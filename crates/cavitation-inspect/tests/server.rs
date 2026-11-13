//! Tests of the inspector server's network edge, its event hub, its command queue and its
//! JSON writer, through plain `TcpStream` clients. Every server binds port 0.
//!
//! Timing bounds are generous where the operating system's scheduling decides the result
//! (a loaded CI machine can stall a thread for hundreds of milliseconds); each one says
//! what it allows and why. The one tight bound is `publish_never_blocks`, which measures a
//! failed `try_lock`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use cavitation_core::Error;
use cavitation_inspect::commands::Command;
use cavitation_inspect::http::{HttpError, Method, parse_request};
use cavitation_inspect::json::JsonWriter;
use cavitation_inspect::sse::{Event, Hub, Snapshot};
use cavitation_inspect::{Inspector, InspectorConfig};

/// How long a test client waits for any one read before it gives up, so that a server bug
/// fails the test instead of hanging the test binary.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

/// A configuration that binds the loopback address on a port the system chooses.
fn config() -> InspectorConfig {
    InspectorConfig {
        address: "127.0.0.1:0".parse().unwrap(),
        ..InspectorConfig::default()
    }
}

/// Starts a server with [`config`].
fn start() -> Inspector {
    Inspector::start(config()).expect("the server starts")
}

/// The address the server is bound to, read from its URL.
fn addr(inspector: &Inspector) -> SocketAddr {
    let url = inspector.url();
    let rest = url.strip_prefix("http://").expect("an http URL");
    rest[..rest.find('/').unwrap()].parse().unwrap()
}

/// The server's token, read from its URL.
fn token(inspector: &Inspector) -> String {
    let url = inspector.url();
    url[url.find("token=").unwrap() + 6..].to_string()
}

/// One response: the status code, the head (status line and headers) and the body.
struct Response {
    /// The status code.
    status: u16,
    /// The status line and headers, as text.
    head: String,
    /// The bytes after the head.
    body: Vec<u8>,
}

impl Response {
    /// The value of the header `name` (case-insensitive), if present.
    fn header(&self, name: &str) -> Option<String> {
        self.head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
    }
}

/// Opens a connection with the client timeouts set.
fn connect(addr: SocketAddr) -> TcpStream {
    let stream = TcpStream::connect(addr).expect("connect");
    stream.set_read_timeout(Some(CLIENT_TIMEOUT)).unwrap();
    stream.set_write_timeout(Some(CLIENT_TIMEOUT)).unwrap();
    stream
}

/// Sends `raw` on a new connection and reads the response until the server closes it.
fn send(addr: SocketAddr, raw: &[u8]) -> Response {
    let mut stream = connect(addr);
    // The server may answer and close before reading everything (an oversized head), so
    // a failed write is not an error here; the response is what is checked.
    let _ = stream.write_all(raw);
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => bytes.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }
    parse_response(&bytes)
}

/// Splits raw response bytes into a [`Response`].
fn parse_response(bytes: &[u8]) -> Response {
    let end = bytes
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no complete head in {:?}", String::from_utf8_lossy(bytes)));
    let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    Response {
        status,
        head,
        body: bytes[end + 4..].to_vec(),
    }
}

/// Sends a request with the server's own `Host` and any extra header lines.
fn request(inspector: &Inspector, method: &str, target: &str, extra: &str) -> Response {
    let a = addr(inspector);
    let raw = format!("{method} {target} HTTP/1.1\r\nHost: {a}\r\n{extra}\r\n");
    send(a, raw.as_bytes())
}

/// Polls `done` every 5 ms until it is true or `limit` has passed; returns whether it became
/// true.
fn wait_until(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if done() {
            return true;
        }
        thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// Opens `/events` with the token and reads the response head; returns the reader, after
/// the head, and the status code.
fn open_events(inspector: &Inspector) -> (BufReader<TcpStream>, u16) {
    let a = addr(inspector);
    let mut stream = connect(a);
    let raw = format!(
        "GET /events?token={} HTTP/1.1\r\nHost: {a}\r\n\r\n",
        token(inspector)
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    let status = status_line.split(' ').nth(1).unwrap().parse().unwrap();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
    }
    (reader, status)
}

/// Reads the next event from a stream, skipping comment lines; `None` at the end of the
/// stream.
fn read_event(reader: &mut BufReader<TcpStream>) -> Option<(String, String)> {
    let (mut name, mut data) = (String::new(), String::new());
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end_matches('\n');
        if line.is_empty() {
            if !name.is_empty() || !data.is_empty() {
                return Some((name, data));
            }
        } else if let Some(value) = line.strip_prefix("event: ") {
            name = value.to_string();
        } else if let Some(value) = line.strip_prefix("data: ") {
            data = value.to_string();
        }
    }
}

/// The pure parser: what it extracts and the status of each refusal.
#[test]
fn parse_request_cases() {
    let r = parse_request(
        b"GET /events?token=abc&x=1&flag HTTP/1.1\r\nhost: 127.0.0.1:9\r\nORIGIN: http://127.0.0.1:9\r\n\r\n",
    )
    .unwrap();
    assert_eq!(r.method, Method::Get);
    assert_eq!(r.path, "/events");
    assert_eq!(
        r.query,
        vec![
            ("token".to_string(), "abc".to_string()),
            ("x".to_string(), "1".to_string()),
            ("flag".to_string(), String::new()),
        ]
    );
    assert_eq!(r.host.as_deref(), Some("127.0.0.1:9"));
    assert_eq!(r.origin.as_deref(), Some("http://127.0.0.1:9"));

    let r = parse_request(
        b"POST /api/pause HTTP/1.1\r\nHost: localhost:9\r\nContent-Length: 0\r\n\r\n",
    )
    .unwrap();
    assert_eq!((r.method, r.path.as_str()), (Method::Post, "/api/pause"));
    assert!(r.query.is_empty() && r.origin.is_none());

    // A head without its final blank line still parses; the reader decides completeness.
    assert!(parse_request(b"GET / HTTP/1.1\r\nHost: localhost:9").is_ok());

    let status = |head: &[u8]| {
        parse_request(head)
            .map(|_| 0)
            .unwrap_or_else(|e| e.status())
    };
    assert_eq!(status(b"PUT / HTTP/1.1\r\nHost: a\r\n\r\n"), 405);
    assert_eq!(status(b"DELETE / HTTP/1.1\r\nHost: a\r\n\r\n"), 405);
    assert_eq!(
        status(b"GET /?token=a%20b HTTP/1.1\r\nHost: a\r\n\r\n"),
        400
    );
    assert_eq!(
        status(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n"),
        413
    );
    assert_eq!(
        status(b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n"),
        413
    );
    assert_eq!(
        status(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: x\r\n\r\n"),
        400
    );
    assert_eq!(status(b"GET / HTTP/1.1\r\nno colon here\r\n\r\n"), 400);
    assert_eq!(status(b"GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n"), 400);
    assert_eq!(status(b"GET /\r\n\r\n"), 400);
    assert_eq!(status(b"GET http://a/ HTTP/1.1\r\n\r\n"), 400);
    assert_eq!(status(b"GET / SPDY/3\r\n\r\n"), 400);
    assert_eq!(status(b"\xff\xfe / HTTP/1.1\r\n\r\n"), 400);
    assert_eq!(status(b""), 400);
    let mut big = b"GET / HTTP/1.1\r\nX: ".to_vec();
    big.resize(9000, b'a');
    big.extend_from_slice(b"\r\n\r\n");
    assert_eq!(status(&big), 431);
    assert_eq!(HttpError::HeadTooLarge.status(), 431);
}

/// A head above 8 KiB is answered 431.
#[test]
fn oversized_head_is_431() {
    let inspector = start();
    let a = addr(&inspector);
    let mut raw = format!("GET / HTTP/1.1\r\nHost: {a}\r\nX-Filler: ").into_bytes();
    raw.resize(9 * 1024, b'a');
    raw.extend_from_slice(b"\r\n\r\n");
    assert_eq!(send(a, &raw).status, 431);
}

/// A request with a body is answered 413, by length or by chunked encoding.
#[test]
fn body_is_413() {
    let inspector = start();
    let a = addr(&inspector);
    let t = token(&inspector);
    let raw = format!(
        "POST /api/pause?token={t} HTTP/1.1\r\nHost: {a}\r\nContent-Length: 10\r\n\r\n0123456789"
    );
    assert_eq!(send(a, raw.as_bytes()).status, 413);
    let raw = format!(
        "POST /api/pause?token={t} HTTP/1.1\r\nHost: {a}\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
    );
    assert_eq!(send(a, raw.as_bytes()).status, 413);
    assert!(inspector.try_command().is_none());
}

/// A head not complete within the read timeout gets the connection closed, at about the
/// timeout.
#[test]
fn half_sent_head_closes_after_read_timeout() {
    let read_timeout = Duration::from_millis(300);
    let inspector = Inspector::start(InspectorConfig {
        read_timeout,
        ..config()
    })
    .unwrap();
    let a = addr(&inspector);
    let mut stream = connect(a);
    let start = Instant::now();
    stream
        .write_all(format!("GET / HTTP/1.1\r\nHost: {a}\r\n").as_bytes())
        .unwrap();
    let mut buf = [0u8; 64];
    // End of stream (or a reset) means the server closed the connection.
    let n = stream.read(&mut buf).unwrap_or(0);
    let elapsed = start.elapsed();
    assert_eq!(n, 0, "no response is sent to a half-sent head");
    // Not before the timeout (less a little for clock granularity); the upper bound allows
    // a loaded machine 3 s of scheduling delay.
    assert!(
        elapsed >= read_timeout - Duration::from_millis(50),
        "{elapsed:?}"
    );
    assert!(
        elapsed < read_timeout + Duration::from_secs(3),
        "{elapsed:?}"
    );
}

/// A `Host` that is not the server's own address is refused.
#[test]
fn foreign_host_is_403() {
    let inspector = start();
    let a = addr(&inspector);
    let t = token(&inspector);
    for host in [
        "evil.example".to_string(),
        format!("evil.example:{}", a.port()),
        format!("127.0.0.1:{}", a.port().wrapping_add(1).max(1)),
        "127.0.0.1".to_string(),
    ] {
        let raw = format!("GET /events?token={t} HTTP/1.1\r\nHost: {host}\r\n\r\n");
        assert_eq!(send(a, raw.as_bytes()).status, 403, "{host}");
    }
    let raw = "GET / HTTP/1.1\r\n\r\n";
    assert_eq!(send(a, raw.as_bytes()).status, 403, "no Host at all");
    // Both loopback names are accepted.
    for host in [
        format!("127.0.0.1:{}", a.port()),
        format!("localhost:{}", a.port()),
    ] {
        let raw = format!("GET / HTTP/1.1\r\nHost: {host}\r\n\r\n");
        assert_eq!(send(a, raw.as_bytes()).status, 200, "{host}");
    }
}

/// A command from another site's page is refused and queues nothing.
#[test]
fn cross_origin_post_is_403() {
    let inspector = start();
    let t = token(&inspector);
    for origin in ["http://evil.example", "null", "https://127.0.0.1"] {
        let r = request(
            &inspector,
            "POST",
            &format!("/api/pause?token={t}"),
            &format!("Origin: {origin}\r\n"),
        );
        assert_eq!(r.status, 403, "{origin}");
    }
    assert!(inspector.try_command().is_none());
}

/// An event stream opened by another site's page is refused and takes no client slot.
#[test]
fn cross_origin_events_is_403() {
    let inspector = start();
    let t = token(&inspector);
    let r = request(
        &inspector,
        "GET",
        &format!("/events?token={t}"),
        "Origin: http://evil.example\r\n",
    );
    assert_eq!(r.status, 403);
    assert_eq!(inspector.clients(), 0);
}

/// The stream, the commands and the export need the token; a wrong token is the same.
#[test]
fn missing_token_is_403() {
    let inspector = start();
    for (method, target) in [
        ("GET", "/events"),
        ("GET", "/events?token="),
        ("GET", "/events?token=00000000000000000000000000000000"),
        ("POST", "/api/mask?linear=0&neuron=0&on=0"),
        ("POST", "/api/mask?token=wrong&linear=0&neuron=0&on=0"),
        ("GET", "/export"),
        ("GET", "/export?token=wrong"),
    ] {
        assert_eq!(
            request(&inspector, method, target, "").status,
            403,
            "{target}"
        );
    }
    assert_eq!(inspector.clients(), 0);
    assert!(inspector.try_command().is_none());
}

/// A command with the token from the server's own page is accepted and queued.
#[test]
fn same_origin_post_with_token_is_204() {
    let inspector = start();
    let a = addr(&inspector);
    let t = token(&inspector);
    let r = request(
        &inspector,
        "POST",
        &format!("/api/mask?token={t}&linear=0&neuron=5&on=0"),
        &format!("Origin: http://{a}\r\n"),
    );
    assert_eq!(r.status, 204);
    assert!(r.body.is_empty());
    assert_eq!(
        inspector.try_command(),
        Some(Command::SetMask {
            linear: 0,
            neuron: 5,
            on: false
        })
    );
    // The `localhost` origin is the same server, and no Origin at all is a same-site tool.
    let r = request(
        &inspector,
        "POST",
        &format!("/api/lr?token={t}&value=0.05"),
        &format!("Origin: http://localhost:{}\r\n", a.port()),
    );
    assert_eq!(r.status, 204);
    assert_eq!(
        inspector.try_command(),
        Some(Command::SetLearningRate(0.05))
    );
    let posts = [
        (format!("/api/pause?token={t}"), Command::Pause),
        (format!("/api/resume?token={t}"), Command::Resume),
        (
            format!("/api/probe?token={t}&index=7"),
            Command::SelectProbe(7),
        ),
        (
            format!("/api/neuron?token={t}&layer=2&neuron=3"),
            Command::SelectNeuron {
                layer: 2,
                neuron: 3,
            },
        ),
        (
            format!("/api/ablate?token={t}&linear=1"),
            Command::Ablate { linear: 1 },
        ),
        (format!("/api/shrink?token={t}"), Command::Shrink),
        (format!("/api/cancel?token={t}"), Command::Cancel),
        (format!("/api/finish?token={t}"), Command::Finish),
        (
            format!("/api/mask?token={t}&linear=1&neuron=2&on=1"),
            Command::SetMask {
                linear: 1,
                neuron: 2,
                on: true,
            },
        ),
    ];
    for (target, command) in posts {
        assert_eq!(
            request(&inspector, "POST", &target, "").status,
            204,
            "{target}"
        );
        assert_eq!(inspector.try_command(), Some(command), "{target}");
    }
    // Bad parameter values are 400 and queue nothing.
    for target in [
        format!("/api/mask?token={t}&linear=0&neuron=5&on=2"),
        format!("/api/mask?token={t}&linear=0&on=1"),
        format!("/api/lr?token={t}&value=-1"),
        format!("/api/lr?token={t}&value=NaN"),
        format!("/api/probe?token={t}&index=x"),
    ] {
        assert_eq!(
            request(&inspector, "POST", &target, "").status,
            400,
            "{target}"
        );
    }
    assert!(inspector.try_command().is_none());
    // GET changes nothing: a command route answers 405 to GET.
    let r = request(&inspector, "GET", &format!("/api/pause?token={t}"), "");
    assert_eq!(r.status, 405);
    assert!(inspector.try_command().is_none());
}

/// No response of any route, refused or not, allows another origin to read it.
#[test]
fn no_response_allows_cors() {
    let inspector = start();
    let t = token(&inspector);
    let origin = "Origin: http://evil.example\r\n";
    let mut responses = vec![
        request(&inspector, "GET", "/", ""),
        request(&inspector, "GET", "/", origin),
        request(&inspector, "GET", &format!("/export?token={t}"), ""),
        request(&inspector, "GET", "/export", ""),
        request(&inspector, "GET", "/events", ""),
        request(&inspector, "GET", &format!("/events?token={t}"), origin),
        request(&inspector, "GET", "/nowhere", ""),
        request(&inspector, "PUT", "/", ""),
        request(&inspector, "GET", &format!("/api/pause?token={t}"), ""),
    ];
    for route in [
        "mask?linear=0&neuron=0&on=1",
        "lr?value=0.1",
        "pause",
        "resume",
        "probe?index=0",
        "neuron?layer=0&neuron=0",
        "ablate?linear=0",
        "shrink",
        "cancel",
        "finish",
    ] {
        let sep = if route.contains('?') { '&' } else { '?' };
        let target = format!("/api/{route}{sep}token={t}");
        responses.push(request(&inspector, "POST", &target, ""));
        responses.push(request(&inspector, "POST", &target, origin));
    }
    // The event stream's own head.
    let a = addr(&inspector);
    let mut stream = connect(a);
    stream
        .write_all(format!("GET /events?token={t} HTTP/1.1\r\nHost: {a}\r\n\r\n").as_bytes())
        .unwrap();
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        head.push(byte[0]);
    }
    responses.push(parse_response(&head));
    for r in &responses {
        assert!(
            r.header("Access-Control-Allow-Origin").is_none(),
            "{}",
            r.head
        );
        assert!(
            !r.head.to_ascii_lowercase().contains("access-control-allow"),
            "{}",
            r.head
        );
    }
}

/// `GET /export` returns a document and changes nothing: no file, no command, no client.
#[test]
fn export_get_writes_nothing() {
    let listing = |dir: &std::path::Path| -> Vec<std::path::PathBuf> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        entries
    };
    let here = std::env::current_dir().unwrap();
    let before = listing(&here);
    let inspector = start();
    let t = token(&inspector);
    let r = request(&inspector, "GET", &format!("/export?token={t}"), "");
    assert_eq!(r.status, 200);
    assert!(r.header("Content-Type").unwrap().starts_with("text/html"));
    let body = String::from_utf8(r.body).unwrap();
    assert!(body.starts_with("<!doctype html>"), "{body}");
    assert_eq!(listing(&here), before);
    assert!(inspector.try_command().is_none());
    assert_eq!(inspector.clients(), 0);
}

/// A path the server does not have is 404.
#[test]
fn unknown_path_is_404() {
    let inspector = start();
    let t = token(&inspector);
    for target in [
        "/nowhere".to_string(),
        "/index.html".to_string(),
        format!("/api/unknown?token={t}"),
        "/events/".to_string(),
    ] {
        let method = if target.starts_with("/api") {
            "POST"
        } else {
            "GET"
        };
        assert_eq!(
            request(&inspector, method, &target, "").status,
            404,
            "{target}"
        );
    }
}

/// The ninth event stream is refused while eight are open.
#[test]
fn ninth_client_is_503() {
    let inspector = start();
    let streams: Vec<_> = (0..8)
        .map(|_| {
            let (reader, status) = open_events(&inspector);
            assert_eq!(status, 200);
            reader
        })
        .collect();
    assert_eq!(inspector.clients(), 8);
    let t = token(&inspector);
    assert_eq!(
        request(&inspector, "GET", &format!("/events?token={t}"), "").status,
        503
    );
    assert_eq!(inspector.clients(), 8);
    drop(streams);
    // Closed streams are noticed and their slots freed (generous: a loaded machine).
    assert!(wait_until(Duration::from_secs(5), || inspector.clients() == 0));
    let (_reader, status) = open_events(&inspector);
    assert_eq!(status, 200);
}

/// Listening on an address in use is `Error::Net` naming the address.
#[test]
fn port_in_use_is_net_error_naming_address() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = taken.local_addr().unwrap();
    match Inspector::start(InspectorConfig {
        address,
        ..InspectorConfig::default()
    }) {
        Err(Error::Net {
            address: named,
            source,
        }) => {
            assert_eq!(named, address.to_string());
            let text = Error::Net {
                address: named,
                source,
            }
            .to_string();
            assert!(text.contains(&address.to_string()), "{text}");
        }
        Err(other) => panic!("expected Error::Net, got {other}"),
        Ok(_) => panic!("expected Error::Net, the server started"),
    }
}

/// With port 0 the URL names the port the system chose, and a 128-bit token.
#[test]
fn port_zero_reports_real_port_and_token() {
    let inspector = start();
    let url = inspector.url();
    assert!(url.starts_with("http://127.0.0.1:"), "{url}");
    assert_ne!(addr(&inspector).port(), 0);
    let t = token(&inspector);
    assert_eq!(t.len(), 32, "{url}");
    assert!(
        t.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "{t}"
    );
    assert_eq!(url, format!("http://{}/?token={t}", addr(&inspector)));
    // Each run has its own token.
    let other = start();
    assert_ne!(token(&other), t);
    assert_ne!(
        cavitation_inspect::token::new_token(),
        cavitation_inspect::token::new_token()
    );
}

/// The default address is the loopback interface, port 7878.
#[test]
fn binds_loopback_by_default() {
    let default = InspectorConfig::default();
    assert_eq!(
        default.address,
        "127.0.0.1:7878".parse::<SocketAddr>().unwrap()
    );
    assert!(default.address.ip().is_loopback());
    let inspector = Inspector::start(InspectorConfig {
        address: SocketAddr::new(default.address.ip(), 0),
        ..InspectorConfig::default()
    })
    .unwrap();
    assert!(addr(&inspector).ip().is_loopback());
}

/// While another thread holds the hub's lock, a snapshot publish returns `false` at once.
#[test]
fn publish_never_blocks() {
    let hub = Arc::new(Hub::new(8, 256, Duration::from_secs(2)));
    let (held_tx, held_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let holder = {
        let hub = Arc::clone(&hub);
        thread::spawn(move || {
            // The fill closure runs under the lock: hold it until told to let go.
            hub.publish_snapshot(Snapshot::Stats, |w| {
                held_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(10));
                w.number(1.0);
            })
        })
    };
    held_rx.recv().unwrap();
    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let published = hub.publish_snapshot(Snapshot::Probe, |w| {
            w.number(2.0);
        });
        times.push(start.elapsed());
        assert!(!published);
    }
    let start = Instant::now();
    assert!(!hub.publish_event(
        Event::Status,
        |w| {
            w.string("x");
        },
        false
    ));
    times.push(start.elapsed());
    release_tx.send(()).unwrap();
    assert!(holder.join().unwrap());
    // A blocking publish would wait for the holder, 10 s; a failed `try_lock` takes
    // microseconds. The fastest call is held to the 1 ms bound, and every call to 100 ms so
    // that one preempted call on a loaded machine does not fail the test.
    let fastest = times.iter().min().unwrap();
    let slowest = times.iter().max().unwrap();
    assert!(*fastest < Duration::from_millis(1), "{times:?}");
    assert!(*slowest < Duration::from_millis(100), "{times:?}");
    // Once released, publishing works again.
    assert!(hub.publish_snapshot(Snapshot::Probe, |w| {
        w.number(3.0);
    }));
}

/// A burst of 1,000 discrete events published with `blocking: true` all reach a reading
/// client, in order, although its queue holds 256.
#[test]
fn discrete_events_never_skipped() {
    let inspector = start();
    let (mut reader, status) = open_events(&inspector);
    assert_eq!(status, 200);
    assert!(wait_until(Duration::from_secs(5), || inspector.clients() == 1));
    let reading = thread::spawn(move || {
        let mut got = Vec::new();
        while got.len() < 1000 {
            match read_event(&mut reader) {
                Some((name, data)) if name == "ablation" => got.push(data),
                Some(_) => {}
                None => break,
            }
        }
        got
    });
    for i in 0..1000u64 {
        let published = inspector.hub().publish_event(
            Event::Ablation,
            |w| {
                w.begin_object().key("i").uint(i).end_object();
            },
            true,
        );
        assert!(published, "event {i}");
    }
    let got = reading.join().unwrap();
    assert_eq!(got.len(), 1000);
    for (i, data) in got.iter().enumerate() {
        assert_eq!(*data, format!("{{\"i\":{i}}}"));
    }
}

/// A client that never reads is dropped once a write makes no progress for the write
/// timeout, and publishing carries on.
#[test]
fn stalled_client_is_dropped() {
    let inspector = Inspector::start(InspectorConfig {
        write_timeout: Duration::from_millis(200),
        ..config()
    })
    .unwrap();
    let a = addr(&inspector);
    let mut stalled = connect(a);
    stalled
        .write_all(
            format!(
                "GET /events?token={} HTTP/1.1\r\nHost: {a}\r\n\r\n",
                token(&inspector)
            )
            .as_bytes(),
        )
        .unwrap();
    assert!(wait_until(Duration::from_secs(5), || inspector.clients() == 1));
    let big = "x".repeat(1 << 20);
    let start = Instant::now();
    // The brief's bound is 2 s; 5 s here allows for a loaded machine.
    while inspector.clients() > 0 && start.elapsed() < Duration::from_secs(5) {
        inspector.hub().publish_snapshot(Snapshot::Stats, |w| {
            w.string(&big);
        });
        thread::sleep(Duration::from_millis(5));
    }
    let elapsed = start.elapsed();
    assert_eq!(inspector.clients(), 0, "still connected after {elapsed:?}");
    println!("stalled client dropped after {elapsed:?}");
    assert!(inspector.hub().publish_snapshot(Snapshot::Stats, |w| {
        w.number(1.0);
    }));
    assert!(inspector.hub().publish_event(
        Event::Epoch,
        |w| {
            w.number(1.0);
        },
        false
    ));
    drop(stalled);
}

/// A blocking event publish waits at most about the write timeout for a client whose
/// queue stays full, then drops that client, even when the client keeps reading a little
/// at a time so its writer never times out.
#[test]
fn blocking_publish_drops_a_slow_client_after_write_timeout() {
    let write_timeout = Duration::from_millis(300);
    let inspector = Inspector::start(InspectorConfig {
        write_timeout,
        client_queue: 2,
        ..config()
    })
    .unwrap();
    let a = addr(&inspector);
    let mut slow = connect(a);
    slow.write_all(
        format!(
            "GET /events?token={} HTTP/1.1\r\nHost: {a}\r\n\r\n",
            token(&inspector)
        )
        .as_bytes(),
    )
    .unwrap();
    assert!(wait_until(Duration::from_secs(5), || inspector.clients() == 1));
    // Reads 64 KiB every 20 ms: progress within every write timeout, far slower than the
    // publisher. It ends when the server closes the connection.
    let reader = thread::spawn(move || {
        let mut chunk = vec![0u8; 64 * 1024];
        loop {
            match slow.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
    });
    let big = "x".repeat(8 << 20);
    let mut slowest = Duration::ZERO;
    let start = Instant::now();
    while inspector.clients() > 0 && start.elapsed() < Duration::from_secs(10) {
        let t = Instant::now();
        assert!(inspector.hub().publish_event(
            Event::Ablation,
            |w| {
                w.string(&big);
            },
            true
        ));
        slowest = slowest.max(t.elapsed());
    }
    assert_eq!(inspector.clients(), 0, "the slow client was never dropped");
    // The wait is capped at the write timeout (300 ms); without the cap, one publish waits
    // for the reader to take a 16 MiB batch at about 3 MB/s, several seconds. The bound allows a loaded
    // machine 1.7 s of scheduling delay.
    assert!(slowest < Duration::from_secs(2), "{slowest:?}");
    println!("slowest blocking publish {slowest:?}");
    drop(inspector);
    reader.join().unwrap();
}

/// Garbage bytes are refused without stopping the server.
#[test]
fn malformed_request_keeps_server_alive() {
    let inspector = start();
    let a = addr(&inspector);
    let r = send(a, b"\x00\x01garbage\xff\xfe\r\n\r\n");
    assert_eq!(r.status, 400);
    // A connection closed with no bytes at all, and one closed mid-head.
    drop(connect(a));
    {
        let mut s = connect(a);
        s.write_all(b"GET / HT").unwrap();
    }
    let r = request(&inspector, "GET", "/", "");
    assert_eq!(r.status, 200);
    assert!(
        String::from_utf8(r.body)
            .unwrap()
            .contains("<!doctype html>")
    );
}

/// Dropping the server with a stalled client and a half-sent request returns within 1 s.
#[test]
fn drop_returns_within_one_second() {
    let inspector = start();
    let a = addr(&inspector);
    let mut stalled = connect(a);
    stalled
        .write_all(
            format!(
                "GET /events?token={} HTTP/1.1\r\nHost: {a}\r\n\r\n",
                token(&inspector)
            )
            .as_bytes(),
        )
        .unwrap();
    assert!(wait_until(Duration::from_secs(5), || inspector.clients() == 1));
    // Fill the socket buffers so the client's writer is in the middle of a write.
    let big = "x".repeat(1 << 20);
    for _ in 0..32 {
        inspector.hub().publish_snapshot(Snapshot::Stats, |w| {
            w.string(&big);
        });
        thread::sleep(Duration::from_millis(2));
    }
    let mut half = connect(a);
    half.write_all(b"GET / HTTP/1.1\r\nHost: 127").unwrap();
    thread::sleep(Duration::from_millis(50));
    let start = Instant::now();
    drop(inspector);
    let elapsed = start.elapsed();
    // The API promises 1 s; the threads notice the shutdown within a few milliseconds.
    assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
    println!("drop took {elapsed:?}");
    drop((stalled, half));
}

/// JSON escapes what could end a `<script>` element or a line, and writes non-finite
/// numbers as `null`.
#[test]
fn json_escapes_and_nulls() {
    let mut w = JsonWriter::new();
    w.begin_object()
        .key("a")
        .number(f64::NAN)
        .key("b")
        .number(f64::INFINITY)
        .key("c")
        .number(f64::NEG_INFINITY)
        .key("s")
        .string("</script>")
        .key("v")
        .f32s(&[1.0, f32::NAN, -0.5, f32::INFINITY])
        .key("t")
        .bool(true)
        .key("n")
        .null()
        .key("e")
        .begin_array()
        .end_array()
        .end_object();
    assert_eq!(
        w.as_str(),
        r#"{"a":null,"b":null,"c":null,"s":"\u003c/script\u003e","v":[1.0,null,-0.5,null],"t":true,"n":null,"e":[]}"#
    );
    w.clear();
    w.string("\"\\\n\r\t\u{1}\u{1f}&<>\u{2028}\u{2029}é");
    assert_eq!(
        w.as_str(),
        r#""\"\\\n\r\t\u0001\u001f\u0026\u003c\u003e\u2028\u2029é""#
    );
    w.clear();
    w.begin_array()
        .number(0.1)
        .int(-3)
        .uint(7)
        .f32(f32::NAN)
        .end_array();
    assert_eq!(w.as_str(), "[0.1,-3,7,null]");
}

/// The 65th command waiting to be read is refused with 503.
#[test]
fn command_queue_is_bounded() {
    let inspector = start();
    let t = token(&inspector);
    let target = format!("/api/pause?token={t}");
    for i in 0..64 {
        assert_eq!(
            request(&inspector, "POST", &target, "").status,
            204,
            "command {i}"
        );
    }
    assert_eq!(request(&inspector, "POST", &target, "").status, 503);
    let mut drained = 0;
    while inspector.try_command().is_some() {
        drained += 1;
    }
    assert_eq!(drained, 64);
    assert_eq!(request(&inspector, "POST", &target, "").status, 204);
}
