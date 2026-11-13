//! `Inspector` and `InspectorConfig`: the server's threads, its checks on every request,
//! and its routes.
//!
//! One accept thread takes connections. Each connection gets a short-lived thread that
//! reads its request head (within the read timeout), checks it, answers it and closes it;
//! an `/events` connection's thread stays on as that client's writer. At most
//! [`MAX_PENDING`] connections may be reading their heads at once; more are closed at once.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cavitation_core::Error;
use cavitation_nn::Dataset;

use crate::commands::{Command, CommandQueue, QUEUE_CAPACITY};
use crate::export::{self, EXPORT_POLICY, PAGE, PAGE_POLICY};
use crate::http::{self, Head, Method, Request};
use crate::snapshot::Training;
use crate::sse::Hub;
use crate::token::new_token;

/// The most connections reading their request heads at once.
pub(crate) const MAX_PENDING: usize = 32;

/// How long `Drop` waits for its wake-up connection to the accept thread.
const WAKE_TIMEOUT: Duration = Duration::from_millis(500);

/// The command routes, after `/api/`.
const COMMANDS: [&str; 10] = [
    "mask", "lr", "pause", "resume", "probe", "neuron", "ablate", "shrink", "cancel", "finish",
];

/// The `Content-Security-Policy` header line of a response carrying `policy`, with
/// `frame-ancestors 'none'` added so no other site can show the page in a frame (a `<meta>`
/// policy cannot carry that directive, so only the header has it).
fn policy_header(policy: &str) -> String {
    format!("Content-Security-Policy: {policy}; frame-ancestors 'none'\r\n")
}

/// How the inspector serves a run.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::InspectorConfig;
/// let config = InspectorConfig { title: "digits".to_string(), ..InspectorConfig::default() };
/// assert_eq!(config.address.to_string(), "127.0.0.1:7878");
/// assert_eq!(config.max_clients, 8);
/// ```
#[derive(Debug, Clone)]
pub struct InspectorConfig {
    /// Where to listen. By default `127.0.0.1:7878`, reachable only from this machine. Port
    /// 0 lets the system choose a free port, which [`Inspector::url`] reports.
    ///
    /// Only loopback addresses are intended. With another address the server answers only
    /// requests whose `Host` names that exact address (or `127.0.0.1` or `localhost` with
    /// the port), over unencrypted HTTP, so the token crosses the network in clear text.
    /// With `0.0.0.0` the port is open to the network while every request from another
    /// machine is refused, since its `Host` names an address other than `0.0.0.0`.
    pub address: SocketAddr,
    /// The most snapshots per second the training side captures; 4 by default.
    pub snapshot_hz: f64,
    /// The most event streams (browser tabs) at once; 8 by default. Another is answered 503.
    pub max_clients: usize,
    /// The most discrete events waiting for one client; 256 by default.
    pub client_queue: usize,
    /// How long a client's connection may accept no bytes before the client is dropped;
    /// 2 s by default.
    pub write_timeout: Duration,
    /// How long a client has to send a complete request head; 5 s by default.
    pub read_timeout: Duration,
    /// How long a paused run waits with no client connected before it resumes; 30 s by
    /// default.
    pub idle_resume: Duration,
    /// The rows the page can probe. `None` (the default) uses the first `eval_rows` training
    /// rows, which the page labels as training rows.
    pub probe: Option<Dataset>,
    /// How many probe rows the held-out evaluation uses; 1,000 by default.
    pub eval_rows: usize,
    /// `Some((height, width))` when each input row is an image of that size, so the page
    /// draws inputs and incoming weights as pictures.
    pub input_image: Option<(usize, usize)>,
    /// The names of the output classes, for the page's labels.
    pub class_names: Option<Vec<String>>,
    /// The page's title; "Cavitation inspector" by default.
    pub title: String,
}

impl Default for InspectorConfig {
    /// The defaults listed on each field.
    ///
    /// ```
    /// let config = cavitation_inspect::InspectorConfig::default();
    /// assert_eq!(config.write_timeout, std::time::Duration::from_secs(2));
    /// ```
    fn default() -> InspectorConfig {
        InspectorConfig {
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7878),
            snapshot_hz: 4.0,
            max_clients: 8,
            client_queue: 256,
            write_timeout: Duration::from_secs(2),
            read_timeout: Duration::from_secs(5),
            idle_resume: Duration::from_secs(30),
            probe: None,
            eval_rows: 1000,
            input_image: None,
            class_names: None,
            title: "Cavitation inspector".to_string(),
        }
    }
}

/// What the server's threads share.
pub(crate) struct Shared {
    /// The snapshots, events and clients.
    pub(crate) hub: Hub,
    /// Commands from the page, waiting for the training side.
    pub(crate) commands: CommandQueue,
    /// This run's token.
    token: String,
    /// The `Host` values that name this server, lowercase.
    hosts: Vec<String>,
    /// How long a client has to send its request head.
    read_timeout: Duration,
    /// How long a write may make no progress.
    write_timeout: Duration,
    /// The connection threads, joined on drop.
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// Connections still reading their request heads.
    pending: AtomicUsize,
}

/// A live view of a training run in the browser: an HTTP and Server-Sent Events server on
/// this machine, built on `std::net` alone.
///
/// [`start`](Self::start) listens and returns at once; the server runs on its own threads
/// until the `Inspector` is dropped. Open [`url`](Self::url) in a browser. The URL carries a
/// token made for this run, without which the event stream, the commands and the export are
/// refused, and the server refuses any request whose `Host` is not its own address or whose
/// `Origin` is another site, so neither other web pages nor other users of the machine can
/// read the run or change it. At most 32 connections may be sending their request at once,
/// each within `read_timeout`; more are closed at once, so a local program that holds 32
/// half-sent requests can keep the page from loading, though not read or change the run.
///
/// Routes: `GET /` (the page), `GET /events?token=` (the event stream), `GET /export?token=`
/// (the page with the run so far embedded, as a download; the server writes no file), and
/// the commands
/// `POST /api/{mask,lr,pause,resume,probe,neuron,ablate,shrink,cancel,finish}?token=…`,
/// each answered 204, or 503 when 64 commands are already waiting.
///
/// To train under it, call [`fit`](crate::fit) in place of `cavitation_nn::fit`; the event
/// stream then carries the events [`wire`](crate::wire) describes.
///
/// The page is one HTML file built into the crate, with its style and script inline. It
/// reads the token from its own address, follows the event stream and sends the commands
/// from its buttons; it loads nothing from anywhere else, and its Content Security Policy
/// allows connections only to this server. It shows the speed and the loss, every layer's
/// values for one probe row as a heat map (outputs, before the activation, gradients or
/// the activation's slope), one neuron opened up with the chain rule written out, where a
/// step's time goes, and in explore mode the ablation ranking and the shrink curve.
/// [`export_html`](Self::export_html) saves the same page with the run so far in it, to
/// open later with no server.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::{Inspector, InspectorConfig};
/// let inspector = Inspector::start(InspectorConfig {
///     address: "127.0.0.1:0".parse().unwrap(),
///     ..InspectorConfig::default()
/// })?;
/// let url = inspector.url();
/// assert!(url.starts_with("http://127.0.0.1:"));
/// assert_eq!(inspector.clients(), 0);
/// drop(inspector); // stops the server and joins its threads
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub struct Inspector {
    /// The configuration it was started with.
    pub(crate) config: InspectorConfig,
    /// The address it is bound to (the real port when the configuration asked for 0).
    address: SocketAddr,
    /// What its threads share.
    pub(crate) shared: Arc<Shared>,
    /// The accept thread, joined on drop.
    accept: Option<JoinHandle<()>>,
    /// The training side's state: the probe, the evaluation, the pause and the events
    /// waiting to be published.
    pub(crate) training: Training,
}

impl fmt::Debug for Inspector {
    /// Shows the address and the number of clients; never the token.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let inspector = Inspector::start(config).unwrap();
    /// assert!(format!("{inspector:?}").contains("127.0.0.1"));
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Inspector")
            .field("address", &self.address)
            .field("clients", &self.clients())
            .finish_non_exhaustive()
    }
}

/// Returns `Error::InvalidData` naming the configuration field when `ok` is false.
fn require(ok: bool, what: &str) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidData(format!(
            "inspector configuration: {what}"
        )))
    }
}

impl Inspector {
    /// Listens on `config.address` and starts the accept thread.
    ///
    /// # Errors
    ///
    /// [`Error::Net`] naming the address when it cannot listen there (for example, the
    /// port is in use) or cannot start its thread; [`Error::InvalidData`] when
    /// `snapshot_hz` is not finite and above 0 or so small that one period does not fit a
    /// [`Duration`], or `max_clients`, `client_queue`, `write_timeout` or `read_timeout`
    /// is 0.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let address = "127.0.0.1:0".parse().unwrap();
    /// let tiny = InspectorConfig { address, snapshot_hz: 1e-20, ..Default::default() };
    /// assert!(Inspector::start(tiny).is_err());
    /// ```
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    /// let config = InspectorConfig { address: busy.local_addr().unwrap(), ..Default::default() };
    /// let error = Inspector::start(config).unwrap_err();
    /// assert!(error.to_string().contains(&busy.local_addr().unwrap().to_string()));
    /// ```
    pub fn start(config: InspectorConfig) -> Result<Inspector, Error> {
        require(
            config.snapshot_hz.is_finite()
                && config.snapshot_hz > 0.0
                && Duration::try_from_secs_f64(1.0 / config.snapshot_hz).is_ok(),
            "snapshot_hz must be finite and above 0, with a period that fits a Duration",
        )?;
        require(config.max_clients > 0, "max_clients must be at least 1")?;
        require(config.client_queue > 0, "client_queue must be at least 1")?;
        require(
            !config.write_timeout.is_zero(),
            "write_timeout must be above 0",
        )?;
        require(
            !config.read_timeout.is_zero(),
            "read_timeout must be above 0",
        )?;
        let net = |source| Error::Net {
            address: config.address.to_string(),
            source,
        };
        let listener = TcpListener::bind(config.address).map_err(net)?;
        let address = listener.local_addr().map_err(net)?;
        let port = address.port();
        let mut hosts = vec![format!("127.0.0.1:{port}"), format!("localhost:{port}")];
        if !hosts.contains(&address.to_string()) {
            hosts.push(address.to_string().to_ascii_lowercase());
        }
        let shared = Arc::new(Shared {
            hub: Hub::new(
                config.max_clients,
                config.client_queue,
                config.write_timeout,
            ),
            commands: CommandQueue::new(QUEUE_CAPACITY),
            token: new_token(),
            hosts,
            read_timeout: config.read_timeout,
            write_timeout: config.write_timeout,
            threads: Mutex::new(Vec::new()),
            pending: AtomicUsize::new(0),
        });
        let accept = {
            let shared = Arc::clone(&shared);
            thread::Builder::new()
                .name("cavitation-inspect-accept".to_string())
                .spawn(move || accept_loop(&listener, &shared))
                .map_err(net)?
        };
        Ok(Inspector {
            config,
            address,
            shared,
            accept: Some(accept),
            training: Training::new(),
        })
    }

    /// The address to open in a browser: `http://<bound address>/?token=<token>`, with the
    /// real port when the configuration asked for port 0.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let inspector = Inspector::start(config).unwrap();
    /// let url = inspector.url();
    /// assert!(!url.contains(":0/"));
    /// assert_eq!(url.len() - url.find("token=").unwrap() - 6, 32);
    /// ```
    pub fn url(&self) -> String {
        format!("http://{}/?token={}", self.address, self.shared.token)
    }

    /// The address the server is bound to.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let inspector = Inspector::start(config).unwrap();
    /// assert_ne!(inspector.address().port(), 0);
    /// ```
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// The configuration the server was started with.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let inspector = Inspector::start(config).unwrap();
    /// assert_eq!(inspector.config().max_clients, 8);
    /// ```
    pub fn config(&self) -> &InspectorConfig {
        &self.config
    }

    /// The number of connected event streams.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// assert_eq!(Inspector::start(config).unwrap().clients(), 0);
    /// ```
    pub fn clients(&self) -> usize {
        self.shared.hub.clients()
    }

    /// The hub the training side publishes snapshots and events to.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// use cavitation_inspect::sse::Snapshot;
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let inspector = Inspector::start(config).unwrap();
    /// assert!(inspector.hub().publish_snapshot(Snapshot::Stats, |w| { w.uint(1); }));
    /// ```
    pub fn hub(&self) -> &Hub {
        &self.shared.hub
    }

    /// The oldest command from the page, without waiting: `None` when there is none, or
    /// when the server is adding one at this moment (try again at the next step).
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// assert!(Inspector::start(config).unwrap().try_command().is_none());
    /// ```
    pub fn try_command(&self) -> Option<Command> {
        self.shared.commands.try_pop()
    }

    /// How many snapshots the training side has captured (statistics, probe row and neuron,
    /// taken together) over the inspector's life. It stays 0 while no client is connected,
    /// since nothing is captured then.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// assert_eq!(Inspector::start(config).unwrap().snapshots_taken(), 0);
    /// ```
    pub fn snapshots_taken(&self) -> u64 {
        self.training.captures
    }

    /// Writes the inspector page to `path` as one self-contained HTML file, with everything
    /// published so far embedded (the model, the latest statistics, probe row, neuron and
    /// profile, the latest status, and the recent epochs, ablations and shrink levels, as
    /// [`Hub::export_json`](crate::sse::Hub::export_json) gathers them). Opened in a
    /// browser, it shows that snapshot with its controls off; a `<meta>` Content Security
    /// Policy forbids it any connection. The JSON is written as the event stream's is, so
    /// no title or name in it can end its `<script>` element.
    ///
    /// `GET /export?token=…` sends the same document as a download and writes no file.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] naming `path` when the file cannot be written.
    ///
    /// ```
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let inspector = Inspector::start(config)?;
    /// let path = std::env::temp_dir().join(format!("inspector-doc-{}.html", std::process::id()));
    /// inspector.export_html(&path)?;
    /// let html = std::fs::read_to_string(&path).unwrap();
    /// assert!(html.contains(r#"<script id="snapshot" type="application/json">"#));
    /// std::fs::remove_file(&path).unwrap();
    /// # Ok::<(), cavitation_core::Error>(())
    /// ```
    pub fn export_html(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        let html = export::document(&self.shared.hub.export_json());
        std::fs::write(path, html).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })
    }
}

impl Drop for Inspector {
    /// Stops the server: closes the hub (every writer and every waiting publisher wakes),
    /// wakes the accept thread with a connection to its own port, and joins every thread.
    /// Every thread notices the shutdown within a few milliseconds, so this returns well
    /// within 1 s, even with a client that stopped reading or a half-sent request.
    ///
    /// If the wake-up connection cannot be made (a firewall refusing loopback connections),
    /// the accept thread is left to end at the next connection instead of being waited for.
    ///
    /// A run paused from the page cannot be waiting here: [`fit`](crate::fit) holds the
    /// inspector for the whole run, so it can only be dropped afterwards. The pause itself
    /// ends when the hub closes, when the page resumes it, or when no client has been
    /// connected for `idle_resume`.
    fn drop(&mut self) {
        self.shared.hub.close();
        let wake = match self.address.ip() {
            IpAddr::V4(ip) if ip.is_unspecified() => {
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.address.port())
            }
            IpAddr::V6(ip) if ip.is_unspecified() => {
                SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), self.address.port())
            }
            _ => self.address,
        };
        let woken = TcpStream::connect_timeout(&wake, WAKE_TIMEOUT).is_ok();
        if let Some(accept) = self.accept.take()
            && woken
        {
            let _ = accept.join();
        }
        let threads = std::mem::take(
            &mut *self
                .shared
                .threads
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        for thread in threads {
            let _ = thread.join();
        }
    }
}

/// Accepts connections until the hub closes, giving each its own thread.
fn accept_loop(listener: &TcpListener, shared: &Arc<Shared>) {
    for stream in listener.incoming() {
        if shared.hub.stopping().load(Ordering::Acquire) {
            return;
        }
        let Ok(stream) = stream else {
            // Out of file descriptors, or a connection reset before it was accepted:
            // wait a little rather than spin.
            thread::sleep(Duration::from_millis(10));
            continue;
        };
        let mut threads = shared
            .threads
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        threads.retain(|t| !t.is_finished());
        if shared.pending.load(Ordering::Acquire) >= MAX_PENDING {
            continue;
        }
        shared.pending.fetch_add(1, Ordering::AcqRel);
        let worker = Arc::clone(shared);
        match thread::Builder::new()
            .name("cavitation-inspect-client".to_string())
            .spawn(move || connection(stream, &worker))
        {
            Ok(handle) => threads.push(handle),
            Err(_) => {
                shared.pending.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
}

/// Counts a connection as reading its head until dropped.
struct Pending<'a>(&'a AtomicUsize);

impl Drop for Pending<'_> {
    /// Stops counting the connection.
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A response to send: everything but the headers every response carries.
struct Reply {
    /// The status code.
    status: u16,
    /// The `Content-Type`.
    content_type: &'static str,
    /// Further header lines, each ending in `\r\n`.
    extra: String,
    /// The body.
    body: String,
}

impl Reply {
    /// A plain-text response.
    fn text(status: u16, body: impl Into<String>) -> Reply {
        Reply {
            status,
            content_type: "text/plain; charset=utf-8",
            extra: String::new(),
            body: body.into(),
        }
    }

    /// A plain-text response with further header lines.
    fn text_with(status: u16, extra: &str, body: impl Into<String>) -> Reply {
        Reply {
            extra: extra.to_string(),
            ..Reply::text(status, body)
        }
    }
}

/// What a request leads to.
enum Outcome {
    /// Send this response and close.
    Reply(Reply),
    /// Open an event stream.
    Events,
}

/// Serves one connection: reads its head, checks and routes it, answers, and for an event
/// stream runs the client's writer loop.
fn connection(mut stream: TcpStream, shared: &Shared) {
    let pending = Pending(&shared.pending);
    if stream.set_nonblocking(true).is_err() {
        return;
    }
    let _ = stream.set_nodelay(true);
    let stop = shared.hub.stopping();
    let mut buf = Vec::with_capacity(1024);
    let outcome = match http::read_head(&mut stream, &mut buf, shared.read_timeout, stop) {
        Head::Complete(len) => match http::parse_request(&buf[..len]) {
            Ok(request) => route(shared, &request),
            Err(e) => Outcome::Reply(Reply::text(e.status(), e.to_string())),
        },
        Head::TooLarge => {
            Outcome::Reply(Reply::text(431, http::HttpError::HeadTooLarge.to_string()))
        }
        Head::Closed | Head::TimedOut | Head::Stopped => return,
    };
    // The head is read: from here the connection no longer counts against the limit on
    // connections sending a request, even while its answer lingers or its stream runs.
    drop(pending);
    match outcome {
        Outcome::Reply(reply) => {
            let sent = http::respond(
                &mut stream,
                reply.status,
                reply.content_type,
                &reply.extra,
                reply.body.as_bytes(),
                shared.write_timeout,
                stop,
            );
            if sent.is_ok() {
                http::finish(&mut stream, stop);
            }
        }
        Outcome::Events => {
            let Some(id) = shared.hub.register() else {
                let reply = Reply::text(503, "too many event streams are open");
                let _ = http::respond(
                    &mut stream,
                    reply.status,
                    reply.content_type,
                    "",
                    reply.body.as_bytes(),
                    shared.write_timeout,
                    stop,
                );
                http::finish(&mut stream, stop);
                return;
            };
            let mut head = http::head_start(200);
            head.push_str("Content-Type: text/event-stream\r\n\r\n");
            if http::write_all(&mut stream, head.as_bytes(), shared.write_timeout, stop).is_err() {
                shared.hub.unregister(id);
                return;
            }
            shared.hub.serve(id, &mut stream, shared.write_timeout);
        }
    }
}

/// Whether `host` (a `Host` header, or an `Origin` without its scheme) names this server.
fn own_host(shared: &Shared, host: &str) -> bool {
    shared.hosts.iter().any(|h| h.eq_ignore_ascii_case(host))
}

/// Compares the request's token with the run's without stopping at the first difference.
fn token_ok(shared: &Shared, request: &Request) -> bool {
    let Some(given) = request.param("token") else {
        return false;
    };
    let (a, b) = (given.as_bytes(), shared.token.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Checks a parsed request and decides its response: `Host` and `Origin` first (403), then
/// the path (404), the method (405) and the token (403); a command is parsed (400) and
/// queued (204, or 503 when the queue is full).
fn route(shared: &Shared, request: &Request) -> Outcome {
    let forbidden = |why: &str| Outcome::Reply(Reply::text(403, why.to_string()));
    if !request.host.as_deref().is_some_and(|h| own_host(shared, h)) {
        return forbidden("the Host header does not name this server");
    }
    if let Some(origin) = &request.origin {
        let own = origin
            .strip_prefix("http://")
            .is_some_and(|h| own_host(shared, h));
        if !own {
            return forbidden("requests from other sites are refused");
        }
    }
    let path = request.path.as_str();
    let command = path.strip_prefix("/api/");
    let known = matches!(path, "/" | "/events" | "/export")
        || command.is_some_and(|c| COMMANDS.contains(&c));
    if !known {
        return Outcome::Reply(Reply::text(404, "not found"));
    }
    match (command.is_some(), request.method) {
        (true, Method::Get) => {
            return Outcome::Reply(Reply::text_with(
                405,
                "Allow: POST\r\n",
                "commands are sent with POST",
            ));
        }
        (false, Method::Post) => {
            return Outcome::Reply(Reply::text_with(405, "Allow: GET\r\n", "use GET"));
        }
        _ => {}
    }
    if path == "/" {
        return Outcome::Reply(document(PAGE.to_string(), PAGE_POLICY, ""));
    }
    if !token_ok(shared, request) {
        return forbidden("a valid token is required");
    }
    match command {
        None if path == "/events" => Outcome::Events,
        None => Outcome::Reply(document(
            export::document(&shared.hub.export_json()),
            EXPORT_POLICY,
            "Content-Disposition: attachment; filename=\"inspector.html\"\r\n",
        )),
        Some(name) => match parse_command(name, request) {
            Err(why) => Outcome::Reply(Reply::text(400, why)),
            Ok(command) => {
                if shared.commands.push(command) {
                    Outcome::Reply(Reply::text(204, ""))
                } else {
                    Outcome::Reply(Reply::text(503, "too many commands are waiting"))
                }
            }
        },
    }
}

/// A 200 HTML response with the Content Security Policy `policy` and `extra` header lines.
fn document(html: String, policy: &str, extra: &str) -> Reply {
    Reply {
        status: 200,
        content_type: "text/html; charset=utf-8",
        extra: format!("{}{extra}", policy_header(policy)),
        body: html,
    }
}

/// Reads query parameter `name` as a `T`, or says what is wrong with it.
fn param<T: std::str::FromStr>(request: &Request, name: &str) -> Result<T, String> {
    let value = request
        .param(name)
        .ok_or_else(|| format!("missing parameter `{name}`"))?;
    value
        .parse()
        .map_err(|_| format!("parameter `{name}` has a bad value: {value}"))
}

/// Builds the command for `POST /api/<name>` from its query parameters.
fn parse_command(name: &str, request: &Request) -> Result<Command, String> {
    Ok(match name {
        "mask" => Command::SetMask {
            linear: param(request, "linear")?,
            neuron: param(request, "neuron")?,
            on: match request.param("on") {
                Some("1") => true,
                Some("0") => false,
                _ => return Err("parameter `on` must be 0 or 1".to_string()),
            },
        },
        "lr" => {
            let value: f64 = param(request, "value")?;
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "the learning rate must be finite and above 0, not {value}"
                ));
            }
            Command::SetLearningRate(value)
        }
        "pause" => Command::Pause,
        "resume" => Command::Resume,
        "probe" => Command::SelectProbe(param(request, "index")?),
        "neuron" => Command::SelectNeuron {
            layer: param(request, "layer")?,
            neuron: param(request, "neuron")?,
        },
        "ablate" => Command::Ablate {
            linear: param(request, "linear")?,
        },
        "shrink" => Command::Shrink,
        "cancel" => Command::Cancel,
        "finish" => Command::Finish,
        other => return Err(format!("unknown command `{other}`")),
    })
}
