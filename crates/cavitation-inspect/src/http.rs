//! The HTTP/1.1 subset the inspector speaks: parsing a request head, reading one from a
//! socket within a time limit, and writing a response.
//!
//! The server reads one request per connection and closes it after the response. Requests
//! carry no body: commands are `POST`s with their arguments in the query string.
//!
//! Sockets are used in non-blocking mode and polled every 2 ms, so that every wait has
//! its own deadline and notices a shutdown within a few milliseconds on every system.
//! (Blocking sockets with `SO_RCVTIMEO` / `SO_SNDTIMEO` would also give deadlines, but
//! Windows leaves a socket in an undefined state after such a timeout.)

use std::fmt;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// The largest request head accepted, terminating blank line included; a larger one is 431.
///
/// ```
/// use cavitation_inspect::http::{HttpError, MAX_HEAD, parse_request};
/// let head = vec![b'G'; MAX_HEAD + 1];
/// assert_eq!(parse_request(&head), Err(HttpError::HeadTooLarge));
/// ```
pub const MAX_HEAD: usize = 8 * 1024;

/// How long a non-blocking socket operation sleeps before trying again.
pub(crate) const POLL: Duration = Duration::from_millis(2);

/// How long the server keeps reading (and discarding) after a response, so that the client
/// receives the response before the connection closes; see [`finish`].
const LINGER: Duration = Duration::from_millis(250);

/// The most bytes [`finish`] reads and discards.
const LINGER_BYTES: usize = 64 * 1024;

/// A request method the server accepts.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::http::{Method, parse_request};
/// let r = parse_request(b"POST /api/pause HTTP/1.1\r\n\r\n").unwrap();
/// assert_eq!(r.method, Method::Post);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `GET`: reads; never changes anything.
    Get,
    /// `POST`: sends a command.
    Post,
}

/// A parsed request head: what the server needs to route and check it.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::http::parse_request;
/// let r =
///     parse_request(b"GET /events?token=ab HTTP/1.1\r\nHost: 127.0.0.1:7878\r\n\r\n").unwrap();
/// assert_eq!(r.path, "/events");
/// assert_eq!(r.param("token"), Some("ab"));
/// assert_eq!(r.host.as_deref(), Some("127.0.0.1:7878"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The method.
    pub method: Method,
    /// The path, before any `?`.
    pub path: String,
    /// The query string's `name=value` pairs in order; a name without `=` has an empty
    /// value. Values are not percent-decoded: a `%` in the query is refused.
    pub query: Vec<(String, String)>,
    /// The `Host` header, if present.
    pub host: Option<String>,
    /// The `Origin` header, if present.
    pub origin: Option<String>,
}

impl Request {
    /// The value of the first query parameter called `name`.
    ///
    /// ```
    /// let r = cavitation_inspect::http::parse_request(b"GET /?a=1&a=2 HTTP/1.1\r\n\r\n").unwrap();
    /// assert_eq!(r.param("a"), Some("1"));
    /// assert_eq!(r.param("b"), None);
    /// ```
    pub fn param(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Why a request head was refused, with the status code it is answered with.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::http::{HttpError, parse_request};
/// let e = parse_request(b"PUT / HTTP/1.1\r\n\r\n").unwrap_err();
/// assert_eq!(e, HttpError::Method);
/// assert_eq!(e.status(), 405);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpError {
    /// Not a well-formed HTTP/1.x request head (400).
    Malformed,
    /// A `%` in the query string, which the server never produces (400).
    Query,
    /// A head above [`MAX_HEAD`] bytes (431).
    HeadTooLarge,
    /// A request body: `Content-Length` above 0, or any `Transfer-Encoding` (413).
    Body,
    /// A method other than `GET` and `POST` (405).
    Method,
}

impl HttpError {
    /// The status code the server answers with.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::http::HttpError::Body.status(), 413);
    /// ```
    pub fn status(self) -> u16 {
        match self {
            HttpError::Malformed | HttpError::Query => 400,
            HttpError::HeadTooLarge => 431,
            HttpError::Body => 413,
            HttpError::Method => 405,
        }
    }
}

impl fmt::Display for HttpError {
    /// Writes what was wrong with the request, as sent in the response body.
    ///
    /// ```
    /// let text = cavitation_inspect::http::HttpError::Query.to_string();
    /// assert!(text.contains('%'));
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            HttpError::Malformed => "malformed request",
            HttpError::Query => "the query string must not contain %",
            HttpError::HeadTooLarge => "request head larger than 8 KiB",
            HttpError::Body => "requests to this server carry no body",
            HttpError::Method => "only GET and POST are supported",
        })
    }
}

impl std::error::Error for HttpError {}

/// Parses a request head: the request line and the headers, with or without the blank line
/// that ends them. Pure: it reads nothing and checks nothing about the server (`Host`,
/// `Origin` and the token are checked by the server).
///
/// # Errors
///
/// [`HttpError::HeadTooLarge`] above [`MAX_HEAD`] bytes, [`HttpError::Method`] for a method
/// other than `GET` and `POST`, [`HttpError::Query`] for a `%` in the query string,
/// [`HttpError::Body`] for a `Content-Length` above 0 or a `Transfer-Encoding`, and
/// [`HttpError::Malformed`] for anything else that is not a well-formed HTTP/1.x head in
/// UTF-8 with an origin-form target (starting with `/`): a header without a colon, a folded
/// header line, a second `Host`, `Origin` or `Content-Length`.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::http::{HttpError, parse_request};
/// let r =
///     parse_request(b"GET /?token=1 HTTP/1.1\r\nOrigin: http://localhost:7878\r\n\r\n").unwrap();
/// assert_eq!(r.origin.as_deref(), Some("http://localhost:7878"));
/// assert_eq!(parse_request(b"GET /?a=%41 HTTP/1.1\r\n\r\n"), Err(HttpError::Query));
/// ```
pub fn parse_request(head: &[u8]) -> Result<Request, HttpError> {
    if head.len() > MAX_HEAD {
        return Err(HttpError::HeadTooLarge);
    }
    let end = head
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or(head.len());
    let text = std::str::from_utf8(&head[..end]).map_err(|_| HttpError::Malformed)?;
    let mut lines = text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l));
    let request_line = lines.next().ok_or(HttpError::Malformed)?;
    let mut parts = request_line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(HttpError::Malformed);
    };
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(HttpError::Malformed);
    }
    if !version.starts_with("HTTP/1.") || !target.starts_with('/') {
        return Err(HttpError::Malformed);
    }
    let method = match method {
        "GET" => Method::Get,
        "POST" => Method::Post,
        _ => return Err(HttpError::Method),
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    if query.contains('%') {
        return Err(HttpError::Query);
    }
    let query = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (key.to_string(), value.to_string())
        })
        .collect();

    let (mut host, mut origin, mut length) = (None, None, None);
    for line in lines {
        if line.is_empty() {
            break;
        }
        if line.starts_with([' ', '\t']) {
            return Err(HttpError::Malformed);
        }
        let (name, value) = line.split_once(':').ok_or(HttpError::Malformed)?;
        if name.is_empty() || name.contains([' ', '\t']) {
            return Err(HttpError::Malformed);
        }
        let value = value.trim_matches([' ', '\t']);
        let once = |slot: &mut Option<String>| {
            if slot.replace(value.to_string()).is_some() {
                Err(HttpError::Malformed)
            } else {
                Ok(())
            }
        };
        if name.eq_ignore_ascii_case("host") {
            once(&mut host)?;
        } else if name.eq_ignore_ascii_case("origin") {
            once(&mut origin)?;
        } else if name.eq_ignore_ascii_case("content-length") {
            once(&mut length)?;
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(HttpError::Body);
        }
    }
    if let Some(length) = length {
        let length: u64 = length.parse().map_err(|_| HttpError::Malformed)?;
        if length > 0 {
            return Err(HttpError::Body);
        }
    }
    Ok(Request {
        method,
        path: path.to_string(),
        query,
        host,
        origin,
    })
}

/// How reading a request head ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Head {
    /// The head is complete and is the first `n` bytes of the buffer.
    Complete(usize),
    /// More than [`MAX_HEAD`] bytes arrived without the head ending.
    TooLarge,
    /// The client closed the connection, or it failed.
    Closed,
    /// The head was not complete within the read timeout.
    TimedOut,
    /// The server is shutting down.
    Stopped,
}

/// Whether an I/O error means "try again later" on a non-blocking socket.
fn would_block(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
    )
}

/// Reads a request head from a non-blocking `stream` into `buf`, giving up after `timeout`
/// in total (not per read, so a client sending one byte at a time cannot hold the
/// connection longer) or as soon as `stop` is set.
pub(crate) fn read_head(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    timeout: Duration,
    stop: &AtomicBool,
) -> Head {
    let start = Instant::now();
    let mut chunk = [0u8; 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return Head::Closed,
            Ok(n) => {
                let from = buf.len().saturating_sub(3);
                buf.extend_from_slice(&chunk[..n]);
                if let Some(p) = buf[from..].windows(4).position(|w| w == b"\r\n\r\n") {
                    let len = from + p + 4;
                    return if len > MAX_HEAD {
                        Head::TooLarge
                    } else {
                        Head::Complete(len)
                    };
                }
                if buf.len() > MAX_HEAD {
                    return Head::TooLarge;
                }
            }
            Err(e) if would_block(&e) => {
                if stop.load(Ordering::Acquire) {
                    return Head::Stopped;
                }
                if start.elapsed() >= timeout {
                    return Head::TimedOut;
                }
                thread::sleep(POLL);
            }
            Err(_) => return Head::Closed,
        }
    }
}

/// Writes all of `data` to a non-blocking `stream`. Fails with `TimedOut` when no byte could
/// be written for `timeout` (a client that stopped reading), and with `Interrupted` as soon
/// as `stop` is set while waiting.
pub(crate) fn write_all(
    stream: &mut TcpStream,
    mut data: &[u8],
    timeout: Duration,
    stop: &AtomicBool,
) -> io::Result<()> {
    let mut progress = Instant::now();
    while !data.is_empty() {
        match stream.write(data) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                data = &data[n..];
                progress = Instant::now();
            }
            Err(e) if would_block(&e) => {
                if stop.load(Ordering::Acquire) {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "shutting down"));
                }
                if progress.elapsed() >= timeout {
                    return Err(io::ErrorKind::TimedOut.into());
                }
                thread::sleep(POLL);
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// The reason phrase for a status code the server sends.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

/// The start of every response head: the status line and the headers every response
/// carries. No response allows another origin to read it (no `Access-Control-Allow-*`),
/// none is cached, framed or sent as a referrer, and every connection closes after it.
pub(crate) fn head_start(status: u16) -> String {
    format!(
        "HTTP/1.1 {status} {}\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         X-Frame-Options: DENY\r\n\
         Referrer-Policy: no-referrer\r\n\
         Connection: close\r\n",
        reason(status)
    )
}

/// Writes a complete response with a body: `extra` holds further header lines, each ending
/// in `\r\n`.
pub(crate) fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    extra: &str,
    body: &[u8],
    timeout: Duration,
    stop: &AtomicBool,
) -> io::Result<()> {
    let mut head = head_start(status);
    if status != 204 {
        head.push_str(&format!(
            "Content-Type: {content_type}\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    head.push_str(extra);
    head.push_str("\r\n");
    let mut bytes = head.into_bytes();
    if status != 204 {
        bytes.extend_from_slice(body);
    }
    write_all(stream, &bytes, timeout, stop)
}

/// Ends a connection after its response: closes the sending side, then reads and discards
/// what the client still sends, for at most [`LINGER`] and [`LINGER_BYTES`], until it closes
/// its side. Closing a socket with unread data makes the system reset the connection, and a
/// reset can destroy the response before the client reads it (an oversized head, a body).
pub(crate) fn finish(stream: &mut TcpStream, stop: &AtomicBool) {
    let _ = stream.shutdown(Shutdown::Write);
    let start = Instant::now();
    let mut read = 0;
    let mut chunk = [0u8; 4096];
    while read < LINGER_BYTES && start.elapsed() < LINGER && !stop.load(Ordering::Acquire) {
        match stream.read(&mut chunk) {
            Ok(0) => return,
            Ok(n) => read += n,
            Err(e) if would_block(&e) => thread::sleep(POLL),
            Err(_) => return,
        }
    }
}
