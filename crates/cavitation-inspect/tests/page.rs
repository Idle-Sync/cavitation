//! Tests of the inspector page and its self-contained HTML export: the page served at `/`
//! with its Content Security Policy, no external request anywhere in it, and the export's
//! embedded snapshot, which must parse and must stay inside its `<script>` element whatever
//! the title, class names or messages hold; and the run's state replayed to a page opened
//! late.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::time::Duration;

use cavitation_core::{ParamStore, Rng, Tensor};
use cavitation_inspect::sse::{Event, Snapshot};
use cavitation_inspect::wire::{self, ModelInfo, RunState, Stats};
use cavitation_inspect::{Inspector, InspectorConfig};
use cavitation_nn::{
    Dataset, EpochReport, FitConfig, Health, Linear, Loss, LrSchedule, OptimizerConfig, Relu,
    Sequential, Targets,
};

mod common;

use common::Json;

/// The page, as the crate serves it.
const PAGE: &str = include_str!("../assets/inspector.html");

/// The policy the page is served with (before `frame-ancestors`, which only a header can
/// carry).
const PAGE_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'";

/// The policy of an exported document: the page's, with no connections at all.
const EXPORT_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'";

/// Text that tries to end the snapshot's element, open an HTML comment (which would change
/// how the browser reads the rest of the script element), and break a line in JavaScript.
const HOSTILE: &str = "</script><!--<script>\u{2028}\u{2029}<b>";

/// The opening tag of the embedded snapshot.
const SNAPSHOT_OPEN: &str = r#"<script id="snapshot" type="application/json">"#;

/// Text that would make a browser fetch something from elsewhere.
const EXTERNAL: [&str; 5] = ["http:", "https:", "src=\"//", "url(//", "@import"];

/// Starts an inspector on a port the system chooses, with `title`.
fn start(title: &str) -> Inspector {
    Inspector::start(InspectorConfig {
        address: "127.0.0.1:0".parse().unwrap(),
        title: title.to_string(),
        ..InspectorConfig::default()
    })
    .expect("the server starts")
}

/// Opens the event stream and returns a reader positioned after the response head.
fn open_events(inspector: &Inspector) -> BufReader<TcpStream> {
    let (addr, token) = endpoint(inspector);
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream
        .write_all(format!("GET /events?token={token} HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes())
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line).expect("the response head");
        if line == "\r\n" {
            return reader;
        }
        assert!(!line.is_empty(), "the stream closed before its head ended");
    }
}

/// The next event on the stream, skipping keep-alive comments: its name and its data,
/// parsed.
fn next_event(reader: &mut BufReader<TcpStream>) -> (String, Json) {
    let mut name = String::new();
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line).expect("an event line");
        assert!(!line.is_empty(), "the stream closed");
        let line = line.trim_end_matches('\n');
        if let Some(n) = line.strip_prefix("event: ") {
            name = n.to_string();
        } else if let Some(data) = line.strip_prefix("data: ") {
            return (name, Json::parse(data).expect("the event parses"));
        }
    }
}

/// 64 rows of 4 features with two classes.
fn small_data() -> Dataset {
    let mut rng = Rng::seed(5);
    let x: Vec<f32> = (0..64 * 4).map(|_| rng.uniform_f32()).collect();
    let labels = x.chunks(4).map(|r| usize::from(r[0] > r[1])).collect();
    Dataset {
        x: Tensor::from_vec(x, &[64, 4]).unwrap(),
        targets: Targets::Classes(labels),
    }
}

/// Three epochs of Adam at 1e-2, batch 16, quiet health checks.
fn fit_config() -> FitConfig<'static> {
    FitConfig {
        epochs: 3,
        batch: 16,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-2,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(1),
        schedule: LrSchedule::Constant,
        health: Health::quiet(),
        on_epoch: None,
        validation: None,
    }
}

/// The address and token of a running inspector, read from its URL.
fn endpoint(inspector: &Inspector) -> (SocketAddr, String) {
    let url = inspector.url();
    let rest = url.strip_prefix("http://").unwrap();
    let addr = rest[..rest.find('/').unwrap()].parse().unwrap();
    let token = url[url.find("token=").unwrap() + 6..].to_string();
    (addr, token)
}

/// Sends `GET target` with the server's own `Host`; returns the head and the body.
fn get(inspector: &Inspector, target: &str) -> (String, String) {
    let (addr, _) = endpoint(inspector);
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream
        .write_all(format!("GET {target} HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes())
        .unwrap();
    let mut bytes = Vec::new();
    let _ = stream.read_to_end(&mut bytes);
    let text = String::from_utf8(bytes).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").expect("a complete head");
    (head.to_string(), body.to_string())
}

/// The value of header `name` in `head`, case-insensitive.
fn header(head: &str, name: &str) -> Option<String> {
    head.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

/// A file name in the system's temporary folder, unique to this process and `name`.
fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "cavitation-inspect-{}-{name}.html",
        std::process::id()
    ))
}

/// Exports `inspector` to a temporary file and returns the file's text, read back from disk.
fn export_to_disk(inspector: &Inspector, name: &str) -> String {
    let path = temp_file(name);
    inspector.export_html(&path).expect("the export is written");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    text
}

/// The embedded snapshot's text: what lies between its opening tag and the next
/// `</script>`.
fn snapshot_text(html: &str) -> &str {
    let start = html.find(SNAPSHOT_OPEN).expect("the snapshot element") + SNAPSHOT_OPEN.len();
    let len = html[start..].find("</script>").expect("its closing tag");
    &html[start..start + len]
}

/// A 4 → 3 → ReLU → 2 network.
fn network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(0);
    Sequential::new()
        .push(Linear::new(params, 4, 3, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 3, 2, &mut rng))
}

/// Publishes a `model` event titled `title` whose first class is also named `title`, a
/// `stats` snapshot at step 42, an `epoch` event and a `status` event whose message is
/// `title`, as a run would.
fn publish_run(inspector: &Inspector, title: &str) {
    let mut params = ParamStore::new();
    let net = network(&mut params);
    let names = vec![title.to_string(), "yes".to_string()];
    let info = ModelInfo {
        title,
        input_width: 4,
        input_image: Some((2, 2)),
        class_names: Some(&names),
        probe_rows: 8,
        probe_is_training: true,
        eval_rows: 8,
        parameters: wire::parameter_count(&net, &params),
        flops_per_sample: wire::flops_per_sample(&net),
    };
    let hub = inspector.hub();
    assert!(hub.publish_event(Event::Model, |w| wire::model(w, &net, &info), true));
    let stats = Stats {
        step: 42,
        epoch: 2,
        epochs: 3,
        samples_per_second: 1234.5,
        gflops: 0.01,
        loss: 0.25,
        learning_rate: 1e-3,
        paused: false,
    };
    assert!(hub.publish_snapshot(Snapshot::Stats, |w| wire::stats(w, &stats)));
    let report = EpochReport::new(1, 0.5, 8, 0.01);
    assert!(hub.publish_event(Event::Epoch, |w| wire::epoch(w, &report, 3, None), true));
    assert!(hub.publish_event(
        Event::Status,
        |w| wire::status(w, RunState::Training, title),
        true
    ));
}

/// `/` serves the page itself, with the policy that lets its own script run and connect
/// only to this server, and nothing else load.
#[test]
fn page_served_with_csp() {
    let inspector = start("digits");
    let (head, body) = get(&inspector, "/");
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(
        header(&head, "Content-Type")
            .unwrap()
            .starts_with("text/html")
    );
    let policy = header(&head, "Content-Security-Policy").expect("a policy");
    assert_eq!(policy, format!("{PAGE_POLICY}; frame-ancestors 'none'"));
    assert_eq!(body, PAGE);
}

/// Nothing in the page reaches outside it: no URL with a scheme or a scheme-relative one,
/// no imported style sheet. The policy header blocks the rest.
#[test]
fn page_makes_no_external_requests() {
    let lower = PAGE.to_ascii_lowercase();
    for pattern in EXTERNAL {
        assert!(!lower.contains(pattern), "the page contains {pattern}");
    }
    // The page reads its token from its own address and talks only to its own server.
    assert!(PAGE.contains("location"));
    assert!(PAGE.contains("new EventSource('/events?token="));
    // One script, on lines of its own (whatever the line endings), so it can be checked on
    // its own.
    assert_eq!(PAGE.lines().filter(|l| *l == "<script>").count(), 1);
    assert_eq!(PAGE.lines().filter(|l| *l == "</script>").count(), 1);
    // An HTML comment on the first line says what the file is.
    let first = PAGE.lines().next().unwrap();
    assert!(first.starts_with("<!doctype html><!--"), "{first}");
}

/// The export embeds the latest snapshot and the events so far, and names no URL.
#[test]
fn export_contains_snapshot_and_no_urls() {
    let inspector = start("digits");
    publish_run(&inspector, "digits");
    let html = export_to_disk(&inspector, "contains");
    let snapshot = Json::parse(snapshot_text(&html)).expect("the snapshot parses");
    assert_eq!(snapshot.get("model").get("title").str(), "digits");
    assert_eq!(snapshot.get("stats").get("step").num(), 42.0);
    assert!(snapshot.get("probe").is_null());
    assert_eq!(snapshot.get("epochs").arr().len(), 1);
    assert_eq!(snapshot.get("epochs").at(0).get("epoch").num(), 1.0);
    assert_eq!(snapshot.get("status").get("state").str(), "training");
    assert!(snapshot.get("ablations").arr().is_empty());
    assert!(snapshot.get("shrinks").arr().is_empty());
    let lower = html.to_ascii_lowercase();
    for pattern in EXTERNAL {
        assert!(!lower.contains(pattern), "the export contains {pattern}");
    }
    // `GET /export` gives the same document, and writes nothing.
    let (_, token) = endpoint(&inspector);
    let (head, body) = get(&inspector, &format!("/export?token={token}"));
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert_eq!(body, html);
}

/// The export carries its own policy in a `<meta>` element before any script, with no
/// connections allowed, and `GET /export` sends the same policy as a header.
#[test]
fn export_has_csp_meta() {
    let inspector = start("digits");
    publish_run(&inspector, "digits");
    let html = export_to_disk(&inspector, "meta");
    let meta = format!(r#"<meta http-equiv="Content-Security-Policy" content="{EXPORT_POLICY}">"#);
    assert_eq!(html.matches(&meta).count(), 1, "{meta}");
    assert!(html.find(&meta).unwrap() < html.find("<script").unwrap());
    let (_, token) = endpoint(&inspector);
    let (head, _) = get(&inspector, &format!("/export?token={token}"));
    let policy = header(&head, "Content-Security-Policy").expect("a policy");
    assert_eq!(policy, format!("{EXPORT_POLICY}; frame-ancestors 'none'"));
}

/// The export is one HTML document: one `<html>`, `<head>` and `<body>`, the page's script
/// and the snapshot, which parses. An export before any run has a snapshot of nulls.
#[test]
fn export_is_valid_html_shape() {
    let inspector = start("digits");
    let empty = export_to_disk(&inspector, "empty");
    let snapshot = Json::parse(snapshot_text(&empty)).expect("the empty snapshot parses");
    assert!(snapshot.get("model").is_null());
    assert!(snapshot.get("stats").is_null());
    assert!(snapshot.get("status").is_null());
    publish_run(&inspector, "digits");
    let html = export_to_disk(&inspector, "shape");
    assert!(html.starts_with("<!doctype html>"));
    for (tag, count) in [
        ("<html", 1),
        ("</html>", 1),
        ("<head>", 1),
        ("</head>", 1),
        ("<body>", 1),
        ("</body>", 1),
        (SNAPSHOT_OPEN, 1),
        ("<script", 2),
        ("</script>", 2),
    ] {
        assert_eq!(html.matches(tag).count(), count, "{tag}");
    }
    assert!(html.find(SNAPSHOT_OPEN).unwrap() < html.find("<script>").unwrap());
    assert!(html.find("</body>").unwrap() < html.find("</html>").unwrap());
    Json::parse(snapshot_text(&html)).expect("the snapshot parses");
}

/// Text that tries to close the snapshot's element, open a comment or break a script line
/// stays inside it, in the title, a class name and a status message alike: the file holds
/// it escaped (checked on disk), the snapshot still parses back to the same text, and the
/// file has exactly one `</script>` for the snapshot.
#[test]
fn title_with_script_tag_stays_inside() {
    let inspector = start(HOSTILE);
    publish_run(&inspector, HOSTILE);
    let html = export_to_disk(&inspector, "title");
    assert!(
        html.contains(r"\u003c/script\u003e\u003c!--\u003cscript\u003e\u2028\u2029\u003cb\u003e")
    );
    assert!(!html.contains("<b>"));
    assert!(!html.contains('\u{2028}') && !html.contains('\u{2029}'));
    assert_eq!(html.matches("</script>").count(), 2);
    let after = html.find(SNAPSHOT_OPEN).unwrap() + SNAPSHOT_OPEN.len();
    assert!(!html[after..].contains("<!--"));
    let snapshot = Json::parse(snapshot_text(&html)).expect("the snapshot parses");
    assert_eq!(snapshot.get("model").get("title").str(), HOSTILE);
    assert_eq!(
        snapshot.get("model").get("class_names").at(0).str(),
        HOSTILE
    );
    assert_eq!(snapshot.get("status").get("message").str(), HOSTILE);
    // The snapshot's closing tag is the first `</script>` after it, and the page's script
    // follows.
    let close = after + html[after..].find("</script>").unwrap();
    assert!(html[close..].contains("<script>"));
}

/// An export after a run trained under the inspector holds the run's model, every epoch and
/// its final status.
#[test]
fn export_after_training() {
    let data = small_data();
    let mut params = ParamStore::new();
    let mut model = network(&mut params);
    let mut inspector = start("after training");
    cavitation_inspect::fit(&mut model, &mut params, &data, fit_config(), &mut inspector).unwrap();
    let html = export_to_disk(&inspector, "trained");
    let snapshot = Json::parse(snapshot_text(&html)).expect("the snapshot parses");
    assert_eq!(snapshot.get("model").get("title").str(), "after training");
    assert_eq!(snapshot.get("model").get("layers").arr().len(), 3);
    assert_eq!(snapshot.get("epochs").arr().len(), 3);
    assert_eq!(snapshot.get("status").get("state").str(), "finished");
    // No client was connected, so nothing was captured.
    assert!(snapshot.get("stats").is_null());
}

/// An export of a run a client watched holds every snapshot (statistics, probe row,
/// neuron, profile), escaped like the rest: the largest payloads, and they name no URL.
#[test]
fn export_of_a_watched_run_has_every_snapshot() {
    let data = small_data();
    let mut params = ParamStore::new();
    let mut model = network(&mut params);
    let mut inspector = Inspector::start(InspectorConfig {
        address: "127.0.0.1:0".parse().unwrap(),
        // A capture at every step: the 12 steps give each snapshot many chances, so one
        // publish refused by a busy lock cannot leave it empty.
        snapshot_hz: 1e6,
        title: HOSTILE.to_string(),
        class_names: Some(vec![HOSTILE.to_string(), "yes".to_string()]),
        ..InspectorConfig::default()
    })
    .unwrap();
    let _stream = open_events(&inspector);
    let start = std::time::Instant::now();
    while inspector.clients() == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the client registers"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    cavitation_inspect::fit(&mut model, &mut params, &data, fit_config(), &mut inspector).unwrap();
    let html = export_to_disk(&inspector, "watched");
    let snapshot = Json::parse(snapshot_text(&html)).expect("the snapshot parses");
    for name in ["stats", "probe", "neuron", "profile"] {
        assert!(!snapshot.get(name).is_null(), "{name} is null");
    }
    assert_eq!(snapshot.get("probe").get("layers").arr().len(), 3);
    assert_eq!(
        snapshot.get("model").get("class_names").at(0).str(),
        HOSTILE
    );
    let text = snapshot_text(&html);
    assert!(!text.contains('<'), "a raw < in the snapshot");
    let lower = html.to_ascii_lowercase();
    for pattern in EXTERNAL {
        assert!(!lower.contains(pattern), "the export contains {pattern}");
    }
}

/// A page that connects after the run has finished is told so at once: the stream starts
/// with the model, the last epoch and the `finished` status.
#[test]
fn status_replayed_to_late_client() {
    let data = small_data();
    let mut params = ParamStore::new();
    let mut model = network(&mut params);
    let mut inspector = start("late");
    cavitation_inspect::fit(&mut model, &mut params, &data, fit_config(), &mut inspector).unwrap();
    let mut stream = open_events(&inspector);
    let (name, model_event) = next_event(&mut stream);
    assert_eq!(name, "model");
    assert_eq!(model_event.get("title").str(), "late");
    let (name, epoch) = next_event(&mut stream);
    assert_eq!(name, "epoch");
    assert_eq!(epoch.get("epoch").num(), 3.0);
    let (name, status) = next_event(&mut stream);
    assert_eq!(name, "status");
    assert_eq!(status.get("state").str(), "finished");
}

/// A path the export cannot be written to is `Error::Io` naming it.
#[test]
fn export_to_missing_folder_is_io_error() {
    let inspector = start("digits");
    let path = std::env::temp_dir()
        .join(format!("cavitation-inspect-missing-{}", std::process::id()))
        .join("inspector.html");
    let error = inspector.export_html(&path).unwrap_err();
    assert!(
        matches!(error, cavitation_core::Error::Io { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains("inspector.html"), "{error}");
}
