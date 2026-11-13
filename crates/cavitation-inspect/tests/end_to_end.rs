//! End-to-end tests of training under the inspector: a real server, a real event stream
//! read by a plain `TcpStream` client, commands sent as `POST` requests, and
//! `cavitation_inspect::fit` running on its own thread, on synthetic data shaped like the
//! digits set (64 features in `[0, 1)`, 10 classes).
//!
//! Every wait has a deadline, so a bug fails a test instead of hanging the binary. A test
//! that pauses training holds the event stream open; if it fails while the run is paused,
//! unwinding closes the stream, the run resumes after its short `idle_resume`, and the
//! training thread ends.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use cavitation_core::kernels::set_threads;
use cavitation_core::{ParamStore, Rng, Tensor};
use cavitation_inspect::{Inspector, InspectorConfig};
use cavitation_nn::{
    Dataset, FitConfig, Health, Linear, Loss, LrSchedule, OptimizerConfig, Relu, Sequential,
    Softmax, Tanh, Targets,
};

/// The longest any one wait in these tests may take.
const DEADLINE: Duration = Duration::from_secs(30);

/// `rows` rows of 64 features in `[0, 1)` with class labels 0 to 9, from `Rng::seed(seed)`;
/// the label is the index of the largest of the first ten features, so the set can be
/// learned.
fn digits(rows: usize, seed: u64) -> Dataset {
    let mut rng = Rng::seed(seed);
    let x: Vec<f32> = (0..rows * 64).map(|_| rng.uniform_f32()).collect();
    let labels = x
        .chunks(64)
        .map(|row| (0..10).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap())
        .collect();
    Dataset {
        x: Tensor::from_vec(x, &[rows, 64]).unwrap(),
        targets: Targets::Classes(labels),
    }
}

/// The 64 → `hidden` → tanh → 10 network from `Rng::seed(1)`: `Linear` 0 is layer 0 and
/// its neurons are read at layer 1, the `Tanh`, where a value is exactly 0 only when it is
/// switched off.
fn network(params: &mut ParamStore, hidden: usize) -> Sequential {
    let mut rng = Rng::seed(1);
    Sequential::new()
        .push(Linear::new(params, 64, hidden, &mut rng))
        .push(Tanh)
        .push(Linear::new(params, hidden, 10, &mut rng))
}

/// A training configuration: Adam at 1e-3, batch 32, shuffled with seed 3, quiet health
/// checks.
fn fit_config(epochs: usize) -> FitConfig {
    FitConfig {
        epochs,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(3),
        schedule: LrSchedule::Constant,
        health: Health::quiet(),
        on_epoch: None,
    }
}

/// [`fit_config`] whose `on_epoch` adds one to `epochs` and then, at the end of each epoch,
/// waits until `gate` is open (for at most [`DEADLINE`]), so a test can be sure its commands
/// reach the run before it ends.
fn gated(epochs: usize, gate: Arc<AtomicBool>, counter: Arc<AtomicUsize>) -> FitConfig {
    FitConfig {
        on_epoch: Some(Box::new(move |_, _, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            let start = Instant::now();
            while !gate.load(Ordering::SeqCst) && start.elapsed() < DEADLINE {
                thread::sleep(Duration::from_millis(1));
            }
        })),
        ..fit_config(epochs)
    }
}

/// A closed gate and an epoch counter for [`gated`].
fn gate() -> (Arc<AtomicBool>, Arc<AtomicUsize>) {
    (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicUsize::new(0)),
    )
}

/// An inspector configuration on a port the system chooses, sampling at 1,000 Hz so every
/// test sees snapshots quickly, and resuming a paused run after 500 ms without a client.
fn inspector_config() -> InspectorConfig {
    InspectorConfig {
        address: "127.0.0.1:0".parse().unwrap(),
        snapshot_hz: 1000.0,
        idle_resume: Duration::from_millis(500),
        ..InspectorConfig::default()
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

/// Sends `POST /api/<command>?token=…&<query>` and returns the status code.
fn post(addr: SocketAddr, token: &str, command: &str, query: &str) -> u16 {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.set_read_timeout(Some(DEADLINE)).unwrap();
    let sep = if query.is_empty() { "" } else { "&" };
    let raw =
        format!("POST /api/{command}?token={token}{sep}{query} HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    stream.write_all(raw.as_bytes()).unwrap();
    let mut bytes = Vec::new();
    let _ = stream.read_to_end(&mut bytes);
    let text = String::from_utf8_lossy(&bytes);
    text.split(' ').nth(1).unwrap().parse().unwrap()
}

/// One event from the stream: its name, its raw data and the data parsed.
#[derive(Debug, Clone)]
struct Message {
    /// The event name.
    name: String,
    /// The `data:` line as sent.
    raw: String,
    /// The data, parsed.
    json: Json,
}

/// A client reading `/events` on its own thread, handing each event over a channel.
struct Client {
    /// The events read so far and not yet taken.
    events: mpsc::Receiver<Message>,
    /// The connection, kept to close it.
    stream: TcpStream,
    /// The reading thread.
    reader: Option<thread::JoinHandle<()>>,
}

impl Client {
    /// Opens the event stream and starts reading it.
    fn open(addr: SocketAddr, token: &str) -> Client {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.set_read_timeout(Some(DEADLINE)).unwrap();
        let raw = format!("GET /events?token={token} HTTP/1.1\r\nHost: {addr}\r\n\r\n");
        stream.write_all(raw.as_bytes()).unwrap();
        let (send, events) = mpsc::channel();
        let read_side = stream.try_clone().unwrap();
        let reader = thread::spawn(move || {
            let mut reader = BufReader::new(read_side);
            let mut line = String::new();
            // The response head.
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                if line == "\r\n" {
                    break;
                }
            }
            let (mut name, mut data) = (String::new(), String::new());
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let text = line.trim_end_matches('\n');
                if text.is_empty() {
                    if !name.is_empty() {
                        let json = Json::parse(&data)
                            .unwrap_or_else(|e| panic!("event {name}: {e} in {data}"));
                        let message = Message {
                            name: std::mem::take(&mut name),
                            raw: std::mem::take(&mut data),
                            json,
                        };
                        if send.send(message).is_err() {
                            return;
                        }
                    }
                } else if let Some(value) = text.strip_prefix("event: ") {
                    name = value.to_string();
                } else if let Some(value) = text.strip_prefix("data: ") {
                    data = value.to_string();
                }
            }
        });
        Client {
            events,
            stream,
            reader: Some(reader),
        }
    }

    /// The next event, or a panic naming `what` after the deadline.
    fn next(&self, what: &str) -> Message {
        self.events
            .recv_timeout(DEADLINE)
            .unwrap_or_else(|_| panic!("no event while waiting for {what}"))
    }

    /// The next event named `name` for which `pick` is true.
    fn wait_for(&self, name: &str, mut pick: impl FnMut(&Json) -> bool) -> Message {
        let start = Instant::now();
        loop {
            assert!(start.elapsed() < DEADLINE, "no matching {name} event");
            let m = self.next(name);
            if m.name == name && pick(&m.json) {
                return m;
            }
        }
    }

    /// Every event that arrives within `period`.
    fn collect_for(&self, period: Duration) -> Vec<Message> {
        let end = Instant::now() + period;
        let mut out = Vec::new();
        while let Some(left) = end.checked_duration_since(Instant::now()) {
            match self.events.recv_timeout(left) {
                Ok(m) => out.push(m),
                Err(_) => break,
            }
        }
        out
    }

    /// Every event up to the `status` event that ends the run, then whatever follows within
    /// 300 ms (the last snapshots are written after the events of the same batch). For a
    /// run that has already ended: nothing is skipped, so no event is lost to a search.
    fn until_finished(&self) -> Vec<Message> {
        let start = Instant::now();
        let mut out = Vec::new();
        loop {
            assert!(start.elapsed() < DEADLINE, "the run never finished");
            let m = self.next("the end of the run");
            let done = m.name == "status" && m.json.get("state").str() == "finished";
            out.push(m);
            if done {
                break;
            }
        }
        out.extend(self.collect_for(Duration::from_millis(300)));
        out
    }

    /// Pauses a run made with [`gated`]: waits for its `model` event (the run has begun),
    /// sends `pause`, opens the gate, and returns the first paused `stats`. The pause is
    /// read in time however fast the machine is, since the run waits at the end of epoch 1
    /// until the gate opens.
    fn pause(&self, addr: SocketAddr, token: &str, gate: &AtomicBool) -> Message {
        self.wait_for("model", |_| true);
        assert_eq!(post(addr, token, "pause", ""), 204);
        gate.store(true, Ordering::SeqCst);
        // The `status` event is published before the paused `stats` snapshot, and events
        // reach the client ahead of snapshots, so it is read first.
        self.wait_for("status", |j| j.get("state").str() == "paused");
        self.wait_for("stats", |j| j.get("paused").boolean())
    }

    /// Closes the connection and waits for the reading thread.
    fn close(mut self) {
        self.shut();
    }

    /// Shuts the connection down and joins the reader.
    fn shut(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for Client {
    /// Closes the connection, so a paused run resumes when a test fails.
    fn drop(&mut self) {
        self.shut();
    }
}

/// The last event named `name` in `events`.
fn find<'a>(events: &'a [Message], name: &str) -> &'a Message {
    events
        .iter()
        .rev()
        .find(|m| m.name == name)
        .unwrap_or_else(|| panic!("no {name} event"))
}

/// Waits until the inspector counts `n` clients.
fn wait_clients(inspector: &Inspector, n: usize) {
    let start = Instant::now();
    while inspector.clients() != n {
        assert!(
            start.elapsed() < DEADLINE,
            "the client count never reached {n}"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

/// The bits of every parameter value of `model`, in order.
fn bits(model: &Sequential, params: &ParamStore) -> Vec<u32> {
    model
        .parameter_names()
        .into_iter()
        .flat_map(|(_, id)| {
            params
                .value(id)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A connected client gets every kind of training event over a 2-epoch run.
#[test]
fn events_arrive_during_training() {
    let data = digits(512, 7);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 32);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        // Opened inside the scope, so a failure closes it while unwinding, the paused run
        // resumes after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            cavitation_inspect::fit(
                &mut model,
                &mut params,
                &data,
                fit_config(2),
                &mut inspector,
            )
        });
        let mut seen = std::collections::BTreeSet::new();
        let want = ["epoch", "model", "probe", "profile", "stats"];
        let start = Instant::now();
        while !want.iter().all(|w| seen.contains(*w)) {
            assert!(start.elapsed() < DEADLINE, "events seen: {seen:?}");
            let m = client.next("the training events");
            seen.insert(m.name.clone());
            if m.name == "model" {
                assert_eq!(m.json.get("input_width").num(), 64.0);
                assert_eq!(m.json.get("layers").arr().len(), 3);
            }
        }
        let history = run.join().unwrap().unwrap();
        assert_eq!(history.epochs.len(), 2);
        let finished = client.wait_for("status", |j| j.get("state").str() == "finished");
        assert!(!finished.raw.is_empty());
        client.close();
    });
}

/// Switching neuron 5 of `Linear` 0 off from the page shows it at exactly 0 in a later
/// probe (layer 1, the `Tanh` that holds the neurons), and the mask stays on the model.
#[test]
fn mask_switch_reaches_the_probe() {
    let data = digits(1024, 8);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 32);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    let (open, epochs) = gate();
    thread::scope(|s| {
        // Opened inside the scope, so a failure closes it while unwinding, the paused run
        // resumes after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            let config = gated(4, Arc::clone(&open), epochs);
            cavitation_inspect::fit(&mut model, &mut params, &data, config, &mut inspector)
        });
        client.pause(addr, &token, &open);
        let before = client.wait_for("probe", |_| true);
        assert_ne!(before.json.get("layers").at(1).get("a").at(5).num(), 0.0);
        assert_eq!(post(addr, &token, "mask", "linear=0&neuron=5&on=0"), 204);
        let after = client.wait_for("probe", |j| {
            let masked = j.get("layers").at(1).get("masked");
            masked.arr().iter().any(|m| m.num() == 5.0)
        });
        let layer = after.json.get("layers").at(1);
        assert_eq!(layer.get("a").at(5).num(), 0.0);
        assert!(
            layer
                .get("a")
                .arr()
                .iter()
                .filter(|v| v.num() == 0.0)
                .count()
                == 1
        );
        assert_eq!(post(addr, &token, "resume", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    assert_eq!(model.neuron_mask(0).unwrap().off().collect::<Vec<_>>(), [5]);
}

/// While paused no `stats` step advances; after `resume` it does.
#[test]
fn pause_and_resume() {
    let data = digits(1024, 9);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 64);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    let (open, epochs) = gate();
    thread::scope(|s| {
        // Opened inside the scope, so a failure closes it while unwinding, the paused run
        // resumes after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            let config = gated(4, Arc::clone(&open), epochs);
            cavitation_inspect::fit(&mut model, &mut params, &data, config, &mut inspector)
        });
        let paused = client.pause(addr, &token, &open);
        let step = paused.json.get("step").num();
        // A command while paused is still served (and recaptures): the step stays put.
        assert_eq!(post(addr, &token, "probe", "index=3"), 204);
        let held = client.collect_for(Duration::from_millis(400));
        for m in held.iter().filter(|m| m.name == "stats") {
            assert_eq!(m.json.get("step").num(), step, "a step ran while paused");
            assert!(m.json.get("paused").boolean());
        }
        assert!(
            held.iter()
                .any(|m| m.name == "probe" && m.json.get("row").num() == 3.0),
            "the probe choice was not served while paused"
        );
        assert_eq!(post(addr, &token, "resume", ""), 204);
        client.wait_for("stats", |j| j.get("step").num() > step);
        run.join().unwrap().unwrap();
        client.close();
    });
}

/// A paused run resumes once no client has been connected for `idle_resume`: training goes
/// on within 1 s of the last client closing.
#[test]
fn pause_resumes_when_client_leaves() {
    let data = digits(1024, 10);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 64);
    let mut inspector = Inspector::start(InspectorConfig {
        idle_resume: Duration::from_millis(300),
        ..inspector_config()
    })
    .unwrap();
    let (addr, token) = endpoint(&inspector);
    let (open, epochs) = gate();
    let counter = Arc::clone(&epochs);
    thread::scope(|s| {
        // Opened inside the scope, so a failure closes it while unwinding, the paused run
        // resumes after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            // Made on the training thread: a `FitConfig` holds callbacks and is not `Send`.
            let config = gated(4, Arc::clone(&open), counter);
            cavitation_inspect::fit(&mut model, &mut params, &data, config, &mut inspector)
        });
        client.pause(addr, &token, &open);
        let held = epochs.load(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(400));
        assert_eq!(
            epochs.load(Ordering::SeqCst),
            held,
            "an epoch ended while paused"
        );
        client.close();
        let closed = Instant::now();
        while epochs.load(Ordering::SeqCst) == held {
            assert!(
                closed.elapsed() < Duration::from_secs(1),
                "training did not go on within 1 s of the client leaving"
            );
            thread::sleep(Duration::from_millis(2));
        }
        run.join().unwrap().unwrap();
    });
}

/// A learning rate sent from the page replaces the cosine schedule for the rest of the run.
///
/// Every step is sampled (`snapshot_hz` 10⁶), and the events are read to the end of the run
/// with the snapshots that follow it, so the last `stats` comes from the last epoch however
/// fast the machine trains.
#[test]
fn learning_rate_command_overrides_schedule() {
    let data = digits(512, 11);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 32);
    let mut inspector = Inspector::start(InspectorConfig {
        snapshot_hz: 1e6,
        ..inspector_config()
    })
    .unwrap();
    let (addr, token) = endpoint(&inspector);
    let (open, epochs) = gate();
    thread::scope(|s| {
        // Opened inside the scope, so a failure closes it while unwinding, the paused run
        // resumes after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            let config = FitConfig {
                optimizer: OptimizerConfig::Sgd {
                    lr: 0.05,
                    momentum: 0.0,
                    weight_decay: 0.0,
                },
                schedule: LrSchedule::Cosine { min_lr: 0.0 },
                ..gated(8, Arc::clone(&open), epochs)
            };
            cavitation_inspect::fit(&mut model, &mut params, &data, config, &mut inspector)
        });
        let paused = client.pause(addr, &token, &open);
        // The run pauses in epoch 1 or early in epoch 2, where the schedule is between 0.048
        // and 0.05.
        let scheduled = paused.json.get("learning_rate").num();
        assert!(scheduled > 0.04 && scheduled <= 0.05, "{scheduled}");
        let paused_epoch = paused.json.get("epoch").num();
        assert!(paused_epoch <= 2.0, "paused in epoch {paused_epoch}");
        assert_eq!(post(addr, &token, "lr", "value=0.00123"), 204);
        // Refused by the server before it reaches the run.
        assert_eq!(post(addr, &token, "lr", "value=-1"), 400);
        assert_eq!(post(addr, &token, "resume", ""), 204);
        run.join().unwrap().unwrap();
        let stats: Vec<Message> = client
            .until_finished()
            .into_iter()
            .filter(|m| m.name == "stats")
            .collect();
        let from = stats
            .iter()
            .position(|m| m.json.get("learning_rate").num() == 0.00123)
            .expect("no stats snapshot shows the new rate");
        // Every later snapshot keeps it, through the epochs the schedule would have set.
        for m in &stats[from..] {
            assert_eq!(m.json.get("learning_rate").num(), 0.00123);
        }
        let last = stats.last().unwrap();
        assert_eq!(
            last.json.get("epoch").num(),
            8.0,
            "the last stats is not from epoch 8"
        );
        client.close();
    });
}

/// With one client reading and a probe run after nearly every step (no commands), training
/// gives the parameters of plain `fit`, bit for bit, on 1 and on 4 threads. The thread count is process-wide, and training is bit-identical at any
/// count, so tests running beside this one are not disturbed.
#[test]
fn probe_leaves_training_untouched() {
    let data = digits(640, 12);
    for threads in [1, 4] {
        set_threads(threads);
        let mut plain_params = ParamStore::new();
        let plain = network(&mut plain_params, 128);
        cavitation_nn::fit(&plain, &mut plain_params, &data, fit_config(2)).unwrap();

        let mut params = ParamStore::new();
        let mut model = network(&mut params, 128);
        // A capture is due after every step; each capture runs the probe.
        let mut inspector = Inspector::start(InspectorConfig {
            snapshot_hz: 1e6,
            ..inspector_config()
        })
        .unwrap();
        let (addr, token) = endpoint(&inspector);
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let history = cavitation_inspect::fit(
            &mut model,
            &mut params,
            &data,
            fit_config(2),
            &mut inspector,
        )
        .unwrap();
        // 640 rows in batches of 32: 20 steps an epoch.
        assert_eq!(history.epochs.len(), 2);
        let steps = 40;
        let probed = inspector.snapshots_taken();
        assert!(
            probed >= steps - 2,
            "the probe ran after only {probed} of {steps} steps"
        );
        let probes = client
            .until_finished()
            .into_iter()
            .filter(|m| m.name == "probe")
            .count();
        assert!(probes > 0, "the client read no probe");
        assert!(
            bits(&plain, &plain_params) == bits(&model, &params),
            "the parameters differ from plain fit at {threads} threads"
        );
        client.close();
    }
}

/// An evaluation that costs more than 1% of an epoch is skipped at the epoch ends that
/// follow it, and the last epoch always carries one.
#[test]
fn evaluation_skipped_when_budget_spent() {
    let data = digits(256, 13);
    let probe = digits(40_000, 14);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 64);
    let mut inspector = Inspector::start(InspectorConfig {
        probe: Some(probe),
        eval_rows: 40_000,
        ..inspector_config()
    })
    .unwrap();
    let (addr, token) = endpoint(&inspector);
    let client = Client::open(addr, &token);
    wait_clients(&inspector, 1);
    let epochs = 6;
    cavitation_inspect::fit(
        &mut model,
        &mut params,
        &data,
        fit_config(epochs),
        &mut inspector,
    )
    .unwrap();
    let mut evaluations = Vec::new();
    for m in client.until_finished().iter().filter(|m| m.name == "epoch") {
        let e = m.json.get("evaluation");
        evaluations.push(!e.is_null());
        if !e.is_null() {
            assert_eq!(e.get("rows").num(), 40_000.0);
            assert!(e.get("misclassified").arr().len() <= 100);
            assert!((0.0..=1.0).contains(&e.get("accuracy").num()));
        }
    }
    assert_eq!(evaluations.len(), epochs);
    assert!(
        evaluations[0],
        "the first epoch has no earlier evaluation to wait for"
    );
    assert!(
        evaluations.iter().any(|e| !e),
        "no evaluation was skipped: {evaluations:?}"
    );
    assert!(evaluations[epochs - 1], "the last epoch has no evaluation");
    client.close();
}

/// A model ending in `Softmax`: its layer has no element-wise derivative, so `z`, f′ and
/// ∂L/∂z are `null` there, and the event is valid JSON.
#[test]
fn softmax_layer_probe_has_nulls() {
    let data = digits(256, 15);
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let mut model = Sequential::new()
        .push(Linear::new(&mut params, 64, 16, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut params, 16, 10, &mut rng))
        .push(Softmax);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    let client = Client::open(addr, &token);
    wait_clients(&inspector, 1);
    cavitation_inspect::fit(
        &mut model,
        &mut params,
        &data,
        fit_config(1),
        &mut inspector,
    )
    .unwrap();
    let events = client.until_finished();
    let probe = find(&events, "probe");
    let layers = probe.json.get("layers");
    let softmax = layers.at(3);
    assert_eq!(softmax.get("a").arr().len(), 10);
    assert!(softmax.get("z").is_null());
    assert!(softmax.get("df").is_null());
    assert!(softmax.get("grad_z").is_null());
    assert_eq!(softmax.get("grad_a").arr().len(), 10);
    // The ReLU has all five.
    let relu = layers.at(1);
    for key in ["a", "z", "df", "grad_a", "grad_z"] {
        assert_eq!(relu.get(key).arr().len(), 16, "{key}");
    }
    // A Linear has no element-wise input or derivative.
    assert!(layers.at(0).get("df").is_null());
    client.close();
}

/// A probe row holding NaN gives `null` values, never the text `NaN`.
#[test]
fn nan_in_probe_is_null() {
    let data = digits(256, 16);
    let mut probe = digits(8, 17);
    let mut x = probe.x.data().to_vec();
    x[0] = f32::NAN;
    probe.x = Tensor::from_vec(x, &[8, 64]).unwrap();
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 16);
    let mut inspector = Inspector::start(InspectorConfig {
        probe: Some(probe),
        ..inspector_config()
    })
    .unwrap();
    let (addr, token) = endpoint(&inspector);
    let client = Client::open(addr, &token);
    wait_clients(&inspector, 1);
    cavitation_inspect::fit(
        &mut model,
        &mut params,
        &data,
        fit_config(1),
        &mut inspector,
    )
    .unwrap();
    let events = client.until_finished();
    let probe = find(&events, "probe");
    assert!(!probe.raw.contains("NaN") && !probe.raw.contains("inf"));
    assert!(probe.json.get("input").at(0).is_null());
    assert!(probe.json.get("layers").at(0).get("a").at(0).is_null());
    assert!(probe.json.get("loss").is_null());
    let neuron = find(&events, "neuron");
    assert!(!neuron.raw.contains("NaN"));
    client.close();
}

/// A network with one output (value targets): the probe has width-one layers at the end,
/// and no label or predicted class.
#[test]
fn width_one_output_probe() {
    let mut rng = Rng::seed(18);
    let x: Vec<f32> = (0..128 * 4).map(|_| rng.uniform_f32()).collect();
    let y: Vec<f32> = x.chunks(4).map(|r| r.iter().sum::<f32>()).collect();
    let data = Dataset {
        x: Tensor::from_vec(x, &[128, 4]).unwrap(),
        targets: Targets::Values(Tensor::from_vec(y, &[128, 1]).unwrap()),
    };
    let mut params = ParamStore::new();
    let mut model = Sequential::new()
        .push(Linear::new(&mut params, 4, 8, &mut rng))
        .push(Tanh)
        .push(Linear::new(&mut params, 8, 1, &mut rng));
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    let client = Client::open(addr, &token);
    wait_clients(&inspector, 1);
    let config = FitConfig {
        loss: Loss::Mse,
        ..fit_config(2)
    };
    cavitation_inspect::fit(&mut model, &mut params, &data, config, &mut inspector).unwrap();
    let events = client.until_finished();
    let probe = find(&events, "probe");
    let last = probe.json.get("layers").at(2);
    assert_eq!(last.get("a").arr().len(), 1);
    assert_eq!(last.get("grad_a").arr().len(), 1);
    assert!(probe.json.get("label").is_null());
    assert!(probe.json.get("predicted").is_null());
    assert_eq!(probe.json.get("target").arr().len(), 1);
    let epoch = events
        .iter()
        .find(|m| m.name == "epoch" && !m.json.get("evaluation").is_null())
        .expect("an epoch with an evaluation");
    let evaluation = epoch.json.get("evaluation");
    assert!(evaluation.get("accuracy").is_null());
    assert!(!evaluation.get("r2").is_null());
    client.close();
}

/// One inspector serves two runs whose parameters live in different stores: the second run
/// probes its own store (the probe is rebuilt for each run) instead of panicking.
#[test]
fn two_runs_on_two_stores() {
    let data = digits(256, 23);
    let mut inspector = Inspector::start(InspectorConfig {
        snapshot_hz: 1e6,
        ..inspector_config()
    })
    .unwrap();
    let (addr, token) = endpoint(&inspector);
    let client = Client::open(addr, &token);
    wait_clients(&inspector, 1);
    let mut first_params = ParamStore::new();
    let mut first = network(&mut first_params, 16);
    cavitation_inspect::fit(
        &mut first,
        &mut first_params,
        &data,
        fit_config(1),
        &mut inspector,
    )
    .unwrap();
    let after_first = inspector.snapshots_taken();
    assert!(after_first > 0);
    client.until_finished();
    let mut second_params = ParamStore::new();
    let mut second = network(&mut second_params, 24);
    cavitation_inspect::fit(
        &mut second,
        &mut second_params,
        &data,
        fit_config(1),
        &mut inspector,
    )
    .unwrap();
    assert!(inspector.snapshots_taken() > after_first);
    let events = client.until_finished();
    let probe = find(&events, "probe");
    assert_eq!(probe.json.get("layers").at(1).get("a").arr().len(), 24);
    client.close();
}

/// With no client connected, nothing is captured.
#[test]
fn no_capture_without_clients() {
    let data = digits(512, 19);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 32);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    cavitation_inspect::fit(
        &mut model,
        &mut params,
        &data,
        fit_config(2),
        &mut inspector,
    )
    .unwrap();
    assert_eq!(inspector.snapshots_taken(), 0);
}

/// Without a probe set the probe rows are training rows, and the `model` event says so; with
/// one, it says they are not.
#[test]
fn probe_defaults_to_training_rows() {
    for given in [false, true] {
        let data = digits(256, 20);
        let mut params = ParamStore::new();
        let mut model = network(&mut params, 16);
        let mut inspector = Inspector::start(InspectorConfig {
            probe: given.then(|| digits(50, 21)),
            ..inspector_config()
        })
        .unwrap();
        let (addr, token) = endpoint(&inspector);
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        cavitation_inspect::fit(
            &mut model,
            &mut params,
            &data,
            fit_config(1),
            &mut inspector,
        )
        .unwrap();
        let events = client.until_finished();
        let model_event = find(&events, "model");
        assert_eq!(model_event.json.get("probe_is_training").boolean(), !given);
        let rows = if given { 50.0 } else { 256.0 };
        assert_eq!(model_event.json.get("probe_rows").num(), rows);
        client.close();
    }
}

/// Commands the run cannot carry out are answered with a `status` message, not a panic: a
/// neuron out of range, a `Linear` that does not exist, a probe row past the end, and the
/// explore-mode commands during training.
#[test]
fn bad_commands_get_a_status_message() {
    let data = digits(512, 22);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 32);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    let (open, epochs) = gate();
    thread::scope(|s| {
        // Opened inside the scope, so a failure closes it while unwinding, the paused run
        // resumes after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            let config = gated(3, Arc::clone(&open), epochs);
            cavitation_inspect::fit(&mut model, &mut params, &data, config, &mut inspector)
        });
        client.pause(addr, &token, &open);
        for (command, query) in [
            ("mask", "linear=0&neuron=32&on=0"),
            ("mask", "linear=2&neuron=0&on=0"),
            // The output layer: switching it off would change what the network answers.
            ("mask", "linear=1&neuron=3&on=0"),
            // Above the page's maximum learning rate.
            ("lr", "value=11"),
            ("lr", "value=1e300"),
            ("probe", "index=512"),
            ("neuron", "layer=3&neuron=0"),
            ("neuron", "layer=1&neuron=32"),
            ("ablate", "linear=0"),
            ("shrink", ""),
        ] {
            assert_eq!(post(addr, &token, command, query), 204);
            let m = client.wait_for("status", |_| true);
            assert_eq!(m.json.get("state").str(), "paused", "{command}");
            assert!(!m.json.get("message").str().is_empty(), "{command}");
        }
        assert_eq!(post(addr, &token, "resume", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    assert!(model.neuron_mask(0).is_none());
    assert!(model.neuron_mask(2).is_none());
}

/// A parsed JSON value: enough of JSON to check the events.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object, its members in order.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses a whole document; an error says where it went wrong.
    fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = p.value()?;
        p.space();
        if p.at != p.bytes.len() {
            return Err(format!("trailing text at byte {}", p.at));
        }
        Ok(value)
    }

    /// Member `key` of an object; panics if there is none.
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(members) => members
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("no member {key} in {self:?}")),
            _ => panic!("not an object: {self:?}"),
        }
    }

    /// Element `i` of an array.
    fn at(&self, i: usize) -> &Json {
        &self.arr()[i]
    }

    /// The elements of an array.
    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            _ => panic!("not an array: {self:?}"),
        }
    }

    /// The value of a number.
    fn num(&self) -> f64 {
        match self {
            Json::Num(x) => *x,
            _ => panic!("not a number: {self:?}"),
        }
    }

    /// The value of a string.
    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string: {self:?}"),
        }
    }

    /// The value of a boolean.
    fn boolean(&self) -> bool {
        match self {
            Json::Bool(b) => *b,
            _ => panic!("not a boolean: {self:?}"),
        }
    }

    /// Whether this is `null`.
    fn is_null(&self) -> bool {
        *self == Json::Null
    }
}

/// A recursive-descent JSON parser over bytes.
struct Parser<'a> {
    /// The text.
    bytes: &'a [u8],
    /// The next byte to read.
    at: usize,
}

impl Parser<'_> {
    /// Skips white space.
    fn space(&mut self) {
        while self.at < self.bytes.len() && b" \t\r\n".contains(&self.bytes[self.at]) {
            self.at += 1;
        }
    }

    /// Reads `word` or fails.
    fn expect(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }

    /// Reads one value.
    fn value(&mut self) -> Result<Json, String> {
        self.space();
        match self.bytes.get(self.at) {
            None => Err("unexpected end".to_string()),
            Some(b'n') => self.expect("null").map(|()| Json::Null),
            Some(b't') => self.expect("true").map(|()| Json::Bool(true)),
            Some(b'f') => self.expect("false").map(|()| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.bytes.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("bad array at byte {}", self.at)),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.space();
                if self.bytes.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(members));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    self.expect(":")?;
                    let value = self.value()?;
                    members.push((key, value));
                    self.space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Json::Obj(members));
                        }
                        _ => return Err(format!("bad object at byte {}", self.at)),
                    }
                }
            }
            Some(_) => self.number(),
        }
    }

    /// Reads a string, decoding its escapes.
    fn string(&mut self) -> Result<String, String> {
        self.expect("\"")?;
        let mut out = String::new();
        loop {
            let Some(&b) = self.bytes.get(self.at) else {
                return Err("unterminated string".to_string());
            };
            self.at += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.bytes[self.at];
                    self.at += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.bytes[self.at..self.at + 4])
                                .map_err(|e| e.to_string())?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            self.at += 4;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("bad escape at byte {}", self.at)),
                    }
                }
                b if b < 0x20 => return Err(format!("raw control byte at {}", self.at)),
                _ => {
                    // Copy the whole UTF-8 sequence.
                    let start = self.at - 1;
                    let len = match b {
                        0xF0.. => 4,
                        0xE0.. => 3,
                        0xC0.. => 2,
                        _ => 1,
                    };
                    self.at = start + len;
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..self.at])
                            .map_err(|e| e.to_string())?,
                    );
                }
            }
        }
    }

    /// Reads a number in JSON's grammar (no NaN, no infinity, no leading `+`).
    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        if self.bytes.get(self.at) == Some(&b'-') {
            self.at += 1;
        }
        let digits = |p: &mut Self| {
            let s = p.at;
            while p.bytes.get(p.at).is_some_and(u8::is_ascii_digit) {
                p.at += 1;
            }
            p.at > s
        };
        if !digits(self) {
            return Err(format!("bad number at byte {start}"));
        }
        if self.bytes.get(self.at) == Some(&b'.') {
            self.at += 1;
            if !digits(self) {
                return Err(format!("bad fraction at byte {start}"));
            }
        }
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.bytes.get(self.at), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !digits(self) {
                return Err(format!("bad exponent at byte {start}"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).unwrap();
        text.parse()
            .map(Json::Num)
            .map_err(|e| format!("{e} at byte {start}"))
    }
}
