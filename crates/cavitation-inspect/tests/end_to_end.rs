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
use cavitation_inspect::{ExploreConfig, Inspector, InspectorConfig};
use cavitation_nn::shrink::{Criterion, RetrainConfig, ShrinkConfig, Validation, shrink};
use cavitation_nn::{
    Dataset, FitConfig, Health, Linear, Loss, LrSchedule, OptimizerConfig, Relu, Sequential,
    Softmax, Tanh, Targets,
};

mod common;

use common::Json;

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
fn fit_config(epochs: usize) -> FitConfig<'static> {
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
        validation: None,
    }
}

/// [`fit_config`] whose `on_epoch` adds one to `epochs` and then, at the end of each epoch,
/// waits until `gate` is open (for at most [`DEADLINE`]), so a test can be sure its commands
/// reach the run before it ends.
fn gated(epochs: usize, gate: Arc<AtomicBool>, counter: Arc<AtomicUsize>) -> FitConfig<'static> {
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
        // The `finished` status can arrive before the last of the snapshots (the final
        // flush writes events first), so one read in this loop is kept.
        let is_finished =
            |m: &Message| m.name == "status" && m.json.get("state").str() == "finished";
        let mut finished = None;
        while !want.iter().all(|w| seen.contains(*w)) {
            assert!(start.elapsed() < DEADLINE, "events seen: {seen:?}");
            let m = client.next("the training events");
            seen.insert(m.name.clone());
            if m.name == "model" {
                assert_eq!(m.json.get("input_width").num(), 64.0);
                assert_eq!(m.json.get("layers").arr().len(), 3);
            }
            if is_finished(&m) {
                finished = Some(m);
            }
        }
        let history = run.join().unwrap().unwrap();
        assert_eq!(history.epochs.len(), 2);
        let finished = finished
            .unwrap_or_else(|| client.wait_for("status", |j| j.get("state").str() == "finished"));
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
/// gives the parameters of plain `fit`, bit for bit, on 1 and on 4 threads. The thread count is
/// process-wide, and training is bit-identical at any count, so tests running beside this one are
/// not disturbed.
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

/// An evaluation that costs more than the epochs after it is skipped at their ends, and the
/// first and the last epoch always carry one. (A 40,000-row evaluation costs more than the
/// four epochs between them, so their ends are under any budget. The budget's edge, 199
/// against 200 times the cost, is checked by the unit test of `evaluation_due`.)
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
    // For each epoch: whether it was evaluated, its seconds, and the evaluation's own cost.
    let mut epochs_seen = Vec::new();
    for m in client.until_finished().iter().filter(|m| m.name == "epoch") {
        let e = m.json.get("evaluation");
        let cost = if e.is_null() {
            None
        } else {
            assert_eq!(e.get("rows").num(), 40_000.0);
            assert!(e.get("misclassified").arr().len() <= 100);
            assert!((0.0..=1.0).contains(&e.get("accuracy").num()));
            Some(e.get("seconds").num())
        };
        epochs_seen.push((cost, m.json.get("seconds").num()));
    }
    assert_eq!(epochs_seen.len(), epochs);
    assert!(
        epochs_seen[0].0.is_some(),
        "the first epoch has no earlier evaluation to wait for"
    );
    assert!(
        epochs_seen[epochs - 1].0.is_some(),
        "the last epoch has no evaluation"
    );
    assert!(
        epochs_seen.iter().any(|e| e.0.is_none()),
        "no evaluation was skipped"
    );
    let (first_cost, middle): (f64, f64) = (
        epochs_seen[0].0.unwrap(),
        epochs_seen[1..epochs - 1].iter().map(|e| e.1).sum(),
    );
    assert!(
        middle < first_cost * f64::from(cavitation_inspect::EVALUATION_FACTOR),
        "the middle epochs ({middle:.4} s) are not under the budget of an evaluation costing \
         {first_cost:.4} s"
    );
    // Every epoch between the first and the last is skipped: each is far under the budget.
    assert!(
        epochs_seen[1..epochs - 1].iter().all(|e| e.0.is_none()),
        "an epoch under the budget was evaluated: {epochs_seen:?}"
    );
    client.close();
}

/// When training ends with a client connected, one last snapshot shows the final step and
/// epoch, even when the sampling period is far longer than the run, so the page does not
/// stop at the last sampled step.
#[test]
fn final_snapshot_shows_the_last_step() {
    let data = digits(256, 24);
    let mut params = ParamStore::new();
    let mut model = network(&mut params, 16);
    let mut inspector = Inspector::start(InspectorConfig {
        snapshot_hz: 0.01,
        ..inspector_config()
    })
    .unwrap();
    let (addr, token) = endpoint(&inspector);
    let client = Client::open(addr, &token);
    wait_clients(&inspector, 1);
    let epochs = 4;
    cavitation_inspect::fit(
        &mut model,
        &mut params,
        &data,
        fit_config(epochs),
        &mut inspector,
    )
    .unwrap();
    let stats: Vec<Message> = client
        .until_finished()
        .into_iter()
        .filter(|m| m.name == "stats")
        .collect();
    let last = stats.last().expect("no stats snapshot arrived");
    // 256 rows in batches of 32: 8 steps an epoch.
    assert_eq!(last.json.get("step").num(), (epochs * 8) as f64);
    assert_eq!(last.json.get("epoch").num(), epochs as f64);
    assert_eq!(last.json.get("epochs").num(), epochs as f64);
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

// ---- Explore mode ----------------------------------------------------------------------

/// The 64 → `hidden` → tanh → 10 network of [`network`], trained for 3 epochs on
/// `digits(512, 11)`, with that training set.
fn trained(hidden: usize) -> (Sequential, ParamStore, Dataset) {
    let data = digits(512, 11);
    let mut params = ParamStore::new();
    let model = network(&mut params, hidden);
    cavitation_nn::fit(&model, &mut params, &data, fit_config(3)).unwrap();
    (model, params, data)
}

/// A small shrink run: levels 1,000 bp to `max_bp` by 1,000, every level run (no
/// patience), one retraining epoch per level, measured on 300 held-out rows, and a stop rule
/// every level meets (100 accuracy points), so the kept level is the last one run.
fn shrink_config(max_bp: u32) -> ShrinkConfig<'static> {
    ShrinkConfig {
        max_removed_bp: max_bp,
        patience: None,
        criterion: Some(Criterion::AccuracyDrop { points: 100.0 }),
        validation: Some(Validation::Given(digits(300, 12))),
        retrain: RetrainConfig {
            epochs: 1,
            ..RetrainConfig::default()
        },
        ..ShrinkConfig::default()
    }
}

/// An explore configuration with cross-entropy, these training rows, and a shrink factory
/// for [`shrink_config`]`(max_bp)` when `max_bp` is given.
fn explore_config(train: Option<Dataset>, max_bp: Option<u32>) -> ExploreConfig<'static> {
    ExploreConfig {
        loss: Loss::CrossEntropy,
        train,
        shrink: max_bp.map(|bp| {
            Box::new(move || shrink_config(bp)) as Box<dyn FnMut() -> ShrinkConfig<'static> + Send>
        }),
    }
}

impl Client {
    /// Every event up to the first `status` whose message starts with `prefix`, that one
    /// included.
    fn until_status(&self, prefix: &str) -> Vec<Message> {
        let start = Instant::now();
        let mut out = Vec::new();
        loop {
            assert!(start.elapsed() < DEADLINE, "no status starting {prefix:?}");
            let m = self.next(prefix);
            let done = m.name == "status" && m.json.get("message").str().starts_with(prefix);
            out.push(m);
            if done {
                return out;
            }
        }
    }

    /// Waits for explore mode to begin: its `status` event in state `explore`.
    fn exploring(&self) -> Message {
        self.wait_for("status", |j| j.get("state").str() == "explore")
    }
}

/// The `shrink` events in `events`, in order.
fn shrink_events(events: &[Message]) -> Vec<&Json> {
    events
        .iter()
        .filter(|m| m.name == "shrink")
        .map(|m| &m.json)
        .collect()
}

/// The numbers in a JSON array.
fn numbers(j: &Json) -> Vec<f64> {
    j.arr().iter().map(Json::num).collect()
}

/// The neurons the neuron mask of `Linear` layer `linear` switches off.
fn off(model: &Sequential, linear: usize) -> Vec<usize> {
    model
        .neuron_mask(linear)
        .map(|m| m.off().collect())
        .unwrap_or_default()
}

/// Switching neuron 5 of `Linear` 0 off in explore mode publishes a probe at once with it
/// at exactly 0 and every other value of its layer unchanged, and the chosen neuron's panel
/// shows it off.
#[test]
fn explore_switch_recomputes_probe() {
    let (mut model, mut params, data) = trained(32);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        // Opened inside the scope: a failure closes it while unwinding, explore mode
        // returns after `idle_resume`, and the scope's join ends.
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), None),
            )
        });
        client.exploring();
        // The probe explore mode publishes when it starts.
        let before = client.wait_for("probe", |_| true);
        let a_before = numbers(before.json.get("layers").at(1).get("a"));
        assert_ne!(a_before[5], 0.0);
        assert_eq!(post(addr, &token, "neuron", "layer=1&neuron=5"), 204);
        client.wait_for("neuron", |j| j.get("neuron").num() == 5.0);
        assert_eq!(post(addr, &token, "mask", "linear=0&neuron=5&on=0"), 204);
        let after = client.wait_for("probe", |j| {
            numbers(j.get("layers").at(1).get("masked")) == [5.0]
        });
        let a_after = numbers(after.json.get("layers").at(1).get("a"));
        assert_eq!(a_after[5], 0.0);
        for (j, (x, y)) in a_before.iter().zip(&a_after).enumerate() {
            if j != 5 {
                assert_eq!(x, y, "neuron {j}");
            }
        }
        let neuron = client.wait_for("neuron", |j| !j.get("on").boolean());
        assert_eq!(neuron.json.get("neuron").num(), 5.0);
        assert_eq!(neuron.json.get("a").num(), 0.0);
        assert_eq!(post(addr, &token, "finish", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    assert_eq!(off(&model, 0), [5]);
}

/// Ranking a 16-wide hidden layer streams 16 `ablation` events in neuron order, then a
/// `status` with the ranking.
#[test]
fn explore_ablation_streams_every_neuron() {
    let (mut model, mut params, data) = trained(16);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), None),
            )
        });
        client.exploring();
        assert_eq!(post(addr, &token, "ablate", "linear=0"), 204);
        let events = client.until_status("ranked");
        let ablations: Vec<&Json> = events
            .iter()
            .filter(|m| m.name == "ablation")
            .map(|m| &m.json)
            .collect();
        let neurons: Vec<f64> = ablations.iter().map(|j| j.get("neuron").num()).collect();
        let want: Vec<f64> = (0..16).map(f64::from).collect();
        assert_eq!(neurons, want);
        assert!(ablations.iter().all(|j| j.get("linear").num() == 0.0));
        assert!(ablations.iter().all(|j| !j.get("loss").is_null()));
        // One ranking, one sweep number.
        let sweep = ablations[0].get("sweep").num();
        assert!(ablations.iter().all(|j| j.get("sweep").num() == sweep));
        let ranking = events.last().unwrap();
        assert_eq!(ranking.json.get("state").str(), "explore");
        let message = ranking.json.get("message").str();
        assert!(message.contains("16 neurons"), "{message}");
        // No probe set was configured: the rows are training rows, and the message says so.
        assert!(message.contains("training rows"), "{message}");
        // A second ranking is a new sweep, so the page does not mix the two.
        assert_eq!(post(addr, &token, "ablate", "linear=0"), 204);
        let again = client.until_status("ranked");
        let second = find(&again, "ablation").json.get("sweep").num();
        assert!(second > sweep, "{second} after {sweep}");
        assert_eq!(post(addr, &token, "finish", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    // The sweep leaves the model as it was.
    assert!(model.neuron_mask(0).is_none());
}

/// A shrink run streams one `shrink` event per level, the same levels a plain `shrink`
/// run finds, and leaves the model and its parameters at the kept level, bit for bit.
#[test]
fn explore_shrink_streams_steps() {
    let (mut model, mut params, data) = trained(32);
    let (mut plain, mut plain_params) = (model.clone(), params.clone());
    let report = shrink(&mut plain, &mut plain_params, &data, shrink_config(3000)).unwrap();
    assert_eq!(report.steps.len(), 3);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), Some(3000)),
            )
        });
        client.exploring();
        assert_eq!(post(addr, &token, "shrink", ""), 204);
        let events = client.until_status("shrink finished");
        let levels = shrink_events(&events);
        assert_eq!(levels.len(), 3);
        for (level, step) in levels.iter().zip(&report.steps) {
            assert_eq!(level.get("removed_bp").num(), f64::from(step.removed_bp));
            let kept: Vec<usize> = level
                .get("kept")
                .arr()
                .iter()
                .map(|k| k.num() as usize)
                .collect();
            assert_eq!(kept, step.kept);
            assert_eq!(level.get("met").boolean(), step.met);
        }
        let number = levels[0].get("run").num();
        assert!(levels.iter().all(|j| j.get("run").num() == number));
        // The network's size and the page's view of switched-off neurons follow the kept
        // level: a `network` snapshot with its parameter count and a probe with its masked
        // neurons, in either order.
        let chosen = &report.steps[report.chosen.unwrap()];
        let masked = off(&plain, 0).len();
        let (mut sized, mut probed) = (false, false);
        let start = Instant::now();
        while !(sized && probed) {
            assert!(
                start.elapsed() < DEADLINE,
                "network {sized}, probe {probed}"
            );
            let m = client.next("the kept level");
            match m.name.as_str() {
                "network" => {
                    let p = m.json.get("parameters");
                    sized |= !p.is_null() && p.num() == chosen.parameters as f64;
                }
                "probe" => {
                    probed |= m.json.get("layers").at(1).get("masked").arr().len() == masked;
                }
                _ => {}
            }
        }
        assert_eq!(post(addr, &token, "finish", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    assert_eq!(off(&model, 0), off(&plain, 0));
    assert!(!off(&model, 0).is_empty());
    assert_eq!(bits(&model, &params), bits(&plain, &plain_params));
}

/// `Cancel` ends a shrink run after the level in progress and keeps the best level so far:
/// the model is the one a plain run of that one level gives. A command that cannot run
/// beside a shrink run is answered with a `status` message.
#[test]
fn explore_shrink_cancel_keeps_best() {
    let (mut model, mut params, data) = trained(32);
    let (mut plain, mut plain_params) = (model.clone(), params.clone());
    shrink(&mut plain, &mut plain_params, &data, shrink_config(1000)).unwrap();
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        // Queued before explore mode starts, so the run reads the switch and the cancel
        // after its first level, whatever the machine's speed.
        assert_eq!(post(addr, &token, "shrink", ""), 204);
        assert_eq!(post(addr, &token, "mask", "linear=0&neuron=1&on=0"), 204);
        assert_eq!(post(addr, &token, "cancel", ""), 204);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), Some(9500)),
            )
        });
        let events = client.until_status("shrink cancelled");
        assert_eq!(shrink_events(&events).len(), 1);
        // The cancel is acknowledged before the level in progress ends.
        let ack = events
            .iter()
            .position(|m| {
                m.name == "status" && m.json.get("message").str().starts_with("cancelling")
            })
            .expect("a cancelling status");
        let level = events.iter().position(|m| m.name == "shrink").unwrap();
        assert!(ack < level, "acknowledged at {ack}, level at {level}");
        assert!(events.iter().any(|m| {
            m.name == "status" && m.json.get("message").str().contains("in progress")
        }));
        assert_eq!(post(addr, &token, "finish", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    assert_eq!(off(&model, 0), off(&plain, 0));
    assert_eq!(bits(&model, &params), bits(&plain, &plain_params));
}

/// The shrink lab runs again on a later click, on the network the first run left.
#[test]
fn explore_shrink_runs_twice() {
    let (mut model, mut params, data) = trained(32);
    let (mut plain, mut plain_params) = (model.clone(), params.clone());
    shrink(&mut plain, &mut plain_params, &data, shrink_config(2000)).unwrap();
    shrink(&mut plain, &mut plain_params, &data, shrink_config(2000)).unwrap();
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), Some(2000)),
            )
        });
        client.exploring();
        for _ in 0..2 {
            assert_eq!(post(addr, &token, "shrink", ""), 204);
            let events = client.until_status("shrink finished");
            let levels: Vec<f64> = shrink_events(&events)
                .iter()
                .map(|j| j.get("removed_bp").num())
                .collect();
            assert_eq!(levels, [1000.0, 2000.0]);
        }
        assert_eq!(post(addr, &token, "finish", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
    assert_eq!(off(&model, 0), off(&plain, 0));
    assert_eq!(bits(&model, &params), bits(&plain, &plain_params));
}

/// Without training rows, or without a shrink factory, `Shrink` is answered with a `status`
/// message naming what is missing, and the model is left alone.
#[test]
fn explore_without_train_refuses_shrink() {
    let (mut model, mut params, data) = trained(16);
    let before = bits(&model, &params);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    for (train, max_bp, missing) in [
        (None, Some(2000), "ExploreConfig::train"),
        (Some(data.clone()), None, "ExploreConfig::shrink"),
    ] {
        thread::scope(|s| {
            let client = Client::open(addr, &token);
            wait_clients(&inspector, 1);
            let (model, params, inspector) = (&mut model, &mut params, &mut inspector);
            // The configuration holds a callback that is not `Send`: it is made on the
            // thread that explores.
            let run =
                s.spawn(move || inspector.explore(model, params, explore_config(train, max_bp)));
            client.exploring();
            assert_eq!(post(addr, &token, "shrink", ""), 204);
            let events = client.until_status("cannot shrink");
            assert!(shrink_events(&events).is_empty());
            let refusal = events.last().unwrap();
            assert_eq!(refusal.json.get("state").str(), "explore");
            assert!(
                refusal.json.get("message").str().contains(missing),
                "{}",
                refusal.raw
            );
            assert_eq!(post(addr, &token, "finish", ""), 204);
            run.join().unwrap().unwrap();
            client.wait_for("status", |j| j.get("state").str() == "finished");
            client.close();
        });
    }
    assert!(model.neuron_mask(0).is_none());
    assert_eq!(bits(&model, &params), before);
}

/// `Finish` from the page ends explore mode: `explore` returns and the page is told.
#[test]
fn explore_finish_returns() {
    let (mut model, mut params, data) = trained(16);
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), None),
            )
        });
        client.exploring();
        // Commands for training only are answered, and explore mode goes on.
        assert_eq!(post(addr, &token, "pause", ""), 204);
        let m = client.wait_for("status", |_| true);
        assert_eq!(m.json.get("state").str(), "explore");
        assert_eq!(post(addr, &token, "finish", ""), 204);
        let start = Instant::now();
        run.join().unwrap().unwrap();
        assert!(start.elapsed() < Duration::from_secs(5));
        client.wait_for("status", |j| j.get("state").str() == "finished");
        client.close();
    });
}

/// With no client for `idle_resume`, `explore` returns on its own.
#[test]
fn explore_returns_when_idle() {
    let (mut model, mut params, data) = trained(16);
    let mut inspector = Inspector::start(InspectorConfig {
        idle_resume: Duration::from_millis(300),
        ..inspector_config()
    })
    .unwrap();
    let start = Instant::now();
    inspector
        .explore(&mut model, &mut params, explore_config(Some(data), None))
        .unwrap();
    let took = start.elapsed();
    assert!(took >= Duration::from_millis(300), "{took:?}");
    assert!(took < Duration::from_secs(1), "{took:?}");
}

/// Ranking the neurons of a model with no hidden `Linear` is answered with a `status`
/// message in the library's words, as is a `Linear` the model does not have; explore mode
/// goes on.
#[test]
fn ablate_on_model_without_hidden_linear_reports_status() {
    let data = digits(256, 13);
    let mut params = ParamStore::new();
    let mut model = Sequential::new().push(Linear::new(&mut params, 64, 10, &mut Rng::seed(2)));
    let mut inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        let run = s.spawn(|| {
            inspector.explore(
                &mut model,
                &mut params,
                explore_config(Some(data.clone()), None),
            )
        });
        client.exploring();
        for (query, words) in [
            ("linear=0", "last Linear layer"),
            ("linear=3", "1 Linear layer"),
        ] {
            assert_eq!(post(addr, &token, "ablate", query), 204);
            let events = client.until_status("cannot rank");
            assert!(events.iter().all(|m| m.name != "ablation"));
            let m = events.last().unwrap();
            assert_eq!(m.json.get("state").str(), "explore");
            assert!(m.json.get("message").str().contains(words), "{}", m.raw);
            assert!(
                !m.json.get("message").str().contains("1 Linear layers"),
                "{}",
                m.raw
            );
        }
        assert_eq!(post(addr, &token, "finish", ""), 204);
        run.join().unwrap().unwrap();
        client.close();
    });
}

/// A program that drops the inspector as soon as explore mode returns still gets its last
/// `status` to the page: `Drop` lets the clients read what is queued before it closes.
#[test]
fn explore_finished_reaches_the_page_before_drop() {
    let (mut model, mut params, data) = trained(16);
    let inspector = Inspector::start(inspector_config()).unwrap();
    let (addr, token) = endpoint(&inspector);
    thread::scope(|s| {
        let client = Client::open(addr, &token);
        wait_clients(&inspector, 1);
        assert_eq!(post(addr, &token, "finish", ""), 204);
        let (model, params) = (&mut model, &mut params);
        let run = s.spawn(move || {
            let mut inspector = inspector;
            let config = ExploreConfig {
                train: Some(data),
                ..ExploreConfig::default()
            };
            let result = inspector.explore(model, params, config);
            let start = Instant::now();
            drop(inspector);
            (result, start.elapsed())
        });
        let finished = client.wait_for("status", |j| j.get("state").str() == "finished");
        assert!(
            finished
                .json
                .get("message")
                .str()
                .contains("finished from the page")
        );
        let (result, dropping) = run.join().unwrap();
        result.unwrap();
        assert!(dropping < Duration::from_secs(1), "{dropping:?}");
        client.close();
    });
}
