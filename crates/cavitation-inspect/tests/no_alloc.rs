//! Counts heap allocations made on the training thread while a network trains under the
//! inspector with one client reading the event stream; any allocation after warm-up fails
//! the run.
//!
//! This target has no test harness: `main` runs the scenario on the main thread, which is
//! the training thread. The counting allocator counts an allocation only while a
//! thread-local flag is set, and only the training thread sets it, between steps it has
//! chosen to count. The server's threads, the client threads of this file and the kernels'
//! thread pool allocate freely; the training thread must not.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cavitation_core::{ParamStore, Rng, Tensor};
use cavitation_inspect::{Inspector, InspectorConfig};
use cavitation_nn::{
    Dataset, EpochReport, FitConfig, Flow, Health, Hooks, Init, Linear, Loss, LrSchedule,
    ModelHandle, OptimizerConfig, Relu, Sequential, StepControl, StepInfo, Targets, TrainerHandle,
    fit_controlled,
};

/// The allocations counted so far: those made while the current thread's [`COUNTING`] flag
/// was set.
static COUNTED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Whether this thread's allocations are counted. Only the training thread ever sets it.
    /// A `const` initialiser with no destructor needs no allocation to reach, so the
    /// allocator can read it.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread's allocations are counted now; `false` on a thread whose
/// thread-locals are gone.
fn counting() -> bool {
    COUNTING.try_with(Cell::get).unwrap_or(false)
}

/// Starts or stops counting the current thread's allocations.
fn set_counting(on: bool) {
    COUNTING.with(|c| c.set(on));
}

/// The system allocator, counting in [`COUNTED`] every allocation and reallocation made
/// while the calling thread's [`COUNTING`] flag is set.
struct CountingAllocator;

// SAFETY: every method forwards its arguments unchanged to `System`, which implements
// `GlobalAlloc` correctly; counting reads a thread-local `Cell` and touches an atomic, and
// neither allocates.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if counting() {
            COUNTED.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if counting() {
            COUNTED.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract, passed on unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if counting() {
            COUNTED.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `ptr` and `layout` came from this allocator, which hands out `System`'s
        // blocks unchanged, and the caller upholds the rest of `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from this allocator, which hands out `System`'s
        // blocks unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Installs the counting allocator for the whole process.
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The longest any one wait in this file may take.
const DEADLINE: Duration = Duration::from_secs(60);

/// Training rows: as many as the digits training set, so batch 128 gives 12 steps an epoch.
const ROWS: usize = 1437;

/// The batch size.
const BATCH: usize = 128;

/// Steps in each epoch.
const STEPS_PER_EPOCH: u64 = ROWS.div_ceil(BATCH) as u64;

/// Epochs counted after the warm-up.
const MEASURED_EPOCHS: usize = 4;

/// The most warm-up epochs before the scenario gives up waiting for a refused publish.
const MAX_WARMUP_EPOCHS: usize = 300;

/// `rows` rows of 64 features in `[0, 1)` with class labels 0 to 9 from `Rng::seed(seed)`,
/// shaped like the digits; the label is the index of the largest of the first ten features.
fn digits_like(rows: usize, seed: u64) -> Dataset {
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

/// The 50k digits network, 64 → 256 → ReLU → 128 → ReLU → 10, He uniform weights from
/// `Rng::seed(1)`. `Linear`s 0 and 1 (layers 0 and 2) are hidden, so their neurons can be
/// switched from the page.
fn network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(1);
    let mut he = |inputs, outputs| {
        Linear::with_init(
            params,
            inputs,
            outputs,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        )
    };
    Sequential::new()
        .push(he(64, 256))
        .push(Relu)
        .push(he(256, 128))
        .push(Relu)
        .push(he(128, 10))
}

/// The digits benchmark's training: SGD at 0.01, batch 128, the rows in order, quiet health
/// checks.
fn fit_config(epochs: usize) -> FitConfig {
    FitConfig {
        epochs,
        batch: BATCH,
        optimizer: OptimizerConfig::Sgd {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: None,
        schedule: LrSchedule::Constant,
        health: Health::quiet(),
        on_epoch: None,
    }
}

/// The commands the page can send, as `POST /api/<name>?<query>`.
const COMMANDS: [(&str, &str); 13] = [
    ("mask", "linear=0&neuron=5&on=0"),
    ("mask", "linear=0&neuron=5&on=1"),
    ("mask", "linear=1&neuron=3&on=0"),
    ("mask", "linear=1&neuron=3&on=1"),
    ("neuron", "layer=0&neuron=7"),
    ("neuron", "layer=1&neuron=200"),
    ("neuron", "layer=2&neuron=4"),
    ("neuron", "layer=3&neuron=100"),
    ("neuron", "layer=4&neuron=9"),
    ("probe", "index=5"),
    ("probe", "index=0"),
    ("lr", "value=0.01"),
    // Refused (there is no layer 9), so the run answers with a `status` event.
    ("neuron", "layer=9&neuron=0"),
];

/// The commands sent at step `k` (counting from 1) of the first warm-up epoch: command
/// `k − 1`, one a step so each neuron choice is captured, and at the epoch's last step also
/// the refused one, so every path runs before counting starts.
fn warm_up_commands(k: u64) -> &'static [usize] {
    const ONE_EACH: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    match k {
        1..=11 => {
            let i = (k - 1) as usize;
            &ONE_EACH[i..=i]
        }
        12 => &[11, 12],
        _ => &[],
    }
}

/// The command sent at step `k` of a counted epoch: neuron switches on both hidden
/// `Linear`s, neuron and probe choices and a learning rate, none of which the run refuses.
fn counted_command(k: u64) -> Option<usize> {
    match k {
        2 => Some(0),
        3 => Some(5),
        4 => Some(2),
        6 => Some(9),
        7 => Some(1),
        8 => Some(3),
        9 => Some(7),
        10 => Some(11),
        11 => Some(10),
        _ => None,
    }
}

/// What the client threads share with the training thread.
struct Shared {
    /// The number of the latest command the training thread asked for.
    asked: AtomicUsize,
    /// Which command that is, as an index into [`COMMANDS`].
    which: AtomicUsize,
    /// The number of the latest command the server has queued.
    queued: AtomicUsize,
    /// Set by the driver when a command was not answered 204.
    failed: AtomicBool,
    /// Set by the training thread when the clients should stop.
    stop: AtomicBool,
    /// Exports fetched so far.
    exports: AtomicUsize,
}

/// Sends `POST /api/<command>?token=…&<query>` and returns the status code (0 when the
/// answer cannot be read).
fn post(addr: SocketAddr, token: &str, command: &str, query: &str) -> u16 {
    let Ok(mut stream) = TcpStream::connect(addr) else {
        return 0;
    };
    let _ = stream.set_read_timeout(Some(DEADLINE));
    let raw = format!("POST /api/{command}?token={token}&{query} HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    if stream.write_all(raw.as_bytes()).is_err() {
        return 0;
    }
    let mut bytes = Vec::new();
    let _ = stream.read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes)
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0)
}

/// Sends each command the training thread asks for, and says when the server has queued it
/// (it answers 204 only after queueing).
fn driver(addr: SocketAddr, token: &str, shared: &Shared) {
    let mut done = 0;
    while !shared.stop.load(Ordering::SeqCst) {
        let asked = shared.asked.load(Ordering::SeqCst);
        if asked == done {
            thread::sleep(Duration::from_micros(200));
            continue;
        }
        let (command, query) = COMMANDS[shared.which.load(Ordering::SeqCst)];
        if post(addr, token, command, query) != 204 {
            shared.failed.store(true, Ordering::SeqCst);
        }
        done = asked;
        shared.queued.store(done, Ordering::SeqCst);
    }
}

/// Fetches `GET /export` every millisecond or so until told to stop. Each export holds the
/// hub's lock while it gathers the run so far, so the training thread's `try_lock` is
/// sometimes refused and takes the path that retries at the next step. The pause keeps the
/// number of connections, each of which leaves a closed socket behind for a while, modest.
fn exporter(addr: SocketAddr, token: &str, shared: &Shared) {
    let raw = format!("GET /export?token={token} HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    let mut sink = Vec::new();
    while !shared.stop.load(Ordering::SeqCst) {
        let Ok(mut stream) = TcpStream::connect(addr) else {
            continue;
        };
        let _ = stream.set_read_timeout(Some(DEADLINE));
        if stream.write_all(raw.as_bytes()).is_ok() {
            sink.clear();
            let _ = stream.read_to_end(&mut sink);
            shared.exports.fetch_add(1, Ordering::SeqCst);
        }
        thread::sleep(Duration::from_millis(1));
    }
}

/// Opens the event stream and reads it until the server closes it, discarding the bytes.
/// Returns the connection, to close it, and the reading thread.
fn reader(addr: SocketAddr, token: &str) -> (TcpStream, thread::JoinHandle<()>) {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!("GET /events?token={token} HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    stream.write_all(raw.as_bytes()).unwrap();
    let mut read_side = stream.try_clone().unwrap();
    let thread = thread::spawn(move || {
        let mut buffer = vec![0u8; 1 << 16];
        while matches!(read_side.read(&mut buffer), Ok(n) if n > 0) {}
    });
    (stream, thread)
}

/// Where the run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Every path is taken once; nothing is counted.
    WarmUp,
    /// Steps are counted.
    Counted,
}

/// What the counted run saw.
#[derive(Debug, Default)]
struct Tally {
    /// Warm-up epochs before counting started.
    warm_up_epochs: usize,
    /// Steps counted.
    steps: usize,
    /// Counted steps on which the inspector captured a snapshot.
    sampled: usize,
    /// Commands sent during the counted epochs.
    commands: usize,
    /// Captures a busy hub refused during the warm-up.
    refused_warm_up: usize,
    /// Captures a busy hub refused during the counted epochs.
    refused_counted: usize,
}

/// The control the run trains with: the inspector's own [`StepControl`], with counting
/// switched on around the steps to count and the page's commands sent at chosen steps.
struct Counted<'a> {
    /// The inspector, which does the real work between steps.
    inspector: &'a mut Inspector,
    /// The state shared with the client threads.
    shared: &'a Shared,
    /// Where the run is.
    phase: Phase,
    /// Counted epochs still to run.
    left: usize,
    /// Whether the last `after_step` captured a snapshot.
    captured: bool,
    /// Commands asked for so far.
    asked: usize,
    /// What the run saw.
    tally: Tally,
}

impl Counted<'_> {
    /// Asks the driver for command `which` and waits until the server has queued it, so the
    /// inspector reads it in the `after_step` that follows.
    fn send(&mut self, which: usize) {
        self.asked += 1;
        self.shared.which.store(which, Ordering::SeqCst);
        self.shared.asked.store(self.asked, Ordering::SeqCst);
        let start = Instant::now();
        while self.shared.queued.load(Ordering::SeqCst) != self.asked {
            assert!(start.elapsed() < DEADLINE, "command {which} was not queued");
            thread::sleep(Duration::from_micros(100));
        }
    }
}

impl StepControl for Counted<'_> {
    /// Stops counting, notes whether the previous capture was refused, sends this step's
    /// command, then runs the inspector's `after_step` (counted in the counted epochs) and
    /// leaves counting on for the next step unless this one ends its epoch.
    fn after_step(
        &mut self,
        info: &StepInfo,
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        set_counting(false);
        // A capture that went through asks for the next step's profile, so a capture with
        // no profile after it was refused by a busy hub.
        if self.captured && trainer.last_profile().is_none() {
            match self.phase {
                Phase::WarmUp => self.tally.refused_warm_up += 1,
                Phase::Counted => self.tally.refused_counted += 1,
            }
        }
        let k = (info.step - 1) % STEPS_PER_EPOCH + 1;
        match self.phase {
            Phase::WarmUp if self.tally.warm_up_epochs == 0 => {
                for &which in warm_up_commands(k) {
                    self.send(which);
                }
            }
            Phase::WarmUp => {}
            Phase::Counted => {
                if let Some(which) = counted_command(k) {
                    self.send(which);
                    self.tally.commands += 1;
                }
            }
        }
        let before = self.inspector.snapshots_taken();
        set_counting(self.phase == Phase::Counted);
        let flow = self.inspector.after_step(info, model, params, trainer);
        set_counting(false);
        self.captured = self.inspector.snapshots_taken() > before;
        if self.phase == Phase::Counted {
            self.tally.steps += 1;
            self.tally.sampled += usize::from(self.captured);
        }
        // The epoch's end (its health checks and report) is not a step, so it is not
        // counted; the next epoch's first step is, from the end of `after_epoch`.
        set_counting(self.phase == Phase::Counted && k != STEPS_PER_EPOCH);
        flow
    }

    /// Runs the inspector's `after_step`, uncounted (it publishes the `epoch` event, which
    /// copies a string per client), then moves the run on: counting starts once the warm-up
    /// has taken every path, and the run stops after [`MEASURED_EPOCHS`] counted epochs.
    fn after_epoch(
        &mut self,
        report: &EpochReport,
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        set_counting(false);
        let flow = self.inspector.after_epoch(report, model, params, trainer);
        match self.phase {
            Phase::WarmUp => {
                self.tally.warm_up_epochs += 1;
                if self.tally.refused_warm_up > 0 {
                    self.phase = Phase::Counted;
                } else if self.tally.warm_up_epochs >= MAX_WARMUP_EPOCHS {
                    return Flow::Stop;
                }
            }
            Phase::Counted => {
                self.left -= 1;
                if self.left == 0 {
                    return Flow::Stop;
                }
            }
        }
        set_counting(self.phase == Phase::Counted);
        flow
    }
}

/// Trains the 50k network on digit-shaped rows under the inspector with one client reading
/// `/events`, while other client threads send the page's commands and fetch exports.
///
/// The warm-up is one epoch of `cavitation_inspect::fit` (which sets the run up: the probe
/// rows, the layer widths, the sampling period, and the first held-out evaluation, which
/// builds the evaluator), then epochs of `fit_controlled` with the inspector as the control
/// until every path has been taken: a snapshot published, a publish refused by a busy hub, a
/// `status` event, commands drained, the writers woken with `notify_all`, and neuron
/// switches through `set_neuron` on both hidden `Linear`s (the first creates each mask).
/// Then [`MEASURED_EPOCHS`] epochs are counted, every step and its `after_step`, with
/// commands at chosen steps; epoch ends are not counted. Returns the allocations counted.
fn inspector_training_thread() -> usize {
    let train = digits_like(ROWS, 2);
    let probe = digits_like(360, 3);
    let mut inspector = Inspector::start(InspectorConfig {
        address: "127.0.0.1:0".parse().unwrap(),
        snapshot_hz: 250.0,
        probe: Some(probe),
        input_image: Some((8, 8)),
        ..InspectorConfig::default()
    })
    .unwrap();
    let url = inspector.url();
    let rest = url.strip_prefix("http://").unwrap();
    let addr: SocketAddr = rest[..rest.find('/').unwrap()].parse().unwrap();
    let token = url[url.find("token=").unwrap() + 6..].to_string();
    let shared = Shared {
        asked: AtomicUsize::new(0),
        which: AtomicUsize::new(0),
        queued: AtomicUsize::new(0),
        failed: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        exports: AtomicUsize::new(0),
    };
    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    let (stream, reading) = reader(addr, &token);
    let start = Instant::now();
    while inspector.clients() == 0 {
        assert!(start.elapsed() < DEADLINE, "the client did not connect");
        thread::sleep(Duration::from_millis(1));
    }
    let tally = thread::scope(|s| {
        s.spawn(|| driver(addr, &token, &shared));
        for _ in 0..2 {
            s.spawn(|| exporter(addr, &token, &shared));
        }
        cavitation_inspect::fit(&mut net, &mut params, &train, fit_config(1), &mut inspector)
            .unwrap();
        let mut control = Counted {
            inspector: &mut inspector,
            shared: &shared,
            phase: Phase::WarmUp,
            left: MEASURED_EPOCHS,
            captured: false,
            asked: 0,
            tally: Tally::default(),
        };
        let config = fit_config(MAX_WARMUP_EPOCHS + MEASURED_EPOCHS);
        let result = fit_controlled(
            &mut net,
            &mut params,
            &train,
            config,
            &mut Hooks::new(),
            &mut control,
        );
        set_counting(false);
        shared.stop.store(true, Ordering::SeqCst);
        result.unwrap();
        control.tally
    });
    assert!(
        !shared.failed.load(Ordering::SeqCst),
        "a command was not answered 204"
    );
    assert!(
        tally.refused_warm_up > 0,
        "no publish was refused in {MAX_WARMUP_EPOCHS} warm-up epochs"
    );
    assert_eq!(inspector.clients(), 1, "the reading client was dropped");
    assert!(tally.sampled > 0, "no counted step captured a snapshot");
    assert_eq!(net.neuron_mask(0).map(|m| m.off().count()), Some(0));
    println!(
        "inspector_training_thread: {} warm-up epochs, then {} steps counted ({} sampled, {} \
         not), {} commands, {} exports fetched, refused publishes: {} in the warm-up, {} counted",
        tally.warm_up_epochs,
        tally.steps,
        tally.sampled,
        tally.steps - tally.sampled,
        tally.commands,
        shared.exports.load(Ordering::SeqCst),
        tally.refused_warm_up,
        tally.refused_counted
    );
    drop(inspector);
    let _ = stream.shutdown(Shutdown::Both);
    reading.join().unwrap();
    COUNTED.load(Ordering::Relaxed)
}

/// Runs the scenario and fails if the training thread allocated in a counted step.
fn main() -> ExitCode {
    let made = inspector_training_thread();
    if made == 0 {
        println!("inspector_training_thread: ok, 0 allocations");
        ExitCode::SUCCESS
    } else {
        println!("inspector_training_thread: FAILED, {made} allocations after warm-up");
        ExitCode::FAILURE
    }
}
