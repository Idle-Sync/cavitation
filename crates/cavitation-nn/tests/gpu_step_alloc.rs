//! Counts host allocations and live host bytes across warm `GpuTrainer` epochs, on full,
//! short, skipped and profiled steps. wgpu allocates on the host for each encoder, pass,
//! submission and poll, so a step is not free of allocations; but after warm-up, the live
//! bytes must be the same at the start of every epoch of the same batches, or memory would
//! grow, and each step must make the same number of allocations in every epoch, give or take
//! one (measured on Metal over 40 epochs: never more than one apart). That no GPU objects
//! are made after warm-up is checked inside the crate, in `gpu::memory_tests`.
//!
//! The counters see every thread, the trainer's own included. This target installs its own
//! global allocator, so it runs without the test harness, and nothing but the trainer runs
//! while a step is counted. It skips without a GPU, unless `CAVITATION_GPU_REQUIRED=1` makes
//! that a failure.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Rng, TensorView};
use cavitation_gpu::Gpu;
use cavitation_nn::{
    BatchNorm2d, BatchView, Conv2d, Dropout, Flatten, GpuTrainer, Health, Linear, Loss, MaxPool2d,
    OnNonFinite, OptimizerConfig, Relu, Sequential, TargetsView, Warning,
};

/// The number of allocations (including reallocations) made so far, from every thread.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// The bytes allocated and not yet freed, from every thread.
static LIVE: AtomicIsize = AtomicIsize::new(0);

/// The system allocator, counting every allocation and reallocation in [`ALLOCATIONS`] and
/// keeping [`LIVE`] up to date.
struct CountingAllocator;

// SAFETY: every method forwards to `System`, which upholds `GlobalAlloc`'s contract; the
// counters are only atomic updates.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: the caller's guarantees for `layout` are passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: the caller's guarantees for `layout` are passed on unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `ptr` was allocated by `System` with `layout`, as the caller guarantees.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        // SAFETY: the caller's guarantees for `ptr`, `layout` and `new_size` are passed on.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Installs the counting allocator for this test binary.
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The epochs run before counting.
const WARM_UP: usize = 2;

/// The epochs counted.
const COUNTED: usize = 6;

/// The batch with a NaN input, left out under [`OnNonFinite::Skip`] every epoch.
const BAD: usize = 2;

/// The batches profiled in every epoch: a full one and the short one.
const PROFILED: [usize; 2] = [1, 3];

/// One batch: its inputs and class labels.
struct Data {
    /// The inputs, row after row.
    x: Vec<f32>,
    /// The class labels, one per row.
    labels: Vec<usize>,
}

impl Data {
    /// `rows` rows of `width` inputs in `[-1, 1)` and labels below `classes`, from `seed`.
    fn new(seed: u64, rows: usize, width: usize, classes: usize) -> Data {
        let mut rng = Rng::seed(seed);
        Data {
            x: (0..rows * width)
                .map(|_| rng.range(-1.0, 1.0) as f32)
                .collect(),
            labels: (0..rows)
                .map(|_| (rng.range(0.0, classes as f64) as usize).min(classes - 1))
                .collect(),
        }
    }

    /// The batch as a trainer takes it.
    fn view(&self) -> BatchView<'_, f32> {
        let rows = self.labels.len();
        BatchView {
            x: TensorView::from_slice(&self.x, &[rows, self.x.len() / rows]).unwrap(),
            targets: TargetsView::Classes(&self.labels),
        }
    }
}

/// A CNN on 2 × 8 × 8 images: a convolution, batch norm, ReLU, max pooling, dropout, flatten
/// and a `Linear`.
fn cnn(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(93);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::new(params, 4, (8, 8)))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 5, &mut rng))
}

/// Per counted epoch: the allocations each step made, and the live bytes when it began.
struct Counts {
    /// The allocations of each counted epoch's steps, `end_epoch` included as the last.
    allocations: Vec<Vec<usize>>,
    /// The live host bytes at the beginning of each counted epoch, and after the last.
    live: Vec<isize>,
}

/// Trains the CNN with `config` at batch 6 on 22 rows (6, 6, 6 with a NaN and a short 4),
/// every health check on and the NaN batch left out under `Skip`, profiling the batches in
/// [`PROFILED`], for [`WARM_UP`] epochs, then counts [`COUNTED`] epochs.
fn count_epochs(gpu: &Gpu, config: &OptimizerConfig) -> Counts {
    let mut params = ParamStore::new();
    let net = cnn(&mut params);
    let mut batches: Vec<Data> = [6, 6, 6, 4]
        .iter()
        .enumerate()
        .map(|(s, &rows)| Data::new(400 + s as u64, rows, 2 * 64, 5))
        .collect();
    batches[BAD].x[5] = f32::NAN;
    let warnings = AtomicUsize::new(0);
    let mut t = GpuTrainer::new(&net, &params, config, Loss::CrossEntropy, 6, gpu).unwrap();
    t.set_health(Health {
        on_non_finite: OnNonFinite::Skip,
        sink: Some(Box::new(|_: &Warning| {
            warnings.fetch_add(1, Ordering::Relaxed);
        })),
        ..Health::quiet()
    })
    .unwrap();
    // Made in full before the first step, so that filling them in allocates nothing.
    let mut counts = Counts {
        allocations: vec![vec![0; batches.len() + 1]; WARM_UP + COUNTED],
        live: vec![0; WARM_UP + COUNTED + 1],
    };
    let mut sum = 0.0f32;
    for epoch in 0..WARM_UP + COUNTED {
        counts.live[epoch] = LIVE.load(Ordering::Relaxed);
        let warned = warnings.load(Ordering::Relaxed);
        for (i, (d, made)) in batches
            .iter()
            .zip(&mut counts.allocations[epoch])
            .enumerate()
        {
            let before = ALLOCATIONS.load(Ordering::Relaxed);
            if PROFILED.contains(&i) {
                t.profile_next_step();
            }
            let loss = t.step(&net, d.view()).unwrap();
            *made = ALLOCATIONS.load(Ordering::Relaxed) - before;
            assert_eq!(
                PROFILED.contains(&i),
                t.last_profile().is_some(),
                "step {i}"
            );
            if i != BAD {
                sum += loss;
            }
        }
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        t.end_epoch(epoch + 1).unwrap();
        counts.allocations[epoch][batches.len()] = ALLOCATIONS.load(Ordering::Relaxed) - before;
        assert!(
            warnings.load(Ordering::Relaxed) > warned,
            "epoch {epoch} sent no warning"
        );
    }
    counts.live[WARM_UP + COUNTED] = LIVE.load(Ordering::Relaxed);
    counts.allocations.drain(..WARM_UP);
    counts.live.drain(..WARM_UP);
    assert!(sum.is_finite() && sum > 0.0, "the steps trained");
    assert_eq!(t.skipped_steps(), (WARM_UP + COUNTED) as u64);
    counts
}

/// Counts SGD with momentum and Adam, and fails if the live bytes at the start of a counted
/// epoch, or after the last, differ from the first counted epoch's, or if a step's
/// allocations differ by more than one between counted epochs.
fn main() -> ExitCode {
    let Some(gpu) = Gpu::new() else {
        if std::env::var_os("CAVITATION_GPU_REQUIRED").is_some() {
            println!("CAVITATION_GPU_REQUIRED is set, but no GPU opened");
            return ExitCode::FAILURE;
        }
        println!("no GPU: skipped");
        return ExitCode::SUCCESS;
    };
    println!("on {}", gpu.info().name);
    let configs = [
        (
            "sgd",
            OptimizerConfig::Sgd {
                lr: 0.05,
                momentum: 0.9,
                weight_decay: 1e-4,
            },
        ),
        (
            "adam",
            OptimizerConfig::Adam {
                lr: 1e-3,
                weight_decay: 0.0,
            },
        ),
    ];
    let mut ok = true;
    for (name, config) in &configs {
        let counts = count_epochs(&gpu, config);
        let steps = counts.allocations[0].len();
        let spread = |i: usize| {
            let at = counts.allocations.iter().map(|e| e[i]);
            (at.clone().min().unwrap(), at.max().unwrap())
        };
        let spreads: Vec<(usize, usize)> = (0..steps).map(spread).collect();
        let steady = spreads.iter().all(|&(least, most)| most - least <= 1);
        let flat = counts.live.iter().all(|&l| l == counts.live[0]);
        if steady && flat {
            println!(
                "{name}: host allocations per step (the last is end_epoch), least and most \
                 over {COUNTED} epochs: {spreads:?}; live host bytes the same at every \
                 epoch's start"
            );
        } else {
            println!(
                "{name}: memory grows or steps allocate differently across epochs; \
                 allocations per step {:?}; live bytes at each epoch's start {:?}",
                counts.allocations, counts.live
            );
            ok = false;
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
