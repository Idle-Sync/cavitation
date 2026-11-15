//! Counts host allocations and live host bytes across warm GPU steps. wgpu allocates on the
//! host for each encoder, pass, submission and poll, so a step is not free of allocations;
//! but after warm-up, the steps of the same work must repeat the same allocation counts with
//! a short period, and the live bytes must come back to the same level every period, or
//! memory would grow.
//!
//! This target has no test harness: `main` runs on the main thread alone, so nothing else
//! allocates while a step is counted. It skips without a GPU, unless
//! `CAVITATION_GPU_REQUIRED=1` makes that a failure.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use cavitation_gpu::{BufferPool, Gpu, PipelineCache, Shader, StepEncoder, StepSizes, wgpu};

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
    /// Counts, then allocates with `System`.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: the caller's guarantees for `layout` are passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    /// Frees with `System`.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `ptr` was allocated by `System` with `layout`, as the caller guarantees.
        unsafe { System.dealloc(ptr, layout) }
    }

    /// Counts, then reallocates with `System`.
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

/// `y = x + 1`, element by element.
const ADD_ONE: Shader = Shader {
    name: "add one",
    source: "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x < arrayLength(&y)) { y[id.x] = x[id.x] + 1.0; }
        }
    ",
};

/// The steps run before counting.
const WARM_UP: usize = 20;

/// The steps counted.
const COUNTED: usize = 60;

/// The longest repeating pattern accepted. wgpu's bookkeeping can alternate between steps
/// (a profiled step alternates between two allocation counts), which is still bounded; a
/// count or a level that drifts or grows has no period.
const LONGEST_PERIOD: usize = 4;

/// Per counted step: the allocations it made, and the live bytes when it began.
struct Counts {
    /// The allocations each counted step made.
    allocations: Vec<usize>,
    /// The live host bytes at the beginning of each counted step, and after the last.
    live: Vec<isize>,
}

/// Runs warm steps of an upload, a span with a dispatch, an upload and another dispatch,
/// a second span and a readback, profiled or not, and counts them.
fn count_steps(gpu: &Gpu, profiling: bool) -> Counts {
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let mut cache = PipelineCache::new(gpu);
    let pipeline = cache.get(ADD_ONE, "main", &[]).expect("build");
    let mut pool = BufferPool::new(gpu);
    let x = pool.take(4096, usage);
    let y = pool.take(4096, usage);
    let mut steps = StepEncoder::new(gpu, StepSizes::default());
    steps.set_profiling(profiling);
    let group = steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.binding(),
            },
        ],
    });
    let input = vec![0u8; 4096];
    let mut counts = Counts {
        allocations: Vec::with_capacity(COUNTED),
        live: Vec::with_capacity(COUNTED + 1),
    };
    let mut checksum = 0u64;
    for step_number in 0..WARM_UP + COUNTED {
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        if step_number >= WARM_UP {
            counts.live.push(LIVE.load(Ordering::Relaxed));
        }
        let mut step = steps.begin();
        step.upload(x.buffer(), 0, &input);
        step.span("first");
        step.dispatch(&pipeline, &group, 4);
        step.upload(x.buffer(), 0, &input);
        step.dispatch(&pipeline, &group, 4);
        step.span("second");
        step.dispatch(&pipeline, &group, 4);
        step.read_back(y.buffer(), 0, 16);
        let output = step.finish();
        checksum += u64::from(output.bytes()[3]);
        let after = ALLOCATIONS.load(Ordering::Relaxed);
        if step_number >= WARM_UP {
            counts.allocations.push(after - before);
        }
    }
    counts.live.push(LIVE.load(Ordering::Relaxed));
    assert!(checksum > 0, "the steps produced their results");
    assert_eq!(pool.created(), 2);
    assert_eq!(steps.bind_groups_made(), 1);
    assert_eq!(cache.pipelines_built(), 1);
    pool.give(x);
    pool.give(y);
    counts
}

/// The shortest period, up to [`LONGEST_PERIOD`], with which `values` repeats exactly.
fn period<T: PartialEq>(values: &[T]) -> Option<usize> {
    (1..=LONGEST_PERIOD).find(|&p| (p..values.len()).all(|i| values[i] == values[i - p]))
}

/// The counted steps whose largest allocation count bounds every later step's.
const BOUNDING_STEPS: usize = 10;

/// Runs the counts unprofiled and profiled, and fails if a run's live bytes do not repeat
/// with a period of at most [`LONGEST_PERIOD`] steps (memory would grow), or if a later
/// step makes more allocations than any of the first [`BOUNDING_STEPS`] counted steps. A
/// step making one allocation fewer now and then (wgpu reusing a freed block) passes.
fn main() -> ExitCode {
    let Some(gpu) = Gpu::new() else {
        println!("no GPU: skipped");
        return ExitCode::SUCCESS;
    };
    let mut ok = true;
    for profiling in [false, true] {
        let counts = count_steps(&gpu, profiling);
        let label = if profiling { "profiled" } else { "unprofiled" };
        let bound = *counts.allocations[..BOUNDING_STEPS]
            .iter()
            .max()
            .expect("steps were counted");
        let least = *counts.allocations.iter().min().expect("steps were counted");
        let most = *counts.allocations.iter().max().expect("steps were counted");
        match period(&counts.live) {
            Some(l) if most <= bound => println!(
                "{label}: {least}..={most} host allocations per warm step; live host bytes \
                 repeat every {l} step(s), ending where they began ({} bytes over {COUNTED} steps)",
                counts.live[counts.live.len() - 1] - counts.live[0],
            ),
            _ => {
                println!(
                    "{label}: memory grows or steps allocate more over time; allocations per \
                     step {:?}; live bytes {:?}",
                    counts.allocations, counts.live
                );
                ok = false;
            }
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
