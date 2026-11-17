//! The step encoder on this machine's GPU: uploads and readbacks are exact, commands run in
//! the order recorded, a split dispatch of 20 million threads runs every one exactly once,
//! warm steps create no buffers, bind groups or pipelines, and profiled spans have GPU
//! times. Each test skips without a GPU, unless `CAVITATION_GPU_REQUIRED=1` makes that a
//! failure.

use std::time::Instant;

use cavitation_gpu::{
    Backend, BufferPool, Gpu, MAX_WORKGROUPS_PER_DIMENSION, PipelineCache, PooledBuffer, Shader,
    StepEncoder, StepSizes, TimedKind, Timing, wgpu,
};

/// Adds 1 to `count[i]` for each thread's linear index `i`, with workgroups numbered over
/// two dimensions, as a split dispatch needs.
const COUNT: Shader = Shader {
    name: "count",
    source: "
        @group(0) @binding(0) var<storage, read_write> count: array<atomic<u32>>;
        @compute @workgroup_size(256)
        fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>,
                @builtin(local_invocation_index) l: u32) {
            let i = (w.x + w.y * n.x) * 256u + l;
            if (i < arrayLength(&count)) { atomicAdd(&count[i], 1u); }
        }
    ",
};

/// `y = x + 1`, element by element, indexed by `global_invocation_id.x` alone.
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

/// Usage for every tensor buffer in these tests.
const TENSOR: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

/// The GPU, or `None` (after saying so) to skip the test.
fn gpu() -> Option<Gpu> {
    let gpu = Gpu::new();
    if gpu.is_none() {
        println!("no GPU: skipped");
    }
    gpu
}

/// Little-endian bytes of `values`.
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The `f32`s in `bytes`.
fn floats(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

/// A bind group for `pipeline` with `buffers` at bindings 0, 1, ….
fn bind(
    steps: &mut StepEncoder,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&PooledBuffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.binding(),
        })
        .collect();
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// A million floats go up through the staging buffer and come back bit for bit.
#[test]
fn million_float_round_trip() {
    let Some(gpu) = gpu() else { return };
    let n = 1 << 20;
    let mut steps = StepEncoder::new(
        &gpu,
        StepSizes {
            upload_bytes: 4 * n,
            readback_bytes: 4 * n,
            ..StepSizes::default()
        },
    );
    let mut pool = BufferPool::new(&gpu);
    let buffer = pool.take(4 * n, TENSOR);
    // Scattered bit patterns, negatives and subnormals included; bit 23 is cleared, so
    // none is a NaN or an infinity.
    let values: Vec<f32> = (0..n as u32)
        .map(|i| f32::from_bits(i.wrapping_mul(2_654_435_761) & 0xff7f_ffff))
        .collect();
    let data = bytes(&values);
    let mut step = steps.begin();
    step.upload(buffer.buffer(), 0, &data);
    let at = step.read_back(buffer.buffer(), 0, 4 * n);
    let output = step.finish();
    assert_eq!(at, 0);
    assert_eq!(output.bytes(), &data[..]);
    pool.give(buffer);
}

/// Commands run in the order recorded: an upload recorded after a dispatch reaches the
/// buffer after the dispatch has read it, and a clear lands between the two readbacks
/// around it.
#[test]
fn commands_run_in_order() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(ADD_ONE, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let x = pool.take(1024, TENSOR);
    let y = pool.take(1024, TENSOR);
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    let group = bind(&mut steps, &pipeline, &[&x, &y]);
    let mut step = steps.begin();
    step.upload(x.buffer(), 0, &bytes(&[1.0; 256]));
    step.dispatch(&pipeline, &group, 1);
    step.upload(x.buffer(), 0, &bytes(&[5.0; 256]));
    let at_y = step.read_back(y.buffer(), 0, 1024);
    step.dispatch(&pipeline, &group, 1);
    let at_y_again = step.read_back(y.buffer(), 0, 1024);
    step.clear(y.buffer(), 0, 1024);
    let at_y_cleared = step.read_back(y.buffer(), 0, 1024);
    let at_x = step.read_back(x.buffer(), 0, 1024);
    let output = step.finish();
    let part = |at: usize| floats(&output.bytes()[at..at + 1024]);
    assert_eq!(part(at_y), vec![2.0; 256]);
    assert_eq!(part(at_y_again), vec![6.0; 256]);
    assert_eq!(part(at_y_cleared), vec![0.0; 256]);
    assert_eq!(part(at_x), vec![5.0; 256]);
    pool.give(x);
    pool.give(y);
}

/// 20 million threads need 78,125 workgroups of 256, more than one dimension holds: the
/// split dispatch runs every thread exactly once, which atomic counts show.
#[test]
fn twenty_million_threads_split_over_two_dimensions() {
    let Some(gpu) = gpu() else { return };
    let n: u32 = 20_000_000;
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(COUNT, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let count = pool.take(4 * u64::from(n), TENSOR);
    let mut steps = StepEncoder::new(
        &gpu,
        StepSizes {
            readback_bytes: 4 * u64::from(n),
            ..StepSizes::default()
        },
    );
    let group = bind(&mut steps, &pipeline, &[&count]);
    let mut step = steps.begin();
    step.clear(count.buffer(), 0, 4 * u64::from(n));
    step.dispatch_split(&pipeline, &group, n.div_ceil(256));
    step.read_back(count.buffer(), 0, 4 * u64::from(n));
    let output = step.finish();
    let (words, _) = output.bytes().as_chunks::<4>();
    assert_eq!(words.len(), n as usize);
    let wrong: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| u32::from_le_bytes(**w) != 1)
        .map(|(i, _)| i)
        .take(10)
        .collect();
    assert!(
        wrong.is_empty(),
        "threads not run exactly once, first: {wrong:?}"
    );
    pool.give(count);
}

/// An unsplit dispatch above one dimension's limit is refused rather than quietly
/// repeating the first row's work in a kernel indexed by `global_invocation_id.x`.
#[test]
fn unsplit_dispatch_above_the_limit_panics() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(ADD_ONE, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let x = pool.take(1024, TENSOR);
    let y = pool.take(1024, TENSOR);
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    let group = bind(&mut steps, &pipeline, &[&x, &y]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut step = steps.begin();
        step.dispatch(&pipeline, &group, MAX_WORKGROUPS_PER_DIMENSION + 1);
    }));
    assert!(result.is_err());
}

/// After one step, a hundred more create no buffers (in the pool or the encoder), bind
/// groups or pipelines, and every step's result is exact.
#[test]
fn warm_steps_create_nothing() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let mut pool = BufferPool::new(&gpu);
    let x = pool.take(4096, TENSOR);
    let y = pool.take(4096, TENSOR);
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    steps.set_profiling(true);
    let mut group = None;
    let mut counts = None;
    for step_number in 0..101u32 {
        let pipeline = cache.get(ADD_ONE, "main", &[]).expect("build");
        let group = group.get_or_insert_with(|| bind(&mut steps, &pipeline, &[&x, &y]));
        let input = vec![step_number as f32; 1024];
        let mut step = steps.begin();
        step.upload(x.buffer(), 0, &bytes(&input));
        step.span("add one");
        step.dispatch(&pipeline, group, 4);
        step.read_back(y.buffer(), 0, 4096);
        let output = step.finish();
        assert_eq!(floats(output.bytes()), vec![step_number as f32 + 1.0; 1024]);
        let now = (
            pool.created(),
            steps.buffers_made(),
            steps.bind_groups_made(),
            cache.pipelines_built(),
        );
        assert_eq!(*counts.get_or_insert(now), now, "step {step_number}");
    }
    let (pooled, encoder_buffers, bind_groups, pipelines) = counts.expect("steps ran");
    assert_eq!((pooled, bind_groups, pipelines), (2, 1, 1));
    assert_eq!(encoder_buffers, if gpu.info().timestamps { 3 } else { 2 });
    assert_eq!(steps.steps_submitted(), 101);
    pool.give(x);
    pool.give(y);
}

/// A profiled step times each span on the GPU: every span that dispatches gets a positive
/// time, a span with sixteen large dispatches takes longer than one with a small one, a
/// span whose dispatches straddle an upload keeps the time of both halves (off Metal), and the spans
/// together take no longer than the whole step did on the clock.
#[test]
fn profiled_spans_have_gpu_times() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(COUNT, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let count = pool.take(4 << 22, TENSOR);
    let small = pool.take(256, TENSOR);
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    steps.set_profiling(true);
    let group = bind(&mut steps, &pipeline, &[&count]);
    let large = (1u32 << 22) / 256;
    // Warm up the clocks, then time.
    for _ in 0..5 {
        let mut step = steps.begin();
        step.span("warm");
        step.dispatch_split(&pipeline, &group, large);
        let _ = step.finish();
    }
    let start = Instant::now();
    let mut step = steps.begin();
    step.span("small");
    step.dispatch(&pipeline, &group, 16);
    step.span("large");
    for _ in 0..16 {
        step.dispatch_split(&pipeline, &group, large);
    }
    step.span("straddles an upload");
    step.dispatch_split(&pipeline, &group, large);
    step.upload(small.buffer(), 0, &[0; 256]);
    step.dispatch_split(&pipeline, &group, large);
    step.span("empty");
    let output = step.finish();
    let wall = start.elapsed().as_secs_f64() * 1e9;
    let spans = output.spans();
    let names: Vec<&str> = spans.iter().map(|s| s.0).collect();
    assert_eq!(names, ["small", "large", "straddles an upload", "empty"]);
    if !gpu.info().timestamps {
        assert!(spans.iter().all(|s| s.1.is_none()));
        println!("no timestamps: span times not checked");
        return;
    }
    let times: Vec<f64> = spans
        .iter()
        .map(|s| s.1.expect("profiled with timestamps"))
        .collect();
    println!("span times (ns): {times:?}; whole step on the clock {wall:.0} ns");
    assert!(times[..3].iter().all(|t| *t > 0.0), "{times:?}");
    assert!(times[3] >= 0.0, "{times:?}");
    assert!(times[1] > times[0], "{times:?}");
    // Two large dispatches against sixteen: the straddling span's time covers both halves,
    // so it is well above one sixteenth of the large span's. On Metal the passes on either
    // side of a copy are not timed on one timeline and one half can read as about 0, so
    // there the span need only have a time.
    if gpu.info().backend != Backend::Metal {
        assert!(times[2] > times[1] / 16.0, "{times:?}");
    }
    assert!(
        times.iter().sum::<f64>() <= wall,
        "{times:?} against {wall}"
    );
    pool.give(count);
    pool.give(small);
}

/// Without profiling, spans are labels only and have no times.
#[test]
fn unprofiled_spans_have_no_times() {
    let Some(gpu) = gpu() else { return };
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    let mut step = steps.begin();
    step.span("a");
    step.span("b");
    let output = step.finish();
    assert_eq!(output.spans(), &[("a", None), ("b", None)]);
}

/// A profiled step's timed entries, in the order recorded: a span begins a pass, an upload
/// ends it, a dispatch begins another and a readback ends that. With encoder timestamps
/// the two runs of copies are entries too. Every entry has a time, the span's time is the
/// sum of its entries', and what is not timed is said by name.
#[test]
fn profiled_steps_time_passes_and_copies() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(ADD_ONE, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let (x, y) = (pool.take(4096, TENSOR), pool.take(4096, TENSOR));
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    steps.set_profiling(true);
    let group = bind(&mut steps, &pipeline, &[&x, &y]);
    let timing = steps.timing();
    let info = gpu.info();
    let want: &[TimedKind] = match timing {
        Timing::Untimed { why } => {
            assert!(!info.timestamps && why.contains("TIMESTAMP_QUERY"), "{why}");
            &[]
        }
        Timing::Passes { copies_untimed } => {
            assert!(info.timestamps && !info.encoder_timestamps);
            assert!(copies_untimed.contains("TIMESTAMP_QUERY_INSIDE_ENCODERS"));
            &[TimedKind::Pass, TimedKind::Pass]
        }
        Timing::PassesAndCopies => {
            assert!(info.timestamps && info.encoder_timestamps);
            &[
                TimedKind::Pass,
                TimedKind::Copies,
                TimedKind::Pass,
                TimedKind::Copies,
            ]
        }
    };
    let mut step = steps.begin();
    step.span("add one");
    step.upload(x.buffer(), 0, &bytes(&[2.0; 1024]));
    step.dispatch(&pipeline, &group, 4);
    step.read_back(y.buffer(), 0, 4096);
    let output = step.finish();
    assert_eq!(floats(output.bytes()), vec![3.0; 1024]);
    let timed = output.timed();
    let kinds: Vec<TimedKind> = timed.iter().map(|t| t.kind).collect();
    assert_eq!(kinds, want, "{timing:?}");
    assert!(timed.iter().all(|t| t.span == 0));
    let times: Vec<f64> = timed
        .iter()
        .map(|t| {
            t.nanoseconds
                .expect("each entry's end follows its beginning")
        })
        .collect();
    let span = output.spans()[0].1;
    if want.is_empty() {
        assert_eq!(span, None);
    } else {
        assert_eq!(span, Some(times.iter().sum::<f64>()));
    }
    println!(
        "{timing:?}: {} of {} entries timed, {kinds:?}, {times:?} ns",
        times.len(),
        want.len()
    );
    pool.give(x);
    pool.give(y);
}

/// A copy between two GPU buffers is timed as an upload is: with encoder timestamps, a
/// profiled step of a copy, a dispatch and a readback gives a run of copies, a pass and
/// another run of copies, as in [`profiled_steps_time_passes_and_copies`].
#[test]
fn profiled_copies_between_buffers_are_timed() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(ADD_ONE, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let (w, x, y) = (
        pool.take(4096, TENSOR),
        pool.take(4096, TENSOR),
        pool.take(4096, TENSOR),
    );
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    let group = bind(&mut steps, &pipeline, &[&x, &y]);
    let mut step = steps.begin();
    step.upload(w.buffer(), 0, &bytes(&[2.0; 1024]));
    let _ = step.finish();
    steps.set_profiling(true);
    let timing = steps.timing();
    let mut step = steps.begin();
    step.span("add one");
    step.copy(w.buffer(), 0, x.buffer(), 0, 4096);
    step.dispatch(&pipeline, &group, 4);
    step.read_back(y.buffer(), 0, 4096);
    let output = step.finish();
    assert_eq!(floats(output.bytes()), vec![3.0; 1024]);
    let kinds: Vec<TimedKind> = output.timed().iter().map(|t| t.kind).collect();
    match timing {
        Timing::PassesAndCopies => assert_eq!(
            kinds,
            [
                TimedKind::Pass,
                TimedKind::Copies,
                TimedKind::Pass,
                TimedKind::Copies
            ]
        ),
        Timing::Passes { .. } => assert_eq!(kinds, [TimedKind::Pass, TimedKind::Pass]),
        Timing::Untimed { .. } => assert!(kinds.is_empty()),
    }
    println!("{timing:?}: {kinds:?}");
    for b in [w, x, y] {
        pool.give(b);
    }
}

/// Each step's times are its own, not the step before's: steps alternate between a small
/// and a large dispatch, 1,024 times the work, and the large steps' median time is above the
/// small steps'. Read from a resolve in the step's own submission, Metal gave each step the
/// previous step's times, which turns the medians around (small 348 µs, large 3.75 µs).
/// Medians, and no wider margin, because the other tests in this file share the GPU and can
/// stretch the small steps well past a few microseconds.
#[test]
fn span_times_belong_to_their_step() {
    let Some(gpu) = gpu() else { return };
    if matches!(
        StepEncoder::new(&gpu, StepSizes::default()).timing(),
        Timing::Untimed { .. }
    ) {
        println!("no timestamps: not checked");
        return;
    }
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache.get(COUNT, "main", &[]).expect("build");
    let mut pool = BufferPool::new(&gpu);
    let count = pool.take(4 << 22, TENSOR);
    let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    steps.set_profiling(true);
    let group = bind(&mut steps, &pipeline, &[&count]);
    let mut time = |workgroups: u32| {
        let mut step = steps.begin();
        step.span("count");
        step.dispatch_split(&pipeline, &group, workgroups);
        step.finish().spans()[0].1.expect("timed")
    };
    let (mut small, mut large) = (Vec::new(), Vec::new());
    for _ in 0..9 {
        small.push(time(16));
        large.push(time(1 << 14));
    }
    println!("small {small:?} ns; large {large:?} ns");
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (small, large) = (median(&mut small), median(&mut large));
    assert!(large > small, "median small {small} ns, large {large} ns");
    pool.give(count);
}
