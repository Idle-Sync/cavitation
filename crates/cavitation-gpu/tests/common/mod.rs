//! The twin harness shared by the GPU kernel tests: input generators for the exact and the
//! rounding twins, the two assertions they end in, the error bounds, a kit that moves
//! `f32` data to and from the GPU, and a loop over every backend this machine has.
//!
//! Every GPU kernel is checked twice against its CPU twin:
//!
//! 1. **Exact.** Inputs are non-zero small integers from {-2, -1, 1, 2}
//!    ([`exact_values`]), chosen so every sum and product the kernel forms is an integer
//!    well inside f32's 24-bit range: exact in any order and with or without fused
//!    multiply-adds, so the GPU must match the CPU bit for bit at any size, and a dropped or
//!    doubled term always changes a result. [`assert_exact`] compares bits.
//! 2. **Rounding.** Seeded inputs in a range ([`seeded_values`]), against an `f64` reference
//!    computed from the same f32 inputs, within a stated bound per element
//!    ([`assert_within`]). Bounds are in units of [`U`] = 2^-24, f32's unit roundoff, plus
//!    the absolute [`FLOOR`] = 2^-126 every bound carries.
//!
//! "Exact" covers normal numbers only: GPUs flush subnormal results to zero (measured on
//! Vulkan and DX12), so [`assert_exact`] flushes the reference's subnormals before
//! comparing, and [`FLOOR`] covers a flushed result in the rounding twin.
//!
//! The suite requires **round to nearest, ties to even**: the rounding bounds take one
//! rounding to be within [`U`] of its exact result, and the twins that compare a rounded
//! result with the CPU's bits need the CPU's ties. WGSL leaves the rounding mode open (a
//! correctly rounded result may be either float beside the exact one, up to `2u` off), so
//! this is a requirement on the backend, not a WGSL guarantee. `tests/rounding.rs` checks
//! it on its own, so a backend that rounds another way fails there by name.

#![allow(dead_code)]

use cavitation_gpu::{
    Backend, BufferPool, Gpu, PipelineCache, PooledBuffer, StepEncoder, StepSizes, wgpu,
};

/// f32's unit roundoff, 2^-24: the largest relative error of one operation rounded to
/// nearest, which the suite requires of the backend (`tests/rounding.rs`).
pub const U: f64 = 1.0 / 16_777_216.0;

/// The absolute floor every rounding bound carries, 2^-126, the smallest normal f32: a
/// result the GPU flushes to zero is at most this far from its reference.
pub const FLOOR: f64 = 1.0 / (1u128 << 126) as f64;

/// The usage every buffer from [`Kit`] has: a storage buffer that can be written and read
/// back.
pub const TENSOR: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

/// A small seeded generator (xorshift64*), the same on every machine, so a failing input
/// can be reproduced from its seed.
#[derive(Clone, Debug)]
pub struct Seeded(u64);

impl Seeded {
    /// A generator from `seed`; every seed, 0 included, gives a usable sequence.
    pub fn new(seed: u64) -> Seeded {
        Seeded(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A uniform value in `[lo, hi)`, rounded to f32.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// `n` values for an exact twin: each one of -2, -1, 1 and 2, drawn from `seed`.
pub fn exact_values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Seeded::new(seed);
    (0..n)
        .map(|_| [-2.0, -1.0, 1.0, 2.0][(rng.next_u64() >> 62) as usize])
        .collect()
}

/// `n` values for a rounding twin: uniform in `[lo, hi)`, drawn from `seed`.
pub fn seeded_values(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut rng = Seeded::new(seed);
    (0..n).map(|_| rng.uniform(lo, hi)).collect()
}

/// `x` with a subnormal flushed to zero (keeping its sign), as a GPU flushes it.
pub fn flush(x: f32) -> f32 {
    if x != 0.0 && x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// The error bound of a sum of `k` terms (a dot product of length `k`, or a reduction),
/// in any order, against an f64 reference: `2(k + 2) · u · Σ|terms|`. This is the GEMM
/// bound `2(k+2)·u·Σ|a·b|`, and the column-sum bound with the column's length for `k`.
pub fn sum_bound(k: usize, abs_sum: f64) -> f64 {
    2.0 * (k as f64 + 2.0) * U * abs_sum
}

/// The GEMM bound for one output: `2(k + 2) · u · Σ|a·b|` over its `k` products.
pub fn gemm_bound(k: usize, abs_products: f64) -> f64 {
    sum_bound(k, abs_products)
}

/// Asserts that `got` equals `want` bit for bit, or `want` with its subnormals flushed to zero
/// (GPUs that flush and GPUs that keep subnormals both pass; a NaN matches any NaN); on failure, names `what`, the number of mismatches and the first few
/// with both values and their bits.
pub fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let mut mismatches = Vec::new();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let same = g.to_bits() == w.to_bits()
            || g.to_bits() == flush(w).to_bits()
            || (g.is_nan() && w.is_nan());
        if !same {
            mismatches.push(i);
        }
    }
    if !mismatches.is_empty() {
        let shown: Vec<String> = mismatches
            .iter()
            .take(5)
            .map(|&i| {
                format!(
                    "[{i}] got {:e} (0x{:08x}), want {:e} (0x{:08x})",
                    got[i],
                    got[i].to_bits(),
                    flush(want[i]),
                    flush(want[i]).to_bits()
                )
            })
            .collect();
        panic!(
            "{what}: {} of {} elements differ; first: {}",
            mismatches.len(),
            got.len(),
            shown.join("; ")
        );
    }
}

/// The worst element of a rounding twin, as [`assert_within`] reports it.
#[derive(Clone, Copy, Debug)]
pub struct Worst {
    /// Its index.
    pub index: usize,
    /// The GPU's value.
    pub got: f32,
    /// The f64 reference.
    pub reference: f64,
    /// `|got - reference|`.
    pub error: f64,
    /// Its bound, the floor included.
    pub bound: f64,
}

impl Worst {
    /// The error as a fraction of the bound: at most 1 when the twin passes.
    pub fn ratio(&self) -> f64 {
        self.error / self.bound
    }
}

/// Asserts that every `got[i]` is within `bound(i)` + [`FLOOR`] of `reference[i]`, and that
/// it is NaN exactly where the reference is; returns the worst element (the largest error
/// relative to its bound), so a test can print how close to its bound a kernel comes. On
/// failure, names `what` and the worst element.
pub fn assert_within(
    got: &[f32],
    reference: &[f64],
    bound: impl Fn(usize) -> f64,
    what: &str,
) -> Worst {
    assert_eq!(got.len(), reference.len(), "{what}: lengths differ");
    let mut worst = Worst {
        index: 0,
        got: 0.0,
        reference: 0.0,
        error: 0.0,
        bound: FLOOR,
    };
    for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
        assert_eq!(
            g.is_nan(),
            r.is_nan(),
            "{what}: element {i} is {g:e} where the reference is {r:e}"
        );
        if r.is_nan() {
            continue;
        }
        let error = if g as f64 == r {
            0.0
        } else {
            (g as f64 - r).abs()
        };
        let b = bound(i) + FLOOR;
        let candidate = Worst {
            index: i,
            got: g,
            reference: r,
            error,
            bound: b,
        };
        if candidate.ratio() > worst.ratio() {
            worst = candidate;
        }
    }
    assert!(
        worst.error <= worst.bound,
        "{what}: element {} is {:e}, the reference {:e}: error {:e} above its bound {:e} ({:.2}x)",
        worst.index,
        worst.got,
        worst.reference,
        worst.error,
        worst.bound,
        worst.ratio()
    );
    worst
}

/// Runs `body` on every backend this platform tries ([`Backend::on_this_platform`]) that
/// has a GPU, after `CAVITATION_GPU_BACKEND`'s restriction, and returns how many it ran
/// on. Without a GPU it says so and runs nothing; with `CAVITATION_GPU_REQUIRED=1`, a
/// missing GPU on an expected backend is a panic instead (from [`Gpu::with_backend`]).
pub fn for_each_gpu(mut body: impl FnMut(&Gpu)) -> usize {
    let mut ran = 0;
    for &backend in Backend::on_this_platform() {
        if let Some(gpu) = Gpu::with_backend(backend) {
            println!("on {backend}: {}", gpu.info().name);
            body(&gpu);
            ran += 1;
        }
    }
    if ran == 0 {
        println!("no GPU: skipped");
    }
    ran
}

/// What a kernel test needs on one GPU: a buffer pool, a pipeline cache and a step
/// encoder, with helpers that move `f32` data to and from the GPU, one step each.
#[derive(Debug)]
pub struct Kit {
    /// The GPU.
    pub gpu: Gpu,
    /// Buffers for the test's tensors.
    pub pool: BufferPool,
    /// Pipelines for the test's kernels.
    pub cache: PipelineCache,
    /// The step encoder the helpers and the test's dispatches use.
    pub steps: StepEncoder,
}

impl Kit {
    /// A kit on `gpu` whose steps can upload and read back up to `max_bytes` each.
    pub fn new(gpu: &Gpu, max_bytes: u64) -> Kit {
        Kit {
            gpu: gpu.clone(),
            pool: BufferPool::new(gpu),
            cache: PipelineCache::new(gpu),
            steps: StepEncoder::new(
                gpu,
                StepSizes {
                    upload_bytes: max_bytes,
                    readback_bytes: max_bytes,
                    ..StepSizes::default()
                },
            ),
        }
    }

    /// A GPU buffer holding `values`.
    pub fn upload(&mut self, values: &[f32]) -> PooledBuffer {
        let buffer = self.pool.take(4 * values.len() as u64, TENSOR);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut step = self.steps.begin();
        step.upload(buffer.buffer(), 0, &bytes);
        let _ = step.finish();
        buffer
    }

    /// A GPU buffer of `n` zeros.
    pub fn zeros(&mut self, n: usize) -> PooledBuffer {
        let buffer = self.pool.take(4 * n as u64, TENSOR);
        let mut step = self.steps.begin();
        step.clear(buffer.buffer(), 0, 4 * n as u64);
        let _ = step.finish();
        buffer
    }

    /// The `f32`s in `buffer` (its first [`PooledBuffer::len`] bytes).
    pub fn read(&mut self, buffer: &PooledBuffer) -> Vec<f32> {
        let mut step = self.steps.begin();
        step.read_back(buffer.buffer(), 0, buffer.len());
        let output = step.finish();
        let (words, _) = output.bytes().as_chunks::<4>();
        words.iter().map(|w| f32::from_le_bytes(*w)).collect()
    }

    /// A bind group for `pipeline`'s group 0 with `buffers` at bindings 0, 1, ….
    pub fn bind(
        &mut self,
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
        self.steps.bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }
}
