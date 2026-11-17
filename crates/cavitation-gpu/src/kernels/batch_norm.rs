//! 2-D batch normalisation on the GPU, as `cavitation_core`'s `Tape::batch_norm2d` computes
//! it on NCHW batches: the statistics from the batch (the mean first, then the centred
//! squares, in fixed trees) or from the running statistics, the normalisation with an
//! optional affine `γ`, `β`, the backward pass in either mode, and the running statistics'
//! update with a momentum or a cumulative average, skipped when the step's finite flag is
//! set.
//!
//! Every kernel works from a stats buffer of `3 · C` values, `[mean | inv | unbiased]`, as the
//! CPU's stats node holds them: the mean, `1 / √(var + eps)` and the unbiased variance (0 for
//! running statistics). The bounds assume every multiply-add may round twice.

use super::bindings::{Kind, bind, check_distinct, check_len, layout, upload_uniform, word_bytes};
use super::reduce::{ChannelSumsCall, SumsTo, TermSums, partials_pass};
use super::{ChannelDims, ChannelSums};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// The kernel's name in panic messages.
const NAME: &str = "batch norm";

/// Threads per workgroup in every kernel here, passed as the `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// Bytes of the `Params` uniform: the shape, the flags, `m`, `m − 1` and eps.
const PARAMS_BYTES: u64 = 32;

/// Bytes of the running update's `Running` uniform: the channels, the momentum and
/// `1 − momentum`.
const RUNNING_BYTES: u64 = 16;

/// Flag bit of `Params`: the layer is affine, so `γ` and `β` apply.
const AFFINE: u32 = 1;

/// Flag bit of `Params`: the statistics are the batch's, so the backward pass goes through
/// them.
const BATCH: u32 = 2;

/// The `WORKGROUP` override and the `Params` uniform every kernel here but the running update
/// shares.
macro_rules! params {
    () => {
        "
        override WORKGROUP: u32;

        struct Params {
            outer: u32,
            channels: u32,
            inner: u32,
            flags: u32,
            m: f32,
            m_minus_1: f32,
            eps: f32,
            pad: f32,
        }
        "
    };
}

/// The centred squares `(x − mean)²` summed per channel for the batch variance.
const CENTRED_SQUARES: Shader = Shader {
    name: "batch norm centred squares",
    source: partials_pass!(
        "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(3) var<storage, read> stats: array<f32>;
        ",
        "
        fn term(i: u32, c: u32) -> f32 {
            let d = x[i] - stats[c];
            return d * d;
        }
        "
    ),
};

/// `dy · x̂ = dy · (x − mean) · inv` summed per channel: `γ`'s gradient, and part of `x`'s.
const DY_XHAT: Shader = Shader {
    name: "batch norm dy xhat",
    source: partials_pass!(
        "
        @group(0) @binding(0) var<storage, read> dy: array<f32>;
        @group(0) @binding(3) var<storage, read> x: array<f32>;
        @group(0) @binding(4) var<storage, read> stats: array<f32>;
        ",
        "
        fn term(i: u32, c: u32) -> f32 {
            return dy[i] * ((x[i] - stats[c]) * stats[dims.channels + c]);
        }
        "
    ),
};

/// The per-channel statistics kernels, one thread per channel: `mean` divides the batch sum
/// by `m`; `finish` turns the sum of centred squares into `inv = 1/√(sq/m + eps)` and the
/// unbiased variance `sq/(m − 1)`; `running` takes the running mean and `1/√(var + eps)`.
const STATS: Shader = Shader {
    name: "batch norm stats",
    source: concat!(
        params!(),
        "
        @group(0) @binding(0) var<storage, read_write> stats: array<f32>;
        @group(0) @binding(1) var<uniform> p: Params;
        @group(0) @binding(2) var<storage, read> a: array<f32>;
        @group(0) @binding(3) var<storage, read> b: array<f32>;

        @compute @workgroup_size(WORKGROUP)
        fn mean(@builtin(global_invocation_id) id: vec3<u32>) {
            let c = id.x;
            if (c >= p.channels) { return; }
            stats[c] = a[c] / p.m;
        }

        @compute @workgroup_size(WORKGROUP)
        fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
            let c = id.x;
            if (c >= p.channels) { return; }
            let sq = a[c];
            stats[p.channels + c] = inverseSqrt(sq / p.m + p.eps);
            stats[2u * p.channels + c] = sq / p.m_minus_1;
        }

        @compute @workgroup_size(WORKGROUP)
        fn running(@builtin(global_invocation_id) id: vec3<u32>) {
            let c = id.x;
            if (c >= p.channels) { return; }
            stats[c] = a[c];
            stats[p.channels + c] = inverseSqrt(b[c] + p.eps);
            stats[2u * p.channels + c] = 0.0;
        }
        "
    ),
};

/// Normalisation, one thread per value: `y = (x − mean) · inv · γ + β`, or without `γ` and
/// `β` when the layer is not affine.
const NORMALISE: Shader = Shader {
    name: "batch norm normalise",
    source: concat!(
        params!(),
        "
        @group(0) @binding(0) var<storage, read_write> y: array<f32>;
        @group(0) @binding(1) var<uniform> p: Params;
        @group(0) @binding(2) var<storage, read> x: array<f32>;
        @group(0) @binding(3) var<storage, read> stats: array<f32>;
        @group(0) @binding(4) var<storage, read> weight: array<f32>;
        @group(0) @binding(5) var<storage, read> bias: array<f32>;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= p.outer * p.channels * p.inner) { return; }
            let c = (i / p.inner) % p.channels;
            let t = (x[i] - stats[c]) * stats[p.channels + c];
            if ((p.flags & 1u) != 0u) {
                y[i] = t * weight[c] + bias[c];
            } else {
                y[i] = t;
            }
        }
        "
    ),
};

/// The input gradient, one thread per value, from `Σ dy` and `Σ dy · x̂` per channel: with
/// `g = γ` (1 without), `dx = inv / m · (m · dy·g − Σdy·g − x̂ · Σ(dy·x̂)·g)` through batch
/// statistics, `dx = dy · g · inv` through running ones.
const BACKWARD: Shader = Shader {
    name: "batch norm backward",
    source: concat!(
        params!(),
        "
        @group(0) @binding(0) var<storage, read_write> grad_in: array<f32>;
        @group(0) @binding(1) var<uniform> p: Params;
        @group(0) @binding(2) var<storage, read> grad_out: array<f32>;
        @group(0) @binding(3) var<storage, read> x: array<f32>;
        @group(0) @binding(4) var<storage, read> stats: array<f32>;
        @group(0) @binding(5) var<storage, read> weight: array<f32>;
        @group(0) @binding(6) var<storage, read> sum_dy: array<f32>;
        @group(0) @binding(7) var<storage, read> sum_dy_xhat: array<f32>;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= p.outer * p.channels * p.inner) { return; }
            let c = (i / p.inner) % p.channels;
            let inv = stats[p.channels + c];
            var g = 1.0;
            if ((p.flags & 1u) != 0u) { g = weight[c]; }
            let gy = grad_out[i] * g;
            if ((p.flags & 2u) == 0u) {
                grad_in[i] = gy * inv;
                return;
            }
            let xhat = (x[i] - stats[c]) * inv;
            let sg = sum_dy[c] * g;
            let sgx = sum_dy_xhat[c] * g;
            grad_in[i] = inv / p.m * (p.m * gy - sg - xhat * sgx);
        }
        "
    ),
};

/// The running statistics' update, one thread per channel, unless the step's finite flag is
/// set: `r ← momentum · batch + (1 − momentum) · r` for the mean and the unbiased variance.
const RUNNING: Shader = Shader {
    name: "batch norm running update",
    source: "
        override WORKGROUP: u32;

        struct Running { channels: u32, momentum: f32, keep: f32, pad: f32 }

        @group(0) @binding(0) var<storage, read_write> running_mean: array<f32>;
        @group(0) @binding(1) var<uniform> r: Running;
        @group(0) @binding(2) var<storage, read> stats: array<f32>;
        @group(0) @binding(3) var<storage, read> finite: array<u32>;
        @group(0) @binding(4) var<storage, read_write> running_var: array<f32>;

        @compute @workgroup_size(WORKGROUP)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let c = id.x;
            if (finite[0] != 0u || c >= r.channels) { return; }
            running_mean[c] = r.momentum * stats[c] + r.keep * running_mean[c];
            running_var[c] = r.momentum * stats[2u * r.channels + c] + r.keep * running_var[c];
        }
    ",
};

/// The `Params` uniform's bytes for `dims`, `flags` and `eps`, with `m = outer · inner` and
/// `m − 1` and eps rounded to `f32` as the CPU rounds them.
fn params(dims: ChannelDims, flags: u32, eps: f64) -> Vec<u8> {
    let m = dims.summed() as f64;
    word_bytes([
        dims.outer,
        dims.channels,
        dims.inner,
        flags,
        (m as f32).to_bits(),
        ((m - 1.0) as f32).to_bits(),
        (eps as f32).to_bits(),
        0,
    ])
}

/// Where a forward pass takes its statistics from.
#[derive(Clone, Copy, Debug)]
pub enum BatchStats<'a> {
    /// The batch's own: its mean and biased variance, which training uses (and evaluation
    /// too when the layer keeps no running statistics).
    Batch,
    /// The running mean and variance, one value per channel each, which evaluation uses.
    Running {
        /// The running mean.
        mean: &'a PooledBuffer,
        /// The running variance.
        var: &'a PooledBuffer,
    },
}

/// 2-D batch normalisation on the GPU over NCHW tensors (`ChannelDims::new(batch, C, h·w)`),
/// as `cavitation_core`'s `Tape::batch_norm2d` computes it: built once with
/// [`new`](BatchNorm2d::new), then per layer a forward call
/// ([`forward_call`](BatchNorm2d::forward_call)), a backward call
/// ([`backward_call`](BatchNorm2d::backward_call)) and a running update
/// ([`running_call`](BatchNorm2d::running_call)) prepared when a model is compiled, and
/// recorded in each step.
///
/// The statistics buffer holds `3 · C` values, `[mean | inv | unbiased]`. Against the exact
/// result of each kernel's own `f32` inputs, with `u = 2⁻²⁴`, `L = m = outer · inner`
/// values per channel and every multiply-add allowed two roundings:
///
/// - **Sums:** `Σx` and `Σ dy` within `2(L + 2)·u·Σ|x|`, the centred squares and `Σ dy·x̂`
///   within `(2(L + 2) + 3)·u·Σ|term|`.
/// - **Statistics:** the mean within `5u·|mean|` (WGSL's division is 2.5 ulp), `inv` within
///   `8u·inv` (and `inverseSqrt` 2 ulp), the unbiased variance within `5u` of itself.
/// - **Normalisation:** for `t = (x − mean)·inv`, `y` within `u·(|y| + 3·|t·γ|)` (affine) or
///   `2u·|t|`.
/// - **Input gradient:** through running statistics within `2u·|dx|`; through batch ones,
///   for `S = m·|dy·γ| + |Σdy·γ| + |x̂·Σ(dy·x̂)·γ|`, within `u·(|dx| + 12·(inv/m)·S)`, the
///   difference's cancellation bounded by the three magnitudes.
/// - **Running update:** within `2u·(|r'| + |momentum·b| + |(1 − momentum)·r|)`, with the
///   `f32` momentum and `1 − momentum` [`RunningUpdate::rates`] gives.
///
/// Each bound adds `2⁻¹²⁶` for every value that may be flushed on the way, times what scales
/// it afterwards:
/// - **Sums:** `(2L − 1)` for the plain sums. The centred squares add `L` more, and
///   `Σ dy·x̂` adds `Σ (|dy|·(inv + 1) + 1)`.
/// - **Statistics:** 1 each.
/// - **Normalisation:** `inv·|γ| + |γ| + 2`.
/// - **Input gradient:** `inv + 1` through running statistics. Through batch ones,
///   `2·inv + (inv/m)·(5 + (1 + inv)·|Σ(dy·x̂)·γ|) + S + 1`.
/// - **Running update:** 3.
///
/// The statistics' bounds assume `var + eps` is normal, at least `2⁻¹²⁶`. A non-zero eps
/// whose `f32` would be subnormal is refused.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{BatchNorm2d, BatchStats, ChannelDims};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let bn = BatchNorm2d::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     // Two images of one channel of two values: the mean is 2.5, the variance 1.25.
///     let dims = ChannelDims::new(2, 1, 2);
///     let (x, y, stats) = (pool.take(16, usage), pool.take(16, usage), pool.take(12, usage));
///     let sizes = StepSizes { upload_bytes: 256, readback_bytes: 16, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let call = bn.forward_call(&mut steps, &mut pool, dims, 0.0, BatchStats::Batch, &x, None, &y, &stats);
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes(&[1.0, 2.0, 3.0, 4.0]));
///     bn.record_forward(&mut step, &call);
///     step.read_back(y.buffer(), 0, 16);
///     let out = step.finish().bytes().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect::<Vec<f32>>();
///     let want = [-1.5f32, -0.5, 0.5, 1.5].map(|v| v / 1.25f32.sqrt());
///     assert!(out.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6));
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct BatchNorm2d {
    /// `Σx` and `Σ dy`.
    sums: ChannelSums,
    /// `Σ (x − mean)²`.
    centred: TermSums,
    /// `Σ dy · x̂`.
    dy_xhat: TermSums,
    /// The statistics kernels' layout.
    stats_group: wgpu::BindGroupLayout,
    /// The batch mean.
    mean: wgpu::ComputePipeline,
    /// `inv` and the unbiased variance from the centred squares.
    finish: wgpu::ComputePipeline,
    /// The statistics from running ones.
    running_stats: wgpu::ComputePipeline,
    /// The normalisation's layout.
    normalise_group: wgpu::BindGroupLayout,
    /// The normalisation.
    normalise: wgpu::ComputePipeline,
    /// The input gradient's layout.
    backward_group: wgpu::BindGroupLayout,
    /// The input gradient.
    backward: wgpu::ComputePipeline,
    /// The running update's layout.
    running_group: wgpu::BindGroupLayout,
    /// The running update.
    running: wgpu::ComputePipeline,
}

impl BatchNorm2d {
    /// Builds every kernel on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::BatchNorm2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     BatchNorm2d::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<BatchNorm2d, GpuError> {
        use Kind::{Read, Uniform, Write};
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        let params = Uniform(PARAMS_BYTES);
        let (stats_group, stats_layout) =
            layout(gpu, "batch norm stats", &[Write, params, Read, Read]);
        let (normalise_group, normalise_layout) = layout(
            gpu,
            "batch norm normalise",
            &[Write, params, Read, Read, Read, Read],
        );
        let (backward_group, backward_layout) = layout(
            gpu,
            "batch norm backward",
            &[Write, params, Read, Read, Read, Read, Read, Read],
        );
        let (running_group, running_layout) = layout(
            gpu,
            "batch norm running update",
            &[Write, Uniform(RUNNING_BYTES), Read, Read, Write],
        );
        let stats = |cache: &mut PipelineCache, entry| {
            cache.get_with_layout(STATS, entry, &workgroup, &stats_layout)
        };
        Ok(BatchNorm2d {
            sums: ChannelSums::new(gpu, cache)?,
            centred: TermSums::new(gpu, cache, CENTRED_SQUARES, 1)?,
            dy_xhat: TermSums::new(gpu, cache, DY_XHAT, 2)?,
            mean: stats(cache, "mean")?,
            finish: stats(cache, "finish")?,
            running_stats: stats(cache, "running")?,
            stats_group,
            normalise: cache.get_with_layout(NORMALISE, "main", &workgroup, &normalise_layout)?,
            normalise_group,
            backward: cache.get_with_layout(BACKWARD, "main", &workgroup, &backward_layout)?,
            backward_group,
            running: cache.get_with_layout(RUNNING, "main", &workgroup, &running_layout)?,
            running_group,
        })
    }

    /// Prepares one forward pass of `x` (shaped `dims`) into `y`, its statistics into
    /// `stats` (`3 · C` values) taken `from` the batch or the running statistics, with
    /// `eps`, and `affine` `(γ, β)` if the layer has them: its uniform, the batch sums of a
    /// batch pass and the bind groups, made once when a model is compiled.
    ///
    /// # Panics
    ///
    /// As `ChannelSums::call` for the shape; if batch statistics would be taken over one
    /// value per channel (the CPU refuses that too), a buffer is shorter than its values,
    /// `y` or `stats` is a buffer it is computed from or written next to, or eps is not
    /// finite and non-negative or is subnormal in `f32`.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d`].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ChannelDims,
        eps: f64,
        from: BatchStats<'_>,
        x: &PooledBuffer,
        affine: Option<(&PooledBuffer, &PooledBuffer)>,
        y: &PooledBuffer,
        stats: &PooledBuffer,
    ) -> BatchNormForward {
        dims.check("batch norm");
        let c = u64::from(dims.channels);
        assert!(
            eps.is_finite() && eps >= 0.0,
            "batch norm: eps must be finite and non-negative, got {eps}"
        );
        assert!(
            eps == 0.0 || eps as f32 >= f32::MIN_POSITIVE,
            "batch norm: eps must be 0 or at least 2⁻¹²⁶ in f32, which flushes smaller ones; \
             got {eps}"
        );
        check_len(NAME, "input", x, dims.len());
        check_len(NAME, "output", y, dims.len());
        check_len(NAME, "statistics", stats, 3 * c);
        let mut read = vec![("input", x)];
        if let Some((w, b)) = affine {
            check_len(NAME, "weight", w, c);
            check_len(NAME, "bias", b, c);
            read.extend([("weight", w), ("bias", b)]);
        }
        match from {
            BatchStats::Running { mean, var } => {
                check_len(NAME, "running mean", mean, c);
                check_len(NAME, "running variance", var, c);
                read.extend([("running mean", mean), ("running variance", var)]);
            }
            BatchStats::Batch => assert!(
                dims.summed() > 1,
                "batch norm: batch statistics need more than one value per channel, got \
                 {dims:?}"
            ),
        }
        check_distinct(NAME, ("statistics", stats), &read);
        read.push(("statistics", stats));
        check_distinct(NAME, ("output", y), &read);
        let flags = if affine.is_some() { AFFINE } else { 0 };
        let uniform = upload_uniform(steps, pool, &params(dims, flags, eps));
        let (batch, running_group) = match from {
            BatchStats::Batch => {
                let readable = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
                let (sums, squares) = (pool.take(4 * c, readable), pool.take(4 * c, readable));
                let sum_x = self.sums.call(steps, pool, dims, x, &sums, false);
                let centred = self.centred.call(
                    steps,
                    pool,
                    "batch norm centred squares",
                    dims,
                    (x, 0),
                    &[(stats, 3 * c)],
                    SumsTo::new(&squares),
                );
                let batch = BatchPass {
                    mean_group: bind(steps, &self.stats_group, &[stats, &uniform, &sums, &sums]),
                    finish_group: bind(
                        steps,
                        &self.stats_group,
                        &[stats, &uniform, &squares, &squares],
                    ),
                    sum_x,
                    centred,
                    sums,
                    squares,
                };
                (Some(batch), None)
            }
            BatchStats::Running { mean, var } => (
                None,
                Some(bind(
                    steps,
                    &self.stats_group,
                    &[stats, &uniform, mean, var],
                )),
            ),
        };
        let (w, b) = affine.unwrap_or((x, x));
        let normalise_group = bind(steps, &self.normalise_group, &[y, &uniform, x, stats, w, b]);
        BatchNormForward {
            dims,
            affine: affine.is_some(),
            input: x.buffer().clone(),
            stats: stats.buffer().clone(),
            uniform,
            batch,
            running_group,
            normalise_group,
        }
    }

    /// Adds the forward pass `call` to `step`: from the batch, its sum, mean, centred squares
    /// and `inv`; from running statistics, one per-channel kernel; then the normalisation.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d`].
    pub fn record_forward(&self, step: &mut Step<'_>, call: &BatchNormForward) {
        let per_channel = call.dims.channels.div_ceil(WORKGROUP);
        if let Some(batch) = &call.batch {
            self.sums.record(step, &batch.sum_x);
            step.dispatch(&self.mean, &batch.mean_group, per_channel);
            self.centred.record(step, &batch.centred);
            step.dispatch(&self.finish, &batch.finish_group, per_channel);
        }
        if let Some(group) = &call.running_group {
            step.dispatch(&self.running_stats, group, per_channel);
        }
        step.dispatch_split(
            &self.normalise,
            &call.normalise_group,
            value_workgroups(call.dims),
        );
    }

    /// Prepares the backward pass of the forward pass `forward`, from the same input `x` and
    /// statistics `stats`: the input gradient `grad_in` from the output gradient `grad_out`,
    /// through the batch statistics or the running ones as `forward` took them, and, for an
    /// affine layer, `(γ, dγ, dβ)`, the weight and the two gradients `Σ dy·x̂` and `Σ dy`.
    /// Every gradient is overwritten. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// As `ChannelSums::call` for the shape; if `x` or `stats` is not the forward pass's,
    /// `affine` is given for a forward pass without one or the other way round, a buffer is
    /// shorter than its values, or a gradient is a buffer it is computed from (or `dγ` is
    /// `dβ`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{BatchNorm2d, BatchStats, ChannelDims};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let bn = BatchNorm2d::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    ///         | wgpu::BufferUsages::COPY_SRC;
    ///     let dims = ChannelDims::new(2, 1, 2);
    ///     let [x, y, dy, dx] = [(); 4].map(|_| pool.take(16, usage));
    ///     let stats = pool.take(12, usage);
    ///     let [w, b, dw, db] = [(); 4].map(|_| pool.take(4, usage));
    ///     let sizes = StepSizes { upload_bytes: 256, readback_bytes: 32, ..StepSizes::default() };
    ///     let mut steps = StepEncoder::new(&gpu, sizes);
    ///     let forward = bn.forward_call(&mut steps, &mut pool, dims, 0.0, BatchStats::Batch, &x, Some((&w, &b)), &y, &stats);
    ///     let backward = bn.backward_call(&mut steps, &mut pool, &forward, &x, &stats, &dy, &dx, Some((&w, &dw, &db)));
    ///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    ///     let mut step = steps.begin();
    ///     step.upload(x.buffer(), 0, &bytes(&[1.0, 2.0, 3.0, 4.0]));
    ///     step.upload(w.buffer(), 0, &bytes(&[1.0]));
    ///     step.upload(b.buffer(), 0, &bytes(&[0.0]));
    ///     step.upload(dy.buffer(), 0, &bytes(&[1.0, 1.0, 1.0, 1.0]));
    ///     bn.record_forward(&mut step, &forward);
    ///     bn.record_backward(&mut step, &backward);
    ///     step.read_back(dx.buffer(), 0, 16);
    ///     step.read_back(db.buffer(), 0, 4);
    ///     let out = step.finish().bytes().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect::<Vec<f32>>();
    ///     // A constant output gradient moves no input: the mean takes it all.
    ///     assert!(out[..4].iter().all(|g| g.abs() < 1e-6));
    ///     assert_eq!(out[4], 4.0); // dβ = Σ dy
    ///     forward.release(&mut pool);
    ///     backward.release(&mut pool);
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn backward_call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        forward: &BatchNormForward,
        x: &PooledBuffer,
        stats: &PooledBuffer,
        grad_out: &PooledBuffer,
        grad_in: &PooledBuffer,
        affine: Option<(&PooledBuffer, &PooledBuffer, &PooledBuffer)>,
    ) -> BatchNormBackward {
        let dims = forward.dims;
        let (len, c) = (dims.len(), u64::from(dims.channels));
        assert!(
            affine.is_some() == forward.affine,
            "batch norm: the backward pass needs γ, dγ and dβ exactly when the forward pass \
             is affine"
        );
        assert!(
            *x.buffer() == forward.input && *stats.buffer() == forward.stats,
            "batch norm: the backward pass must read the forward pass's input and statistics"
        );
        check_len(NAME, "output gradient", grad_out, len);
        check_len(NAME, "input gradient", grad_in, len);
        let mut read = vec![
            ("output gradient", grad_out),
            ("input", x),
            ("statistics", stats),
        ];
        if let Some((w, dw, db)) = affine {
            check_len(NAME, "weight", w, c);
            check_len(NAME, "weight gradient", dw, c);
            check_len(NAME, "bias gradient", db, c);
            check_distinct(
                NAME,
                ("weight gradient", dw),
                &[("bias gradient", db), ("weight", w)],
            );
            check_distinct(
                NAME,
                ("bias gradient", db),
                &[("weight", w), ("input", x), ("statistics", stats)],
            );
            read.extend([
                ("weight", w),
                ("weight gradient", dw),
                ("bias gradient", db),
            ]);
        }
        check_distinct(NAME, ("input gradient", grad_in), &read);
        let batch = forward.batch.is_some();
        let flags = (if forward.affine { AFFINE } else { 0 }) | (if batch { BATCH } else { 0 });
        let uniform = upload_uniform(steps, pool, &params(dims, flags, 0.0));
        let scratch = (affine.is_none() && batch)
            .then(|| [(); 2].map(|_| pool.take(4 * c, wgpu::BufferUsages::STORAGE)));
        let targets = match (affine, &scratch) {
            (Some((_, dw, db)), _) => Some((db, dw)),
            (None, Some([sdy, sdx])) => Some((sdy, sdx)),
            (None, None) => None,
        };
        let sums = targets.map(|(sdy, sdx)| BackwardSums {
            sum_dy: self.sums.call(steps, pool, dims, grad_out, sdy, false),
            dy_xhat: self.dy_xhat.call(
                steps,
                pool,
                "batch norm dy x̂",
                dims,
                (grad_out, 0),
                &[(x, len), (stats, 3 * c)],
                SumsTo::new(sdx),
            ),
        });
        let (sdy, sdx) = targets.unwrap_or((grad_out, grad_out));
        let w = affine.map_or(grad_out, |a| a.0);
        let group = bind(
            steps,
            &self.backward_group,
            &[grad_in, &uniform, grad_out, x, stats, w, sdy, sdx],
        );
        BatchNormBackward {
            dims,
            uniform,
            sums,
            scratch,
            group,
        }
    }

    /// Adds the backward pass `call` to `step`: the sums `Σ dy` and `Σ dy·x̂` when the
    /// gradient needs them, then the input gradient.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::backward_call`].
    pub fn record_backward(&self, step: &mut Step<'_>, call: &BatchNormBackward) {
        if let Some(sums) = &call.sums {
            self.sums.record(step, &sums.sum_dy);
            self.dy_xhat.record(step, &sums.dy_xhat);
        }
        step.dispatch_split(&self.backward, &call.group, value_workgroups(call.dims));
    }

    /// Prepares the running statistics' update after the forward pass `forward`, which took
    /// the batch's statistics into `stats`: the running `(mean, var)` move towards the batch
    /// mean and unbiased variance, unless the `u32` at the start of `finite` is non-zero
    /// (the step's finite flag: 0 when every value was finite). `momentum` is each batch's
    /// weight, or `None` for the cumulative average over `count` batches, the
    /// `num_batches_tracked` the layer keeps (if it keeps one), as
    /// `cavitation_core::Graph::apply_buffer_updates` takes them. The count stays on the
    /// host: each [`record_running`](BatchNorm2d::record_running) uploads the rates for the
    /// next batch, and [`RunningUpdate::commit`] counts it once the step is known to have
    /// run.
    ///
    /// # Panics
    ///
    /// If `forward` took running statistics, `stats` is not its statistics, a buffer is
    /// shorter than its values, or a running statistic is `stats`, `finite` or the other
    /// one.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{BatchNorm2d, BatchStats, ChannelDims};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let bn = BatchNorm2d::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    ///         | wgpu::BufferUsages::COPY_SRC;
    ///     let dims = ChannelDims::new(2, 1, 2);
    ///     let (x, y, stats) = (pool.take(16, usage), pool.take(16, usage), pool.take(12, usage));
    ///     let [mean, var, finite] = [(); 3].map(|_| pool.take(4, usage));
    ///     let sizes = StepSizes { upload_bytes: 256, readback_bytes: 16, ..StepSizes::default() };
    ///     let mut steps = StepEncoder::new(&gpu, sizes);
    ///     let forward = bn.forward_call(&mut steps, &mut pool, dims, 0.0, BatchStats::Batch, &x, None, &y, &stats);
    ///     let mut update = bn.running_call(&mut steps, &mut pool, &forward, &stats, (&mean, &var), &finite, None, Some(0.0));
    ///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    ///     let mut step = steps.begin();
    ///     step.upload(x.buffer(), 0, &bytes(&[1.0, 2.0, 3.0, 4.0]));
    ///     step.upload(mean.buffer(), 0, &bytes(&[0.0]));
    ///     step.upload(var.buffer(), 0, &bytes(&[1.0]));
    ///     step.clear(finite.buffer(), 0, 4);
    ///     bn.record_forward(&mut step, &forward);
    ///     bn.record_running(&mut step, &update); // the first batch: momentum 1
    ///     let at = [step.read_back(mean.buffer(), 0, 4), step.read_back(var.buffer(), 0, 4)];
    ///     let output = step.finish();
    ///     let [m, v] = at.map(|i| f32::from_le_bytes(output.bytes()[i..i + 4].try_into().unwrap()));
    ///     update.commit(true);
    ///     assert_eq!(update.count(), Some(1.0));
    ///     assert!((m - 2.5).abs() < 1e-6 && (v - 5.0 / 3.0).abs() < 1e-6);
    ///     forward.release(&mut pool);
    ///     update.release(&mut pool);
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn running_call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        forward: &BatchNormForward,
        stats: &PooledBuffer,
        (mean, var): (&PooledBuffer, &PooledBuffer),
        finite: &PooledBuffer,
        momentum: Option<f64>,
        count: Option<f32>,
    ) -> RunningUpdate {
        assert!(
            forward.batch.is_some(),
            "batch norm: the running update needs a forward pass from the batch's statistics"
        );
        assert!(
            *stats.buffer() == forward.stats,
            "batch norm: the running update must read the forward pass's statistics"
        );
        let c = u64::from(forward.dims.channels);
        check_len(NAME, "running mean", mean, c);
        check_len(NAME, "running variance", var, c);
        check_len(NAME, "finite flag", finite, 1);
        check_distinct(
            NAME,
            ("running mean", mean),
            &[
                ("statistics", stats),
                ("finite flag", finite),
                ("running variance", var),
            ],
        );
        check_distinct(
            NAME,
            ("running variance", var),
            &[("statistics", stats), ("finite flag", finite)],
        );
        let uniform = pool.take(
            RUNNING_BYTES,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let group = bind(
            steps,
            &self.running_group,
            &[mean, &uniform, stats, finite, var],
        );
        RunningUpdate {
            channels: forward.dims.channels,
            momentum,
            count,
            uniform,
            group,
        }
    }

    /// Adds the running update `update` to `step`: an upload of the rates
    /// [`RunningUpdate::rates`] gives (16 bytes of the step's upload space), then one thread
    /// per channel, which leaves the statistics as they are when the finite flag is set.
    /// Record it after the kernels that write the flag.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::running_call`].
    pub fn record_running(&self, step: &mut Step<'_>, update: &RunningUpdate) {
        let (momentum, keep) = update.rates();
        let bytes = word_bytes([update.channels, momentum.to_bits(), keep.to_bits(), 0]);
        step.upload(update.uniform.buffer(), 0, &bytes);
        step.dispatch(
            &self.running,
            &update.group,
            update.channels.div_ceil(WORKGROUP),
        );
    }
}

/// Workgroups of one thread per value of a tensor shaped `dims`, for `dispatch_split`.
fn value_workgroups(dims: ChannelDims) -> u32 {
    dims.len().div_ceil(u64::from(WORKGROUP)) as u32
}

/// The parts of a forward pass from the batch's statistics.
#[derive(Debug)]
struct BatchPass {
    /// `Σx` per channel.
    sum_x: ChannelSumsCall,
    /// `Σ (x − mean)²` per channel.
    centred: ChannelSumsCall,
    /// The mean kernel's bind group.
    mean_group: wgpu::BindGroup,
    /// The finish kernel's bind group.
    finish_group: wgpu::BindGroup,
    /// `Σx`, one value per channel.
    sums: PooledBuffer,
    /// `Σ (x − mean)²`, one value per channel.
    squares: PooledBuffer,
}

/// One prepared forward pass of [`BatchNorm2d`], from [`BatchNorm2d::forward_call`].
///
/// # Examples
///
/// See [`BatchNorm2d`].
#[derive(Debug)]
pub struct BatchNormForward {
    /// The tensor's shape.
    dims: ChannelDims,
    /// Whether the layer is affine.
    affine: bool,
    /// The input, which the backward pass must read too.
    input: wgpu::Buffer,
    /// The statistics, which the backward pass and the running update must read too.
    stats: wgpu::Buffer,
    /// The `Params` uniform.
    uniform: PooledBuffer,
    /// The batch statistics' passes, when the statistics are the batch's.
    batch: Option<BatchPass>,
    /// The running statistics' bind group, when the statistics are running ones.
    running_group: Option<wgpu::BindGroup>,
    /// The normalisation's bind group.
    normalise_group: wgpu::BindGroup,
}

impl BatchNormForward {
    /// The shape this forward pass was prepared for.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d`].
    pub fn dims(&self) -> ChannelDims {
        self.dims
    }

    /// The batch sums the statistics come from, `(Σx, Σ (x − mean)²)`, one value per
    /// channel each, when the statistics are the batch's; both can be read back.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d`].
    pub fn sums(&self) -> Option<(&PooledBuffer, &PooledBuffer)> {
        self.batch.as_ref().map(|b| (&b.sums, &b.squares))
    }

    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        if let Some(batch) = self.batch {
            batch.sum_x.release(pool);
            batch.centred.release(pool);
            pool.give(batch.sums);
            pool.give(batch.squares);
        }
    }
}

/// The sums a backward pass takes.
#[derive(Debug)]
struct BackwardSums {
    /// `Σ dy` per channel.
    sum_dy: ChannelSumsCall,
    /// `Σ dy · x̂` per channel.
    dy_xhat: ChannelSumsCall,
}

/// One prepared backward pass of [`BatchNorm2d`], from [`BatchNorm2d::backward_call`].
///
/// # Examples
///
/// See [`BatchNorm2d::backward_call`].
#[derive(Debug)]
pub struct BatchNormBackward {
    /// The tensor's shape.
    dims: ChannelDims,
    /// The `Params` uniform.
    uniform: PooledBuffer,
    /// `Σ dy` and `Σ dy · x̂`, when the input gradient or the affine gradients need them.
    sums: Option<BackwardSums>,
    /// Where those sums go when the layer has no `dγ` and `dβ` to hold them.
    scratch: Option<[PooledBuffer; 2]>,
    /// The input gradient's bind group.
    group: wgpu::BindGroup,
}

impl BatchNormBackward {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::backward_call`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        if let Some(sums) = self.sums {
            sums.sum_dy.release(pool);
            sums.dy_xhat.release(pool);
        }
        for buffer in self.scratch.into_iter().flatten() {
            pool.give(buffer);
        }
    }
}

/// One prepared running-statistics update of [`BatchNorm2d`], from
/// [`BatchNorm2d::running_call`], with the batch count it keeps on the host.
///
/// # Examples
///
/// See [`BatchNorm2d::running_call`].
#[derive(Debug)]
pub struct RunningUpdate {
    /// The channels.
    channels: u32,
    /// Each batch's weight, or `None` for the cumulative average.
    momentum: Option<f64>,
    /// The batches counted so far, if the layer keeps a count.
    count: Option<f32>,
    /// The `Running` uniform, uploaded by each record.
    uniform: PooledBuffer,
    /// The update's bind group.
    group: wgpu::BindGroup,
}

impl RunningUpdate {
    /// The `f32` weights the next recorded update gives the batch and the running value,
    /// `(momentum, 1 − momentum)`, rounded as `Graph::apply_buffer_updates` rounds them on
    /// `f32`: the momentum itself, else one over the count after this batch, else 1.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::running_call`].
    pub fn rates(&self) -> (f32, f32) {
        let momentum = match (self.momentum, self.count) {
            (Some(m), _) => m as f32,
            (None, Some(n)) => 1.0 / (n + 1.0),
            (None, None) => 1.0,
        };
        (momentum, 1.0 - momentum)
    }

    /// Counts the batch of the last recorded update if its step `ran` (its finite flag was
    /// clear), as the CPU counts a step it does not skip.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::running_call`].
    pub fn commit(&mut self, ran: bool) {
        if ran && let Some(n) = &mut self.count {
            *n += 1.0;
        }
    }

    /// The batches counted so far, if the layer keeps a count.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::running_call`].
    pub fn count(&self) -> Option<f32> {
        self.count
    }

    /// Sets the batches counted so far to `count`, for a layer that keeps a count: a second
    /// update of the same layer (a training step compiled for another batch size) takes up the
    /// count where the first left it. A layer without a count is left as it is.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{BatchNorm2d, BatchStats, ChannelDims};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let bn = BatchNorm2d::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    ///         | wgpu::BufferUsages::COPY_SRC;
    ///     let dims = ChannelDims::new(2, 1, 2);
    ///     let (x, y, stats) = (pool.take(16, usage), pool.take(16, usage), pool.take(12, usage));
    ///     let [mean, var, finite] = [(); 3].map(|_| pool.take(4, usage));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let forward = bn.forward_call(&mut steps, &mut pool, dims, 0.0, BatchStats::Batch, &x, None, &y, &stats);
    ///     // A cumulative average (no momentum), 3 batches counted so far.
    ///     let mut update = bn.running_call(&mut steps, &mut pool, &forward, &stats, (&mean, &var), &finite, None, Some(3.0));
    ///     update.set_count(7.0);
    ///     assert_eq!(update.count(), Some(7.0));
    ///     // The next batch is the 8th: it gets an eighth of the weight.
    ///     assert_eq!(update.rates(), (0.125, 0.875));
    ///     forward.release(&mut pool);
    ///     update.release(&mut pool);
    /// }
    /// ```
    pub fn set_count(&mut self, count: f32) {
        if let Some(n) = &mut self.count {
            *n = count;
        }
    }

    /// Gives this call's uniform back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`BatchNorm2d::running_call`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
    }
}
