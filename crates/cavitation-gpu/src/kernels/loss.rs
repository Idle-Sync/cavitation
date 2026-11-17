//! The training losses on the GPU, as `cavitation_core`'s tape computes them with
//! `Reduction::Mean`:
//!
//! - [`SoftmaxCrossEntropy`]: softmax cross-entropy over integer class labels, the mean over
//!   the batch.
//! - [`MeanSquaredError`]: the mean over every value.
//!
//! Each writes its loss, one `f32` word, at a word offset of an output buffer, so a step's
//! results can sit one after another in one small readback. Each also writes the gradient of
//! that loss with respect to its input, overwriting it. The mean's sum runs in the
//! per-channel sums' fixed tree, so the results are the same on every run.

use super::ChannelDims;
use super::bindings::{Kind, bind, check_distinct, check_len, layout, upload_uniform, word_bytes};
use super::reduce::{CHANNEL_PARTIALS, ChannelSumsCall, SumsTo, TermSums, partials_pass};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// Threads per workgroup in every kernel here, passed as the `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// Bytes of each kernel's uniform: four words.
const UNIFORM_BYTES: u64 = 16;

/// The most values a loss's input may hold. The mean squared error's sum is one channel of
/// the per-channel sums, which take at most 65,535 chunks of 4,096. The cross-entropy's
/// logits are never summed and only need `u32` indices, but share the limit so both losses
/// take the same sizes.
const MAX_VALUES: u64 = 65_535 * 4_096;

/// Softmax cross-entropy, one thread per row: the row's loss `ln Σ e^(z − max) − (z_label −
/// max)` and its gradient `w · (softmax − onehot)`. The sum leaves out the first maximum's
/// term, `e = Σ_{j≠k} e^(z_j − max)`, and the logarithm is `ln(1 + e)`: for `e ≤ 1` it is
/// `2·atanh(e / (2 + e))` from its series to `y¹⁵`, which keeps `e`'s low bits that `1 + e`
/// rounded to `f32` would drop and avoids WGSL's `log`, whose error near 1 is 2⁻²¹ absolute
/// rather than relative. A row with a
/// logit whose exponent bits are all set (NaN or ±∞), or with a label past the classes, gets
/// NaN for its loss and its gradient. NaN is written as bits through `u32` views of both
/// outputs, since WGSL may fold NaN arithmetic away.
const ROWS: Shader = Shader {
    name: "softmax cross-entropy rows",
    source: "
        override WORKGROUP: u32;

        struct Rows { batch: u32, classes: u32, w: f32, pad: u32 }

        @group(0) @binding(0) var<storage, read_write> grad: array<u32>;
        @group(0) @binding(1) var<uniform> r: Rows;
        @group(0) @binding(2) var<storage, read> logits: array<f32>;
        @group(0) @binding(3) var<storage, read> labels: array<u32>;
        @group(0) @binding(4) var<storage, read_write> row_loss: array<u32>;

        const NAN_BITS: u32 = 0x7fc00000u;

        @compute @workgroup_size(WORKGROUP)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i >= r.batch) { return; }
            let base = i * r.classes;
            let label = labels[i];
            var bad = label >= r.classes;
            var m = logits[base];
            for (var j = 0u; j < r.classes; j = j + 1u) {
                let z = logits[base + j];
                bad = bad || (bitcast<u32>(z) & 0x7f800000u) == 0x7f800000u;
                m = max(m, z);
            }
            if (bad) {
                row_loss[i] = NAN_BITS;
                for (var j = 0u; j < r.classes; j = j + 1u) { grad[base + j] = NAN_BITS; }
                return;
            }
            var k = 0u;
            for (var j = 0u; j < r.classes; j = j + 1u) {
                if (logits[base + j] == m) { k = j; break; }
            }
            var e = 0.0;
            for (var j = 0u; j < r.classes; j = j + 1u) {
                if (j != k) { e = e + exp(logits[base + j] - m); }
            }
            var lse: f32;
            if (e <= 1.0) {
                let y = e / (2.0 + e);
                let q = y * y;
                var t = 1.0 / 15.0;
                t = 1.0 / 13.0 + q * t;
                t = 1.0 / 11.0 + q * t;
                t = 1.0 / 9.0 + q * t;
                t = 1.0 / 7.0 + q * t;
                t = 1.0 / 5.0 + q * t;
                t = 1.0 / 3.0 + q * t;
                t = 1.0 + q * t;
                lse = 2.0 * y * t;
            } else {
                lse = log(1.0 + e);
            }
            row_loss[i] = bitcast<u32>(-(logits[base + label] - m - lse));
            for (var j = 0u; j < r.classes; j = j + 1u) {
                let p = exp(logits[base + j] - m - lse);
                let onehot = select(0.0, 1.0, j == label);
                grad[base + j] = bitcast<u32>(r.w * (p - onehot));
            }
        }
    ",
};

/// `(p − t)²` summed for the mean squared error.
const SQUARED_ERRORS: Shader = Shader {
    name: "mean squared error sum",
    source: partials_pass!(
        "
        @group(0) @binding(0) var<storage, read> pred: array<f32>;
        @group(0) @binding(3) var<storage, read> wanted: array<f32>;
        ",
        "
        fn term(i: u32, c: u32) -> f32 {
            let d = pred[i] - wanted[i];
            return d * d;
        }
        "
    ),
};

/// The mean squared error's gradient, one thread per value over `dispatch_split`'s grid:
/// `2 · (p − t) · w`, as the CPU computes it (no multiply-add, so the CPU's bits).
const MSE_GRAD: Shader = Shader {
    name: "mean squared error gradient",
    source: "
        override WORKGROUP: u32;

        struct Mse { n: u32, w: f32, pad0: u32, pad1: u32 }

        @group(0) @binding(0) var<storage, read_write> grad: array<f32>;
        @group(0) @binding(1) var<uniform> e: Mse;
        @group(0) @binding(2) var<storage, read> pred: array<f32>;
        @group(0) @binding(3) var<storage, read> wanted: array<f32>;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= e.n) { return; }
            grad[i] = 2.0 * (pred[i] - wanted[i]) * e.w;
        }
    ",
};

/// `1 / n` rounded to `f32` from `f64`, as the CPU's upstream gradient of a mean over `n`.
fn reciprocal(n: u64) -> f32 {
    (1.0 / n as f64) as f32
}

/// Softmax cross-entropy over a batch of logits `[batch][classes]` and integer labels, on the
/// GPU, as `Tape::softmax_cross_entropy` with `Reduction::Mean` computes it: the loss (one
/// word) and the logits' gradient `(softmax − onehot) / batch`.
///
/// Each row runs in one thread. It takes the row's maximum, then `ln Σ e^(z − max)` as
/// `ln(1 + e)` with `e` the sum without the first maximum's term, then its loss and gradient.
/// For `e ≤ 1` the logarithm comes from the series of `2·atanh(e / (2 + e))`, so a confident
/// row's small loss keeps its relative accuracy; the CPU rounds `1 + e` to `f32` first, so
/// on such rows it can sit below the exact loss where this kernel does not. The row losses' mean is a per-channel sum in a fixed tree, divided by
/// the batch rounded to `f32`. A row with a NaN or infinite logit, or a label not below the
/// classes, gives a NaN loss and a NaN gradient row, so the step's finite flag is set. The
/// CPU would give a finite loss for a −∞ logit off the label, and panics on a bad label;
/// `fit` refuses bad labels before training, so only a hand-written loop meets that case.
///
/// The bounds are against the exact result of the same `f32` logits. WGSL gives `exp` within
/// `3 + 2|x|` ulp, `log` within 3 ulp off `[0.5, 2]` and division within 2.5 ulp, and every
/// multiply-add may round twice. With `x_j = z_j − max`, `e = Σ_{j≠k} e^(x_j)`, `S = 1 + e`
/// and `ℓ = ln S − x_label`:
///
/// - **`e`:** within `Δe = Σ_{j≠k} (6 + 5|x_j|)·u·e^(x_j) + 2(K + 1)·u·e` for `K` classes.
/// - **`ln S`:** within `Δ = Δe + 10u·ln S` if `e ≤ 1` (the quotient's 6u, the series' 2u
///   with its truncation, the last product's u), else `Δ = (Δe + u·S)/S + 6u·ln S`.
/// - **A row's loss:** within `2u·(|z_label| + |max| + ln S) + Δ`: its two subtractions in
///   either order, as a compiler may reassociate them.
/// - **The loss:** the row losses' bounds summed and divided by the batch, plus
///   `(22 + chunks)·u·Σℓ / B` for the sum (each row loss passes through at most 15 + 8 +
///   (chunks − 1) additions of the fixed tree, in any order within a lane) and `5u` of
///   itself for the division.
/// - **A gradient:** with `t_j = x_j − ln S` and `p_j = e^(t_j)`, within
///   `w·p_j·(2u·(|z_j| + |max| + ln S) + Δ + (6 + 4|t_j|)·u) + 3u·|p_j − onehot|/B` of
///   `(p_j − onehot)/B`: the subtraction of the onehot, `w`'s own rounding and the product.
///
/// Each bound adds 2⁻¹²⁶ for every value that may be flushed.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::SoftmaxCrossEntropy;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let ce = SoftmaxCrossEntropy::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let [logits, grad] = [(); 2].map(|_| pool.take(12, usage));
///     let (labels, out) = (pool.take(4, usage), pool.take(8, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 128, readback_bytes: 8, ..StepSizes::default() });
///     let call = ce.call(&mut steps, &mut pool, (1, 3), &logits, &labels, &grad, (&out, 1));
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(logits.buffer(), 0, &bytes(&[1.0, 2.0, 3.0]));
///     step.upload(labels.buffer(), 0, &2u32.to_le_bytes());
///     ce.record(&mut step, &call);
///     step.read_back(out.buffer(), 4, 4);
///     let loss = f32::from_le_bytes(step.finish().bytes()[..4].try_into().unwrap());
///     assert!((loss - 0.407_606).abs() < 1e-5);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct SoftmaxCrossEntropy {
    /// The rows kernel's layout.
    rows_group: wgpu::BindGroupLayout,
    /// The rows kernel.
    rows: wgpu::ComputePipeline,
    /// The sum of the row losses.
    sums: TermSums,
}

impl SoftmaxCrossEntropy {
    /// Builds the kernels on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`SoftmaxCrossEntropy`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<SoftmaxCrossEntropy, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (rows_group, rows_layout) = layout(
            gpu,
            "softmax cross-entropy rows",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read, Write],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(SoftmaxCrossEntropy {
            rows: cache.get_with_layout(ROWS, "main", &workgroup, &rows_layout)?,
            rows_group,
            sums: TermSums::new(gpu, cache, CHANNEL_PARTIALS, 0)?,
        })
    }

    /// Prepares the loss of `logits` (`batch × classes` row-major `f32`s) against `labels`
    /// (`batch` `u32`s, each below `classes`) into word `out.1` of `out.0` (one word,
    /// overwritten), and its gradient into `grad` (as long as the logits, overwritten). Made
    /// once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If `batch` or `classes` is 0, the batch is more than 65,535 · 256 rows or the logits
    /// more than 268,431,360 values, a buffer is shorter than its values, or an output is a
    /// buffer the kernels read (or the gradient is the loss's buffer).
    ///
    /// # Examples
    ///
    /// See [`SoftmaxCrossEntropy`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        (batch, classes): (u32, u32),
        logits: &PooledBuffer,
        labels: &PooledBuffer,
        grad: &PooledBuffer,
        (out, offset): (&PooledBuffer, u32),
    ) -> CrossEntropyCall {
        let what = "softmax cross-entropy";
        let values = u64::from(batch) * u64::from(classes);
        assert!(
            batch > 0 && classes > 0,
            "{what}: the batch and the classes must be above 0, got {batch} × {classes}"
        );
        assert!(
            values <= MAX_VALUES
                && batch.div_ceil(WORKGROUP) <= crate::MAX_WORKGROUPS_PER_DIMENSION,
            "{what}: {batch} × {classes} logits are more than the kernels take"
        );
        check_len(what, "logits", logits, values);
        check_len(what, "labels", labels, u64::from(batch));
        check_len(what, "gradient", grad, values);
        check_len(what, "loss output", out, u64::from(offset) + 1);
        check_distinct(
            what,
            ("gradient", grad),
            &[("logits", logits), ("labels", labels), ("loss output", out)],
        );
        check_distinct(
            what,
            ("loss output", out),
            &[("logits", logits), ("labels", labels)],
        );
        let uniform = upload_uniform(
            steps,
            pool,
            &word_bytes([batch, classes, reciprocal(u64::from(batch)).to_bits(), 0]),
        );
        let row_loss = pool.take(4 * u64::from(batch), wgpu::BufferUsages::STORAGE);
        let group = bind(
            steps,
            &self.rows_group,
            &[grad, &uniform, logits, labels, &row_loss],
        );
        let to = SumsTo {
            offset,
            divisor: Some(batch as f32),
            ..SumsTo::new(out)
        };
        let dims = ChannelDims::new(1, 1, batch);
        let sum = self
            .sums
            .call(steps, pool, what, dims, (&row_loss, 0), &[], to);
        CrossEntropyCall {
            batch,
            uniform,
            row_loss,
            group,
            sum,
        }
    }

    /// Adds the loss `call` to `step`: the rows, then the mean of their losses.
    ///
    /// # Examples
    ///
    /// See [`SoftmaxCrossEntropy`].
    pub fn record(&self, step: &mut Step<'_>, call: &CrossEntropyCall) {
        step.dispatch(&self.rows, &call.group, call.batch.div_ceil(WORKGROUP));
        self.sums.record(step, &call.sum);
    }
}

/// One prepared softmax cross-entropy, from [`SoftmaxCrossEntropy::call`].
///
/// # Examples
///
/// See [`SoftmaxCrossEntropy`].
#[derive(Debug)]
pub struct CrossEntropyCall {
    /// The rows.
    batch: u32,
    /// The rows kernel's uniform.
    uniform: PooledBuffer,
    /// Each row's loss.
    row_loss: PooledBuffer,
    /// The rows kernel's bind group.
    group: wgpu::BindGroup,
    /// The mean of the row losses.
    sum: ChannelSumsCall,
}

impl CrossEntropyCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`SoftmaxCrossEntropy`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        pool.give(self.row_loss);
        self.sum.release(pool);
    }
}

/// The mean squared error of a prediction against a target, on the GPU, as `Tape::mse` with
/// `Reduction::Mean` computes it: the loss `Σ (p − t)² / n` (one word) and the prediction's
/// gradient `2 · (p − t) · (1/n)`.
///
/// The gradient takes one subtraction, an exact doubling and one product, so on normal values
/// it is the CPU's, bit for bit. The loss's sum runs in the per-channel sums' fixed tree, so
/// each square passes through at most 15 + 8 + (chunks − 1) additions (in any order within a
/// lane) for `chunks = ⌈n / 4096⌉`. Against the exact mean of the same `f32` values it is
/// within `(25 + chunks)·u·Σ (p − t)² / n` (the 3 for the difference and the square), plus
/// `5u` of itself for the division and `4 · 2⁻¹²⁶` for flushes. A NaN or infinite value gives a NaN or infinite loss, and a
/// difference large enough to overflow its square gives an infinite loss with a finite
/// gradient. The finite flag tests the loss, so it catches both.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::MeanSquaredError;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mse = MeanSquaredError::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let [pred, target, grad] = [(); 3].map(|_| pool.take(8, usage));
///     let out = pool.take(4, usage);
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 128, readback_bytes: 16, ..StepSizes::default() });
///     let call = mse.call(&mut steps, &mut pool, 2, &pred, &target, &grad, (&out, 0));
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(pred.buffer(), 0, &bytes(&[1.0, 2.0]));
///     step.upload(target.buffer(), 0, &bytes(&[0.0, 0.0]));
///     mse.record(&mut step, &call);
///     let at = [step.read_back(out.buffer(), 0, 4), step.read_back(grad.buffer(), 0, 8)];
///     let output = step.finish();
///     assert_eq!(&output.bytes()[at[0]..at[0] + 4], &bytes(&[2.5])[..]);
///     assert_eq!(&output.bytes()[at[1]..at[1] + 8], &bytes(&[1.0, 2.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct MeanSquaredError {
    /// The gradient kernel's layout.
    grad_group: wgpu::BindGroupLayout,
    /// The gradient kernel.
    grad: wgpu::ComputePipeline,
    /// The sum of the squared errors.
    sums: TermSums,
}

impl MeanSquaredError {
    /// Builds the kernels on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`MeanSquaredError`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<MeanSquaredError, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (grad_group, grad_layout) = layout(
            gpu,
            "mean squared error gradient",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(MeanSquaredError {
            grad: cache.get_with_layout(MSE_GRAD, "main", &workgroup, &grad_layout)?,
            grad_group,
            sums: TermSums::new(gpu, cache, SQUARED_ERRORS, 1)?,
        })
    }

    /// Prepares the loss of `pred` against `target` (`n` `f32`s each) into word `out.1` of
    /// `out.0` (one word, overwritten), and the prediction's gradient into `grad` (`n`
    /// values, overwritten). Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If `n` is 0 or more than 268,431,360, a buffer is shorter than its values, or an
    /// output is a buffer the kernels read (or the gradient is the loss's buffer).
    ///
    /// # Examples
    ///
    /// See [`MeanSquaredError`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        n: u32,
        pred: &PooledBuffer,
        target: &PooledBuffer,
        grad: &PooledBuffer,
        (out, offset): (&PooledBuffer, u32),
    ) -> MseCall {
        let what = "mean squared error";
        assert!(
            n > 0 && u64::from(n) <= MAX_VALUES,
            "{what}: {n} values, where the kernels take 1 to {MAX_VALUES}"
        );
        let len = u64::from(n);
        check_len(what, "prediction", pred, len);
        check_len(what, "target", target, len);
        check_len(what, "gradient", grad, len);
        check_len(what, "loss output", out, u64::from(offset) + 1);
        check_distinct(
            what,
            ("gradient", grad),
            &[
                ("prediction", pred),
                ("target", target),
                ("loss output", out),
            ],
        );
        check_distinct(
            what,
            ("loss output", out),
            &[("prediction", pred), ("target", target)],
        );
        let uniform = upload_uniform(
            steps,
            pool,
            &word_bytes([n, reciprocal(len).to_bits(), 0, 0]),
        );
        let group = bind(steps, &self.grad_group, &[grad, &uniform, pred, target]);
        let to = SumsTo {
            offset,
            divisor: Some(n as f32),
            ..SumsTo::new(out)
        };
        let dims = ChannelDims::new(1, 1, n);
        let sum = self
            .sums
            .call(steps, pool, what, dims, (pred, 0), &[(target, len)], to);
        MseCall {
            n,
            uniform,
            group,
            sum,
        }
    }

    /// Adds the loss `call` to `step`: the gradient, then the mean of the squared errors.
    ///
    /// # Examples
    ///
    /// See [`MeanSquaredError`].
    pub fn record(&self, step: &mut Step<'_>, call: &MseCall) {
        step.dispatch_split(&self.grad, &call.group, call.n.div_ceil(WORKGROUP));
        self.sums.record(step, &call.sum);
    }
}

/// One prepared mean squared error, from [`MeanSquaredError::call`].
///
/// # Examples
///
/// See [`MeanSquaredError`].
#[derive(Debug)]
pub struct MseCall {
    /// The values.
    n: u32,
    /// The gradient kernel's uniform.
    uniform: PooledBuffer,
    /// The gradient kernel's bind group.
    group: wgpu::BindGroup,
    /// The mean of the squared errors.
    sum: ChannelSumsCall,
}

impl MseCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`MeanSquaredError`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        self.sum.release(pool);
    }
}
