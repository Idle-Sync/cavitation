//! The training losses on the GPU, as `cavitation_core`'s tape computes them with
//! `Reduction::Mean` (and, for binary cross-entropy and KL divergence, `Reduction::Sum` and
//! `Reduction::BatchMean`):
//!
//! - [`SoftmaxCrossEntropy`]: softmax cross-entropy over integer class labels, the mean over
//!   the batch.
//! - [`MeanSquaredError`]: the mean over every value.
//! - [`NegativeLogLikelihood`]: the negative log-likelihood of log-probabilities over
//!   integer class labels, the mean over the batch.
//! - [`RegressionLoss`]: the L1, Huber or smooth L1 loss ([`Regression`]), the mean over
//!   every value.
//! - [`BinaryCrossEntropy`]: binary cross-entropy of probabilities or of logits
//!   ([`BinaryInput`]), the mean, the sum or the batch mean over every value
//!   ([`LossReduction`]), followed by the count of logarithms the tape clamps.
//! - [`KlDivergence`]: the KL divergence of a target distribution from log-probabilities,
//!   reduced as a [`LossReduction`] says.
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
/// row's small loss keeps its relative accuracy, as the CPU's `ln_1p` of the same sum does.
/// The row losses' mean is a per-channel sum in a fixed tree, divided by
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

/// The negative log-likelihood, one thread per row: the row's loss `−lp[label]` (its sign
/// bit flipped, so exactly the CPU's) and its gradient, `−w` at the label and `+0` elsewhere,
/// all as bits, since WGSL may fold NaN arithmetic away. A row with a NaN gets NaN for its
/// loss and keeps the CPU's finite gradient; a row whose label is not below the classes gets
/// NaN for its loss and its whole gradient row.
const NLL_ROWS: Shader = Shader {
    name: "negative log-likelihood rows",
    source: "
        override WORKGROUP: u32;

        struct Rows { batch: u32, classes: u32, w: f32, pad: u32 }

        @group(0) @binding(0) var<storage, read_write> grad: array<u32>;
        @group(0) @binding(1) var<uniform> r: Rows;
        @group(0) @binding(2) var<storage, read> log_probs: array<u32>;
        @group(0) @binding(3) var<storage, read> labels: array<u32>;
        @group(0) @binding(4) var<storage, read_write> row_loss: array<u32>;

        const NAN_BITS: u32 = 0x7fc00000u;
        const INF_BITS: u32 = 0x7f800000u;
        const SIGN_BIT: u32 = 0x80000000u;

        @compute @workgroup_size(WORKGROUP)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i >= r.batch) { return; }
            let base = i * r.classes;
            let label = labels[i];
            if (label >= r.classes) {
                row_loss[i] = NAN_BITS;
                for (var j = 0u; j < r.classes; j = j + 1u) { grad[base + j] = NAN_BITS; }
                return;
            }
            var nan = false;
            for (var j = 0u; j < r.classes; j = j + 1u) {
                nan = nan || (log_probs[base + j] & 0x7fffffffu) > INF_BITS;
            }
            row_loss[i] = select(log_probs[base + label] ^ SIGN_BIT, NAN_BITS, nan);
            let minus_w = bitcast<u32>(r.w) ^ SIGN_BIT;
            for (var j = 0u; j < r.classes; j = j + 1u) {
                grad[base + j] = select(0u, minus_w, j == label);
            }
        }
    ",
};

/// The L1, Huber and smooth L1 losses, one thread per value over `dispatch_split`'s grid:
/// each value's loss into `terms` and the prediction's gradient into `grad`, as bits. `kind`
/// is 0 for L1, 1 for Huber and 2 for smooth L1, `param` the threshold. The difference's
/// non-finite cases come from the inputs' bits: NaN (a NaN input, or two infinities of one
/// sign) gives NaN for the loss and the gradient; ±∞ puts every loss on its linear branch,
/// where the loss is +∞ (NaN for Huber and smooth L1 with an infinite threshold, as
/// `∞ − ∞` gives on the CPU) and the gradient `sign(d)·w` (Huber's `sign(d)·delta·w`). A
/// finite difference runs the CPU's expressions, in its order and without a multiply-add.
const REGRESSION: Shader = Shader {
    name: "regression loss terms",
    source: "
        override WORKGROUP: u32;

        struct Regression { n: u32, w: f32, kind: u32, param: f32 }

        @group(0) @binding(0) var<storage, read_write> grad: array<u32>;
        @group(0) @binding(1) var<uniform> r: Regression;
        @group(0) @binding(2) var<storage, read> pred: array<f32>;
        @group(0) @binding(3) var<storage, read> wanted: array<f32>;
        @group(0) @binding(4) var<storage, read_write> terms: array<u32>;

        const NAN_BITS: u32 = 0x7fc00000u;
        const INF_BITS: u32 = 0x7f800000u;
        const SIGN_BIT: u32 = 0x80000000u;

        // The bits of pred - wanted, its NaN and infinite cases decided from the inputs' bits.
        fn difference(i: u32) -> u32 {
            let pb = bitcast<u32>(pred[i]);
            let tb = bitcast<u32>(wanted[i]);
            let pm = pb & 0x7fffffffu;
            let tm = tb & 0x7fffffffu;
            if (pm > INF_BITS || tm > INF_BITS) { return NAN_BITS; }
            if (pm == INF_BITS && tm == INF_BITS) { return select(NAN_BITS, pb, pb != tb); }
            if (pm == INF_BITS) { return pb; }
            if (tm == INF_BITS) { return tb ^ SIGN_BIT; }
            return bitcast<u32>(pred[i] - wanted[i]);
        }

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= r.n) { return; }
            let db = difference(i);
            let magnitude = db & 0x7fffffffu;
            if (magnitude > INF_BITS) {
                terms[i] = NAN_BITS;
                grad[i] = NAN_BITS;
                return;
            }
            let sign = db & SIGN_BIT;
            let w = bitcast<u32>(r.w);
            if (magnitude == INF_BITS) {
                let infinite_param = (bitcast<u32>(r.param) & 0x7fffffffu) == INF_BITS;
                if (r.kind == 0u) {
                    terms[i] = INF_BITS;
                    grad[i] = w ^ sign;
                } else if (r.kind == 1u) {
                    terms[i] = select(INF_BITS, NAN_BITS, infinite_param);
                    grad[i] = select(bitcast<u32>(r.param * r.w), INF_BITS, infinite_param) ^ sign;
                } else {
                    terms[i] = select(INF_BITS, NAN_BITS, infinite_param);
                    grad[i] = w ^ sign;
                }
                return;
            }
            let d = bitcast<f32>(db);
            let a = abs(d);
            // sign(d) · w, with sign(±0) = 0 giving +0, as the CPU's sign.
            var unit = 0u;
            if (d > 0.0) { unit = w; }
            if (d < 0.0) { unit = w ^ SIGN_BIT; }
            if (r.kind == 0u) {
                terms[i] = bitcast<u32>(a);
                grad[i] = unit;
            } else if (r.kind == 1u) {
                let delta = r.param;
                if (a < delta) {
                    terms[i] = bitcast<u32>(0.5 * d * d);
                    grad[i] = bitcast<u32>(d * r.w);
                } else {
                    terms[i] = bitcast<u32>(delta * (a - 0.5 * delta));
                    grad[i] = bitcast<u32>(select(-delta, delta, d > 0.0) * r.w);
                }
            } else {
                let beta = r.param;
                if (a < beta) {
                    terms[i] = bitcast<u32>(0.5 * d * d / beta);
                    grad[i] = bitcast<u32>(d / beta * r.w);
                } else {
                    terms[i] = bitcast<u32>(a - 0.5 * beta);
                    grad[i] = unit;
                }
            }
        }
    ",
};

/// The negative log-likelihood of log-probabilities `[batch][classes]` (for example a
/// log-softmax's output) against integer labels, on the GPU, as `Tape::nll` with
/// `Reduction::Mean` computes it: the loss `Σ −lp[i, label_i] / batch` (one word) and the
/// log-probabilities' gradient, `−1/batch` at each row's label and 0 elsewhere.
///
/// Each row runs in one thread. Its loss is its label's value with the sign bit flipped,
/// and its gradient the upstream `1/batch` rounded to `f32` as the CPU rounds it, negated:
/// both are the CPU's bits. A row with a NaN anywhere gets a NaN loss, as on the CPU, and
/// keeps its finite gradient; ±∞ passes through (a −∞ at the label gives a +∞ loss). A
/// label not below the classes gives a NaN loss and a NaN gradient row, so the step's
/// finite flag is set; the CPU panics on it, and `fit` refuses it before training.
///
/// The row losses' mean is a per-channel sum in a fixed tree, divided by the batch rounded
/// to `f32`, as the cross-entropy's: each row loss passes through at most 15 + 8 +
/// (chunks − 1) additions, so against the exact mean of the same `f32` values the loss is
/// within `(22 + chunks)·u·Σ|ℓ| / B` plus `5u` of itself for the division, with
/// `u = 2⁻²⁴`, and `2⁻¹²⁶` per flushed value. Two runs give the same bits.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::NegativeLogLikelihood;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let nll = NegativeLogLikelihood::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let [log_probs, grad] = [(); 2].map(|_| pool.take(12, usage));
///     let (labels, out) = (pool.take(4, usage), pool.take(4, usage));
///     let sizes = StepSizes { upload_bytes: 128, readback_bytes: 32, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let call = nll.call(&mut steps, &mut pool, (1, 3), &log_probs, &labels, &grad, (&out, 0));
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(log_probs.buffer(), 0, &bytes(&[-1.0, -2.0, -3.0]));
///     step.upload(labels.buffer(), 0, &1u32.to_le_bytes());
///     nll.record(&mut step, &call);
///     let at = [step.read_back(out.buffer(), 0, 4), step.read_back(grad.buffer(), 0, 12)];
///     let output = step.finish();
///     assert_eq!(&output.bytes()[at[0]..at[0] + 4], &bytes(&[2.0])[..]);
///     assert_eq!(&output.bytes()[at[1]..at[1] + 12], &bytes(&[0.0, -1.0, 0.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct NegativeLogLikelihood {
    /// The rows kernel's layout.
    rows_group: wgpu::BindGroupLayout,
    /// The rows kernel.
    rows: wgpu::ComputePipeline,
    /// The sum of the row losses.
    sums: TermSums,
}

impl NegativeLogLikelihood {
    /// Builds the kernels on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`NegativeLogLikelihood`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<NegativeLogLikelihood, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (rows_group, rows_layout) = layout(
            gpu,
            "negative log-likelihood rows",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read, Write],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(NegativeLogLikelihood {
            rows: cache.get_with_layout(NLL_ROWS, "main", &workgroup, &rows_layout)?,
            rows_group,
            sums: TermSums::new(gpu, cache, CHANNEL_PARTIALS, 0)?,
        })
    }

    /// Prepares the loss of `log_probs` (`batch × classes` row-major `f32`s) against
    /// `labels` (`batch` `u32`s, each below `classes`) into word `out.1` of `out.0` (one
    /// word, overwritten), and its gradient into `grad` (as long as the log-probabilities,
    /// overwritten). Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If `batch` or `classes` is 0, the batch is more than 65,535 · 256 rows or the values
    /// more than 268,431,360, a buffer is shorter than its values, or an output is a buffer
    /// the kernels read (or the gradient is the loss's buffer).
    ///
    /// # Examples
    ///
    /// See [`NegativeLogLikelihood`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        (batch, classes): (u32, u32),
        log_probs: &PooledBuffer,
        labels: &PooledBuffer,
        grad: &PooledBuffer,
        (out, offset): (&PooledBuffer, u32),
    ) -> NllCall {
        let what = "negative log-likelihood";
        let values = u64::from(batch) * u64::from(classes);
        assert!(
            batch > 0 && classes > 0,
            "{what}: the batch and the classes must be above 0, got {batch} × {classes}"
        );
        assert!(
            values <= MAX_VALUES
                && batch.div_ceil(WORKGROUP) <= crate::MAX_WORKGROUPS_PER_DIMENSION,
            "{what}: {batch} × {classes} log-probabilities are more than the kernels take"
        );
        check_len(what, "log-probabilities", log_probs, values);
        check_len(what, "labels", labels, u64::from(batch));
        check_len(what, "gradient", grad, values);
        check_len(what, "loss output", out, u64::from(offset) + 1);
        check_distinct(
            what,
            ("gradient", grad),
            &[
                ("log-probabilities", log_probs),
                ("labels", labels),
                ("loss output", out),
            ],
        );
        check_distinct(
            what,
            ("loss output", out),
            &[("log-probabilities", log_probs), ("labels", labels)],
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
            &[grad, &uniform, log_probs, labels, &row_loss],
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
        NllCall {
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
    /// See [`NegativeLogLikelihood`].
    pub fn record(&self, step: &mut Step<'_>, call: &NllCall) {
        step.dispatch(&self.rows, &call.group, call.batch.div_ceil(WORKGROUP));
        self.sums.record(step, &call.sum);
    }
}

/// One prepared negative log-likelihood, from [`NegativeLogLikelihood::call`].
///
/// # Examples
///
/// See [`NegativeLogLikelihood`].
#[derive(Debug)]
pub struct NllCall {
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

impl NllCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`NegativeLogLikelihood`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        pool.give(self.row_loss);
        self.sum.release(pool);
    }
}

/// Which elementwise loss [`RegressionLoss`] computes, with its threshold as the `f32` the
/// CPU's tape takes (`Loss::Huber { delta }` records `delta as f32`). With `d = p − t`:
///
/// - `L1`: `|d|`, gradient `sign(d)` (0 at `d = 0`), as `Tape::l1`.
/// - `Huber { delta }`: `0.5·d²` when `|d| < delta`, else `delta·(|d| − 0.5·delta)`;
///   gradient `d`, else `delta·sign(d)`, as `Tape::huber`. `delta` must be above 0 (+∞
///   included).
/// - `SmoothL1 { beta }`: `0.5·d² / beta` when `|d| < beta`, else `|d| − 0.5·beta`;
///   gradient `d / beta`, else `sign(d)`, as `Tape::smooth_l1`. `beta` must not be negative
///   or NaN; 0 is the L1 loss.
///
/// At `|d|` equal to the threshold both take the linear branch, as the CPU does.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::Regression;
///
/// let huber = Regression::Huber { delta: 1.0 };
/// assert_ne!(huber, Regression::SmoothL1 { beta: 1.0 });
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Regression {
    /// The absolute error.
    L1,
    /// The Huber loss.
    Huber {
        /// Where the loss turns from quadratic to linear; above 0.
        delta: f32,
    },
    /// The smooth L1 loss.
    SmoothL1 {
        /// Where the loss turns from quadratic to linear; 0 or above.
        beta: f32,
    },
}

impl Regression {
    /// The kernel's `kind` and `param` words, after the CPU tape's checks on the threshold
    /// (its panic message, after `what`, if they fail).
    fn words(self, what: &str) -> (u32, f32) {
        match self {
            Regression::L1 => (0, 0.0),
            Regression::Huber { delta } => {
                assert!(
                    delta > 0.0,
                    "{what}: huber: delta must be positive, but it is {delta:?}"
                );
                (1, delta)
            }
            Regression::SmoothL1 { beta } => {
                assert!(
                    beta >= 0.0,
                    "{what}: smooth_l1: beta must not be negative, but it is {beta:?}"
                );
                (2, beta)
            }
        }
    }
}

/// The L1, Huber or smooth L1 loss of a prediction against a target, on the GPU, as
/// `Tape::l1`, `Tape::huber` and `Tape::smooth_l1` with `Reduction::Mean` compute it: the
/// loss `Σ ℓ(p − t) / n` (one word) and the prediction's gradient `ℓ'(p − t) · (1/n)`, for
/// the [`Regression`] the call names.
///
/// One kernel writes each value's loss and gradient; the mean of the losses is a
/// per-channel sum in a fixed tree, divided by `n` rounded to `f32`, as the mean squared
/// error's. Each value runs the CPU's `f32` expressions in its order with no multiply-add,
/// so on normal values the gradient and each value's loss are the CPU's bits, except on
/// smooth L1's quadratic branch, whose divisions by `beta` WGSL gives within 2.5 ulp: there
/// each is within `8u` of the CPU's, relative (`u = 2⁻²⁴`). Against the exact mean of the
/// same `f32` values the loss is within `(22 + chunks + c)·u·Σ ℓ / n` plus `5u` of itself
/// for the division and `2⁻¹²⁶` per flushed value, where `chunks = ⌈n / 4096⌉` and `c`, a
/// term's own error, is 1 for L1, 4 for Huber and 8 for smooth L1 (`tests/loss.rs` derives
/// it).
///
/// NaN and ±∞ follow the CPU, decided from the bits: a NaN difference gives NaN for that
/// value's loss and gradient; an infinite one an infinite loss (NaN when the threshold is
/// also infinite) and the linear branch's gradient. A finite difference whose loss overflows
/// gives an infinite loss. The finite flag tests the loss, so it catches each.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{Regression, RegressionLoss};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let loss = RegressionLoss::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let [pred, target, grad] = [(); 3].map(|_| pool.take(8, usage));
///     let out = pool.take(4, usage);
///     let sizes = StepSizes { upload_bytes: 128, readback_bytes: 16, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let huber = Regression::Huber { delta: 1.0 };
///     let call = loss.call(&mut steps, &mut pool, huber, 2, &pred, &target, &grad, (&out, 0));
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(pred.buffer(), 0, &bytes(&[0.5, 3.0]));
///     step.upload(target.buffer(), 0, &bytes(&[0.0, 0.0]));
///     loss.record(&mut step, &call);
///     let at = [step.read_back(out.buffer(), 0, 4), step.read_back(grad.buffer(), 0, 8)];
///     let output = step.finish();
///     // (0.125 + 2.5) / 2, and the gradients 0.5 / 2 and 1 / 2.
///     assert_eq!(&output.bytes()[at[0]..at[0] + 4], &bytes(&[1.3125])[..]);
///     assert_eq!(&output.bytes()[at[1]..at[1] + 8], &bytes(&[0.25, 0.5])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct RegressionLoss {
    /// The terms kernel's layout.
    terms_group: wgpu::BindGroupLayout,
    /// The terms kernel: each value's loss and gradient.
    terms: wgpu::ComputePipeline,
    /// The sum of the terms.
    sums: TermSums,
}

impl RegressionLoss {
    /// Builds the kernels on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`RegressionLoss`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<RegressionLoss, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (terms_group, terms_layout) = layout(
            gpu,
            "regression loss terms",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read, Write],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(RegressionLoss {
            terms: cache.get_with_layout(REGRESSION, "main", &workgroup, &terms_layout)?,
            terms_group,
            sums: TermSums::new(gpu, cache, CHANNEL_PARTIALS, 0)?,
        })
    }

    /// Prepares the loss `kind` of `pred` against `target` (`n` `f32`s each) into word
    /// `out.1` of `out.0` (one word, overwritten), and the prediction's gradient into `grad`
    /// (`n` values, overwritten). Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// With the CPU tape's message if Huber's `delta` is not above 0 or smooth L1's `beta`
    /// is below 0 or NaN; if `n` is 0 or more than 268,431,360, a buffer is shorter than its
    /// values, or an output is a buffer the kernels read (or the gradient is the loss's
    /// buffer).
    ///
    /// # Examples
    ///
    /// See [`RegressionLoss`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        kind: Regression,
        n: u32,
        pred: &PooledBuffer,
        target: &PooledBuffer,
        grad: &PooledBuffer,
        (out, offset): (&PooledBuffer, u32),
    ) -> RegressionCall {
        let what = "regression loss";
        let (kind, param) = kind.words(what);
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
            &word_bytes([n, reciprocal(len).to_bits(), kind, param.to_bits()]),
        );
        let terms = pool.take(4 * len, wgpu::BufferUsages::STORAGE);
        let group = bind(
            steps,
            &self.terms_group,
            &[grad, &uniform, pred, target, &terms],
        );
        let to = SumsTo {
            offset,
            divisor: Some(n as f32),
            ..SumsTo::new(out)
        };
        let dims = ChannelDims::new(1, 1, n);
        let sum = self
            .sums
            .call(steps, pool, what, dims, (&terms, 0), &[], to);
        RegressionCall {
            n,
            uniform,
            terms,
            group,
            sum,
        }
    }

    /// Adds the loss `call` to `step`: each value's loss and gradient, then their mean.
    ///
    /// # Examples
    ///
    /// See [`RegressionLoss`].
    pub fn record(&self, step: &mut Step<'_>, call: &RegressionCall) {
        step.dispatch_split(&self.terms, &call.group, call.n.div_ceil(WORKGROUP));
        self.sums.record(step, &call.sum);
    }
}

/// One prepared regression loss, from [`RegressionLoss::call`].
///
/// # Examples
///
/// See [`RegressionLoss`].
#[derive(Debug)]
pub struct RegressionCall {
    /// The values.
    n: u32,
    /// The terms kernel's uniform.
    uniform: PooledBuffer,
    /// Each value's loss.
    terms: PooledBuffer,
    /// The terms kernel's bind group.
    group: wgpu::BindGroup,
    /// The mean of the terms.
    sum: ChannelSumsCall,
}

impl RegressionCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`RegressionLoss`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        pool.give(self.terms);
        self.sum.release(pool);
    }
}

/// The WGSL shared by binary cross-entropy and KL divergence: the bit constants, `atanh2(y)`
/// (`2·atanh(y)` from its series to `y¹⁵`), `ln_bits(b)` (`ln x` of the positive finite `x`
/// whose bits are `b`, subnormals included, as `k·ln 2 + 2·atanh((f − 1)/(f + 1))` for
/// `x = f·2^k` and `f` within `[√½, √2]`, `ln 2` split so that `k·ln2_hi` is exact) and
/// `scaled(b)` (the value whose bits are `b` times 2³², exactly, infinite past the largest
/// float). WGSL's `log` is 2⁻²¹ absolute near 1; `ln_bits` keeps the relative accuracy there.
macro_rules! log_bits {
    () => {
        "
        const NAN_BITS: u32 = 0x7fc00000u;
        const INF_BITS: u32 = 0x7f800000u;
        const SIGN_BIT: u32 = 0x80000000u;
        const MAGNITUDE: u32 = 0x7fffffffu;
        const ONE_BITS: u32 = 0x3f800000u;
        const LN2_HI: f32 = 0.693145751953125;
        const LN2_LO: f32 = 1.4286068203094172e-6;

        // 2·atanh(y), from its series to y^15.
        fn atanh2(y: f32) -> f32 {
            let q = y * y;
            var t = 1.0 / 15.0;
            t = 1.0 / 13.0 + q * t;
            t = 1.0 / 11.0 + q * t;
            t = 1.0 / 9.0 + q * t;
            t = 1.0 / 7.0 + q * t;
            t = 1.0 / 5.0 + q * t;
            t = 1.0 / 3.0 + q * t;
            t = 1.0 + q * t;
            return 2.0 * y * t;
        }

        // ln x for the positive finite x whose bits are b, subnormals included.
        fn ln_bits(b: u32) -> f32 {
            var e: i32;
            var m: u32;
            let field = b >> 23u;
            if (field == 0u) {
                let s = countLeadingZeros(b) - 8u;
                m = (b << s) & 0x7fffffu;
                e = -126 - i32(s);
            } else {
                m = b & 0x7fffffu;
                e = i32(field) - 127;
            }
            var f = bitcast<f32>(ONE_BITS | m);
            if (f > 1.41421356) {
                f = 0.5 * f;
                e = e + 1;
            }
            let k = f32(e);
            return k * LN2_HI + (k * LN2_LO + atanh2((f - 1.0) / (f + 1.0)));
        }

        // The value whose bits are b times 2^32, exactly (infinite past the largest float).
        fn scaled(b: u32) -> f32 {
            let sign = b & SIGN_BIT;
            let field = (b >> 23u) & 0xffu;
            if (field == 0u) {
                let v = f32(b & 0x7fffffu) * bitcast<f32>(10u << 23u);
                return bitcast<f32>(bitcast<u32>(v) | sign);
            }
            if (field >= 223u) { return bitcast<f32>(INF_BITS | sign); }
            return bitcast<f32>(b + (32u << 23u));
        }
        "
    };
}

/// Binary cross-entropy, one thread per value over `dispatch_split`'s grid: each value's loss
/// into `terms`, the prediction's gradient into `grad` (both as bits) and `1.0` into `clamps`
/// where the CPU's tape counts a clamped logarithm. `logits` is 0 for probabilities and 1 for
/// logits. The non-finite cases are decided from the inputs' bits, as the CPU's arithmetic
/// gives them, since WGSL may fold NaN and infinite arithmetic away; the rest runs the CPU's
/// `f32` expressions in its order.
///
/// Probabilities: `ln x` is computed from the bits, subnormals included, as
/// `k·ln 2 + 2·atanh(y)` with `y = (f − 1)/(f + 1)` for `x = f·2^k` and `f` within `[√½, √2]`,
/// so `ln p` near 1 keeps its relative accuracy, which WGSL's `log` (2⁻²¹ absolute near 1)
/// does not. A logarithm is clamped to −100 where the CPU's is: `p` of ±0 or of at most
/// 26·2⁻¹⁴⁹ (`ln` −100.02; 27·2⁻¹⁴⁹ gives −99.98), and `1 − p` of 0. Where `|p| < 2⁻⁶²` the
/// gradient's quotient is taken on `p` and `t` scaled by 2³² from their bits, the same
/// quotient, since a GPU may flush a subnormal `p` to 0; where `|p·(1 − p)| ≥ 2⁶⁴` both sides
/// of the quotient are scaled by 2⁻⁶⁴, since a division through the reciprocal of a
/// denominator near the largest float flushes to 0.
///
/// Logits: `max(z, 0) − z·t + ln(1 + e)` with `e = e^(−|z|)` and `ln(1 + e)` from the series
/// of `2·atanh(e / (2 + e))`, as the cross-entropy's, and the gradient `σ(z) − t` with the
/// CPU's stable sigmoid on the same `e`.
const BINARY: Shader = Shader {
    name: "binary cross-entropy terms",
    source: concat!(
        "
        override WORKGROUP: u32;

        struct Binary { n: u32, w: f32, logits: u32, pad: u32 }

        @group(0) @binding(0) var<storage, read_write> grad: array<u32>;
        @group(0) @binding(1) var<uniform> r: Binary;
        @group(0) @binding(2) var<storage, read> pred: array<u32>;
        @group(0) @binding(3) var<storage, read> wanted: array<u32>;
        @group(0) @binding(4) var<storage, read_write> terms: array<u32>;
        @group(0) @binding(5) var<storage, read_write> clamps: array<f32>;

        ",
        log_bits!(),
        "
        // The magnitude bits of 2^64: from there on p (1 - p) rounds to -infinity in f32,
        // since 1 - p rounds to -p and p^2 reaches 2^128; just below, p^2 rounds to a finite
        // float.
        const PRODUCT_OVERFLOWS: u32 = 0x5f800000u;

        fn probabilities(i: u32, pb: u32, tb: u32) {
            let pm = pb & MAGNITUDE;
            let tm = tb & MAGNITUDE;
            let p_clamped = pm == 0u || pb <= 26u;
            let q_clamped = pb == ONE_BITS;
            var count = 0.0;
            if ((p_clamped && tm != 0u) || (q_clamped && tb != ONE_BITS)) { count = 1.0; }
            clamps[i] = count;
            let p = bitcast<f32>(pb);
            let t = bitcast<f32>(tb);
            // ln p or ln(1 - p) is NaN outside [-0, 1], and an infinite target meets a zero
            // logarithm or two of one sign with weights of opposite signs.
            if ((pb <= ONE_BITS || pb == SIGN_BIT) && tm != INF_BITS) {
                var lp = -100.0;
                if (!p_clamped) { lp = ln_bits(pb); }
                var lq = -100.0;
                if (!q_clamped) { lq = ln_bits(bitcast<u32>(1.0 - p)); }
                terms[i] = bitcast<u32>(-(t * lp + (1.0 - t) * lq));
            } else {
                terms[i] = NAN_BITS;
            }
            if (pm == INF_BITS) {
                grad[i] = NAN_BITS;
            } else if (tm == INF_BITS && pm >= PRODUCT_OVERFLOWS) {
                // p (1 - p) overflows to -infinity in f32 for |p| from 2^64, and
                // infinity over infinity is NaN.
                grad[i] = NAN_BITS;
            } else if (tm == INF_BITS) {
                // -t times infinity over 1, or over p (1 - p), negative for p below 0 or
                // above 1.
                let negative = !p_clamped && !q_clamped
                    && ((pb & SIGN_BIT) != 0u || pb > ONE_BITS);
                grad[i] = INF_BITS
                    | (((tb ^ SIGN_BIT) & SIGN_BIT) ^ select(0u, SIGN_BIT, negative));
            } else if (p_clamped) {
                grad[i] = bitcast<u32>((1.0 - t) * r.w);
            } else if (q_clamped) {
                grad[i] = bitcast<u32>(-t * r.w);
            } else if (pm < (65u << 23u)) {
                let ps = scaled(pb);
                grad[i] = bitcast<u32>((ps - scaled(tb)) / ps * r.w);
            } else {
                // From |p (1 - p)| = 2^64 both sides of the quotient are scaled by 2^-64,
                // exactly, so a division by the reciprocal does not flush to 0.
                var num = p - t;
                var den = p * (1.0 - p);
                if ((bitcast<u32>(den) & MAGNITUDE) >= (191u << 23u)) {
                    num = num * bitcast<f32>(63u << 23u);
                    den = den * bitcast<f32>(63u << 23u);
                }
                grad[i] = bitcast<u32>(num / den * r.w);
            }
        }

        fn logits(i: u32, zb: u32, tb: u32) {
            clamps[i] = 0.0;
            let zm = zb & MAGNITUDE;
            let tm = tb & MAGNITUDE;
            let t = bitcast<f32>(tb);
            if (zm == INF_BITS || tm == INF_BITS) {
                // max(z, 0) - z t with an infinite z or t: infinity times 0 is NaN, as is
                // infinity minus infinity.
                if (zb == INF_BITS) {
                    terms[i] = select(NAN_BITS, INF_BITS, (tb & SIGN_BIT) != 0u && tm != 0u);
                } else if (zm == INF_BITS) {
                    terms[i] = select(INF_BITS | (tb & SIGN_BIT), NAN_BITS, tm == 0u);
                } else {
                    terms[i] = select(
                        INF_BITS | ((zb ^ tb ^ SIGN_BIT) & SIGN_BIT),
                        NAN_BITS,
                        zm == 0u,
                    );
                }
                if (tm == INF_BITS) {
                    grad[i] = INF_BITS | ((tb ^ SIGN_BIT) & SIGN_BIT);
                } else {
                    grad[i] = bitcast<u32>((select(0.0, 1.0, zb == INF_BITS) - t) * r.w);
                }
                return;
            }
            let z = bitcast<f32>(zb);
            let e = exp(-abs(z));
            let positive = select(0.0, z, z > 0.0);
            terms[i] = bitcast<u32>(positive - z * t + atanh2(e / (2.0 + e)));
            var sigma: f32;
            if (z >= 0.0) {
                sigma = 1.0 / (1.0 + e);
            } else {
                sigma = e / (1.0 + e);
            }
            grad[i] = bitcast<u32>((sigma - t) * r.w);
        }

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= r.n) { return; }
            let pb = pred[i];
            let tb = wanted[i];
            if ((pb & MAGNITUDE) > INF_BITS || (tb & MAGNITUDE) > INF_BITS) {
                terms[i] = NAN_BITS;
                grad[i] = NAN_BITS;
                clamps[i] = 0.0;
                return;
            }
            if (r.logits == 0u) {
                probabilities(i, pb, tb);
            } else {
                logits(i, pb, tb);
            }
        }
        "
    ),
};

/// What [`BinaryCrossEntropy`] reads as its prediction.
///
/// - `Probabilities`: probabilities `p`, as `Tape::binary_cross_entropy`: the loss
///   `−[t·ln p + (1 − t)·ln(1 − p)]` with each logarithm clamped to at least −100, and the
///   gradient `(p − t) / (p·(1 − p))`, or only the unclamped term's derivative where a
///   logarithm was clamped.
/// - `Logits`: raw scores `z`, as `Tape::binary_cross_entropy_with_logits`: the loss
///   `max(z, 0) − z·t + ln(1 + e^(−|z|))`, which never overflows, and the gradient `σ(z) − t`.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::BinaryInput;
///
/// assert_ne!(BinaryInput::Probabilities, BinaryInput::Logits);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryInput {
    /// Probabilities, for example a sigmoid's output.
    Probabilities,
    /// Raw scores, the sigmoid fused into the loss.
    Logits,
}

/// How [`BinaryCrossEntropy`] and [`KlDivergence`] reduce their values' losses to one word,
/// as the tape's `Reduction::Mean`, `Reduction::Sum` and `Reduction::BatchMean`.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::LossReduction;
///
/// assert_ne!(LossReduction::Mean, LossReduction::Sum);
/// assert_ne!(LossReduction::Mean, LossReduction::BatchMean { batch: 4 });
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossReduction {
    /// The sum divided by the number of values; each gradient is scaled by `1/n` rounded to
    /// `f32`, as the CPU's upstream gradient.
    Mean,
    /// The sum; each gradient is scaled by 1.
    Sum,
    /// The sum divided by the batch, the first dimension of the values; each gradient is
    /// scaled by `1/batch` rounded to `f32`. The batch must be above 0 and divide the values.
    BatchMean {
        /// The rows of the values.
        batch: u32,
    },
}

impl LossReduction {
    /// Each gradient's scale and the sum's divisor (`None` for the sum) over `n` values,
    /// after checking a batch mean's batch (the panic message after `what` if it fails).
    fn scale(self, what: &str, n: u32) -> (f32, Option<f32>) {
        match self {
            LossReduction::Mean => (reciprocal(u64::from(n)), Some(n as f32)),
            LossReduction::Sum => (1.0, None),
            LossReduction::BatchMean { batch } => {
                assert!(
                    batch > 0 && n.is_multiple_of(batch),
                    "{what}: a batch of {batch} rows does not divide {n} values"
                );
                (reciprocal(u64::from(batch)), Some(batch as f32))
            }
        }
    }
}

/// Binary cross-entropy of a prediction against a target, on the GPU, as
/// `Tape::binary_cross_entropy` and `Tape::binary_cross_entropy_with_logits` with
/// `Reduction::Mean`, `Reduction::Sum` or `Reduction::BatchMean` compute it: the loss (one
/// word), then how many logarithms the tape would clamp and count in `Tape::clamped_logs`
/// (one word, a whole number, 0 for logits), and the prediction's gradient. See
/// [`BinaryInput`] for each form.
///
/// One kernel writes each value's loss, clamp count and gradient; the losses' sum and the
/// counts' sum are per-channel sums in a fixed tree, the loss's divided by `n` (the batch for
/// the batch mean) rounded to `f32`. The count is exact up to 2²⁴ clamps. NaN in either
/// input gives NaN for that value's loss and gradient, and the other non-finite cases are the
/// CPU's, decided from the bits: a probability outside `[−0, 1]` or an infinite target gives a NaN loss (with the
/// CPU's finite gradient, or an infinite one for an infinite target, NaN where the CPU's
/// `p·(1 − p)` overflows, from `|p| = 2⁶⁴`), an infinite logit or target gives the CPU's
/// infinite or NaN loss. The finite flag tests the loss, so it catches
/// each.
///
/// The bounds, with `u = 2⁻²⁴`, `chunks = ⌈n / 4096⌉` and the exact values of the same `f32`
/// inputs (`tests/loss.rs` derives them). WGSL gives `exp` within `3 + 2|x|` ulp and
/// division within 2.5 ulp:
///
/// - **A probability's loss:** within `15u·(|t·ln p| + |(1 − t)·ln(1 − p)|)` of the CPU's
///   expression on the same `p`, `fl(1 − p)` and `fl(1 − t)` evaluated exactly, the clamped
///   logarithms −100: each logarithm within `13u` of its own, then two products and a sum.
/// - **A logit's loss:** within `u·(|z·t| + |a| + |ℓ|) + 10u·L + (6 + 4|z|)·u·e`, where
///   `a = max(z, 0) − z·t`, `e = e^(−|z|)`, `L = ln(1 + e)` and `ℓ = a + L`.
/// - **The loss:** the values' bounds summed, plus `(22 + chunks)·u·Σ|ℓ|` for the sum, then
///   for a mean divided by its divisor and `5u` of itself for the division.
/// - **A probability's gradient:** the CPU's bits where a logarithm was clamped; elsewhere its
///   one division WGSL gives within 2.5 ulp, so within `8u` of the CPU's, relative to it, as
///   smooth L1's.
/// - **A logit's gradient:** within `w·(Δσ + u·|σ(z) − t|) + u·|g|` of `g = (σ(z) − t)·w`,
///   with `w` the `f32` scale and `Δσ = 13u·σ(z)` for `z ≥ 0`, `(18 + 8|z|)·u·σ(z)` below.
///
/// Each bound adds `2⁻¹²⁶·(2 + |t|)` per value for flushed subnormals. Two runs give the same
/// bits.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{BinaryCrossEntropy, BinaryInput, LossReduction};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let bce = BinaryCrossEntropy::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let [pred, target, grad, out] = [(); 4].map(|_| pool.take(8, usage));
///     let sizes = StepSizes { upload_bytes: 128, readback_bytes: 16, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let form = (BinaryInput::Probabilities, LossReduction::Mean);
///     let call = bce.call(&mut steps, &mut pool, form, 2, &pred, &target, &grad, (&out, 0));
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(pred.buffer(), 0, &bytes(&[0.5, 0.0]));
///     step.upload(target.buffer(), 0, &bytes(&[1.0, 1.0]));
///     bce.record(&mut step, &call);
///     let at = [step.read_back(out.buffer(), 0, 8), step.read_back(grad.buffer(), 0, 8)];
///     let output = step.finish();
///     let word = |at: usize| f32::from_le_bytes(output.bytes()[at..at + 4].try_into().unwrap());
///     // (ln 2 + 100) / 2, one clamped logarithm, and the gradients -2 / 2 and 0 / 2.
///     assert!((word(at[0]) - 50.346_574).abs() < 1e-4);
///     assert_eq!(word(at[0] + 4), 1.0);
///     assert_eq!((word(at[1]), word(at[1] + 4)), (-1.0, 0.0));
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct BinaryCrossEntropy {
    /// The terms kernel's layout.
    terms_group: wgpu::BindGroupLayout,
    /// The terms kernel: each value's loss, clamp count and gradient.
    terms: wgpu::ComputePipeline,
    /// The sums of the terms and of the counts.
    sums: TermSums,
}

impl BinaryCrossEntropy {
    /// Builds the kernels on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`BinaryCrossEntropy`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<BinaryCrossEntropy, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (terms_group, terms_layout) = layout(
            gpu,
            "binary cross-entropy terms",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read, Write, Write],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(BinaryCrossEntropy {
            terms: cache.get_with_layout(BINARY, "main", &workgroup, &terms_layout)?,
            terms_group,
            sums: TermSums::new(gpu, cache, CHANNEL_PARTIALS, 0)?,
        })
    }

    /// Prepares the loss `form` of `pred` against `target` (`n` `f32`s each) into word
    /// `out.1` of `out.0` and the clamp count into word `out.1 + 1` (both overwritten), and
    /// the prediction's gradient into `grad` (`n` values, overwritten). Made once, when a
    /// model is compiled.
    ///
    /// # Panics
    ///
    /// If `n` is 0 or more than 268,431,360, a batch mean's batch is 0 or does not divide
    /// `n`, a buffer is shorter than its values, or an output is a buffer the kernels read (or
    /// the gradient is the loss's buffer).
    ///
    /// # Examples
    ///
    /// See [`BinaryCrossEntropy`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        (input, reduction): (BinaryInput, LossReduction),
        n: u32,
        pred: &PooledBuffer,
        target: &PooledBuffer,
        grad: &PooledBuffer,
        (out, offset): (&PooledBuffer, u32),
    ) -> BinaryCall {
        let what = "binary cross-entropy";
        assert!(
            n > 0 && u64::from(n) <= MAX_VALUES,
            "{what}: {n} values, where the kernels take 1 to {MAX_VALUES}"
        );
        let len = u64::from(n);
        check_len(what, "prediction", pred, len);
        check_len(what, "target", target, len);
        check_len(what, "gradient", grad, len);
        check_len(what, "loss output", out, u64::from(offset) + 2);
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
        let (w, divisor) = reduction.scale(what, n);
        let logits = u32::from(input == BinaryInput::Logits);
        let uniform = upload_uniform(steps, pool, &word_bytes([n, w.to_bits(), logits, 0]));
        let terms = pool.take(4 * len, wgpu::BufferUsages::STORAGE);
        let clamps = pool.take(4 * len, wgpu::BufferUsages::STORAGE);
        let group = bind(
            steps,
            &self.terms_group,
            &[grad, &uniform, pred, target, &terms, &clamps],
        );
        let dims = ChannelDims::new(1, 1, n);
        let to = SumsTo {
            offset,
            divisor,
            ..SumsTo::new(out)
        };
        let sum = self
            .sums
            .call(steps, pool, what, dims, (&terms, 0), &[], to);
        let to = SumsTo {
            offset: offset + 1,
            ..SumsTo::new(out)
        };
        let count = self
            .sums
            .call(steps, pool, what, dims, (&clamps, 0), &[], to);
        BinaryCall {
            n,
            uniform,
            terms,
            clamps,
            group,
            sum,
            count,
        }
    }

    /// Adds the loss `call` to `step`: each value's loss, count and gradient, then the
    /// losses' reduction and the counts' sum.
    ///
    /// # Examples
    ///
    /// See [`BinaryCrossEntropy`].
    pub fn record(&self, step: &mut Step<'_>, call: &BinaryCall) {
        step.dispatch_split(&self.terms, &call.group, call.n.div_ceil(WORKGROUP));
        self.sums.record(step, &call.sum);
        self.sums.record(step, &call.count);
    }
}

/// One prepared binary cross-entropy, from [`BinaryCrossEntropy::call`].
///
/// # Examples
///
/// See [`BinaryCrossEntropy`].
#[derive(Debug)]
pub struct BinaryCall {
    /// The values.
    n: u32,
    /// The terms kernel's uniform.
    uniform: PooledBuffer,
    /// Each value's loss.
    terms: PooledBuffer,
    /// Each value's clamp count.
    clamps: PooledBuffer,
    /// The terms kernel's bind group.
    group: wgpu::BindGroup,
    /// The reduction of the losses.
    sum: ChannelSumsCall,
    /// The sum of the clamp counts.
    count: ChannelSumsCall,
}

impl BinaryCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`BinaryCrossEntropy`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        pool.give(self.terms);
        pool.give(self.clamps);
        self.sum.release(pool);
        self.count.release(pool);
    }
}

/// KL divergence, one thread per value over `dispatch_split`'s grid: each value's loss
/// `t·(ln t − x)` into `terms` and the log-probabilities' gradient `−t·w` into `grad`, both
/// as bits, `x` being the log-probability and `t` the target. A target of ±0 gives a loss of
/// 0 (NaN for a NaN `x`) and a gradient of +0. The non-finite cases are decided from the
/// inputs' bits, as the CPU's arithmetic gives them, since WGSL may fold NaN and infinite
/// arithmetic away: NaN in either input gives a NaN loss, a negative target (its logarithm
/// NaN) a NaN loss, a target of +∞ a loss of +∞ (NaN for `x = +∞`), an infinite `x` against
/// a positive finite target a loss of the other sign's infinity. `ln t` comes from its bits;
/// a subnormal target's product is taken on `t·2³²`, exactly, and scaled back by 2⁻³², since
/// a GPU may flush a subnormal operand to 0.
const KL_DIV: Shader = Shader {
    name: "KL divergence terms",
    source: concat!(
        "
        override WORKGROUP: u32;

        struct Kl { n: u32, w: f32, pad0: u32, pad1: u32 }

        @group(0) @binding(0) var<storage, read_write> grad: array<u32>;
        @group(0) @binding(1) var<uniform> r: Kl;
        @group(0) @binding(2) var<storage, read> log_probs: array<u32>;
        @group(0) @binding(3) var<storage, read> wanted: array<u32>;
        @group(0) @binding(4) var<storage, read_write> terms: array<u32>;
        ",
        log_bits!(),
        "
        // 2^-32, which undoes scaled().
        const UNSCALE: f32 = 2.3283064365386963e-10;

        // t (ln t - x) for the bits of x and t, both not NaN.
        fn term(xb: u32, tb: u32) -> u32 {
            if ((tb & MAGNITUDE) == 0u) { return 0u; }
            if ((tb & SIGN_BIT) != 0u) { return NAN_BITS; }
            if (tb == INF_BITS) { return select(INF_BITS, NAN_BITS, xb == INF_BITS); }
            if ((xb & MAGNITUDE) == INF_BITS) { return INF_BITS | ((xb ^ SIGN_BIT) & SIGN_BIT); }
            let d = ln_bits(tb) - bitcast<f32>(xb);
            if ((tb >> 23u) == 0u) { return bitcast<u32>(scaled(tb) * d * UNSCALE); }
            return bitcast<u32>(bitcast<f32>(tb) * d);
        }

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= r.n) { return; }
            let xb = log_probs[i];
            let tb = wanted[i];
            let tm = tb & MAGNITUDE;
            if (tm == 0u) {
                grad[i] = 0u;
            } else if (tm > INF_BITS) {
                grad[i] = NAN_BITS;
            } else if (tm == INF_BITS) {
                grad[i] = INF_BITS | ((tb ^ SIGN_BIT) & SIGN_BIT);
            } else {
                grad[i] = bitcast<u32>(bitcast<f32>(tb) * r.w) ^ SIGN_BIT;
            }
            if ((xb & MAGNITUDE) > INF_BITS || tm > INF_BITS) {
                terms[i] = NAN_BITS;
            } else {
                terms[i] = term(xb, tb);
            }
        }
        "
    ),
};

/// The KL divergence of a target distribution from log-probabilities, on the GPU, as
/// `Tape::kl_div` with `Reduction::Mean`, `Reduction::Sum` or `Reduction::BatchMean`
/// computes it, in PyTorch's `kl_div` convention: the input `x` holds log-probabilities (for
/// example a log-softmax's output) and the target `t` probabilities, and each value's loss
/// is `t·(ln t − x)`, 0 where `t` is ±0 (`0·ln 0 = 0`). The output is the loss (one word);
/// the log-probabilities' gradient is `−t·w`, +0 where `t` is ±0, with `w` the reduction's
/// scale rounded to `f32` as the CPU's upstream gradient. `Loss::KlDiv` trains with the batch
/// mean.
///
/// One kernel writes each value's loss and gradient; the losses' sum is a per-channel sum in
/// a fixed tree, divided by `n` for the mean and by the batch for the batch mean, rounded to
/// `f32`. The gradient is one product, so it is the CPU's bits wherever the CPU's is 0, not
/// finite or normal; where it is subnormal it is within 2⁻¹²⁶ of it, as a GPU may flush.
/// NaN and ±∞ follow the CPU, decided from the bits: NaN in either input gives a NaN loss
/// (also where `t` is 0), a negative target (the caller's error) a NaN loss, a target of +∞
/// a loss of +∞ (NaN for `x = +∞`) and an infinite `x` an infinite loss; the gradient
/// stays the CPU's `−t·w`. A finite value whose loss overflows gives an infinite loss. The
/// finite flag tests the loss, so it catches each.
///
/// The bounds, with `u = 2⁻²⁴`, `chunks = ⌈n / 4096⌉` and the exact values of the same `f32`
/// inputs (`tests/loss.rs` derives them):
///
/// - **A value's loss:** within `15u·|t·ln t| + 3u·|ℓ|` of `ℓ = t·(ln t − x)`, plus
///   `2⁻¹²⁶·(1 + |t|)` for flushed subnormals. `ln t` comes from the bits as binary
///   cross-entropy's logarithm does, within `13u` of its own, so it keeps its relative
///   accuracy near `t = 1`; the subtraction and the product round once each.
/// - **The loss:** the values' bounds summed, plus `(22 + chunks)·u·Σ|ℓ|` for the sum, then
///   for a mean divided by its divisor and `5u` of itself for the division.
///
/// Two runs give the same bits.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{KlDivergence, LossReduction};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let kl = KlDivergence::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let [log_probs, target, grad] = [(); 3].map(|_| pool.take(8, usage));
///     let out = pool.take(4, usage);
///     let sizes = StepSizes { upload_bytes: 128, readback_bytes: 16, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let mean = LossReduction::BatchMean { batch: 1 };
///     let call = kl.call(&mut steps, &mut pool, mean, 2, &log_probs, &target, &grad, (&out, 0));
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     let half = 0.5f32.ln();
///     step.upload(log_probs.buffer(), 0, &bytes(&[half, half]));
///     step.upload(target.buffer(), 0, &bytes(&[0.25, 0.75]));
///     kl.record(&mut step, &call);
///     let at = [step.read_back(out.buffer(), 0, 4), step.read_back(grad.buffer(), 0, 8)];
///     let output = step.finish();
///     let word = |at: usize| f32::from_le_bytes(output.bytes()[at..at + 4].try_into().unwrap());
///     // 0.25 ln 0.5 + 0.75 ln 1.5, and the gradients -0.25 and -0.75.
///     assert!((word(at[0]) - 0.130_812).abs() < 1e-5);
///     assert_eq!((word(at[1]), word(at[1] + 4)), (-0.25, -0.75));
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct KlDivergence {
    /// The terms kernel's layout.
    terms_group: wgpu::BindGroupLayout,
    /// The terms kernel: each value's loss and gradient.
    terms: wgpu::ComputePipeline,
    /// The sum of the terms.
    sums: TermSums,
}

impl KlDivergence {
    /// Builds the kernels on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`KlDivergence`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<KlDivergence, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (terms_group, terms_layout) = layout(
            gpu,
            "KL divergence terms",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read, Write],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(KlDivergence {
            terms: cache.get_with_layout(KL_DIV, "main", &workgroup, &terms_layout)?,
            terms_group,
            sums: TermSums::new(gpu, cache, CHANNEL_PARTIALS, 0)?,
        })
    }

    /// Prepares the loss of `log_probs` against `target` (`n` `f32`s each), reduced by
    /// `reduction`, into word `out.1` of `out.0` (one word, overwritten), and the
    /// log-probabilities' gradient into `grad` (`n` values, overwritten). Made once, when a
    /// model is compiled.
    ///
    /// # Panics
    ///
    /// If `n` is 0 or more than 268,431,360, a batch mean's batch is 0 or does not divide
    /// `n`, a buffer is shorter than its values, or an output is a buffer the kernels read (or
    /// the gradient is the loss's buffer).
    ///
    /// # Examples
    ///
    /// See [`KlDivergence`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        reduction: LossReduction,
        n: u32,
        log_probs: &PooledBuffer,
        target: &PooledBuffer,
        grad: &PooledBuffer,
        (out, offset): (&PooledBuffer, u32),
    ) -> KlDivCall {
        let what = "KL divergence";
        assert!(
            n > 0 && u64::from(n) <= MAX_VALUES,
            "{what}: {n} values, where the kernels take 1 to {MAX_VALUES}"
        );
        let len = u64::from(n);
        check_len(what, "log-probabilities", log_probs, len);
        check_len(what, "target", target, len);
        check_len(what, "gradient", grad, len);
        check_len(what, "loss output", out, u64::from(offset) + 1);
        check_distinct(
            what,
            ("gradient", grad),
            &[
                ("log-probabilities", log_probs),
                ("target", target),
                ("loss output", out),
            ],
        );
        check_distinct(
            what,
            ("loss output", out),
            &[("log-probabilities", log_probs), ("target", target)],
        );
        let (w, divisor) = reduction.scale(what, n);
        let uniform = upload_uniform(steps, pool, &word_bytes([n, w.to_bits(), 0, 0]));
        let terms = pool.take(4 * len, wgpu::BufferUsages::STORAGE);
        let group = bind(
            steps,
            &self.terms_group,
            &[grad, &uniform, log_probs, target, &terms],
        );
        let to = SumsTo {
            offset,
            divisor,
            ..SumsTo::new(out)
        };
        let dims = ChannelDims::new(1, 1, n);
        let sum = self
            .sums
            .call(steps, pool, what, dims, (&terms, 0), &[], to);
        KlDivCall {
            n,
            uniform,
            terms,
            group,
            sum,
        }
    }

    /// Adds the loss `call` to `step`: each value's loss and gradient, then their reduction.
    ///
    /// # Examples
    ///
    /// See [`KlDivergence`].
    pub fn record(&self, step: &mut Step<'_>, call: &KlDivCall) {
        step.dispatch_split(&self.terms, &call.group, call.n.div_ceil(WORKGROUP));
        self.sums.record(step, &call.sum);
    }
}

/// One prepared KL divergence, from [`KlDivergence::call`].
///
/// # Examples
///
/// See [`KlDivergence`].
#[derive(Debug)]
pub struct KlDivCall {
    /// The values.
    n: u32,
    /// The terms kernel's uniform.
    uniform: PooledBuffer,
    /// Each value's loss.
    terms: PooledBuffer,
    /// The terms kernel's bind group.
    group: wgpu::BindGroup,
    /// The reduction of the losses.
    sum: ChannelSumsCall,
}

impl KlDivCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`KlDivergence`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
        pool.give(self.terms);
        self.sum.release(pool);
    }
}
