//! The training losses as this crate's own CUDA kernels, the twins of `cavitation_core`'s
//! `Tape::softmax_cross_entropy`, `Tape::mse`, `Tape::nll`, `Tape::l1`, `Tape::huber` and
//! `Tape::smooth_l1` with `Reduction::Mean`: the loss, one `f32`
//! written at a word offset of an output buffer, and the gradient of that loss with respect
//! to the input, overwriting it. The mean's sum runs in the channel sums' fixed tree
//! (`reduce.cu`), so two runs give the same bits.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::reduce::{self, BLOCK, ChannelDims};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/loss.rs`, compiled after the
/// sums' source.
const SOURCE: &str = include_str!("loss.cu");

/// Panics unless every buffer holds at least `values` four-byte values, naming the first
/// that does not.
fn check_buffers(what: &str, buffers: &[(&DeviceBuffer<'_>, &str, u64)]) {
    for &(buffer, name, values) in buffers {
        let needs = values.saturating_mul(4);
        assert!(
            buffer.bytes() as u64 >= needs,
            "{what}: the {name} buffer holds {} bytes, and {values} values need {needs}",
            buffer.bytes()
        );
    }
}

/// `1 / n` rounded to `f32` from `f64`, as the CPU's upstream gradient of a mean over `n`.
fn reciprocal(n: u64) -> f32 {
    (1.0 / n as f64) as f32
}

/// The kernels of one module and their handles.
struct Kernels<'c> {
    /// The compiled sums and loss kernels.
    module: Module<'c>,
    /// The sums' first pass over the values themselves: the cross-entropy's row losses.
    sum_partials: CuFunction,
    /// The sums' first pass over the squared errors.
    squared_partials: CuFunction,
    /// The second pass of a mean, writing the loss.
    mean_merge: CuFunction,
    /// The cross-entropy's rows.
    ce_rows: CuFunction,
    /// The mean squared error's gradient.
    mse_grad: CuFunction,
    /// The negative log-likelihood's rows.
    nll_rows: CuFunction,
    /// The regression losses' terms and gradients.
    regression_terms: CuFunction,
}

impl<'c> Kernels<'c> {
    /// Compiles the sums' and the losses' source on `cuda`'s card and looks every kernel up.
    fn new(cuda: &'c Cuda) -> Result<Kernels<'c>, CudaError> {
        let module = cuda.compile("loss.cu", &format!("{}{SOURCE}", reduce::SOURCE))?;
        let f = |name: &str| module.function(name);
        Ok(Kernels {
            sum_partials: f("sum_partials")?,
            squared_partials: f("squared_partials")?,
            mean_merge: f("mean_merge")?,
            ce_rows: f("ce_rows")?,
            mse_grad: f("mse_grad")?,
            nll_rows: f("nll_rows")?,
            regression_terms: f("regression_terms")?,
            module,
        })
    }

    /// Launches `function`, one of this module's handles, on `blocks` blocks of [`BLOCK`]
    /// threads with `args`.
    ///
    /// # Safety
    ///
    /// `args` must be the kernel's parameters, and every buffer must hold what the kernel
    /// reads and writes on that grid.
    unsafe fn launch(
        &self,
        function: CuFunction,
        blocks: u32,
        args: &[Arg<'_>],
    ) -> Result<(), CudaError> {
        // SAFETY: every handle was looked up in this module when it was compiled.
        let kernel = unsafe { self.module.kernel_from(function) };
        // SAFETY: the caller passes the kernel's parameters and buffers that hold what it
        // touches on this grid.
        unsafe { kernel.launch([blocks, 1, 1], [BLOCK, 1, 1], args) }
    }

    /// The mean of `dims.len()` terms written to `out[offset]`: the first pass `function`
    /// from `inputs` into `partials`, then the merge in chunk order and the division by
    /// `divisor`.
    ///
    /// # Safety
    ///
    /// `function` must be a first pass and `inputs` its inputs before the partial sums, each
    /// holding what the term reads for `dims`, a shape of one channel that passed
    /// `ChannelDims::check`; `partials` must hold `dims.chunks()` floats and `out`
    /// `offset + 1`, and neither may be an input.
    unsafe fn mean(
        &self,
        function: CuFunction,
        inputs: &[Arg<'_>],
        dims: ChannelDims,
        partials: &DeviceBuffer<'_>,
        (out, offset): (&DeviceBuffer<'_>, u32),
        divisor: f32,
    ) -> Result<(), CudaError> {
        let chunks = dims.chunks();
        let mut args = [Arg::U32(0); 7];
        let n = inputs.len();
        args[..n].copy_from_slice(inputs);
        args[n..n + 5].copy_from_slice(&[
            Arg::Buffer(partials),
            Arg::U32(dims.outer),
            Arg::U32(dims.channels),
            Arg::U32(dims.inner),
            Arg::U32(chunks),
        ]);
        // SAFETY: the first pass takes its inputs, then (float *partials, u32 outer, u32
        // channels, u32 inner, u32 chunks); its grid of chunks (one channel) reads values
        // below dims.len() and writes `chunks` partial sums, which the caller promises the
        // buffers hold.
        unsafe {
            self.launch(function, chunks, &args[..n + 5])?;
        }
        // SAFETY: mean_merge takes (const float *partials, float *out, u32 offset, u32
        // chunks, float divisor), reads `chunks` partial sums and writes out[offset], which
        // the caller promises the buffers hold; one thread does the work.
        unsafe {
            self.launch(
                self.mean_merge,
                1,
                &[
                    Arg::Buffer(partials),
                    Arg::Buffer(out),
                    Arg::U32(offset),
                    Arg::U32(chunks),
                    Arg::F32(divisor),
                ],
            )
        }
    }
}

/// Softmax cross-entropy on the card for one shape, logits `[batch][classes]` against
/// integer labels, as `Tape::softmax_cross_entropy` with `Reduction::Mean` computes it: the
/// loss (one word) and the logits' gradient `(softmax − onehot) / batch`.
///
/// Each row runs in one thread. It takes the row's maximum, then `ln Σ e^(z − max)` as
/// `ln(1 + e)` with `e` the sum without the first maximum's term, then its loss and gradient.
/// For `e ≤ 1` the logarithm comes from the series of `2·atanh(e / (2 + e))`, so a confident
/// row's small loss keeps its relative accuracy, as the CPU's `ln_1p` of the same sum does.
/// For `e > 1` it is
/// CUDA's `logf(1 + e)`. The row losses' mean is a channel sum in `reduce.cu`'s fixed tree,
/// divided by the batch rounded to `f32`. A row with a NaN or infinite logit, or a label not
/// below the classes, gives a NaN loss and a NaN gradient row. The CPU would give a finite
/// loss for a −∞ logit off the label, and panics on a bad label.
///
/// The kernel is `cavitation-gpu`'s WGSL one ported operation for operation, and its bounds
/// are the WGSL kernel's, against the exact result of the same `f32` logits. They allow `exp`
/// within `3 + 2|x|` ulp, `log` within 3 ulp, division within 2.5 ulp and a multiply-add
/// rounded twice; CUDA's `expf` is within 2 ulp and its `logf` within 1, NVRTC divides
/// correctly rounded, and nothing is fused, so they hold here. With `x_j = z_j − max`,
/// `e = Σ_{j≠k} e^(x_j)`, `S = 1 + e`, `ℓ = ln S − x_label` and `u = 2⁻²⁴`:
///
/// - **`e`:** within `Δe = Σ_{j≠k} (6 + 5|x_j|)·u·e^(x_j) + 2(K + 1)·u·e` for `K` classes.
/// - **`ln S`:** within `Δ = Δe + 10u·ln S` if `e ≤ 1`, else `Δ = (Δe + u·S)/S + 6u·ln S`.
/// - **A row's loss:** within `2u·(|z_label| + |max| + ln S) + Δ`.
/// - **The loss:** the row losses' bounds summed and divided by the batch, plus
///   `(22 + chunks)·u·Σℓ / B` for the sum and `5u` of itself for the division.
/// - **A gradient:** with `t_j = x_j − ln S` and `p_j = e^(t_j)`, within
///   `w·p_j·(2u·(|z_j| + |max| + ln S) + Δ + (6 + 4|t_j|)·u) + 3u·|p_j − onehot|/B` of
///   `(p_j − onehot)/B`.
///
/// Each bound also carries `2⁻¹²⁶` absolute for every exponential and every result that
/// may be subnormal, where an ulp is absolute rather than relative: `expf`'s 2 ulp are
/// about 2⁻¹⁴⁸ once `e^x` is below the smallest normal. Nothing is flushed. Two runs give
/// the same bits.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, SoftmaxCrossEntropy};
///
/// if let Some(cuda) = Cuda::new() {
///     let ce = SoftmaxCrossEntropy::new(&cuda, (1, 3)).unwrap();
///     let mut logits = cuda.buffer(12).unwrap();
///     let mut labels = cuda.buffer(4).unwrap();
///     let (mut grad, mut out) = (cuda.buffer(12).unwrap(), cuda.buffer(8).unwrap());
///     logits.write(&[1.0f32, 2.0, 3.0]).unwrap();
///     labels.write(&[2u32]).unwrap();
///     ce.run(&logits, &labels, &mut grad, (&mut out, 1)).unwrap();
///     let mut loss = [0.0f32; 2];
///     out.read(&mut loss).unwrap();
///     assert!((loss[1] - 0.407_606).abs() < 1e-5);
/// }
/// ```
pub struct SoftmaxCrossEntropy<'c> {
    /// The rows.
    batch: u32,
    /// The classes.
    classes: u32,
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// Each row's loss.
    row_loss: DeviceBuffer<'c>,
    /// One partial sum per chunk of 4,096 row losses.
    partials: DeviceBuffer<'c>,
}

impl fmt::Debug for SoftmaxCrossEntropy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SoftmaxCrossEntropy")
            .field("batch", &self.batch)
            .field("classes", &self.classes)
            .finish_non_exhaustive()
    }
}

impl<'c> SoftmaxCrossEntropy<'c> {
    /// Compiles the kernels on `cuda`'s card for `batch` rows of `classes` logits, and
    /// allocates the row losses and their partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `batch` or `classes` is 0 or the logits are more than 2³¹
    /// values; [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses
    /// the kernels or the memory.
    pub fn new(
        cuda: &'c Cuda,
        (batch, classes): (u32, u32),
    ) -> Result<SoftmaxCrossEntropy<'c>, CudaError> {
        ChannelDims::new(batch, 1, classes).check("softmax cross-entropy")?;
        let kernels = Kernels::new(cuda)?;
        let rows = ChannelDims::new(1, 1, batch);
        Ok(SoftmaxCrossEntropy {
            batch,
            classes,
            kernels,
            row_loss: cuda.buffer(4 * batch as usize)?,
            partials: cuda.buffer(4 * rows.chunks() as usize)?,
        })
    }

    /// The shape, `(batch, classes)`.
    pub fn shape(&self) -> (u32, u32) {
        (self.batch, self.classes)
    }

    /// Writes the loss of `logits` (`batch × classes` row-major `f32`s) against `labels`
    /// (`batch` `u32`s, each below `classes`) to word `out.1` of `out.0`, and its gradient to
    /// `grad` (as long as the logits). Nothing else of `out` changes.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values: the logits and the gradient `batch · classes`,
    /// the labels `batch`, and the output `out.1 + 1`.
    pub fn run(
        &self,
        logits: &DeviceBuffer<'_>,
        labels: &DeviceBuffer<'_>,
        grad: &mut DeviceBuffer<'_>,
        (out, offset): (&mut DeviceBuffer<'_>, u32),
    ) -> Result<(), CudaError> {
        const WHAT: &str = "softmax cross-entropy";
        let (batch, classes) = (self.batch, self.classes);
        let values = u64::from(batch) * u64::from(classes);
        check_buffers(
            WHAT,
            &[
                (logits, "logits", values),
                (labels, "labels", u64::from(batch)),
                (grad, "gradient", values),
                (out, "loss output", u64::from(offset) + 1),
            ],
        );
        let k = &self.kernels;
        // SAFETY: ce_rows takes (const float *logits, const u32 *labels, float *grad, float
        // *row_loss, u32 batch, u32 classes, float w), reads `batch · classes` logits and
        // `batch` labels and writes as many gradients and `batch` row losses, which the
        // buffers hold (checked above, and the row losses are this loss's own); grad,
        // borrowed mutably, is no input.
        unsafe {
            k.launch(
                k.ce_rows,
                batch.div_ceil(BLOCK),
                &[
                    Arg::Buffer(logits),
                    Arg::Buffer(labels),
                    Arg::Buffer(grad),
                    Arg::Buffer(&self.row_loss),
                    Arg::U32(batch),
                    Arg::U32(classes),
                    Arg::F32(reciprocal(u64::from(batch))),
                ],
            )?;
        }
        // SAFETY: sum_partials reads `batch` row losses from this loss's own buffer; the
        // partial sums are its own too, sized for the batch, and out, borrowed mutably,
        // holds word `offset` (checked above).
        unsafe {
            k.mean(
                k.sum_partials,
                &[Arg::Buffer(&self.row_loss)],
                ChannelDims::new(1, 1, batch),
                &self.partials,
                (out, offset),
                batch as f32,
            )
        }
    }
}

/// The mean squared error on the card for one length, as `Tape::mse` with `Reduction::Mean`
/// computes it: the loss `Σ (p − t)² / n` (one word) and the prediction's gradient
/// `2 · (p − t) · (1/n)`.
///
/// The gradient is the CPU's own `f32` expression, so it is the CPU's bits, subnormals
/// included. The loss's sum runs in `reduce.cu`'s fixed tree, and the mean is that sum
/// divided by `n` rounded to `f32`, correctly rounded: the bits of the same tree evaluated on
/// the host, and on squares whose partial sums are exact the CPU's loss bit for bit. Against
/// the exact mean of the same `f32` values it is within the WGSL kernel's bound,
/// `(25 + chunks)·u·Σ (p − t)² / n` plus `5u` of itself, for `chunks = ⌈n / 4096⌉`. A NaN or
/// infinite value gives a NaN or infinite loss, and a difference large enough to overflow
/// its square gives an infinite loss with a finite gradient.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, MeanSquaredError};
///
/// if let Some(cuda) = Cuda::new() {
///     let mse = MeanSquaredError::new(&cuda, 2).unwrap();
///     let (mut pred, mut target) = (cuda.buffer(8).unwrap(), cuda.buffer(8).unwrap());
///     let (mut grad, mut out) = (cuda.buffer(8).unwrap(), cuda.buffer(4).unwrap());
///     pred.write(&[1.0f32, 2.0]).unwrap();
///     target.write(&[0.0f32, 0.0]).unwrap();
///     mse.run(&pred, &target, &mut grad, (&mut out, 0)).unwrap();
///     let (mut loss, mut g) = ([0.0f32], [0.0f32; 2]);
///     out.read(&mut loss).unwrap();
///     grad.read(&mut g).unwrap();
///     assert_eq!((loss, g), ([2.5], [1.0, 2.0]));
/// }
/// ```
pub struct MeanSquaredError<'c> {
    /// The values.
    n: u32,
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// One partial sum per chunk of 4,096 squared errors.
    partials: DeviceBuffer<'c>,
}

impl fmt::Debug for MeanSquaredError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeanSquaredError")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl<'c> MeanSquaredError<'c> {
    /// Compiles the kernels on `cuda`'s card for `n` values, and allocates the partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `n` is 0 or more than 2³¹; [`CudaError::Compile`] or
    /// [`CudaError::Call`] if NVRTC or the driver refuses the kernels or the memory.
    pub fn new(cuda: &'c Cuda, n: u32) -> Result<MeanSquaredError<'c>, CudaError> {
        let dims = ChannelDims::new(1, 1, n);
        dims.check("mean squared error")?;
        let kernels = Kernels::new(cuda)?;
        Ok(MeanSquaredError {
            n,
            kernels,
            partials: cuda.buffer(4 * dims.chunks() as usize)?,
        })
    }

    /// The values.
    pub fn len(&self) -> u32 {
        self.n
    }

    /// Whether the loss takes no values: never, since [`new`](Self::new) refuses 0.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Writes the loss of `pred` against `target` (`n` `f32`s each) to word `out.1` of
    /// `out.0`, and the prediction's gradient to `grad` (`n` values). Nothing else of `out`
    /// changes.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values: the prediction, the target and the gradient
    /// `n`, and the output `out.1 + 1`.
    pub fn run(
        &self,
        pred: &DeviceBuffer<'_>,
        target: &DeviceBuffer<'_>,
        grad: &mut DeviceBuffer<'_>,
        (out, offset): (&mut DeviceBuffer<'_>, u32),
    ) -> Result<(), CudaError> {
        const WHAT: &str = "mean squared error";
        let n = self.n;
        let len = u64::from(n);
        check_buffers(
            WHAT,
            &[
                (pred, "prediction", len),
                (target, "target", len),
                (grad, "gradient", len),
                (out, "loss output", u64::from(offset) + 1),
            ],
        );
        let k = &self.kernels;
        // SAFETY: mse_grad takes (const float *pred, const float *target, float *grad, u32
        // n, float w) and touches values below n, which the buffers hold (checked above);
        // grad, borrowed mutably, is no input.
        unsafe {
            k.launch(
                k.mse_grad,
                n.div_ceil(BLOCK),
                &[
                    Arg::Buffer(pred),
                    Arg::Buffer(target),
                    Arg::Buffer(grad),
                    Arg::U32(n),
                    Arg::F32(reciprocal(len)),
                ],
            )?;
        }
        // SAFETY: squared_partials reads n values of pred and target, which hold them; the
        // partial sums are this loss's own, sized for n, and out, borrowed mutably, holds
        // word `offset` (checked above).
        unsafe {
            k.mean(
                k.squared_partials,
                &[Arg::Buffer(pred), Arg::Buffer(target)],
                ChannelDims::new(1, 1, n),
                &self.partials,
                (out, offset),
                n as f32,
            )
        }
    }
}

/// The negative log-likelihood on the card for one shape, log-probabilities
/// `[batch][classes]` (for example a log-softmax's output) against integer labels, as
/// `Tape::nll` with `Reduction::Mean` computes it: the loss `Σ −lp[i, label_i] / batch` (one
/// word) and the log-probabilities' gradient, `−1/batch` at each row's label and 0 elsewhere.
///
/// The kernel is `cavitation-gpu`'s WGSL one ported operation for operation. Each row runs
/// in one thread: its loss is its label's value with the sign bit flipped and its gradient
/// the upstream `1/batch` rounded to `f32` as the CPU rounds it, negated, so both are the
/// CPU's bits. A row with a NaN anywhere gets a NaN loss and keeps its finite gradient, as on
/// the CPU; ±∞ passes through. A label not below the classes gives a NaN loss and a NaN
/// gradient row; the CPU panics on it. The row losses' mean is a channel sum in
/// `reduce.cu`'s fixed tree divided by the batch rounded to `f32`, correctly rounded: within
/// `(22 + chunks)·u·Σ|ℓ| / B` plus `5u` of itself of the exact mean of the same values, with
/// `u = 2⁻²⁴`, as the WGSL kernel's. Nothing is flushed, and two runs give the same bits.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, NegativeLogLikelihood};
///
/// if let Some(cuda) = Cuda::new() {
///     let nll = NegativeLogLikelihood::new(&cuda, (1, 3)).unwrap();
///     let mut log_probs = cuda.buffer(12).unwrap();
///     let mut labels = cuda.buffer(4).unwrap();
///     let (mut grad, mut out) = (cuda.buffer(12).unwrap(), cuda.buffer(4).unwrap());
///     log_probs.write(&[-1.0f32, -2.0, -3.0]).unwrap();
///     labels.write(&[1u32]).unwrap();
///     nll.run(&log_probs, &labels, &mut grad, (&mut out, 0)).unwrap();
///     let (mut loss, mut g) = ([0.0f32], [0.0f32; 3]);
///     out.read(&mut loss).unwrap();
///     grad.read(&mut g).unwrap();
///     assert_eq!((loss, g), ([2.0], [0.0, -1.0, 0.0]));
/// }
/// ```
pub struct NegativeLogLikelihood<'c> {
    /// The rows.
    batch: u32,
    /// The classes.
    classes: u32,
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// Each row's loss.
    row_loss: DeviceBuffer<'c>,
    /// One partial sum per chunk of 4,096 row losses.
    partials: DeviceBuffer<'c>,
}

impl fmt::Debug for NegativeLogLikelihood<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NegativeLogLikelihood")
            .field("batch", &self.batch)
            .field("classes", &self.classes)
            .finish_non_exhaustive()
    }
}

impl<'c> NegativeLogLikelihood<'c> {
    /// Compiles the kernels on `cuda`'s card for `batch` rows of `classes` log-probabilities,
    /// and allocates the row losses and their partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `batch` or `classes` is 0 or the values are more than 2³¹;
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses the
    /// kernels or the memory.
    ///
    /// # Examples
    ///
    /// See [`NegativeLogLikelihood`].
    pub fn new(
        cuda: &'c Cuda,
        (batch, classes): (u32, u32),
    ) -> Result<NegativeLogLikelihood<'c>, CudaError> {
        ChannelDims::new(batch, 1, classes).check("negative log-likelihood")?;
        let kernels = Kernels::new(cuda)?;
        let rows = ChannelDims::new(1, 1, batch);
        Ok(NegativeLogLikelihood {
            batch,
            classes,
            kernels,
            row_loss: cuda.buffer(4 * batch as usize)?,
            partials: cuda.buffer(4 * rows.chunks() as usize)?,
        })
    }

    /// The shape, `(batch, classes)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, NegativeLogLikelihood};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     assert_eq!(NegativeLogLikelihood::new(&cuda, (8, 3)).unwrap().shape(), (8, 3));
    /// }
    /// ```
    pub fn shape(&self) -> (u32, u32) {
        (self.batch, self.classes)
    }

    /// Writes the loss of `log_probs` (`batch × classes` row-major `f32`s) against `labels`
    /// (`batch` `u32`s, each below `classes`) to word `out.1` of `out.0`, and its gradient to
    /// `grad` (as long as the log-probabilities). Nothing else of `out` changes.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values: the log-probabilities and the gradient
    /// `batch · classes`, the labels `batch`, and the output `out.1 + 1`.
    ///
    /// # Examples
    ///
    /// See [`NegativeLogLikelihood`].
    pub fn run(
        &self,
        log_probs: &DeviceBuffer<'_>,
        labels: &DeviceBuffer<'_>,
        grad: &mut DeviceBuffer<'_>,
        (out, offset): (&mut DeviceBuffer<'_>, u32),
    ) -> Result<(), CudaError> {
        const WHAT: &str = "negative log-likelihood";
        let (batch, classes) = (self.batch, self.classes);
        let values = u64::from(batch) * u64::from(classes);
        check_buffers(
            WHAT,
            &[
                (log_probs, "log-probabilities", values),
                (labels, "labels", u64::from(batch)),
                (grad, "gradient", values),
                (out, "loss output", u64::from(offset) + 1),
            ],
        );
        let k = &self.kernels;
        // SAFETY: nll_rows takes (const u32 *log_probs, const u32 *labels, u32 *grad, u32
        // *row_loss, u32 batch, u32 classes, float w), reads `batch · classes` values and
        // `batch` labels and writes as many gradients and `batch` row losses, which the
        // buffers hold (checked above, and the row losses are this loss's own); grad,
        // borrowed mutably, is no input.
        unsafe {
            k.launch(
                k.nll_rows,
                batch.div_ceil(BLOCK),
                &[
                    Arg::Buffer(log_probs),
                    Arg::Buffer(labels),
                    Arg::Buffer(grad),
                    Arg::Buffer(&self.row_loss),
                    Arg::U32(batch),
                    Arg::U32(classes),
                    Arg::F32(reciprocal(u64::from(batch))),
                ],
            )?;
        }
        // SAFETY: sum_partials reads `batch` row losses from this loss's own buffer; the
        // partial sums are its own too, sized for the batch, and out, borrowed mutably,
        // holds word `offset` (checked above).
        unsafe {
            k.mean(
                k.sum_partials,
                &[Arg::Buffer(&self.row_loss)],
                ChannelDims::new(1, 1, batch),
                &self.partials,
                (out, offset),
                batch as f32,
            )
        }
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
/// use cavitation_cuda::Regression;
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
    /// The kernel's `kind` and `param` arguments, after the CPU tape's checks on the
    /// threshold ([`CudaError::Invalid`] with its message, after `what`, if they fail).
    fn words(self, what: &str) -> Result<(u32, f32), CudaError> {
        match self {
            Regression::L1 => Ok((0, 0.0)),
            Regression::Huber { delta } if delta > 0.0 => Ok((1, delta)),
            Regression::Huber { delta } => Err(CudaError::Invalid(format!(
                "{what}: huber: delta must be positive, but it is {delta:?}"
            ))),
            Regression::SmoothL1 { beta } if beta >= 0.0 => Ok((2, beta)),
            Regression::SmoothL1 { beta } => Err(CudaError::Invalid(format!(
                "{what}: smooth_l1: beta must not be negative, but it is {beta:?}"
            ))),
        }
    }
}

/// The L1, Huber or smooth L1 loss on the card for one length and one [`Regression`], as
/// `Tape::l1`, `Tape::huber` and `Tape::smooth_l1` with `Reduction::Mean` compute it: the
/// loss `Σ ℓ(p − t) / n` (one word) and the prediction's gradient `ℓ'(p − t) · (1/n)`.
///
/// The kernel is `cavitation-gpu`'s WGSL one ported operation for operation: one thread
/// per value writes its loss and gradient, each the CPU's own `f32` expression in its order
/// with nothing fused and the divisions correctly rounded, so both are the CPU's bits,
/// subnormals and smooth L1's quadratic branch included. NaN and ±∞ follow the CPU, decided
/// from the bits: a NaN difference gives NaN for that value's loss and gradient, an infinite
/// one an infinite loss (NaN when the threshold is also infinite) and the linear branch's
/// gradient. The losses' mean is a channel sum in `reduce.cu`'s fixed tree divided by `n`
/// rounded to `f32`: within the WGSL kernel's bound, `(22 + chunks + c)·u·Σ ℓ / n` plus `5u`
/// of itself of the exact mean of the same values, `c` being 1 for L1, 4 for Huber and 8
/// for smooth L1 (`u = 2⁻²⁴`). Two runs give the same bits.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, Regression, RegressionLoss};
///
/// if let Some(cuda) = Cuda::new() {
///     let huber = RegressionLoss::new(&cuda, 2, Regression::Huber { delta: 1.0 }).unwrap();
///     let (mut pred, mut target) = (cuda.buffer(8).unwrap(), cuda.buffer(8).unwrap());
///     let (mut grad, mut out) = (cuda.buffer(8).unwrap(), cuda.buffer(4).unwrap());
///     pred.write(&[0.5f32, 3.0]).unwrap();
///     target.write(&[0.0f32, 0.0]).unwrap();
///     huber.run(&pred, &target, &mut grad, (&mut out, 0)).unwrap();
///     let (mut loss, mut g) = ([0.0f32], [0.0f32; 2]);
///     out.read(&mut loss).unwrap();
///     grad.read(&mut g).unwrap();
///     // (0.125 + 2.5) / 2, and the gradients 0.5 / 2 and 1 / 2.
///     assert_eq!((loss, g), ([1.3125], [0.25, 0.5]));
/// }
/// ```
pub struct RegressionLoss<'c> {
    /// The values.
    n: u32,
    /// The loss, and its threshold.
    kind: Regression,
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// Each value's loss.
    terms: DeviceBuffer<'c>,
    /// One partial sum per chunk of 4,096 terms.
    partials: DeviceBuffer<'c>,
}

impl fmt::Debug for RegressionLoss<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegressionLoss")
            .field("n", &self.n)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl<'c> RegressionLoss<'c> {
    /// Compiles the kernels on `cuda`'s card for the loss `kind` of `n` values, and
    /// allocates each value's loss and their partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] with the CPU tape's message if Huber's `delta` is not above 0
    /// or smooth L1's `beta` is below 0 or NaN, and if `n` is 0 or more than 2³¹;
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses the
    /// kernels or the memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, Regression, RegressionLoss};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     assert!(RegressionLoss::new(&cuda, 4, Regression::SmoothL1 { beta: 0.0 }).is_ok());
    ///     assert!(RegressionLoss::new(&cuda, 4, Regression::Huber { delta: 0.0 }).is_err());
    /// }
    /// ```
    pub fn new(cuda: &'c Cuda, n: u32, kind: Regression) -> Result<RegressionLoss<'c>, CudaError> {
        const WHAT: &str = "regression loss";
        kind.words(WHAT)?;
        let dims = ChannelDims::new(1, 1, n);
        dims.check(WHAT)?;
        let kernels = Kernels::new(cuda)?;
        Ok(RegressionLoss {
            n,
            kind,
            kernels,
            terms: cuda.buffer(4 * n as usize)?,
            partials: cuda.buffer(4 * dims.chunks() as usize)?,
        })
    }

    /// The values.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, Regression, RegressionLoss};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     assert_eq!(RegressionLoss::new(&cuda, 6, Regression::L1).unwrap().len(), 6);
    /// }
    /// ```
    pub fn len(&self) -> u32 {
        self.n
    }

    /// Whether the loss takes no values: never, since [`new`](Self::new) refuses 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, Regression, RegressionLoss};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     assert!(!RegressionLoss::new(&cuda, 6, Regression::L1).unwrap().is_empty());
    /// }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The loss and its threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, Regression, RegressionLoss};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let l1 = RegressionLoss::new(&cuda, 6, Regression::L1).unwrap();
    ///     assert_eq!(l1.kind(), Regression::L1);
    /// }
    /// ```
    pub fn kind(&self) -> Regression {
        self.kind
    }

    /// Writes the loss of `pred` against `target` (`n` `f32`s each) to word `out.1` of
    /// `out.0`, and the prediction's gradient to `grad` (`n` values). Nothing else of `out`
    /// changes.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values: the prediction, the target and the gradient
    /// `n`, and the output `out.1 + 1`.
    ///
    /// # Examples
    ///
    /// See [`RegressionLoss`].
    pub fn run(
        &self,
        pred: &DeviceBuffer<'_>,
        target: &DeviceBuffer<'_>,
        grad: &mut DeviceBuffer<'_>,
        (out, offset): (&mut DeviceBuffer<'_>, u32),
    ) -> Result<(), CudaError> {
        const WHAT: &str = "regression loss";
        let n = self.n;
        let len = u64::from(n);
        check_buffers(
            WHAT,
            &[
                (pred, "prediction", len),
                (target, "target", len),
                (grad, "gradient", len),
                (out, "loss output", u64::from(offset) + 1),
            ],
        );
        let (kind, param) = self.kind.words(WHAT)?;
        let k = &self.kernels;
        // SAFETY: regression_terms takes (const float *pred, const float *target, u32 *grad,
        // u32 *terms, u32 n, float w, u32 kind, float param) and touches values below n,
        // which the buffers hold (checked above, and the terms are this loss's own); grad,
        // borrowed mutably, is no input.
        unsafe {
            k.launch(
                k.regression_terms,
                n.div_ceil(BLOCK),
                &[
                    Arg::Buffer(pred),
                    Arg::Buffer(target),
                    Arg::Buffer(grad),
                    Arg::Buffer(&self.terms),
                    Arg::U32(n),
                    Arg::F32(reciprocal(len)),
                    Arg::U32(kind),
                    Arg::F32(param),
                ],
            )?;
        }
        // SAFETY: sum_partials reads n terms from this loss's own buffer; the partial sums
        // are its own too, sized for n, and out, borrowed mutably, holds word `offset`
        // (checked above).
        unsafe {
            k.mean(
                k.sum_partials,
                &[Arg::Buffer(&self.terms)],
                ChannelDims::new(1, 1, n),
                &self.partials,
                (out, offset),
                n as f32,
            )
        }
    }
}
