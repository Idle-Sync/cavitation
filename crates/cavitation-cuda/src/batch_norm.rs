//! 2-D batch normalisation as this crate's own CUDA kernels, the twin of `cavitation_core`'s
//! `Tape::batch_norm2d` on NCHW batches: the statistics from the batch (the mean first, then
//! the centred squares, each summed in a fixed tree) or from running ones, the normalisation
//! with an optional affine `γ`, `β`, the backward pass in either mode, and the running
//! statistics' update, skipped when the step's finite flag is set.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::reduce::{self, BLOCK, ChannelDims};
use crate::sys::{CuFunction, DevicePtr};

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/batch_norm.rs`, compiled after
/// the sums' source.
const SOURCE: &str = include_str!("batch_norm.cu");

/// The most arguments a first pass of the sums takes: up to three inputs, the partial sums
/// and four sizes.
const SUM_ARGS: usize = 8;

/// Where a forward pass takes its statistics from, and so which way its backward pass goes.
#[derive(Clone, Copy, Debug)]
pub enum BatchStats<'a> {
    /// The batch's own: its mean and biased variance, which training uses (and evaluation
    /// too when the layer keeps no running statistics).
    Batch,
    /// The running mean and variance, one value per channel each, which evaluation uses.
    Running {
        /// The running mean.
        mean: &'a DeviceBuffer<'a>,
        /// The running variance.
        var: &'a DeviceBuffer<'a>,
    },
}

/// What the backward pass and the running update must agree with: the last forward pass that
/// succeeded.
#[derive(Clone, Copy, Debug)]
struct LastForward {
    /// Whether it took the batch's statistics.
    batch: bool,
    /// Whether it was affine.
    affine: bool,
    /// The input it read, by its address on the card.
    x: DevicePtr,
    /// The statistics it wrote, by their address on the card.
    stats: DevicePtr,
}

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

/// `n` as a kernel's `float`, rounded as the CPU's `T::from_f64` rounds it for `f32`.
fn as_f32(n: f64) -> f32 {
    n as f32
}

/// The kernels of one module and their handles.
struct Kernels<'c> {
    /// The compiled sums and batch norm kernels.
    module: Module<'c>,
    /// The sums' first pass over the values themselves.
    sum_partials: CuFunction,
    /// The sums' first pass over the centred squares.
    centred_partials: CuFunction,
    /// The sums' first pass over `dy · x̂`.
    dy_xhat_partials: CuFunction,
    /// The sums' second pass.
    merge: CuFunction,
    /// The batch mean.
    stats_mean: CuFunction,
    /// `inv` and the unbiased variance from the centred squares.
    stats_finish: CuFunction,
    /// The statistics from running ones.
    stats_running: CuFunction,
    /// The normalisation.
    normalise: CuFunction,
    /// The input gradient.
    backward: CuFunction,
    /// The running statistics' update.
    running_update: CuFunction,
}

impl<'c> Kernels<'c> {
    /// Compiles the sums' and batch norm's source on `cuda`'s card and looks every kernel up.
    fn new(cuda: &'c Cuda) -> Result<Kernels<'c>, CudaError> {
        let module = cuda.compile("batch_norm.cu", &format!("{}{SOURCE}", reduce::SOURCE))?;
        let f = |name: &str| module.function(name);
        Ok(Kernels {
            sum_partials: f("sum_partials")?,
            centred_partials: f("centred_partials")?,
            dy_xhat_partials: f("dy_xhat_partials")?,
            merge: f("merge")?,
            stats_mean: f("stats_mean")?,
            stats_finish: f("stats_finish")?,
            stats_running: f("stats_running")?,
            normalise: f("normalise")?,
            backward: f("backward")?,
            running_update: f("running_update")?,
            module,
        })
    }

    /// Launches `function`, one of this module's handles, on a grid of `grid` blocks of
    /// [`BLOCK`] threads with `args`.
    ///
    /// # Safety
    ///
    /// `args` must be the kernel's parameters, and every buffer must hold what the kernel
    /// reads and writes on that grid.
    unsafe fn launch(
        &self,
        function: CuFunction,
        grid: [u32; 2],
        args: &[Arg<'_>],
    ) -> Result<(), CudaError> {
        // SAFETY: every handle was looked up in this module when it was compiled.
        let kernel = unsafe { self.module.kernel_from(function) };
        // SAFETY: the caller passes the kernel's parameters and buffers that hold what it
        // touches on this grid.
        unsafe { kernel.launch([grid[0], grid[1], 1], [BLOCK, 1, 1], args) }
    }

    /// Launches a per-channel kernel, one thread per channel.
    ///
    /// # Safety
    ///
    /// As [`Kernels::launch`], for `channels` threads.
    unsafe fn per_channel(
        &self,
        function: CuFunction,
        channels: u32,
        args: &[Arg<'_>],
    ) -> Result<(), CudaError> {
        // SAFETY: as the caller promises.
        unsafe { self.launch(function, [channels.div_ceil(BLOCK), 1], args) }
    }

    /// Sums each channel's terms of a tensor shaped `dims` into `out`: the first pass
    /// `function` from `inputs` into `partials`, then the merge.
    ///
    /// # Safety
    ///
    /// `function` must be one of the three first passes and `inputs` its inputs before the
    /// partial sums, each holding what the term reads for `dims`; `dims` must have passed
    /// `ChannelDims::check`; `partials` must hold `dims.chunks() · channels` floats and `out`
    /// `channels`, and neither may be an input.
    unsafe fn sums(
        &self,
        function: CuFunction,
        inputs: &[Arg<'_>],
        dims: ChannelDims,
        partials: &DeviceBuffer<'_>,
        out: &DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let chunks = dims.chunks();
        let mut args = [Arg::U32(0); SUM_ARGS];
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
        // channels, u32 inner, u32 chunks); its grid of chunks by channels reads values below
        // dims.len() and writes partial sums below chunks · channels, which the caller
        // promises the buffers hold.
        unsafe {
            self.launch(function, [chunks, dims.channels], &args[..n + 5])?;
        }
        // SAFETY: the merge takes (const float *partials, float *sums, u32 channels, u32
        // chunks), reads chunks · channels partial sums and writes channels sums.
        unsafe {
            self.per_channel(
                self.merge,
                dims.channels,
                &[
                    Arg::Buffer(partials),
                    Arg::Buffer(out),
                    Arg::U32(dims.channels),
                    Arg::U32(chunks),
                ],
            )
        }
    }
}

/// 2-D batch normalisation on the card for one shape, over NCHW tensors
/// (`ChannelDims::new(batch, C, h · w)`), as `cavitation_core`'s `Tape::batch_norm2d`
/// computes it: [`forward`](BatchNorm2d::forward) from the batch's statistics or from
/// running ones, [`backward`](BatchNorm2d::backward) through the same statistics, and
/// [`update_running`](BatchNorm2d::update_running) after a forward pass from the batch.
///
/// The layer remembers its last forward pass that succeeded: whether it took the batch's
/// statistics, whether it was affine, and which `x` and `stats` buffers it used. A backward
/// pass or a running update that does not match it, or that comes before any, is refused with
/// [`CudaError::Invalid`] rather than computed with the wrong formula or statistics.
///
/// The statistics buffer holds `3 · C` values, `[mean | inv | unbiased]`, as the CPU's stats
/// node holds them: the mean, `1 / √(var + eps)` and the unbiased variance (0 for running
/// statistics).
///
/// The kernels are `cavitation-gpu`'s WGSL ones ported operation for operation, with two
/// changes: `inv` is the CPU's `1 / √v` rather than an inverse square root, and the
/// normalisation without `γ`, `β` multiplies by 1 and adds 0 as the CPU does. NVRTC keeps
/// subnormals and divides and takes square roots correctly rounded, and nothing is fused, so:
///
/// - **Sums** (`Σx`, `Σ (x − mean)²`, `Σ dy`, `Σ dy·x̂`): a fixed tree that depends on the
///   shape alone. The first pass sums each chunk of 4,096 of a channel's values (in the
///   order `b·inner + p`) in one block, each of 256 threads adding every 256th value in
///   order from +0, then the threads in a halving tree; the second pass adds a channel's
///   chunks in order. It is not the CPU's order, so each sum of `L = outer · inner` terms is
///   within `2(L + 2) · 2⁻²⁴ · Σ|term|` of the exact sum of the same `f32` terms; on terms
///   whose partial sums are exact it is the CPU's sum bit for bit.
/// - **Everything else** (the statistics from the sums, the normalisation, the input
///   gradient, the running update): the CPU's own `f32` expression of the same inputs, bit
///   for bit, NaN aside (CUDA's arithmetic makes its own NaN, so NaN appears where the CPU's
///   does, without its bits).
///
/// The statistics are taken per call; [`sums`](BatchNorm2d::sums) gives the batch sums of
/// the last forward pass from the batch. Two runs on the same inputs give the same bits.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{BatchNorm2d, BatchStats, ChannelDims, Cuda};
///
/// if let Some(cuda) = Cuda::new() {
///     // Two images of one channel of two values: the mean is 2.5, the variance 1.25.
///     let mut bn = BatchNorm2d::new(&cuda, ChannelDims::new(2, 1, 2)).unwrap();
///     let mut x = cuda.buffer(16).unwrap();
///     let (mut y, mut stats) = (cuda.buffer(16).unwrap(), cuda.buffer(12).unwrap());
///     x.write(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
///     bn.forward(0.0, BatchStats::Batch, &x, None, &mut y, &mut stats).unwrap();
///     let mut out = [0.0f32; 4];
///     y.read(&mut out).unwrap();
///     let want = [-1.5f32, -0.5, 0.5, 1.5].map(|v| v * (1.0 / 1.25f32.sqrt()));
///     assert_eq!(out, want);
/// }
/// ```
pub struct BatchNorm2d<'c> {
    /// The shape.
    dims: ChannelDims,
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// One partial sum per chunk of each channel, for whichever sum runs.
    partials: DeviceBuffer<'c>,
    /// `Σx` of the last forward pass from the batch, one value per channel.
    sum_x: DeviceBuffer<'c>,
    /// `Σ (x − mean)²` of the last forward pass from the batch, one value per channel.
    squares: DeviceBuffer<'c>,
    /// `Σ dy` of a backward pass without `dβ` to hold it.
    sum_dy: DeviceBuffer<'c>,
    /// `Σ dy · x̂` of a backward pass without `dγ` to hold it.
    sum_dy_xhat: DeviceBuffer<'c>,
    /// The last forward pass that succeeded, which the backward pass and the running update
    /// are checked against; none before the first and after one that failed.
    last: Option<LastForward>,
}

impl fmt::Debug for BatchNorm2d<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

impl<'c> BatchNorm2d<'c> {
    /// Compiles the kernels on `cuda`'s card for tensors shaped `dims`, and allocates the
    /// sums' scratch memory: one float per chunk of 4,096 values of each channel, and four
    /// per channel.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] for a dimension of 0, more than 2³¹ values or more than 65,535
    /// channels; [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses
    /// the kernels or the memory.
    pub fn new(cuda: &'c Cuda, dims: ChannelDims) -> Result<BatchNorm2d<'c>, CudaError> {
        dims.check("batch_norm2d")?;
        let kernels = Kernels::new(cuda)?;
        let c = dims.channels as usize * 4;
        Ok(BatchNorm2d {
            dims,
            kernels,
            partials: cuda.buffer(dims.chunks() as usize * c)?,
            sum_x: cuda.buffer(c)?,
            squares: cuda.buffer(c)?,
            sum_dy: cuda.buffer(c)?,
            sum_dy_xhat: cuda.buffer(c)?,
            last: None,
        })
    }

    /// The shape.
    pub fn dims(&self) -> ChannelDims {
        self.dims
    }

    /// The batch sums of the last forward pass from the batch's statistics,
    /// `(Σx, Σ (x − mean)²)`, one value per channel each (zeros before the first).
    pub fn sums(&self) -> (&DeviceBuffer<'c>, &DeviceBuffer<'c>) {
        (&self.sum_x, &self.squares)
    }

    /// Normalises `x` into `y`, with its statistics written to `stats` (`3 · C` values)
    /// taken `from` the batch or the running statistics, with `eps`, and `affine` `(γ, β)`
    /// if the layer has them. Every value of `y` and `stats` is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if eps is not finite and non-negative, or if batch statistics
    /// would be taken over one value per channel (the CPU refuses that too);
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values: `x` and `y` hold the tensor, `stats` `3 · C`
    /// values, and `γ`, `β` and the running statistics `C` each.
    pub fn forward(
        &mut self,
        eps: f64,
        from: BatchStats<'_>,
        x: &DeviceBuffer<'_>,
        affine: Option<(&DeviceBuffer<'_>, &DeviceBuffer<'_>)>,
        y: &mut DeviceBuffer<'_>,
        stats: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        const WHAT: &str = "batch_norm2d forward";
        let dims = self.dims;
        self.last = None;
        if !(eps.is_finite() && eps >= 0.0) {
            return Err(CudaError::Invalid(format!(
                "{WHAT}: eps must be finite and non-negative, got {eps}"
            )));
        }
        let (len, c) = (dims.len(), u64::from(dims.channels));
        check_buffers(
            WHAT,
            &[
                (x, "input", len),
                (y, "output", len),
                (stats, "statistics", 3 * c),
            ],
        );
        if let Some((w, b)) = affine {
            check_buffers(WHAT, &[(w, "weight", c), (b, "bias", c)]);
        }
        let k = &self.kernels;
        let m = dims.summed() as f64;
        let eps = as_f32(eps);
        match from {
            BatchStats::Running { mean, var } => {
                check_buffers(
                    WHAT,
                    &[(mean, "running mean", c), (var, "running variance", c)],
                );
                // SAFETY: stats_running takes (const float *mean, const float *var, float
                // *stats, u32 channels, float eps), reads C values of each input and writes
                // 3 · C statistics, which the buffers hold (checked above); stats, borrowed
                // mutably, is neither input.
                unsafe {
                    k.per_channel(
                        k.stats_running,
                        dims.channels,
                        &[
                            Arg::Buffer(mean),
                            Arg::Buffer(var),
                            Arg::Buffer(stats),
                            Arg::U32(dims.channels),
                            Arg::F32(eps),
                        ],
                    )?;
                }
            }
            BatchStats::Batch => {
                if dims.summed() < 2 {
                    return Err(CudaError::Invalid(format!(
                        "{WHAT}: batch statistics need more than one value per channel, got \
                         {dims:?}"
                    )));
                }
                // SAFETY: sum_partials reads the tensor from x, which holds it; the shape
                // was checked when the layer was made, and the partial sums and Σx are this
                // layer's own buffers, sized for it and distinct from x.
                unsafe {
                    k.sums(
                        k.sum_partials,
                        &[Arg::Buffer(x)],
                        dims,
                        &self.partials,
                        &self.sum_x,
                    )?;
                }
                // SAFETY: stats_mean takes (const float *sums, float *stats, u32 channels,
                // float m), reads C sums and writes C means, which the buffers hold.
                unsafe {
                    k.per_channel(
                        k.stats_mean,
                        dims.channels,
                        &[
                            Arg::Buffer(&self.sum_x),
                            Arg::Buffer(stats),
                            Arg::U32(dims.channels),
                            Arg::F32(as_f32(m)),
                        ],
                    )?;
                }
                // SAFETY: centred_partials reads the tensor from x and C means from stats,
                // which hold them; the partial sums and the squares are this layer's own.
                unsafe {
                    k.sums(
                        k.centred_partials,
                        &[Arg::Buffer(x), Arg::Buffer(stats)],
                        dims,
                        &self.partials,
                        &self.squares,
                    )?;
                }
                // SAFETY: stats_finish takes (const float *squares, float *stats, u32
                // channels, float m, float m_minus_1, float eps), reads C sums and writes
                // statistics below 3 · C, which the buffers hold.
                unsafe {
                    k.per_channel(
                        k.stats_finish,
                        dims.channels,
                        &[
                            Arg::Buffer(&self.squares),
                            Arg::Buffer(stats),
                            Arg::U32(dims.channels),
                            Arg::F32(as_f32(m)),
                            Arg::F32(as_f32(m - 1.0)),
                            Arg::F32(eps),
                        ],
                    )?;
                }
            }
        }
        let (w, b) = affine.unwrap_or((x, x));
        // The length fits a u32: the shape was checked when the layer was made.
        let total = len as u32;
        // SAFETY: normalise takes (const float *x, const float *stats, const float *weight,
        // const float *bias, float *y, u32 total, u32 channels, u32 inner, u32 affine), reads
        // `total` values of x, 2 · C statistics and, when affine, C values of γ and β, and
        // writes `total` values of y, which the buffers hold (checked above); y, borrowed
        // mutably, is none of the inputs.
        let launched = unsafe {
            k.launch(
                k.normalise,
                [total.div_ceil(BLOCK), 1],
                &[
                    Arg::Buffer(x),
                    Arg::Buffer(stats),
                    Arg::Buffer(w),
                    Arg::Buffer(b),
                    Arg::Buffer(y),
                    Arg::U32(total),
                    Arg::U32(dims.channels),
                    Arg::U32(dims.inner),
                    Arg::U32(u32::from(affine.is_some())),
                ],
            )
        };
        if launched.is_ok() {
            self.last = Some(LastForward {
                batch: matches!(from, BatchStats::Batch),
                affine: affine.is_some(),
                x: x.pointer,
                stats: stats.pointer,
            });
        }
        launched
    }

    /// The last forward pass, which a call named `what` must follow: refused before any.
    fn last(&self, what: &str) -> Result<LastForward, CudaError> {
        self.last.ok_or_else(|| {
            CudaError::Invalid(format!(
                "{what}: no forward pass has run on this layer, or the last one failed"
            ))
        })
    }

    /// The backward pass of the last forward pass, through the statistics it took (the
    /// batch's or the running ones), from the same input `x` and statistics `stats`: the
    /// input gradient `grad_in` from the output gradient `grad_out`, and, for an affine
    /// layer, `(γ, dγ, dβ)`, the weight and the two gradients `Σ dy·x̂` and `Σ dy`. Every
    /// gradient is overwritten.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] before any forward pass (or after one that failed), if
    /// `affine` is given for a forward pass without `γ`, `β` or the other way round, or if
    /// `x` or `stats` is not the forward pass's own buffer; [`CudaError::Call`] if a launch
    /// is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values: `x`, `grad_out` and `grad_in` hold the
    /// tensor, `stats` `3 · C` values, and `γ`, `dγ` and `dβ` `C` each.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{BatchNorm2d, BatchStats, ChannelDims, Cuda};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut bn = BatchNorm2d::new(&cuda, ChannelDims::new(2, 1, 2)).unwrap();
    ///     let buffer = |data: &[f32]| {
    ///         let mut b = cuda.buffer(4 * data.len()).unwrap();
    ///         b.write(data).unwrap();
    ///         b
    ///     };
    ///     let (x, dy) = (buffer(&[1.0, 2.0, 3.0, 4.0]), buffer(&[1.0; 4]));
    ///     let (w, b, mut dw, mut db) = (buffer(&[1.0]), buffer(&[0.0]), buffer(&[0.0]), buffer(&[0.0]));
    ///     let (mut y, mut dx, mut stats) = (buffer(&[0.0; 4]), buffer(&[0.0; 4]), buffer(&[0.0; 3]));
    ///     bn.forward(0.0, BatchStats::Batch, &x, Some((&w, &b)), &mut y, &mut stats).unwrap();
    ///     bn.backward(&x, &stats, &dy, &mut dx, Some((&w, &mut dw, &mut db))).unwrap();
    ///     let (mut grad, mut bias_grad) = ([1.0f32; 4], [0.0f32]);
    ///     dx.read(&mut grad).unwrap();
    ///     db.read(&mut bias_grad).unwrap();
    ///     // A constant output gradient moves no input: the mean takes it all.
    ///     assert_eq!((grad, bias_grad), ([0.0; 4], [4.0]));
    /// }
    /// ```
    pub fn backward(
        &mut self,
        x: &DeviceBuffer<'_>,
        stats: &DeviceBuffer<'_>,
        grad_out: &DeviceBuffer<'_>,
        grad_in: &mut DeviceBuffer<'_>,
        affine: Option<(
            &DeviceBuffer<'_>,
            &mut DeviceBuffer<'_>,
            &mut DeviceBuffer<'_>,
        )>,
    ) -> Result<(), CudaError> {
        const WHAT: &str = "batch_norm2d backward";
        let dims = self.dims;
        let (len, c) = (dims.len(), u64::from(dims.channels));
        check_buffers(
            WHAT,
            &[
                (x, "input", len),
                (stats, "statistics", 3 * c),
                (grad_out, "output gradient", len),
                (grad_in, "input gradient", len),
            ],
        );
        if let Some((w, dw, db)) = &affine {
            check_buffers(
                WHAT,
                &[
                    (w, "weight", c),
                    (dw, "weight gradient", c),
                    (db, "bias gradient", c),
                ],
            );
        }
        let last = self.last(WHAT)?;
        if affine.is_some() != last.affine {
            return Err(CudaError::Invalid(format!(
                "{WHAT}: γ, dγ and dβ must be given exactly when the forward pass was affine \
                 (it was {})",
                if last.affine { "affine" } else { "not affine" }
            )));
        }
        if x.pointer != last.x || stats.pointer != last.stats {
            return Err(CudaError::Invalid(format!(
                "{WHAT}: the backward pass must read the forward pass's own input and \
                 statistics buffers"
            )));
        }
        let batch = last.batch;
        let k = &self.kernels;
        let (w, targets) = match &affine {
            Some((w, dw, db)) => (*w, Some((&**db, &**dw))),
            None if batch => (grad_out, Some((&self.sum_dy, &self.sum_dy_xhat))),
            None => (grad_out, None),
        };
        if let Some((sum_dy, sum_dy_xhat)) = targets {
            // SAFETY: sum_partials reads the tensor from grad_out, which holds it; the
            // partial sums are this layer's own, and the sums go to dβ (borrowed mutably, so
            // no input) or to this layer's own scratch, C values each.
            unsafe {
                k.sums(
                    k.sum_partials,
                    &[Arg::Buffer(grad_out)],
                    dims,
                    &self.partials,
                    sum_dy,
                )?;
            }
            // SAFETY: dy_xhat_partials reads the tensor from grad_out and x and 2 · C
            // statistics, which the buffers hold; the sums go to dγ (borrowed mutably) or to
            // this layer's own scratch.
            unsafe {
                k.sums(
                    k.dy_xhat_partials,
                    &[Arg::Buffer(grad_out), Arg::Buffer(x), Arg::Buffer(stats)],
                    dims,
                    &self.partials,
                    sum_dy_xhat,
                )?;
            }
        }
        let (sum_dy, sum_dy_xhat) = targets.unwrap_or((grad_out, grad_out));
        let total = len as u32;
        // SAFETY: backward takes (const float *grad_out, const float *x, const float *stats,
        // const float *weight, const float *sum_dy, const float *sum_dy_xhat, float *grad_in,
        // u32 total, u32 channels, u32 inner, u32 affine, u32 batch, float m), reads `total`
        // values of grad_out and x, 2 · C statistics and C values of γ when affine and of
        // the sums when through the batch, and writes `total` values of grad_in, which the
        // buffers hold (checked above); grad_in, borrowed mutably, is none of the inputs.
        unsafe {
            k.launch(
                k.backward,
                [total.div_ceil(BLOCK), 1],
                &[
                    Arg::Buffer(grad_out),
                    Arg::Buffer(x),
                    Arg::Buffer(stats),
                    Arg::Buffer(w),
                    Arg::Buffer(sum_dy),
                    Arg::Buffer(sum_dy_xhat),
                    Arg::Buffer(grad_in),
                    Arg::U32(total),
                    Arg::U32(dims.channels),
                    Arg::U32(dims.inner),
                    Arg::U32(u32::from(affine.is_some())),
                    Arg::U32(u32::from(batch)),
                    Arg::F32(as_f32(dims.summed() as f64)),
                ],
            )
        }
    }

    /// Moves the running `(mean, var)` towards the batch mean and unbiased variance in
    /// `stats`, written by the last forward pass, which must have taken the batch's
    /// statistics, with the rates
    /// `update` gives ([`RunningUpdate::rates`]): `r ← momentum · b + (1 − momentum) · r`,
    /// the CPU's `Graph::apply_buffer_updates` on `f32`. Nothing changes when the `u32` at
    /// the start of `finite` is non-zero (the step's finite flag: 0 when every value was
    /// finite). Counting the batch is the caller's, with [`RunningUpdate::commit`] once the
    /// flag is known.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] before any forward pass (or after one that failed), after one
    /// from running statistics (whose unbiased variance slot is 0), or if `stats` is not its
    /// own buffer; [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If `stats` holds fewer than `3 · C` values, a running statistic fewer than `C`, or
    /// `finite` fewer than one.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{BatchNorm2d, BatchStats, ChannelDims, Cuda, RunningUpdate};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut bn = BatchNorm2d::new(&cuda, ChannelDims::new(2, 1, 2)).unwrap();
    ///     let buffer = |data: &[f32]| {
    ///         let mut b = cuda.buffer(4 * data.len()).unwrap();
    ///         b.write(data).unwrap();
    ///         b
    ///     };
    ///     let (x, mut y, mut stats) = (buffer(&[1.0, 2.0, 3.0, 4.0]), buffer(&[0.0; 4]), buffer(&[0.0; 3]));
    ///     let (mut mean, mut var, finite) = (buffer(&[0.0]), buffer(&[1.0]), buffer(&[0.0]));
    ///     bn.forward(0.0, BatchStats::Batch, &x, None, &mut y, &mut stats).unwrap();
    ///     let mut update = RunningUpdate::new(None, Some(0.0)); // the first batch: momentum 1
    ///     bn.update_running(&update, &stats, (&mut mean, &mut var), &finite).unwrap();
    ///     update.commit(true);
    ///     assert_eq!(update.count(), Some(1.0));
    ///     let (mut m, mut v) = ([0.0f32], [0.0f32]);
    ///     mean.read(&mut m).unwrap();
    ///     var.read(&mut v).unwrap();
    ///     assert_eq!((m, v), ([2.5], [5.0 / 3.0]));
    /// }
    /// ```
    pub fn update_running(
        &self,
        update: &RunningUpdate,
        stats: &DeviceBuffer<'_>,
        (mean, var): (&mut DeviceBuffer<'_>, &mut DeviceBuffer<'_>),
        finite: &DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        const WHAT: &str = "batch_norm2d running update";
        let c = u64::from(self.dims.channels);
        check_buffers(
            WHAT,
            &[
                (stats, "statistics", 3 * c),
                (mean, "running mean", c),
                (var, "running variance", c),
                (finite, "finite flag", 1),
            ],
        );
        let last = self.last(WHAT)?;
        if !last.batch {
            return Err(CudaError::Invalid(format!(
                "{WHAT}: the last forward pass took running statistics, and the update needs \
                 one from the batch's"
            )));
        }
        if stats.pointer != last.stats {
            return Err(CudaError::Invalid(format!(
                "{WHAT}: the update must read the forward pass's own statistics buffer"
            )));
        }
        let (momentum, keep) = update.rates();
        let k = &self.kernels;
        // SAFETY: running_update takes (const float *stats, const u32 *finite, float
        // *running_mean, float *running_var, u32 channels, float momentum, float keep), reads
        // statistics below 3 · C and one flag and updates C values of each running
        // statistic, which the buffers hold (checked above); the two, borrowed mutably, are
        // distinct from each other and from the inputs.
        unsafe {
            k.per_channel(
                k.running_update,
                self.dims.channels,
                &[
                    Arg::Buffer(stats),
                    Arg::Buffer(finite),
                    Arg::Buffer(mean),
                    Arg::Buffer(var),
                    Arg::U32(self.dims.channels),
                    Arg::F32(momentum),
                    Arg::F32(keep),
                ],
            )
        }
    }
}

/// How a layer's running statistics weight each batch, with the batch count kept on the
/// host: a momentum, or the cumulative average over the batches counted so far.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::RunningUpdate;
///
/// // A cumulative average (no momentum), 3 batches counted so far.
/// let mut update = RunningUpdate::new(None, Some(3.0));
/// assert_eq!(update.rates(), (0.25, 0.75)); // the next batch is the 4th
/// update.commit(false); // a skipped step is not counted
/// assert_eq!(update.count(), Some(3.0));
/// update.set_count(7.0);
/// assert_eq!(update.rates(), (0.125, 0.875));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningUpdate {
    /// Each batch's weight, or `None` for the cumulative average.
    momentum: Option<f64>,
    /// The batches counted so far, if the layer keeps a count.
    count: Option<f32>,
}

impl RunningUpdate {
    /// Weights each batch by `momentum`, or with `None` by one over the batches counted
    /// after it, from `count`, the `num_batches_tracked` the layer keeps (if it keeps one),
    /// as `cavitation_core::Graph::apply_buffer_updates` takes them.
    pub fn new(momentum: Option<f64>, count: Option<f32>) -> RunningUpdate {
        RunningUpdate { momentum, count }
    }

    /// The `f32` weights the next update gives the batch and the running value,
    /// `(momentum, 1 − momentum)`, rounded as `Graph::apply_buffer_updates` rounds them on
    /// `f32`: the momentum itself, else one over the count after this batch, else 1.
    pub fn rates(&self) -> (f32, f32) {
        let momentum = match (self.momentum, self.count) {
            (Some(m), _) => m as f32,
            (None, Some(n)) => 1.0 / (n + 1.0),
            (None, None) => 1.0,
        };
        (momentum, 1.0 - momentum)
    }

    /// Counts the batch of the last update if its step `ran` (its finite flag was clear), as
    /// the CPU counts a step it does not skip.
    pub fn commit(&mut self, ran: bool) {
        if ran && let Some(n) = &mut self.count {
            *n += 1.0;
        }
    }

    /// The batches counted so far, if the layer keeps a count.
    pub fn count(&self) -> Option<f32> {
        self.count
    }

    /// Sets the batches counted so far to `count`, for a layer that keeps a count; a layer
    /// without one is left as it is.
    pub fn set_count(&mut self, count: f32) {
        if let Some(n) = &mut self.count {
            *n = count;
        }
    }
}
