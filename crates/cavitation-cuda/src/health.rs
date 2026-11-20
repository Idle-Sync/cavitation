//! The health values a training step reads back, as this crate's own CUDA kernels, the rules
//! of the CPU health checks:
//!
//! - [`SumSquares`]: each parameter gradient's sum of squares, which the gradient-norm checks
//!   read.
//! - [`FiniteFlag`]: the step's finite flag, set when the loss or any gradient value is NaN or
//!   infinite, which [`OptimizerKernel`](crate::OptimizerKernel) and
//!   [`BatchNorm2d::update_running`](crate::BatchNorm2d::update_running) read to skip their
//!   writes.
//! - [`DeadRelu`]: one "ever non-zero" bit per ReLU neuron (per channel after a convolution),
//!   which the dead-ReLU check reads.
//!
//! NaN and infinity are tested by their bits. Every sum and count runs in the channel sums'
//! fixed tree (`reduce.cu`), so two runs give the same bits.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::reduce::{self, BLOCK, ChannelDims};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/health.rs`, compiled after the
/// sums' source.
const SOURCE: &str = include_str!("health.cu");

/// The most values the tensors of a flat buffer may hold together: the kernels index them
/// with `u32`s, with room for a block's threads past the end.
const MAX_VALUES: u64 = 1 << 31;

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

/// Each tensor's `(start, len)` in a flat buffer that holds tensors of `lens` values one
/// after another; refuses, naming `what`, a tensor of no values and more than 2³¹ values
/// together.
fn flat(what: &str, lens: &[usize]) -> Result<Vec<(u32, u32)>, CudaError> {
    let mut start = 0u64;
    let mut tensors = Vec::with_capacity(lens.len());
    for (k, &len) in lens.iter().enumerate() {
        if len == 0 {
            return Err(CudaError::Invalid(format!(
                "{what}: tensor {k} holds no values; every tensor must hold at least one"
            )));
        }
        let end = start.saturating_add(len as u64);
        if end > MAX_VALUES {
            return Err(CudaError::Invalid(format!(
                "{what}: the tensors hold more than the {MAX_VALUES} values the kernels take"
            )));
        }
        tensors.push((start as u32, len as u32));
        start = end;
    }
    Ok(tensors)
}

/// The values the tensors hold together.
fn total(tensors: &[(u32, u32)]) -> u64 {
    tensors
        .last()
        .map_or(0, |&(s, l)| u64::from(s) + u64::from(l))
}

/// Partial sums a buffer needs for the longest of `tensors`, at least one.
fn most_chunks(tensors: &[(u32, u32)]) -> usize {
    tensors
        .iter()
        .map(|&(_, len)| ChannelDims::new(1, 1, len).chunks() as usize)
        .max()
        .unwrap_or(0)
        .max(1)
}

/// The kernels of one module and their handles.
struct Kernels<'c> {
    /// The compiled sums and health kernels.
    module: Module<'c>,
    /// The first pass of a sum of squares.
    squares_partials: CuFunction,
    /// The first pass of a count of non-finite values.
    non_finite_partials: CuFunction,
    /// The first pass of a count of non-zero values per channel.
    non_zero_partials: CuFunction,
    /// The sums' second pass, one thread per channel.
    merge: CuFunction,
    /// The second pass of one channel's sum into a word of the output.
    merge_at: CuFunction,
    /// The finite flag from the counts and the loss.
    finite_fold: CuFunction,
    /// The dead-ReLU bits from the counts.
    dead_relu_pack: CuFunction,
}

impl<'c> Kernels<'c> {
    /// Compiles the sums' and the health values' source on `cuda`'s card and looks every
    /// kernel up.
    fn new(cuda: &'c Cuda) -> Result<Kernels<'c>, CudaError> {
        let module = cuda.compile("health.cu", &format!("{}{SOURCE}", reduce::SOURCE))?;
        let f = |name: &str| module.function(name);
        Ok(Kernels {
            squares_partials: f("squares_partials")?,
            non_finite_partials: f("non_finite_partials")?,
            non_zero_partials: f("non_zero_partials")?,
            merge: f("merge")?,
            merge_at: f("merge_at")?,
            finite_fold: f("finite_fold")?,
            dead_relu_pack: f("dead_relu_pack")?,
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

    /// Sums the terms of the tensor of `len` values that starts `start` values into `input`
    /// into word `at` of `out`: the first pass `function` into `partials`, then the merge in
    /// chunk order.
    ///
    /// # Safety
    ///
    /// `function` must be `squares_partials` or `non_finite_partials`; `input` must hold
    /// `start + len` values, at most 2³¹, with `len` above 0; `partials` must hold the
    /// tensor's chunks and `out` `at + 1` floats, and neither may be `input`.
    unsafe fn tensor_sum(
        &self,
        function: CuFunction,
        (input, start, len): (&DeviceBuffer<'_>, u32, u32),
        partials: &DeviceBuffer<'_>,
        (out, at): (&DeviceBuffer<'_>, u32),
    ) -> Result<(), CudaError> {
        let dims = ChannelDims::new(1, 1, len);
        let chunks = dims.chunks();
        // SAFETY: the first pass takes (const float or u32 *input, u32 start, float
        // *partials, u32 outer, u32 channels, u32 inner, u32 chunks); its grid of chunks
        // (one channel) reads values start..start + len and writes `chunks` partial sums,
        // which the caller promises the buffers hold.
        unsafe {
            self.launch(
                function,
                [chunks, 1],
                &[
                    Arg::Buffer(input),
                    Arg::U32(start),
                    Arg::Buffer(partials),
                    Arg::U32(dims.outer),
                    Arg::U32(dims.channels),
                    Arg::U32(dims.inner),
                    Arg::U32(chunks),
                ],
            )?;
        }
        // SAFETY: merge_at takes (const float *partials, float *out, u32 at, u32 chunks),
        // reads `chunks` partial sums and writes out[at], which the caller promises the
        // buffers hold; one thread does the work.
        unsafe {
            self.launch(
                self.merge_at,
                [1, 1],
                &[
                    Arg::Buffer(partials),
                    Arg::Buffer(out),
                    Arg::U32(at),
                    Arg::U32(chunks),
                ],
            )
        }
    }
}

/// The sum of squares of each parameter gradient on the card: one `f32` per tensor, `Σ g²`
/// over its values in the channel sums' fixed tree, which the gradient-norm checks read as
/// they read the CPU's `GradStats`.
///
/// The gradients lie one after another in one flat buffer, in the order of the `lens` the
/// kernel was made with, as [`OptimizerKernel`](crate::OptimizerKernel) takes them, and tensor
/// `k`'s sum goes to word `offset + k` of the output.
///
/// The tree is `cavitation-gpu`'s: chunks of 4,096 values, each of 256 threads adding every
/// 256th square of its chunk in order from +0, the threads meeting in a halving tree, and the
/// chunks added in order. NVRTC keeps subnormals and fuses nothing, so each sum is the bits
/// of that tree evaluated on the host; on integers whose sums stay below 2²⁴ it is the CPU's
/// `f32` sum bit for bit; against the exact sum of the same `f32` values a tensor of `L`
/// values is within the WGSL kernel's bound, `(2(L + 2) + 1) · 2⁻²⁴ · Σ g² + (3L − 1) ·
/// 2⁻¹²⁶`. The CPU sums in `f64`: in `f32` a sum overflows to infinity once it passes
/// `f32::MAX` (a gradient value near 1.8 · 10¹⁹), and a NaN or infinite gradient value gives a
/// NaN or infinite sum. The finite flag tests the values themselves, so an overflowed sum
/// leaves it clear.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, SumSquares};
///
/// if let Some(cuda) = Cuda::new() {
///     // Two gradients in one flat buffer, [3, 4] and [1, 2, 2]; their sums go to words 1 and 2.
///     let sums = SumSquares::new(&cuda, &[2, 3]).unwrap();
///     let (mut grads, mut out) = (cuda.buffer(20).unwrap(), cuda.buffer(12).unwrap());
///     grads.write(&[3.0f32, 4.0, 1.0, 2.0, 2.0]).unwrap();
///     sums.run(&grads, (&mut out, 1)).unwrap();
///     let mut got = [0.0f32; 3];
///     out.read(&mut got).unwrap();
///     assert_eq!(got, [0.0, 25.0, 9.0]);
/// }
/// ```
pub struct SumSquares<'c> {
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// Each tensor's start and length in the flat buffer.
    tensors: Vec<(u32, u32)>,
    /// One partial sum per chunk of the longest tensor, reused tensor by tensor.
    partials: DeviceBuffer<'c>,
}

impl fmt::Debug for SumSquares<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SumSquares")
            .field("tensors", &self.tensors)
            .finish_non_exhaustive()
    }
}

impl<'c> SumSquares<'c> {
    /// Compiles the kernels on `cuda`'s card for gradients of `lens` values each, one after
    /// another in a flat buffer, and allocates the partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `lens` is empty or holds a 0, or the tensors hold more than
    /// 2³¹ values together; [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the
    /// driver refuses the kernels or the memory.
    pub fn new(cuda: &'c Cuda, lens: &[usize]) -> Result<SumSquares<'c>, CudaError> {
        const WHAT: &str = "sum of squares";
        if lens.is_empty() {
            return Err(CudaError::Invalid(format!(
                "{WHAT}: there must be at least one gradient"
            )));
        }
        let tensors = flat(WHAT, lens)?;
        let kernels = Kernels::new(cuda)?;
        Ok(SumSquares {
            partials: cuda.buffer(4 * most_chunks(&tensors))?,
            kernels,
            tensors,
        })
    }

    /// The number of gradients, and so of sums each run writes.
    pub fn tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Writes the sum of squares of each gradient in `grads` to words `offset` onwards of
    /// `out`, one per gradient in order. Nothing else of `out` changes.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If `grads` holds fewer values than the gradients together, or `out` fewer than
    /// `offset` plus one per gradient.
    pub fn run(
        &self,
        grads: &DeviceBuffer<'_>,
        (out, offset): (&mut DeviceBuffer<'_>, u32),
    ) -> Result<(), CudaError> {
        const WHAT: &str = "sum of squares";
        let words = u64::from(offset) + self.tensors.len() as u64;
        check_buffers(
            WHAT,
            &[
                (grads, "gradients", total(&self.tensors)),
                (out, "output", words),
            ],
        );
        let k = &self.kernels;
        for (at, &(start, len)) in (offset..).zip(&self.tensors) {
            // SAFETY: the gradients hold start + len values, at most 2³¹ (checked when the
            // kernel was made and above), and len is above 0; the partial sums are this
            // kernel's own, sized for the longest tensor; out, borrowed mutably and so no
            // input, holds word `at`, below `offset` plus the tensors (checked above).
            unsafe {
                k.tensor_sum(
                    k.squares_partials,
                    (grads, start, len),
                    &self.partials,
                    (out, at),
                )?;
            }
        }
        Ok(())
    }
}

/// The step's finite flag on the card: a `u32` at word 0 of its own buffer, 0 when the loss
/// and every value of every gradient are finite and 1 when any is NaN or infinite, the rule
/// of the CPU's `OnNonFinite::Skip`. [`OptimizerKernel::update`](crate::OptimizerKernel::update)
/// and [`BatchNorm2d::update_running`](crate::BatchNorm2d::update_running) read that word of
/// the same buffer and skip their writes when it is not 0.
///
/// The values are tested by their exponent bits, so the flag is exact: a subnormal or the
/// largest finite value is finite, and a gradient large enough to overflow its sum of squares
/// does not set it. Each gradient's non-finite values are counted in the channel sums' fixed
/// tree (a count is exact while below 2²⁴, and never 0 once a value is counted), then one
/// thread folds the counts and the loss's bits into the flag. The gradients lie one after
/// another in one flat buffer, as for [`SumSquares`]; a flag of the loss alone takes no
/// gradients.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, FiniteFlag};
///
/// if let Some(cuda) = Cuda::new() {
///     let check = FiniteFlag::new(&cuda, &[3]).unwrap();
///     let (mut loss, mut grads) = (cuda.buffer(4).unwrap(), cuda.buffer(12).unwrap());
///     let mut flag = cuda.buffer(4).unwrap();
///     loss.write(&[0.5f32]).unwrap();
///     grads.write(&[1.0f32, f32::INFINITY, 2.0]).unwrap();
///     check.run((&loss, 0), &grads, &mut flag).unwrap();
///     let mut bad = [0u32];
///     flag.read(&mut bad).unwrap();
///     assert_eq!(bad, [1]);
/// }
/// ```
pub struct FiniteFlag<'c> {
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// Each tensor's start and length in the flat buffer.
    tensors: Vec<(u32, u32)>,
    /// One partial sum per chunk of the longest tensor, reused tensor by tensor.
    partials: DeviceBuffer<'c>,
    /// Each tensor's count of non-finite values (one float when there are none).
    counts: DeviceBuffer<'c>,
}

impl fmt::Debug for FiniteFlag<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FiniteFlag")
            .field("tensors", &self.tensors)
            .finish_non_exhaustive()
    }
}

impl<'c> FiniteFlag<'c> {
    /// Compiles the kernels on `cuda`'s card for gradients of `lens` values each, one after
    /// another in a flat buffer (none for a flag of the loss alone), and allocates the counts
    /// and their partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `lens` holds a 0, or the tensors hold more than 2³¹ values
    /// together; [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver
    /// refuses the kernels or the memory.
    pub fn new(cuda: &'c Cuda, lens: &[usize]) -> Result<FiniteFlag<'c>, CudaError> {
        let tensors = flat("finite flag", lens)?;
        let kernels = Kernels::new(cuda)?;
        Ok(FiniteFlag {
            partials: cuda.buffer(4 * most_chunks(&tensors))?,
            counts: cuda.buffer(4 * tensors.len().max(1))?,
            kernels,
            tensors,
        })
    }

    /// The number of gradients.
    pub fn tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Writes the finite flag of the loss, the `f32` at word `loss.1` of `loss.0`, and of the
    /// gradients in `grads` to word 0 of `flag`, a `u32`, overwritten. Nothing else of `flag`
    /// changes. Run it after the kernels that write the loss and the gradients, and before
    /// those that read the flag.
    ///
    /// The flag is set to 1 (skip) before the first count is launched and the fold overwrites
    /// it, so a run that fails partway leaves a flag that skips the step rather than the last
    /// step's value: a check that did not finish never lets an unchecked step through. No test
    /// provokes a failing launch, which the driver gives no cheap way to cause; the tests check
    /// that a run that succeeds writes the rule's flag over the 1.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused or the 1 cannot be written; when the write
    /// itself fails, `flag` may still hold its old value.
    ///
    /// # Panics
    ///
    /// If the loss's buffer holds no word `loss.1`, `grads` holds fewer values than the
    /// gradients together, or `flag` holds no word.
    pub fn run(
        &self,
        (loss, loss_at): (&DeviceBuffer<'_>, u32),
        grads: &DeviceBuffer<'_>,
        flag: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        const WHAT: &str = "finite flag";
        check_buffers(
            WHAT,
            &[
                (loss, "loss", u64::from(loss_at) + 1),
                (grads, "gradients", total(&self.tensors)),
                (flag, "flag", 1),
            ],
        );
        // Skip until the fold says otherwise: the copy follows the work already queued and
        // precedes the launches below.
        flag.write(&[1u32])?;
        let k = &self.kernels;
        for (at, &(start, len)) in (0u32..).zip(&self.tensors) {
            // SAFETY: the gradients hold start + len values, at most 2³¹ (checked when the
            // kernel was made and above), and len is above 0; the partial sums and the
            // counts, one per tensor, are this kernel's own.
            unsafe {
                k.tensor_sum(
                    k.non_finite_partials,
                    (grads, start, len),
                    &self.partials,
                    (&self.counts, at),
                )?;
            }
        }
        // SAFETY: finite_fold takes (const float *counts, const u32 *loss, u32 loss_at, u32
        // tensors, u32 *flag), reads one count per tensor, which this kernel's own counts
        // hold, and loss[loss_at], which the loss holds (checked above), and writes flag[0];
        // one thread does the work, and flag, borrowed mutably, is no input.
        unsafe {
            k.launch(
                k.finite_fold,
                [1, 1],
                &[
                    Arg::Buffer(&self.counts),
                    Arg::Buffer(loss),
                    Arg::U32(loss_at),
                    Arg::U32(self.tensors.len() as u32),
                    Arg::Buffer(flag),
                ],
            )
        }
    }

    /// Each gradient's count of NaN and infinite values from the last [`run`](Self::run), one
    /// `f32` per gradient in order, 0 for a finite gradient: read back to name the gradients
    /// that set the flag. Zeros before the first run.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, FiniteFlag};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let check = FiniteFlag::new(&cuda, &[2, 3]).unwrap();
    ///     let (mut loss, mut grads) = (cuda.buffer(4).unwrap(), cuda.buffer(20).unwrap());
    ///     let mut flag = cuda.buffer(4).unwrap();
    ///     loss.write(&[0.5f32]).unwrap();
    ///     grads.write(&[1.0f32, 2.0, f32::NAN, 1.0, f32::INFINITY]).unwrap();
    ///     check.run((&loss, 0), &grads, &mut flag).unwrap();
    ///     let mut counts = [0.0f32; 2];
    ///     check.counts().read(&mut counts).unwrap();
    ///     assert_eq!(counts, [0.0, 2.0]);
    /// }
    /// ```
    pub fn counts(&self) -> &DeviceBuffer<'c> {
        &self.counts
    }
}

/// The dead-ReLU check's bits on the card: for a ReLU's output laid out
/// `[outer][channels][inner]` ([`ChannelDims`]), one bit per channel, set when any of its
/// values is not ±0. A dense layer's output is `ChannelDims::rows(batch, width)`, a bit per
/// neuron; a convolution's is `ChannelDims::new(batch, channels, h · w)`, a bit per channel,
/// as the CPU counts a position of a convolution's channel dead only when its whole channel
/// is.
///
/// The test is by bits, as the CPU's `v != 0.0`: −0 counts as 0, and NaN and subnormals as
/// non-zero. Each channel's non-zero values are counted in the channel sums' fixed tree (never
/// 0 once a value is counted), then one thread per word packs the bits, so they are exact.
/// Bit `c % 32` of word `c / 32` is channel `c`'s, so `C` channels take
/// [`DeadRelu::words`]`(C)` words. Accumulating ORs them into the old words, so a buffer
/// cleared at the start of an epoch ends it holding every channel that was ever non-zero.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{ChannelDims, Cuda, DeadRelu};
///
/// if let Some(cuda) = Cuda::new() {
///     // Two rows of three neurons: neuron 1 is 0 in both, so its bit stays clear.
///     let dead = DeadRelu::new(&cuda, ChannelDims::rows(2, 3)).unwrap();
///     let (mut y, mut bits) = (cuda.buffer(24).unwrap(), cuda.buffer(4).unwrap());
///     y.write(&[0.5f32, 0.0, 0.0, 0.0, -0.0, 2.0]).unwrap();
///     dead.run(&y, (&mut bits, 0), false).unwrap();
///     let mut got = [0u32];
///     bits.read(&mut got).unwrap();
///     assert_eq!(got, [0b101]);
/// }
/// ```
pub struct DeadRelu<'c> {
    /// The shape.
    dims: ChannelDims,
    /// The compiled kernels.
    kernels: Kernels<'c>,
    /// One partial sum per chunk of each channel.
    partials: DeviceBuffer<'c>,
    /// Each channel's count of non-zero values.
    counts: DeviceBuffer<'c>,
}

impl fmt::Debug for DeadRelu<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeadRelu")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

impl<'c> DeadRelu<'c> {
    /// Compiles the kernels on `cuda`'s card for ReLU outputs shaped `dims`, and allocates the
    /// counts and their partial sums.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] for a dimension of 0, more than 2³¹ values or more than 65,535
    /// channels; [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses
    /// the kernels or the memory.
    pub fn new(cuda: &'c Cuda, dims: ChannelDims) -> Result<DeadRelu<'c>, CudaError> {
        dims.check("dead relu")?;
        let kernels = Kernels::new(cuda)?;
        let c = dims.channels as usize;
        Ok(DeadRelu {
            dims,
            kernels,
            partials: cuda.buffer(4 * dims.chunks() as usize * c)?,
            counts: cuda.buffer(4 * c)?,
        })
    }

    /// The words the bits of `channels` channels take: one per 32.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::DeadRelu;
    ///
    /// assert_eq!([1, 32, 33].map(DeadRelu::words), [1, 1, 2]);
    /// ```
    pub fn words(channels: u32) -> u32 {
        channels.div_ceil(32)
    }

    /// The shape.
    pub fn dims(&self) -> ChannelDims {
        self.dims
    }

    /// Writes the bits of the ReLU output `y` to words `offset` onwards of `out`,
    /// [`DeadRelu::words`]`(channels)` of them, overwritten or (`accumulate`) OR-ed into.
    /// Nothing else of `out` changes, and the bits of a last word past the channels are 0, or
    /// the old bits when accumulating.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If `y` holds fewer values than the shape, or `out` fewer than `offset` plus the words.
    pub fn run(
        &self,
        y: &DeviceBuffer<'_>,
        (out, offset): (&mut DeviceBuffer<'_>, u32),
        accumulate: bool,
    ) -> Result<(), CudaError> {
        const WHAT: &str = "dead relu";
        let dims = self.dims;
        let words = DeadRelu::words(dims.channels);
        check_buffers(
            WHAT,
            &[
                (y, "ReLU output", dims.len()),
                (out, "output", u64::from(offset) + u64::from(words)),
            ],
        );
        let chunks = dims.chunks();
        let k = &self.kernels;
        // SAFETY: non_zero_partials takes (const u32 *y, float *partials, u32 outer, u32
        // channels, u32 inner, u32 chunks); its grid of chunks by channels reads the values
        // below dims.len(), which y holds (checked above), and writes chunks · channels
        // partial sums, which this kernel's own buffer holds for its checked shape.
        unsafe {
            k.launch(
                k.non_zero_partials,
                [chunks, dims.channels],
                &[
                    Arg::Buffer(y),
                    Arg::Buffer(&self.partials),
                    Arg::U32(dims.outer),
                    Arg::U32(dims.channels),
                    Arg::U32(dims.inner),
                    Arg::U32(chunks),
                ],
            )?;
        }
        // SAFETY: the merge takes (const float *partials, float *sums, u32 channels, u32
        // chunks), reads chunks · channels partial sums and writes one count per channel,
        // both this kernel's own buffers.
        unsafe {
            k.launch(
                k.merge,
                [dims.channels.div_ceil(BLOCK), 1],
                &[
                    Arg::Buffer(&self.partials),
                    Arg::Buffer(&self.counts),
                    Arg::U32(dims.channels),
                    Arg::U32(chunks),
                ],
            )?;
        }
        // SAFETY: dead_relu_pack takes (const float *counts, u32 *bits, u32 channels, u32
        // offset, u32 accumulate), reads one count per channel and reads and writes words
        // offset..offset + words of bits, which out holds (checked above); out, borrowed
        // mutably, is no input.
        unsafe {
            k.launch(
                k.dead_relu_pack,
                [words.div_ceil(BLOCK), 1],
                &[
                    Arg::Buffer(&self.counts),
                    Arg::Buffer(out),
                    Arg::U32(dims.channels),
                    Arg::U32(offset),
                    Arg::U32(u32::from(accumulate)),
                ],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tensors one after another: their starts, and the two refusals.
    #[test]
    fn flat_buffers_are_checked() {
        assert_eq!(flat("t", &[2, 3, 1]).unwrap(), vec![(0, 2), (2, 3), (5, 1)]);
        assert_eq!(total(&flat("t", &[2, 3, 1]).unwrap()), 6);
        assert_eq!(total(&[]), 0);
        assert_eq!(most_chunks(&[(0, 1), (1, 4097)]), 2);
        assert_eq!(most_chunks(&[]), 1);
        assert!(flat("t", &[1 << 30, 1 << 30]).is_ok());
        for (lens, why) in [
            (vec![2, 0], "tensor 1 holds no values"),
            (vec![1 << 30, 1 << 30, 1], "more than the 2147483648"),
            (vec![usize::MAX, 1], "more than the 2147483648"),
        ] {
            let Err(CudaError::Invalid(text)) = flat("t", &lens) else {
                panic!("{lens:?} accepted");
            };
            assert!(text.contains(why), "{text}");
        }
    }
}
