//! PReLU as this crate's own CUDA kernels, the twin of `cavitation_core`'s `Tape::prelu`:
//! the forward pass and the input's gradient one thread per value, and the slopes' gradient
//! summed per channel in the fixed tree of `reduce.cu`.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::reduce::{self, BLOCK, ChannelDims};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/prelu.rs`, compiled after the
/// sums' source.
const SOURCE: &str = include_str!("prelu.cu");

/// One element-wise PReLU operation of [`Prelu`].
///
/// # Examples
///
/// ```
/// use cavitation_cuda::PreluOp;
///
/// assert!(!PreluOp::Forward.takes_g());
/// assert!(PreluOp::InputGrad.takes_g());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PreluOp {
    /// `x` above 0, `slope · x` elsewhere.
    Forward,
    /// `g · (1 above 0, slope elsewhere)`, the input's gradient.
    InputGrad,
}

impl PreluOp {
    /// Every operation, in declaration order.
    ///
    /// ```
    /// use cavitation_cuda::PreluOp;
    ///
    /// assert_eq!(PreluOp::ALL.len(), 2);
    /// ```
    pub const ALL: [PreluOp; 2] = [PreluOp::Forward, PreluOp::InputGrad];

    /// Whether the operation reads the gradient `g`: the input's gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::PreluOp;
    ///
    /// assert!(PreluOp::InputGrad.takes_g());
    /// ```
    pub fn takes_g(self) -> bool {
        matches!(self, PreluOp::InputGrad)
    }

    /// The kernel that computes it.
    fn kernel(self) -> &'static str {
        match self {
            PreluOp::Forward => "prelu_forward",
            PreluOp::InputGrad => "prelu_input_grad",
        }
    }
}

/// PReLU on the card for a `[rows][channels]` batch whose value `i` takes slope
/// `i % channels`: one slope per column (`channels` the width), or one shared slope
/// (`channels = 1`, `rows` every value).
///
/// The kernels are `cavitation-gpu`'s WGSL ones (`PreluKernels`) ported operation for
/// operation and compiled without fused multiply-add; NVRTC keeps subnormals, so:
///
/// - **Forward** and **input gradient:** one multiplication per value, the CPU's bits, NaN
///   aside (CUDA makes its own NaN, so NaN appears where the CPU's does, without its bits).
///   The derivative at exactly 0 is the slope, as on the CPU.
/// - **Slopes' gradient:** each channel's `Σ g·x` over its values not above 0, each term the
///   CPU's product, summed in the fixed tree of [`BatchNorm2d`](crate::BatchNorm2d)'s sums
///   rather than the CPU's order: within `2(L + 2)·2⁻²⁴·Σ|g·x|` of the exact sum of the
///   same `f32` terms for `L = rows` terms per channel, and the CPU's sum bit for bit on
///   terms whose partial sums are exact. A NaN input makes its channel's sum NaN.
///
/// The tree depends on the shape alone, so two runs give the same bits. `tests/prelu.rs`
/// checks each.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, Prelu, PreluOp};
///
/// if let Some(cuda) = Cuda::new() {
///     let kernels = Prelu::new(&cuda).unwrap();
///     let mut x = cuda.buffer(4 * 4).unwrap();
///     let mut slope = cuda.buffer(4 * 2).unwrap();
///     let mut y = cuda.buffer(4 * 4).unwrap();
///     x.write(&[-4.0f32, 2.0, 3.0, -8.0]).unwrap();
///     slope.write(&[0.25f32, 0.5]).unwrap();
///     kernels.run(PreluOp::Forward, &x, &slope, None, &mut y, 2, 2).unwrap();
///     let mut out = [0.0f32; 4];
///     y.read(&mut out).unwrap();
///     assert_eq!(out, [-1.0, 2.0, 3.0, -4.0]);
/// }
/// ```
pub struct Prelu<'c> {
    /// The compiled sums and PReLU kernels.
    module: Module<'c>,
    /// Each element-wise operation's kernel handle, in [`PreluOp::ALL`]'s order, which is the
    /// declaration order.
    functions: [CuFunction; PreluOp::ALL.len()],
    /// The slopes' gradient's first pass.
    slope_partials: CuFunction,
    /// The sums' second pass.
    merge: CuFunction,
}

impl fmt::Debug for Prelu<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Prelu").finish_non_exhaustive()
    }
}

impl<'c> Prelu<'c> {
    /// Compiles the kernels for `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses them.
    ///
    /// # Examples
    ///
    /// See [`Prelu`].
    pub fn new(cuda: &'c Cuda) -> Result<Prelu<'c>, CudaError> {
        let module = cuda.compile("prelu.cu", &format!("{}{SOURCE}", reduce::SOURCE))?;
        let mut functions = [std::ptr::null_mut(); PreluOp::ALL.len()];
        for (function, &op) in functions.iter_mut().zip(&PreluOp::ALL) {
            *function = module.function(op.kernel())?;
        }
        Ok(Prelu {
            slope_partials: module.function("prelu_slope_partials")?,
            merge: module.function("merge")?,
            functions,
            module,
        })
    }

    /// The floats of scratch memory [`slope_grad`](Prelu::slope_grad) needs for `rows` rows
    /// of `channels`: one partial sum per chunk of 4,096 rows of each channel.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Prelu;
    ///
    /// assert_eq!(Prelu::partials_len(10_000, 3), 3 * 3);
    /// ```
    pub fn partials_len(rows: usize, channels: usize) -> usize {
        rows.div_ceil(reduce::CHUNK as usize) * channels
    }

    /// Writes `op` of `rows × channels` values of `x`, with `channels` slopes (and `g`), to
    /// `out`, every one of them. `g` must be given exactly when [`PreluOp::takes_g`] says so.
    /// Nothing runs when `rows` or `channels` is 0.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `g` is missing or given against `op`, or `rows × channels`
    /// does not fit a `u32`; [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer holds fewer values than it needs.
    ///
    /// # Examples
    ///
    /// See [`Prelu`].
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        op: PreluOp,
        x: &DeviceBuffer<'_>,
        slope: &DeviceBuffer<'_>,
        g: Option<&DeviceBuffer<'_>>,
        out: &mut DeviceBuffer<'_>,
        rows: usize,
        channels: usize,
    ) -> Result<(), CudaError> {
        if g.is_some() != op.takes_g() {
            return Err(CudaError::Invalid(format!(
                "{op:?} {} a gradient",
                if op.takes_g() { "needs" } else { "takes no" }
            )));
        }
        let n = values(rows, channels)?;
        check_buffers(&[
            (x, "input", n),
            (slope, "slopes", channels),
            (&*out, "output", n),
        ]);
        if let Some(g) = g {
            check_buffers(&[(g, "gradient", n)]);
        }
        if n == 0 {
            return Ok(());
        }
        let (n, channels) = (n as u32, channels as u32);
        // SAFETY: the handles were looked up in this module when it was compiled, in `ALL`'s
        // order, which is the declaration order (a unit test holds it).
        let kernel = unsafe { self.module.kernel_from(self.functions[op as usize]) };
        // SAFETY: both kernels take (const float *x, const float *slope, const float *g,
        // float *out, u32 n, u32 channels), touch values below n and slopes below channels,
        // which the buffers hold (checked above); g is x when the operation does not read
        // it, and out, borrowed mutably, is none of the inputs.
        unsafe {
            kernel.launch(
                [n.div_ceil(BLOCK), 1, 1],
                [BLOCK, 1, 1],
                &[
                    Arg::Buffer(x),
                    Arg::Buffer(slope),
                    Arg::Buffer(g.unwrap_or(x)),
                    Arg::Buffer(out),
                    Arg::U32(n),
                    Arg::U32(channels),
                ],
            )
        }
    }

    /// Writes the slopes' gradient for `rows × channels` values of the forward input `x` and
    /// the gradient `g` to `out` (`channels` values), every one of them, with `partials`
    /// ([`partials_len`](Prelu::partials_len) floats) as scratch memory.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] for a dimension of 0, more than 2³¹ values or more than 65,535
    /// channels; [`CudaError::Call`] if a launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer holds fewer values than it needs.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, Prelu};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let kernels = Prelu::new(&cuda).unwrap();
    ///     let mut x = cuda.buffer(4 * 4).unwrap();
    ///     let mut g = cuda.buffer(4 * 4).unwrap();
    ///     let mut partials = cuda.buffer(4 * Prelu::partials_len(2, 2)).unwrap();
    ///     let mut ds = cuda.buffer(4 * 2).unwrap();
    ///     x.write(&[-4.0f32, 2.0, -3.0, -8.0]).unwrap();
    ///     g.write(&[1.0f32, 5.0, 2.0, 0.5]).unwrap();
    ///     kernels.slope_grad(&x, &g, &mut partials, &mut ds, 2, 2).unwrap();
    ///     let mut out = [0.0f32; 2];
    ///     ds.read(&mut out).unwrap();
    ///     // Column 0: 1·(−4) + 2·(−3); column 1: 0.5·(−8), the 2 being above 0.
    ///     assert_eq!(out, [-10.0, -4.0]);
    /// }
    /// ```
    pub fn slope_grad(
        &self,
        x: &DeviceBuffer<'_>,
        g: &DeviceBuffer<'_>,
        partials: &mut DeviceBuffer<'_>,
        out: &mut DeviceBuffer<'_>,
        rows: usize,
        channels: usize,
    ) -> Result<(), CudaError> {
        let n = values(rows, channels)?;
        let dims = ChannelDims::rows(rows as u32, channels as u32);
        dims.check("prelu slope gradient")?;
        check_buffers(&[
            (x, "input", n),
            (g, "gradient", n),
            (
                &*partials,
                "partial sums",
                Prelu::partials_len(rows, channels),
            ),
            (&*out, "slope gradient", channels),
        ]);
        let chunks = dims.chunks();
        // SAFETY: the handle was looked up in this module.
        let first = unsafe { self.module.kernel_from(self.slope_partials) };
        // SAFETY: prelu_slope_partials takes (const float *x, const float *g, float
        // *partials, u32 outer, u32 channels, u32 inner, u32 chunks); its grid of chunks by
        // channels reads values below n of x and g and writes partial sums below chunks ·
        // channels, which the buffers hold (checked above); partials, borrowed mutably, is
        // neither input.
        unsafe {
            first.launch(
                [chunks, dims.channels, 1],
                [BLOCK, 1, 1],
                &[
                    Arg::Buffer(x),
                    Arg::Buffer(g),
                    Arg::Buffer(partials),
                    Arg::U32(dims.outer),
                    Arg::U32(dims.channels),
                    Arg::U32(dims.inner),
                    Arg::U32(chunks),
                ],
            )?;
        }
        // SAFETY: the handle was looked up in this module.
        let merge = unsafe { self.module.kernel_from(self.merge) };
        // SAFETY: the merge takes (const float *partials, float *sums, u32 channels, u32
        // chunks), reads chunks · channels partial sums and writes channels sums, which the
        // buffers hold; out, borrowed mutably, is not partials.
        unsafe {
            merge.launch(
                [dims.channels.div_ceil(BLOCK), 1, 1],
                [BLOCK, 1, 1],
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

/// `rows × channels`, refused unless it fits a `u32`.
fn values(rows: usize, channels: usize) -> Result<usize, CudaError> {
    rows.checked_mul(channels)
        .filter(|&n| u32::try_from(n).is_ok())
        .ok_or_else(|| {
            CudaError::Invalid(format!("prelu: {rows} rows of {channels} do not fit a u32"))
        })
}

/// Panics unless every buffer holds at least its count of four-byte values, naming the
/// first that does not.
fn check_buffers(buffers: &[(&DeviceBuffer<'_>, &str, usize)]) {
    for &(buffer, what, n) in buffers {
        assert!(
            buffer.bytes() >= 4 * n,
            "prelu: the {what} buffer holds {} bytes, and {n} values need {}",
            buffer.bytes(),
            4 * n
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ALL` lists every operation once, in declaration order, so `op as usize` indexes it.
    #[test]
    fn all_is_in_declaration_order() {
        for (i, &op) in PreluOp::ALL.iter().enumerate() {
            assert_eq!(op as usize, i, "{op:?}");
        }
    }

    /// The scratch memory is one float per chunk of 4,096 rows of each channel.
    #[test]
    fn partials_cover_every_chunk() {
        assert_eq!(Prelu::partials_len(1, 1), 1);
        assert_eq!(Prelu::partials_len(4096, 2), 2);
        assert_eq!(Prelu::partials_len(4097, 2), 4);
    }
}
