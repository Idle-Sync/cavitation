//! Dropout and the neuron mask as this crate's own CUDA kernels, with a bit-packed mask: the
//! twins of the CPU's dropout multipliers and `Tape::keep_where`, forward and backward alike.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `dropout.wgsl`.
const SOURCE: &str = include_str!("dropout.cu");

/// Threads per block.
const BLOCK: u32 = 256;

/// Dropout and the neuron mask on the card, each over `n` `f32`s with a mask of `u32` words:
/// element `i` is kept when bit `i % 32` of word `i / 32` is set, the low bit first, so `n`
/// elements take [`mask_words`](Dropout::mask_words)`(n)` words.
///
/// - [`dropout`](Dropout::dropout): `x · m`, `m` the scale where the bit is set and 0 where
///   it is clear, as the CPU multiplies by its dropout multipliers: a dropped −3 is −0 and a
///   dropped NaN or infinity is NaN. The backward pass is the same call on the gradient with
///   the same mask.
/// - [`keep_where`](Dropout::keep_where): `x` where the bit is set and `+0` where it is
///   clear, selected rather than multiplied, as `Tape::keep_where` computes it: a cleared −3,
///   −0, NaN or infinity is `+0`. The backward pass is the same call on the gradient.
///
/// Each is one rounding at most, so both give the CPU's bits, ±0, ±∞ and subnormals
/// included; a NaN kept or multiplied stays NaN (CUDA's multiply makes its own NaN, without
/// the operand's payload, where it multiplies one). `tests/dropout.rs` checks both.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, Dropout};
///
/// if let Some(cuda) = Cuda::new() {
///     let kernels = Dropout::new(&cuda).unwrap();
///     let (mut x, mut mask, mut y) =
///         (cuda.buffer(16).unwrap(), cuda.buffer(4).unwrap(), cuda.buffer(16).unwrap());
///     x.write(&[3.0f32, -3.0, 1.0, 2.0]).unwrap();
///     mask.write(&[0b1101u32]).unwrap(); // element 1 dropped
///     let scale = Dropout::scale(0.5).unwrap();
///     kernels.dropout(&x, &mask, &mut y, 4, scale).unwrap();
///     let mut out = [0.0f32; 4];
///     y.read(&mut out).unwrap();
///     assert_eq!(out.map(f32::to_bits), [6.0f32, -0.0, 2.0, 4.0].map(f32::to_bits));
///     kernels.keep_where(&x, &mask, &mut y, 4).unwrap();
///     y.read(&mut out).unwrap();
///     assert_eq!(out.map(f32::to_bits), [3.0f32, 0.0, 1.0, 2.0].map(f32::to_bits));
/// }
/// ```
pub struct Dropout<'c> {
    /// The compiled kernels.
    module: Module<'c>,
    /// The dropout kernel.
    dropout: CuFunction,
    /// The neuron mask's kernel.
    keep_where: CuFunction,
}

impl fmt::Debug for Dropout<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dropout").finish_non_exhaustive()
    }
}

impl<'c> Dropout<'c> {
    /// Compiles the kernels for `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses them.
    pub fn new(cuda: &'c Cuda) -> Result<Dropout<'c>, CudaError> {
        let module = cuda.compile("dropout.cu", SOURCE)?;
        Ok(Dropout {
            dropout: module.function("dropout")?,
            keep_where: module.function("keep_where")?,
            module,
        })
    }

    /// The scale of dropout with drop probability `p`, as the CPU rounds it: `1 / f32(1 − p)`,
    /// which differs from `1 / (1 − p)` rounded once for some `p`.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] unless `p` is in `[0, 1)`, as the CPU requires.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Dropout;
    ///
    /// assert_eq!(Dropout::scale(0.5).unwrap(), 2.0);
    /// assert!(Dropout::scale(1.0).is_err());
    /// ```
    pub fn scale(p: f64) -> Result<f32, CudaError> {
        if !(0.0..1.0).contains(&p) {
            return Err(CudaError::Invalid(format!(
                "dropout: the drop probability must be in [0, 1), got {p}"
            )));
        }
        Ok(1.0 / (1.0 - p) as f32)
    }

    /// The mask words `n` elements take: one per 32.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Dropout;
    ///
    /// assert_eq!([1, 32, 33].map(Dropout::mask_words), [1, 1, 2]);
    /// ```
    pub fn mask_words(n: usize) -> usize {
        n.div_ceil(32)
    }

    /// Writes dropout of the first `n` elements of `x` to `out`: `x · scale` where `mask`'s
    /// bit is set and `x · 0` where it is clear. Every one of the `n` outputs is written.
    /// `scale` is [`Dropout::scale`]`(p)` for the CPU's bits.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `scale` is not finite or `n` does not fit a `u32`;
    /// [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If `x` or `out` holds fewer than `n` floats, or `mask` fewer than
    /// [`mask_words`](Dropout::mask_words)`(n)` words.
    pub fn dropout(
        &self,
        x: &DeviceBuffer<'_>,
        mask: &DeviceBuffer<'_>,
        out: &mut DeviceBuffer<'_>,
        n: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        if !scale.is_finite() {
            return Err(CudaError::Invalid(format!(
                "dropout: the scale must be finite, got {scale}"
            )));
        }
        self.launch(self.dropout, "dropout", x, mask, out, n, Some(scale))
    }

    /// Writes the first `n` elements of `x` where `mask`'s bit is set, and `+0` where it is
    /// clear, to `out`. Every one of the `n` outputs is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `n` does not fit a `u32`; [`CudaError::Call`] if the launch
    /// is refused.
    ///
    /// # Panics
    ///
    /// If `x` or `out` holds fewer than `n` floats, or `mask` fewer than
    /// [`mask_words`](Dropout::mask_words)`(n)` words.
    pub fn keep_where(
        &self,
        x: &DeviceBuffer<'_>,
        mask: &DeviceBuffer<'_>,
        out: &mut DeviceBuffer<'_>,
        n: usize,
    ) -> Result<(), CudaError> {
        self.launch(self.keep_where, "keep_where", x, mask, out, n, None)
    }

    /// Checks the sizes and launches `function`, named `what`, over `n` elements, with
    /// `scale` as its last parameter when it takes one.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        function: CuFunction,
        what: &str,
        x: &DeviceBuffer<'_>,
        mask: &DeviceBuffer<'_>,
        out: &mut DeviceBuffer<'_>,
        n: usize,
        scale: Option<f32>,
    ) -> Result<(), CudaError> {
        let count = u32::try_from(n)
            .map_err(|_| CudaError::Invalid(format!("{what}: {n} elements do not fit a u32")))?;
        let words = Dropout::mask_words(n);
        for (buffer, name, values) in [(x, "input", n), (&*out, "output", n), (mask, "mask", words)]
        {
            let needs = values.saturating_mul(4);
            assert!(
                buffer.bytes() >= needs,
                "{what}: the {name} buffer holds {} bytes, and {n} elements need {needs}",
                buffer.bytes()
            );
        }
        if n == 0 {
            return Ok(());
        }
        // SAFETY: both handles were looked up in this module when it was compiled.
        let kernel = unsafe { self.module.kernel_from(function) };
        let mut args = vec![
            Arg::Buffer(x),
            Arg::Buffer(mask),
            Arg::Buffer(out),
            Arg::U32(count),
        ];
        args.extend(scale.map(Arg::F32));
        // SAFETY: dropout takes (const float *a, const u32 *mask, float *out, u32 n, float
        // scale) and keep_where the same without the scale, which `args` holds for the kernel
        // launched (the scale given exactly for dropout); each touches elements below n of a
        // and out and mask words below ceil(n / 32), which the buffers hold (checked above);
        // out, borrowed mutably, is neither input.
        unsafe { kernel.launch([count.div_ceil(BLOCK), 1, 1], [BLOCK, 1, 1], &args) }
    }
}
