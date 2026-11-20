//! Softmax and log-softmax over each row of a `[rows][width]` batch, forward and backward, as
//! this crate's own CUDA kernels: the twins of `cavitation_core`'s `Tape::softmax` and
//! `Tape::log_softmax` over the last axis, one thread per row in the CPU's order.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/softmax.rs`.
const SOURCE: &str = include_str!("softmax.cu");

/// Threads per block.
const BLOCK: u32 = 256;

/// One row operation of [`RowSoftmax`]: `a` is its first input, `b` the gradient flowing back
/// for a backward pass, and the result goes to `out`.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::RowSoftmaxOp;
///
/// assert!(!RowSoftmaxOp::LogSoftmax.takes_b());
/// assert!(RowSoftmaxOp::SoftmaxBackward.takes_b());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowSoftmaxOp {
    /// `e^(a − max) / Σ e^(a − max)` over each row.
    Softmax,
    /// `a · (b − Σ a·b)` over each row, with `a` the softmax's output.
    SoftmaxBackward,
    /// `a − max − ln Σ e^(a − max)` over each row.
    LogSoftmax,
    /// `b − e^a · Σ b` over each row, with `a` the log-softmax's output.
    LogSoftmaxBackward,
}

impl RowSoftmaxOp {
    /// Every operation, in declaration order.
    ///
    /// ```
    /// use cavitation_cuda::RowSoftmaxOp;
    ///
    /// assert_eq!(RowSoftmaxOp::ALL.iter().filter(|op| op.takes_b()).count(), 2);
    /// ```
    pub const ALL: [RowSoftmaxOp; 4] = [
        RowSoftmaxOp::Softmax,
        RowSoftmaxOp::SoftmaxBackward,
        RowSoftmaxOp::LogSoftmax,
        RowSoftmaxOp::LogSoftmaxBackward,
    ];

    /// Whether the operation reads `b`: the backward passes.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::RowSoftmaxOp;
    ///
    /// assert!(RowSoftmaxOp::LogSoftmaxBackward.takes_b());
    /// assert!(!RowSoftmaxOp::Softmax.takes_b());
    /// ```
    pub fn takes_b(self) -> bool {
        matches!(
            self,
            RowSoftmaxOp::SoftmaxBackward | RowSoftmaxOp::LogSoftmaxBackward
        )
    }

    /// The kernel that computes it.
    fn kernel(self) -> &'static str {
        match self {
            RowSoftmaxOp::Softmax => "softmax",
            RowSoftmaxOp::SoftmaxBackward => "softmax_backward",
            RowSoftmaxOp::LogSoftmax => "log_softmax",
            RowSoftmaxOp::LogSoftmaxBackward => "log_softmax_backward",
        }
    }
}

/// Softmax and log-softmax over each row, forward and backward, on the card.
///
/// Each row runs in one thread, which reads its values in order, as the CPU does, so two
/// runs give the same bits. The forward passes are the WGSL kernels' (`cavitation-gpu`'s
/// `RowSoftmaxKernels`), ported operation for operation with CUDA's `expf` and `logf`: the
/// row's maximum subtracted first, the softmax's terms summed in order, and the
/// log-softmax's `ln Σ e^(z − max)` as `ln(1 + e)` with `e` the sum without the first
/// maximum's term, from its series for `e ≤ 1`. A row with a NaN or `+∞`, or with `−∞`
/// everywhere, gives a NaN row, and a `−∞` elsewhere a softmax of 0 and a log-softmax of
/// `−∞`, as on the CPU. The backward passes are the tape's, `y · (g − Σ y·g)` and
/// `g − e^y · Σ g`, from the forward output `y` and the gradient `g`.
///
/// Compiled without fused multiply-add, the softmax's backward pass gives the CPU's bits;
/// the others take an exponential or a logarithm the CPU takes with std's and are within
/// the WGSL kernels' stated bounds of the `f64` tape. `tests/softmax.rs` checks each.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, RowSoftmax, RowSoftmaxOp};
///
/// if let Some(cuda) = Cuda::new() {
///     let kernels = RowSoftmax::new(&cuda).unwrap();
///     let mut x = cuda.buffer(4 * 4).unwrap();
///     let mut y = cuda.buffer(4 * 4).unwrap();
///     x.write(&[1.0f32, 1.0, 1e4, 0.0]).unwrap();
///     kernels.run(RowSoftmaxOp::Softmax, &x, None, &mut y, 2, 2).unwrap();
///     let mut out = [0.0f32; 4];
///     y.read(&mut out).unwrap();
///     assert_eq!(out, [0.5, 0.5, 1.0, 0.0]);
/// }
/// ```
pub struct RowSoftmax<'c> {
    /// The compiled kernels.
    module: Module<'c>,
    /// Each operation's kernel handle, in [`RowSoftmaxOp::ALL`]'s order, which is the
    /// declaration order.
    functions: [CuFunction; RowSoftmaxOp::ALL.len()],
}

impl fmt::Debug for RowSoftmax<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowSoftmax").finish_non_exhaustive()
    }
}

impl<'c> RowSoftmax<'c> {
    /// Compiles the kernels for `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses them.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmax`].
    pub fn new(cuda: &'c Cuda) -> Result<RowSoftmax<'c>, CudaError> {
        let module = cuda.compile("softmax.cu", SOURCE)?;
        let mut functions = [std::ptr::null_mut(); RowSoftmaxOp::ALL.len()];
        for (function, &op) in functions.iter_mut().zip(&RowSoftmaxOp::ALL) {
            *function = module.function(op.kernel())?;
        }
        Ok(RowSoftmax { module, functions })
    }

    /// Writes `op` of `rows` rows of `width` values of `a` (and `b`) to `out`, every one of
    /// the `rows × width` outputs. `b` must be given exactly when [`RowSoftmaxOp::takes_b`]
    /// says so. Nothing runs when `rows` or `width` is 0.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `b` is missing or given against `op`, or `rows × width`
    /// does not fit a `u32`; [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer holds fewer than `rows × width` floats.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmax`].
    pub fn run(
        &self,
        op: RowSoftmaxOp,
        a: &DeviceBuffer<'_>,
        b: Option<&DeviceBuffer<'_>>,
        out: &mut DeviceBuffer<'_>,
        rows: usize,
        width: usize,
    ) -> Result<(), CudaError> {
        if b.is_some() != op.takes_b() {
            return Err(CudaError::Invalid(format!(
                "{op:?} {} a second input",
                if op.takes_b() { "needs" } else { "takes no" }
            )));
        }
        let n = rows
            .checked_mul(width)
            .filter(|&n| u32::try_from(n).is_ok())
            .ok_or_else(|| {
                CudaError::Invalid(format!("{rows} rows of {width} do not fit a u32"))
            })?;
        let needs = n * 4;
        for (buffer, what) in [(a, "first input"), (&*out, "output")]
            .into_iter()
            .chain(b.map(|b| (b, "second input")))
        {
            assert!(
                buffer.bytes() >= needs,
                "the {what} buffer holds {} bytes, and {rows} rows of {width} need {needs}",
                buffer.bytes()
            );
        }
        if n == 0 {
            return Ok(());
        }
        let (rows, width) = (rows as u32, width as u32);
        // SAFETY: the handles were looked up in this module when it was compiled, in `ALL`'s
        // order, which is the declaration order (a unit test holds it).
        let kernel = unsafe { self.module.kernel_from(self.functions[op as usize]) };
        // SAFETY: every kernel in the source takes (const float *a, const float *b, float
        // *out, u32 rows, u32 width) and touches the `rows × width` values only; a, b and
        // out hold at least that many floats (checked above), b is a when the operation does
        // not read it, and out, borrowed mutably, is neither input.
        unsafe {
            kernel.launch(
                [rows.div_ceil(BLOCK), 1, 1],
                [BLOCK, 1, 1],
                &[
                    Arg::Buffer(a),
                    Arg::Buffer(b.unwrap_or(a)),
                    Arg::Buffer(out),
                    Arg::U32(rows),
                    Arg::U32(width),
                ],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ALL` lists every operation once, in declaration order, so `op as usize` indexes it.
    #[test]
    fn all_is_in_declaration_order() {
        for (i, &op) in RowSoftmaxOp::ALL.iter().enumerate() {
            assert_eq!(op as usize, i, "{op:?}");
        }
    }
}
