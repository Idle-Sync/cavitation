//! Gated linear units as this crate's own CUDA kernels, the twin of `cavitation_core`'s
//! `Tape::glu`: each row of a `[rows][width]` batch split into halves `a` and `b`, the forward
//! pass `a · gate(b)` and the backward pass writing both halves of the input's gradient.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::elementwise::ACTIVATION;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/glu.rs`, compiled after the
/// shared scalar functions.
const SOURCE: &str = include_str!("glu.cu");

/// Threads per block.
const BLOCK: u32 = 256;

/// The activation a GLU applies to the second half of each row: `cavitation_core`'s `Gate`,
/// with GELU's two forms apart.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::GluGate;
///
/// assert_eq!(GluGate::ALL.len(), 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GluGate {
    /// The logistic sigmoid: the original GLU.
    Sigmoid,
    /// GELU's tanh form: GeGLU.
    GeluTanh,
    /// GELU's erf form: GeGLU.
    GeluErf,
    /// ReLU: ReGLU.
    Relu,
    /// SiLU: SwiGLU.
    Silu,
}

impl GluGate {
    /// Every gate, in declaration order.
    ///
    /// ```
    /// use cavitation_cuda::GluGate;
    ///
    /// assert_eq!(GluGate::ALL[0], GluGate::Sigmoid);
    /// ```
    pub const ALL: [GluGate; 5] = [
        GluGate::Sigmoid,
        GluGate::GeluTanh,
        GluGate::GeluErf,
        GluGate::Relu,
        GluGate::Silu,
    ];

    /// The kernels' name for the gate.
    fn name(self) -> &'static str {
        match self {
            GluGate::Sigmoid => "sigmoid",
            GluGate::GeluTanh => "gelu_tanh",
            GluGate::GeluErf => "gelu_erf",
            GluGate::Relu => "relu",
            GluGate::Silu => "silu",
        }
    }
}

/// One GLU pass of [`Glu`].
///
/// # Examples
///
/// ```
/// use cavitation_cuda::GluOp;
///
/// assert!(!GluOp::Forward.takes_g());
/// assert!(GluOp::Backward.takes_g());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GluOp {
    /// `a · gate(b)`, half the input's width.
    Forward,
    /// The input's gradient, both halves: `g · gate(b)` and `g · a · gate′(b)`.
    Backward,
}

impl GluOp {
    /// Every pass, in declaration order.
    ///
    /// ```
    /// use cavitation_cuda::GluOp;
    ///
    /// assert_eq!(GluOp::ALL.len(), 2);
    /// ```
    pub const ALL: [GluOp; 2] = [GluOp::Forward, GluOp::Backward];

    /// Whether the pass reads the gradient `g`: the backward pass.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::GluOp;
    ///
    /// assert!(GluOp::Backward.takes_g());
    /// ```
    pub fn takes_g(self) -> bool {
        matches!(self, GluOp::Backward)
    }

    /// The kernels' name for the pass.
    fn name(self) -> &'static str {
        match self {
            GluOp::Forward => "forward",
            GluOp::Backward => "backward",
        }
    }
}

/// Gated linear units on the card for a `[rows][width]` batch of even width, with any
/// [`GluGate`], one thread per pair `(a, b)`; the backward pass writes both of the pair's
/// gradients, so every value of the input's gradient is written.
///
/// The kernels are `cavitation-gpu`'s WGSL ones (`GluKernels`) ported operation for
/// operation and compiled without fused multiply-add, the gates the element-wise kernels'
/// own functions; NVRTC keeps subnormals, so:
///
/// - **ReLU gate:** one multiplication forward, `g · relu(b)` and `(g · a) · relu′(b)`
///   backward, the CPU's bits, subnormals included, NaN aside (CUDA makes its own NaN, so NaN
///   appears where the CPU's does, without its bits). The derivative at exactly 0 is 0.
/// - **Sigmoid, GELU (both forms) and SiLU gates:** within the WGSL kernels' bounds of the
///   `f64` tape (`tests/glu.rs`), and the CPU's bits on ±0 and ±∞ and NaN for NaN.
///
/// Each thread's operations depend on its pair alone, so two runs give the same bits.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, Glu, GluGate, GluOp};
///
/// if let Some(cuda) = Cuda::new() {
///     let kernels = Glu::new(&cuda).unwrap();
///     let mut x = cuda.buffer(4 * 4).unwrap();
///     let mut y = cuda.buffer(4 * 2).unwrap();
///     x.write(&[2.0f32, 3.0, -1.0, 4.0]).unwrap();
///     kernels.run(GluOp::Forward, GluGate::Relu, &x, None, &mut y, 1, 4).unwrap();
///     let mut out = [0.0f32; 2];
///     y.read(&mut out).unwrap();
///     // 2 · relu(−1) and 3 · relu(4).
///     assert_eq!(out, [0.0, 12.0]);
/// }
/// ```
pub struct Glu<'c> {
    /// The compiled kernels.
    module: Module<'c>,
    /// Each gate's kernel handles, in [`GluGate::ALL`]'s order, each gate's passes in
    /// [`GluOp::ALL`]'s; both are the declaration order.
    functions: [[CuFunction; GluOp::ALL.len()]; GluGate::ALL.len()],
}

impl fmt::Debug for Glu<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Glu").finish_non_exhaustive()
    }
}

impl<'c> Glu<'c> {
    /// Compiles the kernels for `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses them.
    ///
    /// # Examples
    ///
    /// See [`Glu`].
    pub fn new(cuda: &'c Cuda) -> Result<Glu<'c>, CudaError> {
        let module = cuda.compile("glu.cu", &format!("{ACTIVATION}{SOURCE}"))?;
        let mut functions = [[std::ptr::null_mut(); GluOp::ALL.len()]; GluGate::ALL.len()];
        for (row, gate) in functions.iter_mut().zip(GluGate::ALL) {
            for (function, op) in row.iter_mut().zip(GluOp::ALL) {
                *function = module.function(&format!("glu_{}_{}", op.name(), gate.name()))?;
            }
        }
        Ok(Glu { module, functions })
    }

    /// Writes `op` with `gate` over `rows` rows of `width` values of `x` (and `g`, `rows`
    /// rows of `width / 2`) to `out` (`width / 2` values a row forward, `width` backward),
    /// every one of them. `g` must be given exactly when [`GluOp::takes_g`] says so. Nothing
    /// runs when `rows` or `width` is 0.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `g` is missing or given against `op`, `width` is odd, or
    /// `rows × width` does not fit a `u32`; [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer holds fewer values than it needs.
    ///
    /// # Examples
    ///
    /// See [`Glu`].
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        op: GluOp,
        gate: GluGate,
        x: &DeviceBuffer<'_>,
        g: Option<&DeviceBuffer<'_>>,
        out: &mut DeviceBuffer<'_>,
        rows: usize,
        width: usize,
    ) -> Result<(), CudaError> {
        if g.is_some() != op.takes_g() {
            return Err(CudaError::Invalid(format!(
                "glu: {op:?} {} a gradient",
                if op.takes_g() { "needs" } else { "takes no" }
            )));
        }
        if !width.is_multiple_of(2) {
            return Err(CudaError::Invalid(format!(
                "glu: the width must be even, got {width}"
            )));
        }
        let n = rows
            .checked_mul(width)
            .filter(|&n| u32::try_from(n).is_ok())
            .ok_or_else(|| {
                CudaError::Invalid(format!("glu: {rows} rows of {width} do not fit a u32"))
            })?;
        let pairs = n / 2;
        check_buffers(&[
            (x, "input", n),
            (&*out, "output", if op.takes_g() { n } else { pairs }),
        ]);
        if let Some(g) = g {
            check_buffers(&[(g, "gradient", pairs)]);
        }
        if n == 0 {
            return Ok(());
        }
        let (pairs, half) = (pairs as u32, (width / 2) as u32);
        // SAFETY: the handles were looked up in this module when it was compiled, in `ALL`'s
        // orders, which are the declaration orders (a unit test holds them).
        let kernel = unsafe {
            self.module
                .kernel_from(self.functions[gate as usize][op as usize])
        };
        // SAFETY: every kernel takes (const float *x, const float *g, float *out, u32 pairs,
        // u32 half), reads x below 2 · pairs and g below pairs, and writes out below pairs
        // (forward) or 2 · pairs (backward), which the buffers hold (checked above); g is x
        // when the pass does not read it, and out, borrowed mutably, is none of the inputs.
        unsafe {
            kernel.launch(
                [pairs.div_ceil(BLOCK), 1, 1],
                [BLOCK, 1, 1],
                &[
                    Arg::Buffer(x),
                    Arg::Buffer(g.unwrap_or(x)),
                    Arg::Buffer(out),
                    Arg::U32(pairs),
                    Arg::U32(half),
                ],
            )
        }
    }
}

/// Panics unless every buffer holds at least its count of four-byte values, naming the
/// first that does not.
fn check_buffers(buffers: &[(&DeviceBuffer<'_>, &str, usize)]) {
    for &(buffer, what, n) in buffers {
        assert!(
            buffer.bytes() >= 4 * n,
            "glu: the {what} buffer holds {} bytes, and {n} values need {}",
            buffer.bytes(),
            4 * n
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ALL` lists every gate and every pass once, in declaration order, so `as usize`
    /// indexes them.
    #[test]
    fn all_is_in_declaration_order() {
        for (i, &gate) in GluGate::ALL.iter().enumerate() {
            assert_eq!(gate as usize, i, "{gate:?}");
        }
        for (i, &op) in GluOp::ALL.iter().enumerate() {
            assert_eq!(op as usize, i, "{op:?}");
        }
    }
}
