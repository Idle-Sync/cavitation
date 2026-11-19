//! The element-wise operations as this crate's own CUDA kernels: add, multiply, scale and
//! the activations (ReLU, leaky ReLU, tanh, the sigmoid, GELU in both forms), forward and
//! backward, each the twin of its CPU code.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `elementwise.wgsl`.
const SOURCE: &str = include_str!("elementwise.cu");

/// Threads per block.
const BLOCK: u32 = 256;

/// One element-wise operation, applied to every element of `a` (and `b`).
///
/// The one-rounding operations (`Add`, `Mul`, `Scale`, the ReLUs forward and backward) give
/// the CPU's bits, NaN, ±0 and ±∞ included. tanh, the sigmoid and GELU are the CPU's formulas
/// ported operation for operation and are checked against bounds of the f64 tape's values in
/// `tests/elementwise.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ElementwiseOp {
    /// `a + b`.
    Add,
    /// `a · b`.
    Mul,
    /// `alpha · a`.
    Scale,
    /// `max(a, 0)`: 0 at or below 0 (so −0 gives +0), NaN for NaN.
    Relu,
    /// `b · relu'(a)`: `b` above 0, 0 at or below, NaN for a NaN `a`.
    ReluBackward,
    /// `a` above 0, `alpha · a` at or below, NaN for NaN.
    LeakyRelu,
    /// `b · (1 above 0, alpha at or below)`, NaN for a NaN `a`.
    LeakyReluBackward,
    /// `tanh(a)`.
    Tanh,
    /// `b · (1 − a²)`, with `a` the forward output.
    TanhBackward,
    /// `1 / (1 + e^-a)`.
    Sigmoid,
    /// `b · a · (1 − a)`, with `a` the forward output.
    SigmoidBackward,
    /// GELU's tanh form, `0.5 · a · (1 + tanh(√(2/π) · (a + 0.044715 · a³)))`.
    GeluTanh,
    /// `b` times the derivative of GELU's tanh form at the forward input `a`.
    GeluTanhBackward,
    /// GELU's erf form, `0.5 · a · (1 + erf(a / √2))`.
    GeluErf,
    /// `b` times `Φ(a) + a · φ(a)`, the derivative of GELU's erf form at the input `a`.
    GeluErfBackward,
}

impl ElementwiseOp {
    /// Every operation, each once.
    pub const ALL: [ElementwiseOp; 15] = [
        ElementwiseOp::Add,
        ElementwiseOp::Mul,
        ElementwiseOp::Scale,
        ElementwiseOp::Relu,
        ElementwiseOp::ReluBackward,
        ElementwiseOp::LeakyRelu,
        ElementwiseOp::LeakyReluBackward,
        ElementwiseOp::Tanh,
        ElementwiseOp::TanhBackward,
        ElementwiseOp::Sigmoid,
        ElementwiseOp::SigmoidBackward,
        ElementwiseOp::GeluTanh,
        ElementwiseOp::GeluTanhBackward,
        ElementwiseOp::GeluErf,
        ElementwiseOp::GeluErfBackward,
    ];

    /// The name of the operation's kernel in the source.
    fn kernel(self) -> &'static str {
        match self {
            ElementwiseOp::Add => "add",
            ElementwiseOp::Mul => "mul",
            ElementwiseOp::Scale => "scale",
            ElementwiseOp::Relu => "relu",
            ElementwiseOp::ReluBackward => "relu_backward",
            ElementwiseOp::LeakyRelu => "leaky_relu",
            ElementwiseOp::LeakyReluBackward => "leaky_relu_backward",
            ElementwiseOp::Tanh => "tanh_forward",
            ElementwiseOp::TanhBackward => "tanh_backward",
            ElementwiseOp::Sigmoid => "sigmoid_forward",
            ElementwiseOp::SigmoidBackward => "sigmoid_backward",
            ElementwiseOp::GeluTanh => "gelu_tanh_forward",
            ElementwiseOp::GeluTanhBackward => "gelu_tanh_backward",
            ElementwiseOp::GeluErf => "gelu_erf_forward",
            ElementwiseOp::GeluErfBackward => "gelu_erf_backward",
        }
    }

    /// Whether the operation reads `b`: the binary operations and every backward pass.
    pub fn takes_b(self) -> bool {
        matches!(
            self,
            ElementwiseOp::Add
                | ElementwiseOp::Mul
                | ElementwiseOp::ReluBackward
                | ElementwiseOp::LeakyReluBackward
                | ElementwiseOp::TanhBackward
                | ElementwiseOp::SigmoidBackward
                | ElementwiseOp::GeluTanhBackward
                | ElementwiseOp::GeluErfBackward
        )
    }
}

/// The element-wise kernels, compiled once for a card.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, Elementwise, ElementwiseOp};
///
/// if let Some(cuda) = Cuda::new() {
///     let kernels = Elementwise::new(&cuda).unwrap();
///     let mut x = cuda.buffer(4 * 4).unwrap();
///     let mut y = cuda.buffer(4 * 4).unwrap();
///     x.write(&[-2.0f32, -0.0, 0.5, 3.0]).unwrap();
///     kernels.run(ElementwiseOp::Relu, &x, None, &mut y, 4, 0.0).unwrap();
///     let mut out = [1.0f32; 4];
///     y.read(&mut out).unwrap();
///     assert_eq!(out.map(f32::to_bits), [0.0f32, 0.0, 0.5, 3.0].map(f32::to_bits));
/// }
/// ```
pub struct Elementwise<'c> {
    /// The compiled kernels.
    module: Module<'c>,
    /// Each operation's kernel handle, in [`ElementwiseOp::ALL`]'s order.
    functions: [CuFunction; 15],
}

impl fmt::Debug for Elementwise<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Elementwise").finish_non_exhaustive()
    }
}

impl<'c> Elementwise<'c> {
    /// Compiles the kernels for `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses them.
    pub fn new(cuda: &'c Cuda) -> Result<Elementwise<'c>, CudaError> {
        let module = cuda.compile("elementwise.cu", SOURCE)?;
        let mut functions = [std::ptr::null_mut(); 15];
        for (function, op) in functions.iter_mut().zip(ElementwiseOp::ALL) {
            *function = module.function(op.kernel())?;
        }
        Ok(Elementwise { module, functions })
    }

    /// Writes `op` of the first `n` elements of `a` (and `b`) to `out`, with `alpha` for
    /// `Scale` and the leaky ReLU (ignored by the others). Every one of the `n` outputs is
    /// written. `b` must be given exactly when [`ElementwiseOp::takes_b`] says so.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `b` is missing or given against `op`, or `n` does not fit a
    /// `u32`; [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer holds fewer than `n` floats.
    pub fn run(
        &self,
        op: ElementwiseOp,
        a: &DeviceBuffer<'_>,
        b: Option<&DeviceBuffer<'_>>,
        out: &mut DeviceBuffer<'_>,
        n: usize,
        alpha: f32,
    ) -> Result<(), CudaError> {
        if b.is_some() != op.takes_b() {
            return Err(CudaError::Invalid(format!(
                "{op:?} {} a second input",
                if op.takes_b() { "needs" } else { "takes no" }
            )));
        }
        let count = u32::try_from(n)
            .map_err(|_| CudaError::Invalid(format!("{n} elements do not fit a u32")))?;
        let needs = n.saturating_mul(4);
        for (buffer, what) in [(a, "first input"), (&*out, "output")]
            .into_iter()
            .chain(b.map(|b| (b, "second input")))
        {
            assert!(
                buffer.bytes() >= needs,
                "the {what} buffer holds {} bytes, and {n} elements need {needs}",
                buffer.bytes()
            );
        }
        if n == 0 {
            return Ok(());
        }
        let index = ElementwiseOp::ALL
            .iter()
            .position(|&o| o == op)
            .expect("every operation is in ALL");
        let kernel = self.module.kernel_from(self.functions[index]);
        let blocks = count.div_ceil(BLOCK);
        // SAFETY: every kernel in the source takes (const float *a, const float *b, float
        // *out, unsigned n, float alpha) and touches elements below n only; a, b and out hold
        // at least n floats (checked above), b is a when the operation does not read it, and
        // out, borrowed mutably, is neither input.
        unsafe {
            kernel.launch(
                [blocks, 1, 1],
                [BLOCK, 1, 1],
                &[
                    Arg::Buffer(a),
                    Arg::Buffer(b.unwrap_or(a)),
                    Arg::Buffer(out),
                    Arg::U32(count),
                    Arg::F32(alpha),
                ],
            )
        }
    }
}
