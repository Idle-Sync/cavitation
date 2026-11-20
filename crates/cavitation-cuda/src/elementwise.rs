//! The element-wise operations as this crate's own CUDA kernels: add, multiply, scale and
//! the activations (ReLU, leaky ReLU, tanh, the sigmoid, GELU in both forms, ELU, SELU,
//! CELU, SiLU, Mish, softplus, softsign, hardtanh, the hard sigmoid and hard swish, and
//! log-sigmoid), forward and backward, each the twin of its CPU code.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `elementwise.wgsl`, compiled after
/// [`ACTIVATION`].
const SOURCE: &str = include_str!("elementwise.cu");

/// The scalar functions the element-wise and the GLU kernels share, a port of
/// `cavitation-gpu`'s `activation.wgsl`: each module compiles its own text after this one.
pub(crate) const ACTIVATION: &str = include_str!("activation.cu");

/// Threads per block.
const BLOCK: u32 = 256;

/// One element-wise operation, applied to every element of `a` (and `b`).
///
/// The one-rounding operations (`Add`, `Mul`, `Scale`, the ReLUs forward and backward), the clamps
/// (hardtanh, the hard sigmoid and hard swish) and softsign, forward and backward, give the CPU's
/// bits, ±0, ±∞ and subnormals included, and NaN for NaN. tanh, the sigmoid, GELU's tanh form, ELU,
/// SELU, CELU, SiLU, Mish, softplus and log-sigmoid give the bits of the CPU's reference kernels
/// (`Level::Reference`; the CPU's SIMD kernels fuse operations and may differ from these by an
/// ulp). GELU's erf form uses CUDA's `expf`, and the backward passes the CPU takes with std's `exp`
/// (ELU, SELU, CELU, SiLU, Mish, softplus, log-sigmoid) use the kernels' exponential; each is
/// within a stated bound of the f64 tape. `tests/elementwise.rs` checks each.
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
    /// ELU with `alpha`: `a` above 0, `alpha · (e^a − 1)` at or below.
    Elu,
    /// `b · (1 above 0, alpha · e^a at or below)`, with `a` the forward input.
    EluBackward,
    /// SELU, `scale · elu(a)` with PyTorch's `scale` and `alpha`.
    Selu,
    /// `b · (scale above 0, scale · alpha · e^a at or below)`, with `a` the forward input.
    SeluBackward,
    /// CELU with `alpha` (positive): `a` above 0, `alpha · (e^(a/alpha) − 1)` at or below.
    Celu,
    /// `b · (1 above 0, e^(a/alpha) at or below)`, with `a` the forward input.
    CeluBackward,
    /// SiLU, `a · σ(a)`.
    Silu,
    /// `b · σ(a) · (1 + a · (1 − σ(a)))`, with `a` the forward input.
    SiluBackward,
    /// Mish, `a · tanh(softplus(a))`, softplus at `beta` 1 and threshold 20.
    Mish,
    /// `b · (t + a · σ(a) · (1 − t²))`, `t = tanh(softplus(a))`, with `a` the forward input.
    MishBackward,
    /// Softplus with `beta` in `alpha` and the threshold in `scale`: `a` where `beta · a` is above
    /// the threshold, `ln(1 + e^(beta·a)) / beta` at or below.
    Softplus,
    /// `b · (1 where beta · a is above the threshold, σ(beta · a) at or below)`, with `a` the
    /// forward input.
    SoftplusBackward,
    /// Softsign, `a / (1 + |a|)`.
    Softsign,
    /// `b · 1 / (1 + |a|)²`, with `a` the forward input.
    SoftsignBackward,
    /// Hardtanh: `a` clamped to `[alpha, scale]` (ReLU6 is `[0, 6]`), NaN for NaN.
    Hardtanh,
    /// `b · (1 strictly inside (alpha, scale), 0 at and beyond the bounds)`, NaN for a NaN `a`.
    HardtanhBackward,
    /// The hard sigmoid: 0 up to −3, 1 from 3, `a / 6 + 1/2` between.
    Hardsigmoid,
    /// `b · (1/6 strictly inside (−3, 3), 0 elsewhere)`, NaN for a NaN `a`.
    HardsigmoidBackward,
    /// The hard swish: 0 up to −3, `a` from 3, `a · (a + 3) / 6` between.
    Hardswish,
    /// `b · (0 below −3, 1 above 3, a / 3 + 1/2 on [−3, 3])`, NaN for a NaN `a`.
    HardswishBackward,
    /// Log-sigmoid, `ln σ(a)`, as `0 − softplus(−a)`.
    LogSigmoid,
    /// `b · σ(−a)`, with `a` the forward input.
    LogSigmoidBackward,
}

impl ElementwiseOp {
    /// Every operation, each once.
    pub const ALL: &'static [ElementwiseOp] = &[
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
        ElementwiseOp::Elu,
        ElementwiseOp::EluBackward,
        ElementwiseOp::Selu,
        ElementwiseOp::SeluBackward,
        ElementwiseOp::Celu,
        ElementwiseOp::CeluBackward,
        ElementwiseOp::Silu,
        ElementwiseOp::SiluBackward,
        ElementwiseOp::Mish,
        ElementwiseOp::MishBackward,
        ElementwiseOp::Softplus,
        ElementwiseOp::SoftplusBackward,
        ElementwiseOp::Softsign,
        ElementwiseOp::SoftsignBackward,
        ElementwiseOp::Hardtanh,
        ElementwiseOp::HardtanhBackward,
        ElementwiseOp::Hardsigmoid,
        ElementwiseOp::HardsigmoidBackward,
        ElementwiseOp::Hardswish,
        ElementwiseOp::HardswishBackward,
        ElementwiseOp::LogSigmoid,
        ElementwiseOp::LogSigmoidBackward,
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
            ElementwiseOp::Elu => "elu",
            ElementwiseOp::EluBackward => "elu_backward",
            ElementwiseOp::Selu => "selu",
            ElementwiseOp::SeluBackward => "selu_backward",
            ElementwiseOp::Celu => "celu",
            ElementwiseOp::CeluBackward => "celu_backward",
            ElementwiseOp::Silu => "silu",
            ElementwiseOp::SiluBackward => "silu_backward",
            ElementwiseOp::Mish => "mish",
            ElementwiseOp::MishBackward => "mish_backward",
            ElementwiseOp::Softplus => "softplus",
            ElementwiseOp::SoftplusBackward => "softplus_backward",
            ElementwiseOp::Softsign => "softsign",
            ElementwiseOp::SoftsignBackward => "softsign_backward",
            ElementwiseOp::Hardtanh => "hardtanh",
            ElementwiseOp::HardtanhBackward => "hardtanh_backward",
            ElementwiseOp::Hardsigmoid => "hardsigmoid",
            ElementwiseOp::HardsigmoidBackward => "hardsigmoid_backward",
            ElementwiseOp::Hardswish => "hardswish",
            ElementwiseOp::HardswishBackward => "hardswish_backward",
            ElementwiseOp::LogSigmoid => "log_sigmoid",
            ElementwiseOp::LogSigmoidBackward => "log_sigmoid_backward",
        }
    }

    /// Whether the operation reads `b`: the binary operations and every backward pass.
    pub fn takes_b(self) -> bool {
        !matches!(
            self,
            ElementwiseOp::Scale
                | ElementwiseOp::Relu
                | ElementwiseOp::LeakyRelu
                | ElementwiseOp::Tanh
                | ElementwiseOp::Sigmoid
                | ElementwiseOp::GeluTanh
                | ElementwiseOp::GeluErf
                | ElementwiseOp::Elu
                | ElementwiseOp::Selu
                | ElementwiseOp::Celu
                | ElementwiseOp::Silu
                | ElementwiseOp::Mish
                | ElementwiseOp::Softplus
                | ElementwiseOp::Softsign
                | ElementwiseOp::Hardtanh
                | ElementwiseOp::Hardsigmoid
                | ElementwiseOp::Hardswish
                | ElementwiseOp::LogSigmoid
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
    /// Each operation's kernel handle, in [`ElementwiseOp::ALL`]'s order, which is the
    /// declaration order.
    functions: [CuFunction; ElementwiseOp::ALL.len()],
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
        let module = cuda.compile("elementwise.cu", &format!("{ACTIVATION}{SOURCE}"))?;
        let mut functions = [std::ptr::null_mut(); ElementwiseOp::ALL.len()];
        for (function, &op) in functions.iter_mut().zip(ElementwiseOp::ALL) {
            *function = module.function(op.kernel())?;
        }
        Ok(Elementwise { module, functions })
    }

    /// Writes `op` of the first `n` elements of `a` (and `b`) to `out`, with `alpha` for
    /// `Scale`, the leaky ReLU, ELU and CELU, as `beta` for softplus and as the lower bound for
    /// hardtanh (the others ignore it). It is [`run_with`] with a `scale` of 0, which softplus
    /// reads as its threshold and hardtanh as its upper bound: run those two, forward and
    /// backward, through [`run_with`]. Every one of the `n` outputs is written. `b` must be
    /// given exactly when [`ElementwiseOp::takes_b`] says so.
    ///
    /// [`run_with`]: Elementwise::run_with
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
        self.run_with(op, a, b, out, n, alpha, 0.0)
    }

    /// [`run`](Elementwise::run) with both of the operation's settings: `alpha` (for `Scale`,
    /// the leaky ReLU, ELU and CELU; `beta` for softplus; the lower bound for hardtanh) and
    /// `scale` (the threshold for softplus; the upper bound for hardtanh).
    ///
    /// # Errors
    ///
    /// As [`run`](Elementwise::run).
    ///
    /// # Panics
    ///
    /// If a buffer holds fewer than `n` floats.
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
    ///     x.write(&[-2.0f32, 0.5, 7.0, f32::NAN]).unwrap();
    ///     // ReLU6: hardtanh between 0 and 6.
    ///     kernels.run_with(ElementwiseOp::Hardtanh, &x, None, &mut y, 4, 0.0, 6.0).unwrap();
    ///     let mut out = [1.0f32; 4];
    ///     y.read(&mut out).unwrap();
    ///     assert_eq!(&out[..3], &[0.0, 0.5, 6.0]);
    ///     assert!(out[3].is_nan());
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn run_with(
        &self,
        op: ElementwiseOp,
        a: &DeviceBuffer<'_>,
        b: Option<&DeviceBuffer<'_>>,
        out: &mut DeviceBuffer<'_>,
        n: usize,
        alpha: f32,
        scale: f32,
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
        // SAFETY: the handles were looked up in this module when it was compiled, in `ALL`'s
        // order, which is the declaration order (a unit test holds it).
        let kernel = unsafe { self.module.kernel_from(self.functions[op as usize]) };
        let blocks = count.div_ceil(BLOCK);
        // SAFETY: every kernel in the source takes (const float *a, const float *b, float *out,
        // unsigned n, float alpha, float scale) and touches elements below n only; a, b and out
        // hold at least n floats (checked above), b is a when the operation does not read it, and
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
                    Arg::F32(scale),
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
        for (i, &op) in ElementwiseOp::ALL.iter().enumerate() {
            assert_eq!(op as usize, i, "{op:?}");
        }
    }
}
