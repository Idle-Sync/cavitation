//! The operations a tape records, and the one place that dispatches their backward passes.
//!
//! Each operation's forward pass is a [`Tape`](crate::Tape) method in the operation's file.
//! Adding an operation means adding its variant to [`Op`], its arm to [`Op::backward`], and
//! its file's forward and backward functions.

mod activation;
mod elementwise;
mod erf;
mod glu;
mod matmul;
mod reduce;
#[cfg(test)]
mod wrong;

use crate::Real;
use crate::graph::{BackwardCtx, ParamId, Var};
pub use activation::GeluApprox;
use elementwise::Kind;
pub(crate) use erf::{erf_f32, erf_f64};
pub use glu::Gate;

/// A recorded operation: what produced a node, with its inputs and anything its backward
/// pass needs, stored inline so that recording never allocates.
#[derive(Debug)]
pub(crate) enum Op {
    /// A value copied in from outside; takes no gradient.
    Input,
    /// A parameter read in place from the borrowed parameter values.
    Param(ParamId),
    /// Elementwise `a + b`, broadcasting.
    Add(Var, Var),
    /// Elementwise `a - b`, broadcasting.
    Sub(Var, Var),
    /// Elementwise `a * b`, broadcasting.
    Mul(Var, Var),
    /// Matrix product `a · b` of `[m, k]` and `[k, n]`.
    MatMul(Var, Var),
    /// Matrix product `a · bᵀ` of `[m, k]` and `[n, k]`.
    MatMulNt(Var, Var),
    /// Every element of the value times a scalar, kept as `f64` (exact for f32 and f64).
    Scale(Var, f64),
    /// The sum of every element, a rank-0 value.
    Sum(Var),
    /// The mean of every element, a rank-0 value.
    Mean(Var),
    /// The sum along an axis.
    SumAxis(Var, usize),
    /// The mean along an axis.
    MeanAxis(Var, usize),
    /// `max(x, 0)`.
    Relu(Var),
    /// `x` where positive, slope times `x` elsewhere; the slope is kept as `f64`.
    LeakyRelu(Var, f64),
    /// The hyperbolic tangent.
    Tanh(Var),
    /// The logistic sigmoid.
    Sigmoid(Var),
    /// GELU in the given form.
    Gelu(Var, GeluApprox),
    /// ELU with the given alpha (kept as `f64`).
    Elu(Var, f64),
    /// SELU.
    Selu(Var),
    /// CELU with the given alpha (kept as `f64`).
    Celu(Var, f64),
    /// SiLU, `x · σ(x)`.
    Silu(Var),
    /// Mish, `x · tanh(softplus(x))`.
    Mish(Var),
    /// Softplus with the given `beta` and threshold (kept as `f64`).
    Softplus(Var, f64, f64),
    /// Softsign, `x / (1 + |x|)`.
    Softsign(Var),
    /// Clamp to the given minimum and maximum (kept as `f64`); also ReLU6.
    Hardtanh(Var, f64, f64),
    /// Hard sigmoid.
    HardSigmoid(Var),
    /// Hard swish.
    HardSwish(Var),
    /// `log σ(x)`.
    LogSigmoid(Var),
    /// PReLU: the input, then the slope (shape `[1]` or `[C]`).
    Prelu(Var, Var),
    /// A gated linear unit: the last axis split into value and gate halves.
    Glu(Var, Gate),
    /// Test only: the identity, whose backward pass returns twice the true gradient.
    #[cfg(test)]
    WrongIdentity(Var),
    /// Test only: the identity, whose backward pass returns infinity.
    #[cfg(test)]
    InfiniteGrad(Var),
    /// Test only: `x` where `x == 1`, infinity elsewhere; its backward pass is the identity's.
    #[cfg(test)]
    BlowUp(Var),
}

impl Op {
    /// Passes `out_grad`, the gradient of the loss with respect to this node's value, back to
    /// the node's inputs through `ctx`. `out` is the node itself, for operations whose
    /// derivative is cheapest from their own output (`tanh`, `sigmoid`).
    pub(crate) fn backward<T: Real>(&self, out: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
        match *self {
            Op::Input | Op::Param(_) => {}
            Op::Add(a, b) => elementwise::binary_backward(Kind::Add, a, b, out_grad, ctx),
            Op::Sub(a, b) => elementwise::binary_backward(Kind::Sub, a, b, out_grad, ctx),
            Op::Mul(a, b) => elementwise::binary_backward(Kind::Mul, a, b, out_grad, ctx),
            Op::MatMul(a, b) => matmul::backward(false, a, b, out_grad, ctx),
            Op::MatMulNt(a, b) => matmul::backward(true, a, b, out_grad, ctx),
            Op::Scale(a, s) => elementwise::scale_backward(a, s, out_grad, ctx),
            Op::Sum(a) => reduce::full_backward(a, false, out_grad, ctx),
            Op::Mean(a) => reduce::full_backward(a, true, out_grad, ctx),
            Op::SumAxis(a, axis) => reduce::axis_backward(a, axis, false, out_grad, ctx),
            Op::MeanAxis(a, axis) => reduce::axis_backward(a, axis, true, out_grad, ctx),
            Op::Relu(a) => activation::relu_backward(a, out_grad, ctx),
            Op::LeakyRelu(a, slope) => activation::leaky_relu_backward(a, slope, out_grad, ctx),
            Op::Tanh(a) => activation::tanh_backward(a, out, out_grad, ctx),
            Op::Sigmoid(a) => activation::sigmoid_backward(a, out, out_grad, ctx),
            Op::Gelu(a, approx) => activation::gelu_backward(a, approx, out_grad, ctx),
            Op::Elu(a, alpha) => activation::elu_backward(a, alpha, out_grad, ctx),
            Op::Selu(a) => activation::selu_backward(a, out_grad, ctx),
            Op::Celu(a, alpha) => activation::celu_backward(a, alpha, out_grad, ctx),
            Op::Silu(a) => activation::silu_backward(a, out_grad, ctx),
            Op::Mish(a) => activation::mish_backward(a, out_grad, ctx),
            Op::Softplus(a, beta, threshold) => {
                activation::softplus_backward(a, beta, threshold, out_grad, ctx)
            }
            Op::Softsign(a) => activation::softsign_backward(a, out_grad, ctx),
            Op::Hardtanh(a, lo, hi) => activation::hardtanh_backward(a, lo, hi, out_grad, ctx),
            Op::HardSigmoid(a) => activation::hardsigmoid_backward(a, out_grad, ctx),
            Op::HardSwish(a) => activation::hardswish_backward(a, out_grad, ctx),
            Op::LogSigmoid(a) => activation::log_sigmoid_backward(a, out_grad, ctx),
            Op::Prelu(a, slope) => activation::prelu_backward(a, slope, out_grad, ctx),
            Op::Glu(a, gate) => glu::backward(a, gate, out_grad, ctx),
            #[cfg(test)]
            Op::WrongIdentity(a) => wrong::backward(a, out_grad, ctx),
            #[cfg(test)]
            Op::InfiniteGrad(a) => wrong::infinite_backward(a, out_grad, ctx),
            #[cfg(test)]
            Op::BlowUp(a) => wrong::blow_up_backward(a, out_grad, ctx),
        }
    }
}
