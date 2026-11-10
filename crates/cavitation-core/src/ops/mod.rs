//! The operations a tape records, and the one place that dispatches their backward passes.
//!
//! Each operation's forward pass is a [`Tape`](crate::Tape) method in the operation's file.
//! Adding an operation means adding its variant to [`Op`], its arm to [`Op::backward`], and
//! its file's forward and backward functions. Operations defined outside the crate go
//! through [`CustomOp`] instead.

mod activation;
mod custom;
mod elementwise;
mod erf;
mod glu;
mod loss;
mod matmul;
mod reduce;
#[cfg(test)]
mod wrong;

use std::sync::Arc;

use crate::Real;
use crate::graph::{BackwardCtx, ParamId, Var};
pub use activation::GeluApprox;
pub use custom::CustomOp;
use elementwise::Kind;
pub(crate) use erf::{erf_f32, erf_f64};
pub use glu::Gate;
pub use loss::Reduction;

/// A recorded operation: what produced a node, with its inputs and anything its backward
/// pass needs, stored inline so that recording never allocates.
#[derive(Debug)]
pub(crate) enum Op<T: Real> {
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
    /// The input where the condition (broadcast to the input's shape) is non-zero, 0
    /// elsewhere.
    KeepWhere(Var, Var),
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
    /// Softmax over the last axis.
    Softmax(Var),
    /// Log-softmax over the last axis.
    LogSoftmax(Var),
    /// Softmax cross-entropy of `[batch, classes]` logits: the reduction, and where the
    /// batch's labels start in the graph's label buffer.
    SoftmaxCrossEntropy(Var, Reduction, usize),
    /// Mean squared error of prediction and target: the reduction.
    Mse(Var, Var, Reduction),
    /// Binary cross-entropy of probabilities and targets: the reduction.
    BinaryCrossEntropy(Var, Var, Reduction),
    /// Binary cross-entropy of logits and targets: the reduction.
    BceWithLogits(Var, Var, Reduction),
    /// Negative log-likelihood of `[batch, classes]` log-probabilities: the reduction, and
    /// where the batch's labels start in the graph's label buffer.
    Nll(Var, Reduction, usize),
    /// Mean absolute error of prediction and target: the reduction.
    L1(Var, Var, Reduction),
    /// Huber loss with the given delta (kept as `f64`): the reduction.
    Huber(Var, Var, f64, Reduction),
    /// Smooth L1 loss with the given beta (kept as `f64`): the reduction.
    SmoothL1(Var, Var, f64, Reduction),
    /// KL divergence of log-probabilities from target probabilities: the reduction.
    KlDiv(Var, Var, Reduction),
    /// A user-defined operation: the operation, its inputs (slots from the count on are
    /// unused) and the input count.
    Custom(
        Arc<dyn CustomOp<T> + Send + Sync>,
        [Var; custom::MAX_INPUTS],
        u8,
    ),
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

impl<T: Real> Op<T> {
    /// Passes `out_grad`, the gradient of the loss with respect to this node's value, back to
    /// the node's inputs through `ctx`. `out` is the node itself, for operations whose
    /// derivative is cheapest from their own output (`tanh`, `sigmoid`).
    pub(crate) fn backward(&self, out: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
        match *self {
            Op::Input | Op::Param(_) => {}
            Op::Add(a, b) => elementwise::binary_backward(Kind::Add, a, b, out_grad, ctx),
            Op::Sub(a, b) => elementwise::binary_backward(Kind::Sub, a, b, out_grad, ctx),
            Op::Mul(a, b) => elementwise::binary_backward(Kind::Mul, a, b, out_grad, ctx),
            Op::MatMul(a, b) => matmul::backward(false, a, b, out_grad, ctx),
            Op::MatMulNt(a, b) => matmul::backward(true, a, b, out_grad, ctx),
            Op::Scale(a, s) => elementwise::scale_backward(a, s, out_grad, ctx),
            Op::KeepWhere(x, keep) => elementwise::keep_where_backward(x, keep, out_grad, ctx),
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
            Op::Softmax(a) => loss::softmax_backward(a, out, out_grad, ctx),
            Op::LogSoftmax(a) => loss::log_softmax_backward(a, out, out_grad, ctx),
            Op::SoftmaxCrossEntropy(a, reduction, labels) => {
                loss::cross_entropy_backward(a, reduction, labels, out_grad, ctx)
            }
            Op::Mse(p, t, reduction) => loss::mse_backward(p, t, reduction, out_grad, ctx),
            Op::BinaryCrossEntropy(p, t, r) => loss::bce_backward(p, t, r, out_grad, ctx),
            Op::BceWithLogits(z, t, r) => loss::bce_logits_backward(z, t, r, out_grad, ctx),
            Op::Nll(a, reduction, labels) => {
                loss::nll_backward(a, reduction, labels, out_grad, ctx)
            }
            Op::L1(p, t, r) => loss::l1_backward(p, t, r, out_grad, ctx),
            Op::Huber(p, t, delta, r) => loss::huber_backward(p, t, delta, r, out_grad, ctx),
            Op::SmoothL1(p, t, beta, r) => loss::smooth_l1_backward(p, t, beta, r, out_grad, ctx),
            Op::KlDiv(x, t, r) => loss::kl_div_backward(x, t, r, out_grad, ctx),
            Op::Custom(ref op, ref inputs, count) => {
                custom::backward(&**op, &inputs[..usize::from(count)], out, out_grad, ctx)
            }
            #[cfg(test)]
            Op::WrongIdentity(a) => wrong::backward(a, out_grad, ctx),
            #[cfg(test)]
            Op::InfiniteGrad(a) => wrong::infinite_backward(a, out_grad, ctx),
            #[cfg(test)]
            Op::BlowUp(a) => wrong::blow_up_backward(a, out_grad, ctx),
        }
    }
}
