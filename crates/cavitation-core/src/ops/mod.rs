//! The operations a tape records, and the one place that dispatches their backward passes.
//!
//! Each operation's forward pass is a [`Tape`](crate::Tape) method in the operation's file.
//! Adding an operation means adding its variant to [`Op`], its arm to [`Op::backward`], and
//! its file's forward and backward functions.

mod elementwise;
mod matmul;
#[cfg(test)]
mod wrong;

use crate::Real;
use crate::graph::{BackwardCtx, ParamId, Var};
use elementwise::Kind;

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
    /// the node's inputs through `ctx`.
    pub(crate) fn backward<T: Real>(&self, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
        match *self {
            Op::Input | Op::Param(_) => {}
            Op::Add(a, b) => elementwise::binary_backward(Kind::Add, a, b, out_grad, ctx),
            Op::Sub(a, b) => elementwise::binary_backward(Kind::Sub, a, b, out_grad, ctx),
            Op::Mul(a, b) => elementwise::binary_backward(Kind::Mul, a, b, out_grad, ctx),
            Op::MatMul(a, b) => matmul::backward(false, a, b, out_grad, ctx),
            Op::MatMulNt(a, b) => matmul::backward(true, a, b, out_grad, ctx),
            Op::Scale(a, s) => elementwise::scale_backward(a, s, out_grad, ctx),
            #[cfg(test)]
            Op::WrongIdentity(a) => wrong::backward(a, out_grad, ctx),
            #[cfg(test)]
            Op::InfiniteGrad(a) => wrong::infinite_backward(a, out_grad, ctx),
            #[cfg(test)]
            Op::BlowUp(a) => wrong::blow_up_backward(a, out_grad, ctx),
        }
    }
}
