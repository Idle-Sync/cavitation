//! The operations a tape records, and the one place that dispatches their backward passes.
//!
//! Each operation's forward pass is a [`Tape`](crate::Tape) method in the operation's file.
//! Adding an operation means adding its variant to [`Op`], its arm to [`Op::backward`], and
//! its file's forward and backward functions.

mod elementwise;

use crate::Real;
use crate::graph::{BackwardCtx, ParamId, Var};

/// A recorded operation: what produced a node, with its inputs and anything its backward
/// pass needs, stored inline so that recording never allocates.
#[derive(Debug)]
pub(crate) enum Op {
    /// A value copied in from outside; takes no gradient.
    Input,
    /// A parameter read in place from the borrowed parameter values.
    Param(ParamId),
    /// Elementwise `a + b` of two values of the same shape.
    Add(Var, Var),
}

impl Op {
    /// Passes `out_grad`, the gradient of the loss with respect to this node's value, back to
    /// the node's inputs through `ctx`.
    pub(crate) fn backward<T: Real>(&self, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
        match *self {
            Op::Input | Op::Param(_) => {}
            Op::Add(a, b) => elementwise::add_backward(a, b, out_grad, ctx),
        }
    }
}
