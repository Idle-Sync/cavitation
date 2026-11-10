//! A test-only operation with a deliberately wrong backward pass, for checking that the
//! gradient check catches wrong gradients.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Tape};

impl<T: Real> Tape<'_, T> {
    /// Records the identity of `a`, whose backward pass returns twice the true gradient.
    pub(crate) fn wrong_identity(&mut self, a: Var) -> Var {
        let shape = self.shape(a);
        let mut out = self.take_buffer(shape.len());
        out.copy_from_slice(self.data(a));
        self.push(Op::WrongIdentity(a), shape, out)
    }

    /// Records the identity of `a`, whose backward pass returns infinity.
    pub(crate) fn infinite_grad(&mut self, a: Var) -> Var {
        let shape = self.shape(a);
        let mut out = self.take_buffer(shape.len());
        out.copy_from_slice(self.data(a));
        self.push(Op::InfiniteGrad(a), shape, out)
    }

    /// Records `a` where it equals 1 and infinity elsewhere (so any perturbation of an
    /// input equal to 1 makes the output infinite).
    pub(crate) fn blow_up(&mut self, a: Var) -> Var {
        let shape = self.shape(a);
        let mut out = self.take_buffer(shape.len());
        for (o, &x) in out.iter_mut().zip(self.data(a)) {
            *o = if x == T::ONE { x } else { T::ONE / T::ZERO };
        }
        self.push(Op::BlowUp(a), shape, out)
    }
}

/// Backward of the wrong identity: twice the output gradient.
pub(super) fn backward<T: Real>(a: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    if let Some(slot) = ctx.slot(a) {
        let doubled: Vec<T> = out_grad.iter().map(|&g| g + g).collect();
        slot.add_from(&doubled);
    }
}

/// Backward of the infinite-gradient identity: infinity for every element.
pub(super) fn infinite_backward<T: Real>(a: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    if let Some(slot) = ctx.slot(a) {
        let inf: Vec<T> = out_grad.iter().map(|_| T::ONE / T::ZERO).collect();
        slot.add_from(&inf);
    }
}

/// Backward of the blow-up op: the output gradient passes unchanged.
pub(super) fn blow_up_backward<T: Real>(a: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    if let Some(slot) = ctx.slot(a) {
        slot.add_from(out_grad);
    }
}
