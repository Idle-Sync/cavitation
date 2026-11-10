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
}

/// Backward of the wrong identity: twice the output gradient.
pub(super) fn backward<T: Real>(a: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    if let Some(slot) = ctx.slot(a) {
        let doubled: Vec<T> = out_grad.iter().map(|&g| g + g).collect();
        slot.add_from(&doubled);
    }
}
