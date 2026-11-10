//! Elementwise operations: `add`.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Tape};

impl<T: Real> Tape<'_, T> {
    /// Records `a + b`, element by element.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` have different shapes (the message names the operation and both
    /// shapes), or if either was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
    /// let b = Tensor::from_vec(vec![3.0f32, 4.0], &[2]).unwrap();
    /// let (a, b) = (tape.input(a.view()), tape.input(b.view()));
    /// let c = tape.add(a, b);
    /// assert_eq!(tape.value(c).data(), &[4.0, 6.0]);
    /// ```
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let (sa, sb) = (self.shape(a), self.shape(b));
        assert!(
            sa == sb,
            "add: shapes {:?} and {:?} must be equal",
            sa.dims(),
            sb.dims()
        );
        let mut out = self.take_buffer(sa.len());
        for ((o, &x), &y) in out.iter_mut().zip(self.data(a)).zip(self.data(b)) {
            *o = x + y;
        }
        self.push(Op::Add(a, b), sa, out)
    }
}

/// Backward of `a + b`: the output gradient passes unchanged to both inputs.
pub(super) fn add_backward<T: Real>(a: Var, b: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    for v in [a, b] {
        if let Some(slot) = ctx.slot(v) {
            slot.add_from(out_grad);
        }
    }
}

/// Tests for the elementwise operations.
#[cfg(test)]
mod tests {
    use crate::{Graph, ParamStore, Tensor};

    /// [1, 2] + [3, 4] = [4, 6].
    #[test]
    fn add_forward_values() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0], &[2]).unwrap();
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        let c = tape.add(a, b);
        assert_eq!(tape.value(c).data(), &[4.0, 6.0]);
        assert_eq!(tape.shape(c).dims(), &[2]);
    }

    /// d(p1 + p2)/dp1 = d(p1 + p2)/dp2 = 1.
    #[test]
    fn add_backward_to_both_params() {
        let mut store = ParamStore::new();
        let p1 = store.add(Tensor::from_vec(vec![2.0f32], &[1]).unwrap());
        let p2 = store.add(Tensor::from_vec(vec![5.0f32], &[1]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let (a, b) = (tape.param(p1), tape.param(p2));
        let loss = tape.add(a, b);
        assert_eq!(tape.value(loss).data(), &[7.0]);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(p1).data(), &[1.0]);
        assert_eq!(store.grad(p2).data(), &[1.0]);
    }

    /// Both uses of one variable contribute: d(p + p)/dp = 2.
    #[test]
    fn same_var_used_twice() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![3.0f32], &[1]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let loss = tape.add(v, v);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[2.0]);
    }

    /// Adding a [2] to a [3] panics naming the operation and both shapes.
    #[test]
    #[should_panic(expected = "add: shapes [2] and [3]")]
    fn add_shape_mismatch_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::zeros(&[2]);
        let b = Tensor::zeros(&[3]);
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        tape.add(a, b);
    }
}
