//! Gated linear units: `glu` with a sigmoid, GELU, ReLU or SiLU gate.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::ops::activation::{
    GeluApprox, gelu_derivative, gelu_value, relu_derivative, sigmoid_value, silu_derivative,
    silu_value,
};
use crate::{Real, Tape};

/// The activation a gated linear unit applies to the second half of the last axis.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Gate, GeluApprox};
/// assert_ne!(Gate::Sigmoid, Gate::Gelu(GeluApprox::Exact));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    /// The logistic sigmoid: the original GLU, PyTorch's `nn.GLU`.
    Sigmoid,
    /// GELU in the given form: GeGLU.
    Gelu(GeluApprox),
    /// ReLU: ReGLU. NaN passes through the gate.
    Relu,
    /// SiLU: SwiGLU.
    Silu,
}

impl Gate {
    /// Replaces every value with the gate's value at it: the sigmoid, SiLU and the tanh form
    /// of GELU through [`Real::sigmoid_in_place`], [`Real::silu_in_place`] and
    /// [`Real::gelu_tanh_in_place`] (the SIMD kernels in f32), the others one value at a time.
    fn apply_in_place<T: Real>(self, values: &mut [T]) {
        match self {
            Gate::Sigmoid => T::sigmoid_in_place(values),
            Gate::Silu => T::silu_in_place(values),
            Gate::Gelu(GeluApprox::Tanh) => T::gelu_tanh_in_place(values),
            Gate::Gelu(GeluApprox::Exact) | Gate::Relu => {
                for v in values {
                    *v = self.value(*v);
                }
            }
        }
    }

    /// The gate's value at `x`.
    fn value<T: Real>(self, x: T) -> T {
        match self {
            Gate::Sigmoid => sigmoid_value(x),
            Gate::Gelu(approx) => gelu_value(x, approx),
            Gate::Relu => x.relu(),
            Gate::Silu => silu_value(x),
        }
    }

    /// The gate's derivative at `x`.
    fn derivative<T: Real>(self, x: T) -> T {
        match self {
            Gate::Sigmoid => {
                let s = sigmoid_value(x);
                s * (T::ONE - s)
            }
            Gate::Gelu(approx) => gelu_derivative(x, approx),
            Gate::Relu => relu_derivative(x),
            Gate::Silu => silu_derivative(x),
        }
    }
}

impl<T: Real> Tape<'_, T> {
    /// Records a gated linear unit: splits the last axis into a first half `a` and a second
    /// half `b` and returns `a ⊙ gate(b)`, with the last axis halved. With
    /// [`Gate::Sigmoid`] this is PyTorch's `F.glu(x, dim=-1)`; [`Gate::Gelu`], [`Gate::Relu`]
    /// and [`Gate::Silu`] give GeGLU, ReGLU and SwiGLU.
    ///
    /// The gradients are `g ⊙ gate(b)` for `a` and `g ⊙ a ⊙ gate′(b)` for `b`. NaN in either
    /// half reaches the output; NaN in `b` reaches both gradients.
    ///
    /// # Panics
    ///
    /// Panics if `x` has rank 0 or an odd last axis (the message names `glu` and the
    /// shape), or if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Gate, Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![2.0f32, 3.0, -1.0, 4.0], &[4]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.glu(x, Gate::Relu);
    /// assert_eq!(tape.value(y).data(), &[0.0, 12.0]);
    /// ```
    pub fn glu(&mut self, x: Var, gate: Gate) -> Var {
        let shape = self.shape(x);
        let dims = shape.dims();
        assert!(
            dims.last().is_some_and(|w| w % 2 == 0),
            "glu: last axis of shape {dims:?} must be even"
        );
        let width = dims[dims.len() - 1];
        let half = width / 2;
        let mut dims_out = [0usize; crate::Shape::MAX_RANK];
        dims_out[..dims.len()].copy_from_slice(dims);
        dims_out[dims.len() - 1] = half;
        let out_shape = crate::Shape::new(&dims_out[..dims.len()]);
        let mut out = self.take_buffer(out_shape.len());
        if half > 0 {
            for (o, row) in out
                .chunks_exact_mut(half)
                .zip(self.data(x).chunks_exact(width))
            {
                let (a, b) = row.split_at(half);
                o.copy_from_slice(b);
                gate.apply_in_place(o);
                for (o, &a) in o.iter_mut().zip(a) {
                    *o = a * *o;
                }
            }
        }
        self.push(Op::Glu(x, gate), out_shape, out)
    }
}

/// Backward of `glu`: `a` gets `g ⊙ gate(b)` and `b` gets `g ⊙ a ⊙ gate′(b)`, both written
/// into the halves of `x`'s gradient buffer in one pass. Every element is written when the
/// slot overwrites. With a last axis of width 0 there is nothing to write, but the slot is
/// still contributed to, so a parameter's gradient is marked current, as every other
/// operation leaves it.
pub(super) fn backward<T: Real>(x: Var, gate: Gate, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    let width = ctx.shape(x).dims().last().copied().unwrap_or(0);
    let half = width / 2;
    let Some(slot) = ctx.slot(x) else { return };
    if half == 0 {
        slot.update_with(|_, _| {});
        return;
    }
    slot.update_with(|buf, beta| {
        let rows = buf.chunks_exact_mut(width).zip(input.chunks_exact(width));
        for ((d, row), g) in rows.zip(out_grad.chunks_exact(half)) {
            let (a, b) = row.split_at(half);
            let (da, db) = d.split_at_mut(half);
            for i in 0..half {
                let ga = g[i] * gate.value(b[i]);
                let gb = g[i] * a[i] * gate.derivative(b[i]);
                if beta == T::ZERO {
                    da[i] = ga;
                    db[i] = gb;
                } else {
                    da[i] += ga;
                    db[i] += gb;
                }
            }
        }
    });
}

/// Tests for the gated linear units.
#[cfg(test)]
mod tests {
    use super::Gate;
    use crate::{GeluApprox, Graph, ParamStore, Tensor};

    /// `glu([1, 2, 3, 4], Sigmoid)` is `[1·σ(3), 2·σ(4)]`.
    #[test]
    fn glu_by_hand() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[4]).unwrap();
        let x = tape.input(x.view());
        let y = tape.glu(x, Gate::Sigmoid);
        let got = tape.value(y).data();
        assert_eq!(tape.shape(y).dims(), &[2]);
        assert!((got[0] - 0.9525741268224334).abs() < 1e-15);
        assert!((got[1] - 1.964027580075817).abs() < 1e-15);
    }

    /// The other gates apply their activation to the second half.
    #[test]
    fn glu_other_gates_by_hand() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![2.0f64, 3.0, -1.0, 4.0], &[4]).unwrap();
        let x = tape.input(x.view());
        let relu = tape.glu(x, Gate::Relu);
        assert_eq!(tape.value(relu).data(), &[0.0, 12.0]);
        let silu = tape.glu(x, Gate::Silu);
        let s = |v: f64| v / (1.0 + (-v).exp());
        assert!((tape.value(silu).data()[0] - 2.0 * s(-1.0)).abs() < 1e-15);
        assert!((tape.value(silu).data()[1] - 3.0 * s(4.0)).abs() < 1e-15);
        let gelu = tape.glu(x, Gate::Gelu(GeluApprox::Exact));
        assert!((tape.value(gelu).data()[1] - 3.0 * 3.9998732).abs() < 1e-5);
    }

    /// The last axis is halved and the leading axes are kept.
    #[test]
    fn glu_halves_last_axis() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::<f32>::zeros(&[3, 6]).view());
        let y = tape.glu(x, Gate::Sigmoid);
        assert_eq!(tape.shape(y).dims(), &[3, 3]);
    }

    /// Each row is split on its own.
    #[test]
    fn glu_splits_each_row() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x =
            Tensor::from_vec(vec![1.0f32, 2.0, 5.0, -5.0, 3.0, 4.0, 5.0, -5.0], &[2, 4]).unwrap();
        let x = tape.input(x.view());
        let y = tape.glu(x, Gate::Relu);
        assert_eq!(tape.value(y).data(), &[5.0, 0.0, 15.0, 0.0]);
    }

    /// An odd last axis panics, naming the operation and the shape.
    #[test]
    #[should_panic(expected = "glu: last axis of shape [2, 5] must be even")]
    fn glu_odd_width_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::<f32>::zeros(&[2, 5]).view());
        tape.glu(x, Gate::Sigmoid);
    }

    /// The gradient of `x` (12 values in shape `dims`) after `uses` `glu`s of it, summed, as
    /// bits.
    fn glu_grad_bits(gate: Gate, uses: usize, dims: &[usize]) -> Vec<u32> {
        let values: Vec<f32> = (0..12).map(|i| (i as f32 - 5.5) * 0.37).collect();
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(values, dims).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.param(p);
        let mut loss = None;
        for _ in 0..uses {
            let y = tape.glu(x, gate);
            let s = tape.sum(y);
            loss = Some(loss.map_or(s, |l| tape.add(l, s)));
        }
        tape.backward(loss.unwrap(), &mut store.grads);
        store.grad(p).data().iter().map(|g| g.to_bits()).collect()
    }

    /// A second `glu` of the same input adds to its gradient: two uses give exactly twice
    /// the gradient of one, for every gate.
    #[test]
    fn glu_gradient_accumulates() {
        for gate in [
            Gate::Sigmoid,
            Gate::Gelu(GeluApprox::Exact),
            Gate::Relu,
            Gate::Silu,
        ] {
            let once = glu_grad_bits(gate, 1, &[2, 6]);
            let twice = glu_grad_bits(gate, 2, &[2, 6]);
            let doubled: Vec<u32> = once
                .iter()
                .map(|&b| (2.0 * f32::from_bits(b)).to_bits())
                .collect();
            assert_eq!(twice, doubled, "{gate:?}");
        }
    }

    /// A rank-3 input is split along its last axis, row by row: its values and gradient
    /// are those of the same data as rank 2.
    #[test]
    fn glu_rank_three_matches_rank_two() {
        let values: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.21).collect();
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x3 = tape.input(Tensor::from_vec(values.clone(), &[2, 3, 4]).unwrap().view());
        let x2 = tape.input(Tensor::from_vec(values, &[6, 4]).unwrap().view());
        let (y3, y2) = (tape.glu(x3, Gate::Sigmoid), tape.glu(x2, Gate::Sigmoid));
        assert_eq!(tape.shape(y3).dims(), &[2, 3, 2]);
        assert_eq!(tape.value(y3).data(), tape.value(y2).data());
        drop(tape);
        for gate in [Gate::Sigmoid, Gate::Silu] {
            assert_eq!(
                glu_grad_bits(gate, 1, &[2, 3, 2]),
                glu_grad_bits(gate, 1, &[6, 2])
            );
        }
    }

    /// With a last axis of width 0 the output is empty and the backward pass still marks
    /// the parameter's gradient current.
    #[test]
    fn glu_of_width_zero_marks_the_gradient() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::<f32>::zeros(&[3, 0]));
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.param(p);
        let y = tape.glu(x, Gate::Sigmoid);
        assert_eq!(tape.shape(y).dims(), &[3, 0]);
        let loss = tape.sum(y);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert!(!store.grads.is_stale(p));
    }

    /// NaN in either half reaches the output and the gradients, also through a ReLU gate.
    #[test]
    fn glu_propagates_nan() {
        for gate in [
            Gate::Sigmoid,
            Gate::Gelu(GeluApprox::Tanh),
            Gate::Relu,
            Gate::Silu,
        ] {
            for (a, b) in [
                (f32::NAN, 1.0f32),
                (1.0, f32::NAN),
                (f32::NAN, -1.0),
                (-1.0, f32::NAN),
            ] {
                let mut store = ParamStore::new();
                let p = store.add(Tensor::from_vec(vec![a, b], &[2]).unwrap());
                let mut graph = Graph::new();
                let mut tape = graph.tape(&store.values);
                let x = tape.param(p);
                let y = tape.glu(x, gate);
                assert!(tape.value(y).data()[0].is_nan(), "{gate:?} {a} {b}");
                let loss = tape.sum(y);
                tape.backward(loss, &mut store.grads);
                let g = store.grad(p).data();
                assert!(g[1].is_nan(), "{gate:?} {a} {b}: {g:?}");
                if b.is_nan() {
                    assert!(g[0].is_nan(), "{gate:?} {a} {b}: {g:?}");
                }
            }
        }
    }
}
