//! Reductions: `sum`, `sum_axis`, `mean` and `mean_axis`.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// Splits `shape` around `axis` into `(outer, n, inner)`: the product of the dimensions
/// before the axis, the axis length, and the product of the dimensions after it. Element
/// `[o, k, j]` of the value sits at `(o * n + k) * inner + j`.
///
/// # Panics
///
/// Panics, naming the operation, the axis and the shape, if `axis` is not a dimension of
/// `shape`.
fn split_axis(name: &str, shape: &Shape, axis: usize) -> (usize, usize, usize) {
    if axis >= shape.rank() {
        panic!(
            "{name}: axis {axis} is out of range for shape {:?}",
            shape.dims()
        );
    }
    let dims = shape.dims();
    let outer = dims[..axis].iter().product();
    let inner = dims[axis + 1..].iter().product();
    (outer, dims[axis], inner)
}

/// `n` as a `T`, for averaging.
fn count<T: Real>(n: usize) -> T {
    T::from_f64(n as f64)
}

/// Writes into `out` (`outer * inner` elements) the sum over the axis of `x`, divided by
/// `divisor`. Every element of `out` is written.
fn reduce_into<T: Real>(
    x: &[T],
    (outer, n, inner): (usize, usize, usize),
    divisor: T,
    out: &mut [T],
) {
    debug_assert!(x.len() == outer * n * inner && out.len() == outer * inner);
    for (o, row) in out.chunks_exact_mut(inner.max(1)).enumerate() {
        row.fill(T::ZERO);
        for k in 0..n {
            let src = &x[(o * n + k) * inner..][..inner];
            for (d, &s) in row.iter_mut().zip(src) {
                *d += s;
            }
        }
        if divisor != T::ONE {
            for d in row.iter_mut() {
                *d = *d / divisor;
            }
        }
    }
}

impl<T: Real> Tape<'_, T> {
    /// Records the sum of every element of `x`, a rank-0 value.
    ///
    /// # Panics
    ///
    /// Panics if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
    /// let x = tape.input(x.view());
    /// let s = tape.sum(x);
    /// assert_eq!(tape.value(s).data(), &[6.0]);
    /// assert_eq!(tape.shape(s).rank(), 0);
    /// ```
    pub fn sum(&mut self, x: Var) -> Var {
        let len = self.shape(x).len();
        let mut out = self.take_buffer(1);
        reduce_into(self.data(x), (1, len, 1), T::ONE, &mut out);
        self.push(Op::Sum(x), Shape::new(&[]), out)
    }

    /// Records the mean of every element of `x`, a rank-0 value.
    ///
    /// # Panics
    ///
    /// Panics ("mean of empty ...") if `x` has no elements, or if `x` was not recorded on
    /// this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 6.0], &[2, 2]).unwrap();
    /// let x = tape.input(x.view());
    /// let m = tape.mean(x);
    /// assert_eq!(tape.value(m).data(), &[3.0]);
    /// ```
    pub fn mean(&mut self, x: Var) -> Var {
        let shape = self.shape(x);
        let len = shape.len();
        assert!(
            len > 0,
            "mean: mean of empty tensor with shape {:?}",
            shape.dims()
        );
        let mut out = self.take_buffer(1);
        reduce_into(self.data(x), (1, len, 1), count(len), &mut out);
        self.push(Op::Mean(x), Shape::new(&[]), out)
    }

    /// Records the sum of `x` along `axis`. With `keep_dim` the axis stays in the result
    /// with length 1; without it the axis is removed.
    ///
    /// # Panics
    ///
    /// Panics, naming `sum_axis`, the axis and the shape, if `axis` is not a dimension of
    /// `x`; or if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let x = tape.input(x.view());
    /// let s = tape.sum_axis(x, 0, false);
    /// assert_eq!(tape.value(s).data(), &[4.0, 6.0]);
    /// assert_eq!(tape.shape(s).dims(), &[2]);
    /// ```
    pub fn sum_axis(&mut self, x: Var, axis: usize, keep_dim: bool) -> Var {
        self.reduce_axis("sum_axis", x, axis, keep_dim, false)
    }

    /// Records the mean of `x` along `axis`. With `keep_dim` the axis stays in the result
    /// with length 1; without it the axis is removed.
    ///
    /// # Panics
    ///
    /// Panics, naming `mean_axis`, the axis and the shape, if `axis` is not a dimension of
    /// `x`; panics ("mean of empty ...") if that axis has length 0; or if `x` was not
    /// recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let x = tape.input(x.view());
    /// let m = tape.mean_axis(x, 1, true);
    /// assert_eq!(tape.value(m).data(), &[1.5, 3.5]);
    /// assert_eq!(tape.shape(m).dims(), &[2, 1]);
    /// ```
    pub fn mean_axis(&mut self, x: Var, axis: usize, keep_dim: bool) -> Var {
        self.reduce_axis("mean_axis", x, axis, keep_dim, true)
    }

    /// Records the sum (or mean) of `x` along `axis`.
    fn reduce_axis(&mut self, name: &str, x: Var, axis: usize, keep_dim: bool, mean: bool) -> Var {
        let shape = self.shape(x);
        let (outer, n, inner) = split_axis(name, &shape, axis);
        assert!(
            !mean || n > 0,
            "{name}: mean of empty axis {axis} of shape {:?}",
            shape.dims()
        );
        let mut dims = [0usize; Shape::MAX_RANK];
        let mut rank = 0;
        for (d, &len) in shape.dims().iter().enumerate() {
            if d != axis {
                dims[rank] = len;
                rank += 1;
            } else if keep_dim {
                dims[rank] = 1;
                rank += 1;
            }
        }
        let out_shape = Shape::new(&dims[..rank]);
        let mut out = self.take_buffer(out_shape.len());
        let divisor = if mean { count(n) } else { T::ONE };
        reduce_into(self.data(x), (outer, n, inner), divisor, &mut out);
        let op = if mean {
            Op::MeanAxis(x, axis)
        } else {
            Op::SumAxis(x, axis)
        };
        self.push(op, out_shape, out)
    }
}

/// Backward of every reduction: each input element gets the gradient of the output element
/// it was reduced into, divided by `divisor` (1 for a sum, the count for a mean).
/// `(outer, n, inner)` is the split of the input around the reduced axis.
pub(super) fn backward<T: Real>(
    x: Var,
    (outer, n, inner): (usize, usize, usize),
    divisor: T,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    debug_assert_eq!(out_grad.len(), outer * inner);
    let Some(slot) = ctx.slot(x) else { return };
    slot.update_with(|buf, beta| {
        let overwrite = beta == T::ZERO;
        for o in 0..outer {
            for k in 0..n {
                let dst = &mut buf[(o * n + k) * inner..][..inner];
                let src = &out_grad[o * inner..][..inner];
                for (d, &g) in dst.iter_mut().zip(src) {
                    if overwrite {
                        *d = g / divisor;
                    } else {
                        *d += g / divisor;
                    }
                }
            }
        }
    });
}

/// Backward of `sum` and `mean`: every element gets the single output gradient (over the
/// count for a mean).
pub(super) fn full_backward<T: Real>(
    x: Var,
    mean: bool,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let len = ctx.shape(x).len();
    let divisor = if mean { count(len) } else { T::ONE };
    backward(x, (1, len, 1), divisor, out_grad, ctx);
}

/// Backward of `sum_axis` and `mean_axis`.
pub(super) fn axis_backward<T: Real>(
    x: Var,
    axis: usize,
    mean: bool,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let split = split_axis("backward", &ctx.shape(x), axis);
    let divisor = if mean { count(split.1) } else { T::ONE };
    backward(x, split, divisor, out_grad, ctx);
}

/// Tests for the reductions.
#[cfg(test)]
mod tests {
    use crate::{Graph, ParamStore, Tensor};

    /// A 2 x 2 input holding [[1, 2], [3, 4]].
    fn square() -> Tensor<f32> {
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap()
    }

    /// Summing axis 0 gives [4, 6]; axis 1 with keep_dim gives [[3], [7]].
    #[test]
    fn sum_axis_values() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = square();
        let x = tape.input(x.view());
        let s0 = tape.sum_axis(x, 0, false);
        assert_eq!(tape.value(s0).data(), &[4.0, 6.0]);
        assert_eq!(tape.shape(s0).dims(), &[2]);
        let s1 = tape.sum_axis(x, 1, true);
        assert_eq!(tape.value(s1).data(), &[3.0, 7.0]);
        assert_eq!(tape.shape(s1).dims(), &[2, 1]);
    }

    /// `sum` and `mean` give rank-0 results.
    #[test]
    fn sum_and_mean_are_scalars() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = square();
        let x = tape.input(x.view());
        let s = tape.sum(x);
        let m = tape.mean(x);
        assert_eq!(tape.value(s).data(), &[10.0]);
        assert_eq!(tape.value(m).data(), &[2.5]);
        assert_eq!(tape.shape(s).rank(), 0);
        assert_eq!(tape.shape(m).rank(), 0);
        let ma = tape.mean_axis(x, 1, false);
        assert_eq!(tape.value(ma).data(), &[1.5, 3.5]);
    }

    /// The mean of a tensor with no elements panics.
    #[test]
    #[should_panic(expected = "mean of empty")]
    fn mean_of_empty_axis_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(Vec::<f32>::new(), &[0, 3]).unwrap();
        let x = tape.input(x.view());
        tape.mean(x);
    }

    /// The mean along an axis of length 0 panics.
    #[test]
    #[should_panic(expected = "mean of empty")]
    fn mean_axis_of_empty_axis_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(Vec::<f32>::new(), &[0, 3]).unwrap();
        let x = tape.input(x.view());
        tape.mean_axis(x, 0, false);
    }

    /// An axis past the rank panics naming the operation, the axis and the shape.
    #[test]
    #[should_panic(expected = "sum_axis: axis 2 is out of range for shape [2, 2]")]
    fn axis_out_of_range_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = square();
        let x = tape.input(x.view());
        tape.sum_axis(x, 2, false);
    }

    /// The gradient of a mean is 1/n for every element.
    #[test]
    fn mean_backward_spreads_evenly() {
        let mut store = ParamStore::new();
        let p = store.add(square());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let m = tape.mean(v);
        tape.backward(m, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[0.25; 4]);
    }
}
