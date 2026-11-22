//! Operations that move values without arithmetic: `reshape`, `permute` and `transpose`.
//! Each output holds exactly its input's values, so the forward pass and the backward pass
//! are exact, the same bits on every kernel level.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

impl<T: Real> Tape<'_, T> {
    /// Records `x` with the new shape `dims`: the same values in the same row-major order,
    /// like PyTorch's `reshape` on a contiguous tensor. The gradient is the output's
    /// gradient with `x`'s shape. The value is copied, so it never aliases `x`. Every length
    /// is given: there is no `-1` to infer one.
    ///
    /// # Panics
    ///
    /// Panics, naming `reshape` and both shapes, if `dims` does not hold as many values as
    /// `x`; if `dims` has more than [`Shape::MAX_RANK`] dimensions; or if `x` was not
    /// recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap().view());
    /// let y = tape.reshape(x, &[3, 2]);
    /// assert_eq!(tape.shape(y).dims(), &[3, 2]);
    /// assert_eq!(tape.value(y).data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn reshape(&mut self, x: Var, dims: &[usize]) -> Var {
        let shape = self.shape(x);
        assert!(
            dims.len() <= Shape::MAX_RANK,
            "reshape: {dims:?} has more than {} dimensions",
            Shape::MAX_RANK
        );
        // Any length of 0 makes the count 0, however large the others.
        let count = if dims.contains(&0) {
            Some(0)
        } else {
            dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        };
        let wanted = count.filter(|&n| n == shape.len());
        if wanted.is_none() {
            panic!(
                "reshape: shape {:?} holds {} values, {dims:?} does not",
                shape.dims(),
                shape.len()
            );
        }
        let mut out = self.take_buffer(shape.len());
        out.copy_from_slice(self.data(x));
        self.push(Op::Reshape(x), Shape::new(dims), out)
    }

    /// Records `x` with its axes reordered: output axis `i` is input axis `axes[i]`, like
    /// PyTorch's `permute`. The gradient moves the output's gradient back to each value's
    /// place in `x`. Axes count from 0; there are no negative axes.
    ///
    /// # Panics
    ///
    /// Panics, naming `permute`, the axes and the shape, if `axes` is not an ordering of
    /// `0..rank` of `x`, or if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap().view());
    /// let y = tape.permute(x, &[1, 0]);
    /// assert_eq!(tape.shape(y).dims(), &[3, 2]);
    /// assert_eq!(tape.value(y).data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn permute(&mut self, x: Var, axes: &[usize]) -> Var {
        let shape = self.shape(x);
        let rank = shape.rank();
        let mut seen = [false; Shape::MAX_RANK];
        let valid = axes.len() == rank
            && axes
                .iter()
                .all(|&a| a < rank && !std::mem::replace(&mut seen[a], true));
        if !valid {
            panic!(
                "permute: {axes:?} is not an ordering of the {rank} axes of shape {:?}",
                shape.dims()
            );
        }
        let mut dims = [0usize; Shape::MAX_RANK];
        for (i, &a) in axes.iter().enumerate() {
            dims[i] = shape.dims()[a];
        }
        let out_shape = Shape::new(&dims[..rank]);
        let mut out = self.take_buffer(shape.len());
        permute_into(self.data(x), &shape, axes, &mut out, false);
        let start = self.push_labels(axes);
        self.push(Op::Permute(x, start), out_shape, out)
    }

    /// Records `x` with axes `a` and `b` swapped, like PyTorch's `transpose`: a
    /// [`permute`](Tape::permute) that exchanges two axes and keeps the rest in place. Axes
    /// count from 0; there are no negative axes. As in PyTorch, a rank-0 value takes
    /// `transpose(x, 0, 0)`, which changes nothing.
    ///
    /// # Panics
    ///
    /// Panics, naming `transpose`, the axes and the shape, if `a` or `b` is not an axis of
    /// `x` (or, for a rank-0 value, is not 0), or if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec((0..24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap().view());
    /// let y = tape.transpose(x, 0, 2);
    /// assert_eq!(tape.shape(y).dims(), &[4, 3, 2]);
    /// assert_eq!(&tape.value(y).data()[..4], &[0.0, 12.0, 4.0, 16.0]);
    /// ```
    pub fn transpose(&mut self, x: Var, a: usize, b: usize) -> Var {
        let shape = self.shape(x);
        let rank = shape.rank();
        if a >= rank.max(1) || b >= rank.max(1) {
            panic!(
                "transpose: axes {a} and {b} are not both axes of shape {:?}",
                shape.dims()
            );
        }
        let mut axes = [0usize; Shape::MAX_RANK];
        for (i, v) in axes.iter_mut().enumerate().take(rank) {
            *v = i;
        }
        if rank > 0 {
            axes.swap(a, b);
        }
        self.permute(x, &axes[..rank])
    }
}

/// Moves the values of `src` (shape `shape`) to `dst` with the axes reordered by `axes`, as
/// [`Tape::permute`] records it. With `back`, `src` is in the permuted order and the values
/// go to their places in `shape`'s order instead.
fn permute_into<T: Real>(src: &[T], shape: &Shape, axes: &[usize], dst: &mut [T], back: bool) {
    permute_with(src, shape, axes, dst, back, |d, s| *d = s);
}

/// [`permute_into`] with each value combined into `dst` by `put`.
fn permute_with<T: Real>(
    src: &[T],
    shape: &Shape,
    axes: &[usize],
    dst: &mut [T],
    back: bool,
    put: impl Fn(&mut T, T),
) {
    let rank = shape.rank();
    let dims = shape.dims();
    if shape.is_empty() {
        return;
    }
    // The input's row-major strides, then each output axis's stride in the input.
    let mut in_stride = [0usize; Shape::MAX_RANK];
    let mut s = 1;
    for d in (0..rank).rev() {
        in_stride[d] = s;
        s *= dims[d];
    }
    let mut out_dims = [1usize; Shape::MAX_RANK];
    let mut step = [0usize; Shape::MAX_RANK];
    for (i, &a) in axes.iter().enumerate() {
        out_dims[i] = dims[a];
        step[i] = in_stride[a];
    }
    // Walk the output in row-major order, carrying the matching input offset.
    let mut index = [0usize; Shape::MAX_RANK];
    let mut offset = 0usize;
    for o in 0..shape.len() {
        if back {
            put(&mut dst[offset], src[o]);
        } else {
            put(&mut dst[o], src[offset]);
        }
        for i in (0..rank).rev() {
            index[i] += 1;
            offset += step[i];
            if index[i] < out_dims[i] {
                break;
            }
            offset -= step[i] * out_dims[i];
            index[i] = 0;
        }
    }
}

/// Backward of `reshape`: the output's gradient, element for element, into `x`'s.
pub(super) fn reshape_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            if beta == T::ZERO {
                buf.copy_from_slice(out_grad);
            } else {
                for (d, &g) in buf.iter_mut().zip(out_grad) {
                    *d += g;
                }
            }
        });
    }
}

/// Backward of `permute`: each output gradient back to its value's place in `x`, through the
/// axes kept in the labels from `start`.
pub(super) fn permute_backward<T: Real>(
    x: Var,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let shape = ctx.shape(x);
    let axes = &ctx.labels()[start..start + shape.rank()];
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            if beta == T::ZERO {
                permute_into(out_grad, &shape, axes, buf, true);
            } else {
                permute_with(out_grad, &shape, axes, buf, true, |d, g| *d += g);
            }
        });
    }
}

/// Tests for the structural operations.
#[cfg(test)]
mod tests {
    use crate::{Graph, ParamStore, Tensor};

    /// A permutation of a rank-4 value puts every value where the index arithmetic says,
    /// and its backward pass sends each gradient back to the same place.
    #[test]
    fn permute_moves_every_value() {
        let dims = [2usize, 3, 4, 5];
        let n: usize = dims.iter().product();
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut params = ParamStore::new();
        let p = params.add(Tensor::from_vec(v, &dims).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.param(p);
        let axes = [2usize, 0, 3, 1];
        let y = tape.permute(x, &axes);
        assert_eq!(tape.shape(y).dims(), &[4, 2, 5, 3]);
        let out = tape.value(y).data().to_vec();
        let at = |i: [usize; 4]| ((i[0] * 3 + i[1]) * 4 + i[2]) * 5 + i[3];
        let mut k = 0;
        for a in 0..4 {
            for b in 0..2 {
                for c in 0..5 {
                    for d in 0..3 {
                        // Output [a, b, c, d] is input [b, d, a, c].
                        assert_eq!(out[k], at([b, d, a, c]) as f32);
                        k += 1;
                    }
                }
            }
        }
        // Weight each output by its own position: the gradient at input i is then the
        // position its value went to.
        let w: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let w = tape.input(Tensor::from_vec(w, &[4, 2, 5, 3]).unwrap().view());
        let weighted = tape.mul(y, w);
        let loss = tape.sum(weighted);
        tape.backward(loss, &mut params.grads);
        let g = params.grad(p).data();
        for (pos, &value) in out.iter().enumerate() {
            assert_eq!(g[value as usize], pos as f32);
        }
    }

    /// A gradient that reaches `x` through a reshape and another path adds both.
    #[test]
    fn reshape_adds_to_other_gradients() {
        let mut params = ParamStore::new();
        let p = params.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.param(p);
        let r = tape.reshape(x, &[4]);
        let t = tape.transpose(x, 0, 1);
        let a = tape.sum(r);
        let b = tape.sum(t);
        let both = tape.add(a, b);
        tape.backward(both, &mut params.grads);
        assert_eq!(params.grad(p).data(), &[2.0, 2.0, 2.0, 2.0]);
    }

    /// A shape with the wrong number of values is refused, naming both shapes.
    #[test]
    #[should_panic(expected = "reshape: shape [2, 3] holds 6 values, [4, 2] does not")]
    fn reshape_refuses_another_count() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(vec![0.0f32; 6], &[2, 3]).unwrap().view());
        tape.reshape(x, &[4, 2]);
    }

    /// Axes that repeat one, or leave one out, are refused.
    #[test]
    #[should_panic(expected = "permute: [0, 0] is not an ordering of the 2 axes")]
    fn permute_refuses_a_repeated_axis() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(vec![0.0f32; 6], &[2, 3]).unwrap().view());
        tape.permute(x, &[0, 0]);
    }

    /// An axis past the rank is refused.
    #[test]
    #[should_panic(expected = "transpose: axes 0 and 2 are not both axes of shape [2, 3]")]
    fn transpose_refuses_a_missing_axis() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(vec![0.0f32; 6], &[2, 3]).unwrap().view());
        tape.transpose(x, 0, 2);
    }

    /// A rank-0 value permutes with no axes, transposes with axes 0 and 0, and reshapes to
    /// and from one value, its gradient passing through each.
    #[test]
    fn rank_zero_values_pass_through() {
        let mut params = ParamStore::new();
        let p = params.add(Tensor::from_vec(vec![3.0f32], &[]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.param(p);
        let y = tape.permute(x, &[]);
        let z = tape.transpose(y, 0, 0);
        let one = tape.reshape(z, &[1, 1]);
        let back = tape.reshape(one, &[]);
        assert_eq!(tape.shape(back).rank(), 0);
        assert_eq!(tape.value(back).data(), &[3.0]);
        tape.backward(back, &mut params.grads);
        assert_eq!(params.grad(p).data(), &[1.0]);
    }

    /// A shape with a zero length holds no values, so it takes an empty value.
    #[test]
    fn a_zero_length_counts_no_values() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(Vec::<f32>::new(), &[0]).unwrap().view());
        let y = tape.reshape(x, &[0, 7]);
        assert_eq!(tape.shape(y).dims(), &[0, 7]);
    }

    /// Empty values pass through, with the new shape.
    #[test]
    fn empty_values_keep_their_shapes() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(Vec::<f32>::new(), &[0, 3]).unwrap().view());
        let y = tape.permute(x, &[1, 0]);
        assert_eq!(tape.shape(y).dims(), &[3, 0]);
        let z = tape.reshape(y, &[3, 0, 5]);
        assert_eq!(tape.shape(z).dims(), &[3, 0, 5]);
    }
}
