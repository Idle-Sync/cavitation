//! Operations that move values without arithmetic: `reshape`, `permute`, `transpose`,
//! `slice` and `concat`.
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

impl<T: Real> Tape<'_, T> {
    /// Records the part of `x` from `start` up to (not including) `end` along `axis`, like
    /// PyTorch's `x.narrow(axis, start, end - start)` or `x[..., start:end, ...]`. The
    /// gradient of `x` is the output's gradient in that range and 0 elsewhere.
    ///
    /// # Panics
    ///
    /// Panics, naming `slice`, the range and the shape, if `axis` is not an axis of `x` or the
    /// range is not within it (`start <= end <= len`), or if `x` was not recorded on this
    /// tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec((0..12).map(|v| v as f32).collect(), &[3, 4]).unwrap().view());
    /// let y = tape.slice(x, 1, 1, 3);
    /// assert_eq!(tape.shape(y).dims(), &[3, 2]);
    /// assert_eq!(tape.value(y).data(), &[1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
    /// ```
    pub fn slice(&mut self, x: Var, axis: usize, start: usize, end: usize) -> Var {
        let shape = self.shape(x);
        if axis >= shape.rank() || start > end || end > shape.dims()[axis] {
            panic!(
                "slice: {start}..{end} along axis {axis} is not within shape {:?}",
                shape.dims()
            );
        }
        let (outer, n, inner) = around(&shape, axis);
        let m = end - start;
        let mut dims = [0usize; Shape::MAX_RANK];
        dims[..shape.rank()].copy_from_slice(shape.dims());
        dims[axis] = m;
        let out_shape = Shape::new(&dims[..shape.rank()]);
        let mut out = self.take_buffer(out_shape.len());
        let src = self.data(x);
        for o in 0..outer {
            out[o * m * inner..][..m * inner]
                .copy_from_slice(&src[(o * n + start) * inner..][..m * inner]);
        }
        self.push(Op::Slice(x, axis, start), out_shape, out)
    }

    /// Records the values of `xs` joined along `axis`, in order, like PyTorch's
    /// `torch.cat(xs, axis)`. Every value has the same rank and the same lengths on every
    /// other axis. Each input's gradient is its part of the output's gradient. One value may
    /// appear more than once; its gradient then adds each part.
    ///
    /// # Panics
    ///
    /// Panics, naming `concat`, the axis and the shapes, if `xs` is empty, if `axis` is not
    /// an axis of the values, or if the values differ in rank or on another axis; or if one of
    /// them was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = tape.input(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap().view());
    /// let b = tape.input(Tensor::from_vec(vec![5.0f32, 6.0], &[2, 1]).unwrap().view());
    /// let y = tape.concat(&[a, b], 1);
    /// assert_eq!(tape.shape(y).dims(), &[2, 3]);
    /// assert_eq!(tape.value(y).data(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    /// ```
    pub fn concat(&mut self, xs: &[Var], axis: usize) -> Var {
        let Some(&first) = xs.first() else {
            panic!("concat: no values to join along axis {axis}");
        };
        let shape = self.shape(first);
        let rank = shape.rank();
        if axis >= rank {
            panic!(
                "concat: axis {axis} is not an axis of shape {:?}",
                shape.dims()
            );
        }
        let mut total = 0usize;
        for &x in xs {
            let other = self.shape(x);
            let fits = other.rank() == rank
                && (0..rank).all(|d| d == axis || other.dims()[d] == shape.dims()[d]);
            if !fits {
                panic!(
                    "concat: shape {:?} does not join shape {:?} along axis {axis}",
                    other.dims(),
                    shape.dims()
                );
            }
            total = total
                .checked_add(other.dims()[axis])
                .unwrap_or_else(|| panic!("concat: the joined axis {axis} overflows"));
        }
        let mut dims = [0usize; Shape::MAX_RANK];
        dims[..rank].copy_from_slice(shape.dims());
        dims[axis] = total;
        let out_shape = Shape::new(&dims[..rank]);
        let (outer, _, inner) = around(&out_shape, axis);
        let mut out = self.take_buffer(out_shape.len());
        let mut at = 0;
        for &x in xs {
            let n = self.shape(x).dims()[axis];
            let src = self.data(x);
            for o in 0..outer {
                out[(o * total + at) * inner..][..n * inner]
                    .copy_from_slice(&src[o * n * inner..][..n * inner]);
            }
            at += n;
        }
        let start = self.push_labels(&[axis, xs.len()]);
        for &x in xs {
            self.push_labels(&x.to_words());
        }
        self.push(Op::Concat(start), out_shape, out)
    }
}

/// Splits `shape` around `axis` into `(outer, n, inner)`: the product of the lengths before
/// the axis, its length, and the product of the lengths after it.
fn around(shape: &Shape, axis: usize) -> (usize, usize, usize) {
    let dims = shape.dims();
    (
        dims[..axis].iter().product(),
        dims[axis],
        dims[axis + 1..].iter().product(),
    )
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

/// Backward of `slice`: the output's gradient into its range of `x`'s, 0 elsewhere.
pub(super) fn slice_backward<T: Real>(
    x: Var,
    axis: usize,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let shape = ctx.shape(x);
    let (outer, n, inner) = around(&shape, axis);
    let m = out_grad.len().checked_div(outer * inner).unwrap_or(0);
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            if beta == T::ZERO {
                buf.fill(T::ZERO);
            }
            for o in 0..outer {
                let dst = &mut buf[(o * n + start) * inner..][..m * inner];
                for (d, &g) in dst.iter_mut().zip(&out_grad[o * m * inner..][..m * inner]) {
                    *d += g;
                }
            }
        });
    }
}

/// Backward of `concat`: each input's part of the output's gradient, through the axis and
/// the inputs kept in the labels from `start`.
pub(super) fn concat_backward<T: Real>(start: usize, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let labels = ctx.labels();
    let (axis, count) = (labels[start], labels[start + 1]);
    let inputs = &labels[start + 2..][..2 * count];
    let mut total = 0;
    for pair in inputs.as_chunks::<2>().0 {
        total += ctx.shape(Var::from_words(pair)).dims()[axis];
    }
    let mut at = 0;
    for pair in inputs.as_chunks::<2>().0 {
        let x = Var::from_words(pair);
        let shape = ctx.shape(x);
        let (outer, n, inner) = around(&shape, axis);
        if let Some(slot) = ctx.slot(x) {
            slot.update_with(|buf, beta| {
                for o in 0..outer {
                    let src = &out_grad[(o * total + at) * inner..][..n * inner];
                    let dst = &mut buf[o * n * inner..][..n * inner];
                    if beta == T::ZERO {
                        dst.copy_from_slice(src);
                    } else {
                        for (d, &g) in dst.iter_mut().zip(src) {
                            *d += g;
                        }
                    }
                }
            });
        }
        at += n;
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

    /// A slice of the middle axis keeps its range, and its gradient is the output's there and
    /// 0 elsewhere: two backward passes without `zero_grad` add 1 twice inside the range and
    /// leave 0 outside it.
    #[test]
    fn slice_takes_its_range_and_zeroes_the_rest() {
        let mut params = ParamStore::new();
        let p =
            params.add(Tensor::from_vec((0..24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap());
        let mut graph = Graph::new();
        for _ in 0..2 {
            let mut tape = graph.tape(&params.values);
            let x = tape.param(p);
            let y = tape.slice(x, 1, 1, 3);
            assert_eq!(tape.shape(y).dims(), &[2, 2, 4]);
            let want: Vec<f32> = [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23]
                .map(|v| v as f32)
                .to_vec();
            assert_eq!(tape.value(y).data(), &want[..]);
            let loss = tape.sum(y);
            tape.backward(loss, &mut params.grads);
        }
        let g = params.grad(p).data();
        for (i, &v) in g.iter().enumerate() {
            let inside = (i / 4) % 3 != 0;
            assert_eq!(v, if inside { 2.0 } else { 0.0 }, "element {i}");
        }
    }

    /// Joining a value with itself and another adds both parts into its gradient.
    #[test]
    fn concat_adds_a_repeated_input() {
        let mut params = ParamStore::new();
        let a = params.add(Tensor::from_vec(vec![1.0f32, 2.0], &[2, 1]).unwrap());
        let b = params.add(Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], &[2, 2]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let (xa, xb) = (tape.param(a), tape.param(b));
        let y = tape.concat(&[xa, xb, xa], 1);
        assert_eq!(tape.shape(y).dims(), &[2, 4]);
        assert_eq!(
            tape.value(y).data(),
            &[1.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0, 2.0]
        );
        let w = tape.input(
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
                .unwrap()
                .view(),
        );
        let weighted = tape.mul(y, w);
        let loss = tape.sum(weighted);
        tape.backward(loss, &mut params.grads);
        assert_eq!(params.grad(a).data(), &[1.0 + 4.0, 5.0 + 8.0]);
        assert_eq!(params.grad(b).data(), &[2.0, 3.0, 6.0, 7.0]);
    }

    /// A range past the axis is refused.
    #[test]
    #[should_panic(expected = "slice: 2..5 along axis 1 is not within shape [2, 3]")]
    fn slice_refuses_a_range_past_the_axis() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(vec![0.0f32; 6], &[2, 3]).unwrap().view());
        tape.slice(x, 1, 2, 5);
    }

    /// Values that differ off the joined axis are refused.
    #[test]
    #[should_panic(expected = "concat: shape [3, 2] does not join shape [2, 2] along axis 1")]
    fn concat_refuses_another_length() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = tape.input(Tensor::from_vec(vec![0.0f32; 4], &[2, 2]).unwrap().view());
        let b = tape.input(Tensor::from_vec(vec![0.0f32; 6], &[3, 2]).unwrap().view());
        tape.concat(&[a, b], 1);
    }

    /// An empty list is refused.
    #[test]
    #[should_panic(expected = "concat: no values to join along axis 0")]
    fn concat_refuses_no_values() {
        let store = ParamStore::<f32>::new_generic();
        let mut graph = Graph::new_generic();
        let mut tape = graph.tape(&store.values);
        tape.concat(&[], 0);
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
