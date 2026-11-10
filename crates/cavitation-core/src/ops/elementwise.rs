//! Elementwise operations with broadcasting: `add`, `sub`, `mul`, `scale` and the select
//! `keep_where`.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// Which binary operation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
}

impl Kind {
    /// The operation's name, used in panic messages.
    fn name(self) -> &'static str {
        match self {
            Kind::Add => "add",
            Kind::Sub => "sub",
            Kind::Mul => "mul",
        }
    }

    /// The recorded operation for inputs `a` and `b`.
    fn op<T: Real>(self, a: Var, b: Var) -> Op<T> {
        match self {
            Kind::Add => Op::Add(a, b),
            Kind::Sub => Op::Sub(a, b),
            Kind::Mul => Op::Mul(a, b),
        }
    }
}

/// One stride per output dimension: how far an operand's offset moves when that dimension
/// advances by one. Zero for a dimension the operand is broadcast over.
type Strides = [usize; Shape::MAX_RANK];

/// The strides of an operand of shape `s` read as if it had shape `out` (`s` broadcasts to
/// `out`): dimensions are aligned from the right, and a dimension of 1 or a missing one
/// gets stride 0.
fn broadcast_strides(out: &Shape, s: &Shape) -> Strides {
    let mut strides = [0; Shape::MAX_RANK];
    let lead = out.rank() - s.rank();
    let mut stride = 1;
    for d in (0..s.rank()).rev() {
        let n = s.dims()[d];
        strides[lead + d] = if n == 1 { 0 } else { stride };
        stride *= n;
    }
    strides
}

/// Whether `s` is a row vector broadcast over the rows of `out`: `out` has at least two
/// dimensions, `s` holds exactly one row's worth of elements and shares `out`'s last
/// dimension, and the two shapes differ (for example `[N]` or `[1, N]` against `[B, N]`).
fn is_row_of(out: &Shape, s: &Shape) -> bool {
    out.rank() >= 2
        && s.rank() >= 1
        && s != out
        && s.dims()[s.rank() - 1] == out.dims()[out.rank() - 1]
        && s.len() == out.dims()[out.rank() - 1]
}

/// Calls `f(i, o0, o1, o2)` for every output position `i` in row-major order, where `o0`,
/// `o1` and `o2` are the offsets of the three operands whose strides are `strides`.
fn walk(dims: &[usize], strides: [&Strides; 3], mut f: impl FnMut(usize, usize, usize, usize)) {
    let total: usize = dims.iter().product();
    let mut index = [0usize; Shape::MAX_RANK];
    let mut off = [0usize; 3];
    for i in 0..total {
        f(i, off[0], off[1], off[2]);
        for d in (0..dims.len()).rev() {
            index[d] += 1;
            for k in 0..3 {
                off[k] += strides[k][d];
            }
            if index[d] < dims[d] {
                break;
            }
            for k in 0..3 {
                off[k] -= strides[k][d] * dims[d];
            }
            index[d] = 0;
        }
    }
}

/// The shape of `a op b`.
///
/// # Panics
///
/// Panics, naming the operation and both shapes, if they cannot be broadcast together.
fn broadcast_or_panic(kind: Kind, sa: &Shape, sb: &Shape) -> Shape {
    Shape::broadcast(sa, sb).unwrap_or_else(|| {
        panic!(
            "{}: cannot broadcast {:?} with {:?}",
            kind.name(),
            sa.dims(),
            sb.dims()
        )
    })
}

/// Writes `f(a, b)` into every element of `out`, broadcasting `a` (shape `sa`) and `b`
/// (shape `sb`) to `so`. Every element of `out` is written.
fn apply<T: Real>(
    f: impl Fn(T, T) -> T,
    a: &[T],
    b: &[T],
    (sa, sb, so): (&Shape, &Shape, &Shape),
    out: &mut [T],
) {
    debug_assert_eq!(out.len(), so.len());
    if out.is_empty() {
        return;
    }
    if sa == so && sb == so {
        debug_assert!(a.len() == out.len() && b.len() == out.len());
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = f(x, y);
        }
    } else if sa == so && is_row_of(so, sb) {
        // The bias add: every row of `a` meets the whole of `b`.
        debug_assert!(a.len() == out.len() && b.len() == so.dims()[so.rank() - 1]);
        for (orow, arow) in out.chunks_exact_mut(b.len()).zip(a.chunks_exact(b.len())) {
            for ((o, &x), &y) in orow.iter_mut().zip(arow).zip(b) {
                *o = f(x, y);
            }
        }
    } else {
        let zero = [0; Shape::MAX_RANK];
        let (ta, tb) = (broadcast_strides(so, sa), broadcast_strides(so, sb));
        walk(so.dims(), [&ta, &tb, &zero], |i, oa, ob, _| {
            out[i] = f(a[oa], b[ob]);
        });
    }
}

impl<T: Real> Tape<'_, T> {
    /// Records `a + b` with NumPy broadcasting: shapes are aligned from the right, and a
    /// dimension of 1 (or a missing one) stretches to match the other. A `[B, N]` value
    /// plus a `[N]` row vector (the bias add) takes a fast path.
    ///
    /// The gradient sums the incoming gradient over the dimensions each input was
    /// stretched along.
    ///
    /// # Panics
    ///
    /// Panics if the shapes cannot be broadcast together (the message names the operation
    /// and both shapes, for example `add: cannot broadcast [2, 3] with [2]`), or if either
    /// value was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![10.0f32, 20.0], &[2]).unwrap();
    /// let (a, b) = (tape.input(a.view()), tape.input(b.view()));
    /// let c = tape.add(a, b);
    /// assert_eq!(tape.value(c).data(), &[11.0, 22.0, 13.0, 24.0]);
    /// ```
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.binary(Kind::Add, a, b)
    }

    /// Records `a - b` with NumPy broadcasting (see [`add`](Tape::add)).
    ///
    /// # Panics
    ///
    /// Panics if the shapes cannot be broadcast together (the message names the operation
    /// and both shapes), or if either value was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![5.0f32, 6.0], &[2]).unwrap();
    /// let b = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();
    /// let (a, b) = (tape.input(a.view()), tape.input(b.view()));
    /// let c = tape.sub(a, b);
    /// assert_eq!(tape.value(c).data(), &[4.0, 5.0]);
    /// ```
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.binary(Kind::Sub, a, b)
    }

    /// Records `a * b`, element by element, with NumPy broadcasting (see
    /// [`add`](Tape::add)).
    ///
    /// # Panics
    ///
    /// Panics if the shapes cannot be broadcast together (the message names the operation
    /// and both shapes), or if either value was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
    /// let b = Tensor::from_vec(vec![2.0f32], &[]).unwrap();
    /// let (a, b) = (tape.input(a.view()), tape.input(b.view()));
    /// let c = tape.mul(a, b);
    /// assert_eq!(tape.value(c).data(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.binary(Kind::Mul, a, b)
    }

    /// Records `a * s`: every element of `a` times the scalar `s`.
    ///
    /// # Panics
    ///
    /// Panics if `a` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![1.0f32, -2.0], &[2]).unwrap();
    /// let a = tape.input(a.view());
    /// let c = tape.scale(a, 0.5);
    /// assert_eq!(tape.value(c).data(), &[0.5, -1.0]);
    /// ```
    pub fn scale(&mut self, a: Var, s: T) -> Var {
        let shape = self.shape(a);
        let mut out = self.take_buffer(shape.len());
        debug_assert_eq!(out.len(), self.data(a).len());
        for (o, &x) in out.iter_mut().zip(self.data(a)) {
            *o = x * s;
        }
        self.push(Op::Scale(a, s.to_f64()), shape, out)
    }

    /// Records `x` where `keep` is non-zero and exactly 0 where `keep` is 0, with `keep`
    /// broadcast to the shape of `x` (NumPy rules; see [`add`](Tape::add)). A `[N]` row
    /// against a `[B, N]` input selects columns, as a neuron mask does.
    ///
    /// It selects rather than multiplies, so an infinite or NaN element of `x` where `keep`
    /// is 0 still gives 0 (a multiply by 0 would give NaN). A NaN in `keep` counts as
    /// non-zero. The gradient of `x` is the output gradient where `keep` is non-zero and 0
    /// elsewhere; `keep` is a condition and gets no gradient. This matches PyTorch's
    /// `torch.where(keep != 0, x, 0)`.
    ///
    /// # Panics
    ///
    /// Panics if `keep` does not broadcast to the shape of `x` (the message names both
    /// shapes, for example `keep_where: cannot broadcast keep [4] to the input's shape
    /// [2, 3]`), including a `keep` that would stretch `x`, or if either value was not
    /// recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = vec![1.0f32, f32::INFINITY, 3.0, 4.0, f32::NAN, 6.0];
    /// let x = Tensor::from_vec(x, &[2, 3]).unwrap();
    /// let keep = Tensor::from_vec(vec![1.0f32, 0.0, 1.0], &[3]).unwrap();
    /// let (x, keep) = (tape.input(x.view()), tape.input(keep.view()));
    /// let y = tape.keep_where(x, keep);
    /// assert_eq!(tape.value(y).data(), &[1.0, 0.0, 3.0, 4.0, 0.0, 6.0]);
    /// ```
    pub fn keep_where(&mut self, x: Var, keep: Var) -> Var {
        let (sx, sk) = (self.shape(x), self.shape(keep));
        check_keep(&sx, &sk);
        let mut out = self.take_buffer(sx.len());
        let (xd, kd) = (self.data(x), self.data(keep));
        apply(select, xd, kd, (&sx, &sk, &sx), &mut out);
        self.push(Op::KeepWhere(x, keep), sx, out)
    }

    /// Records `a kind b`.
    fn binary(&mut self, kind: Kind, a: Var, b: Var) -> Var {
        let (sa, sb) = (self.shape(a), self.shape(b));
        let so = broadcast_or_panic(kind, &sa, &sb);
        let mut out = self.take_buffer(so.len());
        let (x, y) = (self.data(a), self.data(b));
        let shapes = (&sa, &sb, &so);
        match kind {
            Kind::Add => apply(|p, q| p + q, x, y, shapes, &mut out),
            Kind::Sub => apply(|p, q| p - q, x, y, shapes, &mut out),
            Kind::Mul => apply(|p, q| p * q, x, y, shapes, &mut out),
        }
        self.push(kind.op(a, b), so, out)
    }
}

/// Fills `dst` (the shape of one input, `sv`) with the sum of `sign * g` over the dimensions
/// that input was stretched along; `g` has the output shape `so`. Every element of `dst` is
/// written.
fn sum_to<T: Real>(dst: &mut [T], sv: &Shape, so: &Shape, g: &[T], sign: T) {
    debug_assert!(dst.len() == sv.len() && g.len() == so.len());
    dst.fill(T::ZERO);
    if g.is_empty() {
        return;
    }
    if is_row_of(so, sv) {
        // The bias add: sum the rows.
        for row in g.chunks_exact(dst.len()) {
            for (d, &x) in dst.iter_mut().zip(row) {
                *d += sign * x;
            }
        }
    } else {
        let zero = [0; Shape::MAX_RANK];
        let tv = broadcast_strides(so, sv);
        walk(so.dims(), [&zero, &tv, &zero], |i, _, ov, _| {
            dst[ov] += sign * g[i];
        });
    }
}

/// Fills `dst` (the shape of one input, `sv`) with the sum of `g * c` over the dimensions
/// that input was stretched along, where `c` is the other input (shape `sc`) broadcast to
/// the output shape `so`. Every element of `dst` is written.
fn sum_product_to<T: Real>(
    dst: &mut [T],
    sv: &Shape,
    (c, sc): (&[T], &Shape),
    so: &Shape,
    g: &[T],
) {
    debug_assert!(dst.len() == sv.len() && g.len() == so.len());
    if sv == so && sc == so {
        debug_assert_eq!(c.len(), g.len());
        for ((d, &x), &y) in dst.iter_mut().zip(g).zip(c) {
            *d = x * y;
        }
        return;
    }
    dst.fill(T::ZERO);
    let zero = [0; Shape::MAX_RANK];
    let (tv, tc) = (broadcast_strides(so, sv), broadcast_strides(so, sc));
    walk(so.dims(), [&zero, &tv, &tc], |i, _, ov, oc| {
        dst[ov] += g[i] * c[oc];
    });
}

/// Backward of `a + b`, `a - b` and `a * b`: each input that takes a gradient gets the
/// output gradient (times the other input for `mul`, negated for `sub`'s second input),
/// summed over the dimensions it was stretched along.
pub(super) fn binary_backward<T: Real>(
    kind: Kind,
    a: Var,
    b: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let (sa, sb) = (ctx.shape(a), ctx.shape(b));
    let so = broadcast_or_panic(kind, &sa, &sb);
    for (v, sv, other, sc, is_b) in [(a, sa, b, sb, false), (b, sb, a, sa, true)] {
        if !ctx.takes_grad(v) {
            continue;
        }
        let sign = if kind == Kind::Sub && is_b {
            -T::ONE
        } else {
            T::ONE
        };
        if kind == Kind::Mul {
            let mut dst = ctx.scratch(sv.len());
            sum_product_to(&mut dst, &sv, (ctx.value(other), &sc), &so, out_grad);
            if let Some(slot) = ctx.slot(v) {
                slot.add_from(&dst);
            }
            ctx.release(dst);
        } else if sv == so {
            if let Some(slot) = ctx.slot(v) {
                slot.add_scaled_from(out_grad, sign);
            }
        } else {
            let mut dst = ctx.scratch(sv.len());
            sum_to(&mut dst, &sv, &so, out_grad, sign);
            if let Some(slot) = ctx.slot(v) {
                slot.add_from(&dst);
            }
            ctx.release(dst);
        }
    }
}

/// `v` where `keep` is non-zero, exactly 0 where it is 0.
fn select<T: Real>(v: T, keep: T) -> T {
    if keep == T::ZERO { T::ZERO } else { v }
}

/// Panics, naming both shapes, unless a `keep` of shape `sk` broadcasts to `sx` exactly.
fn check_keep(sx: &Shape, sk: &Shape) {
    assert!(
        Shape::broadcast(sx, sk).as_ref() == Some(sx),
        "keep_where: cannot broadcast keep {:?} to the input's shape {:?}",
        sk.dims(),
        sx.dims()
    );
}

/// Backward of `keep_where`: `x` gets the output gradient where `keep` is non-zero and 0
/// elsewhere; `keep` gets nothing.
pub(super) fn keep_where_backward<T: Real>(
    x: Var,
    keep: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    if !ctx.takes_grad(x) {
        return;
    }
    let (sx, sk) = (ctx.shape(x), ctx.shape(keep));
    let mut dst = ctx.scratch(sx.len());
    apply(select, out_grad, ctx.value(keep), (&sx, &sk, &sx), &mut dst);
    if let Some(slot) = ctx.slot(x) {
        slot.add_from(&dst);
    }
    ctx.release(dst);
}

/// Backward of `a * s`: the output gradient times `s`.
pub(super) fn scale_backward<T: Real>(
    a: Var,
    s: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    if let Some(slot) = ctx.slot(a) {
        slot.add_scaled_from(out_grad, T::from_f64(s));
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

    /// [2, 3] + [3]: each row plus the vector.
    #[test]
    fn broadcast_add_values() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        let c = tape.add(a, b);
        assert_eq!(tape.shape(c).dims(), &[2, 3]);
        assert_eq!(tape.value(c).data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        // The vector on the left gives the same values.
        let d = tape.add(b, a);
        assert_eq!(tape.value(d).data(), tape.value(c).data());
    }

    /// [3, 1] - [1, 4] fills a [3, 4] grid.
    #[test]
    fn broadcast_sub_outer() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[1, 4]).unwrap();
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        let c = tape.sub(a, b);
        assert_eq!(tape.shape(c).dims(), &[3, 4]);
        assert_eq!(
            tape.value(c).data(),
            &[
                1.0, 0.0, -1.0, -2.0, 2.0, 1.0, 0.0, -1.0, 3.0, 2.0, 1.0, 0.0
            ]
        );
    }

    /// The gradient of a [3] bias added to a [2, 3] input is the column count of rows: [2, 2, 2].
    #[test]
    fn bias_add_grad_sums_rows() {
        let mut store = ParamStore::new();
        let bias = store.add(Tensor::from_vec(vec![0.5f32, 1.0, 1.5], &[3]).unwrap());
        let x = Tensor::full(&[2, 3], 1.0f32);
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let (xi, b) = (tape.input(x.view()), tape.param(bias));
        let out = tape.add(xi, b);
        tape.backward_seeded(out, &[1.0; 6], &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(bias).data(), &[2.0, 2.0, 2.0]);
    }

    /// d(a * b)/da = b and d(a * b)/db = a, also with broadcasting ([2, 2] * [2]).
    #[test]
    fn mul_grad_is_other_input() {
        let mut store = ParamStore::new();
        let pa = store.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let pb = store.add(Tensor::from_vec(vec![10.0f32, 100.0], &[2]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let (a, b) = (tape.param(pa), tape.param(pb));
        let out = tape.mul(a, b);
        assert_eq!(tape.value(out).data(), &[10.0, 200.0, 30.0, 400.0]);
        tape.backward_seeded(out, &[1.0; 4], &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(pa).data(), &[10.0, 100.0, 10.0, 100.0]);
        assert_eq!(store.grad(pb).data(), &[4.0, 6.0]);
    }

    /// d(a - b)/da = 1 and d(a - b)/db = -1.
    #[test]
    fn sub_grad_is_minus_one() {
        let mut store = ParamStore::new();
        let pa = store.add(Tensor::from_vec(vec![5.0f32, 6.0], &[2]).unwrap());
        let pb = store.add(Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let (a, b) = (tape.param(pa), tape.param(pb));
        let out = tape.sub(a, b);
        assert_eq!(tape.value(out).data(), &[4.0, 4.0]);
        tape.backward_seeded(out, &[1.0; 2], &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(pa).data(), &[1.0, 1.0]);
        assert_eq!(store.grad(pb).data(), &[-1.0, -1.0]);
    }

    /// Values that cannot be broadcast panic naming the operation and both shapes.
    #[test]
    #[should_panic(expected = "add: cannot broadcast [2, 3] with [2]")]
    fn mismatched_shapes_panic() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::zeros(&[2, 3]);
        let b = Tensor::zeros(&[2]);
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        tape.add(a, b);
    }

    /// `sub` and `mul` name themselves in the panic too.
    #[test]
    #[should_panic(expected = "mul: cannot broadcast [3] with [4]")]
    fn mul_mismatch_names_mul() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::zeros(&[3]);
        let b = Tensor::zeros(&[4]);
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        tape.mul(a, b);
    }

    /// scale multiplies every element, and its gradient is the scalar.
    #[test]
    fn scale_grad() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![1.0f32, -2.0, 4.0], &[3]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let out = tape.scale(v, 2.5);
        assert_eq!(tape.value(out).data(), &[2.5, -5.0, 10.0]);
        tape.backward_seeded(out, &[1.0, 2.0, 3.0], &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[2.5, 5.0, 7.5]);
    }

    /// `keep_where` gives exactly 0 where `keep` is 0, even for an infinite or NaN input, and
    /// the input unchanged elsewhere.
    #[test]
    fn keep_where_zeroes_dropped_positions_even_inf_and_nan() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(
            vec![
                1.0f32,
                f32::INFINITY,
                -3.0,
                4.0,
                f32::NAN,
                f32::NEG_INFINITY,
            ],
            &[2, 3],
        )
        .unwrap();
        let keep = Tensor::from_vec(vec![1.0f32, 0.0, 1.0], &[3]).unwrap();
        let (x, keep) = (tape.input(x.view()), tape.input(keep.view()));
        let out = tape.keep_where(x, keep);
        assert_eq!(tape.shape(out).dims(), &[2, 3]);
        let bits: Vec<u32> = tape.value(out).data().iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = [1.0f32, 0.0, -3.0, 4.0, 0.0, f32::NEG_INFINITY]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits, want);
    }

    /// The input's gradient is the output gradient where `keep` is non-zero and 0 elsewhere;
    /// `keep` itself gets no gradient. A same-shape `keep` works too.
    #[test]
    fn keep_where_grad_is_zero_where_dropped() {
        let mut store = ParamStore::new();
        let px = store.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let pk = store.add(Tensor::from_vec(vec![0.0f32, 2.0], &[2]).unwrap());
        let full = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let (x, k) = (tape.param(px), tape.param(pk));
        let out = tape.keep_where(x, k);
        assert_eq!(tape.value(out).data(), &[0.0, 2.0, 0.0, 4.0]);
        tape.backward_seeded(out, &[5.0, 6.0, 7.0, 8.0], &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(px).data(), &[0.0, 6.0, 0.0, 8.0]);
        assert_eq!(store.grad(pk).data(), &[0.0, 0.0]);

        store.zero_grad();
        let mut tape = graph.tape(&store.values);
        let (x, k) = (tape.param(px), tape.input(full.view()));
        let out = tape.keep_where(x, k);
        assert_eq!(tape.value(out).data(), &[1.0, 0.0, 0.0, 4.0]);
        tape.backward_seeded(out, &[5.0, 6.0, 7.0, 8.0], &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(px).data(), &[5.0, 0.0, 0.0, 8.0]);
    }

    /// A `keep` that does not broadcast to the input's shape panics naming both shapes.
    #[test]
    #[should_panic(expected = "keep_where: cannot broadcast keep [4] to the input's shape [2, 3]")]
    fn keep_where_mismatch_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::zeros(&[2, 3]);
        let keep = Tensor::zeros(&[4]);
        let (x, keep) = (tape.input(x.view()), tape.input(keep.view()));
        tape.keep_where(x, keep);
    }

    /// A `keep` larger than the input (which would stretch the input) panics too.
    #[test]
    #[should_panic(expected = "keep_where: cannot broadcast keep [2, 3] to the input's shape [3]")]
    fn keep_where_larger_keep_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::zeros(&[3]);
        let keep = Tensor::zeros(&[2, 3]);
        let (x, keep) = (tape.input(x.view()), tape.input(keep.view()));
        tape.keep_where(x, keep);
    }

    /// A pooled buffer holding old data is fully overwritten: the second step gives the same
    /// answer as the first, for forward values and gradients.
    #[test]
    fn reused_buffers_give_same_results() {
        let mut store = ParamStore::new();
        let pb = store.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap());
        let x = Tensor::from_vec(vec![1.0f32; 6], &[2, 3]).unwrap();
        let mut graph = Graph::new();
        let mut results = Vec::new();
        for _ in 0..3 {
            store.zero_grad();
            let mut tape = graph.tape(&store.values);
            let (xi, b) = (tape.input(x.view()), tape.param(pb));
            let m = tape.mul(xi, b);
            let out = tape.sub(m, b);
            tape.backward_seeded(out, &[1.0; 6], &mut store.grads);
            results.push((
                tape.value(out).data().to_vec(),
                tape.value(m).data().to_vec(),
            ));
            drop(tape);
            assert_eq!(store.grad(pb).data(), &[0.0, 0.0, 0.0]);
        }
        assert_eq!(results[0], results[2]);
    }
}
