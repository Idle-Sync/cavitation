//! Matrix multiplication: `matmul` and `matmul_nt`, with their backward passes.

use crate::graph::{BackwardCtx, Var};
use crate::kernels::{MatMut, MatRef};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// The operation's name, used in panic messages.
fn name(nt: bool) -> &'static str {
    if nt { "matmul_nt" } else { "matmul" }
}

/// A row-major view of `data` with the given dimensions.
fn mat<T: Real>(data: &[T], rows: usize, cols: usize) -> MatRef<'_, T> {
    MatRef::new(data, rows, cols, cols, 1).expect("buffer holds rows * cols elements")
}

/// A writable row-major view of `data` with the given dimensions.
fn mat_mut<T: Real>(data: &mut [T], rows: usize, cols: usize) -> MatMut<'_, T> {
    MatMut::new(data, rows, cols, cols, 1).expect("buffer holds rows * cols elements")
}

/// The extents `(m, k, n)` of the product of `sa` (`[m, k]`) and `sb` (`[k, n]`, or `[n, k]`
/// when `nt`).
///
/// # Panics
///
/// Panics, naming the operation and both shapes, if either shape is not rank 2 or the inner
/// dimensions differ.
fn dims(nt: bool, sa: &Shape, sb: &Shape) -> (usize, usize, usize) {
    if sa.rank() != 2 || sb.rank() != 2 {
        panic!(
            "{}: both inputs must be rank 2, but the shapes are {:?} and {:?}",
            name(nt),
            sa.dims(),
            sb.dims()
        );
    }
    let (m, k) = (sa.dims()[0], sa.dims()[1]);
    let (n, kb) = if nt {
        (sb.dims()[0], sb.dims()[1])
    } else {
        (sb.dims()[1], sb.dims()[0])
    };
    if k != kb {
        panic!(
            "{}: inner dimensions differ, {:?} times {:?}{}",
            name(nt),
            sa.dims(),
            sb.dims(),
            if nt { " transposed" } else { "" }
        );
    }
    (m, k, n)
}

impl<T: Real> Tape<'_, T> {
    /// Records the matrix product `a · b` of an `[m, k]` and a `[k, n]` value, an `[m, n]`
    /// result.
    ///
    /// The gradients are `dA = dC · Bᵀ` and `dB = Aᵀ · dC`, computed on strided views
    /// without copying a transpose. An input that takes no gradient (a tape input) costs no
    /// work in the backward pass.
    ///
    /// # Panics
    ///
    /// Panics if either value is not rank 2 or the inner dimensions differ (the message
    /// names `matmul` and both shapes), or if either value was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    /// let (a, b) = (tape.input(a.view()), tape.input(b.view()));
    /// let c = tape.matmul(a, b);
    /// assert_eq!(tape.value(c).data(), &[19.0, 22.0, 43.0, 50.0]);
    /// ```
    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        self.product(false, a, b)
    }

    /// Records `a · bᵀ` of an `[m, k]` and an `[n, k]` value, an `[m, n]` result: the forward
    /// pass of a linear layer whose weights are stored `[out, in]`.
    ///
    /// The gradients are `dA = dC · B` and `dB = dCᵀ · A`, on strided views.
    ///
    /// # Panics
    ///
    /// Panics if either value is not rank 2 or the inner dimensions differ (the message
    /// names `matmul_nt` and both shapes), or if either value was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], &[2, 2]).unwrap();
    /// let (a, b) = (tape.input(a.view()), tape.input(b.view()));
    /// let c = tape.matmul_nt(a, b);
    /// assert_eq!(tape.value(c).data(), &[11.0, 17.0]);
    /// ```
    pub fn matmul_nt(&mut self, a: Var, b: Var) -> Var {
        self.product(true, a, b)
    }

    /// Records `a · b`, or `a · bᵀ` when `nt`.
    fn product(&mut self, nt: bool, a: Var, b: Var) -> Var {
        let (sa, sb) = (self.shape(a), self.shape(b));
        let (m, k, n) = dims(nt, &sa, &sb);
        let so = Shape::new(&[m, n]);
        let mut out = self.take_buffer(so.len());
        let x = mat(self.data(a), m, k);
        let y = if nt {
            mat(self.data(b), n, k).transpose()
        } else {
            mat(self.data(b), k, n)
        };
        // beta = 0 writes every element of `out`; the pooled buffer's old contents are unused.
        T::gemm(T::ONE, x, y, T::ZERO, mat_mut(&mut out, m, n));
        let op = if nt {
            Op::MatMulNt(a, b)
        } else {
            Op::MatMul(a, b)
        };
        self.push(op, so, out)
    }
}

/// Backward of `a · b` (`nt` false) and `a · bᵀ` (`nt` true): each input that takes a
/// gradient gets one GEMM straight into its gradient buffer, overwriting it on the first
/// contribution of the step and adding afterwards.
pub(super) fn backward<T: Real>(
    nt: bool,
    a: Var,
    b: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let (sa, sb) = (ctx.shape(a), ctx.shape(b));
    let (m, k, n) = dims(nt, &sa, &sb);
    let (av, bv) = (ctx.value(a), ctx.value(b));
    let dc = mat(out_grad, m, n);
    if ctx.takes_grad(a) {
        // dA [m, k] = dC [m, n] · B (B is [n, k] when nt, else Bᵀ of [k, n]).
        let rhs = if nt {
            mat(bv, n, k)
        } else {
            mat(bv, k, n).transpose()
        };
        if let Some(slot) = ctx.slot(a) {
            slot.update_with(|buf, beta| T::gemm(T::ONE, dc, rhs, beta, mat_mut(buf, m, k)));
        }
    }
    if ctx.takes_grad(b)
        && let Some(slot) = ctx.slot(b)
    {
        if nt {
            // dB [n, k] = dCᵀ [n, m] · A [m, k].
            let lhs = dc.transpose();
            slot.update_with(|buf, beta| {
                T::gemm(T::ONE, lhs, mat(av, m, k), beta, mat_mut(buf, n, k));
            });
        } else {
            // dB [k, n] = Aᵀ [k, m] · dC [m, n].
            let lhs = mat(av, m, k).transpose();
            slot.update_with(|buf, beta| T::gemm(T::ONE, lhs, dc, beta, mat_mut(buf, k, n)));
        }
    }
}

/// Tests for matrix multiplication.
#[cfg(test)]
mod tests {
    use crate::real::GEMM_CALLS;
    use crate::{Graph, ParamStore, Tensor};

    /// [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]].
    #[test]
    fn matmul_by_hand() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        let c = tape.matmul(a, b);
        assert_eq!(tape.shape(c).dims(), &[2, 2]);
        assert_eq!(tape.value(c).data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    /// [[1,2]] · [[3,4],[5,6]]ᵀ = [[11,17]].
    #[test]
    fn matmul_nt_by_hand() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2]).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], &[2, 2]).unwrap();
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        let c = tape.matmul_nt(a, b);
        assert_eq!(tape.value(c).data(), &[11.0, 17.0]);
    }

    /// Mismatched inner dimensions panic naming the operation and both shapes.
    #[test]
    #[should_panic(expected = "matmul: inner dimensions differ, [2, 3] times [4, 5]")]
    fn inner_dim_mismatch_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::<f32>::zeros(&[2, 3]);
        let b = Tensor::<f32>::zeros(&[4, 5]);
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        tape.matmul(a, b);
    }

    /// `matmul_nt` names itself and both shapes too.
    #[test]
    #[should_panic(expected = "matmul_nt: inner dimensions differ, [2, 3] times [4, 5] transposed")]
    fn nt_inner_dim_mismatch_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::<f32>::zeros(&[2, 3]);
        let b = Tensor::<f32>::zeros(&[4, 5]);
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        tape.matmul_nt(a, b);
    }

    /// Inputs that are not rank 2 panic naming the operation and the shapes.
    #[test]
    #[should_panic(
        expected = "matmul: both inputs must be rank 2, but the shapes are [3] and [3, 1]"
    )]
    fn rank_mismatch_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = Tensor::<f32>::zeros(&[3]);
        let b = Tensor::<f32>::zeros(&[3, 1]);
        let (a, b) = (tape.input(a.view()), tape.input(b.view()));
        tape.matmul(a, b);
    }

    /// The GEMM calls made by the backward pass of `matmul(a, w)` then `matmul_nt(h, w)`,
    /// where `a` is a tape input or a parameter and `w` is a parameter.
    fn backward_gemm_calls(a_is_input: bool) -> usize {
        let mut store = ParamStore::new();
        let w = store.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let p = store.add(Tensor::from_vec(vec![1.0f32, 0.5], &[1, 2]).unwrap());
        let x = Tensor::from_vec(vec![1.0f32, 0.5], &[1, 2]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let a = if a_is_input {
            tape.input(x.view())
        } else {
            tape.param(p)
        };
        let wv = tape.param(w);
        let h = tape.matmul(a, wv);
        let out = tape.matmul_nt(h, wv);
        // `out` is [1, 2]; seed it by hand.
        let before = GEMM_CALLS.with(|n| n.get());
        tape.backward_seeded(out, &[1.0, 1.0], &mut store.grads);
        GEMM_CALLS.with(|n| n.get()) - before
    }

    /// An input that takes no gradient costs no GEMM: with `a` an input, the backward pass
    /// of the two products runs 3 GEMMs instead of 4.
    #[test]
    fn no_input_gradient_is_computed() {
        assert_eq!(backward_gemm_calls(false), 4);
        assert_eq!(backward_gemm_calls(true), 3);
    }

    /// `dW = Xᵀ · dC` by hand, with two uses of `W` (the second adds to the first).
    #[test]
    fn backward_by_hand_and_accumulates() {
        let mut store = ParamStore::new();
        let w = store.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let x = Tensor::from_vec(vec![1.0f32, 1.0, 2.0, 0.0], &[2, 2]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let (xi, wv) = (tape.input(x.view()), tape.param(w));
        let y1 = tape.matmul(xi, wv);
        let y2 = tape.matmul(xi, wv);
        let y = tape.add(y1, y2);
        tape.backward_seeded(y, &[1.0; 4], &mut store.grads);
        drop(tape);
        // One use gives dW[p][j] = sum over rows i of X[i][p] = [[3, 3], [1, 1]]; two double it.
        assert_eq!(store.grad(w).data(), &[6.0, 6.0, 2.0, 2.0]);
    }

    /// Pooled buffers and gradient buffers holding old data do not leak into the results:
    /// every step gives the same values and gradients.
    #[test]
    fn reused_buffers_give_same_results() {
        let mut store = ParamStore::new();
        let w =
            store.add(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap());
        let x = Tensor::from_vec(vec![1.0f32, -1.0, 0.5, 2.0], &[2, 2]).unwrap();
        let mut graph = Graph::new();
        let mut first = None;
        for _ in 0..3 {
            store.zero_grad();
            let mut tape = graph.tape(&store.values);
            let (xi, wv) = (tape.input(x.view()), tape.param(w));
            let y = tape.matmul_nt(xi, wv);
            tape.backward_seeded(y, &[1.0; 6], &mut store.grads);
            let value = tape.value(y).data().to_vec();
            drop(tape);
            let got = (value, store.grad(w).data().to_vec());
            match &first {
                None => first = Some(got),
                Some(f) => assert_eq!(f, &got),
            }
        }
        // dW = dCᵀ · X with dC all ones: every row of dW is the column sums of X, [1.5, 1.0].
        assert_eq!(first.unwrap().1, vec![1.5, 1.0, 1.5, 1.0, 1.5, 1.0]);
    }
}
