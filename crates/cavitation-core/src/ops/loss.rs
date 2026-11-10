//! Softmax, log-softmax and the losses built on them.
//!
//! Every loss follows one pattern. It computes an unreduced loss (one value per element or
//! per sample), then [`Reduction`] turns it into the recorded value: `None` keeps it, the
//! others sum it and divide by a count. The backward pass multiplies the incoming gradient
//! by the same count's reciprocal, through [`Upstream`]. A new loss supplies its per-item
//! value to [`reduce_into`] and its per-item weight to [`Upstream::at`].

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// How an unreduced loss (one value per sample or per element) becomes the loss that is
/// recorded. PyTorch's default is [`Reduction::Mean`].
///
/// # Examples
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let p = tape.input(Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap().view());
/// let t = tape.input(Tensor::from_vec(vec![0.0f32, 0.0], &[2]).unwrap().view());
/// let each = tape.mse(p, t, Reduction::None);
/// let sum = tape.mse(p, t, Reduction::Sum);
/// assert_eq!(tape.value(each).data(), &[1.0, 4.0]);
/// assert_eq!(tape.value(sum).data(), &[5.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// The sum of the unreduced loss divided by its number of elements; a rank-0 value.
    Mean,
    /// The sum of the unreduced loss; a rank-0 value.
    Sum,
    /// The unreduced loss itself: `[batch]` for cross-entropy, the prediction's shape for an
    /// elementwise loss.
    None,
    /// The sum of the unreduced loss divided by the batch size, the first dimension of the
    /// input (1 for a rank-0 input); a rank-0 value. PyTorch recommends it for KL divergence.
    BatchMean,
}

impl Reduction {
    /// What the sum of the unreduced loss is divided by: `n` unreduced elements over a batch
    /// of `batch`. `None` does not divide, so it gives 1.
    pub(super) fn divisor(self, n: usize, batch: usize) -> usize {
        match self {
            Reduction::Mean => n,
            Reduction::BatchMean => batch,
            Reduction::Sum | Reduction::None => 1,
        }
    }

    /// The shape of the recorded loss for an unreduced loss of shape `unreduced`.
    pub(super) fn output_shape(self, unreduced: Shape) -> Shape {
        match self {
            Reduction::None => unreduced,
            _ => Shape::new(&[]),
        }
    }
}

/// The batch size of an input of `shape` for `BatchMean`: its first dimension, or 1 for a
/// rank-0 input.
pub(super) fn batch_of(shape: &Shape) -> usize {
    shape.dims().first().copied().unwrap_or(1)
}

/// Fills `out` (the recorded loss, as long as [`Reduction::output_shape`] says) from
/// `per(i)`, the unreduced loss at `i` for `i` in `0..n`: the values themselves for `None`,
/// otherwise their sum divided by the reduction's divisor. Every element of `out` is written.
pub(super) fn reduce_into<T: Real>(
    out: &mut [T],
    reduction: Reduction,
    n: usize,
    batch: usize,
    mut per: impl FnMut(usize) -> T,
) {
    if reduction == Reduction::None {
        for (i, o) in out.iter_mut().enumerate() {
            *o = per(i);
        }
        return;
    }
    let mut sum = T::ZERO;
    for i in 0..n {
        sum += per(i);
    }
    out[0] = match reduction.divisor(n, batch) {
        1 => sum,
        d => sum / T::from_f64(d as f64),
    };
}

/// The incoming gradient of a loss as seen by one unreduced item: for `None` the item's own
/// gradient, otherwise the single incoming gradient times the reciprocal of the divisor.
pub(super) struct Upstream<'a, T> {
    /// The incoming gradient: one element per item for `None`, else one element.
    grad: &'a [T],
    /// Whether `grad` has one element per item.
    per_item: bool,
    /// The reciprocal of the divisor (1 for `None` and `Sum`).
    scale: T,
}

impl<'a, T: Real> Upstream<'a, T> {
    /// The upstream gradient `out_grad` of a loss reduced by `reduction` over `n` items in a
    /// batch of `batch`.
    pub(super) fn new(out_grad: &'a [T], reduction: Reduction, n: usize, batch: usize) -> Self {
        Upstream {
            grad: out_grad,
            per_item: reduction == Reduction::None,
            scale: T::from_f64(1.0 / reduction.divisor(n, batch) as f64),
        }
    }

    /// The gradient of the recorded loss with respect to unreduced item `i`.
    pub(super) fn at(&self, i: usize) -> T {
        if self.per_item {
            self.grad[i]
        } else {
            self.grad[0] * self.scale
        }
    }
}

/// Whether `v` is NaN.
fn is_nan<T: Real>(v: T) -> bool {
    v.partial_cmp(&v).is_none()
}

/// The largest element of the non-empty `row`; NaN if any element is NaN, so that a NaN
/// logit reaches the output instead of being skipped by the comparison.
fn row_max<T: Real>(row: &[T]) -> T {
    let mut m = row[0];
    for &v in &row[1..] {
        if is_nan(v) {
            return v;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// `ln Σ e^(z - max)` of a non-empty row, and the max itself.
fn log_sum_exp<T: Real>(row: &[T]) -> (T, T) {
    let m = row_max(row);
    let mut s = T::ZERO;
    for &z in row {
        s += (z - m).exp();
    }
    (m, s.ln())
}

/// The last axis of `shape`.
///
/// # Panics
///
/// Panics, naming `name` and the shape, if `shape` has rank 0.
fn last_axis(name: &str, shape: &Shape) -> usize {
    match shape.dims().last() {
        Some(&w) => w,
        None => panic!("{name}: the input must have at least one axis, but its shape is []"),
    }
}

impl<T: Real> Tape<'_, T> {
    /// Records the softmax of `x` over its last axis, computed with the row maximum
    /// subtracted so that large logits do not overflow. NaN in a row makes the whole row NaN.
    ///
    /// # Panics
    ///
    /// Panics if `x` has rank 0 (the message names `softmax` and the shape), or if `x` was
    /// not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec(vec![1e4f32, 0.0, -1e4], &[3]).unwrap().view());
    /// let y = tape.softmax(x);
    /// assert_eq!(tape.value(y).data(), &[1.0, 0.0, 0.0]);
    /// ```
    pub fn softmax(&mut self, x: Var) -> Var {
        let shape = self.shape(x);
        let width = last_axis("softmax", &shape);
        let mut out = self.take_buffer(shape.len());
        if width > 0 {
            for (o, row) in out
                .chunks_exact_mut(width)
                .zip(self.data(x).chunks_exact(width))
            {
                let m = row_max(row);
                let mut sum = T::ZERO;
                for (o, &z) in o.iter_mut().zip(row) {
                    let e = (z - m).exp();
                    *o = e;
                    sum += e;
                }
                for o in o.iter_mut() {
                    *o = *o / sum;
                }
            }
        }
        self.push(Op::Softmax(x), shape, out)
    }

    /// Records the log-softmax of `x` over its last axis, `x - max - ln Σ e^(x - max)`, which
    /// never overflows. NaN in a row makes the whole row NaN.
    ///
    /// # Panics
    ///
    /// Panics if `x` has rank 0 (the message names `log_softmax` and the shape), or if `x`
    /// was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec(vec![1e4f32, 0.0], &[2]).unwrap().view());
    /// let y = tape.log_softmax(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, -1e4]);
    /// ```
    pub fn log_softmax(&mut self, x: Var) -> Var {
        let shape = self.shape(x);
        let width = last_axis("log_softmax", &shape);
        let mut out = self.take_buffer(shape.len());
        if width > 0 {
            for (o, row) in out
                .chunks_exact_mut(width)
                .zip(self.data(x).chunks_exact(width))
            {
                let (m, lse) = log_sum_exp(row);
                for (o, &z) in o.iter_mut().zip(row) {
                    *o = z - m - lse;
                }
            }
        }
        self.push(Op::LogSoftmax(x), shape, out)
    }

    /// Records the cross-entropy between softmax of `logits` (`[batch, classes]`) and the
    /// class `labels`, fused: row `i` has loss `-(z[y] - max - ln Σ e^(z - max))`, finite for
    /// finite logits even when a class probability underflows to 0. NaN in a row's logits
    /// gives a NaN loss for that row. The unreduced loss has shape `[batch]`; `reduction`
    /// turns it into the recorded value ([`Reduction::Mean`] and [`Reduction::BatchMean`]
    /// both divide by the batch). The gradient is `(softmax - onehot)` times the upstream
    /// gradient and the reduction's factor, written directly into the logits' gradient.
    ///
    /// # Panics
    ///
    /// Panics, naming `softmax_cross_entropy` and the values, if `logits` is not rank 2, if
    /// `labels.len()` differs from the batch, or if a label is not below the number of
    /// classes; all before any loss is computed. Also panics if `logits` was not recorded on
    /// this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let z = tape.input(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]).unwrap().view());
    /// let loss = tape.softmax_cross_entropy(z, &[2], Reduction::Mean);
    /// assert!((tape.value(loss).data()[0] - 0.407_606).abs() < 1e-5);
    /// ```
    pub fn softmax_cross_entropy(
        &mut self,
        logits: Var,
        labels: &[usize],
        reduction: Reduction,
    ) -> Var {
        let shape = self.shape(logits);
        let name = "softmax_cross_entropy";
        if shape.rank() != 2 {
            panic!(
                "{name}: logits must be rank 2 [batch, classes], but the shape is {:?}",
                shape.dims()
            );
        }
        let (batch, classes) = (shape.dims()[0], shape.dims()[1]);
        if labels.len() != batch {
            panic!("{name}: got {} labels for a batch of {batch}", labels.len());
        }
        if let Some((i, &label)) = labels.iter().enumerate().find(|&(_, &l)| l >= classes) {
            panic!("{name}: label {label} at index {i} is out of range 0..{classes}");
        }
        let start = self.push_labels(labels);
        let out_shape = reduction.output_shape(Shape::new(&[batch]));
        let mut out = self.take_buffer(out_shape.len());
        let rows = self.data(logits);
        reduce_into(&mut out, reduction, batch, batch, |i| {
            let row = &rows[i * classes..][..classes];
            let (m, lse) = log_sum_exp(row);
            -(row[labels[i]] - m - lse)
        });
        self.push(
            Op::SoftmaxCrossEntropy(logits, reduction, start),
            out_shape,
            out,
        )
    }

    /// Records the mean squared error `(pred - target)²` elementwise, reduced by `reduction`.
    /// The unreduced loss has the shape of `pred`. For [`Reduction::BatchMean`] the batch is
    /// the first dimension of `pred` (1 for a rank-0 input). Both inputs take gradients,
    /// `2 (pred - target)` times the upstream gradient and the reduction's factor, and the
    /// opposite sign for the target.
    ///
    /// # Panics
    ///
    /// Panics, naming `mse` and both shapes, if the shapes differ (there is no
    /// broadcasting), or if either variable was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let p = tape.input(Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![0.0f32, 0.0], &[2]).unwrap().view());
    /// let loss = tape.mse(p, t, Reduction::Mean);
    /// assert_eq!(tape.value(loss).data(), &[2.5]);
    /// ```
    pub fn mse(&mut self, pred: Var, target: Var, reduction: Reduction) -> Var {
        let shape = self.shape(pred);
        let other = self.shape(target);
        if shape != other {
            panic!(
                "mse: shapes differ, {:?} and {:?}",
                shape.dims(),
                other.dims()
            );
        }
        let n = shape.len();
        let batch = batch_of(&shape);
        let out_shape = reduction.output_shape(shape);
        let mut out = self.take_buffer(out_shape.len());
        let (p, t) = (self.data(pred), self.data(target));
        reduce_into(&mut out, reduction, n, batch, |i| {
            let d = p[i] - t[i];
            d * d
        });
        self.push(Op::Mse(pred, target, reduction), out_shape, out)
    }
}

/// Backward of `softmax`: `dx = y ⊙ (g - Σ g y)` per row. Every element is written when the
/// slot overwrites.
pub(super) fn softmax_backward<T: Real>(
    x: Var,
    out: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let width = ctx.shape(x).dims().last().copied().unwrap_or(0);
    if width == 0 {
        return;
    }
    let y = ctx.value(out);
    let Some(slot) = ctx.slot(x) else { return };
    slot.update_with(|buf, beta| {
        let rows = buf.chunks_exact_mut(width).zip(y.chunks_exact(width));
        for ((d, y), g) in rows.zip(out_grad.chunks_exact(width)) {
            let mut dot = T::ZERO;
            for (&y, &g) in y.iter().zip(g) {
                dot += y * g;
            }
            for ((d, &y), &g) in d.iter_mut().zip(y).zip(g) {
                let v = y * (g - dot);
                *d = if beta == T::ZERO { v } else { *d + v };
            }
        }
    });
}

/// Backward of `log_softmax`: `dx = g - softmax ⊙ Σ g` per row, with the softmax recovered
/// as `e^y`. Every element is written when the slot overwrites.
pub(super) fn log_softmax_backward<T: Real>(
    x: Var,
    out: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let width = ctx.shape(x).dims().last().copied().unwrap_or(0);
    if width == 0 {
        return;
    }
    let y = ctx.value(out);
    let Some(slot) = ctx.slot(x) else { return };
    slot.update_with(|buf, beta| {
        let rows = buf.chunks_exact_mut(width).zip(y.chunks_exact(width));
        for ((d, y), g) in rows.zip(out_grad.chunks_exact(width)) {
            let mut total = T::ZERO;
            for &g in g {
                total += g;
            }
            for ((d, &y), &g) in d.iter_mut().zip(y).zip(g) {
                let v = g - y.exp() * total;
                *d = if beta == T::ZERO { v } else { *d + v };
            }
        }
    });
}

/// Backward of softmax cross-entropy: row `i` of the logits' gradient is
/// `w_i · (softmax - onehot)`, with `w_i` from [`Upstream`]. The softmax is recomputed from
/// the logits. Every element is written when the slot overwrites.
pub(super) fn cross_entropy_backward<T: Real>(
    logits: Var,
    reduction: Reduction,
    labels_start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let shape = ctx.shape(logits);
    let (batch, classes) = (shape.dims()[0], shape.dims()[1]);
    if classes == 0 {
        return;
    }
    let labels = &ctx.labels()[labels_start..][..batch];
    let rows = ctx.value(logits);
    let up = Upstream::new(out_grad, reduction, batch, batch);
    let Some(slot) = ctx.slot(logits) else { return };
    slot.update_with(|buf, beta| {
        for (i, (d, row)) in buf
            .chunks_exact_mut(classes)
            .zip(rows.chunks_exact(classes))
            .enumerate()
        {
            let (m, lse) = log_sum_exp(row);
            let w = up.at(i);
            for (j, (d, &z)) in d.iter_mut().zip(row).enumerate() {
                let onehot = if j == labels[i] { T::ONE } else { T::ZERO };
                let v = w * ((z - m - lse).exp() - onehot);
                *d = if beta == T::ZERO { v } else { *d + v };
            }
        }
    });
}

/// Backward of MSE: `pred` gets `2 (pred - target) · w` and `target` the negative, with `w`
/// from [`Upstream`]. Every element is written when a slot overwrites.
pub(super) fn mse_backward<T: Real>(
    pred: Var,
    target: Var,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let shape = ctx.shape(pred);
    let n = shape.len();
    let up = Upstream::new(out_grad, reduction, n, batch_of(&shape));
    let (p, t) = (ctx.value(pred), ctx.value(target));
    let two = T::from_f64(2.0);
    for (var, sign) in [(pred, T::ONE), (target, -T::ONE)] {
        if let Some(slot) = ctx.slot(var) {
            slot.update_with(|buf, beta| {
                for (i, d) in buf.iter_mut().enumerate() {
                    let v = sign * two * (p[i] - t[i]) * up.at(i);
                    *d = if beta == T::ZERO { v } else { *d + v };
                }
            });
        }
    }
}

/// Tests for softmax, log-softmax, cross-entropy and MSE.
#[cfg(test)]
mod tests {
    use crate::{Graph, ParamStore, Reduction, Tensor};

    /// Records `f` on a fresh f64 tape over `x` (shape `dims`) and returns the output.
    fn run(
        x: &[f64],
        dims: &[usize],
        f: impl FnOnce(&mut crate::Tape<'_, f64>, crate::Var) -> crate::Var,
    ) -> (Vec<f64>, Vec<usize>) {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::from_vec(x.to_vec(), dims).unwrap().view());
        let y = f(&mut tape, x);
        (tape.value(y).data().to_vec(), tape.shape(y).dims().to_vec())
    }

    /// Asserts `got` and `want` agree element by element within `tol`.
    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tol, "{got:?} vs {want:?}");
        }
    }

    /// Every row of a softmax sums to one.
    #[test]
    fn softmax_rows_sum_to_one() {
        let x = [0.3, -1.2, 2.5, 0.0, 4.0, 4.0, -3.0, 1.0];
        let (y, dims) = run(&x, &[2, 4], |t, x| t.softmax(x));
        assert_eq!(dims, [2, 4]);
        for row in y.chunks(4) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        }
    }

    /// Logits of 1e4 and -1e4 do not overflow.
    #[test]
    fn softmax_large_logits_finite() {
        let (y, _) = run(&[1e4, 0.0, -1e4], &[3], |t, x| t.softmax(x));
        assert_eq!(y, [1.0, 0.0, 0.0]);
    }

    /// `log_softmax([1, 2, 3])` by hand.
    #[test]
    fn log_softmax_by_hand() {
        let (y, _) = run(&[1.0, 2.0, 3.0], &[3], |t, x| t.log_softmax(x));
        let want = [-2.40760596444438, -1.4076059644443801, -0.40760596444438013];
        assert_close(&y, &want, 1e-14);
    }

    /// `log_softmax` of large logits is exact, not `-inf` or NaN.
    #[test]
    fn log_softmax_large_logits_finite() {
        let (y, _) = run(&[1e4, 0.0], &[2], |t, x| t.log_softmax(x));
        assert_eq!(y, [0.0, -1e4]);
    }

    /// Cross-entropy of one row by hand.
    #[test]
    fn cross_entropy_by_hand() {
        let (y, dims) = run(&[1.0, 2.0, 3.0], &[1, 3], |t, x| {
            t.softmax_cross_entropy(x, &[2], Reduction::Mean)
        });
        assert!(dims.is_empty());
        assert_close(&y, &[0.40760596444438], 1e-14);
    }

    /// Every reduction of a two-row cross-entropy.
    #[test]
    fn cross_entropy_reductions() {
        let x = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0];
        let per = [0.40760596444438013, 0.5514447139320509];
        let loss = |r| run(&x, &[2, 3], |t, x| t.softmax_cross_entropy(x, &[2, 0], r));
        let (none, dims) = loss(Reduction::None);
        assert_eq!(dims, [2]);
        assert_close(&none, &per, 1e-14);
        let sum = per[0] + per[1];
        assert_close(&loss(Reduction::Sum).0, &[sum], 1e-14);
        assert_close(&loss(Reduction::Mean).0, &[sum / 2.0], 1e-14);
        assert_close(&loss(Reduction::BatchMean).0, &[sum / 2.0], 1e-14);
    }

    /// A probability of exactly zero gives a large finite loss.
    #[test]
    fn cross_entropy_with_zero_probability_is_finite() {
        let (y, _) = run(&[1e4, -1e4], &[1, 2], |t, x| {
            t.softmax_cross_entropy(x, &[1], Reduction::Mean)
        });
        assert_eq!(y, [2e4]);
    }

    /// NaN in the logits gives a NaN loss, in every reduction and wherever it sits.
    #[test]
    fn cross_entropy_nan_logit_gives_nan_loss() {
        for r in [
            Reduction::None,
            Reduction::Sum,
            Reduction::Mean,
            Reduction::BatchMean,
        ] {
            for x in [[f64::NAN, 1.0], [1.0, f64::NAN]] {
                for label in [0, 1] {
                    let (y, _) = run(&x, &[1, 2], |t, x| t.softmax_cross_entropy(x, &[label], r));
                    assert!(y[0].is_nan(), "{r:?} {x:?} {label}");
                }
            }
        }
    }

    /// A label past the last class panics with the label and the valid range.
    #[test]
    #[should_panic(expected = "label 3 at index 1 is out of range 0..3")]
    fn label_out_of_range_panics() {
        run(&[0.0; 6], &[2, 3], |t, x| {
            t.softmax_cross_entropy(x, &[0, 3], Reduction::Mean)
        });
    }

    /// The label count must match the batch.
    #[test]
    #[should_panic(expected = "softmax_cross_entropy: got 1 labels for a batch of 2")]
    fn label_count_mismatch_panics() {
        run(&[0.0; 6], &[2, 3], |t, x| {
            t.softmax_cross_entropy(x, &[0], Reduction::Mean)
        });
    }

    /// MSE of `[1, 2]` against zeros under each reduction.
    #[test]
    fn mse_by_hand() {
        let loss = |r| {
            let store = ParamStore::<f64>::new_generic();
            let mut graph = Graph::<f64>::new_generic();
            let mut tape = graph.tape(&store.values);
            let p = tape.input(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().view());
            let t = tape.input(Tensor::from_vec(vec![0.0, 0.0], &[2]).unwrap().view());
            let y = tape.mse(p, t, r);
            tape.value(y).data().to_vec()
        };
        assert_eq!(loss(Reduction::Mean), [2.5]);
        assert_eq!(loss(Reduction::Sum), [5.0]);
        assert_eq!(loss(Reduction::None), [1.0, 4.0]);
        assert_eq!(loss(Reduction::BatchMean), [2.5]);
    }

    /// `BatchMean` divides by the first dimension, `Mean` by every element.
    #[test]
    fn mse_batch_mean_divides_by_first_dimension() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let p = tape.input(Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap().view());
        let t = tape.input(Tensor::from_vec(vec![0.0; 6], &[2, 3]).unwrap().view());
        let batch = tape.mse(p, t, Reduction::BatchMean);
        let mean = tape.mse(p, t, Reduction::Mean);
        assert_eq!(tape.value(batch).data(), &[3.0]);
        assert_eq!(tape.value(mean).data(), &[1.0]);
    }

    /// Different shapes panic, naming the operation and both shapes.
    #[test]
    #[should_panic(expected = "mse: shapes differ, [2] and [3]")]
    fn mse_shape_mismatch_panics() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let p = tape.input(Tensor::<f32>::zeros(&[2]).view());
        let t = tape.input(Tensor::<f32>::zeros(&[3]).view());
        tape.mse(p, t, Reduction::Mean);
    }

    /// The cross-entropy gradient is `(softmax - onehot) / batch` for `Mean`, and the labels
    /// of an earlier loss on the same tape are not mixed up with a later one's.
    #[test]
    fn cross_entropy_gradient_by_hand() {
        let mut store = ParamStore::<f64>::new_generic();
        let p = store.add(Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0], &[2, 3]).unwrap());
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let z = tape.param(p);
        let first = tape.softmax_cross_entropy(z, &[2, 0], Reduction::Mean);
        let second = tape.softmax_cross_entropy(z, &[0, 1], Reduction::Sum);
        let zero = tape.scale(second, 0.0);
        let loss = tape.add(first, zero);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        let sm = |row: &[f64]| -> Vec<f64> {
            let e: Vec<f64> = row.iter().map(|v| v.exp()).collect();
            let s: f64 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        };
        let (r0, r1) = (sm(&[1.0, 2.0, 3.0]), sm(&[1.0, 0.0, 0.0]));
        let want = [
            r0[0] / 2.0,
            r0[1] / 2.0,
            (r0[2] - 1.0) / 2.0,
            (r1[0] - 1.0) / 2.0,
            r1[1] / 2.0,
            r1[2] / 2.0,
        ];
        assert_close(store.grad(p).data(), &want, 1e-15);
    }
}
