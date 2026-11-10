//! Softmax, log-softmax and the losses: cross-entropy, NLL, MSE, L1, Huber, smooth L1, binary
//! cross-entropy (on probabilities and on logits) and KL divergence.
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

/// Checks that `shape` is `[batch, classes]` with one label per row, each below `classes`,
/// and returns `(batch, classes)`.
///
/// # Panics
///
/// Panics, naming `name`, `what` (the input's role) and the values, on any mismatch.
fn check_labels(name: &str, what: &str, shape: &Shape, labels: &[usize]) -> (usize, usize) {
    if shape.rank() != 2 {
        panic!(
            "{name}: {what} must be rank 2 [batch, classes], but the shape is {:?}",
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
    (batch, classes)
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
        let (batch, classes) = check_labels("softmax_cross_entropy", "logits", &shape, labels);
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

    /// The shared start of every two-input elementwise loss: checks that `a` and `b` have the
    /// same shape (panicking with `name` and both shapes otherwise), then fills the
    /// recorded value from `per(a_i, b_i)` through [`reduce_into`]. Returns the value's shape
    /// and data.
    fn pairwise_loss(
        &mut self,
        name: &str,
        a: Var,
        b: Var,
        reduction: Reduction,
        mut per: impl FnMut(T, T) -> T,
    ) -> (Shape, Vec<T>) {
        let shape = self.shape(a);
        let other = self.shape(b);
        if shape != other {
            panic!(
                "{name}: shapes differ, {:?} and {:?}",
                shape.dims(),
                other.dims()
            );
        }
        let n = shape.len();
        let batch = batch_of(&shape);
        let out_shape = reduction.output_shape(shape);
        let mut out = self.take_buffer(out_shape.len());
        let (x, y) = (self.data(a), self.data(b));
        reduce_into(&mut out, reduction, n, batch, |i| per(x[i], y[i]));
        (out_shape, out)
    }

    /// Records the binary cross-entropy `-[t ln p + (1 - t) ln(1 - p)]` of probabilities
    /// `probs` against `target`, elementwise, reduced by `reduction`. Each logarithm is
    /// clamped to be at least -100, as PyTorch does, so `p = 0` or `p = 1` gives a finite
    /// loss (100 for a wrong certain prediction). Every clamped logarithm adds one to
    /// [`Tape::clamped_logs`], which the health checks read, so a clamp is never silent.
    /// NaN in `probs` or `target` gives a NaN loss, and a NaN is not a clamp.
    ///
    /// The unreduced loss has the shape of `probs`. For [`Reduction::BatchMean`] the batch is
    /// the first dimension. Both inputs take gradients. For `probs` the gradient per element
    /// is `(p - t) / (p (1 - p))` times the upstream gradient and the reduction's factor,
    /// where neither logarithm was clamped. Where one was clamped, its term is constant and
    /// contributes 0, so only the other term's derivative remains (`-t / p` or
    /// `(1 - t) / (1 - p)`); this never divides by zero. For `target` the gradient is
    /// `ln(1 - p) - ln p` with the clamped logarithms.
    ///
    /// # Panics
    ///
    /// Panics, naming `binary_cross_entropy` and both shapes, if the shapes differ (there is
    /// no broadcasting), or if either variable was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let p = tape.input(Tensor::from_vec(vec![0.8f32, 0.0], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap().view());
    /// let each = tape.binary_cross_entropy(p, t, Reduction::None);
    /// assert!((tape.value(each).data()[0] - 0.223_143_55).abs() < 1e-6);
    /// assert_eq!(tape.value(each).data()[1], 100.0);
    /// assert_eq!(tape.clamped_logs(), 1);
    /// ```
    pub fn binary_cross_entropy(&mut self, probs: Var, target: Var, reduction: Reduction) -> Var {
        let mut clamped = 0usize;
        let (shape, out) =
            self.pairwise_loss("binary_cross_entropy", probs, target, reduction, |p, t| {
                let (lp, lq) = (clamped_ln(p), clamped_ln(T::ONE - p));
                clamped += usize::from(lp.1) + usize::from(lq.1);
                -(t * lp.0 + (T::ONE - t) * lq.0)
            });
        self.add_clamped_logs(clamped);
        self.push(Op::BinaryCrossEntropy(probs, target, reduction), shape, out)
    }

    /// Records the binary cross-entropy of `logits` against `target`, with the sigmoid fused
    /// in: `max(z, 0) - z t + ln(1 + e^(-|z|))`, which never overflows. Elementwise, reduced
    /// by `reduction`. NaN in either input gives a NaN loss. The unreduced loss has the shape
    /// of `logits`; for [`Reduction::BatchMean`] the batch is the first dimension. The
    /// gradient for `logits` is `σ(z) - t` and for `target` is `-z`, each times the upstream
    /// gradient and the reduction's factor.
    ///
    /// # Panics
    ///
    /// Panics, naming `binary_cross_entropy_with_logits` and both shapes, if the shapes
    /// differ, or if either variable was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let z = tape.input(Tensor::from_vec(vec![1000.0f32], &[1]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![0.0f32], &[1]).unwrap().view());
    /// let loss = tape.binary_cross_entropy_with_logits(z, t, Reduction::Mean);
    /// assert_eq!(tape.value(loss).data(), &[1000.0]);
    /// ```
    pub fn binary_cross_entropy_with_logits(
        &mut self,
        logits: Var,
        target: Var,
        reduction: Reduction,
    ) -> Var {
        let (shape, out) = self.pairwise_loss(
            "binary_cross_entropy_with_logits",
            logits,
            target,
            reduction,
            |z, t| {
                let positive = if z > T::ZERO { z } else { T::ZERO };
                positive - z * t + (-z.abs()).exp().ln_1p()
            },
        );
        self.push(Op::BceWithLogits(logits, target, reduction), shape, out)
    }

    /// Records the negative log-likelihood of class `labels` under `log_probs`
    /// (`[batch, classes]`, for example a [`Tape::log_softmax`] output): row `i` has loss
    /// `-log_probs[i, labels[i]]`. The unreduced loss has shape `[batch]`; `reduction` turns
    /// it into the recorded value ([`Reduction::Mean`] and [`Reduction::BatchMean`] both
    /// divide by the batch). NaN anywhere in a row gives that row a NaN loss. The gradient is
    /// `-1` at each row's label times the upstream gradient and the reduction's factor, and 0
    /// elsewhere.
    ///
    /// # Panics
    ///
    /// Panics, naming `nll` and the values, if `log_probs` is not rank 2, if `labels.len()`
    /// differs from the batch, or if a label is not below the number of classes; all before
    /// any loss is computed. Also panics if `log_probs` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let lp = tape.input(Tensor::from_vec(vec![-1.0f32, -2.0, -3.0], &[1, 3]).unwrap().view());
    /// let loss = tape.nll(lp, &[1], Reduction::Mean);
    /// assert_eq!(tape.value(loss).data(), &[2.0]);
    /// ```
    pub fn nll(&mut self, log_probs: Var, labels: &[usize], reduction: Reduction) -> Var {
        let shape = self.shape(log_probs);
        let (batch, classes) = check_labels("nll", "log_probs", &shape, labels);
        let start = self.push_labels(labels);
        let out_shape = reduction.output_shape(Shape::new(&[batch]));
        let mut out = self.take_buffer(out_shape.len());
        let rows = self.data(log_probs);
        reduce_into(&mut out, reduction, batch, batch, |i| {
            let row = &rows[i * classes..][..classes];
            if row.iter().any(|&v| is_nan(v)) {
                T::from_f64(f64::NAN)
            } else {
                -row[labels[i]]
            }
        });
        self.push(Op::Nll(log_probs, reduction, start), out_shape, out)
    }

    /// Records the absolute error `|pred - target|` elementwise, reduced by `reduction`. The
    /// unreduced loss has the shape of `pred`; for [`Reduction::BatchMean`] the batch is the
    /// first dimension. NaN gives a NaN loss. The gradient for `pred` is `sign(pred - target)`
    /// (0 where they are equal) times the upstream gradient and the reduction's factor, and
    /// the negative of it for `target`.
    ///
    /// # Panics
    ///
    /// Panics, naming `l1` and both shapes, if the shapes differ, or if either variable was
    /// not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let p = tape.input(Tensor::from_vec(vec![1.0f32, -2.0], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![0.0f32, 0.0], &[2]).unwrap().view());
    /// let loss = tape.l1(p, t, Reduction::Mean);
    /// assert_eq!(tape.value(loss).data(), &[1.5]);
    /// ```
    pub fn l1(&mut self, pred: Var, target: Var, reduction: Reduction) -> Var {
        let (shape, out) = self.pairwise_loss("l1", pred, target, reduction, |p, t| (p - t).abs());
        self.push(Op::L1(pred, target, reduction), shape, out)
    }

    /// Records the Huber loss (PyTorch `HuberLoss`) of `pred` against `target`, elementwise,
    /// reduced by `reduction`: with `d = pred - target`, `0.5 d²` when `|d| < delta`, else
    /// `delta (|d| - 0.5 delta)`. PyTorch's default `delta` is 1. The unreduced loss has the
    /// shape of `pred`; for [`Reduction::BatchMean`] the batch is the first dimension. NaN
    /// gives a NaN loss. The gradient for `pred` is `d` when `|d| < delta`, else
    /// `delta · sign(d)`, times the upstream gradient and the reduction's factor, and the
    /// negative of it for `target`.
    ///
    /// # Panics
    ///
    /// Panics, naming `huber`, if `delta` is not positive; naming `huber` and both shapes if
    /// the shapes differ; and if either variable was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let p = tape.input(Tensor::from_vec(vec![0.5f32, 3.0], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![0.0f32, 0.0], &[2]).unwrap().view());
    /// let loss = tape.huber(p, t, 1.0, Reduction::None);
    /// assert_eq!(tape.value(loss).data(), &[0.125, 2.5]);
    /// ```
    pub fn huber(&mut self, pred: Var, target: Var, delta: T, reduction: Reduction) -> Var {
        if delta.partial_cmp(&T::ZERO) != Some(std::cmp::Ordering::Greater) {
            panic!("huber: delta must be positive, but it is {delta:?}");
        }
        let (shape, out) = self.pairwise_loss("huber", pred, target, reduction, |p, t| {
            huber_value(p - t, delta)
        });
        let delta = delta.to_f64();
        self.push(Op::Huber(pred, target, delta, reduction), shape, out)
    }

    /// Records the smooth L1 loss (PyTorch `SmoothL1Loss`) of `pred` against `target`,
    /// elementwise, reduced by `reduction`: with `d = pred - target`, `0.5 d² / beta` when
    /// `|d| < beta`, else `|d| - 0.5 beta`. PyTorch's default `beta` is 1; `beta = 0` is the
    /// L1 loss. The unreduced loss has the shape of `pred`; for [`Reduction::BatchMean`] the
    /// batch is the first dimension. NaN gives a NaN loss. The gradient for `pred` is
    /// `d / beta` when `|d| < beta`, else `sign(d)`, times the upstream gradient and the
    /// reduction's factor, and the negative of it for `target`.
    ///
    /// # Panics
    ///
    /// Panics, naming `smooth_l1`, if `beta` is negative; naming `smooth_l1` and both shapes
    /// if the shapes differ; and if either variable was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let p = tape.input(Tensor::from_vec(vec![0.5f32, 3.0], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![0.0f32, 0.0], &[2]).unwrap().view());
    /// let loss = tape.smooth_l1(p, t, 1.0, Reduction::None);
    /// assert_eq!(tape.value(loss).data(), &[0.125, 2.5]);
    /// ```
    pub fn smooth_l1(&mut self, pred: Var, target: Var, beta: T, reduction: Reduction) -> Var {
        if is_nan(beta) || beta < T::ZERO {
            panic!("smooth_l1: beta must not be negative, but it is {beta:?}");
        }
        let (shape, out) = self.pairwise_loss("smooth_l1", pred, target, reduction, |p, t| {
            smooth_l1_value(p - t, beta)
        });
        let beta = beta.to_f64();
        self.push(Op::SmoothL1(pred, target, beta, reduction), shape, out)
    }

    /// Records the KL divergence in PyTorch's `kl_div` convention: `log_probs` holds
    /// log-probabilities and `target` probabilities, and each element is
    /// `t (ln t - input)`, with 0 where `t` is exactly 0 (`0 ln 0 = 0`). A negative target is
    /// the caller's error and gives NaN. NaN in either input gives a NaN loss, also where
    /// `t` is 0. The unreduced loss has the shape of `log_probs`; for [`Reduction::BatchMean`]
    /// (PyTorch's recommended reduction) the batch is the first dimension. The gradient for
    /// `log_probs` is `-t` and for `target` is `ln t - input + 1` (0 where `t` is 0), each
    /// times the upstream gradient and the reduction's factor.
    ///
    /// # Panics
    ///
    /// Panics, naming `kl_div` and both shapes, if the shapes differ, or if either variable
    /// was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let half = 0.5f32.ln();
    /// let x = tape.input(Tensor::from_vec(vec![half, half], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![0.25f32, 0.75], &[2]).unwrap().view());
    /// let loss = tape.kl_div(x, t, Reduction::Sum);
    /// assert!((tape.value(loss).data()[0] - 0.130_812).abs() < 1e-5);
    /// ```
    pub fn kl_div(&mut self, log_probs: Var, target: Var, reduction: Reduction) -> Var {
        let (shape, out) = self.pairwise_loss("kl_div", log_probs, target, reduction, |x, t| {
            kl_value(x, t)
        });
        self.push(Op::KlDiv(log_probs, target, reduction), shape, out)
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

/// `ln(x)` clamped to be at least -100, and whether the clamp applied. NaN is not clamped:
/// the comparison is false for it, so it passes through.
fn clamped_ln<T: Real>(x: T) -> (T, bool) {
    let l = x.ln();
    let floor = T::from_f64(-100.0);
    if l < floor { (floor, true) } else { (l, false) }
}

/// The sign of `d`: 1, -1, or 0 for zero; NaN stays NaN so that a NaN loss has a NaN gradient.
fn sign<T: Real>(d: T) -> T {
    if d > T::ZERO {
        T::ONE
    } else if d < T::ZERO {
        -T::ONE
    } else if is_nan(d) {
        d
    } else {
        T::ZERO
    }
}

/// The logistic sigmoid, evaluated so that `e^z` is never computed for positive `z`.
fn stable_sigmoid<T: Real>(z: T) -> T {
    if z >= T::ZERO {
        T::ONE / (T::ONE + (-z).exp())
    } else {
        let e = z.exp();
        e / (T::ONE + e)
    }
}

/// The Huber loss of the difference `d`. NaN falls to the linear branch and stays NaN.
fn huber_value<T: Real>(d: T, delta: T) -> T {
    let a = d.abs();
    if a < delta {
        T::from_f64(0.5) * d * d
    } else {
        delta * (a - T::from_f64(0.5) * delta)
    }
}

/// The smooth L1 loss of the difference `d`. NaN falls to the linear branch and stays NaN.
fn smooth_l1_value<T: Real>(d: T, beta: T) -> T {
    let a = d.abs();
    if a < beta {
        T::from_f64(0.5) * d * d / beta
    } else {
        a - T::from_f64(0.5) * beta
    }
}

/// One element of the KL divergence: `t (ln t - x)`, 0 where `t` is 0 unless `x` is NaN.
fn kl_value<T: Real>(x: T, t: T) -> T {
    if t == T::ZERO {
        // 0 · ln 0 = 0, but a NaN input still gives a NaN loss.
        if is_nan(x) { x } else { T::ZERO }
    } else {
        t * (t.ln() - x)
    }
}

/// Backward of a loss over two same-shaped inputs `a` and `b`: `grad(a_i, b_i, w_i)` returns
/// the gradients for `a_i` and `b_i`, with `w_i` the upstream gradient of item `i` from
/// [`Upstream`] (so `grad` multiplies by it). Every element is written when a slot
/// overwrites.
fn pairwise_backward<T: Real>(
    a: Var,
    b: Var,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
    grad: impl Fn(T, T, T) -> (T, T),
) {
    let shape = ctx.shape(a);
    let up = Upstream::new(out_grad, reduction, shape.len(), batch_of(&shape));
    let (x, y) = (ctx.value(a), ctx.value(b));
    for (var, second) in [(a, false), (b, true)] {
        if let Some(slot) = ctx.slot(var) {
            slot.update_with(|buf, beta| {
                for (i, d) in buf.iter_mut().enumerate() {
                    let (ga, gb) = grad(x[i], y[i], up.at(i));
                    let v = if second { gb } else { ga };
                    *d = if beta == T::ZERO { v } else { *d + v };
                }
            });
        }
    }
}

/// Backward of binary cross-entropy: see [`Tape::binary_cross_entropy`] for the convention at
/// the clamp.
pub(super) fn bce_backward<T: Real>(
    probs: Var,
    target: Var,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    pairwise_backward(probs, target, reduction, out_grad, ctx, |p, t, w| {
        let ((lp, p_clamped), (lq, q_clamped)) = (clamped_ln(p), clamped_ln(T::ONE - p));
        let dp = match (p_clamped, q_clamped) {
            (false, false) => (p - t) / (p * (T::ONE - p)),
            (true, false) => (T::ONE - t) / (T::ONE - p),
            (false, true) => -t / p,
            (true, true) => T::ZERO,
        };
        (dp * w, (lq - lp) * w)
    });
}

/// Backward of binary cross-entropy with logits: `σ(z) - t` for the logits, `-z` for the
/// target.
pub(super) fn bce_logits_backward<T: Real>(
    logits: Var,
    target: Var,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    pairwise_backward(logits, target, reduction, out_grad, ctx, |z, t, w| {
        ((stable_sigmoid(z) - t) * w, -z * w)
    });
}

/// Backward of NLL: `-w_i` at each row's label, 0 elsewhere, with `w_i` from [`Upstream`].
/// Every element is written when the slot overwrites.
pub(super) fn nll_backward<T: Real>(
    log_probs: Var,
    reduction: Reduction,
    labels_start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let shape = ctx.shape(log_probs);
    let (batch, classes) = (shape.dims()[0], shape.dims()[1]);
    if classes == 0 {
        return;
    }
    let labels = &ctx.labels()[labels_start..][..batch];
    let up = Upstream::new(out_grad, reduction, batch, batch);
    let Some(slot) = ctx.slot(log_probs) else {
        return;
    };
    slot.update_with(|buf, beta| {
        for (i, row) in buf.chunks_exact_mut(classes).enumerate() {
            let w = up.at(i);
            for (j, d) in row.iter_mut().enumerate() {
                let v = if j == labels[i] { -w } else { T::ZERO };
                *d = if beta == T::ZERO { v } else { *d + v };
            }
        }
    });
}

/// Backward of L1: `sign(p - t)` for the prediction, the negative for the target.
pub(super) fn l1_backward<T: Real>(
    pred: Var,
    target: Var,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    pairwise_backward(pred, target, reduction, out_grad, ctx, |p, t, w| {
        let g = sign(p - t) * w;
        (g, -g)
    });
}

/// Backward of Huber: `d` inside `delta`, else `delta · sign(d)`; the negative for the target.
pub(super) fn huber_backward<T: Real>(
    pred: Var,
    target: Var,
    delta: f64,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let delta = T::from_f64(delta);
    pairwise_backward(pred, target, reduction, out_grad, ctx, |p, t, w| {
        let d = p - t;
        let g = if d.abs() < delta { d } else { delta * sign(d) } * w;
        (g, -g)
    });
}

/// Backward of smooth L1: `d / beta` inside `beta`, else `sign(d)`; the negative for the target.
pub(super) fn smooth_l1_backward<T: Real>(
    pred: Var,
    target: Var,
    beta: f64,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let beta = T::from_f64(beta);
    pairwise_backward(pred, target, reduction, out_grad, ctx, |p, t, w| {
        let d = p - t;
        let g = if d.abs() < beta { d / beta } else { sign(d) } * w;
        (g, -g)
    });
}

/// Backward of KL divergence: `-t` for the input, `ln t - x + 1` for the target; both 0 where
/// `t` is 0.
pub(super) fn kl_div_backward<T: Real>(
    log_probs: Var,
    target: Var,
    reduction: Reduction,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    pairwise_backward(log_probs, target, reduction, out_grad, ctx, |x, t, w| {
        if t == T::ZERO {
            (T::ZERO, T::ZERO)
        } else {
            (-t * w, (t.ln() - x + T::ONE) * w)
        }
    });
}

/// Tests for softmax, log-softmax, the losses built on them, and the other losses.
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

    /// Records `f` on a fresh f64 tape over two inputs `a` and `b` (shape `dims`) and returns
    /// the output values.
    fn run2(
        a: &[f64],
        b: &[f64],
        dims: &[usize],
        f: impl FnOnce(&mut crate::Tape<'_, f64>, crate::Var, crate::Var) -> crate::Var,
    ) -> Vec<f64> {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let a = tape.input(Tensor::from_vec(a.to_vec(), dims).unwrap().view());
        let b = tape.input(Tensor::from_vec(b.to_vec(), dims).unwrap().view());
        let y = f(&mut tape, a, b);
        tape.value(y).data().to_vec()
    }

    /// Binary cross-entropy of one probability, and the clamp count it leaves on the tape.
    fn bce_one(p: f64, t: f64) -> (f64, usize) {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let pv = tape.input(Tensor::from_vec(vec![p], &[1]).unwrap().view());
        let tv = tape.input(Tensor::from_vec(vec![t], &[1]).unwrap().view());
        let y = tape.binary_cross_entropy(pv, tv, Reduction::Mean);
        (tape.value(y).data()[0], tape.clamped_logs())
    }

    /// BCE of 0.8 against 1 and against 0.
    #[test]
    fn bce_by_hand() {
        assert_close(&[bce_one(0.8, 1.0).0], &[0.2231435513142097], 1e-15);
        assert_close(&[bce_one(0.8, 0.0).0], &[1.6094379124341003], 1e-15);
    }

    /// A probability of exactly 0 or 1 gives the clamped value 100, and the clamp is counted.
    #[test]
    fn bce_at_zero_and_one_is_finite() {
        assert_eq!(bce_one(0.0, 1.0), (100.0, 1));
        assert_eq!(bce_one(1.0, 0.0), (100.0, 1));
    }

    /// No clamp, no count; a new tape starts again at zero.
    #[test]
    fn bce_without_clamping_counts_zero() {
        assert_eq!(bce_one(0.8, 1.0).1, 0);
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        {
            let mut tape = graph.tape(&store.values);
            let p = tape.input(Tensor::from_vec(vec![0.0, 1.0, 0.5], &[3]).unwrap().view());
            let t = tape.input(Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap().view());
            tape.binary_cross_entropy(p, t, Reduction::Sum);
            assert_eq!(tape.clamped_logs(), 2);
        }
        assert_eq!(graph.tape(&store.values).clamped_logs(), 0);
    }

    /// NaN in the probability or the target gives a NaN loss, and a NaN is not a clamp.
    #[test]
    fn bce_nan_gives_nan_loss() {
        for (p, t) in [(f64::NAN, 1.0), (f64::NAN, 0.0), (0.5, f64::NAN)] {
            let (loss, clamped) = bce_one(p, t);
            assert!(loss.is_nan(), "{p} {t}");
            assert_eq!(clamped, 0);
        }
    }

    /// Where a logarithm is clamped its term has gradient 0; the other term still counts.
    #[test]
    fn bce_gradient_at_the_clamp() {
        let grad = |p: f64, t: f64| {
            let mut store = ParamStore::<f64>::new_generic();
            let id = store.add(Tensor::from_vec(vec![p], &[1]).unwrap());
            let mut graph = Graph::<f64>::new_generic();
            let mut tape = graph.tape(&store.values);
            let pv = tape.param(id);
            let tv = tape.input(Tensor::from_vec(vec![t], &[1]).unwrap().view());
            let y = tape.binary_cross_entropy(pv, tv, Reduction::Sum);
            tape.backward(y, &mut store.grads);
            drop(tape);
            store.grad(id).data()[0]
        };
        assert_eq!(grad(0.0, 1.0), 0.0);
        assert_eq!(grad(0.0, 0.25), 0.75);
        assert_eq!(grad(1.0, 1.0), -1.0);
        assert_eq!(grad(1.0, 0.25), -0.25);
        assert_close(&[grad(0.8, 1.0)], &[-1.25], 1e-15);
        assert_close(&[grad(0.25, 0.5)], &[-0.25 / (0.25 * 0.75)], 1e-15);
    }

    /// Different shapes panic, naming the operation and both shapes.
    #[test]
    #[should_panic(expected = "binary_cross_entropy: shapes differ, [2] and [3]")]
    fn bce_shape_mismatch_panics() {
        run2(&[0.5; 2], &[0.5; 2], &[2], |t, p, _| {
            let other = t.input(Tensor::from_vec(vec![0.5; 3], &[3]).unwrap().view());
            t.binary_cross_entropy(p, other, Reduction::Mean)
        });
    }

    /// BCE with logits at z = 2, t = 1 and z = -3, t = 0.
    #[test]
    fn bce_with_logits_by_hand() {
        let y = run2(&[2.0, -3.0], &[1.0, 0.0], &[2], |t, z, y| {
            t.binary_cross_entropy_with_logits(z, y, Reduction::None)
        });
        assert_close(&y, &[0.1269280110429725, 0.04858735157374206], 1e-15);
    }

    /// A logit of 1000 does not overflow.
    #[test]
    fn bce_with_logits_large_is_finite() {
        let y = run2(&[1000.0, -1000.0], &[0.0, 1.0], &[2], |t, z, y| {
            t.binary_cross_entropy_with_logits(z, y, Reduction::None)
        });
        assert_eq!(y, [1000.0, 1000.0]);
    }

    /// NaN logits or targets give a NaN loss.
    #[test]
    fn bce_with_logits_nan_gives_nan_loss() {
        for (z, y) in [(f64::NAN, 1.0), (f64::NAN, 0.0), (0.0, f64::NAN)] {
            let l = run2(&[z], &[y], &[1], |t, z, y| {
                t.binary_cross_entropy_with_logits(z, y, Reduction::Mean)
            });
            assert!(l[0].is_nan(), "{z} {y}");
        }
    }

    /// NLL of one row picks the negated entry at the label.
    #[test]
    fn nll_by_hand() {
        let (y, dims) = run(&[-1.0, -2.0, -3.0], &[1, 3], |t, x| {
            t.nll(x, &[1], Reduction::Mean)
        });
        assert!(dims.is_empty());
        assert_eq!(y, [2.0]);
    }

    /// Every reduction of a two-row NLL.
    #[test]
    fn nll_reductions() {
        let x = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let loss = |r| run(&x, &[2, 3], |t, x| t.nll(x, &[2, 0], r));
        let (none, dims) = loss(Reduction::None);
        assert_eq!(dims, [2]);
        assert_eq!(none, [3.0, 4.0]);
        assert_eq!(loss(Reduction::Sum).0, [7.0]);
        assert_eq!(loss(Reduction::Mean).0, [3.5]);
        assert_eq!(loss(Reduction::BatchMean).0, [3.5]);
    }

    /// NaN anywhere in a row gives that row a NaN loss.
    #[test]
    fn nll_nan_gives_nan_loss() {
        for x in [[f64::NAN, -1.0], [-1.0, f64::NAN]] {
            let (y, _) = run(&x, &[1, 2], |t, x| t.nll(x, &[0], Reduction::Mean));
            assert!(y[0].is_nan(), "{x:?}");
        }
    }

    /// A label past the last class panics with the label and the valid range.
    #[test]
    #[should_panic(expected = "nll: label 3 at index 1 is out of range 0..3")]
    fn nll_label_out_of_range_panics() {
        run(&[0.0; 6], &[2, 3], |t, x| {
            t.nll(x, &[0, 3], Reduction::Mean)
        });
    }

    /// The label count must match the batch.
    #[test]
    #[should_panic(expected = "nll: got 1 labels for a batch of 2")]
    fn nll_label_count_mismatch_panics() {
        run(&[0.0; 6], &[2, 3], |t, x| t.nll(x, &[0], Reduction::Mean));
    }

    /// The log-probabilities must be rank 2.
    #[test]
    #[should_panic(expected = "nll: log_probs must be rank 2")]
    fn nll_rank_panics() {
        run(&[0.0; 3], &[3], |t, x| t.nll(x, &[0], Reduction::Mean));
    }

    /// L1 of `[1, -2]` against zeros.
    #[test]
    fn l1_by_hand() {
        let l1 = |r| run2(&[1.0, -2.0], &[0.0, 0.0], &[2], |t, p, y| t.l1(p, y, r));
        assert_eq!(l1(Reduction::Mean), [1.5]);
        assert_eq!(l1(Reduction::Sum), [3.0]);
        assert_eq!(l1(Reduction::None), [1.0, 2.0]);
    }

    /// The L1 gradient is the sign of the difference, 0 where they are equal.
    #[test]
    fn l1_gradient_is_sign() {
        let mut store = ParamStore::<f64>::new_generic();
        let id = store.add(Tensor::from_vec(vec![1.0, -2.0, 0.5], &[3]).unwrap());
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let p = tape.param(id);
        let y = tape.input(Tensor::from_vec(vec![0.0, 0.0, 0.5], &[3]).unwrap().view());
        let l = tape.l1(p, y, Reduction::Sum);
        tape.backward(l, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(id).data(), &[1.0, -1.0, 0.0]);
    }

    /// Huber with delta 1 at d = 0.5 and d = 3.
    #[test]
    fn huber_by_hand() {
        let y = run2(&[0.5, 3.0], &[0.0, 0.0], &[2], |t, p, y| {
            t.huber(p, y, 1.0, Reduction::None)
        });
        assert_eq!(y, [0.125, 2.5]);
    }

    /// Smooth L1 in both regimes and with a beta other than 1.
    #[test]
    fn smooth_l1_by_hand() {
        let sl1 = |d: f64, beta: f64| {
            run2(&[d], &[0.0], &[1], |t, p, y| {
                t.smooth_l1(p, y, beta, Reduction::Sum)
            })[0]
        };
        assert_eq!(sl1(0.5, 1.0), 0.125);
        assert_eq!(sl1(1.0, 2.0), 0.25);
        assert_eq!(sl1(3.0, 1.0), 2.5);
        assert_eq!(sl1(3.0, 0.0), 3.0);
    }

    /// NaN gives a NaN loss in L1, Huber and smooth L1.
    #[test]
    fn robust_losses_nan_give_nan_loss() {
        let r = Reduction::Mean;
        for (a, b) in [(f64::NAN, 0.0), (0.0, f64::NAN)] {
            assert!(run2(&[a], &[b], &[1], |t, p, y| t.l1(p, y, r))[0].is_nan());
            assert!(run2(&[a], &[b], &[1], |t, p, y| t.huber(p, y, 1.0, r))[0].is_nan());
            assert!(run2(&[a], &[b], &[1], |t, p, y| t.smooth_l1(p, y, 1.0, r))[0].is_nan());
        }
    }

    /// A non-positive delta, or a negative beta, panics naming the operation.
    #[test]
    #[should_panic(expected = "huber: delta must be positive, but it is 0")]
    fn huber_zero_delta_panics() {
        run2(&[0.0], &[0.0], &[1], |t, p, y| {
            t.huber(p, y, 0.0, Reduction::Mean)
        });
    }

    /// A negative beta panics.
    #[test]
    #[should_panic(expected = "smooth_l1: beta must not be negative, but it is -1")]
    fn smooth_l1_negative_beta_panics() {
        run2(&[0.0], &[0.0], &[1], |t, p, y| {
            t.smooth_l1(p, y, -1.0, Reduction::Mean)
        });
    }

    /// KL divergence of ln[0.5, 0.5] against [0.25, 0.75].
    #[test]
    fn kl_div_by_hand() {
        let half = 0.5f64.ln();
        let y = run2(&[half, half], &[0.25, 0.75], &[2], |t, x, y| {
            t.kl_div(x, y, Reduction::Sum)
        });
        assert_close(&y, &[0.13081203594113697], 1e-15);
    }

    /// A target of exactly 0 contributes 0 to the loss and the gradient.
    #[test]
    fn kl_div_zero_target_is_zero() {
        let mut store = ParamStore::<f64>::new_generic();
        let x = store.add(Tensor::from_vec(vec![-0.7, -0.7], &[2]).unwrap());
        let y = store.add(Tensor::from_vec(vec![0.0, 0.5], &[2]).unwrap());
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let (xv, yv) = (tape.param(x), tape.param(y));
        let l = tape.kl_div(xv, yv, Reduction::None);
        let want = 0.5 * (0.5f64.ln() + 0.7);
        assert_close(tape.value(l).data(), &[0.0, want], 1e-15);
        let s = tape.sum(l);
        tape.backward(s, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(x).data()[0], 0.0);
        assert_eq!(store.grad(y).data()[0], 0.0);
        assert_close(store.grad(x).data(), &[0.0, -0.5], 1e-15);
        assert_close(store.grad(y).data(), &[0.0, 0.5f64.ln() + 0.7 + 1.0], 1e-15);
    }

    /// NaN in the input gives a NaN loss even where the target is 0.
    #[test]
    fn kl_div_nan_gives_nan_loss() {
        for (x, y) in [(f64::NAN, 0.0), (f64::NAN, 0.5), (-1.0, f64::NAN)] {
            let l = run2(&[x], &[y], &[1], |t, x, y| t.kl_div(x, y, Reduction::Mean));
            assert!(l[0].is_nan(), "{x} {y}");
        }
    }

    /// Every two-input loss rejects different shapes, naming itself and both shapes.
    #[test]
    fn two_input_losses_name_themselves_on_shape_mismatch() {
        type Loss = fn(&mut crate::Tape<'_, f64>, crate::Var, crate::Var) -> crate::Var;
        let losses: [(&str, Loss); 5] = [
            ("binary_cross_entropy_with_logits", |t, a, b| {
                t.binary_cross_entropy_with_logits(a, b, Reduction::Mean)
            }),
            ("l1", |t, a, b| t.l1(a, b, Reduction::Mean)),
            ("huber", |t, a, b| t.huber(a, b, 1.0, Reduction::Mean)),
            ("smooth_l1", |t, a, b| {
                t.smooth_l1(a, b, 1.0, Reduction::Mean)
            }),
            ("kl_div", |t, a, b| t.kl_div(a, b, Reduction::Mean)),
        ];
        for (name, f) in losses {
            let caught = std::panic::catch_unwind(|| {
                run2(&[0.5; 2], &[0.5; 2], &[2], |t, a, _| {
                    let other = t.input(Tensor::from_vec(vec![0.5; 3], &[3]).unwrap().view());
                    f(t, a, other)
                })
            });
            let err = caught.unwrap_err();
            let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
            assert!(
                msg.contains(&format!("{name}: shapes differ, [2] and [3]")),
                "{msg}"
            );
        }
    }
}
