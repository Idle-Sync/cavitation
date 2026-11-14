//! Batch normalisation on the tape: `batch_norm2d`, with its backward pass, on images stored
//! as flat NCHW rows.

use crate::graph::{BackwardCtx, BufferUpdate, ParamId, Var};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// The running statistics of a batch normalisation: the buffers it reads in evaluation and
/// moves towards each batch's statistics in training, as PyTorch's `BatchNorm2d` keeps them.
///
/// ```
/// use cavitation_core::{ParamStore, RunningStats, Tensor};
/// let mut store = ParamStore::new();
/// let stats = RunningStats {
///     mean: store.add(Tensor::zeros(&[8])),
///     var: store.add(Tensor::full(&[8], 1.0f32)),
///     count: None,
///     momentum: Some(0.1),
/// };
/// assert_eq!(stats.momentum, Some(0.1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningStats {
    /// The running mean, one value per channel.
    pub mean: ParamId,
    /// The running variance, one value per channel.
    pub var: ParamId,
    /// The number of batches seen (one value), PyTorch's `num_batches_tracked`, if kept.
    pub count: Option<ParamId>,
    /// The weight of each batch's statistics in the update, or `None` for the cumulative
    /// average (which needs `count`).
    pub momentum: Option<f64>,
}

/// The operation's name, used in panic messages.
const NAME: &str = "batch_norm2d";

impl<T: Real> Tape<'_, T> {
    /// Records 2-D batch normalisation of the images in `x`, `[batch, channels · spatial]`
    /// (each channel's `spatial` values together, NCHW order), as PyTorch's `BatchNorm2d`
    /// computes it.
    ///
    /// - On a training tape, or with no `running` statistics, each channel is normalised
    ///   with its batch mean and biased variance over the batch and the image:
    ///   `(x − mean) / √(var + eps)`. On a training tape with `running` statistics, the
    ///   batch mean and unbiased variance are recorded for [`Graph::apply_buffer_updates`]
    ///   (the tape only reads the values, so it cannot write the buffers itself).
    /// - On an evaluation tape with `running` statistics, each channel is normalised with
    ///   them instead, read from the parameter values without being recorded, so they never
    ///   take a gradient.
    ///
    /// Then `weight` (γ, `[channels]`) scales and `bias` (β) shifts each channel, if given.
    /// The gradient is the closed form, through the batch statistics in training.
    ///
    /// # Panics
    ///
    /// Panics, naming the operation and the shapes, if `x` is not `[batch, channels ·
    /// spatial]`, `weight`, `bias` or a running statistic is not `[channels]`, or batch
    /// statistics are taken over one value per channel (PyTorch refuses that too).
    ///
    /// [`Graph::apply_buffer_updates`]: crate::Graph::apply_buffer_updates
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// // Two images of one channel of two values: the channel's mean is 2.5, its variance 1.25.
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.batch_norm2d(x, None, None, None, 1, 2, 0.0);
    /// let want = [-1.5f32, -0.5, 0.5, 1.5].map(|v| v / 1.25f32.sqrt());
    /// for (a, b) in tape.value(y).data().iter().zip(want) {
    ///     assert!((a - b).abs() < 1e-6);
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm2d(
        &mut self,
        x: Var,
        weight: Option<Var>,
        bias: Option<Var>,
        running: Option<RunningStats>,
        channels: usize,
        spatial: usize,
        eps: f64,
    ) -> Var {
        let sx = self.shape(x);
        let width = channels * spatial;
        assert!(
            sx.rank() == 2 && sx.dims()[1] == width,
            "{NAME}: the input must be [batch, {channels} · {spatial}], got {:?}",
            sx.dims()
        );
        let batch = sx.dims()[0];
        for (what, v) in [("weight", weight), ("bias", bias)] {
            if let Some(v) = v {
                assert!(
                    self.shape(v).dims() == [channels],
                    "{NAME}: the {what} must be [{channels}], got {:?}",
                    self.shape(v).dims()
                );
            }
        }
        if let Some(r) = running {
            for (what, id) in [("running mean", r.mean), ("running variance", r.var)] {
                let dims = self.param_value(id).shape().dims().to_vec();
                assert!(
                    dims == [channels],
                    "{NAME}: the {what} must be [{channels}], got {dims:?}"
                );
            }
        }
        let use_running = running.is_some() && !self.is_training();
        let m = batch * spatial;
        // stats: each channel's mean, then 1 / √(var + eps), then (training only) the
        // unbiased variance; an input node of its own, which the backward pass reads.
        let mut stats = self.take_buffer(3 * channels);
        let eps_t = T::from_f64(eps);
        {
            let xv = self.data(x);
            let (means, rest) = stats.split_at_mut(channels);
            let (inv, unbiased) = rest.split_at_mut(channels);
            if use_running {
                let r = running.expect("checked");
                let (rm, rv) = (self.param_value(r.mean), self.param_value(r.var));
                for c in 0..channels {
                    means[c] = rm.data()[c];
                    inv[c] = T::ONE / (rv.data()[c] + eps_t).sqrt();
                    unbiased[c] = T::ZERO;
                }
            } else {
                assert!(
                    m > 1,
                    "{NAME}: batch statistics need more than one value per channel, got \
                     {batch} images of {spatial} values"
                );
                for c in 0..channels {
                    let mut sum = T::ZERO;
                    for n in 0..batch {
                        for &v in &xv[n * width + c * spatial..][..spatial] {
                            sum += v;
                        }
                    }
                    let mean = sum / T::from_f64(m as f64);
                    let mut sq = T::ZERO;
                    for n in 0..batch {
                        for &v in &xv[n * width + c * spatial..][..spatial] {
                            let d = v - mean;
                            sq += d * d;
                        }
                    }
                    means[c] = mean;
                    inv[c] = T::ONE / (sq / T::from_f64(m as f64) + eps_t).sqrt();
                    unbiased[c] = sq / T::from_f64((m - 1) as f64);
                }
            }
        }
        let mut out = self.take_buffer(batch * width);
        {
            let xv = self.data(x);
            let w = weight.map(|v| self.data(v));
            let b = bias.map(|v| self.data(v));
            for n in 0..batch {
                for c in 0..channels {
                    let (mean, inv) = (stats[c], stats[channels + c]);
                    let g = w.map_or(T::ONE, |w| w[c]);
                    let s = b.map_or(T::ZERO, |b| b[c]);
                    let at = n * width + c * spatial;
                    for (o, &v) in out[at..][..spatial].iter_mut().zip(&xv[at..][..spatial]) {
                        *o = (v - mean) * inv * g + s;
                    }
                }
            }
        }
        if let Some(r) = running
            && self.is_training()
        {
            // The batch means, then the unbiased variances, for the update.
            let update = BufferUpdate {
                mean: r.mean,
                var: r.var,
                count: r.count,
                momentum: r.momentum,
                start: 0,
                channels,
            };
            let (means, rest) = stats.split_at(channels);
            let unbiased = &rest[channels..];
            self.push_buffer_update(update, means, unbiased);
        }
        let stats_var = self.push(Op::Input, Shape::new(&[3 * channels]), stats);
        let start = self.push_labels(&[channels, spatial, usize::from(use_running)]);
        let so = Shape::new(&[batch, width]);
        self.push(Op::BatchNorm2d(x, weight, bias, stats_var, start), so, out)
    }
}

/// Backward of `batch_norm2d`. With `x̂ = (x − mean) · inv` and `g = dy · γ` per channel over
/// its `m` values: `dβ = Σ dy`, `dγ = Σ dy · x̂`, and `dx = inv · g` with the running
/// statistics, else `dx = inv / m · (m · g − Σ g − x̂ · Σ g · x̂)` through the batch's.
pub(super) fn backward<T: Real>(
    x: Var,
    weight: Option<Var>,
    bias: Option<Var>,
    stats: Var,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let l = &ctx.labels()[start..start + 3];
    let (channels, spatial, use_running) = (l[0], l[1], l[2] == 1);
    let batch = ctx.shape(x).dims()[0];
    let width = channels * spatial;
    let m = batch * spatial;
    let (xv, sv) = (ctx.value(x), ctx.value(stats));
    let w = weight.map(|v| ctx.value(v));
    // Per channel: Σ dy and Σ dy · x̂.
    let mut sums = ctx.scratch(2 * channels);
    for c in 0..channels {
        let (mean, inv) = (sv[c], sv[channels + c]);
        let (mut s, mut sx) = (T::ZERO, T::ZERO);
        for n in 0..batch {
            let at = n * width + c * spatial;
            for (&g, &v) in out_grad[at..][..spatial].iter().zip(&xv[at..][..spatial]) {
                s += g;
                sx += g * ((v - mean) * inv);
            }
        }
        sums[c] = s;
        sums[channels + c] = sx;
    }
    for (v, offset) in [(bias, 0), (weight, channels)] {
        if let Some(v) = v
            && let Some(slot) = ctx.slot(v)
        {
            slot.add_from(&sums[offset..][..channels]);
        }
    }
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            let mf = T::from_f64(m as f64);
            for c in 0..channels {
                let (mean, inv) = (sv[c], sv[channels + c]);
                let g = w.map_or(T::ONE, |w| w[c]);
                // Σ (dy · γ) and Σ (dy · γ) · x̂ over the channel.
                let (sg, sgx) = (sums[c] * g, sums[channels + c] * g);
                for n in 0..batch {
                    let at = n * width + c * spatial;
                    let dst = &mut buf[at..][..spatial];
                    for ((d, &dy), &v) in dst
                        .iter_mut()
                        .zip(&out_grad[at..][..spatial])
                        .zip(&xv[at..][..spatial])
                    {
                        let gy = dy * g;
                        let dx = if use_running {
                            gy * inv
                        } else {
                            let xhat = (v - mean) * inv;
                            inv / mf * (mf * gy - sg - xhat * sgx)
                        };
                        *d = if beta == T::ZERO { dx } else { *d + dx };
                    }
                }
            }
        });
    }
    ctx.release(sums);
}

/// Tests for batch normalisation on the tape: by hand, the gradient check in both modes, the
/// running statistics and the refusals.
#[cfg(test)]
mod tests {
    use super::RunningStats;
    use crate::{Graph, ParamStore, Rng, Tensor, gradcheck_params};

    /// `n` values uniform in [-2, 2) from `Rng::seed(seed)`.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::seed(seed);
        (0..n).map(|_| rng.range(-2.0, 2.0)).collect()
    }

    /// The gradient check passes with batch statistics, for the input, γ and β, in f64.
    #[test]
    fn batch_norm_grad_check() {
        let (batch, channels, spatial) = (3, 2, 4);
        let mut store = ParamStore::<f64>::new_generic();
        let w = store.add(Tensor::from_vec(values(channels, 1), &[channels]).unwrap());
        let b = store.add(Tensor::from_vec(values(channels, 2), &[channels]).unwrap());
        let x = Tensor::from_vec(
            values(batch * channels * spatial, 4),
            &[batch, channels * spatial],
        )
        .unwrap();
        gradcheck_params(&mut store, &[x], |tape, v| {
            let (wv, bv) = (tape.param(w), tape.param(b));
            let y = tape.batch_norm2d(v[0], Some(wv), Some(bv), None, channels, spatial, 1e-5);
            let t = tape.tanh(y);
            let s = tape.mul(t, t);
            tape.sum(s)
        })
        .assert_passed();
    }

    /// With running statistics on an evaluation tape the operation is a per-channel affine
    /// map: `dx = dy · γ / √(var + eps)`, `dγ = Σ dy · x̂`, `dβ = Σ dy`, and the running
    /// statistics take no gradient (they are read, not recorded).
    #[test]
    fn running_statistics_gradients_by_hand() {
        let mut store = ParamStore::<f64>::new_generic();
        let w = store.add(Tensor::from_vec(vec![2.0], &[1]).unwrap());
        let b = store.add(Tensor::from_vec(vec![0.5], &[1]).unwrap());
        let stats = RunningStats {
            mean: store.add(Tensor::from_vec(vec![1.0], &[1]).unwrap()),
            var: store.add(Tensor::from_vec(vec![4.0], &[1]).unwrap()),
            count: None,
            momentum: Some(0.1),
        };
        let xi = store.add(Tensor::from_vec(vec![3.0, -1.0], &[2, 1]).unwrap());
        let mut graph = Graph::new_generic();
        let mut tape = graph.tape(&store.values);
        let (xv, wv, bv) = (tape.param(xi), tape.param(w), tape.param(b));
        let y = tape.batch_norm2d(xv, Some(wv), Some(bv), Some(stats), 1, 1, 0.0);
        assert_eq!(tape.value(y).data(), &[2.5, -1.5]); // 2 · (x − 1) / 2 + 0.5
        let loss = tape.sum(y);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(xi).data(), &[1.0, 1.0]);
        assert_eq!(store.grad(w).data(), &[0.0]); // x̂ = 1 and -1
        assert_eq!(store.grad(b).data(), &[2.0]);
        assert!(store.grads.is_stale(stats.mean) && store.grads.is_stale(stats.var));
    }

    /// A constant channel normalises to 0, so its output is β; with γ = 1 and β = 0 each
    /// channel of the output has mean 0 and (biased) variance 1 up to `eps`.
    #[test]
    fn batch_statistics_normalise_each_channel() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::new_generic();
        let mut tape = graph.tape(&store.values);
        let mut x = values(4 * 2 * 3, 7);
        for n in 0..4 {
            for s in 0..3 {
                x[n * 6 + 3 + s] = 5.0;
            }
        }
        let xv = tape.input(Tensor::from_vec(x, &[4, 6]).unwrap().view());
        let y = tape.batch_norm2d(xv, None, None, None, 2, 3, 1e-12);
        let out = tape.value(y).data().to_vec();
        let ch0: Vec<f64> = (0..4)
            .flat_map(|n| out[n * 6..n * 6 + 3].to_vec())
            .collect();
        let mean = ch0.iter().sum::<f64>() / 12.0;
        let var = ch0.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 12.0;
        assert!(
            mean.abs() < 1e-12 && (var - 1.0).abs() < 1e-9,
            "{mean} {var}"
        );
        assert!((0..4).all(|n| out[n * 6 + 3..n * 6 + 6].iter().all(|&v| v == 0.0)));
    }

    /// Batch statistics over one value per channel are refused.
    #[test]
    #[should_panic(expected = "batch statistics need more than one value per channel")]
    fn one_value_per_channel_is_refused() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.training_tape(&store.values, 0, 1);
        let x = tape.input(Tensor::full(&[1, 3], 1.0f32).view());
        tape.batch_norm2d(x, None, None, None, 3, 1, 1e-5);
    }

    /// Three training steps move the running statistics as PyTorch does (momentum 0.1, and the
    /// cumulative average with a batch count), applied only after each tape is dropped; an
    /// evaluation tape records no update.
    #[test]
    fn running_statistics_follow_pytorchs_updates() {
        for momentum in [Some(0.1), None] {
            let mut store = ParamStore::<f64>::new_generic();
            let stats = RunningStats {
                mean: store.add(Tensor::zeros_generic(&[1])),
                var: store.add(Tensor::full(&[1], 1.0)),
                count: Some(store.add(Tensor::zeros_generic(&[1]))),
                momentum,
            };
            let mut graph = Graph::new_generic();
            let (mut mean, mut var) = (0.0f64, 1.0f64);
            for step in 1..=3u64 {
                let x: Vec<f64> = (0..4).map(|i| (i as f64) * step as f64).collect();
                let bm = x.iter().sum::<f64>() / 4.0;
                let bv = x.iter().map(|v| (v - bm).powi(2)).sum::<f64>() / 3.0;
                let m = momentum.unwrap_or(1.0 / step as f64);
                mean = m * bm + (1.0 - m) * mean;
                var = m * bv + (1.0 - m) * var;
                let mut tape = graph.training_tape(&store.values, 0, step);
                let xv = tape.input(Tensor::from_vec(x, &[4, 1]).unwrap().view());
                tape.batch_norm2d(xv, None, None, Some(stats), 1, 1, 1e-5);
                drop(tape);
                graph.apply_buffer_updates(&mut store.values);
            }
            assert!((store.value(stats.mean).data()[0] - mean).abs() < 1e-12);
            assert!((store.value(stats.var).data()[0] - var).abs() < 1e-12);
            assert_eq!(store.value(stats.count.unwrap()).data(), &[3.0]);
            let mut tape = graph.tape(&store.values);
            let xv = tape.input(Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap().view());
            tape.batch_norm2d(xv, None, None, Some(stats), 1, 1, 1e-5);
            drop(tape);
            graph.apply_buffer_updates(&mut store.values);
            assert_eq!(store.value(stats.count.unwrap()).data(), &[3.0]);
        }
    }
}
