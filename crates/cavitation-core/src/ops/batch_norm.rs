//! Batch normalisation on the tape: `batch_norm2d`, and `batch_norm2d_relu` with a ReLU
//! after it in the same passes, with their backward pass, on images stored as flat NCHW
//! rows.
//!
//! Both passes run one task per channel on the thread pool. Each channel's sums run in a
//! fixed order on the thread that takes it, so every result is the same for any thread
//! count.

use crate::graph::{BackwardCtx, BufferUpdate, ParamId, Var};
use crate::kernels::conv::Parts;
use crate::kernels::{self, pool};
use crate::ops::Op;
use crate::ops::activation::relu_derivative;
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
        self.batch_norm2d_with(x, weight, bias, running, channels, spatial, eps, false)
    }

    /// Records [`batch_norm2d`](Self::batch_norm2d) followed by ReLU as one node: the same
    /// value, gradients and running statistics, bit for bit, as `batch_norm2d` then
    /// [`relu`](Self::relu), with one pass over memory fewer in each direction: the forward
    /// pass applies ReLU as it writes, and the backward pass masks the gradient inside batch
    /// norm's own two passes instead of in a pass of ReLU's. The node's value is ReLU's
    /// output; batch norm's own output is not kept.
    ///
    /// The backward pass multiplies the incoming gradient by ReLU's derivative at the node's
    /// value, which is positive exactly where ReLU's input was and NaN exactly where it was
    /// NaN, so the product is the one ReLU's own backward pass would pass on.
    ///
    /// # Panics
    ///
    /// As [`batch_norm2d`](Self::batch_norm2d).
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let x = tape.input(x.view());
    /// let fused = tape.batch_norm2d_relu(x, None, None, None, 1, 2, 0.0);
    /// let bn = tape.batch_norm2d(x, None, None, None, 1, 2, 0.0);
    /// let separate = tape.relu(bn);
    /// assert_eq!(tape.value(fused).data(), tape.value(separate).data());
    /// assert_eq!(tape.value(fused).data()[..2], [0.0, 0.0]);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm2d_relu(
        &mut self,
        x: Var,
        weight: Option<Var>,
        bias: Option<Var>,
        running: Option<RunningStats>,
        channels: usize,
        spatial: usize,
        eps: f64,
    ) -> Var {
        self.batch_norm2d_with(x, weight, bias, running, channels, spatial, eps, true)
    }

    /// [`batch_norm2d`](Self::batch_norm2d), with ReLU applied to each output value as it is
    /// written when `relu` is true.
    #[allow(clippy::too_many_arguments)]
    fn batch_norm2d_with(
        &mut self,
        x: Var,
        weight: Option<Var>,
        bias: Option<Var>,
        running: Option<RunningStats>,
        channels: usize,
        spatial: usize,
        eps: f64,
        relu: bool,
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
                let value = self.param_value(id);
                let dims = value.shape().dims();
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
                // One task per channel: each channel's sums run in the same order on one
                // thread, so the statistics are the same for any thread count.
                let parts = [means, inv, unbiased].map(|p| Parts(p.as_mut_ptr()));
                pool::run(kernels::threads(), channels, &|t| {
                    let c = t.index;
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
                    // SAFETY: task `c` alone writes element `c` of each of the three
                    // statistics, each `channels` long.
                    let [mp, ip, up] = parts.each_ref().map(|p| unsafe { &mut p.slice(c, 1)[0] });
                    *mp = mean;
                    *ip = T::ONE / (sq / T::from_f64(m as f64) + eps_t).sqrt();
                    *up = sq / T::from_f64((m - 1) as f64);
                });
            }
        }
        let mut out = self.take_buffer(batch * width);
        {
            let xv = self.data(x);
            let w = weight.map(|v| self.data(v));
            let b = bias.map(|v| self.data(v));
            let outs = Parts(out.as_mut_ptr());
            let stats = &stats;
            pool::run(kernels::threads(), channels, &|t| {
                let c = t.index;
                let (mean, inv) = (stats[c], stats[channels + c]);
                let g = w.map_or(T::ONE, |w| w[c]);
                let s = b.map_or(T::ZERO, |b| b[c]);
                for n in 0..batch {
                    let at = n * width + c * spatial;
                    // SAFETY: task `c` alone writes channel `c`'s values of each image.
                    let o = unsafe { outs.slice(at, spatial) };
                    if relu {
                        for (o, &v) in o.iter_mut().zip(&xv[at..][..spatial]) {
                            *o = ((v - mean) * inv * g + s).relu();
                        }
                    } else {
                        for (o, &v) in o.iter_mut().zip(&xv[at..][..spatial]) {
                            *o = (v - mean) * inv * g + s;
                        }
                    }
                }
            });
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
        let start = self.push_labels(&[
            channels,
            spatial,
            usize::from(use_running),
            usize::from(relu),
        ]);
        let so = Shape::new(&[batch, width]);
        self.push(Op::BatchNorm2d(x, weight, bias, stats_var, start), so, out)
    }
}

/// Values per block when the fused ReLU's gradient is masked ahead of batch norm's loops.
const BLOCK: usize = 256;

/// `out[k] = dy[k] · ReLU'(y[k])`, the product ReLU's own backward pass writes. A loop of its
/// own, so it compiles to a vector select: inside batch norm's sums, which keep their order
/// and so cannot be vectorised, the derivative would be a branch on the sign of each value,
/// mispredicted about half the time.
fn relu_masked<T: Real>(out: &mut [T], dy: &[T], y: &[T]) {
    for ((o, &g), &v) in out.iter_mut().zip(dy).zip(y) {
        *o = g * relu_derivative(v);
    }
}

/// Backward of `batch_norm2d` and `batch_norm2d_relu`. With `x̂ = (x − mean) · inv` and
/// `g = dy · γ` per channel over its `m` values: `dβ = Σ dy`, `dγ = Σ dy · x̂`, and
/// `dx = inv · g` with the running statistics, else `dx = inv / m · (m · g − Σ g − x̂ · Σ g ·
/// x̂)` through the batch's. With the fused ReLU, `dy` is first multiplied by ReLU's
/// derivative at `out`'s value, ReLU's output: on `x`'s first gradient write the sums pass
/// stores that product in `x`'s gradient buffer and the second pass turns it into `dx` in
/// place; when `x` already has a gradient, each pass masks `dy` a block at a time.
// The op's five fields, the node, the gradient and the context; a struct would only rename them.
#[allow(clippy::too_many_arguments)]
pub(super) fn backward<T: Real>(
    x: Var,
    weight: Option<Var>,
    bias: Option<Var>,
    stats: Var,
    start: usize,
    out: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let l = &ctx.labels()[start..start + 4];
    let (channels, spatial, use_running, relu) = (l[0], l[1], l[2] == 1, l[3] == 1);
    let batch = ctx.shape(x).dims()[0];
    let width = channels * spatial;
    let m = batch * spatial;
    let (xv, sv) = (ctx.value(x), ctx.value(stats));
    let w = weight.map(|v| ctx.value(v));
    let yv = ctx.value(out);
    let mut sums = ctx.scratch(2 * channels);
    let threads = kernels::threads();
    let mf = T::from_f64(m as f64);
    // Per channel: Σ dy and Σ dy · x̂, where `dy` is read from `src` (the incoming gradient,
    // masked a block at a time with the fused ReLU) or, when `store` is given, masked into
    // that buffer first and summed from there.
    let sum_pass = |sums: &mut [T], store: Option<&mut [T]>| {
        let sp = Parts(sums.as_mut_ptr());
        let st = store.map(|s| Parts(s.as_mut_ptr()));
        pool::run(threads, channels, &|t| {
            let c = t.index;
            let (mean, inv) = (sv[c], sv[channels + c]);
            let (mut s, mut sx) = (T::ZERO, T::ZERO);
            for n in 0..batch {
                let at = n * width + c * spatial;
                if let Some(st) = &st {
                    // SAFETY: task `c` alone writes channel `c`'s values of each image.
                    let d = unsafe { st.slice(at, spatial) };
                    relu_masked(d, &out_grad[at..][..spatial], &yv[at..][..spatial]);
                    for (&g, &v) in d.iter().zip(&xv[at..][..spatial]) {
                        s += g;
                        sx += g * ((v - mean) * inv);
                    }
                } else if relu {
                    let mut block = [T::ZERO; BLOCK];
                    for k0 in (0..spatial).step_by(BLOCK) {
                        let len = BLOCK.min(spatial - k0);
                        relu_masked(
                            &mut block[..len],
                            &out_grad[at + k0..][..len],
                            &yv[at + k0..][..len],
                        );
                        for (&g, &v) in block[..len].iter().zip(&xv[at + k0..][..len]) {
                            s += g;
                            sx += g * ((v - mean) * inv);
                        }
                    }
                } else {
                    for (&g, &v) in out_grad[at..][..spatial].iter().zip(&xv[at..][..spatial]) {
                        s += g;
                        sx += g * ((v - mean) * inv);
                    }
                }
            }
            // SAFETY: task `c` alone writes elements `c` and `channels + c` of `sums`.
            unsafe {
                sp.slice(c, 1)[0] = s;
                sp.slice(channels + c, 1)[0] = sx;
            }
        });
    };
    let mut summed = false;
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            // With the fused ReLU and the first gradient for `x`, the masked gradient is kept
            // in `buf` by the sums pass and turned into `dx` in place, so it is masked once.
            let in_place = relu && beta == T::ZERO;
            if in_place {
                sum_pass(&mut sums, Some(&mut *buf));
            } else {
                sum_pass(&mut sums, None);
            }
            summed = true;
            let bufs = Parts(buf.as_mut_ptr());
            let sums = &sums;
            pool::run(threads, channels, &|t| {
                let c = t.index;
                let (mean, inv) = (sv[c], sv[channels + c]);
                let g = w.map_or(T::ONE, |w| w[c]);
                // Σ (dy · γ) and Σ (dy · γ) · x̂ over the channel.
                let (sg, sgx) = (sums[c] * g, sums[channels + c] * g);
                let dx = |dy: T, v: T| {
                    let gy = dy * g;
                    if use_running {
                        gy * inv
                    } else {
                        let xhat = (v - mean) * inv;
                        inv / mf * (mf * gy - sg - xhat * sgx)
                    }
                };
                for n in 0..batch {
                    let at = n * width + c * spatial;
                    // SAFETY: task `c` alone writes channel `c`'s values of each image.
                    let dst = unsafe { bufs.slice(at, spatial) };
                    let xs = &xv[at..][..spatial];
                    if in_place {
                        for (d, &v) in dst.iter_mut().zip(xs) {
                            *d = dx(*d, v);
                        }
                        continue;
                    }
                    let mut block = [T::ZERO; BLOCK];
                    let step = if relu { BLOCK } else { spatial.max(1) };
                    for k0 in (0..spatial).step_by(step) {
                        let len = step.min(spatial - k0);
                        let dys: &[T] = if relu {
                            relu_masked(
                                &mut block[..len],
                                &out_grad[at + k0..][..len],
                                &yv[at + k0..][..len],
                            );
                            &block[..len]
                        } else {
                            &out_grad[at + k0..][..len]
                        };
                        for ((d, &dy), &v) in
                            dst[k0..k0 + len].iter_mut().zip(dys).zip(&xs[k0..k0 + len])
                        {
                            let dx = dx(dy, v);
                            *d = if beta == T::ZERO { dx } else { *d + dx };
                        }
                    }
                }
            });
        });
    }
    if !summed {
        sum_pass(&mut sums, None);
    }
    for (v, offset) in [(bias, 0), (weight, channels)] {
        if let Some(v) = v
            && let Some(slot) = ctx.slot(v)
        {
            slot.add_from(&sums[offset..][..channels]);
        }
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

    /// One fused-against-separate comparison in `T`: values, the gradients of the input, γ
    /// and β, and the running statistics after the update, all compared by bits.
    ///
    /// Channel 0 holds values equal to its mean and has β = 0, so those outputs are exactly
    /// 0 (ReLU's derivative there is 0); channel 2 holds a NaN, which makes its statistics
    /// and outputs NaN (the derivative is NaN); the other βs are not 0. With `affine` false
    /// there is no γ or β. The loss is `Σ y · r`, plus `Σ x · q` with `reuse_x`: that term,
    /// recorded after the batch norm, gives `x` its first gradient, so the batch norm's
    /// backward pass adds to it (the block path) rather than writing it (the in-place path).
    fn fused_matches_separate<T: crate::Real>(training: bool, affine: bool, reuse_x: bool) {
        let (batch, channels, spatial) = (3, 3, 4);
        let len = batch * channels * spatial;
        let mut xs: Vec<T> = values(len, 7).into_iter().map(T::from_f64).collect();
        // Channel 0: image 0's values 1, 2, 3, 2 and the other images' all 2, so the
        // channel's mean is exactly 2 and those values normalise to exactly 0.
        for n in 0..batch {
            for (k, v) in [1.0, 2.0, 3.0, 2.0].into_iter().enumerate() {
                xs[n * channels * spatial + k] = T::from_f64(if n == 0 { v } else { 2.0 });
            }
        }
        xs[2 * spatial + 1] = T::from_f64(f64::NAN); // channel 2 of image 0
        let r: Vec<T> = values(len, 9).into_iter().map(T::from_f64).collect();
        let q: Vec<T> = values(len, 11).into_iter().map(T::from_f64).collect();
        let run = |fused: bool| {
            let mut store = ParamStore::<T>::new_generic();
            let w = store.add(
                Tensor::from_vec(
                    values(channels, 1).into_iter().map(T::from_f64).collect(),
                    &[channels],
                )
                .unwrap(),
            );
            let betas = [0.0, 0.3, -0.4].map(T::from_f64).to_vec();
            let b = store.add(Tensor::from_vec(betas, &[channels]).unwrap());
            let stats = RunningStats {
                mean: store
                    .add(Tensor::from_vec(vec![T::from_f64(0.25); channels], &[channels]).unwrap()),
                var: store
                    .add(Tensor::from_vec(vec![T::from_f64(1.5); channels], &[channels]).unwrap()),
                count: None,
                momentum: Some(0.1),
            };
            let xi = store.add(Tensor::from_vec(xs.clone(), &[batch, channels * spatial]).unwrap());
            let mut graph = Graph::new_generic();
            let mut tape = if training {
                graph.training_tape(&store.values, 0, 1)
            } else {
                graph.tape(&store.values)
            };
            let xv = tape.param(xi);
            let (wv, bv) = if affine {
                (Some(tape.param(w)), Some(tape.param(b)))
            } else {
                (None, None)
            };
            let y = if fused {
                tape.batch_norm2d_relu(xv, wv, bv, Some(stats), channels, spatial, 1e-5)
            } else {
                let bn = tape.batch_norm2d(xv, wv, bv, Some(stats), channels, spatial, 1e-5);
                tape.relu(bn)
            };
            let ys = tape.value(y).data().to_vec();
            let rv = tape.input(
                Tensor::from_vec(r.clone(), &[batch, channels * spatial])
                    .unwrap()
                    .view(),
            );
            let yr = tape.mul(y, rv);
            let mut loss = tape.sum(yr);
            if reuse_x {
                let qv = tape.input(
                    Tensor::from_vec(q.clone(), &[batch, channels * spatial])
                        .unwrap()
                        .view(),
                );
                let xq = tape.mul(xv, qv);
                let c = tape.sum(xq);
                loss = tape.add(loss, c);
            }
            tape.backward(loss, &mut store.grads);
            drop(tape);
            graph.apply_buffer_updates(&mut store.values);
            let ids = if affine { vec![xi, w, b] } else { vec![xi] };
            let grads: Vec<Vec<T>> = ids
                .iter()
                .map(|&id| store.grad(id).data().to_vec())
                .collect();
            let running: Vec<Vec<T>> = [stats.mean, stats.var]
                .iter()
                .map(|&id| store.value(id).data().to_vec())
                .collect();
            (ys, grads, running)
        };
        let bits = |v: &[T]| v.iter().map(|x| x.to_f64().to_bits()).collect::<Vec<_>>();
        let (fused, separate) = (run(true), run(false));
        assert_eq!(bits(&fused.0), bits(&separate.0), "values");
        for (k, (a, b)) in fused.1.iter().zip(&separate.1).enumerate() {
            assert_eq!(bits(a), bits(b), "gradient {k} (input, γ, β)");
        }
        for (k, (a, b)) in fused.2.iter().zip(&separate.2).enumerate() {
            assert_eq!(bits(a), bits(b), "running statistic {k} (mean, variance)");
        }
        // The cases the test is for are there: exact zeros, positives, and NaN.
        assert!(fused.0.contains(&T::ZERO));
        assert!(fused.0.iter().any(|v| *v > T::ZERO));
        assert!(fused.0.iter().any(|v| v.to_f64().is_nan()));
    }

    /// `batch_norm2d_relu` gives the bits of `batch_norm2d` then `relu`, in f32 and f64,
    /// with batch statistics on a training tape, with and without γ and β.
    #[test]
    fn fused_relu_matches_separate_in_training() {
        for affine in [true, false] {
            for reuse_x in [false, true] {
                fused_matches_separate::<f32>(true, affine, reuse_x);
                fused_matches_separate::<f64>(true, affine, reuse_x);
            }
        }
    }

    /// The same with the running statistics on an evaluation tape.
    #[test]
    fn fused_relu_matches_separate_in_evaluation() {
        for affine in [true, false] {
            for reuse_x in [false, true] {
                fused_matches_separate::<f32>(false, affine, reuse_x);
                fused_matches_separate::<f64>(false, affine, reuse_x);
            }
        }
    }

    /// The gradient check passes for the fused operation with batch statistics, for the
    /// input, γ and β, in f64. The loss `Σ y · r` has a non-zero slope where ReLU's output
    /// is 0, so a backward pass that left out ReLU's derivative would fail here.
    #[test]
    fn fused_relu_grad_check() {
        let (batch, channels, spatial) = (3, 2, 4);
        let mut store = ParamStore::<f64>::new_generic();
        let w = store.add(Tensor::from_vec(values(channels, 1), &[channels]).unwrap());
        let b = store.add(Tensor::from_vec(values(channels, 2), &[channels]).unwrap());
        let x = Tensor::from_vec(
            values(batch * channels * spatial, 4),
            &[batch, channels * spatial],
        )
        .unwrap();
        let r = Tensor::from_vec(
            values(batch * channels * spatial, 5),
            &[batch, channels * spatial],
        )
        .unwrap();
        gradcheck_params(&mut store, &[x, r], |tape, v| {
            let (wv, bv) = (tape.param(w), tape.param(b));
            let y = tape.batch_norm2d_relu(v[0], Some(wv), Some(bv), None, channels, spatial, 1e-5);
            let s = tape.mul(y, v[1]);
            tape.sum(s)
        })
        .assert_passed();
    }
}
