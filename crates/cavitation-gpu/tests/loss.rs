//! The losses against the tape's, on every backend this machine has.
//!
//! - **Softmax cross-entropy:** the loss and the logits' gradient within their stated bounds
//!   of `Tape::softmax_cross_entropy` on an `f64` store, from one row to 70,000 rows and from
//!   one class to 1,000. A NaN or infinite logit, or a bad label, gives a NaN row that the
//!   finite flag catches. On confident rows the loss carries no bias.
//! - **Mean squared error:** the gradient bit for bit against `Tape::mse` on an `f32` store,
//!   and the loss within its bound of the `f64` tape. Both are checked up to more values than
//!   one grid row of workgroups holds, and an overflowed square and a NaN reach the loss.
//! - **Negative log-likelihood:** the gradient bit for bit against `Tape::nll` on an `f32`
//!   store and the loss within its bound of the `f64` tape, on the cross-entropy's shapes;
//!   NaN and ±∞ as on the CPU, and a bad label giving a NaN row.
//! - **L1, Huber and smooth L1:** the gradient bit for bit against `Tape::l1`, `Tape::huber`
//!   and `Tape::smooth_l1` on an `f32` store (smooth L1's quadratic branch, which divides,
//!   within `8u`), and the loss within the bound derived at `REGRESSIONS` of the `f64`
//!   tape, up to more values than one grid row of workgroups holds; each value at the edges
//!   (±0, the threshold and one ulp either side, a threshold of 0 or +∞, NaN, ±∞ and
//!   overflow) the `f32` tape's bits; the tape's refusal of a bad threshold.
//! - **Binary cross-entropy:** of probabilities and of logits, with the mean and the sum: the
//!   loss within the bound derived at `bce_term` of the exact loss, the gradient against
//!   `Tape::binary_cross_entropy` and `Tape::binary_cross_entropy_with_logits` on an `f32`
//!   store (bit for bit where a logarithm was clamped, within `8u` where the probabilities'
//!   gradient divides, within the bound at `logit_grad` for logits), and the clamp count
//!   `Tape::clamped_logs`, up to more values than one grid row of workgroups holds; each value
//!   at the edges (0 and 1, the clamp's threshold, subnormals, ±100 and larger logits, values
//!   outside `[0, 1]`, NaN and ±∞) as the `f32` tape gives it; a NaN reaching the loss; the
//!   batch mean against `Reduction::BatchMean`.
//! - **KL divergence:** with the mean, the sum and the batch mean: the gradient bit for bit
//!   against `Tape::kl_div` on an `f32` store, and the loss within the bound derived at
//!   `kl_term` of the exact loss, up to more values than one grid row of workgroups holds;
//!   each value at the edges (targets of ±0, subnormals, 1 and one ulp either side, above 1,
//!   negative, NaN and ±∞; log-probabilities of ±0 to ±`f32::MAX`, NaN and ±∞, and an
//!   overflowing loss) as the `f32` tape gives it; a NaN or bad value reaching the loss and
//!   the finite flag; the refusals.
//!
//! Every loss is written at word 1 of a buffer of sentinels, to show nothing else changes. The
//! tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, Real, Reduction, Tensor};
use cavitation_gpu::kernels::{
    BinaryCrossEntropy, BinaryInput, FiniteFlag, KlDivergence, LossReduction, MeanSquaredError,
    NegativeLogLikelihood, Regression, RegressionLoss, SoftmaxCrossEntropy,
};
use cavitation_gpu::{Gpu, PooledBuffer};
use common::{FLOOR, Kit, U, assert_exact, assert_within, for_each_gpu, seeded_values, sum_bound};

/// What fills the words around a loss.
const SENTINEL: f32 = 7.0;

/// A kit whose steps can move `values` values.
fn kit(gpu: &Gpu, values: usize) -> Kit {
    Kit::new(gpu, 4 * values as u64 + 65_536)
}

/// The relative error of a sum of `len` terms in the per-channel sums' fixed tree, in units of
/// `u`: each term passes through at most 15 + 8 + (chunks − 1) additions, whatever order a
/// compiler picks within a lane.
fn depth(len: usize) -> f64 {
    22.0 + len.div_ceil(4096) as f64
}

/// `b` with an allowance for rounding the result it bounds.
fn second(b: f64) -> f64 {
    b * (1.0 + 16.0 * U)
}

/// The loss buffer: the loss at word 1 between two sentinels.
fn loss_buffer(kit: &mut Kit) -> PooledBuffer {
    kit.upload(&[SENTINEL, SENTINEL, SENTINEL])
}

/// The loss at word 1 of `out`, after checking the sentinels around it.
fn read_loss(kit: &mut Kit, out: &PooledBuffer) -> f32 {
    let got = kit.read(out);
    assert_eq!(got[0], SENTINEL, "the word before the loss changed");
    assert_eq!(got[2], SENTINEL, "the word after the loss changed");
    got[1]
}

/// The GPU's cross-entropy of `logits` (`batch × classes`) against `labels`: the loss, the
/// gradient, and the finite flag of the loss alone.
fn gpu_ce(
    kit: &mut Kit,
    (ce, flag): (&SoftmaxCrossEntropy, &FiniteFlag),
    (batch, classes): (u32, u32),
    logits: &[f32],
    labels: &[u32],
) -> (f32, Vec<f32>, u32) {
    let z = kit.upload(logits);
    let l = kit.upload(
        &labels
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>(),
    );
    let grad = kit.upload(&vec![SENTINEL; logits.len()]);
    let out = loss_buffer(kit);
    let flags = kit.zeros(1);
    let call = ce.call(
        &mut kit.steps,
        &mut kit.pool,
        (batch, classes),
        &z,
        &l,
        &grad,
        (&out, 1),
    );
    let check = flag.call(&mut kit.steps, &mut kit.pool, (&out, 1), &[], &flags);
    let mut step = kit.steps.begin();
    ce.record(&mut step, &call);
    flag.record(&mut step, &check);
    let _ = step.finish();
    let loss = read_loss(kit, &out);
    let g = kit.read(&grad);
    let f = kit.read(&flags)[0].to_bits();
    call.release(&mut kit.pool);
    check.release(&mut kit.pool);
    (loss, g, f)
}

/// What `Tape::softmax_cross_entropy` with `Reduction::Mean` gives on an `f64` store from the
/// same `f32` logits: the loss and the logits' gradient.
fn tape_ce(batch: usize, classes: usize, logits: &[f32], labels: &[u32]) -> (f64, Vec<f64>) {
    let mut store = ParamStore::<f64>::new_generic();
    let wide: Vec<f64> = logits.iter().map(|&v| f64::from(v)).collect();
    let id = store.add(Tensor::from_vec(wide, &[batch, classes]).unwrap());
    let labels: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
    let mut graph = Graph::<f64>::new_generic();
    let mut tape = graph.tape(&store.values);
    let z = tape.param(id);
    let loss = tape.softmax_cross_entropy(z, &labels, Reduction::Mean);
    let value = tape.value(loss).data()[0];
    tape.backward(loss, &mut store.grads);
    drop(tape);
    (value, store.grad(id).data().to_vec())
}

/// The stated bounds of the cross-entropy on these logits, from the exact row values: the
/// loss's, and each gradient value's.
fn ce_bounds(batch: usize, classes: usize, logits: &[f32], labels: &[u32]) -> (f64, Vec<f64>) {
    let b = batch as f64;
    let mut grad = Vec::with_capacity(logits.len());
    let (mut row_bounds, mut losses) = (0.0, 0.0);
    for (i, row) in logits.chunks_exact(classes).enumerate() {
        let z: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let x: Vec<f64> = z.iter().map(|v| v - m).collect();
        let k = x.iter().position(|&v| v == 0.0).unwrap();
        let rest = || {
            x.iter()
                .enumerate()
                .filter(move |&(j, _)| j != k)
                .map(|(_, &v)| v)
        };
        let e: f64 = rest().map(f64::exp).sum();
        let d_e: f64 = rest()
            .map(|v| (6.0 + 5.0 * v.abs()) * U * v.exp() + FLOOR)
            .sum::<f64>()
            + sum_bound(classes - 1, e);
        let s = 1.0 + e;
        let lse = e.ln_1p();
        let d = if e <= 1.0 {
            second(d_e + 10.0 * U * lse)
        } else {
            second((d_e + U * s) / s + 6.0 * U * lse)
        };
        let label = labels[i] as usize;
        let loss = lse - x[label];
        losses += loss;
        let order = |zj: f64| 2.0 * U * (zj.abs() + m.abs() + lse.abs());
        row_bounds += order(z[label]) + d;
        for (j, &xj) in x.iter().enumerate() {
            let t = xj - lse;
            let p = t.exp();
            let onehot = if j == label { 1.0 } else { 0.0 };
            let inner = order(z[j]) + d + (6.0 + 4.0 * t.abs()) * U;
            grad.push(second(p * inner / b + 3.0 * U * (p - onehot).abs() / b) + FLOOR / b);
        }
    }
    let mean = losses / b;
    let loss = second((row_bounds + depth(batch) * U * losses) / b + 5.0 * U * mean)
        + 2.0 * batch as f64 * FLOOR;
    (loss, grad)
}

/// `n` labels below `classes`, drawn from `seed`.
fn labels(seed: u64, n: usize, classes: u32) -> Vec<u32> {
    seeded_values(seed, n, 0.0, classes as f32)
        .into_iter()
        .map(|v| (v as u32).min(classes - 1))
        .collect()
}

/// The cross-entropy against the tape on an `f64` store, within its stated bounds, on:
/// - one row of 10 classes;
/// - CIFAR's 128 × 10;
/// - 1,000 rows of 3;
/// - 7 rows of 1,000;
/// - 65 rows of one class (a loss of 0, a gradient of 0);
/// - 70,000 rows of 4 (more than 256 workgroups of rows, and a sum over 18 chunks).
///
/// The logits run over `[−5, 5)` and, for the exponential's wider error, `[−40, 40)`. The
/// flag of a finite loss is clear, and a second run gives the same bits.
#[test]
fn cross_entropy_twin() {
    for_each_gpu(|gpu| {
        for (s, (batch, classes)) in [
            (1u32, 10u32),
            (128, 10),
            (1000, 3),
            (7, 1000),
            (65, 1),
            (70_000, 4),
        ]
        .into_iter()
        .enumerate()
        {
            let n = (batch * classes) as usize;
            let mut kit = kit(gpu, n);
            let ce = SoftmaxCrossEntropy::new(gpu, &mut kit.cache).unwrap();
            let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
            for range in [5.0f32, 40.0] {
                let what = format!("{batch} × {classes}, logits within ±{range}");
                let z = seeded_values(10 * s as u64 + 1, n, -range, range);
                let y = labels(10 * s as u64 + 2, batch as usize, classes);
                let (loss, grad, f) = gpu_ce(&mut kit, (&ce, &flag), (batch, classes), &z, &y);
                let (want_loss, want_grad) = tape_ce(batch as usize, classes as usize, &z, &y);
                let (loss_bound, grad_bound) = ce_bounds(batch as usize, classes as usize, &z, &y);
                let worst = assert_within(
                    &[loss],
                    &[want_loss],
                    |_| loss_bound,
                    &format!("loss, {what}"),
                );
                println!(
                    "cross-entropy loss {what}: worst {:.3} of its bound",
                    worst.ratio()
                );
                let worst = assert_within(
                    &grad,
                    &want_grad,
                    |i| grad_bound[i],
                    &format!("gradient, {what}"),
                );
                println!(
                    "cross-entropy gradient {what}: worst {:.3} of its bound",
                    worst.ratio()
                );
                assert_eq!(f, 0, "the finite flag of a finite loss, {what}");
                let (again, again_grad, _) =
                    gpu_ce(&mut kit, (&ce, &flag), (batch, classes), &z, &y);
                assert_exact(
                    &[again],
                    &[loss],
                    &format!("the loss on a second run, {what}"),
                );
                assert_exact(
                    &again_grad,
                    &grad,
                    &format!("the gradient on a second run, {what}"),
                );
            }
        }
    });
}

/// Rows `[0, z]` labelled 1 for `z` from −15 to −7: the label's probability is far below 1/2,
/// so `p − 1` rounds, and its gradient takes three roundings on `p − onehot`. Each gradient is
/// within its bound of the tape's.
#[test]
fn cross_entropy_label_far_below_the_max() {
    for_each_gpu(|gpu| {
        let batch = 1000u32;
        let mut kit = kit(gpu, 2000);
        let ce = SoftmaxCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let z: Vec<f32> = (0..batch)
            .flat_map(|i| [0.0, -15.0 + 8.0 * i as f32 / batch as f32])
            .collect();
        let y = vec![1u32; batch as usize];
        let (loss, grad, _) = gpu_ce(&mut kit, (&ce, &flag), (batch, 2), &z, &y);
        let (want_loss, want_grad) = tape_ce(batch as usize, 2, &z, &y);
        let (loss_bound, grad_bound) = ce_bounds(batch as usize, 2, &z, &y);
        assert_within(&[loss], &[want_loss], |_| loss_bound, "the loss");
        let worst = assert_within(&grad, &want_grad, |i| grad_bound[i], "the gradient");
        println!(
            "labels far below the max: worst {:.3} of its bound",
            worst.ratio()
        );
    });
}

/// A row with a NaN, +∞ or −∞ logit, or a label not below the classes, gives a NaN loss and a
/// NaN gradient row, so the loss is NaN and the finite flag set. The other rows' gradients
/// are the bits of a run without the bad row.
#[test]
fn cross_entropy_bad_rows_set_the_flag() {
    for_each_gpu(|gpu| {
        let (batch, classes) = (4u32, 5u32);
        let mut kit = kit(gpu, 20);
        let ce = SoftmaxCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let z = seeded_values(1, 20, -3.0, 3.0);
        let y = vec![0u32, 4, 2, 1];
        let (_, clean, f) = gpu_ce(&mut kit, (&ce, &flag), (batch, classes), &z, &y);
        assert_eq!(f, 0);
        for (name, at, value, label) in [
            ("a NaN logit", 7, f32::NAN, 4),
            ("a +inf logit", 5, f32::INFINITY, 4),
            ("a -inf logit off the label", 8, f32::NEG_INFINITY, 4),
            ("a -inf logit on the label", 9, f32::NEG_INFINITY, 4),
            ("label 5 of 5 classes", 6, z[6], 5),
        ] {
            let (mut zb, mut yb) = (z.clone(), y.clone());
            zb[at] = value;
            yb[1] = label;
            let (loss, grad, f) = gpu_ce(&mut kit, (&ce, &flag), (batch, classes), &zb, &yb);
            assert!(loss.is_nan(), "{name}: the loss is {loss}, not NaN");
            assert_eq!(f, 1, "{name}: the finite flag");
            assert!(
                grad[5..10].iter().all(|g| g.is_nan()),
                "{name}: the bad row's gradient"
            );
            let rest: Vec<f32> = grad[..5].iter().chain(&grad[10..]).copied().collect();
            let want: Vec<f32> = clean[..5].iter().chain(&clean[10..]).copied().collect();
            assert_exact(&rest, &want, &format!("{name}: the other rows"));
        }
    });
}

/// The GPU's mean squared error: the loss and the gradient.
fn gpu_mse(kit: &mut Kit, mse: &MeanSquaredError, pred: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let (p, t) = (kit.upload(pred), kit.upload(target));
    let grad = kit.upload(&vec![SENTINEL; pred.len()]);
    let out = loss_buffer(kit);
    let call = mse.call(
        &mut kit.steps,
        &mut kit.pool,
        pred.len() as u32,
        &p,
        &t,
        &grad,
        (&out, 1),
    );
    let mut step = kit.steps.begin();
    mse.record(&mut step, &call);
    let _ = step.finish();
    let loss = read_loss(kit, &out);
    let g = kit.read(&grad);
    call.release(&mut kit.pool);
    (loss, g)
}

/// The mean squared error against the tape: the gradient bit for bit against an `f32`
/// store's, and the loss within `(25 + chunks)·u·Σ d² / n + 5u·loss` of an `f64` store's, for
/// 1, 37, 4,097, 100,000 and 2²⁴ + 1,000 values (more than one grid row of workgroups).
#[test]
fn mse_twin() {
    for_each_gpu(|gpu| {
        for (s, n) in [1usize, 37, 4097, 100_000, (1 << 24) + 1000]
            .into_iter()
            .enumerate()
        {
            let mut kit = kit(gpu, n);
            let mse = MeanSquaredError::new(gpu, &mut kit.cache).unwrap();
            let pred = seeded_values(20 + s as u64, n, -2.0, 2.0);
            let target = seeded_values(30 + s as u64, n, -2.0, 2.0);
            let (loss, grad) = gpu_mse(&mut kit, &mse, &pred, &target);

            let mut store = ParamStore::new();
            let id = store.add(Tensor::from_vec(pred.clone(), &[n]).unwrap());
            let mut graph = Graph::new();
            let mut tape = graph.tape(&store.values);
            let p = tape.param(id);
            let t = tape.input(Tensor::from_vec(target.clone(), &[n]).unwrap().view());
            let l = tape.mse(p, t, Reduction::Mean);
            tape.backward(l, &mut store.grads);
            drop(tape);
            assert_exact(
                &grad,
                store.grad(id).data(),
                &format!("the gradient of {n} values"),
            );

            let squares: f64 = pred
                .iter()
                .zip(&target)
                .map(|(&p, &t)| (f64::from(p) - f64::from(t)).powi(2))
                .sum();
            let want = squares / n as f64;
            let relative = (depth(n) + 3.0) * U * squares / n as f64;
            let bound = second(relative + 5.0 * U * want) + 3.0 * FLOOR;
            let worst = assert_within(
                &[loss],
                &[want],
                |_| bound,
                &format!("the loss of {n} values"),
            );
            println!(
                "mse loss of {n} values: worst {:.3} of its bound",
                worst.ratio()
            );
        }
    });
}

/// A difference whose square overflows gives an infinite loss and a finite gradient, and a NaN
/// prediction a NaN loss and a NaN gradient at that value; the finite flag of the loss
/// catches both.
#[test]
fn mse_non_finite_reaches_the_loss() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 8192);
        let mse = MeanSquaredError::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let target = vec![0.0f32; 8192];
        for (name, value) in [
            ("1e20", 1e20f32),
            ("NaN", f32::NAN),
            ("+inf", f32::INFINITY),
        ] {
            let mut pred = seeded_values(1, 8192, -1.0, 1.0);
            pred[5000] = value;
            let (loss, grad) = gpu_mse(&mut kit, &mse, &pred, &target);
            assert!(!loss.is_finite(), "{name}: the loss is {loss}");
            assert_eq!(
                grad[5000].is_finite(),
                value == 1e20,
                "{name}: its gradient {}",
                grad[5000]
            );
            let losses = kit.upload(&[loss]);
            let flags = kit.zeros(1);
            let check = flag.call(&mut kit.steps, &mut kit.pool, (&losses, 0), &[], &flags);
            let mut step = kit.steps.begin();
            flag.record(&mut step, &check);
            let _ = step.finish();
            assert_eq!(kit.read(&flags)[0].to_bits(), 1, "{name}: the finite flag");
            check.release(&mut kit.pool);
        }
    });
}

/// The documented refusals: empty shapes, buffers too short, and an output that is a buffer
/// the kernels read.
#[test]
fn refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu, 64);
        let ce = SoftmaxCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        let mse = MeanSquaredError::new(gpu, &mut kit.cache).unwrap();
        let [z, l, g, out] = [(); 4].map(|_| kit.zeros(16));
        let one = kit.zeros(1);
        let mut cross = |shape: (u32, u32),
                         z: &PooledBuffer,
                         l: &PooledBuffer,
                         g: &PooledBuffer,
                         out: (&PooledBuffer, u32)| {
            message(&mut || {
                let _ = ce.call(&mut kit.steps, &mut kit.pool, shape, z, l, g, out);
            })
        };
        assert!(cross((0, 4), &z, &l, &g, (&out, 0)).contains("must be above 0"));
        assert!(cross((4, 0), &z, &l, &g, (&out, 0)).contains("must be above 0"));
        assert!(
            cross((1 << 20, 1 << 10), &z, &l, &g, (&out, 0)).contains("more than the kernels take")
        );
        assert!(cross((4, 5), &z, &l, &g, (&out, 0)).contains("the logits needs 20 values"));
        assert!(cross((4, 4), &z, &one, &g, (&out, 0)).contains("the labels needs 4 values"));
        assert!(cross((4, 4), &z, &l, &g, (&out, 16)).contains("the loss output needs 17 values"));
        assert!(
            cross((4, 4), &z, &l, &z, (&out, 0))
                .contains("the gradient and the logits must be different")
        );
        assert!(
            cross((4, 4), &z, &l, &g, (&g, 0))
                .contains("the gradient and the loss output must be different")
        );
        assert!(
            cross((4, 4), &z, &l, &g, (&z, 0))
                .contains("the loss output and the logits must be different")
        );
        let mut squared = |n: u32,
                           p: &PooledBuffer,
                           t: &PooledBuffer,
                           g: &PooledBuffer,
                           out: (&PooledBuffer, u32)| {
            message(&mut || {
                let _ = mse.call(&mut kit.steps, &mut kit.pool, n, p, t, g, out);
            })
        };
        assert!(squared(0, &z, &l, &g, (&out, 0)).contains("where the kernels take 1 to"));
        assert!(squared(17, &z, &l, &g, (&out, 0)).contains("the prediction needs 17 values"));
        assert!(squared(16, &z, &one, &g, (&out, 0)).contains("the target needs 16 values"));
        assert!(
            squared(16, &z, &l, &l, (&out, 0))
                .contains("the gradient and the target must be different")
        );
        assert!(
            squared(16, &z, &l, &g, (&z, 0))
                .contains("the loss output and the prediction must be different")
        );
    });
}

/// Confident rows, as late in training: 4,096 rows of 10 logits in `[−4, 4)` with the label's
/// raised by 7 or by 12 (mean losses near 0.24 and 0.0026), 20 batches each. Each batch's
/// loss is within its bound of the `f64` tape's, and the mean of their relative errors is
/// within 8u: rounding, not a bias. Computing `ln S` with WGSL's `log` gave a mean of −26.5u
/// at a raise of 12 on an Intel UHD Graphics (JSL) through Vulkan, from its 2⁻²¹ absolute
/// error near 1.
#[test]
fn cross_entropy_confident_rows_have_no_bias() {
    for_each_gpu(|gpu| {
        let (batch, classes) = (4096u32, 10u32);
        let n = (batch * classes) as usize;
        let mut kit = kit(gpu, n);
        let ce = SoftmaxCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        for raise in [7.0f32, 12.0] {
            let mut sum = 0.0;
            for seed in 0..20u64 {
                let mut z = seeded_values(100 + seed, n, -4.0, 4.0);
                let y = labels(200 + seed, batch as usize, classes);
                for (i, &l) in y.iter().enumerate() {
                    z[i * classes as usize + l as usize] += raise;
                }
                let (loss, _, _) = gpu_ce(&mut kit, (&ce, &flag), (batch, classes), &z, &y);
                let (want, _) = tape_ce(batch as usize, classes as usize, &z, &y);
                let (bound, _) = ce_bounds(batch as usize, classes as usize, &z, &y);
                assert_within(&[loss], &[want], |_| bound, &format!("loss, raise {raise}"));
                sum += (f64::from(loss) - want) / want;
            }
            let mean = sum / 20.0 / U;
            println!("confident rows, raise {raise}: mean relative error {mean:.2}u");
            assert!(
                mean.abs() <= 8.0,
                "confident rows, raise {raise}: the loss's mean relative error is {mean:.2}u"
            );
        }
    });
}

/// The GPU's negative log-likelihood of `log_probs` (`batch × classes`) against `labels`:
/// the loss and the gradient.
fn gpu_nll(
    kit: &mut Kit,
    nll: &NegativeLogLikelihood,
    (batch, classes): (u32, u32),
    log_probs: &[f32],
    labels: &[u32],
) -> (f32, Vec<f32>) {
    let lp = kit.upload(log_probs);
    let l = kit.upload(
        &labels
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>(),
    );
    let grad = kit.upload(&vec![SENTINEL; log_probs.len()]);
    let out = loss_buffer(kit);
    let call = nll.call(
        &mut kit.steps,
        &mut kit.pool,
        (batch, classes),
        &lp,
        &l,
        &grad,
        (&out, 1),
    );
    let mut step = kit.steps.begin();
    nll.record(&mut step, &call);
    let _ = step.finish();
    let loss = read_loss(kit, &out);
    let g = kit.read(&grad);
    call.release(&mut kit.pool);
    (loss, g)
}

/// What `Tape::nll` with `Reduction::Mean` gives on a store of `T` from the same `f32`
/// log-probabilities: the loss and the gradient.
fn tape_nll<T: Real>(classes: usize, log_probs: &[f32], labels: &[u32]) -> (T, Vec<T>) {
    let batch = labels.len();
    let mut store = ParamStore::<T>::new_generic();
    let wide: Vec<T> = log_probs
        .iter()
        .map(|&v| T::from_f64(f64::from(v)))
        .collect();
    let id = store.add(Tensor::from_vec(wide, &[batch, classes]).unwrap());
    let labels: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&store.values);
    let lp = tape.param(id);
    let loss = tape.nll(lp, &labels, Reduction::Mean);
    let value = tape.value(loss).data()[0];
    tape.backward(loss, &mut store.grads);
    drop(tape);
    (value, store.grad(id).data().to_vec())
}

/// The negative log-likelihood against the tape: the gradient bit for bit against an `f32`
/// store's (a row's label gets `−(1/B)` rounded to `f32`, every other value `+0`), and the
/// loss within `(22 + chunks)·u·Σ|ℓ| / B + 5u·loss` of an `f64` store's (each row's loss is
/// its label's value negated, exact), on the cross-entropy's shapes with log-probabilities
/// in `[−12, 0)`. A second run gives the same bits.
#[test]
fn nll_twin() {
    for_each_gpu(|gpu| {
        for (s, (batch, classes)) in [
            (1u32, 10u32),
            (128, 10),
            (1000, 3),
            (7, 1000),
            (65, 1),
            (70_000, 4),
        ]
        .into_iter()
        .enumerate()
        {
            let n = (batch * classes) as usize;
            let mut kit = kit(gpu, n);
            let nll = NegativeLogLikelihood::new(gpu, &mut kit.cache).unwrap();
            let what = format!("{batch} × {classes}");
            let lp = seeded_values(40 + s as u64, n, -12.0, 0.0);
            let y = labels(50 + s as u64, batch as usize, classes);
            let (loss, grad) = gpu_nll(&mut kit, &nll, (batch, classes), &lp, &y);
            let (_, want_grad) = tape_nll::<f32>(classes as usize, &lp, &y);
            assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
            let (want, _) = tape_nll::<f64>(classes as usize, &lp, &y);
            let total: f64 = (y.iter().enumerate())
                .map(|(i, &l)| f64::from(lp[i * classes as usize + l as usize]).abs())
                .sum();
            let b = f64::from(batch);
            let bound = second(depth(batch as usize) * U * total / b + 5.0 * U * want.abs())
                + 2.0 * b * FLOOR;
            let worst = assert_within(&[loss], &[want], |_| bound, &format!("the loss, {what}"));
            println!("nll loss {what}: worst {:.3} of its bound", worst.ratio());
            let (again, again_grad) = gpu_nll(&mut kit, &nll, (batch, classes), &lp, &y);
            assert_exact(&[again], &[loss], &format!("a second run's loss, {what}"));
            assert_exact(
                &again_grad,
                &grad,
                &format!("a second run's gradient, {what}"),
            );
        }
    });
}

/// NaN and ±∞ as on the CPU: a NaN anywhere in a row gives a NaN loss and the CPU's finite
/// gradient; −∞ at the label gives a +∞ loss; +∞ and −∞ off the label leave the loss finite
/// (the CPU's, bit for bit, here where every partial sum is exact). A label not below the
/// classes, which the CPU panics on, gives a NaN loss and a NaN gradient row, the other rows'
/// gradients unchanged.
#[test]
fn nll_non_finite_rows() {
    for_each_gpu(|gpu| {
        let (batch, classes) = (4u32, 5u32);
        let mut kit = kit(gpu, 20);
        let nll = NegativeLogLikelihood::new(gpu, &mut kit.cache).unwrap();
        // Quarters, so the CPU's sum and the GPU's tree are exact.
        let lp: Vec<f32> = (0..20).map(|i| -0.25 * (i % 7) as f32).collect();
        let y = vec![0u32, 4, 2, 1];
        let (clean, clean_grad) = gpu_nll(&mut kit, &nll, (batch, classes), &lp, &y);
        for (name, at, value) in [
            ("a NaN off the label", 7, f32::NAN),
            ("a NaN at the label", 9, f32::NAN),
            ("-inf at the label", 9, f32::NEG_INFINITY),
            ("+inf off the label", 5, f32::INFINITY),
            ("-inf off the label", 8, f32::NEG_INFINITY),
        ] {
            let mut bad = lp.clone();
            bad[at] = value;
            let (loss, grad) = gpu_nll(&mut kit, &nll, (batch, classes), &bad, &y);
            let (want, want_grad) = tape_nll::<f32>(classes as usize, &bad, &y);
            assert_exact(&[loss], &[want], &format!("{name}: the loss"));
            assert_exact(&grad, &want_grad, &format!("{name}: the gradient"));
        }
        assert_exact(&[clean], &[tape_nll::<f32>(5, &lp, &y).0], "the clean loss");
        let mut bad_label = y.clone();
        bad_label[1] = 5;
        let (loss, grad) = gpu_nll(&mut kit, &nll, (batch, classes), &lp, &bad_label);
        assert!(loss.is_nan(), "label 5 of 5 classes: the loss is {loss}");
        assert!(
            grad[5..10].iter().all(|g| g.is_nan()),
            "the bad row's gradient"
        );
        let rest: Vec<f32> = grad[..5].iter().chain(&grad[10..]).copied().collect();
        let want: Vec<f32> = (clean_grad[..5].iter().chain(&clean_grad[10..]))
            .copied()
            .collect();
        assert_exact(&rest, &want, "label 5 of 5 classes: the other rows");
    });
}

/// Each regression loss the tests run, with its term error `c` in units of `u`: how far a
/// value's computed loss may be from the exact loss of the same `f32` values.
///
/// - **L1:** `|fl(p − t)|`, one rounding: `c = 1`.
/// - **Huber:** the quadratic branch `fl(fl(0.5·d)·d)` with `d` rounded once: `(1 + u)³`,
///   `c = 3`. The linear branch `fl(delta·fl(a − 0.5·delta))` with `a = |d|` rounded once:
///   `a`'s error `u·a` is at most `2u·(a − 0.5·delta)` since `a ≥ delta`, then one rounding
///   for the difference and one for the product: `c = 4`.
/// - **Smooth L1:** the quadratic branch's square `(1 + u)³` and a division WGSL gives within
///   2.5 ulp, at most `5u` relative: `c = 8`. The linear branch as Huber's without the product:
///   `c = 3`.
///
/// Near the threshold the `f32` difference can take the other branch than the exact one;
/// both branches and their slopes meet there, so that costs `O(u²)`, inside `second`.
const REGRESSIONS: [(Regression, f64); 7] = [
    (Regression::L1, 1.0),
    (Regression::Huber { delta: 1.0 }, 4.0),
    (Regression::Huber { delta: 0.25 }, 4.0),
    (Regression::SmoothL1 { beta: 1.0 }, 8.0),
    (Regression::SmoothL1 { beta: 0.75 }, 8.0),
    (Regression::SmoothL1 { beta: 0.0 }, 8.0),
    (
        Regression::Huber {
            delta: f32::INFINITY,
        },
        4.0,
    ),
];

/// The GPU's regression loss `kind`: the loss and the gradient.
fn gpu_regression(
    kit: &mut Kit,
    loss: &RegressionLoss,
    kind: Regression,
    pred: &[f32],
    target: &[f32],
) -> (f32, Vec<f32>) {
    let (p, t) = (kit.upload(pred), kit.upload(target));
    let grad = kit.upload(&vec![SENTINEL; pred.len()]);
    let out = loss_buffer(kit);
    let call = loss.call(
        &mut kit.steps,
        &mut kit.pool,
        kind,
        pred.len() as u32,
        &p,
        &t,
        &grad,
        (&out, 1),
    );
    let mut step = kit.steps.begin();
    loss.record(&mut step, &call);
    let _ = step.finish();
    let value = read_loss(kit, &out);
    let g = kit.read(&grad);
    call.release(&mut kit.pool);
    (value, g)
}

/// What the tape gives for `kind` with `Reduction::Mean` on a store of `T` from the same
/// `f32` values: the loss and the prediction's gradient.
fn tape_regression<T: Real>(kind: Regression, pred: &[f32], target: &[f32]) -> (T, Vec<T>) {
    let n = pred.len();
    let wide = |v: &[f32]| {
        v.iter()
            .map(|&x| T::from_f64(f64::from(x)))
            .collect::<Vec<T>>()
    };
    let mut store = ParamStore::<T>::new_generic();
    let id = store.add(Tensor::from_vec(wide(pred), &[n]).unwrap());
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&store.values);
    let p = tape.param(id);
    let t = tape.input(Tensor::from_vec(wide(target), &[n]).unwrap().view());
    let l = match kind {
        Regression::L1 => tape.l1(p, t, Reduction::Mean),
        Regression::Huber { delta } => {
            tape.huber(p, t, T::from_f64(f64::from(delta)), Reduction::Mean)
        }
        Regression::SmoothL1 { beta } => {
            tape.smooth_l1(p, t, T::from_f64(f64::from(beta)), Reduction::Mean)
        }
    };
    let value = tape.value(l).data()[0];
    tape.backward(l, &mut store.grads);
    drop(tape);
    (value, store.grad(id).data().to_vec())
}

/// Whether value `i` takes smooth L1's quadratic branch, whose divisions WGSL rounds within
/// 2.5 ulp rather than correctly.
fn divides(kind: Regression, p: f32, t: f32) -> bool {
    matches!(kind, Regression::SmoothL1 { beta } if (p - t).abs() < beta)
}

/// The gradient against the `f32` tape's: bit for bit, except on smooth L1's quadratic
/// branch, where `fl(fl(d / beta)·w)` with the division within 2.5 ulp (`5u`) is within
/// `8u` of the CPU's `fl(fl(d / beta)·w)` relative to it: `5u + u` for the GPU's two
/// roundings and `2u` for the CPU's. Returns how many of those were bit for bit, of how
/// many, and the worst as a share of its bound.
fn check_regression_grad(
    kind: Regression,
    (pred, target): (&[f32], &[f32]),
    grad: &[f32],
    want: &[f32],
    what: &str,
) -> (usize, usize, f64) {
    let split = |v: &[f32], quadratic: bool| -> Vec<f32> {
        (0..v.len())
            .filter(|&i| divides(kind, pred[i], target[i]) == quadratic)
            .map(|i| v[i])
            .collect()
    };
    assert_exact(&split(grad, false), &split(want, false), what);
    let (got, wanted) = (split(grad, true), split(want, true));
    let reference: Vec<f64> = wanted.iter().map(|&v| f64::from(v)).collect();
    let worst = assert_within(
        &got,
        &reference,
        |i| second(8.0 * U * reference[i].abs()),
        what,
    );
    let same = got
        .iter()
        .zip(&wanted)
        .filter(|(a, b)| a.to_bits() == b.to_bits());
    (same.count(), got.len(), worst.ratio())
}

/// Each loss of `REGRESSIONS` against the tape: the gradient against the `f32` store's (bit
/// for bit, but for smooth L1's quadratic branch within `8u`), and the loss within
/// `(22 + chunks + c)·u·Σ ℓ / n + 5u·loss` of an `f64` store's, for 1, 37, 4,097 and 100,000
/// values with predictions and targets in `[−2, 2)` (so every branch is taken), and for
/// 2²⁴ + 1,000 values (more than one grid row of workgroups) with L1 and smooth L1. A second
/// run gives the same bits.
#[test]
fn regression_twin() {
    for_each_gpu(|gpu| {
        for (k, (kind, c)) in REGRESSIONS.into_iter().enumerate() {
            let large = k == 0 || k == 3;
            let sizes: &[usize] = if large {
                &[1, 37, 4097, 100_000, (1 << 24) + 1000]
            } else {
                &[1, 37, 4097, 100_000]
            };
            for (s, &n) in sizes.iter().enumerate() {
                let mut kit = kit(gpu, n);
                let loss = RegressionLoss::new(gpu, &mut kit.cache).unwrap();
                let what = format!("{kind:?} of {n} values");
                let seed = 100 * k as u64 + s as u64;
                let pred = seeded_values(seed + 1000, n, -2.0, 2.0);
                let target = seeded_values(seed + 2000, n, -2.0, 2.0);
                let (value, grad) = gpu_regression(&mut kit, &loss, kind, &pred, &target);
                let (_, want_grad) = tape_regression::<f32>(kind, &pred, &target);
                let (same, quadratic, ratio) = check_regression_grad(
                    kind,
                    (&pred, &target),
                    &grad,
                    &want_grad,
                    &format!("the gradient, {what}"),
                );
                let (want, _) = tape_regression::<f64>(kind, &pred, &target);
                // Every term is not negative, so Σ ℓ is the mean times n.
                let total = want * n as f64;
                let bound =
                    second((depth(n) + c) * U * total / n as f64 + 5.0 * U * want) + 3.0 * FLOOR;
                let worst =
                    assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
                println!(
                    "{what}: loss worst {:.3} of its bound; {same} of {quadratic} quadratic \
                     smooth L1 gradients bit for bit, the worst {ratio:.3} of its bound",
                    worst.ratio()
                );
                let (again, again_grad) = gpu_regression(&mut kit, &loss, kind, &pred, &target);
                assert_exact(&[again], &[value], &format!("a second run's loss, {what}"));
                assert_exact(
                    &again_grad,
                    &grad,
                    &format!("a second run's gradient, {what}"),
                );
            }
        }
    });
}

/// Each value's loss and gradient on its own (`n = 1`, so the loss is the value's term, bit
/// for bit through the tree and the division by 1), against the `f32` tape, at the edges:
/// `d` of ±0 (`sign(0)` gives a gradient of +0); `|d|` exactly at the threshold, which takes
/// the linear branch, and one ulp either side; smooth L1 with `beta` 0 (the L1 loss) and +∞
/// (a loss of 0); Huber with `delta` +∞ (half the square); NaN in either input, two
/// infinities of one sign and of opposite signs, one infinite input, and a finite difference
/// that overflows. Every value is the tape's bits, NaN where it is NaN, but for smooth L1's
/// quadratic branch, within `8u` (its losses here are exact quotients, so bit for bit too).
#[test]
fn regression_edges_match_the_tape() {
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let above = |x: f32| f32::from_bits(x.to_bits() + 1);
    let inf = f32::INFINITY;
    let pairs: Vec<(f32, f32)> = vec![
        (1.5, 1.5),
        (-0.0, 0.0),
        (0.0, -0.0),
        (0.5, 0.0),
        (-0.25, 0.25),
        (below(0.5), 0.0),
        (above(0.5), 0.0),
        (0.0, below(0.5)),
        (0.25, 0.0),
        (-3.0, 1.0),
        (1e-3, 0.0),
        (f32::NAN, 0.0),
        (0.0, f32::NAN),
        (inf, inf),
        (-inf, -inf),
        (inf, -inf),
        (-inf, inf),
        (inf, 1.0),
        (1.0, inf),
        (-inf, 0.0),
        (3e38, -3e38),
        (-3e38, 3e38),
        (1e20, 0.0),
    ];
    let kinds = [
        Regression::L1,
        Regression::Huber { delta: 0.5 },
        Regression::Huber { delta: inf },
        Regression::SmoothL1 { beta: 0.5 },
        Regression::SmoothL1 { beta: 0.0 },
        Regression::SmoothL1 { beta: inf },
    ];
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 64);
        let loss = RegressionLoss::new(gpu, &mut kit.cache).unwrap();
        for kind in kinds {
            for &(p, t) in &pairs {
                let what = format!("{kind:?} at p = {p:e}, t = {t:e}");
                let (value, grad) = gpu_regression(&mut kit, &loss, kind, &[p], &[t]);
                let (want, want_grad) = tape_regression::<f32>(kind, &[p], &[t]);
                if divides(kind, p, t) {
                    let r = f64::from(want);
                    assert_within(&[value], &[r], |_| second(8.0 * U * r.abs()), &what);
                } else {
                    assert_exact(&[value], &[want], &format!("the loss, {what}"));
                }
                check_regression_grad(
                    kind,
                    (&[p], &[t]),
                    &grad,
                    &want_grad,
                    &format!("the gradient, {what}"),
                );
            }
        }
    });
}

/// The documented refusals of the negative log-likelihood and the regression losses: the
/// CPU tape's thresholds (Huber's `delta` not above 0 or NaN, smooth L1's `beta` below 0 or
/// NaN, with the tape's message), empty shapes, buffers too short, and an output that is a
/// buffer the kernels read.
#[test]
fn nll_and_regression_refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu, 64);
        let nll = NegativeLogLikelihood::new(gpu, &mut kit.cache).unwrap();
        let loss = RegressionLoss::new(gpu, &mut kit.cache).unwrap();
        let [z, l, g, out] = [(); 4].map(|_| kit.zeros(16));
        let one = kit.zeros(1);
        let mut rows = |shape: (u32, u32),
                        z: &PooledBuffer,
                        l: &PooledBuffer,
                        g: &PooledBuffer,
                        out: (&PooledBuffer, u32)| {
            message(&mut || {
                let _ = nll.call(&mut kit.steps, &mut kit.pool, shape, z, l, g, out);
            })
        };
        assert!(rows((0, 4), &z, &l, &g, (&out, 0)).contains("must be above 0"));
        assert!(rows((4, 0), &z, &l, &g, (&out, 0)).contains("must be above 0"));
        assert!(
            rows((1 << 20, 1 << 10), &z, &l, &g, (&out, 0)).contains("more than the kernels take")
        );
        assert!(
            rows((4, 5), &z, &l, &g, (&out, 0)).contains("the log-probabilities needs 20 values")
        );
        assert!(rows((4, 4), &z, &one, &g, (&out, 0)).contains("the labels needs 4 values"));
        assert!(rows((4, 4), &z, &l, &g, (&out, 16)).contains("the loss output needs 17 values"));
        assert!(
            rows((4, 4), &z, &l, &z, (&out, 0))
                .contains("the gradient and the log-probabilities must be different")
        );
        assert!(
            rows((4, 4), &z, &l, &g, (&l, 0))
                .contains("the loss output and the labels must be different")
        );
        let mut regression = |kind: Regression,
                              n: u32,
                              p: &PooledBuffer,
                              g: &PooledBuffer,
                              out: (&PooledBuffer, u32)| {
            message(&mut || {
                let _ = loss.call(&mut kit.steps, &mut kit.pool, kind, n, p, &l, g, out);
            })
        };
        for (kind, text) in [
            (
                Regression::Huber { delta: 0.0 },
                "huber: delta must be positive, but it is 0.0",
            ),
            (
                Regression::Huber { delta: -1.0 },
                "huber: delta must be positive, but it is -1.0",
            ),
            (
                Regression::Huber { delta: f32::NAN },
                "huber: delta must be positive, but it is NaN",
            ),
            (
                Regression::SmoothL1 { beta: -0.5 },
                "smooth_l1: beta must not be negative, but it is -0.5",
            ),
            (
                Regression::SmoothL1 { beta: f32::NAN },
                "smooth_l1: beta must not be negative, but it is NaN",
            ),
        ] {
            let got = regression(kind, 16, &z, &g, (&out, 0));
            assert!(got.contains(text), "{kind:?}: {got}");
        }
        let l1 = Regression::L1;
        assert!(regression(l1, 0, &z, &g, (&out, 0)).contains("where the kernels take 1 to"));
        assert!(regression(l1, 17, &z, &g, (&out, 0)).contains("the prediction needs 17 values"));
        assert!(regression(l1, 16, &z, &one, (&out, 0)).contains("the gradient needs 16 values"));
        assert!(
            regression(l1, 16, &z, &z, (&out, 0))
                .contains("the gradient and the prediction must be different")
        );
        assert!(
            regression(l1, 16, &z, &g, (&z, 0))
                .contains("the loss output and the prediction must be different")
        );
    });
}

/// A binary cross-entropy's form: what it reads, and how it reduces.
type Form = (BinaryInput, LossReduction);

/// Every form.
const FORMS: [Form; 4] = [
    (BinaryInput::Probabilities, LossReduction::Mean),
    (BinaryInput::Logits, LossReduction::Mean),
    (BinaryInput::Probabilities, LossReduction::Sum),
    (BinaryInput::Logits, LossReduction::Sum),
];

/// The GPU's binary cross-entropy `form`: the loss, the clamp count and the gradient, the
/// loss and the count at words 1 and 2 between two sentinels.
fn gpu_bce(
    kit: &mut Kit,
    bce: &BinaryCrossEntropy,
    form: Form,
    pred: &[f32],
    target: &[f32],
) -> (f32, f32, Vec<f32>) {
    let (p, t) = (kit.upload(pred), kit.upload(target));
    let grad = kit.upload(&vec![SENTINEL; pred.len()]);
    let out = kit.upload(&[SENTINEL; 4]);
    let call = bce.call(
        &mut kit.steps,
        &mut kit.pool,
        form,
        pred.len() as u32,
        &p,
        &t,
        &grad,
        (&out, 1),
    );
    let mut step = kit.steps.begin();
    bce.record(&mut step, &call);
    let _ = step.finish();
    let words = kit.read(&out);
    assert_eq!(words[0], SENTINEL, "the word before the loss changed");
    assert_eq!(words[3], SENTINEL, "the word after the count changed");
    let g = kit.read(&grad);
    call.release(&mut kit.pool);
    (words[1], words[2], g)
}

/// What the tape gives for `form` on a store of `T` from the same `f32` values: the loss,
/// the prediction's gradient and `Tape::clamped_logs`.
fn tape_bce<T: Real>(form: Form, pred: &[f32], target: &[f32]) -> (T, Vec<T>, usize) {
    let n = pred.len();
    let wide = |v: &[f32]| {
        v.iter()
            .map(|&x| T::from_f64(f64::from(x)))
            .collect::<Vec<T>>()
    };
    let mut store = ParamStore::<T>::new_generic();
    let dims = match form.1 {
        LossReduction::BatchMean { batch } => vec![batch as usize, n / batch as usize],
        _ => vec![n],
    };
    let id = store.add(Tensor::from_vec(wide(pred), &dims).unwrap());
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&store.values);
    let p = tape.param(id);
    let t = tape.input(Tensor::from_vec(wide(target), &dims).unwrap().view());
    let reduction = tape_reduction(form.1);
    let l = match form.0 {
        BinaryInput::Probabilities => tape.binary_cross_entropy(p, t, reduction),
        BinaryInput::Logits => tape.binary_cross_entropy_with_logits(p, t, reduction),
    };
    let value = tape.value(l).data()[0];
    let clamped = tape.clamped_logs();
    tape.backward(l, &mut store.grads);
    drop(tape);
    (value, store.grad(id).data().to_vec(), clamped)
}

/// Whether the `f32` tape clamps `ln x`.
fn clamps(x: f32) -> bool {
    x.ln() < -100.0
}

/// One value's exact loss and its stated bound, from the tape's expression on the same `f32`
/// values.
///
/// - **Probabilities:** `−(t·ln p + c·ln q)` with `q = fl(1 − p)` and `c = fl(1 − t)`, the
///   CPU's own subtractions, each logarithm −100 where the tape clamps it. Each logarithm is
///   within `13u` of the exact one: `y = (f − 1)/(f + 1)` has `f − 1` exact (`f` within
///   `[√½, √2]`), `f + 1` rounded once and the division within 2.5 ulp, `6u` together; the
///   series `T` within `1.2u` (its terms past 1 are at most 0.03 of it, its truncation below
///   `10⁻¹²`); the product `2y·T` `u`: `9u` of `r = 2·atanh(y)`. Then `k·ln2_hi` is exact
///   (`|k| ≤ 150`, `ln2_hi` of 15 significant bits), `k·ln2_lo + r` and the last sum round
///   once each, and `|r|` is at most the result for `k ≠ 0` (the result is at least `ln √2`
///   in magnitude, `|r|` at most that): `9u + u + u` and `ln2_lo`'s own error, below
///   `10⁻⁵u`, under `13u`. The two products and the sum add `u` each of
///   `A = |t·ln p| + |c·ln q|`, so the term is within `15u·A`.
/// - **Logits:** `a + L` with `a = max(z, 0) − z·t` and `L = ln(1 + e)`, `e = e^(−|z|)`. The
///   product `z·t`, the difference and the last sum round once each: `u·(|z·t| + |a| + |ℓ|)`.
///   `e` is within `(6 + 4|z|)·u·e` (WGSL's `exp`, `3 + 2|x|` ulp), which moves `L` by at most
///   that, and the series is within `10u·L` of `ln(1 + e)` for the computed `e`, as the
///   cross-entropy's.
///
/// Both add `2⁻¹²⁶·(2 + |t|)` for flushed subnormals (a product with a subnormal target, and
/// an `e` or a `z` below the smallest normal).
fn bce_term(input: BinaryInput, p: f32, t: f32) -> (f64, f64) {
    let t64 = f64::from(t);
    let floor = (2.0 + t64.abs()) * FLOOR;
    match input {
        BinaryInput::Probabilities => {
            let log = |x: f32| {
                if clamps(x) { -100.0 } else { f64::from(x).ln() }
            };
            let a = t64 * log(p);
            let b = f64::from(1.0 - t) * log(1.0 - p);
            (-(a + b), 15.0 * U * (a.abs() + b.abs()) + floor)
        }
        BinaryInput::Logits => {
            let z = f64::from(p);
            let e = (-z.abs()).exp();
            let l = e.ln_1p();
            let a = z.max(0.0) - z * t64;
            let loss = a + l;
            let rounding = U * ((z * t64).abs() + a.abs() + loss.abs());
            (
                loss,
                rounding + 10.0 * U * l + (6.0 + 4.0 * z.abs()) * U * e + floor,
            )
        }
    }
}

/// The exact loss of `form` on these values and its stated bound: the values' bounds summed,
/// `(22 + chunks)·u·Σ|ℓ|` for the fixed tree, and for the mean the division by `n` and `5u`
/// of the mean for it.
fn bce_loss(form: Form, pred: &[f32], target: &[f32]) -> (f64, f64) {
    let (mut sum, mut abs, mut bounds) = (0.0, 0.0, 0.0);
    for (&p, &t) in pred.iter().zip(target) {
        let (v, b) = bce_term(form.0, p, t);
        sum += v;
        abs += v.abs();
        bounds += b;
    }
    let tree = depth(pred.len()) * U * abs;
    match divisor(form.1, pred.len()) {
        Some(d) => {
            let mean = sum / d;
            (
                mean,
                second((bounds + tree) / d + 5.0 * U * mean.abs()) + 3.0 * FLOOR,
            )
        }
        None => (sum, second(bounds + tree) + 3.0 * FLOOR),
    }
}

/// The tape's reduction for `reduction`.
fn tape_reduction(reduction: LossReduction) -> Reduction {
    match reduction {
        LossReduction::Mean => Reduction::Mean,
        LossReduction::Sum => Reduction::Sum,
        LossReduction::BatchMean { .. } => Reduction::BatchMean,
    }
}

/// What `reduction` divides the sum of `n` values by: `n` for the mean, the batch for the
/// batch mean, nothing for the sum.
fn divisor(reduction: LossReduction, n: usize) -> Option<f64> {
    match reduction {
        LossReduction::Mean => Some(n as f64),
        LossReduction::Sum => None,
        LossReduction::BatchMean { batch } => Some(f64::from(batch)),
    }
}

/// The scale of each gradient, the reciprocal of the divisor rounded to `f32` from `f64`
/// (1 for the sum), as the CPU's upstream.
fn bce_scale(form: Form, n: usize) -> f32 {
    divisor(form.1, n).map_or(1.0, |d| (1.0 / d) as f32)
}

/// A logit's exact gradient `g = (σ(z) − t)·w` and its stated bound
/// `w·(Δσ + u·|σ − t|) + u·|g|`. For `z ≥ 0`, `σ = 1/(1 + e)`: `e` within `(6 + 4z)·u·e`
/// moves `1 + e` by at most `6u` of it (`(6 + 4z)·e^(−z)` is largest at 0), the sum rounds
/// once and the division is within 2.5 ulp: `Δσ = 13u·σ`. For `z < 0`, `σ = e/(1 + e)`: `e`'s
/// error twice, the sum and the division, `Δσ = (18 + 8|z|)·u·σ`. For `z = ±∞` it is 1 or 0
/// exactly. The subtraction and the product round once each.
fn logit_grad(z: f32, t: f32, w: f32) -> (f64, f64) {
    let (z, t, w) = (f64::from(z), f64::from(t), f64::from(w));
    let sigma = if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        z.exp() / (1.0 + z.exp())
    };
    let d_sigma = if z.is_infinite() {
        // σ(±∞) is 1 or 0 exactly.
        0.0
    } else if z >= 0.0 {
        13.0 * U * sigma
    } else {
        (18.0 + 8.0 * z.abs()) * U * sigma
    };
    let g = (sigma - t) * w;
    (
        g,
        second(w * (d_sigma + U * (sigma - t).abs()) + U * g.abs()),
    )
}

/// Whether the probability gradient at `(p, t)` divides: not clamped, both finite.
fn bce_divides(input: BinaryInput, p: f32, t: f32) -> bool {
    input == BinaryInput::Probabilities
        && p.is_finite()
        && t.is_finite()
        && !clamps(p)
        && !clamps(1.0 - p)
}

/// The gradient of `form` against the tape. Probabilities: the `f32` tape's bits where a
/// logarithm was clamped or an input is not finite, and within `8u` of it, relative, where
/// the gradient divides: `fl(fl(A / B)·w)` with the division within 2.5 ulp (`5u`) and the
/// product `u` against the CPU's two roundings (`2u`), as smooth L1's. Logits: the `f32`
/// tape's bits where it is not finite, else within [`logit_grad`]'s bound of the exact
/// gradient. Returns the worst share of a bound.
fn check_bce_grad(
    form: Form,
    (pred, target): (&[f32], &[f32]),
    grad: &[f32],
    want: &[f32],
    what: &str,
) -> f64 {
    let w = bce_scale(form, pred.len());
    let exact = |i: usize| match form.0 {
        BinaryInput::Probabilities => !bce_divides(form.0, pred[i], target[i]),
        BinaryInput::Logits => !want[i].is_finite(),
    };
    let pick = |v: &[f32], keep: bool| -> Vec<f32> {
        (0..v.len())
            .filter(|&i| exact(i) == keep)
            .map(|i| v[i])
            .collect()
    };
    assert_exact(&pick(grad, true), &pick(want, true), what);
    let rest: Vec<usize> = (0..grad.len()).filter(|&i| !exact(i)).collect();
    let got: Vec<f32> = rest.iter().map(|&i| grad[i]).collect();
    let (reference, bounds): (Vec<f64>, Vec<f64>) = match form.0 {
        BinaryInput::Probabilities => rest
            .iter()
            .map(|&i| {
                let r = f64::from(want[i]);
                (r, second(8.0 * U * r.abs()))
            })
            .unzip(),
        BinaryInput::Logits => rest
            .iter()
            .map(|&i| logit_grad(pred[i], target[i], w))
            .unzip(),
    };
    assert_within(&got, &reference, |i| bounds[i], what).ratio()
}

/// `n` probabilities: the sigmoid of `[−12, 12)` from `seed`, with every 97th value 0 and
/// every 89th 1, so some logarithms clamp.
fn probabilities(seed: u64, n: usize) -> Vec<f32> {
    let mut p: Vec<f32> = (seeded_values(seed, n, -12.0, 12.0).into_iter())
        .map(|z| 1.0 / (1.0 + (-z).exp()))
        .collect();
    for (i, v) in p.iter_mut().enumerate() {
        if i % 97 == 3 {
            *v = 0.0;
        } else if i % 89 == 5 {
            *v = 1.0;
        }
    }
    p
}

/// `n` logits in `[−20, 20)` from `seed`, every 101st five times larger.
fn logits(seed: u64, n: usize) -> Vec<f32> {
    let mut z = seeded_values(seed, n, -20.0, 20.0);
    for v in z.iter_mut().step_by(101) {
        *v *= 5.0;
    }
    z
}

/// `n` targets in `[0, 1)` from `seed`, every 7th rounded to 0 or 1.
fn targets(seed: u64, n: usize) -> Vec<f32> {
    let mut t = seeded_values(seed, n, 0.0, 1.0);
    for v in t.iter_mut().step_by(7) {
        *v = v.round();
    }
    t
}

/// Each form against the tape on 1, 37, 4,097 and 100,000 values (and probabilities' mean on
/// 2²⁴ + 1,000, more than one grid row of workgroups), the batch sizes no multiple of the
/// workgroup: the loss within [`bce_loss`]'s bound of the exact loss, the gradient as
/// [`check_bce_grad`] checks it, the clamp count the tape's, and a second run's bits the
/// first's.
#[test]
fn binary_cross_entropy_twin() {
    for_each_gpu(|gpu| {
        for (k, form) in FORMS.into_iter().enumerate() {
            let large = k == 0;
            let sizes: &[usize] = if large {
                &[1, 37, 4097, 100_000, (1 << 24) + 1000]
            } else {
                &[1, 37, 4097, 100_000]
            };
            for (s, &n) in sizes.iter().enumerate() {
                let mut kit = kit(gpu, n);
                let bce = BinaryCrossEntropy::new(gpu, &mut kit.cache).unwrap();
                let what = format!("{form:?} of {n} values");
                let seed = 300 + 10 * k as u64 + s as u64;
                let pred = match form.0 {
                    BinaryInput::Probabilities => probabilities(seed, n),
                    BinaryInput::Logits => logits(seed, n),
                };
                let target = targets(seed + 1000, n);
                let (value, count, grad) = gpu_bce(&mut kit, &bce, form, &pred, &target);
                let (_, want_grad, clamped) = tape_bce::<f32>(form, &pred, &target);
                let ratio = check_bce_grad(
                    form,
                    (&pred, &target),
                    &grad,
                    &want_grad,
                    &format!("the gradient, {what}"),
                );
                assert_eq!(count, clamped as f32, "the clamp count, {what}");
                let (want, bound) = bce_loss(form, &pred, &target);
                let worst =
                    assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
                println!(
                    "{what}: loss worst {:.3} of its bound, gradient {ratio:.3}, {clamped} clamped",
                    worst.ratio()
                );
                let (again, again_count, again_grad) =
                    gpu_bce(&mut kit, &bce, form, &pred, &target);
                assert_exact(
                    &[again, again_count],
                    &[value, count],
                    &format!("a second run, {what}"),
                );
                assert_exact(
                    &again_grad,
                    &grad,
                    &format!("a second run's gradient, {what}"),
                );
            }
        }
    });
}

/// Each value on its own (`n = 1`, so the mean's division is by 1) at the edges, against
/// the tape: where the `f32` tape's loss is NaN or infinite the GPU's is the same, elsewhere
/// within [`bce_term`]'s bound of the exact loss; the gradient as [`check_bce_grad`] checks
/// it, so ±∞ and NaN the tape's; the clamp count the tape's.
///
/// Probabilities: ±0 and 1 (clamped), 26·2⁻¹⁴⁹ (clamped) and 27·2⁻¹⁴⁹ (not), other
/// subnormals, 2⁻⁶² and one ulp either side (where the gradient's scaling starts), values
/// near 0 and 1, and outside `[0, 1]` (a NaN loss with the CPU's finite gradient), among them
/// ±1e20 and ±2⁶⁴ and one ulp below it (from 2⁶⁴ the CPU's `p·(1 − p)` overflows, so an
/// infinite target gives a NaN gradient), and NaN or ±∞; against targets of ±0, 1, ¼, a subnormal, NaN and ±∞. Logits: ±0, a subnormal, ±1,
/// ±20, ±88 and ±89 (where `e^(−|z|)` turns subnormal), ±100, ±104, ±1e30, NaN and ±∞; against
/// targets of ±0, 1, ½, 2, NaN and ±∞.
#[test]
fn binary_cross_entropy_edges_match_the_tape() {
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let above = |x: f32| f32::from_bits(x.to_bits() + 1);
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let tiny = 2.0f32.powi(-62);
    // From 2^64 on, the CPU's p (1 - p) overflows to -infinity.
    let huge = 2.0f32.powi(64);
    let probabilities = [
        0.0,
        -0.0,
        1.0,
        f32::from_bits(1),
        f32::from_bits(26),
        f32::from_bits(27),
        1e-40,
        f32::MIN_POSITIVE,
        below(tiny),
        tiny,
        above(tiny),
        1e-30,
        1e-7,
        0.25,
        0.5,
        below(1.0),
        1.0 - 1.0 / 16_777_216.0,
        above(1.0),
        1.5,
        -1e-3,
        -1e-30,
        1e20,
        -1e20,
        below(huge),
        huge,
        -below(huge),
        -huge,
        nan,
        inf,
        -inf,
    ];
    let probability_targets = [0.0, -0.0, 1.0, 0.25, 1e-40, nan, inf, -inf];
    let logits = [
        0.0, -0.0, 1e-40, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0, 89.0, -89.0, 100.0, -100.0, 104.0,
        -104.0, 1e30, -1e30, nan, inf, -inf,
    ];
    let logit_targets = [0.0, -0.0, 1.0, 0.5, 2.0, nan, inf, -inf];
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 64);
        let bce = BinaryCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        for (input, preds, wanted) in [
            (
                BinaryInput::Probabilities,
                &probabilities[..],
                &probability_targets[..],
            ),
            (BinaryInput::Logits, &logits[..], &logit_targets[..]),
        ] {
            let form = (input, LossReduction::Mean);
            for &p in preds {
                for &t in wanted {
                    let what = format!("{input:?} at {p:e}, target {t:e}");
                    let (value, count, grad) = gpu_bce(&mut kit, &bce, form, &[p], &[t]);
                    let (want, want_grad, clamped) = tape_bce::<f32>(form, &[p], &[t]);
                    if want.is_finite() {
                        let (exact, bound) = bce_term(input, p, t);
                        assert_within(&[value], &[exact], |_| second(bound), &what);
                    } else {
                        assert_exact(&[value], &[want], &format!("the loss, {what}"));
                    }
                    check_bce_grad(
                        form,
                        (&[p], &[t]),
                        &grad,
                        &want_grad,
                        &format!("the gradient, {what}"),
                    );
                    assert_eq!(count, clamped as f32, "the clamp count, {what}");
                }
            }
        }
    });
}

/// A NaN or out-of-range value in a batch of 8,192: a NaN probability, a NaN target, a
/// probability of 1.5, an infinite target and an infinite logit give a NaN or infinite loss,
/// which the finite flag catches, and leave every other value's gradient the clean run's
/// bits.
#[test]
fn binary_cross_entropy_non_finite_reaches_the_loss() {
    for_each_gpu(|gpu| {
        let n = 8192;
        let mut kit = kit(gpu, n);
        let bce = BinaryCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let target = targets(1, n);
        for (name, input, at_pred, value) in [
            (
                "a NaN probability",
                BinaryInput::Probabilities,
                true,
                f32::NAN,
            ),
            ("a NaN target", BinaryInput::Probabilities, false, f32::NAN),
            (
                "a probability of 1.5",
                BinaryInput::Probabilities,
                true,
                1.5,
            ),
            (
                "an infinite target",
                BinaryInput::Probabilities,
                false,
                f32::INFINITY,
            ),
            ("a NaN logit", BinaryInput::Logits, true, f32::NAN),
            ("a NaN target", BinaryInput::Logits, false, f32::NAN),
            (
                "an infinite logit",
                BinaryInput::Logits,
                true,
                f32::INFINITY,
            ),
        ] {
            let form = (input, LossReduction::Mean);
            let pred = match input {
                BinaryInput::Probabilities => probabilities(2, n),
                BinaryInput::Logits => logits(2, n),
            };
            let (_, _, clean) = gpu_bce(&mut kit, &bce, form, &pred, &target);
            let (mut p, mut t) = (pred.clone(), target.clone());
            if at_pred {
                p[5000] = value;
            } else {
                t[5000] = value;
            }
            let (loss, _, grad) = gpu_bce(&mut kit, &bce, form, &p, &t);
            assert!(!loss.is_finite(), "{input:?}, {name}: the loss is {loss}");
            let rest: Vec<f32> = grad[..5000].iter().chain(&grad[5001..]).copied().collect();
            let want: Vec<f32> = clean[..5000]
                .iter()
                .chain(&clean[5001..])
                .copied()
                .collect();
            assert_exact(
                &rest,
                &want,
                &format!("{input:?}, {name}: the other values"),
            );
            let losses = kit.upload(&[loss]);
            let flags = kit.zeros(1);
            let check = flag.call(&mut kit.steps, &mut kit.pool, (&losses, 0), &[], &flags);
            let mut step = kit.steps.begin();
            flag.record(&mut step, &check);
            let _ = step.finish();
            assert_eq!(
                kit.read(&flags)[0].to_bits(),
                1,
                "{input:?}, {name}: the finite flag"
            );
            check.release(&mut kit.pool);
        }
    });
}

/// The documented refusals of binary cross-entropy: no values, buffers too short (the output
/// needs the loss's word and the count's), and an output that is a buffer the kernels read.
#[test]
fn binary_cross_entropy_refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu, 64);
        let bce = BinaryCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        let [p, t, g, out] = [(); 4].map(|_| kit.zeros(16));
        let one = kit.zeros(1);
        let form = (BinaryInput::Probabilities, LossReduction::Mean);
        let mut binary = |n: u32,
                          p: &PooledBuffer,
                          t: &PooledBuffer,
                          g: &PooledBuffer,
                          out: (&PooledBuffer, u32)| {
            message(&mut || {
                let _ = bce.call(&mut kit.steps, &mut kit.pool, form, n, p, t, g, out);
            })
        };
        assert!(binary(0, &p, &t, &g, (&out, 0)).contains("where the kernels take 1 to"));
        assert!(binary(17, &p, &t, &g, (&out, 0)).contains("the prediction needs 17 values"));
        assert!(binary(16, &p, &one, &g, (&out, 0)).contains("the target needs 16 values"));
        assert!(binary(16, &p, &t, &one, (&out, 0)).contains("the gradient needs 16 values"));
        assert!(binary(16, &p, &t, &g, (&out, 15)).contains("the loss output needs 17 values"));
        assert!(binary(16, &p, &t, &g, (&one, 0)).contains("the loss output needs 2 values"));
        assert!(
            binary(16, &p, &t, &t, (&out, 0))
                .contains("the gradient and the target must be different")
        );
        assert!(
            binary(16, &p, &t, &g, (&g, 0))
                .contains("the gradient and the loss output must be different")
        );
        assert!(
            binary(16, &p, &t, &g, (&p, 0))
                .contains("the loss output and the prediction must be different")
        );
    });
}

/// Binary cross-entropy's batch mean on 37 rows of 12 values, of probabilities and of
/// logits, against the tape's `Reduction::BatchMean` on `[37, 12]`: the loss within
/// [`bce_loss`]'s bound of the exact loss divided by the batch, the gradient as
/// [`check_bce_grad`] checks it with the scale `1/37`, the clamp count the tape's.
#[test]
fn binary_cross_entropy_batch_mean() {
    for_each_gpu(|gpu| {
        let (batch, width) = (37usize, 12usize);
        let n = batch * width;
        let mut kit = kit(gpu, n);
        let bce = BinaryCrossEntropy::new(gpu, &mut kit.cache).unwrap();
        for input in [BinaryInput::Probabilities, BinaryInput::Logits] {
            let form = (
                input,
                LossReduction::BatchMean {
                    batch: batch as u32,
                },
            );
            let what = format!("{form:?} of {n} values");
            let pred = match input {
                BinaryInput::Probabilities => probabilities(41, n),
                BinaryInput::Logits => logits(41, n),
            };
            let target = targets(42, n);
            let (value, count, grad) = gpu_bce(&mut kit, &bce, form, &pred, &target);
            let (_, want_grad, clamped) = tape_bce::<f32>(form, &pred, &target);
            check_bce_grad(
                form,
                (&pred, &target),
                &grad,
                &want_grad,
                &format!("the gradient, {what}"),
            );
            assert_eq!(count, clamped as f32, "the clamp count, {what}");
            let (want, bound) = bce_loss(form, &pred, &target);
            assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
        }
    });
}

/// The GPU's KL divergence reduced by `reduction`: the loss, at word 1 between two
/// sentinels, and the log-probabilities' gradient.
fn gpu_kl(
    kit: &mut Kit,
    kl: &KlDivergence,
    reduction: LossReduction,
    log_probs: &[f32],
    target: &[f32],
) -> (f32, Vec<f32>) {
    let (x, t) = (kit.upload(log_probs), kit.upload(target));
    let grad = kit.upload(&vec![SENTINEL; log_probs.len()]);
    let out = loss_buffer(kit);
    let call = kl.call(
        &mut kit.steps,
        &mut kit.pool,
        reduction,
        log_probs.len() as u32,
        &x,
        &t,
        &grad,
        (&out, 1),
    );
    let mut step = kit.steps.begin();
    kl.record(&mut step, &call);
    let _ = step.finish();
    let loss = read_loss(kit, &out);
    let g = kit.read(&grad);
    call.release(&mut kit.pool);
    (loss, g)
}

/// What `Tape::kl_div` gives for `reduction` on a store of `T` from the same `f32` values,
/// the values shaped `[batch, n / batch]` for the batch mean and `[n]` otherwise: the loss
/// and the log-probabilities' gradient.
fn tape_kl<T: Real>(reduction: LossReduction, log_probs: &[f32], target: &[f32]) -> (T, Vec<T>) {
    let n = log_probs.len();
    let dims = match reduction {
        LossReduction::BatchMean { batch } => vec![batch as usize, n / batch as usize],
        _ => vec![n],
    };
    let wide = |v: &[f32]| {
        v.iter()
            .map(|&x| T::from_f64(f64::from(x)))
            .collect::<Vec<T>>()
    };
    let mut store = ParamStore::<T>::new_generic();
    let id = store.add(Tensor::from_vec(wide(log_probs), &dims).unwrap());
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&store.values);
    let x = tape.param(id);
    let t = tape.input(Tensor::from_vec(wide(target), &dims).unwrap().view());
    let l = tape.kl_div(x, t, tape_reduction(reduction));
    let value = tape.value(l).data()[0];
    tape.backward(l, &mut store.grads);
    drop(tape);
    (value, store.grad(id).data().to_vec())
}

/// One value's exact loss `ℓ = t·(ln t − x)` (0 where `t` is ±0) and its stated bound
/// `15u·|t·ln t| + 3u·|ℓ| + 2⁻¹²⁶·(1 + |t|)`, for finite values with `t` not negative.
///
/// `ln t` is the kernel's logarithm from the bits, binary cross-entropy's, within `13u` of
/// `ln t` (derived at [`bce_term`]; it holds for subnormal `t`, whose exponent comes from
/// the bits). The subtraction `ln t − x` rounds once (`u·|ln t − x|`, once `ln t`'s own error
/// is carried through, `13u·|ln t|`), and the product once (`u·|ℓ|`); a subnormal `t` takes
/// the product on `t·2³²` and scales back by 2⁻³², both exact, so the same single rounding.
/// A compiler may reassociate the logarithm's last sum, `k·ln2_hi + b`, with the
/// subtraction, as `k·ln2_hi + (b − x)`: `|b|` is at most `ln √2` and `|k·ln2_hi|` at most
/// `2·|ln t|` where `k ≠ 0`, so `|b − x|` is at most `|ln t| + |x|` and `|x|` at most
/// `|ln t| + |ln t − x|`, which adds `2u·|t·ln t| + u·|ℓ|`. A flushed `x` moves the loss by
/// `|t|·2⁻¹²⁶` and a flushed result by 2⁻¹²⁶.
fn kl_term(x: f32, t: f32) -> (f64, f64) {
    if t == 0.0 {
        return (0.0, 0.0);
    }
    let (x, t) = (f64::from(x), f64::from(t));
    let l = t.ln();
    let loss = t * (l - x);
    (
        loss,
        15.0 * U * (t * l).abs() + 3.0 * U * loss.abs() + (1.0 + t.abs()) * FLOOR,
    )
}

/// The exact loss of `reduction` on these values and its stated bound: the values' bounds
/// summed, `(22 + chunks)·u·Σ|ℓ|` for the fixed tree, and for a mean the division by its
/// divisor and `5u` of the mean for it.
fn kl_loss(reduction: LossReduction, log_probs: &[f32], target: &[f32]) -> (f64, f64) {
    let (mut sum, mut abs, mut bounds) = (0.0, 0.0, 0.0);
    for (&x, &t) in log_probs.iter().zip(target) {
        let (v, b) = kl_term(x, t);
        sum += v;
        abs += v.abs();
        bounds += b;
    }
    let tree = depth(log_probs.len()) * U * abs;
    match divisor(reduction, log_probs.len()) {
        Some(d) => {
            let mean = sum / d;
            (
                mean,
                second((bounds + tree) / d + 5.0 * U * mean.abs()) + 3.0 * FLOOR,
            )
        }
        None => (sum, second(bounds + tree) + 3.0 * FLOOR),
    }
}

/// `n` log-probabilities: the logarithm of a value in `[10⁻⁶, 1)` from `seed`, every 17th
/// five times further below 0 and every 23rd above 0, so some are not log-probabilities.
fn kl_inputs(seed: u64, n: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (seeded_values(seed, n, 1e-6, 1.0).into_iter())
        .map(f32::ln)
        .collect();
    for (i, v) in x.iter_mut().enumerate() {
        if i % 17 == 2 {
            *v *= 5.0;
        } else if i % 23 == 4 {
            *v = -*v;
        }
    }
    x
}

/// `n` targets in `[0, 1)` from `seed`, every 5th exactly 0, every 13th 1, every 31st a
/// subnormal and every 37th below 10⁻²⁰.
fn kl_targets(seed: u64, n: usize) -> Vec<f32> {
    let mut t = seeded_values(seed, n, 0.0, 1.0);
    for (i, v) in t.iter_mut().enumerate() {
        if i % 5 == 1 {
            *v = 0.0;
        } else if i % 13 == 3 {
            *v = 1.0;
        } else if i % 31 == 7 {
            *v *= 1e-38;
        } else if i % 37 == 9 {
            *v *= 1e-20;
        }
    }
    t
}

/// The reductions the KL divergence's twin test runs on `n` values: the mean, the sum and
/// the batch mean over the largest batch up to 1,000 that divides `n`.
fn kl_reductions(n: usize) -> [LossReduction; 3] {
    let batch = (1..=n.min(1000))
        .rev()
        .find(|&b| n.is_multiple_of(b))
        .unwrap_or(1) as u32;
    [
        LossReduction::Mean,
        LossReduction::Sum,
        LossReduction::BatchMean { batch },
    ]
}

/// Each reduction against the tape on 1, 37, 4,097 and 100,000 values (and the batch mean
/// on 2²⁴ + 1,000 values over 8 rows, more than one grid row of workgroups), no size a
/// multiple of the workgroup: the gradient the `f32` tape's bits (a subnormal one possibly
/// flushed), the loss within [`kl_loss`]'s bound of the exact loss, and a second run's bits
/// the first's.
#[test]
fn kl_div_twin() {
    for_each_gpu(|gpu| {
        let mut runs: Vec<(usize, LossReduction)> = Vec::new();
        for n in [1, 37, 4097, 100_000] {
            runs.extend(kl_reductions(n).into_iter().map(|r| (n, r)));
        }
        runs.push(((1 << 24) + 1000, LossReduction::BatchMean { batch: 8 }));
        for (k, &(n, reduction)) in runs.iter().enumerate() {
            let mut kit = kit(gpu, n);
            let kl = KlDivergence::new(gpu, &mut kit.cache).unwrap();
            let what = format!("{reduction:?} of {n} values");
            let x = kl_inputs(500 + k as u64, n);
            let t = kl_targets(700 + k as u64, n);
            let (value, grad) = gpu_kl(&mut kit, &kl, reduction, &x, &t);
            let (_, want_grad) = tape_kl::<f32>(reduction, &x, &t);
            assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
            let (want, bound) = kl_loss(reduction, &x, &t);
            let worst = assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
            println!("{what}: loss worst {:.3} of its bound", worst.ratio());
            let (again, again_grad) = gpu_kl(&mut kit, &kl, reduction, &x, &t);
            assert_exact(&[again], &[value], &format!("a second run, {what}"));
            assert_exact(
                &again_grad,
                &grad,
                &format!("a second run's gradient, {what}"),
            );
        }
    });
}

/// Each value on its own (`n = 1`, so every reduction divides by 1) at the edges, against
/// the tape: where the `f32` tape's loss is NaN or infinite the GPU's is the same, elsewhere
/// within [`kl_term`]'s bound of the exact loss; the gradient the `f32` tape's bits.
///
/// Log-probabilities: ±0, a subnormal, −1, `ln ½`, −20, −88, −104, −1e30, 1, 3, ±`f32::MAX`,
/// NaN and ±∞. Targets: ±0, the smallest subnormal, 10⁻⁴⁰, the smallest normal, 10⁻³⁰,
/// ¼, ½, one ulp either side of 1, 1, 2 (where `t·(ln t − x)` overflows against −`f32::MAX`),
/// −¼, NaN and ±∞.
#[test]
fn kl_div_edges_match_the_tape() {
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let above = |x: f32| f32::from_bits(x.to_bits() + 1);
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let inputs = [
        0.0,
        -0.0,
        -1e-40,
        -1.0,
        0.5f32.ln(),
        -20.0,
        -88.0,
        -104.0,
        -1e30,
        1.0,
        3.0,
        f32::MAX,
        -f32::MAX,
        nan,
        inf,
        -inf,
    ];
    let targets = [
        0.0,
        -0.0,
        f32::from_bits(1),
        1e-40,
        f32::MIN_POSITIVE,
        1e-30,
        0.25,
        0.5,
        below(1.0),
        1.0,
        above(1.0),
        2.0,
        -0.25,
        nan,
        inf,
        -inf,
    ];
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 64);
        let kl = KlDivergence::new(gpu, &mut kit.cache).unwrap();
        let mut worst: f64 = 0.0;
        for reduction in [
            LossReduction::Mean,
            LossReduction::Sum,
            LossReduction::BatchMean { batch: 1 },
        ] {
            for &x in &inputs {
                for &t in &targets {
                    let what = format!("{reduction:?} at {x:e}, target {t:e}");
                    let (value, grad) = gpu_kl(&mut kit, &kl, reduction, &[x], &[t]);
                    let (want, want_grad) = tape_kl::<f32>(reduction, &[x], &[t]);
                    if want.is_finite() {
                        let (exact, bound) = kl_term(x, t);
                        let w = assert_within(&[value], &[exact], |_| second(bound), &what);
                        worst = worst.max(w.ratio());
                    } else {
                        assert_exact(&[value], &[want], &format!("the loss, {what}"));
                    }
                    // -t·w with a subnormal t is subnormal, and a GPU may flush t itself
                    // before the product, so there the gradient is held to FLOOR; every
                    // other gradient is the tape's bits.
                    let what = format!("the gradient, {what}");
                    if want_grad[0].is_subnormal() {
                        assert_within(&grad, &[f64::from(want_grad[0])], |_| FLOOR, &what);
                    } else {
                        assert_exact(&grad, &want_grad, &what);
                    }
                }
            }
        }
        println!("edges: loss worst {worst:.3} of its bound");
    });
}

/// A NaN or out-of-range value in a batch of 8,192: a NaN log-probability (where the target
/// is 0 too), a NaN target, a negative target, an infinite target and a log-probability of
/// +∞ give a NaN or infinite loss, which the finite flag catches, and leave every other
/// value's gradient the clean run's bits.
#[test]
fn kl_div_non_finite_reaches_the_loss() {
    for_each_gpu(|gpu| {
        let n = 8192;
        let mut kit = kit(gpu, n);
        let kl = KlDivergence::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let reduction = LossReduction::BatchMean { batch: 64 };
        let x = kl_inputs(3, n);
        let t = kl_targets(4, n);
        // Value 5,001 has a target of 0.
        assert_eq!(t[5001], 0.0);
        let (_, clean) = gpu_kl(&mut kit, &kl, reduction, &x, &t);
        for (name, at, on_input, value) in [
            ("a NaN log-probability", 5000, true, f32::NAN),
            ("a NaN log-probability against 0", 5001, true, f32::NAN),
            ("a NaN target", 5000, false, f32::NAN),
            ("a negative target", 5000, false, -0.5),
            ("an infinite target", 5000, false, f32::INFINITY),
            ("a log-probability of +inf", 5000, true, f32::INFINITY),
        ] {
            let (mut xs, mut ts) = (x.clone(), t.clone());
            if on_input {
                xs[at] = value;
            } else {
                ts[at] = value;
            }
            let (loss, grad) = gpu_kl(&mut kit, &kl, reduction, &xs, &ts);
            assert!(!loss.is_finite(), "{name}: the loss is {loss}");
            let rest: Vec<f32> = grad[..at].iter().chain(&grad[at + 1..]).copied().collect();
            let want: Vec<f32> = clean[..at]
                .iter()
                .chain(&clean[at + 1..])
                .copied()
                .collect();
            assert_exact(&rest, &want, &format!("{name}: the other values"));
            let losses = kit.upload(&[loss]);
            let flags = kit.zeros(1);
            let check = flag.call(&mut kit.steps, &mut kit.pool, (&losses, 0), &[], &flags);
            let mut step = kit.steps.begin();
            flag.record(&mut step, &check);
            let _ = step.finish();
            assert_eq!(kit.read(&flags)[0].to_bits(), 1, "{name}: the finite flag");
            check.release(&mut kit.pool);
        }
    });
}

/// The documented refusals of KL divergence: no values, buffers too short, a batch mean
/// whose batch is 0 or does not divide the values, and an output that is a buffer the
/// kernels read.
#[test]
fn kl_div_refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu, 64);
        let kl = KlDivergence::new(gpu, &mut kit.cache).unwrap();
        let [x, t, g, out] = [(); 4].map(|_| kit.zeros(16));
        let one = kit.zeros(1);
        let mut kl_div = |reduction: LossReduction,
                          n: u32,
                          x: &PooledBuffer,
                          t: &PooledBuffer,
                          g: &PooledBuffer,
                          out: (&PooledBuffer, u32)| {
            message(&mut || {
                let _ = kl.call(&mut kit.steps, &mut kit.pool, reduction, n, x, t, g, out);
            })
        };
        let mean = LossReduction::Mean;
        assert!(kl_div(mean, 0, &x, &t, &g, (&out, 0)).contains("where the kernels take 1 to"));
        assert!(kl_div(mean, 17, &x, &t, &g, (&out, 0)).contains("the log-probabilities needs 17"));
        assert!(kl_div(mean, 16, &x, &one, &g, (&out, 0)).contains("the target needs 16 values"));
        assert!(kl_div(mean, 16, &x, &t, &one, (&out, 0)).contains("the gradient needs 16 values"));
        assert!(
            kl_div(mean, 16, &x, &t, &g, (&out, 16)).contains("the loss output needs 17 values")
        );
        for batch in [0, 3, 32] {
            let r = LossReduction::BatchMean { batch };
            let text = format!("a batch of {batch} rows does not divide 16 values");
            assert!(
                kl_div(r, 16, &x, &t, &g, (&out, 0)).contains(&text),
                "{batch}"
            );
        }
        assert!(
            kl_div(mean, 16, &x, &t, &t, (&out, 0))
                .contains("the gradient and the target must be different")
        );
        assert!(
            kl_div(mean, 16, &x, &t, &g, (&g, 0))
                .contains("the gradient and the loss output must be different")
        );
        assert!(
            kl_div(mean, 16, &x, &t, &g, (&x, 0))
                .contains("the loss output and the log-probabilities must be different")
        );
    });
}
