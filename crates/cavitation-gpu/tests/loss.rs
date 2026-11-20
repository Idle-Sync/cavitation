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
//!
//! Every loss is written at word 1 of a buffer of sentinels, to show nothing else changes. The
//! tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, Real, Reduction, Tensor};
use cavitation_gpu::kernels::{
    FiniteFlag, MeanSquaredError, NegativeLogLikelihood, Regression, RegressionLoss,
    SoftmaxCrossEntropy,
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
