//! The losses on the card against the tape's, as `cavitation-gpu`'s tests check its WGSL
//! kernels:
//!
//! - **Softmax cross-entropy:** the loss and the logits' gradient within the WGSL kernel's
//!   stated bounds of `Tape::softmax_cross_entropy` on an `f64` store, from one row to 70,000
//!   rows and from one class to 1,000, the `cifar-cnn` network's 128 × 10 and its last batch
//!   of 80 included. A NaN or infinite logit, or a bad label, gives a NaN row and a NaN loss.
//!   On confident rows the loss carries no bias.
//! - **Mean squared error:** the gradient bit for bit against `Tape::mse` on an `f32` store,
//!   subnormals included; the loss the bits of the fixed tree evaluated on the host, the
//!   `f32` tape's bits on squares whose sums are exact, and within its bound of the `f64`
//!   tape. Both are checked up to 2²⁴ + 1,000 values, and an overflowed square and a NaN
//!   reach the loss.
//! - **Negative log-likelihood:** the gradient bit for bit against `Tape::nll` on an `f32`
//!   store; the loss the bits of the fixed tree evaluated on the host and within its bound of
//!   the `f64` tape, on the cross-entropy's shapes; NaN and ±∞ as on the CPU, and a bad label
//!   giving a NaN row.
//! - **L1, Huber and smooth L1:** the gradient bit for bit against `Tape::l1`, `Tape::huber`
//!   and `Tape::smooth_l1` on an `f32` store, every branch included; the loss the bits of the
//!   fixed tree evaluated on the host over the tape's terms and within the WGSL kernel's bound
//!   of the `f64` tape, up to 2²⁴ + 1,000 values; each value at the edges (±0, the threshold
//!   and one ulp either side, a threshold of 0 or +∞, NaN, ±∞, overflow and subnormals) the
//!   `f32` tape's bits; the tape's refusal of a bad threshold.
//! - **Binary cross-entropy:** of probabilities and of logits, with the mean and the sum: a
//!   probability's gradient bit for bit against `Tape::binary_cross_entropy` on an `f32`
//!   store, subnormals included, and its loss the bits of the fixed tree evaluated on the host
//!   over the kernel's terms; the logits' gradient, and every loss, within the WGSL kernel's
//!   bounds of the exact values; the clamp count `Tape::clamped_logs`, up to 2²⁴ + 1,000
//!   values; each value at the edges (0 and 1, the clamp's threshold, subnormals, ±100 and
//!   larger logits, values outside `[0, 1]`, NaN and ±∞) as the `f32` tape gives it; the
//!   batch mean against `Reduction::BatchMean`.
//! - **KL divergence:** with the mean, the sum and the batch mean: the gradient bit for bit
//!   against `Tape::kl_div` on an `f32` store, subnormals included; the loss the bits of the
//!   fixed tree evaluated on the host over the kernel's terms and within the WGSL kernel's
//!   bound of the exact loss, up to 2²⁴ + 1,000 values; each value at the edges (targets of
//!   ±0, subnormals, 1 and one ulp either side, above 1, negative, NaN and ±∞;
//!   log-probabilities of ±0 to ±`f32::MAX`, NaN and ±∞) as the `f32` tape gives it; the
//!   refusals.
//!
//! Every output starts as NaN, every loss is written at word 1 between two sentinels, and two
//! runs give the same bits. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1`
//! makes that a failure.

mod common;

use cavitation_core::{Graph, ParamStore, Real, Reduction, Tensor};
use cavitation_cuda::{
    BinaryCrossEntropy, BinaryInput, Cuda, CudaError, DeviceBuffer, KlDivergence, LossReduction,
    MeanSquaredError, NegativeLogLikelihood, Regression, RegressionLoss, SoftmaxCrossEntropy,
};
use common::{
    FLOOR, U, assert_exact, assert_within, exact_values, nans, panic_message, read, seeded_values,
    upload,
};

/// What fills the words around a loss.
const SENTINEL: f32 = 7.0;

/// The error bound of a sum of `k` terms in any order: `2(k + 2) · u · Σ|terms|`.
fn sum_bound(k: usize, abs_sum: f64) -> f64 {
    2.0 * (k as f64 + 2.0) * U * abs_sum
}

/// The relative error of a sum of `len` terms in the channel sums' fixed tree, in units of
/// `u`, as the WGSL bound counts it: each term passes through at most 15 + 8 + (chunks − 1)
/// additions.
fn depth(len: usize) -> f64 {
    22.0 + len.div_ceil(4096) as f64
}

/// `b` with an allowance for rounding the result it bounds.
fn second(b: f64) -> f64 {
    b * (1.0 + 16.0 * U)
}

/// The sum of `terms` in the channel sums' fixed tree, evaluated on the host: chunks of
/// 4,096, each of 256 lanes adding every 256th term of its chunk in order from +0, the lanes
/// meeting in a halving tree, and the chunks added in order.
fn tree(terms: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for (index, chunk) in terms.chunks(4096).enumerate() {
        let mut lanes = [0.0f32; 256];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for t in (lane..chunk.len()).step_by(256) {
                acc += chunk[t];
            }
            *slot = acc;
        }
        let mut width = 128;
        while width > 0 {
            for lane in 0..width {
                lanes[lane] += lanes[lane + width];
            }
            width /= 2;
        }
        total = if index == 0 {
            lanes[0]
        } else {
            total + lanes[0]
        };
    }
    total
}

/// The loss buffer: NaN at word 1 between two sentinels.
fn loss_buffer(cuda: &Cuda) -> DeviceBuffer<'_> {
    upload(cuda, &[SENTINEL, f32::NAN, SENTINEL])
}

/// The loss at word 1 of `out`, after checking the sentinels around it.
fn read_loss(out: &DeviceBuffer<'_>) -> f32 {
    let got = read(out, 3);
    assert_eq!(got[0], SENTINEL, "the word before the loss changed");
    assert_eq!(got[2], SENTINEL, "the word after the loss changed");
    got[1]
}

/// The card's cross-entropy of `logits` against `labels`: the loss and the gradient.
fn cuda_ce(
    cuda: &Cuda,
    ce: &SoftmaxCrossEntropy<'_>,
    logits: &[f32],
    labels: &[u32],
) -> (f32, Vec<f32>) {
    let (z, l) = (upload(cuda, logits), upload(cuda, labels));
    let mut grad = nans(cuda, logits.len());
    let mut out = loss_buffer(cuda);
    ce.run(&z, &l, &mut grad, (&mut out, 1)).expect("run");
    (read_loss(&out), read(&grad, logits.len()))
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
/// - `cifar-cnn`'s 128 × 10 and its last batch of 80 (50,000 = 390 · 128 + 80);
/// - 1,000 rows of 3;
/// - 7 rows of 1,000;
/// - 65 rows of one class (a loss of 0, a gradient of 0, both exact);
/// - 70,000 rows of 4 (more than 256 blocks of rows, and a sum over 18 chunks).
///
/// The logits run over `[−5, 5)` and, for the exponential's wider error, `[−40, 40)`. A
/// second run gives the same bits.
#[test]
fn cross_entropy_twin() {
    let Some(cuda) = Cuda::new() else { return };
    for (s, (batch, classes)) in [
        (1u32, 10u32),
        (128, 10),
        (80, 10),
        (1000, 3),
        (7, 1000),
        (65, 1),
        (70_000, 4),
    ]
    .into_iter()
    .enumerate()
    {
        let n = (batch * classes) as usize;
        let ce = SoftmaxCrossEntropy::new(&cuda, (batch, classes)).expect("cross-entropy");
        assert_eq!(ce.shape(), (batch, classes));
        for range in [5.0f32, 40.0] {
            let what = format!("{batch} × {classes}, logits within ±{range}");
            let z = seeded_values(10 * s as u64 + 1, n, -range, range);
            let y = labels(10 * s as u64 + 2, batch as usize, classes);
            let (loss, grad) = cuda_ce(&cuda, &ce, &z, &y);
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
            if classes == 1 {
                assert_exact(&[loss], &[0.0], &format!("loss, {what}"));
                assert_exact(&grad, &vec![0.0; n], &format!("gradient, {what}"));
            }
            let (again, again_grad) = cuda_ce(&cuda, &ce, &z, &y);
            assert_eq!(
                again.to_bits(),
                loss.to_bits(),
                "the loss on a second run, {what}"
            );
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(
                bits(&again_grad),
                bits(&grad),
                "the gradient on a second run, {what}"
            );
        }
    }
}

/// Rows `[0, z]` labelled 1 for `z` from −15 to −7: the label's probability is far below 1/2,
/// so `p − 1` rounds, and its gradient takes three roundings on `p − onehot`. Each gradient is
/// within its bound of the tape's.
#[test]
fn cross_entropy_label_far_below_the_max() {
    let Some(cuda) = Cuda::new() else { return };
    let batch = 1000u32;
    let ce = SoftmaxCrossEntropy::new(&cuda, (batch, 2)).expect("cross-entropy");
    let z: Vec<f32> = (0..batch)
        .flat_map(|i| [0.0, -15.0 + 8.0 * i as f32 / batch as f32])
        .collect();
    let y = vec![1u32; batch as usize];
    let (loss, grad) = cuda_ce(&cuda, &ce, &z, &y);
    let (want_loss, want_grad) = tape_ce(batch as usize, 2, &z, &y);
    let (loss_bound, grad_bound) = ce_bounds(batch as usize, 2, &z, &y);
    assert_within(&[loss], &[want_loss], |_| loss_bound, "the loss");
    let worst = assert_within(&grad, &want_grad, |i| grad_bound[i], "the gradient");
    println!(
        "labels far below the max: worst {:.3} of its bound",
        worst.ratio()
    );
}

/// Logits far below their row's maximum, where `expf` gives subnormals and 0: rows
/// `[0, −100, −150]` with each label (e^−100 is subnormal and e^−150 underflows to 0), rows of
/// 10 logits in `[−60, 60)` at `cifar-cnn`'s 128 and 80, and 1,000 rows of 3 in `[−200, 200)`.
/// The loss and the gradient are within their bounds, the absolute floor included, of the
/// `f64` tape's.
#[test]
fn cross_entropy_subnormal_exponentials() {
    let Some(cuda) = Cuda::new() else { return };
    let mut cases: Vec<(u32, u32, Vec<f32>, Vec<u32>)> =
        vec![(3, 3, [0.0, -100.0, -150.0].repeat(3), vec![0, 1, 2])];
    for (s, (batch, classes, range)) in [(128u32, 10u32, 60.0f32), (80, 10, 60.0), (1000, 3, 200.0)]
        .into_iter()
        .enumerate()
    {
        let n = (batch * classes) as usize;
        let z = seeded_values(60 + s as u64, n, -range, range);
        cases.push((
            batch,
            classes,
            z,
            labels(70 + s as u64, batch as usize, classes),
        ));
    }
    for (batch, classes, z, y) in cases {
        let what = format!("{batch} × {classes}");
        let ce = SoftmaxCrossEntropy::new(&cuda, (batch, classes)).expect("cross-entropy");
        let (loss, grad) = cuda_ce(&cuda, &ce, &z, &y);
        let (want_loss, want_grad) = tape_ce(batch as usize, classes as usize, &z, &y);
        let (loss_bound, grad_bound) = ce_bounds(batch as usize, classes as usize, &z, &y);
        let worst = assert_within(
            &[loss],
            &[want_loss],
            |_| loss_bound,
            &format!("loss, {what}"),
        );
        println!(
            "far logits {what}: loss worst {:.3} of its bound",
            worst.ratio()
        );
        let worst = assert_within(
            &grad,
            &want_grad,
            |i| grad_bound[i],
            &format!("gradient, {what}"),
        );
        println!(
            "far logits {what}: gradient worst {:.3} of its bound",
            worst.ratio()
        );
        if batch == 3 {
            assert!(
                grad.iter().any(|g| g.is_subnormal()),
                "the [0, −100, −150] rows reach subnormal gradients: {grad:?}"
            );
        }
    }
}

/// A row with a NaN, +∞ or −∞ logit, or a label not below the classes, gives a NaN loss and a
/// NaN gradient row. The other rows' gradients are the bits of a run without the bad row.
#[test]
fn cross_entropy_bad_rows_give_nan() {
    let Some(cuda) = Cuda::new() else { return };
    let ce = SoftmaxCrossEntropy::new(&cuda, (4, 5)).expect("cross-entropy");
    let z = seeded_values(1, 20, -3.0, 3.0);
    let y = vec![0u32, 4, 2, 1];
    let (loss, clean) = cuda_ce(&cuda, &ce, &z, &y);
    assert!(loss.is_finite());
    for (name, at, value, label) in [
        ("a NaN logit", 7, f32::NAN, 4),
        ("a +inf logit", 5, f32::INFINITY, 4),
        ("a -inf logit off the label", 8, f32::NEG_INFINITY, 4),
        ("a -inf logit on the label", 9, f32::NEG_INFINITY, 4),
        ("label 5 of 5 classes", 6, z[6], 5),
        ("label u32::MAX", 6, z[6], u32::MAX),
    ] {
        let (mut zb, mut yb) = (z.clone(), y.clone());
        zb[at] = value;
        yb[1] = label;
        let (loss, grad) = cuda_ce(&cuda, &ce, &zb, &yb);
        assert!(loss.is_nan(), "{name}: the loss is {loss}, not NaN");
        assert!(
            grad[5..10].iter().all(|g| g.is_nan()),
            "{name}: the bad row's gradient"
        );
        let rest: Vec<f32> = grad[..5].iter().chain(&grad[10..]).copied().collect();
        let want: Vec<f32> = clean[..5].iter().chain(&clean[10..]).copied().collect();
        assert_exact(&rest, &want, &format!("{name}: the other rows"));
    }
}

/// Confident rows, as late in training: 4,096 rows of 10 logits in `[−4, 4)` with the label's
/// raised by 7 or by 12 (mean losses near 0.24 and 0.0026), 20 batches each. Each batch's
/// loss is within its bound of the `f64` tape's, and the mean of their relative errors is
/// within 8u: rounding, not a bias.
#[test]
fn cross_entropy_confident_rows_have_no_bias() {
    let Some(cuda) = Cuda::new() else { return };
    let (batch, classes) = (4096u32, 10u32);
    let n = (batch * classes) as usize;
    let ce = SoftmaxCrossEntropy::new(&cuda, (batch, classes)).expect("cross-entropy");
    for raise in [7.0f32, 12.0] {
        let mut sum = 0.0;
        for seed in 0..20u64 {
            let mut z = seeded_values(100 + seed, n, -4.0, 4.0);
            let y = labels(200 + seed, batch as usize, classes);
            for (i, &l) in y.iter().enumerate() {
                z[i * classes as usize + l as usize] += raise;
            }
            let (loss, _) = cuda_ce(&cuda, &ce, &z, &y);
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
}

/// The card's mean squared error: the loss and the gradient.
fn cuda_mse(
    cuda: &Cuda,
    mse: &MeanSquaredError<'_>,
    pred: &[f32],
    target: &[f32],
) -> (f32, Vec<f32>) {
    let (p, t) = (upload(cuda, pred), upload(cuda, target));
    let mut grad = nans(cuda, pred.len());
    let mut out = loss_buffer(cuda);
    mse.run(&p, &t, &mut grad, (&mut out, 1)).expect("run");
    (read_loss(&out), read(&grad, pred.len()))
}

/// `Tape::mse` with `Reduction::Mean` on an `f32` store: the loss and the prediction's
/// gradient.
fn tape_mse(pred: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let n = pred.len();
    let mut store = ParamStore::new();
    let id = store.add(Tensor::from_vec(pred.to_vec(), &[n]).unwrap());
    let mut graph = Graph::new();
    let mut tape = graph.tape(&store.values);
    let p = tape.param(id);
    let t = tape.input(Tensor::from_vec(target.to_vec(), &[n]).unwrap().view());
    let l = tape.mse(p, t, Reduction::Mean);
    let loss = tape.value(l).data()[0];
    tape.backward(l, &mut store.grads);
    drop(tape);
    (loss, store.grad(id).data().to_vec())
}

/// The squared errors as the kernel computes them, in `f32`.
fn squares(pred: &[f32], target: &[f32]) -> Vec<f32> {
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| (p - t) * (p - t))
        .collect()
}

/// The mean squared error against the tape, for 1, 37, 4,097, 100,000 and 2²⁴ + 1,000
/// values: the gradient bit for bit against an `f32` store's; the loss the bits of the fixed
/// tree evaluated on the host divided by `n`, and within
/// `(25 + chunks)·u·Σ d² / n + 5u·loss` of an `f64` store's. A second run gives the same
/// bits.
#[test]
fn mse_twin() {
    let Some(cuda) = Cuda::new() else { return };
    for (s, n) in [1usize, 37, 4097, 100_000, (1 << 24) + 1000]
        .into_iter()
        .enumerate()
    {
        let mse = MeanSquaredError::new(&cuda, n as u32).expect("mse");
        assert_eq!(mse.len() as usize, n);
        let pred = seeded_values(20 + s as u64, n, -2.0, 2.0);
        let target = seeded_values(30 + s as u64, n, -2.0, 2.0);
        let (loss, grad) = cuda_mse(&cuda, &mse, &pred, &target);
        let (_, want_grad) = tape_mse(&pred, &target);
        assert_exact(&grad, &want_grad, &format!("the gradient of {n} values"));
        let host = tree(&squares(&pred, &target)) / n as f32;
        assert_eq!(
            loss.to_bits(),
            host.to_bits(),
            "the loss of {n} values against the host's tree"
        );

        let sum: f64 = pred
            .iter()
            .zip(&target)
            .map(|(&p, &t)| (f64::from(p) - f64::from(t)).powi(2))
            .sum();
        let want = sum / n as f64;
        let relative = (depth(n) + 3.0) * U * sum / n as f64;
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
        let (again, again_grad) = cuda_mse(&cuda, &mse, &pred, &target);
        assert_eq!(again.to_bits(), loss.to_bits(), "a second run's loss");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&again_grad), bits(&grad), "a second run's gradient");
    }
}

/// On differences of ±1 and ±2, whose squares sum exactly, the loss is the `f32` tape's bits;
/// on differences near the smallest normal, whose gradients are subnormal, the gradient is
/// still the tape's bits (nothing is flushed), and so is the loss of 0 their underflowed
/// squares give.
#[test]
fn mse_exact_twins() {
    let Some(cuda) = Cuda::new() else { return };
    for n in [3usize, 4096, 10_000] {
        let mse = MeanSquaredError::new(&cuda, n as u32).expect("mse");
        let pred = exact_values(40 + n as u64, n);
        let target = vec![0.0f32; n];
        let (loss, grad) = cuda_mse(&cuda, &mse, &pred, &target);
        let (want_loss, want_grad) = tape_mse(&pred, &target);
        assert_exact(
            &[loss],
            &[want_loss],
            &format!("the loss of {n} small integers"),
        );
        assert_exact(
            &grad,
            &want_grad,
            &format!("the gradient of {n} small integers"),
        );

        let tiny: Vec<f32> = seeded_values(50 + n as u64, n, -4.0, 4.0)
            .into_iter()
            .map(|v| v * f32::MIN_POSITIVE)
            .collect();
        let (loss, grad) = cuda_mse(&cuda, &mse, &tiny, &target);
        let (want_loss, want_grad) = tape_mse(&tiny, &target);
        assert!(
            want_grad.iter().any(|g| g.is_subnormal()),
            "the twin has subnormal gradients"
        );
        assert_exact(
            &[loss],
            &[want_loss],
            &format!("the loss of {n} tiny values"),
        );
        assert_exact(
            &grad,
            &want_grad,
            &format!("the gradient of {n} tiny values"),
        );
    }
}

/// A difference whose square overflows gives an infinite loss and a finite gradient; a NaN
/// or infinite prediction, or a NaN target, gives a loss that is not finite and a gradient
/// that is not finite at that value. Every other gradient is the tape's bits.
#[test]
fn mse_non_finite_reaches_the_loss() {
    let Some(cuda) = Cuda::new() else { return };
    let n = 8192;
    let mse = MeanSquaredError::new(&cuda, n as u32).expect("mse");
    for (name, value, in_target) in [
        ("1e20", 1e20f32, false),
        ("NaN", f32::NAN, false),
        ("+inf", f32::INFINITY, false),
        ("-inf", f32::NEG_INFINITY, false),
        ("a NaN target", f32::NAN, true),
    ] {
        let mut pred = seeded_values(1, n, -1.0, 1.0);
        let mut target = vec![0.0f32; n];
        if in_target {
            target[5000] = value;
        } else {
            pred[5000] = value;
        }
        let (loss, grad) = cuda_mse(&cuda, &mse, &pred, &target);
        assert!(!loss.is_finite(), "{name}: the loss is {loss}");
        assert_eq!(
            grad[5000].is_finite(),
            value == 1e20,
            "{name}: its gradient {}",
            grad[5000]
        );
        let (want_loss, want_grad) = tape_mse(&pred, &target);
        assert_exact(&grad, &want_grad, &format!("{name}: the gradient"));
        assert_eq!(
            loss.is_nan(),
            want_loss.is_nan(),
            "{name}: the loss {loss} against the tape's {want_loss}"
        );
    }
}

/// The refusals: a shape with a dimension of 0 or more than 2³¹ values is
/// `CudaError::Invalid`; a buffer too short panics before any launch, naming it.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let invalid = |r: Result<(), CudaError>, why: &str| match r {
        Err(CudaError::Invalid(text)) => assert!(text.contains(why), "{text}"),
        other => panic!("accepted: {other:?}"),
    };
    let ce = |shape| SoftmaxCrossEntropy::new(&cuda, shape).map(|_| ());
    invalid(ce((0, 4)), "above 0");
    invalid(ce((4, 0)), "above 0");
    invalid(ce((1 << 20, 1 << 12)), "more than the 2147483648");
    let mse = |n| MeanSquaredError::new(&cuda, n).map(|_| ());
    invalid(mse(0), "above 0");

    let ce = SoftmaxCrossEntropy::new(&cuda, (4, 4)).expect("cross-entropy");
    let (z, l) = (nans(&cuda, 16), upload(&cuda, &[0u32; 16]));
    let one = nans(&cuda, 1);
    let mut short_grad = nans(&cuda, 15);
    let mut grad = nans(&cuda, 16);
    let mut loss = nans(&cuda, 16);
    let m = panic_message(|| {
        let _ = ce.run(&one, &l, &mut grad, (&mut loss, 0));
    });
    assert!(
        m.contains("the logits buffer holds 4 bytes, and 16 values need 64"),
        "{m}"
    );
    let m = panic_message(|| {
        let _ = ce.run(&z, &one, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the labels buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = ce.run(&z, &l, &mut short_grad, (&mut loss, 0));
    });
    assert!(m.contains("the gradient buffer holds 60 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = ce.run(&z, &l, &mut grad, (&mut loss, 16));
    });
    assert!(
        m.contains("the loss output buffer holds 64 bytes, and 17 values need 68"),
        "{m}"
    );
    assert!(
        read(&grad, 16).iter().all(|v| v.is_nan()),
        "a refused call wrote the gradient"
    );

    let mse = MeanSquaredError::new(&cuda, 16).expect("mse");
    let m = panic_message(|| {
        let _ = mse.run(&one, &z, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the prediction buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = mse.run(&z, &one, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the target buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = mse.run(&z, &l, &mut short_grad, (&mut loss, 0));
    });
    assert!(m.contains("the gradient buffer holds 60 bytes"), "{m}");
}

/// The card's negative log-likelihood of `log_probs` against `labels`: the loss and the
/// gradient.
fn cuda_nll(
    cuda: &Cuda,
    nll: &NegativeLogLikelihood<'_>,
    log_probs: &[f32],
    labels: &[u32],
) -> (f32, Vec<f32>) {
    let (lp, l) = (upload(cuda, log_probs), upload(cuda, labels));
    let mut grad = nans(cuda, log_probs.len());
    let mut out = loss_buffer(cuda);
    nll.run(&lp, &l, &mut grad, (&mut out, 1)).expect("run");
    (read_loss(&out), read(&grad, log_probs.len()))
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

/// The negative log-likelihood against the tape, on the cross-entropy's shapes with
/// log-probabilities in `[−12, 0)`: the gradient bit for bit against an `f32` store's; the
/// loss the bits of the fixed tree evaluated on the host over the row losses (each its
/// label's value negated) divided by the batch, and within `(22 + chunks)·u·Σ|ℓ| / B +
/// 5u·loss` of an `f64` store's. A second run gives the same bits.
#[test]
fn nll_twin() {
    let Some(cuda) = Cuda::new() else { return };
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
        let nll = NegativeLogLikelihood::new(&cuda, (batch, classes)).expect("nll");
        assert_eq!(nll.shape(), (batch, classes));
        let what = format!("{batch} × {classes}");
        let lp = seeded_values(40 + s as u64, n, -12.0, 0.0);
        let y = labels(50 + s as u64, batch as usize, classes);
        let (loss, grad) = cuda_nll(&cuda, &nll, &lp, &y);
        let (_, want_grad) = tape_nll::<f32>(classes as usize, &lp, &y);
        assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
        let rows: Vec<f32> = (y.iter().enumerate())
            .map(|(i, &l)| -lp[i * classes as usize + l as usize])
            .collect();
        let host = tree(&rows) / batch as f32;
        assert_eq!(
            loss.to_bits(),
            host.to_bits(),
            "the loss against the host's tree, {what}"
        );
        let (want, _) = tape_nll::<f64>(classes as usize, &lp, &y);
        let total: f64 = rows.iter().map(|&v| f64::from(v).abs()).sum();
        let b = f64::from(batch);
        let bound = second(depth(batch as usize) * U * total / b + 5.0 * U * want.abs());
        let worst = assert_within(&[loss], &[want], |_| bound, &format!("the loss, {what}"));
        println!("nll loss {what}: worst {:.3} of its bound", worst.ratio());
        let (again, again_grad) = cuda_nll(&cuda, &nll, &lp, &y);
        assert_eq!(
            again.to_bits(),
            loss.to_bits(),
            "a second run's loss, {what}"
        );
        assert_exact(
            &again_grad,
            &grad,
            &format!("a second run's gradient, {what}"),
        );
    }
}

/// NaN and ±∞ as on the CPU: a NaN anywhere in a row gives a NaN loss and the CPU's finite
/// gradient; −∞ at the label gives a +∞ loss; ±∞ off the label leaves the loss finite (the
/// CPU's, bit for bit, here where every partial sum is exact). A label not below the classes,
/// which the CPU panics on, gives a NaN loss and a NaN gradient row, the other rows' gradients
/// unchanged.
#[test]
fn nll_non_finite_rows() {
    let Some(cuda) = Cuda::new() else { return };
    let nll = NegativeLogLikelihood::new(&cuda, (4, 5)).expect("nll");
    // Quarters, so the CPU's sum and the card's tree are exact.
    let lp: Vec<f32> = (0..20).map(|i| -0.25 * (i % 7) as f32).collect();
    let y = vec![0u32, 4, 2, 1];
    let (clean, clean_grad) = cuda_nll(&cuda, &nll, &lp, &y);
    assert_exact(&[clean], &[tape_nll::<f32>(5, &lp, &y).0], "the clean loss");
    for (name, at, value) in [
        ("a NaN off the label", 7, f32::NAN),
        ("a NaN at the label", 9, f32::NAN),
        ("-inf at the label", 9, f32::NEG_INFINITY),
        ("+inf off the label", 5, f32::INFINITY),
        ("-inf off the label", 8, f32::NEG_INFINITY),
    ] {
        let mut bad = lp.clone();
        bad[at] = value;
        let (loss, grad) = cuda_nll(&cuda, &nll, &bad, &y);
        let (want, want_grad) = tape_nll::<f32>(5, &bad, &y);
        assert_exact(&[loss], &[want], &format!("{name}: the loss"));
        assert_exact(&grad, &want_grad, &format!("{name}: the gradient"));
    }
    let mut bad_label = y.clone();
    bad_label[1] = 5;
    let (loss, grad) = cuda_nll(&cuda, &nll, &lp, &bad_label);
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
}

/// Each regression loss the tests run, with its term error `c` in units of `u` against the
/// exact loss of the same `f32` values, as `cavitation-gpu`'s `tests/loss.rs` derives it: 1
/// for L1, 4 for Huber and 8 for smooth L1 (the WGSL division's 2.5 ulp; NVRTC's is
/// correctly rounded, so 4 would do here).
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

/// The card's regression loss: the loss and the gradient.
fn cuda_regression(
    cuda: &Cuda,
    loss: &RegressionLoss<'_>,
    pred: &[f32],
    target: &[f32],
) -> (f32, Vec<f32>) {
    let (p, t) = (upload(cuda, pred), upload(cuda, target));
    let mut grad = nans(cuda, pred.len());
    let mut out = loss_buffer(cuda);
    loss.run(&p, &t, &mut grad, (&mut out, 1)).expect("run");
    (read_loss(&out), read(&grad, pred.len()))
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

/// Each value's loss as the CPU's tape computes it in `f32`.
fn regression_terms(kind: Regression, pred: &[f32], target: &[f32]) -> Vec<f32> {
    let term = |p: f32, t: f32| {
        let d = p - t;
        let a = d.abs();
        match kind {
            Regression::L1 => a,
            Regression::Huber { delta } if a < delta => 0.5 * d * d,
            Regression::Huber { delta } => delta * (a - 0.5 * delta),
            Regression::SmoothL1 { beta } if a < beta => 0.5 * d * d / beta,
            Regression::SmoothL1 { beta } => a - 0.5 * beta,
        }
    };
    pred.iter().zip(target).map(|(&p, &t)| term(p, t)).collect()
}

/// Each loss of `REGRESSIONS` against the tape, for 1, 37, 4,097 and 100,000 values with
/// predictions and targets in `[−2, 2)` (every branch taken), and 2²⁴ + 1,000 values with L1
/// and smooth L1: the gradient bit for bit against an `f32` store's, smooth L1's quadratic
/// branch included; the loss the bits of the fixed tree evaluated on the host over the
/// `f32` tape's terms divided by `n`, and within `(22 + chunks + c)·u·Σ ℓ / n + 5u·loss` of an
/// `f64` store's. A second run gives the same bits.
#[test]
fn regression_twin() {
    let Some(cuda) = Cuda::new() else { return };
    for (k, (kind, c)) in REGRESSIONS.into_iter().enumerate() {
        let sizes: &[usize] = if k == 0 || k == 3 {
            &[1, 37, 4097, 100_000, (1 << 24) + 1000]
        } else {
            &[1, 37, 4097, 100_000]
        };
        for (s, &n) in sizes.iter().enumerate() {
            let loss = RegressionLoss::new(&cuda, n as u32, kind).expect("regression loss");
            assert_eq!((loss.len() as usize, loss.kind()), (n, kind));
            let what = format!("{kind:?} of {n} values");
            let seed = 100 * k as u64 + s as u64;
            let pred = seeded_values(seed + 1000, n, -2.0, 2.0);
            let target = seeded_values(seed + 2000, n, -2.0, 2.0);
            let (value, grad) = cuda_regression(&cuda, &loss, &pred, &target);
            let (_, want_grad) = tape_regression::<f32>(kind, &pred, &target);
            assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
            let host = tree(&regression_terms(kind, &pred, &target)) / n as f32;
            assert_eq!(
                value.to_bits(),
                host.to_bits(),
                "the loss against the host's tree, {what}"
            );
            let (want, _) = tape_regression::<f64>(kind, &pred, &target);
            let total = want * n as f64;
            let bound = second((depth(n) + c) * U * total / n as f64 + 5.0 * U * want);
            let worst = assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
            println!("{what}: loss worst {:.3} of its bound", worst.ratio());
            let (again, again_grad) = cuda_regression(&cuda, &loss, &pred, &target);
            assert_eq!(
                again.to_bits(),
                value.to_bits(),
                "a second run's loss, {what}"
            );
            assert_exact(
                &again_grad,
                &grad,
                &format!("a second run's gradient, {what}"),
            );
        }
    }
}

/// Each value's loss and gradient on its own (`n = 1`, so the loss is the value's term) at
/// the edges, the `f32` tape's bits, NaN where it is NaN: `d` of ±0; `|d|` exactly at the
/// threshold (the linear branch) and one ulp either side; smooth L1 with `beta` 0 and +∞;
/// Huber with `delta` +∞; NaN in either input, two infinities of one sign and of opposite
/// signs, one infinite input, a finite difference that overflows; and differences near the
/// smallest normal, whose differences and gradients are subnormal (nothing is flushed).
#[test]
fn regression_edges_match_the_tape() {
    let Some(cuda) = Cuda::new() else { return };
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let above = |x: f32| f32::from_bits(x.to_bits() + 1);
    let inf = f32::INFINITY;
    let tiny = f32::MIN_POSITIVE;
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
        (1.5 * tiny, tiny),
        (-0.75 * tiny, 0.0),
    ];
    let kinds = [
        Regression::L1,
        Regression::Huber { delta: 0.5 },
        Regression::Huber { delta: inf },
        Regression::SmoothL1 { beta: 0.5 },
        Regression::SmoothL1 { beta: 0.0 },
        Regression::SmoothL1 { beta: inf },
    ];
    for kind in kinds {
        let loss = RegressionLoss::new(&cuda, 1, kind).expect("regression loss");
        for &(p, t) in &pairs {
            let what = format!("{kind:?} at p = {p:e}, t = {t:e}");
            let (value, grad) = cuda_regression(&cuda, &loss, &[p], &[t]);
            let (want, want_grad) = tape_regression::<f32>(kind, &[p], &[t]);
            assert_exact(&[value], &[want], &format!("the loss, {what}"));
            assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
        }
    }
}

/// The refusals of the negative log-likelihood and the regression losses: the CPU tape's
/// thresholds (Huber's `delta` not above 0 or NaN, smooth L1's `beta` below 0 or NaN, with the
/// tape's message) and shapes with a dimension of 0 are `CudaError::Invalid`; a buffer too
/// short panics before any launch, naming it.
#[test]
fn nll_and_regression_refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let invalid = |r: Result<(), CudaError>, why: &str| match r {
        Err(CudaError::Invalid(text)) => assert!(text.contains(why), "{text}"),
        other => panic!("accepted: {other:?}"),
    };
    let nll = |shape| NegativeLogLikelihood::new(&cuda, shape).map(|_| ());
    invalid(nll((0, 4)), "above 0");
    invalid(nll((4, 0)), "above 0");
    invalid(nll((1 << 20, 1 << 12)), "more than the 2147483648");
    let regression = |n, kind| RegressionLoss::new(&cuda, n, kind).map(|_| ());
    invalid(regression(0, Regression::L1), "above 0");
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
        invalid(regression(4, kind), text);
    }

    let nll = NegativeLogLikelihood::new(&cuda, (4, 4)).expect("nll");
    let (z, l) = (nans(&cuda, 16), upload(&cuda, &[0u32; 16]));
    let one = nans(&cuda, 1);
    let mut short_grad = nans(&cuda, 15);
    let mut grad = nans(&cuda, 16);
    let mut loss = nans(&cuda, 16);
    let m = panic_message(|| {
        let _ = nll.run(&one, &l, &mut grad, (&mut loss, 0));
    });
    assert!(
        m.contains("the log-probabilities buffer holds 4 bytes, and 16 values need 64"),
        "{m}"
    );
    let m = panic_message(|| {
        let _ = nll.run(&z, &one, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the labels buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = nll.run(&z, &l, &mut short_grad, (&mut loss, 0));
    });
    assert!(m.contains("the gradient buffer holds 60 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = nll.run(&z, &l, &mut grad, (&mut loss, 16));
    });
    assert!(m.contains("the loss output buffer holds 64 bytes"), "{m}");

    let huber = RegressionLoss::new(&cuda, 16, Regression::Huber { delta: 1.0 }).expect("huber");
    let m = panic_message(|| {
        let _ = huber.run(&one, &z, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the prediction buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = huber.run(&z, &one, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the target buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = huber.run(&z, &z, &mut short_grad, (&mut loss, 0));
    });
    assert!(m.contains("the gradient buffer holds 60 bytes"), "{m}");
    assert!(
        read(&grad, 16).iter().all(|v| v.is_nan()),
        "a refused call wrote the gradient"
    );
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

/// The card's binary cross-entropy: the loss, the clamp count and the gradient, the loss and
/// the count written at words 1 and 2 (NaN before) between two sentinels.
fn cuda_bce(
    cuda: &Cuda,
    bce: &BinaryCrossEntropy<'_>,
    pred: &[f32],
    target: &[f32],
) -> (f32, f32, Vec<f32>) {
    let (p, t) = (upload(cuda, pred), upload(cuda, target));
    let mut grad = nans(cuda, pred.len());
    let mut out = upload(cuda, &[SENTINEL, f32::NAN, f32::NAN, SENTINEL]);
    bce.run(&p, &t, &mut grad, (&mut out, 1)).expect("run");
    let words = read(&out, 4);
    assert_eq!(words[0], SENTINEL, "the word before the loss changed");
    assert_eq!(words[3], SENTINEL, "the word after the count changed");
    (words[1], words[2], read(&grad, pred.len()))
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

/// `2·atanh(y)` from its series to `y¹⁵`, as the kernel computes it.
fn atanh2(y: f32) -> f32 {
    let q = y * y;
    let mut t = 1.0f32 / 15.0;
    for c in [13.0f32, 11.0, 9.0, 7.0, 5.0, 3.0] {
        t = 1.0 / c + q * t;
    }
    t = 1.0 + q * t;
    2.0 * y * t
}

/// `ln x` of a positive finite `x`, as the kernel computes it from the bits.
fn ln_bits(x: f32) -> f32 {
    let b = x.to_bits();
    let field = b >> 23;
    let (m, mut e) = if field == 0 {
        let s = b.leading_zeros() - 8;
        ((b << s) & 0x7f_ffff, -126 - s as i32)
    } else {
        (b & 0x7f_ffff, field as i32 - 127)
    };
    let mut f = f32::from_bits(0x3f80_0000 | m);
    if f > std::f32::consts::SQRT_2 {
        f *= 0.5;
        e += 1;
    }
    let k = e as f32;
    let ln2_lo = 1.428_606_820_309_417_2e-6_f64 as f32;
    k * f32::from_bits(0x3f31_7200) + (k * ln2_lo + atanh2((f - 1.0) / (f + 1.0)))
}

/// A probability's loss as the kernel computes it, for `p` within `[−0, 1]` and a finite `t`:
/// nothing is fused and every operation is correctly rounded on the card, so these are its
/// bits.
fn probability_term(p: f32, t: f32) -> f32 {
    let lp = if clamps(p) { -100.0 } else { ln_bits(p) };
    let lq = if clamps(1.0 - p) {
        -100.0
    } else {
        ln_bits(1.0 - p)
    };
    -(t * lp + (1.0 - t) * lq)
}

/// One value's exact loss and its stated bound, from the tape's expression on the same `f32`
/// values: the WGSL kernel's, derived in `cavitation-gpu`'s `tests/loss.rs`. Probabilities:
/// `−(t·ln p + c·ln q)` with `q = fl(1 − p)` and `c = fl(1 − t)`, within `15u` of
/// `|t·ln p| + |c·ln q|`. Logits: `a + L` with `a = max(z, 0) − z·t`, `L = ln(1 + e^(−|z|))`,
/// within `u·(|z·t| + |a| + |ℓ|) + 10u·L + (6 + 4|z|)·u·e`. Both add `2⁻¹²⁶·(2 + |t|)`.
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

/// A logit's exact gradient `g = (σ(z) − t)·w` and the WGSL kernel's bound
/// `w·(Δσ + u·|σ − t|) + u·|g|`, `Δσ` being `13u·σ` for `z ≥ 0`, `(18 + 8|z|)·u·σ` below
/// and 0 for `z = ±∞`.
fn logit_grad(z: f32, t: f32, w: f32) -> (f64, f64) {
    let (z, t, w) = (f64::from(z), f64::from(t), f64::from(w));
    let sigma = if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        z.exp() / (1.0 + z.exp())
    };
    let d_sigma = if z.is_infinite() {
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

/// The gradient of `form` against the tape: probabilities the `f32` tape's bits everywhere;
/// logits the tape's bits where it is not finite, else within [`logit_grad`]'s bound of the
/// exact gradient. Returns the worst share of a bound.
fn check_bce_grad(
    form: Form,
    (pred, target): (&[f32], &[f32]),
    grad: &[f32],
    want: &[f32],
    what: &str,
) -> f64 {
    if form.0 == BinaryInput::Probabilities {
        assert_exact(grad, want, what);
        return 0.0;
    }
    let w = bce_scale(form, pred.len());
    let pick = |v: &[f32], keep: bool| -> Vec<f32> {
        (0..v.len())
            .filter(|&i| !want[i].is_finite() == keep)
            .map(|i| v[i])
            .collect()
    };
    assert_exact(&pick(grad, true), &pick(want, true), what);
    let rest: Vec<usize> = (0..grad.len()).filter(|&i| want[i].is_finite()).collect();
    let got: Vec<f32> = rest.iter().map(|&i| grad[i]).collect();
    let (reference, bounds): (Vec<f64>, Vec<f64>) = rest
        .iter()
        .map(|&i| logit_grad(pred[i], target[i], w))
        .unzip();
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
/// 2²⁴ + 1,000): the gradient as [`check_bce_grad`] checks it; for probabilities, the loss
/// the bits of the fixed tree evaluated on the host over the kernel's terms (divided by `n`
/// for the mean); the loss within [`bce_loss`]'s bound of the exact loss; the clamp count the
/// tape's; a second run's bits the first's.
#[test]
fn binary_cross_entropy_twin() {
    let Some(cuda) = Cuda::new() else { return };
    for (k, form) in FORMS.into_iter().enumerate() {
        let sizes: &[usize] = if k == 0 {
            &[1, 37, 4097, 100_000, (1 << 24) + 1000]
        } else {
            &[1, 37, 4097, 100_000]
        };
        for (s, &n) in sizes.iter().enumerate() {
            let bce = BinaryCrossEntropy::new(&cuda, n as u32, form).expect("binary loss");
            assert_eq!((bce.len() as usize, bce.form()), (n, form));
            let what = format!("{form:?} of {n} values");
            let seed = 300 + 10 * k as u64 + s as u64;
            let pred = match form.0 {
                BinaryInput::Probabilities => probabilities(seed, n),
                BinaryInput::Logits => logits(seed, n),
            };
            let target = targets(seed + 1000, n);
            let (value, count, grad) = cuda_bce(&cuda, &bce, &pred, &target);
            let (_, want_grad, clamped) = tape_bce::<f32>(form, &pred, &target);
            let ratio = check_bce_grad(
                form,
                (&pred, &target),
                &grad,
                &want_grad,
                &format!("the gradient, {what}"),
            );
            assert_eq!(count, clamped as f32, "the clamp count, {what}");
            if form.0 == BinaryInput::Probabilities {
                let terms: Vec<f32> = (pred.iter().zip(&target))
                    .map(|(&p, &t)| probability_term(p, t))
                    .collect();
                let host = match divisor(form.1, n) {
                    Some(d) => tree(&terms) / d as f32,
                    None => tree(&terms),
                };
                assert_eq!(
                    value.to_bits(),
                    host.to_bits(),
                    "the loss against the host's tree, {what}"
                );
            }
            let (want, bound) = bce_loss(form, &pred, &target);
            let worst = assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
            println!(
                "{what}: loss worst {:.3} of its bound, gradient {ratio:.3}, {clamped} clamped",
                worst.ratio()
            );
            let (again, again_count, again_grad) = cuda_bce(&cuda, &bce, &pred, &target);
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
}

/// Each value on its own (`n = 1`) at the edges, against the tape, as `cavitation-gpu`'s
/// edge test checks the WGSL kernel: where the `f32` tape's loss is NaN or infinite the
/// card's is the same, elsewhere within [`bce_term`]'s bound of the exact loss and, for
/// probabilities, the kernel's term evaluated on the host bit for bit; the gradient as
/// [`check_bce_grad`] checks it (probabilities the tape's bits, subnormals included); the
/// clamp count the tape's. The values are the WGSL test's: 0, 1 and the clamp's threshold,
/// subnormals, 2⁻⁶² and one ulp either side, values outside `[0, 1]` (±1e20 and ±2⁶⁴ and one ulp
/// below it among them, where the CPU's `p·(1 − p)` starts to overflow), NaN and ±∞; logits
/// of ±0 to ±1e30, NaN and ±∞.
#[test]
fn binary_cross_entropy_edges_match_the_tape() {
    let Some(cuda) = Cuda::new() else { return };
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
    for (input, preds, wanted) in [
        (
            BinaryInput::Probabilities,
            &probabilities[..],
            &probability_targets[..],
        ),
        (BinaryInput::Logits, &logits[..], &logit_targets[..]),
    ] {
        let form = (input, LossReduction::Mean);
        let bce = BinaryCrossEntropy::new(&cuda, 1, form).expect("binary loss");
        for &p in preds {
            for &t in wanted {
                let what = format!("{input:?} at {p:e}, target {t:e}");
                let (value, count, grad) = cuda_bce(&cuda, &bce, &[p], &[t]);
                let (want, want_grad, clamped) = tape_bce::<f32>(form, &[p], &[t]);
                if want.is_finite() {
                    let (exact, bound) = bce_term(input, p, t);
                    assert_within(&[value], &[exact], |_| second(bound), &what);
                    if input == BinaryInput::Probabilities {
                        let host = tree(&[probability_term(p, t)]);
                        assert_exact(&[value], &[host], &format!("the host's term, {what}"));
                    }
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
}

/// A NaN or out-of-range value in a batch of 8,192: a NaN probability, a NaN target, a
/// probability of 1.5, an infinite target and an infinite logit give a NaN or infinite loss
/// and leave every other value's gradient the clean run's bits.
#[test]
fn binary_cross_entropy_non_finite_reaches_the_loss() {
    let Some(cuda) = Cuda::new() else { return };
    let n = 8192;
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
        let bce = BinaryCrossEntropy::new(&cuda, n as u32, form).expect("binary loss");
        let pred = match input {
            BinaryInput::Probabilities => probabilities(2, n),
            BinaryInput::Logits => logits(2, n),
        };
        let (_, _, clean) = cuda_bce(&cuda, &bce, &pred, &target);
        let (mut p, mut t) = (pred.clone(), target.clone());
        if at_pred {
            p[5000] = value;
        } else {
            t[5000] = value;
        }
        let (loss, _, grad) = cuda_bce(&cuda, &bce, &p, &t);
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
    }
}

/// The documented refusals of binary cross-entropy: no values or more than 2³¹, and buffers
/// too short (the output needs the loss's word and the count's), with nothing written.
#[test]
fn binary_cross_entropy_refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let form = (BinaryInput::Probabilities, LossReduction::Mean);
    for n in [0, u32::MAX] {
        match BinaryCrossEntropy::new(&cuda, n, form) {
            Err(CudaError::Invalid(_)) => {}
            other => panic!("{n} values accepted: {other:?}"),
        }
    }
    let bce = BinaryCrossEntropy::new(&cuda, 16, form).expect("binary loss");
    let (p, t) = (nans(&cuda, 16), nans(&cuda, 16));
    let one = nans(&cuda, 1);
    let mut short_grad = nans(&cuda, 15);
    let mut grad = nans(&cuda, 16);
    let mut loss = nans(&cuda, 16);
    let mut short_loss = nans(&cuda, 1);
    let m = panic_message(|| {
        let _ = bce.run(&one, &t, &mut grad, (&mut loss, 0));
    });
    assert!(
        m.contains("the prediction buffer holds 4 bytes, and 16 values need 64"),
        "{m}"
    );
    let m = panic_message(|| {
        let _ = bce.run(&p, &one, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the target buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = bce.run(&p, &t, &mut short_grad, (&mut loss, 0));
    });
    assert!(m.contains("the gradient buffer holds 60 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = bce.run(&p, &t, &mut grad, (&mut short_loss, 0));
    });
    assert!(
        m.contains("the loss output buffer holds 4 bytes, and 2 values need 8"),
        "{m}"
    );
    let m = panic_message(|| {
        let _ = bce.run(&p, &t, &mut grad, (&mut loss, 15));
    });
    assert!(m.contains("the loss output buffer holds 64 bytes"), "{m}");
    assert!(
        read(&grad, 16).iter().all(|v| v.is_nan()),
        "a refused call wrote the gradient"
    );
}

/// Binary cross-entropy's batch mean on 37 rows of 12 values, of probabilities and of
/// logits, against the tape's `Reduction::BatchMean` on `[37, 12]`: the gradient as
/// [`check_bce_grad`] checks it with the scale `1/37`, the probabilities' loss the host's
/// tree over the kernel's terms divided by 37, every loss within [`bce_loss`]'s bound, the
/// clamp count the tape's.
#[test]
fn binary_cross_entropy_batch_mean() {
    let Some(cuda) = Cuda::new() else { return };
    let (batch, width) = (37usize, 12usize);
    let n = batch * width;
    for input in [BinaryInput::Probabilities, BinaryInput::Logits] {
        let form = (
            input,
            LossReduction::BatchMean {
                batch: batch as u32,
            },
        );
        let bce = BinaryCrossEntropy::new(&cuda, n as u32, form).expect("binary loss");
        let what = format!("{form:?} of {n} values");
        let pred = match input {
            BinaryInput::Probabilities => probabilities(41, n),
            BinaryInput::Logits => logits(41, n),
        };
        let target = targets(42, n);
        let (value, count, grad) = cuda_bce(&cuda, &bce, &pred, &target);
        let (_, want_grad, clamped) = tape_bce::<f32>(form, &pred, &target);
        check_bce_grad(
            form,
            (&pred, &target),
            &grad,
            &want_grad,
            &format!("the gradient, {what}"),
        );
        assert_eq!(count, clamped as f32, "the clamp count, {what}");
        if input == BinaryInput::Probabilities {
            let terms: Vec<f32> = (pred.iter().zip(&target))
                .map(|(&p, &t)| probability_term(p, t))
                .collect();
            let host = tree(&terms) / batch as f32;
            assert_exact(&[value], &[host], &format!("the host's tree, {what}"));
        }
        let (want, bound) = bce_loss(form, &pred, &target);
        assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
    }
    for batch in [0, 5] {
        let form = (BinaryInput::Logits, LossReduction::BatchMean { batch });
        match BinaryCrossEntropy::new(&cuda, 12, form) {
            Err(CudaError::Invalid(m)) => assert!(m.contains("does not divide 12 values"), "{m}"),
            other => panic!("a batch of {batch} accepted: {other:?}"),
        }
    }
}

/// The card's KL divergence: the loss, written at word 1 (NaN before) between two
/// sentinels, and the log-probabilities' gradient.
fn cuda_kl(
    cuda: &Cuda,
    kl: &KlDivergence<'_>,
    log_probs: &[f32],
    target: &[f32],
) -> (f32, Vec<f32>) {
    let (x, t) = (upload(cuda, log_probs), upload(cuda, target));
    let mut grad = nans(cuda, log_probs.len());
    let mut out = upload(cuda, &[SENTINEL, f32::NAN, SENTINEL]);
    kl.run(&x, &t, &mut grad, (&mut out, 1)).expect("run");
    let words = read(&out, 3);
    assert_eq!(words[0], SENTINEL, "the word before the loss changed");
    assert_eq!(words[2], SENTINEL, "the word after the loss changed");
    (words[1], read(&grad, log_probs.len()))
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

/// A value's KL divergence as the kernel computes it, for finite `x` and `t` with `t` not
/// negative: `ln t` from the bits and a subnormal target's product on `t·2³²` scaled back
/// by 2⁻³². Nothing is fused and every operation is correctly rounded on the card, so these
/// are its bits.
fn kl_host_term(x: f32, t: f32) -> f32 {
    if t == 0.0 {
        return 0.0;
    }
    let d = ln_bits(t) - x;
    if t.is_subnormal() {
        let scaled = (t.to_bits() & 0x7f_ffff) as f32 * f32::from_bits(10 << 23);
        scaled * d * f32::from_bits(95 << 23)
    } else {
        t * d
    }
}

/// One value's exact loss `ℓ = t·(ln t − x)` (0 where `t` is ±0) and the WGSL kernel's
/// bound `15u·|t·ln t| + 3u·|ℓ| + 2⁻¹²⁶·(1 + |t|)`, derived in `cavitation-gpu`'s
/// `tests/loss.rs`, for finite values with `t` not negative.
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
/// on 2²⁴ + 1,000 values over 8 rows), as `cavitation-gpu`'s twin test checks the WGSL
/// kernel: the gradient the `f32` tape's bits, subnormals included; the loss the bits of the
/// fixed tree evaluated on the host over the kernel's terms (divided by the divisor for a
/// mean) and within [`kl_loss`]'s bound of the exact loss; a second run's bits the first's.
#[test]
fn kl_div_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let mut runs: Vec<(usize, LossReduction)> = Vec::new();
    for n in [1, 37, 4097, 100_000] {
        runs.extend(kl_reductions(n).into_iter().map(|r| (n, r)));
    }
    runs.push(((1 << 24) + 1000, LossReduction::BatchMean { batch: 8 }));
    for (k, &(n, reduction)) in runs.iter().enumerate() {
        let kl = KlDivergence::new(&cuda, n as u32, reduction).expect("KL divergence");
        assert_eq!((kl.len() as usize, kl.reduction()), (n, reduction));
        let what = format!("{reduction:?} of {n} values");
        let x = kl_inputs(500 + k as u64, n);
        let t = kl_targets(700 + k as u64, n);
        let (value, grad) = cuda_kl(&cuda, &kl, &x, &t);
        let (_, want_grad) = tape_kl::<f32>(reduction, &x, &t);
        assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
        let terms: Vec<f32> = (x.iter().zip(&t))
            .map(|(&x, &t)| kl_host_term(x, t))
            .collect();
        let host = match divisor(reduction, n) {
            Some(d) => tree(&terms) / d as f32,
            None => tree(&terms),
        };
        assert_exact(&[value], &[host], &format!("the host's tree, {what}"));
        let (want, bound) = kl_loss(reduction, &x, &t);
        let worst = assert_within(&[value], &[want], |_| bound, &format!("the loss, {what}"));
        println!("{what}: loss worst {:.3} of its bound", worst.ratio());
        let (again, again_grad) = cuda_kl(&cuda, &kl, &x, &t);
        assert_exact(&[again], &[value], &format!("a second run, {what}"));
        assert_exact(
            &again_grad,
            &grad,
            &format!("a second run's gradient, {what}"),
        );
    }
}

/// Each value on its own (`n = 1`) at the edges, against the tape, with the WGSL test's
/// values: where the `f32` tape's loss is NaN or infinite the card's is the same, elsewhere
/// the kernel's term evaluated on the host bit for bit and within [`kl_term`]'s bound of the
/// exact loss; the gradient the `f32` tape's bits, subnormals included.
#[test]
fn kl_div_edges_match_the_tape() {
    let Some(cuda) = Cuda::new() else { return };
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
    let mut worst: f64 = 0.0;
    for reduction in [
        LossReduction::Mean,
        LossReduction::Sum,
        LossReduction::BatchMean { batch: 1 },
    ] {
        let kl = KlDivergence::new(&cuda, 1, reduction).expect("KL divergence");
        for &x in &inputs {
            for &t in &targets {
                let what = format!("{reduction:?} at {x:e}, target {t:e}");
                let (value, grad) = cuda_kl(&cuda, &kl, &[x], &[t]);
                let (want, want_grad) = tape_kl::<f32>(reduction, &[x], &[t]);
                if want.is_finite() {
                    let (exact, bound) = kl_term(x, t);
                    let w = assert_within(&[value], &[exact], |_| second(bound), &what);
                    worst = worst.max(w.ratio());
                    let host = tree(&[kl_host_term(x, t)]);
                    assert_exact(&[value], &[host], &format!("the host's term, {what}"));
                } else {
                    assert_exact(&[value], &[want], &format!("the loss, {what}"));
                }
                assert_exact(&grad, &want_grad, &format!("the gradient, {what}"));
            }
        }
    }
    println!("edges: loss worst {worst:.3} of its bound");
}

/// A NaN or out-of-range value in a batch of 8,192: a NaN log-probability (where the target
/// is 0 too), a NaN target, a negative target, an infinite target and a log-probability of
/// +∞ give a NaN or infinite loss and leave every other value's gradient the clean run's
/// bits.
#[test]
fn kl_div_non_finite_reaches_the_loss() {
    let Some(cuda) = Cuda::new() else { return };
    let n = 8192;
    let kl = KlDivergence::new(&cuda, n as u32, LossReduction::BatchMean { batch: 64 })
        .expect("KL divergence");
    let x = kl_inputs(3, n);
    let t = kl_targets(4, n);
    // Value 5,001 has a target of 0.
    assert_eq!(t[5001], 0.0);
    let (_, clean) = cuda_kl(&cuda, &kl, &x, &t);
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
        let (loss, grad) = cuda_kl(&cuda, &kl, &xs, &ts);
        assert!(!loss.is_finite(), "{name}: the loss is {loss}");
        let rest: Vec<f32> = grad[..at].iter().chain(&grad[at + 1..]).copied().collect();
        let want: Vec<f32> = clean[..at]
            .iter()
            .chain(&clean[at + 1..])
            .copied()
            .collect();
        assert_exact(&rest, &want, &format!("{name}: the other values"));
    }
}

/// The documented refusals of KL divergence: no values or more than 2³¹, a batch mean whose
/// batch is 0 or does not divide the values, and buffers too short, with nothing written.
#[test]
fn kl_div_refusals() {
    let Some(cuda) = Cuda::new() else { return };
    for n in [0, u32::MAX] {
        match KlDivergence::new(&cuda, n, LossReduction::Mean) {
            Err(CudaError::Invalid(_)) => {}
            other => panic!("{n} values accepted: {other:?}"),
        }
    }
    for batch in [0, 3, 32] {
        match KlDivergence::new(&cuda, 16, LossReduction::BatchMean { batch }) {
            Err(CudaError::Invalid(m)) => assert!(
                m.contains(&format!(
                    "a batch of {batch} rows does not divide 16 values"
                )),
                "{m}"
            ),
            other => panic!("a batch of {batch} accepted: {other:?}"),
        }
    }
    let kl = KlDivergence::new(&cuda, 16, LossReduction::Mean).expect("KL divergence");
    let (x, t) = (nans(&cuda, 16), nans(&cuda, 16));
    let one = nans(&cuda, 1);
    let mut short_grad = nans(&cuda, 15);
    let mut grad = nans(&cuda, 16);
    let mut loss = nans(&cuda, 16);
    let m = panic_message(|| {
        let _ = kl.run(&one, &t, &mut grad, (&mut loss, 0));
    });
    assert!(
        m.contains("the log-probabilities buffer holds 4 bytes, and 16 values need 64"),
        "{m}"
    );
    let m = panic_message(|| {
        let _ = kl.run(&x, &one, &mut grad, (&mut loss, 0));
    });
    assert!(m.contains("the target buffer holds 4 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = kl.run(&x, &t, &mut short_grad, (&mut loss, 0));
    });
    assert!(m.contains("the gradient buffer holds 60 bytes"), "{m}");
    let m = panic_message(|| {
        let _ = kl.run(&x, &t, &mut grad, (&mut loss, 16));
    });
    assert!(
        m.contains("the loss output buffer holds 64 bytes, and 17 values need 68"),
        "{m}"
    );
    assert!(
        read(&grad, 16).iter().all(|v| v.is_nan()),
        "a refused call wrote the gradient"
    );
}
