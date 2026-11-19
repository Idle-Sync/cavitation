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
//!
//! Every output starts as NaN, every loss is written at word 1 between two sentinels, and two
//! runs give the same bits. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1`
//! makes that a failure.

mod common;

use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
use cavitation_cuda::{Cuda, CudaError, DeviceBuffer, MeanSquaredError, SoftmaxCrossEntropy};
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
