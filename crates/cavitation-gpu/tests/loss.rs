//! The losses against the tape's, on every backend this machine has.
//!
//! - **Softmax cross-entropy:** the loss and the logits' gradient within their stated bounds
//!   of `Tape::softmax_cross_entropy` on an `f64` store, from one row to 70,000 rows and from
//!   one class to 1,000. A NaN or infinite logit, or a bad label, gives a NaN row that the
//!   finite flag catches.
//! - **Mean squared error:** the gradient bit for bit against `Tape::mse` on an `f32` store,
//!   and the loss within its bound of the `f64` tape. Both are checked up to more values than
//!   one grid row of workgroups holds, and an overflowed square and a NaN reach the loss.
//!
//! Every loss is written at word 1 of a buffer of sentinels, to show nothing else changes. The
//! tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
use cavitation_gpu::kernels::{FiniteFlag, MeanSquaredError, SoftmaxCrossEntropy};
use cavitation_gpu::{Gpu, PooledBuffer};
use common::{FLOOR, Kit, U, assert_exact, assert_within, for_each_gpu, seeded_values, sum_bound};

/// What fills the words around a loss.
const SENTINEL: f32 = 7.0;

/// A kit whose steps can move `values` values.
fn kit(gpu: &Gpu, values: usize) -> Kit {
    Kit::new(gpu, 4 * values as u64 + 65_536)
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
        let s: f64 = x.iter().map(|v| v.exp()).sum();
        let d_s: f64 = x
            .iter()
            .map(|v| (6.0 + 5.0 * v.abs()) * U * v.exp() + FLOOR)
            .sum::<f64>()
            + sum_bound(classes, s);
        let lse = s.ln();
        let d = second(d_s / s) + 2f64.powi(-21) + 6.0 * U * lse.abs();
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
            grad.push(second(p * inner / b + 2.0 * U * (p - onehot).abs() / b) + FLOOR / b);
        }
    }
    let mean = losses / b;
    let loss = second((row_bounds + sum_bound(batch, losses)) / b + 5.0 * U * mean)
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
            ("a -inf logit off the label", 9, f32::NEG_INFINITY, 4),
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
/// store's, and the loss within `((2(n + 2) + 3)·u·Σ d²)/n + 5u·loss` of an `f64` store's, for
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
            let bound =
                second((sum_bound(n, squares) + 3.0 * U * squares) / n as f64 + 5.0 * U * want)
                    + 3.0 * FLOOR;
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
