//! Softmax and log-softmax over each row on the card against the tape's, as
//! `cavitation-gpu`'s tests check its WGSL kernels:
//!
//! - **Forward:** each output within the WGSL kernels' stated bound (`cavitation-gpu`'s
//!   `RowSoftmaxKernels`) of `Tape::softmax` and `Tape::log_softmax` on an `f64` store, from
//!   the same `f32` inputs, from one row to 70,000 rows and from one value per row to 5,000,
//!   over `[−5, 5)` and `[−40, 40)`. A row of one value gives exactly 1 and exactly 0.
//! - **Backward:** from the `f32` tape's forward output and a seeded gradient, the softmax's
//!   backward pass the bits of the `f32` tape's gradient (the same operations in the same
//!   order, without fused multiply-add); the log-softmax's within its stated bound of the
//!   tape's formula evaluated in `f64` on the same `f32` values.
//! - **Bad rows:** a NaN or infinite value gives a NaN row, and the other rows keep the bits
//!   of a run without it.
//!
//! Every output starts as NaN, so a value left unwritten fails, and two runs give the same
//! bits. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes that a
//! failure.

mod common;

use cavitation_core::{Graph, ParamStore, Real, Tensor};
use cavitation_cuda::{Cuda, CudaError, RowSoftmax, RowSoftmaxOp};
use common::upload;
use common::{FLOOR, U, assert_exact, assert_within, nans, panic_message, read, seeded_values};

/// The shapes every twin runs: one value, CIFAR's 128 × 10, many short rows, a few long
/// ones, rows of one value, more rows than 256 blocks of threads, and very long rows.
const SHAPES: [(usize, usize); 8] = [
    (1, 1),
    (1, 10),
    (128, 10),
    (1000, 3),
    (7, 1000),
    (65, 1),
    (70_000, 4),
    (3, 5000),
];

/// The error bound of a sum of `k` terms in any order: `2(k + 2) · u · Σ|terms|`.
fn sum_bound(k: usize, abs_sum: f64) -> f64 {
    2.0 * (k as f64 + 2.0) * U * abs_sum
}

/// `b` with an allowance for rounding the result it bounds.
fn second(b: f64) -> f64 {
    b * (1.0 + 16.0 * U)
}

/// The card's `op` over `rows × width` values of `a` (and `b`), into an output of NaNs.
fn cuda_run(
    cuda: &Cuda,
    kernels: &RowSoftmax<'_>,
    op: RowSoftmaxOp,
    (rows, width): (usize, usize),
    a: &[f32],
    b: Option<&[f32]>,
) -> Vec<f32> {
    let x = upload(cuda, a);
    let g = b.map(|b| upload(cuda, b));
    let mut out = nans(cuda, a.len());
    kernels
        .run(op, &x, g.as_ref(), &mut out, rows, width)
        .unwrap();
    read(&out, a.len())
}

/// What the tape records for `x` (`rows × width`) through the softmax, or the log-softmax if
/// `log`, on a store of `T`, from the same `f32` values; with a gradient `g` for the output,
/// also the input's gradient.
fn tape<T: Real>(log: bool, width: usize, x: &[f32], g: Option<&[f32]>) -> (Vec<T>, Vec<T>) {
    let wide = |v: &[f32]| -> Vec<T> { v.iter().map(|&v| T::from_f64(f64::from(v))).collect() };
    let rows = x.len() / width;
    let mut store = ParamStore::<T>::new_generic();
    let id = store.add(Tensor::from_vec(wide(x), &[rows, width]).unwrap());
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&store.values);
    let v = tape.param(id);
    let y = if log {
        tape.log_softmax(v)
    } else {
        tape.softmax(v)
    };
    let out = tape.value(y).data().to_vec();
    let Some(g) = g else {
        return (out, Vec::new());
    };
    // Σ y · g has the gradient g for y, exactly.
    let g = tape.input(Tensor::from_vec(wide(g), &[rows, width]).unwrap().view());
    let product = tape.mul(y, g);
    let loss = tape.sum(product);
    tape.backward(loss, &mut store.grads);
    drop(tape);
    (out, store.grad(id).data().to_vec())
}

/// The stated bound of each forward output for `x` (`rows × width`): the WGSL kernels'.
fn forward_bounds(log: bool, width: usize, x: &[f32]) -> Vec<f64> {
    let mut bounds = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        let z: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let xs: Vec<f64> = z.iter().map(|v| v - m).collect();
        let rel = |v: f64| (6.0 + 5.0 * v.abs()) * U;
        if log {
            let k = xs.iter().position(|&v| v == 0.0).unwrap();
            let rest = || {
                xs.iter()
                    .enumerate()
                    .filter(move |&(j, _)| j != k)
                    .map(|(_, &v)| v)
            };
            let e: f64 = rest().map(f64::exp).sum();
            let d_e: f64 =
                rest().map(|v| rel(v) * v.exp() + FLOOR).sum::<f64>() + sum_bound(width - 1, e);
            let s = 1.0 + e;
            let lse = e.ln_1p();
            let d = if e <= 1.0 {
                d_e + 10.0 * U * lse
            } else {
                (d_e + U * s) / s + 6.0 * U * lse
            };
            for &zj in &z {
                bounds.push(second(2.0 * U * (zj.abs() + m.abs() + lse) + d));
            }
        } else {
            let s: f64 = xs.iter().map(|v| v.exp()).sum();
            let d_s: f64 =
                xs.iter().map(|&v| rel(v) * v.exp() + FLOOR).sum::<f64>() + sum_bound(width, s);
            for &v in &xs {
                let y = v.exp() / s;
                bounds.push(second(y * (rel(v) + d_s / s + 5.0 * U)) + FLOOR / s);
            }
        }
    }
    bounds
}

/// The log-softmax's backward formula evaluated in `f64` on the `f32` forward output `y` and
/// gradient `g`, `g − e^y · Σ g`, and the WGSL kernel's stated bound of each output.
fn log_backward_reference(width: usize, y: &[f32], g: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let (mut out, mut bounds) = (Vec::with_capacity(y.len()), Vec::with_capacity(y.len()));
    for (y, g) in y.chunks_exact(width).zip(g.chunks_exact(width)) {
        let y: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
        let g: Vec<f64> = g.iter().map(|&v| f64::from(v)).collect();
        let t: f64 = g.iter().sum();
        let d_t = sum_bound(width, g.iter().map(|v| v.abs()).sum());
        for (&yj, &gj) in y.iter().zip(&g) {
            let p = yj.exp();
            out.push(gj - p * t);
            let b = p * d_t + (7.0 + 4.0 * yj.abs()) * U * p * t.abs() + U * (gj - p * t).abs();
            bounds.push(second(b) + FLOOR);
        }
    }
    (out, bounds)
}

/// Both forward passes against the `f64` tape on every shape, over `[−5, 5)` and
/// `[−40, 40)`, and the same bits on a second run. Rows of one value give exactly 1 and
/// exactly 0.
#[test]
fn forward_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = RowSoftmax::new(&cuda).unwrap();
    for (s, shape) in SHAPES.into_iter().enumerate() {
        let (rows, width) = shape;
        let n = rows * width;
        for range in [5.0f32, 40.0] {
            let x = seeded_values(10 * s as u64 + 1, n, -range, range);
            for log in [false, true] {
                let (name, op) = if log {
                    ("log-softmax", RowSoftmaxOp::LogSoftmax)
                } else {
                    ("softmax", RowSoftmaxOp::Softmax)
                };
                let what = format!("{name} of {rows} × {width} within ±{range}");
                let got = cuda_run(&cuda, &kernels, op, shape, &x, None);
                let (want, _) = tape::<f64>(log, width, &x, None);
                let bounds = forward_bounds(log, width, &x);
                let worst = assert_within(&got, &want, |i| bounds[i], &what);
                println!("{what}: worst {:.3} of its bound", worst.ratio());
                let again = cuda_run(&cuda, &kernels, op, shape, &x, None);
                assert_exact(&again, &got, &format!("{what}, a second run"));
            }
        }
        if width == 1 {
            let x = seeded_values(3, n, -40.0, 40.0);
            let ones = cuda_run(&cuda, &kernels, RowSoftmaxOp::Softmax, shape, &x, None);
            assert_exact(&ones, &vec![1.0; n], "softmax of one value");
            let zeros = cuda_run(&cuda, &kernels, RowSoftmaxOp::LogSoftmax, shape, &x, None);
            assert_exact(&zeros, &vec![0.0; n], "log-softmax of one value");
        }
    }
}

/// Both backward passes from the `f32` tape's forward output and a gradient in `[−1, 1)`:
/// the softmax's the bits of the `f32` tape's gradient, the log-softmax's within its bound
/// of the formula in `f64`; the same bits on a second run.
#[test]
fn backward_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = RowSoftmax::new(&cuda).unwrap();
    for (s, shape) in SHAPES.into_iter().enumerate() {
        let (rows, width) = shape;
        let n = rows * width;
        for range in [5.0f32, 40.0] {
            let x = seeded_values(10 * s as u64 + 2, n, -range, range);
            let g = seeded_values(10 * s as u64 + 3, n, -1.0, 1.0);
            let what = format!("{rows} × {width} within ±{range}");

            let (y, dx) = tape::<f32>(false, width, &x, Some(&g));
            let op = RowSoftmaxOp::SoftmaxBackward;
            let got = cuda_run(&cuda, &kernels, op, shape, &y, Some(&g));
            assert_exact(&got, &dx, &format!("softmax backward, {what}"));

            let (y, _) = tape::<f32>(true, width, &x, None);
            let op = RowSoftmaxOp::LogSoftmaxBackward;
            let got = cuda_run(&cuda, &kernels, op, shape, &y, Some(&g));
            let (want, bounds) = log_backward_reference(width, &y, &g);
            let what = format!("log-softmax backward, {what}");
            let worst = assert_within(&got, &want, |i| bounds[i], &what);
            println!("{what}: worst {:.3} of its bound", worst.ratio());
            let again = cuda_run(&cuda, &kernels, op, shape, &y, Some(&g));
            assert_exact(&again, &got, &format!("{what}, a second run"));
        }
    }
}

/// A row with a NaN, +∞ or −∞ gives a NaN row in both forward passes; the other rows keep
/// the bits of a run without it.
#[test]
fn bad_rows_are_nan() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = RowSoftmax::new(&cuda).unwrap();
    let x = seeded_values(7, 20, -3.0, 3.0);
    for op in [RowSoftmaxOp::Softmax, RowSoftmaxOp::LogSoftmax] {
        let clean = cuda_run(&cuda, &kernels, op, (4, 5), &x, None);
        for (name, value) in [
            ("NaN", f32::NAN),
            ("+inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            let mut bad = x.clone();
            bad[7] = value;
            let got = cuda_run(&cuda, &kernels, op, (4, 5), &bad, None);
            assert!(
                got[5..10].iter().all(|v| v.is_nan()),
                "{op:?} with {name}: the bad row is {:?}",
                &got[5..10]
            );
            let rest: Vec<f32> = got[..5].iter().chain(&got[10..]).copied().collect();
            let want: Vec<f32> = clean[..5].iter().chain(&clean[10..]).copied().collect();
            assert_exact(&rest, &want, &format!("{op:?} with {name}: the other rows"));
        }
    }
}

/// A missing or unwanted gradient and a size past `u32` are refused, and a short buffer
/// panics, naming it.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = RowSoftmax::new(&cuda).unwrap();
    let a = upload(&cuda, &[0.0f32; 8]);
    let mut out = nans(&cuda, 8);
    let err = (kernels.run(RowSoftmaxOp::SoftmaxBackward, &a, None, &mut out, 2, 4)).unwrap_err();
    assert!(
        matches!(&err, CudaError::Invalid(m) if m.contains("needs")),
        "{err}"
    );
    let err = (kernels.run(RowSoftmaxOp::Softmax, &a, Some(&a), &mut out, 2, 4)).unwrap_err();
    assert!(
        matches!(&err, CudaError::Invalid(m) if m.contains("takes no")),
        "{err}"
    );
    let err =
        (kernels.run(RowSoftmaxOp::Softmax, &a, None, &mut out, 1 << 20, 1 << 12)).unwrap_err();
    assert!(
        matches!(&err, CudaError::Invalid(m) if m.contains("u32")),
        "{err}"
    );
    let text = panic_message(|| {
        let _ = kernels.run(RowSoftmaxOp::Softmax, &a, None, &mut out, 3, 4);
    });
    assert!(
        text.contains("first input") && text.contains("48"),
        "{text}"
    );
}
