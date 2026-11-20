//! Softmax and log-softmax over each row against the tape's, on every backend this machine
//! has.
//!
//! - **Forward:** each output within its stated bound (see `RowSoftmaxKernels`) of
//!   `Tape::softmax` and `Tape::log_softmax` on an `f64` store, from the same `f32` inputs,
//!   from one row to 70,000 rows and from one value per row to 5,000, over `[−5, 5)` and
//!   `[−40, 40)`. A row of one value gives exactly 1 and exactly 0.
//! - **Backward:** from the `f32` tape's forward output and a seeded gradient, each output
//!   within its stated bound of the tape's backward formula evaluated in `f64` on the same
//!   `f32` values; the softmax's backward pass bit for bit on small integers, whose sums and
//!   products are exact in any order.
//! - **Bad rows:** as on the CPU, a NaN or +∞, or −∞ everywhere, gives a NaN row, and a −∞
//!   elsewhere the `f32` tape's bits for it, forward and backward; the other rows keep the
//!   bits of a run without it.
//!
//! Every output starts as NaN, so a value left unwritten fails, and a second run gives the
//! same bits. The tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, Real, Tensor};
use cavitation_gpu::Gpu;
use cavitation_gpu::kernels::{RowSoftmax, RowSoftmaxKernels};
use common::{FLOOR, Kit, U, assert_exact, assert_within, exact_values, for_each_gpu};
use common::{seeded_values, sum_bound};

/// The shapes every twin runs: one value, CIFAR's 128 × 10, many short rows, a few long
/// ones, rows of one value, more rows than 256 workgroups of threads, and very long rows.
const SHAPES: [(u32, u32); 8] = [
    (1, 1),
    (1, 10),
    (128, 10),
    (1000, 3),
    (7, 1000),
    (65, 1),
    (70_000, 4),
    (3, 5000),
];

/// A kit whose steps can move `values` values.
fn kit(gpu: &Gpu, values: usize) -> Kit {
    Kit::new(gpu, 4 * values as u64 + 65_536)
}

/// `b` with an allowance for rounding the result it bounds.
fn second(b: f64) -> f64 {
    b * (1.0 + 16.0 * U)
}

/// The GPU's `op` over `rows × width` values of `a` (and `b`), into an output of NaNs.
fn gpu_run(
    kit: &mut Kit,
    kernels: &RowSoftmaxKernels,
    op: RowSoftmax,
    shape: (u32, u32),
    a: &[f32],
    b: Option<&[f32]>,
) -> Vec<f32> {
    let out = kit.upload(&vec![f32::NAN; a.len()]);
    let a = kit.upload(a);
    let b = b.map(|b| kit.upload(b));
    let call = kernels.call(
        &mut kit.steps,
        &mut kit.pool,
        op,
        shape,
        &a,
        b.as_ref(),
        &out,
    );
    let mut step = kit.steps.begin();
    kernels.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&out);
    call.release(&mut kit.pool);
    got
}

/// What the tape records for `x` (`rows × width`) through the softmax, or the log-softmax if
/// `log`, on a store of `T`, from the same `f32` values.
fn tape_forward<T: Real>(log: bool, width: usize, x: &[f32]) -> Vec<T> {
    let store = ParamStore::<T>::new_generic();
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&store.values);
    let wide: Vec<T> = x.iter().map(|&v| T::from_f64(f64::from(v))).collect();
    let rows = x.len() / width;
    let v = tape.input(Tensor::from_vec(wide, &[rows, width]).unwrap().view());
    let y = if log {
        tape.log_softmax(v)
    } else {
        tape.softmax(v)
    };
    tape.value(y).data().to_vec()
}

/// The stated bound of each forward output for `x` (`rows × width`).
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

/// The tape's backward formula evaluated in `f64` on the `f32` forward output `y` and
/// gradient `g`: `y · (g − Σ y·g)` for the softmax, `g − e^y · Σ g` for the log-softmax.
fn backward_reference(log: bool, width: usize, y: &[f32], g: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    for (y, g) in y.chunks_exact(width).zip(g.chunks_exact(width)) {
        let y: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
        let g: Vec<f64> = g.iter().map(|&v| f64::from(v)).collect();
        if log {
            let t: f64 = g.iter().sum();
            out.extend(y.iter().zip(&g).map(|(y, g)| g - y.exp() * t));
        } else {
            let d: f64 = y.iter().zip(&g).map(|(y, g)| y * g).sum();
            out.extend(y.iter().zip(&g).map(|(y, g)| y * (g - d)));
        }
    }
    out
}

/// The stated bound of each backward output for `y` and `g`.
fn backward_bounds(log: bool, width: usize, y: &[f32], g: &[f32]) -> Vec<f64> {
    let mut bounds = Vec::with_capacity(y.len());
    for (y, g) in y.chunks_exact(width).zip(g.chunks_exact(width)) {
        let y: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
        let g: Vec<f64> = g.iter().map(|&v| f64::from(v)).collect();
        if log {
            let t: f64 = g.iter().sum();
            let d_t = sum_bound(width, g.iter().map(|v| v.abs()).sum());
            for (&yj, &gj) in y.iter().zip(&g) {
                let p = yj.exp();
                let b = p * d_t + (7.0 + 4.0 * yj.abs()) * U * p * t.abs() + U * (gj - p * t).abs();
                bounds.push(second(b) + FLOOR);
            }
        } else {
            let d: f64 = y.iter().zip(&g).map(|(y, g)| y * g).sum();
            let d_d = sum_bound(width, y.iter().zip(&g).map(|(y, g)| (y * g).abs()).sum());
            for (&yj, &gj) in y.iter().zip(&g) {
                bounds.push(second(yj.abs() * (d_d + 2.0 * U * (gj - d).abs())) + FLOOR);
            }
        }
    }
    bounds
}

/// The forward pass of `op` over `x` against the `f64` tape, within its bound, and the same
/// bits on a second run; returns the worst ratio of error to bound.
fn check_forward(
    kit: &mut Kit,
    kernels: &RowSoftmaxKernels,
    log: bool,
    shape: (u32, u32),
    x: &[f32],
    what: &str,
) -> f64 {
    let op = if log {
        RowSoftmax::LogSoftmax
    } else {
        RowSoftmax::Softmax
    };
    let width = shape.1 as usize;
    let got = gpu_run(kit, kernels, op, shape, x, None);
    let want = tape_forward::<f64>(log, width, x);
    let bounds = forward_bounds(log, width, x);
    let worst = assert_within(&got, &want, |i| bounds[i], what);
    let again = gpu_run(kit, kernels, op, shape, x, None);
    assert_exact(&again, &got, &format!("{what}, a second run"));
    worst.ratio()
}

/// Both forward passes against the `f64` tape on every shape, over `[−5, 5)` and
/// `[−40, 40)`. Rows of one value give exactly 1 and exactly 0.
#[test]
fn forward_twin() {
    for_each_gpu(|gpu| {
        for (s, (rows, width)) in SHAPES.into_iter().enumerate() {
            let n = (rows * width) as usize;
            let mut kit = kit(gpu, n);
            let kernels = RowSoftmaxKernels::new(gpu, &mut kit.cache).unwrap();
            for range in [5.0f32, 40.0] {
                let x = seeded_values(10 * s as u64 + 1, n, -range, range);
                for log in [false, true] {
                    let name = if log { "log-softmax" } else { "softmax" };
                    let what = format!("{name} of {rows} × {width} within ±{range}");
                    let ratio = check_forward(&mut kit, &kernels, log, (rows, width), &x, &what);
                    println!("{what}: worst {ratio:.3} of its bound");
                }
            }
            if width == 1 {
                let x = seeded_values(3, n, -40.0, 40.0);
                let op = RowSoftmax::Softmax;
                let ones = gpu_run(&mut kit, &kernels, op, (rows, 1), &x, None);
                assert_exact(&ones, &vec![1.0; n], "softmax of one value");
                let op = RowSoftmax::LogSoftmax;
                let zeros = gpu_run(&mut kit, &kernels, op, (rows, 1), &x, None);
                assert_exact(&zeros, &vec![0.0; n], "log-softmax of one value");
            }
        }
    });
}

/// Rows whose values are close together and rows with one value far above the rest (a
/// softmax near one-hot, a log-softmax near 0 at the maximum): within their bounds.
#[test]
fn forward_twin_near_one_hot() {
    for_each_gpu(|gpu| {
        let (rows, width) = (200u32, 10u32);
        let n = (rows * width) as usize;
        let mut kit = kit(gpu, n);
        let kernels = RowSoftmaxKernels::new(gpu, &mut kit.cache).unwrap();
        let mut x = seeded_values(5, n, -1e-3, 1e-3);
        let spread = seeded_values(6, n, -30.0, -5.0);
        for (i, row) in x.chunks_exact_mut(width as usize).enumerate().skip(100) {
            row.copy_from_slice(&spread[i * width as usize..(i + 1) * width as usize]);
            row[i % width as usize] = 20.0;
        }
        for log in [false, true] {
            let what = format!(
                "{} near one-hot",
                if log { "log-softmax" } else { "softmax" }
            );
            let ratio = check_forward(&mut kit, &kernels, log, (rows, width), &x, &what);
            println!("{what}: worst {ratio:.3} of its bound");
        }
    });
}

/// Both backward passes from the `f32` tape's forward output and a gradient in `[−1, 1)`,
/// within their bounds of the tape's formula in `f64`, and the same bits on a second run.
#[test]
fn backward_twin() {
    for_each_gpu(|gpu| {
        for (s, (rows, width)) in SHAPES.into_iter().enumerate() {
            let n = (rows * width) as usize;
            let mut kit = kit(gpu, n);
            let kernels = RowSoftmaxKernels::new(gpu, &mut kit.cache).unwrap();
            for range in [5.0f32, 40.0] {
                let x = seeded_values(10 * s as u64 + 2, n, -range, range);
                let g = seeded_values(10 * s as u64 + 3, n, -1.0, 1.0);
                for log in [false, true] {
                    let (name, op) = if log {
                        ("log-softmax", RowSoftmax::LogSoftmaxBackward)
                    } else {
                        ("softmax", RowSoftmax::SoftmaxBackward)
                    };
                    let what = format!("{name} backward, {rows} × {width} within ±{range}");
                    let y = tape_forward::<f32>(log, width as usize, &x);
                    let shape = (rows, width);
                    let got = gpu_run(&mut kit, &kernels, op, shape, &y, Some(&g));
                    let want = backward_reference(log, width as usize, &y, &g);
                    let bounds = backward_bounds(log, width as usize, &y, &g);
                    let worst = assert_within(&got, &want, |i| bounds[i], &what);
                    println!("{what}: worst {:.3} of its bound", worst.ratio());
                    let again = gpu_run(&mut kit, &kernels, op, shape, &y, Some(&g));
                    assert_exact(&again, &got, &format!("{what}, a second run"));
                }
            }
        }
    });
}

/// The softmax's backward pass on values and gradients from {−2, −1, 1, 2}: every product and
/// sum is an exact integer, so the result is the formula's, bit for bit.
#[test]
fn softmax_backward_exact() {
    for_each_gpu(|gpu| {
        for (s, (rows, width)) in SHAPES.into_iter().enumerate() {
            let n = (rows * width) as usize;
            let mut kit = kit(gpu, n);
            let kernels = RowSoftmaxKernels::new(gpu, &mut kit.cache).unwrap();
            let y = exact_values(10 * s as u64 + 4, n);
            let g = exact_values(10 * s as u64 + 5, n);
            let op = RowSoftmax::SoftmaxBackward;
            let got = gpu_run(&mut kit, &kernels, op, (rows, width), &y, Some(&g));
            let want: Vec<f32> = backward_reference(false, width as usize, &y, &g)
                .into_iter()
                .map(|v| v as f32)
                .collect();
            assert_exact(&got, &want, &format!("softmax backward, {rows} × {width}"));
        }
    });
}

/// What the tape records for `x` (`rows × width`) through the softmax, or the log-softmax if
/// `log`, on an `f32` store, and the input's gradient for a gradient `g` of the output.
fn tape_backward(log: bool, width: usize, x: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let rows = x.len() / width;
    let mut store = ParamStore::new();
    let id = store.add(Tensor::from_vec(x.to_vec(), &[rows, width]).unwrap());
    let mut graph = Graph::new();
    let mut tape = graph.tape(&store.values);
    let v = tape.param(id);
    let y = if log {
        tape.log_softmax(v)
    } else {
        tape.softmax(v)
    };
    let out = tape.value(y).data().to_vec();
    // Σ y · g has the gradient g for y, exactly.
    let g = tape.input(Tensor::from_vec(g.to_vec(), &[rows, width]).unwrap().view());
    let product = tape.mul(y, g);
    let loss = tape.sum(product);
    tape.backward(loss, &mut store.grads);
    drop(tape);
    (out, store.grad(id).data().to_vec())
}

/// The rows of a 4 × 5 batch other than row 1.
fn other_rows(v: &[f32]) -> Vec<f32> {
    v[..5].iter().chain(&v[10..]).copied().collect()
}

/// Bad rows as the CPU has them, in both forward passes:
/// - a NaN or +∞, or −∞ everywhere, gives a NaN row;
/// - a −∞ elsewhere gives the `f32` tape's bits for it (a softmax of +0, a log-softmax of
///   −∞), the row's other values the bits of the same row without it, and through both
///   backward passes from the tape's output the tape's bits for it (`0 · (g − Σ y·g)` and
///   `g − 0 · Σ g`).
///
/// The other rows keep the bits of a run without the bad row.
#[test]
fn bad_rows_match_the_cpu() {
    for_each_gpu(|gpu| {
        let shape = (4u32, 5u32);
        let mut kit = kit(gpu, 20);
        let kernels = RowSoftmaxKernels::new(gpu, &mut kit.cache).unwrap();
        let x = seeded_values(7, 20, -3.0, 3.0);
        let g = seeded_values(8, 20, -1.0, 1.0);
        for log in [false, true] {
            let (op, back) = if log {
                (RowSoftmax::LogSoftmax, RowSoftmax::LogSoftmaxBackward)
            } else {
                (RowSoftmax::Softmax, RowSoftmax::SoftmaxBackward)
            };
            let clean = gpu_run(&mut kit, &kernels, op, shape, &x, None);
            let mut all_neg = x.clone();
            all_neg[5..10].fill(f32::NEG_INFINITY);
            for (name, bad) in [
                ("NaN", with(&x, 7, f32::NAN)),
                ("+inf", with(&x, 7, f32::INFINITY)),
                ("-inf everywhere", all_neg),
            ] {
                let got = gpu_run(&mut kit, &kernels, op, shape, &bad, None);
                let cpu = tape_forward::<f32>(log, 5, &bad);
                assert!(
                    cpu[5..10].iter().all(|v| v.is_nan()),
                    "{op:?} with {name}: CPU"
                );
                assert!(
                    got[5..10].iter().all(|v| v.is_nan()),
                    "{op:?} with {name}: the bad row is {:?}",
                    &got[5..10]
                );
                let what = format!("{op:?} with {name}: the other rows");
                assert_exact(&other_rows(&got), &other_rows(&clean), &what);
            }
            let bad = with(&x, 7, f32::NEG_INFINITY);
            let got = gpu_run(&mut kit, &kernels, op, shape, &bad, None);
            let (cpu, cpu_dx) = tape_backward(log, 5, &bad, &g);
            assert_exact(&got[7..8], &cpu[7..8], &format!("{op:?} of -inf"));
            let short: Vec<f32> = [5, 6, 8, 9].iter().map(|&i| bad[i]).collect();
            let alone = gpu_run(&mut kit, &kernels, op, (1, 4), &short, None);
            let rest = [got[5], got[6], got[8], got[9]];
            assert_exact(&rest, &alone, &format!("{op:?} beside -inf"));
            let what = format!("{op:?} with -inf: the other rows");
            assert_exact(&other_rows(&got), &other_rows(&clean), &what);
            let dx = gpu_run(&mut kit, &kernels, back, shape, &cpu, Some(&g));
            assert_exact(&dx[7..8], &cpu_dx[7..8], &format!("{back:?} at -inf"));
            assert!(dx.iter().all(|v| v.is_finite()), "{back:?}: {dx:?}");
        }
    });
}

/// `x` with value `at` set to `v`.
fn with(x: &[f32], at: usize, v: f32) -> Vec<f32> {
    let mut x = x.to_vec();
    x[at] = v;
    x
}

/// A call refuses an output that is one of its inputs, and a backward pass without its
/// gradient, naming the kernel.
#[test]
fn calls_refuse_bad_buffers() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 8);
        let kernels = RowSoftmaxKernels::new(gpu, &mut kit.cache).unwrap();
        let a = kit.upload(&[0.0; 8]);
        let out = kit.upload(&[0.0; 8]);
        let message = |body: &mut dyn FnMut()| {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_err();
            err.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        let (steps, pool) = (&mut kit.steps, &mut kit.pool);
        let op = RowSoftmax::Softmax;
        let text = message(&mut || {
            kernels.call(steps, pool, op, (2, 4), &a, None, &a);
        });
        assert!(
            text.contains("row softmax") && text.contains("different"),
            "{text}"
        );
        let op = RowSoftmax::SoftmaxBackward;
        let text = message(&mut || {
            kernels.call(steps, pool, op, (2, 4), &a, None, &out);
        });
        assert!(text.contains("none was given"), "{text}");
        let text = message(&mut || {
            kernels.call(steps, pool, RowSoftmax::Softmax, (3, 4), &a, None, &out);
        });
        assert!(text.contains("needs 12 values"), "{text}");
    });
}
