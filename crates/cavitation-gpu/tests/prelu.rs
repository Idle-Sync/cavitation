//! PReLU against the tape's `Tape::prelu`, on every backend this machine has, with one slope
//! per column and with one shared slope:
//!
//! - **Forward** and **input gradient:** the bits of the `f32` tape's output and of its
//!   input gradient (one multiplication each), on seeded values and on zeros of both signs,
//!   infinities, NaN, the extreme normals and ±1; a NaN matches any NaN, and a subnormal
//!   result may be flushed. Positive subnormal inputs are left out: a GPU that flushes them
//!   reads them as 0, so their input gradient is `g · slope` instead of `g` (see
//!   `PreluKernels`).
//! - **Slopes' gradient:** each channel's sum within `2(L + 2)·u·Σ|g·x| + 3L·2⁻¹²⁶` of the
//!   exact (`f64`) sum of the same `f32` terms `g·x` over the values not above 0, for
//!   `L = rows`, the bound `PreluKernels` states; on small integers, whose sums are exact in
//!   any order, the `f32` tape's bits. A NaN input makes its channel's sum NaN and leaves
//!   the others alone, as on the CPU. The `3L·2⁻¹²⁶` covers flushed products and partial
//!   sums, not flushed operands: a subnormal `x` or `g` read as 0 is outside the bound, and
//!   the values here are drawn from `[−1, 1)` with no subnormals among them. The any-order
//!   term holds for large `L` with room to spare: the tree's addition depth is about
//!   `24 + L/4096`, far below `L + 2`.
//!
//! Every output starts as NaN, so a value left unwritten fails, and a second run gives the
//! same bits. The tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, Tensor};
use cavitation_gpu::Gpu;
use cavitation_gpu::kernels::{Prelu, PreluKernels};
use common::{FLOOR, Kit, assert_exact, assert_within, exact_values, for_each_gpu};
use common::{seeded_values, sum_bound};

/// The shapes every twin runs, `(rows, channels)`: one value, CIFAR's 128 × 10, many short
/// rows, a few long ones, a shared slope over one chunk of the sums and over many, and more
/// rows than one chunk.
const SHAPES: [(u32, u32); 8] = [
    (1, 1),
    (128, 10),
    (1000, 3),
    (7, 1000),
    (4000, 1),
    (300_000, 1),
    (70_000, 4),
    (5000, 64),
];

/// Values the forward and input-gradient twins include besides the seeded ones.
const SPECIAL: [f32; 11] = [
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MAX,
    f32::MIN,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    1.0,
    -1.0,
];

/// A kit whose steps can move `values` values.
fn kit(gpu: &Gpu, values: usize) -> Kit {
    Kit::new(gpu, 4 * values as u64 + 65_536)
}

/// The GPU's `op` over `x` with slopes `s` (and `g`), into an output of NaNs.
fn gpu_run(
    kit: &mut Kit,
    kernels: &PreluKernels,
    op: Prelu,
    x: &[f32],
    s: &[f32],
    g: Option<&[f32]>,
) -> Vec<f32> {
    let shape = ((x.len() / s.len()) as u32, s.len() as u32);
    let out = kit.upload(&vec![f32::NAN; x.len()]);
    let (xb, sb) = (kit.upload(x), kit.upload(s));
    let gb = g.map(|g| kit.upload(g));
    let call = kernels.call(
        &mut kit.steps,
        &mut kit.pool,
        op,
        shape,
        &xb,
        &sb,
        gb.as_ref(),
        &out,
    );
    let mut step = kit.steps.begin();
    kernels.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&out);
    call.release(&mut kit.pool);
    got
}

/// The GPU's slopes' gradient for `x` and `g` with `channels` slopes, into an output of
/// NaNs.
fn gpu_slope_grad(
    kit: &mut Kit,
    kernels: &PreluKernels,
    x: &[f32],
    g: &[f32],
    channels: usize,
) -> Vec<f32> {
    let shape = ((x.len() / channels) as u32, channels as u32);
    let out = kit.upload(&vec![f32::NAN; channels]);
    let (xb, gb) = (kit.upload(x), kit.upload(g));
    let call = kernels.slope_grad_call(&mut kit.steps, &mut kit.pool, shape, &xb, &gb, &out);
    let mut step = kit.steps.begin();
    kernels.record_slope_grad(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&out);
    call.release(&mut kit.pool);
    got
}

/// The `f32` tape's PReLU of `x` (`rows × channels`) with the slopes `s`, and, for the
/// gradient `g` of its output, the gradients of `x` and of the slopes.
fn tape(x: &[f32], s: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (c, rows) = (s.len(), x.len() / s.len());
    let mut store = ParamStore::new();
    let xi = store.add(Tensor::from_vec(x.to_vec(), &[rows, c]).unwrap());
    let si = store.add(Tensor::from_vec(s.to_vec(), &[c]).unwrap());
    let mut graph = Graph::new();
    let mut tape = graph.tape(&store.values);
    let (xv, sv) = (tape.param(xi), tape.param(si));
    let y = tape.prelu(xv, sv);
    let out = tape.value(y).data().to_vec();
    // Σ y · g has the gradient g for y, exactly.
    let gv = tape.input(Tensor::from_vec(g.to_vec(), &[rows, c]).unwrap().view());
    let product = tape.mul(y, gv);
    let loss = tape.sum(product);
    tape.backward(loss, &mut store.grads);
    drop(tape);
    (
        out,
        store.grad(xi).data().to_vec(),
        store.grad(si).data().to_vec(),
    )
}

/// The exact sum of each channel's `f32` terms `g·x` over its values not above 0, in `f64`,
/// and the sum of their magnitudes.
fn slope_sums(x: &[f32], g: &[f32], channels: usize) -> (Vec<f64>, Vec<f64>) {
    let mut sums = vec![0.0f64; channels];
    let mut abs = vec![0.0f64; channels];
    for (i, (&v, &gi)) in x.iter().zip(g).enumerate() {
        if v > 0.0 {
            continue;
        }
        let term = gi * v;
        sums[i % channels] += f64::from(term);
        abs[i % channels] += f64::from(term).abs();
    }
    (sums, abs)
}

/// Seeded values in `[−4, 4)` with [`SPECIAL`] at their start when there is room.
fn with_specials(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seeded_values(seed, n, -4.0, 4.0);
    let k = SPECIAL.len().min(n);
    x[..k].copy_from_slice(&SPECIAL[..k]);
    x
}

/// Slopes for a twin: seeded in `[−2, 2)`, with 0 and 1 among them when there are three or
/// more.
fn slopes(seed: u64, channels: usize) -> Vec<f32> {
    let mut s = seeded_values(seed, channels, -2.0, 2.0);
    if channels >= 3 {
        s[1] = 0.0;
        s[2] = 1.0;
    }
    s
}

/// The forward pass and the input's gradient give the `f32` tape's bits on every shape,
/// special values included, and the same bits on a second run.
#[test]
fn forward_and_input_grad_twin() {
    for_each_gpu(|gpu| {
        for (k, (rows, channels)) in SHAPES.into_iter().enumerate() {
            let (n, c) = ((rows * channels) as usize, channels as usize);
            let mut kit = kit(gpu, n);
            let kernels = PreluKernels::new(gpu, &mut kit.cache).unwrap();
            let seed = 10 * k as u64;
            let x = with_specials(seed + 1, n);
            let s = slopes(seed + 2, c);
            let g = seeded_values(seed + 3, n, -1.0, 1.0);
            let (y, dx, _) = tape(&x, &s, &g);
            let what = format!("{rows} × {channels}");
            let got = gpu_run(&mut kit, &kernels, Prelu::Forward, &x, &s, None);
            assert_exact(&got, &y, &format!("forward, {what}"));
            let again = gpu_run(&mut kit, &kernels, Prelu::Forward, &x, &s, None);
            assert_exact(&again, &got, &format!("forward, {what}, a second run"));
            let got = gpu_run(&mut kit, &kernels, Prelu::InputGrad, &x, &s, Some(&g));
            assert_exact(&got, &dx, &format!("input gradient, {what}"));
        }
    });
}

/// The derivative at exactly 0 is the slope, and a NaN input gives NaN both ways, as the
/// tape has them.
#[test]
fn zero_and_nan_inputs() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 8);
        let kernels = PreluKernels::new(gpu, &mut kit.cache).unwrap();
        let x = [0.0, -0.0, f32::NAN, 2.0];
        let s = [0.5, -3.0, 0.25, 0.75];
        let g = [2.0, 1.0, 1.0, 4.0];
        let dx = gpu_run(&mut kit, &kernels, Prelu::InputGrad, &x, &s, Some(&g));
        assert_eq!(dx[..2], [1.0, -3.0]);
        assert!(dx[2].is_nan(), "{dx:?}");
        assert_eq!(dx[3], 4.0);
        let y = gpu_run(&mut kit, &kernels, Prelu::Forward, &x, &s, None);
        assert_eq!(
            [y[0].to_bits(), y[1].to_bits(), y[3].to_bits()],
            [0.0f32.to_bits(), 0.0f32.to_bits(), 2.0f32.to_bits()]
        );
        assert!(y[2].is_nan(), "{y:?}");
    });
}

/// The slopes' gradient on seeded values against the exact sums of the same `f32` terms,
/// within the stated bound, and the same bits on a second run.
#[test]
fn slope_grad_rounding_twin() {
    for_each_gpu(|gpu| {
        for (k, (rows, channels)) in SHAPES.into_iter().enumerate() {
            let (n, c) = ((rows * channels) as usize, channels as usize);
            let mut kit = kit(gpu, n);
            let kernels = PreluKernels::new(gpu, &mut kit.cache).unwrap();
            let seed = 10 * k as u64 + 100;
            let x = seeded_values(seed, n, -1.0, 1.0);
            let g = seeded_values(seed + 1, n, -1.0, 1.0);
            let got = gpu_slope_grad(&mut kit, &kernels, &x, &g, c);
            let (exact, abs) = slope_sums(&x, &g, c);
            let l = rows as usize;
            let what = format!("slopes' gradient, {rows} × {channels}");
            let worst = assert_within(
                &got,
                &exact,
                |j| sum_bound(l, abs[j]) + 3.0 * l as f64 * FLOOR,
                &what,
            );
            println!("{what}: worst {:.3} of its bound", worst.ratio());
            let again = gpu_slope_grad(&mut kit, &kernels, &x, &g, c);
            assert_exact(&again, &got, &format!("{what}, a second run"));
        }
    });
}

/// The slopes' gradient on values and gradients from {−2, −1, 1, 2}: every term and partial
/// sum is an exact integer, so the result is the `f32` tape's, bit for bit.
#[test]
fn slope_grad_exact() {
    for_each_gpu(|gpu| {
        for (k, (rows, channels)) in SHAPES.into_iter().enumerate() {
            let (n, c) = ((rows * channels) as usize, channels as usize);
            let mut kit = kit(gpu, n);
            let kernels = PreluKernels::new(gpu, &mut kit.cache).unwrap();
            let seed = 10 * k as u64 + 200;
            let x = exact_values(seed, n);
            let g = exact_values(seed + 1, n);
            let s = slopes(seed + 2, c);
            let (_, _, ds) = tape(&x, &s, &g);
            let got = gpu_slope_grad(&mut kit, &kernels, &x, &g, c);
            let what = format!("slopes' gradient, {rows} × {channels}");
            assert_exact(&got, &ds, &what);
        }
    });
}

/// A NaN input makes its channel's slope gradient NaN, as on the CPU, and leaves the other
/// channels' sums as they are without it; a +∞ input is above 0 and adds nothing, and a −∞
/// gives the tape's infinity.
#[test]
fn slope_grad_bad_values() {
    for_each_gpu(|gpu| {
        let (rows, c) = (50usize, 4usize);
        let mut kit = kit(gpu, rows * c);
        let kernels = PreluKernels::new(gpu, &mut kit.cache).unwrap();
        let x = exact_values(300, rows * c);
        let g = exact_values(301, rows * c);
        let s = slopes(302, c);
        let clean = gpu_slope_grad(&mut kit, &kernels, &x, &g, c);
        let mut bad = x.clone();
        bad[4 * 7 + 1] = f32::NAN;
        bad[4 * 9 + 2] = f32::INFINITY;
        bad[4 * 11 + 3] = f32::NEG_INFINITY;
        let got = gpu_slope_grad(&mut kit, &kernels, &bad, &g, c);
        let (_, _, ds) = tape(&bad, &s, &g);
        assert!(
            ds[1].is_nan() && got[1].is_nan(),
            "{got:?}, the CPU's {ds:?}"
        );
        assert_exact(&got[..1], &clean[..1], "the channel without a bad value");
        assert_exact(&got, &ds, "the tape's sums with bad values");
        assert!(got[3].is_infinite(), "{got:?}");
    });
}

/// A call refuses an output that is one of its inputs, an input gradient without its
/// gradient, a buffer too short and a slope gradient over more than 65,535 channels, naming
/// the kernel.
#[test]
fn calls_refuse_bad_buffers() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 8);
        let kernels = PreluKernels::new(gpu, &mut kit.cache).unwrap();
        let a = kit.upload(&[0.0; 8]);
        let s = kit.upload(&[0.0; 4]);
        let out = kit.upload(&[0.0; 8]);
        let message = |body: &mut dyn FnMut()| {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_err();
            err.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        let (steps, pool) = (&mut kit.steps, &mut kit.pool);
        let text = message(&mut || {
            kernels.call(steps, pool, Prelu::Forward, (2, 4), &a, &s, None, &a);
        });
        assert!(
            text.contains("prelu") && text.contains("different"),
            "{text}"
        );
        let text = message(&mut || {
            kernels.call(steps, pool, Prelu::InputGrad, (2, 4), &a, &s, None, &out);
        });
        assert!(text.contains("none was given"), "{text}");
        let text = message(&mut || {
            kernels.call(steps, pool, Prelu::Forward, (3, 4), &a, &s, None, &out);
        });
        assert!(text.contains("needs 12 values"), "{text}");
        let text = message(&mut || {
            kernels.call(steps, pool, Prelu::Forward, (1, 8), &a, &s, None, &out);
        });
        assert!(
            text.contains("slopes") && text.contains("8 values"),
            "{text}"
        );
        let text = message(&mut || {
            kernels.slope_grad_call(steps, pool, (2, 4), &a, &out, &a);
        });
        assert!(text.contains("different"), "{text}");
        let text = message(&mut || {
            kernels.slope_grad_call(steps, pool, (1, 65_536), &a, &out, &s);
        });
        assert!(text.contains("more than 65535 in a dimension"), "{text}");
    });
}
