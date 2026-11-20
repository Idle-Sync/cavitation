//! PReLU on the card against the tape's `Tape::prelu`, with one slope per column and with one
//! shared slope, as `cavitation-gpu`'s tests check its WGSL kernels:
//!
//! - **Forward** and **input gradient:** the bits of the `f32` tape's output and of its
//!   input gradient, on seeded values and on zeros of both signs, infinities, NaN, the
//!   extreme normals, subnormals and ±1.
//! - **Slopes' gradient:** each channel's sum within `2(L + 2)·u·Σ|g·x|` of the exact (`f64`)
//!   sum of the same `f32` terms `g·x` over the values not above 0, for `L = rows`, the bound
//!   `Prelu` states (the WGSL kernels' without their flushing allowance); on small integers,
//!   whose sums are exact in any order, the `f32` tape's bits. A NaN input makes its
//!   channel's sum NaN, as on the CPU.
//!
//! Every output starts as NaN, so a value left unwritten fails, and two runs give the same
//! bits. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes that a
//! failure.

mod common;

use cavitation_core::{Graph, ParamStore, Tensor};
use cavitation_cuda::{Cuda, CudaError, Prelu, PreluOp};
use common::{
    U, assert_exact, assert_within, exact_values, nans, panic_message, read, seeded_values, upload,
};

/// The shapes every twin runs, `(rows, channels)`: one value, CIFAR's 128 × 10, many short
/// rows, a few long ones, a shared slope over one chunk of the sums and over many, and more
/// rows than one chunk.
const SHAPES: [(usize, usize); 8] = [
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
const SPECIAL: [f32; 13] = [
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MAX,
    f32::MIN,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    1e-40,
    -1e-40,
    1.0,
    -1.0,
];

// `sum_bound` copies the bound of `crates/cavitation-gpu/tests/common`'s, which the WGSL
// kernels' tests use; a change to one must be made to the other.

/// The error bound of a sum of `k` terms in any order: `2(k + 2) · u · Σ|terms|`.
fn sum_bound(k: usize, abs_sum: f64) -> f64 {
    2.0 * (k as f64 + 2.0) * U * abs_sum
}

/// The card's `op` over `x` with slopes `s` (and `g`), into an output of NaNs.
fn cuda_run(
    cuda: &Cuda,
    kernels: &Prelu<'_>,
    op: PreluOp,
    x: &[f32],
    s: &[f32],
    g: Option<&[f32]>,
) -> Vec<f32> {
    let (xb, sb) = (upload(cuda, x), upload(cuda, s));
    let gb = g.map(|g| upload(cuda, g));
    let mut out = nans(cuda, x.len());
    let rows = x.len() / s.len();
    kernels
        .run(op, &xb, &sb, gb.as_ref(), &mut out, rows, s.len())
        .unwrap();
    read(&out, x.len())
}

/// The card's slopes' gradient for `x` and `g` with `channels` slopes, into an output of
/// NaNs.
fn cuda_slope_grad(
    cuda: &Cuda,
    kernels: &Prelu<'_>,
    x: &[f32],
    g: &[f32],
    channels: usize,
) -> Vec<f32> {
    let rows = x.len() / channels;
    let (xb, gb) = (upload(cuda, x), upload(cuda, g));
    let mut partials = nans(cuda, Prelu::partials_len(rows, channels));
    let mut out = nans(cuda, channels);
    kernels
        .slope_grad(&xb, &gb, &mut partials, &mut out, rows, channels)
        .unwrap();
    read(&out, channels)
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
        let term = f64::from(gi * v);
        sums[i % channels] += term;
        abs[i % channels] += term.abs();
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
/// special values and subnormals included, and the same bits on a second run.
#[test]
fn forward_and_input_grad_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Prelu::new(&cuda).unwrap();
    for (k, (rows, c)) in SHAPES.into_iter().enumerate() {
        let n = rows * c;
        let seed = 10 * k as u64;
        let x = with_specials(seed + 1, n);
        let s = slopes(seed + 2, c);
        let g = seeded_values(seed + 3, n, -1.0, 1.0);
        let (y, dx, _) = tape(&x, &s, &g);
        let what = format!("{rows} × {c}");
        let got = cuda_run(&cuda, &kernels, PreluOp::Forward, &x, &s, None);
        assert_exact(&got, &y, &format!("forward, {what}"));
        let again = cuda_run(&cuda, &kernels, PreluOp::Forward, &x, &s, None);
        assert_exact(&again, &got, &format!("forward, {what}, a second run"));
        let got = cuda_run(&cuda, &kernels, PreluOp::InputGrad, &x, &s, Some(&g));
        assert_exact(&got, &dx, &format!("input gradient, {what}"));
    }
}

/// The derivative at exactly 0 is the slope, and a NaN input gives NaN both ways, as the
/// tape has them.
#[test]
fn zero_and_nan_inputs() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Prelu::new(&cuda).unwrap();
    let x = [0.0, -0.0, f32::NAN, 2.0];
    let s = [0.5, -3.0, 0.25, 0.75];
    let g = [2.0, 1.0, 1.0, 4.0];
    let dx = cuda_run(&cuda, &kernels, PreluOp::InputGrad, &x, &s, Some(&g));
    assert_eq!(dx[..2], [1.0, -3.0]);
    assert!(dx[2].is_nan(), "{dx:?}");
    assert_eq!(dx[3], 4.0);
    let y = cuda_run(&cuda, &kernels, PreluOp::Forward, &x, &s, None);
    assert_eq!(
        [y[0].to_bits(), y[1].to_bits(), y[3].to_bits()],
        [0.0f32.to_bits(), 0.0f32.to_bits(), 2.0f32.to_bits()]
    );
    assert!(y[2].is_nan(), "{y:?}");
}

/// The slopes' gradient on seeded values against the exact sums of the same `f32` terms,
/// within the stated bound, and the same bits on a second run.
#[test]
fn slope_grad_rounding_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Prelu::new(&cuda).unwrap();
    for (k, (rows, c)) in SHAPES.into_iter().enumerate() {
        let n = rows * c;
        let seed = 10 * k as u64 + 100;
        let x = seeded_values(seed, n, -1.0, 1.0);
        let g = seeded_values(seed + 1, n, -1.0, 1.0);
        let got = cuda_slope_grad(&cuda, &kernels, &x, &g, c);
        let (exact, abs) = slope_sums(&x, &g, c);
        let what = format!("slopes' gradient, {rows} × {c}");
        let worst = assert_within(&got, &exact, |j| sum_bound(rows, abs[j]), &what);
        println!("{what}: worst {:.3} of its bound", worst.ratio());
        let again = cuda_slope_grad(&cuda, &kernels, &x, &g, c);
        assert_exact(&again, &got, &format!("{what}, a second run"));
    }
}

/// The slopes' gradient on values and gradients from {−2, −1, 1, 2}: every term and partial
/// sum is an exact integer, so the result is the `f32` tape's, bit for bit.
#[test]
fn slope_grad_exact() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Prelu::new(&cuda).unwrap();
    for (k, (rows, c)) in SHAPES.into_iter().enumerate() {
        let n = rows * c;
        let seed = 10 * k as u64 + 200;
        let x = exact_values(seed, n);
        let g = exact_values(seed + 1, n);
        let s = slopes(seed + 2, c);
        let (_, _, ds) = tape(&x, &s, &g);
        let got = cuda_slope_grad(&cuda, &kernels, &x, &g, c);
        assert_exact(&got, &ds, &format!("slopes' gradient, {rows} × {c}"));
    }
}

/// A NaN input makes its channel's slope gradient NaN, as on the CPU, and leaves the other
/// channels' sums as they are without it; a +∞ input is above 0 and adds nothing, and a −∞
/// gives the tape's infinity.
#[test]
fn slope_grad_bad_values() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Prelu::new(&cuda).unwrap();
    let (rows, c) = (50usize, 4usize);
    let x = exact_values(300, rows * c);
    let g = exact_values(301, rows * c);
    let s = slopes(302, c);
    let clean = cuda_slope_grad(&cuda, &kernels, &x, &g, c);
    let mut bad = x.clone();
    bad[4 * 7 + 1] = f32::NAN;
    bad[4 * 9 + 2] = f32::INFINITY;
    bad[4 * 11 + 3] = f32::NEG_INFINITY;
    let got = cuda_slope_grad(&cuda, &kernels, &bad, &g, c);
    let (_, _, ds) = tape(&bad, &s, &g);
    assert!(
        ds[1].is_nan() && got[1].is_nan(),
        "{got:?}, the CPU's {ds:?}"
    );
    assert_exact(&got[..1], &clean[..1], "the channel without a bad value");
    assert_exact(&got, &ds, "the tape's sums with bad values");
    assert!(got[3].is_infinite(), "{got:?}");
}

/// The refusals: a gradient missing or given against the operation, a shape too large for
/// the kernels' indices or the sums' grid, and a buffer too short.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Prelu::new(&cuda).unwrap();
    let (x, s) = (upload(&cuda, &[0.0f32; 8]), upload(&cuda, &[0.0f32; 4]));
    let mut out = nans(&cuda, 8);
    let err = kernels
        .run(PreluOp::InputGrad, &x, &s, None, &mut out, 2, 4)
        .unwrap_err();
    assert!(
        matches!(err, CudaError::Invalid(ref m) if m.contains("needs")),
        "{err}"
    );
    let err = kernels
        .run(PreluOp::Forward, &x, &s, Some(&x), &mut out, 2, 4)
        .unwrap_err();
    assert!(
        matches!(err, CudaError::Invalid(ref m) if m.contains("takes no")),
        "{err}"
    );
    let err = kernels
        .run(PreluOp::Forward, &x, &s, None, &mut out, 1 << 30, 8)
        .unwrap_err();
    assert!(err.to_string().contains("do not fit a u32"), "{err}");
    let mut partials = nans(&cuda, 70_000);
    let mut ds = nans(&cuda, 70_000);
    let err = kernels
        .slope_grad(&x, &x, &mut partials, &mut ds, 1, 70_000)
        .unwrap_err();
    assert!(err.to_string().contains("65535 channels"), "{err}");
    let text = panic_message(|| {
        kernels
            .run(PreluOp::Forward, &x, &s, None, &mut out, 3, 4)
            .unwrap();
    });
    assert!(
        text.contains("input buffer") && text.contains("12 values"),
        "{text}"
    );
    let text = panic_message(|| {
        kernels
            .run(PreluOp::Forward, &x, &s, None, &mut out, 1, 8)
            .unwrap();
    });
    assert!(text.contains("slopes buffer"), "{text}");
}
