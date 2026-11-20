//! The GLU kernels on the card against the tape's `Tape::glu`, with every gate, as
//! `cavitation-gpu`'s tests check its WGSL kernels:
//!
//! - **ReLU gate:** the bits of the `f32` tape's output and of both halves of its input
//!   gradient, on seeded values and on zeros of both signs, infinities, NaN, the extreme
//!   normals, subnormals and ±1, from one pair to 16,777,160 pairs, a NaN matching any NaN.
//! - **Sigmoid, GELU (both forms) and SiLU gates:** within the WGSL kernels' bounds of the
//!   `f64` tape, derived in `crates/cavitation-gpu/tests/glu.rs` (the gate's own bound from
//!   the element-wise tests carried through the products); on ±0, ±∞ and NaN, in either
//!   half, the `f32` tape's bits.
//!
//! Every output starts as NaN, so a value left unwritten fails, and two runs give the same
//! bits. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes that a
//! failure.

mod common;

use cavitation_core::{Gate, GeluApprox, Graph, ParamStore, Real, Tensor};
use cavitation_cuda::{Cuda, CudaError, Glu, GluGate, GluOp};
use common::{
    FLOOR, U, assert_exact, assert_within, nans, panic_message, read, seeded_values, upload,
};

/// Every gate, the tape's and the card's.
const GATES: [(Gate, GluGate); 5] = [
    (Gate::Sigmoid, GluGate::Sigmoid),
    (Gate::Gelu(GeluApprox::Tanh), GluGate::GeluTanh),
    (Gate::Gelu(GeluApprox::Exact), GluGate::GeluErf),
    (Gate::Relu, GluGate::Relu),
    (Gate::Silu, GluGate::Silu),
];

/// The shapes the ReLU gate's twin runs on the tape, `(rows, width)`: one pair, CIFAR-like
/// rows, many short rows, a few long ones and an odd half.
const SHAPES: [(usize, usize); 5] = [(1, 2), (128, 20), (1000, 6), (7, 2000), (3, 130)];

/// Values the ReLU gate's twin includes besides the seeded ones.
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

/// The card's `op` with `gate` over `x` (rows of `width`), with `g` for the backward pass,
/// into an output of NaNs.
fn cuda_run(
    cuda: &Cuda,
    kernels: &Glu<'_>,
    op: GluOp,
    gate: GluGate,
    width: usize,
    x: &[f32],
    g: Option<&[f32]>,
) -> Vec<f32> {
    let written = if op.takes_g() { x.len() } else { x.len() / 2 };
    let xb = upload(cuda, x);
    let gb = g.map(|g| upload(cuda, g));
    let mut out = nans(cuda, written);
    kernels
        .run(op, gate, &xb, gb.as_ref(), &mut out, x.len() / width, width)
        .unwrap();
    read(&out, written)
}

/// The tape's GLU with `gate` of `x` (rows of `width`) and, for the gradient `g` of its
/// output, `x`'s gradient, in `T`: the CPU's own `f32` arithmetic, or the `f64` reference.
fn tape<T: Real>(x: &[f32], g: &[f32], width: usize, gate: Gate) -> (Vec<T>, Vec<T>) {
    let widen = |v: &[f32]| {
        v.iter()
            .map(|&v| T::from_f64(f64::from(v)))
            .collect::<Vec<T>>()
    };
    let rows = x.len() / width;
    let mut params: ParamStore<T> = ParamStore::default();
    let id = params.add(Tensor::from_vec(widen(x), &[rows, width]).unwrap());
    let gs = Tensor::from_vec(widen(g), &[rows, width / 2]).unwrap();
    let mut graph: Graph<T> = Graph::default();
    let mut tape = graph.tape(&params.values);
    let xv = tape.param(id);
    let y = tape.glu(xv, gate);
    let values = tape.value(y).data().to_vec();
    // Σ y · g has the gradient g for y, exactly.
    let gv = tape.input(gs.view());
    let weighted = tape.mul(y, gv);
    let loss = tape.sum(weighted);
    tape.backward(loss, &mut params.grads);
    drop(tape);
    (values, params.grad(id).data().to_vec())
}

/// Seeded values in `[−4, 4)` with [`SPECIAL`] at their start and again at their middle
/// (so specials land in both halves of a row) when there is room.
fn with_specials(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seeded_values(seed, n, -4.0, 4.0);
    let k = SPECIAL.len().min(n / 2);
    x[..k].copy_from_slice(&SPECIAL[..k]);
    let mid = n / 2 + 1;
    let k = SPECIAL.len().min(n - mid);
    x[mid..mid + k].copy_from_slice(&SPECIAL[..k]);
    x
}

/// The ReLU gate gives the `f32` tape's bits forward and backward on every shape, special
/// values and subnormals included, and the same bits on a second run.
#[test]
fn relu_gate_gives_the_tapes_bits() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Glu::new(&cuda).unwrap();
    for (k, (rows, width)) in SHAPES.into_iter().enumerate() {
        let n = rows * width;
        let seed = 10 * k as u64;
        let x = with_specials(seed + 1, n);
        let g = seeded_values(seed + 2, n / 2, -1.0, 1.0);
        let (y, dx) = tape::<f32>(&x, &g, width, Gate::Relu);
        let what = format!("{rows} × {width}");
        let relu = GluGate::Relu;
        let got = cuda_run(&cuda, &kernels, GluOp::Forward, relu, width, &x, None);
        assert_exact(&got, &y, &format!("forward, {what}"));
        let again = cuda_run(&cuda, &kernels, GluOp::Forward, relu, width, &x, None);
        assert_exact(&again, &got, &format!("forward, {what}, a second run"));
        let got = cuda_run(&cuda, &kernels, GluOp::Backward, relu, width, &x, Some(&g));
        assert_exact(&got, &dx, &format!("backward, {what}"));
    }
}

/// 16,777,160 pairs in rows of one pair: the ReLU gate's bits, from the tape's formulas
/// (`a · relu(b)`, `g · relu(b)` and `(g · a) · relu′(b)`, which the shapes above check
/// against the tape itself).
#[test]
fn relu_gate_over_many_pairs() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Glu::new(&cuda).unwrap();
    let pairs = 65_535 * 256 + 200;
    let n = 2 * pairs;
    let x = seeded_values(500, n, -4.0, 4.0);
    let g = seeded_values(501, pairs, -1.0, 1.0);
    let relu = |v: f32| if v <= 0.0 { 0.0 } else { v };
    let d = |v: f32| if v > 0.0 { 1.0 } else { 0.0 };
    let y: Vec<f32> = (0..pairs).map(|i| x[2 * i] * relu(x[2 * i + 1])).collect();
    let dx: Vec<f32> = (0..n)
        .map(|k| {
            let (i, a, b) = (k / 2, x[k & !1], x[k | 1]);
            if k % 2 == 0 {
                g[i] * relu(b)
            } else {
                (g[i] * a) * d(b)
            }
        })
        .collect();
    let gate = GluGate::Relu;
    let got = cuda_run(&cuda, &kernels, GluOp::Forward, gate, 2, &x, None);
    assert_exact(&got, &y, "forward over many pairs");
    let got = cuda_run(&cuda, &kernels, GluOp::Backward, gate, 2, &x, Some(&g));
    assert_exact(&got, &dx, "backward over many pairs");
}

// `sigma`, `gate_bounds` and `sweep` copy `crates/cavitation-gpu/tests/glu.rs`'s, where the
// bounds are derived; a change to one must be made to the other.

/// The logistic sigmoid in `f64`.
fn sigma(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// For `gate` at `b`, `(V, E, |G|, |D|)`: the bounds of the card's gate value and derivative
/// against the `f64` tape's, before the products, and bounds of the value and the derivative
/// themselves.
fn gate_bounds(gate: Gate, b: f64) -> (f64, f64, f64, f64) {
    let s = sigma(b);
    match gate {
        Gate::Sigmoid => (8.0 * U * s, 10.0 * U * s, s, s),
        Gate::Silu => (
            10.0 * U * b.abs() * s,
            12.0 * U * s * (1.0 + b.abs()),
            b.abs() * s,
            s * (1.0 + b.abs()),
        ),
        Gate::Gelu(approx) => {
            let (u, slope, du) = match approx {
                GeluApprox::Tanh => {
                    let u = 0.797_884_560_802_865_4 * (b + 0.044715 * b.powi(3));
                    let du = 0.797_884_560_802_865_4 * (1.0 + 0.134_145 * b * b);
                    (u, 1.0 / u.cosh().powi(2), du)
                }
                GeluApprox::Exact => {
                    let u = b / std::f64::consts::SQRT_2;
                    (u, (-u * u).exp(), 0.0)
                }
            };
            let size_f = b.abs() + b.abs() * u.abs().max(1.0) * slope;
            let size_b = 1.0 + b.abs() * du + (1.0 + b * b) * (1.0 + u * u) * slope;
            (4.0 * U * size_f, 8.0 * U * size_b, b.abs(), 1.2)
        }
        Gate::Relu => unreachable!("the ReLU gate is exact"),
    }
}

/// The gate's inputs `b`: a dense sweep of [−10, 10] in steps of 1/64, seeded values in
/// [−8, 8], the tails where the gates saturate and the exponential underflows, and the
/// subnormals ±1e-40, which the card keeps.
fn sweep(seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (-640..=640).map(|i| i as f32 / 64.0).collect();
    v.extend(seeded_values(seed, 20_000, -8.0, 8.0));
    v.extend([
        -1e12, -50.0, -20.0, 20.0, 50.0, 1e12, -87.0, -90.0, -104.0, -110.0, 87.0, 104.0, 1e-30,
        -1e-30, 1e-40, -1e-40,
    ]);
    v
}

/// The sigmoid, GELU (both forms) and SiLU gates, forward and backward, within the WGSL
/// kernels' bounds of the `f64` tape, in rows of 7 pairs, and the same bits on a second run.
#[test]
fn smooth_gates_within_their_bounds() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Glu::new(&cuda).unwrap();
    let half = 7;
    let mut b = sweep(20);
    b.resize(b.len().div_ceil(half) * half, 0.5);
    let pairs = b.len();
    let rows = pairs / half;
    let a = seeded_values(21, pairs, -4.0, 4.0);
    let g = seeded_values(22, pairs, -2.0, 2.0);
    let mut x = Vec::with_capacity(2 * pairs);
    for r in 0..rows {
        x.extend_from_slice(&a[r * half..(r + 1) * half]);
        x.extend_from_slice(&b[r * half..(r + 1) * half]);
    }
    for (gate, card) in GATES.into_iter().filter(|&(g, _)| g != Gate::Relu) {
        let (want, grad) = tape::<f64>(&x, &g, 2 * half, gate);
        let flush = |bv: f64| (1.0 + bv.abs()) * FLOOR;
        let mut out_bound = vec![0.0; pairs];
        let mut dx_bound = vec![0.0; 2 * pairs];
        for i in 0..pairs {
            let (av, bv, gv) = (f64::from(a[i]), f64::from(b[i]), f64::from(g[i]));
            let (v, e, gs, ds) = gate_bounds(gate, bv);
            let k = (i / half) * 2 * half + i % half;
            out_bound[i] = av.abs() * (v + flush(bv)) + 2.0 * U * (av * gs).abs();
            dx_bound[k] = gv.abs() * (v + flush(bv)) + 2.0 * U * (gv * gs).abs();
            dx_bound[k + half] = (gv * av).abs() * (e + flush(bv)) + 3.0 * U * (gv * av * ds).abs();
        }
        let what = format!("{card:?}");
        let w = 2 * half;
        let y = cuda_run(&cuda, &kernels, GluOp::Forward, card, w, &x, None);
        let worst = assert_within(&y, &want, |i| out_bound[i], &format!("{what} forward"));
        println!("{what} forward: worst {:.3} of its bound", worst.ratio());
        let again = cuda_run(&cuda, &kernels, GluOp::Forward, card, w, &x, None);
        assert_exact(&again, &y, &format!("{what} forward, a second run"));
        let dx = cuda_run(&cuda, &kernels, GluOp::Backward, card, w, &x, Some(&g));
        let worst = assert_within(&dx, &grad, |i| dx_bound[i], &format!("{what} backward"));
        println!("{what} backward: worst {:.3} of its bound", worst.ratio());
        let again = cuda_run(&cuda, &kernels, GluOp::Backward, card, w, &x, Some(&g));
        assert_exact(&again, &dx, &format!("{what} backward, a second run"));
    }
}

/// On ±0, ±∞ and NaN, in either half, every gate gives the `f32` tape's bits forward and
/// backward (a NaN matching any NaN).
#[test]
fn every_gate_on_special_values() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Glu::new(&cuda).unwrap();
    let firsts = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.5,
    ];
    let seconds = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let mut x = Vec::new();
    for &a in &firsts {
        for &b in &seconds {
            x.extend([a, b]);
        }
    }
    let g: Vec<f32> = (0..x.len() / 2)
        .map(|i| if i % 2 == 0 { 1.5 } else { -0.75 })
        .collect();
    for (gate, card) in GATES {
        let (y, dx) = tape::<f32>(&x, &g, 2, gate);
        let got = cuda_run(&cuda, &kernels, GluOp::Forward, card, 2, &x, None);
        assert_exact(&got, &y, &format!("{card:?} forward"));
        let got = cuda_run(&cuda, &kernels, GluOp::Backward, card, 2, &x, Some(&g));
        assert_exact(&got, &dx, &format!("{card:?} backward"));
    }
}

/// The refusals: a gradient missing or given against the pass, an odd width, a shape too
/// large for the kernels' indices, and a buffer too short.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Glu::new(&cuda).unwrap();
    let x = upload(&cuda, &[0.0f32; 8]);
    let mut out = nans(&cuda, 8);
    let relu = GluGate::Relu;
    let err = kernels
        .run(GluOp::Backward, relu, &x, None, &mut out, 2, 4)
        .unwrap_err();
    assert!(
        matches!(err, CudaError::Invalid(ref m) if m.contains("needs")),
        "{err}"
    );
    let err = kernels
        .run(GluOp::Forward, relu, &x, Some(&x), &mut out, 2, 4)
        .unwrap_err();
    assert!(
        matches!(err, CudaError::Invalid(ref m) if m.contains("takes no")),
        "{err}"
    );
    let err = kernels
        .run(GluOp::Forward, relu, &x, None, &mut out, 1, 5)
        .unwrap_err();
    assert!(err.to_string().contains("must be even"), "{err}");
    let err = kernels
        .run(GluOp::Forward, relu, &x, None, &mut out, 1 << 30, 8)
        .unwrap_err();
    assert!(err.to_string().contains("do not fit a u32"), "{err}");
    let text = panic_message(|| {
        kernels
            .run(GluOp::Forward, relu, &x, None, &mut out, 3, 4)
            .unwrap();
    });
    assert!(
        text.contains("input buffer") && text.contains("12 values"),
        "{text}"
    );
    let short = upload(&cuda, &[0.0f32; 2]);
    let text = panic_message(|| {
        kernels
            .run(GluOp::Backward, relu, &x, Some(&short), &mut out, 2, 4)
            .unwrap();
    });
    assert!(text.contains("gradient buffer"), "{text}");
}
