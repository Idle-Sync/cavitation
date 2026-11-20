//! The GLU kernels against the tape's `Tape::glu`, on every backend this machine has, with
//! every gate:
//!
//! - **ReLU gate:** the bits of the `f32` tape's output and of both halves of its input
//!   gradient (one multiplication forward, `g · relu(b)` and `(g · a) · relu′(b)` backward),
//!   on seeded values and on zeros of both signs, infinities, NaN, the extreme normals and ±1,
//!   over shapes from one pair to more pairs than one dispatch dimension holds; a NaN matches
//!   any NaN, and a subnormal result may be flushed. No input is subnormal: a GPU that
//!   flushes reads a positive subnormal `b` as 0 (see `GluKernels`).
//! - **Sigmoid, GELU (both forms) and SiLU gates:** within bounds of the `f64` tape, each
//!   the gate's own bound from `tests/elementwise.rs` carried through the products, with
//!   `G` the gate's value, `D` its derivative and `F = (1 + |b|) · 2⁻¹²⁶` for a flushed
//!   exponential:
//!   - the output `a · G` within `|a| · (V + F) + 2u|a · G|`;
//!   - `a`'s gradient `g · G` within `|g| · (V + F) + 2u|g · G|`;
//!   - `b`'s gradient `(g · a) · D` within `|g · a| · (E + F) + 3u|g · a · D|`;
//!
//!   where `V` and `E` bound the gate's value and derivative: the sigmoid `8u·σ` and `10u·σ`
//!   (`σ` within 4 ulp, which `σ · (1 − σ)` carries with a condition of at most 1, and two
//!   roundings); SiLU `10u·|b|·σ` and `12u·σ·(1 + |b|)`; GELU `4u` and `8u` times the size of
//!   its terms, as `tests/elementwise.rs` derives them. `|G|` and `|D|` are bounded by their
//!   sizes too: `σ` and `σ`; `|b|·σ` and `σ·(1 + |b|)`; `|b|` and 1.2 for GELU.
//!   On ±0, ±∞ and NaN, in either half, every gate gives the `f32` tape's bits (a NaN
//!   matching any NaN).
//!
//! Every output starts as NaN, so a value left unwritten fails, and a second run gives the
//! same bits. The tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Gate, GeluApprox, Graph, ParamStore, Real, Tensor};
use cavitation_gpu::Gpu;
use cavitation_gpu::kernels::{Glu, GluKernels};
use common::{FLOOR, Kit, U, assert_exact, assert_within, for_each_gpu, seeded_values};

/// Every gate.
const GATES: [Gate; 5] = [
    Gate::Sigmoid,
    Gate::Gelu(GeluApprox::Tanh),
    Gate::Gelu(GeluApprox::Exact),
    Gate::Relu,
    Gate::Silu,
];

/// The shapes the ReLU gate's twin runs on the tape, `(rows, width)`: one pair, CIFAR-like
/// rows, many short rows, a few long ones and an odd half.
const SHAPES: [(u32, u32); 5] = [(1, 2), (128, 20), (1000, 6), (7, 2000), (3, 130)];

/// Values the ReLU gate's twin includes besides the seeded ones.
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

/// The GPU's `op` with `gate` over `x` (rows of `width`), with `g` for the backward pass,
/// into an output of NaNs.
fn gpu_run(
    kit: &mut Kit,
    kernels: &GluKernels,
    op: Glu,
    gate: Gate,
    width: usize,
    x: &[f32],
    g: Option<&[f32]>,
) -> Vec<f32> {
    let rows = x.len() / width;
    let written = if op.takes_g() { x.len() } else { x.len() / 2 };
    let out = kit.upload(&vec![f32::NAN; written]);
    let xb = kit.upload(x);
    let gb = g.map(|g| kit.upload(g));
    let call = kernels.call(
        &mut kit.steps,
        &mut kit.pool,
        op,
        gate,
        (rows as u32, width as u32),
        &xb,
        gb.as_ref(),
        &out,
    );
    let mut step = kit.steps.begin();
    kernels.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&out);
    call.release(&mut kit.pool);
    kit.pool.give(out);
    kit.pool.give(xb);
    if let Some(gb) = gb {
        kit.pool.give(gb);
    }
    got
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

/// The `f32` tape's output and input gradient.
fn tape32(x: &[f32], g: &[f32], width: usize, gate: Gate) -> (Vec<f32>, Vec<f32>) {
    tape::<f32>(x, g, width, gate)
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
/// values included, and the same bits on a second run.
#[test]
fn relu_gate_gives_the_tapes_bits() {
    for_each_gpu(|gpu| {
        for (k, (rows, width)) in SHAPES.into_iter().enumerate() {
            let (n, w) = ((rows * width) as usize, width as usize);
            let mut kit = kit(gpu, n);
            let kernels = GluKernels::new(gpu, &mut kit.cache).unwrap();
            let seed = 10 * k as u64;
            let x = with_specials(seed + 1, n);
            let g = seeded_values(seed + 2, n / 2, -1.0, 1.0);
            let (y, dx) = tape32(&x, &g, w, Gate::Relu);
            let what = format!("{rows} × {width}");
            let got = gpu_run(&mut kit, &kernels, Glu::Forward, Gate::Relu, w, &x, None);
            assert_exact(&got, &y, &format!("forward, {what}"));
            let again = gpu_run(&mut kit, &kernels, Glu::Forward, Gate::Relu, w, &x, None);
            assert_exact(&again, &got, &format!("forward, {what}, a second run"));
            let got = gpu_run(
                &mut kit,
                &kernels,
                Glu::Backward,
                Gate::Relu,
                w,
                &x,
                Some(&g),
            );
            assert_exact(&got, &dx, &format!("backward, {what}"));
        }
    });
}

/// More pairs than one dispatch dimension holds (65,535 workgroups of 256), in rows of one
/// pair: the ReLU gate's bits, from the tape's formulas (`a · relu(b)`, `g · relu(b)` and
/// `(g · a) · relu′(b)`, which the shapes above check against the tape itself).
#[test]
fn relu_gate_over_a_split_grid() {
    for_each_gpu(|gpu| {
        let pairs = 65_535 * 256 + 200;
        let n = 2 * pairs;
        let mut kit = kit(gpu, n);
        let kernels = GluKernels::new(gpu, &mut kit.cache).unwrap();
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
        let got = gpu_run(&mut kit, &kernels, Glu::Forward, Gate::Relu, 2, &x, None);
        assert_exact(&got, &y, "forward over a split grid");
        let got = gpu_run(
            &mut kit,
            &kernels,
            Glu::Backward,
            Gate::Relu,
            2,
            &x,
            Some(&g),
        );
        assert_exact(&got, &dx, "backward over a split grid");
    });
}

/// The logistic sigmoid in `f64`.
fn sigma(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// For `gate` at `b`, `(V, E, |G|, |D|)`: the bounds of the GPU gate's value and derivative
/// against the `f64` tape's, before the products, and bounds of the value and the derivative
/// themselves. See the module's documentation.
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
            // The terms' sizes, as `tests/elementwise.rs`'s `gelu_twins` takes them.
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
/// [−8, 8], and the tails where the gates saturate and the exponential flushes, up to ±1e12
/// (GELU's `f32` derivative is NaN from about 1.8e19 on the CPU and the GPU alike).
fn sweep(seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (-640..=640).map(|i| i as f32 / 64.0).collect();
    v.extend(seeded_values(seed, 20_000, -8.0, 8.0));
    v.extend([
        -1e12, -50.0, -20.0, 20.0, 50.0, 1e12, -87.0, -90.0, -104.0, -110.0, 87.0, 104.0, 1e-30,
        -1e-30,
    ]);
    v
}

/// The sigmoid, GELU (both forms) and SiLU gates, forward and backward, within their stated
/// bounds of the `f64` tape, in rows of 7 pairs, and the same bits on a second run.
#[test]
fn smooth_gates_within_their_bounds() {
    for_each_gpu(|gpu| {
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
        let mut kit = kit(gpu, x.len());
        let kernels = GluKernels::new(gpu, &mut kit.cache).unwrap();
        for gate in GATES.into_iter().filter(|&g| g != Gate::Relu) {
            let (want, grad) = tape::<f64>(&x, &g, 2 * half, gate);
            let flush = |bv: f64| (1.0 + bv.abs()) * FLOOR;
            let mut out_bound = vec![0.0; pairs];
            let mut dx_bound = vec![0.0; 2 * pairs];
            for i in 0..pairs {
                let (av, bv, gv) = (f64::from(a[i]), f64::from(b[i]), f64::from(g[i]));
                let (v, e, gs, ds) = gate_bounds(gate, bv);
                let (r, j) = (i / half, i % half);
                let k = r * 2 * half + j;
                out_bound[i] = av.abs() * (v + flush(bv)) + 2.0 * U * (av * gs).abs();
                dx_bound[k] = gv.abs() * (v + flush(bv)) + 2.0 * U * (gv * gs).abs();
                dx_bound[k + half] =
                    (gv * av).abs() * (e + flush(bv)) + 3.0 * U * (gv * av * ds).abs();
            }
            let what = format!("{gate:?}");
            let y = gpu_run(&mut kit, &kernels, Glu::Forward, gate, 2 * half, &x, None);
            let worst = assert_within(&y, &want, |i| out_bound[i], &format!("{what} forward"));
            println!("{what} forward: worst {:.3} of its bound", worst.ratio());
            let again = gpu_run(&mut kit, &kernels, Glu::Forward, gate, 2 * half, &x, None);
            assert_exact(&again, &y, &format!("{what} forward, a second run"));
            let dx = gpu_run(
                &mut kit,
                &kernels,
                Glu::Backward,
                gate,
                2 * half,
                &x,
                Some(&g),
            );
            let worst = assert_within(&dx, &grad, |i| dx_bound[i], &format!("{what} backward"));
            println!("{what} backward: worst {:.3} of its bound", worst.ratio());
            let again = gpu_run(
                &mut kit,
                &kernels,
                Glu::Backward,
                gate,
                2 * half,
                &x,
                Some(&g),
            );
            assert_exact(&again, &dx, &format!("{what} backward, a second run"));
        }
    });
}

/// On ±0, ±∞ and NaN, in either half, every gate gives the `f32` tape's bits forward and
/// backward (a NaN matching any NaN).
#[test]
fn every_gate_on_special_values() {
    for_each_gpu(|gpu| {
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
        let mut kit = kit(gpu, x.len());
        let kernels = GluKernels::new(gpu, &mut kit.cache).unwrap();
        for gate in GATES {
            let (y, dx) = tape32(&x, &g, 2, gate);
            let got = gpu_run(&mut kit, &kernels, Glu::Forward, gate, 2, &x, None);
            assert_exact(&got, &y, &format!("{gate:?} forward"));
            let got = gpu_run(&mut kit, &kernels, Glu::Backward, gate, 2, &x, Some(&g));
            assert_exact(&got, &dx, &format!("{gate:?} backward"));
        }
    });
}

/// A call refuses an odd width, an output that is one of its inputs, a backward pass without
/// its gradient and a buffer too short, naming the kernel.
#[test]
fn calls_refuse_bad_buffers() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, 8);
        let kernels = GluKernels::new(gpu, &mut kit.cache).unwrap();
        let a = kit.upload(&[0.0; 8]);
        let out = kit.upload(&[0.0; 8]);
        let message = |body: &mut dyn FnMut()| {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_err();
            err.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        let (steps, pool) = (&mut kit.steps, &mut kit.pool);
        let relu = Gate::Relu;
        let text = message(&mut || {
            kernels.call(steps, pool, Glu::Forward, relu, (1, 5), &a, None, &out);
        });
        assert!(text.contains("glu") && text.contains("even"), "{text}");
        let text = message(&mut || {
            kernels.call(steps, pool, Glu::Forward, relu, (2, 4), &a, None, &a);
        });
        assert!(text.contains("glu") && text.contains("different"), "{text}");
        let text = message(&mut || {
            kernels.call(steps, pool, Glu::Backward, relu, (2, 4), &a, None, &out);
        });
        assert!(text.contains("none was given"), "{text}");
        let text = message(&mut || {
            kernels.call(steps, pool, Glu::Forward, relu, (3, 4), &a, None, &out);
        });
        assert!(text.contains("needs 12 values"), "{text}");
        let text = message(&mut || {
            kernels.call(
                steps,
                pool,
                Glu::Backward,
                relu,
                (2, 4),
                &a,
                Some(&out),
                &out,
            );
        });
        assert!(text.contains("different"), "{text}");
    });
}
