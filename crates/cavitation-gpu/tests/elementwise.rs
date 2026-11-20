//! The element-wise kernels against their CPU twins on every backend this machine has.
//! The one-rounding operations (add, multiply, scale, ReLU and leaky ReLU forward and
//! backward, dropout) and the clamps (hardtanh and ReLU6, the hard sigmoid and the hard
//! swish, forward and backward) give the CPU's bits on normal values, NaN, ±0 and ±∞
//! included. tanh, the sigmoid, GELU (both forms), ELU, SELU, CELU, SiLU, Mish, softplus,
//! softsign and log-sigmoid, forward and backward, are within their stated bounds of the f64
//! tape's values, and give the CPU's bits on ±0, ±∞ and NaN. Each test skips without a GPU,
//! unless `CAVITATION_GPU_REQUIRED=1` makes that a failure.

mod common;

use cavitation_core::{GeluApprox, Graph, ParamStore, Real, Tape, Tensor, Var};
use cavitation_gpu::kernels::{Elementwise, ElementwiseKernels};
use common::{
    FLOOR, Kit, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values,
};

/// Values every one-rounding twin includes besides the generated ones: zeros of both signs,
/// infinities, NaN, the extreme normals, and values around 0.
const SPECIAL: [f32; 12] = [
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
    0.5,
];

/// Runs `op` over `a` (and `b`) with `alpha` and `scale`, and returns the result.
fn run(
    kit: &mut Kit,
    kernels: &ElementwiseKernels,
    op: Elementwise,
    a: &[f32],
    b: Option<&[f32]>,
    alpha: f32,
    scale: f32,
) -> Vec<f32> {
    let n = a.len();
    let ba = kit.upload(a);
    let bb = b.map(|b| kit.upload(b));
    let out = kit.zeros(n);
    let call = kernels.call(
        &mut kit.steps,
        &mut kit.pool,
        op,
        n as u32,
        alpha,
        scale,
        &ba,
        bb.as_ref(),
        &out,
    );
    let mut step = kit.steps.begin();
    kernels.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&out);
    call.release(&mut kit.pool);
    kit.pool.give(ba);
    kit.pool.give(out);
    if let Some(bb) = bb {
        kit.pool.give(bb);
    }
    got
}

/// A one-rounding operation as the CPU computes it, from the element of a, of b, and alpha.
type CpuTwin = fn(f32, f32, f32) -> f32;

/// The CPU's ReLU: 0 at or below 0 (−0 included), the value otherwise, NaN for NaN.
fn relu(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { x }
}

/// The CPU's ReLU derivative: 1 above 0, 0 at or below, NaN for NaN.
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x <= 0.0 {
        0.0
    } else {
        f32::NAN
    }
}

/// The exact twins: every one-rounding operation gives the CPU's bits on exact values from
/// {-2, -1, 1, 2}, on the special values, and on values over 65,535 × 256 elements (which
/// split the dispatch over two dimensions).
#[test]
fn one_rounding_operations_give_the_cpus_bits() {
    for_each_gpu(|gpu| {
        let n = 65_535 * 256 + 1000;
        let mut kit = Kit::new(gpu, 4 * n as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let mut a = exact_values(1, n);
        let mut b = exact_values(2, n);
        // Every pair of special values, then each special against an ordinary value.
        let k = SPECIAL.len();
        for (i, &x) in SPECIAL.iter().enumerate() {
            for (j, &y) in SPECIAL.iter().enumerate() {
                a[i * k + j] = x;
                b[i * k + j] = y;
            }
            a[k * k + i] = x;
            b[k * k + i] = [2.0, -1.0][i % 2];
        }
        let alpha = 0.25f32;
        let cases: [(Elementwise, CpuTwin); 7] = [
            (Elementwise::Add, |x, y, _| x + y),
            (Elementwise::Mul, |x, y, _| x * y),
            (Elementwise::Scale, |x, _, al| al * x),
            (Elementwise::Relu, |x, _, _| relu(x)),
            (Elementwise::ReluBackward, |x, g, _| g * relu_derivative(x)),
            (Elementwise::LeakyRelu, |x, _, al| {
                if x > 0.0 || x.is_nan() { x } else { al * x }
            }),
            (Elementwise::LeakyReluBackward, |x, g, al| {
                g * if x > 0.0 {
                    1.0
                } else if x <= 0.0 {
                    al
                } else {
                    f32::NAN
                }
            }),
        ];
        for (op, cpu) in cases {
            let got = run(
                &mut kit,
                &kernels,
                op,
                &a,
                op.takes_b().then_some(&b[..]),
                alpha,
                0.0,
            );
            let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| cpu(x, y, alpha)).collect();
            assert_exact(&got, &want, &format!("{op:?}"));
        }
    });
}

/// Dropout multiplies each element by `1/f32(1 − p)` where its mask bit is set and by 0 where it
/// is clear, as the CPU does: a dropped −3 is −0, a dropped NaN or infinity is NaN.
#[test]
fn dropout_follows_its_mask() {
    for_each_gpu(|gpu| {
        let n = 100_003;
        let mut kit = Kit::new(gpu, 4 * n as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let mut x = seeded_values(3, n, -4.0, 4.0);
        x[..SPECIAL.len()].copy_from_slice(&SPECIAL);
        x[SPECIAL.len()] = -3.0;
        let mut rng = common::Seeded::new(4);
        let words: Vec<u32> = (0..n.div_ceil(32)).map(|_| rng.next_u64() as u32).collect();
        let mask_as_f32: Vec<f32> = words.iter().map(|w| f32::from_bits(*w)).collect();
        // The CPU's scale: 1 / f32(1 − p), which differs from f32(1 / (1 − p)) for some p.
        let p = 0.025f32;
        let scale = 1.0 / (1.0 - p);
        // The mask goes up as raw words: Kit::upload copies the bytes unchanged.
        let got = run(
            &mut kit,
            &kernels,
            Elementwise::Dropout,
            &x,
            Some(&mask_as_f32),
            0.0,
            scale,
        );
        let want: Vec<f32> = (0..n)
            .map(|i| {
                x[i] * if (words[i / 32] >> (i % 32)) & 1 == 1 {
                    scale
                } else {
                    0.0
                }
            })
            .collect();
        // The specials meet both a kept and a dropped bit somewhere in the first words.
        assert!(want[..SPECIAL.len() + 1].iter().any(|v| v.is_nan()));
        assert_exact(&got, &want, "dropout");
    });
}

/// A neuron mask selects each element where its bit is set and gives +0 where it is clear,
/// as `Tape::keep_where` does: a cleared −3, −0, NaN or infinity is +0, a kept one keeps its
/// bits.
#[test]
fn keep_where_follows_its_mask() {
    for_each_gpu(|gpu| {
        let n = 100_003;
        let mut kit = Kit::new(gpu, 4 * n as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let mut x = seeded_values(5, n, -4.0, 4.0);
        x[..SPECIAL.len()].copy_from_slice(&SPECIAL);
        x[SPECIAL.len()] = -3.0;
        let mut rng = common::Seeded::new(6);
        let mut words: Vec<u32> = (0..n.div_ceil(32)).map(|_| rng.next_u64() as u32).collect();
        // The specials and the −3 each meet a cleared bit and, in the next word, a set one.
        let specials = SPECIAL.len() + 1;
        assert!(specials <= 32);
        words[0] = 0;
        words[1] = u32::MAX;
        x.copy_within(..specials, 32);
        let mask_as_f32: Vec<f32> = words.iter().map(|w| f32::from_bits(*w)).collect();
        let got = run(
            &mut kit,
            &kernels,
            Elementwise::KeepWhere,
            &x,
            Some(&mask_as_f32),
            0.0,
            0.0,
        );
        let want: Vec<f32> = (0..n)
            .map(|i| {
                if (words[i / 32] >> (i % 32)) & 1 == 1 {
                    x[i]
                } else {
                    0.0
                }
            })
            .collect();
        assert!(want[..specials].iter().all(|v| v.to_bits() == 0));
        assert!(want[32..32 + specials].iter().any(|v| v.is_nan()));
        assert_exact(&got, &want, "keep_where");
    });
}

/// The f64 tape's values of `f` at `x` and, with `g` flowing back, its gradient.
fn tape_reference(
    x: &[f32],
    g: &[f32],
    f: impl Fn(&mut cavitation_core::Tape<'_, f64>, cavitation_core::Var) -> cavitation_core::Var,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut params: ParamStore<f64> = ParamStore::default();
    let xs: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let id = params.add(Tensor::from_vec(xs, &[n]).expect("shape"));
    let gs = Tensor::from_vec(g.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>(), &[n])
        .expect("shape");
    let mut graph: Graph<f64> = Graph::default();
    let mut tape = graph.tape(&params.values);
    let xv = tape.param(id);
    let gv = tape.input(gs.view());
    let y = f(&mut tape, xv);
    let values = tape.value(y).data().to_vec();
    let weighted = tape.mul(y, gv);
    let loss = tape.sum(weighted);
    tape.backward(loss, &mut params.grads);
    drop(tape);
    (values, params.grad(id).data().to_vec())
}

/// A dense sweep of [-10, 10] in steps of 1/64, then seeded values in [-8, 8].
fn sweep(seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (-640..=640).map(|i| i as f32 / 64.0).collect();
    v.extend(seeded_values(seed, 50_000, -8.0, 8.0));
    // The tails and the exponential's clamp: where tanh and GELU saturate, and where the
    // sigmoid's e^-|x| reaches the subnormal range and the clamp at -104. The tails stop at
    // ±1e12: GELU's forward pass is finite everywhere, but its f32 backward formulas give NaN
    // for |x| ≥ 1.8e19 (x² overflows) on the CPU and the GPU alike, while the f64 reference
    // stays finite. Issue #268 gives the backward passes their limits there.
    v.extend([
        -1e12, -50.0, -20.0, 20.0, 50.0, 1e12, -87.0, -90.0, -104.0, -110.0, 87.0, 104.0,
    ]);
    v
}

/// tanh and the sigmoid, forward from the input and backward from the output, within
/// stated bounds of the f64 tape: the forward values within 8u of their size (4 ulp; the CPU
/// kernels are within 3, and the GPU fuses differently), the backward products within 4u of
/// the size of their terms.
#[test]
fn tanh_and_sigmoid_twins() {
    for_each_gpu(|gpu| {
        let x = sweep(5);
        let g = seeded_values(6, x.len(), -2.0, 2.0);
        let mut kit = Kit::new(gpu, 4 * x.len() as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        for (forward, backward, name) in [
            (Elementwise::Tanh, Elementwise::TanhBackward, "tanh"),
            (
                Elementwise::Sigmoid,
                Elementwise::SigmoidBackward,
                "sigmoid",
            ),
        ] {
            let (want, _) = tape_reference(&x, &g, |t, v| {
                if name == "tanh" {
                    t.tanh(v)
                } else {
                    t.sigmoid(v)
                }
            });
            let y = run(&mut kit, &kernels, forward, &x, None, 0.0, 0.0);
            let worst = assert_within(&y, &want, |i| 8.0 * U * want[i].abs(), name);
            println!("{name} forward: worst {:.3} of its bound", worst.ratio());
            let got = run(&mut kit, &kernels, backward, &y, Some(&g), 0.0, 0.0);
            let (reference, bound): (Vec<f64>, Vec<f64>) = y
                .iter()
                .zip(&g)
                .map(|(&yv, &gv)| {
                    let (yv, gv) = (f64::from(yv), f64::from(gv));
                    if name == "tanh" {
                        (gv * (1.0 - yv * yv), 4.0 * U * gv.abs() * (1.0 + yv * yv))
                    } else {
                        (
                            gv * yv * (1.0 - yv),
                            4.0 * U * gv.abs() * yv.abs() * (1.0 + yv.abs()),
                        )
                    }
                })
                .unzip();
            let worst = assert_within(&got, &reference, |i| bound[i], &format!("{name} backward"));
            println!("{name} backward: worst {:.3} of its bound", worst.ratio());
        }
    });
}

/// GELU, tanh and erf forms, forward and backward from the input, within stated bounds of
/// the f64 tape: `c · u` times the size of the formula's terms (with the inner function's
/// error scaled by its slope), c = 4 forward and 8 backward. The size bounds the terms loosely
/// (|x| + x² forward), so the measured errors stay well inside even near the inner functions'
/// worst points (tanh at 4 ulp, erf with WGSL's exp at 3 + 2|x| ulp).
#[test]
fn gelu_twins() {
    for_each_gpu(|gpu| {
        let x = sweep(7);
        let g = seeded_values(8, x.len(), -2.0, 2.0);
        let mut kit = Kit::new(gpu, 4 * x.len() as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        for (approx, forward, backward) in [
            (
                GeluApprox::Tanh,
                Elementwise::GeluTanh,
                Elementwise::GeluTanhBackward,
            ),
            (
                GeluApprox::Exact,
                Elementwise::GeluErf,
                Elementwise::GeluErfBackward,
            ),
        ] {
            let (want, grad) = tape_reference(&x, &g, |t, v| t.gelu(v, approx));
            // The size of the terms: 0.5·|x|·(1 + |inner|) forward, and for the derivative
            // its two terms plus the inner slope times |x|, each bounded loosely by |x| + 1.
            // The size of the terms: |x| for the product and the inner function's error scaled
            // by its slope, |x| · |u| · s(u) with s = sech²(u) (tanh form) or the Gaussian
            // e^(-u²) (erf form), which vanish in the tails; the backward pass adds the
            // derivative's terms on the same pattern.
            let inner = |v: f64| -> (f64, f64) {
                match approx {
                    GeluApprox::Tanh => {
                        let u = 0.797_884_560_802_865_4 * (v + 0.044715 * v.powi(3));
                        (u, 1.0 / u.cosh().powi(2))
                    }
                    _ => {
                        let u = v / std::f64::consts::SQRT_2;
                        (u, (-u * u).exp())
                    }
                }
            };
            let size_f: Vec<f64> = x
                .iter()
                .map(|&v| {
                    let v = f64::from(v);
                    let (u, s) = inner(v);
                    v.abs() + v.abs() * u.abs().max(1.0) * s
                })
                .collect();
            let size_b: Vec<f64> = x
                .iter()
                .zip(&g)
                .map(|(&v, &gv)| {
                    let v = f64::from(v);
                    let (u, s) = inner(v);
                    // tanh's absolute error (a few u, |t| <= 1) reaches the derivative through
                    // 0.5·(1 + t) and 0.5·x·(1 − t²)·du, even where t has saturated to ±1.
                    let du = match approx {
                        GeluApprox::Tanh => 0.797_884_560_802_865_4 * (1.0 + 0.134_145 * v * v),
                        _ => 0.0,
                    };
                    f64::from(gv).abs() * (1.0 + v.abs() * du + (1.0 + v * v) * (1.0 + u * u) * s)
                })
                .collect();
            let y = run(&mut kit, &kernels, forward, &x, None, 0.0, 0.0);
            let worst = assert_within(&y, &want, |i| 4.0 * U * size_f[i], &format!("{forward:?}"));
            println!("{forward:?}: worst {:.3} of its bound", worst.ratio());
            let gb = run(&mut kit, &kernels, backward, &x, Some(&g), 0.0, 0.0);
            let worst = assert_within(
                &gb,
                &grad,
                |i| 8.0 * U * size_b[i],
                &format!("{backward:?}"),
            );
            println!("{backward:?}: worst {:.3} of its bound", worst.ratio());
        }
    });
}

/// NaN stays NaN through every activation, and ±∞ go where the CPU's do.
#[test]
fn activations_keep_nan_and_infinities() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 12);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let x = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let tanh = run(&mut kit, &kernels, Elementwise::Tanh, &x, None, 0.0, 0.0);
        assert!(tanh[0].is_nan());
        assert_eq!(&tanh[1..], &[1.0, -1.0, -0.0]);
        assert_eq!(tanh[3].to_bits(), (-0.0f32).to_bits());
        let sigmoid = run(&mut kit, &kernels, Elementwise::Sigmoid, &x, None, 0.0, 0.0);
        assert!(sigmoid[0].is_nan());
        assert_eq!(&sigmoid[1..], &[1.0, 0.0, 0.5]);
        for op in [Elementwise::GeluTanh, Elementwise::GeluErf] {
            assert!(
                run(&mut kit, &kernels, op, &x, None, 0.0, 0.0)[0].is_nan(),
                "{op:?}"
            );
        }
    });
}

/// Far in the tails, where x² or the tanh argument's slope overflows, both GELU backward
/// passes take the derivative's limit as the CPU does (#268): exactly 1 for large positive x
/// and 0 for large negative x, up to ±∞, so the gradient is g or g · 0, never NaN. NaN still
/// gives NaN.
#[test]
fn gelu_backward_takes_its_limits() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 12);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let big = [1e12f32, 1e13, 1e20, 3e38, f32::MAX, f32::INFINITY];
        let x: Vec<f32> = big.iter().flat_map(|&v| [v, -v]).collect();
        let g: Vec<f32> = (0..x.len())
            .map(|i| if i % 4 < 2 { 1.5 } else { -0.75 })
            .collect();
        let want: Vec<f32> = x
            .iter()
            .zip(&g)
            .map(|(&x, &g)| g * if x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        for op in [Elementwise::GeluTanhBackward, Elementwise::GeluErfBackward] {
            let got = run(&mut kit, &kernels, op, &x, Some(&g), 0.0, 0.0);
            assert_exact(&got, &want, &format!("{op:?} in the tails"));
            let nan = run(&mut kit, &kernels, op, &[f32::NAN], Some(&[1.0]), 0.0, 0.0);
            assert!(nan[0].is_nan(), "{op:?} of NaN");
        }
    });
}

/// A missing `b` and a short buffer are refused when the operation is prepared.
#[test]
fn misuse_is_refused() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 12);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let (a, short) = (kit.zeros(64), kit.zeros(32));
        let missing_b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kernels.call(
                &mut kit.steps,
                &mut kit.pool,
                Elementwise::Add,
                64,
                0.0,
                0.0,
                &a,
                None,
                &a,
            )
        }));
        assert!(missing_b.is_err());
        let too_short = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kernels.call(
                &mut kit.steps,
                &mut kit.pool,
                Elementwise::Relu,
                64,
                0.0,
                0.0,
                &a,
                None,
                &short,
            )
        }));
        assert!(too_short.is_err());
    });
}

/// An activation the CPU trains, with its settings, as the tests below run it.
#[derive(Clone, Copy, Debug)]
enum Act {
    /// ELU with `alpha`.
    Elu(f32),
    /// SELU.
    Selu,
    /// CELU with `alpha`.
    Celu(f32),
    /// SiLU.
    Silu,
    /// Mish.
    Mish,
    /// Softplus with `beta` and the threshold.
    Softplus(f32, f32),
    /// Softsign.
    Softsign,
    /// Hardtanh with `min` and `max` (ReLU6 is `[0, 6]`).
    Hardtanh(f32, f32),
    /// The hard sigmoid.
    Hardsigmoid,
    /// The hard swish.
    Hardswish,
    /// Log-sigmoid.
    LogSigmoid,
}

impl Act {
    /// The forward and backward operations, with their `alpha` and `scale`.
    fn ops(self) -> (Elementwise, Elementwise, f32, f32) {
        match self {
            Act::Elu(alpha) => (Elementwise::Elu, Elementwise::EluBackward, alpha, 0.0),
            Act::Selu => (Elementwise::Selu, Elementwise::SeluBackward, 0.0, 0.0),
            Act::Celu(alpha) => (Elementwise::Celu, Elementwise::CeluBackward, alpha, 0.0),
            Act::Silu => (Elementwise::Silu, Elementwise::SiluBackward, 0.0, 0.0),
            Act::Mish => (Elementwise::Mish, Elementwise::MishBackward, 0.0, 0.0),
            Act::Softplus(beta, threshold) => (
                Elementwise::Softplus,
                Elementwise::SoftplusBackward,
                beta,
                threshold,
            ),
            Act::Softsign => (
                Elementwise::Softsign,
                Elementwise::SoftsignBackward,
                0.0,
                0.0,
            ),
            Act::Hardtanh(min, max) => (
                Elementwise::Hardtanh,
                Elementwise::HardtanhBackward,
                min,
                max,
            ),
            Act::Hardsigmoid => (
                Elementwise::Hardsigmoid,
                Elementwise::HardsigmoidBackward,
                0.0,
                0.0,
            ),
            Act::Hardswish => (
                Elementwise::Hardswish,
                Elementwise::HardswishBackward,
                0.0,
                0.0,
            ),
            Act::LogSigmoid => (
                Elementwise::LogSigmoid,
                Elementwise::LogSigmoidBackward,
                0.0,
                0.0,
            ),
        }
    }

    /// Records the activation on `tape`, as the CPU's layer does.
    fn record<T: Real>(self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let c = |v: f32| T::from_f64(f64::from(v));
        match self {
            Act::Elu(alpha) => tape.elu(x, c(alpha)),
            Act::Selu => tape.selu(x),
            Act::Celu(alpha) => tape.celu(x, c(alpha)),
            Act::Silu => tape.silu(x),
            Act::Mish => tape.mish(x),
            Act::Softplus(beta, threshold) => tape.softplus(x, c(beta), c(threshold)),
            Act::Softsign => tape.softsign(x),
            Act::Hardtanh(min, max) => tape.hardtanh(x, c(min), c(max)),
            Act::Hardsigmoid => tape.hardsigmoid(x),
            Act::Hardswish => tape.hardswish(x),
            Act::LogSigmoid => tape.log_sigmoid(x),
        }
    }
}

/// The tape's values of `act` at `x` and, with `g` flowing back, its gradient, in `T`: the
/// CPU's own `f32` arithmetic, or the `f64` reference.
fn tape_of<T: Real>(act: Act, x: &[f32], g: &[f32]) -> (Vec<T>, Vec<T>) {
    let n = x.len();
    let widen = |v: &[f32]| {
        v.iter()
            .map(|&v| T::from_f64(f64::from(v)))
            .collect::<Vec<T>>()
    };
    let mut params: ParamStore<T> = ParamStore::default();
    let id = params.add(Tensor::from_vec(widen(x), &[n]).expect("shape"));
    let gs = Tensor::from_vec(widen(g), &[n]).expect("shape");
    let mut graph: Graph<T> = Graph::default();
    let mut tape = graph.tape(&params.values);
    let xv = tape.param(id);
    let gv = tape.input(gs.view());
    let y = act.record(&mut tape, xv);
    let values = tape.value(y).data().to_vec();
    let weighted = tape.mul(y, gv);
    let loss = tape.sum(weighted);
    tape.backward(loss, &mut params.grads);
    drop(tape);
    (values, params.grad(id).data().to_vec())
}

/// The CPU's `f32` values and gradients of `act`, as `f32`.
fn cpu_f32(act: Act, x: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    tape_of::<f32>(act, x, g)
}

/// The values every activation twin includes: zeros of both signs, infinities, NaN, the
/// extreme normals, and the breakpoints of the clamps (±3, ±1, 0, 6, the hardtanh bounds
/// below) with their neighbours.
fn special_values() -> Vec<f32> {
    let mut v = SPECIAL.to_vec();
    for p in [-6.0f32, -3.0, -2.5, -1.0, 0.5, 1.0, 3.0, 6.0, 20.0, 10.0] {
        v.extend([p, p.next_up(), p.next_down()]);
    }
    v.extend([
        1e-30, -1e-30, 2.9e-8, -2.9e-8, 3.0e-8, -3.0e-8, -17.5, -17.6,
    ]);
    v
}

/// The piecewise-linear activations, hardtanh (with ReLU6's bounds and two others), the hard
/// sigmoid and the hard swish, forward and backward from the input, give the CPU's bits on
/// the special values and the breakpoints with their neighbours, and on a sweep between.
#[test]
fn clamps_give_the_cpus_bits() {
    for_each_gpu(|gpu| {
        let mut x = special_values();
        x.extend(sweep(9));
        x.extend((0..20_000).map(|i| -3.0 + 6.0 * i as f32 / 20_000.0));
        let g = seeded_values(10, x.len(), -2.0, 2.0);
        let mut kit = Kit::new(gpu, 4 * x.len() as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        for act in [
            Act::Hardtanh(-1.0, 1.0),
            Act::Hardtanh(0.0, 6.0),
            Act::Hardtanh(-2.5, 0.5),
            Act::Hardsigmoid,
            Act::Hardswish,
        ] {
            let (forward, backward, alpha, scale) = act.ops();
            let (want, grad) = cpu_f32(act, &x, &g);
            let y = run(&mut kit, &kernels, forward, &x, None, alpha, scale);
            assert_exact(&y, &want, &format!("{act:?}"));
            let d = run(&mut kit, &kernels, backward, &x, Some(&g), alpha, scale);
            assert_exact(&d, &grad, &format!("{act:?} backward"));
        }
    });
}

/// The `f64` sigmoid.
fn sigma(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// For `act` at `x` with `g` flowing back, the bounds the forward value and the gradient
/// are held to against the `f64` tape, given its value `y`: each `(c, size, extra)`, the
/// bound being `c · u · size + extra`. See [`exponential_activation_twins`].
fn bounds(act: Act, x: f64, y: f64, g: f64) -> [(f64, f64, f64); 2] {
    let g = g.abs();
    let flush = (1.0 + x.abs()) * FLOOR;
    match act {
        Act::Elu(alpha) => {
            let d = if x > 0.0 {
                1.0
            } else {
                f64::from(alpha).abs() * x.exp()
            };
            [(8.0, y.abs(), 0.0), (8.0, g * d, 0.0)]
        }
        Act::Selu => {
            let d = if x > 0.0 { 1.05 } else { 1.76 * x.exp() };
            [(12.0, y.abs(), 0.0), (8.0, g * d, 0.0)]
        }
        Act::Celu(alpha) => {
            let z = x / f64::from(alpha);
            let d = if x > 0.0 {
                1.0
            } else {
                z.exp() * (1.0 + z.abs())
            };
            [(12.0, y.abs(), 0.0), (8.0, g * d, 0.0)]
        }
        Act::Silu => {
            let s = sigma(x);
            [
                (10.0, y.abs(), flush),
                (12.0, g * s * (1.0 + x.abs()), g * flush),
            ]
        }
        Act::Mish => {
            let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
            let (t, s) = (sp.tanh(), sigma(x));
            [
                (16.0, y.abs(), 2.0 * flush),
                (
                    32.0,
                    g * (t.abs() + x.abs() * s * (1.0 + t * t)),
                    2.0 * g * flush,
                ),
            ]
        }
        Act::Softplus(beta, threshold) => {
            let z = f64::from(beta) * x;
            let s = sigma(z);
            let above = z > f64::from(threshold);
            [
                (12.0, y.abs() + x.abs() * s, FLOOR / f64::from(beta).abs()),
                (
                    10.0,
                    if above {
                        0.0
                    } else {
                        g * s * (1.0 + z.abs() * (1.0 - s))
                    },
                    0.0,
                ),
            ]
        }
        Act::Softsign => [
            (8.0, y.abs(), 0.0),
            (12.0, g / (1.0 + x.abs()).powi(2), 0.0),
        ],
        Act::LogSigmoid => [(8.0, y.abs(), 0.0), (10.0, g * sigma(-x), 0.0)],
        Act::Hardtanh(..) | Act::Hardsigmoid | Act::Hardswish => {
            unreachable!("the clamps are exact")
        }
    }
}

/// The activations built on an exponential, a logarithm or a division, forward and backward
/// from the input, within stated bounds of the `f64` tape, each `c · u` times the size of the
/// value or the gradient, plus `extra` where a flushed subnormal is multiplied by `x`:
///
/// - ELU, `8u|y|` (`e^x − 1` within 2 ulp, one more than the CPU's kernel, and the product
///   by `alpha`); backward `8u|g · alpha · e^x|` (`exp_neg` within 2 ulp, two products).
/// - SELU, `12u|y|` (ELU's, and the two constants and the product by the scale rounded to
///   `f32`); backward `8u|g · d|`.
/// - CELU, `12u|y|` (WGSL's division within 2.5 ulp, which `e^y − 1` carries with a
///   relative condition of at most 1, then ELU's); backward `8u|g| · e^z · (1 + |z|)`,
///   `z = x / alpha`, the division's error scaled by `z`.
/// - SiLU, `10u|y|` (the sigmoid within 4 ulp, as its own twin, and the product), and
///   `12u|g| · σ(x) · (1 + |x|)` backward, the sigmoid's error reaching `1 − σ` and `x`.
/// - Mish, `16u|y|` (softplus within 3 ulp and the tanh within 4, both carried with a
///   condition of at most 1, and the product); backward `32u|g| · (|t| + |x|·σ(x)·(1 + t²))`,
///   `t`'s error doubled in `t²`.
/// - Softplus, `12u · (|y| + |x|·σ(z))`, `z = beta · x` (the rounding of `z` scaled by the
///   slope, softplus within 3 ulp and the division by `beta`); backward
///   `10u|g| · σ(z) · (1 + |z|·(1 − σ(z)))`.
/// - Softsign, `8u|y|` (the sum and the division); backward `12u|g| / (1 + |x|)²`.
/// - Log-sigmoid, `8u|y|` (softplus within 3 ulp); backward `10u|g| · σ(−x)`.
///
/// A flushed sigmoid or softplus (`x` beyond about ±87) is off by at most 2⁻¹²⁶, times `x`
/// where it multiplies `x`: the `extra` term, `(1 + |x|) · 2⁻¹²⁶`.
#[test]
fn exponential_activation_twins() {
    for_each_gpu(|gpu| {
        let mut x = sweep(11);
        x.extend((-240..=240).map(|i| i as f32 / 8.0));
        x.extend([
            1e-30, -1e-30, 2.9e-8, -2.9e-8, 3.0e-8, -3.0e-8, -17.5, -17.6,
        ]);
        let g = seeded_values(12, x.len(), -2.0, 2.0);
        let mut kit = Kit::new(gpu, 4 * x.len() as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        for act in [
            Act::Elu(1.0),
            Act::Elu(0.25),
            Act::Selu,
            Act::Celu(1.0),
            Act::Celu(1.5),
            Act::Silu,
            Act::Mish,
            Act::Softplus(1.0, 20.0),
            Act::Softplus(1.5, 10.0),
            Act::Softsign,
            Act::LogSigmoid,
        ] {
            let (forward, backward, alpha, scale) = act.ops();
            let (want, grad) = tape_of::<f64>(act, &x, &g);
            let b: Vec<[(f64, f64, f64); 2]> = (0..x.len())
                .map(|i| bounds(act, f64::from(x[i]), want[i], f64::from(g[i])))
                .collect();
            let y = run(&mut kit, &kernels, forward, &x, None, alpha, scale);
            let worst = assert_within(
                &y,
                &want,
                |i| b[i][0].0 * U * b[i][0].1 + b[i][0].2,
                &format!("{act:?}"),
            );
            println!("{act:?} forward: worst {:.3} of its bound", worst.ratio());
            let d = run(&mut kit, &kernels, backward, &x, Some(&g), alpha, scale);
            let worst = assert_within(
                &d,
                &grad,
                |i| b[i][1].0 * U * b[i][1].1 + b[i][1].2,
                &format!("{act:?} backward"),
            );
            println!("{act:?} backward: worst {:.3} of its bound", worst.ratio());
        }
    });
}

/// On ±0, ±∞ and NaN the activations built on an exponential, a logarithm or a division
/// give the CPU's bits, forward and backward: −0 stays −0 where the CPU keeps it, and NaN
/// stays NaN.
#[test]
fn exponential_activations_on_special_values() {
    for_each_gpu(|gpu| {
        let x = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let g = [1.5f32, -0.75, 1.5, -0.75, 1.0];
        let mut kit = Kit::new(gpu, 1 << 12);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        for act in [
            Act::Elu(1.0),
            Act::Selu,
            Act::Celu(1.5),
            Act::Silu,
            Act::Mish,
            Act::Softplus(1.0, 20.0),
            Act::Softsign,
            Act::LogSigmoid,
        ] {
            let (forward, backward, alpha, scale) = act.ops();
            let (want, grad) = cpu_f32(act, &x, &g);
            let y = run(&mut kit, &kernels, forward, &x, None, alpha, scale);
            assert_exact(&y, &want, &format!("{act:?}"));
            let d = run(&mut kit, &kernels, backward, &x, Some(&g), alpha, scale);
            assert_exact(&d, &grad, &format!("{act:?} backward"));
        }
    });
}
