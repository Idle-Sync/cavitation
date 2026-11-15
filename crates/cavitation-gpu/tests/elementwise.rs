//! The element-wise kernels against their CPU twins on every backend this machine has.
//! The one-rounding operations (add, multiply, scale, ReLU and leaky ReLU forward and
//! backward, dropout) give the CPU's bits on normal values, NaN, ±0 and ±∞ included. tanh,
//! the sigmoid and GELU (both forms), forward and backward, are within their stated bounds
//! of the f64 tape's values. Each test skips without a GPU, unless `CAVITATION_GPU_REQUIRED=1`
//! makes that a failure.

mod common;

use cavitation_core::{GeluApprox, Graph, ParamStore, Tensor};
use cavitation_gpu::kernels::{Elementwise, ElementwiseKernels};
use common::{Kit, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values};

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
        a[..SPECIAL.len()].copy_from_slice(&SPECIAL);
        b[SPECIAL.len()..2 * SPECIAL.len()].copy_from_slice(&SPECIAL);
        for (i, &s) in SPECIAL.iter().enumerate() {
            b[i] = [2.0, -1.0][i % 2];
            a[SPECIAL.len() + i] = s;
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

/// Dropout keeps exactly the elements whose mask bits are set, scaled by the f32 `1/(1 − p)`
/// in one rounding, and zeroes the rest.
#[test]
fn dropout_follows_its_mask() {
    for_each_gpu(|gpu| {
        let n = 100_003;
        let mut kit = Kit::new(gpu, 4 * n as u64);
        let kernels = ElementwiseKernels::new(gpu, &mut kit.cache).expect("build");
        let x = seeded_values(3, n, -4.0, 4.0);
        let mut rng = common::Seeded::new(4);
        let words: Vec<u32> = (0..n.div_ceil(32)).map(|_| rng.next_u64() as u32).collect();
        let mask_as_f32: Vec<f32> = words.iter().map(|w| f32::from_bits(*w)).collect();
        let scale = (1.0f64 / (1.0 - 0.3)) as f32;
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
                if (words[i / 32] >> (i % 32)) & 1 == 1 {
                    x[i] * scale
                } else {
                    0.0
                }
            })
            .collect();
        assert_exact(&got, &want, "dropout");
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
            let size_f: Vec<f64> = x
                .iter()
                .map(|&v| 0.5 * f64::from(v).abs() * 2.0 + f64::from(v).powi(2))
                .collect();
            let size_b: Vec<f64> = x
                .iter()
                .zip(&g)
                .map(|(&v, &gv)| {
                    f64::from(gv).abs()
                        * (1.0
                            + 2.0 * f64::from(v).abs()
                            + f64::from(v).powi(2) * (1.0 + 0.2 * f64::from(v).powi(2)))
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
