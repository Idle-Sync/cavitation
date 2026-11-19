//! The element-wise kernels against their CPU twins, as `cavitation-gpu`'s tests check its
//! WGSL ones. The one-rounding operations (add, multiply, scale, ReLU and leaky ReLU forward
//! and backward) give the CPU's bits on normal values, NaN, ±0 and ±∞ included. tanh, the
//! sigmoid and GELU (both forms), forward and backward, are within the same stated bounds of
//! the f64 tape's values as the WGSL kernels. Each test skips without CUDA, unless
//! `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

use cavitation_core::{GeluApprox, Graph, ParamStore, Tape, Tensor, Var};
use cavitation_cuda::{Cuda, DeviceBuffer, Elementwise, ElementwiseOp};

/// The unit roundoff of f32, 2⁻²⁴.
const U: f64 = 1.0 / 16_777_216.0;

/// The absolute floor of every bound, 2⁻¹²⁶, the smallest normal f32.
const FLOOR: f64 = 1.0 / (1u128 << 126) as f64;

/// Values every one-rounding twin includes besides the generated ones: zeros of both signs,
/// infinities, NaN, the extreme normals, a subnormal, and values around 0.
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
    1.0,
    -1.0,
    0.5,
];

/// SplitMix64 from `seed`, as `u64`s.
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `n` values uniform in `[lo, hi)`.
fn seeded_values(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut next = stream(seed);
    (0..n)
        .map(|_| lo + (hi - lo) * ((next() >> 40) as f32 / (1u64 << 24) as f32))
        .collect()
}

/// `n` values from {-2, -1, 1, 2} and seeded values in [-4, 4).
fn mixed_values(seed: u64, n: usize) -> Vec<f32> {
    let mut next = stream(seed);
    let wide = seeded_values(seed ^ 0xABCD, n, -4.0, 4.0);
    (0..n)
        .map(|i| {
            if i % 2 == 0 {
                [-2.0, -1.0, 1.0, 2.0][(next() >> 62) as usize]
            } else {
                wide[i]
            }
        })
        .collect()
}

/// A buffer holding `data`.
fn upload<'c>(cuda: &'c Cuda, data: &[f32]) -> DeviceBuffer<'c> {
    let mut buffer = cuda.buffer(data.len() * 4).expect("allocate");
    buffer.write(data).expect("write");
    buffer
}

/// Runs `op` over `a` (and `b`) with `alpha`, into a buffer of NaNs, and returns the result.
fn run(
    cuda: &Cuda,
    kernels: &Elementwise<'_>,
    op: ElementwiseOp,
    a: &[f32],
    b: Option<&[f32]>,
    alpha: f32,
) -> Vec<f32> {
    let ab = upload(cuda, a);
    let bb = b.map(|b| upload(cuda, b));
    let mut out = upload(cuda, &vec![f32::NAN; a.len()]);
    kernels
        .run(op, &ab, bb.as_ref(), &mut out, a.len(), alpha)
        .expect("run");
    let mut got = vec![0.0f32; a.len()];
    out.read(&mut got).expect("read");
    got
}

/// Asserts `got` has `want`'s bits, a NaN of any payload matching a NaN.
fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let same = (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits();
        assert!(
            same,
            "{what}: element {i}: got {g:e} ({:#x}), want {w:e} ({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Asserts every element of `got` is within `bound(i) + FLOOR` of `want[i]` (NaN against NaN
/// matching), and returns the worst ratio of error to bound.
fn assert_within(got: &[f32], want: &[f64], bound: impl Fn(usize) -> f64, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        let g = f64::from(g);
        if g.is_infinite() || w.is_infinite() {
            assert_eq!(g, w, "{what}: element {i}");
            continue;
        }
        let ratio = (g - w).abs() / (bound(i) + FLOOR);
        assert!(
            ratio <= 1.0,
            "{what}: element {i}: got {g:e}, want {w:e}, {ratio:.3} of its bound"
        );
        worst = worst.max(ratio);
    }
    worst
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

/// Every one-rounding operation gives the CPU's bits on mixed values, on every pair of
/// special values, and on a length that leaves the last block part empty.
#[test]
fn one_rounding_operations_give_the_cpus_bits() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Elementwise::new(&cuda).expect("compile");
    let n = (1 << 20) + 1000;
    let mut a = mixed_values(1, n);
    let mut b = mixed_values(2, n);
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
    let cases: [(ElementwiseOp, CpuTwin); 7] = [
        (ElementwiseOp::Add, |x, y, _| x + y),
        (ElementwiseOp::Mul, |x, y, _| x * y),
        (ElementwiseOp::Scale, |x, _, al| al * x),
        (ElementwiseOp::Relu, |x, _, _| relu(x)),
        (ElementwiseOp::ReluBackward, |x, g, _| {
            g * relu_derivative(x)
        }),
        (ElementwiseOp::LeakyRelu, |x, _, al| {
            if x > 0.0 || x.is_nan() { x } else { al * x }
        }),
        (ElementwiseOp::LeakyReluBackward, |x, g, al| {
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
            &cuda,
            &kernels,
            op,
            &a,
            op.takes_b().then_some(&b[..]),
            alpha,
        );
        let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| cpu(x, y, alpha)).collect();
        assert_exact(&got, &want, &format!("{op:?}"));
    }
}

/// The f64 tape's values of `f` at `x` and, with `g` flowing back, its gradient.
fn tape_reference(
    x: &[f32],
    g: &[f32],
    f: impl Fn(&mut Tape<'_, f64>, Var) -> Var,
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

/// A dense sweep of [-10, 10] in steps of 1/64, seeded values in [-8, 8], and the tails and
/// the exponential's clamp.
fn sweep(seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (-640..=640).map(|i| i as f32 / 64.0).collect();
    v.extend(seeded_values(seed, 50_000, -8.0, 8.0));
    v.extend([
        -1e12, -50.0, -20.0, 20.0, 50.0, 1e12, -87.0, -90.0, -104.0, -110.0, 87.0, 104.0,
    ]);
    v
}

/// tanh and the sigmoid, forward from the input and backward from the output, within the
/// WGSL kernels' bounds of the f64 tape: the forward values within 8u of their size, the
/// backward products within 4u of the size of their terms.
#[test]
fn tanh_and_sigmoid_twins() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Elementwise::new(&cuda).expect("compile");
    let x = sweep(5);
    let g = seeded_values(6, x.len(), -2.0, 2.0);
    for (forward, backward, name) in [
        (ElementwiseOp::Tanh, ElementwiseOp::TanhBackward, "tanh"),
        (
            ElementwiseOp::Sigmoid,
            ElementwiseOp::SigmoidBackward,
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
        let y = run(&cuda, &kernels, forward, &x, None, 0.0);
        let worst = assert_within(&y, &want, |i| 8.0 * U * want[i].abs(), name);
        println!("{name} forward: worst {worst:.3} of its bound");
        let got = run(&cuda, &kernels, backward, &y, Some(&g), 0.0);
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
        println!("{name} backward: worst {worst:.3} of its bound");
    }
}

/// GELU, tanh and erf forms, forward and backward from the input, within the WGSL kernels'
/// bounds of the f64 tape: `c · u` times the size of the formula's terms (with the inner
/// function's error scaled by its slope), c = 4 forward and 8 backward.
#[test]
fn gelu_twins() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Elementwise::new(&cuda).expect("compile");
    let x = sweep(7);
    let g = seeded_values(8, x.len(), -2.0, 2.0);
    for (approx, forward, backward) in [
        (
            GeluApprox::Tanh,
            ElementwiseOp::GeluTanh,
            ElementwiseOp::GeluTanhBackward,
        ),
        (
            GeluApprox::Exact,
            ElementwiseOp::GeluErf,
            ElementwiseOp::GeluErfBackward,
        ),
    ] {
        let (want, grad) = tape_reference(&x, &g, |t, v| t.gelu(v, approx));
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
                let du = match approx {
                    GeluApprox::Tanh => 0.797_884_560_802_865_4 * (1.0 + 0.134_145 * v * v),
                    _ => 0.0,
                };
                f64::from(gv).abs() * (1.0 + v.abs() * du + (1.0 + v * v) * (1.0 + u * u) * s)
            })
            .collect();
        let y = run(&cuda, &kernels, forward, &x, None, 0.0);
        let worst = assert_within(&y, &want, |i| 4.0 * U * size_f[i], &format!("{forward:?}"));
        println!("{forward:?}: worst {worst:.3} of its bound");
        let gb = run(&cuda, &kernels, backward, &x, Some(&g), 0.0);
        let worst = assert_within(
            &gb,
            &grad,
            |i| 8.0 * U * size_b[i],
            &format!("{backward:?}"),
        );
        println!("{backward:?}: worst {worst:.3} of its bound");
    }
}

/// NaN stays NaN through every activation, and ±∞ and −0 go where the CPU's do.
#[test]
fn activations_keep_nan_and_infinities() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Elementwise::new(&cuda).expect("compile");
    let x = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
    let tanh = run(&cuda, &kernels, ElementwiseOp::Tanh, &x, None, 0.0);
    assert!(tanh[0].is_nan());
    assert_eq!(&tanh[1..], &[1.0, -1.0, -0.0]);
    assert_eq!(tanh[3].to_bits(), (-0.0f32).to_bits());
    let sigmoid = run(&cuda, &kernels, ElementwiseOp::Sigmoid, &x, None, 0.0);
    assert!(sigmoid[0].is_nan());
    assert_eq!(&sigmoid[1..], &[1.0, 0.0, 0.5]);
    for op in [ElementwiseOp::GeluTanh, ElementwiseOp::GeluErf] {
        assert!(
            run(&cuda, &kernels, op, &x, None, 0.0)[0].is_nan(),
            "{op:?}"
        );
    }
}

/// Far in the tails both GELU backward passes take the derivative's limit as the CPU does:
/// exactly 1 for large positive x and 0 for large negative x, up to ±∞, never NaN. NaN still
/// gives NaN.
#[test]
fn gelu_backward_takes_its_limits() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Elementwise::new(&cuda).expect("compile");
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
    for op in [
        ElementwiseOp::GeluTanhBackward,
        ElementwiseOp::GeluErfBackward,
    ] {
        let got = run(&cuda, &kernels, op, &x, Some(&g), 0.0);
        assert_exact(&got, &want, &format!("{op:?} in the tails"));
        let nan = run(&cuda, &kernels, op, &[f32::NAN], Some(&[1.0]), 0.0);
        assert!(nan[0].is_nan(), "{op:?} of NaN");
    }
}

/// A missing or extra `b` is an error; a short buffer panics before the launch; no elements
/// run nothing.
#[test]
fn misuse_is_refused() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Elementwise::new(&cuda).expect("compile");
    let a = upload(&cuda, &[1.0; 8]);
    let mut out = upload(&cuda, &[0.0; 8]);
    assert!(
        kernels
            .run(ElementwiseOp::Add, &a, None, &mut out, 8, 0.0)
            .is_err()
    );
    assert!(
        kernels
            .run(ElementwiseOp::Relu, &a, Some(&a), &mut out, 8, 0.0)
            .is_err()
    );
    kernels
        .run(ElementwiseOp::Relu, &a, None, &mut out, 0, 0.0)
        .expect("nothing to do");
    let text = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = kernels.run(ElementwiseOp::Relu, &a, None, &mut out, 9, 0.0);
    }))
    .expect_err("a short buffer panics")
    .downcast_ref::<String>()
    .cloned()
    .unwrap_or_default();
    assert!(
        text.contains("the first input buffer holds 32 bytes"),
        "{text}"
    );
}
