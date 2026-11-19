//! cuBLAS's fully connected layer against an f64 reference on the same f32 inputs, over the
//! CIFAR and MNIST networks' layers at their training batches and small odd shapes:
//!
//! - every output is written (the outputs start as NaN);
//! - each pass is within its measured tolerance of the reference: `|got − exact| ≤ t · u ·
//!   Σ|terms|`, `u = 2⁻²⁴`, with `t` per pass in [`TOLERANCE`]. cuBLAS publishes no error
//!   bound, so `t` is the worst ratio measured on an NVIDIA GTX 1660 Super with margin, not
//!   a proof;
//! - two runs give the same bits.
//!
//! Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

use cavitation_cuda::{Cuda, DeviceBuffer, Linear, LinearShape};

/// The unit roundoff of f32, 2⁻²⁴.
const U: f64 = 1.0 / 16_777_216.0;

/// The absolute floor of every tolerance, 2⁻¹²⁶, the smallest normal f32.
const FLOOR: f64 = 1.0 / (1u128 << 126) as f64;

/// The measured tolerance `t` of each pass, in units of `u · Σ|terms|`: forward (with the
/// bias), input gradient, weight gradient, bias gradient. Each is about twice the worst
/// ratio measured on an NVIDIA GTX 1660 Super with cuBLAS 13.2.2 over the shapes here:
/// 3.25, 2.85, 4.28 and 1.28.
const TOLERANCE: [(&str, f64); 4] = [
    ("forward", 8.0),
    ("input gradient", 8.0),
    ("weight gradient", 10.0),
    ("bias gradient", 3.0),
];

/// The shapes tested: the CIFAR network's two layers at its batch of 128, the MNIST
/// network's three at 64, a short last batch, and small odd shapes.
fn shapes() -> Vec<LinearShape> {
    let s = |batch, inputs, outputs| LinearShape {
        batch,
        inputs,
        outputs,
    };
    vec![
        s(128, 4096, 256),
        s(128, 256, 10),
        s(64, 400, 120),
        s(64, 120, 84),
        s(64, 84, 10),
        s(16, 4096, 256),
        s(1, 1, 1),
        s(3, 5, 7),
        s(7, 33, 2),
        s(1, 300, 9),
    ]
}

/// `n` values uniform in [-1, 1), from a fixed seed (SplitMix64).
fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The inputs of a shape: x, w, the bias and dy.
fn inputs(s: &LinearShape) -> [Vec<f32>; 4] {
    [
        seeded(11, s.x_len()),
        seeded(12, s.w_len()),
        seeded(13, s.outputs as usize),
        seeded(14, s.y_len()),
    ]
}

/// A buffer holding `data`.
fn upload<'c>(cuda: &'c Cuda, data: &[f32]) -> DeviceBuffer<'c> {
    let mut buffer = cuda.buffer(data.len() * 4).expect("allocate");
    buffer.write(data).expect("write");
    buffer
}

/// A buffer's first `n` floats.
fn read(buffer: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    buffer.read(&mut out).expect("read");
    out
}

/// The four passes' outputs: y (with the bias), dx, dw, db.
fn run(cuda: &Cuda, layer: &Linear<'_>) -> [Vec<f32>; 4] {
    let s = layer.shape();
    let [x, w, b, dy] = inputs(&s);
    let (x, w, b, dy) = (
        upload(cuda, &x),
        upload(cuda, &w),
        upload(cuda, &b),
        upload(cuda, &dy),
    );
    let nans = |n: usize| upload(cuda, &vec![f32::NAN; n]);
    let (mut y, mut dx, mut dw, mut db) = (
        nans(s.y_len()),
        nans(s.x_len()),
        nans(s.w_len()),
        nans(s.outputs as usize),
    );
    layer.forward(&x, &w, Some(&b), &mut y).expect("forward");
    layer.input_grad(&w, &dy, &mut dx).expect("input gradient");
    layer
        .weight_grad(&x, &dy, &mut dw)
        .expect("weight gradient");
    layer.bias_grad(&dy, &mut db).expect("bias gradient");
    [
        read(&y, s.y_len()),
        read(&dx, s.x_len()),
        read(&dw, s.w_len()),
        read(&db, s.outputs as usize),
    ]
}

/// The exact passes in f64 for `s`, each with the same sums over absolute values.
fn reference(s: &LinearShape) -> [(Vec<f64>, Vec<f64>); 4] {
    let (b, i, o) = (s.batch as usize, s.inputs as usize, s.outputs as usize);
    let [x, w, bias, dy] = inputs(s).map(|v| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>());
    let mut y = (vec![0.0; b * o], vec![0.0; b * o]);
    let mut dx = (vec![0.0; b * i], vec![0.0; b * i]);
    let mut dw = (vec![0.0; o * i], vec![0.0; o * i]);
    let mut db = (vec![0.0; o], vec![0.0; o]);
    for r in 0..b {
        for c in 0..o {
            let (mut sum, mut size) = (bias[c], bias[c].abs());
            for k in 0..i {
                let t = x[r * i + k] * w[c * i + k];
                sum += t;
                size += t.abs();
            }
            y.0[r * o + c] = sum;
            y.1[r * o + c] = size;
        }
        for k in 0..i {
            let (mut sum, mut size) = (0.0, 0.0);
            for c in 0..o {
                let t = dy[r * o + c] * w[c * i + k];
                sum += t;
                size += t.abs();
            }
            dx.0[r * i + k] = sum;
            dx.1[r * i + k] = size;
        }
    }
    for c in 0..o {
        for k in 0..i {
            let (mut sum, mut size) = (0.0, 0.0);
            for r in 0..b {
                let t = dy[r * o + c] * x[r * i + k];
                sum += t;
                size += t.abs();
            }
            dw.0[c * i + k] = sum;
            dw.1[c * i + k] = size;
        }
        for r in 0..b {
            db.0[c] += dy[r * o + c];
            db.1[c] += dy[r * o + c].abs();
        }
    }
    [y, dx, dw, db]
}

/// The worst ratio of `|got − want|` to `u · size + FLOOR`, after checking every value is
/// finite (so written over the NaNs it started as).
fn worst_ratio(got: &[f32], (want, size): &(Vec<f64>, Vec<f64>), what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let mut worst = 0.0f64;
    for (n, ((&g, &w), &s)) in got.iter().zip(want).zip(size).enumerate() {
        assert!(g.is_finite(), "{what}: value {n} is {g}, not written");
        worst = worst.max((f64::from(g) - w).abs() / (U * s + FLOOR));
    }
    worst
}

/// Each pass of each shape is within its tolerance of the f64 reference; the worst ratio of
/// each is printed.
#[test]
fn within_measured_tolerance() {
    let Some(cuda) = Cuda::new() else { return };
    let mut worst = [0.0f64; 4];
    for s in shapes() {
        let layer = Linear::new(&cuda, s).expect("set up the layer");
        let got = run(&cuda, &layer);
        let want = reference(&s);
        let r: Vec<f64> = (0..4)
            .map(|p| worst_ratio(&got[p], &want[p], TOLERANCE[p].0))
            .collect();
        println!(
            "{s:?}: worst |got - exact| / (u * sum|terms|): forward {:.2}, input gradient \
             {:.2}, weight gradient {:.2}, bias gradient {:.2}",
            r[0], r[1], r[2], r[3]
        );
        for (p, &(pass, t)) in TOLERANCE.iter().enumerate() {
            assert!(
                r[p] <= t,
                "{pass} of {s:?}: {:.2} u above its tolerance {t}",
                r[p]
            );
            worst[p] = worst[p].max(r[p]);
        }
    }
    println!("worst over all shapes: {worst:.2?}");
}

/// Two runs of each pass give the same bits.
#[test]
fn runs_are_identical() {
    let Some(cuda) = Cuda::new() else { return };
    for s in shapes() {
        let layer = Linear::new(&cuda, s).expect("set up the layer");
        let (a, b) = (run(&cuda, &layer), run(&cuda, &layer));
        for p in 0..4 {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(bits(&a[p]), bits(&b[p]), "{} of {s:?}", TOLERANCE[p].0);
        }
    }
}

/// A zero size is refused when set up; a buffer smaller than its matrix panics before
/// cuBLAS sees it.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let zero = LinearShape {
        batch: 0,
        inputs: 3,
        outputs: 2,
    };
    assert!(Linear::new(&cuda, zero).is_err());
    let s = LinearShape {
        batch: 2,
        inputs: 3,
        outputs: 2,
    };
    let layer = Linear::new(&cuda, s).expect("set up");
    let (x, small) = (
        cuda.buffer(6 * 4).expect("allocate"),
        cuda.buffer(4).expect("allocate"),
    );
    let mut y = cuda.buffer(4 * 4).expect("allocate");
    let text = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = layer.forward(&x, &small, None, &mut y);
    }))
    .expect_err("a small weight panics")
    .downcast_ref::<String>()
    .cloned()
    .unwrap_or_default();
    assert!(text.contains("the weight buffer holds 4 bytes"), "{text}");
}
