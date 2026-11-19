//! cuDNN's convolution against the CPU's f64 reference on the same f32 inputs, over the CPU
//! tests' geometries and the CIFAR and MNIST networks' convolutions:
//!
//! - every output is written (the outputs start as NaN);
//! - each pass is within its measured tolerance of the reference: `|got − exact| ≤ t · u ·
//!   Σ|terms|`, `u = 2⁻²⁴`, with `t` per pass in [`TOLERANCE`]. cuDNN publishes no error
//!   bound, so `t` is the worst ratio measured on an NVIDIA GTX 1660 Super with margin, not
//!   a proof;
//! - two runs give the same bits.
//!
//! Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_backward_reference, conv2d_reference};
use cavitation_cuda::{Conv2d, ConvShape, Cuda, DeviceBuffer};

/// The unit roundoff of f32, 2⁻²⁴.
const U: f64 = 1.0 / 16_777_216.0;

/// The absolute floor of every tolerance, 2⁻¹²⁶, the smallest normal f32.
const FLOOR: f64 = 1.0 / (1u128 << 126) as f64;

/// The measured tolerance `t` of each pass, in units of `u · Σ|terms|`: forward (with the
/// bias), input gradient, weight gradient, bias gradient. Each is about twice the worst
/// ratio measured on an NVIDIA GTX 1660 Super with cuDNN 9.28.1 over the shapes here (the
/// CPU geometries at 3 images, the networks' at 8 and at their training batches): 54.4 (forward, `WINOGRAD_NONFUSED` on 64 → 64 channels at 16 × 16),
/// 3.8, 17.4 (weight gradient, `WINOGRAD_NONFUSED` on 32 → 32 at 32 × 32) and 0.48. A sum
/// of `k` terms added in a loop is within about `(k − 1) · u · Σ|terms|` (576 u for a 3 × 3
/// kernel over 64 channels).
const TOLERANCE: [(&str, f64); 4] = [
    ("forward", 128.0),
    ("input gradient", 8.0),
    ("weight gradient", 40.0),
    ("bias gradient", 1.0),
];

/// A shape in the CPU tests' tuple order: (in, out, input, kernel, stride, padding,
/// dilation).
type Shape = (
    u32,
    u32,
    (u32, u32),
    (u32, u32),
    (u32, u32),
    (u32, u32),
    (u32, u32),
);

/// The geometries of `cavitation-core`'s convolution tests: strides, padding and dilation,
/// rectangular images and kernels, and a 1 × 1 kernel.
const CPU_GEOMETRIES: [Shape; 6] = [
    (2, 3, (5, 5), (3, 3), (1, 1), (0, 0), (1, 1)),
    (1, 2, (6, 7), (3, 2), (2, 1), (1, 0), (1, 1)),
    (3, 2, (7, 7), (3, 3), (1, 1), (2, 2), (2, 2)),
    (2, 2, (4, 6), (1, 1), (1, 1), (0, 0), (1, 1)),
    (2, 1, (5, 4), (2, 3), (2, 2), (1, 1), (1, 1)),
    (1, 1, (3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
];

/// The CIFAR network's four convolutions (3 × 3, padding 1) and the MNIST network's two
/// (5 × 5, the first with padding 2), as `cavitation-bench` builds them.
const NETWORKS: [Shape; 6] = [
    (3, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)),
    (32, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)),
    (32, 64, (16, 16), (3, 3), (1, 1), (1, 1), (1, 1)),
    (64, 64, (16, 16), (3, 3), (1, 1), (1, 1), (1, 1)),
    (1, 6, (28, 28), (5, 5), (1, 1), (2, 2), (1, 1)),
    (6, 16, (14, 14), (5, 5), (1, 1), (0, 0), (1, 1)),
];

/// The convolution of `s` over `batch` images.
fn shape(batch: u32, s: Shape) -> ConvShape {
    let (cin, cout, input, kernel, stride, padding, dilation) = s;
    ConvShape {
        stride,
        padding,
        dilation,
        ..ConvShape::new(batch, cin, cout, input, kernel)
    }
}

/// The networks' convolutions at the batches `cavitation-bench` trains them with: 128 for
/// the CIFAR network and 64 for the MNIST network. cuDNN's heuristics pick other algorithms
/// there than at 8 images.
fn training_shapes() -> Vec<ConvShape> {
    NETWORKS[..4]
        .iter()
        .map(|&s| shape(128, s))
        .chain(NETWORKS[4..].iter().map(|&s| shape(64, s)))
        .collect()
}

/// Every small shape tested: the CPU geometries at 3 images, the networks' at 8.
fn all_shapes() -> Vec<ConvShape> {
    CPU_GEOMETRIES
        .iter()
        .map(|&s| shape(3, s))
        .chain(NETWORKS.iter().map(|&s| shape(8, s)))
        .collect()
}

/// The CPU geometry of `s`.
fn geom(s: &ConvShape) -> Conv2dGeom {
    let u = |p: (u32, u32)| (p.0 as usize, p.1 as usize);
    Conv2dGeom::with_options(
        s.in_channels as usize,
        s.out_channels as usize,
        u(s.input),
        u(s.kernel),
        u(s.stride),
        u(s.padding),
        u(s.dilation),
    )
    .expect("a valid convolution")
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

/// `v` widened to f64, and its absolute values widened.
fn wide(v: &[f32]) -> (Vec<f64>, Vec<f64>) {
    (
        v.iter().map(|&a| f64::from(a)).collect(),
        v.iter().map(|&a| f64::from(a).abs()).collect(),
    )
}

/// The worst ratio of `|got − want|` to `u · size + FLOOR` over all values, after checking
/// every value is finite (so written over the NaNs it started as).
fn worst_ratio(got: &[f32], want: &[f64], size: &[f64], what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let mut worst = 0.0f64;
    for (i, ((&g, &w), &s)) in got.iter().zip(want).zip(size).enumerate() {
        assert!(g.is_finite(), "{what}: value {i} is {g}, not written");
        worst = worst.max((f64::from(g) - w).abs() / (U * s + FLOOR));
    }
    worst
}

/// A buffer holding `data`.
fn upload<'c>(cuda: &'c Cuda, data: &[f32]) -> DeviceBuffer<'c> {
    let mut buffer = cuda.buffer(data.len() * 4).expect("allocate");
    buffer.write(data).expect("write");
    buffer
}

/// A buffer of `n` NaNs.
fn nans(cuda: &Cuda, n: usize) -> DeviceBuffer<'_> {
    upload(cuda, &vec![f32::NAN; n])
}

/// A buffer's first `n` floats.
fn read(buffer: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    buffer.read(&mut out).expect("read");
    out
}

/// The four passes' outputs for one shape, from fixed inputs.
struct Run {
    /// The forward pass with the bias.
    y: Vec<f32>,
    /// The input gradient.
    dx: Vec<f32>,
    /// The weight gradient.
    dw: Vec<f32>,
    /// The bias gradient.
    db: Vec<f32>,
}

/// The inputs of a shape: x, w, the bias and dy.
fn inputs(s: &ConvShape) -> [Vec<f32>; 4] {
    [
        seeded(1, s.x_len()),
        seeded(2, s.w_len()),
        seeded(3, s.out_channels as usize),
        seeded(4, s.y_len()),
    ]
}

/// Runs the four passes of `s` on the card.
fn run(cuda: &Cuda, conv: &Conv2d<'_>) -> Run {
    let s = conv.shape();
    let [x, w, b, dy] = inputs(&s);
    let (x, w, b, dy) = (
        upload(cuda, &x),
        upload(cuda, &w),
        upload(cuda, &b),
        upload(cuda, &dy),
    );
    let mut y = nans(cuda, s.y_len());
    let mut dx = nans(cuda, s.x_len());
    let mut dw = nans(cuda, s.w_len());
    let mut db = nans(cuda, s.out_channels as usize);
    conv.forward(&x, &w, Some(&b), &mut y).expect("forward");
    conv.input_grad(&w, &dy, &mut dx).expect("input gradient");
    conv.weight_grad(&x, &dy, &mut dw).expect("weight gradient");
    conv.bias_grad(&dy, &mut db).expect("bias gradient");
    Run {
        y: read(&y, s.y_len()),
        dx: read(&dx, s.x_len()),
        dw: read(&dw, s.w_len()),
        db: read(&db, s.out_channels as usize),
    }
}

/// The worst ratio of each pass of `s` to its reference, in [`TOLERANCE`]'s order.
fn ratios(cuda: &Cuda, s: ConvShape) -> ([f64; 4], String) {
    let conv = Conv2d::new(cuda, s).expect("set up the convolution");
    let got = run(cuda, &conv);
    let g = geom(&s);
    let batch = s.batch as usize;
    let [x, w, b, dy] = inputs(&s);
    let ((x, xa), (w, wa), (b, ba), (dy, dya)) = (wide(&x), wide(&w), wide(&b), wide(&dy));

    let mut y = vec![0.0; s.y_len()];
    let mut ysize = vec![0.0; s.y_len()];
    conv2d_reference(&g, batch, &x, &w, Some(&b), &mut y);
    conv2d_reference(&g, batch, &xa, &wa, Some(&ba), &mut ysize);

    let backward = |x: &[f64], w: &[f64], dy: &[f64]| {
        let mut dx = vec![0.0; s.x_len()];
        let mut dw = vec![0.0; s.w_len()];
        let mut db = vec![0.0; s.out_channels as usize];
        conv2d_backward_reference(&g, batch, x, w, dy, &mut dx, &mut dw, &mut db);
        (dx, dw, db)
    };
    let (dx, dw, db) = backward(&x, &w, &dy);
    let (dxa, dwa, dba) = backward(&xa, &wa, &dya);
    (
        [
            worst_ratio(&got.y, &y, &ysize, "forward"),
            worst_ratio(&got.dx, &dx, &dxa, "input gradient"),
            worst_ratio(&got.dw, &dw, &dwa, "weight gradient"),
            worst_ratio(&got.db, &db, &dba, "bias gradient"),
        ],
        conv.algorithms().to_string(),
    )
}

/// Each pass of each shape is within its tolerance of the f64 reference; the worst ratio of
/// each is printed with the algorithms cuDNN ran.
#[test]
fn within_measured_tolerance() {
    let Some(cuda) = Cuda::new() else { return };
    let mut worst = [0.0f64; 4];
    for s in all_shapes() {
        let (r, algorithms) = ratios(&cuda, s);
        println!(
            "{s:?}\n  {algorithms}\n  worst |got - exact| / (u * sum|terms|): forward {:.2}, \
             input gradient {:.2}, weight gradient {:.2}, bias gradient {:.2}",
            r[0], r[1], r[2], r[3]
        );
        for (i, &(pass, t)) in TOLERANCE.iter().enumerate() {
            assert!(
                r[i] <= t,
                "{pass} of {s:?}: {:.2} u above its tolerance {t}",
                r[i]
            );
            worst[i] = worst[i].max(r[i]);
        }
    }
    println!("worst over all shapes: {worst:.2?}");
}

/// Two runs of each pass give the same bits, at the small batches and the training ones.
#[test]
fn runs_are_identical() {
    let Some(cuda) = Cuda::new() else { return };
    for s in all_shapes().into_iter().chain(training_shapes()) {
        let conv = Conv2d::new(&cuda, s).expect("set up the convolution");
        let (a, b) = (run(&cuda, &conv), run(&cuda, &conv));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&a.y), bits(&b.y), "forward of {s:?}");
        assert_eq!(bits(&a.dx), bits(&b.dx), "input gradient of {s:?}");
        assert_eq!(bits(&a.dw), bits(&b.dw), "weight gradient of {s:?}");
        assert_eq!(bits(&a.db), bits(&b.db), "bias gradient of {s:?}");
    }
}

/// A shape where no window fits, or a zero size, is refused when set up; a buffer smaller
/// than its tensor panics before cuDNN sees it.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    assert!(Conv2d::new(&cuda, ConvShape::new(1, 1, 1, (2, 2), (3, 3))).is_err());
    assert!(Conv2d::new(&cuda, ConvShape::new(0, 1, 1, (4, 4), (3, 3))).is_err());
    assert!(Conv2d::new(&cuda, ConvShape::new(1, 0, 1, (4, 4), (3, 3))).is_err());
    let conv = Conv2d::new(&cuda, ConvShape::new(1, 1, 1, (4, 4), (3, 3))).expect("set up");
    let small = cuda.buffer(4).expect("allocate");
    let w = cuda.buffer(9 * 4).expect("allocate");
    let mut y = cuda.buffer(4 * 4).expect("allocate");
    let text = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = conv.forward(&small, &w, None, &mut y);
    }))
    .expect_err("a small input panics")
    .downcast_ref::<String>()
    .cloned()
    .unwrap_or_default();
    assert!(text.contains("the input buffer holds 4 bytes"), "{text}");
}

/// The networks' convolutions at their training batches (128 for the CIFAR network, 64 for
/// the MNIST network) are within the same tolerances.
#[test]
fn within_measured_tolerance_at_training_batches() {
    let Some(cuda) = Cuda::new() else { return };
    for s in training_shapes() {
        let (r, algorithms) = ratios(&cuda, s);
        println!(
            "{s:?}\n  {algorithms}\n  worst |got - exact| / (u * sum|terms|): forward {:.2}, \
             input gradient {:.2}, weight gradient {:.2}, bias gradient {:.2}",
            r[0], r[1], r[2], r[3]
        );
        for (i, &(pass, t)) in TOLERANCE.iter().enumerate() {
            assert!(
                r[i] <= t,
                "{pass} of {s:?}: {:.2} u above its tolerance {t}",
                r[i]
            );
        }
    }
}
