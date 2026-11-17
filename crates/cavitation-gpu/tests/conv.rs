//! The GPU convolution against its CPU twins in f64 on the same f32 inputs, over the CPU
//! tests' geometries, the CIFAR and MNIST networks' convolutions, and shapes that span several
//! tiles:
//!
//! - **Forward pass:** against `conv2d_reference`, bit for bit on small integers and within
//!   the GEMM's bound `2(k + 3) · u · (Σ|w·x| + |bias|)` on seeded values.
//! - **Input gradient:** against `conv2d_backward_reference`'s `dx`, bit for bit on small
//!   integers and within `2(k + 3) · u · Σ|w·dy|`, `k` = out·kh·kw, on seeded values.
//! - **Weight gradient:** against `conv2d_backward_reference`'s `dw`, bit for bit on small
//!   integers, at the CIFAR network's full batch of 128 too, and within the blocked sum's
//!   bound `2(L + B + 2) · u · Σ|dy·x|` on seeded values.

mod common;

use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_backward_reference, conv2d_reference};
use cavitation_gpu::kernels::{
    ConvDims, ConvForward, ConvInputGrad, ConvWeightGrad, WEIGHT_GRAD_BLOCK,
};
use common::{Kit, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values};

/// The CPU geometry with the same shape as `dims`.
fn geom(d: &ConvDims) -> Conv2dGeom {
    let u = |p: (u32, u32)| (p.0 as usize, p.1 as usize);
    Conv2dGeom::with_options(
        d.in_channels as usize,
        d.out_channels as usize,
        u(d.input),
        u(d.kernel),
        u(d.stride),
        u(d.padding),
        u(d.dilation),
    )
    .expect("a valid convolution")
}

/// A shape from the CPU tests' tuple order: (in, out, input, kernel, stride, padding,
/// dilation), at `batch` images.
type Shape = (
    u32,
    u32,
    (u32, u32),
    (u32, u32),
    (u32, u32),
    (u32, u32),
    (u32, u32),
);

/// The convolution of `s` over `batch` images.
fn dims(batch: u32, s: Shape) -> ConvDims {
    let (cin, cout, input, kernel, stride, padding, dilation) = s;
    ConvDims {
        stride,
        padding,
        dilation,
        ..ConvDims::new(batch, cin, cout, input, kernel)
    }
}

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

/// Shapes past one tile in every direction: more than 64 output channels, more than 64 input
/// channels (the input gradient's rows), an image of more than 64 positions whose rows do not
/// divide 64, and a k that is not a multiple of 16.
const SPANNING: [Shape; 4] = [
    (5, 130, (11, 13), (3, 3), (1, 1), (1, 1), (1, 1)),
    (130, 5, (11, 13), (3, 3), (1, 1), (1, 1), (1, 1)),
    (7, 65, (20, 9), (3, 2), (2, 1), (2, 0), (1, 2)),
    (3, 70, (9, 70), (1, 5), (1, 3), (0, 2), (1, 1)),
];

/// The f64 reference for `x`, `w` and `bias`, and the same convolution of their absolute
/// values (Σ|w·x| + |bias| for each output).
fn reference(d: &ConvDims, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> (Vec<f64>, Vec<f64>) {
    let g = geom(d);
    let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
    let abs = |v: &[f32]| v.iter().map(|&a| f64::from(a).abs()).collect::<Vec<f64>>();
    let batch = d.batch as usize;
    let mut want = vec![0.0; d.out_len() as usize];
    let mut size = vec![0.0; d.out_len() as usize];
    let (bw, ba) = (bias.map(wide), bias.map(abs));
    conv2d_reference(&g, batch, &wide(x), &wide(w), bw.as_deref(), &mut want);
    conv2d_reference(&g, batch, &abs(x), &abs(w), ba.as_deref(), &mut size);
    (want, size)
}

/// The GPU's output for `x`, `w` and `bias`, written over a buffer of NaNs (so every output
/// is shown to be written).
fn run(
    kit: &mut Kit,
    conv: &mut ConvForward,
    d: ConvDims,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let (xb, wb) = (kit.upload(x), kit.upload(w));
    let bb = bias.map(|b| kit.upload(b));
    let out = kit.upload(&vec![f32::NAN; d.out_len() as usize]);
    let call = conv.call(
        &mut kit.steps,
        &mut kit.pool,
        d,
        &xb,
        &wb,
        bb.as_ref(),
        &out,
    );
    conv.ensure_wide(&mut kit.cache, &call)
        .expect("the wide tile builds");
    let mut step = kit.steps.begin();
    conv.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&out);
    call.release(&mut kit.pool);
    for buffer in [xb, wb, out].into_iter().chain(bb) {
        kit.pool.give(buffer);
    }
    got
}

/// A shape whose windows leave input positions uncovered on both axes: stride 3 with
/// dilation 2 down (rows 1, 4 and 7 are never read) and stride 2 across (columns 0, 2, 4 and
/// 6 meet no tap), so their input gradients are 0.
const GAPS: Shape = (2, 3, (9, 8), (2, 1), (3, 2), (0, 1), (2, 1));

/// Shapes, each with its batch, that the GEMM's wide tile computes (at most 32 rows, more
/// than 64 columns and at least 64 wide tiles in the dispatch), where every other shape here
/// runs on the square tile. The rows are the output channels forward and the input channels
/// in the input gradient; the columns the output positions and the input positions.
/// - 8 → 8 channels on 12 × 12 with padding 1, 32 images: 144 positions, 2 wide tiles an
///   image, 64 in all, both ways.
/// - 3 → 20 channels on 11 × 9 with a 3 × 2 kernel and padding (1, 0), 64 images: 88 output
///   positions and 99 input positions, 1 wide tile an image with columns left over.
const WIDE: [(u32, Shape); 2] = [
    (32, (8, 8, (12, 12), (3, 3), (1, 1), (1, 1), (1, 1))),
    (64, (3, 20, (11, 9), (3, 2), (1, 1), (1, 0), (1, 1))),
];

/// Every shape at `batch`, the network shapes at a batch of 4, and the wide-tile shapes.
fn all_shapes() -> Vec<ConvDims> {
    let mut out: Vec<ConvDims> = CPU_GEOMETRIES.iter().map(|&s| dims(3, s)).collect();
    out.extend(NETWORKS.iter().map(|&s| dims(4, s)));
    out.extend(SPANNING.iter().map(|&s| dims(2, s)));
    out.push(dims(2, GAPS));
    out.extend(WIDE.iter().map(|&(batch, s)| dims(batch, s)));
    out
}

/// On values from {-2, -1, 1, 2}, every sum is a small integer, so the GPU gives the f64
/// reference's value exactly, with and without a bias.
#[test]
fn exact_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut conv = ConvForward::new(gpu, &mut kit.cache).expect("build");
        for (s, d) in all_shapes().into_iter().enumerate() {
            let seed = 10 * s as u64;
            let x = exact_values(seed, d.x_len() as usize);
            let w = exact_values(seed + 1, d.weight_len() as usize);
            let b = exact_values(seed + 2, d.out_channels as usize);
            for bias in [None, Some(&b[..])] {
                let (want, _) = reference(&d, &x, &w, bias);
                let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
                let got = run(&mut kit, &mut conv, d, &x, &w, bias);
                assert_exact(
                    &got,
                    &want,
                    &format!("conv exact {d:?}, bias {}", bias.is_some()),
                );
            }
        }
    });
}

/// On seeded values in [-1, 1), each output is within the GEMM bound of the f64 reference.
#[test]
fn rounding_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut conv = ConvForward::new(gpu, &mut kit.cache).expect("build");
        for (s, d) in all_shapes().into_iter().enumerate() {
            let seed = 500 + 10 * s as u64;
            let x = seeded_values(seed, d.x_len() as usize, -1.0, 1.0);
            let w = seeded_values(seed + 1, d.weight_len() as usize, -1.0, 1.0);
            let b = seeded_values(seed + 2, d.out_channels as usize, -1.0, 1.0);
            let (want, size) = reference(&d, &x, &w, Some(&b));
            let got = run(&mut kit, &mut conv, d, &x, &w, Some(&b));
            let k = f64::from(d.in_channels * d.kernel.0 * d.kernel.1);
            let worst = assert_within(
                &got,
                &want,
                |i| 2.0 * (k + 3.0) * U * size[i],
                &format!("conv rounding {d:?}"),
            );
            println!("{d:?}: worst error {:.4} of its bound", worst.ratio());
        }
    });
}

/// The same inputs give the same bits on a second run.
#[test]
fn runs_are_identical() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut conv = ConvForward::new(gpu, &mut kit.cache).expect("build");
        let d = dims(4, NETWORKS[1]);
        let x = seeded_values(7, d.x_len() as usize, -1.0, 1.0);
        let w = seeded_values(8, d.weight_len() as usize, -1.0, 1.0);
        let first = run(&mut kit, &mut conv, d, &x, &w, None);
        let second = run(&mut kit, &mut conv, d, &x, &w, None);
        let bits = |v: &[f32]| v.iter().map(|a| a.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&first), bits(&second));
    });
}

/// Misuse is refused when the convolution is prepared: a window that does not fit, a zero
/// stride, a short buffer, and an output that is also an input.
#[test]
fn refusals() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let conv = ConvForward::new(gpu, &mut kit.cache).expect("build");
        let good = ConvDims::new(2, 2, 3, (5, 5), (3, 3));
        let (x, w, out) = (kit.zeros(100), kit.zeros(54), kit.zeros(54));
        let (small, b, tiny) = (kit.zeros(53), kit.zeros(3), kit.zeros(2));
        conv.call(&mut kit.steps, &mut kit.pool, good, &x, &w, Some(&b), &out)
            .release(&mut kit.pool);
        let cases: Vec<(ConvDims, &_, &_, Option<&_>, &_)> = vec![
            (ConvDims::new(2, 2, 3, (2, 2), (3, 3)), &x, &w, None, &out),
            (
                ConvDims {
                    stride: (0, 1),
                    ..good
                },
                &x,
                &w,
                None,
                &out,
            ),
            (
                ConvDims {
                    dilation: (1, 0),
                    ..good
                },
                &x,
                &w,
                None,
                &out,
            ),
            (ConvDims { batch: 0, ..good }, &x, &w, None, &out),
            (good, &x, &small, None, &out),
            (good, &x, &w, None, &small),
            (good, &x, &w, Some(&tiny), &out),
            (good, &x, &w, None, &x),
            (good, &x, &w, Some(&out), &out),
        ];
        for (d, x, w, bias, out) in cases {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                conv.call(&mut kit.steps, &mut kit.pool, d, x, w, bias, out)
            }));
            assert!(result.is_err(), "{d:?} was accepted");
        }
    });
}

/// The f64 input gradient for the weight `w` and the output gradients `dy`, and the same for
/// their absolute values (Σ|w·dy| for each input value).
fn grad_reference(d: &ConvDims, w: &[f32], dy: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let g = geom(d);
    let batch = d.batch as usize;
    let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
    let abs = |v: &[f32]| v.iter().map(|&a| f64::from(a).abs()).collect::<Vec<f64>>();
    let x = vec![0.0; d.x_len() as usize];
    let gradient = |w: &[f64], dy: &[f64]| {
        let mut dx = vec![0.0; d.x_len() as usize];
        let mut dw = vec![0.0; d.weight_len() as usize];
        let mut db = vec![0.0; d.out_channels as usize];
        conv2d_backward_reference(&g, batch, &x, w, dy, &mut dx, &mut dw, &mut db);
        dx
    };
    (gradient(&wide(w), &wide(dy)), gradient(&abs(w), &abs(dy)))
}

/// The GPU's input gradient for `w` and `dy`, written over a buffer of NaNs (so every value
/// is shown to be written).
fn run_grad(
    kit: &mut Kit,
    grad: &mut ConvInputGrad,
    d: ConvDims,
    w: &[f32],
    dy: &[f32],
) -> Vec<f32> {
    let (wb, dyb) = (kit.upload(w), kit.upload(dy));
    let dx = kit.upload(&vec![f32::NAN; d.x_len() as usize]);
    let call = grad.call(&mut kit.steps, &mut kit.pool, d, &wb, &dyb, &dx);
    grad.ensure_wide(&mut kit.cache, &call)
        .expect("the wide tile builds");
    let mut step = kit.steps.begin();
    grad.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&dx);
    call.release(&mut kit.pool);
    for buffer in [wb, dyb, dx] {
        kit.pool.give(buffer);
    }
    got
}

/// On values from {-2, -1, 1, 2}, every input gradient is a small integer, so the GPU gives
/// the f64 reference's value exactly, 0 included where no window covers a position (the
/// `GAPS` shape).
#[test]
fn input_grad_exact_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut grad = ConvInputGrad::new(gpu, &mut kit.cache).expect("build");
        for (s, d) in all_shapes().into_iter().enumerate() {
            let seed = 1000 + 10 * s as u64;
            let w = exact_values(seed, d.weight_len() as usize);
            let dy = exact_values(seed + 1, d.out_len() as usize);
            let (want, _) = grad_reference(&d, &w, &dy);
            let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
            assert_exact(
                &run_grad(&mut kit, &mut grad, d, &w, &dy),
                &want,
                &format!("conv input gradient exact {d:?}"),
            );
        }
    });
}

/// On seeded values in [-1, 1), each input gradient is within `2(k + 3) · u · Σ|w·dy|` of the
/// f64 reference, `k` = out·kh·kw.
#[test]
fn input_grad_rounding_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut grad = ConvInputGrad::new(gpu, &mut kit.cache).expect("build");
        for (s, d) in all_shapes().into_iter().enumerate() {
            let seed = 1500 + 10 * s as u64;
            let w = seeded_values(seed, d.weight_len() as usize, -1.0, 1.0);
            let dy = seeded_values(seed + 1, d.out_len() as usize, -1.0, 1.0);
            let (want, size) = grad_reference(&d, &w, &dy);
            let got = run_grad(&mut kit, &mut grad, d, &w, &dy);
            let k = f64::from(d.out_channels * d.kernel.0 * d.kernel.1);
            let worst = assert_within(
                &got,
                &want,
                |i| 2.0 * (k + 3.0) * U * size[i],
                &format!("conv input gradient rounding {d:?}"),
            );
            println!(
                "input gradient {d:?}: worst error {:.4} of its bound",
                worst.ratio()
            );
        }
    });
}

/// The same inputs give the same input gradient, bit for bit, on a second run.
#[test]
fn input_grad_runs_are_identical() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut grad = ConvInputGrad::new(gpu, &mut kit.cache).expect("build");
        let d = dims(4, NETWORKS[1]);
        let w = seeded_values(9, d.weight_len() as usize, -1.0, 1.0);
        let dy = seeded_values(10, d.out_len() as usize, -1.0, 1.0);
        let first = run_grad(&mut kit, &mut grad, d, &w, &dy);
        let second = run_grad(&mut kit, &mut grad, d, &w, &dy);
        assert_exact(&second, &first, "the input gradient on a second run");
    });
}

/// Misuse of the input gradient is refused when it is prepared: a window that does not fit, a
/// zero stride, short buffers, and an input gradient that is also the weight or `dy`.
#[test]
fn input_grad_refusals() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let grad = ConvInputGrad::new(gpu, &mut kit.cache).expect("build");
        let good = ConvDims::new(2, 2, 3, (5, 5), (3, 3));
        let (w, dy, dx) = (kit.zeros(54), kit.zeros(54), kit.zeros(100));
        let small = kit.zeros(53);
        grad.call(&mut kit.steps, &mut kit.pool, good, &w, &dy, &dx)
            .release(&mut kit.pool);
        let cases: Vec<(ConvDims, &_, &_, &_)> = vec![
            (ConvDims::new(2, 2, 3, (2, 2), (3, 3)), &w, &dy, &dx),
            (
                ConvDims {
                    stride: (0, 1),
                    ..good
                },
                &w,
                &dy,
                &dx,
            ),
            (
                ConvDims {
                    dilation: (1, 0),
                    ..good
                },
                &w,
                &dy,
                &dx,
            ),
            (ConvDims { batch: 0, ..good }, &w, &dy, &dx),
            (
                ConvDims {
                    padding: (1, 1),
                    ..ConvDims::new(1, 1, 1, (0, 0), (1, 1))
                },
                &w,
                &dy,
                &dx,
            ),
            (good, &small, &dy, &dx),
            (good, &w, &small, &dx),
            (good, &w, &dy, &small),
            (good, &dx, &dy, &dx),
            (good, &w, &dx, &dx),
        ];
        for (d, w, dy, dx) in cases {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                grad.call(&mut kit.steps, &mut kit.pool, d, w, dy, dx)
            }));
            assert!(result.is_err(), "{d:?} was accepted");
        }
    });
}

/// Shapes whose positions (batch · oh · ow) meet the weight gradient's block edges: exactly
/// two blocks, and two blocks the second of them short, with an image boundary inside a block.
fn block_edges() -> Vec<ConvDims> {
    let side = (WEIGHT_GRAD_BLOCK as f64).sqrt() as u32;
    vec![
        dims(2, (2, 3, (side, side), (1, 1), (1, 1), (0, 0), (1, 1))),
        dims(3, (2, 3, (27, 27), (3, 3), (1, 1), (0, 0), (1, 1))),
    ]
}

/// The f64 weight gradient for the input images `x` and the output gradients `dy`, and the
/// same for their absolute values (Σ|dy·x| for each weight).
fn weight_grad_reference(d: &ConvDims, x: &[f32], dy: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let g = geom(d);
    let batch = d.batch as usize;
    let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
    let abs = |v: &[f32]| v.iter().map(|&a| f64::from(a).abs()).collect::<Vec<f64>>();
    let w = vec![0.0; d.weight_len() as usize];
    let gradient = |x: &[f64], dy: &[f64]| {
        let mut dx = vec![0.0; d.x_len() as usize];
        let mut dw = vec![0.0; d.weight_len() as usize];
        let mut db = vec![0.0; d.out_channels as usize];
        conv2d_backward_reference(&g, batch, x, &w, dy, &mut dx, &mut dw, &mut db);
        dw
    };
    (gradient(&wide(x), &wide(dy)), gradient(&abs(x), &abs(dy)))
}

/// The GPU's weight gradient for `x` and `dy`, written over a buffer of NaNs (so every value
/// is shown to be written), and the call's blocks and block length.
fn run_weight_grad(
    kit: &mut Kit,
    grad: &mut ConvWeightGrad,
    d: ConvDims,
    x: &[f32],
    dy: &[f32],
) -> (Vec<f32>, u32, u32) {
    let (xb, dyb) = (kit.upload(x), kit.upload(dy));
    let dw = kit.upload(&vec![f32::NAN; d.weight_len() as usize]);
    let call = grad.call(&mut kit.steps, &mut kit.pool, d, &xb, &dyb, &dw);
    grad.ensure_wide(&mut kit.cache, &call)
        .expect("the wide tile builds");
    let (blocks, len) = (call.blocks(), call.block_len());
    let mut step = kit.steps.begin();
    grad.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&dw);
    call.release(&mut kit.pool);
    for buffer in [xb, dyb, dw] {
        kit.pool.give(buffer);
    }
    (got, blocks, len)
}

/// On values from {-2, -1, 1, 2}, every weight gradient is a small integer, so the GPU gives
/// the f64 reference's value exactly, over every shape and the block edges.
#[test]
fn weight_grad_exact_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut grad = ConvWeightGrad::new(gpu, &mut kit.cache).expect("build");
        for (s, d) in all_shapes().into_iter().chain(block_edges()).enumerate() {
            let seed = 2000 + 10 * s as u64;
            let x = exact_values(seed, d.x_len() as usize);
            let dy = exact_values(seed + 1, d.out_len() as usize);
            let (want, _) = weight_grad_reference(&d, &x, &dy);
            let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
            let (got, _, _) = run_weight_grad(&mut kit, &mut grad, d, &x, &dy);
            assert_exact(&got, &want, &format!("conv weight gradient exact {d:?}"));
        }
    });
}

/// At the CIFAR network's batch of 128, the first two convolutions sum 128 · 32 · 32 =
/// 131,072 positions in 128 blocks; on values from {-2, -1, 1, 2} every sum stays below
/// 4 · 131,072 = 2^19, so the GPU still gives the f64 reference's value exactly. The second
/// (32 → 32 channels: 32 rows, 288 columns, 128 blocks of 3 wide tiles) runs on the GEMM's
/// wide tile, the first (27 columns) on the square one.
#[test]
fn weight_grad_exact_at_full_cifar_batch() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 25);
        let mut grad = ConvWeightGrad::new(gpu, &mut kit.cache).expect("build");
        for (s, &shape) in NETWORKS[..2].iter().enumerate() {
            let d = dims(128, shape);
            let seed = 2500 + 10 * s as u64;
            let x = exact_values(seed, d.x_len() as usize);
            let dy = exact_values(seed + 1, d.out_len() as usize);
            let (want, _) = weight_grad_reference(&d, &x, &dy);
            let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
            let (got, blocks, _) = run_weight_grad(&mut kit, &mut grad, d, &x, &dy);
            assert_eq!(blocks, 131_072 / WEIGHT_GRAD_BLOCK);
            assert_exact(&got, &want, &format!("conv weight gradient exact {d:?}"));
        }
    });
}

/// On seeded values in [-1, 1), each weight gradient is within `2(L + B + 2) · u · Σ|dy·x|`
/// of the f64 reference: a block's `L` fused multiply-adds, then `B` blocks added in order.
/// The CIFAR network's first two convolutions at its full batch of 128 are among the shapes.
#[test]
fn weight_grad_rounding_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 25);
        let mut grad = ConvWeightGrad::new(gpu, &mut kit.cache).expect("build");
        let shapes = all_shapes()
            .into_iter()
            .chain(block_edges())
            // At the full batch: the first convolution on the square tile, the second on
            // the wide one.
            .chain([dims(128, NETWORKS[0]), dims(128, NETWORKS[1])]);
        for (s, d) in shapes.enumerate() {
            let seed = 3000 + 10 * s as u64;
            let x = seeded_values(seed, d.x_len() as usize, -1.0, 1.0);
            let dy = seeded_values(seed + 1, d.out_len() as usize, -1.0, 1.0);
            let (want, size) = weight_grad_reference(&d, &x, &dy);
            let (got, blocks, len) = run_weight_grad(&mut kit, &mut grad, d, &x, &dy);
            let n = f64::from(len) + f64::from(blocks);
            let worst = assert_within(
                &got,
                &want,
                |i| 2.0 * (n + 2.0) * U * size[i],
                &format!("conv weight gradient rounding {d:?}"),
            );
            println!(
                "weight gradient {d:?} ({blocks} blocks of {len}): worst error {:.4} of its bound",
                worst.ratio()
            );
        }
    });
}

/// The same inputs give the same weight gradient, bit for bit, on a second run.
#[test]
fn weight_grad_runs_are_identical() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 24);
        let mut grad = ConvWeightGrad::new(gpu, &mut kit.cache).expect("build");
        let d = dims(4, NETWORKS[1]);
        let x = seeded_values(11, d.x_len() as usize, -1.0, 1.0);
        let dy = seeded_values(12, d.out_len() as usize, -1.0, 1.0);
        let (first, _, _) = run_weight_grad(&mut kit, &mut grad, d, &x, &dy);
        let (second, _, _) = run_weight_grad(&mut kit, &mut grad, d, &x, &dy);
        assert_exact(&second, &first, "the weight gradient on a second run");
    });
}

/// Misuse of the weight gradient is refused when it is prepared: a window that does not fit,
/// a zero stride or dilation, no images, short buffers, and a weight gradient that is also `x`
/// or `dy`.
#[test]
fn weight_grad_refusals() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let grad = ConvWeightGrad::new(gpu, &mut kit.cache).expect("build");
        let good = ConvDims::new(2, 2, 3, (5, 5), (3, 3));
        let (x, dy, dw) = (kit.zeros(100), kit.zeros(54), kit.zeros(54));
        let small = kit.zeros(53);
        grad.call(&mut kit.steps, &mut kit.pool, good, &x, &dy, &dw)
            .release(&mut kit.pool);
        let cases: Vec<(ConvDims, &_, &_, &_)> = vec![
            (ConvDims::new(2, 2, 3, (2, 2), (3, 3)), &x, &dy, &dw),
            (
                ConvDims {
                    stride: (0, 1),
                    ..good
                },
                &x,
                &dy,
                &dw,
            ),
            (
                ConvDims {
                    dilation: (1, 0),
                    ..good
                },
                &x,
                &dy,
                &dw,
            ),
            (ConvDims { batch: 0, ..good }, &x, &dy, &dw),
            (good, &small, &dy, &dw),
            (good, &x, &small, &dw),
            (good, &x, &dy, &small),
            (good, &x, &dy, &x),
            (good, &x, &dy, &dy),
        ];
        for (d, x, dy, dw) in cases {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                grad.call(&mut kit.steps, &mut kit.pool, d, x, dy, dw)
            }));
            assert!(result.is_err(), "{d:?} was accepted");
        }
    });
}
