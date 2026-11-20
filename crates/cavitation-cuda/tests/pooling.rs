//! The pooling kernels against their CPU twins in `cavitation_core::kernels::pooling`, as
//! `cavitation-gpu`'s tests check its WGSL ones. Max pooling forward is compared bit for bit,
//! payloads and subnormals included, with its winners, on windows built to hold the ties and
//! NaNs that decide a winner. Max pooling backward is compared bit for bit on sums that round
//! and on subnormal, infinite and NaN gradients: nothing is flushed. Average and adaptive
//! average pooling, forward and backward, are compared bit for bit with the CPU's `f32`
//! kernels on small integers, on seeded values whose sums round and on special values, and
//! within the WGSL kernels' rounding bounds of an `f64` reference. Each test skips without
//! CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

mod common;

use cavitation_core::Pool2dGeom;
use cavitation_core::kernels::pooling::{
    adaptive_avg_pool2d, adaptive_avg_pool2d_backward, avg_pool2d, avg_pool2d_backward,
    max_pool2d_backward, max_pool2d_reference,
};
use cavitation_cuda::{AdaptiveAvgPool2d, AvgPool2d, Cuda, CudaError, MaxPool2d, PoolShape};
use common::{
    Seeded, U, assert_exact, assert_within, exact_values, nans, panic_message, read, read_u32,
    seeded_values, upload,
};

/// The geometries the max-pooling twins run, as `cavitation-gpu`'s tests: the 2 × 2 stride-2
/// pooling of the CNN benchmarks, overlapping padded windows, `ceil_mode`, dilation, unequal
/// windows and strides, and strides longer than the window, plain and dilated, which leave
/// pixels in no window, each with its batch size.
fn max_geometries() -> Vec<(Pool2dGeom, usize)> {
    let with = Pool2dGeom::with_options;
    vec![
        (Pool2dGeom::new(8, (32, 32), (2, 2)).unwrap(), 4),
        (
            with(3, (9, 11), (3, 3), (1, 1), (1, 1), (1, 1), false).unwrap(),
            2,
        ),
        (
            with(2, (10, 10), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap(),
            3,
        ),
        (
            with(2, (12, 9), (3, 2), (2, 1), (1, 1), (2, 2), false).unwrap(),
            2,
        ),
        (
            with(4, (7, 13), (2, 3), (1, 2), (0, 1), (1, 1), false).unwrap(),
            1,
        ),
        (
            with(2, (11, 11), (2, 2), (3, 3), (1, 1), (1, 1), true).unwrap(),
            2,
        ),
        (
            with(1, (10, 10), (2, 2), (4, 4), (0, 0), (2, 2), false).unwrap(),
            3,
        ),
    ]
}

/// The geometries the average-pooling twins run, with their batch size and divisor rule
/// (`count_include_pad`, `divisor_override`): padded overlapping windows counted both ways,
/// `ceil_mode` with its clipped last window, an override, unequal windows, and a stride
/// longer than the window, which leaves pixels in no window.
fn avg_geometries() -> Vec<(Pool2dGeom, usize, bool, Option<usize>)> {
    let with = Pool2dGeom::with_options;
    vec![
        (Pool2dGeom::new(8, (32, 32), (2, 2)).unwrap(), 4, true, None),
        (
            with(3, (9, 11), (3, 3), (1, 1), (1, 1), (1, 1), false).unwrap(),
            2,
            true,
            None,
        ),
        (
            with(3, (9, 11), (3, 3), (1, 1), (1, 1), (1, 1), false).unwrap(),
            2,
            false,
            None,
        ),
        (
            with(2, (10, 10), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap(),
            3,
            true,
            None,
        ),
        (
            with(2, (10, 10), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap(),
            3,
            false,
            None,
        ),
        (
            with(2, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1), false).unwrap(),
            2,
            true,
            Some(7),
        ),
        // The override wins over `count_include_pad = false` too.
        (
            with(2, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1), false).unwrap(),
            2,
            false,
            Some(7),
        ),
        (
            with(4, (7, 13), (2, 3), (1, 2), (0, 1), (1, 1), false).unwrap(),
            1,
            false,
            None,
        ),
        (
            with(2, (11, 11), (2, 2), (3, 3), (1, 1), (1, 1), true).unwrap(),
            2,
            true,
            None,
        ),
    ]
}

/// `geom` at `batch` as this crate's shape.
fn shape(geom: &Pool2dGeom, batch: usize) -> PoolShape {
    let pair = |(a, b): (usize, usize)| (a as u32, b as u32);
    PoolShape {
        batch: batch as u32,
        channels: geom.channels() as u32,
        input: pair(geom.input()),
        kernel: pair(geom.kernel()),
        stride: pair(geom.stride()),
        padding: pair(geom.padding()),
        dilation: pair(geom.dilation()),
        ceil_mode: geom.ceil_mode(),
    }
}

/// `n` seeded values spread over sixteen binades (`[-1, 1)` scaled by 2⁻⁸ to 2⁷), so a sum
/// of two or more of them almost always rounds and its value depends on the order of the
/// additions.
fn mixed_values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Seeded::new(seed);
    (0..n)
        .map(|_| {
            let scale = 2f32.powi((rng.next_u64() >> 60) as i32 - 8);
            rng.uniform(-1.0, 1.0) * scale
        })
        .collect()
}

/// `values` with every seventh value replaced, in turn, by a quiet NaN, a negative NaN with
/// another payload, both infinities, both zeros and both signs of a subnormal.
fn with_specials(mut values: Vec<f32>) -> Vec<f32> {
    let specials = [
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0002),
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        f32::from_bits(0x0000_1234),
        f32::from_bits(0x8000_4321),
    ];
    for (k, v) in values.iter_mut().step_by(7).enumerate() {
        *v = specials[k % specials.len()];
    }
    values
}

/// Asserts that `got` has exactly `want`'s bits, NaN payloads included.
fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{what}: element {i} is 0x{:08x}, want 0x{:08x}",
            got[i].to_bits(),
            want[i].to_bits()
        );
    }
}

/// The card's max-pooling outputs and winners for `x`, written over NaNs and `u32::MAX`es.
fn cuda_max(cuda: &Cuda, geom: &Pool2dGeom, batch: usize, x: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let pool = MaxPool2d::new(cuda, shape(geom, batch)).expect("max pool");
    let n = batch * geom.out_width();
    let xb = upload(cuda, x);
    let (mut out, mut winners) = (nans(cuda, n), upload(cuda, &vec![u32::MAX; n]));
    pool.forward(&xb, &mut out, &mut winners).expect("forward");
    (read(&out, n), read_u32(&winners, n))
}

/// The card's max-pooling input gradient: `dx` plus each of `dy` at its winner.
fn cuda_max_backward(
    cuda: &Cuda,
    geom: &Pool2dGeom,
    batch: usize,
    (dy, winners, dx): (&[f32], &[u32], &[f32]),
) -> Vec<f32> {
    let pool = MaxPool2d::new(cuda, shape(geom, batch)).expect("max pool");
    let (dyb, wb, mut dxb) = (upload(cuda, dy), upload(cuda, winners), upload(cuda, dx));
    pool.backward(&dyb, &wb, &mut dxb).expect("backward");
    read(&dxb, dx.len())
}

/// The CPU reference's max-pooling outputs and winners for `x`.
fn cpu_max(geom: &Pool2dGeom, batch: usize, x: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let n = batch * geom.out_width();
    let (mut out, mut winners) = (vec![0.0; n], vec![0usize; n]);
    max_pool2d_reference(geom, batch, x, &mut out, Some(&mut winners));
    (out, winners.iter().map(|&w| w as u32).collect())
}

/// The CPU's max-pooling input gradient: `dx` plus each of `dy` at its winner.
fn cpu_max_backward(
    geom: &Pool2dGeom,
    batch: usize,
    (dy, winners, dx): (&[f32], &[u32], &[f32]),
) -> Vec<f32> {
    let winners: Vec<usize> = winners.iter().map(|&w| w as usize).collect();
    let mut dx = dx.to_vec();
    max_pool2d_backward(geom, batch, dy, &winners, &mut dx);
    dx
}

/// Max pooling forward on windows of three built to decide each rule a winner depends on,
/// with the CPU reference's bits and winners: of two NaNs the later wins; a NaN after a
/// number wins and a number after a NaN does not; of -0 and +0 the first wins either way; a
/// positive subnormal beats +0 and the larger of two subnormals wins; infinities order with
/// the numbers; equal values go to the first.
#[test]
fn max_pool_forward_decides_ties_and_nans_as_the_cpu() {
    let Some(cuda) = Cuda::new() else { return };
    let nan_a = f32::from_bits(0x7fc0_0001);
    let nan_b = f32::from_bits(0xffc0_0002);
    let sub_small = f32::from_bits(0x0000_0010);
    let sub_large = f32::from_bits(0x0000_1000);
    let windows: [[f32; 3]; 12] = [
        [nan_a, 1.0, nan_b],
        [1.0, nan_a, 2.0],
        [nan_b, -1.0, 5.0],
        [-0.0, 0.0, -1.0],
        [0.0, -0.0, -1.0],
        [0.0, sub_small, -0.0],
        [sub_small, sub_large, sub_small],
        [-sub_large, -sub_small, -0.0],
        [f32::NEG_INFINITY, -1.0, f32::INFINITY],
        [f32::INFINITY, nan_a, 3.0],
        [2.0, 2.0, 2.0],
        [f32::NEG_INFINITY, f32::NEG_INFINITY, -f32::MAX],
    ];
    let x: Vec<f32> = windows.iter().flatten().copied().collect();
    let geom = Pool2dGeom::new(1, (1, x.len()), (1, 3)).unwrap();
    let (out, winners) = cuda_max(&cuda, &geom, 1, &x);
    let (want, want_winners) = cpu_max(&geom, 1, &x);
    assert_eq!(want_winners, [2, 4, 6, 9, 12, 16, 19, 23, 26, 28, 30, 35]);
    assert_bits(&out, &want, "max_pool2d special windows");
    assert_eq!(winners, want_winners, "max_pool2d special windows' winners");
}

/// Max pooling forward on small integers (ties everywhere) mixed with NaNs, infinities,
/// signed zeros and subnormals, and on seeded values: the same bits and the same winners as
/// the CPU reference.
#[test]
fn max_pool_forward_twins() {
    let Some(cuda) = Cuda::new() else { return };
    for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
        let s = s as u64;
        let n = batch * geom.in_width();
        for x in [
            with_specials(exact_values(s, n)),
            seeded_values(100 + s, n, -1.0, 1.0),
        ] {
            let (out, winners) = cuda_max(&cuda, &geom, batch, &x);
            let (want, want_winners) = cpu_max(&geom, batch, &x);
            assert_bits(&out, &want, &format!("max_pool2d {geom:?}"));
            assert_eq!(winners, want_winners, "max_pool2d winners {geom:?}");
        }
    }
}

/// Max pooling backward on small integers: each winner's input gradient gets exactly its
/// outputs' gradients, and every other input keeps its own.
#[test]
fn max_pool_backward_exact_twin() {
    let Some(cuda) = Cuda::new() else { return };
    for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
        let s = s as u64;
        let x = exact_values(200 + s, batch * geom.in_width());
        let (_, winners) = cpu_max(&geom, batch, &x);
        let dy = exact_values(300 + s, batch * geom.out_width());
        let dx = exact_values(400 + s, batch * geom.in_width());
        let got = cuda_max_backward(&cuda, &geom, batch, (&dy, &winners, &dx));
        let want = cpu_max_backward(&geom, batch, (&dy, &winners, &dx));
        assert_exact(&got, &want, &format!("max_pool2d backward {geom:?}"));
    }
}

/// Max pooling backward on gradients spread over sixteen binades, so that an input's sum of
/// its own gradient and its outputs' rounds: the card adds them in the CPU's order, so the
/// bits match. The test counts the inputs whose f32 sum differs from the exact one and
/// requires many, so the order is what it checks. The same gradients with NaNs, infinities,
/// signed zeros and subnormals among them give the CPU's bits too: no sum is flushed.
#[test]
fn max_pool_backward_seeded_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let mut rounded = 0;
    for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
        let s = s as u64;
        let x = seeded_values(500 + s, batch * geom.in_width(), -1.0, 1.0);
        let (_, winners) = cpu_max(&geom, batch, &x);
        let dy = mixed_values(600 + s, batch * geom.out_width());
        let dx = mixed_values(700 + s, batch * geom.in_width());
        let got = cuda_max_backward(&cuda, &geom, batch, (&dy, &winners, &dx));
        let want = cpu_max_backward(&geom, batch, (&dy, &winners, &dx));
        assert_exact(&got, &want, &format!("max_pool2d backward {geom:?}"));
        let (dy_s, dx_s) = (with_specials(dy.clone()), with_specials(dx.clone()));
        let got = cuda_max_backward(&cuda, &geom, batch, (&dy_s, &winners, &dx_s));
        let want_s = cpu_max_backward(&geom, batch, (&dy_s, &winners, &dx_s));
        assert_exact(
            &got,
            &want_s,
            &format!("max_pool2d backward with specials {geom:?}"),
        );
        let mut exact: Vec<f64> = dx.iter().map(|&v| f64::from(v)).collect();
        let plane_out = geom.out_width() / geom.channels();
        let plane_in = geom.in_width() / geom.channels();
        for (o, (&g, &w)) in dy.iter().zip(&winners).enumerate() {
            exact[o / plane_out * plane_in + w as usize] += f64::from(g);
        }
        rounded += (0..exact.len())
            .filter(|&i| {
                let wins = (0..dy.len())
                    .filter(|&o| o / plane_out * plane_in + winners[o] as usize == i)
                    .count();
                wins >= 2 && f64::from(want[i]) != exact[i]
            })
            .count();
    }
    println!("max_pool2d backward: {rounded} inputs with two or more rounded additions");
    assert!(rounded >= 100, "only {rounded} multi-gradient sums rounded");
}

/// A 4100 × 4100 image with 1 × 1 windows (16.8M outputs, more than one row of 65,535
/// workgroups held in the WGSL kernels): both passes of both poolings cover every value
/// exactly once.
#[test]
fn both_poolings_cover_a_large_grid() {
    let Some(cuda) = Cuda::new() else { return };
    let geom = Pool2dGeom::new(1, (4100, 4100), (1, 1)).unwrap();
    let x = exact_values(800, geom.in_width());
    let (out, winners) = cuda_max(&cuda, &geom, 1, &x);
    assert_bits(&out, &x, "max_pool2d 1x1");
    assert!(winners.iter().enumerate().all(|(i, &w)| w as usize == i));
    let dy = exact_values(801, geom.out_width());
    let dx = exact_values(802, geom.in_width());
    let want: Vec<f32> = dx.iter().zip(&dy).map(|(a, b)| a + b).collect();
    let got = cuda_max_backward(&cuda, &geom, 1, (&dy, &winners, &dx));
    assert_exact(&got, &want, "max_pool2d 1x1 backward");
    let case = (&geom, 1, true, None);
    assert_exact(&cuda_avg(&cuda, case, &x), &x, "avg_pool2d 1x1");
    assert_exact(
        &cuda_avg_backward(&cuda, case, &dy, &dx),
        &want,
        "avg_pool2d 1x1 backward",
    );
}

/// One average-pooling case: the geometry, the batch and the divisor rule.
type AvgCase<'a> = (&'a Pool2dGeom, usize, bool, Option<usize>);

/// The card's average-pooling outputs for `x`, written over NaNs.
fn cuda_avg(cuda: &Cuda, (geom, batch, pad, divisor): AvgCase<'_>, x: &[f32]) -> Vec<f32> {
    let divisor = divisor.map(|d| d as u32);
    let pool = AvgPool2d::new(cuda, shape(geom, batch), pad, divisor).expect("avg pool");
    let n = batch * geom.out_width();
    let (xb, mut out) = (upload(cuda, x), nans(cuda, n));
    pool.forward(&xb, &mut out).expect("forward");
    read(&out, n)
}

/// The card's average-pooling input gradient: `dx` plus each output gradient's share.
fn cuda_avg_backward(
    cuda: &Cuda,
    (geom, batch, pad, divisor): AvgCase<'_>,
    dy: &[f32],
    dx: &[f32],
) -> Vec<f32> {
    let divisor = divisor.map(|d| d as u32);
    let pool = AvgPool2d::new(cuda, shape(geom, batch), pad, divisor).expect("avg pool");
    let (dyb, mut dxb) = (upload(cuda, dy), upload(cuda, dx));
    pool.backward(&dyb, &mut dxb).expect("backward");
    read(&dxb, dx.len())
}

/// Average pooling, forward and backward, on small integers, on seeded values whose sums
/// round, and on values with NaNs, infinities, signed zeros and subnormals among them: the
/// CPU's `f32` bits on every geometry, since the card sums in the CPU's order and divides
/// correctly rounded (the WGSL kernels meet this only within a bound).
#[test]
fn avg_pool_gives_the_cpus_bits() {
    let Some(cuda) = Cuda::new() else { return };
    for (s, (geom, batch, pad, divisor)) in avg_geometries().into_iter().enumerate() {
        let s = s as u64;
        let case = (&geom, batch, pad, divisor);
        let (ins, outs) = (batch * geom.in_width(), batch * geom.out_width());
        let inputs = [
            (exact_values(900 + s, ins), exact_values(1000 + s, outs)),
            (mixed_values(1100 + s, ins), mixed_values(1200 + s, outs)),
            (
                with_specials(mixed_values(1300 + s, ins)),
                with_specials(mixed_values(1400 + s, outs)),
            ),
        ];
        for (k, (x, dy)) in inputs.into_iter().enumerate() {
            let mut want = vec![0.0; outs];
            avg_pool2d(&geom, batch, &x, pad, divisor, &mut want);
            let what = format!("avg_pool2d input {k} {geom:?} {pad} {divisor:?}");
            assert_exact(&cuda_avg(&cuda, case, &x), &want, &what);
            let dx = x;
            let mut want = dx.clone();
            avg_pool2d_backward(&geom, batch, &dy, pad, divisor, &mut want);
            let got = cuda_avg_backward(&cuda, case, &dy, &dx);
            assert_exact(&got, &want, &format!("{what} backward"));
        }
    }
}

/// Average pooling on seeded values, forward and backward, within the WGSL kernels' bound of
/// an f64 reference of the same f32 inputs: an output summing at most `n = kh · kw` pixels
/// with divisor `d` within `(2(n + 2) + 6) · u · Σ|x| / d`, and an input gradient that at
/// most `m = ⌈kh / sh⌉ · ⌈kw / sw⌉` outputs cover within `(2(m + 3) + 5) · u · (|dx| +
/// Σ|dy / d|)`.
#[test]
fn avg_pool_rounding_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let (mut worst_forward, mut worst_backward) = (0.0f64, 0.0f64);
    for (s, (geom, batch, pad, divisor)) in avg_geometries().into_iter().enumerate() {
        let s = s as u64;
        let case = (&geom, batch, pad, divisor);
        let (kh, kw) = geom.kernel();
        let (sh, sw) = geom.stride();
        let wide = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
        let abs = |v: &[f64]| v.iter().map(|x| x.abs()).collect::<Vec<f64>>();

        let x = seeded_values(1200 + s, batch * geom.in_width(), -1.0, 1.0);
        let outs = batch * geom.out_width();
        let (mut reference, mut scale) = (vec![0.0; outs], vec![0.0; outs]);
        avg_pool2d(&geom, batch, &wide(&x), pad, divisor, &mut reference);
        avg_pool2d(&geom, batch, &abs(&wide(&x)), pad, divisor, &mut scale);
        let n = (kh * kw) as f64;
        let worst = assert_within(
            &cuda_avg(&cuda, case, &x),
            &reference,
            |i| (2.0 * (n + 2.0) + 6.0) * U * scale[i],
            &format!("avg_pool2d {geom:?}"),
        );
        worst_forward = worst_forward.max(worst.ratio());

        let dy = seeded_values(1300 + s, outs, -1.0, 1.0);
        let dx = seeded_values(1400 + s, batch * geom.in_width(), -1.0, 1.0);
        let (mut reference, mut scale) = (wide(&dx), abs(&wide(&dx)));
        avg_pool2d_backward(&geom, batch, &wide(&dy), pad, divisor, &mut reference);
        avg_pool2d_backward(&geom, batch, &abs(&wide(&dy)), pad, divisor, &mut scale);
        let m = (kh.div_ceil(sh) * kw.div_ceil(sw)) as f64;
        let worst = assert_within(
            &cuda_avg_backward(&cuda, case, &dy, &dx),
            &reference,
            |i| (2.0 * (m + 3.0) + 5.0) * U * scale[i],
            &format!("avg_pool2d backward {geom:?}"),
        );
        worst_backward = worst_backward.max(worst.ratio());
    }
    println!(
        "avg_pool2d: worst {worst_forward:.3} of its bound forward, {worst_backward:.3} backward"
    );
}

/// Two runs of every pass on the same inputs give the same bits.
#[test]
fn two_runs_give_the_same_bits() {
    let Some(cuda) = Cuda::new() else { return };
    let (geom, batch) = max_geometries().swap_remove(2);
    let x = mixed_values(1500, batch * geom.in_width());
    let dy = mixed_values(1501, batch * geom.out_width());
    let dx = mixed_values(1502, batch * geom.in_width());
    let (out, winners) = cuda_max(&cuda, &geom, batch, &x);
    assert_eq!((out.clone(), winners.clone()), {
        let again = cuda_max(&cuda, &geom, batch, &x);
        (again.0, again.1)
    });
    let backward = cuda_max_backward(&cuda, &geom, batch, (&dy, &winners, &dx));
    assert_bits(
        &cuda_max_backward(&cuda, &geom, batch, (&dy, &winners, &dx)),
        &backward,
        "max_pool2d backward again",
    );
    let case = (&geom, batch, false, None);
    assert_bits(
        &cuda_avg(&cuda, case, &x),
        &cuda_avg(&cuda, case, &x),
        "avg_pool2d again",
    );
    assert_bits(
        &cuda_avg_backward(&cuda, case, &dy, &dx),
        &cuda_avg_backward(&cuda, case, &dy, &dx),
        "avg_pool2d backward again",
    );
}

/// Shapes the CPU refuses, and a dilated or zero-divisor average pooling, are refused with
/// `CudaError::Invalid`; a short buffer panics before the launch, naming it.
#[test]
fn misuse_is_refused() {
    let Some(cuda) = Cuda::new() else { return };
    let base = PoolShape::new(1, 1, (4, 4), (2, 2));
    let refused = [
        PoolShape { batch: 0, ..base },
        PoolShape {
            padding: (2, 0),
            ..base
        },
        PoolShape {
            kernel: (5, 2),
            ..base
        },
        PoolShape {
            stride: (0, 1),
            ..base
        },
        // Two taps four apart with padding 1 on a 3-pixel-wide image: the last output's taps
        // fall in the padding on both sides (the CPU's own example).
        PoolShape {
            input: (1, 3),
            kernel: (1, 2),
            stride: (1, 1),
            padding: (0, 1),
            dilation: (1, 4),
            ..base
        },
    ];
    for shape in refused {
        assert!(
            matches!(MaxPool2d::new(&cuda, shape), Err(CudaError::Invalid(_))),
            "{shape:?}"
        );
    }
    let dilated = PoolShape {
        dilation: (2, 2),
        ..base
    };
    assert!(MaxPool2d::new(&cuda, dilated).is_ok());
    assert!(matches!(
        AvgPool2d::new(&cuda, dilated, true, None),
        Err(CudaError::Invalid(_))
    ));
    assert!(matches!(
        AvgPool2d::new(&cuda, base, true, Some(0)),
        Err(CudaError::Invalid(_))
    ));

    let pool = MaxPool2d::new(&cuda, base).expect("max pool");
    let x = upload(&cuda, &[0.0f32; 16]);
    let (mut out, mut short) = (nans(&cuda, 4), nans(&cuda, 3));
    let text = panic_message(|| {
        let _ = pool.forward(&x, &mut out, &mut short);
    });
    assert!(
        text.contains("the winners buffer holds 12 bytes, and 4 values need 16"),
        "{text}"
    );
    let avg = AvgPool2d::new(&cuda, base, true, None).expect("avg pool");
    let text = panic_message(|| {
        let _ = avg.backward(&out, &mut short);
    });
    assert!(
        text.contains("the input gradient buffer holds 12 bytes"),
        "{text}"
    );
}

/// One adaptive-pooling case: channels, input size, output size and batch.
type Adaptive = (usize, (usize, usize), (usize, usize), usize);

/// The shapes the adaptive-pooling twins run, as `cavitation-gpu`'s tests: global pooling of
/// the CNN heads, uneven windows that overlap (7 → 3) beside even ones (5 → 2 across), an
/// output larger than its input (4 → 6, pixels shared by two outputs), overlapping windows of
/// 3 (9 → 4), an output equal to its input, and windows of powers of two (6 → 4, 8 → 1).
fn adaptive_cases() -> Vec<Adaptive> {
    vec![
        (8, (32, 32), (1, 1), 2),
        (3, (7, 5), (3, 2), 2),
        (2, (4, 6), (6, 4), 3),
        (4, (9, 9), (4, 4), 1),
        (2, (5, 5), (5, 5), 2),
        (2, (6, 8), (4, 1), 2),
    ]
}

/// The CPU's window along an axis: `floor(o · size / out) .. ceil((o + 1) · size / out)`.
fn adaptive_window(o: usize, out: usize, size: usize) -> std::ops::Range<usize> {
    o * size / out..((o + 1) * size).div_ceil(out)
}

/// Along one axis of `size` pooled to `out`: the longest window, and the most outputs whose
/// windows hold one pixel.
fn adaptive_axis(size: usize, out: usize) -> (usize, usize) {
    let longest = (0..out)
        .map(|o| adaptive_window(o, out, size).len())
        .max()
        .unwrap();
    let covers = (0..size)
        .map(|p| {
            (0..out)
                .filter(|&o| adaptive_window(o, out, size).contains(&p))
                .count()
        })
        .max()
        .unwrap();
    (longest, covers)
}

/// The card's kernels for an adaptive case.
fn adaptive_pool(cuda: &Cuda, (c, input, output, batch): Adaptive) -> AdaptiveAvgPool2d<'_> {
    let pair = |(a, b): (usize, usize)| (a as u32, b as u32);
    AdaptiveAvgPool2d::new(cuda, batch as u32, c as u32, pair(input), pair(output))
        .expect("adaptive pool")
}

/// The input and output lengths of an adaptive case.
fn adaptive_lens((c, (h, w), (oh, ow), batch): Adaptive) -> (usize, usize) {
    (batch * c * h * w, batch * c * oh * ow)
}

/// The card's adaptive-pooling outputs for `x`, written over NaNs.
fn cuda_adaptive(cuda: &Cuda, case: Adaptive, x: &[f32]) -> Vec<f32> {
    let pool = adaptive_pool(cuda, case);
    let (_, outs) = adaptive_lens(case);
    let (xb, mut out) = (upload(cuda, x), nans(cuda, outs));
    pool.forward(&xb, &mut out).expect("forward");
    read(&out, outs)
}

/// The card's adaptive-pooling input gradient: `dx` plus each output gradient's share.
fn cuda_adaptive_backward(cuda: &Cuda, case: Adaptive, dy: &[f32], dx: &[f32]) -> Vec<f32> {
    let pool = adaptive_pool(cuda, case);
    let (dyb, mut dxb) = (upload(cuda, dy), upload(cuda, dx));
    pool.backward(&dyb, &mut dxb).expect("backward");
    read(&dxb, dx.len())
}

/// Adaptive average pooling, forward and backward, on small integers, on seeded values whose
/// sums round, and on values with NaNs, infinities, signed zeros and subnormals among them:
/// the CPU's `f32` bits on every shape, since the card sums in the CPU's order and divides
/// correctly rounded, by the window's height and then its width.
#[test]
fn adaptive_pool_gives_the_cpus_bits() {
    let Some(cuda) = Cuda::new() else { return };
    for (s, case) in adaptive_cases().into_iter().enumerate() {
        let (c, input, output, batch) = case;
        let s = s as u64;
        let (ins, outs) = adaptive_lens(case);
        let inputs = [
            (exact_values(1600 + s, ins), exact_values(1700 + s, outs)),
            (mixed_values(1800 + s, ins), mixed_values(1900 + s, outs)),
            (
                with_specials(mixed_values(2000 + s, ins)),
                with_specials(mixed_values(2100 + s, outs)),
            ),
        ];
        for (k, (x, dy)) in inputs.into_iter().enumerate() {
            let mut want = vec![0.0; outs];
            adaptive_avg_pool2d(batch, c, input, output, &x, &mut want);
            let what = format!("adaptive_avg_pool2d input {k} {case:?}");
            assert_exact(&cuda_adaptive(&cuda, case, &x), &want, &what);
            let dx = x;
            let mut want = dx.clone();
            adaptive_avg_pool2d_backward(batch, c, input, output, &dy, &mut want);
            let got = cuda_adaptive_backward(&cuda, case, &dy, &dx);
            assert_exact(&got, &want, &format!("{what} backward"));
        }
    }
}

/// Adaptive average pooling on seeded values, forward and backward, within the WGSL kernels'
/// bound of an f64 reference of the same f32 inputs (derived in `cavitation-gpu`'s
/// `tests/pooling.rs`): an output summing at most `n` pixels of a window of `dh · dw` within
/// `(2(n + 2) + 11) · u · Σ|x| / (dh · dw)`, and an input gradient that at most `m` outputs
/// cover within `(2(m + 3) + 10) · u · (|dx| + Σ|dy / (dh · dw)|)`.
#[test]
fn adaptive_pool_rounding_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let (mut worst_forward, mut worst_backward) = (0.0f64, 0.0f64);
    for (s, case) in adaptive_cases().into_iter().enumerate() {
        let (c, input, output, batch) = case;
        let s = s as u64;
        let (ins, outs) = adaptive_lens(case);
        let (rows, rows_cover) = adaptive_axis(input.0, output.0);
        let (cols, cols_cover) = adaptive_axis(input.1, output.1);
        let (n, m) = ((rows * cols) as f64, (rows_cover * cols_cover) as f64);
        let wide = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
        let abs = |v: &[f64]| v.iter().map(|x| x.abs()).collect::<Vec<f64>>();
        let what = format!("adaptive_avg_pool2d {case:?}");

        let x = seeded_values(2200 + s, ins, -1.0, 1.0);
        let (mut reference, mut scale) = (vec![0.0; outs], vec![0.0; outs]);
        adaptive_avg_pool2d(batch, c, input, output, &wide(&x), &mut reference);
        adaptive_avg_pool2d(batch, c, input, output, &abs(&wide(&x)), &mut scale);
        let worst = assert_within(
            &cuda_adaptive(&cuda, case, &x),
            &reference,
            |i| (2.0 * (n + 2.0) + 11.0) * U * scale[i],
            &what,
        );
        worst_forward = worst_forward.max(worst.ratio());

        let dy = seeded_values(2300 + s, outs, -1.0, 1.0);
        let dx = seeded_values(2400 + s, ins, -1.0, 1.0);
        let (mut reference, mut scale) = (wide(&dx), abs(&wide(&dx)));
        adaptive_avg_pool2d_backward(batch, c, input, output, &wide(&dy), &mut reference);
        adaptive_avg_pool2d_backward(batch, c, input, output, &abs(&wide(&dy)), &mut scale);
        let worst = assert_within(
            &cuda_adaptive_backward(&cuda, case, &dy, &dx),
            &reference,
            |i| (2.0 * (m + 3.0) + 10.0) * U * scale[i],
            &format!("{what} backward"),
        );
        worst_backward = worst_backward.max(worst.ratio());
    }
    println!(
        "adaptive_avg_pool2d: worst {worst_forward:.3} of its bound forward, {worst_backward:.3} \
         backward"
    );
}

/// A 4100 × 4100 image pooled to its own size (16.8M values) copies it forward and adds `dy`
/// to `dx` backward, every value exactly once; two runs of both passes give the same bits.
#[test]
fn adaptive_pool_covers_a_large_grid_the_same_every_run() {
    let Some(cuda) = Cuda::new() else { return };
    let case = (1, (4100, 4100), (4100, 4100), 1);
    let n = 4100 * 4100;
    let x = exact_values(2500, n);
    assert_exact(&cuda_adaptive(&cuda, case, &x), &x, "adaptive 4100");
    let dy = exact_values(2501, n);
    let dx = exact_values(2502, n);
    let want: Vec<f32> = dx.iter().zip(&dy).map(|(a, b)| a + b).collect();
    let got = cuda_adaptive_backward(&cuda, case, &dy, &dx);
    assert_exact(&got, &want, "adaptive 4100 backward");

    let case = (3, (7, 5), (3, 2), 2);
    let x = mixed_values(2503, 2 * 3 * 35);
    let dy = mixed_values(2504, 2 * 3 * 6);
    let dx = mixed_values(2505, 2 * 3 * 35);
    assert_bits(
        &cuda_adaptive(&cuda, case, &x),
        &cuda_adaptive(&cuda, case, &x),
        "adaptive again",
    );
    assert_bits(
        &cuda_adaptive_backward(&cuda, case, &dy, &dx),
        &cuda_adaptive_backward(&cuda, case, &dy, &dx),
        "adaptive backward again",
    );
}

/// A count or size of 0, as the CPU refuses a size of 0, and sizes whose window arithmetic
/// or lengths overflow a `u32` are refused with `CudaError::Invalid`; a short buffer panics
/// before the launch, naming it.
#[test]
fn adaptive_misuse_is_refused() {
    let Some(cuda) = Cuda::new() else { return };
    for (batch, channels, input, output) in [
        (0, 1, (4, 4), (2, 2)),
        (1, 0, (4, 4), (2, 2)),
        (1, 1, (4, 4), (0, 2)),
        (1, 1, (0, 4), (2, 2)),
        (1, 1, (70_000, 1), (70_000, 1)),
        (1 << 16, 1 << 16, (1, 1), (1, 1)),
    ] {
        assert!(
            matches!(
                AdaptiveAvgPool2d::new(&cuda, batch, channels, input, output),
                Err(CudaError::Invalid(_))
            ),
            "{batch} {channels} {input:?} {output:?}"
        );
    }
    let pool = AdaptiveAvgPool2d::new(&cuda, 1, 1, (4, 4), (2, 2)).expect("adaptive pool");
    let (x, mut short) = (upload(&cuda, &[0.0f32; 16]), nans(&cuda, 3));
    let text = panic_message(|| {
        let _ = pool.forward(&x, &mut short);
    });
    assert!(
        text.contains("the output buffer holds 12 bytes, and 4 values need 16"),
        "{text}"
    );
}
