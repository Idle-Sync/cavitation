//! The GPU pooling kernels against their CPU twins in `cavitation_core::kernels::pooling`,
//! on every backend this machine has. Max pooling forward is compared bit for bit, payloads
//! and subnormals included, with its winners, on windows built to hold the ties and NaNs
//! that decide a winner. Max pooling backward is compared bit for bit after flushing the
//! reference's subnormals, on sums that round. Average pooling is compared exactly on small
//! integers with power-of-two divisors, and within its rounding bound on seeded values. The
//! tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::Pool2dGeom;
use cavitation_core::kernels::pooling::{
    avg_pool2d, avg_pool2d_backward, max_pool2d_backward, max_pool2d_reference,
};
use cavitation_gpu::kernels::{AvgPool2d, MaxPool2d};
use common::{
    Kit, Seeded, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values,
};

/// The geometries the max-pooling twins run: the 2 × 2 stride-2 pooling of the CNN
/// benchmarks, overlapping padded windows, `ceil_mode`, dilation, unequal windows and
/// strides, and strides longer than the window, plain and dilated, which leave pixels in no
/// window, each with its batch size.
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

/// The geometries the average-pooling rounding twin runs, with their batch size and divisor
/// rule (`count_include_pad`, `divisor_override`): padded overlapping windows counted both
/// ways, `ceil_mode` with its clipped last window, an override, unequal windows, and a
/// stride longer than the window, which leaves pixels in no window.
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
/// another payload, both infinities, both zeros and both signs of a subnormal: the inputs
/// whose ordering a GPU's fast math or flushing could get wrong.
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

/// Asserts that `got` has exactly `want`'s bits, subnormals and NaN payloads included,
/// naming `what` and the first mismatch.
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

/// A kit whose steps can move the largest buffer of `geom` at `batch`.
fn kit(gpu: &cavitation_gpu::Gpu, geom: &Pool2dGeom, batch: usize) -> Kit {
    Kit::new(
        gpu,
        4 * (batch * geom.in_width().max(geom.out_width())) as u64,
    )
}

/// The GPU's max-pooling outputs and winners for `x`.
fn gpu_max(kit: &mut Kit, geom: &Pool2dGeom, batch: usize, x: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let pool2d = MaxPool2d::new(&mut kit.cache, geom, batch).unwrap();
    let n = batch * geom.out_width();
    let (xb, out, winners) = (kit.upload(x), kit.zeros(n), kit.zeros(n));
    let group = pool2d.forward_bind_group(&mut kit.steps, &xb, &out, &winners);
    let mut step = kit.steps.begin();
    pool2d.forward(&mut step, &group);
    let _ = step.finish();
    let result = (
        kit.read(&out),
        kit.read(&winners).iter().map(|w| w.to_bits()).collect(),
    );
    for b in [xb, out, winners] {
        kit.pool.give(b);
    }
    result
}

/// The GPU's max-pooling input gradient: `dx` plus each of `dy` at its winner.
fn gpu_max_backward(
    kit: &mut Kit,
    geom: &Pool2dGeom,
    batch: usize,
    dy: &[f32],
    winners: &[u32],
    dx: &[f32],
) -> Vec<f32> {
    let pool2d = MaxPool2d::new(&mut kit.cache, geom, batch).unwrap();
    let as_bits: Vec<f32> = winners.iter().map(|&w| f32::from_bits(w)).collect();
    let (dyb, wb, dxb) = (kit.upload(dy), kit.upload(&as_bits), kit.upload(dx));
    let group = pool2d.backward_bind_group(&mut kit.steps, &dyb, &wb, &dxb);
    let mut step = kit.steps.begin();
    pool2d.backward(&mut step, &group);
    let _ = step.finish();
    let result = kit.read(&dxb);
    for b in [dyb, wb, dxb] {
        kit.pool.give(b);
    }
    result
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
    dy: &[f32],
    winners: &[u32],
    dx: &[f32],
) -> Vec<f32> {
    let winners: Vec<usize> = winners.iter().map(|&w| w as usize).collect();
    let mut dx = dx.to_vec();
    max_pool2d_backward(geom, batch, dy, &winners, &mut dx);
    dx
}

/// Max pooling forward on windows of three built to decide each rule a winner depends on,
/// with the CPU reference's bits and winners: of two NaNs the later wins; a NaN after a
/// number wins and a number after a NaN does not; of -0 and +0 the first wins either way;
/// a positive subnormal beats +0 and the larger of two subnormals wins (nothing flushed);
/// infinities order with the numbers; equal values go to the first.
#[test]
fn max_pool_forward_decides_ties_and_nans_as_the_cpu() {
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
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu, &geom, 1);
        let (out, winners) = gpu_max(&mut kit, &geom, 1, &x);
        let (want, want_winners) = cpu_max(&geom, 1, &x);
        assert_eq!(want_winners, [2, 4, 6, 9, 12, 16, 19, 23, 26, 28, 30, 35]);
        assert_bits(&out, &want, "max_pool2d special windows");
        assert_eq!(winners, want_winners, "max_pool2d special windows' winners");
    });
}

/// Max pooling forward on small integers (ties everywhere) mixed with NaNs, infinities,
/// signed zeros and subnormals: the same bits and the same winners as the CPU reference.
#[test]
fn max_pool_forward_exact_twin() {
    for_each_gpu(|gpu| {
        for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
            let mut kit = kit(gpu, &geom, batch);
            let x = with_specials(exact_values(s as u64, batch * geom.in_width()));
            let (out, winners) = gpu_max(&mut kit, &geom, batch, &x);
            let (want, want_winners) = cpu_max(&geom, batch, &x);
            assert_bits(&out, &want, &format!("max_pool2d {geom:?}"));
            assert_eq!(winners, want_winners, "max_pool2d winners {geom:?}");
        }
    });
}

/// Max pooling forward on seeded values: no rounding is involved, so the rounding twin is
/// bit for bit too.
#[test]
fn max_pool_forward_seeded_twin() {
    for_each_gpu(|gpu| {
        for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
            let mut kit = kit(gpu, &geom, batch);
            let x = seeded_values(100 + s as u64, batch * geom.in_width(), -1.0, 1.0);
            let (out, winners) = gpu_max(&mut kit, &geom, batch, &x);
            let (want, want_winners) = cpu_max(&geom, batch, &x);
            assert_bits(&out, &want, &format!("max_pool2d {geom:?}"));
            assert_eq!(winners, want_winners, "max_pool2d winners {geom:?}");
        }
    });
}

/// Max pooling backward on small integers: each winner's input gradient gets exactly its
/// outputs' gradients, and every other input keeps its own.
#[test]
fn max_pool_backward_exact_twin() {
    for_each_gpu(|gpu| {
        for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
            let mut kit = kit(gpu, &geom, batch);
            let s = s as u64;
            let x = exact_values(200 + s, batch * geom.in_width());
            let (_, winners) = cpu_max(&geom, batch, &x);
            let dy = exact_values(300 + s, batch * geom.out_width());
            let dx = exact_values(400 + s, batch * geom.in_width());
            let got = gpu_max_backward(&mut kit, &geom, batch, &dy, &winners, &dx);
            let want = cpu_max_backward(&geom, batch, &dy, &winners, &dx);
            assert_exact(&got, &want, &format!("max_pool2d backward {geom:?}"));
        }
    });
}

/// Max pooling backward on gradients spread over sixteen binades, so that an input's sum of
/// its own gradient and its outputs' rounds: the GPU adds them in the CPU's order, so the
/// bits match. The test counts the inputs whose f32 sum differs from the exact one and
/// requires many, so the order is what it checks.
#[test]
fn max_pool_backward_seeded_twin() {
    for_each_gpu(|gpu| {
        let mut rounded = 0;
        for (s, (geom, batch)) in max_geometries().into_iter().enumerate() {
            let mut kit = kit(gpu, &geom, batch);
            let s = s as u64;
            let x = seeded_values(500 + s, batch * geom.in_width(), -1.0, 1.0);
            let (_, winners) = cpu_max(&geom, batch, &x);
            let dy = mixed_values(600 + s, batch * geom.out_width());
            let dx = mixed_values(700 + s, batch * geom.in_width());
            let got = gpu_max_backward(&mut kit, &geom, batch, &dy, &winners, &dx);
            let want = cpu_max_backward(&geom, batch, &dy, &winners, &dx);
            assert_exact(&got, &want, &format!("max_pool2d backward {geom:?}"));
            let mut exact: Vec<f64> = dx.iter().map(|&v| f64::from(v)).collect();
            let plane_out = geom.out_width() / geom.channels();
            let plane_in = geom.in_width() / geom.channels();
            for (o, (&g, &w)) in dy.iter().zip(&winners).enumerate() {
                exact[o / plane_out * plane_in + w as usize] += f64::from(g);
            }
            let multi = (0..exact.len()).filter(|&i| {
                let wins = (0..dy.len())
                    .filter(|&o| o / plane_out * plane_in + winners[o] as usize == i)
                    .count();
                wins >= 2 && f64::from(want[i]) != exact[i]
            });
            rounded += multi.count();
        }
        println!("max_pool2d backward: {rounded} inputs with two or more rounded additions");
        assert!(rounded >= 100, "only {rounded} multi-gradient sums rounded");
    });
}

/// More threads than one row of workgroups holds (a 4100 × 4100 image with 1 × 1 windows,
/// 16.8M outputs): both passes number their workgroups over the split grid and cover every
/// value exactly once.
#[test]
fn max_pool_covers_a_split_grid() {
    for_each_gpu(|gpu| {
        let geom = Pool2dGeom::new(1, (4100, 4100), (1, 1)).unwrap();
        let mut kit = kit(gpu, &geom, 1);
        let x = exact_values(800, geom.in_width());
        let (out, winners) = gpu_max(&mut kit, &geom, 1, &x);
        assert_bits(&out, &x, "max_pool2d 1x1");
        assert!(winners.iter().enumerate().all(|(i, &w)| w as usize == i));
        let dy = exact_values(801, geom.out_width());
        let dx = exact_values(802, geom.in_width());
        let got = gpu_max_backward(&mut kit, &geom, 1, &dy, &winners, &dx);
        let want: Vec<f32> = dx.iter().zip(&dy).map(|(a, b)| a + b).collect();
        assert_exact(&got, &want, "max_pool2d 1x1 backward");
    });
}

/// Average pooling over the split grid, as max pooling above: 1 × 1 windows on a 4100 ×
/// 4100 image copy it forward and add `dy` to `dx` backward, every value exactly once.
#[test]
fn avg_pool_covers_a_split_grid() {
    for_each_gpu(|gpu| {
        let geom = Pool2dGeom::new(1, (4100, 4100), (1, 1)).unwrap();
        let mut kit = kit(gpu, &geom, 1);
        let case = (&geom, 1, true, None);
        let x = exact_values(803, geom.in_width());
        assert_exact(&gpu_avg(&mut kit, case, &x), &x, "avg_pool2d 1x1");
        let dy = exact_values(804, geom.out_width());
        let dx = exact_values(805, geom.in_width());
        let want: Vec<f32> = dx.iter().zip(&dy).map(|(a, b)| a + b).collect();
        assert_exact(
            &gpu_avg_backward(&mut kit, case, &dy, &dx),
            &want,
            "avg_pool2d 1x1 backward",
        );
    });
}

/// The GPU's average-pooling outputs for `x`.
fn gpu_avg(
    kit: &mut Kit,
    (geom, batch, pad, divisor): (&Pool2dGeom, usize, bool, Option<usize>),
    x: &[f32],
) -> Vec<f32> {
    let pool2d = AvgPool2d::new(&mut kit.cache, geom, batch, pad, divisor).unwrap();
    let (xb, out) = (kit.upload(x), kit.zeros(batch * geom.out_width()));
    let group = pool2d.forward_bind_group(&mut kit.steps, &xb, &out);
    let mut step = kit.steps.begin();
    pool2d.forward(&mut step, &group);
    let _ = step.finish();
    let result = kit.read(&out);
    kit.pool.give(xb);
    kit.pool.give(out);
    result
}

/// The GPU's average-pooling input gradient: `dx` plus each output gradient's share.
fn gpu_avg_backward(
    kit: &mut Kit,
    (geom, batch, pad, divisor): (&Pool2dGeom, usize, bool, Option<usize>),
    dy: &[f32],
    dx: &[f32],
) -> Vec<f32> {
    let pool2d = AvgPool2d::new(&mut kit.cache, geom, batch, pad, divisor).unwrap();
    let (dyb, dxb) = (kit.upload(dy), kit.upload(dx));
    let group = pool2d.backward_bind_group(&mut kit.steps, &dyb, &dxb);
    let mut step = kit.steps.begin();
    pool2d.backward(&mut step, &group);
    let _ = step.finish();
    let result = kit.read(&dxb);
    kit.pool.give(dyb);
    kit.pool.give(dxb);
    result
}

/// Average pooling on small integers where every divisor is a power of two (2 × 2 windows,
/// so 4, or an override of 2), forward and backward: every sum is exact in any order, and
/// so is a division by a power of two on the backends measured (WGSL only promises
/// 2.5 ulp), so the GPU matches the CPU bit for bit, overlapping windows included.
#[test]
fn avg_pool_exact_twin() {
    let with = Pool2dGeom::with_options;
    let cases = [
        (Pool2dGeom::new(8, (32, 32), (2, 2)).unwrap(), 4, true, None),
        (
            with(3, (9, 11), (2, 2), (1, 1), (0, 0), (1, 1), false).unwrap(),
            2,
            true,
            None,
        ),
        (
            with(2, (9, 9), (3, 3), (1, 2), (1, 1), (1, 1), true).unwrap(),
            2,
            false,
            Some(2),
        ),
    ];
    for_each_gpu(|gpu| {
        for (s, (geom, batch, pad, divisor)) in cases.iter().enumerate() {
            let (batch, s) = (*batch, s as u64);
            let case = (geom, batch, *pad, *divisor);
            let mut kit = kit(gpu, geom, batch);
            let x = exact_values(900 + s, batch * geom.in_width());
            let mut want = vec![0.0; batch * geom.out_width()];
            avg_pool2d(geom, batch, &x, *pad, *divisor, &mut want);
            assert_exact(
                &gpu_avg(&mut kit, case, &x),
                &want,
                &format!("avg_pool2d {geom:?}"),
            );
            let dy = exact_values(1000 + s, batch * geom.out_width());
            let dx = exact_values(1100 + s, batch * geom.in_width());
            let mut want = dx.clone();
            avg_pool2d_backward(geom, batch, &dy, *pad, *divisor, &mut want);
            let got = gpu_avg_backward(&mut kit, case, &dy, &dx);
            assert_exact(&got, &want, &format!("avg_pool2d backward {geom:?}"));
        }
    });
}

/// Average pooling on seeded values, forward and backward, against an f64 reference of the
/// same f32 inputs: an output summing at most `n = kh · kw` pixels with divisor `d` is within
/// `(2(n + 2) + 6) · u · Σ|x| / d` (the sum in any order, then a 2.5-ulp division), and an
/// input gradient that at most `m = ⌈kh / sh⌉ · ⌈kw / sw⌉` outputs cover is within
/// `(2(m + 3) + 5) · u · (|dx| + Σ|dy / d|)`.
#[test]
fn avg_pool_rounding_twin() {
    for_each_gpu(|gpu| {
        for (s, (geom, batch, pad, divisor)) in avg_geometries().into_iter().enumerate() {
            let s = s as u64;
            let case = (&geom, batch, pad, divisor);
            let mut kit = kit(gpu, &geom, batch);
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
                &gpu_avg(&mut kit, case, &x),
                &reference,
                |i| (2.0 * (n + 2.0) + 6.0) * U * scale[i],
                &format!("avg_pool2d {geom:?}"),
            );
            println!(
                "avg_pool2d {geom:?}: worst error {:.3} of its bound",
                worst.ratio()
            );

            let dy = seeded_values(1300 + s, outs, -1.0, 1.0);
            let dx = seeded_values(1400 + s, batch * geom.in_width(), -1.0, 1.0);
            let (mut reference, mut scale) = (wide(&dx), abs(&wide(&dx)));
            avg_pool2d_backward(&geom, batch, &wide(&dy), pad, divisor, &mut reference);
            avg_pool2d_backward(&geom, batch, &abs(&wide(&dy)), pad, divisor, &mut scale);
            let m = (kh.div_ceil(sh) * kw.div_ceil(sw)) as f64;
            let worst = assert_within(
                &gpu_avg_backward(&mut kit, case, &dy, &dx),
                &reference,
                |i| (2.0 * (m + 3.0) + 5.0) * U * scale[i],
                &format!("avg_pool2d backward {geom:?}"),
            );
            println!(
                "avg_pool2d backward {geom:?}: worst error {:.3} of its bound",
                worst.ratio()
            );
        }
    });
}

/// A kernel's pipelines are built once per geometry and batch: building the same pooling
/// again builds nothing, and another batch size builds its own.
#[test]
fn pipelines_are_built_once() {
    for_each_gpu(|gpu| {
        let geom = Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap();
        let mut kit = kit(gpu, &geom, 2);
        let before = kit.cache.pipelines_built();
        MaxPool2d::new(&mut kit.cache, &geom, 2).unwrap();
        MaxPool2d::new(&mut kit.cache, &geom, 2).unwrap();
        AvgPool2d::new(&mut kit.cache, &geom, 2, true, None).unwrap();
        AvgPool2d::new(&mut kit.cache, &geom, 2, true, None).unwrap();
        assert_eq!(kit.cache.pipelines_built() - before, 4);
        MaxPool2d::new(&mut kit.cache, &geom, 3).unwrap();
        assert_eq!(kit.cache.pipelines_built() - before, 6);
    });
}

/// A buffer of the wrong size is refused when the bind group is made, naming it.
#[test]
fn bind_groups_check_buffer_sizes() {
    for_each_gpu(|gpu| {
        let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
        let mut kit = kit(gpu, &geom, 1);
        let pool2d = MaxPool2d::new(&mut kit.cache, &geom, 1).unwrap();
        let (x, out, short) = (kit.zeros(16), kit.zeros(4), kit.zeros(3));
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool2d.forward_bind_group(&mut kit.steps, &x, &out, &short);
        }));
        let message = refused.expect_err("a short winners buffer is refused");
        let message = message
            .downcast_ref::<String>()
            .expect("a formatted message");
        assert!(
            message.contains("the winners holds 12 bytes, expected 16"),
            "{message}"
        );
    });
}
