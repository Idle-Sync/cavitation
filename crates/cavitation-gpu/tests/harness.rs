//! The twin harness in `tests/common` tested on its own: the generators, the flush, the
//! bounds and the two assertions on made-up data, then both twins on every backend this
//! machine has, with an element-wise add, a dot product, and a broken dot product that
//! skips one term and must be caught by the exact twin. The GPU tests skip without a GPU,
//! unless `CAVITATION_GPU_REQUIRED=1` makes that a failure.

mod common;

use common::{
    FLOOR, Kit, Seeded, U, assert_exact, assert_within, exact_values, flush, for_each_gpu,
    gemm_bound, seeded_values, sum_bound,
};

use cavitation_gpu::Shader;

/// `z = x + y`, element by element.
const ADD: Shader = Shader {
    name: "harness add",
    source: "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read> y: array<f32>;
        @group(0) @binding(2) var<storage, read_write> z: array<f32>;
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x < arrayLength(&z)) { z[id.x] = x[id.x] + y[id.x]; }
        }
    ",
};

/// `out[r] = Σ_k a[r·K + k] · b[k]`, one thread per row, `K` from an `override`; with
/// `SKIP` set to a column, that column's term is left out, as a broken kernel would.
const DOT: Shader = Shader {
    name: "harness dot",
    source: "
        override K: u32 = 1u;
        override SKIP: u32 = 4294967295u;
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let r = id.x;
            if (r >= arrayLength(&out)) { return; }
            var acc = 0.0;
            for (var k = 0u; k < K; k = k + 1u) {
                if (k != SKIP) { acc = fma(a[r * K + k], b[k], acc); }
            }
            out[r] = acc;
        }
    ",
};

/// The exact generator draws only -2, -1, 1 and 2, all four of them, and the same values
/// for the same seed.
#[test]
fn exact_values_are_small_nonzero_integers() {
    let v = exact_values(7, 10_000);
    assert!(v.iter().all(|x| [-2.0, -1.0, 1.0, 2.0].contains(x)));
    for want in [-2.0, -1.0, 1.0, 2.0] {
        assert!(v.contains(&want), "{want} never drawn");
    }
    assert_eq!(v, exact_values(7, 10_000));
    assert_ne!(v, exact_values(8, 10_000));
}

/// Seeded values stay in their range and repeat for a seed.
#[test]
fn seeded_values_stay_in_range() {
    let v = seeded_values(3, 10_000, -0.5, 2.0);
    assert!(v.iter().all(|x| (-0.5..2.0).contains(x)));
    assert_eq!(v, seeded_values(3, 10_000, -0.5, 2.0));
    let mut a = Seeded::new(0);
    let mut b = Seeded::new(0);
    assert_eq!(a.next_u64(), b.next_u64());
}

/// The flush turns subnormals into zeros of the same sign and leaves everything else.
#[test]
fn flush_only_touches_subnormals() {
    let tiny = f32::from_bits(1);
    assert_eq!(flush(tiny).to_bits(), 0.0f32.to_bits());
    assert_eq!(flush(-tiny).to_bits(), (-0.0f32).to_bits());
    for x in [0.0, -0.0, 1.0, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY] {
        assert_eq!(flush(x).to_bits(), x.to_bits());
    }
    assert!(flush(f32::NAN).is_nan());
}

/// The constants and the sum bound have the values the spec states.
#[test]
fn bounds() {
    assert_eq!(U, 2f64.powi(-24));
    assert_eq!(FLOOR, 2f64.powi(-126));
    assert_eq!(FLOOR, f64::from(f32::MIN_POSITIVE));
    assert_eq!(sum_bound(1000, 3.0), 2.0 * 1002.0 * U * 3.0);
    assert_eq!(gemm_bound(8, 1.0), sum_bound(8, 1.0));
}

/// The exact assertion passes on equal bits, on any NaN against any NaN, and on a GPU
/// zero where the reference is subnormal.
#[test]
fn exact_assertion_accepts_what_it_should() {
    let quiet = f32::from_bits(0x7fc0_0000);
    let other = f32::from_bits(0x7fc0_0001);
    assert_exact(
        &[1.0, quiet, 0.0],
        &[1.0, other, f32::from_bits(5)],
        "accepted",
    );
}

/// The exact assertion fails on a one-ulp difference, and its message names the element.
#[test]
#[should_panic(expected = "1 of 3 elements differ; first: [1]")]
fn exact_assertion_catches_one_ulp() {
    assert_exact(&[1.0, 1.0 + f32::EPSILON, 3.0], &[1.0, 1.0, 3.0], "one ulp");
}

/// The rounding assertion passes within the bound, reports the element closest to its
/// bound, and fails outside it.
#[test]
fn rounding_assertion() {
    let worst = assert_within(&[1.0, 2.0], &[1.0, 2.0 + 1e-7], |_| 1e-6, "within");
    assert_eq!(worst.index, 1);
    assert!(
        worst.ratio() > 0.09 && worst.ratio() < 0.11,
        "{}",
        worst.ratio()
    );
    let outside = std::panic::catch_unwind(|| {
        assert_within(&[1.0], &[1.0 + 1e-5], |_| 1e-6, "outside");
    });
    assert!(outside.is_err());
}

/// A NaN where the reference is a number, or a number where it is NaN, fails.
#[test]
#[should_panic(expected = "element 0 is NaN where the reference is 1e0")]
fn rounding_assertion_checks_nan_positions() {
    assert_within(&[f32::NAN], &[1.0], |_| 1.0, "nan");
}

/// Both twins of an element-wise add, on every backend: exact inputs give the CPU's bits,
/// seeded inputs are within one rounding.
#[test]
fn add_twins_on_every_gpu() {
    for_each_gpu(|gpu| {
        let n = 100_000;
        let mut kit = Kit::new(gpu, 4 * n as u64);
        let pipeline = kit.cache.get(ADD, "main", &[]).expect("build");
        for (x, y, exact) in [
            (exact_values(1, n), exact_values(2, n), true),
            (
                seeded_values(3, n, -1e3, 1e3),
                seeded_values(4, n, -1.0, 1.0),
                false,
            ),
        ] {
            let (bx, by, bz) = (kit.upload(&x), kit.upload(&y), kit.zeros(n));
            let group = kit.bind(&pipeline, &[&bx, &by, &bz]);
            let mut step = kit.steps.begin();
            step.dispatch(&pipeline, &group, (n as u32).div_ceil(256));
            let _ = step.finish();
            let got = kit.read(&bz);
            if exact {
                let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
                assert_exact(&got, &want, "add, exact");
            } else {
                let reference: Vec<f64> = x
                    .iter()
                    .zip(&y)
                    .map(|(&a, &b)| a as f64 + b as f64)
                    .collect();
                let worst = assert_within(
                    &got,
                    &reference,
                    |i| U * reference[i].abs(),
                    "add, rounding",
                );
                println!("add: worst error {:.3} of its bound", worst.ratio());
            }
            for b in [bx, by, bz] {
                kit.pool.give(b);
            }
        }
    });
}

/// Both twins of a dot product with k = 4096, on every backend, and the exact twin
/// catching a kernel that drops one of the 4096 terms.
#[test]
fn dot_twins_and_a_dropped_term_on_every_gpu() {
    for_each_gpu(|gpu| {
        let (rows, k) = (512usize, 4096usize);
        let mut kit = Kit::new(gpu, 4 * (rows * k) as u64);
        let good = kit
            .cache
            .get(DOT, "main", &[("K", k as f64)])
            .expect("build");
        let broken = kit
            .cache
            .get(DOT, "main", &[("K", k as f64), ("SKIP", 1000.0)])
            .expect("build");
        let run = |kit: &mut Kit,
                   pipeline: &cavitation_gpu::wgpu::ComputePipeline,
                   a: &[f32],
                   b: &[f32]| {
            let (ba, bb, bo) = (kit.upload(a), kit.upload(b), kit.zeros(rows));
            let group = kit.bind(pipeline, &[&ba, &bb, &bo]);
            let mut step = kit.steps.begin();
            step.dispatch(pipeline, &group, (rows as u32).div_ceil(64));
            let _ = step.finish();
            let got = kit.read(&bo);
            for buffer in [ba, bb, bo] {
                kit.pool.give(buffer);
            }
            got
        };
        // Exact: integer products and sums, so any order and any fusion give the same bits.
        let (a, b) = (exact_values(5, rows * k), exact_values(6, k));
        let want: Vec<f32> = (0..rows)
            .map(|r| (0..k).map(|j| a[r * k + j] * b[j]).sum())
            .collect();
        assert_exact(&run(&mut kit, &good, &a, &b), &want, "dot, exact");
        let dropped = run(&mut kit, &broken, &a, &b);
        let caught =
            std::panic::catch_unwind(|| assert_exact(&dropped, &want, "dot, one term dropped"));
        assert!(caught.is_err(), "the exact twin missed a dropped term");
        // Rounding: against f64, within 2(k+2)·u·Σ|a·b|.
        let (a, b) = (
            seeded_values(7, rows * k, -1.0, 1.0),
            seeded_values(8, k, -1.0, 1.0),
        );
        let (a_ref, b_ref) = (&a, &b);
        let products = |r: usize| (0..k).map(move |j| a_ref[r * k + j] as f64 * b_ref[j] as f64);
        let reference: Vec<f64> = (0..rows).map(|r| products(r).sum()).collect();
        let abs: Vec<f64> = (0..rows).map(|r| products(r).map(f64::abs).sum()).collect();
        let got = run(&mut kit, &good, &a, &b);
        let worst = assert_within(&got, &reference, |r| gemm_bound(k, abs[r]), "dot, rounding");
        println!("dot k={k}: worst error {:.4} of its bound", worst.ratio());
    });
}
