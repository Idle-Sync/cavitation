//! Round to nearest, ties to even, the rounding every GPU twin depends on, checked on its
//! own: f32 `+` and `*` on chosen cases just above, just below and exactly halfway between
//! two floats, with results of both signs, and on seeded ones. WGSL only promises one of the
//! two floats either side (§15.7.4, "WGSL does not specify a rounding mode"). The rounding
//! twins take one rounding to be within `U = 2⁻²⁴` (`tests/common`), which needs round to
//! nearest, and the kernels that match the CPU bit for bit need its ties to even as well. A
//! backend that rounds another way fails here, named, instead of at scattered tolerances
//! across the suite. The GPU tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`
//! makes that a failure.

mod common;

use cavitation_gpu::Shader;
use common::{Kit, Seeded, for_each_gpu};

/// `sum = x + y` and `product = x · y`, element by element, from buffers, so no compiler can
/// fold them.
const OPS: Shader = Shader {
    name: "rounding probe",
    source: "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read> y: array<f32>;
        @group(0) @binding(2) var<storage, read_write> sum: array<f32>;
        @group(0) @binding(3) var<storage, read_write> product: array<f32>;
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i < arrayLength(&sum)) {
                sum[i] = x[i] + y[i];
                product[i] = x[i] * y[i];
            }
        }
    ",
};

/// An operation the probe checks, on the exact values of its operands.
type Op = fn(f64, f64) -> f64;

/// The probe's two operations.
const OPS_CHECKED: [(&str, Op); 2] = [("+", |a, b| a + b), ("*", |a, b| a * b)];

/// Whether `r`, a normal-range value exact in `f64`, lies exactly halfway between two
/// floats.
fn is_tie(r: f64) -> bool {
    let n = r as f32;
    let nf = f64::from(n);
    if nf == r {
        return false;
    }
    let other = if nf < r { n.next_up() } else { n.next_down() };
    (r - nf).abs() == (f64::from(other) - r).abs()
}

/// The chosen pairs: for sums, 1, 1.5 and 3 plus offsets just above, just below and at half
/// their ulp; for products, `(1 + 2⁻¹²)` times `1 + 2⁻¹²` moved by one ulp either way, whose
/// exact products lie just above, just below and at a halfway point. Each is repeated with a
/// negative result, so rounding toward zero, down or up, or ties away from zero, each misses
/// one.
fn chosen() -> Vec<(f32, f32)> {
    let p = |e: i32| 2f32.powi(e);
    let mut pairs = Vec::new();
    for (base, half) in [(1.0, p(-24)), (1.5, p(-24)), (3.0, p(-23))] {
        for off in [half + half * p(-6), half - half * p(-6), half] {
            pairs.push((base, off));
            pairs.push((-base, -off));
        }
    }
    let a = 1.0 + p(-12);
    for b in [a + p(-23), a - p(-23), a] {
        pairs.push((a, b));
        pairs.push((-a, b));
    }
    pairs
}

/// Seeded pairs of magnitude in `[0.5, 4)` with random signs, so every sum and product is
/// exact in `f64`.
fn seeded(seed: u64, n: usize) -> Vec<(f32, f32)> {
    let mut rng = Seeded::new(seed);
    let mut one = || {
        let v = rng.uniform(0.5, 4.0);
        if rng.next_u64() >> 63 == 0 { v } else { -v }
    };
    (0..n).map(|_| (one(), one())).collect()
}

/// The `(index, a, b, got, wanted)` of each result that is not the exact one rounded to
/// nearest, ties to even (Rust's `f64` to `f32` conversion).
fn misses(pairs: &[(f32, f32)], got: &[f32], op: Op) -> Vec<(usize, f32, f32, f32, f32)> {
    pairs
        .iter()
        .zip(got)
        .enumerate()
        .filter_map(|(i, (&(a, b), &g))| {
            let wanted = op(f64::from(a), f64::from(b)) as f32;
            (g.to_bits() != wanted.to_bits()).then_some((i, a, b, g, wanted))
        })
        .collect()
}

/// The chosen cases cover what they claim: for sums and for products, results exactly
/// halfway and results off halfway, each with both signs, and the hand-worked ones right.
#[test]
fn the_chosen_cases_straddle_halfway_points() {
    let pairs = chosen();
    for (op, f) in OPS_CHECKED {
        let results: Vec<f64> = pairs
            .iter()
            .map(|&(a, b)| f(f64::from(a), f64::from(b)))
            .collect();
        for negative in [false, true] {
            let side = |r: &&f64| (**r < 0.0) == negative;
            assert!(
                results.iter().filter(side).any(|&r| is_tie(r)),
                "`{op}`: no exact halfway case, negative {negative}"
            );
            assert!(
                results
                    .iter()
                    .filter(side)
                    .any(|&r| !is_tie(r) && f64::from(r as f32) != r),
                "`{op}`: no inexact case off halfway, negative {negative}"
            );
        }
    }
    let (a, p) = (1.0 + 2f64.powi(-12), |e: i32| 2f32.powi(e));
    assert_eq!((a * (a + 2f64.powi(-23))) as f32, 1.0 + p(-11) + p(-22));
    assert_eq!((a * (a - 2f64.powi(-23))) as f32, 1.0 + p(-11) - p(-23));
    assert!(is_tie(a * a));
    assert_eq!((a * a) as f32, 1.0 + p(-11));
    assert_eq!((1.0 + 2f64.powi(-24) + 2f64.powi(-30)) as f32, 1.0 + p(-23));
    assert_eq!((1.0 + 2f64.powi(-24) - 2f64.powi(-30)) as f32, 1.0);
    assert!(is_tie(1.0 + 2f64.powi(-24)));
    assert_eq!((1.0 + 2f64.powi(-24)) as f32, 1.0);
}

/// `r` rounded toward zero, down, up, or to nearest with ties away from zero, as a backend
/// with that rounding would give it.
fn other_rounding(r: f64, mode: &str) -> f32 {
    let n = r as f32;
    let nf = f64::from(n);
    let (down, up) = if nf == r {
        (n, n)
    } else if nf < r {
        (n, n.next_up())
    } else {
        (n.next_down(), n)
    };
    match mode {
        "down" => down,
        "up" => up,
        "ties away" if is_tie(r) => {
            if r < 0.0 {
                down
            } else {
                up
            }
        }
        "ties away" => n,
        _ if r < 0.0 => up,
        _ => down,
    }
}

/// Rounding toward zero, down or up, or ties away from zero, each gives a result the probe
/// refuses, for `+` and for `*`, on the chosen cases alone; the CPU's own f32 `+` and `*`
/// pass.
#[test]
fn other_roundings_fail_the_probe() {
    let pairs = chosen();
    for mode in ["toward zero", "down", "up", "ties away"] {
        for (op, f) in OPS_CHECKED {
            let got: Vec<f32> = pairs
                .iter()
                .map(|&(a, b)| other_rounding(f(f64::from(a), f64::from(b)), mode))
                .collect();
            assert!(
                !misses(&pairs, &got, f).is_empty(),
                "rounding {mode} passes for `{op}`"
            );
        }
    }
    let sums: Vec<f32> = pairs.iter().map(|&(a, b)| a + b).collect();
    let products: Vec<f32> = pairs.iter().map(|&(a, b)| a * b).collect();
    assert!(misses(&pairs, &sums, OPS_CHECKED[0].1).is_empty());
    assert!(misses(&pairs, &products, OPS_CHECKED[1].1).is_empty());
}

/// Every backend rounds f32 `+` and `*` to nearest, ties to even: the requirement the twins
/// rest on.
#[test]
fn the_gpu_rounds_to_nearest() {
    for_each_gpu(|gpu| {
        let mut pairs = chosen();
        pairs.extend(seeded(1, 1 << 16));
        let n = pairs.len();
        let mut kit = Kit::new(gpu, 4 * n as u64);
        let pipeline = kit.cache.get(OPS, "main", &[]).expect("build");
        let x: Vec<f32> = pairs.iter().map(|p| p.0).collect();
        let y: Vec<f32> = pairs.iter().map(|p| p.1).collect();
        let (bx, by) = (kit.upload(&x), kit.upload(&y));
        let (bs, bp) = (kit.zeros(n), kit.zeros(n));
        let group = kit.bind(&pipeline, &[&bx, &by, &bs, &bp]);
        let mut step = kit.steps.begin();
        step.dispatch(&pipeline, &group, (n as u32).div_ceil(256));
        let _ = step.finish();
        let (sums, products) = (kit.read(&bs), kit.read(&bp));
        for ((op, f), got) in OPS_CHECKED.into_iter().zip([&sums, &products]) {
            let missed = misses(&pairs, got, f);
            assert!(
                missed.is_empty(),
                "{}: f32 `{op}` does not round to nearest, ties to even, on this GPU ({} of {n} \
                 results; first (index, a, b, got, wanted): {:?}). The GPU twins require it: \
                 their bounds take one rounding to be within U = 2^-24, and the kernels that \
                 match the CPU bit for bit need its ties. WGSL leaves the rounding mode open, \
                 so this backend is outside what the suite covers (tests/common/mod.rs).",
                gpu.info().name,
                missed.len(),
                &missed[..missed.len().min(4)]
            );
        }
        for b in [bx, by, bs, bp] {
            kit.pool.give(b);
        }
    });
}
