//! The health values against the CPU health checks' rules, on every backend this machine
//! has:
//!
//! - **Sums of squares:** exact on integers, and within the bound of a sum on seeded values,
//!   for tensors one after another in a flat buffer, written at their word offsets.
//! - **Finite flag:** exact against `GradStats`' and `OnNonFinite::Skip`'s rule (the loss
//!   and every gradient value finite), with NaN and infinity in the first, a middle and the
//!   last value, the extremes of the finite range, and a gradient whose sum of squares
//!   overflows.
//! - **Dead-ReLU bits:** exact against `HealthMonitor`'s `v != 0.0`, for channel counts
//!   around a word's 32, dense and convolution shapes, −0, NaN and subnormals, overwriting
//!   and OR-ing into old bits.
//!
//! `HealthMonitor` lives in `cavitation-nn`, which this crate's tests do not depend on, so
//! its rules are transcribed here. The tests skip without a GPU, unless
//! `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_gpu::kernels::{ChannelDims, DeadRelu, FiniteFlag, SumSquares};
use cavitation_gpu::{Gpu, PooledBuffer};
use common::{
    FLOOR, Kit, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values,
    sum_bound,
};

/// What fills the words around a result, to show nothing else is written.
const SENTINEL: f32 = 7.0;

/// A kit whose steps can move `values` values.
fn kit(gpu: &Gpu, values: usize) -> Kit {
    Kit::new(gpu, 4 * values as u64 + 65_536)
}

/// The words of `v` as `u32`s.
fn words(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Lengths of the tensors laid one after another in a flat gradient buffer: one value, a
/// small tensor, exactly one chunk (4,096), one more, and several chunks.
const LENGTHS: [usize; 5] = [1, 37, 4096, 4097, 100_000];

/// The GPU's sums of squares of the tensors `lengths` long, one after another in `flat`,
/// written to words 1, 2, … of a buffer of sentinels.
fn gpu_sums(kit: &mut Kit, kernel: &SumSquares, flat: &[f32], lengths: &[usize]) -> Vec<f32> {
    let grads = kit.upload(flat);
    let out = kit.upload(&vec![SENTINEL; lengths.len() + 2]);
    let mut start = 0;
    let calls: Vec<_> = lengths
        .iter()
        .enumerate()
        .map(|(k, &len)| {
            let call = kernel.call(
                &mut kit.steps,
                &mut kit.pool,
                (&grads, start as u32, len as u32),
                &out,
                k as u32 + 1,
            );
            start += len;
            call
        })
        .collect();
    let mut step = kit.steps.begin();
    for call in &calls {
        kernel.record(&mut step, call);
    }
    let _ = step.finish();
    let got = kit.read(&out);
    for call in calls {
        call.release(&mut kit.pool);
    }
    assert_eq!(got[0], SENTINEL, "the word before the sums changed");
    assert_eq!(
        got[lengths.len() + 1],
        SENTINEL,
        "the word after the sums changed"
    );
    got[1..=lengths.len()].to_vec()
}

/// Sums of squares on small integers are exact in any order, so each tensor's is the CPU's
/// `f32` sum bit for bit; on seeded values each is within
/// `(2(L + 2) + 1) · u · Σ g² + (3L − 1) · 2⁻¹²⁶` of the exact sum.
#[test]
fn sums_of_squares_twin() {
    for_each_gpu(|gpu| {
        let total: usize = LENGTHS.iter().sum();
        let mut kit = kit(gpu, total);
        let kernel = SumSquares::new(gpu, &mut kit.cache).unwrap();
        let tensors = |flat: &[f32]| {
            let mut start = 0;
            LENGTHS
                .iter()
                .map(|&len| {
                    let t = flat[start..start + len].to_vec();
                    start += len;
                    t
                })
                .collect::<Vec<_>>()
        };
        let flat = exact_values(1, total);
        let want: Vec<f32> = tensors(&flat)
            .iter()
            .map(|t| t.iter().map(|g| g * g).sum())
            .collect();
        assert_exact(
            &gpu_sums(&mut kit, &kernel, &flat, &LENGTHS),
            &want,
            "sums of squares on integers",
        );
        let flat = seeded_values(2, total, -3.0, 3.0);
        let exact: Vec<f64> = tensors(&flat)
            .iter()
            .map(|t| t.iter().map(|&g| f64::from(g).powi(2)).sum())
            .collect();
        let got = gpu_sums(&mut kit, &kernel, &flat, &LENGTHS);
        let worst = assert_within(
            &got,
            &exact,
            |k| {
                let l = LENGTHS[k];
                sum_bound(l, exact[k]) + U * exact[k] + (3 * l - 2) as f64 * FLOOR
            },
            "sums of squares on seeded values",
        );
        println!("sums of squares: worst {:.3} of its bound", worst.ratio());
        let again = gpu_sums(&mut kit, &kernel, &flat, &LENGTHS);
        assert_exact(&again, &got, "sums of squares on a second run");
    });
}

/// The GPU's finite flag of `loss` and the tensors `lengths` long one after another in
/// `flat`, as a `u32`.
fn gpu_flag(kit: &mut Kit, kernel: &FiniteFlag, loss: f32, flat: &[f32], lengths: &[usize]) -> u32 {
    let losses = kit.upload(&[SENTINEL, loss]);
    let grads = kit.upload(if flat.is_empty() { &[0.0] } else { flat });
    let flag = kit.upload(&[SENTINEL, SENTINEL]);
    let mut start = 0;
    let ranges: Vec<(&PooledBuffer, u32, u32)> = lengths
        .iter()
        .map(|&len| {
            let range = (&grads, start as u32, len as u32);
            start += len;
            range
        })
        .collect();
    let call = kernel.call(&mut kit.steps, &mut kit.pool, (&losses, 1), &ranges, &flag);
    let mut step = kit.steps.begin();
    kernel.record(&mut step, &call);
    let _ = step.finish();
    let got = words(&kit.read(&flag));
    call.release(&mut kit.pool);
    assert_eq!(
        got[1],
        SENTINEL.to_bits(),
        "the word after the flag changed"
    );
    got[0]
}

/// The CPU's rule: a step is bad when its loss or any gradient value is not finite.
fn cpu_flag(loss: f32, flat: &[f32]) -> u32 {
    u32::from(!(loss.is_finite() && flat.iter().all(|g| g.is_finite())))
}

/// The finite flag, bit for bit, against the CPU's rule:
/// - a NaN, +∞ or −∞ in the first value, a middle value or the very last value of the last
///   tensor sets it;
/// - so does a NaN or infinite loss with finite gradients;
/// - the largest finite value, the smallest subnormal and −0 leave it clear, as does a
///   gradient of 10³⁰, whose sum of squares overflows to infinity.
#[test]
fn finite_flag_twin() {
    for_each_gpu(|gpu| {
        let total: usize = LENGTHS.iter().sum();
        let mut kit = kit(gpu, total);
        let kernel = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let sums = SumSquares::new(gpu, &mut kit.cache).unwrap();
        let clean = seeded_values(3, total, -1.0, 1.0);
        let mut cases: Vec<(String, f32, Vec<f32>)> =
            vec![("all finite".into(), 0.5, clean.clone())];
        let middle = LENGTHS[0] + LENGTHS[1] + LENGTHS[2] + 100;
        for (name, bad) in [
            ("NaN", f32::NAN),
            ("+inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
            ("a NaN with other bits", f32::from_bits(0xffc0_0001)),
        ] {
            for (place, at) in [("first", 0), ("middle", middle), ("last", total - 1)] {
                let mut flat = clean.clone();
                flat[at] = bad;
                cases.push((format!("{name} in the {place} value"), 0.5, flat));
            }
            cases.push((format!("a {name} loss"), bad, clean.clone()));
        }
        for (name, edge) in [
            ("f32::MAX", f32::MAX),
            ("-f32::MAX", -f32::MAX),
            ("the smallest subnormal", f32::from_bits(1)),
            ("-0", -0.0),
            ("1e30", 1e30),
        ] {
            let mut flat = clean.clone();
            flat[middle] = edge;
            cases.push((format!("{name} in a gradient"), edge, flat));
        }
        for (name, loss, flat) in &cases {
            let got = gpu_flag(&mut kit, &kernel, *loss, flat, &LENGTHS);
            assert_eq!(got, cpu_flag(*loss, flat), "the finite flag with {name}");
        }
        let mut flat = clean.clone();
        flat[middle] = 1e30;
        let squares = gpu_sums(&mut kit, &sums, &flat, &LENGTHS);
        assert_eq!(squares[3], f32::INFINITY, "the overflowed sum of squares");
        assert_eq!(
            gpu_flag(&mut kit, &kernel, 0.5, &flat, &LENGTHS),
            0,
            "1e30's flag"
        );
        assert_eq!(
            gpu_flag(&mut kit, &kernel, f32::NAN, &[], &[]),
            1,
            "a NaN loss and no gradients"
        );
        assert_eq!(
            gpu_flag(&mut kit, &kernel, 1.0, &[], &[]),
            0,
            "no gradients"
        );
    });
}

/// The GPU's dead-ReLU bits of `y` (shaped `dims`) at word 1 of a buffer of `old` words
/// (sentinels around them), overwriting or OR-ing into them.
fn gpu_bits(
    kit: &mut Kit,
    kernel: &DeadRelu,
    dims: ChannelDims,
    y: &[f32],
    old: &[u32],
    accumulate: bool,
) -> Vec<u32> {
    let yb = kit.upload(y);
    let initial: Vec<f32> = std::iter::once(SENTINEL)
        .chain(old.iter().map(|&w| f32::from_bits(w)))
        .chain(std::iter::once(SENTINEL))
        .collect();
    let out = kit.upload(&initial);
    let call = kernel.call(
        &mut kit.steps,
        &mut kit.pool,
        dims,
        &yb,
        &out,
        1,
        accumulate,
    );
    let mut step = kit.steps.begin();
    kernel.record(&mut step, &call);
    let _ = step.finish();
    let got = words(&kit.read(&out));
    call.release(&mut kit.pool);
    let n = old.len();
    assert_eq!(
        got[0],
        SENTINEL.to_bits(),
        "the word before the bits changed"
    );
    assert_eq!(
        got[n + 1],
        SENTINEL.to_bits(),
        "the word after the bits changed"
    );
    got[1..=n].to_vec()
}

/// The CPU's rule: a channel is alive when any of its values is not 0 (`v != 0.0`, so −0 is
/// 0, and NaN and subnormals are not), as bits packed 32 to a word.
fn cpu_bits(dims: ChannelDims, y: &[f32]) -> Vec<u32> {
    let c = dims.channels as usize;
    let mut alive = vec![false; c];
    for (i, &v) in y.iter().enumerate() {
        alive[i / dims.inner as usize % c] |= v != 0.0;
    }
    let mut out = vec![0u32; DeadRelu::words(dims.channels) as usize];
    for (k, &a) in alive.iter().enumerate() {
        if a {
            out[k / 32] |= 1 << (k % 32);
        }
    }
    out
}

/// The dead-ReLU bits, exactly as the CPU's rule gives them:
/// - for 1, 31, 32, 33 and 70 channels, dense and as convolutions;
/// - every channel otherwise 0 or −0, some with a single non-zero value (a positive value, a
///   NaN, the smallest subnormal) at its first or last position;
/// - overwriting old bits, and OR-ed into them.
#[test]
fn dead_relu_twin() {
    for_each_gpu(|gpu| {
        for channels in [1u32, 31, 32, 33, 70] {
            for dims in [
                ChannelDims::rows(5, channels),
                ChannelDims::new(3, channels, 7),
            ] {
                let n = dims.len() as usize;
                let mut kit = kit(gpu, n);
                let kernel = DeadRelu::new(gpu, &mut kit.cache).unwrap();
                let zeros = seeded_values(u64::from(channels), n, -1.0, 1.0);
                let mut y: Vec<f32> = zeros
                    .iter()
                    .map(|&v| if v < 0.0 { -0.0 } else { 0.0 })
                    .collect();
                let mut marks = [0.5f32, f32::NAN, f32::from_bits(1), 3.0]
                    .into_iter()
                    .cycle();
                let c = channels as usize;
                let inner = dims.inner as usize;
                for k in (0..c).filter(|k| k % 3 != 1) {
                    let last = ((dims.outer as usize - 1) * c + k) * inner + inner - 1;
                    let at = if k % 2 == 0 { k * inner } else { last };
                    y[at] = marks.next().unwrap();
                }
                let want = cpu_bits(dims, &y);
                let words = want.len();
                let what = format!("{dims:?}");
                let old = vec![0xffff_ffffu32; words];
                assert_eq!(
                    gpu_bits(&mut kit, &kernel, dims, &y, &old, false),
                    want,
                    "bits overwriting, {what}"
                );
                let old: Vec<u32> = (0..words as u32)
                    .map(|w| 0x1234_5678u32.rotate_left(w))
                    .collect();
                let got = gpu_bits(&mut kit, &kernel, dims, &y, &old, true);
                let mask = |w: usize| {
                    let bits = (c - 32 * w).min(32);
                    if bits == 32 {
                        u32::MAX
                    } else {
                        (1u32 << bits) - 1
                    }
                };
                for w in 0..words {
                    assert_eq!(
                        got[w],
                        old[w] | want[w],
                        "word {w} OR-ed into old bits, {what}"
                    );
                    assert_eq!(
                        got[w] & !mask(w),
                        old[w] & !mask(w),
                        "word {w}'s bits past the channels, {what}"
                    );
                }
            }
        }
    });
}

/// The bits over epochs, as the dead-ReLU check reads them: OR-ed step by step into a buffer
/// cleared at the epoch's start, a channel's bit is set once any step's batch had a non-zero
/// value in it, and stays set.
#[test]
fn dead_relu_bits_accumulate_over_an_epoch() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::new(2, 40, 3);
        let n = dims.len() as usize;
        let mut kit = kit(gpu, n);
        let kernel = DeadRelu::new(gpu, &mut kit.cache).unwrap();
        let (yb, out) = (kit.zeros(n), kit.zeros(2));
        let call = kernel.call(&mut kit.steps, &mut kit.pool, dims, &yb, &out, 0, true);
        let mut want = [0u32; 2];
        for step_k in 0..3usize {
            let mut y = vec![0.0f32; n];
            for k in (step_k..40).step_by(7) {
                y[(40 + k) * 3 + 1] = 1.0;
                want[k / 32] |= 1 << (k % 32);
            }
            let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut step = kit.steps.begin();
            step.upload(yb.buffer(), 0, &bytes);
            kernel.record(&mut step, &call);
            let _ = step.finish();
            assert_eq!(words(&kit.read(&out)), want, "after step {step_k}");
        }
        call.release(&mut kit.pool);
    });
}

/// The documented refusals: an empty tensor, a tensor past its buffer, an output too short
/// for its offset, and an output that is one of the buffers read.
#[test]
fn refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu, 64);
        let sums = SumSquares::new(gpu, &mut kit.cache).unwrap();
        let flag = FiniteFlag::new(gpu, &mut kit.cache).unwrap();
        let dead = DeadRelu::new(gpu, &mut kit.cache).unwrap();
        let (g, out, one) = (kit.zeros(16), kit.zeros(4), kit.zeros(1));
        let mut sum = |range: (&PooledBuffer, u32, u32), out: &PooledBuffer, offset: u32| {
            message(&mut || {
                let _ = sums.call(&mut kit.steps, &mut kit.pool, range, out, offset);
            })
        };
        assert!(sum((&g, 0, 0), &out, 0).contains("at least one value"));
        assert!(sum((&g, 10, 7), &out, 0).contains("the tensor needs 17 values"));
        assert!(sum((&g, 0, 16), &out, 4).contains("the output needs 5 values"));
        assert!(sum((&g, 0, 16), &g, 0).contains("different buffers"));
        let mut finite =
            |loss: (&PooledBuffer, u32), grads: &[(&PooledBuffer, u32, u32)], f: &PooledBuffer| {
                message(&mut || {
                    let _ = flag.call(&mut kit.steps, &mut kit.pool, loss, grads, f);
                })
            };
        assert!(finite((&out, 4), &[], &one).contains("the loss needs 5 values"));
        assert!(finite((&out, 0), &[], &out).contains("the flag and the loss must be different"));
        assert!(
            finite((&out, 0), &[(&g, 0, 4), (&one, 0, 1)], &one)
                .contains("the flag and gradient 1 must be different")
        );
        assert!(finite((&out, 0), &[(&g, 12, 8)], &one).contains("needs 20 values"));
        let mut bits = |dims: ChannelDims, y: &PooledBuffer, out: &PooledBuffer, offset: u32| {
            message(&mut || {
                let _ = dead.call(&mut kit.steps, &mut kit.pool, dims, y, out, offset, false);
            })
        };
        assert!(bits(ChannelDims::rows(2, 33), &g, &one, 0).contains("the output needs 2 values"));
        assert!(bits(ChannelDims::rows(2, 3), &g, &out, 4).contains("the output needs 5 values"));
        assert!(
            bits(ChannelDims::rows(2, 3), &g, &g, 0)
                .contains("the output and the ReLU output must be different")
        );
        assert!(bits(ChannelDims::rows(0, 3), &g, &out, 0).contains("above 0"));
    });
}
