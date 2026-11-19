//! The health values against the CPU health checks' rules, as `cavitation-gpu`'s tests check
//! its WGSL kernels:
//!
//! - **Sums of squares:** the bits of the fixed tree evaluated on the host, the CPU's `f32`
//!   sums on integers, and within the WGSL bound of the exact sum on seeded values, for
//!   tensors one after another in a flat buffer, written at their word offsets; NaN and ±∞
//!   values give NaN and +∞.
//! - **Finite flag:** exact against `GradStats`' and `OnNonFinite::Skip`'s rule (the loss and
//!   every gradient value finite), with NaN and infinity in the first, a middle and the last
//!   value, the extremes of the finite range, and a gradient whose sum of squares overflows;
//!   and the flag it writes is the one `OptimizerKernel` reads.
//! - **Dead-ReLU bits:** exact against `HealthMonitor`'s `v != 0.0`, for channel counts around
//!   a word's 32, dense and convolution shapes, −0, NaN and subnormals, overwriting and OR-ing
//!   into old bits.
//!
//! The `cifar-cnn` network's 16 parameter gradients (1,117,162 values) and its ReLU outputs at
//! batch 128 are covered too. Outputs start as NaN, or as sentinel words for the `u32`s, and
//! two runs give the same bits. `HealthMonitor` lives in `cavitation-nn`, whose rules are
//! transcribed here. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes
//! that a failure.

mod common;

use cavitation_cuda::{
    ChannelDims, Cuda, CudaError, DeadRelu, FiniteFlag, OptimizerKernel, OptimizerRule, SumSquares,
};
use common::{
    FLOOR, U, assert_exact, assert_within, exact_values, nans, panic_message, read, read_u32,
    seeded_values, upload,
};

/// What fills the `f32` words around a result, to show nothing else is written.
const SENTINEL: f32 = 7.0;

/// What fills the `u32` words of a flag or bits before a run, and around them.
const SENTINEL_WORD: u32 = 0xDEAD_BEEF;

/// Lengths of the tensors laid one after another in a flat gradient buffer: one value, a
/// small tensor, exactly one chunk (4,096), one more, and several chunks.
const LENGTHS: [usize; 5] = [1, 37, 4096, 4097, 100_000];

/// The parameters of the `cifar-cnn` network in the order it adds them: four 3 × 3
/// convolutions without a bias, each followed by a batch norm's `γ` and `β`, then
/// `Linear(4096 → 256)` and `Linear(256 → 10)` with their biases; 1,117,162 floats.
const CIFAR: [usize; 16] = [
    3 * 32 * 9,
    32,
    32,
    32 * 32 * 9,
    32,
    32,
    32 * 64 * 9,
    64,
    64,
    64 * 64 * 9,
    64,
    64,
    4096 * 256,
    256,
    256 * 10,
    10,
];

/// The error bound of a sum of `k` terms in any order: `2(k + 2) · u · Σ|terms|`.
fn sum_bound(k: usize, abs_sum: f64) -> f64 {
    2.0 * (k as f64 + 2.0) * U * abs_sum
}

/// The sum of `terms` in the channel sums' fixed tree, evaluated on the host: chunks of
/// 4,096, each of 256 lanes adding every 256th term of its chunk in order from +0, the lanes
/// meeting in a halving tree, and the chunks added in order.
fn tree(terms: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for (index, chunk) in terms.chunks(4096).enumerate() {
        let mut lanes = [0.0f32; 256];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for t in (lane..chunk.len()).step_by(256) {
                acc += chunk[t];
            }
            *slot = acc;
        }
        let mut width = 128;
        while width > 0 {
            for lane in 0..width {
                lanes[lane] += lanes[lane + width];
            }
            width /= 2;
        }
        total = if index == 0 {
            lanes[0]
        } else {
            total + lanes[0]
        };
    }
    total
}

/// The tensors `lens` long one after another in `flat`.
fn split<'a>(flat: &'a [f32], lens: &[usize]) -> Vec<&'a [f32]> {
    let mut start = 0;
    lens.iter()
        .map(|&len| {
            let t = &flat[start..start + len];
            start += len;
            t
        })
        .collect()
}

/// Asserts that two runs gave the same bits, NaNs included.
fn assert_same_bits(again: &[f32], first: &[f32], what: &str) {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert!(bits(again) == bits(first), "{what}: the bits differ");
}

/// The card's sums of squares of the tensors `kernel` was made for, in `flat`, written to
/// words 1, 2, … of a buffer of NaNs between two sentinels.
fn card_sums(cuda: &Cuda, kernel: &SumSquares<'_>, flat: &[f32]) -> Vec<f32> {
    let k = kernel.tensors();
    let grads = upload(cuda, flat);
    let mut initial = vec![f32::NAN; k + 2];
    initial[0] = SENTINEL;
    initial[k + 1] = SENTINEL;
    let mut out = upload(cuda, &initial);
    kernel.run(&grads, (&mut out, 1)).expect("run");
    let got = read(&out, k + 2);
    assert_eq!(got[0], SENTINEL, "the word before the sums changed");
    assert_eq!(got[k + 1], SENTINEL, "the word after the sums changed");
    got[1..=k].to_vec()
}

/// Sums of squares:
/// - on small integers every order is exact, so each tensor's is the CPU's `f32` sum bit for
///   bit;
/// - on seeded values each is the host's evaluation of the fixed tree bit for bit, within
///   `(2(L + 2) + 1) · u · Σ g² + (3L − 1) · 2⁻¹²⁶` of the exact sum, and the same on a
///   second run;
/// - a NaN gives NaN, ±∞ gives +∞, and the other tensors are unchanged.
#[test]
fn sums_of_squares_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let kernel = SumSquares::new(&cuda, &LENGTHS).expect("compile");
    let total: usize = LENGTHS.iter().sum();
    let flat = exact_values(1, total);
    let want: Vec<f32> = split(&flat, &LENGTHS)
        .iter()
        .map(|t| t.iter().map(|g| g * g).sum())
        .collect();
    assert_exact(
        &card_sums(&cuda, &kernel, &flat),
        &want,
        "sums of squares on integers",
    );
    let flat = seeded_values(2, total, -3.0, 3.0);
    let tensors = split(&flat, &LENGTHS);
    let host: Vec<f32> = tensors
        .iter()
        .map(|t| tree(&t.iter().map(|g| g * g).collect::<Vec<_>>()))
        .collect();
    let exact: Vec<f64> = tensors
        .iter()
        .map(|t| t.iter().map(|&g| f64::from(g).powi(2)).sum())
        .collect();
    let got = card_sums(&cuda, &kernel, &flat);
    assert_exact(&got, &host, "sums of squares against the host's tree");
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
    assert_same_bits(
        &card_sums(&cuda, &kernel, &flat),
        &got,
        "sums of squares on a second run",
    );
    let starts: Vec<usize> = LENGTHS
        .iter()
        .scan(0, |s, &l| {
            let start = *s;
            *s += l;
            Some(start)
        })
        .collect();
    for (bad, want_bad) in [
        (f32::NAN, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
    ] {
        let mut flat = flat.clone();
        flat[starts[3] + 2000] = bad;
        let got_bad = card_sums(&cuda, &kernel, &flat);
        let mut want = host.clone();
        want[3] = want_bad;
        assert_exact(&got_bad, &want, &format!("sums of squares with {bad}"));
    }
}

/// The card's finite flag of `loss` and the tensors `kernel` was made for in `flat`, as a
/// `u32`, from a flag word that starts as a sentinel with a sentinel after it.
fn card_flag(cuda: &Cuda, kernel: &FiniteFlag<'_>, loss: f32, flat: &[f32]) -> u32 {
    let losses = upload(cuda, &[SENTINEL, loss]);
    let grads = upload(cuda, if flat.is_empty() { &[0.0f32][..] } else { flat });
    let mut flag = upload(cuda, &[SENTINEL_WORD, SENTINEL_WORD]);
    kernel.run((&losses, 1), &grads, &mut flag).expect("run");
    let got = read_u32(&flag, 2);
    assert_eq!(got[1], SENTINEL_WORD, "the word after the flag changed");
    got[0]
}

/// The CPU's rule: a step is bad when its loss or any gradient value is not finite.
fn cpu_flag(loss: f32, flat: &[f32]) -> u32 {
    u32::from(!(loss.is_finite() && flat.iter().all(|g| g.is_finite())))
}

/// The finite flag, bit for bit, against the CPU's rule:
/// - a NaN, +∞ or −∞ in the first value, a middle value or the very last value of the last
///   tensor sets it, and the count of the tensor that holds it is 1, the others 0;
/// - so does a NaN or infinite loss with finite gradients;
/// - the largest finite value, the smallest subnormal and −0 leave it clear, as does a
///   gradient of 10³⁰, whose sum of squares overflows to infinity.
#[test]
fn finite_flag_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let total: usize = LENGTHS.iter().sum();
    let kernel = FiniteFlag::new(&cuda, &LENGTHS).expect("compile");
    let sums = SumSquares::new(&cuda, &LENGTHS).expect("compile");
    let clean = seeded_values(3, total, -1.0, 1.0);
    let mut cases: Vec<(String, f32, Vec<f32>, Option<usize>)> =
        vec![("all finite".into(), 0.5, clean.clone(), None)];
    let middle = LENGTHS[0] + LENGTHS[1] + LENGTHS[2] + 100;
    for (name, bad) in [
        ("NaN", f32::NAN),
        ("+inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("a NaN with other bits", f32::from_bits(0xffc0_0001)),
    ] {
        for (place, at, tensor) in [
            ("first", 0, 0),
            ("middle", middle, 3),
            ("last", total - 1, 4),
        ] {
            let mut flat = clean.clone();
            flat[at] = bad;
            cases.push((
                format!("{name} in the {place} value"),
                0.5,
                flat,
                Some(tensor),
            ));
        }
        cases.push((format!("a {name} loss"), bad, clean.clone(), None));
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
        cases.push((format!("{name} in a gradient"), edge, flat, None));
    }
    for (name, loss, flat, tensor) in &cases {
        let got = card_flag(&cuda, &kernel, *loss, flat);
        assert_eq!(got, cpu_flag(*loss, flat), "the finite flag with {name}");
        let counts = read(kernel.counts(), LENGTHS.len());
        let want: Vec<f32> = (0..LENGTHS.len())
            .map(|k| f32::from(u8::from(*tensor == Some(k))))
            .collect();
        assert_exact(&counts, &want, &format!("the counts with {name}"));
    }
    let mut flat = clean.clone();
    flat[middle] = 1e30;
    let squares = card_sums(&cuda, &sums, &flat);
    assert_eq!(squares[3], f32::INFINITY, "the overflowed sum of squares");
    assert_eq!(card_flag(&cuda, &kernel, 0.5, &flat), 0, "1e30's flag");
    let loss_alone = FiniteFlag::new(&cuda, &[]).expect("compile");
    assert_eq!(
        card_flag(&cuda, &loss_alone, f32::NAN, &[]),
        1,
        "a NaN loss and no gradients"
    );
    assert_eq!(
        card_flag(&cuda, &loss_alone, f32::NEG_INFINITY, &[]),
        1,
        "a −∞ loss and no gradients"
    );
    assert_eq!(card_flag(&cuda, &loss_alone, 1.0, &[]), 0, "no gradients");
}

/// The flag `FiniteFlag` writes is the one `OptimizerKernel` reads, word 0 of the same buffer
/// as a `u32`: a NaN gradient, an infinite gradient or a NaN loss sets it to 1, and Adam then
/// skips the step (every value, moment and count as it was, `commit` false); with every value
/// finite it is 0 and the step runs.
#[test]
fn the_flag_skips_the_optimiser() {
    let Some(cuda) = Cuda::new() else { return };
    let lens = [300usize, 37, 1];
    let n: usize = lens.iter().sum();
    let check = FiniteFlag::new(&cuda, &lens).expect("compile");
    let mut adam =
        OptimizerKernel::new(&cuda, OptimizerRule::adam(1e-3, 0.0), &lens).expect("compile");
    let start = seeded_values(20, n, -1.0, 1.0);
    let mut values = upload(&cuda, &start);
    let (mut m, mut v) = (
        upload(&cuda, &vec![0.0f32; n]),
        upload(&cuda, &vec![0.0f32; n]),
    );
    let mut flag = upload(&cuda, &[SENTINEL_WORD]);
    let clean = seeded_values(21, n, -1.0, 1.0);
    let mut nan = clean.clone();
    nan[300 + 5] = f32::NAN;
    let mut inf = clean.clone();
    inf[n - 1] = f32::NEG_INFINITY;
    let mut steps = 0;
    for (name, loss, g, runs) in [
        ("a NaN gradient", 0.5f32, &nan, false),
        ("finite values", 0.5, &clean, true),
        ("an infinite gradient", 0.5, &inf, false),
        ("a NaN loss", f32::NAN, &clean, false),
        ("finite values again", 0.25, &clean, true),
    ] {
        let losses = upload(&cuda, &[loss]);
        let grads = upload(&cuda, g);
        let before = [read(&values, n), read(&m, n), read(&v, n)];
        check.run((&losses, 0), &grads, &mut flag).expect("flag");
        assert_eq!(read_u32(&flag, 1), [u32::from(!runs)], "{name}: the flag");
        adam.update(
            &mut values,
            &grads,
            &mut [&mut m, &mut v],
            &flag,
            &[true; 3],
        )
        .expect("update");
        assert_eq!(
            adam.commit().expect("commit"),
            runs,
            "{name}: whether the step ran"
        );
        steps += usize::from(runs);
        assert!(
            adam.steps().iter().all(|&c| c as usize == steps),
            "{name}: the counts"
        );
        let after = [read(&values, n), read(&m, n), read(&v, n)];
        for s in 0..3 {
            if runs {
                assert!(
                    after[s].iter().zip(&before[s]).any(|(a, b)| a != b),
                    "{name}: buffer {s} did not move"
                );
            } else {
                assert_same_bits(&after[s], &before[s], &format!("{name}: buffer {s}"));
            }
        }
    }
}

/// The card's dead-ReLU bits of `y` at words 1.. of a buffer holding `old` between two
/// sentinels, overwriting or OR-ing into them.
fn card_bits(
    cuda: &Cuda,
    kernel: &DeadRelu<'_>,
    y: &[f32],
    old: &[u32],
    accumulate: bool,
) -> Vec<u32> {
    let yb = upload(cuda, y);
    let initial: Vec<u32> = std::iter::once(SENTINEL_WORD)
        .chain(old.iter().copied())
        .chain(std::iter::once(SENTINEL_WORD))
        .collect();
    let mut out = upload(cuda, &initial);
    kernel.run(&yb, (&mut out, 1), accumulate).expect("run");
    let n = old.len();
    let got = read_u32(&out, n + 2);
    assert_eq!(got[0], SENTINEL_WORD, "the word before the bits changed");
    assert_eq!(got[n + 1], SENTINEL_WORD, "the word after the bits changed");
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
///   NaN, the smallest subnormal, +∞) at its first or last position;
/// - overwriting old bits (sentinel words), and OR-ed into them, the bits past the channels
///   kept; the same on a second run.
#[test]
fn dead_relu_twin() {
    let Some(cuda) = Cuda::new() else { return };
    for channels in [1u32, 31, 32, 33, 70] {
        for dims in [
            ChannelDims::rows(5, channels),
            ChannelDims::new(3, channels, 7),
        ] {
            let n = dims.len() as usize;
            let kernel = DeadRelu::new(&cuda, dims).expect("compile");
            let zeros = seeded_values(u64::from(channels), n, -1.0, 1.0);
            let mut y: Vec<f32> = zeros
                .iter()
                .map(|&v| if v < 0.0 { -0.0 } else { 0.0 })
                .collect();
            let mut marks = [0.5f32, f32::NAN, f32::from_bits(1), f32::INFINITY]
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
            let old = vec![SENTINEL_WORD; words];
            let got = card_bits(&cuda, &kernel, &y, &old, false);
            assert_eq!(got, want, "bits overwriting, {what}");
            assert_eq!(
                card_bits(&cuda, &kernel, &y, &old, false),
                got,
                "a second run, {what}"
            );
            let old: Vec<u32> = (0..words as u32)
                .map(|w| 0x1234_5678u32.rotate_left(w))
                .collect();
            let got = card_bits(&cuda, &kernel, &y, &old, true);
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
}

/// The bits over epochs, as the dead-ReLU check reads them: OR-ed step by step into a buffer
/// cleared at the epoch's start, a channel's bit is set once any step's batch had a non-zero
/// value in it, and stays set.
#[test]
fn dead_relu_bits_accumulate_over_an_epoch() {
    let Some(cuda) = Cuda::new() else { return };
    let dims = ChannelDims::new(2, 40, 3);
    let n = dims.len() as usize;
    let kernel = DeadRelu::new(&cuda, dims).expect("compile");
    let mut yb = upload(&cuda, &vec![0.0f32; n]);
    let mut out = upload(&cuda, &[0u32; 2]);
    let mut want = [0u32; 2];
    for step in 0..3usize {
        let mut y = vec![0.0f32; n];
        for k in (step..40).step_by(7) {
            y[(40 + k) * 3 + 1] = 1.0;
            want[k / 32] |= 1 << (k % 32);
        }
        yb.write(&y).expect("write");
        kernel.run(&yb, (&mut out, 0), true).expect("run");
        assert_eq!(read_u32(&out, 2), want, "after step {step}");
    }
}

/// The `cifar-cnn` network at batch 128:
/// - its 16 parameter gradients (1,117,162 values): each sum of squares the host's tree bit
///   for bit and within its bound, the flag clear, then set by a NaN in the last value of the
///   last tensor and by a +∞ in the first value of the first, each counted in its own tensor;
/// - the bits of its ReLU outputs, 128 × 32 × 32 × 32, 128 × 64 × 16 × 16 and 128 × 256, with
///   some channels dead.
///
/// Two runs give the same bits.
#[test]
fn cifar_sizes() {
    let Some(cuda) = Cuda::new() else { return };
    let total: usize = CIFAR.iter().sum();
    assert_eq!(total, 1_117_162);
    let flat = seeded_values(30, total, -0.05, 0.05);
    let sums = SumSquares::new(&cuda, &CIFAR).expect("compile");
    let got = card_sums(&cuda, &sums, &flat);
    let tensors = split(&flat, &CIFAR);
    let host: Vec<f32> = tensors
        .iter()
        .map(|t| tree(&t.iter().map(|g| g * g).collect::<Vec<_>>()))
        .collect();
    assert_exact(
        &got,
        &host,
        "the cifar-cnn sums of squares against the host's tree",
    );
    let exact: Vec<f64> = tensors
        .iter()
        .map(|t| t.iter().map(|&g| f64::from(g).powi(2)).sum())
        .collect();
    let worst = assert_within(
        &got,
        &exact,
        |k| {
            let l = CIFAR[k];
            sum_bound(l, exact[k]) + U * exact[k] + (3 * l - 2) as f64 * FLOOR
        },
        "the cifar-cnn sums of squares",
    );
    println!(
        "cifar-cnn sums of squares: worst {:.3} of its bound",
        worst.ratio()
    );
    assert_same_bits(&card_sums(&cuda, &sums, &flat), &got, "a second run");
    let check = FiniteFlag::new(&cuda, &CIFAR).expect("compile");
    assert_eq!(card_flag(&cuda, &check, 2.3, &flat), 0, "the clean flag");
    for (at, tensor, bad) in [(total - 1, 15, f32::NAN), (0, 0, f32::INFINITY)] {
        let mut flat = flat.clone();
        flat[at] = bad;
        assert_eq!(card_flag(&cuda, &check, 2.3, &flat), 1, "{bad} at {at}");
        let counts = read(check.counts(), CIFAR.len());
        let want: Vec<f32> = (0..CIFAR.len())
            .map(|k| f32::from(u8::from(k == tensor)))
            .collect();
        assert_exact(&counts, &want, &format!("the counts with {bad} at {at}"));
    }
    for dims in [
        ChannelDims::new(128, 32, 32 * 32),
        ChannelDims::new(128, 64, 16 * 16),
        ChannelDims::rows(128, 256),
    ] {
        let kernel = DeadRelu::new(&cuda, dims).expect("compile");
        let c = dims.channels as usize;
        let inner = dims.inner as usize;
        // A ReLU's output: half the values 0, and every fifth channel dead throughout.
        let y: Vec<f32> = seeded_values(u64::from(dims.channels), dims.len() as usize, -1.0, 1.0)
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if ((i / inner) % c).is_multiple_of(5) {
                    0.0
                } else {
                    v.max(0.0)
                }
            })
            .collect();
        let want = cpu_bits(dims, &y);
        let old = vec![SENTINEL_WORD; want.len()];
        let got = card_bits(&cuda, &kernel, &y, &old, false);
        assert_eq!(got, want, "the bits of {dims:?}");
        assert_eq!(
            card_bits(&cuda, &kernel, &y, &old, false),
            got,
            "a second run"
        );
    }
}

/// The refusals: no gradients for a sum of squares, an empty gradient, more than 2³¹ values
/// together, and a shape the channel sums refuse are `CudaError::Invalid`; a gradient buffer
/// shorter than its tensors, an output too short for its offset, a loss word past its buffer,
/// a flag buffer without a word and a ReLU output shorter than its shape panic before the
/// launch, naming the buffer.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let invalid = |result: Result<(), CudaError>, why: &str| {
        let Err(CudaError::Invalid(text)) = result else {
            panic!("accepted where {why} was expected");
        };
        assert!(text.contains(why), "{text}");
    };
    invalid(
        SumSquares::new(&cuda, &[]).map(drop),
        "at least one gradient",
    );
    invalid(
        SumSquares::new(&cuda, &[3, 0]).map(drop),
        "tensor 1 holds no values",
    );
    invalid(
        FiniteFlag::new(&cuda, &[0]).map(drop),
        "tensor 0 holds no values",
    );
    invalid(
        FiniteFlag::new(&cuda, &[1 << 31, 1]).map(drop),
        "more than the 2147483648",
    );
    invalid(
        DeadRelu::new(&cuda, ChannelDims::rows(0, 3)).map(drop),
        "above 0",
    );
    invalid(
        DeadRelu::new(&cuda, ChannelDims::rows(2, 70_000)).map(drop),
        "more than 65535 channels",
    );
    let sums = SumSquares::new(&cuda, &[10, 7]).expect("compile");
    let g = nans(&cuda, 16);
    let mut out = nans(&cuda, 4);
    let text = panic_message(|| {
        let _ = sums.run(&g, (&mut out, 0));
    });
    assert!(
        text.contains("the gradients buffer holds 64 bytes, and 17 values"),
        "{text}"
    );
    let g = nans(&cuda, 17);
    let text = panic_message(|| {
        let _ = sums.run(&g, (&mut out, 3));
    });
    assert!(
        text.contains("the output buffer holds 16 bytes, and 5 values"),
        "{text}"
    );
    let flag_kernel = FiniteFlag::new(&cuda, &[4]).expect("compile");
    let loss = nans(&cuda, 4);
    let mut flag = upload(&cuda, &[SENTINEL_WORD]);
    let text = panic_message(|| {
        let _ = flag_kernel.run((&loss, 4), &g, &mut flag);
    });
    assert!(
        text.contains("the loss buffer holds 16 bytes, and 5 values"),
        "{text}"
    );
    let short = nans(&cuda, 3);
    let text = panic_message(|| {
        let _ = flag_kernel.run((&loss, 0), &short, &mut flag);
    });
    assert!(
        text.contains("the gradients buffer holds 12 bytes, and 4 values"),
        "{text}"
    );
    let mut empty = cuda.buffer(2).expect("allocate");
    let text = panic_message(|| {
        let _ = flag_kernel.run((&loss, 0), &g, &mut empty);
    });
    assert!(
        text.contains("the flag buffer holds 2 bytes, and 1 values"),
        "{text}"
    );
    assert_eq!(read_u32(&flag, 1), [SENTINEL_WORD], "a refused run wrote");
    let dead = DeadRelu::new(&cuda, ChannelDims::rows(2, 33)).expect("compile");
    let y = nans(&cuda, 66);
    let mut bits = upload(&cuda, &[SENTINEL_WORD; 2]);
    let text = panic_message(|| {
        let _ = dead.run(&y, (&mut bits, 1), false);
    });
    assert!(
        text.contains("the output buffer holds 8 bytes, and 3 values"),
        "{text}"
    );
    let text = panic_message(|| {
        let _ = dead.run(&short, (&mut bits, 0), false);
    });
    assert!(
        text.contains("the ReLU output buffer holds 12 bytes, and 66 values"),
        "{text}"
    );
    assert_eq!(
        read_u32(&bits, 2),
        [SENTINEL_WORD; 2],
        "a refused run wrote"
    );
}
