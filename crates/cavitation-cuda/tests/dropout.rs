//! Dropout and the neuron mask against their CPU twins, as `cavitation-gpu`'s tests check its
//! WGSL kernels: dropout gives the bits of the CPU's `x · m` (a dropped −3 is −0, a dropped
//! NaN or infinity is NaN) and the neuron mask the bits of `Tape::keep_where`'s selection (a
//! cleared −3, −0, NaN or infinity is +0), ±0, ±∞, NaN and subnormals included, on 100,003
//! elements and on the `cifar-cnn` network's dropout inputs. Every output starts as NaN, and
//! two runs give the same bits. Each test skips without CUDA, unless
//! `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

mod common;

use cavitation_core::{Graph, ParamStore, Tensor};
use cavitation_cuda::{Cuda, CudaError, Dropout};
use common::{Seeded, assert_exact, nans, panic_message, read, seeded_values, upload};

/// Values every test includes besides the generated ones: zeros of both signs, infinities,
/// NaN, the extreme normals, subnormals and values around 0.
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

/// `words` seeded mask words.
fn mask(seed: u64, words: usize) -> Vec<u32> {
    let mut rng = Seeded::new(seed);
    (0..words).map(|_| rng.next_u64() as u32).collect()
}

/// Whether element `i`'s bit is set in `words`.
fn bit(words: &[u32], i: usize) -> bool {
    (words[i / 32] >> (i % 32)) & 1 == 1
}

/// Runs dropout (with `scale`) or, without one, the neuron mask over `x` with `words`, into a
/// buffer of NaNs, and returns the result.
fn run(
    cuda: &Cuda,
    kernels: &Dropout<'_>,
    x: &[f32],
    words: &[u32],
    scale: Option<f32>,
) -> Vec<f32> {
    let (xb, mb) = (upload(cuda, x), upload(cuda, words));
    let mut out = nans(cuda, x.len());
    match scale {
        Some(s) => kernels.dropout(&xb, &mb, &mut out, x.len(), s),
        None => kernels.keep_where(&xb, &mb, &mut out, x.len()),
    }
    .expect("run");
    read(&out, x.len())
}

/// Asserts that two runs gave the same bits, NaNs included.
fn assert_same_bits(again: &[f32], first: &[f32], what: &str) {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert!(bits(again) == bits(first), "{what}: the bits differ");
}

/// The CPU's dropout: `x` times the multiplier, `scale` for a kept element and 0 for a
/// dropped one.
fn cpu_dropout(x: &[f32], words: &[u32], scale: f32) -> Vec<f32> {
    (0..x.len())
        .map(|i| x[i] * if bit(words, i) { scale } else { 0.0 })
        .collect()
}

/// The CPU's neuron mask: `x` where the bit is set, +0 where it is clear.
fn cpu_keep_where(x: &[f32], words: &[u32]) -> Vec<f32> {
    (0..x.len())
        .map(|i| if bit(words, i) { x[i] } else { 0.0 })
        .collect()
}

/// Dropout multiplies each element by `scale` where its bit is set and by 0 where it is
/// clear, as the CPU does: a dropped −3 is −0, a dropped NaN or infinity is NaN. The scale is
/// the CPU's `1 / f32(1 − p)`.
#[test]
fn dropout_follows_its_mask() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Dropout::new(&cuda).expect("compile");
    let n = 100_003;
    let mut x = seeded_values(3, n, -4.0, 4.0);
    x[..SPECIAL.len()].copy_from_slice(&SPECIAL);
    x[SPECIAL.len()] = -3.0;
    let mut words = mask(4, Dropout::mask_words(n));
    // The specials and the −3 each meet a cleared bit and, in the next word, a set one.
    let specials = SPECIAL.len() + 1;
    words[0] = 0;
    words[1] = u32::MAX;
    x.copy_within(..specials, 32);
    let p = 0.025f64;
    let scale = Dropout::scale(p).expect("p in [0, 1)");
    assert_eq!(scale, 1.0 / (1.0 - p) as f32, "the CPU's scale");
    let got = run(&cuda, &kernels, &x, &words, Some(scale));
    let want = cpu_dropout(&x, &words, scale);
    assert_eq!(
        want[specials - 1].to_bits(),
        (-0.0f32).to_bits(),
        "a dropped −3"
    );
    assert!(want[4].is_nan() && want[2].is_nan(), "a dropped NaN and +∞");
    assert!(want[32 + 4].is_nan() && want[32 + 2] == f32::INFINITY);
    assert_exact(&got, &want, "dropout");
    let again = run(&cuda, &kernels, &x, &words, Some(scale));
    assert_same_bits(&again, &got, "dropout on a second run");
}

/// A neuron mask selects each element where its bit is set and gives +0 where it is clear,
/// as `Tape::keep_where` does: a cleared −3, −0, NaN or infinity is +0, a kept one keeps its
/// bits.
#[test]
fn keep_where_follows_its_mask() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Dropout::new(&cuda).expect("compile");
    let n = 100_003;
    let mut x = seeded_values(5, n, -4.0, 4.0);
    x[..SPECIAL.len()].copy_from_slice(&SPECIAL);
    x[SPECIAL.len()] = -3.0;
    let mut words = mask(6, Dropout::mask_words(n));
    let specials = SPECIAL.len() + 1;
    words[0] = 0;
    words[1] = u32::MAX;
    x.copy_within(..specials, 32);
    let got = run(&cuda, &kernels, &x, &words, None);
    let want = cpu_keep_where(&x, &words);
    assert!(want[..specials].iter().all(|v| v.to_bits() == 0));
    assert!(want[32..32 + specials].iter().any(|v| v.is_nan()));
    assert_exact(&got, &want, "keep_where");
    // A cleared element is +0 itself, not a NaN or a −0 that compares equal.
    assert!(got[..specials].iter().all(|v| v.to_bits() == 0));
    let again = run(&cuda, &kernels, &x, &words, None);
    assert_same_bits(&again, &got, "keep_where on a second run");
}

/// The twins above are the CPU's own operations: the tape's dropout multipliers and
/// `keep_where` give the same bits on the specials, forward and backward, at p = 0.025.
#[test]
fn the_twins_are_the_tapes() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Dropout::new(&cuda).expect("compile");
    let mut x: Vec<f32> = SPECIAL.to_vec();
    x.push(-3.0);
    x.extend(SPECIAL);
    x.push(-3.0);
    let n = x.len();
    let words = [0xAAAA_5555u32];
    let keep: Vec<bool> = (0..n).map(|i| bit(&words, i)).collect();
    // At 0.025 the CPU's 1 / f32(1 − p) and 1 / (1 − p) rounded once differ, so the tape
    // tells the two roundings apart.
    let p = 0.025;
    let scale = Dropout::scale(p).expect("p");
    assert_ne!(
        scale,
        (1.0 / (1.0 - p)) as f32,
        "p tells the roundings apart"
    );
    let g: Vec<f32> = (0..n).map(|i| [1.5, -0.25, f32::NAN][i % 3]).collect();
    let mut params: ParamStore<f32> = ParamStore::default();
    let id = params.add(Tensor::from_vec(x.clone(), &[n]).expect("shape"));
    let mut graph: Graph<f32> = Graph::default();
    let mut tape = graph.tape(&params.values);
    let xv = tape.param(id);
    let gv = tape.input(Tensor::from_vec(g.clone(), &[n]).expect("shape").view());
    let mask_values: Vec<f32> = keep.iter().map(|&k| f32::from(u8::from(k))).collect();
    let kv = tape.input(Tensor::from_vec(mask_values, &[n]).expect("shape").view());
    let dropped = tape.dropout_with_mask(xv, &keep, p);
    let kept = tape.keep_where(xv, kv);
    let tape_dropout = tape.value(dropped).data().to_vec();
    let tape_keep = tape.value(kept).data().to_vec();
    let both = tape.add(dropped, kept);
    let weighted = tape.mul(both, gv);
    let loss = tape.sum(weighted);
    tape.backward(loss, &mut params.grads);
    drop(tape);
    let tape_grad = params.grad(id).data().to_vec();
    assert_exact(
        &run(&cuda, &kernels, &x, &words, Some(scale)),
        &tape_dropout,
        "dropout against the tape",
    );
    assert_exact(
        &run(&cuda, &kernels, &x, &words, None),
        &tape_keep,
        "keep_where against the tape",
    );
    // Backward: the same calls on the gradient, summed as the tape sums its two branches.
    let back_dropout = run(&cuda, &kernels, &g, &words, Some(scale));
    let back_keep = run(&cuda, &kernels, &g, &words, None);
    let sum: Vec<f32> = back_dropout
        .iter()
        .zip(&back_keep)
        .map(|(a, b)| a + b)
        .collect();
    assert_exact(&sum, &tape_grad, "both backward passes against the tape");
}

/// The `cifar-cnn` network's dropout inputs, 128 × 4096 after its last pool and 128 × 256
/// after its first dense layer, at its rates 0.25 and 0.5, with the neuron mask on the same
/// sizes: the CPU's bits, the same on a second run.
#[test]
fn cifar_sizes() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Dropout::new(&cuda).expect("compile");
    for (n, p, seed) in [(128 * 4096, 0.25, 7), (128 * 256, 0.5, 8)] {
        let x = seeded_values(seed, n, -2.0, 2.0);
        let words = mask(seed + 100, Dropout::mask_words(n));
        let scale = Dropout::scale(p).expect("p");
        let got = run(&cuda, &kernels, &x, &words, Some(scale));
        assert_exact(
            &got,
            &cpu_dropout(&x, &words, scale),
            &format!("dropout of {n}"),
        );
        assert_same_bits(
            &run(&cuda, &kernels, &x, &words, Some(scale)),
            &got,
            &format!("dropout of {n} on a second run"),
        );
        let got = run(&cuda, &kernels, &x, &words, None);
        assert_exact(
            &got,
            &cpu_keep_where(&x, &words),
            &format!("keep_where of {n}"),
        );
        assert_same_bits(
            &run(&cuda, &kernels, &x, &words, None),
            &got,
            &format!("keep_where of {n} on a second run"),
        );
    }
}

/// Lengths around a mask word and a block: every one of the `n` outputs is written and none
/// past them.
#[test]
fn lengths_around_a_word_and_a_block() {
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Dropout::new(&cuda).expect("compile");
    for n in [1usize, 31, 32, 33, 255, 256, 257, 1000] {
        let x = seeded_values(n as u64, n, -1.0, 1.0);
        let words = mask(n as u64 + 1, Dropout::mask_words(n));
        let (xb, mb) = (upload(&cuda, &x), upload(&cuda, &words));
        let mut out = nans(&cuda, n + 1);
        kernels
            .dropout(&xb, &mb, &mut out, n, 2.0)
            .expect("dropout");
        let got = read(&out, n + 1);
        assert!(got[n].is_nan(), "{n}: the value past the end was written");
        assert_exact(
            &got[..n],
            &cpu_dropout(&x, &words, 2.0),
            &format!("dropout of {n}"),
        );
        let mut out = nans(&cuda, n + 1);
        kernels
            .keep_where(&xb, &mb, &mut out, n)
            .expect("keep_where");
        let got = read(&out, n + 1);
        assert!(got[n].is_nan(), "{n}: the value past the end was written");
        assert_exact(
            &got[..n],
            &cpu_keep_where(&x, &words),
            &format!("keep_where of {n}"),
        );
    }
}

/// The refusals: a drop probability outside [0, 1) and a scale that is not finite are
/// `CudaError::Invalid`; a short input, output or mask panics before the launch, naming it;
/// no elements run nothing.
#[test]
fn refusals() {
    for p in [-0.1, 1.0, 1.5, f64::NAN, f64::INFINITY] {
        let Err(CudaError::Invalid(text)) = Dropout::scale(p) else {
            panic!("p = {p} accepted");
        };
        assert!(text.contains("[0, 1)"), "{text}");
    }
    assert_eq!(Dropout::scale(0.0).expect("p = 0"), 1.0);
    let Some(cuda) = Cuda::new() else { return };
    let kernels = Dropout::new(&cuda).expect("compile");
    let x = upload(&cuda, &[1.0f32; 64]);
    let one_word = upload(&cuda, &[u32::MAX]);
    let two_words = upload(&cuda, &[u32::MAX; 2]);
    let mut out = nans(&cuda, 64);
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let Err(CudaError::Invalid(text)) = kernels.dropout(&x, &one_word, &mut out, 8, scale)
        else {
            panic!("a scale of {scale} accepted");
        };
        assert!(text.contains("must be finite"), "{text}");
    }
    kernels
        .dropout(&x, &one_word, &mut out, 0, 2.0)
        .expect("nothing to do");
    kernels
        .keep_where(&x, &one_word, &mut out, 0)
        .expect("nothing to do");
    assert!(
        read(&out, 64).iter().all(|v| v.is_nan()),
        "an empty call wrote"
    );
    let text = panic_message(|| {
        let _ = kernels.dropout(&x, &one_word, &mut out, 33, 2.0);
    });
    assert!(text.contains("the mask buffer holds 4 bytes"), "{text}");
    let text = panic_message(|| {
        let _ = kernels.keep_where(&x, &two_words, &mut out, 65);
    });
    assert!(text.contains("the input buffer holds 256 bytes"), "{text}");
    let mut short = nans(&cuda, 8);
    let text = panic_message(|| {
        let _ = kernels.keep_where(&x, &two_words, &mut short, 9);
    });
    assert!(text.contains("the output buffer holds 32 bytes"), "{text}");
}
