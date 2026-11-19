//! Batch norm on the card against its CPU twin, `Tape::batch_norm2d`, as `cavitation-gpu`'s
//! tests check its WGSL kernels:
//!
//! - the four sums (`Σx`, the centred squares, `Σ dy`, `Σ dy·x̂`) give the bits of the same
//!   fixed tree evaluated on the host, are within the bound of a sum in any order of the
//!   exact result, and on small integers give the CPU's in-order sums bit for bit;
//! - every other kernel (the statistics, the normalisation, the input gradient, the running
//!   update) gives the bits of the CPU's own `f32` expression of the card's inputs;
//! - end to end, on inputs whose sums are exact, the forward pass, the gradients and the
//!   running statistics are the `f32` tape's bits, and on seeded values they are within the
//!   WGSL kernels' composed bounds of the tape on an `f64` store;
//! - NaN appears where the CPU's has it; the finite flag, buffers longer than the tensor and
//!   the refusals.
//!
//! The shapes cover a batch of one, `inner = 1`, more than 256 channels, a channel of odd
//! length over two chunks, and the `cifar-cnn` network's two batch norms at batch 128 and at
//! its last batch of 80. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1`
//! makes that a failure.

mod common;

use cavitation_core::{Graph, ParamStore, RunningStats, Tensor};
use cavitation_cuda::{BatchNorm2d, BatchStats, ChannelDims, Cuda, CudaError, RunningUpdate};
use common::{
    FLOOR, U, assert_exact, assert_within, exact_values, nans, panic_message, read, seeded_values,
    upload,
};

/// The eps every test uses but the exact ones, PyTorch's default.
const EPS: f64 = 1e-5;

/// What fills the values past a tensor in a buffer longer than it.
const SENTINEL: f32 = 7.0;

/// The tolerance of the gradients against the `f64` tape, relative to each one's scale: far
/// above the rounding of either side, far below a wrong term.
const TOLERANCE: f64 = 1.0 / 4096.0;

/// The `cifar-cnn` network's two batch norms at batch 128 and at its last batch of 80
/// (50,000 = 390 · 128 + 80).
fn cifar() -> Vec<ChannelDims> {
    vec![
        ChannelDims::new(128, 32, 1024),
        ChannelDims::new(128, 64, 256),
        ChannelDims::new(80, 32, 1024),
        ChannelDims::new(80, 64, 256),
    ]
}

/// The shapes the kernels run, as `(batch, channels, h · w)`: a small batch, a batch of one
/// whose channels hold an odd 4,097 values (two chunks), a 1000-row matrix of 10 channels
/// (`inner = 1`), 300 channels (two blocks of the per-channel kernels), and [`cifar`].
fn shapes() -> Vec<ChannelDims> {
    let mut all = vec![
        ChannelDims::new(2, 7, 9),
        ChannelDims::new(1, 3, 4097),
        ChannelDims::rows(1000, 10),
        ChannelDims::new(2, 300, 3),
    ];
    all.extend(cifar());
    all
}

/// The channel of value `i` of a tensor shaped `dims`.
fn channel(dims: ChannelDims, i: usize) -> usize {
    i / dims.inner as usize % dims.channels as usize
}

/// `v` widened to `f64`.
fn wide(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `n` values of `γ` for a rounding twin: magnitudes in `[0.5, 2)`, either sign.
fn weights(seed: u64, n: usize) -> Vec<f32> {
    let signs = exact_values(seed, n);
    seeded_values(seed + 1, n, 0.5, 2.0)
        .into_iter()
        .zip(signs)
        .map(|(w, s)| w.copysign(s))
        .collect()
}

/// One run on the card: a forward pass of `x` (or, with `stats` given, statistics uploaded
/// by hand and no forward pass), then with `dy` a backward pass.
#[derive(Clone, Copy)]
struct Case<'a> {
    /// The tensor's shape.
    dims: ChannelDims,
    /// eps.
    eps: f64,
    /// The input.
    x: &'a [f32],
    /// `(γ, β)`, for an affine layer.
    affine: Option<(&'a [f32], &'a [f32])>,
    /// The running `(mean, var)` the forward pass takes, else the batch's.
    running: Option<(&'a [f32], &'a [f32])>,
    /// The output gradient, for a backward pass.
    dy: Option<&'a [f32]>,
    /// Statistics `[mean | inv | unbiased]` uploaded in place of the forward pass's.
    stats: Option<&'a [f32]>,
    /// Values of [`SENTINEL`] past the tensor in every tensor buffer.
    pad: usize,
}

impl<'a> Case<'a> {
    /// A forward pass of `x` shaped `dims` from the batch, without `γ`, `β`, with [`EPS`].
    fn new(dims: ChannelDims, x: &'a [f32]) -> Case<'a> {
        Case {
            dims,
            eps: EPS,
            x,
            affine: None,
            running: None,
            dy: None,
            stats: None,
            pad: 0,
        }
    }
}

/// What a [`Case`] read back.
#[derive(Default)]
struct Run {
    /// The output, with its padding.
    y: Vec<f32>,
    /// The statistics `[mean | inv | unbiased]`, with their padding.
    stats: Vec<f32>,
    /// `(Σx, Σ (x − mean)²)` of a forward pass from the batch.
    sums: (Vec<f32>, Vec<f32>),
    /// The input gradient, with its padding.
    dx: Vec<f32>,
    /// `dγ = Σ dy · x̂`, for an affine layer.
    dw: Vec<f32>,
    /// `dβ = Σ dy`, for an affine layer.
    db: Vec<f32>,
}

/// Runs `case` on the card with `bn`, every output written over NaNs (and [`SENTINEL`]s
/// past the tensor).
fn run(cuda: &Cuda, bn: &mut BatchNorm2d<'_>, case: Case<'_>) -> Run {
    let dims = case.dims;
    let (n, c) = (dims.len() as usize, dims.channels as usize);
    let padded = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .copied()
            .chain(std::iter::repeat_n(SENTINEL, case.pad))
            .collect()
    };
    let unwritten = |len: usize| upload(cuda, &padded(&vec![f32::NAN; len]));
    let x = upload(cuda, &padded(case.x));
    let mut y = unwritten(n);
    let mut stats = match case.stats {
        Some(s) => upload(cuda, &padded(s)),
        None => unwritten(3 * c),
    };
    let affine = case.affine.map(|(w, b)| (upload(cuda, w), upload(cuda, b)));
    let running = case
        .running
        .map(|(m, v)| (upload(cuda, m), upload(cuda, v)));
    let from = match &running {
        Some((mean, var)) => BatchStats::Running { mean, var },
        None => BatchStats::Batch,
    };
    if case.stats.is_none() {
        bn.forward(
            case.eps,
            from,
            &x,
            affine.as_ref().map(|(w, b)| (w, b)),
            &mut y,
            &mut stats,
        )
        .expect("forward");
    }
    let mut out = Run {
        y: read(&y, n + case.pad),
        stats: read(&stats, 3 * c + case.pad),
        sums: (read(bn.sums().0, c), read(bn.sums().1, c)),
        ..Run::default()
    };
    if let Some(dy) = case.dy {
        let dy = upload(cuda, &padded(dy));
        let mut dx = unwritten(n);
        let (mut dw, mut db) = (nans(cuda, c), nans(cuda, c));
        let parts = affine.as_ref().map(|(w, _)| (w, &mut dw, &mut db));
        bn.backward(from, &x, &stats, &dy, &mut dx, parts)
            .expect("backward");
        out.dx = read(&dx, n + case.pad);
        if affine.is_some() {
            out.dw = read(&dw, c);
            out.db = read(&db, c);
        }
    }
    out
}

/// The sums' fixed tree evaluated on the host: for each channel, chunks of 4,096 of its
/// `term(i, c)` values in the order `b · inner + p`, each of 256 lanes adding every 256th
/// term of the chunk in order from +0, the lanes meeting in a halving tree, and the chunks
/// added in order.
fn tree(dims: ChannelDims, term: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let (c, p, l) = (
        dims.channels as usize,
        dims.inner as usize,
        dims.summed() as usize,
    );
    (0..c)
        .map(|k| {
            let mut total = 0.0f32;
            for chunk in 0..l.div_ceil(4096) {
                let mut lanes = [0.0f32; 256];
                for (lane, slot) in lanes.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for t in 0..16 {
                        let j = chunk * 4096 + lane + t * 256;
                        if j < l {
                            acc += term((j / p * c + k) * p + j % p, k);
                        }
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
                total = if chunk == 0 {
                    lanes[0]
                } else {
                    total + lanes[0]
                };
            }
            total
        })
        .collect()
}

/// `m`, `m − 1` and eps as the CPU rounds them to `f32`.
fn rounded(dims: ChannelDims, eps: f64) -> (f32, f32, f32) {
    let m = dims.summed() as f64;
    (m as f32, (m - 1.0) as f32, eps as f32)
}

/// The CPU's `f32` statistics `[mean | inv | unbiased]` from the card's sums, or from running
/// ones.
fn cpu_stats(
    dims: ChannelDims,
    eps: f64,
    sums: (&[f32], &[f32]),
    running: Option<(&[f32], &[f32])>,
) -> Vec<f32> {
    let (m, m1, eps) = rounded(dims, eps);
    let c = dims.channels as usize;
    let mut out = vec![0.0f32; 3 * c];
    for k in 0..c {
        match running {
            Some((rm, rv)) => {
                out[k] = rm[k];
                out[c + k] = 1.0 / (rv[k] + eps).sqrt();
                out[2 * c + k] = 0.0;
            }
            None => {
                let (s, q) = (sums.0[k], sums.1[k]);
                out[k] = s / m;
                out[c + k] = 1.0 / (q / m + eps).sqrt();
                out[2 * c + k] = q / m1;
            }
        }
    }
    out
}

/// The CPU's `f32` normalisation of `x` from `stats`: `(x − mean) · inv · γ + β`, with
/// `γ = 1` and `β = 0` without them.
fn cpu_y(
    dims: ChannelDims,
    x: &[f32],
    stats: &[f32],
    affine: Option<(&[f32], &[f32])>,
) -> Vec<f32> {
    let c = dims.channels as usize;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let k = channel(dims, i);
            let (g, s) = affine.map_or((1.0, 0.0), |(w, b)| (w[k], b[k]));
            (v - stats[k]) * stats[c + k] * g + s
        })
        .collect()
}

/// The CPU's `f32` input gradient from `stats` and the card's `(Σ dy, Σ dy·x̂)`.
fn cpu_dx(
    dims: ChannelDims,
    (x, dy): (&[f32], &[f32]),
    stats: &[f32],
    w: Option<&[f32]>,
    sums: (&[f32], &[f32]),
    batch: bool,
) -> Vec<f32> {
    let c = dims.channels as usize;
    let (m, _, _) = rounded(dims, 0.0);
    dy.iter()
        .enumerate()
        .map(|(i, &d)| {
            let k = channel(dims, i);
            let (mean, inv) = (stats[k], stats[c + k]);
            let g = w.map_or(1.0, |w| w[k]);
            let gy = d * g;
            if !batch {
                return gy * inv;
            }
            let (sg, sgx) = (sums.0[k] * g, sums.1[k] * g);
            let xhat = (x[i] - mean) * inv;
            inv / m * (m * gy - sg - xhat * sgx)
        })
        .collect()
}

/// The sums' bound for `l` terms of magnitudes summing to `abs`: `2(l + 2) · u · abs`.
fn sum_bound(l: usize, abs: f64) -> f64 {
    2.0 * (l as f64 + 2.0) * U * abs
}

/// Every kernel against its twin from the card's own inputs, on seeded values, every shape,
/// with and without `γ`, `β`, from the batch and from running statistics:
/// - the four sums are the host's evaluation of the same tree, bit for bit, and within
///   `2(L + 2) · u · Σ|x|` of the exact sum (the centred squares and `Σ dy·x̂` within
///   `3u · Σ|term|` more for their terms' own roundings);
/// - the statistics, the output and the input gradient are the CPU's `f32` expression of
///   the card's sums and statistics, bit for bit.
#[test]
fn kernel_twins() {
    let Some(cuda) = Cuda::new() else { return };
    let mut worst = [0.0f64; 4];
    for (s, dims) in shapes().into_iter().enumerate() {
        let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
        let (c, s) = (dims.channels as usize, 100 * s as u64);
        let l = dims.summed() as usize;
        let x = seeded_values(s, dims.len() as usize, -2.0, 2.0);
        let dy = seeded_values(s + 1, x.len(), -1.0, 1.0);
        let (w, b) = (weights(s + 2, c), seeded_values(s + 4, c, -1.0, 1.0));
        let (rm, rv) = (
            seeded_values(s + 5, c, -1.0, 1.0),
            seeded_values(s + 6, c, 0.5, 2.0),
        );
        for affine in [Some((&w[..], &b[..])), None] {
            for running in [None, Some((&rm[..], &rv[..]))] {
                let what = format!(
                    "{dims:?}, affine {}, running {}",
                    affine.is_some(),
                    running.is_some()
                );
                let case = Case {
                    affine,
                    running,
                    dy: Some(&dy),
                    ..Case::new(dims, &x)
                };
                let got = run(&cuda, &mut bn, case);
                let stats = &got.stats;
                let (sums, squares) = (&got.sums.0[..], &got.sums.1[..]);
                if running.is_none() {
                    assert_exact(sums, &tree(dims, |i, _| x[i]), &format!("Σx, {what}"));
                    let centred = |i: usize, k: usize| {
                        let d = x[i] - stats[k];
                        d * d
                    };
                    assert_exact(
                        squares,
                        &tree(dims, centred),
                        &format!("centred squares, {what}"),
                    );
                }
                let want = cpu_stats(dims, EPS, (sums, squares), running);
                assert_exact(stats, &want, &format!("statistics, {what}"));
                assert_exact(
                    &got.y,
                    &cpu_y(dims, &x, stats, affine),
                    &format!("y, {what}"),
                );
                let xhat = |i: usize, k: usize| (x[i] - stats[k]) * stats[c + k];
                let (sdy, sdx) = match affine {
                    Some(_) => {
                        assert_exact(&got.db, &tree(dims, |i, _| dy[i]), &format!("dβ, {what}"));
                        let term = |i: usize, k: usize| dy[i] * xhat(i, k);
                        assert_exact(&got.dw, &tree(dims, term), &format!("dγ, {what}"));
                        (got.db.clone(), got.dw.clone())
                    }
                    // Without γ the card's sums are its own scratch: the tree's bits, as
                    // with γ.
                    None => (
                        tree(dims, |i, _| dy[i]),
                        tree(dims, |i, k| dy[i] * xhat(i, k)),
                    ),
                };
                let want = cpu_dx(
                    dims,
                    (&x, &dy),
                    stats,
                    affine.map(|a| a.0),
                    (&sdy, &sdx),
                    running.is_none(),
                );
                assert_exact(&got.dx, &want, &format!("dx, {what}"));

                if affine.is_none() || running.is_some() {
                    continue;
                }
                // The sums against the exact sums of the same f32 terms.
                let mut exact = [vec![0.0f64; c], vec![0.0; c], vec![0.0; c], vec![0.0; c]];
                let mut abs = exact.clone();
                for i in 0..x.len() {
                    let k = channel(dims, i);
                    let mean = f64::from(stats[k]);
                    let xh = (f64::from(x[i]) - mean) * f64::from(stats[c + k]);
                    let terms = [
                        f64::from(x[i]),
                        (f64::from(x[i]) - mean).powi(2),
                        f64::from(dy[i]),
                        f64::from(dy[i]) * xh,
                    ];
                    for (t, term) in terms.into_iter().enumerate() {
                        exact[t][k] += term;
                        abs[t][k] += term.abs();
                    }
                }
                for (t, (got, name)) in [
                    (sums, "Σx"),
                    (squares, "centred squares"),
                    (&got.db[..], "dβ"),
                    (&got.dw[..], "dγ"),
                ]
                .into_iter()
                .enumerate()
                {
                    let own = if t % 2 == 1 { 3.0 * U } else { 0.0 };
                    let w = assert_within(
                        got,
                        &exact[t],
                        |k| sum_bound(l, abs[t][k]) + own * abs[t][k],
                        &format!("{name} against the exact sum, {what}"),
                    );
                    worst[t] = worst[t].max(w.ratio());
                }
            }
        }
    }
    println!(
        "sums: worst of the bound Σx {:.3}, centred squares {:.3}, Σ dy {:.3}, Σ dy·x̂ {:.3}",
        worst[0], worst[1], worst[2], worst[3]
    );
}

/// Values of a tensor shaped `dims` for an exact twin of the centred squares: each one of
/// -2, -1, 1 and 2, and every channel summing to exactly 0 (the second half of a channel
/// negates the first, and an odd channel ends 1, 1, -2), so the mean is exactly 0.
fn zero_mean(dims: ChannelDims, seed: u64) -> Vec<f32> {
    let (c, p, l) = (
        dims.channels as usize,
        dims.inner as usize,
        dims.summed() as usize,
    );
    let mut x = vec![0.0f32; dims.len() as usize];
    for k in 0..c {
        let half = if l.is_multiple_of(2) {
            l / 2
        } else {
            (l - 3) / 2
        };
        let v = exact_values(seed + k as u64, half);
        let mut values: Vec<f32> = v.iter().copied().chain(v.iter().map(|x| -x)).collect();
        if !l.is_multiple_of(2) {
            values.extend([1.0, 1.0, -2.0]);
        }
        for (j, value) in values.into_iter().enumerate() {
            x[(j / p * c + k) * p + j % p] = value;
        }
    }
    x
}

/// The four sums on small integers, bit for bit, as the CPU's in-order `f32` sums, on every
/// shape and on the channel sums' own shapes in `cavitation-gpu`'s tests (one chunk exactly,
/// one value more, several chunks, a small matrix). `Σx` and the centred squares come from
/// inputs whose channels sum to 0. `Σ dy` and `Σ dy·x̂` come with statistics uploaded by hand
/// (mean 0.5, inv 2, so every `x̂` is an odd integer). Then the input gradient through running
/// statistics, `dy · γ · inv`, which is exact too.
#[test]
fn sums_exact_twin() {
    let Some(cuda) = Cuda::new() else { return };
    let mut all = shapes();
    all.extend([
        ChannelDims::new(3, 2, 5),
        ChannelDims::new(1, 1, 4096),
        ChannelDims::new(3, 4, 5000),
        ChannelDims::rows(3, 5),
        ChannelDims::rows(3, 300),
    ]);
    for (s, dims) in all.into_iter().enumerate() {
        let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
        let (c, s) = (dims.channels as usize, 100 * s as u64);
        let x = zero_mean(dims, s);
        let got = run(&cuda, &mut bn, Case::new(dims, &x));
        let mut want = (vec![0.0f32; c], vec![0.0f32; c]);
        for (i, &v) in x.iter().enumerate() {
            want.0[channel(dims, i)] += v;
            want.1[channel(dims, i)] += v * v;
        }
        assert_exact(&got.sums.0, &want.0, &format!("Σx {dims:?}"));
        assert_exact(&got.sums.1, &want.1, &format!("centred squares {dims:?}"));

        let stats: Vec<f32> = [0.5f32, 2.0, 0.0]
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, c))
            .collect();
        let x = exact_values(s + 1, x.len());
        let dy = exact_values(s + 2, x.len());
        let w = exact_values(s + 3, c);
        let bias = vec![0.0f32; c];
        let (rm, rv) = (vec![0.0f32; c], vec![1.0f32; c]);
        for running in [None, Some((&rm[..], &rv[..]))] {
            let case = Case {
                affine: Some((&w, &bias)),
                running,
                dy: Some(&dy),
                stats: Some(&stats),
                ..Case::new(dims, &x)
            };
            let got = run(&cuda, &mut bn, case);
            let mut want = (vec![0.0f32; c], vec![0.0f32; c]);
            for (i, (&g, &v)) in dy.iter().zip(&x).enumerate() {
                want.0[channel(dims, i)] += g;
                want.1[channel(dims, i)] += g * ((v - 0.5) * 2.0);
            }
            let what = format!("{dims:?}, running {}", running.is_some());
            assert_exact(&got.db, &want.0, &format!("Σ dy, {what}"));
            assert_exact(&got.dw, &want.1, &format!("Σ dy·x̂, {what}"));
            if running.is_some() {
                let want: Vec<f32> = dy
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| g * w[channel(dims, i)] * 2.0)
                    .collect();
                assert_exact(&got.dx, &want, &format!("dx through running stats, {what}"));
            }
        }
    }
}

/// What the `f32` tape gives for one layer: the output, the input gradient, `dγ`, `dβ` and the
/// running statistics after the step.
struct TapeRun {
    /// The output.
    y: Vec<f32>,
    /// The input gradient.
    dx: Vec<f32>,
    /// `dγ`.
    dw: Vec<f32>,
    /// `dβ`.
    db: Vec<f32>,
    /// The running mean and variance after the step's update.
    running: (Vec<f32>, Vec<f32>),
}

/// `Tape::batch_norm2d` on an `f32` store with `γ`, `β` and running statistics (momentum
/// 0.1), for a loss `Σ y · dy`: a training tape (batch statistics, then the running update)
/// or an evaluation one (the running statistics).
fn tape_f32(
    dims: ChannelDims,
    eps: f64,
    (x, dy): (&[f32], &[f32]),
    (w, bias): (&[f32], &[f32]),
    (rm, rv): (&[f32], &[f32]),
    training: bool,
) -> TapeRun {
    let (b, c, p) = (
        dims.outer as usize,
        dims.channels as usize,
        dims.inner as usize,
    );
    let mut store = ParamStore::new();
    let mut add =
        |v: &[f32], shape: &[usize]| store.add(Tensor::from_vec(v.to_vec(), shape).unwrap());
    let (xi, wi, bi) = (add(x, &[b, c * p]), add(w, &[c]), add(bias, &[c]));
    let stats = RunningStats {
        mean: add(rm, &[c]),
        var: add(rv, &[c]),
        count: None,
        momentum: Some(0.1),
    };
    let mut graph = Graph::new();
    let mut tape = if training {
        graph.training_tape(&store.values, 0, 1)
    } else {
        graph.tape(&store.values)
    };
    let (xv, wv, bv) = (tape.param(xi), tape.param(wi), tape.param(bi));
    let y = tape.batch_norm2d(xv, Some(wv), Some(bv), Some(stats), c, p, eps);
    let r = tape.input(Tensor::from_vec(dy.to_vec(), &[b, c * p]).unwrap().view());
    let weighted = tape.mul(y, r);
    let loss = tape.sum(weighted);
    let y = tape.value(y).data().to_vec();
    tape.backward(loss, &mut store.grads);
    drop(tape);
    graph.apply_buffer_updates(&mut store.values);
    TapeRun {
        y,
        dx: store.grad(xi).data().to_vec(),
        dw: store.grad(wi).data().to_vec(),
        db: store.grad(bi).data().to_vec(),
        running: (
            store.value(stats.mean).data().to_vec(),
            store.value(stats.var).data().to_vec(),
        ),
    }
}

/// End to end against `Tape::batch_norm2d` on an `f32` store, bit for bit, where every sum
/// is exact: each channel of ±1 with mean 0 and eps 0, so the variance and `inv` are 1, and
/// small-integer output gradients. Through the batch statistics: the output, the input
/// gradient, `dγ`, `dβ` and the running statistics after the update. Through running ones
/// (seeded, eps [`EPS`]): the output, the input gradient and `dβ`.
#[test]
fn exact_inputs_give_the_tapes_bits() {
    let Some(cuda) = Cuda::new() else { return };
    let mut all = vec![ChannelDims::new(2, 7, 8), ChannelDims::rows(1000, 10)];
    all.extend(cifar());
    for (s, dims) in all.into_iter().enumerate() {
        let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
        let (c, s) = (dims.channels as usize, 100 * s as u64 + 3);
        let x: Vec<f32> = zero_mean(dims, s).iter().map(|v| v.signum()).collect();
        let dy = exact_values(s + 1, x.len());
        let (w, b) = (weights(s + 2, c), seeded_values(s + 4, c, -1.0, 1.0));
        let (rm, rv) = (
            seeded_values(s + 5, c, -1.0, 1.0),
            seeded_values(s + 6, c, 0.5, 2.0),
        );
        for training in [true, false] {
            let what = format!("{dims:?}, training {training}");
            let eps = if training { 0.0 } else { EPS };
            let want = tape_f32(dims, eps, (&x, &dy), (&w, &b), (&rm, &rv), training);
            let case = Case {
                eps,
                affine: Some((&w, &b)),
                running: (!training).then_some((&rm[..], &rv[..])),
                dy: Some(&dy),
                ..Case::new(dims, &x)
            };
            let got = run(&cuda, &mut bn, case);
            assert_exact(&got.y, &want.y, &format!("y, {what}"));
            assert_exact(&got.dx, &want.dx, &format!("dx, {what}"));
            assert_exact(&got.db, &want.db, &format!("dβ, {what}"));
            if !training {
                continue;
            }
            assert_exact(&got.dw, &want.dw, &format!("dγ, {what}"));
            let stats = upload(&cuda, &got.stats);
            let (mut mean, mut var) = (upload(&cuda, &rm), upload(&cuda, &rv));
            let finite = upload(&cuda, &[0u32]);
            let update = RunningUpdate::new(Some(0.1), None);
            bn.update_running(&update, &stats, (&mut mean, &mut var), &finite)
                .expect("running update");
            assert_exact(
                &read(&mean, c),
                &want.running.0,
                &format!("running mean, {what}"),
            );
            assert_exact(
                &read(&var, c),
                &want.running.1,
                &format!("running variance, {what}"),
            );
        }
    }
}

/// NaN in the input or the output gradient goes where the CPU's goes: the `f32` tape on the
/// exact inputs of [`exact_inputs_give_the_tapes_bits`] with a NaN in one channel's input and
/// an infinity in another's, then a NaN in a third channel's output gradient. Every output
/// and gradient is NaN exactly where the tape's is, and elsewhere has its bits.
#[test]
fn nan_goes_where_the_cpus_goes() {
    let Some(cuda) = Cuda::new() else { return };
    let dims = ChannelDims::new(4, 5, 6);
    let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
    let c = 5;
    let mut x: Vec<f32> = zero_mean(dims, 1).iter().map(|v| v.signum()).collect();
    let mut dy = exact_values(2, x.len());
    let (w, b) = (weights(3, c), seeded_values(5, c, -1.0, 1.0));
    let (rm, rv) = (
        seeded_values(6, c, -1.0, 1.0),
        seeded_values(7, c, 0.5, 2.0),
    );
    x[6] = f32::NAN; // channel 1
    x[14] = f32::INFINITY; // channel 2
    dy[20] = f32::NAN; // channel 3
    for training in [true, false] {
        let what = format!("training {training}");
        let eps = if training { 0.0 } else { EPS };
        let want = tape_f32(dims, eps, (&x, &dy), (&w, &b), (&rm, &rv), training);
        let case = Case {
            eps,
            affine: Some((&w, &b)),
            running: (!training).then_some((&rm[..], &rv[..])),
            dy: Some(&dy),
            ..Case::new(dims, &x)
        };
        let got = run(&cuda, &mut bn, case);
        assert!(want.y.iter().any(|v| v.is_nan()) && want.dx.iter().any(|v| v.is_nan()));
        assert_exact(&got.y, &want.y, &format!("y, {what}"));
        assert_exact(&got.dx, &want.dx, &format!("dx, {what}"));
        assert_exact(&got.db, &want.db, &format!("dβ, {what}"));
        if training {
            assert_exact(&got.dw, &want.dw, &format!("dγ, {what}"));
        }
    }
}

/// The bound of a sum of `l` terms of magnitudes `abs`, with the WGSL kernels' allowance for
/// flushed values.
fn sum_error(l: usize, abs: f64) -> f64 {
    sum_bound(l, abs) + (2 * l) as f64 * FLOOR
}

/// `b` with an allowance for rounding the result it bounds itself.
fn second(b: f64) -> f64 {
    b * (1.0 + 16.0 * U)
}

/// The exact statistics of one channel and the WGSL kernels' bounds on the card's.
#[derive(Clone, Copy, Debug)]
struct Exact {
    /// The mean.
    mean: f64,
    /// The bound on the card's mean.
    e_mean: f64,
    /// `1 / √(var + eps)`.
    inv: f64,
    /// The bound on the card's `inv`.
    e_inv: f64,
}

/// `inv = 1/√a` for `a` computed within `da` of its exact value, with 2 ulp on top:
/// `(inv, bound)`.
fn inv_bound(a: f64, da: f64) -> (f64, f64) {
    let r = da / a;
    assert!(
        r < 0.5,
        "the variance is too uncertain to bound inv: {da:e} of {a:e}"
    );
    let inv = 1.0 / a.sqrt();
    (inv, inv * (r + 4.0 * U * (1.0 + r)) + FLOOR)
}

/// The exact batch statistics of each channel of `x` and the WGSL kernels' composed bounds
/// on the card's: the sum, the division, the centred squares around the card's mean, and
/// `inv` from them.
fn batch_exact(dims: ChannelDims, x: &[f32]) -> Vec<Exact> {
    let (c, l) = (dims.channels as usize, dims.summed() as usize);
    let m = l as f64;
    let eps32 = f64::from(EPS as f32);
    let mut per = vec![Vec::with_capacity(l); c];
    for (i, &v) in x.iter().enumerate() {
        per[channel(dims, i)].push(f64::from(v));
    }
    per.iter()
        .map(|v| {
            let abs: f64 = v.iter().map(|x| x.abs()).sum();
            let mean = v.iter().sum::<f64>() / m;
            let e_sum = sum_error(l, abs);
            let e_mean = e_sum / m + 5.0 * U * (mean.abs() + e_sum / m) + FLOOR;
            let sq: f64 = v.iter().map(|x| (x - mean).powi(2)).sum();
            let dev: f64 = v.iter().map(|x| (x - mean).abs()).sum();
            let terms = sq + 2.0 * e_mean * dev + m * e_mean * e_mean;
            let d_sq = m * e_mean * e_mean + sum_error(l, terms) + 3.0 * U * terms;
            let var = (sq + d_sq) / m;
            let da = d_sq / m
                + 5.0 * U * var
                + U * (var * (1.0 + 5.0 * U) + eps32)
                + (eps32 - EPS).abs()
                + FLOOR;
            let (inv, e_inv) = inv_bound(sq / m + EPS, da);
            Exact {
                mean,
                e_mean,
                inv,
                e_inv,
            }
        })
        .collect()
}

/// The statistics of running ones and the bounds on the card's: the mean as it is, `inv`
/// from `var + eps` rounded once.
fn running_exact(mean: &[f32], var: &[f32]) -> Vec<Exact> {
    let eps32 = f64::from(EPS as f32);
    mean.iter()
        .zip(var)
        .map(|(&rm, &rv)| {
            let rv = f64::from(rv);
            let da = U * (rv + eps32) + (eps32 - EPS).abs() + FLOOR;
            let (inv, e_inv) = inv_bound(rv + EPS, da);
            Exact {
                mean: f64::from(rm),
                e_mean: 0.0,
                inv,
                e_inv,
            }
        })
        .collect()
}

/// What `Tape::batch_norm2d` gives on an `f64` store from the same `f32` values, on an
/// evaluation tape: `y` from the batch statistics, or from `running` ones when given.
fn tape_forward(
    dims: ChannelDims,
    x: &[f32],
    affine: Option<(&[f32], &[f32])>,
    running: Option<(&[f32], &[f32])>,
) -> Vec<f64> {
    let (b, c, p) = (
        dims.outer as usize,
        dims.channels as usize,
        dims.inner as usize,
    );
    let mut store = ParamStore::<f64>::new_generic();
    let mut param = |v: &[f32]| store.add(Tensor::from_vec(wide(v), &[v.len()]).unwrap());
    let affine = affine.map(|(w, bias)| (param(w), param(bias)));
    let running = running.map(|(mean, var)| RunningStats {
        mean: param(mean),
        var: param(var),
        count: None,
        momentum: Some(0.1),
    });
    let mut graph = Graph::<f64>::new_generic();
    let mut tape = graph.tape(&store.values);
    let xv = tape.input(Tensor::from_vec(wide(x), &[b, c * p]).unwrap().view());
    let (w, bias) = match affine {
        Some((w, bias)) => (Some(tape.param(w)), Some(tape.param(bias))),
        None => (None, None),
    };
    let y = tape.batch_norm2d(xv, w, bias, running, c, p, EPS);
    tape.value(y).data().to_vec()
}

/// The forward pass end to end against `Tape::batch_norm2d` on an `f64` store, from the
/// batch statistics and from running ones, with and without `γ`, `β`, within the WGSL
/// kernels' bound composed through the statistics ([`batch_exact`]), which holds for these
/// kernels too: their division and square root round once where the WGSL ones are allowed
/// 2.5 and 2 ulp. A constant channel is included, whose variance is 0.
#[test]
fn forward_matches_the_tape() {
    let Some(cuda) = Cuda::new() else { return };
    let mut all = shapes();
    all.push(ChannelDims::new(4, 2, 3));
    let mut worst = 0.0f64;
    for (s, dims) in all.into_iter().enumerate() {
        let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
        let (c, s) = (dims.channels as usize, 100 * s as u64 + 7);
        let mut x = seeded_values(s, dims.len() as usize, -2.0, 2.0);
        if dims == ChannelDims::new(4, 2, 3) {
            for (i, v) in x.iter_mut().enumerate() {
                if channel(dims, i) == 1 {
                    *v = 5.0;
                }
            }
        }
        let (w, b) = (weights(s + 1, c), seeded_values(s + 3, c, -1.0, 1.0));
        let (rm, rv) = (
            seeded_values(s + 4, c, -1.0, 1.0),
            seeded_values(s + 5, c, 0.5, 2.0),
        );
        for affine in [Some((&w[..], &b[..])), None] {
            for running in [None, Some((&rm[..], &rv[..]))] {
                let what = format!(
                    "{dims:?}, affine {}, running {}",
                    affine.is_some(),
                    running.is_some()
                );
                let case = Case {
                    affine,
                    running,
                    ..Case::new(dims, &x)
                };
                let got = run(&cuda, &mut bn, case);
                let want = tape_forward(dims, &x, affine, running);
                let exact = match running {
                    Some((m, v)) => running_exact(m, v),
                    None => batch_exact(dims, &x),
                };
                let bound = |i: usize| {
                    let k = channel(dims, i);
                    let e = exact[k];
                    let (g, beta) = affine.map_or((1.0, 0.0), |(w, b)| {
                        (f64::from(w[k]).abs(), f64::from(b[k]).abs())
                    });
                    let d = (f64::from(x[i]) - e.mean).abs();
                    let t = d * e.inv;
                    let dt = d * e.e_inv + e.e_mean * (e.inv + e.e_inv);
                    let rounding = if affine.is_some() {
                        U * (want[i].abs() + g * dt + 3.0 * g * (t + dt))
                    } else {
                        2.0 * U * (t + dt)
                    };
                    g * dt + second(rounding) + 64.0 * f64::EPSILON * (g * t + beta)
                };
                let w = assert_within(&got.y, &want, bound, &format!("y, {what}"));
                worst = worst.max(w.ratio());
            }
        }
    }
    println!("y against the f64 tape: worst {worst:.3} of its bound");
}

/// The gradients end to end against the tape's backward pass on an `f64` store, for a loss
/// `Σ y · r` (so `dy = r`), through the batch statistics (a training tape) and through
/// running ones (an evaluation tape): `dβ` within the sums' bound, `dγ` and `dx` within
/// [`TOLERANCE`] of each one's scale. This checks the formula, not the rounding, which the
/// kernel twins pin down.
#[test]
fn backward_matches_the_tape() {
    let Some(cuda) = Cuda::new() else { return };
    let mut worst = [0.0f64; 3];
    for (s, dims) in shapes().into_iter().enumerate() {
        let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
        let (b, c, p) = (
            dims.outer as usize,
            dims.channels as usize,
            dims.inner as usize,
        );
        let (l, m, s) = (
            dims.summed() as usize,
            dims.summed() as f64,
            100 * s as u64 + 13,
        );
        let x = seeded_values(s, dims.len() as usize, -2.0, 2.0);
        let dy = seeded_values(s + 1, x.len(), -1.0, 1.0);
        let (w, bias) = (weights(s + 2, c), seeded_values(s + 4, c, -1.0, 1.0));
        let (rm, rv) = (
            seeded_values(s + 5, c, -1.0, 1.0),
            seeded_values(s + 6, c, 0.5, 2.0),
        );
        for running in [None, Some((&rm[..], &rv[..]))] {
            let what = format!("{dims:?}, running {}", running.is_some());
            let case = Case {
                affine: Some((&w, &bias)),
                running,
                dy: Some(&dy),
                ..Case::new(dims, &x)
            };
            let got = run(&cuda, &mut bn, case);
            let mut store = ParamStore::<f64>::new_generic();
            let xi = store.add(Tensor::from_vec(wide(&x), &[b, c * p]).unwrap());
            let wi = store.add(Tensor::from_vec(wide(&w), &[c]).unwrap());
            let bi = store.add(Tensor::from_vec(wide(&bias), &[c]).unwrap());
            let stats = running.map(|(mean, var)| RunningStats {
                mean: store.add(Tensor::from_vec(wide(mean), &[c]).unwrap()),
                var: store.add(Tensor::from_vec(wide(var), &[c]).unwrap()),
                count: None,
                momentum: Some(0.1),
            });
            let mut graph = Graph::<f64>::new_generic();
            let mut tape = match running {
                Some(_) => graph.tape(&store.values),
                None => graph.training_tape(&store.values, 0, 1),
            };
            let (xv, wv, bv) = (tape.param(xi), tape.param(wi), tape.param(bi));
            let y = tape.batch_norm2d(xv, Some(wv), Some(bv), stats, c, p, EPS);
            let r = tape.input(Tensor::from_vec(wide(&dy), &[b, c * p]).unwrap().view());
            let weighted = tape.mul(y, r);
            let loss = tape.sum(weighted);
            tape.backward(loss, &mut store.grads);
            drop(tape);
            let exact = match running {
                Some((mean, var)) => running_exact(mean, var),
                None => batch_exact(dims, &x),
            };
            let xhat = |i: usize| {
                let e = exact[channel(dims, i)];
                (f64::from(x[i]) - e.mean) * e.inv
            };
            let (mut sdy, mut ady, mut sdx, mut asdx) =
                (vec![0.0; c], vec![0.0; c], vec![0.0; c], vec![0.0; c]);
            for (i, &g) in dy.iter().enumerate() {
                let (k, g) = (channel(dims, i), f64::from(g));
                sdy[k] += g;
                ady[k] += g.abs();
                sdx[k] += g * xhat(i);
                asdx[k] += (g * xhat(i)).abs();
            }
            let r = assert_within(
                &got.db,
                store.grad(bi).data(),
                |k| sum_error(l, ady[k]),
                &format!("dβ against the tape, {what}"),
            );
            worst[0] = worst[0].max(r.ratio());
            let r = assert_within(
                &got.dw,
                store.grad(wi).data(),
                |k| TOLERANCE * asdx[k],
                &format!("dγ against the tape, {what}"),
            );
            worst[1] = worst[1].max(r.ratio());
            let scale = |i: usize| {
                let k = channel(dims, i);
                let (g, iv) = (f64::from(w[k]).abs(), exact[k].inv);
                let gy = f64::from(dy[i]).abs() * g;
                match running {
                    Some(_) => gy * iv,
                    None => iv / m * (m * gy + sdy[k].abs() * g + (xhat(i) * sdx[k]).abs() * g),
                }
            };
            let r = assert_within(
                &got.dx,
                store.grad(xi).data(),
                |i| TOLERANCE * scale(i),
                &format!("dx against the tape, {what}"),
            );
            worst[2] = worst[2].max(r.ratio());
        }
    }
    println!(
        "against the f64 tape: worst dβ {:.3} of its bound, dγ {:.2e} and dx {:.2e} of the \
         tolerance",
        worst[0], worst[1], worst[2]
    );
}

/// The running statistics over three training steps, for each way of weighting a batch (a
/// momentum of 0.1 with a count kept, the cumulative average from a count of 2, neither):
/// after each step the running mean and variance are the CPU's `f32`
/// `momentum · b + (1 − momentum) · r` of the card's batch statistics, bit for bit, and the
/// count is the one `Graph::apply_buffer_updates` keeps.
#[test]
fn running_statistics_match_the_cpu() {
    let Some(cuda) = Cuda::new() else { return };
    let dims = ChannelDims::new(4, 3, 5);
    let (b, c, p) = (4, 3, 5);
    let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
    for (momentum, count) in [(Some(0.1), Some(0.0f32)), (None, Some(2.0)), (None, None)] {
        let what = format!("momentum {momentum:?}, count {count:?}");
        let (rm, rv) = (
            seeded_values(1, c, -1.0, 1.0),
            seeded_values(2, c, 0.5, 2.0),
        );
        let mut store = ParamStore::<f64>::new_generic();
        let ids = RunningStats {
            mean: store.add(Tensor::from_vec(wide(&rm), &[c]).unwrap()),
            var: store.add(Tensor::from_vec(wide(&rv), &[c]).unwrap()),
            count: count.map(|n| store.add(Tensor::from_vec(vec![f64::from(n)], &[1]).unwrap())),
            momentum,
        };
        let mut graph = Graph::<f64>::new_generic();
        let (mut mean, mut var) = (upload(&cuda, &rm), upload(&cuda, &rv));
        let (mut r_mean, mut r_var) = (rm.clone(), rv.clone());
        let finite = upload(&cuda, &[0u32]);
        let mut update = RunningUpdate::new(momentum, count);
        for k in 0..3u64 {
            let x = seeded_values(10 + k, dims.len() as usize, -2.0, 2.0);
            let mut tape = graph.training_tape(&store.values, 0, k);
            let xv = tape.input(Tensor::from_vec(wide(&x), &[b, c * p]).unwrap().view());
            tape.batch_norm2d(xv, None, None, Some(ids), c, p, EPS);
            drop(tape);
            graph.apply_buffer_updates(&mut store.values);

            let xb = upload(&cuda, &x);
            let (mut y, mut stats) = (nans(&cuda, x.len()), nans(&cuda, 3 * c));
            bn.forward(EPS, BatchStats::Batch, &xb, None, &mut y, &mut stats)
                .expect("forward");
            bn.update_running(&update, &stats, (&mut mean, &mut var), &finite)
                .expect("running update");
            let (m, keep) = update.rates();
            update.commit(true);
            assert_eq!(
                update.count().map(f64::from),
                ids.count.map(|id| store.value(id).data()[0]),
                "the count after step {k}, {what}"
            );
            let batch = read(&stats, 3 * c);
            for j in 0..c {
                r_mean[j] = m * batch[j] + keep * r_mean[j];
                r_var[j] = m * batch[2 * c + j] + keep * r_var[j];
            }
            assert_exact(
                &read(&mean, c),
                &r_mean,
                &format!("running mean, step {k}, {what}"),
            );
            assert_exact(
                &read(&var, c),
                &r_var,
                &format!("running variance, step {k}, {what}"),
            );
        }
    }
}

/// A set finite flag leaves the running statistics as they were, bit for bit, and a step
/// committed as not run leaves the count and the next rates as they were; the next clear
/// step updates them as usual.
#[test]
fn a_set_finite_flag_skips_the_update() {
    let Some(cuda) = Cuda::new() else { return };
    let dims = ChannelDims::new(2, 3, 4);
    let c = 3;
    let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
    let x = seeded_values(1, dims.len() as usize, -2.0, 2.0);
    let (rm, rv) = (
        seeded_values(2, c, -1.0, 1.0),
        seeded_values(3, c, 0.5, 2.0),
    );
    let xb = upload(&cuda, &x);
    let (mut y, mut stats) = (nans(&cuda, x.len()), nans(&cuda, 3 * c));
    let (mut mean, mut var) = (upload(&cuda, &rm), upload(&cuda, &rv));
    let mut finite = upload(&cuda, &[1u32]);
    bn.forward(EPS, BatchStats::Batch, &xb, None, &mut y, &mut stats)
        .expect("forward");
    let mut update = RunningUpdate::new(None, Some(3.0));
    let rates = update.rates();
    bn.update_running(&update, &stats, (&mut mean, &mut var), &finite)
        .expect("running update");
    update.commit(false);
    assert_exact(&read(&mean, c), &rm, "the running mean under a set flag");
    assert_exact(&read(&var, c), &rv, "the running variance under a set flag");
    assert_eq!(update.count(), Some(3.0), "the count of a skipped step");
    assert_eq!(update.rates(), rates, "the rates after a skipped step");
    assert_eq!(rates, (0.25, 0.75), "one over the count after this batch");

    finite.write(&[0u32]).expect("clear the flag");
    bn.update_running(&update, &stats, (&mut mean, &mut var), &finite)
        .expect("running update");
    update.commit(true);
    assert_eq!(update.count(), Some(4.0));
    let batch = read(&stats, 3 * c);
    for (what, buffer, old, at) in [("mean", &mean, &rm, 0), ("variance", &var, &rv, 2 * c)] {
        let want: Vec<f32> = old
            .iter()
            .zip(&batch[at..at + c])
            .map(|(&r, &b)| 0.25 * b + 0.75 * r)
            .collect();
        assert_exact(
            &read(buffer, c),
            &want,
            &format!("the running {what} once the flag is clear"),
        );
    }
}

/// Every kernel goes by the shape, not by its buffers' lengths: with every tensor buffer
/// longer than its tensor, the values past it stay as they were, and the results are the
/// bits of exactly sized buffers.
#[test]
fn buffers_longer_than_the_tensor() {
    let Some(cuda) = Cuda::new() else { return };
    let dims = ChannelDims::new(3, 5, 7);
    let (n, c) = (dims.len() as usize, 5);
    let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
    let x = seeded_values(1, n, -2.0, 2.0);
    let dy = seeded_values(2, n, -1.0, 1.0);
    let (w, b) = (weights(3, c), seeded_values(5, c, -1.0, 1.0));
    let case = Case {
        affine: Some((&w, &b)),
        dy: Some(&dy),
        ..Case::new(dims, &x)
    };
    let exact = run(&cuda, &mut bn, case);
    let long = run(&cuda, &mut bn, Case { pad: 300, ..case });
    for (what, got, want, len) in [
        ("y", &long.y, &exact.y, n),
        ("dx", &long.dx, &exact.dx, n),
        ("stats", &long.stats, &exact.stats, 3 * c),
    ] {
        assert_exact(&got[..len], want, &format!("{what} with long buffers"));
        assert!(
            got[len..].iter().all(|&v| v == SENTINEL),
            "{what}: a value past the tensor changed"
        );
    }
    assert_exact(&long.dw, &exact.dw, "dγ with long buffers");
    assert_exact(&long.db, &exact.db, "dβ with long buffers");
}

/// Two runs of every pass on the same inputs give the same bits, on a `cifar-cnn` shape.
#[test]
fn two_runs_give_the_same_bits() {
    let Some(cuda) = Cuda::new() else { return };
    let dims = ChannelDims::new(128, 64, 256);
    let c = 64;
    let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
    let x = seeded_values(1, dims.len() as usize, -2.0, 2.0);
    let dy = seeded_values(2, x.len(), -1.0, 1.0);
    let (w, b) = (weights(3, c), seeded_values(5, c, -1.0, 1.0));
    let case = Case {
        affine: Some((&w, &b)),
        dy: Some(&dy),
        ..Case::new(dims, &x)
    };
    let (first, second) = (run(&cuda, &mut bn, case), run(&cuda, &mut bn, case));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    for (what, a, b) in [
        ("y", &first.y, &second.y),
        ("stats", &first.stats, &second.stats),
        ("dx", &first.dx, &second.dx),
        ("dγ", &first.dw, &second.dw),
        ("dβ", &first.db, &second.db),
    ] {
        assert_eq!(bits(a), bits(b), "{what} on a second run");
    }
}

/// The refusals: a shape with a dimension of 0, more than 2³¹ values or more than 65,535
/// channels, an eps that is negative or not finite, and batch statistics over one value per
/// channel are `CudaError::Invalid`; a buffer too short panics before any launch, naming it.
/// A subnormal eps, which the WGSL kernels refuse because they flush it, is taken, and gives
/// the CPU's statistics.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let invalid = |r: Result<BatchNorm2d<'_>, CudaError>| matches!(r, Err(CudaError::Invalid(_)));
    assert!(invalid(BatchNorm2d::new(&cuda, ChannelDims::new(0, 2, 4))));
    assert!(invalid(BatchNorm2d::new(
        &cuda,
        ChannelDims::new(1 << 16, 1 << 16, 1)
    )));
    assert!(invalid(BatchNorm2d::new(
        &cuda,
        ChannelDims::rows(2, 70_000)
    )));

    let dims = ChannelDims::new(2, 2, 4);
    let mut bn = BatchNorm2d::new(&cuda, dims).expect("batch norm");
    let x = upload(&cuda, &seeded_values(1, 16, -2.0, 2.0));
    let (mut y, mut stats) = (nans(&cuda, 16), nans(&cuda, 6));
    for eps in [-1.0, f64::NAN, f64::INFINITY] {
        let r = bn.forward(eps, BatchStats::Batch, &x, None, &mut y, &mut stats);
        assert!(
            matches!(r, Err(CudaError::Invalid(ref t)) if t.contains("eps")),
            "{eps}"
        );
    }
    let mut one = BatchNorm2d::new(&cuda, ChannelDims::new(1, 2, 1)).expect("batch norm");
    let r = one.forward(EPS, BatchStats::Batch, &x, None, &mut y, &mut stats);
    assert!(matches!(r, Err(CudaError::Invalid(ref t)) if t.contains("more than one value")));
    let (mean, var) = (
        upload(&cuda, &[0.5f32, -1.0]),
        upload(&cuda, &[1.0f32, 2.0]),
    );
    let running = BatchStats::Running {
        mean: &mean,
        var: &var,
    };
    one.forward(EPS, running, &x, None, &mut y, &mut stats)
        .expect("running statistics over one value per channel");

    let eps = 1e-40;
    assert!((eps as f32).is_subnormal());
    bn.forward(eps, BatchStats::Batch, &x, None, &mut y, &mut stats)
        .expect("a subnormal eps");
    let (sx, sq) = (read(bn.sums().0, 2), read(bn.sums().1, 2));
    assert_exact(
        &read(&stats, 6),
        &cpu_stats(dims, eps, (&sx, &sq), None),
        "statistics with a subnormal eps",
    );

    let short = nans(&cuda, 15);
    let mut short_mut = nans(&cuda, 5);
    let text = panic_message(|| {
        let _ = bn.forward(EPS, BatchStats::Batch, &short, None, &mut y, &mut stats);
    });
    assert!(
        text.contains("the input buffer holds 60 bytes, and 16 values need 64"),
        "{text}"
    );
    let text = panic_message(|| {
        let _ = bn.forward(EPS, BatchStats::Batch, &x, None, &mut y, &mut short_mut);
    });
    assert!(
        text.contains("the statistics buffer holds 20 bytes"),
        "{text}"
    );
    let one_value = nans(&cuda, 1);
    let text = panic_message(|| {
        let _ = bn.forward(
            EPS,
            BatchStats::Batch,
            &x,
            Some((&one_value, &mean)),
            &mut y,
            &mut stats,
        );
    });
    assert!(text.contains("the weight buffer holds 4 bytes"), "{text}");
    let short_running = BatchStats::Running {
        mean: &one_value,
        var: &var,
    };
    let text = panic_message(|| {
        let _ = bn.forward(EPS, short_running, &x, None, &mut y, &mut stats);
    });
    assert!(
        text.contains("the running mean buffer holds 4 bytes"),
        "{text}"
    );
    let mut dx = nans(&cuda, 15);
    let text = panic_message(|| {
        let _ = bn.backward(BatchStats::Batch, &x, &stats, &x, &mut dx, None);
    });
    assert!(
        text.contains("the input gradient buffer holds 60 bytes"),
        "{text}"
    );
    let (w, mut dw, mut db) = (upload(&cuda, &[1.0f32; 2]), nans(&cuda, 2), nans(&cuda, 1));
    let mut dx = nans(&cuda, 16);
    let text = panic_message(|| {
        let _ = bn.backward(
            BatchStats::Batch,
            &x,
            &stats,
            &x,
            &mut dx,
            Some((&w, &mut dw, &mut db)),
        );
    });
    assert!(
        text.contains("the bias gradient buffer holds 4 bytes"),
        "{text}"
    );
    let (mut rm, mut rv) = (nans(&cuda, 2), nans(&cuda, 2));
    let empty = cuda.buffer(0).expect("allocate");
    let text = panic_message(|| {
        let _ = bn.update_running(
            &RunningUpdate::new(Some(0.1), None),
            &stats,
            (&mut rm, &mut rv),
            &empty,
        );
    });
    assert!(
        text.contains("the finite flag buffer holds 0 bytes"),
        "{text}"
    );
}

/// The rates are the CPU's own `f32` momentum and `1 − momentum`, bit for bit, for each way
/// of weighting a batch, over four batches: on an `f32` store, a constant batch (mean exactly
/// 1, unbiased variance exactly 0) moves a running mean of 0 to exactly the momentum and a
/// running variance of 1 to exactly `1 − momentum`. Host arithmetic only, so it runs without
/// CUDA.
#[test]
fn rates_are_the_cpus_bits() {
    for (momentum, count) in [
        (Some(0.1), Some(0.0f32)),
        (Some(0.3), None),
        (None, Some(2.0)),
        (None, None),
    ] {
        let mut update = RunningUpdate::new(momentum, count);
        for k in 0..4 {
            let mut store = ParamStore::new();
            let ids = RunningStats {
                mean: store.add(Tensor::zeros(&[1])),
                var: store.add(Tensor::full(&[1], 1.0f32)),
                count: update.count().map(|n| store.add(Tensor::full(&[1], n))),
                momentum,
            };
            let mut graph = Graph::new();
            let mut tape = graph.training_tape(&store.values, 0, 1);
            let xv = tape.input(Tensor::full(&[2, 3], 1.0f32).view());
            tape.batch_norm2d(xv, None, None, Some(ids), 1, 3, EPS);
            drop(tape);
            graph.apply_buffer_updates(&mut store.values);
            let (m, keep) = update.rates();
            let what = format!("batch {k}, momentum {momentum:?}, count {count:?}");
            assert_exact(
                &[m],
                store.value(ids.mean).data(),
                &format!("momentum, {what}"),
            );
            assert_exact(
                &[keep],
                store.value(ids.var).data(),
                &format!("1 − momentum, {what}"),
            );
            update.commit(true);
            if let Some(id) = ids.count {
                assert_eq!(update.count(), Some(store.value(id).data()[0]), "{what}");
            }
        }
    }
}
