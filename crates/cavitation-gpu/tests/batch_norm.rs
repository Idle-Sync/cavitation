//! Batch norm on the GPU against its CPU twin, `Tape::batch_norm2d`, on every backend this
//! machine has:
//!
//! - each kernel from the GPU's own inputs, within its stated bound;
//! - the four sums bit for bit on integers;
//! - the forward pass, the gradients and the running statistics end to end against the tape
//!   on an `f64` store;
//! - the finite flag, buffers longer than the tensor, and the refusals.
//!
//! The shapes cover a batch of one, `inner = 1`, more than 256 channels, a channel of odd
//! length over two chunks, and the CIFAR network's `(128, 32, 1024)`. The tests skip without
//! a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, RunningStats, Tensor};
use cavitation_gpu::Gpu;
use cavitation_gpu::kernels::{BatchNorm2d, BatchStats, ChannelDims};
use common::{
    FLOOR, Kit, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values,
    sum_bound,
};

/// The eps every test uses, PyTorch's default.
const EPS: f64 = 1e-5;

/// What fills the values past a tensor in a buffer longer than it.
const SENTINEL: f32 = 7.0;

/// The tolerance of the gradients against the tape, relative to each one's scale: far above
/// the rounding of either side, far below a wrong term.
const TOLERANCE: f64 = 1.0 / 4096.0;

/// The shapes the kernels run, as `(batch, channels, h · w)`:
/// - a small batch;
/// - a batch of one whose channels hold an odd 4,097 values (two chunks);
/// - a 1000-row matrix of 10 channels (`inner = 1`);
/// - 300 channels (two workgroups of the per-channel kernels);
/// - the CIFAR network's first batch norm at batch 128.
fn shapes() -> Vec<ChannelDims> {
    vec![
        ChannelDims::new(2, 7, 9),
        ChannelDims::new(1, 3, 4097),
        ChannelDims::rows(1000, 10),
        ChannelDims::new(2, 300, 3),
        ChannelDims::new(128, 32, 1024),
    ]
}

/// The channel of value `i` of a tensor shaped `dims`.
fn channel(dims: ChannelDims, i: usize) -> usize {
    i / dims.inner as usize % dims.channels as usize
}

/// A kit whose steps can move a tensor of `dims`.
fn kit(gpu: &Gpu, dims: ChannelDims) -> Kit {
    Kit::new(gpu, 4 * dims.len() + 65_536)
}

/// `v` widened to `f64`.
fn wide(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `b` with an allowance for rounding the result it bounds itself: a bound computed from the
/// reference, applied to the GPU's value, which is within `b` of it.
fn second(b: f64) -> f64 {
    b * (1.0 + 16.0 * U)
}

/// One GPU run: a forward pass of `x` (or, with `stats` given, statistics uploaded by hand
/// and no forward pass), then with `dy` a backward pass.
#[derive(Clone, Copy, Default)]
struct Case<'a> {
    /// The tensor's shape.
    dims: Option<ChannelDims>,
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

/// What a [`Case`] read back.
#[derive(Default)]
struct Run {
    /// The output.
    y: Vec<f32>,
    /// The statistics `[mean | inv | unbiased]`.
    stats: Vec<f32>,
    /// `(Σx, Σ (x − mean)²)` of a batch forward pass.
    sums: Option<(Vec<f32>, Vec<f32>)>,
    /// The input gradient.
    dx: Vec<f32>,
    /// `dγ = Σ dy · x̂`, for an affine layer.
    dw: Vec<f32>,
    /// `dβ = Σ dy`, for an affine layer.
    db: Vec<f32>,
}

/// Runs `case` on the GPU.
fn run(kit: &mut Kit, bn: &BatchNorm2d, case: Case<'_>) -> Run {
    let dims = case.dims.expect("a shape");
    let (n, c) = (dims.len() as usize, dims.channels as usize);
    let padded = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .copied()
            .chain(std::iter::repeat_n(SENTINEL, case.pad))
            .collect()
    };
    let x = kit.upload(&padded(case.x));
    let y = kit.upload(&vec![SENTINEL; n + case.pad]);
    let stats = match case.stats {
        Some(s) => kit.upload(&padded(s)),
        None => kit.upload(&vec![SENTINEL; 3 * c + case.pad]),
    };
    let affine = case.affine.map(|(w, b)| (kit.upload(w), kit.upload(b)));
    let running = case.running.map(|(m, v)| (kit.upload(m), kit.upload(v)));
    let from = match &running {
        Some((mean, var)) => BatchStats::Running { mean, var },
        None => BatchStats::Batch,
    };
    let forward = bn.forward_call(
        &mut kit.steps,
        &mut kit.pool,
        dims,
        EPS,
        from,
        &x,
        affine.as_ref().map(|(w, b)| (w, b)),
        &y,
        &stats,
    );
    let grads = case.dy.map(|dy| {
        (
            kit.upload(&padded(dy)),
            kit.upload(&vec![SENTINEL; n + case.pad]),
            kit.upload(&vec![SENTINEL; c]),
            kit.upload(&vec![SENTINEL; c]),
        )
    });
    let backward = grads.as_ref().map(|(dy, dx, dw, db)| {
        bn.backward_call(
            &mut kit.steps,
            &mut kit.pool,
            &forward,
            &x,
            &stats,
            dy,
            dx,
            affine.as_ref().map(|(w, _)| (w, dw, db)),
        )
    });
    let mut step = kit.steps.begin();
    if case.stats.is_none() {
        bn.record_forward(&mut step, &forward);
    }
    if let Some(b) = &backward {
        bn.record_backward(&mut step, b);
    }
    let _ = step.finish();
    let mut out = Run {
        y: kit.read(&y),
        stats: kit.read(&stats),
        sums: forward.sums().map(|(s, q)| (kit.read(s), kit.read(q))),
        ..Run::default()
    };
    if let Some((_, dx, dw, db)) = &grads {
        out.dx = kit.read(dx);
        if affine.is_some() {
            out.dw = kit.read(dw);
            out.db = kit.read(db);
        }
    }
    forward.release(&mut kit.pool);
    if let Some(b) = backward {
        b.release(&mut kit.pool);
    }
    out
}

/// The values of each channel of `v` (shaped `dims`), in the order `b · inner + p`.
fn per_channel(dims: ChannelDims, v: &[f32]) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(dims.summed() as usize); dims.channels as usize];
    for (i, &x) in v[..dims.len() as usize].iter().enumerate() {
        out[channel(dims, i)].push(f64::from(x));
    }
    out
}

/// The bound of a sum of `l` terms of magnitudes `abs`, flushes included.
fn sum_error(l: usize, abs: f64) -> f64 {
    sum_bound(l, abs) + (2 * l) as f64 * FLOOR
}

/// The exact statistics of one channel and the bounds on the GPU's.
#[derive(Clone, Copy, Debug)]
struct Exact {
    /// The mean.
    mean: f64,
    /// The bound on the GPU's mean.
    e_mean: f64,
    /// `1 / √(var + eps)`.
    inv: f64,
    /// The bound on the GPU's `inv`.
    e_inv: f64,
    /// The unbiased variance.
    unbiased: f64,
    /// The bound on the GPU's unbiased variance.
    e_unbiased: f64,
}

/// `inv = 1/√a` for `a` the GPU computes within `da` of its exact value, with
/// `inverseSqrt`'s 2 ulp on top: `(inv, bound)`.
fn inv_bound(a: f64, da: f64) -> (f64, f64) {
    let r = da / a;
    assert!(
        r < 0.5,
        "the variance is too uncertain to bound inv: {da:e} of {a:e}"
    );
    let inv = 1.0 / a.sqrt();
    (inv, inv * (r + 4.0 * U * (1.0 + r)) + FLOOR)
}

/// The exact batch statistics of each channel of `x` (shaped `dims`) and the bounds on the
/// GPU's, composed from each kernel's bound: the sum, the division, the centred squares
/// around the GPU's mean (`Σ (x − mean')² = Σ (x − mean)² + m·(mean' − mean)²`), and `inv`
/// and the unbiased variance from them.
fn batch_exact(dims: ChannelDims, x: &[f32]) -> Vec<Exact> {
    let l = dims.summed() as usize;
    let m = l as f64;
    let eps32 = f64::from(EPS as f32);
    per_channel(dims, x)
        .iter()
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
            let unbiased = sq / (m - 1.0);
            let e_unbiased = d_sq / (m - 1.0) + 5.0 * U * (unbiased + d_sq / (m - 1.0)) + FLOOR;
            Exact {
                mean,
                e_mean,
                inv,
                e_inv,
                unbiased,
                e_unbiased,
            }
        })
        .collect()
}

/// The statistics of running ones and the bounds on the GPU's: the mean as it is, `inv`
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
                unbiased: 0.0,
                e_unbiased: 0.0,
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

/// `n` values of `γ` for a rounding twin: magnitudes in `[0.5, 2)`, either sign.
fn weights(seed: u64, n: usize) -> Vec<f32> {
    let signs = exact_values(seed, n);
    seeded_values(seed + 1, n, 0.5, 2.0)
        .into_iter()
        .zip(signs)
        .map(|(w, s)| w.copysign(s))
        .collect()
}

/// Each forward kernel against the exact result of its own inputs as the GPU computed them,
/// within its stated bound: `Σx`, the mean, the centred squares around the GPU's mean,
/// `inv`, the unbiased variance and the normalisation, from the batch with and without `γ`,
/// `β`; and from running statistics, whose mean is copied and whose unbiased slot is 0. The
/// tree being fixed, a second run gives the same bits.
#[test]
fn forward_kernels_twin() {
    for_each_gpu(|gpu| {
        for (s, dims) in shapes().into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
            let (c, s) = (dims.channels as usize, 100 * s as u64);
            let (l, m) = (dims.summed() as usize, dims.summed() as f64);
            let x = seeded_values(s, dims.len() as usize, -2.0, 2.0);
            let (w, b) = (weights(s + 1, c), seeded_values(s + 3, c, -1.0, 1.0));
            let (rm, rv) = (
                seeded_values(s + 4, c, -1.0, 1.0),
                seeded_values(s + 5, c, 0.5, 2.0),
            );
            let xs = per_channel(dims, &x);
            for affine in [Some((&w[..], &b[..])), None] {
                for running in [None, Some((&rm[..], &rv[..]))] {
                    let what = format!(
                        "{dims:?}, affine {}, running {}",
                        affine.is_some(),
                        running.is_some()
                    );
                    let case = Case {
                        dims: Some(dims),
                        x: &x,
                        affine,
                        running,
                        ..Case::default()
                    };
                    let got = run(&mut kit, &bn, case);
                    let (mean, rest) = got.stats.split_at(c);
                    let (inv, unbiased) = rest.split_at(c);
                    if let Some((rm, rv)) = running {
                        assert_exact(mean, rm, &format!("running mean copied, {what}"));
                        assert_exact(unbiased, &vec![0.0; c], &format!("unbiased, {what}"));
                        let want: Vec<f64> = rv
                            .iter()
                            .map(|&v| 1.0 / (f64::from(v) + f64::from(EPS as f32)).sqrt())
                            .collect();
                        assert_within(inv, &want, |k| 8.0 * U * want[k], &format!("inv, {what}"));
                    } else {
                        let (sums, squares) = got.sums.as_ref().expect("batch sums");
                        let exact: Vec<f64> = xs.iter().map(|v| v.iter().sum()).collect();
                        let abs: Vec<f64> =
                            xs.iter().map(|v| v.iter().map(|x| x.abs()).sum()).collect();
                        assert_within(
                            sums,
                            &exact,
                            |k| sum_error(l, abs[k]),
                            &format!("Σx, {what}"),
                        );
                        let want: Vec<f64> = sums.iter().map(|&v| f64::from(v) / m).collect();
                        assert_within(
                            mean,
                            &want,
                            |k| 5.0 * U * want[k].abs(),
                            &format!("mean, {what}"),
                        );
                        let terms: Vec<f64> = xs
                            .iter()
                            .zip(mean)
                            .map(|(v, &mu)| v.iter().map(|x| (x - f64::from(mu)).powi(2)).sum())
                            .collect();
                        assert_within(
                            squares,
                            &terms,
                            |k| sum_error(l, terms[k]) + 3.0 * U * terms[k] + l as f64 * FLOOR,
                            &format!("centred squares, {what}"),
                        );
                        let eps32 = f64::from(EPS as f32);
                        let want: Vec<f64> = squares
                            .iter()
                            .map(|&q| 1.0 / (f64::from(q) / m + eps32).sqrt())
                            .collect();
                        assert_within(inv, &want, |k| 8.0 * U * want[k], &format!("inv, {what}"));
                        let want: Vec<f64> =
                            squares.iter().map(|&q| f64::from(q) / (m - 1.0)).collect();
                        assert_within(
                            unbiased,
                            &want,
                            |k| 5.0 * U * want[k],
                            &format!("unbiased variance, {what}"),
                        );
                    }
                    let (g, beta) = affine.unwrap_or_default();
                    let mut bound = Vec::with_capacity(x.len());
                    let want: Vec<f64> = x
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let k = channel(dims, i);
                            let t = (f64::from(v) - f64::from(mean[k])) * f64::from(inv[k]);
                            let iv = f64::from(inv[k]);
                            let (y, b) = match affine {
                                Some(_) => {
                                    let (tg, gk) = (t * f64::from(g[k]), f64::from(g[k]).abs());
                                    let y = tg + f64::from(beta[k]);
                                    let flushes = (iv * gk + gk + 1.0) * FLOOR;
                                    (y, U * (y.abs() + 3.0 * tg.abs()) + flushes)
                                }
                                None => (t, 2.0 * U * t.abs() + iv * FLOOR),
                            };
                            bound.push(second(b));
                            y
                        })
                        .collect();
                    let worst = assert_within(
                        &got.y,
                        &want,
                        |i| bound[i],
                        &format!("normalisation, {what}"),
                    );
                    println!(
                        "normalisation {what}: worst {:.3} of its bound",
                        worst.ratio()
                    );
                    if affine.is_some() && running.is_none() {
                        let again = run(&mut kit, &bn, case);
                        assert_exact(&again.y, &got.y, &format!("a second run, {what}"));
                    }
                }
            }
        }
    });
}

/// Values of a tensor shaped `dims` for an exact twin of the centred squares: each one of
/// -2, -1, 1 and 2, and every channel summing to exactly 0 (the second half of a channel
/// negates the first, and an odd channel ends 1, 1, -2), so the GPU's mean is exactly 0.
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

/// The four sums on small integers, bit for bit, as the CPU's in-order `f32` sums. `Σx` and
/// the centred squares come from inputs whose channels sum to 0. `Σ dy` and `Σ dy·x̂` come
/// with statistics uploaded by hand (mean 0.5, inv 2, so every `x̂` is an odd integer). Then
/// the input gradient through running statistics, `dy · γ · inv`, which is exact too.
#[test]
fn sums_exact_twin() {
    for_each_gpu(|gpu| {
        let mut all = shapes();
        all.push(ChannelDims::new(3, 2, 5));
        for (s, dims) in all.into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
            let (c, s) = (dims.channels as usize, 100 * s as u64);
            let x = zero_mean(dims, s);
            let case = Case {
                dims: Some(dims),
                x: &x,
                ..Case::default()
            };
            let got = run(&mut kit, &bn, case);
            let (sums, squares) = got.sums.expect("batch sums");
            let mut want = (vec![0.0f32; c], vec![0.0f32; c]);
            for (i, &v) in x.iter().enumerate() {
                want.0[channel(dims, i)] += v;
                want.1[channel(dims, i)] += v * v;
            }
            assert_exact(&sums, &want.0, &format!("Σx {dims:?}"));
            assert_exact(&squares, &want.1, &format!("centred squares {dims:?}"));

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
                    dims: Some(dims),
                    x: &x,
                    affine: Some((&w, &bias)),
                    running,
                    dy: Some(&dy),
                    stats: Some(&stats),
                    ..Case::default()
                };
                let got = run(&mut kit, &bn, case);
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
    });
}

/// The forward pass end to end against `Tape::batch_norm2d` on an `f64` store, from the
/// batch statistics and from running ones, with and without `γ`, `β`. The bound composes
/// the kernels' bounds through the statistics ([`batch_exact`]). A constant channel is
/// included, whose variance is 0 and whose GPU mean may still be off by a division's
/// rounding, which `inv = 1/√eps` then magnifies: the bound allows it, so the test asserts
/// no exact `β`. From batch statistics at large `L` the bound is loose: the sum's bound
/// grows with `L`, so a channel's mean is allowed an error near its own size. There, the
/// kernel twins and the small shapes are what catch a wrong formula.
#[test]
fn forward_matches_the_tape() {
    for_each_gpu(|gpu| {
        let mut all = shapes();
        all.push(ChannelDims::new(4, 2, 3));
        for (s, dims) in all.into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
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
                        dims: Some(dims),
                        x: &x,
                        affine,
                        running,
                        ..Case::default()
                    };
                    let got = run(&mut kit, &bn, case);
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
                    let worst = assert_within(&got.y, &want, bound, &format!("y, {what}"));
                    println!("y {what}: worst {:.3} of its bound", worst.ratio());
                }
            }
        }
    });
}

/// Each backward kernel against the exact result of its own inputs as the GPU computed them:
/// - `Σ dy` and `Σ dy·x̂` (from the GPU's statistics) within the sums' bounds;
/// - `dx` within `u·(|dx| + 12·(inv/m)·S)` through batch statistics and `2u·|dx|` through
///   running ones.
///
/// Without `γ` the kernels run with `g = 1`, so they give the bits of an affine run with
/// `γ = 1`.
#[test]
fn backward_kernels_twin() {
    for_each_gpu(|gpu| {
        for (s, dims) in shapes().into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
            let (c, s) = (dims.channels as usize, 100 * s as u64 + 11);
            let (l, m) = (dims.summed() as usize, dims.summed() as f64);
            let x = seeded_values(s, dims.len() as usize, -2.0, 2.0);
            let dy = seeded_values(s + 1, x.len(), -1.0, 1.0);
            let (w, b) = (weights(s + 2, c), seeded_values(s + 4, c, -1.0, 1.0));
            let (rm, rv) = (
                seeded_values(s + 5, c, -1.0, 1.0),
                seeded_values(s + 6, c, 0.5, 2.0),
            );
            let (ones, zeros) = (vec![1.0f32; c], vec![0.0f32; c]);
            for running in [None, Some((&rm[..], &rv[..]))] {
                let what = format!("{dims:?}, running {}", running.is_some());
                let case = Case {
                    dims: Some(dims),
                    x: &x,
                    affine: Some((&w, &b)),
                    running,
                    dy: Some(&dy),
                    ..Case::default()
                };
                let got = run(&mut kit, &bn, case);
                let (mean, inv) = (&got.stats[..c], &got.stats[c..2 * c]);
                let xhat = |i: usize| {
                    let k = channel(dims, i);
                    (f64::from(x[i]) - f64::from(mean[k])) * f64::from(inv[k])
                };
                let (mut sdy, mut adx, mut sdx, mut asdx, mut flushes) = (
                    vec![0.0; c],
                    vec![0.0; c],
                    vec![0.0; c],
                    vec![0.0; c],
                    vec![0.0; c],
                );
                for (i, &g) in dy.iter().enumerate() {
                    let (k, g) = (channel(dims, i), f64::from(g));
                    sdy[k] += g;
                    adx[k] += g.abs();
                    sdx[k] += g * xhat(i);
                    asdx[k] += (g * xhat(i)).abs();
                    flushes[k] += (g.abs() * (f64::from(inv[k]) + 1.0) + 1.0) * FLOOR;
                }
                assert_within(
                    &got.db,
                    &sdy,
                    |k| sum_error(l, adx[k]),
                    &format!("dβ, {what}"),
                );
                assert_within(
                    &got.dw,
                    &sdx,
                    |k| sum_error(l, asdx[k]) + 3.0 * U * asdx[k] + flushes[k],
                    &format!("dγ, {what}"),
                );
                let mut bound = Vec::with_capacity(x.len());
                let want: Vec<f64> = dy
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| {
                        let k = channel(dims, i);
                        let (g, iv) = (f64::from(w[k]), f64::from(inv[k]));
                        let gy = f64::from(d) * g;
                        if running.is_some() {
                            bound.push(second(2.0 * U * (gy * iv).abs()) + iv * FLOOR);
                            return gy * iv;
                        }
                        let sg = f64::from(got.db[k]) * g;
                        let sgx = xhat(i) * f64::from(got.dw[k]) * g;
                        let dx = iv / m * (m * gy - sg - sgx);
                        let scale = iv / m * (m * gy.abs() + sg.abs() + sgx.abs());
                        let s = m * gy.abs() + sg.abs() + sgx.abs();
                        let raw = (f64::from(got.dw[k]) * g).abs();
                        let flushes = 2.0 * iv + iv / m * (5.0 + (1.0 + iv) * raw) + s;
                        bound.push(second(U * (dx.abs() + 12.0 * scale)) + flushes * FLOOR);
                        dx
                    })
                    .collect();
                let worst = assert_within(&got.dx, &want, |i| bound[i], &format!("dx, {what}"));
                println!("dx {what}: worst {:.3} of its bound", worst.ratio());
                let plain = Case {
                    affine: None,
                    ..case
                };
                let unit = Case {
                    affine: Some((&ones, &zeros)),
                    ..case
                };
                let (plain, unit) = (run(&mut kit, &bn, plain), run(&mut kit, &bn, unit));
                assert_exact(&plain.dx, &unit.dx, &format!("dx without γ, {what}"));
            }
        }
    });
}

/// The gradients end to end against the tape's backward pass on an `f64` store, for a loss
/// `Σ y · r` (so `dy = r`), through the batch statistics (a training tape) and through
/// running ones (an evaluation tape). The tolerance is [`TOLERANCE`] of each gradient's
/// scale. This checks the formula, not the rounding, which the kernel twins bound.
#[test]
fn backward_matches_the_tape() {
    for_each_gpu(|gpu| {
        for (s, dims) in shapes().into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
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
                let got = run(
                    &mut kit,
                    &bn,
                    Case {
                        dims: Some(dims),
                        x: &x,
                        affine: Some((&w, &bias)),
                        running,
                        dy: Some(&dy),
                        ..Case::default()
                    },
                );
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
                assert_within(
                    &got.db,
                    store.grad(bi).data(),
                    |k| sum_error(l, ady[k]),
                    &format!("dβ against the tape, {what}"),
                );
                assert_within(
                    &got.dw,
                    store.grad(wi).data(),
                    |k| TOLERANCE * asdx[k],
                    &format!("dγ against the tape, {what}"),
                );
                let scale = |i: usize| {
                    let k = channel(dims, i);
                    let (g, iv) = (f64::from(w[k]).abs(), exact[k].inv);
                    let gy = f64::from(dy[i]).abs() * g;
                    match running {
                        Some(_) => gy * iv,
                        None => iv / m * (m * gy + sdy[k].abs() * g + (xhat(i) * sdx[k]).abs() * g),
                    }
                };
                assert_within(
                    &got.dx,
                    store.grad(xi).data(),
                    |i| TOLERANCE * scale(i),
                    &format!("dx against the tape, {what}"),
                );
            }
        }
    });
}

/// The running statistics over three training steps against `Graph::apply_buffer_updates`
/// on an `f64` store, for each way of weighting a batch:
/// - a momentum of 0.1 with a count kept;
/// - the cumulative average from a count of 2;
/// - neither (momentum 1).
///
/// The count after each step is the CPU's. The bound carries each step's error into the
/// next: the batch statistics' bounds ([`batch_exact`]), the kernel's rounding, and the
/// difference between the `f32` rates and the CPU's `f64` ones.
#[test]
fn running_statistics_match_the_cpu() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::new(4, 3, 5);
        let (b, c, p) = (4, 3, 5);
        for (momentum, count) in [(Some(0.1), Some(0.0f32)), (None, Some(2.0)), (None, None)] {
            let what = format!("momentum {momentum:?}, count {count:?}");
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
            let (rm, rv) = (
                seeded_values(1, c, -1.0, 1.0),
                seeded_values(2, c, 0.5, 2.0),
            );
            let mut store = ParamStore::<f64>::new_generic();
            let stats_ids = RunningStats {
                mean: store.add(Tensor::from_vec(wide(&rm), &[c]).unwrap()),
                var: store.add(Tensor::from_vec(wide(&rv), &[c]).unwrap()),
                count: count
                    .map(|n| store.add(Tensor::from_vec(vec![f64::from(n)], &[1]).unwrap())),
                momentum,
            };
            let mut graph = Graph::<f64>::new_generic();
            let (xb, y, stats) = (
                kit.zeros(dims.len() as usize),
                kit.zeros(dims.len() as usize),
                kit.zeros(3 * c),
            );
            let (mean, var, finite) = (kit.upload(&rm), kit.upload(&rv), kit.zeros(1));
            let forward = bn.forward_call(
                &mut kit.steps,
                &mut kit.pool,
                dims,
                EPS,
                BatchStats::Batch,
                &xb,
                None,
                &y,
                &stats,
            );
            let mut update = bn.running_call(
                &mut kit.steps,
                &mut kit.pool,
                &forward,
                &stats,
                (&mean, &var),
                &finite,
                momentum,
                count,
            );
            let mut e_r = [vec![0.0f64; c], vec![0.0f64; c]];
            let mut previous = [wide(&rm), wide(&rv)];
            for k in 0..3u64 {
                let x = seeded_values(10 + k, dims.len() as usize, -2.0, 2.0);
                let mut tape = graph.training_tape(&store.values, 0, k);
                let xv = tape.input(Tensor::from_vec(wide(&x), &[b, c * p]).unwrap().view());
                tape.batch_norm2d(xv, None, None, Some(stats_ids), c, p, EPS);
                drop(tape);
                let cpu_count = stats_ids.count.map(|id| store.value(id).data()[0]);
                let m64 = momentum.unwrap_or_else(|| cpu_count.map_or(1.0, |n| 1.0 / (n + 1.0)));
                graph.apply_buffer_updates(&mut store.values);

                let (m32, k32) = update.rates();
                let mut step = kit.steps.begin();
                let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                step.upload(xb.buffer(), 0, &bytes);
                bn.record_forward(&mut step, &forward);
                bn.record_running(&mut step, &update);
                let _ = step.finish();
                update.commit(true);
                assert_eq!(
                    update.count().map(f64::from),
                    stats_ids.count.map(|id| store.value(id).data()[0]),
                    "the count after step {k}, {what}"
                );

                let exact = batch_exact(dims, &x);
                let (m32, k32) = (f64::from(m32), f64::from(k32));
                let (dm, dk) = ((m32 - m64).abs(), (k32 - (1.0 - m64)).abs());
                assert!(
                    dm <= U * m64 && dk <= U * (1.0 - m64) + dm,
                    "step {k}, {what}: the rates ({m32}, {k32}) are not the CPU's momentum \
                     {m64} rounded to f32"
                );
                for (slot, (buffer, id)) in [(&mean, stats_ids.mean), (&var, stats_ids.var)]
                    .into_iter()
                    .enumerate()
                {
                    let want = store.value(id).data().to_vec();
                    let bound: Vec<f64> = (0..c)
                        .map(|j| {
                            let (b, e_b) = match slot {
                                0 => (exact[j].mean, exact[j].e_mean),
                                _ => (exact[j].unbiased, exact[j].e_unbiased),
                            };
                            let r = previous[slot][j];
                            let rounding = 2.0
                                * U
                                * (want[j].abs()
                                    + m32 * (b.abs() + e_b)
                                    + k32 * (r.abs() + e_r[slot][j]));
                            m32 * e_b
                                + k32 * e_r[slot][j]
                                + dm * b.abs()
                                + dk * r.abs()
                                + second(rounding)
                                + 64.0 * f64::EPSILON * want[j].abs()
                                + FLOOR
                        })
                        .collect();
                    let got = kit.read(buffer);
                    assert_within(
                        &got,
                        &want,
                        |j| bound[j],
                        &format!(
                            "running {} after step {k}, {what}",
                            ["mean", "variance"][slot]
                        ),
                    );
                    e_r[slot] = bound;
                    previous[slot] = want;
                }
            }
            forward.release(&mut kit.pool);
            update.release(&mut kit.pool);
        }
    });
}

/// A set finite flag leaves the running statistics as they were, bit for bit, and a step
/// committed as not run leaves the count and the next rates as they were; the next clear
/// step updates them as usual.
#[test]
fn a_set_finite_flag_skips_the_update() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::new(2, 3, 4);
        let c = 3;
        let mut kit = kit(gpu, dims);
        let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
        let x = seeded_values(1, dims.len() as usize, -2.0, 2.0);
        let (rm, rv) = (
            seeded_values(2, c, -1.0, 1.0),
            seeded_values(3, c, 0.5, 2.0),
        );
        let (xb, y, stats) = (kit.upload(&x), kit.zeros(x.len()), kit.zeros(3 * c));
        let (mean, var) = (kit.upload(&rm), kit.upload(&rv));
        let finite = kit.upload(&[f32::from_bits(1)]);
        let forward = bn.forward_call(
            &mut kit.steps,
            &mut kit.pool,
            dims,
            EPS,
            BatchStats::Batch,
            &xb,
            None,
            &y,
            &stats,
        );
        let mut update = bn.running_call(
            &mut kit.steps,
            &mut kit.pool,
            &forward,
            &stats,
            (&mean, &var),
            &finite,
            None,
            Some(3.0),
        );
        let rates = update.rates();
        let mut step = kit.steps.begin();
        bn.record_forward(&mut step, &forward);
        bn.record_running(&mut step, &update);
        let _ = step.finish();
        update.commit(false);
        assert_exact(&kit.read(&mean), &rm, "the running mean under a set flag");
        assert_exact(
            &kit.read(&var),
            &rv,
            "the running variance under a set flag",
        );
        assert_eq!(update.count(), Some(3.0), "the count of a skipped step");
        assert_eq!(update.rates(), rates, "the rates after a skipped step");
        assert_eq!(rates, (0.25, 0.75), "one over the count after this batch");

        let mut step = kit.steps.begin();
        step.clear(finite.buffer(), 0, 4);
        bn.record_running(&mut step, &update);
        let _ = step.finish();
        update.commit(true);
        assert_eq!(update.count(), Some(4.0));
        let batch = kit.read(&stats);
        for (what, buffer, old, at) in [("mean", &mean, &rm, 0), ("variance", &var, &rv, 2 * c)] {
            let b = &batch[at..at + c];
            let want: Vec<f64> = old
                .iter()
                .zip(b)
                .map(|(&r, &b)| 0.25 * f64::from(b) + 0.75 * f64::from(r))
                .collect();
            assert_within(
                &kit.read(buffer),
                &want,
                |j| {
                    2.0 * U
                        * (want[j].abs()
                            + 0.25 * f64::from(b[j]).abs()
                            + 0.75 * f64::from(old[j]).abs())
                        + 2.0 * FLOOR
                },
                &format!("the running {what} once the flag is clear"),
            );
        }
        forward.release(&mut kit.pool);
        update.release(&mut kit.pool);
    });
}

/// Every kernel goes by the shape, not by its buffers' lengths: with every tensor buffer
/// longer than its tensor, the values past it stay as they were, and the results are the
/// bits of exactly sized buffers.
#[test]
fn buffers_longer_than_the_tensor() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::new(3, 5, 7);
        let (n, c) = (dims.len() as usize, 5);
        let mut kit = kit(gpu, dims);
        let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
        let x = seeded_values(1, n, -2.0, 2.0);
        let dy = seeded_values(2, n, -1.0, 1.0);
        let (w, b) = (weights(3, c), seeded_values(5, c, -1.0, 1.0));
        let case = Case {
            dims: Some(dims),
            x: &x,
            affine: Some((&w, &b)),
            dy: Some(&dy),
            ..Case::default()
        };
        let exact = run(&mut kit, &bn, case);
        let long = run(&mut kit, &bn, Case { pad: 300, ..case });
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
    });
}

/// The documented refusals:
/// - a zero dimension, a negative eps, and batch statistics over one value per channel;
/// - a buffer too short;
/// - a written buffer that is also one of the buffers it is computed from;
/// - a backward pass whose affine parts do not match its forward pass;
/// - a running update after a forward pass from running statistics.
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
        let dims = ChannelDims::new(2, 2, 4);
        let mut kit = kit(gpu, dims);
        let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
        let [x, y, dy, dx] = [(); 4].map(|_| kit.zeros(16));
        let [stats, other] = [(); 2].map(|_| kit.zeros(16));
        let [w, b, dw, db, mean, var, finite] = [(); 7].map(|_| kit.zeros(2));
        let short = kit.zeros(1);
        let mut forward =
            |dims: ChannelDims,
             eps: f64,
             from: BatchStats<'_>,
             x: &cavitation_gpu::PooledBuffer,
             affine: Option<(&cavitation_gpu::PooledBuffer, &cavitation_gpu::PooledBuffer)>,
             y: &cavitation_gpu::PooledBuffer,
             stats: &cavitation_gpu::PooledBuffer| {
                message(&mut || {
                    let _ = bn.forward_call(
                        &mut kit.steps,
                        &mut kit.pool,
                        dims,
                        eps,
                        from,
                        x,
                        affine,
                        y,
                        stats,
                    );
                })
            };
        let batch = BatchStats::Batch;
        let ok = dims;
        assert!(
            forward(ChannelDims::new(0, 2, 4), EPS, batch, &x, None, &y, &stats)
                .contains("above 0")
        );
        assert!(forward(ok, -1.0, batch, &x, None, &y, &stats).contains("eps must be finite"));
        assert!(forward(ok, f64::NAN, batch, &x, None, &y, &stats).contains("eps must be finite"));
        assert!(forward(ok, 1e-40, batch, &x, None, &y, &stats).contains("at least 2⁻¹²⁶"));
        assert!(
            forward(ChannelDims::new(1, 2, 1), EPS, batch, &x, None, &y, &stats)
                .contains("more than one value per channel")
        );
        assert!(forward(ok, EPS, batch, &short, None, &y, &stats).contains("the input needs 16"));
        assert!(forward(ok, EPS, batch, &x, None, &y, &short).contains("the statistics needs 6"));
        assert!(forward(ok, EPS, batch, &x, None, &short, &stats).contains("the output needs 16"));
        let short_mean = BatchStats::Running {
            mean: &short,
            var: &var,
        };
        assert!(
            forward(ok, EPS, short_mean, &x, None, &y, &stats).contains("the running mean needs 2")
        );
        assert!(
            forward(ok, EPS, batch, &x, None, &y, &x)
                .contains("the statistics and the input must be different")
        );
        assert!(
            forward(ok, EPS, batch, &x, Some((&other, &b)), &y, &other)
                .contains("the statistics and the weight must be different")
        );
        let into_mean = BatchStats::Running {
            mean: &dy,
            var: &var,
        };
        assert!(
            forward(ok, EPS, into_mean, &x, None, &dy, &stats)
                .contains("the output and the running mean must be different")
        );
        assert!(
            forward(ok, EPS, batch, &x, Some((&short, &b)), &y, &stats)
                .contains("the weight needs 2")
        );
        assert!(
            forward(ok, EPS, batch, &x, None, &x, &stats)
                .contains("the output and the input must be different")
        );
        assert!(
            forward(ok, EPS, batch, &x, None, &stats, &stats)
                .contains("the output and the statistics must be different")
        );
        assert!(
            forward(ok, EPS, batch, &x, Some((&w, &y)), &y, &stats)
                .contains("the output and the bias must be different")
        );
        assert!(
            forward(
                ok,
                EPS,
                BatchStats::Running {
                    mean: &stats,
                    var: &var
                },
                &x,
                None,
                &y,
                &stats
            )
            .contains("the statistics and the running mean must be different")
        );

        let plain = bn.forward_call(
            &mut kit.steps,
            &mut kit.pool,
            ok,
            EPS,
            batch,
            &x,
            None,
            &y,
            &stats,
        );
        let affine = bn.forward_call(
            &mut kit.steps,
            &mut kit.pool,
            ok,
            EPS,
            batch,
            &x,
            Some((&w, &b)),
            &y,
            &stats,
        );
        let evaluation = bn.forward_call(
            &mut kit.steps,
            &mut kit.pool,
            ok,
            EPS,
            BatchStats::Running {
                mean: &mean,
                var: &var,
            },
            &x,
            None,
            &y,
            &other,
        );
        let mut backward = |forward: &cavitation_gpu::kernels::BatchNormForward,
                            grad_out: &cavitation_gpu::PooledBuffer,
                            grad_in: &cavitation_gpu::PooledBuffer,
                            parts: Option<(
            &cavitation_gpu::PooledBuffer,
            &cavitation_gpu::PooledBuffer,
            &cavitation_gpu::PooledBuffer,
        )>| {
            message(&mut || {
                let _ = bn.backward_call(
                    &mut kit.steps,
                    &mut kit.pool,
                    forward,
                    &x,
                    &stats,
                    grad_out,
                    grad_in,
                    parts,
                );
            })
        };
        assert!(
            backward(&plain, &dy, &dx, Some((&w, &dw, &db)))
                .contains("exactly when the forward pass is affine")
        );
        assert!(
            backward(&affine, &dy, &dx, None).contains("exactly when the forward pass is affine")
        );
        assert!(backward(&plain, &dy, &short, None).contains("the input gradient needs 16"));
        assert!(backward(&plain, &short, &dx, None).contains("the output gradient needs 16"));
        assert!(
            backward(&affine, &dy, &dx, Some((&w, &dw, &x)))
                .contains("the bias gradient and the input must be different")
        );
        assert!(
            backward(&plain, &dy, &dy, None)
                .contains("the input gradient and the output gradient must be different")
        );
        assert!(
            backward(&plain, &dy, &x, None)
                .contains("the input gradient and the input must be different")
        );
        assert!(
            backward(&affine, &dy, &dx, Some((&w, &dw, &dw)))
                .contains("the weight gradient and the bias gradient must be different")
        );
        assert!(
            backward(&affine, &dy, &dx, Some((&w, &w, &db)))
                .contains("the weight gradient and the weight must be different")
        );
        let mut running = |forward: &cavitation_gpu::kernels::BatchNormForward,
                           mean: &cavitation_gpu::PooledBuffer,
                           var: &cavitation_gpu::PooledBuffer,
                           finite: &cavitation_gpu::PooledBuffer| {
            message(&mut || {
                let _ = bn.running_call(
                    &mut kit.steps,
                    &mut kit.pool,
                    forward,
                    &stats,
                    (mean, var),
                    finite,
                    Some(0.1),
                    None,
                );
            })
        };
        assert!(running(&evaluation, &mean, &var, &finite).contains("from the batch's statistics"));
        assert!(
            running(&plain, &mean, &mean, &finite)
                .contains("the running mean and the running variance must be different")
        );
        assert!(
            running(&plain, &stats, &var, &finite)
                .contains("the running mean and the statistics must be different")
        );
        assert!(
            running(&plain, &mean, &var, &mean)
                .contains("the running mean and the finite flag must be different")
        );
        assert!(
            running(&plain, &mean, &var, &var)
                .contains("the running variance and the finite flag must be different")
        );
        assert!(
            running(&plain, &mean, &stats, &finite)
                .contains("the running variance and the statistics must be different")
        );
        let not_its_own = message(&mut || {
            let _ = bn.backward_call(
                &mut kit.steps,
                &mut kit.pool,
                &plain,
                &x,
                &other,
                &dy,
                &dx,
                None,
            );
        });
        assert!(not_its_own.contains("must read the forward pass's input and statistics"));
        let not_its_own = message(&mut || {
            let _ = bn.running_call(
                &mut kit.steps,
                &mut kit.pool,
                &plain,
                &other,
                (&mean, &var),
                &finite,
                None,
                None,
            );
        });
        assert!(not_its_own.contains("must read the forward pass's statistics"));
    });
}

/// The rates are the CPU's own `f32` momentum and `1 − momentum`, bit for bit, for each way
/// of weighting a batch, over four batches. On an `f32` store, a constant batch (mean exactly
/// 1, unbiased variance exactly 0) moves a running mean of 0 to exactly the momentum and a
/// running variance of 1 to exactly `1 − momentum`.
#[test]
fn rates_are_the_cpus_bits() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::new(2, 1, 3);
        for (momentum, count) in [
            (Some(0.1), Some(0.0f32)),
            (Some(0.3), None),
            (None, Some(2.0)),
            (None, None),
        ] {
            let mut kit = kit(gpu, dims);
            let bn = BatchNorm2d::new(gpu, &mut kit.cache).unwrap();
            let [x, y] = [(); 2].map(|_| kit.zeros(6));
            let stats = kit.zeros(3);
            let [mean, var, finite] = [(); 3].map(|_| kit.zeros(1));
            let forward = bn.forward_call(
                &mut kit.steps,
                &mut kit.pool,
                dims,
                EPS,
                BatchStats::Batch,
                &x,
                None,
                &y,
                &stats,
            );
            let mut update = bn.running_call(
                &mut kit.steps,
                &mut kit.pool,
                &forward,
                &stats,
                (&mean, &var),
                &finite,
                momentum,
                count,
            );
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
            forward.release(&mut kit.pool);
            update.release(&mut kit.pool);
        }
    });
}
