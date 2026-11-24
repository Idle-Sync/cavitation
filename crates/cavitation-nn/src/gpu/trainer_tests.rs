//! `GpuTrainer` against the CPU: each parameter's gradient after one step against the `f64`
//! training tape's from the same weights, batch and dropout masks, on the test MLP, a small
//! CNN (convolution, batch norm, max pooling, dropout), a CNN through average and adaptive
//! average pooling, `mnist-cnn` and `cifar-cnn`, with the max-pooling winners that differ from
//! the `f64` tape's listed, and every loss besides cross-entropy on an MLP (binary
//! cross-entropy after a sigmoid, and on logits; KL divergence after a log-softmax, on
//! target distributions); the parameters
//! after one SGD step within the update's bound; two runs giving the same bits; and a step
//! on non-finite inputs leaving the parameters and the optimiser's state as they were.
//!
//! The gradient check is a tolerance, not a derived bound: each tensor's worst difference
//! from the `f64` gradient must be under 2⁻¹² of the tensor's largest `f64` gradient plus
//! 2⁻²⁰ of the network's largest, and the tests print the measured worst ratio to that
//! tolerance. The second term covers a gradient that is 0 in exact arithmetic, such as a
//! convolution's bias before a batch norm (the norm cancels it), where the GPU's sum gives
//! rounding of the size of the terms it cancels.
//!
//! Without a GPU every test says so and checks nothing; `CAVITATION_GPU_REQUIRED=1` makes
//! that a failure.

use cavitation_core::{
    Conv2dGeom, Gate, Graph, ParamStore, Pool2dGeom, Real, Rng, Tensor, TensorView,
};
use cavitation_gpu::{Backend, Gpu};

use super::GpuTrainer;
use super::forward_tests::{
    ADAPTIVE_OUTPUTS, GLU_GATES, Twin, activation_mlp, glu_mlp, mlp, more_activations, pool_cnn,
    prelu_mlp, softmax_mlp, twin,
};
use crate::{
    BatchNorm2d, BatchView, Conv2d, Dropout, Flatten, InferenceLayer, Init, Linear, LogSoftmax,
    Loss, MaxPool2d, NeuronMask, OptimizerConfig, Relu, Sequential, Sigmoid, TargetsView, Trace,
};

/// The unit roundoff of `f32`.
const U: f64 = 1.0 / (1u64 << 24) as f64;

/// Runs `body` on each GPU this platform has, one per backend, and returns how many ran.
fn for_each_gpu(mut body: impl FnMut(&Gpu)) -> usize {
    let mut ran = 0;
    for &backend in Backend::on_this_platform() {
        if let Some(gpu) = Gpu::with_backend(backend) {
            println!("on {backend}: {}", gpu.info().name);
            body(&gpu);
            ran += 1;
        }
    }
    if ran == 0 {
        println!("no GPU: skipped");
    }
    ran
}

/// `n` values in `[lo, hi)` from `seed`.
fn uniform(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..n).map(|_| rng.range(lo, hi) as f32).collect()
}

/// A small CNN on 2 × 8 × 8 images: a convolution with a bias, batch norm, ReLU, max pooling,
/// dropout, flatten and a `Linear`, in either precision.
fn small_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new_generic()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::new(params, 4, (8, 8)))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 5, &mut rng))
}

/// `mnist-cnn`, as the benchmark builds it.
fn mnist_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let g1 = Conv2dGeom::with_options(1, 6, (28, 28), (5, 5), (1, 1), (2, 2), (1, 1)).unwrap();
    let g2 = Conv2dGeom::new(6, 16, (14, 14), (5, 5)).unwrap();
    let pool = |c, s| MaxPool2d::new(Pool2dGeom::new(c, (s, s), (2, 2)).unwrap());
    Sequential::new_generic()
        .push(Conv2d::new(params, g1, &mut rng))
        .push(Relu)
        .push(pool(6, 28))
        .push(Conv2d::new(params, g2, &mut rng))
        .push(Relu)
        .push(pool(16, 10))
        .push(Flatten)
        .push(Linear::new(params, 400, 120, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 120, 84, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 84, 10, &mut rng))
}

/// `cifar-cnn`, as the benchmark builds it.
fn cifar_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let mut net = Sequential::new_generic();
    for (i, o, s) in [(3, 32, 32), (32, 32, 32), (32, 64, 16), (64, 64, 16)] {
        let g = Conv2dGeom::with_options(i, o, (s, s), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        net = net
            .push(Conv2d::with_init(
                params,
                g,
                Init::TorchDefault,
                None,
                &mut rng,
            ))
            .push(BatchNorm2d::new(params, o, (s, s)))
            .push(Relu);
        if i == o {
            net = net.push(MaxPool2d::new(Pool2dGeom::new(o, (s, s), (2, 2)).unwrap()));
        }
    }
    net.push(Dropout::new(0.25).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 4096, 256, &mut rng))
        .push(Relu)
        .push(Dropout::new(0.5).unwrap())
        .push(Linear::new(params, 256, 10, &mut rng))
}

/// One batch: inputs and class labels.
struct Data {
    /// `[rows, width]` inputs.
    x: Vec<f32>,
    /// The rows.
    rows: usize,
    /// One label per row.
    labels: Vec<usize>,
    /// The loss the step minimises.
    loss: Loss,
    /// For a loss on values, `[rows, outputs]` targets; empty otherwise.
    values: Vec<f32>,
}

impl Data {
    /// `rows` rows of `width` inputs in `[lo, hi)` and labels below `classes`, from `seed`.
    fn new(seed: u64, rows: usize, width: usize, classes: usize, (lo, hi): (f64, f64)) -> Data {
        let mut rng = Rng::seed(seed + 1);
        Data {
            x: uniform(seed, rows * width, lo, hi),
            rows,
            labels: (0..rows)
                .map(|_| (rng.range(0.0, classes as f64) as usize).min(classes - 1))
                .collect(),
            loss: Loss::CrossEntropy,
            values: Vec::new(),
        }
    }

    /// The batch for `loss`: for a loss on values, with `outputs` targets per row in
    /// `[−1, 1)` from `seed`, for binary cross-entropy in `[0, 1)` with every fifth 0 or 1,
    /// or for KL divergence a distribution per row.
    fn for_loss(self, loss: Loss, outputs: usize, seed: u64) -> Data {
        let values = if loss.needs_classes() {
            Vec::new()
        } else if is_binary(loss) {
            binary_targets(seed, self.rows * outputs)
        } else if loss == Loss::KlDiv {
            distributions(seed, self.rows, outputs)
        } else {
            uniform(seed, self.rows * outputs, -1.0, 1.0)
        };
        Data {
            loss,
            values,
            ..self
        }
    }

    /// The batch as a trainer takes it.
    fn view(&self) -> BatchView<'_, f32> {
        BatchView {
            x: TensorView::from_slice(&self.x, &[self.rows, self.x.len() / self.rows]).unwrap(),
            targets: if self.loss.needs_classes() {
                TargetsView::Classes(&self.labels)
            } else {
                let outputs = self.values.len() / self.rows;
                TargetsView::Values(
                    TensorView::from_slice(&self.values, &[self.rows, outputs]).unwrap(),
                )
            },
        }
    }
}

/// A max pooling's layer and its input on the `f64` tape.
type Pooled = (usize, Vec<f64>);

/// Each parameter's gradient on the `f64` training tape (seed 0, step 1, as a trainer's
/// first step), with the batch's loss, in the `f64` store's parameter order, and each max
/// pooling's input, by layer.
fn tape_gradients(t: &mut Twin, data: &Data) -> (Vec<Vec<f64>>, Vec<Pooled>) {
    let x64: Vec<f64> = data.x.iter().map(|&v| f64::from(v)).collect();
    let input = Tensor::from_vec(x64, &[data.rows, data.x.len() / data.rows]).unwrap();
    let mut graph = Graph::<f64>::default();
    let mut pooled = Vec::new();
    {
        let mut tape = graph.training_tape(&t.params64.values, 0, 1);
        let v = tape.input(input.view());
        let mut trace = Trace::new();
        let out = t.net64.forward_traced(&mut tape, v, &mut trace);
        for i in 0..t.net.len() {
            if matches!(
                t.net.layer(i).inference(&t.params.values),
                Some(InferenceLayer::MaxPool2d(_))
            ) {
                pooled.push((i, tape.value(trace.input(i)).data().to_vec()));
            }
        }
        let wide: Vec<f64> = data.values.iter().map(|&v| f64::from(v)).collect();
        let values = (!wide.is_empty())
            .then(|| TensorView::from_slice(&wide, &[data.rows, wide.len() / data.rows]).unwrap());
        let loss = data.loss.record_f64(&mut tape, out, &data.labels, values);
        tape.backward(loss, &mut t.params64.grads);
    }
    let grads = (t.net64.parameter_names().into_iter())
        .map(|(_, id)| t.params64.grad(id).data().to_vec())
        .collect();
    (grads, pooled)
}

/// The max-pooling windows whose winner on the `f64` tape's input differs from the GPU's:
/// `(layer, output index)`, with the window's two largest `f64` values. The winner is the
/// window's first largest value, by its position in the input plane.
fn flipped_winners(
    t: &Twin,
    pooled: &[Pooled],
    gpu: &[Vec<u32>],
    rows: usize,
) -> Vec<(usize, usize, f64, f64)> {
    let mut flips = Vec::new();
    for ((layer, input), winners) in pooled.iter().zip(gpu) {
        let Some(InferenceLayer::MaxPool2d(g)) = t.net.layer(*layer).inference(&t.params.values)
        else {
            unreachable!("only max pooling layers are listed")
        };
        let (h, w) = g.input();
        let (kh, kw) = g.kernel();
        let (sh, sw) = g.stride();
        let out_w = (w - kw) / sw + 1;
        let out_h = (h - kh) / sh + 1;
        for img in 0..rows * g.channels() {
            let plane = &input[img * h * w..(img + 1) * h * w];
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut best = (f64::NEG_INFINITY, 0usize, f64::NEG_INFINITY);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let at = (oy * sh + ky) * w + ox * sw + kx;
                            let v = plane[at];
                            if v > best.0 {
                                best = (v, at, best.0);
                            } else if v > best.2 {
                                best.2 = v;
                            }
                        }
                    }
                    let o = img * out_h * out_w + oy * out_w + ox;
                    if winners[o] as usize != best.1 {
                        flips.push((*layer, o, best.0, best.2));
                    }
                }
            }
        }
    }
    flips
}

/// One trainer step with a learning rate of 0, and each parameter's GPU gradient against the
/// `f64` tape's, on every GPU, with the flipped max-pooling winners listed.
fn check_gradients(mut t: Twin, data: &Data, what: &str) {
    let (want, pooled) = tape_gradients(&mut t, data);
    let names: Vec<String> = t
        .net
        .parameter_names()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for_each_gpu(|gpu| {
        let still = OptimizerConfig::Sgd {
            lr: 0.0,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut trainer =
            GpuTrainer::new(&t.net, &t.params, &still, data.loss, data.rows, gpu).unwrap();
        trainer.step(&t.net, data.view()).unwrap();
        let got = trainer.last_gradients();
        let winners = trainer.last_winners();
        let flips = flipped_winners(&t, &pooled, &winners, data.rows);
        println!(
            "{what}: {} max-pooling winners differ from the f64 tape's",
            flips.len()
        );
        for (layer, o, a, b) in flips.iter().take(20) {
            println!("  layer {layer}, output {o}: f64 window's top two {a:e}, {b:e}");
        }
        // The GPU's parameters are in first-use order, which is the parameter order here.
        assert_eq!(
            (names.len(), want.len()),
            (got.len(), got.len()),
            "{what}: parameter counts"
        );
        let ids: Vec<_> = t
            .net
            .parameter_names()
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert!(
            got.iter().map(|(id, _)| *id).eq(ids),
            "{what}: parameter order"
        );
        // A flipped winner sends a gradient to another input, which no tolerance covers, so
        // with flips only the layers after the last max pooling are asserted.
        let after_pools = pooled.last().map_or(0, |(layer, _)| layer + 1);
        let mut worst = (0.0f64, String::new());
        let largest = want.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
        for ((name, w), (_, g)) in names.iter().zip(&want).zip(&got) {
            let scale = w.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let tolerance = scale / 4096.0 + largest / 1_048_576.0;
            let off = g
                .iter()
                .zip(w)
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            let ratio = off / tolerance;
            let layer: usize = name.split('.').next().and_then(|l| l.parse().ok()).unwrap();
            if flips.is_empty() || layer >= after_pools {
                assert!(
                    ratio <= 1.0,
                    "{what}: {name}: off by {off:e} of a largest gradient {scale:e}"
                );
            }
            println!(
                "{what} {name}: worst {ratio:.4} of the tolerance (largest gradient {scale:.3e})"
            );
            if ratio > worst.0 {
                worst = (ratio, name.clone());
            }
        }
        println!(
            "{what}: worst {:.4} of the tolerance ({})",
            worst.0, worst.1
        );
    });
}

/// The test MLP (with a neuron mask) and a small CNN with batch norm, pooling and dropout.
#[test]
fn gradients_match_the_tape() {
    let mut t = twin(mlp::<f64>, mlp::<f32>, 21);
    let mask = NeuronMask::from_off(16, &[1, 5]);
    t.net.set_mask(1, mask.clone());
    t.net64.set_mask(1, mask);
    check_gradients(t, &Data::new(1, 24, 12, 4, (-2.0, 2.0)), "mlp");
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 22);
    check_gradients(t, &Data::new(2, 6, 2 * 64, 5, (-1.0, 1.0)), "small cnn");
}

/// Each activation beyond ReLU, tanh, the sigmoid and GELU, twice in an MLP: the GPU's
/// gradients against the `f64` tape's, within the tolerance above.
#[test]
fn every_activation_matches_the_tape() {
    for (i, kind) in more_activations().into_iter().enumerate() {
        let seed = 60 + i as u64;
        let t = twin(
            |p, s| activation_mlp(p, s, kind),
            |p, s| activation_mlp(p, s, kind),
            seed,
        );
        check_gradients(
            t,
            &Data::new(seed, 24, 12, 4, (-2.0, 2.0)),
            &format!("{kind:?}"),
        );
    }
}

/// Softmax and log-softmax after each `Linear` of an MLP: the GPU's gradients against the
/// `f64` tape's, within the tolerance above.
#[test]
fn softmax_layers_match_the_tape() {
    for (seed, log) in [(90, false), (91, true)] {
        let t = twin(
            |p, s| softmax_mlp(p, s, log),
            |p, s| softmax_mlp(p, s, log),
            seed,
        );
        let what = if log { "log-softmax" } else { "softmax" };
        check_gradients(t, &Data::new(seed, 24, 12, 4, (-2.0, 2.0)), what);
    }
}

/// A PReLU before each `Linear` of an MLP, the first layer among them, with one slope per
/// column and with one shared slope: the GPU's gradients, the slopes' among them, against
/// the `f64` tape's, within the tolerance above.
#[test]
fn prelu_layers_match_the_tape() {
    for (seed, shared) in [(94, false), (95, true)] {
        let t = twin(
            |p, s| prelu_mlp(p, s, shared),
            |p, s| prelu_mlp(p, s, shared),
            seed,
        );
        let what = if shared { "shared PReLU" } else { "PReLU" };
        check_gradients(t, &Data::new(seed, 24, 12, 4, (-2.0, 2.0)), what);
    }
}

/// A GLU of each gate after the first two `Linear`s of an MLP, and with the ReLU and SiLU
/// gates a GLU as the first layer too (the input 24 wide): the GPU's gradients against the
/// `f64` tape's, within the tolerance above.
#[test]
fn glu_layers_match_the_tape() {
    let leading = [(Gate::Relu, true), (Gate::Silu, true)];
    let cases = GLU_GATES.into_iter().map(|g| (g, false)).chain(leading);
    for (k, (gate, lead)) in cases.enumerate() {
        let seed = 120 + k as u64;
        let t = twin(
            |p, s| glu_mlp(p, s, gate, lead),
            |p, s| glu_mlp(p, s, gate, lead),
            seed,
        );
        let width = if lead { 24 } else { 12 };
        let what = format!("GLU with {gate:?}{}", if lead { ", leading" } else { "" });
        check_gradients(t, &Data::new(seed, 24, width, 4, (-2.0, 2.0)), &what);
    }
}

/// A CNN through average pooling and adaptive average pooling to each of
/// `ADAPTIVE_OUTPUTS` (overlapping windows, global pooling, an output larger than its input):
/// the GPU's gradients against the `f64` tape's, within the tolerance above.
#[test]
fn pool_layers_match_the_tape() {
    for (k, to) in ADAPTIVE_OUTPUTS.into_iter().enumerate() {
        let seed = 150 + k as u64;
        let t = twin(|p, s| pool_cnn(p, s, to), |p, s| pool_cnn(p, s, to), seed);
        let what = format!("adaptive pooling to {to:?}");
        check_gradients(t, &Data::new(seed, 6, 98, 5, (-1.0, 1.0)), &what);
    }
}

/// The losses on value targets the GPU trains, with thresholds that put the test MLP's
/// outputs against targets in `[−1, 1)` on both branches, and smooth L1's `beta` of 0.
const VALUE_LOSSES: [Loss; 7] = [
    Loss::Mse,
    Loss::L1,
    Loss::Huber { delta: 0.5 },
    Loss::Huber { delta: 1.0 },
    Loss::SmoothL1 { beta: 0.5 },
    Loss::SmoothL1 { beta: 1.0 },
    Loss::SmoothL1 { beta: 0.0 },
];

/// Each loss besides cross-entropy: the negative log-likelihood after an MLP whose every
/// `Linear` ends in a log-softmax, and each of `VALUE_LOSSES` after the test MLP on targets
/// in `[−1, 1)`: the GPU's gradients against the `f64` tape's, within the tolerance above.
#[test]
fn every_loss_matches_the_tape() {
    let t = twin(
        |p, s| softmax_mlp(p, s, true),
        |p, s| softmax_mlp(p, s, true),
        170,
    );
    let data = Data::new(170, 24, 12, 4, (-2.0, 2.0)).for_loss(Loss::Nll, 4, 270);
    check_gradients(t, &data, "Nll");
    for (k, loss) in VALUE_LOSSES.into_iter().enumerate() {
        let seed = 171 + k as u64;
        let t = twin(mlp::<f64>, mlp::<f32>, seed);
        let data = Data::new(seed, 24, 12, 4, (-2.0, 2.0)).for_loss(loss, 4, seed + 100);
        check_gradients(t, &data, &format!("{loss:?}"));
    }
}

/// Whether `loss` is binary cross-entropy, with or without logits.
fn is_binary(loss: Loss) -> bool {
    matches!(
        loss,
        Loss::BinaryCrossEntropy | Loss::BinaryCrossEntropyWithLogits
    )
}

/// `n` targets in `[0, 1)` from `seed`, every fifth rounded to 0 or 1.
fn binary_targets(seed: u64, n: usize) -> Vec<f32> {
    let mut t = uniform(seed, n, 0.0, 1.0);
    for v in t.iter_mut().step_by(5) {
        *v = v.round();
    }
    t
}

/// The test MLP with a sigmoid after its last `Linear`, in either precision: the
/// probabilities binary cross-entropy takes.
fn sigmoid_mlp<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    mlp(params, seed).push(Sigmoid)
}

/// Binary cross-entropy after the test MLP with a sigmoid, and on the test MLP's logits, on
/// targets in `[0, 1)` with every fifth 0 or 1, at a batch of 24 and of 37 (no multiple of
/// the workgroup, nor of 4): the GPU's gradients against the `f64` tape's, within the
/// tolerance above.
#[test]
fn binary_losses_match_the_tape() {
    for (k, rows) in [24usize, 37].into_iter().enumerate() {
        let seed = 200 + 10 * k as u64;
        let t = twin(sigmoid_mlp::<f64>, sigmoid_mlp::<f32>, seed);
        let loss = Loss::BinaryCrossEntropy;
        let data = Data::new(seed, rows, 12, 4, (-2.0, 2.0)).for_loss(loss, 4, seed + 100);
        check_gradients(t, &data, &format!("{loss:?}, {rows} rows"));
        let t = twin(mlp::<f64>, mlp::<f32>, seed + 1);
        let loss = Loss::BinaryCrossEntropyWithLogits;
        let data = Data::new(seed + 1, rows, 12, 4, (-2.0, 2.0)).for_loss(loss, 4, seed + 101);
        check_gradients(t, &data, &format!("{loss:?}, {rows} rows"));
    }
}

/// `rows` target distributions over `outputs` values from `seed`, as KL divergence takes
/// them: values in `[0, 1)` with every third exactly 0, each row divided by its sum.
fn distributions(seed: u64, rows: usize, outputs: usize) -> Vec<f32> {
    let mut t = uniform(seed, rows * outputs, 0.0, 1.0);
    for v in t.iter_mut().skip(1).step_by(3) {
        *v = 0.0;
    }
    for row in t.chunks_mut(outputs) {
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            row.iter_mut().for_each(|v| *v /= sum);
        } else {
            row[0] = 1.0;
        }
    }
    t
}

/// The test MLP with a log-softmax after its last `Linear`, in either precision: the
/// log-probabilities KL divergence takes.
fn log_softmax_mlp<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    mlp(params, seed).push(LogSoftmax)
}

/// KL divergence after the test MLP with a log-softmax, on target distributions with every
/// third value 0, at a batch of 24 and of 37 (no multiple of the workgroup, nor of 4), so the
/// batch mean divides by each: the GPU's gradients against the `f64` tape's, within the
/// tolerance above.
#[test]
fn kl_div_matches_the_tape() {
    for (k, rows) in [24usize, 37].into_iter().enumerate() {
        let seed = 230 + 10 * k as u64;
        let t = twin(log_softmax_mlp::<f64>, log_softmax_mlp::<f32>, seed);
        let loss = Loss::KlDiv;
        let data = Data::new(seed, rows, 12, 4, (-2.0, 2.0)).for_loss(loss, 4, seed + 100);
        check_gradients(t, &data, &format!("{loss:?}, {rows} rows"));
    }
}

/// One `mnist-cnn` step and one `cifar-cnn` step.
#[test]
fn benchmark_networks_match_the_tape() {
    let t = twin(mnist_cnn::<f64>, mnist_cnn::<f32>, 23);
    check_gradients(t, &Data::new(3, 16, 28 * 28, 10, (0.0, 1.0)), "mnist-cnn");
    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 24);
    check_gradients(
        t,
        &Data::new(4, 8, 3 * 32 * 32, 10, (-1.0, 1.0)),
        "cifar-cnn",
    );
}

/// The parameters after one SGD step are the old ones minus the learning rate times the
/// GPU's gradient, within the update's bound: `u·|p'| + u·(1 + u)·lr·|g|`.
#[test]
fn sgd_updates_within_its_bound() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 25);
    let data = Data::new(5, 6, 2 * 64, 5, (-1.0, 1.0));
    let lr = 0.1;
    for_each_gpu(|gpu| {
        let sgd = OptimizerConfig::Sgd {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut trainer =
            GpuTrainer::new(&t.net, &t.params, &sgd, Loss::CrossEntropy, 6, gpu).unwrap();
        trainer.step(&t.net, data.view()).unwrap();
        let grads = trainer.last_gradients();
        let mut after = t.params.clone();
        trainer.sync_to(&mut after);
        for (id, g) in grads {
            let (p0, p1) = (t.params.value(id).data(), after.value(id).data());
            for ((&a, &b), &g) in p0.iter().zip(p1).zip(&g) {
                let want = f64::from(a) - lr * f64::from(g);
                let bound = U * f64::from(b).abs() + U * (1.0 + U) * lr * f64::from(g).abs();
                assert!(
                    (f64::from(b) - want).abs() <= bound,
                    "{a} - {lr}·{g} gave {b}"
                );
            }
        }
    });
}

/// Two runs of three Adam steps from the same start give the same bits, in the
/// parameters, the running statistics, the batch counts and the optimiser's state.
#[test]
fn two_runs_give_the_same_bits() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 26);
    let batches: Vec<Data> = (0..3)
        .map(|s| Data::new(10 + s, 6, 2 * 64, 5, (-1.0, 1.0)))
        .collect();
    for_each_gpu(|gpu| {
        let run = || {
            let adam = OptimizerConfig::Adam {
                lr: 1e-2,
                weight_decay: 0.0,
            };
            let mut trainer =
                GpuTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 6, gpu).unwrap();
            let losses: Vec<u32> = batches
                .iter()
                .map(|d| trainer.step(&t.net, d.view()).unwrap().to_bits())
                .collect();
            let mut out = t.params.clone();
            trainer.sync_to(&mut out);
            let bits: Vec<Vec<u32>> = (t.net.state_names().into_iter())
                .map(|(_, id)| out.value(id).data().iter().map(|v| v.to_bits()).collect())
                .collect();
            (losses, bits, trainer.optimizer_state())
        };
        let (a, b) = (run(), run());
        assert_eq!(a.0, b.0, "losses");
        assert_eq!(a.1, b.1, "parameters and buffers");
        assert_eq!(a.2, b.2, "optimiser state");
        assert_eq!(a.2.steps, vec![3; a.2.steps.len()]);
        // The batch count went up once per step.
        let count = (t.net.layer(1).as_batch_norm2d())
            .and_then(|bn| bn.running())
            .and_then(|r| r.count)
            .unwrap();
        let mut out = t.params.clone();
        let adam = OptimizerConfig::Adam {
            lr: 1e-2,
            weight_decay: 0.0,
        };
        let mut trainer =
            GpuTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 6, gpu).unwrap();
        for d in &batches {
            trainer.step(&t.net, d.view()).unwrap();
        }
        trainer.sync_to(&mut out);
        assert_eq!(out.value(count).data()[0], 3.0);
    });
}

/// A batch with a NaN input gives `Error::NonFinite`, and leaves the parameters, the running
/// statistics, the batch count and the optimiser's update counts as they were; the step is
/// counted.
#[test]
fn a_non_finite_step_changes_nothing() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 27);
    let good = Data::new(6, 6, 2 * 64, 5, (-1.0, 1.0));
    let mut bad = Data::new(7, 6, 2 * 64, 5, (-1.0, 1.0));
    bad.x[3] = f32::NAN;
    for_each_gpu(|gpu| {
        let adam = OptimizerConfig::Adam {
            lr: 1e-2,
            weight_decay: 0.0,
        };
        let mut trainer =
            GpuTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 6, gpu).unwrap();
        trainer.step(&t.net, good.view()).unwrap();
        let mut before = t.params.clone();
        trainer.sync_to(&mut before);
        let state = trainer.optimizer_state();
        let err = trainer.step(&t.net, bad.view()).unwrap_err();
        assert!(
            err.to_string().contains("non-finite value at step 2"),
            "{err}"
        );
        assert_eq!(trainer.steps(), 2);
        let mut after = t.params.clone();
        trainer.sync_to(&mut after);
        for (name, id) in t.net.state_names() {
            let bits = |p: &ParamStore| {
                p.value(id)
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<u32>>()
            };
            assert_eq!(bits(&before), bits(&after), "{name}");
        }
        assert_eq!(trainer.optimizer_state(), state);
        // The next good step trains: nothing of the bad one is left in the gradients.
        assert!(trainer.step(&t.net, good.view()).unwrap().is_finite());
        assert_eq!(trainer.optimizer_state().steps, vec![2; state.steps.len()]);
    });
}

/// Five steps of the GPU trainer against five of the CPU's `Trainer` from the same start, on
/// the small CNN (batch norm, max pooling, dropout) with SGD and momentum, and on the MLP with
/// Adam: each step's loss within 2⁻¹⁰ of the CPU's, and the parameters, running statistics and
/// batch counts after the last within 2⁻¹⁰ of each tensor's largest value plus 2⁻¹⁶. Later
/// steps read what earlier ones left: the momentum, the moments, the running statistics, the
/// next step's dropout masks and the gradient buffers.
#[test]
fn steps_track_the_cpu_trainer() {
    let cases = [
        (
            twin(small_cnn::<f64>, small_cnn::<f32>, 29),
            2 * 64,
            5,
            OptimizerConfig::Sgd {
                lr: 0.05,
                momentum: 0.9,
                weight_decay: 1e-4,
            },
            "small cnn, sgd",
        ),
        (
            twin(mlp::<f64>, mlp::<f32>, 30),
            12,
            4,
            OptimizerConfig::Adam {
                lr: 1e-2,
                weight_decay: 0.0,
            },
            "mlp, adam",
        ),
    ];
    for (t, width, classes, config, what) in cases {
        let batches: Vec<Data> = (0..5)
            .map(|s| Data::new(40 + s, 6, width, classes, (-1.0, 1.0)))
            .collect();
        let mut cpu = t.params.clone();
        let mut trainer = crate::Trainer::new(&t.net, config.build(), Loss::CrossEntropy, 6);
        let cpu_losses: Vec<f32> = batches
            .iter()
            .map(|d| trainer.step(&t.net, &mut cpu, d.view()).unwrap())
            .collect();
        for_each_gpu(|gpu| {
            let mut gpu_trainer =
                GpuTrainer::new(&t.net, &t.params, &config, Loss::CrossEntropy, 6, gpu).unwrap();
            for (s, d) in batches.iter().enumerate() {
                let loss = gpu_trainer.step(&t.net, d.view()).unwrap();
                let want = cpu_losses[s];
                assert!(
                    (loss - want).abs() <= want.abs() / 1024.0,
                    "{what}: step {}: loss {loss}, the CPU's {want}",
                    s + 1
                );
            }
            let mut out = t.params.clone();
            gpu_trainer.sync_to(&mut out);
            for (name, id) in t.net.state_names() {
                let (g, c) = (out.value(id).data(), cpu.value(id).data());
                let scale = c.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = g
                    .iter()
                    .zip(c)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    off <= scale / 1024.0 + 1.0 / 65536.0,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                println!("{what} {name}: off by {off:.3e} of a largest value {scale:.3e}");
            }
        });
    }
}

/// The share of a `cifar-cnn` training step at batch 128 spent copying the gradients into the
/// optimiser's flat buffer and the values back out: 20 copy-only steps against 20 training
/// steps, each less an empty step's submit and wait. A measurement, printed, not asserted.
/// Run with `--ignored`.
#[test]
#[ignore = "a measurement: run with --ignored"]
fn copy_share_of_a_cifar_step() {
    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 28);
    let data = Data::new(8, 128, 3 * 32 * 32, 10, (-1.0, 1.0));
    for_each_gpu(|gpu| {
        let sgd = OptimizerConfig::Sgd {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
        };
        let mut trainer =
            GpuTrainer::new(&t.net, &t.params, &sgd, Loss::CrossEntropy, 128, gpu).unwrap();
        for _ in 0..3 {
            trainer.step(&t.net, data.view()).unwrap();
        }
        let n = 20;
        let start = std::time::Instant::now();
        for _ in 0..n {
            trainer.step(&t.net, data.view()).unwrap();
        }
        let steps = start.elapsed().as_secs_f64();
        let (copies, empty) = trainer.time_copies(n);
        let (step_ms, copy_ms) = (
            1e3 * (steps - empty) / n as f64,
            1e3 * (copies - empty) / n as f64,
        );
        println!(
            "{}: a step {step_ms:.3} ms, its copies {copy_ms:.3} ms ({:.2}% of the step), an empty step {:.3} ms",
            gpu.info().backend,
            100.0 * copy_ms / step_ms,
            1e3 * empty / n as f64
        );
    });
}

/// The backward time of each `Linear` and `Conv2d` of a `cifar-cnn` training step at batch
/// 128 with Adam, with its parameter gradients and its input gradient apart: each the median
/// of 11 profiled steps after 5 warm-up steps and 21 steps without a profile, whose median
/// is the step's own time. A measurement, printed, not asserted. Run with `--ignored`.
#[test]
#[ignore = "a measurement: run with --ignored"]
fn backward_parts_of_a_cifar_step() {
    use std::time::{Duration, Instant};

    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 28);
    let data = Data::new(8, 128, 3 * 32 * 32, 10, (-1.0, 1.0));
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    for_each_gpu(|gpu| {
        let adam = OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        };
        let mut trainer =
            GpuTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 128, gpu).unwrap();
        for _ in 0..5 {
            trainer.step(&t.net, data.view()).unwrap();
        }
        let mut plain = Vec::new();
        for _ in 0..21 {
            let start = Instant::now();
            trainer.step(&t.net, data.view()).unwrap();
            plain.push(ms(start.elapsed()));
        }
        let mut profiles = Vec::new();
        for _ in 0..11 {
            trainer.profile_next_step();
            trainer.step(&t.net, data.view()).unwrap();
            let profile = trainer.last_profile();
            profiles.push(profile.expect("the step was asked for a profile").clone());
        }
        let backend = gpu.info().backend;
        println!(
            "{backend}: a step without a profile {:.3} ms",
            median(plain)
        );
        let row = |pick: &dyn Fn(&super::GpuStepProfile) -> Option<Duration>| {
            let times: Option<Vec<f64>> = profiles.iter().map(|p| pick(p).map(ms)).collect();
            times.map(median)
        };
        for i in 0..t.net.len() {
            let name = t.net.layer(i).name();
            if name != "linear" && name != "conv2d" {
                continue;
            }
            let Some(whole) = row(&|p| p.backward[i]) else {
                println!("{backend}: {i} {name}: not timed");
                continue;
            };
            let input = row(&|p| p.backward_input[i]);
            let params = row(&|p| {
                let rest = |x| p.backward[i].map(|w: Duration| w.saturating_sub(x));
                p.backward_input[i].map_or(p.backward[i], rest)
            });
            println!(
                "{backend}: {i} {name}: backward {whole:.3} ms, parameter gradients {:.3} ms, \
                 input gradient {}",
                params.unwrap_or(whole),
                input.map_or("none".to_string(), |v| format!("{v:.3} ms"))
            );
        }
    });
}

/// A step with `Loss::Nll` whose labels hold one equal to the output width is
/// `Error::InvalidData` naming the row, before anything runs: the parameters stay as they
/// were, and the trainer has taken no step.
#[test]
fn an_nll_label_past_the_outputs_is_refused() {
    let t = twin(
        |p, s| softmax_mlp(p, s, true),
        |p, s| softmax_mlp(p, s, true),
        190,
    );
    let mut data = Data::new(190, 8, 12, 4, (-2.0, 2.0)).for_loss(Loss::Nll, 4, 0);
    data.labels[5] = 4;
    for_each_gpu(|gpu| {
        let sgd = OptimizerConfig::Sgd {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut trainer = GpuTrainer::new(&t.net, &t.params, &sgd, Loss::Nll, 8, gpu).unwrap();
        let error = trainer.step(&t.net, data.view()).unwrap_err();
        assert!(
            matches!(error, cavitation_core::Error::InvalidData(_)),
            "{error}"
        );
        assert!(
            error.to_string().contains("the label of row 5 is 4"),
            "{error}"
        );
        assert_eq!(trainer.steps(), 0);
        let mut after = t.params.clone();
        trainer.sync_to(&mut after);
        for (name, id) in t.net.state_names() {
            let bits = |p: &ParamStore| -> Vec<u32> {
                p.value(id).data().iter().map(|v| v.to_bits()).collect()
            };
            assert_eq!(bits(&after), bits(&t.params), "{name}");
        }
    });
}
