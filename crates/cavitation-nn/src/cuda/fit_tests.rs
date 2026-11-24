//! [`fit`] with [`Device::Cuda`]: the run is the [`CudaTrainer`]'s own steps (the same
//! batches, rates, masks and short last batch, bit for bit against a loop written by hand);
//! the parameters reach the host before `on_epoch`, when early stopping keeps a best epoch,
//! before an error of a step or of an epoch's end, and at the end; the run tracks the CPU's
//! `fit` within 2⁻¹⁰, with every loss CUDA trains; and what CUDA does not run, or the CPU's
//! tape would panic on, is refused before any training.
//! [`Device::Auto`] trains on CUDA when it opens, to the bits of `Device::Cuda`, and leaves
//! what CUDA refuses to the next device; `CAVITATION_DEVICE=cuda` keeps it on CUDA.
//!
//! Without CUDA every test of `Device::Cuda` says so and checks nothing;
//! `CAVITATION_CUDA_REQUIRED=1` makes that a failure. The tests of `Device::Auto` open CUDA
//! with `Cuda::open`, which that variable does not govern.

use std::sync::{Arc, Mutex};

use cavitation_core::{Conv2dGeom, Error, Gate, Graph, ParamStore, Pool2dGeom, Rng, Tensor};
use cavitation_cuda::Cuda;

use super::CudaTrainer;
use super::test_util::{
    ADAPTIVE_OUTPUTS, GLU_GATES, activation_mlp, bits, classes, cumulative_cnn, glu_mlp, mlp,
    more_activations, off, pool_cnn, prelu_mlp, softmax_mlp, with_cuda,
};
use crate::train::{Choice, fit_choosing};
use crate::{
    Batch, BatchNorm2d, Batcher, Conv2d, CudaModel, Dataset, Device, DeviceKind, EarlyStopping,
    Evaluation, FitConfig, Flatten, Health, Hooks, Linear, LogSoftmax, Loss, LrSchedule, Monitor,
    OnNonFinite, OptimizerConfig, Sequential, Sigmoid, StepControl, Targets, TargetsView, devices,
    evaluate, fit, fit_controlled, fit_with_hooks,
};

/// `config` on `cuda`.
fn on<'a>(cuda: &'a Cuda, config: FitConfig<'a>) -> FitConfig<'a> {
    FitConfig {
        device: Device::Cuda(cuda),
        ..config
    }
}

/// An evaluation's loss and score by their bits.
fn evaluation_bits(e: &Evaluation) -> (u64, Option<u64>, Option<u64>) {
    (
        e.loss.to_bits(),
        e.accuracy.map(f64::to_bits),
        e.r2.map(f64::to_bits),
    )
}

/// What each epoch of a run left: the mean loss, the validation, and the parameters as the
/// host saw them, all by their bits.
type Epochs = Vec<(u64, (u64, Option<u64>, Option<u64>), Vec<u32>)>;

/// The settings of the run [`fit_runs_the_trainers_steps`] makes both ways: three epochs of
/// 22 rows at batch 6 (so a last batch of 4), shuffled, with momentum, a rate that changes
/// every step and a dropout seed of its own.
fn settings() -> FitConfig<'static> {
    FitConfig {
        epochs: 3,
        batch: 6,
        optimizer: OptimizerConfig::Sgd {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 1e-4,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(7),
        dropout_seed: 11,
        schedule: LrSchedule::WarmupCosine {
            warmup_steps: 3,
            min_lr: 1e-4,
        },
        health: Health::quiet(),
        ..FitConfig::default()
    }
}

/// `fit` on CUDA is the loop a caller would write with a `CudaTrainer`: the batcher's batches
/// in its order, the schedule's rate before each step, the configured dropout seed, the short
/// last batch, the end of each epoch, the validation set through the trainer's model and the
/// parameters copied back. Each epoch's mean loss, validation and parameters (as `on_epoch`
/// sees them) and the final parameters are the hand-written loop's, bit for bit, on a CNN
/// with batch norm's running statistics and dropout; the history says CUDA.
#[test]
fn fit_runs_the_trainers_steps() {
    let mut start = ParamStore::new();
    let net = cumulative_cnn(&mut start, 31);
    let train = classes(32, 22, 2 * 64, 5);
    let held_out = classes(33, 9, 2 * 64, 5);
    with_cuda(|cuda| {
        let config = settings();
        let mut by_hand = start.clone();
        let mut trainer =
            CudaTrainer::new(&net, &by_hand, &config.optimizer, config.loss, 6, cuda).unwrap();
        trainer.set_health(Health::quiet()).unwrap();
        trainer.set_dropout_seed(config.dropout_seed);
        let mut batcher = Batcher::new(train.len(), 6, config.shuffle_seed);
        let mut batch = Batch::new(&train, 6);
        let per_epoch = batcher.batches().count();
        assert_eq!(per_epoch, 4);
        let mut want: Epochs = Vec::new();
        for epoch in 1..=config.epochs {
            batcher.epoch();
            let mut total = 0.0f64;
            for (position, range) in batcher.batches().enumerate() {
                let step = (epoch - 1) * per_epoch + position;
                trainer.set_learning_rate(config.schedule.learning_rate(
                    0.05,
                    step,
                    per_epoch,
                    config.epochs,
                ));
                let rows = range.len();
                let view = batcher.gather(&train, range, &mut batch);
                total += f64::from(trainer.step(&net, view).unwrap()) * rows as f64;
            }
            trainer.end_epoch(epoch).unwrap();
            let validation = trainer
                .model()
                .evaluate(&net, &held_out, config.loss)
                .unwrap();
            trainer.sync_to(&mut by_hand);
            want.push((
                (total / 22.0).to_bits(),
                evaluation_bits(&validation),
                bits(&net, &by_hand),
            ));
        }
        assert_eq!(trainer.steps(), 12);
        drop(trainer);

        let seen: Arc<Mutex<Epochs>> = Arc::default();
        let sink = Arc::clone(&seen);
        let mut params = start.clone();
        let config = FitConfig {
            validation: Some(held_out.clone()),
            on_epoch: Some(Box::new(move |report, net, params| {
                let validation = report.validation.as_ref().expect("a validation set");
                sink.lock().unwrap().push((
                    report.mean_loss.to_bits(),
                    evaluation_bits(validation),
                    bits(net, params),
                ));
            })),
            ..on(cuda, settings())
        };
        let history = fit(&net, &mut params, &train, config).unwrap();
        assert_eq!(history.device, DeviceKind::Cuda);
        let got = seen.lock().unwrap();
        assert_eq!(got.len(), 3);
        for (e, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.0, w.0, "epoch {}: the mean loss", e + 1);
            assert_eq!(g.1, w.1, "epoch {}: the validation", e + 1);
            assert!(g.2 == w.2, "epoch {}: the parameters on_epoch saw", e + 1);
        }
        assert!(history.epochs.iter().all(|r| r.rows == 22));
        assert!(bits(&net, &params) == want[2].2, "the final parameters");
        assert_ne!(want[0].2, want[2].2, "the run trained");

        let mut quiet = start.clone();
        fit(&net, &mut quiet, &train, on(cuda, settings())).unwrap();
        assert!(bits(&net, &quiet) == want[2].2, "without a callback");
    });
}

/// The CUDA run tracks the CPU run of the same `FitConfig`: each epoch's mean loss and
/// validation loss within 2⁻¹⁰ of the CPU's, and the final parameters within 2⁻¹⁰ of each
/// tensor's largest value plus 2⁻¹⁶, on the MLP with Adam over 50 rows at batch 16 (a last
/// batch of 2) with a plateau schedule that lowers the rate during the run.
#[test]
fn fit_tracks_the_cpu_fit() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 41);
    let train = classes(42, 50, 12, 4);
    let held_out = classes(43, 21, 12, 4);
    let config = || FitConfig {
        epochs: 6,
        batch: 16,
        optimizer: OptimizerConfig::Adam {
            lr: 0.01,
            weight_decay: 0.0,
        },
        shuffle_seed: Some(3),
        schedule: LrSchedule::ReduceOnPlateau {
            monitor: Monitor::TrainingLoss,
            factor: 0.5,
            patience: 1,
            threshold: crate::Threshold::Absolute(1e9),
            cooldown: 0,
            min_lr: 0.0,
        },
        health: Health::quiet(),
        validation: Some(held_out.clone()),
        ..FitConfig::default()
    };
    let mut cpu = start.clone();
    let on_cpu = fit(&net, &mut cpu, &train, config()).unwrap();
    with_cuda(|cuda| {
        let mut params = start.clone();
        let on_cuda = fit(&net, &mut params, &train, on(cuda, config())).unwrap();
        assert_eq!(on_cuda.epochs.len(), 6);
        for (g, c) in on_cuda.epochs.iter().zip(&on_cpu.epochs) {
            assert_eq!((g.epoch, g.rows), (c.epoch, c.rows));
            let (gv, cv) = (g.validation.unwrap().loss, c.validation.unwrap().loss);
            for (got, want, what) in [
                (g.mean_loss, c.mean_loss, "mean loss"),
                (gv, cv, "validation loss"),
            ] {
                assert!(
                    (got - want).abs() <= want.abs() / 1024.0,
                    "epoch {}: {what} {got}, the CPU's {want}",
                    g.epoch
                );
            }
        }
        for (name, id) in net.state_names() {
            let (diff, scale) = off(params.value(id).data(), cpu.value(id).data());
            assert!(
                diff <= scale / 1024.0 + 1.0 / 65536.0,
                "{name} off by {diff:e} of {scale:e}"
            );
        }
    });
}

/// Each activation beyond ReLU, tanh, the sigmoid and GELU, twice in an MLP: a run on
/// CUDA tracks the CPU run of the same `FitConfig`, each epoch's mean loss within 2⁻¹⁰ of
/// the CPU's and the final parameters within 2⁻¹⁰ of each tensor's largest value plus 2⁻¹⁶,
/// over 3 epochs of 50 rows at batch 16 (a last batch of 2) with Adam.
#[test]
fn every_activation_tracks_the_cpu_fit() {
    let train = classes(70, 50, 12, 4);
    let config = || FitConfig {
        epochs: 3,
        batch: 16,
        optimizer: OptimizerConfig::Adam {
            lr: 0.01,
            weight_decay: 0.0,
        },
        shuffle_seed: Some(3),
        health: Health::quiet(),
        ..FitConfig::default()
    };
    for (i, kind) in more_activations().into_iter().enumerate() {
        let mut start = ParamStore::new();
        let net = activation_mlp::<f32>(&mut start, 80 + i as u64, kind);
        let mut cpu = start.clone();
        let on_cpu = fit(&net, &mut cpu, &train, config()).unwrap();
        with_cuda(|cuda| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(cuda, config())).unwrap();
            let mut worst = 0.0f64;
            for (g, c) in on_device.epochs.iter().zip(&on_cpu.epochs) {
                let off = (g.mean_loss - c.mean_loss).abs() / c.mean_loss.abs();
                assert!(
                    off <= 1.0 / 1024.0,
                    "{kind:?}: epoch {}: mean loss {}, the CPU's {}",
                    g.epoch,
                    g.mean_loss,
                    c.mean_loss
                );
                worst = worst.max(off);
            }
            let mut worst_param = 0.0f32;
            for (name, id) in net.state_names() {
                let (a, b) = (params.value(id).data(), cpu.value(id).data());
                let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let tolerance = scale / 1024.0 + 1.0 / 65536.0;
                assert!(
                    off <= tolerance,
                    "{kind:?}: {name} off by {off:e} of {scale:e}"
                );
                worst_param = worst_param.max(off / tolerance);
            }
            println!(
                "{kind:?}: worst loss off by {worst:.3e} of the CPU's; parameters {worst_param:.4} of the tolerance"
            );
        });
    }
}

/// Softmax and log-softmax after each `Linear` of an MLP: a run on CUDA tracks the CPU run of
/// the same `FitConfig`, each epoch's mean loss and validation loss (through the trainer's
/// model) within 2⁻¹⁰ of the CPU's and the final parameters within 2⁻¹⁰ of each tensor's
/// largest value plus 2⁻¹⁶, over 3 epochs of 50 rows at batch 16 (a last batch of 2) with
/// Adam.
#[test]
fn softmax_layers_track_the_cpu_fit() {
    let train = classes(72, 50, 12, 4);
    let held_out = classes(73, 20, 12, 4);
    let config = || FitConfig {
        epochs: 3,
        batch: 16,
        optimizer: OptimizerConfig::Adam {
            lr: 0.01,
            weight_decay: 0.0,
        },
        shuffle_seed: Some(3),
        health: Health::quiet(),
        validation: Some(held_out.clone()),
        ..FitConfig::default()
    };
    for (seed, log) in [(92, false), (93, true)] {
        let what = if log { "log-softmax" } else { "softmax" };
        let mut start = ParamStore::new();
        let net = softmax_mlp::<f32>(&mut start, seed, log);
        let mut cpu = start.clone();
        let on_cpu = fit(&net, &mut cpu, &train, config()).unwrap();
        with_cuda(|cuda| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(cuda, config())).unwrap();
            let mut worst = 0.0f64;
            for (g, c) in on_device.epochs.iter().zip(&on_cpu.epochs) {
                let (gv, cv) = (
                    g.validation.as_ref().unwrap(),
                    c.validation.as_ref().unwrap(),
                );
                for (got, want, which) in [
                    (g.mean_loss, c.mean_loss, "mean loss"),
                    (gv.loss, cv.loss, "validation loss"),
                ] {
                    let off = (got - want).abs() / want.abs();
                    assert!(
                        off <= 1.0 / 1024.0,
                        "{what}: epoch {}: {which} {got}, the CPU's {want}",
                        g.epoch
                    );
                    worst = worst.max(off);
                }
            }
            let mut worst_param = 0.0f32;
            for (name, id) in net.state_names() {
                let (a, b) = (params.value(id).data(), cpu.value(id).data());
                let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let tolerance = scale / 1024.0 + 1.0 / 65536.0;
                assert!(
                    off <= tolerance,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                worst_param = worst_param.max(off / tolerance);
            }
            println!(
                "{what}: worst loss off by {worst:.3e} of the CPU's; parameters {worst_param:.4} \
                 of the tolerance"
            );
        });
    }
}

/// A PReLU before each `Linear` of an MLP, with one slope per column and with one shared slope:
/// a run on CUDA tracks the CPU run of the same `FitConfig`, each epoch's mean loss and
/// validation loss (through the trainer's model) within 2⁻¹⁰ of the CPU's and the final
/// parameters, the slopes among them, within 2⁻¹⁰ of each tensor's largest value plus 2⁻¹⁶,
/// over 3 epochs of 50 rows at batch 16 (a last batch of 2) with Adam, and once with AdamW's
/// decoupled weight decay of 0.05; the slopes move.
#[test]
fn prelu_layers_track_the_cpu_fit() {
    let train = classes(72, 50, 12, 4);
    let held_out = classes(73, 20, 12, 4);
    // Adam, or AdamW with the decay when there is one.
    let config = |decay: f64| FitConfig {
        epochs: 3,
        batch: 16,
        optimizer: if decay == 0.0 {
            OptimizerConfig::Adam {
                lr: 0.01,
                weight_decay: 0.0,
            }
        } else {
            OptimizerConfig::AdamW {
                lr: 0.01,
                weight_decay: decay,
            }
        },
        shuffle_seed: Some(3),
        health: Health::quiet(),
        validation: Some(held_out.clone()),
        ..FitConfig::default()
    };
    for (seed, shared, decay) in [(98, false, 0.0), (99, true, 0.0), (100, false, 0.05)] {
        let what = format!(
            "{}{}",
            if shared { "shared PReLU" } else { "PReLU" },
            if decay == 0.0 {
                ""
            } else {
                " with AdamW's decay"
            }
        );
        let mut start = ParamStore::new();
        let net = prelu_mlp::<f32>(&mut start, seed, shared);
        let mut cpu = start.clone();
        let on_cpu = fit(&net, &mut cpu, &train, config(decay)).unwrap();
        with_cuda(|cuda| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(cuda, config(decay))).unwrap();
            let mut worst = 0.0f64;
            for (g, c) in on_device.epochs.iter().zip(&on_cpu.epochs) {
                let (gv, cv) = (
                    g.validation.as_ref().unwrap(),
                    c.validation.as_ref().unwrap(),
                );
                for (got, want, which) in [
                    (g.mean_loss, c.mean_loss, "mean loss"),
                    (gv.loss, cv.loss, "validation loss"),
                ] {
                    let off = (got - want).abs() / want.abs();
                    assert!(
                        off <= 1.0 / 1024.0,
                        "{what}: epoch {}: {which} {got}, the CPU's {want}",
                        g.epoch
                    );
                    worst = worst.max(off);
                }
            }
            let mut worst_param = 0.0f32;
            for (name, id) in net.state_names() {
                let (a, b) = (params.value(id).data(), cpu.value(id).data());
                let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let tolerance = scale / 1024.0 + 1.0 / 65536.0;
                assert!(
                    off <= tolerance,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                worst_param = worst_param.max(off / tolerance);
            }
            let slopes = net.layer(2).as_prelu().unwrap().weight();
            assert_ne!(
                params.value(slopes).data(),
                start.value(slopes).data(),
                "{what}: the slopes did not train"
            );
            println!(
                "{what}: worst loss off by {worst:.3e} of the CPU's; parameters {worst_param:.4} \
                 of the tolerance"
            );
        });
    }
}

/// A GLU of each gate after the first two `Linear`s of an MLP, and with the SiLU gate a GLU as
/// the first layer too: a run on CUDA tracks the CPU run of the same `FitConfig`, each
/// epoch's mean loss and validation loss (through the trainer's model) within 2⁻¹⁰ of the
/// CPU's and the final parameters within 2⁻¹⁰ of each tensor's largest value plus 2⁻¹⁶, over
/// 3 epochs of 50 rows at batch 16 (a last batch of 2) with Adam.
#[test]
fn glu_layers_track_the_cpu_fit() {
    let cases = GLU_GATES
        .into_iter()
        .map(|g| (g, false))
        .chain([(Gate::Silu, true)]);
    for (k, (gate, lead)) in cases.enumerate() {
        let width = if lead { 24 } else { 12 };
        let train = classes(74, 50, width, 4);
        let held_out = classes(75, 20, width, 4);
        let config = || FitConfig {
            epochs: 3,
            batch: 16,
            optimizer: OptimizerConfig::Adam {
                lr: 0.01,
                weight_decay: 0.0,
            },
            shuffle_seed: Some(3),
            health: Health::quiet(),
            validation: Some(held_out.clone()),
            ..FitConfig::default()
        };
        let what = format!("GLU with {gate:?}{}", if lead { ", leading" } else { "" });
        let mut start = ParamStore::new();
        let net = glu_mlp::<f32>(&mut start, 130 + k as u64, gate, lead);
        let mut cpu = start.clone();
        let on_cpu = fit(&net, &mut cpu, &train, config()).unwrap();
        with_cuda(|cuda| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(cuda, config())).unwrap();
            let mut worst = 0.0f64;
            for (g, c) in on_device.epochs.iter().zip(&on_cpu.epochs) {
                let (gv, cv) = (
                    g.validation.as_ref().unwrap(),
                    c.validation.as_ref().unwrap(),
                );
                for (got, want, which) in [
                    (g.mean_loss, c.mean_loss, "mean loss"),
                    (gv.loss, cv.loss, "validation loss"),
                ] {
                    let off = (got - want).abs() / want.abs();
                    assert!(
                        off <= 1.0 / 1024.0,
                        "{what}: epoch {}: {which} {got}, the CPU's {want}",
                        g.epoch
                    );
                    worst = worst.max(off);
                }
            }
            let mut worst_param = 0.0f32;
            for (name, id) in net.state_names() {
                let (a, b) = (params.value(id).data(), cpu.value(id).data());
                let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let tolerance = scale / 1024.0 + 1.0 / 65536.0;
                assert!(
                    off <= tolerance,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                worst_param = worst_param.max(off / tolerance);
            }
            println!(
                "{what}: worst loss off by {worst:.3e} of the CPU's; parameters {worst_param:.4} \
                 of the tolerance"
            );
        });
    }
}

/// `rows` rows of `width` values in `[-1, 1)` from `seed`, with `outputs` target values per
/// row in `[-1, 1)`.
fn values(seed: u64, rows: usize, width: usize, outputs: usize) -> Dataset {
    let mut rng = Rng::seed(seed);
    let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect() };
    let x = draw(rows * width);
    let t = draw(rows * outputs);
    Dataset {
        x: Tensor::from_vec(x, &[rows, width]).unwrap(),
        targets: Targets::Values(Tensor::from_vec(t, &[rows, outputs]).unwrap()),
    }
}

/// `values` with its targets in `[0, 1)` and every fifth rounded to 0 or 1, as binary
/// cross-entropy takes them.
fn binary_values(seed: u64, rows: usize, width: usize, outputs: usize) -> Dataset {
    let mut data = values(seed, rows, width, outputs);
    if let Targets::Values(t) = &mut data.targets {
        for (i, v) in t.data_mut().iter_mut().enumerate() {
            *v = (*v + 1.0) / 2.0;
            if i % 5 == 0 {
                *v = v.round();
            }
        }
    }
    data
}

/// `values` with each row's targets a distribution, as KL divergence takes them: the
/// magnitudes, every third 0, each row divided by its sum.
fn distribution_values(seed: u64, rows: usize, width: usize, outputs: usize) -> Dataset {
    let mut data = values(seed, rows, width, outputs);
    if let Targets::Values(t) = &mut data.targets {
        for (i, v) in t.data_mut().iter_mut().enumerate() {
            *v = if i % 3 == 1 { 0.0 } else { v.abs() };
        }
        for row in t.data_mut().chunks_mut(outputs) {
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                row.iter_mut().for_each(|v| *v /= sum);
            } else {
                row[0] = 1.0;
            }
        }
    }
    data
}

/// The losses besides cross-entropy, each with whether it trains on labels: the negative
/// log-likelihood after an MLP whose every `Linear` ends in a log-softmax, on labels; MSE, L1,
/// Huber (`delta` 0.5) and smooth L1 (`beta` 0.5 and 0) after the test MLP, on targets in
/// `[−1, 1)`; binary cross-entropy after the test MLP with a sigmoid, and on its logits, on
/// targets in `[0, 1)` with every fifth 0 or 1; KL divergence after the test MLP with a
/// log-softmax, on target distributions with every third value 0.
fn other_losses() -> Vec<(Loss, bool)> {
    vec![
        (Loss::Nll, true),
        (Loss::Mse, false),
        (Loss::L1, false),
        (Loss::Huber { delta: 0.5 }, false),
        (Loss::SmoothL1 { beta: 0.5 }, false),
        (Loss::SmoothL1 { beta: 0.0 }, false),
        (Loss::BinaryCrossEntropy, false),
        (Loss::BinaryCrossEntropyWithLogits, false),
        (Loss::KlDiv, false),
    ]
}

/// Each loss of `other_losses`: a run on CUDA tracks the CPU run of the same `FitConfig`,
/// each epoch's mean loss and validation loss (through the trainer's model) within 2⁻¹⁰ of
/// the CPU's and the final parameters within 2⁻¹⁰ of each tensor's largest value plus 2⁻¹⁶,
/// over 3 epochs of 50 rows at batch 16 (a last batch of 2) with Adam; and
/// `CudaModel::evaluate` on the trained parameters measures the held-out loss within 2⁻¹⁰ of
/// the CPU's `evaluate`.
#[test]
fn every_loss_tracks_the_cpu_fit() {
    for (k, (loss, classes_in)) in other_losses().into_iter().enumerate() {
        let seed = 180 + 3 * k as u64;
        let binary = matches!(
            loss,
            Loss::BinaryCrossEntropy | Loss::BinaryCrossEntropyWithLogits
        );
        let (train, held_out) = if classes_in {
            (classes(seed, 50, 12, 4), classes(seed + 1, 20, 12, 4))
        } else if binary {
            (
                binary_values(seed, 50, 12, 4),
                binary_values(seed + 1, 20, 12, 4),
            )
        } else if loss == Loss::KlDiv {
            (
                distribution_values(seed, 50, 12, 4),
                distribution_values(seed + 1, 20, 12, 4),
            )
        } else {
            (values(seed, 50, 12, 4), values(seed + 1, 20, 12, 4))
        };
        let config = || FitConfig {
            epochs: 3,
            batch: 16,
            optimizer: OptimizerConfig::Adam {
                lr: 0.01,
                weight_decay: 0.0,
            },
            loss,
            shuffle_seed: Some(3),
            health: Health::quiet(),
            validation: Some(held_out.clone()),
            ..FitConfig::default()
        };
        let what = format!("{loss:?}");
        let mut start = ParamStore::new();
        let net = if classes_in {
            softmax_mlp::<f32>(&mut start, seed + 2, true)
        } else if loss == Loss::BinaryCrossEntropy {
            mlp::<f32>(&mut start, seed + 2).push(Sigmoid)
        } else if loss == Loss::KlDiv {
            mlp::<f32>(&mut start, seed + 2).push(LogSoftmax)
        } else {
            mlp::<f32>(&mut start, seed + 2)
        };
        let mut cpu = start.clone();
        let on_cpu = fit(&net, &mut cpu, &train, config()).unwrap();
        with_cuda(|cuda| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(cuda, config())).unwrap();
            let mut worst = 0.0f64;
            for (g, c) in on_device.epochs.iter().zip(&on_cpu.epochs) {
                let (gv, cv) = (
                    g.validation.as_ref().unwrap(),
                    c.validation.as_ref().unwrap(),
                );
                for (got, want, which) in [
                    (g.mean_loss, c.mean_loss, "mean loss"),
                    (gv.loss, cv.loss, "validation loss"),
                ] {
                    let off = (got - want).abs() / want.abs();
                    assert!(
                        off <= 1.0 / 1024.0,
                        "{what}: epoch {}: {which} {got}, the CPU's {want}",
                        g.epoch
                    );
                    worst = worst.max(off);
                }
            }
            let mut worst_param = 0.0f32;
            for (name, id) in net.state_names() {
                let (a, b) = (params.value(id).data(), cpu.value(id).data());
                let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = (a.iter().zip(b))
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let tolerance = scale / 1024.0 + 1.0 / 65536.0;
                assert!(
                    off <= tolerance,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                worst_param = worst_param.max(off / tolerance);
            }
            let mut model = CudaModel::compile(&net, &params, 16, cuda).unwrap();
            let got = model.evaluate(&net, &held_out, loss).unwrap();
            let want = evaluate(&net, &params, &held_out, loss, 16).unwrap();
            let evaluated = (got.loss - want.loss).abs() / want.loss.abs();
            assert!(
                evaluated <= 1.0 / 1024.0,
                "{what}: evaluate {got:?} against the CPU's {want:?}"
            );
            println!(
                "{what}: worst loss off by {worst:.3e} of the CPU's; parameters {worst_param:.4} \
                 of the tolerance; evaluate's loss off by {evaluated:.3e}"
            );
        });
    }
}

/// The thresholds the CPU's tape panics on, each with the start of the message it panics
/// with: a Huber `delta` not above 0 and a smooth L1 `beta` below 0, NaN either way and as
/// the `f32` the tape takes (`1e-50` rounds to 0).
fn bad_thresholds() -> [(Loss, &'static str); 6] {
    let huber = "huber: delta must be positive, but it is";
    let smooth = "smooth_l1: beta must not be negative, but it is";
    [
        (Loss::Huber { delta: 0.0 }, huber),
        (Loss::Huber { delta: -1.0 }, huber),
        (Loss::Huber { delta: f64::NAN }, huber),
        (Loss::Huber { delta: 1e-50 }, huber),
        (Loss::SmoothL1 { beta: -0.5 }, smooth),
        (Loss::SmoothL1 { beta: f64::NAN }, smooth),
    ]
}

/// The message of the panic the CPU's tape raises when `loss` is recorded on 8 × 4 outputs.
fn tape_panic(loss: Loss) -> String {
    let panic = std::panic::catch_unwind(|| {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let zeros = Tensor::zeros(&[8, 4]);
        let out = tape.input(zeros.view());
        loss.record(&mut tape, out, TargetsView::Values(zeros.view()));
    })
    .expect_err("the tape panics");
    (panic.downcast_ref::<String>().cloned())
        .or_else(|| panic.downcast_ref::<&str>().map(|m| m.to_string()))
        .unwrap_or_default()
}

/// A threshold of `bad_thresholds`, on which the CPU's tape panics, is `Error::InvalidData`
/// with the tape's message (the value as the `f32` the tape takes) before any training on
/// every device: `fit` on the CPU, on CUDA and with `Device::Auto` (which returns the
/// error itself rather than leaving CUDA for the CPU) gives the same error, and
/// `CudaTrainer::new` names itself; the parameters are untouched. A smooth L1 `beta` of 0 or
/// −0 trains.
#[test]
fn a_bad_threshold_is_refused_on_every_device() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 195);
    let data = values(196, 16, 12, 4);
    let config = |loss| FitConfig {
        epochs: 1,
        batch: 8,
        loss,
        health: Health::quiet(),
        ..FitConfig::default()
    };
    let sgd = OptimizerConfig::Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    for (loss, text) in bad_thresholds() {
        let problem = tape_panic(loss);
        assert!(
            problem.starts_with(text),
            "{loss:?}: the tape's panic {problem}"
        );
        let mut cpu = start.clone();
        let cpu_error = fit(&net, &mut cpu, &data, config(loss)).unwrap_err();
        assert!(
            matches!(cpu_error, Error::InvalidData(_)),
            "{loss:?}: {cpu_error}"
        );
        let message = format!("invalid data: Loss: {problem}");
        assert_eq!(cpu_error.to_string(), message);
        assert_eq!(bits(&net, &cpu), bits(&net, &start), "{loss:?} on the CPU");
        let mut auto = start.clone();
        let auto_config = FitConfig {
            device: Device::Auto,
            ..config(loss)
        };
        let auto_error = fit_choosing(&net, &mut auto, &data, auto_config, Choice::Best);
        assert_eq!(
            auto_error.unwrap_err().to_string(),
            message,
            "{loss:?} on Auto"
        );
        assert_eq!(bits(&net, &auto), bits(&net, &start), "{loss:?} on Auto");
        with_cuda(|cuda| {
            let mut params = start.clone();
            let error = fit(&net, &mut params, &data, on(cuda, config(loss))).unwrap_err();
            assert_eq!(error.to_string(), message, "{loss:?}");
            assert_eq!(bits(&net, &params), bits(&net, &start), "{loss:?}");
            let made = CudaTrainer::new(&net, &start, &sgd, loss, 8, cuda).unwrap_err();
            assert!(matches!(made, Error::InvalidData(_)), "{loss:?}: {made}");
            assert_eq!(
                made.to_string(),
                format!("invalid data: CudaTrainer: {problem}")
            );
        });
    }
    for beta in [0.0, -0.0] {
        with_cuda(|cuda| {
            let mut params = start.clone();
            let loss = Loss::SmoothL1 { beta };
            let run = fit(&net, &mut params, &data, on(cuda, config(loss))).unwrap();
            assert!(run.epochs[0].mean_loss.is_finite(), "beta {beta}");
        });
    }
}

/// A CNN through average pooling and adaptive average pooling to each of `ADAPTIVE_OUTPUTS`:
/// a run on CUDA tracks the CPU run of the same `FitConfig`, each epoch's mean loss and
/// validation loss (through the trainer's model) within 2⁻¹⁰ of the CPU's and the final
/// parameters within 2⁻¹⁰ of each tensor's largest value plus 2⁻¹⁶, over 3 epochs of 50
/// rows at batch 16 (a last batch of 2) with Adam.
#[test]
fn pool_layers_track_the_cpu_fit() {
    let train = classes(76, 50, 98, 5);
    let held_out = classes(77, 20, 98, 5);
    for (k, to) in ADAPTIVE_OUTPUTS.into_iter().enumerate() {
        let config = || FitConfig {
            epochs: 3,
            batch: 16,
            optimizer: OptimizerConfig::Adam {
                lr: 0.01,
                weight_decay: 0.0,
            },
            shuffle_seed: Some(3),
            health: Health::quiet(),
            validation: Some(held_out.clone()),
            ..FitConfig::default()
        };
        let what = format!("adaptive pooling to {to:?}");
        let mut start = ParamStore::new();
        let net = pool_cnn::<f32>(&mut start, 160 + k as u64, to);
        let mut cpu = start.clone();
        let on_cpu = fit(&net, &mut cpu, &train, config()).unwrap();
        with_cuda(|cuda| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(cuda, config())).unwrap();
            let mut worst = 0.0f64;
            for (g, c) in on_device.epochs.iter().zip(&on_cpu.epochs) {
                let (gv, cv) = (
                    g.validation.as_ref().unwrap(),
                    c.validation.as_ref().unwrap(),
                );
                for (got, want, which) in [
                    (g.mean_loss, c.mean_loss, "mean loss"),
                    (gv.loss, cv.loss, "validation loss"),
                ] {
                    let off = (got - want).abs() / want.abs();
                    assert!(
                        off <= 1.0 / 1024.0,
                        "{what}: epoch {}: {which} {got}, the CPU's {want}",
                        g.epoch
                    );
                    worst = worst.max(off);
                }
            }
            let mut worst_param = 0.0f32;
            for (name, id) in net.state_names() {
                let (a, b) = (params.value(id).data(), cpu.value(id).data());
                let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let off = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let tolerance = scale / 1024.0 + 1.0 / 65536.0;
                assert!(
                    off <= tolerance,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                worst_param = worst_param.max(off / tolerance);
            }
            println!(
                "{what}: worst loss off by {worst:.3e} of the CPU's; parameters {worst_param:.4} \
                 of the tolerance"
            );
        });
    }
}

/// Early stopping on CUDA ends the run with the best epoch's parameters: with a rule only the
/// first epoch can meet, a run of 5 epochs stops after 2 and returns the parameters
/// `on_epoch` saw after epoch 1, with or without the callback; without `restore_best` it
/// returns epoch 2's.
#[test]
fn early_stopping_restores_the_best_epoch() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 51);
    let train = classes(52, 40, 12, 4);
    with_cuda(|cuda| {
        let config = |restore_best| FitConfig {
            epochs: 5,
            batch: 16,
            health: Health::quiet(),
            early_stopping: Some(EarlyStopping {
                min_delta: 1e9,
                restore_best,
                ..EarlyStopping::new(Monitor::TrainingLoss, 1)
            }),
            device: Device::Cuda(cuda),
            ..FitConfig::default()
        };
        let seen: Arc<Mutex<Vec<Vec<u32>>>> = Arc::default();
        let sink = Arc::clone(&seen);
        let mut params = start.clone();
        let history = fit(
            &net,
            &mut params,
            &train,
            FitConfig {
                on_epoch: Some(Box::new(move |_, net, params| {
                    sink.lock().unwrap().push(bits(net, params));
                })),
                ..config(true)
            },
        )
        .unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(
            (
                history.epochs.len(),
                history.best_epoch,
                history.stopped_early
            ),
            (2, Some(1), true)
        );
        assert_ne!(seen[0], seen[1], "epoch 2 trained");
        assert!(bits(&net, &params) == seen[0], "epoch 1's parameters");

        let mut no_callback = start.clone();
        fit(&net, &mut no_callback, &train, config(true)).unwrap();
        assert!(bits(&net, &no_callback) == seen[0], "without a callback");

        let mut last = start.clone();
        let history = fit(&net, &mut last, &train, config(false)).unwrap();
        assert_eq!((history.epochs.len(), history.stopped_early), (2, true));
        assert!(bits(&net, &last) == seen[1], "without restore_best");
    });
}

/// A step whose loss is infinite ends the CUDA run under `Stop` with the CPU's error, and
/// `params` holds the parameters before that step (the first step's, bit for bit); under
/// `Skip` the run goes on without the batch, which the epoch's rows leave out. An epoch in
/// which every step is left out ends the run with the CPU's error.
#[test]
fn non_finite_steps_stop_or_are_left_out() {
    let mut start = ParamStore::new();
    let net = Sequential::new().push(Linear::new(&mut start, 2, 1, &mut Rng::seed(61)));
    let mut x = vec![0.5f32; 24];
    x[8..16].fill(1e30);
    let overflowing = Dataset {
        x: Tensor::from_vec(x, &[12, 2]).unwrap(),
        targets: Targets::Values(Tensor::full(&[12, 1], 1.0)),
    };
    let exploding = Dataset {
        x: Tensor::full(&[4, 2], 1e18),
        targets: Targets::Values(Tensor::full(&[4, 1], 1.0)),
    };
    let sgd = OptimizerConfig::Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let config = |on_non_finite, epochs| FitConfig {
        epochs,
        batch: 4,
        optimizer: sgd,
        loss: Loss::Mse,
        shuffle_seed: None,
        health: Health {
            on_non_finite,
            ..Health::quiet()
        },
        ..FitConfig::default()
    };
    let one_step = |cuda: &Cuda, data: &Dataset| {
        let mut params = start.clone();
        let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 4, cuda).unwrap();
        trainer.set_health(Health::quiet()).unwrap();
        let mut batch = Batch::new(data, 4);
        let view = Batcher::new(data.len(), 4, None).gather(data, 0..4, &mut batch);
        trainer.step(&net, view).unwrap();
        trainer.sync_to(&mut params);
        bits(&net, &params)
    };
    let mut cpu = start.clone();
    let stopped = fit(&net, &mut cpu, &overflowing, config(OnNonFinite::Stop, 2)).unwrap_err();
    let mut cpu = start.clone();
    let skipped = fit(&net, &mut cpu, &exploding, config(OnNonFinite::Skip, 3)).unwrap_err();
    assert!(matches!(skipped, Error::NonFinite { .. }), "{skipped}");
    with_cuda(|cuda| {
        let mut params = start.clone();
        let error = fit(
            &net,
            &mut params,
            &overflowing,
            on(cuda, config(OnNonFinite::Stop, 2)),
        )
        .unwrap_err();
        assert!(matches!(error, Error::NonFinite { step: 2, .. }), "{error}");
        assert_eq!(error.to_string(), stopped.to_string());
        assert_eq!(bits(&net, &params), one_step(cuda, &overflowing));

        let mut params = start.clone();
        let history = fit(
            &net,
            &mut params,
            &overflowing,
            on(cuda, config(OnNonFinite::Skip, 2)),
        )
        .unwrap();
        assert!(history.epochs.iter().all(|e| e.rows == 8), "{history:?}");
        assert!(history.epochs.iter().all(|e| e.mean_loss.is_finite()));

        let mut params = start.clone();
        let error = fit(
            &net,
            &mut params,
            &exploding,
            on(cuda, config(OnNonFinite::Skip, 3)),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), skipped.to_string());
        assert_eq!(bits(&net, &params), one_step(cuda, &exploding));
        assert_ne!(bits(&net, &params), bits(&net, &start));
    });
}

/// What the validation loss and parameters were after an epoch.
type Seen = (f64, Vec<u32>);

/// Early stopping on a validation measure, which the card measures: over 6 epochs with
/// `restore_best` the run ends with the parameters `on_epoch` saw at `History::best_epoch`,
/// with or without the callback.
#[test]
fn early_stopping_keeps_the_best_validation_epoch() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 91);
    let train = classes(92, 48, 12, 4);
    let held_out = classes(93, 24, 12, 4);
    with_cuda(|cuda| {
        for lr in [0.007, 1e-4] {
            let config = || FitConfig {
                epochs: 6,
                batch: 16,
                optimizer: OptimizerConfig::Adam {
                    lr,
                    weight_decay: 0.0,
                },
                health: Health::quiet(),
                validation: Some(held_out.clone()),
                early_stopping: Some(EarlyStopping {
                    restore_best: true,
                    ..EarlyStopping::new(Monitor::ValidationLoss, 6)
                }),
                device: Device::Cuda(cuda),
                ..FitConfig::default()
            };
            let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
            let sink = Arc::clone(&seen);
            let mut params = start.clone();
            let history = fit(
                &net,
                &mut params,
                &train,
                FitConfig {
                    on_epoch: Some(Box::new(move |report, net, params| {
                        let loss = report.validation.expect("a validation set").loss;
                        sink.lock().unwrap().push((loss, bits(net, params)));
                    })),
                    ..config()
                },
            )
            .unwrap();
            let seen = seen.lock().unwrap();
            assert_eq!((seen.len(), history.stopped_early), (6, false));
            let lowest = seen.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
            let best = 1 + seen.iter().position(|s| s.0 == lowest).unwrap();
            assert_eq!(history.best_epoch, Some(best), "lr {lr}");
            assert!(bits(&net, &params) == seen[best - 1].1, "lr {lr}");

            let mut no_callback = start.clone();
            let again = fit(&net, &mut no_callback, &train, config()).unwrap();
            assert_eq!(again.best_epoch, Some(best), "lr {lr}");
            assert!(bits(&net, &no_callback) == seen[best - 1].1, "lr {lr}");
        }
    });
}

/// A control that does nothing.
struct NoControl;
impl StepControl for NoControl {}

/// What CUDA does not run is refused with `Error::Unsupported` before any training, with the
/// parameters untouched: `fit_controlled`, a registered hook, an optimiser without a CUDA
/// kernel, and `OnNonFinite::Warn`, as `Device::Gpu` refuses them. A validation set
/// whose last batch a batch-norm layer without running statistics could not normalise is
/// `InvalidData`.
#[test]
fn what_cuda_does_not_run_is_refused() {
    let mut start = ParamStore::new();
    let mut net = mlp::<f32>(&mut start, 71);
    let train = classes(72, 16, 12, 4);
    with_cuda(|cuda| {
        let config = || FitConfig {
            epochs: 1,
            batch: 8,
            health: Health::quiet(),
            device: Device::Cuda(cuda),
            ..FitConfig::default()
        };
        let mut params = start.clone();
        let refused = |result: Result<_, Error>, what: &str, params: &ParamStore| {
            let error = result.expect_err(what);
            assert!(matches!(error, Error::Unsupported(_)), "{what}: {error}");
            assert_eq!(bits(&net, params), bits(&net, &start), "{what}");
            error.to_string()
        };
        let mut hooks = Hooks::new();
        hooks.on_forward(0, |_| {});
        let r = fit_with_hooks(&net, &mut params, &train, config(), &mut hooks);
        assert!(refused(r, "a hook", &params).contains("fit_with_hooks"));
        drop(hooks);
        let rms = OptimizerConfig::RmsProp {
            lr: 1e-3,
            alpha: 0.99,
            momentum: 0.0,
            weight_decay: 0.0,
            centered: false,
        };
        let r = fit(
            &net,
            &mut params,
            &train,
            FitConfig {
                optimizer: rms,
                ..config()
            },
        );
        assert!(refused(r, "RMSprop", &params).contains("CudaTrainer"));
        let warn = Health {
            on_non_finite: OnNonFinite::Warn,
            ..Health::quiet()
        };
        let r = fit(
            &net,
            &mut params,
            &train,
            FitConfig {
                health: warn,
                ..config()
            },
        );
        refused(r, "OnNonFinite::Warn", &params);
        let r = fit_with_hooks(&net, &mut params, &train, config(), &mut Hooks::new());
        assert_eq!(r.unwrap().epochs.len(), 1);
        let before = bits(&net, &start);
        let mut params = start.clone();
        let error = fit_controlled(
            &mut net,
            &mut params,
            &train,
            config(),
            &mut Hooks::new(),
            &mut NoControl,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        assert!(error.to_string().contains("fit_controlled"));
        assert_eq!(bits(&net, &params), before);
    });

    let mut start = ParamStore::new();
    let mut rng = Rng::seed(73);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    let net = Sequential::new()
        .push(Conv2d::new(&mut start, g, &mut rng))
        .push(
            BatchNorm2d::with_options(&mut start, 4, (8, 8), 1e-5, Some(0.1), true, false).unwrap(),
        )
        .push(Flatten)
        .push(Linear::new(&mut start, 256, 5, &mut rng));
    let train = classes(74, 8, 2 * 64, 5);
    with_cuda(|cuda| {
        let mut params = start.clone();
        let config = FitConfig {
            epochs: 1,
            batch: 4,
            validation: Some(classes(75, 10, 2 * 64, 5)),
            device: Device::Cuda(cuda),
            ..FitConfig::default()
        };
        let error = fit(&net, &mut params, &train, config).unwrap_err();
        assert!(
            matches!(&error, Error::InvalidData(msg) if msg.starts_with("validation: ")),
            "{error}"
        );
        assert_eq!(bits(&net, &params), bits(&net, &start));
    });
}

/// Whether `CAVITATION_DEVICE` is unset or empty, so that `Device::Auto` picks by its own
/// rule, as the tests of it assume; says so when it is not.
fn auto_is_unrestricted() -> bool {
    let unrestricted = std::env::var_os("CAVITATION_DEVICE").is_none_or(|v| v.is_empty());
    if !unrestricted {
        println!("CAVITATION_DEVICE is set: the Device::Auto tests check nothing");
    }
    unrestricted
}

/// What a run left: the device it trained on, the parameters and each epoch's mean loss, by
/// their bits.
type Outcome = (DeviceKind, Vec<u32>, Vec<u64>);

/// The run of [`settings`] on the CNN and `data`, on `device`.
fn cnn_run(device: Device<'_>, data: &Dataset) -> Outcome {
    let mut params = ParamStore::new();
    let net = cumulative_cnn(&mut params, 5);
    let config = FitConfig {
        device,
        ..settings()
    };
    let history = fit(&net, &mut params, data, config).unwrap();
    let losses = history.epochs.iter().map(|e| e.mean_loss.to_bits());
    (history.device, bits(&net, &params), losses.collect())
}

/// `Device::Auto` trains on CUDA when it opens, to the bits of `Device::Cuda`, and says so in
/// the history and in `devices`, which lists CUDA first and chooses it.
#[test]
fn auto_trains_on_cuda_when_it_opens() {
    if !auto_is_unrestricted() {
        return;
    }
    let data = classes(31, 22, 2 * 64, 5);
    let auto = cnn_run(Device::Auto, &data);
    let list = devices().unwrap();
    assert_eq!(list[0].kind, DeviceKind::Cuda);
    match Cuda::open() {
        Ok(cuda) => {
            assert_eq!(auto.0, DeviceKind::Cuda);
            assert_eq!(auto, cnn_run(Device::Cuda(&cuda), &data));
            assert!(list[0].available && list[0].chosen, "{}", list[0]);
            assert_eq!(list[0].name, cuda.info().name);
        }
        Err(why) => {
            println!("no CUDA ({why}): Device::Auto trains elsewhere");
            assert_ne!(auto.0, DeviceKind::Cuda);
            assert!(!list[0].available && !list[0].chosen);
            assert_eq!(list[0].note.as_deref(), Some(why.to_string().as_str()));
        }
    }
}

/// A layer of your own with no parameters that passes its input on and declares a dilated
/// average pooling as its inference form, which the CPU's `AvgPool2d` never builds.
struct DilatedAvg;

impl<T: cavitation_core::Real> crate::Layer<T> for DilatedAvg {
    fn name(&self) -> &str {
        "dilated_avg"
    }
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }
    fn forward(
        &self,
        _: &mut cavitation_core::Tape<'_, T>,
        x: cavitation_core::Var,
    ) -> cavitation_core::Var {
        x
    }
    fn parameters(&self) -> Vec<(&'static str, cavitation_core::ParamId)> {
        Vec::new()
    }
    fn inference(&self, _: &cavitation_core::ParamValues<f32>) -> Option<crate::InferenceLayer> {
        let geom = Pool2dGeom::with_options(1, (6, 6), (2, 2), (2, 2), (0, 0), (2, 2), false);
        Some(crate::InferenceLayer::AvgPool2d {
            geom: geom.unwrap(),
            count_include_pad: true,
            divisor_override: None,
        })
    }
}

/// `Device::Auto` with a layer whose inference form is a dilated average pooling, which the
/// card refuses as invalid data: the run leaves CUDA (and, on a build with the `gpu` feature,
/// the wgpu GPU, which refuses it too) and trains on the CPU, to the bits of `Device::Cpu`;
/// when CUDA opened, its history says why a device was left.
#[test]
fn auto_leaves_a_dilated_average_pooling_to_the_cpu() {
    if !auto_is_unrestricted() {
        return;
    }
    let mut start = ParamStore::new();
    let net =
        Sequential::new()
            .push(DilatedAvg)
            .push(Linear::new(&mut start, 36, 4, &mut Rng::seed(73)));
    let train = classes(74, 16, 36, 4);
    let cuda_opens = Cuda::open().is_ok();
    let mut outcomes = Vec::new();
    for device in [Device::Auto, Device::Cpu] {
        let mut params = start.clone();
        let config = FitConfig {
            epochs: 2,
            batch: 8,
            health: Health::quiet(),
            device,
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &train, config).unwrap();
        assert_eq!(history.device, DeviceKind::Cpu);
        let left = matches!(device, Device::Auto) && cuda_opens;
        assert_eq!(history.device_note.is_some(), left);
        if let Some(note) = &history.device_note {
            println!("left CUDA: {note}");
            assert!(
                note.contains("layer 0 (dilated_avg) is an average pooling with dilation (2, 2)"),
                "{note}"
            );
        }
        let losses: Vec<u64> = (history.epochs.iter())
            .map(|e| e.mean_loss.to_bits())
            .collect();
        outcomes.push((bits(&net, &params), losses));
    }
    assert_eq!(outcomes[0], outcomes[1]);
}
/// `Device::Auto` leaves what CUDA refuses (an optimiser without a CUDA kernel,
/// `OnNonFinite::Warn`) to the next device, and its history says why CUDA was left; on a
/// build without the `gpu` feature the next device is the CPU, to the bits of
/// `Device::Cpu`.
#[test]
fn auto_leaves_what_cuda_refuses() {
    if !auto_is_unrestricted() {
        return;
    }
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 71);
    let train = classes(72, 16, 12, 4);
    let cuda_opens = Cuda::open().is_ok();
    for case in ["RMSprop", "OnNonFinite::Warn"] {
        let mut outcomes = Vec::new();
        for device in [Device::Auto, Device::Cpu] {
            let mut config = FitConfig {
                epochs: 2,
                batch: 8,
                health: Health::quiet(),
                device,
                ..FitConfig::default()
            };
            match case {
                "RMSprop" => {
                    config.optimizer = OptimizerConfig::RmsProp {
                        lr: 1e-3,
                        alpha: 0.99,
                        momentum: 0.0,
                        weight_decay: 0.0,
                        centered: false,
                    };
                }
                _ => config.health.on_non_finite = OnNonFinite::Warn,
            }
            let mut params = start.clone();
            let history = fit(&net, &mut params, &train, config).unwrap();
            assert_eq!(history.device, DeviceKind::Cpu, "{case}");
            let left = matches!(device, Device::Auto) && cuda_opens;
            assert_eq!(history.device_note.is_some(), left, "{case}");
            if let Some(note) = &history.device_note {
                assert!(note.contains("CudaTrainer"), "{case}: {note}");
            }
            let losses: Vec<u64> = (history.epochs.iter())
                .map(|e| e.mean_loss.to_bits())
                .collect();
            outcomes.push((bits(&net, &params), losses));
        }
        assert_eq!(outcomes[0], outcomes[1], "{case}");
    }
}

/// With `CAVITATION_DEVICE=cuda`, `Device::Auto` does not fall back: a run CUDA does not
/// train is the error `Device::Cuda` returns, with the parameters untouched, and one it
/// trains is `Device::Cuda`'s run.
#[test]
fn the_variable_keeps_auto_on_cuda() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 71);
    let train = classes(72, 16, 12, 4);
    let config = |device: Device<'static>| FitConfig {
        epochs: 2,
        batch: 8,
        health: Health::quiet(),
        device,
        ..FitConfig::default()
    };
    let only_cuda = Choice::Only(DeviceKind::Cuda);
    let Ok(cuda) = Cuda::open() else {
        let mut params = start.clone();
        let error =
            fit_choosing(&net, &mut params, &train, config(Device::Auto), only_cuda).unwrap_err();
        assert!(
            error.to_string().contains("CAVITATION_DEVICE=cuda"),
            "{error}"
        );
        return;
    };
    let mut params = start.clone();
    let rms = FitConfig {
        optimizer: OptimizerConfig::RmsProp {
            lr: 1e-3,
            alpha: 0.99,
            momentum: 0.0,
            weight_decay: 0.0,
            centered: false,
        },
        ..config(Device::Auto)
    };
    let error = fit_choosing(&net, &mut params, &train, rms, only_cuda).unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    assert_eq!(bits(&net, &params), bits(&net, &start));

    let mut asked = start.clone();
    let history = fit_choosing(&net, &mut asked, &train, config(Device::Auto), only_cuda).unwrap();
    assert_eq!(
        (history.device, &history.device_note),
        (DeviceKind::Cuda, &None)
    );
    let mut named = start.clone();
    let on_cuda = FitConfig {
        device: Device::Cuda(&cuda),
        ..config(Device::Cpu)
    };
    fit(&net, &mut named, &train, on_cuda).unwrap();
    assert_eq!(bits(&net, &asked), bits(&net, &named));
}
