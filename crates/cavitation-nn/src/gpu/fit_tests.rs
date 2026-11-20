//! [`fit`] with [`Device::Gpu`]: the run is the [`GpuTrainer`]'s own steps (the same batches,
//! rates, masks and short last batch, bit for bit against a loop written by hand); the
//! parameters reach the host before `on_epoch`, when early stopping keeps a best epoch,
//! before an error of a step or of an epoch's end, and at the end; the run tracks the CPU's `fit` within 2⁻¹⁰; and what
//! the GPU does not run is refused before any training. [`Device::Auto`] trains on the GPU
//! it opens, to the bits of `Device::Gpu` on that GPU, and on the CPU, to the bits of
//! `Device::Cpu`, what the GPU does not run.
//!
//! Without a GPU every test of `Device::Gpu` says so and checks nothing;
//! `CAVITATION_GPU_REQUIRED=1` makes that a failure. The tests of `Device::Auto` open the GPU
//! with `Gpu::open`, which that variable does not govern: without a GPU they check that the
//! run trains on the CPU.

use std::sync::{Arc, Mutex};

use cavitation_core::{Conv2dGeom, Error, ParamStore, Pool2dGeom, Rng, Tensor};

use super::GpuTrainer;
use super::forward_tests::{activation_mlp, mlp, more_activations, softmax_mlp};
use super::health_tests::for_each_gpu;
use crate::{
    Batch, BatchNorm2d, Batcher, Conv2d, Dataset, Device, DeviceKind, Dropout, EarlyStopping,
    Evaluation, FitConfig, Flatten, Health, Hooks, Linear, Loss, LrSchedule, MaxPool2d, Monitor,
    OnNonFinite, OptimizerConfig, Relu, Sequential, StepControl, Targets, fit, fit_controlled,
    fit_with_hooks,
};

/// A CNN on 2 × 8 × 8 images: a convolution, batch norm with running statistics, ReLU, max
/// pooling, dropout, flatten and a `Linear` to 5 classes.
fn cnn(params: &mut ParamStore, seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::with_options(params, 4, (8, 8), 1e-5, None, true, true).unwrap())
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 5, &mut rng))
}

/// `config` on `gpu`.
fn on<'a>(gpu: &'a cavitation_gpu::Gpu, config: FitConfig<'a>) -> FitConfig<'a> {
    FitConfig {
        device: Device::Gpu(gpu),
        ..config
    }
}

/// `rows` rows of `width` values in `[-1, 1)` from `seed`, with labels below `classes`.
fn classes(seed: u64, rows: usize, width: usize, classes: usize) -> Dataset {
    let mut rng = Rng::seed(seed);
    let x: Vec<f32> = (0..rows * width)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let labels = (0..rows)
        .map(|_| rng.range(0.0, classes as f64) as usize)
        .collect();
    Dataset {
        x: Tensor::from_vec(x, &[rows, width]).unwrap(),
        targets: Targets::Classes(labels),
    }
}

/// Every parameter and buffer value of `net` in `params`, by its bits, in the model's order.
fn bits(net: &Sequential, params: &ParamStore) -> Vec<u32> {
    net.state_names()
        .iter()
        .flat_map(|(_, id)| params.value(*id).data().iter().map(|v| v.to_bits()))
        .collect()
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

const WARMUP_COSINE: LrSchedule = LrSchedule::WarmupCosine {
    warmup_steps: 3,
    min_lr: 1e-4,
};

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
        schedule: WARMUP_COSINE,
        health: Health::quiet(),
        ..FitConfig::default()
    }
}

/// `fit` on the GPU is the loop a caller would write with a `GpuTrainer`: the batcher's
/// batches in its order, the schedule's rate before each step, the configured dropout seed,
/// the short last batch, the end of each epoch, the validation set through the trainer's
/// model and the parameters copied back. Each epoch's mean loss, validation and parameters
/// (as `on_epoch` sees them) and the final parameters are the hand-written loop's, bit for
/// bit, on a CNN with batch norm's running statistics and dropout.
#[test]
fn fit_runs_the_trainers_steps() {
    let mut start = ParamStore::new();
    let net = cnn(&mut start, 31);
    let train = classes(32, 22, 2 * 64, 5);
    let held_out = classes(33, 9, 2 * 64, 5);
    for_each_gpu(|gpu| {
        // By hand.
        let config = settings();
        let mut by_hand = start.clone();
        let mut trainer =
            GpuTrainer::new(&net, &by_hand, &config.optimizer, config.loss, 6, gpu).unwrap();
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

        // Through `fit`.
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
            device: Device::Gpu(gpu),
            ..settings()
        };
        let history = fit(&net, &mut params, &train, config).unwrap();
        let got = seen.lock().unwrap();
        assert_eq!(got.len(), 3);
        for (e, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.0, w.0, "epoch {}: the mean loss", e + 1);
            assert_eq!(g.1, w.1, "epoch {}: the validation", e + 1);
            assert!(g.2 == w.2, "epoch {}: the parameters on_epoch saw", e + 1);
        }
        assert!(history.epochs.iter().all(|r| r.rows == 22));
        assert!(
            bits(&net, &params) == want[2].2,
            "the final parameters are the last epoch's"
        );
        assert_ne!(want[0].2, want[2].2, "the run trained");

        // With no callback the parameters are copied back at the end, to the same bits.
        let mut quiet = start.clone();
        let config = FitConfig {
            device: Device::Gpu(gpu),
            ..settings()
        };
        fit(&net, &mut quiet, &train, config).unwrap();
        assert!(bits(&net, &quiet) == want[2].2, "without a callback");
    });
}

/// The GPU run tracks the CPU run of the same `FitConfig`: each epoch's mean loss and
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
        // A threshold no epoch meets, so the rate halves after every second epoch.
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
    for_each_gpu(|gpu| {
        let mut params = start.clone();
        let on_gpu = fit(&net, &mut params, &train, on(gpu, config())).unwrap();
        assert_eq!(on_gpu.epochs.len(), 6);
        let mut worst = 0.0f64;
        for (g, c) in on_gpu.epochs.iter().zip(&on_cpu.epochs) {
            assert_eq!((g.epoch, g.rows), (c.epoch, c.rows));
            let (gv, cv) = (g.validation.unwrap().loss, c.validation.unwrap().loss);
            for (got, want, what) in [
                (g.mean_loss, c.mean_loss, "mean loss"),
                (gv, cv, "validation loss"),
            ] {
                let off = (got - want).abs() / want.abs();
                assert!(
                    off <= 1.0 / 1024.0,
                    "epoch {}: {what} {got}, the CPU's {want}",
                    g.epoch
                );
                worst = worst.max(off);
            }
        }
        println!("6 epochs: worst loss off by {worst:.3e} of the CPU's");
        for (name, id) in net.state_names() {
            let (a, b) = (params.value(id).data(), cpu.value(id).data());
            let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let off = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                off <= scale / 1024.0 + 1.0 / 65536.0,
                "{name} off by {off:e} of {scale:e}"
            );
        }
    });
}

/// Each activation beyond ReLU, tanh, the sigmoid and GELU, twice in an MLP: a run on
/// the GPU tracks the CPU run of the same `FitConfig`, each epoch's mean loss within 2⁻¹⁰ of
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
        for_each_gpu(|gpu| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(gpu, config())).unwrap();
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

/// Softmax and log-softmax after each `Linear` of an MLP: a run on the GPU tracks the CPU run of
/// the same `FitConfig`, each epoch's mean loss and validation loss (through the trainer's
/// model) within 2⁻¹⁰ of the CPU's and the final parameters within 2⁻¹⁰ of each tensor's
/// largest value plus 2⁻¹⁶, over 3 epochs of 50 rows at batch 16 (a last batch of 2) with Adam.
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
        for_each_gpu(|gpu| {
            let mut params = start.clone();
            let on_device = fit(&net, &mut params, &train, on(gpu, config())).unwrap();
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

/// Early stopping on the GPU ends the run with the best epoch's parameters, copied back when
/// that epoch was the best so far: with a rule only the first epoch can meet, a run of 5
/// epochs stops after 2 and returns the parameters `on_epoch` saw after epoch 1, with or
/// without the callback; without `restore_best` it returns epoch 2's.
#[test]
fn early_stopping_restores_the_best_epoch() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 51);
    let train = classes(52, 40, 12, 4);
    for_each_gpu(|gpu| {
        let config = |restore_best| FitConfig {
            epochs: 5,
            batch: 16,
            health: Health::quiet(),
            // No later epoch improves on the first by 1e9, so the second is the last.
            early_stopping: Some(EarlyStopping {
                min_delta: 1e9,
                restore_best,
                ..EarlyStopping::new(Monitor::TrainingLoss, 1)
            }),
            device: Device::Gpu(gpu),
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

/// 12 rows of 2 values with value targets; rows 4 to 7 (the second batch of 4) hold 1e30,
/// finite values whose squared error overflows.
fn overflowing() -> Dataset {
    let mut x = vec![0.5f32; 24];
    x[8..16].fill(1e30);
    Dataset {
        x: Tensor::from_vec(x, &[12, 2]).unwrap(),
        targets: Targets::Values(Tensor::full(&[12, 1], 1.0)),
    }
}

/// A step whose loss is infinite ends the GPU run under `Stop` with the CPU's error, and
/// `params` holds the parameters before that step (the first step's, bit for bit); under
/// `Skip` the run goes on without the batch, which the epoch's rows leave out.
#[test]
fn a_non_finite_step_stops_or_is_left_out() {
    let mut start = ParamStore::new();
    let net = Sequential::new().push(Linear::new(&mut start, 2, 1, &mut Rng::seed(61)));
    let data = overflowing();
    let sgd = OptimizerConfig::Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let config = |on_non_finite| FitConfig {
        epochs: 2,
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
    let mut cpu = start.clone();
    let cpu_error = fit(&net, &mut cpu, &data, config(OnNonFinite::Stop))
        .unwrap_err()
        .to_string();
    for_each_gpu(|gpu| {
        let mut params = start.clone();
        let error = fit(&net, &mut params, &data, on(gpu, config(OnNonFinite::Stop))).unwrap_err();
        assert!(matches!(error, Error::NonFinite { step: 2, .. }), "{error}");
        assert_eq!(error.to_string(), cpu_error);
        let mut one_step = start.clone();
        let mut trainer = GpuTrainer::new(&net, &one_step, &sgd, Loss::Mse, 4, gpu).unwrap();
        trainer.set_health(Health::quiet()).unwrap();
        let mut batch = Batch::new(&data, 4);
        let view = Batcher::new(12, 4, None).gather(&data, 0..4, &mut batch);
        trainer.step(&net, view).unwrap();
        trainer.sync_to(&mut one_step);
        assert_eq!(bits(&net, &params), bits(&net, &one_step));
        assert_ne!(bits(&net, &params), bits(&net, &start));

        let mut params = start.clone();
        let history = fit(&net, &mut params, &data, on(gpu, config(OnNonFinite::Skip))).unwrap();
        assert!(history.epochs.iter().all(|e| e.rows == 8), "{history:?}");
        assert!(history.epochs.iter().all(|e| e.mean_loss.is_finite()));
        assert!(params.value(net.state_names()[0].1).data()[0].is_finite());
    });
}

/// 4 rows of 2 inputs of 1e18 with value targets, one batch: the first step's loss and
/// gradient are finite and its update so large that every later loss is infinite.
fn exploding() -> Dataset {
    Dataset {
        x: Tensor::full(&[4, 2], 1e18),
        targets: Targets::Values(Tensor::full(&[4, 1], 1.0)),
    }
}

/// An epoch in which every step is left out under `Skip` ends the GPU run with the CPU's
/// error, from the end of that epoch's health checks, and `params` holds the parameters the
/// epoch before left (one step's, bit for bit), copied back before the error is returned.
#[test]
fn an_epoch_of_skipped_steps_ends_the_run() {
    let mut start = ParamStore::new();
    let net = Sequential::new().push(Linear::new(&mut start, 2, 1, &mut Rng::seed(81)));
    let data = exploding();
    let sgd = OptimizerConfig::Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let config = || FitConfig {
        epochs: 3,
        batch: 4,
        optimizer: sgd,
        loss: Loss::Mse,
        shuffle_seed: None,
        health: Health {
            on_non_finite: OnNonFinite::Skip,
            ..Health::quiet()
        },
        ..FitConfig::default()
    };
    let mut cpu = start.clone();
    let cpu_error = fit(&net, &mut cpu, &data, config()).unwrap_err();
    assert!(matches!(cpu_error, Error::NonFinite { .. }), "{cpu_error}");
    for_each_gpu(|gpu| {
        let mut params = start.clone();
        let error = fit(&net, &mut params, &data, on(gpu, config())).unwrap_err();
        assert_eq!(error.to_string(), cpu_error.to_string());
        let mut one_step = start.clone();
        let mut trainer = GpuTrainer::new(&net, &one_step, &sgd, Loss::Mse, 4, gpu).unwrap();
        trainer.set_health(Health::quiet()).unwrap();
        let mut batch = Batch::new(&data, 4);
        let view = Batcher::new(4, 4, None).gather(&data, 0..4, &mut batch);
        assert!(trainer.step(&net, view).unwrap().is_finite());
        trainer.sync_to(&mut one_step);
        assert_eq!(bits(&net, &params), bits(&net, &one_step));
        assert_ne!(bits(&net, &params), bits(&net, &start));
    });
}

/// What `on_epoch` saw after an epoch: the validation loss, and the parameters by their bits.
type Seen = (f64, Vec<u32>);

/// Early stopping on a validation measure, which the GPU model measures: over 6 epochs with
/// `restore_best` the run ends with the parameters `on_epoch` saw at `History::best_epoch`,
/// with or without the callback, when several epochs in turn were the best so far (each
/// copied back in its turn) and when the best is the last epoch.
#[test]
fn early_stopping_keeps_the_best_validation_epoch() {
    let mut start = ParamStore::new();
    let net = mlp::<f32>(&mut start, 91);
    let train = classes(92, 48, 12, 4);
    let held_out = classes(93, 24, 12, 4);
    for_each_gpu(|gpu| {
        // At 0.007 the validation loss falls for some epochs and then rises; at 1e-4 it
        // falls through the last epoch.
        for (lr, last_is_best) in [(0.007, false), (1e-4, true)] {
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
                device: Device::Gpu(gpu),
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
            // The best epoch is the first with the lowest validation loss.
            let lowest = seen.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
            let best = 1 + seen.iter().position(|s| s.0 == lowest).unwrap();
            assert_eq!(history.best_epoch, Some(best), "lr {lr}");
            assert_eq!(best == 6, last_is_best, "lr {lr}: the best epoch is {best}");
            let improved = (1..6).filter(|&e| seen[e].0 < seen[e - 1].0).count();
            assert!(
                improved >= 2,
                "lr {lr}: {improved} epochs improved after the first"
            );
            assert!(bits(&net, &params) == seen[best - 1].1, "lr {lr}");

            let mut no_callback = start.clone();
            let again = fit(&net, &mut no_callback, &train, config()).unwrap();
            assert_eq!(again.best_epoch, Some(best), "lr {lr}");
            assert!(
                bits(&net, &no_callback) == seen[best - 1].1,
                "lr {lr}: without a callback"
            );
        }
    });
}

/// A control that does nothing.
struct NoControl;
impl StepControl for NoControl {}

/// What the GPU does not run is refused with `Error::Unsupported` before any training, with
/// the parameters untouched: `fit_controlled`, a registered hook, an optimiser or loss
/// without a GPU kernel, and `OnNonFinite::Warn`. A validation set whose last batch a
/// batch-norm layer without running statistics could not normalise is `InvalidData`.
#[test]
fn what_the_gpu_does_not_run_is_refused() {
    let mut start = ParamStore::new();
    let mut net = mlp::<f32>(&mut start, 71);
    let train = classes(72, 16, 12, 4);
    let ran = for_each_gpu(|gpu| {
        let config = || FitConfig {
            epochs: 1,
            batch: 8,
            health: Health::quiet(),
            device: Device::Gpu(gpu),
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
        refused(r, "RMSprop", &params);
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
        let values = Dataset {
            x: train.x.clone(),
            targets: Targets::Values(Tensor::zeros(&[16, 4])),
        };
        let r = fit(
            &net,
            &mut params,
            &values,
            FitConfig {
                loss: Loss::L1,
                ..config()
            },
        );
        refused(r, "Loss::L1", &params);
        // No hook: `fit_with_hooks` trains.
        let r = fit_with_hooks(&net, &mut params, &train, config(), &mut Hooks::new());
        assert_eq!(r.unwrap().epochs.len(), 1);
    });
    // `fit_controlled` borrows the model mutably, so it runs apart from the others.
    if ran > 0 {
        for_each_gpu(|gpu| {
            let mut params = start.clone();
            let config = FitConfig {
                epochs: 1,
                batch: 8,
                device: Device::Gpu(gpu),
                ..FitConfig::default()
            };
            let before = bits(&net, &params);
            let error = fit_controlled(
                &mut net,
                &mut params,
                &train,
                config,
                &mut Hooks::new(),
                &mut NoControl,
            )
            .unwrap_err();
            assert!(matches!(error, Error::Unsupported(_)), "{error}");
            assert!(error.to_string().contains("fit_controlled"));
            assert_eq!(bits(&net, &params), before);
        });
    }

    // Batch norm without running statistics normalises with the batch's own, so a validation
    // set of 10 rows at batch 4 would end on a short batch.
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
    for_each_gpu(|gpu| {
        let mut params = start.clone();
        let config = FitConfig {
            epochs: 1,
            batch: 4,
            validation: Some(classes(75, 10, 2 * 64, 5)),
            device: Device::Gpu(gpu),
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

/// The default device is the CPU, and a `FitConfig` that names a GPU still moves to another
/// thread.
#[test]
fn the_default_device_is_the_cpu() {
    assert!(matches!(FitConfig::default().device, Device::Cpu));
    assert!(matches!(Device::default(), Device::Cpu));
    fn is_send<T: Send>(_: &T) {}
    is_send(&FitConfig::default());
    for_each_gpu(|gpu| {
        let config = FitConfig {
            device: Device::Gpu(gpu),
            ..FitConfig::default()
        };
        assert!(format!("{config:?}").contains("device: Gpu("));
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

/// Whether CUDA opens on a build with the `cuda` feature, where `Device::Auto` picks it
/// before the wgpu GPU; says so when it does.
fn cuda_picked_first() -> bool {
    #[cfg(feature = "cuda")]
    if cavitation_cuda::Cuda::open().is_ok() {
        println!("CUDA opens: Device::Auto picks it before the wgpu GPU, so this checks nothing");
        return true;
    }
    false
}

/// What a run left: the device it trained on, the parameters and each epoch's mean loss, by
/// their bits.
type Outcome = (DeviceKind, Vec<u32>, Vec<u64>);

/// The run of [`settings`] on the CNN and `data`, on `device`.
fn cnn_run(device: Device<'_>, data: &Dataset) -> Outcome {
    let mut params = ParamStore::new();
    let net = cnn(&mut params, 5);
    let config = FitConfig {
        device,
        ..settings()
    };
    let history = fit(&net, &mut params, data, config).unwrap();
    let losses = history.epochs.iter().map(|e| e.mean_loss.to_bits());
    (history.device, bits(&net, &params), losses.collect())
}

/// `Device::Auto` trains a model the GPU runs on the GPU `Gpu::open` picks on this
/// platform's backends, to the bits of `Device::Gpu` on that GPU, and says so in the
/// history; on a machine without a GPU it trains on the CPU, to the bits of `Device::Cpu`.
#[test]
fn auto_trains_on_the_gpu_it_opens() {
    use cavitation_gpu::{Backend, Gpu};

    if !auto_is_unrestricted() || cuda_picked_first() {
        return;
    }
    let data = classes(31, 22, 2 * 64, 5);
    let auto = cnn_run(Device::Auto, &data);
    match Gpu::open(Backend::on_this_platform()) {
        Ok(gpu) => {
            assert_eq!(auto.0, DeviceKind::Gpu);
            assert_eq!(auto, cnn_run(Device::Gpu(&gpu), &data));
        }
        Err(why) => {
            println!("no GPU ({why}): Device::Auto trains on the CPU");
            assert_eq!(auto, cnn_run(Device::Cpu, &data));
            assert_eq!(auto.0, DeviceKind::Cpu);
        }
    }
}

/// `Device::Auto` leaves to the CPU what the GPU refuses under `Device::Gpu` (an optimiser
/// without a GPU kernel, `OnNonFinite::Warn`, a registered hook and `fit_controlled`): each
/// run trains, to the bits of `Device::Cpu`, and its history says the CPU.
#[test]
fn auto_leaves_to_the_cpu_what_the_gpu_does_not_run() {
    if !auto_is_unrestricted() || cuda_picked_first() {
        return;
    }
    let mut start = ParamStore::new();
    let mut net = mlp::<f32>(&mut start, 71);
    let train = classes(72, 16, 12, 4);
    let rms = OptimizerConfig::RmsProp {
        lr: 1e-3,
        alpha: 0.99,
        momentum: 0.0,
        weight_decay: 0.0,
        centered: false,
    };
    let gpu_opens = cavitation_gpu::Gpu::open(cavitation_gpu::Backend::on_this_platform()).is_ok();
    // The settings of case `case` on `device`.
    let config = |case: &str, device: Device<'static>| {
        let mut config = FitConfig {
            epochs: 2,
            batch: 8,
            health: Health::quiet(),
            device,
            ..FitConfig::default()
        };
        match case {
            "RMSprop" => config.optimizer = rms,
            "OnNonFinite::Warn" => config.health.on_non_finite = OnNonFinite::Warn,
            _ => {}
        }
        config
    };
    for case in ["RMSprop", "OnNonFinite::Warn", "a hook", "a control"] {
        let mut outcomes = Vec::new();
        for device in [Device::Auto, Device::Cpu] {
            let mut params = start.clone();
            let mut hooks = Hooks::new();
            if case == "a hook" {
                hooks.on_forward(0, |_| {});
            }
            let config = config(case, device);
            let history = if case == "a control" {
                let mut control = NoControl;
                fit_controlled(
                    &mut net,
                    &mut params,
                    &train,
                    config,
                    &mut hooks,
                    &mut control,
                )
            } else {
                fit_with_hooks(&net, &mut params, &train, config, &mut hooks)
            }
            .unwrap_or_else(|e| panic!("{case}: {e}"));
            drop(hooks);
            let losses: Vec<u64> = (history.epochs.iter())
                .map(|e| e.mean_loss.to_bits())
                .collect();
            assert_eq!(history.device, DeviceKind::Cpu, "{case}");
            // The history says why a GPU that opened was left: the reason `Device::Gpu`
            // gives. A hook or a control keeps the run on the CPU without opening one.
            let left = matches!(device, Device::Auto)
                && gpu_opens
                && matches!(case, "RMSprop" | "OnNonFinite::Warn");
            assert_eq!(history.device_note.is_some(), left, "{case}");
            if let Some(note) = &history.device_note {
                assert!(note.contains("GpuTrainer"), "{case}: {note}");
            }
            outcomes.push((bits(&net, &params), losses));
        }
        assert_eq!(outcomes[0], outcomes[1], "{case}");
    }
}

/// The epoch time of a small network on each device: the digits network's shape
/// (64 → 256 → 128 → 10, 50,826 parameters) on 1,152 random rows at batch 32 with Adam, 12
/// epochs on `Device::Cpu` and on each GPU, the median of the last 10. A measurement,
/// printed, not asserted. Run with `--ignored`.
#[test]
#[ignore = "a measurement: run with --ignored"]
fn epoch_time_of_a_small_network_by_device() {
    let data = classes(91, 1152, 64, 10);
    let run = |device: Device<'_>| {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(92);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 64, 256, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 256, 128, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 128, 10, &mut rng));
        let config = FitConfig {
            epochs: 12,
            batch: 32,
            health: Health::quiet(),
            device,
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        let mut seconds: Vec<f64> = history.epochs[2..].iter().map(|e| e.seconds).collect();
        seconds.sort_by(f64::total_cmp);
        1e3 * seconds[seconds.len() / 2]
    };
    println!("cpu: an epoch of 36 steps {:.2} ms", run(Device::Cpu));
    for_each_gpu(|gpu| {
        println!(
            "{}: an epoch of 36 steps {:.2} ms",
            gpu.info().backend,
            run(Device::Gpu(gpu))
        );
    });
}

/// With `CAVITATION_DEVICE` asking for a GPU, `Device::Auto` does not fall back: a run the
/// GPU does not train is the error `Device::Gpu` returns, with the parameters untouched, and
/// one it trains is `Device::Gpu`'s run. Asking for the CPU trains there with a GPU present.
#[test]
fn the_variable_keeps_auto_on_the_kind_it_names() {
    use crate::train::{Choice, fit_choosing};
    use cavitation_gpu::{Backend, Gpu};

    let Ok(gpu) = Gpu::open(Backend::on_this_platform()) else {
        println!("no GPU: nothing to check");
        return;
    };
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
    let only_gpu = Choice::Only(DeviceKind::Gpu);

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
    let error = fit_choosing(&net, &mut params, &train, rms, only_gpu).unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    assert_eq!(bits(&net, &params), bits(&net, &start));

    let mut asked = start.clone();
    let history = fit_choosing(&net, &mut asked, &train, config(Device::Auto), only_gpu).unwrap();
    assert_eq!(
        (history.device, &history.device_note),
        (DeviceKind::Gpu, &None)
    );
    let mut named = start.clone();
    let on_gpu = FitConfig {
        device: Device::Gpu(&gpu),
        ..config(Device::Cpu)
    };
    fit(&net, &mut named, &train, on_gpu).unwrap();
    assert_eq!(bits(&net, &asked), bits(&net, &named));

    let mut on_cpu = start.clone();
    let only_cpu = Choice::Only(DeviceKind::Cpu);
    let history = fit_choosing(&net, &mut on_cpu, &train, config(Device::Auto), only_cpu).unwrap();
    assert_eq!(
        (history.device, &history.device_note),
        (DeviceKind::Cpu, &None)
    );
    let mut cpu = start.clone();
    fit(&net, &mut cpu, &train, config(Device::Cpu)).unwrap();
    assert_eq!(bits(&net, &on_cpu), bits(&net, &cpu));
}
