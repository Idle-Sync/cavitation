//! `CudaTrainer`'s health checks against the CPU `Trainer`'s, run side by side from the same
//! start on the same batches: a NaN batch stopped or left out as on the CPU, a gradient that
//! overflows to infinity while the loss stays finite named as on the CPU, a finite gradient
//! whose `f32` sum of squares overflows reported as exploding, an epoch of skipped steps
//! refused as on the CPU, and the warnings of seeded runs (the norm checks on every
//! parameter, and dead ReLU and ReLU6 neurons after a `Linear` and after a convolution)
//! matching the CPU's kind for kind, step for step and name for name, with each norm within
//! 2⁻¹⁰ of the CPU's; and binary cross-entropy's clamped logarithms warned of as on the CPU,
//! with the CPU's count.
//!
//! Without CUDA every test says so and checks nothing; `CAVITATION_CUDA_REQUIRED=1` makes
//! that a failure.

use std::sync::{Arc, Mutex};

use cavitation_core::{Conv2dGeom, Error, ParamStore, Pool2dGeom, Rng, Tensor};
use cavitation_cuda::Cuda;

use super::CudaTrainer;
use super::test_util::{Data, mlp, prelu_mlp, twin, with_cuda};
use crate::{
    Conv2d, Flatten, Health, Init, Linear, LogSoftmax, Loss, MaxPool2d, OnNonFinite,
    OptimizerConfig, Relu, Relu6, Sequential, Sigmoid, Trainer, Warning, WarningKind,
};

/// The warnings a sink has received.
type Seen = Arc<Mutex<Vec<Warning>>>;

/// `health` with a sink that keeps every warning, and the warnings it keeps.
fn collecting(health: Health<'static>) -> (Health<'static>, Seen) {
    let seen: Seen = Arc::default();
    let into = Arc::clone(&seen);
    let health = Health {
        sink: Some(Box::new(move |w: &Warning| {
            into.lock().unwrap().push(w.clone())
        })),
        ..health
    };
    (health, seen)
}

/// What one run of a trainer did: each step's loss (by its bits) or error, the steps taken and
/// left out, each epoch's end, and the warnings.
#[derive(Debug)]
struct Run {
    /// Each step's loss bits, or its error's text.
    steps: Vec<Result<u32, String>>,
    /// The steps counted, and the steps left out.
    counts: (u64, u64),
    /// Each epoch's end: `Ok`, or the error's text.
    epochs: Vec<Result<(), String>>,
    /// The warnings, in order.
    warnings: Vec<Warning>,
}

impl Run {
    /// Each step's error, if it had one.
    fn errors(&self) -> Vec<Option<String>> {
        self.steps
            .iter()
            .map(|s| s.as_ref().err().cloned())
            .collect()
    }
}

/// Steps through `epochs` (a list of batches each) with `step`, ending each epoch with
/// `end`; stops at the first step that fails, as `fit` does.
fn run(
    epochs: &[Vec<&Data>],
    mut step: impl FnMut(&Data) -> Result<f32, Error>,
    mut end: impl FnMut(usize) -> Result<(), Error>,
    counts: impl Fn() -> (u64, u64),
    seen: &Seen,
) -> Run {
    let (mut steps, mut ends) = (Vec::new(), Vec::new());
    'epochs: for (e, batches) in epochs.iter().enumerate() {
        for d in batches {
            let r = step(d);
            let stopped = r.is_err();
            steps.push(r.map(f32::to_bits).map_err(|e| e.to_string()));
            if stopped {
                break 'epochs;
            }
        }
        ends.push(end(e + 1).map_err(|e| e.to_string()));
    }
    Run {
        steps,
        counts: counts(),
        epochs: ends,
        warnings: seen.lock().unwrap().clone(),
    }
}

/// The CPU `Trainer`'s run of `net` from `params`, at the first batch's rows.
fn cpu_run(
    net: &Sequential,
    params: &ParamStore,
    config: &OptimizerConfig,
    loss: Loss,
    health: Health<'static>,
    epochs: &[Vec<&Data>],
) -> Run {
    let (health, seen) = collecting(health);
    let mut params = params.clone();
    let batch = epochs[0][0].rows;
    let mut trainer = Trainer::new(net, config.build(), loss, batch);
    trainer.set_health(health);
    let trainer = std::cell::RefCell::new(trainer);
    run(
        epochs,
        |d| trainer.borrow_mut().step(net, &mut params, d.view()),
        |e| trainer.borrow_mut().end_epoch(e),
        || {
            let t = trainer.borrow();
            (t.steps(), t.skipped_steps())
        },
        &seen,
    )
}

/// The `CudaTrainer`'s run, at the first batch's rows, and its optimiser's update counts
/// after it.
fn cuda_run(
    cuda: &Cuda,
    net: &Sequential,
    params: &ParamStore,
    config: &OptimizerConfig,
    loss: Loss,
    health: Health<'static>,
    epochs: &[Vec<&Data>],
) -> (Run, Vec<u32>) {
    let (health, seen) = collecting(health);
    let batch = epochs[0][0].rows;
    let mut trainer = CudaTrainer::new(net, params, config, loss, batch, cuda).unwrap();
    trainer.set_health(health).unwrap();
    let trainer = std::cell::RefCell::new(trainer);
    let r = run(
        epochs,
        |d| trainer.borrow_mut().step(net, d.view()),
        |e| trainer.borrow_mut().end_epoch(e),
        || {
            let t = trainer.borrow();
            (t.steps(), t.skipped_steps())
        },
        &seen,
    );
    let updates = trainer.borrow_mut().optimizer_state().steps;
    (r, updates)
}

/// Checks that the card's warnings are the CPU's: the same kinds, steps, epochs, locations
/// and layers in the same order, the same dead neurons and inputs, and each norm within 2⁻¹⁰.
fn same_warnings(card: &[Warning], cpu: &[Warning], what: &str) {
    let show = |w: &[Warning]| w.iter().map(|w| format!("{w}\n")).collect::<String>();
    assert_eq!(
        card.len(),
        cpu.len(),
        "{what}: the card sent\n{}the CPU\n{}",
        show(card),
        show(cpu)
    );
    for (g, c) in card.iter().zip(cpu) {
        let key = |w: &Warning| {
            (
                w.kind(),
                w.step(),
                w.epoch(),
                w.location().map(String::from),
                w.layer(),
            )
        };
        assert_eq!(key(g), key(c), "{what}: {g} against {c}");
        match (g, c) {
            (Warning::Exploding { norm: a, .. }, Warning::Exploding { norm: b, .. })
            | (Warning::Vanishing { norm: a, .. }, Warning::Vanishing { norm: b, .. }) => {
                assert!(
                    (a - b).abs() <= b.abs() / 1024.0,
                    "{what}: norm {a:e} against the CPU's {b:e}"
                );
            }
            _ => assert_eq!(g, c, "{what}"),
        }
    }
}

/// How many of `warnings` are of `kind`.
fn count(warnings: &[Warning], kind: WarningKind) -> usize {
    warnings.iter().filter(|w| w.kind() == kind).count()
}

/// Adam, which counts each parameter's updates.
fn adam() -> OptimizerConfig {
    OptimizerConfig::Adam {
        lr: 1e-2,
        weight_decay: 0.0,
    }
}

/// A NaN in a batch's inputs stops training under `Stop` as on the CPU (the same error text,
/// steps counted and warning); under `Skip` the step is left out as on the CPU, with Adam's
/// update counts one lower than the steps.
#[test]
fn a_nan_batch_is_stopped_or_skipped_as_on_the_cpu() {
    let t = twin(mlp::<f64>, mlp::<f32>, 41);
    let good: Vec<Data> = (0..3).map(|s| Data::new(50 + s, 6, 12, 4)).collect();
    let mut bad = Data::new(60, 6, 12, 4);
    bad.x[17] = f32::NAN;
    let epochs = vec![vec![&good[0], &bad, &good[1], &good[2]]];
    for policy in [OnNonFinite::Stop, OnNonFinite::Skip] {
        let health = || Health {
            on_non_finite: policy,
            ..Health::quiet()
        };
        let loss = Loss::CrossEntropy;
        let cpu = cpu_run(&t.net, &t.params, &adam(), loss, health(), &epochs);
        assert_eq!(count(&cpu.warnings, WarningKind::NonFinite), 1, "{cpu:?}");
        with_cuda(|cuda| {
            let (g, updates) = cuda_run(cuda, &t.net, &t.params, &adam(), loss, health(), &epochs);
            let what = format!("{policy:?}");
            assert_eq!(g.counts, cpu.counts, "{what}");
            assert_eq!(g.epochs, cpu.epochs, "{what}");
            assert_eq!(g.errors(), cpu.errors(), "{what}");
            same_warnings(&g.warnings, &cpu.warnings, &what);
            let trained =
                (cpu.counts.0 - cpu.counts.1) as u32 - u32::from(policy == OnNonFinite::Stop);
            assert_eq!(updates, vec![trained; updates.len()], "{what}");
        });
    }
}

/// An epoch whose every step is left out under `Skip` ends with the CPU's error.
#[test]
fn an_epoch_of_skipped_steps_ends_as_on_the_cpu() {
    let t = twin(mlp::<f64>, mlp::<f32>, 43);
    let mut bad = Data::new(61, 6, 12, 4);
    bad.x[0] = f32::INFINITY;
    let epochs = vec![vec![&bad, &bad]];
    let health = || Health {
        on_non_finite: OnNonFinite::Skip,
        ..Health::quiet()
    };
    let loss = Loss::CrossEntropy;
    let cpu = cpu_run(&t.net, &t.params, &adam(), loss, health(), &epochs);
    assert!(cpu.epochs[0].is_err(), "{cpu:?}");
    with_cuda(|cuda| {
        let (g, updates) = cuda_run(cuda, &t.net, &t.params, &adam(), loss, health(), &epochs);
        assert_eq!(g.epochs, cpu.epochs);
        assert_eq!(g.counts, cpu.counts);
        same_warnings(&g.warnings, &cpu.warnings, "skipped epoch");
        assert_eq!(updates, vec![0; updates.len()]);
    });
}

/// A `Linear` 12 → 4 with no weights and biases of ±300, then a sigmoid: every output is
/// exactly 1 or 0 on both devices, so binary cross-entropy clamps a logarithm wherever the
/// target is not the output. Two batches of 6 rows, with targets of 0, ½ and 1 from `seed`.
fn saturated(seed: u64) -> (Sequential, ParamStore, Vec<Data>) {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(seed);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 12, 4, &mut rng))
        .push(Sigmoid);
    let l = net.layer(0).as_linear().unwrap();
    *params.value_mut(l.weight()) = Tensor::zeros(&[4, 12]);
    *params.value_mut(l.bias().unwrap()) =
        Tensor::from_vec(vec![300.0, -300.0, 300.0, -300.0], &[4]).unwrap();
    let batches = (0..2)
        .map(|b| {
            let mut data = Data::new(seed + b, 6, 12, 4);
            let targets = (0..24).map(|_| (rng.range(0.0, 3.0) as usize).min(2) as f32 / 2.0);
            data.values = Some(targets.collect());
            data
        })
        .collect();
    (net, params, batches)
}

/// Binary cross-entropy on outputs of exactly 0 and 1 sends `Warning::ClampedLogs` as on the
/// CPU over two epochs of two batches: once per epoch, at the epoch's first step, with the
/// CPU's count of clamped logarithms.
#[test]
fn clamped_logs_warn_as_on_the_cpu() {
    let (net, params, batches) = saturated(91);
    let epoch: Vec<&Data> = batches.iter().collect();
    let epochs = vec![epoch.clone(), epoch];
    let loss = Loss::BinaryCrossEntropy;
    let health = || Health {
        dead_relu: false,
        ..Health::quiet()
    };
    let cpu = cpu_run(&net, &params, &adam(), loss, health(), &epochs);
    assert_eq!(count(&cpu.warnings, WarningKind::ClampedLogs), 2, "{cpu:?}");
    with_cuda(|cuda| {
        let (g, _) = cuda_run(cuda, &net, &params, &adam(), loss, health(), &epochs);
        assert_eq!(g.counts, cpu.counts, "the steps taken and left out");
        same_warnings(&g.warnings, &cpu.warnings, "clamped logarithms");
    });
}

/// Two ReLU layers, the first 40 wide (two words of dead-ReLU bits) and the second 9 (one
/// more), with neurons 3 and 35 of the first and 2 of the second pushed far below 0 by their
/// biases, then a log-softmax: six parameter tensors and three words of bits read back after
/// the loss. Two batches of 6 rows with a target distribution per row from `seed`, every
/// third value 0.
fn deep_log_softmax(seed: u64) -> (Sequential, ParamStore, Vec<Data>) {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(seed);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 12, 40, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut params, 40, 9, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut params, 9, 4, &mut rng))
        .push(LogSoftmax);
    for (layer, killed) in [(0, &[3usize, 35][..]), (2, &[2][..])] {
        let bias = net.layer(layer).as_linear().unwrap().bias().unwrap();
        let mut b = params.value(bias).data().to_vec();
        for &k in killed {
            b[k] = -100.0;
        }
        let n = b.len();
        *params.value_mut(bias) = Tensor::from_vec(b, &[n]).unwrap();
    }
    let batches = (0..2)
        .map(|b| {
            let mut data = Data::new(seed + 1 + b, 6, 12, 4);
            let mut t: Vec<f32> = (0..24)
                .map(|i| {
                    let v = rng.range(0.0, 1.0) as f32;
                    if i % 3 == 1 { 0.0 } else { v }
                })
                .collect();
            for row in t.chunks_mut(4) {
                let sum: f32 = row.iter().sum();
                row.iter_mut().for_each(|v| *v /= sum);
            }
            data.values = Some(t);
            data
        })
        .collect();
    (net, params, batches)
}

/// Each step's loss of `run` within 2⁻¹⁰ of the CPU's, relative, with the same steps failing.
fn same_losses(run: &[Result<u32, String>], cpu: &[Result<u32, String>], what: &str) {
    assert_eq!(run.len(), cpu.len(), "{what}: the steps");
    for (k, (g, c)) in run.iter().zip(cpu).enumerate() {
        let (g, c) = (
            f32::from_bits(*g.as_ref().unwrap()),
            f32::from_bits(*c.as_ref().unwrap()),
        );
        assert!(
            (g - c).abs() <= c.abs() / 1024.0,
            "{what}: step {k}'s loss {g}, the CPU's {c}"
        );
    }
}

/// KL divergence on [`deep_log_softmax`] with every norm check out of bounds (a high bound
/// of 0) and the dead-ReLU check on, over two epochs: each step's loss within 2⁻¹⁰ of the
/// CPU's, the steps the CPU's, and the warnings (one `Exploding` per parameter per epoch,
/// each norm within 2⁻¹⁰, and the dead neurons of both ReLU layers) the CPU's, so the norms
/// of six parameters and three words of bits are read back from the right offsets.
#[test]
fn kl_div_on_a_deep_model_checks_health_as_on_the_cpu() {
    let (net, params, batches) = deep_log_softmax(97);
    let epoch: Vec<&Data> = batches.iter().collect();
    let epochs = vec![epoch.clone(), epoch];
    let loss = Loss::KlDiv;
    let health = || Health {
        grad_norm_high: 0.0,
        grad_norm_low: 0.0,
        dead_relu: true,
        ..Health::quiet()
    };
    let cpu = cpu_run(&net, &params, &adam(), loss, health(), &epochs);
    assert_eq!(count(&cpu.warnings, WarningKind::Exploding), 12, "{cpu:?}");
    assert_eq!(count(&cpu.warnings, WarningKind::DeadNeurons), 4, "{cpu:?}");
    with_cuda(|cuda| {
        let (g, _) = cuda_run(cuda, &net, &params, &adam(), loss, health(), &epochs);
        assert_eq!(g.counts, cpu.counts, "the steps taken and left out");
        same_losses(&g.steps, &cpu.steps, "KL divergence");
        same_warnings(&g.warnings, &cpu.warnings, "KL divergence");
    });
}

/// Two `Linear` layers set so that the output is 1 (the loss is 1) and the first weight's
/// gradient is 2 · `w` · `x`: with the second weight `w` and the inputs `x`.
fn large_gradient(w: f32, x: f32) -> (Sequential, ParamStore, Data) {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 1, 1, &mut rng))
        .push(Linear::new(&mut params, 1, 1, &mut rng));
    let set = |params: &mut ParamStore, layer: usize, w: f32, b: f32| {
        let l = net.layer(layer).as_linear().unwrap();
        *params.value_mut(l.weight()) = Tensor::from_vec(vec![w], &[1, 1]).unwrap();
        *params.value_mut(l.bias().unwrap()) = Tensor::from_vec(vec![b], &[1]).unwrap();
    };
    set(&mut params, 0, 0.0, 1.0 / w);
    set(&mut params, 1, w, 0.0);
    let data = Data {
        x: vec![x, x],
        rows: 2,
        labels: Vec::new(),
        values: Some(vec![0.0, 0.0]),
    };
    (net, params, data)
}

/// A gradient that is infinite while the loss is finite is named as on the CPU: under `Stop`
/// by the error and its warning, under `Skip` by the warning, with the step left out; the
/// card leaves the update out either way.
#[test]
fn an_infinite_gradient_is_named_as_on_the_cpu() {
    let (net, params, data) = large_gradient(1e20, 1e20);
    let epochs = vec![vec![&data]];
    for policy in [OnNonFinite::Stop, OnNonFinite::Skip] {
        let health = || Health {
            on_non_finite: policy,
            ..Health::quiet()
        };
        let cpu = cpu_run(&net, &params, &adam(), Loss::Mse, health(), &epochs);
        let names: Vec<_> = cpu.warnings.iter().filter_map(Warning::location).collect();
        assert_eq!(names, ["0.weight"], "{policy:?}: {cpu:?}");
        with_cuda(|cuda| {
            let (g, updates) = cuda_run(cuda, &net, &params, &adam(), Loss::Mse, health(), &epochs);
            let what = format!("{policy:?}");
            assert_eq!(updates, vec![0; updates.len()], "{what}: no update applied");
            assert_eq!(g.counts, cpu.counts, "{what}");
            assert_eq!(g.errors(), cpu.errors(), "{what}");
            same_warnings(&g.warnings, &cpu.warnings, &what);
        });
    }
}

/// A finite gradient whose `f32` sum of squares overflows is exploding, not NaN: the card
/// sends the CPU's warnings, with that one norm reported as √f32::MAX, and the step trains.
#[test]
fn an_overflowing_sum_of_squares_is_exploding() {
    let (net, params, data) = large_gradient(1e10, 1e9);
    let epochs = vec![vec![&data]];
    let cpu = cpu_run(&net, &params, &adam(), Loss::Mse, Health::quiet(), &epochs);
    let weight = |w: &[Warning]| {
        (w.iter())
            .find(|w| w.location() == Some("0.weight"))
            .cloned()
            .unwrap()
    };
    let Warning::Exploding { norm, .. } = weight(&cpu.warnings) else {
        panic!("{cpu:?}")
    };
    assert!((norm - 2e19).abs() < 1e14, "{norm:e}");
    with_cuda(|cuda| {
        let quiet = Health::quiet();
        let (g, updates) = cuda_run(cuda, &net, &params, &adam(), Loss::Mse, quiet, &epochs);
        assert_eq!(updates, vec![1; updates.len()], "the step trains");
        assert_eq!(g.counts, cpu.counts);
        let Warning::Exploding { norm, .. } = weight(&g.warnings) else {
            panic!("{g:?}")
        };
        assert_eq!(norm, f64::from(f32::MAX).sqrt());
        let others = |w: &[Warning]| {
            (w.iter())
                .filter(|w| w.location() != Some("0.weight"))
                .cloned()
                .collect::<Vec<_>>()
        };
        assert!(!others(&cpu.warnings).is_empty());
        same_warnings(&others(&g.warnings), &others(&cpu.warnings), "overflow");
    });
}

/// A model of one bias-free `Linear` (one parameter tensor of one value) trains.
#[test]
fn one_parameter_tensor_trains() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(3);
    let net = Sequential::new().push(Linear::without_bias(
        &mut params,
        1,
        1,
        Init::TorchDefault,
        &mut rng,
    ));
    let data = Data {
        x: vec![1.0, 2.0],
        rows: 2,
        labels: Vec::new(),
        values: Some(vec![3.0, 5.0]),
    };
    let sgd = OptimizerConfig::Sgd {
        lr: 0.1,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    with_cuda(|cuda| {
        let mut t = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 2, cuda).unwrap();
        t.set_health(Health::quiet()).unwrap();
        let first = t.step(&net, data.view()).unwrap();
        let second = t.step(&net, data.view()).unwrap();
        assert!(second < first, "{first} then {second}");
    });
}

/// Two epochs of three batches on the MLP with every parameter's norm out of bounds, first
/// above the high bound (0) and then below the low one (∞): one warning per parameter per
/// epoch, as on the CPU, each norm within 2⁻¹⁰ of the CPU's.
#[test]
fn norm_warnings_match_the_cpu() {
    let t = twin(mlp::<f64>, mlp::<f32>, 47);
    let batches: Vec<Data> = (0..3).map(|s| Data::new(70 + s, 6, 12, 4)).collect();
    let epoch: Vec<&Data> = batches.iter().collect();
    let epochs = vec![epoch.clone(), epoch];
    let params = t.net.parameter_names().len();
    for (high, low, kind) in [
        (0.0, 0.0, WarningKind::Exploding),
        (f64::INFINITY, f64::INFINITY, WarningKind::Vanishing),
    ] {
        let health = || Health {
            grad_norm_high: high,
            grad_norm_low: low,
            dead_relu: false,
            ..Health::quiet()
        };
        let loss = Loss::CrossEntropy;
        let cpu = cpu_run(&t.net, &t.params, &adam(), loss, health(), &epochs);
        assert_eq!(count(&cpu.warnings, kind), 2 * params, "{kind:?}: {cpu:?}");
        with_cuda(|cuda| {
            let (g, _) = cuda_run(cuda, &t.net, &t.params, &adam(), loss, health(), &epochs);
            same_warnings(&g.warnings, &cpu.warnings, &format!("{kind:?}"));
        });
    }
}

/// The norm checks of the MLP with a PReLU before each `Linear`, as above: one warning per
/// parameter per epoch, the slopes among them, as on the CPU, each norm within 2⁻¹⁰ of the
/// CPU's.
#[test]
fn prelu_slope_norms_warn_as_on_the_cpu() {
    let t = twin(
        |p, s| prelu_mlp(p, s, false),
        |p, s| prelu_mlp(p, s, false),
        47,
    );
    let batches: Vec<Data> = (0..3).map(|s| Data::new(70 + s, 6, 12, 4)).collect();
    let epoch: Vec<&Data> = batches.iter().collect();
    let epochs = vec![epoch.clone(), epoch];
    let params = t.net.parameter_names().len();
    for (high, low, kind) in [
        (0.0, 0.0, WarningKind::Exploding),
        (f64::INFINITY, f64::INFINITY, WarningKind::Vanishing),
    ] {
        let health = || Health {
            grad_norm_high: high,
            grad_norm_low: low,
            dead_relu: false,
            ..Health::quiet()
        };
        let loss = Loss::CrossEntropy;
        let cpu = cpu_run(&t.net, &t.params, &adam(), loss, health(), &epochs);
        assert_eq!(count(&cpu.warnings, kind), 2 * params, "{kind:?}: {cpu:?}");
        with_cuda(|cuda| {
            let (g, _) = cuda_run(cuda, &t.net, &t.params, &adam(), loss, health(), &epochs);
            same_warnings(&g.warnings, &cpu.warnings, &format!("{kind:?}"));
        });
    }
}

/// A `Linear` → ReLU, a convolution → ReLU and two ReLU layers (the first 40 wide, two words
/// of bits) with some neurons (and one channel) pushed far below 0 by their biases: the
/// dead-neuron reports of two epochs match the CPU's, and with `dead_relu` off neither sends
/// one.
#[test]
fn dead_relus_match_the_cpu() {
    let mut dense = twin(mlp::<f64>, mlp::<f32>, 53);
    let bias = dense.net.layer(2).as_linear().unwrap().bias().unwrap();
    let mut b = dense.params.value(bias).data().to_vec();
    for k in [1, 4, 7] {
        b[k] = -100.0;
    }
    *dense.params.value_mut(bias) = Tensor::from_vec(b, &[9]).unwrap();
    let dense_batches: Vec<Data> = (0..2).map(|s| Data::new(80 + s, 6, 12, 4)).collect();
    let mut cnn_params = ParamStore::new();
    let mut rng = Rng::seed(59);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    let cnn = Sequential::new()
        .push(Conv2d::new(&mut cnn_params, g, &mut rng))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Flatten)
        .push(Linear::new(&mut cnn_params, 64, 3, &mut rng));
    let conv_bias = cnn.layer(0).as_conv2d().unwrap().bias().unwrap();
    *cnn_params.value_mut(conv_bias) =
        Tensor::from_vec(vec![0.1, -0.2, -100.0, 0.3], &[4]).unwrap();
    let cnn_batches: Vec<Data> = (0..2).map(|s| Data::new(90 + s, 6, 128, 3)).collect();
    let mut wide_params = ParamStore::new();
    let mut rng = Rng::seed(61);
    let wide = Sequential::new()
        .push(Linear::new(&mut wide_params, 12, 40, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut wide_params, 40, 9, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut wide_params, 9, 4, &mut rng));
    for (layer, killed) in [(0, &[3usize, 35][..]), (2, &[2][..])] {
        let bias = wide.layer(layer).as_linear().unwrap().bias().unwrap();
        let mut b = wide_params.value(bias).data().to_vec();
        for &k in killed {
            b[k] = -100.0;
        }
        let n = b.len();
        *wide_params.value_mut(bias) = Tensor::from_vec(b, &[n]).unwrap();
    }
    let cases = [
        (&dense.net, &dense.params, &dense_batches, "dense", 2),
        (&cnn, &cnn_params, &cnn_batches, "conv", 2),
        (&wide, &wide_params, &dense_batches, "two ReLU layers", 4),
    ];
    for (net, params, batches, what, reports) in cases {
        let epoch: Vec<&Data> = batches.iter().collect();
        let epochs = vec![epoch.clone(), epoch];
        for dead_relu in [true, false] {
            let health = || Health {
                dead_relu,
                ..Health::quiet()
            };
            let loss = Loss::CrossEntropy;
            let cpu = cpu_run(net, params, &adam(), loss, health(), &epochs);
            let dead = count(&cpu.warnings, WarningKind::DeadNeurons);
            assert_eq!(dead, if dead_relu { reports } else { 0 }, "{what}: {cpu:?}");
            with_cuda(|cuda| {
                let (g, _) = cuda_run(cuda, net, params, &adam(), loss, health(), &epochs);
                let what = format!("{what}, dead_relu {dead_relu}");
                same_warnings(&g.warnings, &cpu.warnings, &what);
            });
        }
    }
}

/// A `Linear` → ReLU6 with three of its nine neurons pushed far below 0 by their biases,
/// under the default health checks: ReLU6 is checked for dead neurons as ReLU is, and the
/// reports of two epochs (and every other warning) match the CPU's.
#[test]
fn dead_relu6_neurons_match_the_cpu() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(67);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 12, 9, &mut rng))
        .push(Relu6)
        .push(Linear::new(&mut params, 9, 4, &mut rng));
    let bias = net.layer(0).as_linear().unwrap().bias().unwrap();
    let mut b = params.value(bias).data().to_vec();
    for k in [1, 4, 7] {
        b[k] = -100.0;
    }
    *params.value_mut(bias) = Tensor::from_vec(b, &[9]).unwrap();
    let batches: Vec<Data> = (0..2).map(|s| Data::new(84 + s, 6, 12, 4)).collect();
    let epoch: Vec<&Data> = batches.iter().collect();
    let epochs = vec![epoch.clone(), epoch];
    let loss = Loss::CrossEntropy;
    let cpu = cpu_run(&net, &params, &adam(), loss, Health::default(), &epochs);
    assert_eq!(count(&cpu.warnings, WarningKind::DeadNeurons), 2, "{cpu:?}");
    with_cuda(|cuda| {
        let (g, _) = cuda_run(
            cuda,
            &net,
            &params,
            &adam(),
            loss,
            Health::default(),
            &epochs,
        );
        same_warnings(&g.warnings, &cpu.warnings, "relu6");
    });
}
