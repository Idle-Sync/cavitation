//! `GpuTrainer`'s health checks against the CPU `Trainer`'s, run side by side from the same
//! start on the same batches: a NaN batch stopped or left out as on the CPU (with the same
//! step and skip counts, and Adam's update counts unchanged by a skipped step), a gradient
//! that overflows to infinity while the loss stays finite named as on the CPU, an epoch of
//! skipped steps refused as on the CPU, and the warnings of seeded runs (the norm checks on
//! every parameter, and dead ReLU and ReLU6 neurons after a `Linear` and after a convolution)
//! matching the CPU's kind for kind, step for step and name for name, with each norm within
//! 2⁻¹⁰ of the CPU's.
//!
//! Each comparison first checks that the CPU run sent the warnings it is about, so two empty
//! lists never pass. Without a GPU every test says so and checks nothing;
//! `CAVITATION_GPU_REQUIRED=1` makes that a failure.

use std::sync::{Arc, Mutex};

use cavitation_core::{Conv2dGeom, Error, ParamStore, Pool2dGeom, Rng, Tensor, TensorView};
use cavitation_gpu::{Backend, Gpu};

use super::GpuTrainer;
use super::forward_tests::{mlp, twin};
use crate::{
    BatchView, Conv2d, Flatten, Health, Init, Linear, Loss, MaxPool2d, OnNonFinite,
    OptimizerConfig, Relu, Relu6, Sequential, TargetsView, Trainer, Warning, WarningKind,
};

/// Runs `body` on each GPU this platform has, one per backend, and returns how many ran.
pub(super) fn for_each_gpu(mut body: impl FnMut(&Gpu)) -> usize {
    let mut ran = 0;
    for &backend in Backend::on_this_platform() {
        if let Some(gpu) = Gpu::with_backend(backend) {
            println!("on {backend}: {}", gpu.info().name);
            body(&gpu);
            ran += 1;
        }
    }
    if ran == 0 {
        if std::env::var_os("CAVITATION_GPU_REQUIRED").is_some() {
            panic!("CAVITATION_GPU_REQUIRED is set, but no GPU opened");
        }
        println!("no GPU: skipped");
    }
    ran
}

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

/// One batch: rows of inputs, and class labels or target values.
pub(super) struct Data {
    /// The inputs, row after row.
    pub(super) x: Vec<f32>,
    /// The rows.
    pub(super) rows: usize,
    /// Class labels, for cross-entropy.
    pub(super) labels: Vec<usize>,
    /// Target values, for MSE.
    pub(super) values: Option<Vec<f32>>,
}

impl Data {
    /// `rows` rows of `width` inputs in `[-1, 1)` and labels below `classes`, from `seed`.
    pub(super) fn new(seed: u64, rows: usize, width: usize, classes: usize) -> Data {
        let mut rng = Rng::seed(seed);
        Data {
            x: (0..rows * width)
                .map(|_| rng.range(-1.0, 1.0) as f32)
                .collect(),
            rows,
            labels: (0..rows)
                .map(|_| (rng.range(0.0, classes as f64) as usize).min(classes - 1))
                .collect(),
            values: None,
        }
    }

    /// The batch as a trainer takes it.
    pub(super) fn view(&self) -> BatchView<'_, f32> {
        let x = TensorView::from_slice(&self.x, &[self.rows, self.x.len() / self.rows]).unwrap();
        let targets = match &self.values {
            Some(v) => TargetsView::Values(
                TensorView::from_slice(v, &[self.rows, v.len() / self.rows]).unwrap(),
            ),
            None => TargetsView::Classes(&self.labels),
        };
        BatchView { x, targets }
    }
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

/// The CPU `Trainer`'s run of `net` from `params`.
fn cpu_run(
    net: &Sequential,
    params: &ParamStore,
    config: &OptimizerConfig,
    loss: Loss,
    batch: usize,
    health: Health<'static>,
    epochs: &[Vec<&Data>],
) -> Run {
    let (health, seen) = collecting(health);
    let mut params = params.clone();
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

/// The `GpuTrainer`'s run, and its Adam (or SGD) update counts after it.
#[allow(clippy::too_many_arguments)]
fn gpu_run(
    gpu: &Gpu,
    net: &Sequential,
    params: &ParamStore,
    config: &OptimizerConfig,
    loss: Loss,
    batch: usize,
    health: Health<'static>,
    epochs: &[Vec<&Data>],
) -> (Run, Vec<u32>) {
    let (health, seen) = collecting(health);
    let mut trainer = GpuTrainer::new(net, params, config, loss, batch, gpu).unwrap();
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

/// Checks that the GPU's warnings are the CPU's: the same kinds, steps, epochs, locations and
/// layers in the same order, the same dead neurons and inputs, and each norm within 2⁻¹⁰.
fn same_warnings(gpu: &[Warning], cpu: &[Warning], what: &str) {
    let show = |w: &[Warning]| w.iter().map(|w| format!("{w}\n")).collect::<String>();
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "{what}: the GPU sent\n{}the CPU\n{}",
        show(gpu),
        show(cpu)
    );
    for (g, c) in gpu.iter().zip(cpu) {
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

/// A NaN in a batch's inputs stops training under `Stop` as on the CPU: the same error text
/// (step, location and the input's position), the same steps counted and the same warning;
/// under `Skip` the step is left out as on the CPU, counted in the steps and the skipped steps,
/// with Adam's update counts one lower than the steps.
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
        let cpu = cpu_run(
            &t.net,
            &t.params,
            &adam(),
            Loss::CrossEntropy,
            6,
            health(),
            &epochs,
        );
        assert_eq!(count(&cpu.warnings, WarningKind::NonFinite), 1, "{cpu:?}");
        match policy {
            OnNonFinite::Stop => assert!(cpu.steps[1].is_err() && cpu.steps.len() == 2),
            _ => assert_eq!(cpu.counts, (4, 1)),
        }
        for_each_gpu(|gpu| {
            let (g, updates) = gpu_run(
                gpu,
                &t.net,
                &t.params,
                &adam(),
                Loss::CrossEntropy,
                6,
                health(),
                &epochs,
            );
            let what = format!("{policy:?}");
            assert_eq!(g.counts, cpu.counts, "{what}");
            assert_eq!(g.epochs, cpu.epochs, "{what}");
            // The losses differ in their last bits; which steps failed, and how, must not.
            let errors = |r: &Run| {
                r.steps
                    .iter()
                    .map(|s| s.as_ref().err().cloned())
                    .collect::<Vec<_>>()
            };
            assert_eq!(errors(&g), errors(&cpu), "{what}");
            same_warnings(&g.warnings, &cpu.warnings, &what);
            let trained =
                (cpu.counts.0 - cpu.counts.1) as u32 - u32::from(policy == OnNonFinite::Stop);
            assert_eq!(
                updates,
                vec![trained; updates.len()],
                "{what}: Adam's update counts"
            );
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
    let cpu = cpu_run(
        &t.net,
        &t.params,
        &adam(),
        Loss::CrossEntropy,
        6,
        health(),
        &epochs,
    );
    assert!(cpu.epochs[0].is_err(), "{cpu:?}");
    for_each_gpu(|gpu| {
        let (g, updates) = gpu_run(
            gpu,
            &t.net,
            &t.params,
            &adam(),
            Loss::CrossEntropy,
            6,
            health(),
            &epochs,
        );
        assert_eq!(g.epochs, cpu.epochs);
        assert_eq!(g.counts, cpu.counts);
        same_warnings(&g.warnings, &cpu.warnings, "skipped epoch");
        assert_eq!(updates, vec![0; updates.len()]);
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

/// A gradient that is infinite while the loss is finite (2 · 10²⁰ · 10²⁰ overflows) is named
/// as on the CPU: under `Stop` by the error and its warning, under `Skip` by the warning, with
/// the step left out. Either way the GPU leaves the update out (under `Stop` the CPU has
/// applied it): Adam's update counts stay 0.
#[test]
fn an_infinite_gradient_is_named_as_on_the_cpu() {
    let (net, params, data) = large_gradient(1e20, 1e20);
    let sgd = adam();
    let epochs = vec![vec![&data]];
    for policy in [OnNonFinite::Stop, OnNonFinite::Skip] {
        let health = || Health {
            on_non_finite: policy,
            ..Health::quiet()
        };
        let cpu = cpu_run(&net, &params, &sgd, Loss::Mse, 2, health(), &epochs);
        let names: Vec<_> = cpu.warnings.iter().filter_map(Warning::location).collect();
        assert_eq!(names, ["0.weight"], "{policy:?}: {cpu:?}");
        for_each_gpu(|gpu| {
            let (g, updates) = gpu_run(gpu, &net, &params, &sgd, Loss::Mse, 2, health(), &epochs);
            let what = format!("{policy:?}");
            assert_eq!(updates, vec![0; updates.len()], "{what}: no update applied");
            assert_eq!(g.counts, cpu.counts, "{what}");
            let errors = |r: &Run| {
                r.steps
                    .iter()
                    .map(|s| s.as_ref().err().cloned())
                    .collect::<Vec<_>>()
            };
            assert_eq!(errors(&g), errors(&cpu), "{what}");
            same_warnings(&g.warnings, &cpu.warnings, &what);
        });
    }
}

/// A finite gradient whose `f32` sum of squares overflows (2 · 10¹⁰ · 10⁹ = 2 · 10¹⁹, squared
/// above `f32::MAX`) is exploding, not NaN: the GPU sends the CPU's warnings, naming the same
/// parameters at the same step, with that one norm reported as √f32::MAX, a lower bound of the
/// CPU's 2 · 10¹⁹, and the step trains.
#[test]
fn an_overflowing_sum_of_squares_is_exploding() {
    let (net, params, data) = large_gradient(1e10, 1e9);
    let epochs = vec![vec![&data]];
    let cpu = cpu_run(
        &net,
        &params,
        &adam(),
        Loss::Mse,
        2,
        Health::quiet(),
        &epochs,
    );
    let weight = |w: &[Warning]| {
        w.iter()
            .find(|w| w.location() == Some("0.weight"))
            .cloned()
            .unwrap()
    };
    let Warning::Exploding { norm, .. } = weight(&cpu.warnings) else {
        panic!("{cpu:?}")
    };
    assert!((norm - 2e19).abs() < 1e14, "{norm:e}");
    for_each_gpu(|gpu| {
        let (g, updates) = gpu_run(
            gpu,
            &net,
            &params,
            &adam(),
            Loss::Mse,
            2,
            Health::quiet(),
            &epochs,
        );
        assert_eq!(updates, vec![1; updates.len()], "the step trains");
        assert_eq!(g.counts, cpu.counts);
        let Warning::Exploding { norm, .. } = weight(&g.warnings) else {
            panic!("{g:?}")
        };
        assert_eq!(norm, f64::from(f32::MAX).sqrt());
        // The rest are the CPU's, norms included.
        let others = |w: &[Warning]| {
            w.iter()
                .filter(|w| w.location() != Some("0.weight"))
                .cloned()
                .collect::<Vec<_>>()
        };
        assert!(!others(&cpu.warnings).is_empty());
        same_warnings(&others(&g.warnings), &others(&cpu.warnings), "overflow");
    });
}

/// A model of one bias-free `Linear` (one parameter tensor of one value) trains: the step's
/// readback holds the health values of an odd number of tensors.
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
    for_each_gpu(|gpu| {
        let mut t = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 2, gpu).unwrap();
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
        let cpu = cpu_run(
            &t.net,
            &t.params,
            &adam(),
            Loss::CrossEntropy,
            6,
            health(),
            &epochs,
        );
        assert_eq!(count(&cpu.warnings, kind), 2 * params, "{kind:?}: {cpu:?}");
        for_each_gpu(|gpu| {
            let (g, _) = gpu_run(
                gpu,
                &t.net,
                &t.params,
                &adam(),
                Loss::CrossEntropy,
                6,
                health(),
                &epochs,
            );
            same_warnings(&g.warnings, &cpu.warnings, &format!("{kind:?}"));
        });
    }
}

/// A `Linear` → ReLU and a convolution → ReLU with some neurons (and one channel) pushed far
/// below 0 by their biases: the dead-neuron reports of two epochs match the CPU's, and with
/// `dead_relu` off neither sends one.
#[test]
fn dead_relus_match_the_cpu() {
    // The MLP's ReLU (layer 3) follows a `Linear` of 9 outputs; three of them are killed.
    let mut dense = twin(mlp::<f64>, mlp::<f32>, 53);
    let bias = dense.net.layer(2).as_linear().unwrap().bias().unwrap();
    let mut b = dense.params.value(bias).data().to_vec();
    for k in [1, 4, 7] {
        b[k] = -100.0;
    }
    *dense.params.value_mut(bias) = Tensor::from_vec(b, &[9]).unwrap();
    let dense_batches: Vec<Data> = (0..2).map(|s| Data::new(80 + s, 6, 12, 4)).collect();
    // A convolution of 4 channels on 2 × 8 × 8 images, then ReLU; channel 2 is killed.
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
    // Two ReLU layers, the first 40 wide (two words of bits): neurons 3 and 35 of the first
    // and 2 of the second are killed.
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
            let cpu = cpu_run(
                net,
                params,
                &adam(),
                Loss::CrossEntropy,
                6,
                health(),
                &epochs,
            );
            let dead = count(&cpu.warnings, WarningKind::DeadNeurons);
            assert_eq!(dead, if dead_relu { reports } else { 0 }, "{what}: {cpu:?}");
            for_each_gpu(|gpu| {
                let (g, _) = gpu_run(
                    gpu,
                    net,
                    params,
                    &adam(),
                    Loss::CrossEntropy,
                    6,
                    health(),
                    &epochs,
                );
                same_warnings(
                    &g.warnings,
                    &cpu.warnings,
                    &format!("{what}, dead_relu {dead_relu}"),
                );
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
    let cpu = cpu_run(
        &net,
        &params,
        &adam(),
        Loss::CrossEntropy,
        6,
        Health::default(),
        &epochs,
    );
    assert_eq!(count(&cpu.warnings, WarningKind::DeadNeurons), 2, "{cpu:?}");
    for_each_gpu(|gpu| {
        let (g, _) = gpu_run(
            gpu,
            &net,
            &params,
            &adam(),
            Loss::CrossEntropy,
            6,
            Health::default(),
            &epochs,
        );
        same_warnings(&g.warnings, &cpu.warnings, "relu6");
    });
}
