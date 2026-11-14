//! Timed runs: the matrix products of one training step, each timed on its own, the
//! training and inference runs that are compared with the numpy baseline, the parts of the
//! `shrink` run, the runs of the `inspect` benchmark, and the `activation` timings.

use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

use cavitation_core::kernels::activation::{sigmoid_f32_with, tanh_f32_with};
use cavitation_core::kernels::gemm::{gemm_f32_with, threads_used};
use cavitation_core::kernels::{Level, MatMut, MatRef};
use cavitation_core::{ParamStore, Rng, Tensor};
use cavitation_datasets::{cifar10, digits, mnist};
use cavitation_inspect::{Inspector, InspectorConfig};
use cavitation_nn::shrink::{ShrinkConfig, ShrinkStep, Validation, shrink};
use cavitation_nn::{
    Adam, Batch, Batcher, Dataset, Evaluation, FitConfig, Health, History, InferenceModel, Init,
    Linear, Loss, LrSchedule, Optimizer, OptimizerConfig, Relu, Sequential, Sgd, Tanh, Targets,
    Trainer, Warning, WarningKind, accuracy, fit,
};

use crate::numpy;

/// One matrix product of a training step, `C [m, n] = A [m, k] · B [k, n]`, with operands
/// stored the way the step stores them.
#[derive(Clone, Copy, Debug)]
pub struct Product {
    /// What the product is in the step.
    pub name: &'static str,
    /// Rows of `A` and `C`.
    pub m: usize,
    /// Columns of `B` and `C`.
    pub n: usize,
    /// Columns of `A`, rows of `B`: the length of each sum.
    pub k: usize,
    /// `A` is the transpose of a row-major `[k, m]` matrix (a weight gradient's `dYᵀ`).
    pub a_transposed: bool,
    /// `B` is the transpose of a row-major `[n, k]` matrix (a forward pass's `Wᵀ`).
    pub b_transposed: bool,
}

impl Product {
    /// The floating-point operations of one product: `2 · m · n · k`.
    pub fn flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }
}

/// The eight products of one batch-128 training step of an MLP with layer widths `widths`
/// (input first): the forward products, the weight gradients, and the input gradients of
/// every layer but the first (the first layer's input gradient is never needed).
fn step_products(widths: [usize; 4], names: &'static [&'static str; 8]) -> [Product; 8] {
    let batch = 128;
    let [w0, w1, w2, w3] = widths;
    let p = |i: usize, m, n, k, a_transposed, b_transposed| Product {
        name: names[i],
        m,
        n,
        k,
        a_transposed,
        b_transposed,
    };
    [
        p(0, batch, w1, w0, false, true),
        p(1, batch, w2, w1, false, true),
        p(2, batch, w3, w2, false, true),
        p(3, w3, w2, batch, true, false),
        p(4, w2, w1, batch, true, false),
        p(5, w1, w0, batch, true, false),
        p(6, batch, w2, w3, false, false),
        p(7, batch, w1, w2, false, false),
    ]
}

/// The step's product names, in the order [`step_products`] lists them.
const NAMES: [&str; 8] = [
    "forward 1   X·W1ᵀ",
    "forward 2   H1·W2ᵀ",
    "forward 3   H2·W3ᵀ",
    "weights 3   dY3ᵀ·H2",
    "weights 2   dY2ᵀ·H1",
    "weights 1   dY1ᵀ·X",
    "input 3     dY3·W3",
    "input 2     dY2·W2",
];

/// The products of one batch-128 step of a named network: `mnist128` (784 → 1024 → 256 →
/// 10, 1,068,810 parameters) or `digits128` (64 → 256 → 128 → 10, 50,826 parameters).
///
/// # Errors
///
/// Returns a message naming `name` and the accepted names if it is neither.
pub fn shapes(name: &str) -> Result<[Product; 8], String> {
    match name {
        "mnist128" => Ok(step_products([784, 1024, 256, 10], &NAMES)),
        "digits128" => Ok(step_products([64, 256, 128, 10], &NAMES)),
        _ => Err(format!(
            "unknown shape set `{name}`; expected mnist128 or digits128"
        )),
    }
}

/// The timing of one product.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// The median time of one call, in seconds.
    pub seconds: f64,
    /// The threads the call used.
    pub threads: usize,
}

/// Times `product` at `level` on `threads` threads: warms up for about 50 ms, then takes 31
/// samples, each the mean of enough back-to-back calls to last about 2 ms, and returns the
/// median sample. Inputs are fixed pseudo-random values; `beta` is 0, as for the first
/// contribution to a gradient.
pub fn time_product(level: Level, threads: usize, product: Product) -> Timing {
    let Product { m, n, k, .. } = product;
    let fill = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 7919 + seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    };
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let a_view = if product.a_transposed {
        MatRef::new(&a, k, m, m, 1).unwrap().transpose()
    } else {
        MatRef::new(&a, m, k, k, 1).unwrap()
    };
    let b_view = if product.b_transposed {
        MatRef::new(&b, n, k, k, 1).unwrap().transpose()
    } else {
        MatRef::new(&b, k, n, n, 1).unwrap()
    };
    let mut call = || {
        gemm_f32_with(
            level,
            threads,
            black_box(1.0),
            black_box(a_view),
            black_box(b_view),
            black_box(0.0),
            MatMut::new(&mut c, m, n, n, 1).unwrap(),
        );
        black_box(&mut c);
    };
    let start = Instant::now();
    let mut calls = 0usize;
    while start.elapsed().as_secs_f64() < 0.05 || calls < 5 {
        call();
        calls += 1;
    }
    let per_call = start.elapsed().as_secs_f64() / calls as f64;
    let reps = ((2e-3 / per_call).ceil() as usize).max(1);
    let mut samples: Vec<f64> = (0..31)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                call();
            }
            t.elapsed().as_secs_f64() / reps as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    Timing {
        seconds: samples[samples.len() / 2],
        threads: threads_used(level, threads, m, n, k),
    }
}

/// Untimed training steps (or inference calls for `latency`) before the timed part of a
/// training run.
const TRAIN_WARMUP: usize = 20;

/// The layer widths of the 1M network: 784 to 1024 to 256 to 10, 1,068,810 parameters.
pub const MNIST_NET: [usize; 4] = [784, 1024, 256, 10];

/// The layer widths of the 50k network: 64 to 256 to 128 to 10, 50,826 parameters.
pub const DIGITS_NET: [usize; 4] = [64, 256, 128, 10];

/// The layer widths of the CIFAR-10 network: 3072 to 1024 to 512 to 10, 3,676,682
/// parameters.
pub const CIFAR_NET: [usize; 4] = [3072, 1024, 512, 10];

/// The median of `values` (the upper middle one for an even count).
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// A ReLU MLP with the given layer widths, He normal weights and zero biases, seeded with 1.
fn build_net(params: &mut ParamStore, widths: &[usize]) -> Sequential {
    let mut rng = Rng::seed(1);
    let mut net = Sequential::new();
    for (i, pair) in widths.windows(2).enumerate() {
        net = net.push(Linear::with_init(
            params,
            pair[0],
            pair[1],
            Init::HeNormal,
            Init::Zeros,
            &mut rng,
        ));
        if i + 2 < widths.len() {
            net = net.push(Relu);
        }
    }
    net
}

/// One compared measurement: the same work timed on cavitation and on numpy.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// What was run, for example `mnist batch 128`.
    pub name: String,
    /// The details both sides share: network, optimiser, steps.
    pub detail: String,
    /// Seconds per epoch or per round, as cavitation measured them.
    pub ours: Vec<f64>,
    /// Seconds per epoch or per round, as numpy measured them at the setting compared
    /// against: the fastest tried (see [`numpy_setting`](Self::numpy_setting)).
    pub numpy: Vec<f64>,
    /// Seconds per epoch or per round of numpy at its default settings (the same run as
    /// `numpy` when only the default was tried).
    pub numpy_default: Vec<f64>,
    /// The setting `numpy` was measured at: `default`, `1 thread`, `core Nehalem`, ...
    pub numpy_setting: String,
    /// Every numpy setting tried, in order, with its median seconds.
    pub numpy_tried: Vec<(String, f64)>,
    /// The work in each timed sample: samples per epoch or steps per timed run (training),
    /// or calls per round (latency).
    pub work: f64,
    /// What each side's timed region includes.
    pub timing_note: String,
    /// The numpy version that ran the baseline.
    pub numpy_version: String,
    /// The BLAS numpy ran on at the setting compared against (library, version, OpenBLAS
    /// core, threads).
    pub blas: String,
    /// The BLAS numpy ran on at its default settings.
    pub blas_default: String,
    /// The health warnings numpy's own checks raised (training only).
    pub numpy_warnings: Option<usize>,
    /// How to show a median: `s/epoch`, `samples/s`, `steps/s` or `us/call`.
    pub unit: Unit,
    /// The health warnings the run raised, by kind (see [`HealthLog::summary`]); empty for
    /// runs that do not train.
    pub health: String,
}

/// The health warnings of one training run, grouped by kind and place.
#[derive(Debug, Default)]
pub struct HealthLog {
    /// For each kind and place (layer or location) in order of first appearance: the epoch of
    /// its first warning (0 for the untimed warm-up) and how many came.
    groups: Vec<(WarningKind, Option<String>, usize, usize)>,
}

impl HealthLog {
    /// Records `warning`, raised during `epoch` (0 for the untimed warm-up steps). A warning
    /// that knows its own epoch (dead neurons) uses that one.
    pub fn record(&mut self, warning: &Warning, epoch: usize) {
        let kind = warning.kind();
        let place = match (warning.layer(), warning.location()) {
            (Some(layer), _) => Some(format!("layer {layer}")),
            (None, Some(location)) => Some(location.to_string()),
            (None, None) => None,
        };
        if let Some(group) = self.groups.iter_mut().find(|g| g.0 == kind && g.1 == place) {
            group.3 += 1;
            return;
        }
        self.groups
            .push((kind, place, warning.epoch().unwrap_or(epoch), 1));
    }

    /// How many warnings were recorded in all.
    pub fn total(&self) -> usize {
        self.groups.iter().map(|g| g.3).sum()
    }

    /// One line naming each kind and layer (or location) and first epoch, then its
    /// count, for example `health: 4 warnings: DeadNeurons (layer 3, from epoch 1) x3;
    /// DeadNeurons (layer 5, from epoch 1) x1`; `health: no warnings` if there were none.
    pub fn summary(&self) -> String {
        if self.groups.is_empty() {
            return "health: no warnings".to_string();
        }
        let kinds: Vec<String> = self
            .groups
            .iter()
            .map(|(kind, place, epoch, count)| {
                let from = if *epoch == 0 {
                    "from the warm-up".to_string()
                } else {
                    format!("from epoch {epoch}")
                };
                match place {
                    Some(place) => format!("{kind:?} ({place}, {from}) x{count}"),
                    None => format!("{kind:?} ({from}) x{count}"),
                }
            })
            .collect();
        let total = self.total();
        let noun = if total == 1 { "warning" } else { "warnings" };
        format!("health: {total} {noun}: {}", kinds.join("; "))
    }
}

/// How a median time is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Seconds per epoch.
    EpochSeconds,
    /// Training samples per second.
    SamplesPerSecond,
    /// Training steps per second.
    StepsPerSecond,
    /// Microseconds per call.
    MicrosPerCall,
}

impl Comparison {
    /// The value of a median time `seconds` in this comparison's unit.
    pub fn value(&self, seconds: f64) -> f64 {
        match self.unit {
            Unit::EpochSeconds => seconds,
            Unit::SamplesPerSecond | Unit::StepsPerSecond => self.work / seconds,
            Unit::MicrosPerCall => seconds / self.work * 1e6,
        }
    }

    /// The unit's label.
    pub fn unit_label(&self) -> &'static str {
        match self.unit {
            Unit::EpochSeconds => "s/epoch",
            Unit::SamplesPerSecond => "samples/s",
            Unit::StepsPerSecond => "steps/s",
            Unit::MicrosPerCall => "us/call",
        }
    }

    /// How many times faster cavitation is: numpy's median time (at the setting compared
    /// against) over ours.
    pub fn ratio(&self) -> f64 {
        median(&self.numpy) / median(&self.ours)
    }

    /// How many times faster cavitation is than numpy at its default settings.
    pub fn default_ratio(&self) -> f64 {
        median(&self.numpy_default) / median(&self.ours)
    }

    /// The numpy settings tried with their medians in this comparison's unit, for example
    /// `default 12.31, 1 thread 9.80`.
    pub fn tried_summary(&self) -> String {
        self.numpy_tried
            .iter()
            .map(|(label, seconds)| format!("{label} {:.2}", self.value(*seconds)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What a training run measures and compares.
#[derive(Clone, Debug)]
pub struct TrainSpec {
    /// The label of the run.
    pub name: String,
    /// `mnist`, `digits` or `cifar`.
    pub data: &'static str,
    /// Layer widths, input first.
    pub widths: [usize; 4],
    /// The batch size.
    pub batch: usize,
    /// Adam with learning rate 1e-3 if true, plain SGD with 0.01 if false.
    pub adam: bool,
    /// Timed epochs, or timed runs of `steps` steps.
    pub epochs: usize,
    /// `Some(n)`: each timed sample is `n` steps instead of a whole epoch, for machines on
    /// which an epoch is slow. The steps go through the batches in order, wrapping round.
    pub steps: Option<usize>,
    /// How to show the median.
    pub unit: Unit,
}

/// Times `spec.epochs` epochs (or runs of `spec.steps` steps) of training through the public
/// path (`Trainer`, `Batcher`, the default health checks on) after [`TRAIN_WARMUP`] untimed
/// steps, and returns the seconds of each, with the health warnings raised.
///
/// # Errors
///
/// Returns a message if the training stops (a non-finite value) or the data is missing.
fn train_ours(spec: &TrainSpec, data: &Dataset) -> Result<(Vec<f64>, HealthLog), String> {
    let mut params = ParamStore::new();
    let net = build_net(&mut params, &spec.widths);
    let optimizer = if spec.adam {
        OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        }
    } else {
        OptimizerConfig::Sgd {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    };
    let mut trainer = Trainer::new(&net, optimizer.build(), Loss::CrossEntropy, spec.batch);
    // The default checks, with the warnings recorded instead of printed to standard error:
    // the sink adds to a count (the first of each kind also keeps its place), and the
    // summary is formatted after the timing.
    let warnings = Rc::new(RefCell::new(HealthLog::default()));
    let epoch_now = Rc::new(Cell::new(0usize));
    let (log, now) = (Rc::clone(&warnings), Rc::clone(&epoch_now));
    trainer.set_health(Health {
        sink: Some(Box::new(move |w: &Warning| {
            log.borrow_mut().record(w, now.get());
        })),
        ..Health::default()
    });
    let mut batcher = Batcher::new(data.len(), spec.batch, None);
    let mut buffer = Batch::new(data, spec.batch);
    let ranges: Vec<_> = batcher.batches().collect();
    for k in 0..TRAIN_WARMUP {
        let view = batcher.gather(data, ranges[k % ranges.len()].clone(), &mut buffer);
        black_box(
            trainer
                .step(&net, &mut params, view)
                .map_err(|e| e.to_string())?,
        );
    }
    // Step k of a timed sample uses batch k modulo the batch count; a whole epoch is every
    // batch once, in order.
    let count = spec.steps.unwrap_or(ranges.len());
    let mut seconds = Vec::with_capacity(spec.epochs);
    for epoch in 1..=spec.epochs {
        epoch_now.set(epoch);
        let first = (epoch - 1) * count;
        let start = Instant::now();
        batcher.epoch();
        for k in first..first + count {
            let view = batcher.gather(data, ranges[k % ranges.len()].clone(), &mut buffer);
            black_box(
                trainer
                    .step(&net, &mut params, view)
                    .map_err(|e| e.to_string())?,
            );
        }
        trainer.end_epoch(epoch).map_err(|e| e.to_string())?;
        seconds.push(start.elapsed().as_secs_f64());
    }
    Ok((seconds, warnings.take()))
}

/// Loads the training data a run needs: MNIST's 60,000 images from `data/mnist/`,
/// CIFAR-10's 50,000 from `data/cifar10/`, or the 1,437 training digits.
fn load_training_data(name: &str) -> Result<Dataset, String> {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let loaded = match name {
        "mnist" => mnist::load(&data.join("mnist")),
        "cifar" => cifar10::load(&data.join("cifar10")),
        _ => digits::load(),
    };
    loaded.map(|(train, _)| train).map_err(|e| e.to_string())
}

/// The network as the script's `--net` argument takes it: `784,1024,256,10`.
fn net_arg(widths: &[usize]) -> String {
    widths
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// What both sides of a training comparison time, in each step and at each epoch's end.
pub const TRAIN_TIMED_STEPS: &str = "both sides time the same steps: the batch copy, the \
     forward pass, the loss value with a finite check, dead-ReLU tracking, the backward pass, \
     the update, each parameter's gradient norm against the health bounds, and the \
     end-of-epoch report and reset";

/// What both sides of the latency comparison time.
pub const LATENCY_TIMED_STEPS: &str =
    "both sides time back-to-back single-input forward passes, biases added";

/// Builds a comparison from cavitation's times and numpy's runs.
fn comparison(
    name: String,
    detail: String,
    ours: Vec<f64>,
    numpy: numpy::NumpyRuns,
    timing_note: &str,
    (work, unit): (f64, Unit),
    health: String,
) -> Comparison {
    Comparison {
        name,
        detail,
        ours,
        numpy: numpy.best.samples,
        numpy_default: numpy.default.samples,
        numpy_setting: numpy.best_label,
        numpy_tried: numpy.tried,
        work,
        timing_note: timing_note.to_string(),
        numpy_version: numpy.best.numpy_version,
        blas: numpy.best.blas,
        blas_default: numpy.default.blas,
        numpy_warnings: numpy.best.warnings,
        unit,
        health,
    }
}

/// Runs a training comparison: cavitation first, then the numpy script with the same
/// network, batch, data, optimiser, warm-up, epoch count and steps per timed sample, at
/// numpy's default settings and, with `numpy_best`, at the others
/// [`numpy::run_settings`] tries.
///
/// # Errors
///
/// Returns a message if either side fails.
pub fn compare_training(
    spec: &TrainSpec,
    python: &str,
    numpy_best: bool,
) -> Result<Comparison, String> {
    let data = load_training_data(spec.data)?;
    let (ours, health) = train_ours(spec, &data)?;
    let rows = data.len();
    let epoch_steps = rows.div_ceil(spec.batch);
    let steps = spec.steps.unwrap_or(epoch_steps);
    // The numpy child loads its own copy; CIFAR-10's is 614 MB as f32.
    drop(data);
    let mut args: Vec<String> = [
        "--mode",
        "train",
        "--net",
        &net_arg(&spec.widths),
        "--batch",
        &spec.batch.to_string(),
        "--data",
        spec.data,
        "--optimizer",
        if spec.adam { "adam" } else { "sgd" },
        "--epochs",
        &spec.epochs.to_string(),
        "--warmup",
        &TRAIN_WARMUP.to_string(),
    ]
    .map(String::from)
    .to_vec();
    if let Some(n) = spec.steps {
        args.extend(["--steps".to_string(), n.to_string()]);
    }
    let theirs = numpy::run_settings(python, &args, numpy_best, |r| {
        r.check("train", steps, spec.epochs)
    })?;
    let timed = match spec.steps {
        Some(n) => format!("runs of {n} steps ({epoch_steps} steps per epoch)"),
        None => format!("epochs of {epoch_steps} steps"),
    };
    let work = if spec.unit == Unit::StepsPerSecond {
        steps as f64
    } else {
        rows as f64
    };
    Ok(comparison(
        spec.name.clone(),
        format!(
            "{} network, batch {}, {}, {} untimed warm-up steps, \
             {} timed {timed}, Health::default() on ({} warnings counted)",
            net_arg(&spec.widths),
            spec.batch,
            if spec.adam {
                "Adam lr 0.001"
            } else {
                "SGD lr 0.01"
            },
            TRAIN_WARMUP,
            spec.epochs,
            health.total()
        ),
        ours,
        theirs,
        TRAIN_TIMED_STEPS,
        (work, spec.unit),
        health.summary(),
    ))
}

/// Calls in each timed round of the latency run.
const LATENCY_CALLS: usize = 10_000;
/// Timed rounds of the latency run.
const LATENCY_ROUNDS: usize = 10;
/// Untimed calls before the timed rounds.
const LATENCY_WARMUP: usize = 1_000;

/// Times one input through the 50k network with `InferenceModel::infer`: warm-up calls, then
/// rounds of back-to-back calls. Returns the seconds of each round.
fn latency_ours() -> Result<Vec<f64>, String> {
    let mut params = ParamStore::new();
    let net = build_net(&mut params, &DIGITS_NET);
    let mut model = InferenceModel::compile(&net, &params).map_err(|e| e.to_string())?;
    let mut rng = Rng::seed(2);
    let input: Vec<f32> = (0..DIGITS_NET[0]).map(|_| rng.normal() as f32).collect();
    for _ in 0..LATENCY_WARMUP {
        black_box(model.infer(black_box(&input)));
    }
    Ok((0..LATENCY_ROUNDS)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..LATENCY_CALLS {
                black_box(model.infer(black_box(&input)));
            }
            start.elapsed().as_secs_f64()
        })
        .collect())
}

/// Runs the latency comparison: cavitation, then the numpy script with the same network,
/// warm-up, rounds and calls, at the numpy settings [`compare_training`] uses.
///
/// # Errors
///
/// Returns a message if either side fails.
pub fn compare_latency(python: &str, numpy_best: bool) -> Result<Comparison, String> {
    let ours = latency_ours()?;
    let args: Vec<String> = [
        "--mode",
        "infer",
        "--net",
        &net_arg(&DIGITS_NET),
        "--calls",
        &LATENCY_CALLS.to_string(),
        "--rounds",
        &LATENCY_ROUNDS.to_string(),
        "--warmup",
        &LATENCY_WARMUP.to_string(),
    ]
    .map(String::from)
    .to_vec();
    let theirs = numpy::run_settings(python, &args, numpy_best, |r| {
        r.check("infer", LATENCY_CALLS, LATENCY_ROUNDS)
    })?;
    Ok(comparison(
        "latency".to_string(),
        format!(
            "{} network, one input per call, {LATENCY_WARMUP} warm-up calls, \
             {LATENCY_ROUNDS} rounds of {LATENCY_CALLS} calls",
            net_arg(&DIGITS_NET)
        ),
        ours,
        theirs,
        LATENCY_TIMED_STEPS,
        (LATENCY_CALLS as f64, Unit::MicrosPerCall),
        String::new(),
    ))
}

/// Tests for the timed runs.
#[cfg(test)]
mod tests {
    use super::*;

    /// Warnings group by kind and place, keeping the first one's epoch, and the summary
    /// names each group once with its count.
    #[test]
    fn health_log_groups_by_kind() {
        let mut log = HealthLog::default();
        assert_eq!(log.summary(), "health: no warnings");
        let dead = |layer| Warning::DeadNeurons {
            epoch: 1,
            layer,
            width: 8,
            neurons: vec![0],
        };
        for layer in [3, 3, 5, 3] {
            log.record(&dead(layer), 1);
        }
        assert_eq!(
            log.summary(),
            "health: 4 warnings: DeadNeurons (layer 3, from epoch 1) x3; \
             DeadNeurons (layer 5, from epoch 1) x1"
        );
        let vanishing = Warning::Vanishing {
            step: 9,
            layer: "linear_2".to_string(),
            norm: 1e-9,
        };
        log.record(&vanishing, 0);
        assert_eq!(log.total(), 5);
        assert_eq!(
            log.summary(),
            "health: 5 warnings: DeadNeurons (layer 3, from epoch 1) x3; \
             DeadNeurons (layer 5, from epoch 1) x1; Vanishing (linear_2, from the warm-up) x1"
        );
    }

    /// The off/off spread is the typical distance of a round from 1, so a noisy run whose
    /// ratios straddle 1 is not met, as a median of the ratios would make it.
    #[test]
    fn aa_spread_measures_noise() {
        // The nine off/off ratios of a run on a busy machine.
        let busy = [
            1.2324, 1.6936, 0.6832, 0.9811, 0.9599, 0.8331, 1.0346, 1.0018, 1.0437,
        ];
        assert!((median(&busy) - 1.0).abs() < 0.01);
        assert!((aa_spread(&busy) - 0.0437).abs() < 1e-9);
        assert!(aa_spread(&busy) > 0.01);
        let quiet = [1.003, 0.998, 1.001, 0.996, 1.004, 0.999, 1.002, 0.997];
        assert!(aa_spread(&quiet) <= 0.01);
    }

    /// The 1M network's eight products do 4,799,488 FLOPs per sample (6 × weights minus the
    /// first layer's input gradient), and the digits network's 269,824.
    #[test]
    fn step_flops_per_sample() {
        let work = |name| {
            shapes(name)
                .unwrap()
                .iter()
                .map(Product::flops)
                .sum::<f64>()
                / 128.0
        };
        assert_eq!(work("mnist128"), 4_799_488.0);
        assert_eq!(work("digits128"), 269_824.0);
        assert!(shapes("cifar").is_err());
    }
}

/// Elements of the one parameter of the `subnormal` run: 1,068,810, the size of the 1M
/// network.
pub const SUBNORMAL_ELEMENTS: usize = 1_068_810;

/// Steps per block of the `subnormal` run's report.
pub const BLOCK: usize = 100;

/// One optimiser's update times from the `subnormal` run.
#[derive(Clone, Debug)]
pub struct SubnormalRun {
    /// The optimiser's name.
    pub name: &'static str,
    /// The median update time, in seconds, of each block of [`BLOCK`] steps.
    pub blocks: Vec<f64>,
    /// The median of the blocks from step 901 on over the median of the blocks of steps 1 to
    /// 700 (`None` when the run is shorter than 1,000 steps).
    pub ratio: Option<f64>,
}

/// Times `steps` updates of `opt` on one parameter of [`SUBNORMAL_ELEMENTS`] elements whose
/// gradient is 1e-3 on the first step and 0 on every later one, so the optimiser's state
/// decays through the subnormal range. Only the update is timed.
fn time_decay<O: Optimizer<f32>>(name: &'static str, mut opt: O, steps: usize) -> SubnormalRun {
    let mut store = ParamStore::new();
    let p = store.add(Tensor::full(&[SUBNORMAL_ELEMENTS], 1.0f32));
    let mut times = Vec::with_capacity(steps);
    for step in 0..steps {
        store.zero_grad();
        let g = if step == 0 { 1e-3 } else { 0.0 };
        store.grads.accumulate(p).0.data_mut().fill(g);
        let start = Instant::now();
        opt.step(&mut store);
        times.push(start.elapsed().as_secs_f64());
    }
    let blocks: Vec<f64> = times.chunks(BLOCK).map(median).collect();
    let ratio = (steps >= 1000).then(|| median(&blocks[9..]) / median(&blocks[..7]));
    SubnormalRun {
        name,
        blocks,
        ratio,
    }
}

/// Runs the `subnormal` benchmark: SGD, SGD with momentum 0.9 and Adam (lr 1e-3) for `steps`
/// steps each, one after the other.
pub fn subnormal(steps: usize) -> Vec<SubnormalRun> {
    vec![
        time_decay("sgd", Sgd::new(0.01), steps),
        time_decay("sgd momentum 0.9", Sgd::new(0.01).with_momentum(0.9), steps),
        time_decay("adam lr 1e-3", Adam::new(1e-3), steps),
    ]
}

/// The layer widths of the `small` run's network: 4 to 32 to 32 to 1, tanh between layers,
/// 1,249 parameters.
pub const SMALL_NET: [usize; 4] = [4, 32, 32, 1];

/// Rows of the `small` run's training data.
pub const SMALL_ROWS: usize = 20_000;

/// The batch size of the `small` run.
pub const SMALL_BATCH: usize = 64;

/// Timed epochs of the `small` run.
pub const SMALL_EPOCHS: usize = 10;

/// What one `small` training run measured.
#[derive(Clone, Copy, Debug)]
pub struct SmallRun {
    /// Wall-clock seconds of the timed epochs.
    pub wall: f64,
    /// CPU seconds (all threads, user plus system) the process used in the timed epochs.
    pub cpu: f64,
    /// The bits of the last step's loss, which must be the same for every thread count.
    pub loss_bits: u32,
}

/// The `small` run's data: [`SMALL_ROWS`] rows of 4 inputs uniform in `[-1, 1)` from seed
/// 3, each with the target `sin(π·x₀)·x₁ + x₂·x₃`.
fn small_data() -> Result<Dataset, String> {
    let mut rng = Rng::seed(3);
    let x: Vec<f32> = (0..SMALL_ROWS * SMALL_NET[0])
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let y: Vec<f32> = x
        .as_chunks::<{ SMALL_NET[0] }>()
        .0
        .iter()
        .map(|&[a, b, c, d]| (std::f32::consts::PI * a).sin() * b + c * d)
        .collect();
    Ok(Dataset {
        x: Tensor::from_vec(x, &[SMALL_ROWS, SMALL_NET[0]]).map_err(|e| e.to_string())?,
        targets: Targets::Values(Tensor::from_vec(y, &[SMALL_ROWS, 1]).map_err(|e| e.to_string())?),
    })
}

/// Trains the `small` network (4 → 32 → tanh → 32 → tanh → 1, Xavier uniform weights from
/// seed 1, zero biases) with mean squared error and Adam (lr 1e-3) in batches of
/// [`SMALL_BATCH`], shuffled with seed 7, through the public path with the default health
/// checks on, at the current thread count: [`TRAIN_WARMUP`] untimed steps, then
/// [`SMALL_EPOCHS`] timed epochs, whose wall and CPU time it returns.
///
/// # Errors
///
/// Returns a message if the training stops or the CPU time cannot be read (only Linux
/// reports it).
pub fn small_once() -> Result<SmallRun, String> {
    let data = small_data()?;
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let mut net = Sequential::new();
    for (i, pair) in SMALL_NET.windows(2).enumerate() {
        net = net.push(Linear::with_init(
            &mut params,
            pair[0],
            pair[1],
            Init::XavierUniform,
            Init::Zeros,
            &mut rng,
        ));
        if i + 2 < SMALL_NET.len() {
            net = net.push(Tanh);
        }
    }
    let optimizer = OptimizerConfig::Adam {
        lr: 1e-3,
        weight_decay: 0.0,
    };
    let mut trainer = Trainer::new(&net, optimizer.build(), Loss::Mse, SMALL_BATCH);
    trainer.set_health(Health {
        sink: Some(Box::new(|_: &Warning| {})),
        ..Health::default()
    });
    let mut batcher = Batcher::new(data.len(), SMALL_BATCH, Some(7));
    let mut buffer = Batch::new(&data, SMALL_BATCH);
    let ranges: Vec<_> = batcher.batches().collect();
    let mut loss = 0.0f32;
    for k in 0..TRAIN_WARMUP {
        let view = batcher.gather(&data, ranges[k % ranges.len()].clone(), &mut buffer);
        loss = trainer
            .step(&net, &mut params, view)
            .map_err(|e| e.to_string())?;
    }
    let cpu = || crate::machine::process_cpu_seconds().ok_or("no CPU time on this system");
    let (cpu0, start) = (cpu()?, Instant::now());
    for epoch in 1..=SMALL_EPOCHS {
        batcher.epoch();
        for range in &ranges {
            let view = batcher.gather(&data, range.clone(), &mut buffer);
            loss = trainer
                .step(&net, &mut params, view)
                .map_err(|e| e.to_string())?;
        }
        trainer.end_epoch(epoch).map_err(|e| e.to_string())?;
    }
    let wall = start.elapsed().as_secs_f64();
    Ok(SmallRun {
        wall,
        cpu: cpu()? - cpu0,
        loss_bits: black_box(loss).to_bits(),
    })
}

/// The training rows the `shrink` run trains on: the first 1,149 of the 1,437 training
/// digits. The other 288 are its validation set.
pub const SHRINK_FIT_ROWS: usize = 1149;

/// Timed rounds of the `shrink` run's latency comparison.
pub const SHRINK_ROUNDS: usize = 301;

/// Calls per model in each timed round of the `shrink` run.
pub const SHRINK_CALLS: usize = 1_000;

/// Untimed calls per model before the `shrink` run's timed rounds.
pub const SHRINK_WARMUP: usize = 20_000;

/// The digits network trained for the `shrink` run, with the rows it was not trained on.
pub struct ShrinkStart {
    /// The trained network, with no masks.
    pub net: Sequential,
    /// Its trained parameters.
    pub params: ParamStore,
    /// The rows it was trained on, which shrinking retrains on.
    pub fit_rows: Dataset,
    /// The held-out training rows: the validation set.
    pub validation: Dataset,
    /// The 360 test digits, used only to check the results.
    pub test: Dataset,
}

/// Trains the 64 → 256 → 128 → 10 ReLU network as the digits known result does (He uniform
/// weights, zero biases, `Rng::seed(1)`, Adam lr 0.002, batch 32, 20 epochs, shuffle seed
/// 7), on the first [`SHRINK_FIT_ROWS`] training rows, with quiet health checks.
///
/// # Errors
///
/// Returns a message if the data cannot be loaded or training stops.
pub fn shrink_start() -> Result<ShrinkStart, String> {
    let (train, test) = digits::load().map_err(|e| e.to_string())?;
    let (fit_rows, validation) = train.split_at(SHRINK_FIT_ROWS);
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let mut net = Sequential::new();
    for (i, pair) in DIGITS_NET.windows(2).enumerate() {
        net = net.push(Linear::with_init(
            &mut params,
            pair[0],
            pair[1],
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        ));
        if i + 2 < DIGITS_NET.len() {
            net = net.push(Relu);
        }
    }
    let config = FitConfig {
        epochs: 20,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 0.002,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(7),
        schedule: LrSchedule::Constant,
        health: Health::quiet(),
        on_epoch: None,
        validation: None,
    };
    fit(&net, &mut params, &fit_rows, config).map_err(|e| e.to_string())?;
    Ok(ShrinkStart {
        net,
        params,
        fit_rows,
        validation,
        test,
    })
}

/// The rows of `data` a network classifies correctly, from its accuracy.
pub fn rows_right(acc: f64, data: &Dataset) -> usize {
    (acc * data.len() as f64).round() as usize
}

/// One step of a `shrink` run, with the test accuracy of its network.
pub struct CurvePoint {
    /// The step as the run recorded it, with its checkpoint.
    pub step: ShrinkStep,
    /// The test digits its network classifies correctly.
    pub test_right: usize,
}

/// What one `shrink` run found.
pub struct ShrinkRun {
    /// The unpruned network's measures on the validation set.
    pub baseline: Evaluation,
    /// Every step, in order.
    pub curve: Vec<CurvePoint>,
    /// The index of the chosen step in `curve`, if a step met the criterion.
    pub chosen: Option<usize>,
    /// The chosen network (the unpruned one if no step met the criterion), with its masks.
    pub net: Sequential,
    /// The chosen network's parameters.
    pub params: ParamStore,
}

/// Runs `shrink` from `start` with `config`, the validation set and checkpoints filled in,
/// and measures each step's network on the test digits. `start` is not changed.
///
/// # Errors
///
/// Returns a message if the run fails.
pub fn shrink_run(start: &ShrinkStart, config: ShrinkConfig) -> Result<ShrinkRun, String> {
    let mut net = start.net.clone();
    let mut params = start.params.clone();
    let config = ShrinkConfig {
        validation: Some(Validation::Given(start.validation.clone())),
        keep_checkpoints: true,
        ..config
    };
    let report =
        shrink(&mut net, &mut params, &start.fit_rows, config).map_err(|e| e.to_string())?;
    let curve = report
        .steps
        .into_iter()
        .map(|step| {
            let (masks, store) = step.checkpoint.as_ref().expect("checkpoints are kept");
            let mut at = start.net.clone();
            for (linear, mask) in masks {
                at.set_neuron_mask(*linear, mask.clone());
            }
            let test_right = rows_right(accuracy(&at, store, &start.test, 128), &start.test);
            CurvePoint { step, test_right }
        })
        .collect();
    Ok(ShrinkRun {
        baseline: report.baseline,
        curve,
        chosen: report.chosen,
        net,
        params,
    })
}

/// How `a` and `b` compare value by value: the count of values whose bits differ, and the
/// largest absolute difference.
pub fn bit_diff(a: &[f32], b: &[f32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "bit_diff: lengths differ");
    a.iter().zip(b).fold((0, 0.0f32), |(n, worst), (x, y)| {
        (
            n + usize::from(x.to_bits() != y.to_bits()),
            worst.max((x - y).abs()),
        )
    })
}

/// The outputs of `model` on every row of `data`, one `infer` call per row.
pub fn infer_rows(model: &mut InferenceModel, data: &Dataset) -> Vec<f32> {
    let width = data.features();
    data.x
        .data()
        .chunks(width)
        .flat_map(|row| model.infer(row).to_vec())
        .collect()
}

/// The outputs of `model` on all rows of `data` in one `infer_batch` call.
pub fn infer_all(model: &mut InferenceModel, data: &Dataset) -> Vec<f32> {
    model.infer_batch(data.x.view()).data().to_vec()
}

/// The per-call times of the `shrink` run's latency comparison, one list per model.
pub struct Latencies {
    /// For each model, the mean seconds per call of each timed round.
    pub per_call: Vec<Vec<f64>>,
}

/// Times one input through each of `models` with `InferenceModel::infer`: each model is
/// warmed up with [`SHRINK_WARMUP`] calls, then timed in [`SHRINK_ROUNDS`] rounds; each round
/// runs [`SHRINK_CALLS`] back-to-back calls of every model, starting from a different model
/// each round, so that every model sees the same machine state.
pub fn time_infer(models: &mut [InferenceModel], input: &[f32]) -> Latencies {
    for model in models.iter_mut() {
        for _ in 0..SHRINK_WARMUP {
            black_box(model.infer(black_box(input)));
        }
    }
    let n = models.len();
    let mut per_call = vec![Vec::with_capacity(SHRINK_ROUNDS); n];
    for round in 0..SHRINK_ROUNDS {
        for k in 0..n {
            let m = (round + k) % n;
            let model = &mut models[m];
            let start = Instant::now();
            for _ in 0..SHRINK_CALLS {
                black_box(model.infer(black_box(input)));
            }
            per_call[m].push(start.elapsed().as_secs_f64() / SHRINK_CALLS as f64);
        }
    }
    Latencies { per_call }
}

/// The `p`-th share (0 to 1) of `values` by rank, nearest rank below.
pub fn quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).floor() as usize]
}

/// The parameter values `net` holds in `params`, every element of every parameter counted
/// (masked neurons' included).
pub fn elements(net: &Sequential, params: &ParamStore) -> usize {
    net.parameter_names()
        .iter()
        .map(|(_, id)| params.value(*id).data().len())
        .sum()
}

/// The ways the `inspect` run trains, in the order of its first round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InspectMode {
    /// `cavitation_nn::fit`, no inspector: the baseline.
    Off,
    /// The baseline again, for the A/A pair.
    OffAgain,
    /// `cavitation_inspect::fit` with one client reading `/events` on its own thread.
    Reading,
    /// `cavitation_inspect::fit` with one client connected to `/events` and never reading.
    Connected,
}

impl InspectMode {
    /// The four modes in the order of the first round; round `r` starts at mode `r` mod 4.
    pub const ALL: [InspectMode; 4] = [
        InspectMode::Off,
        InspectMode::OffAgain,
        InspectMode::Reading,
        InspectMode::Connected,
    ];

    /// The mode's name in the report.
    pub fn name(self) -> &'static str {
        match self {
            InspectMode::Off => "off",
            InspectMode::OffAgain => "off again",
            InspectMode::Reading => "on, reading",
            InspectMode::Connected => "on, not reading",
        }
    }

    /// Whether the run trains under the inspector.
    pub fn inspected(self) -> bool {
        matches!(self, InspectMode::Reading | InspectMode::Connected)
    }
}

/// Timed epochs in each run of the `inspect` benchmark: the `digits` benchmark's 100.
pub const INSPECT_EPOCHS: usize = 100;

/// Rounds of the `inspect` benchmark: each runs the four modes once. A multiple of 4, so
/// that every mode runs first, second, third and last equally often.
pub const INSPECT_ROUNDS: usize = 8;

/// The off/off spread of the `inspect` benchmark: the median, over the rounds, of
/// `|1 − ratio|`, where each ratio is one round's off time over its off-again time. Two
/// identical runs give 1 on a quiet machine, so this is how far a typical round strays from
/// 1: the noise a mode's ratio carries too. It is not `|1 − median(ratio)|`, which stays near
/// 0 whenever the noise is symmetric, however large it is.
pub fn aa_spread(ratios: &[f64]) -> f64 {
    let deviations: Vec<f64> = ratios.iter().map(|r| (1.0 - r).abs()).collect();
    median(&deviations)
}

/// The least time the `inspect` benchmark trains, untimed, before its rounds: past the
/// package power-limit window of a low-power CPU (the N5105 runs at 15 W for 27.98 s, then
/// at 10 W), so that every round runs at the same power limit.
pub const INSPECT_WARMUP: Duration = Duration::from_secs(30);

/// The digits the `inspect` benchmark trains on and probes.
pub struct InspectData {
    /// The 1,437 training digits.
    pub train: Dataset,
    /// The 360 test digits: the inspector's probe set.
    pub test: Dataset,
}

/// Loads the digits for the `inspect` benchmark.
///
/// # Errors
///
/// Returns a message if the digits cannot be loaded.
pub fn inspect_data() -> Result<InspectData, String> {
    let (train, test) = digits::load().map_err(|e| e.to_string())?;
    Ok(InspectData { train, test })
}

/// What one run of the `inspect` benchmark measured.
#[derive(Clone, Copy, Debug)]
pub struct InspectRun {
    /// Wall-clock seconds of the `fit` call.
    pub seconds: f64,
    /// Snapshots the inspector captured during the run (0 without an inspector).
    pub snapshots: u64,
    /// Clients connected when training ended (0 without an inspector).
    pub clients_at_end: usize,
    /// The bits of the last epoch's mean loss, the same in every mode.
    pub loss_bits: u64,
}

/// The digits benchmark's training: SGD at 0.01, batch 128, the rows in order, the health
/// checks on with no output, cross-entropy.
fn inspect_fit_config(epochs: usize) -> FitConfig {
    FitConfig {
        epochs,
        batch: 128,
        optimizer: OptimizerConfig::Sgd {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: None,
        schedule: LrSchedule::Constant,
        health: Health::quiet(),
        on_epoch: None,
        validation: None,
    }
}

/// The client side of an inspected run: a reader thread, or a connection that never reads.
enum InspectClient {
    /// A thread reading the event stream until the server closes it.
    Reading(TcpStream, std::thread::JoinHandle<()>),
    /// A connection that asked for the event stream and reads nothing.
    Connected(TcpStream),
}

/// Opens `/events` on the inspector at `url` (`http://<address>/?token=<token>`).
///
/// # Errors
///
/// Returns a message if the connection or the request fails.
fn open_events(url: &str) -> Result<TcpStream, String> {
    let rest = url.strip_prefix("http://").ok_or("the inspector's URL")?;
    let address = &rest[..rest.find('/').ok_or("the inspector's URL")?];
    let token = &url[url.find("token=").ok_or("the inspector's token")? + 6..];
    let mut stream = TcpStream::connect(address).map_err(|e| format!("{address}: {e}"))?;
    let request = format!("GET /events?token={token} HTTP/1.1\r\nHost: {address}\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("{address}: {e}"))?;
    Ok(stream)
}

/// Trains the 50k network (He normal weights from seed 1) on the training digits for
/// `epochs` epochs in `mode`, and times the `fit` call. Under the inspector the 360 test
/// digits are the probe set (8×8 images), with the default `eval_rows` and `snapshot_hz`; the
/// client connects, and the hub has registered it, before the clock starts.
///
/// # Errors
///
/// Returns a message if the inspector cannot start, the client cannot connect within 5 s,
/// or training fails.
pub fn inspect_once(
    mode: InspectMode,
    data: &InspectData,
    epochs: usize,
) -> Result<InspectRun, String> {
    let mut params = ParamStore::new();
    let mut net = build_net(&mut params, &DIGITS_NET);
    let config = inspect_fit_config(epochs);
    if !mode.inspected() {
        let start = Instant::now();
        let history = fit(&net, &mut params, &data.train, config).map_err(|e| e.to_string())?;
        let seconds = start.elapsed().as_secs_f64();
        return Ok(InspectRun {
            seconds,
            snapshots: 0,
            clients_at_end: 0,
            loss_bits: last_loss_bits(&history),
        });
    }
    let mut inspector = Inspector::start(InspectorConfig {
        address: "127.0.0.1:0".parse().expect("a socket address"),
        probe: Some(data.test.clone()),
        input_image: Some((8, 8)),
        title: "digits benchmark".to_string(),
        ..InspectorConfig::default()
    })
    .map_err(|e| e.to_string())?;
    let stream = open_events(&inspector.url())?;
    let client = match mode {
        InspectMode::Reading => {
            let mut reader = stream.try_clone().map_err(|e| e.to_string())?;
            let thread = std::thread::spawn(move || {
                let mut buffer = vec![0u8; 1 << 16];
                while matches!(reader.read(&mut buffer), Ok(n) if n > 0) {}
            });
            InspectClient::Reading(stream, thread)
        }
        _ => InspectClient::Connected(stream),
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while inspector.clients() == 0 {
        if Instant::now() > deadline {
            return Err("the client did not connect within 5 s".to_string());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let before = inspector.snapshots_taken();
    let start = Instant::now();
    let history =
        cavitation_inspect::fit(&mut net, &mut params, &data.train, config, &mut inspector)
            .map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    let run = InspectRun {
        seconds,
        snapshots: inspector.snapshots_taken() - before,
        clients_at_end: inspector.clients(),
        loss_bits: last_loss_bits(&history),
    };
    // Closing the client first lets the inspector's drop finish without waiting for it.
    match client {
        InspectClient::Reading(stream, thread) => {
            drop(inspector);
            let _ = stream.shutdown(Shutdown::Both);
            let _ = thread.join();
        }
        InspectClient::Connected(stream) => {
            let _ = stream.shutdown(Shutdown::Both);
            drop(inspector);
        }
    }
    Ok(run)
}

/// The bits of the last epoch's mean loss in `history`.
fn last_loss_bits(history: &History) -> u64 {
    history
        .epochs
        .last()
        .map_or(0, |report| report.mean_loss.to_bits())
}

/// The slice lengths the `activation` run times: one hidden layer of a small controller,
/// and one batch of a wide layer.
pub const ACTIVATION_LENGTHS: [usize; 2] = [64, 4096];

/// Timed rounds of the `activation` run's kernel timings.
pub const ACTIVATION_ROUNDS: usize = 301;

/// An activation computed over a slice in place.
pub type SliceFn = Box<dyn Fn(&mut [f32])>;

/// One way to compute an activation over a slice in place, for the `activation` run.
pub struct ActivationCase {
    /// What it computes and how, for example `"tanh sse4.2"`.
    pub name: String,
    /// The computation.
    pub run: SliceFn,
}

/// The `activation` run's cases: `tanh` and the sigmoid, each as the plain scalar loop the
/// library ran before the SIMD kernels (std's `f32::tanh`; the overflow-free
/// `1 / (1 + e^-x)` or `e^x / (1 + e^x)` with std's `f32::exp`), then through the kernels at
/// every level in `levels`.
pub fn activation_cases(levels: &[Level]) -> Vec<ActivationCase> {
    let mut cases = vec![ActivationCase {
        name: "tanh scalar std".to_string(),
        run: Box::new(|v: &mut [f32]| {
            for x in v {
                *x = x.tanh();
            }
        }),
    }];
    for &level in levels {
        cases.push(ActivationCase {
            name: format!("tanh {level}"),
            run: Box::new(move |v: &mut [f32]| tanh_f32_with(level, v)),
        });
    }
    cases.push(ActivationCase {
        name: "sigmoid scalar std".to_string(),
        run: Box::new(|v: &mut [f32]| {
            for x in v {
                *x = if *x >= 0.0 {
                    1.0 / (1.0 + (-*x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
            }
        }),
    });
    for &level in levels {
        cases.push(ActivationCase {
            name: format!("sigmoid {level}"),
            run: Box::new(move |v: &mut [f32]| sigmoid_f32_with(level, v)),
        });
    }
    cases
}

/// Times every case on `n` values spread evenly over [-4, 4) (each call first copies them
/// into the working buffer, so every call sees the same inputs; the copy is included):
/// [`ACTIVATION_ROUNDS`] rounds, each running every case, starting one case later each
/// round, for enough calls to take about 20 µs. Returns, for each case, the median over
/// rounds of nanoseconds per value.
pub fn time_activations(cases: &[ActivationCase], n: usize) -> Vec<f64> {
    let input: Vec<f32> = (0..n).map(|i| -4.0 + 8.0 * i as f32 / n as f32).collect();
    let mut buf = input.clone();
    let calls = (20_000 / n).max(1);
    for case in cases {
        for _ in 0..calls * 10 {
            buf.copy_from_slice(&input);
            (case.run)(black_box(&mut buf));
        }
    }
    let k = cases.len();
    let mut per_value = vec![Vec::with_capacity(ACTIVATION_ROUNDS); k];
    for round in 0..ACTIVATION_ROUNDS {
        for j in 0..k {
            let c = (round + j) % k;
            let start = Instant::now();
            for _ in 0..calls {
                buf.copy_from_slice(&input);
                (cases[c].run)(black_box(&mut buf));
            }
            per_value[c].push(start.elapsed().as_secs_f64() * 1e9 / (calls * n) as f64);
        }
    }
    per_value.iter().map(|t| median(t)).collect()
}

/// The `activation` run's controller pair: the `small` run's widths (4 → 32 → 32 → 1) with
/// tanh, and the same with ReLU, Xavier uniform weights from seed 1 and zero biases,
/// compiled for inference.
pub fn activation_controllers() -> Result<Vec<InferenceModel>, String> {
    let mut models = Vec::new();
    for tanh in [true, false] {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(1);
        let mut net = Sequential::new();
        for (i, pair) in SMALL_NET.windows(2).enumerate() {
            net = net.push(Linear::with_init(
                &mut params,
                pair[0],
                pair[1],
                Init::XavierUniform,
                Init::Zeros,
                &mut rng,
            ));
            if i + 2 < SMALL_NET.len() {
                net = if tanh { net.push(Tanh) } else { net.push(Relu) };
            }
        }
        models.push(InferenceModel::compile(&net, &params).map_err(|e| e.to_string())?);
    }
    Ok(models)
}
