//! Timed runs: the matrix products of one training step, each timed on its own, and the
//! training and inference runs that are compared with the numpy baseline.

use std::cell::Cell;
use std::hint::black_box;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use cavitation_core::kernels::gemm::{gemm_f32_with, threads_used};
use cavitation_core::kernels::{Level, MatMut, MatRef};
use cavitation_core::{ParamStore, Rng, Tensor};
use cavitation_datasets::{cifar10, digits, mnist};
use cavitation_nn::{
    Adam, Batch, Batcher, Dataset, Health, InferenceModel, Init, Linear, Loss, Optimizer,
    OptimizerConfig, Relu, Sequential, Sgd, Tanh, Targets, Trainer, Warning,
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
    /// Seconds per epoch or per round, as numpy measured them.
    pub numpy: Vec<f64>,
    /// The work in each timed sample: samples per epoch or steps per timed run (training),
    /// or calls per round (latency).
    pub work: f64,
    /// What each side's timed region includes.
    pub timing_note: String,
    /// The numpy version that ran the baseline.
    pub numpy_version: String,
    /// The BLAS thread setting the baseline saw.
    pub blas_threads: String,
    /// How to show a median: `s/epoch`, `samples/s`, `steps/s` or `us/call`.
    pub unit: Unit,
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

    /// How many times faster cavitation is: numpy's median time over ours.
    pub fn ratio(&self) -> f64 {
        median(&self.numpy) / median(&self.ours)
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
/// steps, and returns the seconds of each, with the count of health warnings.
///
/// # Errors
///
/// Returns a message if the training stops (a non-finite value) or the data is missing.
fn train_ours(spec: &TrainSpec, data: &Dataset) -> Result<(Vec<f64>, usize), String> {
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
    // The default checks, with the warnings counted instead of printed to standard error.
    let warnings = Rc::new(Cell::new(0usize));
    let counter = Rc::clone(&warnings);
    trainer.set_health(Health {
        sink: Some(Box::new(move |_: &Warning| counter.set(counter.get() + 1))),
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
    Ok((seconds, warnings.get()))
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

/// Runs a training comparison: cavitation first, then the numpy script with the same
/// network, batch, data, optimiser, warm-up, epoch count and steps per timed sample.
///
/// # Errors
///
/// Returns a message if either side fails.
pub fn compare_training(spec: &TrainSpec, python: &str) -> Result<Comparison, String> {
    let data = load_training_data(spec.data)?;
    let (ours, warnings) = train_ours(spec, &data)?;
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
    let theirs = numpy::run(python, &args)?;
    theirs.check("train", steps, spec.epochs)?;
    let timed = match spec.steps {
        Some(n) => format!("runs of {n} steps ({epoch_steps} steps per epoch)"),
        None => format!("epochs of {epoch_steps} steps"),
    };
    Ok(Comparison {
        name: spec.name.clone(),
        detail: format!(
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
            warnings
        ),
        ours,
        numpy: theirs.samples,
        timing_note: "ours times the batch copy, the step with its loss value and health \
                      checks, and end_epoch; numpy times the gradient step only"
            .to_string(),
        numpy_version: theirs.numpy_version,
        blas_threads: theirs.blas_threads,
        work: if spec.unit == Unit::StepsPerSecond {
            steps as f64
        } else {
            rows as f64
        },
        unit: spec.unit,
    })
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
/// warm-up, rounds and calls.
///
/// # Errors
///
/// Returns a message if either side fails.
pub fn compare_latency(python: &str) -> Result<Comparison, String> {
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
    let theirs = numpy::run(python, &args)?;
    theirs.check("infer", LATENCY_CALLS, LATENCY_ROUNDS)?;
    Ok(Comparison {
        name: "latency".to_string(),
        detail: format!(
            "{} network, one input per call, {LATENCY_WARMUP} warm-up calls, \
             {LATENCY_ROUNDS} rounds of {LATENCY_CALLS} calls",
            net_arg(&DIGITS_NET)
        ),
        ours,
        numpy: theirs.samples,
        timing_note: "both time back-to-back single-input forward passes".to_string(),
        numpy_version: theirs.numpy_version,
        blas_threads: theirs.blas_threads,
        work: LATENCY_CALLS as f64,
        unit: Unit::MicrosPerCall,
    })
}

/// Tests for the timed runs.
#[cfg(test)]
mod tests {
    use super::*;

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
