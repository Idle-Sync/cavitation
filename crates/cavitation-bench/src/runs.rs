//! Timed runs: the matrix products of one training step, each timed on its own, and the
//! training and inference runs that are compared with the numpy baseline.

use std::cell::Cell;
use std::hint::black_box;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use cavitation_core::kernels::gemm::{gemm_f32_with, threads_used};
use cavitation_core::kernels::{Level, MatMut, MatRef};
use cavitation_core::{ParamStore, Rng};
use cavitation_datasets::{digits, mnist};
use cavitation_nn::{
    Batch, Batcher, Dataset, Health, InferenceModel, Init, Linear, Loss, OptimizerConfig, Relu,
    Sequential, Trainer, Warning,
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
    /// Samples per epoch (training) or calls per round (latency): the work in each timed sample.
    pub work: f64,
    /// What each side's timed region includes.
    pub timing_note: String,
    /// The numpy version that ran the baseline.
    pub numpy_version: String,
    /// The BLAS thread setting the baseline saw.
    pub blas_threads: String,
    /// How to show a median: `epoch`, `samples/s` or `us/call`.
    pub unit: Unit,
}

/// How a median time is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Seconds per epoch.
    EpochSeconds,
    /// Training samples per second.
    SamplesPerSecond,
    /// Microseconds per call.
    MicrosPerCall,
}

impl Comparison {
    /// The value of a median time `seconds` in this comparison's unit.
    pub fn value(&self, seconds: f64) -> f64 {
        match self.unit {
            Unit::EpochSeconds => seconds,
            Unit::SamplesPerSecond => self.work / seconds,
            Unit::MicrosPerCall => seconds / self.work * 1e6,
        }
    }

    /// The unit's label.
    pub fn unit_label(&self) -> &'static str {
        match self.unit {
            Unit::EpochSeconds => "s/epoch",
            Unit::SamplesPerSecond => "samples/s",
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
    /// `mnist` or `digits`.
    pub data: &'static str,
    /// Layer widths, input first.
    pub widths: [usize; 4],
    /// The batch size.
    pub batch: usize,
    /// Adam with learning rate 1e-3 if true, plain SGD with 0.01 if false.
    pub adam: bool,
    /// Timed epochs.
    pub epochs: usize,
    /// How to show the median.
    pub unit: Unit,
}

/// Times `spec.epochs` epochs of training through the public path (`Trainer`, `Batcher`,
/// the default health checks on) after [`TRAIN_WARMUP`] untimed steps, and returns the seconds of
/// each epoch.
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
        black_box(trainer.step(&net, &mut params, view));
    }
    let mut seconds = Vec::with_capacity(spec.epochs);
    for epoch in 1..=spec.epochs {
        let start = Instant::now();
        batcher.epoch();
        for range in &ranges {
            let view = batcher.gather(data, range.clone(), &mut buffer);
            black_box(trainer.step(&net, &mut params, view));
        }
        trainer.end_epoch(epoch).map_err(|e| e.to_string())?;
        seconds.push(start.elapsed().as_secs_f64());
    }
    Ok((seconds, warnings.get()))
}

/// Loads the training data a run needs: MNIST's 60,000 images from `data/mnist/` or the
/// 1,437 training digits.
fn load_training_data(name: &str) -> Result<Dataset, String> {
    let loaded = if name == "mnist" {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/mnist");
        mnist::load(&dir)
    } else {
        digits::load()
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
/// network, batch, data, optimiser, warm-up and epoch count.
///
/// # Errors
///
/// Returns a message if either side fails.
pub fn compare_training(spec: &TrainSpec, python: &str) -> Result<Comparison, String> {
    let data = load_training_data(spec.data)?;
    let (ours, warnings) = train_ours(spec, &data)?;
    let args: Vec<String> = [
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
    let theirs = numpy::run(python, &args)?;
    let steps = data.len().div_ceil(spec.batch);
    theirs.check("train", steps, spec.epochs)?;
    Ok(Comparison {
        name: spec.name.clone(),
        detail: format!(
            "{} network, batch {}, {} steps per epoch, {}, {} untimed warm-up steps, \
             {} timed epochs, Health::default() on ({} warnings counted)",
            net_arg(&spec.widths),
            spec.batch,
            steps,
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
        work: data.len() as f64,
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
