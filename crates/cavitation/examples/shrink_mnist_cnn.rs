//! Trains the LeNet-5-style CNN of `mnist_cnn` on 12,000 MNIST digits, shrinks its two
//! convolutions by channel and its two hidden `Linear` layers by neuron with `shrink`,
//! measured on 3,000 digits it never trained on, compacts the result into a smaller
//! network, and trains a network of the found size from new random weights to compare.
//! It needs the data files; run `scripts/data/get-mnist.sh` once to fetch them.
//!
//! The guide's chapter "Image models" walks through this file; the `ANCHOR` comments mark
//! the parts it includes.
//!
//! Run it with `cargo run --release --example shrink_mnist_cnn -p cavitation`.

use std::path::Path;
use std::process::ExitCode;

use cavitation::cavitation_nn::shrink::{
    Criterion, FromScratch, RetrainConfig, ShrinkConfig, Validation, compact, shrink,
    train_found_size,
};
use cavitation::prelude::*;

/// The rows the network trains and retrains on.
const TRAIN_ROWS: usize = 12_000;
/// The rows after them that `shrink` measures each step on.
const HELD_OUT_ROWS: usize = 3_000;
/// The epochs of the first training.
const EPOCHS: usize = 5;

/// Loads MNIST from `data/mnist/`, then trains, shrinks and compares.
fn main() -> ExitCode {
    // `data/mnist/` at the repository root, wherever the example is started from.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/mnist");
    let (train, test) = match cavitation_datasets::mnist::load(&dir) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("could not load MNIST from {}: {err}", dir.display());
            eprintln!("run scripts/data/get-mnist.sh to download it, then run this example again");
            return ExitCode::FAILURE;
        }
    };
    match run(&train, &test) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// The `mnist_cnn` network: two 5 × 5 convolutions with ReLU and 2 × 2 max pooling, then
/// `Linear` layers of 120, 84 and 10 neurons. Layers 0 and 3 are the convolutions, 7 and
/// 9 the hidden `Linear` layers.
fn lenet(params: &mut ParamStore, rng: &mut Rng) -> Result<Sequential, Error> {
    let conv1 = Conv2dGeom::with_options(1, 6, (28, 28), (5, 5), (1, 1), (2, 2), (1, 1))?;
    let conv2 = Conv2dGeom::new(6, 16, (14, 14), (5, 5))?;
    Ok(Sequential::new()
        .push(Conv2d::new(params, conv1, rng)) // 6 × 28 × 28
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(6, (28, 28), (2, 2))?)) // 6 × 14 × 14
        .push(Conv2d::new(params, conv2, rng)) // 16 × 10 × 10
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(16, (10, 10), (2, 2))?)) // 16 × 5 × 5
        .push(Flatten)
        .push(Linear::new(params, 400, 120, rng))
        .push(Relu)
        .push(Linear::new(params, 120, 84, rng))
        .push(Relu)
        .push(Linear::new(params, 84, 10, rng)))
}

/// Adam at 1e-3 in batches of 64 for `epochs` epochs.
fn recipe(epochs: usize) -> FitConfig<'static> {
    FitConfig {
        epochs,
        batch: 64,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        ..FitConfig::default()
    }
}

/// The number of values in the parameters of `net`.
fn parameter_count(net: &Sequential, params: &ParamStore) -> usize {
    (0..net.len())
        .flat_map(|i| net.layer(i).parameters())
        .map(|(_, id)| params.value(id).data().len())
        .sum()
}

/// Trains, shrinks, compacts and trains the found size from scratch.
fn run(all_train: &Dataset, test: &Dataset) -> Result<(), Error> {
    // ANCHOR: split
    let train = all_train.subset(0..TRAIN_ROWS);
    let held_out = all_train.subset(TRAIN_ROWS..TRAIN_ROWS + HELD_OUT_ROWS);
    // ANCHOR_END: split
    let mut params = ParamStore::new();
    let mut net = lenet(&mut params, &mut Rng::seed(0))?;
    fit(&net, &mut params, &train, recipe(EPOCHS))?;
    println!(
        "trained: {} parameters, test {:.2}%",
        parameter_count(&net, &params),
        accuracy(&net, &params, test, 500) * 100.0
    );

    // ANCHOR: shrink
    let config = ShrinkConfig {
        validation: Some(Validation::Given(held_out)),
        criterion: Some(Criterion::AccuracyDrop { points: 0.5 }),
        retrain: RetrainConfig {
            epochs: 1,
            batch: 64,
            ..RetrainConfig::default()
        },
        ..ShrinkConfig::default()
    };
    let report = shrink(&mut net, &mut params, &train, config)?;
    let baseline = report.baseline.accuracy.unwrap_or(f64::NAN);
    println!("unpruned: validation {:.2}%", baseline * 100.0);
    for step in &report.steps {
        println!(
            "{:>5.1}% removed: kept {:?}, {} parameters, validation {:.2}%, met {}",
            f64::from(step.removed_bp) / 100.0,
            step.kept,
            step.parameters,
            step.validation.accuracy.unwrap_or(f64::NAN) * 100.0,
            step.met
        );
    }
    // ANCHOR_END: shrink
    let Some(chosen) = report.chosen else {
        println!("no step kept validation accuracy within half a point; nothing to compact");
        return Ok(());
    };

    // ANCHOR: compact
    let (small, small_params) = compact(&net, &params)?;
    let channels = |i: usize| {
        small
            .layer(i)
            .as_conv2d()
            .map_or(0, |c| c.geom().out_channels())
    };
    let neurons = |i: usize| small.linear(i).map_or(0, |l| l.outputs());
    println!(
        "compacted: {} and {} channels, {} and {} neurons, {} parameters, test {:.2}%",
        channels(0),
        channels(3),
        neurons(7),
        neurons(9),
        parameter_count(&small, &small_params),
        accuracy(&small, &small_params, test, 500) * 100.0
    );
    // ANCHOR_END: compact

    // ANCHOR: scratch
    // As many epochs as the pruned network had: the first training and one per step.
    let epochs = EPOCHS + chosen + 1;
    let scratch = FromScratch {
        fit: recipe(epochs),
        ..FromScratch::default()
    };
    let found = train_found_size(&net, &params, &train, test, scratch)?;
    println!(
        "{epochs} epochs each: pruned test {:.2}%, same size from new random weights {:.2}%",
        found.pruned.accuracy.unwrap_or(f64::NAN) * 100.0,
        found.scratch.accuracy.unwrap_or(f64::NAN) * 100.0
    );
    // ANCHOR_END: scratch
    Ok(())
}
