//! Trains a LeNet-5-style CNN (61,706 parameters) on MNIST with `fit` and Adam, and prints
//! each epoch's test accuracy. This is the network and schedule of `cavitation-bench
//! mnist-cnn`. It needs the data files; run `scripts/data/get-mnist.sh` once to fetch them.
//!
//! The guide's chapter "Image models" walks through this file; the `ANCHOR` comments mark
//! the parts it includes.
//!
//! Run it with `cargo run --release --example mnist_cnn -p cavitation`.

use std::path::Path;
use std::process::ExitCode;

use cavitation::prelude::*;

/// Loads MNIST from `data/mnist/`, trains for 5 epochs and prints the test accuracy.
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
            eprintln!("training failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Builds the network, trains it and prints each epoch's test accuracy.
fn run(train: &Dataset, test: &Dataset) -> Result<(), Error> {
    // ANCHOR: network
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    // Each image is one row of 784 values: one channel of 28 × 28 pixels, row by row.
    let conv1 = Conv2dGeom::with_options(1, 6, (28, 28), (5, 5), (1, 1), (2, 2), (1, 1))?;
    let conv2 = Conv2dGeom::new(6, 16, (14, 14), (5, 5))?;
    let net = Sequential::new()
        .push(Conv2d::new(&mut params, conv1, &mut rng)) // 6 × 28 × 28
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(6, (28, 28), (2, 2))?)) // 6 × 14 × 14
        .push(Conv2d::new(&mut params, conv2, &mut rng)) // 16 × 10 × 10
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(16, (10, 10), (2, 2))?)) // 16 × 5 × 5
        .push(Flatten)
        .push(Linear::new(&mut params, 400, 120, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut params, 120, 84, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut params, 84, 10, &mut rng));
    // ANCHOR_END: network

    // ANCHOR: fit
    let config = FitConfig {
        epochs: 5,
        batch: 64,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        // The dead-ReLU report counts each position of a convolution's output as a neuron,
        // and the blank border of a digit leaves many positions at 0 for a whole epoch.
        health: Health {
            dead_relu: false,
            ..Health::default()
        },
        on_epoch: Some(Box::new(|report, net, params| {
            println!(
                "epoch {}: mean loss {:.4}, test accuracy {:.2}%, {:.1} s",
                report.epoch,
                report.mean_loss,
                accuracy(net, params, test, 500) * 100.0,
                report.seconds
            );
        })),
        ..FitConfig::default()
    };
    fit(&net, &mut params, train, config)?;
    // ANCHOR_END: fit
    Ok(())
}
