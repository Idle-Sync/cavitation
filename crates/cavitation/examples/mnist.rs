//! Trains a 784 -> 256 -> 128 -> 10 ReLU network on MNIST with `fit` and prints the test
//! accuracy. It needs the data files; run `scripts/data/get-mnist.sh` once to fetch them.
//!
//! Run it with `cargo run --release --example mnist -p cavitation`.

use std::path::Path;
use std::process::ExitCode;

use cavitation::prelude::*;

/// Loads MNIST from `data/mnist/`, trains with Adam, prints the test accuracy.
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

/// Trains for 10 epochs and prints the test accuracy.
fn run(train: &Dataset, test: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let mut he = |inputs, outputs| {
        Linear::with_init(
            &mut params,
            inputs,
            outputs,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        )
    };
    let net = Sequential::new()
        .push(he(784, 256))
        .push(Relu)
        .push(he(256, 128))
        .push(Relu)
        .push(he(128, 10));

    // A dead-ReLU warning means a ReLU neuron output exactly 0 for every sample of an epoch,
    // so it passed no signal forward and received no gradient. The warnings are counted and
    // summarised at the end rather than printed every epoch.
    let count = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let counter = std::rc::Rc::clone(&count);
    let health = Health {
        sink: Some(Box::new(move |_: &Warning| counter.set(counter.get() + 1))),
        ..Health::default()
    };
    let config = FitConfig {
        epochs: 10,
        batch: 128,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(7),
        health,
        on_epoch: Some(Box::new(|r: &EpochReport| {
            println!(
                "epoch {:2}: mean loss {:.6}, {:.3} s",
                r.epoch, r.mean_loss, r.seconds
            );
        })),
    };
    fit(&net, &mut params, train, config)?;
    let acc = accuracy(&net, &params, test, 256);
    println!(
        "test accuracy: {:.2}% on {} images",
        acc * 100.0,
        test.len()
    );
    println!("health: {} warnings", count.get());
    Ok(())
}
