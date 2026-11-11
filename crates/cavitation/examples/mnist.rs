//! Trains a 784 -> 1024 -> 256 -> 10 ReLU network (1,068,810 parameters) on MNIST with `fit`,
//! Adam and a cosine learning-rate schedule, and prints the test accuracy. This is the network
//! and schedule of the MNIST known-result test, which reaches 98.63% on the 10,000 test images
//! after 10 epochs. It needs the data files; run `scripts/data/get-mnist.sh` once to fetch
//! them.
//!
//! Run it with `cargo run --release --example mnist -p cavitation`.

use std::path::Path;
use std::process::ExitCode;

use cavitation::prelude::*;

/// Loads MNIST from `data/mnist/`, trains with Adam and cosine decay, prints the test
/// accuracy.
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

/// Trains for 10 epochs with the learning rate decayed by a cosine and prints the test
/// accuracy.
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
        .push(he(784, 1024))
        .push(Relu)
        .push(he(1024, 256))
        .push(Relu)
        .push(he(256, 10));

    // A dead-ReLU warning means a ReLU neuron output exactly 0 for every sample of an epoch,
    // so it passed no signal forward and received no gradient. The warnings are counted, every
    // epoch's, and summarised at the end; without a sink each layer's first report would be
    // printed.
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
        // The rate starts at 1e-3 and falls along a cosine towards 0 over the 10 epochs.
        schedule: LrSchedule::Cosine { min_lr: 0.0 },
        health,
        on_epoch: Some(Box::new(
            |r: &EpochReport, _: &Sequential, _: &ParamStore| {
                println!(
                    "epoch {:2}: mean loss {:.6}, {:.3} s",
                    r.epoch, r.mean_loss, r.seconds
                );
            },
        )),
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
