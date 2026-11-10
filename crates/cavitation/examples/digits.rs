//! Trains a 64 -> 256 -> 128 -> 10 ReLU network on the 8x8 digits data set with `fit`, prints
//! the loss of every epoch and the test accuracy, then times one inference.
//!
//! Run it with `cargo run --release --example digits -p cavitation`.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use cavitation::prelude::*;

/// Loads the digits, trains with Adam, prints accuracy and the time of one inference.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train, test) = cavitation_datasets::digits::load()?;
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
        .push(he(64, 256))
        .push(Relu)
        .push(he(256, 128))
        .push(Relu)
        .push(he(128, 10));

    // Health checks stay on. A dead-ReLU warning means a ReLU neuron output exactly 0 for
    // every sample of an epoch: it passed no signal forward and received no gradient, so
    // training cannot revive it. The warnings are collected here and summarised at the end
    // instead of printed every epoch.
    let warnings: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink = Rc::clone(&warnings);
    let health = Health {
        sink: Some(Box::new(move |w: &Warning| {
            sink.borrow_mut().push(w.to_string())
        })),
        ..Health::default()
    };

    let config = FitConfig {
        epochs: 20,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 0.002,
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
    fit(&net, &mut params, &train, config)?;

    let acc = accuracy(&net, &params, &test, 128);
    println!(
        "test accuracy: {:.2}% on {} held-out images",
        acc * 100.0,
        test.len()
    );
    let warnings = warnings.borrow();
    match warnings.last() {
        Some(last) => println!("health: {} warnings; last: {last}", warnings.len()),
        None => println!("health: no warnings"),
    }

    // Time one inference: compile the network once, then call it on one image many times
    // and take the mean per call. This is a rough figure from a plain timer loop.
    let mut model = InferenceModel::compile(&net, &params)?;
    let image = &test.x.data()[..64];
    let calls = 20_000;
    let start = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(model.infer(std::hint::black_box(image)));
    }
    let per_call = start.elapsed().as_secs_f64() * 1e6 / f64::from(calls);
    println!("one inference: {per_call:.2} us per call (mean of {calls} calls)");
    Ok(())
}
