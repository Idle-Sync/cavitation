//! Known results: real data sets that a correct library must learn to a stated accuracy.

use cavitation_core::{ParamStore, Rng};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use cavitation_nn::{
    Adam, Batch, Batcher, EpochReport, FitConfig, Health, Init, Linear, Loss, LrSchedule,
    OptimizerConfig, Relu, Sequential, Trainer, Warning, accuracy, fit,
};

/// The 64 → 256 → 128 → 10 ReLU network (He uniform weights, zero biases, `Rng::seed(1)`),
/// trained with Adam at learning rate 0.002 in batches of 32 for 20 epochs, shuffled with
/// seed 7, reaches at least 97.5% on the 360 held-out scikit-learn digits.
#[test]
fn digits_known_result() {
    let (train, test) = cavitation_datasets::digits::load().unwrap();
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
        health: Health::default(),
        on_epoch: Some(Box::new(
            |r: &EpochReport, _: &Sequential, _: &ParamStore| {
                println!(
                    "epoch {:2}: mean loss {:.6}, {:.3} s",
                    r.epoch, r.mean_loss, r.seconds
                );
            },
        )),
    };
    let history = fit(&net, &mut params, &train, config).unwrap();
    assert_eq!(history.epochs.len(), 20);
    let acc = accuracy(&net, &params, &test, 128);
    println!("digits test accuracy: {:.2}%", acc * 100.0);
    assert!(acc >= 0.975, "digits test accuracy {acc} is below 0.975");
}

/// The 784 → 1024 → 256 → 10 ReLU network (He uniform weights, zero biases, `Rng::seed(1)`),
/// trained with Adam at learning rate 1e-3 in batches of 128 for 10 epochs, shuffled with
/// seed 7, reaches at least 98.0% on the 10,000 MNIST test images after the 10th epoch.
///
/// Needs `data/mnist/` (run `scripts/data/get-mnist.sh`), so it is ignored by default; run it
/// with `cargo test --release -p cavitation -- --ignored mnist_known_result --nocapture`.
/// The loop below is the one `fit` runs (same optimiser, same shuffle order, same health
/// checks), written out so the test accuracy can be measured after every epoch.
#[test]
#[ignore = "needs the MNIST files in data/mnist/ and takes about two minutes"]
fn mnist_known_result() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/mnist");
    let (train, test) = cavitation_datasets::mnist::load(&dir).unwrap();
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

    // Warnings are collected, not printed, so the per-epoch table stays readable.
    let warnings: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink_warnings = Rc::clone(&warnings);
    let health = Health {
        sink: Some(Box::new(move |w: &Warning| {
            sink_warnings.borrow_mut().push(format!("{w:?}"));
        })),
        ..Health::default()
    };
    let (epochs, batch_size) = (10, 128);
    let mut trainer = Trainer::new(&net, Adam::new(1e-3), Loss::CrossEntropy, batch_size);
    trainer.set_health(health);
    let mut batcher = Batcher::new(train.len(), batch_size, Some(7));
    let mut batch = Batch::new(&train, batch_size);
    params.zero_grad();

    let mut accuracies = Vec::with_capacity(epochs);
    for epoch in 1..=epochs {
        let start = Instant::now();
        batcher.epoch();
        let mut total = 0.0f64;
        for range in batcher.batches() {
            let rows = range.len() as f64;
            let view = batcher.gather(&train, range, &mut batch);
            total += f64::from(trainer.step(&net, &mut params, view)) * rows;
        }
        trainer.end_epoch(epoch).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let acc = accuracy(&net, &params, &test, 256);
        println!(
            "epoch {epoch:2}: mean loss {:.6}, test accuracy {:.2}%, {seconds:.3} s",
            total / train.len() as f64,
            acc * 100.0
        );
        accuracies.push(acc);
    }
    println!("health: {} warnings", warnings.borrow().len());
    let acc = accuracies[epochs - 1];
    println!("mnist test accuracy: {:.2}%", acc * 100.0);
    assert!(acc >= 0.98, "MNIST test accuracy {acc} is below 0.98");
}
