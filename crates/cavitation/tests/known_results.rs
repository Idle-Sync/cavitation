//! Known results: real data sets that a correct library must learn to a stated accuracy.

use cavitation_core::{ParamStore, Rng};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use cavitation_nn::{
    EpochReport, FitConfig, Health, Init, Linear, Loss, LrSchedule, OptimizerConfig, Relu,
    Sequential, Warning, accuracy, fit,
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
/// trained with Adam starting at learning rate 1e-3 and decayed by a cosine schedule to 0
/// (`LrSchedule::Cosine { min_lr: 0.0 }`), in batches of 128 for 10 epochs, shuffled with
/// seed 7, reaches at least 98.0% on the 10,000 MNIST test images after the 10th epoch.
///
/// Needs `data/mnist/` (run `scripts/data/get-mnist.sh`), so it is ignored by default; run it
/// with `cargo test --release -p cavitation -- --ignored mnist_known_result --nocapture`.
/// The test accuracy is measured after every epoch from `on_epoch` and printed with the
/// epoch's loss; the health warnings are printed after the table.
#[test]
#[ignore = "needs data/mnist/ (scripts/data/get-mnist.sh); trains the 1M network 10 epochs"]
fn mnist_known_result() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/mnist");
    let (train, test) = match cavitation_datasets::mnist::load(&dir) {
        Ok(data) => data,
        // The error's message names the file, and for a missing file the download script.
        Err(err) => panic!("could not load MNIST from {}: {err}", dir.display()),
    };
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

    // Warnings are collected, not printed as they come, so the per-epoch table stays readable.
    let warnings: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink = Rc::clone(&warnings);
    let health = Health {
        sink: Some(Box::new(move |w: &Warning| {
            sink.borrow_mut().push(w.to_string())
        })),
        ..Health::default()
    };
    let accuracies: Rc<RefCell<Vec<f64>>> = Rc::default();
    let measured = Rc::clone(&accuracies);
    let epochs = 10;
    let config = FitConfig {
        epochs,
        batch: 128,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(7),
        schedule: LrSchedule::Cosine { min_lr: 0.0 },
        health,
        on_epoch: Some(Box::new(
            move |r: &EpochReport, net: &Sequential, params: &ParamStore| {
                let acc = accuracy(net, params, &test, 256);
                println!(
                    "epoch {:2}: mean loss {:.6}, test accuracy {:.2}%, {:.3} s",
                    r.epoch,
                    r.mean_loss,
                    acc * 100.0,
                    r.seconds
                );
                measured.borrow_mut().push(acc);
            },
        )),
    };
    fit(&net, &mut params, &train, config).unwrap();

    let warnings = warnings.borrow();
    println!("health: {} warnings", warnings.len());
    for w in warnings.iter() {
        println!("  {w}");
    }
    let accuracies = accuracies.borrow();
    assert_eq!(accuracies.len(), epochs);
    let acc = accuracies[epochs - 1];
    println!("mnist test accuracy: {:.2}%", acc * 100.0);
    assert!(acc >= 0.98, "MNIST test accuracy {acc} is below 0.98");
}
