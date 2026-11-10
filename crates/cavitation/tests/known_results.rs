//! Known results: real data sets that a correct library must learn to a stated accuracy.

use cavitation_core::{ParamStore, Rng};
use cavitation_nn::{
    EpochReport, FitConfig, Health, Init, Linear, Loss, OptimizerConfig, Relu, Sequential,
    accuracy, fit,
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
        health: Health::default(),
        on_epoch: Some(Box::new(|r: &EpochReport| {
            println!(
                "epoch {:2}: mean loss {:.6}, {:.3} s",
                r.epoch, r.mean_loss, r.seconds
            );
        })),
    };
    let history = fit(&net, &mut params, &train, config).unwrap();
    assert_eq!(history.epochs.len(), 20);
    let acc = accuracy(&net, &params, &test, 128);
    println!("digits test accuracy: {:.2}%", acc * 100.0);
    assert!(acc >= 0.975, "digits test accuracy {acc} is below 0.975");
}
