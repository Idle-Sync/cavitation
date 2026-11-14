//! Trains a small CNN to read the steering angle off synthetic road images: 32 by 32 grey
//! images of a road edge leaning up to 30° either way (`cavitation_datasets::steering`),
//! with the angle in degrees as the target. It prints the mean absolute error on held-out
//! images after each epoch, then a few test images' angles beside the network's answers.
//!
//! Run it with `cargo run --release --example steering_cnn -p cavitation`.

use cavitation::prelude::*;
use cavitation_datasets::steering;

/// The mean absolute difference between the network's answers on `data` and its angles.
fn mean_abs_error(model: &mut InferenceModel, data: &Dataset) -> f64 {
    let Targets::Values(angles) = &data.targets else {
        unreachable!("steering has value targets")
    };
    let out = model.infer_batch(data.x.view());
    let sum: f64 = (out.data().iter().zip(angles.data()))
        .map(|(&p, &a)| f64::from((p - a).abs()))
        .sum();
    sum / data.len() as f64
}

/// Trains `steering::network` for 8 epochs on 2,000 images and checks it on 500 others.
fn main() -> Result<(), Error> {
    let train = steering::generate(2000, 1);
    let test = steering::generate(500, 2);
    // 1 × 32 × 32 → two strided convolutions → 16 × 4 × 4 after pooling → 32 → the angle.
    let mut params = ParamStore::new();
    let net = steering::network(&mut params, 0);

    let config = FitConfig {
        on_epoch: Some(Box::new(|report, net, params| {
            let mut model = InferenceModel::compile(net, params).expect("the network compiles");
            println!(
                "epoch {}: train loss {:.3}, test mean absolute error {:.2}°",
                report.epoch,
                report.mean_loss,
                mean_abs_error(&mut model, &test)
            );
        })),
        ..steering::recipe(8)
    };
    fit(&net, &mut params, &train, config)?;

    let mut model = InferenceModel::compile(&net, &params)?;
    let Targets::Values(angles) = &test.targets else {
        unreachable!("steering has value targets")
    };
    let out = model.infer_batch(test.x.view()).data().to_vec();
    for (p, a) in out.iter().zip(angles.data()).take(5) {
        println!("angle {a:6.1}°, network {p:6.1}°");
    }
    println!(
        "final test mean absolute error {:.2}°",
        mean_abs_error(&mut model, &test)
    );
    Ok(())
}
