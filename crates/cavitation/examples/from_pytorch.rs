//! Shows, on the digits data, the points where a PyTorch user's habits meet Cavitation's
//! conventions: the data split, the mean loss of an epoch whose last batch is short, the
//! activation defaults, and saving and loading in PyTorch's `.safetensors` layout.
//!
//! The guide's chapter "Coming from PyTorch" walks through this file; the `ANCHOR` comments
//! mark the parts it includes.
//!
//! Run it with `cargo run --release --example from_pytorch -p cavitation`.

use cavitation::prelude::*;

/// A 64 -> 32 -> 10 digits network with ReLU between the layers.
fn network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(1);
    Sequential::new()
        .push(Linear::new(params, 64, 32, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 32, 10, &mut rng))
}

/// Runs the four parts in the order the chapter tells them.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: split
    let (train, test) = cavitation_datasets::digits::load()?;
    println!("train {} rows, test {} rows", train.len(), test.len());
    // ANCHOR_END: split

    // 1,437 rows in batches of 100 are 14 full batches and one of 37 rows.
    let batch_size = 100;
    let epoch = FitConfig {
        epochs: 1,
        batch: batch_size,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        shuffle_seed: None,
        health: Health::quiet(),
        ..FitConfig::default()
    };

    // ANCHOR: loop
    // One epoch by hand, keeping each batch's loss (the mean over its rows) and its row count.
    let mut params = ParamStore::new();
    let net = network(&mut params);
    let mut trainer = Trainer::new(&net, Adam::new(1e-3), Loss::CrossEntropy, batch_size);
    let batcher = Batcher::new(train.len(), batch_size, None);
    let mut batch = Batch::new(&train, batch_size);
    let (mut sum_of_means, mut weighted, mut rows, mut batches) = (0.0f64, 0.0f64, 0usize, 0);
    for range in batcher.batches() {
        let n = range.len();
        let view = batcher.gather(&train, range, &mut batch);
        let loss = f64::from(trainer.step(&net, &mut params, view)?);
        sum_of_means += loss; // PyTorch's `total += loss.item()`, divided by the batch count
        weighted += loss * n as f64; // the row-weighted sum
        rows += n;
        batches += 1;
    }
    println!(
        "average of the batch means {:.6}, row-weighted {:.6}",
        sum_of_means / f64::from(batches),
        weighted / rows as f64
    );
    // ANCHOR_END: loop

    // ANCHOR: fit
    let mut fit_params = ParamStore::new();
    let fit_net = network(&mut fit_params);
    let history = fit(&fit_net, &mut fit_params, &train, epoch)?;
    println!("fit's mean_loss {:.6}", history.epochs[0].mean_loss);
    // ANCHOR_END: fit
    assert!((history.epochs[0].mean_loss - weighted / rows as f64).abs() < 1e-9);

    // ANCHOR: defaults
    assert_eq!(LeakyRelu::default().slope, 0.01);
    assert_eq!(Elu::default().alpha, 1.0);
    assert_eq!(Celu::default().alpha, 1.0);
    assert_eq!(Gelu::default().approx, GeluApprox::Exact);
    let softplus = Softplus::default();
    assert_eq!((softplus.beta, softplus.threshold), (1.0, 20.0));
    // ANCHOR_END: defaults

    // ANCHOR: save
    let path = std::env::temp_dir().join("from-pytorch-digits.safetensors");
    save(&path, &fit_net, &fit_params)?;
    for (name, id) in fit_net.parameter_names() {
        println!("{name}: {:?}", fit_params.value(id).shape().dims());
    }
    // A fresh network with other weights takes the file's values.
    let mut loaded_params = ParamStore::new();
    let loaded = network(&mut loaded_params);
    load(&path, &loaded, &mut loaded_params)?;
    let accuracy = |net: &Sequential, params: &ParamStore| {
        evaluate(net, params, &test, Loss::CrossEntropy, 256).map(|e| e.accuracy)
    };
    assert_eq!(
        accuracy(&fit_net, &fit_params)?,
        accuracy(&loaded, &loaded_params)?
    );
    // ANCHOR_END: save
    std::fs::remove_file(&path).ok();
    Ok(())
}
