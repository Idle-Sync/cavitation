//! Training gives bit-identical parameters on 1 and on 4 threads, dropout masks and
//! convolutions included, and the v0.3 CIFAR network gives bit-identical parameters and
//! batch-norm buffers at 1, 2, 3 and 4 threads and from run to run.
//!
//! This target holds a single test because it sets the process-wide thread count.

mod common;

use cavitation_core::kernels::gemm::threads_used;
use cavitation_core::kernels::{Level, level, set_threads};
use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Rng, Tensor};
use cavitation_nn::{
    BatchNorm2d, Conv2d, Dataset, Dropout, FitConfig, Health, Init, Linear, Loss, LrSchedule,
    MaxPool2d, OptimizerConfig, Relu, Sequential, Targets, fit,
};

/// The batch size of the run: large enough that every product of the step is split.
const BATCH: usize = 128;

/// The 64 → 256 → 128 → 10 ReLU network with He uniform weights, from `Rng::seed(1)`, with
/// dropout (p = 0.2) after each ReLU.
fn network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(1);
    let mut he = |inputs, outputs| {
        Linear::with_init(
            params,
            inputs,
            outputs,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        )
    };
    Sequential::new()
        .push(he(64, 256))
        .push(Relu)
        .push(Dropout::new(0.2).unwrap())
        .push(he(256, 128))
        .push(Relu)
        .push(Dropout::new(0.2).unwrap())
        .push(he(128, 10))
}

/// 640 rows of 64 features in `[0, 1)` with class labels 0 to 9, from `Rng::seed(4)`.
fn dataset() -> Dataset {
    let n = 640;
    let mut rng = Rng::seed(4);
    let x: Vec<f32> = (0..n * 64).map(|_| rng.uniform_f32()).collect();
    Dataset {
        x: Tensor::from_vec(x, &[n, 64]).unwrap(),
        targets: Targets::Classes((0..n).map(|_| rng.below(10) as usize).collect()),
    }
}

/// Trains for 2 epochs with Adam on the current thread count and returns the bits of every
/// parameter value, in order.
fn train() -> Vec<u32> {
    let data = dataset();
    let mut params = ParamStore::new();
    let net = network(&mut params);
    let config = FitConfig {
        epochs: 2,
        batch: BATCH,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(5),
        dropout_seed: 3,
        schedule: LrSchedule::Constant,
        health: Health::default(),
        on_epoch: None,
        validation: None,
        early_stopping: None,
    };
    fit(&net, &mut params, &data, config).unwrap();
    net.parameter_names()
        .into_iter()
        .flat_map(|(_, id)| {
            params
                .value(id)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A small CNN on the same rows read as 1 × 8 × 8 images: a 3 × 3 convolution to 16
/// channels with padding 1, ReLU, dropout, a 3 × 3 convolution to 16 channels with stride 2
/// and no bias, batch normalisation, ReLU, 2 × 2 max pooling and a `Linear` head, trained
/// for one epoch; returns the bits of every parameter and buffer.
fn train_cnn() -> Vec<u32> {
    let data = dataset();
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(6);
    let g1 = Conv2dGeom::with_options(1, 16, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    let g2 = Conv2dGeom::with_options(16, 16, (8, 8), (3, 3), (2, 2), (1, 1), (1, 1)).unwrap();
    let net = Sequential::new()
        .push(Conv2d::new(&mut params, g1, &mut rng))
        .push(Relu)
        .push(Dropout::new(0.2).unwrap())
        .push(Conv2d::with_init(
            &mut params,
            g2,
            Init::TorchDefault,
            None,
            &mut rng,
        ))
        .push(BatchNorm2d::new(&mut params, 16, (4, 4)))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(16, (4, 4), (2, 2)).unwrap()))
        .push(Linear::new(&mut params, 16 * 2 * 2, 10, &mut rng));
    let config = FitConfig {
        epochs: 1,
        batch: BATCH,
        shuffle_seed: Some(5),
        dropout_seed: 3,
        ..FitConfig::default()
    };
    fit(&net, &mut params, &data, config).unwrap();
    net.state_names()
        .into_iter()
        .flat_map(|(_, id)| {
            params
                .value(id)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The CIFAR network of `common` trained for 20 steps of 16 rows (one epoch of 320 seeded
/// rows of 3 × 32 × 32 images in `[0, 1)` with classes 0 to 9) with Adam, shuffled and with
/// dropout; returns the bits of every parameter and buffer.
fn train_cifar() -> Vec<u32> {
    let n = 320;
    let mut rng = Rng::seed(8);
    let x: Vec<f32> = (0..n * 3072).map(|_| rng.uniform_f32()).collect();
    let data = Dataset {
        x: Tensor::from_vec(x, &[n, 3072]).unwrap(),
        targets: Targets::Classes((0..n).map(|_| rng.below(10) as usize).collect()),
    };
    let mut params = ParamStore::new();
    let net = common::cifar_cnn(&mut params, &mut Rng::seed(9));
    let config = FitConfig {
        epochs: 1,
        batch: 16,
        shuffle_seed: Some(5),
        dropout_seed: 3,
        health: Health::quiet(),
        ..FitConfig::default()
    };
    fit(&net, &mut params, &data, config).unwrap();
    net.state_names()
        .into_iter()
        .flat_map(|(_, id)| {
            params
                .value(id)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The same run with `set_threads(1)` and with `set_threads(4)` gives bit-identical
/// parameters. On a CPU with a packed GEMM kernel, every matrix product of a full batch
/// (forward, input gradient and weight gradient of each layer) runs on 4 threads at 4; at
/// the reference and Accelerate levels each product is one call on one thread.
#[test]
fn threads_1_and_4_identical() {
    // (m, n, k) of each product in one step: layer outputs, input gradients, weight gradients.
    let products = [
        (BATCH, 256, 64),
        (BATCH, 128, 256),
        (BATCH, 10, 128),
        (BATCH, 256, 128),
        (BATCH, 128, 10),
        (256, 64, BATCH),
        (128, 256, BATCH),
        (10, 128, BATCH),
    ];
    let level = level();
    if matches!(level, Level::Reference | Level::Accelerate) {
        println!("kernel level is {level}: each product is one call on one thread at any count");
    } else {
        for (m, n, k) in products {
            assert_eq!(
                threads_used(level, 4, m, n, k),
                4,
                "the {m} by {k} times {k} by {n} product does not split across 4 threads"
            );
        }
        println!("kernel level {level}: all 8 products of a step run on 4 threads at 4");
    }
    set_threads(1);
    let one = train();
    set_threads(4);
    let four = train();
    assert_eq!(one.len(), 50_826);
    assert!(one == four, "parameters differ between 1 and 4 threads");
    set_threads(1);
    let one = train_cnn();
    set_threads(4);
    let four = train_cnn();
    assert!(one == four, "CNN parameters differ between 1 and 4 threads");
    set_threads(1);
    let one = train_cifar();
    // 1,117,162 parameters and 388 buffer values (each batch norm's running mean and
    // variance, one value per channel each, and its count).
    assert_eq!(one.len(), 1_117_550);
    for threads in 2..=4 {
        set_threads(threads);
        assert!(
            train_cifar() == one,
            "CIFAR parameters and buffers differ between 1 and {threads} threads"
        );
    }
    assert!(
        train_cifar() == one,
        "CIFAR parameters and buffers differ from run to run"
    );
}
