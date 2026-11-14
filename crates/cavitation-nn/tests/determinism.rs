//! Training gives bit-identical parameters on 1 and on 4 threads.
//!
//! This target holds a single test because it sets the process-wide thread count.

use cavitation_core::kernels::gemm::threads_used;
use cavitation_core::kernels::{Level, level, set_threads};
use cavitation_core::{ParamStore, Rng, Tensor};
use cavitation_nn::{
    Dataset, FitConfig, Health, Init, Linear, Loss, LrSchedule, OptimizerConfig, Relu, Sequential,
    Targets, fit,
};

/// The batch size of the run: large enough that every product of the step is split.
const BATCH: usize = 128;

/// The 64 → 256 → 128 → 10 ReLU network with He uniform weights, from `Rng::seed(1)`.
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
        .push(he(256, 128))
        .push(Relu)
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
        schedule: LrSchedule::Constant,
        health: Health::default(),
        on_epoch: None,
        validation: None,
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
}
