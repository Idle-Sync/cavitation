//! What the default health checks print to standard error on real runs: a collapsed network
//! prints its dead layers as they get worse, and a healthy run prints nothing. Each run
//! happens in a child process (this test binary run again on one ignored test), so the
//! parent can read the child's standard error.

use std::process::{Command, Output};

use cavitation::prelude::*;

/// Runs the ignored test `name` of this test binary alone in a child process and returns what
/// the child wrote, after checking that it ran and passed.
fn run_child(name: &str) -> Output {
    let exe = std::env::current_exe().expect("the test binary's path");
    // The printed counts depend on rounding, so the child runs the reference level on one
    // thread, whatever level the test run itself uses.
    let out = Command::new(exe)
        .env("CAVITATION_KERNEL", "reference")
        .env("CAVITATION_THREADS", "1")
        .args([
            "--ignored",
            "--exact",
            "--nocapture",
            "--test-threads=1",
            name,
        ])
        .output()
        .expect("the test binary runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "the child should run {name} and pass; stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// The 64 → 256 → 128 → 10 ReLU network of the digits known result (He uniform weights, zero
/// biases, `Rng::seed(1)`), trained on the digits in batches of 32 for `epochs` epochs with
/// `optimizer`, shuffled with seed 7, with the default health checks (which print).
fn train_digits(optimizer: OptimizerConfig, epochs: usize) {
    let (train, _) = cavitation_datasets::digits::load().unwrap();
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
        epochs,
        batch: 32,
        optimizer,
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(7),
        schedule: LrSchedule::Constant,
        health: Health::default(),
        on_epoch: None,
        validation: None,
    };
    fit(&net, &mut params, &train, config).unwrap();
}

/// The lines of `stderr` that report dead neurons of layer `layer`.
fn dead_lines(stderr: &str, layer: usize) -> Vec<&str> {
    let needle = format!("neurons of layer {layer} ");
    stderr.lines().filter(|l| l.contains(&needle)).collect()
}

/// Training the digits network with SGD at a learning rate of 100 collapses it. The default
/// output has exactly one dead line for layer 1, at epoch 2, saying 247 of 256 (96%), and one
/// for layer 3 saying 128 of 128 (100%), not a line for every epoch.
#[test]
fn lr_100_collapse_is_printed() {
    let out = run_child("lr_100_collapse_child");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let first = dead_lines(&stderr, 1);
    assert_eq!(first.len(), 1, "stderr:\n{stderr}");
    assert!(
        first[0].contains("epoch 2: 247 of 256 (96%)"),
        "{}",
        first[0]
    );
    let last = dead_lines(&stderr, 3);
    assert_eq!(last.len(), 1, "stderr:\n{stderr}");
    assert!(last[0].contains("128 of 128 (100%)"), "{}", last[0]);
}

/// Helper for `lr_100_collapse_is_printed`. Ignored by default; it is run in a child process.
#[test]
#[ignore = "run in a child process by lr_100_collapse_is_printed"]
fn lr_100_collapse_child() {
    let sgd = OptimizerConfig::Sgd {
        lr: 100.0,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    train_digits(sgd, 5);
}

/// The known-result run (Adam, learning rate 0.002, 20 epochs) trains fine and prints no
/// health warning, though its layer 3 has about 16 to 18% dead neurons, below the first
/// threshold.
#[test]
fn healthy_digits_prints_nothing() {
    let out = run_child("healthy_digits_child");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("cavitation: warning"), "stderr:\n{stderr}");
}

/// Helper for `healthy_digits_prints_nothing`. Ignored by default; it is run in a child
/// process.
#[test]
#[ignore = "run in a child process by healthy_digits_prints_nothing"]
fn healthy_digits_child() {
    let adam = OptimizerConfig::Adam {
        lr: 0.002,
        weight_decay: 0.0,
    };
    train_digits(adam, 20);
}
