//! Trains the same digits network twice, on the CPU and on a GPU, from the same weights and
//! the same shuffles, and prints each run's last training loss, test accuracy and time, with
//! the devices this machine offers. Without a GPU it says so and trains on the CPU only.
//!
//! Run it with `cargo run --release --example gpu_digits -p cavitation --features gpu`.

use std::time::Instant;

use cavitation::cavitation_gpu::Gpu;
use cavitation::prelude::*;

// ANCHOR: run
/// Builds the 64 → 64 → 32 → 10 network from `seed` and trains it on `device` for 10 epochs
/// with Adam; returns the run's history and its accuracy on `test`, measured on the CPU.
fn train(
    device: Device<'_>,
    seed: u64,
    train: &Dataset,
    test: &Dataset,
) -> Result<(History, f64), Box<dyn std::error::Error>> {
    let mut params = ParamStore::new();
    let net = cavitation_datasets::digits::network(&mut params, [64, 32], seed);
    let config = FitConfig {
        shuffle_seed: Some(seed),
        device,
        ..cavitation_datasets::digits::recipe(10)
    };
    let history = fit(&net, &mut params, train, config)?;
    // `fit` copied the trained parameters back into `params`, so the CPU measures them.
    let acc = accuracy(&net, &params, test, 128);
    Ok((history, acc))
}
// ANCHOR_END: run

/// Lists the devices, then trains on the CPU and, when one opens, on a GPU.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: devices
    for entry in devices()? {
        println!("{entry}");
    }
    // ANCHOR_END: devices
    let (train_set, test_set) = cavitation_datasets::digits::load()?;
    let seed = 0;

    let start = Instant::now();
    let (cpu, cpu_acc) = train(Device::Cpu, seed, &train_set, &test_set)?;
    report(&cpu, cpu_acc, start);

    // ANCHOR: gpu
    let Some(gpu) = Gpu::new() else {
        println!("no GPU opened: trained on the CPU only");
        return Ok(());
    };
    println!("GPU: {}", gpu.info());
    let start = Instant::now();
    let (on_gpu, gpu_acc) = train(Device::Gpu(&gpu), seed, &train_set, &test_set)?;
    // ANCHOR_END: gpu
    report(&on_gpu, gpu_acc, start);

    let (c, g) = (last_loss(&cpu), last_loss(&on_gpu));
    println!(
        "GPU - CPU last loss: {:.3e} ({:.2e} of the CPU's)",
        g - c,
        (g - c) / c
    );
    Ok(())
}

/// The last epoch's mean training loss.
fn last_loss(history: &History) -> f64 {
    history.epochs.last().map_or(f64::NAN, |e| e.mean_loss)
}

/// Prints where a run trained, its last loss, its accuracy and its time since `start`.
fn report(history: &History, acc: f64, start: Instant) {
    println!(
        "{}: last loss {:.6}, test accuracy {:.2}%, {:.2} s",
        history.device,
        last_loss(history),
        acc * 100.0,
        start.elapsed().as_secs_f64()
    );
}
