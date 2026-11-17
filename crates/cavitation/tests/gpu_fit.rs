//! `fit` on the GPU against `fit` on the CPU on the scikit-learn digits, over five seeds: each
//! GPU run ends within 2⁻¹⁶ of the CPU run's training loss and within one held-out row of its
//! accuracy. The differences between two CPU kernel levels on the same seeds are printed
//! beside the GPU's. Without a GPU the test says so and checks nothing;
//! `CAVITATION_GPU_REQUIRED=1` makes that a failure.

#![cfg(feature = "gpu")]

use cavitation_core::ParamStore;
use cavitation_core::kernels::{Level, detected, level, set_level};
use cavitation_datasets::digits;
use cavitation_gpu::{Backend, Gpu};
use cavitation_nn::{Dataset, Device, FitConfig, accuracy, fit};

/// The seeds: each is the network's initial weights and the shuffle of the rows.
const SEEDS: [u64; 5] = [0, 1, 2, 3, 4];
/// The epochs of each run.
const EPOCHS: usize = 5;
/// Student's t at 97.5% with 4 degrees of freedom: the half-width, in standard deviations,
/// of the 95% interval of five differences.
const T_95_OF_5: f64 = 2.776;

/// Trains the 64 → 64 → 32 → 10 digits network from `seed` for [`EPOCHS`] epochs on
/// `device` with the known result's recipe (Adam at 0.002, batch 32, so a last batch of 29),
/// shuffled with `seed`; returns the last epoch's mean training loss and the accuracy on the
/// held-out rows, measured on the CPU.
fn run(seed: u64, device: Device<'_>, train: &Dataset, test: &Dataset) -> (f64, f64) {
    let mut params = ParamStore::new();
    let net = digits::network(&mut params, [64, 32], seed);
    let config = FitConfig {
        shuffle_seed: Some(seed),
        device,
        ..digits::recipe(EPOCHS)
    };
    let history = fit(&net, &mut params, train, config).unwrap();
    assert_eq!(history.epochs.len(), EPOCHS);
    assert!(history.epochs.iter().all(|e| e.rows == train.len()));
    (
        history.epochs[EPOCHS - 1].mean_loss,
        accuracy(&net, &params, test, 128),
    )
}

/// The mean and the sample standard deviation of `values`.
fn mean_and_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

/// `values` in scientific notation, three decimals each.
fn sci(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| format!("{v:.3e}")).collect();
    format!("[{}]", parts.join(", "))
}

/// On seeds 0 to 4, each GPU run of the digits ends within 2⁻¹⁶ of the CPU run's final
/// training loss, and its accuracy on the 360 held-out rows is within one row of the CPU's.
///
/// The test also prints the rule v0.4 accepts `cifar-cnn` by: whether the mean GPU − CPU
/// loss difference is inside the 95% interval of the differences between two CPU kernel
/// levels (the reference kernels against the fastest this machine runs). It is not asserted
/// here. On this small network the two CPU levels end within one `f32` step of each other,
/// so their interval is narrower than any GPU's rounding; measured on an NVIDIA GTX 1660
/// Super, the interval is ±1.3e-8 and the GPU's mean difference 7.3e-8 (Vulkan) and 8.1e-8
/// (DX12), on losses of 0.15 to 0.19, with the CPU's accuracy on every seed.
///
/// The test sets the kernel level, which the whole process shares, so it is the only test
/// in this file.
#[test]
fn digits_on_the_gpu_end_where_the_cpu_does() {
    let gpus: Vec<(Backend, Gpu)> = Backend::on_this_platform()
        .iter()
        .filter_map(|&b| Gpu::with_backend(b).map(|gpu| (b, gpu)))
        .collect();
    if gpus.is_empty() {
        if std::env::var_os("CAVITATION_GPU_REQUIRED").is_some() {
            panic!("CAVITATION_GPU_REQUIRED is set, but no GPU opened");
        }
        println!("no GPU: skipped");
        return;
    }
    let (train, test) = digits::load().unwrap();
    let before = level();
    let fast = detected();
    let at = |kernels: Level| -> Vec<(f64, f64)> {
        set_level(kernels).unwrap();
        SEEDS
            .iter()
            .map(|&seed| run(seed, Device::Cpu, &train, &test))
            .collect()
    };
    let cpu = at(fast);
    let reference = at(Level::Reference);
    set_level(fast).unwrap();
    let losses = |runs: &[(f64, f64)]| runs.iter().map(|r| r.0).collect::<Vec<_>>();
    let accuracies = |runs: &[(f64, f64)]| runs.iter().map(|r| r.1).collect::<Vec<_>>();
    let between_levels: Vec<f64> = reference.iter().zip(&cpu).map(|(r, c)| r.0 - c.0).collect();
    let (mean, sd) = mean_and_sd(&between_levels);
    let (low, high) = (mean - T_95_OF_5 * sd, mean + T_95_OF_5 * sd);
    println!("CPU ({fast:?}) losses {}", sci(&losses(&cpu)));
    println!(
        "CPU, Reference - {fast:?}: loss differences {}, 95% interval [{low:.3e}, {high:.3e}]",
        sci(&between_levels)
    );
    let one_row = 1.0 / test.len() as f64;
    for (backend, gpu) in &gpus {
        let on_gpu: Vec<(f64, f64)> = SEEDS
            .iter()
            .map(|&seed| run(seed, Device::Gpu(gpu), &train, &test))
            .collect();
        let differences: Vec<f64> = on_gpu.iter().zip(&cpu).map(|(g, c)| g.0 - c.0).collect();
        let (gpu_mean, _) = mean_and_sd(&differences);
        println!(
            "{backend} ({}): GPU - CPU loss differences {}, mean {gpu_mean:.3e}, inside the CPU \
             levels' interval: {}; accuracies {:.4?} against the CPU's {:.4?}",
            gpu.info().name,
            sci(&differences),
            (low..=high).contains(&gpu_mean),
            accuracies(&on_gpu),
            accuracies(&cpu),
        );
        for (seed, (g, c)) in SEEDS.iter().zip(on_gpu.iter().zip(&cpu)) {
            assert!(
                (g.0 - c.0).abs() <= c.0 / 65536.0,
                "{backend}, seed {seed}: loss {} against the CPU's {}",
                g.0,
                c.0
            );
            assert!(
                (g.1 - c.1).abs() <= one_row * 1.000_001,
                "{backend}, seed {seed}: accuracy {} against the CPU's {}",
                g.1,
                c.1
            );
        }
    }
    set_level(before).unwrap();
}
