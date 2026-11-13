//! Trains the 64 -> 256 -> 128 -> 10 digits network on most of the training digits, shrinks
//! it with `shrink` on the rest, compacts the result into a smaller dense network and saves
//! it. With `--ablate` it first switches neurons off one at a time and reports what each
//! costs; with `--lottery` it also retrains the found size twice from the start: once from
//! the kept neurons' initial weights, once from new random weights.
//!
//! The guide's chapter "Shrinking a network" walks through this file; the `ANCHOR`
//! comments mark the parts it includes.
//!
//! Run it with `cargo run --release --example shrink_digits -p cavitation`, adding
//! `-- --ablate`, `-- --lottery` or both for the extra parts.

use cavitation::cavitation_nn::shrink::{ShrinkConfig, Validation, ablate, compact, shrink};
use cavitation::prelude::*;

/// The known-result network for the digits: He uniform weights, zero biases, `Rng::seed(seed)`,
/// with the given hidden widths.
fn network(params: &mut ParamStore, hidden: [usize; 2], seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
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
        .push(he(64, hidden[0]))
        .push(Relu)
        .push(he(hidden[0], hidden[1]))
        .push(Relu)
        .push(he(hidden[1], 10))
}

/// The known result's training: Adam at 0.002, batch 32, 20 epochs, shuffle seed 7.
fn train(net: &Sequential, params: &mut ParamStore, data: &Dataset) -> Result<(), Error> {
    let config = FitConfig {
        epochs: 20,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 0.002,
            weight_decay: 0.0,
        },
        shuffle_seed: Some(7),
        health: Health::quiet(),
        ..FitConfig::default()
    };
    fit(net, params, data, config).map(|_| ())
}

/// Switches off the neurons of both hidden layers, first one by hand and then every one in
/// turn with `ablate`, and prints what each costs on `held_out`.
fn ablation(net: &mut Sequential, params: &ParamStore, held_out: &Dataset) -> Result<(), Error> {
    // ANCHOR: by_hand
    let before = evaluate(net, params, held_out, Loss::CrossEntropy, 256)?;
    net.set_neuron_mask(0, NeuronMask::from_off(256, &[0]));
    let without = evaluate(net, params, held_out, Loss::CrossEntropy, 256)?;
    net.clear_neuron_mask(0);
    // ANCHOR_END: by_hand
    println!(
        "neuron 0 of layer 0 off: validation loss {:.4} -> {:.4}, accuracy {:.2}% -> {:.2}%",
        before.loss,
        without.loss,
        before.accuracy.unwrap_or(f64::NAN) * 100.0,
        without.accuracy.unwrap_or(f64::NAN) * 100.0
    );
    // ANCHOR: sweep
    let ranked = ablate(net, params, held_out, &[0, 2], Loss::CrossEntropy, 256)?;
    for a in ranked.iter().take(5) {
        println!(
            "layer {} neuron {:3}: {:+.2} points",
            a.linear, a.neuron, a.drop
        );
    }
    // ANCHOR_END: sweep
    let count = |f: fn(f64) -> bool| ranked.iter().filter(|a| f(a.drop)).count();
    println!(
        "of {} neurons: {} cost accuracy alone, {} change no answer, {} help when off",
        ranked.len(),
        count(|d| d > 0.0),
        count(|d| d == 0.0),
        count(|d| d < 0.0)
    );
    Ok(())
}

/// Trains, shrinks, compacts and saves; with `--ablate`, runs the ablation first; with
/// `--lottery`, retrains the found size twice.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let flag = |name: &str| std::env::args().any(|a| a == name);
    let (ablate_first, lottery) = (flag("--ablate"), flag("--lottery"));
    let (all_train, test) = cavitation_datasets::digits::load()?;
    // Hold out the last 288 training rows before the first training: the stop rule must be
    // measured on rows the network has never seen. The test set is only used at the end.
    // ANCHOR: split
    let (train_rows, held_out) = all_train.split_at(1149);
    // ANCHOR_END: split

    let mut params = ParamStore::new();
    let mut net = network(&mut params, [256, 128], 1);
    // ANCHOR: initial
    let initial = params.clone();
    // ANCHOR_END: initial
    train(&net, &mut params, &train_rows)?;
    println!(
        "unpruned: test {:.2}%",
        accuracy(&net, &params, &test, 128) * 100.0
    );
    if ablate_first {
        ablation(&mut net, &params, &held_out)?;
    }

    // ANCHOR: shrink
    let config = ShrinkConfig {
        validation: Some(Validation::Given(held_out)),
        ..ShrinkConfig::default()
    };
    let report = shrink(&mut net, &mut params, &train_rows, config)?;
    for step in &report.steps {
        let acc = step.validation.accuracy.unwrap_or(f64::NAN);
        let (bp, kept, met) = (step.removed_bp, &step.kept, step.met);
        println!(
            "{:>5.1}% removed: kept {kept:?}, validation {:.2}%, met {met}",
            f64::from(bp) / 100.0,
            acc * 100.0
        );
    }
    // ANCHOR_END: shrink
    let Some(chosen) = report.chosen else {
        println!("no step kept validation accuracy within a point; nothing to compact");
        return Ok(());
    };
    let kept = report.steps[chosen].kept.clone();
    println!(
        "chosen: {kept:?}, test {:.2}%",
        accuracy(&net, &params, &test, 128) * 100.0
    );

    // Remove the switched-off neurons for good: a smaller dense network, same outputs.
    // ANCHOR: compact
    let (small, small_params) = compact(&net, &params)?;
    let path = std::env::temp_dir().join("shrunk-digits.safetensors");
    save(&path, &small, &small_params)?;
    // ANCHOR_END: compact
    println!(
        "saved the compacted network ({kept:?} hidden) to {}",
        path.display()
    );

    if lottery {
        // ANCHOR: lottery
        // The kept neurons with the weights they started from, trained alone.
        let (ticket, mut ticket_params) = compact(&net, &initial)?;
        train(&ticket, &mut ticket_params, &train_rows)?;
        let ticket_acc = accuracy(&ticket, &ticket_params, &test, 128);
        println!(
            "kept neurons, initial weights, retrained: test {:.2}%",
            ticket_acc * 100.0
        );
        // ANCHOR_END: lottery
        // The same size from new random weights.
        let mut fresh_params = ParamStore::new();
        let fresh = network(&mut fresh_params, [kept[0], kept[1]], 1);
        train(&fresh, &mut fresh_params, &train_rows)?;
        let fresh_acc = accuracy(&fresh, &fresh_params, &test, 128);
        println!(
            "same size, new random weights: test {:.2}%",
            fresh_acc * 100.0
        );
    }
    Ok(())
}
