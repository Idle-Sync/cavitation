//! Trains the 64 -> 256 -> 128 -> 10 digits network while the inspector shows it in the
//! browser, prints the chain rule at one neuron with the probe's numbers, then keeps the
//! trained network open in explore mode, where the page switches neurons off, ranks them and
//! shrinks the network, until the page's Finish button (or 30 s with no page open).
//!
//! The guide's chapter "Watching a network train" walks through this file; the `ANCHOR`
//! comments mark the parts it includes.
//!
//! Run it with `cargo run --release --example inspect_digits -p cavitation --features inspect`
//! and open the address it prints. Training starts when the page connects, or after 60 s
//! without one. `-- --epochs N` trains for `N` epochs instead of 20, to watch it longer.

use std::time::{Duration, Instant};

use cavitation::cavitation_inspect::probe::Probe;
use cavitation::cavitation_inspect::{self, ExploreConfig, Inspector, InspectorConfig};
use cavitation::cavitation_nn::shrink::{ShrinkConfig, Validation};
use cavitation::prelude::*;

/// The digits network of the known result: He uniform weights, zero biases, `Rng::seed(1)`.
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

/// `x` with four significant digits.
fn sig(x: f32) -> String {
    let magnitude = if x == 0.0 {
        0
    } else {
        x.abs().log10().floor() as i32
    };
    let decimals = usize::try_from(3 - magnitude).unwrap_or(0);
    format!("{x:.decimals$}")
}

/// Prints the chain rule at one neuron of the first hidden layer, with the probe's numbers,
/// on the first test digit the network gets wrong (the first one if it gets none wrong): the
/// neuron with the steepest gradient among those that fire on it.
///
/// # Errors
///
/// Returns the probe's error if a row cannot be run.
fn chain_rule(net: &Sequential, params: &ParamStore, test: &Dataset) -> Result<(), Error> {
    // ANCHOR: probe
    let mut probe = Probe::new();
    let mut row = 0;
    for r in 0..test.len() {
        probe.run(net, params, test, r, Loss::CrossEntropy)?;
        if probe.predicted() != probe.label() {
            row = r;
            break;
        }
    }
    probe.run(net, params, test, row, Loss::CrossEntropy)?;
    // Layer 1 is the first ReLU: its z is the first Linear's output, its a the neurons' output.
    let relu = &probe.layers()[1];
    let z = relu.z().unwrap_or(&[]).to_vec();
    let a = relu.a().to_vec();
    let df = relu.df().unwrap_or(&[]).to_vec();
    let grad_a = relu.grad_a().unwrap_or(&[]).to_vec();
    let grad_z = relu.grad_z().unwrap_or(&[]).to_vec();
    // ANCHOR_END: probe
    let Some(k) = (0..a.len())
        .filter(|&k| z[k] > 0.0)
        .max_by(|&i, &j| grad_z[i].abs().total_cmp(&grad_z[j].abs()))
    else {
        println!("no neuron of layer 1 fires on test digit {row}");
        return Ok(());
    };
    println!(
        "test digit {row}: label {}, predicted {}, loss {}",
        probe.label().map_or("?".to_string(), |l| l.to_string()),
        probe.predicted().map_or("?".to_string(), |p| p.to_string()),
        sig(probe.loss())
    );
    println!(
        "layer 1 (ReLU), neuron {k}: z = {}, a = {}, f'(z) = {}",
        sig(z[k]),
        sig(a[k]),
        sig(df[k])
    );
    // What the neuron's output costs is what it does to the 128 neurons of the next Linear
    // (layer 2), each weighted by how it reads this neuron: dL/da = sum over j of
    // w[j][k] * dL/dy[j].
    let next = net.linear(2).expect("layer 2 is a Linear");
    let w = params.value(next.weight()).data();
    let grad_y = probe.layers()[2].grad_a().unwrap_or(&[]).to_vec();
    let inputs = next.inputs();
    let mut terms: Vec<(usize, f32)> = (0..next.outputs())
        .map(|j| (j, w[j * inputs + k] * grad_y[j]))
        .collect();
    let sum: f64 = terms.iter().map(|&(_, t)| f64::from(t)).sum();
    terms.sort_by(|x, y| y.1.abs().total_cmp(&x.1.abs()));
    let largest: Vec<String> = terms
        .iter()
        .take(3)
        .map(|&(j, t)| format!("neuron {j}: {}", sig(t)))
        .collect();
    println!(
        "dL/da = sum over the {} neurons of layer 2 of w[j][{k}] * dL/dy[j] = {} \
         (the probe's dL/da: {}); largest terms {}",
        next.outputs(),
        sig(sum as f32),
        sig(grad_a[k]),
        largest.join(", ")
    );
    println!(
        "dL/dz = dL/da * f'(z) = {} * {} = {} (the probe's dL/dz: {})",
        sig(grad_a[k]),
        sig(df[k]),
        sig(grad_a[k] * df[k]),
        sig(grad_z[k])
    );
    let neuron = probe.neuron(net, params, 1, k)?;
    if let Some(top) = neuron.top_inputs().first() {
        println!(
            "its strongest input is pixel ({}, {}): x = {}, w = {}; dL/dw = dL/dz * x = {} * {} = {}",
            top.input / 8,
            top.input % 8,
            sig(top.x),
            sig(top.weight),
            sig(grad_z[k]),
            sig(top.x),
            sig(grad_z[k] * top.x)
        );
    }
    println!("on the page: probe row {row}, then click neuron {k} of layer 1");
    Ok(())
}

/// Starts the inspector, trains under it, prints the chain rule at one neuron, then serves
/// the trained network in explore mode and saves the page as one file.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut epochs = 20;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match (arg.as_str(), args.next()) {
            ("--epochs", Some(n)) => epochs = n.parse()?,
            _ => return Err(format!("unknown argument `{arg}`; try --epochs N").into()),
        }
    }
    let (all_train, test) = cavitation_datasets::digits::load()?;
    // As in the chapter on shrinking: the last 288 training rows are held out for the shrink
    // tool's stop rule.
    let (train, validation) = all_train.split_at(1149);

    // ANCHOR: start
    let mut inspector = Inspector::start(InspectorConfig {
        probe: Some(test.clone()),
        input_image: Some((8, 8)),
        class_names: Some((0..10).map(|d| d.to_string()).collect()),
        title: "Digits, 64 → 256 → 128 → 10".to_string(),
        ..InspectorConfig::default()
    })?;
    println!("open {}", inspector.url());
    // ANCHOR_END: start
    println!("training starts when the page connects, or in 60 s");
    let waiting = Instant::now();
    while inspector.clients() == 0 && waiting.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(50));
    }

    // ANCHOR: fit
    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    let config = FitConfig {
        epochs,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 0.002,
            weight_decay: 0.0,
        },
        shuffle_seed: Some(7),
        health: Health::quiet(),
        ..FitConfig::default()
    };
    let history = cavitation_inspect::fit(&mut net, &mut params, &train, config, &mut inspector)?;
    // ANCHOR_END: fit
    let last = history.epochs.last().map_or(f64::NAN, |e| e.mean_loss);
    println!(
        "trained {} epochs: last epoch's loss {last:.4}, test accuracy {:.2}%",
        history.epochs.len(),
        accuracy(&net, &params, &test, 128) * 100.0
    );
    chain_rule(&net, &params, &test)?;

    // ANCHOR: explore
    let explore = ExploreConfig {
        loss: Loss::CrossEntropy,
        train: Some(train),
        shrink: Some(Box::new(move || ShrinkConfig {
            validation: Some(Validation::Given(validation.clone())),
            ..ShrinkConfig::default()
        })),
    };
    println!("explore mode: switch, rank and shrink from the page; Finish ends it");
    inspector.explore(&mut net, &mut params, explore)?;
    // ANCHOR_END: explore
    for (layer, linear) in net.linears() {
        let on = net
            .neuron_mask(layer)
            .map_or(linear.outputs(), |m| linear.outputs() - m.off().count());
        println!(
            "Linear at layer {layer}: {on} of {} neurons on",
            linear.outputs()
        );
    }
    println!(
        "test accuracy as the network stands: {:.2}%",
        accuracy(&net, &params, &test, 128) * 100.0
    );
    // ANCHOR: export
    let path = std::env::temp_dir().join("inspect-digits.html");
    inspector.export_html(&path)?;
    // ANCHOR_END: export
    println!("saved the page with the run in it to {}", path.display());
    Ok(())
}
