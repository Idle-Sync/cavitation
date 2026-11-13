//! Known results: real data sets that a correct library must learn to a stated accuracy.

use cavitation_core::{Graph, ParamStore, Reduction, Rng, TensorView};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use cavitation_nn::shrink::{Ranking, ShrinkConfig, Validation, shrink};
use cavitation_nn::{
    Dataset, EpochReport, FitConfig, Health, Init, Linear, Loss, LrSchedule, OptimizerConfig, Relu,
    Sequential, Targets, Warning, accuracy, evaluate, fit,
};

/// The 64 → 256 → 128 → 10 ReLU network (He uniform weights, zero biases, `Rng::seed(1)`),
/// trained with Adam at learning rate 0.002 in batches of 32 for 20 epochs, shuffled with
/// seed 7, reaches at least 97.5% on the 360 held-out scikit-learn digits; `evaluate` on
/// them gives that accuracy and the loss a loop over the batches on the tape gives.
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
        schedule: LrSchedule::Constant,
        health: Health::default(),
        on_epoch: Some(Box::new(
            |r: &EpochReport, _: &Sequential, _: &ParamStore| {
                println!(
                    "epoch {:2}: mean loss {:.6}, {:.3} s",
                    r.epoch, r.mean_loss, r.seconds
                );
            },
        )),
    };
    let history = fit(&net, &mut params, &train, config).unwrap();
    assert_eq!(history.epochs.len(), 20);
    let acc = accuracy(&net, &params, &test, 128);
    println!("digits test accuracy: {:.2}%", acc * 100.0);
    assert!(acc >= 0.975, "digits test accuracy {acc} is below 0.975");
    let evaluation = evaluate(&net, &params, &test, Loss::CrossEntropy, 128).unwrap();
    assert_eq!(evaluation.accuracy, Some(acc));
    let labels = match &test.targets {
        Targets::Classes(labels) => labels,
        Targets::Values(_) => unreachable!("the digits have class labels"),
    };
    let mut graph = Graph::new();
    let mut total = 0.0;
    for start in (0..test.len()).step_by(128) {
        let end = (start + 128).min(test.len());
        let x = TensorView::from_slice(&test.x.data()[start * 64..end * 64], &[end - start, 64])
            .unwrap();
        let mut tape = graph.tape(&params.values);
        let input = tape.input(x);
        let out = net.forward(&mut tape, input);
        let loss = tape.softmax_cross_entropy(out, &labels[start..end], Reduction::Mean);
        total += f64::from(tape.value(loss).data()[0]) * (end - start) as f64;
    }
    let manual = total / test.len() as f64;
    println!("digits test loss: {:.6}", evaluation.loss);
    assert!(
        (evaluation.loss - manual).abs() <= 1e-6 * manual,
        "evaluate's loss {} against the manual loop's {manual}",
        evaluation.loss
    );
}

/// The 784 → 1024 → 256 → 10 ReLU network (He uniform weights, zero biases, `Rng::seed(1)`),
/// trained with Adam starting at learning rate 1e-3 and decayed by a cosine schedule to 0
/// (`LrSchedule::Cosine { min_lr: 0.0 }`), in batches of 128 for 10 epochs, shuffled with
/// seed 7, reaches at least 98.0% on the 10,000 MNIST test images after the 10th epoch.
///
/// Needs `data/mnist/` (run `scripts/data/get-mnist.sh`), so it is ignored by default; run it
/// with `cargo test --release -p cavitation -- --ignored mnist_known_result --nocapture`.
/// The test accuracy is measured after every epoch from `on_epoch` and printed with the
/// epoch's loss; the health warnings are printed after the table.
#[test]
#[ignore = "needs data/mnist/ (scripts/data/get-mnist.sh); trains the 1M network 10 epochs"]
fn mnist_known_result() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/mnist");
    let (train, test) = match cavitation_datasets::mnist::load(&dir) {
        Ok(data) => data,
        // The error's message names the file, and for a missing file the download script.
        Err(err) => panic!("could not load MNIST from {}: {err}", dir.display()),
    };
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
        .push(he(784, 1024))
        .push(Relu)
        .push(he(1024, 256))
        .push(Relu)
        .push(he(256, 10));

    // Warnings are collected, not printed as they come, so the per-epoch table stays readable.
    let warnings: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink = Rc::clone(&warnings);
    let health = Health {
        sink: Some(Box::new(move |w: &Warning| {
            sink.borrow_mut().push(w.to_string())
        })),
        ..Health::default()
    };
    let accuracies: Rc<RefCell<Vec<f64>>> = Rc::default();
    let measured = Rc::clone(&accuracies);
    let epochs = 10;
    let config = FitConfig {
        epochs,
        batch: 128,
        optimizer: OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
        loss: Loss::CrossEntropy,
        shuffle_seed: Some(7),
        schedule: LrSchedule::Cosine { min_lr: 0.0 },
        health,
        on_epoch: Some(Box::new(
            move |r: &EpochReport, net: &Sequential, params: &ParamStore| {
                let acc = accuracy(net, params, &test, 256);
                println!(
                    "epoch {:2}: mean loss {:.6}, test accuracy {:.2}%, {:.3} s",
                    r.epoch,
                    r.mean_loss,
                    acc * 100.0,
                    r.seconds
                );
                measured.borrow_mut().push(acc);
            },
        )),
    };
    fit(&net, &mut params, &train, config).unwrap();

    let warnings = warnings.borrow();
    println!("health: {} warnings", warnings.len());
    for w in warnings.iter() {
        println!("  {w}");
    }
    let accuracies = accuracies.borrow();
    assert_eq!(accuracies.len(), epochs);
    let acc = accuracies[epochs - 1];
    println!("mnist test accuracy: {:.2}%", acc * 100.0);
    assert!(acc >= 0.98, "MNIST test accuracy {acc} is below 0.98");
}

/// The digits known-result network, trained on the first 1,149 of the 1,437 training digits
/// with the known result's recipe, beside the rows it was not trained on.
struct ShrinkStart {
    /// The trained network, with no masks.
    net: Sequential,
    /// Its trained parameters.
    params: ParamStore,
    /// The 1,149 rows it was trained on, which shrinking retrains on.
    fit_rows: Dataset,
    /// The last 288 training rows, held out: the validation set.
    validation: Dataset,
    /// The 360 test digits, used only to check the result.
    test: Dataset,
}

/// Builds the 64 → 256 → 128 → 10 ReLU network as the digits known result does (He uniform
/// weights, zero biases, `Rng::seed(1)`) and trains it the same way (Adam lr 0.002, batch 32,
/// 20 epochs, shuffle seed 7), on the first 1,149 training rows only.
fn shrink_start() -> ShrinkStart {
    let (train, test) = cavitation_datasets::digits::load().unwrap();
    let (fit_rows, validation) = train.split_at(1149);
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
        schedule: LrSchedule::Constant,
        health: Health::quiet(),
        on_epoch: None,
    };
    fit(&net, &mut params, &fit_rows, config).unwrap();
    ShrinkStart {
        net,
        params,
        fit_rows,
        validation,
        test,
    }
}

/// The rows of `data` a network classifies correctly, from its accuracy.
fn rows_right(acc: f64, data: &Dataset) -> usize {
    (acc * data.len() as f64).round() as usize
}

/// What a shrink run chose, checked against the test set.
struct ShrinkOutcome {
    /// The chosen level, in basis points.
    removed_bp: u32,
    /// The hidden neurons kept, both layers together.
    kept: usize,
    /// The test digits the chosen network classifies correctly.
    test_right: usize,
}

/// Runs `shrink` from `start` with `config` (its validation set and checkpoints filled in),
/// prints the curve, one line per step with validation and test accuracy, and returns the
/// chosen step. The model and parameters of `start` are left as they were.
fn shrink_curve(name: &str, start: &ShrinkStart, config: ShrinkConfig) -> ShrinkOutcome {
    let mut net = start.net.clone();
    let mut params = start.params.clone();
    let config = ShrinkConfig {
        validation: Some(Validation::Given(start.validation.clone())),
        keep_checkpoints: true,
        ..config
    };
    let report = shrink(&mut net, &mut params, &start.fit_rows, config).unwrap();
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    let val_rows = start.validation.len();
    let base = report.baseline.accuracy.unwrap();
    println!(
        "{name}: unpruned validation {}/{val_rows} = {:.2}%",
        rows_right(base, &start.validation),
        base * 100.0
    );
    println!("removed  kept     params  validation        val loss  test              met");
    for step in &report.steps {
        let (masks, store) = step.checkpoint.as_ref().unwrap();
        let mut at = start.net.clone();
        for (linear, mask) in masks {
            at.set_neuron_mask(*linear, mask.clone());
        }
        let test_acc = accuracy(&at, store, &start.test, 128);
        let val = step.validation.accuracy.unwrap();
        println!(
            "{:>5.1}%  {:>3}+{:<3} {:>6}  {:>3}/{val_rows} {:>6.2}%  {:>8.4}  {:>3}/{} {:>6.2}%  {}",
            f64::from(step.removed_bp) / 100.0,
            step.kept[0],
            step.kept[1],
            step.parameters,
            rows_right(val, &start.validation),
            val * 100.0,
            step.validation.loss,
            rows_right(test_acc, &start.test),
            start.test.len(),
            test_acc * 100.0,
            if step.met { "yes" } else { "no" }
        );
    }
    let i = report
        .chosen
        .unwrap_or_else(|| panic!("{name}: no step met the criterion"));
    let step = &report.steps[i];
    let test_right = rows_right(accuracy(&net, &params, &start.test, 128), &start.test);
    let kept = step.kept.iter().sum();
    println!(
        "{name}: chose {:.1}% ({}+{}, {} of 384 hidden neurons removed), test {test_right}/{}",
        f64::from(step.removed_bp) / 100.0,
        step.kept[0],
        step.kept[1],
        384 - kept,
        start.test.len()
    );
    ShrinkOutcome {
        removed_bp: step.removed_bp,
        kept,
        test_right,
    }
}

/// The digits known-result network, trained on the first 1,149 training rows, shrinks with
/// the last 288 as its validation set: ranking by outgoing-weight norm without removing
/// never-firing neurons first and running every level, it removes at least 75% of the 384
/// hidden neurons; with the defaults, at least 70%. Each keeps test accuracy no more than 1
/// point below the unpruned network's. Both curves are printed.
///
/// Run it with `cargo test --release -p cavitation --test known_results -- --ignored
/// shrink_known_result --nocapture`.
#[test]
#[ignore = "trains the digits network and shrinks it twice; run in release mode"]
fn shrink_known_result() {
    let start = shrink_start();
    let unpruned = accuracy(&start.net, &start.params, &start.test, 128);
    let unpruned_right = rows_right(unpruned, &start.test);
    println!(
        "unpruned test accuracy: {unpruned_right}/{} = {:.2}%",
        start.test.len(),
        unpruned * 100.0
    );
    let within_a_point = |outcome: &ShrinkOutcome| {
        let lost = unpruned_right as f64 - outcome.test_right as f64;
        lost * 100.0 / start.test.len() as f64 <= 1.0
    };

    let out_norm = ShrinkConfig {
        ranking: Ranking::OutNorm,
        dead_first: false,
        patience: None,
        ..ShrinkConfig::default()
    };
    let first = shrink_curve("out-norm, dead_first off", &start, out_norm);
    let defaults = shrink_curve("defaults", &start, ShrinkConfig::default());

    // At least 75% of the 384 hidden neurons removed is at most 96 kept; at least 70% is at
    // most 115 kept (the 70% level keeps 77 + 39 = 116, since each layer rounds its kept
    // count up, so it removes 69.8%).
    assert!(
        first.kept <= 96,
        "out-norm kept {} hidden neurons at {} bp, more than 96 (75% of 384 removed)",
        first.kept,
        first.removed_bp
    );
    assert!(
        within_a_point(&first),
        "out-norm: test accuracy fell more than 1 point"
    );
    assert!(
        defaults.kept <= 115,
        "defaults kept {} hidden neurons at {} bp, more than 115 (70% of 384 removed)",
        defaults.kept,
        defaults.removed_bp
    );
    assert!(
        within_a_point(&defaults),
        "defaults: test accuracy fell more than 1 point"
    );
}
