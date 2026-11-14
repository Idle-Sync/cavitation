//! Fits the Concrete Compressive Strength data (eight mixture inputs, the strength in MPa) by
//! ordinary least squares and by ridge regression, each two ways: exactly, by QR, and by
//! gradient descent on a one-layer network. It prints the coefficients side by side with
//! their largest difference, the R² on the train and test rows, what happens to the gradient
//! path when the inputs are not standardised, and then folds the standardisation into the
//! network so that it takes raw inputs and answers in MPa, and saves it.
//!
//! The guide's chapter "Regression, exactly and by gradient" walks through this file; the
//! `ANCHOR` comments mark the parts it includes.
//!
//! Run it with `cargo run --release --example linear_regression -p cavitation`.

use cavitation::cavitation_ml::{GradientConfig, LinearModel, LinearRegression};
use cavitation::prelude::*;
use cavitation_datasets::concrete;

/// The largest `|a - b| / max(1, |b|)` over the coefficients and then the intercepts of two
/// models, the measure the library's tests use for "the same answer".
fn largest_difference(a: &LinearModel, b: &LinearModel) -> f64 {
    let ours = a.coef.data().iter().chain(&a.intercept);
    let theirs = b.coef.data().iter().chain(&b.intercept);
    ours.zip(theirs)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// Prints one table row: the name, the exact value, the gradient value and the one of the
/// unscaled data, for `name`'s coefficient.
fn print_row(name: &str, values: [f64; 4]) {
    println!(
        "{name:<18} {:>11.4} {:>11.4} {:>11.4} {:>11.4}",
        values[0], values[1], values[2], values[3]
    );
}

/// Fits OLS and ridge on the Concrete data both ways, prints them side by side, then folds
/// the scaling into a `Sequential` and saves it.
fn main() -> Result<(), Error> {
    let (raw_train, raw_test) = concrete::load()?;
    let Targets::Values(strength) = &raw_train.targets else {
        panic!("Concrete has value targets");
    };

    // ANCHOR: standardise
    // The statistics come from the train rows only; the test rows are scaled with them.
    let x_scale = Standardizer::fit(raw_train.x.view());
    let y_scale = Standardizer::fit(strength.view());
    let mut train = raw_train.clone();
    let mut test = raw_test.clone();
    for data in [&mut train, &mut test] {
        x_scale.transform_inputs(data);
        y_scale.transform_targets(data)?;
    }
    // ANCHOR_END: standardise

    // ANCHOR: fit
    let ols = LinearRegression::default();
    let ridge = LinearRegression {
        alpha: 1.0,
        ..LinearRegression::default()
    };
    let exact_ols = ols.fit(&train)?;
    let exact_ridge = ridge.fit(&train)?;
    let (gradient_ols, history) = ols.fit_gradient(&train, GradientConfig::default())?;
    let (gradient_ridge, _) = ridge.fit_gradient(&train, GradientConfig::default())?;
    // ANCHOR_END: fit

    println!("Coefficients on the standardised data (strength in standard deviations):");
    println!(
        "{:<18} {:>11} {:>11} {:>11} {:>11}",
        "", "OLS exact", "OLS grad", "ridge exact", "ridge grad"
    );
    let names = [
        "cement",
        "blast furnace slag",
        "fly ash",
        "water",
        "superplasticizer",
        "coarse aggregate",
        "fine aggregate",
        "age",
    ];
    for (i, name) in names.iter().enumerate() {
        print_row(
            name,
            [
                exact_ols.coef.data()[i],
                gradient_ols.coef.data()[i],
                exact_ridge.coef.data()[i],
                gradient_ridge.coef.data()[i],
            ],
        );
    }
    print_row(
        "intercept",
        [
            exact_ols.intercept[0],
            gradient_ols.intercept[0],
            exact_ridge.intercept[0],
            gradient_ridge.intercept[0],
        ],
    );
    println!(
        "largest difference, gradient against exact (relative to max(1, |exact|)): \
         OLS {:.1e}, ridge {:.1e}",
        largest_difference(&gradient_ols, &exact_ols),
        largest_difference(&gradient_ridge, &exact_ridge)
    );
    println!(
        "the gradient path trained {} epochs, last mean loss {:.4}",
        history.epochs.len(),
        history.epochs.last().map_or(f64::NAN, |e| e.mean_loss)
    );

    // ANCHOR: unscaled
    // The same OLS fit by gradient descent on the raw inputs and the raw strengths. The health
    // monitor warns of exploding weights on every epoch here; this run silences it.
    let quiet = GradientConfig {
        health: Health::quiet(),
        ..GradientConfig::default()
    };
    let (unscaled, _) = ols.fit_gradient(&raw_train, quiet)?;
    let exact_raw = ols.fit(&raw_train)?;
    // ANCHOR_END: unscaled
    println!(
        "on the raw data the gradient path ends {:.1e} from the exact answer (largest \
         difference as above); the exact path does not care about scale",
        largest_difference(&unscaled, &exact_raw)
    );

    // ANCHOR: score
    println!("R² of the exact OLS fit:");
    println!(
        "  by LinearModel::score        train {:.4}, test {:.4}",
        exact_ols.score(&train)?,
        exact_ols.score(&test)?
    );
    let mut params = ParamStore::new();
    let net = exact_ols.to_sequential(&mut params);
    let on_train = evaluate(&net, &params, &train, Loss::Mse, 256)?;
    let on_test = evaluate(&net, &params, &test, Loss::Mse, 256)?;
    println!(
        "  by evaluate on the Sequential train {:.4}, test {:.4}",
        on_train.r2.unwrap_or(f64::NAN),
        on_test.r2.unwrap_or(f64::NAN)
    );
    // ANCHOR_END: score

    // The unfolded pipeline: standardise the raw test rows, predict, and undo the target scaling.
    let mut scaled_x = raw_test.x.clone();
    x_scale.transform(&mut scaled_x);
    let mut unfolded = exact_ols.predict(scaled_x.view());
    y_scale.inverse_transform(&mut unfolded);

    // ANCHOR: fold
    // After this the network takes raw inputs and answers in MPa. Fold once.
    x_scale.fold_into_input(&net, &mut params)?;
    y_scale.fold_into_output(&net, &mut params)?;
    let folded = evaluate(&net, &params, &raw_test, Loss::Mse, 256)?;
    // ANCHOR_END: fold
    let mut compiled = InferenceModel::compile(&net, &params)?;
    let answers = compiled.infer_batch(raw_test.x.view());
    let worst = answers
        .data()
        .iter()
        .zip(unfolded.data())
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    println!(
        "folded network on the raw test rows: R² {:.4}; its answers differ from the unfolded \
         pipeline's by at most {worst:.1e} MPa",
        folded.r2.unwrap_or(f64::NAN)
    );
    println!(
        "first test row: {:.2} MPa measured, {:.2} MPa predicted",
        match &raw_test.targets {
            Targets::Values(y) => y.data()[0],
            Targets::Classes(_) => f32::NAN,
        },
        answers.data()[0]
    );
    assert!(worst < 1e-3, "folding changed the answers by {worst}");

    // ANCHOR: save
    let name = format!(
        "cavitation-concrete-linear-{}.safetensors",
        std::process::id()
    );
    let path = std::env::temp_dir().join(name);
    save(&path, &net, &params)?;
    println!("saved to {}", path.display());
    // ANCHOR_END: save
    std::fs::remove_file(&path).ok();
    Ok(())
}
