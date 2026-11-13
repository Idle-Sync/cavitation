//! Classifies the Breast Cancer Wisconsin (Diagnostic) tumours (30 measurements each, 0
//! malignant and 1 benign) with L2-penalised logistic regression, fitted two ways: exactly,
//! by Newton's method, and by gradient descent on a one-layer network. It prints Newton's
//! iterations, the test accuracy and the objective of each path, how far apart the two end,
//! and the same comparison with a weaker penalty (`c = 10`), where the gradient path stops
//! short of the exact answer.
//!
//! The guide's chapter "Regression, exactly and by gradient" walks through this file; the
//! `ANCHOR` comments mark the parts it includes.
//!
//! Run it with `cargo run --release --example logistic_regression -p cavitation`.

use cavitation::cavitation_ml::{GradientConfig, LogisticModel, LogisticRegression};
use cavitation::prelude::*;
use cavitation_datasets::breast_cancer;

/// The largest `|a - b| / max(1, |b|)` over the coefficients and the intercept of two models.
fn largest_difference(a: &LogisticModel, b: &LogisticModel) -> f64 {
    let ours = a.coef.data().iter().chain(&a.intercept);
    let theirs = b.coef.data().iter().chain(&b.intercept);
    ours.zip(theirs)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// Fits `logistic` both ways on `train`, prints how each does on `train` and `test`, and
/// returns the largest difference between the two.
fn compare(logistic: LogisticRegression, train: &Dataset, test: &Dataset) -> Result<f64, Error> {
    // ANCHOR: fit
    let exact = logistic.fit(train)?;
    let (gradient, history) = logistic.fit_gradient(train, GradientConfig::default())?;
    // ANCHOR_END: fit
    let c = logistic.c.unwrap_or(f64::INFINITY);
    println!("c = {c}:");
    println!(
        "  exact    {:>3} Newton iterations      test accuracy {:.4}  objective {:.6}",
        exact.iterations,
        exact.score(test)?,
        exact.objective(train)?
    );
    println!(
        "  gradient {:>3} epochs                 test accuracy {:.4}  objective {:.6}",
        history.epochs.len(),
        gradient.score(test)?,
        gradient.objective(train)?
    );
    let gap = largest_difference(&gradient, &exact);
    println!("  largest coefficient difference (relative to max(1, |exact|)): {gap:.1e}");
    Ok(gap)
}

fn main() -> Result<(), Error> {
    let (mut train, mut test) = breast_cancer::load()?;
    println!(
        "{} train rows, {} test rows, {} measurements each",
        train.len(),
        test.len(),
        train.features()
    );

    // ANCHOR: standardise
    let scale = Standardizer::fit(train.x.view());
    scale.transform_inputs(&mut train);
    scale.transform_inputs(&mut test);
    // ANCHOR_END: standardise

    let default = compare(LogisticRegression::default(), &train, &test)?;
    let weak = LogisticRegression {
        c: Some(10.0),
        ..LogisticRegression::default()
    };
    compare(weak, &train, &test)?;
    assert!(
        default <= 1e-3,
        "the paths should agree at c = 1: {default}"
    );
    Ok(())
}
