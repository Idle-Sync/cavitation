//! The gradient path against the exact path: `LinearRegression::fit_gradient` with the
//! default `GradientConfig` reaches the exact least-squares (and ridge) solution of
//! `LinearRegression::fit` within `1e-3 · max(1, |exact|)` per coefficient and intercept.
//!
//! Each fit is 1,000 full-batch steps over 824 rows, a fraction of a second in a debug
//! build, so the test runs in every build.

use std::time::Instant;

use cavitation_ml::{GradientConfig, LinearRegression};
use cavitation_nn::Standardizer;

/// On the Concrete training rows with standardised inputs, OLS and ridge (α = 1) trained by
/// the gradient path with `GradientConfig::default()` agree with the exact solution within
/// `1e-3 · max(1, |exact|)` for every coefficient and intercept.
#[test]
fn concrete_gradient_matches_exact() {
    let (mut train, _) = cavitation_datasets::concrete::load().unwrap();
    Standardizer::fit(train.x.view()).transform_inputs(&mut train);
    for alpha in [0.0, 1.0] {
        let regression = LinearRegression {
            alpha,
            ..LinearRegression::default()
        };
        let exact = regression.fit(&train).unwrap();
        let start = Instant::now();
        let (gradient, history) = regression
            .fit_gradient(&train, GradientConfig::default())
            .unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let ours = gradient.coef.data().iter().chain(&gradient.intercept);
        let exacts = exact.coef.data().iter().chain(&exact.intercept);
        let mut worst = 0.0f64;
        for (i, (&g, &e)) in ours.zip(exacts).enumerate() {
            let gap = (g - e).abs() / e.abs().max(1.0);
            worst = worst.max(gap);
            assert!(
                gap <= 1e-3,
                "alpha {alpha}: value {i} is {g} by gradient, {e} exactly (gap {gap:e})"
            );
        }
        let last = history.epochs.last().unwrap().mean_loss;
        eprintln!(
            "alpha {alpha}: largest gap {worst:e} relative to max(1, |exact|), last epoch's \
             loss {last}, {seconds:.2} s"
        );
    }
}
