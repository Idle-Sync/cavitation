//! Measures a digits classifier beyond accuracy. It trains the 64 → 256 → 128 → 10 network
//! with the digits recipe, then prints its confusion matrix and classification report on
//! the test digits. It then treats "is it a 9?" as a yes-or-no question: ROC AUC and average
//! precision from the network's probability of 9, next to a baseline that always answers
//! "no", whose accuracy looks good and whose recall is 0.
//!
//! The guide's chapter "Measuring a classifier" walks through this file; the `ANCHOR`
//! comments mark the parts it includes.
//!
//! Run it with `cargo run --release --example classification_metrics -p cavitation`.

use cavitation::cavitation_nn::metrics::{Average, ConfusionMatrix, average_precision, roc_auc};
use cavitation::prelude::*;
use cavitation_datasets::digits::{self, network, recipe};

/// Trains on the digits and prints the metrics described in the file's header.
fn main() -> Result<(), Error> {
    let (train, test) = digits::load()?;
    let mut params = ParamStore::new();
    let net = network(&mut params, [256, 128], 1);
    fit(&net, &mut params, &train, recipe(20))?;

    // ANCHOR: report
    let m = confusion_matrix(&net, &params, &test, 256)?;
    println!("{m}");
    // ANCHOR_END: report

    // ANCHOR: nine
    let labels = match &test.targets {
        Targets::Classes(labels) => labels.clone(),
        Targets::Values(_) => unreachable!("the digits have class targets"),
    };
    let is_nine: Vec<bool> = labels.iter().map(|&l| l == 9).collect();
    let mut evaluator = Evaluator::new(&net, &params, Loss::CrossEntropy, 256)?;
    let mut scores = Vec::new();
    evaluator.class_scores(&test, 9, &mut scores)?;
    println!(
        "is it a 9?  network: ROC AUC {:.4}, average precision {:.4}",
        roc_auc(&scores, &is_nine)?,
        average_precision(&scores, &is_nine)?
    );
    // ANCHOR_END: nine

    // ANCHOR: baseline
    let truth: Vec<usize> = is_nine.iter().map(|&n| usize::from(n)).collect();
    let never = ConfusionMatrix::from_labels(2, &truth, &vec![0; truth.len()])?;
    println!(
        "always \"no\": accuracy {:.4}, recall of 9 {:.4}, macro F1 {:.4}",
        never.accuracy(),
        never.recall(1),
        never.f_beta_average(1.0, Average::Macro)
    );
    // ANCHOR_END: baseline
    assert!(m.accuracy() > 0.9, "accuracy {}", m.accuracy());
    Ok(())
}
