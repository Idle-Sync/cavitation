//! The classification metrics against scikit-learn: each fixture in `tests/fixtures/metrics/`
//! (written by `scripts/fixtures/make_metrics_fixtures.py`) gives the labels or scores and
//! scikit-learn's confusion matrix, accuracy, per-class rates, averages, ROC AUC and average
//! precision. Counts must match exactly, rates to 1e-12.

use std::fs;
use std::path::Path;

use cavitation_nn::metrics::{Average, ConfusionMatrix, average_precision, roc_auc};
use serde_json::Value;

/// The fixtures folder.
fn folder() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metrics")
}

/// Every fixture of `kind`, with its file name.
fn fixtures(kind: &str) -> Vec<(String, Value)> {
    let mut out: Vec<(String, Value)> = fs::read_dir(folder())
        .expect("the metrics fixtures folder")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let v: Value = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
            (p.file_name().unwrap().to_string_lossy().into_owned(), v)
        })
        .filter(|(_, v)| v["kind"] == kind)
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!out.is_empty(), "no {kind} fixtures");
    out
}

/// A JSON array of numbers as f64s.
fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// A JSON array of non-negative integers as usizes.
fn labels(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

/// Asserts `got` is within 1e-12 of `want`, naming the case and the quantity.
fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-12,
        "{what}: {got} against scikit-learn's {want}"
    );
}

/// The confusion matrix and accuracy match exactly, and every per-class rate, F2 included,
/// and every average match scikit-learn to 1e-12.
#[test]
fn class_metrics_match_sklearn() {
    for (name, f) in fixtures("classes") {
        let classes = f["classes"].as_u64().unwrap() as usize;
        let m =
            ConfusionMatrix::from_labels(classes, &labels(&f["targets"]), &labels(&f["predicted"]))
                .unwrap();
        for (t, row) in f["confusion"].as_array().unwrap().iter().enumerate() {
            for (p, want) in labels(row).into_iter().enumerate() {
                assert_eq!(m.count(t, p) as usize, want, "{name}: confusion[{t}][{p}]");
            }
        }
        close(
            m.accuracy(),
            f["accuracy"].as_f64().unwrap(),
            &format!("{name}: accuracy"),
        );
        let per = &f["per_class"];
        let (precision, recall) = (floats(&per["precision"]), floats(&per["recall"]));
        let (f1, f2, support) = (
            floats(&per["f1"]),
            floats(&per["f2"]),
            labels(&per["support"]),
        );
        for (i, c) in labels(&f["labels"]).into_iter().enumerate() {
            close(
                m.precision(c),
                precision[i],
                &format!("{name}: precision({c})"),
            );
            close(m.recall(c), recall[i], &format!("{name}: recall({c})"));
            close(m.f1(c), f1[i], &format!("{name}: f1({c})"));
            close(m.f_beta(c, 2.0), f2[i], &format!("{name}: f2({c})"));
            assert_eq!(m.support(c) as usize, support[i], "{name}: support({c})");
        }
        for (key, avg) in [
            ("macro", Average::Macro),
            ("weighted", Average::Weighted),
            ("micro", Average::Micro),
        ] {
            let a = &f["averages"][key];
            let what = |q: &str| format!("{name}: {q} {key}");
            close(
                m.precision_average(avg),
                a["precision"].as_f64().unwrap(),
                &what("precision"),
            );
            close(
                m.recall_average(avg),
                a["recall"].as_f64().unwrap(),
                &what("recall"),
            );
            close(
                m.f_beta_average(1.0, avg),
                a["f1"].as_f64().unwrap(),
                &what("f1"),
            );
            close(
                m.f_beta_average(2.0, avg),
                a["f2"].as_f64().unwrap(),
                &what("f2"),
            );
        }
    }
}

/// ROC AUC and average precision match scikit-learn to 1e-12, ties included.
#[test]
fn score_metrics_match_sklearn() {
    for (name, f) in fixtures("scores") {
        let scores: Vec<f32> = floats(&f["scores"]).into_iter().map(|s| s as f32).collect();
        let positive: Vec<bool> = f["positive"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_bool().unwrap())
            .collect();
        close(
            roc_auc(&scores, &positive).unwrap(),
            f["roc_auc"].as_f64().unwrap(),
            &format!("{name}: roc_auc"),
        );
        close(
            average_precision(&scores, &positive).unwrap(),
            f["average_precision"].as_f64().unwrap(),
            &format!("{name}: average_precision"),
        );
    }
}
