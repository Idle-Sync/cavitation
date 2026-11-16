//! Evaluation metrics, as scikit-learn computes them: for classifiers, the confusion matrix,
//! precision, recall and F-beta per class and averaged (macro, weighted, micro), log loss
//! and top-k accuracy, and, for a binary classifier's scores, ROC AUC and average
//! precision; for regression, the mean absolute and squared errors, RMSE and R² per output
//! and averaged ([`RegressionMetrics`]).
//!
//! Accuracy alone hides a model that ignores a rare class: on a set with 95% of one class,
//! always answering that class scores 95%. The per-class rates show it at once, as a recall
//! of 0 for the rare class.
//!
//! Conventions, all scikit-learn's defaults:
//!
//! - A rate whose denominator is 0 (precision of a class never predicted, recall of a class
//!   with no rows) is 0, as `zero_division=0`.
//! - The macro and weighted averages run over the classes that appear among the targets or
//!   the predictions, as scikit-learn's default `labels`.
//! - A row whose prediction is missing (a model output that is all NaN, which
//!   [`evaluate`](crate::evaluate) counts as wrong) counts against its class's recall and
//!   against accuracy, and as no class's false positive.
//!
//! # Examples
//!
//! ```
//! use cavitation_nn::metrics::{Average, ConfusionMatrix};
//! // 3 classes; class 2 is rare and the model never predicts it.
//! let targets = [0, 0, 1, 1, 1, 2];
//! let predicted = [0, 1, 1, 1, 0, 1];
//! let m = ConfusionMatrix::from_labels(3, &targets, &predicted).unwrap();
//! assert_eq!(m.count(1, 1), 2);
//! assert_eq!(m.recall(2), 0.0);
//! assert_eq!(m.precision(0), 0.5);
//! assert!((m.f_beta_average(1.0, Average::Macro) - (0.5 + 4.0 / 7.0 + 0.0) / 3.0).abs() < 1e-12);
//! ```

use std::fmt;

use cavitation_core::Error;

/// How [`ConfusionMatrix`] averages a per-class rate over the classes, as scikit-learn's
/// `average` argument.
///
/// ```
/// use cavitation_nn::metrics::{Average, ConfusionMatrix};
/// let m = ConfusionMatrix::from_labels(2, &[0, 0, 0, 1], &[0, 0, 1, 1]).unwrap();
/// // Micro: the rates of all rows pooled, here 3 right out of 4.
/// assert_eq!(m.precision_average(Average::Micro), 0.75);
/// // Macro: the mean of class 0's 1.0 and class 1's 0.5.
/// assert_eq!(m.precision_average(Average::Macro), 0.75);
/// // Weighted: by each class's rows, 3 · 1.0 and 1 · 0.5 over 4.
/// assert_eq!(m.precision_average(Average::Weighted), 0.875);
/// ```
///
/// The enum is `#[non_exhaustive]`: scikit-learn has more averages (`binary`, `samples`), so
/// a `match` on it outside this crate needs a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Average {
    /// The plain mean of the classes' rates: every class counts the same, however rare.
    Macro,
    /// The mean weighted by each class's number of rows (its support).
    Weighted,
    /// One rate from the true and false positives and negatives of all classes added up.
    Micro,
}

/// Counts of rows by their true class (row) and predicted class (column), plus the rows of
/// each class that had no prediction. Built from labels with [`from_labels`] or
/// [`from_predictions`], or one row at a time with [`add`].
///
/// [`from_labels`]: ConfusionMatrix::from_labels
/// [`from_predictions`]: ConfusionMatrix::from_predictions
/// [`add`]: ConfusionMatrix::add
///
/// ```
/// use cavitation_nn::metrics::ConfusionMatrix;
/// let mut m = ConfusionMatrix::new(2);
/// m.add(0, Some(0));
/// m.add(1, Some(0));
/// m.add(1, None); // no prediction for this row
/// assert_eq!((m.count(1, 0), m.unpredicted(1), m.support(1)), (1, 1, 2));
/// assert_eq!(m.accuracy(), 1.0 / 3.0);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfusionMatrix {
    /// The number of classes.
    classes: usize,
    /// `counts[t * classes + p]`: rows of class `t` predicted as `p`.
    counts: Vec<u64>,
    /// `unpredicted[t]`: rows of class `t` with no prediction.
    unpredicted: Vec<u64>,
}

impl ConfusionMatrix {
    /// An empty matrix for `classes` classes.
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::new(3);
    /// assert_eq!((m.classes(), m.total()), (3, 0));
    /// ```
    pub fn new(classes: usize) -> ConfusionMatrix {
        ConfusionMatrix {
            classes,
            counts: vec![0; classes * classes],
            unpredicted: vec![0; classes],
        }
    }

    /// The matrix of `targets` against `predicted`, row by row, every label below
    /// `classes`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if the two have different lengths or a label is `classes` or
    /// more, naming the row.
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[0, 1, 1], &[0, 0, 1]).unwrap();
    /// assert_eq!((m.count(0, 0), m.count(1, 0), m.count(1, 1)), (1, 1, 1));
    /// assert!(ConfusionMatrix::from_labels(2, &[0, 2], &[0, 1]).is_err());
    /// ```
    pub fn from_labels(
        classes: usize,
        targets: &[usize],
        predicted: &[usize],
    ) -> Result<ConfusionMatrix, Error> {
        check_lengths(targets.len(), predicted.len())?;
        let mut m = ConfusionMatrix::new(classes);
        for (row, (&t, &p)) in targets.iter().zip(predicted).enumerate() {
            m.check_label(row, "target", t)?;
            m.check_label(row, "prediction", p)?;
            m.add(t, Some(p));
        }
        Ok(m)
    }

    /// As [`from_labels`](Self::from_labels), with `None` for a row that has no prediction.
    ///
    /// # Errors
    ///
    /// As [`from_labels`](Self::from_labels).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_predictions(2, &[0, 1], &[Some(0), None]).unwrap();
    /// assert_eq!((m.unpredicted(1), m.recall(1)), (1, 0.0));
    /// ```
    pub fn from_predictions(
        classes: usize,
        targets: &[usize],
        predicted: &[Option<usize>],
    ) -> Result<ConfusionMatrix, Error> {
        check_lengths(targets.len(), predicted.len())?;
        let mut m = ConfusionMatrix::new(classes);
        for (row, (&t, &p)) in targets.iter().zip(predicted).enumerate() {
            m.check_label(row, "target", t)?;
            if let Some(p) = p {
                m.check_label(row, "prediction", p)?;
            }
            m.add(t, p);
        }
        Ok(m)
    }

    /// An error naming `row` if `label` is not a class.
    fn check_label(&self, row: usize, what: &str, label: usize) -> Result<(), Error> {
        if label < self.classes {
            Ok(())
        } else {
            Err(Error::InvalidData(format!(
                "confusion matrix: row {row}'s {what} is {label}, but there are {} classes",
                self.classes
            )))
        }
    }

    /// Counts one row of class `target`, predicted as `predicted` (`None`: no prediction).
    ///
    /// # Panics
    ///
    /// Panics if `target` or `predicted` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let mut m = ConfusionMatrix::new(2);
    /// m.add(1, Some(1));
    /// assert_eq!(m.count(1, 1), 1);
    /// ```
    pub fn add(&mut self, target: usize, predicted: Option<usize>) {
        assert!(
            target < self.classes && predicted.is_none_or(|p| p < self.classes),
            "confusion matrix: target {target} or prediction {predicted:?} is not below {} classes",
            self.classes
        );
        match predicted {
            Some(p) => self.counts[target * self.classes + p] += 1,
            None => self.unpredicted[target] += 1,
        }
    }

    /// The number of classes.
    ///
    /// ```
    /// assert_eq!(cavitation_nn::metrics::ConfusionMatrix::new(4).classes(), 4);
    /// ```
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Every row counted, those without a prediction included.
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_predictions(2, &[0, 1], &[Some(1), None]).unwrap();
    /// assert_eq!(m.total(), 2);
    /// ```
    pub fn total(&self) -> u64 {
        self.counts.iter().sum::<u64>() + self.unpredicted.iter().sum::<u64>()
    }

    /// The rows of class `target` predicted as `predicted`.
    ///
    /// # Panics
    ///
    /// Panics if either is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[1, 1], &[0, 1]).unwrap();
    /// assert_eq!((m.count(1, 0), m.count(1, 1), m.count(0, 0)), (1, 1, 0));
    /// ```
    pub fn count(&self, target: usize, predicted: usize) -> u64 {
        assert!(
            target < self.classes && predicted < self.classes,
            "confusion matrix: ({target}, {predicted}) is outside {} classes",
            self.classes
        );
        self.counts[target * self.classes + predicted]
    }

    /// The rows of class `target` that had no prediction.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_predictions(2, &[0, 0], &[None, Some(0)]).unwrap();
    /// assert_eq!(m.unpredicted(0), 1);
    /// ```
    pub fn unpredicted(&self, target: usize) -> u64 {
        self.unpredicted[target]
    }

    /// The rows whose true class is `class` (scikit-learn's support).
    ///
    /// # Panics
    ///
    /// Panics if `class` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[1, 1, 0], &[0, 1, 0]).unwrap();
    /// assert_eq!((m.support(0), m.support(1)), (1, 2));
    /// ```
    pub fn support(&self, class: usize) -> u64 {
        let row = &self.counts[class * self.classes..][..self.classes];
        row.iter().sum::<u64>() + self.unpredicted[class]
    }

    /// The rows predicted as `class`.
    ///
    /// # Panics
    ///
    /// Panics if `class` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[1, 1, 0], &[0, 1, 0]).unwrap();
    /// assert_eq!((m.predicted(0), m.predicted(1)), (2, 1));
    /// ```
    pub fn predicted(&self, class: usize) -> u64 {
        (0..self.classes).map(|t| self.count(t, class)).sum()
    }

    /// The share of all rows whose prediction is their class; 0 with no rows.
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(3, &[0, 1, 2, 2], &[0, 1, 1, 2]).unwrap();
    /// assert_eq!(m.accuracy(), 0.75);
    /// ```
    pub fn accuracy(&self) -> f64 {
        let right: u64 = (0..self.classes).map(|c| self.count(c, c)).sum();
        ratio(right, self.total())
    }

    /// Of the rows predicted as `class`, the share that are `class`: `tp / (tp + fp)`, 0
    /// if none is predicted as `class`.
    ///
    /// # Panics
    ///
    /// Panics if `class` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[0, 1, 1], &[1, 1, 0]).unwrap();
    /// assert_eq!(m.precision(1), 0.5);
    /// ```
    pub fn precision(&self, class: usize) -> f64 {
        ratio(self.count(class, class), self.predicted(class))
    }

    /// Of the rows of `class`, the share predicted as `class`: `tp / (tp + fn)`, 0 if
    /// `class` has no rows.
    ///
    /// # Panics
    ///
    /// Panics if `class` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[0, 1, 1], &[1, 1, 0]).unwrap();
    /// assert_eq!(m.recall(1), 0.5);
    /// ```
    pub fn recall(&self, class: usize) -> f64 {
        ratio(self.count(class, class), self.support(class))
    }

    /// The F-beta score of `class`, `(1 + β²) · tp / ((1 + β²) · tp + β² · fn + fp)`: the
    /// weighted harmonic mean of precision and recall, recall counting `beta` times as much
    /// (F1 at `beta = 1`, F2 at 2). 0 when the class has neither rows nor predictions.
    ///
    /// # Panics
    ///
    /// Panics if `class` is not below [`classes`](Self::classes), or `beta` is negative or
    /// not finite.
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// // Class 1: tp 1, fn 2, fp 0, so precision 1 and recall 1/3.
    /// let m = ConfusionMatrix::from_labels(2, &[1, 1, 1, 0], &[1, 0, 0, 0]).unwrap();
    /// assert_eq!(m.f_beta(1, 1.0), 0.5);
    /// assert!((m.f_beta(1, 2.0) - 5.0 / 13.0).abs() < 1e-15);
    /// ```
    pub fn f_beta(&self, class: usize, beta: f64) -> f64 {
        let (tp, fp, fne) = self.outcomes(class);
        f_beta_of(tp, fp, fne, check_beta(beta))
    }

    /// The F1 score of `class`: [`f_beta`](Self::f_beta) at `beta = 1`.
    ///
    /// # Panics
    ///
    /// Panics if `class` is not below [`classes`](Self::classes).
    ///
    /// ```
    /// use cavitation_nn::metrics::ConfusionMatrix;
    /// let m = ConfusionMatrix::from_labels(2, &[1, 1, 1, 0], &[1, 0, 0, 0]).unwrap();
    /// assert_eq!(m.f1(1), 0.5);
    /// ```
    pub fn f1(&self, class: usize) -> f64 {
        self.f_beta(class, 1.0)
    }

    /// Precision averaged over the classes by `average` (see [`Average`]); 0 with no rows.
    ///
    /// ```
    /// use cavitation_nn::metrics::{Average, ConfusionMatrix};
    /// let m = ConfusionMatrix::from_labels(2, &[0, 1], &[0, 0]).unwrap();
    /// assert_eq!(m.precision_average(Average::Macro), 0.25);
    /// ```
    pub fn precision_average(&self, average: Average) -> f64 {
        self.averaged(average, |tp, fp, _| ratio(tp, tp + fp))
    }

    /// Recall averaged over the classes by `average` (see [`Average`]); 0 with no rows.
    ///
    /// ```
    /// use cavitation_nn::metrics::{Average, ConfusionMatrix};
    /// let m = ConfusionMatrix::from_labels(2, &[0, 1], &[0, 0]).unwrap();
    /// assert_eq!(m.recall_average(Average::Macro), 0.5);
    /// ```
    pub fn recall_average(&self, average: Average) -> f64 {
        self.averaged(average, |tp, _, fne| ratio(tp, tp + fne))
    }

    /// F-beta averaged over the classes by `average` (see [`Average`]); 0 with no rows.
    /// The micro average is the F-beta of the pooled counts, as scikit-learn.
    ///
    /// # Panics
    ///
    /// Panics if `beta` is negative or not finite.
    ///
    /// ```
    /// use cavitation_nn::metrics::{Average, ConfusionMatrix};
    /// let m = ConfusionMatrix::from_labels(2, &[0, 0, 1, 1], &[0, 0, 0, 1]).unwrap();
    /// // Class 0: F1 = 4/5; class 1: F1 = 2/3.
    /// assert!((m.f_beta_average(1.0, Average::Macro) - (0.8 + 2.0 / 3.0) / 2.0).abs() < 1e-15);
    /// ```
    pub fn f_beta_average(&self, beta: f64, average: Average) -> f64 {
        let beta = check_beta(beta);
        self.averaged(average, |tp, fp, fne| f_beta_of(tp, fp, fne, beta))
    }

    /// Class `class`'s true positives, false positives and false negatives.
    fn outcomes(&self, class: usize) -> (u64, u64, u64) {
        let tp = self.count(class, class);
        (tp, self.predicted(class) - tp, self.support(class) - tp)
    }

    /// The classes scikit-learn's default `labels` would hold: those with rows or
    /// predictions.
    fn present(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.classes).filter(|&c| self.support(c) > 0 || self.predicted(c) > 0)
    }

    /// `rate` (of true positives, false positives and false negatives) averaged by
    /// `average` over the present classes.
    fn averaged(&self, average: Average, rate: impl Fn(u64, u64, u64) -> f64) -> f64 {
        match average {
            Average::Micro => {
                let (mut tp, mut fp, mut fne) = (0, 0, 0);
                for c in 0..self.classes {
                    let (t, p, n) = self.outcomes(c);
                    (tp, fp, fne) = (tp + t, fp + p, fne + n);
                }
                rate(tp, fp, fne)
            }
            Average::Macro => {
                let (mut sum, mut n) = (0.0, 0usize);
                for c in self.present() {
                    let (tp, fp, fne) = self.outcomes(c);
                    sum += rate(tp, fp, fne);
                    n += 1;
                }
                if n == 0 { 0.0 } else { sum / n as f64 }
            }
            Average::Weighted => {
                let (mut sum, mut weight) = (0.0, 0u64);
                for c in self.present() {
                    let (tp, fp, fne) = self.outcomes(c);
                    let s = self.support(c);
                    sum += s as f64 * rate(tp, fp, fne);
                    weight += s;
                }
                if weight == 0 {
                    0.0
                } else {
                    sum / weight as f64
                }
            }
        }
    }
}

/// A table of the counts (true class down, predicted class across, a last column of rows
/// without a prediction when there are any), then one line per present class with its
/// precision, recall, F1 and support, then accuracy and the macro and weighted averages, as
/// scikit-learn's `classification_report`.
///
/// ```
/// use cavitation_nn::metrics::ConfusionMatrix;
/// let m = ConfusionMatrix::from_labels(2, &[0, 1, 1], &[0, 1, 0]).unwrap();
/// let text = m.to_string();
/// assert!(text.contains("precision") && text.contains("macro avg"));
/// ```
impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let missing = self.unpredicted.iter().any(|&n| n > 0);
        write!(f, "{:>12}", "true \\ pred")?;
        for p in 0..self.classes {
            write!(f, " {p:>8}")?;
        }
        if missing {
            write!(f, " {:>8}", "none")?;
        }
        writeln!(f)?;
        for t in 0..self.classes {
            write!(f, "{t:>12}")?;
            for p in 0..self.classes {
                write!(f, " {:>8}", self.count(t, p))?;
            }
            if missing {
                write!(f, " {:>8}", self.unpredicted(t))?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "{:>12} {:>9} {:>9} {:>9} {:>9}",
            "class", "precision", "recall", "f1", "support"
        )?;
        for c in self.present() {
            writeln!(
                f,
                "{c:>12} {:>9.4} {:>9.4} {:>9.4} {:>9}",
                self.precision(c),
                self.recall(c),
                self.f1(c),
                self.support(c)
            )?;
        }
        writeln!(
            f,
            "{:>12} {:>29.4} {:>9}",
            "accuracy",
            self.accuracy(),
            self.total()
        )?;
        for (name, avg) in [
            ("macro avg", Average::Macro),
            ("weighted avg", Average::Weighted),
        ] {
            writeln!(
                f,
                "{name:>12} {:>9.4} {:>9.4} {:>9.4} {:>9}",
                self.precision_average(avg),
                self.recall_average(avg),
                self.f_beta_average(1.0, avg),
                self.total()
            )?;
        }
        Ok(())
    }
}

/// `num / den` as f64, 0 when `den` is 0 (scikit-learn's `zero_division=0`).
fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// F-beta from the counts, 0 when its denominator is 0.
fn f_beta_of(tp: u64, fp: u64, fne: u64, beta: f64) -> f64 {
    let b2 = beta * beta;
    let num = (1.0 + b2) * tp as f64;
    let den = num + b2 * fne as f64 + fp as f64;
    if den == 0.0 { 0.0 } else { num / den }
}

/// `beta`, checked to be finite and not negative.
fn check_beta(beta: f64) -> f64 {
    assert!(
        beta.is_finite() && beta >= 0.0,
        "f_beta: beta must be finite and not negative, got {beta}"
    );
    beta
}

/// An error unless the targets and the predictions have the same length.
fn check_lengths(targets: usize, predicted: usize) -> Result<(), Error> {
    check_pair(targets, "targets", predicted, "predictions")
}

/// An error unless `a` and `b`, named `a_name` and `b_name`, are equal.
fn check_pair(a: usize, a_name: &str, b: usize, b_name: &str) -> Result<(), Error> {
    if a == b {
        Ok(())
    } else {
        Err(Error::InvalidData(format!(
            "metrics: {a} {a_name} but {b} {b_name}"
        )))
    }
}

/// The points of a ROC curve, `(false positives, true positives)` counted so far at each
/// distinct score from the highest down, with the totals of negatives and positives.
type Curve = (Vec<(u64, u64)>, u64, u64);

/// The ROC curve's points from scores and labels, one per distinct score from the highest
/// down, as `(false positives, true positives)` counted so far; with the totals of
/// negatives and positives.
///
/// # Errors
///
/// As [`roc_auc`].
fn curve(scores: &[f32], positive: &[bool]) -> Result<Curve, Error> {
    check_pair(positive.len(), "labels", scores.len(), "scores")?;
    if let Some(i) = scores.iter().position(|s| s.is_nan()) {
        return Err(Error::InvalidData(format!("metrics: score {i} is NaN")));
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut points = Vec::new();
    let (mut fp, mut tp) = (0u64, 0u64);
    for (k, &i) in order.iter().enumerate() {
        if positive[i] {
            tp += 1;
        } else {
            fp += 1;
        }
        // A point after the last row of each group of equal scores: tied rows move the
        // curve together, as scikit-learn's thresholds do. -0.0 and 0.0 are one score.
        let last = order.get(k + 1).is_none_or(|&j| scores[j] != scores[i]);
        if last {
            points.push((fp, tp));
        }
    }
    Ok((points, fp, tp))
}

/// The area under the ROC curve of a binary classifier's `scores` (higher meaning more
/// likely positive) against which rows are `positive`, as scikit-learn's `roc_auc_score`:
/// the trapezoid area under true positive rate against false positive rate, tied scores
/// moving together. 0.5 is no better than chance, 1 a perfect ranking. An infinite score
/// ranks above (or below) every finite one; scikit-learn refuses infinite scores.
///
/// # Errors
///
/// [`Error::InvalidData`] if the lengths differ, a score is NaN, or the rows are all one
/// class (the curve is then undefined, where scikit-learn raises too).
///
/// ```
/// use cavitation_nn::metrics::roc_auc;
/// let scores = [0.1, 0.4, 0.35, 0.8];
/// let positive = [false, false, true, true];
/// assert_eq!(roc_auc(&scores, &positive).unwrap(), 0.75);
/// assert!(roc_auc(&[0.5, 0.6], &[true, true]).is_err());
/// ```
pub fn roc_auc(scores: &[f32], positive: &[bool]) -> Result<f64, Error> {
    let (points, negatives, positives) = curve(scores, positive)?;
    if negatives == 0 || positives == 0 {
        return Err(Error::InvalidData(format!(
            "roc_auc: needs rows of both classes, got {positives} positive and {negatives} \
             negative"
        )));
    }
    let (mut area, mut prev) = (0.0, (0u64, 0u64));
    for &(fp, tp) in &points {
        area += (fp - prev.0) as f64 * (tp + prev.1) as f64 / 2.0;
        prev = (fp, tp);
    }
    Ok(area / (negatives as f64 * positives as f64))
}

/// The average precision of a binary classifier's `scores` against which rows are
/// `positive`, as scikit-learn's `average_precision_score`: the precision at each distinct
/// score threshold, from the highest down, weighted by how much recall rose there,
/// `Σ (Rₙ − Rₙ₋₁) · Pₙ`. It summarises the precision-recall curve without interpolating.
/// Infinite scores rank as in [`roc_auc`].
///
/// # Errors
///
/// [`Error::InvalidData`] if the lengths differ, a score is NaN, or no row is positive.
/// Recall is then undefined; scikit-learn warns and returns 0 there, which a caller could
/// mistake for a real score, so this refuses instead.
///
/// ```
/// use cavitation_nn::metrics::average_precision;
/// let scores = [0.1, 0.4, 0.35, 0.8];
/// let positive = [false, false, true, true];
/// // Thresholds 0.8 (P 1, R 1/2), 0.4 (P 1/2, R 1/2), 0.35 (P 2/3, R 1).
/// let ap = average_precision(&scores, &positive).unwrap();
/// assert!((ap - (0.5 * 1.0 + 0.5 * 2.0 / 3.0)).abs() < 1e-15);
/// ```
pub fn average_precision(scores: &[f32], positive: &[bool]) -> Result<f64, Error> {
    let (points, _, positives) = curve(scores, positive)?;
    if positives == 0 {
        return Err(Error::InvalidData(
            "average_precision: no row is positive, so recall is undefined".into(),
        ));
    }
    let (mut ap, mut prev_tp) = (0.0, 0u64);
    for &(fp, tp) in &points {
        if tp > prev_tp {
            let precision = tp as f64 / (tp + fp) as f64;
            ap += (tp - prev_tp) as f64 / positives as f64 * precision;
        }
        prev_tp = tp;
    }
    Ok(ap)
}

/// Regression errors per output of `[rows, outputs]` targets and predictions, as
/// scikit-learn's `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`
/// and `r2_score` with `multioutput='raw_values'`; the `*_mean` methods are their
/// `uniform_average`, the mean over the outputs. Sums run in f64.
///
/// R² of an output whose targets are all equal is 1 if every prediction is exact and 0
/// otherwise, as scikit-learn's `force_finite=True`. One row is such a case here too, where
/// scikit-learn warns that R² is not defined and returns NaN.
///
/// ```
/// use cavitation_nn::metrics::RegressionMetrics;
/// // Two rows, two outputs, row-major.
/// let targets = [1.0, 10.0, 3.0, 20.0];
/// let predicted = [2.0, 10.0, 3.0, 16.0];
/// let m = RegressionMetrics::new(&targets, &predicted, 2).unwrap();
/// assert_eq!((m.mae(0), m.mae(1)), (0.5, 2.0));
/// assert_eq!((m.mse(0), m.mse(1)), (0.5, 8.0));
/// assert_eq!(m.rmse(1), 8f64.sqrt());
/// assert_eq!(m.r2(0), 0.5); // 1 − 1 / 2
/// assert_eq!(m.mae_mean(), 1.25);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionMetrics {
    /// The rows.
    rows: usize,
    /// Mean absolute error per output.
    mae: Vec<f64>,
    /// Mean squared error per output.
    mse: Vec<f64>,
    /// R² per output.
    r2: Vec<f64>,
}

impl RegressionMetrics {
    /// The errors of `predicted` against `targets`, both `[rows, outputs]` row-major.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `outputs` is 0, the lengths differ or are not a whole
    /// number of rows, there are no rows, or a value is not finite.
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert!(RegressionMetrics::new(&[1.0, 2.0], &[1.0], 1).is_err());
    /// assert!(RegressionMetrics::new(&[f32::NAN], &[1.0], 1).is_err());
    /// ```
    pub fn new(
        targets: &[f32],
        predicted: &[f32],
        outputs: usize,
    ) -> Result<RegressionMetrics, Error> {
        check_pair(
            targets.len(),
            "target values",
            predicted.len(),
            "predicted values",
        )?;
        if outputs == 0 || targets.is_empty() || !targets.len().is_multiple_of(outputs) {
            return Err(Error::InvalidData(format!(
                "regression metrics: {} values are not a whole, non-empty number of rows of {outputs} outputs",
                targets.len()
            )));
        }
        if let Some(i) = targets.iter().chain(predicted).position(|v| !v.is_finite()) {
            let (what, at) = if i < targets.len() {
                ("target", i)
            } else {
                ("prediction", i - targets.len())
            };
            return Err(Error::InvalidData(format!(
                "regression metrics: the {what} of row {}, output {}, is not finite",
                at / outputs,
                at % outputs
            )));
        }
        let rows = targets.len() / outputs;
        let (mut abs, mut sq, mut mean) =
            (vec![0.0; outputs], vec![0.0; outputs], vec![0.0; outputs]);
        for (t, p) in targets
            .chunks_exact(outputs)
            .zip(predicted.chunks_exact(outputs))
        {
            for o in 0..outputs {
                let e = f64::from(t[o]) - f64::from(p[o]);
                abs[o] += e.abs();
                sq[o] += e * e;
                mean[o] += f64::from(t[o]);
            }
        }
        let n = rows as f64;
        for m in &mut mean {
            *m /= n;
        }
        let mut tot = vec![0.0; outputs];
        for t in targets.chunks_exact(outputs) {
            for o in 0..outputs {
                let d = f64::from(t[o]) - mean[o];
                tot[o] += d * d;
            }
        }
        let r2 = (0..outputs)
            .map(|o| match (tot[o] == 0.0, sq[o] == 0.0) {
                (true, true) => 1.0,
                (true, false) => 0.0,
                _ => 1.0 - sq[o] / tot[o],
            })
            .collect();
        Ok(RegressionMetrics {
            rows,
            mae: abs.into_iter().map(|a| a / n).collect(),
            mse: sq.into_iter().map(|q| q / n).collect(),
            r2,
        })
    }

    /// The number of outputs.
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[1.0, 2.0], &[1.0, 2.0], 2).unwrap().outputs(), 2);
    /// ```
    pub fn outputs(&self) -> usize {
        self.mae.len()
    }

    /// The number of rows.
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[1.0, 2.0], &[1.0, 2.0], 1).unwrap().rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The mean absolute error of output `o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` is not below [`outputs`](Self::outputs).
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[1.0, 4.0], &[2.0, 2.0], 1).unwrap().mae(0), 1.5);
    /// ```
    pub fn mae(&self, o: usize) -> f64 {
        self.mae[o]
    }

    /// The mean squared error of output `o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` is not below [`outputs`](Self::outputs).
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[1.0, 4.0], &[2.0, 2.0], 1).unwrap().mse(0), 2.5);
    /// ```
    pub fn mse(&self, o: usize) -> f64 {
        self.mse[o]
    }

    /// The root mean squared error of output `o`, in the targets' units.
    ///
    /// # Panics
    ///
    /// Panics if `o` is not below [`outputs`](Self::outputs).
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[0.0, 0.0], &[3.0, 4.0], 2).unwrap().rmse(1), 4.0);
    /// ```
    pub fn rmse(&self, o: usize) -> f64 {
        self.mse[o].sqrt()
    }

    /// The coefficient of determination R² of output `o`: 1 minus the squared error over
    /// the targets' spread about their mean.
    ///
    /// # Panics
    ///
    /// Panics if `o` is not below [`outputs`](Self::outputs).
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// // Predicting the targets' mean scores 0.
    /// assert_eq!(RegressionMetrics::new(&[1.0, 3.0], &[2.0, 2.0], 1).unwrap().r2(0), 0.0);
    /// ```
    pub fn r2(&self, o: usize) -> f64 {
        self.r2[o]
    }

    /// The mean absolute error averaged over the outputs.
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[0.0, 0.0], &[1.0, 3.0], 2).unwrap().mae_mean(), 2.0);
    /// ```
    pub fn mae_mean(&self) -> f64 {
        mean_of(&self.mae)
    }

    /// The mean squared error averaged over the outputs.
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[0.0, 0.0], &[1.0, 3.0], 2).unwrap().mse_mean(), 5.0);
    /// ```
    pub fn mse_mean(&self) -> f64 {
        mean_of(&self.mse)
    }

    /// The root mean squared error averaged over the outputs: each output's RMSE, then
    /// their mean, as scikit-learn's `root_mean_squared_error` (not the root of
    /// [`mse_mean`](Self::mse_mean)).
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// assert_eq!(RegressionMetrics::new(&[0.0, 0.0], &[1.0, 3.0], 2).unwrap().rmse_mean(), 2.0);
    /// ```
    pub fn rmse_mean(&self) -> f64 {
        self.mse.iter().map(|m| m.sqrt()).sum::<f64>() / self.mse.len() as f64
    }

    /// R² averaged over the outputs.
    ///
    /// ```
    /// use cavitation_nn::metrics::RegressionMetrics;
    /// let m = RegressionMetrics::new(&[1.0, 1.0, 3.0, 3.0], &[1.0, 2.0, 3.0, 2.0], 2).unwrap();
    /// assert_eq!((m.r2(0), m.r2(1), m.r2_mean()), (1.0, 0.0, 0.5));
    /// ```
    pub fn r2_mean(&self) -> f64 {
        mean_of(&self.r2)
    }
}

/// The mean of `values`.
fn mean_of(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The mean cross-entropy of predicted class probabilities against labels, as
/// scikit-learn's `log_loss` with `labels = 0..classes`: `−mean ln p[label]` over the rows,
/// each probability first clipped to `[ε, 1 − ε]` with ε = `f32::EPSILON` (scikit-learn
/// clips at the epsilon of the input's type). `probabilities` is `[rows, classes]`
/// row-major.
///
/// # Errors
///
/// [`Error::InvalidData`] if the lengths do not fit, a label is not below `classes`, a
/// probability is outside `[0, 1]` or NaN, or a row does not sum to 1 within
/// `√f32::EPSILON` relative. scikit-learn only warns at that last check; a row that does
/// not sum to 1 is not a probability, so this refuses it.
///
/// ```
/// use cavitation_nn::metrics::log_loss;
/// let p = [0.5, 0.5, 0.25, 0.75];
/// let loss = log_loss(&p, 2, &[0, 1]).unwrap();
/// assert!((loss - (2f64.ln() + (4.0f64 / 3.0).ln()) / 2.0).abs() < 1e-12);
/// assert!(log_loss(&[0.5, 0.6], 2, &[0]).is_err());
/// ```
pub fn log_loss(probabilities: &[f32], classes: usize, labels: &[usize]) -> Result<f64, Error> {
    let rows = check_rows("log_loss", probabilities.len(), classes, labels)?;
    let eps = f64::from(f32::EPSILON);
    let tol = f64::from(f32::EPSILON).sqrt();
    let mut total = 0.0;
    for (r, (row, &label)) in probabilities.chunks_exact(classes).zip(labels).enumerate() {
        if let Some(v) = row.iter().find(|v| !(0.0..=1.0).contains(*v)) {
            return Err(Error::InvalidData(format!(
                "log_loss: row {r} has {v}, which is not a probability"
            )));
        }
        let sum: f64 = row.iter().map(|&v| f64::from(v)).sum();
        if (sum - 1.0).abs() > tol {
            return Err(Error::InvalidData(format!(
                "log_loss: row {r} sums to {sum}, not 1"
            )));
        }
        total -= f64::from(row[label]).clamp(eps, 1.0 - eps).ln();
    }
    Ok(total / rows as f64)
}

/// The share of rows whose label is among the `k` highest of their `scores` (`[rows,
/// classes]` row-major), as scikit-learn's `top_k_accuracy_score` with `labels =
/// 0..classes`: among equal scores, the higher class index ranks first. `k = 1` is plain
/// accuracy of the largest score, with that tie rule, and a `k` of `classes` or more scores
/// 1 (scikit-learn warns there). With 2 classes this ranks the two columns; scikit-learn's
/// two-class path instead thresholds one score column, so the two agree from 3 classes on.
///
/// # Errors
///
/// [`Error::InvalidData`] if `k` is 0, the lengths do not fit, a label is not below
/// `classes`, or a score is NaN.
///
/// ```
/// use cavitation_nn::metrics::top_k_accuracy;
/// let scores = [0.1, 0.6, 0.3, 0.5, 0.2, 0.3];
/// // Row 0 ranks 1, 2, 0; row 1 ranks 0, 2, 1.
/// assert_eq!(top_k_accuracy(&scores, 3, &[2, 2], 1).unwrap(), 0.0);
/// assert_eq!(top_k_accuracy(&scores, 3, &[2, 2], 2).unwrap(), 1.0);
/// ```
pub fn top_k_accuracy(
    scores: &[f32],
    classes: usize,
    labels: &[usize],
    k: usize,
) -> Result<f64, Error> {
    if k == 0 {
        return Err(Error::InvalidData(
            "top_k_accuracy: k must be at least 1".into(),
        ));
    }
    let rows = check_rows("top_k_accuracy", scores.len(), classes, labels)?;
    let mut hits = 0usize;
    for (r, (row, &label)) in scores.chunks_exact(classes).zip(labels).enumerate() {
        if row.iter().any(|v| v.is_nan()) {
            return Err(Error::InvalidData(format!(
                "top_k_accuracy: row {r} has a NaN score"
            )));
        }
        let s = row[label];
        // The classes ranked above the label: higher scores, and equal scores at a higher
        // index.
        let above = row
            .iter()
            .enumerate()
            .filter(|&(c, &v)| v > s || (v == s && c > label))
            .count();
        if above < k {
            hits += 1;
        }
    }
    Ok(hits as f64 / rows as f64)
}

/// The number of rows of a `[rows, classes]` buffer of `len` values with one label each,
/// checked: `what` names the caller in the error.
fn check_rows(what: &str, len: usize, classes: usize, labels: &[usize]) -> Result<usize, Error> {
    if classes == 0 || labels.is_empty() || len != labels.len() * classes {
        return Err(Error::InvalidData(format!(
            "{what}: {len} values do not make {} rows of {classes} classes",
            labels.len()
        )));
    }
    if let Some((r, &l)) = labels.iter().enumerate().find(|&(_, &l)| l >= classes) {
        return Err(Error::InvalidData(format!(
            "{what}: row {r}'s label is {l}, but there are {classes} classes"
        )));
    }
    Ok(labels.len())
}

/// Tests for the metrics by hand; the scikit-learn fixtures are in
/// `tests/metrics.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every count, rate and average of a small three-class case, by hand.
    #[test]
    fn three_classes_by_hand() {
        // Class 2 has rows but is never predicted; class 1 is over-predicted.
        let m = ConfusionMatrix::from_labels(3, &[0, 0, 1, 1, 1, 2], &[0, 1, 1, 1, 0, 1]).unwrap();
        assert_eq!((m.total(), m.support(1), m.predicted(1)), (6, 3, 4));
        assert_eq!(m.accuracy(), 0.5);
        assert_eq!((m.precision(0), m.recall(0)), (0.5, 0.5));
        assert_eq!((m.precision(1), m.recall(1)), (0.5, 2.0 / 3.0));
        assert_eq!((m.precision(2), m.recall(2), m.f1(2)), (0.0, 0.0, 0.0));
        assert_eq!(m.precision_average(Average::Micro), 0.5);
        assert_eq!(m.recall_average(Average::Micro), 0.5);
        let macro_r = (0.5 + 2.0 / 3.0 + 0.0) / 3.0;
        assert!((m.recall_average(Average::Macro) - macro_r).abs() < 1e-15);
        let weighted_r = (2.0 * 0.5 + 3.0 * (2.0 / 3.0) + 0.0) / 6.0;
        assert!((m.recall_average(Average::Weighted) - weighted_r).abs() < 1e-15);
    }

    /// A class with neither rows nor predictions is left out of the macro average, as
    /// scikit-learn's default labels leave it out.
    #[test]
    fn absent_classes_are_left_out_of_averages() {
        let m = ConfusionMatrix::from_labels(4, &[0, 1], &[0, 1]).unwrap();
        assert_eq!(m.precision_average(Average::Macro), 1.0);
        assert_eq!(m.f_beta_average(2.0, Average::Weighted), 1.0);
        let empty = ConfusionMatrix::new(3);
        assert_eq!(
            (empty.accuracy(), empty.recall_average(Average::Macro)),
            (0.0, 0.0)
        );
    }

    /// Rows without a prediction lower recall and accuracy but add no false positive.
    #[test]
    fn unpredicted_rows() {
        let m =
            ConfusionMatrix::from_predictions(2, &[0, 0, 1], &[Some(0), None, Some(1)]).unwrap();
        assert_eq!((m.recall(0), m.precision(0)), (0.5, 1.0));
        assert_eq!(m.accuracy(), 2.0 / 3.0);
        assert_eq!(m.precision_average(Average::Micro), 1.0);
        assert_eq!(m.recall_average(Average::Micro), 2.0 / 3.0);
        assert!(m.to_string().contains("none"));
    }

    /// Bad lengths and labels are refused, naming the row.
    #[test]
    fn bad_input_is_refused() {
        let e = ConfusionMatrix::from_labels(2, &[0], &[0, 1]).unwrap_err();
        assert!(e.to_string().contains("1 targets but 2 predictions"), "{e}");
        let e = ConfusionMatrix::from_labels(2, &[0, 1], &[0, 5]).unwrap_err();
        assert!(e.to_string().contains("row 1's prediction is 5"), "{e}");
        assert!(roc_auc(&[f32::NAN, 0.0], &[true, false]).is_err());
        assert!(average_precision(&[0.1, 0.2], &[false, false]).is_err());
    }

    /// Tied scores move the curves together: all rows tied give an AUC of one half, and a
    /// perfect ranking gives 1.
    #[test]
    fn ties_and_extremes() {
        assert_eq!(
            roc_auc(&[0.5; 4], &[true, false, true, false]).unwrap(),
            0.5
        );
        assert_eq!(
            average_precision(&[0.5; 4], &[true, false, true, false]).unwrap(),
            0.5
        );
        assert_eq!(
            roc_auc(&[0.9, 0.8, 0.2], &[true, true, false]).unwrap(),
            1.0
        );
        assert_eq!(
            roc_auc(&[0.1, 0.8, 0.9], &[true, false, false]).unwrap(),
            0.0
        );
        assert_eq!(roc_auc(&[-0.0, 0.0], &[true, false]).unwrap(), 0.5);
        let inf = [f32::INFINITY, 0.0, f32::NEG_INFINITY];
        assert_eq!(roc_auc(&inf, &[true, false, false]).unwrap(), 1.0);
        assert_eq!(average_precision(&inf, &[false, true, false]).unwrap(), 0.5);
    }

    /// Log loss clips a probability of 0 to ε, refuses rows that are not probabilities, and
    /// top-k ranks a tie by the higher index first.
    #[test]
    fn log_loss_and_top_k_edges() {
        let eps = f64::from(f32::EPSILON);
        let loss = log_loss(&[1.0, 0.0], 2, &[1]).unwrap();
        assert!((loss + eps.ln()).abs() < 1e-12, "{loss}");
        assert!(log_loss(&[1.5, -0.5], 2, &[0]).is_err());
        assert!(log_loss(&[0.5, 0.5], 2, &[2]).is_err());
        assert!(log_loss(&[0.5, f32::NAN], 2, &[0]).is_err());
        // Equal scores: class 1 ranks above class 0.
        assert_eq!(top_k_accuracy(&[0.5, 0.5], 2, &[1], 1).unwrap(), 1.0);
        assert_eq!(top_k_accuracy(&[0.5, 0.5], 2, &[0], 1).unwrap(), 0.0);
        assert!(top_k_accuracy(&[0.5, 0.5], 2, &[0], 0).is_err());
    }

    /// R² of constant targets is 1 when predicted exactly and 0 otherwise; bad shapes and
    /// non-finite values are refused.
    #[test]
    fn regression_edges() {
        let exact = RegressionMetrics::new(&[2.0, 2.0], &[2.0, 2.0], 1).unwrap();
        let missed = RegressionMetrics::new(&[2.0, 2.0], &[2.0, 3.0], 1).unwrap();
        assert_eq!((exact.r2(0), missed.r2(0)), (1.0, 0.0));
        assert!(RegressionMetrics::new(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 2).is_err());
        assert!(RegressionMetrics::new(&[], &[], 1).is_err());
        let e = RegressionMetrics::new(&[1.0, 2.0], &[1.0, f32::INFINITY], 1).unwrap_err();
        assert!(
            e.to_string().contains("prediction of row 1, output 0"),
            "{e}"
        );
        // One row: the constant-column rule (scikit-learn returns NaN).
        let one = RegressionMetrics::new(&[2.0], &[3.0], 1).unwrap();
        assert_eq!(one.r2(0), 0.0);
        assert_eq!(top_k_accuracy(&[0.1, 0.9, 0.5], 3, &[0], 3).unwrap(), 1.0);
    }

    /// A negative or non-finite beta is refused.
    #[test]
    #[should_panic(expected = "beta must be finite")]
    fn bad_beta_panics() {
        ConfusionMatrix::new(2).f_beta(0, -1.0);
    }
}
