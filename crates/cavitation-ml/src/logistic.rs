//! Logistic regression: the exact path ([`LogisticRegression::fit`], Newton's method as
//! iteratively reweighted least squares solved by QR in `f64`), the gradient path
//! ([`LogisticRegression::fit_gradient`], one `Linear` layer trained on cross-entropy), and
//! the fitted [`LogisticModel`]: probabilities, predictions, accuracy, the objective, and the
//! conversion to a one-layer [`Sequential`].

use cavitation_core::{Error, ParamStore, Real, Tensor, TensorView};
use cavitation_nn::{Dataset, History, Sequential, Targets};

use crate::gradient::{GradientConfig, Setup, StepLoss, train};
use crate::linalg::Qr;
use crate::linear::LinearModel;

/// The smallest value `p` and `1 − p` each take in a Newton step's weight
/// `w = p · (1 − p)`. A row whose probability is closer to 0 or 1 than this gets the weight
/// of one this close, so the weight never reaches 0 and the working response
/// `z = x·β + (y − p)/w` stays finite. The residual `y − p` itself is not clamped, so the
/// gradient, and with it the solution Newton's method converges to, is unchanged; only the
/// curvature of those rows is overstated, which shortens their steps. It sits below the
/// `1e-8` per row at which separation is detected, so a separable problem is reported
/// before the clamp slows it.
const MIN_PROBABILITY: f64 = 1e-10;

/// The most times a Newton step is halved looking for a lower objective; past it the fit
/// reports that it stalled.
const MAX_HALVINGS: usize = 64;

/// How far, in multiples of `f64::EPSILON · (terms + 2) · |objective|`, an objective may
/// rise and still count as unchanged. The objective is a sum of non-negative terms (one
/// log-loss per row and one square per weight), each computed to within a few ulp, and a
/// running sum of `terms` non-negative values carries an error of at most
/// `(terms − 1) · ε` times the sum; two evaluations of it can therefore differ by about
/// twice that with no real change. A rise inside this band is rounding, not a worse point.
const ROUNDING: f64 = 2.0;

/// Logistic regression: the coefficients `w` and intercept `b` that minimise
/// `c · Σ log(1 + exp(−s·(x·w + b))) + ½‖w‖²` over every row, with `s = 1` for class 1 and
/// `s = −1` for class 0, scikit-learn's `LogisticRegression` with an L2 penalty. The
/// intercept is an unpenalised column of ones (the inputs are not centred); `c: None` drops
/// the penalty and the factor `c`, leaving the log-loss alone.
///
/// [`fit`](Self::fit) solves the two-class problem exactly by Newton's method;
/// [`fit_gradient`](Self::fit_gradient) trains a [`Linear`](cavitation_nn::Linear) layer
/// towards the same answer by gradient descent, and also fits more than two classes
/// (multinomial: softmax cross-entropy, with `½‖W‖²` over every class's weights).
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::LogisticRegression;
/// use cavitation_nn::{Dataset, Targets};
/// let data = Dataset {
///     x: Tensor::from_vec(vec![-2.0f32, -1.0, -0.5, 0.5, 1.0, 2.0], &[6, 1]).unwrap(),
///     targets: Targets::Classes(vec![0, 0, 1, 0, 1, 1]),
/// };
/// let model = LogisticRegression::default().fit(&data).unwrap();
/// assert!(model.coef.data()[0] > 0.0);
/// assert_eq!(model.predict(data.x.view()), [0, 0, 0, 1, 1, 1]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogisticRegression {
    /// Whether to fit an intercept. Without one the decision boundary passes through the
    /// origin.
    pub fit_intercept: bool,
    /// The inverse strength of the L2 penalty, as scikit-learn's `C`: smaller is stronger.
    /// `None` is no penalty, which has no finite answer when the classes are separable.
    pub c: Option<f64>,
    /// The most Newton steps [`fit`](Self::fit) takes.
    pub max_iter: usize,
    /// [`fit`](Self::fit) stops once a step changes every coefficient by at most
    /// `tol · (1 + |coefficient|)`. Values below about 1e-14 can be finer than `f64`
    /// resolves the answer, and then end in an error saying Newton's method stalled.
    pub tol: f64,
}

impl Default for LogisticRegression {
    /// scikit-learn's default problem: an intercept and `c = Some(1.0)`; at most 100 Newton
    /// steps, stopping when a step changes no coefficient by more than `1e-10` relative.
    fn default() -> LogisticRegression {
        LogisticRegression {
            fit_intercept: true,
            c: Some(1.0),
            max_iter: 100,
            tol: 1e-10,
        }
    }
}

/// The rows of a classification problem: the inputs widened to `f64`, row-major, and one
/// class label per row.
struct Classes {
    /// The number of rows.
    rows: usize,
    /// The number of input columns.
    features: usize,
    /// The inputs, `rows · features` values, row-major.
    x: Vec<f64>,
    /// The class label of each row.
    labels: Vec<usize>,
}

impl LogisticRegression {
    /// Fits two classes exactly: Newton's method, each step solved as a weighted
    /// least-squares problem by Householder QR in `f64`, from the data's values (an `f32`
    /// data set is widened exactly, never rounded).
    ///
    /// Starting from all coefficients 0, each step computes every row's probability
    /// `p = σ(x·β)`, its weight `w = p(1 − p)` and working response `z = x·β + (y − p)/w`
    /// (with `x` led by a 1 for the intercept), and solves the least-squares problem with
    /// rows `√(c·w)·x` and right-hand side `√(c·w)·z`, plus, with a penalty, one row per
    /// weight (not the intercept) that is 1 in that weight's column with right-hand side 0;
    /// without a penalty `c` is 1 in these rows and there are no extra rows. Its normal
    /// equations, `(c·XᵀWX + I)·β' = c·XᵀWz`, are the Newton step for the objective, and its
    /// solution is the next `β`, halved towards the current one until the objective
    /// decreases. A step along which the objective falls at first (its directional
    /// derivative is negative) and whose new objective is above the current one by no more
    /// than the rounding of the sum, `2 · ε · (terms + 2) · |objective|`, is taken as it is:
    /// near the optimum, and with rows whose probabilities are far into the tails, the
    /// decrease is smaller than `f64` resolves, and requiring it would stop Newton short.
    /// The fit stops when a step changes every coefficient by at most
    /// `tol · (1 + |coefficient|)`. In the weight, `p` and `1 − p` are each taken as at
    /// least 1e-10, so that `w` never reaches 0; the residual `y − p` is exact, so this does
    /// not move the answer.
    ///
    /// The targets are class labels 0 and 1, or `[rows, 1]` value targets holding 0 and 1.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`]:
    ///
    /// - if `c` is not a finite number above 0, `tol` is negative or not finite, or
    ///   `max_iter` is 0;
    /// - if `data.x` is not a matrix with at least one column, or a value is NaN or
    ///   infinite (naming the row and column), or the targets do not give one class label
    ///   per row;
    /// - if the labels are not 0 and 1: naming them, and suggesting
    ///   [`fit_gradient`](Self::fit_gradient) for more than two classes; and if only one
    ///   class appears, naming it;
    /// - without a penalty, if there are more coefficients than rows, and if a column is a
    ///   linear combination of the columns before it (and the intercept's column of ones),
    ///   naming the column;
    /// - without a penalty, if the classes are separable, so that the log-loss falls
    ///   towards its infimum while the coefficients grow without bound. This is detected
    ///   when the objective falls below `1e-8 · rows` while the coefficients' norm still
    ///   grows, when `max_iter` is reached while it grows and the objective has not risen,
    ///   or when the weighted problem becomes rank deficient after the first step while the
    ///   objective is still falling. The message says the classes are separable and
    ///   suggests setting `c`;
    /// - if `max_iter` steps do not converge otherwise, or if no point along a Newton step
    ///   lowers the objective (beyond rounding) while the step is still larger than `tol`
    ///   allows: the message suggests raising `tol`, and, when the fit has no penalty and
    ///   the coefficients are moving outward or some rows' probabilities are within 1e-10
    ///   of 0 or 1, says the classes may be separable.
    ///
    /// A model is returned only when a step passed the `tol` test.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LogisticRegression;
    /// use cavitation_nn::{Dataset, Targets};
    /// // Separable classes: no finite answer without a penalty.
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![-2.0f32, -1.0, 1.0, 2.0], &[4, 1]).unwrap(),
    ///     targets: Targets::Classes(vec![0, 0, 1, 1]),
    /// };
    /// let none = LogisticRegression { c: None, ..LogisticRegression::default() };
    /// let err = none.fit(&data).unwrap_err();
    /// assert!(err.to_string().contains("separable"));
    /// // With the default penalty, c = 1, Newton converges in a few steps.
    /// let model = LogisticRegression::default().fit(&data).unwrap();
    /// assert!(model.iterations < 10);
    /// ```
    pub fn fit<T: Real>(&self, data: &Dataset<T>) -> Result<LogisticModel, Error> {
        self.check_settings()?;
        let problem = check_classes(data)?;
        let mut seen = problem.labels.clone();
        seen.sort_unstable();
        seen.dedup();
        match *seen.as_slice() {
            [0, 1] => {}
            [only] => return Err(one_class(only)),
            [a, b] => {
                return Err(Error::InvalidData(format!(
                    "logistic regression: the class labels are {a} and {b}, but fit expects \
                     0 and 1; relabel the classes"
                )));
            }
            _ => {
                return Err(Error::InvalidData(format!(
                    "logistic regression: the targets hold {} classes (labels {}), but fit \
                     handles two, labelled 0 and 1; use fit_gradient for more classes",
                    seen.len(),
                    list(&seen)
                )));
            }
        }
        let (beta, iterations) = self.newton(&problem)?;
        let offset = usize::from(self.fit_intercept);
        let intercept = if self.fit_intercept { beta[0] } else { 0.0 };
        Ok(LogisticModel {
            coef: Tensor::from_vec(beta[offset..].to_vec(), &[1, problem.features])?,
            intercept: vec![intercept],
            classes: 2,
            iterations,
            c: self.c,
        })
    }

    /// Returns [`Error::InvalidData`] unless `c` is `None` or finite and above 0, `tol` is
    /// finite and not negative, and `max_iter` is at least 1.
    fn check_settings(&self) -> Result<(), Error> {
        self.check_c()?;
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(Error::InvalidData(format!(
                "logistic regression: tol is {}, but it must be a finite number at least 0",
                self.tol
            )));
        }
        if self.max_iter == 0 {
            return Err(Error::InvalidData(
                "logistic regression: max_iter is 0, but Newton's method needs at least one \
                 step"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Returns [`Error::InvalidData`] unless `c` is `None` or finite and above 0.
    fn check_c(&self) -> Result<(), Error> {
        match self.c {
            Some(c) if !(c.is_finite() && c > 0.0) => Err(Error::InvalidData(format!(
                "logistic regression: c is {c}, but it must be a finite number above 0 (None \
                 for no penalty)"
            ))),
            _ => Ok(()),
        }
    }

    /// Newton's method on the checked two-class `problem`: returns the coefficients, the
    /// intercept first when there is one, and the number of steps taken.
    fn newton(&self, problem: &Classes) -> Result<(Vec<f64>, usize), Error> {
        let Classes {
            rows,
            features,
            ref x,
            ref labels,
        } = *problem;
        let offset = usize::from(self.fit_intercept);
        let cols = offset + features;
        let penalised = self.c.is_some();
        if !penalised && cols > rows {
            let what = if self.fit_intercept {
                format!("{features} input columns and an intercept")
            } else {
                format!("{features} input columns")
            };
            return Err(Error::InvalidData(format!(
                "logistic regression without a penalty: {what} need {cols} coefficients, but \
                 {rows} rows cannot determine them; set c to add a penalty, or use fewer \
                 columns"
            )));
        }
        let weight = self.c.unwrap_or(1.0);
        let total = rows + if penalised { features } else { 0 };
        let positive: Vec<bool> = labels.iter().map(|&l| l == 1).collect();
        let objective = |beta: &[f64]| -> f64 {
            let loss: f64 = x
                .chunks_exact(features)
                .zip(&positive)
                .map(|(row, &pos)| log_loss(score(beta, row, offset), pos))
                .sum();
            if penalised {
                let squares: f64 = beta[offset..].iter().map(|w| w * w).sum();
                weight * loss + 0.5 * squares
            } else {
                loss
            }
        };
        let norm = |beta: &[f64]| beta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let terms = rows + if penalised { features } else { 0 };
        let noise = |value: f64| ROUNDING * f64::EPSILON * (terms + 2) as f64 * value.abs();
        let mut beta = vec![0.0; cols];
        let mut current = objective(&beta);
        let mut previous = f64::INFINITY;
        let mut previous_norm = 0.0;
        for step_number in 1..=self.max_iter {
            let mut a = vec![0.0; total * cols];
            let mut b = vec![0.0; total];
            let mut gradient = vec![0.0; cols];
            let mut clamped = false;
            for (i, (row, &pos)) in x.chunks_exact(features).zip(&positive).enumerate() {
                let eta = score(&beta, row, offset);
                let (p, q) = (sigmoid(eta), sigmoid(-eta));
                clamped |= p < MIN_PROBABILITY || q < MIN_PROBABILITY;
                let w = p.max(MIN_PROBABILITY) * q.max(MIN_PROBABILITY);
                let residual = if pos { q } else { -p };
                let s = (weight * w).sqrt();
                if self.fit_intercept {
                    a[i] = s;
                }
                for (j, &v) in row.iter().enumerate() {
                    a[(offset + j) * total + i] = s * v;
                }
                b[i] = s * (eta + residual / w);
                if self.fit_intercept {
                    gradient[0] -= weight * residual;
                }
                for (g, &v) in gradient[offset..].iter_mut().zip(row) {
                    *g -= weight * residual * v;
                }
            }
            if penalised {
                for (g, &w) in gradient[offset..].iter_mut().zip(&beta[offset..]) {
                    *g += w;
                }
            }
            if penalised {
                for j in 0..features {
                    a[(offset + j) * total + rows + j] = 1.0;
                }
            }
            let qr = Qr::from_columns(a, total, cols).map_err(in_context)?;
            if let Some(col) = qr.dependent_column() {
                if step_number == 1 {
                    return Err(self.dependent(col));
                }
                if !penalised && current < previous {
                    return Err(separable(
                        "the weighted least-squares problem lost rank while the log-loss \
                         was still falling",
                    ));
                }
                return Err(Error::InvalidData(format!(
                    "logistic regression: Newton step {step_number} met a weighted \
                     least-squares problem without a unique solution (column {col} of the \
                     coefficients); set c to add a penalty, or a larger c"
                )));
            }
            let b = Tensor::from_vec(b, &[total, 1])?;
            let next = qr.solve(b.view()).map_err(in_context)?;
            let mut step: Vec<f64> = next.data().iter().zip(&beta).map(|(n, o)| n - o).collect();
            let small = |step: &[f64], at: &[f64]| {
                step.iter()
                    .zip(at)
                    .all(|(d, v)| d.abs() <= self.tol * (1.0 + v.abs()))
            };
            // A full step already within `tol` is the converged answer, taken without asking
            // for a decrease. Otherwise the step is halved until the objective decreases, or,
            // along a descent direction, rises by no more than the rounding of the sum (see
            // `ROUNDING`): a change that small is not resolved in `f64`, and refusing it
            // would stall Newton short of the optimum. If no halving is accepted, the fit
            // stops with an error rather than returning a point the `tol` test did not pass.
            let full: Vec<f64> = beta.iter().zip(&step).map(|(v, d)| v + d).collect();
            if small(&step, &full) {
                return Ok((full, step_number));
            }
            let descent = gradient.iter().zip(&step).map(|(g, d)| g * d).sum::<f64>() < 0.0;
            let slack = noise(current);
            let mut halvings = 0;
            let accepted = loop {
                let trial: Vec<f64> = beta.iter().zip(&step).map(|(v, d)| v + d).collect();
                let value = objective(&trial);
                if value < current || (descent && value <= current + slack) {
                    break Some((trial, value));
                }
                if halvings == MAX_HALVINGS || small(&step, &beta) {
                    break None;
                }
                step.iter_mut().for_each(|d| *d *= 0.5);
                halvings += 1;
            };
            let Some(accepted) = accepted else {
                let outward = norm(&full) > norm(&beta) || norm(&beta) > previous_norm;
                return Err(self.stalled(step_number, !penalised && (outward || clamped)));
            };
            let (trial, value) = accepted;
            let converged = small(&step, &trial);
            previous_norm = norm(&beta);
            previous = current;
            beta = trial;
            current = value;
            if converged {
                return Ok((beta, step_number));
            }
            if !penalised && current < 1e-8 * rows as f64 && norm(&beta) > previous_norm {
                return Err(separable(&format!(
                    "the log-loss fell to {current:e} over {rows} rows while the coefficients \
                     kept growing"
                )));
            }
        }
        if !penalised && norm(&beta) > previous_norm && current <= previous + noise(previous) {
            return Err(separable(&format!(
                "after max_iter = {} Newton steps the coefficients were still growing and the \
                 log-loss was not rising; if they are not separable, raise max_iter",
                self.max_iter
            )));
        }
        Err(Error::InvalidData(format!(
            "logistic regression: Newton's method did not converge in max_iter = {} steps; \
             raise max_iter or tol",
            self.max_iter
        )))
    }

    /// The error for a Newton step, numbered `step_number`, that no halving made acceptable
    /// while it was still larger than `tol` allows. When `separating` (no penalty, and the
    /// stalled step or the last accepted one pushes the coefficients outward, or some
    /// row's probability is within 1e-10 of 0 or 1) the classes may be separable, which the
    /// message says first, suggesting `c`; in either case `tol` may be below what `f64`
    /// resolves for these inputs. Setting `c` is suggested only when it is not set. This
    /// choice only words an error that is returned either way.
    fn stalled(&self, step_number: usize, separating: bool) -> Error {
        let start = format!(
            "logistic regression: Newton's method stalled at step {step_number}: no point along \
             the Newton step lowers the objective, but the step is larger than tol = {:e} allows",
            self.tol
        );
        if separating {
            return Error::InvalidData(format!(
                "{start}, with the coefficients moving outward or some rows' probabilities \
                 within 1e-10 of 0 or 1. The classes may be separable, or nearly so, with no \
                 finite answer; set c to add a penalty. If they are not, tol may be below what \
                 f64 resolves for these inputs; raise tol"
            ));
        }
        let penalty = if self.c.is_none() {
            ", or set c to add a penalty"
        } else {
            ""
        };
        Error::InvalidData(format!(
            "{start}, which can mean tol is below what f64 resolves for these inputs; raise tol, \
             or rescale the inputs{penalty}"
        ))
    }

    /// The error for column `col` of the coefficients (the intercept first, when there is
    /// one) being a linear combination of the columns before it at the first Newton step,
    /// naming the input column.
    fn dependent(&self, col: usize) -> Error {
        if self.fit_intercept && col == 0 {
            return Error::InvalidData(
                "logistic regression: the intercept's column has no weight; set c to add a \
                 penalty"
                    .to_string(),
            );
        }
        let j = col - usize::from(self.fit_intercept);
        let before = if self.fit_intercept {
            " and the intercept's column of ones"
        } else {
            ""
        };
        Error::InvalidData(format!(
            "logistic regression: column {j} is a linear combination of the columns before \
             it{before}, so the coefficients are not unique; drop it, or set c to add a \
             penalty"
        ))
    }

    /// Fits by gradient descent: trains one [`Linear`](cavitation_nn::Linear) layer giving
    /// logits on `data` and returns it as a [`LogisticModel`] with each epoch's report.
    /// [`GradientConfig`] says how, as for
    /// [`LinearRegression::fit_gradient`](crate::LinearRegression::fit_gradient), whose
    /// notes on batches and standardised inputs apply here too.
    ///
    /// Two classes train one output on binary cross-entropy with logits; more train one
    /// output per class on softmax cross-entropy (the multinomial model). Each step
    /// minimises, on a batch of `b` rows out of `n`,
    ///
    /// ```text
    /// L = 1/b · Σ_batch loss  +  1/(2·c·n) · ‖W‖²
    /// ```
    ///
    /// the mean loss plus the penalty on the weight only (none with `c: None`). With a full
    /// batch this is exactly `1/(c·n)` times the objective [`fit`](Self::fit) minimises, so
    /// the minimiser is the same, and with minibatches the expected gradient of an epoch is
    /// a positive multiple of it, by the argument on `LinearRegression::fit_gradient`. The
    /// weights start from Xavier uniform with `config.init_seed`, the intercepts at the
    /// classes' log frequencies (the optimum for weights of 0), or at 0 held there without
    /// `fit_intercept`. The model's `iterations` is the number of epochs.
    ///
    /// The labels are `0..classes` with every class present, as class labels or (for two
    /// classes) `[rows, 1]` value targets of 0 and 1. Without a penalty, separable classes
    /// have no minimiser, and the weights keep growing for as long as training runs.
    ///
    /// A weaker penalty (larger `c`) leaves the objective flatter along correlated inputs,
    /// and the defaults need more epochs to settle. On the Breast Cancer training rows with
    /// standardised inputs, the defaults reach the exact solution within about 5e-7 with
    /// `c = Some(1.0)` for every initialisation seed tried, but with `c = Some(10.0)` they
    /// end up to about 4e-2 away for some seeds; 3,000 epochs bring that below 1e-5.
    /// [`fit`](Self::fit) is the reference to compare with.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `c` is not `None` or a finite number above 0; for the
    /// data problems [`fit`](Self::fit) reports; if only one class appears, or a class
    /// below the largest label does not appear; and if `config.batch` is 0. During
    /// training, the health checks' errors, as on `LinearRegression::fit_gradient`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::{GradientConfig, LogisticRegression};
    /// use cavitation_nn::{Dataset, Targets};
    /// let x: Vec<f32> = (0..60).map(|i| (i % 20) as f32 / 10.0 - 1.0).collect();
    /// let labels: Vec<usize> = (0..60).map(|i| (i % 20 + i / 20) / 8).collect();
    /// let data = Dataset {
    ///     x: Tensor::from_vec(x, &[60, 1]).unwrap(),
    ///     targets: Targets::Classes(labels),
    /// };
    /// let logistic = LogisticRegression::default();
    /// let (model, history) = logistic.fit_gradient(&data, GradientConfig::default()).unwrap();
    /// assert_eq!(model.coef.shape().dims(), &[3, 1]);
    /// assert!(history.epochs.last().unwrap().mean_loss < history.epochs[0].mean_loss);
    /// ```
    pub fn fit_gradient(
        &self,
        data: &Dataset,
        config: GradientConfig,
    ) -> Result<(LogisticModel, History), Error> {
        self.check_c()?;
        let problem = check_classes(data)?;
        let rows = problem.rows;
        let classes = problem.labels.iter().max().map_or(0, |m| m + 1);
        let mut counts = vec![0usize; classes];
        for &l in &problem.labels {
            counts[l] += 1;
        }
        if let Some(only) = (counts.iter().filter(|&&n| n > 0).count() == 1)
            .then(|| counts.iter().position(|&n| n > 0))
            .flatten()
        {
            return Err(one_class(only));
        }
        if let Some(missing) = counts.iter().position(|&n| n == 0) {
            return Err(Error::InvalidData(format!(
                "logistic regression: class {missing} does not appear in the targets, but the \
                 labels must be 0 to {} with every class present; relabel the classes",
                classes - 1
            )));
        }
        let log_share = |n: usize| ((n as f64) / rows as f64).ln();
        let (loss, outputs, bias) = if classes == 2 {
            let odds = log_share(counts[1]) - log_share(counts[0]);
            (StepLoss::BceWithLogits, 1, vec![odds as f32])
        } else {
            let start = counts.iter().map(|&n| log_share(n) as f32).collect();
            (StepLoss::CrossEntropy, classes, start)
        };
        let owned = match (loss, &data.targets) {
            (StepLoss::BceWithLogits, Targets::Classes(labels)) => {
                let values = labels.iter().map(|&l| l as f32).collect();
                Some(Targets::Values(Tensor::from_vec(values, &[rows, 1])?))
            }
            (StepLoss::CrossEntropy, Targets::Values(_)) => {
                Some(Targets::Classes(problem.labels.clone()))
            }
            _ => None,
        }
        .map(|targets| Dataset {
            x: data.x.clone(),
            targets,
        });
        let penalty = self.c.map_or(0.0, |c| 1.0 / (2.0 * c * rows as f64)) as f32;
        let setup = Setup {
            features: problem.features,
            outputs,
            loss,
            penalty,
            fit_intercept: self.fit_intercept,
            bias: self.fit_intercept.then_some(bias),
        };
        let epochs = config.epochs;
        let (model, params, history) = train(owned.as_ref().unwrap_or(data), config, setup)?;
        let LinearModel { coef, intercept } = LinearModel::from_sequential(&model, &params)?;
        Ok((
            LogisticModel {
                coef,
                intercept,
                classes,
                iterations: epochs,
                c: self.c,
            },
            history,
        ))
    }
}

/// The error for a data set whose targets hold only class `only`.
fn one_class(only: usize) -> Error {
    Error::InvalidData(format!(
        "logistic regression needs two classes, but only class {only} appears in the targets"
    ))
}

/// The error for separable classes without a penalty, with `how` saying what showed it.
fn separable(how: &str) -> Error {
    Error::InvalidData(format!(
        "logistic regression without a penalty: the classes are separable, or nearly so, so \
         the log-loss has no minimum and the coefficients grow without bound ({how}); set c \
         to add a penalty (c = Some(1.0) is scikit-learn's default)"
    ))
}

/// Prefixes an [`Error::InvalidData`] from the least-squares solver with "logistic
/// regression: ". Other errors pass through unchanged.
fn in_context(error: Error) -> Error {
    match error {
        Error::InvalidData(message) => {
            Error::InvalidData(format!("logistic regression: {message}"))
        }
        other => other,
    }
}

/// The labels in `seen`, as "0, 1 and 2".
fn list(seen: &[usize]) -> String {
    match seen.split_last() {
        Some((last, [])) => last.to_string(),
        Some((last, rest)) => {
            let rest: Vec<String> = rest.iter().map(usize::to_string).collect();
            format!("{} and {last}", rest.join(", "))
        }
        None => String::new(),
    }
}

/// The linear score `x·w + b` of one row, from coefficients laid out as `[b, w…]` when
/// `offset` is 1 and `[w…]` when it is 0.
fn score(beta: &[f64], row: &[f64], offset: usize) -> f64 {
    let intercept = if offset == 1 { beta[0] } else { 0.0 };
    let sum: f64 = beta[offset..].iter().zip(row).map(|(w, v)| w * v).sum();
    sum + intercept
}

/// The logistic function `1 / (1 + e^(−z))`, computed without overflow on either side.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// The log-loss of score `z` for a row of class 1 (`positive`) or class 0:
/// `log(1 + e^(−s·z))` with `s = ±1`, as `max(−s·z, 0) + log1p(e^(−|z|))`, which neither
/// overflows nor loses the small values.
fn log_loss(z: f64, positive: bool) -> f64 {
    let t = if positive { -z } else { z };
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// Checks that `data` is a classification problem: `x` a matrix with at least one row and
/// one column, every input finite, and one class label per row, given as class targets or
/// as `[rows, 1]` value targets holding whole numbers at least 0.
fn check_classes<T: Real>(data: &Dataset<T>) -> Result<Classes, Error> {
    let &[rows, features] = data.x.shape().dims() else {
        return Err(Error::InvalidData(format!(
            "logistic regression: the inputs must have shape [rows, features], found {:?}",
            data.x.shape().dims()
        )));
    };
    if rows == 0 || features == 0 {
        return Err(Error::InvalidData(format!(
            "logistic regression needs at least one row and one input column, but the inputs \
             are [{rows}, {features}]"
        )));
    }
    if let Some(at) = data.x.data().iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidData(format!(
            "logistic regression: row {}, column {} of the inputs is {}; the values must be \
             finite",
            at / features,
            at % features,
            data.x.data()[at].to_f64()
        )));
    }
    let labels = labels_of(data, rows, "logistic regression")?;
    Ok(Classes {
        rows,
        features,
        x: data.x.data().iter().map(|v| v.to_f64()).collect(),
        labels,
    })
}

/// The class label of each of the `rows` rows of `data`: its class targets, or its
/// `[rows, 1]` value targets read as whole numbers at least 0. Errors start with `who`.
fn labels_of<T: Real>(data: &Dataset<T>, rows: usize, who: &str) -> Result<Vec<usize>, Error> {
    match &data.targets {
        Targets::Classes(labels) if labels.len() == rows => Ok(labels.clone()),
        Targets::Classes(labels) => Err(Error::InvalidData(format!(
            "{who}: there are {} class labels for {rows} input rows",
            labels.len()
        ))),
        Targets::Values(t) if t.shape().dims() == [rows, 1] => t
            .data()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let v = v.to_f64();
                if v >= 0.0 && v.fract() == 0.0 && v < 1e15 {
                    Ok(v as usize)
                } else {
                    Err(Error::InvalidData(format!(
                        "{who}: row {i} of the value targets is {v}, but the targets must be \
                         class labels (0 and 1 for two classes)"
                    )))
                }
            })
            .collect(),
        Targets::Values(t) => Err(Error::InvalidData(format!(
            "{who}: the value targets are {:?}, but class labels as values must be [{rows}, 1]",
            t.shape().dims()
        ))),
    }
}

/// A fitted logistic regression in `f64`: `coef` and `intercept` give each row's logits,
/// `x·coefᵀ + intercept`.
///
/// For two classes there is one logit, the log-odds of class 1, and `coef` is
/// `[1, features]`, as scikit-learn's `coef_`; for more there is one logit per class and
/// the probabilities are their softmax. Fields are public, so a model can also be written
/// by hand.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::LogisticModel;
/// // log-odds of class 1 = 2·x
/// let model = LogisticModel {
///     coef: Tensor::from_vec(vec![2.0], &[1, 1]).unwrap(),
///     intercept: vec![0.0],
///     classes: 2,
///     iterations: 0,
///     c: None,
/// };
/// let x = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], &[3, 1]).unwrap();
/// assert_eq!(model.predict(x.view()), [0, 0, 1]);
/// assert_eq!(model.predict_proba(x.view()).data()[3], 0.5);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticModel {
    /// The coefficients: `[1, features]` for two classes (the log-odds of class 1),
    /// `[classes, features]` for more (one row of logit weights per class).
    pub coef: Tensor<f64>,
    /// The intercept of each row of `coef` (zeros for a model fitted without one).
    pub intercept: Vec<f64>,
    /// The number of classes, at least 2.
    pub classes: usize,
    /// The Newton steps [`LogisticRegression::fit`] took, or the epochs
    /// [`LogisticRegression::fit_gradient`] trained.
    pub iterations: usize,
    /// The `c` of the regression that fitted the model, which
    /// [`objective`](Self::objective) uses: `None` for no penalty.
    pub c: Option<f64>,
}

impl LogisticModel {
    /// The number of input columns and logits, `(features, logits)`.
    ///
    /// # Panics
    ///
    /// Panics, naming the shapes, if `coef` is not a matrix, if `intercept` does not have
    /// one value per row of `coef`, or if `coef` does not have 1 row for 2 classes and one
    /// row per class otherwise.
    fn dims(&self, what: &str) -> (usize, usize) {
        let logits = if self.classes == 2 { 1 } else { self.classes };
        match *self.coef.shape().dims() {
            [k, features] if k == logits && self.intercept.len() == k && self.classes >= 2 => {
                (features, k)
            }
            _ => panic!(
                "LogisticModel::{what}: coef is {:?}, intercept has {} values and classes is \
                 {}; coef must be [1, features] for 2 classes and [classes, features] for \
                 more, with one intercept per row",
                self.coef.shape().dims(),
                self.intercept.len(),
                self.classes
            ),
        }
    }

    /// The logits of one input row, in `f64`, into `out` (one per row of `coef`).
    fn logits<T: Real>(&self, row: &[T], out: &mut [f64]) {
        let features = row.len();
        for ((o, w), b) in out
            .iter_mut()
            .zip(self.coef.data().chunks_exact(features.max(1)))
            .zip(&self.intercept)
        {
            let sum: f64 = w.iter().zip(row).map(|(w, x)| w * x.to_f64()).sum();
            *o = sum + b;
        }
    }

    /// The rows of `x` checked against the model's width, for `what`.
    ///
    /// # Panics
    ///
    /// Panics, naming both shapes, if `x` is not `[rows, features]`.
    fn rows(&self, x: &TensorView<'_, f32>, features: usize, what: &str) -> usize {
        match *x.shape().dims() {
            [rows, f] if f == features => rows,
            _ => panic!(
                "LogisticModel::{what}: x is {:?}, but the model takes [rows, {features}]",
                x.shape().dims()
            ),
        }
    }

    /// Each row's class probabilities, `[rows, classes]`: for two classes `1 − p` and `p`
    /// with `p = σ(logit)`, as scikit-learn's `predict_proba`; for more, the softmax of the
    /// logits. Computed in `f64` and rounded once to `f32`.
    ///
    /// # Panics
    ///
    /// Panics, naming both shapes, if `x` is not a matrix with one column per feature, and
    /// as described on [`LogisticModel`]'s fields if they do not agree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LogisticModel;
    /// // Three classes with equal logits at x = 0.
    /// let model = LogisticModel {
    ///     coef: Tensor::from_vec(vec![1.0, 0.0, -1.0], &[3, 1]).unwrap(),
    ///     intercept: vec![0.0; 3],
    ///     classes: 3,
    ///     iterations: 0,
    ///     c: None,
    /// };
    /// let x = Tensor::from_vec(vec![0.0f32], &[1, 1]).unwrap();
    /// let p = model.predict_proba(x.view());
    /// assert!(p.data().iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-7));
    /// ```
    pub fn predict_proba(&self, x: TensorView<'_, f32>) -> Tensor<f32> {
        let (features, k) = self.dims("predict_proba");
        let rows = self.rows(&x, features, "predict_proba");
        let mut out = Vec::with_capacity(rows * self.classes);
        let mut z = vec![0.0; k];
        for row in x.data().chunks_exact(features.max(1)).take(rows) {
            self.logits(row, &mut z);
            if k == 1 {
                out.extend([sigmoid(-z[0]) as f32, sigmoid(z[0]) as f32]);
            } else {
                let max = z.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                let sum: f64 = z.iter().map(|v| (v - max).exp()).sum();
                out.extend(z.iter().map(|v| ((v - max).exp() / sum) as f32));
            }
        }
        Tensor::from_vec(out, &[rows, self.classes]).expect("one probability per class")
    }

    /// The most probable class of each row of `x`: for two classes, 1 where the logit is
    /// above 0 (as scikit-learn); for more, the class with the largest logit, the first on a
    /// tie.
    ///
    /// # Panics
    ///
    /// As [`predict_proba`](Self::predict_proba).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LogisticModel;
    /// let model = LogisticModel {
    ///     coef: Tensor::from_vec(vec![1.0, -1.0], &[1, 2]).unwrap(),
    ///     intercept: vec![0.0],
    ///     classes: 2,
    ///     iterations: 0,
    ///     c: None,
    /// };
    /// let x = Tensor::from_vec(vec![2.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
    /// assert_eq!(model.predict(x.view()), [1, 0]);
    /// ```
    pub fn predict(&self, x: TensorView<'_, f32>) -> Vec<usize> {
        let (features, k) = self.dims("predict");
        let rows = self.rows(&x, features, "predict");
        let mut z = vec![0.0; k];
        x.data()
            .chunks_exact(features.max(1))
            .take(rows)
            .map(|row| {
                self.logits(row, &mut z);
                self.class_of(&z)
            })
            .collect()
    }

    /// The predicted class for the logits `z`.
    fn class_of(&self, z: &[f64]) -> usize {
        if z.len() == 1 {
            return usize::from(z[0] > 0.0);
        }
        let mut best = 0;
        for (c, &v) in z.iter().enumerate() {
            if v > z[best] {
                best = c;
            }
        }
        best
    }

    /// The accuracy of the model on `data`: the share of rows whose predicted class equals
    /// the label, with the logits computed in `f64` from the data's values.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `data.x` is not a matrix with one column per feature and
    /// at least one row, if the targets do not give one class label per row (as on
    /// [`LogisticRegression::fit`]), or if a label is not below `classes`.
    ///
    /// # Panics
    ///
    /// As described on [`LogisticModel`]'s fields if they do not agree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LogisticModel;
    /// use cavitation_nn::{Dataset, Targets};
    /// let model = LogisticModel {
    ///     coef: Tensor::from_vec(vec![1.0], &[1, 1]).unwrap(),
    ///     intercept: vec![0.0],
    ///     classes: 2,
    ///     iterations: 0,
    ///     c: None,
    /// };
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![-1.0f32, 1.0, 2.0, -3.0], &[4, 1]).unwrap(),
    ///     targets: Targets::Classes(vec![0, 1, 0, 0]),
    /// };
    /// assert_eq!(model.score(&data).unwrap(), 0.75);
    /// ```
    pub fn score<T: Real>(&self, data: &Dataset<T>) -> Result<f64, Error> {
        let (rows, labels) = self.checked(data, "score")?;
        let k = self.dims("score").1;
        let mut z = vec![0.0; k];
        let features = self.dims("score").0;
        let right = data
            .x
            .data()
            .chunks_exact(features.max(1))
            .zip(&labels)
            .filter(|&(row, &label)| {
                self.logits(row, &mut z);
                self.class_of(&z) == label
            })
            .count();
        Ok(right as f64 / rows as f64)
    }

    /// The objective the model's regression minimises, on `data`, in the form scikit-learn's
    /// fixtures record it: for two classes `c · Σ log(1 + e^(−s·z)) + ½‖coef‖²`, with
    /// `s = ±1` for labels 1 and 0 and `z` the logit; for more,
    /// `c · Σ −log softmax(z)_label + ½‖coef‖²` (every class's coefficients); without a
    /// penalty (`c: None`) the sum of the losses alone. The intercept is never penalised.
    /// Each loss is computed stably, as `max(t, 0) + log1p(e^(−|t|))` or with the largest
    /// logit taken out of the softmax, in `f64`.
    ///
    /// # Errors
    ///
    /// As [`score`](Self::score).
    ///
    /// # Panics
    ///
    /// As described on [`LogisticModel`]'s fields if they do not agree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LogisticModel;
    /// use cavitation_nn::{Dataset, Targets};
    /// // All logits 0: each row's loss is ln 2, and the penalty ½·1² is added once.
    /// let model = LogisticModel {
    ///     coef: Tensor::from_vec(vec![1.0], &[1, 1]).unwrap(),
    ///     intercept: vec![0.0],
    ///     classes: 2,
    ///     iterations: 0,
    ///     c: Some(2.0),
    /// };
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![0.0f32, 0.0, 0.0], &[3, 1]).unwrap(),
    ///     targets: Targets::Classes(vec![0, 1, 1]),
    /// };
    /// let expected = 2.0 * 3.0 * 2f64.ln() + 0.5;
    /// assert!((model.objective(&data).unwrap() - expected).abs() < 1e-14);
    /// ```
    pub fn objective<T: Real>(&self, data: &Dataset<T>) -> Result<f64, Error> {
        let (_, labels) = self.checked(data, "objective")?;
        let (features, k) = self.dims("objective");
        let mut z = vec![0.0; k];
        let mut loss = 0.0;
        for (row, &label) in data.x.data().chunks_exact(features.max(1)).zip(&labels) {
            self.logits(row, &mut z);
            loss += if k == 1 {
                log_loss(z[0], label == 1)
            } else {
                let max = z.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                let sum: f64 = z.iter().map(|v| (v - max).exp()).sum();
                max + sum.ln() - z[label]
            };
        }
        Ok(match self.c {
            Some(c) => {
                let squares: f64 = self.coef.data().iter().map(|w| w * w).sum();
                c * loss + 0.5 * squares
            }
            None => loss,
        })
    }

    /// Checks `data` against the model for `what`: its rows and their labels.
    fn checked<T: Real>(
        &self,
        data: &Dataset<T>,
        what: &str,
    ) -> Result<(usize, Vec<usize>), Error> {
        let (features, _) = self.dims(what);
        let who = format!("LogisticModel::{what}");
        let rows = match *data.x.shape().dims() {
            [rows, f] if f == features && rows > 0 => rows,
            _ => {
                return Err(Error::InvalidData(format!(
                    "{who}: the inputs are {:?}, but the model takes [rows, {features}] with at \
                     least one row",
                    data.x.shape().dims()
                )));
            }
        };
        let labels = labels_of(data, rows, &who)?;
        if let Some(&label) = labels.iter().find(|&&l| l >= self.classes) {
            return Err(Error::InvalidData(format!(
                "{who}: label {label} is not one of the model's {} classes (0 to {})",
                self.classes,
                self.classes - 1
            )));
        }
        Ok((rows, labels))
    }

    /// The model as a network of one [`Linear`](cavitation_nn::Linear) layer giving the
    /// logits (weight `coef`, bias `intercept`, each rounded once to `f32`), whose
    /// parameters are added to `params`; as [`LinearModel::to_sequential`], which it calls.
    /// Saved, it is a PyTorch `nn.Linear(features, 1)` (two classes) or
    /// `nn.Linear(features, classes)`; follow it with a sigmoid or softmax for
    /// probabilities.
    ///
    /// # Panics
    ///
    /// As described on [`LogisticModel`]'s fields if they do not agree, and as
    /// [`LinearModel::to_sequential`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_ml::LogisticModel;
    /// let model = LogisticModel {
    ///     coef: Tensor::from_vec(vec![2.0, -1.0], &[1, 2]).unwrap(),
    ///     intercept: vec![0.5],
    ///     classes: 2,
    ///     iterations: 0,
    ///     c: None,
    /// };
    /// let mut params = ParamStore::new();
    /// let net = model.to_sequential(&mut params);
    /// assert_eq!(params.value(net.linear(0).unwrap().bias().unwrap()).data(), &[0.5]);
    /// ```
    pub fn to_sequential(&self, params: &mut ParamStore) -> Sequential {
        self.dims("to_sequential");
        LinearModel {
            coef: self.coef.clone(),
            intercept: self.intercept.clone(),
        }
        .to_sequential(params)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the exact Newton path's errors on degenerate classes, its convergence, and
    //! the fitted model's predictions and conversion.

    use cavitation_core::Tensor;
    use cavitation_nn::{Dataset, Targets};

    use super::*;

    /// A one-feature data set from inputs and class labels.
    fn classes(x: Vec<f32>, labels: Vec<usize>) -> Dataset {
        let rows = labels.len();
        Dataset {
            x: Tensor::from_vec(x, &[rows, 1]).unwrap(),
            targets: Targets::Classes(labels),
        }
    }

    /// No penalty.
    fn unpenalised() -> LogisticRegression {
        LogisticRegression {
            c: None,
            ..LogisticRegression::default()
        }
    }

    /// Asserts that `err` reports separable classes: it says "separable", suggests `c`, and
    /// does not call a column a linear combination.
    fn assert_separable(err: &Error) {
        let text = err.to_string();
        assert!(text.contains("separable"), "{text}");
        assert!(text.contains("set c"), "{text}");
        assert!(!text.contains("linear combination"), "{text}");
    }

    /// Two points per class on either side of 0 are separable: without a penalty the
    /// coefficients grow without bound, which is reported, not returned.
    #[test]
    fn separable_without_penalty_is_an_error() {
        let d = classes(vec![-2.0, -1.0, 1.0, 2.0], vec![0, 0, 1, 1]);
        assert_separable(&unpenalised().fit(&d).unwrap_err());
    }

    /// Classes that overlap except at one end are quasi-separable: column 0 holds both
    /// classes at -1, 0 and 1 (so no line through it alone separates them), and column 1,
    /// `max(x0 − 1.5, 0)`, is non-zero only at the top end, where every row is class 1. Its
    /// coefficient grows without bound while the objective stays above 0: the same error.
    #[test]
    fn quasi_separable_without_penalty_is_an_error() {
        let x0 = [-2.0f32, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0];
        let x: Vec<f32> = x0.iter().flat_map(|&v| [v, (v - 1.5).max(0.0)]).collect();
        let d = Dataset {
            x: Tensor::from_vec(x, &[9, 2]).unwrap(),
            targets: Targets::Classes(vec![0, 0, 1, 0, 1, 0, 1, 1, 1]),
        };
        assert_separable(&unpenalised().fit(&d).unwrap_err());
        // Given many more steps, the separated rows' probabilities fall below what changes
        // the objective in `f64`, and the step cannot be accepted: still reported.
        let long = LogisticRegression {
            max_iter: 2000,
            ..unpenalised()
        };
        let err = long.fit(&d).unwrap_err();
        assert!(err.to_string().contains("max_iter = 2000"), "{err}");
        assert_separable(&err);
        // By 5,000 steps the separated rows' losses no longer change the objective in
        // `f64`, so it has stopped falling; rule 2 still fires, because it asks only that
        // the objective not rise beyond rounding while the coefficients grow.
        let longer = LogisticRegression {
            max_iter: 5000,
            ..unpenalised()
        };
        let err = longer.fit(&d).unwrap_err();
        assert!(err.to_string().contains("max_iter = 5000"), "{err}");
        assert_separable(&err);
        // With a penalty the same rows fit.
        assert!(LogisticRegression::default().fit(&d).is_ok());
    }

    /// Overlapping classes in column 0 (both at -1, 0 and 1), and a column 1 equal to it
    /// except on one extra class-1 row, where it is larger by `offset`: moving along column
    /// 1 minus column 0 separates that row alone.
    fn one_row_apart(offset: f64) -> Dataset<f64> {
        let x0 = [-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.5];
        let mut x = Vec::new();
        for (i, &v) in x0.iter().enumerate() {
            x.extend([v, if i == 8 { v + offset } else { v }]);
        }
        wide(x, vec![0, 0, 1, 0, 1, 0, 1, 1, 1], 2)
    }

    /// A `tol` finer than `f64` resolves the answer stalls Newton at a converged point: an
    /// error (never a model whose step did not pass the `tol` test) that suggests raising
    /// `tol`, and does not suggest setting `c` when it is set.
    #[test]
    fn tolerance_below_resolution_stalls() {
        let d = correlated(4);
        let tight = LogisticRegression {
            tol: 0.0,
            ..LogisticRegression::default()
        };
        let err = tight.fit(&d).unwrap_err().to_string();
        assert!(
            err.contains("stalled") && err.contains("raise tol"),
            "{err}"
        );
        assert!(
            !err.contains("set c") && !err.contains("separable"),
            "{err}"
        );
        let unpenalised_tight = LogisticRegression { c: None, ..tight };
        let err = unpenalised_tight.fit(&d).unwrap_err().to_string();
        assert!(
            err.contains("stalled") && err.contains("raise tol"),
            "{err}"
        );
    }

    /// Quasi-separable data whose separating direction cancels in `f64` (the extra row 1e-6
    /// apart) stalls with coefficients near ±2.7e7 and the separated row's probability past
    /// the clamp: the error says the classes may be separable and suggests `c`.
    #[test]
    fn stall_while_growing_suggests_separable() {
        let err = unpenalised().fit(&one_row_apart(1e-6)).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("stalled"), "{text}");
        assert_separable(&err);
    }

    /// The weighted problem losing rank after the first step while the objective falls is
    /// reported as separation. Column 0 overlaps (both classes at -1, 0 and 1); column 1
    /// equals it except on one extra class-1 row, where it is larger by 1e-11. Moving along
    /// column 1 minus column 0 separates that row alone, so its weight collapses; once it
    /// does, column 1 is a linear combination of the others to working precision, though it
    /// was not at the first step.
    #[test]
    fn rank_lost_while_falling_is_separable() {
        let d = one_row_apart(1e-11);
        let err = unpenalised().fit(&d).unwrap_err();
        assert!(err.to_string().contains("lost rank"), "{err}");
        assert_separable(&err);
        assert!(LogisticRegression::default().fit(&d).is_ok());
    }

    /// With a penalty, separable classes have a finite optimum, which Newton finds: the
    /// gradient `c·Xᵀ(p − y) + w` is 0 there, and every point is classified correctly.
    #[test]
    fn separable_with_penalty_fits() {
        let d = classes(vec![-2.0, -1.0, 1.0, 2.0], vec![0, 0, 1, 1]);
        let model = LogisticRegression::default().fit(&d).unwrap();
        let (w, b) = (model.coef.data()[0], model.intercept[0]);
        let x = [-2.0, -1.0, 1.0, 2.0];
        let y = [0.0, 0.0, 1.0, 1.0];
        let p = |v: f64| 1.0 / (1.0 + (-(w * v + b)).exp());
        let grad_w: f64 = x.iter().zip(y).map(|(&v, t)| (p(v) - t) * v).sum::<f64>() + w;
        let grad_b: f64 = x.iter().zip(y).map(|(&v, t)| p(v) - t).sum();
        assert!(
            grad_w.abs() < 1e-12 && grad_b.abs() < 1e-12,
            "{grad_w} {grad_b}"
        );
        assert_eq!(model.predict(d.x.view()), [0, 0, 1, 1]);
        assert_eq!(model.score(&d).unwrap(), 1.0);
        // Asymmetric classes put the optimal intercept away from 0, so the unpenalised
        // intercept shows: its gradient `Σ(p − y)` has no `+ b` term and is 0 at the answer.
        let x = [-2.0, -1.0, 3.0, 4.0];
        let d = classes(x.iter().map(|&v| v as f32).collect(), vec![0, 0, 1, 1]);
        let model = LogisticRegression::default().fit(&d).unwrap();
        let (w, b) = (model.coef.data()[0], model.intercept[0]);
        let p = |v: f64| 1.0 / (1.0 + (-(w * v + b)).exp());
        let grad_w: f64 = x.iter().zip(y).map(|(&v, t)| (p(v) - t) * v).sum::<f64>() + w;
        let grad_b: f64 = x.iter().zip(y).map(|(&v, t)| p(v) - t).sum();
        assert!(b.abs() > 0.1, "intercept {b}");
        assert!(
            grad_w.abs() < 1e-12 && grad_b.abs() < 1e-12,
            "{grad_w} {grad_b}"
        );
    }

    /// The reviewer-style generator: xorshift64 uniforms in [0, 1) and Box-Muller normals,
    /// so the data sets below are fixed bit for bit.
    struct XorShift(u64);

    impl XorShift {
        /// The next uniform in [0, 1).
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        /// The next standard normal.
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.next().max(1e-300), self.next());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// An `f64` data set of `features` columns from row-major inputs and labels.
    fn wide(x: Vec<f64>, labels: Vec<usize>, features: usize) -> Dataset<f64> {
        let rows = labels.len();
        Dataset {
            x: Tensor::from_vec(x, &[rows, features]).unwrap(),
            targets: Targets::Classes(labels),
        }
    }

    /// The largest component of the objective's gradient at `model`:
    /// `c·Xᵀ(p − y) + w` (with `c = 1` and no `w` term without a penalty), the intercept's
    /// component included.
    fn largest_gradient(model: &LogisticModel, d: &Dataset<f64>) -> f64 {
        let features = d.features();
        let Targets::Classes(y) = &d.targets else {
            unreachable!()
        };
        let (w, b) = (model.coef.data(), model.intercept[0]);
        let c = model.c.unwrap_or(1.0);
        let mut g = vec![0.0; features + 1];
        for (row, &l) in d.x.data().chunks(features).zip(y) {
            let z: f64 = row.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + b;
            let r = sigmoid(z) - l as f64;
            g[features] += c * r;
            for (gj, v) in g.iter_mut().zip(row) {
                *gj += c * r * v;
            }
        }
        if model.c.is_some() {
            for (gj, wj) in g.iter_mut().zip(w) {
                *gj += wj;
            }
        }
        g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// Not separable, though nearly: 2,000 rows with `x0` evenly spaced on [-1, 1], labels
    /// random inside `|x0| < 0.01` and `x0 > 0` outside, and a standard-normal noise column,
    /// for two generator seeds that were once called separable. The unpenalised optimum is
    /// finite (a slope near 210) with rows far past the 1e-10 probability clamp, where a
    /// Newton step's decrease is below what the objective resolves in `f64`; Newton must
    /// fit it, with the objective's gradient below 1e-9 in every component (about 4e-15
    /// is reached).
    #[test]
    fn narrow_band_overlap_fits() {
        for seed in [1u64, 2] {
            let mut rng = XorShift(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for i in 0..2000 {
                let v = -1.0 + 2.0 * (f64::from(i) + 0.5) / 2000.0;
                let label = if v.abs() < 0.01 {
                    usize::from(rng.next() < 0.5)
                } else {
                    usize::from(v > 0.0)
                };
                x.extend([v, rng.normal()]);
                y.push(label);
            }
            let d = wide(x, y, 2);
            let model = unpenalised()
                .fit(&d)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            let g = largest_gradient(&model, &d);
            assert!(g < 1e-9, "seed {seed}: gradient {g:e}, model {model:?}");
            assert!(model.coef.data()[0] > 50.0, "seed {seed}: {model:?}");
        }
    }

    /// The correlated set of [`correlated_rows_with_far_points_fit`] for generator `seed`.
    fn correlated(seed: u64) -> Dataset<f64> {
        let mut rng = XorShift(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..3000 {
            let base = rng.normal();
            let mut row: Vec<f64> = (0..4)
                .map(|j| base + 0.01 * (f64::from(j) + 1.0) * rng.normal())
                .collect();
            let mut label = usize::from(rng.next() < sigmoid(row[0] + 0.3 * row[1]));
            if i < 3 {
                row.iter_mut().for_each(|v| *v *= 40.0);
                label = usize::from(row[0] > 0.0);
            }
            x.extend(row);
            y.push(label);
        }
        wide(x, y, 4)
    }

    /// 3,000 rows of four correlated columns (`base + 0.01·k·N(0, 1)`), Bernoulli labels
    /// from `σ(x0 + 0.3·x1)`, and three rows scaled ×40 with the correct label, for the
    /// generator seeds that once stalled Newton (3, 4, 7, 17). Not separable: without a
    /// penalty and with `c = 1` each fit converges, with the objective's gradient below
    /// `1e-8 · max(1, c)` in every component at the returned answer (converged fits reach
    /// about 1e-12; a stalled fit at `c = 1` stopped at 2.9e-6).
    #[test]
    fn correlated_rows_with_far_points_fit() {
        for seed in [3u64, 4, 7, 17] {
            let d = correlated(seed);
            for r in [unpenalised(), LogisticRegression::default()] {
                let model = r
                    .fit(&d)
                    .unwrap_or_else(|e| panic!("seed {seed}, c {:?}: {e}", r.c));
                let g = largest_gradient(&model, &d);
                assert!(g < 1e-8, "seed {seed}, c {:?}: gradient {g:e}", r.c);
            }
        }
    }

    /// A single class is refused, naming it, by both paths.
    #[test]
    fn one_class_is_an_error() {
        let d = classes(vec![1.0, 2.0, 3.0], vec![1, 1, 1]);
        let err = LogisticRegression::default()
            .fit(&d)
            .unwrap_err()
            .to_string();
        assert!(err.contains("only class 1"), "{err}");
        let config = GradientConfig {
            health: cavitation_nn::Health::quiet(),
            ..GradientConfig::default()
        };
        assert!(
            LogisticRegression::default()
                .fit_gradient(&d, config)
                .is_err()
        );
    }

    /// Labels 1 and 2 are refused by the exact path, naming them and saying 0 and 1 are
    /// expected; value targets of 1 and 2 too.
    #[test]
    fn labels_one_and_two_rejected() {
        let d = classes(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2]);
        let err = LogisticRegression::default()
            .fit(&d)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 and 2"), "{err}");
        assert!(err.contains("0 and 1"), "{err}");
        let values = Dataset {
            x: d.x.clone(),
            targets: Targets::Values(Tensor::from_vec(vec![1.0, 2.0, 1.0, 2.0], &[4, 1]).unwrap()),
        };
        let err = LogisticRegression::default()
            .fit(&values)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 and 2") && err.contains("0 and 1"), "{err}");
    }

    /// Three classes on the exact path are refused with a pointer to `fit_gradient`, which
    /// fits them.
    #[test]
    fn three_classes_exact_suggests_gradient() {
        let d = classes(vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], vec![0, 1, 2, 0, 1, 2]);
        let err = LogisticRegression::default()
            .fit(&d)
            .unwrap_err()
            .to_string();
        assert!(err.contains("3 classes"), "{err}");
        assert!(err.contains("fit_gradient"), "{err}");
        let config = GradientConfig {
            epochs: 10,
            health: cavitation_nn::Health::quiet(),
            ..GradientConfig::default()
        };
        let (model, _) = LogisticRegression::default()
            .fit_gradient(&d, config)
            .unwrap();
        assert_eq!((model.classes, model.coef.shape().dims()), (3, &[3, 1][..]));
    }

    /// Value targets of 0 and 1 are the same problem as class labels 0 and 1.
    #[test]
    fn value_targets_match_labels() {
        let x = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];
        let labels = vec![0, 1, 0, 1, 0, 1];
        let values: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        let d = classes(x.clone(), labels);
        let v = Dataset {
            x: Tensor::from_vec(x, &[6, 1]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(values, &[6, 1]).unwrap()),
        };
        let r = LogisticRegression::default();
        assert_eq!(r.fit(&d).unwrap(), r.fit(&v).unwrap());
    }

    /// A constant column without a penalty is rank deficient at the first iteration: the
    /// error names the column (as a linear combination of the intercept's ones), not
    /// separation; a penalty fits it.
    #[test]
    fn constant_column_named() {
        let x = vec![1.0, 5.0, -1.0, 5.0, 0.5, 5.0, -0.5, 5.0, 2.0, 5.0];
        let d = Dataset {
            x: Tensor::from_vec(x, &[5, 2]).unwrap(),
            targets: Targets::Classes(vec![1, 0, 0, 1, 1]),
        };
        let err = unpenalised().fit(&d).unwrap_err().to_string();
        assert!(err.contains("column 1"), "{err}");
        assert!(!err.contains("separable"), "{err}");
        assert!(LogisticRegression::default().fit(&d).is_ok());
    }

    /// Bad settings and data are `InvalidData` naming the problem.
    #[test]
    fn bad_settings_are_errors() {
        let d = classes(vec![-1.0, 0.0, 1.0, 2.0], vec![0, 1, 0, 1]);
        for c in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let r = LogisticRegression {
                c: Some(c),
                ..LogisticRegression::default()
            };
            assert!(r.fit(&d).unwrap_err().to_string().contains("c is"), "{c}");
        }
        let r = LogisticRegression {
            tol: -1.0,
            ..LogisticRegression::default()
        };
        assert!(r.fit(&d).unwrap_err().to_string().contains("tol"));
        let r = LogisticRegression {
            max_iter: 0,
            ..LogisticRegression::default()
        };
        assert!(r.fit(&d).unwrap_err().to_string().contains("max_iter"));
        let mut bad = d.clone();
        bad.x.data_mut()[2] = f32::NAN;
        let err = LogisticRegression::default()
            .fit(&bad)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("row 2, column 0 of the inputs is NaN"),
            "{err}"
        );
        let short = Dataset {
            x: d.x.clone(),
            targets: Targets::Classes(vec![0, 1]),
        };
        assert!(LogisticRegression::default().fit(&short).is_err());
        let fraction = Dataset {
            x: d.x.clone(),
            targets: Targets::Values(Tensor::from_vec(vec![0.0, 0.5, 1.0, 1.0], &[4, 1]).unwrap()),
        };
        let err = LogisticRegression::default()
            .fit(&fraction)
            .unwrap_err()
            .to_string();
        assert!(err.contains("0.5"), "{err}");
    }

    /// Without a penalty, more coefficients than rows is refused before solving, with a
    /// message suggesting `c`.
    #[test]
    fn more_features_than_rows_suggests_c() {
        let x: Vec<f32> = (0..12).map(|i| (i * i % 7) as f32).collect();
        let d = Dataset {
            x: Tensor::from_vec(x, &[3, 4]).unwrap(),
            targets: Targets::Classes(vec![0, 1, 0]),
        };
        let err = unpenalised().fit(&d).unwrap_err().to_string();
        assert!(err.contains("set c"), "{err}");
        assert!(LogisticRegression::default().fit(&d).is_ok());
    }

    /// `predict_proba` gives each row's class probabilities (summing to 1), `predict` the
    /// most probable class, and `score` the share predicted correctly; the `Sequential`
    /// gives the same logits.
    #[test]
    fn predictions_and_sequential_agree() {
        let model = LogisticModel {
            coef: Tensor::from_vec(vec![2.0, -1.0], &[1, 2]).unwrap(),
            intercept: vec![0.5],
            classes: 2,
            iterations: 0,
            c: Some(1.0),
        };
        let x = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 3.0, -1.0, 1.0], &[3, 2]).unwrap();
        let p = model.predict_proba(x.view());
        assert_eq!(p.shape().dims(), &[3, 2]);
        let z = [2.5f64, -2.5, -2.5];
        for (r, &zr) in z.iter().enumerate() {
            let one = 1.0 / (1.0 + (-zr).exp());
            assert!((f64::from(p.data()[r * 2 + 1]) - one).abs() < 1e-7);
            assert!((f64::from(p.data()[r * 2]) - (1.0 - one)).abs() < 1e-7);
        }
        assert_eq!(model.predict(x.view()), [1, 0, 0]);
        let d = Dataset {
            x: x.clone(),
            targets: Targets::Classes(vec![1, 1, 0]),
        };
        assert!((model.score(&d).unwrap() - 2.0 / 3.0).abs() < 1e-15);
        let mut params = cavitation_core::ParamStore::new();
        let net = model.to_sequential(&mut params);
        let mut compiled = cavitation_nn::InferenceModel::compile(&net, &params).unwrap();
        let logits = compiled.infer_batch(x.view());
        for (g, e) in logits.data().iter().zip(z) {
            assert!((f64::from(*g) - e).abs() < 1e-6, "{g} against {e}");
        }
    }

    /// Multinomial probabilities are the softmax of the logits, and `predict` takes the
    /// first largest.
    #[test]
    fn multinomial_predictions() {
        let model = LogisticModel {
            coef: Tensor::from_vec(vec![1.0, -1.0, 0.0], &[3, 1]).unwrap(),
            intercept: vec![0.0, 0.0, 0.5],
            classes: 3,
            iterations: 0,
            c: None,
        };
        let x = Tensor::from_vec(vec![2.0f32, -2.0, 0.0], &[3, 1]).unwrap();
        assert_eq!(model.predict(x.view()), [0, 1, 2]);
        let p = model.predict_proba(x.view());
        for row in p.data().chunks(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let d = Dataset {
            x,
            targets: Targets::Classes(vec![0, 1, 1]),
        };
        // −log softmax: row 0 at its largest logit 2, row 1 at 2, row 2 at 0 against 0.5.
        let lse = |z: [f64; 3]| z.iter().map(|v| v.exp()).sum::<f64>().ln();
        let expected = (lse([2.0, -2.0, 0.5]) - 2.0)
            + (lse([-2.0, 2.0, 0.5]) - 2.0)
            + (lse([0.0, 0.0, 0.5]) - 0.0);
        assert!((model.objective(&d).unwrap() - expected).abs() < 1e-12);
    }

    /// A penalised fit on overlapping classes converges on the step tolerance in 2 to 14
    /// Newton steps, and one step is not enough: `max_iter = 1` is an error naming
    /// `max_iter`.
    #[test]
    fn newton_iterations_reported() {
        let x: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) / 7.0).collect();
        let labels: Vec<usize> = (0..40)
            .map(|i| usize::from((i * 7919) % 13 < i / 3))
            .collect();
        let d = classes(x, labels);
        let model = LogisticRegression::default().fit(&d).unwrap();
        assert!(
            model.iterations >= 2 && model.iterations < 15,
            "{}",
            model.iterations
        );
        let short = LogisticRegression {
            max_iter: 1,
            ..LogisticRegression::default()
        };
        let err = short.fit(&d).unwrap_err().to_string();
        assert!(err.contains("max_iter"), "{err}");
    }
}
