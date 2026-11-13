//! Linear regression and ridge regression solved exactly by least squares
//! ([`LinearRegression::fit`]), and the fitted [`LinearModel`]: predictions, R², and the
//! conversion to and from a one-layer [`Sequential`].

use cavitation_core::{Error, ParamStore, Real, Tensor, TensorView};
use cavitation_nn::{Dataset, Linear, Sequential, Targets};

use crate::linalg::Qr;

/// Linear regression: the coefficients `w` and intercept `b` that minimise
/// `‖y − X·wᵀ − b‖² + alpha·‖w‖²`, summed over every row and output.
///
/// With `alpha = 0` (the default) this is ordinary least squares, scikit-learn's
/// `LinearRegression`; with `alpha > 0` it is ridge regression, scikit-learn's `Ridge`, whose
/// penalty leaves the intercept out. Each output column of the targets is its own regression
/// on the same inputs.
///
/// [`fit`](Self::fit) solves the problem exactly, by Householder QR in `f64`;
/// [`fit_gradient`](Self::fit_gradient) trains a [`Linear`] layer towards the same answer by
/// gradient descent, the way a neural network is trained.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::LinearRegression;
/// use cavitation_nn::{Dataset, Targets};
/// // y = 1 + 2·x exactly.
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap()),
/// };
/// let model = LinearRegression::default().fit(&data).unwrap();
/// assert!((model.coef.data()[0] - 2.0).abs() < 1e-12);
/// assert!((model.intercept[0] - 1.0).abs() < 1e-12);
///
/// // Ridge pulls the slope towards 0.
/// let ridge = LinearRegression { alpha: 1.0, ..LinearRegression::default() }.fit(&data).unwrap();
/// assert!(ridge.coef.data()[0] < 2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearRegression {
    /// Whether to fit an intercept. Without one the fitted plane passes through the origin.
    pub fit_intercept: bool,
    /// The ridge penalty's strength: 0 for ordinary least squares, above 0 for ridge.
    pub alpha: f64,
}

impl Default for LinearRegression {
    /// Ordinary least squares with an intercept: `fit_intercept` true, `alpha` 0.
    fn default() -> LinearRegression {
        LinearRegression {
            fit_intercept: true,
            alpha: 0.0,
        }
    }
}

impl LinearRegression {
    /// Fits the regression exactly: the least-squares (or ridge) solution, computed in `f64`
    /// from the data's values (an `f32` data set is widened exactly, never rounded).
    ///
    /// The steps, as scikit-learn takes them:
    ///
    /// 1. With `fit_intercept`, centre each input column and each target column on its mean.
    /// 2. Leave out of the solve every column that is zero after centring: a column whose
    ///    values are all the same (all zero, without an intercept). Such a column cannot
    ///    explain anything the intercept does not, so it gets the coefficient 0 exactly.
    /// 3. Solve the rest by Householder QR ([`Qr`]). Ridge appends the rows `√alpha·I` below
    ///    the centred inputs and zeros below the targets, whose least-squares solution is the
    ///    ridge solution: `‖y − Xw‖² + ‖0 − √alpha·w‖²` is the ridge objective.
    /// 4. The intercept is `ȳ − x̄·w` (0 without `fit_intercept`).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `alpha` is negative or not finite; if `data.x` is not a
    /// matrix with at least one column, or has fewer than 2 rows; if the targets are class
    /// labels rather than values, or are not a `[rows, outputs]` matrix with at least one
    /// output and one row per input row; if a value is NaN or infinite (naming the row and
    /// column); if a column's values are within a small factor of `f64::MAX`; with
    /// `alpha = 0`, if more columns are left to solve than the rows can determine (the
    /// rows, or the rows less one with an intercept), suggesting `alpha > 0`; and, naming
    /// the column, if a column is a linear combination of the columns before it (after
    /// centring), suggesting to drop it or set `alpha > 0`. Whether a column depends on the
    /// ones before it is judged against the column's own norm, so the columns' units do not
    /// matter.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LinearRegression;
    /// use cavitation_nn::{Dataset, Targets};
    /// // Column 1 is twice column 0: no unique answer without a penalty.
    /// let x = vec![1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0, 5.0, 10.0];
    /// let y = vec![1.0f32, 2.0, 3.0, 5.0];
    /// let data = Dataset {
    ///     x: Tensor::from_vec(x, &[4, 2]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(y, &[4, 1]).unwrap()),
    /// };
    /// let err = LinearRegression::default().fit(&data).unwrap_err();
    /// assert!(err.to_string().contains("column 1"));
    /// assert!(LinearRegression { alpha: 0.1, ..LinearRegression::default() }.fit(&data).is_ok());
    /// ```
    pub fn fit<T: Real>(&self, data: &Dataset<T>) -> Result<LinearModel, Error> {
        check_alpha(self.alpha)?;
        let shape = check_values(data)?;
        let Shape {
            features, outputs, ..
        } = shape;
        let x: Vec<f64> = data.x.data().iter().map(|v| v.to_f64()).collect();
        let y: Vec<f64> = targets(data).data().iter().map(|v| v.to_f64()).collect();
        let x_mean = self.offsets(&x, features);
        let y_mean = self.offsets(&y, outputs);
        let used: Vec<usize> = (0..features)
            .filter(|&j| !self.is_flat(&x, features, j))
            .collect();
        let mut coef = vec![0.0; outputs * features];
        if !used.is_empty() {
            let solution = self.solve(&x, &y, &x_mean, &y_mean, &used, shape)?;
            for (c, &j) in used.iter().enumerate() {
                for q in 0..outputs {
                    coef[q * features + j] = solution[c * outputs + q];
                }
            }
        }
        let intercept = (0..outputs)
            .map(|q| {
                let row = &coef[q * features..(q + 1) * features];
                let shift: f64 = row.iter().zip(&x_mean).map(|(w, m)| w * m).sum();
                y_mean[q] - shift
            })
            .collect();
        Ok(LinearModel {
            coef: Tensor::from_vec(coef, &[outputs, features])?,
            intercept,
        })
    }

    /// Each column's offset: its mean with `fit_intercept` (the exact value for a column
    /// whose values are all the same), 0 without.
    fn offsets(&self, values: &[f64], width: usize) -> Vec<f64> {
        if !self.fit_intercept {
            return vec![0.0; width];
        }
        let rows = values.len() / width;
        (0..width)
            .map(|j| {
                let first = values[j];
                if values.iter().skip(j).step_by(width).all(|&v| v == first) {
                    return first;
                }
                let sum: f64 = values.iter().skip(j).step_by(width).sum();
                sum / rows as f64
            })
            .collect()
    }

    /// Whether input column `j` is zero after centring: all its values the same with
    /// `fit_intercept`, all zero without.
    fn is_flat(&self, x: &[f64], features: usize, j: usize) -> bool {
        let first = if self.fit_intercept { x[j] } else { 0.0 };
        x.iter().skip(j).step_by(features).all(|&v| v == first)
    }

    /// Solves the centred problem for the input columns `used` and returns the solution as
    /// `[used.len(), outputs]`, row-major.
    fn solve(
        &self,
        x: &[f64],
        y: &[f64],
        x_mean: &[f64],
        y_mean: &[f64],
        used: &[usize],
        shape: Shape,
    ) -> Result<Vec<f64>, Error> {
        let Shape {
            rows,
            features,
            outputs,
        } = shape;
        let m = used.len();
        let ridge = self.alpha > 0.0;
        let determined = rows - usize::from(self.fit_intercept);
        if !ridge && m > determined {
            let why = if self.fit_intercept {
                format!("{rows} rows, less one for the intercept, determine at most {determined}")
            } else {
                format!("{rows} rows determine at most {determined}")
            };
            return Err(Error::InvalidData(format!(
                "linear regression: {m} input columns vary, but {why} coefficients, so the \
                 least-squares solution is not unique; set alpha > 0 (ridge) or use fewer columns"
            )));
        }
        let total = rows + if ridge { m } else { 0 };
        // Column-major, built once and handed to the factorisation without a copy.
        let mut a = vec![0.0; total * m];
        let limit = f64::MAX / (8.0 * (total as f64).sqrt());
        let root = self.alpha.sqrt();
        for (c, &j) in used.iter().enumerate() {
            let column = &mut a[c * total..(c + 1) * total];
            for (i, value) in column[..rows].iter_mut().enumerate() {
                *value = x[i * features + j] - x_mean[j];
            }
            if column[..rows].iter().any(|v| v.is_nan() || v.abs() > limit) {
                return Err(Error::InvalidData(format!(
                    "linear regression: column {j} holds values too close to the largest f64 \
                     to solve with; rescale the column"
                )));
            }
            if ridge {
                column[rows + c] = root;
            }
        }
        let qr = Qr::from_columns(a, total, m).map_err(in_context)?;
        if let Some(c) = qr.dependent_column() {
            let j = used[c];
            let hint = if ridge {
                "drop it, or set a larger alpha"
            } else {
                "drop it, or set alpha > 0 (ridge)"
            };
            let after = if self.fit_intercept {
                " (after centring)"
            } else {
                ""
            };
            return Err(Error::InvalidData(format!(
                "linear regression: column {j} is a linear combination of the columns before \
                 it{after}, so the solution is not unique; {hint}"
            )));
        }
        let mut b = vec![0.0; total * outputs];
        for (i, row) in b.chunks_exact_mut(outputs).take(rows).enumerate() {
            for (q, value) in row.iter_mut().enumerate() {
                *value = y[i * outputs + q] - y_mean[q];
                if value.is_nan() || value.abs() > limit {
                    return Err(Error::InvalidData(format!(
                        "linear regression: target column {q} holds values too close to the \
                         largest f64 to solve with; rescale the targets"
                    )));
                }
            }
        }
        let b = Tensor::from_vec(b, &[total, outputs])?;
        Ok(qr.solve(b.view()).map_err(in_context)?.data().to_vec())
    }
}

/// Prefixes an [`Error::InvalidData`] from the least-squares solver with "linear
/// regression: ", so that an overflow found inside QR says which call met it. Other errors
/// pass through unchanged.
fn in_context(error: Error) -> Error {
    match error {
        Error::InvalidData(message) => Error::InvalidData(format!("linear regression: {message}")),
        other => other,
    }
}

/// The sizes of a regression problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    /// The number of rows.
    pub(crate) rows: usize,
    /// The number of input columns.
    pub(crate) features: usize,
    /// The number of target columns.
    pub(crate) outputs: usize,
}

/// Returns [`Error::InvalidData`] unless `alpha` is finite and not negative.
pub(crate) fn check_alpha(alpha: f64) -> Result<(), Error> {
    if !(alpha.is_finite() && alpha >= 0.0) {
        return Err(Error::InvalidData(format!(
            "linear regression: alpha is {alpha}, but it must be a finite number at least 0 \
             (0 for ordinary least squares, above 0 for ridge)"
        )));
    }
    Ok(())
}

/// The value targets of a data set [`check_values`] passed.
pub(crate) fn targets<T: Real>(data: &Dataset<T>) -> &Tensor<T> {
    match &data.targets {
        Targets::Values(t) => t,
        Targets::Classes(_) => unreachable!("check_values refuses class targets"),
    }
}

/// Checks that `data` is a regression problem: `x` a matrix with at least 2 rows and 1
/// column, value targets `[rows, outputs]` with at least 1 output, every value finite.
pub(crate) fn check_values<T: Real>(data: &Dataset<T>) -> Result<Shape, Error> {
    let &[rows, features] = data.x.shape().dims() else {
        return Err(Error::InvalidData(format!(
            "linear regression: the inputs must have shape [rows, features], found {:?}",
            data.x.shape().dims()
        )));
    };
    let Targets::Values(t) = &data.targets else {
        return Err(Error::InvalidData(
            "linear regression needs value targets, but the data has class targets".to_string(),
        ));
    };
    let outputs = match *t.shape().dims() {
        [r, outputs] if r == rows => outputs,
        _ => {
            return Err(Error::InvalidData(format!(
                "linear regression: the value targets are {:?}, but they must be \
                 [{rows}, outputs] to match the inputs [{rows}, {features}]",
                t.shape().dims()
            )));
        }
    };
    if rows < 2 {
        return Err(Error::InvalidData(format!(
            "linear regression needs at least 2 rows, but the data has {rows}"
        )));
    }
    if features == 0 || outputs == 0 {
        return Err(Error::InvalidData(format!(
            "linear regression needs at least one input and one output column, but the inputs \
             are [{rows}, {features}] and the targets [{rows}, {outputs}]"
        )));
    }
    let finite =
        |values: &[T], width: usize, what: &str| match values.iter().position(|v| !v.is_finite()) {
            Some(at) => Err(Error::InvalidData(format!(
                "linear regression: row {}, column {} of {what} is {}; the values must be finite",
                at / width,
                at % width,
                values[at].to_f64()
            ))),
            None => Ok(()),
        };
    finite(data.x.data(), features, "the inputs")?;
    finite(t.data(), outputs, "the value targets")?;
    Ok(Shape {
        rows,
        features,
        outputs,
    })
}

/// A fitted linear model: `y = x·coefᵀ + intercept`, in `f64`.
///
/// Fields are public, so a model can also be written by hand.
/// [`to_sequential`](Self::to_sequential) turns it into a one-layer network that saves as a
/// PyTorch `nn.Linear`, and [`from_sequential`](Self::from_sequential) reads one back.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::LinearModel;
/// // y = 2·x0 − x1 + 0.5
/// let model = LinearModel {
///     coef: Tensor::from_vec(vec![2.0, -1.0], &[1, 2]).unwrap(),
///     intercept: vec![0.5],
/// };
/// let x = Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 2.0], &[2, 2]).unwrap();
/// assert_eq!(model.predict(x.view()).data(), &[1.5, -1.5]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LinearModel {
    /// The coefficients, `[outputs, features]`: row `q` gives output `q`'s weight on each
    /// input column (the layout of a PyTorch `nn.Linear` weight and scikit-learn's `coef_`).
    pub coef: Tensor<f64>,
    /// The intercept of each output (zeros for a model fitted without one).
    pub intercept: Vec<f64>,
}

impl LinearModel {
    /// The number of input columns and outputs, `(features, outputs)`.
    ///
    /// # Panics
    ///
    /// Panics if `coef` is not a matrix or `intercept` does not have one value per row of
    /// `coef`, naming both shapes.
    fn dims(&self, what: &str) -> (usize, usize) {
        match *self.coef.shape().dims() {
            [outputs, features] if self.intercept.len() == outputs => (features, outputs),
            _ => panic!(
                "LinearModel::{what}: coef is {:?} and intercept has {} values; coef must be \
                 [outputs, features] with one intercept per output",
                self.coef.shape().dims(),
                self.intercept.len()
            ),
        }
    }

    /// The model's prediction for each row of `x` (`[rows, features]`), as
    /// `[rows, outputs]`: each value computed in `f64` from the `f32` inputs and rounded once
    /// to `f32`.
    ///
    /// # Panics
    ///
    /// Panics, naming both shapes, if `x` is not a matrix with one column per feature, and
    /// as described on the model's fields if `coef` and `intercept` do not agree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LinearModel;
    /// // Two outputs: x and 10 − x.
    /// let model = LinearModel {
    ///     coef: Tensor::from_vec(vec![1.0, -1.0], &[2, 1]).unwrap(),
    ///     intercept: vec![0.0, 10.0],
    /// };
    /// let x = Tensor::from_vec(vec![3.0f32], &[1, 1]).unwrap();
    /// assert_eq!(model.predict(x.view()).data(), &[3.0, 7.0]);
    /// ```
    pub fn predict(&self, x: TensorView<'_, f32>) -> Tensor<f32> {
        let (features, outputs) = self.dims("predict");
        let rows = match *x.shape().dims() {
            [rows, f] if f == features => rows,
            _ => panic!(
                "LinearModel::predict: x is {:?}, but the model takes [rows, {features}]",
                x.shape().dims()
            ),
        };
        let mut out = Vec::with_capacity(rows * outputs);
        for row in x.data().chunks_exact(features) {
            out.extend(self.row_prediction(row).map(|v| v as f32));
        }
        Tensor::from_vec(out, &[rows, outputs]).expect("one prediction per row and output")
    }

    /// The prediction for one input row, in `f64`, one value per output.
    fn row_prediction<'a, T: Real>(&'a self, row: &'a [T]) -> impl Iterator<Item = f64> + 'a {
        let features = row.len();
        self.coef
            .data()
            .chunks_exact(features.max(1))
            .zip(&self.intercept)
            .map(move |(w, b)| {
                let sum: f64 = w.iter().zip(row).map(|(w, x)| w * x.to_f64()).sum();
                sum + b
            })
    }

    /// The coefficient of determination R² of the model's predictions on `data`, as
    /// scikit-learn's `r2_score` with `multioutput='uniform_average'` and
    /// [`Evaluation::r2`](cavitation_nn::Evaluation::r2): for each output,
    /// `1 − Σ(y − ŷ)² / Σ(y − ȳ)²` over every row, then the mean over the outputs. An output
    /// whose targets do not vary scores 1 when it is predicted exactly and 0 otherwise. The
    /// predictions and sums are in `f64`, so nothing is rounded to `f32` on the way.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `data.x` is not a matrix with one column per feature or
    /// has no rows, or if the targets are class labels or not `[rows, outputs]`.
    ///
    /// # Panics
    ///
    /// Panics as described on the model's fields if `coef` and `intercept` do not agree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::LinearModel;
    /// use cavitation_nn::{Dataset, Targets};
    /// let model = LinearModel {
    ///     coef: Tensor::from_vec(vec![1.0], &[1, 1]).unwrap(),
    ///     intercept: vec![0.0],
    /// };
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3, 1]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 2.0, 4.0], &[3, 1]).unwrap()),
    /// };
    /// // Squared errors 0, 0, 1 against deviations 13/9 + 4/9 + 25/9 = 42/9 from the mean 7/3.
    /// assert!((model.score(&data).unwrap() - (1.0 - 9.0 / 42.0)).abs() < 1e-15);
    /// ```
    pub fn score<T: Real>(&self, data: &Dataset<T>) -> Result<f64, Error> {
        let (features, outputs) = self.dims("score");
        let rows = match *data.x.shape().dims() {
            [rows, f] if f == features && rows > 0 => rows,
            _ => {
                return Err(Error::InvalidData(format!(
                    "LinearModel::score: the inputs are {:?}, but the model takes \
                     [rows, {features}] with at least one row",
                    data.x.shape().dims()
                )));
            }
        };
        let t = match &data.targets {
            Targets::Values(t) if t.shape().dims() == [rows, outputs] => t,
            Targets::Values(t) => {
                return Err(Error::InvalidData(format!(
                    "LinearModel::score: the value targets are {:?}, but the model outputs \
                     [{rows}, {outputs}]",
                    t.shape().dims()
                )));
            }
            Targets::Classes(_) => {
                return Err(Error::InvalidData(
                    "LinearModel::score needs value targets, but the data has class targets"
                        .to_string(),
                ));
            }
        };
        let y = t.data();
        let mut mean = vec![0.0; outputs];
        for row in y.chunks_exact(outputs) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += v.to_f64();
            }
        }
        for m in &mut mean {
            *m /= rows as f64;
        }
        let (mut ss_res, mut ss_tot) = (vec![0.0; outputs], vec![0.0; outputs]);
        for (x, row) in data
            .x
            .data()
            .chunks_exact(features)
            .zip(y.chunks_exact(outputs))
        {
            for (q, p) in self.row_prediction(x).enumerate() {
                let v = row[q].to_f64();
                ss_res[q] += (v - p) * (v - p);
                ss_tot[q] += (v - mean[q]) * (v - mean[q]);
            }
        }
        Ok(r2(&ss_res, &ss_tot))
    }

    /// The model as a network of one [`Linear`] layer (weight `coef`, bias `intercept`,
    /// each rounded once to `f32`), whose parameters are added to `params`. Saved with
    /// [`save`](cavitation_nn::save), it is a PyTorch `nn.Linear(features, outputs)`'s
    /// `state_dict` (keys `0.weight` and `0.bias`, inside an `nn.Sequential`).
    ///
    /// A coefficient or intercept beyond the `f32` range (about ±3.4e38, possible for a model
    /// fitted on `f64` data) becomes ±infinity in the layer, and a saved file carries the
    /// infinity; debug builds panic instead, naming the value. Rescale the data so the
    /// coefficients fit before converting.
    ///
    /// # Panics
    ///
    /// Panics if the model has no features or no outputs (a `Linear` needs both), and as
    /// described on the model's fields if `coef` and `intercept` do not agree. In debug
    /// builds, also panics if a finite coefficient or intercept rounds to an infinite `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_ml::LinearModel;
    /// let model = LinearModel {
    ///     coef: Tensor::from_vec(vec![2.0, -1.0], &[1, 2]).unwrap(),
    ///     intercept: vec![0.5],
    /// };
    /// let mut params = ParamStore::new();
    /// let net = model.to_sequential(&mut params);
    /// let linear = net.linear(0).unwrap();
    /// assert_eq!(params.value(linear.weight()).data(), &[2.0, -1.0]);
    /// assert_eq!(LinearModel::from_sequential(&net, &params).unwrap(), model);
    /// ```
    pub fn to_sequential(&self, params: &mut ParamStore) -> Sequential {
        let (features, outputs) = self.dims("to_sequential");
        assert!(
            features > 0 && outputs > 0,
            "LinearModel::to_sequential: a Linear layer needs at least one input and one output, \
             but coef is [{outputs}, {features}]"
        );
        let weight: Vec<f32> = self.coef.data().iter().map(|&w| w as f32).collect();
        let bias: Vec<f32> = self.intercept.iter().map(|&b| b as f32).collect();
        debug_assert!(
            {
                let wide = self.coef.data().iter().zip(&weight);
                let narrow = self.intercept.iter().zip(&bias);
                wide.chain(narrow)
                    .all(|(&v, r)| !v.is_finite() || r.is_finite())
            },
            "LinearModel::to_sequential: a coefficient or intercept is beyond the f32 range: {:?}",
            self.coef
                .data()
                .iter()
                .chain(&self.intercept)
                .find(|v| v.is_finite() && !(**v as f32).is_finite())
        );
        let weight = Tensor::from_vec(weight, &[outputs, features]).expect("coef's shape");
        let bias = Tensor::from_vec(bias, &[outputs]).expect("one bias per output");
        let linear = Linear::from_values(params, weight, bias).expect("checked shapes");
        Sequential::new().push(linear)
    }

    /// Reads a model back from a network of exactly one [`Linear`] layer, such as
    /// [`to_sequential`](Self::to_sequential) builds or a loaded PyTorch `nn.Linear`: `coef`
    /// is its weight and `intercept` its bias, widened exactly to `f64`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `model` is not exactly one layer, if that layer is not a
    /// `Linear`, or if it has a mask with any neuron off (a masked output is 0, which no
    /// linear model's coefficients describe).
    ///
    /// # Panics
    ///
    /// Panics if `model` was built from a different [`ParamStore`] than `params`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_ml::LinearModel;
    /// use cavitation_nn::{Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// let model = LinearModel::from_sequential(&net, &params).unwrap();
    /// assert_eq!(model.coef.shape().dims(), &[2, 3]);
    ///
    /// let deeper = Sequential::new()
    ///     .push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// assert!(LinearModel::from_sequential(&deeper, &params).is_err());
    /// ```
    pub fn from_sequential(model: &Sequential, params: &ParamStore) -> Result<LinearModel, Error> {
        let linear = match (model.len(), model.linear(0)) {
            (1, Some(linear)) => linear,
            (1, None) => {
                return Err(Error::InvalidData(format!(
                    "LinearModel::from_sequential: the layer is a {}, not a linear layer",
                    model.layer(0).name()
                )));
            }
            (layers, _) => {
                return Err(Error::InvalidData(format!(
                    "LinearModel::from_sequential: the network has {layers} layers, but a linear \
                     model is exactly one linear layer"
                )));
            }
        };
        if let Some(mask) = model.mask(0).filter(|m| m.count_on() < m.width()) {
            return Err(Error::InvalidData(format!(
                "LinearModel::from_sequential: the layer has {} of {} outputs switched off by a \
                 mask, which a linear model cannot hold; clear the mask first",
                mask.width() - mask.count_on(),
                mask.width()
            )));
        }
        let widen = |v: &Tensor<f32>| v.data().iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let weight = params.value(linear.weight());
        Ok(LinearModel {
            coef: Tensor::from_vec(widen(weight), weight.shape().dims())?,
            intercept: widen(params.value(linear.bias())),
        })
    }
}

/// R² averaged over the outputs from each output's sum of squared errors `ss_res` and sum
/// of squared deviations from its mean `ss_tot`: `1 − ss_res / ss_tot`, or, for an output
/// whose targets do not vary, 1 when it is predicted exactly and 0 otherwise. The same
/// definition as [`Evaluation::r2`](cavitation_nn::Evaluation::r2) and scikit-learn's
/// `r2_score`.
fn r2(ss_res: &[f64], ss_tot: &[f64]) -> f64 {
    let sum: f64 = ss_res
        .iter()
        .zip(ss_tot)
        .map(|(&res, &tot)| {
            if tot != 0.0 {
                1.0 - res / tot
            } else if res == 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .sum();
    sum / ss_res.len() as f64
}

#[cfg(test)]
mod tests {
    //! Tests for the exact path's degenerate inputs, its errors, R², and the round trip
    //! through a `Sequential` and a saved file.

    use cavitation_core::{ParamStore, Rng, Tensor};
    use cavitation_nn::{Dataset, Evaluator, Loss, NeuronMask, Targets, load, save};

    use super::*;

    /// A data set of `rows` rows from row-major inputs and one-output targets.
    fn data(x: Vec<f32>, features: usize, y: Vec<f32>) -> Dataset {
        let rows = y.len();
        Dataset {
            x: Tensor::from_vec(x, &[rows, features]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[rows, 1]).unwrap()),
        }
    }

    /// Seeded inputs `[rows, features]` and targets `x·w + 1` plus a little noise, for
    /// `outputs` outputs with weights `1, 2, ...` scaled per output.
    fn noisy(rows: usize, features: usize, outputs: usize, seed: u64) -> Dataset {
        let mut rng = Rng::seed(seed);
        let x: Vec<f32> = (0..rows * features).map(|_| rng.normal() as f32).collect();
        let mut y = Vec::with_capacity(rows * outputs);
        for row in x.chunks(features) {
            for q in 0..outputs {
                let s: f32 = row
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v * (j + 1) as f32 * (q + 1) as f32)
                    .sum();
                y.push(s + 1.0 + 0.1 * rng.normal() as f32);
            }
        }
        Dataset {
            x: Tensor::from_vec(x, &[rows, features]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[rows, outputs]).unwrap()),
        }
    }

    /// A column equal to an earlier one is refused, naming it and suggesting `alpha > 0`;
    /// ridge solves the same data.
    #[test]
    fn duplicate_column_named() {
        let mut d = noisy(20, 3, 1, 1);
        let x = d.x.data_mut();
        for row in x.chunks_mut(3) {
            row[2] = row[0];
        }
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(err.contains("column 2 is a linear combination"), "{err}");
        assert!(err.contains("alpha > 0"), "{err}");
        let ridge = LinearRegression {
            alpha: 1.0,
            ..LinearRegression::default()
        };
        let model = ridge.fit(&d).unwrap();
        // The penalty splits the weight evenly between the two equal columns.
        let w = model.coef.data();
        assert!((w[0] - w[2]).abs() < 1e-12, "{w:?}");
    }

    /// A column that is a multiple of an earlier one, in units about 1e12 apart (`2^40`,
    /// so the `f32` values are exact multiples), is still found: the test is relative to
    /// each column's own norm.
    #[test]
    fn dependent_column_found_across_units() {
        let mut d = noisy(20, 3, 1, 2);
        for row in d.x.data_mut().chunks_mut(3) {
            row[1] *= 2f32.powi(20);
            row[2] = row[0] * 2f32.powi(-20);
        }
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(err.contains("column 2 is a linear combination"), "{err}");
    }

    /// Independent columns in units about 1e12 apart (`2^20` and `2^-20`) fit to the
    /// coefficients of the same data in common units, rescaled, within 1e-12 relative.
    #[test]
    fn units_far_apart_fit() {
        let d = noisy(30, 2, 1, 3);
        let mut scaled = d.clone();
        let (up, down) = (2f32.powi(20), 2f32.powi(-20));
        for row in scaled.x.data_mut().chunks_mut(2) {
            row[0] *= up;
            row[1] *= down;
        }
        let a = LinearRegression::default().fit(&d).unwrap();
        let b = LinearRegression::default().fit(&scaled).unwrap();
        let (wa, wb) = (a.coef.data(), b.coef.data());
        let back = [wb[0] * f64::from(up), wb[1] * f64::from(down)];
        for (x, y) in back.iter().zip(wa) {
            assert!((x - y).abs() <= 1e-12 * y.abs(), "{back:?} against {wa:?}");
        }
        assert!((a.intercept[0] - b.intercept[0]).abs() <= 1e-12 * a.intercept[0].abs());
    }

    /// More varying columns than the rows can determine is refused with a message that
    /// suggests `alpha > 0`; ridge fits it.
    #[test]
    fn more_features_than_rows_suggests_alpha() {
        let d = noisy(4, 6, 1, 4);
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(err.contains("6 input columns vary"), "{err}");
        assert!(err.contains("alpha > 0"), "{err}");
        // With an intercept, 4 rows determine only 3 coefficients.
        let d = noisy(4, 4, 1, 5);
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(err.contains("less one for the intercept"), "{err}");
        let no_intercept = LinearRegression {
            fit_intercept: false,
            alpha: 0.0,
        };
        assert!(no_intercept.fit(&d).is_ok());
        let ridge = LinearRegression {
            alpha: 0.5,
            ..LinearRegression::default()
        };
        assert!(ridge.fit(&noisy(4, 6, 1, 4)).is_ok());
    }

    /// One row is refused, as is no row.
    #[test]
    fn one_row_is_an_error() {
        let err = LinearRegression::default()
            .fit(&data(vec![1.0, 2.0], 2, vec![3.0]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("at least 2 rows") && err.contains("has 1"),
            "{err}"
        );
        assert!(
            LinearRegression::default()
                .fit(&data(vec![], 2, vec![]))
                .is_err()
        );
    }

    /// A constant column gets the coefficient 0 exactly, with an intercept; without one, an
    /// all-zero column does, and a constant non-zero column takes part in the solve.
    #[test]
    fn constant_column_gets_zero() {
        let mut d = noisy(10, 3, 2, 6);
        for row in d.x.data_mut().chunks_mut(3) {
            row[1] = 0.1;
        }
        let model = LinearRegression::default().fit(&d).unwrap();
        assert_eq!((model.coef.data()[1], model.coef.data()[4]), (0.0, 0.0));
        let no_intercept = LinearRegression {
            fit_intercept: false,
            alpha: 0.0,
        };
        let model = no_intercept.fit(&d).unwrap();
        assert_ne!(model.coef.data()[1], 0.0);
        assert_eq!(model.intercept, [0.0, 0.0]);
        for row in d.x.data_mut().chunks_mut(3) {
            row[1] = 0.0;
        }
        assert_eq!(no_intercept.fit(&d).unwrap().coef.data()[1], 0.0);
        // Every column constant: the intercept is each target's mean.
        let flat = data(vec![2.0; 3], 1, vec![1.0, 2.0, 6.0]);
        let model = LinearRegression::default().fit(&flat).unwrap();
        assert_eq!((model.coef.data()[0], model.intercept[0]), (0.0, 3.0));
    }

    /// Bad `alpha`, class targets, mismatched rows, no columns and non-finite values are
    /// `InvalidData` naming the problem.
    #[test]
    fn bad_inputs_are_errors() {
        let d = noisy(10, 2, 1, 7);
        for alpha in [-1.0, f64::NAN, f64::INFINITY] {
            let r = LinearRegression {
                alpha,
                ..LinearRegression::default()
            };
            assert!(r.fit(&d).unwrap_err().to_string().contains("alpha"));
        }
        let classes = Dataset {
            x: d.x.clone(),
            targets: Targets::Classes(vec![0; 10]),
        };
        let err = LinearRegression::default().fit(&classes).unwrap_err();
        assert!(err.to_string().contains("class targets"), "{err}");
        let short = Dataset {
            x: d.x.clone(),
            targets: Targets::Values(Tensor::zeros(&[9, 1])),
        };
        assert!(LinearRegression::default().fit(&short).is_err());
        let mut bad = d.clone();
        bad.x.data_mut()[5] = f32::NAN;
        let err = LinearRegression::default()
            .fit(&bad)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("row 2, column 1 of the inputs is NaN"),
            "{err}"
        );
        let empty = data(vec![], 0, vec![1.0, 2.0]);
        assert!(LinearRegression::default().fit(&empty).is_err());
    }

    /// Values whose centred column would overflow f64 are refused, naming the column.
    #[test]
    fn huge_values_refused() {
        let x = vec![1.0, 1e308, 2.0, -1e308, 3.0, 1e308];
        let d = Dataset {
            x: Tensor::from_vec(x, &[3, 2]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap()),
        };
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(err.contains("column 1 holds values too close"), "{err}");
    }

    /// Targets whose centred values would overflow f64 are refused, naming the target
    /// column, and a solution that overflows inside QR is reported with the regression's
    /// context.
    #[test]
    fn overflow_errors_have_context() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.5e308, -1.6e308, 1.7e308];
        let d = Dataset {
            x: Tensor::from_vec(x, &[3, 1]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[3, 1]).unwrap()),
        };
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(
            err.contains("target column 0 holds values too close"),
            "{err}"
        );
        // Inputs near 1e-300 and targets near 1e10: the slope is near 1e310.
        let x = vec![1e-300, 2e-300, 4e-300];
        let y = vec![1e10, 2e10, 5e10];
        let d = Dataset {
            x: Tensor::from_vec(x, &[3, 1]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[3, 1]).unwrap()),
        };
        let err = LinearRegression::default().fit(&d).unwrap_err().to_string();
        assert!(
            err.contains("linear regression: the least-squares solution"),
            "{err}"
        );
        assert!(err.contains("overflows"), "{err}");
    }

    /// In debug builds `to_sequential` refuses a coefficient beyond the `f32` range.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "beyond the f32 range")]
    fn to_sequential_refuses_out_of_range() {
        let model = LinearModel {
            coef: Tensor::from_vec(vec![1e39], &[1, 1]).unwrap(),
            intercept: vec![0.0],
        };
        model.to_sequential(&mut ParamStore::new());
    }

    /// Several outputs are fitted at once, each as its own regression.
    #[test]
    fn outputs_are_independent() {
        let d = noisy(40, 3, 2, 8);
        let both = LinearRegression::default().fit(&d).unwrap();
        let Targets::Values(t) = &d.targets else {
            unreachable!()
        };
        for q in 0..2 {
            let y: Vec<f32> = t.data().iter().skip(q).step_by(2).copied().collect();
            let one = LinearRegression::default()
                .fit(&data(d.x.data().to_vec(), 3, y))
                .unwrap();
            for j in 0..3 {
                assert!((one.coef.data()[j] - both.coef.data()[q * 3 + j]).abs() < 1e-12);
            }
            assert!((one.intercept[0] - both.intercept[q]).abs() < 1e-12);
        }
    }

    /// The fitted model's R² equals `Evaluation::r2` of its `Sequential` within the `f32`
    /// rounding of the network's outputs.
    #[test]
    fn score_matches_evaluation() {
        let d = noisy(50, 3, 2, 9);
        let model = LinearRegression::default().fit(&d).unwrap();
        let mut params = ParamStore::new();
        let net = model.to_sequential(&mut params);
        let e = Evaluator::new(&net, &params, Loss::Mse, 16)
            .unwrap()
            .evaluate(&d)
            .unwrap();
        assert!((e.r2.unwrap() - model.score(&d).unwrap()).abs() < 1e-6);
    }

    /// Predictions through the `Sequential` equal `predict`'s within the GEMM bound, before
    /// and after a save and load; `from_sequential` reads back the `f32`-rounded model.
    #[test]
    fn to_sequential_round_trip() {
        let d = noisy(30, 4, 3, 10);
        let model = LinearRegression::default().fit(&d).unwrap();
        let mut params = ParamStore::new();
        let net = model.to_sequential(&mut params);
        let path = std::env::temp_dir().join(format!(
            "cavitation-ml-linear-round-trip-{}.safetensors",
            std::process::id()
        ));
        save(&path, &net, &params).unwrap();
        let mut fresh_params = ParamStore::new();
        let fresh = Sequential::new().push(Linear::new(&mut fresh_params, 4, 3, &mut Rng::seed(1)));
        load(&path, &fresh, &mut fresh_params).unwrap();
        std::fs::remove_file(&path).ok();
        let expected = model.predict(d.x.view());
        let x = d.x.data();
        for (n, p) in [(&net, &params), (&fresh, &fresh_params)] {
            let mut compiled = cavitation_nn::InferenceModel::compile(n, p).unwrap();
            let got = compiled.infer_batch(d.x.view());
            for (i, (&g, &e)) in got.data().iter().zip(expected.data()).enumerate() {
                // GEMM bound: k · ε · Σ|w·x| + |b|-sized rounding, k = features.
                let (r, q) = (i / 3, i % 3);
                let row = &x[r * 4..(r + 1) * 4];
                let w = &model.coef.data()[q * 4..(q + 1) * 4];
                let magnitude: f64 = row
                    .iter()
                    .zip(w)
                    .map(|(x, w)| (f64::from(*x) * w).abs())
                    .sum::<f64>()
                    + model.intercept[q].abs();
                let bound = 4.0 * f64::from(f32::EPSILON) * magnitude + 1e-6;
                assert!(
                    (f64::from(g) - f64::from(e)).abs() <= bound,
                    "row {r}, output {q}: {g} against {e}"
                );
            }
        }
        let back = LinearModel::from_sequential(&fresh, &fresh_params).unwrap();
        for (&a, &b) in back.coef.data().iter().zip(model.coef.data()) {
            assert_eq!(a, f64::from(b as f32));
        }
    }

    /// `from_sequential` refuses a model that is not one `Linear`, and one with a neuron
    /// masked off.
    #[test]
    fn from_sequential_refuses_other_models() {
        let mut params = ParamStore::new();
        let relu = Sequential::new().push(cavitation_nn::Relu);
        let err = LinearModel::from_sequential(&relu, &params).unwrap_err();
        assert!(err.to_string().contains("not a linear layer"), "{err}");
        let mut net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
        let mut mask = NeuronMask::all_on(2);
        mask.set(1, false);
        net.set_mask(0, mask);
        let err = LinearModel::from_sequential(&net, &params).unwrap_err();
        assert!(err.to_string().contains("switched off"), "{err}");
    }

    /// `score` refuses class targets and inputs of the wrong width.
    #[test]
    fn score_errors() {
        let d = noisy(10, 2, 1, 11);
        let model = LinearRegression::default().fit(&d).unwrap();
        let classes = Dataset {
            x: d.x.clone(),
            targets: Targets::Classes(vec![0; 10]),
        };
        assert!(model.score(&classes).is_err());
        let wide = noisy(10, 3, 1, 11);
        assert!(model.score(&wide).is_err());
    }
}
