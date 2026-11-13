//! The scikit-learn fixtures: the loader that reads one case from
//! `tests/fixtures/sklearn/` (written by `scripts/fixtures/make_sklearn_fixtures.py`, whose
//! docstring describes every key), and checks that every case is complete and consistent.

use std::path::PathBuf;

use cavitation_nn::{Dataset, Targets};
use serde_json::Value;

/// The scikit-learn version every fixture must have been written with.
const SKLEARN_VERSION: &str = "1.9.1";

/// A row-major `f64` matrix (or vector) read from a fixture.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    /// The dimensions, outermost first.
    pub shape: Vec<usize>,
    /// The values, row-major.
    pub data: Vec<f64>,
}

/// One scikit-learn fixture case.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// The file name without `.json`, for example `ols-small`.
    pub name: String,
    /// The scikit-learn version that wrote the file.
    pub sklearn_version: String,
    /// `LinearRegression`, `Ridge`, `LogisticRegression`, `StandardScaler` or `r2_score`.
    pub estimator: String,
    /// The estimator's parameters as passed; for a logistic regression without a penalty,
    /// `"C"` is `null`.
    pub params: Value,
    /// The inputs `[rows, features]`, with a data set reference already resolved to the
    /// rows (standardised when the fixture says so); `None` for `r2_score`.
    pub x: Option<Matrix>,
    /// The targets: `[rows, outputs]` for regression and `r2_score`'s true values, `[rows]`
    /// of class labels for classification; `None` for `StandardScaler`.
    pub y: Option<Matrix>,
    /// scikit-learn's `coef_` as `[outputs or classes, features]`, for the fitted models.
    pub coef: Option<Matrix>,
    /// One intercept per output or class (zeros without an intercept); empty when the case
    /// has none.
    pub intercept: Vec<f64>,
    /// The objective at scikit-learn's solution, in the form `objective_form` names.
    pub objective: Option<f64>,
    /// `sum_sq`, `logistic` or `multinomial` (see the fixture script).
    pub objective_form: Option<String>,
    /// scikit-learn's `n_iter_`, for the logistic fits.
    pub n_iter: Option<usize>,
    /// The whole file, for the keys particular to one estimator (`mean`, `scale`, `var`,
    /// `y_pred`, `score`, `raw_values`).
    pub json: Value,
}

/// The folder holding the fixture files.
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sklearn")
}

/// The value at `key`, or a panic naming the file.
fn field<'a>(v: &'a Value, key: &str, file: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("{file}: missing `{key}`"))
}

/// Reads an array of numbers.
fn numbers(v: &Value, file: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{file}: expected an array of numbers"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("{file}: bad number")))
        .collect()
}

/// Reads a `{shape, data}` matrix and checks the shape matches the data.
fn matrix(v: &Value, file: &str) -> Matrix {
    let shape: Vec<usize> = field(v, "shape", file)
        .as_array()
        .unwrap_or_else(|| panic!("{file}: `shape` is not an array"))
        .iter()
        .map(|d| {
            d.as_u64()
                .unwrap_or_else(|| panic!("{file}: bad dimension")) as usize
        })
        .collect();
    let data = numbers(field(v, "data", file), file);
    assert_eq!(
        shape.iter().product::<usize>(),
        data.len(),
        "{file}: shape {shape:?} does not match {} values",
        data.len()
    );
    Matrix { shape, data }
}

/// The training split of the named data set, from `cavitation-datasets`.
fn train_split(name: &str, file: &str) -> Dataset {
    let (train, _) = match name {
        "concrete" => cavitation_datasets::concrete::load(),
        "breast-cancer" => cavitation_datasets::breast_cancer::load(),
        other => panic!("{file}: unknown data set {other}"),
    }
    .unwrap_or_else(|e| panic!("{file}: cannot load {name}: {e}"));
    train
}

/// Resolves `x`: a `{shape, data}` matrix, or a data set reference, which is that set's
/// training rows widened to `f64`, checked against the reference's `shape`, `first_row` and
/// `sum`, then standardised as `(x − mean) / scale` per column when it has `standardize`.
fn resolve_x(v: &Value, file: &str) -> Matrix {
    let Some(name) = v.get("dataset") else {
        return matrix(v, file);
    };
    let name = name
        .as_str()
        .unwrap_or_else(|| panic!("{file}: bad `dataset`"));
    assert_eq!(
        field(v, "split", file),
        "train",
        "{file}: only train splits are used"
    );
    let train = train_split(name, file);
    let (rows, features) = (train.len(), train.features());
    let shape: Vec<usize> = numbers(field(v, "shape", file), file)
        .iter()
        .map(|&d| d as usize)
        .collect();
    assert_eq!(shape, [rows, features], "{file}: shape");
    let mut data: Vec<f64> = train.x.data().iter().map(|&v| f64::from(v)).collect();
    assert_eq!(
        numbers(field(v, "first_row", file), file),
        data[..features],
        "{file}: the first training row differs from the loader's"
    );
    let sum: f64 = data.iter().sum();
    let expected = field(v, "sum", file).as_f64().unwrap();
    assert!(
        (sum - expected).abs() <= 1e-11 * expected.abs(),
        "{file}: training rows sum to {sum}, expected {expected}"
    );
    if let Some(standardize) = v.get("standardize") {
        let mean = numbers(field(standardize, "mean", file), file);
        let scale = numbers(field(standardize, "scale", file), file);
        assert_eq!(
            (mean.len(), scale.len()),
            (features, features),
            "{file}: standardize"
        );
        for row in data.chunks_mut(features) {
            for ((value, m), s) in row.iter_mut().zip(&mean).zip(&scale) {
                *value = (*value - m) / s;
            }
        }
    }
    Matrix { shape, data }
}

/// Resolves `y`: a `{shape, data}` matrix, or a data set reference, which is that set's
/// training targets: `[rows, outputs]` values or `[rows]` class labels.
fn resolve_y(v: &Value, file: &str) -> Matrix {
    let Some(name) = v.get("dataset") else {
        return matrix(v, file);
    };
    let name = name
        .as_str()
        .unwrap_or_else(|| panic!("{file}: bad `dataset`"));
    assert_eq!(
        field(v, "split", file),
        "train",
        "{file}: only train splits are used"
    );
    match train_split(name, file).targets {
        Targets::Values(t) => Matrix {
            shape: t.shape().dims().to_vec(),
            data: t.data().iter().map(|&v| f64::from(v)).collect(),
        },
        Targets::Classes(labels) => Matrix {
            shape: vec![labels.len()],
            data: labels.iter().map(|&l| l as f64).collect(),
        },
    }
}

/// Reads the fixture `tests/fixtures/sklearn/<name>.json`, resolving data set references.
///
/// # Panics
///
/// Panics, naming the file and the problem, if the file is missing, is not valid JSON, or
/// lacks `sklearn_version`, `estimator` or `params`, or if a data set reference does not
/// match the loader's rows.
pub fn fixture(name: &str) -> Fixture {
    let path = fixture_dir().join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let v: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: invalid JSON: {e}"));
    let text_of = |key: &str| {
        field(&v, key, name)
            .as_str()
            .unwrap_or_else(|| panic!("{name}: `{key}` is not a string"))
            .to_string()
    };
    let present = |key: &str| v.get(key).filter(|x| !x.is_null());
    Fixture {
        name: name.to_string(),
        sklearn_version: text_of("sklearn_version"),
        estimator: text_of("estimator"),
        params: field(&v, "params", name).clone(),
        x: present("x").map(|x| resolve_x(x, name)),
        y: present("y").map(|y| resolve_y(y, name)),
        coef: present("coef").map(|c| matrix(c, name)),
        intercept: present("intercept")
            .map(|i| numbers(i, name))
            .unwrap_or_default(),
        objective: present("objective").map(|o| o.as_f64().unwrap()),
        objective_form: present("objective_form").map(|f| f.as_str().unwrap().to_string()),
        n_iter: present("n_iter").map(|n| n.as_u64().unwrap() as usize),
        json: v,
    }
}

/// The names of every fixture file, in file-name order.
fn fixture_names() -> Vec<String> {
    let dir = fixture_dir();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the fixture directory {}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Every fixture records the scikit-learn version that wrote it, and all use the pinned one.
#[test]
fn fixture_versions_recorded() {
    let names = fixture_names();
    assert_eq!(names.len(), 20, "fixtures found: {names:?}");
    for name in &names {
        let f = fixture(name);
        assert_eq!(
            f.sklearn_version, SKLEARN_VERSION,
            "{name}: sklearn_version"
        );
    }
}

/// Every case has the keys its estimator needs, with shapes that agree: as many target rows
/// as input rows, `coef` as `[outputs or classes, features]`, one intercept per row of
/// `coef`, a finite objective, and `n_iter` exactly for the logistic fits.
#[test]
fn fixtures_consistent() {
    for name in fixture_names() {
        let f = fixture(&name);
        match f.estimator.as_str() {
            "LinearRegression" | "Ridge" | "LogisticRegression" => {
                let (x, y, coef) = (f.x.unwrap(), f.y.unwrap(), f.coef.unwrap());
                assert_eq!(x.shape.len(), 2, "{name}: x");
                assert_eq!(y.shape[0], x.shape[0], "{name}: rows");
                assert_eq!(coef.shape[1], x.shape[1], "{name}: coef columns");
                assert_eq!(f.intercept.len(), coef.shape[0], "{name}: intercept");
                assert!(f.objective.unwrap().is_finite(), "{name}: objective");
                let logistic = f.estimator == "LogisticRegression";
                let form = f.objective_form.unwrap();
                if logistic {
                    assert_eq!(y.shape.len(), 1, "{name}: class labels");
                    let classes = y.data.iter().fold(0.0_f64, |m, &l| m.max(l)) as usize + 1;
                    assert_eq!(coef.shape[0], if classes == 2 { 1 } else { classes });
                    assert!(
                        form == "logistic" || form == "multinomial",
                        "{name}: {form}"
                    );
                    assert!(f.n_iter.unwrap() >= 1, "{name}: n_iter");
                    assert!(f.params.get("C").is_some(), "{name}: C");
                } else {
                    assert_eq!(y.shape, [x.shape[0], coef.shape[0]], "{name}: y");
                    assert_eq!(form, "sum_sq", "{name}");
                    assert_eq!(f.n_iter, None, "{name}: n_iter");
                }
            }
            "StandardScaler" => {
                let features = f.x.unwrap().shape[1];
                for key in ["mean", "scale", "var"] {
                    assert_eq!(
                        numbers(&f.json[key], &name).len(),
                        features,
                        "{name}: {key}"
                    );
                }
                assert!(f.y.is_none() && f.coef.is_none() && f.intercept.is_empty());
                assert!(f.objective.is_none() && f.objective_form.is_none() && f.n_iter.is_none());
            }
            "r2_score" => {
                let y = f.y.unwrap();
                assert_eq!(matrix(&f.json["y_pred"], &name).shape, y.shape, "{name}");
                assert_eq!(numbers(&f.json["raw_values"], &name).len(), y.shape[1]);
                assert!(
                    f.json["score"].as_f64().unwrap().is_finite(),
                    "{name}: score"
                );
                assert!(f.x.is_none(), "{name}: x");
            }
            other => panic!("{name}: unknown estimator {other}"),
        }
    }
}

/// A data set reference resolves to the rows scikit-learn fitted: least squares on the
/// resolved `concrete-ols` rows, with a column of ones for the intercept, gives scikit-learn's
/// coefficients and intercept within 1e-9 · max(1, |value|).
#[test]
fn dataset_reference_reproduces_the_fit() {
    use cavitation_core::Tensor;
    let f = fixture("concrete-ols");
    let (x, y, coef) = (f.x.unwrap(), f.y.unwrap(), f.coef.unwrap());
    let (rows, features) = (x.shape[0], x.shape[1]);
    let a: Vec<f64> = x
        .data
        .chunks(features)
        .flat_map(|row| row.iter().copied().chain([1.0]))
        .collect();
    let a = Tensor::from_vec(a, &[rows, features + 1]).unwrap();
    let b = Tensor::from_vec(y.data, &[rows, 1]).unwrap();
    let solution = cavitation_ml::linalg::lstsq(a.view(), b.view()).unwrap();
    let expected = coef.data.iter().chain(&f.intercept);
    for (i, (got, want)) in solution.data().iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "coefficient {i}: {got} against scikit-learn's {want}"
        );
    }
}

/// `Standardizer::fit` on the Concrete training rows gives scikit-learn's `StandardScaler`
/// mean and scale within 1e-12 relative. The loader's rows are `f32` and scikit-learn
/// fitted the same values widened to `f64`, as `fixture` does.
#[test]
fn standard_scaler_matches() {
    use cavitation_core::Tensor;
    use cavitation_nn::Standardizer;
    let f = fixture("standard-scaler-concrete");
    let x = f.x.as_ref().unwrap();
    let rows: Vec<f32> = x.data.iter().map(|&v| v as f32).collect();
    let rows = Tensor::from_vec(rows, &x.shape).unwrap();
    let s = Standardizer::fit(rows.view());
    for (key, ours) in [("mean", s.mean()), ("scale", s.scale())] {
        let theirs = numbers(&f.json[key], "standard-scaler-concrete");
        assert_eq!(ours.len(), theirs.len(), "{key}");
        for (j, (&a, &b)) in ours.iter().zip(&theirs).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs(),
                "{key}[{j}]: {a} against scikit-learn's {b}"
            );
        }
    }
}

/// The fixture's rows as a data set of `T`: `x` as `[rows, features]` and `y` as
/// `[rows, outputs]` value targets. For `f32` the values must be exactly representable,
/// which holds for the synthetic cases (rounded through `f32` before scikit-learn fitted
/// them); the standardised data set cases are not, and are read as `f64`.
fn value_dataset<T: cavitation_core::Real>(x: &Matrix, y: &Matrix) -> Dataset<T> {
    use cavitation_core::Tensor;
    let convert = |m: &Matrix| -> Vec<T> {
        m.data
            .iter()
            .map(|&v| {
                let t = T::from_f64(v);
                assert_eq!(t.to_f64(), v, "{v} is not exactly representable");
                t
            })
            .collect()
    };
    Dataset {
        x: Tensor::from_vec(convert(x), &x.shape).unwrap(),
        targets: Targets::Values(Tensor::from_vec(convert(y), &y.shape).unwrap()),
    }
}

/// The `LinearRegression` the fixture's parameters describe: `fit_intercept` (default
/// true) and, for `Ridge`, `alpha`.
fn regression_for(f: &Fixture) -> cavitation_ml::LinearRegression {
    let fit_intercept = f
        .params
        .get("fit_intercept")
        .is_none_or(|v| v.as_bool().unwrap());
    let alpha = f.params.get("alpha").map_or(0.0, |v| v.as_f64().unwrap());
    cavitation_ml::LinearRegression {
        fit_intercept,
        alpha,
    }
}

/// Fits the fixture `name` (as `f32` data when `f32` is true, otherwise `f64`) and checks
/// every coefficient and intercept against scikit-learn's within `1e-9 · max(1, |sklearn|)`.
/// Returns the fitted model and the largest difference relative to `max(1, |sklearn|)`,
/// which the test prints.
fn check_regression(name: &str, f32_data: bool) -> (cavitation_ml::LinearModel, f64) {
    let f = fixture(name);
    let (x, y) = (f.x.as_ref().unwrap(), f.y.as_ref().unwrap());
    let regression = regression_for(&f);
    let model = if f32_data {
        regression.fit(&value_dataset::<f32>(x, y)).unwrap()
    } else {
        regression.fit(&value_dataset::<f64>(x, y)).unwrap()
    };
    let coef = f.coef.as_ref().unwrap();
    assert_eq!(model.coef.shape().dims(), coef.shape.as_slice(), "{name}");
    let ours = model.coef.data().iter().chain(&model.intercept);
    let theirs = coef.data.iter().chain(&f.intercept);
    let mut worst = 0.0f64;
    for (i, (&a, &b)) in ours.zip(theirs).enumerate() {
        let gap = (a - b).abs() / b.abs().max(1.0);
        worst = worst.max(gap);
        assert!(
            gap <= 1e-9,
            "{name}: value {i} is {a}, scikit-learn's {b} (gap {gap:e})"
        );
    }
    eprintln!("{name}: largest gap {worst:e} relative to max(1, |scikit-learn|)");
    (model, worst)
}

/// Exact least squares matches scikit-learn's `LinearRegression` on every `ols-*` case,
/// with and without an intercept, one output and three.
#[test]
fn ols_fixtures_match() {
    for name in ["ols-small", "ols-multi", "ols-no-intercept"] {
        check_regression(name, true);
    }
}

/// A constant column gets the coefficient 0 exactly (scikit-learn's is within 1e-12 of 0),
/// and the other coefficients match.
#[test]
fn ols_constant_column_matches() {
    let (model, _) = check_regression("ols-constant-column", true);
    assert_eq!(model.coef.data()[2], 0.0);
}

/// Ridge (rows of √α·I appended to the centred matrix) matches scikit-learn's `Ridge` with
/// the `cholesky` solver for α = 0.1 and α = 10.
#[test]
fn ridge_fixtures_match() {
    for name in ["ridge-0.1", "ridge-10"] {
        check_regression(name, true);
    }
}

/// OLS and ridge (α = 1) on the standardised Concrete training rows match scikit-learn. The
/// rows are standardised in `f64`, as scikit-learn fitted them, so they are read as `f64`.
#[test]
fn concrete_fixtures_match() {
    for name in ["concrete-ols", "concrete-ridge-1"] {
        check_regression(name, false);
    }
}

/// `LinearModel::score` gives scikit-learn's `r2_score` (uniform average over outputs)
/// within 1e-12 relative, for one output and three. The model is the identity (coefficients
/// `I`, intercept 0), so its predictions are the fixture's `y_pred` exactly.
#[test]
fn r2_score_matches() {
    use cavitation_core::Tensor;
    for name in ["r2-score", "r2-score-multi"] {
        let f = fixture(name);
        let y = f.y.as_ref().unwrap();
        let y_pred = matrix(&f.json["y_pred"], name);
        let k = y.shape[1];
        let identity: Vec<f64> = (0..k * k)
            .map(|i| if i % (k + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let model = cavitation_ml::LinearModel {
            coef: Tensor::from_vec(identity, &[k, k]).unwrap(),
            intercept: vec![0.0; k],
        };
        let score = model.score(&value_dataset::<f32>(&y_pred, y)).unwrap();
        let expected = f.json["score"].as_f64().unwrap();
        assert!(
            (score - expected).abs() <= 1e-12 * expected.abs(),
            "{name}: R² {score} against scikit-learn's {expected}"
        );
    }
}
