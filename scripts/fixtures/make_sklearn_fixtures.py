"""Writes the scikit-learn reference fixtures that the `cavitation-ml` tests compare against.

One JSON file per case goes under `crates/cavitation-ml/tests/fixtures/sklearn/` (hyphenated
names, for example `logistic-l2-1-newton-cholesky.json`). A file holds:

- `sklearn_version`: the scikit-learn version that wrote it;
- `estimator`: `LinearRegression`, `Ridge`, `LogisticRegression`, `StandardScaler` or
  `r2_score`;
- `params`: the estimator's parameters as passed. A logistic regression without a penalty has
  `"C": null`; scikit-learn 1.9 spells it `C=numpy.inf` (its `penalty` argument is
  deprecated), which JSON cannot hold;
- `x` and `y`: the data, either `{"shape", "data"}` (row-major) or, for the Concrete and
  Breast Cancer cases, a reference to the training rows of the library's own loader, so the
  data set is not copied into every file (see "Data set references" below). `y` is
  `[rows, outputs]` for regression and `[rows]` of class labels for classification;
- `coef` (`[outputs or classes, features]`, scikit-learn's `coef_` as a matrix), `intercept`
  (one per output or class), `objective` and `n_iter` (scikit-learn's `n_iter_`, or `null` for
  the estimators that have none), where they apply;
- `objective_form`: the objective `objective` is the value of, at scikit-learn's solution:
  `sum_sq` is `Σ (y − x·coefᵀ − intercept)²` over every row and output, plus `alpha · ‖coef‖²`
  for ridge; `logistic` is `C · Σ log(1 + exp(−s·z)) + ½‖coef‖²` with `s = ±1` for labels
  1 and 0 and `z` the linear score (no `½‖coef‖²` and no `C` without a penalty);
  `multinomial` is `C · Σ −log softmax(z)_label + ½‖coef‖²` (the Frobenius norm); the
  intercept is never penalised;
- `StandardScaler` adds `mean`, `scale` (the population standard deviation, as scikit-learn)
  and `var`; `r2_score` adds `y_pred`, `score` (the uniform average over outputs) and
  `raw_values` (one per output).

Synthetic inputs come from `numpy.random.default_rng(0)`, drawn in a fixed order, and every
input and target is rounded through float32 before fitting in float64, so the library (which
reads data as f32 and widens it) and scikit-learn solve exactly the same problem. The `lbfgs`
and `newton-cholesky` logistic fits use `tol=1e-12` and `max_iter=100000`. Every warning is an
error, so a fit that did not converge cannot be written as a fixture.

## Data set references

For the Concrete and Breast Cancer cases, `x` is

    {"dataset": "concrete" | "breast-cancer", "split": "train", "shape": [rows, features],
     "first_row": [...], "sum": ..., "standardize": {"mean": [...], "scale": [...]} or absent}

meaning: the training rows of `cavitation_datasets::concrete::load()` (or `breast_cancer`),
widened from f32 to f64, then, if `standardize` is present, `(x − mean) / scale` per column
with these exact `mean` and `scale` (scikit-learn's `StandardScaler` fitted on those rows,
which computes the same two operations). `first_row` (before standardising) and `sum` (of
every value before standardising) let the reader check it built the same rows. `y` is
`{"dataset", "split"}` and means that split's targets. The script reads `data/concrete/
concrete.csv` and `data/breast-cancer/breast-cancer.csv`, checks their SHA-256, reads each
value as float64 and rounds it to float32 (as the loaders do), and shuffles the rows with
mulberry32 seed 42 exactly as `data/digits/README.md` describes; the first 824 (Concrete) or
455 (Breast Cancer) rows of the shuffled order are the training rows. It also checks the
Breast Cancer CSV against the copy bundled with scikit-learn (`load_breast_cancer`): the same
569 rows, values and classes.

Run it from the repository root, inside the fixture environment (see
`scripts/fixtures/requirements.txt`):

    scripts/fixtures/.venv/bin/python -I scripts/fixtures/make_sklearn_fixtures.py --out crates

Floats are written with 17 significant digits, enough to read back the exact float64 values.
Every check exits with status 1 and a message, also under `python -O`.
"""

import warnings

warnings.simplefilter("error")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DIGITS = 17
SKLEARN_VERSION = sklearn.__version__
LBFGS_TOL = 1e-12
MAX_ITER = 100000

# The data sets the loaders read: path, SHA-256, rows, features, training rows.
DATASETS = {
    "concrete": (
        "data/concrete/concrete.csv",
        "858bd88ee4817fa50a3915312403ea425d91c34927e77c8bcfc334eee6544995",
        1030,
        8,
        824,
    ),
    "breast-cancer": (
        "data/breast-cancer/breast-cancer.csv",
        "432ff316e7bfb60b70a275064b4401315cc39f09c9099d031013a23647e98687",
        569,
        30,
        455,
    ),
}


def check(condition, message):
    """Exits with status 1 and `message` unless `condition` holds (works under `python -O`)."""
    if not condition:
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)


def number(value):
    """A float rounded to 17 significant digits, as a JSON number."""
    value = float(value)
    check(math.isfinite(value), f"non-finite value {value}")
    return float(format(value, f".{DIGITS}g"))


def matrix_json(a):
    """The `{shape, data}` form of an array, row-major."""
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "data": [number(v) for v in a.reshape(-1).tolist()]}


def via_f32(a):
    """`a` rounded to float32 and widened back to float64."""
    return np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


def mulberry32(seed):
    """The mulberry32 generator: numbers in [0, 1), as `cavitation_datasets::Mulberry32`."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        yield (t ^ (t >> 14)) / 4294967296


def split_order(n):
    """The loaders' shuffled row order: Fisher-Yates from the end with mulberry32 seed 42."""
    rng = mulberry32(42)
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        k = int(next(rng) * (i + 1))
        order[i], order[k] = order[k], order[i]
    return order


def read_dataset(name):
    """Reads a data set's CSV: (features as f32-rounded float64, targets, rows as float64)."""
    path, sha256, rows, features, _ = DATASETS[name]
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    check(digest == sha256, f"{path} has SHA-256 {digest}, expected {sha256}")
    reader = csv.reader(raw.decode("ascii").splitlines())
    next(reader)
    values = np.array([[float(v) for v in row] for row in reader], dtype=np.float64)
    check(values.shape == (rows, features + 1), f"{path} is {values.shape}")
    return via_f32(values[:, :features]), via_f32(values[:, features]), values


def check_breast_cancer_against_sklearn(values):
    """The Breast Cancer CSV holds the same rows, values and classes as scikit-learn's copy."""
    bundled = load_breast_cancer()
    check(bundled.data.shape == (569, 30), "scikit-learn's breast cancer copy is not 569 by 30")
    check(np.array_equal(values[:, :30], bundled.data), "breast-cancer.csv values differ from scikit-learn's")
    check(np.array_equal(values[:, 30], bundled.target), "breast-cancer.csv classes differ from scikit-learn's")
    check(list(bundled.target_names) == ["malignant", "benign"], "scikit-learn's class names changed")


def train_rows(name):
    """The training rows of a data set, as the library's loader builds them."""
    x, y, _ = read_dataset(name)
    train = DATASETS[name][4]
    order = split_order(len(x))[:train]
    return x[order], y[order]


def dataset_ref(name, x, standardize=None):
    """The reference form of `x` (training rows of `name`), optionally standardised."""
    ref = {
        "dataset": name,
        "split": "train",
        "shape": list(x.shape),
        "first_row": [number(v) for v in x[0]],
        "sum": number(x.sum()),
    }
    if standardize is not None:
        ref["standardize"] = {
            "mean": [number(v) for v in standardize.mean_],
            "scale": [number(v) for v in standardize.scale_],
        }
    return ref


def linear_objective(x, y, coef, intercept, alpha):
    """`Σ (y − x·coefᵀ − intercept)² + alpha · ‖coef‖²`, with `y` as `[rows, outputs]`."""
    residual = y - x @ coef.T - intercept
    return float((residual**2).sum() + alpha * (coef**2).sum())


def logistic_objective(x, y, coef, intercept, c):
    """`C · Σ log(1 + exp(−s·z)) + ½‖coef‖²`; without a penalty (`c` None) just the sum."""
    z = x @ coef[0] + intercept[0]
    s = np.where(y == 1, 1.0, -1.0)
    loss = float(np.logaddexp(0.0, -s * z).sum())
    if c is None:
        return loss
    return c * loss + 0.5 * float((coef**2).sum())


def multinomial_objective(x, y, coef, intercept, c):
    """`C · Σ −log softmax(z)_label + ½‖coef‖²`."""
    z = x @ coef.T + intercept
    zmax = z.max(axis=1, keepdims=True)
    log_softmax = z - zmax - np.log(np.exp(z - zmax).sum(axis=1, keepdims=True))
    loss = float(-log_softmax[np.arange(len(y)), y.astype(int)].sum())
    return c * loss + 0.5 * float((coef**2).sum())


def as_rows(y):
    """`y` as a `[rows, outputs]` matrix."""
    y = np.asarray(y, dtype=np.float64)
    return y.reshape(len(y), -1)


def linear_case(estimator, params, x, y, x_json=None, y_json=None):
    """Fits LinearRegression or Ridge and returns the fixture body."""
    model = (LinearRegression if estimator == "LinearRegression" else Ridge)(**params)
    y_fit = y if y.shape[1] > 1 else y[:, 0]
    model.fit(x, y_fit)
    coef = np.asarray(model.coef_, dtype=np.float64).reshape(y.shape[1], x.shape[1])
    intercept = np.atleast_1d(np.asarray(model.intercept_, dtype=np.float64))
    if not params.get("fit_intercept", True):
        intercept = np.zeros(y.shape[1])
    alpha = params.get("alpha", 0.0)
    n_iter = getattr(model, "n_iter_", None)
    check(n_iter is None, f"{estimator} reported n_iter {n_iter}")
    return {
        "sklearn_version": SKLEARN_VERSION,
        "estimator": estimator,
        "params": params,
        "x": x_json if x_json is not None else matrix_json(x),
        "y": y_json if y_json is not None else matrix_json(y),
        "coef": matrix_json(coef),
        "intercept": [number(v) for v in intercept],
        "objective": number(linear_objective(x, y, coef, intercept, alpha)),
        "objective_form": "sum_sq",
        "n_iter": None,
    }


def logistic_case(c, solver, x, y, x_json=None, y_json=None):
    """Fits LogisticRegression (`c` None for no penalty) and returns the fixture body."""
    params = {"C": c, "solver": solver, "tol": LBFGS_TOL, "max_iter": MAX_ITER}
    model = LogisticRegression(C=np.inf if c is None else c, solver=solver, tol=LBFGS_TOL, max_iter=MAX_ITER)
    model.fit(x, y.astype(int))
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    classes = sorted(set(y.astype(int).tolist()))
    check(list(model.classes_) == classes, "unexpected classes")
    if len(classes) == 2:
        objective = logistic_objective(x, y, coef, intercept, c)
        form = "logistic"
    else:
        objective = multinomial_objective(x, y, coef, intercept, c)
        form = "multinomial"
    n_iter = int(np.max(model.n_iter_))
    check(n_iter < MAX_ITER, f"{solver} used every iteration")
    return {
        "sklearn_version": SKLEARN_VERSION,
        "estimator": "LogisticRegression",
        "params": params,
        "x": x_json if x_json is not None else matrix_json(x),
        "y": y_json if y_json is not None else matrix_json(y),
        "coef": matrix_json(coef),
        "intercept": [number(v) for v in intercept],
        "objective": number(objective),
        "objective_form": form,
        "n_iter": n_iter,
    }


def sigmoid(z):
    """The logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def build_cases():
    """Every fixture case, as (name, body) pairs in a fixed order."""
    rng = np.random.default_rng(0)
    cases = []

    # Ordinary least squares.
    x = via_f32(rng.normal(size=(50, 3)))
    y = via_f32(as_rows(x @ np.array([1.5, -2.0, 0.5]) + 3.0 + 0.1 * rng.normal(size=50)))
    cases.append(("ols-small", linear_case("LinearRegression", {"fit_intercept": True}, x, y)))

    x = via_f32(rng.normal(size=(200, 8)))
    w = rng.normal(size=(8, 3))
    y = via_f32(x @ w + np.array([1.0, -1.0, 0.5]) + 0.1 * rng.normal(size=(200, 3)))
    cases.append(("ols-multi", linear_case("LinearRegression", {"fit_intercept": True}, x, y)))

    x = via_f32(rng.normal(size=(50, 3)))
    y = via_f32(as_rows(x @ np.array([0.7, 1.2, -0.4]) + 0.5 + 0.1 * rng.normal(size=50)))
    cases.append(("ols-no-intercept", linear_case("LinearRegression", {"fit_intercept": False}, x, y)))

    x = via_f32(rng.normal(size=(60, 4)))
    x[:, 2] = 3.0
    y = via_f32(as_rows(x @ np.array([1.0, -0.5, 2.0, 0.25]) + 1.0 + 0.1 * rng.normal(size=60)))
    body = linear_case("LinearRegression", {"fit_intercept": True}, x, y)
    constant_coef = body["coef"]["data"][2]
    check(abs(constant_coef) <= 1e-12, f"constant column's coefficient is {constant_coef}")
    cases.append(("ols-constant-column", body))

    # Ridge.
    x = via_f32(rng.normal(size=(100, 5)))
    y = via_f32(as_rows(x @ np.array([2.0, -1.0, 0.0, 0.5, 1.5]) - 0.5 + 0.3 * rng.normal(size=100)))
    for alpha, name in [(0.1, "ridge-0.1"), (10.0, "ridge-10")]:
        params = {"alpha": alpha, "fit_intercept": True, "solver": "cholesky"}
        cases.append((name, linear_case("Ridge", params, x, y)))

    # Binary logistic regression on overlapping classes.
    x = via_f32(rng.normal(size=(300, 4)))
    p = sigmoid(x @ np.array([1.0, -1.0, 0.5, 0.25]) + 0.3)
    y = (rng.uniform(size=300) < p).astype(np.float64)
    for c, name in [(None, "logistic-none"), (0.5, "logistic-l2-0.5"), (1.0, "logistic-l2-1")]:
        for solver in ["newton-cholesky", "lbfgs"]:
            cases.append((f"{name}-{solver}", logistic_case(c, solver, x, y)))

    # Multinomial logistic regression, three classes.
    x = via_f32(rng.normal(size=(300, 4)))
    w = rng.normal(size=(4, 3))
    z = x @ w + np.array([0.2, -0.1, 0.0])
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p], dtype=np.float64)
    cases.append(("multinomial-l2-1", logistic_case(1.0, "lbfgs", x, y)))

    # r2_score, one output and three outputs.
    y_true = via_f32(rng.normal(size=(40, 1)))
    y_pred = via_f32(y_true + 0.3 * rng.normal(size=(40, 1)))
    y_true3 = via_f32(rng.normal(size=(40, 3)) * np.array([1.0, 5.0, 0.2]))
    y_pred3 = via_f32(y_true3 + 0.3 * rng.normal(size=(40, 3)))
    for name, yt, yp in [("r2-score", y_true, y_pred), ("r2-score-multi", y_true3, y_pred3)]:
        yt_fit = yt if yt.shape[1] > 1 else yt[:, 0]
        yp_fit = yp if yp.shape[1] > 1 else yp[:, 0]
        score = r2_score(yt_fit, yp_fit)
        raw = np.atleast_1d(r2_score(yt_fit, yp_fit, multioutput="raw_values"))
        cases.append(
            (
                name,
                {
                    "sklearn_version": SKLEARN_VERSION,
                    "estimator": "r2_score",
                    "params": {"multioutput": "uniform_average"},
                    "y": matrix_json(yt),
                    "y_pred": matrix_json(yp),
                    "score": number(score),
                    "raw_values": [number(v) for v in raw],
                },
            )
        )

    # Concrete: the scaler, then OLS and ridge on the standardised training rows.
    x, y = train_rows("concrete")
    scaler = StandardScaler().fit(x)
    cases.append(
        (
            "standard-scaler-concrete",
            {
                "sklearn_version": SKLEARN_VERSION,
                "estimator": "StandardScaler",
                "params": {"with_mean": True, "with_std": True},
                "x": dataset_ref("concrete", x),
                "mean": [number(v) for v in scaler.mean_],
                "scale": [number(v) for v in scaler.scale_],
                "var": [number(v) for v in scaler.var_],
            },
        )
    )
    xs = scaler.transform(x)
    x_json = dataset_ref("concrete", x, scaler)
    y_json = {"dataset": "concrete", "split": "train"}
    cases.append(
        ("concrete-ols", linear_case("LinearRegression", {"fit_intercept": True}, xs, as_rows(y), x_json, y_json))
    )
    params = {"alpha": 1.0, "fit_intercept": True, "solver": "cholesky"}
    cases.append(("concrete-ridge-1", linear_case("Ridge", params, xs, as_rows(y), x_json, y_json)))

    # Breast Cancer: L2 logistic regression with C = 1 on the standardised training rows.
    _, _, values = read_dataset("breast-cancer")
    check_breast_cancer_against_sklearn(values)
    x, y = train_rows("breast-cancer")
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    x_json = dataset_ref("breast-cancer", x, scaler)
    y_json = {"dataset": "breast-cancer", "split": "train"}
    for solver in ["newton-cholesky", "lbfgs"]:
        cases.append(
            (f"breast-cancer-logistic-l2-1-{solver}", logistic_case(1.0, solver, xs, y, x_json, y_json))
        )
    return cases


def main():
    """Writes every fixture under `--out`/cavitation-ml/tests/fixtures/sklearn."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="crates", help="the crates folder (default: crates)")
    args = parser.parse_args()
    cases = build_cases()
    out_dir = os.path.join(args.out, "cavitation-ml", "tests", "fixtures", "sklearn")
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".json"):
            os.remove(os.path.join(out_dir, old))
    names = set()
    for name, body in cases:
        check(name not in names, f"duplicate fixture name {name}")
        names.add(name)
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(body, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
    print(f"wrote {len(cases)} fixtures to {out_dir} with scikit-learn {SKLEARN_VERSION}", file=sys.stderr)


if __name__ == "__main__":
    main()
