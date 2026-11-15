"""Writes the scikit-learn fixtures that `crates/cavitation-nn/tests/metrics.rs` compares the
classification metrics against.

One JSON file per case goes under `crates/cavitation-nn/tests/fixtures/metrics/`:

- `sklearn_version`: the scikit-learn version that wrote it;
- class cases (`kind: "classes"`): `classes`, `targets` and `predicted` (labels below
  `classes`); `confusion` (`confusion_matrix` with `labels=range(classes)`, true class down);
  `accuracy`; `labels` (the classes scikit-learn's default labels hold: those among the
  targets or the predictions); `per_class` (`precision`, `recall`, `f1`, `f2`, `support`, one
  value per entry of `labels`, from `precision_recall_fscore_support` with
  `zero_division=0`); and `averages` (`macro`, `weighted`, `micro`, each with `precision`,
  `recall`, `f1` and `f2`);
- score cases (`kind: "scores"`): `scores` (float32 values, as the library takes them),
  `positive` (booleans), `roc_auc` (`roc_auc_score`) and `average_precision`
  (`average_precision_score`);
- regression cases (`kind: "regression"`): `outputs`, `targets` and `predicted` (float32
  values, `[rows, outputs]` row-major), and per output (`raw_values`) and averaged
  (`uniform_average`): `mae`, `mse`, `rmse` (`root_mean_squared_error`) and `r2`;
- probability cases (`kind: "probabilities"`): `classes`, `labels`, `probabilities` (float32
  values, `[rows, classes]`, softmax of seeded logits),
  `log_loss` (with `labels=range(classes)`, on the float32 values widened to float64; no
  probability is near enough to 0 or 1 to be clipped), `scores` (float32, with ties) and `top_k` (a map from k to
  `top_k_accuracy_score` on `scores`).

Inputs come from `numpy.random.default_rng(seed)`, a seed per case. Every warning is an
error, so no case relies on scikit-learn's warning paths, except where `zero_division=0` is
passed explicitly and the one `log_loss` warning `quiet_log_loss` describes.

Run with the fixtures environment (`scripts/fixtures/requirements.txt`):

    scripts/fixtures/.venv/bin/python -I scripts/fixtures/make_metrics_fixtures.py --out crates
"""

import argparse
import json
import pathlib
import warnings

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    top_k_accuracy_score,
)


def class_case(name, classes, targets, predicted):
    """A class case: the confusion matrix, accuracy, per-class rates and averages."""
    targets = [int(t) for t in targets]
    predicted = [int(p) for p in predicted]
    labels = sorted(set(targets) | set(predicted))
    p, r, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=labels, zero_division=0
    )
    f2 = fbeta_score(targets, predicted, beta=2.0, labels=labels, average=None, zero_division=0)
    averages = {}
    for avg in ["macro", "weighted", "micro"]:
        ap, ar, af1, _ = precision_recall_fscore_support(
            targets, predicted, average=avg, zero_division=0
        )
        af2 = fbeta_score(targets, predicted, beta=2.0, average=avg, zero_division=0)
        averages[avg] = {
            "precision": float(ap),
            "recall": float(ar),
            "f1": float(af1),
            "f2": float(af2),
        }
    return name, {
        "sklearn_version": sklearn.__version__,
        "kind": "classes",
        "classes": classes,
        "targets": targets,
        "predicted": predicted,
        "confusion": confusion_matrix(targets, predicted, labels=list(range(classes))).tolist(),
        "accuracy": float(accuracy_score(targets, predicted)),
        "labels": labels,
        "per_class": {
            "precision": [float(v) for v in p],
            "recall": [float(v) for v in r],
            "f1": [float(v) for v in f1],
            "f2": [float(v) for v in f2],
            "support": [int(v) for v in support],
        },
        "averages": averages,
    }


def score_case(name, scores, positive):
    """A score case: ROC AUC and average precision of float32 scores."""
    scores = np.asarray(scores, dtype=np.float32)
    positive = np.asarray(positive, dtype=bool)
    return name, {
        "sklearn_version": sklearn.__version__,
        "kind": "scores",
        "scores": [float(s) for s in scores],
        "positive": [bool(b) for b in positive],
        "roc_auc": float(roc_auc_score(positive, scores.astype(np.float64))),
        "average_precision": float(average_precision_score(positive, scores.astype(np.float64))),
    }


def regression_case(name, targets, predicted):
    """A regression case: the four errors per output and averaged."""
    t = np.asarray(targets, dtype=np.float32).astype(np.float64)
    p = np.asarray(predicted, dtype=np.float32).astype(np.float64)
    out = {
        "sklearn_version": sklearn.__version__,
        "kind": "regression",
        "outputs": int(t.shape[1]),
        "targets": [float(v) for v in t.ravel()],
        "predicted": [float(v) for v in p.ravel()],
    }
    for key, fn in [
        ("mae", mean_absolute_error),
        ("mse", mean_squared_error),
        ("rmse", root_mean_squared_error),
        ("r2", r2_score),
    ]:
        out[key] = {
            "raw": [float(v) for v in fn(t, p, multioutput="raw_values")],
            "mean": float(fn(t, p, multioutput="uniform_average")),
        }
    return name, out


def quiet_log_loss(labels, prob32, classes):
    """`log_loss` on float32 probabilities widened to float64, the values the library reads.

    Rounded to float32, a row's sum is off 1 by about 1e-8, above the float64 tolerance
    `log_loss` warns at, so that one warning is allowed here and no other.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
        return float(log_loss(labels, prob32.astype(np.float64), labels=list(range(classes))))


def probability_case(name, classes, labels, logits, scores):
    """A probability case: log loss of softmax probabilities, top-k accuracy of scores."""
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    prob32 = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    s = np.asarray(scores, dtype=np.float32)
    return name, {
        "sklearn_version": sklearn.__version__,
        "kind": "probabilities",
        "classes": classes,
        "labels": [int(l) for l in labels],
        "probabilities": [float(v) for v in prob32.ravel()],
        "log_loss": quiet_log_loss(labels, prob32, classes),
        "scores": [float(v) for v in s.ravel()],
        "top_k": {
            str(k): float(top_k_accuracy_score(labels, s, k=k, labels=list(range(classes))))
            for k in range(1, classes)
        },
    }


def cases():
    """Every case, in a fixed order."""
    out = []
    # Imbalanced four classes, the rarest never predicted; a fifth class appears only among
    # the predictions.
    rng = np.random.default_rng(1)
    t = rng.choice(4, size=300, p=[0.6, 0.25, 0.1, 0.05])
    p = np.where(rng.random(300) < 0.7, t, rng.choice(3, size=300))
    p = np.where(p == 3, 0, p)
    p[[5, 17]] = 4
    out.append(class_case("classes-imbalanced", 5, t, p))
    # Three balanced classes with a fair share of mistakes.
    rng = np.random.default_rng(2)
    t = rng.integers(0, 3, size=90)
    p = np.where(rng.random(90) < 0.6, t, rng.integers(0, 3, size=90))
    out.append(class_case("classes-balanced", 3, t, p))
    # Two classes, every prediction right.
    out.append(class_case("classes-perfect", 2, [0, 1, 1, 0, 1], [0, 1, 1, 0, 1]))
    # Scores with many ties (rounded to one decimal).
    rng = np.random.default_rng(3)
    pos = rng.random(200) < 0.3
    s = np.round(np.clip(rng.normal(np.where(pos, 0.65, 0.4), 0.2), 0, 1), 1)
    out.append(score_case("scores-ties", s, pos))
    # Continuous scores, a rare positive class.
    rng = np.random.default_rng(4)
    pos = rng.random(500) < 0.08
    s = rng.normal(np.where(pos, 1.0, 0.0), 1.0)
    out.append(score_case("scores-rare", s, pos))
    # A perfect ranking and its reverse.
    out.append(score_case("scores-perfect", [0.9, 0.8, 0.3, 0.1], [True, True, False, False]))
    out.append(score_case("scores-reversed", [0.9, 0.8, 0.3, 0.1], [False, False, True, True]))
    # Three outputs on different scales; the third has constant targets predicted exactly.
    rng = np.random.default_rng(5)
    t = np.stack([rng.normal(0, 1, 60), rng.normal(50, 10, 60), np.full(60, 2.5)], axis=1)
    p = t + np.stack([rng.normal(0, 0.3, 60), rng.normal(0, 4, 60), np.zeros(60)], axis=1)
    out.append(regression_case("regression-three-outputs", t, p))
    # One output, constant targets missed: R² is 0 under force_finite.
    out.append(regression_case("regression-constant-missed", np.full((5, 1), 1.0), [[1.0], [1.5], [1.0], [0.5], [1.0]]))
    # Five classes: probabilities from seeded logits, and scores rounded so rows hold ties.
    rng = np.random.default_rng(6)
    labels = rng.integers(0, 5, size=120)
    logits = rng.normal(0, 1, (120, 5)) + 1.5 * np.eye(5)[labels]
    scores = np.round(rng.random((120, 5)) + 0.4 * np.eye(5)[labels], 1)
    out.append(probability_case("probabilities-five-classes", 5, labels, logits, scores))
    return out


def main():
    """Writes every case."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="the crates directory")
    args = parser.parse_args()
    folder = pathlib.Path(args.out) / "cavitation-nn" / "tests" / "fixtures" / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, case in cases():
            (folder / f"{name}.json").write_text(json.dumps(case, indent=1) + "\n")
            print(f"wrote {name}.json")


if __name__ == "__main__":
    main()
