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
  (`average_precision_score`).

Inputs come from `numpy.random.default_rng(seed)`, a seed per case. Every warning is an
error, so no case relies on scikit-learn's warning paths except where `zero_division=0` is
passed explicitly.

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
    precision_recall_fscore_support,
    roc_auc_score,
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
