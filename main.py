"""Train and audit a baseline credit-default model on a local CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TEST_SIZE = 0.25
IDENTIFIER_COLUMNS = {"id", "customerid", "member_id", "encounter_id"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(
    numeric: list[str], categorical: list[str], *, class_weight: str | None = None
) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessing),
            ("classifier", LogisticRegression(max_iter=1500, class_weight=class_weight)),
        ]
    )


def encode_positive_target(values: pd.Series, positive_label: str) -> np.ndarray:
    """Map the configured positive label to 1 regardless of lexical class order."""
    labels = values.astype(str)
    observed = set(labels.unique())
    if len(observed) != 2 or positive_label not in observed:
        raise ValueError("positive_label must identify one of exactly two observed classes")
    return (labels.to_numpy() == positive_label).astype(int)


def decode_binary_target(values: np.ndarray, positive_label: str, negative_label: str) -> np.ndarray:
    """Map the model's binary 1/0 convention back to the selected string labels."""
    return np.where(np.asarray(values) == 1, positive_label, negative_label)


def select_weighting_by_validation(candidate_scores: dict[str, float]) -> str:
    """Choose the candidate with the lowest validation Brier score."""
    if not candidate_scores or any(not np.isfinite(score) for score in candidate_scores.values()):
        raise ValueError("candidate validation Brier scores must be finite and non-empty")
    return min(candidate_scores, key=candidate_scores.get)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials or z <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and z > 0")
    if trials == 0:
        return None
    proportion = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (proportion + z2 / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z2 / (4 * trials * trials)) / denominator
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def probability_quality_metrics(
    train_labels: np.ndarray, test_labels: np.ndarray, probabilities: np.ndarray
) -> dict:
    """Compare probability forecasts with a constant prior fitted on training data only."""
    train_labels = np.asarray(train_labels, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if train_labels.size == 0 or test_labels.size == 0:
        raise ValueError("training and test labels must be non-empty")
    if probabilities.shape != test_labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and match test labels")
    if not set(np.unique(train_labels)).issubset({0, 1}) or not set(np.unique(test_labels)).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")
    prior = float(train_labels.mean())
    constant = np.full(test_labels.shape, prior, dtype=float)
    model_brier = float(brier_score_loss(test_labels, probabilities))
    baseline_brier = float(brier_score_loss(test_labels, constant))
    return {
        "model_brier_score": model_brier,
        "train_prevalence_constant_baseline_brier_score": baseline_brier,
        "brier_skill_score": float(1 - model_brier / baseline_brier) if baseline_brier > 0 else None,
        "model_log_loss": float(log_loss(test_labels, probabilities, labels=[0, 1])),
        "train_prevalence_constant_baseline_log_loss": float(log_loss(test_labels, constant, labels=[0, 1])),
        "training_prevalence": prior,
    }


def subgroup_audit(groups: pd.Series, truth: np.ndarray, predictions: np.ndarray) -> list[dict]:
    audit = []
    values = groups.fillna("UNKNOWN").astype(str).to_numpy()
    for value in sorted(set(values)):
        mask = values == value
        actual = truth[mask]
        predicted = predictions[mask]
        positives = int(actual.sum())
        negatives = int(len(actual) - positives)
        if len(actual) < 10:
            continue
        tp = int(((predicted == 1) & (actual == 1)).sum())
        fp = int(((predicted == 1) & (actual == 0)).sum())
        audit.append(
            {
                "group": value,
                "n": int(len(actual)),
                "positive_support": positives,
                "negative_support": negatives,
                "selection_rate": float(predicted.mean()),
                "tpr": float(tp / positives) if positives else None,
                "fpr": float(fp / negatives) if negatives else None,
                "tpr_wilson_95_ci": wilson_interval(tp, positives),
                "fpr_wilson_95_ci": wilson_interval(fp, negatives),
            }
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing predictors and target")
    parser.add_argument("--target", required=True, help="Binary target column")
    parser.add_argument("--group", help="Optional column for subgroup error-rate auditing")
    parser.add_argument("--positive-label", help="Label treated as the positive/default class")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between 0 and 1")
    if args.false_positive_cost < 0 or args.false_negative_cost < 0:
        parser.error("error costs must be nonnegative")

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    raw_input_rows = len(frame)
    if args.target not in frame.columns:
        parser.error(f"target column {args.target!r} is missing")
    if args.group and args.group not in frame.columns:
        parser.error(f"group column {args.group!r} is missing")

    frame = frame.loc[frame[args.target].notna()].copy()
    if frame.empty:
        parser.error("no rows with a non-missing target were found")

    groups = frame[args.group].reset_index(drop=True) if args.group else None
    labels = LabelEncoder().fit(frame[args.target].astype(str))
    if len(labels.classes_) != 2:
        parser.error("this baseline expects exactly two target classes")
    positive_label = str(args.positive_label) if args.positive_label is not None else str(labels.classes_[-1])
    if positive_label not in labels.classes_:
        parser.error(f"positive label must be one of {labels.classes_.tolist()}")
    negative_label = str(next(label for label in labels.classes_ if label != positive_label))
    truth_all = encode_positive_target(frame[args.target], positive_label)

    excluded = {args.target}
    if args.group:
        excluded.add(args.group)
    excluded.update(column for column in frame if column.lower() in IDENTIFIER_COLUMNS)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        parser.error("no model features remain after excluding the target, group, and identifiers")
    features = frame[feature_columns]
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in feature_columns if column not in numeric]

    train_x, test_x, train_y, test_y, _train_indices, test_indices = train_test_split(
        features,
        truth_all,
        np.arange(len(frame)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=truth_all,
    )
    fit_x, validation_x, fit_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=0.20,
        random_state=RANDOM_SEED + 1,
        stratify=train_y,
    )
    candidate_validation = {}
    for class_weight in (None, "balanced"):
        candidate = build_pipeline(numeric, categorical, class_weight=class_weight)
        candidate.fit(fit_x, fit_y)
        validation_probabilities = candidate.predict_proba(validation_x)[:, 1]
        candidate_validation["unweighted" if class_weight is None else "balanced"] = {
            "brier_score": float(brier_score_loss(validation_y, validation_probabilities)),
            "log_loss": float(log_loss(validation_y, validation_probabilities, labels=[0, 1])),
        }
    selected_weighting = select_weighting_by_validation(
        {name: result["brier_score"] for name, result in candidate_validation.items()}
    )
    selected_class_weight = None if selected_weighting == "unweighted" else "balanced"
    model = build_pipeline(numeric, categorical, class_weight=selected_class_weight)
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions, labels=[0, 1]).ravel()

    probability_quality = probability_quality_metrics(train_y, test_y, probabilities)

    source_metadata_path = input_path.parent / "source.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    metrics = {
        "input_rows": int(len(frame)),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "positive_label": positive_label,
        "threshold": args.threshold,
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "average_precision": float(average_precision_score(test_y, probabilities)),
        "brier_score": probability_quality["model_brier_score"],
        "log_loss": probability_quality["model_log_loss"],
        "probability_quality": probability_quality,
        "validation_model_selection": {
            "selection_metric": "Brier score (lower is better)",
            "selected_weighting": selected_weighting,
            "candidate_metrics": candidate_validation,
            "validation_fraction_of_training_partition": 0.20,
        },
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "group_audit": subgroup_audit(groups.iloc[test_indices], test_y, predictions) if groups is not None else [],
        "data": {
            "input_sha256": sha256_file(input_path),
            "source_archive_sha256": source_metadata.get("archive_sha256"),
            "source_url": source_metadata.get("source_url"),
            "source_rows_without_target": int(raw_input_rows - len(frame)),
        },
        "protocol": {
            "split": "stratified 60% fit / 15% validation / 25% final holdout; refit on train+validation",
            "test_fraction": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "features": feature_columns,
            "excluded_identifiers": sorted(column for column in frame if column.lower() in IDENTIFIER_COLUMNS),
            "threshold_costs": {
                "false_positive": args.false_positive_cost,
                "false_negative": args.false_negative_cost,
            },
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    sweep = []
    for threshold in np.linspace(0.05, 0.95, 19):
        decision = probabilities >= threshold
        false_positives = int(((decision == 1) & (test_y == 0)).sum())
        false_negatives = int(((decision == 0) & (test_y == 1)).sum())
        true_positives = int(((decision == 1) & (test_y == 1)).sum())
        sweep.append(
            {
                "threshold": float(threshold),
                "precision": float(true_positives / max(1, int(decision.sum()))),
                "recall": float(true_positives / max(1, int(test_y.sum()))),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "weighted_error_cost": float(
                    false_positives * args.false_positive_cost
                    + false_negatives * args.false_negative_cost
                ),
            }
        )

    pd.DataFrame(sweep).to_csv(output / "threshold_tradeoffs.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "actual": decode_binary_target(test_y, positive_label, negative_label),
            "probability_positive": probabilities,
            "predicted": decode_binary_target(predictions, positive_label, negative_label),
        }
    ).to_csv(output / "holdout_predictions.csv", index=False)
    joblib.dump(
        {
            "pipeline": model,
            "classes": labels.classes_.tolist(),
            "positive_label": positive_label,
            "positive_model_class": 1,
            "threshold": args.threshold,
            "features": feature_columns,
        },
        output / "model.joblib",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

