"""Train and audit a baseline credit-default model on a local CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
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
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )


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
    encoded_target = labels.transform(frame[args.target].astype(str))
    if len(labels.classes_) != 2:
        parser.error("this baseline expects exactly two target classes")
    positive_label = str(args.positive_label) if args.positive_label is not None else str(labels.classes_[-1])
    if positive_label not in labels.classes_:
        parser.error(f"positive label must be one of {labels.classes_.tolist()}")
    positive_class = int(np.where(labels.classes_ == positive_label)[0][0])
    truth_all = (encoded_target == positive_class).astype(int)

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
    model = build_pipeline(numeric, categorical)
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions, labels=[0, 1]).ravel()

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
        "brier_score": float(brier_score_loss(test_y, probabilities)),
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
            "split": "stratified random holdout",
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
            "actual": labels.inverse_transform(test_y),
            "probability_positive": probabilities,
            "predicted": np.where(predictions == 1, positive_label, labels.classes_[1 - positive_class]),
        }
    ).to_csv(output / "holdout_predictions.csv", index=False)
    joblib.dump(
        {
            "pipeline": model,
            "classes": labels.classes_.tolist(),
            "positive_label": positive_label,
            "threshold": args.threshold,
            "features": feature_columns,
        },
        output / "model.joblib",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

