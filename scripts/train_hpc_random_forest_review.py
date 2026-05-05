#!/usr/bin/env python3
"""Train a scikit-learn RandomForest motif classifier and export review candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from export_hpc_label_review import collect_candidates, load_labels
from train_hpc_mlp_review import select_review_rows
from train_hpc_round_review import (
    build_feature_matrix,
    candidate_key,
    read_reviewed_labels,
    write_review_csv,
    write_review_html,
)


def write_report(path: Path, labels: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    path.write_text(report)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument(
        "--predict-raw-dir",
        type=Path,
        default=None,
        help="Optional raw directory to predict/export after training on --raw-dir.",
    )
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_random_forest_round"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--target-labels", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    all_schema_labels = load_labels(args.schema)
    reviewed_rows = read_reviewed_labels(args.labels_csv)
    train_source_candidates, times = collect_candidates(args.raw_dir)
    by_key = {candidate_key(row): row for row in train_source_candidates}

    train_candidates = []
    y_labels = []
    for row in reviewed_rows:
        candidate = by_key.get(candidate_key(row))
        if candidate is None:
            continue
        train_candidates.append(candidate)
        y_labels.append(row["reviewed_label"].strip())
    if not train_candidates:
        raise SystemExit("No labels matched HPC candidates.")

    trained_labels = sorted(set(y_labels))
    if args.target_labels:
        unknown_targets = [label for label in args.target_labels if label not in trained_labels]
        if unknown_targets:
            raise SystemExit(f"Target labels have no training examples: {', '.join(unknown_targets)}")
        review_labels = args.target_labels
    else:
        review_labels = trained_labels

    x = build_feature_matrix(train_candidates)
    y = np.array(y_labels, dtype=object)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    if args.predict_raw_dir is None:
        predict_candidates = train_source_candidates
        exclude_keys = {candidate_key(row) for row in train_candidates}
    else:
        predict_candidates, times = collect_candidates(args.predict_raw_dir)
        exclude_keys = set()

    x_all = build_feature_matrix(predict_candidates)
    probs = model.predict_proba(x_all)
    model_labels = model.classes_.tolist()
    selected = select_review_rows(predict_candidates, probs, model_labels, review_labels, exclude_keys, args.per_class)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_review_csv(args.out_dir / "hpc_random_forest_review_candidates.csv", selected)
    write_review_html(args.out_dir / "hpc_random_forest_review.html", selected, all_schema_labels, times)
    joblib.dump(model, args.out_dir / "hpc_random_forest_model.joblib")
    write_report(args.out_dir / "hpc_random_forest_validation_report.txt", trained_labels, y_val, val_pred)
    write_metrics_csv(
        args.out_dir / "hpc_random_forest_metrics.csv",
        [
            {"metric": "train_accuracy", "value": accuracy_score(y_train, train_pred)},
            {"metric": "val_accuracy", "value": accuracy_score(y_val, val_pred)},
            {"metric": "n_labeled", "value": len(train_candidates)},
            {"metric": "n_train", "value": len(y_train)},
            {"metric": "n_val", "value": len(y_val)},
            {"metric": "n_prediction_candidates", "value": len(predict_candidates)},
        ],
    )

    print(f"matched labeled rows: {len(train_candidates)}")
    print("training label counts:")
    for label in trained_labels:
        print(f"  {label}: {int(np.sum(y == label))}")
    untrained = [label for label in all_schema_labels if label not in trained_labels]
    if untrained:
        print("untrained labels with zero examples:")
        for label in untrained:
            print(f"  {label}")
    print(f"train_accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"val_accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print("selected review counts:")
    for label in review_labels:
        print(f"  {label}: {sum(row['predicted_label'] == label for row in selected)}")
    print(f"prediction candidates: {len(predict_candidates)}")
    print(f"html: {args.out_dir / 'hpc_random_forest_review.html'}")
    print(f"candidate csv: {args.out_dir / 'hpc_random_forest_review_candidates.csv'}")
    print(f"model: {args.out_dir / 'hpc_random_forest_model.joblib'}")
    print(f"validation report: {args.out_dir / 'hpc_random_forest_validation_report.txt'}")


if __name__ == "__main__":
    main()
