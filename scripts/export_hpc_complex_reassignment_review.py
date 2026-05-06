#!/usr/bin/env python3
"""Export human-labeled complex traces plus RF complex-like HPC candidates for review."""

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
from train_hpc_round_review import build_feature_matrix, candidate_key, read_reviewed_labels, write_review_html


def write_review_csv(path: Path, selected: list[dict]) -> None:
    fieldnames = [
        "source_file",
        "source_path",
        "trace_index",
        "fiber_id",
        "nd2_name",
        "tile_name",
        "baseline_9p5_10d",
        "shifted_max",
        "shifted_min",
        "shifted_amplitude",
        "review_source",
        "predicted_label",
        "model_top_label",
        "prediction_confidence",
        "target_label_distance",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_complex_reassignment_round"))
    parser.add_argument("--predicted-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    all_schema_labels = load_labels(args.schema)
    reviewed_rows = read_reviewed_labels(args.labels_csv)
    candidates, times = collect_candidates(args.raw_dir)
    by_key = {candidate_key(row): row for row in candidates}

    train_candidates = []
    y_labels = []
    human_complex_keys = set()
    for row in reviewed_rows:
        label = row["reviewed_label"].strip()
        candidate = by_key.get(candidate_key(row))
        if candidate is None:
            continue
        train_candidates.append(candidate)
        y_labels.append(label)
        if label == "complex":
            human_complex_keys.add(candidate_key(row))
    if not train_candidates:
        raise SystemExit("No labels matched HPC candidates.")
    if "complex" not in set(y_labels):
        raise SystemExit("No human-labeled complex traces found.")

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

    x_all = build_feature_matrix(candidates)
    probs = model.predict_proba(x_all)
    model_labels = model.classes_.tolist()
    exclude_keys = {candidate_key(row) for row in train_candidates}
    predicted_complex = select_review_rows(
        candidates,
        probs,
        model_labels,
        ["complex"],
        exclude_keys,
        args.predicted_count,
    )
    for row in predicted_complex:
        row["review_source"] = "rf_complex_candidate"

    human_complex = []
    complex_idx = model_labels.index("complex")
    probs_by_key = {candidate_key(row): prob for row, prob in zip(candidates, probs, strict=False)}
    for row in train_candidates:
        key = candidate_key(row)
        if key not in human_complex_keys:
            continue
        out = dict(row)
        prob = probs_by_key.get(key)
        out["review_source"] = "human_label_complex"
        out["predicted_label"] = "complex"
        out["model_top_label"] = "human_label_complex"
        out["prediction_confidence"] = float(prob[complex_idx]) if prob is not None else ""
        out["target_label_distance"] = ""
        human_complex.append(out)

    human_complex.sort(key=lambda row: (row["source_file"], row["trace_index"]))
    selected = human_complex + predicted_complex

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_review_csv(args.out_dir / "hpc_complex_reassignment_candidates.csv", selected)
    write_review_html(args.out_dir / "hpc_complex_reassignment_review.html", selected, all_schema_labels, times)
    joblib.dump(model, args.out_dir / "hpc_complex_reassignment_random_forest_model.joblib")
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    (args.out_dir / "hpc_complex_reassignment_validation_report.txt").write_text(
        classification_report(y_val, val_pred, labels=all_schema_labels, zero_division=0),
        encoding="utf-8",
    )
    write_metrics_csv(
        args.out_dir / "hpc_complex_reassignment_metrics.csv",
        [
            {"metric": "train_accuracy", "value": accuracy_score(y_train, train_pred)},
            {"metric": "val_accuracy", "value": accuracy_score(y_val, val_pred)},
            {"metric": "n_labeled", "value": len(train_candidates)},
            {"metric": "n_human_complex", "value": len(human_complex)},
            {"metric": "n_rf_complex_candidates", "value": len(predicted_complex)},
            {"metric": "n_review_rows", "value": len(selected)},
        ],
    )

    direct_rf_complex = sum(row["model_top_label"] == "complex" for row in predicted_complex)
    print(f"matched labeled rows: {len(train_candidates)}")
    print(f"human complex rows: {len(human_complex)}")
    print(f"rf complex review candidates: {len(predicted_complex)}")
    print(f"rf candidates with complex as top label: {direct_rf_complex}")
    print(f"train_accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"val_accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"html: {args.out_dir / 'hpc_complex_reassignment_review.html'}")
    print(f"candidate csv: {args.out_dir / 'hpc_complex_reassignment_candidates.csv'}")


if __name__ == "__main__":
    main()
