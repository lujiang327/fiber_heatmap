#!/usr/bin/env python3
"""Export per-file HPC motif prediction summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from export_hpc_label_review import collect_candidates, load_labels
from train_hpc_round_review import build_feature_matrix, candidate_key, read_reviewed_labels


def count_total_traces(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        return int(data["recovered_cfos"].shape[0])


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    raw_dir: Path,
    labels_csv: Path,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> tuple[Pipeline, list[dict], dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates, _ = collect_candidates(raw_dir)
    by_key = {candidate_key(row): row for row in candidates}
    reviewed_rows = read_reviewed_labels(labels_csv)

    train_candidates = []
    y_labels = []
    for row in reviewed_rows:
        candidate = by_key.get(candidate_key(row))
        if candidate is None:
            continue
        train_candidates.append(candidate)
        y_labels.append(row["reviewed_label"].strip())

    if not train_candidates:
        raise RuntimeError("No reviewed labels matched filtered HPC candidates.")

    x = build_feature_matrix(train_candidates)
    y = np.array(y_labels, dtype=object)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    train_label_by_key = dict(zip((candidate_key(row) for row in train_candidates), y_labels, strict=False))
    return model, candidates, train_label_by_key, x_train, x_val, y_train, y_val


def build_prediction_tables(raw_dir: Path, candidates: list[dict], predictions: np.ndarray, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_by_key = {candidate_key(row): str(label) for row, label in zip(candidates, predictions, strict=False)}
    kept_by_file: dict[str, int] = {}
    counts_by_file: dict[str, dict[str, int]] = {}
    for row in candidates:
        file_name = row["source_file"]
        kept_by_file[file_name] = kept_by_file.get(file_name, 0) + 1
        predicted = pred_by_key[candidate_key(row)]
        counts_by_file.setdefault(file_name, {})
        counts_by_file[file_name][predicted] = counts_by_file[file_name].get(predicted, 0) + 1

    per_file_wide = []
    per_file_long = []
    for npz_path in sorted(raw_dir.glob("*.npz")):
        file_name = npz_path.name
        n_total_before_filter = count_total_traces(npz_path)
        n_kept_after_filter = int(kept_by_file.get(file_name, 0))
        n_filtered_out = n_total_before_filter - n_kept_after_filter
        filtered_fraction = (n_filtered_out / n_total_before_filter) if n_total_before_filter else 0.0

        row = {
            "file": file_name,
            "n_total_before_filter": n_total_before_filter,
            "n_kept_after_filter": n_kept_after_filter,
            "n_filtered_out": n_filtered_out,
            "filtered_fraction": filtered_fraction,
        }
        for label in labels:
            n_label = int(counts_by_file.get(file_name, {}).get(label, 0))
            fraction = (n_label / n_kept_after_filter) if n_kept_after_filter else 0.0
            row[label] = n_label
            row[f"{label}_fraction"] = fraction
            per_file_long.append(
                {
                    "file": file_name,
                    "label": label,
                    "n_total_before_filter": n_total_before_filter,
                    "n_kept_after_filter": n_kept_after_filter,
                    "n_filtered_out": n_filtered_out,
                    "filtered_fraction": filtered_fraction,
                    "n_label_traces": n_label,
                    "label_fraction_in_file": fraction,
                }
            )
        per_file_wide.append(row)

    wide_df = pd.DataFrame(per_file_wide).sort_values("file").reset_index(drop=True)
    long_df = pd.DataFrame(per_file_long).sort_values(["file", "label"]).reset_index(drop=True)

    overall = {
        "scope": "all_hpc_files",
        "n_files": int(len(wide_df)),
        "n_total_before_filter": int(wide_df["n_total_before_filter"].sum()),
        "n_kept_after_filter": int(wide_df["n_kept_after_filter"].sum()),
        "n_filtered_out": int(wide_df["n_filtered_out"].sum()),
    }
    overall["filtered_fraction"] = (
        overall["n_filtered_out"] / overall["n_total_before_filter"]
        if overall["n_total_before_filter"]
        else 0.0
    )
    for label in labels:
        n_label = int(wide_df[label].sum())
        overall[label] = n_label
        overall[f"{label}_fraction"] = (
            n_label / overall["n_kept_after_filter"] if overall["n_kept_after_filter"] else 0.0
        )
    overall_df = pd.DataFrame([overall])
    return wide_df, long_df, overall_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/reports/hpc_file_predictions/<timestamp>_hpc_random_forest_prediction_tables.",
    )
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = Path("outputs/reports/hpc_file_predictions") / f"{timestamp}_hpc_random_forest_prediction_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.schema)
    model, candidates, train_label_by_key, x_train, x_val, y_train, y_val = train_model(
        args.raw_dir,
        args.labels_csv,
        args.seed,
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
    )
    x_all = build_feature_matrix(candidates)
    predictions = model.predict(x_all)
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    wide_df, long_df, overall_df = build_prediction_tables(args.raw_dir, candidates, predictions, labels)

    wide_df.to_csv(out_dir / "hpc_file_label_stats_wide.csv", index=False)
    long_df.to_csv(out_dir / "hpc_file_label_stats_long.csv", index=False)
    overall_df.to_csv(out_dir / "hpc_overall_label_summary.csv", index=False)
    for _, row in wide_df.iterrows():
        pd.DataFrame([row.to_dict()]).to_csv(out_dir / f"{Path(row['file']).stem}__label_stats_wide.csv", index=False)

    joblib.dump(model, out_dir / "hpc_random_forest_model.joblib")
    (out_dir / "hpc_random_forest_validation_report.txt").write_text(
        classification_report(y_val, val_pred, labels=labels, zero_division=0),
        encoding="utf-8",
    )
    write_metrics_csv(
        out_dir / "hpc_random_forest_metrics.csv",
        [
            {"metric": "train_accuracy", "value": accuracy_score(y_train, train_pred)},
            {"metric": "val_accuracy", "value": accuracy_score(y_val, val_pred)},
            {"metric": "n_labeled", "value": len(train_label_by_key)},
            {"metric": "n_train", "value": len(y_train)},
            {"metric": "n_val", "value": len(y_val)},
            {"metric": "n_prediction_candidates", "value": len(candidates)},
        ],
    )

    metadata = {
        "label_classes": labels,
        "model": "RandomForestClassifier",
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced_subsample",
            "random_state": args.seed,
        },
        "n_train_labels_matched": int(len(train_label_by_key)),
        "n_hpc_files": int(len(wide_df)),
        "quality_filter": "baseline-subtracted max(abs(max(trace)), abs(min(trace))) >= 0.2",
        "filtered_traces_reassigned_to_no_response": False,
        "train_data_scope": "merged manually reviewed HPC labels",
        "prediction_scope": "filtered traces from all HPC files",
        "raw_dir": str(args.raw_dir),
        "labels_csv": str(args.labels_csv),
        "schema": str(args.schema),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(out_dir)
    print(wide_df[["file", "n_total_before_filter", "n_kept_after_filter", "n_filtered_out"]].to_string(index=False))
    print()
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
