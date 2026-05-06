#!/usr/bin/env python3
"""Predict brain-wide traces and export ID tables for selected peak labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from export_brainwide_prediction_bundles import extract_tile_key, load_major_group_map, load_region_map
from export_hpc_label_review import collect_candidates, load_labels
from train_hpc_round_review import build_feature_matrix, candidate_key, read_reviewed_labels


def count_raw_traces(raw_dir: Path) -> int:
    total = 0
    for path in raw_dir.glob("*.npz"):
        with np.load(path, allow_pickle=True) as data:
            total += int(data["recovered_cfos"].shape[0])
    return total


def build_training_set(raw_dir: Path, labels_csv: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    reviewed_rows = read_reviewed_labels(labels_csv)
    candidates, _ = collect_candidates(raw_dir)
    by_key = {candidate_key(row): row for row in candidates}
    train_candidates = []
    y_labels = []
    for row in reviewed_rows:
        candidate = by_key.get(candidate_key(row))
        if candidate is None:
            continue
        train_candidates.append(candidate)
        y_labels.append(row["reviewed_label"].strip())
    if not train_candidates:
        raise SystemExit("No labels matched training candidates.")
    return build_feature_matrix(train_candidates), np.array(y_labels, dtype=object), train_candidates


def fit_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> tuple[Pipeline, dict, str]:
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
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }
    report = classification_report(y_val, val_pred, labels=sorted(set(y.tolist())), zero_division=0)
    return model, metrics, report


def add_optional_region_metadata(rows: list[dict], region_map_path: Path, major_group_map_path: Path) -> None:
    region_map = load_region_map(region_map_path)
    major_group_map = load_major_group_map(major_group_map_path)
    for row in rows:
        key = extract_tile_key(row["source_file"])
        region = region_map.get(key, "")
        row["key"] = key
        row["region"] = region
        row["major_group"] = major_group_map.get(region, "Unmapped")


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc-raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--brainwide-raw-dir", type=Path, default=Path("data/raw/brain_wide"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument(
        "--region-map",
        type=Path,
        default=Path("/Users/louis/Desktop/lab/yj/fiber_umap/data/raw/brain wide config/final_combined_region.csv"),
    )
    parser.add_argument(
        "--major-group-map",
        type=Path,
        default=Path(
            "/Users/louis/Desktop/lab/yj/fiber_umap/data/raw/brain wide config/region_fiber_major_group_with_info_ranked.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/brainwide_peak_prediction_ids/brainwide_peak_prediction_ids"),
    )
    parser.add_argument("--target-labels", nargs="*", default=["dual_peak", "second_peak"])
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    schema_labels = load_labels(args.schema)
    unknown_targets = [label for label in args.target_labels if label not in schema_labels]
    if unknown_targets:
        raise SystemExit(f"Unknown target labels: {', '.join(unknown_targets)}")

    x_train, y_train, train_candidates = build_training_set(args.hpc_raw_dir, args.labels_csv)
    model, metrics, report = fit_random_forest(
        x_train,
        y_train,
        args.seed,
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
    )

    brainwide_total_before_filter = count_raw_traces(args.brainwide_raw_dir)
    brainwide_candidates, _ = collect_candidates(args.brainwide_raw_dir)
    add_optional_region_metadata(brainwide_candidates, args.region_map, args.major_group_map)

    x_brainwide = build_feature_matrix(brainwide_candidates)
    proba = model.predict_proba(x_brainwide)
    pred = model.predict(x_brainwide)
    model_labels = model.classes_.tolist()
    label_to_idx = {label: idx for idx, label in enumerate(model_labels)}

    detail_rows = []
    for i, row in enumerate(brainwide_candidates):
        predicted_label = str(pred[i])
        if predicted_label not in args.target_labels:
            continue
        out = {
            "predicted_label": predicted_label,
            "source_file": row["source_file"],
            "source_path": row["source_path"],
            "trace_index": int(row["trace_index"]),
            "fiber_id": int(row["fiber_id"]),
            "nd2_name": row["nd2_name"],
            "tile_name": row["tile_name"],
            "key": row.get("key", ""),
            "region": row.get("region", ""),
            "major_group": row.get("major_group", ""),
            "prediction_confidence": float(proba[i].max()),
            "baseline_9p5_10d": float(row["baseline_9p5_10d"]),
            "shifted_max": float(row["shifted_max"]),
            "shifted_min": float(row["shifted_min"]),
            "shifted_amplitude": float(row["shifted_amplitude"]),
        }
        for label in args.target_labels:
            out[f"prob_{label}"] = float(proba[i, label_to_idx[label]]) if label in label_to_idx else ""
        detail_rows.append(out)

    detail_rows.sort(
        key=lambda row: (
            row["predicted_label"],
            row["source_file"],
            int(row["trace_index"]),
            int(row["fiber_id"]),
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "predicted_label",
        "source_file",
        "source_path",
        "trace_index",
        "fiber_id",
        "nd2_name",
        "tile_name",
        "key",
        "region",
        "major_group",
        "prediction_confidence",
    ] + [f"prob_{label}" for label in args.target_labels] + [
        "baseline_9p5_10d",
        "shifted_max",
        "shifted_min",
        "shifted_amplitude",
    ]
    write_rows(args.out_dir / "brainwide_dual_second_peak_prediction_ids.csv", detail_rows, fieldnames)
    for label in args.target_labels:
        write_rows(
            args.out_dir / f"brainwide_{label}_prediction_ids.csv",
            [row for row in detail_rows if row["predicted_label"] == label],
            fieldnames,
        )

    summary_rows = []
    for label in schema_labels:
        summary_rows.append(
            {
                "label": label,
                "n_predicted_after_amplitude_filter": int(np.sum(pred == label)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "brainwide_prediction_summary.csv", index=False)
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        args.out_dir / "random_forest_metrics.csv",
        index=False,
    )
    (args.out_dir / "random_forest_validation_report.txt").write_text(report, encoding="utf-8")
    joblib.dump(model, args.out_dir / "brainwide_peak_random_forest_model.joblib")
    metadata = {
        "target_labels": args.target_labels,
        "model_labels": model_labels,
        "training_label_counts": {label: int(np.sum(y_train == label)) for label in sorted(set(y_train.tolist()))},
        "n_train_labels_matched": int(len(train_candidates)),
        "n_total_brainwide_before_filter": int(brainwide_total_before_filter),
        "n_total_brainwide_kept_after_amplitude_filter": int(len(brainwide_candidates)),
        "quality_filter": "baseline-subtracted max(abs(max(trace)), abs(min(trace))) >= 0.2",
        "region_metadata_note": "Region and major_group are included when tile keys are present in the config maps; unmapped rows are retained.",
        "raw_dir": str(args.brainwide_raw_dir),
        "labels_csv": str(args.labels_csv),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"out: {args.out_dir}")
    print(f"n_train: {len(train_candidates)}")
    print(f"brainwide before filter: {brainwide_total_before_filter}")
    print(f"brainwide kept after amplitude filter: {len(brainwide_candidates)}")
    for label in args.target_labels:
        print(f"{label}: {sum(row['predicted_label'] == label for row in detail_rows)}")
    print(f"combined csv: {args.out_dir / 'brainwide_dual_second_peak_prediction_ids.csv'}")


if __name__ == "__main__":
    main()
