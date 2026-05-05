#!/usr/bin/env python3
"""Predict all brain-wide traces and export per-label CSV bundles."""

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

from export_hpc_label_review import BASELINE_END_DAYS, TIME_END_DAYS, TIME_START_DAYS, collect_candidates, load_labels
from train_hpc_round_review import build_feature_matrix, candidate_key, read_reviewed_labels


EXCLUDED_MAJOR_GROUPS = {"", "Unassigned", "Unmapped"}


def load_region_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["key"].strip()
            region = row["region"].strip()
            if key and key != "Image_Tile" and region:
                mapping[key] = region
    return mapping


def load_major_group_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    df["region"] = df["region"].astype(str).str.strip()
    df["major_group"] = df["major_group"].astype(str).str.strip()
    df = df.loc[df["region"].ne("") & df["major_group"].ne("")]
    return dict(zip(df["region"], df["major_group"]))


def extract_tile_key(source_file: str) -> str:
    if source_file.startswith("OUTPUT_") and source_file.endswith("_profiles.npz"):
        return source_file[len("OUTPUT_") : -len("_profiles.npz")]
    return Path(source_file).stem


def count_raw_traces(raw_dir: Path) -> int:
    total = 0
    for path in raw_dir.glob("*.npz"):
        with np.load(path, allow_pickle=True) as data:
            total += int(data["recovered_cfos"].shape[0])
    return total


def build_training_set(raw_dir: Path, labels_csv: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
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
    return build_feature_matrix(train_candidates), np.array(y_labels, dtype=object), sorted(set(y_labels)), train_candidates


def fit_random_forest(x: np.ndarray, y: np.ndarray, seed: int, n_estimators: int, max_depth: int, min_samples_leaf: int):
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


def add_region_metadata(candidates: list[dict], region_map: dict[str, str], major_group_map: dict[str, str]) -> list[dict]:
    rows = []
    for row in candidates:
        tile_key = extract_tile_key(row["source_file"])
        region = region_map.get(tile_key, "")
        major_group = major_group_map.get(region, "Unmapped")
        if major_group in EXCLUDED_MAJOR_GROUPS:
            continue
        updated = dict(row)
        updated["key"] = tile_key
        updated["region"] = region
        updated["major_group"] = major_group
        rows.append(updated)
    return rows


def build_boundaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    major_rows = []
    region_rows = []
    cursor = 0
    for major_group in ordered_unique(row["major_group"] for row in rows):
        major_start = cursor
        major_subset = [row for row in rows if row["major_group"] == major_group]
        for region in ordered_unique(row["region"] for row in major_subset):
            region_start = cursor
            n_rows = sum(1 for row in major_subset if row["region"] == region)
            cursor += n_rows
            region_rows.append(
                {
                    "major_group": major_group,
                    "region": region,
                    "row_start": region_start,
                    "row_end": cursor - 1,
                    "n_rows": n_rows,
                }
            )
        major_rows.append(
            {
                "major_group": major_group,
                "row_start": major_start,
                "row_end": cursor - 1,
                "n_rows": len(major_subset),
            }
        )
    return major_rows, region_rows


def ordered_unique(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def write_dict_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_trace_matrix(path: Path, rows: list[dict], class_labels: list[str], label: str) -> None:
    raw_cols = [f"raw_{i:04d}" for i in range(1000)]
    fieldnames = [
        "major_group",
        "region",
        "key",
        "file",
        "trace_index",
        "fiber_id",
        "prediction_confidence",
        f"prob_{label}",
        "baseline",
    ] + raw_cols
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            trace = np.asarray(row["trace"], dtype=float)
            writer.writerow(
                [
                    row["major_group"],
                    row["region"],
                    row["key"],
                    row["source_file"],
                    row["trace_index"],
                    row["fiber_id"],
                    row["prediction_confidence"],
                    row.get(f"prob_{label}", ""),
                    row["baseline_9p5_10d"],
                ]
                + trace.tolist()
            )


def write_region_stats(path: Path, rows: list[dict], label_order: list[str]) -> None:
    records = []
    group_keys = sorted({(row["major_group"], row["region"]) for row in rows})
    for major_group, region in group_keys:
        subset = [row for row in rows if row["major_group"] == major_group and row["region"] == region]
        rec = {"major_group": major_group, "region": region, "n_total_traces": len(subset)}
        for label in label_order:
            n = sum(1 for row in subset if row["predicted_label"] == label)
            rec[label] = n
            rec[f"{label}_fraction"] = n / len(subset) if subset else 0.0
        records.append(rec)
    fieldnames = ["major_group", "region", "n_total_traces"] + label_order + [f"{label}_fraction" for label in label_order]
    write_dict_rows(path, records, fieldnames)


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
        default=Path("outputs/reports/brainwide_region_overlays/brainwide_random_forest_label_csv_bundles"),
    )
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    label_order = [label for label in load_labels(args.schema)]
    x_train, y_train, trained_labels, train_candidates = build_training_set(args.hpc_raw_dir, args.labels_csv)
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
    region_rows = add_region_metadata(
        brainwide_candidates,
        load_region_map(args.region_map),
        load_major_group_map(args.major_group_map),
    )
    x_brainwide = build_feature_matrix(region_rows)
    proba = model.predict_proba(x_brainwide)
    pred = model.predict(x_brainwide)
    model_labels = model.classes_.tolist()
    for i, row in enumerate(region_rows):
        row["predicted_label"] = str(pred[i])
        row["prediction_confidence"] = float(proba[i].max())
        for j, cls in enumerate(model_labels):
            row[f"prob_{cls}"] = float(proba[i, j])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "brainwide_random_forest_model.joblib")
    (args.out_dir / "random_forest_validation_report.txt").write_text(report)
    write_dict_rows(
        args.out_dir / "random_forest_metrics.csv",
        [{"metric": key, "value": value} for key, value in metrics.items()],
        ["metric", "value"],
    )
    write_region_stats(args.out_dir / "region_label_stats_wide.csv", region_rows, label_order)

    summary_rows = []
    for label in label_order:
        label_rows = [row for row in region_rows if row["predicted_label"] == label]
        label_rows.sort(
            key=lambda row: (
                row["major_group"],
                row["region"],
                -float(row.get(f"prob_{label}", 0.0)),
                -float(row["prediction_confidence"]),
                row["source_file"],
                int(row["trace_index"]),
            )
        )
        label_dir = args.out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        write_trace_matrix(label_dir / f"{label}_trace_matrix.csv", label_rows, model_labels, label)
        major_bounds, region_bounds = build_boundaries(label_rows)
        write_dict_rows(
            label_dir / f"{label}_major_group_row_boundaries.csv",
            major_bounds,
            ["major_group", "row_start", "row_end", "n_rows"],
        )
        write_dict_rows(
            label_dir / f"{label}_region_row_boundaries.csv",
            region_bounds,
            ["major_group", "region", "row_start", "row_end", "n_rows"],
        )
        (label_dir / "render_heatmap_from_csv.py").write_text(
            "#!/usr/bin/env python3\nprint('Use the project heatmap renderer with this CSV bundle.')\n"
        )
        metadata = {
            "predicted_label": label,
            "n_train": int(len(train_candidates)),
            "n_total_brainwide_before_filter": int(brainwide_total_before_filter),
            "n_total_brainwide_kept_after_filter": int(len(brainwide_candidates)),
            "n_total_brainwide_kept_after_region_filter": int(len(region_rows)),
            "n_predicted_label": int(len(label_rows)),
            "quality_filter": "max(abs(max(recovered_cfos)), abs(min(recovered_cfos))) >= 0.2 after baseline subtraction",
            "baseline_window_days": [TIME_START_DAYS, BASELINE_END_DAYS],
            "time_window_days": [TIME_START_DAYS, TIME_END_DAYS],
            "model": "RandomForestClassifier",
            "model_labels": model_labels,
        }
        (label_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        summary_rows.append({"label": label, "folder": str(label_dir), "n_predicted_label": int(len(label_rows))})

    write_dict_rows(args.out_dir / "summary.csv", summary_rows, ["label", "folder", "n_predicted_label"])
    root_metadata = {
        "n_train": int(len(train_candidates)),
        "training_label_counts": {label: int(np.sum(y_train == label)) for label in trained_labels},
        "n_total_brainwide_before_filter": int(brainwide_total_before_filter),
        "n_total_brainwide_kept_after_filter": int(len(brainwide_candidates)),
        "n_total_brainwide_kept_after_region_filter": int(len(region_rows)),
        "label_order": label_order,
        "model_labels": model_labels,
        "baseline_window_days": [TIME_START_DAYS, BASELINE_END_DAYS],
        "time_window_days": [TIME_START_DAYS, TIME_END_DAYS],
        "validation_metrics": metrics,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(root_metadata, indent=2))

    print(f"out: {args.out_dir}")
    print(f"n_train: {len(train_candidates)}")
    print(f"brainwide before filter: {brainwide_total_before_filter}")
    print(f"brainwide kept after filter: {len(brainwide_candidates)}")
    print(f"brainwide kept after region filter: {len(region_rows)}")
    for row in summary_rows:
        print(f"{row['label']}: {row['n_predicted_label']}")


if __name__ == "__main__":
    main()
