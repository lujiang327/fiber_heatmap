#!/usr/bin/env python3
"""Merge low-amplitude filtered brain-wide traces into no_response in an existing bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_brainwide_prediction_bundles import (
    EXCLUDED_MAJOR_GROUPS,
    build_boundaries,
    count_raw_traces,
    extract_tile_key,
    load_major_group_map,
    load_region_map,
    write_dict_rows,
    write_trace_matrix,
)
from export_hpc_label_review import BASELINE_END_DAYS, TIME_END_DAYS, TIME_START_DAYS, trace_to_float


def iter_filtered_rows(raw_dir: Path, region_map: dict[str, str], major_group_map: dict[str, str], threshold: float):
    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    for npz_path in sorted(raw_dir.glob("*.npz")):
        tile_key = extract_tile_key(npz_path.name)
        region = region_map.get(tile_key, "")
        major_group = major_group_map.get(region, "Unmapped")
        if major_group in EXCLUDED_MAJOR_GROUPS:
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            recovered = data["recovered_cfos"]
            fiber_ids = data["fiber_ids"]
            nd2_names = data["nd2_name"]
            tile_names = data["tile_name"]
            for trace_index in range(recovered.shape[0]):
                raw_trace = trace_to_float(recovered[trace_index])
                if raw_trace.size != times.size or not np.isfinite(raw_trace).any():
                    continue
                baseline = float(np.nanmean(raw_trace[baseline_mask]))
                shifted = raw_trace - baseline
                max_value = float(np.nanmax(shifted))
                min_value = float(np.nanmin(shifted))
                amplitude = max(abs(max_value), abs(min_value))
                if not np.isfinite(amplitude) or amplitude >= threshold:
                    continue
                yield {
                    "major_group": major_group,
                    "region": region,
                    "key": tile_key,
                    "source_file": npz_path.name,
                    "source_path": str(npz_path),
                    "trace_index": int(trace_index),
                    "fiber_id": int(fiber_ids[trace_index]),
                    "nd2_name": str(nd2_names[trace_index]),
                    "tile_name": str(tile_names[trace_index]),
                    "baseline_9p5_10d": baseline,
                    "shifted_max": max_value,
                    "shifted_min": min_value,
                    "shifted_amplitude": amplitude,
                    "predicted_label": "no_response",
                    "prediction_confidence": 1.0,
                    "prob_no_response": 1.0,
                    "trace": shifted,
                }


def read_no_response_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        raw_cols = [col for col in reader.fieldnames or [] if col.startswith("raw_")]
        for row in reader:
            trace = np.array([float(row[col]) for col in raw_cols], dtype=float)
            rows.append(
                {
                    "major_group": row["major_group"],
                    "region": row["region"],
                    "key": row["key"],
                    "source_file": row["file"],
                    "source_path": "",
                    "trace_index": int(row["trace_index"]),
                    "fiber_id": int(row["fiber_id"]),
                    "baseline_9p5_10d": float(row["baseline"]),
                    "predicted_label": "no_response",
                    "prediction_confidence": float(row["prediction_confidence"]),
                    "prob_no_response": float(row["prob_no_response"]),
                    "trace": trace,
                }
            )
    return rows


def update_summary(summary_path: Path, no_response_count: int) -> None:
    rows = list(csv.DictReader(summary_path.open(newline="")))
    for row in rows:
        if row["label"] == "no_response":
            row["n_predicted_label"] = str(no_response_count)
            break
    write_dict_rows(summary_path, rows, ["label", "folder", "n_predicted_label"])


def update_region_stats(path: Path, filtered_rows: list[dict], label_order: list[str]) -> None:
    df = pd.read_csv(path)
    filtered_counts = (
        pd.DataFrame(
            [{"major_group": row["major_group"], "region": row["region"]} for row in filtered_rows]
        )
        .groupby(["major_group", "region"])
        .size()
        .reset_index(name="n_filtered_no_response")
    )
    if filtered_counts.empty:
        return
    df = df.merge(filtered_counts, on=["major_group", "region"], how="outer")
    df["n_filtered_no_response"] = df["n_filtered_no_response"].fillna(0).astype(int)
    for col in ["n_total_traces", "no_response"]:
        df[col] = df[col].fillna(0).astype(int)
    df["n_total_traces"] += df["n_filtered_no_response"]
    df["no_response"] += df["n_filtered_no_response"]
    for label in label_order:
        if label not in df.columns:
            df[label] = 0
        df[label] = df[label].fillna(0).astype(int)
        df[f"{label}_fraction"] = np.where(df["n_total_traces"] > 0, df[label] / df["n_total_traces"], 0.0)
    df = df.drop(columns=["n_filtered_no_response"])
    fieldnames = ["major_group", "region", "n_total_traces"] + label_order + [f"{label}_fraction" for label in label_order]
    df[fieldnames].sort_values(["major_group", "region"]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--brainwide-raw-dir", type=Path, default=Path("data/raw/brain_wide"))
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
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    label_order = [row["label"] for row in csv.DictReader((args.bundle_dir / "summary.csv").open(newline=""))]
    no_response_dir = args.bundle_dir / "no_response"
    no_response_matrix = no_response_dir / "no_response_trace_matrix.csv"
    existing_rows = read_no_response_rows(no_response_matrix)
    filtered_rows = list(
        iter_filtered_rows(
            args.brainwide_raw_dir,
            load_region_map(args.region_map),
            load_major_group_map(args.major_group_map),
            args.threshold,
        )
    )
    merged_rows = existing_rows + filtered_rows
    merged_rows.sort(
        key=lambda row: (
            row["major_group"],
            row["region"],
            -float(row.get("prob_no_response", 0.0)),
            -float(row["prediction_confidence"]),
            row["source_file"],
            int(row["trace_index"]),
        )
    )
    write_trace_matrix(no_response_matrix, merged_rows, ["no_response"], "no_response")
    major_bounds, region_bounds = build_boundaries(merged_rows)
    write_dict_rows(
        no_response_dir / "no_response_major_group_row_boundaries.csv",
        major_bounds,
        ["major_group", "row_start", "row_end", "n_rows"],
    )
    write_dict_rows(
        no_response_dir / "no_response_region_row_boundaries.csv",
        region_bounds,
        ["major_group", "region", "row_start", "row_end", "n_rows"],
    )
    update_summary(args.bundle_dir / "summary.csv", len(merged_rows))
    update_region_stats(args.bundle_dir / "region_label_stats_wide.csv", filtered_rows, label_order)

    bundle_metadata_path = args.bundle_dir / "run_metadata.json"
    metadata = json.loads(bundle_metadata_path.read_text())
    metadata["filtered_out_reassigned_to"] = "no_response"
    metadata["n_filtered_out_merged_to_no_response"] = len(filtered_rows)
    metadata["n_total_brainwide_before_filter"] = count_raw_traces(args.brainwide_raw_dir)
    metadata["n_total_brainwide_after_no_response_merge"] = metadata.get(
        "n_total_brainwide_kept_after_region_filter", 0
    ) + len(filtered_rows)
    bundle_metadata_path.write_text(json.dumps(metadata, indent=2))

    no_response_metadata_path = no_response_dir / "run_metadata.json"
    no_response_metadata = json.loads(no_response_metadata_path.read_text())
    no_response_metadata["n_predicted_label"] = len(merged_rows)
    no_response_metadata["n_filtered_out_merged_to_no_response"] = len(filtered_rows)
    no_response_metadata["filtered_out_reassigned_to"] = "no_response"
    no_response_metadata_path.write_text(json.dumps(no_response_metadata, indent=2))

    print(f"existing no_response rows: {len(existing_rows)}")
    print(f"filtered rows merged to no_response: {len(filtered_rows)}")
    print(f"merged no_response rows: {len(merged_rows)}")
    print(f"updated bundle: {args.bundle_dir}")


if __name__ == "__main__":
    main()
