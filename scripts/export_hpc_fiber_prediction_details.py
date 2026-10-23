#!/usr/bin/env python3
"""Export one-row-per-fiber HPC prediction details from a saved prediction run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np

from export_hpc_label_review import BASELINE_END_DAYS, TIME_END_DAYS, TIME_START_DAYS, trace_to_float
from train_hpc_round_review import build_feature_matrix


def load_reviewed_labels(path: Path) -> dict[tuple[str, int], str]:
    if not path.exists():
        return {}
    out: dict[tuple[str, int], str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            source_file = row.get("source_file", "").strip()
            trace_index = row.get("trace_index", "").strip()
            label = (row.get("reviewed_label") or row.get("label") or "").strip()
            if source_file and trace_index and label:
                out[(source_file, int(trace_index))] = label
    return out


def collect_all_hpc_rows(raw_dir: Path, min_amplitude: float) -> tuple[list[dict], list[dict]]:
    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    all_rows: list[dict] = []
    kept_rows: list[dict] = []

    for npz_path in sorted(raw_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            recovered = data["recovered_cfos"]
            fiber_ids = data["fiber_ids"]
            nd2_names = data["nd2_name"]
            tile_names = data["tile_name"]

            for trace_index in range(recovered.shape[0]):
                raw_trace = trace_to_float(recovered[trace_index])
                valid_trace = raw_trace.size == times.size and np.isfinite(raw_trace).any()
                baseline = ""
                shifted_max = ""
                shifted_min = ""
                amplitude = ""
                passed_filter = False
                shifted = None
                if valid_trace:
                    baseline_float = float(np.nanmean(raw_trace[baseline_mask]))
                    shifted = raw_trace - baseline_float
                    shifted_max_float = float(np.nanmax(shifted))
                    shifted_min_float = float(np.nanmin(shifted))
                    amplitude_float = max(abs(shifted_max_float), abs(shifted_min_float))
                    baseline = baseline_float
                    shifted_max = shifted_max_float
                    shifted_min = shifted_min_float
                    amplitude = amplitude_float
                    passed_filter = math.isfinite(amplitude_float) and amplitude_float >= min_amplitude

                row = {
                    "source_file": npz_path.name,
                    "source_path": str(npz_path),
                    "trace_index": trace_index,
                    "fiber_id": int(fiber_ids[trace_index]),
                    "nd2_name": str(nd2_names[trace_index]),
                    "tile_name": str(tile_names[trace_index]),
                    "passed_filter": passed_filter,
                    "baseline_9p5_10d": baseline,
                    "shifted_max": shifted_max,
                    "shifted_min": shifted_min,
                    "shifted_amplitude": amplitude,
                }
                all_rows.append(row)
                if passed_filter and shifted is not None:
                    kept = dict(row)
                    kept["trace"] = shifted
                    kept_rows.append(kept)
    return all_rows, kept_rows


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--min-amplitude", type=float, default=0.2)
    args = parser.parse_args()

    model_path = args.run_dir / "hpc_random_forest_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    model = joblib.load(model_path)
    model_labels = list(model.classes_)

    all_rows, kept_rows = collect_all_hpc_rows(args.raw_dir, args.min_amplitude)
    x = build_feature_matrix(kept_rows)
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    reviewed_by_key = load_reviewed_labels(args.labels_csv)

    kept_detail_rows = []
    predicted_by_key: dict[tuple[str, int], dict] = {}
    for row, predicted_label, prob in zip(kept_rows, predictions, probabilities, strict=False):
        detail = {key: value for key, value in row.items() if key != "trace"}
        detail["predicted_label"] = str(predicted_label)
        detail["prediction_confidence"] = float(np.max(prob))
        detail["reviewed_label"] = reviewed_by_key.get((detail["source_file"], int(detail["trace_index"])), "")
        for label, value in zip(model_labels, prob, strict=False):
            detail[f"prob_{label}"] = float(value)
        kept_detail_rows.append(detail)
        predicted_by_key[(detail["source_file"], int(detail["trace_index"]))] = detail

    all_detail_rows = []
    for row in all_rows:
        key = (row["source_file"], int(row["trace_index"]))
        if key in predicted_by_key:
            all_detail_rows.append(predicted_by_key[key])
            continue
        detail = dict(row)
        detail["predicted_label"] = "filtered_out"
        detail["prediction_confidence"] = ""
        detail["reviewed_label"] = reviewed_by_key.get(key, "")
        for label in model_labels:
            detail[f"prob_{label}"] = ""
        all_detail_rows.append(detail)

    base_fields = [
        "source_file",
        "source_path",
        "trace_index",
        "fiber_id",
        "nd2_name",
        "tile_name",
        "passed_filter",
        "predicted_label",
        "prediction_confidence",
        "reviewed_label",
    ]
    prob_fields = [f"prob_{label}" for label in model_labels]
    metric_fields = ["baseline_9p5_10d", "shifted_max", "shifted_min", "shifted_amplitude"]
    fieldnames = base_fields + prob_fields + metric_fields

    kept_path = args.run_dir / "hpc_fiber_prediction_details.csv"
    all_path = args.run_dir / "hpc_fiber_prediction_details_with_filtered.csv"
    write_rows(kept_path, kept_detail_rows, fieldnames)
    write_rows(all_path, all_detail_rows, fieldnames)

    summary = {
        "run_dir": str(args.run_dir),
        "model": str(model_path),
        "raw_dir": str(args.raw_dir),
        "labels_csv": str(args.labels_csv),
        "n_total_before_filter": len(all_detail_rows),
        "n_kept_after_filter": len(kept_detail_rows),
        "n_filtered_out": len(all_detail_rows) - len(kept_detail_rows),
        "model_labels": model_labels,
        "kept_details_csv": kept_path.name,
        "all_details_csv": all_path.name,
    }
    (args.run_dir / "hpc_fiber_prediction_details_metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
