#!/usr/bin/env python3
"""Compare old brain-wide peak predictions with current peak prediction ID tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from export_hpc_label_review import BASELINE_END_DAYS, TIME_END_DAYS, TIME_START_DAYS, trace_to_float


RAW_PREFIX = "raw_"


def row_key(row: dict | pd.Series, file_col: str) -> tuple[str, int, int]:
    return str(row[file_col]), int(row["trace_index"]), int(row["fiber_id"])


def load_old_matrix(path: Path, old_label: str, current_label: str) -> tuple[dict[tuple[str, int, int], dict], dict[tuple[str, int, int], np.ndarray]]:
    df = pd.read_csv(path)
    raw_cols = [col for col in df.columns if col.startswith(RAW_PREFIX)]
    rows: dict[tuple[str, int, int], dict] = {}
    traces: dict[tuple[str, int, int], np.ndarray] = {}
    for row in df.to_dict("records"):
        key = row_key(row, "file")
        rows[key] = {
            "source": "old",
            "old_label": old_label,
            "current_label": current_label,
            "source_file": row["file"],
            "trace_index": int(row["trace_index"]),
            "fiber_id": int(row["fiber_id"]),
            "key": row.get("key", ""),
            "region": row.get("region", ""),
            "major_group": row.get("major_group", ""),
            "old_prediction_confidence": row.get("prediction_confidence", ""),
            f"old_prob_{old_label}": row.get(f"prob_{old_label}", ""),
            "baseline": row.get("baseline", ""),
        }
        traces[key] = np.array([row[col] for col in raw_cols], dtype=np.float64)
    return rows, traces


def load_new_ids(path: Path, current_label: str) -> dict[tuple[str, int, int], dict]:
    df = pd.read_csv(path)
    rows: dict[tuple[str, int, int], dict] = {}
    for row in df.to_dict("records"):
        key = row_key(row, "source_file")
        rows[key] = {
            "source": "new",
            "current_label": current_label,
            "source_file": row["source_file"],
            "source_path": row.get("source_path", ""),
            "trace_index": int(row["trace_index"]),
            "fiber_id": int(row["fiber_id"]),
            "nd2_name": row.get("nd2_name", ""),
            "tile_name": row.get("tile_name", ""),
            "key": row.get("key", ""),
            "region": row.get("region", ""),
            "major_group": row.get("major_group", ""),
            "new_prediction_confidence": row.get("prediction_confidence", ""),
            "prob_dual_peak": row.get("prob_dual_peak", ""),
            "prob_second_peak": row.get("prob_second_peak", ""),
            "baseline": row.get("baseline_9p5_10d", ""),
            "shifted_amplitude": row.get("shifted_amplitude", ""),
        }
    return rows


def load_new_traces(raw_dir: Path, keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], np.ndarray]:
    by_file: dict[str, list[tuple[str, int, int]]] = {}
    for key in keys:
        by_file.setdefault(key[0], []).append(key)

    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    traces: dict[tuple[str, int, int], np.ndarray] = {}
    for file_name, file_keys in sorted(by_file.items()):
        npz_path = raw_dir / file_name
        if not npz_path.exists():
            continue
        needed = {(trace_index, fiber_id): key for key, trace_index, fiber_id in file_keys}
        with np.load(npz_path, allow_pickle=True) as data:
            recovered = data["recovered_cfos"]
            fiber_ids = data["fiber_ids"]
            for trace_index in sorted({trace_index for _, trace_index, _ in file_keys}):
                if trace_index < 0 or trace_index >= recovered.shape[0]:
                    continue
                fiber_id = int(fiber_ids[trace_index])
                key = needed.get((trace_index, fiber_id))
                if key is None:
                    continue
                raw_trace = trace_to_float(recovered[trace_index])
                baseline = float(np.nanmean(raw_trace[baseline_mask]))
                traces[key] = raw_trace - baseline
    return traces


def compact_svg(values: np.ndarray, width: int = 520, height: int = 190, max_points: int = 280) -> str:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float64)
    if values.size > max_points:
        idx = np.linspace(0, values.size - 1, max_points).astype(int)
        plot_values = values[idx]
        plot_times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, values.size)[idx]
    else:
        plot_values = values
        plot_times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, values.size)

    pad_left = 40
    pad_right = 12
    pad_top = 12
    pad_bottom = 26
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    finite = np.isfinite(plot_values)
    if finite.any():
        y_min = float(np.nanmin(plot_values))
        y_max = float(np.nanmax(plot_values))
        extent = max(abs(y_min), abs(y_max), 0.2)
    else:
        extent = 0.2
    y_min, y_max = -extent, extent

    xs = pad_left + (plot_times - TIME_START_DAYS) / (TIME_END_DAYS - TIME_START_DAYS) * plot_w
    denom = y_max - y_min if y_max > y_min else 1.0
    ys = pad_top + (y_max - plot_values) / denom * plot_h
    points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y, ok in zip(xs, ys, finite, strict=False) if ok
    )

    def x_pos(day: float) -> float:
        return pad_left + (day - TIME_START_DAYS) / (TIME_END_DAYS - TIME_START_DAYS) * plot_w

    def y_pos(value: float) -> float:
        return pad_top + (y_max - value) / denom * plot_h

    ticks = []
    for day in [9.5, 10, 12, 14, 15]:
        x = x_pos(day)
        ticks.append(
            f'<line x1="{x:.1f}" y1="{pad_top + plot_h}" x2="{x:.1f}" y2="{pad_top + plot_h + 4}" stroke="#64748b"/>'
            f'<text x="{x:.1f}" y="{height - 7}" text-anchor="middle" font-size="10" fill="#475569">{day:g}</text>'
        )
    zero_y = y_pos(0.0)
    x_10 = x_pos(10.0)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="100%" height="100%" fill="#fff"/>'
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#e2e8f0"/>'
        f'<rect x="{pad_left}" y="{pad_top}" width="{x_10 - pad_left:.1f}" height="{plot_h}" fill="#ecfdf5"/>'
        f'<line x1="{pad_left}" y1="{zero_y:.1f}" x2="{pad_left + plot_w}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-dasharray="4 4"/>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1.5"/>'
        f'{"".join(ticks)}'
        f'<text x="{pad_left - 5}" y="{y_pos(y_max) + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{y_max:.2f}</text>'
        f'<text x="{pad_left - 5}" y="{y_pos(y_min) + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{y_min:.2f}</text>'
        "</svg>"
    )


def display_float(value) -> str:
    try:
        if value == "" or value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "comparison",
        "current_label",
        "old_label",
        "source_file",
        "trace_index",
        "fiber_id",
        "key",
        "region",
        "major_group",
        "old_prediction_confidence",
        "new_prediction_confidence",
        "prob_dual_peak",
        "prob_second_peak",
        "baseline",
        "shifted_amplitude",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def section_html(title: str, rows: list[dict], traces: dict[tuple[str, int, int], np.ndarray]) -> str:
    body_rows = []
    for row in rows:
        key = (row["source_file"], int(row["trace_index"]), int(row["fiber_id"]))
        trace = traces.get(key, np.zeros(1000, dtype=np.float64))
        body_rows.append(
            f"""
            <tr>
              <td>{compact_svg(trace)}</td>
              <td>{html.escape(row.get("comparison", ""))}</td>
              <td>{html.escape(row.get("current_label", ""))}</td>
              <td>{html.escape(row.get("old_label", ""))}</td>
              <td>{html.escape(row["source_file"])}</td>
              <td>{row["trace_index"]}</td>
              <td>{row["fiber_id"]}</td>
              <td>{html.escape(str(row.get("key", "")))}</td>
              <td>{html.escape(str(row.get("region", "")))}</td>
              <td>{html.escape(str(row.get("major_group", "")))}</td>
              <td>{display_float(row.get("old_prediction_confidence", ""))}</td>
              <td>{display_float(row.get("new_prediction_confidence", ""))}</td>
              <td>{display_float(row.get("prob_dual_peak", ""))}</td>
              <td>{display_float(row.get("prob_second_peak", ""))}</td>
            </tr>
            """
        )
    return f"""
    <details open>
      <summary>{html.escape(title)} ({len(rows)})</summary>
      <table>
        <thead>
          <tr>
            <th>Trace</th><th>Comparison</th><th>Current Label</th><th>Old Label</th>
            <th>File</th><th>Trace Index</th><th>Fiber ID</th><th>Key</th><th>Region</th><th>Major Group</th>
            <th>Old Conf</th><th>New Conf</th><th>Prob Dual</th><th>Prob Second</th>
          </tr>
        </thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
    </details>
    """


def write_html(path: Path, sections: list[tuple[str, list[dict], dict]], summary: dict) -> None:
    section_docs = [section_html(title, rows, traces) for title, rows, traces in sections]
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Brain-wide peak prediction comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    .meta {{ color: #4b5563; margin-bottom: 14px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin: 12px 0 18px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; background: #f9fafb; }}
    .card .label {{ color: #4b5563; font-size: 12px; }}
    .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    details {{ margin: 16px 0; }}
    summary {{ cursor: pointer; font-weight: 700; font-size: 17px; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f3f4f6; z-index: 2; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Brain-wide dual/second peak prediction comparison</h1>
  <div class="meta">Old mapping: dual_peak -> dual_peak, recall_only -> second_peak. Matched by file + trace_index + fiber_id.</div>
  <div class="cards">
    {"".join(f'<div class="card"><div class="label">{html.escape(k)}</div><div class="value">{v}</div></div>' for k, v in summary.items())}
  </div>
  {"".join(section_docs)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def make_diff_rows(
    old_rows: dict[tuple[str, int, int], dict],
    new_rows: dict[tuple[str, int, int], dict],
    old_label: str,
    current_label: str,
) -> tuple[list[dict], list[dict]]:
    old_keys = set(old_rows)
    new_keys = set(new_rows)
    old_only = []
    for key in sorted(old_keys - new_keys):
        row = dict(old_rows[key])
        row["comparison"] = f"old_{old_label}_not_predicted_by_current_{current_label}"
        old_only.append(row)
    new_only = []
    for key in sorted(new_keys - old_keys):
        row = dict(new_rows[key])
        row["old_label"] = old_label
        row["comparison"] = f"current_{current_label}_not_in_old_{old_label}"
        new_only.append(row)
    return old_only, new_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-bundle",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--new-id-dir",
        type=Path,
        default=Path("outputs/reports/brainwide_peak_prediction_ids/20260505_dual_second_peak_ids"),
    )
    parser.add_argument("--brainwide-raw-dir", type=Path, default=Path("data/raw/brain_wide"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/brainwide_peak_prediction_comparison/brainwide_peak_comparison"),
    )
    args = parser.parse_args()

    old_dual, old_dual_traces = load_old_matrix(
        args.old_bundle / "dual_peak" / "dual_peak_trace_matrix.csv",
        "dual_peak",
        "dual_peak",
    )
    old_recall, old_recall_traces = load_old_matrix(
        args.old_bundle / "recall_only" / "recall_only_trace_matrix.csv",
        "recall_only",
        "second_peak",
    )
    new_dual = load_new_ids(args.new_id_dir / "brainwide_dual_peak_prediction_ids.csv", "dual_peak")
    new_second = load_new_ids(args.new_id_dir / "brainwide_second_peak_prediction_ids.csv", "second_peak")

    dual_old_only, dual_new_only = make_diff_rows(old_dual, new_dual, "dual_peak", "dual_peak")
    second_old_only, second_new_only = make_diff_rows(old_recall, new_second, "recall_only", "second_peak")

    new_only_keys = {
        (row["source_file"], int(row["trace_index"]), int(row["fiber_id"]))
        for row in dual_new_only + second_new_only
    }
    new_traces = load_new_traces(args.brainwide_raw_dir, new_only_keys)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "dual_peak_old_only_missing_by_current.csv", dual_old_only)
    write_csv(args.out_dir / "dual_peak_current_only_extra.csv", dual_new_only)
    write_csv(args.out_dir / "second_peak_old_recall_only_missing_by_current.csv", second_old_only)
    write_csv(args.out_dir / "second_peak_current_only_extra.csv", second_new_only)
    write_csv(
        args.out_dir / "all_peak_prediction_differences.csv",
        dual_old_only + dual_new_only + second_old_only + second_new_only,
    )

    summary = {
        "old dual_peak": len(old_dual),
        "current dual_peak": len(new_dual),
        "dual overlap": len(set(old_dual) & set(new_dual)),
        "dual old-only": len(dual_old_only),
        "dual current-only": len(dual_new_only),
        "old recall_only": len(old_recall),
        "current second_peak": len(new_second),
        "second overlap": len(set(old_recall) & set(new_second)),
        "second old-only": len(second_old_only),
        "second current-only": len(second_new_only),
    }
    sections = [
        ("Dual peak: old predicted but current model did not", dual_old_only, old_dual_traces),
        ("Dual peak: current model predicted but old did not", dual_new_only, new_traces),
        ("Second peak: old recall_only predicted but current second_peak did not", second_old_only, old_recall_traces),
        ("Second peak: current second_peak predicted but old recall_only did not", second_new_only, new_traces),
    ]
    write_html(args.out_dir / "brainwide_peak_prediction_comparison.html", sections, summary)
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"out: {args.out_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"html: {args.out_dir / 'brainwide_peak_prediction_comparison.html'}")


if __name__ == "__main__":
    main()
