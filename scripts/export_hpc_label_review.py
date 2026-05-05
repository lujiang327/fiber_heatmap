#!/usr/bin/env python3
"""Export a self-contained HTML review set for manual HPC trace labels."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from random import Random

import numpy as np


TIME_START_DAYS = 9.5
TIME_END_DAYS = 15.0
BASELINE_END_DAYS = 10.0


def load_labels(path: Path) -> list[str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row.get("display_order") or 0))
    return [row["label"] for row in rows if row.get("label")]


def trace_to_float(row: np.ndarray) -> np.ndarray:
    out = np.asarray(row, dtype=np.float64)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def make_svg(times: np.ndarray, values: np.ndarray, width: int = 720, height: int = 260) -> str:
    pad_left = 48
    pad_right = 16
    pad_top = 14
    pad_bottom = 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    finite = np.isfinite(values)
    if not finite.any():
        y_min, y_max = -1.0, 1.0
        points = ""
    else:
        y_min = float(np.nanmin(values))
        y_max = float(np.nanmax(values))
        extent = max(abs(y_min), abs(y_max), 0.2)
        y_min, y_max = -extent, extent
        xs = pad_left + (times - TIME_START_DAYS) / (TIME_END_DAYS - TIME_START_DAYS) * plot_w
        ys = pad_top + (y_max - values) / (y_max - y_min) * plot_h
        points = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y, ok in zip(xs, ys, finite, strict=False)
            if ok
        )

    def x_pos(day: float) -> float:
        return pad_left + (day - TIME_START_DAYS) / (TIME_END_DAYS - TIME_START_DAYS) * plot_w

    def y_pos(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * plot_h

    y_zero = y_pos(0.0)
    x_10 = x_pos(10.0)
    ticks = []
    for day in [9.5, 10, 11, 12, 13, 14, 15]:
        x = x_pos(day)
        label = f"{day:g}"
        ticks.append(
            f'<line x1="{x:.1f}" y1="{pad_top + plot_h}" x2="{x:.1f}" y2="{pad_top + plot_h + 5}" stroke="#555"/>'
            f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle" font-size="11" fill="#333">{label}</text>'
        )

    y_ticks = []
    for value in [y_min, 0.0, y_max]:
        y = y_pos(value)
        y_ticks.append(
            f'<line x1="{pad_left - 5}" y1="{y:.1f}" x2="{pad_left}" y2="{y:.1f}" stroke="#555"/>'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#333">{value:.2f}</text>'
        )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="100%" height="100%" fill="#fff"/>'
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>'
        f'<rect x="{pad_left}" y="{pad_top}" width="{x_10 - pad_left:.1f}" height="{plot_h}" fill="#ecfdf5"/>'
        f'<line x1="{pad_left}" y1="{y_zero:.1f}" x2="{pad_left + plot_w}" y2="{y_zero:.1f}" stroke="#94a3b8" stroke-dasharray="4 4"/>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1.6"/>'
        f'{"".join(ticks)}{"".join(y_ticks)}'
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 1}" text-anchor="middle" font-size="11" fill="#333">days</text>'
        f'</svg>'
    )


def collect_candidates(raw_dir: Path) -> tuple[list[dict], np.ndarray]:
    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    candidates: list[dict] = []

    for npz_path in sorted(raw_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
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
            if not math.isfinite(amplitude) or amplitude < 0.2:
                continue
            candidates.append(
                {
                    "source_file": npz_path.name,
                    "source_path": str(npz_path),
                    "trace_index": trace_index,
                    "fiber_id": int(fiber_ids[trace_index]),
                    "nd2_name": str(nd2_names[trace_index]),
                    "tile_name": str(tile_names[trace_index]),
                    "baseline_9p5_10d": baseline,
                    "shifted_max": max_value,
                    "shifted_min": min_value,
                    "shifted_amplitude": amplitude,
                    "trace": shifted,
                }
            )

    return candidates, times


def write_candidates_csv(path: Path, selected: list[dict]) -> None:
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
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row[key] for key in fieldnames})


def write_html(path: Path, selected: list[dict], labels: list[str], times: np.ndarray) -> None:
    label_buttons = "".join(
        f'<button type="button" class="label-btn" data-label="{html.escape(label)}">{html.escape(label)}</button>'
        for label in labels
    )
    rows = []
    review_rows = []
    for review_index, row in enumerate(selected, start=1):
        svg = make_svg(times, row["trace"])
        row_id = f"r{review_index}"
        review_rows.append(
            {
                "review_index": review_index,
                "source_file": row["source_file"],
                "source_path": row["source_path"],
                "trace_index": row["trace_index"],
                "fiber_id": row["fiber_id"],
                "nd2_name": row["nd2_name"],
                "tile_name": row["tile_name"],
                "baseline_9p5_10d": row["baseline_9p5_10d"],
                "shifted_max": row["shifted_max"],
                "shifted_min": row["shifted_min"],
                "shifted_amplitude": row["shifted_amplitude"],
            }
        )
        rows.append(
            f"""
            <tr data-row-id="{row_id}">
              <td>{svg}</td>
              <td>{html.escape(row["source_file"])}</td>
              <td>{row["trace_index"]}</td>
              <td>{row["fiber_id"]}</td>
              <td>{row["baseline_9p5_10d"]:.4f}</td>
              <td>
                <div class="label-picker">
                  <div class="label-btn-row">{label_buttons}</div>
                  <button type="button" class="clear-btn">Clear</button>
                  <input type="hidden" class="review-label" value="">
                </div>
              </td>
              <td><textarea class="review-notes" rows="3"></textarea></td>
            </tr>
            """
        )

    review_json = json.dumps(review_rows)
    labels_json = json.dumps(labels)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HPC random trace label review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ margin: 0 0 16px; color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f5f5f5; z-index: 2; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    textarea {{ width: 220px; }}
    .toolbar {{ position: fixed; right: 16px; bottom: 16px; z-index: 1000; display:flex; flex-direction:column; gap:10px; align-items:stretch; max-width: 340px; background: rgba(255,255,255,0.96); border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.12); }}
    .toolbar button {{ width: 100%; padding:8px 12px; }}
    .toolbar .status {{ font-size: 12px; color: #374151; }}
    .label-picker {{ display:flex; flex-direction:column; gap:6px; min-width:360px; }}
    .label-btn-row {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .label-btn {{ padding:4px 8px; border:1px solid #cbd5e1; background:#fff; border-radius:6px; cursor:pointer; font-size:12px; }}
    .label-btn.active {{ background:#dbeafe; border-color:#2563eb; color:#1d4ed8; font-weight:600; }}
    .clear-btn {{ align-self:flex-start; padding:3px 8px; border:1px solid #d1d5db; background:#f9fafb; border-radius:6px; cursor:pointer; font-size:12px; }}
  </style>
</head>
<body>
  <h1>HPC random trace label review</h1>
  <p class="meta">100 traces sampled after baseline correction and amplitude filtering. Baseline: recovered_cfos mean from 9.5 to 10 days. Filter: max(abs(max), abs(min)) &gt;= 0.2 on shifted trace.</p>
  <div class="toolbar">
    <button type="button" onclick="downloadReviewCsv()">Download Reviewed CSV</button>
    <div class="status" id="status"></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Trace Plot</th>
        <th>File</th>
        <th>Trace Index</th>
        <th>Fiber ID</th>
        <th>Baseline</th>
        <th>Reviewed Label</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <script>
    const reviewRows = {review_json};
    const labels = {labels_json};

    function csvEscape(value) {{
      const text = value === null || value === undefined ? "" : String(value);
      return '"' + text.replaceAll('"', '""') + '"';
    }}

    function updateStatus() {{
      const labeled = Array.from(document.querySelectorAll(".review-label")).filter(el => el.value).length;
      document.getElementById("status").textContent = `${{labeled}} / ${{reviewRows.length}} labeled`;
    }}

    document.querySelectorAll("tr[data-row-id]").forEach((tr) => {{
      const input = tr.querySelector(".review-label");
      tr.querySelectorAll(".label-btn").forEach((btn) => {{
        btn.addEventListener("click", () => {{
          input.value = btn.dataset.label;
          tr.querySelectorAll(".label-btn").forEach((other) => other.classList.remove("active"));
          btn.classList.add("active");
          updateStatus();
        }});
      }});
      tr.querySelector(".clear-btn").addEventListener("click", () => {{
        input.value = "";
        tr.querySelectorAll(".label-btn").forEach((other) => other.classList.remove("active"));
        updateStatus();
      }});
    }});

    function downloadReviewCsv() {{
      const headers = [
        "review_index", "source_file", "source_path", "trace_index", "fiber_id",
        "nd2_name", "tile_name", "baseline_9p5_10d", "shifted_max", "shifted_min",
        "shifted_amplitude", "reviewed_label", "notes"
      ];
      const lines = [headers.map(csvEscape).join(",")];
      document.querySelectorAll("tr[data-row-id]").forEach((tr, idx) => {{
        const meta = reviewRows[idx];
        const label = tr.querySelector(".review-label").value;
        const notes = tr.querySelector(".review-notes").value;
        const row = headers.map((header) => {{
          if (header === "reviewed_label") return label;
          if (header === "notes") return notes;
          return meta[header];
        }});
        lines.push(row.map(csvEscape).join(","));
      }});
      const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "hpc_random100_reviewed_labels.csv";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }}

    updateStatus();
  </script>
</body>
</html>
"""
    path.write_text(doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--schema", type=Path, default=Path("configs/motif_label_schema.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_random100"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260504)
    args = parser.parse_args()

    labels = load_labels(args.schema)
    candidates, times = collect_candidates(args.raw_dir)
    if len(candidates) < args.n:
        raise SystemExit(f"Only {len(candidates)} candidates survived filtering; need {args.n}.")

    rng = Random(args.seed)
    selected = rng.sample(candidates, args.n)
    selected.sort(key=lambda row: (row["source_file"], row["trace_index"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_candidates_csv(args.out_dir / "hpc_random100_candidates.csv", selected)
    write_html(args.out_dir / "hpc_random100_label_review.html", selected, labels, times)

    print(f"total eligible candidates: {len(candidates)}")
    print(f"selected traces: {len(selected)}")
    print(f"html: {args.out_dir / 'hpc_random100_label_review.html'}")
    print(f"candidate csv: {args.out_dir / 'hpc_random100_candidates.csv'}")


if __name__ == "__main__":
    main()
