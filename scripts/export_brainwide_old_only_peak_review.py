#!/usr/bin/env python3
"""Export old-only brain-wide peak differences into a manual review HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_brainwide_peak_comparison_review import compact_svg, load_old_matrix, row_key
from export_hpc_label_review import load_labels


def load_old_only_rows(csv_path: Path, matrix_rows: dict, matrix_traces: dict) -> list[dict]:
    df = pd.read_csv(csv_path)
    rows = []
    for row in df.to_dict("records"):
        key = row_key(row, "source_file")
        meta = dict(matrix_rows.get(key, {}))
        meta.update(row)
        meta["trace"] = matrix_traces.get(key, np.zeros(1000, dtype=np.float64))
        rows.append(meta)
    return rows


def label_buttons(labels: list[str], selected: str) -> str:
    return "".join(
        (
            f'<button type="button" class="label-btn active" data-label="{html.escape(label)}">{html.escape(label)}</button>'
            if label == selected
            else f'<button type="button" class="label-btn" data-label="{html.escape(label)}">{html.escape(label)}</button>'
        )
        for label in labels
    )


def write_review_html(path: Path, rows: list[dict], labels: list[str]) -> None:
    html_rows = []
    review_rows = []
    for review_index, row in enumerate(rows, start=1):
        suggested = str(row.get("current_label", ""))
        if suggested not in labels:
            suggested = labels[0]
        review_rows.append(
            {
                "review_index": review_index,
                "comparison": row.get("comparison", ""),
                "old_label": row.get("old_label", ""),
                "suggested_label": row.get("current_label", ""),
                "source_file": row.get("source_file", ""),
                "source_path": row.get("source_path", ""),
                "trace_index": row.get("trace_index", ""),
                "fiber_id": row.get("fiber_id", ""),
                "key": row.get("key", ""),
                "region": row.get("region", ""),
                "major_group": row.get("major_group", ""),
                "old_prediction_confidence": row.get("old_prediction_confidence", ""),
                "new_prediction_confidence": row.get("new_prediction_confidence", ""),
                "prob_dual_peak": row.get("prob_dual_peak", ""),
                "prob_second_peak": row.get("prob_second_peak", ""),
                "baseline": row.get("baseline", ""),
                "shifted_amplitude": row.get("shifted_amplitude", ""),
            }
        )
        html_rows.append(
            f"""
            <tr data-row-id="r{review_index}">
              <td>{compact_svg(row["trace"], width=560, height=205)}</td>
              <td>{html.escape(str(row.get("comparison", "")))}</td>
              <td>{html.escape(str(row.get("old_label", "")))}</td>
              <td>{html.escape(str(row.get("current_label", "")))}</td>
              <td>{html.escape(str(row.get("source_file", "")))}</td>
              <td>{html.escape(str(row.get("trace_index", "")))}</td>
              <td>{html.escape(str(row.get("fiber_id", "")))}</td>
              <td>{html.escape(str(row.get("region", "")))}</td>
              <td>{html.escape(str(row.get("major_group", "")))}</td>
              <td>
                <div class="label-picker">
                  <div class="label-btn-row">{label_buttons(labels, suggested)}</div>
                  <button type="button" class="clear-btn">Clear</button>
                  <input type="hidden" class="review-label" value="{html.escape(suggested)}">
                </div>
              </td>
              <td><textarea class="review-notes" rows="3"></textarea></td>
            </tr>
            """
        )

    review_json = json.dumps(review_rows)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Brain-wide old-only peak review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ margin: 0 0 16px; color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 7px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f3f4f6; z-index: 2; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    textarea {{ width: 210px; }}
    .toolbar {{ position: fixed; right: 16px; bottom: 16px; z-index: 1000; display:flex; flex-direction:column; gap:10px; align-items:stretch; max-width: 360px; background: rgba(255,255,255,0.96); border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.12); }}
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
  <h1>Brain-wide old-only peak review</h1>
  <p class="meta">{len(review_rows)} old-only traces. Old dual_peak is preselected as dual_peak; old recall_only is preselected as second_peak.</p>
  <div class="toolbar">
    <button type="button" onclick="downloadReviewCsv()">Download Reviewed CSV</button>
    <div class="status" id="status"></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Trace</th>
        <th>Comparison</th>
        <th>Old Label</th>
        <th>Suggested Label</th>
        <th>File</th>
        <th>Trace Index</th>
        <th>Fiber ID</th>
        <th>Region</th>
        <th>Major Group</th>
        <th>Reviewed Label</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>{"".join(html_rows)}</tbody>
  </table>
  <script>
    const reviewRows = {review_json};
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
        "review_index", "comparison", "old_label", "suggested_label", "source_file", "source_path",
        "trace_index", "fiber_id", "key", "region", "major_group",
        "old_prediction_confidence", "new_prediction_confidence", "prob_dual_peak", "prob_second_peak",
        "baseline", "shifted_amplitude", "reviewed_label", "notes"
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
      link.download = "brainwide_old_only_peak_reviewed_labels.csv";
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
    path.write_text(doc, encoding="utf-8")


def write_seed_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "comparison",
        "old_label",
        "current_label",
        "source_file",
        "trace_index",
        "fiber_id",
        "key",
        "region",
        "major_group",
        "old_prediction_confidence",
        "baseline",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-bundle",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("outputs/reports/brainwide_peak_prediction_comparison/20260505_old_vs_current_dual_second"),
    )
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/review_exports/brainwide_old_only_peak_review"),
    )
    args = parser.parse_args()

    labels = load_labels(args.schema)
    old_dual_rows, old_dual_traces = load_old_matrix(
        args.old_bundle / "dual_peak" / "dual_peak_trace_matrix.csv",
        "dual_peak",
        "dual_peak",
    )
    old_recall_rows, old_recall_traces = load_old_matrix(
        args.old_bundle / "recall_only" / "recall_only_trace_matrix.csv",
        "recall_only",
        "second_peak",
    )
    dual_old_only = load_old_only_rows(
        args.comparison_dir / "dual_peak_old_only_missing_by_current.csv",
        old_dual_rows,
        old_dual_traces,
    )
    second_old_only = load_old_only_rows(
        args.comparison_dir / "second_peak_old_recall_only_missing_by_current.csv",
        old_recall_rows,
        old_recall_traces,
    )
    rows = dual_old_only + second_old_only

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_seed_csv(args.out_dir / "brainwide_old_only_peak_review_candidates.csv", rows)
    write_review_html(args.out_dir / "brainwide_old_only_peak_review.html", rows, labels)
    print(f"dual old-only: {len(dual_old_only)}")
    print(f"second old-only: {len(second_old_only)}")
    print(f"total: {len(rows)}")
    print(f"html: {args.out_dir / 'brainwide_old_only_peak_review.html'}")
    print(f"csv: {args.out_dir / 'brainwide_old_only_peak_review_candidates.csv'}")


if __name__ == "__main__":
    main()
