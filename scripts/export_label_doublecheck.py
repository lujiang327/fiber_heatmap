#!/usr/bin/env python3
"""Export all merged human labels into one double-check HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from export_hpc_label_review import collect_candidates, load_labels, make_svg
from train_hpc_round_review import candidate_key


def read_labels(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if (row.get("reviewed_label") or "").strip()]


def write_html(path: Path, rows: list[dict], labels: list[str], candidate_by_key: dict, times) -> None:
    label_buttons_cache = {
        current: "".join(
            (
                f'<button type="button" class="label-btn active" data-label="{html.escape(label)}">{html.escape(label)}</button>'
                if label == current
                else f'<button type="button" class="label-btn" data-label="{html.escape(label)}">{html.escape(label)}</button>'
            )
            for label in labels
        )
        for current in labels
    }

    html_rows = []
    review_rows = []
    for review_index, row in enumerate(rows, start=1):
        key = candidate_key(row)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        current_label = row["reviewed_label"].strip()
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
                "label_round_id": row.get("label_round_id", ""),
                "label_file": row.get("label_file", ""),
                "original_reviewed_label": current_label,
            }
        )
        html_rows.append(
            f"""
            <tr data-row-id="r{review_index}">
              <td>{make_svg(times, candidate["trace"])}</td>
              <td>{html.escape(row["source_file"])}</td>
              <td>{html.escape(str(row["trace_index"]))}</td>
              <td>{html.escape(str(row["fiber_id"]))}</td>
              <td>{html.escape(row.get("label_round_id", ""))}</td>
              <td>{html.escape(current_label)}</td>
              <td>
                <div class="label-picker">
                  <div class="label-btn-row">{label_buttons_cache.get(current_label, label_buttons_cache[labels[0]])}</div>
                  <button type="button" class="clear-btn">Clear</button>
                  <input type="hidden" class="review-label" value="{html.escape(current_label)}">
                </div>
              </td>
              <td><textarea class="review-notes" rows="3">{html.escape(row.get("notes", ""))}</textarea></td>
            </tr>
            """
        )

    review_json = json.dumps(review_rows)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HPC merged labels double-check</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ margin: 0 0 16px; color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f5f5f5; z-index: 2; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    textarea {{ width: 220px; }}
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
  <h1>HPC merged labels double-check</h1>
  <p class="meta">{len(review_rows)} merged human-labeled traces. Current label is preselected; correct uncertain labels and download the CSV.</p>
  <div class="toolbar">
    <button type="button" onclick="downloadReviewCsv()">Download Double-Checked CSV</button>
    <div class="status" id="status"></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Trace Plot</th>
        <th>File</th>
        <th>Trace Index</th>
        <th>Fiber ID</th>
        <th>Last Round</th>
        <th>Original Label</th>
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
        "review_index", "source_file", "source_path", "trace_index", "fiber_id", "nd2_name", "tile_name",
        "baseline_9p5_10d", "shifted_max", "shifted_min", "shifted_amplitude",
        "label_round_id", "label_file", "original_reviewed_label", "reviewed_label", "notes"
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
      link.download = "hpc_merged_labels_double_checked.csv";
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
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_merged_label_doublecheck"))
    args = parser.parse_args()

    labels = load_labels(args.schema)
    label_rows = read_labels(args.labels_csv)
    candidates, times = collect_candidates(args.raw_dir)
    candidate_by_key = {candidate_key(row): row for row in candidates}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_html(args.out_dir / "hpc_merged_label_doublecheck.html", label_rows, labels, candidate_by_key, times)
    print(f"labels: {len(label_rows)}")
    print(f"html: {args.out_dir / 'hpc_merged_label_doublecheck.html'}")


if __name__ == "__main__":
    main()
