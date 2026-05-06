#!/usr/bin/env python3
"""Compare old HPC peak bundle to current HPC RF predictions and export old-only review HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from export_brainwide_peak_comparison_review import compact_svg
from export_hpc_label_review import BASELINE_END_DAYS, TIME_END_DAYS, TIME_START_DAYS, collect_candidates, load_labels, trace_to_float
from train_hpc_round_review import build_feature_matrix, candidate_key, read_reviewed_labels


def key_from_row(row: dict | pd.Series, file_col: str = "source_file") -> tuple[str, int]:
    return str(row[file_col]), int(row["fiber_id"])


def load_old_matrix(path: Path, old_label: str, current_label: str) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], np.ndarray]]:
    df = pd.read_csv(path)
    raw_cols = [col for col in df.columns if col.startswith("raw_")]
    rows: dict[tuple[str, int], dict] = {}
    traces: dict[tuple[str, int], np.ndarray] = {}
    for row in df.to_dict("records"):
        key = key_from_row(row, "file")
        rows[key] = {
            "old_label": old_label,
            "suggested_label": current_label,
            "source_file": row["file"],
            "source_path": str(Path("data/raw/hpc") / str(row["file"])),
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


def load_current_raw_traces(raw_dir: Path, keys: set[tuple[str, int]]) -> dict[tuple[str, int], np.ndarray]:
    by_file: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        by_file.setdefault(key[0], []).append(key)

    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    traces: dict[tuple[str, int], np.ndarray] = {}
    for file_name, file_keys in sorted(by_file.items()):
        npz_path = raw_dir / file_name
        if not npz_path.exists():
            continue
        needed_by_fiber = {fiber_id: (key_file, fiber_id) for key_file, fiber_id in file_keys}
        with np.load(npz_path, allow_pickle=True) as data:
            recovered = data["recovered_cfos"]
            fiber_ids = data["fiber_ids"]
            for trace_index, fiber_id in enumerate(fiber_ids):
                key = needed_by_fiber.get(int(fiber_id))
                if key is None:
                    continue
                raw_trace = trace_to_float(recovered[trace_index])
                if raw_trace.size != times.size or not np.isfinite(raw_trace).any():
                    continue
                baseline = float(np.nanmean(raw_trace[baseline_mask]))
                traces[key] = raw_trace - baseline
    return traces


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
        raise SystemExit("No labels matched current HPC candidates.")
    return build_feature_matrix(train_candidates), np.array(y_labels, dtype=object), train_candidates


def fit_current_model(
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


def predict_current_hpc(raw_dir: Path, model: Pipeline, target_labels: list[str]) -> tuple[dict[tuple[str, int, int], dict], list[dict], list[str]]:
    candidates, _ = collect_candidates(raw_dir)
    x_all = build_feature_matrix(candidates)
    proba = model.predict_proba(x_all)
    pred = model.predict(x_all)
    model_labels = model.classes_.tolist()
    label_to_idx = {label: idx for idx, label in enumerate(model_labels)}
    by_key = {}
    target_rows = []
    for i, row in enumerate(candidates):
        out = {
            "source_file": row["source_file"],
            "source_path": row["source_path"],
            "trace_index": int(row["trace_index"]),
            "fiber_id": int(row["fiber_id"]),
            "nd2_name": row.get("nd2_name", ""),
            "tile_name": row.get("tile_name", ""),
            "current_predicted_label": str(pred[i]),
            "current_prediction_confidence": float(proba[i].max()),
            "baseline_9p5_10d": float(row["baseline_9p5_10d"]),
            "shifted_amplitude": float(row["shifted_amplitude"]),
        }
        for label in target_labels:
            out[f"prob_{label}"] = float(proba[i, label_to_idx[label]]) if label in label_to_idx else ""
        key = key_from_row(out)
        by_key[key] = out
        if out["current_predicted_label"] in target_labels:
            target_rows.append(out)
    return by_key, target_rows, model_labels


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_old_only_rows(
    old_rows: dict[tuple[str, int], dict],
    current_by_key: dict[tuple[str, int], dict],
    target_current_label: str,
    comparison_name: str,
) -> list[dict]:
    rows = []
    for key in sorted(old_rows):
        current = current_by_key.get(key)
        if current is not None and current.get("current_predicted_label") == target_current_label:
            continue
        row = dict(old_rows[key])
        row["comparison"] = comparison_name
        if current is None:
            row["current_predicted_label"] = "not_in_current_filtered_candidates"
            row["current_prediction_confidence"] = ""
            row["prob_dual_peak"] = ""
            row["prob_second_peak"] = ""
            row["shifted_amplitude"] = ""
        else:
            row.update(
                {
                    "current_predicted_label": current.get("current_predicted_label", ""),
                    "current_prediction_confidence": current.get("current_prediction_confidence", ""),
                    "prob_dual_peak": current.get("prob_dual_peak", ""),
                    "prob_second_peak": current.get("prob_second_peak", ""),
                    "shifted_amplitude": current.get("shifted_amplitude", ""),
                    "nd2_name": current.get("nd2_name", ""),
                    "tile_name": current.get("tile_name", ""),
                }
            )
        rows.append(row)
    return rows


def display_float(value) -> str:
    try:
        if value == "" or value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def label_buttons(labels: list[str], selected: str) -> str:
    return "".join(
        (
            f'<button type="button" class="label-btn active" data-label="{html.escape(label)}">{html.escape(label)}</button>'
            if label == selected
            else f'<button type="button" class="label-btn" data-label="{html.escape(label)}">{html.escape(label)}</button>'
        )
        for label in labels
    )


def write_review_html(
    path: Path,
    rows: list[dict],
    old_traces: dict[tuple[str, int], np.ndarray],
    current_traces: dict[tuple[str, int], np.ndarray],
    labels: list[str],
) -> None:
    html_rows = []
    review_rows = []
    for review_index, row in enumerate(rows, start=1):
        key = key_from_row(row)
        suggested = row.get("suggested_label", "")
        if suggested not in labels:
            suggested = labels[0]
        review_rows.append(
            {
                "review_index": review_index,
                "comparison": row.get("comparison", ""),
                "old_label": row.get("old_label", ""),
                "suggested_label": row.get("suggested_label", ""),
                "current_predicted_label": row.get("current_predicted_label", ""),
                "source_file": row.get("source_file", ""),
                "source_path": row.get("source_path", ""),
                "trace_index": row.get("trace_index", ""),
                "fiber_id": row.get("fiber_id", ""),
                "nd2_name": row.get("nd2_name", ""),
                "tile_name": row.get("tile_name", ""),
                "old_prediction_confidence": row.get("old_prediction_confidence", ""),
                "current_prediction_confidence": row.get("current_prediction_confidence", ""),
                "prob_dual_peak": row.get("prob_dual_peak", ""),
                "prob_second_peak": row.get("prob_second_peak", ""),
                "baseline": row.get("baseline", ""),
                "shifted_amplitude": row.get("shifted_amplitude", ""),
            }
        )
        html_rows.append(
            f"""
            <tr data-row-id="r{review_index}">
              <td>
                <div class="trace-label">Old trace from old bundle</div>
                {compact_svg(old_traces.get(key, np.zeros(1000, dtype=np.float64)), width=500, height=190)}
              </td>
              <td>
                <div class="trace-label">New trace from current HPC npz</div>
                {compact_svg(current_traces.get(key, np.zeros(1000, dtype=np.float64)), width=500, height=190)}
              </td>
              <td>{html.escape(str(row.get("old_label", "")))}</td>
              <td>{html.escape(str(row.get("current_predicted_label", "")))}</td>
              <td>{html.escape(str(row.get("source_file", "")))}</td>
              <td>{html.escape(str(row.get("trace_index", "")))}</td>
              <td>{html.escape(str(row.get("fiber_id", "")))}</td>
              <td>{display_float(row.get("old_prediction_confidence", ""))}</td>
              <td>{display_float(row.get("current_prediction_confidence", ""))}</td>
              <td>{display_float(row.get("prob_dual_peak", ""))}</td>
              <td>{display_float(row.get("prob_second_peak", ""))}</td>
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
  <title>HPC old-only peak review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ margin: 0 0 16px; color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 7px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f3f4f6; z-index: 2; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    textarea {{ width: 210px; }}
    .trace-label {{ font-size: 11px; font-weight: 700; color: #374151; margin-bottom: 4px; }}
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
  <h1>HPC old-only peak review</h1>
  <p class="meta">{len(review_rows)} old-only traces. Old dual_peak is preselected as dual_peak; old recall_only is preselected as second_peak.</p>
  <div class="toolbar">
    <button type="button" onclick="downloadReviewCsv()">Download Reviewed CSV</button>
    <div class="status" id="status"></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Old Trace</th><th>New Trace</th><th>Old Label</th><th>Current Prediction</th>
        <th>File</th><th>Trace Index</th><th>Fiber ID</th>
        <th>Old Conf</th><th>Current Conf</th><th>Prob Dual</th><th>Prob Second</th>
        <th>Reviewed Label</th><th>Notes</th>
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
        "review_index", "comparison", "old_label", "suggested_label", "current_predicted_label",
        "source_file", "source_path", "trace_index", "fiber_id", "nd2_name", "tile_name",
        "old_prediction_confidence", "current_prediction_confidence", "prob_dual_peak", "prob_second_peak",
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
      link.download = "hpc_old_only_peak_reviewed_labels.csv";
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_old_only_peak_review"))
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    labels = load_labels(args.schema)
    x_train, y_train, train_candidates = build_training_set(args.raw_dir, args.labels_csv)
    model, metrics, report = fit_current_model(
        x_train,
        y_train,
        args.seed,
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
    )

    current_by_key, current_peak_rows, model_labels = predict_current_hpc(args.raw_dir, model, ["dual_peak", "second_peak"])
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

    dual_old_only = make_old_only_rows(
        old_dual,
        current_by_key,
        "dual_peak",
        "old_dual_peak_not_predicted_as_current_dual_peak",
    )
    recall_old_only = make_old_only_rows(
        old_recall,
        current_by_key,
        "second_peak",
        "old_recall_only_not_predicted_as_current_second_peak",
    )
    rows = dual_old_only + recall_old_only
    old_traces = {**old_dual_traces, **old_recall_traces}
    current_traces = load_current_raw_traces(
        args.raw_dir,
        {key_from_row(row) for row in rows},
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "current_hpc_random_forest_model.joblib")
    (args.out_dir / "current_hpc_random_forest_validation_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        args.out_dir / "current_hpc_random_forest_metrics.csv",
        index=False,
    )
    current_fieldnames = [
        "source_file",
        "source_path",
        "trace_index",
        "fiber_id",
        "nd2_name",
        "tile_name",
        "current_predicted_label",
        "current_prediction_confidence",
        "prob_dual_peak",
        "prob_second_peak",
        "baseline_9p5_10d",
        "shifted_amplitude",
    ]
    write_csv(args.out_dir / "current_hpc_dual_second_peak_prediction_ids.csv", current_peak_rows, current_fieldnames)
    review_fieldnames = [
        "comparison",
        "old_label",
        "suggested_label",
        "current_predicted_label",
        "source_file",
        "source_path",
        "trace_index",
        "fiber_id",
        "nd2_name",
        "tile_name",
        "old_prediction_confidence",
        "current_prediction_confidence",
        "prob_dual_peak",
        "prob_second_peak",
        "baseline",
        "shifted_amplitude",
    ]
    write_csv(args.out_dir / "hpc_old_only_peak_review_candidates.csv", rows, review_fieldnames)
    write_csv(args.out_dir / "hpc_old_dual_peak_old_only.csv", dual_old_only, review_fieldnames)
    write_csv(args.out_dir / "hpc_old_recall_only_old_only.csv", recall_old_only, review_fieldnames)
    write_review_html(args.out_dir / "hpc_old_only_peak_review.html", rows, old_traces, current_traces, labels)

    summary = {
        "old_dual_peak": len(old_dual),
        "current_dual_peak": sum(row["current_predicted_label"] == "dual_peak" for row in current_peak_rows),
        "dual_old_only": len(dual_old_only),
        "old_recall_only": len(old_recall),
        "current_second_peak": sum(row["current_predicted_label"] == "second_peak" for row in current_peak_rows),
        "recall_old_only": len(recall_old_only),
        "n_review_rows": len(rows),
        "n_review_rows_with_current_trace": len(current_traces),
        "n_train_labels_matched": len(train_candidates),
        "model_labels": model_labels,
        "metrics": metrics,
    }
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"out: {args.out_dir}")
    for key, value in summary.items():
        if key not in {"model_labels", "metrics"}:
            print(f"{key}: {value}")
    print(f"html: {args.out_dir / 'hpc_old_only_peak_review.html'}")


if __name__ == "__main__":
    main()
