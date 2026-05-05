#!/usr/bin/env python3
"""Train a first-pass HPC motif classifier and export per-class review traces."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from export_hpc_label_review import collect_candidates, load_labels, make_svg


def read_reviewed_labels(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if (row.get("reviewed_label") or "").strip()]


def candidate_key(row: dict) -> tuple[str, int]:
    return str(row["source_file"]), int(row["trace_index"])


def trace_features(trace: np.ndarray, row: dict) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    chunks = np.array_split(trace, 100)
    downsample = np.array([np.nanmean(chunk) for chunk in chunks], dtype=np.float64)
    finite = np.isfinite(trace)
    if not finite.any():
        trace = np.zeros_like(trace)
    else:
        trace = np.where(finite, trace, np.nanmedian(trace[finite]))

    max_value = float(np.max(trace))
    min_value = float(np.min(trace))
    amplitude = max(abs(max_value), abs(min_value))
    argmax = float(np.argmax(trace) / max(len(trace) - 1, 1))
    argmin = float(np.argmin(trace) / max(len(trace) - 1, 1))
    pos_area = float(np.mean(np.clip(trace, 0, None)))
    neg_area = float(np.mean(np.clip(trace, None, 0)))
    early = float(np.mean(trace[:182]))
    mid = float(np.mean(trace[182:636]))
    late = float(np.mean(trace[636:]))
    final_delta = float(np.mean(trace[-91:]) - np.mean(trace[:91]))
    stats = np.array(
        [
            max_value,
            min_value,
            amplitude,
            argmax,
            argmin,
            pos_area,
            neg_area,
            early,
            mid,
            late,
            final_delta,
            float(row["shifted_max"]),
            float(row["shifted_min"]),
            float(row["shifted_amplitude"]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([downsample, stats])


def build_feature_matrix(candidates: list[dict]) -> np.ndarray:
    return np.vstack([trace_features(row["trace"], row) for row in candidates])


def fit_centroid_model(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-9] = 1.0
    xz = (x_train - mean) / std
    labels = np.array(sorted(set(y_train.tolist())), dtype=object)
    centroids = np.vstack([xz[y_train == label].mean(axis=0) for label in labels])
    return {"mean": mean, "std": std, "labels": labels, "centroids": centroids}


def predict_centroid(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xz = (x - model["mean"]) / model["std"]
    distances = np.linalg.norm(xz[:, None, :] - model["centroids"][None, :, :], axis=2)
    best = np.argmin(distances, axis=1)
    sorted_distances = np.sort(distances, axis=1)
    if distances.shape[1] > 1:
        confidence = (sorted_distances[:, 1] - sorted_distances[:, 0]) / (sorted_distances[:, 1] + 1e-9)
    else:
        confidence = np.ones(len(x), dtype=np.float64)
    return model["labels"][best], confidence


def centroid_distances(model: dict, x: np.ndarray) -> np.ndarray:
    xz = (x - model["mean"]) / model["std"]
    return np.linalg.norm(xz[:, None, :] - model["centroids"][None, :, :], axis=2)


def select_for_review(
    candidates: list[dict],
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    distances: np.ndarray,
    model_labels: list[str],
    exclude_keys: set[tuple[str, int]],
    labels: list[str],
    per_class: int,
) -> list[dict]:
    selected: list[dict] = []
    selected_keys: set[tuple[str, int]] = set()
    for label in labels:
        idxs = [
            i
            for i, row in enumerate(candidates)
            if predicted_labels[i] == label
            and candidate_key(row) not in exclude_keys
            and candidate_key(row) not in selected_keys
        ]
        label_distance_col = model_labels.index(label)
        idxs.sort(key=lambda i: (-float(confidence[i]), candidates[i]["source_file"], candidates[i]["trace_index"]))
        if len(idxs) <= per_class:
            chosen = idxs
        else:
            high = idxs[: per_class // 2]
            rest = idxs[per_class // 2 :]
            step = max(len(rest) / max(per_class - len(high), 1), 1)
            spread = [rest[int(j * step)] for j in range(per_class - len(high))]
            chosen = high + spread
        if len(chosen) < per_class:
            backfill = [
                i
                for i, row in enumerate(candidates)
                if candidate_key(row) not in exclude_keys
                and candidate_key(row) not in selected_keys
                and i not in chosen
            ]
            backfill.sort(
                key=lambda i: (
                    float(distances[i, label_distance_col]),
                    candidates[i]["source_file"],
                    candidates[i]["trace_index"],
                )
            )
            chosen.extend(backfill[: per_class - len(chosen)])
        for i in chosen[:per_class]:
            row = dict(candidates[i])
            row["predicted_label"] = label
            row["model_top_label"] = str(predicted_labels[i])
            row["prediction_confidence"] = float(confidence[i])
            row["target_label_distance"] = float(distances[i, label_distance_col])
            selected.append(row)
            selected_keys.add(candidate_key(row))
    selected.sort(key=lambda row: (row["predicted_label"], -row["prediction_confidence"]))
    return selected


def write_review_csv(path: Path, selected: list[dict]) -> None:
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
        "predicted_label",
        "model_top_label",
        "prediction_confidence",
        "target_label_distance",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row[key] for key in fieldnames})


def write_review_html(path: Path, selected: list[dict], all_labels: list[str], times: np.ndarray) -> None:
    rows = []
    review_rows = []
    for review_index, row in enumerate(selected, start=1):
        row_id = f"r{review_index}"
        label_buttons = "".join(
            (
                f'<button type="button" class="label-btn active" data-label="{html.escape(label)}">{html.escape(label)}</button>'
                if label == row["predicted_label"]
                else f'<button type="button" class="label-btn" data-label="{html.escape(label)}">{html.escape(label)}</button>'
            )
            for label in all_labels
        )
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
                "predicted_label": row["predicted_label"],
                "model_top_label": row.get("model_top_label", row["predicted_label"]),
                "prediction_confidence": row["prediction_confidence"],
                "target_label_distance": row.get("target_label_distance", ""),
            }
        )
        rows.append(
            f"""
            <tr data-row-id="{row_id}">
              <td>{make_svg(times, row["trace"])}</td>
              <td>{html.escape(row["source_file"])}</td>
              <td>{row["trace_index"]}</td>
              <td>{row["fiber_id"]}</td>
              <td>{html.escape(row["predicted_label"])}</td>
              <td>{html.escape(row.get("model_top_label", row["predicted_label"]))}</td>
              <td>{row["prediction_confidence"]:.4f}</td>
              <td>
                <div class="label-picker">
                  <div class="label-btn-row">{label_buttons}</div>
                  <button type="button" class="clear-btn">Clear</button>
                  <input type="hidden" class="review-label" value="{html.escape(row["predicted_label"])}">
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
  <title>HPC model round review</title>
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
  <h1>HPC model round review</h1>
  <p class="meta">Predicted labels are preselected. Correct labels as needed, then download the reviewed CSV.</p>
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
        <th>Predicted</th>
        <th>Model Top</th>
        <th>Confidence</th>
        <th>Reviewed Label</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
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
        "predicted_label", "model_top_label", "prediction_confidence", "target_label_distance", "reviewed_label", "notes"
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
      link.download = "hpc_model_round_reviewed_labels.csv";
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
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/labels/review_rounds/hpc_random100_reviewed_labels.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_model_round1"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument(
        "--target-labels",
        nargs="*",
        default=None,
        help="Optional subset of trained labels to export for review.",
    )
    args = parser.parse_args()

    all_labels = load_labels(args.schema)
    reviewed_rows = read_reviewed_labels(args.labels_csv)
    candidates, times = collect_candidates(args.raw_dir)
    by_key = {candidate_key(row): row for row in candidates}

    train_candidates = []
    y_train = []
    missing = []
    for row in reviewed_rows:
        key = candidate_key(row)
        candidate = by_key.get(key)
        if candidate is None:
            missing.append(key)
            continue
        train_candidates.append(candidate)
        y_train.append(row["reviewed_label"].strip())
    if not train_candidates:
        raise SystemExit("No labeled rows matched the HPC candidate set.")

    x_train = build_feature_matrix(train_candidates)
    y_train_array = np.array(y_train, dtype=object)
    model = fit_centroid_model(x_train, y_train_array)

    x_all = build_feature_matrix(candidates)
    predicted, confidence = predict_centroid(model, x_all)
    distances = centroid_distances(model, x_all)
    exclude_keys = {candidate_key(row) for row in train_candidates}
    trained_labels = model["labels"].tolist()
    if args.target_labels:
        unknown_targets = [label for label in args.target_labels if label not in trained_labels]
        if unknown_targets:
            raise SystemExit(f"Target labels have no training examples: {', '.join(unknown_targets)}")
        review_labels = args.target_labels
    else:
        review_labels = trained_labels
    selected = select_for_review(
        candidates,
        predicted,
        confidence,
        distances,
        trained_labels,
        exclude_keys,
        review_labels,
        args.per_class,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_review_csv(args.out_dir / "hpc_model_round1_candidates.csv", selected)
    write_review_html(args.out_dir / "hpc_model_round1_review.html", selected, all_labels, times)
    np.savez(
        args.out_dir / "hpc_model_round1_centroid_model.npz",
        labels=model["labels"],
        mean=model["mean"],
        std=model["std"],
        centroids=model["centroids"],
    )

    train_counts = {label: int(np.sum(y_train_array == label)) for label in trained_labels}
    pred_counts = {label: int(np.sum(predicted == label)) for label in trained_labels}
    selected_counts = {label: sum(row["predicted_label"] == label for row in selected) for label in review_labels}
    print(f"matched labeled rows: {len(train_candidates)}")
    print(f"missing labeled rows: {len(missing)}")
    print("training label counts:")
    for label in trained_labels:
        print(f"  {label}: {train_counts[label]}")
    untrained = [label for label in all_labels if label not in trained_labels]
    if untrained:
        print("untrained labels with zero examples:")
        for label in untrained:
            print(f"  {label}")
    print("predicted HPC candidate counts:")
    for label in trained_labels:
        print(f"  {label}: {pred_counts[label]}")
    print("selected review counts:")
    for label in review_labels:
        print(f"  {label}: {selected_counts[label]}")
    print(f"html: {args.out_dir / 'hpc_model_round1_review.html'}")
    print(f"candidate csv: {args.out_dir / 'hpc_model_round1_candidates.csv'}")
    print(f"model: {args.out_dir / 'hpc_model_round1_centroid_model.npz'}")


if __name__ == "__main__":
    main()
