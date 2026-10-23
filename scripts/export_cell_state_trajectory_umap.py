#!/usr/bin/env python3
"""Embed per-cell state windows and render trajectory bundle contours.

This is different from ``export_brainwide_population_manifolds.py``:

  population manifold:
    sample = time point, feature = cell/fiber

  this script:
    sample = one cell/fiber in one behavioral state window
    feature = trace shape inside that state window

Each retained cell/fiber therefore has one embedded point per state, and its
state sequence is represented as a short trajectory through UMAP space.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fiber_heatmap_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TIME_START_DAYS = 9.5
TIME_END_DAYS = 15.0
BASELINE_END_DAYS = 10.0
DEFAULT_STATES = "home_cage_1:9.5:10,encoding:10:11.5,home_cage_2:11.5:13.5,recall:13.5:15"
STATE_COLORS = {
    "home_cage_1": "#2563eb",
    "encoding": "#db2777",
    "home_cage_2": "#16a34a",
    "recall": "#f59e0b",
}


def parse_states(spec: str) -> list[tuple[str, float, float]]:
    states = []
    for chunk in spec.split(","):
        name, start, end = chunk.split(":")
        states.append((name.strip(), float(start), float(end)))
    return states


def trace_to_float(row: np.ndarray) -> np.ndarray:
    out = np.asarray(row, dtype=np.float64)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def resample_segment(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == n_bins:
        return values
    old_x = np.linspace(0.0, 1.0, values.size)
    new_x = np.linspace(0.0, 1.0, n_bins)
    return np.interp(new_x, old_x, values)


def load_cell_state_features(
    npz_path: Path,
    states: list[tuple[str, float, float]],
    n_bins: int,
    min_amplitude: float,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    with np.load(npz_path, allow_pickle=True) as data:
        recovered = data["recovered_cfos"]
        fiber_ids = np.asarray(data["fiber_ids"])

    features = []
    rows = []
    for trace_index in range(recovered.shape[0]):
        raw = trace_to_float(recovered[trace_index])
        if raw.size != times.size or not np.isfinite(raw).any():
            continue
        baseline = float(np.nanmean(raw[baseline_mask]))
        shifted = raw - baseline
        amplitude = float(max(abs(np.nanmax(shifted)), abs(np.nanmin(shifted))))
        if not math.isfinite(amplitude) or amplitude < min_amplitude:
            continue
        for state_index, (state, start, end) in enumerate(states):
            if state_index == len(states) - 1:
                mask = (times >= start) & (times <= end)
            else:
                mask = (times >= start) & (times < end)
            segment = shifted[mask]
            if not np.isfinite(segment).any() or segment.size < 2:
                continue
            feature = resample_segment(np.nan_to_num(segment, nan=0.0), n_bins)
            features.append(feature)
            rows.append(
                {
                    "source_file": npz_path.name,
                    "trace_index": trace_index,
                    "fiber_id": int(fiber_ids[trace_index]),
                    "state": state,
                    "state_index": state_index,
                    "start_day": start,
                    "end_day": end,
                    "baseline": baseline,
                    "amplitude": amplitude,
                }
            )
    if not features:
        raise SystemExit(f"No usable cell-state samples from {npz_path}")
    return np.vstack(features), rows, times


def embed_features(
    features: np.ndarray,
    method: str,
    n_pcs: int,
    seed: int | None,
    umap_neighbors: int,
    umap_min_dist: float,
) -> tuple[np.ndarray, float, str]:
    scaled = StandardScaler().fit_transform(features)
    n_components = max(2, min(n_pcs, scaled.shape[0], scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(scaled)
    cumulative_variance = float(np.sum(pca.explained_variance_ratio_))
    if method == "pca":
        embedding = pcs[:, :2]
        return embedding, cumulative_variance, "pca"
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise SystemExit("UMAP requested but umap-learn is not installed. Install with: pip install umap-learn") from exc
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state.*",
            category=UserWarning,
            module="umap.umap_",
        )
        embedding = reducer.fit_transform(pcs)
    return embedding, cumulative_variance, "umap"


def add_embedding_to_rows(rows: list[dict], embedding: np.ndarray) -> list[dict]:
    out = []
    for row, point in zip(rows, embedding, strict=False):
        updated = dict(row)
        updated["dim1"] = float(point[0])
        updated["dim2"] = float(point[1])
        out.append(updated)
    return out


def build_line_density(rows: list[dict], grid_size: int, samples_per_segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_fiber: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_fiber.setdefault((row["source_file"], int(row["fiber_id"])), []).append(row)

    points = np.array([[row["dim1"], row["dim2"]] for row in rows], dtype=float)
    x_pad = max((points[:, 0].max() - points[:, 0].min()) * 0.06, 1e-6)
    y_pad = max((points[:, 1].max() - points[:, 1].min()) * 0.06, 1e-6)
    x_edges = np.linspace(points[:, 0].min() - x_pad, points[:, 0].max() + x_pad, grid_size + 1)
    y_edges = np.linspace(points[:, 1].min() - y_pad, points[:, 1].max() + y_pad, grid_size + 1)

    line_points = []
    for fiber_rows in by_fiber.values():
        ordered = sorted(fiber_rows, key=lambda row: int(row["state_index"]))
        for start, end in zip(ordered[:-1], ordered[1:], strict=False):
            xs = np.linspace(float(start["dim1"]), float(end["dim1"]), samples_per_segment)
            ys = np.linspace(float(start["dim2"]), float(end["dim2"]), samples_per_segment)
            line_points.extend(zip(xs, ys, strict=False))
    line_points_arr = np.asarray(line_points, dtype=float)
    density, _, _ = np.histogram2d(line_points_arr[:, 1], line_points_arr[:, 0], bins=[y_edges, x_edges])
    try:
        from scipy.ndimage import gaussian_filter

        density = gaussian_filter(density, sigma=1.2)
    except Exception:
        pass
    return density, x_edges, y_edges


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_centroids(rows: list[dict], states: list[tuple[str, float, float]]) -> list[dict]:
    out = []
    for state, _, _ in states:
        subset = [row for row in rows if row["state"] == state]
        if not subset:
            continue
        coords = np.array([[row["dim1"], row["dim2"]] for row in subset], dtype=float)
        centroid = coords.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(coords - centroid, axis=1)))
        out.append(
            {
                "state": state,
                "n_cells": len(subset),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "within_state_spread": spread,
            }
        )
    return out


def plot_bundle(
    path: Path,
    rows: list[dict],
    centroids: list[dict],
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    states: list[tuple[str, float, float]],
    title: str,
    method: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    ax.imshow(density, origin="lower", extent=extent, cmap="Greys", alpha=0.45, aspect="auto")
    positive = density[density > 0]
    if positive.size:
        levels = np.quantile(positive, [0.55, 0.7, 0.82, 0.92, 0.98])
        levels = np.unique(levels)
        if levels.size > 1:
            ax.contour(
                (x_edges[:-1] + x_edges[1:]) / 2,
                (y_edges[:-1] + y_edges[1:]) / 2,
                density,
                levels=levels,
                colors="#111827",
                linewidths=0.8,
                alpha=0.8,
            )

    for state, _, _ in states:
        subset = [row for row in rows if row["state"] == state]
        if not subset:
            continue
        coords = np.array([[row["dim1"], row["dim2"]] for row in subset], dtype=float)
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.22, color=STATE_COLORS.get(state), label=state)

    centroid_by_state = {row["state"]: row for row in centroids}
    ordered_centroids = []
    for state, _, _ in states:
        row = centroid_by_state.get(state)
        if row:
            ordered_centroids.append(row)
            ax.scatter(row["centroid_x"], row["centroid_y"], s=90, color=STATE_COLORS.get(state), edgecolor="black", zorder=5)
            ax.text(row["centroid_x"], row["centroid_y"], state, fontsize=9, weight="bold", ha="left", va="bottom")
    for start, end in zip(ordered_centroids[:-1], ordered_centroids[1:], strict=False):
        ax.annotate(
            "",
            xy=(end["centroid_x"], end["centroid_y"]),
            xytext=(start["centroid_x"], start["centroid_y"]),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "black"},
        )

    ax.set_title(f"{title}\nCell-state {method.upper()} trajectory bundle density")
    ax.set_xlabel("UMAP 1" if method == "umap" else "PC 1")
    ax.set_ylabel("UMAP 2" if method == "umap" else "PC 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_index(path: Path, summary: dict) -> None:
    plot = html.escape(summary["plot"])
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cell-State Trajectory UMAP</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    img {{ max-width: 1100px; width: 100%; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Cell-State Trajectory UMAP</h1>
  <p><code>{html.escape(summary["file"])}</code></p>
  <p>Cells kept: {summary["n_cells"]} | Samples embedded: {summary["n_samples"]} | Method: {html.escape(summary["method"])}</p>
  <a href="{plot}"><img src="{plot}" alt="trajectory bundle"></a>
</body>
</html>
"""
    path.write_text(doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--method", choices=["umap", "pca"], default="umap")
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--min-amplitude", type=float, default=0.2)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--samples-per-segment", type=int, default=40)
    args = parser.parse_args()

    states = parse_states(args.states)
    out_dir = args.out_dir or Path("outputs/reports/cell_state_trajectory_umap") / f"{datetime.now():%Y%m%d_%H%M%S}_{args.npz.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    features, rows, _ = load_cell_state_features(args.npz, states, args.n_bins, args.min_amplitude)
    embedding, pca_variance, method = embed_features(
        features,
        args.method,
        args.n_pcs,
        None if args.no_seed else args.seed,
        args.umap_neighbors,
        args.umap_min_dist,
    )
    embedded_rows = add_embedding_to_rows(rows, embedding)
    centroids = compute_centroids(embedded_rows, states)
    density, x_edges, y_edges = build_line_density(embedded_rows, args.grid_size, args.samples_per_segment)

    coords_path = out_dir / "cell_state_embedding.csv"
    centroids_path = out_dir / "state_centroids.csv"
    density_path = out_dir / "line_density.npy"
    plot_path = out_dir / "cell_state_trajectory_bundle.png"
    write_csv(
        coords_path,
        embedded_rows,
        [
            "source_file",
            "trace_index",
            "fiber_id",
            "state",
            "state_index",
            "start_day",
            "end_day",
            "baseline",
            "amplitude",
            "dim1",
            "dim2",
        ],
    )
    write_csv(centroids_path, centroids, ["state", "n_cells", "centroid_x", "centroid_y", "within_state_spread"])
    np.save(density_path, density)
    plot_bundle(plot_path, embedded_rows, centroids, density, x_edges, y_edges, states, args.npz.name, method)

    summary = {
        "file": args.npz.name,
        "method": method,
        "n_cells": len({(row["source_file"], row["fiber_id"]) for row in embedded_rows}),
        "n_samples": len(embedded_rows),
        "n_bins": args.n_bins,
        "min_amplitude": args.min_amplitude,
        "pca_cumulative_variance": pca_variance,
        "plot": plot_path.name,
        "coords_csv": coords_path.name,
        "centroids_csv": centroids_path.name,
        "density_npy": density_path.name,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2))
    write_index(out_dir / "index.html", summary)
    print(f"out: {out_dir}")
    print(f"cells kept: {summary['n_cells']}")
    print(f"samples embedded: {summary['n_samples']}")


if __name__ == "__main__":
    main()
