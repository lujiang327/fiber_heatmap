#!/usr/bin/env python3
"""Generate population-state manifolds for brain-wide NPZ files.

Each NPZ is treated as one neural population:
  samples = recovered time points
  features = fibers in that NPZ

This is different from the trace classifier pipeline, where each fiber trace is
one sample. Here the goal is to visualize time/state transitions of the whole
population within each NPZ file.
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
from sklearn.manifold import TSNE
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


def trace_to_float(row: np.ndarray) -> np.ndarray:
    out = np.asarray(row, dtype=np.float64)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def parse_states(spec: str) -> list[tuple[str, float, float]]:
    states = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise SystemExit(f"Invalid state spec {chunk!r}; expected name:start_day:end_day")
        name, start, end = parts
        states.append((name.strip(), float(start), float(end)))
    if not states:
        raise SystemExit("At least one state window is required.")
    return states


def assign_states(times: np.ndarray, states: list[tuple[str, float, float]]) -> np.ndarray:
    labels = np.full(times.shape, "unassigned", dtype=object)
    for i, (name, start, end) in enumerate(states):
        if i == len(states) - 1:
            mask = (times >= start) & (times <= end)
        else:
            mask = (times >= start) & (times < end)
        labels[mask] = name
    return labels


def gaussian_smooth_matrix(matrix: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return matrix
    radius = max(1, int(math.ceil(4 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_bins**2))
    kernel = kernel / kernel.sum()
    padded = np.pad(matrix, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(matrix)
    for col in range(matrix.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out


def load_shifted_population(
    npz_path: Path,
    min_amplitude: float,
    smooth_sigma_bins: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    times = np.linspace(TIME_START_DAYS, TIME_END_DAYS, 1000)
    baseline_mask = (times >= TIME_START_DAYS) & (times <= BASELINE_END_DAYS)
    with np.load(npz_path, allow_pickle=True) as data:
        recovered = data["recovered_cfos"]
        fiber_ids = np.asarray(data["fiber_ids"])

    shifted_rows = []
    kept_fiber_ids = []
    amplitudes = []
    for trace_index in range(recovered.shape[0]):
        raw = trace_to_float(recovered[trace_index])
        if raw.size != times.size or not np.isfinite(raw).any():
            continue
        baseline = float(np.nanmean(raw[baseline_mask]))
        shifted = raw - baseline
        amp = float(max(abs(np.nanmax(shifted)), abs(np.nanmin(shifted))))
        if not math.isfinite(amp) or amp < min_amplitude:
            continue
        shifted_rows.append(shifted)
        kept_fiber_ids.append(int(fiber_ids[trace_index]))
        amplitudes.append(amp)

    if not shifted_rows:
        return np.empty((times.size, 0)), times, {
            "n_total_fibers": int(recovered.shape[0]),
            "n_kept_fibers": 0,
            "kept_fiber_ids": [],
            "mean_amplitude": "",
        }

    trace_matrix = np.vstack(shifted_rows)
    population = trace_matrix.T
    population = gaussian_smooth_matrix(population, smooth_sigma_bins)
    population = np.nan_to_num(population, nan=0.0, posinf=0.0, neginf=0.0)
    meta = {
        "n_total_fibers": int(recovered.shape[0]),
        "n_kept_fibers": int(population.shape[1]),
        "kept_fiber_ids": kept_fiber_ids,
        "mean_amplitude": float(np.mean(amplitudes)),
    }
    return population, times, meta


def embed_population(
    population: np.ndarray,
    method: str,
    n_pcs: int,
    seed: int | None,
    umap_neighbors: int,
    umap_min_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    scaled = StandardScaler().fit_transform(population)
    n_components = max(1, min(n_pcs, scaled.shape[0], scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(scaled)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit("UMAP requested but umap-learn is not installed. Install with: pip install umap-learn") from exc
        reducer = umap.UMAP(
            n_components=3,
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
        actual_method = "umap"
    elif method == "tsne":
        embedding = TSNE(
            n_components=3,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, (pcs.shape[0] - 1) // 3)),
            random_state=seed,
        ).fit_transform(pcs)
        actual_method = "tsne"
    else:
        embedding = np.zeros((pcs.shape[0], 3), dtype=float)
        keep = min(3, pcs.shape[1])
        embedding[:, :keep] = pcs[:, :keep]
        actual_method = "pca"
    return embedding, pcs, pca.explained_variance_ratio_, actual_method


def compute_metrics(embedding: np.ndarray, state_labels: np.ndarray, state_order: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    centroid_rows = []
    distance_rows = []
    angle_rows = []
    centroids: dict[str, np.ndarray] = {}

    for state in state_order:
        mask = state_labels == state
        if not mask.any():
            continue
        pts = embedding[mask, :3]
        centroid = pts.mean(axis=0)
        centroids[state] = centroid
        spread = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))
        centroid_rows.append(
            {
                "state": state,
                "n_timepoints": int(mask.sum()),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "within_state_spread": spread,
            }
        )

    for a in state_order:
        for b in state_order:
            if a in centroids and b in centroids:
                distance = float(np.linalg.norm(centroids[a] - centroids[b]))
            else:
                distance = ""
            distance_rows.append({"state_a": a, "state_b": b, "centroid_distance": distance})

    transitions = []
    for prev_state, next_state in zip(state_order[:-1], state_order[1:], strict=False):
        if prev_state in centroids and next_state in centroids:
            transitions.append((prev_state, next_state, centroids[next_state] - centroids[prev_state]))
    for (a0, a1, v1), (b0, b1, v2) in zip(transitions[:-1], transitions[1:], strict=False):
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = ""
        if denom > 0:
            cos_value = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_value)))
        angle_rows.append({"transition_a": f"{a0}->{a1}", "transition_b": f"{b0}->{b1}", "angle_degrees": angle})

    return centroid_rows, distance_rows, angle_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_manifold(
    path: Path,
    embedding: np.ndarray,
    explained: np.ndarray,
    distance_rows: list[dict],
    state_labels: np.ndarray,
    state_order: list[str],
    title: str,
    method: str,
    connect_time_order: bool,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), marker="o", linewidth=1.5)
    ax1.set_xlabel("PC")
    ax1.set_ylabel("Cumulative explained variance")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("PCA scree")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(2, 2, 2)
    if connect_time_order:
        ax2.plot(
            embedding[:, 0],
            embedding[:, 1],
            color="#111827",
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
    for state in state_order:
        mask = state_labels == state
        if mask.any():
            ax2.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=14,
                alpha=0.75,
                label=state,
                color=STATE_COLORS.get(state),
                zorder=2,
            )
    centroids = []
    for state in state_order:
        mask = state_labels == state
        if mask.any():
            centroids.append(embedding[mask, :2].mean(axis=0))
    for start, end in zip(centroids[:-1], centroids[1:], strict=False):
        ax2.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4})
    suffix = " with time-ordered path" if connect_time_order else ""
    ax2.set_title(f"2D {method.upper()} state map{suffix}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    for state in state_order:
        mask = state_labels == state
        if mask.any():
            pts = embedding[mask]
            ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.0, color=STATE_COLORS.get(state), label=state)
            step = max(1, len(pts) // 12)
            deltas = np.diff(pts, axis=0)
            if len(deltas):
                ax3.quiver(
                    pts[:-1:step, 0],
                    pts[:-1:step, 1],
                    pts[:-1:step, 2],
                    deltas[::step, 0],
                    deltas[::step, 1],
                    deltas[::step, 2],
                    length=1.0,
                    normalize=False,
                    color=STATE_COLORS.get(state),
                    alpha=0.6,
                )
    ax3.set_title("3D temporal trajectory")

    ax4 = fig.add_subplot(2, 2, 4)
    dist = np.full((len(state_order), len(state_order)), np.nan)
    index = {state: i for i, state in enumerate(state_order)}
    for row in distance_rows:
        value = row["centroid_distance"]
        if value != "":
            dist[index[row["state_a"]], index[row["state_b"]]] = float(value)
    im = ax4.imshow(dist, cmap="viridis")
    ax4.set_xticks(range(len(state_order)), state_order, rotation=35, ha="right")
    ax4.set_yticks(range(len(state_order)), state_order)
    for i in range(len(state_order)):
        for j in range(len(state_order)):
            if np.isfinite(dist[i, j]):
                ax4.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax4.set_title("Centroid distance matrix")
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_file(npz_path: Path, out_dir: Path, args: argparse.Namespace, states: list[tuple[str, float, float]]) -> dict:
    population, times, meta = load_shifted_population(npz_path, args.min_amplitude, args.smooth_sigma_bins)
    state_order = [name for name, _, _ in states]
    state_labels = assign_states(times, states)
    assigned_mask = state_labels != "unassigned"

    rel_base = npz_path.stem
    file_dir = out_dir / "files" / rel_base
    file_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "file": npz_path.name,
        "n_total_fibers": meta["n_total_fibers"],
        "n_kept_fibers": meta["n_kept_fibers"],
        "mean_amplitude": meta["mean_amplitude"],
        "method": args.method,
        "status": "ok",
        "plot": str(file_dir.relative_to(out_dir) / "manifold.png"),
        "embedding_csv": str(file_dir.relative_to(out_dir) / "embedding.csv"),
    }
    if population.shape[1] < args.min_fibers:
        summary["status"] = "skipped_too_few_fibers"
        return summary
    if assigned_mask.sum() < 10:
        summary["status"] = "skipped_too_few_assigned_timepoints"
        return summary

    embedding, _, explained, actual_method = embed_population(
        population[assigned_mask],
        args.method,
        args.n_pcs,
        None if args.no_seed else args.seed,
        args.umap_neighbors,
        args.umap_min_dist,
    )
    assigned_times = times[assigned_mask]
    assigned_states = state_labels[assigned_mask]

    centroid_rows, distance_rows, angle_rows = compute_metrics(embedding, assigned_states, state_order)
    summary["method"] = actual_method
    summary["pca_cumulative_variance"] = float(np.sum(explained))

    embedding_rows = [
        {
            "time_day": float(time),
            "state": str(state),
            "dim1": float(point[0]),
            "dim2": float(point[1]),
            "dim3": float(point[2]),
        }
        for time, state, point in zip(assigned_times, assigned_states, embedding, strict=False)
    ]
    write_csv(file_dir / "embedding.csv", embedding_rows, ["time_day", "state", "dim1", "dim2", "dim3"])
    write_csv(file_dir / "state_centroids.csv", centroid_rows, ["state", "n_timepoints", "centroid_x", "centroid_y", "centroid_z", "within_state_spread"])
    write_csv(file_dir / "state_distances.csv", distance_rows, ["state_a", "state_b", "centroid_distance"])
    write_csv(file_dir / "transition_angles.csv", angle_rows, ["transition_a", "transition_b", "angle_degrees"])

    title = f"{npz_path.name} | fibers kept {meta['n_kept_fibers']}/{meta['n_total_fibers']}"
    plot_manifold(
        file_dir / "manifold.png",
        embedding,
        explained,
        distance_rows,
        assigned_states,
        state_order,
        title,
        actual_method,
        args.connect_time_order,
    )
    return summary


def write_index(out_dir: Path, summaries: list[dict], args: argparse.Namespace, states: list[tuple[str, float, float]]) -> None:
    rows = []
    for row in summaries:
        plot_cell = ""
        if row["status"] == "ok":
            plot = html.escape(row["plot"])
            plot_cell = f'<a href="{plot}"><img src="{plot}" loading="lazy" alt="{html.escape(row["file"])} manifold"></a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['file'])}</td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{row['n_kept_fibers']}/{row['n_total_fibers']}</td>"
            f"<td>{html.escape(str(row.get('method', '')))}</td>"
            f"<td>{html.escape(str(row.get('pca_cumulative_variance', '')))}</td>"
            f"<td>{plot_cell}</td>"
            "</tr>"
        )
    state_text = ", ".join(f"{name} [{start:g}, {end:g}]" for name, start, end in states)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Brain-wide Population Manifolds</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f3f4f6; }}
    img {{ width: 520px; max-width: 100%; }}
    .meta {{ color: #4b5563; margin-bottom: 14px; }}
  </style>
</head>
<body>
  <h1>Brain-wide Population Manifolds</h1>
  <div class="meta">
    Method: {html.escape(args.method)} |
    Min amplitude: {args.min_amplitude:g} |
    State windows: {html.escape(state_text)}
  </div>
  <table>
    <thead>
      <tr><th>File</th><th>Status</th><th>Kept fibers</th><th>Method</th><th>PCA cumulative variance</th><th>Plot</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainwide-raw-dir", type=Path, default=Path("data/raw/brain_wide"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rebuild-index-only", type=Path, default=None, help="Rebuild index.html for an existing output directory.")
    parser.add_argument("--states", default=DEFAULT_STATES, help="Comma-separated name:start_day:end_day windows.")
    parser.add_argument("--method", choices=["pca", "umap", "tsne"], default="pca")
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--min-amplitude", type=float, default=0.2)
    parser.add_argument("--min-fibers", type=int, default=5)
    parser.add_argument("--smooth-sigma-bins", type=float, default=3.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-seed", action="store_true", help="Disable fixed random seed. For UMAP this allows parallelism but results are not exactly reproducible.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--file", action="append", default=None, help="Process only matching NPZ filename. Can be used more than once.")
    parser.add_argument("--connect-time-order", action="store_true", help="Draw a line connecting all 2D manifold points in chronological order.")
    args = parser.parse_args()

    states = parse_states(args.states)
    if args.rebuild_index_only is not None:
        out_dir = args.rebuild_index_only
        with (out_dir / "summary.csv").open(newline="") as f:
            summaries = list(csv.DictReader(f))
        write_index(out_dir, summaries, args, states)
        print(f"rebuilt index: {out_dir / 'index.html'}")
        return

    out_dir = args.out_dir or Path("outputs/reports/brainwide_manifolds") / f"{datetime.now():%Y%m%d_%H%M%S}_brainwide_population_manifolds"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(args.brainwide_raw_dir.glob("*.npz"))
    if args.file:
        wanted = {Path(name).name for name in args.file}
        npz_paths = [path for path in npz_paths if path.name in wanted]
        missing = sorted(wanted - {path.name for path in npz_paths})
        if missing:
            raise SystemExit(f"Requested NPZ file(s) not found: {', '.join(missing)}")
    if args.max_files is not None:
        npz_paths = npz_paths[: args.max_files]
    summaries = []
    for i, npz_path in enumerate(npz_paths, start=1):
        if i % 100 == 0 or i == 1:
            print(f"processing {i}/{len(npz_paths)}: {npz_path.name}", flush=True)
        summaries.append(process_file(npz_path, out_dir, args, states))

    write_csv(
        out_dir / "summary.csv",
        summaries,
        ["file", "n_total_fibers", "n_kept_fibers", "mean_amplitude", "method", "status", "plot", "embedding_csv", "pca_cumulative_variance"],
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "brainwide_raw_dir": str(args.brainwide_raw_dir),
                "method": args.method,
                "n_pcs": args.n_pcs,
                "min_amplitude": args.min_amplitude,
                "min_fibers": args.min_fibers,
                "smooth_sigma_bins": args.smooth_sigma_bins,
                "states": [{"name": name, "start_day": start, "end_day": end} for name, start, end in states],
                "n_files": len(npz_paths),
            },
            indent=2,
        )
    )
    write_index(out_dir, summaries, args, states)
    print(f"out: {out_dir}")
    ok = sum(1 for row in summaries if row["status"] == "ok")
    print(f"files ok: {ok}/{len(summaries)}")


if __name__ == "__main__":
    main()
