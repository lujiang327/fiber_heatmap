#!/usr/bin/env python3
"""Build a deduplicated training label table from reviewed label rounds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_FIELDS = [
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
    "reviewed_label",
    "notes",
    "label_round_id",
    "label_file",
]


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def trace_key(row: dict) -> tuple[str, int]:
    return str(row["source_file"]), int(row["trace_index"])


def build_dataset(labels_dir: Path, manifest_path: Path) -> list[dict]:
    manifest_rows = read_manifest(manifest_path)
    by_key: dict[tuple[str, int], dict] = {}
    for manifest_row in manifest_rows:
        label_file = manifest_row["label_file"]
        round_id = manifest_row["round_id"]
        csv_path = labels_dir / label_file
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                label = (row.get("reviewed_label") or "").strip()
                if not label:
                    continue
                normalized = {field: row.get(field, "") for field in BASE_FIELDS}
                normalized["reviewed_label"] = label
                normalized["label_round_id"] = round_id
                normalized["label_file"] = label_file
                by_key[trace_key(row)] = normalized
    return list(by_key.values())


def write_dataset(path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda row: (row["source_file"], int(row["trace_index"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--manifest", type=Path, default=Path("data/labels/label_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    args = parser.parse_args()

    rows = build_dataset(args.labels_dir, args.manifest)
    write_dataset(args.out, rows)

    counts: dict[str, int] = {}
    for row in rows:
        label = row["reviewed_label"]
        counts[label] = counts.get(label, 0) + 1
    print(f"merged labeled traces: {len(rows)}")
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{label}: {count}")
    print(f"out: {args.out}")


if __name__ == "__main__":
    main()
