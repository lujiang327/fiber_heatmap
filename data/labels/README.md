# Manual Label Store

This directory is the project-owned source of truth for human-reviewed motif labels.

## Layout

- `review_rounds/`: raw reviewed CSV exports from each manual review round.
- `label_manifest.csv`: one row per reviewed CSV, recording provenance and intent.

## Current Label Contract

Each reviewed CSV should include these identifier columns:

- `source_file`
- `source_path`
- `trace_index`
- `fiber_id`
- `reviewed_label`

Optional but useful columns:

- `review_index`
- `predicted_label`
- `prediction_confidence`
- `notes`
- `baseline_9p5_10d`
- `shifted_max`
- `shifted_min`
- `shifted_amplitude`

Training scripts should read labels from this directory rather than from `~/Downloads`.
