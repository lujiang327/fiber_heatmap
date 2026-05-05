#!/usr/bin/env python3
"""Train a NumPy MLP motif classifier and export model-review candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from export_hpc_label_review import collect_candidates, load_labels
from train_hpc_round_review import (
    build_feature_matrix,
    candidate_key,
    read_reviewed_labels,
    write_review_csv,
    write_review_html,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def stratified_split(y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in np.unique(y):
        idxs = np.flatnonzero(y == label)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction))) if len(idxs) >= 3 else 1
        n_val = min(n_val, max(len(idxs) - 1, 1))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def init_model(n_features: int, n_hidden: int, n_classes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden)),
        "b1": np.zeros(n_hidden, dtype=np.float64),
        "w2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_classes)),
        "b2": np.zeros(n_classes, dtype=np.float64),
    }


def forward(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_pre = x @ params["w1"] + params["b1"]
    hidden = relu(hidden_pre)
    logits = hidden @ params["w2"] + params["b2"]
    return hidden_pre, hidden, logits


def accuracy(params: dict, x: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(forward(params, x)[2], axis=1)
    return float(np.mean(pred == y))


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict, list[dict]]:
    n_features = x_train.shape[1]
    n_classes = int(y_train.max()) + 1
    params = init_model(n_features, n_hidden, n_classes, seed)
    y_oh = one_hot(y_train, n_classes)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1.0))
    sample_weights = class_weights[y_train]
    sample_weights = sample_weights / sample_weights.mean()

    adam = {key: {"m": np.zeros_like(value), "v": np.zeros_like(value)} for key, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history: list[dict] = []

    best_params = {key: value.copy() for key, value in params.items()}
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        hidden_pre, hidden, logits = forward(params, x_train)
        probs = softmax(logits)
        grad_logits = (probs - y_oh) * sample_weights[:, None] / len(x_train)

        grads = {
            "w2": hidden.T @ grad_logits + weight_decay * params["w2"],
            "b2": grad_logits.sum(axis=0),
        }
        grad_hidden = grad_logits @ params["w2"].T
        grad_hidden[hidden_pre <= 0.0] = 0.0
        grads["w1"] = x_train.T @ grad_hidden + weight_decay * params["w1"]
        grads["b1"] = grad_hidden.sum(axis=0)

        for key in params:
            adam[key]["m"] = beta1 * adam[key]["m"] + (1 - beta1) * grads[key]
            adam[key]["v"] = beta2 * adam[key]["v"] + (1 - beta2) * (grads[key] ** 2)
            m_hat = adam[key]["m"] / (1 - beta1**epoch)
            v_hat = adam[key]["v"] / (1 - beta2**epoch)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            train_acc = accuracy(params, x_train, y_train)
            val_acc = accuracy(params, x_val, y_val)
            history.append({"epoch": epoch, "train_accuracy": train_acc, "val_accuracy": val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_params = {key: value.copy() for key, value in params.items()}

    return best_params, history


def select_review_rows(
    candidates: list[dict],
    probs: np.ndarray,
    labels: list[str],
    target_labels: list[str],
    exclude_keys: set[tuple[str, int]],
    per_class: int,
) -> list[dict]:
    top_idx = np.argmax(probs, axis=1)
    selected: list[dict] = []
    selected_keys: set[tuple[str, int]] = set()
    for label in target_labels:
        class_idx = labels.index(label)
        direct = [
            i
            for i, row in enumerate(candidates)
            if top_idx[i] == class_idx
            and candidate_key(row) not in exclude_keys
            and candidate_key(row) not in selected_keys
        ]
        direct.sort(key=lambda i: (-float(probs[i, class_idx]), candidates[i]["source_file"], candidates[i]["trace_index"]))
        chosen = direct[:per_class]
        if len(chosen) < per_class:
            backfill = [
                i
                for i, row in enumerate(candidates)
                if candidate_key(row) not in exclude_keys
                and candidate_key(row) not in selected_keys
                and i not in chosen
            ]
            backfill.sort(
                key=lambda i: (-float(probs[i, class_idx]), candidates[i]["source_file"], candidates[i]["trace_index"])
            )
            chosen.extend(backfill[: per_class - len(chosen)])
        for i in chosen[:per_class]:
            top_label = labels[int(top_idx[i])]
            row = dict(candidates[i])
            row["predicted_label"] = label
            row["model_top_label"] = top_label
            row["prediction_confidence"] = float(probs[i, class_idx])
            row["target_label_distance"] = ""
            selected.append(row)
            selected_keys.add(candidate_key(row))
    selected.sort(key=lambda row: (row["predicted_label"], -row["prediction_confidence"]))
    return selected


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_accuracy", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hpc"))
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels/hpc_reviewed_labels_merged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_exports/hpc_mlp_round"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument(
        "--target-labels",
        nargs="*",
        default=None,
        help="Optional subset of trained labels to export for review.",
    )
    args = parser.parse_args()

    all_schema_labels = load_labels(args.schema)
    reviewed_rows = read_reviewed_labels(args.labels_csv)
    candidates, times = collect_candidates(args.raw_dir)
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
        raise SystemExit("No labels matched HPC candidates.")

    trained_labels = sorted(set(y_labels))
    label_to_idx = {label: idx for idx, label in enumerate(trained_labels)}
    y = np.array([label_to_idx[label] for label in y_labels], dtype=int)

    x = build_feature_matrix(train_candidates)
    train_idx, val_idx = stratified_split(y, 0.2, args.seed)
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std < 1e-9] = 1.0
    xz = (x - mean) / std

    params, history = train_mlp(
        xz[train_idx],
        y[train_idx],
        xz[val_idx],
        y[val_idx],
        args.hidden,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.seed,
    )

    x_all = build_feature_matrix(candidates)
    probs = softmax(forward(params, (x_all - mean) / std)[2])
    exclude_keys = {candidate_key(row) for row in train_candidates}
    if args.target_labels:
        unknown_targets = [label for label in args.target_labels if label not in trained_labels]
        if unknown_targets:
            raise SystemExit(f"Target labels have no training examples: {', '.join(unknown_targets)}")
        review_labels = args.target_labels
    else:
        review_labels = trained_labels
    selected = select_review_rows(candidates, probs, trained_labels, review_labels, exclude_keys, args.per_class)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_review_csv(args.out_dir / "hpc_mlp_review_candidates.csv", selected)
    write_review_html(args.out_dir / "hpc_mlp_review.html", selected, all_schema_labels, times)
    write_history(args.out_dir / "hpc_mlp_training_history.csv", history)
    np.savez(
        args.out_dir / "hpc_mlp_model.npz",
        labels=np.array(trained_labels, dtype=object),
        mean=mean,
        std=std,
        w1=params["w1"],
        b1=params["b1"],
        w2=params["w2"],
        b2=params["b2"],
    )

    print(f"matched labeled rows: {len(train_candidates)}")
    print("training label counts:")
    for label in trained_labels:
        print(f"  {label}: {sum(np.array(y_labels, dtype=object) == label)}")
    untrained = [label for label in all_schema_labels if label not in trained_labels]
    if untrained:
        print("untrained labels with zero examples:")
        for label in untrained:
            print(f"  {label}")
    print("training history last:")
    for key, value in history[-1].items():
        print(f"  {key}: {value}")
    selected_counts = {label: sum(row["predicted_label"] == label for row in selected) for label in review_labels}
    print("selected review counts:")
    for label in review_labels:
        print(f"  {label}: {selected_counts[label]}")
    print(f"html: {args.out_dir / 'hpc_mlp_review.html'}")
    print(f"candidate csv: {args.out_dir / 'hpc_mlp_review_candidates.csv'}")
    print(f"model: {args.out_dir / 'hpc_mlp_model.npz'}")


if __name__ == "__main__":
    main()
