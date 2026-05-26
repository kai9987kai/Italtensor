from __future__ import annotations

from typing import Any, Sequence

import numpy as np


EPSILON = 1e-12


def run_neighborhood_hardness_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    k: int = 5,
    max_rows: int = 12,
) -> dict[str, Any]:
    """Estimate local learnability with leave-one-out k-nearest-neighbor votes."""
    x, y = _validate_inputs(features, labels)
    k = _validate_positive_int(k, "k")
    max_rows = _validate_positive_int(max_rows, "max_rows")
    neighbor_count = min(k, x.shape[0] - 1)
    scaled = _standardize(x)
    distances = _pairwise_distances(scaled)
    neighbor_indices = np.argsort(distances, axis=1)[:, :neighbor_count]
    rows = [_row_score(index, y, neighbor_indices[index], distances[index]) for index in range(x.shape[0])]
    rows.sort(
        key=lambda row: (
            -float(row["hardness_score"]),
            -float(row["opposite_vote_rate"]),
            -float(row["vote_entropy"]),
            int(row["row_index"]),
        )
    )
    summary = _summary(rows, y)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "k": int(neighbor_count),
        "class_counts": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "summary": summary,
        "rows": rows[:max_rows],
    }


def format_neighborhood_hardness_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_row = summary.get("top_hard_row")
    top_text = "-" if top_row is None else str(int(top_row))
    return (
        "Neighborhood hardness: "
        f"k={int(report.get('k', 0))}, "
        f"loo_acc={float(summary.get('loo_accuracy', 0.0)):.4f}, "
        f"hard={int(summary.get('hard_row_count', 0))}, "
        f"ambiguous={int(summary.get('ambiguous_row_count', 0))}, "
        f"label_issue={int(summary.get('label_issue_candidate_count', 0))}, "
        f"top_row={top_text}"
    )


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Neighborhood hardness features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Neighborhood hardness features must be a 2D array.")
    if x.shape[0] < 6:
        raise ValueError("Neighborhood hardness needs at least six rows.")
    if x.shape[1] == 0:
        raise ValueError("Neighborhood hardness needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Neighborhood hardness features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Neighborhood hardness labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Neighborhood hardness feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Neighborhood hardness labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Neighborhood hardness requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Neighborhood hardness requires binary labels 0 or 1.")
    for class_value in (0, 1):
        if int(np.sum(y == class_value)) < 2:
            raise ValueError("Neighborhood hardness needs at least two rows per class.")
    return x, y


def _validate_positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Neighborhood hardness {name} must be positive.")
    return parsed


def _standardize(x: np.ndarray) -> np.ndarray:
    scale = np.std(x, axis=0)
    scale = np.where(scale > EPSILON, scale, 1.0)
    return (x - np.mean(x, axis=0)) / scale


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def _row_score(index: int, labels: np.ndarray, neighbors: np.ndarray, distances: np.ndarray) -> dict[str, Any]:
    label = int(labels[index])
    neighbor_labels = labels[neighbors]
    positive_vote_rate = float(np.mean(neighbor_labels == 1))
    predicted_label = 1 if positive_vote_rate >= 0.5 else 0
    opposite_vote_rate = float(np.mean(neighbor_labels != label))
    vote_margin = abs(positive_vote_rate - 0.5) * 2.0
    vote_entropy = _binary_entropy(positive_vote_rate)
    same_distances = distances[neighbors[neighbor_labels == label]]
    opposite_distances = distances[neighbors[neighbor_labels != label]]
    nearest_same = float(np.min(same_distances)) if same_distances.size else None
    nearest_opposite = float(np.min(opposite_distances)) if opposite_distances.size else None
    mean_neighbor_distance = float(np.mean(distances[neighbors]))
    misvoted = predicted_label != label
    hardness_score = float(
        np.clip(
            0.50 * opposite_vote_rate
            + 0.25 * vote_entropy
            + 0.15 * (1.0 - vote_margin)
            + 0.10 * float(misvoted),
            0.0,
            1.0,
        )
    )
    return {
        "row_index": int(index),
        "label": label,
        "predicted_label": int(predicted_label),
        "positive_vote_rate": positive_vote_rate,
        "opposite_vote_rate": opposite_vote_rate,
        "vote_margin": float(vote_margin),
        "vote_entropy": vote_entropy,
        "hardness_score": hardness_score,
        "misvoted": bool(misvoted),
        "nearest_same_distance": nearest_same,
        "nearest_opposite_distance": nearest_opposite,
        "mean_neighbor_distance": mean_neighbor_distance,
        "neighbor_indices": [int(value) for value in neighbors],
        "neighbor_labels": [int(value) for value in neighbor_labels],
        "risk_flags": _risk_flags(
            opposite_vote_rate=opposite_vote_rate,
            vote_entropy=vote_entropy,
            misvoted=bool(misvoted),
            hardness_score=hardness_score,
        ),
    }


def _binary_entropy(positive_rate: float) -> float:
    p = float(np.clip(positive_rate, EPSILON, 1.0 - EPSILON))
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(entropy)


def _risk_flags(
    *,
    opposite_vote_rate: float,
    vote_entropy: float,
    misvoted: bool,
    hardness_score: float,
) -> list[str]:
    flags: list[str] = []
    if misvoted and opposite_vote_rate >= 0.80:
        flags.append("label_issue_candidate")
    if vote_entropy >= 0.90:
        flags.append("ambiguous_neighborhood")
    if hardness_score >= 0.65:
        flags.append("hard_row")
    if opposite_vote_rate == 0.0 and vote_entropy <= 0.05:
        flags.append("locally_easy")
    return flags


def _summary(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    correct = [not bool(row["misvoted"]) for row in rows]
    hard_count = int(sum(float(row["hardness_score"]) >= 0.65 for row in rows))
    ambiguous_count = int(sum("ambiguous_neighborhood" in row["risk_flags"] for row in rows))
    label_issue_count = int(sum("label_issue_candidate" in row["risk_flags"] for row in rows))
    easy_count = int(sum("locally_easy" in row["risk_flags"] for row in rows))
    top = rows[0] if rows else None
    warnings: list[str] = []
    if label_issue_count:
        warnings.append(f"{label_issue_count} row(s) are surrounded by the opposite class")
    if hard_count:
        warnings.append(f"{hard_count} locally hard row(s)")
    return {
        "loo_accuracy": float(np.mean(correct)) if correct else 0.0,
        "loo_error_rate": float(1.0 - np.mean(correct)) if correct else 0.0,
        "hard_row_count": hard_count,
        "ambiguous_row_count": ambiguous_count,
        "label_issue_candidate_count": label_issue_count,
        "locally_easy_count": easy_count,
        "top_hard_row": None if top is None else int(top["row_index"]),
        "top_hardness_score": 0.0 if top is None else float(top["hardness_score"]),
        "mean_hardness_score": float(np.mean([row["hardness_score"] for row in rows])) if rows else 0.0,
        "label_prevalence": float(np.mean(labels)) if labels.size else 0.0,
        "warning": "; ".join(warnings) if warnings else None,
    }
