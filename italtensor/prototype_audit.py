from __future__ import annotations

from typing import Any, Sequence

import numpy as np


EPSILON = 1e-12


def run_prototype_audit(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    k: int = 5,
    max_rows: int = 8,
) -> dict[str, Any]:
    """Rank representative, boundary, isolated, and locally contradictory rows."""
    x, y = _validate_inputs(features, labels)
    k = _validate_positive_int(k, "k")
    max_rows = _validate_positive_int(max_rows, "max_rows")
    neighbor_count = min(k, x.shape[0] - 1)
    scaled = _standardize(x)
    distances = _pairwise_distances(scaled)
    neighbor_indices = np.argsort(distances, axis=1)[:, :neighbor_count]

    base_rows = _base_neighborhood_rows(y, distances, neighbor_indices)
    class_medians = {
        class_value: _positive_median([row["same_mean_distance"] for row in base_rows if row["label"] == class_value])
        for class_value in (0, 1)
    }
    rows = [_score_row(row, class_medians[int(row["label"])]) for row in base_rows]

    prototypes = _select_prototypes(rows, max_rows=max_rows)
    boundary_rows = _top_rows(rows, "boundary_score", max_rows=max_rows)
    isolated_rows = _top_rows(rows, "isolation_score", max_rows=max_rows)
    label_contradictions = _top_rows(rows, "label_contradiction_score", max_rows=max_rows)
    review_rows = _review_rows(rows, max_rows=max_rows)
    summary = _summary(prototypes, boundary_rows, isolated_rows, label_contradictions, rows)

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "k": int(neighbor_count),
        "class_counts": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "summary": summary,
        "prototypes": prototypes,
        "boundary_rows": boundary_rows,
        "isolated_rows": isolated_rows,
        "label_contradictions": label_contradictions,
        "rows": review_rows,
    }


def format_prototype_audit_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_boundary = summary.get("top_boundary_row")
    top_boundary_text = "-" if top_boundary is None else str(int(top_boundary))
    top_contradiction = summary.get("top_label_contradiction_row")
    top_contradiction_text = "-" if top_contradiction is None else str(int(top_contradiction))
    return (
        "Prototype audit: "
        f"k={int(report.get('k', 0))}, "
        f"prototypes={int(summary.get('prototype_count', 0))}, "
        f"boundary={int(summary.get('boundary_row_count', 0))}, "
        f"isolated={int(summary.get('isolated_row_count', 0))}, "
        f"contradictions={int(summary.get('label_contradiction_count', 0))}, "
        f"top_boundary={top_boundary_text}, "
        f"top_contradiction={top_contradiction_text}"
    )


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prototype audit features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Prototype audit features must be a 2D array.")
    if x.shape[0] < 6:
        raise ValueError("Prototype audit needs at least six rows.")
    if x.shape[1] == 0:
        raise ValueError("Prototype audit needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Prototype audit features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prototype audit labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Prototype audit feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Prototype audit labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Prototype audit requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Prototype audit requires binary labels 0 or 1.")
    for class_value in (0, 1):
        if int(np.sum(y == class_value)) < 2:
            raise ValueError("Prototype audit needs at least two rows per class.")
    return x, y


def _validate_positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Prototype audit {name} must be positive.")
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


def _base_neighborhood_rows(
    labels: np.ndarray,
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(labels.shape[0]):
        label = int(labels[index])
        same_mask = labels == label
        same_mask[index] = False
        opposite_mask = labels != label
        same_distances = np.sort(distances[index, same_mask])
        opposite_distances = np.sort(distances[index, opposite_mask])
        neighbors = neighbor_indices[index]
        neighbor_labels = labels[neighbors]
        rows.append(
            {
                "row_index": int(index),
                "label": label,
                "nearest_same_distance": float(same_distances[0]),
                "nearest_opposite_distance": float(opposite_distances[0]),
                "same_mean_distance": float(np.mean(same_distances[: max(1, min(same_distances.shape[0], neighbors.shape[0]))])),
                "local_opposite_fraction": float(np.mean(neighbor_labels != label)),
                "neighbor_indices": [int(value) for value in neighbors],
                "neighbor_labels": [int(value) for value in neighbor_labels],
            }
        )
    return rows


def _score_row(row: dict[str, Any], class_median_same_distance: float) -> dict[str, Any]:
    nearest_same = float(row["nearest_same_distance"])
    nearest_opposite = float(row["nearest_opposite_distance"])
    same_mean = float(row["same_mean_distance"])
    local_opposite_fraction = float(row["local_opposite_fraction"])
    opposite_closeness = nearest_same / (nearest_same + nearest_opposite + EPSILON)
    same_margin = nearest_opposite / (nearest_same + nearest_opposite + EPSILON)
    density_score = class_median_same_distance / (same_mean + class_median_same_distance + EPSILON)
    isolation_raw = same_mean / (class_median_same_distance + EPSILON) - 1.0

    prototype_score = np.clip(0.45 * (1.0 - local_opposite_fraction) + 0.35 * density_score + 0.20 * same_margin, 0.0, 1.0)
    boundary_score = np.clip(0.65 * local_opposite_fraction + 0.35 * opposite_closeness, 0.0, 1.0)
    isolation_score = _bounded_positive(isolation_raw)
    label_contradiction_score = np.clip(0.70 * local_opposite_fraction + 0.30 * opposite_closeness, 0.0, 1.0)

    scored = dict(row)
    scored.update(
        {
            "prototype_score": float(prototype_score),
            "boundary_score": float(boundary_score),
            "isolation_score": float(isolation_score),
            "label_contradiction_score": float(label_contradiction_score),
            "risk_flags": _risk_flags(
                prototype_score=float(prototype_score),
                boundary_score=float(boundary_score),
                isolation_score=float(isolation_score),
                label_contradiction_score=float(label_contradiction_score),
                local_opposite_fraction=local_opposite_fraction,
            ),
        }
    )
    return scored


def _bounded_positive(value: float) -> float:
    clipped = max(0.0, float(value))
    return clipped / (1.0 + clipped)


def _risk_flags(
    *,
    prototype_score: float,
    boundary_score: float,
    isolation_score: float,
    label_contradiction_score: float,
    local_opposite_fraction: float,
) -> list[str]:
    flags: list[str] = []
    if prototype_score >= 0.65 and local_opposite_fraction <= 0.20:
        flags.append("class_prototype")
    if boundary_score >= 0.35:
        flags.append("class_boundary")
    if isolation_score >= 0.45:
        flags.append("isolated_within_class")
    if label_contradiction_score >= 0.55:
        flags.append("possible_label_contradiction")
    return flags


def _positive_median(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value) and value >= 0.0], dtype=np.float64)
    if finite.size == 0:
        return 1.0
    return max(float(np.median(finite)), EPSILON)


def _select_prototypes(rows: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    per_class = max(1, min(4, max_rows // 2 if max_rows > 1 else 1))
    selected: list[dict[str, Any]] = []
    for class_value in (0, 1):
        class_rows = [row for row in rows if int(row["label"]) == class_value]
        class_rows.sort(
            key=lambda row: (
                -float(row["prototype_score"]),
                float(row["local_opposite_fraction"]),
                float(row["same_mean_distance"]),
                int(row["row_index"]),
            )
        )
        selected.extend(_compact_row(row) for row in class_rows[:per_class])
    return selected


def _top_rows(rows: list[dict[str, Any]], key: str, *, max_rows: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[key]),
            -float(row["local_opposite_fraction"]),
            int(row["row_index"]),
        ),
    )
    return [_compact_row(row) for row in ordered[:max_rows]]


def _review_rows(rows: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -max(
                float(row["boundary_score"]),
                float(row["isolation_score"]),
                float(row["label_contradiction_score"]),
            ),
            int(row["row_index"]),
        ),
    )
    return [_compact_row(row) for row in ordered[:max_rows]]


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "row_index",
        "label",
        "prototype_score",
        "boundary_score",
        "isolation_score",
        "label_contradiction_score",
        "nearest_same_distance",
        "nearest_opposite_distance",
        "local_opposite_fraction",
        "neighbor_indices",
        "neighbor_labels",
        "risk_flags",
    )
    return {key: row[key] for key in keys}


def _summary(
    prototypes: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    isolated_rows: list[dict[str, Any]],
    label_contradictions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    boundary_count = int(sum(float(row["boundary_score"]) >= 0.35 for row in rows))
    isolated_count = int(sum(float(row["isolation_score"]) >= 0.45 for row in rows))
    contradiction_count = int(sum(float(row["label_contradiction_score"]) >= 0.55 for row in rows))
    warnings: list[str] = []
    if contradiction_count:
        warnings.append(f"{contradiction_count} row(s) have mostly opposite-label neighbors")
    if isolated_count:
        warnings.append(f"{isolated_count} row(s) are sparse within their own class")
    return {
        "prototype_count": int(len(prototypes)),
        "boundary_row_count": boundary_count,
        "isolated_row_count": isolated_count,
        "label_contradiction_count": contradiction_count,
        "top_boundary_row": boundary_rows[0]["row_index"] if boundary_rows else None,
        "top_boundary_score": float(boundary_rows[0]["boundary_score"]) if boundary_rows else 0.0,
        "top_isolated_row": isolated_rows[0]["row_index"] if isolated_rows else None,
        "top_isolation_score": float(isolated_rows[0]["isolation_score"]) if isolated_rows else 0.0,
        "top_label_contradiction_row": label_contradictions[0]["row_index"] if label_contradictions else None,
        "top_label_contradiction_score": (
            float(label_contradictions[0]["label_contradiction_score"]) if label_contradictions else 0.0
        ),
        "mean_boundary_score": float(np.mean([row["boundary_score"] for row in rows])) if rows else 0.0,
        "mean_isolation_score": float(np.mean([row["isolation_score"] for row in rows])) if rows else 0.0,
        "warning": "; ".join(warnings) if warnings else None,
    }
