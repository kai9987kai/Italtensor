"""Lightweight row-value and curation diagnostics for small binary datasets."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


EPSILON = 1e-12


def run_data_value_scout(
    features: Any,
    labels: Any,
    *,
    k: int = 7,
    max_rows: int = 10,
) -> dict[str, Any]:
    """Estimate row support, harm, redundancy, and coverage value without retraining."""
    x, y = _validate_inputs(features, labels)
    k = max(1, int(k))
    max_rows = max(1, int(max_rows))
    neighbor_count = min(k, x.shape[0] - 1)
    scaled = _standardize(x)
    distances = _pairwise_distances(scaled)
    neighbor_indices = np.argsort(distances, axis=1)[:, :neighbor_count]
    robust_z = _robust_z_scores(x)
    tail_scores = np.max(np.abs(robust_z), axis=1)
    row_scores = _row_scores(x, y, distances, neighbor_indices, tail_scores)
    recommendations = _recommendations(row_scores)
    summary = _summary(row_scores, recommendations)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "k": int(neighbor_count),
        "dataset_fingerprint": data_value_dataset_fingerprint(x, y),
        "summary": summary,
        "recommendations": recommendations,
        "high_value_rows": _top_rows(row_scores, "value_score", max_rows),
        "review_rows": _top_rows(row_scores, "review_score", max_rows),
        "redundant_rows": _top_rows(row_scores, "redundancy_score", max_rows),
        "coverage_rows": _top_rows(row_scores, "coverage_score", max_rows),
        "rows": _top_rows(row_scores, "curation_priority", max_rows),
    }


def format_data_value_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Data value scout: "
        f"verdict={summary.get('verdict', '-')}, "
        f"priority={summary.get('priority', '-')}, "
        f"review={int(summary.get('review_row_count', 0) or 0)}, "
        f"redundant={int(summary.get('redundant_row_count', 0) or 0)}, "
        f"anchors={int(summary.get('high_value_row_count', 0) or 0)}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def data_value_dataset_fingerprint(features: Any, labels: Any) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_inputs(features: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Data value scout features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Data value scout needs a 2D feature matrix.")
    if x.shape[0] < 6:
        raise ValueError("Data value scout needs at least six labeled rows.")
    if x.shape[1] < 1:
        raise ValueError("Data value scout needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Data value scout features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Data value scout labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Data value scout feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Data value scout labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Data value scout requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Data value scout requires binary labels 0 or 1.")
    for label in (0, 1):
        if int(np.sum(y == label)) < 2:
            raise ValueError("Data value scout needs at least two rows per class.")
    return x, y


def _standardize(x: np.ndarray) -> np.ndarray:
    scale = np.std(x, axis=0)
    scale = np.where(scale > EPSILON, scale, 1.0)
    return (x - np.mean(x, axis=0)) / scale


def _robust_z_scores(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    scale = np.where(mad > EPSILON, 1.4826 * mad, np.std(x, axis=0))
    scale = np.where(scale > EPSILON, scale, 1.0)
    return (x - median) / scale


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def _row_scores(
    x: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    tail_scores: np.ndarray,
) -> list[dict[str, Any]]:
    neighbor_support, neighbor_harm = _neighbor_contributions(y, neighbor_indices)
    mean_finite_distance = float(np.mean(distances[np.isfinite(distances)]))
    rows: list[dict[str, Any]] = []
    for index in range(x.shape[0]):
        neighbors = neighbor_indices[index]
        neighbor_labels = y[neighbors]
        same_mask = neighbor_labels == y[index]
        same_fraction = float(np.mean(same_mask))
        opposite_fraction = 1.0 - same_fraction
        nearest_same_distance = _nearest_distance(distances[index], y, int(y[index]))
        nearest_opposite_distance = _nearest_distance(distances[index], y, 1 - int(y[index]))
        mean_same_distance = _mean_distance(distances[index], y, int(y[index]))
        same_duplicates = int(np.sum((distances[index] <= 1e-6) & (y == y[index])))
        conflict_duplicates = int(np.sum((distances[index] <= 1e-6) & (y != y[index])))
        isolation_raw = mean_same_distance / max(mean_finite_distance, EPSILON) - 1.0
        isolation_score = _bounded_positive(isolation_raw)
        boundary_score = float(1.0 - abs(nearest_same_distance - nearest_opposite_distance) / (nearest_same_distance + nearest_opposite_distance + EPSILON))
        boundary_score = float(np.clip(boundary_score, 0.0, 1.0))
        support_score = float(neighbor_support[index])
        harm_score = float(neighbor_harm[index])
        redundancy_score = float(np.clip(0.45 * same_duplicates + 0.30 * same_fraction + 0.25 * max(0.0, 1.0 - isolation_score), 0.0, 1.0))
        coverage_score = float(np.clip(0.55 * min(float(tail_scores[index]) / 4.5, 1.0) + 0.45 * isolation_score, 0.0, 1.0))
        review_score = float(
            np.clip(
                0.42 * opposite_fraction
                + 0.24 * boundary_score
                + 0.22 * harm_score
                + 0.12 * min(conflict_duplicates, 1),
                0.0,
                1.0,
            )
        )
        value_score = float(
            np.clip(
                0.48 * support_score
                + 0.20 * same_fraction
                + 0.18 * coverage_score
                + 0.14 * boundary_score
                - 0.35 * review_score
                - 0.18 * redundancy_score,
                0.0,
                1.0,
            )
        )
        curation_priority = float(
            np.clip(
                max(value_score, review_score, redundancy_score * 0.72, coverage_score * 0.82),
                0.0,
                1.0,
            )
        )
        rows.append(
            {
                "row_index": int(index),
                "label": int(y[index]),
                "value_score": value_score,
                "support_score": support_score,
                "harm_score": harm_score,
                "review_score": review_score,
                "redundancy_score": redundancy_score,
                "coverage_score": coverage_score,
                "curation_priority": curation_priority,
                "same_neighbor_fraction": same_fraction,
                "opposite_neighbor_fraction": opposite_fraction,
                "nearest_same_distance": float(nearest_same_distance),
                "nearest_opposite_distance": float(nearest_opposite_distance),
                "tail_score": float(tail_scores[index]),
                "same_duplicate_count": same_duplicates,
                "conflict_duplicate_count": conflict_duplicates,
                "neighbor_indices": [int(value) for value in neighbors[:5]],
                "risk_flags": _risk_flags(
                    value_score=value_score,
                    review_score=review_score,
                    redundancy_score=redundancy_score,
                    coverage_score=coverage_score,
                    conflict_duplicates=conflict_duplicates,
                ),
                "recommended_action": _recommended_action(
                    value_score=value_score,
                    review_score=review_score,
                    redundancy_score=redundancy_score,
                    coverage_score=coverage_score,
                    conflict_duplicates=conflict_duplicates,
                ),
                "feature_preview": [float(value) for value in x[index, :8]],
            }
        )
    return rows


def _neighbor_contributions(y: np.ndarray, neighbor_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.zeros(y.shape[0], dtype=np.float64)
    harm = np.zeros(y.shape[0], dtype=np.float64)
    if neighbor_indices.shape[1] == 0:
        return support, harm
    weights = 1.0 / (np.arange(neighbor_indices.shape[1], dtype=np.float64) + 1.0)
    total = float(np.sum(weights))
    for query_index, neighbors in enumerate(neighbor_indices):
        for rank, train_index in enumerate(neighbors):
            contribution = weights[rank] / total
            if y[train_index] == y[query_index]:
                support[train_index] += contribution
            else:
                harm[train_index] += contribution
    denom = np.maximum(support + harm, EPSILON)
    support_score = support / denom
    harm_score = harm / denom
    return support_score, harm_score


def _nearest_distance(row_distances: np.ndarray, y: np.ndarray, label: int) -> float:
    values = row_distances[y == label]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.min(finite))


def _mean_distance(row_distances: np.ndarray, y: np.ndarray, label: int) -> float:
    values = row_distances[y == label]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.mean(finite))


def _bounded_positive(value: float) -> float:
    clipped = max(0.0, float(value))
    return clipped / (1.0 + clipped)


def _risk_flags(
    *,
    value_score: float,
    review_score: float,
    redundancy_score: float,
    coverage_score: float,
    conflict_duplicates: int,
) -> list[str]:
    flags: list[str] = []
    if review_score >= 0.58 or conflict_duplicates:
        flags.append("review_or_relabel")
    if redundancy_score >= 0.72 and review_score < 0.45:
        flags.append("redundant_anchor")
    if coverage_score >= 0.35 and review_score < 0.55:
        flags.append("rare_coverage")
    if value_score >= 0.50 and review_score < 0.50:
        flags.append("high_value_anchor")
    if not flags:
        flags.append("ordinary_support")
    return flags


def _recommended_action(
    *,
    value_score: float,
    review_score: float,
    redundancy_score: float,
    coverage_score: float,
    conflict_duplicates: int,
) -> str:
    if conflict_duplicates:
        return "Resolve exact-feature conflicting labels before using this row as evidence."
    if review_score >= 0.58:
        return "Review the label or collect nearby examples before promotion."
    if redundancy_score >= 0.72 and value_score < 0.50:
        return "Keep one representative row and consider downweighting duplicates."
    if coverage_score >= 0.35:
        return "Keep as a rare coverage case and add a canary or schema note."
    if value_score >= 0.50:
        return "Keep as a high-value class anchor for validation and canary examples."
    return "Keep as ordinary support; no immediate curation action."


def _recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_count = sum(1 for row in rows if row["review_score"] >= 0.58 or row["conflict_duplicate_count"])
    redundant_count = sum(1 for row in rows if row["redundancy_score"] >= 0.72 and row["review_score"] < 0.45)
    coverage_count = sum(1 for row in rows if row["coverage_score"] >= 0.35 and row["review_score"] < 0.55)
    anchor_count = sum(1 for row in rows if row["value_score"] >= 0.50 and row["review_score"] < 0.50)
    recs: list[dict[str, Any]] = []

    def add(score: float, priority: str, category: str, title: str, reason: str, action: str) -> None:
        recs.append(
            {
                "priority": priority,
                "priority_score": float(score),
                "category": category,
                "title": title,
                "reason": reason,
                "action": action,
            }
        )

    if review_count:
        add(
            91.0,
            "high" if review_count >= 3 else "medium",
            "row_review",
            "Review possible harmful training rows",
            f"{review_count} row(s) have high opposing-neighbor or conflict evidence.",
            "Inspect review rows first; fix labels, collect nearby rows, or quarantine them before promotion.",
        )
    if redundant_count:
        add(
            68.0,
            "medium",
            "deduplication",
            "Reduce redundant anchor weight",
            f"{redundant_count} row(s) look like repeated same-label anchors.",
            "Keep representative examples, but avoid letting duplicates dominate validation or review queues.",
        )
    if coverage_count:
        add(
            62.0,
            "medium",
            "coverage",
            "Preserve rare valid coverage rows",
            f"{coverage_count} row(s) look rare but locally plausible.",
            "Keep these rows as coverage canaries and collect more examples around valid rare regions.",
        )
    if anchor_count:
        add(
            48.0,
            "low",
            "anchors",
            "Promote high-value anchors into checks",
            f"{anchor_count} row(s) strongly support nearby same-label examples.",
            "Use the clearest anchors as preset prediction examples or canaries.",
        )
    if not recs:
        add(
            20.0,
            "low",
            "ready",
            "No urgent row-value action",
            "Neighbor support, redundancy, and review evidence look stable.",
            "Proceed with validation, model search, and external holdout checks.",
        )

    recs.sort(key=lambda item: (-float(item["priority_score"]), item["category"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs


def _summary(rows: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    review_count = sum(1 for row in rows if row["review_score"] >= 0.58 or row["conflict_duplicate_count"])
    redundant_count = sum(1 for row in rows if row["redundancy_score"] >= 0.72 and row["review_score"] < 0.45)
    coverage_count = sum(1 for row in rows if row["coverage_score"] >= 0.35 and row["review_score"] < 0.55)
    anchor_count = sum(1 for row in rows if row["value_score"] >= 0.50 and row["review_score"] < 0.50)
    high_priority_count = sum(1 for item in recommendations if item["priority"] == "high")
    medium_priority_count = sum(1 for item in recommendations if item["priority"] == "medium")
    readiness = 100.0 - 12.0 * high_priority_count - 6.0 * medium_priority_count
    readiness -= min(16.0, review_count * 2.5)
    readiness -= min(8.0, redundant_count * 0.8)
    readiness = max(0.0, min(100.0, readiness))
    if high_priority_count:
        verdict = "curate_before_model_selection"
        priority = "high"
    elif medium_priority_count:
        verdict = "targeted_curation_recommended"
        priority = "medium"
    else:
        verdict = "row_values_ready_for_experiment"
        priority = "low"
    top = recommendations[0] if recommendations else {}
    return {
        "verdict": verdict,
        "priority": priority,
        "readiness_score": round(float(readiness), 1),
        "review_row_count": int(review_count),
        "redundant_row_count": int(redundant_count),
        "coverage_row_count": int(coverage_count),
        "high_value_row_count": int(anchor_count),
        "mean_value_score": float(np.mean([row["value_score"] for row in rows])),
        "max_review_score": float(max(row["review_score"] for row in rows)),
        "recommendation_count": int(len(recommendations)),
        "recommended_next_step": top.get("action"),
    }


def _top_rows(rows: list[dict[str, Any]], key: str, max_rows: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row[key]), int(row["row_index"])))
    output = []
    for rank, item in enumerate(ranked[:max_rows], start=1):
        row = dict(item)
        row["rank"] = rank
        output.append(row)
    return output
