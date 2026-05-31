from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_capacity_planner(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    capacity_grid: Sequence[float] | None = None,
    benefit_tp: float = 5.0,
    cost_fp: float = 1.0,
    cost_action: float = 0.25,
    max_rows: int = 12,
) -> dict[str, Any]:
    """Simulate finite review/action budgets over rows ranked by model probability."""
    x, y = _validate_inputs(features, labels)
    benefit_tp = float(benefit_tp)
    cost_fp = float(cost_fp)
    cost_action = float(cost_action)
    max_rows = max(1, int(max_rows))
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Capacity planner preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    order = np.lexsort((np.arange(probabilities.shape[0]), -probabilities))
    grid = _capacity_grid(capacity_grid, x.shape[0])
    points = [
        _capacity_point(
            fraction,
            order,
            y,
            probabilities,
            benefit_tp=benefit_tp,
            cost_fp=cost_fp,
            cost_action=cost_action,
        )
        for fraction in grid
    ]
    best_utility = max(points, key=lambda point: (float(point["net_utility"]), float(point["precision_at_k"]), -float(point["capacity_fraction"])))
    best_f1 = max(points, key=lambda point: (float(point["f1_at_k"]), float(point["recall_captured"]), float(point["precision_at_k"])))
    top_rows = _top_rows(order, y, probabilities, max_rows=max_rows)
    summary = _summary(points, best_utility, best_f1, y)
    recommendations = _recommendations(summary, best_utility)
    summary["recommendation"] = recommendations[0]["action"] if recommendations else None
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "positive_count": int(np.sum(y == 1)),
        "capacity_grid": [float(value) for value in grid],
        "capacity_fractions": [float(value) for value in grid],
        "benefit_tp": benefit_tp,
        "cost_fp": cost_fp,
        "cost_action": cost_action,
        "utility_model": {
            "true_positive_value": benefit_tp,
            "false_positive_cost": cost_fp,
            "review_cost": cost_action,
        },
        "dataset_fingerprint": capacity_planner_dataset_fingerprint(x, y),
        "summary": summary,
        "best_utility": best_utility,
        "best_f1": best_f1,
        "points": points,
        "capacity_points": points,
        "top_rows": top_rows,
        "recommendations": recommendations,
    }


def format_capacity_planner_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Capacity planner: "
        f"verdict={summary.get('verdict', '-')}, "
        f"best_budget={float(summary.get('best_capacity_fraction', 0.0)):.3f}, "
        f"precision={float(summary.get('best_precision_at_k', 0.0)):.4f}, "
        f"recall={float(summary.get('best_recall_captured', 0.0)):.4f}, "
        f"lift={float(summary.get('best_lift', 0.0)):.4f}, "
        f"utility={float(summary.get('best_net_utility', 0.0)):.4f}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def capacity_planner_dataset_fingerprint(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> str:
    x, y = _validate_inputs(features, labels)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(value) for value in x.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    hasher.update(str(tuple(int(value) for value in y.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return hasher.hexdigest()


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Capacity planner features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Capacity planner features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Capacity planner feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("Capacity planner needs at least one sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Capacity planner features must be finite numbers.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Capacity planner labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Capacity planner labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Capacity planner labels must be binary 0/1.")
    return y.astype(np.int32)


def _capacity_grid(capacity_grid: Sequence[float] | None, sample_count: int) -> np.ndarray:
    if capacity_grid is None:
        raw = np.asarray([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0], dtype=np.float64)
    else:
        raw = np.asarray(list(capacity_grid), dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("Capacity planner capacity grid must contain at least one finite fraction.")
    raw = np.clip(raw, 1.0 / max(1, sample_count), 1.0)
    by_k: dict[int, float] = {}
    for value in np.unique(raw):
        k = min(sample_count, max(1, int(np.ceil(float(value) * sample_count))))
        by_k[k] = float(k / sample_count)
    return np.asarray([by_k[k] for k in sorted(by_k)], dtype=np.float64)


def _capacity_point(
    fraction: float,
    order: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    benefit_tp: float,
    cost_fp: float,
    cost_action: float,
) -> dict[str, float | int]:
    total = int(labels.shape[0])
    positives = int(np.sum(labels == 1))
    k = min(total, max(1, int(np.ceil(float(fraction) * total))))
    selected = order[:k]
    selected_labels = labels[selected]
    tp = int(np.sum(selected_labels == 1))
    fp = int(k - tp)
    fn = int(positives - tp)
    precision = float(tp / k) if k else 0.0
    recall = float(tp / positives) if positives else 0.0
    prevalence = float(positives / total) if total else 0.0
    lift = float(precision / prevalence) if prevalence > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    utility = float(benefit_tp * tp - cost_fp * fp - cost_action * k)
    random_expected_tp = float(k * prevalence)
    return {
        "capacity_fraction": float(k / total),
        "requested_capacity_fraction": float(fraction),
        "k": k,
        "threshold_floor": float(probabilities[selected[-1]]) if k else 1.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative_remaining": fn,
        "precision_at_k": precision,
        "recall_captured": recall,
        "f1_at_k": f1,
        "lift": lift,
        "net_utility": utility,
        "utility_per_action": float(utility / k) if k else 0.0,
        "random_expected_tp": random_expected_tp,
        "gain_over_random_tp": float(tp - random_expected_tp),
    }


def _top_rows(order: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, *, max_rows: int) -> list[dict[str, Any]]:
    rows = []
    for rank, index in enumerate(order[:max_rows], start=1):
        rows.append(
            {
                "rank": int(rank),
                "row_index": int(index),
                "label": int(labels[index]),
                "probability": float(probabilities[index]),
            }
        )
    return rows


def _summary(
    points: list[dict[str, float | int]],
    best_utility: dict[str, float | int],
    best_f1: dict[str, float | int],
    labels: np.ndarray,
) -> dict[str, Any]:
    positives = int(np.sum(labels == 1))
    prevalence = float(positives / max(1, labels.shape[0]))
    best_net = float(best_utility["net_utility"])
    best_precision = float(best_utility["precision_at_k"])
    best_recall = float(best_utility["recall_captured"])
    best_lift = float(best_utility["lift"])
    verdict = _verdict(best_net, best_precision, best_recall, positives)
    return {
        "verdict": verdict,
        "prevalence": prevalence,
        "best_capacity_fraction": float(best_utility["capacity_fraction"]),
        "best_k": int(best_utility["k"]),
        "best_threshold_floor": float(best_utility["threshold_floor"]),
        "best_precision_at_k": best_precision,
        "best_recall_captured": best_recall,
        "best_lift": best_lift,
        "best_net_utility": best_net,
        "best_utility_per_action": float(best_utility["utility_per_action"]),
        "best_f1_capacity_fraction": float(best_f1["capacity_fraction"]),
        "best_f1_at_k": float(best_f1["f1_at_k"]),
        "max_recall_at_half_capacity": max(
            (float(point["recall_captured"]) for point in points if float(point["capacity_fraction"]) <= 0.5),
            default=0.0,
        ),
        "positive_count": positives,
    }


def _verdict(best_net: float, best_precision: float, best_recall: float, positive_count: int) -> str:
    if positive_count == 0:
        return "no_positive_evidence"
    if best_net <= 0.0 or best_precision <= 0.0:
        return "not_actionable"
    if best_recall < 0.20:
        return "low_capture_review"
    return "actionable_capacity_plan"


def _recommendations(summary: dict[str, Any], best_utility: dict[str, float | int]) -> list[dict[str, Any]]:
    verdict = str(summary.get("verdict", "not_actionable"))
    recs: list[dict[str, Any]] = []

    def add(score: float, priority: str, category: str, title: str, reason: str, action: str) -> None:
        recs.append(
            {
                "priority_score": float(score),
                "priority": priority,
                "category": category,
                "title": title,
                "reason": reason,
                "action": action,
            }
        )

    if verdict == "actionable_capacity_plan":
        add(
            72.0,
            "medium",
            "capacity",
            "Use the best utility budget as a starting point",
            f"Best budget captures {summary['best_recall_captured']:.1%} of positives at precision {summary['best_precision_at_k']:.1%}.",
            f"Plan review/action capacity around top {int(best_utility['k'])} row(s), then validate on fresh rows.",
        )
    elif verdict == "low_capture_review":
        add(
            62.0,
            "medium",
            "capacity",
            "Capacity plan captures too little signal",
            f"Best utility point captures only {summary['best_recall_captured']:.1%} of positives.",
            "Increase capacity, improve model ranking, or use a lower threshold before operational use.",
        )
    elif verdict == "no_positive_evidence":
        add(
            58.0,
            "medium",
            "evidence",
            "No positive rows are available for capacity planning",
            "Loaded labels contain no positives.",
            "Load labeled data with observed positives before using capacity planning.",
        )
    else:
        add(
            76.0,
            "high",
            "capacity",
            "Do not use this ranked action plan yet",
            f"Best net utility is {summary['best_net_utility']:.3f}.",
            "Review labels, costs, and model ranking before assigning action capacity.",
        )
    recs.sort(key=lambda item: (-item["priority_score"], item["category"], item["title"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs[:6]
