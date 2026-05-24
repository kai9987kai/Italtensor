from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_threshold_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    current_threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    recall_target: float = 0.9,
    precision_target: float = 0.9,
    grid_size: int = 101,
) -> dict[str, Any]:
    """Evaluate classification metrics across probability thresholds."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Threshold diagnostics features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Threshold diagnostics feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Threshold diagnostics need at least one sample.")
    if not 0.0 <= current_threshold <= 1.0:
        raise ValueError("current_threshold must be between 0 and 1.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    thresholds = _threshold_grid(probabilities, current_threshold, grid_size)
    points = [
        _threshold_point(y, probabilities, float(threshold), fp_cost=fp_cost, fn_cost=fn_cost)
        for threshold in thresholds
    ]
    current = min(points, key=lambda point: abs(float(point["threshold"]) - float(current_threshold)))
    best_f1 = max(points, key=lambda point: (float(point["f1"]), float(point["accuracy"]), -float(point["cost"])))
    best_balanced = max(points, key=lambda point: (float(point["balanced_accuracy"]), float(point["f1"]), -float(point["cost"])))
    min_cost = min(points, key=lambda point: (float(point["cost"]), -float(point["f1"])))
    high_recall = _constraint_choice(points, "recall", recall_target, sort_key=("precision", "f1"))
    high_precision = _constraint_choice(points, "precision", precision_target, sort_key=("recall", "f1"))
    compact_points = _compact_points(points, current_threshold)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "current_threshold": float(current_threshold),
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
        "recall_target": float(recall_target),
        "precision_target": float(precision_target),
        "current": current,
        "best_f1": best_f1,
        "best_balanced_accuracy": best_balanced,
        "min_cost": min_cost,
        "high_recall": high_recall,
        "high_precision": high_precision,
        "points": compact_points,
        "summary": {
            "best_f1_threshold": float(best_f1["threshold"]),
            "best_balanced_accuracy_threshold": float(best_balanced["threshold"]),
            "min_cost_threshold": float(min_cost["threshold"]),
            "current_f1": float(current["f1"]),
            "best_f1": float(best_f1["f1"]),
            "current_cost": float(current["cost"]),
            "min_cost": float(min_cost["cost"]),
        },
    }


def format_threshold_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    high_recall = report.get("high_recall")
    high_precision = report.get("high_precision")
    parts = [
        "Threshold sweep:",
        f"current={float(report.get('current_threshold', 0.5)):.4f}",
        f"best_f1_t={float(summary.get('best_f1_threshold', 0.5)):.4f}",
        f"min_cost_t={float(summary.get('min_cost_threshold', 0.5)):.4f}",
        f"best_f1={float(summary.get('best_f1', 0.0)):.4f}",
        f"min_cost={float(summary.get('min_cost', 0.0)):.4f}",
    ]
    if isinstance(high_recall, dict):
        parts.append(f"recall_target_t={float(high_recall['threshold']):.4f}")
    if isinstance(high_precision, dict):
        parts.append(f"precision_target_t={float(high_precision['threshold']):.4f}")
    return " ".join(parts)


def _threshold_grid(probabilities: np.ndarray, current_threshold: float, grid_size: int) -> np.ndarray:
    grid_count = max(3, int(grid_size))
    values = np.concatenate(
        [
            np.linspace(0.0, 1.0, grid_count, dtype=np.float32),
            np.asarray([current_threshold], dtype=np.float32),
            np.asarray(probabilities, dtype=np.float32),
        ]
    )
    return np.unique(np.clip(values, 0.0, 1.0))


def _threshold_point(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    fp_cost: float,
    fn_cost: float,
) -> dict[str, float | int]:
    metrics = evaluate_predictions(labels, probabilities, threshold)
    fp = int(metrics["false_positive"])
    fn = int(metrics["false_negative"])
    tp = int(metrics["true_positive"])
    tn = int(metrics["true_negative"])
    cost = (float(fp_cost) * fp + float(fn_cost) * fn) / max(1, int(labels.shape[0]))
    return {
        "threshold": float(threshold),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_positive_rate": float((tp + fp) / max(1, int(labels.shape[0]))),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "cost": float(cost),
    }


def _constraint_choice(
    points: list[dict[str, float | int]],
    metric: str,
    target: float,
    *,
    sort_key: tuple[str, str],
) -> dict[str, float | int] | None:
    candidates = [point for point in points if float(point[metric]) >= float(target)]
    if not candidates:
        return None
    first_key, second_key = sort_key
    return max(
        candidates,
        key=lambda point: (
            float(point[first_key]),
            float(point[second_key]),
            -float(point["cost"]),
        ),
    )


def _compact_points(points: list[dict[str, float | int]], current_threshold: float) -> list[dict[str, float | int]]:
    if len(points) <= 25:
        return points
    selected_indices = set(np.linspace(0, len(points) - 1, 21, dtype=np.int32).tolist())
    current_index = min(
        range(len(points)),
        key=lambda index: abs(float(points[index]["threshold"]) - float(current_threshold)),
    )
    selected_indices.add(current_index)
    for rank_key in ("f1", "balanced_accuracy"):
        selected_indices.add(max(range(len(points)), key=lambda index: float(points[index][rank_key])))
    selected_indices.add(min(range(len(points)), key=lambda index: float(points[index]["cost"])))
    return [points[index] for index in sorted(selected_indices)]
