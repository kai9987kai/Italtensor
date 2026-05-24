from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DEFAULT_TARGET_RISKS = (0.0, 0.05, 0.1, 0.2)


def run_selective_risk_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    grid_size: int = 21,
    epsilon: float = 1e-6,
    target_risks: Sequence[float] = DEFAULT_TARGET_RISKS,
) -> dict[str, Any]:
    """Evaluate risk/coverage tradeoffs when abstaining on uncertain rows."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Selective risk features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Selective risk feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Selective risk diagnostics need at least one sample.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Selective risk threshold must be between 0 and 1.")
    if int(grid_size) < 2:
        raise ValueError("Selective risk grid_size must be at least 2.")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("Selective risk epsilon must be between 0 and 0.5.")

    effective_threshold = float(np.clip(float(threshold), float(epsilon), 1.0 - float(epsilon)))
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    predictions = (probabilities >= effective_threshold).astype(np.int32)
    confidence = _confidence_scores(probabilities, effective_threshold)
    cutoffs = _cutoff_grid(confidence, int(grid_size))
    full_metrics = evaluate_predictions(y, probabilities, effective_threshold)
    base = _base_metrics(full_metrics)
    points = [_point(y, probabilities, predictions, confidence, cutoff, effective_threshold, base) for cutoff in cutoffs]
    valid_points = [point for point in points if int(point["covered_count"]) > 0]
    ranked_cutoffs = sorted(
        valid_points,
        key=lambda item: (
            float(item["error_rate"]),
            -float(item["f1"]),
            -float(item["coverage"]),
            float(item["confidence_cutoff"]),
        ),
    )
    best_by_accuracy = max(valid_points, key=lambda item: (float(item["accuracy"]), float(item["coverage"])))
    best_by_f1 = max(valid_points, key=lambda item: (float(item["f1"]), float(item["coverage"])))
    lowest_error = ranked_cutoffs[0]
    min_risk = min((float(item["error_rate"]) for item in valid_points), default=1.0)
    target_points = {
        _target_key(target): _best_point_for_target(valid_points, float(target))
        for target in target_risks
    }

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "current_threshold": float(threshold),
        "effective_threshold": effective_threshold,
        "score": "normalized_margin_from_threshold",
        "base": base,
        "points": points,
        "ranked_cutoffs": ranked_cutoffs[:12],
        "target_points": target_points,
        "summary": {
            "best_f1_cutoff": float(best_by_f1["confidence_cutoff"]),
            "best_accuracy_cutoff": float(best_by_accuracy["confidence_cutoff"]),
            "lowest_error_cutoff": float(lowest_error["confidence_cutoff"]),
            "recommended_cutoff": float(lowest_error["confidence_cutoff"]),
            "full_coverage_risk": float(base["error_rate"]),
            "min_selective_risk": float(min_risk),
            "best_selective_accuracy": float(best_by_accuracy["accuracy"]),
            "best_selective_coverage": float(best_by_accuracy["coverage"]),
            "max_error_reduction": float(base["error_rate"] - min_risk),
            "coverage_at_10pct_risk": _coverage_for_target(target_points, 0.1),
            "area_under_risk_coverage": _area_under_risk_coverage(valid_points),
            "warning": _warning(y),
        },
    }


def format_selective_risk_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Selective risk: "
        f"full_risk={float(summary.get('full_coverage_risk', 0.0)):.4f}, "
        f"min_risk={float(summary.get('min_selective_risk', 0.0)):.4f}, "
        f"best_acc={float(summary.get('best_selective_accuracy', 0.0)):.4f}, "
        f"coverage={float(summary.get('best_selective_coverage', 0.0)):.4f}, "
        f"AURC={float(summary.get('area_under_risk_coverage', 0.0)):.4f}"
    )


def _confidence_scores(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    denom = max(float(threshold), 1.0 - float(threshold), 1e-6)
    scores = np.abs(np.asarray(probabilities, dtype=np.float32).reshape(-1) - float(threshold)) / denom
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def _cutoff_grid(confidence: np.ndarray, grid_size: int) -> list[float]:
    max_confidence = float(np.max(confidence)) if confidence.size else 0.0
    values = list(np.linspace(0.0, max_confidence, grid_size))
    values.extend(float(item) for item in np.unique(confidence))
    return sorted({round(float(value), 10) for value in values if 0.0 <= float(value) <= max_confidence})


def _base_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "coverage": 1.0,
        "accuracy": float(metrics["accuracy"]),
        "error_rate": float(1.0 - float(metrics["accuracy"])),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "true_positive": int(metrics["true_positive"]),
        "true_negative": int(metrics["true_negative"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
    }


def _point(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    cutoff: float,
    threshold: float,
    base: dict[str, float | int],
) -> dict[str, Any]:
    selected = confidence >= cutoff
    selected_count = int(np.sum(selected))
    abstained_count = int(labels.shape[0] - selected_count)
    coverage = float(selected_count / labels.shape[0])
    if selected_count == 0:
        return {
            "confidence_cutoff": float(cutoff),
            "uncertainty_cutoff": float(1.0 - float(cutoff)),
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "covered_count": 0,
            "selected_count": 0,
            "abstained_count": abstained_count,
            "accuracy": None,
            "error_rate": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy_delta": None,
            "error_reduction": None,
            "f1_delta": None,
            "abstained_error_rate": _abstained_error_rate(labels, predictions, ~selected),
            "mean_confidence": 0.0,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }

    selected_metrics = evaluate_predictions(labels[selected], probabilities[selected], threshold)
    selected_predictions = predictions[selected]
    return {
        "confidence_cutoff": float(cutoff),
        "uncertainty_cutoff": float(1.0 - float(cutoff)),
        "coverage": coverage,
        "abstention_rate": float(1.0 - coverage),
        "covered_count": selected_count,
        "selected_count": selected_count,
        "abstained_count": abstained_count,
        "accuracy": float(selected_metrics["accuracy"]),
        "error_rate": float(1.0 - float(selected_metrics["accuracy"])),
        "f1": float(selected_metrics["f1"]),
        "precision": float(selected_metrics["precision"]),
        "recall": float(selected_metrics["recall"]),
        "accuracy_delta": float(float(selected_metrics["accuracy"]) - float(base["accuracy"])),
        "error_reduction": float(float(base["error_rate"]) - (1.0 - float(selected_metrics["accuracy"]))),
        "f1_delta": float(float(selected_metrics["f1"]) - float(base["f1"])),
        "abstained_error_rate": _abstained_error_rate(labels, predictions, ~selected),
        "mean_confidence": float(np.mean(confidence[selected])),
        "predicted_positive_rate": float(np.mean(selected_predictions)),
        "true_positive": int(selected_metrics["true_positive"]),
        "true_negative": int(selected_metrics["true_negative"]),
        "false_positive": int(selected_metrics["false_positive"]),
        "false_negative": int(selected_metrics["false_negative"]),
    }


def _abstained_error_rate(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    count = int(np.sum(mask))
    if count == 0:
        return 0.0
    return float(np.mean(predictions[mask] != labels[mask]))


def _best_point_for_target(points: list[dict[str, Any]], target_risk: float) -> dict[str, Any] | None:
    candidates = [point for point in points if float(point["error_rate"]) <= target_risk]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (float(item["coverage"]), -float(item["error_rate"])))


def _target_key(target: float) -> str:
    return f"risk_le_{target:.2f}"


def _coverage_for_target(target_points: dict[str, dict[str, Any] | None], target: float) -> float:
    point = target_points.get(_target_key(target))
    return float(point["coverage"]) if isinstance(point, dict) else 0.0


def _area_under_risk_coverage(points: list[dict[str, Any]]) -> float:
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda item: float(item["coverage"]))
    coverages = np.asarray([float(item["coverage"]) for item in ordered], dtype=np.float32)
    risks = np.asarray([float(item["error_rate"]) for item in ordered], dtype=np.float32)
    return float(np.trapezoid(risks, coverages))


def _warning(labels: np.ndarray) -> str | None:
    unique = set(int(item) for item in np.unique(labels))
    if unique == {0}:
        return "All labels are negative; selective risk cannot evaluate missed positives."
    if unique == {1}:
        return "All labels are positive; selective risk cannot evaluate false positives."
    return None
