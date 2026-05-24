from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_decision_curve_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    current_threshold: float = 0.5,
    grid_size: int = 101,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Compute decision-curve-style net benefit across action thresholds."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Decision curve features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Decision curve feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Decision curve diagnostics need at least one sample.")
    if int(grid_size) < 2:
        raise ValueError("Decision curve grid_size must be at least 2.")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("Decision curve epsilon must be between 0 and 0.5.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    effective_current = float(np.clip(float(current_threshold), epsilon, 1.0 - epsilon))
    thresholds = _threshold_grid(int(grid_size), float(epsilon), effective_current)
    prevalence = float(np.mean(y))
    points = [_point(y, probabilities, threshold, prevalence) for threshold in thresholds]
    current = min(points, key=lambda item: abs(float(item["threshold"]) - effective_current))
    best = max(points, key=lambda item: (float(item["net_benefit_model"]), float(item["delta_vs_best_default"])))
    useful_points = [point for point in points if float(point["delta_vs_best_default"]) > 0.0]

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "current_threshold": float(current_threshold),
        "effective_current_threshold": effective_current,
        "prevalence": prevalence,
        "points": points,
        "current": current,
        "best": best,
        "summary": {
            "best_threshold": float(best["threshold"]),
            "best_net_benefit": float(best["net_benefit_model"]),
            "max_delta_vs_best_default": max(
                (float(point["delta_vs_best_default"]) for point in points),
                default=0.0,
            ),
            "useful_threshold_count": len(useful_points),
            "useful_threshold_ranges": _useful_ranges(useful_points, thresholds),
            "warning": _warning(prevalence),
        },
    }


def format_decision_curve_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    ranges = summary.get("useful_threshold_ranges") or []
    range_text = "none"
    if ranges:
        range_text = ", ".join(f"{float(left):.4f}-{float(right):.4f}" for left, right in ranges[:3])
    return (
        "Decision curve: "
        f"best_t={float(summary.get('best_threshold', 0.0)):.4f}, "
        f"best_nb={float(summary.get('best_net_benefit', 0.0)):.4f}, "
        f"max_gain={float(summary.get('max_delta_vs_best_default', 0.0)):.4f}, "
        f"useful_ranges={range_text}"
    )


def _threshold_grid(grid_size: int, epsilon: float, current_threshold: float) -> list[float]:
    grid = np.linspace(epsilon, 1.0 - epsilon, grid_size).tolist()
    grid.append(float(current_threshold))
    return sorted({round(float(value), 10) for value in grid if epsilon <= float(value) <= 1.0 - epsilon})


def _point(labels: np.ndarray, probabilities: np.ndarray, threshold: float, prevalence: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int32)
    n = max(labels.shape[0], 1)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    odds = threshold / (1.0 - threshold)
    net_benefit_model = float(tp / n - fp / n * odds)
    net_benefit_treat_all = float(prevalence - (1.0 - prevalence) * odds)
    net_benefit_treat_none = 0.0
    best_default = max(net_benefit_treat_all, net_benefit_treat_none)
    delta_vs_best_default = float(net_benefit_model - best_default)
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_positive_rate": float(np.mean(predicted)),
        "net_benefit_model": net_benefit_model,
        "net_benefit_treat_all": net_benefit_treat_all,
        "net_benefit_treat_none": net_benefit_treat_none,
        "delta_vs_treat_all": float(net_benefit_model - net_benefit_treat_all),
        "delta_vs_treat_none": net_benefit_model,
        "delta_vs_best_default": delta_vs_best_default,
        "best_default_strategy": "treat_all" if net_benefit_treat_all >= net_benefit_treat_none else "treat_none",
        "net_interventions_avoided_per_100": float(delta_vs_best_default / odds * 100.0) if odds > 0 else 0.0,
    }


def _useful_ranges(points: list[dict[str, Any]], thresholds: list[float]) -> list[list[float]]:
    useful = {float(point["threshold"]) for point in points}
    ranges: list[list[float]] = []
    start: float | None = None
    previous: float | None = None
    for threshold in thresholds:
        is_useful = threshold in useful
        if is_useful and start is None:
            start = threshold
        if not is_useful and start is not None and previous is not None:
            ranges.append([float(start), float(previous)])
            start = None
        previous = threshold
    if start is not None and previous is not None:
        ranges.append([float(start), float(previous)])
    return ranges


def _warning(prevalence: float) -> str | None:
    if prevalence <= 0.0:
        return "All labels are negative; treat-none will usually dominate and false-negative utility is not observable."
    if prevalence >= 1.0:
        return "All labels are positive; treat-all will usually dominate and false-positive harm is not observable."
    return None
