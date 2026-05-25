from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_model_response_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    grid_size: int = 9,
) -> dict[str, Any]:
    """Sweep each raw feature over an observed grid and summarize mean model response."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Model response features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Model response features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Model response feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("Model response diagnostics need at least one sample.")
    if set(int(item) for item in np.unique(y)) - {0, 1}:
        raise ValueError("Model response diagnostics require binary labels 0 or 1.")
    if int(grid_size) < 3:
        raise ValueError("Model response grid_size must be at least 3.")

    scenarios: list[tuple[str, int | None, float | None, np.ndarray]] = [("baseline", None, None, x)]
    feature_grids: dict[int, list[float]] = {}
    for feature_index in range(x.shape[1]):
        grid = _feature_grid(x[:, feature_index], int(grid_size))
        feature_grids[feature_index] = grid
        for value in grid:
            scenario = x.copy()
            scenario[:, feature_index] = float(value)
            scenarios.append(("feature", feature_index, float(value), scenario))

    stacked = np.vstack([scenario for _, _, _, scenario in scenarios])
    prepared = preprocessor.transform(stacked) if preprocessor is not None else stacked
    probabilities = predict_probability(model, prepared)
    if probabilities.shape[0] != stacked.shape[0]:
        raise ValueError("Model returned a probability count that does not match the response grid.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)

    cursor = 0
    baseline_probs = probabilities[cursor: cursor + x.shape[0]]
    cursor += x.shape[0]
    feature_reports: list[dict[str, Any]] = []
    for feature_index in range(x.shape[1]):
        points: list[dict[str, Any]] = []
        for value in feature_grids[feature_index]:
            probs = probabilities[cursor: cursor + x.shape[0]]
            cursor += x.shape[0]
            points.append(_grid_point(value, probs))
        feature_reports.append(_feature_report(feature_index, x[:, feature_index], points, baseline_probs))

    ranked = sorted(
        feature_reports,
        key=lambda item: (-float(item["response_range"]), -int(item["direction_changes"]), int(item["feature_index"])),
    )
    top = ranked[0] if ranked else None
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "grid_size": int(grid_size),
        "baseline": {
            "mean_probability": float(np.mean(baseline_probs)),
            "std_probability": float(np.std(baseline_probs)),
        },
        "features": ranked,
        "summary": {
            "top_feature": int(top["feature_index"]) if top else None,
            "top_response_range": float(top["response_range"]) if top else 0.0,
            "top_direction": top["direction"] if top else None,
            "nonmonotonic_feature_count": int(sum(1 for item in ranked if item["direction"] == "nonmonotonic")),
            "high_impact_feature_count": int(sum(1 for item in ranked if float(item["response_range"]) >= 0.2)),
            "warning": _warning(ranked),
        },
    }


def format_model_response_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top = summary.get("top_feature")
    top_text = "-" if top is None else f"x{int(top) + 1}"
    return (
        "Model response: "
        f"top={top_text}, "
        f"range={float(summary.get('top_response_range', 0.0)):.4f}, "
        f"direction={summary.get('top_direction') or '-'}, "
        f"nonmonotonic={int(summary.get('nonmonotonic_feature_count', 0))}, "
        f"high_impact={int(summary.get('high_impact_feature_count', 0))}"
    )


def _feature_grid(values: np.ndarray, grid_size: int) -> list[float]:
    unique = np.unique(values.astype(np.float32))
    if unique.shape[0] == 1:
        return [float(unique[0])]
    left, right = np.quantile(values, [0.05, 0.95])
    if float(left) == float(right):
        left, right = float(np.min(values)), float(np.max(values))
    grid = np.linspace(float(left), float(right), grid_size)
    return [float(item) for item in np.unique(grid.astype(np.float32))]


def _grid_point(value: float, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "value": float(value),
        "mean_probability": float(np.mean(probabilities)),
        "std_probability": float(np.std(probabilities)),
        "p10": float(np.quantile(probabilities, 0.1)),
        "p90": float(np.quantile(probabilities, 0.9)),
    }


def _feature_report(
    feature_index: int,
    values: np.ndarray,
    points: list[dict[str, float]],
    baseline_probs: np.ndarray,
) -> dict[str, Any]:
    means = np.asarray([point["mean_probability"] for point in points], dtype=np.float32)
    response_range = float(np.max(means) - np.min(means)) if means.size else 0.0
    signed_change = float(means[-1] - means[0]) if means.size >= 2 else 0.0
    direction_changes = _direction_changes(means)
    return {
        "feature_index": int(feature_index),
        "observed_min": float(np.min(values)),
        "observed_max": float(np.max(values)),
        "observed_mean": float(np.mean(values)),
        "baseline_mean_probability": float(np.mean(baseline_probs)),
        "response_range": response_range,
        "signed_change": signed_change,
        "direction": _direction(means, direction_changes),
        "direction_changes": int(direction_changes),
        "min_response_value": float(points[int(np.argmin(means))]["value"]) if points else None,
        "max_response_value": float(points[int(np.argmax(means))]["value"]) if points else None,
        "points": points,
        "risk_flags": _risk_flags(response_range, direction_changes),
    }


def _direction(means: np.ndarray, direction_changes: int) -> str:
    if means.shape[0] < 2 or float(np.max(means) - np.min(means)) < 1e-6:
        return "flat"
    if direction_changes > 0:
        return "nonmonotonic"
    return "increasing" if float(means[-1] - means[0]) > 0.0 else "decreasing"


def _direction_changes(means: np.ndarray) -> int:
    if means.shape[0] < 3:
        return 0
    diffs = np.diff(means)
    scale = max(float(np.max(means) - np.min(means)), 1e-6)
    signs = np.sign(np.where(np.abs(diffs) < 0.01 * scale, 0.0, diffs))
    signs = signs[signs != 0]
    if signs.shape[0] < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _risk_flags(response_range: float, direction_changes: int) -> list[str]:
    flags: list[str] = []
    if response_range >= 0.2:
        flags.append("high_impact")
    if direction_changes > 0:
        flags.append("nonmonotonic")
    return flags


def _warning(features: list[dict[str, Any]]) -> str | None:
    if not features:
        return "No features were available for response diagnostics."
    if all(float(item["response_range"]) < 1e-6 for item in features):
        return "Model response is flat across the observed feature grids."
    return None
