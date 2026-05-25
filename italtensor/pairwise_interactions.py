from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_pairwise_interaction_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    grid_size: int = 5,
    max_features: int = 8,
) -> dict[str, Any]:
    """Estimate pairwise feature interactions with centered 2D partial-dependence grids."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Pairwise interaction features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Pairwise interaction features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Pairwise interaction feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("Pairwise interaction diagnostics need at least one sample.")
    if x.shape[1] < 2:
        raise ValueError("Pairwise interaction diagnostics need at least two features.")
    if set(int(item) for item in np.unique(y)) - {0, 1}:
        raise ValueError("Pairwise interaction diagnostics require binary labels 0 or 1.")
    if int(grid_size) < 3:
        raise ValueError("Pairwise interaction grid_size must be at least 3.")
    if int(max_features) < 2:
        raise ValueError("max_features must be at least 2.")

    selected_features = _selected_features(x, int(max_features))
    grids = {feature_index: _feature_grid(x[:, feature_index], int(grid_size)) for feature_index in selected_features}
    pair_specs = list(combinations(selected_features, 2))
    if not pair_specs:
        raise ValueError("Pairwise interaction diagnostics found fewer than two non-constant features.")

    scenarios: list[np.ndarray] = []
    for left, right in pair_specs:
        for left_value in grids[left]:
            for right_value in grids[right]:
                scenario = x.copy()
                scenario[:, left] = float(left_value)
                scenario[:, right] = float(right_value)
                scenarios.append(scenario)

    stacked = np.vstack(scenarios)
    prepared = preprocessor.transform(stacked) if preprocessor is not None else stacked
    probabilities = predict_probability(model, prepared)
    if probabilities.shape[0] != stacked.shape[0]:
        raise ValueError("Model returned a probability count that does not match the interaction grid.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)

    pair_reports: list[dict[str, Any]] = []
    cursor = 0
    for left, right in pair_specs:
        left_grid = grids[left]
        right_grid = grids[right]
        surface = np.zeros((len(left_grid), len(right_grid)), dtype=np.float32)
        for left_index in range(len(left_grid)):
            for right_index in range(len(right_grid)):
                values = probabilities[cursor: cursor + x.shape[0]]
                cursor += x.shape[0]
                surface[left_index, right_index] = float(np.mean(values))
        pair_reports.append(_pair_report(x, left, right, left_grid, right_grid, surface))

    ranked_pairs = sorted(
        pair_reports,
        key=lambda item: (
            -float(item["interaction_strength"]),
            -float(item["max_abs_interaction"]),
            -float(item["response_range"]),
            int(item["feature_i"]),
            int(item["feature_j"]),
        ),
    )
    top = ranked_pairs[0]
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "grid_size": int(grid_size),
        "max_features": int(max_features),
        "selected_features": [int(item) for item in selected_features],
        "omitted_feature_count": int(max(0, x.shape[1] - len(selected_features))),
        "pairs": ranked_pairs,
        "summary": {
            "evaluated_pair_count": int(len(ranked_pairs)),
            "top_pair": [int(top["feature_i"]), int(top["feature_j"])],
            "top_interaction_strength": float(top["interaction_strength"]),
            "top_max_abs_interaction": float(top["max_abs_interaction"]),
            "top_response_range": float(top["response_range"]),
            "strong_pair_count": int(sum(1 for item in ranked_pairs if float(item["interaction_strength"]) >= 0.25)),
            "threshold_crossing_pair_count": int(sum(1 for item in ranked_pairs if int(item["threshold_crossings"]) > 0)),
            "warning": _warning(x, ranked_pairs, len(selected_features)),
        },
    }


def format_pairwise_interaction_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    pair = summary.get("top_pair")
    pair_text = "-" if not isinstance(pair, list) else f"x{int(pair[0]) + 1}:x{int(pair[1]) + 1}"
    return (
        "Pairwise interactions: "
        f"pairs={int(summary.get('evaluated_pair_count', 0))}, "
        f"top={pair_text}, "
        f"H={float(summary.get('top_interaction_strength', 0.0)):.4f}, "
        f"max_abs={float(summary.get('top_max_abs_interaction', 0.0)):.4f}, "
        f"strong={int(summary.get('strong_pair_count', 0))}"
    )


def _selected_features(x: np.ndarray, max_features: int) -> list[int]:
    variances = np.var(x, axis=0)
    nonconstant = [int(index) for index, variance in enumerate(variances) if float(variance) > 1e-12]
    ranked = sorted(nonconstant, key=lambda index: (-float(variances[index]), index))
    selected = ranked[:max_features]
    if len(selected) >= 2:
        return selected
    return list(range(min(x.shape[1], max_features)))


def _feature_grid(values: np.ndarray, grid_size: int) -> list[float]:
    unique = np.unique(values.astype(np.float32))
    if unique.shape[0] == 1:
        return [float(unique[0])]
    left, right = np.quantile(values, [0.05, 0.95])
    if float(left) == float(right):
        left, right = float(np.min(values)), float(np.max(values))
    return [float(item) for item in np.unique(np.linspace(float(left), float(right), grid_size).astype(np.float32))]


def _pair_report(
    x: np.ndarray,
    left: int,
    right: int,
    left_grid: list[float],
    right_grid: list[float],
    surface: np.ndarray,
) -> dict[str, Any]:
    grand = float(np.mean(surface))
    left_effect = np.mean(surface, axis=1, keepdims=True)
    right_effect = np.mean(surface, axis=0, keepdims=True)
    additive = left_effect + right_effect - grand
    residual = surface - additive
    centered = surface - grand
    denom = float(np.sum(np.square(centered)))
    strength = float(np.sqrt(float(np.sum(np.square(residual))) / denom)) if denom > 1e-12 else 0.0
    abs_residual = np.abs(residual)
    max_index = np.unravel_index(int(np.argmax(abs_residual)), residual.shape)
    threshold_crossings = _threshold_crossings(surface)
    correlation = _correlation(x[:, left], x[:, right])
    max_abs = float(abs_residual[max_index])
    mean_abs = float(np.mean(abs_residual))
    return {
        "feature_i": int(left),
        "feature_j": int(right),
        "grid_i": [float(item) for item in left_grid],
        "grid_j": [float(item) for item in right_grid],
        "interaction_strength": min(max(strength, 0.0), 1.0),
        "max_abs_interaction": max_abs,
        "mean_abs_interaction": mean_abs,
        "signed_interaction": float(np.mean(residual)),
        "response_range": float(np.max(surface) - np.min(surface)),
        "threshold_crossings": int(threshold_crossings),
        "raw_correlation": correlation,
        "strongest_cell": {
            "feature_i_value": float(left_grid[max_index[0]]),
            "feature_j_value": float(right_grid[max_index[1]]),
            "interaction": float(residual[max_index]),
        },
        "surface": [[float(value) for value in row] for row in surface],
        "interaction_surface": [[float(value) for value in row] for row in residual],
        "risk_flags": _risk_flags(strength, max_abs, threshold_crossings, correlation),
    }


def _threshold_crossings(surface: np.ndarray, threshold: float = 0.5) -> int:
    above = surface >= threshold
    horizontal = int(np.sum(above[:, 1:] != above[:, :-1]))
    vertical = int(np.sum(above[1:, :] != above[:-1, :]))
    return horizontal + vertical


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _risk_flags(interaction_strength: float, max_abs_interaction: float, threshold_crossings: int, correlation: float) -> list[str]:
    flags: list[str] = []
    if interaction_strength >= 0.25:
        flags.append("strong_interaction")
    if max_abs_interaction >= 0.1:
        flags.append("non_additive")
    if threshold_crossings > 0:
        flags.append("threshold_crossing")
    if abs(correlation) >= 0.8:
        flags.append("correlated_features")
    return flags


def _warning(x: np.ndarray, pairs: list[dict[str, Any]], selected_count: int) -> str | None:
    messages: list[str] = []
    if selected_count < x.shape[1]:
        messages.append(f"Only the {selected_count} highest-variance features were scanned.")
    if any("correlated_features" in pair.get("risk_flags", []) for pair in pairs):
        messages.append("Highly correlated feature pairs can make two-way partial dependence grids unrealistic.")
    return " ".join(messages) if messages else None
