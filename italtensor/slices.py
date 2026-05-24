from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_slice_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    max_features: int = 12,
    bins: int = 4,
    min_count: int | None = None,
) -> dict[str, Any]:
    """Rank raw-feature ranges where model performance is weakest."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Slice diagnostics features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Slice diagnostics feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Slice diagnostics need at least one sample.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    base_metrics = _compact_metrics(y, probabilities, threshold)
    minimum = int(min_count if min_count is not None else max(3, round(x.shape[0] * 0.08)))
    minimum = max(1, minimum)

    slices: list[dict[str, Any]] = []
    for feature_index in range(min(x.shape[1], max(1, int(max_features)))):
        values = x[:, feature_index]
        if np.allclose(values, values[0]):
            continue
        for left, right, is_last in _quantile_ranges(values, bins):
            if is_last:
                mask = (values >= left) & (values <= right)
            else:
                mask = (values >= left) & (values < right)
            count = int(np.sum(mask))
            if count < minimum:
                continue
            slice_probabilities = probabilities[mask]
            slice_labels = y[mask]
            metrics = _compact_metrics(slice_labels, slice_probabilities, threshold)
            predicted_labels = (slice_probabilities >= threshold).astype(np.int32)
            slices.append(
                {
                    "feature_index": int(feature_index),
                    "left": float(left),
                    "right": float(right),
                    "count": count,
                    "coverage": float(count / x.shape[0]),
                    "f1": float(metrics["f1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_delta": float(float(metrics["f1"]) - float(base_metrics["f1"])),
                    "accuracy_delta": float(float(metrics["accuracy"]) - float(base_metrics["accuracy"])),
                    "error_rate": float(1.0 - float(metrics["accuracy"])),
                    "label_prevalence": float(np.mean(slice_labels)),
                    "predicted_positive_rate": float(np.mean(predicted_labels)),
                }
            )

    slices.sort(key=lambda item: (float(item["f1_delta"]), float(item["accuracy_delta"]), -float(item["count"])))
    top_slices = slices[:12]
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "bin_count": int(bins),
        "min_count": minimum,
        "base": base_metrics,
        "slices": top_slices,
        "summary": {
            "slice_count": len(slices),
            "worst_slice": _slice_name(top_slices[0]) if top_slices else "none",
            "worst_f1_delta": float(top_slices[0]["f1_delta"]) if top_slices else 0.0,
            "worst_accuracy_delta": float(top_slices[0]["accuracy_delta"]) if top_slices else 0.0,
        },
    }


def format_slice_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    base = report.get("base", {})
    return (
        "Slice diagnostics: "
        f"base_f1={float(base.get('f1', 0.0)):.4f}, "
        f"slices={int(summary.get('slice_count', 0))}, "
        f"worst={summary.get('worst_slice', 'none')}, "
        f"f1_delta={float(summary.get('worst_f1_delta', 0.0)):.4f}, "
        f"acc_delta={float(summary.get('worst_accuracy_delta', 0.0)):.4f}"
    )


def _compact_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = evaluate_predictions(labels, probabilities, threshold)
    keys = ("f1", "accuracy", "balanced_accuracy", "precision", "recall")
    return {key: metrics[key] for key in keys}


def _quantile_ranges(values: np.ndarray, bins: int) -> list[tuple[float, float, bool]]:
    bin_count = max(2, int(bins))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 2:
        return []
    ranges: list[tuple[float, float, bool]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if float(left) == float(right):
            continue
        ranges.append((float(left), float(right), index == edges.size - 2))
    return ranges


def _slice_name(item: dict[str, Any]) -> str:
    return f"x{int(item['feature_index']) + 1}[{float(item['left']):.4g}, {float(item['right']):.4g}]"
