from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_sample_review(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    max_items: int = 12,
    confident_error_probability: float = 0.8,
) -> dict[str, Any]:
    """Rank dataset rows for manual label and hardness review."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Sample review features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Sample review feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Sample review needs at least one sample.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    predicted = (probabilities >= threshold).astype(np.int32)
    true_probability = np.where(y == 1, probabilities, 1.0 - probabilities)
    clipped_true_probability = np.clip(true_probability, 1e-7, 1.0)
    losses = -np.log(clipped_true_probability)
    disagreement = predicted != y
    opposing_confidence = np.where(predicted == 1, probabilities, 1.0 - probabilities)
    margin = np.abs(probabilities - threshold)
    items = [
        _row_item(index, x[index], int(y[index]), int(predicted[index]), float(probabilities[index]), float(losses[index]), float(margin[index]))
        for index in range(x.shape[0])
    ]

    label_issues = [
        item
        for item, disagrees, confidence in zip(items, disagreement, opposing_confidence, strict=True)
        if bool(disagrees) and float(confidence) >= confident_error_probability
    ]
    label_issues.sort(key=lambda item: (-float(item["loss"]), -abs(float(item["probability"]) - threshold), int(item["row_index"])))

    hard_examples = sorted(items, key=lambda item: (-float(item["loss"]), int(item["row_index"])))
    ambiguous_examples = sorted(items, key=lambda item: (float(item["margin"]), -float(item["loss"]), int(item["row_index"])))
    max_count = max(1, int(max_items))
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "confident_error_probability": float(confident_error_probability),
        "summary": {
            "label_issue_count": len(label_issues),
            "disagreement_count": int(np.sum(disagreement)),
            "mean_loss": float(np.mean(losses)),
            "max_loss": float(np.max(losses)),
            "ambiguous_count": int(np.sum(margin <= 0.05)),
        },
        "label_issues": label_issues[:max_count],
        "hard_examples": hard_examples[:max_count],
        "ambiguous_examples": ambiguous_examples[:max_count],
    }


def format_sample_review_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Sample review: "
        f"label_issues={int(summary.get('label_issue_count', 0))}, "
        f"disagreements={int(summary.get('disagreement_count', 0))}, "
        f"ambiguous={int(summary.get('ambiguous_count', 0))}, "
        f"mean_loss={float(summary.get('mean_loss', 0.0)):.4f}, "
        f"max_loss={float(summary.get('max_loss', 0.0)):.4f}"
    )


def _row_item(
    row_index: int,
    features: np.ndarray,
    label: int,
    predicted_label: int,
    probability: float,
    loss: float,
    margin: float,
) -> dict[str, Any]:
    return {
        "row_index": int(row_index),
        "label": int(label),
        "predicted_label": int(predicted_label),
        "probability": float(probability),
        "loss": float(loss),
        "margin": float(margin),
        "feature_preview": [float(value) for value in features[:8]],
    }
