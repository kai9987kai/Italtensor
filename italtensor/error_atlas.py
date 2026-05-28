from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_error_atlas(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    max_items: int = 10,
    high_confidence: float = 0.80,
    near_threshold_width: float = 0.05,
) -> dict[str, Any]:
    """Build a row-level confusion atlas for the active model on the loaded dataset."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Error atlas features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Error atlas feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Error atlas needs at least one sample.")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    max_items = max(1, int(max_items))
    high_confidence = float(high_confidence)
    near_threshold_width = max(0.0, float(near_threshold_width))

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    predicted = (probabilities >= threshold).astype(np.int32)
    losses = _binary_log_loss(y, probabilities)
    margins = np.abs(probabilities - threshold)
    confidence = np.where(predicted == 1, probabilities, 1.0 - probabilities)

    masks = {
        "true_positive": (y == 1) & (predicted == 1),
        "true_negative": (y == 0) & (predicted == 0),
        "false_positive": (y == 0) & (predicted == 1),
        "false_negative": (y == 1) & (predicted == 0),
    }
    error_mask = predicted != y
    correct_mask = ~error_mask
    high_conf_error_mask = error_mask & (confidence >= high_confidence)
    near_threshold_mask = margins <= near_threshold_width
    confusion = _confusion_summary(masks, y)
    row_items = [
        _row_item(
            index,
            x[index],
            int(y[index]),
            int(predicted[index]),
            float(probabilities[index]),
            float(losses[index]),
            float(margins[index]),
            float(confidence[index]),
        )
        for index in range(x.shape[0])
    ]
    buckets = {
        name: _rank_rows(row_items, np.where(mask)[0], threshold=threshold, limit=max_items)
        for name, mask in masks.items()
    }
    high_confidence_errors = _rank_rows(row_items, np.where(high_conf_error_mask)[0], threshold=threshold, limit=max_items)
    near_threshold_rows = _rank_rows(row_items, np.where(near_threshold_mask)[0], threshold=threshold, limit=max_items, near=True)
    feature_shifts = _feature_error_shifts(x, error_mask, correct_mask)
    error_count = int(np.sum(error_mask))
    high_confidence_error_count = int(np.sum(high_conf_error_mask))
    dominant_error_type = _dominant_error_type(confusion)
    recommendations = _recommendations(
        confusion=confusion,
        error_count=error_count,
        high_confidence_error_count=high_confidence_error_count,
        near_threshold_count=int(np.sum(near_threshold_mask)),
        feature_shifts=feature_shifts,
    )
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": threshold,
        "high_confidence": high_confidence,
        "near_threshold_width": near_threshold_width,
        "summary": {
            "error_count": error_count,
            "error_rate": float(error_count / x.shape[0]),
            "high_confidence_error_count": high_confidence_error_count,
            "near_threshold_count": int(np.sum(near_threshold_mask)),
            "mean_loss": float(np.mean(losses)),
            "max_loss": float(np.max(losses)),
            "dominant_error_type": dominant_error_type,
            "top_error_row": high_confidence_errors[0]["row_index"] if high_confidence_errors else _top_error_row(row_items, error_mask),
            "recommendation": recommendations[0]["action"] if recommendations else None,
        },
        "confusion": confusion,
        "buckets": buckets,
        "high_confidence_errors": high_confidence_errors,
        "near_threshold_rows": near_threshold_rows,
        "feature_error_shifts": feature_shifts,
        "recommendations": recommendations,
    }


def format_error_atlas_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    confusion = report.get("confusion", {})
    return (
        "Error atlas: "
        f"errors={int(summary.get('error_count', 0))}/{int(report.get('sample_count', 0))}, "
        f"rate={float(summary.get('error_rate', 0.0)):.4f}, "
        f"FP={int(confusion.get('false_positive', 0))}, "
        f"FN={int(confusion.get('false_negative', 0))}, "
        f"high_conf={int(summary.get('high_confidence_error_count', 0))}, "
        f"near_threshold={int(summary.get('near_threshold_count', 0))}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))


def _confusion_summary(masks: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(masks["true_positive"]))
    tn = int(np.sum(masks["true_negative"]))
    fp = int(np.sum(masks["false_positive"]))
    fn = int(np.sum(masks["false_negative"]))
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    total = int(labels.shape[0])
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "false_positive_rate": float(fp / negatives) if negatives else 0.0,
        "false_negative_rate": float(fn / positives) if positives else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def _row_item(
    row_index: int,
    features: np.ndarray,
    label: int,
    predicted_label: int,
    probability: float,
    loss: float,
    margin: float,
    confidence: float,
) -> dict[str, Any]:
    if label == 0 and predicted_label == 1:
        bucket = "false_positive"
    elif label == 1 and predicted_label == 0:
        bucket = "false_negative"
    elif label == 1:
        bucket = "true_positive"
    else:
        bucket = "true_negative"
    return {
        "row_index": int(row_index),
        "label": int(label),
        "predicted_label": int(predicted_label),
        "bucket": bucket,
        "probability": float(probability),
        "loss": float(loss),
        "margin": float(margin),
        "confidence": float(confidence),
        "feature_preview": [float(value) for value in features[:8]],
    }


def _rank_rows(
    items: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    threshold: float,
    limit: int,
    near: bool = False,
) -> list[dict[str, Any]]:
    selected = [items[int(index)] for index in indices]
    if near:
        selected.sort(key=lambda item: (float(item["margin"]), -float(item["loss"]), int(item["row_index"])))
    else:
        selected.sort(
            key=lambda item: (
                int(item["predicted_label"] == item["label"]),
                -float(item["loss"]),
                -abs(float(item["probability"]) - threshold),
                int(item["row_index"]),
            )
        )
    return selected[:limit]


def _feature_error_shifts(
    features: np.ndarray,
    error_mask: np.ndarray,
    correct_mask: np.ndarray,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not np.any(error_mask) or not np.any(correct_mask):
        return []
    error_values = features[error_mask]
    correct_values = features[correct_mask]
    pooled_scale = np.std(features, axis=0)
    pooled_scale = np.where(pooled_scale < 1e-8, 1.0, pooled_scale)
    error_mean = np.mean(error_values, axis=0)
    correct_mean = np.mean(correct_values, axis=0)
    standardized_shift = (error_mean - correct_mean) / pooled_scale
    rows = [
        {
            "feature_index": int(index),
            "error_mean": float(error_mean[index]),
            "correct_mean": float(correct_mean[index]),
            "standardized_shift": float(standardized_shift[index]),
            "abs_standardized_shift": float(abs(standardized_shift[index])),
        }
        for index in range(features.shape[1])
    ]
    rows.sort(key=lambda item: (-float(item["abs_standardized_shift"]), int(item["feature_index"])))
    return rows[: max(1, int(limit))]


def _dominant_error_type(confusion: dict[str, Any]) -> str:
    fp = int(confusion.get("false_positive", 0))
    fn = int(confusion.get("false_negative", 0))
    if fp > fn:
        return "false_positive"
    if fn > fp:
        return "false_negative"
    if fp == 0 and fn == 0:
        return "none"
    return "balanced_errors"


def _top_error_row(items: list[dict[str, Any]], error_mask: np.ndarray) -> int | None:
    errors = [items[index] for index in np.where(error_mask)[0]]
    if not errors:
        return None
    errors.sort(key=lambda item: (-float(item["loss"]), int(item["row_index"])))
    return int(errors[0]["row_index"])


def _recommendations(
    *,
    confusion: dict[str, Any],
    error_count: int,
    high_confidence_error_count: int,
    near_threshold_count: int,
    feature_shifts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []

    def add(priority: str, category: str, title: str, action: str, reason: str, score: float) -> None:
        recommendations.append(
            {
                "priority": priority,
                "category": category,
                "title": title,
                "action": action,
                "reason": reason,
                "priority_score": float(score),
            }
        )

    if error_count == 0:
        add(
            "low",
            "validation",
            "Preserve the clean error atlas",
            "Export the report and verify on fresh or held-out rows before promotion.",
            "No errors were observed on the loaded dataset.",
            35.0,
        )
    if high_confidence_error_count:
        add(
            "high",
            "label_review",
            "Review high-confidence errors first",
            "Inspect the high-confidence error rows for label mistakes, leakage, or missing features.",
            f"{high_confidence_error_count} error row(s) are far from the decision threshold.",
            90.0,
        )
    if int(confusion.get("false_negative", 0)) > int(confusion.get("false_positive", 0)):
        add(
            "medium",
            "threshold",
            "Reduce missed positives",
            "Run Threshold tradeoff and compare a higher-recall operating point.",
            "False negatives dominate the current error profile.",
            68.0,
        )
    elif int(confusion.get("false_positive", 0)) > int(confusion.get("false_negative", 0)):
        add(
            "medium",
            "threshold",
            "Reduce false alarms",
            "Run Decision curve or Threshold tradeoff with false-positive costs.",
            "False positives dominate the current error profile.",
            66.0,
        )
    if near_threshold_count >= max(3, int(0.10 * sum(int(confusion.get(key, 0)) for key in ("true_positive", "true_negative", "false_positive", "false_negative")))):
        add(
            "medium",
            "active_learning",
            "Prioritize boundary labels",
            "Export batch predictions or review near-threshold rows before retraining.",
            f"{near_threshold_count} row(s) sit near the current threshold.",
            60.0,
        )
    if feature_shifts and float(feature_shifts[0]["abs_standardized_shift"]) >= 0.75:
        top = feature_shifts[0]
        add(
            "medium",
            "feature_quality",
            "Inspect the strongest error-shifted feature",
            f"Review feature x{int(top['feature_index']) + 1}; error rows differ from correct rows by {float(top['standardized_shift']):.2f} standard deviations.",
            "Errors are concentrated in a shifted feature region.",
            58.0,
        )
    ordered = sorted(recommendations, key=lambda item: (-item["priority_score"], item["category"], item["title"]))
    for index, item in enumerate(ordered, start=1):
        item["rank"] = index
    return ordered[:6]
