"""Estimate how much individual labels affect active-model evaluation metrics."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_label_sensitivity(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    max_items: int = 12,
    material_f1_delta: float = 0.05,
) -> dict[str, Any]:
    """Rank rows by fixed-prediction metric movement if their label were flipped."""
    x, y = _validate_inputs(features, labels)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    max_items = max(1, int(max_items))
    material_f1_delta = max(0.0, float(material_f1_delta))

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")

    baseline = evaluate_predictions(y, probabilities, threshold=threshold)
    baseline_metrics = {
        key: float(value) if isinstance(value, (float, np.floating)) else int(value)
        for key, value in baseline.items()
        if key
        in {
            "f1",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "brier_score",
            "log_loss",
            "ece",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        }
    }
    predicted = (probabilities >= threshold).astype(np.int32)
    rows = [
        _row_sensitivity(
            index=index,
            features=x[index],
            labels=y,
            probabilities=probabilities,
            predicted=int(predicted[index]),
            baseline=baseline,
            threshold=threshold,
            material_f1_delta=material_f1_delta,
        )
        for index in range(x.shape[0])
    ]
    rows.sort(key=lambda row: (-float(row["sensitivity_score"]), int(row["row_index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    recommendations = _recommendations(rows, material_f1_delta=material_f1_delta)
    summary = _summary(rows, recommendations, material_f1_delta=material_f1_delta)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": threshold,
        "material_f1_delta": material_f1_delta,
        "dataset_fingerprint": label_sensitivity_dataset_fingerprint(x, y),
        "primary_metric": "f1",
        "observed": baseline_metrics,
        "baseline_metrics": baseline_metrics,
        "summary": summary,
        "recommendations": recommendations,
        "rows": rows[:max_items],
        "suspect_label_rows": [row for row in rows if row["label_flip_direction"] == "improves_metrics"][:max_items],
        "anchor_rows": [row for row in rows if row["label_flip_direction"] == "hurts_metrics"][:max_items],
    }


def run_label_sensitivity_diagnostics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the post-hoc label-sensitivity diagnostic."""
    return run_label_sensitivity(*args, **kwargs)


def format_label_sensitivity_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Label sensitivity: "
        f"verdict={summary.get('verdict', '-')}, "
        f"priority={summary.get('priority', '-')}, "
        f"suspect={int(summary.get('suspect_label_count', 0) or 0)}, "
        f"anchors={int(summary.get('anchor_row_count', 0) or 0)}, "
        f"max_abs_f1_delta={float(summary.get('max_abs_f1_delta', 0.0) or 0.0):.4f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def label_sensitivity_dataset_fingerprint(features: Any, labels: Any) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_inputs(features: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Label sensitivity features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Label sensitivity features must be a 2D array.")
    if x.shape[0] < 2:
        raise ValueError("Label sensitivity needs at least two labeled rows.")
    if x.shape[1] < 1:
        raise ValueError("Label sensitivity needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Label sensitivity features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Label sensitivity labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Label sensitivity feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Label sensitivity labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Label sensitivity requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Label sensitivity requires binary labels 0 or 1.")
    return x, y


def _row_sensitivity(
    *,
    index: int,
    features: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted: int,
    baseline: dict[str, float | int],
    threshold: float,
    material_f1_delta: float,
) -> dict[str, Any]:
    flipped = labels.copy()
    flipped[index] = 1 - flipped[index]
    flipped_metrics = evaluate_predictions(flipped, probabilities, threshold=threshold)
    f1_delta = float(flipped_metrics["f1"] - baseline["f1"])
    accuracy_delta = float(flipped_metrics["accuracy"] - baseline["accuracy"])
    balanced_delta = float(flipped_metrics["balanced_accuracy"] - baseline["balanced_accuracy"])
    brier_delta = float(flipped_metrics["brier_score"] - baseline["brier_score"])
    log_loss_delta = float(flipped_metrics["log_loss"] - baseline["log_loss"])
    ece_delta = float(flipped_metrics["ece"] - baseline["ece"])
    current_loss = _binary_loss(int(labels[index]), float(probabilities[index]))
    flipped_loss = _binary_loss(int(flipped[index]), float(probabilities[index]))
    confidence = float(probabilities[index] if predicted == 1 else 1.0 - probabilities[index])
    sensitivity_score = float(
        abs(f1_delta)
        + 0.60 * abs(balanced_delta)
        + 0.35 * abs(accuracy_delta)
        + 0.20 * min(abs(log_loss_delta), 1.0)
    )
    if f1_delta >= material_f1_delta or (f1_delta > 0.0 and flipped_loss < current_loss):
        direction = "improves_metrics"
    elif f1_delta <= -material_f1_delta or (f1_delta < 0.0 and flipped_loss > current_loss):
        direction = "hurts_metrics"
    else:
        direction = "neutral"
    return {
        "row_index": int(index),
        "label": int(labels[index]),
        "flipped_label": int(flipped[index]),
        "predicted_label": int(predicted),
        "probability": float(probabilities[index]),
        "confidence": confidence,
        "current_loss": current_loss,
        "flipped_loss": flipped_loss,
        "f1_after_flip": float(flipped_metrics["f1"]),
        "f1_delta_if_flipped": f1_delta,
        "accuracy_delta_if_flipped": accuracy_delta,
        "balanced_accuracy_delta_if_flipped": balanced_delta,
        "brier_delta_if_flipped": brier_delta,
        "log_loss_delta_if_flipped": log_loss_delta,
        "ece_delta_if_flipped": ece_delta,
        "sensitivity_score": sensitivity_score,
        "label_flip_direction": direction,
        "risk_flags": _risk_flags(
            label=int(labels[index]),
            predicted=predicted,
            probability=float(probabilities[index]),
            f1_delta=f1_delta,
            current_loss=current_loss,
            flipped_loss=flipped_loss,
            material_f1_delta=material_f1_delta,
        ),
        "recommended_action": _recommended_action(direction, f1_delta, material_f1_delta),
        "feature_preview": [float(value) for value in features[:8]],
    }


def _binary_loss(label: int, probability: float) -> float:
    clipped = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    return float(-(label * np.log(clipped) + (1 - label) * np.log(1.0 - clipped)))


def _risk_flags(
    *,
    label: int,
    predicted: int,
    probability: float,
    f1_delta: float,
    current_loss: float,
    flipped_loss: float,
    material_f1_delta: float,
) -> list[str]:
    flags: list[str] = []
    confidence = probability if predicted == 1 else 1.0 - probability
    if f1_delta >= material_f1_delta:
        flags.append("flip_improves_f1")
    if f1_delta <= -material_f1_delta:
        flags.append("metric_anchor_label")
    if predicted != label and confidence >= 0.80:
        flags.append("confident_disagreement")
    if flipped_loss + 0.25 < current_loss:
        flags.append("flipped_label_fits_probability")
    if abs(f1_delta) < material_f1_delta and current_loss < 0.35:
        flags.append("stable_low_loss_label")
    if not flags:
        flags.append("low_sensitivity_label")
    return flags


def _recommended_action(direction: str, f1_delta: float, material_f1_delta: float) -> str:
    if direction == "improves_metrics":
        return "Review this label first; flipping it would improve fixed-prediction validation metrics."
    if direction == "hurts_metrics":
        return "Keep or document this label as a metric anchor; flipping it would weaken validation evidence."
    if abs(f1_delta) >= material_f1_delta / 2:
        return "Review if domain context is uncertain; this label has moderate metric sensitivity."
    return "No immediate label-sensitivity action."


def _recommendations(rows: list[dict[str, Any]], *, material_f1_delta: float) -> list[dict[str, Any]]:
    suspect = [row for row in rows if row["label_flip_direction"] == "improves_metrics"]
    anchors = [row for row in rows if row["label_flip_direction"] == "hurts_metrics"]
    high_loss_fit = [row for row in rows if "flipped_label_fits_probability" in row["risk_flags"]]
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

    if suspect:
        top = suspect[0]
        add(
            90.0,
            "high" if float(top["f1_delta_if_flipped"]) >= material_f1_delta else "medium",
            "label_quality",
            "Review labels that improve metrics when flipped",
            f"{len(suspect)} row(s) improve fixed-prediction F1 or loss when their label is flipped.",
            "Review suspect-label rows before promotion, then rerun Sample review, Error atlas, and Label sensitivity.",
        )
    if high_loss_fit:
        add(
            74.0,
            "medium",
            "label_quality",
            "Check labels that fit the opposite probability better",
            f"{len(high_loss_fit)} row(s) have lower loss under the flipped label.",
            "Use domain review before changing labels; do not auto-relabel from model probabilities alone.",
        )
    if anchors:
        add(
            46.0,
            "low",
            "metric_anchors",
            "Document highly sensitive anchor labels",
            f"{len(anchors)} row(s) materially hurt F1 if flipped.",
            "Use anchor rows as canaries or release-note examples when the labels are trusted.",
        )
    if not recs:
        add(
            20.0,
            "low",
            "ready",
            "No urgent label-sensitivity action",
            "No loaded label materially improves fixed-prediction F1 when flipped.",
            "Proceed with normal validation, external holdout, and promotion checks.",
        )

    recs.sort(key=lambda item: (-float(item["priority_score"]), item["category"], item["title"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs


def _summary(
    rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    *,
    material_f1_delta: float,
) -> dict[str, Any]:
    suspect = [row for row in rows if row["label_flip_direction"] == "improves_metrics"]
    anchors = [row for row in rows if row["label_flip_direction"] == "hurts_metrics"]
    max_abs = max((abs(float(row["f1_delta_if_flipped"])) for row in rows), default=0.0)
    max_improve = max((float(row["f1_delta_if_flipped"]) for row in suspect), default=0.0)
    high_count = sum(1 for item in recommendations if item["priority"] == "high")
    medium_count = sum(1 for item in recommendations if item["priority"] == "medium")
    if high_count or max_improve >= material_f1_delta:
        verdict = "review_sensitive_labels"
        priority = "high"
    elif medium_count or suspect:
        verdict = "targeted_label_review"
        priority = "medium"
    else:
        verdict = "label_sensitivity_stable"
        priority = "low"
    readiness = 100.0 - 18.0 * high_count - 8.0 * medium_count - min(18.0, len(suspect) * 4.0)
    top = recommendations[0] if recommendations else {}
    return {
        "verdict": verdict,
        "priority": priority,
        "readiness_score": round(max(0.0, min(100.0, readiness)), 1),
        "suspect_label_count": int(len(suspect)),
        "anchor_row_count": int(len(anchors)),
        "max_abs_f1_delta": float(max_abs),
        "max_improving_f1_delta": float(max_improve),
        "mean_abs_f1_delta": float(np.mean([abs(float(row["f1_delta_if_flipped"])) for row in rows])),
        "top_suspect_row": suspect[0]["row_index"] if suspect else None,
        "top_anchor_row": anchors[0]["row_index"] if anchors else None,
        "recommendation_count": int(len(recommendations)),
        "recommended_next_step": top.get("action"),
    }
