from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions, probability_diagnostics
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_external_holdout_evaluation(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    reference_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    reference_labels: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the active model on a separate labeled holdout dataset."""
    if model is None:
        raise ValueError("External holdout evaluation needs an active model.")
    threshold = _validate_threshold(threshold)
    x, y = _validate_inputs(features, labels)
    probabilities = _predict_probabilities(model, x, preprocessor)
    metrics = _selected_metrics(evaluate_predictions(y, probabilities, threshold))
    diagnostics = probability_diagnostics(y, probabilities)
    reference = _reference_comparison(
        x,
        y,
        reference_features=reference_features,
        reference_labels=reference_labels,
    )
    summary = _summary(metrics, diagnostics, reference)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": threshold,
        "metrics": metrics,
        "probability_diagnostics": {
            "brier_score": float(diagnostics.get("brier_score", 0.0)),
            "log_loss": float(diagnostics.get("log_loss", 0.0)),
            "expected_calibration_error": float(diagnostics.get("expected_calibration_error", 0.0)),
            "max_calibration_error": float(diagnostics.get("max_calibration_error", 0.0)),
            "mean_probability": float(diagnostics.get("mean_probability", 0.0)),
            "label_prevalence": float(diagnostics.get("label_prevalence", 0.0)),
            "predicted_positive_rate": float(diagnostics.get("predicted_positive_rate", 0.0)),
        },
        "reference_comparison": reference,
        "summary": summary,
        "recommendations": _recommendations(summary),
    }


def format_external_holdout_summary(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    summary = report.get("summary", {})
    return (
        "External holdout: "
        f"verdict={summary.get('verdict', '-')}, "
        f"n={int(report.get('sample_count', 0))}, "
        f"F1={float(metrics.get('f1', 0.0)):.4f}, "
        f"balanced_acc={float(metrics.get('balanced_accuracy', 0.0)):.4f}, "
        f"ECE={float(report.get('probability_diagnostics', {}).get('expected_calibration_error', 0.0)):.4f}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def _validate_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("External holdout threshold must be finite and between 0 and 1.") from exc
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("External holdout threshold must be finite and between 0 and 1.")
    return threshold


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("External holdout features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("External holdout features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("External holdout feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("External holdout needs at least one sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("External holdout features must be finite numbers.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("External holdout labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("External holdout labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("External holdout labels must be binary 0/1.")
    return y.astype(np.int32)


def _predict_probabilities(model: Any, x: np.ndarray, preprocessor: FeatureStandardizer | None) -> np.ndarray:
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("External holdout preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    return np.clip(probabilities, 0.0, 1.0)


def _selected_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "validation_loss",
        "brier_score",
        "ece",
        "roc_auc",
        "average_precision",
        "predicted_positive_rate",
        "label_prevalence",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _reference_comparison(
    holdout_features: np.ndarray,
    holdout_labels: np.ndarray,
    *,
    reference_features: Sequence[Sequence[float]] | np.ndarray | None,
    reference_labels: Sequence[int] | np.ndarray | None,
) -> dict[str, Any] | None:
    if reference_features is None or reference_labels is None:
        return None
    reference_x, reference_y = _validate_inputs(reference_features, reference_labels)
    if reference_x.shape[1] != holdout_features.shape[1]:
        raise ValueError("External holdout reference feature count does not match holdout input dimension.")
    ref_mean = reference_x.mean(axis=0)
    holdout_mean = holdout_features.mean(axis=0)
    ref_std = reference_x.std(axis=0)
    ref_std = np.where(ref_std < 1e-6, 1.0, ref_std)
    standardized_shift = np.abs((holdout_mean - ref_mean) / ref_std)
    top_index = int(np.argmax(standardized_shift)) if standardized_shift.size else None
    prevalence_shift = float(np.mean(holdout_labels) - np.mean(reference_y))
    return {
        "reference_count": int(reference_x.shape[0]),
        "holdout_count": int(holdout_features.shape[0]),
        "max_standardized_mean_shift": float(np.max(standardized_shift)) if standardized_shift.size else 0.0,
        "top_shift_feature": top_index,
        "top_shift_reference_mean": float(ref_mean[top_index]) if top_index is not None else None,
        "top_shift_holdout_mean": float(holdout_mean[top_index]) if top_index is not None else None,
        "label_prevalence_shift": prevalence_shift,
        "reference_label_prevalence": float(np.mean(reference_y)),
        "holdout_label_prevalence": float(np.mean(holdout_labels)),
    }


def _summary(
    metrics: dict[str, float | int],
    diagnostics: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    f1 = float(metrics.get("f1", 0.0) or 0.0)
    balanced_accuracy = float(metrics.get("balanced_accuracy", 0.0) or 0.0)
    ece = float(diagnostics.get("expected_calibration_error", 0.0) or 0.0)
    brier = float(diagnostics.get("brier_score", 0.0) or 0.0)
    max_shift = float((reference or {}).get("max_standardized_mean_shift", 0.0) or 0.0)
    prevalence_shift = abs(float((reference or {}).get("label_prevalence_shift", 0.0) or 0.0))
    verdict = _verdict(f1, balanced_accuracy, ece, brier, max_shift, prevalence_shift)
    return {
        "verdict": verdict,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "expected_calibration_error": ece,
        "brier_score": brier,
        "max_standardized_mean_shift": max_shift,
        "label_prevalence_shift_abs": prevalence_shift,
        "recommendation": _next_step(verdict),
    }


def _verdict(
    f1: float,
    balanced_accuracy: float,
    ece: float,
    brier: float,
    max_shift: float,
    prevalence_shift: float,
) -> str:
    if f1 < 0.55 or balanced_accuracy < 0.55:
        return "holdout_failure"
    if ece >= 0.14 or brier >= 0.30:
        return "holdout_calibration_review"
    if max_shift >= 1.0 or prevalence_shift >= 0.20:
        return "holdout_shift_review"
    if f1 < 0.70 or balanced_accuracy < 0.70:
        return "holdout_performance_review"
    return "holdout_pass"


def _next_step(verdict: str) -> str:
    if verdict == "holdout_failure":
        return "Do not promote yet; investigate external holdout errors before saving the model."
    if verdict == "holdout_calibration_review":
        return "Run calibration diagnostics on representative evaluation rows, then re-score this external holdout before using probabilities."
    if verdict == "holdout_shift_review":
        return "Compare holdout rows with the loaded dataset and validate whether this shift matches deployment."
    if verdict == "holdout_performance_review":
        return "Review holdout errors and consider collecting more representative labeled rows."
    return "Keep this external holdout result with the model evidence."


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = str(summary.get("verdict", "holdout_performance_review"))
    priority = "low" if verdict == "holdout_pass" else ("high" if verdict == "holdout_failure" else "medium")
    category = "evidence" if verdict == "holdout_pass" else "external_validation"
    return [
        {
            "rank": 1,
            "priority_score": 90.0 if priority == "high" else (65.0 if priority == "medium" else 25.0),
            "priority": priority,
            "category": category,
            "title": "External holdout review" if verdict != "holdout_pass" else "Retain external holdout evidence",
            "action": str(summary.get("recommendation") or _next_step(verdict)),
        }
    ]
