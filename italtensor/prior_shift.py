from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DEFAULT_PREVALENCE_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_POPULATION_SIZE = 1000


def run_prior_shift_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    prevalence_grid: Sequence[float] | None = None,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, Any]:
    """Simulate predictive values under deployment prevalence/base-rate shifts."""
    if model is None:
        raise ValueError("Prior shift needs an active model.")
    threshold = _validate_threshold(threshold)
    population_size = max(1, int(population_size))
    x, y = _validate_inputs(features, labels)
    probabilities = _predict_probabilities(model, x, preprocessor)
    predicted = (probabilities >= threshold).astype(np.int32)
    current = _current_metrics(y, predicted, threshold)
    grid = _prevalence_grid(prevalence_grid, current["observed_prevalence"])
    points = [
        _simulate_point(
            prevalence,
            sensitivity=float(current.get("sensitivity", 0.0) or 0.0),
            specificity=float(current.get("specificity", 0.0) or 0.0),
            population_size=population_size,
        )
        for prevalence in grid
    ]
    summary = _summary(current, points)
    recommendations = _recommendations(summary)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": threshold,
        "population_size": population_size,
        "prevalence_grid": [float(value) for value in grid],
        "dataset_fingerprint": prior_shift_dataset_fingerprint(x, y),
        "current": current,
        "points": points,
        "summary": summary,
        "recommendations": recommendations,
    }


def format_prior_shift_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Prior shift: "
        f"verdict={summary.get('verdict', '-')}, "
        f"observed_prev={float(summary.get('observed_prevalence', 0.0)):.4f}, "
        f"min_ppv={float(summary.get('min_ppv', 0.0)):.4f}, "
        f"max_alerts_per_1000={float(summary.get('max_predicted_positive_per_1000', 0.0)):.1f}, "
        f"max_fp_per_1000={float(summary.get('max_false_positive_per_1000', 0.0)):.1f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def prior_shift_dataset_fingerprint(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> str:
    x, y = _validate_inputs(features, labels)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(value) for value in x.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    hasher.update(str(tuple(int(value) for value in y.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return hasher.hexdigest()


def _validate_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prior shift threshold must be finite and between 0 and 1.") from exc
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("Prior shift threshold must be finite and between 0 and 1.")
    return threshold


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prior shift features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Prior shift features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Prior shift feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("Prior shift needs at least one sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Prior shift features must be finite numbers.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prior shift labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Prior shift labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Prior shift labels must be binary 0/1.")
    return y.astype(np.int32)


def _predict_probabilities(model: Any, x: np.ndarray, preprocessor: FeatureStandardizer | None) -> np.ndarray:
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Prior shift preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    return np.clip(probabilities, 0.0, 1.0)


def _current_metrics(labels: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, Any]:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    predicted_positive = int(tp + fp)
    predicted_negative = int(tn + fn)
    sensitivity = float(tp / positives) if positives else 0.0
    specificity = float(tn / negatives) if negatives else 0.0
    ppv = float(tp / predicted_positive) if predicted_positive else 0.0
    npv = float(tn / predicted_negative) if predicted_negative else 0.0
    fpr = float(fp / negatives) if negatives else 0.0
    fnr = float(fn / positives) if positives else 0.0
    return {
        "threshold": float(threshold),
        "positive_count": positives,
        "negative_count": negatives,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "observed_prevalence": float(positives / max(1, labels.shape[0])),
        "predicted_positive_rate": float(predicted_positive / max(1, labels.shape[0])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "positive_likelihood_ratio": float(sensitivity / fpr) if fpr > 0.0 else None,
        "negative_likelihood_ratio": float(fnr / specificity) if specificity > 0.0 else None,
        "warning": None if positives and negatives else "Prior shift simulation needs both positive and negative labels.",
    }


def _prevalence_grid(prevalence_grid: Sequence[float] | None, observed_prevalence: float) -> np.ndarray:
    provided = prevalence_grid is not None
    raw = np.asarray(DEFAULT_PREVALENCE_GRID if prevalence_grid is None else list(prevalence_grid), dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if provided and raw.size == 0:
        raise ValueError("Prior shift prevalence grid must contain at least one finite prevalence.")
    if observed_prevalence > 0.0:
        raw = np.concatenate([raw, np.asarray([observed_prevalence], dtype=np.float64)])
    if raw.size == 0:
        raise ValueError("Prior shift prevalence grid must contain at least one finite prevalence.")
    raw = np.clip(raw, 0.0, 1.0)
    return np.unique(raw)


def _simulate_point(
    prevalence: float,
    *,
    sensitivity: float,
    specificity: float,
    population_size: int,
) -> dict[str, float]:
    prevalence = float(prevalence)
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    specificity = float(np.clip(specificity, 0.0, 1.0))
    fpr = 1.0 - specificity
    fnr = 1.0 - sensitivity
    tp_rate = prevalence * sensitivity
    fp_rate = (1.0 - prevalence) * fpr
    fn_rate = prevalence * fnr
    tn_rate = (1.0 - prevalence) * specificity
    predicted_positive_rate = tp_rate + fp_rate
    predicted_negative_rate = tn_rate + fn_rate
    ppv = float(tp_rate / predicted_positive_rate) if predicted_positive_rate > 0.0 else 0.0
    npv = float(tn_rate / predicted_negative_rate) if predicted_negative_rate > 0.0 else 0.0
    return {
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_negative_rate": predicted_negative_rate,
        "false_discovery_rate": float(1.0 - ppv) if predicted_positive_rate > 0.0 else 0.0,
        "false_omission_rate": float(1.0 - npv) if predicted_negative_rate > 0.0 else 0.0,
        "expected_true_positive": float(tp_rate * population_size),
        "expected_false_positive": float(fp_rate * population_size),
        "expected_false_negative": float(fn_rate * population_size),
        "expected_true_negative": float(tn_rate * population_size),
        "expected_predicted_positive": float(predicted_positive_rate * population_size),
        "expected_predicted_negative": float(predicted_negative_rate * population_size),
    }


def _summary(current: dict[str, Any], points: list[dict[str, float]]) -> dict[str, Any]:
    positive_count = int(current.get("positive_count", 0) or 0)
    negative_count = int(current.get("negative_count", 0) or 0)
    observed_prevalence = float(current.get("observed_prevalence", 0.0) or 0.0)
    current_ppv = float(current.get("positive_predictive_value", 0.0) or 0.0)
    min_ppv_point = min(points, key=lambda item: float(item.get("positive_predictive_value", 0.0)), default={})
    max_fp_point = max(points, key=lambda item: float(item.get("expected_false_positive", 0.0)), default={})
    max_alert_point = max(points, key=lambda item: float(item.get("expected_predicted_positive", 0.0)), default={})
    min_npv_point = min(points, key=lambda item: float(item.get("negative_predictive_value", 0.0)), default={})
    min_ppv = float(min_ppv_point.get("positive_predictive_value", 0.0) or 0.0)
    ppv_drop = float(max(0.0, current_ppv - min_ppv))
    verdict = _verdict(current, min_ppv=min_ppv, ppv_drop=ppv_drop)
    return {
        "verdict": verdict,
        "observed_prevalence": observed_prevalence,
        "sensitivity": float(current.get("sensitivity", 0.0) or 0.0),
        "specificity": float(current.get("specificity", 0.0) or 0.0),
        "current_ppv": current_ppv,
        "current_npv": float(current.get("negative_predictive_value", 0.0) or 0.0),
        "min_ppv": min_ppv,
        "min_ppv_prevalence": float(min_ppv_point.get("prevalence", 0.0) or 0.0),
        "min_npv": float(min_npv_point.get("negative_predictive_value", 0.0) or 0.0),
        "min_npv_prevalence": float(min_npv_point.get("prevalence", 0.0) or 0.0),
        "max_false_positive_per_1000": float(max_fp_point.get("expected_false_positive", 0.0) or 0.0),
        "max_false_positive_prevalence": float(max_fp_point.get("prevalence", 0.0) or 0.0),
        "max_predicted_positive_per_1000": float(max_alert_point.get("expected_predicted_positive", 0.0) or 0.0),
        "max_predicted_positive_prevalence": float(max_alert_point.get("prevalence", 0.0) or 0.0),
        "ppv_drop_from_observed_to_min_grid": ppv_drop,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "recommended_next_step": _next_step(verdict),
    }


def _verdict(current: dict[str, Any], *, min_ppv: float, ppv_drop: float) -> str:
    positives = int(current.get("positive_count", 0) or 0)
    negatives = int(current.get("negative_count", 0) or 0)
    sensitivity = float(current.get("sensitivity", 0.0) or 0.0)
    specificity = float(current.get("specificity", 0.0) or 0.0)
    if positives == 0 or negatives == 0:
        return "no_two_class_evidence"
    if sensitivity < 0.50 or specificity < 0.70:
        return "prior_shift_review"
    if min_ppv < 0.20 or ppv_drop >= 0.25:
        return "prevalence_shift_risk"
    if min_ppv < 0.40 or ppv_drop >= 0.15:
        return "prior_shift_review"
    return "prior_shift_stable"


def _next_step(verdict: str) -> str:
    if verdict == "prior_shift_stable":
        return "Document expected PPV/NPV across plausible deployment prevalences and rerun after threshold changes."
    if verdict == "prevalence_shift_risk":
        return "Do not rely on current PPV at lower base rates; validate prevalence, threshold, and review capacity on deployment-like rows."
    if verdict == "no_two_class_evidence":
        return "Load validation rows containing both labels before simulating deployment prevalence."
    return "Review threshold, false-positive rate, and plausible deployment prevalence before promotion."


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = str(summary.get("verdict", "prior_shift_review"))
    if verdict == "prior_shift_stable":
        priority = "low"
        score = 42.0
        title = "Keep prevalence assumptions with the model evidence"
    elif verdict == "prevalence_shift_risk":
        priority = "high"
        score = 84.0
        title = "Validate base-rate shift before operational use"
    elif verdict == "no_two_class_evidence":
        priority = "medium"
        score = 65.0
        title = "Add two-class validation evidence"
    else:
        priority = "medium"
        score = 68.0
        title = "Review prior-shift sensitivity"
    return [
        {
            "rank": 1,
            "priority_score": score,
            "priority": priority,
            "category": "prevalence",
            "title": title,
            "reason": f"Prior-shift verdict is {verdict}.",
            "action": summary.get("recommended_next_step"),
        }
    ]
