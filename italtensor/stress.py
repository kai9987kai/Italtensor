from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DEFAULT_NOISE_LEVELS = (0.05, 0.1, 0.2)
DEFAULT_DROPOUT_RATES = (0.1, 0.25)


def run_stress_suite(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    seed: int = 42,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    dropout_rates: Sequence[float] = DEFAULT_DROPOUT_RATES,
    shift_magnitude: float = 1.0,
    max_feature_shifts: int = 8,
) -> dict[str, Any]:
    """Run deterministic perturbation checks against a trained model."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Stress features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Stress feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Stress suite needs at least one sample.")

    rng = np.random.default_rng(seed)
    base_probabilities = _predict_raw(model, x, preprocessor)
    base_metrics = _compact_metrics(y, base_probabilities, threshold)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale).astype(np.float32)

    perturbations: list[dict[str, Any]] = []
    for level in noise_levels:
        noisy = x + rng.normal(0.0, float(level), size=x.shape).astype(np.float32) * feature_scale
        perturbations.append(
            _perturbation_result("gaussian_noise", float(level), model, x, noisy, y, base_probabilities, preprocessor, threshold)
        )

    for rate in dropout_rates:
        clipped_rate = min(max(float(rate), 0.0), 1.0)
        mask = rng.random(size=x.shape) < clipped_rate
        dropped = x.copy()
        dropped[mask] = np.broadcast_to(feature_mean, x.shape)[mask]
        perturbations.append(
            _perturbation_result("feature_dropout", clipped_rate, model, x, dropped, y, base_probabilities, preprocessor, threshold)
        )

    shift_results: list[dict[str, Any]] = []
    for feature_index in range(x.shape[1]):
        if feature_index >= int(max_feature_shifts):
            break
        for direction in (-1.0, 1.0):
            shifted = x.copy()
            shifted[:, feature_index] += direction * float(shift_magnitude) * feature_scale[feature_index]
            shift_results.append(
                _perturbation_result(
                    "feature_shift",
                    float(direction * shift_magnitude),
                    model,
                    x,
                    shifted,
                    y,
                    base_probabilities,
                    preprocessor,
                    threshold,
                    feature_index=feature_index,
                )
            )
    shift_results.sort(key=lambda item: (float(item["f1_delta"]), -float(item["label_flip_rate"])))
    perturbations.extend(shift_results[: min(len(shift_results), int(max_feature_shifts))])

    worst_f1 = min((float(item["f1"]) for item in perturbations), default=float(base_metrics["f1"]))
    max_flip = max((float(item["label_flip_rate"]) for item in perturbations), default=0.0)
    base_f1 = max(float(base_metrics["f1"]), 1e-9)
    return {
        "seed": int(seed),
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "base": base_metrics,
        "perturbations": perturbations,
        "summary": {
            "worst_f1": float(worst_f1),
            "base_f1": float(base_metrics["f1"]),
            "stress_f1_ratio": float(worst_f1 / base_f1),
            "max_label_flip_rate": float(max_flip),
            "worst_case": _worst_case_name(perturbations),
        },
    }


def format_stress_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    base = report.get("base", {})
    return (
        "Stress suite: "
        f"base_f1={float(base.get('f1', 0.0)):.4f}, "
        f"worst_f1={float(summary.get('worst_f1', 0.0)):.4f}, "
        f"ratio={float(summary.get('stress_f1_ratio', 0.0)):.4f}, "
        f"max_flip={float(summary.get('max_label_flip_rate', 0.0)):.4f}, "
        f"worst={summary.get('worst_case', '-')}"
    )


def _predict_raw(
    model: Any,
    raw_features: np.ndarray,
    preprocessor: FeatureStandardizer | None,
) -> np.ndarray:
    prepared = preprocessor.transform(raw_features) if preprocessor is not None else raw_features
    return predict_probability(model, prepared)


def _compact_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = evaluate_predictions(labels, probabilities, threshold)
    keys = (
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "validation_loss",
        "brier_score",
        "ece",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _perturbation_result(
    kind: str,
    level: float,
    model: Any,
    original_features: np.ndarray,
    perturbed_features: np.ndarray,
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    preprocessor: FeatureStandardizer | None,
    threshold: float,
    *,
    feature_index: int | None = None,
) -> dict[str, Any]:
    probabilities = _predict_raw(model, perturbed_features, preprocessor)
    metrics = _compact_metrics(labels, probabilities, threshold)
    base_pred = (base_probabilities >= threshold).astype(np.int32)
    perturbed_pred = (probabilities >= threshold).astype(np.int32)
    result: dict[str, Any] = {
        "kind": kind,
        "level": float(level),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1_delta": float(float(metrics["f1"]) - float(evaluate_predictions(labels, base_probabilities, threshold)["f1"])),
        "mean_probability_shift": float(np.mean(np.abs(probabilities - base_probabilities))),
        "label_flip_rate": float(np.mean(perturbed_pred != base_pred)),
    }
    if feature_index is not None:
        result["feature_index"] = int(feature_index)
    return result


def _worst_case_name(perturbations: list[dict[str, Any]]) -> str:
    if not perturbations:
        return "none"
    worst = min(perturbations, key=lambda item: (float(item["f1"]), -float(item["label_flip_rate"])))
    if "feature_index" in worst:
        return f"{worst['kind']}[x{int(worst['feature_index']) + 1}]@{float(worst['level']):.2f}"
    return f"{worst['kind']}@{float(worst['level']):.2f}"
