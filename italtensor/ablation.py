from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_ablation_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    seed: int = 42,
    max_features: int = 16,
) -> dict[str, Any]:
    """Rank raw features by how much model quality drops when each one is disrupted."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Ablation diagnostics features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Ablation diagnostics feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Ablation diagnostics need at least one sample.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Ablation diagnostics threshold must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    base_probabilities = _predict_raw(model, x, preprocessor)
    base_metrics = _compact_metrics(y, base_probabilities, threshold)
    base_predictions = (base_probabilities >= threshold).astype(np.int32)
    neutral_values = np.median(x, axis=0)
    selected_indices = set(preprocessor.selected_indices) if preprocessor is not None and preprocessor.selected_indices is not None else None

    results: list[dict[str, Any]] = []
    for feature_index in range(min(x.shape[1], max(1, int(max_features)))):
        neutralized = x.copy()
        neutralized[:, feature_index] = neutral_values[feature_index]
        neutral_probabilities = _predict_raw(model, neutralized, preprocessor)
        neutral_metrics = _compact_metrics(y, neutral_probabilities, threshold)
        neutral_predictions = (neutral_probabilities >= threshold).astype(np.int32)

        permuted = x.copy()
        permuted[:, feature_index] = permuted[rng.permutation(x.shape[0]), feature_index]
        permutation_probabilities = _predict_raw(model, permuted, preprocessor)
        permutation_metrics = _compact_metrics(y, permutation_probabilities, threshold)
        permutation_predictions = (permutation_probabilities >= threshold).astype(np.int32)

        neutral_f1_drop = float(base_metrics["f1"] - neutral_metrics["f1"])
        permutation_f1_drop = float(base_metrics["f1"] - permutation_metrics["f1"])
        neutral_flip_rate = float(np.mean(neutral_predictions != base_predictions))
        permutation_flip_rate = float(np.mean(permutation_predictions != base_predictions))
        neutral_probability_shift = float(np.mean(np.abs(neutral_probabilities - base_probabilities)))
        permutation_probability_shift = float(np.mean(np.abs(permutation_probabilities - base_probabilities)))
        label_correlation = _safe_correlation(x[:, feature_index], y)
        reliance_score = float(
            max(0.0, neutral_f1_drop, permutation_f1_drop)
            + 0.25 * max(neutral_flip_rate, permutation_flip_rate)
            + 0.1 * max(neutral_probability_shift, permutation_probability_shift)
        )

        results.append(
            {
                "feature_index": int(feature_index),
                "selected_by_preprocessor": selected_indices is None or feature_index in selected_indices,
                "neutral_value": float(neutral_values[feature_index]),
                "raw_mean": float(np.mean(x[:, feature_index])),
                "raw_std": float(np.std(x[:, feature_index])),
                "label_correlation": float(label_correlation),
                "f1": float(neutral_metrics["f1"]),
                "accuracy": float(neutral_metrics["accuracy"]),
                "balanced_accuracy": float(neutral_metrics["balanced_accuracy"]),
                "f1_drop": neutral_f1_drop,
                "accuracy_drop": float(base_metrics["accuracy"] - neutral_metrics["accuracy"]),
                "label_flip_rate": neutral_flip_rate,
                "mean_probability_shift": neutral_probability_shift,
                "permutation_f1": float(permutation_metrics["f1"]),
                "permutation_accuracy": float(permutation_metrics["accuracy"]),
                "permutation_f1_drop": permutation_f1_drop,
                "permutation_accuracy_drop": float(base_metrics["accuracy"] - permutation_metrics["accuracy"]),
                "permutation_label_flip_rate": permutation_flip_rate,
                "permutation_mean_probability_shift": permutation_probability_shift,
                "reliance_score": reliance_score,
                "risk_flags": _risk_flags(
                    neutral_f1_drop=neutral_f1_drop,
                    permutation_f1_drop=permutation_f1_drop,
                    label_flip_rate=max(neutral_flip_rate, permutation_flip_rate),
                    label_correlation=label_correlation,
                    raw_std=float(np.std(x[:, feature_index])),
                ),
            }
        )

    results.sort(
        key=lambda item: (
            -float(item["reliance_score"]),
            -max(float(item["f1_drop"]), float(item["permutation_f1_drop"])),
            int(item["feature_index"]),
        )
    )
    top_results = results[: min(len(results), int(max_features))]
    max_f1_drop = max((max(float(item["f1_drop"]), float(item["permutation_f1_drop"])) for item in top_results), default=0.0)
    max_flip_rate = max(
        (
            max(float(item["label_flip_rate"]), float(item["permutation_label_flip_rate"]))
            for item in top_results
        ),
        default=0.0,
    )
    return {
        "seed": int(seed),
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "method": "median_neutralization_and_permutation",
        "base": base_metrics,
        "features": top_results,
        "summary": {
            "feature_count": len(results),
            "top_feature": _feature_name(top_results[0]) if top_results else "none",
            "max_f1_drop": float(max_f1_drop),
            "max_label_flip_rate": float(max_flip_rate),
            "high_reliance_count": sum(1 for item in results if float(item["reliance_score"]) >= 0.1),
            "label_proxy_count": sum(1 for item in results if abs(float(item["label_correlation"])) >= 0.85),
        },
    }


def format_ablation_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    base = report.get("base", {})
    return (
        "Ablation diagnostics: "
        f"base_f1={float(base.get('f1', 0.0)):.4f}, "
        f"top={summary.get('top_feature', 'none')}, "
        f"max_f1_drop={float(summary.get('max_f1_drop', 0.0)):.4f}, "
        f"max_flip={float(summary.get('max_label_flip_rate', 0.0)):.4f}, "
        f"proxy_flags={int(summary.get('label_proxy_count', 0))}"
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
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    )
    return {key: metrics[key] for key in keys}


def _safe_correlation(values: np.ndarray, labels: np.ndarray) -> float:
    feature = np.asarray(values, dtype=np.float32).reshape(-1)
    target = np.asarray(labels, dtype=np.float32).reshape(-1)
    if feature.size < 2 or float(np.std(feature)) < 1e-8 or float(np.std(target)) < 1e-8:
        return 0.0
    return float(np.corrcoef(feature, target)[0, 1])


def _risk_flags(
    *,
    neutral_f1_drop: float,
    permutation_f1_drop: float,
    label_flip_rate: float,
    label_correlation: float,
    raw_std: float,
) -> list[str]:
    flags: list[str] = []
    if max(neutral_f1_drop, permutation_f1_drop) >= 0.15:
        flags.append("high_f1_reliance")
    if label_flip_rate >= 0.2:
        flags.append("high_flip_rate")
    if abs(label_correlation) >= 0.85:
        flags.append("label_proxy")
    if raw_std < 1e-8:
        flags.append("constant_feature")
    return flags


def _feature_name(item: dict[str, Any]) -> str:
    return f"x{int(item['feature_index']) + 1}"
