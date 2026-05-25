from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DEFAULT_PERMUTATIONS = 200


def run_permutation_null_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    permutation_count: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare fixed model predictions with a shuffled-label null distribution."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Permutation null features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Permutation null features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Permutation null feature and label counts do not match.")
    if x.shape[0] < 4:
        raise ValueError("Permutation null diagnostics need at least four samples.")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Permutation null diagnostics require binary labels 0 or 1.")
    if np.unique(y).size < 2:
        raise ValueError("Permutation null diagnostics need both classes present.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Permutation null threshold must be between 0 and 1.")
    if int(permutation_count) < 10:
        raise ValueError("Permutation null diagnostics need at least ten permutations.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared).reshape(-1)
    if probabilities.shape[0] != y.shape[0]:
        raise ValueError("Model returned a probability count that does not match the dataset.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)

    observed = _selected_metrics(evaluate_predictions(y, probabilities, threshold))
    rng = np.random.default_rng(seed)
    null_metrics = {
        "f1": np.empty(int(permutation_count), dtype=np.float32),
        "accuracy": np.empty(int(permutation_count), dtype=np.float32),
        "balanced_accuracy": np.empty(int(permutation_count), dtype=np.float32),
    }
    for index in range(int(permutation_count)):
        shuffled = rng.permutation(y)
        shuffled_metrics = evaluate_predictions(shuffled, probabilities, threshold)
        for key in null_metrics:
            null_metrics[key][index] = float(shuffled_metrics[key])

    distribution = {
        key: _distribution_summary(values, observed[key])
        for key, values in null_metrics.items()
    }
    p_values = {
        key: _empirical_p_value(values, observed[key])
        for key, values in null_metrics.items()
    }
    predicted = (probabilities >= float(threshold)).astype(np.int32)
    warnings = _warnings(y, predicted, observed, p_values)
    f1_dist = distribution["f1"]
    summary = {
        "primary_metric": "f1",
        "observed_f1": float(observed["f1"]),
        "null_mean_f1": float(f1_dist["mean"]),
        "f1_gap": float(observed["f1"] - f1_dist["mean"]),
        "f1_z_score": float(f1_dist["z_score"]),
        "f1_p_value": float(p_values["f1"]),
        "observed_accuracy": float(observed["accuracy"]),
        "accuracy_p_value": float(p_values["accuracy"]),
        "verdict": _verdict(observed["f1"], f1_dist["mean"], p_values["f1"]),
        "warning": "; ".join(warnings) if warnings else None,
    }
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "seed": int(seed),
        "permutation_count": int(permutation_count),
        "class_counts": {
            "0": int(np.sum(y == 0)),
            "1": int(np.sum(y == 1)),
        },
        "predicted_class_counts": {
            "0": int(np.sum(predicted == 0)),
            "1": int(np.sum(predicted == 1)),
        },
        "observed": observed,
        "null_distribution": distribution,
        "p_values": p_values,
        "summary": summary,
    }


def format_permutation_null_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Permutation null: "
        f"F1={float(summary.get('observed_f1', 0.0)):.4f}, "
        f"null_mean={float(summary.get('null_mean_f1', 0.0)):.4f}, "
        f"gap={float(summary.get('f1_gap', 0.0)):.4f}, "
        f"p={float(summary.get('f1_p_value', 1.0)):.4f}, "
        f"verdict={summary.get('verdict', '-')}"
    )


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
        "predicted_positive_rate",
        "label_prevalence",
    )
    count_keys = {"true_positive", "true_negative", "false_positive", "false_negative"}
    return {
        key: int(metrics[key]) if key in count_keys else float(metrics[key])
        for key in keys
        if key in metrics
    }


def _distribution_summary(values: np.ndarray, observed_value: float) -> dict[str, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    z_score = 0.0 if std < 1e-9 else float((float(observed_value) - mean) / std)
    return {
        "mean": mean,
        "std": std,
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "observed_gap": float(float(observed_value) - mean),
        "z_score": z_score,
    }


def _empirical_p_value(values: np.ndarray, observed_value: float) -> float:
    exceedances = int(np.sum(values >= float(observed_value)))
    return float((exceedances + 1) / (values.shape[0] + 1))


def _verdict(observed_f1: float, null_mean_f1: float, f1_p_value: float) -> str:
    if float(observed_f1) <= float(null_mean_f1):
        return "no_observed_lift"
    if float(f1_p_value) <= 0.01:
        return "strong_signal"
    if float(f1_p_value) <= 0.05:
        return "signal"
    if float(f1_p_value) <= 0.10:
        return "weak_signal"
    return "not_significant"


def _warnings(
    labels: np.ndarray,
    predicted: np.ndarray,
    observed: dict[str, float | int],
    p_values: dict[str, float],
) -> list[str]:
    warnings: list[str] = []
    if np.unique(predicted).size < 2:
        warnings.append("model predicts one class at the current threshold")
    minority = int(min(np.sum(labels == 0), np.sum(labels == 1)))
    if minority < 5:
        warnings.append("minority class has fewer than five rows")
    if (
        labels.shape[0] < 30
        and float(observed.get("f1", 0.0)) >= 0.95
        and float(p_values.get("f1", 1.0)) <= 0.05
    ):
        warnings.append("tiny dataset with near-perfect observed score")
    return warnings
