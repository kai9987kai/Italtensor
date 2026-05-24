from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import conformal_label_set, conformal_quantile
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3)


def run_conformal_set_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    calibration_fraction: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate split-conformal prediction-set behavior on the loaded dataset."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Conformal set features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Conformal set features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Conformal set feature and label counts do not match.")
    if x.shape[0] < 2:
        raise ValueError("Conformal set diagnostics need at least two samples.")
    if set(int(item) for item in np.unique(y)) - {0, 1}:
        raise ValueError("Conformal set diagnostics require binary labels 0 or 1.")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    clean_alphas = [float(alpha) for alpha in alphas]
    if not clean_alphas:
        raise ValueError("At least one conformal alpha is required.")
    if any(not 0.0 < alpha < 1.0 for alpha in clean_alphas):
        raise ValueError("Conformal alpha values must be between 0 and 1.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    if probabilities.shape[0] != y.shape[0]:
        raise ValueError("Model returned a probability count that does not match the dataset.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)

    calibration_indices, evaluation_indices, split_source = _split_calibration_evaluation(
        y,
        calibration_fraction=float(calibration_fraction),
        seed=int(seed),
    )
    calibration_labels = y[calibration_indices]
    calibration_probabilities = probabilities[calibration_indices]
    evaluation_labels = y[evaluation_indices]
    evaluation_probabilities = probabilities[evaluation_indices]
    points = [
        _alpha_point(
            alpha,
            calibration_labels,
            calibration_probabilities,
            evaluation_labels,
            evaluation_probabilities,
        )
        for alpha in sorted(clean_alphas)
    ]
    recommended = _recommended_point(points)
    efficient = min(points, key=lambda item: (float(item["mean_set_size"]), -float(item["empirical_coverage"])))
    reliable = max(points, key=lambda item: (float(item["coverage_gap"]), -float(item["mean_set_size"])))
    warning = _warning(y, split_source, evaluation_labels)

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "split": {
            "source": split_source,
            "calibration_fraction": float(calibration_fraction),
            "calibration_count": int(calibration_labels.shape[0]),
            "evaluation_count": int(evaluation_labels.shape[0]),
            "seed": int(seed),
        },
        "points": points,
        "summary": {
            "recommended_alpha": float(recommended["alpha"]),
            "recommended_quantile": float(recommended["quantile"]),
            "recommended_target_coverage": float(recommended["target_coverage"]),
            "recommended_empirical_coverage": float(recommended["empirical_coverage"]),
            "recommended_mean_set_size": float(recommended["mean_set_size"]),
            "recommended_singleton_rate": float(recommended["singleton_rate"]),
            "recommended_ambiguous_rate": float(recommended["ambiguous_rate"]),
            "best_efficiency_alpha": float(efficient["alpha"]),
            "most_reliable_alpha": float(reliable["alpha"]),
            "warning": warning,
        },
    }


def format_conformal_set_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Conformal sets: "
        f"alpha={float(summary.get('recommended_alpha', 0.0)):.4f}, "
        f"target={float(summary.get('recommended_target_coverage', 0.0)):.4f}, "
        f"coverage={float(summary.get('recommended_empirical_coverage', 0.0)):.4f}, "
        f"mean_size={float(summary.get('recommended_mean_set_size', 0.0)):.4f}, "
        f"singleton={float(summary.get('recommended_singleton_rate', 0.0)):.4f}, "
        f"ambiguous={float(summary.get('recommended_ambiguous_rate', 0.0)):.4f}"
    )


def _alpha_point(
    alpha: float,
    calibration_labels: np.ndarray,
    calibration_probabilities: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_probabilities: np.ndarray,
) -> dict[str, Any]:
    quantile = conformal_quantile(calibration_labels, calibration_probabilities, alpha=alpha)
    sets = [conformal_label_set(float(probability), quantile) for probability in evaluation_probabilities]
    set_sizes = np.asarray([len(item) for item in sets], dtype=np.int32)
    contains_truth = np.asarray(
        [int(label) in label_set for label, label_set in zip(evaluation_labels, sets, strict=True)],
        dtype=bool,
    )
    singleton_mask = set_sizes == 1
    singleton_predictions = np.asarray([label_set[0] if len(label_set) == 1 else -1 for label_set in sets])
    singleton_count = int(np.sum(singleton_mask))
    singleton_accuracy = (
        float(np.mean(singleton_predictions[singleton_mask] == evaluation_labels[singleton_mask]))
        if singleton_count
        else None
    )
    return {
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "quantile": float(quantile),
        "calibration_count": int(calibration_labels.shape[0]),
        "evaluation_count": int(evaluation_labels.shape[0]),
        "empirical_coverage": float(np.mean(contains_truth)) if contains_truth.size else 0.0,
        "coverage_gap": float(np.mean(contains_truth) - (1.0 - alpha)) if contains_truth.size else float(-(1.0 - alpha)),
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes.size else 0.0,
        "singleton_rate": float(np.mean(singleton_mask)) if set_sizes.size else 0.0,
        "ambiguous_rate": float(np.mean(set_sizes == 2)) if set_sizes.size else 0.0,
        "empty_rate": float(np.mean(set_sizes == 0)) if set_sizes.size else 0.0,
        "singleton_accuracy": singleton_accuracy,
        "singleton_count": singleton_count,
        "ambiguous_count": int(np.sum(set_sizes == 2)),
        "empty_count": int(np.sum(set_sizes == 0)),
        "missed_count": int(np.sum(~contains_truth)),
        "negative_coverage": _class_coverage(evaluation_labels, contains_truth, 0),
        "positive_coverage": _class_coverage(evaluation_labels, contains_truth, 1),
        "set_size_counts": {
            "0": int(np.sum(set_sizes == 0)),
            "1": int(np.sum(set_sizes == 1)),
            "2": int(np.sum(set_sizes == 2)),
        },
    }


def _split_calibration_evaluation(
    labels: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) == 2 and np.all(counts >= 2):
        calibration: list[int] = []
        evaluation: list[int] = []
        for label in unique:
            indices = np.where(labels == label)[0]
            rng.shuffle(indices)
            calibration_count = int(round(indices.shape[0] * calibration_fraction))
            calibration_count = max(1, min(indices.shape[0] - 1, calibration_count))
            calibration.extend(int(item) for item in indices[:calibration_count])
            evaluation.extend(int(item) for item in indices[calibration_count:])
        calibration_indices = np.asarray(calibration, dtype=np.int32)
        evaluation_indices = np.asarray(evaluation, dtype=np.int32)
        rng.shuffle(calibration_indices)
        rng.shuffle(evaluation_indices)
        return calibration_indices, evaluation_indices, "posthoc_stratified_split"

    indices = np.arange(labels.shape[0], dtype=np.int32)
    rng.shuffle(indices)
    calibration_count = int(round(indices.shape[0] * calibration_fraction))
    calibration_count = max(1, min(indices.shape[0] - 1, calibration_count))
    return indices[:calibration_count], indices[calibration_count:], "posthoc_global_split"


def _recommended_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [point for point in points if float(point["empirical_coverage"]) >= float(point["target_coverage"])]
    if candidates:
        return min(
            candidates,
            key=lambda item: (
                float(item["mean_set_size"]),
                -float(item["singleton_rate"]),
                float(item["alpha"]),
            ),
        )
    return max(
        points,
        key=lambda item: (
            float(item["coverage_gap"]),
            -float(item["mean_set_size"]),
            -float(item["singleton_rate"]),
        ),
    )


def _class_coverage(labels: np.ndarray, contains_truth: np.ndarray, class_label: int) -> float | None:
    mask = labels == int(class_label)
    if not np.any(mask):
        return None
    return float(np.mean(contains_truth[mask]))


def _warning(labels: np.ndarray, split_source: str, evaluation_labels: np.ndarray) -> str | None:
    unique = set(int(item) for item in np.unique(labels))
    messages: list[str] = []
    if unique == {0}:
        messages.append("All labels are negative; positive-class coverage is unavailable.")
    elif unique == {1}:
        messages.append("All labels are positive; negative-class coverage is unavailable.")
    if split_source == "posthoc_global_split":
        messages.append("Too few examples per class for a stratified calibration/evaluation split.")
    if len(set(int(item) for item in np.unique(evaluation_labels))) < 2:
        messages.append("Evaluation split has one observed class, so class-conditional coverage is partial.")
    return " ".join(messages) if messages else None
