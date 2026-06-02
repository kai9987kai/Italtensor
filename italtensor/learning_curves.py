"""Learning-curve diagnostics: performance vs. training set size."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .experiments import balanced_class_weights, evaluate_predictions, fixed_threshold_metrics, optimize_threshold
from .modeling import ModelConfig, predict_probability, train_model
from .preprocessing import FeatureStandardizer


def learning_curve_points(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Train at increasing training fractions and evaluate one fixed holdout."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Learning curves need a 2D feature matrix.")
    if x.shape[0] != y.shape[0] or x.shape[0] < 8:
        raise ValueError("Learning curves need at least 8 samples.")

    rng = np.random.default_rng(seed)
    train_pool, val_idx = _fixed_holdout_indices(y, rng)
    x_val, y_val = x[val_idx], y[val_idx]

    points: list[dict[str, Any]] = []
    for fraction in fractions:
        frac = min(max(float(fraction), 0.1), 1.0)
        train_idx = _stratified_fraction_indices(y, train_pool, frac, rng)
        subset_x = x[train_idx]
        subset_y = y[train_idx]
        result = train_fixed_holdout_model(subset_x, subset_y, x_val, y_val, config)
        metrics = result["metrics"]
        points.append(
            {
                "train_fraction": frac,
                "train_samples": int(subset_x.shape[0]),
                "validation_samples": int(x_val.shape[0]),
                "f1": float(metrics.get("f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "threshold": float(metrics.get("threshold", 0.5)),
                "validation_loss": float(metrics.get("validation_loss", 0.0)),
            }
        )
    return points


def train_fixed_holdout_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
) -> dict[str, Any]:
    """Train with the app backend stack and evaluate a supplied holdout."""
    train_x = np.asarray(x_train, dtype=np.float32)
    train_y = np.asarray(y_train, dtype=np.int32).reshape(-1)
    val_x = np.asarray(x_val, dtype=np.float32)
    val_y = np.asarray(y_val, dtype=np.int32).reshape(-1)
    if train_x.ndim != 2 or val_x.ndim != 2:
        raise ValueError("Learning-curve train and validation features must be 2D arrays.")
    if train_x.shape[0] != train_y.shape[0] or val_x.shape[0] != val_y.shape[0]:
        raise ValueError("Learning-curve feature and label counts do not match.")
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Learning-curve train and validation feature widths do not match.")
    if np.unique(train_y).size < 2 or np.unique(val_y).size < 2:
        raise ValueError("Learning-curve train and validation splits both need two classes.")

    if getattr(config, "feature_selection_k", None) is not None:
        preprocessor = FeatureStandardizer.fit_with_selection(train_x, train_y, k=config.feature_selection_k)
    else:
        preprocessor = FeatureStandardizer.fit(train_x)
    train_x_std = preprocessor.transform(train_x)
    val_x_std = preprocessor.transform(val_x)
    model, history = train_model(
        train_x_std,
        train_y,
        config,
        validation_data=(val_x_std, val_y),
        class_weight=balanced_class_weights(train_y),
    )
    probabilities = predict_probability(model, val_x_std)
    threshold = optimize_threshold(val_y, probabilities)
    metrics = evaluate_predictions(val_y, probabilities, threshold)
    metrics.update(fixed_threshold_metrics(val_y, probabilities))
    metrics["threshold"] = float(threshold)
    metrics["threshold_gain_f1"] = float(metrics["f1"] - metrics["fixed_threshold_f1"])
    metrics["threshold_gain_balanced_accuracy"] = float(
        metrics["balanced_accuracy"] - metrics["fixed_threshold_balanced_accuracy"]
    )
    if history.get("val_loss"):
        metrics["training_final_tuning_loss"] = float(history["val_loss"][-1])
    return {"metrics": metrics, "history": history}


def run_learning_curve_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Return a durable learning-curve report for reports and sidecars."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    points = learning_curve_points(x, y, config, fractions=fractions, seed=seed)
    summary = _summary(points)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]) if x.ndim == 2 else None,
        "fractions": [float(point["train_fraction"]) for point in points],
        "dataset_fingerprint": learning_curve_dataset_fingerprint(x, y),
        "config": config.to_dict(),
        "summary": summary,
        "points": points,
    }


def format_learning_curve_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Learning curve: "
        f"verdict={summary.get('verdict', '-')}, "
        f"best_F1={float(summary.get('best_f1', 0.0)):.4f}, "
        f"final_F1={float(summary.get('final_f1', 0.0)):.4f}, "
        f"gain={float(summary.get('f1_gain', 0.0)):.4f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def learning_curve_dataset_fingerprint(features: np.ndarray, labels: np.ndarray) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _fixed_holdout_indices(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=np.int32).reshape(-1)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_value in (0, 1):
        class_indices = np.where(labels == class_value)[0]
        if class_indices.shape[0] < 2:
            raise ValueError("Learning curves need at least two samples from each class.")
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        holdout_count = max(1, int(round(shuffled.shape[0] * 0.2)))
        holdout_count = min(holdout_count, shuffled.shape[0] - 1)
        validation_indices.extend(int(index) for index in shuffled[:holdout_count])
        train_indices.extend(int(index) for index in shuffled[holdout_count:])
    train_array = np.asarray(train_indices, dtype=np.int64)
    validation_array = np.asarray(validation_indices, dtype=np.int64)
    rng.shuffle(train_array)
    rng.shuffle(validation_array)
    return train_array, validation_array


def _stratified_fraction_indices(
    y: np.ndarray,
    train_pool: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int32).reshape(-1)
    selected: list[int] = []
    for class_value in (0, 1):
        class_pool = np.asarray([index for index in train_pool if labels[int(index)] == class_value], dtype=np.int64)
        if class_pool.shape[0] == 0:
            raise ValueError("Learning-curve training pool must include both classes.")
        count = max(1, int(round(class_pool.shape[0] * fraction)))
        count = min(count, class_pool.shape[0])
        selected.extend(int(index) for index in class_pool[:count])
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    return selected_array


def _summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {
            "verdict": "no_points",
            "best_f1": 0.0,
            "final_f1": 0.0,
            "f1_gain": 0.0,
            "recommended_next_step": "Run learning curve after adding enough labeled data.",
        }
    first = points[0]
    final = points[-1]
    best = max(points, key=lambda item: (float(item.get("f1", 0.0)), float(item.get("accuracy", 0.0))))
    first_f1 = float(first.get("f1", 0.0))
    final_f1 = float(final.get("f1", 0.0))
    best_f1 = float(best.get("f1", 0.0))
    gain = final_f1 - first_f1
    best_gap = best_f1 - final_f1
    if final_f1 < 0.60:
        verdict = "underfit_or_noisy"
        next_step = "Improve features, labels, or model search before relying on this configuration."
    elif gain >= 0.08:
        verdict = "more_data_helpful"
        next_step = "Collect more labeled rows; validation F1 is still improving with training size."
    elif best_gap >= 0.06:
        verdict = "capacity_or_regularization_review"
        next_step = "Review regularization, feature selection, and threshold stability; later fractions did not dominate."
    else:
        verdict = "stable_enough"
        next_step = "Use external holdout or robustness diagnostics before promotion."
    return {
        "verdict": verdict,
        "first_f1": first_f1,
        "final_f1": final_f1,
        "best_f1": best_f1,
        "best_fraction": float(best.get("train_fraction", 0.0)),
        "f1_gain": float(gain),
        "best_gap_vs_final": float(best_gap),
        "recommended_next_step": next_step,
    }
