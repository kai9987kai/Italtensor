"""Repeated stratified validation diagnostics for split-sensitive model evidence."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .experiments import (
    balanced_class_weights,
    evaluate_predictions,
    optimize_threshold,
    stratified_kfold_indices,
)
from .modeling import ModelConfig, predict_probability, train_numpy_model
from .preprocessing import FeatureStandardizer


METRIC_KEYS = (
    "f1",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "brier_score",
    "log_loss",
    "ece",
    "roc_auc",
    "average_precision",
)


def run_validation_stability_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 4,
    repeats: int = 3,
    seed: int = 42,
    max_epochs: int = 35,
    feature_map: str = "linear",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure how much validation evidence moves across repeated stratified folds."""
    x, y = _validate_inputs(features, labels)
    n_splits = _validate_splits(n_splits, y)
    repeats = _bounded_int(repeats, "repeats", minimum=1, maximum=20)
    max_epochs = _bounded_int(max_epochs, "max_epochs", minimum=1, maximum=200)
    feature_map = _validate_feature_map(feature_map)
    threshold = _validate_threshold(threshold)

    folds: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        repeat_seed = int(seed) + repeat_index * 1009
        splits = stratified_kfold_indices(y, n_splits=n_splits, seed=repeat_seed)
        oof_probabilities = np.full(y.shape[0], np.nan, dtype=np.float64)
        repeat_fold_rows: list[dict[str, Any]] = []

        for fold_index, (outer_train_indices, validation_indices) in enumerate(splits):
            model_seed = repeat_seed + (fold_index + 1) * 97
            fit_local, calibration_local = _inner_train_calibration_indices(
                y[outer_train_indices],
                seed=model_seed,
            )
            fit_indices = outer_train_indices[fit_local]
            calibration_indices = outer_train_indices[calibration_local]
            probabilities, calibration_probabilities, history = _train_fold(
                x[fit_indices],
                y[fit_indices],
                x[calibration_indices],
                y[calibration_indices],
                x[validation_indices],
                y[validation_indices],
                feature_map=feature_map,
                max_epochs=max_epochs,
                seed=model_seed,
            )
            oof_probabilities[validation_indices] = probabilities
            metrics = _metric_snapshot(evaluate_predictions(y[validation_indices], probabilities, threshold))
            tuned_threshold = float(optimize_threshold(y[calibration_indices], calibration_probabilities))
            tuned_metrics = _metric_snapshot(
                evaluate_predictions(y[validation_indices], probabilities, tuned_threshold)
            )
            row = {
                "repeat": int(repeat_index + 1),
                "fold": int(fold_index + 1),
                "split_seed": int(repeat_seed),
                "model_seed": int(model_seed),
                "train_sample_count": int(fit_indices.shape[0]),
                "calibration_sample_count": int(calibration_indices.shape[0]),
                "validation_sample_count": int(validation_indices.shape[0]),
                "train_class_counts": _class_counts(y[fit_indices]),
                "calibration_class_counts": _class_counts(y[calibration_indices]),
                "validation_class_counts": _class_counts(y[validation_indices]),
                "validation_row_indices": [int(value) for value in np.sort(validation_indices)],
                "metrics": metrics,
                "tuned_threshold": tuned_threshold,
                "tuned_metrics": tuned_metrics,
                "epochs_run": int(len(history.get("loss", []))),
                "final_train_loss": _last_or_none(history.get("loss")),
                "final_validation_loss": _last_or_none(history.get("val_loss")),
            }
            folds.append(row)
            repeat_fold_rows.append(row)

        if not np.all(np.isfinite(oof_probabilities)):
            raise RuntimeError("Validation stability did not produce one out-of-fold probability per row.")
        pooled_metrics = _metric_snapshot(evaluate_predictions(y, oof_probabilities, threshold))
        repeat_rows.append(
            {
                "repeat": int(repeat_index + 1),
                "split_seed": int(repeat_seed),
                "fold_count": int(len(repeat_fold_rows)),
                "fixed_threshold": float(threshold),
                "pooled_metrics": pooled_metrics,
                "fold_f1_mean": float(np.mean([row["metrics"]["f1"] for row in repeat_fold_rows])),
                "fold_f1_std": float(np.std([row["metrics"]["f1"] for row in repeat_fold_rows])),
                "fold_tuned_f1_mean": float(
                    np.mean([row["tuned_metrics"]["f1"] for row in repeat_fold_rows])
                ),
            }
        )

    aggregate = {key: _distribution([float(row["metrics"][key]) for row in folds]) for key in METRIC_KEYS}
    tuned_aggregate = {
        key: _distribution([float(row["tuned_metrics"][key]) for row in folds])
        for key in METRIC_KEYS
    }
    repeat_aggregate = {
        key: _distribution([float(row["pooled_metrics"][key]) for row in repeat_rows])
        for key in METRIC_KEYS
    }
    threshold_distribution = _distribution([float(row["tuned_threshold"]) for row in folds])
    summary = _summary(folds, aggregate, repeat_aggregate, threshold_distribution)
    recommendations = _recommendations(summary)
    summary["recommended_next_step"] = recommendations[0]["action"] if recommendations else None

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dataset_fingerprint": validation_stability_dataset_fingerprint(x, y),
        "class_counts": _class_counts(y),
        "n_splits": int(n_splits),
        "repeats": int(repeats),
        "total_fold_count": int(len(folds)),
        "seed": int(seed),
        "max_epochs": int(max_epochs),
        "feature_map": feature_map,
        "threshold": float(threshold),
        "aggregate": aggregate,
        "tuned_aggregate": tuned_aggregate,
        "repeat_aggregate": repeat_aggregate,
        "calibration_threshold_distribution": threshold_distribution,
        "summary": summary,
        "recommendations": recommendations,
        "repeats_detail": repeat_rows,
        "folds": folds,
        "interpretation_note": (
            "Fold quantiles are empirical split-sensitivity summaries, not formal confidence intervals."
        ),
    }


def run_validation_stability(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for repeated validation stability diagnostics."""
    return run_validation_stability_diagnostics(*args, **kwargs)


def format_validation_stability_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Validation stability: "
        f"verdict={summary.get('verdict', '-')}, "
        f"priority={summary.get('priority', '-')}, "
        f"folds={int(report.get('total_fold_count', 0) or 0)}, "
        f"mean_f1={float(summary.get('mean_fold_f1', 0.0) or 0.0):.4f}, "
        f"f1_std={float(summary.get('fold_f1_std', 0.0) or 0.0):.4f}, "
        f"q10={float(summary.get('fold_f1_q10', 0.0) or 0.0):.4f}, "
        f"worst={float(summary.get('worst_fold_f1', 0.0) or 0.0):.4f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def validation_stability_dataset_fingerprint(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> str:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(value) for value in x.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(x).tobytes())
    hasher.update(str(tuple(int(value) for value in y.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return hasher.hexdigest()[:16]


def _train_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    feature_map: str,
    max_epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    standardizer = FeatureStandardizer.fit(x_train)
    train_prepared = standardizer.transform(x_train)
    calibration_prepared = standardizer.transform(x_calibration)
    evaluation_prepared = standardizer.transform(x_evaluation)
    config = ModelConfig(
        learning_rate=0.05,
        batch_size=16,
        max_epochs=max_epochs,
        patience=max(3, min(8, max_epochs // 3 if max_epochs >= 3 else 1)),
        random_seed=int(seed),
        feature_map=feature_map,
        backend="numpy",
    )
    model, history = train_numpy_model(
        train_prepared,
        y_train,
        config,
        validation_data=(calibration_prepared, y_calibration),
        class_weight=balanced_class_weights(y_train),
    )
    calibration_probabilities = predict_probability(model, calibration_prepared).reshape(-1).astype(np.float64)
    probabilities = predict_probability(model, evaluation_prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != y_evaluation.shape[0]:
        raise ValueError("Validation stability model probability count does not match validation rows.")
    if calibration_probabilities.shape[0] != y_calibration.shape[0]:
        raise ValueError("Validation stability model probability count does not match calibration rows.")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(calibration_probabilities)):
        raise ValueError("Validation stability model returned non-finite probabilities.")
    return (
        np.clip(probabilities, 0.0, 1.0),
        np.clip(calibration_probabilities, 0.0, 1.0),
        history,
    )


def _validate_inputs(features: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Validation stability features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Validation stability features must be a 2D array.")
    if x.shape[0] < 12:
        raise ValueError("Validation stability needs at least 12 labeled rows.")
    if x.shape[1] < 1:
        raise ValueError("Validation stability needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Validation stability features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Validation stability labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Validation stability feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Validation stability labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Validation stability requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Validation stability requires binary labels 0 or 1.")
    if np.unique(y).size < 2:
        raise ValueError("Validation stability needs both labels 0 and 1.")
    return x, y


def _validate_splits(value: int, labels: np.ndarray) -> int:
    n_splits = _bounded_int(value, "n_splits", minimum=2, maximum=10)
    minority_count = int(min(np.sum(labels == 0), np.sum(labels == 1)))
    if minority_count < n_splits:
        raise ValueError(
            f"Validation stability needs at least {n_splits} rows in each class; minority count is {minority_count}."
        )
    largest_outer_fold = int(np.ceil(minority_count / n_splits))
    if minority_count - largest_outer_fold < 2:
        raise ValueError(
            "Validation stability needs enough minority rows to keep both training and calibration examples inside every outer fold."
        )
    return n_splits


def _inner_train_calibration_indices(labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    for class_value in (0, 1):
        indices = rng.permutation(np.where(labels == class_value)[0]).astype(np.int32)
        if indices.shape[0] < 2:
            raise ValueError("Validation stability inner split needs at least two rows from each class.")
        calibration_count = max(1, min(indices.shape[0] - 1, int(round(indices.shape[0] * 0.2))))
        calibration_parts.append(indices[:calibration_count])
        train_parts.append(indices[calibration_count:])
    train_indices = np.concatenate(train_parts).astype(np.int32)
    calibration_indices = np.concatenate(calibration_parts).astype(np.int32)
    rng.shuffle(train_indices)
    rng.shuffle(calibration_indices)
    return train_indices, calibration_indices


def _bounded_int(value: int, name: str, *, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"Validation stability {name} must be between {minimum} and {maximum}.")
    return parsed


def _validate_feature_map(value: str) -> str:
    parsed = str(value).strip().lower()
    if parsed not in {"linear", "quadratic", "rff"}:
        raise ValueError("Validation stability feature_map must be one of: linear, quadratic, rff.")
    return parsed


def _validate_threshold(value: float) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("Validation stability threshold must be between 0 and 1.")
    return parsed


def _metric_snapshot(metrics: dict[str, float | int]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in METRIC_KEYS}


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "q10": 0.0, "median": 0.0, "q90": 0.0, "max": 0.0, "range": 0.0}
    q10, median, q90 = np.quantile(array, [0.10, 0.50, 0.90])
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": minimum,
        "q10": float(q10),
        "median": float(median),
        "q90": float(q90),
        "max": maximum,
        "range": float(maximum - minimum),
    }


def _summary(
    folds: list[dict[str, Any]],
    aggregate: dict[str, dict[str, float]],
    repeat_aggregate: dict[str, dict[str, float]],
    threshold_distribution: dict[str, float],
) -> dict[str, Any]:
    f1 = aggregate["f1"]
    balanced = aggregate["balanced_accuracy"]
    worst_fold = min(folds, key=lambda row: float(row["metrics"]["f1"]))
    weak_cutoff = max(0.0, float(f1["mean"]) - 0.10)
    weak_fold_count = sum(float(row["metrics"]["f1"]) < weak_cutoff for row in folds)
    repeat_f1_std = float(repeat_aggregate["f1"]["std"])

    if f1["std"] >= 0.12 or f1["range"] >= 0.35 or repeat_f1_std >= 0.10:
        verdict = "validation_unstable"
        priority = "high"
    elif f1["std"] >= 0.06 or f1["range"] >= 0.18 or balanced["std"] >= 0.07 or weak_fold_count > 0:
        verdict = "validation_stability_review"
        priority = "medium"
    else:
        verdict = "validation_stable"
        priority = "low"

    penalty = min(
        100.0,
        260.0 * float(f1["std"])
        + 90.0 * float(f1["range"])
        + 120.0 * float(balanced["std"])
        + 80.0 * repeat_f1_std,
    )
    return {
        "verdict": verdict,
        "priority": priority,
        "stability_score": round(max(0.0, 100.0 - penalty), 1),
        "mean_fold_f1": float(f1["mean"]),
        "fold_f1_std": float(f1["std"]),
        "fold_f1_q10": float(f1["q10"]),
        "fold_f1_q90": float(f1["q90"]),
        "fold_f1_range": float(f1["range"]),
        "repeat_oof_f1_std": repeat_f1_std,
        "balanced_accuracy_std": float(balanced["std"]),
        "weak_fold_count": int(weak_fold_count),
        "worst_fold_repeat": int(worst_fold["repeat"]),
        "worst_fold_index": int(worst_fold["fold"]),
        "worst_fold_f1": float(worst_fold["metrics"]["f1"]),
        "worst_fold_validation_rows": list(worst_fold["validation_row_indices"]),
        "calibration_threshold_std": float(threshold_distribution["std"]),
    }


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = str(summary.get("verdict", "validation_stable"))
    rows = summary.get("worst_fold_validation_rows", [])
    row_text = ", ".join(str(int(value)) for value in rows[:12]) or "none"
    if verdict == "validation_unstable":
        return [
            {
                "rank": 1,
                "priority": "high",
                "category": "validation",
                "title": "Do not trust one favorable split",
                "reason": (
                    f"Fold F1 std={float(summary['fold_f1_std']):.3f} and "
                    f"range={float(summary['fold_f1_range']):.3f}."
                ),
                "action": "Review the worst-fold rows, collect representative labels, and require external or grouped validation before promotion.",
            },
            {
                "rank": 2,
                "priority": "medium",
                "category": "error_analysis",
                "title": "Inspect worst-fold coverage",
                "reason": f"Worst-fold validation row indices: {row_text}.",
                "action": "Compare these rows with Error atlas, Data value scout, and subgroup diagnostics.",
            },
        ]
    if verdict == "validation_stability_review":
        return [
            {
                "rank": 1,
                "priority": "medium",
                "category": "validation",
                "title": "Document split sensitivity",
                "reason": (
                    f"Empirical fold F1 q10={float(summary['fold_f1_q10']):.3f}, "
                    f"std={float(summary['fold_f1_std']):.3f}."
                ),
                "action": "Use the lower empirical fold range for planning and confirm performance on a fresh external holdout.",
            },
            {
                "rank": 2,
                "priority": "medium",
                "category": "error_analysis",
                "title": "Review the weakest fold",
                "reason": f"Worst-fold validation row indices: {row_text}.",
                "action": "Inspect whether rare pockets, conflicts, or subgroup coverage explain the weak split.",
            },
        ]
    return [
        {
            "rank": 1,
            "priority": "low",
            "category": "evidence",
            "title": "Retain validation stability evidence",
            "reason": (
                f"Fold F1 std={float(summary['fold_f1_std']):.3f} and "
                f"q10={float(summary['fold_f1_q10']):.3f}."
            ),
            "action": "Keep this empirical spread with the model report and still use a final external holdout.",
        }
    ]


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))}


def _last_or_none(values: list[float] | None) -> float | None:
    if not values:
        return None
    return float(values[-1])
