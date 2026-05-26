from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions, probability_diagnostics
from .modeling import ModelConfig, predict_probability, train_numpy_model
from .preprocessing import FeatureStandardizer


def run_chronological_holdout_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    reference_fraction: float = 0.6,
    reference_validation_fraction: float = 0.25,
    current_validation_fraction: float = 0.5,
    threshold: float = 0.5,
    permutation_repeats: int = 3,
    seed: int = 42,
    max_epochs: int = 90,
    feature_map: str = "linear",
) -> dict[str, Any]:
    """Train on earlier rows and evaluate on later rows as a deployment replay."""
    x = np.asarray(features, dtype=np.float32)
    y = _coerce_binary_labels(labels)
    if x.ndim != 2:
        raise ValueError("Chronological holdout features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Chronological holdout features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Chronological holdout feature and label counts do not match.")
    if x.shape[0] < 16:
        raise ValueError("Chronological holdout diagnostics need at least sixteen rows.")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Chronological holdout diagnostics require binary labels 0 or 1.")
    if not 0.35 <= float(reference_fraction) <= 0.85:
        raise ValueError("Chronological holdout reference_fraction must be between 0.35 and 0.85.")
    if not 0.15 <= float(reference_validation_fraction) <= 0.5:
        raise ValueError("Chronological holdout reference_validation_fraction must be between 0.15 and 0.5.")
    if not 0.25 <= float(current_validation_fraction) <= 0.75:
        raise ValueError("Chronological holdout current_validation_fraction must be between 0.25 and 0.75.")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Chronological holdout threshold must be between 0 and 1.")
    if int(permutation_repeats) < 1:
        raise ValueError("Chronological holdout permutation_repeats must be at least one.")
    if feature_map not in {"linear", "quadratic", "rff"}:
        raise ValueError("Chronological holdout feature_map must be linear, quadratic, or rff.")

    reference_count = int(round(x.shape[0] * float(reference_fraction)))
    reference_count = min(max(reference_count, 8), x.shape[0] - 4)
    reference_x = x[:reference_count]
    reference_y = y[:reference_count]
    current_x = x[reference_count:]
    current_y = y[reference_count:]
    warnings: list[str] = []

    reference_train_idx, reference_eval_idx = _stratified_indices(
        reference_y,
        validation_fraction=float(reference_validation_fraction),
        seed=int(seed),
        purpose="Chronological holdout reference slice",
    )
    x_train = reference_x[reference_train_idx]
    y_train = reference_y[reference_train_idx]
    x_reference_eval = reference_x[reference_eval_idx]
    y_reference_eval = reference_y[reference_eval_idx]

    preprocessor = FeatureStandardizer.fit(x_train)
    x_train_std = preprocessor.transform(x_train)
    x_reference_eval_std = preprocessor.transform(x_reference_eval)
    x_current_std = preprocessor.transform(current_x)
    config = _diagnostic_config(
        train_count=int(x_train.shape[0]),
        seed=int(seed),
        max_epochs=int(max_epochs),
        feature_map=feature_map,
    )
    model, history = train_numpy_model(
        x_train_std,
        y_train,
        config,
        validation_data=(x_reference_eval_std, y_reference_eval),
    )
    reference_probabilities = predict_probability(model, x_reference_eval_std).reshape(-1)
    current_probabilities = predict_probability(model, x_current_std).reshape(-1)
    reference_metrics = _selected_metrics(
        evaluate_predictions(y_reference_eval, reference_probabilities, threshold=float(threshold))
    )
    current_metrics = _selected_metrics(evaluate_predictions(current_y, current_probabilities, threshold=float(threshold)))
    deltas = _metric_deltas(reference_metrics, current_metrics)
    reliance = _permutation_reliance(
        model,
        x_current_std,
        current_y,
        current_probabilities,
        current_metrics,
        threshold=float(threshold),
        seed=int(seed) + 31,
        repeats=int(permutation_repeats),
    )
    label_shift = _label_shift(reference_y, current_y)
    probability_diagnostics = _probability_diagnostics(
        reference_probabilities,
        current_probabilities,
        y_reference_eval,
        current_y,
        reference_metrics,
        current_metrics,
    )
    current_baseline = _current_baseline(
        reference_model=model,
        reference_preprocessor=preprocessor,
        current_features=current_x,
        current_labels=current_y,
        current_validation_fraction=float(current_validation_fraction),
        seed=int(seed) + 101,
        max_epochs=int(max_epochs),
        feature_map=feature_map,
        threshold=float(threshold),
        warnings=warnings,
    )
    summary = _summary(
        reference_metrics,
        current_metrics,
        deltas,
        reliance,
        label_shift,
        current_y,
        current_baseline,
        warnings,
    )
    report_warnings = summary["warning"].split("; ") if summary.get("warning") else []
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "split_source": "row_order_reference_then_current",
        "reference_fraction": float(reference_fraction),
        "reference_count": int(reference_count),
        "reference_train_count": int(y_train.shape[0]),
        "reference_evaluation_count": int(y_reference_eval.shape[0]),
        "current_count": int(current_y.shape[0]),
        "feature_map": feature_map,
        "threshold": float(threshold),
        "permutation_repeats": int(permutation_repeats),
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "metric_deltas": deltas,
        "label_shift": label_shift,
        "current_probability_diagnostics": probability_diagnostics,
        "permutation_reliance": reliance,
        "current_feature_reliance": reliance,
        "current_baseline": current_baseline,
        "summary": summary,
        "warnings": report_warnings,
        "training": {
            "epochs_run": int(len(history.get("loss", []))),
            "final_loss": float(history.get("loss", [0.0])[-1]) if history.get("loss") else 0.0,
            "final_reference_validation_loss": float(history.get("val_loss", [0.0])[-1]) if history.get("val_loss") else 0.0,
        },
    }


def format_chronological_holdout_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_feature = summary.get("top_current_reliance_feature")
    top_text = "-" if top_feature is None else f"x{int(top_feature) + 1}"
    return (
        "Chronological holdout: "
        f"ref_F1={float(summary.get('reference_f1', 0.0)):.4f}, "
        f"current_F1={float(summary.get('current_f1', 0.0)):.4f}, "
        f"delta={float(summary.get('f1_delta', 0.0)):.4f}, "
        f"top_current_feature={top_text}, "
        f"verdict={summary.get('verdict', '-')}"
    )


def _diagnostic_config(*, train_count: int, seed: int, max_epochs: int, feature_map: str) -> ModelConfig:
    return ModelConfig(
        learning_rate=0.05,
        max_epochs=max(5, int(max_epochs)),
        patience=12,
        batch_size=min(32, max(4, int(train_count))),
        random_seed=int(seed),
        feature_map=feature_map,
        backend="numpy",
    )


def _coerce_binary_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        label_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chronological holdout labels must be numeric binary values 0 or 1.") from exc
    if label_values.size == 0:
        raise ValueError("Chronological holdout diagnostics need labels.")
    if not np.all(np.isfinite(label_values)):
        raise ValueError("Chronological holdout labels must be finite binary values 0 or 1.")
    if not np.all(np.equal(label_values, np.round(label_values))):
        raise ValueError("Chronological holdout labels must be integer binary values 0 or 1.")
    as_int = label_values.astype(np.int32)
    if set(np.unique(as_int).tolist()) - {0, 1}:
        raise ValueError("Chronological holdout diagnostics require binary labels 0 or 1.")
    return as_int


def _stratified_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
    purpose: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for cls in (0, 1):
        cls_indices = np.where(labels == cls)[0]
        if cls_indices.shape[0] < 2:
            raise ValueError(f"{purpose} needs at least two rows per class.")
        rng.shuffle(cls_indices)
        eval_count = max(1, int(round(cls_indices.shape[0] * validation_fraction)))
        eval_count = min(eval_count, cls_indices.shape[0] - 1)
        eval_indices.extend(cls_indices[:eval_count].tolist())
        train_indices.extend(cls_indices[eval_count:].tolist())
    train = np.asarray(train_indices, dtype=np.int32)
    eval_idx = np.asarray(eval_indices, dtype=np.int32)
    rng.shuffle(train)
    rng.shuffle(eval_idx)
    return train, eval_idx


def _selected_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "brier_score",
        "log_loss",
        "roc_auc",
        "average_precision",
        "predicted_positive_rate",
        "label_prevalence",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    )
    selected: dict[str, float | int] = {}
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        selected[key] = int(value) if key.startswith(("true_", "false_")) else float(value)
    return selected


def _metric_deltas(
    reference_metrics: dict[str, float | int],
    current_metrics: dict[str, float | int],
) -> dict[str, float]:
    return {
        "f1_delta": float(current_metrics.get("f1", 0.0) - reference_metrics.get("f1", 0.0)),
        "accuracy_delta": float(current_metrics.get("accuracy", 0.0) - reference_metrics.get("accuracy", 0.0)),
        "balanced_accuracy_delta": float(
            current_metrics.get("balanced_accuracy", 0.0) - reference_metrics.get("balanced_accuracy", 0.0)
        ),
        "precision_delta": float(current_metrics.get("precision", 0.0) - reference_metrics.get("precision", 0.0)),
        "recall_delta": float(current_metrics.get("recall", 0.0) - reference_metrics.get("recall", 0.0)),
        "brier_score_delta": float(current_metrics.get("brier_score", 0.0) - reference_metrics.get("brier_score", 0.0)),
        "log_loss_delta": float(current_metrics.get("log_loss", 0.0) - reference_metrics.get("log_loss", 0.0)),
    }


def _probability_diagnostics(
    reference_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
    reference_labels: np.ndarray,
    current_labels: np.ndarray,
    reference_metrics: dict[str, float | int],
    current_metrics: dict[str, float | int],
) -> dict[str, Any]:
    reference_quality = _compact_probability_diagnostics(
        probability_diagnostics(reference_labels, reference_probabilities)
    )
    current_quality = _compact_probability_diagnostics(probability_diagnostics(current_labels, current_probabilities))
    return {
        "reference_mean_probability": float(np.mean(reference_probabilities)) if reference_probabilities.size else 0.0,
        "current_mean_probability": float(np.mean(current_probabilities)) if current_probabilities.size else 0.0,
        "mean_probability_delta": float(np.mean(current_probabilities) - np.mean(reference_probabilities))
        if current_probabilities.size and reference_probabilities.size
        else 0.0,
        "current_low_confidence_rate": float(np.mean(np.abs(current_probabilities - 0.5) <= 0.10))
        if current_probabilities.size
        else 0.0,
        "current_high_confidence_rate": float(np.mean((current_probabilities <= 0.10) | (current_probabilities >= 0.90)))
        if current_probabilities.size
        else 0.0,
        "predicted_positive_rate_delta": float(
            current_metrics.get("predicted_positive_rate", 0.0) - reference_metrics.get("predicted_positive_rate", 0.0)
        ),
        "label_prevalence_delta": float(
            current_metrics.get("label_prevalence", 0.0) - reference_metrics.get("label_prevalence", 0.0)
        ),
        "reference_quality": reference_quality,
        "current_quality": current_quality,
        "current_ece": float(current_quality.get("expected_calibration_error", 0.0)),
        "current_max_calibration_error": float(current_quality.get("max_calibration_error", 0.0)),
        "current_calibration_bins": current_quality.get("calibration_bins", []),
    }


def _compact_probability_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "brier_score",
        "log_loss",
        "roc_auc",
        "average_precision",
        "mean_probability",
        "predicted_positive_rate",
        "label_prevalence",
        "expected_calibration_error",
        "max_calibration_error",
        "quantiles_by_class",
        "calibration_bins",
    )
    return {key: diagnostics[key] for key in keys if key in diagnostics}


def _permutation_reliance(
    model: Any,
    current_features: np.ndarray,
    current_labels: np.ndarray,
    base_probabilities: np.ndarray,
    base_metrics: dict[str, float | int],
    *,
    threshold: float,
    seed: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base_f1 = float(base_metrics.get("f1", 0.0))
    base_log_loss = float(base_metrics.get("log_loss", 0.0))
    rows: list[dict[str, Any]] = []
    for feature_index in range(current_features.shape[1]):
        f1_drops: list[float] = []
        log_loss_increases: list[float] = []
        probability_shifts: list[float] = []
        for _ in range(int(repeats)):
            permuted = current_features.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            probabilities = predict_probability(model, permuted).reshape(-1)
            metrics = evaluate_predictions(current_labels, probabilities, threshold=float(threshold))
            f1_drops.append(float(base_f1 - float(metrics.get("f1", 0.0))))
            log_loss_increases.append(float(float(metrics.get("log_loss", 0.0)) - base_log_loss))
            probability_shifts.append(float(np.mean(np.abs(probabilities - base_probabilities))))
        f1_drop = float(np.mean(f1_drops))
        log_loss_increase = float(np.mean(log_loss_increases))
        probability_shift = float(np.mean(probability_shifts))
        rows.append(
            {
                "feature_index": int(feature_index),
                "permutation_repeats": int(repeats),
                "f1_drop": f1_drop,
                "log_loss_increase": log_loss_increase,
                "mean_probability_shift": probability_shift,
                "risk_score": float(max(0.0, f1_drop) + 0.25 * max(0.0, log_loss_increase) + 0.25 * probability_shift),
                "risk_flags": _feature_flags(f1_drop, log_loss_increase, probability_shift),
            }
        )
    rows.sort(key=lambda item: (-float(item["risk_score"]), int(item["feature_index"])))
    return rows


def _current_baseline(
    *,
    reference_model: Any,
    reference_preprocessor: FeatureStandardizer,
    current_features: np.ndarray,
    current_labels: np.ndarray,
    current_validation_fraction: float,
    seed: int,
    max_epochs: int,
    feature_map: str,
    threshold: float,
    warnings: list[str],
) -> dict[str, Any]:
    if current_labels.shape[0] < 8:
        reason = "current slice has fewer than eight rows"
        warnings.append(reason)
        return {"available": False, "reason": reason}
    try:
        current_train_idx, current_eval_idx = _stratified_indices(
            current_labels,
            validation_fraction=current_validation_fraction,
            seed=seed,
            purpose="Chronological holdout current-baseline slice",
        )
    except ValueError as exc:
        reason = str(exc)
        warnings.append(reason)
        return {"available": False, "reason": reason}

    x_train = current_features[current_train_idx]
    y_train = current_labels[current_train_idx]
    x_eval = current_features[current_eval_idx]
    y_eval = current_labels[current_eval_idx]
    preprocessor = FeatureStandardizer.fit(x_train)
    x_train_std = preprocessor.transform(x_train)
    x_eval_std = preprocessor.transform(x_eval)
    config = _diagnostic_config(
        train_count=int(x_train.shape[0]),
        seed=seed,
        max_epochs=max_epochs,
        feature_map=feature_map,
    )
    model, history = train_numpy_model(
        x_train_std,
        y_train,
        config,
        validation_data=(x_eval_std, y_eval),
    )
    current_model_probabilities = predict_probability(model, x_eval_std).reshape(-1)
    current_model_metrics = _selected_metrics(
        evaluate_predictions(y_eval, current_model_probabilities, threshold=float(threshold))
    )

    reference_model_probabilities = predict_probability(reference_model, reference_preprocessor.transform(x_eval)).reshape(-1)
    reference_model_metrics = _selected_metrics(
        evaluate_predictions(y_eval, reference_model_probabilities, threshold=float(threshold))
    )
    return {
        "available": True,
        "current_train_count": int(y_train.shape[0]),
        "current_evaluation_count": int(y_eval.shape[0]),
        "reference_model_metrics_on_current_eval": reference_model_metrics,
        "current_model_metrics": current_model_metrics,
        "metric_deltas_vs_reference_model": _metric_deltas(reference_model_metrics, current_model_metrics),
        "training": {
            "epochs_run": int(len(history.get("loss", []))),
            "final_loss": float(history.get("loss", [0.0])[-1]) if history.get("loss") else 0.0,
            "final_validation_loss": float(history.get("val_loss", [0.0])[-1]) if history.get("val_loss") else 0.0,
        },
    }


def _label_shift(reference_labels: np.ndarray, current_labels: np.ndarray) -> dict[str, Any]:
    reference_prevalence = float(np.mean(reference_labels == 1)) if reference_labels.size else 0.0
    current_prevalence = float(np.mean(current_labels == 1)) if current_labels.size else 0.0
    return {
        "reference_prevalence": reference_prevalence,
        "current_prevalence": current_prevalence,
        "prevalence_shift": float(abs(current_prevalence - reference_prevalence)),
        "reference_counts": {
            "0": int(np.sum(reference_labels == 0)),
            "1": int(np.sum(reference_labels == 1)),
        },
        "current_counts": {
            "0": int(np.sum(current_labels == 0)),
            "1": int(np.sum(current_labels == 1)),
        },
    }


def _summary(
    reference_metrics: dict[str, float | int],
    current_metrics: dict[str, float | int],
    deltas: dict[str, float],
    reliance: list[dict[str, Any]],
    label_shift: dict[str, Any],
    current_labels: np.ndarray,
    current_baseline: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    top = reliance[0] if reliance else {}
    baseline_gap = 0.0
    if current_baseline.get("available"):
        baseline_delta = current_baseline.get("metric_deltas_vs_reference_model", {})
        baseline_gap = float(baseline_delta.get("f1_delta", 0.0))
    warning = _warning(deltas, label_shift, current_labels, current_baseline, warnings)
    return {
        "reference_f1": float(reference_metrics.get("f1", 0.0)),
        "current_f1": float(current_metrics.get("f1", 0.0)),
        "f1_delta": float(deltas["f1_delta"]),
        "accuracy_delta": float(deltas["accuracy_delta"]),
        "balanced_accuracy_delta": float(deltas["balanced_accuracy_delta"]),
        "brier_score_delta": float(deltas["brier_score_delta"]),
        "log_loss_delta": float(deltas["log_loss_delta"]),
        "label_prevalence_shift": float(label_shift["prevalence_shift"]),
        "top_current_reliance_feature": top.get("feature_index"),
        "top_current_reliance_f1_drop": float(top.get("f1_drop", 0.0)),
        "current_baseline_f1_gain": baseline_gap,
        "verdict": _verdict(deltas, baseline_gap),
        "warning": warning,
    }


def _verdict(deltas: dict[str, float], current_baseline_f1_gain: float) -> str:
    if deltas["f1_delta"] <= -0.25 or deltas["balanced_accuracy_delta"] <= -0.20:
        if current_baseline_f1_gain >= 0.10:
            return "severe_temporal_degradation_current_relearns"
        return "severe_temporal_degradation"
    if deltas["f1_delta"] <= -0.10 or deltas["balanced_accuracy_delta"] <= -0.10:
        if current_baseline_f1_gain >= 0.10:
            return "moderate_temporal_degradation_current_relearns"
        return "moderate_temporal_degradation"
    if deltas["brier_score_delta"] >= 0.08 or deltas["log_loss_delta"] >= 0.20:
        return "probability_degradation"
    return "stable_holdout"


def _feature_flags(f1_drop: float, log_loss_increase: float, probability_shift: float) -> list[str]:
    flags: list[str] = []
    if f1_drop >= 0.08:
        flags.append("current_f1_driver")
    if log_loss_increase >= 0.10:
        flags.append("current_log_loss_driver")
    if probability_shift >= 0.08:
        flags.append("current_probability_driver")
    return flags


def _warning(
    deltas: dict[str, float],
    label_shift: dict[str, Any],
    current_labels: np.ndarray,
    current_baseline: dict[str, Any],
    baseline_warnings: list[str],
) -> str | None:
    warnings = list(baseline_warnings)
    if current_labels.shape[0] < 20:
        warnings.append("small current holdout")
    if np.unique(current_labels).size < 2:
        warnings.append("current holdout has one class")
    if deltas["f1_delta"] <= -0.10:
        warnings.append("current F1 is lower than reference validation")
    if float(label_shift["prevalence_shift"]) >= 0.20:
        warnings.append("label prevalence shifted")
    if not current_baseline.get("available"):
        warnings.append("current-only baseline unavailable")
    return "; ".join(dict.fromkeys(warnings)) if warnings else None
