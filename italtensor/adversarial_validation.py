from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import ModelConfig, predict_probability, train_numpy_model
from .preprocessing import FeatureStandardizer


def run_adversarial_validation_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    reference_fraction: float = 0.5,
    validation_fraction: float = 0.35,
    seed: int = 42,
    max_epochs: int = 80,
) -> dict[str, Any]:
    """Train a lightweight domain classifier to detect multivariate row-order drift."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Adversarial validation features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Adversarial validation features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Adversarial validation feature and label counts do not match.")
    if x.shape[0] < 12:
        raise ValueError("Adversarial validation diagnostics need at least twelve rows.")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Adversarial validation diagnostics require binary labels 0 or 1.")
    if not 0.1 <= float(reference_fraction) <= 0.9:
        raise ValueError("Adversarial validation reference_fraction must be between 0.1 and 0.9.")
    if not 0.1 <= float(validation_fraction) <= 0.6:
        raise ValueError("Adversarial validation validation_fraction must be between 0.1 and 0.6.")

    reference_count = int(round(x.shape[0] * float(reference_fraction)))
    reference_count = min(max(reference_count, 4), x.shape[0] - 4)
    reference = x[:reference_count]
    current = x[reference_count:]
    domain_labels = np.asarray([0] * reference.shape[0] + [1] * current.shape[0], dtype=np.int32)
    domain_features = np.vstack([reference, current]).astype(np.float32)

    train_idx, validation_idx = _domain_train_validation_indices(
        reference.shape[0],
        current.shape[0],
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )
    x_train = domain_features[train_idx]
    y_train = domain_labels[train_idx]
    x_val = domain_features[validation_idx]
    y_val = domain_labels[validation_idx]

    preprocessor = FeatureStandardizer.fit(x_train)
    x_train_std = preprocessor.transform(x_train)
    x_val_std = preprocessor.transform(x_val)
    config = ModelConfig(
        learning_rate=0.05,
        max_epochs=max(5, int(max_epochs)),
        patience=12,
        batch_size=min(32, max(4, int(x_train.shape[0]))),
        random_seed=int(seed),
        feature_map="quadratic",
        backend="numpy",
    )
    model, history = train_numpy_model(
        x_train_std,
        y_train,
        config,
        validation_data=(x_val_std, y_val),
    )
    probabilities = predict_probability(model, x_val_std).reshape(-1)
    metrics = evaluate_predictions(y_val, probabilities, threshold=0.5)
    feature_rows = _permutation_importance(model, x_val_std, y_val, probabilities, metrics, seed=int(seed) + 17)
    label_shift = _label_shift(y[:reference_count], y[reference_count:])
    summary = _summary(metrics, feature_rows, label_shift, y_val, probabilities)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "reference_fraction": float(reference_fraction),
        "reference_count": int(reference.shape[0]),
        "current_count": int(current.shape[0]),
        "validation_fraction": float(validation_fraction),
        "validation_samples": int(y_val.shape[0]),
        "split_source": "row_order_domain_classifier",
        "domain_label_schema": {"reference": 0, "current": 1},
        "domain_metrics": _selected_metrics(metrics),
        "label_shift": label_shift,
        "features": feature_rows,
        "summary": summary,
        "training": {
            "epochs_run": int(len(history.get("loss", []))),
            "final_loss": float(history.get("loss", [0.0])[-1]) if history.get("loss") else 0.0,
            "final_validation_loss": float(history.get("val_loss", [0.0])[-1]) if history.get("val_loss") else 0.0,
        },
    }


def format_adversarial_validation_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_feature = summary.get("top_feature")
    top_text = "-" if top_feature is None else f"x{int(top_feature) + 1}"
    return (
        "Adversarial validation: "
        f"AUC={float(summary.get('domain_auc', 0.0)):.4f}, "
        f"acc={float(summary.get('domain_accuracy', 0.0)):.4f}, "
        f"detectability={float(summary.get('detectability', 0.0)):.4f}, "
        f"top={top_text}, "
        f"verdict={summary.get('verdict', '-')}"
    )


def _domain_train_validation_indices(
    reference_count: int,
    current_count: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    offset = 0
    for count in (reference_count, current_count):
        local = np.arange(offset, offset + count)
        rng.shuffle(local)
        validation_count = max(1, int(round(count * validation_fraction)))
        validation_count = min(validation_count, count - 1)
        validation_indices.extend(local[:validation_count].tolist())
        train_indices.extend(local[validation_count:].tolist())
        offset += count
    train = np.asarray(train_indices, dtype=np.int32)
    validation = np.asarray(validation_indices, dtype=np.int32)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _permutation_importance(
    model: Any,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    base_probabilities: np.ndarray,
    base_metrics: dict[str, float | int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base_auc = float(base_metrics.get("roc_auc", 0.0))
    base_accuracy = float(base_metrics.get("accuracy", 0.0))
    rows: list[dict[str, Any]] = []
    for feature_index in range(validation_features.shape[1]):
        permuted = validation_features.copy()
        permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
        probabilities = predict_probability(model, permuted).reshape(-1)
        metrics = evaluate_predictions(validation_labels, probabilities, threshold=0.5)
        auc_drop = float(base_auc - float(metrics.get("roc_auc", 0.0)))
        accuracy_drop = float(base_accuracy - float(metrics.get("accuracy", 0.0)))
        probability_shift = float(np.mean(np.abs(probabilities - base_probabilities)))
        rows.append(
            {
                "feature_index": int(feature_index),
                "auc_drop": auc_drop,
                "accuracy_drop": accuracy_drop,
                "mean_probability_shift": probability_shift,
                "risk_score": float(max(0.0, auc_drop) + 0.5 * max(0.0, accuracy_drop) + 0.25 * probability_shift),
                "risk_flags": _feature_flags(auc_drop, accuracy_drop, probability_shift),
            }
        )
    rows.sort(key=lambda item: (-float(item["risk_score"]), int(item["feature_index"])))
    return rows


def _selected_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "brier_score",
        "log_loss",
        "predicted_positive_rate",
    )
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _label_shift(reference_labels: np.ndarray, current_labels: np.ndarray) -> dict[str, float]:
    reference_prevalence = float(np.mean(reference_labels == 1)) if reference_labels.size else 0.0
    current_prevalence = float(np.mean(current_labels == 1)) if current_labels.size else 0.0
    return {
        "reference_prevalence": reference_prevalence,
        "current_prevalence": current_prevalence,
        "prevalence_shift": float(abs(current_prevalence - reference_prevalence)),
    }


def _summary(
    metrics: dict[str, float | int],
    feature_rows: list[dict[str, Any]],
    label_shift: dict[str, float],
    validation_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    domain_auc = float(metrics.get("roc_auc", 0.0))
    detectability = float(max(domain_auc, 1.0 - domain_auc))
    top = feature_rows[0] if feature_rows else {}
    warning = _warning(detectability, label_shift, validation_labels, probabilities)
    return {
        "domain_auc": domain_auc,
        "domain_accuracy": float(metrics.get("accuracy", 0.0)),
        "domain_balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "detectability": detectability,
        "top_feature": top.get("feature_index"),
        "top_feature_auc_drop": float(top.get("auc_drop", 0.0)),
        "important_feature_count": int(sum(bool(row["risk_flags"]) for row in feature_rows)),
        "label_prevalence_shift": float(label_shift["prevalence_shift"]),
        "verdict": _verdict(detectability),
        "warning": warning,
    }


def _verdict(detectability: float) -> str:
    if detectability >= 0.85:
        return "strong_multivariate_shift"
    if detectability >= 0.75:
        return "moderate_multivariate_shift"
    if detectability >= 0.65:
        return "weak_multivariate_shift"
    return "no_detectable_multivariate_shift"


def _feature_flags(auc_drop: float, accuracy_drop: float, probability_shift: float) -> list[str]:
    flags: list[str] = []
    if auc_drop >= 0.08:
        flags.append("domain_auc_driver")
    if accuracy_drop >= 0.08:
        flags.append("domain_accuracy_driver")
    if probability_shift >= 0.08:
        flags.append("probability_shift_driver")
    return flags


def _warning(
    detectability: float,
    label_shift: dict[str, float],
    validation_labels: np.ndarray,
    probabilities: np.ndarray,
) -> str | None:
    warnings: list[str] = []
    if validation_labels.shape[0] < 20:
        warnings.append("small domain-validation split")
    predicted = (probabilities >= 0.5).astype(np.int32)
    if np.unique(predicted).size < 2:
        warnings.append("domain classifier predicts one class")
    if detectability >= 0.75:
        warnings.append("reference/current rows are distinguishable")
    if float(label_shift["prevalence_shift"]) >= 0.20:
        warnings.append("label prevalence shifted")
    return "; ".join(warnings) if warnings else None
