"""Train temporary models under synthetic label noise and measure degradation."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .data import Dataset
from .experiments import balanced_class_weights, evaluate_predictions, split_train_validation
from .modeling import ModelConfig, predict_probability, train_numpy_model
from .preprocessing import FeatureStandardizer


DEFAULT_NOISE_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)


def run_label_noise_stress_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    noise_rates: Sequence[float] = DEFAULT_NOISE_RATES,
    repeats: int = 3,
    train_ratio: float = 0.70,
    seed: int = 42,
    max_epochs: int = 40,
    feature_map: str = "linear",
    threshold: float = 0.5,
    material_f1_drop: float = 0.05,
) -> dict[str, Any]:
    """Stress validation evidence by retraining temporary NumPy models on flipped labels."""
    dataset = _validate_inputs(features, labels)
    rates = _validate_noise_rates(noise_rates)
    repeats = _positive_int(repeats, "repeats")
    max_epochs = _positive_int(max_epochs, "max_epochs")
    train_ratio = _train_ratio(train_ratio)
    threshold = _threshold(threshold)
    material_f1_drop = max(0.0, float(material_f1_drop))
    feature_map = _feature_map(feature_map)

    x_train, y_train, x_val, y_val = split_train_validation(dataset, train_ratio=train_ratio, seed=int(seed))
    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        raise ValueError("Label noise stress needs both classes in train and validation splits.")

    standardizer = FeatureStandardizer.fit(x_train)
    x_train_prepared = standardizer.transform(x_train)
    x_val_prepared = standardizer.transform(x_val)
    rng = np.random.default_rng(int(seed))
    run_rows: list[dict[str, Any]] = []

    for noise_rate in rates:
        rate_repeats = 1 if noise_rate == 0.0 else repeats
        for repeat in range(rate_repeats):
            run_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            noisy_labels, flipped_indices = _flip_labels(y_train, noise_rate, run_seed)
            config = ModelConfig(
                learning_rate=0.05,
                batch_size=16,
                max_epochs=max_epochs,
                patience=max(3, min(8, max_epochs // 3 if max_epochs >= 3 else 1)),
                random_seed=run_seed,
                feature_map=feature_map,
                backend="numpy",
            )
            model, history = train_numpy_model(
                x_train_prepared,
                noisy_labels,
                config,
                validation_data=(x_val_prepared, y_val),
                class_weight=balanced_class_weights(noisy_labels),
            )
            probabilities = predict_probability(model, x_val_prepared)
            metrics = _metric_snapshot(evaluate_predictions(y_val, probabilities, threshold=threshold))
            run_rows.append(
                {
                    "noise_rate": float(noise_rate),
                    "repeat": int(repeat + 1),
                    "seed": int(run_seed),
                    "flipped_train_label_count": int(flipped_indices.shape[0]),
                    "train_sample_count": int(y_train.shape[0]),
                    "validation_sample_count": int(y_val.shape[0]),
                    "metrics": metrics,
                    "final_train_loss": _last_or_none(history.get("loss")),
                    "final_validation_loss": _last_or_none(history.get("val_loss")),
                }
            )

    baseline_runs = [row for row in run_rows if row["noise_rate"] == 0.0]
    baseline = _aggregate_metric_runs(baseline_runs)
    rates_summary = []
    for noise_rate in rates:
        rate_runs = [row for row in run_rows if row["noise_rate"] == noise_rate]
        aggregate = _aggregate_metric_runs(rate_runs)
        degradation = _degradation(baseline, aggregate)
        rates_summary.append(
            {
                "noise_rate": float(noise_rate),
                "repeat_count": int(len(rate_runs)),
                "mean_metrics": aggregate["mean_metrics"],
                "std_metrics": aggregate["std_metrics"],
                "degradation": degradation,
            }
        )

    summary = _summary(rates_summary, material_f1_drop=material_f1_drop)
    return {
        "sample_count": int(dataset.sample_count),
        "input_dim": int(dataset.input_dim),
        "dataset_fingerprint": label_noise_stress_dataset_fingerprint(dataset.features, dataset.labels),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "feature_map": feature_map,
        "threshold": float(threshold),
        "max_epochs": int(max_epochs),
        "material_f1_drop": float(material_f1_drop),
        "split": {
            "train_sample_count": int(y_train.shape[0]),
            "validation_sample_count": int(y_val.shape[0]),
            "train_class_counts": {"0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))},
            "validation_class_counts": {"0": int(np.sum(y_val == 0)), "1": int(np.sum(y_val == 1))},
        },
        "baseline": baseline,
        "rates": rates_summary,
        "summary": summary,
        "runs": run_rows,
    }


def format_label_noise_stress_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    first_rate = summary.get("first_material_noise_rate")
    first_text = "-" if first_rate is None else f"{float(first_rate):.2f}"
    return (
        "Label noise stress: "
        f"verdict={summary.get('verdict', '-')}, "
        f"priority={summary.get('priority', '-')}, "
        f"baseline_f1={float(summary.get('baseline_f1', 0.0) or 0.0):.4f}, "
        f"worst_f1_drop={float(summary.get('worst_mean_f1_drop', 0.0) or 0.0):.4f}, "
        f"first_material_rate={first_text}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def label_noise_stress_dataset_fingerprint(features: Any, labels: Any) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def run_label_noise_stress(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the label-noise stress diagnostic."""
    return run_label_noise_stress_diagnostics(*args, **kwargs)


def label_noise_dataset_fingerprint(features: Any, labels: Any) -> str:
    """Compatibility alias for the label-noise stress dataset fingerprint."""
    return label_noise_stress_dataset_fingerprint(features, labels)


def _validate_inputs(features: Any, labels: Any) -> Dataset:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Label noise stress features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Label noise stress features must be a 2D array.")
    if x.shape[0] < 12:
        raise ValueError("Label noise stress needs at least 12 labeled rows.")
    if x.shape[1] < 1:
        raise ValueError("Label noise stress needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Label noise stress features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Label noise stress labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Label noise stress feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Label noise stress labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Label noise stress requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Label noise stress requires binary labels 0 or 1.")
    if np.unique(y).size < 2:
        raise ValueError("Label noise stress needs both labels 0 and 1.")
    return Dataset(features=x, labels=y, input_dim=int(x.shape[1]))


def _flip_labels(labels: np.ndarray, noise_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
    flip_count = int(round(float(noise_rate) * y.shape[0]))
    if flip_count <= 0:
        return y, np.asarray([], dtype=np.int32)
    flip_count = min(flip_count, y.shape[0])
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(np.arange(y.shape[0]), size=flip_count, replace=False)).astype(np.int32)
    y[indices] = 1 - y[indices]
    return y, indices


def _metric_snapshot(metrics: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "brier_score",
        "log_loss",
        "ece",
        "validation_loss",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    )
    snapshot: dict[str, float | int] = {}
    for key in keys:
        value = metrics.get(key, 0.0)
        if isinstance(value, (int, np.integer)):
            snapshot[key] = int(value)
        else:
            snapshot[key] = float(value)
    return snapshot


def _aggregate_metric_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"mean_metrics": {}, "std_metrics": {}}
    metric_keys = [key for key in runs[0]["metrics"] if isinstance(runs[0]["metrics"][key], (int, float))]
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for key in metric_keys:
        values = np.asarray([float(row["metrics"][key]) for row in runs], dtype=np.float64)
        mean_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values))
    return {"mean_metrics": mean_metrics, "std_metrics": std_metrics}


def _degradation(baseline: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, float]:
    base = baseline.get("mean_metrics", {})
    current = aggregate.get("mean_metrics", {})
    f1_drop = float(base.get("f1", 0.0) - current.get("f1", 0.0))
    accuracy_drop = float(base.get("accuracy", 0.0) - current.get("accuracy", 0.0))
    balanced_drop = float(base.get("balanced_accuracy", 0.0) - current.get("balanced_accuracy", 0.0))
    brier_increase = float(current.get("brier_score", 0.0) - base.get("brier_score", 0.0))
    log_loss_increase = float(current.get("log_loss", 0.0) - base.get("log_loss", 0.0))
    ece_increase = float(current.get("ece", 0.0) - base.get("ece", 0.0))
    degradation_score = max(0.0, f1_drop) + 0.50 * max(0.0, balanced_drop) + 0.25 * max(0.0, brier_increase)
    return {
        "f1_drop": f1_drop,
        "accuracy_drop": accuracy_drop,
        "balanced_accuracy_drop": balanced_drop,
        "brier_increase": brier_increase,
        "log_loss_increase": log_loss_increase,
        "ece_increase": ece_increase,
        "degradation_score": float(degradation_score),
    }


def _summary(rates: list[dict[str, Any]], *, material_f1_drop: float) -> dict[str, Any]:
    baseline_metrics = rates[0].get("mean_metrics", {}) if rates else {}
    nonzero_rates = [row for row in rates if float(row.get("noise_rate", 0.0)) > 0.0]
    worst = max(nonzero_rates, key=lambda row: float(row["degradation"].get("f1_drop", 0.0)), default={})
    worst_drop = float((worst.get("degradation") or {}).get("f1_drop", 0.0) or 0.0)
    first_material = None
    for row in nonzero_rates:
        if float(row["degradation"].get("f1_drop", 0.0)) >= material_f1_drop:
            first_material = float(row["noise_rate"])
            break
    if worst_drop >= material_f1_drop * 2.0:
        verdict = "label_noise_fragile"
        priority = "high"
        next_step = "Review label quality and rerun label sensitivity/sample review before more model search."
    elif first_material is not None:
        verdict = "label_noise_review"
        priority = "medium"
        next_step = "Treat validation evidence as label-noise sensitive; review labels and collect clean holdout rows."
    else:
        verdict = "label_noise_stable"
        priority = "low"
        next_step = "Proceed with normal validation, while keeping human label review for high-impact rows."
    return {
        "verdict": verdict,
        "priority": priority,
        "baseline_f1": float(baseline_metrics.get("f1", 0.0) or 0.0),
        "baseline_accuracy": float(baseline_metrics.get("accuracy", 0.0) or 0.0),
        "worst_noise_rate": worst.get("noise_rate"),
        "worst_mean_f1_drop": float(worst_drop),
        "first_material_noise_rate": first_material,
        "material_f1_drop": float(material_f1_drop),
        "recommended_next_step": next_step,
    }


def _validate_noise_rates(values: Sequence[float]) -> list[float]:
    value_list = list(values)
    if not value_list:
        raise ValueError("Label noise stress needs at least one noise rate.")
    rates = {0.0}
    for value in value_list:
        parsed = float(value)
        if not 0.0 <= parsed <= 0.5:
            raise ValueError("Label noise stress noise rates must be between 0.0 and 0.5.")
        rates.add(round(parsed, 4))
    return sorted(rates)


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Label noise stress {name} must be at least 1.")
    return parsed


def _train_ratio(value: float) -> float:
    parsed = float(value)
    if not 0.5 <= parsed <= 0.85:
        raise ValueError("Label noise stress train_ratio must be between 0.5 and 0.85.")
    return parsed


def _threshold(value: float) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("Label noise stress threshold must be between 0 and 1.")
    return parsed


def _feature_map(value: str) -> str:
    parsed = str(value).strip().lower()
    if parsed not in {"linear", "quadratic", "rff"}:
        raise ValueError("Label noise stress feature_map must be one of: linear, quadratic, rff.")
    return parsed


def _last_or_none(values: list[float] | None) -> float | None:
    if not values:
        return None
    return float(values[-1])
