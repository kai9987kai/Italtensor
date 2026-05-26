from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import ModelConfig, predict_probability, train_numpy_model
from .preprocessing import FeatureStandardizer


EPSILON = 1e-12


def run_bootstrap_stability_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    model_count: int = 12,
    train_fraction: float = 0.8,
    seed: int = 42,
    max_epochs: int = 45,
    feature_map: str = "linear",
    threshold: float = 0.5,
    max_rows: int = 12,
) -> dict[str, Any]:
    """Train stratified bootstrap replicas and rank rows by model instability."""
    x, y = _validate_inputs(features, labels)
    model_count = _validate_minimum_int(model_count, "model_count", 2)
    max_epochs = _validate_minimum_int(max_epochs, "max_epochs", 1)
    max_rows = _validate_minimum_int(max_rows, "max_rows", 1)
    threshold = _validate_probability_threshold(threshold)
    train_fraction = _validate_train_fraction(train_fraction)
    feature_map = _validate_feature_map(feature_map)

    rng = np.random.default_rng(int(seed))
    probabilities = np.empty((model_count, x.shape[0]), dtype=np.float32)
    training_runs: list[dict[str, Any]] = []

    for model_index in range(model_count):
        train_indices, heldout_indices = _stratified_subsample(y, train_fraction, rng)
        model_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        standardizer = FeatureStandardizer.fit(x[train_indices])
        train_features = standardizer.transform(x[train_indices])
        all_features = standardizer.transform(x)
        config = ModelConfig(
            learning_rate=0.05,
            max_epochs=max_epochs,
            patience=max(3, min(8, max_epochs // 3 if max_epochs >= 3 else 1)),
            random_seed=model_seed,
            feature_map=feature_map,
            backend="numpy",
        )
        model, history = train_numpy_model(train_features, y[train_indices], config)
        probabilities[model_index] = _predict_all_rows(model, all_features, x.shape[0])
        training_runs.append(
            _training_run_summary(
                model_index=model_index,
                model_seed=model_seed,
                train_indices=train_indices,
                heldout_indices=heldout_indices,
                labels=y,
                history=history,
            )
        )

    mean_probability = np.mean(probabilities, axis=0)
    probability_std = np.std(probabilities, axis=0)
    model_labels = (probabilities >= threshold).astype(np.int32)
    positive_vote_rate = np.mean(model_labels, axis=0)
    label_disagreement_rate = np.minimum(positive_vote_rate, 1.0 - positive_vote_rate)
    predicted_label = (mean_probability >= threshold).astype(np.int32)
    correctness = predicted_label == y
    boundary_score = _boundary_score(mean_probability, threshold)
    instability_score = _instability_score(
        probability_std=probability_std,
        label_disagreement_rate=label_disagreement_rate,
        boundary_score=boundary_score,
    )

    ensemble_metrics = _ensemble_metrics(
        labels=y,
        mean_probability=mean_probability,
        probability_std=probability_std,
        label_disagreement_rate=label_disagreement_rate,
        instability_score=instability_score,
        threshold=threshold,
    )
    rows = _rank_rows(
        labels=y,
        mean_probability=mean_probability,
        probability_std=probability_std,
        label_disagreement_rate=label_disagreement_rate,
        positive_vote_rate=positive_vote_rate,
        predicted_label=predicted_label,
        correctness=correctness,
        boundary_score=boundary_score,
        instability_score=instability_score,
        max_rows=max_rows,
    )
    summary = _summary(rows, ensemble_metrics, instability_score)

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "model_count": int(model_count),
        "feature_map": feature_map,
        "threshold": float(threshold),
        "ensemble_metrics": ensemble_metrics,
        "summary": summary,
        "rows": rows,
        "training": {
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "max_epochs": int(max_epochs),
            "class_counts": {
                "0": int(np.sum(y == 0)),
                "1": int(np.sum(y == 1)),
            },
            "train_size_min": int(min(run["train_size"] for run in training_runs)),
            "train_size_max": int(max(run["train_size"] for run in training_runs)),
            "train_size_mean": float(np.mean([run["train_size"] for run in training_runs])),
            "heldout_size_min": int(min(run["heldout_size"] for run in training_runs)),
            "heldout_size_max": int(max(run["heldout_size"] for run in training_runs)),
            "heldout_size_mean": float(np.mean([run["heldout_size"] for run in training_runs])),
            "runs": training_runs,
        },
    }


def format_bootstrap_stability_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    metrics = report.get("ensemble_metrics", {})
    top_row = summary.get("top_row_index")
    top_text = "-" if top_row is None else str(int(top_row))
    return (
        "Bootstrap stability: "
        f"models={int(report.get('model_count', 0))}, "
        f"accuracy={float(metrics.get('accuracy', 0.0)):.4f}, "
        f"mean_std={float(summary.get('mean_probability_std', 0.0)):.4f}, "
        f"mean_disagreement={float(summary.get('mean_label_disagreement_rate', 0.0)):.4f}, "
        f"top_row={top_text}, "
        f"top_instability={float(summary.get('top_instability_score', 0.0)):.4f}"
    )


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bootstrap stability features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Bootstrap stability features must be a 2D array.")
    if x.shape[0] < 8:
        raise ValueError("Bootstrap stability diagnostics need at least 8 rows.")
    if x.shape[1] == 0:
        raise ValueError("Bootstrap stability needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Bootstrap stability features must be finite numbers.")

    raw_labels = np.asarray(labels)
    if raw_labels.ndim == 0 or raw_labels.ndim > 2:
        raise ValueError("Bootstrap stability labels must be a flat binary integer array.")
    if raw_labels.ndim == 2 and 1 not in raw_labels.shape:
        raise ValueError("Bootstrap stability labels must be a flat binary integer array.")
    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bootstrap stability labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Bootstrap stability feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Bootstrap stability labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Bootstrap stability requires strict binary integer labels 0 or 1.")
    if set(np.unique(y_values).tolist()) - {0.0, 1.0}:
        raise ValueError("Bootstrap stability requires binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if np.unique(y).size < 2:
        raise ValueError("Bootstrap stability diagnostics need both classes present.")
    return x, y


def _validate_minimum_int(value: int, name: str, minimum: int) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"Bootstrap stability {name} must be at least {minimum}.")
    return parsed


def _validate_train_fraction(value: float) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise ValueError("Bootstrap stability train_fraction must be between 0 and 1.")
    return parsed


def _validate_probability_threshold(value: float) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("Bootstrap stability threshold must be between 0 and 1.")
    return parsed


def _validate_feature_map(value: str) -> str:
    parsed = str(value).lower().strip()
    if parsed not in {"linear", "quadratic", "rff"}:
        raise ValueError("Bootstrap stability feature_map must be one of: linear, quadratic, rff.")
    return parsed


def _stratified_subsample(
    labels: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    heldout_parts: list[np.ndarray] = []
    for class_value in (0, 1):
        class_indices = np.where(labels == class_value)[0]
        shuffled = rng.permutation(class_indices)
        class_count = int(shuffled.shape[0])
        if class_count == 1:
            train_count = 1
        else:
            requested = int(round(class_count * train_fraction))
            train_count = max(1, min(class_count - 1, requested))
        train_parts.append(shuffled[:train_count])
        heldout_parts.append(shuffled[train_count:])
    train_indices = np.concatenate(train_parts).astype(np.int32)
    heldout_indices = np.concatenate(heldout_parts).astype(np.int32)
    rng.shuffle(train_indices)
    rng.shuffle(heldout_indices)
    return train_indices, heldout_indices


def _predict_all_rows(model: Any, features: np.ndarray, expected_rows: int) -> np.ndarray:
    probabilities = predict_probability(model, features).reshape(-1)
    if probabilities.shape[0] != expected_rows:
        raise ValueError("Bootstrap stability model probability count does not match features.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Bootstrap stability model returned non-finite probabilities.")
    return np.clip(probabilities.astype(np.float32), 0.0, 1.0)


def _training_run_summary(
    *,
    model_index: int,
    model_seed: int,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    labels: np.ndarray,
    history: dict[str, list[float]],
) -> dict[str, Any]:
    final_loss = None
    if history.get("loss"):
        final_loss = float(history["loss"][-1])
    return {
        "model_index": int(model_index),
        "seed": int(model_seed),
        "train_size": int(train_indices.shape[0]),
        "heldout_size": int(heldout_indices.shape[0]),
        "train_class_counts": {
            "0": int(np.sum(labels[train_indices] == 0)),
            "1": int(np.sum(labels[train_indices] == 1)),
        },
        "heldout_class_counts": {
            "0": int(np.sum(labels[heldout_indices] == 0)),
            "1": int(np.sum(labels[heldout_indices] == 1)),
        },
        "epochs_run": int(len(history.get("loss", []))),
        "final_loss": final_loss,
    }


def _boundary_score(mean_probability: np.ndarray, threshold: float) -> np.ndarray:
    width = max(float(threshold), 1.0 - float(threshold), EPSILON)
    return np.clip(1.0 - np.abs(mean_probability - float(threshold)) / width, 0.0, 1.0)


def _instability_score(
    *,
    probability_std: np.ndarray,
    label_disagreement_rate: np.ndarray,
    boundary_score: np.ndarray,
) -> np.ndarray:
    std_component = np.clip(probability_std / 0.25, 0.0, 1.0)
    disagreement_component = np.clip(label_disagreement_rate / 0.5, 0.0, 1.0)
    score = 0.40 * std_component + 0.35 * disagreement_component + 0.25 * boundary_score
    return np.clip(score, 0.0, 1.0)


def _ensemble_metrics(
    *,
    labels: np.ndarray,
    mean_probability: np.ndarray,
    probability_std: np.ndarray,
    label_disagreement_rate: np.ndarray,
    instability_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    selected = _selected_metrics(evaluate_predictions(labels, mean_probability, threshold))
    selected.update(
        {
            "mean_probability": float(np.mean(mean_probability)),
            "mean_probability_std": float(np.mean(probability_std)),
            "max_probability_std": float(np.max(probability_std)),
            "mean_label_disagreement_rate": float(np.mean(label_disagreement_rate)),
            "max_label_disagreement_rate": float(np.max(label_disagreement_rate)),
            "mean_instability_score": float(np.mean(instability_score)),
            "max_instability_score": float(np.max(instability_score)),
            "unstable_row_count": int(np.sum(instability_score >= 0.5)),
        }
    )
    return selected


def _selected_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "brier_score",
        "log_loss",
        "roc_auc",
        "average_precision",
        "predicted_positive_rate",
        "label_prevalence",
    )
    count_keys = {"true_positive", "true_negative", "false_positive", "false_negative"}
    return {
        key: int(metrics[key]) if key in count_keys else float(metrics[key])
        for key in keys
        if key in metrics
    }


def _rank_rows(
    *,
    labels: np.ndarray,
    mean_probability: np.ndarray,
    probability_std: np.ndarray,
    label_disagreement_rate: np.ndarray,
    positive_vote_rate: np.ndarray,
    predicted_label: np.ndarray,
    correctness: np.ndarray,
    boundary_score: np.ndarray,
    instability_score: np.ndarray,
    max_rows: int,
) -> list[dict[str, Any]]:
    order = sorted(
        range(labels.shape[0]),
        key=lambda index: (
            -float(instability_score[index]),
            -float(probability_std[index]),
            -float(label_disagreement_rate[index]),
            -float(boundary_score[index]),
            int(index),
        ),
    )
    rows: list[dict[str, Any]] = []
    for index in order[:max_rows]:
        disagreement = float(label_disagreement_rate[index])
        std = float(probability_std[index])
        boundary = float(boundary_score[index])
        instability = float(instability_score[index])
        rows.append(
            {
                "row_index": int(index),
                "label": int(labels[index]),
                "mean_probability": float(mean_probability[index]),
                "probability_std": std,
                "label_disagreement_rate": disagreement,
                "disagreement_rate": disagreement,
                "positive_vote_rate": float(positive_vote_rate[index]),
                "predicted_label": int(predicted_label[index]),
                "correct": bool(correctness[index]),
                "boundary_score": boundary,
                "instability_score": instability,
                "risk_flags": _risk_flags(
                    instability_score=instability,
                    probability_std=std,
                    disagreement_rate=disagreement,
                    boundary_score=boundary,
                    correct=bool(correctness[index]),
                ),
            }
        )
    return rows


def _risk_flags(
    *,
    instability_score: float,
    probability_std: float,
    disagreement_rate: float,
    boundary_score: float,
    correct: bool,
) -> list[str]:
    flags: list[str] = []
    if instability_score >= 0.50:
        flags.append("high_instability")
    if disagreement_rate >= 0.25:
        flags.append("committee_disagreement")
    if probability_std >= 0.15:
        flags.append("probability_variance")
    if boundary_score >= 0.85:
        flags.append("decision_boundary")
    if not correct:
        flags.append("ensemble_misclassified")
    return flags


def _summary(
    rows: list[dict[str, Any]],
    ensemble_metrics: dict[str, float | int],
    instability_score: np.ndarray,
) -> dict[str, Any]:
    top_row = rows[0] if rows else None
    quantiles = np.quantile(instability_score, [0.5, 0.75, 0.9, 0.95])
    unstable_count = int(ensemble_metrics.get("unstable_row_count", 0))
    warnings: list[str] = []
    if unstable_count:
        warnings.append(f"{unstable_count} row(s) exceeded the instability review threshold")
    return {
        "top_row_index": None if top_row is None else int(top_row["row_index"]),
        "top_instability_score": 0.0 if top_row is None else float(top_row["instability_score"]),
        "top_probability_std": 0.0 if top_row is None else float(top_row["probability_std"]),
        "top_label_disagreement_rate": 0.0 if top_row is None else float(top_row["label_disagreement_rate"]),
        "top_disagreement_rate": 0.0 if top_row is None else float(top_row["disagreement_rate"]),
        "mean_probability_std": float(ensemble_metrics.get("mean_probability_std", 0.0)),
        "max_probability_std": float(ensemble_metrics.get("max_probability_std", 0.0)),
        "mean_label_disagreement_rate": float(ensemble_metrics.get("mean_label_disagreement_rate", 0.0)),
        "max_label_disagreement_rate": float(ensemble_metrics.get("max_label_disagreement_rate", 0.0)),
        "mean_disagreement_rate": float(ensemble_metrics.get("mean_label_disagreement_rate", 0.0)),
        "max_disagreement_rate": float(ensemble_metrics.get("max_label_disagreement_rate", 0.0)),
        "mean_instability_score": float(ensemble_metrics.get("mean_instability_score", 0.0)),
        "max_instability_score": float(ensemble_metrics.get("max_instability_score", 0.0)),
        "instability_p50": float(quantiles[0]),
        "instability_p75": float(quantiles[1]),
        "instability_p90": float(quantiles[2]),
        "instability_p95": float(quantiles[3]),
        "unstable_row_count": unstable_count,
        "warning": "; ".join(warnings) if warnings else None,
    }
