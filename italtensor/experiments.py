from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .data import Dataset, validate_dataset
from .modeling import ModelConfig, predict_probability, train_model
from .preprocessing import FeatureStandardizer


@dataclass
class ExperimentResult:
    config: ModelConfig
    metrics: dict[str, float | int]
    history: dict[str, list[float]]
    model: object | None = None
    threshold: float = 0.5
    preprocessor: FeatureStandardizer | None = None
    feature_importances: list[dict[str, float | int]] = field(default_factory=list)


Trainer = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ModelConfig], ExperimentResult]


def generate_random_configs(trials: int = 8, seed: int = 42) -> list[ModelConfig]:
    if trials <= 0:
        raise ValueError("trials must be greater than zero.")

    search_space = [
        ModelConfig(hidden_layers=hidden, learning_rate=lr, batch_size=batch_size, max_epochs=epochs)
        for hidden in ((16,), (32,), (64,), (64, 32))
        for lr in (0.01, 0.001, 0.0003)
        for batch_size in (8, 16, 32)
        for epochs in (20, 50)
    ]
    if trials > len(search_space):
        raise ValueError(f"trials cannot exceed {len(search_space)}.")
    rng = random.Random(seed)
    rng.shuffle(search_space)
    return [
        ModelConfig(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            random_seed=seed + index,
        )
        for index, config in enumerate(search_space[:trials])
    ]


def train_single_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    seed: int = 42,
) -> ExperimentResult:
    dataset = validate_dataset(features.tolist(), labels.tolist(), min_samples=4, require_two_classes=True)
    x_train, y_train, x_val, y_val = split_train_validation(dataset, seed=seed)
    return _default_trainer(x_train, y_train, x_val, y_val, config)


def run_experiments(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    configs: Iterable[ModelConfig] | None = None,
    trials: int = 8,
    seed: int = 42,
    trainer: Trainer | None = None,
    progress_callback: Callable[[int, int, ExperimentResult], None] | None = None,
) -> list[ExperimentResult]:
    dataset = validate_dataset(features.tolist(), labels.tolist(), min_samples=4, require_two_classes=True)
    x_train, y_train, x_val, y_val = split_train_validation(dataset, seed=seed)
    selected_configs = list(configs if configs is not None else generate_random_configs(trials, seed))
    if not selected_configs:
        raise ValueError("At least one experiment config is required.")

    run_trial = trainer or _default_trainer
    results: list[ExperimentResult] = []
    for index, config in enumerate(selected_configs, start=1):
        result = run_trial(x_train, y_train, x_val, y_val, config)
        results.append(result)
        if progress_callback is not None:
            progress_callback(index, len(selected_configs), result)
    return results


def select_best_result(results: Iterable[ExperimentResult]) -> ExperimentResult:
    result_list = list(results)
    if not result_list:
        raise ValueError("No experiment results to select from.")
    return max(
        result_list,
        key=lambda result: (
            float(result.metrics.get("f1", 0.0)),
            float(result.metrics.get("accuracy", 0.0)),
            -float(result.metrics.get("validation_loss", float("inf"))),
        ),
    )


def split_train_validation(dataset: Dataset, validation_fraction: float = 0.2, seed: int = 42):
    if dataset.sample_count < 2:
        raise ValueError("At least two samples are required for training.")

    rng = random.Random(seed)
    labels = dataset.labels.tolist()
    indices_by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(labels):
        indices_by_label[int(label)].append(index)

    if any(len(indices) < 2 for indices in indices_by_label.values()):
        raise ValueError("Training and validation require at least two samples for each class.")

    val_indices: list[int] = []
    train_indices: list[int] = []
    for indices in indices_by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * validation_fraction))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return (
        dataset.features[train_indices],
        dataset.labels[train_indices],
        dataset.features[val_indices],
        dataset.labels[val_indices],
    )


def evaluate_predictions(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.int32).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if truth.shape[0] != probs.shape[0]:
        raise ValueError("labels and probabilities must have the same length.")
    predicted = (probs >= threshold).astype(np.int32)

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    total = max(1, int(truth.shape[0]))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def optimize_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return 0.5
    unique_probs = sorted(float(probability) for probability in np.unique(probs))
    midpoints = [
        (left + right) / 2.0
        for left, right in zip(unique_probs, unique_probs[1:], strict=False)
    ]
    candidates = sorted({0.0, 0.5, 1.0, *unique_probs, *midpoints})

    def score(threshold: float) -> tuple[float, float, float]:
        metrics = evaluate_predictions(labels, probs, threshold)
        return (
            float(metrics["f1"]),
            float(metrics["balanced_accuracy"]),
            -abs(threshold - 0.5),
        )

    best_threshold = max(candidates, key=score)
    return float(best_threshold)


def balanced_class_weights(labels: np.ndarray) -> dict[int, float] | None:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    counts = np.bincount(values, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        return None
    total = float(counts.sum())
    return {0: total / (2.0 * float(counts[0])), 1: total / (2.0 * float(counts[1]))}


def permutation_feature_importance(
    model: object,
    preprocessor: FeatureStandardizer,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    max_features: int = 10,
    repeats: int = 3,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Features must be a 2D array.")
    if values.shape[0] < 2:
        return []

    base_probabilities = predict_probability(model, preprocessor.transform(values))
    baseline_f1 = float(evaluate_predictions(labels, base_probabilities, threshold)["f1"])
    variances = values.var(axis=0)
    candidate_indices = np.argsort(variances)[::-1][: max(1, min(max_features, values.shape[1]))]
    rng = np.random.default_rng(seed)
    importances: list[dict[str, float | int]] = []

    for feature_index in candidate_indices:
        scores: list[float] = []
        for _ in range(max(1, repeats)):
            permuted = values.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            probabilities = predict_probability(model, preprocessor.transform(permuted))
            scores.append(float(evaluate_predictions(labels, probabilities, threshold)["f1"]))
        mean_permuted_f1 = float(np.mean(scores))
        importances.append(
            {
                "feature_index": int(feature_index),
                "importance": float(baseline_f1 - mean_permuted_f1),
                "baseline_f1": baseline_f1,
                "permuted_f1": mean_permuted_f1,
            }
        )

    return sorted(importances, key=lambda item: float(item["importance"]), reverse=True)


def _default_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
) -> ExperimentResult:
    preprocessor = FeatureStandardizer.fit(x_train)
    x_train_prepared = preprocessor.transform(x_train)
    x_val_prepared = preprocessor.transform(x_val)
    class_weight = balanced_class_weights(y_train)
    model, history = train_model(
        x_train_prepared,
        y_train,
        config,
        validation_data=(x_val_prepared, y_val),
        class_weight=class_weight,
    )
    probabilities = predict_probability(model, x_val_prepared)
    threshold = optimize_threshold(y_val, probabilities)
    metrics = evaluate_predictions(y_val, probabilities, threshold)
    val_losses = history.get("val_loss") or []
    metrics["validation_loss"] = float(min(val_losses)) if val_losses else float("inf")
    if class_weight is not None:
        metrics["class_weight_0"] = float(class_weight[0])
        metrics["class_weight_1"] = float(class_weight[1])
    feature_importances = permutation_feature_importance(
        model,
        preprocessor,
        x_val,
        y_val,
        threshold=threshold,
        max_features=10,
        repeats=2,
        seed=config.random_seed,
    )
    return ExperimentResult(
        config=config,
        metrics=metrics,
        history=history,
        model=model,
        threshold=threshold,
        preprocessor=preprocessor,
        feature_importances=feature_importances,
    )
