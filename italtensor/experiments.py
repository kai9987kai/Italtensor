from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .data import Dataset, validate_dataset
from .modeling import ModelConfig, predict_probability, train_model


@dataclass
class ExperimentResult:
    config: ModelConfig
    metrics: dict[str, float | int]
    history: dict[str, list[float]]
    model: object | None = None


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
    dataset = validate_dataset(features.tolist(), labels.tolist(), min_samples=2)
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
    dataset = validate_dataset(features.tolist(), labels.tolist(), min_samples=2)
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

    if all(len(indices) >= 2 for indices in indices_by_label.values()):
        val_indices: list[int] = []
        train_indices: list[int] = []
        for indices in indices_by_label.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            val_count = max(1, round(len(shuffled) * validation_fraction))
            val_count = min(val_count, len(shuffled) - 1)
            val_indices.extend(shuffled[:val_count])
            train_indices.extend(shuffled[val_count:])
    else:
        all_indices = list(range(dataset.sample_count))
        rng.shuffle(all_indices)
        val_count = max(1, round(dataset.sample_count * validation_fraction))
        val_count = min(val_count, dataset.sample_count - 1)
        val_indices = all_indices[:val_count]
        train_indices = all_indices[val_count:]

    return (
        dataset.features[train_indices],
        dataset.labels[train_indices],
        dataset.features[val_indices],
        dataset.labels[val_indices],
    )


def evaluate_predictions(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.int32).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    predicted = (probs >= threshold).astype(np.int32)

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    total = max(1, int(truth.shape[0]))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _default_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
) -> ExperimentResult:
    model, history = train_model(x_train, y_train, config, validation_data=(x_val, y_val))
    probabilities = predict_probability(model, x_val)
    metrics = evaluate_predictions(y_val, probabilities)
    val_losses = history.get("val_loss") or []
    metrics["validation_loss"] = float(val_losses[-1]) if val_losses else float("inf")
    return ExperimentResult(config=config, metrics=metrics, history=history, model=model)
