"""Learning-curve diagnostics: performance vs. training set size."""

from __future__ import annotations

from typing import Any

import numpy as np

from .experiments import train_single_model
from .modeling import ModelConfig


def learning_curve_points(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Train at increasing training fractions and return validation F1 / accuracy."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 8:
        raise ValueError("Learning curves need at least 8 samples.")

    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    holdout_count = max(2, int(round(x.shape[0] * 0.2)))
    val_idx = indices[:holdout_count]
    train_pool = indices[holdout_count:]
    x_val, y_val = x[val_idx], y[val_idx]

    points: list[dict[str, Any]] = []
    for fraction in fractions:
        frac = min(max(float(fraction), 0.1), 1.0)
        n_train = max(4, int(round(len(train_pool) * frac)))
        n_train = min(n_train, len(train_pool))
        train_idx = train_pool[:n_train]
        subset_x = x[train_idx]
        subset_y = y[train_idx]
        merged_x = np.concatenate([subset_x, x_val], axis=0)
        merged_y = np.concatenate([subset_y, y_val], axis=0)
        result = train_single_model(merged_x, merged_y, config)
        points.append(
            {
                "train_fraction": frac,
                "train_samples": int(n_train),
                "f1": float(result.metrics.get("f1", 0.0)),
                "accuracy": float(result.metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(result.metrics.get("balanced_accuracy", 0.0)),
                "validation_loss": float(result.metrics.get("validation_loss", 0.0)),
            }
        )
    return points
