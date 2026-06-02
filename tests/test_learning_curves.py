from __future__ import annotations

import numpy as np
import pytest

from italtensor.learning_curves import (
    format_learning_curve_summary,
    learning_curve_dataset_fingerprint,
    learning_curve_points,
    run_learning_curve_diagnostics,
)
from italtensor.modeling import ModelConfig


def test_learning_curve_report_wraps_points_and_summary(monkeypatch):
    rng = np.random.default_rng(5)
    features = rng.normal(size=(32, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int32)
    config = ModelConfig(max_epochs=2, batch_size=8, backend="numpy", random_seed=3)

    def fake_train_fixed_holdout_model(x_train, y_train, x_val, y_val, trial_config):
        sample_score = min(1.0, x_train.shape[0] / 24.0)
        return {
            "metrics": {
                "f1": sample_score,
                "accuracy": sample_score,
                "balanced_accuracy": sample_score,
                "precision": sample_score,
                "recall": sample_score,
                "threshold": 0.5,
                "validation_loss": 1.0 - sample_score,
            }
        }

    monkeypatch.setattr("italtensor.learning_curves.train_fixed_holdout_model", fake_train_fixed_holdout_model)

    report = run_learning_curve_diagnostics(features, labels, config, fractions=(0.25, 0.5, 1.0), seed=11)

    assert report["sample_count"] == 32
    assert report["input_dim"] == 3
    assert len(report["points"]) == 3
    assert report["summary"]["best_f1"] == pytest.approx(report["points"][-1]["f1"])
    assert report["summary"]["verdict"] in {"more_data_helpful", "stable_enough"}
    assert "Learning curve" in format_learning_curve_summary(report)


def test_learning_curve_points_use_one_fixed_holdout(monkeypatch):
    features = np.column_stack(
        [
            np.arange(40, dtype=np.float32),
            np.linspace(-1.0, 1.0, 40, dtype=np.float32),
        ]
    )
    labels = np.asarray([0, 1] * 20, dtype=np.int32)
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_train_fixed_holdout_model(x_train, y_train, x_val, y_val, trial_config):
        seen.append((x_train[:, 0].copy(), x_val[:, 0].copy()))
        score = min(1.0, x_train.shape[0] / 32.0)
        return {
            "metrics": {
                "f1": score,
                "accuracy": score,
                "balanced_accuracy": score,
                "precision": score,
                "recall": score,
                "threshold": 0.4,
                "validation_loss": 1.0 - score,
            }
        }

    monkeypatch.setattr("italtensor.learning_curves.train_fixed_holdout_model", fake_train_fixed_holdout_model)

    points = learning_curve_points(
        features,
        labels,
        ModelConfig(backend="numpy"),
        fractions=(0.25, 0.5, 1.0),
        seed=23,
    )

    assert len(seen) == 3
    validation_ids = tuple(seen[0][1].tolist())
    for train_ids, val_ids in seen:
        assert tuple(val_ids.tolist()) == validation_ids
        assert set(train_ids.tolist()).isdisjoint(set(validation_ids))
    assert all(point["validation_samples"] == len(validation_ids) for point in points)
    assert [point["train_samples"] for point in points] == sorted(point["train_samples"] for point in points)


def test_learning_curve_points_validate_minimum_rows():
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.3], [0.3, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="at least 8"):
        learning_curve_points(features, labels, ModelConfig())


def test_learning_curve_fingerprint_is_order_sensitive():
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.3], [0.3, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    assert learning_curve_dataset_fingerprint(features, labels) != learning_curve_dataset_fingerprint(
        features[::-1],
        labels[::-1],
    )
