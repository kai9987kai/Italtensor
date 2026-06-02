from __future__ import annotations

import numpy as np
import pytest

from italtensor.mps_diagnostics import (
    format_mps_order_sweep_summary,
    format_mps_sweep_summary,
    run_mps_bond_sweep,
    run_mps_order_sweep,
)
from italtensor.modeling import ModelConfig


def test_mps_bond_sweep_ranks_dimensions():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(40, 4)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int32)
    config = ModelConfig(max_epochs=8, batch_size=8, mps_bond_dim=8, backend="mps", random_seed=3)
    report = run_mps_bond_sweep(features, labels, config, bond_dims=(4, 8))
    assert len(report["results"]) == 2
    assert report["recommended_bond_dim"] in (4, 8)
    summary = format_mps_sweep_summary(report)
    assert "MPS bond sweep" in summary


def test_mps_order_sweep_ranks_candidate_orders_with_fake_trainer(monkeypatch):
    rng = np.random.default_rng(4)
    features = rng.normal(size=(48, 5)).astype(np.float32)
    labels = (features[:, 2] > 0.0).astype(np.int32)
    config = ModelConfig(max_epochs=3, batch_size=8, mps_bond_dim=4, backend="mps", random_seed=5)

    class FakeMpsModel:
        def predict(self, x):
            return (1.0 / (1.0 + np.exp(-4.0 * x[:, 0]))).reshape(-1, 1)

    def fake_train_model(x_train, y_train, trial_config, validation_data=None):
        return FakeMpsModel(), {"loss": [0.4, 0.2], "val_loss": [0.45, 0.25]}

    monkeypatch.setattr("italtensor.mps_diagnostics.train_mps_model", fake_train_model)

    report = run_mps_order_sweep(features, labels, config, max_orders=4)

    assert len(report["results"]) <= 4
    assert "original" in report["orders_tested"]
    assert sorted(report["recommended_order"]) == list(range(5))
    assert report["recommended_feature_order_1_based"][0] == 3
    assert report["recommended_f1"] >= 0.9
    assert report["original_f1"] is not None
    assert "site-order sensitivity" in report["adoption_note"]
    assert "MPS order sweep" in format_mps_order_sweep_summary(report)


def test_mps_order_sweep_rejects_bad_permutations():
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.2], [0.2, -1.0]] * 3, dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1] * 3, dtype=np.int32)
    config = ModelConfig(max_epochs=1, batch_size=4, mps_bond_dim=4, backend="mps", random_seed=1)

    with pytest.raises(ValueError, match="permutation"):
        run_mps_order_sweep(features, labels, config, orders={"bad": [0, 0]})


def test_mps_order_sweep_validates_inputs():
    config = ModelConfig(max_epochs=1, batch_size=4, mps_bond_dim=4, backend="mps", random_seed=1)
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.2], [0.2, -1.0]] * 2, dtype=np.float32)

    with pytest.raises(ValueError, match="both classes"):
        run_mps_order_sweep(features, np.zeros(8, dtype=np.int32), config)
    with pytest.raises(ValueError, match="validation_fraction"):
        run_mps_order_sweep(features, np.asarray([0, 1, 0, 1] * 2, dtype=np.int32), config, validation_fraction=1.0)
