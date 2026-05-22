from __future__ import annotations

import numpy as np
import pytest

from italtensor.experiments import aps_label_set, aps_metrics, aps_nonconformity_score, train_single_model
from italtensor.modeling import ModelConfig
from italtensor.mps import MPSBinaryClassifier, train_mps_model
from italtensor.persistence import model_metadata_path, save_model_bundle, load_model_bundle


def test_mps_train_and_predict():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 5)).astype(np.float32)
    labels = (features[:, 0] + 0.3 * features[:, 1] > 0).astype(np.int32)
    config = ModelConfig(backend="mps", max_epochs=15, mps_bond_dim=4, learning_rate=0.05, batch_size=8)
    model, history = train_mps_model(features, labels, config)
    assert isinstance(model, MPSBinaryClassifier)
    assert len(history["loss"]) >= 1
    probs = model.predict(features[:5])
    assert probs.shape == (5, 1)
    assert np.all((probs >= 0) & (probs <= 1))


def test_mps_train_single_model_pipeline():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(50, 4)).astype(np.float32)
    labels = (features.sum(axis=1) > 0).astype(np.int32)
    config = ModelConfig(backend="mps", max_epochs=12, mps_bond_dim=4, batch_size=10)
    result = train_single_model(features, labels, config)
    assert result.metrics.get("f1", 0) >= 0
    assert "aps_coverage" in result.metrics
    assert result.uncertainty.get("aps_tau") is not None


def test_aps_nonconformity_and_sets():
    assert aps_nonconformity_score(1, 0.9) == pytest.approx(0.9)
    assert aps_nonconformity_score(0, 0.9) == pytest.approx(1.0)
    label_set = aps_label_set(0.6, tau=0.75)
    assert 1 in label_set


def test_aps_metrics_coverage():
    labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.int32)
    probs = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1], dtype=np.float32)
    summary = aps_metrics(labels, probs, alpha=0.2)
    assert 0.0 <= summary["aps_coverage"] <= 1.0
    assert summary["aps_mean_set_size"] >= 0.0


def test_metadata_path_no_double_json(tmp_path):
    model, _ = train_mps_model(
        np.random.normal(size=(20, 3)).astype(np.float32),
        np.array([0, 1] * 10, dtype=np.int32),
        ModelConfig(backend="mps", max_epochs=5, mps_bond_dim=3),
    )
    model_path, meta_path = save_model_bundle(
        model,
        tmp_path / "chain",
        input_dim=3,
        config=ModelConfig(backend="mps"),
    )
    assert model_path.suffix == ".json"
    assert model_path.name.endswith(".italtensor-mps.json")
    assert meta_path.name.endswith(".italtensor-meta.json")
    assert meta_path.name.count(".json") == 1
    assert model_metadata_path(model_path) == meta_path
    loaded, metadata = load_model_bundle(model_path)
    assert isinstance(loaded, MPSBinaryClassifier)
    assert metadata.get("model_backend") == "mps-binary"
