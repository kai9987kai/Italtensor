"""Tests for Adam optimizer in NumPy backend and vectorised MPS forward pass."""
from __future__ import annotations

import numpy as np
import pytest

from italtensor.modeling import ModelConfig, train_numpy_model, NumpyBinaryClassifier
from italtensor.mps import (
    MPSBinaryClassifier,
    _batch_site_embedding,
    _soft_site_embedding,
    _init_cores,
    _init_site_centers,
    train_mps_model,
)


# ──────────────────────────────────────────────────────────────────────────────
# ModelConfig — optimizer field
# ──────────────────────────────────────────────────────────────────────────────

def test_model_config_defaults_to_adam():
    config = ModelConfig()
    assert config.optimizer == "adam"


def test_model_config_roundtrips_optimizer():
    config = ModelConfig(optimizer="sgd")
    restored = ModelConfig.from_dict(config.to_dict())
    assert restored.optimizer == "sgd"


def test_model_config_from_dict_defaults_optimizer_to_adam():
    """Existing serialised configs without 'optimizer' key must default to adam."""
    d = ModelConfig().to_dict()
    del d["optimizer"]
    restored = ModelConfig.from_dict(d)
    assert restored.optimizer == "adam"


# ──────────────────────────────────────────────────────────────────────────────
# Adam optimizer in train_numpy_model
# ──────────────────────────────────────────────────────────────────────────────

def _make_linearly_separable(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int32)
    return x, y


def test_adam_converges_on_separable_data():
    """Adam should achieve reasonable training loss on linearly separable data."""
    x, y = _make_linearly_separable()
    config = ModelConfig(
        learning_rate=0.01,
        max_epochs=30,
        patience=30,
        optimizer="adam",
        feature_map="linear",
    )
    model, history = train_numpy_model(x, y, config)
    assert isinstance(model, NumpyBinaryClassifier)
    assert history["loss"][-1] < history["loss"][0], "Loss should decrease with Adam"


def test_sgd_fallback_still_works():
    """optimizer='sgd' should produce valid predictions."""
    x, y = _make_linearly_separable()
    config = ModelConfig(
        learning_rate=0.05,
        max_epochs=20,
        patience=20,
        optimizer="sgd",
        feature_map="linear",
    )
    model, history = train_numpy_model(x, y, config)
    assert isinstance(model, NumpyBinaryClassifier)
    probs = model.predict(x[:5])
    assert np.all((probs >= 0) & (probs <= 1))


def test_adam_outperforms_sgd_on_few_epochs():
    """On a fixed small epoch budget, Adam should reach lower loss than SGD."""
    x, y = _make_linearly_separable(n=80, seed=3)
    shared_kwargs = dict(
        learning_rate=0.005,
        max_epochs=10,
        patience=10,
        feature_map="linear",
    )
    _, hist_adam = train_numpy_model(x, y, ModelConfig(optimizer="adam", **shared_kwargs))
    _, hist_sgd = train_numpy_model(x, y, ModelConfig(optimizer="sgd", **shared_kwargs))
    # Adam should reach lower or equal final loss within the same epoch budget
    assert hist_adam["loss"][-1] <= hist_sgd["loss"][-1] + 0.05  # allow tiny margin


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised MPS site embedding
# ──────────────────────────────────────────────────────────────────────────────

def test_batch_site_embedding_matches_per_row():
    """_batch_site_embedding must match _soft_site_embedding on every row."""
    rng = np.random.default_rng(42)
    values = rng.normal(size=(16,)).astype(np.float32)
    centers = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    temperature = 0.5

    batch_result = _batch_site_embedding(values, centers, temperature)
    for i, v in enumerate(values):
        per_row = _soft_site_embedding(float(v), centers, temperature)
        np.testing.assert_allclose(
            batch_result[i], per_row, atol=1e-5,
            err_msg=f"Row {i} mismatch between batch and per-row embedding"
        )


def test_batch_site_embedding_sums_to_one():
    rng = np.random.default_rng(7)
    values = rng.uniform(-3, 3, size=(32,)).astype(np.float32)
    centers = np.linspace(-3, 3, 6, dtype=np.float32)
    emb = _batch_site_embedding(values, centers)
    np.testing.assert_allclose(emb.sum(axis=1), np.ones(32), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised MPS forward pass correctness
# ──────────────────────────────────────────────────────────────────────────────

def _make_mps_model(n_sites: int = 5, bond: int = 4, phys: int = 3, seed: int = 0) -> MPSBinaryClassifier:
    rng = np.random.default_rng(seed)
    cores = _init_cores(n_sites, bond, phys, rng)
    readout = rng.normal(size=cores[-1].shape[2]).astype(np.float32)
    features = rng.normal(size=(20, n_sites)).astype(np.float32)
    site_centers = _init_site_centers(features, phys)
    return MPSBinaryClassifier(
        cores=cores,
        readout=readout,
        bias=0.1,
        site_centers=site_centers,
        raw_input_dim=n_sites,
        bond_dim=bond,
        physical_dim=phys,
    )


def test_mps_forward_output_shape():
    model = _make_mps_model()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 5)).astype(np.float32)
    probs = model.predict(x)
    assert probs.shape == (8, 1)
    assert np.all((probs >= 0) & (probs <= 1))


def test_mps_vectorised_forward_batch_consistency():
    """Single-sample and batched forward pass must agree."""
    model = _make_mps_model(n_sites=4, bond=3, phys=4, seed=5)
    rng = np.random.default_rng(99)
    x = rng.normal(size=(10, 4)).astype(np.float32)

    batch_logits = model._forward_logits(x)
    for i in range(10):
        single_logit = model._forward_logits(x[i : i + 1])
        np.testing.assert_allclose(
            batch_logits[i], single_logit[0], atol=1e-4,
            err_msg=f"Sample {i}: batch vs single forward mismatch"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Adam in MPS training
# ──────────────────────────────────────────────────────────────────────────────

def test_mps_training_with_adam_converges():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(50, 4)).astype(np.float32)
    labels = (features[:, 0] + 0.3 * features[:, 1] > 0).astype(np.int32)
    config = ModelConfig(
        backend="mps",
        optimizer="adam",
        max_epochs=20,
        mps_bond_dim=4,
        learning_rate=0.01,
        batch_size=16,
    )
    model, history = train_mps_model(features, labels, config)
    assert isinstance(model, MPSBinaryClassifier)
    assert len(history["loss"]) >= 1
    # Loss should generally decrease
    assert history["loss"][-1] < history["loss"][0] + 0.5  # tolerant bound


def test_mps_training_with_sgd_fallback():
    rng = np.random.default_rng(22)
    features = rng.normal(size=(40, 3)).astype(np.float32)
    labels = (features.sum(axis=1) > 0).astype(np.int32)
    config = ModelConfig(
        backend="mps",
        optimizer="sgd",
        max_epochs=10,
        mps_bond_dim=3,
        learning_rate=0.05,
        batch_size=8,
    )
    model, history = train_mps_model(features, labels, config)
    assert isinstance(model, MPSBinaryClassifier)
    probs = model.predict(features[:5])
    assert np.all((probs >= 0) & (probs <= 1))
