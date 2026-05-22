from __future__ import annotations

import numpy as np
import pytest

from italtensor.modeling import ModelConfig, train_numpy_model, NumpyBinaryClassifier
from italtensor.analysis import weight_statistics, model_similarity, bootstrap_confidence_intervals, compute_weight_histogram
from italtensor.experiments import EnsemblePredictor, train_single_model
from italtensor.preprocessing import FeatureStandardizer


def test_weight_statistics_and_histogram():
    model = NumpyBinaryClassifier(
        weights=np.array([0.0, 1.0, -1.0, 0.0, 2.0], dtype=np.float32),
        bias=0.5,
        raw_input_dim=5,
    )
    stats = weight_statistics(model)
    # weights + bias = [0.0, 1.0, -1.0, 0.0, 2.0, 0.5]
    assert stats["mean"] == pytest.approx(0.4166666)
    assert stats["min"] == -1.0
    assert stats["max"] == 2.0
    # 2 out of 6 elements are exactly zero (the two 0.0s in weights)
    assert stats["sparsity"] == pytest.approx(33.333333)
    assert stats["l1_norm"] == pytest.approx(4.5)  # 0+1+1+0+2 + 0.5 = 4.5
    assert stats["l2_norm"] == pytest.approx(2.4999999) # sqrt(0+1+1+0+4 + 0.25) = sqrt(6.25) = 2.5

    hist = compute_weight_histogram(model, bins=3)
    assert len(hist["counts"]) == 3
    assert len(hist["bin_edges"]) == 4


def test_model_similarity():
    model_a = NumpyBinaryClassifier(
        weights=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bias=0.0,
        raw_input_dim=3,
    )
    model_b = NumpyBinaryClassifier(
        weights=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bias=0.0,
        raw_input_dim=3,
    )
    assert model_similarity(model_a, model_b) == pytest.approx(1.0)

    model_c = NumpyBinaryClassifier(
        weights=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        bias=0.0,
        raw_input_dim=3,
    )
    assert model_similarity(model_a, model_c) == pytest.approx(-1.0)


def test_bootstrap_confidence_intervals():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_probs = np.array([0.9, 0.1, 0.8, 0.85, 0.2, 0.15, 0.95, 0.05, 0.75, 0.12])
    ci = bootstrap_confidence_intervals(y_true, y_probs, B=50, alpha=0.05, threshold=0.5)
    
    assert "f1" in ci
    assert "accuracy" in ci
    assert "balanced_accuracy" in ci
    assert ci["f1"][0] <= ci["f1"][1]
    assert ci["accuracy"][0] <= ci["accuracy"][1]


def test_lr_schedules_and_gradient_clip():
    features = np.random.normal(size=(20, 4)).astype(np.float32)
    labels = np.array([1, 0] * 10, dtype=np.float32)

    # Test cosine annealing
    config_cos = ModelConfig(
        max_epochs=10,
        learning_rate=0.01,
        lr_schedule="cosine",
        gradient_clip=0.5,
    )
    model, history = train_numpy_model(features, labels, config_cos)
    assert len(history["loss"]) == 10

    # Test step decay
    config_step = ModelConfig(
        max_epochs=10,
        learning_rate=0.01,
        lr_schedule="step_decay",
        patience=3,
        gradient_clip=0.1,
    )
    model, history = train_numpy_model(features, labels, config_step)
    assert len(history["loss"]) == 10


def test_ensemble_predictor():
    model_a = NumpyBinaryClassifier(
        weights=np.array([1.0, -1.0], dtype=np.float32),
        bias=0.0,
        raw_input_dim=2,
    )
    model_b = NumpyBinaryClassifier(
        weights=np.array([-1.0, 1.0], dtype=np.float32),
        bias=0.0,
        raw_input_dim=2,
    )

    preproc = FeatureStandardizer.identity(2)
    ensemble = EnsemblePredictor([(model_a, preproc), (model_b, preproc)])

    samples = np.array([[1.0, 2.0]], dtype=np.float32)
    # model_a logits: 1 - 2 = -1. prob: sigmoid(-1)
    # model_b logits: -1 + 2 = 1. prob: sigmoid(1)
    # avg prob = (sigmoid(-1) + sigmoid(1)) / 2 = (0.26894 + 0.73106) / 2 = 0.5
    mean_probs = ensemble.predict(samples)
    assert mean_probs[0, 0] == pytest.approx(0.5, abs=1e-4)

    mean, std = ensemble.predict_with_uncertainty(samples)
    assert mean[0] == pytest.approx(0.5, abs=1e-4)
    assert std[0] > 0.0

    # Test dictionary serialization
    dct = ensemble.to_dict()
    assert dct["ensemble_format_version"] == 2
    assert dct.get("fusion") == "mean"
    assert len(dct["models"]) == 2

    ensemble_loaded = EnsemblePredictor.from_dict(dct)
    assert len(ensemble_loaded.models) == 2
    assert isinstance(ensemble_loaded.models[0][0], NumpyBinaryClassifier)
