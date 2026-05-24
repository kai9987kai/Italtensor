import importlib.util

import numpy as np
import pytest

from italtensor.modeling import ModelConfig, NumpyBinaryClassifier, predict_probability, train_numpy_model
from italtensor.persistence import load_model_bundle, save_model_bundle
from italtensor.preprocessing import FeatureStandardizer


def test_numpy_fallback_trains_predicts_and_tracks_validation_loss():
    features = np.asarray(
        [[-1.0, -1.0], [-0.8, -0.9], [0.8, 0.9], [1.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    config = ModelConfig(learning_rate=0.05, max_epochs=20, patience=3, random_seed=3)

    model, history = train_numpy_model(
        features[:2],
        labels[:2],
        config,
        validation_data=(features[2:], labels[2:]),
        class_weight={0: 1.0, 1: 1.0},
    )
    probabilities = predict_probability(model, features)

    assert isinstance(model, NumpyBinaryClassifier)
    assert model.input_dim == 2
    assert model.input_shape == (None, 2)
    assert probabilities.shape == (4,)
    assert all(0.0 <= float(probability) <= 1.0 for probability in probabilities)
    assert history["loss"]
    assert history["val_loss"]


def test_numpy_fallback_predict_rejects_wrong_feature_count():
    model = NumpyBinaryClassifier(weights=np.asarray([1.0, -1.0], dtype=np.float32), bias=0.0)

    try:
        model.predict([[1.0, 2.0, 3.0]])
    except ValueError as exc:
        assert "Expected 2 features" in str(exc)
    else:
        raise AssertionError("Expected wrong feature count to fail.")


def test_numpy_fallback_model_json_includes_shape_metadata():
    model = NumpyBinaryClassifier(weights=np.asarray([1.0, -1.0], dtype=np.float32), bias=0.25)
    payload = model.to_dict()

    assert payload["model_format_version"] == 1
    assert payload["backend"] == "numpy-logistic"
    assert payload["input_dim"] == 2
    assert NumpyBinaryClassifier.from_dict(payload).input_dim == 2


def test_quadratic_feature_map_can_learn_xor_pattern():
    features = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]] * 12,
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 0] * 12, dtype=np.int32)
    config = ModelConfig(feature_map="quadratic", learning_rate=0.1, max_epochs=150, patience=20, random_seed=5)

    model, history = train_numpy_model(features, labels, config)
    probabilities = predict_probability(model, features)
    predictions = (probabilities >= 0.5).astype(np.int32)

    assert model.feature_map == "quadratic"
    assert model.input_dim == 2
    assert float(np.mean(predictions == labels)) >= 0.95
    assert history["loss"][-1] < history["loss"][0]


def test_rff_feature_map_round_trips_with_random_parameters():
    features = np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    model, _ = train_numpy_model(
        features,
        labels,
        ModelConfig(feature_map="rff", rff_components=8, learning_rate=0.05, max_epochs=4, random_seed=9),
    )

    restored = NumpyBinaryClassifier.from_dict(model.to_dict())

    assert restored.feature_map == "rff"
    assert restored.input_dim == 1
    np.testing.assert_allclose(predict_probability(restored, features), predict_probability(model, features))


def test_rff_model_rejects_bad_parameter_shapes():
    payload = {
        "backend": "numpy-logistic",
        "input_dim": 2,
        "feature_map": "rff",
        "weights": [0.1, 0.2, 0.3],
        "bias": 0.0,
        "rff_weights": [[0.1, 0.2, 0.3]],
        "rff_bias": [0.1, 0.2],
    }

    try:
        NumpyBinaryClassifier.from_dict(payload)
    except ValueError as exc:
        assert "RFF input_dim" in str(exc) or "RFF bias length" in str(exc)
    else:
        raise AssertionError("Expected invalid RFF metadata to fail.")


def test_numpy_fallback_model_bundle_round_trip(tmp_path):
    model = NumpyBinaryClassifier(weights=np.asarray([1.0, -1.0], dtype=np.float32), bias=0.25)
    preprocessor = FeatureStandardizer.identity(2)

    model_path, metadata_path = save_model_bundle(
        model,
        tmp_path / "fallback.keras",
        input_dim=2,
        config=ModelConfig(max_epochs=5),
        metrics={"f1": 0.8},
        threshold=0.4,
        preprocessor=preprocessor,
        feature_importances=[{"feature_index": 0, "importance": 0.5}],
        trial_history=[{"metrics": {"f1": 0.8}}],
        uncertainty_metadata={"conformal_quantile": 0.33, "conformal_coverage": 0.9},
        ablation_report={"summary": {"top_feature": "x1"}},
        sample_review_report={"summary": {"label_issue_count": 1}},
        threshold_report={"summary": {"best_f1_threshold": 0.3}},
        slice_report={"summary": {"worst_slice": "x1[0, 1]"}},
        stress_report={"summary": {"worst_f1": 0.7}},
    )
    loaded, metadata = load_model_bundle(model_path)
    probabilities = predict_probability(loaded, [[1.0, 0.0]])

    assert model_path.name == "fallback.italtensor-model.json"
    assert metadata_path.name == "fallback.italtensor-meta.json"
    assert isinstance(loaded, NumpyBinaryClassifier)
    assert metadata["model_format_version"] == 1
    assert metadata["model_backend"] == "numpy-logistic"
    assert metadata["model_feature_map"] == "linear"
    assert metadata["threshold"] == 0.4
    assert metadata["preprocessing"]["method"] == "standardize"
    assert metadata["trial_history"][0]["metrics"]["f1"] == 0.8
    assert metadata["uncertainty"]["conformal_quantile"] == 0.33
    assert metadata["feature_ablation_diagnostics"]["summary"]["top_feature"] == "x1"
    assert metadata["sample_review"]["summary"]["label_issue_count"] == 1
    assert metadata["threshold_diagnostics"]["summary"]["best_f1_threshold"] == 0.3
    assert metadata["slice_diagnostics"]["summary"]["worst_slice"] == "x1[0, 1]"
    assert metadata["stress_lab"]["summary"]["worst_f1"] == 0.7
    assert probabilities.shape == (1,)


def test_keras_load_without_tensorflow_has_actionable_error(tmp_path):
    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("TensorFlow is installed; no-TensorFlow error path is not active.")
    model_path = tmp_path / "model.keras"
    model_path.write_text("not a real keras archive", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"\.keras models require the optional TensorFlow backend"):
        load_model_bundle(model_path)
