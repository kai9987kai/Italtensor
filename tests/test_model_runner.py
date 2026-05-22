import numpy as np
import pytest

from italtensor.experiments import EnsemblePredictor, train_single_model
from italtensor.model_communication import ModelPanel, PanelMember, fit_stacking_weights
from italtensor.model_runner import (
    ModelRunQueue,
    available_backends,
    resolve_backend,
    run_model_queue,
    select_best_from_runs,
)
from italtensor.modeling import ModelConfig, NumpyBinaryClassifier
from italtensor.persistence import load_model_registry, save_model_registry
from italtensor.preprocessing import FeatureStandardizer
from italtensor.registry import ModelSlot


def _tiny_dataset():
    features = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]] * 4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1] * 4, dtype=np.int32)
    return features, labels


def test_resolve_backend_numpy():
    assert resolve_backend("numpy") == "numpy"


def test_resolve_backend_auto_without_tf():
    if "keras" in available_backends():
        pytest.skip("TensorFlow installed; auto may resolve to keras.")
    assert resolve_backend("auto") == "numpy"


def test_model_run_queue_numpy_only():
    features, labels = _tiny_dataset()
    config = ModelConfig(max_epochs=5, feature_map="linear", backend="numpy")
    queue = ModelRunQueue.multi_backend_sweep(config, include_keras=False)
    results = run_model_queue(features, labels, queue)
    assert len(results) == 1
    assert isinstance(results[0].model, NumpyBinaryClassifier)


def test_select_best_from_runs():
    features, labels = _tiny_dataset()
    cfg_a = ModelConfig(max_epochs=5, backend="numpy", random_seed=1)
    cfg_b = ModelConfig(max_epochs=5, backend="numpy", random_seed=2)
    results = [
        train_single_model(features, labels, cfg_a),
        train_single_model(features, labels, cfg_b),
    ]
    best = select_best_from_runs(results)
    assert best in results


def test_model_panel_consensus():
    model = NumpyBinaryClassifier(weights=np.array([1.0, 0.5]), bias=0.0, raw_input_dim=2)
    members = [
        PanelMember("a", model, FeatureStandardizer.identity(2), threshold=0.5),
        PanelMember("b", model, FeatureStandardizer.identity(2), threshold=0.5),
    ]
    panel = ModelPanel(members, fusion="mean")
    prediction = panel.predict([0.5, 0.5])
    assert len(prediction.messages) >= 3
    assert 0.0 <= float(prediction.consensus[0]) <= 1.0


def test_ensemble_weighted_fusion():
    model = NumpyBinaryClassifier(weights=np.array([1.0]), bias=0.0, raw_input_dim=1)
    prep = FeatureStandardizer.identity(1)
    ensemble = EnsemblePredictor([(model, prep), (model, prep)], fusion="weighted", member_weights=[0.25, 0.75])
    probs = ensemble.predict([[1.0]]).reshape(-1)
    assert probs.shape == (1,)


def test_registry_round_trip(tmp_path):
    model = NumpyBinaryClassifier(weights=np.array([1.0]), bias=0.0, raw_input_dim=1)
    slot = ModelSlot(
        model=model,
        config=ModelConfig(),
        metrics={"f1": 0.5},
        preprocessor=FeatureStandardizer.identity(1),
        threshold=0.5,
        name="test-slot",
    )
    path = tmp_path / "registry.json"
    save_model_registry(path, [slot], input_dim=1)
    loaded, input_dim = load_model_registry(path)
    assert input_dim == 1
    assert len(loaded) == 1
    assert loaded[0].name == "test-slot"
    assert isinstance(loaded[0].model, NumpyBinaryClassifier)


def test_fit_stacking_weights():
    features, labels = _tiny_dataset()
    model = NumpyBinaryClassifier(weights=np.array([0.5, 0.5]), bias=0.0, raw_input_dim=2)
    members = [PanelMember("m", model, FeatureStandardizer.identity(2))]
    coef = fit_stacking_weights(members, features, labels)
    assert coef.shape[0] == 2
