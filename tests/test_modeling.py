import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from italtensor.experiments import train_single_model
from italtensor.modeling import ModelConfig, build_model, predict_probability
from italtensor.persistence import load_model_bundle, save_model_bundle


def test_model_builder_outputs_one_probability_per_sample():
    model = build_model(3, ModelConfig(hidden_layers=(8,), max_epochs=1, batch_size=2))
    output = model(np.zeros((4, 3), dtype=np.float32)).numpy()

    assert output.shape == (4, 1)


@pytest.mark.slow
def test_training_save_load_smoke(tmp_path):
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32)
    config = ModelConfig(hidden_layers=(8,), learning_rate=0.01, batch_size=2, max_epochs=2, patience=1)

    result = train_single_model(features, labels, config)
    model_path, metadata_path = save_model_bundle(
        result.model,
        tmp_path / "model.keras",
        input_dim=2,
        config=result.config,
        metrics=result.metrics,
        threshold=result.threshold,
        preprocessor=result.preprocessor,
        feature_importances=result.feature_importances,
    )
    loaded_model, metadata = load_model_bundle(model_path)
    probabilities = predict_probability(loaded_model, [0.2, 0.1])

    assert model_path.exists()
    assert metadata_path.exists()
    assert metadata["input_dim"] == 2
    assert metadata["threshold"] == result.threshold
    assert metadata["preprocessing"]["method"] == "standardize"
    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
