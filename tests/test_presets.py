import json

import pytest

from italtensor.app import AppState, _replace_dataset
from italtensor.data import DataValidationError, validate_dataset
from italtensor.experiments import split_train_validation
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer
from italtensor.presets import BUILT_IN_PRESETS, generate_builtin_preset, load_preset_file, save_preset_file


def test_preset_round_trip_preserves_dataset_and_metadata(tmp_path):
    dataset = validate_dataset([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.7]], [0, 1, 0, 1])
    path = save_preset_file(tmp_path / "preset.json", dataset, name="Demo", description="Reusable demo")

    loaded, metadata = load_preset_file(path)

    assert loaded.input_dim == dataset.input_dim
    assert loaded.features.tolist() == dataset.features.tolist()
    assert loaded.labels.tolist() == dataset.labels.tolist()
    assert metadata["name"] == "Demo"
    assert metadata["description"] == "Reusable demo"
    assert metadata["schema_version"] == 1
    assert metadata["training_defaults"]["trials"] == 8


def test_load_preset_accepts_plain_dataset_json(tmp_path):
    dataset = validate_dataset([[0.1], [0.8]], [0, 1])
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "input_dim": 1,
                "samples": [
                    {"features": [0.1], "label": 0},
                    {"features": [0.8], "label": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded, metadata = load_preset_file(path)

    assert loaded.features.tolist() == dataset.features.tolist()
    assert metadata["name"] == "dataset"


def test_load_preset_rejects_missing_dataset_or_bad_schema_version(tmp_path):
    missing_dataset = tmp_path / "missing.json"
    missing_dataset.write_text(json.dumps({"schema_version": 1, "name": "bad"}), encoding="utf-8")
    bad_version = tmp_path / "bad-version.json"
    bad_version.write_text(
        json.dumps({"schema_version": 999, "name": "bad", "dataset": {"input_dim": 1, "samples": []}}),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="contain a dataset"):
        load_preset_file(missing_dataset)
    with pytest.raises(DataValidationError, match="Unsupported preset"):
        load_preset_file(bad_version)


def test_builtin_presets_are_trainable():
    for preset in BUILT_IN_PRESETS:
        dataset = generate_builtin_preset(preset.name, sample_count=preset.default_samples, seed=7)
        assert dataset.sample_count >= preset.min_samples
        assert set(dataset.labels.tolist()) == {0, 1}
        x_train, y_train, x_val, y_val = split_train_validation(dataset)
        assert x_train.shape[0] >= 2
        assert x_val.shape[0] >= 2
        assert set(y_train.tolist()) == {0, 1}
        assert set(y_val.tolist()) == {0, 1}


def test_experimental_builtin_presets_are_available():
    names = {preset.name for preset in BUILT_IN_PRESETS}

    assert {"Concentric rings", "Two moons", "Rare event signal"}.issubset(names)


def test_save_as_preset_uses_existing_dataset_json_shape(tmp_path):
    dataset = validate_dataset([[1.0, 2.0], [3.0, 4.0]], [0, 1])
    path = save_preset_file(tmp_path / "shape.json", dataset, name="Shape")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset"]["input_dim"] == 2
    assert payload["dataset"]["samples"][0] == {"features": [1.0, 2.0], "label": 0}


def test_replace_dataset_from_preset_invalidates_old_model_state():
    state = AppState(
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 1.0},
        latest_threshold=0.8,
        preprocessor=FeatureStandardizer.identity(2),
    )
    dataset = generate_builtin_preset("Linear blobs", sample_count=12, seed=3)

    _replace_dataset(state, dataset)

    assert len(state.labels) == 12
    assert state.model is None
    assert state.latest_metrics == {}
    assert state.preprocessor is None
