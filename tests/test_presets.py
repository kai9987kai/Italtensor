import json

import pytest

from italtensor.app import AppState, _replace_dataset
from italtensor.data import DataValidationError, validate_dataset
from italtensor.experiments import split_train_validation
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer
from italtensor.presets import BUILT_IN_PRESETS, generate_builtin_preset, load_preset_file, preset_metadata, save_preset_file


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

    assert {
        "Concentric rings",
        "Two moons",
        "Rare event signal",
        "Overlapping margin",
        "Noisy labels",
        "Sparse interaction signal",
        "Deployment drift probe",
        "Active learning margin",
        "Spurious shortcut",
        "Subgroup blind spot",
        "Cost-sensitive screening",
        "Decision utility tradeoff",
        "Selective abstention triage",
        "Label audit traps",
        "Proxy leakage lab",
    }.issubset(names)


def test_builtin_preset_metadata_has_recommendations_and_examples():
    metadata = preset_metadata("XOR pattern")

    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["training_defaults"]["feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["x1", "x2"]
    assert metadata["label_names"] == {"0": "negative", "1": "positive"}
    assert len(metadata["prediction_examples"]) >= 2


def test_sparse_interaction_preset_applies_feature_selection_defaults():
    metadata = preset_metadata("Sparse interaction signal")

    assert metadata["input_dim"] == 16
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["training_defaults"]["l1_penalty"] == 0.001
    assert metadata["training_defaults"]["feature_selection_k"] == 6


def test_deployment_drift_preset_has_shifted_prediction_example():
    metadata = preset_metadata("Deployment drift probe")

    assert metadata["input_dim"] == 4
    assert any(example["name"] == "Drift review row" for example in metadata["prediction_examples"])


def test_noisy_labels_preset_has_margin_example():
    metadata = preset_metadata("Noisy labels")

    assert metadata["recommended_feature_map"] == "linear"
    assert any(example["name"] == "Ambiguous noisy margin" for example in metadata["prediction_examples"])


def test_active_learning_margin_preset_has_boundary_query_example():
    metadata = preset_metadata("Active learning margin")

    assert metadata["recommended_feature_map"] == "linear"
    assert any(example["name"] == "Boundary query" for example in metadata["prediction_examples"])


def test_spurious_shortcut_preset_has_conflict_example():
    metadata = preset_metadata("Spurious shortcut")

    assert metadata["input_dim"] == 3
    assert metadata["feature_names"] == ["stable_signal", "context_noise", "shortcut_signal"]
    assert any(example["name"] == "Shortcut conflict" for example in metadata["prediction_examples"])


def test_subgroup_blind_spot_preset_recommends_interactions():
    metadata = preset_metadata("Subgroup blind spot")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["primary_signal", "subgroup_marker", "context_noise"]
    assert any(example["name"] == "Minority flipped rule" for example in metadata["prediction_examples"])


def test_cost_sensitive_screening_preset_has_borderline_example():
    metadata = preset_metadata("Cost-sensitive screening")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["risk_score", "secondary_signal", "background_noise"]
    assert any(example["name"] == "Borderline review" for example in metadata["prediction_examples"])


def test_decision_utility_tradeoff_preset_has_gray_zone_example():
    metadata = preset_metadata("Decision utility tradeoff")

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["risk_score", "triage_signal", "noise_marker", "gray_zone"]
    assert any(example["name"] == "Decision gray zone" for example in metadata["prediction_examples"])


def test_selective_abstention_triage_preset_has_abstention_candidate():
    metadata = preset_metadata("Selective abstention triage")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["triage_score", "support_signal", "ambiguity_marker"]
    assert any(example["name"] == "Abstention candidate" for example in metadata["prediction_examples"])


def test_label_audit_traps_preset_has_suspicious_example():
    metadata = preset_metadata("Label audit traps")

    assert metadata["recommended_feature_map"] == "linear"
    assert any(example["name"] == "Suspicious positive-shaped negative" for example in metadata["prediction_examples"])


def test_proxy_leakage_lab_preset_supports_ablation_diagnostics():
    metadata = preset_metadata("Proxy leakage lab")

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["real_signal", "weak_signal", "proxy_code", "background_noise"]
    assert any(example["name"] == "Proxy conflict" for example in metadata["prediction_examples"])


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
