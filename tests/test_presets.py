import json

import numpy as np
import pytest

from italtensor.app import AppState, _replace_dataset
from italtensor.data import DataValidationError, validate_dataset
from italtensor.data_acquisition import run_data_acquisition_planner
from italtensor.data_value import run_data_value_scout
from italtensor.experiments import split_train_validation
from italtensor.label_sensitivity import run_label_sensitivity
from italtensor.label_noise_stress import run_label_noise_stress
from italtensor.leakage_sentinel import run_leakage_sentinel
from italtensor.modeling import ModelConfig, NumpyBinaryClassifier
from italtensor.preprocessing import FeatureStandardizer
from italtensor.presets import BUILT_IN_PRESETS, generate_builtin_preset, load_preset_file, preset_metadata, save_preset_file
from italtensor.validation_plan import run_validation_plan


def test_preset_round_trip_preserves_dataset_and_metadata(tmp_path):
    dataset = validate_dataset([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.7]], [0, 1, 0, 1])
    path = save_preset_file(
        tmp_path / "preset.json",
        dataset,
        name="Demo",
        description="Reusable demo",
        training_defaults={
            "epochs": 44,
            "batch_size": 8,
            "trials": 6,
            "feature_map": "quadratic",
            "backend": "numpy",
            "gradient_clip": 0.5,
            "l1_penalty": 0.001,
            "feature_selection_k": 2,
        },
        recommended_feature_map="quadratic",
        feature_names=["left", "right"],
        label_names={"0": "reject", "1": "accept"},
        prediction_examples=[{"name": "Review row", "features": [0.4, 0.5], "expected_label": None}],
        policy_checks=[
            {
                "name": "Left should increase",
                "feature_index": 0,
                "feature_name": "left",
                "direction": "increasing",
            }
        ],
    )

    loaded, metadata = load_preset_file(path)

    assert loaded.input_dim == dataset.input_dim
    assert loaded.features.tolist() == dataset.features.tolist()
    assert loaded.labels.tolist() == dataset.labels.tolist()
    assert metadata["name"] == "Demo"
    assert metadata["description"] == "Reusable demo"
    assert metadata["schema_version"] == 1
    assert metadata["training_defaults"]["epochs"] == 44
    assert metadata["training_defaults"]["trials"] == 6
    assert metadata["training_defaults"]["feature_map"] == "quadratic"
    assert metadata["training_defaults"]["backend"] == "numpy"
    assert metadata["training_defaults"]["gradient_clip"] == 0.5
    assert metadata["training_defaults"]["l1_penalty"] == 0.001
    assert metadata["training_defaults"]["feature_selection_k"] == 2
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["left", "right"]
    assert metadata["label_names"] == {"0": "reject", "1": "accept"}
    assert metadata["prediction_examples"][0]["features"] == [0.4, 0.5]
    assert metadata["policy_checks"][0]["feature_name"] == "left"
    assert metadata["policy_checks"][0]["direction"] == "increasing"


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


def test_load_preset_rejects_malformed_prediction_examples(tmp_path):
    bad_examples = tmp_path / "bad-examples.json"
    bad_examples.write_text(
        json.dumps(
            {
                "kind": "italtensor.dataset_preset",
                "schema_version": 1,
                "name": "bad examples",
                "prediction_examples": [{"name": "bad width", "features": [0.1], "expected_label": 1}],
                "dataset": {
                    "input_dim": 2,
                    "samples": [
                        {"features": [0.1, 0.2], "label": 0},
                        {"features": [0.8, 0.9], "label": 1},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="prediction example"):
        load_preset_file(bad_examples)


def test_load_preset_rejects_malformed_policy_checks(tmp_path):
    bad_policy = tmp_path / "bad-policy.json"
    bad_policy.write_text(
        json.dumps(
            {
                "kind": "italtensor.dataset_preset",
                "schema_version": 1,
                "name": "bad policy",
                "policy_checks": [{"name": "bad direction", "feature_index": 0, "direction": "sideways"}],
                "dataset": {
                    "input_dim": 1,
                    "samples": [
                        {"features": [0.1], "label": 0},
                        {"features": [0.8], "label": 1},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="direction"):
        load_preset_file(bad_policy)


def test_save_preset_rejects_inconsistent_metadata(tmp_path):
    dataset = validate_dataset([[0.1, 0.2], [0.8, 0.9]], [0, 1])

    with pytest.raises(DataValidationError, match="feature_names"):
        save_preset_file(tmp_path / "bad-features.json", dataset, name="Bad", feature_names=["only_one"])
    with pytest.raises(DataValidationError, match="recommended_feature_map"):
        save_preset_file(tmp_path / "bad-map.json", dataset, name="Bad", recommended_feature_map="unknown")
    with pytest.raises(DataValidationError, match="prediction example"):
        save_preset_file(
            tmp_path / "bad-example.json",
            dataset,
            name="Bad",
            prediction_examples=[{"name": "Bad width", "features": [0.1], "expected_label": 1}],
        )


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
        "Subgroup disparity lab",
        "Response curve lab",
        "Interaction surface lab",
        "Calibration repair lab",
        "Reliability atlas lab",
        "Calibration slice lab",
        "Permutation null lab",
        "Population drift lab",
        "Prior shift lab",
        "External holdout lab",
        "Adversarial validation lab",
        "Chronological holdout lab",
        "Shadow replay lab",
        "Threshold stability lab",
        "Capacity planner lab",
        "Rank lift lab",
        "Cost-sensitive screening",
        "Decision utility tradeoff",
        "Selective abstention triage",
        "Conformal coverage lab",
        "Label audit traps",
        "Label sensitivity lab",
        "Label noise stress lab",
        "Error atlas lab",
        "OOD sentinel lab",
        "Bootstrap stability lab",
        "Prototype coverage lab",
        "Separability lens lab",
        "Neighborhood hardness lab",
        "Dataset triage lab",
        "Data acquisition lab",
        "Data value scout lab",
        "Validation plan lab",
        "Learning curve lab",
        "Schema guard lab",
        "Canary regression lab",
        "Monotonic policy lab",
        "Experiment advisor lab",
        "Proxy leakage lab",
        "Leakage sentinel lab",
        "MPS locality lab",
        "Promotion gate lab",
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


def test_mps_locality_lab_preset_uses_mps_defaults():
    metadata = preset_metadata("MPS locality lab")
    dataset = generate_builtin_preset("MPS locality lab", sample_count=80, seed=9)

    assert metadata["input_dim"] == 6
    assert metadata["training_defaults"]["backend"] == "mps"
    assert metadata["training_defaults"]["mps_bond_dim"] == 4
    assert metadata["feature_names"][:4] == ["left_gate", "left_partner", "right_gate", "right_partner"]
    assert any(example["name"] == "Order-sensitivity review" for example in metadata["prediction_examples"])
    assert dataset.sample_count == 80
    assert set(dataset.labels.tolist()) == {0, 1}


def test_validation_plan_lab_preset_preserves_ordered_split_risk():
    metadata = preset_metadata("Validation plan lab")
    dataset = generate_builtin_preset("Validation plan lab", sample_count=120, seed=11)
    report = run_validation_plan(dataset.features, dataset.labels)

    assert metadata["input_dim"] == 5
    assert metadata["training_defaults"]["use_cv"] is True
    assert metadata["feature_names"] == [
        "stable_margin",
        "support_signal",
        "regime_shift",
        "prevalence_marker",
        "background_noise",
    ]
    assert report["summary"]["recommended_strategy"] == "chronological_holdout"
    assert report["summary"]["row_order_risk"] is True


def test_data_acquisition_lab_preset_has_collection_gaps():
    metadata = preset_metadata("Data acquisition lab")
    dataset = generate_builtin_preset("Data acquisition lab", sample_count=120, seed=13)
    report = run_data_acquisition_planner(dataset.features, dataset.labels, min_class_count=40)

    assert metadata["input_dim"] == 5
    assert metadata["feature_names"] == [
        "margin_signal",
        "support_signal",
        "boundary_band",
        "tail_probe",
        "collection_source",
    ]
    assert any(example["name"] == "Boundary acquisition row" for example in metadata["prediction_examples"])
    assert report["summary"]["priority"] == "high"
    assert any(item["category"] == "class_balance" for item in report["recommendations"])
    assert int(np.sum(dataset.features[:, 2] > 0.7)) >= 18
    assert float(np.max(dataset.features[:, 3])) > 3.5


def test_data_value_scout_lab_preset_has_row_curation_cases():
    metadata = preset_metadata("Data value scout lab")
    dataset = generate_builtin_preset("Data value scout lab", sample_count=120, seed=14)
    report = run_data_value_scout(dataset.features, dataset.labels, k=5)

    assert metadata["input_dim"] == 5
    assert metadata["feature_names"] == [
        "primary_margin",
        "support_signal",
        "duplicate_marker",
        "rare_coverage",
        "conflict_marker",
    ]
    assert any(example["name"] == "Conflict review row" for example in metadata["prediction_examples"])
    assert report["summary"]["priority"] == "high"
    assert report["summary"]["review_row_count"] >= 2
    assert report["summary"]["redundant_row_count"] >= 2
    assert report["summary"]["coverage_row_count"] >= 1


def test_learning_curve_lab_preset_has_boundary_rows():
    metadata = preset_metadata("Learning curve lab")
    dataset = generate_builtin_preset("Learning curve lab", sample_count=120, seed=12)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == [
        "margin_signal",
        "support_signal",
        "boundary_band",
        "label_noise_probe",
    ]
    assert any(example["name"] == "Boundary learning row" for example in metadata["prediction_examples"])
    assert dataset.sample_count == 120
    assert set(dataset.labels.tolist()) == {0, 1}
    assert int(np.sum(dataset.features[:, 2] > 0.7)) >= 20
    assert int(np.sum(dataset.features[:, 3] > 1.0)) >= 2


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


def test_leakage_sentinel_lab_preset_has_direct_label_code():
    metadata = preset_metadata("Leakage sentinel lab")
    dataset = generate_builtin_preset("Leakage sentinel lab", sample_count=120, seed=12)
    report = run_leakage_sentinel(dataset.features, dataset.labels)

    assert metadata["input_dim"] == 5
    assert metadata["feature_names"] == [
        "stable_signal",
        "proxy_bucket",
        "label_code",
        "review_timestamp",
        "background_noise",
    ]
    assert any(example["name"] == "Proxy conflict review" for example in metadata["prediction_examples"])
    assert report["summary"]["risk_level"] == "high"
    assert report["summary"]["top_feature"] == 2
    assert "direct_label_copy_candidate" in report["features"][0]["risk_flags"]


def test_subgroup_blind_spot_preset_recommends_interactions():
    metadata = preset_metadata("Subgroup blind spot")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["primary_signal", "subgroup_marker", "context_noise"]
    assert any(example["name"] == "Minority flipped rule" for example in metadata["prediction_examples"])


def test_subgroup_disparity_lab_preset_has_marker_feature():
    metadata = preset_metadata("Subgroup disparity lab")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["primary_signal", "group_marker", "context_noise"]
    assert any(example["name"] == "Minority conflict" for example in metadata["prediction_examples"])


def test_response_curve_lab_preset_has_nonlinear_feature():
    metadata = preset_metadata("Response curve lab")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["linear_driver", "nonlinear_arc", "weak_noise"]
    assert any(example["name"] == "Arc peak" for example in metadata["prediction_examples"])


def test_interaction_surface_lab_preset_has_factor_features():
    metadata = preset_metadata("Interaction surface lab")

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == ["left_factor", "right_factor", "weak_signal", "distractor_noise"]
    assert any(example["name"] == "Both factors aligned" for example in metadata["prediction_examples"])


def test_calibration_repair_lab_preset_has_confidence_trap():
    metadata = preset_metadata("Calibration repair lab")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["margin_score", "confidence_trap", "background_noise"]
    assert any(example["name"] == "Miscalibrated shoulder" for example in metadata["prediction_examples"])


def test_reliability_atlas_lab_preset_has_calibration_bands():
    metadata = preset_metadata("Reliability atlas lab")
    dataset = generate_builtin_preset("Reliability atlas lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "raw_score",
        "overconfidence_band",
        "underconfidence_band",
        "background_noise",
    ]
    assert any(example["name"] == "Overconfident review" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 1] > 0.5)) >= 15
    assert int(np.sum(dataset.features[:, 2] > 0.5)) >= 15


def test_permutation_null_lab_preset_has_decoy_and_boundary_example():
    metadata = preset_metadata("Permutation null lab")

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["real_margin", "weak_support", "label_noise_marker", "decoy_noise"]
    assert any(example["name"] == "Boundary null check" for example in metadata["prediction_examples"])


def test_population_drift_lab_preset_has_shift_probe_features():
    metadata = preset_metadata("Population drift lab")

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "stable_signal",
        "shifted_signal",
        "variance_drift",
        "tail_probe",
        "decoy_noise",
    ]
    assert any(example["name"] == "Current shifted review" for example in metadata["prediction_examples"])


def test_adversarial_validation_lab_preset_has_multivariate_shift_features():
    metadata = preset_metadata("Adversarial validation lab")

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "quadratic"
    assert metadata["feature_names"] == [
        "stable_signal",
        "domain_shift_axis",
        "interaction_shift",
        "variance_marker",
        "decoy_noise",
    ]
    assert any(example["name"] == "Current multivariate shift" for example in metadata["prediction_examples"])


def test_chronological_holdout_lab_preset_preserves_order_and_has_temporal_features():
    metadata = preset_metadata("Chronological holdout lab")
    dataset = generate_builtin_preset("Chronological holdout lab", sample_count=80, seed=9)

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "early_signal",
        "late_signal_decay",
        "threshold_drift",
        "calibration_wobble",
        "decoy_noise",
    ]
    assert any(example["name"] == "Late degraded review" for example in metadata["prediction_examples"])
    assert float(dataset.features[:48, 2].mean()) < float(dataset.features[48:, 2].mean())


def test_shadow_replay_lab_preset_preserves_order_and_late_regime():
    metadata = preset_metadata("Shadow replay lab")
    dataset = generate_builtin_preset("Shadow replay lab", sample_count=100, seed=9)

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "stable_margin",
        "late_counter_signal",
        "regime_marker",
        "review_band",
        "decoy_noise",
    ]
    assert any(example["name"] == "Late replay review" for example in metadata["prediction_examples"])
    assert float(dataset.features[:55, 2].mean()) < -0.8
    assert float(dataset.features[-20:, 2].mean()) > 0.8


def test_calibration_slice_lab_preset_has_local_miscalibration_pocket():
    metadata = preset_metadata("Calibration slice lab")
    dataset = generate_builtin_preset("Calibration slice lab", sample_count=160, seed=10)
    pocket_mask = dataset.features[:, 1] > 0.5

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "raw_score",
        "calibration_pocket",
        "support_signal",
        "pocket_noise",
        "background_noise",
    ]
    assert any(example["name"] == "Overconfident pocket review" for example in metadata["prediction_examples"])
    assert int(np.sum(pocket_mask)) >= 25
    assert float(dataset.features[pocket_mask, 0].mean()) > 0.6
    assert float(np.mean(dataset.labels[pocket_mask] == 1)) < 0.3


def test_threshold_stability_lab_preset_has_boundary_band():
    metadata = preset_metadata("Threshold stability lab")
    dataset = generate_builtin_preset("Threshold stability lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["risk_score", "support_signal", "boundary_band", "decoy_noise"]
    assert any(example["name"] == "Boundary threshold review" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 0.7)) >= 30
    assert 0.35 <= float(np.mean(dataset.labels == 1)) <= 0.65


def test_capacity_planner_lab_preset_has_false_alarm_decoys():
    metadata = preset_metadata("Capacity planner lab")
    dataset = generate_builtin_preset("Capacity planner lab", sample_count=160, seed=10)
    positive_rate = float(np.mean(dataset.labels == 1))

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "priority_score",
        "support_signal",
        "false_alarm_decoy",
        "capacity_band",
        "background_noise",
    ]
    assert any(example["name"] == "False-alarm review" for example in metadata["prediction_examples"])
    assert 0.10 <= positive_rate <= 0.25
    assert int(np.sum((dataset.labels == 0) & (dataset.features[:, 2] > 0.8))) >= 12


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


def test_conformal_coverage_lab_preset_has_both_label_candidate():
    metadata = preset_metadata("Conformal coverage lab")

    assert metadata["input_dim"] == 3
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["score_signal", "calibration_noise", "overlap_band"]
    assert any(example["name"] == "Both-label set candidate" for example in metadata["prediction_examples"])


def test_ood_sentinel_lab_preset_has_leverage_review_example():
    metadata = preset_metadata("OOD sentinel lab")
    dataset = generate_builtin_preset("OOD sentinel lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["main_signal", "support_signal", "leverage_axis", "artifact_code"]
    assert any(example["name"] == "Leverage review row" for example in metadata["prediction_examples"])
    assert float(np.max(np.abs(dataset.features[:, 2]))) > 3.0


def test_bootstrap_stability_lab_preset_has_boundary_band():
    metadata = preset_metadata("Bootstrap stability lab")
    dataset = generate_builtin_preset("Bootstrap stability lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["stable_margin", "support_signal", "boundary_band", "decoy_noise"]
    assert any(example["name"] == "Unstable boundary" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 0.8)) >= 12


def test_prototype_coverage_lab_preset_has_boundary_and_island_markers():
    metadata = preset_metadata("Prototype coverage lab")
    dataset = generate_builtin_preset("Prototype coverage lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["coverage_axis", "class_axis", "boundary_bridge", "island_marker"]
    assert any(example["name"] == "Sparse island review" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 0.7)) >= 12
    assert float(np.max(np.abs(dataset.features[:, 3]))) > 1.5


def test_separability_lens_lab_preset_has_shortcut_and_redundant_features():
    metadata = preset_metadata("Separability lens lab")
    dataset = generate_builtin_preset("Separability lens lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "strong_signal",
        "weak_signal",
        "overlap_noise",
        "shortcut_code",
        "redundant_signal",
    ]
    assert any(example["name"] == "Shortcut conflict" for example in metadata["prediction_examples"])
    shortcut_alignment = np.mean(np.sign(dataset.features[:, 3]) == np.where(dataset.labels == 1, 1.0, -1.0))
    assert float(shortcut_alignment) >= 0.90
    assert float(np.corrcoef(dataset.features[:, 0], dataset.features[:, 4])[0, 1]) > 0.95


def test_neighborhood_hardness_lab_preset_has_boundary_and_hard_islands():
    metadata = preset_metadata("Neighborhood hardness lab")
    dataset = generate_builtin_preset("Neighborhood hardness lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["local_signal", "support_signal", "ambiguity_band", "island_marker"]
    assert any(example["name"] == "Hard island review" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 0.7)) >= 12
    assert float(np.max(np.abs(dataset.features[:, 3]))) > 1.3


def test_dataset_triage_lab_preset_has_conflicts_redundancy_and_tails():
    metadata = preset_metadata("Dataset triage lab")
    dataset = generate_builtin_preset("Dataset triage lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 6
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "real_margin",
        "shortcut_code",
        "redundant_margin",
        "background_noise",
        "constant_code",
        "tail_marker",
    ]
    assert any(example["name"] == "Tail review" for example in metadata["prediction_examples"])
    assert float(np.std(dataset.features[:, 4])) == pytest.approx(0.0)
    assert float(np.corrcoef(dataset.features[:, 0], dataset.features[:, 2])[0, 1]) > 0.95
    assert float(np.max(np.abs(dataset.features[:, 5]))) > 4.0


def test_schema_guard_lab_preset_has_contract_risks():
    metadata = preset_metadata("Schema guard lab")
    dataset = generate_builtin_preset("Schema guard lab", sample_count=160, seed=10)

    assert metadata["input_dim"] == 6
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "continuous_signal",
        "wide_scale_amount",
        "near_constant_sensor",
        "status_code",
        "sparse_indicator",
        "tail_probe",
    ]
    assert any(example["name"] == "Schema warning row" for example in metadata["prediction_examples"])
    assert int(np.unique(dataset.features[:, 3]).shape[0]) <= 3
    assert int(np.sum(dataset.features[:, 4] > 0.5)) >= 4
    assert float(np.max(np.abs(dataset.features[:, 5]))) > 4.0


def test_canary_regression_lab_preset_has_checkable_canaries():
    metadata = preset_metadata("Canary regression lab")
    dataset = generate_builtin_preset("Canary regression lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["stable_margin", "support_signal", "shortcut_marker", "boundary_band"]
    assert any(example["name"] == "Canary shortcut conflict" for example in metadata["prediction_examples"])
    assert sum(example["expected_label"] is not None for example in metadata["prediction_examples"]) >= 3
    assert int(np.sum(np.sign(dataset.features[:, 2]) != np.where(dataset.labels == 1, 1.0, -1.0))) >= 4
    assert int(np.sum(np.abs(dataset.features[:, 3]) < 0.10)) >= 4


def test_monotonic_policy_lab_preset_has_policy_checks():
    metadata = preset_metadata("Monotonic policy lab")
    dataset = generate_builtin_preset("Monotonic policy lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["risk_score", "protective_signal", "exposure_amount", "background_noise"]
    assert len(metadata["policy_checks"]) == 3
    directions = {check["feature_name"]: check["direction"] for check in metadata["policy_checks"]}
    assert directions == {
        "risk_score": "increasing",
        "protective_signal": "decreasing",
        "exposure_amount": "increasing",
    }
    assert any(example["name"] == "Policy boundary review" for example in metadata["prediction_examples"])
    assert float(np.corrcoef(dataset.features[:, 0], dataset.labels)[0, 1]) > 0.45
    assert float(np.corrcoef(dataset.features[:, 1], dataset.labels)[0, 1]) < -0.25


def test_prior_shift_lab_preset_has_base_rate_simulation_risk():
    metadata = preset_metadata("Prior shift lab")
    dataset = generate_builtin_preset("Prior shift lab", sample_count=160, seed=14)

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "screening_score",
        "support_signal",
        "false_positive_shoulder",
        "deployment_mix",
        "background_noise",
    ]
    assert any(example["name"] == "False-positive shoulder review" for example in metadata["prediction_examples"])
    positive_rate = float(np.mean(dataset.labels == 1))
    assert 0.24 <= positive_rate <= 0.36
    false_positive_shoulder = dataset.features[dataset.labels == 0, 2]
    assert float(np.mean(false_positive_shoulder > 0.7)) > 0.10
    assert float(np.corrcoef(dataset.features[:, 0], dataset.labels)[0, 1]) > 0.45


def test_external_holdout_lab_preset_has_shifted_deployment_marker():
    metadata = preset_metadata("External holdout lab")
    dataset = generate_builtin_preset("External holdout lab", sample_count=160, seed=15)

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == [
        "deployment_score",
        "stable_support",
        "shift_marker",
        "prevalence_marker",
        "background_noise",
    ]
    assert any(example["name"] == "Shifted holdout review" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 0.9)) >= 40
    assert 0.35 <= float(np.mean(dataset.labels == 1)) <= 0.60
    assert float(np.corrcoef(dataset.features[:, 0], dataset.labels)[0, 1]) > 0.45


def test_rank_lift_lab_preset_has_concentrated_ranking_signal():
    metadata = preset_metadata("Rank lift lab")
    dataset = generate_builtin_preset("Rank lift lab", sample_count=160, seed=13)

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["rank_score", "support_signal", "decoy_score", "tail_noise", "review_band"]
    assert any(example["name"] == "Decoy high-score review" for example in metadata["prediction_examples"])
    positive_rate = float(np.mean(dataset.labels == 1))
    assert 0.10 <= positive_rate <= 0.20
    top_k = max(1, int(round(dataset.features.shape[0] * 0.10)))
    top_by_rank_score = np.argsort(-dataset.features[:, 0])[:top_k]
    top_positive_rate = float(np.mean(dataset.labels[top_by_rank_score] == 1))
    assert top_positive_rate > positive_rate * 2.0
    assert float(np.mean(dataset.features[dataset.labels == 0, 2])) > 0.15


def test_experiment_advisor_lab_preset_has_imbalanced_nonlinear_boundary():
    metadata = preset_metadata("Experiment advisor lab")
    dataset = generate_builtin_preset("Experiment advisor lab", sample_count=160, seed=11)
    positive_rate = float(np.mean(dataset.labels == 1))

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "rff"
    assert metadata["training_defaults"]["use_smote"] is True
    assert metadata["feature_names"] == ["arc_x", "arc_y", "support_noise", "tail_probe", "boundary_marker"]
    assert any(example["name"] == "Rare outer positive" for example in metadata["prediction_examples"])
    assert 0.20 <= positive_rate <= 0.35
    assert float(np.max(np.abs(dataset.features[:, 3]))) > 3.0


def test_label_audit_traps_preset_has_suspicious_example():
    metadata = preset_metadata("Label audit traps")

    assert metadata["recommended_feature_map"] == "linear"
    assert any(example["name"] == "Suspicious positive-shaped negative" for example in metadata["prediction_examples"])


def test_label_sensitivity_lab_preset_has_planted_review_rows():
    metadata = preset_metadata("Label sensitivity lab")
    dataset = generate_builtin_preset("Label sensitivity lab", sample_count=120, seed=10)
    model = NumpyBinaryClassifier(
        weights=np.asarray([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
        bias=0.0,
    )
    report = run_label_sensitivity(model, dataset.features, dataset.labels, material_f1_delta=0.01, max_items=12)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["model_score", "support_signal", "review_marker", "trusted_anchor"]
    assert any(example["name"] == "Sensitive label review" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 1.0)) >= 4
    assert report["summary"]["priority"] in {"medium", "high"}
    assert report["summary"]["suspect_label_count"] >= 2
    assert report["summary"]["max_improving_f1_delta"] > 0.0


def test_label_noise_stress_lab_preset_has_boundary_stress_rows():
    metadata = preset_metadata("Label noise stress lab")
    dataset = generate_builtin_preset("Label noise stress lab", sample_count=80, seed=7)
    report = run_label_noise_stress(
        dataset.features,
        dataset.labels,
        noise_rates=(0.0, 0.10, 0.25),
        repeats=2,
        max_epochs=12,
        seed=3,
    )

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["clean_margin", "support_signal", "boundary_band", "review_marker"]
    assert any(example["name"] == "Noise-stress boundary" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 0.7)) >= 12
    assert report["summary"]["worst_mean_f1_drop"] > 0.0
    assert report["rates"][0]["noise_rate"] == 0.0


def test_error_atlas_lab_preset_has_asymmetric_error_pockets():
    metadata = preset_metadata("Error atlas lab")
    dataset = generate_builtin_preset("Error atlas lab", sample_count=120, seed=10)

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["margin_score", "support_signal", "false_alarm_pocket", "miss_pocket"]
    assert any(example["name"] == "False-alarm review pocket" for example in metadata["prediction_examples"])
    assert int(np.sum(dataset.features[:, 2] > 1.0)) >= 4
    assert int(np.sum(dataset.features[:, 3] > 1.0)) >= 4


def test_proxy_leakage_lab_preset_supports_ablation_diagnostics():
    metadata = preset_metadata("Proxy leakage lab")

    assert metadata["input_dim"] == 4
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["feature_names"] == ["real_signal", "weak_signal", "proxy_code", "background_noise"]
    assert any(example["name"] == "Proxy conflict" for example in metadata["prediction_examples"])


def test_promotion_gate_lab_preset_has_review_risks():
    metadata = preset_metadata("Promotion gate lab")
    dataset = generate_builtin_preset("Promotion gate lab", sample_count=160, seed=12)
    positive_rate = float(np.mean(dataset.labels == 1))

    assert metadata["input_dim"] == 5
    assert metadata["recommended_feature_map"] == "linear"
    assert metadata["training_defaults"]["trials"] == 16
    assert metadata["feature_names"] == [
        "primary_margin",
        "support_signal",
        "shortcut_proxy",
        "tail_shift",
        "review_band",
    ]
    assert any(example["name"] == "Boundary promotion review" for example in metadata["prediction_examples"])
    assert 0.25 <= positive_rate <= 0.45
    assert float(np.max(np.abs(dataset.features[:, 3]))) > 4.0
    assert int(np.sum(np.abs(dataset.features[:, 4]) < 0.10)) >= 6


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
