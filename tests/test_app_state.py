import json
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch
from italtensor.app import (
    AppState,
    _apply_preset_metadata,
    _format_uncertainty,
    _invalidate_model_artifacts,
    _replace_dataset,
    _store_model_slot,
    _activate_model_slot,
    _build_ensemble,
    _compare_models,
    _run_weight_analysis,
    _handle_worker_done,
    _import_reviewed_labels,
    _export_report,
    _save_preset,
    _run_shap_analysis,
    _run_decision_boundary,
)
from italtensor.data import DataValidationError, validate_dataset
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer
from italtensor.registry import ModelSlot


class FakeElement:
    def __init__(self, values=None):
        self.value = ""
        self.values = values or []

    def update(self, value=None, values=None, append=False, **kwargs):
        if value is not None:
            if append:
                self.value = str(self.value) + str(value)
            else:
                self.value = value
        if values is not None:
            self.values = values

    def get_list_values(self):
        return self.values


class FakeWindow(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AllKeysDict = {}

    def __missing__(self, key):
        self.AllKeysDict[key] = True
        self[key] = FakeElement()
        return self[key]


def test_invalidate_model_artifacts_keeps_dataset_shape_but_clears_model_state():
    state = AppState(
        features=[[1.0]],
        labels=[1],
        input_dim=1,
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 1.0},
        latest_threshold=0.7,
        preprocessor=FeatureStandardizer.identity(1),
        feature_importances=[{"feature_index": 0, "importance": 0.5}],
        trial_history=[{"metrics": {"f1": 1.0}}],
        uncertainty_metadata={"conformal_quantile": 0.3},
        latest_ablation_report={"summary": {"top_feature": "x1"}},
        latest_decision_curve_report={"summary": {"best_threshold": 0.4}},
        latest_conformal_set_report={"summary": {"recommended_alpha": 0.1}},
        latest_calibration_repair_report={"summary": {"recommended_method": "platt"}},
        latest_selective_risk_report={"summary": {"recommended_cutoff": 0.2}},
        latest_sample_review_report={"summary": {"label_issue_count": 1}},
        latest_threshold_report={"summary": {"best_f1": 1.0}},
        latest_model_response_report={"summary": {"top_feature": 0}},
        latest_pairwise_interaction_report={"summary": {"top_pair": [0, 1]}},
        latest_slice_report={"summary": {"worst_f1_delta": -0.5}},
        latest_subgroup_disparity_report={"summary": {"max_disparity": 0.4}},
        latest_stress_report={"summary": {"worst_f1": 0.5}},
        latest_permutation_null_report={"summary": {"verdict": "signal"}},
        latest_population_drift_report={"summary": {"top_feature": 1}},
        latest_adversarial_validation_report={"summary": {"verdict": "strong_multivariate_shift"}},
        latest_chronological_holdout_report={"summary": {"verdict": "severe_temporal_degradation"}},
        latest_cartography_report={"region_counts": {"easy_to_learn": 1}},
        latest_ood_sentinel_report={"summary": {"top_row_index": 3}},
        latest_bootstrap_stability_report={"summary": {"top_row_index": 4}},
        latest_prototype_audit_report={"summary": {"top_boundary_row": 5}},
        latest_feature_separability_report={"summary": {"top_feature": 2}},
        latest_neighborhood_hardness_report={"summary": {"top_hard_row": 6}},
        latest_mps_sweep_report={"recommended_bond_dim": 4},
    )

    _invalidate_model_artifacts(state)

    assert state.features == [[1.0]]
    assert state.labels == [1]
    assert state.input_dim == 1
    assert state.model is None
    assert state.latest_config is None
    assert state.latest_metrics == {}
    assert state.latest_threshold == 0.5
    assert state.preprocessor is None
    assert state.feature_importances == []
    assert state.trial_history == []
    assert state.uncertainty_metadata == {}
    assert state.latest_ablation_report is None
    assert state.latest_decision_curve_report is None
    assert state.latest_conformal_set_report is None
    assert state.latest_calibration_repair_report is None
    assert state.latest_selective_risk_report is None
    assert state.latest_sample_review_report is None
    assert state.latest_threshold_report is None
    assert state.latest_model_response_report is None
    assert state.latest_pairwise_interaction_report is None
    assert state.latest_slice_report is None
    assert state.latest_subgroup_disparity_report is None
    assert state.latest_stress_report is None
    assert state.latest_permutation_null_report is None
    assert state.latest_population_drift_report is None
    assert state.latest_adversarial_validation_report is None
    assert state.latest_chronological_holdout_report is None
    assert state.latest_cartography_report is None
    assert state.latest_ood_sentinel_report is None
    assert state.latest_bootstrap_stability_report is None
    assert state.latest_prototype_audit_report is None
    assert state.latest_feature_separability_report is None
    assert state.latest_neighborhood_hardness_report is None
    assert state.latest_mps_sweep_report is None


def test_replace_dataset_invalidates_old_model_state():
    state = AppState(
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 1.0},
        latest_threshold=0.7,
        preprocessor=FeatureStandardizer.identity(1),
        uncertainty_metadata={"conformal_quantile": 0.3},
    )
    dataset = validate_dataset([[1.0, 2.0], [3.0, 4.0]], [0, 1])

    _replace_dataset(state, dataset)

    assert state.features == [[1.0, 2.0], [3.0, 4.0]]
    assert state.labels == [0, 1]
    assert state.input_dim == 2
    assert state.model is None
    assert state.latest_metrics == {}
    assert state.uncertainty_metadata == {}


def test_format_uncertainty_includes_source_and_coverage():
    summary = _format_uncertainty(
        {
            "conformal_source": "dedicated_calibration",
            "conformal_alpha": 0.1,
            "conformal_target_coverage": 0.9,
            "conformal_coverage": 0.85,
        }
    )

    assert "source=dedicated_calibration" in summary
    assert "conformal_target_coverage=0.9000" in summary
    assert "conformal_coverage=0.8500" in summary


def test_export_report_allows_dataset_only_diagnostics(tmp_path):
    window = FakeWindow()
    state = AppState(
        features=[[0.1, 0.2], [0.8, 0.9]],
        labels=[0, 1],
        input_dim=2,
        latest_feature_separability_report={"summary": {"top_feature": 1}},
        latest_prototype_audit_report={"summary": {"top_boundary_row": 0}},
        latest_neighborhood_hardness_report={"summary": {"top_hard_row": 1}},
    )
    path = tmp_path / "dataset-report.json"

    _export_report(window, state, {"-REPORT_PATH-": str(path)})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"]["available"] is True
    assert payload["model"]["config"] is None
    assert payload["metrics"] == {}
    assert payload["feature_separability"]["summary"]["top_feature"] == 1
    assert payload["prototype_audit"]["summary"]["top_boundary_row"] == 0
    assert payload["neighborhood_hardness"]["summary"]["top_hard_row"] == 1
    assert "Exported report" in window["-LOG-"].value


def test_export_report_rejects_empty_workspace(tmp_path):
    window = FakeWindow()
    state = AppState()

    with pytest.raises(ValueError, match="Load a dataset"):
        _export_report(window, state, {"-REPORT_PATH-": str(tmp_path / "empty.json")})


def test_training_preserves_dataset_only_diagnostics():
    window = FakeWindow()
    state = AppState(
        latest_feature_separability_report={"summary": {"top_feature": 1}},
        latest_prototype_audit_report={"summary": {"top_boundary_row": 0}},
        latest_neighborhood_hardness_report={"summary": {"top_hard_row": 2}},
        busy=True,
    )
    training_result = SimpleNamespace(
        model=object(),
        config=ModelConfig(),
        metrics={"f1": 0.5},
        threshold=0.5,
        preprocessor=None,
        feature_importances=[],
        history={"loss": [1.0]},
        uncertainty={},
    )

    _handle_worker_done(window, state, ("single", training_result))

    assert state.latest_feature_separability_report == {"summary": {"top_feature": 1}}
    assert state.latest_prototype_audit_report == {"summary": {"top_boundary_row": 0}}
    assert state.latest_neighborhood_hardness_report == {"summary": {"top_hard_row": 2}}
    assert state.latest_metrics == {"f1": 0.5}


def test_handle_worker_done_stores_stress_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "base": {"f1": 0.9},
        "summary": {
            "worst_f1": 0.7,
            "stress_f1_ratio": 0.7777,
            "max_label_flip_rate": 0.25,
            "worst_case": "feature_dropout@0.25",
        },
        "perturbations": [
            {
                "kind": "feature_dropout",
                "level": 0.25,
                "f1": 0.7,
                "label_flip_rate": 0.25,
                "mean_probability_shift": 0.12,
            }
        ],
    }

    _handle_worker_done(window, state, ("stress_test", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_stress_report == report
    assert state.busy is False
    assert "Stress suite" in window["-LOG-"].value


def test_handle_worker_done_stores_ablation_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "base": {"f1": 0.9},
        "summary": {
            "top_feature": "x1",
            "max_f1_drop": 0.4,
            "max_label_flip_rate": 0.25,
            "label_proxy_count": 1,
        },
        "features": [
            {
                "feature_index": 0,
                "f1_drop": 0.4,
                "permutation_f1_drop": 0.3,
                "label_flip_rate": 0.25,
                "permutation_label_flip_rate": 0.1,
                "label_correlation": 0.9,
                "risk_flags": ["label_proxy"],
            }
        ],
    }

    _handle_worker_done(window, state, ("ablation_diagnostics", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_ablation_report == report
    assert state.busy is False
    assert "Ablation diagnostics" in window["-LOG-"].value


def test_handle_worker_done_stores_decision_curve_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "best_threshold": 0.4,
            "best_net_benefit": 0.25,
            "max_delta_vs_best_default": 0.1,
            "useful_threshold_ranges": [[0.2, 0.6]],
        },
        "points": [
            {
                "threshold": 0.4,
                "net_benefit_model": 0.25,
                "net_benefit_treat_all": 0.1,
                "delta_vs_best_default": 0.15,
                "best_default_strategy": "treat_all",
            }
        ],
    }

    _handle_worker_done(window, state, ("decision_curve", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_decision_curve_report == report
    assert state.busy is False
    assert "Decision curve" in window["-LOG-"].value


def test_handle_worker_done_stores_selective_risk_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "full_coverage_risk": 0.2,
            "min_selective_risk": 0.0,
            "best_selective_accuracy": 1.0,
            "best_selective_coverage": 0.5,
            "area_under_risk_coverage": 0.1,
        },
        "ranked_cutoffs": [
            {
                "confidence_cutoff": 0.2,
                "coverage": 0.5,
                "error_rate": 0.0,
                "accuracy": 1.0,
                "f1": 1.0,
            }
        ],
    }

    _handle_worker_done(window, state, ("selective_risk", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_selective_risk_report == report
    assert state.busy is False
    assert "Selective risk" in window["-LOG-"].value


def test_handle_worker_done_stores_conformal_sets_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "recommended_alpha": 0.1,
            "recommended_target_coverage": 0.9,
            "recommended_empirical_coverage": 1.0,
            "recommended_mean_set_size": 1.25,
            "recommended_singleton_rate": 0.75,
            "recommended_ambiguous_rate": 0.25,
        },
        "points": [
            {
                "alpha": 0.1,
                "target_coverage": 0.9,
                "empirical_coverage": 1.0,
                "mean_set_size": 1.25,
                "singleton_accuracy": 1.0,
            }
        ],
    }

    _handle_worker_done(window, state, ("conformal_sets", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_conformal_set_report == report
    assert state.busy is False
    assert "Conformal sets" in window["-LOG-"].value


def test_handle_worker_done_stores_calibration_repair_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "recommended_method": "platt",
            "recommended_brier_score": 0.1,
            "recommended_ece": 0.05,
            "best_brier_improvement": 0.08,
            "best_ece_improvement": 0.03,
        },
        "methods": [
            {
                "method": "raw",
                "brier_score": 0.18,
                "ece": 0.08,
                "log_loss": 0.5,
                "brier_improvement": 0.0,
            },
            {
                "method": "platt",
                "brier_score": 0.1,
                "ece": 0.05,
                "log_loss": 0.4,
                "brier_improvement": 0.08,
            },
        ],
    }

    _handle_worker_done(window, state, ("calibration_repair", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_calibration_repair_report == report
    assert state.busy is False
    assert "Calibration repair" in window["-LOG-"].value


def test_handle_worker_done_stores_permutation_null_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "observed_f1": 0.9,
            "null_mean_f1": 0.45,
            "f1_gap": 0.45,
            "f1_p_value": 0.01,
            "verdict": "strong_signal",
        },
        "observed": {"f1": 0.9, "accuracy": 0.85, "balanced_accuracy": 0.84},
        "p_values": {"f1": 0.01, "accuracy": 0.02, "balanced_accuracy": 0.03},
        "null_distribution": {
            "f1": {"mean": 0.45, "p95": 0.7},
            "accuracy": {"mean": 0.5, "p95": 0.75},
            "balanced_accuracy": {"mean": 0.5, "p95": 0.74},
        },
    }

    _handle_worker_done(window, state, ("permutation_null", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_permutation_null_report == report
    assert state.busy is False
    assert "Permutation null" in window["-LOG-"].value


def test_handle_worker_done_stores_population_drift_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "top_feature": 1,
            "max_psi": 0.4,
            "max_ks_statistic": 0.5,
            "max_mean_shift_std": 1.2,
            "label_prevalence_shift": 0.2,
            "drifted_feature_count": 1,
        },
        "features": [
            {
                "feature_index": 1,
                "psi": 0.4,
                "ks_statistic": 0.5,
                "mean_shift_std": 1.2,
                "outside_reference_rate": 0.25,
                "risk_flags": ["major_psi_shift"],
            }
        ],
    }

    _handle_worker_done(window, state, ("population_drift", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_population_drift_report == report
    assert state.busy is False
    assert "Population drift" in window["-LOG-"].value


def test_handle_worker_done_stores_adversarial_validation_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "domain_auc": 0.88,
            "domain_accuracy": 0.8,
            "detectability": 0.88,
            "top_feature": 2,
            "verdict": "strong_multivariate_shift",
        },
        "features": [
            {
                "feature_index": 2,
                "auc_drop": 0.2,
                "accuracy_drop": 0.15,
                "mean_probability_shift": 0.12,
                "risk_flags": ["domain_auc_driver"],
            }
        ],
    }

    _handle_worker_done(window, state, ("adversarial_validation", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_adversarial_validation_report == report
    assert state.busy is False
    assert "Adversarial validation" in window["-LOG-"].value


def test_handle_worker_done_stores_chronological_holdout_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "reference_f1": 0.9,
            "current_f1": 0.55,
            "f1_delta": -0.35,
            "top_current_reliance_feature": 1,
            "verdict": "severe_temporal_degradation",
            "warning": "current F1 is lower than reference validation",
        },
        "metric_deltas": {"f1_delta": -0.35, "accuracy_delta": -0.25, "brier_score_delta": 0.12},
        "current_baseline": {
            "available": True,
            "current_model_metrics": {"f1": 0.75},
            "metric_deltas_vs_reference_model": {"f1_delta": 0.2},
        },
        "permutation_reliance": [
            {
                "feature_index": 1,
                "f1_drop": 0.2,
                "log_loss_increase": 0.15,
                "mean_probability_shift": 0.12,
                "risk_flags": ["current_f1_driver"],
            }
        ],
    }

    _handle_worker_done(window, state, ("chronological_holdout", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_chronological_holdout_report == report
    assert state.busy is False
    assert "Chronological holdout" in window["-LOG-"].value


def test_handle_worker_done_stores_ood_sentinel_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "model_used": True,
        "sample_count": 4,
        "summary": {
            "top_row_index": 2,
            "max_ood_score": 3.5,
            "max_abs_robust_z": 4.0,
            "max_nearest_neighbor_distance": 3.0,
            "flagged_row_count": 1,
        },
        "rows": [
            {
                "row_index": 2,
                "ood_score": 3.5,
                "max_abs_robust_z": 4.0,
                "nearest_neighbor_distance": 3.0,
                "loss": 1.2,
                "probability": 0.9,
                "risk_flags": ["robust_outlier"],
            }
        ],
    }

    _handle_worker_done(window, state, ("ood_sentinel", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_ood_sentinel_report == report
    assert state.busy is False
    assert "OOD sentinel" in window["-LOG-"].value


def test_handle_worker_done_stores_bootstrap_stability_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "model_count": 8,
        "feature_map": "linear",
        "threshold": 0.4,
        "ensemble_metrics": {"f1": 0.8, "accuracy": 0.75, "brier_score": 0.2},
        "summary": {
            "top_row_index": 3,
            "max_probability_std": 0.22,
            "max_disagreement_rate": 0.5,
            "unstable_row_count": 1,
        },
        "rows": [
            {
                "row_index": 3,
                "instability_score": 0.7,
                "probability_std": 0.22,
                "disagreement_rate": 0.5,
                "mean_probability": 0.48,
                "risk_flags": ["committee_disagreement"],
            }
        ],
    }

    _handle_worker_done(window, state, ("bootstrap_stability", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_bootstrap_stability_report == report
    assert state.busy is False
    assert "Bootstrap stability" in window["-LOG-"].value


def test_handle_worker_done_stores_prototype_audit_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "k": 3,
        "summary": {
            "prototype_count": 2,
            "boundary_row_count": 1,
            "isolated_row_count": 1,
            "label_contradiction_count": 1,
            "top_boundary_row": 3,
            "top_label_contradiction_row": 3,
        },
        "prototypes": [
            {
                "row_index": 0,
                "label": 0,
                "prototype_score": 0.8,
                "local_opposite_fraction": 0.0,
                "risk_flags": ["class_prototype"],
            }
        ],
        "boundary_rows": [
            {
                "row_index": 3,
                "label": 1,
                "boundary_score": 0.6,
                "local_opposite_fraction": 0.67,
                "risk_flags": ["class_boundary"],
            }
        ],
    }

    _handle_worker_done(window, state, ("prototype_audit", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_prototype_audit_report == report
    assert state.busy is False
    assert "Prototype audit" in window["-LOG-"].value


def test_handle_worker_done_stores_feature_separability_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "top_feature": 1,
            "top_auc": 0.95,
            "top_balanced_accuracy": 0.9,
            "near_perfect_feature_count": 1,
            "weak_feature_count": 1,
            "redundant_pair_count": 1,
        },
        "features": [
            {
                "feature_index": 1,
                "auc": 0.95,
                "best_balanced_accuracy": 0.9,
                "standardized_mean_difference": 2.4,
                "risk_flags": ["strong_single_feature"],
            }
        ],
        "redundant_pairs": [
            {
                "left_feature_index": 0,
                "right_feature_index": 2,
                "correlation": 0.98,
                "risk_flags": ["redundant_features"],
            }
        ],
    }

    _handle_worker_done(window, state, ("feature_separability", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_feature_separability_report == report
    assert state.busy is False
    assert "Feature separability" in window["-LOG-"].value


def test_handle_worker_done_stores_neighborhood_hardness_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "k": 3,
        "summary": {
            "loo_accuracy": 0.75,
            "hard_row_count": 2,
            "ambiguous_row_count": 1,
            "label_issue_candidate_count": 1,
            "top_hard_row": 4,
        },
        "rows": [
            {
                "row_index": 4,
                "label": 0,
                "predicted_label": 1,
                "hardness_score": 0.9,
                "opposite_vote_rate": 1.0,
                "risk_flags": ["label_issue_candidate", "hard_row"],
            }
        ],
    }

    _handle_worker_done(window, state, ("neighborhood_hardness", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_neighborhood_hardness_report == report
    assert state.busy is False
    assert "Neighborhood hardness" in window["-LOG-"].value


def test_handle_worker_done_stores_slice_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "base": {"f1": 0.9},
        "summary": {
            "slice_count": 1,
            "worst_slice": "x2[0, 1]",
            "worst_f1_delta": -0.4,
            "worst_accuracy_delta": -0.25,
        },
        "slices": [
            {
                "feature_index": 1,
                "left": 0.0,
                "right": 1.0,
                "count": 4,
                "f1": 0.5,
                "accuracy": 0.5,
                "f1_delta": -0.4,
            }
        ],
    }

    _handle_worker_done(window, state, ("slice_diagnostics", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_slice_report == report
    assert state.busy is False
    assert "Slice diagnostics" in window["-LOG-"].value


def test_handle_worker_done_stores_model_response_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "top_feature": 0,
            "top_response_range": 0.5,
            "top_direction": "increasing",
            "nonmonotonic_feature_count": 1,
            "high_impact_feature_count": 2,
        },
        "features": [
            {
                "feature_index": 0,
                "response_range": 0.5,
                "signed_change": 0.5,
                "direction": "increasing",
                "risk_flags": ["high_impact"],
            }
        ],
    }

    _handle_worker_done(window, state, ("model_response", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_model_response_report == report
    assert state.busy is False
    assert "Model response" in window["-LOG-"].value


def test_handle_worker_done_stores_pairwise_interactions_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "evaluated_pair_count": 1,
            "top_pair": [0, 1],
            "top_interaction_strength": 0.5,
            "top_max_abs_interaction": 0.2,
            "strong_pair_count": 1,
        },
        "pairs": [
            {
                "feature_i": 0,
                "feature_j": 1,
                "interaction_strength": 0.5,
                "max_abs_interaction": 0.2,
                "threshold_crossings": 2,
                "risk_flags": ["strong_interaction"],
            }
        ],
    }

    _handle_worker_done(window, state, ("pairwise_interactions", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_pairwise_interaction_report == report
    assert state.busy is False
    assert "Pairwise interactions" in window["-LOG-"].value


def test_handle_worker_done_stores_subgroup_disparity_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "evaluated_feature_count": 1,
            "evaluated_subgroup_count": 2,
            "worst_feature": 1,
            "worst_metric": "false_negative_rate_gap",
            "max_disparity": 0.6,
        },
        "subgroups": [
            {
                "label": "x2=1",
                "count": 4,
                "risk_score": 0.6,
                "worst_metric": "false_negative_rate_gap",
                "risk_flags": ["fnr_gap"],
            }
        ],
    }

    _handle_worker_done(window, state, ("subgroup_disparity", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_subgroup_disparity_report == report
    assert state.busy is False
    assert "Subgroup disparity" in window["-LOG-"].value


def test_handle_worker_done_stores_threshold_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "current_threshold": 0.4,
        "summary": {
            "best_f1_threshold": 0.3,
            "min_cost_threshold": 0.2,
            "best_f1": 0.95,
            "min_cost": 0.1,
            "current_cost": 0.2,
        },
        "best_f1": {"threshold": 0.3, "f1": 0.95, "precision": 0.9, "recall": 1.0, "cost": 0.15},
        "best_balanced_accuracy": {"threshold": 0.35, "f1": 0.9, "precision": 0.9, "recall": 0.9, "cost": 0.2},
        "min_cost": {"threshold": 0.2, "f1": 0.85, "precision": 0.8, "recall": 1.0, "cost": 0.1},
    }

    _handle_worker_done(window, state, ("threshold_diagnostics", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_threshold_report == report
    assert state.latest_threshold == 0.4
    assert state.busy is False
    assert "Threshold sweep" in window["-LOG-"].value


def test_handle_worker_done_stores_sample_review_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "label_issue_count": 1,
            "disagreement_count": 2,
            "ambiguous_count": 1,
            "mean_loss": 0.3,
            "max_loss": 1.2,
        },
        "label_issues": [
            {"row_index": 2, "label": 0, "predicted_label": 1, "probability": 0.95, "loss": 2.9}
        ],
        "hard_examples": [],
        "ambiguous_examples": [],
    }

    _handle_worker_done(window, state, ("sample_review", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_sample_review_report == report
    assert state.latest_threshold == 0.4
    assert state.busy is False
    assert "Sample review" in window["-LOG-"].value


def test_import_reviewed_labels_appends_rows_and_invalidates_model(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "\n".join(
            [
                "x1,x2,italtensor_probability,italtensor_label,italtensor_review_label",
                "0.2,0.3,0.4,0,1",
                "0.4,0.5,0.6,1,",
                "0.6,0.7,0.8,1,0",
            ]
        ),
        encoding="utf-8",
    )
    window = FakeWindow()
    state = AppState(
        features=[[0.0, 0.0], [1.0, 1.0]],
        labels=[0, 1],
        input_dim=2,
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 0.8},
    )

    _import_reviewed_labels(window, state, {"-BATCH_OUTPUT_PATH-": str(path)})

    assert len(state.labels) == 4
    assert state.labels[-2:] == [1, 0]
    np.testing.assert_allclose(state.features[-2:], [[0.2, 0.3], [0.6, 0.7]])
    assert state.model is None
    assert state.latest_metrics == {}
    assert "Imported 2 reviewed label" in window["-LOG-"].value


def test_import_reviewed_labels_with_no_reviewed_rows_does_not_mutate(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text("x1,italtensor_probability,italtensor_review_label\n0.2,0.4,\n", encoding="utf-8")
    window = FakeWindow()
    state = AppState(features=[[0.0]], labels=[0], input_dim=1)

    with pytest.raises(DataValidationError):
        _import_reviewed_labels(window, state, {"-BATCH_OUTPUT_PATH-": str(path)})

    assert state.features == [[0.0]]
    assert state.labels == [0]


def test_apply_preset_metadata_updates_sparse_training_defaults():
    window = FakeWindow()

    _apply_preset_metadata(
        window,
        {
            "recommended_feature_map": "quadratic",
            "training_defaults": {
                "epochs": 90,
                "batch_size": 16,
                "trials": 16,
                "feature_map": "quadratic",
                "l1_penalty": 0.001,
                "feature_selection_k": 6,
            },
        },
    )

    assert window["-FEATURE_MAP-"].value == "quadratic"
    assert window["-L1_PENALTY-"].value == "0.001"
    assert window["-FEATURE_K-"].value == "6"


def test_apply_preset_metadata_updates_extended_training_defaults():
    window = FakeWindow()

    _apply_preset_metadata(
        window,
        {
            "recommended_feature_map": "rff",
            "training_defaults": {
                "epochs": 55,
                "batch_size": 12,
                "trials": 9,
                "feature_map": "quadratic",
                "backend": "mps",
                "lr_schedule": "cosine",
                "gradient_clip": 0.5,
                "mps_bond_dim": 6,
                "mps_physical_dim": 5,
            },
        },
    )

    assert window["-BACKEND-"].value == "mps"
    assert window["-LR_SCHEDULE-"].value == "cosine"
    assert window["-GRADIENT_CLIP-"].value == "0.5"
    assert window["-MPS_BOND-"].value == "6"
    assert window["-MPS_PHYS-"].value == "5"
    assert window["-FEATURE_MAP-"].value == "rff"


def test_save_preset_uses_current_gui_training_defaults_and_prediction_example(tmp_path):
    window = FakeWindow()
    state = AppState(features=[[0.1, 0.2], [0.8, 0.9]], labels=[0, 1], input_dim=2)
    path = tmp_path / "custom-preset.json"
    values = {
        "-PRESET_PATH-": str(path),
        "-PRESET_SAVE_NAME-": "Custom useful preset",
        "-PRESET_DESCRIPTION-": "Keeps GUI defaults",
        "-EPOCHS-": "77",
        "-BATCH_SIZE-": "11",
        "-TRIALS-": "5",
        "-FEATURE_MAP-": "quadratic",
        "-BACKEND-": "numpy",
        "-LR_SCHEDULE-": "cosine",
        "-GRADIENT_CLIP-": "0.25",
        "-L1_PENALTY-": "0.002",
        "-FEATURE_K-": "2",
        "-MPS_BOND-": "7",
        "-MPS_PHYS-": "3",
        "-PREDICTION_VECTOR-": "[0.3, 0.4]",
    }

    _save_preset(window, state, values)

    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload["training_defaults"]
    assert defaults["epochs"] == 77
    assert defaults["batch_size"] == 11
    assert defaults["trials"] == 5
    assert defaults["feature_map"] == "quadratic"
    assert defaults["backend"] == "numpy"
    assert defaults["lr_schedule"] == "cosine"
    assert defaults["gradient_clip"] == 0.25
    assert defaults["l1_penalty"] == 0.002
    assert defaults["feature_selection_k"] == 2
    assert defaults["mps_bond_dim"] == 7
    assert defaults["mps_physical_dim"] == 3
    assert payload["recommended_feature_map"] == "quadratic"
    assert payload["prediction_examples"][0]["features"] == [0.3, 0.4]


def test_store_model_slot():
    window = FakeWindow()
    state = AppState()
    state.model = object()
    state.latest_config = ModelConfig()
    state.latest_metrics = {"f1": 0.8}
    state.latest_threshold = 0.5

    with patch("PySimpleGUI.popup_get_text", return_value="Model A"):
        _store_model_slot(window, state, {})

    assert len(state.model_slots) == 1
    assert state.model_slots[0].name == "Model A"
    assert state.model_slots[0].metrics["f1"] == 0.8
    assert state.active_slot_index == 0


def test_activate_model_slot():
    window = FakeWindow()
    state = AppState()
    slot = ModelSlot(
        model=object(),
        config=ModelConfig(learning_rate=0.05),
        metrics={"f1": 0.9},
        preprocessor=None,
        threshold=0.6,
        name="Model B",
    )
    state.model_slots.append(slot)
    state.active_slot_index = 0

    window["-MODEL_SLOTS-"].values = ["* Model B (F1: 0.9000)"]

    _activate_model_slot(window, state, {"-MODEL_SLOTS-": ["* Model B (F1: 0.9000)"]})

    assert state.active_slot_index == 0
    assert state.latest_config.learning_rate == 0.05
    assert state.latest_threshold == 0.6


def test_build_ensemble():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier
    from italtensor.preprocessing import FeatureStandardizer

    model_a = NumpyBinaryClassifier(weights=np.array([1.0]), bias=0.0, raw_input_dim=1)
    slot = ModelSlot(
        model=model_a,
        config=ModelConfig(),
        metrics={"f1": 0.8},
        preprocessor=FeatureStandardizer.identity(1),
        threshold=0.5,
        name="Model A",
    )
    state.model_slots.append(slot)
    state.features = [[1.0], [2.0], [3.0], [4.0]]
    state.labels = [0, 1, 0, 1]

    _build_ensemble(window, state, {})

    assert len(state.model_slots) == 2
    assert "Ensemble" in state.model_slots[1].name


def test_compare_models():
    window = FakeWindow()
    state = AppState()
    slot = ModelSlot(
        model=object(),
        config=ModelConfig(),
        metrics={"f1": 0.8, "accuracy": 0.85, "brier_score": 0.1, "ece": 0.05},
        preprocessor=None,
        threshold=0.5,
        name="Model A",
    )
    state.model_slots.append(slot)

    _compare_models(window, state, {})
    assert "Model A" in window["-LOG-"].value


def test_run_weight_analysis():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier

    state.model = NumpyBinaryClassifier(weights=np.array([1.0, 0.0]), bias=0.5, raw_input_dim=2)

    _run_weight_analysis(window, state, {})
    assert "Weight Analysis" in window["-LOG-"].value
    assert "Sparsity" in window["-LOG-"].value


def test_run_shap_analysis():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier
    from italtensor.preprocessing import FeatureStandardizer

    # 1. Test fallback when model is None
    _run_shap_analysis(window, state, {})
    assert "Train or load a model first" in window["-LOG-"].value

    # 2. Test when prediction vector is missing
    state.model = NumpyBinaryClassifier(weights=np.array([1.0, 0.0]), bias=0.5, raw_input_dim=2)
    window["-LOG-"].value = ""
    _run_shap_analysis(window, state, {})
    assert "Please enter a prediction vector JSON" in window["-LOG-"].value

    # 3. Test successful SHAP run
    window["-LOG-"].value = ""
    state.preprocessor = FeatureStandardizer(mean=np.array([0.0, 0.0]), scale=np.array([1.0, 1.0]), selected_indices=[0, 1])
    values = {"-PREDICTION_VECTOR-": "[1.0, -1.0]"}
    _run_shap_analysis(window, state, values)
    assert "SHAP Local Feature Attributions" in window["-LOG-"].value
    assert "x1" in window["-LOG-"].value


def test_run_decision_boundary():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier
    from italtensor.preprocessing import FeatureStandardizer

    # 1. Test fallback when model is None
    _run_decision_boundary(window, state, {})
    assert "Train or load a model first" in window["-LOG-"].value

    # 2. Test when dataset is missing
    state.model = NumpyBinaryClassifier(weights=np.array([1.0, 0.0]), bias=0.5, raw_input_dim=2)
    window["-LOG-"].value = ""
    _run_decision_boundary(window, state, {})
    assert "No dataset loaded" in window["-LOG-"].value

    # 3. Test successful decision boundary run
    window["-LOG-"].value = ""
    state.features = [[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]]
    state.labels = [0, 1, 0, 1]
    state.preprocessor = FeatureStandardizer(mean=np.array([0.0, 0.0]), scale=np.array([1.0, 1.0]), selected_indices=[0, 1])
    
    _run_decision_boundary(window, state, {})
    assert "Decision Boundary Visualization" in window["-LOG-"].value
