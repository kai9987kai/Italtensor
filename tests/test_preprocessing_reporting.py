import json

import numpy as np
import pytest

from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer
from italtensor.reporting import build_experiment_report, export_experiment_report


def test_standardizer_fits_training_statistics_only():
    train_features = np.asarray([[1.0, 10.0], [3.0, 10.0]], dtype=np.float32)
    validation_features = np.asarray([[101.0, 10.0]], dtype=np.float32)

    standardizer = FeatureStandardizer.fit(train_features)
    transformed = standardizer.transform(validation_features)

    assert standardizer.mean.tolist() == pytest.approx([2.0, 10.0])
    assert standardizer.scale.tolist() == pytest.approx([1.0, 1.0])
    np.testing.assert_allclose(transformed, np.asarray([[99.0, 0.0]], dtype=np.float32))


def test_standardizer_metadata_round_trip():
    standardizer = FeatureStandardizer.fit(np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32))

    restored = FeatureStandardizer.from_dict(json.loads(json.dumps(standardizer.to_dict())))

    assert restored is not None
    assert restored.mean.tolist() == pytest.approx(standardizer.mean.tolist())
    assert restored.scale.tolist() == pytest.approx(standardizer.scale.tolist())


def test_standardizer_metadata_validates_input_dimension():
    standardizer = FeatureStandardizer.identity(2)

    with pytest.raises(ValueError, match="model expects 3"):
        FeatureStandardizer.from_dict(standardizer.to_dict(), input_dim=3)


def test_report_export_json_and_markdown(tmp_path):
    report = build_experiment_report(
        sample_count=4,
        input_dim=2,
        labels=[0, 0, 1, 1],
        features=[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        config=ModelConfig(hidden_layers=(16,), max_epochs=3),
        metrics={"f1": 0.75, "threshold": 0.4},
        threshold=0.4,
        preprocessor=FeatureStandardizer.identity(2),
        feature_importances=[{"feature_index": 0, "importance": 0.25}],
        trial_history=[
            {
                "config": {"feature_map": "rff"},
                "metrics": {"f1": 0.75, "brier_score": 0.1, "log_loss": 0.3},
            }
        ],
        uncertainty_metadata={
            "conformal_source": "dedicated_calibration",
            "conformal_alpha": 0.1,
            "conformal_quantile": 0.35,
            "conformal_target_coverage": 0.9,
            "conformal_coverage": 1.0,
            "conformal_calibration_count": 8,
            "conformal_evaluation_count": 8,
            "conformal_singleton_rate": 0.75,
        },
        ablation_report={
            "base": {"f1": 0.75},
            "summary": {
                "top_feature": "x1",
                "max_f1_drop": 0.25,
                "max_label_flip_rate": 0.2,
                "high_reliance_count": 1,
                "label_proxy_count": 1,
            },
            "features": [
                {
                    "feature_index": 0,
                    "f1_drop": 0.25,
                    "permutation_f1_drop": 0.2,
                    "label_flip_rate": 0.2,
                    "permutation_label_flip_rate": 0.1,
                    "label_correlation": 0.9,
                    "risk_flags": ["label_proxy"],
                }
            ],
        },
        sample_review_report={
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
        },
        error_atlas_report={
            "sample_count": 4,
            "summary": {
                "error_count": 2,
                "error_rate": 0.5,
                "high_confidence_error_count": 1,
                "near_threshold_count": 1,
                "dominant_error_type": "balanced_errors",
                "recommendation": "Review high-confidence errors first.",
            },
            "confusion": {"false_positive": 1, "false_negative": 1},
            "high_confidence_errors": [
                {"row_index": 2, "label": 0, "predicted_label": 1, "probability": 0.95, "loss": 2.9, "margin": 0.55}
            ],
            "near_threshold_rows": [
                {"row_index": 1, "label": 1, "predicted_label": 1, "probability": 0.51, "loss": 0.67, "margin": 0.01}
            ],
            "feature_error_shifts": [
                {"feature_index": 0, "standardized_shift": 1.2, "error_mean": 1.0, "correct_mean": 0.0}
            ],
        },
        reliability_atlas_report={
            "summary": {
                "risk_level": "medium",
                "brier_score": 0.18,
                "log_loss": 0.5,
                "expected_calibration_error": 0.09,
                "max_calibration_error": 0.25,
                "bin_count": 3,
                "sparse_bin_count": 1,
                "recommendation": "Run Calibration repair.",
            },
            "worst_bins": [
                {
                    "left": 0.8,
                    "right": 1.0,
                    "count": 4,
                    "confidence": 0.9,
                    "accuracy": 0.5,
                    "absolute_error": 0.4,
                    "calibration_direction": "overconfident",
                }
            ],
        },
        threshold_report={
            "current_threshold": 0.4,
            "summary": {
                "best_f1_threshold": 0.3,
                "best_balanced_accuracy_threshold": 0.35,
                "min_cost_threshold": 0.25,
                "current_cost": 0.5,
                "min_cost": 0.25,
            },
            "best_f1": {"threshold": 0.3, "f1": 0.8, "precision": 0.75, "recall": 0.85, "cost": 0.3},
            "best_balanced_accuracy": {"threshold": 0.35, "f1": 0.75, "precision": 0.7, "recall": 0.8, "cost": 0.4},
            "min_cost": {"threshold": 0.25, "f1": 0.7, "precision": 0.65, "recall": 0.9, "cost": 0.25},
        },
        decision_curve_report={
            "prevalence": 0.5,
            "summary": {
                "best_threshold": 0.4,
                "best_net_benefit": 0.25,
                "max_delta_vs_best_default": 0.2,
                "useful_threshold_ranges": [[0.2, 0.6]],
                "warning": None,
            },
            "current": {"threshold": 0.4, "net_benefit_model": 0.25, "delta_vs_best_default": 0.2},
            "points": [
                {
                    "threshold": 0.4,
                    "net_benefit_model": 0.25,
                    "net_benefit_treat_all": 0.1,
                    "net_benefit_treat_none": 0.0,
                    "delta_vs_best_default": 0.15,
                }
            ],
        },
        conformal_set_report={
            "split": {
                "source": "posthoc_stratified_split",
                "calibration_count": 4,
                "evaluation_count": 4,
            },
            "summary": {
                "recommended_alpha": 0.1,
                "recommended_target_coverage": 0.9,
                "recommended_empirical_coverage": 1.0,
                "recommended_mean_set_size": 1.25,
                "recommended_singleton_rate": 0.75,
                "recommended_ambiguous_rate": 0.25,
                "warning": None,
            },
            "points": [
                {
                    "alpha": 0.1,
                    "target_coverage": 0.9,
                    "empirical_coverage": 1.0,
                    "coverage_gap": 0.1,
                    "mean_set_size": 1.25,
                    "singleton_accuracy": 1.0,
                }
            ],
        },
        calibration_repair_report={
            "split": {
                "source": "posthoc_stratified_split",
                "calibration_count": 4,
                "evaluation_count": 4,
            },
            "summary": {
                "recommended_method": "platt",
                "recommended_brier_score": 0.1,
                "recommended_ece": 0.05,
                "recommended_log_loss": 0.4,
                "best_brier_improvement": 0.08,
                "best_ece_improvement": 0.03,
                "warning": None,
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
        },
        permutation_null_report={
            "permutation_count": 80,
            "seed": 42,
            "summary": {
                "observed_f1": 0.9,
                "null_mean_f1": 0.45,
                "f1_gap": 0.45,
                "f1_z_score": 3.1,
                "f1_p_value": 0.01,
                "accuracy_p_value": 0.02,
                "verdict": "strong_signal",
                "warning": None,
            },
            "observed": {"f1": 0.9, "accuracy": 0.85, "balanced_accuracy": 0.84},
            "p_values": {"f1": 0.01, "accuracy": 0.02, "balanced_accuracy": 0.03},
            "null_distribution": {
                "f1": {"mean": 0.45, "p95": 0.7},
                "accuracy": {"mean": 0.5, "p95": 0.75},
                "balanced_accuracy": {"mean": 0.5, "p95": 0.74},
            },
        },
        population_drift_report={
            "split_source": "row_order_first_reference_then_current",
            "reference_count": 2,
            "current_count": 2,
            "summary": {
                "top_feature": 1,
                "max_psi": 0.4,
                "max_ks_statistic": 0.5,
                "max_mean_shift_std": 1.2,
                "max_outside_reference_rate": 0.25,
                "drifted_feature_count": 1,
                "warning": None,
            },
            "label_shift": {"prevalence_shift": 0.25},
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
        },
        adversarial_validation_report={
            "split_source": "row_order_domain_classifier",
            "reference_count": 2,
            "current_count": 2,
            "validation_samples": 2,
            "summary": {
                "domain_auc": 0.88,
                "domain_accuracy": 0.8,
                "detectability": 0.88,
                "top_feature": 1,
                "important_feature_count": 1,
                "verdict": "strong_multivariate_shift",
                "warning": None,
            },
            "domain_metrics": {"roc_auc": 0.88, "accuracy": 0.8},
            "label_shift": {"prevalence_shift": 0.25},
            "features": [
                {
                    "feature_index": 1,
                    "auc_drop": 0.2,
                    "accuracy_drop": 0.15,
                    "mean_probability_shift": 0.12,
                    "risk_flags": ["domain_auc_driver"],
                }
            ],
        },
        chronological_holdout_report={
            "split_source": "row_order_reference_then_current",
            "reference_count": 3,
            "reference_evaluation_count": 1,
            "current_count": 2,
            "feature_map": "linear",
            "reference_metrics": {"f1": 0.9, "accuracy": 0.9},
            "current_metrics": {"f1": 0.5, "accuracy": 0.6},
            "metric_deltas": {"f1_delta": -0.4, "accuracy_delta": -0.3, "brier_score_delta": 0.15, "log_loss_delta": 0.4},
            "current_probability_diagnostics": {"mean_probability_delta": 0.2},
            "label_shift": {"prevalence_shift": 0.25},
            "summary": {
                "top_current_reliance_feature": 1,
                "current_baseline_f1_gain": 0.2,
                "verdict": "severe_temporal_degradation_current_relearns",
                "warning": None,
            },
            "current_baseline": {
                "available": True,
                "current_train_count": 4,
                "current_evaluation_count": 2,
                "current_model_metrics": {"f1": 0.7},
                "metric_deltas_vs_reference_model": {"f1_delta": 0.2},
            },
            "permutation_reliance": [
                {
                    "feature_index": 1,
                    "f1_drop": 0.2,
                    "log_loss_increase": 0.1,
                    "mean_probability_shift": 0.15,
                    "risk_flags": ["current_f1_driver"],
                }
            ],
        },
        selective_risk_report={
            "base": {"error_rate": 0.5},
            "summary": {
                "min_selective_risk": 0.0,
                "recommended_cutoff": 0.2,
                "best_selective_accuracy": 1.0,
                "best_selective_coverage": 0.5,
                "max_error_reduction": 0.5,
                "coverage_at_10pct_risk": 0.5,
                "area_under_risk_coverage": 0.1,
                "warning": None,
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
        },
        model_response_report={
            "summary": {
                "top_feature": 0,
                "top_response_range": 0.5,
                "top_direction": "increasing",
                "nonmonotonic_feature_count": 1,
                "high_impact_feature_count": 2,
                "warning": None,
            },
            "features": [
                {
                    "feature_index": 0,
                    "response_range": 0.5,
                    "signed_change": 0.45,
                    "direction": "increasing",
                    "min_response_value": -1.0,
                    "max_response_value": 1.0,
                    "risk_flags": ["high_impact"],
                }
            ],
        },
        pairwise_interaction_report={
            "summary": {
                "evaluated_pair_count": 1,
                "top_pair": [0, 1],
                "top_interaction_strength": 0.55,
                "top_max_abs_interaction": 0.22,
                "strong_pair_count": 1,
                "threshold_crossing_pair_count": 1,
                "warning": None,
            },
            "pairs": [
                {
                    "feature_i": 0,
                    "feature_j": 1,
                    "interaction_strength": 0.55,
                    "max_abs_interaction": 0.22,
                    "mean_abs_interaction": 0.1,
                    "threshold_crossings": 2,
                    "risk_flags": ["strong_interaction"],
                }
            ],
        },
        slice_report={
            "base": {"f1": 0.75},
            "summary": {
                "slice_count": 1,
                "worst_slice": "x1[0, 1]",
                "worst_f1_delta": -0.25,
                "worst_accuracy_delta": -0.25,
            },
            "slices": [
                {
                    "feature_index": 0,
                    "left": 0.0,
                    "right": 1.0,
                    "count": 2,
                    "f1": 0.5,
                    "f1_delta": -0.25,
                }
            ],
        },
        subgroup_disparity_report={
            "summary": {
                "evaluated_feature_count": 1,
                "evaluated_subgroup_count": 2,
                "worst_feature": 1,
                "worst_subgroup": "x2=1",
                "worst_metric": "false_negative_rate_gap",
                "max_disparity": 0.6,
                "max_false_negative_rate_gap": 0.6,
                "max_false_positive_rate_gap": 0.2,
                "max_predicted_positive_rate_gap": 0.3,
                "warning": "Numeric feature slices are proxy subgroup diagnostics.",
            },
            "subgroups": [
                {
                    "label": "x2=1",
                    "count": 4,
                    "coverage": 0.5,
                    "risk_score": 0.6,
                    "worst_metric": "false_negative_rate_gap",
                    "risk_flags": ["fnr_gap"],
                }
            ],
        },
        stress_report={
            "base": {"f1": 0.75},
            "summary": {
                "worst_f1": 0.5,
                "stress_f1_ratio": 0.6667,
                "max_label_flip_rate": 0.25,
                "worst_case": "feature_dropout@0.25",
            },
            "perturbations": [
                {
                    "kind": "feature_dropout",
                    "level": 0.25,
                    "f1": 0.5,
                    "label_flip_rate": 0.25,
                }
            ],
        },
        cartography_report={
            "sample_count": 4,
            "threshold": 0.4,
            "median_confidence": 0.75,
            "median_variability": 0.05,
            "region_counts": {
                "easy_to_learn": 2,
                "ambiguous": 1,
                "hard_to_learn": 1,
                "overconfident_wrong": 0,
            },
            "regions": {
                "ambiguous": [
                    {
                        "row_index": 1,
                        "label": 0,
                        "predicted_label": 1,
                        "confidence": 0.45,
                        "variability": 0.2,
                    }
                ]
            },
        },
        ood_sentinel_report={
            "sample_count": 4,
            "input_dim": 2,
            "threshold": 0.4,
            "model_used": True,
            "summary": {
                "top_row_index": 3,
                "max_ood_score": 3.2,
                "max_abs_robust_z": 4.1,
                "max_nearest_neighbor_distance": 2.5,
                "flagged_row_count": 1,
                "warning": None,
            },
            "rows": [
                {
                    "row_index": 3,
                    "ood_score": 3.2,
                    "max_abs_robust_z": 4.1,
                    "nearest_neighbor_distance": 2.5,
                    "loss": 1.1,
                    "probability": 0.9,
                    "risk_flags": ["robust_outlier"],
                }
            ],
        },
        bootstrap_stability_report={
            "sample_count": 4,
            "input_dim": 2,
            "model_count": 8,
            "feature_map": "linear",
            "threshold": 0.4,
            "ensemble_metrics": {"f1": 0.8, "accuracy": 0.75, "brier_score": 0.2},
            "summary": {
                "top_row_index": 2,
                "mean_probability_std": 0.08,
                "max_probability_std": 0.22,
                "max_disagreement_rate": 0.5,
                "unstable_row_count": 1,
                "warning": None,
            },
            "rows": [
                {
                    "row_index": 2,
                    "instability_score": 0.7,
                    "probability_std": 0.22,
                    "disagreement_rate": 0.5,
                    "mean_probability": 0.48,
                    "risk_flags": ["committee_disagreement"],
                }
            ],
        },
        prototype_audit_report={
            "sample_count": 4,
            "input_dim": 2,
            "k": 3,
            "summary": {
                "prototype_count": 2,
                "boundary_row_count": 1,
                "isolated_row_count": 1,
                "label_contradiction_count": 1,
                "top_boundary_row": 2,
                "top_label_contradiction_row": 2,
                "warning": None,
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
                    "row_index": 2,
                    "label": 1,
                    "boundary_score": 0.6,
                    "label_contradiction_score": 0.7,
                    "risk_flags": ["class_boundary", "possible_label_contradiction"],
                }
            ],
        },
        feature_separability_report={
            "sample_count": 4,
            "input_dim": 2,
            "summary": {
                "top_feature": 1,
                "top_auc": 0.95,
                "top_balanced_accuracy": 0.9,
                "near_perfect_feature_count": 1,
                "weak_feature_count": 1,
                "redundant_pair_count": 1,
                "warning": None,
            },
            "features": [
                {
                    "feature_index": 1,
                    "auc": 0.95,
                    "best_balanced_accuracy": 0.9,
                    "standardized_mean_difference": 2.4,
                    "direction": "positive_high",
                    "risk_flags": ["strong_single_feature"],
                }
            ],
            "redundant_pairs": [
                {
                    "left_feature_index": 0,
                    "right_feature_index": 1,
                    "correlation": 0.97,
                    "risk_flags": ["redundant_features"],
                }
            ],
        },
        neighborhood_hardness_report={
            "sample_count": 4,
            "input_dim": 2,
            "k": 3,
            "summary": {
                "loo_accuracy": 0.75,
                "hard_row_count": 1,
                "ambiguous_row_count": 1,
                "label_issue_candidate_count": 1,
                "locally_easy_count": 2,
                "top_hard_row": 2,
                "warning": None,
            },
            "rows": [
                {
                    "row_index": 2,
                    "label": 0,
                    "predicted_label": 1,
                    "hardness_score": 0.8,
                    "opposite_vote_rate": 1.0,
                    "vote_entropy": 0.0,
                    "risk_flags": ["label_issue_candidate"],
                }
            ],
        },
        dataset_triage_report={
            "sample_count": 4,
            "input_dim": 2,
            "class_counts": {"0": 2, "1": 2},
            "summary": {
                "readiness_score": 71.0,
                "risk_level": "medium",
                "blocking_issue_count": 1,
                "penalty_points": 29.0,
                "warning": "Review same-feature rows with conflicting labels.",
                "top_actions": [
                    "Review same-feature rows with conflicting labels.",
                    "Inspect OOD-sentinel rows for artifacts, leverage, or data-entry issues.",
                ],
            },
        },
        experiment_advisor_report={
            "summary": {
                "recommendation_count": 1,
                "top_priority": "high",
                "top_category": "thresholding",
                "recommended_next_step": "Promote threshold tuning",
                "needs_training": False,
            },
            "recommendations": [
                {
                    "rank": 1,
                    "priority": "high",
                    "category": "thresholding",
                    "title": "Promote threshold tuning",
                    "action": "Run Threshold tradeoff and Decision curve.",
                }
            ],
        },
        trial_inspector_report={
            "trial_count": 2,
            "valid_trial_count": 2,
            "invalid_trial_count": 0,
            "summary": {
                "best_trial_index": 1,
                "best_backend": "numpy",
                "best_feature_map": "rff",
                "best_f1": 0.75,
                "leader_margin_f1": 0.05,
                "recommendation": "Run another bounded auto-experiment sweep.",
                "warning": None,
            },
            "leaderboard": [
                {
                    "rank": 1,
                    "trial_index": 1,
                    "backend": "numpy",
                    "feature_map": "rff",
                    "f1": 0.75,
                    "accuracy": 0.75,
                    "validation_loss": 0.3,
                }
            ],
            "groups": [
                {
                    "group": "numpy/rff",
                    "count": 2,
                    "best_f1": 0.75,
                    "avg_f1": 0.72,
                }
            ],
        },
        promotion_gate_report={
            "summary": {
                "verdict": "needs_review",
                "promotion_score": 74.0,
                "blocker_count": 0,
                "caution_count": 2,
                "required_next_step": "Run Trial inspector.",
                "warning": "Review 2 caution(s) before relying on this model.",
            },
            "checks": [
                {
                    "rank": 1,
                    "severity": "caution",
                    "category": "model_selection",
                    "title": "Trial inspector has not been run",
                    "action": "Run Trial inspector.",
                }
            ],
            "release_note": {"recommended_use": "guarded_local_use", "must_include": ["validation metrics"]},
        },
        mps_sweep_report={
            "input_dim": 2,
            "physical_dim": 4,
            "validation_samples": 2,
            "bond_dims_tested": [4, 8],
            "recommended_bond_dim": 8,
            "recommended_f1": 0.8,
            "results": [{"bond_dim": 8, "f1": 0.8, "accuracy": 0.75, "brier_score": 0.2, "ece": 0.1}],
        },
    )

    json_path = export_experiment_report(tmp_path / "report.json", report)
    markdown_path = export_experiment_report(tmp_path / "report.md", report)

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    saved_markdown = markdown_path.read_text(encoding="utf-8")
    assert saved_json["dataset"]["class_counts"] == {"0": 2, "1": 2}
    assert saved_json["dataset"]["available"] is True
    assert saved_json["dataset"]["audit"]["duplicate_row_count"] == 1
    assert saved_json["dataset"]["audit"]["duplicate_rows"]["duplicate_group_count"] == 1
    assert saved_json["dataset"]["audit"]["class_balance"]["minority_fraction"] == 0.5
    assert saved_json["model"]["threshold"] == 0.4
    assert saved_json["uncertainty"]["conformal_source"] == "dedicated_calibration"
    assert saved_json["uncertainty"]["conformal_quantile"] == 0.35
    assert saved_json["uncertainty"]["conformal_calibration_count"] == 8
    assert saved_json["feature_ablation_diagnostics"]["summary"]["top_feature"] == "x1"
    assert saved_json["sample_review"]["summary"]["label_issue_count"] == 1
    assert saved_json["error_atlas"]["summary"]["error_count"] == 2
    assert saved_json["reliability_atlas"]["summary"]["risk_level"] == "medium"
    assert saved_json["threshold_diagnostics"]["summary"]["best_f1_threshold"] == 0.3
    assert saved_json["decision_curve_diagnostics"]["summary"]["best_threshold"] == 0.4
    assert saved_json["posthoc_conformal_diagnostics"]["summary"]["recommended_alpha"] == 0.1
    assert saved_json["posthoc_calibration_repair_diagnostics"]["summary"]["recommended_method"] == "platt"
    assert saved_json["posthoc_permutation_null_diagnostics"]["summary"]["verdict"] == "strong_signal"
    assert saved_json["population_drift_diagnostics"]["summary"]["top_feature"] == 1
    assert saved_json["adversarial_validation_diagnostics"]["summary"]["verdict"] == "strong_multivariate_shift"
    assert saved_json["chronological_holdout_diagnostics"]["summary"]["verdict"] == "severe_temporal_degradation_current_relearns"
    assert saved_json["selective_prediction_diagnostics"]["summary"]["recommended_cutoff"] == 0.2
    assert saved_json["model_response_diagnostics"]["summary"]["top_feature"] == 0
    assert saved_json["pairwise_interaction_diagnostics"]["summary"]["top_pair"] == [0, 1]
    assert saved_json["slice_diagnostics"]["summary"]["worst_slice"] == "x1[0, 1]"
    assert saved_json["subgroup_disparity_diagnostics"]["summary"]["max_disparity"] == 0.6
    assert saved_json["stress_lab"]["summary"]["worst_f1"] == 0.5
    assert saved_json["dataset_cartography"]["region_counts"]["ambiguous"] == 1
    assert saved_json["ood_sentinel"]["summary"]["top_row_index"] == 3
    assert saved_json["bootstrap_stability_diagnostics"]["summary"]["top_row_index"] == 2
    assert saved_json["prototype_audit"]["summary"]["top_boundary_row"] == 2
    assert saved_json["feature_separability"]["summary"]["top_feature"] == 1
    assert saved_json["neighborhood_hardness"]["summary"]["top_hard_row"] == 2
    assert saved_json["dataset_triage"]["summary"]["readiness_score"] == 71.0
    assert saved_json["experiment_advisor"]["summary"]["recommended_next_step"] == "Promote threshold tuning"
    assert saved_json["trial_inspector"]["summary"]["best_trial_index"] == 1
    assert saved_json["promotion_gate"]["summary"]["verdict"] == "needs_review"
    assert saved_json["mps_bond_sweep"]["recommended_bond_dim"] == 8
    assert saved_json["trial_history"][0]["config"]["feature_map"] == "rff"
    assert "Feature 0" in saved_markdown
    assert "## Dataset Audit" in saved_markdown
    assert "Trial 1" in saved_markdown
    assert "## Uncertainty" in saved_markdown
    assert "conformal_source" in saved_markdown
    assert "## Ablation Diagnostics" in saved_markdown
    assert "Label-proxy flags" in saved_markdown
    assert "## Sample Review" in saved_markdown
    assert "label_issue row 2" in saved_markdown
    assert "## Error Atlas" in saved_markdown
    assert "high-confidence error row 2" in saved_markdown
    assert "## Reliability Atlas" in saved_markdown
    assert "Run Calibration repair" in saved_markdown
    assert "## Threshold Tradeoffs" in saved_markdown
    assert "Best F1 threshold" in saved_markdown
    assert "## Decision Curve / Utility" in saved_markdown
    assert "Useful threshold ranges" in saved_markdown
    assert "## Post-Hoc Conformal Diagnostics" in saved_markdown
    assert "Recommended alpha" in saved_markdown
    assert "## Post-Hoc Calibration Repair" in saved_markdown
    assert "Recommended method" in saved_markdown
    assert "## Post-Hoc Permutation-Null Diagnostic" in saved_markdown
    assert "F1 p-value" in saved_markdown
    assert "## Population Drift Diagnostics" in saved_markdown
    assert "Max PSI" in saved_markdown
    assert "## Adversarial Validation" in saved_markdown
    assert "Domain AUC" in saved_markdown
    assert "## Chronological Holdout Diagnostics" in saved_markdown
    assert "Current-baseline F1 gain" in saved_markdown
    assert "## Selective Prediction / Risk-Coverage" in saved_markdown
    assert "Recommended cutoff" in saved_markdown
    assert "## Model Response / Partial Dependence" in saved_markdown
    assert "Top response range" in saved_markdown
    assert "## Pairwise Feature Interactions" in saved_markdown
    assert "Top interaction strength" in saved_markdown
    assert "## Slice Diagnostics" in saved_markdown
    assert "x1[0.0000, 1.0000]" in saved_markdown
    assert "## Subgroup Disparity Diagnostics" in saved_markdown
    assert "Max FNR gap" in saved_markdown
    assert "## Robustness Stress Lab" in saved_markdown
    assert "feature_dropout" in saved_markdown
    assert "## Dataset Cartography" in saved_markdown
    assert "Ambiguous rows" in saved_markdown
    assert "## Feature Separability Lens" in saved_markdown
    assert "Near-perfect features" in saved_markdown
    assert "## Neighborhood Hardness" in saved_markdown
    assert "Leave-one-out accuracy" in saved_markdown
    assert "## Dataset Triage" in saved_markdown
    assert "Readiness score" in saved_markdown
    assert "## Experiment Advisor" in saved_markdown
    assert "Promote threshold tuning" in saved_markdown
    assert "## Trial Inspector" in saved_markdown
    assert "Rank 1: trial 1" in saved_markdown
    assert "## Promotion Gate" in saved_markdown
    assert "Trial inspector has not been run" in saved_markdown
    assert "## OOD Sentinel" in saved_markdown
    assert "Max OOD score" in saved_markdown
    assert "## Bootstrap Stability Diagnostics" in saved_markdown
    assert "Mean probability std" in saved_markdown
    assert "## Prototype Audit" in saved_markdown
    assert "Possible label contradictions" in saved_markdown
    assert "## MPS Bond Sweep" in saved_markdown
    assert "Recommended chi" in saved_markdown


def test_report_marks_dataset_unavailable_for_model_only_export():
    report = build_experiment_report(
        sample_count=0,
        input_dim=2,
        labels=[],
        config=ModelConfig(),
        metrics={"f1": 0.0},
        threshold=0.5,
        preprocessor=None,
        feature_importances=[],
    )

    assert report["dataset"]["available"] is False
    assert report["dataset"]["class_counts"] is None
