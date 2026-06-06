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
        label_sensitivity_report={
            "observed": {"f1": 0.75, "accuracy": 0.75},
            "summary": {
                "verdict": "review_sensitive_labels",
                "priority": "high",
                "suspect_label_count": 2,
                "anchor_row_count": 1,
                "max_abs_f1_delta": 0.12,
                "max_improving_f1_delta": 0.12,
                "top_suspect_row": 2,
                "top_anchor_row": 0,
                "recommended_next_step": "Review sensitive labels.",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "priority": "high",
                    "category": "label_quality",
                    "action": "Review sensitive labels.",
                }
            ],
            "suspect_label_rows": [
                {
                    "row_index": 2,
                    "label": 0,
                    "flipped_label": 1,
                    "probability": 0.95,
                    "f1_delta_if_flipped": 0.12,
                    "risk_flags": ["flip_improves_f1"],
                }
            ],
            "anchor_rows": [{"row_index": 0, "label": 0, "probability": 0.10, "f1_delta_if_flipped": -0.08}],
        },
        label_noise_stress_report={
            "summary": {
                "verdict": "label_noise_review",
                "priority": "medium",
                "baseline_f1": 0.75,
                "baseline_accuracy": 0.80,
                "worst_noise_rate": 0.20,
                "worst_mean_f1_drop": 0.07,
                "first_material_noise_rate": 0.10,
                "recommended_next_step": "Review noisy labels before another sweep.",
            },
            "split": {
                "train_sample_count": 3,
                "validation_sample_count": 1,
                "train_class_counts": {"0": 2, "1": 1},
                "validation_class_counts": {"0": 0, "1": 1},
            },
            "rates": [
                {
                    "noise_rate": 0.0,
                    "repeat_count": 1,
                    "mean_metrics": {"f1": 0.75, "accuracy": 0.80},
                    "degradation": {"f1_drop": 0.0, "brier_increase": 0.0},
                },
                {
                    "noise_rate": 0.20,
                    "repeat_count": 3,
                    "mean_metrics": {"f1": 0.68, "accuracy": 0.72},
                    "degradation": {"f1_drop": 0.07, "brier_increase": 0.03},
                },
            ],
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
        calibration_slice_report={
            "summary": {
                "risk_level": "high",
                "slice_count": 2,
                "worst_slice": "x2[0, 1]",
                "max_absolute_confidence_gap": 0.42,
                "max_weighted_calibration_impact": 0.12,
                "high_risk_slice_count": 1,
                "recommendation": "Review x2[0, 1].",
            },
            "base": {
                "expected_calibration_error": 0.08,
                "brier_score": 0.16,
            },
            "slices": [
                {
                    "feature_index": 1,
                    "left": 0.0,
                    "right": 1.0,
                    "count": 6,
                    "label_prevalence": 0.25,
                    "mean_probability": 0.67,
                    "signed_confidence_gap": 0.42,
                    "weighted_calibration_impact": 0.12,
                    "calibration_direction": "overconfident",
                }
            ],
        },
        external_holdout_report={
            "sample_count": 12,
            "summary": {
                "verdict": "holdout_shift_review",
                "f1": 0.72,
                "balanced_accuracy": 0.74,
                "recommendation": "Compare holdout rows with the loaded dataset.",
            },
            "metrics": {
                "f1": 0.72,
                "balanced_accuracy": 0.74,
                "precision": 0.75,
                "recall": 0.70,
            },
            "probability_diagnostics": {
                "expected_calibration_error": 0.09,
                "brier_score": 0.18,
            },
            "reference_comparison": {
                "top_shift_feature": 1,
                "max_standardized_mean_shift": 1.2,
                "label_prevalence_shift": 0.25,
            },
        },
        shadow_replay_report={
            "summary": {
                "verdict": "ordered_degradation_review",
                "first_window_f1": 0.9,
                "last_window_f1": 0.6,
                "worst_window_f1": 0.6,
                "max_f1_drop": 0.3,
                "max_accuracy_drop": 0.2,
                "max_brier_increase": 0.12,
                "degradation_window_count": 1,
                "recommendation": "Inspect degraded replay windows.",
            },
            "threshold": 0.4,
            "window_count": 2,
            "row_order_assumption": "loaded_row_order",
            "worst_windows": [
                {
                    "window_index": 1,
                    "start_row": 2,
                    "end_row_exclusive": 4,
                    "f1": 0.6,
                    "accuracy": 0.5,
                    "brier_score": 0.3,
                    "f1_delta_vs_first": -0.3,
                }
            ],
            "error_runs": [
                {"start_row": 2, "end_row_exclusive": 4, "length": 2, "mean_loss": 1.1, "mean_confidence": 0.8}
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
        threshold_stability_report={
            "current_threshold": 0.4,
            "bootstrap_samples": 24,
            "threshold_interval": {"q05": 0.25, "q50": 0.35, "q95": 0.55},
            "summary": {
                "verdict": "threshold_stability_review",
                "full_best_threshold": 0.35,
                "median_best_threshold": 0.35,
                "threshold_spread": 0.3,
                "current_inside_interval": True,
                "median_f1_gain_vs_current": 0.04,
                "selection_agreement_rate": 0.5,
                "recommendation": "Review threshold stability.",
            },
        },
        capacity_planner_report={
            "utility_model": {"true_positive_value": 5.0, "false_positive_cost": 1.0, "review_cost": 0.25},
            "summary": {
                "verdict": "actionable_capacity_plan",
                "best_capacity_fraction": 0.5,
                "best_k": 2,
                "best_threshold_floor": 0.65,
                "best_precision_at_k": 0.5,
                "best_recall_captured": 0.5,
                "best_lift": 1.0,
                "best_net_utility": 3.5,
                "recommendation": "Plan review/action capacity around top 2 row(s).",
            },
            "capacity_points": [
                {
                    "capacity_fraction": 0.5,
                    "k": 2,
                    "precision_at_k": 0.5,
                    "recall_captured": 0.5,
                    "lift": 1.0,
                    "net_utility": 3.5,
                }
            ],
            "top_rows": [{"rank": 1, "row_index": 0, "label": 1, "probability": 0.9}],
        },
        rank_lift_report={
            "summary": {
                "verdict": "concentrated_ranking",
                "prevalence": 0.5,
                "top_10_lift": 2.0,
                "top_10_positive_capture": 0.5,
                "top_20_lift": 2.0,
                "top_20_positive_capture": 0.5,
                "normalized_gains_auc": 0.8,
                "score_gini": 0.3,
                "recommended_next_step": "Validate top-decile lift.",
            },
            "points": [
                {
                    "top_fraction": 0.25,
                    "k": 1,
                    "precision_at_k": 1.0,
                    "positive_capture": 0.5,
                    "lift": 2.0,
                    "probability_mass_capture": 0.4,
                }
            ],
            "deciles_table": [
                {
                    "bucket": 1,
                    "rank_start": 1,
                    "rank_end": 1,
                    "count": 1,
                    "positive_count": 1,
                    "response_rate": 1.0,
                    "lift": 2.0,
                    "cumulative_positive_capture": 0.5,
                }
            ],
        },
        prior_shift_report={
            "summary": {
                "verdict": "prevalence_shift_risk",
                "observed_prevalence": 0.5,
                "sensitivity": 0.8,
                "specificity": 0.75,
                "false_positive_rate": 0.25,
                "false_negative_rate": 0.2,
                "current_ppv": 0.7,
                "current_npv": 0.85,
                "min_ppv": 0.2,
                "max_predicted_positive_per_1000": 260.0,
                "max_false_positive_per_1000": 245.0,
                "recommended_next_step": "Validate deployment prevalence.",
            },
            "current": {
                "true_positive": 2,
                "false_positive": 1,
                "false_negative": 0,
                "true_negative": 1,
            },
            "points": [
                {
                    "prevalence": 0.02,
                    "positive_predictive_value": 0.2,
                    "negative_predictive_value": 0.99,
                    "expected_predicted_positive": 260.0,
                    "expected_false_positive": 245.0,
                }
            ],
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
        learning_curve_report={
            "sample_count": 4,
            "input_dim": 2,
            "summary": {
                "verdict": "more_data_helpful",
                "first_f1": 0.55,
                "final_f1": 0.75,
                "best_f1": 0.75,
                "best_fraction": 1.0,
                "f1_gain": 0.20,
                "best_gap_vs_final": 0.0,
                "recommended_next_step": "Collect more labeled rows.",
            },
            "points": [
                {
                    "train_fraction": 0.25,
                    "train_samples": 2,
                    "f1": 0.55,
                    "accuracy": 0.6,
                    "balanced_accuracy": 0.6,
                    "validation_loss": 0.7,
                },
                {
                    "train_fraction": 1.0,
                    "train_samples": 4,
                    "f1": 0.75,
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.8,
                    "validation_loss": 0.4,
                },
            ],
        },
        validation_stability_report={
            "n_splits": 3,
            "repeats": 2,
            "total_fold_count": 6,
            "threshold": 0.5,
            "summary": {
                "verdict": "validation_stability_review",
                "priority": "medium",
                "stability_score": 68.0,
                "fold_f1_std": 0.08,
                "fold_f1_q10": 0.62,
                "worst_fold_f1": 0.58,
                "weak_fold_count": 1,
                "worst_fold_validation_rows": [1, 3],
                "recommended_next_step": "Confirm the lower range on an external holdout.",
            },
            "aggregate": {
                "f1": {"mean": 0.76, "std": 0.08, "q10": 0.62, "q90": 0.84},
            },
            "tuned_aggregate": {
                "f1": {"mean": 0.79, "std": 0.06, "q10": 0.68, "q90": 0.86},
            },
            "calibration_threshold_distribution": {"std": 0.07},
            "folds": [
                {
                    "repeat": 1,
                    "fold": 1,
                    "train_sample_count": 2,
                    "calibration_sample_count": 1,
                    "validation_sample_count": 1,
                    "metrics": {"f1": 0.58, "balanced_accuracy": 0.60},
                    "tuned_threshold": 0.42,
                    "tuned_metrics": {"f1": 0.66},
                }
            ],
            "interpretation_note": "Fold quantiles are empirical split-sensitivity summaries, not formal confidence intervals.",
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
        canary_suite_report={
            "preset_name": "Canary regression lab",
            "summary": {
                "verdict": "canary_review",
                "checked_count": 2,
                "passed_count": 2,
                "failed_count": 0,
                "review_count": 1,
                "informational_count": 1,
                "pass_rate": 1.0,
                "min_probability_margin_observed": 0.02,
                "recommended_next_step": "Review low-margin canaries.",
            },
            "examples": [
                {
                    "name": "Canary boundary review",
                    "status": "review",
                    "probability": 0.52,
                    "predicted_label": 1,
                    "expected_label": 1,
                    "margin_to_threshold": 0.02,
                    "schema_status": "pass",
                }
            ],
        },
        policy_guard_report={
            "preset_name": "Monotonic policy lab",
            "summary": {
                "verdict": "policy_review",
                "check_count": 2,
                "pair_count": 24,
                "violation_count": 1,
                "violation_rate": 0.0417,
                "max_violation": 0.02,
                "worst_check": "Risk score should rise",
                "recommended_next_step": "Review weak policy violations.",
            },
            "checks": [
                {
                    "name": "Risk score should rise",
                    "status": "review",
                    "feature_name": "risk_score",
                    "direction": "increasing",
                    "pair_count": 12,
                    "violation_count": 1,
                    "violation_rate": 0.0833,
                    "max_violation": 0.02,
                }
            ],
        },
        schema_guard_report={
            "sample_count": 4,
            "input_dim": 2,
            "summary": {
                "risk_level": "medium",
                "readiness_score": 78.0,
                "constant_feature_count": 1,
                "near_constant_feature_count": 0,
                "low_cardinality_feature_count": 1,
                "outlier_feature_count": 1,
                "recommended_next_step": "Review schema warnings.",
            },
            "features": [
                {
                    "feature_index": 0,
                    "feature_name": "dead_sensor",
                    "min": 1.0,
                    "q01": 1.0,
                    "q99": 1.0,
                    "max": 1.0,
                    "unique_count": 1,
                    "outlier_count": 0,
                    "risk_flags": ["constant_feature", "low_cardinality_numeric"],
                }
            ],
        },
        leakage_sentinel_report={
            "sample_count": 4,
            "input_dim": 2,
            "summary": {
                "risk_level": "high",
                "top_feature": 1,
                "max_risk_score": 0.97,
                "high_risk_feature_count": 1,
                "medium_risk_feature_count": 0,
                "direct_label_copy_candidate_count": 1,
                "low_cardinality_label_mapping_count": 1,
                "recommendation": "Quarantine x2.",
            },
            "features": [
                {
                    "feature_index": 1,
                    "risk_level": "high",
                    "risk_score": 0.97,
                    "auc": 1.0,
                    "best_balanced_accuracy": 1.0,
                    "label_mapping_balanced_accuracy": 1.0,
                    "unique_count": 2,
                    "risk_flags": ["direct_label_copy_candidate"],
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
        validation_plan_report={
            "sample_count": 4,
            "input_dim": 2,
            "class_counts": {"0": 2, "1": 2},
            "summary": {
                "recommended_strategy": "stratified_kfold",
                "risk_level": "medium",
                "readiness_score": 76.0,
                "minority_class_count": 2,
                "imbalance_ratio": 1.0,
                "kfold_splits": 2,
                "validation_fraction": None,
                "row_order_risk": False,
                "warning": None,
            },
            "split_blueprint": {"shuffle": True, "stratify": True},
            "row_order": {"prevalence_delta": 0.0, "max_standardized_mean_shift": 0.1},
            "recommendations": [
                {
                    "rank": 1,
                    "priority": "high",
                    "category": "validation",
                    "action": "Use stratified cross-validation.",
                }
            ],
        },
        data_acquisition_report={
            "sample_count": 4,
            "input_dim": 2,
            "class_counts": {"0": 2, "1": 2},
            "summary": {
                "verdict": "collect_before_model_selection",
                "priority": "high",
                "readiness_score": 64.0,
                "recommended_label_budget": 14,
                "boundary_candidate_count": 2,
                "tail_candidate_count": 1,
                "feature_review_count": 1,
                "recommended_next_step": "Collect more class 1 labels.",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "priority": "high",
                    "category": "class_balance",
                    "action": "Collect more class 1 labels.",
                }
            ],
            "row_candidates": [{"row_index": 2, "candidate_type": "boundary", "score": 0.88, "label": 1}],
        },
        data_value_report={
            "sample_count": 4,
            "input_dim": 2,
            "k": 3,
            "summary": {
                "verdict": "curate_before_model_selection",
                "priority": "high",
                "readiness_score": 68.0,
                "review_row_count": 2,
                "redundant_row_count": 1,
                "coverage_row_count": 1,
                "high_value_row_count": 1,
                "max_review_score": 0.9,
                "recommended_next_step": "Inspect review rows first.",
            },
            "recommendations": [
                {
                    "rank": 1,
                    "priority": "high",
                    "category": "row_review",
                    "action": "Inspect review rows first.",
                }
            ],
            "rows": [
                {
                    "row_index": 2,
                    "value_score": 0.2,
                    "review_score": 0.9,
                    "redundancy_score": 0.1,
                    "coverage_score": 0.2,
                    "risk_flags": ["review_or_relabel"],
                }
            ],
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
        mps_order_sweep_report={
            "input_dim": 2,
            "physical_dim": 4,
            "bond_dim": 8,
            "validation_samples": 2,
            "orders_tested": ["original", "reversed"],
            "recommended_order_name": "reversed",
            "recommended_order": [1, 0],
            "recommended_feature_order_1_based": [2, 1],
            "recommended_f1": 0.85,
            "original_f1": 0.75,
            "best_delta_f1_vs_original": 0.1,
            "material_gain": True,
            "adoption_note": "This is site-order sensitivity evidence.",
            "results": [
                {
                    "order_name": "reversed",
                    "feature_order_1_based": [2, 1],
                    "f1": 0.85,
                    "accuracy": 0.8,
                    "brier_score": 0.18,
                    "ece": 0.08,
                }
            ],
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
    assert saved_json["posthoc_label_sensitivity_diagnostics"]["summary"]["priority"] == "high"
    assert saved_json["label_noise_stress_diagnostics"]["summary"]["worst_noise_rate"] == 0.20
    assert saved_json["error_atlas"]["summary"]["error_count"] == 2
    assert saved_json["reliability_atlas"]["summary"]["risk_level"] == "medium"
    assert saved_json["calibration_slice_diagnostics"]["summary"]["risk_level"] == "high"
    assert saved_json["external_holdout"]["summary"]["verdict"] == "holdout_shift_review"
    assert saved_json["shadow_replay"]["summary"]["verdict"] == "ordered_degradation_review"
    assert saved_json["threshold_diagnostics"]["summary"]["best_f1_threshold"] == 0.3
    assert saved_json["threshold_stability"]["summary"]["verdict"] == "threshold_stability_review"
    assert saved_json["capacity_planner"]["summary"]["best_k"] == 2
    assert saved_json["rank_lift"]["summary"]["verdict"] == "concentrated_ranking"
    assert saved_json["prior_shift"]["summary"]["verdict"] == "prevalence_shift_risk"
    assert saved_json["decision_curve_diagnostics"]["summary"]["best_threshold"] == 0.4
    assert saved_json["posthoc_conformal_diagnostics"]["summary"]["recommended_alpha"] == 0.1
    assert saved_json["posthoc_calibration_repair_diagnostics"]["summary"]["recommended_method"] == "platt"
    assert saved_json["posthoc_permutation_null_diagnostics"]["summary"]["verdict"] == "strong_signal"
    assert saved_json["population_drift_diagnostics"]["summary"]["top_feature"] == 1
    assert saved_json["adversarial_validation_diagnostics"]["summary"]["verdict"] == "strong_multivariate_shift"
    assert saved_json["chronological_holdout_diagnostics"]["summary"]["verdict"] == "severe_temporal_degradation_current_relearns"
    assert saved_json["learning_curve"]["summary"]["verdict"] == "more_data_helpful"
    assert saved_json["validation_stability_diagnostics"]["summary"]["fold_f1_std"] == 0.08
    assert saved_json["selective_prediction_diagnostics"]["summary"]["recommended_cutoff"] == 0.2
    assert saved_json["model_response_diagnostics"]["summary"]["top_feature"] == 0
    assert saved_json["pairwise_interaction_diagnostics"]["summary"]["top_pair"] == [0, 1]
    assert saved_json["slice_diagnostics"]["summary"]["worst_slice"] == "x1[0, 1]"
    assert saved_json["subgroup_disparity_diagnostics"]["summary"]["max_disparity"] == 0.6
    assert saved_json["stress_lab"]["summary"]["worst_f1"] == 0.5
    assert saved_json["dataset_cartography"]["region_counts"]["ambiguous"] == 1
    assert saved_json["ood_sentinel"]["summary"]["top_row_index"] == 3
    assert saved_json["bootstrap_stability_diagnostics"]["summary"]["top_row_index"] == 2
    assert saved_json["canary_suite"]["summary"]["verdict"] == "canary_review"
    assert saved_json["policy_guard"]["summary"]["verdict"] == "policy_review"
    assert saved_json["schema_guard"]["summary"]["risk_level"] == "medium"
    assert saved_json["leakage_sentinel"]["summary"]["risk_level"] == "high"
    assert saved_json["prototype_audit"]["summary"]["top_boundary_row"] == 2
    assert saved_json["feature_separability"]["summary"]["top_feature"] == 1
    assert saved_json["neighborhood_hardness"]["summary"]["top_hard_row"] == 2
    assert saved_json["dataset_triage"]["summary"]["readiness_score"] == 71.0
    assert saved_json["validation_plan"]["summary"]["recommended_strategy"] == "stratified_kfold"
    assert saved_json["data_acquisition_plan"]["summary"]["priority"] == "high"
    assert saved_json["data_value_scout"]["summary"]["review_row_count"] == 2
    assert saved_json["experiment_advisor"]["summary"]["recommended_next_step"] == "Promote threshold tuning"
    assert saved_json["trial_inspector"]["summary"]["best_trial_index"] == 1
    assert saved_json["promotion_gate"]["summary"]["verdict"] == "needs_review"
    assert saved_json["mps_bond_sweep"]["recommended_bond_dim"] == 8
    assert saved_json["mps_site_order_sweep"]["recommended_order_name"] == "reversed"
    assert saved_json["trial_history"][0]["config"]["feature_map"] == "rff"
    assert "Feature 0" in saved_markdown
    assert "## Dataset Audit" in saved_markdown
    assert "## Schema Guard" in saved_markdown
    assert "dead_sensor" in saved_markdown
    assert "## Leakage Sentinel" in saved_markdown
    assert "Quarantine x2" in saved_markdown
    assert "Trial 1" in saved_markdown
    assert "## Uncertainty" in saved_markdown
    assert "conformal_source" in saved_markdown
    assert "## Ablation Diagnostics" in saved_markdown
    assert "Label-proxy flags" in saved_markdown
    assert "## Sample Review" in saved_markdown
    assert "label_issue row 2" in saved_markdown
    assert "## Post-Hoc Label Sensitivity" in saved_markdown
    assert "suspect row 2" in saved_markdown
    assert "## Label Noise Stress" in saved_markdown
    assert "Review noisy labels" in saved_markdown
    assert "## Error Atlas" in saved_markdown
    assert "high-confidence error row 2" in saved_markdown
    assert "## Reliability Atlas" in saved_markdown
    assert "## Calibration Slice Diagnostics" in saved_markdown
    assert "Max confidence gap" in saved_markdown
    assert "Review x2[0, 1]" in saved_markdown
    assert "## External Holdout Evaluation" in saved_markdown
    assert "Compare holdout rows with the loaded dataset" in saved_markdown
    assert "## Capacity Planner" in saved_markdown
    assert "budget=0.5000" in saved_markdown
    assert "## Rank Lift / Gains" in saved_markdown
    assert "Top 10% lift" in saved_markdown
    assert "Validate top-decile lift" in saved_markdown
    assert "## Prior Shift / Prevalence" in saved_markdown
    assert "False-positive rate" in saved_markdown
    assert "Min simulated PPV" in saved_markdown
    assert "Validate deployment prevalence" in saved_markdown
    assert "Run Calibration repair" in saved_markdown
    assert "## Shadow Replay Diagnostics" in saved_markdown
    assert "Inspect degraded replay windows" in saved_markdown
    assert "## Threshold Tradeoffs" in saved_markdown
    assert "Best F1 threshold" in saved_markdown
    assert "## Threshold Stability" in saved_markdown
    assert "Review threshold stability" in saved_markdown
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
    assert "## Learning Curve" in saved_markdown
    assert "Collect more labeled rows" in saved_markdown
    assert "## Validation Stability" in saved_markdown
    assert "Confirm the lower range" in saved_markdown
    assert "not formal confidence intervals" in saved_markdown
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
    assert "## MPS Site-Order Sweep" in saved_markdown
    assert "site-order sensitivity" in saved_markdown
    assert "Leave-one-out accuracy" in saved_markdown
    assert "## Dataset Triage" in saved_markdown
    assert "Readiness score" in saved_markdown
    assert "## Validation Plan" in saved_markdown
    assert "Use stratified cross-validation" in saved_markdown
    assert "## Data Acquisition Planner" in saved_markdown
    assert "Collect more class 1 labels." in saved_markdown
    assert "## Data Value Scout" in saved_markdown
    assert "Inspect review rows first." in saved_markdown
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
    assert "## Canary Suite" in saved_markdown
    assert "Canary boundary review" in saved_markdown
    assert "## Policy Guard" in saved_markdown
    assert "Risk score should rise" in saved_markdown
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
