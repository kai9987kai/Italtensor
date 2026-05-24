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
    assert saved_json["threshold_diagnostics"]["summary"]["best_f1_threshold"] == 0.3
    assert saved_json["decision_curve_diagnostics"]["summary"]["best_threshold"] == 0.4
    assert saved_json["selective_prediction_diagnostics"]["summary"]["recommended_cutoff"] == 0.2
    assert saved_json["slice_diagnostics"]["summary"]["worst_slice"] == "x1[0, 1]"
    assert saved_json["stress_lab"]["summary"]["worst_f1"] == 0.5
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
    assert "## Threshold Tradeoffs" in saved_markdown
    assert "Best F1 threshold" in saved_markdown
    assert "## Decision Curve / Utility" in saved_markdown
    assert "Useful threshold ranges" in saved_markdown
    assert "## Selective Prediction / Risk-Coverage" in saved_markdown
    assert "Recommended cutoff" in saved_markdown
    assert "## Slice Diagnostics" in saved_markdown
    assert "x1[0.0000, 1.0000]" in saved_markdown
    assert "## Robustness Stress Lab" in saved_markdown
    assert "feature_dropout" in saved_markdown


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
