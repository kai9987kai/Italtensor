from italtensor.experiment_advisor import build_experiment_advisor, format_experiment_advisor_summary
from italtensor.modeling import ModelConfig


def _triage_report():
    return {
        "summary": {
            "readiness_score": 54.0,
            "risk_level": "high",
            "top_actions": [
                "Review same-feature rows with conflicting labels.",
                "Inspect OOD-sentinel rows for artifacts, leverage, or data-entry issues.",
            ],
        },
        "feature_separability": {
            "input_dim": 3,
            "summary": {
                "weak_feature_count": 3,
                "strong_feature_count": 0,
                "near_perfect_feature_count": 0,
            },
        },
        "neighborhood_hardness": {"summary": {"loo_accuracy": 0.72}},
    }


def test_experiment_advisor_recommends_loading_data_without_dataset():
    report = build_experiment_advisor(sample_count=0, input_dim=None, labels=[])

    assert report["summary"]["needs_training"] is True
    assert report["recommendations"][0]["category"] == "data"
    assert "Load" in report["recommendations"][0]["title"]


def test_experiment_advisor_prioritizes_triage_before_training():
    report = build_experiment_advisor(
        sample_count=24,
        input_dim=3,
        labels=[0, 1] * 12,
        config=ModelConfig(feature_map="linear"),
        dataset_triage_report=_triage_report(),
    )

    top = report["recommendations"][0]
    assert top["source"] == "dataset_triage"
    assert top["priority"] == "high"
    training = [item for item in report["recommendations"] if item["category"] == "training"][0]
    assert training["suggested_config"]["feature_map"] == "rff"
    assert training["suggested_config"]["use_cv"] is True


def test_experiment_advisor_uses_metrics_for_next_runs():
    report = build_experiment_advisor(
        sample_count=100,
        input_dim=4,
        labels=[0, 1] * 50,
        config=ModelConfig(feature_map="linear"),
        metrics={
            "f1": 0.52,
            "fixed_threshold_f1": 0.40,
            "threshold_gain_f1": 0.12,
            "ece": 0.11,
            "brier_score": 0.24,
            "precision": 0.82,
            "recall": 0.40,
        },
        trial_history=[{"metrics": {"f1": 0.52}}],
    )
    categories = {item["category"] for item in report["recommendations"]}

    assert {"model_selection", "thresholding", "calibration", "search"}.issubset(categories)
    assert report["summary"]["needs_training"] is False
    assert report["summary"]["model_f1"] == 0.52


def test_experiment_advisor_recommends_external_holdout_when_metrics_exist():
    report = build_experiment_advisor(
        sample_count=100,
        input_dim=4,
        labels=[0, 1] * 50,
        metrics={"f1": 0.82, "accuracy": 0.84, "precision": 0.80, "recall": 0.85},
        trial_history=[{"metrics": {"f1": 0.82}}] * 3,
    )

    holdout = [item for item in report["recommendations"] if item["source"] == "missing_external_holdout"][0]
    assert holdout["category"] == "external_validation"
    assert holdout["priority"] == "medium"
    assert "separate labeled CSV" in holdout["reason"]


def test_experiment_advisor_prioritizes_high_risk_leakage_before_training():
    report = build_experiment_advisor(
        sample_count=80,
        input_dim=3,
        labels=[0, 1] * 40,
        leakage_sentinel_report={
            "summary": {
                "risk_level": "high",
                "top_feature": 2,
                "max_risk_score": 0.98,
                "recommendation": "Quarantine x3.",
            }
        },
    )

    top = report["recommendations"][0]
    assert top["source"] == "leakage_sentinel"
    assert top["priority"] == "high"
    assert top["category"] == "leakage"
    assert "Quarantine x3" in top["action"]


def test_experiment_advisor_prioritizes_external_holdout_failure():
    report = build_experiment_advisor(
        sample_count=120,
        input_dim=3,
        labels=[0, 1] * 60,
        metrics={"f1": 0.86, "accuracy": 0.88, "precision": 0.84, "recall": 0.88},
        trial_history=[{"metrics": {"f1": 0.86}}] * 3,
        external_holdout_report={
            "summary": {
                "verdict": "holdout_failure",
                "f1": 0.42,
                "balanced_accuracy": 0.50,
                "recommendation": "Investigate external holdout errors.",
            }
        },
    )

    top = report["recommendations"][0]
    assert top["source"] == "external_holdout"
    assert top["priority"] == "high"
    assert top["category"] == "external_validation"
    assert "Investigate external holdout errors" in top["action"]


def test_experiment_advisor_holdout_failure_overrides_triage_queue():
    report = build_experiment_advisor(
        sample_count=120,
        input_dim=3,
        labels=[0, 1] * 60,
        metrics={"f1": 0.86, "accuracy": 0.88, "precision": 0.84, "recall": 0.88},
        trial_history=[{"metrics": {"f1": 0.86}}] * 3,
        dataset_triage_report=_triage_report(),
        external_holdout_report={
            "summary": {
                "verdict": "holdout_failure",
                "f1": 0.42,
                "balanced_accuracy": 0.50,
                "recommendation": "Investigate external holdout errors.",
            }
        },
    )

    assert report["recommendations"][0]["source"] == "external_holdout"


def test_experiment_advisor_recommends_localized_calibration_review():
    report = build_experiment_advisor(
        sample_count=120,
        input_dim=3,
        labels=[0, 1] * 60,
        metrics={"f1": 0.82, "accuracy": 0.84, "precision": 0.80, "recall": 0.85},
        trial_history=[{"metrics": {"f1": 0.82}}] * 3,
        external_holdout_report={"summary": {"verdict": "holdout_pass", "f1": 0.82}},
        calibration_slice_report={
            "summary": {
                "risk_level": "high",
                "worst_slice": "x2[0.40, 0.80]",
                "max_absolute_confidence_gap": 0.31,
                "recommendation": "Review x2[0.40, 0.80].",
            }
        },
    )

    calibration = [item for item in report["recommendations"] if item["source"] == "calibration_slices"][0]
    assert calibration["priority"] == "high"
    assert calibration["category"] == "calibration"
    assert "x2[0.40, 0.80]" in calibration["reason"]


def test_experiment_advisor_reacts_to_rising_learning_curve():
    report = build_experiment_advisor(
        sample_count=120,
        input_dim=4,
        labels=[0, 1] * 60,
        metrics={"f1": 0.78, "accuracy": 0.80, "precision": 0.76, "recall": 0.81},
        trial_history=[{"metrics": {"f1": 0.78}}] * 3,
        external_holdout_report={"summary": {"verdict": "holdout_pass", "f1": 0.77}},
        learning_curve_report={
            "summary": {
                "verdict": "more_data_helpful",
                "final_f1": 0.76,
                "best_f1": 0.76,
                "f1_gain": 0.14,
                "recommended_next_step": "Collect more labeled rows.",
            }
        },
    )

    curve = [item for item in report["recommendations"] if item["source"] == "learning_curve"][0]
    assert curve["category"] == "data_volume"
    assert curve["priority"] == "medium"
    assert "Collect more labeled rows" in curve["action"]


def test_experiment_advisor_prioritizes_high_priority_data_acquisition_plan():
    report = build_experiment_advisor(
        sample_count=80,
        input_dim=4,
        labels=[0, 1] * 40,
        metrics={"f1": 0.78, "accuracy": 0.80, "precision": 0.76, "recall": 0.81},
        trial_history=[{"metrics": {"f1": 0.78}}] * 3,
        external_holdout_report={"summary": {"verdict": "holdout_pass", "f1": 0.77}},
        data_acquisition_report={
            "summary": {
                "priority": "high",
                "recommended_label_budget": 18,
                "recommended_next_step": "Collect more class 1 labels.",
            }
        },
    )

    top = report["recommendations"][0]
    assert top["source"] == "data_acquisition_plan"
    assert top["priority"] == "high"
    assert "Collect more class 1 labels" in top["action"]


def test_experiment_advisor_prioritizes_high_priority_data_value_scout():
    report = build_experiment_advisor(
        sample_count=80,
        input_dim=4,
        labels=[0, 1] * 40,
        metrics={"f1": 0.78, "accuracy": 0.80, "precision": 0.76, "recall": 0.81},
        trial_history=[{"metrics": {"f1": 0.78}}] * 3,
        external_holdout_report={"summary": {"verdict": "holdout_pass", "f1": 0.77}},
        data_value_report={
            "summary": {
                "priority": "high",
                "review_row_count": 5,
                "redundant_row_count": 2,
                "recommended_next_step": "Inspect review rows first.",
            }
        },
    )

    top = report["recommendations"][0]
    assert top["source"] == "data_value_scout"
    assert top["category"] == "data_curation"
    assert "Inspect review rows" in top["action"]


def test_experiment_advisor_prioritizes_high_priority_label_sensitivity():
    report = build_experiment_advisor(
        sample_count=80,
        input_dim=4,
        labels=[0, 1] * 40,
        metrics={"f1": 0.78, "accuracy": 0.80, "precision": 0.76, "recall": 0.81},
        trial_history=[{"metrics": {"f1": 0.78}}] * 3,
        external_holdout_report={"summary": {"verdict": "holdout_pass", "f1": 0.77}},
        label_sensitivity_report={
            "summary": {
                "priority": "high",
                "suspect_label_count": 3,
                "max_improving_f1_delta": 0.09,
                "recommended_next_step": "Review sensitive labels first.",
            }
        },
    )

    top = report["recommendations"][0]
    assert top["source"] == "label_sensitivity"
    assert top["category"] == "label_review"
    assert top["priority"] == "high"
    assert "Review sensitive labels" in top["action"]


def test_experiment_advisor_prioritizes_label_noise_stress_fragility():
    report = build_experiment_advisor(
        sample_count=80,
        input_dim=4,
        labels=[0, 1] * 40,
        metrics={"f1": 0.78, "accuracy": 0.80, "precision": 0.76, "recall": 0.81},
        trial_history=[{"metrics": {"f1": 0.78}}] * 3,
        external_holdout_report={"summary": {"verdict": "holdout_pass", "f1": 0.77}},
        label_noise_stress_report={
            "summary": {
                "priority": "high",
                "worst_mean_f1_drop": 0.14,
                "first_material_noise_rate": 0.10,
                "recommended_next_step": "Audit labels before the next sweep.",
            }
        },
    )

    top = report["recommendations"][0]
    assert top["source"] == "label_noise_stress"
    assert top["category"] == "label_review"
    assert top["priority"] == "high"
    assert "Audit labels" in top["action"]


def test_experiment_advisor_is_deterministic_and_formats_summary():
    kwargs = {
        "sample_count": 24,
        "input_dim": 3,
        "labels": [0, 1] * 12,
        "dataset_triage_report": _triage_report(),
    }

    first = build_experiment_advisor(**kwargs)
    second = build_experiment_advisor(**kwargs)

    assert first == second
    summary = format_experiment_advisor_summary(first)
    assert summary.startswith("Experiment advisor:")
    assert "recommendations=" in summary
