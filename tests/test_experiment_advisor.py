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
