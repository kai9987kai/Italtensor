from italtensor.modeling import ModelConfig
from italtensor.promotion_gate import build_promotion_gate, format_promotion_gate_summary


def test_promotion_gate_blocks_without_dataset_or_metrics():
    report = build_promotion_gate(sample_count=0, input_dim=None, labels=[], metrics={})

    assert report["summary"]["verdict"] == "blocked"
    assert report["summary"]["blocker_count"] >= 2
    assert report["checks"][0]["severity"] == "blocker"
    assert "Load" in report["summary"]["required_next_step"] or "Train" in report["summary"]["required_next_step"]


def test_promotion_gate_promotes_strong_evidence():
    report = build_promotion_gate(
        sample_count=120,
        input_dim=3,
        labels=[0, 1] * 60,
        config=ModelConfig(feature_map="linear"),
        metrics={
            "f1": 0.88,
            "accuracy": 0.90,
            "balanced_accuracy": 0.89,
            "brier_score": 0.12,
            "ece": 0.03,
            "threshold_gain_f1": 0.02,
        },
        trial_history=[{"metrics": {"f1": 0.86}}, {"metrics": {"f1": 0.88}}, {"metrics": {"f1": 0.84}}],
        dataset_triage_report={"summary": {"risk_level": "low", "readiness_score": 92.0, "blocking_issue_count": 0}},
        trial_inspector_report={"valid_trial_count": 4, "summary": {"leader_margin_f1": 0.06}},
    )

    assert report["summary"]["verdict"] == "promotable"
    assert report["summary"]["promotion_score"] >= 82.0
    assert report["summary"]["blocker_count"] == 0


def test_promotion_gate_blocks_high_risk_dataset_and_poor_scores():
    report = build_promotion_gate(
        sample_count=36,
        input_dim=5,
        labels=[0] * 30 + [1] * 6,
        metrics={
            "f1": 0.51,
            "accuracy": 0.82,
            "balanced_accuracy": 0.57,
            "brier_score": 0.30,
            "ece": 0.14,
            "threshold_gain_f1": 0.12,
        },
        dataset_triage_report={
            "summary": {
                "risk_level": "high",
                "readiness_score": 48.0,
                "blocking_issue_count": 2,
                "top_actions": ["Review shortcut conflicts."],
            }
        },
        trial_inspector_report={"valid_trial_count": 2, "summary": {"leader_margin_f1": 0.01}},
    )
    categories = {item["category"] for item in report["checks"]}

    assert report["summary"]["verdict"] == "blocked"
    assert {"data_quality", "model", "calibration", "model_selection"}.issubset(categories)
    assert report["summary"]["blocker_count"] >= 3


def test_promotion_gate_needs_review_for_thin_or_unstable_search():
    report = build_promotion_gate(
        sample_count=80,
        input_dim=2,
        labels=[0, 1] * 40,
        metrics={"f1": 0.82, "accuracy": 0.83, "balanced_accuracy": 0.82, "brier_score": 0.16, "ece": 0.05},
        trial_history=[{"metrics": {"f1": 0.82}}, {"metrics": {"f1": 0.81}}],
        dataset_triage_report={"summary": {"risk_level": "low", "readiness_score": 88.0, "blocking_issue_count": 0}},
        trial_inspector_report={"valid_trial_count": 2, "summary": {"leader_margin_f1": 0.01}},
        error_atlas_report={"summary": {"high_confidence_error_count": 1, "error_rate": 0.10}},
        reliability_atlas_report={"summary": {"risk_level": "medium", "expected_calibration_error": 0.09, "max_calibration_error": 0.25}},
    )

    assert report["summary"]["verdict"] == "needs_review"
    assert any(item["title"] == "Leaderboard winner is unstable" for item in report["checks"])
    assert any(item["category"] == "error_analysis" for item in report["checks"])
    assert any(item["title"] == "Reliability atlas needs calibration review" for item in report["checks"])
    summary = format_promotion_gate_summary(report)
    assert summary.startswith("Promotion gate:")
    assert "verdict=needs_review" in summary
