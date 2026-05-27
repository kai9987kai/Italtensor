import numpy as np
import pytest

from italtensor.dataset_triage import format_dataset_triage_summary, run_dataset_triage


def _triage_fixture():
    features = [
        [-1.2, -1.0, -1.18, 0.1, 1.0, 0.0],
        [-1.0, -1.0, -0.98, -0.2, 1.0, 0.0],
        [-0.8, -1.0, -0.79, 0.1, 1.0, 0.0],
        [-0.2, -1.0, -0.21, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.02, 0.0, 1.0, 0.0],
        [0.2, 1.0, 0.19, 0.1, 1.0, 0.0],
        [0.8, 1.0, 0.82, -0.1, 1.0, 0.0],
        [1.0, 1.0, 1.01, 0.2, 1.0, 0.0],
        [1.2, 1.0, 1.22, -0.1, 1.0, 0.0],
        [0.1, -1.0, 0.1, 0.0, 1.0, 5.0],
        [0.1, -1.0, 0.1, 0.0, 1.0, 5.0],
        [-0.1, 1.0, -0.1, 0.0, 1.0, -5.0],
    ]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
    return features, labels


def test_dataset_triage_runs_components_and_ranks_actions():
    features, labels = _triage_fixture()

    report = run_dataset_triage(features, labels)

    assert report["sample_count"] == len(labels)
    assert report["input_dim"] == 6
    assert report["class_counts"] == {"0": 6, "1": 6}
    assert set(report) >= {
        "dataset_audit",
        "feature_separability",
        "prototype_audit",
        "neighborhood_hardness",
        "ood_sentinel",
    }
    assert report["dataset_audit"]["label_conflict_count"] >= 1
    assert report["feature_separability"]["summary"]["redundant_pair_count"] >= 1
    assert report["neighborhood_hardness"]["summary"]["label_issue_candidate_count"] >= 1
    assert report["ood_sentinel"]["model_used"] is False
    assert 0.0 <= report["summary"]["readiness_score"] <= 100.0
    assert report["summary"]["risk_level"] in {"low", "medium", "high"}
    assert report["summary"]["top_actions"]


def test_dataset_triage_is_deterministic():
    features, labels = _triage_fixture()

    first = run_dataset_triage(features, labels)
    second = run_dataset_triage(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32))

    assert first["summary"] == second["summary"]
    assert first["feature_separability"]["summary"] == second["feature_separability"]["summary"]
    assert first["neighborhood_hardness"]["summary"] == second["neighborhood_hardness"]["summary"]


def test_dataset_triage_rejects_insufficient_data():
    with pytest.raises(ValueError, match="at least 6"):
        run_dataset_triage([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

    with pytest.raises(ValueError, match="at least two rows per class"):
        run_dataset_triage([[float(index)] for index in range(6)], [0, 0, 0, 0, 0, 1])


def test_format_dataset_triage_summary():
    features, labels = _triage_fixture()

    summary = format_dataset_triage_summary(run_dataset_triage(features, labels))

    assert summary.startswith("Dataset triage:")
    assert "readiness=" in summary
    assert "actions=" in summary
