import numpy as np
import pytest

from italtensor.data_acquisition import (
    data_acquisition_dataset_fingerprint,
    format_data_acquisition_summary,
    run_data_acquisition_planner,
)


def test_data_acquisition_planner_prioritizes_minority_and_boundary_rows():
    rng = np.random.default_rng(4)
    negatives = rng.normal(loc=(-1.0, -0.5), scale=0.25, size=(36, 2))
    positives = rng.normal(loc=(1.0, 0.5), scale=0.25, size=(6, 2))
    boundary = np.asarray([[0.02, 0.0], [-0.03, 0.02], [0.04, -0.02], [4.5, 0.0]], dtype=np.float32)
    features = np.vstack([negatives, positives, boundary]).astype(np.float32)
    labels = np.asarray([0] * 36 + [1] * 6 + [0, 1, 0, 1], dtype=np.int32)

    report = run_data_acquisition_planner(features, labels, min_class_count=16)

    assert report["sample_count"] == 46
    assert report["input_dim"] == 2
    assert report["summary"]["priority"] == "high"
    assert report["summary"]["recommended_label_budget"] > 0
    assert any(item["category"] == "class_balance" for item in report["recommendations"])
    assert any(item["candidate_type"] == "boundary" for item in report["row_candidates"])
    assert "Data acquisition plan" in format_data_acquisition_summary(report)


def test_data_acquisition_planner_flags_missing_class():
    features = np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 0], dtype=np.int32)

    report = run_data_acquisition_planner(features, labels, min_class_count=5)

    top = report["recommendations"][0]
    assert top["category"] == "class_balance"
    assert "class 1" in top["title"]
    assert report["summary"]["verdict"] == "collect_before_model_selection"


def test_data_acquisition_rejects_bad_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_data_acquisition_planner([0.1, 0.2], [0, 1])

    with pytest.raises(ValueError, match="binary"):
        run_data_acquisition_planner([[0.1], [0.2]], [0, 2])


def test_data_acquisition_fingerprint_is_order_sensitive():
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.3], [0.3, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    assert data_acquisition_dataset_fingerprint(features, labels) != data_acquisition_dataset_fingerprint(
        features[::-1],
        labels[::-1],
    )
