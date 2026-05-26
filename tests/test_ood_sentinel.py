from __future__ import annotations

import numpy as np
import pytest

from italtensor.modeling import NumpyBinaryClassifier
from italtensor.ood_sentinel import format_ood_sentinel_summary, run_ood_sentinel


def test_ood_sentinel_ranks_geometric_outlier_first():
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            [8.0, 8.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 0, 1], dtype=np.int32)

    report = run_ood_sentinel(None, features, labels, max_rows=3)

    assert report["sample_count"] == 6
    assert report["input_dim"] == 2
    assert report["model_used"] is False
    assert len(report["rows"]) == 3
    assert report["summary"]["top_row_index"] == 5
    assert report["rows"][0]["row_index"] == 5
    assert report["rows"][0]["ood_score"] > report["rows"][1]["ood_score"]
    assert report["rows"][0]["max_abs_robust_z"] > report["rows"][1]["max_abs_robust_z"]
    assert report["rows"][0]["nearest_neighbor_distance"] > report["rows"][1]["nearest_neighbor_distance"]
    assert "feature_tail" in report["rows"][0]["risk_flags"]
    assert "OOD sentinel" in format_ood_sentinel_summary(report)


def test_ood_sentinel_model_free_mode_omits_model_metrics():
    features = np.asarray(
        [[0.0, 0.0], [0.2, 0.1], [-0.1, 0.1], [0.1, -0.2]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    report = run_ood_sentinel(None, features, labels)

    assert report["model_used"] is False
    assert report["summary"]["model_loss_available"] is False
    assert report["summary"]["misclassification_count"] == 0
    assert all(row["probability"] is None for row in report["rows"])
    assert all(row["model_loss"] is None for row in report["rows"])


def test_ood_sentinel_model_loss_contributes_to_ranking():
    model = NumpyBinaryClassifier(
        weights=np.asarray([0.0], dtype=np.float32),
        bias=float(np.log(0.95 / 0.05)),
    )
    features = np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    labels = np.asarray([1, 1, 1, 0], dtype=np.int32)

    report = run_ood_sentinel(model, features, labels, max_rows=4)

    top = report["rows"][0]
    correct = report["rows"][-1]
    assert report["model_used"] is True
    assert report["summary"]["model_loss_available"] is True
    assert report["summary"]["misclassification_count"] == 1
    assert top["row_index"] == 3
    assert top["misclassified"] is True
    assert top["model_loss"] > correct["model_loss"]
    assert top["model_score"] > correct["model_score"]
    assert "high_model_loss" in top["risk_flags"]
    assert "model=on" in format_ood_sentinel_summary(report)


def test_ood_sentinel_rejects_invalid_inputs():
    valid_features = np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32)
    valid_labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="2D"):
        run_ood_sentinel(None, [0.0, 1.0, 2.0, 3.0], valid_labels)
    with pytest.raises(ValueError, match="numeric"):
        run_ood_sentinel(None, [["bad"], [0.1], [0.2], [0.3]], valid_labels)
    with pytest.raises(ValueError, match="finite"):
        run_ood_sentinel(None, [[0.0], [0.1], [float("nan")], [0.3]], valid_labels)
    with pytest.raises(ValueError, match="counts"):
        run_ood_sentinel(None, valid_features, [0, 1])
    with pytest.raises(ValueError, match="binary"):
        run_ood_sentinel(None, valid_features, [0, 1, 2, 1])
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_ood_sentinel(None, valid_features, valid_labels, threshold=1.2)
    with pytest.raises(ValueError, match="max_rows"):
        run_ood_sentinel(None, valid_features, valid_labels, max_rows=0)
    with pytest.raises(ValueError, match="min_reference_samples"):
        run_ood_sentinel(None, valid_features[:3], valid_labels[:3], min_reference_samples=4)
