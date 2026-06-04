import numpy as np
import pytest

from italtensor.data_value import (
    data_value_dataset_fingerprint,
    format_data_value_summary,
    run_data_value_scout,
)


def _curation_dataset():
    features = np.asarray(
        [
            [-1.2, -0.6, 1.0, 0.0, 0.0],
            [-1.2, -0.6, 1.0, 0.0, 0.0],
            [-1.2, -0.6, 1.0, 0.0, 0.0],
            [1.2, 0.6, 1.0, 0.0, 0.0],
            [1.2, 0.6, 1.0, 0.0, 0.0],
            [1.2, 0.6, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.1, 0.0, 0.0, 0.0, 2.2],
            [0.1, 0.0, 0.0, 0.0, 2.2],
            [1.2, 0.6, 0.0, 4.8, 0.0],
            [1.1, 0.5, 0.0, 4.9, 0.0],
            [-1.0, -0.5, 0.0, 0.0, 0.0],
            [0.9, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)
    return features, labels


def test_data_value_scout_flags_review_redundancy_and_coverage_rows():
    features, labels = _curation_dataset()

    report = run_data_value_scout(features, labels, k=5, max_rows=6)

    assert report["sample_count"] == features.shape[0]
    assert report["input_dim"] == features.shape[1]
    assert report["summary"]["priority"] == "high"
    assert report["summary"]["review_row_count"] >= 2
    assert report["summary"]["redundant_row_count"] >= 2
    assert report["summary"]["coverage_row_count"] >= 1
    assert any("review_or_relabel" in row["risk_flags"] for row in report["review_rows"])
    assert any("redundant_anchor" in row["risk_flags"] for row in report["redundant_rows"])
    assert any("rare_coverage" in row["risk_flags"] for row in report["coverage_rows"])
    assert "Data value scout" in format_data_value_summary(report)


def test_data_value_scout_rejects_bad_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_data_value_scout([0.1, 0.2], [0, 1])

    with pytest.raises(ValueError, match="at least six"):
        run_data_value_scout([[0.1], [0.2], [0.3], [0.4]], [0, 0, 1, 1])

    with pytest.raises(ValueError, match="binary"):
        run_data_value_scout([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], [0, 1, 2, 0, 1, 0])


def test_data_value_fingerprint_is_order_sensitive():
    features, labels = _curation_dataset()

    assert data_value_dataset_fingerprint(features, labels) != data_value_dataset_fingerprint(
        features[::-1],
        labels[::-1],
    )
