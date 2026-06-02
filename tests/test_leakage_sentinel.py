import numpy as np
import pytest

from italtensor.leakage_sentinel import (
    format_leakage_sentinel_summary,
    leakage_sentinel_dataset_fingerprint,
    run_leakage_sentinel,
)


def test_leakage_sentinel_flags_direct_label_copy():
    labels = [0, 0, 0, 1, 1, 1, 0, 1]
    features = [
        [-1.0, 0.0, 0.2],
        [-0.8, 0.0, -0.1],
        [-0.7, 0.0, 0.1],
        [0.7, 1.0, 0.0],
        [0.8, 1.0, -0.1],
        [1.0, 1.0, 0.1],
        [0.2, 0.0, -0.2],
        [-0.1, 1.0, 0.2],
    ]

    report = run_leakage_sentinel(features, labels)

    assert report["summary"]["risk_level"] == "high"
    assert report["summary"]["top_feature"] == 1
    assert report["features"][0]["risk_level"] == "high"
    assert "direct_label_copy_candidate" in report["features"][0]["risk_flags"]
    assert "Quarantine" in report["summary"]["recommendation"]
    assert format_leakage_sentinel_summary(report).startswith("Leakage sentinel:")


def test_leakage_sentinel_flags_low_cardinality_proxy_with_conflicts():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int32)
    proxy = np.asarray([10, 10, 10, 10, 20, 20, 20, 21, 20, 20], dtype=np.float32)
    stable = np.asarray([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.0, 0.4, -0.4, 0.1], dtype=np.float32)
    noise = np.asarray([0.0, 0.2, -0.1, 0.1, 0.0, -0.2, 0.1, 0.2, -0.3, 0.3], dtype=np.float32)
    features = np.column_stack([stable, proxy, noise])

    report = run_leakage_sentinel(features, labels)

    top = report["features"][0]
    assert report["summary"]["risk_level"] in {"medium", "high"}
    assert top["feature_index"] == 1
    assert "low_cardinality_label_mapping" in top["risk_flags"]
    assert top["mixed_value_count"] >= 1
    assert top["label_mapping_balanced_accuracy"] >= 0.75


def test_leakage_sentinel_keeps_low_risk_dataset_as_evidence():
    rng = np.random.default_rng(12)
    labels = np.asarray([0, 1] * 20, dtype=np.int32)
    features = rng.normal(0.0, 1.0, size=(labels.shape[0], 4)).astype(np.float32)

    report = run_leakage_sentinel(features, labels)

    assert report["summary"]["risk_level"] in {"low", "medium"}
    assert report["sample_count"] == labels.shape[0]
    assert len(report["features"]) == 4
    assert leakage_sentinel_dataset_fingerprint(features, labels) == report["dataset_fingerprint"]


def test_leakage_sentinel_validates_inputs():
    with pytest.raises(ValueError, match="at least six"):
        run_leakage_sentinel([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

    with pytest.raises(ValueError, match="at least two rows per class"):
        run_leakage_sentinel([[float(index)] for index in range(6)], [0, 0, 0, 0, 0, 1])

    with pytest.raises(ValueError, match="finite"):
        run_leakage_sentinel([[0.0], [1.0], [2.0], [3.0], [4.0], [float("inf")]], [0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="binary"):
        run_leakage_sentinel([[float(index)] for index in range(6)], [0, 1, 2, 1, 0, 1])
