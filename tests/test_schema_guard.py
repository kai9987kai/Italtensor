import numpy as np
import pytest

from italtensor.schema_guard import (
    check_vector_against_schema,
    format_schema_guard_summary,
    run_schema_guard,
    schema_guard_dataset_fingerprint,
)


def test_schema_guard_builds_numeric_contract_and_flags_schema_risks():
    rows = 80
    signal = np.linspace(-1.0, 1.0, rows)
    wide = np.linspace(10.0, 1000.0, rows)
    near_constant = np.ones(rows)
    near_constant[0] = 1.04
    status = np.tile([0.0, 1.0, 2.0, 0.0], rows // 4)
    sparse = np.zeros(rows)
    sparse[:4] = 1.0
    tail = np.zeros(rows)
    tail[-3:] = [4.5, -5.0, 6.0]
    features = np.column_stack([signal, wide, near_constant, status, sparse, tail])
    labels = (signal > 0).astype(np.int32)

    report = run_schema_guard(
        features,
        labels,
        feature_names=[
            "signal",
            "wide",
            "near_constant",
            "status",
            "sparse",
            "tail",
        ],
    )

    by_name = {row["feature_name"]: row for row in report["features"]}
    assert report["sample_count"] == rows
    assert report["input_dim"] == 6
    assert report["class_counts"] == {"0": 40, "1": 40}
    assert report["summary"]["risk_level"] in {"medium", "high"}
    assert "near_constant_feature" in by_name["near_constant"]["risk_flags"]
    assert "low_cardinality_numeric" in by_name["status"]["risk_flags"]
    assert "mostly_constant_with_spikes" in by_name["tail"]["risk_flags"]
    assert report["dataset_fingerprint"] == schema_guard_dataset_fingerprint(features)
    assert format_schema_guard_summary(report).startswith("Schema guard:")


def test_schema_guard_fingerprint_is_row_order_insensitive():
    features = np.asarray([[0.2, 1.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float32)

    assert schema_guard_dataset_fingerprint(features) == schema_guard_dataset_fingerprint(features[[2, 0, 1]])


def test_schema_guard_vector_check_flags_out_of_contract_values():
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [5.0, 1.0],
        ],
        dtype=np.float32,
    )
    report = run_schema_guard(features, feature_names=["score", "code"])

    fail = check_vector_against_schema([9.0, 3.0], report)
    ok = check_vector_against_schema([3.0, 1.0], report)

    assert fail["status"] == "fail"
    assert any(item["kind"] == "outside_observed_range" for item in fail["failures"])
    assert any(item["kind"] == "unseen_low_cardinality_value" for item in fail["failures"])
    assert ok["status"] == "pass"


def test_schema_guard_validates_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_schema_guard([1.0, 2.0])
    with pytest.raises(ValueError, match="at least two"):
        run_schema_guard([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        run_schema_guard([[1.0], [np.inf]])
    with pytest.raises(ValueError, match="feature_names"):
        run_schema_guard([[1.0], [2.0]], feature_names=["one", "two"])

