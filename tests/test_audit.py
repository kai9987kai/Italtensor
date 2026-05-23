from italtensor.audit import audit_dataset, format_audit_summary


def test_audit_dataset_flags_duplicates_conflicts_and_constant_features():
    audit = audit_dataset(
        [
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 0.0],
            [3.0, 6.0, 0.0],
        ],
        [0, 0, 1, 0, 0],
    )

    assert audit["duplicate_row_count"] == 2
    assert audit["label_conflict_count"] == 1
    assert audit["constant_features"] == [2]
    assert "duplicate feature rows" in audit["warnings"]
    assert "same features appear with both labels" in audit["warnings"]
    assert "constant features" in audit["warnings"]


def test_audit_dataset_reports_class_imbalance_and_high_correlations():
    audit = audit_dataset(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ],
        [0, 0, 0, 0, 0, 1],
    )

    assert audit["class_counts"] == {"0": 5, "1": 1}
    assert audit["imbalance_ratio"] == 5.0
    assert audit["high_correlations"][0]["left"] == 0
    assert audit["high_correlations"][0]["right"] == 1
    assert "strong class imbalance" in audit["warnings"]
    assert "highly correlated features" in audit["warnings"]


def test_format_audit_summary_is_compact():
    summary = format_audit_summary(
        {
            "sample_count": 4,
            "input_dim": 2,
            "class_counts": {"0": 2, "1": 2},
            "imbalance_ratio": 1.0,
            "duplicate_row_count": 0,
            "label_conflict_count": 0,
            "constant_features": [],
            "high_correlations": [],
            "warnings": [],
        }
    )

    assert "Dataset audit:" in summary
    assert "warnings=no warnings" in summary
