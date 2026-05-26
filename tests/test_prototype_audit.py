from __future__ import annotations

import numpy as np
import pytest

from italtensor.prototype_audit import format_prototype_audit_summary, run_prototype_audit


def _prototype_dataset() -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [-1.20, -0.90],
            [-1.10, -0.80],
            [-0.95, -0.75],
            [1.20, 0.90],
            [1.10, 0.80],
            [0.95, 0.75],
            [-0.04, 0.02],
            [0.04, -0.02],
            [0.00, 0.05],
            [2.90, -2.90],
            [-2.90, 2.90],
            [0.08, -0.05],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    return features, labels


def test_prototype_audit_ranks_prototypes_boundaries_and_sparse_rows():
    features, labels = _prototype_dataset()

    report = run_prototype_audit(features, labels, k=3, max_rows=5)

    assert report["sample_count"] == 12
    assert report["input_dim"] == 2
    assert report["k"] == 3
    assert report["class_counts"] == {"0": 6, "1": 6}
    assert report["summary"]["prototype_count"] >= 2
    assert report["summary"]["boundary_row_count"] >= 3
    assert report["summary"]["isolated_row_count"] >= 1
    assert {row["label"] for row in report["prototypes"]} == {0, 1}
    assert all("neighbor_indices" in row for row in report["rows"])
    assert report["boundary_rows"][0]["boundary_score"] >= report["boundary_rows"][-1]["boundary_score"]
    assert report["isolated_rows"][0]["row_index"] in {9, 10}
    assert "class_boundary" in report["boundary_rows"][0]["risk_flags"]


def test_prototype_audit_is_deterministic():
    features, labels = _prototype_dataset()

    first = run_prototype_audit(features, labels, k=4, max_rows=4)
    second = run_prototype_audit(features, labels, k=4, max_rows=4)

    assert first == second


def test_prototype_audit_rejects_invalid_inputs():
    features, labels = _prototype_dataset()

    with pytest.raises(ValueError, match="2D"):
        run_prototype_audit([0.0] * 6, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="numeric"):
        run_prototype_audit([["bad"], [0], [1], [2], [3], [4]], [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        run_prototype_audit([[0.0], [1.0], [2.0], [3.0], [4.0], [float("nan")]], [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="counts"):
        run_prototype_audit(features, labels[:-1])
    with pytest.raises(ValueError, match="at least six"):
        run_prototype_audit(features[:5], labels[:5])
    with pytest.raises(ValueError, match="integer binary"):
        run_prototype_audit(features, labels.astype(float) + 0.1)
    with pytest.raises(ValueError, match="two rows per class"):
        run_prototype_audit(features, np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    with pytest.raises(ValueError, match="k"):
        run_prototype_audit(features, labels, k=0)
    with pytest.raises(ValueError, match="max_rows"):
        run_prototype_audit(features, labels, max_rows=0)


def test_format_prototype_audit_summary():
    features, labels = _prototype_dataset()
    report = run_prototype_audit(features, labels, k=3, max_rows=4)

    summary = format_prototype_audit_summary(report)

    assert "Prototype audit" in summary
    assert "k=3" in summary
    assert "prototypes=" in summary
    assert "top_boundary=" in summary
