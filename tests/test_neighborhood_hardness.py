from __future__ import annotations

import numpy as np
import pytest

from italtensor.neighborhood_hardness import (
    format_neighborhood_hardness_summary,
    run_neighborhood_hardness_diagnostics,
)


def _hardness_dataset() -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [-2.0, -1.0],
            [-1.8, -0.9],
            [-1.6, -0.8],
            [2.0, 1.0],
            [1.8, 0.9],
            [1.6, 0.8],
            [-0.05, 0.02],
            [0.05, -0.02],
            [0.00, 0.06],
            [1.9, 1.1],
            [-1.9, -1.1],
            [0.08, -0.06],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    return features, labels


def test_neighborhood_hardness_ranks_local_label_issues_and_ambiguity():
    features, labels = _hardness_dataset()

    report = run_neighborhood_hardness_diagnostics(features, labels, k=3, max_rows=6)

    assert report["sample_count"] == 12
    assert report["input_dim"] == 2
    assert report["k"] == 3
    assert report["class_counts"] == {"0": 6, "1": 6}
    assert report["summary"]["loo_accuracy"] < 1.0
    assert report["summary"]["hard_row_count"] >= 1
    assert report["summary"]["label_issue_candidate_count"] >= 1
    assert report["rows"][0]["hardness_score"] >= report["rows"][-1]["hardness_score"]
    assert any("label_issue_candidate" in row["risk_flags"] for row in report["rows"])
    assert any("ambiguous_neighborhood" in row["risk_flags"] for row in report["rows"])


def test_neighborhood_hardness_is_deterministic():
    features, labels = _hardness_dataset()

    first = run_neighborhood_hardness_diagnostics(features, labels, k=4, max_rows=4)
    second = run_neighborhood_hardness_diagnostics(features, labels, k=4, max_rows=4)

    assert first == second


def test_neighborhood_hardness_rejects_invalid_inputs():
    features, labels = _hardness_dataset()

    with pytest.raises(ValueError, match="2D"):
        run_neighborhood_hardness_diagnostics([0.0] * 6, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="numeric"):
        run_neighborhood_hardness_diagnostics([["bad"], [0], [1], [2], [3], [4]], [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        run_neighborhood_hardness_diagnostics(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [float("nan")]],
            [0, 0, 0, 1, 1, 1],
        )
    with pytest.raises(ValueError, match="counts"):
        run_neighborhood_hardness_diagnostics(features, labels[:-1])
    with pytest.raises(ValueError, match="at least six"):
        run_neighborhood_hardness_diagnostics(features[:5], labels[:5])
    with pytest.raises(ValueError, match="integer binary"):
        run_neighborhood_hardness_diagnostics(features, labels.astype(float) + 0.1)
    with pytest.raises(ValueError, match="two rows per class"):
        run_neighborhood_hardness_diagnostics(features, np.asarray([0] * 11 + [1]))
    with pytest.raises(ValueError, match="k"):
        run_neighborhood_hardness_diagnostics(features, labels, k=0)
    with pytest.raises(ValueError, match="max_rows"):
        run_neighborhood_hardness_diagnostics(features, labels, max_rows=0)


def test_format_neighborhood_hardness_summary():
    features, labels = _hardness_dataset()
    report = run_neighborhood_hardness_diagnostics(features, labels, k=3, max_rows=4)

    summary = format_neighborhood_hardness_summary(report)

    assert "Neighborhood hardness" in summary
    assert "k=3" in summary
    assert "loo_acc=" in summary
    assert "label_issue=" in summary
