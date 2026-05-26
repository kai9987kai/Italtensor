from __future__ import annotations

import numpy as np
import pytest

from italtensor.feature_separability import (
    format_feature_separability_summary,
    run_feature_separability_diagnostics,
)


def _separability_dataset() -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([0] * 8 + [1] * 8, dtype=np.int32)
    strong = np.asarray([-2.1, -1.8, -1.6, -1.3, -1.1, -0.9, -0.7, -0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 1.8, 2.1])
    weak = np.asarray([-0.2, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0, 0.0, -0.1, 0.2, -0.2, 0.1, 0.0, -0.1, 0.2])
    noise = np.asarray([0.3, -0.4, 0.1, 0.8, -0.8, 0.5, -0.2, 0.0, -0.3, 0.4, -0.1, -0.8, 0.8, -0.5, 0.2, 0.0])
    shortcut = np.asarray([-1.0] * 8 + [1.0] * 8)
    redundant = strong + 0.02
    features = np.column_stack([strong, weak, noise, shortcut, redundant]).astype(np.float32)
    return features, labels


def test_feature_separability_ranks_shortcut_and_redundancy():
    features, labels = _separability_dataset()

    report = run_feature_separability_diagnostics(features, labels, max_features=5)

    assert report["sample_count"] == 16
    assert report["input_dim"] == 5
    assert report["class_counts"] == {"0": 8, "1": 8}
    assert report["summary"]["near_perfect_feature_count"] >= 2
    assert report["summary"]["weak_feature_count"] >= 1
    assert report["summary"]["redundant_pair_count"] >= 1
    assert report["features"][0]["auc"] == pytest.approx(1.0)
    assert "near_perfect_single_feature" in report["features"][0]["risk_flags"]
    assert any(
        {pair["left_feature_index"], pair["right_feature_index"]} == {0, 4}
        for pair in report["redundant_pairs"]
    )


def test_feature_separability_is_deterministic():
    features, labels = _separability_dataset()

    first = run_feature_separability_diagnostics(features, labels)
    second = run_feature_separability_diagnostics(features, labels)

    assert first == second


def test_feature_separability_rejects_invalid_inputs():
    features, labels = _separability_dataset()

    with pytest.raises(ValueError, match="2D"):
        run_feature_separability_diagnostics([0.0] * 6, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="numeric"):
        run_feature_separability_diagnostics([["bad"], [0], [1], [2], [3], [4]], [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        run_feature_separability_diagnostics(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [float("nan")]],
            [0, 0, 0, 1, 1, 1],
        )
    with pytest.raises(ValueError, match="counts"):
        run_feature_separability_diagnostics(features, labels[:-1])
    with pytest.raises(ValueError, match="at least six"):
        run_feature_separability_diagnostics(features[:5], labels[:5])
    with pytest.raises(ValueError, match="integer binary"):
        run_feature_separability_diagnostics(features, labels.astype(float) + 0.1)
    with pytest.raises(ValueError, match="two rows per class"):
        run_feature_separability_diagnostics(features, np.asarray([0] * 15 + [1]))
    with pytest.raises(ValueError, match="max_features"):
        run_feature_separability_diagnostics(features, labels, max_features=0)


def test_format_feature_separability_summary():
    features, labels = _separability_dataset()
    report = run_feature_separability_diagnostics(features, labels)

    summary = format_feature_separability_summary(report)

    assert "Feature separability" in summary
    assert "top=x" in summary
    assert "near_perfect=" in summary
    assert "redundant_pairs=" in summary
