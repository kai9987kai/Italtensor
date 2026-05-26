from __future__ import annotations

import numpy as np
import pytest

from italtensor.bootstrap_stability import (
    format_bootstrap_stability_summary,
    run_bootstrap_stability_diagnostics,
)


def _boundary_dataset() -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [-3.0, -1.0],
            [-2.5, -0.9],
            [-2.0, -1.1],
            [-1.5, -0.6],
            [1.5, 0.6],
            [2.0, 1.1],
            [2.5, 0.9],
            [3.0, 1.0],
            [-0.08, 0.05],
            [0.08, -0.05],
            [0.02, 0.04],
            [-0.02, -0.04],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0], dtype=np.int32)
    return features, labels


def test_bootstrap_stability_ranks_boundary_rows_as_unstable():
    features, labels = _boundary_dataset()

    report = run_bootstrap_stability_diagnostics(
        features,
        labels,
        model_count=8,
        train_fraction=0.75,
        seed=11,
        max_epochs=25,
        max_rows=6,
    )

    scores = [row["instability_score"] for row in report["rows"]]
    top_rows = {row["row_index"] for row in report["rows"][:4]}

    assert report["sample_count"] == 12
    assert report["input_dim"] == 2
    assert report["model_count"] == 8
    assert report["summary"]["unstable_row_count"] >= 2
    assert scores == sorted(scores, reverse=True)
    assert top_rows == {8, 9, 10, 11}
    assert report["rows"][0]["label_disagreement_rate"] > 0.0
    assert report["rows"][0]["boundary_score"] > 0.9
    assert "correct" in report["rows"][0]
    assert all(run["train_class_counts"]["0"] >= 1 for run in report["training"]["runs"])
    assert all(run["train_class_counts"]["1"] >= 1 for run in report["training"]["runs"])
    assert all(run["heldout_class_counts"]["0"] >= 1 for run in report["training"]["runs"])
    assert all(run["heldout_class_counts"]["1"] >= 1 for run in report["training"]["runs"])


def test_bootstrap_stability_is_deterministic_for_fixed_seed():
    features, labels = _boundary_dataset()

    first = run_bootstrap_stability_diagnostics(
        features,
        labels,
        model_count=5,
        train_fraction=0.75,
        seed=19,
        max_epochs=12,
        max_rows=5,
    )
    second = run_bootstrap_stability_diagnostics(
        features,
        labels,
        model_count=5,
        train_fraction=0.75,
        seed=19,
        max_epochs=12,
        max_rows=5,
    )

    assert first == second


def test_bootstrap_stability_rejects_fractional_binary_labels():
    features, _ = _boundary_dataset()
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0.9, 1, 1, 0], dtype=np.float32)

    with pytest.raises(ValueError, match="strict binary integer"):
        run_bootstrap_stability_diagnostics(features, labels, model_count=2, max_epochs=2)


def test_bootstrap_stability_rejects_invalid_inputs():
    features, labels = _boundary_dataset()

    with pytest.raises(ValueError, match="2D"):
        run_bootstrap_stability_diagnostics([0.0] * 8, [0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="numeric"):
        run_bootstrap_stability_diagnostics([["bad"], [0], [1], [2], [3], [4], [5], [6]], [0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        run_bootstrap_stability_diagnostics(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [float("nan")]],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
    with pytest.raises(ValueError, match="counts"):
        run_bootstrap_stability_diagnostics(features, labels[:-1])
    with pytest.raises(ValueError, match="at least 8"):
        run_bootstrap_stability_diagnostics(features[:7], labels[:7])
    with pytest.raises(ValueError, match="both classes"):
        run_bootstrap_stability_diagnostics(features, np.ones(features.shape[0], dtype=np.int32))
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_bootstrap_stability_diagnostics(features, labels, train_fraction=1.0)
    with pytest.raises(ValueError, match="threshold"):
        run_bootstrap_stability_diagnostics(features, labels, threshold=-0.1)
    with pytest.raises(ValueError, match="model_count"):
        run_bootstrap_stability_diagnostics(features, labels, model_count=1)
    with pytest.raises(ValueError, match="feature_map"):
        run_bootstrap_stability_diagnostics(features, labels, feature_map="bad")
    with pytest.raises(ValueError, match="max_rows"):
        run_bootstrap_stability_diagnostics(features, labels, max_rows=0)


def test_format_bootstrap_stability_summary():
    features, labels = _boundary_dataset()
    report = run_bootstrap_stability_diagnostics(
        features,
        labels,
        model_count=4,
        train_fraction=0.75,
        seed=3,
        max_epochs=8,
        max_rows=3,
    )

    summary = format_bootstrap_stability_summary(report)

    assert "Bootstrap stability" in summary
    assert "models=4" in summary
    assert "top_row=" in summary
    assert "top_instability=" in summary
