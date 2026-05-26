from __future__ import annotations

import numpy as np
import pytest

from italtensor.chronological_holdout import (
    format_chronological_holdout_summary,
    run_chronological_holdout_diagnostics,
)
from italtensor.presets import generate_builtin_preset


def test_chronological_holdout_detects_temporal_degradation_and_baseline_relearns():
    dataset = generate_builtin_preset("Chronological holdout lab", sample_count=180, seed=11)

    report = run_chronological_holdout_diagnostics(
        dataset.features,
        dataset.labels,
        seed=5,
        max_epochs=120,
        feature_map="linear",
    )

    assert report["split_source"] == "row_order_reference_then_current"
    assert report["reference_count"] == 108
    assert report["current_count"] == 72
    assert report["threshold"] == pytest.approx(0.5)
    assert report["permutation_repeats"] == 3
    assert report["summary"]["reference_f1"] >= 0.8
    assert report["summary"]["current_f1"] < report["summary"]["reference_f1"]
    assert report["summary"]["f1_delta"] <= -0.25
    assert report["summary"]["verdict"].startswith("severe_temporal_degradation")
    assert report["current_baseline"]["available"] is True
    assert report["current_baseline"]["metric_deltas_vs_reference_model"]["f1_delta"] > 0.1
    assert report["current_probability_diagnostics"]["current_low_confidence_rate"] >= 0.0
    assert np.isfinite(report["current_probability_diagnostics"]["current_ece"])
    assert report["current_probability_diagnostics"]["current_calibration_bins"]
    assert report["permutation_reliance"][0]["risk_score"] >= 0.0
    assert report["permutation_reliance"][0]["permutation_repeats"] == 3
    assert report["permutation_reliance"][0]["feature_index"] != 4
    assert report["current_feature_reliance"] == report["permutation_reliance"]
    assert "Chronological holdout" in format_chronological_holdout_summary(report)

    repeated = run_chronological_holdout_diagnostics(
        dataset.features,
        dataset.labels,
        seed=5,
        max_epochs=120,
        feature_map="linear",
    )
    assert repeated["permutation_reliance"] == report["permutation_reliance"]


def test_chronological_holdout_handles_small_current_baseline_unavailable():
    features = np.asarray(
        [[-1.0], [-0.9], [0.9], [1.0], [-1.1], [-0.8], [0.8], [1.1], [-1.2], [1.2], [-1.3], [1.3], [0.0], [0.1], [0.2], [0.3]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)

    report = run_chronological_holdout_diagnostics(
        features,
        labels,
        reference_fraction=0.75,
        seed=2,
        max_epochs=10,
    )

    assert report["current_count"] == 4
    assert report["current_baseline"]["available"] is False
    assert "current-only baseline unavailable" in report["summary"]["warning"]
    assert "current-only baseline unavailable" in report["warnings"]


def test_chronological_holdout_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_chronological_holdout_diagnostics([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        run_chronological_holdout_diagnostics(
            [[float(index)] for index in range(15)] + [[float("nan")]],
            [0, 1] * 8,
        )
    with pytest.raises(ValueError, match="counts"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1])
    with pytest.raises(ValueError, match="at least sixteen"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(12)], [0, 1] * 6)
    with pytest.raises(ValueError, match="binary"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1, 2, 1] * 4)
    with pytest.raises(ValueError, match="integer binary"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0.0, 0.9] * 8)
    with pytest.raises(ValueError, match="reference_fraction"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1] * 8, reference_fraction=0.9)
    with pytest.raises(ValueError, match="reference_validation_fraction"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1] * 8, reference_validation_fraction=0.9)
    with pytest.raises(ValueError, match="threshold"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1] * 8, threshold=1.0)
    with pytest.raises(ValueError, match="permutation_repeats"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1] * 8, permutation_repeats=0)
    with pytest.raises(ValueError, match="feature_map"):
        run_chronological_holdout_diagnostics([[float(index)] for index in range(16)], [0, 1] * 8, feature_map="bad")
