from __future__ import annotations

import numpy as np
import pytest

from italtensor.population_drift import (
    format_population_drift_summary,
    run_population_drift_diagnostics,
)


def test_population_drift_ranks_shifted_feature_first():
    reference = np.asarray(
        [[-1.0, 0.0], [-0.5, 0.1], [0.0, -0.1], [0.5, 0.0], [1.0, 0.1]],
        dtype=np.float32,
    )
    current = np.asarray(
        [[-1.0, 2.0], [-0.5, 2.1], [0.0, 1.9], [0.5, 2.0], [1.0, 2.2]],
        dtype=np.float32,
    )
    labels = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]

    report = run_population_drift_diagnostics(np.vstack([reference, current]), labels, n_bins=4)

    assert report["reference_count"] == 5
    assert report["current_count"] == 5
    assert report["summary"]["top_feature"] == 1
    assert report["features"][0]["psi"] > report["features"][1]["psi"]
    assert "major_psi_shift" in report["features"][0]["risk_flags"]
    assert report["summary"]["label_prevalence_shift"] == pytest.approx(0.4)
    assert "Population drift" in format_population_drift_summary(report)


def test_population_drift_constant_reference_stays_finite_and_flags_outside():
    features = np.asarray(
        [[1.0], [1.0], [1.0], [1.0], [2.0], [2.5], [3.0], [3.5]],
        dtype=np.float32,
    )
    labels = [0, 0, 1, 1, 0, 1, 0, 1]

    report = run_population_drift_diagnostics(features, labels, n_bins=3)
    row = report["features"][0]

    assert np.isfinite(row["psi"])
    assert np.isfinite(row["ks_statistic"])
    assert np.isfinite(row["mean_shift_std"])
    assert row["outside_reference_rate"] == pytest.approx(1.0)
    assert "outside_reference_range" in row["risk_flags"]


def test_population_drift_odd_count_split_is_predictable():
    features = np.asarray([[float(index), 0.0] for index in range(7)], dtype=np.float32)
    labels = [0, 1, 0, 1, 0, 1, 0]

    report = run_population_drift_diagnostics(features, labels, reference_fraction=0.5)

    assert report["reference_count"] == 4
    assert report["current_count"] == 3


def test_population_drift_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_population_drift_diagnostics([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        run_population_drift_diagnostics([[0.0], [1.0], [2.0], [float("nan")], [4.0], [5.0]], [0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="counts"):
        run_population_drift_diagnostics([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], [0, 1])
    with pytest.raises(ValueError, match="at least six"):
        run_population_drift_diagnostics([[0.0], [1.0], [2.0], [3.0], [4.0]], [0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="binary"):
        run_population_drift_diagnostics([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], [0, 1, 2, 1, 0, 1])
    with pytest.raises(ValueError, match="reference_fraction"):
        run_population_drift_diagnostics([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], [0, 1, 0, 1, 0, 1], reference_fraction=0.95)
    with pytest.raises(ValueError, match="bins"):
        run_population_drift_diagnostics([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], [0, 1, 0, 1, 0, 1], n_bins=1)
