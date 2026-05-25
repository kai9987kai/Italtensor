from __future__ import annotations

import numpy as np
import pytest

from italtensor.adversarial_validation import (
    format_adversarial_validation_summary,
    run_adversarial_validation_diagnostics,
)


def test_adversarial_validation_detects_multivariate_shift():
    rng = np.random.default_rng(4)
    reference_axis = rng.normal(0.0, 1.0, size=50)
    current_axis = rng.normal(0.0, 1.0, size=50)
    reference = np.column_stack(
        [
            rng.normal(0.0, 1.0, size=50),
            reference_axis,
            reference_axis + rng.normal(0.0, 0.1, size=50),
        ]
    )
    current = np.column_stack(
        [
            rng.normal(0.0, 1.0, size=50),
            current_axis,
            -current_axis + rng.normal(0.0, 0.1, size=50),
        ]
    )
    features = np.vstack([reference, current]).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int32)

    report = run_adversarial_validation_diagnostics(features, labels, seed=3, max_epochs=120)

    assert report["reference_count"] == 50
    assert report["current_count"] == 50
    assert report["summary"]["detectability"] >= 0.75
    assert report["summary"]["verdict"] in {"moderate_multivariate_shift", "strong_multivariate_shift"}
    assert report["features"][0]["risk_score"] >= 0.0
    assert "Adversarial validation" in format_adversarial_validation_summary(report)


def test_adversarial_validation_no_shift_stays_finite():
    rng = np.random.default_rng(5)
    reference = rng.normal(0.0, 1.0, size=(30, 3))
    current = rng.normal(0.0, 1.0, size=(30, 3))
    features = np.vstack([reference, current]).astype(np.float32)
    labels = np.asarray([0, 1] * 30, dtype=np.int32)

    report = run_adversarial_validation_diagnostics(features, labels, seed=8, max_epochs=30)

    assert np.isfinite(report["summary"]["domain_auc"])
    assert np.isfinite(report["summary"]["detectability"])
    assert report["summary"]["verdict"] in {
        "no_detectable_multivariate_shift",
        "weak_multivariate_shift",
        "moderate_multivariate_shift",
        "strong_multivariate_shift",
    }


def test_adversarial_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_adversarial_validation_diagnostics([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        run_adversarial_validation_diagnostics(
            [[float(index)] for index in range(11)] + [[float("nan")]],
            [0, 1] * 6,
        )
    with pytest.raises(ValueError, match="counts"):
        run_adversarial_validation_diagnostics([[float(index)] for index in range(12)], [0, 1])
    with pytest.raises(ValueError, match="at least twelve"):
        run_adversarial_validation_diagnostics([[float(index)] for index in range(10)], [0, 1] * 5)
    with pytest.raises(ValueError, match="binary"):
        run_adversarial_validation_diagnostics([[float(index)] for index in range(12)], [0, 1, 2] * 4)
    with pytest.raises(ValueError, match="reference_fraction"):
        run_adversarial_validation_diagnostics([[float(index)] for index in range(12)], [0, 1] * 6, reference_fraction=0.95)
    with pytest.raises(ValueError, match="validation_fraction"):
        run_adversarial_validation_diagnostics([[float(index)] for index in range(12)], [0, 1] * 6, validation_fraction=0.9)
