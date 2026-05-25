from __future__ import annotations

import numpy as np
import pytest

from italtensor.permutation_null import (
    format_permutation_null_summary,
    run_permutation_null_diagnostics,
)


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.calls = 0

    def predict(self, samples, verbose=0):
        self.calls += 1
        sample_count = np.asarray(samples).shape[0]
        return self.probabilities[:sample_count]


class CountingPreprocessor:
    def __init__(self):
        self.calls = 0

    def transform(self, features):
        self.calls += 1
        return np.asarray(features, dtype=np.float32) * 2.0


def test_permutation_null_detects_signal_deterministically():
    features = np.asarray([[float(index), 0.0] for index in range(8)], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    model = FixedProbabilityModel([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95])

    report = run_permutation_null_diagnostics(
        model,
        features,
        labels,
        permutation_count=80,
        seed=7,
    )
    repeated = run_permutation_null_diagnostics(
        FixedProbabilityModel(model.probabilities),
        features,
        labels,
        permutation_count=80,
        seed=7,
    )

    assert report["observed"]["f1"] == pytest.approx(1.0)
    assert report["summary"]["f1_p_value"] <= 0.05
    assert report["summary"]["verdict"] in {"signal", "strong_signal"}
    assert report["null_distribution"]["f1"]["mean"] == pytest.approx(repeated["null_distribution"]["f1"]["mean"])
    assert "Permutation null" in format_permutation_null_summary(report)


def test_permutation_null_applies_preprocessor_and_predicts_once():
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    model = FixedProbabilityModel([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    preprocessor = CountingPreprocessor()

    report = run_permutation_null_diagnostics(
        model,
        features,
        labels,
        preprocessor=preprocessor,
        permutation_count=20,
    )

    assert report["sample_count"] == 6
    assert preprocessor.calls == 1
    assert model.calls == 1


def test_permutation_null_warns_for_constant_predictions():
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    model = FixedProbabilityModel([0.1] * 6)

    report = run_permutation_null_diagnostics(model, features, labels, permutation_count=20)

    assert report["predicted_class_counts"]["1"] == 0
    assert "one class" in report["summary"]["warning"]
    assert np.isfinite(report["null_distribution"]["f1"]["mean"])


def test_permutation_null_rejects_invalid_inputs():
    model = FixedProbabilityModel([0.5, 0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="2D"):
        run_permutation_null_diagnostics(model, [1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="counts"):
        run_permutation_null_diagnostics(model, [[1.0], [2.0], [3.0], [4.0]], [0, 1])
    with pytest.raises(ValueError, match="at least four"):
        run_permutation_null_diagnostics(model, [[1.0], [2.0], [3.0]], [0, 1, 0])
    with pytest.raises(ValueError, match="binary"):
        run_permutation_null_diagnostics(model, [[1.0], [2.0], [3.0], [4.0]], [0, 1, 2, 1])
    with pytest.raises(ValueError, match="both classes"):
        run_permutation_null_diagnostics(model, [[1.0], [2.0], [3.0], [4.0]], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="threshold"):
        run_permutation_null_diagnostics(model, [[1.0], [2.0], [3.0], [4.0]], [0, 1, 0, 1], threshold=1.5)
    with pytest.raises(ValueError, match="permutations"):
        run_permutation_null_diagnostics(model, [[1.0], [2.0], [3.0], [4.0]], [0, 1, 0, 1], permutation_count=9)


def test_permutation_null_rejects_bad_model_probabilities():
    with pytest.raises(ValueError, match="probability count"):
        run_permutation_null_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0], [3.0], [4.0]], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        run_permutation_null_diagnostics(
            FixedProbabilityModel([0.1, float("nan"), 0.8, 0.9]),
            [[1.0], [2.0], [3.0], [4.0]],
            [0, 0, 1, 1],
        )
