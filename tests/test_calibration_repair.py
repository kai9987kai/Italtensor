import numpy as np
import pytest

from italtensor.calibration_repair import (
    format_calibration_repair_summary,
    run_calibration_repair_diagnostics,
)


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(self, samples, verbose=0):
        sample_count = np.asarray(samples).shape[0]
        return self.probabilities[:sample_count].reshape(-1, 1)


class CountingPreprocessor:
    def __init__(self):
        self.calls = 0

    def transform(self, values):
        self.calls += 1
        return np.asarray(values, dtype=np.float32)


def test_calibration_repair_compares_raw_platt_and_isotonic():
    features = [[float(index)] for index in range(12)]
    labels = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    probabilities = [0.05, 0.1, 0.2, 0.35, 0.65, 0.8, 0.15, 0.3, 0.7, 0.82, 0.9, 0.96]

    report = run_calibration_repair_diagnostics(
        FixedProbabilityModel(probabilities),
        features,
        labels,
        seed=7,
        n_bins=4,
    )

    methods = {item["method"]: item for item in report["methods"]}
    assert set(methods) == {"raw", "platt", "isotonic"}
    assert report["split"]["source"] == "posthoc_stratified_split"
    assert report["summary"]["recommended_method"] in methods
    assert all("brier_score" in item and "ece" in item for item in methods.values())
    assert "Calibration repair" in format_calibration_repair_summary(report)


def test_calibration_repair_uses_preprocessor_once():
    preprocessor = CountingPreprocessor()

    report = run_calibration_repair_diagnostics(
        FixedProbabilityModel([0.05, 0.1, 0.8, 0.9]),
        [[0.0], [1.0], [2.0], [3.0]],
        [0, 0, 1, 1],
        preprocessor=preprocessor,  # type: ignore[arg-type]
        n_bins=2,
    )

    assert preprocessor.calls == 1
    assert report["split"]["calibration_count"] == 2


def test_calibration_repair_warns_for_one_class_global_split():
    report = run_calibration_repair_diagnostics(
        FixedProbabilityModel([0.05, 0.1, 0.2, 0.3]),
        [[0.0], [1.0], [2.0], [3.0]],
        [0, 0, 0, 0],
        n_bins=2,
    )

    assert report["split"]["source"] == "posthoc_global_split"
    assert "All labels are negative" in report["summary"]["warning"]
    assert report["summary"]["recommended_method"] in {"raw", "platt", "isotonic"}


def test_calibration_repair_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D array"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5]), [1.0], [1])
    with pytest.raises(ValueError, match="counts do not match"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [1])
    with pytest.raises(ValueError, match="at least two samples"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5]), [[1.0]], [1])
    with pytest.raises(ValueError, match="binary labels"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[float("nan")], [2.0]], [0, 1])
    with pytest.raises(ValueError, match="calibration_fraction"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 1], calibration_fraction=1.0)
    with pytest.raises(ValueError, match="n_bins"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 1], n_bins=1)


def test_calibration_repair_rejects_bad_model_probabilities():
    with pytest.raises(ValueError, match="probability count"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        run_calibration_repair_diagnostics(FixedProbabilityModel([float("nan"), 0.5]), [[1.0], [2.0]], [0, 1])
