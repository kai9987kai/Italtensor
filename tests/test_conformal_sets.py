import numpy as np
import pytest

from italtensor.conformal_sets import format_conformal_set_summary, run_conformal_set_diagnostics
from italtensor.preprocessing import FeatureStandardizer


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


def test_conformal_set_diagnostics_sweeps_alpha_prediction_sets():
    features = [[float(index), float(index % 2)] for index in range(8)]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probabilities = [0.02, 0.08, 0.25, 0.42, 0.58, 0.75, 0.9, 0.98]

    report = run_conformal_set_diagnostics(
        FixedProbabilityModel(probabilities),
        features,
        labels,
        alphas=[0.2, 0.1],
        seed=3,
    )

    assert report["split"]["source"] == "posthoc_stratified_split"
    assert report["split"]["calibration_count"] == 4
    assert report["split"]["evaluation_count"] == 4
    assert [point["alpha"] for point in report["points"]] == [0.1, 0.2]
    assert 0.0 <= report["summary"]["recommended_empirical_coverage"] <= 1.0
    assert 0.0 <= report["summary"]["recommended_singleton_rate"] <= 1.0
    assert "Conformal sets" in format_conformal_set_summary(report)


def test_conformal_set_diagnostics_uses_preprocessor_once():
    features = [[-1.0], [-0.5], [0.5], [1.0]]
    labels = [0, 0, 1, 1]
    preprocessor = CountingPreprocessor()

    report = run_conformal_set_diagnostics(
        FixedProbabilityModel([0.1, 0.2, 0.8, 0.9]),
        features,
        labels,
        preprocessor=preprocessor,  # type: ignore[arg-type]
        alphas=[0.25],
    )

    assert preprocessor.calls == 1
    assert report["points"][0]["calibration_count"] == 2


def test_conformal_set_diagnostics_warns_for_one_class_global_split():
    report = run_conformal_set_diagnostics(
        FixedProbabilityModel([0.05, 0.1, 0.2, 0.3]),
        [[0.0], [1.0], [2.0], [3.0]],
        [0, 0, 0, 0],
        alphas=[0.1],
    )

    warning = report["summary"]["warning"]
    assert report["split"]["source"] == "posthoc_global_split"
    assert warning is not None
    assert "All labels are negative" in warning
    assert report["points"][0]["positive_coverage"] is None


def test_conformal_set_diagnostics_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D array"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5]), [1.0], [1])
    with pytest.raises(ValueError, match="counts do not match"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [1])
    with pytest.raises(ValueError, match="at least two samples"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5]), [[1.0]], [1])
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 1], alphas=[0.0])
    with pytest.raises(ValueError, match="binary labels"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[float("nan")], [2.0]], [0, 1])


def test_conformal_set_diagnostics_rejects_bad_model_probabilities():
    with pytest.raises(ValueError, match="probability count"):
        run_conformal_set_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        run_conformal_set_diagnostics(FixedProbabilityModel([float("nan"), 0.5]), [[1.0], [2.0]], [0, 1])


def test_conformal_set_diagnostics_accepts_standardizer():
    standardizer = FeatureStandardizer.fit(np.asarray([[0.0], [1.0]], dtype=np.float32))

    report = run_conformal_set_diagnostics(
        FixedProbabilityModel([0.1, 0.2, 0.8, 0.9]),
        [[0.0], [0.2], [0.8], [1.0]],
        [0, 0, 1, 1],
        preprocessor=standardizer,
        alphas=[0.2],
    )

    assert report["input_dim"] == 1
    assert report["summary"]["recommended_alpha"] == pytest.approx(0.2)
