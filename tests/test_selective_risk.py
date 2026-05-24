import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.selective_risk import format_selective_risk_summary, run_selective_risk_diagnostics


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)
        return self.probabilities[: values.shape[0]].reshape(-1, 1)


class FirstFeatureModel:
    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)
        return np.where(values[:, 0] > 0.0, 0.9, 0.1).reshape(-1, 1)


def test_selective_risk_cutoff_zero_matches_base_metrics():
    features = np.zeros((4, 1), dtype=np.float32)
    labels = np.asarray([1, 0, 0, 1], dtype=np.int32)
    report = run_selective_risk_diagnostics(
        FixedProbabilityModel([0.9, 0.2, 0.55, 0.45]),
        features,
        labels,
        threshold=0.5,
        grid_size=5,
    )

    point_zero = next(point for point in report["points"] if point["confidence_cutoff"] == 0.0)
    assert point_zero["coverage"] == 1.0
    assert point_zero["accuracy"] == pytest.approx(report["base"]["accuracy"])
    assert point_zero["error_rate"] == pytest.approx(report["base"]["error_rate"])
    assert "Selective risk" in format_selective_risk_summary(report)


def test_selective_risk_abstaining_low_confidence_wrong_rows_improves_accuracy():
    features = np.zeros((4, 1), dtype=np.float32)
    labels = np.asarray([1, 0, 0, 1], dtype=np.int32)
    report = run_selective_risk_diagnostics(
        FixedProbabilityModel([0.9, 0.2, 0.55, 0.45]),
        features,
        labels,
        threshold=0.5,
        grid_size=5,
    )

    ranked = report["ranked_cutoffs"][0]
    assert report["base"]["accuracy"] == pytest.approx(0.5)
    assert ranked["coverage"] == pytest.approx(0.5)
    assert ranked["accuracy"] == pytest.approx(1.0)
    assert ranked["error_rate"] == pytest.approx(0.0)
    assert report["summary"]["max_error_reduction"] == pytest.approx(0.5)


def test_selective_risk_uses_selected_preprocessor_once():
    features = np.asarray(
        [
            [999.0, -2.0],
            [999.0, -1.0],
            [999.0, 1.0],
            [999.0, 2.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_selective_risk_diagnostics(
        FirstFeatureModel(),
        features,
        labels,
        preprocessor=preprocessor,
        threshold=0.5,
    )

    assert report["base"]["accuracy"] == 1.0
    assert report["summary"]["min_selective_risk"] == 0.0


def test_selective_risk_clamps_threshold_endpoints_and_warns_on_one_class():
    features = np.zeros((3, 1), dtype=np.float32)
    negative = run_selective_risk_diagnostics(FixedProbabilityModel([0.0, 0.0, 0.0]), features, [0, 0, 0], threshold=0.0)
    positive = run_selective_risk_diagnostics(FixedProbabilityModel([1.0, 1.0, 1.0]), features, [1, 1, 1], threshold=1.0)

    assert negative["effective_threshold"] > 0.0
    assert positive["effective_threshold"] < 1.0
    assert "All labels are negative" in negative["summary"]["warning"]
    assert "All labels are positive" in positive["summary"]["warning"]


def test_selective_risk_validates_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_selective_risk_diagnostics(FixedProbabilityModel([0.5]), [1.0], [1])
    with pytest.raises(ValueError, match="counts"):
        run_selective_risk_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [1])
    with pytest.raises(ValueError, match="grid_size"):
        run_selective_risk_diagnostics(FixedProbabilityModel([0.5]), [[1.0]], [1], grid_size=1)
