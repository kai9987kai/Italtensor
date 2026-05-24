import numpy as np
import pytest

from italtensor.decision_curve import format_decision_curve_summary, run_decision_curve_diagnostics
from italtensor.preprocessing import FeatureStandardizer


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


def test_decision_curve_net_benefit_math_at_current_threshold():
    features = np.zeros((4, 1), dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int32)
    model = FixedProbabilityModel([0.9, 0.8, 0.7, 0.1])

    report = run_decision_curve_diagnostics(
        model,
        features,
        labels,
        current_threshold=0.5,
        grid_size=3,
        epsilon=0.1,
    )

    current = report["current"]
    assert current["threshold"] == pytest.approx(0.5)
    assert current["true_positive"] == 2
    assert current["false_positive"] == 1
    assert current["net_benefit_model"] == pytest.approx(0.25)
    assert current["net_benefit_treat_all"] == pytest.approx(0.0)
    assert current["delta_vs_best_default"] == pytest.approx(0.25)
    assert "Decision curve" in format_decision_curve_summary(report)


def test_decision_curve_detects_useful_threshold_ranges():
    features = np.zeros((4, 1), dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    model = FixedProbabilityModel([0.1, 0.2, 0.8, 0.9])

    report = run_decision_curve_diagnostics(model, features, labels, grid_size=5, epsilon=0.1)

    assert report["summary"]["useful_threshold_count"] > 0
    assert report["summary"]["max_delta_vs_best_default"] > 0.0
    assert report["summary"]["useful_threshold_ranges"]
    assert report["best"]["net_benefit_model"] >= report["best"]["net_benefit_treat_none"]


def test_decision_curve_uses_selected_preprocessor_once():
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

    report = run_decision_curve_diagnostics(
        FirstFeatureModel(),
        features,
        labels,
        preprocessor=preprocessor,
        current_threshold=0.5,
    )

    assert report["current"]["true_positive"] == 2
    assert report["current"]["false_positive"] == 0
    assert report["current"]["net_benefit_model"] == pytest.approx(0.5)


def test_decision_curve_clamps_threshold_endpoints_and_stays_finite():
    features = np.zeros((2, 1), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    model = FixedProbabilityModel([0.2, 0.8])

    report = run_decision_curve_diagnostics(model, features, labels, current_threshold=1.0, grid_size=4)

    assert report["effective_current_threshold"] < 1.0
    for point in report["points"]:
        assert 0.0 < point["threshold"] < 1.0
        assert np.isfinite(point["net_benefit_model"])
        assert np.isfinite(point["net_benefit_treat_all"])


def test_decision_curve_one_class_datasets_emit_warnings():
    features = np.zeros((3, 1), dtype=np.float32)

    negative = run_decision_curve_diagnostics(FixedProbabilityModel([0.1, 0.2, 0.3]), features, [0, 0, 0])
    positive = run_decision_curve_diagnostics(FixedProbabilityModel([0.7, 0.8, 0.9]), features, [1, 1, 1])

    assert "All labels are negative" in negative["summary"]["warning"]
    assert "All labels are positive" in positive["summary"]["warning"]


def test_decision_curve_validates_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_decision_curve_diagnostics(FixedProbabilityModel([0.5]), [1.0], [1])
    with pytest.raises(ValueError, match="counts"):
        run_decision_curve_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [1])
    with pytest.raises(ValueError, match="grid_size"):
        run_decision_curve_diagnostics(FixedProbabilityModel([0.5]), [[1.0]], [1], grid_size=1)
