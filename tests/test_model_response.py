import numpy as np
import pytest

from italtensor.model_response import format_model_response_summary, run_model_response_diagnostics


class LinearProbabilityModel:
    def predict(self, samples, verbose=0):
        x = np.asarray(samples, dtype=np.float32)
        logits = 1.5 * x[:, 0] - 1.2 * np.square(x[:, 1])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities.reshape(-1, 1)


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


def test_model_response_ranks_high_impact_feature_and_detects_direction():
    features = np.asarray(
        [[-1.5, -1.0], [-0.8, -0.5], [0.0, 0.0], [0.8, 0.5], [1.5, 1.0]],
        dtype=np.float32,
    )
    labels = [0, 0, 1, 1, 1]

    report = run_model_response_diagnostics(LinearProbabilityModel(), features, labels, grid_size=7)

    assert report["summary"]["top_feature"] == 0
    assert report["features"][0]["response_range"] > 0.4
    assert report["features"][0]["direction"] == "increasing"
    assert "high_impact" in report["features"][0]["risk_flags"]
    assert "Model response" in format_model_response_summary(report)


def test_model_response_can_detect_nonmonotonic_curve():
    features = np.asarray(
        [[0.0, -1.8], [0.0, -0.9], [0.0, 0.0], [0.0, 0.9], [0.0, 1.8]],
        dtype=np.float32,
    )
    labels = [0, 1, 1, 1, 0]

    report = run_model_response_diagnostics(LinearProbabilityModel(), features, labels, grid_size=9)
    nonlinear = next(item for item in report["features"] if item["feature_index"] == 1)

    assert nonlinear["direction"] == "nonmonotonic"
    assert nonlinear["direction_changes"] >= 1
    assert "nonmonotonic" in nonlinear["risk_flags"]


def test_model_response_applies_preprocessor_once_to_stacked_grid():
    preprocessor = CountingPreprocessor()

    report = run_model_response_diagnostics(
        LinearProbabilityModel(),
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        [0, 1, 1],
        preprocessor=preprocessor,  # type: ignore[arg-type]
        grid_size=5,
    )

    assert preprocessor.calls == 1
    assert report["grid_size"] == 5


def test_model_response_handles_constant_feature():
    report = run_model_response_diagnostics(
        LinearProbabilityModel(),
        [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
        [0, 1, 1],
        grid_size=5,
    )

    constant_feature = next(item for item in report["features"] if item["feature_index"] == 0)
    assert constant_feature["direction"] == "flat"
    assert constant_feature["response_range"] == pytest.approx(0.0)


def test_model_response_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D array"):
        run_model_response_diagnostics(LinearProbabilityModel(), [1.0], [1])
    with pytest.raises(ValueError, match="counts do not match"):
        run_model_response_diagnostics(LinearProbabilityModel(), [[1.0], [2.0]], [1])
    with pytest.raises(ValueError, match="at least one sample"):
        run_model_response_diagnostics(LinearProbabilityModel(), np.empty((0, 1)), [])
    with pytest.raises(ValueError, match="binary labels"):
        run_model_response_diagnostics(LinearProbabilityModel(), [[1.0], [2.0]], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        run_model_response_diagnostics(LinearProbabilityModel(), [[float("nan")]], [0])
    with pytest.raises(ValueError, match="grid_size"):
        run_model_response_diagnostics(LinearProbabilityModel(), [[1.0]], [0], grid_size=2)


def test_model_response_rejects_bad_model_probabilities():
    with pytest.raises(ValueError, match="probability count"):
        run_model_response_diagnostics(FixedProbabilityModel([0.5]), [[1.0]], [0], grid_size=3)
    with pytest.raises(ValueError, match="non-finite"):
        run_model_response_diagnostics(FixedProbabilityModel([0.5, float("nan"), 0.5, 0.5]), [[1.0]], [0], grid_size=3)
