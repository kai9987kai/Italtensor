import numpy as np
import pytest

from italtensor.policy_guard import format_policy_guard_summary, run_policy_guard, sanitize_policy_checks
from italtensor.preprocessing import FeatureStandardizer


class LinearProbabilityModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = float(bias)

    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        probabilities = np.clip(values @ self.weights + self.bias, 0.0, 1.0)
        return probabilities.reshape(-1, 1)


def test_policy_guard_passes_monotonic_increasing_and_decreasing_checks():
    features = [[0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0.9, 0.1]]
    report = run_policy_guard(
        LinearProbabilityModel([0.5, -0.4]),
        features,
        [
            {"name": "risk up", "feature_index": 0, "direction": "increasing"},
            {"name": "protection down", "feature_index": 1, "direction": "decreasing"},
        ],
        input_dim=2,
        step_fraction=0.1,
    )

    assert report["summary"]["verdict"] == "policy_pass"
    assert report["summary"]["violation_count"] == 0
    assert {item["status"] for item in report["checks"]} == {"pass"}
    assert format_policy_guard_summary(report).startswith("Policy guard:")


def test_policy_guard_fails_when_probability_moves_against_direction():
    features = [[0.1], [0.2], [0.5], [0.8], [0.9]]
    report = run_policy_guard(
        LinearProbabilityModel([-0.5], bias=0.9),
        features,
        [{"name": "risk should rise", "feature_index": 0, "direction": "increasing"}],
        input_dim=1,
        step_fraction=0.2,
    )

    assert report["summary"]["verdict"] == "policy_fail"
    assert report["summary"]["failed_check_count"] == 1
    assert report["checks"][0]["violation_count"] > 0


def test_policy_guard_reports_low_variance_feature_as_not_testable():
    report = run_policy_guard(
        LinearProbabilityModel([0.5]),
        [[1.0], [1.0], [1.0]],
        [{"name": "constant", "feature_index": 0, "direction": "increasing"}],
        input_dim=1,
    )

    assert report["summary"]["verdict"] == "policy_review"
    assert report["checks"][0]["status"] == "not_testable"
    assert "too little" in report["checks"][0]["warning"]


def test_policy_guard_uses_preprocessor_and_flags_dropped_selected_feature():
    features = [[0.1, 0.2], [0.3, 0.4], [0.7, 0.8], [0.9, 1.0]]
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )
    report = run_policy_guard(
        LinearProbabilityModel([0.5]),
        features,
        [
            {"name": "selected feature", "feature_index": 1, "direction": "increasing"},
            {"name": "dropped feature", "feature_index": 0, "direction": "increasing"},
        ],
        input_dim=2,
        preprocessor=preprocessor,
        step_fraction=0.1,
    )

    assert report["checks"][0]["status"] == "pass"
    assert report["checks"][1]["status"] == "not_testable"
    assert "not used" in report["checks"][1]["warning"]
    assert report["summary"]["verdict"] == "policy_review"


def test_policy_guard_rejects_invalid_inputs_and_sanitizes_aliases():
    checks = sanitize_policy_checks(
        [{"name": "alias", "feature_index": "0", "direction": "+1"}],
        input_dim=1,
    )
    assert checks[0]["direction"] == "increasing"

    with pytest.raises(ValueError, match="at least one"):
        run_policy_guard(LinearProbabilityModel([0.5]), [[0.1], [0.2]], [], input_dim=1)
    with pytest.raises(ValueError, match="out of bounds"):
        sanitize_policy_checks([{"feature_index": 2, "direction": "increasing"}], input_dim=1)
    with pytest.raises(ValueError, match="direction"):
        sanitize_policy_checks([{"feature_index": 0, "direction": "sideways"}], input_dim=1)
    with pytest.raises(ValueError, match="finite"):
        run_policy_guard(
            LinearProbabilityModel([0.5]),
            [[0.1], [float("nan")]],
            [{"feature_index": 0, "direction": "increasing"}],
            input_dim=1,
        )
