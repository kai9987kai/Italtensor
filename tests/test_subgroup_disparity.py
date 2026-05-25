import numpy as np
import pytest

from italtensor.subgroup_disparity import (
    format_subgroup_disparity_summary,
    run_subgroup_disparity_diagnostics,
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


def test_subgroup_disparity_finds_marker_group_gap():
    features = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
    ]
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.2, 0.3]

    report = run_subgroup_disparity_diagnostics(
        FixedProbabilityModel(probabilities),
        features,
        labels,
        threshold=0.5,
    )

    marker_groups = [item for item in report["subgroups"] if item["label"] == "x2=1"]
    assert marker_groups
    assert marker_groups[0]["recall_gap"] == pytest.approx(1.0)
    assert marker_groups[0]["false_negative_rate_gap"] == pytest.approx(1.0)
    assert "large_gap" in marker_groups[0]["risk_flags"]
    assert report["summary"]["max_false_negative_rate_gap"] == pytest.approx(1.0)
    assert "Subgroup disparity" in format_subgroup_disparity_summary(report)


def test_subgroup_disparity_uses_preprocessor_once():
    preprocessor = CountingPreprocessor()

    report = run_subgroup_disparity_diagnostics(
        FixedProbabilityModel([0.1, 0.2, 0.8, 0.9]),
        [[0.0], [0.0], [1.0], [1.0]],
        [0, 0, 1, 1],
        preprocessor=preprocessor,  # type: ignore[arg-type]
    )

    assert preprocessor.calls == 1
    assert report["summary"]["evaluated_subgroup_count"] >= 2


def test_subgroup_disparity_warns_when_no_usable_groups():
    report = run_subgroup_disparity_diagnostics(
        FixedProbabilityModel([0.1, 0.2, 0.3, 0.4]),
        [[1.0], [1.0], [1.0], [1.0]],
        [0, 0, 0, 0],
    )

    assert report["summary"]["evaluated_subgroup_count"] == 0
    assert "No feature produced" in report["summary"]["warning"]
    assert "All labels are negative" in report["summary"]["warning"]


def test_subgroup_disparity_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D array"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5]), [1.0], [1])
    with pytest.raises(ValueError, match="counts do not match"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [1])
    with pytest.raises(ValueError, match="at least two samples"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5]), [[1.0]], [1])
    with pytest.raises(ValueError, match="binary labels"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[float("nan")], [2.0]], [0, 1])
    with pytest.raises(ValueError, match="threshold"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5, 0.6]), [[1.0], [2.0]], [0, 1], threshold=1.5)


def test_subgroup_disparity_rejects_bad_model_probabilities():
    with pytest.raises(ValueError, match="probability count"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([0.5]), [[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        run_subgroup_disparity_diagnostics(FixedProbabilityModel([float("nan"), 0.5]), [[1.0], [2.0]], [0, 1])
