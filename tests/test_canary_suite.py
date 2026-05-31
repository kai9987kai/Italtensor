import numpy as np
import pytest

from italtensor.canary_suite import format_canary_suite_summary, run_canary_suite
from italtensor.preprocessing import FeatureStandardizer
from italtensor.schema_guard import run_schema_guard


class FirstColumnProbabilityModel:
    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values[:, :1]


class ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return np.full((values.shape[0], 1), self.probability, dtype=np.float32)


def test_canary_suite_scores_expected_and_informational_examples():
    report = run_canary_suite(
        FirstColumnProbabilityModel(),
        [
            {"name": "negative", "features": [0.1], "expected_label": 0},
            {"name": "positive", "features": [0.9], "expected_label": 1},
            {"name": "review", "features": [0.45], "expected_label": None},
        ],
        input_dim=1,
        threshold=0.5,
    )

    assert report["summary"]["verdict"] == "canary_pass"
    assert report["summary"]["passed_count"] == 2
    assert report["summary"]["failed_count"] == 0
    assert report["summary"]["informational_count"] == 1
    assert report["examples"][2]["passed"] is None
    assert format_canary_suite_summary(report).startswith("Canary suite:")


def test_canary_suite_marks_expected_label_mismatch_as_failure():
    report = run_canary_suite(
        FirstColumnProbabilityModel(),
        [{"name": "regression", "features": [0.9], "expected_label": 0}],
        input_dim=1,
        threshold=0.5,
    )

    assert report["summary"]["verdict"] == "canary_fail"
    assert report["summary"]["failed_count"] == 1
    assert report["examples"][0]["status"] == "fail"


def test_canary_suite_uses_raw_vector_preprocessor_before_prediction():
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_canary_suite(
        FirstColumnProbabilityModel(),
        [{"name": "selected positive", "features": [99.0, 0.8], "expected_label": 1}],
        input_dim=2,
        threshold=0.5,
        preprocessor=preprocessor,
    )

    assert report["examples"][0]["probability"] == pytest.approx(0.8)
    assert report["examples"][0]["passed"] is True


def test_canary_suite_schema_guard_failures_affect_verdict():
    schema = run_schema_guard([[0.0], [0.2], [0.8], [1.0]])

    report = run_canary_suite(
        ConstantProbabilityModel(0.8),
        [{"name": "outside schema", "features": [1.2], "expected_label": 1}],
        input_dim=1,
        threshold=0.5,
        schema_guard_report=schema,
    )

    assert report["summary"]["verdict"] == "canary_fail"
    assert report["summary"]["schema_failure_count"] == 1
    assert report["examples"][0]["status"] == "schema_fail"
    assert report["examples"][0]["schema_status"] == "fail"


def test_canary_suite_review_when_all_examples_are_informational():
    report = run_canary_suite(
        FirstColumnProbabilityModel(),
        [{"name": "review only", "features": [0.3], "expected_label": None}],
        input_dim=1,
        threshold=0.5,
    )

    assert report["summary"]["verdict"] == "no_checkable_canaries"
    assert report["summary"]["checked_count"] == 0
    assert report["summary"]["pass_rate"] is None


def test_canary_suite_rejects_invalid_examples():
    with pytest.raises(ValueError, match="at least one"):
        run_canary_suite(FirstColumnProbabilityModel(), [], input_dim=1)
    with pytest.raises(ValueError, match="1 feature"):
        run_canary_suite(
            FirstColumnProbabilityModel(),
            [{"name": "bad width", "features": [0.1, 0.2], "expected_label": 1}],
            input_dim=1,
        )
    with pytest.raises(ValueError, match="finite"):
        run_canary_suite(
            FirstColumnProbabilityModel(),
            [{"name": "bad value", "features": [float("nan")], "expected_label": 1}],
            input_dim=1,
        )
    with pytest.raises(ValueError, match="expected_label"):
        run_canary_suite(
            FirstColumnProbabilityModel(),
            [{"name": "bad label", "features": [0.8], "expected_label": 2}],
            input_dim=1,
        )
