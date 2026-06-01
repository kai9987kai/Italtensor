import numpy as np
import pytest

from italtensor.external_holdout import format_external_holdout_summary, run_external_holdout_evaluation
from italtensor.preprocessing import FeatureStandardizer


class ProbabilityModel:
    def predict(self, samples, verbose=0):
        return np.asarray(samples, dtype=np.float32)[:, :1]


class ShortModel:
    def predict(self, samples, verbose=0):
        return np.asarray([[0.2]], dtype=np.float32)


class NanModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), np.nan, dtype=np.float32)


class OutOfRangeModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 1.2, dtype=np.float32)


def test_external_holdout_scores_labeled_csv_shape_without_mutating_reference():
    holdout_features = np.asarray([[0.05, 1.0], [0.15, 1.0], [0.85, 1.0], [0.95, 1.0]], dtype=np.float32)
    holdout_labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    reference_features = np.asarray([[0.05, 0.0], [0.15, 0.0], [0.85, 0.0], [0.95, 0.0]], dtype=np.float32)
    reference_labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = run_external_holdout_evaluation(
        ProbabilityModel(),
        holdout_features,
        holdout_labels,
        threshold=0.5,
        reference_features=reference_features,
        reference_labels=reference_labels,
    )

    assert report["summary"]["verdict"] in {"holdout_pass", "holdout_shift_review"}
    assert report["metrics"]["f1"] == pytest.approx(1.0)
    assert report["probability_diagnostics"]["brier_score"] < 0.05
    assert report["reference_comparison"]["top_shift_feature"] == 1
    assert report["reference_comparison"]["max_standardized_mean_shift"] > 0.0
    assert format_external_holdout_summary(report).startswith("External holdout:")


def test_external_holdout_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.9], [999.0, 0.8], [999.0, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_external_holdout_evaluation(
        ProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        threshold=0.5,
    )

    assert report["input_dim"] == 2
    assert report["metrics"]["f1"] == pytest.approx(1.0)


def test_external_holdout_flags_poor_external_performance():
    report = run_external_holdout_evaluation(
        ProbabilityModel(),
        np.asarray([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        threshold=0.5,
    )

    assert report["summary"]["verdict"] == "holdout_failure"
    assert report["recommendations"][0]["priority"] == "high"


def test_external_holdout_validates_inputs_and_model_output():
    features = np.asarray([[0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_external_holdout_evaluation(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_external_holdout_evaluation(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_external_holdout_evaluation(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_external_holdout_evaluation(ProbabilityModel(), features, np.asarray([0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_external_holdout_evaluation(ProbabilityModel(), np.asarray([[0.2], [np.inf]], dtype=np.float32), labels)
    with pytest.raises(ValueError, match="2D array"):
        run_external_holdout_evaluation(ProbabilityModel(), [0.2, 0.8], labels)
    with pytest.raises(ValueError, match="threshold"):
        run_external_holdout_evaluation(ProbabilityModel(), features, labels, threshold=-0.1)
    with pytest.raises(ValueError, match="reference feature count"):
        run_external_holdout_evaluation(
            ProbabilityModel(),
            features,
            labels,
            reference_features=np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32),
            reference_labels=labels,
        )
