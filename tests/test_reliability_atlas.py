import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.reliability_atlas import (
    format_reliability_atlas_summary,
    reliability_dataset_fingerprint,
    run_reliability_atlas,
)


class ProbabilityModel:
    def predict(self, samples, verbose=0):
        return np.asarray(samples, dtype=np.float32)[:, :1]


class ShortModel:
    def predict(self, samples, verbose=0):
        return np.asarray([[0.2]], dtype=np.float32)


class NanModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), np.nan, dtype=np.float32)


class ConstantModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 0.5, dtype=np.float32)


class OutOfRangeModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 1.2, dtype=np.float32)


def test_reliability_atlas_ranks_worst_bins_and_recommendations():
    features = np.asarray([[0.05], [0.15], [0.85], [0.95], [0.55], [0.65]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)

    report = run_reliability_atlas(ProbabilityModel(), features, labels, n_bins=5, min_bin_count=2)

    assert report["summary"]["expected_calibration_error"] > 0.0
    assert report["summary"]["max_calibration_error"] > 0.0
    assert report["dataset_fingerprint"] == reliability_dataset_fingerprint(features, labels)
    assert report["summary"]["clipped_probability_count"] == 0
    assert report["worst_bins"][0]["absolute_error"] >= report["worst_bins"][-1]["absolute_error"]
    assert report["highest_impact_bins"][0]["weighted_error"] >= report["highest_impact_bins"][-1]["weighted_error"]
    assert report["summary"]["risk_level"] in {"medium", "high"}
    assert report["recommendations"][0]["category"] in {"calibration", "bin_review"}
    assert format_reliability_atlas_summary(report).startswith("Reliability atlas:")


def test_reliability_atlas_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.9], [999.0, 0.8], [999.0, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_reliability_atlas(ProbabilityModel(), features, labels, preprocessor=preprocessor, n_bins=4)

    assert report["sample_count"] == 4
    assert report["input_dim"] == 2
    assert report["summary"]["brier_score"] < 0.1


def test_reliability_atlas_validates_model_output_and_labels():
    features = np.asarray([[0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_reliability_atlas(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_reliability_atlas(NanModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_reliability_atlas(ProbabilityModel(), features, np.asarray([0, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="binary"):
        run_reliability_atlas(ProbabilityModel(), features, np.asarray([0.2, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_reliability_atlas(ConstantModel(), np.asarray([[0.2], [np.inf]], dtype=np.float32), labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_reliability_atlas(OutOfRangeModel(), features, labels)
