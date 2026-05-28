import numpy as np
import pytest

from italtensor.error_atlas import format_error_atlas_summary, run_error_atlas
from italtensor.preprocessing import FeatureStandardizer


class ProbabilityModel:
    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)[:, 0]
        return values.reshape(-1, 1)


class ShortModel:
    def predict(self, samples, verbose=0):
        return np.asarray([[0.2]], dtype=np.float32)


class NanModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), np.nan, dtype=np.float32)


def test_error_atlas_splits_confusion_buckets_and_orders_errors():
    features = np.asarray([[0.95], [0.10], [0.90], [0.40], [0.51], [0.49]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 1, 0, 0], dtype=np.int32)

    report = run_error_atlas(ProbabilityModel(), features, labels, threshold=0.5)

    assert report["confusion"]["false_positive"] == 2
    assert report["confusion"]["false_negative"] == 2
    assert report["summary"]["high_confidence_error_count"] == 2
    assert report["high_confidence_errors"][0]["row_index"] == 0
    assert report["high_confidence_errors"][1]["row_index"] == 1
    assert report["near_threshold_rows"][0]["row_index"] == 4
    assert report["summary"]["dominant_error_type"] == "balanced_errors"
    assert report["feature_error_shifts"][0]["feature_index"] == 0
    assert "Error atlas:" in format_error_atlas_summary(report)


def test_error_atlas_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.9]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_error_atlas(ProbabilityModel(), features, labels, preprocessor=preprocessor)

    assert report["summary"]["error_count"] == 0
    assert report["buckets"]["true_negative"][0]["feature_preview"] == [999.0, 0.10000000149011612]


def test_error_atlas_validates_threshold_shape_and_probability_values():
    features = np.asarray([[0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="between 0 and 1"):
        run_error_atlas(ProbabilityModel(), features, labels, threshold=1.1)
    with pytest.raises(ValueError, match="different number"):
        run_error_atlas(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_error_atlas(NanModel(), features, labels)
