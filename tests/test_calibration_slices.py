import numpy as np
import pytest

from italtensor.calibration_slices import (
    calibration_slice_dataset_fingerprint,
    format_calibration_slice_summary,
    run_calibration_slice_diagnostics,
)
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


def test_calibration_slices_find_local_overconfidence():
    features = np.asarray(
        [
            [0.05, 0.0],
            [0.15, 0.0],
            [0.85, 0.0],
            [0.95, 0.0],
            [0.78, 1.0],
            [0.82, 1.0],
            [0.88, 1.0],
            [0.92, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)

    report = run_calibration_slice_diagnostics(
        ProbabilityModel(),
        features,
        labels,
        bins=2,
        min_count=4,
        n_probability_bins=4,
    )

    assert report["dataset_fingerprint"] == calibration_slice_dataset_fingerprint(features, labels)
    assert report["summary"]["risk_level"] == "high"
    assert report["summary"]["max_absolute_confidence_gap"] >= 0.75
    assert any(
        item["feature_index"] == 1 and item["calibration_direction"] == "overconfident"
        for item in report["slices"]
    )
    assert format_calibration_slice_summary(report).startswith("Calibration slices:")


def test_calibration_slices_use_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.9], [999.0, 0.8], [999.0, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_calibration_slice_diagnostics(
        ProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        bins=2,
        min_count=2,
    )

    assert report["sample_count"] == 4
    assert report["input_dim"] == 2
    assert report["base"]["brier_score"] < 0.05
    assert all(item["feature_index"] == 1 for item in report["slices"])


def test_calibration_slices_validate_inputs_and_probabilities():
    features = np.asarray([[0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_calibration_slice_diagnostics(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_calibration_slice_diagnostics(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_calibration_slice_diagnostics(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_calibration_slice_diagnostics(ProbabilityModel(), features, np.asarray([0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_calibration_slice_diagnostics(ProbabilityModel(), np.asarray([[0.2], [np.inf]], dtype=np.float32), labels)
    with pytest.raises(ValueError, match="2D array"):
        run_calibration_slice_diagnostics(ProbabilityModel(), [0.2, 0.8], labels)


def test_calibration_slice_fingerprint_is_order_sensitive():
    features = np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    order = [2, 0, 3, 1]

    assert calibration_slice_dataset_fingerprint(features, labels) != calibration_slice_dataset_fingerprint(
        features[order],
        labels[order],
    )
