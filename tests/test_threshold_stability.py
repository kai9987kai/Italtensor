import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.threshold_stability import (
    format_threshold_stability_summary,
    run_threshold_stability,
    threshold_stability_dataset_fingerprint,
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


class OutOfRangeModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 1.2, dtype=np.float32)


def test_threshold_stability_flags_unstable_current_threshold_deterministically():
    probabilities = np.asarray([0.05, 0.10, 0.20, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90, 0.95], dtype=np.float32)
    features = probabilities.reshape(-1, 1)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)

    report = run_threshold_stability(
        ProbabilityModel(),
        features,
        labels,
        current_threshold=0.9,
        bootstrap_samples=24,
        seed=7,
        grid_size=21,
    )
    repeat = run_threshold_stability(
        ProbabilityModel(),
        features,
        labels,
        current_threshold=0.9,
        bootstrap_samples=24,
        seed=7,
        grid_size=21,
    )

    assert report["summary"] == repeat["summary"]
    assert report["summary"]["verdict"] == "unstable_threshold"
    assert report["summary"]["current_inside_interval"] is False
    assert report["summary"]["median_f1_gain_vs_current"] > 0.1
    assert report["summary"]["completed_bootstrap_count"] == 24
    assert report["summary"]["skipped_bootstrap_count"] == 0
    assert report["dataset_fingerprint"] == threshold_stability_dataset_fingerprint(features, labels)
    assert report["recommendations"][0]["category"] == "threshold"
    assert format_threshold_stability_summary(report).startswith("Threshold stability:")


def test_threshold_stability_fingerprint_is_order_sensitive():
    features = np.asarray([[0.1], [0.9], [0.2], [0.8], [0.3], [0.7]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
    order = [2, 0, 4, 5, 3, 1]

    assert threshold_stability_dataset_fingerprint(features, labels) != threshold_stability_dataset_fingerprint(
        features[order],
        labels[order],
    )


def test_threshold_stability_uses_selected_preprocessor_once():
    features = np.asarray(
        [[999.0, 0.05], [999.0, 0.15], [999.0, 0.35], [999.0, 0.65], [999.0, 0.85], [999.0, 0.95]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_threshold_stability(
        ProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        bootstrap_samples=12,
        seed=3,
    )

    assert report["sample_count"] == 6
    assert report["input_dim"] == 2
    assert report["full_dataset"]["best_f1"]["f1"] == pytest.approx(1.0)


def test_threshold_stability_validates_inputs_and_probabilities():
    features = np.asarray([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_threshold_stability(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_threshold_stability(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_threshold_stability(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_threshold_stability(ProbabilityModel(), features, np.asarray([0, 0, 0.5, 1, 1, 1], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_threshold_stability(ProbabilityModel(), np.asarray([[0.1], [0.2], [0.3], [0.7], [0.8], [np.inf]]), labels)
    with pytest.raises(ValueError, match="both classes"):
        run_threshold_stability(ProbabilityModel(), features, np.zeros(6, dtype=np.int32))
