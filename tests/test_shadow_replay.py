import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.shadow_replay import (
    format_shadow_replay_summary,
    run_shadow_replay,
    shadow_replay_dataset_fingerprint,
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


def test_shadow_replay_flags_ordered_degradation_and_error_runs():
    early_prob = np.asarray([0.05, 0.12, 0.88, 0.92, 0.08, 0.95, 0.15, 0.85], dtype=np.float32)
    late_prob = np.asarray([0.90, 0.88, 0.12, 0.10, 0.85, 0.80, 0.20, 0.15], dtype=np.float32)
    features = np.concatenate([early_prob, late_prob]).reshape(-1, 1)
    labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1] * 2, dtype=np.int32)

    report = run_shadow_replay(ProbabilityModel(), features, labels, window_count=2, min_window_size=4)

    assert report["summary"]["verdict"] == "severe_ordered_degradation"
    assert report["summary"]["max_f1_drop"] > 0.5
    assert report["dataset_fingerprint"] == shadow_replay_dataset_fingerprint(features, labels)
    assert len(report["segments"]) == len(report["windows"]) == 2
    assert report["worst_windows"][0]["window_index"] == 1
    assert report["error_runs"][0]["length"] >= 4
    assert report["recommendations"][0]["category"] == "temporal_validation"
    assert format_shadow_replay_summary(report).startswith("Shadow replay:")


def test_shadow_replay_fingerprint_is_order_sensitive():
    features = np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    shuffled = [2, 0, 3, 1]

    assert shadow_replay_dataset_fingerprint(features, labels) != shadow_replay_dataset_fingerprint(
        features[shuffled],
        labels[shuffled],
    )


def test_shadow_replay_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.9], [999.0, 0.8], [999.0, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_shadow_replay(
        ProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        window_count=2,
        min_window_size=2,
    )

    assert report["sample_count"] == 4
    assert report["input_dim"] == 2
    assert report["summary"]["verdict"] in {"stable_ordered_replay", "thin_ordered_evidence"}


def test_shadow_replay_validates_inputs_and_probabilities():
    features = np.asarray([[0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_shadow_replay(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_shadow_replay(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_shadow_replay(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_shadow_replay(ProbabilityModel(), features, np.asarray([0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_shadow_replay(ProbabilityModel(), np.asarray([[0.2], [np.inf]], dtype=np.float32), labels)
