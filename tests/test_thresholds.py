import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.thresholds import format_threshold_summary, run_threshold_diagnostics


class FirstFeatureProbabilityModel:
    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)[:, 0]
        return values.reshape(-1, 1)


def test_threshold_diagnostics_selects_best_f1_and_cost_points():
    features = np.asarray([[0.1], [0.2], [0.55], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 1], dtype=np.int32)

    report = run_threshold_diagnostics(
        FirstFeatureProbabilityModel(),
        features,
        labels,
        current_threshold=0.5,
        fp_cost=1.0,
        fn_cost=4.0,
        grid_size=5,
    )

    assert report["best_f1"]["f1"] == 1.0
    assert report["best_f1"]["threshold"] == pytest.approx(0.2)
    assert report["min_cost"]["threshold"] == pytest.approx(0.2)
    assert report["best_f1"]["false_positive_rate"] == 0.0
    assert report["best_f1"]["false_negative_rate"] == 0.0
    assert report["summary"]["best_f1"] == 1.0
    assert "Threshold sweep" in format_threshold_summary(report)


def test_threshold_diagnostics_honors_precision_and_recall_targets():
    features = np.asarray([[0.1], [0.35], [0.45], [0.9]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = run_threshold_diagnostics(
        FirstFeatureProbabilityModel(),
        features,
        labels,
        recall_target=1.0,
        precision_target=1.0,
        grid_size=5,
    )

    assert report["high_recall"]["recall"] >= 1.0
    assert report["high_precision"]["precision"] >= 1.0


def test_threshold_diagnostics_handles_identical_probabilities_and_one_class():
    features = np.asarray([[0.4], [0.4], [0.4]], dtype=np.float32)
    labels = np.asarray([0, 0, 0], dtype=np.int32)

    report = run_threshold_diagnostics(FirstFeatureProbabilityModel(), features, labels, grid_size=11)

    thresholds = [point["threshold"] for point in report["points"]]
    assert thresholds == sorted(set(thresholds))
    assert report["best_f1"]["f1"] == 0.0
    assert report["best_balanced_accuracy"]["balanced_accuracy"] >= 0.0


def test_threshold_diagnostics_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.8]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_threshold_diagnostics(
        FirstFeatureProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        current_threshold=0.5,
    )

    assert report["current"]["f1"] == 1.0
