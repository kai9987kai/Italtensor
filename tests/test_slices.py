import numpy as np

from italtensor.preprocessing import FeatureStandardizer
from italtensor.slices import format_slice_summary, run_slice_diagnostics


class PrimarySignalModel:
    def predict(self, samples, verbose=0):
        probabilities = 1.0 / (1.0 + np.exp(-4.0 * np.asarray(samples)[:, 0]))
        return probabilities.reshape(-1, 1)


def test_slice_diagnostics_finds_minority_subgroup_blind_spot():
    features = np.asarray(
        [
            [-1.0, 0.0],
            [-0.8, 0.0],
            [0.8, 0.0],
            [1.0, 0.0],
            [0.8, 1.0],
            [1.0, 1.0],
            [-0.8, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32)

    report = run_slice_diagnostics(PrimarySignalModel(), features, labels, bins=2, min_count=2)

    assert report["summary"]["slice_count"] > 0
    assert report["slices"][0]["f1_delta"] < 0.0
    assert any(item["feature_index"] == 1 and item["f1"] == 0.0 for item in report["slices"])
    assert "Slice diagnostics" in format_slice_summary(report)


def test_slice_diagnostics_skips_constant_features_and_includes_confusion_counts():
    features = np.asarray([[-1.0, 1.0], [-0.5, 1.0], [0.5, 1.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = run_slice_diagnostics(PrimarySignalModel(), features, labels, bins=2, min_count=2)

    assert all(item["feature_index"] == 0 for item in report["slices"])
    assert {"true_positive", "true_negative", "false_positive", "false_negative"}.issubset(report["base"])


def test_slice_diagnostics_uses_raw_bins_and_selected_preprocessor_once():
    features = np.asarray([[999.0, -1.0], [999.0, -0.8], [999.0, 0.8], [999.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_slice_diagnostics(
        PrimarySignalModel(),
        features,
        labels,
        preprocessor=preprocessor,
        bins=2,
        min_count=2,
    )

    assert report["base"]["f1"] == 1.0
    assert all(item["feature_index"] == 1 for item in report["slices"])


def test_slice_diagnostics_respects_min_count():
    features = np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = run_slice_diagnostics(PrimarySignalModel(), features, labels, bins=4, min_count=3)

    assert report["summary"]["slice_count"] == 0
    assert report["slices"] == []
