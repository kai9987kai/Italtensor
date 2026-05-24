import numpy as np
import pytest

from italtensor.ablation import format_ablation_summary, run_ablation_diagnostics
from italtensor.preprocessing import FeatureStandardizer


class FirstFeatureModel:
    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)
        probabilities = np.where(values[:, 0] > 0.0, 0.9, 0.1)
        return probabilities.reshape(-1, 1)


def test_ablation_diagnostics_ranks_reliant_feature_first():
    features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = run_ablation_diagnostics(FirstFeatureModel(), features, labels, threshold=0.5, seed=3)

    assert report["base"]["f1"] == 1.0
    assert report["summary"]["top_feature"] == "x1"
    assert report["features"][0]["feature_index"] == 0
    assert report["features"][0]["f1_drop"] > 0.9
    assert "high_f1_reliance" in report["features"][0]["risk_flags"]
    assert "Ablation diagnostics" in format_ablation_summary(report)


def test_ablation_diagnostics_uses_preprocessor_on_raw_features():
    features = np.asarray(
        [
            [999.0, -2.0],
            [999.0, -1.0],
            [999.0, 1.0],
            [999.0, 2.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(mean=np.asarray([0.0], dtype=np.float32), scale=np.asarray([1.0], dtype=np.float32), selected_indices=[1])

    report = run_ablation_diagnostics(FirstFeatureModel(), features, labels, preprocessor=preprocessor)

    assert report["features"][0]["feature_index"] == 1
    assert report["features"][0]["f1_drop"] > 0.9
    assert report["features"][0]["selected_by_preprocessor"] is True
    assert report["features"][1]["feature_index"] == 0
    assert report["features"][1]["selected_by_preprocessor"] is False
    assert "constant_feature" in report["features"][1]["risk_flags"]


def test_ablation_diagnostics_validates_inputs():
    with pytest.raises(ValueError, match="2D"):
        run_ablation_diagnostics(FirstFeatureModel(), [1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="counts"):
        run_ablation_diagnostics(FirstFeatureModel(), [[1.0], [2.0]], [0])
    with pytest.raises(ValueError, match="threshold"):
        run_ablation_diagnostics(FirstFeatureModel(), [[1.0]], [1], threshold=1.1)
