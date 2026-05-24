import numpy as np

from italtensor.preprocessing import FeatureStandardizer
from italtensor.stress import format_stress_summary, run_stress_suite


class SumModel:
    def predict(self, samples, verbose=0):
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(samples).sum(axis=1)))
        return probabilities.reshape(-1, 1)


def test_stress_suite_is_deterministic_with_fixed_seed():
    features = np.asarray([[-1.0], [-0.8], [0.8], [1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    first = run_stress_suite(SumModel(), features, labels, seed=7)
    second = run_stress_suite(SumModel(), features, labels, seed=7)

    assert first == second
    assert first["summary"]["worst_case"] != "none"
    assert "Stress suite" in format_stress_summary(first)


def test_stress_dropout_degrades_fragile_toy_model():
    features = np.asarray([[-1.0], [-0.8], [0.8], [1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = run_stress_suite(
        SumModel(),
        features,
        labels,
        seed=3,
        noise_levels=(),
        dropout_rates=(1.0,),
        max_feature_shifts=0,
    )

    dropout = report["perturbations"][0]
    assert report["base"]["f1"] == 1.0
    assert dropout["kind"] == "feature_dropout"
    assert dropout["f1"] < report["base"]["f1"]
    assert dropout["label_flip_rate"] > 0.0


def test_stress_suite_perturbs_raw_features_before_selected_preprocessor():
    features = np.asarray([[100.0, -1.0], [100.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_stress_suite(
        SumModel(),
        features,
        labels,
        preprocessor=preprocessor,
        noise_levels=(),
        dropout_rates=(),
        shift_magnitude=1.0,
        max_feature_shifts=4,
    )

    shifts = [item for item in report["perturbations"] if item["kind"] == "feature_shift"]
    unselected = [item for item in shifts if item["feature_index"] == 0]
    selected = [item for item in shifts if item["feature_index"] == 1]
    assert all(item["mean_probability_shift"] == 0.0 for item in unselected)
    assert any(item["mean_probability_shift"] > 0.0 for item in selected)
