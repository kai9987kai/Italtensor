import numpy as np
import pytest

from italtensor.counterfactual import find_counterfactual, format_counterfactual_result
from italtensor.preprocessing import FeatureStandardizer


class SumModel:
    def predict(self, samples, verbose=0):
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(samples).sum(axis=1)))
        return probabilities.reshape(-1, 1)


class NegativeSumModel:
    def predict(self, samples, verbose=0):
        probabilities = 1.0 / (1.0 + np.exp(np.asarray(samples).sum(axis=1)))
        return probabilities.reshape(-1, 1)


def test_counterfactual_finds_nearby_flip_for_linear_model():
    result = find_counterfactual(
        SumModel(),
        [-2.0, 0.0],
        threshold=0.5,
        max_steps=12,
        samples_per_step=0,
    )

    assert result.success is True
    assert result.original_label == 0
    assert result.target_label == 1
    assert result.candidate_probability >= 0.5
    assert result.normalized_l1 > 0
    assert result.changed_features
    assert "Counterfactual target=1" in format_counterfactual_result(result)


def test_counterfactual_can_target_negative_class():
    result = find_counterfactual(
        NegativeSumModel(),
        [-2.0, 0.0],
        threshold=0.5,
        target_label=0,
        max_steps=12,
        samples_per_step=0,
    )

    assert result.success is True
    assert result.target_label == 0
    assert result.candidate_probability < 0.5


def test_counterfactual_respects_selected_preprocessor_features():
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    result = find_counterfactual(
        SumModel(),
        [100.0, -2.0],
        preprocessor=preprocessor,
        threshold=0.5,
        max_steps=12,
        samples_per_step=0,
    )

    assert result.success is True
    assert {item["feature_index"] for item in result.changed_features} == {1}


def test_counterfactual_rejects_bad_input():
    with pytest.raises(ValueError, match="finite"):
        find_counterfactual(SumModel(), [float("nan")])
