import numpy as np
import pytest

from italtensor.pairwise_interactions import (
    format_pairwise_interaction_summary,
    run_pairwise_interaction_diagnostics,
)


class AdditiveModel:
    def predict(self, samples, verbose=0):
        x = np.asarray(samples, dtype=np.float32)
        logits = 0.9 * x[:, 0] + 0.7 * x[:, 1]
        return (1.0 / (1.0 + np.exp(-logits))).reshape(-1, 1)


class MultiplicativeModel:
    def predict(self, samples, verbose=0):
        x = np.asarray(samples, dtype=np.float32)
        logits = 3.0 * x[:, 0] * x[:, 1] + 0.2 * x[:, 2]
        return (1.0 / (1.0 + np.exp(-logits))).reshape(-1, 1)


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(self, samples, verbose=0):
        sample_count = np.asarray(samples).shape[0]
        return self.probabilities[:sample_count].reshape(-1, 1)


class CountingPreprocessor:
    def __init__(self):
        self.calls = 0

    def transform(self, values):
        self.calls += 1
        return np.asarray(values, dtype=np.float32)


def _interaction_features():
    values = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
    rows = []
    for left in values:
        for right in values:
            rows.append([float(left), float(right), 0.0])
    return np.asarray(rows, dtype=np.float32)


def test_pairwise_interactions_rank_true_multiplicative_pair():
    features = _interaction_features()
    labels = (features[:, 0] * features[:, 1] > 0).astype(int).tolist()

    report = run_pairwise_interaction_diagnostics(MultiplicativeModel(), features, labels, grid_size=5)

    assert report["summary"]["top_pair"] == [0, 1]
    assert report["summary"]["top_interaction_strength"] > 0.25
    assert "strong_interaction" in report["pairs"][0]["risk_flags"]
    assert "Pairwise interactions" in format_pairwise_interaction_summary(report)


def test_pairwise_interactions_additive_model_has_smaller_interaction():
    features = _interaction_features()
    labels = (features[:, 0] + features[:, 1] > 0).astype(int).tolist()

    additive = run_pairwise_interaction_diagnostics(AdditiveModel(), features, labels, grid_size=5)
    multiplicative = run_pairwise_interaction_diagnostics(MultiplicativeModel(), features, labels, grid_size=5)

    assert additive["summary"]["top_interaction_strength"] < multiplicative["summary"]["top_interaction_strength"]


def test_pairwise_interactions_applies_preprocessor_once_and_caps_features():
    preprocessor = CountingPreprocessor()
    features = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 12),
            np.linspace(1.0, -1.0, 12),
            np.linspace(-0.5, 0.5, 12),
            np.zeros(12),
        ]
    )
    labels = (features[:, 0] * features[:, 1] > 0).astype(int).tolist()

    report = run_pairwise_interaction_diagnostics(
        MultiplicativeModel(),
        features,
        labels,
        preprocessor=preprocessor,  # type: ignore[arg-type]
        max_features=3,
    )

    assert preprocessor.calls == 1
    assert len(report["selected_features"]) == 3
    assert report["summary"]["evaluated_pair_count"] == 3
    assert report["omitted_feature_count"] == 1


def test_pairwise_interactions_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D array"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), [1.0], [1])
    with pytest.raises(ValueError, match="counts do not match"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), [[1.0, 2.0], [3.0, 4.0]], [1])
    with pytest.raises(ValueError, match="at least one sample"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), np.empty((0, 2)), [])
    with pytest.raises(ValueError, match="at least two features"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), [[1.0]], [1])
    with pytest.raises(ValueError, match="binary labels"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), [[1.0, 2.0], [3.0, 4.0]], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), [[float("nan"), 2.0]], [0])
    with pytest.raises(ValueError, match="grid_size"):
        run_pairwise_interaction_diagnostics(AdditiveModel(), [[1.0, 2.0]], [0], grid_size=2)


def test_pairwise_interactions_rejects_bad_model_probabilities():
    features = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    labels = [0, 1, 1]
    with pytest.raises(ValueError, match="probability count"):
        run_pairwise_interaction_diagnostics(FixedProbabilityModel([0.5]), features, labels, grid_size=3)
    with pytest.raises(ValueError, match="non-finite"):
        probabilities = [0.5] * 27
        probabilities[1] = float("nan")
        run_pairwise_interaction_diagnostics(
            FixedProbabilityModel(probabilities),
            features,
            labels,
            grid_size=3,
        )
