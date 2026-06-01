import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.rank_lift import (
    format_rank_lift_summary,
    rank_lift_dataset_fingerprint,
    run_rank_lift_diagnostics,
)


class ProbabilityModel:
    def predict(self, samples, verbose=0):
        return np.asarray(samples, dtype=np.float32)[:, :1]


class ConstantModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 0.5, dtype=np.float32)


class ShortModel:
    def predict(self, samples, verbose=0):
        return np.asarray([[0.2]], dtype=np.float32)


class NanModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), np.nan, dtype=np.float32)


class OutOfRangeModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 1.2, dtype=np.float32)


def test_rank_lift_reports_concentrated_top_ranked_signal():
    features = np.asarray(
        [[0.99], [0.96], [0.88], [0.72], [0.40], [0.30], [0.20], [0.10], [0.05], [0.01]],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    report = run_rank_lift_diagnostics(
        ProbabilityModel(),
        features,
        labels,
        top_fractions=[0.1, 0.2, 0.5, 1.0],
        deciles=5,
        max_rows=3,
    )

    assert report["summary"]["verdict"] == "concentrated_ranking"
    assert report["summary"]["top_10_lift"] == pytest.approx(10 / 3)
    assert report["summary"]["top_20_positive_capture"] == pytest.approx(2 / 3)
    assert report["summary"]["normalized_gains_auc"] > 0.8
    assert report["points"][0]["k"] == 1
    assert report["deciles_table"][0]["positive_count"] == 2
    assert report["top_rows"][0]["row_index"] == 0
    assert report["dataset_fingerprint"] == rank_lift_dataset_fingerprint(features, labels)
    assert format_rank_lift_summary(report).startswith("Rank lift:")


def test_rank_lift_uses_selected_feature_preprocessor():
    features = np.asarray([[99.0, 0.90], [99.0, 0.70], [99.0, 0.20], [99.0, 0.10]], dtype=np.float32)
    labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_rank_lift_diagnostics(
        ProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        top_fractions=[0.5, 1.0],
    )

    assert report["input_dim"] == 2
    assert report["points"][0]["precision_at_k"] == pytest.approx(1.0)
    assert report["summary"]["top_20_lift"] == pytest.approx(2.0)


def test_rank_lift_flags_flat_scores_and_no_positive_evidence():
    flat = run_rank_lift_diagnostics(
        ConstantModel(),
        np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        top_fractions=[0.5, 1.0],
    )
    no_positive = run_rank_lift_diagnostics(
        ProbabilityModel(),
        np.asarray([[0.9], [0.8], [0.1]], dtype=np.float32),
        np.asarray([0, 0, 0], dtype=np.int32),
        top_fractions=[0.5, 1.0],
    )

    assert flat["summary"]["verdict"] == "flat_scores"
    assert flat["summary"]["score_spread"] == 0.0
    assert no_positive["summary"]["verdict"] == "no_positive_evidence"
    assert no_positive["summary"]["top_10_positive_capture"] == 0.0


def test_rank_lift_validates_inputs_and_probabilities():
    features = np.asarray([[0.1], [0.9]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_rank_lift_diagnostics(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_rank_lift_diagnostics(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_rank_lift_diagnostics(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_rank_lift_diagnostics(ProbabilityModel(), features, np.asarray([0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_rank_lift_diagnostics(ProbabilityModel(), np.asarray([[0.1], [np.inf]], dtype=np.float32), labels)
    with pytest.raises(ValueError, match="top fractions"):
        run_rank_lift_diagnostics(ProbabilityModel(), features, labels, top_fractions=[float("nan")])
    with pytest.raises(ValueError, match="deciles"):
        run_rank_lift_diagnostics(ProbabilityModel(), features, labels, deciles=0)


def test_rank_lift_fingerprint_is_order_sensitive():
    features = np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    order = [2, 0, 3, 1]

    assert rank_lift_dataset_fingerprint(features, labels) != rank_lift_dataset_fingerprint(
        features[order],
        labels[order],
    )
