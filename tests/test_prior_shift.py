import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.prior_shift import (
    format_prior_shift_summary,
    prior_shift_dataset_fingerprint,
    run_prior_shift_diagnostics,
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


def test_prior_shift_simulates_predictive_values_under_prevalence_grid():
    features = np.asarray([[0.95], [0.85], [0.70], [0.65], [0.40], [0.30], [0.20], [0.10]], dtype=np.float32)
    labels = np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)

    report = run_prior_shift_diagnostics(
        ProbabilityModel(),
        features,
        labels,
        threshold=0.5,
        prevalence_grid=[0.02, 0.10, 0.50],
        population_size=1000,
    )

    assert report["summary"]["verdict"] == "prevalence_shift_risk"
    assert report["current"]["sensitivity"] == pytest.approx(1.0)
    assert report["current"]["specificity"] == pytest.approx(0.8)
    low_prevalence = report["points"][0]
    assert low_prevalence["prevalence"] == pytest.approx(0.02)
    assert low_prevalence["positive_predictive_value"] == pytest.approx(0.02 / (0.02 + 0.98 * 0.2))
    assert low_prevalence["expected_false_positive"] == pytest.approx(196.0)
    assert report["dataset_fingerprint"] == prior_shift_dataset_fingerprint(features, labels)
    assert format_prior_shift_summary(report).startswith("Prior shift:")


def test_prior_shift_uses_selected_feature_preprocessor():
    features = np.asarray([[999.0, 0.90], [999.0, 0.80], [999.0, 0.20], [999.0, 0.10]], dtype=np.float32)
    labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_prior_shift_diagnostics(
        ProbabilityModel(),
        features,
        labels,
        preprocessor=preprocessor,
        threshold=0.5,
        prevalence_grid=[0.5],
    )

    assert report["input_dim"] == 2
    assert report["current"]["true_positive"] == 2
    assert report["current"]["false_positive"] == 0
    assert report["summary"]["verdict"] == "prior_shift_stable"


def test_prior_shift_flags_missing_two_class_evidence():
    report = run_prior_shift_diagnostics(
        ProbabilityModel(),
        np.asarray([[0.9], [0.8], [0.7]], dtype=np.float32),
        np.asarray([0, 0, 0], dtype=np.int32),
        prevalence_grid=[0.1, 0.5],
    )

    assert report["summary"]["verdict"] == "no_two_class_evidence"
    assert report["current"]["warning"]


def test_prior_shift_validates_inputs_and_probabilities():
    features = np.asarray([[0.1], [0.9]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_prior_shift_diagnostics(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_prior_shift_diagnostics(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_prior_shift_diagnostics(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_prior_shift_diagnostics(ProbabilityModel(), features, np.asarray([0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_prior_shift_diagnostics(ProbabilityModel(), np.asarray([[0.1], [np.inf]], dtype=np.float32), labels)
    with pytest.raises(ValueError, match="prevalence grid"):
        run_prior_shift_diagnostics(ProbabilityModel(), features, labels, prevalence_grid=[float("nan")])
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_prior_shift_diagnostics(ProbabilityModel(), features, labels, prevalence_grid=[-0.1])
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_prior_shift_diagnostics(ProbabilityModel(), features, labels, prevalence_grid=[1.1])
    with pytest.raises(ValueError, match="threshold"):
        run_prior_shift_diagnostics(ProbabilityModel(), features, labels, threshold=1.5)


def test_prior_shift_fingerprint_is_order_sensitive():
    features = np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    order = [2, 0, 3, 1]

    assert prior_shift_dataset_fingerprint(features, labels) != prior_shift_dataset_fingerprint(
        features[order],
        labels[order],
    )
