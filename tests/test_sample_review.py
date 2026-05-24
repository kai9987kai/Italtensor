import numpy as np
import pytest

from italtensor.preprocessing import FeatureStandardizer
from italtensor.sample_review import format_sample_review_summary, run_sample_review


class ProbabilityModel:
    def predict(self, samples, verbose=0):
        values = np.asarray(samples, dtype=np.float32)[:, 0]
        return values.reshape(-1, 1)


def test_sample_review_flags_confident_disagreement_as_label_issue():
    features = np.asarray([[0.05], [0.95], [0.9], [0.1]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 0], dtype=np.int32)

    report = run_sample_review(ProbabilityModel(), features, labels, threshold=0.5)

    assert report["summary"]["label_issue_count"] == 1
    assert report["label_issues"][0]["row_index"] == 2
    assert report["label_issues"][0]["predicted_label"] == 1
    assert "Sample review" in format_sample_review_summary(report)


def test_sample_review_sorts_high_loss_and_ambiguous_rows():
    features = np.asarray([[0.02], [0.49], [0.9], [0.99]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int32)

    report = run_sample_review(ProbabilityModel(), features, labels, threshold=0.5)

    assert report["hard_examples"][0]["row_index"] == 3
    assert report["ambiguous_examples"][0]["row_index"] == 1


def test_sample_review_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.1], [999.0, 0.9]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_sample_review(ProbabilityModel(), features, labels, preprocessor=preprocessor)

    assert report["summary"]["disagreement_count"] == 0
    assert report["hard_examples"][0]["feature_preview"] == [999.0, 0.10000000149011612]


def test_sample_review_allows_threshold_edges_and_validates_inputs():
    features = np.asarray([[0.2]], dtype=np.float32)
    labels = np.asarray([0], dtype=np.int32)

    assert run_sample_review(ProbabilityModel(), features, labels, threshold=0.0)["sample_count"] == 1
    assert run_sample_review(ProbabilityModel(), features, labels, threshold=1.0)["sample_count"] == 1
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_sample_review(ProbabilityModel(), features, labels, threshold=-0.1)
