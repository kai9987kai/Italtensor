import numpy as np
import pytest

from italtensor.label_sensitivity import (
    format_label_sensitivity_summary,
    label_sensitivity_dataset_fingerprint,
    run_label_sensitivity,
    run_label_sensitivity_diagnostics,
)
from italtensor.modeling import NumpyBinaryClassifier
from italtensor.preprocessing import FeatureStandardizer


def _margin_model(input_dim=1):
    weights = np.zeros(input_dim, dtype=np.float32)
    weights[0] = 3.0
    return NumpyBinaryClassifier(weights=weights, bias=0.0)


def test_label_sensitivity_ranks_metric_sensitive_label_flips():
    features = np.asarray([[-2.0], [-1.5], [1.4], [1.8], [2.2], [-2.2]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32)

    report = run_label_sensitivity(_margin_model(), features, labels, material_f1_delta=0.01, max_items=10)

    suspect_rows = {item["row_index"] for item in report["suspect_label_rows"]}
    assert {4, 5}.issubset(suspect_rows)
    assert report["primary_metric"] == "f1"
    assert report["observed"]["f1"] == pytest.approx(report["baseline_metrics"]["f1"])
    assert report["summary"]["priority"] == "high"
    assert report["summary"]["max_improving_f1_delta"] > 0.01
    assert report["rows"][0]["rank"] == 1
    assert "flip_improves_f1" in report["suspect_label_rows"][0]["risk_flags"]

    summary = format_label_sensitivity_summary(report)
    assert summary.startswith("Label sensitivity:")
    assert "suspect=" in summary


def test_label_sensitivity_alias_and_preprocessor_work():
    raw = np.asarray([[8.0], [9.0], [12.0], [13.0], [14.0], [6.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32)
    preprocessor = FeatureStandardizer.fit(raw)

    report = run_label_sensitivity_diagnostics(
        _margin_model(),
        raw,
        labels,
        preprocessor=preprocessor,
        material_f1_delta=0.01,
    )

    assert report["input_dim"] == 1
    assert report["summary"]["suspect_label_count"] >= 1


def test_label_sensitivity_rejects_bad_inputs():
    model = _margin_model()

    with pytest.raises(ValueError, match="2D"):
        run_label_sensitivity(model, [0.1, 0.2], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        run_label_sensitivity(model, [[0.1], [np.nan]], [0, 1])
    with pytest.raises(ValueError, match="binary labels"):
        run_label_sensitivity(model, [[0.1], [0.2]], [0, 2])
    with pytest.raises(ValueError, match="counts do not match"):
        run_label_sensitivity(model, [[0.1], [0.2]], [0])
    with pytest.raises(ValueError, match="threshold"):
        run_label_sensitivity(model, [[0.1], [0.2]], [0, 1], threshold=1.2)


def test_label_sensitivity_rejects_bad_model_probabilities():
    class BadModel:
        def predict(self, samples, verbose=0):
            return np.asarray([0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="different number"):
        run_label_sensitivity(BadModel(), [[0.1], [0.2]], [0, 1])


def test_label_sensitivity_dataset_fingerprint_is_order_sensitive():
    left = label_sensitivity_dataset_fingerprint([[0.1], [0.9]], [0, 1])
    right = label_sensitivity_dataset_fingerprint([[0.9], [0.1]], [1, 0])

    assert left != right
