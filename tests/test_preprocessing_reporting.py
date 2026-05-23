import json

import numpy as np
import pytest

from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer
from italtensor.reporting import build_experiment_report, export_experiment_report


def test_standardizer_fits_training_statistics_only():
    train_features = np.asarray([[1.0, 10.0], [3.0, 10.0]], dtype=np.float32)
    validation_features = np.asarray([[101.0, 10.0]], dtype=np.float32)

    standardizer = FeatureStandardizer.fit(train_features)
    transformed = standardizer.transform(validation_features)

    assert standardizer.mean.tolist() == pytest.approx([2.0, 10.0])
    assert standardizer.scale.tolist() == pytest.approx([1.0, 1.0])
    np.testing.assert_allclose(transformed, np.asarray([[99.0, 0.0]], dtype=np.float32))


def test_standardizer_metadata_round_trip():
    standardizer = FeatureStandardizer.fit(np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32))

    restored = FeatureStandardizer.from_dict(json.loads(json.dumps(standardizer.to_dict())))

    assert restored is not None
    assert restored.mean.tolist() == pytest.approx(standardizer.mean.tolist())
    assert restored.scale.tolist() == pytest.approx(standardizer.scale.tolist())


def test_standardizer_metadata_validates_input_dimension():
    standardizer = FeatureStandardizer.identity(2)

    with pytest.raises(ValueError, match="model expects 3"):
        FeatureStandardizer.from_dict(standardizer.to_dict(), input_dim=3)


def test_report_export_json_and_markdown(tmp_path):
    report = build_experiment_report(
        sample_count=4,
        input_dim=2,
        labels=[0, 0, 1, 1],
        features=[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        config=ModelConfig(hidden_layers=(16,), max_epochs=3),
        metrics={"f1": 0.75, "threshold": 0.4},
        threshold=0.4,
        preprocessor=FeatureStandardizer.identity(2),
        feature_importances=[{"feature_index": 0, "importance": 0.25}],
        trial_history=[
            {
                "config": {"feature_map": "rff"},
                "metrics": {"f1": 0.75, "brier_score": 0.1, "log_loss": 0.3},
            }
        ],
        uncertainty_metadata={
            "conformal_source": "dedicated_calibration",
            "conformal_alpha": 0.1,
            "conformal_quantile": 0.35,
            "conformal_target_coverage": 0.9,
            "conformal_coverage": 1.0,
            "conformal_calibration_count": 8,
            "conformal_evaluation_count": 8,
            "conformal_singleton_rate": 0.75,
        },
    )

    json_path = export_experiment_report(tmp_path / "report.json", report)
    markdown_path = export_experiment_report(tmp_path / "report.md", report)

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    saved_markdown = markdown_path.read_text(encoding="utf-8")
    assert saved_json["dataset"]["class_counts"] == {"0": 2, "1": 2}
    assert saved_json["dataset"]["available"] is True
    assert saved_json["dataset"]["audit"]["duplicate_row_count"] == 1
    assert saved_json["model"]["threshold"] == 0.4
    assert saved_json["uncertainty"]["conformal_source"] == "dedicated_calibration"
    assert saved_json["uncertainty"]["conformal_quantile"] == 0.35
    assert saved_json["uncertainty"]["conformal_calibration_count"] == 8
    assert saved_json["trial_history"][0]["config"]["feature_map"] == "rff"
    assert "Feature 0" in saved_markdown
    assert "## Dataset Audit" in saved_markdown
    assert "Trial 1" in saved_markdown
    assert "## Uncertainty" in saved_markdown
    assert "conformal_source" in saved_markdown


def test_report_marks_dataset_unavailable_for_model_only_export():
    report = build_experiment_report(
        sample_count=0,
        input_dim=2,
        labels=[],
        config=ModelConfig(),
        metrics={"f1": 0.0},
        threshold=0.5,
        preprocessor=None,
        feature_importances=[],
    )

    assert report["dataset"]["available"] is False
    assert report["dataset"]["class_counts"] is None
