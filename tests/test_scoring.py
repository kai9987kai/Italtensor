import csv

import numpy as np
import pytest

from italtensor.data import DataValidationError
from italtensor.preprocessing import FeatureStandardizer
from italtensor.scoring import load_prediction_csv, load_reviewed_prediction_csv, score_prediction_csv, score_prediction_rows


class SumModel:
    def predict(self, samples, verbose=0):
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(samples).sum(axis=1)))
        return probabilities.reshape(-1, 1)


def test_load_prediction_csv_accepts_header_and_numeric_rows(tmp_path):
    path = tmp_path / "predict.csv"
    path.write_text("a,b\n0.1,0.2\n0.3,0.4\n", encoding="utf-8")

    table = load_prediction_csv(path, expected_dim=2)

    assert table.feature_names == ["a", "b"]
    np.testing.assert_allclose(table.features, np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))


def test_load_prediction_csv_rejects_non_numeric_data_row(tmp_path):
    path = tmp_path / "predict.csv"
    path.write_text("0.1,0.2\nbad,0.4\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="row 2"):
        load_prediction_csv(path, expected_dim=2)


def test_load_prediction_csv_rejects_wrong_feature_count(tmp_path):
    path = tmp_path / "predict.csv"
    path.write_text("a,b\n0.1,0.2,0.3\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Expected 2 features"):
        load_prediction_csv(path, expected_dim=2)


def test_score_prediction_rows_adds_label_set_and_review_priority():
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0, 0.0], dtype=np.float32),
        scale=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    scored = score_prediction_rows(
        SumModel(),
        np.asarray([[0.0, 0.0], [4.0, 4.0]], dtype=np.float32),
        preprocessor=preprocessor,
        threshold=0.5,
        uncertainty_metadata={"conformal_quantile": 0.55},
    )

    assert scored[0]["conformal_set"] == "{0,1}"
    assert scored[0]["review_priority"] == "high"
    assert scored[0]["active_query_rank"] == 1
    assert scored[0]["active_query_score"] > scored[1]["active_query_score"]
    assert scored[1]["label"] == 1
    assert scored[1]["ood_flag"] == 1
    assert scored[1]["review_priority"] == "high"


def test_score_prediction_csv_exports_scores_and_preserves_features(tmp_path):
    input_path = tmp_path / "predict.csv"
    output_path = tmp_path / "scored.csv"
    input_path.write_text("x1,x2,x3\n0.0,2.0,99.0\n1.0,2.0,99.0\n", encoding="utf-8")
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0, 0.0], dtype=np.float32),
        scale=np.asarray([1.0, 1.0], dtype=np.float32),
        selected_indices=[0, 1],
    )

    path, count = score_prediction_csv(
        SumModel(),
        input_path,
        output_path,
        expected_dim=3,
        preprocessor=preprocessor,
        threshold=0.5,
        uncertainty_metadata={"conformal_quantile": 0.2},
    )

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert count == 2
    assert rows[0] == [
        "x1",
        "x2",
        "x3",
        "italtensor_probability",
        "italtensor_label",
        "italtensor_conformal_set",
        "italtensor_uncertainty_score",
        "italtensor_active_query_score",
        "italtensor_active_query_rank",
        "italtensor_drift_score",
        "italtensor_max_abs_z",
        "italtensor_ood_flag",
        "italtensor_review_priority",
        "italtensor_review_label",
    ]
    assert rows[1][:3] == ["0.0", "2.0", "99.0"]
    assert rows[1][5] in {"{0}", "{1}", "{0,1}", "abstain", "unavailable"}
    assert rows[1][11] in {"0", "1", "unavailable"}
    assert rows[1][13] == ""


def test_score_prediction_rows_marks_missing_conformal_as_unavailable():
    scored = score_prediction_rows(
        SumModel(),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        threshold=0.5,
        uncertainty_metadata={},
    )

    assert scored[0]["conformal_set"] == "unavailable"
    assert scored[0]["drift_score"] == "unavailable"
    assert scored[0]["ood_flag"] == "unavailable"
    assert scored[0]["active_query_rank"] == 1


def test_drift_diagnostics_use_selected_standardized_features_only():
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0, 0.0], dtype=np.float32),
        scale=np.asarray([1.0, 2.0], dtype=np.float32),
        selected_indices=[0, 2],
    )

    scored = score_prediction_rows(
        SumModel(),
        np.asarray([[0.0, 999.0, 2.0], [3.0, 999.0, 0.0]], dtype=np.float32),
        preprocessor=preprocessor,
        threshold=0.5,
        uncertainty_metadata={},
    )

    assert scored[0]["max_abs_z"] == pytest.approx(1.0)
    assert scored[0]["ood_flag"] == 0
    assert scored[1]["max_abs_z"] == pytest.approx(3.0)
    assert scored[1]["ood_flag"] == 1


def test_load_reviewed_prediction_csv_imports_only_labeled_rows(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "\n".join(
            [
                "x1,x2,italtensor_probability,italtensor_review_label",
                "0.1,0.2,0.4,1",
                "0.3,0.4,0.6,",
                "0.5,0.6,0.7,0",
            ]
        ),
        encoding="utf-8",
    )

    table, labels = load_reviewed_prediction_csv(path, expected_dim=2)

    assert table.feature_names == ["x1", "x2"]
    np.testing.assert_allclose(table.features, np.asarray([[0.1, 0.2], [0.5, 0.6]], dtype=np.float32))
    np.testing.assert_array_equal(labels, np.asarray([1, 0], dtype=np.int32))


def test_load_reviewed_prediction_csv_rejects_bad_review_labels(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text("x1,italtensor_probability,italtensor_review_label\n0.1,0.2,maybe\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="review label must be 0 or 1"):
        load_reviewed_prediction_csv(path, expected_dim=1)


def test_load_reviewed_prediction_csv_rejects_missing_review_column(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text("x1,italtensor_probability\n0.1,0.2\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="italtensor_review_label"):
        load_reviewed_prediction_csv(path, expected_dim=1)


def test_load_reviewed_prediction_csv_rejects_no_reviewed_rows(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "x1,italtensor_probability,italtensor_review_label\n0.1,0.2,\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="no rows"):
        load_reviewed_prediction_csv(path, expected_dim=1)


def test_load_reviewed_prediction_csv_rejects_unscored_csv(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text("x1,italtensor_review_label\n0.1,1\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="scored Italtensor metadata"):
        load_reviewed_prediction_csv(path, expected_dim=1)


def test_load_reviewed_prediction_csv_rejects_wrong_feature_count(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "x1,x2,italtensor_probability,italtensor_review_label\n0.1,0.2,0.3,1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="expected 1"):
        load_reviewed_prediction_csv(path, expected_dim=1)


def test_load_reviewed_prediction_csv_rejects_bad_feature_cell(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "x1,italtensor_probability,italtensor_review_label\nbad,0.3,1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="row 2"):
        load_reviewed_prediction_csv(path, expected_dim=1)
