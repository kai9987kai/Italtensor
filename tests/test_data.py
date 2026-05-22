import json

import pytest

from italtensor.data import (
    DataValidationError,
    dataset_from_jsonable,
    dataset_to_jsonable,
    load_csv_dataset,
    parse_prediction_vector,
    parse_training_example,
    validate_dataset,
)


def test_parse_training_example_accepts_flat_vector_and_label():
    features, label = parse_training_example("[[0.1, 2, -3.5], 1]")

    assert features == [0.1, 2.0, -3.5]
    assert label == 1


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "__import__('os').system('echo unsafe')",
        "[[0.1, 0.2]]",
        "[[0.1, true], 1]",
        "[[], 1]",
        "[[0.1, 0.2], 2]",
        "[[0.1, 0.2], true]",
    ],
)
def test_parse_training_example_rejects_bad_input(raw):
    with pytest.raises(DataValidationError):
        parse_training_example(raw)


def test_prediction_vector_enforces_expected_dimension():
    assert parse_prediction_vector("[1, 2]", expected_dim=2) == [1.0, 2.0]

    with pytest.raises(DataValidationError, match="Expected 3 features"):
        parse_prediction_vector("[1, 2]", expected_dim=3)


def test_validate_dataset_rejects_mixed_feature_lengths():
    with pytest.raises(DataValidationError, match="Expected 2 features"):
        validate_dataset([[0.1, 0.2], [0.3]], [0, 1])


def test_validate_dataset_can_require_two_classes():
    with pytest.raises(DataValidationError, match="both labels"):
        validate_dataset([[0.1], [0.2]], [1, 1], require_two_classes=True)


def test_load_csv_dataset_accepts_header_and_final_label(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("f1,f2,label\n0.1,0.2,1\n0.3,0.4,0\n", encoding="utf-8")

    dataset = load_csv_dataset(csv_path)

    assert dataset.sample_count == 2
    assert dataset.input_dim == 2
    assert dataset.labels.tolist() == [1, 0]


@pytest.mark.parametrize(
    "content",
    [
        "f1,f2,label\n0.1,nope,1\n",
        "f1,f2,label\n0.1,0.2,3\n",
        "f1,f2,label\n0.1\n",
    ],
)
def test_load_csv_dataset_rejects_bad_rows(tmp_path, content):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(DataValidationError):
        load_csv_dataset(csv_path)


def test_load_csv_dataset_does_not_hide_bad_first_data_row(tmp_path):
    csv_path = tmp_path / "bad-first-row.csv"
    csv_path.write_text("0.1,nope,1\n0.2,0.3,0\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="CSV row 1"):
        load_csv_dataset(csv_path)


def test_dataset_json_round_trip():
    dataset = validate_dataset([[0.1, 0.2], [0.3, 0.4]], [1, 0])
    encoded = json.loads(json.dumps(dataset_to_jsonable(dataset)))

    decoded = dataset_from_jsonable(encoded)

    assert decoded.features.tolist() == dataset.features.tolist()
    assert decoded.labels.tolist() == [1, 0]
    assert decoded.input_dim == 2
