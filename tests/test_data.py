import dataclasses

import pytest

from dengai.data import load_raw_data, validate_raw_data


def test_missing_files_give_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="drivendata.org"):
        load_raw_data(tmp_path)


def test_valid_data_passes(raw):
    validate_raw_data(raw)


def test_overlapping_train_and_test_is_rejected(raw):
    leaky_test = raw.train_features.groupby("city").tail(5)
    bad = dataclasses.replace(raw, test_features=leaky_test, submission_format=leaky_test[["city", "year", "weekofyear"]].assign(total_cases=0))
    with pytest.raises(ValueError, match="overlap"):
        validate_raw_data(bad)


def test_negative_labels_are_rejected(raw):
    labels = raw.train_labels.copy()
    labels.loc[0, "total_cases"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_raw_data(dataclasses.replace(raw, train_labels=labels))


def test_test_weights_follow_test_set(raw):
    assert raw.test_counts == {"sj": 52, "iq": 52}
    assert raw.test_weights == {"sj": 0.5, "iq": 0.5}
