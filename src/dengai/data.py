"""Loading and validating the raw DrivenData files."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CITIES, KEYS, RAW_FILES, TARGET


@dataclass(frozen=True)
class RawData:
    train_features: pd.DataFrame
    train_labels: pd.DataFrame
    test_features: pd.DataFrame
    submission_format: pd.DataFrame

    @property
    def test_counts(self) -> dict:
        """Number of test weeks per city (also the length of the held-out report window)."""
        return self.test_features.city.value_counts().reindex(CITIES).astype(int).to_dict()

    @property
    def test_weights(self) -> dict:
        """Share of each city in the test set; used to weight per-city errors into one number."""
        return self.test_features.city.value_counts(normalize=True).reindex(CITIES).to_dict()


def load_raw_data(raw_dir: Path) -> RawData:
    """Read the four competition files. Raises a helpful error if any is missing."""
    raw_dir = Path(raw_dir)
    missing = [name for name in RAW_FILES.values() if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Download the competition data from "
            "https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/ "
            "and place the CSV files there (they cannot be redistributed with this repository)."
        )
    date = ["week_start_date"]
    return RawData(
        train_features=pd.read_csv(raw_dir / RAW_FILES["train_features"], parse_dates=date),
        train_labels=pd.read_csv(raw_dir / RAW_FILES["train_labels"]),
        test_features=pd.read_csv(raw_dir / RAW_FILES["test_features"], parse_dates=date),
        submission_format=pd.read_csv(raw_dir / RAW_FILES["submission_format"]),
    )


def validate_raw_data(raw: RawData) -> None:
    """Integrity checks. Raises ValueError describing the first problem found."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    frames = {"train_features": raw.train_features, "train_labels": raw.train_labels,
              "test_features": raw.test_features, "submission_format": raw.submission_format}
    for name, frame in frames.items():
        require(not frame[KEYS].isna().any().any(), f"{name}: missing key values")
        require(not frame.duplicated(KEYS).any(), f"{name}: duplicate keys")

    require(set(raw.train_features.columns) == set(raw.test_features.columns), "train/test feature columns differ")
    require(set(raw.train_features.city) == set(raw.test_features.city) == set(CITIES), f"cities must be {CITIES}")

    joined = raw.train_features.merge(raw.train_labels, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    require(joined["_merge"].eq("both").all(), "training features and labels do not match one-to-one")
    labels = raw.train_labels[TARGET]
    require(bool(np.isfinite(labels).all() and labels.ge(0).all() and labels.mod(1).eq(0).all()),
            "labels must be non-negative integers")

    require(raw.train_features[KEYS].merge(raw.test_features[KEYS], on=KEYS).empty, "train and test keys overlap")
    check = raw.test_features[KEYS].merge(raw.submission_format[KEYS], on=KEYS, how="outer",
                                          validate="one_to_one", indicator=True)
    require(check["_merge"].eq("both").all(), "test keys differ from the submission template")

    for city in CITIES:
        last_train = raw.train_features.loc[raw.train_features.city.eq(city), "week_start_date"].max()
        first_test = raw.test_features.loc[raw.test_features.city.eq(city), "week_start_date"].min()
        require(last_train < first_test, f"test period does not follow training for {city}")
