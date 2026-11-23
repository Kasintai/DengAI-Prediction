"""Synthetic stand-in for the DrivenData files (same schema), so the tests never need the restricted data."""
import numpy as np
import pandas as pd
import pytest

from dengai.data import RawData
from dengai.features import build_modelling_frame, split_train_test
from dengai.splits import locked_split

WEATHER_COLUMNS = [
    "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm",
    "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c",
    "station_max_temp_c", "station_min_temp_c", "station_precip_mm",
]
TEST_WEEKS = 52
# (city, first Monday, number of training weeks)
CITY_SPECS = [("sj", "2000-01-03", 600), ("iq", "2003-01-06", 400)]


def _make_city(rng, city, start, n_train):
    dates = pd.date_range(start, periods=n_train + TEST_WEEKS, freq="7D")
    iso = dates.isocalendar()
    phase = 2 * np.pi * (dates.dayofyear - 1) / 365 + (0 if city == "sj" else np.pi)
    frame = pd.DataFrame({"city": city, "year": iso.year.to_numpy(), "weekofyear": iso.week.to_numpy(), "week_start_date": dates})
    for column in WEATHER_COLUMNS:
        base = 300.0 if column.endswith("_k") and "tdtr" not in column else 20.0
        frame[column] = base + 3 * np.sin(phase) + rng.normal(0, 1, len(dates))
    frame["reanalysis_sat_precip_amt_mm"] = np.abs(rng.normal(30, 10, len(dates)))
    frame["precipitation_amt_mm"] = frame["reanalysis_sat_precip_amt_mm"]  # exact duplicate, as in the real data
    frame["total_cases"] = rng.poisson(np.exp(2.5 + np.sin(phase)))
    return frame


@pytest.fixture(scope="session")
def raw() -> RawData:
    rng = np.random.default_rng(0)
    full = pd.concat([_make_city(rng, *spec) for spec in CITY_SPECS], ignore_index=True)
    keys = ["city", "year", "weekofyear"]
    is_test = full.groupby("city").cumcount(ascending=False).lt(TEST_WEEKS)
    features = full.drop(columns="total_cases")
    # sprinkle missing values so imputation is exercised
    for column in ("ndvi_ne", "station_avg_temp_c"):
        features.loc[rng.choice(len(features), 40, replace=False), column] = np.nan
    test_features = features[is_test].reset_index(drop=True)
    submission = test_features[keys].assign(total_cases=0)
    return RawData(
        train_features=features[~is_test].reset_index(drop=True),
        train_labels=full.loc[~is_test, keys + ["total_cases"]].reset_index(drop=True),
        test_features=test_features,
        submission_format=submission,
    )


@pytest.fixture(scope="session")
def frames(raw):
    """(weather, season_cols, training, test, dev, report)."""
    weather, season_cols = build_modelling_frame(raw)
    training, test = split_train_test(weather)
    dev, report = locked_split(training, raw.test_counts)
    return weather, season_cols, training, test, dev, report
