"""Cleaning, causal imputation and feature construction.

Everything here looks backwards only: a value at week t never depends on weeks after t. That is
what lets the same table serve every validation fold and the final forecast without leakage.
"""
import numpy as np
import pandas as pd

from .config import (ABSOLUTE_TEMP_K, CORE, DROP_DUPLICATE, FFILL_LIMIT, KEYS, LAG_BLOCKS, MAX_HISTORY,
                     N_HARMONICS, TARGET, WEATHER_SETS)
from .data import RawData


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kelvin -> Celsius, and drop the exact duplicate rainfall column."""
    if not df[DROP_DUPLICATE].equals(df["reanalysis_sat_precip_amt_mm"]):
        raise ValueError(f"{DROP_DUPLICATE} is no longer identical to reanalysis_sat_precip_amt_mm")
    df = df.copy()
    df[ABSOLUTE_TEMP_K] = df[ABSOLUTE_TEMP_K] - 273.15
    df = df.rename(columns={c: c[:-2] + "_c" for c in ABSOLUTE_TEMP_K + ["reanalysis_tdtr_k"]})
    return df.drop(columns=DROP_DUPLICATE)


def impute_weather(weather: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Causal imputation: forward fill (at most FFILL_LIMIT weeks), then the city's expanding median
    of strictly earlier values. `weather` must be sorted by city and date."""
    weather = weather.copy()
    observed = weather[columns]
    past_median = observed.groupby(weather.city).transform(lambda s: s.expanding().median().shift(1))
    weather[columns] = observed.groupby(weather.city).ffill(limit=FFILL_LIMIT)
    weather[columns] = weather[columns].fillna(past_median)
    return weather


def seasonal_columns(weather: pd.DataFrame) -> list:
    """Add N_HARMONICS sine/cosine pairs of the day of the year (in place). Returns the new column names."""
    phase = 2 * np.pi * (weather.week_start_date.dt.day_of_year - 1) / 365
    names = []
    for k in range(1, N_HARMONICS + 1):
        weather[f"season_sin_{k}"], weather[f"season_cos_{k}"] = np.sin(k * phase), np.cos(k * phase)
        names += [f"season_sin_{k}", f"season_cos_{k}"]
    return names


def lag_columns(weather: pd.DataFrame) -> list:
    """For each core variable add the current week (`__L0`) and the mean of each lag block, built within city
    from earlier rows only (in place). Returns the new column names."""
    names = []
    for column in CORE:
        weather[f"{column}__L0"] = weather[column]
        names.append(f"{column}__L0")
        for block, (first, last) in LAG_BLOCKS.items():
            width = last - first + 1
            weather[f"{column}__{block}"] = weather.groupby("city")[column].transform(
                lambda s, f=first, w=width: s.shift(f).rolling(w, min_periods=w).mean())
            names.append(f"{column}__{block}")
    return names


def weather_columns(feature_set: str) -> list:
    """Weather columns a feature set adds to the seasonal curve."""
    return [f"{column}__{suffix}" for suffix in WEATHER_SETS[feature_set] for column in CORE]


def build_modelling_frame(raw: RawData) -> tuple:
    """Train + test weeks in one chronological table with all features.

    Returns (weather, season_cols). `weather` has a `split` column ('train' / 'test') and the
    `total_cases` label for training rows (NaN for test rows).
    """
    train, test = clean_features(raw.train_features), clean_features(raw.test_features)
    columns = [c for c in train if c not in KEYS + ["week_start_date"]]

    weather = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    weather = weather.sort_values(["city", "week_start_date"]).reset_index(drop=True)
    if weather.duplicated(["city", "week_start_date"]).any():
        raise ValueError("duplicate (city, week_start_date) rows")

    weather = impute_weather(weather, columns)
    weather["precip_log1p"] = np.log1p(weather.reanalysis_sat_precip_amt_mm.clip(lower=0))
    season_cols = seasonal_columns(weather)
    lag_cols = lag_columns(weather)
    assert max(last for _, last in LAG_BLOCKS.values()) <= MAX_HISTORY

    weather = weather.merge(raw.train_labels, on=KEYS, how="left", validate="one_to_one")
    weather = weather.sort_values(["city", "week_start_date"]).reset_index(drop=True)
    # A row can be used for fitting once its longest lag window is complete
    weather["eligible"] = weather.groupby("city").cumcount().ge(MAX_HISTORY)
    train_rows = weather.split.eq("train") & weather.eligible
    if weather.loc[train_rows, season_cols + lag_cols].isna().any().any():
        raise ValueError("eligible training rows still contain missing features")
    return weather, season_cols


def split_train_test(weather: pd.DataFrame) -> tuple:
    """(training rows with labels, test rows without labels), both sorted by city and date."""
    training = weather.loc[weather.split.eq("train")].reset_index(drop=True)
    test = weather.loc[weather.split.eq("test")].drop(columns=TARGET).reset_index(drop=True)
    return training, test
