"""Naive reference forecasts. A model is only worth its complexity if it beats `seasonal_median` (separate)."""
import numpy as np
import pandas as pd

from .config import TARGET
from .splits import fitting_rows

BASELINES = ["median", "seasonal_median", "last_value"]
BASELINE_VARIANTS = [(kind, architecture) for kind in BASELINES
                     for architecture in (["separate"] if kind == "last_value" else ["separate", "pooled"])]
REFERENCE_BASELINE = ("seasonal_median", "separate")


def baseline_forecast(kind: str, architecture: str, available: pd.DataFrame, valid: pd.DataFrame, city: str) -> np.ndarray:
    """Point forecast of one baseline for the weeks in `valid`, using only the rows in `available`."""
    if kind == "last_value":
        history = available.loc[available.city.eq(city)].sort_values("week_start_date")
        return np.full(len(valid), float(history[TARGET].iloc[-1]))
    fit = fitting_rows(available, architecture, city)
    overall = float(fit[TARGET].median())
    if kind == "median":
        return np.full(len(valid), overall)
    if kind == "seasonal_median":  # median of the same week of year; falls back to the overall median
        weekly = fit.groupby("weekofyear")[TARGET].median()
        return valid.weekofyear.map(weekly).fillna(overall).to_numpy(float)
    raise ValueError(kind)
