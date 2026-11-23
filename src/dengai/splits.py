"""Time-ordered splitting: the locked report window, and validation folds inside the development region."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CITIES, DEV_LONG_WEEKS, FOLD_WEEKS, MIN_OTHER_CITY_ROWS, N_FOLDS


@dataclass(frozen=True)
class Split:
    """One validation split. All fitting rows are dated before `cutoff`, the first validation week."""
    evaluation: str          # 'annual' or 'devlong'
    city: str
    fold: int
    cutoff: pd.Timestamp
    valid_index: np.ndarray  # row labels of the validation weeks in the frame the split was built from


def locked_split(training: pd.DataFrame, report_weeks: dict) -> tuple:
    """Hold out the last `report_weeks[city]` training weeks of each city as the report window.

    The window has the same length as the real test period and sits at the end of each city's own history.
    Returns (dev, report). Model selection may only look at `dev`.
    """
    dev_parts, report_parts = [], []
    for city in CITIES:
        rows = training.loc[training.city.eq(city)].sort_values("week_start_date")
        dev_parts.append(rows.iloc[:-report_weeks[city]])
        report_parts.append(rows.iloc[-report_weeks[city]:])
    dev, report = pd.concat(dev_parts), pd.concat(report_parts)
    for city in CITIES:
        if not dev.loc[dev.city.eq(city), "week_start_date"].max() < report.loc[report.city.eq(city), "week_start_date"].min():
            raise ValueError(f"development region does not precede the report window for {city}")
    return dev, report


def fitting_rows(available: pd.DataFrame, architecture: str, city: str) -> pd.DataFrame:
    """Rows a model for `city` may be fitted on, from `available` (already restricted to dates before the cutoff).

    'separate' uses the city's own rows. 'pooled' adds the other city only once it has
    MIN_OTHER_CITY_ROWS usable weeks; before that a pooled model would be a separate one in disguise.
    """
    fit = available.loc[available.eligible]
    if architecture == "separate":
        return fit.loc[fit.city.eq(city)]
    counts = fit.city.value_counts()
    keep = [c for c in CITIES if c == city or counts.get(c, 0) >= MIN_OTHER_CITY_ROWS]
    return fit.loc[fit.city.isin(keep)]


def make_splits(dev: pd.DataFrame) -> tuple:
    """Per city: N_FOLDS annual folds (the last 52-week blocks of `dev`, expanding history) and one long fold
    (the last DEV_LONG_WEEKS weeks in a single pass). Returns (annual, devlong) lists of Split."""
    annual, devlong = [], []
    for city in CITIES:
        rows = dev.loc[dev.city.eq(city)].sort_values("week_start_date")
        n = len(rows)
        for fold in range(1, N_FOLDS + 1):
            stop = n - (N_FOLDS - fold) * FOLD_WEEKS
            valid = rows.iloc[stop - FOLD_WEEKS:stop]
            annual.append(Split("annual", city, fold, valid.week_start_date.min(), valid.index.to_numpy()))
        valid = rows.iloc[-DEV_LONG_WEEKS:]
        devlong.append(Split("devlong", city, 1, valid.week_start_date.min(), valid.index.to_numpy()))
    return annual, devlong


def fold_data(split: Split, frame: pd.DataFrame) -> tuple:
    """(history before the cutoff, validation weeks in time order) for one split."""
    available = frame.loc[frame.week_start_date.lt(split.cutoff)]
    valid = frame.loc[split.valid_index].sort_values("week_start_date")
    return available, valid


def audit_splits(splits: list, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per split: how much history each architecture gets. Also asserts there is no overlap in time."""
    rows = []
    for split in splits:
        available, valid = fold_data(split, frame)
        fitting = fitting_rows(available, "pooled", split.city)
        if not (fitting.week_start_date.max() < valid.week_start_date.min() and set(fitting.index).isdisjoint(valid.index)):
            raise ValueError(f"fold overlaps in time: {split.city} {split.evaluation} {split.fold}")
        rows.append({"evaluation": split.evaluation, "target_city": split.city, "fold": split.fold,
                     "cutoff": split.cutoff, "n_fit_target_city": int(fitting.city.eq(split.city).sum()),
                     "n_fit_other_city": int((~fitting.city.eq(split.city)).sum()), "n_valid": len(valid),
                     "valid_first": valid.week_start_date.min(), "valid_last": valid.week_start_date.max()})
    return pd.DataFrame(rows)
