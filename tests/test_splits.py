import pytest

from dengai.config import CITIES, DEV_LONG_WEEKS, FOLD_WEEKS, N_FOLDS
from dengai.splits import audit_splits, fitting_rows, fold_data, make_splits


def test_report_window_follows_development_region(frames, raw):
    *_, dev, report = frames
    assert set(dev.index).isdisjoint(report.index)
    for city in CITIES:
        assert (dev.loc[dev.city.eq(city), "week_start_date"].max()
                < report.loc[report.city.eq(city), "week_start_date"].min())
        assert report.city.eq(city).sum() == raw.test_counts[city]


def test_folds_stay_inside_development_region(frames):
    *_, dev, report = frames
    annual, devlong = make_splits(dev)
    assert len(annual) == N_FOLDS * len(CITIES) and len(devlong) == len(CITIES)
    for split in annual + devlong:
        available, valid = fold_data(split, dev)
        assert set(valid.index) <= set(dev.index)
        assert available.week_start_date.max() < valid.week_start_date.min()
        assert len(valid) == (FOLD_WEEKS if split.evaluation == "annual" else DEV_LONG_WEEKS)
    audit_splits(annual + devlong, dev)  # raises if any fold overlaps in time


def test_pooled_only_adds_other_city_once_it_has_enough_history(frames):
    *_, dev, _ = frames
    # iq starts 2003-01 in the synthetic data (12 weeks of it are history only)
    early = dev.loc[dev.week_start_date.lt("2003-09-01")]    # ~34 iq weeks, ~22 usable: below MIN_OTHER_CITY_ROWS
    later = dev.loc[dev.week_start_date.lt("2005-01-01")]    # ~104 iq weeks: enough
    assert fitting_rows(early, "pooled", "sj").city.eq("sj").all()
    assert fitting_rows(later, "pooled", "sj").city.eq("iq").any()
    assert fitting_rows(later, "separate", "sj").city.eq("sj").all()
    # the target city itself is always kept, however little history it has
    assert fitting_rows(early, "pooled", "iq").city.eq("iq").any()


def test_fitting_rows_exclude_ineligible_history(frames):
    *_, dev, _ = frames
    assert fitting_rows(dev, "separate", "sj").eligible.all()


def test_overlapping_fold_is_detected(frames):
    *_, dev, _ = frames
    annual, _ = make_splits(dev)
    bad = annual[0].__class__(**{**annual[0].__dict__, "cutoff": dev.week_start_date.max()})
    with pytest.raises(ValueError, match="overlaps"):
        audit_splits([bad], dev)
