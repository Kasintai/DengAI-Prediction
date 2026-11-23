import numpy as np
import pytest

from dengai.baselines import baseline_forecast
from dengai.config import TARGET
from dengai.models import fit_candidate, predict_candidate
from dengai.splits import fold_data, make_splits

CANDIDATES = [
    ("negative_binomial", "pooled", "with_weather_lag"),
    ("negative_binomial_ridge_1", "pooled", "with_weather_lag"),
    ("lightgbm_mae", "separate", "with_weather_lag"),
]


@pytest.mark.parametrize("family,architecture,feature_set", CANDIDATES)
def test_scrambling_labels_after_the_cutoff_changes_nothing(frames, family, architecture, feature_set):
    """Leakage test: a fold's predictions may only depend on labels dated before its cutoff."""
    _, season_cols, _, _, dev, _ = frames
    split = make_splits(dev)[0][2]  # sj, fold 3
    shuffled = dev.copy()
    later = shuffled.week_start_date.ge(split.cutoff)
    shuffled.loc[later, TARGET] = np.random.default_rng(1).integers(0, 500, size=int(later.sum()))
    outputs = []
    for frame in (dev, shuffled):
        available, valid = fold_data(split, frame)
        model = fit_candidate(family, architecture, feature_set, available, split.city, season_cols)
        outputs.append(predict_candidate(model, valid).point)
    assert np.array_equal(outputs[0], outputs[1])


def test_nb_ridge_forecasts_are_non_negative_integers(frames):
    _, season_cols, _, _, dev, report = frames
    model = fit_candidate("negative_binomial_ridge_1", "pooled", "seasonal", dev, "sj", season_cols,
                          {"lam": 0.1, "alpha": 2.0})
    pred = predict_candidate(model, report.loc[report.city.eq("sj")])
    assert model.alpha == 2.0 and model.converged
    assert (pred.point >= 0).all() and np.array_equal(pred.point, np.round(pred.point))
    assert np.isfinite(pred.loglik).all()


def test_forecast_needs_no_labels(frames):
    """Seasonal-only forecasts must work on rows without a `total_cases` column (the real test set)."""
    _, season_cols, _, test, dev, _ = frames
    model = fit_candidate("negative_binomial_ridge_1", "separate", "seasonal", dev, "iq", season_cols, {"lam": 0.1})
    pred = predict_candidate(model, test.loc[test.city.eq("iq")])
    assert len(pred.point) == 52 and np.isnan(pred.loglik).all()


def test_seasonal_median_baseline_uses_only_history(frames):
    *_, dev, report = frames
    valid = report.loc[report.city.eq("sj")]
    point = baseline_forecast("seasonal_median", "separate", dev, valid, "sj")
    history = dev.loc[dev.city.eq("sj") & dev.eligible]
    week = valid.weekofyear.iloc[0]
    assert point[0] == history.loc[history.weekofyear.eq(week), TARGET].median()
