import numpy as np
import pandas as pd

from dengai.config import CORE, LAG_BLOCKS, MAX_HISTORY
from dengai.features import build_modelling_frame, clean_features, impute_weather, weather_columns
from dengai.models import make_design, tree_matrix


def test_kelvin_converted_and_duplicate_dropped(raw):
    cleaned = clean_features(raw.train_features)
    assert "precipitation_amt_mm" not in cleaned
    assert "reanalysis_air_temp_k" not in cleaned
    assert cleaned["reanalysis_air_temp_c"].between(-50, 60).all()


def test_imputation_is_causal():
    """Changing a future value must not change any earlier imputed value."""
    base = pd.DataFrame({"city": "sj", "x": [1.0, np.nan, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0]})
    changed = base.copy()
    changed.loc[8, "x"] = 1000.0
    a, b = impute_weather(base, ["x"]), impute_weather(changed, ["x"])
    assert a.x.iloc[:8].equals(b.x.iloc[:8])


def test_forward_fill_is_limited_then_past_median_is_used():
    values = [1.0, 2.0, 3.0] + [np.nan] * 6
    out = impute_weather(pd.DataFrame({"city": "sj", "x": values}), ["x"])
    assert out.x.iloc[3:7].eq(3.0).all()        # carried forward for FFILL_LIMIT = 4 weeks
    assert out.x.iloc[7] == 2.0                 # then the median of the observed earlier values (1, 2, 3)


def test_lag_block_is_mean_of_earlier_weeks(frames):
    weather = frames[0]
    sj = weather.loc[weather.city.eq("sj")].reset_index(drop=True)
    t = 40
    for block, (first, last) in LAG_BLOCKS.items():
        expected = sj[CORE[0]].iloc[t - last:t - first + 1].mean()
        assert np.isclose(sj[f"{CORE[0]}__{block}"].iloc[t], expected)


def test_first_weeks_are_not_eligible_and_features_complete(frames):
    weather, season_cols, training, *_ = frames
    assert weather.groupby("city").eligible.apply(lambda s: (~s).sum()).eq(MAX_HISTORY).all()
    eligible = training.loc[training.eligible]
    assert not eligible[season_cols + weather_columns("with_weather_lag")].isna().any().any()


def test_year_is_never_a_model_input(frames):
    _, season_cols, training, *_ = frames
    rows = training.loc[training.eligible].head(60)
    design = make_design(rows, "pooled", "with_weather_lag", season_cols, ["sj", "iq"],
                         {c: (rows[weather_columns("with_weather_lag")].mean(), rows[weather_columns("with_weather_lag")].std())
                          for c in ["sj", "iq"]})
    assert "year" not in design.columns
    assert tree_matrix(rows, "pooled", "with_weather_lag", season_cols).shape[1] == 4 + 16 + 1


def test_building_features_does_not_depend_on_test_labels(raw):
    """Test rows carry no label, and training labels attach one-to-one."""
    weather, _ = build_modelling_frame(raw)
    assert weather.loc[weather.split.eq("test"), "total_cases"].isna().all()
    assert weather.loc[weather.split.eq("train"), "total_cases"].notna().all()
