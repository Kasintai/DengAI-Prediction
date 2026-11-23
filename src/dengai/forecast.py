"""Forecasting a block of future weeks: the locked report window (an honest estimate) and the real test set.

Both use the same function, so the model that is submitted is built exactly the way it was validated:
for each city, fit on rows dated before that city's first target week, then predict every target week in one
pass. The final model uses only the seasonal curve, so no forecast depends on earlier forecasts or on cases.
"""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .baselines import BASELINE_VARIANTS, baseline_forecast
from .config import CITIES, KEYS, N_THREADS, TARGET
from .evaluation import test_mix
from .models import fit_candidate, predict_candidate


def forecast_block(config: dict, history: pd.DataFrame, targets: pd.DataFrame, season_cols: list) -> tuple:
    """Fit `config` per city on `history` rows before that city's first target week and forecast its target weeks.

    Returns (predictions with KEYS, week_start_date, pred and, if known, the label; fitted models by city).
    """
    parts, models = [], {}
    with threadpool_limits(limits=N_THREADS):
        for city in CITIES:
            rows = targets.loc[targets.city.eq(city)].sort_values("week_start_date")
            available = history.loc[history.week_start_date.lt(rows.week_start_date.min())]
            if not available.week_start_date.max() < rows.week_start_date.min():
                raise ValueError("history overlaps the forecast period")
            model = fit_candidate(config["family"], config["architecture"], config["feature_set"], available, city,
                                  season_cols, config.get("hyper"))
            models[city] = model
            pred = predict_candidate(model, rows)
            out = rows[KEYS + ["week_start_date"] + ([TARGET] if TARGET in rows else [])].copy()
            out["pred"] = np.maximum(pred.point, 0)
            out["weeks_ahead"] = np.arange(1, len(out) + 1)
            parts.append(out)
    return pd.concat(parts, ignore_index=True), models


def score_report_window(report_pred: pd.DataFrame, config: dict, dev: pd.DataFrame, report: pd.DataFrame,
                        weights: dict, label: str) -> pd.DataFrame:
    """MAE of the final model and every baseline on the report window (baselines fitted on the same dev rows)."""
    def mix(errors_by_city):
        mae = {c: float(np.mean(errors_by_city[c])) for c in CITIES}
        return {"mae_sj": mae["sj"], "mae_iq": mae["iq"], "mae_test_mix": test_mix(mae, weights)}

    errors = {c: np.abs(g[TARGET].to_numpy(float) - g.pred.to_numpy(float))
              for c, g in report_pred.groupby("city")}
    rows = {label: mix(errors)}
    for kind, architecture in BASELINE_VARIANTS:
        base_errors = {}
        for city in CITIES:
            valid = report.loc[report.city.eq(city)].sort_values("week_start_date")
            available = dev.loc[dev.week_start_date.lt(valid.week_start_date.min())]
            base_errors[city] = np.abs(valid[TARGET].to_numpy(float)
                                       - baseline_forecast(kind, architecture, available, valid, city))
        rows[f"{kind} ({architecture})"] = mix(base_errors)
    return pd.DataFrame(rows).T.sort_values("mae_test_mix")


def error_by_forecast_year(report_pred: pd.DataFrame) -> pd.DataFrame:
    """MAE per 52-week block of the report window (year 1 = first 52 weeks)."""
    frame = report_pred.assign(window_year=(report_pred.weeks_ahead - 1) // 52 + 1,
                               abs_error=(report_pred[TARGET] - report_pred.pred).abs())
    return frame.pivot_table(index="window_year", columns="city", values="abs_error", aggfunc="mean")[CITIES]


def make_submission(test_pred: pd.DataFrame, submission_format: pd.DataFrame) -> pd.DataFrame:
    """Predictions in the exact row order and columns of the competition's submission template."""
    submission = submission_format[KEYS].merge(test_pred[KEYS + ["pred"]], on=KEYS, how="left", validate="one_to_one")
    if submission.pred.isna().any():
        raise ValueError("some submission rows have no prediction")
    submission[TARGET] = np.rint(submission.pred).astype(int)
    if (submission[TARGET] < 0).any():
        raise ValueError("negative predictions")
    return submission[list(submission_format.columns)]
