"""End-to-end workflow: validate -> features -> split -> backtest -> tune -> report-window check -> submission.

Selection only ever sees the development region. The report window is read once, after the configuration is
fixed, and the final model is then refitted on all training data to forecast the real test period.
"""
import json
import logging
import platform
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from . import __version__
from .config import CITIES, FINAL_FAMILY, FINAL_FEATURE_SET, N_THREADS
from .data import RawData, load_raw_data, validate_raw_data
from .evaluation import (CANDIDATES, Backtest, baseline_comparison, build_comparison, evaluate_splits, fit_health,
                         reference_baseline_scores)
from .features import build_modelling_frame, split_train_test
from .forecast import error_by_forecast_year, forecast_block, make_submission, score_report_window
from .splits import audit_splits, locked_split, make_splits
from .tuning import best_config, tune_nb_ridge

log = logging.getLogger("dengai")
FINAL_LABEL = "NB ridge, seasonal (final)"


def _save(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    frame.to_csv(path, index=index)
    log.info("wrote %s", path)


def run(raw_dir: Path, output_dir: Path, compare: bool = True) -> dict:
    """Run the full workflow and write all artefacts to `output_dir`. Returns the final configuration and scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw: RawData = load_raw_data(raw_dir)
    validate_raw_data(raw)
    log.info("raw data validated; test weeks per city: %s", raw.test_counts)

    weather, season_cols = build_modelling_frame(raw)
    training, test = split_train_test(weather)

    # ---- Locked report window: from here on, selection uses `dev` only ----
    dev, report = locked_split(training, raw.test_counts)
    annual, devlong = make_splits(dev)
    bt = Backtest(dev, season_cols, raw.test_weights, annual, devlong)
    _save(audit_splits(annual + devlong, dev), output_dir / "split_audit.csv")

    baselines = baseline_comparison(bt)
    _save(baselines, output_dir / "baseline_comparison.csv")
    baseline_annual, baseline_long = reference_baseline_scores(baselines)
    log.info("reference baseline (seasonal median, separate): annual %.3f, long %.3f", baseline_annual, baseline_long)

    if compare:
        log.info("comparing %d candidates on all development folds", len(CANDIDATES))
        with threadpool_limits(limits=N_THREADS):
            annual_pred, annual_log = evaluate_splits(bt, annual)
            long_pred, long_log = evaluate_splits(bt, devlong)
        fit_log = pd.concat([annual_log, long_log], ignore_index=True)
        comparison = build_comparison(annual_pred, long_pred, fit_log, bt.weights)
        _save(comparison, output_dir / "candidate_comparison.csv")
        _save(fit_health(fit_log), output_dir / "fit_health.csv", index=True)
        log.info("best candidate by selection_score:\n%s",
                 comparison.head(5)[["family", "architecture", "feature_set", "mae_test_mix",
                                     "devlong_mae_test_mix", "selection_score"]].to_string())

    # ---- Tune the Negative Binomial ridge on development folds ----
    tuning = {fs: tune_nb_ridge(bt, fs) for fs in ("seasonal", "with_weather_lag")}
    for fs, table in tuning.items():
        _save(table, output_dir / f"tuning_nb_ridge_{fs}.csv")
    final = best_config(tuning[FINAL_FEATURE_SET], FINAL_FAMILY, FINAL_FEATURE_SET)
    dev_score = float(tuning[FINAL_FEATURE_SET].iloc[0].selection_score)
    log.info("final configuration %s (development selection_score %.3f)", final, dev_score)

    # ---- Report window: one honest look, with the configuration already fixed ----
    report_pred, _ = forecast_block(final, dev, report, season_cols)
    score = score_report_window(report_pred, final, dev, report, raw.test_weights, FINAL_LABEL)
    _save(report_pred, output_dir / "report_window_predictions.csv")
    _save(score, output_dir / "report_window_score.csv", index=True)
    _save(error_by_forecast_year(report_pred), output_dir / "report_window_error_by_year.csv", index=True)
    log.info("report-window scores:\n%s", score.round(3).to_string())

    # ---- Refit on all training data and forecast the real test period ----
    test_pred, models = forecast_block(final, training, test, season_cols)
    for city in CITIES:
        log.info("%s: final fit on %d own + %d other-city weeks, alpha %.2f", city,
                 models[city].n_fit_target_city, models[city].n_fit_other_city, models[city].alpha)
    submission = make_submission(test_pred, raw.submission_format)
    _save(submission, output_dir / "submission.csv")

    summary = {
        "final_configuration": final,
        "development_selection_score": dev_score,
        "report_window_mae_test_mix": float(score.loc[FINAL_LABEL, "mae_test_mix"]),
        "reference_baseline_report_window_mae_test_mix": float(score.loc["seasonal_median (separate)", "mae_test_mix"]),
        "n_submission_rows": len(submission),
        "versions": {"python": platform.python_version(), "dengai": __version__},
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
