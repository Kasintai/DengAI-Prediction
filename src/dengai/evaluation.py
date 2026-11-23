"""Backtesting inside the development region and scoring.

Metric: MAE per city, combined with the test set's city weights (`mae_test_mix`). `selection_score` averages the
annual-fold view and the long-fold view, because the real test mixes forecast horizons from 1 to 260 weeks.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import BASELINE_VARIANTS, REFERENCE_BASELINE, baseline_forecast
from .config import ARCHITECTURES, CITIES, FAMILIES, GLM_FAMILIES, TARGET, WEATHER_SETS
from .models import fit_candidate, predict_candidate
from .splits import Split, fold_data

CAND_KEYS = ["family", "architecture", "feature_set"]
CANDIDATES = [(f, a, s) for f in FAMILIES for a in ARCHITECTURES for s in WEATHER_SETS]


def test_mix(by_city: dict, weights: dict) -> float:
    """Weighted average of a per-city quantity using the test set's city weights."""
    return float(sum(by_city[c] * weights[c] for c in CITIES))


@dataclass
class Backtest:
    """Everything needed to score a configuration on the development folds."""
    dev: pd.DataFrame
    season_cols: list
    weights: dict
    annual: list   # Split
    devlong: list  # Split


def evaluate_splits(bt: Backtest, splits: list, candidates=CANDIDATES) -> tuple:
    """Fit every candidate on every split. Returns (validation predictions, fit log)."""
    blocks, log = [], []
    for split in splits:
        available, valid = fold_data(split, bt.dev)
        for family, architecture, feature_set in candidates:
            model = fit_candidate(family, architecture, feature_set, available, split.city, bt.season_cols)
            pred = predict_candidate(model, valid)
            log.append({"evaluation": split.evaluation, "city": split.city, "fold": split.fold, "family": family,
                        "architecture": architecture, "feature_set": feature_set, "alpha": model.alpha,
                        "converged": model.converged, "solver": model.solver, "alpha_at_bound": model.at_bound,
                        "n_fit_target_city": model.n_fit_target_city, "n_fit_other_city": model.n_fit_other_city})
            block = valid[["city", "week_start_date", TARGET]].copy()
            block["pred"], block["mean_pred"], block["loglik"] = pred.point, pred.mean, pred.loglik
            for name, value in zip(CAND_KEYS, (family, architecture, feature_set)):
                block[name] = value
            block["fold"], block["evaluation"] = split.fold, split.evaluation
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(log)


def _point_errors(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred[TARGET].to_numpy(float)
    out = pred[CAND_KEYS + ["city", "loglik"]].copy()
    point = np.maximum(pred["pred"].to_numpy(float), 0)
    out["abs_error"] = np.abs(y - point)
    out["abs_error_integer"] = np.abs(y - np.rint(point))
    out["abs_error_mean"] = np.abs(y - np.maximum(pred["mean_pred"].to_numpy(float), 0))
    return out


def summarize(pred: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """One row per candidate: per-city and test-mix MAE, mean log-likelihood, pooled MAEs."""
    errors = _point_errors(pred)
    by_city = errors.groupby(CAND_KEYS + ["city"]).agg(
        mae=("abs_error", "mean"), loglik=("loglik", "mean"), mae_mean=("abs_error_mean", "mean")).unstack("city")
    out = pd.DataFrame({"mae_sj": by_city[("mae", "sj")], "mae_iq": by_city[("mae", "iq")]})
    out["mae_test_mix"] = [test_mix({"sj": s, "iq": i}, weights) for s, i in zip(out.mae_sj, out.mae_iq)]
    out["loglik_test_mix"] = [test_mix({"sj": s, "iq": i}, weights)
                              for s, i in zip(by_city[("loglik", "sj")], by_city[("loglik", "iq")])]
    out["mean_pred_mae_test_mix"] = [test_mix({"sj": s, "iq": i}, weights)
                                     for s, i in zip(by_city[("mae_mean", "sj")], by_city[("mae_mean", "iq")])]
    overall = errors.groupby(CAND_KEYS).agg(mae_overall=("abs_error", "mean"),
                                            mae_integer_overall=("abs_error_integer", "mean"),
                                            n_valid=("abs_error", "size"))
    return out.join(overall).reset_index()


def build_comparison(annual_pred, long_pred, fit_log, weights) -> pd.DataFrame:
    """Annual-fold scores, long-fold scores, convergence counts and the selection score, one row per candidate."""
    comparison = summarize(annual_pred, weights)
    long_summary = summarize(long_pred, weights)[
        CAND_KEYS + ["mae_sj", "mae_iq", "mae_test_mix", "mae_overall", "mean_pred_mae_test_mix", "n_valid"]]
    long_summary = long_summary.rename(columns={c: "devlong_" + c for c in long_summary.columns if c not in CAND_KEYS})
    comparison = comparison.merge(long_summary, on=CAND_KEYS, validate="one_to_one")
    not_converged = fit_log.groupby(CAND_KEYS).agg(fits_not_converged=("converged", lambda s: int((~s).sum()))).reset_index()
    comparison = comparison.merge(not_converged, on=CAND_KEYS, validate="one_to_one")
    comparison["selection_score"] = 0.5 * comparison.mae_test_mix + 0.5 * comparison.devlong_mae_test_mix
    return comparison.sort_values(["selection_score"] + CAND_KEYS).reset_index(drop=True)


def baseline_comparison(bt: Backtest) -> pd.DataFrame:
    """Baseline MAEs on the same folds as the models (annual_* and devlong_* columns)."""
    def table(splits):
        rows = []
        for split in splits:
            available, valid = fold_data(split, bt.dev)
            for kind, architecture in BASELINE_VARIANTS:
                point = baseline_forecast(kind, architecture, available, valid, split.city)
                rows.append({"baseline": kind, "architecture": architecture, "city": split.city,
                             "abs_error": np.abs(valid[TARGET].to_numpy(float) - point)})
        long = pd.DataFrame(rows).explode("abs_error").astype({"abs_error": float})
        mae = long.groupby(["baseline", "architecture", "city"]).abs_error.mean().unstack("city")
        mae["mae_test_mix"] = [test_mix({"sj": s, "iq": i}, bt.weights) for s, i in zip(mae.sj, mae.iq)]
        return mae

    out = table(bt.annual).add_prefix("annual_").join(table(bt.devlong).add_prefix("devlong_"))
    return out.reset_index().sort_values("annual_mae_test_mix").reset_index(drop=True)


def reference_baseline_scores(baselines: pd.DataFrame) -> tuple:
    """(annual, devlong) mae_test_mix of the baseline a model has to beat."""
    kind, architecture = REFERENCE_BASELINE
    row = baselines.set_index(["baseline", "architecture"]).loc[(kind, architecture)]
    return float(row.annual_mae_test_mix), float(row.devlong_mae_test_mix)


def cv_score(bt: Backtest, family: str, architecture: str, feature_set: str, hyper=None) -> dict:
    """Annual and long-fold mae_test_mix and their average (`selection_score`) for one configuration."""
    errors = {"annual": {c: [] for c in CITIES}, "devlong": {c: [] for c in CITIES}}
    for splits in (bt.annual, bt.devlong):
        for split in splits:
            available, valid = fold_data(split, bt.dev)
            model = fit_candidate(family, architecture, feature_set, available, split.city, bt.season_cols, hyper)
            point = predict_candidate(model, valid).point
            errors[split.evaluation][split.city].append(np.abs(valid[TARGET].to_numpy(float) - np.maximum(point, 0)))
    mix = {k: test_mix({c: np.concatenate(v[c]).mean() for c in CITIES}, bt.weights) for k, v in errors.items()}
    return {"annual": mix["annual"], "devlong": mix["devlong"],
            "selection_score": (mix["annual"] + mix["devlong"]) / 2}


def fit_health(fit_log: pd.DataFrame) -> pd.DataFrame:
    """Convergence summary of the GLM fits."""
    glm = fit_log.loc[fit_log.family.isin(GLM_FAMILIES)]
    return glm.groupby(["family", "architecture"], sort=False).agg(
        fits=("converged", "size"), not_converged=("converged", lambda s: int((~s).sum())),
        solved_by_fallback=("solver", lambda s: int(s.isin(["bfgs", "lbfgs"]).sum())),
        alpha_at_bound=("alpha_at_bound", "sum"), alpha_median=("alpha", "median"))
