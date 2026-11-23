"""One interface for every candidate model: `fit_candidate` and `predict_candidate`.

A candidate is (family, architecture, feature_set). For a given feature set and architecture every family
receives the same input columns, so differences between rows of the comparison are differences between models.
"""
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import nbinom, poisson
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .config import CITIES, GLM_FAMILIES, RANDOM_STATE, TARGET, TREE_FAMILIES
from .features import weather_columns
from .glm import fit_negative_binomial, fit_poisson, fit_ridge, nb_loglik
from .splits import fitting_rows


class Prediction(NamedTuple):
    point: np.ndarray  # primary point forecast: GLM median (minimises absolute error), tree output
    mean: np.ndarray   # GLM mean (NaN for trees)
    loglik: np.ndarray # log-likelihood of the observed counts (NaN for trees, and when labels are unknown)


@dataclass
class FittedModel:
    family: str
    architecture: str
    feature_set: str
    season_cols: list
    kind: str                       # 'glm' or 'tree'
    n_fit_target_city: int
    n_fit_other_city: int
    estimator: Any = None           # tree estimator
    params: Optional[np.ndarray] = None
    alpha: float = np.nan
    stats: dict = field(default_factory=dict)   # per-city (mean, sd) used to standardise weather columns
    columns: list = field(default_factory=list)
    cities: list = field(default_factory=list)
    converged: bool = True
    at_bound: bool = False
    solver: str = "tree"


def _glm_numeric_columns(feature_set: str) -> list:
    return weather_columns(feature_set)


def make_design(rows, architecture, feature_set, season_cols, cities, stats) -> pd.DataFrame:
    """GLM design matrix: intercept, (pooled) city indicator, seasonal columns, then weather columns
    standardised per city with `stats` (computed on fitting rows only)."""
    numeric = _glm_numeric_columns(feature_set)
    z = pd.DataFrame(index=rows.index, columns=numeric, dtype=float)
    for city in cities:
        mask = rows.city.eq(city)
        mean, sd = stats[city]
        z.loc[mask] = (rows.loc[mask, numeric] - mean) / sd
    design = {"const": np.ones(len(rows))}
    if architecture == "pooled":
        for city in cities[1:]:
            design[f"city_{city}"] = rows.city.eq(city).to_numpy(float)
    elif architecture != "separate":
        raise ValueError(architecture)
    for c in season_cols:
        design[c] = rows[c].to_numpy(float)
    for c in numeric:
        design[c] = z[c].to_numpy(float)
    return pd.DataFrame(design, index=rows.index)


def tree_matrix(rows, architecture, feature_set, season_cols) -> np.ndarray:
    X = rows[weather_columns(feature_set) + season_cols].copy()
    if architecture == "pooled":
        X["is_iq"] = rows.city.eq("iq").to_numpy(float)
    return X.to_numpy(float)


def make_tree(family: str, hyper: Optional[dict] = None):
    if family == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=1.0,
                                     random_state=RANDOM_STATE, n_jobs=1)
    if family == "lightgbm_mae":
        settings = dict(objective="l1", n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                        reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        return LGBMRegressor(**{**settings, **(hyper or {})})
    if family == "xgboost_mae":  # mirrors the LightGBM settings so they differ by library, not by configuration
        return XGBRegressor(objective="reg:absoluteerror", n_estimators=200, learning_rate=0.05, tree_method="hist",
                            grow_policy="lossguide", max_leaves=7, max_depth=0, min_child_weight=20, reg_lambda=1.0,
                            random_state=RANDOM_STATE, n_jobs=1)
    raise ValueError(family)


def fit_candidate(family, architecture, feature_set, available, city, season_cols, hyper=None) -> FittedModel:
    """Fit one model for `city` on the eligible rows of `available` (dated before the cutoff).

    `hyper` overrides defaults: {'lam', 'alpha'} for NB ridge, LightGBM settings for lightgbm_mae.
    """
    hyper = hyper or {}
    fit_rows = fitting_rows(available, architecture, city)
    base = dict(family=family, architecture=architecture, feature_set=feature_set, season_cols=season_cols,
                n_fit_target_city=int(fit_rows.city.eq(city).sum()),
                n_fit_other_city=int((~fit_rows.city.eq(city)).sum()))
    y = fit_rows[TARGET].to_numpy(float)

    if family in TREE_FAMILIES:
        estimator = make_tree(family, hyper).fit(tree_matrix(fit_rows, architecture, feature_set, season_cols), y)
        return FittedModel(kind="tree", estimator=estimator, **base)
    if family not in GLM_FAMILIES:
        raise ValueError(f"unknown family {family}")

    cities = [c for c in CITIES if c in set(fit_rows.city)]
    numeric = _glm_numeric_columns(feature_set)
    stats = {}
    for c in cities:
        part = fit_rows.loc[fit_rows.city.eq(c), numeric]
        stats[c] = (part.mean(), part.std().replace(0, 1.0).fillna(1.0))

    X = make_design(fit_rows, architecture, feature_set, season_cols, cities, stats)
    if "_ridge_" in family:
        base_family, lam = family.rsplit("_ridge_", 1)
        fit = fit_ridge(X.to_numpy(float), y, base_family, hyper.get("lam", float(lam)), list(X.columns),
                        fixed_alpha=hyper.get("alpha"))
    else:
        fit = (fit_poisson if family == "poisson" else fit_negative_binomial)(X.to_numpy(float), y)
    return FittedModel(kind="glm", params=fit.params, alpha=fit.alpha, stats=stats, columns=list(X.columns),
                       cities=cities, converged=fit.converged, at_bound=fit.at_bound, solver=fit.solver, **base)


def predict_candidate(model: FittedModel, rows: pd.DataFrame) -> Prediction:
    """Forecast `rows`. Labels in `rows`, if present, are used only to compute the log-likelihood."""
    n = len(rows)
    if model.kind == "tree":
        point = np.maximum(model.estimator.predict(
            tree_matrix(rows, model.architecture, model.feature_set, model.season_cols)), 0)
        return Prediction(point, np.full(n, np.nan), np.full(n, np.nan))

    X = make_design(rows, model.architecture, model.feature_set, model.season_cols, model.cities, model.stats)
    assert list(X.columns) == model.columns
    mu = np.exp(np.clip(X.to_numpy(float) @ model.params, -30, 30))
    if model.family.startswith("poisson"):
        median = poisson.ppf(0.5, mu)
        dist_loglik = lambda y: poisson.logpmf(y, mu)
    else:
        median = nbinom.ppf(0.5, 1.0 / model.alpha, 1.0 / (1.0 + model.alpha * mu))
        dist_loglik = lambda y: nb_loglik(y, mu, model.alpha)
    loglik = dist_loglik(rows[TARGET].to_numpy(float)) if TARGET in rows and rows[TARGET].notna().all() else np.full(n, np.nan)
    return Prediction(median, mu, loglik)
