"""Poisson and Negative Binomial (NB2) GLM fitters, with optional ridge penalty.

NB2 variance is mu + alpha * mu^2. alpha is not estimated jointly: it is either fixed (tuning) or set from the
moment formula on a Poisson fit, then the NB GLM is fitted once with that alpha.
"""
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy.stats import nbinom

from .config import ALPHA_BOUNDS, UNPENALISED_COLUMNS


class GLMFit(NamedTuple):
    params: np.ndarray
    alpha: float       # NaN for Poisson
    converged: bool
    at_bound: bool     # alpha ended at the edge of ALPHA_BOUNDS
    solver: str


def nb_loglik(y, mu, alpha):
    return nbinom.logpmf(y, 1.0 / alpha, 1.0 / (1.0 + alpha * mu))


def moment_alpha(y, mu) -> float:
    """E[(y - mu)^2] = mu + alpha * mu^2  =>  alpha = sum((y - mu)^2 - y) / sum(mu^2), clipped to ALPHA_BOUNDS."""
    return float(np.clip(np.sum((y - mu) ** 2 - y) / np.sum(mu ** 2), *ALPHA_BOUNDS))


def alpha_hits_bound(alpha: float) -> bool:
    return bool(alpha <= ALPHA_BOUNDS[0] * 1.01 or alpha >= ALPHA_BOUNDS[1] * 0.99)


def _robust_fit(model, start_params=None):
    """IRLS first; if it does not converge, BFGS then L-BFGS from the IRLS solution. Returns (result, converged, solver)."""
    result = model.fit(start_params=start_params, maxiter=200)
    if result.converged and np.all(np.isfinite(result.params)):
        return result, True, "irls"
    for method in ("bfgs", "lbfgs"):
        try:
            alt = model.fit(start_params=np.asarray(result.params), method=method, maxiter=5000, disp=0)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if bool(alt.mle_retvals["converged"]) and np.all(np.isfinite(alt.params)):
            return alt, True, method
    return result, False, "irls"


def fit_poisson(X, y) -> GLMFit:
    result, converged, solver = _robust_fit(sm.GLM(y, X, family=sm.families.Poisson()))
    return GLMFit(np.asarray(result.params), np.nan, converged, False, solver)


def fit_negative_binomial(X, y) -> GLMFit:
    poisson_fit = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=200)
    alpha = moment_alpha(y, np.asarray(poisson_fit.fittedvalues))
    result, converged, solver = _robust_fit(sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)),
                                            start_params=np.asarray(poisson_fit.params))
    return GLMFit(np.asarray(result.params), alpha, converged, alpha_hits_bound(alpha), solver)


def penalty_weights(lam: float, columns: list) -> np.ndarray:
    """Ridge weight per column; the intercept and the city indicator are not penalised."""
    return lam * np.array([0.0 if c in UNPENALISED_COLUMNS else 1.0 for c in columns])


def fit_ridge(X, y, base_family: str, lam: float, columns: list, fixed_alpha=None) -> GLMFit:
    """Ridge GLM minimising -loglik/n + lam/2 * sum(beta^2).

    statsmodels' ridge routine reports no convergence flag, so `converged` only means finite coefficients.
    For NB, alpha comes from the moment formula on the current penalised fit (3 refits), or is fixed if given.
    """
    weights = penalty_weights(lam, columns)
    params = np.asarray(sm.GLM(y, X, family=sm.families.Poisson()).fit_regularized(alpha=weights, L1_wt=0.0).params)
    alpha = np.nan
    if base_family == "negative_binomial":
        for _ in range(1 if fixed_alpha is not None else 3):
            mu = np.exp(np.clip(X @ params, -30, 30))
            alpha = float(fixed_alpha) if fixed_alpha is not None else moment_alpha(y, mu)
            params = np.asarray(sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit_regularized(
                alpha=weights, L1_wt=0.0, start_params=params).params)
    at_bound = bool(base_family == "negative_binomial" and alpha_hits_bound(alpha))
    return GLMFit(params, alpha, bool(np.all(np.isfinite(params))), at_bound, "ridge")
