"""Grid search for the Negative Binomial ridge model, on development folds only.

Caution: the winner is chosen on the same folds that score it, so its score is optimistic. The report window
(see forecast.py) is the unbiased check.
"""
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ARCHITECTURES, N_THREADS, NB_ALPHAS, NB_LAMBDAS
from .evaluation import Backtest, cv_score

NB_RIDGE = "negative_binomial_ridge_1"  # family name only selects the model type; lam/alpha come from the grid


def tune_nb_ridge(bt: Backtest, feature_set: str) -> pd.DataFrame:
    """Grid over architecture x lambda x alpha, best `selection_score` first. alpha='moment' estimates it per fit."""
    rows = []
    with threadpool_limits(limits=N_THREADS):
        for architecture in ARCHITECTURES:
            for lam in NB_LAMBDAS:
                for alpha in NB_ALPHAS:
                    hyper = {"lam": lam} if alpha == "moment" else {"lam": lam, "alpha": alpha}
                    rows.append({"architecture": architecture, "lam": lam, "alpha": alpha,
                                 **cv_score(bt, NB_RIDGE, architecture, feature_set, hyper)})
    return pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)


def best_config(tuning: pd.DataFrame, family: str, feature_set: str) -> dict:
    """The lowest-selection-score row of a tuning table as a model configuration."""
    best = tuning.iloc[0]
    hyper = {"lam": float(best.lam)}
    if best.alpha != "moment":
        hyper["alpha"] = float(best.alpha)
    return {"family": family, "architecture": best.architecture, "feature_set": feature_set, "hyper": hyper}
