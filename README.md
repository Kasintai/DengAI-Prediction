# DengAI: forecasting weekly dengue cases in San Juan and Iquitos

Take-home solution for the DrivenData [DengAI](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) challenge:
predict weekly `total_cases` for `sj` (San Juan, 260 test weeks) and `iq` (Iquitos, 156 test weeks) from climate data.
The metric is mean absolute error (MAE).

## Summary

- **Recommendation:** a per-city *seasonal curve* fitted with a Negative Binomial ridge model. No weather inputs, no lagged cases.
- **Why it wins:** on a held-out window the same length as the real test, it beats the best honest baseline (seasonal median)
  by about 12%: **MAE 11.9 vs 13.5**. It also beats 53 other candidates (Poisson, Negative Binomial, random forest,
  LightGBM, XGBoost; with and without lagged weather) on time-ordered validation folds.
- **Key finding:** weather did not help out of sample. In the controlled comparison, adding same-week or lagged weather made most
  models *worse*, even though lagged weather correlates with cases in the EDA (that correlation is inflated by shared seasonality).
- **What it cannot do:** predict outbreaks. Cases are dominated by a few huge, unforecastable years, so the forecast is
  a typical-year seasonal level, and the error is large in outbreak years (see [Limitations](#limitations)).
- **Honest caveat:** the gain over the baseline is modest, and on Iquitos the model is *worse* than the seasonal median on the report window (8.5 vs 7.5).

## Reproduce

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 1. Download the four CSVs from the DrivenData "Data" tab into data/raw/
#    (dengue_features_train.csv, dengue_labels_train.csv, dengue_features_test.csv, submission_format.csv).
#    They are restricted competition data, so they are NOT in this repository.

# 2. Run everything (about 2 minutes): validation, comparison, tuning, report-window check, submission
python -m dengai run

# Faster: skip the 54-candidate comparison (it does not affect the submission)
python -m dengai run --skip-compare

# Tests (synthetic data; no competition files needed)
pip install -r requirements-dev.txt && pytest
```

The run is deterministic (fixed seeds, no randomness in the final model) and writes to `outputs/`:

| File | Contents |
|---|---|
| `submission.csv` | **Test-set predictions** in the exact format of `submission_format.csv` |
| `run_summary.json` | Final configuration and headline scores |
| `report_window_score.csv`, `report_window_predictions.csv`, `report_window_error_by_year.csv` | The honest estimate and its breakdown |
| `candidate_comparison.csv`, `baseline_comparison.csv`, `fit_health.csv` | Development-fold comparison of all candidates and baselines |
| `tuning_nb_ridge_*.csv` | Hyperparameter grid results |
| `split_audit.csv` | How much history each validation fold has |

Options: `--raw-dir PATH`, `--output-dir PATH`. Everything else is in `src/dengai/config.py`.

## Problem and approach

### Data and what the EDA showed
Full analysis in [`notebooks/EDA_Final_23_11_2026.ipynb`](notebooks/EDA_Final_23_11_2026.ipynb). The points that drove the design:

| Finding | Consequence |
|---|---|
| Test is a **forecast**: each city's test weeks come strictly after its training weeks (5 years `sj`, 3 years `iq`) | Time-ordered validation only; no random splits |
| Counts are strongly **overdispersed** (variance/mean ≈ 77 `sj`, 15 `iq`; NB α ≈ 1.6) | Negative Binomial rather than Poisson |
| The two cities differ: scale (mean 34 vs 7.6), seasonal peak (week ~42 vs ~4), four features change sign, cases are negatively correlated | Compare separate vs pooled models; always give pooled models a city term |
| **Level shifts** between years (`sj` yearly mean 9 to 126; `iq` ~0 in 2000–01) | A `year` trend cannot describe the jumps, so `year` is never a feature |
| Case autocorrelation is very high, but test has no case history | Forecasts must not use past cases (would need compounding recursive forecasts) |
| Missingness is small except `ndvi_ne` (~20% of `sj`); one exact duplicate rainfall column; temperatures in Kelvin | Causal imputation, drop the duplicate, convert to °C |

### Validation (the part to scrutinise)
1. **Locked report window.** Before any modelling, the last 260 (`sj`) / 156 (`iq`) training weeks of each city, the same length as the
   test period, are set aside. All selection uses only the earlier *development region*. The window is read once, after the configuration is fixed.
2. **Folds inside development:** per city, three annual folds (expanding history) plus one 104-week "long" fold predicted in one pass, because the real test spans horizons of 1 to 260 weeks.
   `selection_score` = mean of the annual and long-fold MAE.
3. **City weights:** per-city MAE is combined with the test set's weights (`sj` 0.625, `iq` 0.375), so the metric mirrors the real one.
4. **No leakage by construction:** imputation and lag features look backwards only (forward-fill ≤ 4 weeks, then an expanding median of earlier values); weather is standardised on fitting rows only;
   a pooled model only borrows the other city once it has ≥ 52 usable weeks.
   A test scrambles every label after a fold's cutoff and asserts the predictions do not change (`tests/test_models.py`).

### Features and models
- **Features:** two annual sine/cosine harmonics of the day of year ("seasonal"); optionally four core weather variables
  (specific humidity, station temperature, log rainfall, diurnal range) as current-week values and three non-overlapping 4-week lag blocks (weeks t-1..4, t-5..8, t-9..12).
- **Candidates (54):** 9 model families × {separate, pooled} × 3 feature sets (`seasonal`, `no_weather_lag`, `with_weather_lag`).
  Families: Poisson and Negative Binomial GLMs (each plain and ridge λ=0.1 / 1), random forest, LightGBM and XGBoost with MAE loss.
  Every family gets the same columns, so differences are differences between models.
- **Point forecast:** the median of the predictive distribution, which minimises absolute error (the competition metric).
- **Baselines:** overall median, **seasonal median** (median of the same week of year; the bar to beat), and last observed value (not a usable forecast, since it needs case history at forecast time).
- **Tuning:** NB ridge over architecture × λ × α, on development folds only.
- **Final model:** pooled NB ridge (λ = 0.03, α = 3) on the seasonal curve. I fixed the feature set to `seasonal` rather than letting the grid pick: it scored 9.45 vs 9.50 for the weather-lag variant, which is within fold-to-fold noise,
  so I took the simpler model.

## Results

Report window (260 `sj` weeks, 156 `iq` weeks; never used for selection), MAE:

| Model | `sj` | `iq` | Test-weighted |
|---|---|---|---|
| **NB ridge, seasonal (final)** | 13.98 | 8.48 | **11.92** |
| Seasonal median (separate) | 17.11 | **7.53** | 13.52 |
| Seasonal median (pooled) | 15.17 | 11.05 | 13.62 |
| Median (pooled) | 15.96 | 10.51 | 13.92 |
| Median (separate) | 17.96 | 7.61 | 14.08 |
| Last value | 18.17 | 8.45 | 14.53 |

Development folds (used for choosing): best candidate 10.46, tuned final 9.45, seasonal-median baseline 12.10.
Best model per family: NB ridge 10.46 (→ 9.45 tuned), NB 11.48, LightGBM 12.14, XGBoost 12.15; Poisson and random forest ≈ 18.
15 of 54 candidates beat the baseline on the annual folds and 12 on both views.

**Reading these numbers.**
- Development (9.45) is lower than the report window (11.92). That gap is the optimism of picking the best of many configurations on the same folds, which is why the report window exists. **11.92 is the number to quote**, not 9.45.
- Report-window error by forecast year (the notebook run): `sj` 9.0 / 4.0 / 25.6 / 4.1 / 27.1; `iq` 9.1 / 11.7 / 4.6. Error is dominated by two outbreak years in `sj`, not by the horizon.
- Predictions are flat by design: roughly 4–14 cases per week for `sj` and 1–4 for `iq` in the test set.

## Limitations

- **No outbreak forecasting.** The model gives the typical seasonal level; large outbreaks (e.g. `sj` 1994, `iq` 2004) are invisible to it. Reducing MAE further needs information about outbreak onset that the supplied weather does not provide.
- **Weather features are unused.** I did not establish *why* they hurt (plausible: lag windows add noisy columns on ~1,000 autocorrelated rows; year-level shifts swamp weather signal). That is a hypothesis, not a finding.
- **Tuning hit the edge of its grid.** The best α = 3.0 is the largest value searched. The result may improve with a wider grid; I kept the grid as it was so the numbers stay reproducible.
- **Optimistic selection.** Winners are picked on the folds that score them (see gap above). Differences below about half a case are within noise.
- **Pooling is a mixed bag.** The pooled model shares one seasonal shape between cities that peak about six months apart (only the level differs by city), and for early `sj` folds there is no `iq` data, so "pooled" is effectively separate there.
  It still won on the folds, but it conflicts with the EDA's recommendation of city-specific seasonality.
- **`iq` is worse than a plain seasonal median** on the report window (8.48 vs 7.53). `iq` has little data (508 usable weeks) and irregular seasonality.
- **Small, single window.** The report window is one 3 to 5 year stretch per city, a noisy estimate.
- **Final refit and pooling.** For the real forecast each city's model is fitted on all training rows before that city's first test week, and a pooled fit borrows only other-city rows before that date, the same rule that was validated.
- **No leaderboard submission** was made; the leaderboard score is unknown.

## Next steps
1. Widen the α grid and tune λ jointly with it; test city-specific seasonal harmonics inside the pooled model.
2. Model the outbreak scale directly, e.g. a mixture or quantile approach to separate "typical year" from "outbreak year", and report prediction intervals, not just a point forecast.
3. Rolling-origin evaluation with more origins to shrink the noise of a single report window.
4. Revisit weather with a method that handles short, autocorrelated series (e.g. distributed-lag or Bayesian shrinkage), and check whether lagged *cases* help at short horizons in a genuine multi-step setting.
5. Submit to the DrivenData leaderboard as an external check.

## Repository layout

```
src/dengai/
  config.py       all constants (columns, folds, grids, final-model settings)
  data.py         load + validate the raw files (clear error if data is missing)
  features.py     cleaning, causal imputation, seasonal and lag features
  splits.py       locked report window and time-ordered folds
  glm.py          Poisson / Negative Binomial GLMs with optional ridge
  models.py       one fit/predict interface over GLMs and tree models
  baselines.py    naive reference forecasts
  evaluation.py   backtesting, scoring, candidate comparison
  tuning.py       NB ridge grid search
  forecast.py     report-window scoring, final forecast, submission file
  pipeline.py     orchestrates the workflow    __main__.py  CLI
tests/            26 tests on synthetic data: leakage, causality, splits, submission format, end-to-end
notebooks/        exploratory record: EDA, and the original model-selection notebook
data/raw/         put the DrivenData CSVs here (git-ignored)
```

The notebooks are the exploration history. The package reproduces the notebook's model-selection results exactly
(development score 9.451, report-window MAE 11.916); the notebook also contains a LightGBM random search that is not part of the pipeline, since LightGBM did not beat the final model.
