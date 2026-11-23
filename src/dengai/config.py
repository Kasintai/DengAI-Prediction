"""Constants shared across the pipeline. Nothing here reads data or has side effects."""
from pathlib import Path

# ---- Paths (overridable from the CLI) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_FILES = {
    "train_features": "dengue_features_train.csv",
    "train_labels": "dengue_labels_train.csv",
    "test_features": "dengue_features_test.csv",
    "submission_format": "submission_format.csv",
}

# ---- Schema ----
KEYS = ["city", "year", "weekofyear"]  # `year` is a row key only; it is never a model input
TARGET = "total_cases"
CITIES = ["sj", "iq"]
CITY_NAMES = {"sj": "San Juan", "iq": "Iquitos"}

# ---- Cleaning ----
ABSOLUTE_TEMP_K = [
    "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
]
DROP_DUPLICATE = "precipitation_amt_mm"  # exact duplicate of reanalysis_sat_precip_amt_mm
FFILL_LIMIT = 4                          # carry a value forward at most this many weeks

# ---- Features ----
N_HARMONICS = 2
CORE = ["reanalysis_specific_humidity_g_per_kg", "station_avg_temp_c", "precip_log1p", "reanalysis_tdtr_c"]
LAG_BLOCKS = {"b1_4": (1, 4), "b5_8": (5, 8), "b9_12": (9, 12)}  # mean of weeks t-first ... t-last
MAX_HISTORY = 12  # the first 12 weeks of each city are history only (needed by the longest lag block)

# Which weather columns each feature set adds on top of the seasonal curve
WEATHER_SETS = {
    "seasonal": [],
    "no_weather_lag": ["L0"],
    "with_weather_lag": ["L0"] + list(LAG_BLOCKS),
}

# ---- Models ----
GLM_FAMILIES = [
    "poisson", "poisson_ridge_0.1", "poisson_ridge_1",
    "negative_binomial", "negative_binomial_ridge_0.1", "negative_binomial_ridge_1",
]
TREE_FAMILIES = ["random_forest", "lightgbm_mae", "xgboost_mae"]
FAMILIES = GLM_FAMILIES + TREE_FAMILIES
ARCHITECTURES = ["separate", "pooled"]  # pooled = shared coefficients + city intercept
ALPHA_BOUNDS = (1e-3, 20.0)             # allowed range of the Negative Binomial dispersion
UNPENALISED_COLUMNS = ("const", "city_iq")

# ---- Validation ----
N_FOLDS = 3                # annual folds per city
FOLD_WEEKS = 52
DEV_LONG_WEEKS = 104       # one long fold: the last 104 development weeks predicted in one pass
MIN_OTHER_CITY_ROWS = 52   # a pooled fit only includes the other city if it has this many fitting rows

# ---- Tuning grid (Negative Binomial ridge) ----
NB_LAMBDAS = [0.03, 0.1, 0.3, 1, 3, 10, 30]
NB_ALPHAS = ["moment", 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # 'moment' = estimate alpha by moments

# ---- Final model ----
# Chosen by looking at the development folds (see README). The feature set is fixed to the
# simplest one because the weather-lag variant was within fold-to-fold noise of it.
FINAL_FEATURE_SET = "seasonal"
FINAL_FAMILY = "negative_binomial_ridge_1"  # only carries the model type; lambda/alpha come from tuning

# ---- Runtime ----
RANDOM_STATE = 42
N_THREADS = 2
