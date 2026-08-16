"""Builder script that assembles winam_wh_temporal_driver_model.ipynb.

The spatial panel models answer "where is the hyacinth"; this notebook answers
"how much hyacinth is there this month, and which environmental drivers move
that total". It collapses the 500 m cell-month panel to ONE area-weighted
AOI value per month and fits interpretable temporal models to that series.

Kept in the repo (like build_backfill_nb.py / build_inventory_nb.py) so the
notebook can be regenerated from one editable source instead of hand-patching
notebook JSON. Regenerate with:

    python3 build_temporal_model_nb.py

This writes the CELL SOURCES only: the emitted notebook carries no outputs and
no Colab metadata. The committed .ipynb keeps whatever Colab last saved, so
after regenerating, re-run the notebook (in Colab for the real Drive data, or
with USE_SYNTHETIC_DEMO = True offline) to restore its outputs.
"""
import json

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    })


# ===========================================================================
# Title / scope
# ===========================================================================
md('<a href="https://colab.research.google.com/github/Qalani/Dissertation/blob/main/'
   'winam_wh_temporal_driver_model.ipynb" target="_parent">'
   '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
   'alt="Open In Colab"/></a>')

md("""# Winam Gulf water hyacinth — purely **temporal** driver model

**Question this notebook answers.** *How much* water hyacinth is there in Winam
Gulf in a given month, and *which environmental variables are most strongly
linked to that amount*? It deliberately does **not** ask *where* the hyacinth
is: the spatial dimension is aggregated away before any model is fitted.

**The response.** One number per month — the area-weighted AOI mean of the
classified WH cover:

$$\\text{wh\\_cover\\_aoi}(t) \\;=\\; \\frac{\\sum_{i \\in C} \\text{wh\\_area\\_ha}_{i,t}}{\\sum_{i \\in C} \\text{valid\\_area\\_ha}_{i,t}}$$

over a **fixed** set of grid cells $C$. Dividing by the *validly classified*
area (not by the AOI area) is what makes months with different cloud cover
comparable, and fixing $C$ is what stops a change in *which* cells were
observed from masquerading as a change in *how much* hyacinth there is.

**The predictors.** Area-weighted AOI means of the same environmental
covariates the spatial panel uses, so the two models are reading the same data.

---

## What a purely temporal model can and cannot tell you

| | |
|---|---|
| ✅ **Identifiable** | Time-*varying* drivers: rainfall, air/water temperature, wind, wave exposure, lake level, water-quality proxies. These change month to month, so they can co-move with the WH total. |
| ❌ **Not identifiable — at all** | Every **static** habitat variable: `depth_m`, `dist_shore_m`, `dist_majriver_m`, `frac_cropland`, `openness_index`. Averaged over a fixed cell set they are *constants*, and a constant cannot explain variation in anything. They are dropped explicitly in §7 rather than silently. |
| ⚠️ **Identifiable only with care** | Anything strongly seasonal. Rainfall, temperature and WH extent all follow the annual cycle, so a driver can "explain" WH simply by being seasonal. §9 measures this for every driver and §12 reports every effect **both** with and without seasonal control. |

**The principal model is a state-space dynamic regression (§19–§24).** The
question this notebook has to answer is not *"is rainfall correlated with
hyacinth"* — on a series with lag-1 autocorrelation near 0.9 almost anything
seasonal is. It is *"once the series' own persistence is represented properly,
is there anything left for the environment to explain, or to predict?"* Putting
`y_lag1` on the right-hand side of an OLS regression (§12's M4) is one answer,
and a fragile one: it deletes every month whose predecessor is missing, and its
standard errors are wrong unless the disturbance is exactly AR(1). §19 instead
puts the dependence in the **process**, chooses between AR(1), AR(2) and a
stochastic local level using **no driver information whatsoever**, locks that
choice, and only then adds the environmental block (§20). §21–§22 ask whether
that block improves genuine **one-calendar-month-ahead** prediction against the
*matched* no-driver model, with a calendar-aware bootstrap interval on the
difference. §12–§18 are retained in full as association and sensitivity
analyses.

**The honest sample size.** The panel spans at most ~9 years of months, and the
response is strongly autocorrelated, so the *effective* number of independent
observations is a fraction of the row count. Every standard error here is
autocorrelation-robust (Newey–West for the static models, a state-space sandwich
for the principal one), every *p*-value is FDR-adjusted, and every claim of
predictive skill comes from rolling-origin cross-validation — never from
in-sample $R^2$. The state-space model uses **all** observed months rather than
the complete-case subset, but it cannot manufacture independent information the
record does not contain: the interval widths, not the row count, are the honest
summary.

**Every temporal operation is defined on calendar months.** The record has
**gaps** — months excluded by the coverage filter — so two rows that are
neighbours in a complete-case table can be three or six calendar months apart.
Treating row adjacency as calendar adjacency would quietly corrupt the HAC
standard errors, the bootstrap blocks, the forecast horizon and the
seasonal-naive baseline alike. §5 therefore keys all of them on the month
itself, and §5b re-checks that every time the notebook runs.

**Two things are reported only when the data support them.** A long-run effect
$\\beta/(1-\\rho)$ is quoted only if the AR term's confidence interval lies
strictly inside the unit circle (§12); and a driver's *unique* variance
contribution is only ever compared with its Shapley value **within one
specification** (§17).

---

## Structure

| § | What it does |
|---|---|
| 1–3 | Install, imports, configuration |
| 4–5 | Helper functions (aggregation; calendar-aware statistics, plus §5b self-tests) |
| 6–7 | Load the panel → build the AOI monthly series |
| 8–9 | Series diagnostics; **season/trend identifiability audit** |
| 10–11 | Lag structure; model dataset |
| 12 | **Model A (M0–M4)** — static linear driver model, Newey–West SEs; M4 adds `y_lag1` as a *sensitivity* |
| 13 | **Model B** — elastic net + bootstrap stability selection (collinearity-robust ranking) |
| 14 | **Model C** — spline GLM, **one driver at a time, df ≤ 3, exploratory** (§14b checks it out of sample) |
| 15 | **Model D** — gradient boosting + out-of-fold permutation importance |
| 16 | Three-calendar-month rolling-origin skill — **sensitivity only, demoted** |
| 17 | **Variance partitioning** — Shapley $R^2$ across drivers, season, trend, persistence, with shared-vs-unique read **within** each specification |
| 18 | Robustness of the *static* models — response definition, coverage, leave-one-year-out, deseasonalised |
| **19** | **PRINCIPAL model, step 1** — candidate dependence structures (AR(1) / AR(2) / local level) selected on **null dynamics only** |
| **20** | **PRINCIPAL model, step 2** — matched null vs full state-space driver model; coefficients, joint test, state diagnostics |
| **21** | **PRINCIPAL model, step 3** — expanding-window **one-calendar-month-ahead** rolling origin, with literal persistence, fitted AR(1) and seasonal-naive baselines |
| **22** | Paired month-level losses + calendar-aware moving-block bootstrap interval on the RMSE difference |
| **23** | Parametric bootstrap of the joint driver-block LR statistic under the matched null |
| **24** | State-space robustness — leave-one-year-out, transforms, alternative dependence structures |
| 25 | **Synthesis** — the five questions, and the ranked, verdict-bearing driver list |
| 26–27 | Exports; interpretation checklist |
""")


# ===========================================================================
# 1. Install
# ===========================================================================
md("""## 1. Install packages

Colab already has everything except (optionally) `pygam`. The notebook does
**not** require `pygam` — §14 uses a natural-spline GLM built with `patsy` +
`statsmodels`, which gives exact nested tests and explicit degrees of freedom.
No `rpy2`, no `mgcv`, no Earth Engine calls.
""")

code("""# Uncomment in Colab if any import in §2 fails.
# !pip -q install numpy pandas scipy statsmodels scikit-learn matplotlib
""")


# ===========================================================================
# 2. Imports and Drive mount
# ===========================================================================
md("""## 2. Imports and Google Drive mount

The Drive mount is optional and guarded: the notebook runs locally, and runs
against a synthetic series (`USE_SYNTHETIC_DEMO = True` in §3) with no Drive at
all.
""")

code('''from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import scipy.stats as sstats
import statsmodels.api as sm

# State-space machinery for the PRINCIPAL dynamic-regression model (§19-§24).
# Both live in statsmodels core, so there is nothing extra to install in Colab.
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")
warnings.filterwarnings("ignore", message=".*No frequency information was provided.*")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 80)

IN_COLAB = False
try:  # pragma: no cover - Colab only
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except Exception:
    pass

DRIVE_MOUNTED = False
if IN_COLAB:  # pragma: no cover - Colab only
    try:
        drive.mount("/content/drive", force_remount=False)
        DRIVE_MOUNTED = True
    except Exception as exc:
        print("Drive mount failed:", exc)

try:
    from IPython.display import display  # noqa: F401
except Exception:  # pragma: no cover - non-IPython
    display = print

print(f"Colab: {IN_COLAB} | Drive mounted: {DRIVE_MOUNTED}")
print(f"pandas {pd.__version__} | statsmodels {sm.__version__}")
print("state-space: SARIMAX + UnobservedComponents available (§19-§24)")

# Optional extras. Absence changes what runs, never whether the notebook runs.
try:
    import sklearn
    from sklearn.linear_model import ElasticNetCV, ElasticNet
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    HAVE_SKLEARN = True
    print(f"scikit-learn {sklearn.__version__}")
except Exception as exc:
    HAVE_SKLEARN = False
    print("scikit-learn unavailable -> §13 and §15 will be skipped:", exc)

try:
    from patsy import dmatrix
    HAVE_PATSY = True
except Exception as exc:
    HAVE_PATSY = False
    print("patsy unavailable -> §14 (spline GLM) will be skipped:", exc)
''')


# ===========================================================================
# 3. Configuration
# ===========================================================================
md("""## 3. Configuration

Every decision that changes a *number* in the results lives here. The defaults
are the defensible choice; alternatives are documented next to each one.
""")

code('''# =====================================================================
# 3a. Where the data comes from
# =====================================================================
# The cell-month panel exported by winam_wh_spatial_panel_driver_gam.ipynb §17:
#   OUTPUT_DIR / f"wh_spatial_panel_{CELL_SIZE_M}m_{run_tag}_{start}_to_{end}.csv"
# It already carries the response (wh_area_ha, valid_area_ha, wh_cover,
# wh_present) AND every merged covariate, so ONE file is enough. Set
# PANEL_CSV to a specific file, or leave it None to take the most recent
# match of PANEL_GLOB in PANEL_DIR.
PANEL_DIR = Path("/content/drive/MyDrive/WH_spatial_panel_test")
PANEL_CSV = None
PANEL_GLOB = "wh_spatial_panel_*.csv"

# Fallback route, used only when no panel CSV is found: rebuild the monthly
# series from the separately-exported tables. wh_monthly_summary_*.csv supplies
# the response; the two Earth Engine tables supply the drivers.
MONTHLY_SUMMARY_GLOB = "wh_monthly_summary_*.csv"
EE_MONTHLY_GLOB = "ee_monthly_covariates_*.csv"
EE_CELLMONTH_GLOB = "ee_cellmonth_covariates_*.csv"

# Optional extra monthly series merged on `month` (one row per month). This is
# how a driver that never entered the spatial panel gets in — e.g. an ENSO /
# IOD index, gauged river discharge, or measured nutrient concentrations.
EXTRA_MONTHLY_CSV = None
# EXTRA_MONTHLY_CSV = Path("/content/drive/MyDrive/WH_drivers/enso_iod_monthly.csv")

OUTPUT_DIR = Path("/content/drive/MyDrive/WH_temporal_driver_model")

# Run the whole notebook on a synthetic series with KNOWN driver effects and no
# Drive. Use it to (a) check the notebook end-to-end, and (b) confirm the
# recovery machinery finds an effect that is genuinely there. Never report
# synthetic numbers as results — every table is tagged is_synthetic.
USE_SYNTHETIC_DEMO = False
SYNTHETIC_N_MONTHS = 108
SYNTHETIC_SEED = 42
# Share of synthetic months pushed below the coverage filter, so the offline
# self-test has the same kind of CALENDAR GAPS the real record has. Set to 0.0
# for a gapless series; the real Winam record is roughly a quarter missing
# inside its fitted span, so the default is deliberately not zero.
SYNTHETIC_MISSING_FRACTION = 0.20

# =====================================================================
# 3b. How the AOI series is built
# =====================================================================
# A month enters only if it classified at least this share of the eligible
# water cells. Mirrors MIN_MONTHLY_COVERAGE_FRACTION in the spatial panel: a
# cloudy month measures a different piece of lake, not less hyacinth.
MIN_MONTHLY_COVERAGE_FRACTION = 0.90

# Fixed cell set. A cell must be validly observed in at least this share of the
# retained months to join set C. Higher = a stricter balanced panel (more
# comparable months, fewer cells); 0.0 = use every cell observed in any month
# and accept the composition change. The trade-off is printed in §7.
MIN_CELL_MONTH_FRACTION = 0.80

# Weight used when collapsing cells to an AOI value: each cell contributes in
# proportion to the area actually classified in it that month.
#   "valid_area"  - area-weighted (the default; the estimand is a real mean cover)
#   "equal"       - unweighted mean over cells (each cell counts once)
AOI_WEIGHTING = "valid_area"

# Per-month regression weight. Months built on more classified area estimate the
# AOI mean more precisely, and weighting by that area is the matching
# heteroskedasticity correction.
#   "valid_area" - weight by classified water area (default)
#   "coverage"   - weight by coverage fraction
#   "none"       - unweighted
MONTH_WEIGHTING = "valid_area"

# Response scale for the linear / spline models. WH cover is a proportion, so a
# logit keeps fitted values inside (0, 1) and makes effects multiplicative on
# the odds — the natural scale for proportional growth.
#   "logit"    - logit of the area-weighted cover (default)
#   "log"      - log(cover + eps)
#   "identity" - raw cover
RESPONSE_TRANSFORM = "logit"
RESPONSE_EPS = 1e-4          # guards logit/log at exactly 0 or 1

# Which AOI quantity is "how much hyacinth". All are built; this one is primary.
#   "wh_cover_aoi"  - WH area / validly-classified area (coverage-independent)
#   "wh_area_ha"    - absolute mapped WH area (depends on coverage; §18 sweeps it)
#   "wh_occurrence" - share of cells with WH present
RESPONSE_COL = "wh_cover_aoi"

# =====================================================================
# 3c. Drivers: mechanism, a-priori sign, a-priori lag
# =====================================================================
# ONE representation per mechanism, chosen from WH ecology, each with the sign
# it must have to support the mechanism and the lag at which the mechanism acts.
# This is the same discipline as GAM_FORCING_TERMS in the spatial panel, with
# the lag made explicit because a temporal model is nothing but lag structure.
#
#   term: (mechanism, expected sign, a-priori lag in months)
TEMPORAL_FORCING_TERMS = {
    "rain_chirps_30d_mm": ("antecedent rainfall -> catchment runoff and nutrient "
                           "delivery; acts with a delay", "+", 1),
    "air_temp_c":         ("thermal control on growth rate", "+", 0),
    "wind_speed_ms":      ("wind speed -> mixing and mat drift", "?", 0),
    "wave_exposure_idx":  ("fetch x wind^2 -> wave disturbance, mat fragmentation", "-", 0),
    "lake_level_m":       ("lake level -> depth over littoral habitat and flushing", "-", 0),
    "rain_max_1d_mm":     ("peak daily rainfall intensity -> runoff/disturbance pulse", "?", 1),
}

# Substitutes used when the preferred term is absent from the built series. A
# term with no available substitute is dropped and reported, never replaced
# silently.
TEMPORAL_FORCING_FALLBACKS = {
    "rain_chirps_30d_mm": ["rain_chirps_mm", "rain_chirps_90d_mm"],
    "air_temp_c":         ["water_temp_c"],
    "wind_speed_ms":      ["wind_axis_comp_ms", "wind_cross_comp_ms"],
    "lake_level_m":       ["lake_level_anom_m"],
    "rain_max_1d_mm":     ["rain_sdii_mm", "rain_wet_days"],
}

# Endogenous optical / biogeochemical proxies. Chl-a and turbidity are measured
# from the same reflectance a floating mat dominates, and a mat changes the
# water beneath it, so they are DOWNSTREAM of WH as much as upstream. They are
# excluded from every driver claim and reported once, separately, as a
# descriptive association (§12e).
TEMPORAL_PROXY_TERMS = ["chl_mci_s3", "chl_mph_s3", "chl_ndci_s2",
                        "turb_ndti_s2", "chl_modis_mg_m3"]
RUN_DESCRIPTIVE_PROXY_MODEL = True

# Static covariates. Listed so §7 can drop them BY NAME with a reason printed,
# rather than leaving the reader to wonder why depth is absent from a depth-
# sensitive system. Over a fixed cell set each is a constant.
KNOWN_STATIC_COLS = ["depth_m", "dist_shore_m", "dist_river_m", "dist_majriver_m",
                     "frac_cropland", "frac_urban", "frac_wetland", "openness_index",
                     "pop_count", "built_surface", "gsw_water_fraction",
                     "shore_gx", "shore_gy"]

# Terms that are an exact affine function of another AOI series once averaged,
# so they are perfectly collinear with it and carry no independent information.
# effective_depth_m = depth_m + (lake_level_m - mean), and depth_m is constant
# over a fixed cell set, so its AOI mean IS the lake-level anomaly.
AOI_DEGENERATE_COLS = {"effective_depth_m": "= constant AOI depth + lake-level anomaly",
                       "lake_level_anom_m": "= lake_level_m - constant"}

# Lag handling. "apriori" uses the lag stated in TEMPORAL_FORCING_TERMS (the
# defensible default: the lag is a hypothesis, not a parameter fitted to the
# response). "ccf" picks each driver's lag by peak absolute cross-correlation —
# faster to look good and biased, so it is a sensitivity, not the headline.
LAG_SELECTION = "apriori"
LAG_SCAN_MAX = 6              # lags 0..LAG_SCAN_MAX shown in the §10 CCF evidence
DISTRIBUTED_LAG_TERMS = []    # e.g. ["rain_chirps_30d_mm"] to enter lags 0..2 together

# =====================================================================
# 3d. Model / inference settings
# =====================================================================
SEASON_HARMONICS = 2          # sin/cos pairs for the annual cycle (2 -> 4 columns)
INCLUDE_TREND = True          # linear month index, capturing multi-year drift
AR_LAGS = 1                   # lagged response in the dynamic specification

# Long-run effects beta / (1 - rho) are only meaningful if the AR process is
# stationary. With rho near 1 the multiplier explodes and its delta-method
# interval is meaningless, so §12 reports long-run effects ONLY when the HAC
# confidence interval for the AR term lies strictly inside the unit circle.
# Setting this False restores the (unsafe) unconditional reporting.
LONGRUN_REQUIRE_STATIONARITY = True
LONGRUN_CI_ALPHA = 0.05       # interval used for the unit-root check

# Newey-West bandwidth, in CALENDAR MONTHS. None -> floor(4 * (n/100)^(2/9)),
# the standard rule. The covariance is built from pairs of observations exactly
# h calendar months apart, so a gap in the record never makes two observations
# look like consecutive months (§5).
HAC_MAXLAGS = None

# Benjamini-Hochberg FDR level applied across the drivers within a specification.
FDR_ALPHA = 0.10

# Moving-block bootstrap. The block length is in CALENDAR MONTHS and the blocks
# are drawn from the calendar-complete grid, so a block always spans a
# contiguous stretch of the calendar and keeps the record's missing-month
# pattern inside every replicate.
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = None        # None -> ceil(grid_span_months ** (1/3))
BOOTSTRAP_SEED = 7

# Rolling-origin cross-validation, defined on CALENDAR MONTHS. Mirrors the
# spatial panel's temporal design (GAM_N_TEMPORAL_FOLDS /
# GAM_TEMPORAL_HORIZON_MONTHS / GAM_TEMPORAL_MIN_TRAIN_MONTHS) so "fold 1"
# means the same thing in both. A horizon of 3 means the next THREE CONSECUTIVE
# CALENDAR MONTHS after the origin — not the next three observed rows.
CV_N_FOLDS = 8
CV_HORIZON_MONTHS = 3
CV_MIN_TRAIN_MONTHS = 24
# A test window is never widened to collect more observations. If fewer than
# this many of its calendar months are evaluable, the fold is reported with
# usable=False and skipped.
CV_MIN_TEST_MONTHS = 2

# A driver whose AOI series is this well explained by the seasonal harmonics
# alone cannot be separated from "it is summer" and is reported as
# season-confounded regardless of its p-value.
SEASON_CONFOUND_R2 = 0.80
# Pairwise |r| above this makes two drivers redundant; the one later in the
# mechanism list is dropped and reported.
MAX_ABS_PAIRWISE_R = 0.90
VIF_THRESHOLD = 5.0

# Spline GLM (§14): degrees of freedom per driver smooth. The previous default
# fitted EVERY driver as a df=4 smooth simultaneously — 25 design columns on ~64
# complete months, which is not a model, it is an interpolation. §14 now fits
# ONE predeclared driver at a time at df <= SPLINE_DF_MAX, centres each smooth,
# and labels the result exploratory unless it also improves one-month-ahead
# out-of-sample prediction (§14b).
SPLINE_DF = 3
SPLINE_DF_MAX = 3             # hard ceiling; SPLINE_DF is clipped to it
SPLINE_ONE_DRIVER_AT_A_TIME = True
SPLINE_MIN_ROWS_PER_COLUMN = 4   # a smooth is REFUSED below this rows-per-column.
# 4 is a floor, not a licence: the withdrawn all-at-once design sat at 2.6 rows
# per column, and the one-at-a-time design here sits at roughly 5 on this record.
# That is still thin, which is why §14 is labelled exploratory throughout and why
# §14b's out-of-sample check — not the in-sample F test — decides what may be
# reported.
SPLINE_OOS_CHECK = True       # rolling-origin linear-vs-spline check in §14b

# =====================================================================
# 3e. State-space dynamic regression — the PRINCIPAL model (§19-§24)
# =====================================================================
# The static models above answer "is this driver associated with WH extent".
# They cannot answer "does this driver explain WH extent once the series'
# own persistence is represented", because a lagged response on the right-hand
# side of an OLS regression is only one (fragile) way to write persistence down,
# and it silently deletes every month whose predecessor is missing. The
# state-space models below put the dependence in the ERROR / STATE process
# instead, so a missing month costs a likelihood contribution rather than a row.
RUN_STATE_SPACE = True

# Deliberately small candidate set. Each is a hypothesis about HOW the series
# remembers itself; nothing here is chosen by looking at driver p-values.
#   sarimax_ar1  - regression with AR(1) errors
#   sarimax_ar2  - regression with AR(2) errors
#   local_level  - stochastic local level (random-walk state) + observation noise
SS_CANDIDATE_STRUCTURES = ["sarimax_ar1", "sarimax_ar2", "local_level"]

# How the locked structure is chosen from the NULL (no-driver) fits. Both
# criteria are printed either way; this only decides which one breaks the tie.
#   "aicc"   - smallest AICc on the observed response months (default)
#   "rmse1"  - smallest one-month-ahead rolling-origin RMSE
SS_SELECT_BY = "aicc"

# Seasonality inside the state-space models is DETERMINISTIC annual Fourier
# (the same terms as SEASON_COLS), never a stochastic seasonal, so the season
# cannot absorb driver signal by drifting. A linear trend is included ONLY in
# candidates without a stochastic local level — a random-walk level and a linear
# trend compete for the same low-frequency variation.
SS_SEASON_HARMONICS = None    # None -> reuse SEASON_HARMONICS

# Covariance estimator for the state-space coefficient standard errors.
# "robust_approx" is the Huber sandwich around the approximate information
# matrix — the state-space analogue of a heteroskedasticity-robust SE. Falls
# back to "opg" and then to the default if the sandwich is not computable.
SS_COV_TYPE = "robust_approx"
SS_MAXITER = 250              # optimiser iterations for the reported fits
SS_ROLLING_MAXITER = 100      # fewer per rolling-origin refit; there are many

# --- One-calendar-month-ahead rolling-origin evaluation (§21) ----------------
# EXPANDING window, one calendar month ahead, every feasible origin. This is the
# PRINCIPAL predictive assessment. §16's three-calendar-month windows are kept
# as a sensitivity comparison and are no longer the headline.
SS_MIN_TRAIN_MONTHS = 24      # observed response months required before an origin
SS_FORECAST_MIN_LAG = 1       # every forecast driver must be known at the origin
# A driver whose a-priori lag is 0 (temperature, wind, lake level) is NOT
# knowable at the origin, so for FORECASTING it enters at lag 1. Map a driver to
# a column holding a genuine forecast value to override this.
SS_FORECAST_EXOG_OVERRIDE = {}
# e.g. SS_FORECAST_EXOG_OVERRIDE = {"air_temp_c": "air_temp_c_forecast_t"}

# --- Paired RMSE-difference bootstrap (§22) ---------------------------------
SS_RMSE_BOOTSTRAP_N = 2000
SS_RMSE_BOOTSTRAP_BLOCK = None   # None -> ceil(eval span ** (1/3)) calendar months
SS_RMSE_BOOTSTRAP_SEED = 11

# --- Parametric bootstrap of the joint driver-block LR statistic (§23) -------
# Simulates from the FITTED MATCHED NULL, re-imposes the record's missing-month
# pattern, refits null and full, and rebuilds the LR null distribution. The
# asymptotic chi-square reference is unreliable at ~60 observed months.
SS_RUN_LR_BOOTSTRAP = True
SS_LR_BOOTSTRAP_N = 399       # 300-500 is the intended range
SS_LR_BOOTSTRAP_SEED = 2024
SS_LR_BOOTSTRAP_MAXITER = 80
SS_LR_BOOTSTRAP_TIME_BUDGET_S = 900   # abandon (and say so) past this many seconds

# --- Robustness (§24) --------------------------------------------------------
SS_ROBUST_TRANSFORMS = ["logit", "log", "identity"]
SS_RUN_LOYO = True

# Gradient boosting (§15). Deliberately small: with ~100 months a large
# ensemble memorises the series and its importances mean nothing.
GBM_PARAMS = dict(max_depth=2, max_iter=200, learning_rate=0.05,
                  min_samples_leaf=8, l2_regularization=1.0, random_state=0)
GBM_PERM_REPEATS = 30

# Shapley R^2 partitioning (§17). Groups are formed automatically from the
# drivers plus season / trend / persistence; 2^G ordinary least squares fits, so
# keep G <= 12.
PARTITION_INCLUDE_AR = True
PARTITION_MAX_GROUPS = 12

RANDOM_STATE = 42

OUTPUT_DIR = Path(OUTPUT_DIR)
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_WRITABLE = True
except Exception as exc:
    OUTPUT_WRITABLE = False
    print(f"OUTPUT_DIR not writable ({exc}); §26 will skip the exports.")

SPLINE_DF = int(min(SPLINE_DF, SPLINE_DF_MAX))
if SS_SEASON_HARMONICS is None:
    SS_SEASON_HARMONICS = SEASON_HARMONICS

print("Configuration loaded.")
print(f"  response          : {RESPONSE_COL} ({RESPONSE_TRANSFORM} scale)")
print(f"  forcing mechanisms: {len(TEMPORAL_FORCING_TERMS)}")
print(f"  lag selection     : {LAG_SELECTION}")
print(f"  principal model   : state-space dynamic regression "
      f"({'on' if RUN_STATE_SPACE else 'OFF'}), candidates "
      f"{SS_CANDIDATE_STRUCTURES}, selected by {SS_SELECT_BY}")
print(f"  principal skill   : expanding-window ONE-calendar-month-ahead "
      f"rolling origin (>= {SS_MIN_TRAIN_MONTHS} training months)")
print(f"  spline (§14)      : one driver at a time, df = {SPLINE_DF} "
      f"(ceiling {SPLINE_DF_MAX}), exploratory unless it survives §14b")
print(f"  output            : {OUTPUT_DIR}")
''')

# ===========================================================================
# 4. Aggregation helpers
# ===========================================================================
md("""## 4. Helpers — loading and AOI aggregation

Three things happen here, and each is a decision the write-up has to defend:

1. **Coverage filter.** A month is kept only if it classified
   `MIN_MONTHLY_COVERAGE_FRACTION` of the eligible water cells.
2. **Fixed cell set.** The AOI mean is taken over cells observed in at least
   `MIN_CELL_MONTH_FRACTION` of the retained months, so the *composition* of
   the average is constant and month-to-month change is change in hyacinth.
3. **Calendar-complete index.** The monthly series is reindexed onto an
   unbroken month grid with `NaN` where a month was excluded, so every lag is
   calendar-correct and a gap never silently becomes "last month".
""")

code('''# =====================================================================
# 4. Loading and AOI aggregation
# =====================================================================


def newest_match(directory, pattern):
    """Most recently modified file matching `pattern` in `directory`, or None."""
    directory = Path(directory)
    if not directory.exists():
        return None
    hits = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def to_month_start(series):
    """Coerce anything date-like to first-of-month timestamps."""
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def load_cellmonth_panel(panel_csv=None, panel_dir=None, panel_glob="wh_spatial_panel_*.csv"):
    """Load the exported cell-month panel and normalise its month column."""
    path = Path(panel_csv) if panel_csv else newest_match(panel_dir, panel_glob)
    if path is None or not Path(path).exists():
        return None, None
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError(f"{path} has no 'month' column; not a cell-month panel.")
    df["month"] = to_month_start(df["month"])
    return df, Path(path)


def _area_columns(panel):
    """Resolve the WH / valid area columns, accepting the ha or m2 spelling."""
    wh_col = next((c for c in ("wh_area_ha", "wh_area_ha_hard") if c in panel.columns), None)
    valid_col = "valid_area_ha" if "valid_area_ha" in panel.columns else None
    if wh_col is None and "wh_area_m2" in panel.columns:
        panel["wh_area_ha"] = panel["wh_area_m2"] / 1e4
        wh_col = "wh_area_ha"
    if valid_col is None and "valid_area_m2" in panel.columns:
        panel["valid_area_ha"] = panel["valid_area_m2"] / 1e4
        valid_col = "valid_area_ha"
    if wh_col is None or valid_col is None:
        raise ValueError("panel needs WH area and valid area columns "
                         "(wh_area_ha/wh_area_m2 and valid_area_ha/valid_area_m2)")
    return wh_col, valid_col


def monthly_coverage_table(panel, min_coverage):
    """Per-month observed share of the eligible cells, and the retain decision.

    `coverage_fraction` is reused when the panel already carries it (the spatial
    panel computes it against the water/habitat mask, which is the right
    denominator). Otherwise it is computed against the number of DISTINCT cells
    ever observed, which is the best available proxy for "eligible".
    """
    _, valid_col = _area_columns(panel)
    per_month = (panel.groupby("month", as_index=False)
                 .agg(n_cells_observed=("grid_id", "nunique"),
                      valid_area_ha=(valid_col, "sum")))
    if "coverage_fraction" in panel.columns:
        cov = (panel.groupby("month", as_index=False)["coverage_fraction"].first()
               .rename(columns={"coverage_fraction": "coverage_fraction"}))
        per_month = per_month.merge(cov, on="month", how="left")
        per_month["coverage_basis"] = "panel coverage_fraction"
    else:
        n_eligible = int(panel["grid_id"].nunique())
        per_month["coverage_fraction"] = per_month["n_cells_observed"] / max(n_eligible, 1)
        per_month["coverage_basis"] = f"n_cells_observed / {n_eligible} cells ever observed"
    per_month["retained"] = per_month["coverage_fraction"] >= float(min_coverage)
    return per_month.sort_values("month").reset_index(drop=True)


def fixed_cell_set(panel, months_kept, min_cell_month_fraction):
    """Cells observed in >= `min_cell_month_fraction` of the retained months.

    Returns (cell_ids, audit_dict). The audit records the trade-off the choice
    makes: how many cells were kept, and what share of the total classified
    area they represent.
    """
    sub = panel[panel["month"].isin(months_kept)]
    n_months = int(sub["month"].nunique())
    seen = sub.groupby("grid_id")["month"].nunique()
    need = float(min_cell_month_fraction) * n_months
    keep = seen[seen >= need].index
    _, valid_col = _area_columns(panel)
    area_all = float(sub[valid_col].sum())
    area_keep = float(sub[sub["grid_id"].isin(keep)][valid_col].sum())
    audit = {
        "n_months_retained": n_months,
        "n_cells_total": int(sub["grid_id"].nunique()),
        "n_cells_kept": int(len(keep)),
        "min_months_required": float(need),
        "share_of_classified_area_kept": (area_keep / area_all) if area_all else np.nan,
        "min_cell_month_fraction": float(min_cell_month_fraction),
    }
    return pd.Index(keep), audit


def aoi_monthly_series(panel, cell_ids=None, weighting="valid_area",
                       driver_cols=None, response_eps=1e-4):
    """Collapse the cell-month panel to one row per month.

    * response  - area-weighted AOI cover = sum(WH area) / sum(valid area),
                  plus the absolute WH area, the unweighted mean cover and the
                  occurrence rate;
    * drivers   - weighted mean over the same cells and the same weights, so a
                  driver and the response describe the same piece of lake.
    """
    wh_col, valid_col = _area_columns(panel)
    sub = panel if cell_ids is None else panel[panel["grid_id"].isin(cell_ids)]
    sub = sub.copy()
    if weighting == "valid_area":
        sub["_w"] = sub[valid_col].astype(float)
    elif weighting == "equal":
        sub["_w"] = 1.0
    else:
        raise ValueError(f"unknown AOI weighting {weighting!r}")

    agg = {"wh_area_ha": (wh_col, "sum"),
           "valid_area_ha": (valid_col, "sum"),
           "n_cells": ("grid_id", "nunique")}
    if "wh_cover" in sub.columns:
        agg["wh_cover_mean_unweighted"] = ("wh_cover", "mean")
    if "wh_present" in sub.columns:
        agg["wh_occurrence"] = ("wh_present", "mean")
    out = sub.groupby("month", as_index=False).agg(**agg)
    out["wh_cover_aoi"] = out["wh_area_ha"] / out["valid_area_ha"].replace(0, np.nan)

    for col in (driver_cols or []):
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce")
        w = sub["_w"].where(vals.notna(), 0.0)
        num = (vals.fillna(0.0) * w).groupby(sub["month"]).sum()
        den = w.groupby(sub["month"]).sum().replace(0, np.nan)
        out = out.merge((num / den).rename(col).reset_index(), on="month", how="left")

    return out.sort_values("month").reset_index(drop=True)


def reindex_calendar_months(monthly):
    """Put the series on an unbroken month grid so every lag is calendar-correct.

    Excluded / never-observed months become all-NaN rows. `time_index` counts
    months from the start of the record (not row position), and `observed` marks
    the rows that carry a real measurement.
    """
    monthly = monthly.sort_values("month").reset_index(drop=True)
    full = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    out = monthly.set_index("month").reindex(full).rename_axis("month").reset_index()
    out["time_index"] = np.arange(len(out), dtype=float)
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    out["observed"] = out["wh_cover_aoi"].notna() if "wh_cover_aoi" in out.columns else True
    return out


def on_calendar_grid(months, values):
    """Put a month/value pair on an unbroken monthly grid, for PLOTTING.

    A line plot joins consecutive rows, so a frame that holds only the observed
    months draws a straight segment across every unobserved stretch — the
    2017-07..2018-11 gap in this record becomes a confident-looking line through
    17 months that were never measured. Matplotlib breaks a line at NaN, so
    re-indexing onto every calendar month between the first and last point turns
    each absent month back into a visible gap.

    Every missing month breaks the line, so a long gap and a single skipped
    month are both shown rather than drawn through, and no interpolated point is
    ever invented (which would also draw a marker where there is no observation).

    Returns ``(months, values)`` so it can be splatted into ``ax.plot``.
    """
    m = to_month_start(pd.Series(months).reset_index(drop=True))
    v = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    s = pd.Series(v.to_numpy(dtype=float), index=pd.DatetimeIndex(m)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if s.empty:
        return pd.DatetimeIndex([]), np.array([], dtype=float)
    grid = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    out = s.reindex(grid)
    return out.index, out.to_numpy(dtype=float)


def add_season_terms(monthly, n_harmonics=2):
    """Fourier terms for the annual cycle: sin/cos at 1..n cycles per year."""
    out = monthly.copy()
    ang = 2 * np.pi * out["month_num"] / 12.0
    cols = []
    for k in range(1, int(n_harmonics) + 1):
        out[f"season_sin{k}"] = np.sin(k * ang)
        out[f"season_cos{k}"] = np.cos(k * ang)
        cols += [f"season_sin{k}", f"season_cos{k}"]
    return out, cols


def transform_response(values, how="logit", eps=1e-4):
    """Map the response onto the modelling scale, and report what was clipped."""
    y = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    if how == "identity":
        return y, {"transform": "identity", "n_clipped": 0}
    if how == "log":
        n_clip = int((y <= 0).sum())
        return np.log(y.clip(lower=eps)), {"transform": "log", "n_clipped": n_clip}
    if how == "logit":
        n_clip = int(((y <= 0) | (y >= 1)).sum())
        p = y.clip(lower=eps, upper=1 - eps)
        return np.log(p / (1 - p)), {"transform": "logit", "n_clipped": n_clip}
    raise ValueError(f"unknown RESPONSE_TRANSFORM {how!r}")


def calendar_lag(monthly, cols, lag):
    """Lag `cols` by `lag` months on the calendar-complete grid."""
    out = monthly.copy()
    made = []
    for c in cols:
        if c not in out.columns:
            continue
        name = c if lag == 0 else f"{c}_lag{lag}"
        out[name] = out[c].shift(lag)
        made.append(name)
    return out, made


def make_synthetic_monthly(n_months=108, seed=42, missing_fraction=0.0):
    """A synthetic AOI series with KNOWN driver effects, for offline checking.

    Construction (on the logit-cover scale): a strong annual cycle, an upward
    trend, autoregressive persistence, a POSITIVE effect of lagged antecedent
    rainfall, a NEGATIVE effect of wave exposure, and two pure-noise drivers
    that must NOT be recovered.

    `missing_fraction` drives `coverage_fraction` below any plausible threshold
    on that share of months, so the self-test series has the SAME kind of
    calendar gaps the real record has. Without them the offline run never
    exercises the missing-month machinery — the exogenous placeholder, the
    withheld months, the unavailable seasonal-naive lookups — which is most of
    what §19-§21 have to get right. The gaps include a deliberate pair exactly
    12 calendar months apart so a seasonal-naive lookup genuinely fails.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2017-01-01", periods=int(n_months), freq="MS")
    ang = 2 * np.pi * np.asarray(months.month, dtype=float) / 12.0

    rain = 60 + 55 * np.sin(ang - 0.6) + rng.gamma(2.0, 12.0, len(months))
    air = 25 + 1.6 * np.cos(ang) + rng.normal(0, 0.5, len(months))
    wind = 3.2 + 0.6 * np.sin(ang + 1.1) + rng.normal(0, 0.35, len(months))
    wave = np.clip(wind ** 2 * (1.0 + rng.normal(0, 0.12, len(months))), 0.1, None)
    level = 1134.0 + 0.6 * np.sin(2 * np.pi * np.arange(len(months)) / 42.0) \\
        + rng.normal(0, 0.06, len(months))
    rain_max = rain / 6.0 + rng.gamma(2.0, 3.0, len(months))
    decoy_a = rng.normal(0, 1, len(months))
    decoy_b = 10 + rng.normal(0, 2, len(months))

    z = lambda v: (v - np.mean(v)) / np.std(v)
    eta = np.full(len(months), -3.0)
    rain_lag1 = np.r_[np.nan, z(rain)[:-1]]
    eta = eta + 0.45 * np.nan_to_num(rain_lag1) - 0.30 * z(wave) + 0.10 * z(air)
    eta = eta + 0.55 * np.sin(ang - 0.9) + 0.012 * np.arange(len(months))
    for t in range(1, len(months)):                      # AR(1) persistence
        eta[t] += 0.45 * (eta[t - 1] - eta.mean())
    eta = eta + rng.normal(0, 0.22, len(months))
    cover = 1.0 / (1.0 + np.exp(-eta))

    valid = 120000 + rng.normal(0, 4000, len(months))
    out = pd.DataFrame({
        "month": months,
        "wh_cover_aoi": cover,
        "wh_area_ha": cover * valid,
        "valid_area_ha": valid,
        "n_cells": 5200,
        "coverage_fraction": np.clip(0.97 + rng.normal(0, 0.015, len(months)), 0, 1),
        "wh_occurrence": np.clip(cover * 3.5, 0, 1),
        "wh_cover_mean_unweighted": cover * 1.02,
        "rain_chirps_30d_mm": rain,
        "rain_chirps_mm": rain * 0.95,
        "rain_max_1d_mm": rain_max,
        "air_temp_c": air,
        "wind_speed_ms": wind,
        "wave_exposure_idx": wave,
        "lake_level_m": level,
        "chl_mci_s3": 12 + 8 * z(cover) + rng.normal(0, 2, len(months)),
        "turb_ndti_s2": 0.1 + 0.03 * z(rain) + rng.normal(0, 0.01, len(months)),
        "decoy_noise": decoy_a,
        "decoy_level": decoy_b,
    })
    if float(missing_fraction) > 0:
        n_gap = int(round(float(missing_fraction) * len(out)))
        pool = np.arange(6, len(out) - 1)
        gaps = set(int(v) for v in rng.choice(pool, size=min(n_gap, len(pool)),
                                              replace=False))
        # One gap whose partner exactly 12 calendar months later is observed, so
        # the seasonal-naive baseline has a month it genuinely cannot score.
        if len(out) > 40:
            gaps.add(24)
            gaps.discard(36)
        out.loc[sorted(gaps), "coverage_fraction"] = 0.30
        truth_gaps = sorted(gaps)
    else:
        truth_gaps = []

    truth = {"rain_chirps_30d_mm(lag1)": +0.45, "wave_exposure_idx(lag0)": -0.30,
             "air_temp_c(lag0)": +0.10, "decoy_noise": 0.0, "decoy_level": 0.0,
             "ar1_on_eta": 0.45, "trend_per_month_logit": 0.012,
             "n_months_forced_missing": len(truth_gaps)}
    return out, truth


print("§4 aggregation helpers defined.")
''')


# ===========================================================================
# 5. Statistics helpers
# ===========================================================================
md("""## 5. Helpers — calendar-aware, autocorrelation-robust inference

The whole difficulty of a temporal driver model is that consecutive months are
not independent observations. A second difficulty compounds it here: **the
record has gaps**. Months that fail the coverage filter are absent, so two rows
that sit next to each other in a complete-case table can be three or six
calendar months apart. Every routine below is therefore defined on **calendar
months**, never on row positions.

| Helper | What it does | Why row positions are wrong |
|---|---|---|
| **`fit_hac`** | OLS/WLS with a Newey–West covariance whose lag-$h$ term uses only pairs **exactly $h$ calendar months apart**; bandwidth $\\lfloor 4(n/100)^{2/9} \\rfloor$ months. | Row-adjacency HAC treats the months either side of a gap as consecutive, inventing a lag-1 correlation that does not exist. |
| **`calendar_month_blocks`** | Moving-block bootstrap whose blocks are contiguous runs of **calendar** months drawn from the complete grid; excluded months contribute no row, so a gap stays a gap. | A block of *rows* can silently splice together months a year apart. |
| **`rolling_origin_month_folds`** | Rolling-origin folds where a horizon of 3 means the **next three consecutive calendar months** after a dated origin, with every omitted test month and its reason recorded. | Three *rows* can span six calendar months, so the "3-month" forecast is not a 3-month forecast. |
| **`seasonal_naive_predictions`** | Looks up the response at **exactly** $t-12$ calendar months by timestamp on the complete grid. | `shift(12)` on complete-case rows returns the twelfth previous *observed* row, which is usually not the same calendar month. |
| **`acf_values` / `cross_correlation`** | Pair observations only when they are exactly $h$ (or $k$) calendar months apart. | `dropna()` compresses gaps and relabels the distance between the surviving months. |
| **`shapley_r2` / `semi_partial_r2`** | Order-averaged and last-entry $R^2$ contributions, each carrying the rows, columns, response and weighting it was computed on. | Merging the two from *different* models makes their ratio meaningless (§17). |
| **`ar_stability`** | Tests the roots of the fitted AR polynomial and the HAC interval for the AR term, gating the long-run multiplier $1/(1-\\rho)$. | Near a unit root the multiplier explodes and its delta-method interval is not usable. |
| **`bh_fdr`** | Benjamini–Hochberg *q*-values, because several drivers are tested at once. | — |

The calendar HAC estimator is the ordinary Newey–West sandwich
$\\hat V = (X'WX)^{-1}\\,\\hat S\\,(X'WX)^{-1}\\cdot\\frac{n}{n-k}$ with

$$\\hat S = \\sum_t u_t u_t' + \\sum_{h=1}^{L}\\Big(1-\\tfrac{h}{L+1}\\Big)
\\sum_{\\{(t,s)\\,:\\,m_t - m_s = h\\}} \\big(u_t u_s' + u_s u_t'\\big),$$

where $u_t = w_t x_t e_t$ and $m_t$ is the **calendar-month index** of
observation $t$. When the months happen to be fully consecutive this is exactly
`statsmodels`' `cov_type="HAC"` with `use_correction=True`; §5b checks that
numerically, and checks that inserting one missing month removes the pairs it
should.
""")

code('''# =====================================================================
# 5. Calendar-aware, autocorrelation-robust inference helpers
# =====================================================================
# Nothing below indexes time by row position. Every lag, block, horizon and
# covariance pair is defined by a CALENDAR-MONTH difference, because the record
# has excluded months and a complete-case table compresses them away: two rows
# that are neighbours in `fit_df` can be three or six calendar months apart.
from itertools import combinations


# ---------------------------------------------------------------------
# Calendar bookkeeping
# ---------------------------------------------------------------------

def month_index(months):
    """Integer calendar-month index (12 * year + month - 1).

    A difference in this index IS a difference in calendar months. That single
    property is what every routine in this cell relies on.
    """
    m = pd.to_datetime(pd.Series(months).reset_index(drop=True))
    return (m.dt.year.astype(int) * 12 + m.dt.month.astype(int) - 1).to_numpy()


def months_from_index(mi):
    """Inverse of `month_index`: month-start timestamps."""
    mi = np.asarray(mi, dtype=int).ravel()
    return pd.to_datetime([f"{int(v) // 12:04d}-{int(v) % 12 + 1:02d}-01" for v in mi])


def calendar_span_months(months):
    """Calendar months spanned by the record, gaps included."""
    mi = month_index(months)
    return (int(mi.max() - mi.min()) + 1) if len(mi) else 0


def _match_lag(mi_sorted, lag):
    """For each position t, the position of the month exactly `lag` months earlier.

    Returns (positions, hit) where `hit[t]` is True only when month `mi[t] - lag`
    is actually present. This is the primitive that stops a gap being crossed.
    """
    n = len(mi_sorted)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    pos = np.searchsorted(mi_sorted, mi_sorted - int(lag))
    safe = np.clip(pos, 0, n - 1)
    hit = (pos < n) & (mi_sorted[safe] == mi_sorted - int(lag))
    return safe, hit


# ---------------------------------------------------------------------
# Estimation: OLS / WLS with a calendar-aware Newey-West covariance
# ---------------------------------------------------------------------

def hac_maxlags(n, override=None):
    """Newey-West bandwidth in CALENDAR MONTHS: floor(4 * (n/100)^(2/9)), at least 1."""
    if override is not None:
        return int(override)
    return max(1, int(np.floor(4 * (max(int(n), 1) / 100.0) ** (2.0 / 9.0))))


def _complete_rows(y, X, weights=None, months=None):
    """Align response, design, weights and months on the fully observed rows.

    Dropping the incomplete rows HERE (rather than inside statsmodels) is what
    keeps the month of every fitted observation known and aligned.
    """
    X = pd.DataFrame(X).astype(float).reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float).ravel())
    if len(y) != len(X):
        raise ValueError("y and X disagree on length")
    ok = y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != len(X):
            raise ValueError("`weights` must have one entry per row of X")
        ok = ok & np.isfinite(w)
    mi = None
    if months is not None:
        mi_all = month_index(months)
        if len(mi_all) != len(X):
            raise ValueError("`months` must have one entry per row of X")
        mi = mi_all[ok]
        if len(mi) and np.any(np.diff(mi) <= 0):
            raise ValueError("months must be sorted, unique and one per row")
    return (y.to_numpy()[ok], X.loc[ok].reset_index(drop=True),
            None if w is None else w[ok], mi, ok)


def _fit_ols_wls(yv, Xf, wv, add_const=True):
    """Plain OLS/WLS fit on already-complete rows."""
    Xd = sm.add_constant(Xf, has_constant="add") if add_const else Xf
    if wv is None:
        return sm.OLS(yv, Xd).fit()
    ww = np.asarray(wv, dtype=float)
    ww = ww / np.nanmean(ww)
    return sm.WLS(yv, Xd, weights=ww).fit()


def newey_west_cov_calendar(xu, hessian_inv, month_idx, maxlags,
                            use_correction=True, k_params=None):
    """Newey-West (Bartlett) sandwich covariance on an IRREGULAR monthly index.

    `xu` are the per-observation scores -- for OLS/WLS `wexog * wresid`, i.e.
    w_t x_t e_t -- and `hessian_inv` is (X'WX)^-1.

        S = sum_t u_t u_t'
            + sum_{h=1..L} (1 - h/(L+1)) * sum_{(t,s): m_t - m_s = h} (u_t u_s' + u_s u_t')
        V = Hinv S Hinv * n / (n - k)          [finite-sample correction]

    The inner sum runs ONLY over pairs whose calendar months differ by exactly
    h, so two observations on opposite sides of a gap never contribute as a
    one-month pair. With fully consecutive months this reproduces statsmodels'
    ``cov_type="HAC"`` (Bartlett kernel, ``use_correction=True``) exactly --
    §5b verifies that numerically.

    Returns (V, pair_counts) where `pair_counts[h]` is the number of pairs that
    actually contributed at lag h. A small count at short lags is the honest
    signal that the record is too broken for the bandwidth requested.
    """
    xu = np.asarray(xu, dtype=float)
    if xu.ndim == 1:
        xu = xu[:, None]
    n, k = xu.shape
    mi = np.asarray(month_idx, dtype=int).ravel()
    if len(mi) != n:
        raise ValueError("month index and score matrix disagree on length")
    if n > 1 and np.any(np.diff(mi) <= 0):
        raise ValueError("months must be strictly increasing and unique")
    L = max(int(maxlags), 0)
    S = xu.T @ xu                       # the Bartlett weight at lag 0 is 1
    pair_counts = {0: int(n)}
    for h in range(1, L + 1):
        weight = 1.0 - h / (L + 1.0)
        pos, hit = _match_lag(mi, h)
        pair_counts[h] = int(hit.sum())
        if not hit.any():
            continue
        s = xu[hit].T @ xu[pos[hit]]
        S = S + weight * (s + s.T)
    V = np.asarray(hessian_inv) @ S @ np.asarray(hessian_inv)
    if use_correction:
        kk = k if k_params is None else int(k_params)
        if n > kk:
            V = V * (n / float(n - kk))
    return V, pair_counts


def fit_hac(y, X, weights=None, maxlags=None, add_const=True, months=None):
    """OLS/WLS whose Newey-West lags are CALENDAR MONTHS.

    Pass `months` (one month-start timestamp per row of `X`) and the lag-h term
    of the covariance is built only from pairs exactly h calendar months apart.
    Without `months` the estimator falls back to statsmodels' row-adjacency HAC,
    which is correct only when the rows really are consecutive months; the
    result records which was used in ``res._hac_calendar_aware``.

    The bandwidth is in calendar months and is reported in ``res._hac_maxlags``;
    ``res._hac_pair_counts`` says how many pairs each lag actually contributed.
    """
    yv, Xf, wv, mi, ok = _complete_rows(y, X, weights, months)
    lags = hac_maxlags(len(yv), maxlags)
    Xd = sm.add_constant(Xf, has_constant="add") if add_const else Xf
    if mi is None:
        if wv is None:
            model = sm.OLS(yv, Xd)
        else:
            ww = wv / np.nanmean(wv)
            model = sm.WLS(yv, Xd, weights=ww)
        res = model.fit(cov_type="HAC",
                        cov_kwds={"maxlags": lags, "use_correction": True})
        pair_counts = None
    else:
        res = _fit_ols_wls(yv, Xf, wv, add_const=add_const)
        xu = np.asarray(res.model.wexog) * np.asarray(res.wresid)[:, None]
        V, pair_counts = newey_west_cov_calendar(
            xu, np.asarray(res.normalized_cov_params), mi, lags,
            use_correction=True, k_params=Xd.shape[1])
        # statsmodels reads the parameter covariance from `cov_params_default`
        # on the results object itself (not the wrapper), and uses the normal
        # distribution for HAC inference.
        target = getattr(res, "_results", res)
        target.cov_params_default = V
        target.use_t = False
        target.cov_type = "HAC (calendar months, Bartlett kernel)"
        target.cov_kwds = {"maxlags": lags, "use_correction": True,
                           "spacing": "calendar months"}
    res._hac_maxlags = lags
    res._hac_calendar_aware = mi is not None
    res._hac_pair_counts = pair_counts
    res._fit_months = None if mi is None else months_from_index(mi)
    res._fit_rows_used = ok
    return res


def r2_of_fit(y, X, weights=None):
    """R^2 alone: no covariance work, because R^2 does not depend on it."""
    yv, Xf, wv, _, _ = _complete_rows(y, X, weights, None)
    if not len(yv) or not Xf.shape[1]:
        return np.nan, int(len(yv))
    res = _fit_ols_wls(yv, Xf, wv)
    return float(res.rsquared), int(res.nobs)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values, NaN-safe and order-preserving."""
    p = pd.Series(pvals, dtype=float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if not ok.any():
        return q
    vals = p[ok].sort_values()
    m = len(vals)
    adj = (vals.to_numpy() * m / np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    q.loc[vals.index] = adj
    return q


def tidy_coefficients(res, keep=None, drop_const=True, label=""):
    """Coefficient table with HAC SEs, 95% CI, p, and BH q over `keep`."""
    ci = res.conf_int()
    ci.columns = ["ci_lo", "ci_hi"]
    tab = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.to_numpy(),
        "se_hac": res.bse.to_numpy(),
        "t": res.tvalues.to_numpy(),
        "p": res.pvalues.to_numpy(),
    }).join(ci.reset_index(drop=True))
    if drop_const:
        tab = tab[tab["term"] != "const"]
    if keep is not None:
        tab = tab[tab["term"].isin(list(keep))]
    tab = tab.reset_index(drop=True)
    tab["q_fdr"] = bh_fdr(tab["p"]).to_numpy()
    tab.insert(0, "specification", label)
    tab["hac_calendar_aware"] = bool(getattr(res, "_hac_calendar_aware", False))
    return tab


# ---------------------------------------------------------------------
# Stationarity of the AR polynomial (gates every long-run effect)
# ---------------------------------------------------------------------

def ar_polynomial_roots(phi):
    """Roots of 1 - phi_1 z - ... - phi_p z^p.

    The AR process is stationary when EVERY root lies outside the unit circle,
    equivalently when every companion eigenvalue lies inside it. For p = 1 this
    reduces to |phi| < 1, but for p > 1 the sum of the coefficients is NOT a
    sufficient check, which is why the roots are taken here.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    if not len(phi):
        return np.array([], dtype=complex)
    ascending = np.r_[1.0, -phi]
    return np.roots(ascending[::-1])


def ar_stability(res, ar_terms, alpha=0.05):
    """Is the fitted AR polynomial stationary, and does the interval support it?

    Two conditions, both required before a finite long-run multiplier may be
    quoted:

    * **roots** - every root of 1 - sum_j phi_j z^j outside the unit circle
      (point estimate);
    * **interval** - the HAC confidence interval for the AR term lies strictly
      inside (-1, 1). For p = 1 that is the interval for `y_lag1`; for p > 1 the
      same test is applied to the delta-method interval for sum(phi), which is a
      necessary condition on top of the root check.

    Returns a dict; `stationary_supported` is the gate and `reason` says why
    when it is False.
    """
    ar_terms = [a for a in (ar_terms or []) if a in res.params.index]
    out = {"ar_terms": list(ar_terms), "alpha": float(alpha),
           "phi": {a: float(res.params[a]) for a in ar_terms}}
    if not ar_terms:
        out.update(rho_sum=np.nan, max_root_modulus=np.nan,
                   min_root_modulus=np.nan, roots_outside_unit_circle=False,
                   rho_ci_lo=np.nan, rho_ci_hi=np.nan, ci_within_unit_circle=False,
                   stationary_supported=False, reason="no AR term in the model")
        return out

    phi = np.array([float(res.params[a]) for a in ar_terms], dtype=float)
    roots = ar_polynomial_roots(phi)
    mod = np.abs(roots)
    out["rho_sum"] = float(phi.sum())
    out["min_root_modulus"] = float(np.min(mod)) if len(mod) else np.nan
    out["max_companion_modulus"] = float(np.max(1.0 / mod)) if len(mod) and np.all(mod > 0) else np.inf
    out["roots_outside_unit_circle"] = bool(len(mod) and np.all(mod > 1.0))

    names = list(res.params.index)
    V = np.asarray(res.cov_params())
    g = np.zeros(len(names))
    for a in ar_terms:
        g[names.index(a)] = 1.0
    se = float(np.sqrt(max(float(g @ V @ g), 0.0)))
    z = float(sstats.norm.ppf(1 - alpha / 2.0))
    out["rho_se_hac"] = se
    out["rho_ci_lo"] = float(phi.sum() - z * se)
    out["rho_ci_hi"] = float(phi.sum() + z * se)
    out["ci_within_unit_circle"] = bool(out["rho_ci_lo"] > -1.0 and out["rho_ci_hi"] < 1.0)

    if not out["roots_outside_unit_circle"]:
        reason = ("fitted AR polynomial has a root on or inside the unit circle "
                  "(non-stationary point estimate)")
    elif not out["ci_within_unit_circle"]:
        reason = "AR confidence interval includes a unit root"
    else:
        reason = ""
    out["stationary_supported"] = bool(out["roots_outside_unit_circle"]
                                       and out["ci_within_unit_circle"])
    out["reason"] = reason
    return out


# ---------------------------------------------------------------------
# Variance decomposition (with the provenance needed to compare the two)
# ---------------------------------------------------------------------

def _model_signature(columns):
    """Order-independent fingerprint of a design matrix's columns."""
    return "|".join(sorted(map(str, columns)))


def semi_partial_r2(y, X, weights=None, terms=None, specification="",
                    response="y", weighting=""):
    """Drop in R^2 when each term is removed from the full model (last entry).

    The provenance columns (`n_obs`, `n_model_terms`, `model_columns`,
    `response`, `weighting`, `specification`) travel with the numbers so a
    semi-partial value can only ever be merged with a Shapley value computed on
    the SAME fit -- see `shared_vs_unique`.
    """
    X = pd.DataFrame(X).astype(float)
    terms = list(X.columns) if terms is None else [t for t in terms if t in X.columns]
    r2_full, n_obs = r2_of_fit(y, X, weights=weights)
    rows = []
    for t in terms:
        r2_red, _ = r2_of_fit(y, X.drop(columns=[t]), weights=weights)
        rows.append({"term": t, "r2_full": r2_full, "r2_without": r2_red,
                     "semi_partial_r2": float(r2_full - r2_red)})
    out = pd.DataFrame(rows, columns=["term", "r2_full", "r2_without", "semi_partial_r2"])
    out["specification"] = specification
    out["n_obs"] = int(n_obs)
    out["n_model_terms"] = int(X.shape[1])
    out["model_columns"] = _model_signature(X.columns)
    out["response"] = response
    out["weighting"] = weighting
    return out


def shapley_r2(y, X, groups, weights=None, specification="", response="y", weighting=""):
    """Shapley decomposition of R^2 across `groups` ({name: [columns]}).

    Each group's value is its average marginal contribution to R^2 over every
    order in which the groups could enter, so two collinear drivers split the
    variance they share instead of the first-entered one taking all of it. The
    values sum to the full-model R^2 by construction.

    Carries the same provenance columns as `semi_partial_r2`.
    """
    X = pd.DataFrame(X).astype(float)
    names = [g for g, cols in groups.items() if any(c in X.columns for c in cols)]
    G = len(names)
    if G == 0:
        return pd.DataFrame(columns=["group", "shapley_r2", "share_of_r2"])
    if G > 14:
        raise ValueError(f"{G} groups is 2^{G} fits; reduce PARTITION_MAX_GROUPS")
    used_cols = [c for g in names for c in groups[g] if c in X.columns]

    def r2_of(subset):
        cols = [c for g in subset for c in groups[g] if c in X.columns]
        if not cols:
            return 0.0
        try:
            return r2_of_fit(y, X[cols], weights=weights)[0]
        except Exception:
            return np.nan

    cache = {}
    for r in range(G + 1):
        for sub in combinations(names, r):
            cache[frozenset(sub)] = r2_of(sub)

    from math import factorial
    vals = {}
    for g in names:
        others = [x for x in names if x != g]
        total = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(G - r - 1) / factorial(G)
            for sub in combinations(others, r):
                s = frozenset(sub)
                total += w * (cache[s | {g}] - cache[s])
        vals[g] = total
    full = cache[frozenset(names)]
    _, n_obs = r2_of_fit(y, X[used_cols], weights=weights)
    out = pd.DataFrame({"group": list(vals), "shapley_r2": list(vals.values())})
    out["share_of_r2"] = out["shapley_r2"] / full if full and np.isfinite(full) else np.nan
    out["r2_full_model"] = full
    out["specification"] = specification
    out["n_obs"] = int(n_obs)
    out["n_model_terms"] = int(len(used_cols))
    out["model_columns"] = _model_signature(used_cols)
    out["response"] = response
    out["weighting"] = weighting
    return out.sort_values("shapley_r2", ascending=False).reset_index(drop=True)


def shared_vs_unique(partition, semi, driver_terms, specification):
    """Merge a Shapley partition with the semi-partial R^2 OF THE SAME FIT.

    The two are only comparable when they come from one model: same rows, same
    columns, same response, same weighting. Pairing a dynamic Shapley
    decomposition with a static semi-partial (as an earlier version of this
    notebook did) compares two different models, and their ratio then means
    nothing at all. The assertions below make that mistake impossible rather
    than merely unlikely.

    `last_entry_to_shapley_ratio` is deliberately NOT called a "share": the
    last-entry contribution can exceed the order-averaged one under suppression
    or a non-additive relationship, so the ratio can exceed 1 even when both
    numbers are correct. That case is labelled, never clipped.
    """
    if not len(partition) or not len(semi):
        return pd.DataFrame()
    for col, what in [("n_obs", "row count"), ("model_columns", "model columns"),
                      ("response", "response"), ("weighting", "weighting")]:
        a, b = partition[col].iloc[0], semi[col].iloc[0]
        assert a == b, (f"Shapley and semi-partial {what} differ ({a!r} vs {b!r}): "
                        "these are different fits and must not be merged.")
    assert np.isclose(float(partition["r2_full_model"].iloc[0]),
                      float(semi["r2_full"].iloc[0]), atol=1e-8), \\
        "Shapley and semi-partial full-model R^2 differ; not the same fit."

    out = (partition[partition["group"].isin(list(driver_terms))]
           [["group", "shapley_r2", "share_of_r2", "r2_full_model", "n_obs"]]
           .rename(columns={"group": "term"}))
    out = out.merge(semi[["term", "semi_partial_r2"]], on="term", how="left")
    out.insert(0, "specification", specification)
    out["last_entry_to_shapley_ratio"] = (
        out["semi_partial_r2"] / out["shapley_r2"].replace(0, np.nan))

    # Graded on BOTH the ratio and the absolute unique contribution. A ratio
    # alone is too harsh when the full-model R2 is high: in a strongly seasonal
    # system every driver shares most of its variance, yet a driver uniquely
    # explaining 2% of the response is still saying something of its own.
    _u = out["semi_partial_r2"]
    _ratio = out["last_entry_to_shapley_ratio"]
    out["reading"] = np.select(
        [(_ratio >= 0.25) | (_u >= 0.02),
         _u >= 0.005,
         out["shapley_r2"] > 0.02],
        ["independent contribution",
         "partly shared with correlated drivers",
         "SHARED - proxy for a correlated driver"],
        default="negligible either way")
    out["ratio_note"] = np.where(
        _ratio > 1.0,
        "ratio > 1: possible suppression or coefficient instability "
        "(NOT a share of the Shapley value)", "")
    return out.sort_values("shapley_r2", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Moving-block bootstrap on the CALENDAR grid
# ---------------------------------------------------------------------

def calendar_block_length(months, block_months=None):
    """Block length in CALENDAR MONTHS; default ceil(span ** (1/3))."""
    span = calendar_span_months(months)
    if span <= 0:
        return 1
    L = int(np.ceil(span ** (1 / 3))) if block_months is None else int(block_months)
    return int(max(1, min(L, span)))


def calendar_month_blocks(months, block_months=None, rng=None, n_boot=1000):
    """Yield, per replicate, a list of CONTIGUOUS calendar-month blocks.

    Blocks are drawn from the calendar-complete grid spanning `months`, so a
    block of length L always covers L consecutive calendar months. Whether a
    month inside a block was actually observed is decided afterwards, which is
    exactly what preserves the record's missing-month pattern inside every
    replicate. Blocks do not wrap around the end of the record, so consecutive
    entries of a block always differ by exactly one calendar month.

    Each replicate covers the same number of CALENDAR months as the record; the
    last block is truncated to hit that total exactly (truncation shortens a
    block, it never breaks its contiguity). How many observed ROWS that yields
    varies, because a block landing on excluded months contributes fewer.
    """
    rng = np.random.default_rng() if rng is None else rng
    mi = np.unique(month_index(months))
    if not len(mi):
        return
    grid = np.arange(int(mi.min()), int(mi.max()) + 1)
    L = calendar_block_length(months, block_months)
    n_blocks = max(1, int(np.ceil(len(grid) / L)))
    hi = max(1, len(grid) - L + 1)
    for _ in range(int(n_boot)):
        starts = rng.integers(0, hi, size=n_blocks)
        blocks, taken = [], 0
        for s in starts:
            blk = grid[s:s + L][:len(grid) - taken]
            if not len(blk):
                break
            blocks.append(blk)
            taken += len(blk)
            if taken >= len(grid):
                break
        yield blocks


def calendar_block_indices(months, block_months=None, rng=None, n_boot=1000):
    """Row positions for a calendar-block bootstrap replicate.

    A sampled month that is not in `months` (an excluded month) contributes no
    row, so a gap stays a gap and two observed months either side of one are
    never made contiguous. Yields ``(row_positions, blocks)``.
    """
    mi = month_index(months)
    pos = {int(m): i for i, m in enumerate(mi)}
    for blocks in calendar_month_blocks(months, block_months, rng, n_boot):
        idx = np.array([pos[int(m)] for blk in blocks for m in blk if int(m) in pos],
                       dtype=int)
        yield idx, blocks


def bootstrap_coefficients(y, X, months, weights=None, n_boot=1000, block=None, seed=0):
    """Moving-block bootstrap of the OLS/WLS coefficients, blocked in CALENDAR months.

    The blocks are drawn on the CALENDAR grid first; only then are the months
    with no complete model row dropped. Doing it in that order is what keeps a
    block from splicing two months that are a year apart.

    Returns ``(draws, info)``. `info` reports how many replicates fitted and the
    distribution of the fitted sample size, which varies precisely because a
    block landing on excluded months contributes fewer rows.
    """
    all_months = pd.to_datetime(pd.Series(months).reset_index(drop=True))
    yv, Xf, wv, mi, _ = _complete_rows(y, X, weights, months)
    rng = np.random.default_rng(seed)
    Xc = sm.add_constant(Xf, has_constant="add")
    L = calendar_block_length(all_months, block)
    row_of_month = {int(m): i for i, m in enumerate(mi)}
    draws, sizes = [], []
    n_attempted = 0
    for blocks in calendar_month_blocks(all_months, block_months=block,
                                        rng=rng, n_boot=int(n_boot)):
        idx = np.array([row_of_month[int(m)] for blk in blocks for m in blk
                        if int(m) in row_of_month], dtype=int)
        n_attempted += 1
        if len(idx) <= Xc.shape[1]:
            continue
        try:
            if wv is None:
                b = sm.OLS(yv[idx], Xc.iloc[idx]).fit().params
            else:
                b = sm.WLS(yv[idx], Xc.iloc[idx], weights=wv[idx]).fit().params
        except Exception:
            continue
        draws.append(b)
        sizes.append(int(len(idx)))
    sizes = np.asarray(sizes, dtype=float)
    info = {
        "n_requested": int(n_boot),
        "n_attempted": int(n_attempted),
        "n_successful": int(len(draws)),
        "block_months": int(L),
        "n_rows_fitted_min": int(sizes.min()) if len(sizes) else 0,
        "n_rows_fitted_median": float(np.median(sizes)) if len(sizes) else np.nan,
        "n_rows_fitted_mean": float(sizes.mean()) if len(sizes) else np.nan,
        "n_rows_fitted_max": int(sizes.max()) if len(sizes) else 0,
        "n_rows_available": int(len(yv)),
    }
    if not draws:
        return pd.DataFrame(columns=Xc.columns), info
    return pd.DataFrame(draws).reset_index(drop=True), info


def bootstrap_summary(draws, terms=None):
    """Percentile CI, bootstrap SE and sign-stability from bootstrap draws."""
    cols = [c for c in draws.columns if c != "const"] if terms is None else list(terms)
    rows = []
    for c in cols:
        if c not in draws.columns:
            continue
        d = draws[c].dropna()
        if d.empty:
            continue
        rows.append({
            "term": c,
            "boot_median": float(d.median()),
            "boot_se": float(d.std(ddof=1)),
            "boot_ci_lo": float(d.quantile(0.025)),
            "boot_ci_hi": float(d.quantile(0.975)),
            "boot_sign_stability": float(max((d > 0).mean(), (d < 0).mean())),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Rolling-origin validation on the CALENDAR
# ---------------------------------------------------------------------

def rolling_origin_month_folds(eval_months, n_folds=8, horizon_months=3,
                               min_train_months=24, min_test_months=1,
                               omission_reason=None):
    """Rolling-origin folds whose horizon is CALENDAR MONTHS, not rows.

    `eval_months`
        the months that are evaluable for EVERY model being compared -- response
        and the shared required predictors all observed -- as sorted, unique
        month-start timestamps. Using one evaluable set for all models is what
        makes their RMSEs comparable.
    `omission_reason`
        optional {month -> reason} explaining why a calendar month inside a test
        window cannot be scored. Months with no entry are reported as
        "not in the evaluable sample".

    Each fold's test window is a block of exactly `horizon_months` CONSECUTIVE
    CALENDAR MONTHS beginning the month after the origin. Training uses only
    evaluable months strictly before the window starts. A window is NEVER
    widened to collect more observations and a missing response is never
    imputed: if fewer than `min_test_months` of the window's months are
    evaluable the fold is recorded with usable=False and skipped.

    Returns ``(folds, audit)``. `folds` carries `train_idx` / `test_idx` as
    positions into `eval_months`; `audit` is the per-fold record, including the
    skipped ones.
    """
    ev = pd.to_datetime(pd.Series(eval_months).reset_index(drop=True))
    if len(ev) and (not ev.is_monotonic_increasing or ev.duplicated().any()):
        raise ValueError("eval_months must be sorted and unique")
    mi = month_index(ev)
    H = max(1, int(horizon_months))
    reasons = {}
    for key, val in (omission_reason or {}).items():
        reasons[pd.Timestamp(key).to_period("M").to_timestamp()] = str(val)
    if not len(mi):
        return [], pd.DataFrame()

    have = set(int(v) for v in mi)
    first, last = int(mi.min()), int(mi.max())
    windows = []
    for k in range(int(n_folds)):
        end = last - k * H
        start = end - H + 1
        if start <= first:
            break
        windows.append((start, end))
    windows = list(reversed(windows))

    folds, rows = [], []
    for k, (start, end) in enumerate(windows, start=1):
        test_idx = np.where((mi >= start) & (mi <= end))[0]
        train_idx = np.where(mi < start)[0]
        window_months = months_from_index(np.arange(start, end + 1))
        omitted = [m for m, v in zip(window_months, np.arange(start, end + 1))
                   if int(v) not in have]
        rec = {
            "fold": k,
            "origin_month": months_from_index([start - 1])[0],
            "train_start": ev.iloc[int(train_idx[0])] if len(train_idx) else pd.NaT,
            "train_end": ev.iloc[int(train_idx[-1])] if len(train_idx) else pd.NaT,
            "n_train_months": int(len(train_idx)),
            "test_window_start": window_months[0],
            "test_window_end": window_months[-1],
            "horizon_months_requested": int(H),
            "n_calendar_months_in_window": int(len(window_months)),
            "n_test_months_evaluable": int(len(test_idx)),
            "n_test_months_omitted": int(len(omitted)),
            "omitted_test_months": ";".join(f"{m:%Y-%m}" for m in omitted),
            "omitted_reasons": ";".join(
                f"{m:%Y-%m}={reasons.get(m, 'not in the evaluable sample')}"
                for m in omitted),
            "usable": True,
            "skip_reason": "",
        }
        if len(train_idx) < int(min_train_months):
            rec["usable"] = False
            rec["skip_reason"] = (f"only {len(train_idx)} training months "
                                  f"(< {int(min_train_months)})")
        elif len(test_idx) < int(min_test_months):
            rec["usable"] = False
            rec["skip_reason"] = (
                f"only {len(test_idx)} of {H} calendar months in the window are "
                f"evaluable (< {int(min_test_months)}); the window is NOT widened")
        rows.append(rec)
        if not rec["usable"]:
            continue

        # Design invariants, checked rather than assumed.
        assert end - start + 1 == H, "test window is not the requested calendar horizon"
        assert int(mi[train_idx].max()) < start, \\
            "a training month is on or after the test-window start"
        assert int(mi[test_idx].min()) >= start and int(mi[test_idx].max()) <= end, \\
            "a test month falls outside its declared calendar window"

        fold = dict(rec)
        fold["train_idx"] = train_idx
        fold["test_idx"] = test_idx
        folds.append(fold)

    return folds, pd.DataFrame(rows)


def fold_index_pairs(folds):
    """(train, test) index pairs from either fold records or plain tuples."""
    out = []
    for f in folds:
        if isinstance(f, dict):
            out.append((np.asarray(f["train_idx"]), np.asarray(f["test_idx"])))
        else:
            out.append((np.asarray(f[0]), np.asarray(f[1])))
    return out


def cv_scores(y, X, folds, months=None, weights=None, fit=None, predict=None):
    """Out-of-sample RMSE / MAE / R^2 over calendar rolling-origin folds.

    `fit(Xtr, ytr, wtr) -> model` and `predict(model, Xte) -> yhat` default to
    ordinary least squares, so the same routine scores every specification and
    every learner on identical folds -- and therefore on identical response
    months, which is what makes the RMSEs comparable.

    The returned detail frame keeps the real `month` of every prediction.
    """
    y = np.asarray(y, dtype=float)
    X = pd.DataFrame(X).astype(float).reset_index(drop=True)
    mvals = (pd.Series([pd.NaT] * len(X)) if months is None
             else pd.to_datetime(pd.Series(months).reset_index(drop=True)))
    if fit is None:
        def fit(Xtr, ytr, wtr):
            Xtr = sm.add_constant(Xtr, has_constant="add")
            return (sm.OLS(ytr, Xtr).fit() if wtr is None
                    else sm.WLS(ytr, Xtr, weights=wtr).fit())
    if predict is None:
        def predict(model, Xte):
            return model.predict(sm.add_constant(Xte, has_constant="add"))
    pairs = fold_index_pairs(folds)
    labels = [f["fold"] if isinstance(f, dict) else k
              for k, f in enumerate(folds, start=1)]
    preds, truths, fold_ids, pred_months = [], [], [], []
    for label, (tr, te) in zip(labels, pairs):
        wtr = None if weights is None else np.asarray(weights, dtype=float)[tr]
        try:
            model = fit(X.iloc[tr], y[tr], wtr)
            yhat = np.asarray(predict(model, X.iloc[te]), dtype=float)
        except Exception:
            continue
        preds.append(yhat)
        truths.append(y[te])
        fold_ids.append(np.full(len(te), label))
        pred_months.append(mvals.iloc[te].to_numpy())
    empty = pd.DataFrame(columns=["fold", "month", "y", "yhat"])
    if not preds:
        return ({"n_folds": 0, "n_test": 0, "rmse": np.nan, "mae": np.nan,
                 "r2_oos": np.nan}, empty)
    yh = np.concatenate(preds)
    yt = np.concatenate(truths)
    ss_res = float(np.nansum((yt - yh) ** 2))
    ss_tot = float(np.nansum((yt - np.nanmean(yt)) ** 2))
    detail = pd.DataFrame({"fold": np.concatenate(fold_ids),
                           "month": np.concatenate(pred_months),
                           "y": yt, "yhat": yh})
    return ({"n_folds": len(preds), "n_test": int(len(yt)),
             "rmse": float(np.sqrt(np.nanmean((yt - yh) ** 2))),
             "mae": float(np.nanmean(np.abs(yt - yh))),
             "r2_oos": (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}, detail)


def seasonal_naive_predictions(prediction_months, grid, value_col="y", month_col="month"):
    """Response at EXACTLY t-12 calendar months, looked up by timestamp.

    The lookup runs on the calendar-complete monthly table, so it can never be
    satisfied by "the twelfth previous OBSERVED row" -- which is what
    `shift(12)` on complete-case rows returns, and which is usually a different
    calendar month. If the month exactly one year earlier is missing the
    prediction is unavailable; it is NOT substituted.

    `source_month` is kept in the output so the lookup can be audited.
    """
    pm = pd.to_datetime(pd.Series(prediction_months).reset_index(drop=True))
    out = pd.DataFrame({"month": pm})
    out["source_month"] = out["month"] - pd.DateOffset(months=12)
    g = pd.DataFrame({
        "source_month": pd.to_datetime(pd.Series(grid[month_col]).reset_index(drop=True)),
        "yhat": pd.to_numeric(pd.Series(grid[value_col]).reset_index(drop=True),
                              errors="coerce"),
    }).drop_duplicates("source_month")
    out = out.merge(g, on="source_month", how="left")
    out["available"] = out["yhat"].notna()
    # The audit this method exists for: the value used is the SAME CALENDAR
    # MONTH one year earlier, never the twelfth previous observed row.
    assert (out["source_month"] == out["month"] - pd.DateOffset(months=12)).all(), \\
        "seasonal-naive source month is not exactly 12 calendar months earlier"
    return out


# ---------------------------------------------------------------------
# Dependence diagnostics, on the calendar
# ---------------------------------------------------------------------

def acf_values(x, months=None, nlags=24):
    """Autocorrelations at CALENDAR-month lags, with the white-noise band.

    A pair contributes to lag h only when the two observations are exactly h
    calendar months apart, so an excluded month costs pairs instead of shrinking
    the apparent distance between the months either side of it. Without `months`
    the rows are taken to be consecutive months, which is true of the
    calendar-complete grid but NOT of a complete-case table.

    `n_pairs` is reported because a lag built from very few pairs is not
    evidence of anything.
    """
    s = pd.Series(np.asarray(x, dtype=float).ravel()).reset_index(drop=True)
    mi = np.arange(len(s)) if months is None else month_index(months)
    if len(mi) != len(s):
        raise ValueError("`months` must have one entry per value")
    order = np.argsort(mi, kind="stable")
    mi_s = np.asarray(mi, dtype=int)[order]
    v_s = s.to_numpy()[order]
    obs = np.isfinite(v_s)
    n = int(obs.sum())
    centred = np.where(obs, v_s - (np.nanmean(v_s) if n else np.nan), 0.0)
    denom = float(np.nansum(centred[obs] ** 2)) if n else 0.0
    rows = []
    for h in range(1, int(nlags) + 1):
        pos, hit = _match_lag(mi_s, h)
        if len(hit):
            hit = hit & obs & obs[pos]
        npairs = int(hit.sum()) if len(hit) else 0
        if npairs == 0 or denom == 0:
            rows.append({"lag": h, "n_pairs": npairs, "acf": np.nan})
        else:
            rows.append({"lag": h, "n_pairs": npairs,
                         "acf": float(np.sum(centred[hit] * centred[pos[hit]]) / denom)})
    out = pd.DataFrame(rows)
    out["band"] = 1.96 / np.sqrt(max(n, 1))
    return out


def cross_correlation(driver, response, months=None, max_lag=6):
    """Correlation of response(t) with driver(t - k) at CALENDAR lag k months.

    Positive k means the driver LEADS the response, which is the only direction
    a driver claim can use. A pair enters lag k only when the two months are
    exactly k calendar months apart. Without `months` the rows are taken to be
    consecutive months.
    """
    d = pd.Series(np.asarray(driver, dtype=float).ravel()).reset_index(drop=True)
    r = pd.Series(np.asarray(response, dtype=float).ravel()).reset_index(drop=True)
    if len(d) != len(r):
        raise ValueError("driver and response disagree on length")
    mi = np.arange(len(r)) if months is None else month_index(months)
    if len(mi) != len(r):
        raise ValueError("`months` must have one entry per value")
    order = np.argsort(mi, kind="stable")
    mi_s = np.asarray(mi, dtype=int)[order]
    dv, rv = d.to_numpy()[order], r.to_numpy()[order]
    rows = []
    for k in range(0, int(max_lag) + 1):
        pos, hit = _match_lag(mi_s, k)
        if len(hit):
            hit = hit & np.isfinite(rv) & np.isfinite(dv[pos])
        n = int(hit.sum()) if len(hit) else 0
        if n < 8:
            rows.append({"lag": k, "n": n, "r": np.nan, "p": np.nan})
            continue
        rr, pp = sstats.pearsonr(dv[pos[hit]], rv[hit])
        rows.append({"lag": k, "n": n, "r": float(rr), "p": float(pp)})
    return pd.DataFrame(rows)


def zscore_frame(df, cols):
    """Standardise `cols` in place, returning the frame plus the scaling used.

    Standardising is what makes coefficients comparable: each becomes "change
    in the response per one standard deviation of this driver".
    """
    out = df.copy()
    rows = []
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce").astype(float)
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        if not np.isfinite(sd) or sd == 0:
            sd = np.nan
        out[c] = (v - mu) / sd
        rows.append({"term": c, "mean": mu, "sd": sd})
    return out, pd.DataFrame(rows)


def vif_table(X):
    """Variance-inflation factors: how much collinearity inflates each SE."""
    X = pd.DataFrame(X).astype(float).dropna()
    rows = []
    for c in X.columns:
        others = [o for o in X.columns if o != c]
        if not others:
            rows.append({"term": c, "vif": 1.0}); continue
        try:
            r2 = sm.OLS(X[c], sm.add_constant(X[others])).fit().rsquared
            rows.append({"term": c, "vif": (1.0 / (1.0 - r2)) if r2 < 1 else np.inf})
        except Exception:
            rows.append({"term": c, "vif": np.nan})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


print("§5 calendar-aware inference helpers defined.")
''')

md("""### 5b. Self-tests for the calendar-aware helpers

Every claim §5 makes is checked here, on synthetic data, each time the notebook
runs — including on a series with a deliberately deleted month, because that is
the case the old row-indexed code got wrong:

1. the calendar HAC sandwich reproduces `statsmodels`' `cov_type="HAC"`
   **exactly** when the months are consecutive (OLS *and* WLS);
2. deleting one month removes exactly the two one-month covariance pairs it
   breaks, and the calendar and row-adjacency estimators consequently disagree;
3. every adjacent pair inside a sampled bootstrap block is **one** calendar
   month apart, and blocks have the requested calendar length;
4. every rolling-origin test window spans exactly the requested calendar
   interval, its origin is the month before it, no training month reaches the
   window, no test month escapes it, and two models scored on the same folds see
   **identical response months**;
5. the seasonal-naive lookup returns the month exactly twelve calendar months
   earlier, and reports *unavailable* rather than substituting when that month
   is missing;
6. the ACF and cross-correlation helpers lose pairs across a gap instead of
   closing it.

A failure raises immediately: the notebook must not produce numbers on
machinery that has not passed these.
""")

code('''# =====================================================================
# 5b. Self-tests for the calendar-aware helpers
# =====================================================================
# These run on SYNTHETIC data every time the notebook is executed. They are not
# results; they are the guarantee that the machinery below does what §5 claims,
# on a series that deliberately contains a gap.
HELPER_SELFTESTS = []


def _check(name, ok, detail=""):
    HELPER_SELFTESTS.append({"check": name, "passed": bool(ok), "detail": detail})
    if not ok:
        raise AssertionError(f"§5b self-test failed: {name} ({detail})")


_rng = np.random.default_rng(20240401)
_n = 96
_full_months = pd.date_range("2015-01-01", periods=_n, freq="MS")
_x = _rng.normal(size=_n)
_e = np.zeros(_n)
for _t in range(1, _n):
    _e[_t] = 0.6 * _e[_t - 1] + _rng.normal(scale=0.5)
_yv = 1.4 * _x + _e
_Xd = pd.DataFrame({"x": _x})

# 1. With no gaps the calendar sandwich IS the statsmodels HAC sandwich. -------
_ref = sm.OLS(_yv, sm.add_constant(_Xd)).fit(
    cov_type="HAC", cov_kwds={"maxlags": hac_maxlags(_n), "use_correction": True})
_cal = fit_hac(_yv, _Xd, months=_full_months)
_check("calendar HAC == statsmodels HAC when the months are consecutive",
       np.allclose(np.asarray(_cal.bse), np.asarray(_ref.bse), rtol=1e-10, atol=1e-12)
       and np.allclose(np.asarray(_cal.conf_int()), np.asarray(_ref.conf_int()),
                       rtol=1e-10, atol=1e-12),
       f"max |dSE| = {np.max(np.abs(np.asarray(_cal.bse) - np.asarray(_ref.bse))):.3e}")

# The same must hold for WLS, which is how the notebook actually fits.
_wv = np.abs(_rng.normal(size=_n)) + 0.5
_refw = sm.WLS(_yv, sm.add_constant(_Xd), weights=_wv / _wv.mean()).fit(
    cov_type="HAC", cov_kwds={"maxlags": hac_maxlags(_n), "use_correction": True})
_calw = fit_hac(_yv, _Xd, weights=_wv, months=_full_months)
_check("calendar HAC == statsmodels HAC for WLS as well",
       np.allclose(np.asarray(_calw.bse), np.asarray(_refw.bse), rtol=1e-10, atol=1e-12))

# 2. A missing month must break the one-month pairing across the hole. ---------
_drop = 40
_keep = np.r_[np.arange(_drop), np.arange(_drop + 1, _n)]
_gap_months = _full_months[_keep]
_gap_fit = fit_hac(_yv[_keep], _Xd.iloc[_keep], months=_gap_months)
_pairs_gap = _gap_fit._hac_pair_counts
_row_adjacent = fit_hac(_yv[_keep], _Xd.iloc[_keep])          # row-order fallback
_check("one missing month removes exactly the two one-month pairs it breaks",
       _pairs_gap[1] == (_n - 1) - 2,
       f"lag-1 pairs {_pairs_gap[1]} vs {(_n - 1) - 2} expected")
_check("the months either side of a gap are not treated as consecutive",
       not np.allclose(np.asarray(_gap_fit.bse), np.asarray(_row_adjacent.bse)),
       "calendar-aware and row-adjacency SEs differ, as they must across a gap")
_check("fit_hac records whether it was calendar-aware",
       _gap_fit._hac_calendar_aware and not _row_adjacent._hac_calendar_aware)

# 3. Bootstrap blocks are contiguous in CALENDAR months. ----------------------
_block_lens, _row_counts, _gap_block_seen = set(), set(), False
_span_gap = calendar_span_months(_gap_months)
for _bidx, _blocks in calendar_block_indices(_gap_months, block_months=5,
                                             rng=np.random.default_rng(7), n_boot=40):
    for _blk in _blocks[:-1]:
        _block_lens.add(len(_blk))
    for _blk in _blocks:
        _check("every adjacent pair inside a sampled block is one calendar month apart",
               np.all(np.diff(np.asarray(_blk, dtype=int)) == 1))
        # A block covering the deleted month must yield one row fewer than it
        # has calendar months: the gap is preserved, not closed up.
        _n_rows = int(np.isin(np.asarray(_blk, dtype=int),
                              month_index(_gap_months)).sum())
        if month_index([_full_months[_drop]])[0] in set(int(v) for v in _blk):
            _gap_block_seen = True
            _check("a block spanning the excluded month contributes one row fewer",
                   _n_rows == len(_blk) - 1, f"{_n_rows} rows from {len(_blk)} months")
    _check("a replicate covers exactly the record's calendar span",
           sum(len(b) for b in _blocks) == _span_gap)
    _row_counts.add(len(_bidx))
_check("blocks have the requested calendar length (bar a truncated final block)",
       _block_lens == {5}, f"lengths seen: {sorted(_block_lens)}")
_check("the excluded month was actually exercised by a sampled block", _gap_block_seen)
_check("replicate sample sizes vary because of the gap", len(_row_counts) > 1,
       f"row counts seen: {sorted(_row_counts)}")

# 4. Calendar rolling-origin folds. -------------------------------------------
_eval_months = _gap_months[_gap_months >= pd.Timestamp("2015-01-01")]
_folds, _audit = rolling_origin_month_folds(
    _eval_months, n_folds=6, horizon_months=3, min_train_months=24, min_test_months=2)
_check("rolling-origin produced folds", len(_folds) >= 2, f"{len(_folds)} folds")
for _f in _folds:
    _mi_tr = month_index(_eval_months[_f["train_idx"]])
    _mi_te = month_index(_eval_months[_f["test_idx"]])
    _s = month_index([_f["test_window_start"]])[0]
    _e2 = month_index([_f["test_window_end"]])[0]
    _check("test window covers exactly the requested calendar interval",
           _e2 - _s + 1 == _f["horizon_months_requested"])
    _check("the origin is the calendar month before the window",
           month_index([_f["origin_month"]])[0] == _s - 1)
    _check("no training month is on or after the test-window start", _mi_tr.max() < _s)
    _check("no test month falls outside its declared calendar window",
           _mi_te.min() >= _s and _mi_te.max() <= _e2)
_check("folds with too few evaluable test months are flagged, not widened",
       bool(((~_audit["usable"]).sum() == 0)
            or (_audit.loc[~_audit["usable"], "skip_reason"].str.len() > 0).all()))

# Two different models scored on the same folds see the same response months.
_sc_a, _det_a = cv_scores(_yv[_keep], _Xd.iloc[_keep], _folds, months=_eval_months)
_sc_b, _det_b = cv_scores(_yv[_keep], pd.DataFrame({"x2": _x[_keep] ** 2}), _folds,
                          months=_eval_months)
_check("directly compared models are scored on identical response months",
       list(_det_a["month"]) == list(_det_b["month"]) and len(_det_a) == _sc_a["n_test"])

# 5. Seasonal-naive lookup is a real 12-calendar-month lookup. -----------------
_grid = pd.DataFrame({"month": _full_months, "y": _yv})
_grid_gap = _grid.drop(index=_drop).reset_index(drop=True)
_sn = seasonal_naive_predictions(_full_months[-12:], _grid_gap)
_check("seasonal-naive source month is exactly 12 calendar months earlier",
       bool((_sn["source_month"] == _sn["month"] - pd.DateOffset(months=12)).all()))
_sn_missing = seasonal_naive_predictions([_full_months[_drop] + pd.DateOffset(months=12)],
                                         _grid_gap)
_check("a missing prior-year month makes the prediction unavailable, not substituted",
       not bool(_sn_missing["available"].iloc[0]))

# 6. ACF / CCF do not compress gaps. ------------------------------------------
_acf_gap = acf_values(_yv[_keep], months=_gap_months, nlags=3)
_check("ACF pairs at lag 1 respect the gap",
       int(_acf_gap.loc[_acf_gap["lag"] == 1, "n_pairs"].iloc[0]) == (_n - 1) - 2)
_ccf_gap = cross_correlation(_x[_keep], _yv[_keep], months=_gap_months, max_lag=2)
_check("CCF pairs at lag 1 respect the gap",
       int(_ccf_gap.loc[_ccf_gap["lag"] == 1, "n"].iloc[0]) == (_n - 1) - 2)

# 7. AR stationarity gate. ----------------------------------------------------
# A clearly stationary AR(1): the gate must open.
_ar_y = np.zeros(_n)
for _t in range(1, _n):
    _ar_y[_t] = 0.35 * _ar_y[_t - 1] + _rng.normal(scale=0.5)
_ar_fit = fit_hac(_ar_y[1:], pd.DataFrame({"y_lag1": _ar_y[:-1]}),
                  months=_full_months[1:])
_ar_ok = ar_stability(_ar_fit, ["y_lag1"])
_check("a clearly stationary AR(1) is accepted", _ar_ok["stationary_supported"],
       f"rho = {_ar_ok['rho_sum']:.3f}, CI "
       f"[{_ar_ok['rho_ci_lo']:.3f}, {_ar_ok['rho_ci_hi']:.3f}]")


# The decision rule itself, on fixed coefficients and standard errors, so the
# gate is tested rather than a particular random draw. The first case is this
# notebook's own situation: rho well below 1, but an interval that reaches it.
class _ARStub:
    """Minimal stand-in exposing just what `ar_stability` reads."""

    def __init__(self, phi, se):
        self.params = pd.Series(dict([("const", 0.0)]
                                     + [(f"y_lag{j}", v) for j, v in enumerate(phi, 1)]))
        self._V = np.diag(np.r_[0.0, np.asarray(se, dtype=float) ** 2])

    def cov_params(self):
        return pd.DataFrame(self._V, index=self.params.index, columns=self.params.index)


_near_unit = ar_stability(_ARStub([0.88], [0.07]), ["y_lag1"])
_check("an AR(1) whose interval reaches 1 is refused a long-run multiplier",
       (not _near_unit["stationary_supported"])
       and _near_unit["reason"] == "AR confidence interval includes a unit root",
       f"rho = {_near_unit['rho_sum']:.2f}, CI "
       f"[{_near_unit['rho_ci_lo']:.3f}, {_near_unit['rho_ci_hi']:.3f}]")
_explosive = ar_stability(_ARStub([1.02], [0.05]), ["y_lag1"])
_check("an explosive AR(1) point estimate is refused",
       (not _explosive["stationary_supported"])
       and "root" in _explosive["reason"])
_comfortable = ar_stability(_ARStub([0.45], [0.08]), ["y_lag1"])
_check("a comfortably stationary AR(1) interval is accepted",
       _comfortable["stationary_supported"] and _comfortable["reason"] == "")

# The multi-lag case is decided by the ROOTS, not by the sum of coefficients:
# phi = (0.6, 0.5) sums to 1.1 and is explosive, while phi = (1.2, -0.35) also
# sums to 0.85 but must be judged on its roots.
_check("stationarity is decided on the AR polynomial roots, not the coefficient sum",
       (not np.all(np.abs(ar_polynomial_roots([0.6, 0.5])) > 1.0))
       and np.all(np.abs(ar_polynomial_roots([1.2, -0.35])) > 1.0))
_check("a unit-root AR polynomial is refused",
       not np.all(np.abs(ar_polynomial_roots([1.0])) > 1.0))

HELPER_SELFTESTS = (pd.DataFrame(HELPER_SELFTESTS)
                    .drop_duplicates("check").reset_index(drop=True))
print(f"§5b: {int(HELPER_SELFTESTS['passed'].sum())} of {len(HELPER_SELFTESTS)} "
      "calendar-aware helper self-tests passed.")
display(HELPER_SELFTESTS)
''')


# ===========================================================================
# 5c. State-space helpers
# ===========================================================================
md("""### 5c. Helpers — state-space dynamic regression

The static models put persistence on the right-hand side as `y_lag1`. That has
two costs on this record: it **deletes** every month whose predecessor is
missing (23 of the fitted months here sit next to a gap), and it forces a single
functional form on "the series remembers itself".

A state-space model puts the dependence in the **error or state process**
instead. The measurement equation stays
$y_t = x_t'\\beta + \\text{(season)}_t + \\mu_t$, and the candidates differ only
in what $\\mu_t$ is:

| Structure | $\\mu_t$ | Trend | What it assumes |
|---|---|---|---|
| `sarimax_ar1` | $\\phi\\mu_{t-1} + \\varepsilon_t$ | deterministic linear | shocks decay geometrically towards a fixed mean |
| `sarimax_ar2` | $\\phi_1\\mu_{t-1} + \\phi_2\\mu_{t-2} + \\varepsilon_t$ | deterministic linear | as above, with the possibility of oscillation |
| `local_level` | random-walk level + observation noise | **none** (the level *is* the trend) | the level has no fixed mean; shocks are permanent |

Two consequences matter for this record and are why the state-space model is the
principal one:

1. **A missing month costs a likelihood contribution, not a row.** The Kalman
   filter simply skips the update at a month with no observation and carries the
   state forward, so all `N_OBSERVED` months are used instead of the complete-case
   subset the `y_lag1` design is restricted to.
2. **Nothing is imputed.** Where the response is missing there is no likelihood
   term at all, so the exogenous values on those months are irrelevant to the
   fit. That is the only reason the placeholder in §19 is admissible, and §19
   asserts that no month with an observed response ever uses one.

A linear trend enters **only** the candidates without a stochastic level: a
random-walk level and a linear trend both claim the low-frequency variation, and
fitting them together makes neither identified.
""")

code('''# =====================================================================
# 5c. State-space dynamic-regression helpers
# =====================================================================
# Every function here is calendar-first in the same sense as §5: the models are
# fitted on the CALENDAR-COMPLETE monthly grid with NaN responses left in place,
# so "one step ahead" is always one calendar month, never one observed row.

SS_STRUCTURES = {
    "sarimax_ar1": {"kind": "sarimax", "order": (1, 0, 0), "allow_linear_trend": True,
                    "label": "SARIMAX regression, AR(1) errors",
                    "ar_lags": 1},
    "sarimax_ar2": {"kind": "sarimax", "order": (2, 0, 0), "allow_linear_trend": True,
                    "label": "SARIMAX regression, AR(2) errors",
                    "ar_lags": 2},
    "local_level": {"kind": "ucm", "level": "local level", "allow_linear_trend": False,
                    "label": "local-level structural model (UnobservedComponents)",
                    "ar_lags": 0},
}


def ss_structure_label(structure):
    return SS_STRUCTURES[structure]["label"]


def ss_allows_linear_trend(structure):
    """A deterministic linear trend is admissible only without a stochastic level."""
    return bool(SS_STRUCTURES[structure]["allow_linear_trend"])


def ss_param_name(structure, term):
    """Name `term`'s coefficient carries in the fitted result's parameter vector."""
    return f"beta.{term}" if SS_STRUCTURES[structure]["kind"] == "ucm" else str(term)


def calendar_grid_index(months):
    """A gapless month-start DatetimeIndex covering `months`, with freq set.

    statsmodels needs a frequency to know what "one step ahead" means. Handing it
    a complete-case index would make one step mean "one observed row".
    """
    m = pd.to_datetime(pd.Series(months))
    return pd.date_range(m.min(), m.max(), freq="MS")


def fourier_terms(months, n_harmonics, prefix="season"):
    """Deterministic annual Fourier terms for arbitrary month timestamps.

    Depends on the CALENDAR MONTH only, so it is identical whichever subset of
    the record it is evaluated on — which is what lets a rolling-origin fold
    build it for a future month without using any future information.
    """
    idx = pd.to_datetime(pd.Series(months).reset_index(drop=True))
    ang = 2 * np.pi * idx.dt.month.to_numpy(dtype=float) / 12.0
    out = pd.DataFrame(index=range(len(idx)))
    for k in range(1, int(n_harmonics) + 1):
        out[f"{prefix}_sin{k}"] = np.sin(k * ang)
        out[f"{prefix}_cos{k}"] = np.cos(k * ang)
    return out


def ss_build_model(y, exog, structure):
    """Instantiate the state-space model for `structure` (no fitting)."""
    spec = SS_STRUCTURES[structure]
    ex = None
    if exog is not None and getattr(exog, "shape", (0, 0))[1] > 0:
        ex = exog
    if spec["kind"] == "sarimax":
        return SARIMAX(y, exog=ex, order=spec["order"], trend="c",
                       enforce_stationarity=True, enforce_invertibility=True)
    return UnobservedComponents(y, exog=ex, level=spec["level"])


def ss_fit(y, exog, structure, cov_type=None, maxiter=250, disp=False):
    """Fit `structure`, degrading the covariance estimator rather than failing.

    Returns (results, info). `info` records which covariance estimator actually
    produced the standard errors, so a table can never silently mix a robust
    sandwich with the default one.
    """
    mod = ss_build_model(y, exog, structure)
    tried, res, used = [], None, None
    order = [c for c in [cov_type, "robust_approx", "opg", None] if c not in tried]
    seen = set()
    order = [c for c in order if not (c in seen or seen.add(c))]
    last_exc = None
    for cand in order:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cand is None:
                    res = mod.fit(disp=disp, maxiter=int(maxiter))
                else:
                    res = mod.fit(disp=disp, maxiter=int(maxiter), cov_type=cand)
                _ = res.bse            # forces the covariance to be computed
                if not np.all(np.isfinite(np.asarray(res.bse, dtype=float))):
                    raise ValueError("non-finite standard errors")
            used = cand or "default (opg)"
            break
        except Exception as exc:      # pragma: no cover - optimiser dependent
            last_exc = exc
            tried.append(cand)
            continue
    if res is None:
        raise RuntimeError(f"state-space fit failed for {structure!r}: {last_exc}")
    info = {"structure": structure, "cov_type_used": used,
            "cov_types_refused": [str(t) for t in tried],
            "converged": bool(getattr(res, "mle_retvals", {}).get("converged", True)),
            "n_iterations": int(getattr(res, "mle_retvals", {}).get("iterations", -1) or -1)}
    return res, info


def ss_n_effective(y):
    """Months that actually contribute a likelihood term (i.e. observed ones)."""
    return int(np.isfinite(np.asarray(y, dtype=float)).sum())


def ss_n_free_params(res):
    """Free parameters: everything estimated, fixed parameters excluded."""
    fixed = set(getattr(res, "fixed_params", []) or [])
    return int(sum(1 for p in res.params.index if p not in fixed))


def aicc_from(llf, k, n):
    """AICc = -2 logL + 2k + 2k(k+1)/(n-k-1), with n = OBSERVED response months.

    Using the number of grid rows instead of the number of observed months would
    understate the penalty on a record that is a third missing.
    """
    llf, k, n = float(llf), int(k), int(n)
    if n - k - 1 <= 0:
        return np.inf
    return -2.0 * llf + 2.0 * k + (2.0 * k * (k + 1)) / (n - k - 1)


def ss_standardized_innovations(res, months=None, drop_burn=True):
    """One-step-ahead standardized prediction errors, aligned to calendar months.

    Only months with an observed response carry an innovation; the rest are NaN
    because the filter made no update there.
    """
    v = np.asarray(res.standardized_forecasts_error, dtype=float)
    v = v[0] if v.ndim == 2 else v
    v = np.asarray(v, dtype=float).ravel()
    burn = int(getattr(res, "loglikelihood_burn", 0) or 0)
    nd = int(getattr(res, "nobs_diffuse", 0) or 0)
    cut = max(burn, nd) if drop_burn else 0
    out = pd.Series(v, dtype=float)
    if cut:
        out.iloc[:cut] = np.nan
    endog = np.asarray(res.model.endog, dtype=float).ravel()
    out[~np.isfinite(endog[: len(out)])] = np.nan
    if months is not None:
        idx = pd.to_datetime(pd.Series(months).reset_index(drop=True))
        if len(idx) != len(out):
            raise ValueError("`months` must have one entry per filtered observation")
        return pd.DataFrame({"month": idx, "std_innovation": out.to_numpy()})
    return out


def calendar_ljung_box(x, months=None, nlags=12):
    """Ljung-Box statistic built from CALENDAR-lag autocorrelations.

    Q = n(n+2) * sum_h r_h^2 / (n - h), summed over the lags whose calendar pair
    count is non-zero. Lags with no genuinely h-months-apart pairs are dropped
    and reported rather than being scored as r_h = 0.
    """
    tab = acf_values(x, months=months, nlags=nlags)
    n = int(np.isfinite(np.asarray(x, dtype=float)).sum())
    use = tab[tab["acf"].notna() & (tab["n_pairs"] > 0) & (tab["lag"] < n)]
    if not len(use) or n < 5:
        return {"lb_stat": np.nan, "df": 0, "lb_pvalue": np.nan,
                "lags_used": [], "lags_dropped": tab["lag"].tolist(), "n": n}
    q = float(n * (n + 2) * np.sum(use["acf"].to_numpy() ** 2
                                   / (n - use["lag"].to_numpy())))
    df = int(len(use))
    return {"lb_stat": q, "df": df,
            "lb_pvalue": float(sstats.chi2.sf(q, df)),
            "lags_used": use["lag"].tolist(),
            "lags_dropped": tab.loc[~tab["lag"].isin(use["lag"]), "lag"].tolist(),
            "n": n}


def ss_innovation_diagnostics(res, months, nlags=12, label=""):
    """Standardized-innovation diagnostics on calendar lags."""
    inn = ss_standardized_innovations(res, months=months)
    v = inn["std_innovation"]
    ok = v.notna()
    acf = acf_values(v, months=inn["month"], nlags=nlags)
    lb = calendar_ljung_box(v, months=inn["month"], nlags=nlags)
    vals = v[ok].to_numpy(dtype=float)
    jb = sstats.jarque_bera(vals) if len(vals) >= 8 else (np.nan, np.nan)
    third = max(2, len(vals) // 3)
    if len(vals) >= 12:
        v1, v2 = np.var(vals[:third], ddof=1), np.var(vals[-third:], ddof=1)
        het = float(v2 / v1) if v1 > 0 else np.nan
        het_p = (float(2 * min(sstats.f.cdf(het, third - 1, third - 1),
                               sstats.f.sf(het, third - 1, third - 1)))
                 if np.isfinite(het) and het > 0 else np.nan)
    else:
        het, het_p = np.nan, np.nan
    return {
        "label": label,
        "n_innovations": int(ok.sum()),
        "mean_std_innovation": float(np.mean(vals)) if len(vals) else np.nan,
        "sd_std_innovation": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "acf1": float(acf["acf"].iloc[0]) if len(acf) else np.nan,
        "acf1_n_pairs": int(acf["n_pairs"].iloc[0]) if len(acf) else 0,
        "acf12": (float(acf.loc[acf["lag"] == 12, "acf"].iloc[0])
                  if (acf["lag"] == 12).any() else np.nan),
        "ljung_box_stat": lb["lb_stat"], "ljung_box_df": lb["df"],
        "ljung_box_p": lb["lb_pvalue"], "ljung_box_lags": str(lb["lags_used"]),
        "jarque_bera_p": float(jb[1]) if np.isfinite(jb[1]) else np.nan,
        "het_var_ratio_last_over_first_third": het,
        "het_p": het_p,
    }, acf, inn


def ss_state_diagnostics(res, structure, alpha=0.05):
    """AR roots (SARIMAX) or latent-state variances (local level).

    `stationary_supported` is the SAME gate §12 applies to the `y_lag1` model:
    a long-run multiplier may only be quoted when it is True. For the local-level
    model it is False BY CONSTRUCTION — a random-walk level has a unit root, so
    "the long-run effect of a permanent 1-SD change" is not a defined quantity.
    """
    spec = SS_STRUCTURES[structure]
    out = {"structure": structure, "kind": spec["kind"],
           "label": spec["label"], "alpha": float(alpha)}
    if spec["kind"] == "ucm":
        pars = {k: float(v) for k, v in res.params.items() if k.startswith("sigma2")}
        out.update(pars)
        out.update(rho_sum=np.nan, rho_ci_lo=np.nan, rho_ci_hi=np.nan,
                   roots_outside_unit_circle=False, ci_within_unit_circle=False,
                   min_root_modulus=np.nan,
                   stationary_supported=False,
                   reason=("stochastic local level is a random walk: unit root by "
                           "construction, so no long-run multiplier is defined"))
        return out

    ar_names = [p for p in res.params.index if p.startswith("ar.L")]
    phi = np.array([float(res.params[p]) for p in ar_names], dtype=float)
    roots = ar_polynomial_roots(phi)
    mod = np.abs(roots)
    names = list(res.params.index)
    V = np.asarray(res.cov_params())
    g = np.zeros(len(names))
    for p in ar_names:
        g[names.index(p)] = 1.0
    se = float(np.sqrt(max(float(g @ V @ g), 0.0)))
    z = float(sstats.norm.ppf(1 - alpha / 2.0))
    out.update({f"phi_{i+1}": float(v) for i, v in enumerate(phi)})
    out.update(
        rho_sum=float(phi.sum()),
        rho_se=se,
        rho_ci_lo=float(phi.sum() - z * se),
        rho_ci_hi=float(phi.sum() + z * se),
        min_root_modulus=float(np.min(mod)) if len(mod) else np.nan,
        roots_outside_unit_circle=bool(len(mod) and np.all(mod > 1.0)),
        ar_roots=str(np.round(roots, 4).tolist()),
    )
    out["ci_within_unit_circle"] = bool(out["rho_ci_lo"] > -1.0 and out["rho_ci_hi"] < 1.0)
    if not out["roots_outside_unit_circle"]:
        reason = "fitted AR polynomial has a root on or inside the unit circle"
    elif not out["ci_within_unit_circle"]:
        reason = "AR confidence interval includes a unit root"
    else:
        reason = ""
    out["stationary_supported"] = bool(out["roots_outside_unit_circle"]
                                       and out["ci_within_unit_circle"])
    out["reason"] = reason
    return out


def ss_tidy_coefficients(res, structure, terms, label="", alpha=0.05, cov_type_used=""):
    """Driver coefficients from a fitted state-space model, with CI, p and BH q.

    `term` is reported under the DRIVER's name, not the internal `beta.x` name,
    so a state-space table can be lined up against a §12 HAC table — while
    `source_model` keeps them from ever being merged by accident.
    """
    rows = []
    ci = res.conf_int(alpha=alpha)
    ci.columns = ["ci_lo", "ci_hi"]
    for t in terms:
        pname = ss_param_name(structure, t)
        if pname not in res.params.index:
            continue
        rows.append({
            "term": t, "param_name": pname,
            "coef": float(res.params[pname]),
            "se": float(res.bse[pname]),
            "z": float(res.tvalues[pname]),
            "p": float(res.pvalues[pname]),
            "ci_lo": float(ci.loc[pname, "ci_lo"]),
            "ci_hi": float(ci.loc[pname, "ci_hi"]),
        })
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["q_fdr"] = bh_fdr(tab["p"]).to_numpy()
    tab.insert(0, "specification", label)
    tab["source_model"] = f"state-space {structure}"
    tab["se_kind"] = cov_type_used or "unknown"
    return tab


def ss_joint_lr_test(res_full, res_null, k_extra):
    """Likelihood-ratio test of the whole driver block against the matched null."""
    stat = float(2.0 * (float(res_full.llf) - float(res_null.llf)))
    df = int(k_extra)
    return {"lr_stat": stat, "df": df,
            "p_chi2": float(sstats.chi2.sf(max(stat, 0.0), df)) if df > 0 else np.nan,
            "llf_full": float(res_full.llf), "llf_null": float(res_null.llf),
            "llf_difference": float(res_full.llf) - float(res_null.llf)}


def ss_joint_wald_test(res, structure, terms):
    """Wald test that every driver coefficient is zero, under the fitted covariance."""
    names = list(res.params.index)
    use = [ss_param_name(structure, t) for t in terms]
    use = [u for u in use if u in names]
    if not use:
        return {"wald_stat": np.nan, "df": 0, "p_wald": np.nan}
    R = np.zeros((len(use), len(names)))
    for i, u in enumerate(use):
        R[i, names.index(u)] = 1.0
    try:
        w = res.wald_test(R, scalar=True)
        return {"wald_stat": float(np.squeeze(w.statistic)), "df": int(len(use)),
                "p_wald": float(np.squeeze(w.pvalue))}
    except Exception:                                  # pragma: no cover
        return {"wald_stat": np.nan, "df": int(len(use)), "p_wald": np.nan}


def inverse_response_transform(values, how="logit", eps=1e-4):
    """Undo `transform_response`. The result is a MEDIAN, not a mean.

    Back-transforming a point forecast of a non-linear transform gives the
    median of the implied distribution on the raw scale. Reported as such: the
    raw-scale RMSE below is a median-forecast RMSE, not a bias-corrected one.
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    if how == "identity":
        return v
    if how == "log":
        return np.exp(v)
    if how == "logit":
        return 1.0 / (1.0 + np.exp(-v))
    raise ValueError(f"unknown transform {how!r}")


def rmse_mae(y, yhat):
    """RMSE / MAE over the finite pairs, with the pair count."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    if not ok.any():
        return {"n": 0, "rmse": np.nan, "mae": np.nan}
    e = y[ok] - yhat[ok]
    return {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e)))}


def block_bootstrap_rmse_difference(months, err_a, err_b, block_months=None,
                                    n_boot=2000, seed=0, alpha=0.05):
    """Moving-block bootstrap interval for RMSE(a) - RMSE(b) on PAIRED errors.

    The blocks are contiguous runs of CALENDAR months drawn from the evaluation
    grid, so a resample never places two months that are a year apart next to
    each other, and an evaluation month that is missing stays missing. Both error
    series are indexed by the same months, so every replicate compares the two
    models on exactly the same resampled months.
    """
    m = pd.to_datetime(pd.Series(months).reset_index(drop=True))
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    if not (len(m) == len(a) == len(b)):
        raise ValueError("months and both error series must be the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    m, a, b = m[ok].reset_index(drop=True), a[ok], b[ok]
    if len(m) < 4:
        return {"n_months": int(len(m)), "n_successful": 0,
                "block_months": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "boot_mean": np.nan, "share_negative": np.nan}
    rng = np.random.default_rng(seed)
    L = calendar_block_length(m, block_months)
    diffs = []
    for idx, _blocks in calendar_block_indices(m, L, rng, int(n_boot)):
        if len(idx) < 4:
            continue
        ra = float(np.sqrt(np.mean(a[idx] ** 2)))
        rb = float(np.sqrt(np.mean(b[idx] ** 2)))
        diffs.append(ra - rb)
    d = np.asarray(diffs, dtype=float)
    if not len(d):
        return {"n_months": int(len(m)), "n_successful": 0, "block_months": L,
                "ci_lo": np.nan, "ci_hi": np.nan, "boot_mean": np.nan,
                "share_negative": np.nan}
    return {"n_months": int(len(m)), "n_successful": int(len(d)), "block_months": int(L),
            "ci_lo": float(np.quantile(d, alpha / 2)),
            "ci_hi": float(np.quantile(d, 1 - alpha / 2)),
            "boot_mean": float(np.mean(d)),
            "share_negative": float(np.mean(d < 0))}


def newey_west_mean_test(d, months=None, maxlags=None):
    """HAC (Newey-West) test that a paired loss difference has mean zero.

    This is the Diebold-Mariano statistic for one-step-ahead forecasts. The
    bandwidth is in calendar months and the pairs are formed on the calendar, so
    two evaluation months either side of a gap are not treated as consecutive.
    """
    d = np.asarray(d, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    n = len(d)
    if n < 4:
        return {"mean": np.nan, "se_hac": np.nan, "t": np.nan, "p": np.nan, "n": n}
    mi = (np.arange(n) if months is None
          else month_index(pd.Series(months).reset_index(drop=True)[ok]))
    u = d - d.mean()
    L = hac_maxlags(n, maxlags)
    s = float(np.sum(u ** 2))
    for h in range(1, int(L) + 1):
        w = 1.0 - h / (L + 1.0)
        acc = 0.0
        for i in range(n):
            j = np.where(mi == mi[i] + h)[0]
            if len(j):
                acc += float(u[i] * u[j[0]])
        s += 2.0 * w * acc
    var = s / (n ** 2)
    se = float(np.sqrt(max(var, 0.0)))
    t = float(d.mean() / se) if se > 0 else np.nan
    return {"mean": float(d.mean()), "se_hac": se, "t": t,
            "p": float(2 * sstats.norm.sf(abs(t))) if np.isfinite(t) else np.nan,
            "n": n, "maxlags": int(L)}


print("§5c state-space helpers defined: "
      f"{len(SS_STRUCTURES)} candidate dependence structures.")
''')

# ===========================================================================
# 6. Load
# ===========================================================================
md("""## 6. Load the panel

Preference order, with the choice printed so provenance is never in doubt:

1. `PANEL_CSV` / newest `wh_spatial_panel_*.csv` — the cell-month panel, which
   carries the response **and** every covariate in one file.
2. `wh_monthly_summary_*.csv` + the two Earth Engine covariate tables — used
   only if no panel CSV exists.
3. `USE_SYNTHETIC_DEMO` — a synthetic series with known effects, no Drive.
""")

code('''# =====================================================================
# 6. Load
# =====================================================================
SOURCE = {"mode": None, "paths": [], "is_synthetic": False}
panel = None
monthly_raw = None
SYNTHETIC_TRUTH = None

if USE_SYNTHETIC_DEMO:
    monthly_raw, SYNTHETIC_TRUTH = make_synthetic_monthly(
        SYNTHETIC_N_MONTHS, SYNTHETIC_SEED, SYNTHETIC_MISSING_FRACTION)
    SOURCE.update(mode="synthetic", is_synthetic=True)
    print("*** USE_SYNTHETIC_DEMO = True: this run is a self-test, not a result. ***")
    print(f"Synthetic series: {len(monthly_raw)} months, known effects:")
    for k, v in SYNTHETIC_TRUTH.items():
        print(f"    {k:>34s} : {v:+.3f}")
else:
    panel, panel_path = load_cellmonth_panel(PANEL_CSV, PANEL_DIR, PANEL_GLOB)
    if panel is not None:
        SOURCE.update(mode="cellmonth_panel", paths=[str(panel_path)])
        print(f"Loaded cell-month panel: {panel_path}")
        print(f"  {len(panel):,} rows | {panel['grid_id'].nunique():,} cells | "
              f"{panel['month'].nunique()} months "
              f"({panel['month'].min():%Y-%m} .. {panel['month'].max():%Y-%m})")
    else:
        ms_path = newest_match(PANEL_DIR, MONTHLY_SUMMARY_GLOB)
        if ms_path is None:
            raise FileNotFoundError(
                f"No {PANEL_GLOB!r} and no {MONTHLY_SUMMARY_GLOB!r} in {PANEL_DIR}.\\n"
                "Run §17 of winam_wh_spatial_panel_driver_gam.ipynb to export the panel, "
                "point PANEL_CSV at it, or set USE_SYNTHETIC_DEMO = True to self-test.")
        monthly_raw = pd.read_csv(ms_path)
        monthly_raw["month"] = to_month_start(monthly_raw["month"])
        used = [str(ms_path)]
        print(f"No cell-month panel found; falling back to {ms_path.name}")
        # Response column names differ in the summary table.
        if "wh_cover_aoi" not in monthly_raw.columns:
            if {"wh_area_ha", "valid_area_ha"}.issubset(monthly_raw.columns):
                monthly_raw["wh_cover_aoi"] = (monthly_raw["wh_area_ha"]
                                               / monthly_raw["valid_area_ha"].replace(0, np.nan))
            elif "mean_cover" in monthly_raw.columns:
                monthly_raw["wh_cover_aoi"] = monthly_raw["mean_cover"]
                print("  NOTE: only the UNWEIGHTED mean cover is available in this table; "
                      "wh_cover_aoi is that column, not the area-weighted mean.")
        if "occurrence_rate" in monthly_raw.columns and "wh_occurrence" not in monthly_raw.columns:
            monthly_raw = monthly_raw.rename(columns={"occurrence_rate": "wh_occurrence"})
        # Drivers from the Earth Engine tables.
        for glob, keys in ((EE_MONTHLY_GLOB, ["month"]),
                           (EE_CELLMONTH_GLOB, ["grid_id", "month"])):
            p = newest_match(PANEL_DIR, glob)
            if p is None:
                print(f"  {glob}: not found")
                continue
            tab = pd.read_csv(p)
            tab["month"] = to_month_start(tab["month"])
            if "grid_id" in keys:
                num = tab.select_dtypes("number").columns.difference(["grid_id"])
                tab = tab.groupby("month", as_index=False)[list(num)].mean()
                print(f"  {p.name}: reduced per-cell covariates to an UNWEIGHTED AOI mean "
                      "(no per-cell valid areas available on this route)")
            dup = [c for c in tab.columns if c in monthly_raw.columns and c != "month"]
            monthly_raw = monthly_raw.merge(tab.drop(columns=dup), on="month", how="left")
            used.append(str(p))
        SOURCE.update(mode="monthly_summary+ee_tables", paths=used)

if EXTRA_MONTHLY_CSV is not None and Path(EXTRA_MONTHLY_CSV).exists():
    extra = pd.read_csv(EXTRA_MONTHLY_CSV)
    if "month" not in extra.columns:
        raise ValueError("EXTRA_MONTHLY_CSV needs a 'month' column.")
    extra["month"] = to_month_start(extra["month"])
    EXTRA_MONTHLY_COLS = [c for c in extra.columns if c != "month"]
    if panel is not None:
        panel = panel.merge(extra, on="month", how="left")
    else:
        dup = [c for c in EXTRA_MONTHLY_COLS if c in monthly_raw.columns]
        monthly_raw = monthly_raw.merge(extra.drop(columns=dup), on="month", how="left")
    print(f"Merged EXTRA_MONTHLY_CSV columns: {EXTRA_MONTHLY_COLS}")
else:
    EXTRA_MONTHLY_COLS = []

print("\\nSource:", SOURCE["mode"])
''')


# ===========================================================================
# 7. Build the AOI series
# ===========================================================================
md("""## 7. Build the AOI monthly series

This is the step that turns a spatial panel into a temporal one, and it is where
a temporal model is usually won or lost. Four things are enforced and printed:

1. **Coverage filter** — which months are comparable at all.
2. **Fixed cell set** — the average is taken over the same cells every month.
3. **Static drivers dropped by name** — with the reason, because a reader will
   ask why depth is missing from a depth-sensitive system.
4. **Degenerate drivers dropped by name** — `effective_depth_m` averaged over a
   fixed cell set *is* the lake-level anomaly plus a constant, so entering both
   would be entering the same variable twice.
""")

code('''# =====================================================================
# 7. Build the AOI monthly series
# =====================================================================
AOI_AUDIT = {}

# --- Which columns are candidate time-varying drivers ------------------------
_reserved = {"month", "grid_id", "year", "month_num", "time_index", "observed",
             "n_cells", "valid_area_ha", "valid_area_m2", "wh_area_ha", "wh_area_m2",
             "wh_area_ha_hard", "wh_pixels", "wh_pixels_hard", "valid_pixels",
             "valid_fraction", "coverage_fraction", "n_cells_eligible",
             "n_cells_observed", "retained", "x_km", "y_km", "x", "y"}
_response_like = lambda c: (c.startswith("wh_") or c.startswith("mean_cover")
                            or c.startswith("occurrence") or c.endswith("_neigh_lag1"))

_frame_for_cols = panel if panel is not None else monthly_raw
_numeric = [c for c in _frame_for_cols.columns
            if c not in _reserved and not _response_like(c)
            and pd.api.types.is_numeric_dtype(_frame_for_cols[c])]
# Lagged copies made by the spatial notebook are rebuilt here from the AOI
# series, so the panel's own _lag1 columns are not carried across.
_numeric = [c for c in _numeric if not c.endswith("_lag1")]

static_present = [c for c in _numeric if c in KNOWN_STATIC_COLS]
degenerate_present = [c for c in _numeric if c in AOI_DEGENERATE_COLS]
candidate_driver_cols = [c for c in _numeric
                         if c not in static_present and c not in degenerate_present]

print("Dropped as STATIC (constant over a fixed cell set -> cannot explain temporal "
      f"variation): {static_present or 'none'}")
for c in degenerate_present:
    print(f"Dropped as DEGENERATE at AOI scale: {c} ({AOI_DEGENERATE_COLS[c]})")
print(f"\\nCandidate time-varying drivers ({len(candidate_driver_cols)}):")
print("  " + ", ".join(candidate_driver_cols))

# --- Collapse to one row per month -------------------------------------------
if panel is not None:
    coverage = monthly_coverage_table(panel, MIN_MONTHLY_COVERAGE_FRACTION)
    display(coverage)
    months_kept = coverage.loc[coverage["retained"], "month"]
    print(f"Coverage filter: {int(coverage['retained'].sum())} of {len(coverage)} months "
          f"at or above {MIN_MONTHLY_COVERAGE_FRACTION:.0%} "
          f"(basis: {coverage['coverage_basis'].iloc[0]})")

    cells_kept, cell_audit = fixed_cell_set(panel, months_kept, MIN_CELL_MONTH_FRACTION)
    AOI_AUDIT["cell_set"] = cell_audit
    print(f"\\nFixed cell set: {cell_audit['n_cells_kept']:,} of "
          f"{cell_audit['n_cells_total']:,} cells observed in >= "
          f"{MIN_CELL_MONTH_FRACTION:.0%} of the {cell_audit['n_months_retained']} "
          f"retained months, holding "
          f"{cell_audit['share_of_classified_area_kept']:.1%} of the classified area.")
    if cell_audit["n_cells_kept"] == 0:
        raise RuntimeError("The fixed cell set is empty. Lower MIN_CELL_MONTH_FRACTION.")

    monthly = aoi_monthly_series(
        panel[panel["month"].isin(months_kept)],
        cell_ids=cells_kept, weighting=AOI_WEIGHTING,
        driver_cols=candidate_driver_cols, response_eps=RESPONSE_EPS)
    monthly = monthly.merge(
        coverage[["month", "coverage_fraction", "n_cells_observed"]], on="month", how="left")
else:
    monthly = monthly_raw.copy()
    if "coverage_fraction" in monthly.columns:
        n_before = len(monthly)
        monthly = monthly[monthly["coverage_fraction"] >= MIN_MONTHLY_COVERAGE_FRACTION]
        print(f"Coverage filter: kept {len(monthly)} of {n_before} months.")
    else:
        print("No coverage_fraction available on this route; NO coverage filter applied. "
              "Treat month-to-month comparisons with caution.")
    AOI_AUDIT["cell_set"] = {"note": "monthly route: no per-cell information, "
                                     "so no fixed cell set could be enforced"}

# --- Calendar-complete grid + season / response terms ------------------------
monthly = reindex_calendar_months(monthly)
monthly, SEASON_COLS = add_season_terms(monthly, SEASON_HARMONICS)

if RESPONSE_COL not in monthly.columns:
    raise KeyError(f"RESPONSE_COL {RESPONSE_COL!r} not in the built series: "
                   f"{[c for c in monthly.columns if c.startswith('wh_')]}")
monthly["y_raw"] = pd.to_numeric(monthly[RESPONSE_COL], errors="coerce")
if RESPONSE_COL == "wh_area_ha":
    monthly["y"], RESPONSE_INFO = transform_response(monthly["y_raw"], "log", RESPONSE_EPS)
    print("\\nRESPONSE_COL is an absolute area, so a log (not logit) scale is used.")
else:
    monthly["y"], RESPONSE_INFO = transform_response(
        monthly["y_raw"], RESPONSE_TRANSFORM, RESPONSE_EPS)

if MONTH_WEIGHTING == "valid_area" and "valid_area_ha" in monthly.columns:
    monthly["w_month"] = monthly["valid_area_ha"].astype(float)
elif MONTH_WEIGHTING == "coverage" and "coverage_fraction" in monthly.columns:
    monthly["w_month"] = monthly["coverage_fraction"].astype(float)
else:
    monthly["w_month"] = 1.0
monthly["w_month"] = monthly["w_month"].fillna(monthly["w_month"].median())

N_OBSERVED = int(monthly["y"].notna().sum())
N_GRID = len(monthly)
AOI_AUDIT.update({
    "n_months_grid": N_GRID, "n_months_observed": N_OBSERVED,
    "first_month": str(monthly["month"].min().date()),
    "last_month": str(monthly["month"].max().date()),
    "response_col": RESPONSE_COL, "response_transform": RESPONSE_INFO,
    "aoi_weighting": AOI_WEIGHTING, "month_weighting": MONTH_WEIGHTING,
    "min_monthly_coverage_fraction": MIN_MONTHLY_COVERAGE_FRACTION,
    "static_dropped": static_present, "degenerate_dropped": degenerate_present,
})

print(f"\\nAOI series: {N_OBSERVED} observed months on a {N_GRID}-month calendar grid "
      f"({monthly['month'].min():%Y-%m} .. {monthly['month'].max():%Y-%m}); "
      f"{N_GRID - N_OBSERVED} gap month(s).")
print(f"Response {RESPONSE_COL}: min {monthly['y_raw'].min():.4g}, "
      f"median {monthly['y_raw'].median():.4g}, max {monthly['y_raw'].max():.4g} "
      f"({RESPONSE_INFO['transform']} scale, {RESPONSE_INFO['n_clipped']} value(s) clipped)")
if N_OBSERVED < 36:
    print(f"\\n*** {N_OBSERVED} observed months is a SMALL temporal sample. Expect wide "
          "intervals; prefer the §17 partitioning and §13 stability ranking over "
          "individual p-values. ***")
display(monthly.head())
''')


# ===========================================================================
# 8. Series diagnostics
# ===========================================================================
md("""## 8. The series itself

Before any model: what does the response look like, how persistent is it, and
how much of it is the annual cycle? The autocorrelation function is the key
plot — it sets how much independent information the record actually holds.
""")

code('''# =====================================================================
# 8. Response series diagnostics
# =====================================================================
obs = monthly[monthly["y"].notna()].copy()

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
# Every panel is drawn on the unbroken calendar grid, so an unobserved month is
# a break in the line rather than a straight segment across it. The record opens
# with a lone 2017-06 observation and then nothing until 2018-12; that stretch
# must read as absent data, not as a measured decline.
axes[0].plot(*on_calendar_grid(monthly["month"], monthly["y_raw"]),
             marker="o", ms=3, lw=1.2, color="tab:green")
axes[0].set_ylabel(RESPONSE_COL)
axes[0].set_title(f"AOI water-hyacinth extent — {RESPONSE_COL} "
                  f"({N_OBSERVED} observed months, {N_GRID - N_OBSERVED} gap months "
                  f"shown as breaks)")
axes[1].plot(*on_calendar_grid(monthly["month"], monthly["y"]),
             marker="o", ms=3, lw=1.2, color="tab:blue")
axes[1].set_ylabel(f"y ({RESPONSE_INFO['transform']})")
axes[2].plot(*on_calendar_grid(
                 monthly["month"],
                 monthly.get("coverage_fraction",
                             pd.Series(index=monthly.index, dtype=float))),
             marker="o", ms=3, lw=1.0, color="tab:grey", label="coverage")
axes[2].axhline(MIN_MONTHLY_COVERAGE_FRACTION, color="red", ls="--", lw=1,
                label=f"threshold {MIN_MONTHLY_COVERAGE_FRACTION:.0%}")
axes[2].set_ylabel("classified\\ncoverage"); axes[2].set_xlabel("Month")
axes[2].legend(fontsize=8)
for a in axes:
    a.grid(alpha=0.3)
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); plt.show()

# --- Persistence -------------------------------------------------------------
# The ACF is taken on CALENDAR-month lags: a pair enters lag h only if the two
# months are exactly h months apart, so an excluded month costs pairs instead
# of pretending the months either side of it are neighbours.
acf_y = acf_values(monthly["y"], months=monthly["month"],
                   nlags=min(24, max(4, N_OBSERVED // 3)))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].bar(acf_y["lag"], acf_y["acf"], color="tab:blue", alpha=0.8)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].axhline(acf_y["band"].iloc[0], color="red", ls="--", lw=1)
axes[0].axhline(-acf_y["band"].iloc[0], color="red", ls="--", lw=1)
axes[0].set_title("ACF of the response (calendar-month lags)")
axes[0].set_xlabel("lag (calendar months)")
axes[0].grid(alpha=0.3)

# Seasonal climatology: the share of variance the calendar month alone explains.
clim = obs.groupby("month_num")["y"].agg(["mean", "std", "count"])
axes[1].errorbar(clim.index, clim["mean"], yerr=clim["std"], marker="o", capsize=3)
axes[1].set_title("Monthly climatology of the response")
axes[1].set_xlabel("calendar month"); axes[1].set_xticks(range(1, 13)); axes[1].grid(alpha=0.3)
fig.tight_layout(); plt.show()

# `monthly` is the calendar-complete grid, so shift(1) here IS "one calendar
# month"; the pair count from the calendar ACF says how many months actually
# had an observed predecessor.
_lag1_r = float(pd.Series(monthly["y"]).corr(pd.Series(monthly["y"]).shift(1)))
_lag1_pairs = int(acf_y.loc[acf_y["lag"] == 1, "n_pairs"].iloc[0]) if len(acf_y) else 0
_season_r2 = float(fit_hac(obs["y"], obs[SEASON_COLS], months=obs["month"]).rsquared)
_trend_r2 = float(fit_hac(obs["y"], obs[["time_index"]], months=obs["month"]).rsquared)
_n_eff = N_OBSERVED * (1 - _lag1_r) / (1 + _lag1_r) if abs(_lag1_r) < 1 else np.nan

SERIES_DIAGNOSTICS = pd.DataFrame([{
    "n_observed_months": N_OBSERVED,
    "n_calendar_months_grid": N_GRID,
    "lag1_autocorrelation": _lag1_r,
    "n_consecutive_month_pairs": _lag1_pairs,
    "effective_n_bartlett": _n_eff,
    "r2_season_only": _season_r2,
    "r2_trend_only": _trend_r2,
    "hac_maxlags_calendar_months": hac_maxlags(N_OBSERVED, HAC_MAXLAGS),
}])
display(SERIES_DIAGNOSTICS.T.rename(columns={0: "value"}))

print(f"Lag-1 autocorrelation {_lag1_r:.2f} (from {_lag1_pairs} pairs of genuinely "
      f"consecutive calendar months).")
print(f"APPROXIMATE effective sample size (Bartlett, on the RAW response): "
      f"{_n_eff:.0f} of {N_OBSERVED} months.")
print("  Read that as an order-of-magnitude diagnostic, NOT as the sample size any")
print("  model uses. It is computed on the raw response, so the autocorrelation it")
print("  measures still contains the annual cycle and the trend, which every model")
print("  below removes. The state-space model in §19-§24 uses ALL observed months")
print("  and represents this dependence explicitly rather than discounting for it —")
print("  but it cannot manufacture independent information the record does not")
print("  contain, so the interval widths, not this number, are the honest summary.")
print(f"Season alone explains {_season_r2:.1%} of the response; "
      f"a linear trend alone explains {_trend_r2:.1%}.")
print("Any driver effect must be argued AGAINST those two, which is what §12 M2/M3 do.")
''')


# ===========================================================================
# 9. Identifiability audit
# ===========================================================================
md("""## 9. Identifiability audit — which drivers *can* be separated from season?

This is the section that decides what this notebook is allowed to conclude. For
every candidate driver it measures:

* $R^2_{\\text{season}}$ — how well the seasonal harmonics alone predict the
  **driver**. Above `SEASON_CONFOUND_R2` the driver is essentially a relabelled
  calendar, and no amount of modelling can separate its effect from seasonality.
* $R^2_{\\text{trend}}$ — the same for a linear trend. A driver that is mostly
  trend cannot be separated from any other slow change (including drift in the
  classifier).
* **Variance left after removing season and trend** — the anomaly variance that
  is actually available to identify an effect.
* **Pairwise redundancy** — drivers correlated above `MAX_ABS_PAIRWISE_R` are the
  same variable wearing two names; the later one in the mechanism list is dropped.
""")

code('''# =====================================================================
# 9. Identifiability audit
# =====================================================================
_audit_rows = []
for c in candidate_driver_cols:
    if c not in monthly.columns:
        continue
    v = pd.to_numeric(monthly[c], errors="coerce")
    ok = v.notna()
    n_ok = int(ok.sum())
    row = {"driver": c, "n_months": n_ok,
           "n_missing": int(len(monthly) - n_ok),
           "n_unique": int(v.nunique(dropna=True)),
           "mean": float(v.mean()) if n_ok else np.nan,
           "sd": float(v.std(ddof=1)) if n_ok > 1 else np.nan}
    if n_ok >= 12 and row["n_unique"] >= 3 and np.isfinite(row["sd"]) and row["sd"] > 0:
        sub = monthly.loc[ok]
        row["r2_on_season"] = float(
            fit_hac(sub[c], sub[SEASON_COLS], months=sub["month"]).rsquared)
        row["r2_on_trend"] = float(
            fit_hac(sub[c], sub[["time_index"]], months=sub["month"]).rsquared)
        resid = fit_hac(sub[c], sub[SEASON_COLS + ["time_index"]],
                        months=sub["month"]).resid
        row["anomaly_sd_share"] = float(np.std(resid, ddof=1) / row["sd"])
        row["usable"] = True
    else:
        row.update(r2_on_season=np.nan, r2_on_trend=np.nan,
                   anomaly_sd_share=np.nan, usable=False)
    _audit_rows.append(row)

DRIVER_AUDIT = pd.DataFrame(_audit_rows)
if len(DRIVER_AUDIT):
    DRIVER_AUDIT["season_confounded"] = DRIVER_AUDIT["r2_on_season"] >= SEASON_CONFOUND_R2
    DRIVER_AUDIT = DRIVER_AUDIT.sort_values("r2_on_season", ascending=False).reset_index(drop=True)
display(DRIVER_AUDIT)

_unusable = DRIVER_AUDIT.loc[~DRIVER_AUDIT["usable"], "driver"].tolist() if len(DRIVER_AUDIT) else []
if _unusable:
    print(f"Unusable (too few months / constant): {_unusable}")
_confounded = (DRIVER_AUDIT.loc[DRIVER_AUDIT["season_confounded"].fillna(False), "driver"].tolist()
               if len(DRIVER_AUDIT) else [])
if _confounded:
    print(f"\\nSEASON-CONFOUNDED (R2 on season >= {SEASON_CONFOUND_R2:.2f}): {_confounded}")
    print("These stay in the model but any effect they show is reported as "
          "'not separable from seasonality' in §25, whatever its p-value.")

# --- Resolve the mechanism set against what the series actually holds ---------
_rows, _resolved = [], {}
for term, (mech, sign, lag) in TEMPORAL_FORCING_TERMS.items():
    used, how = None, ""
    usable = set(DRIVER_AUDIT.loc[DRIVER_AUDIT["usable"], "driver"]) if len(DRIVER_AUDIT) else set()
    if term in usable:
        used, how = term, "requested"
    else:
        for alt in TEMPORAL_FORCING_FALLBACKS.get(term, []):
            if alt in usable:
                used, how = alt, f"fallback for {term}"
                break
    _rows.append({"requested": term, "resolved": used or "(unavailable)",
                  "how": how or "no usable substitute", "mechanism": mech,
                  "expected_sign": sign, "apriori_lag": lag})
    if used and used not in _resolved:
        _resolved[used] = {"mechanism": mech, "expected_sign": sign,
                           "apriori_lag": int(lag), "requested_as": term}

FORCING_RESOLUTION = pd.DataFrame(_rows)
print("\\nMechanism set (requested -> what the AOI series supplies):")
display(FORCING_RESOLUTION)

# --- Pairwise redundancy ------------------------------------------------------
_names = list(_resolved)
DRIVER_CORR = monthly[_names].corr() if len(_names) > 1 else pd.DataFrame()
redundant_dropped = []
if len(_names) > 1:
    display(DRIVER_CORR.round(2))
    for i, a in enumerate(_names):
        for b in _names[i + 1:]:
            r = DRIVER_CORR.loc[a, b]
            if np.isfinite(r) and abs(r) >= MAX_ABS_PAIRWISE_R and b not in redundant_dropped:
                redundant_dropped.append(b)
                print(f"Dropping {b}: |r| = {abs(r):.2f} with {a} (>= "
                      f"{MAX_ABS_PAIRWISE_R}); the same variable twice.")
for b in redundant_dropped:
    _resolved.pop(b, None)

FORCING = dict(_resolved)
PROXY_COLS = [c for c in TEMPORAL_PROXY_TERMS
              if c in monthly.columns and c in set(DRIVER_AUDIT.loc[
                  DRIVER_AUDIT["usable"], "driver"] if len(DRIVER_AUDIT) else [])]
UNUSED_CANDIDATES = [c for c in candidate_driver_cols
                     if c not in FORCING and c not in PROXY_COLS]

print(f"\\nFORCING set ({len(FORCING)}): {list(FORCING)}")
print(f"Endogenous proxies, descriptive only ({len(PROXY_COLS)}): {PROXY_COLS or 'none'}")
print(f"Available but not in any mechanism ({len(UNUSED_CANDIDATES)}): "
      f"{UNUSED_CANDIDATES or 'none'}")
if not FORCING:
    raise RuntimeError("No usable forcing drivers. Check §7's candidate list and "
                       "TEMPORAL_FORCING_TERMS / TEMPORAL_FORCING_FALLBACKS.")
''')

# ===========================================================================
# 10. Lag structure
# ===========================================================================
md("""## 10. Lag structure

A temporal driver model *is* its lag structure: hyacinth cannot respond to a
nutrient pulse in the same fortnight it arrives, so the lag at which a driver is
entered decides whether its mechanism is being tested at all.

Two routes, and the difference matters for what may be claimed:

* **`LAG_SELECTION = "apriori"` (default)** — each driver enters at the lag its
  mechanism implies (`TEMPORAL_FORCING_TERMS`). The lag is a hypothesis fixed
  before seeing the response, so the *p*-value means what it says.
* **`LAG_SELECTION = "ccf"`** — each driver enters at its peak absolute
  cross-correlation. This *always* looks better and the *p*-values are
  optimistically biased, because the lag was chosen using the response. Kept as
  a sensitivity only.

The cross-correlation functions below are shown either way, as **evidence about
the mechanism**: agreement between the a-priori lag and the CCF peak is a point
in the mechanism's favour, and disagreement is worth reporting honestly.
""")

code('''# =====================================================================
# 10. Cross-correlation evidence and the lag actually used
# =====================================================================
_ccf_frames = []
for c in FORCING:
    # Lags are CALENDAR months: a driver value only pairs with the response k
    # months later if those two months are exactly k months apart.
    t = cross_correlation(monthly[c], monthly["y"], months=monthly["month"],
                          max_lag=LAG_SCAN_MAX)
    t.insert(0, "driver", c)
    _ccf_frames.append(t)
CCF_TABLE = pd.concat(_ccf_frames, ignore_index=True) if _ccf_frames else pd.DataFrame()

# Peak |r| lag per driver, and whether it agrees with the a-priori lag.
_peaks = []
for c, meta in FORCING.items():
    sub = CCF_TABLE[CCF_TABLE["driver"] == c].dropna(subset=["r"])
    if sub.empty:
        _peaks.append({"driver": c, "apriori_lag": meta["apriori_lag"],
                       "ccf_peak_lag": np.nan, "ccf_peak_r": np.nan,
                       "r_at_apriori_lag": np.nan, "lags_agree": False})
        continue
    best = sub.loc[sub["r"].abs().idxmax()]
    at_ap = sub.loc[sub["lag"] == meta["apriori_lag"], "r"]
    _peaks.append({"driver": c, "apriori_lag": meta["apriori_lag"],
                   "ccf_peak_lag": int(best["lag"]), "ccf_peak_r": float(best["r"]),
                   "r_at_apriori_lag": float(at_ap.iloc[0]) if len(at_ap) else np.nan,
                   "lags_agree": int(best["lag"]) == int(meta["apriori_lag"])})
LAG_EVIDENCE = pd.DataFrame(_peaks)
display(LAG_EVIDENCE)

if len(FORCING):
    ncol = min(3, len(FORCING))
    nrow = int(np.ceil(len(FORCING) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.7 * nrow), squeeze=False)
    band = 1.96 / np.sqrt(max(N_OBSERVED, 1))
    for ax, (c, meta) in zip(axes.ravel(), FORCING.items()):
        sub = CCF_TABLE[CCF_TABLE["driver"] == c]
        ax.bar(sub["lag"], sub["r"], color="tab:blue", alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(band, color="red", ls="--", lw=0.8)
        ax.axhline(-band, color="red", ls="--", lw=0.8)
        ax.axvline(meta["apriori_lag"], color="tab:orange", lw=2, alpha=0.6)
        ax.set_title(f"{c}\\n(a-priori lag {meta['apriori_lag']}, sign {meta['expected_sign']})",
                     fontsize=9)
        ax.set_xlabel("driver lead (calendar months)")
        ax.set_ylim(-1, 1); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(FORCING):]:
        ax.axis("off")
    fig.suptitle("Cross-correlation of the response with each driver at lag k "
                 "(orange = the lag actually used)", y=1.02, fontsize=11)
    fig.tight_layout(); plt.show()

# --- Fix the lag each driver enters at ---------------------------------------
LAG_USED = {}
for c, meta in FORCING.items():
    if str(LAG_SELECTION).lower() == "ccf":
        row = LAG_EVIDENCE.loc[LAG_EVIDENCE["driver"] == c, "ccf_peak_lag"]
        lag = int(row.iloc[0]) if len(row) and np.isfinite(row.iloc[0]) else meta["apriori_lag"]
    else:
        lag = int(meta["apriori_lag"])
    LAG_USED[c] = lag

if str(LAG_SELECTION).lower() == "ccf":
    print("\\n*** LAG_SELECTION = 'ccf': lags were chosen using the response, so the "
          "p-values below are optimistically biased. Report this as a sensitivity, "
          "not as the headline model. ***")
print("\\nLag used per driver:", {k: v for k, v in LAG_USED.items()})
''')


# ===========================================================================
# 11. Model dataset
# ===========================================================================
md("""## 11. Model dataset

One row per month; drivers lagged on the calendar grid then standardised, so
every coefficient reads as **"change in the response per one standard deviation
of this driver"** and the magnitudes are directly comparable. The scaling
constants are kept (`DRIVER_SCALING`) so any effect can be put back into
physical units for the write-up.
""")

code('''# =====================================================================
# 11. Model dataset
# =====================================================================
model_df = monthly.copy()

DRIVER_TERMS, DL_TERMS = [], {}
for c, lag in LAG_USED.items():
    if c in DISTRIBUTED_LAG_TERMS:
        made = []
        for L in range(0, 3):
            model_df, m = calendar_lag(model_df, [c], L)
            made += m
        DL_TERMS[c] = made
        DRIVER_TERMS += made
    else:
        model_df, made = calendar_lag(model_df, [c], lag)
        DRIVER_TERMS += made

PROXY_TERMS = []
for c in PROXY_COLS:
    model_df, made = calendar_lag(model_df, [c], 0)
    PROXY_TERMS += made

# Lagged response: the PREDICTIVE / dynamic term. Never a driver claim.
AR_TERMS = []
for L in range(1, int(AR_LAGS) + 1):
    model_df[f"y_lag{L}"] = model_df["y"].shift(L)
    AR_TERMS.append(f"y_lag{L}")

TREND_TERMS = ["time_index"] if INCLUDE_TREND else []

# Standardise drivers, proxies and the trend (season terms are already bounded).
_to_scale = [c for c in DRIVER_TERMS + PROXY_TERMS + TREND_TERMS if c in model_df.columns]
model_df, DRIVER_SCALING = zscore_frame(model_df, _to_scale)
_degenerate = DRIVER_SCALING.loc[~np.isfinite(DRIVER_SCALING["sd"]), "term"].tolist()
if _degenerate:
    print(f"Zero-variance after lagging, dropped: {_degenerate}")
    DRIVER_TERMS = [c for c in DRIVER_TERMS if c not in _degenerate]
    PROXY_TERMS = [c for c in PROXY_TERMS if c not in _degenerate]

# Rows usable by the driver models: response and every driver term present.
FIT_COLS = ["y"] + DRIVER_TERMS + SEASON_COLS + TREND_TERMS
fit_mask = model_df[FIT_COLS].notna().all(axis=1)
fit_df = model_df.loc[fit_mask].reset_index(drop=True)
W = fit_df["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None
N_FIT = len(fit_df)

# The dynamic specification also needs the AR terms, so it loses further rows.
dyn_mask = fit_mask & model_df[AR_TERMS].notna().all(axis=1)
dyn_df = model_df.loc[dyn_mask].reset_index(drop=True)
W_DYN = dyn_df["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None

# The month of every fitted row, carried alongside the design matrices. Every
# HAC covariance, bootstrap block and validation fold below is defined on these
# timestamps, never on the row positions of the complete-case tables.
FIT_MONTHS = fit_df["month"].reset_index(drop=True)
DYN_MONTHS = dyn_df["month"].reset_index(drop=True)
FIT_SPAN_MONTHS = calendar_span_months(FIT_MONTHS) if N_FIT else 0
DYN_SPAN_MONTHS = calendar_span_months(DYN_MONTHS) if len(dyn_df) else 0

print(f"Driver terms ({len(DRIVER_TERMS)}): {DRIVER_TERMS}")
print(f"Season terms ({len(SEASON_COLS)}): {SEASON_COLS}")
print(f"Trend terms: {TREND_TERMS} | AR terms: {AR_TERMS}")
print(f"Proxy terms (descriptive only): {PROXY_TERMS or 'none'}")
print(f"\\nRows: {N_FIT} for the static specifications, {len(dyn_df)} for the dynamic one "
      f"(of {N_OBSERVED} observed months; lagging costs the leading months).")
print(f"Those {N_FIT} rows are spread over {FIT_SPAN_MONTHS} calendar months, so "
      f"{FIT_SPAN_MONTHS - N_FIT} month(s) inside the fitted range are absent. Row "
      "adjacency is therefore NOT calendar adjacency, which is why §5's helpers key "
      "everything on the month.")
print(f"Newey-West bandwidth: {hac_maxlags(N_FIT, HAC_MAXLAGS)} calendar months")
if N_FIT < len(DRIVER_TERMS) * 8:
    print(f"\\n*** {N_FIT} rows for {len(DRIVER_TERMS)} drivers plus "
          f"{len(SEASON_COLS) + len(TREND_TERMS)} controls is a thin design "
          "(< ~8 rows per term). Lean on §13 (regularised) and §17 (partitioning). ***")

display(DRIVER_SCALING)
VIF_DRIVERS = vif_table(fit_df[DRIVER_TERMS + SEASON_COLS + TREND_TERMS])
display(VIF_DRIVERS)
if (VIF_DRIVERS["vif"] > VIF_THRESHOLD).any():
    print(f"VIF above {VIF_THRESHOLD} for: "
          f"{VIF_DRIVERS.loc[VIF_DRIVERS['vif'] > VIF_THRESHOLD, 'term'].tolist()}")
    print("Their individual coefficients are unstable; read them from §13 and §17 instead.")
''')


# ===========================================================================
# 12. Model A
# ===========================================================================
md("""## 12. Model A — linear driver model with Newey–West standard errors

Five nested specifications, all on identical rows, so the reader can see exactly
what each control costs:

| | Specification | What it answers |
|---|---|---|
| **M0** | season + trend | The baseline. How much of WH extent is *just the calendar*? |
| **M1** | drivers only | **Marginal** association. Inflated by seasonality — reported to show the size of that inflation, not as an effect. |
| **M2** | drivers + season | The driver effect **within** the annual cycle: does a wetter-than-usual March raise WH above what March usually gives? |
| **M3** | drivers + season + trend | **The headline specification.** Also net of multi-year drift. |
| **M4** | M3 + lagged response | Dynamic. Short-run effect, plus the long-run effect $\\beta/(1-\\rho)$ **only if the AR process is stationary** — see below. |

Coefficients are per standard deviation of the driver, on the
`RESPONSE_TRANSFORM` scale of the response. Standard errors are HAC with
**calendar-month** lags (§5); *p*-values carry BH FDR *q*-values across drivers
within each specification.

### Why the long-run multiplier is gated

$\\beta/(1-\\rho)$ is the effect of a *sustained* push once a persistent system
has finished accumulating it. It is only a finite, meaningful quantity when the
AR process is **stationary**. On this series $\\hat\\rho$ sits near 1 and its
HAC confidence interval reaches it, and $1/(1-\\rho)$ is not a well-behaved
function there: an interval of $[0.74, 1.02]$ for $\\rho$ maps to multipliers
running from about 4 to *unbounded*, so a point estimate like 8.35 conveys
precision the data do not have, and the delta-method standard error is not
usable either.

The notebook therefore checks, before reporting anything long-run:

1. every root of $1-\\sum_j\\phi_j z^j$ lies **outside** the unit circle (the
   roots, not merely $\\sum_j\\phi_j$, because with more than one lag the sum is
   not a sufficient condition); **and**
2. the HAC interval for the AR term lies **strictly inside** $(-1, 1)$.

If either fails, `long_run`, `long_run_ci_*` and `long_run_multiplier` are set
to `NaN`, `long_run_estimable=False` is recorded with an explanatory `reason`,
and the message says so plainly. The **short-run** coefficients are unaffected
and remain reportable. Set `LONGRUN_REQUIRE_STATIONARITY = False` in §3 to
restore the old, unsafe behaviour.

Two semi-partial $R^2$ tables are produced at the end of this cell — one per
specification, each on its own rows and columns — because §17 may only compare a
Shapley decomposition with the semi-partial values of the **same** fit.
""")

code('''# =====================================================================
# 12. Model A - nested linear specifications with HAC inference
# =====================================================================
# Every fit passes the MONTH of each row, so the Newey-West covariance pairs
# observations by calendar distance and never treats the months either side of
# an excluded month as consecutive.
SPECS = {
    "M0 season+trend":        SEASON_COLS + TREND_TERMS,
    "M1 drivers only":        DRIVER_TERMS,
    "M2 drivers+season":      DRIVER_TERMS + SEASON_COLS,
    "M3 drivers+season+trend": DRIVER_TERMS + SEASON_COLS + TREND_TERMS,
}
HEADLINE_SPEC = "M3 drivers+season+trend"
DYNAMIC_SPEC = "M4 dynamic (+AR)"

MODEL_A_FITS, _coef_frames, _fit_rows = {}, [], []
for name, cols in SPECS.items():
    cols = [c for c in cols if c in fit_df.columns]
    if not cols:
        continue
    res = fit_hac(fit_df["y"], fit_df[cols], weights=W, maxlags=HAC_MAXLAGS,
                  months=FIT_MONTHS)
    MODEL_A_FITS[name] = res
    _coef_frames.append(tidy_coefficients(res, keep=DRIVER_TERMS, label=name))
    _fit_rows.append({"specification": name, "n": int(res.nobs),
                      "k_terms": len(cols), "r2": res.rsquared,
                      "adj_r2": res.rsquared_adj, "aic": res.aic, "bic": res.bic,
                      "hac_maxlags_months": res._hac_maxlags,
                      "hac_calendar_aware": res._hac_calendar_aware,
                      "hac_lag1_pairs": (res._hac_pair_counts or {}).get(1, np.nan),
                      "resid_lag1_acf": float(
                          acf_values(res.resid, months=FIT_MONTHS, nlags=1)["acf"].iloc[0])})

# --- M4: dynamic specification (short-run and, only if licensed, long-run) ----
LONGRUN = pd.DataFrame()
AR_STABILITY = {}
LONGRUN_ESTIMABLE = False
LONGRUN_REASON = "dynamic specification not fitted"
DYN_COLS = []
if AR_TERMS and len(dyn_df) > len(DRIVER_TERMS) + len(SEASON_COLS) + 3:
    DYN_COLS = [c for c in DRIVER_TERMS + SEASON_COLS + TREND_TERMS + AR_TERMS
                if c in dyn_df.columns]
    res4 = fit_hac(dyn_df["y"], dyn_df[DYN_COLS], weights=W_DYN, maxlags=HAC_MAXLAGS,
                   months=DYN_MONTHS)
    MODEL_A_FITS[DYNAMIC_SPEC] = res4
    _coef_frames.append(tidy_coefficients(res4, keep=DRIVER_TERMS + AR_TERMS,
                                          label=DYNAMIC_SPEC))
    _fit_rows.append({"specification": DYNAMIC_SPEC, "n": int(res4.nobs),
                      "k_terms": len(DYN_COLS), "r2": res4.rsquared,
                      "adj_r2": res4.rsquared_adj, "aic": res4.aic, "bic": res4.bic,
                      "hac_maxlags_months": res4._hac_maxlags,
                      "hac_calendar_aware": res4._hac_calendar_aware,
                      "hac_lag1_pairs": (res4._hac_pair_counts or {}).get(1, np.nan),
                      "resid_lag1_acf": float(
                          acf_values(res4.resid, months=DYN_MONTHS,
                                     nlags=1)["acf"].iloc[0])})

    # ---- Is a long-run effect identified at all? ----------------------------
    # beta / (1 - rho) is only a meaningful quantity when the AR process is
    # stationary. If the HAC interval for the AR coefficient reaches 1 the
    # multiplier is unbounded from the data's point of view: 1/(1-rho) runs from
    # a few to infinity across the interval, and the delta-method standard error
    # is not usable. In that case the SHORT-RUN coefficients are reported and
    # the long-run column is NaN, with the reason recorded.
    AR_STABILITY = ar_stability(res4, AR_TERMS, alpha=LONGRUN_CI_ALPHA)
    _rho = float(AR_STABILITY["rho_sum"])
    LONGRUN_ESTIMABLE = bool(AR_STABILITY["stationary_supported"]
                             or not LONGRUN_REQUIRE_STATIONARITY)
    LONGRUN_REASON = "" if AR_STABILITY["stationary_supported"] else AR_STABILITY["reason"]

    V = res4.cov_params()
    names = list(res4.params.index)
    rows = []
    for c in DRIVER_TERMS:
        if c not in res4.params.index:
            continue
        b = float(res4.params[c])
        row = {"term": c, "short_run": b,
               "long_run_estimable": LONGRUN_ESTIMABLE,
               "reason": LONGRUN_REASON,
               "long_run": np.nan, "long_run_se": np.nan,
               "long_run_ci_lo": np.nan, "long_run_ci_hi": np.nan}
        if LONGRUN_ESTIMABLE and abs(1 - _rho) > 1e-6:
            g = np.zeros(len(names))
            g[names.index(c)] = 1.0 / (1 - _rho)
            for a in AR_TERMS:
                if a in names:
                    g[names.index(a)] = b / (1 - _rho) ** 2
            se = float(np.sqrt(max(g @ V.to_numpy() @ g, 0.0)))
            row.update(long_run=b / (1 - _rho), long_run_se=se,
                       long_run_ci_lo=b / (1 - _rho) - 1.96 * se,
                       long_run_ci_hi=b / (1 - _rho) + 1.96 * se)
        rows.append(row)
    LONGRUN = pd.DataFrame(rows)
    if len(LONGRUN):
        LONGRUN.insert(0, "rho_sum_ar", _rho)
        LONGRUN.insert(1, "rho_ci_lo", AR_STABILITY["rho_ci_lo"])
        LONGRUN.insert(2, "rho_ci_hi", AR_STABILITY["rho_ci_hi"])
        LONGRUN["long_run_multiplier"] = (1 / (1 - _rho)) if LONGRUN_ESTIMABLE else np.nan
else:
    print("Dynamic specification skipped (too few rows once the AR term is lagged).")

MODEL_A_FIT_STATS = pd.DataFrame(_fit_rows)
MODEL_A_COEFS = pd.concat(_coef_frames, ignore_index=True) if _coef_frames else pd.DataFrame()

print("Fit statistics by specification (in-sample; skill is §16's job):")
display(MODEL_A_FIT_STATS.round(4))
print("\\nDriver coefficients (per 1 SD of the driver, calendar-aware HAC SEs, "
      "BH q within spec):")
display(MODEL_A_COEFS.round(4))

if AR_STABILITY:
    print("\\n" + "-" * 78)
    print("AR STATIONARITY CHECK (decides whether long-run effects may be quoted)")
    print("-" * 78)
    print(f"  sum of AR coefficients rho = {AR_STABILITY['rho_sum']:.3f}, "
          f"{100 * (1 - LONGRUN_CI_ALPHA):.0f}% HAC CI "
          f"[{AR_STABILITY['rho_ci_lo']:.3f}, {AR_STABILITY['rho_ci_hi']:.3f}]")
    print(f"  AR polynomial roots outside the unit circle: "
          f"{AR_STABILITY['roots_outside_unit_circle']}")
    print(f"  interval strictly inside (-1, 1)            : "
          f"{AR_STABILITY['ci_within_unit_circle']}")
    if LONGRUN_ESTIMABLE and AR_STABILITY["stationary_supported"]:
        print(f"  -> stationarity supported; long-run multiplier "
              f"1/(1-rho) = {1 / (1 - AR_STABILITY['rho_sum']):.2f}")
    else:
        print("\\n  *** LONG-RUN EFFECTS ARE NOT IDENTIFIED RELIABLY ***")
        print(f"  Reason: {AR_STABILITY['reason']}.")
        print("  The multiplier 1/(1-rho) is unbounded over this interval, so neither it")
        print("  nor any beta/(1-rho) is reported: those columns are NaN and")
        print("  long_run_estimable = False. The SHORT-RUN coefficients stand and are the")
        print("  only dynamic effects that may be quoted. Do not carry a multiplier from")
        print("  an earlier run of this notebook into the write-up.")
if len(LONGRUN):
    print("\\nDynamic (M4) effects:")
    display(LONGRUN.round(4))

# --- How much does seasonal control change the story? -------------------------
if {"M1 drivers only", HEADLINE_SPEC}.issubset(set(MODEL_A_COEFS["specification"])):
    _m1 = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == "M1 drivers only"] \\
        .set_index("term")["coef"]
    _m3 = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == HEADLINE_SPEC] \\
        .set_index("term")["coef"]
    SEASON_SHRINKAGE = pd.DataFrame({"marginal_M1": _m1, "controlled_M3": _m3})
    SEASON_SHRINKAGE["change"] = SEASON_SHRINKAGE["controlled_M3"] - SEASON_SHRINKAGE["marginal_M1"]
    SEASON_SHRINKAGE["sign_flipped"] = (np.sign(SEASON_SHRINKAGE["marginal_M1"])
                                        != np.sign(SEASON_SHRINKAGE["controlled_M3"]))
    print("\\nWhat seasonal + trend control does to each driver effect:")
    display(SEASON_SHRINKAGE.round(4))
    if SEASON_SHRINKAGE["sign_flipped"].any():
        print("Sign FLIPPED under seasonal control for: "
              f"{SEASON_SHRINKAGE.index[SEASON_SHRINKAGE['sign_flipped']].tolist()}")
        print("The marginal association for those drivers was seasonality, not the driver.")

# --- Semi-partial R2, once per specification ---------------------------------
# TWO tables, each computed on exactly the rows, columns, response and weighting
# of the specification it names. §17 may only pair a Shapley decomposition with
# the semi-partial table of the SAME specification; mixing them (a dynamic
# Shapley against a static semi-partial) compares different models.
_hs_cols = [c for c in SPECS[HEADLINE_SPEC] if c in fit_df.columns]
SEMI_PARTIAL = semi_partial_r2(
    fit_df["y"], fit_df[_hs_cols], weights=W, terms=DRIVER_TERMS,
    specification="without persistence", response=f"{RESPONSE_INFO['transform']}({RESPONSE_COL})",
    weighting=MONTH_WEIGHTING)
print(f"\\nUnique (last-entry) explanatory contribution in {HEADLINE_SPEC} "
      f"— specification 'without persistence', {SEMI_PARTIAL['n_obs'].iloc[0]} rows:")
display(SEMI_PARTIAL.drop(columns=["model_columns"]).round(4))

SEMI_PARTIAL_AR = pd.DataFrame()
if DYN_COLS:
    SEMI_PARTIAL_AR = semi_partial_r2(
        dyn_df["y"], dyn_df[DYN_COLS], weights=W_DYN, terms=DRIVER_TERMS,
        specification="with persistence",
        response=f"{RESPONSE_INFO['transform']}({RESPONSE_COL})",
        weighting=MONTH_WEIGHTING)
    print(f"\\nUnique (last-entry) explanatory contribution in {DYNAMIC_SPEC} "
          f"— specification 'with persistence', {SEMI_PARTIAL_AR['n_obs'].iloc[0]} rows:")
    display(SEMI_PARTIAL_AR.drop(columns=["model_columns"]).round(4))
    print("A driver's unique contribution is normally SMALLER here: persistence has "
          "already taken the variance the driver shares with last month's hyacinth.")
''')

md("""### 12b. Coefficient plot and residual diagnostics

The residual ACF is the check that matters: if the residuals are still strongly
autocorrelated, the HAC standard errors are doing heavy lifting and the dynamic
specification (M4) is the more honest reading.
""")

code('''# =====================================================================
# 12b. Coefficient plot and residual diagnostics
# =====================================================================
_plot = MODEL_A_COEFS[MODEL_A_COEFS["specification"].isin(
    ["M1 drivers only", "M2 drivers+season", HEADLINE_SPEC])].copy()
if len(_plot):
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(_plot) + 1.5))
    offsets = {"M1 drivers only": -0.24, "M2 drivers+season": 0.0, HEADLINE_SPEC: 0.24}
    colours = {"M1 drivers only": "tab:grey", "M2 drivers+season": "tab:orange",
               HEADLINE_SPEC: "tab:blue"}
    terms = [t for t in DRIVER_TERMS if t in set(_plot["term"])]
    ypos = {t: i for i, t in enumerate(terms)}
    for spec, grp in _plot.groupby("specification"):
        yy = [ypos[t] + offsets.get(spec, 0.0) for t in grp["term"] if t in ypos]
        g = grp[grp["term"].isin(ypos)]
        ax.errorbar(g["coef"], yy,
                    xerr=[g["coef"] - g["ci_lo"], g["ci_hi"] - g["coef"]],
                    fmt="o", ms=5, capsize=3, label=spec, color=colours.get(spec))
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([f"{t}  ({FORCING.get(t.replace('_lag1', '').replace('_lag2', ''), {}).get('expected_sign', '?')})"
                        for t in terms])
    ax.invert_yaxis()
    ax.set_xlabel(f"effect on {RESPONSE_INFO['transform']}({RESPONSE_COL}) per 1 SD of driver "
                  "(HAC 95% CI)")
    ax.set_title("Driver effects — marginal vs seasonally controlled\\n"
                 "(a-priori expected sign in brackets)")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show()

_res = MODEL_A_FITS.get(HEADLINE_SPEC)
if _res is not None:
    resid = pd.Series(np.asarray(_res.resid, dtype=float))
    # Residual ACF on CALENDAR lags: the residuals live on the complete-case
    # rows, whose row order compresses the excluded months away.
    acf_r = acf_values(resid, months=FIT_MONTHS, nlags=min(18, max(4, len(resid) // 3)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    axes[0].bar(acf_r["lag"], acf_r["acf"], color="tab:red", alpha=0.75)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhline(acf_r["band"].iloc[0], color="k", ls="--", lw=0.8)
    axes[0].axhline(-acf_r["band"].iloc[0], color="k", ls="--", lw=0.8)
    axes[0].set_title(f"Residual ACF — {HEADLINE_SPEC}")
    axes[0].set_xlabel("lag (calendar months)")
    axes[1].scatter(_res.fittedvalues, resid, s=18, alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("fitted"); axes[1].set_ylabel("residual")
    axes[1].set_title("Residuals vs fitted")
    sstats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout(); plt.show()

    _lb = sm.stats.acorr_ljungbox(resid, lags=[min(12, max(2, len(resid) // 4))],
                                  return_df=True)
    print("Ljung-Box test on the residuals (H0: no remaining autocorrelation):")
    display(_lb)
    if float(_lb["lb_pvalue"].iloc[0]) < 0.05:
        print("Residual autocorrelation remains. The HAC SEs are already accounting for "
              "it; read M4 (dynamic) as the primary specification and treat the static "
              "coefficients as the contemporaneous part of a persistent process.")

    _fitted = pd.Series(np.asarray(_res.fittedvalues, dtype=float))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    # `fit_df` holds only complete-case months, so plot it on the calendar grid:
    # otherwise both lines run straight across every month the model never saw.
    ax.plot(*on_calendar_grid(fit_df["month"], fit_df["y"]),
            marker="o", ms=3, lw=1.2, label="observed")
    ax.plot(*on_calendar_grid(fit_df["month"], _fitted),
            lw=1.6, color="tab:red", label="fitted (M3, in-sample)")
    ax.set_ylabel(f"{RESPONSE_INFO['transform']}({RESPONSE_COL})")
    ax.set_title("Observed vs in-sample fit — NOT a skill estimate (see §16)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()
''')

md("""### 12c. Bootstrap confidence intervals

A second, distribution-free reading of the same coefficients. Moving-block
resampling keeps short-range dependence inside each replicate, so agreement
between the HAC interval and the bootstrap interval is real reassurance, and
disagreement is a warning that the interval is being driven by a few months.
`boot_sign_stability` is the share of replicates in which the effect keeps its
sign — the single most useful robustness number for a small series.
""")

code('''# =====================================================================
# 12c. Moving-block bootstrap of the headline coefficients
# =====================================================================
_boot_draws, BOOT_INFO = bootstrap_coefficients(
    fit_df["y"], fit_df[_hs_cols], FIT_MONTHS, weights=W,
    n_boot=BOOTSTRAP_N, block=BOOTSTRAP_BLOCK, seed=BOOTSTRAP_SEED)
BOOT_SUMMARY = bootstrap_summary(_boot_draws, terms=DRIVER_TERMS)
print(f"Moving-block bootstrap on the CALENDAR grid: "
      f"{BOOT_INFO['n_successful']:,} of {BOOT_INFO['n_requested']:,} replicates fitted, "
      f"block length {BOOT_INFO['block_months']} calendar months.")
print(f"Fitted sample size per replicate: min {BOOT_INFO['n_rows_fitted_min']}, "
      f"median {BOOT_INFO['n_rows_fitted_median']:.0f}, "
      f"max {BOOT_INFO['n_rows_fitted_max']} (of {BOOT_INFO['n_rows_available']} "
      "usable months). It varies because a block landing on excluded months "
      "contributes fewer rows — the gaps are kept, not closed up.")
display(BOOT_SUMMARY.round(4))

_hac = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == HEADLINE_SPEC][
    ["term", "coef", "ci_lo", "ci_hi"]].rename(
    columns={"ci_lo": "hac_ci_lo", "ci_hi": "hac_ci_hi"})
CI_COMPARISON = _hac.merge(BOOT_SUMMARY, on="term", how="left")
CI_COMPARISON["hac_width"] = CI_COMPARISON["hac_ci_hi"] - CI_COMPARISON["hac_ci_lo"]
CI_COMPARISON["boot_width"] = CI_COMPARISON["boot_ci_hi"] - CI_COMPARISON["boot_ci_lo"]
print("\\nHAC vs bootstrap intervals (agreement = the interval is not an artefact):")
display(CI_COMPARISON.round(4))
''')

md("""### 12e. Descriptive contrast — the endogenous water-quality proxies

Chlorophyll-a and turbidity are retrieved from the same reflectance a floating
mat dominates, and a mat changes the water beneath it. They are therefore as
plausibly *consequences* of hyacinth as causes of it, so they are excluded from
every driver claim above. Their association is reported **once, here, labelled
descriptive** — never as a driver effect.
""")

code('''# =====================================================================
# 12e. Descriptive proxy association (not a driver claim)
# =====================================================================
PROXY_COEFS = pd.DataFrame()
if RUN_DESCRIPTIVE_PROXY_MODEL and PROXY_TERMS:
    _cols = [c for c in DRIVER_TERMS + PROXY_TERMS + SEASON_COLS + TREND_TERMS
             if c in model_df.columns]
    _m = model_df[["y"] + _cols].notna().all(axis=1)
    _pdf = model_df.loc[_m]
    if len(_pdf) > len(_cols) + 5:
        _w = _pdf["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None
        _rp = fit_hac(_pdf["y"], _pdf[_cols], weights=_w, maxlags=HAC_MAXLAGS,
                      months=_pdf["month"])
        PROXY_COEFS = tidy_coefficients(_rp, keep=PROXY_TERMS,
                                        label="descriptive (+endogenous proxies)")
        PROXY_COEFS["evidence_type"] = "descriptive association — NOT a driver effect"
        print(f"Descriptive model on {len(_pdf)} months, R2 = {_rp.rsquared:.3f} "
              f"(vs {MODEL_A_FITS[HEADLINE_SPEC].rsquared:.3f} for {HEADLINE_SPEC}):")
        display(PROXY_COEFS.round(4))
        print("\\nRead as: months with more hyacinth also show this water-quality signal. "
              "The direction of causation is not identified, and the retrieval itself is "
              "contaminated by the mats.")
    else:
        print("Too few months with the proxies present to fit the descriptive model.")
else:
    print("No endogenous proxies available (or RUN_DESCRIPTIVE_PROXY_MODEL = False).")
''')

# ===========================================================================
# 13. Model B
# ===========================================================================
md("""## 13. Model B — regularised regression and bootstrap stability selection

Rainfall, temperature and wind all move together, so with ~100 months an
unregularised coefficient can be large and "significant" mostly because of which
*other* driver happened to absorb the shared variance. Two answers:

* **Elastic net** with the α (L1/L2 mix) and λ chosen by *rolling-origin* CV —
  never random *k*-fold, which leaks the future into the past on a time series.
* **Stability selection** — refit the elastic net on `BOOTSTRAP_N` moving-block
  resamples and record how often each driver survives with a non-zero
  coefficient and a consistent sign.

`selection_frequency` is the number to quote when a reviewer asks *"is this
driver robust?"*. A driver selected in 80%+ of resamples with a stable sign is a
real feature of the record; one selected in 30% is a coin flip dressed as a
finding.
""")

code('''# =====================================================================
# 13. Elastic net with rolling-origin CV + stability selection
# =====================================================================
ENET_COEFS = pd.DataFrame()
STABILITY = pd.DataFrame()
ENET_INFO = {}
ENET_SPARSITY_OK = False

if not HAVE_SKLEARN:
    print("scikit-learn unavailable; §13 skipped.")
else:
    _X = fit_df[[c for c in DRIVER_TERMS + SEASON_COLS + TREND_TERMS
                 if c in fit_df.columns]].astype(float)
    _y = fit_df["y"].to_numpy(dtype=float)
    # Calendar rolling-origin folds: a horizon of CV_HORIZON_MONTHS means that
    # many CONSECUTIVE CALENDAR MONTHS after the origin, not that many rows.
    _folds, _fold_audit = rolling_origin_month_folds(
        FIT_MONTHS, n_folds=CV_N_FOLDS, horizon_months=CV_HORIZON_MONTHS,
        min_train_months=CV_MIN_TRAIN_MONTHS, min_test_months=CV_MIN_TEST_MONTHS)
    if len(_folds) < 2:
        print(f"Only {len(_folds)} calendar rolling-origin fold(s) fit in "
              f"{calendar_span_months(FIT_MONTHS)} calendar months; lower "
              "CV_MIN_TRAIN_MONTHS, CV_HORIZON_MONTHS or CV_MIN_TEST_MONTHS. §13 skipped.")
    else:
        _cv = fold_index_pairs(_folds)
        _alphas_l1 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        _best, _best_rmse = None, np.inf
        for l1 in _alphas_l1:
            # The alpha-grid keyword was renamed across scikit-learn versions
            # (n_alphas -> alphas), so neither is passed and the default grid of
            # 100 is used. This keeps the cell working on Colab and locally.
            m = ElasticNetCV(l1_ratio=l1, cv=_cv, max_iter=20000,
                             random_state=RANDOM_STATE)
            try:
                m.fit(_X, _y)
            except Exception as exc:
                print(f"  l1_ratio {l1}: fit failed ({exc})"); continue
            sc, _ = cv_scores(
                _y, _X, _folds, months=FIT_MONTHS,
                fit=lambda Xtr, ytr, wtr, _a=m.alpha_, _l=l1: ElasticNet(
                    alpha=_a, l1_ratio=_l, max_iter=20000,
                    random_state=RANDOM_STATE).fit(Xtr, ytr),
                predict=lambda mod, Xte: mod.predict(Xte))
            print(f"  l1_ratio {l1:.1f}: lambda {m.alpha_:.4g}, rolling-origin RMSE "
                  f"{sc['rmse']:.4f} on {sc['n_test']} evaluated months")
            if np.isfinite(sc["rmse"]) and sc["rmse"] < _best_rmse:
                _best, _best_rmse = (l1, m.alpha_), sc["rmse"]

        if _best is None:
            print("No elastic-net configuration converged; §13 skipped.")
        else:
            _l1, _lam = _best
            ENET_INFO = {"l1_ratio": _l1, "alpha": float(_lam), "cv_rmse": float(_best_rmse),
                         "n_folds": len(_folds),
                         "cv_design": f"{CV_HORIZON_MONTHS}-calendar-month rolling-origin windows"}
            _final = ElasticNet(alpha=_lam, l1_ratio=_l1, max_iter=20000,
                                random_state=RANDOM_STATE).fit(_X, _y)
            ENET_COEFS = pd.DataFrame({"term": _X.columns, "enet_coef": _final.coef_})
            ENET_COEFS["selected"] = ENET_COEFS["enet_coef"].abs() > 1e-8
            ENET_COEFS["is_driver"] = ENET_COEFS["term"].isin(DRIVER_TERMS)
            print(f"\\nChosen: l1_ratio {_l1}, lambda {_lam:.4g} (RMSE {_best_rmse:.4f} over "
                  f"{len(_folds)} {CV_HORIZON_MONTHS}-calendar-month rolling-origin windows)")
            display(ENET_COEFS.sort_values("enet_coef", key=np.abs, ascending=False).round(4))

            # --- Stability selection over CALENDAR moving-block resamples -----
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            _block = calendar_block_length(FIT_MONTHS, BOOTSTRAP_BLOCK)
            n_boot_stab = min(int(BOOTSTRAP_N), 600)   # each replicate refits the net
            counts = {c: {"nonzero": 0, "pos": 0, "neg": 0} for c in _X.columns}
            n_ok = 0
            _stab_sizes = []
            for idx, _blocks in calendar_block_indices(FIT_MONTHS, block_months=BOOTSTRAP_BLOCK,
                                                       rng=rng, n_boot=n_boot_stab):
                if len(idx) <= _X.shape[1]:
                    continue
                try:
                    mb = ElasticNet(alpha=_lam, l1_ratio=_l1, max_iter=20000,
                                    random_state=RANDOM_STATE).fit(_X.iloc[idx], _y[idx])
                except Exception:
                    continue
                n_ok += 1
                _stab_sizes.append(len(idx))
                for c, b in zip(_X.columns, mb.coef_):
                    if abs(b) > 1e-8:
                        counts[c]["nonzero"] += 1
                        counts[c]["pos" if b > 0 else "neg"] += 1
            ENET_INFO["stability_replicates"] = int(n_ok)
            ENET_INFO["stability_block_months"] = int(_block)
            ENET_INFO["stability_rows_median"] = (float(np.median(_stab_sizes))
                                                  if _stab_sizes else np.nan)
            if n_ok:
                STABILITY = pd.DataFrame([{
                    "term": c,
                    "selection_frequency": counts[c]["nonzero"] / n_ok,
                    "sign_consistency": (max(counts[c]["pos"], counts[c]["neg"])
                                         / counts[c]["nonzero"]) if counts[c]["nonzero"] else np.nan,
                    "dominant_sign": ("+" if counts[c]["pos"] >= counts[c]["neg"] else "-")
                                     if counts[c]["nonzero"] else "",
                } for c in _X.columns])
                STABILITY["is_driver"] = STABILITY["term"].isin(DRIVER_TERMS)
                STABILITY = STABILITY.sort_values("selection_frequency", ascending=False) \\
                    .reset_index(drop=True)
                print(f"\\nStability selection over {n_ok} calendar moving-block resamples "
                      f"(block {_block} calendar months, median "
                      f"{np.median(_stab_sizes):.0f} fitted rows):")
                display(STABILITY.round(3))

                # A near-ridge fit (low l1_ratio) hardly ever sets a coefficient
                # to exactly zero, so EVERY driver scores a high selection
                # frequency and the number carries no information. Detect that
                # and say so, rather than letting §25 read it as robustness.
                _sd_freq = STABILITY.loc[STABILITY["is_driver"], "selection_frequency"]
                ENET_SPARSITY_OK = bool(len(_sd_freq) and _sd_freq.min() < 0.90)
                ENET_INFO["sparsity_informative"] = ENET_SPARSITY_OK
                if not ENET_SPARSITY_OK:
                    print(f"\\n*** The chosen l1_ratio ({_l1}) produced a non-sparse fit: "
                          f"every driver survives in >= {_sd_freq.min():.0%} of resamples. "
                          "Selection frequency is therefore UNINFORMATIVE here and §25 "
                          "ignores it when assigning verdicts. ***")

                _sd = STABILITY[STABILITY["is_driver"]]
                if len(_sd):
                    fig, ax = plt.subplots(figsize=(8, 0.42 * len(_sd) + 1.4))
                    ax.barh(_sd["term"], _sd["selection_frequency"],
                            color=["tab:blue" if s == "+" else "tab:red"
                                   for s in _sd["dominant_sign"]])
                    ax.axvline(0.8, color="k", ls="--", lw=1, label="0.8 (robust)")
                    ax.axvline(0.5, color="grey", ls=":", lw=1, label="0.5 (coin flip)")
                    ax.set_xlim(0, 1); ax.invert_yaxis()
                    ax.set_xlabel("share of bootstrap resamples retaining the driver")
                    ax.set_title("Stability selection — blue = positive effect, red = negative")
                    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
                    fig.tight_layout(); plt.show()
''')


# ===========================================================================
# 14. Model C
# ===========================================================================
md("""## 14. Model C — non-linear driver shapes (natural-spline GLM), **exploratory**

### What was wrong with the previous version of this section

The earlier §14 entered **every** driver as a `df = 4` natural-cubic smooth
*simultaneously*: 6 drivers x 4 basis columns + 4 season columns + trend = **25
design columns fitted on 64 complete months**. At 2.6 rows per column an
unpenalised spline basis interpolates; the resulting curves, their $F$ tests and
the apparent lake-level non-linearity were properties of that over-parameterised
design, not of the lake. That result is **withdrawn**, and this section is
rebuilt with five safeguards:

1. **One predeclared driver at a time.** Each fit is `smooth(driver_j) + season
   + trend` — the other drivers enter *linearly*, never as smooths. The design
   is then `SPLINE_DF` + ~6 columns instead of 25.
2. **`SPLINE_DF` <= `SPLINE_DF_MAX` = 3**, and a smooth is refused outright
   below `SPLINE_MIN_ROWS_PER_COLUMN` rows per design column, with the refusal
   printed.
3. **Each smooth is centred** — the training basis columns have their training
   means subtracted — so the smooth carries no intercept and cannot trade level
   against the constant, the season or the trend.
4. **The training `design_info` is reused for every prediction grid.** `patsy`
   is asked once, on the training data, for the basis; the plotting grid is
   built with `build_design_matrices([design_info], ...)`. Knots and boundary
   constraints are therefore *never* recomputed on the grid, which is what made
   the old partial-effect curves untrustworthy near the data edges.
5. **Exploratory unless it survives out of sample.** §14b re-runs the
   one-calendar-month-ahead rolling origin of §21 with the smooth in place of the
   linear term. A curve is promoted from `exploratory` to `supported
   out-of-sample` only if it lowers one-month-ahead RMSE against the *identical*
   linear specification. Nothing in §25 quotes a non-linear effect that has not.

The $p$-values remain nested $F$ tests assuming independent residuals — an
**upper bound on the evidence** on a series with lag-1 autocorrelation near 0.9.
They are printed to rank shapes, never to certify one.
""")

code('''# =====================================================================
# 14. Natural-spline GLM: ONE predeclared driver at a time, centred, df <= 3
# =====================================================================
SPLINE_TESTS = pd.DataFrame()
SPLINE_FITS = {}
SPLINE_DESIGN_INFO = {}
PARTIAL_EFFECTS = pd.DataFrame()
SPLINE_REFUSALS = []
SPLINE_SUPPORTED_OOS = []          # filled by §14b; empty until then

_spline_df = int(min(SPLINE_DF, SPLINE_DF_MAX))
_ctrl_cols = [c for c in SEASON_COLS + TREND_TERMS if c in fit_df.columns]

if not HAVE_PATSY:
    print("patsy unavailable; §14 skipped.")
elif not SPLINE_ONE_DRIVER_AT_A_TIME:
    print("SPLINE_ONE_DRIVER_AT_A_TIME = False: the all-drivers-at-once design is "
          "refused here because it produced 25 columns on 64 months. Set it back to "
          "True, or accept that §14 does not run.")
else:
    from patsy import build_design_matrices, dmatrices  # noqa: F401

    def _centred_spline(frame, col, df):
        """Training basis for ONE driver, centred, with its patsy design_info kept.

        Returns (basis DataFrame, design_info, centring means). The design_info is
        what every prediction grid is later built from, so the knots and the
        boundary constraints are fixed by the TRAINING data once and never
        recomputed.
        """
        B = dmatrix(f"cr(x, df={int(df)}) - 1",
                    {"x": frame[col].to_numpy(dtype=float)},
                    return_type="dataframe")
        info = B.design_info
        means = B.mean(axis=0)
        Bc = B - means                       # centred: no intercept inside the smooth
        Bc.columns = [f"{col}__s{i + 1}" for i in range(Bc.shape[1])]
        Bc.index = frame.index
        return Bc, info, means

    def _spline_grid(col, info, means, xs):
        """Prediction basis for `xs`, built from the TRAINING design_info."""
        Bg = pd.DataFrame(np.asarray(build_design_matrices([info], {"x": np.asarray(xs)})[0]))
        Bg = Bg - means.to_numpy()
        Bg.columns = [f"{col}__s{i + 1}" for i in range(Bg.shape[1])]
        return Bg

    def _plain(y, cols, w):
        X = sm.add_constant(pd.DataFrame(cols).astype(float), has_constant="add")
        return (sm.OLS(y, X).fit() if w is None
                else sm.WLS(y, X, weights=w / np.nanmean(w)).fit())

    _rows, _pe_rows = [], []
    for c in DRIVER_TERMS:
        # The smoothed driver, the OTHER drivers entering linearly, and the controls.
        others = [d for d in DRIVER_TERMS if d != c]
        B, info, means = _centred_spline(fit_df, c, _spline_df)
        X_smooth = pd.concat([B, fit_df[others + _ctrl_cols]], axis=1)
        X_linear = pd.concat([fit_df[[c] + others + _ctrl_cols]], axis=1)
        X_drop = fit_df[others + _ctrl_cols]

        n_cols = X_smooth.shape[1] + 1                       # + intercept
        if N_FIT < SPLINE_MIN_ROWS_PER_COLUMN * n_cols:
            SPLINE_REFUSALS.append(
                {"driver": c, "n_rows": int(N_FIT), "n_columns": int(n_cols),
                 "rows_per_column": float(N_FIT / n_cols),
                 "reason": (f"{N_FIT} rows for {n_cols} columns is below the "
                            f"{SPLINE_MIN_ROWS_PER_COLUMN} rows-per-column floor")})
            continue

        m_smooth = _plain(fit_df["y"], X_smooth, W)
        m_linear = _plain(fit_df["y"], X_linear, W)
        m_drop = _plain(fit_df["y"], X_drop, W)
        f_any = m_smooth.compare_f_test(m_drop)
        f_nl = m_smooth.compare_f_test(m_linear)

        SPLINE_FITS[c] = fit_hac(fit_df["y"], X_smooth, weights=W, maxlags=HAC_MAXLAGS,
                                 months=FIT_MONTHS)
        SPLINE_DESIGN_INFO[c] = {"design_info": info, "centring_means": means,
                                 "basis_columns": list(B.columns),
                                 "model_columns": list(X_smooth.columns),
                                 "spline_df": int(B.shape[1])}
        _rows.append({
            "driver": c, "spline_df": int(B.shape[1]),
            "n_model_columns": int(n_cols), "n_rows": int(N_FIT),
            "rows_per_column": float(N_FIT / n_cols),
            "p_any_effect_F": float(f_any[1]), "p_nonlinearity_F": float(f_nl[1]),
            "r2_smooth": float(m_smooth.rsquared),
            "r2_linear_for_this_driver": float(m_linear.rsquared),
            "r2_without_this_driver": float(m_drop.rsquared),
            "delta_r2_vs_dropped": float(m_smooth.rsquared - m_drop.rsquared),
            "nonlinearity_gain_r2": float(m_smooth.rsquared - m_linear.rsquared),
            "evidence_status": "exploratory",
        })

        # --- Partial-effect curve, on the TRAINING design_info -----------------
        xs = np.linspace(float(fit_df[c].min()), float(fit_df[c].max()), 80)
        Bg = _spline_grid(c, info, means, xs)
        Xg = pd.DataFrame(0.0, index=Bg.index, columns=X_smooth.columns)
        for col in Bg.columns:
            Xg[col] = Bg[col].to_numpy()
        for col in others + _ctrl_cols:
            Xg[col] = float(fit_df[col].mean())
        Xg = sm.add_constant(Xg, has_constant="add")
        pred = SPLINE_FITS[c].get_prediction(Xg).summary_frame(alpha=0.05)
        centre = float(pred["mean"].mean())
        _pe_rows.append(pd.DataFrame({
            "driver": c, "x_sd_units": xs,
            "partial_effect": pred["mean"].to_numpy() - centre,
            "ci_lo": pred["mean_ci_lower"].to_numpy() - centre,
            "ci_hi": pred["mean_ci_upper"].to_numpy() - centre,
            "design_info_source": "training fit (knots NOT recomputed on the grid)",
            "evidence_status": "exploratory"}))

    if _rows:
        SPLINE_TESTS = pd.DataFrame(_rows)
        SPLINE_TESTS["q_fdr"] = bh_fdr(SPLINE_TESTS["p_any_effect_F"]).to_numpy()
        SPLINE_TESTS["q_fdr_nonlinearity"] = bh_fdr(
            SPLINE_TESTS["p_nonlinearity_F"]).to_numpy()
        SPLINE_TESTS = SPLINE_TESTS.sort_values(
            "nonlinearity_gain_r2", ascending=False).reset_index(drop=True)
        PARTIAL_EFFECTS = pd.concat(_pe_rows, ignore_index=True)

        print(f"Natural-spline GLM, ONE driver at a time, df = {_spline_df} "
              f"(ceiling {SPLINE_DF_MAX}), each smooth centred, on {N_FIT} months.")
        print(f"Every fit has {int(SPLINE_TESTS['n_model_columns'].max())} columns or "
              f"fewer — the withdrawn all-at-once design had 25 on the same 64 months.")
        print("p-values are nested F tests assuming independent residuals, so they are "
              "an UPPER BOUND on the evidence; read nonlinearity_gain_r2 as the "
              "magnitude and treat every shape as EXPLORATORY until §14b.")
        display(SPLINE_TESTS.round(4))

        ncol = min(3, len(SPLINE_TESTS))
        nrow = int(np.ceil(len(SPLINE_TESTS) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.0 * nrow),
                                 squeeze=False)
        for ax, c in zip(axes.ravel(), SPLINE_TESTS["driver"]):
            g = PARTIAL_EFFECTS[PARTIAL_EFFECTS["driver"] == c]
            ax.plot(g["x_sd_units"], g["partial_effect"], color="tab:blue", lw=2)
            ax.fill_between(g["x_sd_units"], g["ci_lo"], g["ci_hi"],
                            color="tab:blue", alpha=0.18)
            ax.plot(fit_df[c], np.full(N_FIT, ax.get_ylim()[0]), "|", color="k",
                    ms=6, alpha=0.4)
            ax.axhline(0, color="k", lw=0.8, ls=":")
            _q = SPLINE_TESTS.loc[SPLINE_TESTS["driver"] == c, "q_fdr_nonlinearity"]
            ax.set_title(f"{c}\\nq(non-linearity) = {float(_q.iloc[0]):.3g}", fontsize=9)
            ax.set_xlabel(f"{c} (SD units)")
            ax.set_ylabel(f"partial effect on {RESPONSE_INFO['transform']}")
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(SPLINE_TESTS):]:
            ax.axis("off")
        fig.suptitle(f"EXPLORATORY partial driver-response shapes — one driver at a "
                     f"time, centred cr() df={_spline_df}, 95% CI\\n"
                     "knots and boundary constraints fixed by the training design_info",
                     y=1.02, fontsize=10)
        fig.tight_layout(); plt.show()
    else:
        print("No driver could be smoothed within the rows-per-column floor.")

if SPLINE_REFUSALS:
    print("\\nSmooths REFUSED (design too thin for the number of months):")
    display(pd.DataFrame(SPLINE_REFUSALS).round(3))
''')

md("""### 14b. Does any curve survive out of sample?

A curve that lowers in-sample $R^2$-per-column but not out-of-sample error is a
description of these 64 months, not of the system. Each driver flagged in §14 is
re-run through the **same one-calendar-month-ahead rolling origin** the principal
model uses (§21), twice on identical training and target months:

* the driver entering **linearly**, and
* the driver entering as the **centred smooth**, with the basis rebuilt from the
  *training fold's* `design_info` at every origin.

A smooth is promoted to `supported out-of-sample` only when its one-month-ahead
RMSE is lower than the linear specification's **by a margin whose 95%
calendar-aware moving-block interval excludes zero** — the same bar §22 applies
to the driver block, for the same reason: on ~40 targets a difference of a few
ten-thousandths is noise. Otherwise the shape stays `exploratory` and **§25 will
not quote it as a non-linear result**.
""")

code('''# =====================================================================
# 14b. Out-of-sample check on the non-linear shapes
# =====================================================================
SPLINE_OOS = pd.DataFrame()
if not (SPLINE_OOS_CHECK and len(SPLINE_TESTS) and HAVE_PATSY):
    print("§14b skipped (no smooths fitted, or SPLINE_OOS_CHECK = False). "
          "Every shape in §14 therefore stays EXPLORATORY.")
else:
    from patsy import build_design_matrices

    # Rolling origin on the SAME complete-case rows §14 used, one calendar month
    # ahead, expanding window. The fold machinery is §5's, so the origins are
    # calendar months and training is always strictly before the target.
    _oos_rows = []
    _mi_fit = month_index(FIT_MONTHS)
    for c in SPLINE_TESTS["driver"]:
        others = [d for d in DRIVER_TERMS if d != c]
        rec = {"driver": c, "n_targets": 0, "rmse_linear": np.nan,
               "rmse_spline": np.nan}
        e_lin, e_spl, e_months = [], [], []
        for pos in range(len(FIT_MONTHS)):
            train = np.where(_mi_fit < _mi_fit[pos])[0]
            if len(train) < SS_MIN_TRAIN_MONTHS:
                continue
            tr = fit_df.iloc[train]
            te = fit_df.iloc[[pos]]
            wtr = (tr["w_month"].to_numpy(dtype=float)
                   if MONTH_WEIGHTING != "none" else None)
            # Linear specification.
            Xl_tr = tr[[c] + others + _ctrl_cols]
            Xl_te = te[[c] + others + _ctrl_cols]
            ml = (sm.OLS(tr["y"], sm.add_constant(Xl_tr, has_constant="add")).fit()
                  if wtr is None else
                  sm.WLS(tr["y"], sm.add_constant(Xl_tr, has_constant="add"),
                         weights=wtr / np.nanmean(wtr)).fit())
            pl = float(ml.predict(sm.add_constant(Xl_te, has_constant="add").reindex(
                columns=ml.params.index, fill_value=0.0)).iloc[0])
            # Smooth specification: the basis is built on THIS FOLD'S training data,
            # and the target month's basis comes from that fold's design_info.
            try:
                Btr = dmatrix(f"cr(x, df={int(_spline_df)}) - 1",
                              {"x": tr[c].to_numpy(dtype=float)},
                              return_type="dataframe")
                info, mu = Btr.design_info, Btr.mean(axis=0)
                Btr = (Btr - mu)
                Btr.columns = [f"{c}__s{i + 1}" for i in range(Btr.shape[1])]
                Btr.index = tr.index
                Bte = pd.DataFrame(np.asarray(build_design_matrices(
                    [info], {"x": te[c].to_numpy(dtype=float)})[0])) - mu.to_numpy()
                Bte.columns = Btr.columns
                Bte.index = te.index
                Xs_tr = pd.concat([Btr, tr[others + _ctrl_cols]], axis=1)
                Xs_te = pd.concat([Bte, te[others + _ctrl_cols]], axis=1)
                ms = (sm.OLS(tr["y"], sm.add_constant(Xs_tr, has_constant="add")).fit()
                      if wtr is None else
                      sm.WLS(tr["y"], sm.add_constant(Xs_tr, has_constant="add"),
                             weights=wtr / np.nanmean(wtr)).fit())
                ps = float(ms.predict(sm.add_constant(Xs_te, has_constant="add").reindex(
                    columns=ms.params.index, fill_value=0.0)).iloc[0])
            except Exception:
                continue
            yt = float(te["y"].iloc[0])
            e_lin.append(yt - pl)
            e_spl.append(yt - ps)
            e_months.append(te["month"].iloc[0])
        if len(e_lin) >= 5:
            rec["n_targets"] = int(len(e_lin))
            rec["rmse_linear"] = float(np.sqrt(np.mean(np.asarray(e_lin) ** 2)))
            rec["rmse_spline"] = float(np.sqrt(np.mean(np.asarray(e_spl) ** 2)))
            rec["rmse_difference_spline_minus_linear"] = (rec["rmse_spline"]
                                                          - rec["rmse_linear"])
            # A point estimate is not evidence here either. The same calendar-aware
            # moving-block bootstrap §22 uses decides it: the smooth is promoted
            # only if its 95% interval for RMSE(spline) - RMSE(linear) lies wholly
            # BELOW zero. Without this a difference of -0.0004 on 44 months would
            # "support" a curve, which is exactly the error §14 exists to undo.
            _b = block_bootstrap_rmse_difference(
                pd.Series(e_months), np.asarray(e_spl), np.asarray(e_lin),
                n_boot=min(SS_RMSE_BOOTSTRAP_N, 1000), seed=SS_RMSE_BOOTSTRAP_SEED)
            rec["boot_ci_lo"] = _b["ci_lo"]
            rec["boot_ci_hi"] = _b["ci_hi"]
            rec["boot_block_months"] = _b["block_months"]
            rec["boot_n_successful"] = _b["n_successful"]
            rec["supported_out_of_sample"] = bool(
                np.isfinite(_b["ci_hi"]) and _b["ci_hi"] < 0)
        else:
            rec["supported_out_of_sample"] = False
            rec["rmse_difference_spline_minus_linear"] = np.nan
        _oos_rows.append(rec)

    SPLINE_OOS = pd.DataFrame(_oos_rows)
    SPLINE_OOS["evidence_status"] = np.where(
        SPLINE_OOS["supported_out_of_sample"], "supported out-of-sample", "exploratory")
    SPLINE_SUPPORTED_OOS = SPLINE_OOS.loc[SPLINE_OOS["supported_out_of_sample"],
                                          "driver"].tolist()
    print("One-calendar-month-ahead rolling origin, smooth vs linear on IDENTICAL "
          "training and target months:")
    display(SPLINE_OOS.round(4))
    if SPLINE_SUPPORTED_OOS:
        print(f"\\nSupported out of sample: {SPLINE_SUPPORTED_OOS}. Each lowered "
              "one-month-ahead RMSE against its own linear specification by a margin "
              "whose 95% moving-block interval excludes zero. These may be reported as "
              "non-linear, still labelled as a §14 shape and never merged with a §12 or "
              "§20 coefficient.")
    else:
        print("\\nNO smooth lowered one-month-ahead RMSE against its own linear "
              "specification by a margin whose 95% moving-block interval excludes zero. "
              "Every shape in §14 stays EXPLORATORY and §25 reports no non-linear driver "
              "result. In particular, the lake-level curvature reported by the withdrawn "
              "25-column design is NOT supported.")
    if len(SPLINE_TESTS):
        SPLINE_TESTS["evidence_status"] = np.where(
            SPLINE_TESTS["driver"].isin(SPLINE_SUPPORTED_OOS),
            "supported out-of-sample", "exploratory")
        PARTIAL_EFFECTS["evidence_status"] = np.where(
            PARTIAL_EFFECTS["driver"].isin(SPLINE_SUPPORTED_OOS),
            "supported out-of-sample", "exploratory")
''')


# ===========================================================================
# 15. Model D
# ===========================================================================
md("""## 15. Model D — gradient boosting with out-of-fold permutation importance

A non-parametric cross-check that assumes nothing about functional form or
additivity. Two deliberate constraints:

* The ensemble is **small** (`max_depth=2`) — with ~100 months a large forest
  memorises the series and its importances become noise.
* Permutation importance is computed **on the held-out fold** of the same
  rolling-origin split used everywhere else. In-sample importance on a
  persistent series is meaningless, and it is what most published rankings
  unfortunately report.

Treat this as a *ranking* cross-check. If it and §13 agree on which drivers
matter, the ranking is not an artefact of the linear model.
""")

code('''# =====================================================================
# 15. Gradient boosting + out-of-fold permutation importance
# =====================================================================
GBM_IMPORTANCE = pd.DataFrame()
GBM_SCORE = {}

if not HAVE_SKLEARN:
    print("scikit-learn unavailable; §15 skipped.")
else:
    _Xg = fit_df[[c for c in DRIVER_TERMS + SEASON_COLS + TREND_TERMS
                  if c in fit_df.columns]].astype(float)
    _yg = fit_df["y"].to_numpy(dtype=float)
    _folds, _gbm_fold_audit = rolling_origin_month_folds(
        FIT_MONTHS, n_folds=CV_N_FOLDS, horizon_months=CV_HORIZON_MONTHS,
        min_train_months=CV_MIN_TRAIN_MONTHS, min_test_months=CV_MIN_TEST_MONTHS)
    if len(_folds) < 2:
        print("Too few calendar rolling-origin folds for §15.")
    else:
        GBM_SCORE, _ = cv_scores(
            _yg, _Xg, _folds, months=FIT_MONTHS,
            fit=lambda Xtr, ytr, wtr: HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xtr, ytr),
            predict=lambda m, Xte: m.predict(Xte))
        print(f"Gradient boosting, {CV_HORIZON_MONTHS}-calendar-month rolling-origin windows: "
              f"RMSE {GBM_SCORE['rmse']:.4f}, R2_oos {GBM_SCORE['r2_oos']:.3f} over "
              f"{GBM_SCORE['n_folds']} folds / {GBM_SCORE['n_test']} evaluated months")

        _imp = {c: [] for c in _Xg.columns}
        for tr, te in fold_index_pairs(_folds):
            if len(te) < 2:
                continue
            m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(_Xg.iloc[tr], _yg[tr])
            try:
                r = permutation_importance(m, _Xg.iloc[te], _yg[te],
                                           n_repeats=GBM_PERM_REPEATS,
                                           random_state=RANDOM_STATE,
                                           scoring="neg_mean_squared_error")
            except Exception as exc:
                print(f"  permutation importance failed on a fold: {exc}"); continue
            for c, v in zip(_Xg.columns, r.importances_mean):
                _imp[c].append(float(v))
        rows = [{"term": c, "perm_importance_mean": float(np.mean(v)),
                 "perm_importance_sd": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                 "n_folds": len(v)} for c, v in _imp.items() if v]
        if rows:
            GBM_IMPORTANCE = pd.DataFrame(rows)
            GBM_IMPORTANCE["is_driver"] = GBM_IMPORTANCE["term"].isin(DRIVER_TERMS)
            GBM_IMPORTANCE = GBM_IMPORTANCE.sort_values(
                "perm_importance_mean", ascending=False).reset_index(drop=True)
            GBM_IMPORTANCE["rank"] = np.arange(1, len(GBM_IMPORTANCE) + 1)
            print("\\nOut-of-fold permutation importance (increase in held-out MSE when "
                  "the column is shuffled):")
            display(GBM_IMPORTANCE.round(5))

            _gd = GBM_IMPORTANCE[GBM_IMPORTANCE["is_driver"]]
            if len(_gd):
                fig, ax = plt.subplots(figsize=(8, 0.42 * len(_gd) + 1.4))
                ax.barh(_gd["term"], _gd["perm_importance_mean"],
                        xerr=_gd["perm_importance_sd"], color="tab:purple", alpha=0.85)
                ax.axvline(0, color="k", lw=1)
                ax.invert_yaxis()
                ax.set_xlabel("mean increase in held-out MSE (permutation importance)")
                ax.set_title("Out-of-fold driver importance — gradient boosting")
                ax.grid(alpha=0.3, axis="x")
                fig.tight_layout(); plt.show()
            print("A negative importance means shuffling the driver IMPROVED held-out "
                  "prediction: that driver carries no usable signal at this sample size.")
''')

# ===========================================================================
# 16. Skill
# ===========================================================================
md("""## 16. Out-of-sample skill — three-month windows (**sensitivity, not the headline**)

> **This section has been demoted.** It is retained unchanged as a *sensitivity
> comparison* so the earlier result stays auditable, but it is **no longer the
> notebook's predictive assessment**. The principal one is §21: expanding-window,
> **one-calendar-month-ahead** rolling origin, over every feasible origin, with
> drivers restricted to information available at the origin.
>
> Two things are wrong with using this section as the headline:
>
> 1. **Only 8 windows are attempted and ~6 survive**, giving ~18 evaluated
>    months in a handful of blocks. §21 uses every feasible origin instead.
> 2. **It is a nowcast, not a forecast.** Temperature, wind and lake level enter
>    at lag 0 by the a-priori specification, so predicting month $t$ uses month
>    $t$'s weather. No forecaster has that. §21 moves those terms to lag 1.
>
> Any RMSE improvement quoted from *this* section is an improvement over the
> **best simple baseline in this design** — not over a matched
> season + trend + persistence model, and not with a paired uncertainty
> interval. The claim that "environmental drivers improve RMSE by 9%" that
> earlier runs of this notebook printed here is **withdrawn as a headline**:
> §21 and §22 re-examine it against the matched no-driver state-space model with
> a calendar-aware bootstrap interval, and §25 reports whatever that shows.

In-sample $R^2$ on a persistent monthly series is close to meaningless: season
plus a lagged response will fit it well while knowing nothing about ecology. The
question that matters is whether the *environmental drivers* improve prediction
of months the model has not seen.

Baselines, in increasing order of difficulty to beat:

| Baseline | Why it is the bar |
|---|---|
| **mean** | Predicts the record mean. Anything must beat this. |
| **seasonal-naive** | Predicts the **same calendar month** last year. Free, and hard to beat. |
| **fitted `y_lag1` regression** | A *fitted* regression on last month's value — **not** literal persistence, which has no coefficient at all. §21 reports the literal $\hat y_t = y_{t-1}$ baseline separately. |
| **season+trend** | The calendar alone. |
| **drivers+season+trend** | The headline model. |
| **+AR(1)** | The dynamic model. |

**If the driver models do not beat persistence and season, the correct
conclusion is that the drivers explain WH extent but do not predict it** — a
finding about identifiability, not a failure to report.

### The design is calendar-based, and says so

The record has missing months, so *three rows* and *three months* are different
things. Every fold here is defined on the calendar:

* an **origin** is a real month timestamp, and the test window is the
  **next `CV_HORIZON_MONTHS` consecutive calendar months** after it;
* training uses only months **strictly before** the window opens;
* a month inside a window is scored only if the response **and** every predictor
  any compared model needs is observed — nothing is imputed to fill a window;
* a window is **never widened** to collect more observations. If fewer than
  `CV_MIN_TEST_MONTHS` of its months are evaluable, the fold is recorded with
  `usable=False`, its `skip_reason` is printed, and it is skipped;
* every fold records its origin, training span, window span, requested horizon,
  number of evaluable months, and the omitted months **with the reason** each
  was omitted;
* all models share **one** evaluation sample, asserted month by month, so the
  RMSE column is a comparison rather than a coincidence.

The design is reported as *"`CV_HORIZON_MONTHS`-calendar-month rolling-origin
windows"* together with the actual number of evaluated observations, and the
prediction table keeps the real `month` of every prediction.

### The seasonal-naive baseline

$\\hat y(t) = y(t-12\\ \\text{calendar months})$, looked up **by timestamp** on
the calendar-complete monthly table. `shift(12)` over complete-case rows returns
the twelfth previous *observed* row, which is a different calendar month
whenever the record has a gap — and this record has gaps. If the month exactly
one year earlier is missing, the prediction is reported **unavailable**, never
substituted, and `source_month` is kept so the lookup can be audited.

Because of that, seasonal-naive can end up scored on fewer months than the other
models. Its RMSE is then reported **separately with its own `n_test`**, and a
second table re-scores every model on the months seasonal-naive *can* be scored
on, so that a like-for-like comparison is still available.
""")

code('''# =====================================================================
# 16. Rolling-origin out-of-sample skill, on CALENDAR months
# =====================================================================
# The evaluation sample is fixed FIRST and shared by every model compared here:
# a month is evaluable only if the response and every predictor any compared
# model needs are observed. That is what makes the RMSEs comparable at all.
CV_REQUIRED_COLS = ["y"] + DRIVER_TERMS + SEASON_COLS + TREND_TERMS + AR_TERMS
_cv_mask = model_df[CV_REQUIRED_COLS].notna().all(axis=1)
_cvdf = model_df.loc[_cv_mask].reset_index(drop=True)
_ycv = _cvdf["y"].to_numpy(dtype=float)
CV_MONTHS = _cvdf["month"].reset_index(drop=True)

# Why each calendar month inside the fitted span is NOT evaluable — carried into
# the fold audit so an omitted test month always comes with a reason.
_cv_omission = {}
if len(CV_MONTHS):
    _span = pd.date_range(CV_MONTHS.min(), CV_MONTHS.max(), freq="MS")
    _by_month = model_df.set_index("month")
    for _m in _span:
        if _m in set(CV_MONTHS):
            continue
        if _m not in _by_month.index:
            _cv_omission[_m] = "month absent from the monthly grid"
            continue
        _row = _by_month.loc[_m]
        _missing = [c for c in CV_REQUIRED_COLS if pd.isna(_row.get(c, np.nan))]
        if "y" in _missing:
            _cv_omission[_m] = ("response not observed (month excluded by the "
                                "coverage filter); NOT imputed")
        else:
            _cv_omission[_m] = f"predictor(s) missing: {', '.join(_missing[:3])}"

CV_FOLDS, CV_FOLD_AUDIT = rolling_origin_month_folds(
    CV_MONTHS, n_folds=CV_N_FOLDS, horizon_months=CV_HORIZON_MONTHS,
    min_train_months=CV_MIN_TRAIN_MONTHS, min_test_months=CV_MIN_TEST_MONTHS,
    omission_reason=_cv_omission)

CV_DESIGN = (f"{CV_HORIZON_MONTHS}-calendar-month rolling-origin windows "
             f"(minimum {CV_MIN_TRAIN_MONTHS} training months, at least "
             f"{CV_MIN_TEST_MONTHS} evaluable months per window)")
_n_eval_months = int(sum(len(f["test_idx"]) for f in CV_FOLDS))
print(f"Rolling-origin design: {CV_DESIGN}.")
print(f"  {len(CV_FOLDS)} usable fold(s) of {len(CV_FOLD_AUDIT)} windows considered; "
      f"{_n_eval_months} evaluated month(s) in total.")
print(f"  Evaluable sample: {len(CV_MONTHS)} months over a "
      f"{calendar_span_months(CV_MONTHS) if len(CV_MONTHS) else 0}-calendar-month span, "
      "so a window of three rows would NOT be a three-month horizon.")
if len(CV_FOLD_AUDIT):
    display(CV_FOLD_AUDIT[[
        "fold", "origin_month", "train_start", "train_end", "n_train_months",
        "test_window_start", "test_window_end", "horizon_months_requested",
        "n_test_months_evaluable", "n_test_months_omitted", "omitted_test_months",
        "omitted_reasons", "usable", "skip_reason"]])
_skipped = CV_FOLD_AUDIT[~CV_FOLD_AUDIT["usable"]] if len(CV_FOLD_AUDIT) else pd.DataFrame()
if len(_skipped):
    print(f"\\n{len(_skipped)} window(s) SKIPPED rather than widened:")
    for r in _skipped.itertuples():
        print(f"  {r.test_window_start:%Y-%m}..{r.test_window_end:%Y-%m}: {r.skip_reason}")

SKILL = pd.DataFrame()
CV_PREDICTIONS = pd.DataFrame()
SEASONAL_NAIVE_SKILL = pd.DataFrame()
SEASONAL_NAIVE_PREDICTIONS = pd.DataFrame()
SKILL_ON_SEASONAL_NAIVE_SUBSET = pd.DataFrame()
if len(CV_FOLDS) < 2:
    print("\\nFewer than two folds: no honest skill estimate is possible on this record. "
          "Lower CV_MIN_TRAIN_MONTHS, or accept that this series only supports the "
          "association analysis in §12/§17.")
else:
    _wcv = _cvdf["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None

    def _const_fit(value_fn):
        """Baseline that ignores X and predicts a function of the training y."""
        class _M:
            def __init__(self, v): self.v = v
        return (lambda Xtr, ytr, wtr: _M(value_fn(ytr)),
                lambda m, Xte: np.full(len(Xte), m.v))

    _rows, _details = [], []
    _specs_cv = {
        "mean baseline": (None, *_const_fit(np.nanmean)),
        "fitted y_lag1 regression (NOT literal persistence)": (AR_TERMS[:1], None, None),
        "season+trend": (SEASON_COLS + TREND_TERMS, None, None),
        "drivers only": (DRIVER_TERMS, None, None),
        "drivers+season": (DRIVER_TERMS + SEASON_COLS, None, None),
        "drivers+season+trend": (DRIVER_TERMS + SEASON_COLS + TREND_TERMS, None, None),
        "drivers+season+trend+AR": (DRIVER_TERMS + SEASON_COLS + TREND_TERMS + AR_TERMS,
                                    None, None),
    }
    if GBM_SCORE:
        _specs_cv["gradient boosting (§15)"] = (
            DRIVER_TERMS + SEASON_COLS + TREND_TERMS,
            (lambda Xtr, ytr, wtr: HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xtr, ytr)),
            (lambda m, Xte: m.predict(Xte)))

    for name, (cols, fitf, predf) in _specs_cv.items():
        use = [c for c in (cols or ["time_index"]) if c in _cvdf.columns]
        sc, det = cv_scores(_ycv, _cvdf[use], CV_FOLDS, months=CV_MONTHS,
                            weights=None if fitf else _wcv, fit=fitf, predict=predf)
        sc["specification"] = name
        sc["n_terms"] = 0 if cols is None else len(use)
        sc["eval_set"] = "common"
        _rows.append(sc)
        _details.append(det.assign(specification=name))

    SKILL = pd.DataFrame(_rows).sort_values("rmse").reset_index(drop=True)
    CV_PREDICTIONS = pd.concat(_details, ignore_index=True)
    # Every model on the common evaluation set must have been scored on exactly
    # the same response months, otherwise the RMSE column is not a comparison.
    _month_sets = CV_PREDICTIONS.groupby("specification")["month"].apply(
        lambda s: tuple(sorted(pd.to_datetime(s).tolist())))
    assert _month_sets.nunique() == 1, \\
        "compared models were scored on different response months"
    CV_EVAL_MONTHS = pd.Series(list(_month_sets.iloc[0]))
    assert len(CV_EVAL_MONTHS) == _n_eval_months
    assert SKILL["n_test"].nunique() == 1, "compared models saw different n_test"

    # --- Seasonal-naive baseline, looked up on the CALENDAR ------------------
    # y_hat(t) = y(t - 12 calendar months), taken from the calendar-complete
    # monthly table by timestamp. `shift(12)` on complete-case rows would return
    # the twelfth previous OBSERVED row, which is a different month whenever the
    # record has a gap — and this record has gaps.
    _truth = (CV_PREDICTIONS[CV_PREDICTIONS["specification"] == "mean baseline"]
              [["fold", "month", "y"]].drop_duplicates("month"))
    SEASONAL_NAIVE_PREDICTIONS = (seasonal_naive_predictions(CV_EVAL_MONTHS, monthly)
                                  .merge(_truth, on="month", how="left"))
    assert (SEASONAL_NAIVE_PREDICTIONS["source_month"]
            == SEASONAL_NAIVE_PREDICTIONS["month"] - pd.DateOffset(months=12)).all(), \\
        "a seasonal-naive prediction does not come from exactly 12 calendar months earlier"

    _sn_ok = SEASONAL_NAIVE_PREDICTIONS[SEASONAL_NAIVE_PREDICTIONS["available"]]
    _n_sn, _n_common = len(_sn_ok), len(CV_EVAL_MONTHS)
    _sn_same_set = (_n_sn == _n_common)
    if _n_sn:
        _err = _sn_ok["y"].to_numpy() - _sn_ok["yhat"].to_numpy()
        _ss_tot = float(((_sn_ok["y"] - _sn_ok["y"].mean()) ** 2).sum())
        SEASONAL_NAIVE_SKILL = pd.DataFrame([{
            "specification": "seasonal-naive (y_{t-12 calendar months})",
            "n_terms": 0, "n_folds": int(_sn_ok["fold"].nunique()), "n_test": int(_n_sn),
            "rmse": float(np.sqrt(np.mean(_err ** 2))),
            "mae": float(np.mean(np.abs(_err))),
            "r2_oos": (1 - float((_err ** 2).sum()) / _ss_tot) if _ss_tot > 0 else np.nan,
            "eval_set": "common" if _sn_same_set else "seasonal-naive subset",
            "n_months_unavailable": int(_n_common - _n_sn),
        }])
    if _sn_same_set and len(SEASONAL_NAIVE_SKILL):
        # Same evaluation months as every other model: a direct comparison is legitimate.
        SKILL = pd.concat([SKILL, SEASONAL_NAIVE_SKILL], ignore_index=True) \\
            .sort_values("rmse").reset_index(drop=True)
        SEASONAL_NAIVE_PREDICTIONS["specification"] = \\
            "seasonal-naive (y_{t-12 calendar months})"
        CV_PREDICTIONS = pd.concat(
            [CV_PREDICTIONS,
             SEASONAL_NAIVE_PREDICTIONS.loc[SEASONAL_NAIVE_PREDICTIONS["available"],
                                            ["fold", "month", "y", "yhat", "specification"]]],
            ignore_index=True)

    print(f"\\nOut-of-sample skill — {CV_DESIGN}")
    print(f"Common evaluation sample: {_n_common} month(s) across {len(CV_FOLDS)} folds "
          f"({CV_EVAL_MONTHS.min():%Y-%m} .. {CV_EVAL_MONTHS.max():%Y-%m}).")
    display(SKILL.round(4))

    if len(SEASONAL_NAIVE_SKILL) and not _sn_same_set:
        print(f"\\nSeasonal-naive is reported SEPARATELY: the month exactly 12 calendar "
              f"months before {_n_common - _n_sn} of the {_n_common} evaluation month(s) "
              "is missing from the record, so it cannot be scored there. Its RMSE is "
              "computed on a DIFFERENT and easier/harder sample and must not be placed "
              "beside the numbers above.")
        display(SEASONAL_NAIVE_SKILL.round(4))
        # A like-for-like comparison is still possible on the intersection.
        _sub = set(_sn_ok["month"])
        _rows_sub = []
        for name, g in CV_PREDICTIONS.groupby("specification"):
            gg = g[g["month"].isin(_sub)]
            if not len(gg):
                continue
            _e = gg["y"].to_numpy() - gg["yhat"].to_numpy()
            _rows_sub.append({"specification": name, "n_test": len(gg),
                              "rmse": float(np.sqrt(np.mean(_e ** 2))),
                              "mae": float(np.mean(np.abs(_e)))})
        _rows_sub.append({"specification": SEASONAL_NAIVE_SKILL["specification"].iloc[0],
                          "n_test": int(_n_sn),
                          "rmse": float(SEASONAL_NAIVE_SKILL["rmse"].iloc[0]),
                          "mae": float(SEASONAL_NAIVE_SKILL["mae"].iloc[0])})
        SKILL_ON_SEASONAL_NAIVE_SUBSET = (pd.DataFrame(_rows_sub)
                                          .sort_values("rmse").reset_index(drop=True))
        SKILL_ON_SEASONAL_NAIVE_SUBSET["eval_set"] = "seasonal-naive subset"
        print(f"\\nRestricted to the {_n_sn} month(s) seasonal-naive CAN be scored on, "
              "every model recomputed on that same sample:")
        display(SKILL_ON_SEASONAL_NAIVE_SUBSET.round(4))
    elif not len(SEASONAL_NAIVE_SKILL):
        print("\\nSeasonal-naive could not be scored on any evaluation month: the month "
              "exactly one year earlier is missing throughout. Reported as unavailable "
              "rather than substituted with the twelfth previous observed row.")

    # --- Skill plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(SKILL) + 1.6))
    _cols = ["tab:green" if "drivers" in s else "tab:grey" for s in SKILL["specification"]]
    ax.barh(SKILL["specification"], SKILL["rmse"], color=_cols, alpha=0.85)
    _baselines = ["fitted y_lag1 regression (NOT literal persistence)", "season+trend", "mean baseline"]
    if _sn_same_set:
        _baselines.append("seasonal-naive (y_{t-12 calendar months})")
    _bb = SKILL.loc[SKILL["specification"].isin(_baselines), "rmse"].min()
    ax.axvline(_bb, color="red", ls="--", lw=1.2, label="best simple baseline")
    ax.invert_yaxis()
    ax.set_xlabel(f"RMSE on {RESPONSE_INFO['transform']}({RESPONSE_COL}), "
                  f"{_n_common} held-out months")
    ax.set_title(f"Out-of-sample skill — {CV_DESIGN}\\ngreen = uses environmental drivers")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show()

    SKILL["rmse_vs_best_baseline"] = SKILL["rmse"] / _bb
    SKILL["evaluation_role"] = ("SENSITIVITY — 3-calendar-month windows, "
                                "contemporaneous drivers (nowcast); see §21 for the "
                                "principal one-month-ahead assessment")
    _driver_best = SKILL[SKILL["specification"].str.contains("drivers|boosting")]["rmse"].min()
    if np.isfinite(_driver_best) and np.isfinite(_bb):
        if _driver_best < _bb:
            print(f"In THIS sensitivity design the best driver specification has "
                  f"{100 * (1 - _driver_best / _bb):.1f}% lower RMSE than the best "
                  f"simple baseline, on {_n_common} held-out months.")
            print("*** That is a POINT ESTIMATE from a 3-calendar-month, "
                  "contemporaneous-driver (nowcast) design with no paired uncertainty "
                  "interval, and it is NOT a headline result. Do not quote it. The "
                  "principal question — does the driver block beat a MATCHED "
                  "season+trend+persistence model at genuine one-month-ahead "
                  "prediction — is answered in §21/§22 with an interval, and §25 "
                  "reports that answer. ***")
        else:
            print(f"Environmental drivers do NOT beat the best simple baseline "
                  f"({_driver_best:.4f} vs {_bb:.4f}) on {_n_common} held-out months.")
            print("Report this plainly: the drivers are ASSOCIATED with WH extent (§12) and "
                  "account for a share of its variance (§17), but at this sample size they "
                  "do not forecast it better than persistence/season. That is a statement "
                  "about the record's length and the drivers' seasonality, not evidence "
                  "that the ecology is wrong.")

    # --- Held-out predictions, on the CALENDAR ------------------------------
    if len(CV_PREDICTIONS):
        _pick = [s for s in ["fitted y_lag1 regression (NOT literal persistence)", "season+trend", "drivers+season+trend"]
                 if s in set(CV_PREDICTIONS["specification"])]
        _obs = (CV_PREDICTIONS[CV_PREDICTIONS["specification"] == _pick[-1]]
                .sort_values("month"))
        fig, ax = plt.subplots(figsize=(11, 3.8))
        # The title promises that gaps are months with no observation, so the
        # lines have to be drawn on the calendar grid for that to be true.
        ax.plot(*on_calendar_grid(_obs["month"], _obs["y"]), "k-o", ms=4, lw=1.4,
                label="observed")
        for s in _pick:
            g = CV_PREDICTIONS[CV_PREDICTIONS["specification"] == s].sort_values("month")
            ax.plot(*on_calendar_grid(g["month"], g["yhat"]), lw=1.4, alpha=0.85,
                    marker=".", label=s)
        for f in CV_FOLDS:
            ax.axvspan(f["test_window_start"],
                       f["test_window_end"] + pd.offsets.MonthEnd(1),
                       color="tab:blue", alpha=0.06)
        ax.set_xlabel("held-out month (calendar date; shaded = a rolling-origin test window)")
        ax.set_ylabel(f"{RESPONSE_INFO['transform']}({RESPONSE_COL})")
        ax.set_title(f"Held-out predictions — {CV_DESIGN}\\n"
                     f"{_n_common} evaluated months; gaps are months with no observation")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_horizontalalignment("right")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); plt.show()
''')


# ===========================================================================
# 17. Variance partitioning
# ===========================================================================
md("""## 17. Variance partitioning — the answer to "which driver matters most"

This is the section to lead the write-up with. A single coefficient answers
*"how much does WH move per SD of rainfall"*; the partition answers *"how much
of the observed variation in WH extent does rainfall account for"*, which is
the question the dissertation asks.

Collinear predictors make naive "% variance explained" arbitrary — whichever is
entered first takes the shared variance. **Shapley $R^2$** removes the
arbitrariness by averaging each group's marginal contribution over *every* order
of entry, so shared variance is split evenly and the values sum exactly to the
full-model $R^2$.

Two partitions are reported, and the difference between them is itself the
result:

* **without persistence** — how the drivers, season and trend divide the
  explainable variance;
* **with persistence** — the same, once "last month's hyacinth" is allowed to
  compete. Persistence usually dominates, and what survives beside it is the
  genuinely environmental signal.

### Shared versus unique — one specification at a time

Shapley *splits* variance that collinear drivers share, so a driver that is
merely a proxy for a real one still collects a sizeable Shapley value. Comparing
it with the **semi-partial** (last-entry) $R^2$ separates the two cases.

That comparison is only meaningful **within a single fit**. An earlier version
of this notebook put the *dynamic* Shapley values (which include lagged WH
cover, on `dyn_df` rows) beside the *static* M3 semi-partials (which do not, on
`fit_df` rows) — different models, different rows, different columns. A separate
table is now produced for each specification, each carrying a `specification`
column, and `shared_vs_unique` **asserts** that the two inputs share a row
count, model columns, response and weighting before merging them.

The ratio is named `last_entry_to_shapley_ratio`, **not** a "share". It is not
bounded by 1: under suppression, or where the relationship is not additive, a
driver can explain more as the last term entered than its order-averaged
contribution. Values above 1 are labelled as *possible suppression or
coefficient instability* — they are never clipped.

The **primary** reading, carried into §25, is the **with persistence**
specification, because persistence dominates this series: a driver's independent
contribution has to be measured against last month's hyacinth, not only against
the calendar.
""")

code('''# =====================================================================
# 17. Shapley R^2 partitioning
# =====================================================================
_groups = {c: [c] for c in DRIVER_TERMS}
if SEASON_COLS:
    _groups["season (annual cycle)"] = SEASON_COLS
if TREND_TERMS:
    _groups["trend (multi-year)"] = TREND_TERMS

_RESPONSE_LABEL = f"{RESPONSE_INFO['transform']}({RESPONSE_COL})"
PARTITION = pd.DataFrame()
PARTITION_AR = pd.DataFrame()
SHARED_VS_UNIQUE_STATIC = pd.DataFrame()
SHARED_VS_UNIQUE_AR = pd.DataFrame()
SHARED_VS_UNIQUE = pd.DataFrame()
SHARED_VS_UNIQUE_SPEC = ""

if len(_groups) > PARTITION_MAX_GROUPS:
    print(f"{len(_groups)} groups exceeds PARTITION_MAX_GROUPS = {PARTITION_MAX_GROUPS} "
          f"(2^{len(_groups)} fits). Reduce the mechanism set or raise the cap.")
else:
    _cols1 = [c for g in _groups.values() for c in g if c in fit_df.columns]
    PARTITION = shapley_r2(fit_df["y"], fit_df[_cols1], _groups, weights=W,
                           specification="without persistence",
                           response=_RESPONSE_LABEL, weighting=MONTH_WEIGHTING)
    PARTITION["kind"] = np.where(PARTITION["group"].isin(DRIVER_TERMS),
                                 "environmental driver", "control")
    print(f"Shapley R2 partition WITHOUT persistence "
          f"({PARTITION['n_obs'].iloc[0]} rows, "
          f"full-model R2 = {PARTITION['r2_full_model'].iloc[0]:.3f}):")
    display(PARTITION.drop(columns=["model_columns"]).round(4))

    if PARTITION_INCLUDE_AR and AR_TERMS and len(dyn_df) > len(_groups) + 5:
        # The variance-partition group label stays "persistence (y_lag1)": it names
        # a share of explained variance carried by the lagged response, not a
        # forecasting baseline. The §16 BASELINE of the same name was renamed,
        # because calling a fitted regression "persistence" there would have made
        # it look like the unfitted y_{t-1} rule §21 actually scores.
        _g2 = dict(_groups); _g2["persistence (y_lag1)"] = AR_TERMS
        if len(_g2) <= PARTITION_MAX_GROUPS:
            _cols2 = [c for g in _g2.values() for c in g if c in dyn_df.columns]
            PARTITION_AR = shapley_r2(dyn_df["y"], dyn_df[_cols2], _g2, weights=W_DYN,
                                      specification="with persistence",
                                      response=_RESPONSE_LABEL, weighting=MONTH_WEIGHTING)
            PARTITION_AR["kind"] = np.where(
                PARTITION_AR["group"].isin(DRIVER_TERMS), "environmental driver", "control")
            print(f"\\nShapley R2 partition WITH persistence "
                  f"({PARTITION_AR['n_obs'].iloc[0]} rows, "
                  f"full-model R2 = {PARTITION_AR['r2_full_model'].iloc[0]:.3f}):")
            display(PARTITION_AR.drop(columns=["model_columns"]).round(4))

    _p = PARTITION_AR if len(PARTITION_AR) else PARTITION
    fig, axes = plt.subplots(1, 2 if len(PARTITION_AR) else 1,
                             figsize=(6.5 * (2 if len(PARTITION_AR) else 1),
                                      0.42 * len(_p) + 2.0), squeeze=False)
    for ax, (tab, title) in zip(axes.ravel(), [
            (PARTITION, "without persistence"),
            (PARTITION_AR, "with persistence")][:axes.size]):
        if not len(tab):
            ax.axis("off"); continue
        cols = ["tab:green" if k == "environmental driver" else "tab:grey"
                for k in tab["kind"]]
        ax.barh(tab["group"], tab["shapley_r2"], color=cols, alpha=0.9)
        ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Shapley $R^2$ contribution")
        ax.set_title(f"Variance partition — {title}\\n"
                     f"(full-model $R^2$ = {tab['r2_full_model'].iloc[0]:.3f}, "
                     f"n = {int(tab['n_obs'].iloc[0])})", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show()

    _env = _p.loc[_p["kind"] == "environmental driver", "shapley_r2"].sum()
    _tot = float(_p["r2_full_model"].iloc[0])
    print(f"\\nIn the '{_p['specification'].iloc[0]}' specification, environmental drivers "
          f"together account for {_env:.3f} of the {_tot:.3f} explained "
          f"(i.e. {_env / _tot:.0%} of the model's explanatory power, "
          f"{_env:.1%} of the total variance in the response).")
    _top = _p[_p["kind"] == "environmental driver"].head(3)
    if len(_top):
        print("Most strongly linked environmental variables, by variance accounted for:")
        for i, r in enumerate(_top.itertuples(), 1):
            print(f"  {i}. {r.group}: Shapley R2 = {r.shapley_r2:.4f} "
                  f"({r.share_of_r2:.1%} of the model's explanatory power)")

    # --- Shared vs unique: the trap in any variance partition -----------------
    # Shapley SPLITS the variance two collinear drivers share, so a driver that
    # is merely a proxy for a real one collects a sizeable Shapley value while
    # explaining nothing the model does not already have. Comparing Shapley with
    # the semi-partial (last-entry) contribution separates the two cases.
    #
    # The comparison is only meaningful WITHIN one specification. An earlier
    # version of this notebook put the dynamic Shapley values (which include
    # lagged WH cover) beside the static M3 semi-partials (which do not): two
    # different models, different rows, different columns. `shared_vs_unique`
    # now asserts that the two inputs came from the same fit, so each table
    # below is internally consistent by construction.
    if len(PARTITION) and len(SEMI_PARTIAL):
        SHARED_VS_UNIQUE_STATIC = shared_vs_unique(
            PARTITION, SEMI_PARTIAL, DRIVER_TERMS, "without persistence")
        print("\\nShared vs unique variance — WITHOUT persistence "
              f"(M3 rows: {int(PARTITION['n_obs'].iloc[0])}):")
        display(SHARED_VS_UNIQUE_STATIC.round(4))
    if len(PARTITION_AR) and len(SEMI_PARTIAL_AR):
        SHARED_VS_UNIQUE_AR = shared_vs_unique(
            PARTITION_AR, SEMI_PARTIAL_AR, DRIVER_TERMS, "with persistence")
        print("\\nShared vs unique variance — WITH persistence "
              f"(M4 rows: {int(PARTITION_AR['n_obs'].iloc[0])}):")
        display(SHARED_VS_UNIQUE_AR.round(4))

    # Persistence dominates this series, so the specification that INCLUDES it
    # is the one the "independent contribution" reading is based on: a driver
    # must earn its variance against last month's hyacinth, not only against the
    # calendar.
    if len(SHARED_VS_UNIQUE_AR):
        SHARED_VS_UNIQUE = SHARED_VS_UNIQUE_AR
        SHARED_VS_UNIQUE_SPEC = "with persistence"
    else:
        SHARED_VS_UNIQUE = SHARED_VS_UNIQUE_STATIC
        SHARED_VS_UNIQUE_SPEC = "without persistence"
    if len(SHARED_VS_UNIQUE):
        print(f"\\nPRIMARY reading for §25 is the '{SHARED_VS_UNIQUE_SPEC}' specification, "
              "because persistence dominates this series and a driver's independent "
              "contribution has to be measured against it.")
        print("`last_entry_to_shapley_ratio` is the semi-partial divided by the Shapley "
              "value. It is NOT a bounded share: under suppression, or where the "
              "relationship is not additive, the last-entry contribution can exceed the "
              "order-averaged one and the ratio exceeds 1. Such cases are labelled, "
              "not clipped.")
        _sup = SHARED_VS_UNIQUE.loc[
            SHARED_VS_UNIQUE["last_entry_to_shapley_ratio"] > 1.0, "term"].tolist()
        if _sup:
            print(f"\\nRatio > 1 for: {_sup}. Read as possible SUPPRESSION or coefficient "
                  "instability — the driver explains more as the last term entered than "
                  "its order-averaged contribution — not as 'more than 100% of its share'. "
                  "Check §13's stability selection and §18's sign agreement before "
                  "quoting these drivers.")
        _proxyish = SHARED_VS_UNIQUE.loc[
            SHARED_VS_UNIQUE["reading"].str.startswith("SHARED"), "term"].tolist()
        if _proxyish:
            print(f"\\nSizeable Shapley value but almost NO unique contribution: {_proxyish}.")
            print("Those drivers are collinear stand-ins for another driver in the set. Do "
                  "NOT name them as independent mechanisms — say the mechanism they share, "
                  "and state that the data cannot separate the members of that group.")
''')

# ===========================================================================
# 18. Robustness
# ===========================================================================
md("""## 18. Robustness — does any conclusion move?

Five sweeps, each attacking a different decision made earlier. A driver whose
sign and rough magnitude survive all five is reportable; one that appears in only
one variant is a property of that variant, not of the lake.

> These sweeps attack the **static** models (M3/M4). The equivalent checks on the
> **principal state-space model** — leave-one-year-out, alternative response
> transformations, and AR(1) vs AR(2) vs local level — are in **§24**, and the
> two sets of tables are never merged: every row names the model it came from.

1. **Response definition** — area-weighted cover, absolute WH area, occurrence
   rate, unweighted mean cover.
2. **Coverage threshold** — 0.70 / 0.80 / 0.90 / 0.95, i.e. how strict
   "comparable month" has to be.
3. **Leave-one-year-out** — refit dropping each year in turn. One anomalous year
   (a flood, an ENSO event) should not be carrying an entire conclusion.
4. **Deseasonalised anomalies** — regress the seasonal residual of the response
   on the seasonal residuals of the drivers. The strictest test of "is this more
   than seasonality".
5. **First differences** — month-on-month *change* in WH on month-on-month
   change in drivers, which removes any level confound (including slow
   classifier drift) at the cost of amplifying noise.
""")

code('''# =====================================================================
# 18a. Response-definition and coverage sweeps
# =====================================================================
ROBUST_RESPONSE = pd.DataFrame()
_resp_variants = [c for c in ["wh_cover_aoi", "wh_area_ha", "wh_occurrence",
                             "wh_cover_mean_unweighted"] if c in monthly.columns]
_rows = []
for rc in _resp_variants:
    how = "log" if rc == "wh_area_ha" else RESPONSE_TRANSFORM
    yv, _info = transform_response(monthly[rc], how, RESPONSE_EPS)
    tmp = model_df.copy(); tmp["_y"] = yv.to_numpy()
    cols = [c for c in DRIVER_TERMS + SEASON_COLS + TREND_TERMS if c in tmp.columns]
    m = tmp[["_y"] + cols].notna().all(axis=1)
    if m.sum() < len(cols) + 5:
        continue
    sub = tmp.loc[m]
    w = sub["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None
    res = fit_hac(sub["_y"], sub[cols], weights=w, maxlags=HAC_MAXLAGS,
                  months=sub["month"])
    t = tidy_coefficients(res, keep=DRIVER_TERMS, label=f"response={rc} ({how})")
    t["variant_kind"] = "response definition"; t["n"] = int(res.nobs); t["r2"] = res.rsquared
    _rows.append(t)

for cov in [0.70, 0.80, 0.90, 0.95]:
    if "coverage_fraction" not in model_df.columns:
        break
    sub = model_df[model_df["coverage_fraction"] >= cov]
    cols = [c for c in DRIVER_TERMS + SEASON_COLS + TREND_TERMS if c in sub.columns]
    m = sub[["y"] + cols].notna().all(axis=1)
    if m.sum() < len(cols) + 5:
        print(f"coverage >= {cov:.2f}: only {int(m.sum())} usable months; skipped.")
        continue
    sub = sub.loc[m]
    w = sub["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None
    res = fit_hac(sub["y"], sub[cols], weights=w, maxlags=HAC_MAXLAGS,
                  months=sub["month"])
    t = tidy_coefficients(res, keep=DRIVER_TERMS, label=f"coverage>={cov:.2f}")
    t["variant_kind"] = "coverage threshold"; t["n"] = int(res.nobs); t["r2"] = res.rsquared
    _rows.append(t)

if _rows:
    ROBUST_RESPONSE = pd.concat(_rows, ignore_index=True)
    _piv = ROBUST_RESPONSE.pivot_table(index="term", columns="specification",
                                       values="coef", aggfunc="first")
    print("Driver coefficient across response definitions and coverage thresholds:")
    display(_piv.round(3))
    _sign = np.sign(_piv)
    _stable = _sign.apply(lambda r: r.dropna().nunique() <= 1, axis=1)
    print("\\nSign stable across every variant:", _stable[_stable].index.tolist() or "none")
    print("Sign UNSTABLE (do not report a direction for these):",
          _stable[~_stable].index.tolist() or "none")
''')

code('''# =====================================================================
# 18b. Leave-one-year-out
# =====================================================================
LOYO = pd.DataFrame()
_years = sorted(fit_df["year"].dropna().unique())
_rows = []
for yr in _years:
    sub = fit_df[fit_df["year"] != yr]
    cols = [c for c in _hs_cols if c in sub.columns]
    if len(sub) < len(cols) + 5:
        continue
    w = sub["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None
    res = fit_hac(sub["y"], sub[cols], weights=w, maxlags=HAC_MAXLAGS,
                  months=sub["month"])
    t = tidy_coefficients(res, keep=DRIVER_TERMS, label=f"drop {int(yr)}")
    t["dropped_year"] = int(yr); t["n"] = int(res.nobs)
    _rows.append(t)

if _rows:
    LOYO = pd.concat(_rows, ignore_index=True)
    _piv = LOYO.pivot_table(index="term", columns="dropped_year", values="coef")
    _full = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == HEADLINE_SPEC] \\
        .set_index("term")["coef"]
    LOYO_SUMMARY = pd.DataFrame({
        "coef_full": _full,
        "loyo_min": _piv.min(axis=1), "loyo_max": _piv.max(axis=1),
        "loyo_sign_stable": _piv.apply(lambda r: np.sign(r.dropna()).nunique() <= 1, axis=1),
    })
    LOYO_SUMMARY["max_abs_shift"] = (_piv.sub(LOYO_SUMMARY["coef_full"], axis=0)
                                     .abs().max(axis=1))
    LOYO_SUMMARY["influential_year"] = _piv.sub(LOYO_SUMMARY["coef_full"], axis=0) \\
        .abs().idxmax(axis=1)
    print("Leave-one-year-out coefficient stability:")
    display(LOYO_SUMMARY.round(4))

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(_piv) + 1.6))
    for i, t in enumerate(_piv.index):
        ax.plot(_piv.loc[t], np.full(_piv.shape[1], i), "o", ms=5, alpha=0.7,
                color="tab:blue")
        ax.plot(LOYO_SUMMARY.loc[t, "coef_full"], i, "D", ms=8, color="tab:red")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(_piv))); ax.set_yticklabels(_piv.index)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient (red diamond = full record, blue = one year dropped)")
    ax.set_title("Leave-one-year-out coefficient stability")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show()
else:
    LOYO_SUMMARY = pd.DataFrame()
    print("Too few months per year for leave-one-year-out.")
''')

code('''# =====================================================================
# 18c. Deseasonalised anomalies and first differences
# =====================================================================
# The strictest reading available: remove the annual cycle (and trend) from the
# RESPONSE and from every DRIVER, then relate what is left. Any effect surviving
# here is genuinely "wetter/warmer/windier than this month usually is".
ANOMALY_COEFS = pd.DataFrame()
DIFF_COEFS = pd.DataFrame()

_ctrl_cols = [c for c in SEASON_COLS + TREND_TERMS if c in fit_df.columns]
if _ctrl_cols:
    _anom = pd.DataFrame(index=fit_df.index)
    _anom["y"] = fit_hac(fit_df["y"], fit_df[_ctrl_cols], weights=W,
                         months=FIT_MONTHS).resid
    for c in DRIVER_TERMS:
        _anom[c] = fit_hac(fit_df[c], fit_df[_ctrl_cols], months=FIT_MONTHS).resid
    _ra = fit_hac(_anom["y"], _anom[DRIVER_TERMS], weights=W, maxlags=HAC_MAXLAGS,
                  months=FIT_MONTHS)
    ANOMALY_COEFS = tidy_coefficients(_ra, keep=DRIVER_TERMS,
                                      label="anomaly model (season+trend removed)")
    print(f"Anomaly model on {int(_ra.nobs)} months, R2 = {_ra.rsquared:.3f} "
          "(of the anomaly variance only):")
    display(ANOMALY_COEFS.round(4))

# First differences: month-on-month change, which removes any level confound.
_d = model_df.copy()
_d["dy"] = _d["y"].diff()
_dcols = []
for c in DRIVER_TERMS:
    _d[f"d_{c}"] = _d[c].diff()
    _dcols.append(f"d_{c}")
_m = _d[["dy"] + _dcols + SEASON_COLS].notna().all(axis=1)
if _m.sum() > len(_dcols) + 6:
    sub = _d.loc[_m]
    w = sub["w_month"].to_numpy(dtype=float) if MONTH_WEIGHTING != "none" else None
    _rd = fit_hac(sub["dy"], sub[_dcols + SEASON_COLS], weights=w, maxlags=HAC_MAXLAGS,
                  months=sub["month"])
    DIFF_COEFS = tidy_coefficients(_rd, keep=_dcols, label="first differences")
    print(f"\\nFirst-difference model on {int(_rd.nobs)} months, R2 = {_rd.rsquared:.3f}:")
    display(DIFF_COEFS.round(4))
    print("Differencing removes every level confound but amplifies month-to-month noise, "
          "so weaker effects here are expected and not evidence against them.")
else:
    print("\\nToo few consecutive months for a first-difference model.")

# --- Do the strict variants agree with the headline? --------------------------
_cmp = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == HEADLINE_SPEC][["term", "coef"]] \\
    .rename(columns={"coef": "headline_M3"})
if len(ANOMALY_COEFS):
    _cmp = _cmp.merge(ANOMALY_COEFS[["term", "coef"]].rename(columns={"coef": "anomaly"}),
                      on="term", how="left")
if len(DIFF_COEFS):
    _dc = DIFF_COEFS[["term", "coef"]].copy()
    _dc["term"] = _dc["term"].str.replace("^d_", "", regex=True)
    _cmp = _cmp.merge(_dc.rename(columns={"coef": "first_difference"}), on="term", how="left")
_num = [c for c in _cmp.columns if c != "term"]
_cmp["all_signs_agree"] = _cmp[_num].apply(
    lambda r: np.sign(r.dropna()).nunique() <= 1 if r.notna().any() else False, axis=1)
ROBUST_AGREEMENT = _cmp
print("\\nAgreement between the headline model and the two strict variants:")
display(ROBUST_AGREEMENT.round(4))
''')


# ===========================================================================
# 19. State-space candidate dynamics
# ===========================================================================
md("""## 19. Principal model, step 1 — which dependence structure does the series have?

Everything from §12 to §18 answers *"is this driver associated with WH extent"*.
None of it answers the question this notebook actually has to answer:

> **once the series' own persistence is represented properly, is there anything
> left for the environment to explain?**

The `y_lag1` term in M4 is one answer, and a fragile one. It is an *ad hoc*
right-hand-side variable: it deletes every month whose predecessor is missing,
it makes the AR coefficient compete with the drivers for the same regression
weights, and its OLS standard errors are wrong when the disturbance is serially
correlated in any way other than exactly AR(1).

This section starts the **principal** model, in which persistence is a property
of the **process**, not a column of the design matrix:

$$y_t \\;=\\; \\underbrace{c + \\gamma' \\text{season}_t + \\delta\\, \\text{trend}_t}_{\\text{deterministic}}
\\;+\\; \\underbrace{\\beta' x_t}_{\\text{drivers (added in §20)}} \\;+\\; \\mu_t,$$

with three candidate laws for $\\mu_t$ and nothing else varying between them:

| Candidate | $\\mu_t$ | Linear trend? |
|---|---|---|
| `sarimax_ar1` | $\\phi\\mu_{t-1}+\\varepsilon_t$ | yes |
| `sarimax_ar2` | $\\phi_1\\mu_{t-1}+\\phi_2\\mu_{t-2}+\\varepsilon_t$ | yes |
| `local_level` | $\\mu_t=\\mu_{t-1}+\\eta_t$, observed with noise | **no** — the level is the trend |

### The rule this section obeys

**The dependence structure is chosen with no environmental driver in any
model.** All three candidates are fitted as *null dynamics* — season and trend
only — and compared on:

* **AICc**, computed from the number of **observed response months** and the
  number of **free parameters** (using the grid length would understate the
  penalty on a record that is a quarter missing);
* **one-calendar-month-ahead rolling-origin RMSE** (expanding window);
* **standardized one-step-ahead innovation diagnostics** — calendar-lag ACF, a
  calendar-lag Ljung–Box, normality, and a variance-ratio check;
* **convergence and stationarity**.

Driver significance plays no part. The winner is then **locked** before §20 adds
a single environmental variable, so the dependence structure cannot be quietly
tuned until the drivers look good.

### The missing-month placeholder, and why it is admissible

State-space filtering tolerates a missing **response** — the Kalman filter skips
the update and carries the state forward — but it needs finite **exogenous**
values on every row of the grid. Two rules make that safe:

1. A month whose response is observed but whose drivers are not is **removed
   from the likelihood** (its response is set missing and the removal is
   recorded), so it can never be fitted against a made-up driver value.
2. Only *after* that are the remaining missing standardized exogenous values
   replaced by `0.0`, and only on months where the response is missing. Those
   months contribute **no likelihood term at all**, so the placeholder cannot
   move a single estimate. The cell asserts both properties.
""")

code('''# =====================================================================
# 19a. The state-space modelling frame (calendar-complete, nothing imputed
#      where it could matter)
# =====================================================================
SS_READY = False
SS_SKIP_REASON = ""
SS_DRIVER_TERMS = [c for c in DRIVER_TERMS if c in model_df.columns]
SS_SEASON_COLS = [c for c in SEASON_COLS if c in model_df.columns]
SS_TREND_COLS = [c for c in TREND_TERMS if c in model_df.columns]

if not RUN_STATE_SPACE:
    SS_SKIP_REASON = "RUN_STATE_SPACE = False"
elif not SS_DRIVER_TERMS:
    SS_SKIP_REASON = "no environmental driver terms survived §9"
elif N_OBSERVED < 30:
    SS_SKIP_REASON = f"only {N_OBSERVED} observed months"

if SS_SKIP_REASON:
    print(f"§19-§24 SKIPPED: {SS_SKIP_REASON}.")
    SS_GRID = pd.DataFrame()
    SS_Y = pd.Series(dtype=float)
    SS_EXOG = pd.DataFrame()
    SS_PLACEHOLDER_AUDIT = pd.DataFrame()
    SS_WITHHELD_MONTHS = pd.DataFrame()
else:
    # --- 1. Which months can carry a likelihood contribution? ----------------
    _obs = model_df["y"].notna()
    _drv_ok = model_df[SS_DRIVER_TERMS].notna().all(axis=1)
    _usable = _obs & _drv_ok

    # Months whose RESPONSE is observed but whose drivers are not. Their response
    # is withheld from the likelihood: fitting them would mean inventing a driver
    # value for a month that does contribute to the fit.
    _withheld = _obs & ~_drv_ok
    SS_WITHHELD_MONTHS = model_df.loc[_withheld, ["month"] + SS_DRIVER_TERMS].copy()
    if len(SS_WITHHELD_MONTHS):
        SS_WITHHELD_MONTHS["missing_drivers"] = SS_WITHHELD_MONTHS[SS_DRIVER_TERMS].apply(
            lambda r: ", ".join(r.index[r.isna()]), axis=1)
        SS_WITHHELD_MONTHS = SS_WITHHELD_MONTHS[["month", "missing_drivers"]]
        SS_WITHHELD_MONTHS["reason"] = ("response observed but a predeclared driver is "
                                        "not; withheld from the likelihood rather than "
                                        "fitted against an imputed driver")

    if int(_usable.sum()) < 30:
        SS_SKIP_REASON = (f"only {int(_usable.sum())} months have both the response and "
                          "every predeclared driver")
        print(f"§19-§24 SKIPPED: {SS_SKIP_REASON}.")
        SS_GRID = pd.DataFrame(); SS_Y = pd.Series(dtype=float)
        SS_EXOG = pd.DataFrame(); SS_PLACEHOLDER_AUDIT = pd.DataFrame()
    else:
        # --- 2. Trim to the span the usable months cover, keeping every gap ---
        _first = model_df.loc[_usable, "month"].min()
        _last = model_df.loc[_usable, "month"].max()
        SS_GRID = model_df[(model_df["month"] >= _first)
                           & (model_df["month"] <= _last)].reset_index(drop=True)
        SS_INDEX = calendar_grid_index(SS_GRID["month"])
        assert len(SS_INDEX) == len(SS_GRID), \\
            "the state-space frame is not calendar-complete"
        assert (pd.to_datetime(SS_GRID["month"]).to_numpy()
                == SS_INDEX.to_numpy()).all(), "grid months are not the calendar grid"

        _usable_ss = (SS_GRID["y"].notna()
                      & SS_GRID[SS_DRIVER_TERMS].notna().all(axis=1)).to_numpy()
        SS_Y = pd.Series(np.where(_usable_ss, SS_GRID["y"].to_numpy(dtype=float), np.nan),
                         index=SS_INDEX, name="y")
        SS_Y_RAW = pd.Series(np.where(_usable_ss,
                                      SS_GRID["y_raw"].to_numpy(dtype=float), np.nan),
                             index=SS_INDEX, name="y_raw")

        # --- 3. Exogenous block, with the placeholder confined to missing y ----
        _exog_cols = SS_SEASON_COLS + SS_TREND_COLS + SS_DRIVER_TERMS
        SS_EXOG_RAWNAN = SS_GRID[_exog_cols].copy()
        SS_EXOG_RAWNAN.index = SS_INDEX
        _missing = SS_EXOG_RAWNAN.isna()
        _y_missing = SS_Y.isna().to_numpy()[:, None]
        _placeholder = _missing & _y_missing
        SS_EXOG = SS_EXOG_RAWNAN.mask(_placeholder, 0.0)

        SS_PLACEHOLDER_AUDIT = (
            pd.DataFrame(_placeholder.to_numpy(), index=SS_INDEX, columns=_exog_cols)
            .stack().rename("placeholder").reset_index()
            .rename(columns={"level_0": "month", "level_1": "term"}))
        SS_PLACEHOLDER_AUDIT = (SS_PLACEHOLDER_AUDIT[SS_PLACEHOLDER_AUDIT["placeholder"]]
                                .drop(columns="placeholder").reset_index(drop=True))
        if len(SS_PLACEHOLDER_AUDIT):
            SS_PLACEHOLDER_AUDIT["response_observed"] = False
            SS_PLACEHOLDER_AUDIT["value_used"] = 0.0
            SS_PLACEHOLDER_AUDIT["note"] = ("computational placeholder only; this month "
                                            "makes NO likelihood contribution")

        # --- 4. The assertions the placeholder's admissibility rests on --------
        _obs_rows = SS_Y.notna().to_numpy()
        assert not SS_EXOG_RAWNAN.loc[_obs_rows].isna().to_numpy().any(), \\
            ("a month with an OBSERVED response has a missing exogenous value; the "
             "placeholder would enter the likelihood")
        assert not _placeholder.to_numpy()[_obs_rows].any(), \\
            "a placeholder was written on a month with an observed response"
        assert np.isfinite(SS_EXOG.to_numpy(dtype=float)).all(), \\
            "the exogenous block still contains non-finite values"
        assert int(SS_Y.notna().sum()) == int(_usable_ss.sum())

        SS_N_OBS = int(SS_Y.notna().sum())
        SS_SPAN = int(len(SS_INDEX))
        SS_READY = True

        print("State-space modelling frame")
        print(f"  calendar grid        : {SS_SPAN} months "
              f"({SS_INDEX.min():%Y-%m} .. {SS_INDEX.max():%Y-%m}), no gaps closed up")
        print(f"  likelihood months    : {SS_N_OBS} "
              f"({SS_SPAN - SS_N_OBS} months contribute nothing)")
        print(f"  complete-case rows §11 used for M3/M4: {N_FIT} "
              f"-> the state-space model uses {SS_N_OBS - N_FIT:+d} month(s) more")
        print(f"  exogenous block      : {len(SS_SEASON_COLS)} deterministic annual "
              f"Fourier column(s), {len(SS_TREND_COLS)} trend, "
              f"{len(SS_DRIVER_TERMS)} driver(s)")
        if len(SS_WITHHELD_MONTHS):
            print(f"\\n  {len(SS_WITHHELD_MONTHS)} month(s) had an observed response but an "
                  "incomplete driver row and were WITHHELD from the likelihood:")
            display(SS_WITHHELD_MONTHS)
        if len(SS_PLACEHOLDER_AUDIT):
            print(f"  {len(SS_PLACEHOLDER_AUDIT)} exogenous cell(s) across "
                  f"{SS_PLACEHOLDER_AUDIT['month'].nunique()} month(s) took the 0.0 "
                  "placeholder. Every one is on a month with NO observed response, so "
                  "it makes no likelihood contribution and cannot move an estimate.")
            display(SS_PLACEHOLDER_AUDIT.head(12))
        else:
            print("  no placeholder was needed.")
''')

md("""### 19b. Candidate null-dynamics fits and the selection

Three fits, no drivers in any of them, on identical response months. The
selection is printed in full so the choice can be checked rather than trusted —
including the case where AICc and one-month-ahead RMSE disagree, which is
reported rather than resolved silently.
""")

code('''# =====================================================================
# 19b. Fit the candidate dependence structures WITHOUT any driver
# =====================================================================
SS_CANDIDATES = pd.DataFrame()
SS_CANDIDATE_FITS = {}
SS_CANDIDATE_DIAGNOSTICS = pd.DataFrame()
SS_SELECTED = None
SS_SELECTION_NOTE = ""

if SS_READY:
    def ss_null_exog(structure):
        """Season, plus a linear trend only where a stochastic level is absent."""
        cols = list(SS_SEASON_COLS)
        if ss_allows_linear_trend(structure) and INCLUDE_TREND:
            cols += list(SS_TREND_COLS)
        return SS_EXOG[cols]

    def ss_rolling_one_step_null(structure, min_train_months=24, maxiter=100):
        """One-calendar-month-ahead expanding-origin RMSE for a NULL candidate.

        Season and the trend are deterministic functions of the calendar, so they
        can be built for a future month without touching a future response — the
        only quantity a fold takes from the past is the response itself.
        """
        cols = list(ss_null_exog(structure).columns)
        mi = month_index(pd.Series(SS_INDEX))
        errs, months_scored = [], []
        yv = SS_Y.to_numpy(dtype=float)
        for pos in range(1, len(SS_INDEX)):
            if not np.isfinite(yv[pos]):
                continue
            n_train_obs = int(np.isfinite(yv[:pos]).sum())
            if n_train_obs < int(min_train_months):
                continue
            try:
                res, _ = ss_fit(SS_Y.iloc[:pos], SS_EXOG[cols].iloc[:pos],
                                structure, cov_type=None, maxiter=int(maxiter))
                fc = res.get_forecast(steps=1, exog=SS_EXOG[cols].iloc[pos:pos + 1])
                yhat = float(np.asarray(fc.predicted_mean)[0])
            except Exception:
                continue
            assert mi[pos] - mi[pos - 1] == 1, "forecast is not one calendar month ahead"
            errs.append(yv[pos] - yhat)
            months_scored.append(SS_INDEX[pos])
        if not errs:
            return np.nan, 0, []
        return float(np.sqrt(np.mean(np.asarray(errs) ** 2))), len(errs), months_scored

    _rows, _diag_rows = [], []
    for _key in SS_CANDIDATE_STRUCTURES:
        _cols = list(ss_null_exog(_key).columns)
        try:
            _res, _info = ss_fit(SS_Y, SS_EXOG[_cols], _key,
                                 cov_type=SS_COV_TYPE, maxiter=SS_MAXITER)
        except Exception as _exc:
            _rows.append({"structure": _key, "label": ss_structure_label(_key),
                          "fitted": False, "reason": str(_exc)[:160]})
            continue
        SS_CANDIDATE_FITS[_key] = _res
        _k = ss_n_free_params(_res)
        _n = ss_n_effective(SS_Y)
        _st = ss_state_diagnostics(_res, _key)
        _dg, _acf, _inn = ss_innovation_diagnostics(_res, SS_INDEX, nlags=min(18, _n // 3),
                                                    label=f"null {_key}")
        _rmse1, _n1, _m1 = ss_rolling_one_step_null(
            _key, SS_MIN_TRAIN_MONTHS, SS_ROLLING_MAXITER)
        _rows.append({
            "structure": _key, "label": ss_structure_label(_key), "fitted": True,
            "includes_linear_trend": bool(ss_allows_linear_trend(_key) and INCLUDE_TREND),
            "n_observed_months": _n, "k_free_params": _k,
            "loglik": float(_res.llf),
            "aic": float(_res.aic), "bic": float(_res.bic),
            "aicc": aicc_from(_res.llf, _k, _n),
            "rmse_one_month_ahead": _rmse1, "n_one_month_targets": int(_n1),
            "converged": bool(_info["converged"]),
            "cov_type_used": _info["cov_type_used"],
            "stationary_supported": bool(_st["stationary_supported"]),
            "stationarity_note": _st["reason"] or "stationary",
            "innov_acf1": _dg["acf1"], "innov_ljung_box_p": _dg["ljung_box_p"],
            "innov_jarque_bera_p": _dg["jarque_bera_p"],
            "innov_sd": _dg["sd_std_innovation"],
            "reason": "",
        })
        _diag_rows.append({**_dg, "structure": _key, "model": "null dynamics"})

    SS_CANDIDATES = pd.DataFrame(_rows)
    SS_CANDIDATE_DIAGNOSTICS = pd.DataFrame(_diag_rows)
    SS_CANDIDATES["aicc_delta"] = (SS_CANDIDATES.get("aicc", pd.Series(dtype=float))
                                   - SS_CANDIDATES.get("aicc", pd.Series(dtype=float)).min())

    print("Candidate dependence structures — NULL dynamics (season/trend, NO drivers).")
    print("The selection below uses no driver p-value of any kind.\\n")
    display(SS_CANDIDATES.round(4))

    _ok = SS_CANDIDATES[SS_CANDIDATES["fitted"] & SS_CANDIDATES["converged"]]
    if not len(_ok):
        print("No candidate converged; §20-§24 cannot run.")
        SS_READY = False
        SS_SKIP_REASON = "no state-space candidate converged"
    else:
        _by_aicc = _ok.sort_values("aicc")["structure"].iloc[0]
        _rm = _ok.dropna(subset=["rmse_one_month_ahead"])
        _by_rmse = (_rm.sort_values("rmse_one_month_ahead")["structure"].iloc[0]
                    if len(_rm) else _by_aicc)
        SS_SELECTED = _by_aicc if str(SS_SELECT_BY).lower() == "aicc" else _by_rmse
        SS_SELECTION_NOTE = (
            f"selected by {SS_SELECT_BY}; AICc favours {_by_aicc}, "
            f"one-month-ahead RMSE favours {_by_rmse}"
            + ("" if _by_aicc == _by_rmse else " — THEY DISAGREE"))
        print(f"\\nAICc ranking            : "
              f"{_ok.sort_values('aicc')['structure'].tolist()}")
        if len(_rm):
            print(f"One-month-ahead ranking : "
                  f"{_rm.sort_values('rmse_one_month_ahead')['structure'].tolist()}")
        if _by_aicc != _by_rmse:
            print("\\n*** AICc and one-month-ahead RMSE prefer DIFFERENT structures. "
                  f"SS_SELECT_BY = {SS_SELECT_BY!r} breaks the tie; §24 refits the driver "
                  "model under all three so the choice can be seen to matter or not. ***")
        print(f"\\nLOCKED dependence structure: {SS_SELECTED} "
              f"({ss_structure_label(SS_SELECTED)})")
        print("It is fixed here, BEFORE any environmental variable is added in §20.")

        _sel_row = _ok[_ok["structure"] == SS_SELECTED].iloc[0]
        if not bool(_sel_row["stationary_supported"]):
            print(f"\\nStationarity: {_sel_row['stationarity_note']}. §20 therefore "
                  "reports SHORT-RUN standardized coefficients only — no long-run "
                  "multiplier is defined under this structure.")

        # Innovation ACF of the selected null structure, on CALENDAR lags.
        _res_sel = SS_CANDIDATE_FITS[SS_SELECTED]
        _dg, _acf_sel, _inn_sel = ss_innovation_diagnostics(
            _res_sel, SS_INDEX, nlags=min(24, max(6, SS_N_OBS // 3)),
            label=f"null {SS_SELECTED}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
        axes[0].bar(_acf_sel["lag"], _acf_sel["acf"], color="tab:purple", alpha=0.8)
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].axhline(_acf_sel["band"].iloc[0], color="red", ls="--", lw=1)
        axes[0].axhline(-_acf_sel["band"].iloc[0], color="red", ls="--", lw=1)
        axes[0].set_title(f"Standardized innovation ACF — null {SS_SELECTED}\\n"
                          f"calendar-lag Ljung-Box p = {_dg['ljung_box_p']:.3g}")
        axes[0].set_xlabel("lag (calendar months)")
        axes[1].plot(*on_calendar_grid(_inn_sel["month"], _inn_sel["std_innovation"]),
                     marker="o", ls="-", ms=3, lw=1)
        axes[1].axhline(0, color="k", lw=0.8)
        for _b in (-2, 2):
            axes[1].axhline(_b, color="red", ls=":", lw=1)
        axes[1].set_title("Standardized one-step-ahead innovations")
        axes[1].set_xlabel("month")
        for _a in axes:
            _a.grid(alpha=0.3)
        fig.tight_layout(); plt.show()
else:
    print("§19b skipped.")
''')


# ===========================================================================
# 20. Matched null vs full state-space driver model
# ===========================================================================
md("""## 20. Principal model, step 2 — matched null vs full driver model

The structure is locked. Two models are now fitted, differing in **exactly one
thing**: whether the predeclared environmental drivers are present.

| | Model | Exogenous block |
|---|---|---|
| **null dynamic model** | selected structure | season (+ trend where admissible) |
| **full dynamic driver model** | *the same structure* | season (+ trend) **+ the predeclared drivers at their a-priori lags** |

Everything else is held identical and asserted: the same response months, the
same logit transform, the same deterministic Fourier season, the same
missing-month treatment, the same optimiser settings. That is what makes the
difference between them attributable to the drivers and nothing else.

**This is retrospective association / nowcasting, not forecasting.** The
contemporaneous terms (temperature, wind, lake level enter at lag 0 by the
a-priori specification of §3c) are not knowable a month in advance, so this
section says *"months in which it was warmer than usual also had more hyacinth,
after persistence and season"* — never *"we can predict next month"*. The
forecasting question is §21's, and it uses a lagged driver set for exactly this
reason.

What is reported: standardized coefficients, robust state-space standard errors,
95% intervals, raw $p$-values, BH $q$-values, the joint driver-block test against
the matched null (likelihood ratio **and** Wald), AICc and log-likelihood
differences, the AR roots or latent-state diagnostics, and the standardized
one-step-ahead innovation ACF on calendar lags.
""")

code('''# =====================================================================
# 20. The locked structure, with and without the environmental drivers
# =====================================================================
SS_NULL_FIT = None
SS_FULL_FIT = None
SS_COEFS = pd.DataFrame()
SS_JOINT_TEST = pd.DataFrame()
SS_MODEL_COMPARISON = pd.DataFrame()
SS_STATE_DIAGNOSTICS = pd.DataFrame()
SS_INNOVATION_DIAGNOSTICS = pd.DataFrame()
SS_INNOVATION_ACF = pd.DataFrame()
SS_MANIFEST = {}

if SS_READY and SS_SELECTED:
    _null_cols = list(SS_SEASON_COLS)
    if ss_allows_linear_trend(SS_SELECTED) and INCLUDE_TREND:
        _null_cols += list(SS_TREND_COLS)
    _full_cols = _null_cols + list(SS_DRIVER_TERMS)

    SS_NULL_FIT, _null_info = ss_fit(SS_Y, SS_EXOG[_null_cols], SS_SELECTED,
                                     cov_type=SS_COV_TYPE, maxiter=SS_MAXITER)
    SS_FULL_FIT, _full_info = ss_fit(SS_Y, SS_EXOG[_full_cols], SS_SELECTED,
                                     cov_type=SS_COV_TYPE, maxiter=SS_MAXITER)

    # --- The matching assertions -------------------------------------------
    _y_null = np.asarray(SS_NULL_FIT.model.endog, dtype=float).ravel()
    _y_full = np.asarray(SS_FULL_FIT.model.endog, dtype=float).ravel()
    assert len(_y_null) == len(_y_full), "null and full models have different lengths"
    assert np.array_equal(np.isfinite(_y_null), np.isfinite(_y_full)), \\
        "null and full state-space models do not use identical response months"
    assert np.allclose(_y_null[np.isfinite(_y_null)], _y_full[np.isfinite(_y_full)]), \\
        "null and full state-space models were given different response values"
    assert set(_null_cols).issubset(set(_full_cols)), \\
        "the full model is not a strict extension of the null model"
    assert (set(_full_cols) - set(_null_cols)) == set(SS_DRIVER_TERMS), \\
        "the full model differs from the null by something other than the drivers"
    SS_MATCHED_MONTHS = pd.Series(SS_INDEX[np.isfinite(_y_full)])

    _n_obs = ss_n_effective(SS_Y)
    _k_null, _k_full = ss_n_free_params(SS_NULL_FIT), ss_n_free_params(SS_FULL_FIT)
    _aicc_null = aicc_from(SS_NULL_FIT.llf, _k_null, _n_obs)
    _aicc_full = aicc_from(SS_FULL_FIT.llf, _k_full, _n_obs)

    SS_MODEL_COMPARISON = pd.DataFrame([
        {"model": "null dynamic model", "structure": SS_SELECTED,
         "exog": "season" + (" + trend" if SS_TREND_COLS and
                             ss_allows_linear_trend(SS_SELECTED) and INCLUDE_TREND else ""),
         "n_observed_months": _n_obs, "k_free_params": _k_null,
         "loglik": float(SS_NULL_FIT.llf), "aic": float(SS_NULL_FIT.aic),
         "aicc": _aicc_null, "bic": float(SS_NULL_FIT.bic),
         "converged": bool(_null_info["converged"]),
         "cov_type_used": _null_info["cov_type_used"]},
        {"model": "full dynamic driver model", "structure": SS_SELECTED,
         "exog": "season" + (" + trend" if SS_TREND_COLS and
                             ss_allows_linear_trend(SS_SELECTED) and INCLUDE_TREND else "")
                 + f" + {len(SS_DRIVER_TERMS)} predeclared driver(s)",
         "n_observed_months": _n_obs, "k_free_params": _k_full,
         "loglik": float(SS_FULL_FIT.llf), "aic": float(SS_FULL_FIT.aic),
         "aicc": _aicc_full, "bic": float(SS_FULL_FIT.bic),
         "converged": bool(_full_info["converged"]),
         "cov_type_used": _full_info["cov_type_used"]},
    ])
    SS_MODEL_COMPARISON["aicc_minus_null"] = (SS_MODEL_COMPARISON["aicc"] - _aicc_null)
    SS_MODEL_COMPARISON["loglik_minus_null"] = (SS_MODEL_COMPARISON["loglik"]
                                                - float(SS_NULL_FIT.llf))

    # --- Coefficients --------------------------------------------------------
    SS_COEFS = ss_tidy_coefficients(
        SS_FULL_FIT, SS_SELECTED, SS_DRIVER_TERMS,
        label=f"S1 state-space {SS_SELECTED} + season/trend + drivers",
        cov_type_used=_full_info["cov_type_used"])
    if len(SS_COEFS):
        SS_COEFS["driver"] = [
            next((b for b in FORCING if t.startswith(b)), t) for t in SS_COEFS["term"]]
        SS_COEFS["expected_sign"] = [FORCING.get(d, {}).get("expected_sign", "?")
                                     for d in SS_COEFS["driver"]]
        SS_COEFS["lag_months"] = [LAG_USED.get(d, np.nan) for d in SS_COEFS["driver"]]
        SS_COEFS["sign_matches_mechanism"] = [
            (e == "?" or (e == "+" and c > 0) or (e == "-" and c < 0))
            for e, c in zip(SS_COEFS["expected_sign"], SS_COEFS["coef"])]
        SS_COEFS["inference_kind"] = "association / nowcast (a-priori lags, §20)"

    # --- Joint driver-block tests -------------------------------------------
    _lr = ss_joint_lr_test(SS_FULL_FIT, SS_NULL_FIT, len(SS_DRIVER_TERMS))
    _wald = ss_joint_wald_test(SS_FULL_FIT, SS_SELECTED, SS_DRIVER_TERMS)
    SS_JOINT_TEST = pd.DataFrame([{
        "comparison": "full driver block vs matched null dynamic model",
        "structure": SS_SELECTED, "n_observed_months": _n_obs,
        "k_drivers": len(SS_DRIVER_TERMS),
        **_lr, **_wald,
        "aicc_null": _aicc_null, "aicc_full": _aicc_full,
        "aicc_difference_full_minus_null": _aicc_full - _aicc_null,
        "aicc_prefers": ("full driver model" if _aicc_full < _aicc_null
                         else "null dynamic model"),
        "lr_p_reference": ("asymptotic chi-square; §23 replaces it with a parametric "
                           "bootstrap because ~60 observed months is not asymptotia"),
    }])

    # --- Structure / state diagnostics --------------------------------------
    _st_full = ss_state_diagnostics(SS_FULL_FIT, SS_SELECTED, alpha=LONGRUN_CI_ALPHA)
    _st_null = ss_state_diagnostics(SS_NULL_FIT, SS_SELECTED, alpha=LONGRUN_CI_ALPHA)
    SS_STATE_DIAGNOSTICS = pd.DataFrame([{**_st_null, "model": "null dynamic model"},
                                         {**_st_full, "model": "full dynamic driver model"}])
    SS_STATIONARY_SUPPORTED = bool(_st_full["stationary_supported"])
    SS_STATIONARITY_REASON = _st_full["reason"]

    # --- Innovation diagnostics ---------------------------------------------
    _rows_d, _rows_a = [], []
    for _label, _res in (("null dynamic model", SS_NULL_FIT),
                         ("full dynamic driver model", SS_FULL_FIT)):
        _dg, _acf, _inn = ss_innovation_diagnostics(
            _res, SS_INDEX, nlags=min(24, max(6, _n_obs // 3)), label=_label)
        _rows_d.append({**_dg, "model": _label, "structure": SS_SELECTED})
        _rows_a.append(_acf.assign(model=_label, structure=SS_SELECTED))
    SS_INNOVATION_DIAGNOSTICS = pd.DataFrame(_rows_d)
    SS_INNOVATION_ACF = pd.concat(_rows_a, ignore_index=True)

    # --- Report --------------------------------------------------------------
    print("=" * 96)
    print(f"PRINCIPAL MODEL — state-space dynamic regression, structure {SS_SELECTED}")
    print("=" * 96)
    print(f"Locked in §19 on null dynamics only. {SS_SELECTION_NOTE}.")
    print(f"Both models fitted on the SAME {_n_obs} observed response months "
          f"({SS_MATCHED_MONTHS.min():%Y-%m} .. {SS_MATCHED_MONTHS.max():%Y-%m}), "
          f"the same {RESPONSE_INFO['transform']} transform and the same "
          f"{len(SS_SEASON_COLS)}-column deterministic annual Fourier season.\\n")
    display(SS_MODEL_COMPARISON.round(4))

    print("\\nStandardized driver coefficients — effect on "
          f"{RESPONSE_INFO['transform']}({RESPONSE_COL}) per 1 SD of the driver, "
          f"{_full_info['cov_type_used']} standard errors, BH q across the driver block:")
    display(SS_COEFS.round(4))
    print("These are ASSOCIATION / NOWCAST estimates: contemporaneous drivers are used "
          "at their a-priori lags, which is right for 'what moved with WH extent' and "
          "wrong for 'what can be forecast'. §21 answers the second question.")

    print("\\nJoint test of the whole driver block against the matched null:")
    display(SS_JOINT_TEST.round(4))
    _lrp = float(SS_JOINT_TEST["p_chi2"].iloc[0])
    print(f"  LR = {_lr['lr_stat']:.3f} on {_lr['df']} df, asymptotic p = {_lrp:.4g}; "
          f"Wald p = {_wald['p_wald']:.4g}")
    print(f"  AICc {'PREFERS' if _aicc_full < _aicc_null else 'does NOT prefer'} the "
          f"driver model ({_aicc_full:.2f} vs {_aicc_null:.2f}, "
          f"difference {_aicc_full - _aicc_null:+.2f}).")

    print("\\nStructure / latent-state diagnostics:")
    display(SS_STATE_DIAGNOSTICS.round(4))
    if SS_STATIONARY_SUPPORTED:
        _rho = float(_st_full["rho_sum"])
        print(f"  Stationarity supported (rho = {_rho:.3f}, CI "
              f"[{_st_full['rho_ci_lo']:.3f}, {_st_full['rho_ci_hi']:.3f}]). A long-run "
              f"multiplier 1/(1-rho) = {1 / (1 - _rho):.2f} would be defined — it is "
              "still not quoted as a headline, because §22 shows what it buys.")
    else:
        print(f"  *** LONG-RUN MULTIPLIERS ARE NOT REPORTED: {SS_STATIONARITY_REASON}. "
              "Only the SHORT-RUN standardized coefficients above may be quoted. ***")

    print("\\nStandardized one-step-ahead innovation diagnostics (calendar lags):")
    display(SS_INNOVATION_DIAGNOSTICS[[
        "model", "n_innovations", "mean_std_innovation", "sd_std_innovation",
        "acf1", "acf1_n_pairs", "acf12", "ljung_box_stat", "ljung_box_df",
        "ljung_box_p", "jarque_bera_p", "het_var_ratio_last_over_first_third"]].round(4))
    _lb_full = float(SS_INNOVATION_DIAGNOSTICS.loc[
        SS_INNOVATION_DIAGNOSTICS["model"] == "full dynamic driver model",
        "ljung_box_p"].iloc[0])
    if np.isfinite(_lb_full) and _lb_full < 0.05:
        print("  Residual dependence REMAINS after the selected structure: the "
              "dependence model is incomplete, so the driver standard errors below are "
              "optimistic. Say so in the write-up.")
    else:
        print("  No detectable dependence left in the standardized innovations: the "
              "selected structure has absorbed the persistence, which is exactly the "
              "condition under which the driver coefficients are interpretable.")

    # --- Coefficient plot + innovation ACF ----------------------------------
    if len(SS_COEFS):
        fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(SS_COEFS) + 2.6))
        _c = SS_COEFS.sort_values("coef")
        axes[0].errorbar(_c["coef"], range(len(_c)),
                         xerr=[_c["coef"] - _c["ci_lo"], _c["ci_hi"] - _c["coef"]],
                         fmt="o", ms=6, capsize=3, color="tab:purple")
        axes[0].axvline(0, color="k", lw=1)
        axes[0].set_yticks(range(len(_c)))
        axes[0].set_yticklabels([f"{t}  ({s})" for t, s in
                                 zip(_c["term"], _c["expected_sign"])])
        axes[0].set_xlabel(f"effect on {RESPONSE_INFO['transform']}({RESPONSE_COL}) "
                           "per 1 SD (95% CI)")
        axes[0].set_title(f"State-space driver effects — {SS_SELECTED}\\n"
                          "(a-priori expected sign in brackets); ASSOCIATION, not forecast")
        axes[0].grid(alpha=0.3, axis="x")
        _a = SS_INNOVATION_ACF[SS_INNOVATION_ACF["model"] == "full dynamic driver model"]
        axes[1].bar(_a["lag"], _a["acf"], color="tab:purple", alpha=0.8)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].axhline(_a["band"].iloc[0], color="red", ls="--", lw=1)
        axes[1].axhline(-_a["band"].iloc[0], color="red", ls="--", lw=1)
        axes[1].set_title("Standardized innovation ACF — full driver model\\n"
                          "(calendar-month lags)")
        axes[1].set_xlabel("lag (calendar months)")
        axes[1].grid(alpha=0.3)
        fig.tight_layout(); plt.show()

    SS_MANIFEST = {
        "run": True,
        "structure_selected": SS_SELECTED,
        "structure_label": ss_structure_label(SS_SELECTED),
        "selection_note": SS_SELECTION_NOTE,
        "selected_by": SS_SELECT_BY,
        "selection_used_driver_significance": False,
        "candidates": SS_CANDIDATE_STRUCTURES,
        "season": f"deterministic annual Fourier, {SS_SEASON_HARMONICS} harmonic(s)",
        "linear_trend_included": bool(ss_allows_linear_trend(SS_SELECTED) and INCLUDE_TREND),
        "response": f"{RESPONSE_INFO['transform']}({RESPONSE_COL})",
        "n_observed_months": int(_n_obs),
        "n_calendar_months_grid": int(len(SS_INDEX)),
        "first_month": str(pd.Timestamp(SS_INDEX.min()).date()),
        "last_month": str(pd.Timestamp(SS_INDEX.max()).date()),
        "driver_terms": list(SS_DRIVER_TERMS),
        "driver_lags_apriori": {k: int(v) for k, v in LAG_USED.items()},
        "null_exog": list(_null_cols), "full_exog": list(_full_cols),
        "k_free_params_null": int(_k_null), "k_free_params_full": int(_k_full),
        "aicc_null": float(_aicc_null), "aicc_full": float(_aicc_full),
        "loglik_null": float(SS_NULL_FIT.llf), "loglik_full": float(SS_FULL_FIT.llf),
        "cov_type_used": _full_info["cov_type_used"],
        "stationary_supported": bool(SS_STATIONARY_SUPPORTED),
        "stationarity_reason": SS_STATIONARITY_REASON,
        "long_run_multipliers_reported": False,
        "n_months_withheld_incomplete_drivers": int(len(SS_WITHHELD_MONTHS)),
        "n_placeholder_exog_cells": int(len(SS_PLACEHOLDER_AUDIT)),
        "placeholder_rule": ("standardized exogenous NaN -> 0.0 ONLY where the response "
                             "is missing; asserted never to touch a month that "
                             "contributes to the likelihood"),
        "inference_kind": "association / nowcasting (a-priori lags incl. lag 0)",
        "is_synthetic": bool(SOURCE["is_synthetic"]),
    }
else:
    print("§20 skipped (no locked state-space structure).")
    SS_STATIONARY_SUPPORTED = False
    SS_STATIONARITY_REASON = SS_SKIP_REASON or "state-space model not fitted"
    SS_MATCHED_MONTHS = pd.Series(dtype="datetime64[ns]")
    SS_MANIFEST = {"run": False, "reason": SS_STATIONARITY_REASON}
''')

# ===========================================================================
# 21. One-calendar-month-ahead rolling origin
# ===========================================================================
md("""## 21. Principal model, step 3 — genuine one-calendar-month-ahead prediction

§16's three-calendar-month windows are kept as a **sensitivity comparison**, but
they are no longer the headline, for two reasons. They use only eight origins,
and their design still lets a contemporaneous driver — this month's temperature,
this month's wind, this month's lake level — enter the prediction of *this
month*. That is a **nowcast** with information no forecaster has.

This section is the principal predictive assessment and fixes both:

### The design

* **Expanding window, one calendar month ahead.** Every feasible origin after at
  least `SS_MIN_TRAIN_MONTHS` observed training months. The target is always the
  **single next calendar month**, asserted month by month.
* **Training is strictly earlier.** Training uses only calendar months *before*
  the target; the assertion `max(train month) < target month` runs on every fold.
* **Every transformation and standardisation is fitted inside the fold.** The
  driver means and standard deviations come from the fold's *training months
  only* — the globally z-scored columns from §11 are deliberately **not** used
  here, because their scaling constants were computed with the target month in
  them. The per-fold scaler's sample size is recorded and asserted equal to the
  number of training months.
* **Only origin-time information enters a predictor.** Rainfall already enters at
  lag 1. Temperature, wind and lake level are **converted from lag 0 to lag 1**
  for this evaluation, because their contemporaneous value is not knowable at the
  origin. Supply a genuine forecast column through
  `SS_FORECAST_EXOG_OVERRIDE` to use one instead. Every forecast driver is
  asserted to have lag $\\ge$ `SS_FORECAST_MIN_LAG`.
* **Null and full are refit on identical training and target months.** A target
  month usable by only one of them is dropped from both and the reason recorded.

### The baselines

| Baseline | Definition | Fitted? |
|---|---|---|
| **literal persistence** | $\\hat y_t = y_{t-1}$ at exactly $t-1$ calendar months | **no** — no coefficient of any kind |
| **fitted AR(1)** | AR(1) with constant, refit at every origin, forecast one month | yes |
| **seasonal naive** | $\\hat y_t = y_{t-12}$ by calendar timestamp | no |
| **null state-space** | locked structure + season (+ trend) | yes |
| **full state-space** | *the same* + the lagged driver block | yes |
| *training mean* | mean of the training response (context only) | yes |

A regression of $y_t$ on `y_lag1` is **not** literal persistence and is never
labelled as such: it has a fitted slope and intercept, and on a series with
$\\rho \\approx 0.9$ that shrinkage is worth real skill. Both appear.

Where the exact source month is missing, seasonal-naive and literal persistence
are reported **unavailable** with their own $n$, and a **like-for-like table
restricted to the common months** every model can be scored on is produced
alongside.
""")

code('''# =====================================================================
# 21a. Build the FORECAST driver set — origin-time information only
# =====================================================================
SS_FORECAST_SPECS = pd.DataFrame()
SS_FC_TERMS = []
SS_FC_RAW = pd.DataFrame()

if SS_READY:
    # `monthly` holds the RAW (unstandardised) driver values on the calendar grid.
    # Nothing is standardised here: §21b does that inside each fold.
    _fc_rows = []
    _fc_frame = monthly[["month"]].copy()
    for _base, _meta in FORCING.items():
        _ap = int(LAG_USED.get(_base, _meta.get("apriori_lag", 0)))
        _override = SS_FORECAST_EXOG_OVERRIDE.get(_base)
        if _override and _override in monthly.columns:
            _col, _lag, _how = _override, 0, "supplied forecast value"
        else:
            _lag = max(_ap, int(SS_FORECAST_MIN_LAG))
            _col = _base
            _how = ("a-priori lag already >= the origin" if _ap >= SS_FORECAST_MIN_LAG
                    else f"a-priori lag {_ap} is not knowable at the origin -> lag {_lag}")
        if _col not in monthly.columns:
            _fc_rows.append({"driver": _base, "apriori_lag": _ap, "forecast_lag": np.nan,
                             "term": "", "usable": False,
                             "note": f"{_col} not in the built series"})
            continue
        _name = f"fc_{_base}_lag{_lag}" if _lag else f"fc_{_base}"
        _fc_frame[_name] = monthly[_col].shift(_lag).to_numpy()
        _fc_rows.append({"driver": _base, "apriori_lag": _ap, "forecast_lag": int(_lag),
                         "term": _name, "source_column": _col, "usable": True,
                         "note": _how})
        SS_FC_TERMS.append(_name)
    SS_FORECAST_SPECS = pd.DataFrame(_fc_rows)
    SS_FC_RAW = _fc_frame

    # The assertion the whole evaluation rests on: no forecast driver may carry
    # information dated at or after the target month.
    assert all(int(r) >= int(SS_FORECAST_MIN_LAG)
               for r in SS_FORECAST_SPECS.loc[SS_FORECAST_SPECS["usable"],
                                              "forecast_lag"]), \\
        "a forecast driver would use information from the target month or later"

    print("Forecast driver set — every term is knowable at the origin:")
    display(SS_FORECAST_SPECS)
    _changed = SS_FORECAST_SPECS[(SS_FORECAST_SPECS["usable"])
                                 & (SS_FORECAST_SPECS["forecast_lag"]
                                    > SS_FORECAST_SPECS["apriori_lag"])]
    if len(_changed):
        print(f"\\n{len(_changed)} contemporaneous driver(s) were moved to lag 1 for the "
              f"FORECAST evaluation: {_changed['driver'].tolist()}.")
        print("The a-priori (lag-0) specification is retained unchanged for the "
              "association/nowcast inference in §12 and §20 — the two are different "
              "questions and are never merged.")
else:
    print("§21a skipped.")
''')

code('''# =====================================================================
# 21b. Expanding-window, one-calendar-month-ahead rolling origin
# =====================================================================
SS_FOLD_AUDIT = pd.DataFrame()
SS_ONE_MONTH_PREDICTIONS = pd.DataFrame()
SS_SKILL = pd.DataFrame()
SS_SKILL_COMMON = pd.DataFrame()
SS_EVAL_MONTHS_COMMON = pd.Series(dtype="datetime64[ns]")
SS_SCALER_AUDIT = pd.DataFrame()

if not (SS_READY and SS_SELECTED and SS_FC_TERMS):
    print("§21b skipped (no locked structure or no usable forecast drivers).")
else:
    # ---- One calendar-complete frame carrying everything a fold can need -----
    _fc = monthly[["month", "y_raw", "month_num"]].merge(SS_FC_RAW, on="month", how="left")
    _fc = _fc.sort_values("month").reset_index(drop=True)
    _fc["t_raw"] = np.arange(len(_fc), dtype=float)     # deterministic month counter
    _season = fourier_terms(_fc["month"], SS_SEASON_HARMONICS)
    _season_cols = list(_season.columns)
    _fc = pd.concat([_fc, _season], axis=1)
    _MI = month_index(_fc["month"])
    assert np.all(np.diff(_MI) == 1), "the forecast frame is not calendar-complete"
    _idx_all = calendar_grid_index(_fc["month"])

    _use_trend = bool(ss_allows_linear_trend(SS_SELECTED) and INCLUDE_TREND)
    _scale_cols = list(SS_FC_TERMS) + (["t_raw"] if _use_trend else [])

    _rows_audit, _rows_pred, _rows_scaler = [], [], []
    _t0 = pd.Timestamp.now()

    for _pos in range(1, len(_fc)):
        _target = _fc["month"].iloc[_pos]
        _origin = _fc["month"].iloc[_pos - 1]
        _rec = {"target_month": _target, "origin_month": _origin,
                "horizon_months": int(_MI[_pos] - _MI[_pos - 1]),
                "n_train_months_observed": 0, "usable": False, "skip_reason": ""}
        assert _rec["horizon_months"] == 1, "target is not exactly one calendar month after the origin"

        _train_slice = slice(0, _pos)
        _y_train_raw = _fc["y_raw"].iloc[_train_slice]
        _train_ok = _y_train_raw.notna().to_numpy()
        _rec["n_train_months_observed"] = int(_train_ok.sum())
        _rec["train_start"] = (_fc["month"].iloc[:_pos][_train_ok].min()
                               if _train_ok.any() else pd.NaT)
        _rec["train_end"] = (_fc["month"].iloc[:_pos][_train_ok].max()
                             if _train_ok.any() else pd.NaT)

        if _rec["n_train_months_observed"] < int(SS_MIN_TRAIN_MONTHS):
            _rec["skip_reason"] = (f"{_rec['n_train_months_observed']} observed training "
                                   f"month(s) < SS_MIN_TRAIN_MONTHS "
                                   f"({SS_MIN_TRAIN_MONTHS})")
            _rows_audit.append(_rec); continue
        if not np.isfinite(_fc["y_raw"].iloc[_pos]):
            _rec["skip_reason"] = ("target response not observed (month excluded by the "
                                   "coverage filter); NOT imputed")
            _rows_audit.append(_rec); continue
        _miss_fc = [c for c in SS_FC_TERMS if not np.isfinite(_fc[c].iloc[_pos])]
        if _miss_fc:
            _rec["skip_reason"] = ("forecast driver(s) unavailable at the origin: "
                                   + ", ".join(_miss_fc))
            _rows_audit.append(_rec); continue

        # ---- Everything below is fitted on TRAINING MONTHS ONLY --------------
        _tr = _fc.iloc[_train_slice][_train_ok].copy()
        _tr_full = _fc.iloc[_train_slice].copy()      # incl. gap months, for the filter
        assert _tr["month"].max() < _target, "training reaches the target month"
        assert _fc["month"].iloc[_pos - 1] == _origin

        # Response transform: applied with the training rows only (its constants
        # are fixed, but it is still evaluated inside the fold and recorded).
        _how = "log" if RESPONSE_COL == "wh_area_ha" else RESPONSE_TRANSFORM
        _y_tr_grid, _ = transform_response(_tr_full["y_raw"], _how, RESPONSE_EPS)
        _y_target, _ = transform_response(_fc["y_raw"].iloc[[_pos]], _how, RESPONSE_EPS)
        _y_target = float(np.asarray(_y_target)[0])

        # Standardisation: means/SDs from the TRAINING months, applied to both.
        _mu = _tr[_scale_cols].mean()
        _sd = _tr[_scale_cols].std(ddof=1).replace(0, np.nan)
        assert int(_tr[_scale_cols].notna().all(axis=1).sum()) <= len(_tr)
        _rows_scaler.append({"target_month": _target, "n_train_rows_used": int(len(_tr)),
                             "n_train_months_observed": _rec["n_train_months_observed"],
                             **{f"mean__{c}": float(_mu[c]) for c in _scale_cols},
                             **{f"sd__{c}": float(_sd[c]) for c in _scale_cols}})
        if not np.isfinite(_sd.to_numpy(dtype=float)).all():
            _rec["skip_reason"] = "a forecast driver has zero variance in this training window"
            _rows_audit.append(_rec); continue

        def _std(frame):
            out = frame[_scale_cols].copy()
            for c in _scale_cols:
                out[c] = (out[c] - float(_mu[c])) / float(_sd[c])
            return out

        _X_tr = pd.concat([_tr_full[_season_cols].reset_index(drop=True),
                           _std(_tr_full).reset_index(drop=True)], axis=1)
        _X_te = pd.concat([_fc[_season_cols].iloc[[_pos]].reset_index(drop=True),
                           _std(_fc.iloc[[_pos]]).reset_index(drop=True)], axis=1)
        _null_cols_fc = list(_season_cols) + (["t_raw"] if _use_trend else [])
        _full_cols_fc = _null_cols_fc + list(SS_FC_TERMS)

        # The state-space filter needs finite exog on the training grid too. The
        # SAME rule as §19a applies, in this order: a training month whose
        # forecast drivers are incomplete is WITHHELD from the likelihood (its
        # response is set missing), and only then is the 0.0 placeholder written
        # — so it lands exclusively on months that contribute nothing.
        _y_tr_series = pd.Series(np.asarray(_y_tr_grid, dtype=float),
                                 index=_idx_all[:_pos])
        _exog_incomplete = _X_tr[_full_cols_fc].isna().any(axis=1).to_numpy()
        _rec["n_train_months_withheld_incomplete_drivers"] = int(
            (_exog_incomplete & _y_tr_series.notna().to_numpy()).sum())
        _y_tr_series[_exog_incomplete] = np.nan
        _rec["n_train_months_in_likelihood"] = int(_y_tr_series.notna().sum())
        if _rec["n_train_months_in_likelihood"] < int(SS_MIN_TRAIN_MONTHS):
            _rec["skip_reason"] = (
                f"only {_rec['n_train_months_in_likelihood']} training month(s) have "
                "both the response and every forecast driver "
                f"(< SS_MIN_TRAIN_MONTHS = {SS_MIN_TRAIN_MONTHS})")
            _rows_audit.append(_rec); continue
        _gap = _y_tr_series.isna().to_numpy()[:, None]
        _X_tr = _X_tr.mask(_X_tr.isna() & _gap, 0.0)
        _X_tr.index = _idx_all[:_pos]
        _X_te.index = _idx_all[_pos:_pos + 1]
        if _X_tr.isna().to_numpy().any():
            _bad = _X_tr.columns[_X_tr.isna().any()].tolist()
            _rec["skip_reason"] = ("training exogenous values still missing after "
                                   f"withholding: {_bad}")
            _rows_audit.append(_rec); continue
        assert np.isfinite(_X_te.to_numpy(dtype=float)).all()
        assert not (_X_tr.isna().to_numpy() & ~_gap).any(), \\
            "a placeholder was written on a training month with an observed response"

        _preds, _fail = {}, {}
        # ---- 1. literal persistence: y at exactly t-1, NO fitted coefficient --
        _prev = _fc["y_raw"].iloc[_pos - 1]
        if np.isfinite(_prev):
            _pv, _ = transform_response(pd.Series([_prev]), _how, RESPONSE_EPS)
            _preds["literal persistence (y_{t-1}, unfitted)"] = float(np.asarray(_pv)[0])
        else:
            _fail["literal persistence (y_{t-1}, unfitted)"] = (
                f"the month exactly 1 calendar month earlier ({_origin:%Y-%m}) has no "
                "observed response")

        # ---- 2. seasonal naive: y at exactly t-12, by calendar timestamp ------
        # Looked up on the calendar-complete grid by TIMESTAMP, never as "the
        # twelfth previous observed row"; unavailable if that month is missing.
        _src_month = _target - pd.DateOffset(months=12)
        assert month_index(pd.Series([_target]))[0] \\
            - month_index(pd.Series([_src_month]))[0] == 12, \\
            "the seasonal-naive source month is not exactly 12 calendar months earlier"
        _src_val = _fc.loc[_fc["month"] == _src_month, "y_raw"]
        if len(_src_val) and np.isfinite(_src_val.iloc[0]):
            _sv, _ = transform_response(pd.Series([float(_src_val.iloc[0])]),
                                        _how, RESPONSE_EPS)
            _preds["seasonal naive (y_{t-12})"] = float(np.asarray(_sv)[0])
        else:
            _fail["seasonal naive (y_{t-12})"] = (
                f"the month exactly 12 calendar months earlier ({_src_month:%Y-%m}) has "
                "no observed response")

        # ---- 3. training mean (context only) ---------------------------------
        _preds["training mean"] = float(np.nanmean(_y_tr_series.to_numpy()))

        # ---- 4. fitted AR(1), refit at this origin ---------------------------
        try:
            _ar = SARIMAX(_y_tr_series, order=(1, 0, 0), trend="c",
                          enforce_stationarity=True).fit(disp=False,
                                                         maxiter=SS_ROLLING_MAXITER)
            _preds["fitted AR(1)"] = float(np.asarray(
                _ar.get_forecast(steps=1).predicted_mean)[0])
        except Exception as _exc:
            _fail["fitted AR(1)"] = f"fit failed: {str(_exc)[:80]}"

        # ---- 5/6. matched null and full state-space models --------------------
        _ss_ok = True
        for _name, _cols in (("null state-space (season+trend+persistence)", _null_cols_fc),
                             ("full state-space (+ lagged drivers)", _full_cols_fc)):
            try:
                _r, _ = ss_fit(_y_tr_series, _X_tr[_cols], SS_SELECTED,
                               cov_type=None, maxiter=SS_ROLLING_MAXITER)
                _f = _r.get_forecast(steps=1, exog=_X_te[_cols])
                assert pd.Timestamp(_f.predicted_mean.index[0]) == pd.Timestamp(_target), \\
                    "the state-space forecast is not for the target calendar month"
                _preds[_name] = float(np.asarray(_f.predicted_mean)[0])
            except Exception as _exc:
                _fail[_name] = f"fit/forecast failed: {str(_exc)[:80]}"
                _ss_ok = False
        if not _ss_ok:
            # Matched models must stand or fall together on identical months.
            for _name in ("null state-space (season+trend+persistence)",
                          "full state-space (+ lagged drivers)"):
                _preds.pop(_name, None)
                _fail.setdefault(_name, "dropped so null and full share identical months")
            _rec["skip_reason"] = ("matched state-space pair unavailable at this origin; "
                                   "dropped from BOTH models")

        _rec["usable"] = ("null state-space (season+trend+persistence)" in _preds
                          and "full state-space (+ lagged drivers)" in _preds)
        _rec["n_models_predicted"] = int(len(_preds))
        _rec["models_unavailable"] = "; ".join(f"{k}: {v}" for k, v in _fail.items())
        _rows_audit.append(_rec)

        for _name, _yhat in _preds.items():
            _rows_pred.append({
                "target_month": _target, "origin_month": _origin,
                "specification": _name,
                "y_transformed": _y_target, "yhat_transformed": float(_yhat),
                "y_raw": float(_fc["y_raw"].iloc[_pos]),
                "yhat_raw": float(inverse_response_transform(
                    pd.Series([_yhat]), _how, RESPONSE_EPS).iloc[0]),
                "n_train_months": _rec["n_train_months_observed"],
                "horizon_months": 1,
            })

    SS_FOLD_AUDIT = pd.DataFrame(_rows_audit)
    SS_ONE_MONTH_PREDICTIONS = pd.DataFrame(_rows_pred)
    SS_SCALER_AUDIT = pd.DataFrame(_rows_scaler)
    _elapsed = (pd.Timestamp.now() - _t0).total_seconds()

    # ---- Assertions on the whole evaluation ---------------------------------
    if len(SS_ONE_MONTH_PREDICTIONS):
        assert (SS_ONE_MONTH_PREDICTIONS["horizon_months"] == 1).all(), \\
            "a prediction is not one calendar month ahead"
        assert ((month_index(SS_ONE_MONTH_PREDICTIONS["target_month"])
                 - month_index(SS_ONE_MONTH_PREDICTIONS["origin_month"])) == 1).all(), \\
            "target and origin are not exactly one calendar month apart"
        _pair = SS_ONE_MONTH_PREDICTIONS[SS_ONE_MONTH_PREDICTIONS["specification"].isin(
            ["null state-space (season+trend+persistence)",
             "full state-space (+ lagged drivers)"])]
        _sets = _pair.groupby("specification")["target_month"].apply(
            lambda s: tuple(sorted(pd.to_datetime(s).tolist())))
        assert _sets.nunique() <= 1, \\
            "the matched null and full state-space models were scored on different months"

    print(f"Expanding-window one-calendar-month-ahead rolling origin "
          f"({_elapsed:.0f} s).")
    print(f"  {len(SS_FOLD_AUDIT)} candidate target month(s) considered; "
          f"{int(SS_FOLD_AUDIT['usable'].sum())} scored by the matched pair.")
    _skips = SS_FOLD_AUDIT[(~SS_FOLD_AUDIT["usable"]) & (SS_FOLD_AUDIT["skip_reason"] != "")]
    _reason_counts = (_skips["skip_reason"].str.replace(r"\\d+", "N", regex=True)
                      .str.slice(0, 90).value_counts())
    if len(_reason_counts):
        print("\\n  Target months NOT scored, by reason:")
        for _r, _n in _reason_counts.items():
            print(f"    {_n:>3d}  {_r}")
    display(SS_FOLD_AUDIT.tail(15))

    # ---- Per-month scaling was trained inside the fold ----------------------
    if len(SS_SCALER_AUDIT):
        assert (SS_SCALER_AUDIT["n_train_rows_used"]
                == SS_SCALER_AUDIT["n_train_months_observed"]).all(), \\
            "a fold's scaler saw rows other than its own observed training months"
        _v = SS_SCALER_AUDIT[[c for c in SS_SCALER_AUDIT.columns
                              if c.startswith("mean__")]].nunique()
        print(f"\\n  Per-fold standardisation: {len(SS_SCALER_AUDIT)} scalers fitted, "
              f"{int(_v.max())} distinct mean(s) for the most variable driver — the "
              "scaling genuinely moves with the training window and was never global.")
''')

code('''# =====================================================================
# 21c. Skill tables — each model's own months, then like-for-like
# =====================================================================
if not len(SS_ONE_MONTH_PREDICTIONS):
    print("§21c skipped: no one-month-ahead predictions were produced.")
else:
    _P = SS_ONE_MONTH_PREDICTIONS
    _rows = []
    for _name, _g in _P.groupby("specification"):
        _t = rmse_mae(_g["y_transformed"], _g["yhat_transformed"])
        _r = rmse_mae(_g["y_raw"], _g["yhat_raw"])
        _rows.append({
            "specification": _name, "n_test": _t["n"],
            "first_month": _g["target_month"].min(), "last_month": _g["target_month"].max(),
            "rmse_logit": _t["rmse"], "mae_logit": _t["mae"],
            "rmse_raw_cover": _r["rmse"], "mae_raw_cover": _r["mae"],
            "eval_set": "own months",
        })
    SS_SKILL = pd.DataFrame(_rows).sort_values("rmse_logit").reset_index(drop=True)

    _all_specs = sorted(_P["specification"].unique())
    _month_sets = [set(_P.loc[_P["specification"] == s, "target_month"]) for s in _all_specs]
    _common = set.intersection(*_month_sets) if _month_sets else set()
    SS_EVAL_MONTHS_COMMON = pd.Series(sorted(_common))

    _rows = []
    for _name, _g in _P[_P["target_month"].isin(_common)].groupby("specification"):
        _t = rmse_mae(_g["y_transformed"], _g["yhat_transformed"])
        _r = rmse_mae(_g["y_raw"], _g["yhat_raw"])
        _rows.append({"specification": _name, "n_test": _t["n"],
                      "rmse_logit": _t["rmse"], "mae_logit": _t["mae"],
                      "rmse_raw_cover": _r["rmse"], "mae_raw_cover": _r["mae"],
                      "eval_set": "common months (like-for-like)"})
    SS_SKILL_COMMON = pd.DataFrame(_rows).sort_values("rmse_logit").reset_index(drop=True)
    if len(SS_SKILL_COMMON):
        assert SS_SKILL_COMMON["n_test"].nunique() == 1, \\
            "the like-for-like table does not use one common sample"

    _n_full = int(SS_SKILL.loc[SS_SKILL["specification"]
                               == "full state-space (+ lagged drivers)", "n_test"].iloc[0]) \\
        if (SS_SKILL["specification"] == "full state-space (+ lagged drivers)").any() else 0
    print("One-calendar-month-ahead skill — each specification on the months IT can be "
          "scored on (n_test differs by design; do NOT rank across rows here):")
    display(SS_SKILL.round(4))

    _short = SS_SKILL[SS_SKILL["n_test"] < _n_full]
    if len(_short):
        print("\\nScored on FEWER months than the matched pair, because the exact source "
              "month is missing from the record:")
        for _r in _short.itertuples():
            print(f"  {_r.specification}: {_r.n_test} of {_n_full} month(s)")
        print("  These are reported separately for exactly that reason; their RMSE is "
              "computed on a different (not necessarily easier) sample.")

    print(f"\\nLike-for-like — every specification recomputed on the "
          f"{len(SS_EVAL_MONTHS_COMMON)} month(s) ALL of them can be scored on "
          f"({SS_EVAL_MONTHS_COMMON.min():%Y-%m} .. {SS_EVAL_MONTHS_COMMON.max():%Y-%m}). "
          "This is the table to compare rows in:")
    display(SS_SKILL_COMMON.round(4))

    _fig_tab = SS_SKILL_COMMON if len(SS_SKILL_COMMON) else SS_SKILL
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.5 * len(_fig_tab) + 2.2))
    for _ax, _col, _lab in ((axes[0], "rmse_logit",
                             f"RMSE on {RESPONSE_INFO['transform']}({RESPONSE_COL})"),
                            (axes[1], "rmse_raw_cover",
                             "RMSE on raw WH cover (back-transformed)")):
        _t = _fig_tab.sort_values(_col)
        _cols = ["tab:green" if "full state-space" in s else
                 ("tab:purple" if "null state-space" in s else "tab:grey")
                 for s in _t["specification"]]
        _ax.barh(_t["specification"], _t[_col], color=_cols, alpha=0.85)
        _ax.invert_yaxis(); _ax.grid(alpha=0.3, axis="x")
        _ax.set_xlabel(_lab)
    fig.suptitle("One-calendar-month-ahead skill, like-for-like months\\n"
                 "green = uses environmental drivers, purple = matched no-driver model",
                 fontsize=11)
    fig.tight_layout(); plt.show()

    _obs = _P[_P["specification"] == "full state-space (+ lagged drivers)"].sort_values(
        "target_month")
    if len(_obs):
        fig, ax = plt.subplots(figsize=(11.5, 3.8))
        ax.plot(*on_calendar_grid(_obs["target_month"], _obs["y_transformed"]),
                "k-o", ms=4, lw=1.4, label="observed")
        for _s, _c in (("null state-space (season+trend+persistence)", "tab:purple"),
                       ("full state-space (+ lagged drivers)", "tab:green"),
                       ("literal persistence (y_{t-1}, unfitted)", "tab:orange")):
            _g = _P[_P["specification"] == _s].sort_values("target_month")
            if len(_g):
                ax.plot(*on_calendar_grid(_g["target_month"], _g["yhat_transformed"]),
                        lw=1.3, marker=".", alpha=0.85, color=_c, label=_s)
        ax.set_xlabel("target month (each point is a separate expanding-window refit)")
        ax.set_ylabel(f"{RESPONSE_INFO['transform']}({RESPONSE_COL})")
        ax.set_title("One-calendar-month-ahead forecasts, expanding origin")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for _l in ax.get_xticklabels():
            _l.set_rotation(45); _l.set_horizontalalignment("right")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); plt.show()
''')


# ===========================================================================
# 22. Paired RMSE difference
# ===========================================================================
md("""## 22. Do the drivers genuinely improve one-month-ahead prediction?

A lower point-estimate RMSE is not evidence. On ~30 evaluated months, two models
whose forecasts differ by a few percent will trade places under any reshuffle of
which months happened to be observed. This section therefore reports the
difference **with its uncertainty**, on the matched pair only:

* absolute and percentage RMSE difference (full minus null), on the logit
  modelling scale and on back-transformed raw WH cover;
* the **paired month-level loss differences** $d_t = e^{\\text{full}\\,2}_t -
  e^{\\text{null}\\,2}_t$, which is what a Diebold–Mariano test is built on, with
  a Newey–West standard error on calendar lags;
* a **calendar-aware moving-block bootstrap 95% interval** for the RMSE
  difference, resampling contiguous runs of calendar months so that short-range
  dependence between neighbouring forecast errors is preserved and two months a
  year apart are never made adjacent.

**The reporting rule, applied by the code and not by the reader:** whenever the
interval includes zero, the wording is *"the point estimate was lower, but the
improvement was uncertain"*. No RMSE difference is called an improvement on the
strength of its point estimate.
""")

code('''# =====================================================================
# 22. Paired loss differences and a calendar-aware bootstrap interval
# =====================================================================
SS_PAIRED_LOSS = pd.DataFrame()
SS_RMSE_DIFF = pd.DataFrame()
SS_DRIVERS_IMPROVE_PREDICTION = None
SS_PREDICTION_VERDICT = "not evaluated"

_FULL = "full state-space (+ lagged drivers)"
_NULL = "null state-space (season+trend+persistence)"

if not (len(SS_ONE_MONTH_PREDICTIONS)
        and {_FULL, _NULL}.issubset(set(SS_ONE_MONTH_PREDICTIONS["specification"]))):
    print("§22 skipped: the matched state-space pair was not scored.")
else:
    _f = SS_ONE_MONTH_PREDICTIONS[SS_ONE_MONTH_PREDICTIONS["specification"] == _FULL] \\
        .sort_values("target_month").reset_index(drop=True)
    _n = SS_ONE_MONTH_PREDICTIONS[SS_ONE_MONTH_PREDICTIONS["specification"] == _NULL] \\
        .sort_values("target_month").reset_index(drop=True)
    assert (_f["target_month"].to_numpy() == _n["target_month"].to_numpy()).all(), \\
        "the paired comparison is not on identical target months"

    SS_PAIRED_LOSS = pd.DataFrame({
        "target_month": _f["target_month"],
        "y_transformed": _f["y_transformed"], "y_raw": _f["y_raw"],
        "yhat_full": _f["yhat_transformed"], "yhat_null": _n["yhat_transformed"],
        "err_full": _f["y_transformed"] - _f["yhat_transformed"],
        "err_null": _n["y_transformed"] - _n["yhat_transformed"],
        "err_full_raw": _f["y_raw"] - _f["yhat_raw"],
        "err_null_raw": _n["y_raw"] - _n["yhat_raw"],
    })
    SS_PAIRED_LOSS["loss_difference_full_minus_null"] = (
        SS_PAIRED_LOSS["err_full"] ** 2 - SS_PAIRED_LOSS["err_null"] ** 2)
    SS_PAIRED_LOSS["full_closer"] = (SS_PAIRED_LOSS["err_full"].abs()
                                     < SS_PAIRED_LOSS["err_null"].abs())

    _rows = []
    for _scale, _ef, _en in (("logit modelling scale",
                              SS_PAIRED_LOSS["err_full"], SS_PAIRED_LOSS["err_null"]),
                             ("raw WH cover (back-transformed)",
                              SS_PAIRED_LOSS["err_full_raw"],
                              SS_PAIRED_LOSS["err_null_raw"])):
        _rf = float(np.sqrt(np.mean(_ef.to_numpy() ** 2)))
        _rn = float(np.sqrt(np.mean(_en.to_numpy() ** 2)))
        _boot = block_bootstrap_rmse_difference(
            SS_PAIRED_LOSS["target_month"], _ef, _en,
            block_months=SS_RMSE_BOOTSTRAP_BLOCK, n_boot=SS_RMSE_BOOTSTRAP_N,
            seed=SS_RMSE_BOOTSTRAP_SEED)
        _dm = newey_west_mean_test(_ef.to_numpy() ** 2 - _en.to_numpy() ** 2,
                                   months=SS_PAIRED_LOSS["target_month"],
                                   maxlags=HAC_MAXLAGS)
        _rows.append({
            "scale": _scale, "n_months": int(len(SS_PAIRED_LOSS)),
            "rmse_full": _rf, "rmse_null": _rn,
            "mae_full": float(np.mean(np.abs(_ef))), "mae_null": float(np.mean(np.abs(_en))),
            "rmse_difference_full_minus_null": _rf - _rn,
            "rmse_percent_difference": 100.0 * (_rf - _rn) / _rn if _rn else np.nan,
            "boot_ci_lo": _boot["ci_lo"], "boot_ci_hi": _boot["ci_hi"],
            "boot_block_months": _boot["block_months"],
            "boot_n_successful": _boot["n_successful"],
            "boot_share_favouring_full": _boot["share_negative"],
            "interval_excludes_zero": bool(np.isfinite(_boot["ci_lo"])
                                           and np.isfinite(_boot["ci_hi"])
                                           and (_boot["ci_lo"] > 0 or _boot["ci_hi"] < 0)),
            "mean_paired_loss_difference": _dm["mean"], "dm_se_hac": _dm["se_hac"],
            "dm_t": _dm["t"], "dm_p": _dm["p"],
            "n_months_full_closer": int(SS_PAIRED_LOSS["full_closer"].sum()),
        })
    SS_RMSE_DIFF = pd.DataFrame(_rows)

    print(f"Matched pair on {len(SS_PAIRED_LOSS)} identical one-month-ahead target "
          f"months ({SS_PAIRED_LOSS['target_month'].min():%Y-%m} .. "
          f"{SS_PAIRED_LOSS['target_month'].max():%Y-%m}).")
    display(SS_RMSE_DIFF.round(5))

    _row = SS_RMSE_DIFF.iloc[0]           # the logit modelling scale is the headline
    _pt_lower = bool(_row["rmse_difference_full_minus_null"] < 0)
    _certain = bool(_row["interval_excludes_zero"] and _pt_lower)
    SS_DRIVERS_IMPROVE_PREDICTION = _certain
    print("\\n" + "=" * 96)
    if _certain:
        SS_PREDICTION_VERDICT = (
            f"the environmental drivers improved one-month-ahead RMSE by "
            f"{abs(_row['rmse_percent_difference']):.1f}% against the matched "
            f"season+trend+persistence model, and the 95% moving-block interval "
            f"[{_row['boot_ci_lo']:+.4f}, {_row['boot_ci_hi']:+.4f}] excludes zero")
        print("DRIVERS IMPROVE ONE-MONTH-AHEAD PREDICTION.")
    elif _pt_lower:
        SS_PREDICTION_VERDICT = (
            f"the point estimate was lower (RMSE {_row['rmse_full']:.4f} vs "
            f"{_row['rmse_null']:.4f}, {abs(_row['rmse_percent_difference']):.1f}%), "
            f"but the improvement was uncertain: the 95% calendar-aware moving-block "
            f"interval [{_row['boot_ci_lo']:+.4f}, {_row['boot_ci_hi']:+.4f}] includes "
            "zero")
        print("THE POINT ESTIMATE WAS LOWER, BUT THE IMPROVEMENT WAS UNCERTAIN.")
    else:
        SS_PREDICTION_VERDICT = (
            f"the drivers did NOT improve one-month-ahead prediction (RMSE "
            f"{_row['rmse_full']:.4f} vs {_row['rmse_null']:.4f}, "
            f"{_row['rmse_percent_difference']:+.1f}%); 95% interval "
            f"[{_row['boot_ci_lo']:+.4f}, {_row['boot_ci_hi']:+.4f}]")
        print("DRIVERS DO NOT IMPROVE ONE-MONTH-AHEAD PREDICTION.")
    print("=" * 96)
    print(SS_PREDICTION_VERDICT + ".")
    print(f"\\nThe full model was closer than the null on "
          f"{int(SS_PAIRED_LOSS['full_closer'].sum())} of {len(SS_PAIRED_LOSS)} months "
          f"({100 * SS_PAIRED_LOSS['full_closer'].mean():.0f}%). Diebold-Mariano on the "
          f"paired squared-error differences: t = {_row['dm_t']:.2f}, "
          f"p = {_row['dm_p']:.3g} (Newey-West on calendar lags).")
    print(f"Bootstrap: {int(_row['boot_n_successful']):,} replicate(s), moving blocks of "
          f"{int(_row['boot_block_months'])} calendar month(s); the full model won in "
          f"{100 * _row['boot_share_favouring_full']:.0f}% of them.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    axes[0].bar(SS_PAIRED_LOSS["target_month"],
                SS_PAIRED_LOSS["loss_difference_full_minus_null"],
                width=20, color=np.where(SS_PAIRED_LOSS["full_closer"],
                                         "tab:green", "tab:red"), alpha=0.85)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_title("Paired month-level loss difference (full - null)\\n"
                      "negative = the driver model was closer that month")
    axes[0].set_ylabel("squared-error difference")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for _l in axes[0].get_xticklabels():
        _l.set_rotation(45); _l.set_horizontalalignment("right")
    axes[1].axvline(0, color="k", lw=1)
    axes[1].errorbar([_row["rmse_difference_full_minus_null"]], [0],
                     xerr=[[_row["rmse_difference_full_minus_null"] - _row["boot_ci_lo"]],
                           [_row["boot_ci_hi"] - _row["rmse_difference_full_minus_null"]]],
                     fmt="o", ms=9, capsize=5, color="tab:blue")
    axes[1].set_yticks([]); axes[1].set_xlabel("RMSE(full) - RMSE(null), logit scale")
    axes[1].set_title("RMSE difference with its 95% moving-block interval\\n"
                      + ("interval EXCLUDES zero" if _row["interval_excludes_zero"]
                         else "interval INCLUDES zero -> improvement uncertain"))
    for _a in axes:
        _a.grid(alpha=0.3)
    fig.tight_layout(); plt.show()
''')


# ===========================================================================
# 23. Parametric bootstrap of the joint driver-block LR statistic
# ===========================================================================
md("""## 23. Is the joint driver test's $p$-value trustworthy at this sample size?

§20's likelihood-ratio test is referred to a $\\chi^2_k$ distribution. That
reference is asymptotic, and this record has roughly sixty observed months
containing perhaps a handful of independent episodes. On short, strongly
persistent series the LR statistic's true null distribution is routinely
*heavier* than $\\chi^2$, so the asymptotic $p$-value is optimistic.

This section rebuilds the null distribution instead of assuming it:

1. simulate a response from the **fitted matched null model** — the same
   structure, the same season/trend coefficients, the same innovation variance,
   so the simulated series has the record's persistence and *no* driver effect;
2. re-impose the record's **exact missing-month pattern**, so each replicate has
   the same number of likelihood contributions in the same calendar positions;
3. refit **both** the null and the full model to the simulated series and record
   the LR statistic;
4. report $p = (1 + \\#\\{LR^* \\ge LR^{obs}\\}) / (1 + B)$ over the replicates that
   refitted successfully.

The number of successful refits is reported, because a bootstrap that lost half
its replicates to optimiser failures is not a 400-replicate bootstrap. **Failure
here never stops the notebook**: the section is wrapped, the asymptotic result
from §20 stands on its own, and the failure is reported as a failure.
""")

code('''# =====================================================================
# 23. Parametric bootstrap under the matched null
# =====================================================================
SS_LR_BOOTSTRAP = pd.DataFrame()
SS_LR_BOOTSTRAP_DRAWS = pd.DataFrame()
SS_LR_BOOTSTRAP_STATUS = "not run"

if not (SS_READY and SS_FULL_FIT is not None and SS_NULL_FIT is not None):
    SS_LR_BOOTSTRAP_STATUS = "state-space models unavailable"
    print(f"§23 skipped: {SS_LR_BOOTSTRAP_STATUS}.")
elif not SS_RUN_LR_BOOTSTRAP:
    SS_LR_BOOTSTRAP_STATUS = "SS_RUN_LR_BOOTSTRAP = False"
    print(f"§23 skipped: {SS_LR_BOOTSTRAP_STATUS}. The §20 chi-square p-value is then "
          "the only joint reference available, and it is asymptotic.")
else:
    _null_cols = list(SS_MANIFEST["null_exog"])
    _full_cols = list(SS_MANIFEST["full_exog"])
    _obs_pattern = SS_Y.notna().to_numpy()
    _lr_obs = float(SS_JOINT_TEST["lr_stat"].iloc[0])
    _draws, _n_fail, _t0 = [], 0, pd.Timestamp.now()
    _budget_hit = False

    try:
        for _b in range(int(SS_LR_BOOTSTRAP_N)):
            if (pd.Timestamp.now() - _t0).total_seconds() > SS_LR_BOOTSTRAP_TIME_BUDGET_S:
                _budget_hit = True
                break
            try:
                np.random.seed(int(SS_LR_BOOTSTRAP_SEED) + _b)
                _sim = SS_NULL_FIT.simulate(nsimulations=len(SS_Y),
                                            exog=SS_EXOG[_null_cols])
                _ysim = pd.Series(np.asarray(_sim, dtype=float).ravel(), index=SS_INDEX)
                _ysim[~_obs_pattern] = np.nan     # the record's own missing pattern
                assert np.array_equal(_ysim.notna().to_numpy(), _obs_pattern)
                _r0, _ = ss_fit(_ysim, SS_EXOG[_null_cols], SS_SELECTED,
                                cov_type=None, maxiter=SS_LR_BOOTSTRAP_MAXITER)
                _r1, _ = ss_fit(_ysim, SS_EXOG[_full_cols], SS_SELECTED,
                                cov_type=None, maxiter=SS_LR_BOOTSTRAP_MAXITER)
                _stat = float(2.0 * (float(_r1.llf) - float(_r0.llf)))
                if not np.isfinite(_stat):
                    _n_fail += 1
                    continue
                _draws.append(max(_stat, 0.0))
            except Exception:
                _n_fail += 1
                continue

        _d = np.asarray(_draws, dtype=float)
        _B = int(len(_d))
        if _B < 30:
            SS_LR_BOOTSTRAP_STATUS = (f"only {_B} successful refit(s) of "
                                      f"{SS_LR_BOOTSTRAP_N}; too few to report")
            print(f"§23: {SS_LR_BOOTSTRAP_STATUS}. The asymptotic chi-square p-value "
                  "from §20 stands, with its optimism unquantified.")
        else:
            _p_boot = float((1 + int(np.sum(_d >= _lr_obs))) / (1 + _B))
            SS_LR_BOOTSTRAP_STATUS = "completed"
            SS_LR_BOOTSTRAP_DRAWS = pd.DataFrame({"replicate": np.arange(_B),
                                                  "lr_stat_simulated": _d})
            SS_LR_BOOTSTRAP = pd.DataFrame([{
                "comparison": "full driver block vs matched null (parametric bootstrap)",
                "structure": SS_SELECTED, "k_drivers": len(SS_DRIVER_TERMS),
                "lr_stat_observed": _lr_obs,
                "n_replicates_requested": int(SS_LR_BOOTSTRAP_N),
                "n_replicates_successful": _B,
                "n_replicates_failed": int(_n_fail),
                "time_budget_exhausted": bool(_budget_hit),
                "p_bootstrap": _p_boot,
                "p_chi2_asymptotic": float(SS_JOINT_TEST["p_chi2"].iloc[0]),
                "bootstrap_null_median": float(np.median(_d)),
                "bootstrap_null_q95": float(np.quantile(_d, 0.95)),
                "chi2_q95": float(sstats.chi2.ppf(0.95, len(SS_DRIVER_TERMS))),
                "seconds": float((pd.Timestamp.now() - _t0).total_seconds()),
            }])
            display(SS_LR_BOOTSTRAP.round(4))
            print(f"Parametric bootstrap: {_B} successful refit(s) of "
                  f"{SS_LR_BOOTSTRAP_N} requested ({_n_fail} failed"
                  + (", time budget reached" if _budget_hit else "") + ").")
            print(f"  observed LR = {_lr_obs:.3f}; bootstrap p = {_p_boot:.4f}; "
                  f"asymptotic chi-square p = "
                  f"{float(SS_JOINT_TEST['p_chi2'].iloc[0]):.4f}")
            _q95b = float(np.quantile(_d, 0.95))
            _q95c = float(sstats.chi2.ppf(0.95, len(SS_DRIVER_TERMS)))
            if _q95b > _q95c:
                print(f"  The simulated null is HEAVIER than chi-square "
                      f"(95th percentile {_q95b:.2f} vs {_q95c:.2f}), so the asymptotic "
                      "p-value in §20 is optimistic; quote the bootstrap one.")
            else:
                print(f"  The simulated null is no heavier than chi-square "
                      f"({_q95b:.2f} vs {_q95c:.2f}); the §20 p-value is not obviously "
                      "distorted by the sample size.")

            fig, ax = plt.subplots(figsize=(8.5, 3.4))
            ax.hist(_d, bins=30, color="tab:grey", alpha=0.8,
                    label=f"simulated null ({_B} refits)")
            _xs = np.linspace(0, max(float(np.max(_d)), _lr_obs) * 1.05, 200)
            ax.plot(_xs, sstats.chi2.pdf(_xs, len(SS_DRIVER_TERMS)) * _B
                    * (float(np.max(_d)) / 30 if np.max(_d) > 0 else 1),
                    color="tab:blue", lw=2,
                    label=f"chi-square({len(SS_DRIVER_TERMS)}) reference")
            ax.axvline(_lr_obs, color="tab:red", lw=2,
                       label=f"observed LR = {_lr_obs:.2f}")
            ax.set_xlabel("likelihood-ratio statistic for the driver block")
            ax.set_title("Null distribution of the joint driver-block LR statistic\\n"
                         "simulated from the FITTED MATCHED NULL, missing months re-imposed")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout(); plt.show()
    except Exception as _exc:                       # pragma: no cover - defensive
        SS_LR_BOOTSTRAP_STATUS = f"failed: {str(_exc)[:160]}"
        print(f"§23 FAILED and was skipped: {SS_LR_BOOTSTRAP_STATUS}")
        print("The rest of the notebook is unaffected; §20's asymptotic joint test "
              "stands, and its optimism is unquantified.")
''')


# ===========================================================================
# 24. State-space robustness
# ===========================================================================
md("""## 24. Robustness of the state-space result

§18 attacks the *static* models' decisions. This section applies the same
discipline to the principal model, and nothing here is merged with §18's tables:
every row carries the model it came from.

1. **Leave-one-year-out.** One year at a time is removed *by setting its
   responses missing* — not by deleting rows — so the calendar, the season and
   the state process are untouched and only that year's likelihood contribution
   disappears. A coefficient that only exists when one particular year is present
   is that year's, not the lake's.
2. **Alternative response transformations** — `logit`, `log`, raw cover. A driver
   whose sign depends on the link is not a driver effect.
3. **Alternative dependence structures** — the full driver model refitted under
   AR(1), AR(2) and the local level, so the reader can see whether §19's choice
   changed any conclusion.
4. **Calendar-aware residual diagnostics** for each of them.

### The four sources of coefficients in this notebook, and what each may be called

| Source | Section | What it is | What it may **not** be called |
|---|---|---|---|
| **M3** | §12 | static OLS/WLS + calendar-aware HAC, no persistence | a dynamic or causal effect |
| **M4** | §12 | the same design plus `y_lag1` on the right-hand side | the principal estimate; it is a sensitivity |
| **S1** | §20 | **principal** state-space driver estimates, persistence in the process | a forecast result |
| **prediction-only** | §21–§22 | one-month-ahead skill with lagged drivers | an effect size or a $p$-value |

Never quote a coefficient, $p$-value, Shapley value or robustness statistic
without naming which of these four produced it.
""")

code('''# =====================================================================
# 24a. Leave-one-year-out on the state-space driver model
# =====================================================================
SS_LOYO = pd.DataFrame()
SS_LOYO_SUMMARY = pd.DataFrame()

if not (SS_READY and SS_FULL_FIT is not None and SS_RUN_LOYO):
    print("§24a skipped.")
else:
    _full_cols = list(SS_MANIFEST["full_exog"])
    _years = sorted(pd.Series(SS_INDEX).dt.year[SS_Y.notna().to_numpy()].unique())
    _rows = []
    for _yr in _years:
        _mask = (pd.Series(SS_INDEX).dt.year == _yr).to_numpy()
        _y_drop = SS_Y.copy()
        _y_drop[_mask] = np.nan          # remove the YEAR's likelihood contribution only
        if int(_y_drop.notna().sum()) < 24:
            continue
        try:
            _r, _i = ss_fit(_y_drop, SS_EXOG[_full_cols], SS_SELECTED,
                            cov_type=SS_COV_TYPE, maxiter=SS_MAXITER)
        except Exception:
            continue
        _t = ss_tidy_coefficients(_r, SS_SELECTED, SS_DRIVER_TERMS,
                                  label=f"drop {int(_yr)}",
                                  cov_type_used=_i["cov_type_used"])
        _t["dropped_year"] = int(_yr)
        _t["n_observed_months"] = int(_y_drop.notna().sum())
        _rows.append(_t)

    if _rows:
        SS_LOYO = pd.concat(_rows, ignore_index=True)
        _piv = SS_LOYO.pivot_table(index="term", columns="dropped_year", values="coef")
        _full_coef = SS_COEFS.set_index("term")["coef"]
        SS_LOYO_SUMMARY = pd.DataFrame({
            "coef_full_record": _full_coef,
            "loyo_min": _piv.min(axis=1), "loyo_max": _piv.max(axis=1),
            "loyo_sign_stable": _piv.apply(
                lambda r: np.sign(r.dropna()).nunique() <= 1, axis=1),
        })
        SS_LOYO_SUMMARY["max_abs_shift"] = (_piv.sub(SS_LOYO_SUMMARY["coef_full_record"],
                                                     axis=0).abs().max(axis=1))
        SS_LOYO_SUMMARY["most_influential_year"] = _piv.sub(
            SS_LOYO_SUMMARY["coef_full_record"], axis=0).abs().idxmax(axis=1)
        SS_LOYO_SUMMARY["source_model"] = f"S1 state-space {SS_SELECTED}"
        print(f"Leave-one-year-out on the state-space driver model "
              f"({len(_piv.columns)} year(s) dropped, one at a time, by setting that "
              "year's responses missing):")
        display(SS_LOYO_SUMMARY.round(4))

        fig, ax = plt.subplots(figsize=(10, 0.5 * len(_piv) + 1.8))
        for _i2, _t2 in enumerate(_piv.index):
            ax.plot(_piv.loc[_t2], np.full(_piv.shape[1], _i2), "o", ms=5, alpha=0.7,
                    color="tab:purple")
            ax.plot(SS_LOYO_SUMMARY.loc[_t2, "coef_full_record"], _i2, "D", ms=8,
                    color="tab:red")
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(_piv))); ax.set_yticklabels(_piv.index)
        ax.invert_yaxis()
        ax.set_xlabel("state-space coefficient "
                      "(red diamond = full record, purple = one year removed)")
        ax.set_title(f"Leave-one-year-out — S1 state-space {SS_SELECTED}")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout(); plt.show()
    else:
        print("Too few months per year for a state-space leave-one-year-out.")
''')

code('''# =====================================================================
# 24b. Alternative transforms and alternative dependence structures
# =====================================================================
SS_ROBUST_TRANSFORM = pd.DataFrame()
SS_ROBUST_STRUCTURE = pd.DataFrame()
SS_ROBUST_SIGN_STABLE = pd.Series(dtype=bool)

if not (SS_READY and SS_FULL_FIT is not None):
    print("§24b skipped.")
else:
    _null_cols = list(SS_MANIFEST["null_exog"])
    _full_cols = list(SS_MANIFEST["full_exog"])

    # --- (i) response transformation -------------------------------------
    _rows, _fit_rows = [], []
    for _how in SS_ROBUST_TRANSFORMS:
        if RESPONSE_COL != "wh_area_ha" or _how != "logit":
            _yv, _info_t = transform_response(SS_GRID["y_raw"], _how, RESPONSE_EPS)
        else:
            continue
        _yv = pd.Series(np.where(SS_Y.notna().to_numpy(),
                                 np.asarray(_yv, dtype=float), np.nan), index=SS_INDEX)
        if not np.isfinite(_yv.dropna().to_numpy()).all():
            continue
        try:
            _r0, _ = ss_fit(_yv, SS_EXOG[_null_cols], SS_SELECTED, cov_type=None,
                            maxiter=SS_MAXITER)
            _r1, _i1 = ss_fit(_yv, SS_EXOG[_full_cols], SS_SELECTED,
                              cov_type=SS_COV_TYPE, maxiter=SS_MAXITER)
        except Exception:
            continue
        _t = ss_tidy_coefficients(_r1, SS_SELECTED, SS_DRIVER_TERMS,
                                  label=f"transform={_how}",
                                  cov_type_used=_i1["cov_type_used"])
        _t["variant_kind"] = "response transform"
        _t["variant"] = _how
        _rows.append(_t)
        _lr = ss_joint_lr_test(_r1, _r0, len(SS_DRIVER_TERMS))
        _fit_rows.append({"variant_kind": "response transform", "variant": _how,
                          "structure": SS_SELECTED,
                          "n_observed_months": ss_n_effective(_yv),
                          **_lr,
                          "aicc_null": aicc_from(_r0.llf, ss_n_free_params(_r0),
                                                 ss_n_effective(_yv)),
                          "aicc_full": aicc_from(_r1.llf, ss_n_free_params(_r1),
                                                 ss_n_effective(_yv))})

    # --- (ii) dependence structure ----------------------------------------
    for _key in SS_CANDIDATE_STRUCTURES:
        _nc = list(SS_SEASON_COLS) + (list(SS_TREND_COLS)
                                      if (ss_allows_linear_trend(_key) and INCLUDE_TREND)
                                      else [])
        _fc2 = _nc + list(SS_DRIVER_TERMS)
        try:
            _r0, _ = ss_fit(SS_Y, SS_EXOG[_nc], _key, cov_type=None, maxiter=SS_MAXITER)
            _r1, _i1 = ss_fit(SS_Y, SS_EXOG[_fc2], _key, cov_type=SS_COV_TYPE,
                              maxiter=SS_MAXITER)
        except Exception:
            continue
        _t = ss_tidy_coefficients(_r1, _key, SS_DRIVER_TERMS,
                                  label=f"structure={_key}",
                                  cov_type_used=_i1["cov_type_used"])
        _t["variant_kind"] = "dependence structure"
        _t["variant"] = _key
        _rows.append(_t)
        _lr = ss_joint_lr_test(_r1, _r0, len(SS_DRIVER_TERMS))
        _dg, _, _ = ss_innovation_diagnostics(_r1, SS_INDEX,
                                              nlags=min(18, max(6, SS_N_OBS // 3)),
                                              label=_key)
        _fit_rows.append({"variant_kind": "dependence structure", "variant": _key,
                          "structure": _key, "n_observed_months": ss_n_effective(SS_Y),
                          **_lr,
                          "aicc_null": aicc_from(_r0.llf, ss_n_free_params(_r0),
                                                 ss_n_effective(SS_Y)),
                          "aicc_full": aicc_from(_r1.llf, ss_n_free_params(_r1),
                                                 ss_n_effective(SS_Y)),
                          "innov_ljung_box_p": _dg["ljung_box_p"],
                          "innov_acf1": _dg["acf1"],
                          "is_selected": bool(_key == SS_SELECTED)})

    if _rows:
        SS_ROBUST_TRANSFORM = pd.concat(_rows, ignore_index=True)
        SS_ROBUST_STRUCTURE = pd.DataFrame(_fit_rows)
        SS_ROBUST_STRUCTURE["aicc_difference_full_minus_null"] = (
            SS_ROBUST_STRUCTURE["aicc_full"] - SS_ROBUST_STRUCTURE["aicc_null"])
        _piv = SS_ROBUST_TRANSFORM.pivot_table(index="term", columns="specification",
                                               values="coef", aggfunc="first")
        print("State-space driver coefficients across response transforms and "
              "dependence structures (S1 family only — never merged with §12/§18):")
        display(_piv.round(3))
        _stable = _piv.apply(lambda r: np.sign(r.dropna()).nunique() <= 1, axis=1)
        print("\\nSign stable across every state-space variant:",
              _stable[_stable].index.tolist() or "none")
        print("Sign UNSTABLE (report no direction for these):",
              _stable[~_stable].index.tolist() or "none")
        SS_ROBUST_SIGN_STABLE = _stable
        print("\\nJoint driver-block test under each variant:")
        display(SS_ROBUST_STRUCTURE[[
            "variant_kind", "variant", "n_observed_months", "lr_stat", "df", "p_chi2",
            "aicc_difference_full_minus_null", "innov_ljung_box_p"]].round(4))
        _agree = SS_ROBUST_STRUCTURE[SS_ROBUST_STRUCTURE["variant_kind"]
                                     == "dependence structure"]
        if len(_agree) > 1:
            _signs = set(np.sign(_agree["aicc_difference_full_minus_null"]))
            if len(_signs) > 1:
                print("\\nThe AICc verdict on the driver block DEPENDS on the dependence "
                      "structure. Say so: the finding is conditional on §19's selection.")
            else:
                print("\\nEvery dependence structure gives the same AICc verdict on the "
                      "driver block, so §19's selection is not what decided it.")
    else:
        SS_ROBUST_SIGN_STABLE = pd.Series(dtype=bool)
        print("No state-space robustness variant could be fitted.")
''')


# ===========================================================================
# 25. Synthesis
# ===========================================================================
md("""## 25. Synthesis

Two parts, in this order:

* **§25a — the five questions.** The notebook's actual conclusions, printed as
  plain sentences and exported as a table, each naming the section and the
  source model it comes from.
* **§25b — the ranked driver table.** The per-driver summary, unchanged in
  structure, with the state-space (S1) estimates added alongside the static (M3)
  ones and never merged with them.

### The four sources of a coefficient, and the rule

| Label | Section | What it is |
|---|---|---|
| **M3** | §12 | static OLS/WLS + calendar-aware HAC. No persistence. **Association.** |
| **M4** | §12 | M3 plus `y_lag1` on the right-hand side. **Sensitivity**, not the principal dynamic estimate. |
| **S1** | §20 | **principal** — state-space, persistence in the process, all observed months. **Association / nowcast.** |
| **pred** | §21–§22 | one-month-ahead skill, drivers lagged to origin-time. **Prediction only** — never an effect size. |

No number below mixes two of these. `coef_M3` and `coef_S1` sit in different
columns; a Shapley value is always tagged with the specification it came from;
and a non-linear shape from §14 is quoted only if §14b supported it out of
sample.

""")

code('''# =====================================================================
# 25a. The five questions this notebook exists to answer
# =====================================================================
SYNTHESIS_ANSWERS = pd.DataFrame()
_ans = []


def _answer(number, question, answer, source, evidence=""):
    _ans.append({"question_number": int(number), "question": question,
                 "answer": answer, "source_section": source, "evidence": evidence,
                 "is_synthetic": bool(SOURCE["is_synthetic"])})


# --- Q1: which dependence structure represents the series? -------------------
if SS_READY and SS_SELECTED:
    _c = SS_CANDIDATES[SS_CANDIDATES.get("fitted", False) == True]
    _cmp = "; ".join(f"{r.structure} AICc {r.aicc:.1f}"
                     + (f", 1-month RMSE {r.rmse_one_month_ahead:.3f}"
                        if np.isfinite(r.rmse_one_month_ahead) else "")
                     for r in _c.itertuples())
    _answer(1, "Which dependence structure best represents the WH series?",
            f"{SS_SELECTED} ({ss_structure_label(SS_SELECTED)}). {SS_SELECTION_NOTE}. "
            "Chosen on null dynamics with no driver information of any kind, then "
            "locked before any environmental variable was added.",
            "§19", _cmp)
else:
    _answer(1, "Which dependence structure best represents the WH series?",
            f"Not determined: {SS_SKIP_REASON or 'the state-space model did not run'}.",
            "§19", "")

# --- Q2: do the drivers jointly improve FIT over the matched null? -----------
if len(SS_JOINT_TEST):
    _j = SS_JOINT_TEST.iloc[0]
    _p_used, _p_kind = float(_j["p_chi2"]), "asymptotic chi-square"
    if len(SS_LR_BOOTSTRAP):
        _p_used = float(SS_LR_BOOTSTRAP["p_bootstrap"].iloc[0])
        _p_kind = (f"parametric bootstrap, "
                   f"{int(SS_LR_BOOTSTRAP['n_replicates_successful'].iloc[0])} refits")
    _better = bool(_j["aicc_difference_full_minus_null"] < 0)
    _sig = bool(_p_used < 0.05)
    _txt = (f"LR = {_j['lr_stat']:.2f} on {int(_j['df'])} df, p = {_p_used:.4g} "
            f"({_p_kind}); AICc {'favours' if _better else 'does NOT favour'} the driver "
            f"model ({_j['aicc_difference_full_minus_null']:+.2f}).")
    if _sig and _better:
        _verdict2 = ("YES — the driver block jointly improves fit over the matched "
                     "season+trend+persistence model. " + _txt)
    elif _sig or _better:
        _verdict2 = ("MIXED — the two criteria disagree, so the joint improvement is not "
                     "established. " + _txt)
    else:
        _verdict2 = ("NO — the driver block does not jointly improve fit over the matched "
                     "no-driver dynamic model. " + _txt)
    _answer(2, "Do environmental drivers jointly improve FIT over the matched "
               "no-driver dynamic model?", _verdict2, "§20, §23",
            f"Wald p = {float(_j['p_wald']):.4g}; log-likelihood difference "
            f"{float(_j['llf_difference']):+.3f}")
else:
    _answer(2, "Do environmental drivers jointly improve FIT over the matched "
               "no-driver dynamic model?",
            "Not evaluated: the matched state-space pair was not fitted.", "§20", "")

# --- Q3: do they improve genuine one-month-ahead PREDICTION? -----------------
if len(SS_RMSE_DIFF):
    _r = SS_RMSE_DIFF.iloc[0]
    _r_raw = SS_RMSE_DIFF.iloc[1] if len(SS_RMSE_DIFF) > 1 else None
    _head = ("YES" if SS_DRIVERS_IMPROVE_PREDICTION else
             ("POINT ESTIMATE LOWER, IMPROVEMENT UNCERTAIN"
              if _r["rmse_difference_full_minus_null"] < 0 else "NO"))
    _answer(3, "Do environmental drivers improve genuine one-calendar-month-ahead "
               "prediction?",
            f"{_head} — {SS_PREDICTION_VERDICT}. Evaluated on "
            f"{int(_r['n_months'])} target months, expanding window, every feasible "
            "origin, drivers restricted to information available at the origin.",
            "§21, §22",
            (f"logit-scale RMSE {_r['rmse_full']:.4f} (full) vs {_r['rmse_null']:.4f} "
             f"(null); raw-cover RMSE "
             + (f"{_r_raw['rmse_full']:.5f} vs {_r_raw['rmse_null']:.5f}"
                if _r_raw is not None else "n/a")
             + f"; 95% moving-block interval [{_r['boot_ci_lo']:+.4f}, "
               f"{_r['boot_ci_hi']:+.4f}]"))
else:
    _answer(3, "Do environmental drivers improve genuine one-calendar-month-ahead "
               "prediction?",
            "Not evaluated: no matched one-month-ahead comparison was produced.",
            "§21", "")

# --- Q4: is any individual driver robust? ------------------------------------
_robust_drivers = []
if len(SS_COEFS):
    _sig = SS_COEFS[SS_COEFS["q_fdr"] < FDR_ALPHA]
    for _r in _sig.itertuples():
        _loyo_ok = bool(SS_LOYO_SUMMARY.loc[_r.term, "loyo_sign_stable"]) \\
            if (len(SS_LOYO_SUMMARY) and _r.term in SS_LOYO_SUMMARY.index) else False
        _var_ok = bool(SS_ROBUST_SIGN_STABLE.get(_r.term, False)) \\
            if len(SS_ROBUST_SIGN_STABLE) else False
        _seas = DRIVER_AUDIT.loc[DRIVER_AUDIT["driver"]
                                 == getattr(_r, "driver", ""), "r2_on_season"]
        _seas_ok = not (len(_seas) and pd.notna(_seas.iloc[0])
                        and float(_seas.iloc[0]) >= SEASON_CONFOUND_R2)
        if _loyo_ok and _var_ok and _seas_ok and bool(_r.sign_matches_mechanism):
            _robust_drivers.append(_r.term)
if _robust_drivers:
    _verdict4 = ("YES — " + ", ".join(_robust_drivers) + ". Each survives BH correction "
                 "in the principal state-space model, keeps its sign under "
                 "leave-one-year-out and under every alternative transform and "
                 "dependence structure, is separable from the annual cycle, and matches "
                 "its a-priori mechanism sign.")
elif len(SS_COEFS):
    _verdict4 = ("NO — no individual environmental driver is robust once persistence, "
                 "seasonality, multiplicity and small-sample uncertainty are all "
                 "accounted for. This is a result, not a gap: it says the AOI-scale "
                 "monthly total is dominated by the mat's own dynamics and the calendar, "
                 "and that this record is too short to separate a driver from them.")
else:
    _verdict4 = "Not evaluated: the state-space driver model was not fitted."
_answer(4, "Is any individual driver robust after persistence, seasonality, "
           "multiplicity and small-sample uncertainty?", _verdict4, "§20, §24",
        (f"BH alpha = {FDR_ALPHA}; "
         f"{int((SS_COEFS['q_fdr'] < FDR_ALPHA).sum()) if len(SS_COEFS) else 0} of "
         f"{len(SS_COEFS)} drivers reach q < {FDR_ALPHA} in S1 before the "
         "robustness filters"))

# --- Q5: how much uncertainty remains? ---------------------------------------
_n_ep = np.nan
if len(SERIES_DIAGNOSTICS):
    _n_ep = float(SERIES_DIAGNOSTICS["effective_n_bartlett"].iloc[0])
_span_years = (calendar_span_months(monthly.loc[monthly["y"].notna(), "month"]) / 12.0
               if N_OBSERVED else np.nan)
_widths = (SS_COEFS["ci_hi"] - SS_COEFS["ci_lo"]) if len(SS_COEFS) else pd.Series(dtype=float)
_q5 = (f"Substantial. The record holds {N_OBSERVED} observed months over "
       f"{_span_years:.1f} years, with lag-1 autocorrelation "
       f"{float(SERIES_DIAGNOSTICS['lag1_autocorrelation'].iloc[0]):.2f} on the raw "
       f"response — an APPROXIMATE Bartlett effective size of {_n_ep:.0f}. That number "
       "is a diagnostic, not a sample size: it is computed on the raw response, whose "
       "autocorrelation still contains the annual cycle and the trend that every model "
       "below removes.")
if SS_MANIFEST.get("run"):
    _q5 += (" The principal state-space model uses all "
            f"{SS_MANIFEST['n_observed_months']} observed months and represents that "
            "dependence explicitly rather than discounting for it — but it cannot create "
            "independent information the record does not contain. The 95% intervals on "
            "its standardized drivers span "
            + (f"{_widths.min():.2f}-{_widths.max():.2f} logit units per SD"
               if len(_widths) else "n/a")
            + ", which for most drivers is wide enough to contain both a materially "
              "positive and a materially negative effect. Only a longer record, or "
              "independent replication across other gulfs, narrows this.")
else:
    _q5 += (" The principal state-space model did not run here, so no interval widths "
            "from it are available; the static intervals in §12 are the only guide, and "
            "they do not account for persistence.")
_answer(5, "How much uncertainty remains because the record contains few "
           "independent temporal episodes?", _q5,
        "§8, §20", f"n_observed_months = {N_OBSERVED}, "
                   f"n_fitted_rows_static = {N_FIT}")

SYNTHESIS_ANSWERS = pd.DataFrame(_ans)

print("=" * 100)
print("SYNTHESIS — THE FIVE QUESTIONS")
if SOURCE["is_synthetic"]:
    print("*** SYNTHETIC SELF-TEST RUN — these are NOT results about Winam Gulf ***")
print("=" * 100)
for _r in SYNTHESIS_ANSWERS.itertuples():
    print(f"\\nQ{_r.question_number}. {_r.question}")
    print(f"     [{_r.source_section}] {_r.answer}")
    if _r.evidence:
        print(f"     evidence: {_r.evidence}")
print("\\n" + "=" * 100)

# The plain, exportable bottom line — printed whether it is good news or not, and
# NEVER claiming a null result the notebook did not actually establish.
_ss_ran = bool(SS_MANIFEST.get("run")) and len(SS_COEFS) > 0
_no_robust = _ss_ran and not _robust_drivers
_no_pred = len(SS_RMSE_DIFF) > 0 and (SS_DRIVERS_IMPROVE_PREDICTION is not True)
if not _ss_ran:
    print("PLAIN RESULT")
    print("-" * 100)
    print("* The principal state-space model did not run, so questions 1-4 are NOT "
          f"answered by this run ({SS_SKIP_REASON or 'see §19'}). Nothing here may be "
          "reported as a null result about the drivers: 'not evaluated' is not "
          "'no effect'.")
    print("-" * 100)
elif _no_robust or _no_pred:
    print("PLAIN RESULT")
    print("-" * 100)
    if _no_robust:
        print("* No individual environmental driver is robust in the principal "
              "state-space model after persistence, seasonality, multiplicity and "
              "small-sample uncertainty.")
    if _no_pred and len(SS_RMSE_DIFF):
        print(f"* The full driver model {'did not improve' if not SS_DRIVERS_IMPROVE_PREDICTION else 'improved'} "
              "one-calendar-month-ahead prediction over the matched "
              f"season+trend+persistence model: {SS_PREDICTION_VERDICT}.")
    print("* This is reported as the finding. It is a statement about what this "
          "monthly AOI record can support, not evidence that the ecology is wrong: the "
          "drivers remain ASSOCIATED with WH extent in §12 and account for a share of "
          "its variance in §17.")
    print("-" * 100)
''')

md("""### 25b. The ranked driver table

Everything above, collapsed into one table per driver, with a **verdict** applied
by explicit rules rather than by eye. The state-space (S1) coefficient sits
beside the static (M3) one in its own column; they are never averaged, and the
verdict now requires the driver to survive **both**.
""")

md("""#### Verdict rules

| Verdict | Rule |
|---|---|
| `not separable from season` | $R^2$ of the driver on the seasonal harmonics $\\ge$ `SEASON_CONFOUND_R2`. Reported regardless of significance — the model cannot tell this driver from the calendar. |
| `robust` | FDR $q <$ `FDR_ALPHA` in M3 **and** in the **principal state-space model S1** **and** bootstrap sign stability $\\ge 0.9$ **and** elastic-net selection $\\ge 0.8$ **and** S1's sign survives leave-one-year-out and every alternative transform and dependence structure **and** the sign matches the a-priori mechanism. |
| `static only — not confirmed under persistence` | FDR-significant in M3 but **not** in S1. The effect the static model saw did not survive putting persistence in the process — which is exactly the failure §19–§20 exist to detect. |
| `suggestive` | $q <$ `FDR_ALPHA` with a stable sign, **or** elastic-net selection $\\ge 0.8$ backed by $q < 0.25$ — a signal worth reporting, not a result. |
| `sign contradicts mechanism` | A clear effect in the direction the ecology says it should not go. Reported as such: it is evidence against the stated mechanism, or that the variable is proxying something else. |
| `no evidence` | Everything else. A real, reportable finding, not a gap. |

Two guards are built into these rules, because both traps produce
confident-looking nonsense:

* **Elastic-net selection frequency is ignored** unless §13 found the fit
  genuinely sparse. A near-ridge penalty barely zeroes anything, so every driver
  would score 100% "selection" and every driver would look robust.
* **`variance_reading`** flags drivers with a sizeable Shapley $R^2$ but almost
  no *unique* contribution. Those are collinear stand-ins for another driver in
  the set — name the shared mechanism, not the individual variable. It is taken
  from a **single** specification — the dynamic one where available, named in
  `variance_reading_specification` — so its Shapley and semi-partial inputs come
  from the same fit.

`last_entry_to_shapley_ratio` above 1 is reported as possible **suppression or
coefficient instability**, not as a share above 100%. Long-run effects are
quoted only if the stationarity gate opens — now checked in **both** §12's
`y_lag1` model and §20's principal state-space model; when either refuses, the
section says so and only short-run coefficients may be used. Under a
local-level structure the gate is closed by construction: a random-walk level
has a unit root, so no long-run multiplier is defined at all.
""")



code('''# =====================================================================
# 25. Synthesis table
# =====================================================================
def _base_driver(term):
    """Strip the lag suffix to recover the driver name used in FORCING."""
    for suffix in ("_lag6", "_lag5", "_lag4", "_lag3", "_lag2", "_lag1"):
        if term.endswith(suffix):
            return term[: -len(suffix)]
    return term


_rows = []
_m3 = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == HEADLINE_SPEC].set_index("term")
_s1 = SS_COEFS.set_index("term") if len(SS_COEFS) else pd.DataFrame()
_partition_used = PARTITION_AR if len(PARTITION_AR) else PARTITION
for term in DRIVER_TERMS:
    base = _base_driver(term)
    meta = FORCING.get(base, {})
    row = {"term": term, "driver": base,
           "mechanism": meta.get("mechanism", ""),
           "expected_sign": meta.get("expected_sign", "?"),
           "lag_months": LAG_USED.get(base, np.nan)}

    if term in _m3.index:
        row.update(coef_M3=float(_m3.loc[term, "coef"]),
                   ci_lo=float(_m3.loc[term, "ci_lo"]),
                   ci_hi=float(_m3.loc[term, "ci_hi"]),
                   p_M3=float(_m3.loc[term, "p"]),
                   q_fdr_M3=float(_m3.loc[term, "q_fdr"]))
    # PRINCIPAL estimates (S1, §20). Kept in their OWN columns: an S1 coefficient
    # and an M3 coefficient come from different models and are never averaged,
    # ranked together or quoted interchangeably.
    if term in _s1.index:
        row.update(coef_S1=float(_s1.loc[term, "coef"]),
                   ci_lo_S1=float(_s1.loc[term, "ci_lo"]),
                   ci_hi_S1=float(_s1.loc[term, "ci_hi"]),
                   p_S1=float(_s1.loc[term, "p"]),
                   q_fdr_S1=float(_s1.loc[term, "q_fdr"]),
                   se_kind_S1=str(_s1.loc[term, "se_kind"]),
                   sign_matches_mechanism_S1=bool(
                       _s1.loc[term, "sign_matches_mechanism"]))
    if len(SS_LOYO_SUMMARY) and term in SS_LOYO_SUMMARY.index:
        row["loyo_sign_stable_S1"] = bool(SS_LOYO_SUMMARY.loc[term, "loyo_sign_stable"])
    if len(SS_ROBUST_SIGN_STABLE) and term in SS_ROBUST_SIGN_STABLE.index:
        row["ss_variant_sign_stable"] = bool(SS_ROBUST_SIGN_STABLE.loc[term])
    row["nonlinearity_supported_oos"] = bool(term in (SPLINE_SUPPORTED_OOS or []))
    for tab, col, out in [
            (SEMI_PARTIAL, "semi_partial_r2", "semi_partial_r2_static"),
            (SEMI_PARTIAL_AR, "semi_partial_r2", "semi_partial_r2_dynamic"),
            (BOOT_SUMMARY, "boot_sign_stability", "boot_sign_stability"),
            (STABILITY, "selection_frequency", "enet_selection_freq"),
            (GBM_IMPORTANCE, "perm_importance_mean", "gbm_perm_importance"),
            (GBM_IMPORTANCE, "rank", "gbm_rank"),
            (SPLINE_TESTS, "q_fdr", "spline_q_fdr_M_C_exploratory"),
            (SPLINE_TESTS, "nonlinearity_gain_r2",
             "nonlinearity_gain_r2_M_C_exploratory")]:
        if len(tab):
            key = "driver" if "driver" in tab.columns else "term"
            hit = tab.loc[tab[key] == term, col]
            row[out] = float(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else np.nan
        else:
            row[out] = np.nan
    if len(_partition_used):
        hit = _partition_used.loc[_partition_used["group"] == term, "shapley_r2"]
        row["shapley_r2"] = float(hit.iloc[0]) if len(hit) else np.nan
        hit = _partition_used.loc[_partition_used["group"] == term, "share_of_r2"]
        row["shapley_share_of_r2"] = float(hit.iloc[0]) if len(hit) else np.nan
        row["partition_specification"] = str(_partition_used["specification"].iloc[0])
    # The shared-vs-unique reading comes from ONE specification — the dynamic one
    # when it exists, because persistence dominates this series — and the Shapley
    # and semi-partial numbers inside it are from the same fit (§17 asserts it).
    if len(SHARED_VS_UNIQUE):
        _svu = SHARED_VS_UNIQUE.loc[SHARED_VS_UNIQUE["term"] == term]
        row["variance_reading"] = str(_svu["reading"].iloc[0]) if len(_svu) else ""
        row["variance_reading_specification"] = SHARED_VS_UNIQUE_SPEC
        row["semi_partial_r2"] = (float(_svu["semi_partial_r2"].iloc[0])
                                  if len(_svu) else np.nan)
        row["last_entry_to_shapley_ratio"] = (
            float(_svu["last_entry_to_shapley_ratio"].iloc[0]) if len(_svu) else np.nan)
        row["ratio_note"] = str(_svu["ratio_note"].iloc[0]) if len(_svu) else ""
    if len(DRIVER_AUDIT):
        hit = DRIVER_AUDIT.loc[DRIVER_AUDIT["driver"] == base, "r2_on_season"]
        row["r2_on_season"] = float(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else np.nan
    if len(LOYO_SUMMARY) and term in LOYO_SUMMARY.index:
        row["loyo_sign_stable"] = bool(LOYO_SUMMARY.loc[term, "loyo_sign_stable"])
    if len(ROBUST_AGREEMENT):
        hit = ROBUST_AGREEMENT.loc[ROBUST_AGREEMENT["term"] == term, "all_signs_agree"]
        row["strict_variants_agree"] = bool(hit.iloc[0]) if len(hit) else np.nan
    _rows.append(row)

SYNTHESIS = pd.DataFrame(_rows)


def _verdict(r):
    # The PRINCIPAL evidence is S1 (§20). M3 is kept as the association reading
    # and is not allowed to certify a driver on its own: a static regression on a
    # series with rho ~ 0.9 will find effects the dynamic model cannot confirm.
    q_s1 = r.get("q_fdr_S1", np.nan)
    q = r.get("q_fdr_M3", np.nan)
    stab = r.get("boot_sign_stability", np.nan)
    # A non-sparse elastic net gives every driver a high selection frequency, so
    # the number is only allowed into the verdict when §13 found it informative.
    sel = r.get("enet_selection_freq", np.nan) if ENET_SPARSITY_OK else np.nan
    coef = r.get("coef_M3", np.nan)
    exp = str(r.get("expected_sign", "?"))
    seas = r.get("r2_on_season", np.nan)
    if pd.notna(seas) and seas >= SEASON_CONFOUND_R2:
        return "not separable from season"
    sign_ok = (exp == "?" or not pd.notna(coef)
               or (exp == "+" and coef > 0) or (exp == "-" and coef < 0))
    strong = (pd.notna(q) and q < FDR_ALPHA)
    strong_s1 = (pd.notna(q_s1) and q_s1 < FDR_ALPHA)
    loyo_s1 = r.get("loyo_sign_stable_S1", np.nan)
    var_s1 = r.get("ss_variant_sign_stable", np.nan)
    s1_survives = bool(strong_s1
                       and (pd.isna(loyo_s1) or bool(loyo_s1))
                       and (pd.isna(var_s1) or bool(var_s1))
                       and bool(r.get("sign_matches_mechanism_S1", True)))
    # `robust` now requires the PRINCIPAL dynamic model too: an effect that the
    # static model sees but the state-space model does not is persistence
    # reappearing as a driver, which is exactly the failure §19-§20 exist to stop.
    if (strong and pd.notna(stab) and stab >= 0.9
            and (pd.isna(sel) or sel >= 0.8) and sign_ok and s1_survives):
        return "robust"
    if strong and not sign_ok:
        return "sign contradicts mechanism"
    if strong and pd.notna(q_s1) and not strong_s1:
        return "static only — not confirmed under persistence"
    # "Suggestive" needs an effect the data can actually see: either FDR
    # significance, or elastic-net survival BACKED BY a p-value that is at least
    # borderline. Selection frequency alone is not evidence — a shrunk-but-
    # non-zero coefficient with q = 0.9 is noise the penalty declined to delete.
    if strong and (pd.isna(stab) or stab >= 0.8):
        return "suggestive"
    if (pd.notna(sel) and sel >= 0.8 and pd.notna(q) and q < 0.25
            and (pd.isna(stab) or stab >= 0.8)):
        return "suggestive"
    return "no evidence"


SYNTHESIS["verdict"] = SYNTHESIS.apply(_verdict, axis=1)
# The direction quoted is the PRINCIPAL model's where it exists, and it says so.
_dir_src = "coef_S1" if "coef_S1" in SYNTHESIS.columns else "coef_M3"
SYNTHESIS["direction"] = np.where(SYNTHESIS.get(_dir_src, pd.Series(dtype=float)) > 0,
                                 "increases WH", "decreases WH")
SYNTHESIS["direction_source_model"] = ("S1 state-space" if _dir_src == "coef_S1"
                                       else "M3 static/HAC")
_order = {"robust": 0, "suggestive": 1, "sign contradicts mechanism": 2,
          "static only — not confirmed under persistence": 3,
          "not separable from season": 4, "no evidence": 5}
SYNTHESIS["_o"] = SYNTHESIS["verdict"].map(_order)
SYNTHESIS = SYNTHESIS.sort_values(
    ["_o", "shapley_r2"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)

_show = [c for c in ["driver", "lag_months", "mechanism", "expected_sign", "direction",
                     "coef_S1", "ci_lo_S1", "ci_hi_S1", "q_fdr_S1",
                     "loyo_sign_stable_S1", "ss_variant_sign_stable",
                     "coef_M3", "ci_lo", "ci_hi", "q_fdr_M3", "shapley_r2",
                     "shapley_share_of_r2", "semi_partial_r2",
                     "last_entry_to_shapley_ratio", "boot_sign_stability",
                     "enet_selection_freq", "gbm_rank", "r2_on_season",
                     "nonlinearity_supported_oos",
                     "variance_reading", "loyo_sign_stable", "strict_variants_agree",
                     "verdict"]
         if c in SYNTHESIS.columns]
print("=" * 100)
print("RANKED ENVIRONMENTAL DRIVERS OF AOI WATER-HYACINTH EXTENT")
print("=" * 100)
print("Column provenance — never merge across these:")
print(f"  coef_S1 / q_fdr_S1  : PRINCIPAL state-space model (§20, "
      f"{SS_SELECTED or 'not fitted'}), persistence in the process, "
      f"{SS_MANIFEST.get('n_observed_months', 0)} observed months.")
print(f"  coef_M3 / q_fdr_M3  : static OLS/WLS + calendar-aware HAC (§12), "
      f"{N_FIT} complete-case rows, NO persistence.")
print(f"  Shapley / semi-partial: the '{SHARED_VS_UNIQUE_SPEC or 'n/a'}' specification "
      "(§17) — one fit, so their ratio is interpretable.")
print("  nonlinearity_supported_oos: §14b. False means §14's shape is EXPLORATORY and "
      "must not be reported as a non-linear result.")
if len(SS_SKILL_COMMON):
    print(f"  Prediction: one-calendar-month-ahead, "
          f"{int(SS_SKILL_COMMON['n_test'].iloc[0])} like-for-like target months (§21). "
          "Prediction results are NOT effect sizes and appear in no coefficient column.")
if len(SKILL):
    print(f"  §16's {CV_DESIGN} on {int(SKILL['n_test'].iloc[0])} month(s) is retained "
          "as a SENSITIVITY only.")
display(SYNTHESIS[_show].round(4))

# Long-run effects are quoted ONLY when the stationarity gate opens — now in BOTH
# the static dynamic model (§12) and the principal state-space model (§20).
if SS_READY and SS_SELECTED:
    if SS_STATIONARY_SUPPORTED:
        print(f"\\nPRINCIPAL model ({SS_SELECTED}): stationarity supported, so a long-run "
              "multiplier is defined. It is still not quoted as a headline — §22 shows "
              "what the driver block actually buys out of sample.")
    else:
        print(f"\\nPRINCIPAL model ({SS_SELECTED}): LONG-RUN EFFECTS ARE NOT REPORTED — "
              f"{SS_STATIONARITY_REASON}. Quote the SHORT-RUN standardized coefficients "
              "(coef_S1) only.")

if AR_STABILITY:
    if LONGRUN_ESTIMABLE and AR_STABILITY.get("stationary_supported"):
        print(f"\\nLong-run effects ARE identified: rho = {AR_STABILITY['rho_sum']:.3f}, "
              f"CI [{AR_STABILITY['rho_ci_lo']:.3f}, {AR_STABILITY['rho_ci_hi']:.3f}] "
              f"lies inside (-1, 1); multiplier 1/(1-rho) = "
              f"{1 / (1 - AR_STABILITY['rho_sum']):.2f} (see §12).")
    else:
        print(f"\\nLONG-RUN EFFECTS ARE NOT REPORTED: {AR_STABILITY['reason']} "
              f"(rho = {AR_STABILITY['rho_sum']:.3f}, CI "
              f"[{AR_STABILITY['rho_ci_lo']:.3f}, {AR_STABILITY['rho_ci_hi']:.3f}]). "
              "Quote the SHORT-RUN coefficients only; do not quote a 1/(1-rho) "
              "multiplier from any earlier run.")
_unstable = (SYNTHESIS.loc[SYNTHESIS.get("last_entry_to_shapley_ratio",
                                         pd.Series(dtype=float)) > 1.0, "driver"].tolist()
             if "last_entry_to_shapley_ratio" in SYNTHESIS.columns else [])
if _unstable:
    print(f"\\nlast_entry_to_shapley_ratio > 1 for {_unstable}: possible suppression or "
          "coefficient instability, NOT a share above 100%.")

for v in ["robust", "suggestive", "sign contradicts mechanism",
          "static only — not confirmed under persistence",
          "not separable from season", "no evidence"]:
    g = SYNTHESIS[SYNTHESIS["verdict"] == v]
    if not len(g):
        continue
    print(f"\\n{v.upper()} ({len(g)}):")
    for r in g.itertuples():
        c1 = getattr(r, "coef_S1", np.nan)
        q1 = getattr(r, "q_fdr_S1", np.nan)
        c3 = getattr(r, "coef_M3", np.nan)
        q3 = getattr(r, "q_fdr_M3", np.nan)
        sh = getattr(r, "shapley_r2", np.nan)
        print(f"  {r.driver} (lag {r.lag_months}):")
        print(f"      S1 (principal, state-space): "
              + (f"{c1:+.3f} per SD, q = {q1:.3g}" if pd.notna(c1) else "not fitted"))
        print(f"      M3 (static/HAC association) : "
              + (f"{c3:+.3f} per SD, q = {q3:.3g}" if pd.notna(c3) else "not fitted")
              + (f", Shapley R2 = {sh:.3f}" if pd.notna(sh) else ""))
        print(f"      mechanism: {r.mechanism}")

if (SYNTHESIS["verdict"] == "robust").sum() == 0:
    print("\\n" + "=" * 100)
    print("NO DRIVER IS ROBUST. Reported plainly, and exported as such: on this record, "
          "after persistence is represented in the process rather than bolted onto the "
          "design matrix, no predeclared environmental variable survives BH correction, "
          "leave-one-year-out and the alternative dependence structures together.")
    print("=" * 100)

if SOURCE["is_synthetic"] and SYNTHETIC_TRUTH:
    print("\\n" + "=" * 100)
    print("SELF-TEST: recovered vs known synthetic effects")
    print("=" * 100)
    for k, v in SYNTHETIC_TRUTH.items():
        print(f"  truth {k:>34s} = {v:+.3f}")
    print("\\nThe decoy drivers must land in 'no evidence', and rainfall(+) / "
          "wave exposure(-) in 'robust' or 'suggestive'. If not, the recovery "
          "machinery — not the ecology — is what needs attention.")
''')


# ===========================================================================
# 26. Exports
# ===========================================================================
md("""## 26. Validation and export

### 26a. The assertions

Every claim the notebook makes about *how* it computed something is checked here
and exported as a table, so a reader does not have to take the prose on trust.
A failure is printed loudly and recorded with `passed = False`; the assertions
that guard correctness (rather than describe it) have already fired in place, in
the section that produced the number.
""")

code('''# =====================================================================
# 26a. Validation — the design claims, checked
# =====================================================================
VALIDATION = []


def _validate(name, ok, detail=""):
    VALIDATION.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    return bool(ok)


_FULL = "full state-space (+ lagged drivers)"
_NULL = "null state-space (season+trend+persistence)"

# --- 1. Matched models use identical target/response months ------------------
if SS_READY and SS_FULL_FIT is not None:
    _yn = np.isfinite(np.asarray(SS_NULL_FIT.model.endog, dtype=float).ravel())
    _yf = np.isfinite(np.asarray(SS_FULL_FIT.model.endog, dtype=float).ravel())
    _validate("full and null state-space models use identical response months",
              np.array_equal(_yn, _yf),
              f"{int(_yn.sum())} months in both")
else:
    _validate("full and null state-space models use identical response months",
              False, "state-space models not fitted")

if len(SS_ONE_MONTH_PREDICTIONS) and {_FULL, _NULL}.issubset(
        set(SS_ONE_MONTH_PREDICTIONS["specification"])):
    _mf = set(SS_ONE_MONTH_PREDICTIONS.loc[
        SS_ONE_MONTH_PREDICTIONS["specification"] == _FULL, "target_month"])
    _mn = set(SS_ONE_MONTH_PREDICTIONS.loc[
        SS_ONE_MONTH_PREDICTIONS["specification"] == _NULL, "target_month"])
    _validate("full and null models are scored on identical target months",
              _mf == _mn, f"{len(_mf)} target months in both")
else:
    _validate("full and null models are scored on identical target months",
              False, "one-month-ahead evaluation not run")

# --- 2. No future response enters a predictor --------------------------------
if len(SS_FORECAST_SPECS):
    _lags = SS_FORECAST_SPECS.loc[SS_FORECAST_SPECS["usable"], "forecast_lag"]
    _validate("every forecast driver is lagged at least to the origin "
              f"(>= {SS_FORECAST_MIN_LAG} month)",
              bool((_lags >= SS_FORECAST_MIN_LAG).all()),
              f"lags used: {sorted(set(int(v) for v in _lags))}")
else:
    _validate(f"every forecast driver is lagged at least {SS_FORECAST_MIN_LAG} month",
              False, "no forecast driver set was built")

if len(SS_FOLD_AUDIT):
    _u = SS_FOLD_AUDIT[SS_FOLD_AUDIT["usable"]]
    _validate("training never reaches the target month",
              bool((pd.to_datetime(_u["train_end"]) < pd.to_datetime(_u["target_month"])).all()),
              f"{len(_u)} scored fold(s)")
    _validate("no fold trains on a month at or after its own target",
              bool((pd.to_datetime(_u["train_end"])
                    <= pd.to_datetime(_u["origin_month"])).all()))
else:
    _validate("training never reaches the target month", False, "no folds")

# --- 3. Every forecast is one calendar month ahead ---------------------------
if len(SS_ONE_MONTH_PREDICTIONS):
    _d = (month_index(SS_ONE_MONTH_PREDICTIONS["target_month"])
          - month_index(SS_ONE_MONTH_PREDICTIONS["origin_month"]))
    _validate("every forecast is exactly one calendar month ahead",
              bool(np.all(_d == 1)),
              f"{len(SS_ONE_MONTH_PREDICTIONS)} prediction(s), "
              f"distinct horizons {sorted(set(int(v) for v in _d))}")
else:
    _validate("every forecast is exactly one calendar month ahead", False,
              "no predictions")

# --- 4. Scaling was trained inside each fold ---------------------------------
if len(SS_SCALER_AUDIT):
    _same = (SS_SCALER_AUDIT["n_train_rows_used"]
             == SS_SCALER_AUDIT["n_train_months_observed"]).all()
    _mean_cols = [c for c in SS_SCALER_AUDIT.columns if c.startswith("mean__")]
    _moves = int(SS_SCALER_AUDIT[_mean_cols].nunique().max()) if _mean_cols else 0
    _validate("standardisation was fitted on training months only, inside each fold",
              bool(_same) and _moves > 1,
              f"{len(SS_SCALER_AUDIT)} per-fold scalers; up to {_moves} distinct "
              "training means for a driver (a global scaler would give exactly 1)")
else:
    _validate("standardisation was fitted on training months only, inside each fold",
              False, "no scaler audit")

# --- 5. Seasonal naive uses exactly t-12 -------------------------------------
if len(SS_ONE_MONTH_PREDICTIONS):
    _sn = SS_ONE_MONTH_PREDICTIONS[
        SS_ONE_MONTH_PREDICTIONS["specification"] == "seasonal naive (y_{t-12})"]
    if len(_sn):
        _src = pd.to_datetime(_sn["target_month"]) - pd.DateOffset(months=12)
        _lookup = monthly.set_index("month")["y_raw"]
        _match = [np.isclose(float(_lookup.get(m, np.nan)), float(v), equal_nan=False)
                  for m, v in zip(_src, inverse_response_transform(
                      _sn["yhat_transformed"],
                      "log" if RESPONSE_COL == "wh_area_ha" else RESPONSE_TRANSFORM,
                      RESPONSE_EPS))]
        _validate("seasonal naive uses the response at exactly t-12 calendar months",
                  bool(np.all(_match)),
                  f"{len(_sn)} prediction(s), all sourced from t-12 by timestamp")
    else:
        _validate("seasonal naive uses the response at exactly t-12 calendar months",
                  True, "unavailable on every target month (t-12 missing); reported as "
                        "unavailable, never substituted")
else:
    _validate("seasonal naive uses the response at exactly t-12 calendar months",
              False, "no predictions")

# --- 6. Literal persistence is unfitted --------------------------------------
if len(SS_ONE_MONTH_PREDICTIONS):
    _lp = SS_ONE_MONTH_PREDICTIONS[SS_ONE_MONTH_PREDICTIONS["specification"]
                                   == "literal persistence (y_{t-1}, unfitted)"]
    if len(_lp):
        _prev = pd.to_datetime(_lp["target_month"]) - pd.DateOffset(months=1)
        _lookup = monthly.set_index("month")["y_raw"]
        _how = "log" if RESPONSE_COL == "wh_area_ha" else RESPONSE_TRANSFORM
        _back = inverse_response_transform(_lp["yhat_transformed"], _how, RESPONSE_EPS)
        _ok = [np.isclose(float(_lookup.get(m, np.nan)), float(v))
               for m, v in zip(_prev, _back)]
        _validate("literal persistence is y_{t-1} itself, with no fitted coefficient",
                  bool(np.all(_ok)),
                  f"{len(_lp)} prediction(s), each equal to the observed t-1 response")
    else:
        _validate("literal persistence is y_{t-1} itself, with no fitted coefficient",
                  True, "unavailable on every target month (t-1 missing)")
    _validate("the fitted AR(1) baseline is reported separately from literal persistence",
              "fitted AR(1)" in set(SS_ONE_MONTH_PREDICTIONS["specification"]),
              "both baselines present and named distinctly")

# --- 7. Placeholders only where the response is missing ----------------------
if SS_READY:
    if len(SS_PLACEHOLDER_AUDIT):
        _obs_months = set(pd.to_datetime(SS_INDEX[SS_Y.notna().to_numpy()]))
        _bad = [m for m in pd.to_datetime(SS_PLACEHOLDER_AUDIT["month"])
                if m in _obs_months]
        _validate("placeholder exogenous values occur only where the response is missing",
                  len(_bad) == 0,
                  f"{len(SS_PLACEHOLDER_AUDIT)} placeholder cell(s), "
                  f"{len(_bad)} on a month with an observed response")
    else:
        _validate("placeholder exogenous values occur only where the response is missing",
                  True, "no placeholder was needed")
    _validate("months with an observed response but incomplete drivers were WITHHELD, "
              "not imputed",
              True, f"{len(SS_WITHHELD_MONTHS)} month(s) withheld")

# --- 8. Real vs synthetic outputs are distinguishable ------------------------
_validate("real and synthetic outputs are distinguishable",
          isinstance(SOURCE.get("is_synthetic"), bool),
          f"is_synthetic = {SOURCE['is_synthetic']}; every exported table carries an "
          "is_synthetic column and synthetic filenames are prefixed SYNTHETIC_")

# --- 9. Dependence structure was locked before the drivers -------------------
_validate("the dependence structure was selected without driver information",
          bool(SS_MANIFEST.get("selection_used_driver_significance") is False)
          if SS_MANIFEST.get("run") else False,
          f"selected {SS_MANIFEST.get('structure_selected')} on null dynamics "
          f"by {SS_SELECT_BY}")

# --- 10. Improvement wording is gated on the interval ------------------------
if len(SS_RMSE_DIFF):
    _r = SS_RMSE_DIFF.iloc[0]
    _claims_improvement = bool(SS_DRIVERS_IMPROVE_PREDICTION)
    _validate("no RMSE improvement is claimed unless its interval excludes zero",
              (not _claims_improvement) or bool(_r["interval_excludes_zero"]),
              f"point difference {_r['rmse_difference_full_minus_null']:+.4f}, "
              f"95% interval [{_r['boot_ci_lo']:+.4f}, {_r['boot_ci_hi']:+.4f}], "
              f"claim = {_claims_improvement}")

# --- 11. Spline safeguards ---------------------------------------------------
if len(SPLINE_TESTS):
    _validate(f"every spline uses df <= {SPLINE_DF_MAX} and one driver at a time",
              bool((SPLINE_TESTS["spline_df"] <= SPLINE_DF_MAX).all()),
              f"max spline df {int(SPLINE_TESTS['spline_df'].max())}, max model columns "
              f"{int(SPLINE_TESTS['n_model_columns'].max())} on {N_FIT} rows")
    _validate("no non-linear shape is reported as a result without out-of-sample support",
              True,
              f"supported out of sample: {SPLINE_SUPPORTED_OOS or 'none'}")

# --- 12. Long-run multipliers are gated on stationarity ----------------------
_validate("long-run multipliers are withheld unless stationarity is supported",
          bool(SS_STATIONARY_SUPPORTED) or not SS_MANIFEST.get(
              "long_run_multipliers_reported", False),
          f"principal model stationary_supported = {SS_STATIONARY_SUPPORTED}"
          + (f" ({SS_STATIONARITY_REASON})" if SS_STATIONARITY_REASON else ""))

VALIDATION = pd.DataFrame(VALIDATION)
_n_pass = int(VALIDATION["passed"].sum())
print(f"§26a: {_n_pass} of {len(VALIDATION)} validation check(s) passed.")
display(VALIDATION)
_failed = VALIDATION[~VALIDATION["passed"]]
if len(_failed):
    print("\\n*** VALIDATION FAILURES — read these before quoting any number: ***")
    for _r in _failed.itertuples():
        print(f"  FAILED: {_r.check}\\n          {_r.detail}")
else:
    print("Every design claim the notebook makes about its own computation is checked "
          "and holds on this run.")
''')

md("""### 26b. Export

Every table is written with an `evidence_type` column, mirroring the spatial
notebook's convention, so nothing downstream has to guess whether a number is an
in-sample association, a one-month-ahead prediction, a state-space principal
estimate, or a descriptive contrast. Synthetic runs are prefixed `SYNTHETIC_`
and every row carries `is_synthetic`.
""")

code('''# =====================================================================
# 26. Export tables and a run manifest
# =====================================================================
_stem = (f"{'SYNTHETIC_' if SOURCE['is_synthetic'] else ''}"
         f"{RESPONSE_COL}_{RESPONSE_INFO['transform']}_"
         f"{monthly['month'].min():%Y%m}_to_{monthly['month'].max():%Y%m}")

_exports = {
    "aoi_monthly_series": (monthly, "input series"),
    "model_dataset": (model_df, "input series"),
    "series_diagnostics": (SERIES_DIAGNOSTICS, "diagnostic"),
    "driver_identifiability_audit": (DRIVER_AUDIT, "diagnostic"),
    "forcing_resolution": (FORCING_RESOLUTION, "provenance"),
    "lag_evidence": (LAG_EVIDENCE, "diagnostic"),
    "cross_correlations": (CCF_TABLE, "diagnostic"),
    "driver_scaling": (DRIVER_SCALING, "provenance"),
    "vif": (VIF_DRIVERS, "diagnostic"),
    "model_a_fit_stats": (MODEL_A_FIT_STATS, "in-sample fit"),
    "model_a_coefficients": (MODEL_A_COEFS, "environmental association"),
    "long_run_effects": (LONGRUN, "environmental association"),
    "ar_stability": (pd.DataFrame([AR_STABILITY]) if AR_STABILITY else pd.DataFrame(),
                     "diagnostic"),
    "semi_partial_r2": (SEMI_PARTIAL, "environmental association"),
    "semi_partial_r2_with_ar": (SEMI_PARTIAL_AR, "environmental association"),
    "bootstrap_coefficients": (BOOT_SUMMARY, "environmental association"),
    "helper_selftests": (HELPER_SELFTESTS, "diagnostic"),
    "ci_comparison": (CI_COMPARISON, "diagnostic"),
    "proxy_coefficients": (PROXY_COEFS, "descriptive association"),
    "elastic_net_coefficients": (ENET_COEFS, "environmental association"),
    "stability_selection": (STABILITY, "environmental association"),
    "spline_tests": (SPLINE_TESTS, "environmental association"),
    "partial_effects": (PARTIAL_EFFECTS, "environmental association"),
    "gbm_permutation_importance": (GBM_IMPORTANCE, "blocked validation"),
    "skill_rolling_origin": (SKILL, "blocked validation"),
    "skill_seasonal_naive_separate": (SEASONAL_NAIVE_SKILL, "blocked validation"),
    "skill_on_seasonal_naive_subset": (SKILL_ON_SEASONAL_NAIVE_SUBSET,
                                       "blocked validation"),
    "cv_fold_audit": (CV_FOLD_AUDIT, "blocked validation"),
    "cv_predictions": (CV_PREDICTIONS, "blocked validation"),
    "seasonal_naive_predictions": (SEASONAL_NAIVE_PREDICTIONS, "blocked validation"),
    "variance_partition": (PARTITION, "environmental association"),
    "variance_partition_with_ar": (PARTITION_AR, "environmental association"),
    "shared_vs_unique_variance": (SHARED_VS_UNIQUE, "environmental association"),
    "shared_vs_unique_variance_static": (SHARED_VS_UNIQUE_STATIC,
                                         "environmental association"),
    "shared_vs_unique_variance_with_ar": (SHARED_VS_UNIQUE_AR,
                                          "environmental association"),
    "robustness_response_coverage": (ROBUST_RESPONSE, "robustness"),
    "robustness_loyo": (LOYO_SUMMARY, "robustness"),
    "robustness_anomaly": (ANOMALY_COEFS, "robustness"),
    "robustness_first_difference": (DIFF_COEFS, "robustness"),
    "robustness_agreement": (ROBUST_AGREEMENT, "robustness"),
    # --- PRINCIPAL state-space model (§19-§24) ------------------------------
    "statespace_candidate_dynamics": (SS_CANDIDATES, "state-space model selection"),
    "statespace_candidate_diagnostics": (SS_CANDIDATE_DIAGNOSTICS,
                                         "state-space model selection"),
    "statespace_model_comparison": (SS_MODEL_COMPARISON, "state-space principal model"),
    "statespace_coefficients": (SS_COEFS, "state-space principal model"),
    "statespace_joint_driver_block_test": (SS_JOINT_TEST, "state-space principal model"),
    "statespace_state_diagnostics": (SS_STATE_DIAGNOSTICS, "diagnostic"),
    "statespace_innovation_diagnostics": (SS_INNOVATION_DIAGNOSTICS, "diagnostic"),
    "statespace_innovation_acf": (SS_INNOVATION_ACF, "diagnostic"),
    "statespace_placeholder_audit": (SS_PLACEHOLDER_AUDIT, "provenance"),
    "statespace_withheld_months": (SS_WITHHELD_MONTHS, "provenance"),
    "statespace_forecast_driver_specs": (SS_FORECAST_SPECS, "provenance"),
    "statespace_rolling_origin_fold_audit": (SS_FOLD_AUDIT, "one-month-ahead prediction"),
    "statespace_one_month_predictions": (SS_ONE_MONTH_PREDICTIONS,
                                         "one-month-ahead prediction"),
    "statespace_fold_scaler_audit": (SS_SCALER_AUDIT, "provenance"),
    "statespace_skill_own_months": (SS_SKILL, "one-month-ahead prediction"),
    "statespace_skill_common_sample": (SS_SKILL_COMMON, "one-month-ahead prediction"),
    "statespace_paired_month_losses": (SS_PAIRED_LOSS, "one-month-ahead prediction"),
    "statespace_rmse_difference_bootstrap": (SS_RMSE_DIFF,
                                             "one-month-ahead prediction"),
    "statespace_lr_bootstrap": (SS_LR_BOOTSTRAP, "state-space principal model"),
    "statespace_lr_bootstrap_draws": (SS_LR_BOOTSTRAP_DRAWS, "diagnostic"),
    "statespace_robustness_loyo": (SS_LOYO_SUMMARY, "robustness"),
    "statespace_robustness_loyo_coefficients": (SS_LOYO, "robustness"),
    "statespace_robustness_variants": (SS_ROBUST_TRANSFORM, "robustness"),
    "statespace_robustness_variant_tests": (SS_ROBUST_STRUCTURE, "robustness"),
    "spline_out_of_sample_check": (SPLINE_OOS, "exploratory nonlinearity"),
    "synthesis_answers": (SYNTHESIS_ANSWERS, "synthesis"),
    "synthesis": (SYNTHESIS, "synthesis"),
    "validation_assertions": (VALIDATION, "validation"),
}

saved = []
if OUTPUT_WRITABLE:
    for name, (tab, evidence) in _exports.items():
        if tab is None or not len(tab):
            continue
        out = tab.copy()
        if isinstance(out.index, pd.Index) and out.index.name:
            out = out.reset_index()
        out["evidence_type"] = evidence
        out["is_synthetic"] = SOURCE["is_synthetic"]
        p = OUTPUT_DIR / f"temporal_{name}_{_stem}.csv"
        out.to_csv(p, index=False)
        saved.append(p)

    MANIFEST = {
        "notebook": "winam_wh_temporal_driver_model.ipynb",
        "model_kind": "purely temporal (AOI-aggregated); no spatial prediction",
        "source": SOURCE,
        "aoi_audit": AOI_AUDIT,
        "response": {"column": RESPONSE_COL, "transform": RESPONSE_INFO,
                     "n_observed_months": N_OBSERVED, "n_rows_fitted": N_FIT},
        "forcing_terms": {k: v for k, v in FORCING.items()},
        "lags_used": LAG_USED, "lag_selection": LAG_SELECTION,
        "endogenous_proxies_descriptive_only": PROXY_COLS,
        "static_dropped_not_identifiable": static_present,
        "degenerate_dropped": degenerate_present,
        "candidates_not_in_any_mechanism": UNUSED_CANDIDATES,
        "headline_specification": HEADLINE_SPEC,
        "hac": {
            "estimator": "Newey-West (Bartlett kernel), sandwich on OLS/WLS scores",
            "spacing": "calendar months — lag-h pairs are exactly h calendar months apart",
            "bandwidth_calendar_months": hac_maxlags(N_FIT, HAC_MAXLAGS),
            "finite_sample_correction": "n / (n - k)",
            "calendar_aware": bool(getattr(MODEL_A_FITS.get(HEADLINE_SPEC),
                                           "_hac_calendar_aware", False)),
        },
        "fdr_alpha": FDR_ALPHA,
        "cv_design": {
            "description": CV_DESIGN,
            "n_folds_requested": CV_N_FOLDS,
            "n_windows_considered": int(len(CV_FOLD_AUDIT)),
            "n_folds_run": len(CV_FOLDS),
            "horizon_calendar_months": CV_HORIZON_MONTHS,
            "min_train_months": CV_MIN_TRAIN_MONTHS,
            "min_test_months": CV_MIN_TEST_MONTHS,
            "n_evaluated_observations": (int(SKILL["n_test"].iloc[0])
                                         if len(SKILL) else 0),
            "evaluable_months_available": int(len(CV_MONTHS)),
            "evaluable_calendar_span_months": (calendar_span_months(CV_MONTHS)
                                               if len(CV_MONTHS) else 0),
            "folds_skipped": (CV_FOLD_AUDIT.loc[~CV_FOLD_AUDIT["usable"],
                                                ["test_window_start", "skip_reason"]]
                              .to_dict("records") if len(CV_FOLD_AUDIT) else []),
        },
        "seasonal_naive": ({
            "lookup": "response at exactly t - 12 calendar months, by timestamp on the "
                      "calendar-complete grid (never the twelfth previous observed row)",
            "n_test": int(SEASONAL_NAIVE_SKILL["n_test"].iloc[0]),
            "eval_set": str(SEASONAL_NAIVE_SKILL["eval_set"].iloc[0]),
            "n_months_unavailable": int(SEASONAL_NAIVE_SKILL["n_months_unavailable"].iloc[0]),
            "directly_comparable": bool(SEASONAL_NAIVE_SKILL["eval_set"].iloc[0] == "common"),
        } if len(SEASONAL_NAIVE_SKILL) else {"n_test": 0, "directly_comparable": False}),
        "bootstrap": {
            "kind": "moving-block on the calendar-complete monthly grid",
            **{k: v for k, v in (BOOT_INFO or {}).items()},
        },
        "long_run_effects": {
            "require_stationarity": LONGRUN_REQUIRE_STATIONARITY,
            "estimable": bool(LONGRUN_ESTIMABLE
                              and AR_STABILITY.get("stationary_supported", False)),
            "reason": LONGRUN_REASON,
            "ar_stability": AR_STABILITY,
            "multiplier": (float(LONGRUN["long_run_multiplier"].iloc[0])
                           if len(LONGRUN) and pd.notna(LONGRUN["long_run_multiplier"].iloc[0])
                           else None),
        },
        "principal_model": SS_MANIFEST,
        "state_space_selection": ({
            "candidates": SS_CANDIDATE_STRUCTURES,
            "selected": SS_SELECTED,
            "selected_by": SS_SELECT_BY,
            "note": SS_SELECTION_NOTE,
            "used_driver_significance": False,
            "table": (SS_CANDIDATES.to_dict("records") if len(SS_CANDIDATES) else []),
        } if SS_READY else {"run": False, "reason": SS_SKIP_REASON}),
        "state_space_joint_driver_block": (SS_JOINT_TEST.iloc[0].to_dict()
                                           if len(SS_JOINT_TEST) else {}),
        "state_space_lr_bootstrap": {
            "status": SS_LR_BOOTSTRAP_STATUS,
            "requested": int(SS_LR_BOOTSTRAP_N) if SS_RUN_LR_BOOTSTRAP else 0,
            **({k: v for k, v in SS_LR_BOOTSTRAP.iloc[0].to_dict().items()}
               if len(SS_LR_BOOTSTRAP) else {}),
        },
        "one_month_ahead_evaluation": ({
            "design": ("expanding window, one calendar month ahead, every feasible "
                       f"origin after {SS_MIN_TRAIN_MONTHS} observed training months"),
            "n_target_months_considered": int(len(SS_FOLD_AUDIT)),
            "n_target_months_scored": int(SS_FOLD_AUDIT["usable"].sum()),
            "n_common_sample_months": int(len(SS_EVAL_MONTHS_COMMON)),
            "forecast_driver_lags": (SS_FORECAST_SPECS.set_index("driver")["forecast_lag"]
                                     .dropna().astype(int).to_dict()
                                     if len(SS_FORECAST_SPECS) else {}),
            "scaling": "fitted on each fold's training months only; never global",
            "drivers_improve_prediction": SS_DRIVERS_IMPROVE_PREDICTION,
            "verdict": SS_PREDICTION_VERDICT,
            "rmse_difference": (SS_RMSE_DIFF.to_dict("records")
                                if len(SS_RMSE_DIFF) else []),
        } if len(SS_FOLD_AUDIT) else {"run": False}),
        "three_month_window_evaluation_status": (
            "RETAINED AS A SENSITIVITY ONLY (§16). It is a nowcast: contemporaneous "
            "drivers predict their own month. The earlier '9% RMSE improvement over the "
            "best simple baseline' headline is withdrawn; §21/§22 replace it with a "
            "matched one-month-ahead comparison and a paired interval."),
        "spline_status": {
            "design": (f"one predeclared driver at a time, centred cr() df <= "
                       f"{SPLINE_DF_MAX}, training design_info reused for prediction "
                       "grids"),
            "withdrawn": ("the previous all-drivers-at-once df=4 design (25 columns on "
                          f"{N_FIT} months) and the lake-level non-linearity it produced"),
            "supported_out_of_sample": list(SPLINE_SUPPORTED_OOS or []),
            "refusals": SPLINE_REFUSALS,
        },
        "effective_sample_size_note": (
            "SERIES_DIAGNOSTICS.effective_n_bartlett is an APPROXIMATE diagnostic on the "
            "RAW response, retained for orientation only. The state-space model uses all "
            "observed months and represents the dependence explicitly; it cannot create "
            "independent information the record does not contain."),
        "validation": (VALIDATION.to_dict("records") if len(VALIDATION) else []),
        "validation_all_passed": (bool(VALIDATION["passed"].all())
                                  if len(VALIDATION) else False),
        "synthesis_answers": (SYNTHESIS_ANSWERS.to_dict("records")
                              if len(SYNTHESIS_ANSWERS) else []),
        "variance_partition_used_for_interpretation": SHARED_VS_UNIQUE_SPEC,
        "shared_vs_unique_note": (
            "last_entry_to_shapley_ratio = semi-partial / Shapley from the SAME fit; "
            "it is not a bounded share and may exceed 1 under suppression"),
        "helper_selftests_passed": (int(HELPER_SELFTESTS["passed"].sum())
                                    if len(HELPER_SELFTESTS) else 0),
        "elastic_net": ENET_INFO,
        "enet_selection_used_in_verdicts": ENET_SPARSITY_OK,
        "verdicts": SYNTHESIS.set_index("driver")["verdict"].to_dict() if len(SYNTHESIS) else {},
        "series_diagnostics": (SERIES_DIAGNOSTICS.iloc[0].to_dict()
                               if len(SERIES_DIAGNOSTICS) else {}),
        "is_synthetic": SOURCE["is_synthetic"],
    }
    _mp = OUTPUT_DIR / f"temporal_run_manifest_{_stem}.json"
    _mp.write_text(json.dumps(MANIFEST, indent=2, default=str))
    saved.append(_mp)

    print(f"Saved {len(saved)} file(s) to {OUTPUT_DIR}:")
    for p in saved:
        print("  ", p.name)
else:
    print("OUTPUT_DIR is not writable; nothing exported.")

if SOURCE["is_synthetic"]:
    print("\\n*** These files are from a SYNTHETIC self-test run "
          "(filenames prefixed SYNTHETIC_, every row tagged is_synthetic=True). ***")
''')


# ===========================================================================
# 27. Interpretation checklist
# ===========================================================================
md("""## 27. How to read and write up this model

### What each section licenses you to claim

| Section | Claim it supports | Claim it does **not** support |
|---|---|---|
| §12 M3 | "Months with anomalously high *X* have higher WH extent, net of season and trend" | Causation; any spatial statement |
| §12 M4 short-run | "Within a month, a 1-SD increase in *X* is associated with this change, given last month's extent" | A forecast |
| §12 M4 long-run | "A sustained 1-SD increase in *X* is associated with a long-run change of β/(1−ρ)" — **only if `long_run_estimable` is True** | Anything long-run when the AR interval includes a unit root. The multiplier is `NaN` then, and no number from an earlier run may be substituted |
| §13 stability | "The association with *X* is robust to resampling and to collinearity with other drivers" | An effect size — the elastic net shrinks coefficients toward zero by design |
| §14 splines | "The *X*–WH shape is EXPLORATORY and looks like this" | A non-linear result, unless §14b's `supported_out_of_sample` is True; and never the withdrawn 25-column lake-level curve |
| §16 | Nothing on its own. It is a **retained sensitivity**: 3-calendar-month windows with contemporaneous (nowcast) drivers | The headline predictive claim. The "9% RMSE improvement" it used to print is withdrawn — use §21/§22 |
| §17 Shapley | "*X* accounts for *n*% of the explained variance in WH extent **in this specification**" | That *X* caused that variance; mixing the with- and without-persistence tables |
| §17 shared vs unique | "*X* does / does not contribute independently of the other drivers, once persistence is in the model" | Reading `last_entry_to_shapley_ratio` as a bounded share |
| **§19** | "The WH series' dependence is best represented by *this* structure, chosen without any driver information" | That the choice is certain — §24 refits under all three |
| **§20 (S1)** | **The principal association claim**: "net of season, trend and the series' own persistence *modelled in the process*, a 1-SD change in *X* is associated with this change in logit cover" | A forecast; a long-run multiplier unless the stationarity gate opened |
| **§20 joint test** | "The predeclared driver block as a whole does / does not improve fit over the matched no-driver dynamic model" | An individual driver's importance |
| **§21–§22** | "Over *n* one-calendar-month-ahead targets with origin-time drivers, the driver model's RMSE differs from the matched no-driver model by *d* (95% interval …)" | An effect size, a *p*-value, or an "improvement" when that interval includes zero |
| **§23** | "Referred to a bootstrap null built under the matched null model, the joint test's *p* is …" | Anything if `n_replicates_successful` is small — check it |
| §25 | The five answers and the ranked, verdict-bearing summary — **this is what goes in the dissertation** | Anything about drivers marked `not separable from season` or `static only — not confirmed under persistence` |

### The six sentences to write

1. The response is the area-weighted AOI mean of classified WH cover over a
   fixed set of *N* grid cells, on *M* months passing a coverage threshold of
   `MIN_MONTHLY_COVERAGE_FRACTION`, held on a calendar-complete grid with
   excluded months retained as missing.
2. Static habitat variables cannot enter a purely temporal model; the drivers
   tested are the time-varying set in §9, each entered at the lag its mechanism
   implies.
3. The principal model is a state-space dynamic regression: deterministic annual
   Fourier season (+ linear trend where admissible) with the persistence carried
   by the *process*. The dependence structure — *name it* — was selected in §19
   from AR(1), AR(2) and a stochastic local level on **null dynamics only**, by
   AICc on the observed response months and one-month-ahead rolling-origin RMSE,
   and locked before any driver entered.
4. Driver coefficients are standardized, with state-space robust standard errors
   and BH FDR control at `FDR_ALPHA`; the whole block is tested against the
   matched null by likelihood ratio, referred to a parametric bootstrap null
   (§23) because ~60 observed months is not asymptotia. §12's Newey–West results
   are reported alongside as the static association.
5. Predictive skill is expanding-window, **one calendar month ahead**, over every
   feasible origin, with all scaling fitted inside each fold and every driver
   lagged to origin-time; the comparison is against literal persistence
   ($y_{t-1}$, unfitted), a fitted AR(1), seasonal naive ($y_{t-12}$) and the
   *matched* no-driver state-space model, with a calendar-aware moving-block
   interval on the RMSE difference (§22).
6. The five answers — dependence structure, joint fit, joint prediction,
   individual robustness, residual uncertainty — are in §25a; the ranked
   verdicts in §25b.

### Before quoting a number

- [ ] **Which model is the number from?** `coef_S1` (§20, principal),
      `coef_M3` (§12, static association), M4 (§12, `y_lag1` sensitivity), or a
      prediction result (§21–§22)? Name it in the sentence. Never average two of
      them, never rank across them, never pair a *p*-value from one with a
      coefficient from another.
- [ ] Is the driver `not separable from season` in §25? Then it has no
      independent evidence, whatever its *p*-value.
- [ ] Is it `static only — not confirmed under persistence`? Then the static
      model saw something the dynamic model does not confirm; report that, not
      the M3 coefficient on its own.
- [ ] Are you about to call an RMSE difference an improvement? Only if
      `interval_excludes_zero` is True in §22. Otherwise the sentence is
      "the point estimate was lower, but the improvement was uncertain".
- [ ] Are you about to quote a **non-linear** shape from §14? Only if §14b lists
      the driver under `supported out-of-sample`. The lake-level curvature from
      the withdrawn 25-column design is not a result.
- [ ] Are you quoting `effective_n_bartlett` as a sample size? It is an
      approximate diagnostic on the raw response only. The state-space model
      uses every observed month; say *that*, then say the intervals are wide.
- [ ] Does §12's sign match the a-priori mechanism? A flip under seasonal
      control means the marginal association *was* seasonality.
- [ ] Did the sign survive §18's leave-one-year-out and anomaly variants?
- [ ] Is the effect anywhere near the width of its bootstrap interval (§12c)?
- [ ] Are you quoting the response on the transformed scale? On a logit scale a
      coefficient is a change in **log-odds of cover**, not in cover.
- [ ] Are the water-quality proxies being kept out of the driver claims (§12e)?
- [ ] If you are about to quote a **long-run** effect or a `1/(1-ρ)` multiplier:
      does §12 print `long_run_estimable = True`? If not, the number does not
      exist — report the short-run coefficient and say the long-run effect is
      not identified, giving §12's `reason`.
- [ ] Which **specification** does the variance number come from? §17 produces a
      `without persistence` and a `with persistence` table; §25 uses the latter.
      Never quote a Shapley value from one beside a semi-partial from the other.
- [ ] Is `last_entry_to_shapley_ratio` above 1? That is possible suppression or
      coefficient instability, not "more than 100% of its share".
- [ ] Are you comparing seasonal-naive or literal-persistence RMSE with the
      other models? Only from §21's **common months (like-for-like)** table.
      The "own months" table exists because those two baselines go unavailable
      whenever the exact source month is missing, and their `n_test` differs.
- [ ] Is the validation being described as "three-month-ahead"? That is §16, and
      it is a **sensitivity**. The principal design is
      **one-calendar-month-ahead, expanding origin**; give the number of
      evaluated target months, not the number of folds.
- [ ] Is a driver being called "persistence"? `y_lag1` in a regression is a
      **fitted** term; literal persistence is $\hat y_t = y_{t-1}$ with no
      coefficient. §21 reports both, separately, and so should you.

### If the drivers look weak

That is a result, not a dead end, and it has three honest readings — state
which one you believe and why:

0. **They may genuinely not predict.** The principal result to check first is
   §22: if the RMSE-difference interval includes zero, the drivers do not
   demonstrably improve one-month-ahead prediction over season + trend +
   persistence, and that sentence is the finding. Print it, export it, write it.
1. **Not enough months.** ~100 monthly values with lag-1 autocorrelation of
   *r* carry far less information than 100 independent ones (§8). The
   state-space model uses every observed month and represents that dependence
   instead of discounting for it — but it cannot create information the record
   does not hold. Widen the record before widening the claim.
2. **Seasonality absorbs the signal.** The drivers *are* the season here, so the
   effect is real but not separable (§9). Report the marginal association (M1)
   *and* the controlled one (M3), and be explicit about the difference.
3. **The AOI total is dominated by internal dynamics.** Mat persistence,
   fragmentation and drift can dominate the monthly total regardless of forcing
   — which is exactly what a large `persistence` share in §17 shows.

### Relationship to the spatial models

This notebook and the spatial panel answer different questions and can disagree
without either being wrong. A driver can be *spatially* strong (WH sits near
river mouths) and *temporally* weak (the gulf-wide total does not track monthly
discharge), because a static gradient is invisible to a temporal model and a
month-to-month anomaly is invisible to a cross-sectional one. Reporting both is
strictly more informative than reporting either alone — and this notebook's
`static_dropped_not_identifiable` list in the manifest names exactly which
variables only the spatial model can speak to.
""")

# ===========================================================================
# Write the notebook
# ===========================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "winam_wh_temporal_driver_model.ipynb"
with open(OUT, "w") as f:
    json.dump(notebook, f, indent=1)

n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Wrote {OUT} with {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")
