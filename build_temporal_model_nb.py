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

**The honest sample size.** The panel spans at most ~9 years of months, and the
response is strongly autocorrelated, so the *effective* number of independent
observations is a fraction of the row count. Every standard error here is
autocorrelation-robust (Newey–West), every *p*-value is FDR-adjusted, and every
claim of predictive skill comes from rolling-origin cross-validation — never
from in-sample $R^2$. §16 reports whether the drivers beat *persistence plus
season* out of sample; if they do not, that is the finding.

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
| 12 | **Model A** — linear driver model, Newey–West SEs, nested specifications |
| 13 | **Model B** — elastic net + bootstrap stability selection (collinearity-robust ranking) |
| 14 | **Model C** — spline GLM (non-linear driver–response shapes, partial effects) |
| 15 | **Model D** — gradient boosting + out-of-fold permutation importance |
| 16 | Out-of-sample skill vs persistence / seasonal baselines, on **calendar-month** rolling-origin windows |
| 17 | **Variance partitioning** — Shapley $R^2$ across drivers, season, trend, persistence, with shared-vs-unique read **within** each specification |
| 18 | Robustness — response definition, coverage, leave-one-year-out, deseasonalised |
| 19 | **Synthesis table** — the ranked, verdict-bearing driver list |
| 20–21 | Exports; interpretation checklist |
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

warnings.filterwarnings("ignore", category=FutureWarning)
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

# Spline GLM (§14): degrees of freedom per driver smooth. 3-4 is the ceiling
# worth attempting on ~100 monthly observations.
SPLINE_DF = 4

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
    print(f"OUTPUT_DIR not writable ({exc}); §20 will skip the exports.")

print("Configuration loaded.")
print(f"  response          : {RESPONSE_COL} ({RESPONSE_TRANSFORM} scale)")
print(f"  forcing mechanisms: {len(TEMPORAL_FORCING_TERMS)}")
print(f"  lag selection     : {LAG_SELECTION}")
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


def make_synthetic_monthly(n_months=108, seed=42):
    """A synthetic AOI series with KNOWN driver effects, for offline checking.

    Construction (on the logit-cover scale): a strong annual cycle, an upward
    trend, autoregressive persistence, a POSITIVE effect of lagged antecedent
    rainfall, a NEGATIVE effect of wave exposure, and two pure-noise drivers
    that must NOT be recovered.
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
    truth = {"rain_chirps_30d_mm(lag1)": +0.45, "wave_exposure_idx(lag0)": -0.30,
             "air_temp_c(lag0)": +0.10, "decoy_noise": 0.0, "decoy_level": 0.0,
             "ar1_on_eta": 0.45, "trend_per_month_logit": 0.012}
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
    monthly_raw, SYNTHETIC_TRUTH = make_synthetic_monthly(SYNTHETIC_N_MONTHS, SYNTHETIC_SEED)
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
axes[0].plot(monthly["month"], monthly["y_raw"], marker="o", ms=3, lw=1.2, color="tab:green")
axes[0].set_ylabel(RESPONSE_COL)
axes[0].set_title(f"AOI water-hyacinth extent — {RESPONSE_COL} "
                  f"({N_OBSERVED} observed months)")
axes[1].plot(monthly["month"], monthly["y"], marker="o", ms=3, lw=1.2, color="tab:blue")
axes[1].set_ylabel(f"y ({RESPONSE_INFO['transform']})")
axes[2].plot(monthly["month"], monthly.get("coverage_fraction", pd.Series(index=monthly.index)),
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
      f"consecutive calendar months) -> roughly {_n_eff:.0f} independent "
      f"observations behind {N_OBSERVED} months.")
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
          "'not separable from seasonality' in §19, whatever its p-value.")

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
    ax.plot(fit_df["month"], fit_df["y"], marker="o", ms=3, lw=1.2, label="observed")
    ax.plot(fit_df["month"], _fitted, lw=1.6, color="tab:red", label="fitted (M3, in-sample)")
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
                # and say so, rather than letting §19 read it as robustness.
                _sd_freq = STABILITY.loc[STABILITY["is_driver"], "selection_frequency"]
                ENET_SPARSITY_OK = bool(len(_sd_freq) and _sd_freq.min() < 0.90)
                ENET_INFO["sparsity_informative"] = ENET_SPARSITY_OK
                if not ENET_SPARSITY_OK:
                    print(f"\\n*** The chosen l1_ratio ({_l1}) produced a non-sparse fit: "
                          f"every driver survives in >= {_sd_freq.min():.0%} of resamples. "
                          "Selection frequency is therefore UNINFORMATIVE here and §19 "
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
md("""## 14. Model C — non-linear driver shapes (natural-spline GLM)

Hyacinth responses are not linear: temperature has an optimum, and wave
disturbance plausibly matters only above a threshold. Each driver enters as a
natural cubic spline with `SPLINE_DF` degrees of freedom, and two **nested $F$
tests** are reported per driver:

* `p_any_effect_F` — the whole spline block against dropping the driver;
* `p_nonlinearity_F` — the spline block against entering the driver as a
  straight line, i.e. *does the curvature buy anything*.

These are $F$ tests, not HAC Wald tests, and that is deliberate: a spline basis
is collinear by construction, and a HAC Wald test on four collinear restrictions
at $n \\approx 100$ is numerically unstable (it returns $p = 0$ for blocks whose
$\\Delta R^2$ is 0.001). The $F$ tests assume independent residuals, so they are
an **upper bound on the evidence** — autocorrelation-robust inference lives in
§12, and this section is about *shape*. Read `delta_r2` and
`nonlinearity_gain_r2` as the magnitudes that matter.

The partial-effect curves are the interpretable output: the *shape* of each
driver–response relationship with everything else held at its mean, with
HAC-based confidence bands. With ~100 months the honest default is that most
drivers are indistinguishable from linear, and that is itself a reportable
result.
""")

code('''# =====================================================================
# 14. Natural-spline GLM: non-linear driver shapes
# =====================================================================
SPLINE_TESTS = pd.DataFrame()
SPLINE_FIT = None
PARTIAL_EFFECTS = pd.DataFrame()

if not HAVE_PATSY:
    print("patsy unavailable; §14 skipped.")
elif N_FIT < (len(DRIVER_TERMS) * SPLINE_DF + len(SEASON_COLS) + 10):
    print(f"{N_FIT} rows cannot support {len(DRIVER_TERMS)} smooths at df={SPLINE_DF} "
          f"({len(DRIVER_TERMS) * SPLINE_DF} spline columns). §14 skipped — "
          "lower SPLINE_DF or reduce the mechanism set.")
else:
    def _basis(frame, col, df=SPLINE_DF):
        """Natural cubic-spline basis for one driver, named for readability."""
        B = np.asarray(dmatrix(f"cr(x, df={int(df)}) - 1",
                               {"x": frame[col].to_numpy(dtype=float)},
                               return_type="dataframe"))
        return pd.DataFrame(B, columns=[f"{col}__s{i+1}" for i in range(B.shape[1])],
                            index=frame.index)

    _blocks = {c: _basis(fit_df, c) for c in DRIVER_TERMS}
    _ctrl = fit_df[[c for c in SEASON_COLS + TREND_TERMS if c in fit_df.columns]]
    _Xfull = pd.concat([*_blocks.values(), _ctrl], axis=1)
    SPLINE_FIT = fit_hac(fit_df["y"], _Xfull, weights=W, maxlags=HAC_MAXLAGS,
                         months=FIT_MONTHS)

    # Nested F tests, not HAC Wald tests. A natural-spline basis is by
    # construction highly collinear, and a HAC Wald test on 4 collinear
    # restrictions at n ~ 100 is numerically unstable — it returns p = 0 for
    # blocks whose delta R2 is 0.001. The F tests below are stable and
    # interpretable, at the cost of assuming independent residuals, so they are
    # an UPPER BOUND on the evidence and are labelled as such. HAC inference on
    # the linear terms is in §12; this section is about SHAPE.
    def _plain(cols):
        X = sm.add_constant(pd.DataFrame(cols).astype(float), has_constant="add")
        return (sm.OLS(fit_df["y"], X).fit() if W is None
                else sm.WLS(fit_df["y"], X, weights=W / np.nanmean(W)).fit())

    _full_plain = _plain(_Xfull)
    _rows = []
    for c, B in _blocks.items():
        keep = [k for k in _Xfull.columns if k not in B.columns]
        red = _plain(_Xfull[keep])
        # (a) any effect at all: the whole spline block vs dropping the driver.
        f_any = _full_plain.compare_f_test(red)
        # (b) shape beyond a straight line: spline block vs the linear term.
        lin_cols = pd.concat([fit_df[[c]], _Xfull[keep]], axis=1)
        lin = _plain(lin_cols)
        f_nl = _full_plain.compare_f_test(lin)
        _rows.append({"driver": c, "spline_df": B.shape[1],
                      "p_any_effect_F": float(f_any[1]),
                      "p_nonlinearity_F": float(f_nl[1]),
                      "r2_full": _full_plain.rsquared, "r2_without": red.rsquared,
                      "delta_r2": float(_full_plain.rsquared - red.rsquared),
                      "r2_linear_for_this_driver": lin.rsquared,
                      "nonlinearity_gain_r2": float(_full_plain.rsquared - lin.rsquared)})
    SPLINE_TESTS = pd.DataFrame(_rows)
    SPLINE_TESTS["q_fdr"] = bh_fdr(SPLINE_TESTS["p_any_effect_F"]).to_numpy()
    SPLINE_TESTS["q_fdr_nonlinearity"] = bh_fdr(SPLINE_TESTS["p_nonlinearity_F"]).to_numpy()
    SPLINE_TESTS = SPLINE_TESTS.sort_values("delta_r2", ascending=False).reset_index(drop=True)

    print(f"Spline GLM on {N_FIT} months: R2 = {SPLINE_FIT.rsquared:.3f} "
          f"({_Xfull.shape[1]} columns), vs "
          f"{MODEL_A_FITS[HEADLINE_SPEC].rsquared:.3f} for the linear {HEADLINE_SPEC}.")
    print("p-values are nested F tests assuming independent residuals, so they are an "
          "UPPER BOUND on the evidence; use §12 for autocorrelation-robust inference "
          "and read delta_r2 / nonlinearity_gain_r2 as the magnitudes that matter.")
    display(SPLINE_TESTS.round(4))

    # --- Partial-effect curves ------------------------------------------------
    ncol = min(3, len(DRIVER_TERMS))
    nrow = int(np.ceil(len(DRIVER_TERMS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.0 * nrow), squeeze=False)
    _pe_rows = []
    for ax, c in zip(axes.ravel(), DRIVER_TERMS):
        grid = pd.DataFrame({c: np.linspace(fit_df[c].min(), fit_df[c].max(), 80)})
        Bg = pd.DataFrame(
            np.asarray(dmatrix(f"cr(x, df={int(SPLINE_DF)}) - 1",
                               {"x": grid[c].to_numpy()}, return_type="dataframe")),
            columns=_blocks[c].columns)
        Xg = pd.DataFrame(0.0, index=Bg.index, columns=_Xfull.columns)
        for col in Bg.columns:
            Xg[col] = Bg[col].to_numpy()
        for col in _ctrl.columns:
            Xg[col] = float(fit_df[col].mean())
        Xg = sm.add_constant(Xg, has_constant="add")
        pred = SPLINE_FIT.get_prediction(Xg).summary_frame(alpha=0.05)
        centre = float(pred["mean"].mean())
        ax.plot(grid[c], pred["mean"] - centre, color="tab:blue", lw=2)
        ax.fill_between(grid[c], pred["mean_ci_lower"] - centre,
                        pred["mean_ci_upper"] - centre, color="tab:blue", alpha=0.18)
        ax.plot(fit_df[c], np.full(N_FIT, ax.get_ylim()[0]), "|", color="k",
                ms=6, alpha=0.4)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        _q = SPLINE_TESTS.loc[SPLINE_TESTS["driver"] == c, "q_fdr"]
        ax.set_title(f"{c}\\nq = {float(_q.iloc[0]):.3g}" if len(_q) else c, fontsize=9)
        ax.set_xlabel(f"{c} (SD units)")
        ax.set_ylabel(f"partial effect on {RESPONSE_INFO['transform']}")
        ax.grid(alpha=0.3)
        _pe = pd.DataFrame({"driver": c, "x_sd_units": grid[c],
                            "partial_effect": pred["mean"] - centre,
                            "ci_lo": pred["mean_ci_lower"] - centre,
                            "ci_hi": pred["mean_ci_upper"] - centre})
        _pe_rows.append(_pe)
    for ax in axes.ravel()[len(DRIVER_TERMS):]:
        ax.axis("off")
    fig.suptitle("Partial driver–response shapes (natural cubic splines, 95% CI)",
                 y=1.01, fontsize=11)
    fig.tight_layout(); plt.show()
    PARTIAL_EFFECTS = pd.concat(_pe_rows, ignore_index=True)

    _nl = SPLINE_TESTS[SPLINE_TESTS["nonlinearity_gain_r2"] > 0.02]
    if len(_nl):
        print("Drivers where the curve buys real explanatory power over a straight line "
              f"(delta R2 > 0.02): {_nl['driver'].tolist()}")
    else:
        print("No driver gains materially from a non-linear shape; the linear "
              "coefficients in §12 are an adequate summary.")
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
md("""## 16. Out-of-sample skill — do the drivers actually add anything?

In-sample $R^2$ on a persistent monthly series is close to meaningless: season
plus a lagged response will fit it well while knowing nothing about ecology. The
question that matters is whether the *environmental drivers* improve prediction
of months the model has not seen.

Baselines, in increasing order of difficulty to beat:

| Baseline | Why it is the bar |
|---|---|
| **mean** | Predicts the record mean. Anything must beat this. |
| **seasonal-naive** | Predicts the **same calendar month** last year. Free, and hard to beat. |
| **persistence** | Predicts last month. On a system this autocorrelated, usually the strongest baseline. |
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
        "persistence (y_lag1)": (AR_TERMS[:1], None, None),
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
    _baselines = ["persistence (y_lag1)", "season+trend", "mean baseline"]
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
    _driver_best = SKILL[SKILL["specification"].str.contains("drivers|boosting")]["rmse"].min()
    if np.isfinite(_driver_best) and np.isfinite(_bb):
        if _driver_best < _bb:
            print(f"Environmental drivers IMPROVE out-of-sample RMSE by "
                  f"{100 * (1 - _driver_best / _bb):.1f}% over the best simple baseline, "
                  f"on {_n_common} held-out months.")
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
        _pick = [s for s in ["persistence (y_lag1)", "season+trend", "drivers+season+trend"]
                 if s in set(CV_PREDICTIONS["specification"])]
        _obs = (CV_PREDICTIONS[CV_PREDICTIONS["specification"] == _pick[-1]]
                .sort_values("month"))
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(_obs["month"], _obs["y"], "k-o", ms=4, lw=1.4, label="observed")
        for s in _pick:
            g = CV_PREDICTIONS[CV_PREDICTIONS["specification"] == s].sort_values("month")
            ax.plot(g["month"], g["yhat"], lw=1.4, alpha=0.85, marker=".", label=s)
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

The **primary** reading, carried into §19, is the **with persistence**
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
        print(f"\\nPRIMARY reading for §19 is the '{SHARED_VS_UNIQUE_SPEC}' specification, "
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
# 19. Synthesis
# ===========================================================================
md("""## 19. Synthesis — the ranked driver table

Everything above, collapsed into one table per driver, with a **verdict** applied
by explicit rules rather than by eye:

| Verdict | Rule |
|---|---|
| `not separable from season` | $R^2$ of the driver on the seasonal harmonics $\\ge$ `SEASON_CONFOUND_R2`. Reported regardless of significance — the model cannot tell this driver from the calendar. |
| `robust` | FDR $q <$ `FDR_ALPHA` in M3 **and** bootstrap sign stability $\\ge 0.9$ **and** elastic-net selection $\\ge 0.8$ **and** the sign matches the a-priori mechanism. |
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
quoted here only if §12's stationarity gate opened; when it did not, the section
says so and only short-run coefficients may be used.
""")

code('''# =====================================================================
# 19. Synthesis table
# =====================================================================
def _base_driver(term):
    """Strip the lag suffix to recover the driver name used in FORCING."""
    for suffix in ("_lag6", "_lag5", "_lag4", "_lag3", "_lag2", "_lag1"):
        if term.endswith(suffix):
            return term[: -len(suffix)]
    return term


_rows = []
_m3 = MODEL_A_COEFS[MODEL_A_COEFS["specification"] == HEADLINE_SPEC].set_index("term")
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
    for tab, col, out in [
            (SEMI_PARTIAL, "semi_partial_r2", "semi_partial_r2_static"),
            (SEMI_PARTIAL_AR, "semi_partial_r2", "semi_partial_r2_dynamic"),
            (BOOT_SUMMARY, "boot_sign_stability", "boot_sign_stability"),
            (STABILITY, "selection_frequency", "enet_selection_freq"),
            (GBM_IMPORTANCE, "perm_importance_mean", "gbm_perm_importance"),
            (GBM_IMPORTANCE, "rank", "gbm_rank"),
            (SPLINE_TESTS, "q_fdr", "spline_q_fdr"),
            (SPLINE_TESTS, "nonlinearity_gain_r2", "nonlinearity_gain_r2")]:
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
    if (strong and pd.notna(stab) and stab >= 0.9
            and (pd.isna(sel) or sel >= 0.8) and sign_ok):
        return "robust"
    if strong and not sign_ok:
        return "sign contradicts mechanism"
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
SYNTHESIS["direction"] = np.where(SYNTHESIS.get("coef_M3", pd.Series(dtype=float)) > 0,
                                 "increases WH", "decreases WH")
_order = {"robust": 0, "suggestive": 1, "sign contradicts mechanism": 2,
          "not separable from season": 3, "no evidence": 4}
SYNTHESIS["_o"] = SYNTHESIS["verdict"].map(_order)
SYNTHESIS = SYNTHESIS.sort_values(
    ["_o", "shapley_r2"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)

_show = [c for c in ["driver", "lag_months", "mechanism", "expected_sign", "direction",
                     "coef_M3", "ci_lo", "ci_hi", "q_fdr_M3", "shapley_r2",
                     "shapley_share_of_r2", "semi_partial_r2",
                     "last_entry_to_shapley_ratio", "boot_sign_stability",
                     "enet_selection_freq", "gbm_rank", "r2_on_season",
                     "variance_reading", "loyo_sign_stable", "strict_variants_agree",
                     "verdict"]
         if c in SYNTHESIS.columns]
print("=" * 100)
print("RANKED ENVIRONMENTAL DRIVERS OF AOI WATER-HYACINTH EXTENT")
print("=" * 100)
print(f"Variance columns are the '{SHARED_VS_UNIQUE_SPEC or 'n/a'}' specification: the "
      "Shapley and semi-partial values come from ONE fit (same rows, columns, response "
      "and weighting), so their ratio is interpretable.")
if len(SKILL):
    print(f"Predictive skill is from {CV_DESIGN}, on "
          f"{int(SKILL['n_test'].iloc[0])} evaluated month(s).")
display(SYNTHESIS[_show].round(4))

# Long-run effects are quoted ONLY when §12's stationarity gate opened.
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
          "not separable from season", "no evidence"]:
    g = SYNTHESIS[SYNTHESIS["verdict"] == v]
    if not len(g):
        continue
    print(f"\\n{v.upper()} ({len(g)}):")
    for r in g.itertuples():
        c = getattr(r, "coef_M3", np.nan)
        q = getattr(r, "q_fdr_M3", np.nan)
        sh = getattr(r, "shapley_r2", np.nan)
        print(f"  {r.driver} (lag {r.lag_months}): "
              f"{'+' if pd.notna(c) and c > 0 else ''}{c:.3f} per SD"
              f"{f', q = {q:.3g}' if pd.notna(q) else ''}"
              f"{f', Shapley R2 = {sh:.3f}' if pd.notna(sh) else ''}")
        print(f"      mechanism: {r.mechanism}")

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
# 20. Exports
# ===========================================================================
md("""## 20. Export

Every table is written with an `evidence_type` column, mirroring the spatial
notebook's convention, so nothing downstream has to guess whether a number is an
in-sample association, a blocked validation, or a descriptive contrast.
""")

code('''# =====================================================================
# 20. Export tables and a run manifest
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
    "synthesis": (SYNTHESIS, "synthesis"),
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
# 21. Interpretation checklist
# ===========================================================================
md("""## 21. How to read and write up this model

### What each section licenses you to claim

| Section | Claim it supports | Claim it does **not** support |
|---|---|---|
| §12 M3 | "Months with anomalously high *X* have higher WH extent, net of season and trend" | Causation; any spatial statement |
| §12 M4 short-run | "Within a month, a 1-SD increase in *X* is associated with this change, given last month's extent" | A forecast |
| §12 M4 long-run | "A sustained 1-SD increase in *X* is associated with a long-run change of β/(1−ρ)" — **only if `long_run_estimable` is True** | Anything long-run when the AR interval includes a unit root. The multiplier is `NaN` then, and no number from an earlier run may be substituted |
| §13 stability | "The association with *X* is robust to resampling and to collinearity with other drivers" | An effect size — the elastic net shrinks coefficients toward zero by design |
| §14 splines | "The *X*–WH relationship is (non-)linear, with this shape" | A threshold you can manage to, on ~100 months |
| §16 | "Over *k* three-calendar-month rolling-origin windows and *n* evaluated months, the drivers do / do not improve prediction over persistence and season" | Anything about in-sample fit; comparing seasonal-naive with the rest when `eval_set` differs |
| §17 Shapley | "*X* accounts for *n*% of the explained variance in WH extent **in this specification**" | That *X* caused that variance; mixing the with- and without-persistence tables |
| §17 shared vs unique | "*X* does / does not contribute independently of the other drivers, once persistence is in the model" | Reading `last_entry_to_shapley_ratio` as a bounded share |
| §19 | The ranked, verdict-bearing summary — **this is the table to put in the dissertation** | Anything about drivers marked `not separable from season` |

### The five sentences to write

1. The response is the area-weighted AOI mean of classified WH cover over a
   fixed set of *N* grid cells, on *M* months passing a coverage threshold of
   `MIN_MONTHLY_COVERAGE_FRACTION`.
2. Static habitat variables cannot enter a purely temporal model; the drivers
   tested are the time-varying set in §9, each entered at the lag its mechanism
   implies.
3. All inference is autocorrelation-robust (Newey–West, bandwidth *L*) with BH
   FDR control at `FDR_ALPHA`; effective sample size is roughly *n*<sub>eff</sub>
   from §8.
4. The drivers account for *x*% of the explained variance (§17), of which
   season accounts for *y*% and persistence *z*%.
5. Out of sample, the driver model does / does not beat persistence and season
   (§16), and the ranked robustness verdicts are in §19.

### Before quoting a number

- [ ] Is the driver `not separable from season` in §19? Then it has no
      independent evidence, whatever its *p*-value.
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
      `without persistence` and a `with persistence` table; §19 uses the latter.
      Never quote a Shapley value from one beside a semi-partial from the other.
- [ ] Is `last_entry_to_shapley_ratio` above 1? That is possible suppression or
      coefficient instability, not "more than 100% of its share".
- [ ] Are you comparing seasonal-naive RMSE with the other models? Only if
      `eval_set == "common"`. Otherwise quote its own `n_test`, or use the
      restricted table §16 prints for the months it *can* be scored on.
- [ ] Is the validation being described as "three-month-ahead"? Say
      **three-calendar-month rolling-origin windows**, and give the number of
      evaluated months — not the number of folds alone.

### If the drivers look weak

That is a result, not a dead end, and it has three honest readings — state
which one you believe and why:

1. **Not enough months.** ~100 monthly values with lag-1 autocorrelation of
   *r* carry far less information than 100 independent ones (§8). Widen the
   record before widening the claim.
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
