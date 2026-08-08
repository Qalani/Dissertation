"""Builder script that assembles winam_wh_temporal_driver_model.ipynb.

The spatial panel models answer "where is the hyacinth"; this notebook answers
"how much hyacinth is there this month, and which environmental drivers move
that total". It collapses the 500 m cell-month panel to ONE area-weighted
AOI value per month and fits interpretable temporal models to that series.

Kept in the repo (like build_backfill_nb.py / build_inventory_nb.py) so the
notebook can be regenerated from one editable source instead of hand-patching
notebook JSON. Regenerate with:

    python3 build_temporal_model_nb.py
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

---

## Structure

| § | What it does |
|---|---|
| 1–3 | Install, imports, configuration |
| 4–5 | Helper functions (aggregation; statistics) |
| 6–7 | Load the panel → build the AOI monthly series |
| 8–9 | Series diagnostics; **season/trend identifiability audit** |
| 10–11 | Lag structure; model dataset |
| 12 | **Model A** — linear driver model, Newey–West SEs, nested specifications |
| 13 | **Model B** — elastic net + bootstrap stability selection (collinearity-robust ranking) |
| 14 | **Model C** — spline GLM (non-linear driver–response shapes, partial effects) |
| 15 | **Model D** — gradient boosting + out-of-fold permutation importance |
| 16 | Out-of-sample skill vs persistence / seasonal baselines |
| 17 | **Variance partitioning** — Shapley $R^2$ across drivers, season, trend, persistence |
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

# Newey-West bandwidth. None -> floor(4 * (n/100)^(2/9)), the standard rule.
HAC_MAXLAGS = None

# Benjamini-Hochberg FDR level applied across the drivers within a specification.
FDR_ALPHA = 0.10

# Moving-block bootstrap: block length preserves short-range dependence.
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = None        # None -> ceil(n ** (1/3))
BOOTSTRAP_SEED = 7

# Rolling-origin cross-validation. Mirrors the spatial panel's temporal design
# (GAM_N_TEMPORAL_FOLDS / GAM_TEMPORAL_HORIZON_MONTHS /
# GAM_TEMPORAL_MIN_TRAIN_MONTHS) so "fold 1" means the same thing in both.
CV_N_FOLDS = 8
CV_HORIZON_MONTHS = 3
CV_MIN_TRAIN_MONTHS = 24

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
md("""## 5. Helpers — autocorrelation-robust inference

The whole difficulty of a temporal driver model is that consecutive months are
not independent observations. Everything in this cell exists to stop that fact
turning into a false discovery:

* **`fit_hac`** — OLS/WLS with Newey–West (HAC) covariance, bandwidth
  $\\lfloor 4(n/100)^{2/9} \\rfloor$. Naive SEs on a series this persistent are
  routinely 2–3× too small.
* **`bh_fdr`** — Benjamini–Hochberg *q*-values, because several drivers are
  tested at once.
* **`moving_block_bootstrap`** — resamples contiguous blocks, so a bootstrap
  replicate keeps the short-range dependence the model has to survive.
* **`rolling_origin_folds`** — expanding-window folds; the *only* honest way to
  claim skill on a series.
* **`shapley_r2`** — averages each group's $R^2$ contribution over every order
  of entry, so collinear drivers are not judged by who happened to be entered
  first.
* **`semi_partial_r2`** — the drop in $R^2$ from removing one driver, i.e. what
  that driver explains that nothing else in the model does.
""")

code('''# =====================================================================
# 5. Autocorrelation-robust inference helpers
# =====================================================================
from itertools import combinations


def hac_maxlags(n, override=None):
    """Newey-West bandwidth: floor(4 * (n/100)^(2/9)), at least 1."""
    if override is not None:
        return int(override)
    return max(1, int(np.floor(4 * (max(int(n), 1) / 100.0) ** (2.0 / 9.0))))


def fit_hac(y, X, weights=None, maxlags=None, add_const=True):
    """Fit OLS/WLS with Newey-West covariance. Returns the statsmodels result."""
    y = np.asarray(y, dtype=float)
    X = pd.DataFrame(X).astype(float)
    if add_const:
        X = sm.add_constant(X, has_constant="add")
    lags = hac_maxlags(len(y), maxlags)
    if weights is None:
        model = sm.OLS(y, X, missing="drop")
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.nanmean(w)
        model = sm.WLS(y, X, weights=w, missing="drop")
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    res._hac_maxlags = lags
    return res


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
    return tab


def semi_partial_r2(y, X, weights=None, terms=None):
    """Drop in R^2 when each term is removed from the full model."""
    y = np.asarray(y, dtype=float)
    X = pd.DataFrame(X).astype(float)
    terms = list(X.columns) if terms is None else [t for t in terms if t in X.columns]
    full = fit_hac(y, X, weights=weights)
    rows = []
    for t in terms:
        red = fit_hac(y, X.drop(columns=[t]), weights=weights)
        rows.append({"term": t, "r2_full": full.rsquared,
                     "r2_without": red.rsquared,
                     "semi_partial_r2": float(full.rsquared - red.rsquared)})
    return pd.DataFrame(rows)


def moving_block_bootstrap_indices(n, block=None, rng=None, n_boot=1000):
    """Yield index arrays of length n built from contiguous blocks (with wrap)."""
    rng = np.random.default_rng() if rng is None else rng
    L = int(np.ceil(n ** (1 / 3))) if block is None else int(block)
    L = max(1, min(L, n))
    n_blocks = int(np.ceil(n / L))
    for _ in range(int(n_boot)):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]
        yield idx


def bootstrap_coefficients(y, X, weights=None, n_boot=1000, block=None, seed=0):
    """Moving-block bootstrap distribution of the OLS/WLS coefficients."""
    y = np.asarray(y, dtype=float)
    X = pd.DataFrame(X).astype(float)
    rng = np.random.default_rng(seed)
    Xc = sm.add_constant(X, has_constant="add")
    w = None if weights is None else np.asarray(weights, dtype=float)
    draws = []
    for idx in moving_block_bootstrap_indices(len(y), block=block, rng=rng, n_boot=n_boot):
        Xi, yi = Xc.iloc[idx], y[idx]
        try:
            if w is None:
                b = sm.OLS(yi, Xi).fit().params
            else:
                b = sm.WLS(yi, Xi, weights=w[idx]).fit().params
            draws.append(b)
        except Exception:
            continue
    if not draws:
        return pd.DataFrame(columns=Xc.columns)
    return pd.DataFrame(draws).reset_index(drop=True)


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


def rolling_origin_folds(n, n_folds=8, horizon=3, min_train=24):
    """Expanding-window (train, test) index pairs covering the end of the record.

    The last `n_folds * horizon` observations are split into consecutive test
    blocks, each trained on everything before it. Folds that would leave fewer
    than `min_train` training rows are dropped and the shortfall is reported by
    the caller.
    """
    folds = []
    total_test = int(n_folds) * int(horizon)
    start = max(int(min_train), n - total_test)
    cut = start
    while cut < n and len(folds) < int(n_folds):
        test = np.arange(cut, min(cut + int(horizon), n))
        train = np.arange(0, cut)
        if len(train) >= int(min_train) and len(test) > 0:
            folds.append((train, test))
        cut += int(horizon)
    return folds


def cv_scores(y, X, folds, weights=None, fit=None, predict=None):
    """Out-of-sample RMSE / MAE / R^2 over rolling-origin folds.

    `fit(Xtr, ytr, wtr) -> model` and `predict(model, Xte) -> yhat` default to
    ordinary least squares, so the same routine scores every specification and
    every learner on identical folds.
    """
    y = np.asarray(y, dtype=float)
    X = pd.DataFrame(X).astype(float)
    if fit is None:
        def fit(Xtr, ytr, wtr):
            Xtr = sm.add_constant(Xtr, has_constant="add")
            return (sm.OLS(ytr, Xtr).fit() if wtr is None
                    else sm.WLS(ytr, Xtr, weights=wtr).fit())
    if predict is None:
        def predict(model, Xte):
            return model.predict(sm.add_constant(Xte, has_constant="add"))
    preds, truths, fold_ids = [], [], []
    for k, (tr, te) in enumerate(folds, start=1):
        wtr = None if weights is None else np.asarray(weights, dtype=float)[tr]
        try:
            model = fit(X.iloc[tr], y[tr], wtr)
            yhat = np.asarray(predict(model, X.iloc[te]), dtype=float)
        except Exception:
            continue
        preds.append(yhat); truths.append(y[te]); fold_ids.append(np.full(len(te), k))
    if not preds:
        return {"n_folds": 0, "n_test": 0, "rmse": np.nan, "mae": np.nan, "r2_oos": np.nan}, \\
            pd.DataFrame(columns=["fold", "y", "yhat"])
    yh = np.concatenate(preds); yt = np.concatenate(truths)
    ss_res = float(np.nansum((yt - yh) ** 2))
    ss_tot = float(np.nansum((yt - np.nanmean(yt)) ** 2))
    detail = pd.DataFrame({"fold": np.concatenate(fold_ids), "y": yt, "yhat": yh})
    return {"n_folds": len(preds), "n_test": int(len(yt)),
            "rmse": float(np.sqrt(np.nanmean((yt - yh) ** 2))),
            "mae": float(np.nanmean(np.abs(yt - yh))),
            "r2_oos": (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}, detail


def shapley_r2(y, X, groups, weights=None):
    """Shapley decomposition of R^2 across `groups` ({name: [columns]}).

    Each group's value is its average marginal contribution to R^2 over every
    order in which the groups could enter, so two collinear drivers split the
    variance they share instead of the first-entered one taking all of it. The
    values sum to the full-model R^2 by construction.
    """
    y = np.asarray(y, dtype=float)
    X = pd.DataFrame(X).astype(float)
    names = [g for g, cols in groups.items() if any(c in X.columns for c in cols)]
    G = len(names)
    if G == 0:
        return pd.DataFrame(columns=["group", "shapley_r2", "share_of_r2"])
    if G > 14:
        raise ValueError(f"{G} groups is 2^{G} fits; reduce PARTITION_MAX_GROUPS")

    def r2_of(subset):
        cols = [c for g in subset for c in groups[g] if c in X.columns]
        if not cols:
            return 0.0
        try:
            return float(fit_hac(y, X[cols], weights=weights).rsquared)
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
    out = pd.DataFrame({"group": list(vals), "shapley_r2": list(vals.values())})
    out["share_of_r2"] = out["shapley_r2"] / full if full and np.isfinite(full) else np.nan
    out["r2_full_model"] = full
    return out.sort_values("shapley_r2", ascending=False).reset_index(drop=True)


def acf_values(x, nlags=24):
    """Autocorrelations 1..nlags with the +-1.96/sqrt(n) white-noise band."""
    s = pd.Series(x, dtype=float).dropna()
    n = len(s)
    v = s - s.mean()
    denom = float((v ** 2).sum())
    out = []
    for k in range(1, int(nlags) + 1):
        if k >= n or denom == 0:
            out.append(np.nan); continue
        out.append(float((v.iloc[k:].to_numpy() * v.iloc[:-k].to_numpy()).sum() / denom))
    return pd.DataFrame({"lag": np.arange(1, int(nlags) + 1), "acf": out,
                         "band": 1.96 / np.sqrt(max(n, 1))})


def cross_correlation(driver, response, max_lag=6):
    """Correlation of response(t) with driver(t - k) for k = 0..max_lag.

    Positive k means the driver LEADS the response, which is the only direction
    a driver claim can use. Pairs are those where both are observed.
    """
    d = pd.Series(driver, dtype=float).reset_index(drop=True)
    r = pd.Series(response, dtype=float).reset_index(drop=True)
    rows = []
    for k in range(0, int(max_lag) + 1):
        dk = d.shift(k)
        ok = dk.notna() & r.notna()
        n = int(ok.sum())
        if n < 8:
            rows.append({"lag": k, "n": n, "r": np.nan, "p": np.nan}); continue
        rr, pp = sstats.pearsonr(dk[ok], r[ok])
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


print("§5 inference helpers defined.")
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
acf_y = acf_values(monthly["y"], nlags=min(24, max(4, N_OBSERVED // 3)))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].bar(acf_y["lag"], acf_y["acf"], color="tab:blue", alpha=0.8)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].axhline(acf_y["band"].iloc[0], color="red", ls="--", lw=1)
axes[0].axhline(-acf_y["band"].iloc[0], color="red", ls="--", lw=1)
axes[0].set_title("ACF of the response"); axes[0].set_xlabel("lag (months)")
axes[0].grid(alpha=0.3)

# Seasonal climatology: the share of variance the calendar month alone explains.
clim = obs.groupby("month_num")["y"].agg(["mean", "std", "count"])
axes[1].errorbar(clim.index, clim["mean"], yerr=clim["std"], marker="o", capsize=3)
axes[1].set_title("Monthly climatology of the response")
axes[1].set_xlabel("calendar month"); axes[1].set_xticks(range(1, 13)); axes[1].grid(alpha=0.3)
fig.tight_layout(); plt.show()

_lag1_r = float(pd.Series(monthly["y"]).corr(pd.Series(monthly["y"]).shift(1)))
_season_r2 = float(fit_hac(obs["y"], obs[SEASON_COLS]).rsquared)
_trend_r2 = float(fit_hac(obs["y"], obs[["time_index"]]).rsquared)
_n_eff = N_OBSERVED * (1 - _lag1_r) / (1 + _lag1_r) if abs(_lag1_r) < 1 else np.nan

SERIES_DIAGNOSTICS = pd.DataFrame([{
    "n_observed_months": N_OBSERVED,
    "lag1_autocorrelation": _lag1_r,
    "effective_n_bartlett": _n_eff,
    "r2_season_only": _season_r2,
    "r2_trend_only": _trend_r2,
    "hac_maxlags": hac_maxlags(N_OBSERVED, HAC_MAXLAGS),
}])
display(SERIES_DIAGNOSTICS.T.rename(columns={0: "value"}))

print(f"Lag-1 autocorrelation {_lag1_r:.2f} -> roughly {_n_eff:.0f} independent "
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
        row["r2_on_season"] = float(fit_hac(sub[c], sub[SEASON_COLS]).rsquared)
        row["r2_on_trend"] = float(fit_hac(sub[c], sub[["time_index"]]).rsquared)
        resid = fit_hac(sub[c], sub[SEASON_COLS + ["time_index"]]).resid
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
    t = cross_correlation(monthly[c], monthly["y"], max_lag=LAG_SCAN_MAX)
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
        ax.set_xlabel("driver lead (months)"); ax.set_ylim(-1, 1); ax.grid(alpha=0.3)
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

print(f"Driver terms ({len(DRIVER_TERMS)}): {DRIVER_TERMS}")
print(f"Season terms ({len(SEASON_COLS)}): {SEASON_COLS}")
print(f"Trend terms: {TREND_TERMS} | AR terms: {AR_TERMS}")
print(f"Proxy terms (descriptive only): {PROXY_TERMS or 'none'}")
print(f"\\nRows: {N_FIT} for the static specifications, {len(dyn_df)} for the dynamic one "
      f"(of {N_OBSERVED} observed months; lagging costs the leading months).")
print(f"Newey-West bandwidth: {hac_maxlags(N_FIT, HAC_MAXLAGS)} months")
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
| **M4** | M3 + lagged response | Dynamic. Short-run effect plus the implied **long-run** effect $\\beta/(1-\\rho)$, since a persistent system keeps accumulating a sustained push. |

Coefficients are per standard deviation of the driver, on the
`RESPONSE_TRANSFORM` scale of the response. Standard errors are HAC; *p*-values
carry BH FDR *q*-values across drivers within each specification.
""")

code('''# =====================================================================
# 12. Model A - nested linear specifications with HAC inference
# =====================================================================
SPECS = {
    "M0 season+trend":        SEASON_COLS + TREND_TERMS,
    "M1 drivers only":        DRIVER_TERMS,
    "M2 drivers+season":      DRIVER_TERMS + SEASON_COLS,
    "M3 drivers+season+trend": DRIVER_TERMS + SEASON_COLS + TREND_TERMS,
}
HEADLINE_SPEC = "M3 drivers+season+trend"

MODEL_A_FITS, _coef_frames, _fit_rows = {}, [], []
for name, cols in SPECS.items():
    cols = [c for c in cols if c in fit_df.columns]
    if not cols:
        continue
    res = fit_hac(fit_df["y"], fit_df[cols], weights=W, maxlags=HAC_MAXLAGS)
    MODEL_A_FITS[name] = res
    _coef_frames.append(tidy_coefficients(res, keep=DRIVER_TERMS, label=name))
    _fit_rows.append({"specification": name, "n": int(res.nobs),
                      "k_terms": len(cols), "r2": res.rsquared,
                      "adj_r2": res.rsquared_adj, "aic": res.aic, "bic": res.bic,
                      "hac_maxlags": res._hac_maxlags,
                      "resid_lag1_acf": float(pd.Series(res.resid).autocorr(1))})

# --- M4: dynamic specification (short-run and long-run effects) ---------------
LONGRUN = pd.DataFrame()
if AR_TERMS and len(dyn_df) > len(DRIVER_TERMS) + len(SEASON_COLS) + 3:
    _cols = [c for c in DRIVER_TERMS + SEASON_COLS + TREND_TERMS + AR_TERMS
             if c in dyn_df.columns]
    res4 = fit_hac(dyn_df["y"], dyn_df[_cols], weights=W_DYN, maxlags=HAC_MAXLAGS)
    MODEL_A_FITS["M4 dynamic (+AR)"] = res4
    _coef_frames.append(tidy_coefficients(res4, keep=DRIVER_TERMS + AR_TERMS,
                                          label="M4 dynamic (+AR)"))
    _fit_rows.append({"specification": "M4 dynamic (+AR)", "n": int(res4.nobs),
                      "k_terms": len(_cols), "r2": res4.rsquared,
                      "adj_r2": res4.rsquared_adj, "aic": res4.aic, "bic": res4.bic,
                      "hac_maxlags": res4._hac_maxlags,
                      "resid_lag1_acf": float(pd.Series(res4.resid).autocorr(1))})

    # Long-run multiplier beta / (1 - rho), SE by the delta method on the HAC
    # covariance. A sustained 1-SD push accumulates in a persistent system, so
    # the long-run effect is the ecologically meaningful magnitude.
    rho = float(sum(res4.params.get(a, 0.0) for a in AR_TERMS))
    if abs(1 - rho) > 1e-6:
        V = res4.cov_params()
        rows = []
        for c in DRIVER_TERMS:
            if c not in res4.params.index:
                continue
            b = float(res4.params[c])
            g = np.zeros(len(res4.params))
            names = list(res4.params.index)
            g[names.index(c)] = 1.0 / (1 - rho)
            for a in AR_TERMS:
                if a in names:
                    g[names.index(a)] = b / (1 - rho) ** 2
            se = float(np.sqrt(max(g @ V.to_numpy() @ g, 0.0)))
            rows.append({"term": c, "short_run": b, "long_run": b / (1 - rho),
                         "long_run_se": se,
                         "long_run_ci_lo": b / (1 - rho) - 1.96 * se,
                         "long_run_ci_hi": b / (1 - rho) + 1.96 * se})
        LONGRUN = pd.DataFrame(rows)
        LONGRUN.insert(0, "rho_sum_ar", rho)
else:
    print("Dynamic specification skipped (too few rows once the AR term is lagged).")

MODEL_A_FIT_STATS = pd.DataFrame(_fit_rows)
MODEL_A_COEFS = pd.concat(_coef_frames, ignore_index=True) if _coef_frames else pd.DataFrame()

print("Fit statistics by specification (in-sample; skill is §16's job):")
display(MODEL_A_FIT_STATS.round(4))
print("\\nDriver coefficients (per 1 SD of the driver, HAC SEs, BH q within spec):")
display(MODEL_A_COEFS.round(4))
if len(LONGRUN):
    print(f"\\nLong-run effects (sum of AR coefficients rho = {LONGRUN['rho_sum_ar'].iloc[0]:.3f}; "
          f"multiplier 1/(1-rho) = {1 / (1 - LONGRUN['rho_sum_ar'].iloc[0]):.2f}):")
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

# --- Semi-partial R2 in the headline specification ---------------------------
_hs_cols = [c for c in SPECS[HEADLINE_SPEC] if c in fit_df.columns]
SEMI_PARTIAL = semi_partial_r2(fit_df["y"], fit_df[_hs_cols], weights=W, terms=DRIVER_TERMS)
print(f"\\nUnique explanatory contribution in {HEADLINE_SPEC} "
      "(drop in R2 when the driver is removed):")
display(SEMI_PARTIAL.round(4))
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
    acf_r = acf_values(resid, nlags=min(18, max(4, len(resid) // 3)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    axes[0].bar(acf_r["lag"], acf_r["acf"], color="tab:red", alpha=0.75)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhline(acf_r["band"].iloc[0], color="k", ls="--", lw=0.8)
    axes[0].axhline(-acf_r["band"].iloc[0], color="k", ls="--", lw=0.8)
    axes[0].set_title(f"Residual ACF — {HEADLINE_SPEC}"); axes[0].set_xlabel("lag")
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
_boot_draws = bootstrap_coefficients(
    fit_df["y"], fit_df[_hs_cols], weights=W,
    n_boot=BOOTSTRAP_N, block=BOOTSTRAP_BLOCK, seed=BOOTSTRAP_SEED)
BOOT_SUMMARY = bootstrap_summary(_boot_draws, terms=DRIVER_TERMS)
_bl = BOOTSTRAP_BLOCK or int(np.ceil(N_FIT ** (1 / 3)))
print(f"Moving-block bootstrap: {len(_boot_draws):,} replicates, block length {_bl} months")
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
        _rp = fit_hac(_pdf["y"], _pdf[_cols], weights=_w, maxlags=HAC_MAXLAGS)
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
    _folds = rolling_origin_folds(len(_y), CV_N_FOLDS, CV_HORIZON_MONTHS, CV_MIN_TRAIN_MONTHS)
    if len(_folds) < 2:
        print(f"Only {len(_folds)} rolling-origin fold(s) fit in {len(_y)} rows; "
              "lower CV_MIN_TRAIN_MONTHS or CV_HORIZON_MONTHS. §13 skipped.")
    else:
        _cv = [(tr, te) for tr, te in _folds]
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
                _y, _X, _folds,
                fit=lambda Xtr, ytr, wtr, _a=m.alpha_, _l=l1: ElasticNet(
                    alpha=_a, l1_ratio=_l, max_iter=20000,
                    random_state=RANDOM_STATE).fit(Xtr, ytr),
                predict=lambda mod, Xte: mod.predict(Xte))
            print(f"  l1_ratio {l1:.1f}: lambda {m.alpha_:.4g}, rolling-origin RMSE "
                  f"{sc['rmse']:.4f}")
            if np.isfinite(sc["rmse"]) and sc["rmse"] < _best_rmse:
                _best, _best_rmse = (l1, m.alpha_), sc["rmse"]

        if _best is None:
            print("No elastic-net configuration converged; §13 skipped.")
        else:
            _l1, _lam = _best
            ENET_INFO = {"l1_ratio": _l1, "alpha": float(_lam), "cv_rmse": float(_best_rmse),
                         "n_folds": len(_folds)}
            _final = ElasticNet(alpha=_lam, l1_ratio=_l1, max_iter=20000,
                                random_state=RANDOM_STATE).fit(_X, _y)
            ENET_COEFS = pd.DataFrame({"term": _X.columns, "enet_coef": _final.coef_})
            ENET_COEFS["selected"] = ENET_COEFS["enet_coef"].abs() > 1e-8
            ENET_COEFS["is_driver"] = ENET_COEFS["term"].isin(DRIVER_TERMS)
            print(f"\\nChosen: l1_ratio {_l1}, lambda {_lam:.4g} "
                  f"(rolling-origin RMSE {_best_rmse:.4f} over {len(_folds)} folds)")
            display(ENET_COEFS.sort_values("enet_coef", key=np.abs, ascending=False).round(4))

            # --- Stability selection over moving-block resamples --------------
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            _block = BOOTSTRAP_BLOCK or int(np.ceil(len(_y) ** (1 / 3)))
            n_boot_stab = min(int(BOOTSTRAP_N), 600)   # each replicate refits the net
            counts = {c: {"nonzero": 0, "pos": 0, "neg": 0} for c in _X.columns}
            n_ok = 0
            for idx in moving_block_bootstrap_indices(len(_y), block=BOOTSTRAP_BLOCK,
                                                      rng=rng, n_boot=n_boot_stab):
                try:
                    mb = ElasticNet(alpha=_lam, l1_ratio=_l1, max_iter=20000,
                                    random_state=RANDOM_STATE).fit(_X.iloc[idx], _y[idx])
                except Exception:
                    continue
                n_ok += 1
                for c, b in zip(_X.columns, mb.coef_):
                    if abs(b) > 1e-8:
                        counts[c]["nonzero"] += 1
                        counts[c]["pos" if b > 0 else "neg"] += 1
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
                print(f"\\nStability selection over {n_ok} moving-block resamples "
                      f"(block {_block} months):")
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
    SPLINE_FIT = fit_hac(fit_df["y"], _Xfull, weights=W, maxlags=HAC_MAXLAGS)

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
    _folds = rolling_origin_folds(len(_yg), CV_N_FOLDS, CV_HORIZON_MONTHS, CV_MIN_TRAIN_MONTHS)
    if len(_folds) < 2:
        print("Too few rolling-origin folds for §15.")
    else:
        GBM_SCORE, _ = cv_scores(
            _yg, _Xg, _folds,
            fit=lambda Xtr, ytr, wtr: HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xtr, ytr),
            predict=lambda m, Xte: m.predict(Xte))
        print(f"Gradient boosting, rolling-origin: RMSE {GBM_SCORE['rmse']:.4f}, "
              f"R2_oos {GBM_SCORE['r2_oos']:.3f} over {GBM_SCORE['n_folds']} folds")

        _imp = {c: [] for c in _Xg.columns}
        for tr, te in _folds:
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
| **seasonal-naive** | Predicts the same calendar month last year. Free, and hard to beat. |
| **persistence** | Predicts last month. On a system this autocorrelated, usually the strongest baseline. |
| **season+trend** | The calendar alone. |
| **drivers+season+trend** | The headline model. |
| **+AR(1)** | The dynamic model. |

**If the driver models do not beat persistence and season, the correct
conclusion is that the drivers explain WH extent but do not predict it** — a
finding about identifiability, not a failure to report.
""")

code('''# =====================================================================
# 16. Rolling-origin out-of-sample skill
# =====================================================================
_cvdf = model_df.loc[
    model_df[["y"] + DRIVER_TERMS + SEASON_COLS + TREND_TERMS + AR_TERMS]
    .notna().all(axis=1)].reset_index(drop=True)
_ycv = _cvdf["y"].to_numpy(dtype=float)
CV_FOLDS = rolling_origin_folds(len(_ycv), CV_N_FOLDS, CV_HORIZON_MONTHS, CV_MIN_TRAIN_MONTHS)
print(f"Rolling-origin design: {len(CV_FOLDS)} fold(s), horizon {CV_HORIZON_MONTHS} month(s), "
      f"minimum training window {CV_MIN_TRAIN_MONTHS} months, on {len(_ycv)} complete rows.")
for k, (tr, te) in enumerate(CV_FOLDS, 1):
    print(f"  fold {k}: train {_cvdf['month'].iloc[tr[0]]:%Y-%m}..{_cvdf['month'].iloc[tr[-1]]:%Y-%m} "
          f"({len(tr)} mo) -> test {_cvdf['month'].iloc[te[0]]:%Y-%m}.."
          f"{_cvdf['month'].iloc[te[-1]]:%Y-%m} ({len(te)} mo)")

SKILL = pd.DataFrame()
CV_PREDICTIONS = pd.DataFrame()
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
        sc, det = cv_scores(_ycv, _cvdf[use], CV_FOLDS,
                            weights=None if fitf else _wcv, fit=fitf, predict=predf)
        sc["specification"] = name
        sc["n_terms"] = 0 if cols is None else len(use)
        _rows.append(sc)
        det = det.assign(specification=name)
        _details.append(det)

    # Seasonal-naive: same calendar month one year earlier.
    _sn = _cvdf["y"].shift(12)
    _sn_rows = []
    for k, (tr, te) in enumerate(CV_FOLDS, 1):
        yy, yh = _ycv[te], _sn.to_numpy()[te]
        ok = np.isfinite(yy) & np.isfinite(yh)
        if ok.any():
            _sn_rows.append(pd.DataFrame({"fold": k, "y": yy[ok], "yhat": yh[ok],
                                          "specification": "seasonal-naive (y_{t-12})"}))
    if _sn_rows:
        sn = pd.concat(_sn_rows, ignore_index=True)
        ss_tot = float(((sn["y"] - sn["y"].mean()) ** 2).sum())
        _rows.append({"specification": "seasonal-naive (y_{t-12})", "n_terms": 0,
                      "n_folds": sn["fold"].nunique(), "n_test": len(sn),
                      "rmse": float(np.sqrt(((sn["y"] - sn["yhat"]) ** 2).mean())),
                      "mae": float((sn["y"] - sn["yhat"]).abs().mean()),
                      "r2_oos": 1 - float(((sn["y"] - sn["yhat"]) ** 2).sum()) / ss_tot
                      if ss_tot > 0 else np.nan})
        _details.append(sn)

    SKILL = pd.DataFrame(_rows).sort_values("rmse").reset_index(drop=True)
    CV_PREDICTIONS = pd.concat(_details, ignore_index=True)
    SKILL["rmse_vs_best_baseline"] = SKILL["rmse"] / SKILL.loc[
        SKILL["specification"].isin(["mean baseline", "persistence (y_lag1)",
                                     "seasonal-naive (y_{t-12})", "season+trend"]),
        "rmse"].min()
    display(SKILL.round(4))

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(SKILL) + 1.4))
    _cols = ["tab:green" if "drivers" in s else "tab:grey" for s in SKILL["specification"]]
    ax.barh(SKILL["specification"], SKILL["rmse"], color=_cols, alpha=0.85)
    _bb = SKILL.loc[SKILL["specification"].isin(
        ["persistence (y_lag1)", "seasonal-naive (y_{t-12})", "season+trend"]), "rmse"].min()
    ax.axvline(_bb, color="red", ls="--", lw=1.2, label="best simple baseline")
    ax.invert_yaxis()
    ax.set_xlabel(f"rolling-origin RMSE on {RESPONSE_INFO['transform']}({RESPONSE_COL})")
    ax.set_title("Out-of-sample skill — green = uses environmental drivers")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show()

    _driver_best = SKILL[SKILL["specification"].str.contains("drivers|boosting")]["rmse"].min()
    if np.isfinite(_driver_best) and np.isfinite(_bb):
        if _driver_best < _bb:
            print(f"Environmental drivers IMPROVE out-of-sample RMSE by "
                  f"{100 * (1 - _driver_best / _bb):.1f}% over the best simple baseline.")
        else:
            print(f"Environmental drivers do NOT beat the best simple baseline "
                  f"({_driver_best:.4f} vs {_bb:.4f}).")
            print("Report this plainly: the drivers are ASSOCIATED with WH extent (§12) and "
                  "account for a share of its variance (§17), but at this sample size they "
                  "do not forecast it better than persistence/season. That is a statement "
                  "about the record's length and the drivers' seasonality, not evidence "
                  "that the ecology is wrong.")

    if len(CV_PREDICTIONS):
        _pick = ["persistence (y_lag1)", "season+trend", "drivers+season+trend"]
        fig, ax = plt.subplots(figsize=(11, 3.6))
        _obs = CV_PREDICTIONS[CV_PREDICTIONS["specification"] == _pick[-1]]
        ax.plot(np.arange(len(_obs)), _obs["y"], "k-o", ms=4, lw=1.4, label="observed")
        for s in _pick:
            g = CV_PREDICTIONS[CV_PREDICTIONS["specification"] == s]
            if len(g) == len(_obs):
                ax.plot(np.arange(len(g)), g["yhat"], lw=1.4, alpha=0.85, label=s)
        ax.set_xlabel("held-out month (concatenated rolling-origin folds)")
        ax.set_ylabel(f"{RESPONSE_INFO['transform']}({RESPONSE_COL})")
        ax.set_title("Held-out predictions across the rolling-origin folds")
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
""")

code('''# =====================================================================
# 17. Shapley R^2 partitioning
# =====================================================================
_groups = {c: [c] for c in DRIVER_TERMS}
if SEASON_COLS:
    _groups["season (annual cycle)"] = SEASON_COLS
if TREND_TERMS:
    _groups["trend (multi-year)"] = TREND_TERMS

PARTITION = pd.DataFrame()
PARTITION_AR = pd.DataFrame()
SHARED_VS_UNIQUE = pd.DataFrame()

if len(_groups) > PARTITION_MAX_GROUPS:
    print(f"{len(_groups)} groups exceeds PARTITION_MAX_GROUPS = {PARTITION_MAX_GROUPS} "
          f"(2^{len(_groups)} fits). Reduce the mechanism set or raise the cap.")
else:
    PARTITION = shapley_r2(fit_df["y"], fit_df[[c for g in _groups.values() for c in g]],
                           _groups, weights=W)
    PARTITION["kind"] = np.where(PARTITION["group"].isin(DRIVER_TERMS),
                                 "environmental driver", "control")
    print(f"Shapley R2 partition WITHOUT persistence "
          f"(full-model R2 = {PARTITION['r2_full_model'].iloc[0]:.3f}):")
    display(PARTITION.round(4))

    if PARTITION_INCLUDE_AR and AR_TERMS and len(dyn_df) > len(_groups) + 5:
        _g2 = dict(_groups); _g2["persistence (y_lag1)"] = AR_TERMS
        if len(_g2) <= PARTITION_MAX_GROUPS:
            _cols2 = [c for g in _g2.values() for c in g if c in dyn_df.columns]
            PARTITION_AR = shapley_r2(dyn_df["y"], dyn_df[_cols2], _g2, weights=W_DYN)
            PARTITION_AR["kind"] = np.where(
                PARTITION_AR["group"].isin(DRIVER_TERMS), "environmental driver", "control")
            print(f"\\nShapley R2 partition WITH persistence "
                  f"(full-model R2 = {PARTITION_AR['r2_full_model'].iloc[0]:.3f}):")
            display(PARTITION_AR.round(4))

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
                     f"(full-model $R^2$ = {tab['r2_full_model'].iloc[0]:.3f})", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show()

    _env = _p.loc[_p["kind"] == "environmental driver", "shapley_r2"].sum()
    _tot = float(_p["r2_full_model"].iloc[0])
    print(f"\\nEnvironmental drivers together account for {_env:.3f} of the "
          f"{_tot:.3f} explained (i.e. {_env / _tot:.0%} of the model's explanatory power, "
          f"{_env:.1%} of the total variance in the response).")
    _top = _p[_p["kind"] == "environmental driver"].head(3)
    if len(_top):
        print("Most strongly linked environmental variables, by variance accounted for:")
        for i, r in enumerate(_top.itertuples(), 1):
            print(f"  {i}. {r.group}: Shapley R2 = {r.shapley_r2:.4f} "
                  f"({r.share_of_r2:.1%} of the model's explanatory power)")

    # --- Shared vs unique: the trap in any variance partition -----------------
    # Shapley SPLITS variance two collinear drivers share, so a driver that is
    # merely a proxy for a real one collects a sizeable Shapley value while
    # explaining nothing the model does not already have. Comparing Shapley with
    # the semi-partial (unique) contribution separates the two cases, and the
    # distinction decides which drivers may be named as mechanisms.
    SHARED_VS_UNIQUE = _p[_p["kind"] == "environmental driver"][
        ["group", "shapley_r2", "share_of_r2"]].rename(columns={"group": "term"})
    SHARED_VS_UNIQUE = SHARED_VS_UNIQUE.merge(
        SEMI_PARTIAL[["term", "semi_partial_r2"]], on="term", how="left")
    SHARED_VS_UNIQUE["unique_share_of_shapley"] = (
        SHARED_VS_UNIQUE["semi_partial_r2"] / SHARED_VS_UNIQUE["shapley_r2"].replace(0, np.nan))
    # Graded on BOTH the ratio and the absolute unique contribution. A ratio
    # alone is too harsh when the full-model R2 is high: in a strongly seasonal
    # system every driver shares most of its variance, yet a driver uniquely
    # explaining 2% of the response is still saying something of its own.
    _u = SHARED_VS_UNIQUE["semi_partial_r2"]
    _ratio = SHARED_VS_UNIQUE["unique_share_of_shapley"]
    SHARED_VS_UNIQUE["reading"] = np.select(
        [(_ratio >= 0.25) | (_u >= 0.02),
         _u >= 0.005,
         SHARED_VS_UNIQUE["shapley_r2"] > 0.02],
        ["independent contribution",
         "partly shared with correlated drivers",
         "SHARED — proxy for a correlated driver"],
        default="negligible either way")
    SHARED_VS_UNIQUE = SHARED_VS_UNIQUE.sort_values("shapley_r2", ascending=False) \\
        .reset_index(drop=True)
    print("\\nShared vs unique variance — which drivers are proxies for each other:")
    display(SHARED_VS_UNIQUE.round(4))
    _proxyish = SHARED_VS_UNIQUE.loc[
        SHARED_VS_UNIQUE["reading"].str.startswith("SHARED"), "term"].tolist()
    if _proxyish:
        print(f"Sizeable Shapley value but almost NO unique contribution: {_proxyish}.")
        print("Those drivers are collinear stand-ins for another driver in the set. Do NOT "
              "name them as independent mechanisms — say the mechanism they share, and "
              "state that the data cannot separate the members of that group.")
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
    res = fit_hac(sub["_y"], sub[cols], weights=w, maxlags=HAC_MAXLAGS)
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
    res = fit_hac(sub["y"], sub[cols], weights=w, maxlags=HAC_MAXLAGS)
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
    res = fit_hac(sub["y"], sub[cols], weights=w, maxlags=HAC_MAXLAGS)
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
    _anom["y"] = fit_hac(fit_df["y"], fit_df[_ctrl_cols], weights=W).resid
    for c in DRIVER_TERMS:
        _anom[c] = fit_hac(fit_df[c], fit_df[_ctrl_cols]).resid
    _ra = fit_hac(_anom["y"], _anom[DRIVER_TERMS], weights=W, maxlags=HAC_MAXLAGS)
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
    _rd = fit_hac(sub["dy"], sub[_dcols + SEASON_COLS], weights=w, maxlags=HAC_MAXLAGS)
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
  the set — name the shared mechanism, not the individual variable.
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
            (SEMI_PARTIAL, "semi_partial_r2", "semi_partial_r2"),
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
    if len(SHARED_VS_UNIQUE):
        hit = SHARED_VS_UNIQUE.loc[SHARED_VS_UNIQUE["term"] == term, "reading"]
        row["variance_reading"] = str(hit.iloc[0]) if len(hit) else ""
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
                     "shapley_share_of_r2", "semi_partial_r2", "boot_sign_stability",
                     "enet_selection_freq", "gbm_rank", "r2_on_season",
                     "variance_reading", "loyo_sign_stable", "strict_variants_agree",
                     "verdict"]
         if c in SYNTHESIS.columns]
print("=" * 100)
print("RANKED ENVIRONMENTAL DRIVERS OF AOI WATER-HYACINTH EXTENT")
print("=" * 100)
display(SYNTHESIS[_show].round(4))

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
    "semi_partial_r2": (SEMI_PARTIAL, "environmental association"),
    "bootstrap_coefficients": (BOOT_SUMMARY, "environmental association"),
    "ci_comparison": (CI_COMPARISON, "diagnostic"),
    "proxy_coefficients": (PROXY_COEFS, "descriptive association"),
    "elastic_net_coefficients": (ENET_COEFS, "environmental association"),
    "stability_selection": (STABILITY, "environmental association"),
    "spline_tests": (SPLINE_TESTS, "environmental association"),
    "partial_effects": (PARTIAL_EFFECTS, "environmental association"),
    "gbm_permutation_importance": (GBM_IMPORTANCE, "blocked validation"),
    "skill_rolling_origin": (SKILL, "blocked validation"),
    "cv_predictions": (CV_PREDICTIONS, "blocked validation"),
    "variance_partition": (PARTITION, "environmental association"),
    "variance_partition_with_ar": (PARTITION_AR, "environmental association"),
    "shared_vs_unique_variance": (SHARED_VS_UNIQUE, "environmental association"),
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
        "hac_maxlags": hac_maxlags(N_FIT, HAC_MAXLAGS),
        "fdr_alpha": FDR_ALPHA,
        "cv_design": {"n_folds_requested": CV_N_FOLDS, "n_folds_run": len(CV_FOLDS),
                      "horizon_months": CV_HORIZON_MONTHS,
                      "min_train_months": CV_MIN_TRAIN_MONTHS},
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
| §12 M4 long-run | "A sustained 1-SD increase in *X* is associated with a long-run change of β/(1−ρ)" | A forecast |
| §13 stability | "The association with *X* is robust to resampling and to collinearity with other drivers" | An effect size — the elastic net shrinks coefficients toward zero by design |
| §14 splines | "The *X*–WH relationship is (non-)linear, with this shape" | A threshold you can manage to, on ~100 months |
| §16 | "The drivers do / do not improve out-of-sample prediction over persistence and season" | Anything about in-sample fit |
| §17 Shapley | "*X* accounts for *n*% of the explained variance in WH extent" | That *X* caused that variance |
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
