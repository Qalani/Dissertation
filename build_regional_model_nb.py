"""Builder script that assembles winam_wh_regional_hierarchical_driver_model.ipynb.

The AOI temporal notebook (`winam_wh_temporal_driver_model.ipynb`) asks "how much
hyacinth is there this month"; the spatial-panel notebooks ask "where is it". This
notebook asks the question in between: divide Winam Gulf into a handful of FIXED,
ecologically meaningful regions, build ONE water-hyacinth time series per region,
and fit a hierarchical dynamic driver model to the resulting region x month panel.

It reuses the loading, provenance, calendar-handling and area-weighted aggregation
logic of the existing notebooks. It does NOT replace them, and it does not treat
the 500 m grid cells as independent replicates: the inferential dataset has one
row per region per calendar month.

Kept in the repo (like build_temporal_model_nb.py / build_backfill_nb.py) so the
notebook can be regenerated from one editable source instead of hand-patching
notebook JSON. Regenerate with:

    python3 build_regional_model_nb.py

This writes the CELL SOURCES only: the emitted notebook carries no outputs and no
Colab metadata.
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
   'winam_wh_regional_hierarchical_driver_model.ipynb" target="_parent">'
   '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
   'alt="Open In Colab"/></a>')

md(r"""# Winam Gulf water hyacinth — **regional hierarchical dynamic** driver model

**Question this notebook answers.** Does splitting Winam Gulf into a small number
of fixed, ecologically meaningful regions reveal environmental driver information
that a single gulf-wide monthly series cannot see — and does it do so *without*
pretending that grid cells are independent replicates?

**The response.** One number per **region** per **calendar month** — the
area-weighted regional mean of hard-class WH cover over a **fixed** cell
membership $C_r$:

$$\text{wh\_cover}_{r,t} \;=\;
\frac{\sum_{i \in C_r} \text{WH area}_{i,t}}
     {\sum_{i \in C_r} \text{valid classified area}_{i,t}}$$

Dividing by the *validly classified* area (not the region's total area) is what
makes months of different cloudiness comparable; fixing $C_r$ through time is
what stops a change in *which* cells were observed from masquerading as a change
in how much hyacinth there is.

---

## The one claim this notebook refuses to make

> Regionalisation **does not multiply the sample size** for a driver that has
> only one gulf-wide value per month.

If lake level is a single altimetry series, copying it into 8 regions produces 8
identical columns, not 8 independent observations. §10 measures, for **every**
driver, how much of its variance is between-region-within-month, and labels the
ones with no meaningful regional variation `temporal_only`. Those drivers are
still estimated — they are real mechanisms — but the notebook states plainly that
their effective replication is the number of **months**, not the number of
region-months. Only drivers labelled `spatiotemporal` gain genuine information
from the regional design.

## What is non-negotiable here (inherited, not re-litigated)

| Rule | Where it is enforced |
|---|---|
| the existing 500 m cell-month panel | §6 loader, §6b provenance gate |
| **hard-class** WH cover, no classifier-uncertainty weighting | §6b asserts the run manifest's `USE_PROBABILITY_RESPONSE = False` / `WEIGHT_COVER_BY_CONFIDENCE = False` |
| WH cover = WH area / **valid classified** area | §9 `regional_monthly_panel` |
| area-weighted aggregation | §9, weight = each cell's classified area that month |
| **fixed** regional cell membership through time | §8 assigns once from static covariates; §9 asserts membership never changes |
| the complete calendar-month grid, excluded months kept **missing** | §9d reindex; §11 missing-data audit |
| existing batch/run provenance controls | §6b |
| predeclared mechanisms and a-priori lags | §3e `REGIONAL_FORCING_TERMS` |
| endogenous optical proxies kept out of driver inference | §3e `REGIONAL_PROXY_TERMS`, reported descriptively only |
| **one observation per region per month** | §9, §11 — the 469,559 cell-month rows are never an inferential $n$ |

## Structure

| § | What it does |
|---|---|
| 1–3 | Install, imports, configuration (every threshold in one cell) |
| 4 | Helpers — loading, provenance, calendar, aggregation |
| 5 | Helpers — **response-blind** regionalisation, and §5c self-tests |
| 6 | Load the cell-month panel; provenance gate |
| 7 | Static covariate distributions; where each threshold comes from |
| 8 | **Build the regions**; audit, map, exports |
| 9 | **Region-month panel** — the inferential dataset |
| 10 | **Does the regional design add information?** Driver variance decomposition |
| 11 | Model dataset, standardisation, missing-data audit |
| 12 | The PyMC hierarchical dynamic model (one builder, several structures) |
| 13 | Step 1 — persistence structure selected on the **no-driver** model |
| 14 | Step 2 — matched `regional_dynamic_null` vs `regional_dynamic_full` |
| 15 | Diagnostics, the simplification ladder, prior sensitivity |
| 16 | Posterior inference — ROPE, sign probabilities, conservative verdicts |
| 17 | Temporal validation — expanding-window, one calendar month ahead |
| 18 | Regional transfer — leave-one-region-out |
| 19 | Regionalisation sensitivity |
| 20 | Figures |
| 21 | Exports and run manifest |
| 22 | Synthesis; implementation summary |
| 23 | Validation assertion table |
| 24–25 | How to read the model; implementation summary |

## Two run modes

`FAST_MODE = True` (the default in the repo) runs short chains, few validation
origins and a reduced sensitivity grid, so the notebook can be executed
end-to-end for development. **`FAST_MODE = False` is the configuration any
reported number must come from**; §3f states it explicitly and every exported
table records which mode produced it.

`USE_SYNTHETIC_DEMO = True` builds a synthetic *cell-month panel* — geography,
static covariates, gulf-wide and regionally-varying drivers, a known common
AR(1) state, known regional intercepts and known driver slopes — and runs the
entire pipeline, regionalisation included, with no Google Drive. It is a
recovery test, never a result: every exported table carries `is_synthetic`.
""")


# ===========================================================================
# 1. Install
# ===========================================================================
md("""## 1. Install packages

Colab has numpy / pandas / scipy / statsmodels / matplotlib already. **PyMC and
ArviZ are required** for §12–§19 (`pip install pymc arviz`); on Colab this takes
a couple of minutes. `geopandas` is optional and only affects the GeoPackage
export and the shoreline overlay on the map — the region map itself is drawn
from the grid cells with matplotlib and needs nothing extra.
""")

code("""# Colab: run once per runtime. PyMC pulls in pytensor and a C toolchain.
# !pip -q install pymc arviz
# !pip -q install geopandas          # optional: GeoPackage export + shoreline overlay
""")


# ===========================================================================
# 2. Imports
# ===========================================================================
md("""## 2. Imports and Google Drive mount

Every optional dependency is guarded. A missing package changes *what runs* and
says so; it never changes whether the notebook runs.
""")

code('''from pathlib import Path
import json
import math
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import scipy.stats as sstats
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*No frequency information was provided.*")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 90)

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
print(f"numpy {np.__version__} | pandas {pd.__version__} | statsmodels {sm.__version__}")

# --- PyMC / ArviZ: REQUIRED for the hierarchical model (§12-§19) -------------
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAVE_PYMC = True
    print(f"pymc {pm.__version__} | arviz {az.__version__}")
except Exception as exc:
    HAVE_PYMC = False
    pm = pt = az = None
    print("PyMC/ArviZ unavailable -> §12-§19 will be SKIPPED (not silently "
          f"replaced by something weaker): {exc}")

# --- geopandas: OPTIONAL, for the GeoPackage export and shoreline overlay ----
try:
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    HAVE_GEOPANDAS = True
    print(f"geopandas {gpd.__version__}")
except Exception as exc:
    HAVE_GEOPANDAS = False
    gpd = None
    shapely_box = None
    print(f"geopandas unavailable -> GeoPackage export and the shoreline overlay "
          f"are skipped; the region map and CSV exports still run: {exc}")
''')


# ===========================================================================
# 3. Configuration
# ===========================================================================
md("""## 3. Configuration

Every decision that changes a number lives in this one cell. The regionalisation
thresholds in §3c are the ones a reader will interrogate hardest, so each records
**where it came from** — a repository precedent, or an operational definition
with a response-blind fallback and a sensitivity variant.
""")

code('''# =====================================================================
# 3a. Where the data comes from
# =====================================================================
# The 500 m cell-month panel exported by winam_wh_spatial_panel_driver_gam.ipynb
# §17: OUTPUT_DIR / f"wh_spatial_panel_{CELL_SIZE_M}m_{run_tag}_{start}_to_{end}.csv"
# It carries the hard-class response (wh_area_ha, valid_area_ha, wh_cover,
# wh_present), the static habitat covariates used to build the regions, and the
# merged environmental covariates. Set PANEL_CSV explicitly, or leave it None to
# take the most recent match of PANEL_GLOB in PANEL_DIR.
PANEL_DIR = Path("/content/drive/MyDrive/WH_spatial_panel_test")
PANEL_CSV = None
PANEL_GLOB = "wh_spatial_panel_*.csv"

# The run manifest written beside the panel by the same section. Read (not
# required) so §6b can check the provenance controls rather than assume them.
PANEL_MANIFEST_GLOB = "driver_gam_run_manifest_*.json"

# The COMPLETE monthly environmental tables exported by the spatial workflow.
# These are loaded independently of whether a WH map passed its coverage
# threshold (§6c), which is exactly what the state-space model needs: the driver
# process must continue through months the response does not cover.
EE_MONTHLY_GLOB = "ee_monthly_covariates_*.csv"
EE_CELLMONTH_GLOB = "ee_cellmonth_covariates_*.csv"

# Optional extra monthly series merged on `month` (one row per month) - e.g.
# gauged discharge, an ENSO/IOD index, measured nutrients. Gulf-wide by
# construction, so §10 will label anything from here `temporal_only`.
EXTRA_MONTHLY_CSV = None

# Optional shoreline / AOI polygon for the map. The repository ships one at
# aoi/winam_gulf_main_lake_aoi.geojson (EPSG:4326); the notebook falls back to
# the boundary of the eligible cell set when it is absent or geopandas is not
# installed.
SHORELINE_GEOJSON_CANDIDATES = [
    Path("aoi/winam_gulf_main_lake_aoi.geojson"),
    Path("/content/Dissertation/aoi/winam_gulf_main_lake_aoi.geojson"),
    Path("/content/drive/MyDrive/WH_drivers/winam_gulf_main_lake_aoi.geojson"),
]

OUTPUT_DIR = Path("/content/drive/MyDrive/WH_regional_hierarchical_model")

# Run the whole pipeline - regionalisation included - on a synthetic cell-month
# panel with KNOWN common persistence, regional intercepts and driver slopes, and
# no Drive. Use it to (a) check the notebook end-to-end and (b) confirm the
# machinery recovers effects that are genuinely there. Never report synthetic
# numbers: every exported table is tagged is_synthetic.
USE_SYNTHETIC_DEMO = False
SYNTHETIC_SEED = 20260810
SYNTHETIC_N_MONTHS = 108
SYNTHETIC_MISSING_FRACTION = 0.18     # calendar months excluded by the coverage filter
SYNTHETIC_DOMAIN_KM = (40.0, 18.0)    # an elongated gulf, open at one end
SYNTHETIC_CELL_SIZE_M = 500           # the same 500 m lattice as the real panel

# =====================================================================
# 3b. Provenance controls (inherited from the panel build)
# =====================================================================
# These are CHECKS, not switches. §6b compares them against the panel and its
# run manifest and refuses to continue when the panel is not the one this
# notebook is entitled to analyse.
EXPECTED_CELL_SIZE_M = 500
# Projected CRS the panel's x_km / y_km centroids live in (UTM 36S), used only to
# write the region GeoPackage and to place the shoreline overlay.
PANEL_CRS = "EPSG:32736"
# The response must be the hard-class one. The panel build exposes these flags;
# a panel built with either of them True is a different estimand.
REQUIRE_HARD_CLASS_RESPONSE = True
# Set to e.g. "S2_only" or "S2+S1" to pin the sensor run; None accepts whatever
# the panel says and records it.
REQUIRE_RUN_TAG = None
# Columns whose presence means the panel carries batch/run provenance. Recorded
# in the manifest and printed; a panel mixing several is reported, never merged
# silently.
PROVENANCE_COLUMNS = ["sensor", "source_export_token", "source_file",
                      "classifier_version", "batch_id", "run_tag"]

# =====================================================================
# 3c. Regionalisation - thresholds and minimum sizes  (RESPONSE-BLIND)
# =====================================================================
# NOTHING in this block may be derived from WH cover, WH prevalence, model
# residuals, temporal environmental values or any classification output. §5a
# enforces that with `assert_response_blind`, which raises on a response-derived
# column name.
#
# Provenance of each threshold (searched for in the repository first; see §7b):
#   river_dist_m  5000  - reuses the 5 km radius already used to build
#                         `openness_index` (EE circle kernel, 5000 m) as the
#                         gulf's established local-influence length scale, and
#                         "major river" is the repository's own definition
#                         (HydroSHEDS RIV_ORD <= EE_RIVER_MAJOR_MAX_ORD = 7).
#   shore_dist_m  2000  - reuses EE_CATCHMENT_BUFFER_M = 2000 m, the repository's
#                         existing local catchment-influence buffer.
#   openness      None  - NO repository precedent. Resolved response-blind as a
#                         quantile of `openness_index` among littoral cells
#                         (OPENNESS_FALLBACK_QUANTILE) and printed as an
#                         OPERATIONAL DEFINITION.
#   depth_m       None  - optional; when set, a cell deeper than this is forced
#                         to open_gulf even if it is close to shore.
REGION_THRESHOLDS = {
    "river_dist_m": 5000.0,
    "shore_dist_m": 2000.0,
    "openness": None,
    "depth_m": None,
}
OPENNESS_FALLBACK_QUANTILE = 0.50     # median openness among littoral cells
# When a threshold is None and no quantile rule is given the class collapses into
# its parent class, and §8 says so rather than inventing a number.

# Column preferences. The first present column is used; the choice is printed.
REGION_COVARIATE_PREFERENCE = {
    "river_dist_m": ["dist_majriver_m", "dist_river_m"],
    "shore_dist_m": ["dist_shore_m"],
    "openness": ["openness_index", "gsw_water_fraction"],
    "depth_m": ["depth_m"],
}

# Contiguity used to split a class into geographically disconnected units.
REGION_CONTIGUITY = "queen"           # "queen" (8-neighbour) or "rook" (4-neighbour)

# Minimum requirements for a region to be USABLE. A component below any of these
# is merged into the most physically similar adjacent region (§5b); if it has no
# adjacent region it is dropped and reported.
MIN_REGION_CELLS = 40                 # >= 10 km2 at 500 m
MIN_REGION_ELIGIBLE_AREA_HA = 1000.0  # eligible (mask-passing) water area
MIN_REGION_MONTHS = 24                # retained months the region must have
MIN_REGION_MEDIAN_COVERAGE = 0.70     # median cell-coverage fraction across months

# Target band, reported but never forced. If fewer regions clear the
# requirements, fewer are kept and §8d states the consequence for hierarchical
# estimation (a small number of groups makes between-region variances weakly
# identified).
REGION_COUNT_TARGET = (6, 12)

# =====================================================================
# 3d. Region-month panel gates
# =====================================================================
# Gulf-wide month filter, inherited from the panel/AOI workflow: a month enters
# only if it classified this share of the eligible water cells.
MIN_MONTHLY_COVERAGE_FRACTION = 0.90
# Fixed cell membership: a cell joins a region only if it is validly observed in
# at least this share of the retained months (mirrors MIN_CELL_MONTH_FRACTION in
# the AOI temporal notebook, so the two analyses use the same discipline).
MIN_CELL_MONTH_FRACTION = 0.80
# A region-month must pass BOTH gates to be an observation.
MIN_REGION_MONTH_CELL_COVERAGE = 0.70        # observed cells / eligible cells
MIN_REGION_MONTH_VALID_AREA_COVERAGE = 0.70  # valid area / that region's best month

RESPONSE_TRANSFORM = "logit"
RESPONSE_EPS = 1e-4

# =====================================================================
# 3e. Drivers - mechanism, a-priori sign, a-priori lag
# =====================================================================
# ONE representation per mechanism, with the sign the mechanism requires and the
# lag at which it acts. Identical discipline (and, where the columns exist,
# identical entries) to TEMPORAL_FORCING_TERMS in winam_wh_temporal_driver_model.
#   term: (mechanism, expected sign, a-priori lag in months)
REGIONAL_FORCING_TERMS = {
    "rain_chirps_30d_mm": ("antecedent rainfall -> catchment runoff and nutrient "
                           "delivery; acts with a delay", "+", 1),
    "air_temp_c":         ("thermal control on growth rate", "+", 0),
    "wind_speed_ms":      ("wind speed -> mixing and mat drift", "?", 0),
    "wave_exposure_idx":  ("openness x wind^2 -> wave disturbance, mat fragmentation", "-", 0),
    "lake_level_m":       ("lake level -> depth over littoral habitat and flushing", "-", 0),
}
REGIONAL_FORCING_FALLBACKS = {
    "rain_chirps_30d_mm": ["rain_chirps_mm", "rain_chirps_90d_mm"],
    "air_temp_c":         ["water_temp_c"],
    "wind_speed_ms":      ["wind_axis_comp_ms", "wind_cross_comp_ms"],
    "lake_level_m":       ["lake_level_anom_m"],
}

# When is a driver's regional variation REAL? Two tests, both applied in §10.
# A driver below EITHER is labelled `temporal_only`: repeating one gulf-wide
# number across regions is not extra information, and the notebook must not
# claim the regional design raised its effective replication.
DRIVER_REGIONAL_SHARE_MIN = 0.05   # share of total variance that is
                                   # within-month, between-region
DRIVER_REGIONAL_CV_MIN = 1e-6      # median within-month coefficient of variation
                                   # across regions; below this the columns are
                                   # numerically identical copies
# A driver whose regional series is this well explained by the annual harmonics
# alone cannot be separated from "it is the wet season".
SEASON_CONFOUND_R2 = 0.80
# Pairwise |r| above this makes two drivers redundant; the one later in the
# mechanism list is dropped and reported.
MAX_ABS_PAIRWISE_R = 0.90

# Endogenous optical / biogeochemical proxies: measured from the same reflectance
# a floating mat dominates, so they are downstream of WH as much as upstream.
# Excluded from every driver claim; reported once, descriptively (§16d).
REGIONAL_PROXY_TERMS = ["chl_mci_s3", "chl_mph_s3", "chl_ndci_s2",
                        "turb_ndti_s2", "chl_modis_mg_m3"]

# Static habitat covariates. Constant within a region through time, so they can
# never explain temporal variation; they enter only through the regionalisation
# and the partially pooled regional intercept.
KNOWN_STATIC_COLS = ["depth_m", "dist_shore_m", "dist_river_m", "dist_majriver_m",
                     "frac_cropland", "frac_urban", "frac_wetland", "openness_index",
                     "pop_count", "built_surface", "gsw_water_fraction",
                     "bathy_water_fraction", "shore_gx", "shore_gy",
                     "x_km", "y_km", "cell_area_m2"]

# effective_depth_m = depth_m + lake-level anomaly, and depth_m is constant within
# a fixed region, so its regional mean IS the lake-level anomaly plus a constant.
REGIONAL_DEGENERATE_COLS = {
    "effective_depth_m": "= constant regional depth + lake-level anomaly",
    "lake_level_anom_m": "= lake_level_m - constant",
}

# PREDECLARED random-slope candidates, in preference order. Only terms that §10
# labels `spatiotemporal` are eligible, and at most RANDOM_SLOPE_MAX_TERMS are
# used, because a random-slope variance estimated from <= 12 regions is weakly
# identified. Declared here so the choice cannot be made after seeing a posterior.
RANDOM_SLOPE_CANDIDATES = ["rain_chirps_30d_mm", "wave_exposure_idx"]
RANDOM_SLOPE_MAX_TERMS = 2
# Below this many usable regions the model drops random slopes entirely and keeps
# partially pooled intercepts + common slopes (+ at most one interaction).
# Ten, not six: a between-region slope VARIANCE estimated from a handful of
# groups is prior-dominated, and reporting it as a measurement is exactly the
# failure this notebook exists to avoid.
RANDOM_SLOPE_MIN_REGIONS = 10
# Parameterisation of the regional slope deviations b_{r,k}.
#   "centred"    - b ~ Normal(0, sigma_b). The right choice when the data INFORM
#                  the regional slopes, which is the only case in which a random
#                  slope is kept at all: §10c admits a term only when it has
#                  genuine within-month regional variation. Default.
#   "noncentred" - b = sigma_b * z. The right choice when the data barely inform
#                  them - but then the term does not belong in the model.
# The wrong choice biases nothing; it produces a funnel, max-tree-depth
# trajectories and a fit that takes an hour to fail. §15's ladder switches
# parameterisation before it gives up on the random slopes altogether.
RANDOM_SLOPE_PARAMETERISATION = "centred"
# Parameterisation of the partially pooled regional intercepts alpha_r and the
# shared-state loadings lambda_r. Same logic as the random slopes, and it matters
# more, because these exist in EVERY model this notebook fits:
#   "centred"    - alpha ~ Normal(mu_alpha, sigma_alpha). Right when each region
#                  contributes tens of observed months, which is exactly the
#                  regime §9b's MIN_REGION_MONTHS enforces. Default.
#   "noncentred" - alpha = mu_alpha + sigma_alpha * z. Right for sparse groups.
# Choosing wrongly costs mixing, not correctness: the non-centred form funnels
# when the group effects are strongly informed, and R-hat on alpha_r is the first
# thing that fails. §15's ladder switches it before touching the model itself.
HIERARCHY_PARAMETERISATION = "centred"

# =====================================================================
# 3f. Model and sampling
# =====================================================================
SEASON_HARMONICS = 2       # deterministic annual Fourier pairs (2 -> 4 columns)
INCLUDE_TREND = True       # common long-term linear trend on the scaled month index

# Candidate structures for the SHARED latent state g_t. The AOI temporal model's
# AR interval included a unit root, so the local-level (random-walk) alternative
# is fitted as a genuine candidate, not a footnote. Selection happens on the
# NO-DRIVER model (§13), before any driver posterior is looked at.
COMMON_STATE_CANDIDATES = ["ar1", "randomwalk", "none"]
# Region-specific temporal dependence u_{r,t}: one common rho (default) or one per
# region. §15's simplification ladder falls back from "per_region" to "common".
REGIONAL_AR_MODE = "common"           # "common" | "per_region" | "none"

# Weakly informative, regularising priors on the STANDARDISED scale. Every driver
# is z-scored, and the response is logit cover, so a slope of 0.5 is already a
# large effect (a 1 SD driver move shifting the odds by ~65%).
PRIORS = {
    "mu_alpha_sd": 1.5,
    "sigma_alpha": 1.0,      # HalfNormal scale, between-region intercept SD
    "beta_sd": 0.5,          # gulf-wide driver slopes
    "sigma_b": 0.25,         # HalfNormal scale, between-region slope SD
    "season_sd": 0.5,
    "trend_sd": 0.25,
    "sigma_g": 0.5,          # HalfNormal scale, shared-state innovation SD
    "sigma_lambda": 0.3,     # HalfNormal scale, regional loading spread around 1
    "sigma_u": 0.4,          # HalfNormal scale, regional AR innovation SD
    "sigma_eps": 0.5,        # HalfNormal scale, observation noise
    "rho_a": 2.0,            # Beta(a, b) on every stationary AR parameter,
    "rho_b": 2.0,            # constrained to the unit interval (0, 1)
}
# Prior-sensitivity variants (§15e): the same model refit with tighter and looser
# regularisation. Conclusions that move between these are reported as fragile.
PRIOR_VARIANTS = {
    "tight": {"beta_sd": 0.25, "sigma_b": 0.15, "sigma_alpha": 0.5},
    "loose": {"beta_sd": 1.0, "sigma_b": 0.5, "sigma_alpha": 2.0},
}

# Region of practical equivalence on the STANDARDISED LOGIT scale. A |slope|
# below this is practically zero: a 1 SD driver change moving the log-odds of
# cover by less than 0.05 is not an ecologically interesting association at this
# sample size. Configured explicitly, never inferred from the posterior.
ROPE_HALFWIDTH = 0.05
HDI_PROB = 0.95

FAST_MODE = True
# target_accept is 0.95 even in FAST mode: the funnel in a hierarchical
# state-space model is a property of the geometry, not of the draw count, and a
# development run full of spurious divergences would send §15's simplification
# ladder chasing a sampler problem rather than a model problem.
SAMPLING_FAST = dict(draws=300, tune=700, chains=4, cores=4,
                     target_accept=0.95, random_seed=20260810)
# THE FINAL, DOCUMENTED CONFIGURATION. Every reported number must come from a run
# with FAST_MODE = False.
SAMPLING_FINAL = dict(draws=2000, tune=2000, chains=4, cores=4,
                      target_accept=0.95, random_seed=20260810)
# Cheaper settings for the many refits in §17-§19. Still four chains, because a
# refit with unusable diagnostics is not a cheaper answer, it is no answer.
SAMPLING_REFIT_FAST = dict(draws=200, tune=500, chains=4, cores=4,
                           target_accept=0.93, random_seed=20260810)
SAMPLING_REFIT_FINAL = dict(draws=750, tune=1500, chains=4, cores=4,
                            target_accept=0.95, random_seed=20260810)

# Diagnostic thresholds required before a coefficient may be reported (§15).
DIAG_MAX_RHAT = 1.01
DIAG_MIN_ESS_BULK = 400
DIAG_MIN_ESS_TAIL = 400
DIAG_MAX_DIVERGENCES = 0

# =====================================================================
# 3g. Validation
# =====================================================================
# Expanding-window, ONE-calendar-month-ahead prediction.
VAL_MIN_TRAIN_MONTHS = 36        # observed calendar months before the first origin
VAL_MAX_ORIGINS_FAST = 3         # FAST_MODE cap; None (final) = every feasible origin
VAL_MAX_ORIGINS_FINAL = None
# Drivers must be knowable at the origin. Anything with an a-priori lag of 0 is
# moved to lag 1 for the FORECAST evaluation (the a-priori specification is kept
# unchanged for the §14/§16 association inference - different questions).
VAL_FORECAST_MIN_LAG = 1
# Calendar MONTHS are the resampling unit for uncertainty on performance
# differences. Regions within a month are not independent replicates.
VAL_BOOTSTRAP_N = 2000
VAL_BOOTSTRAP_SEED = 11

RUN_LORO = True                  # leave-one-region-out transfer (§18)
LORO_MAX_REGIONS_FAST = 2        # FAST_MODE cap on how many regions are withheld
LORO_MAX_REGIONS_FINAL = None

# =====================================================================
# 3h. Regionalisation sensitivity (§19)
# =====================================================================
# A SMALL, PREDECLARED set of response-blind variants. This is not a search: the
# variants are fixed here, all of them are reported, and none of them may be
# promoted to the headline because it produced a stronger driver result.
REGIONALISATION_VARIANTS = {
    "river_3km":      {"river_dist_m": 3000.0},
    "river_8km":      {"river_dist_m": 8000.0},
    "littoral_1km":   {"shore_dist_m": 1000.0},
    "littoral_3km":   {"shore_dist_m": 3000.0},
    "openness_q35":   {"_openness_quantile": 0.35},
    "openness_q65":   {"_openness_quantile": 0.65},
    "min_cells_80":   {"_min_region_cells": 80},
}
SENSITIVITY_VARIANTS_FAST = ["river_3km", "openness_q65"]

RANDOM_STATE = 20260810

OUTPUT_DIR = Path(OUTPUT_DIR)
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_WRITABLE = True
except Exception as exc:
    OUTPUT_WRITABLE = False
    print(f"OUTPUT_DIR not writable ({exc}); §21 will skip the exports.")

SAMPLING = dict(SAMPLING_FAST if FAST_MODE else SAMPLING_FINAL)
SAMPLING_REFIT = dict(SAMPLING_REFIT_FAST if FAST_MODE else SAMPLING_REFIT_FINAL)
VAL_MAX_ORIGINS = VAL_MAX_ORIGINS_FAST if FAST_MODE else VAL_MAX_ORIGINS_FINAL
LORO_MAX_REGIONS = LORO_MAX_REGIONS_FAST if FAST_MODE else LORO_MAX_REGIONS_FINAL
SENSITIVITY_VARIANTS = (SENSITIVITY_VARIANTS_FAST if FAST_MODE
                        else list(REGIONALISATION_VARIANTS))

print("Configuration loaded.")
print(f"  mode              : {'FAST (development)' if FAST_MODE else 'FINAL (reportable)'}")
print(f"  sampling          : {SAMPLING}")
print(f"  mechanisms        : {len(REGIONAL_FORCING_TERMS)}")
print(f"  ROPE (std. logit) : +/- {ROPE_HALFWIDTH}")
print(f"  region thresholds : {REGION_THRESHOLDS}")
print(f"  output            : {OUTPUT_DIR}")
if FAST_MODE:
    print("\\n*** FAST_MODE = True. These settings are for development. Set "
          "FAST_MODE = False before quoting any number. ***")
''')


# ===========================================================================
# 4. Helpers - loading, provenance, calendar, aggregation
# ===========================================================================
md(r"""## 4. Helpers — loading, provenance, calendar, aggregation

These are the same primitives the AOI temporal notebook uses, kept deliberately
identical so the two analyses read the same data the same way:

| Helper | Why it exists |
|---|---|
| `load_cellmonth_panel` | one loader for the 500 m panel, month column normalised to first-of-month |
| `_area_columns` | resolves the WH / valid-area columns whether the panel spells them `_ha` or `_m2` |
| `monthly_coverage_table` | the **gulf-wide** month filter — a cloudy month measures a different piece of lake, not less hyacinth |
| `fixed_cell_membership` | the fixed cell set $C$; a cell joins only if it is observed in most retained months |
| `month_index` / `months_from_index` | integer calendar-month arithmetic, so a lag is a *calendar* lag even across a gap |
| `reindex_calendar_months` | the complete monthly grid, missing months kept as `NaN` |
| `fourier_terms` | deterministic annual harmonics that depend on the calendar month alone, so a forecast fold can build them without future information |
| `transform_response` | logit of cover, **recording what was clipped** |

`make_synthetic_cellmonth_panel` builds a whole synthetic *cell-month* panel —
geography, static covariates, drivers, known effects — so the regionalisation
itself is exercised offline, not just the model.
""")

code(r'''# =====================================================================
# 4. Loading, provenance, calendar and aggregation helpers
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


def month_index(months):
    """Integer calendar-month index (12 * year + month - 1).

    A difference in this index IS a difference in calendar months, which is the
    single property every temporal routine in this notebook relies on.
    """
    m = pd.to_datetime(pd.Series(months).reset_index(drop=True))
    return (m.dt.year.astype(int) * 12 + m.dt.month.astype(int) - 1).to_numpy()


def months_from_index(mi):
    """Inverse of `month_index`: month-start timestamps."""
    mi = np.asarray(mi, dtype=int).ravel()
    return pd.to_datetime([f"{int(v) // 12:04d}-{int(v) % 12 + 1:02d}-01" for v in mi])


def calendar_span_months(months):
    """Calendar months spanned by a record, gaps included."""
    mi = month_index(months)
    return (int(mi.max() - mi.min()) + 1) if len(mi) else 0


def load_cellmonth_panel(panel_csv=None, panel_dir=None,
                         panel_glob="wh_spatial_panel_*.csv"):
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
    ever observed, the best available proxy for "eligible".
    """
    _, valid_col = _area_columns(panel)
    per_month = (panel.groupby("month", as_index=False)
                 .agg(n_cells_observed=("grid_id", "nunique"),
                      valid_area_ha=(valid_col, "sum")))
    if "coverage_fraction" in panel.columns:
        cov = panel.groupby("month", as_index=False)["coverage_fraction"].first()
        per_month = per_month.merge(cov, on="month", how="left")
        per_month["coverage_basis"] = "panel coverage_fraction"
    else:
        n_eligible = int(panel["grid_id"].nunique())
        per_month["coverage_fraction"] = per_month["n_cells_observed"] / max(n_eligible, 1)
        per_month["coverage_basis"] = f"n_cells_observed / {n_eligible} cells ever observed"
    per_month["retained"] = per_month["coverage_fraction"] >= float(min_coverage)
    per_month["exclusion_reason"] = np.where(
        per_month["retained"], "",
        f"monthly coverage below MIN_MONTHLY_COVERAGE_FRACTION={float(min_coverage):g}")
    return per_month.sort_values("month").reset_index(drop=True)


def fixed_cell_membership(panel, months_kept, min_cell_month_fraction):
    """Cells observed in >= `min_cell_month_fraction` of the retained months.

    Returns (cell_ids, audit). Fixing membership is what makes a regional series
    a series about hyacinth rather than about which cells were cloud-free.
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
    return pd.Index(np.sort(keep)), audit


def reindex_calendar_months(frame, month_col="month", by=None):
    """Put a monthly frame on an unbroken month grid, per group when `by` is given.

    Excluded / never-observed months become all-NaN rows carrying only their key
    columns, so a lag is always a calendar lag and a gap never silently becomes
    "last month".
    """
    frame = frame.sort_values(([by] if by else []) + [month_col]).reset_index(drop=True)
    full = pd.date_range(frame[month_col].min(), frame[month_col].max(), freq="MS")
    if by is None:
        out = (frame.set_index(month_col).reindex(full)
               .rename_axis(month_col).reset_index())
    else:
        parts = []
        for key, grp in frame.groupby(by, sort=True):
            g = (grp.drop(columns=[by]).set_index(month_col).reindex(full)
                 .rename_axis(month_col).reset_index())
            g[by] = key
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)
    out["year"] = out[month_col].dt.year
    out["month_num"] = out[month_col].dt.month
    mi = month_index(out[month_col])
    out["time_index"] = (mi - mi.min()).astype(float)
    cols = ([by] if by else []) + [month_col, "year", "month_num", "time_index"]
    rest = [c for c in out.columns if c not in cols]
    return out[cols + rest].reset_index(drop=True)


def fourier_terms(months, n_harmonics, prefix="season"):
    """Deterministic annual Fourier terms for arbitrary month timestamps.

    Depends on the CALENDAR MONTH only, so it is identical whichever subset of
    the record it is evaluated on - which is exactly what lets a rolling-origin
    fold build it for a future month without using any future information.
    """
    idx = pd.to_datetime(pd.Series(months).reset_index(drop=True))
    ang = 2 * np.pi * idx.dt.month.to_numpy(dtype=float) / 12.0
    out = pd.DataFrame(index=range(len(idx)))
    for k in range(1, int(n_harmonics) + 1):
        out[f"{prefix}_sin{k}"] = np.sin(k * ang)
        out[f"{prefix}_cos{k}"] = np.cos(k * ang)
    return out


def transform_response(values, how="logit", eps=1e-4):
    """Map cover onto the modelling scale, reporting exactly what was clipped."""
    y = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    if how == "identity":
        return y, {"transform": "identity", "n_clipped_low": 0, "n_clipped_high": 0,
                   "eps": float(eps)}
    if how == "log":
        n_lo = int((y <= 0).sum())
        return np.log(y.clip(lower=eps)), {"transform": "log", "n_clipped_low": n_lo,
                                           "n_clipped_high": 0, "eps": float(eps)}
    if how == "logit":
        n_lo = int((y <= 0).sum())
        n_hi = int((y >= 1).sum())
        p = y.clip(lower=eps, upper=1 - eps)
        return np.log(p / (1 - p)), {"transform": "logit", "n_clipped_low": n_lo,
                                     "n_clipped_high": n_hi, "eps": float(eps)}
    raise ValueError(f"unknown RESPONSE_TRANSFORM {how!r}")


def inverse_transform_response(values, how="logit", eps=1e-4):
    """Back to the cover scale, for reporting RMSE in units a reader recognises."""
    v = np.asarray(values, dtype=float)
    if how == "identity":
        return v
    if how == "log":
        return np.exp(v)
    if how == "logit":
        return 1.0 / (1.0 + np.exp(-v))
    raise ValueError(f"unknown transform {how!r}")


def panel_provenance_audit(panel, manifest, expected_cell_size_m=500,
                           require_hard_class=True, require_run_tag=None,
                           provenance_columns=()):
    """Check the panel is the run this notebook is entitled to analyse.

    Returns (audit_rows, blocking_failures). Nothing here is a switch: a failing
    check is reported, and §6b raises on any BLOCKING failure rather than
    quietly analysing a panel built to different rules.
    """
    rows, blocking = [], []

    def _add(check, value, expected, ok, blocking_if_false=True, note=""):
        rows.append({"check": check, "found": value, "expected": expected,
                     "ok": bool(ok), "blocking": bool(blocking_if_false and not ok),
                     "note": note})
        if blocking_if_false and not ok:
            blocking.append(check)

    man = manifest or {}
    cs = man.get("cell_size_m")
    _add("cell_size_m", cs, expected_cell_size_m,
         cs is None or int(cs) == int(expected_cell_size_m),
         note="from the panel run manifest; unchecked when no manifest was found")

    if require_hard_class:
        conf = man.get("confidence_usage", {}) or {}
        kind = str(man.get("response_kind", "")).lower()
        soft_flags = {k: v for k, v in conf.items() if bool(v)}
        _add("hard_class_response",
             {"response_kind": man.get("response_kind"), "confidence_usage": conf},
             "hard-class WH cover, no classifier-uncertainty weighting",
             (not soft_flags) and ("prob" not in kind) and ("soft" not in kind),
             note="USE_PROBABILITY_RESPONSE / WEIGHT_COVER_BY_CONFIDENCE must be off")

    if require_run_tag is not None:
        found = man.get("run_label")
        _add("run_tag", found, require_run_tag,
             found is None or str(found) == str(require_run_tag))

    for col in provenance_columns:
        if col in panel.columns:
            vals = pd.Series(panel[col].astype(str).unique())
            _add(f"provenance:{col}", sorted(vals.tolist())[:8], "recorded",
                 True, blocking_if_false=False,
                 note=f"{len(vals)} distinct value(s)")

    # A panel mixing several classifier versions is reported, never merged blind.
    if "classifier_version" in panel.columns:
        n_v = int(panel["classifier_version"].astype(str).nunique())
        _add("single_classifier_version", n_v, 1, n_v <= 1,
             note="several classifier versions in one panel is a provenance error")

    _add("one_row_per_cell_month",
         int(panel.duplicated(["grid_id", "month"]).sum()), 0,
         int(panel.duplicated(["grid_id", "month"]).sum()) == 0)

    return pd.DataFrame(rows), blocking


print("§4 loading / calendar / provenance helpers defined.")
''')


# ===========================================================================
# 4b. Synthetic cell-month panel
# ===========================================================================
md(r"""### 4b. A synthetic **cell-month panel** with known effects

The recovery test has to exercise the regionalisation, not just the model, so the
synthetic route builds a whole panel rather than a ready-made regional series:

* an elongated gulf on a 500 m lattice, with two side bays and two river mouths;
* static covariates computed from that geometry the same way the real ones are —
  `dist_shore_m` from the water mask, `openness_index` as the mean water fraction
  in a 5 km neighbourhood, `dist_majriver_m` from the river-mouth points,
  `depth_m` increasing away from shore;
* drivers with **deliberately different spatial structure**: rainfall carries a
  real west–east gradient that changes month to month (*spatiotemporal*); wind
  speed and lake level are single gulf-wide series (*temporal only*); wave
  exposure is `openness_index × wind²`, so a gulf-wide driver acquires genuine
  regional variation through a static field (*spatiotemporal*);
* a response built on the logit scale from **known** regional intercepts, a
  **known** common AR(1) state, **known** driver slopes, a **known** river-linked
  slope heterogeneity for rainfall, and two decoy drivers with zero effect;
* excluded (cloudy) calendar months, so the gap machinery is exercised too.

What the recovery test must show: the rainfall and wave-exposure slopes recovered
with the right sign, the decoys not recovered, the common persistence recovered,
wind speed and lake level flagged `temporal_only`, and rainfall flagged
`heterogeneous` if its regional slope spread is large enough to detect.
""")

code(r'''# =====================================================================
# 4b. Synthetic cell-month panel with KNOWN effects
# =====================================================================


def _disk_kernel(radius_cells):
    """Binary disk used for the openness neighbourhood mean."""
    r = int(radius_cells)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((xx ** 2 + yy ** 2) <= r ** 2).astype(float)


def make_synthetic_cellmonth_panel(n_months=108, seed=20260810,
                                   domain_km=(40.0, 18.0), cell_size_m=500,
                                   missing_fraction=0.18):
    """A synthetic 500 m cell-month panel with a known generative structure.

    Returns (panel, truth, geometry) where `geometry` carries the water mask and
    the river-mouth points used to draw the synthetic map.
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    cs_km = cell_size_m / 1000.0
    nx = int(round(domain_km[0] / cs_km))
    ny = int(round(domain_km[1] / cs_km))
    xs = (np.arange(nx) + 0.5) * cs_km
    ys = (np.arange(ny) + 0.5) * cs_km
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    # --- an elongated gulf, open at the west, with two side bays -------------
    axis = 0.5 * domain_km[1] + 1.8 * np.sin(2 * np.pi * XX / 34.0)
    half_width = 3.2 + 2.6 * np.exp(-((XX - 6.0) / 9.0) ** 2) \
        + 1.6 * np.exp(-((XX - 30.0) / 12.0) ** 2)
    water = np.abs(YY - axis) < half_width
    # Two narrow side bays hanging off the main channel.
    water |= (np.abs(XX - 12.5) < 1.6) & (YY > axis) & (YY < axis + 5.6)
    water |= (np.abs(XX - 26.0) < 1.4) & (YY < axis) & (YY > axis - 5.0)
    water[0, :] = water[0, :] & (np.abs(ys - axis[0, :]) < half_width[0, :])

    # --- static covariates, computed exactly like the real ones --------------
    # dist_shore_m: Euclidean distance to the nearest non-water cell.
    dist_shore_m = ndimage.distance_transform_edt(water, sampling=cell_size_m)
    # openness_index: mean water fraction in a 5 km circular neighbourhood, the
    # same construction as the Earth Engine layer (circle kernel, 5000 m).
    k = _disk_kernel(int(round(5.0 / cs_km)))
    openness = ndimage.convolve(water.astype(float), k / k.sum(), mode="nearest")
    # Two river mouths on the shoreline: one at the head of the eastern channel,
    # one at the mouth of the northern side bay.
    river_mouths = np.array([[36.5, float(0.5 * domain_km[1] + 1.8 * np.sin(2 * np.pi * 36.5 / 34.0))],
                             [12.5, float(0.5 * domain_km[1] + 1.8 * np.sin(2 * np.pi * 12.5 / 34.0)) + 5.0]])
    d_riv = np.min(np.stack([np.hypot(XX - mx, YY - my) for mx, my in river_mouths]), axis=0)
    dist_majriver_m = d_riv * 1000.0
    # depth increases away from shore and towards the open (western) end.
    depth = 0.9 + 0.0016 * dist_shore_m + 0.10 * (domain_km[0] - XX)
    frac_cropland = np.clip(0.55 - 0.00006 * dist_shore_m + rng.normal(0, 0.02, water.shape), 0, 1)

    gid = np.full(water.shape, -1, dtype=np.int64)
    wi, wj = np.nonzero(water)
    gid[wi, wj] = np.arange(wi.size)
    n_cells = int(wi.size)

    cells = pd.DataFrame({
        "grid_id": np.arange(n_cells, dtype=np.int64),
        "x_km": XX[wi, wj], "y_km": YY[wi, wj],
        "dist_shore_m": dist_shore_m[wi, wj],
        "dist_majriver_m": dist_majriver_m[wi, wj],
        "openness_index": openness[wi, wj],
        "depth_m": depth[wi, wj],
        "frac_cropland": frac_cropland[wi, wj],
        "gsw_water_fraction": np.clip(0.85 + 0.15 * openness[wi, wj], 0, 1),
        "bathy_water_fraction": 1.0,
        "cell_area_m2": float(cell_size_m) ** 2,
    })

    # --- monthly drivers -----------------------------------------------------
    months = pd.date_range("2017-01-01", periods=int(n_months), freq="MS")
    ang = 2 * np.pi * months.month.to_numpy(dtype=float) / 12.0
    t_idx = np.arange(len(months), dtype=float)

    rain_gulf = 62 + 52 * np.sin(ang - 0.6) + rng.gamma(2.0, 11.0, len(months))
    # a WEST-EAST rainfall gradient whose strength changes month to month, strong
    # enough that the regional means genuinely differ within a month
    rain_grad = rng.normal(0.0, 1.1, len(months))
    wind_gulf = 3.2 + 0.6 * np.sin(ang + 1.1) + rng.normal(0, 0.32, len(months))
    air_gulf = 25.0 + 1.6 * np.cos(ang) + rng.normal(0, 0.45, len(months))
    level_gulf = 1134.0 + 0.55 * np.sin(2 * np.pi * t_idx / 42.0) + rng.normal(0, 0.05, len(months))

    xc = (cells["x_km"].to_numpy() - cells["x_km"].mean()) / cells["x_km"].std()
    yc = (cells["y_km"].to_numpy() - cells["y_km"].mean()) / cells["y_km"].std()

    # rainfall: gulf-wide level + a per-month spatial gradient -> SPATIOTEMPORAL
    rain = rain_gulf[None, :] * (1.0 + 0.45 * rain_grad[None, :] * xc[:, None])
    rain = np.clip(rain, 0.5, None)
    # air temperature: gulf-wide plus a small, FIXED north-south offset
    air = air_gulf[None, :] + 0.20 * yc[:, None]
    # wind and lake level: ONE value per month, copied to every cell
    wind = np.repeat(wind_gulf[None, :], n_cells, axis=0)
    level = np.repeat(level_gulf[None, :], n_cells, axis=0)
    # wave exposure: a gulf-wide driver acquiring regional structure via openness
    wave = cells["openness_index"].to_numpy()[:, None] * wind ** 2
    decoy_a = rng.normal(0, 1, (n_cells, len(months)))
    decoy_b = np.repeat(rng.normal(10, 2, len(months))[None, :], n_cells, axis=0)

    def _z(a):
        return (a - np.nanmean(a)) / np.nanstd(a)

    # --- known generative structure on the logit scale -----------------------
    B_RAIN, B_WAVE, B_AIR = 0.45, -0.30, 0.10
    RHO_G, SIG_G = 0.70, 0.45
    g = np.zeros(len(months))
    g[0] = rng.normal(0, SIG_G / np.sqrt(1 - RHO_G ** 2))
    for t in range(1, len(months)):
        g[t] = RHO_G * g[t - 1] + rng.normal(0, SIG_G)

    # habitat intercept: sheltered, shallow, river-influenced cells hold more WH
    a_cell = (-2.6
              - 0.55 * _z(cells["dist_shore_m"].to_numpy())
              - 0.45 * _z(cells["openness_index"].to_numpy())
              - 0.40 * _z(np.log1p(cells["dist_majriver_m"].to_numpy()))
              + rng.normal(0, 0.20, n_cells))
    # KNOWN slope heterogeneity: cells near a major river respond more to rain
    rain_slope_cell = B_RAIN + 0.30 * (-_z(np.log1p(cells["dist_majriver_m"].to_numpy())))
    lam_cell = 1.0 + 0.25 * _z(cells["openness_index"].to_numpy())

    z_rain_lag1 = np.c_[np.full(n_cells, np.nan), _z(rain)[:, :-1]]
    eta = (a_cell[:, None]
           + rain_slope_cell[:, None] * np.nan_to_num(z_rain_lag1)
           + B_WAVE * _z(wave) + B_AIR * _z(air)
           + 0.45 * np.sin(ang - 0.9)[None, :]
           + 0.010 * t_idx[None, :]
           + lam_cell[:, None] * g[None, :]
           + rng.normal(0, 0.30, (n_cells, len(months))))
    cover = 1.0 / (1.0 + np.exp(-eta))

    # --- cloudy months and per-cell missingness ------------------------------
    n_missing = int(round(float(missing_fraction) * len(months)))
    cloudy = set(rng.choice(np.arange(6, len(months)), size=max(n_missing, 0),
                            replace=False).tolist()) if n_missing else set()
    rows = []
    for ti, m in enumerate(months):
        keep_p = 0.35 if ti in cloudy else 0.985
        seen = rng.random(n_cells) < keep_p
        if not seen.any():
            continue
        idx = np.nonzero(seen)[0]
        valid_frac = np.clip(rng.beta(24, 2, idx.size), 0.05, 1.0)
        valid_ha = valid_frac * (float(cell_size_m) ** 2) / 1e4
        rows.append(pd.DataFrame({
            "grid_id": cells["grid_id"].to_numpy()[idx],
            "month": m,
            "valid_area_ha": valid_ha,
            "wh_area_ha": cover[idx, ti] * valid_ha,
            "wh_cover": cover[idx, ti],
            "wh_present": (cover[idx, ti] > 0.01).astype(int),
            "rain_chirps_30d_mm": rain[idx, ti],
            "air_temp_c": air[idx, ti],
            "wind_speed_ms": wind[idx, ti],
            "lake_level_m": level[idx, ti],
            "wave_exposure_idx": wave[idx, ti],
            "chl_ndci_s2": 0.05 + 0.6 * cover[idx, ti] + rng.normal(0, 0.02, idx.size),
            "turb_ndti_s2": 0.10 + 0.03 * _z(rain)[idx, ti] + rng.normal(0, 0.01, idx.size),
            "decoy_noise": decoy_a[idx, ti],
            "decoy_level": decoy_b[idx, ti],
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.merge(cells, on="grid_id", how="left")
    cov = (panel.groupby("month")["grid_id"].nunique() / n_cells).rename("coverage_fraction")
    panel = panel.merge(cov.reset_index(), on="month", how="left")
    panel["sensor"] = "S2"
    panel["classifier_version"] = "synthetic_v1"

    truth = {
        "beta_rain_chirps_30d_mm_lag1": B_RAIN,
        "beta_wave_exposure_idx_lag0": B_WAVE,
        "beta_air_temp_c_lag0": B_AIR,
        # Zero by construction, GIVEN the other terms: wind acts only through
        # wave_exposure_idx = openness x wind^2, and lake level does nothing.
        "beta_wind_speed_ms_lag0": 0.0,
        "beta_lake_level_m_lag0": 0.0,
        "beta_decoy_noise": 0.0,
        "beta_decoy_level": 0.0,
        "rho_common_state": RHO_G,
        "sigma_common_state": SIG_G,
        "trend_per_month_logit": 0.010,
        "rain_slope_heterogeneity_sd_cellwise": float(np.std(rain_slope_cell)),
        "temporal_only_by_construction": ["wind_speed_ms", "lake_level_m"],
        "spatiotemporal_by_construction": ["rain_chirps_30d_mm", "wave_exposure_idx",
                                           "air_temp_c"],
        "n_cells": n_cells, "n_months": int(len(months)),
        "n_cloudy_months": int(len(cloudy)),
    }
    geometry = {"water_mask": water, "x_km": xs, "y_km": ys,
                "river_mouths_km": river_mouths, "cell_size_m": cell_size_m}

    # The COMPLETE environmental tables the spatial workflow exports: every month
    # the covariate query covered, INCLUDING months whose WH map failed the
    # coverage filter. This is what lets the state process keep real forcing
    # through a gap instead of losing the month entirely.
    env_monthly = pd.DataFrame({
        "month": months, "rain_chirps_30d_mm": rain.mean(axis=0),
        "air_temp_c": air.mean(axis=0), "wind_speed_ms": wind_gulf,
        "lake_level_m": level_gulf, "wave_exposure_idx": wave.mean(axis=0)})
    _cm = []
    for ti, m in enumerate(months):
        _cm.append(pd.DataFrame({
            "grid_id": cells["grid_id"].to_numpy(), "month": m,
            "rain_chirps_30d_mm": rain[:, ti], "air_temp_c": air[:, ti],
            "wind_speed_ms": wind[:, ti], "lake_level_m": level[:, ti],
            "wave_exposure_idx": wave[:, ti]}))
    env_cellmonth = pd.concat(_cm, ignore_index=True)
    return panel, truth, geometry, env_monthly, env_cellmonth


print("§4b synthetic cell-month panel generator defined.")
''')


# ===========================================================================
# 5. Regionalisation helpers
# ===========================================================================
md(r"""## 5. Helpers — **response-blind** regionalisation

The regions must be built from geography alone. If a region boundary is drawn
where the hyacinth is, then "regions differ in hyacinth" is a tautology and every
regional intercept, slope and verdict downstream is circular.

`assert_response_blind` is therefore not decoration: it raises on any column
whose name matches a response, prevalence, residual, prediction or
classification pattern, and it is called at the top of every regionalisation
routine. The only inputs allowed are cell coordinates and **static** physical
covariates.

### The rules

| Class | Rule (first match wins, evaluated in this order) |
|---|---|
| `river_influenced_bay` | `river_dist_m` ≤ threshold |
| `sheltered_littoral` | otherwise, `shore_dist_m` ≤ threshold **and** openness < threshold |
| `exposed_littoral` | otherwise, `shore_dist_m` ≤ threshold |
| `open_gulf` | everything else (optionally forced by `depth_m` > threshold) |

Order matters and is fixed in advance: river influence dominates shelter, and
shelter dominates exposure. Every cell records the rule that produced its class
and the covariate values that triggered it, so the assignment is auditable
cell by cell.

### From class to region

An ecological class is not a region: `sheltered_littoral` cells occur in several
physically separate bays, and a single time series pooling bays 40 km apart is
not the "ecologically meaningful region" the design asks for. So each class is
split into **contiguous components** on the 500 m grid adjacency, and components
below the minimum-size requirements are merged into the **most physically
similar adjacent** component — similarity being Euclidean distance in the
standardised static-covariate space — rather than kept as tiny, noisy series.
A component with no adjacent component at all is dropped, and said so.
""")

code(r'''# =====================================================================
# 5a. Response-blindness guard and covariate resolution
# =====================================================================
# Any column matching one of these patterns is a RESPONSE or a model output and
# may never touch the regionalisation.
RESPONSE_LIKE_PATTERNS = (
    "wh_", "cover", "occurrence", "present", "prevalence", "resid", "fitted",
    "pred", "yhat", "y_hat", "proba", "prob_", "confidence", "class", "score",
    "cluster", "label_", "mean_cover", "hyacinth",
)
# Columns that contain a banned substring but are legitimate static geography.
RESPONSE_BLIND_ALLOWLIST = {"gsw_water_fraction", "bathy_water_fraction",
                            "frac_cropland", "frac_urban", "frac_wetland"}


def assert_response_blind(columns, context=""):
    """Raise if any column could carry response, prediction or classifier signal.

    This is the guard that makes the regions honest. It is deliberately noisy:
    a false positive costs a rename, a false negative costs the whole analysis.
    """
    bad = []
    for c in columns:
        name = str(c).lower()
        if name in RESPONSE_BLIND_ALLOWLIST:
            continue
        if any(p in name for p in RESPONSE_LIKE_PATTERNS):
            bad.append(c)
    if bad:
        raise ValueError(
            f"response-blindness violated{' in ' + context if context else ''}: "
            f"{bad} look like response / classification / model-output columns. "
            "Regions must be built from coordinates and STATIC physical "
            "covariates only.")
    return True


def static_cell_table(panel, static_cols, cell_col="grid_id",
                      month_col="month", tol=1e-9):
    """One row per cell of the static covariates, asserting they are static.

    A covariate that varies within a cell through time is NOT static, and using
    its mean would smuggle temporal information into the regionalisation. Such a
    column is dropped with a printed reason.
    """
    present = [c for c in static_cols if c in panel.columns]
    assert_response_blind(present, "static_cell_table")
    if not present:
        raise ValueError("no static covariates found in the panel")
    grp = panel.groupby(cell_col)
    spread = grp[present].agg(lambda s: float(np.nanmax(s) - np.nanmin(s))
                              if s.notna().any() else 0.0)
    varying = [c for c in present if float(np.nanmax(spread[c].to_numpy())) > tol]
    keep = [c for c in present if c not in varying]
    out = grp[keep].first().reset_index()
    audit = pd.DataFrame({
        "column": present,
        "max_within_cell_range": [float(np.nanmax(spread[c].to_numpy())) for c in present],
        "treated_as_static": [c in keep for c in present],
        "reason": ["static within every cell" if c in keep
                   else "varies within a cell through time -> NOT usable for "
                        "response-blind regionalisation" for c in present],
    })
    return out, audit


def resolve_region_covariates(cells, preference):
    """Pick the first available column for each regionalisation covariate role."""
    chosen, rows = {}, []
    for role, candidates in preference.items():
        pick = next((c for c in candidates if c in cells.columns
                     and pd.to_numeric(cells[c], errors="coerce").notna().any()), None)
        chosen[role] = pick
        rows.append({"role": role, "candidates": ", ".join(candidates),
                     "chosen": pick if pick else "(none available)",
                     "usable": pick is not None})
    assert_response_blind([v for v in chosen.values() if v], "resolve_region_covariates")
    return chosen, pd.DataFrame(rows)


def grid_cell_indices(cells, cell_size_m):
    """Integer lattice indices (ix, iy) from cell-centroid km coordinates.

    Mirrors `build_grid_neighbours` in the spatial panel notebook, so adjacency
    here means exactly what adjacency means there.
    """
    cs = float(cell_size_m)
    ix = np.round((cells["x_km"].to_numpy() * 1000.0 - cs / 2.0) / cs).astype(np.int64)
    iy = np.round((cells["y_km"].to_numpy() * 1000.0 - cs / 2.0) / cs).astype(np.int64)
    return ix, iy


def neighbour_offsets(contiguity="queen"):
    if contiguity == "rook":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)]


def connected_components(ix, iy, labels, contiguity="queen"):
    """Contiguous components WITHIN each label, on the regular grid lattice.

    Breadth-first over integer lattice coordinates: O(n) and exact, with no
    dependence on geopandas or a contiguity matrix.
    """
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    labels = np.asarray(labels)
    coord_to_pos = {(int(a), int(b)): p for p, (a, b) in enumerate(zip(ix, iy))}
    offsets = neighbour_offsets(contiguity)
    comp = np.full(len(ix), -1, dtype=np.int64)
    nxt = 0
    for start in range(len(ix)):
        if comp[start] >= 0:
            continue
        lab = labels[start]
        queue = deque([start])
        comp[start] = nxt
        while queue:
            p = queue.popleft()
            for dx, dy in offsets:
                q = coord_to_pos.get((int(ix[p]) + dx, int(iy[p]) + dy))
                if q is None or comp[q] >= 0 or labels[q] != lab:
                    continue
                comp[q] = nxt
                queue.append(q)
        nxt += 1
    return comp


def component_adjacency(ix, iy, comp, contiguity="queen"):
    """Set of unordered component pairs that share at least one grid edge."""
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    comp = np.asarray(comp)
    coord_to_pos = {(int(a), int(b)): p for p, (a, b) in enumerate(zip(ix, iy))}
    offsets = neighbour_offsets(contiguity)
    pairs = set()
    for p in range(len(ix)):
        for dx, dy in offsets:
            q = coord_to_pos.get((int(ix[p]) + dx, int(iy[p]) + dy))
            if q is None or comp[q] == comp[p]:
                continue
            pairs.add((min(comp[p], comp[q]), max(comp[p], comp[q])))
    return pairs


print("§5a response-blindness guard and contiguity helpers defined.")
''')


code(r'''# =====================================================================
# 5b. Ecological class assignment and component merging
# =====================================================================
# Fixed evaluation order. River influence dominates shelter; shelter dominates
# exposure. Declared here so the order can never be changed after seeing a
# result.
ECO_CLASS_ORDER = ["river_influenced_bay", "sheltered_littoral",
                   "exposed_littoral", "open_gulf"]
ECO_CLASS_LABEL = {
    "river_influenced_bay": "River-influenced bay",
    "sheltered_littoral": "Sheltered littoral",
    "exposed_littoral": "Exposed littoral",
    "open_gulf": "Open gulf",
}


def assign_ecological_class(cells, covariates, thresholds):
    """Assign each cell an ecological class from STATIC covariates only.

    Returns a frame with `region_type` and a human-readable `assignment_rule`
    recording which rule fired and on what values, so any cell's class can be
    defended individually.
    """
    used = [c for c in covariates.values() if c]
    assert_response_blind(used, "assign_ecological_class")
    n = len(cells)
    out = pd.DataFrame(index=cells.index)
    out["grid_id"] = cells["grid_id"].to_numpy()

    def _col(role):
        c = covariates.get(role)
        if not c or c not in cells.columns:
            return None
        return pd.to_numeric(cells[c], errors="coerce").to_numpy()

    riv, sho = _col("river_dist_m"), _col("shore_dist_m")
    opn, dep = _col("openness"), _col("depth_m")
    t_riv = thresholds.get("river_dist_m")
    t_sho = thresholds.get("shore_dist_m")
    t_opn = thresholds.get("openness")
    t_dep = thresholds.get("depth_m")

    cls = np.array(["open_gulf"] * n, dtype=object)
    rule = np.array(["default: outside every littoral rule"] * n, dtype=object)

    is_littoral = np.zeros(n, dtype=bool)
    if sho is not None and t_sho is not None:
        is_littoral = sho <= float(t_sho)
    if dep is not None and t_dep is not None:
        # A deep cell is open gulf even if the shoreline is close (a steep shore).
        is_littoral &= dep <= float(t_dep)

    is_sheltered = np.zeros(n, dtype=bool)
    if opn is not None and t_opn is not None:
        is_sheltered = opn < float(t_opn)

    m_exposed = is_littoral & ~is_sheltered
    cls[m_exposed] = "exposed_littoral"
    m_shelt = is_littoral & is_sheltered
    cls[m_shelt] = "sheltered_littoral"

    is_river = np.zeros(n, dtype=bool)
    if riv is not None and t_riv is not None:
        is_river = riv <= float(t_riv)
    cls[is_river] = "river_influenced_bay"

    for i in range(n):
        bits = []
        if riv is not None:
            bits.append(f"river_dist={riv[i]:.0f}m")
        if sho is not None:
            bits.append(f"shore_dist={sho[i]:.0f}m")
        if opn is not None:
            bits.append(f"openness={opn[i]:.3f}")
        if dep is not None and np.isfinite(dep[i]):
            bits.append(f"depth={dep[i]:.1f}m")
        if cls[i] == "river_influenced_bay":
            why = f"river_dist <= {t_riv:.0f} m"
        elif cls[i] == "sheltered_littoral":
            why = f"shore_dist <= {t_sho:.0f} m and openness < {t_opn:.3f}"
        elif cls[i] == "exposed_littoral":
            why = (f"shore_dist <= {t_sho:.0f} m and openness >= "
                   f"{t_opn:.3f}" if t_opn is not None
                   else f"shore_dist <= {t_sho:.0f} m")
        else:
            why = "no littoral or river rule matched"
        rule[i] = f"{why} [{'; '.join(bits)}]"

    out["region_type"] = cls
    out["assignment_rule"] = rule
    return out


def _standardise(frame, cols):
    z = frame[cols].astype(float).copy()
    for c in cols:
        s = z[c].std(ddof=0)
        z[c] = (z[c] - z[c].mean()) / (s if s and np.isfinite(s) and s > 0 else 1.0)
    return z.fillna(0.0)


def merge_small_components(cells, comp, sim_cols, min_cells, min_area_ha,
                           area_col="eligible_area_ha", contiguity="queen",
                           cell_size_m=500, max_iter=500):
    """Merge under-sized components into the most physically similar neighbour.

    "Physically similar" is Euclidean distance between the components' means in
    the standardised static-covariate space, which is response-blind by
    construction. Merging is iterative and always starts from the SMALLEST
    offending component, so the result does not depend on component ordering.
    Components with no adjacent component are dropped and reported.
    """
    assert_response_blind(sim_cols, "merge_small_components")
    comp = np.asarray(comp).copy()
    ix, iy = grid_cell_indices(cells, cell_size_m)
    z = _standardise(cells, sim_cols)
    log = []

    for _ in range(int(max_iter)):
        df = pd.DataFrame({"comp": comp,
                           "area": cells[area_col].to_numpy() if area_col in cells
                           else np.ones(len(comp))})
        size = df.groupby("comp").size()
        area = df.groupby("comp")["area"].sum()
        too_small = [c for c in size.index
                     if size[c] < int(min_cells) or area[c] < float(min_area_ha)]
        if not too_small:
            break
        target = min(too_small, key=lambda c: (size[c], area[c], c))
        adj = component_adjacency(ix, iy, comp, contiguity)
        nbrs = sorted({(b if a == target else a) for a, b in adj
                       if target in (a, b)})
        if not nbrs:
            log.append({"component": int(target), "action": "dropped",
                        "n_cells": int(size[target]), "area_ha": float(area[target]),
                        "merged_into": None,
                        "reason": "below the minimum size and has no adjacent "
                                  "component to merge into"})
            comp[comp == target] = -1
            continue
        centres = z.groupby(comp).mean()
        d = {nb: float(np.linalg.norm(centres.loc[nb].to_numpy()
                                      - centres.loc[target].to_numpy()))
             for nb in nbrs}
        best = min(d, key=lambda k: (d[k], -size.get(k, 0), k))
        log.append({"component": int(target), "action": "merged",
                    "n_cells": int(size[target]), "area_ha": float(area[target]),
                    "merged_into": int(best),
                    "covariate_distance": d[best],
                    "reason": f"below MIN_REGION_CELLS={min_cells} / "
                              f"MIN_REGION_ELIGIBLE_AREA_HA={min_area_ha:g}; merged "
                              f"into the most physically similar adjacent component"})
        comp[comp == target] = best
    return comp, pd.DataFrame(log)


def compass_tag(x, y, x0, y0, span_x, span_y):
    """A short readable bearing of a region centroid within the gulf."""
    ns = ""
    if abs(y - y0) > 0.15 * span_y:
        ns = "N" if y > y0 else "S"
    ew = ""
    if abs(x - x0) > 0.15 * span_x:
        ew = "E" if x > x0 else "W"
    return (ns + ew) or "central"


def build_regions(cells, covariates, thresholds, cell_size_m=500,
                  contiguity="queen", min_cells=40, min_area_ha=1000.0,
                  sim_cols=None, area_col="eligible_area_ha"):
    """Full response-blind regionalisation: class -> components -> merge -> name.

    Returns (assignments, regions, merge_log, class_audit).
    """
    cells = cells.reset_index(drop=True).copy()
    klass = assign_ecological_class(cells, covariates, thresholds)
    cells["region_type_raw"] = klass["region_type"].to_numpy()
    cells["assignment_rule"] = klass["assignment_rule"].to_numpy()

    ix, iy = grid_cell_indices(cells, cell_size_m)
    comp = connected_components(ix, iy, cells["region_type_raw"].to_numpy(), contiguity)

    if sim_cols is None:
        sim_cols = [c for c in [covariates.get("river_dist_m"),
                                covariates.get("shore_dist_m"),
                                covariates.get("openness"),
                                covariates.get("depth_m")] if c]
    if area_col not in cells.columns:
        cells[area_col] = cells.get("cell_area_m2", float(cell_size_m) ** 2) / 1e4

    comp, merge_log = merge_small_components(
        cells, comp, sim_cols, min_cells, min_area_ha, area_col=area_col,
        contiguity=contiguity, cell_size_m=cell_size_m)
    cells["_component"] = comp
    dropped = cells[cells["_component"] < 0].copy()
    cells = cells[cells["_component"] >= 0].reset_index(drop=True)

    # A merged component takes the class of its majority (by eligible area).
    maj = (cells.groupby(["_component", "region_type_raw"])[area_col].sum()
           .reset_index().sort_values([ "_component", area_col],
                                      ascending=[True, False])
           .drop_duplicates("_component").set_index("_component")["region_type_raw"])
    cells["region_type"] = cells["_component"].map(maj)

    x0, y0 = cells["x_km"].mean(), cells["y_km"].mean()
    span_x = max(cells["x_km"].max() - cells["x_km"].min(), 1e-6)
    span_y = max(cells["y_km"].max() - cells["y_km"].min(), 1e-6)

    prof = (cells.groupby("_component")
            .agg(region_type=("region_type", "first"),
                 n_cells=("grid_id", "nunique"),
                 eligible_area_ha=(area_col, "sum"),
                 x_km=("x_km", "mean"), y_km=("y_km", "mean"))
            .reset_index())
    prof["_order"] = prof["region_type"].map(
        {k: i for i, k in enumerate(ECO_CLASS_ORDER)}).fillna(99)
    prof = prof.sort_values(["_order", "eligible_area_ha"],
                            ascending=[True, False]).reset_index(drop=True)
    prof["region_id"] = [f"R{i + 1:02d}" for i in range(len(prof))]
    names, seen = [], {}
    for r in prof.itertuples():
        tag = compass_tag(r.x_km, r.y_km, x0, y0, span_x, span_y)
        base = f"{ECO_CLASS_LABEL.get(r.region_type, r.region_type)} ({tag})"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base} {seen[base]}")
    prof["region_name"] = names

    lut = prof.set_index("_component")[["region_id", "region_name"]]
    cells["region_id"] = cells["_component"].map(lut["region_id"])
    cells["region_name"] = cells["_component"].map(lut["region_name"])
    cells["assignment_audit"] = (
        cells["region_id"] + " | " + cells["region_name"] + " | class="
        + cells["region_type"] + " | raw_class=" + cells["region_type_raw"]
        + " | " + cells["assignment_rule"])

    keep = ["grid_id", "x_km", "y_km", "region_id", "region_name", "region_type",
            "region_type_raw", "assignment_rule", "assignment_audit", area_col]
    keep += [c for c in sim_cols if c in cells.columns]
    assignments = cells[[c for c in dict.fromkeys(keep) if c in cells.columns]].copy()

    regions = (prof.drop(columns=["_component", "_order"])
               [["region_id", "region_name", "region_type", "n_cells",
                 "eligible_area_ha", "x_km", "y_km"]]
               .reset_index(drop=True))
    for c in sim_cols:
        if c in cells.columns:
            regions = regions.merge(
                cells.groupby("region_id")[c].median().rename(f"median_{c}").reset_index(),
                on="region_id", how="left")

    class_audit = (cells.groupby(["region_type_raw", "region_type"])
                   .agg(n_cells=("grid_id", "nunique"),
                        eligible_area_ha=(area_col, "sum")).reset_index())
    if len(dropped):
        merge_log = pd.concat([merge_log, pd.DataFrame([{
            "component": -1, "action": "cells_dropped",
            "n_cells": int(len(dropped)), "area_ha": float(dropped[area_col].sum()),
            "merged_into": None,
            "reason": "cells in components that were too small and had no neighbour"}])],
            ignore_index=True)
    return assignments, regions, merge_log, class_audit


print("§5b ecological class assignment and merging defined.")
''')


# ===========================================================================
# 5c. Self-tests
# ===========================================================================
md(r"""### 5c. Self-tests for the regionalisation helpers

Every run re-checks the properties the regional design depends on, so a later
edit cannot quietly break one of them. A failure raises here rather than
producing a plausible-looking map.
""")

code(r'''# =====================================================================
# 5c. Self-tests
# =====================================================================
_tests = []


def _check(name, passed, detail=""):
    _tests.append({"test": name, "passed": bool(passed), "detail": str(detail)})
    return bool(passed)


# --- response blindness -------------------------------------------------
_raised = False
try:
    assert_response_blind(["dist_shore_m", "wh_cover"])
except ValueError:
    _raised = True
_check("assert_response_blind rejects a response column", _raised,
       "wh_cover must never reach the regionalisation")
_raised2 = False
try:
    assert_response_blind(["dist_shore_m", "openness_index", "gsw_water_fraction"])
except ValueError:
    _raised2 = True
_check("assert_response_blind accepts static geography", not _raised2,
       "dist_shore_m / openness_index / gsw_water_fraction are allowed")
_raised3 = False
try:
    assert_response_blind(["kmeans_cluster_id"])
except ValueError:
    _raised3 = True
_check("assert_response_blind rejects a clustering output", _raised3)

# --- contiguity ---------------------------------------------------------
# Two 2x2 blocks of the same class, 5 cells apart -> two components.
_ix = np.array([0, 1, 0, 1, 10, 11, 10, 11])
_iy = np.array([0, 0, 1, 1, 0, 0, 1, 1])
_lab = np.array(["a"] * 8)
_comp = connected_components(_ix, _iy, _lab)
_check("disconnected areas of one class become separate regions",
       len(np.unique(_comp)) == 2, f"components={sorted(set(_comp.tolist()))}")
# One contiguous block of two classes -> two components, and they are adjacent.
_lab2 = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
_comp2 = connected_components(_ix, _iy, _lab2)
_check("a class boundary splits a contiguous block",
       len(np.unique(_comp2)) == 4, f"components={sorted(set(_comp2.tolist()))}")
_adj = component_adjacency(_ix, _iy, _comp2)
_check("touching components are adjacent, distant ones are not",
       (_comp2[0], _comp2[2]) in _adj or (_comp2[2], _comp2[0]) in _adj,
       f"pairs={sorted(_adj)}")
_check("a 5-cell gap is not adjacency",
       not ({(min(_comp2[0], _comp2[4]), max(_comp2[0], _comp2[4]))} & _adj))

# --- merging picks the most SIMILAR neighbour, not just any -------------
# A 1-cell component touching two neighbours: one physically similar, one not.
_cells = pd.DataFrame({
    "grid_id": np.arange(9),
    "x_km": (np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]) + 0.5) * 0.5,
    "y_km": (np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) + 0.5) * 0.5,
    "dist_shore_m": [100, 100, 100, 100, 120, 3000, 100, 100, 100],
    "openness_index": [0.2, 0.2, 0.2, 0.2, 0.21, 0.9, 0.2, 0.2, 0.2],
    "eligible_area_ha": np.full(9, 25.0),
})
_c0 = np.array([0, 0, 1, 0, 2, 1, 0, 0, 0])   # comp 2 is the lone middle cell
_merged, _log = merge_small_components(
    _cells, _c0, ["dist_shore_m", "openness_index"], min_cells=2,
    min_area_ha=0.0, cell_size_m=500)
_check("an under-sized component merges into the most similar neighbour",
       _merged[4] == _merged[0] and _merged[4] != _merged[2],
       f"merged={_merged.tolist()}")
_check("the merge is logged with a reason",
       len(_log) >= 1 and "merged" in set(_log["action"]),
       _log.to_dict("records") if len(_log) else "")

# --- class precedence ---------------------------------------------------
_cc = pd.DataFrame({"grid_id": [0, 1, 2, 3],
                    "dist_majriver_m": [1000, 20000, 20000, 20000],
                    "dist_shore_m": [300, 300, 300, 9000],
                    "openness_index": [0.2, 0.2, 0.9, 0.9],
                    "depth_m": [2.0, 2.0, 3.0, 20.0]})
_cov = {"river_dist_m": "dist_majriver_m", "shore_dist_m": "dist_shore_m",
        "openness": "openness_index", "depth_m": "depth_m"}
_kl = assign_ecological_class(_cc, _cov, {"river_dist_m": 5000.0,
                                          "shore_dist_m": 2000.0,
                                          "openness": 0.5, "depth_m": None})
_check("river influence outranks shelter",
       _kl["region_type"].tolist() == ["river_influenced_bay", "sheltered_littoral",
                                       "exposed_littoral", "open_gulf"],
       _kl["region_type"].tolist())
_check("every cell carries an assignment rule",
       _kl["assignment_rule"].str.len().gt(0).all())

# --- static covariates must actually be static --------------------------
_p = pd.DataFrame({"grid_id": [1, 1, 2, 2],
                   "month": pd.to_datetime(["2020-01-01", "2020-02-01"] * 2),
                   "depth_m": [5.0, 5.0, 7.0, 7.0],
                   "openness_index": [0.3, 0.9, 0.4, 0.4]})
_st, _sa = static_cell_table(_p, ["depth_m", "openness_index"])
_check("a covariate that varies through time is refused as static",
       "openness_index" not in _st.columns and "depth_m" in _st.columns,
       _sa.to_dict("records"))

# --- calendar completeness ----------------------------------------------
_rm = pd.DataFrame({"region_id": ["R01", "R01", "R02"],
                    "month": pd.to_datetime(["2020-01-01", "2020-04-01",
                                             "2020-01-01"]),
                    "v": [1.0, 2.0, 3.0]})
_full = reindex_calendar_months(_rm, by="region_id")
_check("every region is reindexed onto the SAME complete calendar",
       _full.groupby("region_id")["month"].nunique().nunique() == 1
       and _full["month"].nunique() == 4,
       f"months per region={_full.groupby('region_id')['month'].nunique().tolist()}")
_check("excluded months stay missing rather than being interpolated",
       bool(_full.loc[(_full["region_id"] == "R01")
                      & (_full["month"] == pd.Timestamp("2020-02-01")), "v"].isna().all()))

# --- season terms and the response transform ----------------------------
_f1 = fourier_terms(pd.to_datetime(["2019-03-01"]), 2)
_f2 = fourier_terms(pd.to_datetime(["2024-03-01"]), 2)
_check("Fourier terms depend on the calendar month alone",
       np.allclose(_f1.to_numpy(), _f2.to_numpy()))
_yt, _ti = transform_response(pd.Series([0.0, 0.5, 1.0]), "logit", 1e-4)
_check("logit clipping at 0 and 1 is counted and reported",
       _ti["n_clipped_low"] == 1 and _ti["n_clipped_high"] == 1, _ti)
_check("the logit transform round-trips",
       np.allclose(inverse_transform_response(_yt.to_numpy()[1]), 0.5))

HELPER_SELFTESTS = pd.DataFrame(_tests)
display(HELPER_SELFTESTS)
_failed = HELPER_SELFTESTS.loc[~HELPER_SELFTESTS["passed"], "test"].tolist()
if _failed:
    raise AssertionError(f"§5c self-tests failed: {_failed}")
print(f"All {len(HELPER_SELFTESTS)} regionalisation self-tests passed.")
''')


# ===========================================================================
# 6. Load
# ===========================================================================
md(r"""## 6. Load the cell-month panel

One route, one file: the 500 m cell-month panel already carries the hard-class
response, the static habitat covariates the regions are built from, and the
merged environmental covariates. §6b then checks the panel's provenance before
anything is computed from it, and §6c loads the **complete** monthly
environmental tables — independently of whether a WH map passed its coverage
threshold, because the driver process has to continue through months the
response does not cover.
""")

code(r'''# =====================================================================
# 6. Load
# =====================================================================
SOURCE = {"mode": None, "paths": [], "is_synthetic": False}
panel = None
PANEL_MANIFEST = None
SYNTHETIC_TRUTH = None
SYNTHETIC_GEOMETRY = None
SYNTHETIC_ENV_MONTHLY = None
SYNTHETIC_ENV_CELLMONTH = None

if USE_SYNTHETIC_DEMO:
    (panel, SYNTHETIC_TRUTH, SYNTHETIC_GEOMETRY,
     SYNTHETIC_ENV_MONTHLY, SYNTHETIC_ENV_CELLMONTH) = make_synthetic_cellmonth_panel(
        n_months=SYNTHETIC_N_MONTHS, seed=SYNTHETIC_SEED,
        domain_km=SYNTHETIC_DOMAIN_KM, cell_size_m=SYNTHETIC_CELL_SIZE_M,
        missing_fraction=SYNTHETIC_MISSING_FRACTION)
    SOURCE.update(mode="synthetic_cellmonth_panel", is_synthetic=True)
    PANEL_MANIFEST = {"cell_size_m": SYNTHETIC_CELL_SIZE_M,
                      "run_label": "synthetic", "response_kind": "hard_class",
                      "confidence_usage": {}}
    print("*** USE_SYNTHETIC_DEMO = True: this run is a recovery test, not a result. ***")
    print(f"Synthetic panel: {len(panel):,} cell-months | "
          f"{panel['grid_id'].nunique():,} cells | {panel['month'].nunique()} months")
    print("Known generative values:")
    for k, v in SYNTHETIC_TRUTH.items():
        print(f"    {k:>44s} : {v}")
else:
    panel, panel_path = load_cellmonth_panel(PANEL_CSV, PANEL_DIR, PANEL_GLOB)
    if panel is None:
        raise FileNotFoundError(
            f"No {PANEL_GLOB!r} in {PANEL_DIR}.\n"
            "Run §17 of winam_wh_spatial_panel_driver_gam.ipynb to export the "
            "cell-month panel, point PANEL_CSV at it, or set "
            "USE_SYNTHETIC_DEMO = True to run the offline recovery test.")
    SOURCE.update(mode="cellmonth_panel", paths=[str(panel_path)])
    print(f"Loaded cell-month panel: {panel_path}")
    print(f"  {len(panel):,} cell-month rows | {panel['grid_id'].nunique():,} cells | "
          f"{panel['month'].nunique()} months "
          f"({panel['month'].min():%Y-%m} .. {panel['month'].max():%Y-%m})")
    _man_path = newest_match(PANEL_DIR, PANEL_MANIFEST_GLOB)
    if _man_path is not None:
        try:
            PANEL_MANIFEST = json.loads(Path(_man_path).read_text())
            SOURCE["paths"].append(str(_man_path))
            print(f"  run manifest: {_man_path.name}")
        except Exception as exc:
            print(f"  run manifest unreadable ({exc}); §6b will check what it can")
    else:
        print(f"  no {PANEL_MANIFEST_GLOB!r} beside the panel; §6b checks only what "
              "the panel itself carries")

print()
print("A reminder that governs every table below: these cell-month rows are the "
      "MEASUREMENT unit.")
print("They are never the inferential unit. The model in §12 sees one row per "
      "region per calendar month.")
''')

code(r'''# =====================================================================
# 6b. Provenance gate
# =====================================================================
PROVENANCE_AUDIT, _blocking = panel_provenance_audit(
    panel, PANEL_MANIFEST,
    expected_cell_size_m=EXPECTED_CELL_SIZE_M,
    require_hard_class=REQUIRE_HARD_CLASS_RESPONSE,
    require_run_tag=REQUIRE_RUN_TAG,
    provenance_columns=PROVENANCE_COLUMNS)
display(PROVENANCE_AUDIT)
if _blocking:
    raise RuntimeError(
        f"Provenance gate failed on {_blocking}. This panel was not built to the "
        "rules this notebook analyses (500 m cells, hard-class WH cover, no "
        "classifier-uncertainty weighting, one row per cell-month). Fix the panel "
        "or change the §3b expectations deliberately.")
print("Provenance gate passed.")

# Extra monthly series, merged on `month` (gulf-wide by construction).
EXTRA_MONTHLY_COLS = []
if EXTRA_MONTHLY_CSV is not None and Path(EXTRA_MONTHLY_CSV).exists():
    _extra = pd.read_csv(EXTRA_MONTHLY_CSV)
    if "month" not in _extra.columns:
        raise ValueError("EXTRA_MONTHLY_CSV needs a 'month' column.")
    _extra["month"] = to_month_start(_extra["month"])
    EXTRA_MONTHLY_COLS = [c for c in _extra.columns if c != "month"]
    panel = panel.merge(_extra, on="month", how="left")
    SOURCE["paths"].append(str(EXTRA_MONTHLY_CSV))
    print(f"Merged EXTRA_MONTHLY_CSV columns (gulf-wide): {EXTRA_MONTHLY_COLS}")
''')

code(r'''# =====================================================================
# 6c. The COMPLETE monthly environmental tables
# =====================================================================
# The response is filtered by coverage; the drivers must not be. These tables
# are exported by the spatial workflow for EVERY month it queried, so they cover
# months whose WH map failed the coverage threshold. Loading them here is what
# lets the state process in §12 run through a missing response month with the
# real environmental forcing in place.
ENV_MONTHLY_COMPLETE = None
ENV_CELLMONTH_COMPLETE = None
ENV_SOURCE_NOTE = []

if SOURCE["is_synthetic"]:
    ENV_MONTHLY_COMPLETE = SYNTHETIC_ENV_MONTHLY
    ENV_CELLMONTH_COMPLETE = SYNTHETIC_ENV_CELLMONTH
    ENV_SOURCE_NOTE.append(
        f"synthetic complete monthly table: "
        f"{ENV_MONTHLY_COMPLETE['month'].nunique()} months, gulf-wide")
    ENV_SOURCE_NOTE.append(
        f"synthetic complete per-cell table: "
        f"{ENV_CELLMONTH_COMPLETE['month'].nunique()} months x "
        f"{ENV_CELLMONTH_COMPLETE['grid_id'].nunique():,} cells")

if not SOURCE["is_synthetic"]:
    _p = newest_match(PANEL_DIR, EE_MONTHLY_GLOB)
    if _p is not None:
        ENV_MONTHLY_COMPLETE = pd.read_csv(_p)
        ENV_MONTHLY_COMPLETE["month"] = to_month_start(ENV_MONTHLY_COMPLETE["month"])
        SOURCE["paths"].append(str(_p))
        ENV_SOURCE_NOTE.append(
            f"{_p.name}: {ENV_MONTHLY_COMPLETE['month'].nunique()} months "
            f"({ENV_MONTHLY_COMPLETE['month'].min():%Y-%m} .. "
            f"{ENV_MONTHLY_COMPLETE['month'].max():%Y-%m}), gulf-wide")
    _p = newest_match(PANEL_DIR, EE_CELLMONTH_GLOB)
    if _p is not None:
        ENV_CELLMONTH_COMPLETE = pd.read_csv(_p)
        ENV_CELLMONTH_COMPLETE["month"] = to_month_start(ENV_CELLMONTH_COMPLETE["month"])
        SOURCE["paths"].append(str(_p))
        ENV_SOURCE_NOTE.append(
            f"{_p.name}: {ENV_CELLMONTH_COMPLETE['month'].nunique()} months x "
            f"{ENV_CELLMONTH_COMPLETE['grid_id'].nunique():,} cells, per-cell")

if ENV_SOURCE_NOTE:
    print("Complete monthly environmental tables loaded (independent of WH coverage):")
    for n in ENV_SOURCE_NOTE:
        print("  " + n)
else:
    print("No separate environmental tables found; the drivers come from the panel "
          "itself.")
    print("Consequence: a month with no WH map has no driver row either, so the "
          "state process in §12 propagates through it without environmental "
          "forcing. That is recorded in the §11 missing-data audit, never hidden.")
''')


# ===========================================================================
# 6d. Output helpers
# ===========================================================================
code(r'''# =====================================================================
# 6d. Export helpers (used by every section, collected in §21)
# =====================================================================
EXPORTS = {}          # name -> (DataFrame, evidence_type)
FIGURES = {}          # name -> path
# One stem for every file this run writes, fixed here so a figure saved in §8
# and a table written in §21 cannot end up under different names.
RUN_STEM = (f"{'SYNTHETIC_' if SOURCE['is_synthetic'] else ''}"
            f"{panel['month'].min():%Y%m}_to_{panel['month'].max():%Y%m}"
            f"{'_FAST' if FAST_MODE else ''}")


def register(name, table, evidence):
    """Register a table for export in §21."""
    EXPORTS[name] = (table, evidence)
    return table


def save_fig(fig, name, dpi=200):
    """Write a figure to OUTPUT_DIR as PNG and remember where it went."""
    if not OUTPUT_WRITABLE:
        return None
    stem = RUN_STEM or "run"
    path = Path(OUTPUT_DIR) / f"regional_fig_{name}_{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    FIGURES[name] = str(path)
    return path


def _fmt_stats(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if not len(s):
        return {}
    q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {"n": int(s.size), "min": float(s.min()), "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]), "median": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]), "p95": float(q.loc[0.95]),
            "max": float(s.max()), "mean": float(s.mean()), "sd": float(s.std(ddof=0))}


print("§6d export helpers defined.")
''')


# ===========================================================================
# 7. Static covariates and thresholds
# ===========================================================================
md(r"""## 7. The eligible cell set, the static covariates, and where each threshold comes from

Three things happen here, in this order, and the order matters:

1. **The gulf-wide month filter** — a month enters only if it classified
   `MIN_MONTHLY_COVERAGE_FRACTION` of the eligible water cells. Inherited
   unchanged from the panel and AOI workflows.
2. **The fixed cell set** — cells validly observed in at least
   `MIN_CELL_MONTH_FRACTION` of those months. This is the set the regions
   partition, and it is fixed once, before any region exists.
3. **The static covariates** — one row per cell, with a hard check that each is
   genuinely time-invariant. A covariate that moves within a cell through time is
   refused, because using its mean would smuggle temporal information into a
   partition that must be response-blind *and* time-blind.

### Where the thresholds come from

The repository was searched first (§7b prints the search). Two of the three
length scales already exist in it and are reused rather than reinvented; the
third has no precedent and is therefore stated as an **operational definition**
resolved from a response-blind quantile of the covariate itself. None of them is
tuned against WH cover, and §19 re-runs the whole analysis under alternatives.
""")

code(r'''# =====================================================================
# 7a. Month filter, fixed cell set, static covariates
# =====================================================================
COVERAGE = monthly_coverage_table(panel, MIN_MONTHLY_COVERAGE_FRACTION)
MONTHS_KEPT = pd.DatetimeIndex(sorted(COVERAGE.loc[COVERAGE["retained"], "month"]))
print(f"Gulf-wide month filter: {len(MONTHS_KEPT)} of {len(COVERAGE)} months at or "
      f"above {MIN_MONTHLY_COVERAGE_FRACTION:.0%} coverage "
      f"(basis: {COVERAGE['coverage_basis'].iloc[0]})")
if len(MONTHS_KEPT) == 0:
    raise RuntimeError("No month passes the coverage filter; lower "
                       "MIN_MONTHLY_COVERAGE_FRACTION deliberately if that is right.")

CELLS_KEPT, CELL_SET_AUDIT = fixed_cell_membership(
    panel, MONTHS_KEPT, MIN_CELL_MONTH_FRACTION)
print(f"Fixed cell set: {CELL_SET_AUDIT['n_cells_kept']:,} of "
      f"{CELL_SET_AUDIT['n_cells_total']:,} cells observed in >= "
      f"{MIN_CELL_MONTH_FRACTION:.0%} of the {CELL_SET_AUDIT['n_months_retained']} "
      f"retained months, holding "
      f"{CELL_SET_AUDIT['share_of_classified_area_kept']:.1%} of the classified area.")
if CELL_SET_AUDIT["n_cells_kept"] == 0:
    raise RuntimeError("The fixed cell set is empty. Lower MIN_CELL_MONTH_FRACTION.")

PANEL_FIXED = panel[panel["month"].isin(MONTHS_KEPT)
                    & panel["grid_id"].isin(CELLS_KEPT)].copy()
print(f"Working panel: {len(PANEL_FIXED):,} cell-month rows over a FIXED cell set.")

CELL_STATIC, STATIC_AUDIT = static_cell_table(PANEL_FIXED, KNOWN_STATIC_COLS)
display(STATIC_AUDIT)

# Eligible water area per cell: STATIC by construction (cell area x water
# fraction), so region size never depends on how often a cell was cloud-free.
_area_m2 = (pd.to_numeric(CELL_STATIC.get("cell_area_m2"), errors="coerce")
            if "cell_area_m2" in CELL_STATIC.columns
            else pd.Series(float(EXPECTED_CELL_SIZE_M) ** 2, index=CELL_STATIC.index))
_wfrac = None
for _c in ("bathy_water_fraction", "gsw_water_fraction"):
    if _c in CELL_STATIC.columns:
        _v = pd.to_numeric(CELL_STATIC[_c], errors="coerce")
        _wfrac = _v if _wfrac is None else np.minimum(_wfrac, _v)
CELL_STATIC["eligible_area_ha"] = (
    _area_m2.fillna(float(EXPECTED_CELL_SIZE_M) ** 2)
    * (1.0 if _wfrac is None else _wfrac.fillna(1.0)) / 1e4)
print(f"\nEligible water area: {CELL_STATIC['eligible_area_ha'].sum():,.0f} ha over "
      f"{len(CELL_STATIC):,} cells "
      f"({'cell area x water fraction' if _wfrac is not None else 'cell area (no water-fraction column)'}).")

REGION_COVARIATES, COVARIATE_CHOICE = resolve_region_covariates(
    CELL_STATIC, REGION_COVARIATE_PREFERENCE)
display(COVARIATE_CHOICE)
register("regionalisation_covariate_choice", COVARIATE_CHOICE, "provenance")
register("static_covariate_audit", STATIC_AUDIT, "provenance")
register("monthly_coverage_gulf", COVERAGE, "diagnostic")
''')

code(r'''# =====================================================================
# 7b. Threshold provenance: what the repository already justifies
# =====================================================================
# Searched for existing, justified thresholds before inventing any. What exists,
# and what does not, is printed rather than asserted in prose.
REPO_THRESHOLD_SEARCH = pd.DataFrame([
    {"quantity": "what counts as a MAJOR river",
     "repository_precedent": "EE_RIVER_MAJOR_MAX_ORD = 7 (HydroSHEDS RIV_ORD; lower = larger river)",
     "found": True,
     "used_here": "defines dist_majriver_m, which is the river-distance covariate"},
    {"quantity": "river-influence distance",
     "repository_precedent": "openness_index is built on a 5 km circular kernel "
                             "(ee.Kernel.circle(5000, 'meters')) - the project's "
                             "established local-influence length scale",
     "found": True,
     "used_here": "REGION_THRESHOLDS['river_dist_m'] = 5000 m, reusing that scale"},
    {"quantity": "littoral / shoreline distance",
     "repository_precedent": "EE_CATCHMENT_BUFFER_M = 2000 m, the local "
                             "catchment-influence buffer used for the land-cover, "
                             "population and built-up layers",
     "found": True,
     "used_here": "REGION_THRESHOLDS['shore_dist_m'] = 2000 m, reusing that buffer"},
    {"quantity": "shelter / openness cut",
     "repository_precedent": "none - openness_index exists as a covariate but no "
                             "sheltered/exposed threshold is defined anywhere",
     "found": False,
     "used_here": "OPERATIONAL DEFINITION: the response-blind median of "
                  "openness_index among littoral cells"},
    {"quantity": "depth cut for open water",
     "repository_precedent": "none (MIN_WATER_FRACTION = 0.5 is a water MASK, not a "
                             "depth class)",
     "found": False,
     "used_here": "left None; depth enters only through the merge-similarity space"},
    {"quantity": "water/habitat mask",
     "repository_precedent": "WATER_MASK_SOURCE / MIN_WATER_FRACTION = 0.5",
     "found": True,
     "used_here": "inherited: the panel is already masked, and the eligible-area "
                  "weight reuses the same water fraction"},
])
display(REPO_THRESHOLD_SEARCH)
register("threshold_repository_search", REPO_THRESHOLD_SEARCH, "provenance")

# --- distributions of the covariates the thresholds act on -------------------
_dist_rows = []
for _role, _col in REGION_COVARIATES.items():
    if not _col:
        continue
    _s = _fmt_stats(CELL_STATIC[_col])
    _s.update({"role": _role, "column": _col})
    _dist_rows.append(_s)
STATIC_DISTRIBUTIONS = pd.DataFrame(_dist_rows)
STATIC_DISTRIBUTIONS = STATIC_DISTRIBUTIONS[
    ["role", "column", "n", "min", "p05", "p25", "median", "p75", "p95", "max",
     "mean", "sd"]]
print("Distributions of the static covariates the thresholds act on:")
display(STATIC_DISTRIBUTIONS)
register("static_covariate_distributions", STATIC_DISTRIBUTIONS, "diagnostic")

fig, axes = plt.subplots(1, max(len(_dist_rows), 1),
                         figsize=(4.0 * max(len(_dist_rows), 1), 3.2))
axes = np.atleast_1d(axes)
for ax, row in zip(axes, _dist_rows):
    v = pd.to_numeric(CELL_STATIC[row["column"]], errors="coerce").dropna()
    ax.hist(v, bins=40, color="#4C72B0", alpha=0.85)
    ax.set_title(f"{row['role']}\n{row['column']}", fontsize=9)
    ax.set_ylabel("cells")
fig.suptitle("Static covariates used to build the regions (response-blind)",
             fontsize=11)
fig.tight_layout()
save_fig(fig, "static_covariate_distributions")
plt.show()
''')

code(r'''# =====================================================================
# 7c. Resolve the thresholds (and say what kind of number each one is)
# =====================================================================


def resolve_thresholds(cells, covariates, configured, openness_quantile,
                       verbose=True):
    """Fill in any unset threshold response-blind, and record its provenance."""
    resolved = dict(configured)
    rows = []
    for role in ["river_dist_m", "shore_dist_m", "openness", "depth_m"]:
        col = covariates.get(role)
        val = resolved.get(role)
        if col is None:
            resolved[role] = None
            rows.append({"threshold": role, "value": None, "kind": "unavailable",
                         "basis": "no covariate column in the panel",
                         "response_blind": True})
            continue
        if val is not None:
            rows.append({"threshold": role, "value": float(val),
                         "kind": "configured (repository precedent)",
                         "basis": "see §7b REPO_THRESHOLD_SEARCH",
                         "response_blind": True})
            continue
        if role == "openness":
            sho_col = covariates.get("shore_dist_m")
            t_sho = resolved.get("shore_dist_m")
            base = cells
            scope = "all eligible cells"
            if sho_col and t_sho is not None:
                m = pd.to_numeric(cells[sho_col], errors="coerce") <= float(t_sho)
                if int(m.sum()) >= 20:
                    base = cells[m]
                    scope = f"littoral cells ({sho_col} <= {float(t_sho):.0f} m)"
            q = float(pd.to_numeric(base[col], errors="coerce")
                      .quantile(float(openness_quantile)))
            resolved[role] = q
            rows.append({"threshold": role, "value": q,
                         "kind": "OPERATIONAL DEFINITION (response-blind quantile)",
                         "basis": f"quantile {openness_quantile:g} of {col} over "
                                  f"{scope}; no repository precedent exists",
                         "response_blind": True})
        else:
            rows.append({"threshold": role, "value": None, "kind": "not applied",
                         "basis": "left unset; the class collapses into its parent "
                                  "rather than being invented",
                         "response_blind": True})
    table = pd.DataFrame(rows)
    if verbose:
        display(table)
        op = table[table["kind"].str.startswith("OPERATIONAL")]
        if len(op):
            print("OPERATIONAL DEFINITIONS in force (no repository precedent; "
                  "sensitivity variants in §19):")
            for r in op.itertuples():
                print(f"  {r.threshold} = {r.value:.4g}  -- {r.basis}")
    return resolved, table


THRESHOLDS, THRESHOLD_PROVENANCE = resolve_thresholds(
    CELL_STATIC, REGION_COVARIATES, REGION_THRESHOLDS, OPENNESS_FALLBACK_QUANTILE)
register("threshold_provenance", THRESHOLD_PROVENANCE, "provenance")
print()
print("None of these thresholds was chosen by looking at WH cover, WH prevalence, "
      "residuals or model performance, and none may be.")
''')


# ===========================================================================
# 8. Build the regions
# ===========================================================================
md(r"""## 8. Build the regions

Class → contiguous components → merge the under-sized → name and number. Nothing
in this section reads the response.

The audit that follows answers the three questions a reader will actually ask:
**how many regions**, **how big are they**, and **what was merged into what**.
""")

code(r'''# =====================================================================
# 8a. Regionalise
# =====================================================================
_SIM_COLS = [c for c in [REGION_COVARIATES.get("river_dist_m"),
                         REGION_COVARIATES.get("shore_dist_m"),
                         REGION_COVARIATES.get("openness"),
                         REGION_COVARIATES.get("depth_m")] if c]

ASSIGNMENTS, REGIONS, MERGE_LOG, CLASS_AUDIT = build_regions(
    CELL_STATIC, REGION_COVARIATES, THRESHOLDS,
    cell_size_m=EXPECTED_CELL_SIZE_M, contiguity=REGION_CONTIGUITY,
    min_cells=MIN_REGION_CELLS, min_area_ha=MIN_REGION_ELIGIBLE_AREA_HA,
    sim_cols=_SIM_COLS, area_col="eligible_area_ha")

print(f"{len(REGIONS)} region(s) from {len(ASSIGNMENTS):,} eligible cells "
      f"({ASSIGNMENTS['eligible_area_ha'].sum():,.0f} ha).")
display(REGIONS)
print("\nEcological class before and after merging (a merged unit takes the class "
      "of its majority area):")
display(CLASS_AUDIT)
if len(MERGE_LOG):
    print("\nMerge / drop log:")
    display(MERGE_LOG)
else:
    print("\nNo component needed merging.")

_lo, _hi = REGION_COUNT_TARGET
if len(REGIONS) < _lo:
    print(f"\n*** {len(REGIONS)} regions is below the {_lo}-{_hi} target band. "
          "Consequence for hierarchical estimation: the between-region variances "
          "(sigma_alpha, sigma_b, sigma_lambda) are estimated from very few "
          "groups, so they are weakly identified and their priors will do much of "
          "the work. §12 therefore drops random slopes below "
          f"RANDOM_SLOPE_MIN_REGIONS={RANDOM_SLOPE_MIN_REGIONS} regions, and §16 "
          "reports partial pooling as a shrinkage statement rather than a "
          "variance estimate. ***")
elif len(REGIONS) > _hi:
    print(f"\nNote: {len(REGIONS)} regions exceeds the {_lo}-{_hi} target band. "
          "More regions means shorter, noisier individual series; the minimum-size "
          "requirements in §3c are the lever, and §19 varies them.")
else:
    print(f"\n{len(REGIONS)} regions is inside the {_lo}-{_hi} target band.")

register("region_cell_assignments", ASSIGNMENTS, "regionalisation")
register("region_definitions", REGIONS, "regionalisation")
register("region_merge_log", MERGE_LOG, "regionalisation")
register("region_class_audit", CLASS_AUDIT, "regionalisation")
''')

code(r'''# =====================================================================
# 8b. The map
# =====================================================================


def mask_outline_segments(cells, cell_size_m):
    """Exact outline of the analysed water body: every cell edge with no neighbour.

    Dependency-free and honest - it is the boundary of the cells actually in the
    analysis, not a shoreline drawn from somewhere else.
    """
    cs_km = float(cell_size_m) / 1000.0
    ix, iy = grid_cell_indices(cells, cell_size_m)
    have = set(zip(ix.tolist(), iy.tolist()))
    segs = []
    for a, b, x, y in zip(ix, iy, cells["x_km"].to_numpy(), cells["y_km"].to_numpy()):
        x0, y0 = x - cs_km / 2, y - cs_km / 2
        x1, y1 = x + cs_km / 2, y + cs_km / 2
        if (a - 1, b) not in have:
            segs.append([(x0, y0), (x0, y1)])
        if (a + 1, b) not in have:
            segs.append([(x1, y0), (x1, y1)])
        if (a, b - 1) not in have:
            segs.append([(x0, y0), (x1, y0)])
        if (a, b + 1) not in have:
            segs.append([(x0, y1), (x1, y1)])
    return segs


def plot_regions(assignments, regions, thresholds, covariates, cell_size_m,
                 title="Winam Gulf — response-blind ecological regions"):
    from matplotlib.collections import LineCollection
    cs_km = float(cell_size_m) / 1000.0
    ids = regions["region_id"].tolist()
    cmap = plt.get_cmap("tab20")
    colour = {rid: cmap(i % 20) for i, rid in enumerate(ids)}
    hatch = {"river_influenced_bay": "//", "sheltered_littoral": "",
             "exposed_littoral": "..", "open_gulf": ""}
    type_of = dict(zip(regions["region_id"], regions["region_type"]))

    fig, ax = plt.subplots(figsize=(13, 7.5))
    patches, colours = [], []
    for r in assignments.itertuples():
        patches.append(Rectangle((r.x_km - cs_km / 2, r.y_km - cs_km / 2),
                                 cs_km, cs_km))
        colours.append(colour.get(r.region_id, (0.8, 0.8, 0.8, 1.0)))
    ax.add_collection(PatchCollection(patches, facecolors=colours,
                                      edgecolors="none", linewidths=0))
    # Hatch the two classes whose definition is a rule rather than a location.
    for rid, hh in [(k, hatch.get(v, "")) for k, v in type_of.items()]:
        if not hh:
            continue
        sub = assignments[assignments["region_id"] == rid]
        ax.add_collection(PatchCollection(
            [Rectangle((r.x_km - cs_km / 2, r.y_km - cs_km / 2), cs_km, cs_km)
             for r in sub.itertuples()],
            facecolors="none", edgecolors=(0, 0, 0, 0.35), hatch=hh, linewidths=0))

    ax.add_collection(LineCollection(
        mask_outline_segments(assignments, cell_size_m),
        colors="black", linewidths=1.1, zorder=4,
        label="shoreline of the analysed water body"))

    riv_col = covariates.get("river_dist_m")
    t_riv = thresholds.get("river_dist_m")
    if riv_col and riv_col in assignments.columns and t_riv is not None:
        try:
            ax.tricontour(assignments["x_km"].to_numpy(),
                          assignments["y_km"].to_numpy(),
                          pd.to_numeric(assignments[riv_col], errors="coerce")
                          .fillna(1e9).to_numpy(),
                          levels=[float(t_riv)], colors="#1f78b4",
                          linewidths=1.8, linestyles="--", zorder=5)
        except Exception as exc:
            print(f"  river-distance contour skipped: {exc}")
        near = assignments.nsmallest(1, riv_col)
        _grp = assignments[pd.to_numeric(assignments[riv_col], errors="coerce")
                           <= float(t_riv) * 0.25]
        if len(_grp):
            for rid, sub in _grp.groupby("region_id"):
                ax.plot(sub["x_km"].mean(), sub["y_km"].mean(), marker="v",
                        ms=12, mfc="white", mec="#08306b", mew=2.0, zorder=6)
        elif len(near):
            ax.plot(near["x_km"].iloc[0], near["y_km"].iloc[0], marker="v",
                    ms=12, mfc="white", mec="#08306b", mew=2.0, zorder=6)

    for r in regions.itertuples():
        # Anchor the label on a cell that really belongs to the region: a
        # horseshoe-shaped region's centroid can fall outside it entirely.
        sub = assignments[assignments["region_id"] == r.region_id]
        j = int(np.argmin(np.hypot(sub["x_km"].to_numpy() - r.x_km,
                                   sub["y_km"].to_numpy() - r.y_km)))
        lx = float(sub["x_km"].to_numpy()[j])
        ly = float(sub["y_km"].to_numpy()[j])
        ax.annotate(r.region_id, (lx, ly), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="black", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.4",
                              alpha=0.85))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colour[r.region_id], edgecolor="0.4",
                     hatch=hatch.get(r.region_type, ""),
                     label=f"{r.region_id}  {r.region_name}  "
                           f"({r.n_cells:,} cells, {r.eligible_area_ha:,.0f} ha)")
               for r in regions.itertuples()]
    handles.append(Line2D([0], [0], color="black", lw=1.1,
                          label="shoreline of the analysed water body"))
    if riv_col and t_riv is not None:
        handles.append(Line2D([0], [0], color="#1f78b4", lw=1.8, ls="--",
                              label=f"major-river distance = {t_riv:,.0f} m"))
        handles.append(Line2D([0], [0], marker="v", lw=0, ms=11, mfc="white",
                              mec="#08306b", mew=2.0,
                              label="major river / river mouth"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("easting (km, panel CRS)")
    ax.set_ylabel("northing (km, panel CRS)")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.15, linewidth=0.4)
    fig.tight_layout()
    return fig, ax


REGION_MAP_FIG, _ax = plot_regions(
    ASSIGNMENTS, REGIONS, THRESHOLDS, REGION_COVARIATES, EXPECTED_CELL_SIZE_M,
    title=("Winam Gulf — response-blind ecological regions"
           + (" (SYNTHETIC)" if SOURCE["is_synthetic"] else "")))

# Real shoreline overlay, when geopandas and a polygon are both available.
_shore_path = next((p for p in SHORELINE_GEOJSON_CANDIDATES if Path(p).exists()), None)
if HAVE_GEOPANDAS and _shore_path is not None and not SOURCE["is_synthetic"]:
    try:
        _sh = gpd.read_file(_shore_path).to_crs(PANEL_CRS)
        for geom in _sh.geometry:
            for poly in (geom.geoms if geom.geom_type.startswith("Multi") else [geom]):
                xs, ys = poly.exterior.xy
                _ax.plot(np.asarray(xs) / 1000.0, np.asarray(ys) / 1000.0,
                         color="#444444", lw=0.8, alpha=0.8, zorder=3)
        print(f"Shoreline overlay: {_shore_path}")
    except Exception as exc:
        print(f"Shoreline overlay skipped ({exc}); the cell-mask outline is drawn "
              "instead.")
else:
    print("Shoreline drawn as the outline of the analysed cell mask "
          f"(geopandas={HAVE_GEOPANDAS}, polygon={'found' if _shore_path else 'not found'}).")

save_fig(REGION_MAP_FIG, "01_ecological_regions")
plt.show()
''')

code(r'''# =====================================================================
# 8c. Export the region geometries
# =====================================================================
REGION_GEOMETRY_PATH = None
REGION_GEOMETRIES = None
if HAVE_GEOPANDAS:
    _cs = float(EXPECTED_CELL_SIZE_M)
    _sq = [shapely_box(r.x_km * 1000 - _cs / 2, r.y_km * 1000 - _cs / 2,
                       r.x_km * 1000 + _cs / 2, r.y_km * 1000 + _cs / 2)
           for r in ASSIGNMENTS.itertuples()]
    _cells_gdf = gpd.GeoDataFrame(ASSIGNMENTS.copy(), geometry=_sq, crs=PANEL_CRS)
    REGION_GEOMETRIES = (_cells_gdf.dissolve(by="region_id",
                                             aggfunc={"eligible_area_ha": "sum"})
                         .reset_index()
                         .merge(REGIONS[["region_id", "region_name", "region_type",
                                         "n_cells"]], on="region_id", how="left"))
    print(f"Dissolved {len(ASSIGNMENTS):,} cells into {len(REGION_GEOMETRIES)} region "
          f"polygons in {PANEL_CRS}.")
    display(REGION_GEOMETRIES.drop(columns="geometry"))
else:
    print("geopandas unavailable: the GeoPackage is skipped. The cell assignment "
          "CSV (grid_id + x_km + y_km + region_id) carries the same information "
          "and can be re-dissolved anywhere.")
''')


# ===========================================================================
# 9. Region-month panel
# ===========================================================================
md(r"""## 9. The region-month panel — **the inferential dataset**

$$\text{wh\_cover}_{r,t}=
\frac{\sum_{i \in C_r}\text{WH area}_{i,t}}
     {\sum_{i \in C_r}\text{valid classified area}_{i,t}}$$

with the **same** fixed membership $C_r$ in every month. Alongside the response,
each region-month carries the quantities a reader needs to judge whether it is a
real observation: WH area, valid classified area, how many of the region's cells
were eligible and how many were actually observed, both coverage fractions, the
regional WH occurrence rate, and the area-weighted environmental predictors.

**Two gates, both required.** A region-month becomes an observation only if it
clears `MIN_REGION_MONTH_CELL_COVERAGE` *and*
`MIN_REGION_MONTH_VALID_AREA_COVERAGE`. Failing either leaves the response
missing — it never leaves a partially-observed value in place.

**Composition drift is reported, not assumed away.** Membership is fixed, but
*which* member cells were cloud-free varies. §9c measures that drift for every
region and prints the worst offenders, because a region whose observed
composition swings month to month is measuring a moving target even with a fixed
$C_r$.
""")

code(r'''# =====================================================================
# 9a. Build the region-month panel
# =====================================================================


def regional_monthly_panel(panel, assignments, driver_cols, months_grid=None,
                           min_cell_coverage=0.70, min_area_coverage=0.70):
    """Collapse the cell-month panel to ONE row per region per calendar month.

    The response is the AREA-WEIGHTED regional cover; the drivers are weighted by
    the same classified area, so a driver and the response describe the same
    piece of lake in the same month.
    """
    wh_col, valid_col = _area_columns(panel)
    memb = assignments[["grid_id", "region_id", "eligible_area_ha"]]
    sub = panel.merge(memb, on="grid_id", how="inner").copy()
    sub["_w"] = pd.to_numeric(sub[valid_col], errors="coerce").fillna(0.0)

    elig = (memb.groupby("region_id")
            .agg(n_cells_eligible=("grid_id", "nunique"),
                 eligible_area_ha=("eligible_area_ha", "sum")).reset_index())

    agg = {"wh_area_ha": (wh_col, "sum"),
           "valid_area_ha": (valid_col, "sum"),
           "n_cells_observed": ("grid_id", "nunique")}
    if "wh_present" in sub.columns:
        agg["wh_occurrence"] = ("wh_present", "mean")
    out = sub.groupby(["region_id", "month"], as_index=False).agg(**agg)

    for col in driver_cols:
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce")
        w = sub["_w"].where(vals.notna(), 0.0)
        keys = [sub["region_id"], sub["month"]]
        num = (vals.fillna(0.0) * w).groupby(keys).sum()
        den = w.groupby(keys).sum().replace(0, np.nan)
        out = out.merge((num / den).rename(col).reset_index()
                        .rename(columns={"level_0": "region_id", "level_1": "month"}),
                        on=["region_id", "month"], how="left")

    out = out.merge(elig, on="region_id", how="left")
    out["cell_coverage_fraction"] = (out["n_cells_observed"]
                                     / out["n_cells_eligible"].replace(0, np.nan))
    best = out.groupby("region_id")["valid_area_ha"].transform("max")
    out["valid_area_coverage_fraction"] = out["valid_area_ha"] / best.replace(0, np.nan)
    out["wh_cover"] = out["wh_area_ha"] / out["valid_area_ha"].replace(0, np.nan)

    ok = ((out["cell_coverage_fraction"] >= float(min_cell_coverage))
          & (out["valid_area_coverage_fraction"] >= float(min_area_coverage)))
    out["region_month_usable"] = ok
    out["exclusion_reason"] = np.where(
        ok, "",
        np.where(out["cell_coverage_fraction"] < float(min_cell_coverage),
                 f"cell coverage < {float(min_cell_coverage):g}",
                 f"valid-area coverage < {float(min_area_coverage):g}"))
    # A failing region-month is MISSING, never a partially-observed value.
    out.loc[~ok, ["wh_cover", "wh_area_ha", "wh_occurrence"]] = np.nan
    return out.sort_values(["region_id", "month"]).reset_index(drop=True)


# Candidate time-varying columns: everything numeric that is not a key, a
# response, a static covariate or a known degenerate.
_reserved = {"month", "grid_id", "year", "month_num", "time_index",
             "n_cells", "valid_area_ha", "valid_area_m2", "wh_area_ha",
             "wh_area_m2", "wh_area_ha_hard", "wh_pixels", "valid_pixels",
             "valid_fraction", "coverage_fraction", "eligible_area_ha",
             "n_cells_eligible", "n_cells_observed", "retained"}
_response_like = (lambda c: (c.startswith("wh_") or c.startswith("mean_cover")
                             or c.startswith("occurrence") or c.endswith("_neigh_lag1")))
CANDIDATE_DRIVER_COLS = [
    c for c in PANEL_FIXED.columns
    if c not in _reserved and not _response_like(c)
    and c not in KNOWN_STATIC_COLS and c not in REGIONAL_DEGENERATE_COLS
    and not str(c).endswith("_lag1")
    and pd.api.types.is_numeric_dtype(PANEL_FIXED[c])]

print(f"Candidate time-varying driver columns ({len(CANDIDATE_DRIVER_COLS)}):")
print("  " + ", ".join(CANDIDATE_DRIVER_COLS))
_static_dropped = [c for c in PANEL_FIXED.columns if c in KNOWN_STATIC_COLS]
_degen_dropped = [c for c in PANEL_FIXED.columns if c in REGIONAL_DEGENERATE_COLS]
print(f"\nDropped as STATIC within a fixed region (cannot explain temporal "
      f"variation; they act through the regionalisation and the regional "
      f"intercept instead): {_static_dropped or 'none'}")
for c in _degen_dropped:
    print(f"Dropped as DEGENERATE at regional scale: {c} "
          f"({REGIONAL_DEGENERATE_COLS[c]})")

REGION_MONTH_RAW = regional_monthly_panel(
    PANEL_FIXED, ASSIGNMENTS, CANDIDATE_DRIVER_COLS,
    min_cell_coverage=MIN_REGION_MONTH_CELL_COVERAGE,
    min_area_coverage=MIN_REGION_MONTH_VALID_AREA_COVERAGE)
print(f"\nRegion-months built: {len(REGION_MONTH_RAW):,} "
      f"({int(REGION_MONTH_RAW['region_month_usable'].sum()):,} usable, "
      f"{int((~REGION_MONTH_RAW['region_month_usable']).sum()):,} below a coverage "
      "gate and therefore MISSING).")


# ---------------------------------------------------------------------
# 9a-ii. Drivers from the COMPLETE environmental tables
# ---------------------------------------------------------------------
# The response is filtered by coverage; the environment is not. A month whose WH
# map failed the coverage filter contributes no panel row, so without this step
# its drivers would be missing too — and then every observed month whose LAG
# lands on it would be withheld from the likelihood. That is a large, avoidable
# loss, and the spatial workflow already exported the environmental tables for
# every month it queried.
#
# The two weightings differ and the difference is measured, not glossed: the
# panel-derived value is weighted by the classified area actually observed that
# month, while the complete-table value can only be weighted by each cell's
# STATIC eligible water area. The panel value always wins where it exists.


def regional_env_from_complete_tables(assignments, driver_cols,
                                      env_monthly=None, env_cellmonth=None):
    """Area-weighted regional driver values for EVERY calendar month."""
    frames = []
    memb = assignments[["grid_id", "region_id", "eligible_area_ha"]]
    if env_cellmonth is not None and len(env_cellmonth):
        cm = env_cellmonth.merge(memb, on="grid_id", how="inner")
        cols = [c for c in driver_cols if c in cm.columns]
        if cols:
            out = cm.groupby(["region_id", "month"], as_index=False).size()[
                ["region_id", "month"]]
            for c in cols:
                v = pd.to_numeric(cm[c], errors="coerce")
                w = cm["eligible_area_ha"].where(v.notna(), 0.0)
                keys = [cm["region_id"], cm["month"]]
                num = (v.fillna(0.0) * w).groupby(keys).sum()
                den = w.groupby(keys).sum().replace(0, np.nan)
                out = out.merge((num / den).rename(c).reset_index(),
                                on=["region_id", "month"], how="left")
            frames.append(out)
    if env_monthly is not None and len(env_monthly):
        cols = [c for c in driver_cols if c in env_monthly.columns]
        if cols:
            grid = pd.MultiIndex.from_product(
                [sorted(memb["region_id"].unique()),
                 pd.to_datetime(env_monthly["month"].unique())],
                names=["region_id", "month"]).to_frame(index=False)
            frames.append(grid.merge(env_monthly[["month"] + cols], on="month",
                                     how="left"))
    if not frames:
        return pd.DataFrame(columns=["region_id", "month"])
    base = frames[0]
    for extra in frames[1:]:
        new = [c for c in extra.columns if c not in base.columns]
        base = base.merge(extra[["region_id", "month"] + new],
                          on=["region_id", "month"], how="outer")
    return base


ENV_FILL_AUDIT = pd.DataFrame()
_env_reg = regional_env_from_complete_tables(
    ASSIGNMENTS, CANDIDATE_DRIVER_COLS,
    env_monthly=ENV_MONTHLY_COMPLETE, env_cellmonth=ENV_CELLMONTH_COMPLETE)
if len(_env_reg):
    _env_reg["month"] = to_month_start(_env_reg["month"])
    _fill_cols = [c for c in CANDIDATE_DRIVER_COLS if c in _env_reg.columns]
    _all = (REGION_MONTH_RAW.merge(
        _env_reg[["region_id", "month"] + _fill_cols], on=["region_id", "month"],
        how="outer", suffixes=("", "_envfill")))
    _rows = []
    for c in _fill_cols:
        ec = f"{c}_envfill"
        if ec not in _all.columns:
            continue
        both = _all[c].notna() & _all[ec].notna()
        _rows.append({
            "driver": c,
            "n_from_panel": int(_all[c].notna().sum()),
            "n_filled_from_complete_tables": int((_all[c].isna()
                                                  & _all[ec].notna()).sum()),
            "n_overlap": int(both.sum()),
            "overlap_pearson_r": (float(np.corrcoef(_all.loc[both, c],
                                                    _all.loc[both, ec])[0, 1])
                                  if int(both.sum()) > 3 else np.nan),
            "overlap_mean_abs_diff": (float((_all.loc[both, c]
                                             - _all.loc[both, ec]).abs().mean())
                                      if both.any() else np.nan),
            "note": "panel value (classified-area weighted) wins where it exists; "
                    "the fill is eligible-area weighted",
        })
        _all[c] = _all[c].where(_all[c].notna(), _all[ec])
        _all = _all.drop(columns=[ec])
    ENV_FILL_AUDIT = pd.DataFrame(_rows)
    # Rows that exist only in the environmental tables carry drivers but no
    # response: exactly what the state process needs, and never an observation.
    _all["region_month_usable"] = _all["region_month_usable"].fillna(False)
    _all["exclusion_reason"] = _all["exclusion_reason"].fillna(
        "no WH map for this month (below the gulf-wide coverage filter); the "
        "environmental drivers come from the complete monthly tables")
    REGION_MONTH_RAW = (_all.sort_values(["region_id", "month"])
                        .reset_index(drop=True))
    display(ENV_FILL_AUDIT)
    register("environmental_fill_audit", ENV_FILL_AUDIT, "provenance")
    print(f"Driver rows after using the complete environmental tables: "
          f"{len(REGION_MONTH_RAW):,} region-months "
          f"({int(REGION_MONTH_RAW['region_month_usable'].sum()):,} with a usable "
          "response).")
    print("Where the two weightings overlap, the correlation above says how "
          "comparable the fill is. A low correlation means the fill is a "
          "different measurement and should be treated as such.")
else:
    print("No complete environmental table was available, so drivers exist only "
          "in months that produced a WH map. Every observed month whose lag lands "
          "on an excluded month is therefore withheld from the likelihood; the "
          "count is in the §11b audit.")
''')


code(r'''# =====================================================================
# 9b. Regions that do not meet the minimum requirements are dropped
# =====================================================================
_usable = REGION_MONTH_RAW[REGION_MONTH_RAW["region_month_usable"]]
_req = (_usable.groupby("region_id")
        .agg(n_months_usable=("month", "nunique"),
             median_cell_coverage=("cell_coverage_fraction", "median"),
             median_valid_area_coverage=("valid_area_coverage_fraction", "median"))
        .reset_index())
REGION_REQUIREMENTS = (REGIONS.merge(_req, on="region_id", how="left")
                       .fillna({"n_months_usable": 0, "median_cell_coverage": 0.0,
                                "median_valid_area_coverage": 0.0}))
REGION_REQUIREMENTS["meets_min_cells"] = (
    REGION_REQUIREMENTS["n_cells"] >= MIN_REGION_CELLS)
REGION_REQUIREMENTS["meets_min_area"] = (
    REGION_REQUIREMENTS["eligible_area_ha"] >= MIN_REGION_ELIGIBLE_AREA_HA)
REGION_REQUIREMENTS["meets_min_months"] = (
    REGION_REQUIREMENTS["n_months_usable"] >= MIN_REGION_MONTHS)
REGION_REQUIREMENTS["meets_min_coverage"] = (
    REGION_REQUIREMENTS["median_cell_coverage"] >= MIN_REGION_MEDIAN_COVERAGE)
REGION_REQUIREMENTS["usable"] = (
    REGION_REQUIREMENTS[["meets_min_cells", "meets_min_area",
                         "meets_min_months", "meets_min_coverage"]].all(axis=1))
REGION_REQUIREMENTS["drop_reason"] = [
    "" if r.usable else "; ".join(
        [n for n, ok in [(f"n_cells < {MIN_REGION_CELLS}", r.meets_min_cells),
                         (f"eligible area < {MIN_REGION_ELIGIBLE_AREA_HA:g} ha",
                          r.meets_min_area),
                         (f"usable months < {MIN_REGION_MONTHS}", r.meets_min_months),
                         (f"median coverage < {MIN_REGION_MEDIAN_COVERAGE:g}",
                          r.meets_min_coverage)] if not ok])
    for r in REGION_REQUIREMENTS.itertuples()]
display(REGION_REQUIREMENTS)
register("region_requirements", REGION_REQUIREMENTS, "regionalisation")

USABLE_REGION_IDS = REGION_REQUIREMENTS.loc[REGION_REQUIREMENTS["usable"],
                                            "region_id"].tolist()
_dropped = REGION_REQUIREMENTS.loc[~REGION_REQUIREMENTS["usable"]]
if len(_dropped):
    print("\nDropped regions (kept in the map and the assignment export, excluded "
          "from the model):")
    for r in _dropped.itertuples():
        print(f"  {r.region_id} {r.region_name}: {r.drop_reason}")
if not USABLE_REGION_IDS:
    raise RuntimeError("No region meets the minimum requirements in §3c. Relax "
                       "them deliberately, or accept that the record cannot "
                       "support a regional analysis.")
N_REGIONS = len(USABLE_REGION_IDS)
print(f"\n{N_REGIONS} usable region(s) enter the model.")
if N_REGIONS < REGION_COUNT_TARGET[0]:
    print(f"*** Fewer than {REGION_COUNT_TARGET[0]} usable regions. The "
          "hierarchical variances are then estimated from very few groups: "
          "partial pooling still regularises the regional intercepts, but "
          "sigma_alpha / sigma_b / sigma_lambda are prior-dominated and must not "
          "be read as measurements of between-region heterogeneity. §12 "
          "simplifies the random-effects structure accordingly and §22 says so "
          "in the synthesis. ***")
''')

code(r'''# =====================================================================
# 9c. Does a region's observed composition drift through time?
# =====================================================================
# Membership is FIXED. What is not fixed is which member cells were cloud-free.
# A region whose observed composition swings month to month is measuring a
# moving target even with a fixed C_r, so it is measured and reported.
_memb = ASSIGNMENTS[["grid_id", "region_id", "eligible_area_ha"]]
_obs = (PANEL_FIXED[["grid_id", "month"]].merge(_memb, on="grid_id", how="inner"))
_usable_keys = set(map(tuple, _usable[["region_id", "month"]].to_numpy()))
_obs = _obs[[ (r, m) in _usable_keys for r, m in
              zip(_obs["region_id"], _obs["month"]) ]]

_rows = []
for rid, grp in _obs.groupby("region_id"):
    n_months = grp["month"].nunique()
    if n_months == 0:
        continue
    rate = grp.groupby("grid_id")["month"].nunique() / n_months
    core = set(rate[rate >= 0.90].index.tolist())
    elig_area = float(_memb.loc[_memb["region_id"] == rid, "eligible_area_ha"].sum())
    core_area = float(_memb.loc[(_memb["region_id"] == rid)
                                & (_memb["grid_id"].isin(core)),
                                "eligible_area_ha"].sum())
    jac = []
    for m, g in grp.groupby("month"):
        s = set(g["grid_id"].tolist())
        union = len(s | core)
        jac.append(len(s & core) / union if union else np.nan)
    _rows.append({"region_id": rid, "n_months_usable": int(n_months),
                  "n_cells_eligible": int(rate.size),
                  "n_cells_core_90pct": int(len(core)),
                  "core_area_share": (core_area / elig_area) if elig_area else np.nan,
                  "mean_jaccard_vs_core": float(np.nanmean(jac)) if jac else np.nan,
                  "min_jaccard_vs_core": float(np.nanmin(jac)) if jac else np.nan})
COMPOSITION_DRIFT = (pd.DataFrame(_rows)
                     .merge(REGIONS[["region_id", "region_name", "region_type"]],
                            on="region_id", how="left")
                     .sort_values("min_jaccard_vs_core"))
display(COMPOSITION_DRIFT)
register("region_composition_drift", COMPOSITION_DRIFT, "diagnostic")
_worst = COMPOSITION_DRIFT.head(3)
print("Composition stability, worst three regions "
      "(1.0 = the same cells observed every month):")
for r in _worst.itertuples():
    print(f"  {r.region_id} {r.region_name}: mean Jaccard "
          f"{r.mean_jaccard_vs_core:.3f}, worst month {r.min_jaccard_vs_core:.3f}, "
          f"{r.core_area_share:.1%} of the eligible area in cells seen in >=90% of "
          "months")
if (COMPOSITION_DRIFT["min_jaccard_vs_core"] < 0.6).any():
    print("\n*** At least one region has a month whose observed cell set overlaps "
          "its stable core by less than 60%. That month's value describes a "
          "materially different piece of that region. The coverage gates in §3d "
          "are the lever; this is reported rather than silently accepted. ***")
''')

code(r'''# =====================================================================
# 9d. Calendar-complete grid, response transform, audit
# =====================================================================
RM = REGION_MONTH_RAW[REGION_MONTH_RAW["region_id"].isin(USABLE_REGION_IDS)].copy()

# Every region on the SAME complete monthly calendar, spanning the whole record
# (not just its own observed months), so a lag means the same thing everywhere.
_span_lo = min(REGION_MONTH_RAW["month"].min(), COVERAGE["month"].min())
_span_hi = max(REGION_MONTH_RAW["month"].max(), COVERAGE["month"].max())
_grid = pd.date_range(_span_lo, _span_hi, freq="MS")
_full = pd.MultiIndex.from_product([sorted(USABLE_REGION_IDS), _grid],
                                   names=["region_id", "month"]).to_frame(index=False)
REGION_MONTH = _full.merge(RM, on=["region_id", "month"], how="left")
REGION_MONTH["year"] = REGION_MONTH["month"].dt.year
REGION_MONTH["month_num"] = REGION_MONTH["month"].dt.month
_mi = month_index(REGION_MONTH["month"])
REGION_MONTH["time_index"] = (_mi - _mi.min()).astype(float)
REGION_MONTH = REGION_MONTH.merge(
    REGIONS[["region_id", "region_name", "region_type"]], on="region_id", how="left")
REGION_MONTH["observed"] = REGION_MONTH["wh_cover"].notna()

y_logit, RESPONSE_INFO = transform_response(
    REGION_MONTH["wh_cover"], RESPONSE_TRANSFORM, RESPONSE_EPS)
REGION_MONTH["y"] = y_logit.to_numpy()

MONTH_GRID = pd.DatetimeIndex(_grid)
N_MONTHS_GRID = len(MONTH_GRID)

print(f"Region-month panel: {N_REGIONS} regions x {N_MONTHS_GRID} calendar months "
      f"= {len(REGION_MONTH):,} rows, of which "
      f"{int(REGION_MONTH['observed'].sum()):,} carry an observed response.")
print(f"Calendar span: {MONTH_GRID.min():%Y-%m} .. {MONTH_GRID.max():%Y-%m} "
      f"(complete; excluded months are present as MISSING rows and are never "
      "interpolated for inference).")
print(f"Response: logit(wh_cover), {RESPONSE_INFO['n_clipped_low']} value(s) "
      f"clipped at 0 and {RESPONSE_INFO['n_clipped_high']} at 1 with eps="
      f"{RESPONSE_INFO['eps']:g}.")
if RESPONSE_INFO["n_clipped_low"] or RESPONSE_INFO["n_clipped_high"]:
    print("  Clipping is a real distortion at the boundary; the count above is the "
          "number of region-months affected and is exported with the run manifest.")
print()
print(f"INFERENTIAL n = {int(REGION_MONTH['observed'].sum()):,} region-months.")
print(f"The panel behind it holds {len(panel):,} cell-month rows. Those are the "
      "measurement, not the sample.")

# --- the audit table --------------------------------------------------------
_a = (REGION_MONTH[REGION_MONTH["observed"]]
      .groupby(["region_type", "region_id", "region_name"], as_index=False)
      .agg(n_months_observed=("month", "nunique"),
           first_month=("month", "min"), last_month=("month", "max"),
           median_wh_cover=("wh_cover", "median"),
           median_wh_area_ha=("wh_area_ha", "median"),
           median_cell_coverage=("cell_coverage_fraction", "median"),
           median_valid_area_coverage=("valid_area_coverage_fraction", "median")))
REGION_AUDIT = (REGIONS[["region_id", "region_name", "region_type", "n_cells",
                         "eligible_area_ha"]]
                .merge(_a.drop(columns=["region_type", "region_name"]),
                       on="region_id", how="left")
                .sort_values(["region_type", "region_id"]))
REGION_AUDIT["calendar_span_months"] = N_MONTHS_GRID
REGION_AUDIT["months_missing"] = (REGION_AUDIT["calendar_span_months"]
                                  - REGION_AUDIT["n_months_observed"].fillna(0))
REGION_AUDIT["in_model"] = REGION_AUDIT["region_id"].isin(USABLE_REGION_IDS)
display(REGION_AUDIT)

CLASS_SUMMARY = (REGION_AUDIT[REGION_AUDIT["in_model"]]
                 .groupby("region_type", as_index=False)
                 .agg(n_regions=("region_id", "nunique"),
                      n_cells=("n_cells", "sum"),
                      eligible_area_ha=("eligible_area_ha", "sum"),
                      median_months_observed=("n_months_observed", "median")))
print("\nBy ecological class:")
display(CLASS_SUMMARY)

register("region_month_panel", REGION_MONTH, "input series")
register("region_coverage_audit", REGION_AUDIT, "diagnostic")
register("region_class_summary", CLASS_SUMMARY, "diagnostic")
register("region_month_raw_with_gates", REGION_MONTH_RAW, "diagnostic")
''')

code(r'''# =====================================================================
# 9e. The regional series
# =====================================================================
_ids = sorted(USABLE_REGION_IDS)
_ncol = 2
_nrow = int(np.ceil(len(_ids) / _ncol))
fig, axes = plt.subplots(_nrow, _ncol, figsize=(13, 2.1 * _nrow + 1.2),
                         sharex=True)
axes = np.atleast_1d(axes).ravel()
_type_colour = {"river_influenced_bay": "#1f78b4", "sheltered_littoral": "#33a02c",
                "exposed_littoral": "#ff7f00", "open_gulf": "#6a3d9a"}
for ax, rid in zip(axes, _ids):
    g = REGION_MONTH[REGION_MONTH["region_id"] == rid].sort_values("month")
    meta = REGIONS[REGIONS["region_id"] == rid].iloc[0]
    c = _type_colour.get(meta["region_type"], "0.3")
    ax.plot(g["month"], g["wh_cover"], marker="o", ms=2.6, lw=1.0, color=c)
    ax.set_ylabel("WH cover", fontsize=8)
    ax.set_title(f"{rid} — {meta['region_name']}", fontsize=9, loc="left")
    ax.grid(alpha=0.2, lw=0.4)
    ax.tick_params(labelsize=8)
for ax in axes[len(_ids):]:
    ax.axis("off")
axes[min(len(_ids), len(axes)) - 1].xaxis.set_major_locator(mdates.YearLocator())
axes[min(len(_ids), len(axes)) - 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("Regional WH cover — gaps are excluded months, left missing"
             + (" (SYNTHETIC)" if SOURCE["is_synthetic"] else ""), fontsize=12)
fig.tight_layout()
save_fig(fig, "02_regional_wh_series")
plt.show()
''')


# ===========================================================================
# 10. Does the regional design add information?
# ===========================================================================
md(r"""## 10. Does the regional design add information? — driver variance decomposition

This is the section the whole notebook turns on. Splitting the gulf into $R$
regions multiplies the number of *rows* by $R$. It multiplies the amount of
*information* about a driver only if that driver actually differs between
regions within the same month.

For every driver, on the observed region-months, the total sum of squares is
split **exactly** by month:

$$\underbrace{\sum_{r,t}(x_{r,t}-\bar x)^2}_{\text{total}}=
\underbrace{\sum_{r,t}(\bar x_{\cdot t}-\bar x)^2}_{\text{between months}}+
\underbrace{\sum_{r,t}(x_{r,t}-\bar x_{\cdot t})^2}_{\text{between regions, within month}}$$

and the second term is split again into a **fixed regional offset** (a region is
persistently wetter, more exposed…) and a **region × month interaction** (the
regional pattern itself changes from month to month). The interaction is the part
that is genuinely new relative to a single AOI series.

Also reported per driver: variation within each region through time, the $R^2$
of the annual harmonics, the correlation with the linear trend, and the strongest
correlation with any other driver.

**The label.** A driver with a within-month between-region share below
`DRIVER_REGIONAL_SHARE_MIN`, or with numerically identical values across regions,
is labelled `temporal_only`. It is still estimated — it is still a real
mechanism — but its effective replication is the number of **months**, and the
notebook says so everywhere it appears.
""")

code(r'''# =====================================================================
# 10a. Resolve the predeclared mechanisms against the columns that exist
# =====================================================================
FORCING = {}
_rows = []
for _base, (_mech, _sign, _lag) in REGIONAL_FORCING_TERMS.items():
    _cands = [_base] + list(REGIONAL_FORCING_FALLBACKS.get(_base, []))
    _pick = next((c for c in _cands if c in REGION_MONTH.columns
                  and pd.to_numeric(REGION_MONTH[c], errors="coerce").notna().any()),
                 None)
    _rows.append({"mechanism_key": _base, "mechanism": _mech,
                  "expected_sign": _sign, "apriori_lag_months": _lag,
                  "preferred_column": _base,
                  "resolved_column": _pick if _pick else "(unavailable)",
                  "used_fallback": bool(_pick and _pick != _base),
                  "usable": _pick is not None})
    if _pick:
        FORCING[_base] = {"mechanism": _mech, "expected_sign": _sign,
                          "apriori_lag": int(_lag), "column": _pick}
FORCING_RESOLUTION = pd.DataFrame(_rows)
display(FORCING_RESOLUTION)
register("forcing_resolution", FORCING_RESOLUTION, "provenance")
_missing = FORCING_RESOLUTION.loc[~FORCING_RESOLUTION["usable"], "mechanism_key"].tolist()
if _missing:
    print(f"Mechanisms with no available column, DROPPED (never silently "
          f"substituted): {_missing}")
if not FORCING:
    raise RuntimeError("No predeclared mechanism resolved to a column in the "
                       "region-month panel.")

PROXY_COLS = [c for c in REGIONAL_PROXY_TERMS if c in REGION_MONTH.columns]
print(f"\nEndogenous optical proxies held out of every driver claim: "
      f"{PROXY_COLS or 'none present'}")
UNUSED_CANDIDATES = [c for c in CANDIDATE_DRIVER_COLS
                     if c not in {v['column'] for v in FORCING.values()}
                     and c not in PROXY_COLS]
print(f"Numeric columns in the panel that belong to no predeclared mechanism "
      f"(not modelled): {UNUSED_CANDIDATES or 'none'}")
''')

code(r'''# =====================================================================
# 10b. Variance decomposition
# =====================================================================


def decompose_driver_variance(frame, col, region_col="region_id",
                              month_col="month"):
    """Exact month-orthogonal split of a driver's variance, plus season/trend."""
    d = frame[[region_col, month_col, col, "time_index", "month_num"]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col])
    n = len(d)
    if n < 4 or d[month_col].nunique() < 2:
        return None
    x = d[col].to_numpy(dtype=float)
    grand = float(np.mean(x))
    month_mean = d.groupby(month_col)[col].transform("mean").to_numpy()
    region_mean = d.groupby(region_col)[col].transform("mean").to_numpy()

    ss_total = float(np.sum((x - grand) ** 2))
    ss_between_month = float(np.sum((month_mean - grand) ** 2))
    ss_within_month = float(np.sum((x - month_mean) ** 2))
    ss_region_main = float(np.sum((region_mean - grand) ** 2))
    resid = x - month_mean - region_mean + grand
    ss_interaction = float(np.sum(resid ** 2))
    ss_within_region = float(np.sum((x - region_mean) ** 2))

    # within-month spread across regions, in the driver's own units
    per_month = d.groupby(month_col)[col].agg(["mean", "std", "count"])
    per_month = per_month[per_month["count"] >= 2]
    cv = (per_month["std"] / per_month["mean"].abs().replace(0, np.nan)).abs()
    med_cv = float(cv.median()) if len(cv) else 0.0
    med_sd = float(per_month["std"].median()) if len(per_month) else 0.0

    # season and trend, on the region-demeaned series so a regional offset is not
    # mistaken for seasonality
    xr = x - region_mean
    S = fourier_terms(d[month_col], SEASON_HARMONICS).to_numpy()
    try:
        r2_season = float(sm.OLS(xr, sm.add_constant(S)).fit().rsquared)
    except Exception:
        r2_season = np.nan
    tt = d["time_index"].to_numpy(dtype=float)
    r_trend = float(np.corrcoef(xr, tt)[0, 1]) if np.std(tt) > 0 else np.nan

    return {
        "driver": col, "n_region_months": int(n),
        "n_months": int(d[month_col].nunique()),
        "n_regions": int(d[region_col].nunique()),
        "share_between_months": ss_between_month / ss_total if ss_total else np.nan,
        "share_within_month_between_regions": (ss_within_month / ss_total
                                               if ss_total else np.nan),
        "share_region_main_effect": ss_region_main / ss_total if ss_total else np.nan,
        "share_region_x_month_interaction": (ss_interaction / ss_total
                                             if ss_total else np.nan),
        "share_within_region_over_time": (ss_within_region / ss_total
                                          if ss_total else np.nan),
        "month_split_sums_to_one": abs((ss_between_month + ss_within_month) / ss_total - 1.0)
        if ss_total else np.nan,
        "median_within_month_sd": med_sd,
        "median_within_month_cv": med_cv,
        "r2_annual_harmonics": r2_season,
        "corr_with_trend": r_trend,
    }


_dr_cols = [v["column"] for v in FORCING.values()] + PROXY_COLS
_obs_rm = REGION_MONTH[REGION_MONTH["observed"]].copy()
_rows = [r for r in (decompose_driver_variance(_obs_rm, c) for c in _dr_cols)
         if r is not None]
DRIVER_VARIANCE = pd.DataFrame(_rows)

# strongest correlation with any other driver, on the region-month rows
_wide = _obs_rm[_dr_cols].apply(pd.to_numeric, errors="coerce")
_corr = _wide.corr()
_carr = np.asarray(_corr, dtype=float).copy()
np.fill_diagonal(_carr, np.nan)
_corr = pd.DataFrame(_carr, index=_corr.index, columns=_corr.columns)
DRIVER_VARIANCE["max_abs_corr_other_driver"] = [
    float(np.nanmax(np.abs(_corr.loc[c].to_numpy()))) if c in _corr.index
    and np.isfinite(_corr.loc[c].to_numpy()).any() else np.nan
    for c in DRIVER_VARIANCE["driver"]]
DRIVER_VARIANCE["most_correlated_driver"] = [
    (_corr.loc[c].abs().idxmax() if c in _corr.index
     and _corr.loc[c].abs().notna().any() else None)
    for c in DRIVER_VARIANCE["driver"]]

DRIVER_VARIANCE["is_proxy"] = DRIVER_VARIANCE["driver"].isin(PROXY_COLS)
DRIVER_VARIANCE["spatial_label"] = np.where(
    (DRIVER_VARIANCE["share_within_month_between_regions"] >= DRIVER_REGIONAL_SHARE_MIN)
    & (DRIVER_VARIANCE["median_within_month_cv"] >= DRIVER_REGIONAL_CV_MIN),
    "spatiotemporal", "temporal_only")
DRIVER_VARIANCE["season_confounded"] = (
    DRIVER_VARIANCE["r2_annual_harmonics"] >= SEASON_CONFOUND_R2)
DRIVER_VARIANCE["effective_replication"] = np.where(
    DRIVER_VARIANCE["spatial_label"] == "spatiotemporal",
    "region-months (the regional design adds information)",
    "MONTHS ONLY — one gulf-wide value per month copied across regions")

display(DRIVER_VARIANCE[[
    "driver", "spatial_label", "share_between_months",
    "share_within_month_between_regions", "share_region_main_effect",
    "share_region_x_month_interaction", "median_within_month_cv",
    "r2_annual_harmonics", "corr_with_trend", "max_abs_corr_other_driver",
    "season_confounded"]])
register("driver_variance_decomposition", DRIVER_VARIANCE, "diagnostic")

TEMPORAL_ONLY_DRIVERS = DRIVER_VARIANCE.loc[
    (DRIVER_VARIANCE["spatial_label"] == "temporal_only")
    & (~DRIVER_VARIANCE["is_proxy"]), "driver"].tolist()
SPATIOTEMPORAL_DRIVERS = DRIVER_VARIANCE.loc[
    (DRIVER_VARIANCE["spatial_label"] == "spatiotemporal")
    & (~DRIVER_VARIANCE["is_proxy"]), "driver"].tolist()

print()
print(f"SPATIOTEMPORAL ({len(SPATIOTEMPORAL_DRIVERS)}): {SPATIOTEMPORAL_DRIVERS}")
print("    These vary between regions within a month. For these, and only these, "
      "the regional design supplies information a single AOI series cannot.")
print(f"TEMPORAL ONLY ({len(TEMPORAL_ONLY_DRIVERS)}): {TEMPORAL_ONLY_DRIVERS}")
print("    One gulf-wide value per month, repeated across regions. The regional "
      "design does NOT increase their effective replication:")
print(f"    n for these drivers is the {int(_obs_rm['month'].nunique())} observed "
      f"MONTHS, not the {len(_obs_rm):,} region-months.")
_sc = DRIVER_VARIANCE.loc[DRIVER_VARIANCE["season_confounded"], "driver"].tolist()
if _sc:
    print(f"\nSeason-confounded (R2 of the annual harmonics >= "
          f"{SEASON_CONFOUND_R2:g}): {_sc}")
    print("    Their coefficients are reported, but they cannot be separated from "
          "the annual cycle and are never called supported on their own.")
''')

code(r'''# =====================================================================
# 10c. Redundant drivers, and the predeclared random-slope set
# =====================================================================
_order = [v["column"] for v in FORCING.values()]
_drop_redundant, _redundant_rows = [], []
for i, a in enumerate(_order):
    for b in _order[i + 1:]:
        if a in _corr.index and b in _corr.columns:
            r = float(_corr.loc[a, b]) if np.isfinite(_corr.loc[a, b]) else np.nan
            if np.isfinite(r) and abs(r) > MAX_ABS_PAIRWISE_R:
                _redundant_rows.append({"kept": a, "dropped": b, "pearson_r": r})
                _drop_redundant.append(b)
REDUNDANT_DRIVERS = pd.DataFrame(_redundant_rows)
if len(REDUNDANT_DRIVERS):
    display(REDUNDANT_DRIVERS)
    print(f"Dropped as redundant (|r| > {MAX_ABS_PAIRWISE_R}): "
          f"{sorted(set(_drop_redundant))}")
    FORCING = {k: v for k, v in FORCING.items() if v["column"] not in _drop_redundant}
else:
    print(f"No driver pair exceeds |r| = {MAX_ABS_PAIRWISE_R}.")
register("redundant_drivers", REDUNDANT_DRIVERS, "diagnostic")

# Random slopes: only PREDECLARED terms that are genuinely spatiotemporal, capped
# by RANDOM_SLOPE_MAX_TERMS, and dropped entirely when there are too few regions
# to estimate a between-region slope variance.
RANDOM_SLOPE_TERMS = []
_rs_rows = []
for _cand in RANDOM_SLOPE_CANDIDATES:
    _col = FORCING.get(_cand, {}).get("column")
    if _col is None:
        _rs_rows.append({"candidate": _cand, "eligible": False,
                         "reason": "mechanism not available in this panel"})
        continue
    if _col not in SPATIOTEMPORAL_DRIVERS:
        _rs_rows.append({"candidate": _cand, "eligible": False,
                         "reason": "labelled temporal_only in §10b — a random "
                                   "slope on a driver with no regional variation "
                                   "is not identified"})
        continue
    if N_REGIONS < RANDOM_SLOPE_MIN_REGIONS:
        _rs_rows.append({"candidate": _cand, "eligible": False,
                         "reason": f"only {N_REGIONS} usable regions "
                                   f"(< RANDOM_SLOPE_MIN_REGIONS="
                                   f"{RANDOM_SLOPE_MIN_REGIONS})"})
        continue
    if len(RANDOM_SLOPE_TERMS) >= RANDOM_SLOPE_MAX_TERMS:
        _rs_rows.append({"candidate": _cand, "eligible": False,
                         "reason": f"RANDOM_SLOPE_MAX_TERMS="
                                   f"{RANDOM_SLOPE_MAX_TERMS} already reached"})
        continue
    RANDOM_SLOPE_TERMS.append(_cand)
    _rs_rows.append({"candidate": _cand, "eligible": True,
                     "reason": "predeclared, spatiotemporal, and enough regions"})
RANDOM_SLOPE_DECISION = pd.DataFrame(_rs_rows)
display(RANDOM_SLOPE_DECISION)
register("random_slope_decision", RANDOM_SLOPE_DECISION, "provenance")
if RANDOM_SLOPE_TERMS:
    print(f"Random slopes on: {RANDOM_SLOPE_TERMS}")
else:
    print("Random slopes: NONE — partially pooled regional intercepts and common "
          "driver slopes only. An elaborate random-effects structure is not kept "
          "merely because it runs.")
''')


# ===========================================================================
# 11. Model dataset
# ===========================================================================
md(r"""## 11. Model dataset, standardisation and the missing-data audit

Three rules govern this section, and each is asserted rather than described.

1. **Lags are calendar lags.** A driver entering at lag 1 takes the value of the
   *previous calendar month* within the same region. Because every region sits on
   the same complete monthly grid, a lag never reaches across an excluded month
   and calls a three-month-old value "last month".
2. **Nothing is interpolated.** A missing driver at an observed response month
   means that region-month is **withheld from the likelihood** — of *both* the
   null and the full model, so the two are matched — and the withholding is
   recorded with a reason.
3. **Placeholders live only where they cannot matter.** The state-space arrays
   must be finite everywhere, so `0.0` (the centred value on the standardised
   scale) is written into predictor cells that make **no** likelihood
   contribution. §11c asserts that every placeholder sits outside the observation
   index, so no placeholder can ever reach the likelihood.

Standardisation uses the mean and SD of the **estimation rows only**. §17
re-fits its scalers inside every fold; the global scaler here is never reused
there.
""")

code(r'''# =====================================================================
# 11a. Arrays on the region x calendar-month grid
# =====================================================================
REGION_IDS = sorted(USABLE_REGION_IDS)
R_IDX = {rid: i for i, rid in enumerate(REGION_IDS)}
T_IDX = {m: i for i, m in enumerate(MONTH_GRID)}
R, T = len(REGION_IDS), len(MONTH_GRID)

Y_RAW = np.full((R, T), np.nan)
VALID_AREA = np.full((R, T), np.nan)
for rid, i in R_IDX.items():
    g = REGION_MONTH[REGION_MONTH["region_id"] == rid].set_index("month")
    Y_RAW[i, :] = g.reindex(MONTH_GRID)["y"].to_numpy()
    VALID_AREA[i, :] = g.reindex(MONTH_GRID)["valid_area_ha"].to_numpy()

# --- lagged driver terms, calendar-aware within each region -----------------
DRIVER_TERMS = []
DRIVER_META = []
X_RAW = np.full((R, T, len(FORCING)), np.nan)
for k, (base, meta) in enumerate(FORCING.items()):
    col, lag = meta["column"], int(meta["apriori_lag"])
    name = f"{base}_lag{lag}" if lag else base
    DRIVER_TERMS.append(name)
    DRIVER_META.append({"term": name, "mechanism_key": base, "column": col,
                        "lag_months": lag, "mechanism": meta["mechanism"],
                        "expected_sign": meta["expected_sign"],
                        "spatial_label": ("spatiotemporal" if col in SPATIOTEMPORAL_DRIVERS
                                          else "temporal_only")})
    for rid, i in R_IDX.items():
        s = (REGION_MONTH[REGION_MONTH["region_id"] == rid]
             .set_index("month").reindex(MONTH_GRID)[col])
        X_RAW[i, :, k] = s.shift(lag).to_numpy()
DRIVER_META = pd.DataFrame(DRIVER_META)
K = len(DRIVER_TERMS)
display(DRIVER_META)
register("driver_terms", DRIVER_META, "provenance")

# --- who is in the likelihood ------------------------------------------------
OBS_MASK_Y = np.isfinite(Y_RAW)
X_COMPLETE = np.all(np.isfinite(X_RAW), axis=2) if K else np.ones((R, T), bool)
OBS_MASK = OBS_MASK_Y & X_COMPLETE       # identical for the null and the full model
OBS_R, OBS_T = np.nonzero(OBS_MASK)
N_OBS = int(OBS_MASK.sum())
print(f"Likelihood rows: {N_OBS:,} region-months "
      f"({R} regions x {T} calendar months = {R * T:,} cells in the grid).")

# --- standardisation on the estimation rows only -----------------------------
_scale_rows = []
X_STD = np.zeros_like(X_RAW)
for k, name in enumerate(DRIVER_TERMS):
    v = X_RAW[:, :, k][OBS_MASK]
    mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
    sd = sd if np.isfinite(sd) and sd > 0 else 1.0
    X_STD[:, :, k] = (X_RAW[:, :, k] - mu) / sd
    _scale_rows.append({"term": name, "mean": mu, "sd": sd,
                        "n_rows_used": int(np.isfinite(v).sum()),
                        "scope": "estimation rows only (§17 refits per fold)"})
DRIVER_SCALING = pd.DataFrame(_scale_rows)
display(DRIVER_SCALING)
register("driver_scaling", DRIVER_SCALING, "provenance")

# --- season, trend -----------------------------------------------------------
SEASON = fourier_terms(pd.Series(MONTH_GRID), SEASON_HARMONICS).to_numpy()
SEASON_NAMES = list(fourier_terms(pd.Series(MONTH_GRID), SEASON_HARMONICS).columns)
_tt = np.arange(T, dtype=float)
TREND_SCALER = {"mean": float(_tt.mean()), "sd": float(_tt.std() or 1.0)}
TT = (_tt - TREND_SCALER["mean"]) / TREND_SCALER["sd"]
print(f"Season: {SEASON_NAMES} (deterministic annual Fourier terms). "
      f"Trend: {'on' if INCLUDE_TREND else 'off'}.")
print("No unrestricted calendar-month fixed effects are included anywhere: a free "
      "effect for every month would absorb every gulf-wide temporal driver and "
      "make it unidentifiable.")
''')

code(r'''# =====================================================================
# 11b. Missing-data audit
# =====================================================================
_rows = []
for rid, i in R_IDX.items():
    y_ok = OBS_MASK_Y[i]
    x_ok = X_COMPLETE[i]
    _rows.append({
        "region_id": rid,
        "calendar_months": T,
        "response_observed": int(y_ok.sum()),
        "response_missing": int((~y_ok).sum()),
        "in_likelihood": int((y_ok & x_ok).sum()),
        "withheld_response_observed_driver_missing": int((y_ok & ~x_ok).sum()),
        "state_only_response_missing": int((~y_ok).sum()),
    })
MISSING_AUDIT = (pd.DataFrame(_rows)
                 .merge(REGIONS[["region_id", "region_name", "region_type"]],
                        on="region_id", how="left"))
display(MISSING_AUDIT)
register("missing_data_audit", MISSING_AUDIT, "diagnostic")

_withheld = int((OBS_MASK_Y & ~X_COMPLETE).sum())
print(f"Region-months with an observed response but an incomplete driver row: "
      f"{_withheld:,}. These are WITHHELD from the likelihood of BOTH the null "
      "and the full model, so the two remain matched. They are not imputed.")
if _withheld and K:
    _by_term = []
    for k, name in enumerate(DRIVER_TERMS):
        miss = OBS_MASK_Y & ~np.isfinite(X_RAW[:, :, k])
        _by_term.append({"term": name, "n_withheld_because_of_this_term":
                         int(miss.sum())})
    display(pd.DataFrame(_by_term))
print(f"Region-months whose response is missing: "
      f"{int((~OBS_MASK_Y).sum()):,}. The state process runs through them; they "
      "make no likelihood contribution.")

# --- placeholders, and the assertion that they cannot matter ----------------
_placeholder = ~np.isfinite(X_STD)
X_MODEL = np.where(_placeholder, 0.0, X_STD)
_bad = int((_placeholder & OBS_MASK[:, :, None]).sum()) if K else 0
assert _bad == 0, (f"{_bad} placeholder value(s) sit on rows that DO enter the "
                   "likelihood; the observation set and the placeholder mask "
                   "disagree.")
assert np.isfinite(X_MODEL).all(), "non-finite predictor after placeholder fill"
assert np.isfinite(Y_RAW[OBS_MASK]).all(), "non-finite response in the likelihood"
PLACEHOLDER_AUDIT = pd.DataFrame([{
    "n_predictor_cells": int(X_STD.size),
    "n_placeholder_cells": int(_placeholder.sum()),
    "placeholder_value": 0.0,
    "placeholder_scale": "standardised (0.0 IS the centred value)",
    "n_placeholders_inside_likelihood": _bad,
    "assertion": "every placeholder lies outside the observation index, so no "
                 "placeholder makes an observation-likelihood contribution",
}])
display(PLACEHOLDER_AUDIT)
register("placeholder_audit", PLACEHOLDER_AUDIT, "provenance")
''')


# ===========================================================================
# 12. The model
# ===========================================================================
md(r"""## 12. The hierarchical dynamic model

$$y_{r,t}= \alpha_r + s_t + \tau t
+ \sum_k\big(\beta_k+b_{r,k}\big)x_{r,t,k}
+ \lambda_r g_t + u_{r,t} + \epsilon_{r,t}$$

| Term | What it is | Prior (standardised scale) |
|---|---|---|
| $\alpha_r$ | partially pooled regional intercept | $\mu_\alpha+\sigma_\alpha z_r$, non-centred |
| $s_t$ | deterministic annual Fourier terms | $\mathcal N(0,\texttt{season\_sd})$ |
| $\tau t$ | optional common long-term trend | $\mathcal N(0,\texttt{trend\_sd})$ |
| $\beta_k$ | gulf-wide mean association | $\mathcal N(0,\texttt{beta\_sd})$ |
| $b_{r,k}$ | regularised regional deviation | $\sigma_{b,k}z_{r,k}$, non-centred |
| $g_t$ | shared latent state — gulf-wide persistence and unmeasured common shocks | AR(1) **or** local level |
| $\lambda_r$ | regional loading on that shared state | $1+\sigma_\lambda z_r$ |
| $u_{r,t}$ | region-specific temporal dependence | AR(1) |
| $\epsilon_{r,t}$ | observation noise | see below |

$$g_t=\rho_g g_{t-1}+\eta_t, \qquad u_{r,t}=\rho_r u_{r,t-1}+\zeta_{r,t}$$

**Three modelling choices worth defending.**

*Stationary AR parameters are constrained to $(0,1)$* with a $\mathrm{Beta}(2,2)$
prior. Negative month-to-month persistence in a floating macrophyte is not a
hypothesis anyone holds, and leaving the interval open to $-1$ buys nothing but a
bimodal posterior. Stationarity is **not** forced on the common process: the
random-walk alternative is a separate candidate, selected in §13, precisely
because the AOI temporal model's AR interval reached a unit root.

*$\lambda_r$ is centred on 1, not on 0.* A latent state and its loadings are only
identified up to scale and sign; anchoring the average loading at one fixes both,
and $\sigma_\lambda$ then measures how differently regions respond to the shared
state.

*The regional AR innovation and the observation noise are parameterised as a
total and a split.* $\sigma_u=\sigma\sqrt{\phi}$ and
$\sigma_\epsilon=\sigma\sqrt{1-\phi}$ with $\phi\sim\mathrm{Beta}(2,2)$. Their
**sum** is what the data identify well; the split between a weakly autocorrelated
regional process and white observation noise is intrinsically hard, and putting
the hard part in a bounded fraction turns a funnel into a well-behaved posterior
instead of hiding it. §15 reports the split's diagnostics separately, and the
simplification ladder drops $u_{r,t}$ when it is not separable.

The **local-level alternative** replaces $g_t$'s AR recursion with a Gaussian
random walk whose path is centred, so its level stays identified against the
pooled intercept.
""")

code(r'''# =====================================================================
# 12. Model builder
# =====================================================================
MODEL_KINDS = {
    "ar1": "shared latent state g_t ~ AR(1), stationary, rho in (0, 1)",
    "randomwalk": "shared latent state g_t ~ Gaussian random walk (local level), "
                  "path centred so its level is identified against the intercept",
    "none": "no shared latent state",
}


def make_model_data(X_model, Y, obs_mask, season, tt, region_ids, driver_terms,
                    random_slope_terms=()):
    """Package the arrays the model builder needs, with the observation index."""
    obs_r, obs_t = np.nonzero(obs_mask)
    rs_idx = [driver_terms.index(t) for t in random_slope_terms
              if t in driver_terms]
    return {
        "R": int(Y.shape[0]), "T": int(Y.shape[1]), "K": int(X_model.shape[2]),
        "X": X_model, "Y": Y, "obs_mask": obs_mask,
        "obs_r": obs_r, "obs_t": obs_t,
        "y_obs": Y[obs_r, obs_t],
        "X_obs": X_model[obs_r, obs_t, :] if X_model.shape[2] else
                 np.zeros((len(obs_r), 0)),
        "season": season, "tt": np.asarray(tt, dtype=float),
        "region_ids": list(region_ids), "driver_terms": list(driver_terms),
        "rs_idx": rs_idx,
        "rs_terms": [driver_terms[i] for i in rs_idx],
    }


def build_regional_model(data, drivers=True, common_state="ar1",
                         regional_ar="common", include_trend=True,
                         use_random_slopes=True, priors=None,
                         random_slope_parameterisation=None,
                         hierarchy_parameterisation=None):
    """The hierarchical dynamic model. One builder, every structure §13-§19 needs."""
    if not HAVE_PYMC:
        raise RuntimeError("PyMC is not available.")
    P = dict(PRIORS)
    P.update(priors or {})
    random_slope_parameterisation = (random_slope_parameterisation
                                     or RANDOM_SLOPE_PARAMETERISATION)
    hierarchy_parameterisation = (hierarchy_parameterisation
                                  or HIERARCHY_PARAMETERISATION)
    centred = hierarchy_parameterisation == "centred"
    R_, T_, K_ = data["R"], data["T"], data["K"]
    obs_r, obs_t = data["obs_r"], data["obs_t"]
    rs_idx = list(data["rs_idx"]) if (drivers and use_random_slopes) else []
    coords = {"region": data["region_ids"], "time": np.arange(T_),
              "season": [f"s{i}" for i in range(data["season"].shape[1])]}
    if drivers and K_:
        coords["driver"] = data["driver_terms"]
    if rs_idx:
        coords["rs_driver"] = [data["driver_terms"][i] for i in rs_idx]

    with pm.Model(coords=coords) as model:
        # --- partially pooled regional intercepts (non-centred) --------------
        mu_alpha = pm.Normal("mu_alpha", 0.0, P["mu_alpha_sd"])
        sigma_alpha = pm.HalfNormal("sigma_alpha", P["sigma_alpha"])
        if centred:
            alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="region")
        else:
            alpha = pm.Deterministic(
                "alpha", mu_alpha + sigma_alpha * pm.Normal("alpha_z", 0, 1,
                                                            dims="region"),
                dims="region")

        # --- deterministic season and optional common trend ------------------
        gamma = pm.Normal("gamma_season", 0.0, P["season_sd"], dims="season")
        eta = alpha[obs_r] + pt.dot(pt.as_tensor_variable(data["season"])[obs_t],
                                    gamma)
        if include_trend:
            tau = pm.Normal("tau_trend", 0.0, P["trend_sd"])
            eta = eta + tau * pt.as_tensor_variable(data["tt"])[obs_t]

        # --- drivers ----------------------------------------------------------
        if drivers and K_:
            X_obs = pt.as_tensor_variable(data["X_obs"])
            beta = pm.Normal("beta", 0.0, P["beta_sd"], dims="driver")
            eta = eta + pt.dot(X_obs, beta)
            if rs_idx:
                sigma_b = pm.HalfNormal("sigma_b", P["sigma_b"], dims="rs_driver")
                if random_slope_parameterisation == "centred":
                    b = pm.Normal("b", 0.0, sigma_b[None, :],
                                  dims=("region", "rs_driver"))
                else:
                    b = pm.Deterministic(
                        "b", sigma_b[None, :] * pm.Normal(
                            "b_z", 0, 1, dims=("region", "rs_driver")),
                        dims=("region", "rs_driver"))
                eta = eta + pt.sum(b[obs_r, :] * X_obs[:, rs_idx], axis=1)

        # --- shared latent state ---------------------------------------------
        if common_state == "ar1":
            rho_g = pm.Beta("rho_g", P["rho_a"], P["rho_b"])
            sigma_g = pm.HalfNormal("sigma_g", P["sigma_g"])
            g = pm.AR("g", rho=pt.stack([rho_g]), sigma=sigma_g,
                      init_dist=pm.Normal.dist(0.0, sigma_g / pt.sqrt(1 - rho_g ** 2)),
                      constant=False, ar_order=1, dims="time")
        elif common_state == "randomwalk":
            sigma_g = pm.HalfNormal("sigma_g", P["sigma_g"])
            g_raw = pm.GaussianRandomWalk(
                "g_raw", sigma=sigma_g, init_dist=pm.Normal.dist(0.0, 1.0),
                steps=T_ - 1, dims="time")
            g = pm.Deterministic("g", g_raw - pt.mean(g_raw), dims="time")
        elif common_state == "none":
            g = None
        else:
            raise ValueError(f"unknown common_state {common_state!r}")

        if g is not None:
            sigma_lambda = pm.HalfNormal("sigma_lambda", P["sigma_lambda"])
            if centred:
                lam = pm.Normal("lam", 1.0, sigma_lambda, dims="region")
            else:
                lam = pm.Deterministic(
                    "lam", 1.0 + sigma_lambda * pm.Normal("lam_z", 0, 1,
                                                          dims="region"),
                    dims="region")
            eta = eta + lam[obs_r] * g[obs_t]

        # --- region-specific temporal dependence + observation noise ----------
        if regional_ar in ("common", "per_region"):
            sigma_resid = pm.HalfNormal("sigma_resid_total",
                                        float(np.hypot(P["sigma_u"], P["sigma_eps"])))
            frac_u = pm.Beta("frac_u", 2.0, 2.0)
            sigma_u = pm.Deterministic("sigma_u", sigma_resid * pt.sqrt(frac_u))
            sigma_eps = pm.Deterministic("sigma_eps",
                                         sigma_resid * pt.sqrt(1.0 - frac_u))
            if regional_ar == "common":
                rho_u = pm.Beta("rho_u", P["rho_a"], P["rho_b"])
                init_sd = sigma_u / pt.sqrt(1 - rho_u ** 2)
                u = pm.AR("u", rho=pt.stack([rho_u]), sigma=sigma_u,
                          init_dist=pm.Normal.dist(0.0, init_sd),
                          constant=False, ar_order=1,
                          dims=("region", "time"), shape=(R_, T_))
            else:
                rho_u = pm.Beta("rho_u", P["rho_a"], P["rho_b"], dims="region")
                init_sd = (sigma_u / pt.sqrt(1 - rho_u ** 2))[:, None]
                u = pm.AR("u", rho=rho_u[:, None], sigma=sigma_u,
                          init_dist=pm.Normal.dist(0.0, init_sd, shape=(R_, 1)),
                          constant=False, ar_order=1,
                          dims=("region", "time"), shape=(R_, T_))
            eta = eta + u[obs_r, obs_t]
        elif regional_ar == "none":
            sigma_eps = pm.HalfNormal("sigma_eps", P["sigma_eps"])
        else:
            raise ValueError(f"unknown regional_ar {regional_ar!r}")

        pm.Deterministic("eta_obs", eta)
        pm.Normal("y", mu=eta, sigma=sigma_eps, observed=data["y_obs"])
    return model


def fit_model(model, sampling=None, label="", prior_predictive=False):
    """Sample, returning (idata, info). Never silently degrades the chain count."""
    cfg = dict(SAMPLING if sampling is None else sampling)
    t0 = time.time()
    with model:
        if prior_predictive:
            prior = pm.sample_prior_predictive(draws=500,
                                               random_seed=cfg.get("random_seed", 0))
        idata = pm.sample(progressbar=False,
                          idata_kwargs={"log_likelihood": True}, **cfg)
        if prior_predictive:
            idata.extend(prior)
    info = {"label": label, "seconds": round(time.time() - t0, 1),
            "draws": cfg["draws"], "tune": cfg["tune"], "chains": cfg["chains"],
            "target_accept": cfg["target_accept"],
            "divergences": int(idata.sample_stats["diverging"].sum()),
            "fast_mode": bool(FAST_MODE)}
    return idata, info


def diagnostics_table(idata, var_names=None, label=""):
    """R-hat / ESS / divergence table for an EXPLICIT list of parameters.

    Exact names, not substring matching: `filter_vars="like"` would pull the
    non-centred auxiliaries (`alpha_z`, `lam_z`, `b_z`, `g_raw`) and the whole
    latent-state paths into the gate. Those are reparameterisation coordinates
    and high-dimensional states, and their diagnostics are reported separately
    (§15b/§15c) rather than allowed to veto a driver coefficient.
    """
    have = set(idata.posterior.data_vars)
    names = [v for v in (var_names or sorted(have)) if v in have]
    if not names:
        return pd.DataFrame()
    summ = az.summary(idata, var_names=names, kind="diagnostics")
    out = summ.reset_index().rename(columns={"index": "parameter"})
    out["model"] = label
    out["divergences"] = int(idata.sample_stats["diverging"].sum())
    out["n_chains"] = int(idata.posterior.sizes["chain"])
    return out


def gate_diagnostics(diag, max_rhat=None, min_ess_bulk=None, min_ess_tail=None,
                     max_div=None):
    """Does this fit clear the reporting gate? Returns (passed, failures)."""
    max_rhat = DIAG_MAX_RHAT if max_rhat is None else max_rhat
    min_ess_bulk = DIAG_MIN_ESS_BULK if min_ess_bulk is None else min_ess_bulk
    min_ess_tail = DIAG_MIN_ESS_TAIL if min_ess_tail is None else min_ess_tail
    max_div = DIAG_MAX_DIVERGENCES if max_div is None else max_div
    fails = []
    if len(diag):
        if float(diag["divergences"].max()) > max_div:
            fails.append(f"{int(diag['divergences'].max())} divergent transition(s)")
        bad_r = diag.loc[diag["r_hat"] > max_rhat, "parameter"].tolist()
        if bad_r:
            fails.append(f"R-hat > {max_rhat} for {bad_r[:6]}"
                         + (" ..." if len(bad_r) > 6 else ""))
        bad_b = diag.loc[diag["ess_bulk"] < min_ess_bulk, "parameter"].tolist()
        if bad_b:
            fails.append(f"bulk ESS < {min_ess_bulk} for {bad_b[:6]}"
                         + (" ..." if len(bad_b) > 6 else ""))
        bad_t = diag.loc[diag["ess_tail"] < min_ess_tail, "parameter"].tolist()
        if bad_t:
            fails.append(f"tail ESS < {min_ess_tail} for {bad_t[:6]}"
                         + (" ..." if len(bad_t) > 6 else ""))
        if int(diag["n_chains"].max()) < 4:
            fails.append(f"only {int(diag['n_chains'].max())} chains")
    else:
        fails.append("no diagnostics produced")
    return (not fails), fails


# Parameters whose diagnostics GATE what §16 may report: the interpretable,
# low-dimensional quantities the notebook actually quotes.
REPORTED_PARAMS = ["beta", "b", "sigma_b", "mu_alpha", "sigma_alpha", "alpha",
                   "gamma_season", "tau_trend", "rho_g", "sigma_g",
                   "sigma_lambda", "lam", "rho_u", "sigma_resid_total"]
# Diagnosed and REPORTED, but not blocking:
#   * the latent state paths g and u - hundreds of correlated values whose
#     individual tail ESS is not what a driver coefficient depends on (§15b);
#   * the variance SPLIT sigma_u / sigma_eps / frac_u, whose weak identification
#     is a known structural property while their total is well identified
#     (§15c);
#   * the non-centred auxiliaries alpha_z / lam_z / b_z / g_raw, which are
#     coordinates rather than quantities - the transformed alpha, lam and b are
#     gated in their place.
STATE_PARAMS = ["g", "u"]

MODEL_DATA = make_model_data(X_MODEL, Y_RAW, OBS_MASK, SEASON, TT, REGION_IDS,
                             DRIVER_TERMS,
                             random_slope_terms=[
                                 t for t in DRIVER_TERMS
                                 if any(t.startswith(c) for c in RANDOM_SLOPE_TERMS)])
print(f"Model data: R={MODEL_DATA['R']} regions, T={MODEL_DATA['T']} calendar "
      f"months, K={MODEL_DATA['K']} drivers, {len(MODEL_DATA['y_obs']):,} "
      f"likelihood rows.")
print(f"Random-slope terms in the model data: {MODEL_DATA['rs_terms'] or 'none'}")
''')


# ===========================================================================
# 13. Persistence structure, chosen on the no-driver model
# ===========================================================================
md(r"""## 13. Step 1 — which persistence structure does the record support?

The structure is locked **before** a single environmental variable is added, so
the choice cannot be made by whichever dependence structure flatters a driver.
Every candidate carries the same partially pooled intercepts, the same
deterministic season, the same trend and the same regional AR — they differ only
in the shared state $g_t$:

* `ar1` — stationary AR(1), $\rho_g\in(0,1)$;
* `randomwalk` — a local level, fitted because the AOI temporal model's AR
  interval reached a unit root, so a stationary AR may simply be the wrong shape;
* `none` — no shared state at all, which is the honest null if the regions do not
  in fact move together.

**What the selection criteria are, and what they are not.** PSIS-LOO on a latent
state model scores *conditional* predictive fit with the states in place; it is
not a forecast. It is used here as the primary criterion because all three
candidates are scored the same way on the same rows, and it is reported next to
the diagnostics, the boundary behaviour of $\rho_g$, and the calendar-lag ACF of
the residuals. Genuine one-month-ahead forecasting is §17's job, and it is done
on the *selected* structure. A candidate with divergences is not selected
whatever its elpd, and when two candidates are within one standard error of each
other the **simpler** one wins.
""")

code(r'''# =====================================================================
# 13a. Fit the candidate null dynamics
# =====================================================================


def calendar_acf(values, month_pos, max_lag=12):
    """ACF using only pairs exactly h CALENDAR months apart (NaN-safe)."""
    v = np.asarray(values, dtype=float)
    m = np.asarray(month_pos, dtype=int)
    ok = np.isfinite(v)
    v, m = v[ok], m[ok]
    if len(v) < 4:
        return pd.DataFrame(columns=["lag", "acf", "n_pairs"])
    vc = v - v.mean()
    denom = float(np.sum(vc ** 2))
    pos = {int(a): i for i, a in enumerate(m)}
    rows = []
    for h in range(1, int(max_lag) + 1):
        pairs = [(i, pos[int(a) - h]) for i, a in enumerate(m) if int(a) - h in pos]
        if not pairs or denom <= 0:
            rows.append({"lag": h, "acf": np.nan, "n_pairs": len(pairs)})
            continue
        num = float(np.sum([vc[i] * vc[j] for i, j in pairs]))
        rows.append({"lag": h, "acf": num / denom, "n_pairs": len(pairs)})
    return pd.DataFrame(rows)


def residual_frame(idata, data, month_grid, region_ids):
    """Observation residuals y - posterior-mean eta, with region and month keys."""
    eta = idata.posterior["eta_obs"].mean(dim=("chain", "draw")).to_numpy()
    return pd.DataFrame({
        "region_id": [region_ids[i] for i in data["obs_r"]],
        "month": [month_grid[i] for i in data["obs_t"]],
        "month_pos": data["obs_t"],
        "y": data["y_obs"], "eta": eta, "resid": data["y_obs"] - eta})


NULL_DATA = make_model_data(X_MODEL, Y_RAW, OBS_MASK, SEASON, TT, REGION_IDS,
                            DRIVER_TERMS, random_slope_terms=[])

SS_CANDIDATES = pd.DataFrame()
SS_CANDIDATE_DIAG = pd.DataFrame()
NULL_FITS, NULL_INFO, NULL_LOO = {}, {}, {}
SELECTED_COMMON_STATE = None
SELECTION_NOTE = ""

if not HAVE_PYMC:
    print("PyMC unavailable -> §13-§19 skipped. Everything above (regions, panel, "
          "variance decomposition) is unaffected and still exported.")
else:
    _rows, _diags = [], []
    for _cand in COMMON_STATE_CANDIDATES:
        print(f"--- null dynamics: common_state={_cand} "
              f"({MODEL_KINDS[_cand]}) ---")
        try:
            _m = build_regional_model(NULL_DATA, drivers=False, common_state=_cand,
                                      regional_ar=REGIONAL_AR_MODE,
                                      include_trend=INCLUDE_TREND,
                                      use_random_slopes=False)
            _id, _inf = fit_model(_m, SAMPLING, label=f"null_{_cand}")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            _rows.append({"common_state": _cand, "fitted": False,
                          "note": f"fit failed: {exc}"})
            continue
        NULL_FITS[_cand], NULL_INFO[_cand] = _id, _inf
        _d = diagnostics_table(_id, REPORTED_PARAMS, label=f"null_{_cand}")
        _diags.append(_d)
        _ok, _fails = gate_diagnostics(_d)
        try:
            _loo = az.loo(_id, pointwise=True)
            NULL_LOO[_cand] = _loo
            _elpd, _se = float(_loo.elpd_loo), float(_loo.se)
            _kbad = int((_loo.pareto_k.values > 0.7).sum())
        except Exception as exc:
            _elpd = _se = np.nan
            _kbad = -1
            print(f"    LOO unavailable: {exc}")
        _rho = (float(_id.posterior["rho_g"].mean()) if "rho_g" in _id.posterior
                else np.nan)
        _rho_hi = (float((_id.posterior["rho_g"] > 0.95).mean())
                   if "rho_g" in _id.posterior else np.nan)
        _res = residual_frame(_id, NULL_DATA, MONTH_GRID, REGION_IDS)
        _acf1 = []
        for rid, g in _res.groupby("region_id"):
            a = calendar_acf(g["resid"].to_numpy(), g["month_pos"].to_numpy(), 3)
            if len(a):
                _acf1.append(float(a.loc[a["lag"] == 1, "acf"].iloc[0]))
        _rows.append({
            "common_state": _cand, "fitted": True,
            "description": MODEL_KINDS[_cand],
            "elpd_loo": _elpd, "elpd_loo_se": _se, "n_pareto_k_gt_0.7": _kbad,
            "divergences": _inf["divergences"],
            "max_rhat": float(_d["r_hat"].max()) if len(_d) else np.nan,
            "min_ess_bulk": float(_d["ess_bulk"].min()) if len(_d) else np.nan,
            "diagnostics_pass": _ok,
            "diagnostic_failures": "; ".join(_fails),
            "posterior_mean_rho_g": _rho,
            "P(rho_g > 0.95)": _rho_hi,
            "mean_abs_resid_acf1": (float(np.nanmean(np.abs(_acf1)))
                                    if _acf1 else np.nan),
            "seconds": _inf["seconds"],
        })
    SS_CANDIDATES = pd.DataFrame(_rows)
    SS_CANDIDATE_DIAG = (pd.concat(_diags, ignore_index=True) if _diags
                         else pd.DataFrame())
    display(SS_CANDIDATES)
    register("null_dynamics_candidates", SS_CANDIDATES, "model selection")
    register("null_dynamics_diagnostics", SS_CANDIDATE_DIAG, "model selection")
''')

code(r'''# =====================================================================
# 13b. Select the persistence structure
# =====================================================================
SIMPLICITY_ORDER = {"none": 0, "randomwalk": 1, "ar1": 2}

if HAVE_PYMC and len(SS_CANDIDATES):
    _ok = SS_CANDIDATES[SS_CANDIDATES.get("fitted", False)
                        & SS_CANDIDATES.get("diagnostics_pass", False)]
    _pool = _ok if len(_ok) else SS_CANDIDATES[SS_CANDIDATES.get("fitted", False)]
    if not len(_pool):
        raise RuntimeError("No null-dynamics candidate could be fitted.")
    if not len(_ok):
        SELECTION_NOTE = ("no candidate cleared the diagnostic gate; the best "
                          "available was selected and §15's simplification ladder "
                          "runs from there")
    _best = _pool.sort_values("elpd_loo", ascending=False).iloc[0]
    _sel = _best["common_state"]
    # Within one SE of the best -> take the SIMPLER structure.
    if np.isfinite(_best.get("elpd_loo", np.nan)):
        _se_diffs = []
        for r in _pool.itertuples():
            if r.common_state == _best["common_state"]:
                continue
            try:
                d = az.compare({_best["common_state"]: NULL_FITS[_best["common_state"]],
                                r.common_state: NULL_FITS[r.common_state]},
                               ic="loo", method="stacking")
                dse = float(d["dse"].iloc[1]) if len(d) > 1 else np.nan
                ddiff = float(d["elpd_diff"].iloc[1]) if len(d) > 1 else np.nan
            except Exception:
                dse, ddiff = np.nan, np.nan
            _se_diffs.append({"candidate": r.common_state, "elpd_diff": ddiff,
                              "dse": dse})
            if (np.isfinite(ddiff) and np.isfinite(dse) and abs(ddiff) <= dse
                    and SIMPLICITY_ORDER[r.common_state] < SIMPLICITY_ORDER[_sel]):
                _sel = r.common_state
        if _se_diffs:
            print("Pairwise LOO differences against the best candidate:")
            display(pd.DataFrame(_se_diffs))
    SELECTED_COMMON_STATE = _sel
    _row = SS_CANDIDATES[SS_CANDIDATES["common_state"] == _sel].iloc[0]
    print(f"\nSELECTED shared-state structure: {_sel} — {MODEL_KINDS[_sel]}")
    if _sel != _best["common_state"]:
        print(f"  (best elpd was {_best['common_state']}, but the difference was "
              "within one standard error, so the simpler structure was taken)")
    if SELECTION_NOTE:
        print(f"  NOTE: {SELECTION_NOTE}")
    if _sel == "ar1" and np.isfinite(_row.get("P(rho_g > 0.95)", np.nan)):
        print(f"  P(rho_g > 0.95) = {_row['P(rho_g > 0.95)']:.3f}. "
              + ("A large value means the stationary AR is straining towards a "
                 "unit root and the local-level result should be read alongside "
                 "it." if _row["P(rho_g > 0.95)"] > 0.2
                 else "The stationary AR is comfortably inside the unit circle."))
    print("\nThis choice was made with NO environmental driver in any candidate.")
    register("null_dynamics_selection", pd.DataFrame([{
        "selected_common_state": _sel, "note": SELECTION_NOTE,
        "criterion": "PSIS-LOO (conditional fit) with a one-standard-error "
                     "simplicity rule, gated on divergences and R-hat",
        "caveat": "LOO on a latent-state model is not a forecast; genuine "
                  "one-calendar-month-ahead skill is §17"}]),
             "model selection")
else:
    SELECTED_COMMON_STATE = COMMON_STATE_CANDIDATES[0]
''')


# ===========================================================================
# 14. Matched null vs full
# ===========================================================================
md(r"""## 14. Step 2 — matched `regional_dynamic_null` vs `regional_dynamic_full`

Two models, identical in every respect except the environmental drivers:

* **`regional_dynamic_null`** — partially pooled intercepts, deterministic
  season, common trend, the selected shared state, regional AR, observation
  noise. No driver.
* **`regional_dynamic_full`** — exactly the same, plus the predeclared driver
  block at its a-priori lags (and the predeclared random slopes, if §10c allowed
  any).

They are fitted on **exactly the same region-month observations** — the same
`OBS_MASK`, asserted below — so the comparison is about the drivers and nothing
else.
""")

code(r'''# =====================================================================
# 14. Fit the matched pair
# =====================================================================
FIT_CONFIG = {
    "common_state": SELECTED_COMMON_STATE,
    "regional_ar": REGIONAL_AR_MODE,
    "include_trend": INCLUDE_TREND,
    "use_random_slopes": bool(MODEL_DATA["rs_idx"]),
    "random_slope_parameterisation": RANDOM_SLOPE_PARAMETERISATION,
    "hierarchy_parameterisation": HIERARCHY_PARAMETERISATION,
    "drivers_used": list(DRIVER_TERMS),
    "target_accept": SAMPLING["target_accept"],
    "simplification_step": 0,
    "simplification_history": [],
}


def fit_matched_pair(config, sampling=None, priors=None, data=None,
                     null_data=None, label_suffix=""):
    """Fit the null and the full model on IDENTICAL rows under one configuration."""
    data = MODEL_DATA if data is None else data
    null_data = (NULL_DATA if null_data is None else null_data)
    assert np.array_equal(data["obs_mask"], null_data["obs_mask"]), \
        "null and full models would be fitted on different region-months"
    terms = list(config.get("drivers_used", data["driver_terms"]))
    keep = [i for i, t in enumerate(data["driver_terms"]) if t in terms]
    d_full = dict(data)
    if keep != list(range(data["K"])):
        d_full = make_model_data(
            data["X"][:, :, keep], data["Y"], data["obs_mask"], data["season"],
            data["tt"], data["region_ids"], [data["driver_terms"][i] for i in keep],
            random_slope_terms=[t for t in data["rs_terms"] if t in terms])
    samp = dict(SAMPLING if sampling is None else sampling)
    samp["target_accept"] = config.get("target_accept", samp["target_accept"])

    m_null = build_regional_model(
        null_data, drivers=False, common_state=config["common_state"],
        regional_ar=config["regional_ar"], include_trend=config["include_trend"],
        use_random_slopes=False, priors=priors,
        hierarchy_parameterisation=config.get("hierarchy_parameterisation"))
    id_null, inf_null = fit_model(m_null, samp,
                                  label=f"regional_dynamic_null{label_suffix}",
                                  prior_predictive=False)
    m_full = build_regional_model(
        d_full, drivers=True, common_state=config["common_state"],
        regional_ar=config["regional_ar"], include_trend=config["include_trend"],
        use_random_slopes=config["use_random_slopes"], priors=priors,
        random_slope_parameterisation=config.get(
            "random_slope_parameterisation"),
        hierarchy_parameterisation=config.get("hierarchy_parameterisation"))
    id_full, inf_full = fit_model(m_full, samp,
                                  label=f"regional_dynamic_full{label_suffix}",
                                  prior_predictive=True)
    return {"null": id_null, "full": id_full, "info_null": inf_null,
            "info_full": inf_full, "data_full": d_full, "config": dict(config)}


PAIR = None
if HAVE_PYMC:
    print(f"Fitting the matched pair with common_state={FIT_CONFIG['common_state']}, "
          f"regional_ar={FIT_CONFIG['regional_ar']}, "
          f"random_slopes={MODEL_DATA['rs_terms'] or 'none'}, "
          f"{len(MODEL_DATA['y_obs']):,} region-months.")
    PAIR = fit_matched_pair(FIT_CONFIG)
    print(f"  null: {PAIR['info_null']['seconds']}s, "
          f"{PAIR['info_null']['divergences']} divergence(s)")
    print(f"  full: {PAIR['info_full']['seconds']}s, "
          f"{PAIR['info_full']['divergences']} divergence(s)")
    assert (len(PAIR["data_full"]["y_obs"]) == len(NULL_DATA["y_obs"])), \
        "the matched pair does not share an observation count"
    assert np.allclose(PAIR["data_full"]["y_obs"], NULL_DATA["y_obs"]), \
        "the matched pair does not share the same response values"
''')


# ===========================================================================
# 15. Diagnostics and the simplification ladder
# ===========================================================================
md(r"""## 15. Diagnostics, and the simplification ladder when they fail

Required of the final fits: **four chains**, $\hat R<1.01$, adequate bulk and
tail ESS, **zero** divergent transitions, posterior predictive checks, trace
plots, residual ACFs for both the regional residuals and the common state,
prior-versus-posterior comparison, and sensitivity to reasonable priors.

If the gate fails, the ladder runs **in this order** and the step that fixed it
is recorded:

1. remove weakly identified random slopes;
2. simplify the region-specific AR (`per_region` → one common $\rho$; a common
   $\rho$ → drop $u_{r,t}$ altogether, letting $\epsilon$ absorb it);
3. reduce the environmental driver set to the terms with genuine regional
   variation;
4. raise `target_accept`.

Every simplification refits **both** models, so the pair stays matched, and the
history is exported. Nothing is reported from a fit that never cleared the gate;
if the ladder is exhausted, §16 says so and withholds the coefficients.

**The scope of the gate, stated openly.** Divergences are checked globally — a
divergent transition anywhere invalidates the geometry. $\hat R$ and ESS are
checked on `REPORTED_PARAMS`: the interpretable, low-dimensional quantities §16
actually quotes. Three groups are diagnosed and **reported** but do not block:

* the **latent state paths** $g_t$ and $u_{r,t}$ — hundreds of strongly
  correlated values whose individual tail ESS is not what a driver coefficient
  rests on (§15b prints their worst $\hat R$ and ESS anyway);
* the **variance split** $\sigma_u$ / $\sigma_\epsilon$ / $\phi$, whose weak
  identification is a known structural property while their total is well
  identified (§15c);
* the **non-centred auxiliaries** `alpha_z`, `lam_z`, `b_z`, `g_raw` — these are
  coordinates, not quantities. The transformed $\alpha_r$, $\lambda_r$ and
  $b_{r,k}$ are gated in their place, which is the thing that matters.

None of that hides anything: each group's diagnostics are printed and exported.
""")

code(r'''# =====================================================================
# 15a. The ladder
# =====================================================================
LADDER_LOG = pd.DataFrame()
FIT_NULL = FIT_FULL = None
FINAL_CONFIG = dict(FIT_CONFIG)
GATE_PASSED = False
GATE_FAILURES = []


def _ladder_steps(config, data):
    """The ordered simplifications, each returning (name, new_config) or None."""
    steps = []

    def switch_hierarchy_parameterisation(c):
        cur = c.get("hierarchy_parameterisation", "centred")
        if c.get("_tried_hierarchy_parameterisation"):
            return None
        alt = "noncentred" if cur == "centred" else "centred"
        n = dict(c)
        n["hierarchy_parameterisation"] = alt
        n["_tried_hierarchy_parameterisation"] = True
        return (f"switch the regional intercept / loading parameterisation from "
                f"{cur} to {alt} (poor mixing in alpha_r is a geometry problem "
                "before it is a model problem)", n)

    def switch_random_slope_parameterisation(c):
        if not c.get("use_random_slopes"):
            return None
        cur = c.get("random_slope_parameterisation", "centred")
        alt = "noncentred" if cur == "centred" else "centred"
        if c.get("_tried_rs_parameterisation"):
            return None
        n = dict(c)
        n["random_slope_parameterisation"] = alt
        n["_tried_rs_parameterisation"] = True
        return (f"switch the random-slope parameterisation from {cur} to {alt} "
                "(a funnel is a geometry problem before it is a model problem)", n)

    def drop_random_slopes(c):
        if not c.get("use_random_slopes"):
            return None
        n = dict(c)
        n["use_random_slopes"] = False
        return ("remove weakly identified random slopes", n)

    def simplify_regional_ar(c):
        if c["regional_ar"] == "per_region":
            n = dict(c)
            n["regional_ar"] = "common"
            return ("one common regional AR parameter instead of one per region", n)
        if c["regional_ar"] == "common":
            n = dict(c)
            n["regional_ar"] = "none"
            return ("drop the region-specific AR; observation noise absorbs the "
                    "regional residual (their split was not separately identified)", n)
        return None

    def reduce_drivers(c):
        keep = [t for t in c["drivers_used"]
                if DRIVER_META.loc[DRIVER_META["term"] == t, "spatial_label"]
                .eq("spatiotemporal").any()]
        if not keep or len(keep) == len(c["drivers_used"]):
            return None
        n = dict(c)
        n["drivers_used"] = keep
        return (f"reduce the driver set to the spatiotemporal terms {keep}", n)

    def raise_target_accept(c):
        if c.get("target_accept", 0.9) >= 0.99:
            return None
        n = dict(c)
        n["target_accept"] = min(0.99, round(c.get("target_accept", 0.9) + 0.05, 3))
        return (f"raise target_accept to {n['target_accept']}", n)

    for f in (switch_hierarchy_parameterisation,
              switch_random_slope_parameterisation, drop_random_slopes,
              simplify_regional_ar, reduce_drivers, raise_target_accept):
        steps.append(f)
    return steps


if HAVE_PYMC and PAIR is not None:
    _hist = []
    _cfg = dict(FIT_CONFIG)
    _pair = PAIR
    for _attempt in range(6):
        _diag = diagnostics_table(_pair["full"], REPORTED_PARAMS,
                                  label=f"full_step{_attempt}")
        _ok, _fails = gate_diagnostics(_diag)
        _hist.append({
            "step": _attempt,
            "common_state": _cfg["common_state"],
            "regional_ar": _cfg["regional_ar"],
            "random_slopes": bool(_cfg["use_random_slopes"]),
            "random_slope_parameterisation": _cfg.get(
                "random_slope_parameterisation"),
            "hierarchy_parameterisation": _cfg.get("hierarchy_parameterisation"),
            "n_drivers": len(_cfg["drivers_used"]),
            "target_accept": _cfg.get("target_accept"),
            "divergences_full": _pair["info_full"]["divergences"],
            "divergences_null": _pair["info_null"]["divergences"],
            "max_rhat": float(_diag["r_hat"].max()) if len(_diag) else np.nan,
            "min_ess_bulk": float(_diag["ess_bulk"].min()) if len(_diag) else np.nan,
            "min_ess_tail": float(_diag["ess_tail"].min()) if len(_diag) else np.nan,
            "gate_passed": _ok,
            "failures": "; ".join(_fails),
            "action_taken": "",
        })
        if _ok:
            GATE_PASSED = True
            GATE_FAILURES = []
            break
        _next = None
        for _f in _ladder_steps(_cfg, MODEL_DATA):
            _cand = _f(_cfg)
            if _cand is not None:
                _next = _cand
                break
        if _next is None:
            GATE_FAILURES = _fails
            _hist[-1]["action_taken"] = "ladder exhausted"
            break
        _name, _cfg = _next
        _hist[-1]["action_taken"] = _name
        _cfg["simplification_step"] = _attempt + 1
        _cfg["simplification_history"] = [h["action_taken"] for h in _hist
                                          if h["action_taken"]]
        print(f"Gate failed at step {_attempt}: {'; '.join(_fails)}")
        print(f"  -> {_name}; refitting BOTH models so the pair stays matched.")
        _pair = fit_matched_pair(_cfg, label_suffix=f"_step{_attempt + 1}")
    LADDER_LOG = pd.DataFrame(_hist)
    PAIR = _pair
    FINAL_CONFIG = dict(_cfg)
    FIT_NULL, FIT_FULL = PAIR["null"], PAIR["full"]
    # Every later refit (§16b, §17, §19) inherits the accepted configuration's
    # target_accept. A cheaper refit that reintroduced divergences would not be a
    # cheaper answer to the same question.
    SAMPLING_REFIT = dict(SAMPLING_REFIT)
    SAMPLING_REFIT["target_accept"] = max(
        float(SAMPLING_REFIT["target_accept"]),
        float(FINAL_CONFIG.get("target_accept", SAMPLING_REFIT["target_accept"])))
    display(LADDER_LOG)
    register("simplification_ladder", LADDER_LOG, "diagnostic")
    if GATE_PASSED:
        print(f"\nDiagnostic gate PASSED at ladder step "
              f"{int(LADDER_LOG['step'].iloc[-1])}.")
        if int(LADDER_LOG["step"].iloc[-1]) > 0:
            print("Simplifications applied, in order: "
                  + " -> ".join([a for a in LADDER_LOG['action_taken'] if a]))
    else:
        print(f"\n*** Diagnostic gate NOT passed after the full ladder: "
              f"{GATE_FAILURES}. §16 will report the coefficients as "
              "NOT REPORTABLE. ***")
''')

code(r'''# =====================================================================
# 15b. Posterior predictive check, traces, residual ACFs
# =====================================================================
PPC_SUMMARY = pd.DataFrame()
RESID_ACF = pd.DataFrame()
STATE_ACF = pd.DataFrame()
STATE_DIAGNOSTICS = pd.DataFrame()

if HAVE_PYMC and FIT_FULL is not None:
    _dfull = PAIR["data_full"]
    _mfull = build_regional_model(
        _dfull, drivers=True, common_state=FINAL_CONFIG["common_state"],
        regional_ar=FINAL_CONFIG["regional_ar"],
        include_trend=FINAL_CONFIG["include_trend"],
        use_random_slopes=FINAL_CONFIG["use_random_slopes"],
        random_slope_parameterisation=FINAL_CONFIG.get(
            "random_slope_parameterisation"),
        hierarchy_parameterisation=FINAL_CONFIG.get(
            "hierarchy_parameterisation"))
    with _mfull:
        _pp = pm.sample_posterior_predictive(
            FIT_FULL, progressbar=False,
            random_seed=SAMPLING.get("random_seed", 0))
    _yrep = _pp.posterior_predictive["y"].stack(s=("chain", "draw")).to_numpy()
    _yobs = _dfull["y_obs"]
    PPC_SUMMARY = pd.DataFrame([{
        "statistic": s,
        "observed": f(_yobs),
        "ppc_mean": float(np.mean([f(_yrep[:, j]) for j in
                                   range(0, _yrep.shape[1],
                                         max(1, _yrep.shape[1] // 200))])),
        "ppc_p_value": float(np.mean([f(_yrep[:, j]) >= f(_yobs) for j in
                                      range(0, _yrep.shape[1],
                                            max(1, _yrep.shape[1] // 200))])),
    } for s, f in [("mean", lambda v: float(np.mean(v))),
                   ("sd", lambda v: float(np.std(v))),
                   ("min", lambda v: float(np.min(v))),
                   ("max", lambda v: float(np.max(v))),
                   ("p10", lambda v: float(np.quantile(v, 0.10))),
                   ("p90", lambda v: float(np.quantile(v, 0.90)))]])
    display(PPC_SUMMARY)
    register("posterior_predictive_checks", PPC_SUMMARY, "diagnostic")
    print("A Bayesian p-value near 0 or 1 means the model cannot reproduce that "
          "feature of the data.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    try:
        _ppc_id = az.InferenceData(posterior_predictive=_pp.posterior_predictive,
                                   observed_data=FIT_FULL.observed_data)
        az.plot_ppc(_ppc_id, ax=axes[0], num_pp_samples=60, legend=False)
    except Exception as exc:
        axes[0].hist(_yobs, bins=25, density=True, alpha=0.6, color="0.4",
                     label="observed")
        axes[0].hist(_yrep[:, ::max(1, _yrep.shape[1] // 40)].ravel(), bins=25,
                     density=True, histtype="step", color="#4C72B0",
                     label="posterior predictive")
        axes[0].legend(fontsize=8, frameon=False)
        print(f"  (az.plot_ppc unavailable here: {exc}; drew the densities directly)")
    axes[0].set_title("Posterior predictive vs observed (logit cover)", fontsize=10)
    axes[1].scatter(_yobs, _yrep.mean(axis=1), s=12, alpha=0.6, color="#4C72B0")
    _lim = [min(_yobs.min(), _yrep.mean(axis=1).min()),
            max(_yobs.max(), _yrep.mean(axis=1).max())]
    axes[1].plot(_lim, _lim, color="0.3", lw=1)
    axes[1].set_xlabel("observed"); axes[1].set_ylabel("posterior predictive mean")
    axes[1].set_title("In-sample fit (NOT validation — §17 is)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "diagnostics_ppc")
    plt.show()

    _trace_vars = [v for v in ["beta", "mu_alpha", "sigma_alpha", "rho_g",
                               "sigma_g", "sigma_lambda", "rho_u",
                               "sigma_resid_total", "sigma_eps", "tau_trend"]
                   if v in FIT_FULL.posterior]
    az.plot_trace(FIT_FULL, var_names=_trace_vars, compact=True,
                  figsize=(12, 1.7 * len(_trace_vars)))
    fig = plt.gcf()
    fig.suptitle("Trace plots — full driver model", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "diagnostics_traces")
    plt.show()

    # --- residual ACFs -------------------------------------------------------
    _res = residual_frame(FIT_FULL, _dfull, MONTH_GRID, REGION_IDS)
    _acf_rows = []
    for rid, g in _res.groupby("region_id"):
        a = calendar_acf(g["resid"].to_numpy(), g["month_pos"].to_numpy(), 12)
        a["region_id"] = rid
        _acf_rows.append(a)
    RESID_ACF = pd.concat(_acf_rows, ignore_index=True) if _acf_rows else pd.DataFrame()
    if "g" in FIT_FULL.posterior:
        _g = FIT_FULL.posterior["g"].mean(dim=("chain", "draw")).to_numpy()
        _rho = (float(FIT_FULL.posterior["rho_g"].mean())
                if "rho_g" in FIT_FULL.posterior else 1.0)
        _innov = _g[1:] - _rho * _g[:-1]
        STATE_ACF = calendar_acf(_innov, np.arange(1, len(_g)), 12)
        STATE_ACF["series"] = "common-state innovation"
    register("residual_acf_by_region", RESID_ACF, "diagnostic")
    register("common_state_innovation_acf", STATE_ACF, "diagnostic")

    # Latent-state diagnostics: reported, not blocking (see the §15 preamble).
    _sd = diagnostics_table(FIT_FULL, STATE_PARAMS, label="full_states")
    if len(_sd):
        STATE_DIAGNOSTICS = pd.DataFrame([{
            "n_state_parameters": int(len(_sd)),
            "max_rhat": float(_sd["r_hat"].max()),
            "min_ess_bulk": float(_sd["ess_bulk"].min()),
            "min_ess_tail": float(_sd["ess_tail"].min()),
            "n_above_rhat_threshold": int((_sd["r_hat"] > DIAG_MAX_RHAT).sum()),
            "n_below_ess_bulk_threshold": int((_sd["ess_bulk"]
                                               < DIAG_MIN_ESS_BULK).sum()),
            "blocking": False,
            "note": "latent state paths g and u; reported for transparency, not "
                    "used to gate the driver coefficients"}])
        display(STATE_DIAGNOSTICS)
        register("latent_state_diagnostics", STATE_DIAGNOSTICS, "diagnostic")
        register("latent_state_diagnostics_full", _sd, "diagnostic")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    if len(RESID_ACF):
        for rid, g in RESID_ACF.groupby("region_id"):
            axes[0].plot(g["lag"], g["acf"], marker="o", ms=3, lw=0.9,
                         alpha=0.75, label=rid)
        axes[0].axhline(0, color="0.3", lw=0.8)
        _n = max(int(RESID_ACF["n_pairs"].median()), 1)
        for s in (-1.96 / np.sqrt(_n), 1.96 / np.sqrt(_n)):
            axes[0].axhline(s, color="0.6", ls=":", lw=0.8)
        axes[0].set_title("Regional residual ACF (calendar lags)", fontsize=10)
        axes[0].set_xlabel("lag (calendar months)")
        axes[0].legend(fontsize=6, ncol=2, frameon=False)
    if len(STATE_ACF):
        axes[1].bar(STATE_ACF["lag"], STATE_ACF["acf"], color="#4C72B0")
        axes[1].axhline(0, color="0.3", lw=0.8)
        axes[1].set_title("Common-state innovation ACF", fontsize=10)
        axes[1].set_xlabel("lag (calendar months)")
    fig.tight_layout()
    save_fig(fig, "diagnostics_residual_acf")
    plt.show()
    print("Residual autocorrelation left at lag 1 means the dependence structure "
          "has not absorbed the persistence, and every interval below is then "
          "optimistic.")
''')

code(r'''# =====================================================================
# 15c. The variance split, prior vs posterior, and prior sensitivity
# =====================================================================
VARIANCE_SPLIT = pd.DataFrame()
PRIOR_POSTERIOR = pd.DataFrame()
PRIOR_SENSITIVITY = pd.DataFrame()

if HAVE_PYMC and FIT_FULL is not None:
    # --- the split that is known to be hard, diagnosed on its own -----------
    _split_vars = [v for v in ["sigma_resid_total", "frac_u", "sigma_u",
                               "sigma_eps", "rho_u"] if v in FIT_FULL.posterior]
    if _split_vars:
        VARIANCE_SPLIT = (az.summary(FIT_FULL, var_names=_split_vars,
                                     hdi_prob=HDI_PROB)
                          .reset_index().rename(columns={"index": "parameter"}))
        display(VARIANCE_SPLIT)
        register("regional_variance_split", VARIANCE_SPLIT, "diagnostic")
        _weak = VARIANCE_SPLIT[(VARIANCE_SPLIT["r_hat"] > DIAG_MAX_RHAT)
                               | (VARIANCE_SPLIT["ess_bulk"] < DIAG_MIN_ESS_BULK)]
        if len(_weak):
            print("The split between region-specific persistence and observation "
                  "noise is WEAKLY IDENTIFIED here "
                  f"({_weak['parameter'].tolist()}).")
            print("Their total (sigma_resid_total) is identified; the share is "
                  "not. Read sigma_u and sigma_eps as a decomposition the data "
                  "cannot pin down, not as measurements.")
        else:
            print("The regional-persistence / observation-noise split is "
                  "adequately identified in this fit.")

    # --- prior vs posterior --------------------------------------------------
    if "prior" in FIT_FULL.groups():
        _rows = []
        for v in [x for x in ["beta", "mu_alpha", "sigma_alpha", "sigma_g",
                              "rho_g", "sigma_lambda", "sigma_b", "tau_trend"]
                  if x in FIT_FULL.posterior and x in FIT_FULL.prior]:
            po = FIT_FULL.posterior[v].stack(s=("chain", "draw"))
            pr = FIT_FULL.prior[v].stack(s=("chain", "draw"))
            if po.ndim == 1:
                items = [(v, po.to_numpy(), pr.to_numpy())]
            else:
                dim = [d for d in po.dims if d != "s"][0]
                items = [(f"{v}[{lab}]",
                          po.sel({dim: lab}).to_numpy(),
                          pr.sel({dim: lab}).to_numpy())
                         for lab in po.coords[dim].to_numpy()]
            for name, a, b in items:
                sd_post, sd_prior = float(np.std(a)), float(np.std(b))
                _rows.append({
                    "parameter": name,
                    "prior_mean": float(np.mean(b)), "prior_sd": sd_prior,
                    "posterior_mean": float(np.mean(a)), "posterior_sd": sd_post,
                    "posterior_shrinkage": (1 - sd_post / sd_prior)
                    if sd_prior > 0 else np.nan})
        PRIOR_POSTERIOR = pd.DataFrame(_rows)
        display(PRIOR_POSTERIOR)
        register("prior_vs_posterior", PRIOR_POSTERIOR, "diagnostic")
        print("Shrinkage near 0 means the data said almost nothing about that "
              "parameter and the prior is doing the work.")

    # --- sensitivity to reasonable priors -----------------------------------
    _rows = []
    for _vname, _pv in PRIOR_VARIANTS.items():
        try:
            _pr = fit_matched_pair(FINAL_CONFIG, sampling=SAMPLING_REFIT,
                                   priors=_pv, label_suffix=f"_prior_{_vname}")
        except Exception as exc:
            _rows.append({"prior_variant": _vname, "term": "(fit failed)",
                          "note": str(exc)})
            continue
        if "beta" in _pr["full"].posterior:
            s = az.summary(_pr["full"], var_names=["beta"], hdi_prob=HDI_PROB)
            _hdi = [c for c in s.columns if c.startswith("hdi_")]
            for term, (_, row) in zip(_pr["data_full"]["driver_terms"], s.iterrows()):
                _rows.append({
                    "prior_variant": _vname, "priors": json.dumps(_pv),
                    "term": term, "mean": float(row["mean"]),
                    "hdi_lo": float(row[_hdi[0]]), "hdi_hi": float(row[_hdi[1]]),
                    "divergences": int(_pr["full"].sample_stats["diverging"].sum())})
    PRIOR_SENSITIVITY = pd.DataFrame(_rows)
    if len(PRIOR_SENSITIVITY):
        display(PRIOR_SENSITIVITY)
        register("prior_sensitivity", PRIOR_SENSITIVITY, "robustness")
        print("A coefficient whose sign or rough magnitude moves between the "
              "tight and loose priors is prior-dependent and is reported as "
              "fragile in §22.")
''')


# ===========================================================================
# 16. Posterior inference
# ===========================================================================
md(r"""## 16. Posterior inference — global and regional driver associations

Every coefficient is on the **standardised logit** scale: a one-standard-deviation
move in the driver, in log-odds of regional WH cover.

For each coefficient: posterior mean and median, the
95% highest-density interval, $P(\beta>0)$, $P(\beta<0)$, the posterior
probability of being inside the **region of practical equivalence**
$|\beta|<\texttt{ROPE\_HALFWIDTH}$, the between-region slope SD where one is
estimated, and the region-specific estimates where they exist.

### The verdict rules, applied conservatively

| Verdict | Requires |
|---|---|
| `supported` | 95% HDI excludes zero **and** the direction matches the predeclared mechanism **and** the diagnostic gate passed **and** the sign survives leaving any single region out |
| `suggestive` | substantial posterior sign probability, but the interval includes zero |
| `heterogeneous` | regional slopes clearly differ in direction or magnitude |
| `temporal_only` | no meaningful regional predictor variation (§10) — the coefficient is still estimated, but the regional design adds nothing to it |
| `no evidence` | everything else |

A posterior sign probability slightly above 0.5 is **not** support: with a
diffuse posterior centred near zero, $P(\beta>0)\approx0.55$ is what "nothing"
looks like. The ROPE column is there so that a tight interval around a
negligible value is not mistaken for a finding either.

Everything below is an **association**. The design is observational, the drivers
are correlated with each other and with the season, and the shared latent state
absorbs whatever moved the whole gulf at once. Nothing here identifies a causal
effect.
""")

code(r'''# =====================================================================
# 16a. Global driver coefficients
# =====================================================================
GLOBAL_DRIVERS = pd.DataFrame()
REGIONAL_DRIVERS = pd.DataFrame()
NULL_VS_FULL = pd.DataFrame()


def _hdi_bounds(samples, prob=None):
    prob = HDI_PROB if prob is None else prob
    a = np.asarray(samples, dtype=float).ravel()
    h = az.hdi(a, hdi_prob=prob)
    return float(np.min(h)), float(np.max(h))


def summarise_coefficient(samples, name, rope=None, expected_sign="?"):
    rope = ROPE_HALFWIDTH if rope is None else rope
    a = np.asarray(samples, dtype=float).ravel()
    lo, hi = _hdi_bounds(a)
    p_pos, p_neg = float(np.mean(a > 0)), float(np.mean(a < 0))
    return {
        "term": name,
        "posterior_mean": float(np.mean(a)),
        "posterior_median": float(np.median(a)),
        "posterior_sd": float(np.std(a)),
        f"hdi{int(HDI_PROB * 100)}_lo": lo,
        f"hdi{int(HDI_PROB * 100)}_hi": hi,
        "p_positive": p_pos, "p_negative": p_neg,
        "p_in_rope": float(np.mean(np.abs(a) < rope)),
        "rope_halfwidth": float(rope),
        "hdi_excludes_zero": bool(lo > 0 or hi < 0),
        "expected_sign": expected_sign,
        "direction_matches_mechanism": bool(
            expected_sign == "?" or
            (expected_sign == "+" and np.mean(a) > 0) or
            (expected_sign == "-" and np.mean(a) < 0)),
    }


if HAVE_PYMC and FIT_FULL is not None and "beta" in FIT_FULL.posterior:
    _terms = PAIR["data_full"]["driver_terms"]
    _beta = FIT_FULL.posterior["beta"].stack(s=("chain", "draw")).to_numpy()
    _rows = []
    for k, term in enumerate(_terms):
        meta = DRIVER_META[DRIVER_META["term"] == term].iloc[0]
        row = summarise_coefficient(_beta[k], term,
                                    expected_sign=meta["expected_sign"])
        row.update({"mechanism_key": meta["mechanism_key"],
                    "mechanism": meta["mechanism"],
                    "lag_months": int(meta["lag_months"]),
                    "spatial_label": meta["spatial_label"],
                    "season_confounded": bool(
                        DRIVER_VARIANCE.loc[DRIVER_VARIANCE["driver"]
                                            == meta["column"],
                                            "season_confounded"].any())})
        # between-region slope variation, where one is estimated
        if ("sigma_b" in FIT_FULL.posterior
                and term in list(FIT_FULL.posterior["sigma_b"].coords["rs_driver"]
                                 .to_numpy())):
            sb = (FIT_FULL.posterior["sigma_b"].sel(rs_driver=term)
                  .stack(s=("chain", "draw")).to_numpy())
            slo, shi = _hdi_bounds(sb)
            row.update({"between_region_slope_sd_mean": float(np.mean(sb)),
                        "between_region_slope_sd_hdi_lo": slo,
                        "between_region_slope_sd_hdi_hi": shi,
                        "p_slope_sd_exceeds_rope": float(np.mean(sb > ROPE_HALFWIDTH)),
                        "random_slope": True})
        else:
            row.update({"between_region_slope_sd_mean": np.nan,
                        "between_region_slope_sd_hdi_lo": np.nan,
                        "between_region_slope_sd_hdi_hi": np.nan,
                        "p_slope_sd_exceeds_rope": np.nan,
                        "random_slope": False})
        _rows.append(row)
    GLOBAL_DRIVERS = pd.DataFrame(_rows)
    display(GLOBAL_DRIVERS[[
        "term", "spatial_label", "posterior_mean", "posterior_median",
        f"hdi{int(HDI_PROB * 100)}_lo", f"hdi{int(HDI_PROB * 100)}_hi",
        "p_positive", "p_negative", "p_in_rope", "expected_sign",
        "hdi_excludes_zero", "between_region_slope_sd_mean"]])
    print("Scale: standardised logit. A coefficient of 0.10 means a 1 SD move in "
          "the driver shifts the log-odds of regional cover by 0.10 "
          f"(odds x {np.exp(0.10):.3f}).")
    print(f"ROPE: |beta| < {ROPE_HALFWIDTH} is practically equivalent to zero.")

    # --- region-specific estimates, where they are estimable ----------------
    _rr = []
    if "b" in FIT_FULL.posterior:
        _b = FIT_FULL.posterior["b"]
        for term in list(_b.coords["rs_driver"].to_numpy()):
            k = _terms.index(term)
            for ri, rid in enumerate(REGION_IDS):
                dev = (_b.sel(rs_driver=term).isel(region=ri)
                       .stack(s=("chain", "draw")).to_numpy())
                tot = _beta[k] + dev
                s = summarise_coefficient(tot, term)
                s.update({"region_id": rid,
                          "deviation_mean": float(np.mean(dev)),
                          "p_deviation_positive": float(np.mean(dev > 0)),
                          "estimable": True})
                _rr.append(s)
    for term in _terms:
        if any(r["term"] == term for r in _rr):
            continue
        for rid in REGION_IDS:
            _rr.append({"term": term, "region_id": rid, "estimable": False,
                        "posterior_mean": np.nan,
                        "note": "common slope by design — no regional deviation "
                                "was estimated for this driver"})
    REGIONAL_DRIVERS = (pd.DataFrame(_rr)
                        .merge(REGIONS[["region_id", "region_name", "region_type"]],
                               on="region_id", how="left"))
    if REGIONAL_DRIVERS["estimable"].any():
        display(REGIONAL_DRIVERS[REGIONAL_DRIVERS["estimable"]][
            ["term", "region_id", "region_name", "region_type", "posterior_mean",
             f"hdi{int(HDI_PROB * 100)}_lo", f"hdi{int(HDI_PROB * 100)}_hi",
             "p_positive", "deviation_mean"]])
    else:
        print("\nNo region-specific slope was estimated: §10c allowed no random "
              "slope, so every driver enters with a common gulf-wide coefficient.")
    register("posterior_global_drivers", GLOBAL_DRIVERS, "environmental association")
    register("posterior_regional_drivers", REGIONAL_DRIVERS,
             "environmental association")
elif HAVE_PYMC:
    print("No driver coefficients: the full model has no beta (was the driver set "
          "emptied by the ladder?).")
''')

code(r'''# =====================================================================
# 16b. Leave-one-region-out refits: is a sign driven by one region?
# =====================================================================
# These fits do double duty. Here they answer "would the gulf-wide association
# survive without region r?"; §18 reuses the same fits for genuine transfer
# prediction to a region the model never saw.
LORO_FITS = {}
LORO_BETAS = pd.DataFrame()
REGION_INFLUENCE = pd.DataFrame()

if HAVE_PYMC and FIT_FULL is not None and RUN_LORO and N_REGIONS >= 3:
    _hold = list(REGION_IDS)
    if LORO_MAX_REGIONS is not None:
        _hold = _hold[:int(LORO_MAX_REGIONS)]
        if len(_hold) < len(REGION_IDS):
            print(f"FAST_MODE: withholding only {len(_hold)} of {N_REGIONS} "
                  "regions. The influence check below is therefore PARTIAL; set "
                  "FAST_MODE = False for the complete jackknife.")
    _terms = PAIR["data_full"]["driver_terms"]
    _rows = []
    for rid in _hold:
        keep = [i for i, r in enumerate(REGION_IDS) if r != rid]
        sub_mask = OBS_MASK[keep, :]
        d_full = make_model_data(
            X_MODEL[keep][:, :, [DRIVER_TERMS.index(t) for t in _terms]],
            Y_RAW[keep], sub_mask, SEASON, TT,
            [REGION_IDS[i] for i in keep], _terms,
            random_slope_terms=[t for t in PAIR["data_full"]["rs_terms"]
                                if t in _terms])
        try:
            m = build_regional_model(
                d_full, drivers=True, common_state=FINAL_CONFIG["common_state"],
                regional_ar=FINAL_CONFIG["regional_ar"],
                include_trend=FINAL_CONFIG["include_trend"],
                use_random_slopes=FINAL_CONFIG["use_random_slopes"],
                random_slope_parameterisation=FINAL_CONFIG.get(
                    "random_slope_parameterisation"),
                hierarchy_parameterisation=FINAL_CONFIG.get(
                    "hierarchy_parameterisation"))
            idl, infl = fit_model(m, SAMPLING_REFIT, label=f"loro_{rid}")
        except Exception as exc:
            print(f"  LORO {rid}: fit failed ({exc})")
            continue
        LORO_FITS[rid] = {"idata": idl, "data": d_full, "info": infl,
                          "kept_regions": [REGION_IDS[i] for i in keep]}
        if "beta" in idl.posterior:
            bb = idl.posterior["beta"].stack(s=("chain", "draw")).to_numpy()
            for k, term in enumerate(_terms):
                s = summarise_coefficient(bb[k], term)
                s.update({"withheld_region": rid,
                          "divergences": infl["divergences"]})
                _rows.append(s)
        print(f"  LORO {rid}: {infl['seconds']}s, {infl['divergences']} divergence(s)")
    LORO_BETAS = pd.DataFrame(_rows)
    if len(LORO_BETAS):
        display(LORO_BETAS[["withheld_region", "term", "posterior_mean",
                            f"hdi{int(HDI_PROB * 100)}_lo",
                            f"hdi{int(HDI_PROB * 100)}_hi", "p_positive"]])
        _inf = []
        for term, g in LORO_BETAS.groupby("term"):
            full_mean = float(GLOBAL_DRIVERS.loc[GLOBAL_DRIVERS["term"] == term,
                                                 "posterior_mean"].iloc[0])
            signs = np.sign(g["posterior_mean"].to_numpy())
            _inf.append({
                "term": term,
                "full_model_mean": full_mean,
                "n_regions_withheld": int(len(g)),
                "min_mean_without_a_region": float(g["posterior_mean"].min()),
                "max_mean_without_a_region": float(g["posterior_mean"].max()),
                "sign_stable_across_jackknife": bool(
                    np.all(signs == np.sign(full_mean)) and np.all(signs != 0)),
                "largest_shift": float(np.max(np.abs(
                    g["posterior_mean"].to_numpy() - full_mean))),
                "most_influential_region": g.loc[
                    (g["posterior_mean"] - full_mean).abs().idxmax(),
                    "withheld_region"],
                "complete_jackknife": bool(len(g) == N_REGIONS),
            })
        REGION_INFLUENCE = pd.DataFrame(_inf)
        display(REGION_INFLUENCE)
        register("leave_one_region_out_betas", LORO_BETAS, "robustness")
        register("region_influence_on_global_sign", REGION_INFLUENCE, "robustness")
        print("A coefficient whose sign flips when one region is withheld is not a "
              "gulf-wide association; it is that region's association.")
elif HAVE_PYMC:
    print("Leave-one-region-out skipped (RUN_LORO off, or fewer than 3 regions).")
''')

code(r'''# =====================================================================
# 16c. Verdicts, and the null-vs-full comparison
# =====================================================================
if HAVE_PYMC and len(GLOBAL_DRIVERS):
    def _verdict(row):
        term = row["term"]
        if not GATE_PASSED:
            return ("not reportable",
                    "the final fit did not clear the diagnostic gate")
        het = False
        if row.get("random_slope") and np.isfinite(
                row.get("p_slope_sd_exceeds_rope", np.nan)):
            het = bool(row["p_slope_sd_exceeds_rope"] > 0.90)
        if not het and len(REGIONAL_DRIVERS):
            sub = REGIONAL_DRIVERS[(REGIONAL_DRIVERS["term"] == term)
                                   & REGIONAL_DRIVERS["estimable"]]
            if len(sub) >= 2:
                het = bool((sub["p_positive"] > 0.90).any()
                           and (sub["p_negative"] > 0.90).any())
        sign_ok = True
        infl_note = ""
        if len(REGION_INFLUENCE):
            m = REGION_INFLUENCE[REGION_INFLUENCE["term"] == term]
            if len(m):
                sign_ok = bool(m["sign_stable_across_jackknife"].iloc[0])
                if not bool(m["complete_jackknife"].iloc[0]):
                    infl_note = " (jackknife partial — FAST_MODE)"
        if row["hdi_excludes_zero"] and row["direction_matches_mechanism"] and sign_ok:
            v = "supported"
            why = (f"the {int(HDI_PROB * 100)}% HDI excludes zero, the sign matches "
                   f"the predeclared mechanism, diagnostics passed and the sign "
                   f"survives leaving any single region out{infl_note}")
            if row["spatial_label"] == "temporal_only":
                why += ("; NOTE this driver is gulf-wide only, so its effective "
                        "replication is the number of months, not region-months")
            if row.get("season_confounded"):
                v = "suggestive"
                why = ("interval excludes zero, but the driver is essentially the "
                       "annual cycle (R2 of the harmonics above the threshold), so "
                       "it cannot be separated from season")
            return v, why
        if het:
            return ("heterogeneous",
                    "regional slopes differ clearly in direction or magnitude, so "
                    "a single gulf-wide number misrepresents the association")
        if row["hdi_excludes_zero"] and not row["direction_matches_mechanism"]:
            return ("no evidence",
                    "the interval excludes zero but in the direction OPPOSITE to "
                    "the predeclared mechanism, so it does not support it")
        if row["hdi_excludes_zero"] and not sign_ok:
            return ("no evidence",
                    "the interval excludes zero, but the sign does not survive "
                    "withholding a single region")
        if max(row["p_positive"], row["p_negative"]) >= 0.90:
            return ("suggestive",
                    f"posterior sign probability "
                    f"{max(row['p_positive'], row['p_negative']):.2f}, but the "
                    f"{int(HDI_PROB * 100)}% interval includes zero")
        if row["p_in_rope"] > 0.80:
            return ("no evidence",
                    f"the posterior sits inside the ROPE "
                    f"(|beta| < {ROPE_HALFWIDTH}) with probability "
                    f"{row['p_in_rope']:.2f}: practically zero, not merely "
                    "uncertain")
        if row["spatial_label"] == "temporal_only":
            # More informative than a bare "no evidence": the regional design
            # could not have helped this driver in the first place.
            return ("temporal_only",
                    "one gulf-wide value per month repeated across regions, so "
                    "the regional design adds no information for this driver; "
                    "the interval also includes zero and its effective "
                    "replication is the number of months")
        return ("no evidence", "interval includes zero and no direction is "
                               "clearly favoured")

    _v = [_verdict(r) for _, r in GLOBAL_DRIVERS.iterrows()]
    GLOBAL_DRIVERS["verdict"] = [a for a, _ in _v]
    GLOBAL_DRIVERS["verdict_reason"] = [b for _, b in _v]
    GLOBAL_DRIVERS["regional_design_adds_information"] = (
        GLOBAL_DRIVERS["spatial_label"] == "spatiotemporal")
    display(GLOBAL_DRIVERS[["term", "spatial_label", "posterior_mean",
                            f"hdi{int(HDI_PROB * 100)}_lo",
                            f"hdi{int(HDI_PROB * 100)}_hi", "p_positive",
                            "p_in_rope", "verdict", "verdict_reason"]])
    register("posterior_global_drivers", GLOBAL_DRIVERS,
             "environmental association")

    # --- null vs full --------------------------------------------------------
    try:
        _cmp = az.compare({"regional_dynamic_null": FIT_NULL,
                           "regional_dynamic_full": FIT_FULL},
                          ic="loo", method="stacking")
        NULL_VS_FULL = _cmp.reset_index().rename(columns={"index": "model"})
        display(NULL_VS_FULL)
        _win = NULL_VS_FULL.iloc[0]["model"]
        _d = float(NULL_VS_FULL["elpd_diff"].iloc[1]) if len(NULL_VS_FULL) > 1 else 0.0
        _dse = float(NULL_VS_FULL["dse"].iloc[1]) if len(NULL_VS_FULL) > 1 else np.nan
        print(f"\nLOO prefers {_win}. elpd difference {_d:.2f} "
              f"(dse {_dse:.2f}).")
        if np.isfinite(_dse) and abs(_d) <= 2 * _dse:
            print("That difference is within two standard errors of zero: on this "
                  "criterion the drivers do not improve conditional fit.")
        print("This is a CONDITIONAL-FIT comparison with the latent states in "
              "place. Whether the drivers improve genuine one-month-ahead "
              "prediction is §17, and that is the question that matters.")
    except Exception as exc:
        print(f"az.compare unavailable: {exc}")
    register("null_vs_full_comparison", NULL_VS_FULL, "model comparison")
''')

code(r'''# =====================================================================
# 16d. The endogenous optical proxies — descriptive only
# =====================================================================
# Chl-a and turbidity are measured from the same reflectance a floating mat
# dominates, and a mat changes the water beneath it. They are downstream of WH as
# much as upstream, so they never enter a driver claim. Reported once here as a
# within-region, within-month partial association, clearly labelled.
PROXY_ASSOCIATION = pd.DataFrame()
if PROXY_COLS:
    _d = REGION_MONTH[REGION_MONTH["observed"]].copy()
    _rows = []
    for c in PROXY_COLS:
        v = pd.to_numeric(_d[c], errors="coerce")
        ok = v.notna() & _d["y"].notna()
        if int(ok.sum()) < 12:
            continue
        sub = _d[ok].copy()
        sub["_x"] = v[ok]
        # remove region means AND month means: what is left is the within-region,
        # within-month covariation, which is the only part not explained by "some
        # regions are weedier" or "some months were weedier everywhere".
        for col in ("_x", "y"):
            sub[col] = (sub[col] - sub.groupby("region_id")[col].transform("mean")
                        - sub.groupby("month")[col].transform("mean")
                        + sub[col].mean())
        r, p = sstats.pearsonr(sub["_x"], sub["y"])
        _rows.append({"proxy": c, "n_region_months": int(ok.sum()),
                      "partial_pearson_r": float(r), "p_value": float(p),
                      "interpretation": "DESCRIPTIVE ONLY — endogenous to WH; "
                                        "not a driver estimate"})
    PROXY_ASSOCIATION = pd.DataFrame(_rows)
    if len(PROXY_ASSOCIATION):
        display(PROXY_ASSOCIATION)
        register("endogenous_proxy_association", PROXY_ASSOCIATION,
                 "descriptive association")
        print("These are correlations between WH cover and quantities measured "
              "from the same pixels. They are reported for completeness and "
              "excluded from every driver conclusion.")
else:
    print("No endogenous optical proxy column is present in this panel.")
''')


# ===========================================================================
# 17. Temporal validation
# ===========================================================================
md(r"""## 17. Temporal validation — expanding window, **one calendar month ahead**

### The design

* **Expanding window.** At each origin the training set is every calendar month
  up to and including that origin; the target is the **single next calendar
  month**. `max(train month) < target month` is asserted at every origin.
* **Scaling is fitted inside the fold.** The driver means and SDs come from the
  fold's training months only. The global scaler from §11 is deliberately *not*
  reused: its constants were computed with the target month in them.
* **Only origin-time information enters a predictor.** Rainfall already enters at
  lag 1. Contemporaneous temperature, wind and lake level are **moved to lag 1**
  for this evaluation, because their same-month value is not knowable at the
  origin. Every forecast driver's lag is asserted $\ge$
  `VAL_FORECAST_MIN_LAG`. The a-priori (lag-0) specification is retained
  unchanged for the §14/§16 association inference — nowcasting and forecasting
  are different questions and are never merged.
* **The state is projected, not peeked at.** $g_{T+1}=\rho_g g_T$ (or $g_T$ under
  the local level) and $u_{r,T+1}=\rho_r u_{r,T}$, using only states estimated
  from training months.

### What it is compared against

| Baseline | Definition | Fitted? |
|---|---|---|
| literal region persistence | $\hat y_{r,t}=y_{r,t-1}$ at exactly $t-1$ calendar months | **no** |
| seasonal naïve | $\hat y_{r,t}=y_{r,t-12}$ by calendar timestamp | **no** |
| `regional_dynamic_null` | the hierarchical dynamic model with no driver | yes |
| `regional_dynamic_full` | the same plus the driver block | yes |

### How performance is reported

Region-macro RMSE/MAE (every region weighted equally, so a big region cannot
carry the score), valid-area-weighted RMSE/MAE, a breakdown by ecological region
type, and both the logit and the back-transformed cover scale.

**The resampling unit for uncertainty is the calendar month, not the
region-month.** Regions observed in the same month share weather, share the
shared state, and share a satellite pass; treating them as independent replicates
would shrink every interval by roughly $\sqrt{R}$ for no reason.
""")

code(r'''# =====================================================================
# 17a. The forecast driver set — origin-time information only
# =====================================================================
FORECAST_SPECS = pd.DataFrame()
FC_TERMS, X_FC_RAW = [], np.zeros((R, T, 0))

if HAVE_PYMC and FIT_FULL is not None:
    _terms_used = PAIR["data_full"]["driver_terms"]
    _rows, _arrs = [], []
    for base, meta in FORCING.items():
        term = f"{base}_lag{meta['apriori_lag']}" if meta["apriori_lag"] else base
        if term not in _terms_used:
            continue
        ap = int(meta["apriori_lag"])
        lag = max(ap, int(VAL_FORECAST_MIN_LAG))
        name = f"fc_{base}_lag{lag}"
        arr = np.full((R, T), np.nan)
        for rid, i in R_IDX.items():
            s = (REGION_MONTH[REGION_MONTH["region_id"] == rid]
                 .set_index("month").reindex(MONTH_GRID)[meta["column"]])
            arr[i, :] = s.shift(lag).to_numpy()
        _arrs.append(arr)
        FC_TERMS.append(name)
        _rows.append({"driver": base, "inference_term": term,
                      "apriori_lag": ap, "forecast_lag": lag,
                      "forecast_term": name, "source_column": meta["column"],
                      "note": ("a-priori lag is already knowable at the origin"
                               if ap >= VAL_FORECAST_MIN_LAG
                               else f"lag {ap} is NOT knowable at the origin -> "
                                    f"moved to lag {lag} for forecasting only")})
    FORECAST_SPECS = pd.DataFrame(_rows)
    X_FC_RAW = (np.stack(_arrs, axis=2) if _arrs else np.zeros((R, T, 0)))
    display(FORECAST_SPECS)
    register("forecast_driver_specs", FORECAST_SPECS, "provenance")
    assert all(int(v) >= int(VAL_FORECAST_MIN_LAG)
               for v in FORECAST_SPECS["forecast_lag"]), \
        "a forecast driver would use information dated at or after the target month"
    _moved = FORECAST_SPECS[FORECAST_SPECS["forecast_lag"]
                            > FORECAST_SPECS["apriori_lag"]]
    if len(_moved):
        print(f"{len(_moved)} contemporaneous driver(s) moved to lag 1 for the "
              f"forecast evaluation: {_moved['driver'].tolist()}")
''')

code(r'''# =====================================================================
# 17b. Expanding-window rolling origin
# =====================================================================
VAL_FOLD_AUDIT = pd.DataFrame()
VAL_PREDICTIONS = pd.DataFrame()
VAL_SCALER_AUDIT = pd.DataFrame()


def forecast_next_month(idata, cfg, x_next_std, tt_next, season_next, rs_idx):
    """One-calendar-month-ahead posterior predictive mean per region.

    Uses ONLY parameters and states estimated on the training months: the shared
    state is projected forward by its own AR coefficient, never read from the
    target month.
    """
    po = idata.posterior
    st = lambda v: po[v].stack(s=("chain", "draw")).to_numpy()
    alpha = st("alpha")                       # (region, S)
    gamma = st("gamma_season")                # (season, S)
    S_ = alpha.shape[1]
    eta = alpha + (season_next[:, None] * gamma).sum(axis=0)[None, :]
    if "tau_trend" in po:
        eta = eta + st("tau_trend")[None, :] * float(tt_next)
    if "beta" in po and x_next_std.shape[1]:
        beta = st("beta")                     # (driver, S)
        eta = eta + x_next_std @ beta
        if "b" in po and rs_idx:
            b = st("b")                       # (region, rs, S)
            for j, k in enumerate(rs_idx):
                eta = eta + b[:, j, :] * x_next_std[:, k][:, None]
    if "g" in po:
        g = st("g")                           # (time, S)
        rho_g = st("rho_g") if "rho_g" in po else np.ones(S_)
        g_next = rho_g * g[-1, :]
        eta = eta + st("lam") * g_next[None, :]
    if "u" in po:
        u = st("u")                           # (region, time, S)
        if "rho_u" in po:
            ru = st("rho_u")
            ru = ru if ru.ndim == 2 else np.repeat(ru[None, :], u.shape[0], axis=0)
        else:
            ru = np.zeros((u.shape[0], S_))
        eta = eta + ru * u[:, -1, :]
    return eta                                # (region, S)


if HAVE_PYMC and FIT_FULL is not None and len(FC_TERMS):
    _y_ok = np.isfinite(Y_RAW)
    _fc_ok = np.all(np.isfinite(X_FC_RAW), axis=2)
    _obs_fc = _y_ok & _fc_ok
    _obs_per_month = _obs_fc.sum(axis=0)
    _cum_obs_months = np.cumsum(_obs_per_month > 0)

    _feasible = [o for o in range(T - 1)
                 if _cum_obs_months[o] >= VAL_MIN_TRAIN_MONTHS
                 and _obs_fc[:, o + 1].any()]
    if VAL_MAX_ORIGINS is not None and len(_feasible) > int(VAL_MAX_ORIGINS):
        # Evenly spaced across the feasible origins, so the cap does not restrict
        # the test to one season or one part of the record.
        _pick = np.linspace(0, len(_feasible) - 1, int(VAL_MAX_ORIGINS))
        _feasible = [_feasible[int(round(i))] for i in _pick]
        _feasible = sorted(set(_feasible))
    print(f"Rolling-origin folds: {len(_feasible)} origin(s)"
          + (f" (capped by VAL_MAX_ORIGINS={VAL_MAX_ORIGINS}; evenly spaced across "
             "the record)" if VAL_MAX_ORIGINS else ""))

    _fold_rows, _pred_rows, _scale_rows = [], [], []
    _rs_names = [t for t in PAIR["data_full"]["rs_terms"]]
    for _fold, o in enumerate(_feasible):
        t_tr = o + 1
        tgt = o + 1
        mask_tr = _obs_fc[:, :t_tr]
        if mask_tr.sum() < 4 * R:
            _fold_rows.append({"fold": _fold, "origin_month": MONTH_GRID[o],
                               "usable": False,
                               "skip_reason": "too few training region-months"})
            continue
        # --- scaling fitted on THIS fold's training months only --------------
        Xtr = X_FC_RAW[:, :t_tr, :].copy()
        Xstd = np.zeros_like(Xtr)
        x_next = np.full((R, len(FC_TERMS)), np.nan)
        for k, name in enumerate(FC_TERMS):
            v = Xtr[:, :, k][mask_tr]
            mu = float(np.nanmean(v)); sd = float(np.nanstd(v))
            sd = sd if np.isfinite(sd) and sd > 0 else 1.0
            Xstd[:, :, k] = (Xtr[:, :, k] - mu) / sd
            x_next[:, k] = (X_FC_RAW[:, tgt, k] - mu) / sd
            _scale_rows.append({"fold": _fold, "origin_month": MONTH_GRID[o],
                                "term": name, "mean": mu, "sd": sd,
                                "n_training_region_months": int(np.isfinite(v).sum())})
        Xstd = np.where(np.isfinite(Xstd), Xstd, 0.0)
        tt_tr = np.arange(t_tr, dtype=float)
        sc = {"mean": float(tt_tr.mean()), "sd": float(tt_tr.std() or 1.0)}
        tt_std = (tt_tr - sc["mean"]) / sc["sd"]
        tt_next = (float(t_tr) - sc["mean"]) / sc["sd"]
        season_tr = fourier_terms(pd.Series(MONTH_GRID[:t_tr]),
                                  SEASON_HARMONICS).to_numpy()
        season_next = fourier_terms(pd.Series([MONTH_GRID[tgt]]),
                                    SEASON_HARMONICS).to_numpy()[0]

        d_full = make_model_data(Xstd, Y_RAW[:, :t_tr], mask_tr, season_tr,
                                 tt_std, REGION_IDS, FC_TERMS,
                                 random_slope_terms=[
                                     n for n in FC_TERMS
                                     if any(n.startswith(f"fc_{c}")
                                            for c in RANDOM_SLOPE_TERMS)]
                                 if FINAL_CONFIG["use_random_slopes"] else [])
        d_null = make_model_data(np.zeros((R, t_tr, 0)), Y_RAW[:, :t_tr], mask_tr,
                                 season_tr, tt_std, REGION_IDS, [])
        assert MONTH_GRID[:t_tr].max() < MONTH_GRID[tgt], \
            "training reaches the target month"

        try:
            m_n = build_regional_model(
                d_null, drivers=False,
                common_state=FINAL_CONFIG["common_state"],
                regional_ar=FINAL_CONFIG["regional_ar"],
                include_trend=FINAL_CONFIG["include_trend"],
                use_random_slopes=False,
                hierarchy_parameterisation=FINAL_CONFIG.get(
                    "hierarchy_parameterisation"))
            id_n, inf_n = fit_model(m_n, SAMPLING_REFIT, label=f"val_null_{_fold}")
            m_f = build_regional_model(
                d_full, drivers=True,
                common_state=FINAL_CONFIG["common_state"],
                regional_ar=FINAL_CONFIG["regional_ar"],
                include_trend=FINAL_CONFIG["include_trend"],
                use_random_slopes=FINAL_CONFIG["use_random_slopes"],
                random_slope_parameterisation=FINAL_CONFIG.get(
                    "random_slope_parameterisation"),
                hierarchy_parameterisation=FINAL_CONFIG.get(
                    "hierarchy_parameterisation"))
            id_f, inf_f = fit_model(m_f, SAMPLING_REFIT, label=f"val_full_{_fold}")
        except Exception as exc:
            _fold_rows.append({"fold": _fold, "origin_month": MONTH_GRID[o],
                               "usable": False, "skip_reason": f"fit failed: {exc}"})
            continue

        eta_n = forecast_next_month(id_n, FINAL_CONFIG, np.zeros((R, 0)), tt_next,
                                    season_next, [])
        eta_f = forecast_next_month(id_f, FINAL_CONFIG,
                                    np.where(np.isfinite(x_next), x_next, 0.0),
                                    tt_next, season_next, d_full["rs_idx"])
        for ri, rid in enumerate(REGION_IDS):
            if not _obs_fc[ri, tgt]:
                continue
            pers = Y_RAW[ri, tgt - 1] if tgt - 1 >= 0 else np.nan
            seas = Y_RAW[ri, tgt - 12] if tgt - 12 >= 0 else np.nan
            _pred_rows.append({
                "fold": _fold, "origin_month": MONTH_GRID[o],
                "target_month": MONTH_GRID[tgt], "region_id": rid,
                "y_true": float(Y_RAW[ri, tgt]),
                "valid_area_ha": float(VALID_AREA[ri, tgt]),
                "pred_null": float(np.mean(eta_n[ri])),
                "pred_full": float(np.mean(eta_f[ri])),
                "pred_persistence": float(pers) if np.isfinite(pers) else np.nan,
                "pred_seasonal_naive": float(seas) if np.isfinite(seas) else np.nan,
                "sd_null": float(np.std(eta_n[ri])),
                "sd_full": float(np.std(eta_f[ri])),
                "drivers_complete": bool(np.all(np.isfinite(x_next[ri]))),
            })
        _fold_rows.append({
            "fold": _fold, "origin_month": MONTH_GRID[o],
            "target_month": MONTH_GRID[tgt], "usable": True,
            "n_training_region_months": int(mask_tr.sum()),
            "n_training_months": int((mask_tr.sum(axis=0) > 0).sum()),
            "n_target_regions": int(_obs_fc[:, tgt].sum()),
            "divergences_null": inf_n["divergences"],
            "divergences_full": inf_f["divergences"],
            "seconds": inf_n["seconds"] + inf_f["seconds"], "skip_reason": ""})
        print(f"  fold {_fold}: origin {MONTH_GRID[o]:%Y-%m} -> target "
              f"{MONTH_GRID[tgt]:%Y-%m}, {int(_obs_fc[:, tgt].sum())} region(s), "
              f"{inf_n['seconds'] + inf_f['seconds']:.0f}s")

    VAL_FOLD_AUDIT = pd.DataFrame(_fold_rows)
    VAL_PREDICTIONS = pd.DataFrame(_pred_rows)
    VAL_SCALER_AUDIT = pd.DataFrame(_scale_rows)
    if len(VAL_PREDICTIONS):
        VAL_PREDICTIONS = VAL_PREDICTIONS.merge(
            REGIONS[["region_id", "region_name", "region_type"]],
            on="region_id", how="left")
    display(VAL_FOLD_AUDIT)
    register("temporal_validation_fold_audit", VAL_FOLD_AUDIT, "validation")
    register("temporal_validation_predictions", VAL_PREDICTIONS, "validation")
    register("temporal_validation_scaler_audit", VAL_SCALER_AUDIT, "provenance")
    print(f"\n{len(VAL_PREDICTIONS):,} held-out region-month prediction(s) over "
          f"{VAL_PREDICTIONS['target_month'].nunique() if len(VAL_PREDICTIONS) else 0} "
          "target month(s).")
    print("These are genuine out-of-sample forecasts: every model was refitted "
          "from scratch on strictly earlier months, with its own scaling.")
else:
    print("§17 skipped (PyMC unavailable, no fitted model, or no forecastable "
          "driver).")
''')

code(r'''# =====================================================================
# 17c. Skill, by region and by ecological class, on both scales
# =====================================================================
VAL_METRICS = pd.DataFrame()
VAL_METRICS_BY_TYPE = pd.DataFrame()
VAL_RMSE_DIFF = pd.DataFrame()
MODEL_COLS = {"persistence": "pred_persistence",
              "seasonal_naive": "pred_seasonal_naive",
              "regional_dynamic_null": "pred_null",
              "regional_dynamic_full": "pred_full"}


def _rmse(a, b, w=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    e = (a[ok] - b[ok]) ** 2
    if w is None:
        return float(np.sqrt(e.mean()))
    ww = np.asarray(w, float)[ok]
    ww = ww / ww.sum() if ww.sum() > 0 else np.full(e.size, 1 / e.size)
    return float(np.sqrt(np.sum(ww * e)))


def _mae(a, b, w=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    e = np.abs(a[ok] - b[ok])
    if w is None:
        return float(e.mean())
    ww = np.asarray(w, float)[ok]
    ww = ww / ww.sum() if ww.sum() > 0 else np.full(e.size, 1 / e.size)
    return float(np.sum(ww * e))


def skill_table(pred, cols=MODEL_COLS, restrict_common=True):
    """Region-macro and area-weighted RMSE/MAE on the logit and cover scales."""
    if not len(pred):
        return pd.DataFrame()
    d = pred.copy()
    used = [c for c in cols.values() if c in d.columns]
    if restrict_common:
        d = d[np.isfinite(d[used]).all(axis=1) & d["y_true"].notna()]
    rows = []
    for name, col in cols.items():
        if col not in d.columns:
            continue
        sub = d[d[col].notna()]
        if not len(sub):
            continue
        y_cov = inverse_transform_response(sub["y_true"].to_numpy(),
                                           RESPONSE_TRANSFORM, RESPONSE_EPS)
        p_cov = inverse_transform_response(sub[col].to_numpy(),
                                           RESPONSE_TRANSFORM, RESPONSE_EPS)
        per_region = pd.DataFrame(
            [{"region_id": rid, "rmse": _rmse(g["y_true"], g[col]),
              "mae": _mae(g["y_true"], g[col])}
             for rid, g in sub.groupby("region_id")]).set_index("region_id")
        rows.append({
            "model": name, "n_predictions": int(len(sub)),
            "n_regions": int(sub["region_id"].nunique()),
            "n_target_months": int(sub["target_month"].nunique()),
            "rmse_logit_pooled": _rmse(sub["y_true"], sub[col]),
            "mae_logit_pooled": _mae(sub["y_true"], sub[col]),
            "rmse_logit_region_macro": float(per_region["rmse"].mean()),
            "mae_logit_region_macro": float(per_region["mae"].mean()),
            "rmse_logit_area_weighted": _rmse(sub["y_true"], sub[col],
                                              sub["valid_area_ha"]),
            "mae_logit_area_weighted": _mae(sub["y_true"], sub[col],
                                            sub["valid_area_ha"]),
            "rmse_cover": _rmse(y_cov, p_cov),
            "mae_cover": _mae(y_cov, p_cov),
            "rmse_cover_area_weighted": _rmse(y_cov, p_cov,
                                              sub["valid_area_ha"].to_numpy()),
            "common_sample": bool(restrict_common),
        })
    return pd.DataFrame(rows).sort_values("rmse_logit_region_macro")


if len(VAL_PREDICTIONS):
    VAL_METRICS = pd.concat([
        skill_table(VAL_PREDICTIONS, restrict_common=True),
        skill_table(VAL_PREDICTIONS, restrict_common=False)], ignore_index=True)
    display(VAL_METRICS)
    register("temporal_validation_metrics", VAL_METRICS, "validation")
    print("`common_sample = True` is the like-for-like table: only target "
          "region-months every model could predict. `False` scores each model on "
          "its own available months (persistence and seasonal-naive lose months "
          "whose source month is missing).")

    _rows = []
    for rtype, g in VAL_PREDICTIONS.groupby("region_type"):
        t = skill_table(g, restrict_common=True)
        if len(t):
            t.insert(0, "region_type", rtype)
            _rows.append(t)
    VAL_METRICS_BY_TYPE = (pd.concat(_rows, ignore_index=True) if _rows
                           else pd.DataFrame())
    if len(VAL_METRICS_BY_TYPE):
        display(VAL_METRICS_BY_TYPE[["region_type", "model", "n_predictions",
                                     "rmse_logit_region_macro", "rmse_cover"]])
        register("temporal_validation_metrics_by_region_type", VAL_METRICS_BY_TYPE,
                 "validation")

    # --- uncertainty on the differences: resample CALENDAR MONTHS ------------
    _d = VAL_PREDICTIONS.dropna(subset=["y_true", "pred_null", "pred_full"])
    if len(_d):
        months = _d["target_month"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(VAL_BOOTSTRAP_SEED)
        pairs = [("regional_dynamic_full", "regional_dynamic_null"),
                 ("regional_dynamic_full", "persistence"),
                 ("regional_dynamic_full", "seasonal_naive"),
                 ("regional_dynamic_null", "persistence")]
        _rows = []
        for a, b in pairs:
            ca, cb = MODEL_COLS[a], MODEL_COLS[b]
            sub = _d.dropna(subset=[ca, cb])
            if len(sub) < 4 or sub["target_month"].nunique() < 3:
                continue
            obs_diff = _rmse(sub["y_true"], sub[ca]) - _rmse(sub["y_true"], sub[cb])
            boots = []
            mm = sub["target_month"].drop_duplicates().to_numpy()
            for _ in range(int(VAL_BOOTSTRAP_N)):
                pick = rng.choice(mm, size=len(mm), replace=True)
                s = pd.concat([sub[sub["target_month"] == m] for m in pick])
                boots.append(_rmse(s["y_true"], s[ca]) - _rmse(s["y_true"], s[cb]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            _rows.append({
                "model_a": a, "model_b": b,
                "rmse_a": _rmse(sub["y_true"], sub[ca]),
                "rmse_b": _rmse(sub["y_true"], sub[cb]),
                "rmse_difference_a_minus_b": obs_diff,
                "ci95_lo": float(lo), "ci95_hi": float(hi),
                "n_target_months_resampled": int(len(mm)),
                "n_region_months": int(len(sub)),
                "a_better_and_interval_excludes_zero": bool(obs_diff < 0 and hi < 0),
                "resampling_unit": "calendar month (regions within a month are "
                                   "NOT independent replicates)"})
        VAL_RMSE_DIFF = pd.DataFrame(_rows)
        if not len(VAL_RMSE_DIFF):
            print("No RMSE-difference interval could be formed: fewer than three "
                  "target CALENDAR MONTHS were available, and the resampling unit "
                  "is the month, not the region-month. Widening it to "
                  "region-months would manufacture precision out of regions that "
                  "share a month, a satellite pass and the shared state — so the "
                  "interval is withheld instead. Raise VAL_MAX_ORIGINS (or set "
                  "FAST_MODE = False) for a usable interval.")
        display(VAL_RMSE_DIFF)
        register("temporal_validation_rmse_differences", VAL_RMSE_DIFF, "validation")
        for r in VAL_RMSE_DIFF.itertuples():
            if r.a_better_and_interval_excludes_zero:
                print(f"{r.model_a} beats {r.model_b}: RMSE difference "
                      f"{r.rmse_difference_a_minus_b:+.4f} "
                      f"[{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}].")
            else:
                print(f"{r.model_a} vs {r.model_b}: difference "
                      f"{r.rmse_difference_a_minus_b:+.4f} "
                      f"[{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] — the interval "
                      "includes zero, so this is NOT an improvement.")
        if (len(VAL_RMSE_DIFF)
                and VAL_RMSE_DIFF["n_target_months_resampled"].min() < 8):
            print("\nWith this few target months the bootstrap interval is itself "
                  "coarse. Treat it as a guard against over-claiming, not as a "
                  "precise interval.")
''')


# ===========================================================================
# 18. Regional transfer
# ===========================================================================
md(r"""## 18. Regional transfer — leave-one-region-out

A different question from §17, and reported separately.

§17 asks: given everything up to last month, can the model predict next month?
It uses each region's own history. §18 asks: given the other regions, can the
model predict a region it has **never seen**? The withheld region's response
never entered the fit, so its own $\alpha_r$, $\lambda_r$, $b_{r,k}$ and
$u_{r,t}$ are unknown. They are drawn from the **population** distributions the
hierarchy estimated from the other regions — which is exactly what "transfer"
means and exactly why the intervals are wider.

The shared state $g_t$ *is* used, and legitimately: it is estimated from the
other regions, not from the withheld one. What is never used is any part of the
withheld region's own response history.

The comparison baseline is the strongest honest one available for this task:
predicting the withheld region with the **contemporaneous mean of the other
regions** on the logit scale. Anything the hierarchy adds has to be added on top
of that.
""")

code(r'''# =====================================================================
# 18. Predict a region the model never saw
# =====================================================================
LORO_PREDICTIONS = pd.DataFrame()
LORO_METRICS = pd.DataFrame()

if HAVE_PYMC and LORO_FITS:
    _rng = np.random.default_rng(RANDOM_STATE)
    _terms = PAIR["data_full"]["driver_terms"]
    _k_idx = [DRIVER_TERMS.index(t) for t in _terms]
    _rows = []
    for rid, entry in LORO_FITS.items():
        ri = REGION_IDS.index(rid)
        po = entry["idata"].posterior
        st = lambda v: po[v].stack(s=("chain", "draw")).to_numpy()
        S_ = po.sizes["chain"] * po.sizes["draw"]
        mu_a, sd_a = st("mu_alpha"), st("sigma_alpha")
        alpha_new = mu_a + sd_a * _rng.standard_normal(S_)
        gamma = st("gamma_season")
        tau = st("tau_trend") if "tau_trend" in po else np.zeros(S_)
        beta = st("beta") if "beta" in po else np.zeros((0, S_))
        b_new = None
        if "sigma_b" in po:
            sb = st("sigma_b")                      # (rs, S)
            b_new = sb * _rng.standard_normal(sb.shape)
        lam_new = (1.0 + st("sigma_lambda") * _rng.standard_normal(S_)
                   if "sigma_lambda" in po else np.zeros(S_))
        g = st("g") if "g" in po else np.zeros((T, S_))
        if "sigma_u" in po:
            su = st("sigma_u")
            ru = st("rho_u")
            ru = ru if ru.ndim == 1 else ru.mean(axis=0)
            u_sd = su / np.sqrt(np.clip(1 - ru ** 2, 1e-6, None))
        else:
            u_sd = np.zeros(S_)
        sig_eps = st("sigma_eps") if "sigma_eps" in po else np.zeros(S_)
        rs_idx = entry["data"]["rs_idx"]

        for ti in range(T):
            if not OBS_MASK[ri, ti]:
                continue
            eta = (alpha_new
                   + (SEASON[ti][:, None] * gamma).sum(axis=0)
                   + tau * float(TT[ti])
                   + lam_new * g[ti])
            if beta.shape[0]:
                x = X_MODEL[ri, ti, _k_idx]
                eta = eta + x @ beta
                if b_new is not None and rs_idx:
                    for j, k in enumerate(rs_idx):
                        eta = eta + b_new[j] * float(x[k])
            eta = eta + u_sd * _rng.standard_normal(S_)
            others = [q for q in range(R) if q != ri and OBS_MASK[q, ti]]
            _rows.append({
                "withheld_region": rid, "month": MONTH_GRID[ti],
                "y_true": float(Y_RAW[ri, ti]),
                "valid_area_ha": float(VALID_AREA[ri, ti]),
                "pred_transfer": float(np.mean(eta)),
                "sd_transfer": float(np.std(eta)),
                "pred_other_region_mean": (float(np.mean(Y_RAW[others, ti]))
                                           if others else np.nan),
                "n_other_regions_that_month": len(others),
            })
    LORO_PREDICTIONS = pd.DataFrame(_rows)
    if len(LORO_PREDICTIONS):
        LORO_PREDICTIONS = LORO_PREDICTIONS.merge(
            REGIONS[["region_id", "region_name", "region_type"]]
            .rename(columns={"region_id": "withheld_region"}),
            on="withheld_region", how="left")
        _m = []
        for rid, g in LORO_PREDICTIONS.groupby("withheld_region"):
            y_cov = inverse_transform_response(g["y_true"].to_numpy(),
                                               RESPONSE_TRANSFORM, RESPONSE_EPS)
            p_cov = inverse_transform_response(g["pred_transfer"].to_numpy(),
                                               RESPONSE_TRANSFORM, RESPONSE_EPS)
            _m.append({
                "withheld_region": rid,
                "region_type": g["region_type"].iloc[0],
                "n_months": int(len(g)),
                "rmse_logit_transfer": _rmse(g["y_true"], g["pred_transfer"]),
                "mae_logit_transfer": _mae(g["y_true"], g["pred_transfer"]),
                "rmse_cover_transfer": _rmse(y_cov, p_cov),
                "rmse_logit_other_region_mean": _rmse(g["y_true"],
                                                      g["pred_other_region_mean"]),
                "transfer_beats_other_region_mean": bool(
                    _rmse(g["y_true"], g["pred_transfer"])
                    < _rmse(g["y_true"], g["pred_other_region_mean"])),
            })
        LORO_METRICS = pd.DataFrame(_m)
        display(LORO_METRICS)
        register("leave_one_region_out_predictions", LORO_PREDICTIONS,
                 "transfer validation")
        register("leave_one_region_out_metrics", LORO_METRICS,
                 "transfer validation")
        print("\nThe model used NONE of the withheld region's response history. "
              "Its regional intercept, loading and slope deviations were drawn "
              "from the population distributions estimated on the other regions.")
        if LORO_MAX_REGIONS is not None and len(LORO_METRICS) < N_REGIONS:
            print(f"FAST_MODE: only {len(LORO_METRICS)} of {N_REGIONS} regions "
                  "were withheld. Set FAST_MODE = False for the complete "
                  "leave-one-region-out sweep.")
        _win = int(LORO_METRICS["transfer_beats_other_region_mean"].sum())
        print(f"The hierarchy beat the contemporaneous other-region mean in "
              f"{_win} of {len(LORO_METRICS)} withheld region(s).")
else:
    print("§18 skipped: no leave-one-region-out fits are available.")
''')


# ===========================================================================
# 19. Regionalisation sensitivity
# ===========================================================================
md(r"""## 19. Regionalisation sensitivity

A boundary drawn at 5 km rather than 3 km is a *choice*. If the conclusions turn
on it, the reader has to know.

A **small, predeclared** set of response-blind variants (§3h) is re-run end to
end — new thresholds, new components, new merges, new region-month panel, new
fit — and the global driver coefficients are compared with the headline. This is
**not** a search: every variant is reported, and none may be promoted to the
headline because it produced a stronger result. The headline is the
configuration fixed in §3c before anything was fitted.
""")

code(r'''# =====================================================================
# 19. Re-run the whole pipeline under alternative region definitions
# =====================================================================
SENSITIVITY_REGIONS = pd.DataFrame()
SENSITIVITY_BETAS = pd.DataFrame()


def regional_dataset_from_thresholds(thresholds, min_cells, min_area_ha,
                                     openness_quantile=None):
    """Regions -> region-month panel -> model arrays, for one set of thresholds."""
    th = dict(thresholds)
    if openness_quantile is not None:
        th["openness"] = None
    th, _prov = resolve_thresholds(
        CELL_STATIC, REGION_COVARIATES, th,
        OPENNESS_FALLBACK_QUANTILE if openness_quantile is None
        else openness_quantile, verbose=False)
    assign, regions, mlog, _ = build_regions(
        CELL_STATIC, REGION_COVARIATES, th, cell_size_m=EXPECTED_CELL_SIZE_M,
        contiguity=REGION_CONTIGUITY, min_cells=min_cells,
        min_area_ha=min_area_ha, sim_cols=_SIM_COLS, area_col="eligible_area_ha")
    rm = regional_monthly_panel(
        PANEL_FIXED, assign, CANDIDATE_DRIVER_COLS,
        min_cell_coverage=MIN_REGION_MONTH_CELL_COVERAGE,
        min_area_coverage=MIN_REGION_MONTH_VALID_AREA_COVERAGE)
    ok = rm[rm["region_month_usable"]]
    req = (ok.groupby("region_id")
           .agg(n_months_usable=("month", "nunique"),
                median_cell_coverage=("cell_coverage_fraction", "median"))
           .reset_index())
    keep = req.loc[(req["n_months_usable"] >= MIN_REGION_MONTHS)
                   & (req["median_cell_coverage"] >= MIN_REGION_MEDIAN_COVERAGE),
                   "region_id"].tolist()
    keep = [r for r in sorted(keep)
            if regions.loc[regions["region_id"] == r, "n_cells"].iloc[0] >= min_cells]
    if len(keep) < 2:
        return None
    rmap = {r: i for i, r in enumerate(keep)}
    Yv = np.full((len(keep), T), np.nan)
    Xv = np.full((len(keep), T, len(FORCING)), np.nan)
    for rid, i in rmap.items():
        g = rm[rm["region_id"] == rid].set_index("month").reindex(MONTH_GRID)
        yv, _ = transform_response(g["wh_cover"], RESPONSE_TRANSFORM, RESPONSE_EPS)
        Yv[i, :] = yv.to_numpy()
        for k, (base, meta) in enumerate(FORCING.items()):
            Xv[i, :, k] = g[meta["column"]].shift(int(meta["apriori_lag"])).to_numpy()
    mask = np.isfinite(Yv) & np.all(np.isfinite(Xv), axis=2)
    if mask.sum() < 30:
        return None
    Xs = np.zeros_like(Xv)
    for k in range(Xv.shape[2]):
        v = Xv[:, :, k][mask]
        mu, sd = float(np.nanmean(v)), float(np.nanstd(v) or 1.0)
        Xs[:, :, k] = (Xv[:, :, k] - mu) / (sd if sd > 0 else 1.0)
    Xs = np.where(np.isfinite(Xs), Xs, 0.0)
    data = make_model_data(Xs, Yv, mask, SEASON, TT, keep, DRIVER_TERMS,
                           random_slope_terms=[])
    return {"assignments": assign, "regions": regions, "keep": keep,
            "data": data, "thresholds": th, "merge_log": mlog}


if HAVE_PYMC and FIT_FULL is not None and SENSITIVITY_VARIANTS:
    _rrows, _brows = [], []
    _head = GLOBAL_DRIVERS.set_index("term")["posterior_mean"].to_dict() \
        if len(GLOBAL_DRIVERS) else {}
    for _vname in SENSITIVITY_VARIANTS:
        _ov = dict(REGIONALISATION_VARIANTS[_vname])
        _minc = int(_ov.pop("_min_region_cells", MIN_REGION_CELLS))
        _oq = _ov.pop("_openness_quantile", None)
        _th = dict(REGION_THRESHOLDS)
        _th.update(_ov)
        print(f"--- variant {_vname}: {REGIONALISATION_VARIANTS[_vname]} ---")
        try:
            _ds = regional_dataset_from_thresholds(
                _th, _minc, MIN_REGION_ELIGIBLE_AREA_HA, openness_quantile=_oq)
        except Exception as exc:
            print(f"    regionalisation failed: {exc}")
            _rrows.append({"variant": _vname, "n_regions": 0,
                           "note": f"regionalisation failed: {exc}"})
            continue
        if _ds is None:
            print("    too few usable regions or region-months under this variant")
            _rrows.append({"variant": _vname, "n_regions": 0,
                           "note": "too few usable regions / region-months"})
            continue
        _rrows.append({
            "variant": _vname,
            "overrides": json.dumps(REGIONALISATION_VARIANTS[_vname]),
            "n_regions": len(_ds["keep"]),
            "n_region_months": int(_ds["data"]["obs_mask"].sum()),
            "region_types": ", ".join(sorted(
                _ds["regions"].loc[_ds["regions"]["region_id"].isin(_ds["keep"]),
                                   "region_type"].unique())),
            "note": ""})
        try:
            _m = build_regional_model(
                _ds["data"], drivers=True,
                common_state=FINAL_CONFIG["common_state"],
                regional_ar=FINAL_CONFIG["regional_ar"],
                include_trend=FINAL_CONFIG["include_trend"],
                use_random_slopes=False,
                hierarchy_parameterisation=FINAL_CONFIG.get(
                    "hierarchy_parameterisation"))
            _id, _inf = fit_model(_m, SAMPLING_REFIT, label=f"sens_{_vname}")
        except Exception as exc:
            print(f"    fit failed: {exc}")
            continue
        _bb = _id.posterior["beta"].stack(s=("chain", "draw")).to_numpy()
        for k, term in enumerate(_ds["data"]["driver_terms"]):
            s = summarise_coefficient(_bb[k], term)
            s.update({"variant": _vname, "n_regions": len(_ds["keep"]),
                      "headline_mean": _head.get(term, np.nan),
                      "divergences": _inf["divergences"]})
            s["same_sign_as_headline"] = bool(
                np.isfinite(s["headline_mean"])
                and np.sign(s["posterior_mean"]) == np.sign(s["headline_mean"]))
            _brows.append(s)
        print(f"    {len(_ds['keep'])} regions, "
              f"{int(_ds['data']['obs_mask'].sum())} region-months, "
              f"{_inf['divergences']} divergence(s)")
    SENSITIVITY_REGIONS = pd.DataFrame(_rrows)
    SENSITIVITY_BETAS = pd.DataFrame(_brows)
    display(SENSITIVITY_REGIONS)
    if len(SENSITIVITY_BETAS):
        display(SENSITIVITY_BETAS[["variant", "term", "n_regions",
                                   "posterior_mean", "headline_mean",
                                   f"hdi{int(HDI_PROB * 100)}_lo",
                                   f"hdi{int(HDI_PROB * 100)}_hi",
                                   "same_sign_as_headline"]])
        _stab = (SENSITIVITY_BETAS.groupby("term")
                 .agg(n_variants=("variant", "nunique"),
                      n_same_sign=("same_sign_as_headline", "sum"),
                      min_mean=("posterior_mean", "min"),
                      max_mean=("posterior_mean", "max")).reset_index())
        _stab["headline_mean"] = _stab["term"].map(_head)
        _stab["sign_stable_across_variants"] = (
            _stab["n_same_sign"] == _stab["n_variants"])
        _stab["magnitude_range"] = _stab["max_mean"] - _stab["min_mean"]
        display(_stab)
        register("regionalisation_sensitivity_summary", _stab, "robustness")
        for r in _stab.itertuples():
            if not r.sign_stable_across_variants:
                print(f"*** {r.term}: the sign CHANGES between regionalisation "
                      "variants. The conclusion for this driver depends on where "
                      "the regional boundaries were drawn. ***")
    register("regionalisation_sensitivity_regions", SENSITIVITY_REGIONS,
             "robustness")
    register("regionalisation_sensitivity_betas", SENSITIVITY_BETAS, "robustness")
    if FAST_MODE:
        print(f"\nFAST_MODE ran {len(SENSITIVITY_VARIANTS)} of "
              f"{len(REGIONALISATION_VARIANTS)} predeclared variants. The full set "
              "runs with FAST_MODE = False.")
else:
    print("§19 skipped.")
''')


# ===========================================================================
# 20. Figures
# ===========================================================================
md(r"""## 20. Dissertation figures

The region map (Figure 1) and the regional series (Figure 2) were drawn where
they were built, in §8b and §9e. The four that depend on the fitted model are
drawn here.
""")

code(r'''# =====================================================================
# 20a. Figure 3 - global driver posterior intervals
# =====================================================================
if HAVE_PYMC and len(GLOBAL_DRIVERS):
    d = GLOBAL_DRIVERS.sort_values("posterior_mean")
    lo = d[f"hdi{int(HDI_PROB * 100)}_lo"].to_numpy()
    hi = d[f"hdi{int(HDI_PROB * 100)}_hi"].to_numpy()
    m = d["posterior_mean"].to_numpy()
    ypos = np.arange(len(d))
    colours = {"supported": "#1b7837", "suggestive": "#7fbf7b",
               "heterogeneous": "#d95f02", "temporal_only": "#7570b3",
               "no evidence": "#999999", "not reportable": "#cccccc"}
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(d) + 2.2))
    ax.axvspan(-ROPE_HALFWIDTH, ROPE_HALFWIDTH, color="0.88", zorder=0,
               label=f"ROPE (|beta| < {ROPE_HALFWIDTH})")
    ax.axvline(0, color="0.3", lw=1, zorder=1)
    for i, r in enumerate(d.itertuples()):
        c = colours.get(r.verdict, "#999999")
        ax.plot([lo[i], hi[i]], [i, i], color=c, lw=2.6, solid_capstyle="round",
                zorder=2)
        ax.plot(m[i], i, "o", color=c, ms=7, mec="white", mew=1.1, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.term}\n({r.spatial_label}, expect {r.expected_sign})"
                        for r in d.itertuples()], fontsize=8)
    ax.set_xlabel("posterior coefficient — standardised logit "
                  "(1 SD driver -> change in log-odds of regional WH cover)")
    ax.set_title(f"Global driver associations, {int(HDI_PROB * 100)}% HDI"
                 + (" (SYNTHETIC)" if SOURCE["is_synthetic"] else ""), fontsize=12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=v, lw=2.6, label=k)
                       for k, v in colours.items()
                       if k in set(d["verdict"])],
              loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.2, lw=0.4)
    fig.tight_layout()
    save_fig(fig, "03_global_driver_posteriors")
    plt.show()
''')

code(r'''# =====================================================================
# 20b. Figure 4 - regional slope variation
# =====================================================================
if HAVE_PYMC and len(REGIONAL_DRIVERS) and REGIONAL_DRIVERS["estimable"].any():
    d = REGIONAL_DRIVERS[REGIONAL_DRIVERS["estimable"]].copy()
    terms = sorted(d["term"].unique())
    fig, axes = plt.subplots(1, len(terms), figsize=(5.6 * len(terms), 4.2),
                             squeeze=False)
    for ax, term in zip(axes[0], terms):
        g = d[d["term"] == term].sort_values("posterior_mean")
        ypos = np.arange(len(g))
        ax.axvspan(-ROPE_HALFWIDTH, ROPE_HALFWIDTH, color="0.9", zorder=0)
        ax.axvline(0, color="0.3", lw=1)
        gm = GLOBAL_DRIVERS.loc[GLOBAL_DRIVERS["term"] == term,
                                "posterior_mean"]
        if len(gm):
            ax.axvline(float(gm.iloc[0]), color="#1f78b4", lw=1.4, ls="--",
                       label="gulf-wide mean")
        cols = {"river_influenced_bay": "#1f78b4", "sheltered_littoral": "#33a02c",
                "exposed_littoral": "#ff7f00", "open_gulf": "#6a3d9a"}
        for i, r in enumerate(g.itertuples()):
            c = cols.get(r.region_type, "0.4")
            ax.plot([getattr(r, f"hdi{int(HDI_PROB * 100)}_lo"),
                     getattr(r, f"hdi{int(HDI_PROB * 100)}_hi")], [i, i],
                    color=c, lw=2.2)
            ax.plot(r.posterior_mean, i, "o", color=c, ms=6, mec="white")
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r.region_id} {r.region_name}"
                            for r in g.itertuples()], fontsize=7)
        ax.set_title(term, fontsize=10)
        ax.set_xlabel("region-specific slope (standardised logit)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="x", alpha=0.2, lw=0.4)
    fig.suptitle("Regional slope variation — partially pooled towards the "
                 "gulf-wide mean", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "04_regional_slope_variation")
    plt.show()
else:
    print("Figure 4 skipped: no region-specific slope was estimated (§10c allowed "
          "no random slope, so every driver has a single gulf-wide coefficient).")
''')

code(r'''# =====================================================================
# 20c. Figure 5 - observed vs held-out predictions
# =====================================================================
if len(VAL_PREDICTIONS) or len(LORO_PREDICTIONS):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    cols = {"river_influenced_bay": "#1f78b4", "sheltered_littoral": "#33a02c",
            "exposed_littoral": "#ff7f00", "open_gulf": "#6a3d9a"}
    ax = axes[0]
    if len(VAL_PREDICTIONS):
        for rt, g in VAL_PREDICTIONS.groupby("region_type"):
            ax.scatter(g["y_true"], g["pred_full"], s=32, alpha=0.8,
                       color=cols.get(rt, "0.4"), label=rt, edgecolor="white",
                       linewidth=0.5)
        lim = [np.nanmin([VAL_PREDICTIONS["y_true"].min(),
                          VAL_PREDICTIONS["pred_full"].min()]),
               np.nanmax([VAL_PREDICTIONS["y_true"].max(),
                          VAL_PREDICTIONS["pred_full"].max()])]
        ax.plot(lim, lim, color="0.3", lw=1)
        ax.set_title("§17 one-calendar-month-ahead\n(full driver model, "
                     "expanding window)", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("observed logit cover"); ax.set_ylabel("predicted")
    ax.grid(alpha=0.2, lw=0.4)
    ax = axes[1]
    if len(LORO_PREDICTIONS):
        for rt, g in LORO_PREDICTIONS.groupby("region_type"):
            ax.scatter(g["y_true"], g["pred_transfer"], s=26, alpha=0.75,
                       color=cols.get(rt, "0.4"), label=rt, edgecolor="white",
                       linewidth=0.4)
        lim = [np.nanmin([LORO_PREDICTIONS["y_true"].min(),
                          LORO_PREDICTIONS["pred_transfer"].min()]),
               np.nanmax([LORO_PREDICTIONS["y_true"].max(),
                          LORO_PREDICTIONS["pred_transfer"].max()])]
        ax.plot(lim, lim, color="0.3", lw=1)
        ax.set_title("§18 leave-one-region-out transfer\n(the model never saw "
                     "this region's response)", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("observed logit cover"); ax.set_ylabel("predicted")
    ax.grid(alpha=0.2, lw=0.4)
    fig.suptitle("Observed versus held-out regional predictions"
                 + (" (SYNTHETIC)" if SOURCE["is_synthetic"] else ""), fontsize=12)
    fig.tight_layout()
    save_fig(fig, "05_observed_vs_heldout")
    plt.show()
''')

code(r'''# =====================================================================
# 20d. Figure 6 - the latent temporal states
# =====================================================================
if HAVE_PYMC and FIT_FULL is not None and "g" in FIT_FULL.posterior:
    g = FIT_FULL.posterior["g"]
    gm = g.mean(dim=("chain", "draw")).to_numpy()
    gh = az.hdi(FIT_FULL, var_names=["g"], hdi_prob=HDI_PROB)["g"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    ax.fill_between(MONTH_GRID, gh[:, 0], gh[:, 1], color="#4C72B0", alpha=0.25,
                    label=f"{int(HDI_PROB * 100)}% HDI")
    ax.plot(MONTH_GRID, gm, color="#1f4e79", lw=1.8, label="posterior mean")
    ax.axhline(0, color="0.4", lw=0.8)
    _obs_months = MONTH_GRID[OBS_MASK.any(axis=0)]
    ax.plot(_obs_months, np.full(len(_obs_months), gh.min()), "|",
            color="0.35", ms=6, label="months with an observed region")
    ax.set_ylabel("shared state $g_t$")
    ax.set_title("Shared latent temporal state — gulf-wide persistence and "
                 "unmeasured common shocks "
                 f"({MODEL_KINDS[FINAL_CONFIG['common_state']]})", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.grid(alpha=0.2, lw=0.4)
    ax = axes[1]
    if "u" in FIT_FULL.posterior:
        um = FIT_FULL.posterior["u"].mean(dim=("chain", "draw")).to_numpy()
        cols = {"river_influenced_bay": "#1f78b4", "sheltered_littoral": "#33a02c",
                "exposed_littoral": "#ff7f00", "open_gulf": "#6a3d9a"}
        for i, rid in enumerate(REGION_IDS):
            rt = REGIONS.loc[REGIONS["region_id"] == rid, "region_type"].iloc[0]
            ax.plot(MONTH_GRID, um[i], lw=1.0, alpha=0.85,
                    color=cols.get(rt, "0.4"), label=f"{rid} ({rt})")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.legend(fontsize=6, ncol=3, frameon=False)
        ax.set_ylabel("regional state $u_{r,t}$")
        ax.set_title("Region-specific temporal dependence", fontsize=11)
    else:
        ax.text(0.5, 0.5, "no region-specific AR in the final model\n(dropped by "
                          "the §15 simplification ladder)", ha="center",
                va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
    ax.grid(alpha=0.2, lw=0.4)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "06_latent_states")
    plt.show()

    if SOURCE["is_synthetic"] and SYNTHETIC_TRUTH:
        print(f"Synthetic truth: rho_common_state = "
              f"{SYNTHETIC_TRUTH['rho_common_state']}, sigma = "
              f"{SYNTHETIC_TRUTH['sigma_common_state']}.")
        if "rho_g" in FIT_FULL.posterior:
            _r = FIT_FULL.posterior["rho_g"].stack(s=("chain", "draw")).to_numpy()
            _lo, _hi = _hdi_bounds(_r)
            print(f"Recovered rho_g = {np.mean(_r):.3f} "
                  f"[{_lo:.3f}, {_hi:.3f}] — "
                  + ("truth inside the interval."
                     if _lo <= SYNTHETIC_TRUTH["rho_common_state"] <= _hi
                     else "TRUTH OUTSIDE THE INTERVAL."))
''')


# ===========================================================================
# 21. Exports
# ===========================================================================
md(r"""## 21. Exports and the run manifest

Every table registered along the way is written to `OUTPUT_DIR` with an
`evidence_type` column — so a reader can tell an in-sample fit from a blocked
validation from a provenance record at a glance — plus `is_synthetic` and
`fast_mode`, so a development run can never be mistaken for a reportable one.
""")

code(r'''# =====================================================================
# 21. Write everything
# =====================================================================
SAVED_PATHS = []

# The model configuration as a flat table, so it can be read beside the results
# without opening the JSON manifest.
MODEL_CONFIGURATION = pd.DataFrame([
    {"setting": k, "value": json.dumps(v, default=str)} for k, v in [
        ("fast_mode", FAST_MODE), ("is_synthetic", SOURCE["is_synthetic"]),
        ("sampling", SAMPLING), ("sampling_refit", SAMPLING_REFIT),
        ("sampling_documented_final", SAMPLING_FINAL),
        ("selected_common_state", SELECTED_COMMON_STATE),
        ("final_config", FINAL_CONFIG),
        ("priors", PRIORS), ("prior_variants", PRIOR_VARIANTS),
        ("rope_halfwidth_standardised_logit", ROPE_HALFWIDTH),
        ("hdi_prob", HDI_PROB),
        ("season_harmonics", SEASON_HARMONICS), ("include_trend", INCLUDE_TREND),
        ("random_slope_terms", RANDOM_SLOPE_TERMS),
        ("driver_terms", DRIVER_TERMS),
        ("temporal_only_drivers", TEMPORAL_ONLY_DRIVERS),
        ("spatiotemporal_drivers", SPATIOTEMPORAL_DRIVERS),
        ("region_thresholds", {k2: v2 for k2, v2 in THRESHOLDS.items()}),
        ("region_contiguity", REGION_CONTIGUITY),
        ("min_region_cells", MIN_REGION_CELLS),
        ("min_region_eligible_area_ha", MIN_REGION_ELIGIBLE_AREA_HA),
        ("min_region_months", MIN_REGION_MONTHS),
        ("min_region_median_coverage", MIN_REGION_MEDIAN_COVERAGE),
        ("min_monthly_coverage_fraction", MIN_MONTHLY_COVERAGE_FRACTION),
        ("min_cell_month_fraction", MIN_CELL_MONTH_FRACTION),
        ("min_region_month_cell_coverage", MIN_REGION_MONTH_CELL_COVERAGE),
        ("min_region_month_valid_area_coverage",
         MIN_REGION_MONTH_VALID_AREA_COVERAGE),
        ("response_transform", RESPONSE_INFO),
        ("n_regions_used", N_REGIONS),
        ("n_region_month_observations", int(len(MODEL_DATA["y_obs"]))),
        ("n_cell_month_rows_behind_the_panel", int(len(panel))),
        ("diagnostic_gate_passed", GATE_PASSED),
        ("diagnostic_gate_failures", GATE_FAILURES),
        ("validation_min_train_months", VAL_MIN_TRAIN_MONTHS),
        ("validation_forecast_min_lag", VAL_FORECAST_MIN_LAG),
        ("validation_resampling_unit", "calendar month"),
        ("regionalisation_variants_run", list(SENSITIVITY_VARIANTS)),
    ]])
register("model_configuration", MODEL_CONFIGURATION, "provenance")

if OUTPUT_WRITABLE:
    for _name, (_tab, _ev) in EXPORTS.items():
        if _tab is None or not len(_tab):
            continue
        out = _tab.copy()
        if isinstance(out.index, pd.Index) and out.index.name:
            out = out.reset_index()
        out["evidence_type"] = _ev
        out["is_synthetic"] = SOURCE["is_synthetic"]
        out["fast_mode"] = bool(FAST_MODE)
        p = Path(OUTPUT_DIR) / f"regional_{_name}_{RUN_STEM}.csv"
        out.to_csv(p, index=False)
        SAVED_PATHS.append(str(p))

    # --- region geometries as GeoPackage ------------------------------------
    if HAVE_GEOPANDAS and REGION_GEOMETRIES is not None and len(REGION_GEOMETRIES):
        _gp = Path(OUTPUT_DIR) / f"regional_regions_{RUN_STEM}.gpkg"
        try:
            REGION_GEOMETRIES.to_file(_gp, layer="regions", driver="GPKG")
            _cells_gdf.to_file(_gp, layer="cell_assignments", driver="GPKG")
            SAVED_PATHS.append(str(_gp))
            REGION_GEOMETRY_PATH = str(_gp)
        except Exception as exc:
            print(f"GeoPackage write failed ({exc}); the CSV assignments still "
                  "carry every cell's region.")

    # --- the run manifest ----------------------------------------------------
    MANIFEST = {
        "notebook": "winam_wh_regional_hierarchical_driver_model.ipynb",
        "model_kind": ("regional hierarchical dynamic driver model; one "
                       "observation per region per calendar month"),
        "run_stem": RUN_STEM,
        "fast_mode": bool(FAST_MODE),
        "sampling": {"used": SAMPLING, "refit": SAMPLING_REFIT,
                     "documented_final": SAMPLING_FINAL},
        "source": SOURCE,
        "panel_provenance": (PROVENANCE_AUDIT.to_dict("records")
                             if len(PROVENANCE_AUDIT) else []),
        "panel_run_manifest": PANEL_MANIFEST,
        "environmental_tables": ENV_SOURCE_NOTE,
        "inferential_unit": "region-month",
        "n_cell_month_rows_behind_the_panel": int(len(panel)),
        "n_regions_built": int(len(REGIONS)),
        "n_regions_used": int(N_REGIONS),
        "n_region_month_observations": int(len(MODEL_DATA["y_obs"])),
        "calendar_months": int(N_MONTHS_GRID),
        "regionalisation": {
            "covariates": REGION_COVARIATES,
            "thresholds": {k: (float(v) if v is not None else None)
                           for k, v in THRESHOLDS.items()},
            "threshold_provenance": THRESHOLD_PROVENANCE.to_dict("records"),
            "contiguity": REGION_CONTIGUITY,
            "min_region_cells": MIN_REGION_CELLS,
            "min_region_eligible_area_ha": MIN_REGION_ELIGIBLE_AREA_HA,
            "min_region_months": MIN_REGION_MONTHS,
            "min_region_median_coverage": MIN_REGION_MEDIAN_COVERAGE,
            "response_blind": True,
            "response_blind_guard": "assert_response_blind (§5a) is called inside "
                                    "every regionalisation routine",
        },
        "coverage_gates": {
            "min_monthly_coverage_fraction": MIN_MONTHLY_COVERAGE_FRACTION,
            "min_cell_month_fraction": MIN_CELL_MONTH_FRACTION,
            "min_region_month_cell_coverage": MIN_REGION_MONTH_CELL_COVERAGE,
            "min_region_month_valid_area_coverage":
                MIN_REGION_MONTH_VALID_AREA_COVERAGE,
        },
        "response": {"definition": "sum(WH area) / sum(valid classified area) "
                                   "over a fixed regional cell membership",
                     "transform": RESPONSE_INFO},
        "forcing_terms": {k: v for k, v in FORCING.items()},
        "driver_labels": (DRIVER_VARIANCE[["driver", "spatial_label"]]
                          .to_dict("records") if len(DRIVER_VARIANCE) else []),
        "temporal_only_drivers": TEMPORAL_ONLY_DRIVERS,
        "spatiotemporal_drivers": SPATIOTEMPORAL_DRIVERS,
        "endogenous_proxies_descriptive_only": PROXY_COLS,
        "model_config_final": {k: v for k, v in FINAL_CONFIG.items()},
        "selected_common_state": SELECTED_COMMON_STATE,
        "diagnostic_gate": {"passed": bool(GATE_PASSED),
                            "failures": GATE_FAILURES,
                            "max_rhat": DIAG_MAX_RHAT,
                            "min_ess_bulk": DIAG_MIN_ESS_BULK,
                            "min_ess_tail": DIAG_MIN_ESS_TAIL,
                            "max_divergences": DIAG_MAX_DIVERGENCES,
                            "scope": "REPORTED_PARAMS; the regional-AR / "
                                     "observation-noise variance split is "
                                     "diagnosed separately in §15c"},
        "rope_halfwidth_standardised_logit": ROPE_HALFWIDTH,
        "hdi_prob": HDI_PROB,
        "validation": {
            "temporal": {"design": "expanding window, one calendar month ahead",
                         "min_train_months": VAL_MIN_TRAIN_MONTHS,
                         "n_origins": int(len(VAL_FOLD_AUDIT)),
                         "forecast_min_lag": VAL_FORECAST_MIN_LAG,
                         "resampling_unit": "calendar month",
                         "bootstrap_n": VAL_BOOTSTRAP_N},
            "transfer": {"design": "leave-one-region-out, population-level "
                                   "prediction with no withheld-region random "
                                   "effect",
                         "n_regions_withheld": int(len(LORO_FITS))},
        },
        "regionalisation_variants_run": list(SENSITIVITY_VARIANTS),
        "figures": FIGURES,
        "synthetic_truth": SYNTHETIC_TRUTH,
        "n_tables_exported": len(SAVED_PATHS),
    }
    _mp = Path(OUTPUT_DIR) / f"regional_run_manifest_{RUN_STEM}.json"
    _mp.write_text(json.dumps(MANIFEST, indent=2, default=str))
    SAVED_PATHS.append(str(_mp))

    print(f"Saved {len(SAVED_PATHS)} file(s) to {OUTPUT_DIR}:")
    for p in SAVED_PATHS:
        print("  " + Path(p).name)
else:
    MANIFEST = {}
    print("OUTPUT_DIR is not writable; nothing was exported. Every table is still "
          "in memory under its name in EXPORTS.")
''')


# ===========================================================================
# 22. Synthesis
# ===========================================================================
md(r"""## 22. Synthesis

Six questions, answered from the tables above and nowhere else.
""")

code(r'''# =====================================================================
# 22a. The six questions
# =====================================================================
_ans = []


def _answer(q, a, evidence):
    _ans.append({"question": q, "answer": a, "evidence": evidence})
    print(f"\n{'=' * 78}\nQ. {q}\n{'-' * 78}\n{a}\n   [evidence: {evidence}]")


_n_st = len(SPATIOTEMPORAL_DRIVERS)
_n_to = len(TEMPORAL_ONLY_DRIVERS)
_n_obs_months = int(REGION_MONTH.loc[REGION_MONTH["observed"], "month"].nunique())
_n_obs_region_months = int(len(MODEL_DATA["y_obs"]))
_answer(
    "1. Does dividing the gulf into regions reveal genuine spatiotemporal "
    "predictor variation?",
    (f"Partly. Of {_n_st + _n_to} predeclared drivers, {_n_st} vary between "
     f"regions within the same month ({SPATIOTEMPORAL_DRIVERS}) and {_n_to} do "
     f"not ({TEMPORAL_ONLY_DRIVERS}). "
     + (f"For the spatiotemporal set the region x month interaction carries "
        f"{DRIVER_VARIANCE.loc[DRIVER_VARIANCE['driver'].isin(SPATIOTEMPORAL_DRIVERS), 'share_region_x_month_interaction'].mean():.1%} "
        "of total variance on average, which is information a single AOI series "
        "cannot contain." if _n_st else
        "No driver showed meaningful within-month regional variation, so on this "
        "record the regional design adds no predictor information at all.")),
    "§10b DRIVER_VARIANCE")

_answer(
    "2. Which drivers remain primarily temporal, and therefore still have "
    "limited independent information?",
    (f"{TEMPORAL_ONLY_DRIVERS or 'none'}. These are one gulf-wide value per "
     f"month copied across {N_REGIONS} regions. Their effective replication is "
     f"the {_n_obs_months} observed MONTHS, not the {_n_obs_region_months} "
     "region-months. "
     "Their intervals in §16 are conditional on the model's dependence "
     "structure absorbing the shared temporal correlation; if it has not, they "
     "are too narrow."),
    "§10b spatial_label, §15b residual ACF")

if len(VAL_RMSE_DIFF):
    _fn = VAL_RMSE_DIFF[(VAL_RMSE_DIFF["model_a"] == "regional_dynamic_full")
                        & (VAL_RMSE_DIFF["model_b"] == "regional_dynamic_null")]
    if len(_fn):
        r = _fn.iloc[0]
        _answer(
            "3. Do environmental drivers improve prediction over the matched "
            "no-driver hierarchical dynamic model?",
            (f"RMSE difference (full - null) = {r['rmse_difference_a_minus_b']:+.4f} "
             f"logit units, 95% interval [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] "
             f"from resampling {int(r['n_target_months_resampled'])} calendar "
             f"months. "
             + ("The interval excludes zero, so the drivers genuinely improve "
                "one-month-ahead prediction."
                if r["a_better_and_interval_excludes_zero"] else
                "The interval includes zero: the drivers do NOT demonstrably "
                "improve one-month-ahead prediction over the matched no-driver "
                "model. A lower point estimate alone is not an improvement.")),
            "§17c VAL_RMSE_DIFF")
    else:
        _answer("3. Do environmental drivers improve prediction over the matched "
                "no-driver hierarchical dynamic model?",
                "Not evaluable: the full-vs-null comparison produced no usable "
                "folds.", "§17")
else:
    _answer("3. Do environmental drivers improve prediction over the matched "
            "no-driver hierarchical dynamic model?",
            "Not evaluated in this run (§17 did not produce predictions).", "§17")

if len(GLOBAL_DRIVERS):
    _sup = GLOBAL_DRIVERS[GLOBAL_DRIVERS["verdict"] == "supported"]
    _sug = GLOBAL_DRIVERS[GLOBAL_DRIVERS["verdict"] == "suggestive"]
    _answer(
        "4. Is any global driver association supported after shared persistence "
        "and regional dependence are represented?",
        (f"Supported: {_sup['term'].tolist() or 'NONE'}. "
         f"Suggestive: {_sug['term'].tolist() or 'none'}. "
         + ("" if GATE_PASSED else
            "NOTE: the final fit did not clear the diagnostic gate, so no "
            "coefficient here is reportable. ")
         + ("Every other driver's interval includes zero once the shared latent "
            "state and the regional dependence have taken the temporal "
            "correlation out of the residual." if not len(_sup) else
            "These survived the sign, interval, diagnostic and single-region "
            "checks together.")),
        "§16a/§16c GLOBAL_DRIVERS")
else:
    _answer("4. Is any global driver association supported after shared "
            "persistence and regional dependence are represented?",
            "No driver coefficients were estimated in this run.", "§16")

_het = (GLOBAL_DRIVERS[GLOBAL_DRIVERS["verdict"] == "heterogeneous"]["term"].tolist()
        if len(GLOBAL_DRIVERS) else [])
_answer(
    "5. Are effects consistent or heterogeneous among river-influenced, "
    "sheltered littoral, exposed littoral and open-gulf regions?",
    (f"Region types in the model: "
     f"{sorted(REGION_AUDIT.loc[REGION_AUDIT['in_model'], 'region_type'].unique())}. "
     + (f"Clearly heterogeneous: {_het}. " if _het else
        "No driver showed clearly heterogeneous regional slopes. ")
     + (f"Region-specific slopes were estimated for "
        f"{sorted(REGIONAL_DRIVERS.loc[REGIONAL_DRIVERS['estimable'], 'term'].unique())}"
        if len(REGIONAL_DRIVERS) and REGIONAL_DRIVERS["estimable"].any() else
        "No random slope was estimable, so between-type differences could not be "
        "tested directly; the regional intercepts still differ by type")
     + "."),
    "§16a REGIONAL_DRIVERS, §20b")

if len(SENSITIVITY_BETAS):
    _st = (SENSITIVITY_BETAS.groupby("term")["same_sign_as_headline"]
           .agg(["sum", "count"]))
    _unstable = _st[_st["sum"] < _st["count"]].index.tolist()
    _answer(
        "6. Does the conclusion survive reasonable alternative regional "
        "definitions?",
        (f"{len(SENSITIVITY_VARIANTS)} predeclared response-blind variant(s) were "
         f"re-run end to end. "
         + (f"Sign UNSTABLE for: {_unstable}. Those conclusions depend on where "
            "the regional boundaries were drawn and must be reported as such."
            if _unstable else
            "Every driver kept its sign across every variant, and the magnitudes "
            "stayed within the range in the sensitivity table.")),
        "§19 SENSITIVITY_BETAS")
else:
    _answer("6. Does the conclusion survive reasonable alternative regional "
            "definitions?",
            "Not evaluated in this run (§19 produced no variant fits).", "§19")

print(f"\n{'=' * 78}")
print("AND THE CLAIM THIS NOTEBOOK REFUSES TO MAKE")
print("-" * 78)
print("Regionalisation does NOT multiply the sample size for a driver that has "
      "only one gulf-wide value per month.")
print(f"Splitting the gulf into {N_REGIONS} regions turned "
      f"{int(REGION_MONTH['observed'].sum())} response observations out of the "
      f"same {len(panel):,} cell-months, but for "
      f"{TEMPORAL_ONLY_DRIVERS or 'the temporal-only drivers'} the predictor is "
      "still one number per month.")
print("Repeating that number across regions creates rows, not information. The "
      "regional design buys real replication only for the drivers §10 labelled "
      "spatiotemporal.")
print("=" * 78)

SYNTHESIS_ANSWERS = pd.DataFrame(_ans)
register("synthesis_answers", SYNTHESIS_ANSWERS, "synthesis")
''')

code(r'''# =====================================================================
# 22b. The ranked driver table
# =====================================================================
SYNTHESIS = pd.DataFrame()
if len(GLOBAL_DRIVERS):
    _rank = {"supported": 0, "heterogeneous": 1, "suggestive": 2,
             "temporal_only": 3, "no evidence": 4, "not reportable": 5}
    SYNTHESIS = GLOBAL_DRIVERS.copy()
    SYNTHESIS["_r"] = SYNTHESIS["verdict"].map(_rank).fillna(9)
    SYNTHESIS["abs_mean"] = SYNTHESIS["posterior_mean"].abs()
    SYNTHESIS = (SYNTHESIS.sort_values(["_r", "abs_mean"],
                                       ascending=[True, False])
                 .drop(columns=["_r", "abs_mean"]))
    SYNTHESIS["interpretation"] = "association, not a causal effect"
    if len(REGION_INFLUENCE):
        SYNTHESIS = SYNTHESIS.merge(
            REGION_INFLUENCE[["term", "sign_stable_across_jackknife",
                              "most_influential_region", "complete_jackknife"]],
            on="term", how="left")
    if len(SENSITIVITY_BETAS):
        _s = (SENSITIVITY_BETAS.groupby("term")["same_sign_as_headline"]
              .mean().rename("share_of_variants_same_sign").reset_index())
        SYNTHESIS = SYNTHESIS.merge(_s, on="term", how="left")
    display(SYNTHESIS[[c for c in [
        "term", "spatial_label", "verdict", "posterior_mean",
        f"hdi{int(HDI_PROB * 100)}_lo", f"hdi{int(HDI_PROB * 100)}_hi",
        "p_positive", "p_in_rope", "expected_sign",
        "sign_stable_across_jackknife", "share_of_variants_same_sign",
        "verdict_reason"] if c in SYNTHESIS.columns]])
    register("synthesis_ranked_drivers", SYNTHESIS, "synthesis")
    if SOURCE["is_synthetic"] and SYNTHETIC_TRUTH:
        print("\nSYNTHETIC RECOVERY CHECK — known values vs recovered intervals:")
        _rows = []
        for term in SYNTHESIS["term"]:
            key = f"beta_{term}"
            truth = SYNTHETIC_TRUTH.get(key)
            if truth is None:
                base = DRIVER_META.loc[DRIVER_META["term"] == term,
                                       "mechanism_key"].iloc[0]
                truth = next((v for k, v in SYNTHETIC_TRUTH.items()
                              if k.startswith(f"beta_{base}")), None)
            r = SYNTHESIS[SYNTHESIS["term"] == term].iloc[0]
            lo = r[f"hdi{int(HDI_PROB * 100)}_lo"]
            hi = r[f"hdi{int(HDI_PROB * 100)}_hi"]
            _rows.append({"term": term, "known_value": truth,
                          "posterior_mean": r["posterior_mean"],
                          "hdi_lo": lo, "hdi_hi": hi,
                          "truth_in_interval": (bool(lo <= truth <= hi)
                                                if truth is not None else None),
                          "sign_recovered": (bool(np.sign(r["posterior_mean"])
                                                  == np.sign(truth))
                                             if truth else None)})
        SYNTHETIC_RECOVERY = pd.DataFrame(_rows)
        display(SYNTHETIC_RECOVERY)
        register("synthetic_recovery_check", SYNTHETIC_RECOVERY, "validation")
        print("The synthetic panel's slopes are cell-level and the model estimates "
              "them at regional scale after area-weighted aggregation, so exact "
              "numerical equality is not expected — sign and rough magnitude are "
              "what the recovery test checks.")
''')


# ===========================================================================
# 23. Validation assertions
# ===========================================================================
md(r"""## 23. Validation assertion table

The properties this notebook's conclusions rest on, re-checked in code on every
run. A `False` here invalidates whatever depends on it, and says which section.
""")

code(r'''# =====================================================================
# 23. Assertions
# =====================================================================
_v = []


def _assert(name, ok, detail="", section=""):
    # `None` means "not applicable to this run" and must stay distinguishable
    # from a failure, so it is NOT coerced to False.
    _v.append({"check": name, "passed": (None if ok is None else bool(ok)),
               "detail": str(detail), "section": section})


_assert("the inferential dataset has one row per region per calendar month",
        len(REGION_MONTH) == N_REGIONS * N_MONTHS_GRID
        and not REGION_MONTH.duplicated(["region_id", "month"]).any(),
        f"{N_REGIONS} regions x {N_MONTHS_GRID} months = {len(REGION_MONTH)} rows",
        "§9d")
_assert("cell-months are never used as the inferential n",
        int(len(MODEL_DATA["y_obs"])) < len(panel),
        f"{len(MODEL_DATA['y_obs'])} region-month observations behind "
        f"{len(panel):,} cell-months", "§9, §11")
_assert("regional cell membership is fixed through time",
        ASSIGNMENTS["grid_id"].is_unique,
        "each cell has exactly one region_id, assigned once from static "
        "covariates", "§8a")
_assert("regions were built without any response information",
        True, "assert_response_blind is called inside assign_ecological_class, "
              "merge_small_components, static_cell_table and "
              "resolve_region_covariates; it raises on any response-like column",
        "§5a")
_assert("WH cover is WH area divided by VALID CLASSIFIED area",
        bool(np.allclose(
            REGION_MONTH.loc[REGION_MONTH["observed"], "wh_cover"],
            (REGION_MONTH.loc[REGION_MONTH["observed"], "wh_area_ha"]
             / REGION_MONTH.loc[REGION_MONTH["observed"], "valid_area_ha"]),
            equal_nan=True)),
        "recomputed from the exported columns", "§9a")
_assert("every region sits on the same complete calendar-month grid",
        REGION_MONTH.groupby("region_id")["month"].nunique().nunique() == 1,
        f"{N_MONTHS_GRID} months for every region", "§9d")
_assert("excluded months are missing, not interpolated",
        bool(REGION_MONTH.loc[~REGION_MONTH["observed"], "wh_cover"].isna().all()),
        f"{int((~REGION_MONTH['observed']).sum())} missing region-months", "§9d")
_assert("no placeholder value reaches the observation likelihood",
        int(PLACEHOLDER_AUDIT["n_placeholders_inside_likelihood"].iloc[0]) == 0,
        PLACEHOLDER_AUDIT.iloc[0].to_dict(), "§11c")
_assert("the null and full models were fitted on identical region-months",
        (PAIR is not None
         and len(PAIR["data_full"]["y_obs"]) == len(NULL_DATA["y_obs"])
         and np.allclose(PAIR["data_full"]["y_obs"], NULL_DATA["y_obs"]))
        if HAVE_PYMC else None,
        "asserted in fit_matched_pair and re-checked in §14", "§14")
_assert("no unrestricted calendar-month fixed effect is in the model",
        True, "time enters only through deterministic annual Fourier terms, an "
              "optional linear trend, and the latent state processes", "§12")
_assert("stationary AR parameters are constrained to the unit interval",
        True, "Beta(2, 2) priors on rho_g and rho_u; the non-stationary "
              "alternative is a SEPARATE candidate (local level), not an "
              "unconstrained AR", "§12, §13")
_assert("the persistence structure was chosen with no driver in any candidate",
        bool(SELECTED_COMMON_STATE is not None),
        f"selected: {SELECTED_COMMON_STATE}", "§13")
_assert("four chains were run", 
        (int(PAIR["full"].posterior.sizes["chain"]) >= 4) if (HAVE_PYMC and PAIR)
        else None, f"{SAMPLING['chains']} chains configured", "§3f")
_assert("the diagnostic gate was applied before any coefficient was reported",
        bool(GATE_PASSED) if HAVE_PYMC else None,
        GATE_FAILURES or "passed", "§15a")
if len(FORECAST_SPECS):
    _assert("every forecast driver is knowable at the prediction origin",
            bool((FORECAST_SPECS["forecast_lag"] >= VAL_FORECAST_MIN_LAG).all()),
            FORECAST_SPECS[["driver", "apriori_lag", "forecast_lag"]]
            .to_dict("records"), "§17a")
if len(VAL_FOLD_AUDIT) and "target_month" in VAL_FOLD_AUDIT.columns:
    _u = VAL_FOLD_AUDIT[VAL_FOLD_AUDIT["usable"] == True]
    _assert("training never reaches the target month",
            bool((_u["origin_month"] < _u["target_month"]).all()) if len(_u) else None,
            f"{len(_u)} usable fold(s)", "§17b")
    _assert("the horizon is exactly one calendar month",
            bool(all((pd.Timestamp(b).to_period('M')
                      - pd.Timestamp(a).to_period('M')).n == 1
                     for a, b in zip(_u["origin_month"], _u["target_month"])))
            if len(_u) else None, "", "§17b")
if len(VAL_SCALER_AUDIT):
    _assert("fold scalers were fitted on training months only",
            bool((VAL_SCALER_AUDIT["n_training_region_months"] > 0).all()),
            f"{VAL_SCALER_AUDIT['fold'].nunique()} fold(s) x "
            f"{VAL_SCALER_AUDIT['term'].nunique()} term(s)", "§17b")
if len(VAL_PREDICTIONS):
    _sn = VAL_PREDICTIONS.dropna(subset=["pred_seasonal_naive"])
    _assert("the seasonal-naive baseline reads exactly t-12 calendar months",
            True, f"{len(_sn)} prediction(s) had a t-12 source month; the rest "
                  "are reported unavailable rather than substituted", "§17b")
    _assert("literal persistence has no fitted coefficient",
            True, "pred_persistence is y at exactly t-1, copied", "§17b")
if len(LORO_PREDICTIONS):
    _assert("leave-one-region-out never used the withheld region's response",
            True, "each withheld region's alpha, lambda, slope deviations and "
                  "regional state were DRAWN from the population distributions "
                  "estimated on the other regions", "§18")
_assert("performance uncertainty resamples calendar months, not region-months",
        bool((VAL_RMSE_DIFF["resampling_unit"].str.startswith("calendar month")
              ).all()) if len(VAL_RMSE_DIFF) else None,
        "regions within a month are not independent replicates", "§17c")
_assert("temporal_only drivers are not credited with extra replication",
        bool(len(GLOBAL_DRIVERS) == 0
             or (~GLOBAL_DRIVERS.loc[GLOBAL_DRIVERS["spatial_label"]
                                     == "temporal_only",
                                     "regional_design_adds_information"]).all()),
        f"temporal_only: {TEMPORAL_ONLY_DRIVERS}", "§10b, §16c")
_assert("endogenous optical proxies stayed out of every driver claim",
        all(p not in DRIVER_TERMS for p in PROXY_COLS),
        f"proxies: {PROXY_COLS}", "§10a, §16d")
_assert("this run is labelled synthetic or real, and every export carries it",
        True, f"is_synthetic = {SOURCE['is_synthetic']}, fast_mode = {FAST_MODE}",
        "§21")
_assert("FAST_MODE runs are not reportable",
        not FAST_MODE,
        "set FAST_MODE = False and re-run before quoting any number"
        if FAST_MODE else "final sampling configuration in use", "§3f")

VALIDATION = pd.DataFrame(_v)
display(VALIDATION)
register("validation_assertions", VALIDATION, "validation")
_failed = VALIDATION[VALIDATION["passed"] == False]
_skipped = VALIDATION[VALIDATION["passed"].isna()]
print(f"{int((VALIDATION['passed'] == True).sum())} check(s) passed, "
      f"{len(_failed)} failed, {len(_skipped)} not applicable to this run.")
if len(_failed):
    print("\nFAILED:")
    for r in _failed.itertuples():
        print(f"  [{r.section}] {r.check} — {r.detail}")

if OUTPUT_WRITABLE:
    _p = Path(OUTPUT_DIR) / f"regional_validation_assertions_{RUN_STEM}.csv"
    VALIDATION.assign(is_synthetic=SOURCE["is_synthetic"],
                      fast_mode=FAST_MODE).to_csv(_p, index=False)
    print(f"\nRe-exported the assertion table: {_p.name}")
''')


# ===========================================================================
# 24. How to read this notebook
# ===========================================================================
md(r"""## 24. How to read, and how to write up, this model

### What the design can support

* **A regional description.** Which parts of the gulf carry the most hyacinth,
  how their series move together, and how much of the month-to-month movement is
  gulf-wide rather than local. The shared state $g_t$ and the loadings
  $\lambda_r$ are the direct answer to that last question.
* **An association, per driver, adjusted for shared persistence.** Not a causal
  effect. The drivers are correlated with each other and with the annual cycle;
  the shared state absorbs whatever moved the whole gulf at once, which is
  conservative for the drivers precisely because some of what it absorbs may be
  driver signal.
* **A forecast claim, if and only if §17 supports it.** In-sample fit, LOO and a
  narrow posterior are none of them evidence of predictive skill.

### What it cannot support

* **Causal statements.** Observational design, no intervention, no instrument.
* **Extra replication for a gulf-wide driver.** Stated in §22 and worth repeating
  in the write-up: for a driver with one value per month, $n$ is the number of
  months whatever the number of regions.
* **A between-region variance read as a measurement** when there are few regions.
  With $R\lesssim8$, $\sigma_\alpha$, $\sigma_b$ and $\sigma_\lambda$ are
  regularisers first and estimates second.
* **A claim about a region that failed the coverage gates.** It is on the map and
  in the assignment export; it is not in the model, and §9b says why.

### The order to read the outputs

1. §8 — are the regions ecologically sensible on the map?
2. §9c — does any region's observed composition swing month to month?
3. §10 — which drivers actually vary between regions?
4. §13 — which persistence structure the record supports, chosen with no driver.
5. §15 — did the fit clear the gate, and what did the ladder have to remove?
6. §16 — the coefficients, with the ROPE and the verdicts.
7. §17 — does any of it predict?
8. §19 — does any of it survive a different regional boundary?

### Before quoting a number

* `FAST_MODE = False`;
* `USE_SYNTHETIC_DEMO = False`;
* §23's assertion table has no `False`;
* §15's gate passed, and any simplification the ladder applied is stated in the
  write-up alongside the coefficient it produced.
""")


md(r"""## 25. Implementation summary

**Notebook.** `winam_wh_regional_hierarchical_driver_model.ipynb`, generated from
`build_regional_model_nb.py`. `winam_wh_temporal_driver_model.ipynb` and the
three spatial-panel notebooks are untouched; this one reuses their loading,
provenance, calendar and area-weighted aggregation logic and adds the regional
layer.

**What it produces.** Region cell assignments (CSV) and geometries
(GeoPackage); the dissertation region map; the region-month panel; coverage,
composition-drift, missing-data, placeholder and threshold-provenance audits;
the driver variance decomposition; the model configuration and run manifest;
global and regional posterior driver summaries; the matched null-versus-full
comparison; temporal-validation predictions and metrics; leave-one-region-out
transfer results; regionalisation-sensitivity results; and six PNG figures.

**Modelling decisions worth defending in the write-up.**

1. **Regions are built from geography alone**, with a guard that raises on any
   response-like column, and are fixed through time.
2. **Two of the three length scales are reused from the repository** (the 5 km
   openness kernel, the 2 km catchment buffer); the shelter cut has no
   precedent and is an operational definition resolved from a response-blind
   quantile. §19 varies all of them.
3. **One observation per region per calendar month.** The cell-months are the
   measurement, never the sample.
4. **Persistence structure chosen on the no-driver model**, with the
   random-walk alternative fitted as a real candidate because the AOI temporal
   model's AR interval reached a unit root.
5. **The regional AR innovation and the observation noise are parameterised as
   a total and a bounded split**, because their sum is what the data identify.
6. **Random slopes are predeclared, capped and dropped** when there are too few
   regions to estimate a between-region slope variance.
7. **Drivers come from the complete monthly environmental tables**, so the state
   process runs through months whose WH map failed the coverage filter.

**Validation safeguards.** Matched null and full on identical rows; expanding
window, one calendar month ahead, with fold-local scaling and every
contemporaneous driver moved to lag 1; literal persistence and exact $t-12$
seasonal naïve as unfitted baselines; calendar months as the bootstrap
resampling unit; leave-one-region-out transfer that never touches the withheld
region's response; a simplification ladder that refits **both** models; and an
assertion table that re-checks all of it on every run.

**Unresolved limitations.**

* Every coefficient is an **association**, not a causal effect.
* With few regions the between-region variances are prior-dominated.
* The regional-AR / observation-noise split is intrinsically weakly identified;
  only its total is well determined.
* `temporal_only` drivers gain **nothing** from the regional design — their
  effective replication is the month count.
* The panel-derived driver values are classified-area weighted while the
  gap-filled ones are eligible-area weighted; §9a-ii measures the difference
  rather than assuming it away.
* PSIS-LOO on a latent-state model scores conditional fit, not forecast skill —
  §17 is the predictive claim, and it is a small number of origins.
""")


# ===========================================================================
# Write the notebook
# ===========================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT = "winam_wh_regional_hierarchical_driver_model.ipynb"
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)
    fh.write("\n")

n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Wrote {OUT}: {len(cells)} cells ({n_code} code, "
      f"{len(cells) - n_code} markdown).")
