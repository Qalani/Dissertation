"""Behavioural tests for the shared spatial-panel corrections (§4e and the cells that use it).

The helpers live in the ``## 4e. Shared corrections`` cell of every
``winam_wh_spatial_panel*.ipynb``, so they are executed straight out of the notebook
JSON rather than copied — the tests therefore track the real notebooks instead of a
snapshot that can drift away from them. Pure numpy/pandas: no Drive, no Earth Engine,
no R, no LightGBM. The GeoPackage test skips itself when geopandas is unavailable.

What is pinned:

1. the modelling response is hard classified WH area / valid classified area, and a
   soft (confidence-weighted) response is rejected loudly;
2. months below the minimum monthly coverage are excluded, and lags rebuilt after
   that exclusion leave a genuine calendar gap instead of reaching across it;
3. the all-zero cover baseline is present alongside persistence and climatology
   everywhere skill is scored;
4. the climatology baseline's residual about the anomaly reference is identically
   zero, so its temporal-residual correlation is undefined (N/A) rather than ~0.85;
5. a constant prediction yields NaN, not a ``ConstantInputWarning``;
6. writing the same GeoPackage layer twice succeeds and leaves other layers alone.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(REPO.glob("winam_wh_spatial_panel*.ipynb"))
SHARED_MARKER = "4e. Shared corrections"
COVERAGE_MARKER = "9d. Monthly coverage: measure it, audit it"


def _code_cells(notebook):
    return ["".join(c["source"]) for c in json.loads(notebook.read_text())["cells"]
            if c["cell_type"] == "code"]


def _cell(notebook, marker):
    hits = [s for s in _code_cells(notebook) if marker in s]
    assert hits, f"{notebook.name}: no code cell contains {marker!r}"
    return hits[0]


def _shared_namespace(notebook):
    ns: dict = {}
    exec(compile(_cell(notebook, SHARED_MARKER), "<shared>", "exec"), ns)
    return ns


@pytest.fixture(params=NOTEBOOKS, ids=lambda p: p.stem)
def notebook(request):
    return request.param


@pytest.fixture
def ns(notebook):
    return _shared_namespace(notebook)


# ---------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------
def _panel(n_cells=40, n_months=36, seed=0, cell_area_m2=250_000.0):
    """A hard-class panel: wh_cover is always wh_area_m2 / valid_area_m2."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    gid = np.repeat(np.arange(n_cells), n_months)
    mo = pd.DatetimeIndex(np.tile(months.values, n_cells))
    n = len(gid)
    valid = np.full(n, cell_area_m2)
    cover = np.clip(rng.beta(0.3, 8, n), 0, 1)
    panel = pd.DataFrame({
        "grid_id": gid, "month": mo, "month_num": mo.month,
        "valid_pixels": 1000, "valid_area_m2": valid,
        "wh_area_m2": cover * valid,
    })
    panel["wh_cover"] = panel["wh_area_m2"] / panel["valid_area_m2"]
    panel["wh_cover_hard"] = panel["wh_cover"]
    panel["wh_present"] = (panel["wh_cover"] >= 0.02).astype(int)
    return panel.sort_values(["grid_id", "month"]).reset_index(drop=True)


# =====================================================================
# 1. Hard-cover response construction
# =====================================================================
def test_hard_cover_response_accepts_a_hard_class_panel(ns):
    assert ns["assert_hard_cover_response"](_panel()) is True


def test_hard_cover_response_rejects_a_confidence_weighted_response(ns):
    """A soft response is exactly what the assertion exists to catch."""
    panel = _panel()
    # The old behaviour: swap in a confidence-weighted cover while leaving the hard
    # WH area untouched, so cover no longer equals wh_area / valid_area.
    panel["wh_cover"] = panel["wh_cover"] * 0.87
    with pytest.raises(AssertionError, match="HARD classified WH fraction"):
        ns["assert_hard_cover_response"](panel)


def test_hard_cover_response_rejects_a_response_that_left_wh_cover_hard_behind(ns):
    panel = _panel()
    panel["wh_cover"] = panel["wh_cover"] * 0.9
    panel["wh_area_m2"] = panel["wh_cover"] * panel["valid_area_m2"]  # self-consistent...
    with pytest.raises(AssertionError, match="wh_cover_hard"):       # ...but not hard
        ns["assert_hard_cover_response"](panel)


def test_confidence_columns_are_refused_in_a_feature_set(ns):
    ns["assert_no_confidence_columns"](["wh_cover_lag1", "depth_m", "month_sin"])
    for bad in ("wh_conf_mean", "wh_conf_all_mean", "wh_cover_soft"):
        with pytest.raises(AssertionError):
            ns["assert_no_confidence_columns"](["depth_m", bad])


def test_notebook_configures_a_hard_class_response(notebook):
    code = "\n".join(_code_cells(notebook))
    assert "USE_PROBABILITY_RESPONSE = False" in code
    assert "WEIGHT_COVER_BY_CONFIDENCE = False" in code
    assert "USE_PROBABILITY_RESPONSE = True" not in code
    assert "WEIGHT_COVER_BY_CONFIDENCE = True" not in code


def test_provenance_checks_are_preserved(notebook):
    """The strict classifier-run / export-token / filename-prefix checks must survive."""
    code = "\n".join(_code_cells(notebook))
    assert "ACTIVE_CLASSIFIER_VERSION" in code
    assert "REQUIRED_EXPORT_TOKEN_BY_SENSOR" in code
    assert "REQUIRED_CLASSIFIED_PREFIX_BY_SENSOR" in code
    assert "SUPPLEMENT_RUN_LOG_WITH_FOLDER = False" in code
    assert "def validate_panel_provenance" in code


# =====================================================================
# 2. Monthly-coverage filtering and calendar-gap handling
# =====================================================================
def _thin_months(panel, months, keep_cells):
    """Drop all but `keep_cells` cells in the given months (a partly-observed month)."""
    drop = panel["month"].isin(months) & (panel["grid_id"] >= keep_cells)
    return panel[~drop].reset_index(drop=True)


def test_coverage_audit_measures_observed_fraction_and_valid_area(ns):
    panel = _panel(n_cells=40, n_months=6)
    eligible = np.sort(panel["grid_id"].unique())
    thin = pd.Timestamp("2019-03-01")
    panel = _thin_months(panel, [thin], keep_cells=10)   # 10/40 = 25% coverage

    audit = ns["monthly_coverage_audit"](panel, eligible, min_coverage=0.90)
    row = audit.set_index("month").loc[thin]
    assert row["n_cells_eligible"] == 40
    assert row["n_cells_observed"] == 10
    assert row["coverage_fraction"] == pytest.approx(0.25)
    assert not row["retained"]
    assert "MIN_MONTHLY_COVERAGE_FRACTION" in row["exclusion_reason"]
    # Valid water area is reported and is proportionally smaller in the thin month.
    full = audit[audit["retained"]]["valid_area_ha"].iloc[0]
    assert row["valid_area_ha"] == pytest.approx(full * 0.25)
    assert audit["retained"].sum() == 5


def test_coverage_filter_removes_only_the_thin_months(ns):
    panel = _panel(n_cells=40, n_months=6)
    eligible = np.sort(panel["grid_id"].unique())
    thin = pd.Timestamp("2019-03-01")
    panel = _thin_months(panel, [thin], keep_cells=10)
    audit = ns["monthly_coverage_audit"](panel, eligible, min_coverage=0.90)

    filtered = ns["apply_monthly_coverage_filter"](panel, audit)
    assert thin not in set(filtered["month"])
    assert filtered["month"].nunique() == 5
    assert set(ns["sufficiently_covered_months"](audit)) == set(filtered["month"])


def test_threshold_is_configurable(ns):
    panel = _panel(n_cells=40, n_months=4)
    eligible = np.sort(panel["grid_id"].unique())
    panel = _thin_months(panel, [pd.Timestamp("2019-02-01")], keep_cells=30)  # 75%
    assert ns["monthly_coverage_audit"](panel, eligible, min_coverage=0.90)["retained"].sum() == 3
    assert ns["monthly_coverage_audit"](panel, eligible, min_coverage=0.50)["retained"].sum() == 4


def test_calendar_lag_leaves_a_gap_missing_after_filtering(ns):
    """A dropped month must NOT be treated as the preceding month."""
    panel = _panel(n_cells=5, n_months=6)
    dropped = pd.Timestamp("2019-03-01")
    filtered = panel[panel["month"] != dropped].reset_index(drop=True)

    lagged = ns["calendar_lag"](filtered, ["wh_cover"], periods=1)
    april = lagged[lagged["month"] == pd.Timestamp("2019-04-01")]
    assert april["wh_cover_lag1"].isna().all(), (
        "April's lag must be NaN once March is excluded, not February's cover")
    # A month whose predecessor survives still gets a true t-1 value.
    may = lagged[lagged["month"] == pd.Timestamp("2019-05-01")].set_index("grid_id")
    april_obs = filtered[filtered["month"] == pd.Timestamp("2019-04-01")].set_index("grid_id")
    assert np.allclose(may["wh_cover_lag1"], april_obs.loc[may.index, "wh_cover"])


def test_positional_shift_would_have_reached_across_the_gap(ns):
    """The failure the calendar-aware lag prevents, pinned so it cannot come back."""
    panel = _panel(n_cells=3, n_months=6)
    filtered = panel[panel["month"] != pd.Timestamp("2019-03-01")].reset_index(drop=True)
    positional = filtered.groupby("grid_id")["wh_cover"].shift(1)
    april_mask = filtered["month"] == pd.Timestamp("2019-04-01")
    assert positional[april_mask].notna().all(), "sanity: the naive shift does fill the gap"
    calendar = ns["calendar_lag"](filtered, ["wh_cover"])["wh_cover_lag1"]
    assert calendar[april_mask].isna().all()


def test_calendar_gap_report_counts_the_gaps(ns):
    panel = _panel(n_cells=4, n_months=5)
    filtered = panel[panel["month"] != pd.Timestamp("2019-03-01")].reset_index(drop=True)
    report = ns["calendar_gap_report"](filtered)
    # First month (4 cells) + the month after the gap (4 cells).
    assert report["n_rows_without_previous_month"] == 8
    assert report["n_rows"] == 16


def test_notebook_filters_coverage_before_feature_engineering(notebook):
    """The audit/filter cell must sit ahead of the lag-building cell."""
    cells = _code_cells(notebook)
    cov = next(i for i, s in enumerate(cells) if COVERAGE_MARKER in s)
    lags = next(i for i, s in enumerate(cells) if 'panel["month_sin"] = np.sin' in s)
    assert cov < lags, "monthly-coverage filtering must precede feature engineering"
    code = "\n".join(cells)
    assert "MIN_MONTHLY_COVERAGE_FRACTION = 0.90" in code
    assert "monthly_coverage_audit_" in code, "the audit must be exported"
    assert "calendar_lag(panel, time_varying_cols" in code, (
        "covariate lags must be calendar-aware after the coverage filter")


# =====================================================================
# 3. The zero baseline
# =====================================================================
def test_cover_baselines_include_zero_persistence_and_climatology(ns):
    panel = ns["calendar_lag"](_panel(n_cells=12, n_months=24), ["wh_cover"])
    train = panel[panel["month"] < pd.Timestamp("2020-07-01")]
    test = panel[panel["month"] >= pd.Timestamp("2020-07-01")]

    preds = ns["cover_baselines"](train, test)
    assert set(ns["BASELINE_MODELS"]) <= set(preds)
    assert np.all(preds["zero"] == 0.0)
    assert len(preds["zero"]) == len(test)
    # Persistence is the observed t-1 cover (0 where the lag is missing).
    assert np.allclose(preds["persistence"],
                       test["wh_cover_lag1"].fillna(0.0).to_numpy())
    # Every baseline is finite and inside the response support.
    for name, p in preds.items():
        assert np.isfinite(p).all(), name
        assert (p >= 0).all() and (p <= 1).all(), name


def test_zero_baseline_scores_are_defined_except_the_correlation(ns):
    panel = _panel(n_cells=10, n_months=12)
    obs = panel["wh_cover"].to_numpy()
    zero = np.zeros(len(panel))
    row = ns["cover_skill_row"](obs, zero, reference=np.full(len(panel), obs.mean()),
                                area_frame=panel.assign(obs=obs, pred=zero))
    assert np.isfinite(row["rmse"]) and row["rmse"] > 0
    assert np.isfinite(row["mae"])
    assert np.isnan(row["spearman"]), "a constant prediction has no rank correlation"
    assert np.isfinite(row["msss_clim"])
    assert np.isfinite(row["area_mae_ha"]) and np.isfinite(row["area_bias_ha"])
    # Predicting zero everywhere under-forecasts area by exactly the observed total.
    assert row["area_bias_ha"] < 0


def test_notebooks_report_the_zero_baseline(notebook):
    code = "\n".join(_code_cells(notebook))
    assert 'BASELINE_MODELS = ("zero", "persistence", "climatology", "climatology_cell")' in code
    # Every notebook scores a zero baseline somewhere it evaluates predictive skill.
    assert ('cover_baselines(' in code) or ('model = "zero"' in code)


# =====================================================================
# 4. The climatology anomaly invariant
# =====================================================================
def _oof_with_baselines(ns, n_cells=25, n_months=30, seed=3):
    """Concatenated one-step OOF rows for a model and every baseline."""
    panel = ns["calendar_lag"](_panel(n_cells=n_cells, n_months=n_months, seed=seed),
                               ["wh_cover"])
    split = panel["month"].sort_values().unique()[n_months // 2]
    train = panel[panel["month"] < split]
    test = panel[panel["month"] >= split].copy()
    preds = ns["cover_baselines"](train, test)
    clim_cell, _levels = ns["fold_cell_month_climatology"](train, test)

    rng = np.random.default_rng(seed)
    preds["model"] = np.clip(test["wh_cover"].to_numpy() * 0.6
                             + rng.normal(0, 0.01, len(test)), 0, 1)
    frames = []
    for name, p in preds.items():
        frames.append(pd.DataFrame({
            "grid_id": test["grid_id"].to_numpy(),
            "month": test["month"].to_numpy(),
            "model": name,
            "obs": test["wh_cover"].to_numpy(),
            "pred": np.asarray(p, float),
            "clim_cell_month": clim_cell,
        }))
    return pd.concat(frames, ignore_index=True)


def test_climatology_baseline_residual_is_identically_zero(ns):
    oof = _oof_with_baselines(ns)
    sub = oof[oof["model"] == "climatology_cell"]
    assert np.allclose(sub["pred"], sub["clim_cell_month"])


def test_climatology_anomaly_correlation_is_undefined_not_zero_point_eight_five(ns):
    """The invariant the old cross-sectional metric broke."""
    oof = _oof_with_baselines(ns)
    result = ns["temporal_residual_corr_by_cell"](oof[oof["model"] == "climatology_cell"])
    assert np.isnan(result["temporal_resid_corr"]), (
        "a zero predicted residual admits no rank correlation; it must be N/A")
    assert result["n_cells_scored"] == 0
    assert ns["assert_climatology_residual_invariant"](oof) is True


def test_the_invariant_assertion_fires_when_the_baseline_drifts(ns):
    oof = _oof_with_baselines(ns)
    tampered = oof.copy()
    mask = tampered["model"] == "climatology_cell"
    tampered.loc[mask, "pred"] = tampered.loc[mask, "pred"] + 0.01
    with pytest.raises(AssertionError, match="not\\s+identically zero"):
        ns["assert_climatology_residual_invariant"](tampered)


def test_a_skilful_model_still_gets_a_defined_temporal_residual_correlation(ns):
    oof = _oof_with_baselines(ns)
    result = ns["temporal_residual_corr_by_cell"](oof[oof["model"] == "model"])
    assert result["n_cells_scored"] > 0
    assert np.isfinite(result["temporal_resid_corr"])
    assert result["temporal_resid_corr"] > 0.3, "a near-perfect model must score well"


def test_temporal_residual_metric_uses_many_months_not_one(ns):
    """A single target month cannot produce a within-cell temporal correlation."""
    oof = _oof_with_baselines(ns)
    one_month = oof[(oof["model"] == "model")
                    & (oof["month"] == oof["month"].min())]
    result = ns["temporal_residual_corr_by_cell"](one_month)
    assert np.isnan(result["temporal_resid_corr"])
    assert result["n_cells_too_few_months"] == result["n_cells_total"]


def test_climatology_baseline_falls_back_hierarchically(ns):
    """cell x calendar-month -> cell -> calendar-month -> global, documented and used."""
    panel = _panel(n_cells=6, n_months=24)
    train = panel[panel["month"] < pd.Timestamp("2020-01-01")]
    target = panel[panel["month"] >= pd.Timestamp("2020-01-01")].copy()
    # An unseen cell can only fall back to the calendar-month or global level.
    unseen = target.iloc[:3].copy()
    unseen["grid_id"] = 999
    target = pd.concat([target, unseen], ignore_index=True)

    values, levels = ns["fold_cell_month_climatology"](train, target)
    assert np.isfinite(values).all()
    assert set(np.unique(levels)) <= {"cell_month", "cell", "month", "global"}
    assert (levels[target["grid_id"].to_numpy() == 999] == "month").all()


def test_predictive_ml_notebook_replaced_the_old_metric():
    nb = REPO / "winam_wh_spatial_panel_predictive_ml.ipynb"
    code = "\n".join(_code_cells(nb))
    assert "temporal_residual_corr_by_cell" in code
    assert "assert_climatology_residual_invariant" in code
    # The faulty metric must not be computed anywhere any more.
    assert "spearman_anom=" not in code
    assert "spearman_anom = " not in code
    assert 'r["spearman_anom"]' not in code


# =====================================================================
# 5. Constant-correlation handling
# =====================================================================
def test_safe_spearman_is_nan_for_constant_inputs_without_warning(ns):
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any warning becomes a failure
        assert np.isnan(ns["safe_spearman"](y, np.zeros(50)))
        assert np.isnan(ns["safe_spearman"](np.ones(50), y))
        assert np.isnan(ns["safe_spearman"](np.zeros(50), np.zeros(50)))
        assert np.isnan(ns["safe_spearman"]([1.0, 2.0], [1.0, 2.0]))   # too few points
        assert ns["safe_spearman"](y, y) == pytest.approx(1.0)
        assert ns["safe_spearman"](y, -y) == pytest.approx(-1.0)


def test_safe_spearman_ignores_non_finite_pairs(ns):
    a = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    b = np.array([2.0, 4.0, 6.0, 8.0, 1.0])
    assert ns["safe_spearman"](a, b) == pytest.approx(1.0)


def test_undefined_metrics_render_as_not_available(ns):
    assert ns["fmt_metric"](float("nan")) == "N/A"
    assert ns["fmt_metric"](None) == "N/A"
    assert ns["fmt_metric"](0.5) == "0.5000"
    assert ns["fmt_metric"](0.5, "{:+.2f}") == "+0.50"


def test_notebooks_do_not_call_scipy_spearman_on_possibly_constant_predictions(notebook):
    code = "\n".join(_code_cells(notebook))
    assert "safe_spearman" in code
    assert "def safe_spearman" in code


# =====================================================================
# 6. Repeatable GeoPackage layer writes
# =====================================================================
@pytest.fixture
def gpd_module():
    return pytest.importorskip("geopandas", reason="geopandas is not installed")


def _points(gpd_module, n, value):
    from shapely.geometry import Point
    return gpd_module.GeoDataFrame(
        {"value": np.full(n, value, dtype=float)},
        geometry=[Point(float(i), float(i)) for i in range(n)],
        crs="EPSG:4326")


def test_write_gpkg_layer_is_idempotent(ns, gpd_module, tmp_path):
    path = tmp_path / "out.gpkg"
    ns["write_gpkg_layer"](_points(gpd_module, 3, 1.0), path, "grid_summary")
    # Re-running the export cell must not raise "Layer already exists".
    ns["write_gpkg_layer"](_points(gpd_module, 5, 2.0), path, "grid_summary")
    ns["write_gpkg_layer"](_points(gpd_module, 5, 2.0), path, "grid_summary")

    back = gpd_module.read_file(path, layer="grid_summary")
    assert len(back) == 5
    assert np.allclose(back["value"], 2.0)
    assert ns["gpkg_layers"](path) == ["grid_summary"]


def test_write_gpkg_layer_preserves_unrelated_layers(ns, gpd_module, tmp_path):
    path = tmp_path / "out.gpkg"
    ns["write_gpkg_layer"](_points(gpd_module, 4, 9.0), path, "keep_me")
    ns["write_gpkg_layer"](_points(gpd_module, 3, 1.0), path, "grid_summary")
    ns["write_gpkg_layer"](_points(gpd_module, 6, 2.0), path, "grid_summary")

    assert set(ns["gpkg_layers"](path)) == {"keep_me", "grid_summary"}
    kept = gpd_module.read_file(path, layer="keep_me")
    assert len(kept) == 4 and np.allclose(kept["value"], 9.0)
    assert len(gpd_module.read_file(path, layer="grid_summary")) == 6


def test_notebooks_route_gpkg_writes_through_the_helper(notebook):
    code = "\n".join(_code_cells(notebook))
    assert "def write_gpkg_layer(" in code
    assert 'write_gpkg_layer(grid, grid_path_gpkg, "grid")' in code
    assert 'write_gpkg_layer(grid_summary, grid_summary_gpkg, "grid_summary")' in code
    assert 'driver="GPKG", mode="w"' not in code
    # No raw to_file GeoPackage write may remain outside the helper.
    outside = [line for line in code.splitlines()
               if ".to_file(" in line and "GPKG" in line
               and "gdf.to_file" not in line and "table.to_file" not in line]
    assert not outside, f"un-helpered GeoPackage writes remain: {outside}"
