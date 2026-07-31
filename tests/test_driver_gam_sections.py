"""Behavioural tests for the driver-GAM notebook's Python sections.

The notebook cells for §7c (sensor composition), §9c (Earth Engine column audit),
§12 (predictor sets) and the §13 frame builder are pure pandas, so they are executed
here against a synthetic panel — no Drive, no Earth Engine, no R, no rpy2. Together
with ``test_gam_guardrails.py`` (the helper API) and ``test_driver_gam_notebook.py``
(the structural contract) this pins the behaviour that the corrections introduced:

* an S2-only run is reported as S2-only and loses its constant sensor indicator;
* anonymous ``mean`` / ``mean_x`` / ``mean_y`` columns are renamed on a physical
  signature or dropped, and degenerate covariates never reach the model;
* the forcing set is exogenous, and the lagged-response terms only ever appear in
  the predictive specification;
* classifier confidence is used exactly once;
* the fold design validates far more than the final four months.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "winam_wh_spatial_panel_driver_gam.ipynb"


def _cell(marker):
    for c in json.loads(NOTEBOOK.read_text())["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise AssertionError(f"no code cell contains {marker!r}")


def _base_namespace(tmp_path):
    """Guardrail helpers + the methodological-corrections config block."""
    ns = {"pd": pd, "np": np, "Path": Path, "OUTPUT_DIR": tmp_path,
          "CELL_SIZE_M": 500, "TEST_START": "2019-01-01", "TEST_END": "2023-12-31",
          "ACTIVE_CLASSIFIER_VERSION": "route_b_test_v1",
          "TRAIN_FRACTION": 0.75, "RANDOM_STATE": 42,
          "display": lambda *a, **k: None}
    exec(compile(_cell("4d. Methodological guardrails"), "<guardrails>", "exec"), ns)
    cfg = _cell("# Methodological corrections to the driver GAM")
    cfg = cfg[cfg.index("# Methodological corrections to the driver GAM"):]
    exec(compile(cfg, "<config>", "exec"), ns)
    return ns


def _synthetic_panel(n_cells=120, n_months=60, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    gid = np.repeat(np.arange(n_cells), n_months)
    mo = pd.DatetimeIndex(np.tile(months.values, n_cells))
    n = len(gid)
    panel = pd.DataFrame({
        "grid_id": gid, "month": mo, "sensor": "S2", "sensor_is_s1": 0,
        "x_km": np.repeat(rng.uniform(0, 90, n_cells), n_months),
        "y_km": np.repeat(rng.uniform(0, 55, n_cells), n_months),
        "month_num": mo.month, "time_index": np.tile(np.arange(n_months), n_cells),
        "valid_pixels": 368.0, "valid_area_m2": 36800.0, "valid_fraction": 0.9,
        "wh_pixels": rng.integers(0, 40, n),
        "rain_chirps_30d_mm_lag1": rng.gamma(2, 40, n),
        "air_temp_c": rng.normal(25, 2, n),
        "wind_onshore_ms": rng.normal(0, 2, n),
        "wave_exposure_idx": rng.gamma(2, 3, n),
        "effective_depth_m": rng.uniform(0.5, 40, n),
        "dist_majriver_m": np.repeat(rng.uniform(0, 3e4, n_cells), n_months),
        "dist_shore_m": np.repeat(rng.uniform(0, 1e4, n_cells), n_months),
        "frac_cropland": np.repeat(rng.uniform(0, 1, n_cells), n_months),
        "turb_ndti_s2": rng.normal(0, 0.2, n),
        "chl_mci_s3": rng.gamma(2, 10, n),
    })
    cover = np.clip(rng.beta(0.3, 8, n), 0, 1)
    panel["wh_cover_hard"] = cover
    panel["wh_cover_soft"] = np.where(rng.random(n) < 0.02, np.nan, cover * 0.9)
    panel["wh_cover"] = panel["wh_cover_soft"].fillna(panel["wh_cover_hard"])
    panel["wh_present"] = (panel["wh_cover"] >= 0.02).astype(int)
    # confidence is defined ONLY where WH pixels exist -- the asymmetry that made the
    # old `fillna(1.0)` treat every absence as a certain observation.
    panel["wh_conf_mean"] = np.where(panel["wh_cover"] > 0, rng.uniform(0.3, 1, n), np.nan)
    panel = panel.sort_values(["grid_id", "month"]).reset_index(drop=True)
    panel["wh_cover_lag1"] = panel.groupby("grid_id")["wh_cover"].shift(1)
    panel["wh_cover_neigh_lag1"] = panel.groupby("grid_id")["wh_cover"].shift(1) * 0.8
    n_missing = max(1, len(panel) // 15)
    panel.loc[panel.sample(n_missing, random_state=1).index, "air_temp_c"] = np.nan
    return panel


# ---------------------------------------------------------------------
# §7c — sensor composition
# ---------------------------------------------------------------------
@pytest.fixture
def sensor_ns(tmp_path):
    ns = _base_namespace(tmp_path)
    ns.update({"panel": _synthetic_panel(n_cells=40, n_months=12),
               "ENABLE_S1_GAPFILL": True, "SENSOR_FEATURE_ENABLED": True})
    exec(compile(_cell("7c. What sensor(s) does this run ACTUALLY use"), "<s7c>", "exec"), ns)
    return ns


def test_s2_only_run_is_labelled_and_loses_the_sensor_indicator(sensor_ns):
    assert sensor_ns["PANEL_IS_S2_ONLY"] is True
    assert sensor_ns["PANEL_SENSOR_MODE"] == "S2_only"
    assert sensor_ns["S1_GAPFILL_REQUESTED"] is True
    assert sensor_ns["S1_GAPFILL_APPLIED"] is False
    assert "sensor_is_s1" not in sensor_ns["panel"].columns
    assert sensor_ns["SENSOR_FEATURE_ENABLED"] is False


def test_s2_only_run_records_the_unmitigated_mnar_limitation(sensor_ns, tmp_path):
    prov = sensor_ns["PANEL_SENSOR_PROVENANCE"]
    assert prov["s1_gapfill_applied"] is False
    assert "unmitigated" in prov["cloud_mnar_limitation"]
    assert "MNAR" in prov["cloud_mnar_limitation"]
    written = json.loads((tmp_path / "panel_sensor_provenance_500m_S2_only.json").read_text())
    assert written["panel_sensor_mode"] == "S2_only"


def test_multi_sensor_run_keeps_the_indicator(tmp_path):
    ns = _base_namespace(tmp_path)
    panel = _synthetic_panel(n_cells=40, n_months=12)
    panel.loc[panel.index[:100], "sensor"] = "S1"
    panel.loc[panel.index[:100], "sensor_is_s1"] = 1
    ns.update({"panel": panel, "ENABLE_S1_GAPFILL": True, "SENSOR_FEATURE_ENABLED": True})
    exec(compile(_cell("7c. What sensor(s) does this run ACTUALLY use"), "<s7c>", "exec"), ns)
    assert ns["PANEL_IS_S2_ONLY"] is False
    assert ns["S1_GAPFILL_APPLIED"] is True
    assert "sensor_is_s1" in ns["panel"].columns


# ---------------------------------------------------------------------
# §9c — Earth Engine column audit
# ---------------------------------------------------------------------
@pytest.fixture
def audit_ns(tmp_path):
    ns = _base_namespace(tmp_path)
    rng = np.random.default_rng(0)
    n_cells, n_months = 200, 40
    months = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    n = n_cells * n_months
    panel = pd.DataFrame({
        "grid_id": np.repeat(np.arange(n_cells), n_months),
        "month": pd.DatetimeIndex(np.tile(months.values, n_cells)),
        "wh_cover": np.clip(rng.beta(0.3, 8, n), 0, 1),
        "mean_x": rng.uniform(23, 30, n),          # MODIS LST over water, deg C
        "mean_y": rng.uniform(1e4, 9e4, n),        # matches no signature
        "mean": np.repeat(rng.uniform(0, 1, n_cells), n_months),   # JRC water fraction
        "air_temp_c": rng.normal(25, 2, n),
        "constant_col": 1.0,
        "near_constant_col": np.where(np.arange(n) < 5, 1.0, 0.0),
        "all_missing_col": np.nan,
    })
    ns.update({"panel": panel, "MONTHLY_COVARIATE_COLS": [],
               "CELLMONTH_COVARIATE_COLS": ["mean_x", "mean_y", "air_temp_c",
                                            "constant_col", "near_constant_col",
                                            "all_missing_col"],
               "STATIC_COVARIATE_COLS": ["mean"]})
    exec(compile(_cell("9c. Earth Engine covariate audit"), "<s9c>", "exec"), ns)
    return ns


def test_ambiguous_columns_are_renamed_on_a_signature(audit_ns):
    assert audit_ns["EE_COLUMN_AUDIT_RENAMES"] == {"mean_x": "water_temp_c",
                                                   "mean": "gsw_water_fraction"}
    assert "water_temp_c" in audit_ns["panel"].columns
    assert "gsw_water_fraction" in audit_ns["panel"].columns


def test_unidentified_column_is_dropped_not_modelled(audit_ns):
    assert "mean_y" not in audit_ns["panel"].columns
    assert "mean_y" not in audit_ns["CELLMONTH_COVARIATE_COLS"]
    dropped = audit_ns["EE_COLUMN_AUDIT_DROPPED"]
    assert "mean_y" in set(dropped["column"])


def test_degenerate_covariates_leave_the_registries(audit_ns):
    reg = (audit_ns["CELLMONTH_COVARIATE_COLS"] + audit_ns["STATIC_COVARIATE_COLS"]
           + audit_ns["MONTHLY_COVARIATE_COLS"])
    assert reg == ["water_temp_c", "air_temp_c", "gsw_water_fraction"]
    for gone in ["constant_col", "near_constant_col", "all_missing_col"]:
        assert gone not in reg


# ---------------------------------------------------------------------
# §12 + §13 frame builder
# ---------------------------------------------------------------------
@pytest.fixture
def model_ns(tmp_path):
    ns = _base_namespace(tmp_path)
    ns.update({
        "panel": _synthetic_panel(),
        "PANEL_IS_S2_ONLY": True,
        "USE_PROBABILITY_RESPONSE": True,
        "MONTHLY_COVARIATE_COLS": [],
        "CELLMONTH_COVARIATE_COLS": ["turb_ndti_s2", "chl_mci_s3", "air_temp_c"],
        "STATIC_COVARIATE_COLS": ["dist_majriver_m", "dist_shore_m", "frac_cropland"],
    })
    ns["panel"] = ns["panel"].drop(columns=["sensor_is_s1", "sensor"])
    exec(compile(_cell("12. Model dataset: a parsimonious FORCING set"), "<s12>", "exec"), ns)
    exec(compile(_cell("# --- Build the GAM modelling frame ---"), "<s13>", "exec"), ns)
    return ns


def test_forcing_set_is_parsimonious_and_exogenous(model_ns):
    forcing = model_ns["forcing_cols"]
    assert 1 <= len(forcing) <= 12
    assert not any(model_ns["is_response_derived"](c) for c in forcing)
    assert not any(model_ns["is_endogenous_optical"](c) for c in forcing)
    # each term carries a stated mechanism and expected sign
    roles = model_ns["predictor_roles_table"].set_index("term")
    for c in forcing:
        assert roles.loc[c, "mechanism"].strip()
        assert roles.loc[c, "expected_sign"] in {"+", "-", "?"}


def test_optical_proxies_are_excluded_from_forcing_but_still_tracked(model_ns):
    assert set(model_ns["proxy_cols"]) == {"turb_ndti_s2", "chl_mci_s3"}
    assert not set(model_ns["proxy_cols"]) & set(model_ns["forcing_cols"])


def test_lagged_response_terms_are_predictive_only(model_ns):
    assert model_ns["predictive_lag_cols"] == ["wh_cover_lag1", "wh_cover_neigh_lag1"]
    for term in model_ns["predictive_lag_cols"]:
        assert term not in model_ns["r_formula_forcing"]
        assert term not in model_ns["r_formula_cv"]
        assert term in model_ns["r_formula_predictive"]


def test_forcing_formulas_pass_the_exogeneity_gate(model_ns):
    known = list(model_ns["gam_df"].columns)
    for key in ["r_formula_forcing", "r_formula_cv"]:
        assert model_ns["assert_forcing_formula_exogenous"](
            model_ns[key], known, label=key, allow=("grid_id",))


def test_confidence_is_used_exactly_once(model_ns):
    usage = model_ns["CONFIDENCE_USAGE"]
    assert usage["mode"] == "soft_response"
    assert usage["weight_col"] is None
    assert model_ns["RESPONSE_IS_SOFT"] is True
    assert model_ns["WEIGHTS_ARE_CONFIDENCE"] is False
    assert (model_ns["gam_df"]["obs_weight"] == 1.0).all()


def test_likelihood_weights_are_refused_without_absence_confidence(tmp_path):
    ns = _base_namespace(tmp_path)
    panel = _synthetic_panel()
    ns.update({"panel": panel, "PANEL_IS_S2_ONLY": True, "USE_PROBABILITY_RESPONSE": True,
               "MONTHLY_COVARIATE_COLS": [],
               "CELLMONTH_COVARIATE_COLS": ["turb_ndti_s2", "chl_mci_s3", "air_temp_c"],
               "STATIC_COVARIATE_COLS": ["dist_majriver_m", "dist_shore_m", "frac_cropland"],
               "GAM_CONFIDENCE_MODE": "likelihood_weights"})
    ns["panel"] = ns["panel"].drop(columns=["sensor_is_s1", "sensor"])
    exec(compile(_cell("12. Model dataset: a parsimonious FORCING set"), "<s12>", "exec"), ns)
    with pytest.raises(AssertionError, match="ALL valid pixels"):
        exec(compile(_cell("# --- Build the GAM modelling frame ---"), "<s13>", "exec"), ns)


def test_rows_are_not_dropped_for_missing_covariates(model_ns):
    """Missing drivers are imputed inside folds, so they must not shrink the panel."""
    assert len(model_ns["gam_df"]) == len(model_ns["panel"])
    # ... and the raw NaNs survive into the frame, to be imputed per fold
    assert model_ns["gam_df"]["air_temp_c"].isna().sum() > 0


def test_first_observation_rows_only_leave_the_predictive_spec(model_ns):
    md = model_ns["model_df"]
    n_first = int((~md["has_response_lag"]).sum())
    assert n_first == md["grid_id"].nunique()      # one per cell
    assert len(md) > n_first


def test_shared_folds_cover_the_record(model_ns):
    meta = model_ns["CV_TEMPORAL_META"]
    assert meta["months_validated"] >= 12
    assert meta["fraction_validated"] > 0.1
    design = model_ns["CV_TEMPORAL_DESIGN"]
    assert design["train_months"].min() >= meta["min_train_months"]
    gam_df = model_ns["gam_df"]
    assert sorted(gam_df["fold_sp"].unique()) == model_ns["CV_SPATIAL_FOLDS"]
    assert sorted(f for f in gam_df["fold_tm"].unique() if f > 0) == model_ns["CV_TEMPORAL_FOLDS"]


def test_spatial_folds_keep_blocks_whole(model_ns):
    gam_df = model_ns["gam_df"]
    block = (np.floor(gam_df["x_km"] / model_ns["GAM_SPATIAL_BLOCK_KM"]).astype(int).astype(str)
             + "_" + np.floor(gam_df["y_km"] / model_ns["GAM_SPATIAL_BLOCK_KM"]).astype(int).astype(str))
    assert pd.DataFrame({"b": block, "f": gam_df["fold_sp"]}).groupby("b")["f"].nunique().max() == 1


def test_no_sensor_term_survives_into_the_frame(model_ns):
    assert "sensor_is_s1" not in model_ns["gam_df"].columns
    assert "sensor_is_s1" not in model_ns["control_cols"]
    for key in ["r_formula_forcing", "r_formula_predictive", "r_formula_cv"]:
        assert "sensor_is_s1" not in model_ns[key]


def test_downstream_builders_execute_on_the_shared_frame(model_ns):
    """§13c, §13e and §13f must build from the §13 frame without extra plumbing."""
    ns = model_ns
    exec(compile(_cell("13c. Causal contrast: does the spatial term confound the drivers?"),
                 "<s13c>", "exec"), ns)
    assert ns["r_formula_causal"].startswith("wh_cover ~")
    exec(compile(_cell("13e. Model-family comparison -- same observations"), "<s13e>", "exec"), ns)
    assert ns["mfc_run_r"] is True
    assert len(ns["mfc_df"]) == len(ns["gam_df"])          # identical rows by construction
    assert "ordbeta" in ns["mfc_families_str"]
    assert "hurdle" in ns["mfc_families_str"]
    exec(compile(_cell("13f. Robustness: does any conclusion depend on a preprocessing choice?"),
                 "<s13f>", "exec"), ns)
    fams = set(ns["robust_variants"]["family"])
    assert {"response", "measurement_error", "spatial_support", "coverage_threshold"} <= fams
