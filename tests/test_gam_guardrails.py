"""Tests for the driver-GAM methodological guardrails.

The guardrails live in ``winam_wh_spatial_panel_driver_gam.ipynb`` (section 4d),
so they are extracted from the notebook JSON and executed here rather than being
copied -- the tests therefore track the real notebook instead of a snapshot that
can drift away from it. Pure numpy/pandas: no Drive, no Earth Engine, no R.

What is pinned:

1. a forcing formula containing any ``wh_*`` lag, neighbour term or endogenous
   optical proxy is rejected;
2. constant, near-constant, mostly-missing and reducer-named ("mean", "mean_x")
   covariates never reach the model, and ambiguous Earth Engine columns are only
   renamed when a physical signature identifies them uniquely;
3. classifier confidence is used exactly once, and absence rows are never handed
   full confidence just because WH-pixel confidence is missing;
4. the shared fold design covers a meaningful span of the record and every
   requested fold has to come back with metrics;
5. predictions are checked against the response bounds;
6. saved summaries have to declare whether they are in-sample fit, blocked
   validation or environmental association.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "winam_wh_spatial_panel_driver_gam.ipynb"
CELL_MARKER = "4d. Methodological guardrails"


def _load_guardrails():
    """Exec the guardrail cell in an isolated namespace and return it."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    sources = [
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code" and CELL_MARKER in "".join(c["source"])
    ]
    assert len(sources) == 1, f"expected exactly one guardrail cell, found {len(sources)}"
    ns: dict = {}
    exec(compile(sources[0], "<guardrails>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def g():
    return _load_guardrails()


# ---------------------------------------------------------------------
# 1. Exogeneity of the forcing formula
# ---------------------------------------------------------------------
FORCING_COLS = [
    "rain_chirps_30d_mm_lag1", "air_temp_c", "wind_onshore_ms",
    "wave_exposure_idx", "effective_depth_m", "dist_majriver_m",
    "dist_shore_m", "frac_cropland",
]
ENDOGENOUS_COLS = ["turb_ndti_s2", "red_reflectance_s2", "chl_mci_s3", "chl_mph_s3"]
RESPONSE_COLS = ["wh_cover_lag1", "wh_present_lag1", "wh_cover_neigh_lag1",
                 "wh_present_neigh_lag1", "wh_conf_mean"]
KNOWN = FORCING_COLS + ENDOGENOUS_COLS + RESPONSE_COLS + [
    "month_num", "x_km", "y_km", "grid_id", "time_index"]


def _formula(cols):
    smooths = " + ".join(f"s({c}, k=10, bs='tp')" for c in cols)
    return (f"wh_cover ~ {smooths} + s(month_num, bs='cc', k=8) "
            f"+ te(x_km, y_km, k=c(15,15)) + time_index")


def test_roles(g):
    for c in FORCING_COLS:
        assert g["classify_predictor"](c) == "forcing", c
    for c in ENDOGENOUS_COLS:
        assert g["classify_predictor"](c) == "endogenous_optical", c
    for c in RESPONSE_COLS:
        assert g["classify_predictor"](c) == "response_derived", c
    assert g["classify_predictor"]("month_num") == "structural"


def test_clean_forcing_formula_passes(g):
    assert g["assert_forcing_formula_exogenous"](_formula(FORCING_COLS), KNOWN)


@pytest.mark.parametrize("offender", [
    "wh_cover_lag1", "wh_present_lag1", "wh_cover_neigh_lag1",
    "turb_ndti_s2", "chl_mci_s3", "red_reflectance_s2",
])
def test_forcing_formula_rejects_endogenous_terms(g, offender):
    with pytest.raises(AssertionError):
        g["assert_forcing_formula_exogenous"](_formula(FORCING_COLS + [offender]), KNOWN)


def test_forcing_formula_rejects_response_term_not_in_known_columns(g):
    """A wh_* term still trips the check even if it is absent from the column list."""
    with pytest.raises(AssertionError):
        g["assert_forcing_formula_exogenous"](_formula(FORCING_COLS + ["wh_area_ha_lag1"]),
                                              KNOWN)


def test_mrf_and_re_structure_terms_do_not_trip_the_check(g):
    f = (_formula(FORCING_COLS)
         + " + s(grid_id, bs='mrf', xt=xt_mrf, k=200) + s(grid_id, bs='re')")
    assert g["assert_forcing_formula_exogenous"](f, KNOWN)


def test_split_forcing_and_response_terms(g):
    forcing, response, optical = g["split_forcing_and_response_terms"](
        FORCING_COLS + RESPONSE_COLS + ENDOGENOUS_COLS)
    assert forcing == FORCING_COLS
    assert set(response) == set(RESPONSE_COLS)
    assert set(optical) == set(ENDOGENOUS_COLS)


# ---------------------------------------------------------------------
# 2. Covariate identification and screening
# ---------------------------------------------------------------------
def test_ambiguous_column_detection(g):
    for name in ["mean", "mean_x", "mean_y", "median", "sum_x"]:
        assert g["is_ambiguous_column"](name), name
    for name in ["air_temp_c", "mean_depth_m", "rain_chirps_mm"]:
        assert not g["is_ambiguous_column"](name), name


def test_identify_water_temperature_column(g):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "mean_x": rng.uniform(23.0, 30.0, 500),      # MODIS LST over water, deg C
        "mean_y": rng.uniform(1e4, 9e4, 500),        # not covered by any signature
    })
    renames, report = g["identify_ambiguous_columns"](df, ["mean_x", "mean_y"], "cellmonth")
    assert renames == {"mean_x": "water_temp_c"}
    assert set(report["decision"]) == {"renamed", "unidentified"}
    assert report.loc[report["column"] == "mean_y", "decision"].iloc[0] == "unidentified"


def test_identify_gsw_water_fraction_column(g):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"mean": rng.uniform(0.0, 1.0, 400)})
    renames, _ = g["identify_ambiguous_columns"](df, ["mean"], "static")
    assert renames == {"mean": "gsw_water_fraction"}


def test_identification_scope_is_respected(g):
    """A 0-1 column in the cell/month table is NOT silently called water fraction."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"mean": rng.uniform(0.0, 1.0, 400)})
    renames, report = g["identify_ambiguous_columns"](df, ["mean"], "cellmonth")
    assert renames == {}
    assert report["decision"].iloc[0] == "unidentified"


def test_identification_does_not_overwrite_an_existing_name(g):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"mean": rng.uniform(23.0, 30.0, 300),
                       "water_temp_c": rng.uniform(23.0, 30.0, 300)})
    renames, _ = g["identify_ambiguous_columns"](df, ["mean"], "cellmonth")
    assert renames == {}


def test_screen_covariates_drops_the_right_things(g):
    rng = np.random.default_rng(4)
    n = 500
    df = pd.DataFrame({
        "good": rng.normal(size=n),
        "constant": np.ones(n),
        "near_constant": np.concatenate([np.zeros(n - 2), [1.0, 2.0]]),
        "all_missing": np.full(n, np.nan),
        "mostly_missing": np.where(np.arange(n) < 5, rng.normal(size=n), np.nan),
        "mean_x": rng.uniform(20, 30, n),
    })
    keep, dropped = g["screen_covariates"](df, list(df.columns))
    assert keep == ["good"]
    reasons = dict(zip(dropped["column"], dropped["reason"]))
    assert reasons["constant"] == "constant_or_too_few_levels"
    assert reasons["near_constant"] == "near_constant"
    assert reasons["all_missing"] == "all_missing"
    assert reasons["mostly_missing"] == "mostly_missing"
    assert reasons["mean_x"] == "unidentified_reducer_name"


def test_screen_covariates_reports_absent_columns(g):
    df = pd.DataFrame({"a": np.arange(10.0)})
    keep, dropped = g["screen_covariates"](df, ["a", "nope"])
    assert keep == ["a"]
    assert dropped.loc[dropped["column"] == "nope", "reason"].iloc[0] == "absent_from_panel"


def test_assert_all_identified(g):
    assert g["assert_all_identified"](["air_temp_c", "rain_chirps_mm"])
    with pytest.raises(AssertionError):
        g["assert_all_identified"](["air_temp_c", "mean_y"])


def test_sensor_flag_is_dropped_when_constant(g):
    """An S2-only panel makes sensor_is_s1 constant, so it never reaches a model."""
    df = pd.DataFrame({"sensor_is_s1": np.zeros(1000), "air_temp_c": np.arange(1000.0)})
    keep, dropped = g["screen_covariates"](df, ["sensor_is_s1", "air_temp_c"])
    assert keep == ["air_temp_c"]
    assert dropped.loc[dropped["column"] == "sensor_is_s1", "reason"].iloc[0] == \
        "constant_or_too_few_levels"


# ---------------------------------------------------------------------
# 3. Classifier confidence used exactly once
# ---------------------------------------------------------------------
def test_soft_response_uses_unit_weights(g):
    r = g["resolve_confidence_usage"]("soft_response",
                                      {"soft": True, "hard": True, "conf_wh": True})
    assert r["response_col"] == "soft"
    assert r["weight_col"] is None


def test_likelihood_weights_uses_hard_response(g):
    r = g["resolve_confidence_usage"]("likelihood_weights",
                                      {"soft": True, "hard": True, "conf_all": True})
    assert r["response_col"] == "hard"
    assert r["weight_col"] == "conf_all"


def test_likelihood_weights_refuses_wh_only_confidence(g):
    """Absence rows have no WH-pixel confidence -- refuse rather than assign 1.0."""
    with pytest.raises(AssertionError, match="ALL valid pixels"):
        g["resolve_confidence_usage"]("likelihood_weights",
                                      {"soft": True, "hard": True,
                                       "conf_all": False, "conf_wh": True})


def test_likelihood_weights_partial_variant_is_flagged(g):
    r = g["resolve_confidence_usage"]("likelihood_weights",
                                      {"hard": True, "conf_all": False, "conf_wh": True},
                                      allow_partial=True)
    assert r["weight_col"] == "conf_wh_partial"
    assert "never 1.0" in r["note"]


def test_none_mode_and_bad_mode(g):
    r = g["resolve_confidence_usage"]("none", {"hard": True})
    assert r["response_col"] == "hard" and r["weight_col"] is None
    with pytest.raises(ValueError):
        g["resolve_confidence_usage"]("both", {"hard": True})


def test_build_confidence_weights_does_not_default_missing_to_one(g):
    w = g["build_confidence_weights"](pd.Series([0.5, np.nan, 0.9]))
    assert np.isnan(w.iloc[1])
    w2 = g["build_confidence_weights"](pd.Series([0.01, np.nan]), fill=0.4)
    assert w2.iloc[0] == pytest.approx(0.1)   # clipped to the floor
    assert w2.iloc[1] == pytest.approx(0.4)


# ---------------------------------------------------------------------
# 4. Shared fold design
# ---------------------------------------------------------------------
def test_rolling_origin_covers_more_than_the_final_four_months(g):
    months = pd.date_range("2018-01-01", periods=96, freq="MS")
    design, meta = g["rolling_origin_folds"](months, n_folds=8, horizon_months=3,
                                             min_train_months=24)
    assert meta["n_folds"] == 8
    assert meta["months_validated"] == 24
    assert meta["fraction_validated"] == pytest.approx(0.25)
    # Expanding window: every fold trains only on months before its test block.
    assert design["cut_index"].is_monotonic_increasing
    assert design["train_months"].min() >= 24


def test_rolling_origin_trims_when_the_record_is_short(g):
    months = pd.date_range("2020-01-01", periods=18, freq="MS")
    design, meta = g["rolling_origin_folds"](months, n_folds=8, horizon_months=3,
                                             min_train_months=12)
    assert 0 < meta["n_folds"] < 8
    assert design["train_months"].min() >= 12


def test_assign_temporal_folds_matches_the_design(g):
    months = pd.date_range("2018-01-01", periods=24, freq="MS")
    design, _ = g["rolling_origin_folds"](months, n_folds=3, horizon_months=2,
                                          min_train_months=12)
    ranks = np.arange(24)
    folds = g["assign_temporal_folds"](ranks, design)
    assert set(np.unique(folds)) == {0, 1, 2, 3}
    for row in design.itertuples(index=False):
        block = ranks[folds == row.fold]
        assert block.min() == row.cut_index
        assert block.size == row.test_months


def test_spatial_block_folds_keep_blocks_whole(g):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 100, 4000)
    y = rng.uniform(0, 60, 4000)
    folds, keys = g["spatial_block_folds"](x, y, block_km=10.0, n_folds=5, seed=7)
    assert set(np.unique(folds)) == {1, 2, 3, 4, 5}
    # every cell of a block belongs to exactly one fold
    assert pd.DataFrame({"k": keys, "f": folds}).groupby("k")["f"].nunique().max() == 1


def test_fold_coverage_table(g):
    df = pd.DataFrame({
        "grid_id": np.repeat(np.arange(10), 6),
        "time_rank": np.tile(np.arange(6), 10),
        "fold_sp": np.repeat([1, 2], 30),
        "fold_tm": np.tile([0, 0, 0, 0, 1, 2], 10),
    })
    cov = g["fold_coverage_table"](df)
    assert set(cov["kind"]) == {"spatial", "temporal"}
    tm = cov[cov["kind"] == "temporal"].set_index("fold")
    assert tm.loc[1, "n_train"] == 40      # months 0-3
    assert tm.loc[2, "n_train"] == 50      # months 0-4


def test_assert_folds_scored(g):
    ok = pd.DataFrame({"kind": ["spatial"] * 2 + ["temporal"] * 2,
                       "fold": [1, 2, 1, 2],
                       "spearman": [0.3, 0.2, 0.1, 0.4],
                       "rmse": [0.1, 0.1, 0.1, 0.1]})
    # A clean pass returns the (empty) list of undefined-by-construction metrics.
    assert g["assert_folds_scored"](ok, {"spatial": [1, 2], "temporal": [1, 2]}) == []
    with pytest.raises(AssertionError):
        g["assert_folds_scored"](ok, {"spatial": [1, 2, 3], "temporal": [1, 2]})
    bad = ok.copy()
    bad.loc[bad.index[-1], ["spearman", "rmse"]] = np.nan
    with pytest.raises(AssertionError):
        g["assert_folds_scored"](bad, {"spatial": [1, 2], "temporal": [1, 2]})
    # A fold string/float fold column (rpy2 hands back doubles) still matches.
    astr = ok.copy()
    astr["fold"] = astr["fold"].astype(str)
    assert g["assert_folds_scored"](astr, {"spatial": [1, 2], "temporal": [1, 2]}) == []


def test_assert_folds_scored_excuses_undefined_rank_on_constant_folds(g):
    """An all-zero held-out block has no ranking, so a NaN Spearman is not a failure.

    Its error metrics are still finite and still required, and a NaN Spearman on a
    fold that DOES vary (a constant prediction) is still a failure.
    """
    base = pd.DataFrame({"kind": ["spatial"] * 2 + ["temporal"] * 2,
                         "fold": [1, 2, 1, 2],
                         "n": [500, 500, 500, 500],
                         "n_pos": [60, 60, 0, 40],
                         "spearman": [0.3, 0.2, np.nan, 0.4],
                         "rmse": [0.1, 0.1, 0.1, 0.1]})
    expected = {"spatial": [1, 2], "temporal": [1, 2]}
    assert g["assert_folds_scored"](base, expected) == ["temporal:1:spearman"]

    # Fewer than three held-out rows is the other undefined case.
    tiny = base.copy()
    tiny.loc[0, ["n", "n_pos", "spearman"]] = [2, 1, np.nan]
    assert g["assert_folds_scored"](tiny, expected) == [
        "spatial:1:spearman", "temporal:1:spearman"]

    # A NaN Spearman where the block varies is a real degeneracy.
    varying = base.copy()
    varying.loc[1, "spearman"] = np.nan
    with pytest.raises(AssertionError, match="spatial:2:spearman"):
        g["assert_folds_scored"](varying, expected)

    # The excuse is rank-only: a NaN error metric on the same fold still raises.
    no_rmse = base.copy()
    no_rmse.loc[2, "rmse"] = np.nan
    with pytest.raises(AssertionError, match="temporal:1:rmse"):
        g["assert_folds_scored"](no_rmse, expected)

    # An undefined rank metric never hides a fold that came back empty.
    with pytest.raises(AssertionError, match="spatial:3"):
        g["assert_folds_scored"](base, {"spatial": [1, 2, 3], "temporal": [1, 2]})


# The three reasons a rank correlation can be NaN, and what each one means for the
# fold: only the last is a model failure.
EXPECTED_2 = {"spatial": [1], "temporal": [1]}
HEALTHY = {"kind": "spatial", "fold": 1, "n": 400, "n_pos": 120, "n_high": 20,
           "spearman": 0.3, "rmse": 0.10, "pred_min": 0.0, "pred_max": 0.40}


def _nan_fold(**over):
    """A temporal fold 1 whose Spearman came back NaN but whose rmse did not."""
    row = {"kind": "temporal", "fold": 1, "n": 400, "n_pos": 120, "n_high": 20,
           "spearman": np.nan, "rmse": 0.11, "pred_min": 0.01, "pred_max": 0.30}
    row.update(over)
    return pd.DataFrame([HEALTHY, row])


def test_assert_folds_scored_reads_the_recorded_na_reason(g):
    """The sweep records WHY each Spearman is NaN; the guardrail must act on it.

    Row counts cannot tell a block that admits no ranking from a fit that returned
    one value for every held-out row, so the reason is recorded beside the value
    and read back here rather than re-derived.
    """
    for why in ("constant_obs", "too_few"):
        assert g["assert_folds_scored"](
            _nan_fold(spearman_na_reason=why), EXPECTED_2) == ["temporal:1:spearman"]

    # A constant prediction against a block that DOES rank is a real degeneracy: the
    # fold has no skill to report, only an rmse that reads like one.
    with pytest.raises(AssertionError, match="temporal:1:spearman .constant_pred.") as e:
        g["assert_folds_scored"](_nan_fold(spearman_na_reason="constant_pred"),
                                 EXPECTED_2)
    assert "ONE value for every held-out row" in str(e.value)

    # A recorded reason never excuses a non-finite ERROR metric on the same fold.
    with pytest.raises(AssertionError, match="temporal:1:rmse"):
        g["assert_folds_scored"](
            _nan_fold(spearman_na_reason="constant_obs", rmse=np.nan), EXPECTED_2)


def test_assert_folds_scored_diagnoses_results_written_without_a_reason(g):
    """Older cv_results -- and folds resumed from checkpoints written by them --
    carry no reason column, so it is reconstructed from what they do carry.

    A Spearman goes NaN for exactly three reasons, tested in that order, so ruling
    out the first two identifies the third for the whole-block correlation. Guessing
    instead (the row counts alone) failed folds whose held-out block was simply
    constant, which is not a failure.
    """
    afs = g["assert_folds_scored"]

    # Predictions vary and there are rows to rank, so the block itself must be the
    # constant one -- undefined by construction, not a failed fold.
    assert afs(_nan_fold(), EXPECTED_2) == ["temporal:1:spearman"]

    # One predicted value across the block: a degeneracy, named as such.
    with pytest.raises(AssertionError, match="temporal:1:spearman .constant_pred."):
        afs(_nan_fold(pred_min=0.07, pred_max=0.07), EXPECTED_2)

    # Too few held-out rows outranks both.
    assert afs(_nan_fold(n=2, pred_min=0.07, pred_max=0.07), EXPECTED_2) == [
        "temporal:1:spearman"]

    # With no prediction range recorded either, nothing can be concluded and the
    # fold still raises rather than being quietly excused.
    with pytest.raises(AssertionError, match="temporal:1:spearman"):
        afs(_nan_fold().drop(columns=["pred_min", "pred_max"]), EXPECTED_2)

    # A resumed fold carries the column but no value in it (rpy2 may surface R's
    # character NA as a string); that is "unstated", not a reason.
    for blank in (np.nan, "NA", ""):
        assert afs(_nan_fold(spearman_na_reason=blank), EXPECTED_2) == [
            "temporal:1:spearman"]


def test_assert_folds_scored_uses_each_rank_metrics_own_subset(g):
    """_pos ranks only the rows with cover and _high only those above the threshold,
    so each is undefined on its OWN row count, not on the block's."""
    frame = _nan_fold(spearman=0.2, spearman_high=np.nan, n_high=2)
    frame.loc[0, "spearman_high"] = 0.1
    assert g["assert_folds_scored"](
        frame, EXPECTED_2,
        metric_cols=("spearman", "spearman_high", "rmse")) == ["temporal:1:spearman_high"]

    # A block with plenty of high-cover rows and a varying prediction is not excused.
    frame2 = _nan_fold(spearman=0.2, spearman_high=np.nan, n_high=50)
    frame2.loc[0, "spearman_high"] = 0.1
    with pytest.raises(AssertionError, match="temporal:1:spearman_high"):
        g["assert_folds_scored"](frame2, EXPECTED_2,
                                 metric_cols=("spearman", "spearman_high", "rmse"))


def test_a_prediction_pinned_to_a_response_bound_is_named_as_saturation(g):
    """A constant prediction ON a response bound is a saturated link, not a shrunk fit.

    §13-CV temporal fold 1 of the forcing spec returned 2.22e-16 -- mgcv's clamped
    inverse logit -- for all 5,400 held-out rows. Both faults are failures, but they
    are found in different places: a saturated link means the linear predictor was
    driven off the scale (extrapolation past the fold's training range), while a
    model shrunk to its intercept lands at the response MEAN. Reporting the first as
    the second sent the reader looking at select=TRUE for a problem that was not there.
    """
    afs = g["assert_folds_scored"]
    floor_ = float(np.finfo(float).eps)          # what betar's linkinv clamps to

    for bound in (floor_, 1.0 - floor_):
        with pytest.raises(AssertionError,
                           match="temporal:1:spearman .saturated_pred.") as e:
            afs(_nan_fold(pred_min=bound, pred_max=bound), EXPECTED_2)
        assert "saturated the link" in str(e.value)
        assert "extrapolation" in str(e.value)

    # A constant prediction in the INTERIOR is still the shrinkage-shaped failure.
    with pytest.raises(AssertionError, match="temporal:1:spearman .constant_pred.") as e:
        afs(_nan_fold(pred_min=0.07, pred_max=0.07), EXPECTED_2)
    assert "select=TRUE" in str(e.value)

    # An EXACT 0 is a deliberate constant, not a clamp: the zero baseline predicts it
    # on purpose, and calling that "saturated" would invent an extrapolation fault.
    with pytest.raises(AssertionError, match="temporal:1:spearman .constant_pred."):
        afs(_nan_fold(pred_min=0.0, pred_max=0.0), EXPECTED_2)

    # A reason recorded by the sweep itself is acted on the same way, and neither
    # form is ever excused as undefined-by-construction.
    with pytest.raises(AssertionError, match="saturated_pred"):
        afs(_nan_fold(spearman_na_reason="saturated_pred"), EXPECTED_2)
    assert "saturated_pred" in g["RANK_NA_DEGENERATE"]
    assert "saturated_pred" not in g["RANK_NA_UNDEFINED"]


def test_spearman_why_separates_saturation_from_a_flat_prediction(g):
    """The Python metric mirror must speak the same distinction as the R sweep."""
    obs = np.array([0.0, 0.1, 0.4, 0.9])
    eps = float(np.finfo(float).eps)
    assert g["_spearman_why"](obs, np.full(4, 0.07))[1] == "constant_pred"
    assert g["_spearman_why"](obs, np.full(4, 0.0))[1] == "constant_pred"   # zero baseline
    assert g["_spearman_why"](obs, np.full(4, eps))[1] == "saturated_pred"
    assert g["_spearman_why"](obs, np.full(4, 1.0 - eps))[1] == "saturated_pred"
    assert g["_spearman_why"](obs, np.array([0.1, 0.2, 0.3, 0.4]))[1] == ""


def test_cover_metrics_records_why_each_spearman_is_nan(g):
    """cover_metrics is the Python mirror of the R sweep's metrics_fn, so it has to
    speak the same reason vocabulary -- that is what makes the two cross-checkable."""
    cm, reasons = g["cover_metrics"], (
        "spearman_na_reason", "spearman_pos_na_reason", "spearman_high_na_reason")
    # Three rows above the high-cover threshold, so every subset can be ranked.
    obs = np.array([0.0] * 10 + [0.05, 0.25, 0.45, 0.65])
    pred = np.linspace(0.01, 0.30, obs.size)

    assert all(cm(obs, pred)[k] == "" for k in reasons)

    zero = cm(np.zeros(obs.size), pred)
    assert zero["spearman_na_reason"] == "constant_obs"
    assert zero["spearman_pos_na_reason"] == "too_few"      # no positive rows at all
    assert np.isnan(zero["spearman"]) and np.isfinite(zero["rmse"])

    flat = cm(obs, np.full(obs.size, 0.07))
    assert all(flat[k] == "constant_pred" for k in reasons)
    assert flat["pred_min"] == flat["pred_max"]             # what the fallback reads

    # Every reason is one the guardrail knows how to act on.
    vocabulary = set(g["RANK_NA_UNDEFINED"]) | set(g["RANK_NA_DEGENERATE"]) | {""}
    for case in (cm(obs, pred), zero, flat, cm(obs[:2], pred[:2])):
        assert {case[k] for k in reasons} <= vocabulary


def test_cv_sweep_records_the_na_reason_and_survives_a_schema_change(g):
    """§13-CV must write the reason columns and bind folds resumed from checkpoints
    written before they existed -- rbind() refuses a mismatched column set outright,
    which would strand every checkpoint the first time a metric column is added."""
    src = [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text())["cells"]
        if c["cell_type"] == "code"
        and "Dependence-aware cross-validation" in "".join(c["source"])
    ]
    assert len(src) == 1, f"expected 1 CV sweep cell, found {len(src)}"
    cell = src[0]
    for reason in ("too_few", "constant_obs", "constant_pred"):
        assert f'return("{reason}")' in cell, f"sp_why does not name {reason}"
    for col in ("spearman_na_reason", "spearman_pos_na_reason",
                "spearman_high_na_reason"):
        assert cell.count(col) >= 2, f"{col} missing from metrics_fn or na_metrics"
    assert "rbind_aligned" in cell and "do.call(rbind, res)" not in cell


def test_cv_fold_coverage_is_asserted_against_the_budgeted_folds(g):
    """§13-CV and §13h must expect the folds the sweep was ASKED for.

    A CV budget (GAM_CV_MAX_SPATIAL_FOLDS / _TEMPORAL_FOLDS) deliberately scores a
    subset of the design; asserting against the full design failed a run that did
    exactly what it was configured to do.
    """
    cells = [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text())["cells"]
        if c["cell_type"] == "code" and "assert_folds_scored(" in "".join(c["source"])
        and "def assert_folds_scored" not in "".join(c["source"])
    ]
    assert len(cells) == 2, f"expected 2 calling cells, found {len(cells)}"
    for src in cells:
        assert "CV_SPATIAL_FOLDS_USED" in src and "CV_TEMPORAL_FOLDS_USED" in src
        # The full design must still be named, so the reduction is reported.
        assert "REDUCED DESIGN" in src


# ---------------------------------------------------------------------
# 5. Metrics and prediction bounds
# ---------------------------------------------------------------------
def test_cover_metrics_are_complete_and_sane(g):
    rng = np.random.default_rng(6)
    obs = np.clip(rng.beta(0.4, 6.0, 4000), 0, 1)
    pred = np.clip(obs * 0.8 + rng.normal(0, 0.01, obs.size), 0, 1)
    m = g["cover_metrics"](obs, pred, high_threshold=0.2)
    for key in ["spearman", "rmse", "mae", "cal_slope", "cal_intercept",
                "area_recovery", "spearman_pos", "rmse_pos", "spearman_high",
                "pred_max", "obs_q99", "out_of_bounds_frac"]:
        assert key in m
    assert m["spearman"] > 0.9
    assert m["area_recovery"] == pytest.approx(0.8, abs=0.1)
    assert m["out_of_bounds_frac"] == 0.0


def test_cover_metrics_flags_out_of_bounds_predictions(g):
    obs = np.array([0.0, 0.1, 0.5])
    pred = np.array([-0.2, 0.1, 1.4])
    m = g["cover_metrics"](obs, pred)
    assert m["out_of_bounds_frac"] == pytest.approx(2 / 3)
    assert m["pred_max"] <= 1.0 and m["pred_min"] >= 0.0


def test_assert_predictions_in_bounds(g):
    assert g["assert_predictions_in_bounds"](np.array([0.0, 0.5, 1.0]))
    with pytest.raises(AssertionError):
        g["assert_predictions_in_bounds"](np.array([0.0, 1.5]))


# ---------------------------------------------------------------------
# 6. Residual dependence gates inference
# ---------------------------------------------------------------------
def test_dependence_verdict_blocks_on_magnitude_not_p_value(g):
    v = g["dependence_verdict"](moran_i=0.74, moran_p=0.001, lag1=0.02)
    assert v["inference_ok"] is False
    assert "INFERENCE BLOCKED" in v["verdict"]
    v2 = g["dependence_verdict"](moran_i=0.05, moran_p=0.001, lag1=0.02)
    assert v2["inference_ok"] is True          # significant but small -> interpretable
    v3 = g["dependence_verdict"](moran_i=0.05, moran_p=0.5, lag1=-0.42)
    assert v3["inference_ok"] is False


def test_residual_lag1_only_pairs_consecutive_months(g):
    # One cell with a gap between rank 1 and rank 5: the gap must not be paired.
    resid = pd.DataFrame({
        "grid_id": ["a"] * 6,
        "time_rank": [0, 1, 5, 6, 7, 8],
        "resid": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
    })
    rho, n_pairs = g["residual_lag1"](resid)
    assert n_pairs == 4          # (0,1), (5,6), (6,7), (7,8) -- not (1,5)
    assert np.isnan(rho)         # below the 100-pair floor -> reported as unavailable


# ---------------------------------------------------------------------
# 7. Evidence labelling
# ---------------------------------------------------------------------
def test_tag_and_assert_evidence(g):
    t = g["tag_evidence"](pd.DataFrame({"a": [1, 2]}), "blocked_validation",
                          model="gam", spec="forcing")
    assert set(t["evidence_type"]) == {"blocked_validation"}
    assert g["assert_evidence_labelled"]({"cv": t})
    with pytest.raises(ValueError):
        g["tag_evidence"](pd.DataFrame({"a": [1]}), "vibes")
    with pytest.raises(AssertionError):
        g["assert_evidence_labelled"]({"raw": pd.DataFrame({"a": [1]})})
