"""Validity tests for ``winam_wh_spatial_panel_predictive_ml.ipynb``.

The helpers are executed straight out of the notebook JSON (the ``## 4e. Shared
corrections`` and ``## 15c (helpers)`` cells), so these tests track the real notebook
rather than a snapshot that can drift away from it. Pure numpy/pandas/scikit-learn:
no Drive, no Earth Engine, no LightGBM.

What is pinned:

1. **Forecast timing** — every rolling origin is issued at t-1, its label cutoff is
   the issue month minus the embargo, and no training month reaches past it.
2. **Training-history length** — an origin needs BOTH >= 24 retained months and >= 24
   elapsed calendar months, the 36-month specification is a strict subset with
   identical training sets, and the inventory reports counts and date ranges.
3. **Threshold construction** — the fractional-cover and fixed-area occurrence
   definitions agree on a fully-classified cell and diverge on a partly-classified
   one; the primary definition is predeclared and a test-set-selected one is refused.
4. **Coverage filtering** — a month must clear BOTH the eligible-cell and the valid
   water-area criterion, and an excluded month leaves a genuine calendar gap.
5. **No test-label leakage** — training months never include the target month, and
   the notebook keeps its inline leakage assertions.
6. **Persistence as the principal benchmark** — squared-error skill, win rates and
   block-bootstrap intervals behave as advertised.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "winam_wh_spatial_panel_predictive_ml.ipynb"
SHARED_MARKER = "4e. Shared corrections"
ORIGIN_MARKER = "15c (helpers)"
EVAL_MARKER = "15e. Repeated one-month-ahead predictive evaluation"
SWEEP_MARKER = "15e2. Thresholded-occurrence definitions evaluated on IDENTICAL folds"
HISTORY_MARKER = "15e3. Training-history sensitivity"
IMPORTANCE_MARKER = "15i. MAIN feature importance"
CONFIG_MARKER = "CV_TEMPORAL_MIN_TRAIN_MONTHS"


def _cells(kind="code"):
    return ["".join(c["source"]) for c in json.loads(NB_PATH.read_text())["cells"]
            if c["cell_type"] == kind]


def _cell(marker, kind="code"):
    hits = [s for s in _cells(kind) if marker in s]
    assert hits, f"no {kind} cell contains {marker!r}"
    return hits[0]


@pytest.fixture(scope="module")
def shared():
    ns: dict = {}
    exec(compile(_cell(SHARED_MARKER), "<shared>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def origins_ns(shared):
    """The §15c namespace, with the §4e helpers already in scope."""
    ns = dict(shared)
    ns["RUN_ML_WORKHORSE"] = True
    exec(compile(_cell(ORIGIN_MARKER), "<origins>", "exec"), ns)
    return ns


# ---------------------------------------------------------------------
# 1. Forecast timing
# ---------------------------------------------------------------------
def _months(start="2018-01-01", n=60):
    return pd.date_range(start, periods=n, freq="MS")


def test_every_origin_is_issued_one_month_before_its_target(origins_ns):
    origins = origins_ns["nf_forecast_origins"](_months())
    assert origins, "no origins built from 60 contiguous months"
    for o in origins:
        t = pd.Timestamp(o["target_month"]).to_period("M")
        assert pd.Timestamp(o["issue_month"]).to_period("M") == t - 1
        assert pd.Timestamp(o["predictor_cutoff"]).to_period("M") == t - 1


def test_label_cutoff_is_the_issue_month_minus_the_embargo(origins_ns):
    embargo = origins_ns["NF_EMBARGO_MONTHS"]
    for o in origins_ns["nf_forecast_origins"](_months()):
        t = pd.Timestamp(o["target_month"]).to_period("M")
        assert pd.Timestamp(o["label_cutoff"]).to_period("M") == t - (1 + embargo)


def test_no_training_month_reaches_past_the_label_cutoff(origins_ns):
    for o in origins_ns["nf_forecast_origins"](_months()):
        tm = pd.DatetimeIndex(pd.to_datetime(o["train_months"]))
        assert tm.max().to_period("M") <= pd.Timestamp(o["label_cutoff"]).to_period("M")
        assert pd.Timestamp(o["target_month"]) not in set(tm)


def test_forecast_timing_check_rejects_a_two_step_origin(origins_ns):
    o = dict(origins_ns["nf_forecast_origins"](_months())[-1])
    o["issue_month"] = (pd.Timestamp(o["target_month"]).to_period("M") - 2).to_timestamp()
    with pytest.raises(AssertionError):
        origins_ns["nf_check_forecast_timing"](o, [], set(), nowcast_ok=True)


def test_origin_history_assertion_fires_on_a_short_history(origins_ns):
    origins = origins_ns["nf_forecast_origins"](_months())
    bad = dict(origins[0])
    bad["train_months"] = pd.DatetimeIndex(pd.to_datetime(bad["train_months"]))[-3:].values
    with pytest.raises(AssertionError):
        origins_ns["nf_assert_origin_history"]([bad])


# ---------------------------------------------------------------------
# 2. Training-history length
# ---------------------------------------------------------------------
def test_minimum_training_history_is_twenty_four_months_not_six(origins_ns):
    assert origins_ns["NF_MIN_TRAIN_MONTHS"] == 24
    assert origins_ns["NF_MIN_ELAPSED_MONTHS"] == 24
    assert origins_ns["NF_SENS_MIN_TRAIN_MONTHS"] == 36
    assert origins_ns["NF_SENS_MIN_ELAPSED_MONTHS"] == 36


def test_configuration_cell_declares_the_same_history_requirements():
    cfg = _cell(CONFIG_MARKER)
    assert "CV_TEMPORAL_MIN_TRAIN_MONTHS = 24" in cfg
    assert "CV_TEMPORAL_MIN_ELAPSED_MONTHS = 24" in cfg
    assert "CV_TEMPORAL_SENSITIVITY_MIN_TRAIN_MONTHS = 36" in cfg
    assert "CV_TEMPORAL_SENSITIVITY_MIN_ELAPSED_MONTHS = 36" in cfg
    assert "CV_TEMPORAL_MIN_TRAIN_MONTHS = 6" not in cfg


def test_an_origin_needs_twenty_four_retained_months(origins_ns):
    # 26 contiguous months: with a one-month embargo only the last months can have a
    # 24-month history at the label cutoff.
    origins = origins_ns["nf_forecast_origins"](_months(n=26))
    for o in origins:
        assert len(o["train_months"]) >= 24
    assert all(len(o["train_months"]) >= 24 for o in origins)
    assert not origins_ns["nf_forecast_origins"](_months(n=20))


def test_elapsed_calendar_history_is_required_as_well_as_retained_months(origins_ns):
    """24 clear months scattered over five years must not pass as 24 months of record.

    The reverse case is the one the elapsed rule catches: a long span with few
    retained months. Here 12 retained months spread over 60 calendar months clear the
    elapsed bar but fail the retained-month bar, and must be refused.
    """
    sparse = pd.DatetimeIndex(sorted(_months(n=60)[::5]))   # 12 months over 5 years
    assert len(sparse) == 12
    assert not origins_ns["nf_forecast_origins"](sparse, min_train_months=24,
                                                 min_elapsed_months=24)
    # Relaxing only the retained-month rule lets them through, which shows the two
    # requirements are independent rather than one implying the other.
    assert origins_ns["nf_forecast_origins"](sparse, min_train_months=6,
                                             min_elapsed_months=24)


def test_elapsed_rule_refuses_a_dense_but_short_history(origins_ns):
    dense_short = _months(n=30)
    assert origins_ns["nf_forecast_origins"](dense_short, min_train_months=24,
                                             min_elapsed_months=24)
    assert not origins_ns["nf_forecast_origins"](dense_short, min_train_months=24,
                                                 min_elapsed_months=36)


def test_thirty_six_month_origins_are_a_subset_with_identical_training_sets(origins_ns):
    mm = _months(n=72)
    primary = origins_ns["nf_forecast_origins"](mm, min_train_months=24,
                                                min_elapsed_months=24)
    sens = origins_ns["nf_forecast_origins"](mm, min_train_months=36,
                                             min_elapsed_months=36)
    assert 0 < len(sens) < len(primary)
    by_month = {pd.Timestamp(o["target_month"]): o for o in primary}
    for o in sens:
        t = pd.Timestamp(o["target_month"])
        assert t in by_month, "the 36-month run admits an origin the 24-month run does not"
        np.testing.assert_array_equal(
            pd.to_datetime(o["train_months"]).values,
            pd.to_datetime(by_month[t]["train_months"]).values)


def test_origin_summary_reports_the_count_and_the_date_range(origins_ns):
    mm = _months(n=60)
    spec = dict(min_train_months=24, min_elapsed_months=24)
    origins = origins_ns["nf_forecast_origins"](mm, **spec)
    summary = origins_ns["nf_origin_summary"](origins, "primary_24m", spec)
    assert summary["n_origins"] == len(origins)
    assert summary["first_origin"] == pd.Timestamp(
        origins[0]["target_month"]).strftime("%Y-%m")
    assert summary["last_origin"] == pd.Timestamp(
        origins[-1]["target_month"]).strftime("%Y-%m")
    assert summary["min_train_months_used"] >= 24
    assert summary["min_elapsed_used"] >= 24


def test_origin_summary_is_empty_safe(origins_ns):
    summary = origins_ns["nf_origin_summary"]([], "primary_24m",
                                              dict(min_train_months=24,
                                                   min_elapsed_months=24))
    assert summary["n_origins"] == 0 and summary["first_origin"] is None


def test_notebook_reports_the_origin_inventory_and_the_common_origin_comparison():
    feats = _cell("15d. Nowcast vs forecast feature sets")
    assert "nf_origin_inventory" in feats and "nf_origin_summary" in feats
    hist = _cell(HISTORY_MARKER)
    for token in ("common_", "sensitivity_36m", "nf_history_sensitivity",
                  "nf_history_inventory"):
        assert token in hist, f"§15e3 does not mention {token}"
    assert "agree exactly" in hist


# ---------------------------------------------------------------------
# 3. Threshold construction
# ---------------------------------------------------------------------
def test_the_evaluated_thresholds_are_the_declared_grid(shared):
    defs = shared["build_presence_definitions"](cell_size_m=500)
    assert [d["name"] for d in defs] == [
        "cover_0.005", "cover_0.01", "cover_0.02",
        "area_0.125ha", "area_0.25ha", "area_0.5ha"]
    cfg = _cell(CONFIG_MARKER)
    assert "PRESENCE_COVER_THRESHOLD_GRID   = (0.005, 0.01, 0.02)" in cfg
    assert "PRESENCE_AREA_HA_THRESHOLD_GRID = (0.125, 0.25, 0.5)" in cfg


def test_cover_and_area_thresholds_agree_on_a_fully_classified_cell(shared):
    """0.5%/1%/2% of a full 25 ha cell is 0.125/0.25/0.5 ha, so the labels coincide."""
    cover = np.array([0.0, 0.004, 0.006, 0.011, 0.021, 0.4])
    full = pd.DataFrame({"wh_cover": cover,
                         "wh_area_ha": cover * 25.0,          # 500 m cell, fully valid
                         "valid_area_m2": 250_000.0})
    for frac, ha in ((0.005, 0.125), (0.01, 0.25), (0.02, 0.5)):
        a = shared["apply_presence_definition"](
            full, shared["make_presence_definition"]("cover", frac, 500))
        b = shared["apply_presence_definition"](
            full, shared["make_presence_definition"]("area_ha", ha, 500))
        np.testing.assert_array_equal(a, b)


def test_cover_and_area_thresholds_diverge_on_a_half_classified_cell(shared):
    """A half-observed cell reaches 1% cover on half the hectares — the reason for both."""
    half = pd.DataFrame({"wh_cover": [0.012], "wh_area_ha": [0.012 * 12.5],
                         "valid_area_m2": [125_000.0]})
    by_cover = shared["apply_presence_definition"](
        half, shared["make_presence_definition"]("cover", 0.01, 500))
    by_area = shared["apply_presence_definition"](
        half, shared["make_presence_definition"]("area_ha", 0.25, 500))
    assert bool(by_cover[0]) and not bool(by_area[0])


def test_every_definition_is_labelled_as_thresholded_occurrence(shared):
    for d in shared["build_presence_definitions"](cell_size_m=500):
        assert "thresholded occurrence" in d["label"]
        assert "not literal biological presence" in d["wording"]


def test_inventory_reports_prevalence_and_the_share_of_total_wh_area(shared):
    cover = np.array([0.0, 0.001, 0.006, 0.03, 0.5])
    frame = pd.DataFrame({"wh_cover": cover, "wh_area_ha": cover * 25.0,
                          "valid_area_m2": 250_000.0})
    inv = shared["presence_definition_inventory"](
        frame, shared["build_presence_definitions"](cell_size_m=500))
    assert {"prevalence", "share_of_total_wh_area"} <= set(inv.columns)
    row = inv[inv["definition"] == "cover_0.005"].iloc[0]
    assert row["prevalence"] == pytest.approx(3 / 5)
    assert row["share_of_total_wh_area"] == pytest.approx(
        (0.006 + 0.03 + 0.5) / cover.sum())
    # A stricter threshold labels fewer cells but never more area.
    strict = inv[inv["definition"] == "cover_0.02"].iloc[0]
    assert strict["prevalence"] < row["prevalence"]
    assert strict["share_of_total_wh_area"] <= row["share_of_total_wh_area"]


def test_the_primary_definition_is_predeclared_and_test_selection_is_refused(shared):
    defs = shared["build_presence_definitions"](cell_size_m=500)
    assert shared["assert_predeclared_presence_definition"](defs, "cover_0.02",
                                                            "predeclared")
    assert shared["assert_predeclared_presence_definition"](defs, "cover_0.01",
                                                            "nested_training_only")
    with pytest.raises(AssertionError):
        shared["assert_predeclared_presence_definition"](defs, "cover_0.01",
                                                         "best_outer_test_auc")
    with pytest.raises(AssertionError):
        shared["assert_predeclared_presence_definition"](defs, "cover_0.03",
                                                         "predeclared")


def test_notebook_declares_a_predeclared_primary_and_never_selects_on_test():
    cfg = _cell(CONFIG_MARKER)
    assert 'PRESENCE_PRIMARY_DEFINITION = "cover_0.02"' in cfg
    assert 'PRESENCE_THRESHOLD_SELECTION_SOURCE = "predeclared"' in cfg
    sweep = _cell(SWEEP_MARKER)
    assert "assert_predeclared_presence_definition" in sweep
    assert "_train_only_threshold" in sweep
    assert "SENSITIVITY display" in sweep


def test_the_sweep_reports_every_required_metric_on_identical_folds():
    sweep = _cell(SWEEP_MARKER)
    for token in ("prevalence", "share_of_total_wh_area", "presence_auc",
                  "presence_ap", "presence_brier", "colonisation_precision",
                  "colonisation_recall", "colonisation_f1", "hurdle_gbm_rmse",
                  "hurdle_gbm_mae", "hurdle_gbm_msss_clim", "area_mae_ha",
                  "area_bias_ha"):
        assert token in sweep, f"the occurrence sweep does not report {token}"
    # identical folds: the same primary origins drive every definition
    assert "NF_ORIGIN_SPECS[NF_PRIMARY_SPEC]" in sweep


def test_the_continuous_response_is_untouched_by_the_threshold_work():
    """`wh_cover` stays hard WH area / valid area; no confidence weighting returns."""
    cfg = _cell(CONFIG_MARKER)
    assert "USE_PROBABILITY_RESPONSE = False" in cfg
    assert "WEIGHT_COVER_BY_CONFIDENCE = False" in cfg
    sweep = _cell(SWEEP_MARKER)
    assert "continuous response wh_cover is UNTOUCHED" in sweep
    for src in _cells():
        assert "ML_CONFIDENCE_WEIGHTING = True" not in src
        assert 'NF_CONF_WEIGHT_MODE = "confidence"' not in src


# ---------------------------------------------------------------------
# 4. Coverage filtering
# ---------------------------------------------------------------------
def _panel(n_cells=20, n_months=12, cell_area=250_000.0, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    gid = np.repeat(np.arange(n_cells), n_months)
    mo = pd.DatetimeIndex(np.tile(months.values, n_cells))
    cover = np.clip(rng.beta(0.3, 8, len(gid)), 0, 1)
    return pd.DataFrame({
        "grid_id": gid, "month": mo, "month_num": mo.month,
        "valid_pixels": 1000, "valid_area_m2": cell_area,
        "valid_fraction": 1.0,
        "wh_area_m2": cover * cell_area, "wh_cover": cover,
        "wh_area_ha": cover * cell_area / 1e4})


def test_a_month_must_clear_both_coverage_criteria(shared):
    panel = _panel()
    cells = np.sort(panel["grid_id"].unique())
    thin_area = pd.Timestamp("2020-05-01")
    # Every cell still observed, but each classified only 40% of its area.
    mask = panel["month"] == thin_area
    panel.loc[mask, "valid_area_m2"] *= 0.4
    panel.loc[mask, "valid_fraction"] = 0.4

    cells_only = shared["monthly_coverage_audit"](panel, cells, min_coverage=0.9)
    assert bool(cells_only.set_index("month").loc[thin_area, "retained"]), \
        "the cell criterion alone should pass a fully-observed but thinly-classified month"

    both = shared["monthly_coverage_audit"](panel, cells, min_coverage=0.9,
                                            min_area_fraction=0.9)
    row = both.set_index("month").loc[thin_area]
    assert bool(row["coverage_ok"]) and not bool(row["area_ok"])
    assert not bool(row["retained"])
    assert "valid water area" in row["exclusion_reason"]


def test_the_area_criterion_measures_the_eligible_footprint(shared):
    panel = _panel()
    cells = np.sort(panel["grid_id"].unique())
    audit = shared["monthly_coverage_audit"](panel, cells, min_coverage=0.9,
                                             min_area_fraction=0.9)
    # Every month is complete, so each observes the whole eligible footprint.
    assert np.allclose(audit["valid_area_fraction_of_eligible"], 1.0)
    assert audit["retained"].all()


def test_a_month_failing_only_the_cell_criterion_is_still_excluded(shared):
    panel = _panel()
    cells = np.sort(panel["grid_id"].unique())
    thin_cells = pd.Timestamp("2020-07-01")
    drop = panel[(panel["month"] == thin_cells) & (panel["grid_id"] >= 5)].index
    panel = panel.drop(index=drop)
    audit = shared["monthly_coverage_audit"](panel, cells, min_coverage=0.9,
                                             min_area_fraction=0.9)
    row = audit.set_index("month").loc[thin_cells]
    assert not bool(row["coverage_ok"]) and not bool(row["retained"])
    assert "cell coverage" in row["exclusion_reason"]


def test_an_excluded_month_leaves_a_genuine_calendar_gap(shared):
    panel = _panel(n_months=12)
    cells = np.sort(panel["grid_id"].unique())
    gap = pd.Timestamp("2020-06-01")
    mask = panel["month"] == gap
    panel.loc[mask, "valid_area_m2"] *= 0.3
    audit = shared["monthly_coverage_audit"](panel, cells, min_coverage=0.9,
                                             min_area_fraction=0.9)
    kept = shared["apply_monthly_coverage_filter"](panel, audit)
    assert gap not in set(kept["month"])

    lagged = shared["calendar_lag"](kept, ["wh_cover"], periods=1)
    after = lagged[lagged["month"] == pd.Timestamp("2020-07-01")]
    assert len(after) and after["wh_cover_lag1"].isna().all(), \
        "the month after an excluded month must have a missing lag, not May's value"
    # ... and a positional shift would have reached straight across it.
    positional = kept.sort_values(["grid_id", "month"]).copy()
    positional["shifted"] = positional.groupby("grid_id")["wh_cover"].shift(1)
    assert positional[positional["month"] == pd.Timestamp("2020-07-01")][
        "shifted"].notna().any()


def test_cell_month_validity_applies_both_criteria(shared):
    panel = _panel(n_cells=6, n_months=4)
    panel.loc[panel.index[:3], "valid_pixels"] = 2          # fails the pixel rule
    panel.loc[panel.index[3:6], "valid_fraction"] = 0.2     # fails the fraction rule
    out, report = shared["apply_cell_month_validity_filter"](
        panel, min_valid_pixels=10, min_valid_fraction=0.5)
    assert report["n_rows_removed_valid_pixels"] == 3
    assert report["n_rows_removed_valid_fraction"] == 3
    assert len(out) == len(panel) - 6
    assert (out["valid_pixels"] >= 10).all() and (out["valid_fraction"] >= 0.5).all()


def test_cell_month_validity_is_a_no_op_when_unconfigured(shared):
    panel = _panel(n_cells=4, n_months=3)
    out, report = shared["apply_cell_month_validity_filter"](panel)
    assert len(out) == len(panel)
    assert not report["valid_pixels_applied"] and not report["valid_fraction_applied"]


def test_valid_fraction_is_derived_from_the_nominal_cell_area(shared):
    panel = pd.DataFrame({"valid_area_m2": [250_000.0, 125_000.0, 400_000.0]})
    out = shared["cell_month_valid_fraction"](panel, cell_size_m=500)
    np.testing.assert_allclose(out["valid_fraction"], [1.0, 0.5, 1.0])


def test_notebook_filters_validity_before_the_fractional_response_is_formed():
    seven = _cell("DUPLICATE_MONTH_METHOD must be either")
    validity = seven.index("apply_cell_month_validity_filter")
    cover = seven.index('panel_sensor["wh_cover"] = np.where')
    assert validity < cover, \
        "the cell-month validity filter must run before wh_cover is computed"
    assert "MIN_VALID_FRACTION_PER_CELL_MONTH" in seven


def test_notebook_configures_a_minimum_valid_fraction_and_area_coverage():
    cfg = _cell(CONFIG_MARKER)
    assert "MIN_VALID_FRACTION_PER_CELL_MONTH = 0.5" in cfg
    assert "MIN_MONTHLY_VALID_AREA_FRACTION = 0.90" in cfg
    ninth = _cell("9d. Monthly coverage: measure it, audit it")
    assert "min_area_fraction=MIN_MONTHLY_VALID_AREA_FRACTION" in ninth
    assert "apply_cell_month_validity_filter" in ninth


# ---------------------------------------------------------------------
# 5. No test-label leakage
# ---------------------------------------------------------------------
def test_notebook_keeps_its_inline_leakage_assertions():
    ev = _cell(EVAL_MARKER)
    assert 'assert te["month"].min() > pd.Timestamp(_o["label_cutoff"])' in ev
    sweep = _cell(SWEEP_MARKER)
    assert 'assert te["month"].min() > pd.Timestamp(_o["label_cutoff"])' in sweep


def test_training_rows_never_include_the_target_month(origins_ns):
    mm = _months(n=60)
    frame = pd.DataFrame({"month": np.repeat(mm, 5),
                          "grid_id": np.tile(np.arange(5), len(mm))})
    for o in origins_ns["nf_forecast_origins"](mm):
        tr = frame[frame["month"].isin(o["train_months"])]
        te = frame[frame["month"] == o["target_month"]]
        assert len(te) and not set(tr["month"]) & set(te["month"])
        assert tr["month"].max() <= pd.Timestamp(o["label_cutoff"])


def test_forecast_feature_set_still_refuses_contemporaneous_proxies(origins_ns):
    contemporaneous = {"rain_chirps_30d_mm"}
    with pytest.raises(AssertionError):
        origins_ns["nf_assert_forecast_safe"](["rain_chirps_30d_mm"], contemporaneous)
    with pytest.raises(AssertionError):
        origins_ns["nf_assert_forecast_safe"](["turb_ndti_s2"], set())
    assert origins_ns["nf_assert_forecast_safe"](
        ["turb_ndti_s2_lag1", "wh_cover_lag1", "depth_m"], contemporaneous)


# ---------------------------------------------------------------------
# 6. Persistence as the principal benchmark
# ---------------------------------------------------------------------
def test_squared_error_skill_against_persistence(shared):
    obs = np.array([0.0, 0.1, 0.2, 0.3])
    persistence = np.array([0.0, 0.0, 0.1, 0.2])
    assert shared["msss_vs_persistence"](obs, persistence, persistence) == pytest.approx(0.0)
    assert shared["msss_vs_persistence"](obs, obs, persistence) == pytest.approx(1.0)
    worse = persistence - 0.5
    assert shared["msss_vs_persistence"](obs, worse, persistence) < 0


def test_change_metrics_are_defined_for_every_baseline_including_zero(shared):
    rng = np.random.default_rng(1)
    persistence = rng.random(200) * 0.2
    obs = np.clip(persistence + rng.normal(0, 0.02, 200), 0, 1)
    for pred in (np.zeros(200), persistence, obs):
        out = shared["change_skill_vs_persistence"](obs, pred, persistence)
        assert np.isfinite(out["change_msss"])
    # the change skill IS the persistence skill, by construction
    pred = np.clip(persistence + 0.5 * (obs - persistence), 0, 1)
    assert shared["change_skill_vs_persistence"](obs, pred, persistence)["change_msss"] \
        == pytest.approx(shared["msss_vs_persistence"](obs, pred, persistence))


def test_change_direction_hit_rate_rewards_getting_the_sign_right(shared):
    persistence = np.array([0.1, 0.1, 0.1, 0.1])
    obs = np.array([0.2, 0.2, 0.0, 0.0])
    perfect = shared["change_skill_vs_persistence"](obs, obs, persistence)
    inverted = shared["change_skill_vs_persistence"](
        obs, 2 * persistence - obs, persistence)
    assert perfect["change_sign_hit"] == pytest.approx(1.0)
    assert inverted["change_sign_hit"] == pytest.approx(0.0)


def test_win_rate_counts_the_blocks_a_model_beats_persistence_in(shared):
    frame = pd.DataFrame({
        "block": np.repeat([1, 2, 3, 4], 4),
        "model": ["persistence", "persistence", "good", "good"] * 4,
        "obs": [1.0, 0.0] * 8,
        "pred": ([0.9, 0.1] + [0.99, 0.01]) * 3 + ([0.9, 0.1] + [0.2, 0.8]),
    })
    per_block, summary = shared["beats_reference_rate"](frame, "block")
    good = summary.set_index("model").loc["good"]
    assert good["n_blocks"] == 4 and good["n_blocks_won"] == 3
    assert good["win_rate_vs_persistence"] == pytest.approx(0.75)
    assert summary.set_index("model").loc["persistence",
                                          "win_rate_vs_persistence"] == 0.0


def test_block_bootstrap_resamples_whole_blocks_and_brackets_the_point(shared):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"block": np.repeat(np.arange(30), 20),
                          "v": rng.normal(0.4, 0.1, 600)})
    ci = shared["block_bootstrap_ci"](frame, "block",
                                      lambda s: float(np.mean(s["v"])),
                                      n_boot=400, random_state=0)
    assert ci["n_blocks"] == 30 and ci["n_boot"] > 0
    assert ci["lo"] < ci["point"] < ci["hi"]
    # A block bootstrap is wider than a naive row-wise standard error would be.
    assert ci["hi"] - ci["lo"] > 0


def test_block_bootstrap_is_degenerate_safe(shared):
    frame = pd.DataFrame({"block": [1, 1, 1], "v": [0.1, 0.2, 0.3]})
    ci = shared["block_bootstrap_ci"](frame, "block", lambda s: float(np.mean(s["v"])))
    assert np.isnan(ci["lo"]) and np.isnan(ci["hi"])


def test_mse_skill_stat_matches_the_direct_computation(shared):
    frame = pd.DataFrame({
        "model": ["persistence"] * 4 + ["tweedie_gbm"] * 4,
        "obs": [0.1, 0.2, 0.3, 0.4] * 2,
        "pred": [0.0, 0.1, 0.2, 0.3] + [0.1, 0.2, 0.3, 0.5]})
    stat = shared["mse_skill_stat"](reference="persistence", model="tweedie_gbm")
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    expected = shared["msss_vs_persistence"](obs, np.array([0.1, 0.2, 0.3, 0.5]),
                                             np.array([0.0, 0.1, 0.2, 0.3]))
    assert stat(frame) == pytest.approx(expected)


def test_evaluation_leads_on_persistence_and_carries_intervals_and_win_rates():
    ev = _cell(EVAL_MARKER)
    for token in ("msss_persistence", "nf_persistence_skill", "nf_persistence_ci",
                  "nf_persistence_wins", "block_bootstrap_ci",
                  "origin_win_rate_vs_persistence", "window_win_rate_vs_persistence"):
        assert token in ev, f"§15e does not report {token}"
    assert "PRINCIPAL BENCHMARK = persistence" in ev


def test_temporal_residual_correlation_is_demoted_to_a_diagnostic():
    ev = _cell(EVAL_MARKER)
    assert "DIAGNOSTIC ONLY" in ev
    assert "NOT headline evidence of dynamic skill" in ev
    # ... and the exported file says so in its name.
    assert "nf_temporal_residual_corr_DIAGNOSTIC_" in ev
    summary = _cell("15k. Summary", kind="markdown")
    assert "temporal_resid_corr" in summary
    assert "diagnostic" in summary.lower()


def test_tweedie_is_the_principal_continuous_cover_model():
    ev = _cell(EVAL_MARKER)
    assert "PRINCIPAL continuous-cover ML model = tweedie_gbm" in ev
    summary = _cell("15k. Summary", kind="markdown")
    assert "tweedie" in summary.lower() and "secondary" in summary.lower()


# ---------------------------------------------------------------------
# 7. Importance breadth and framing
# ---------------------------------------------------------------------
def test_importance_is_not_limited_to_the_four_latest_origins():
    imp = _cell(IMPORTANCE_MARKER)
    assert "NF_PERM_N_ORIGINS         = 4" not in imp
    assert 'NF_PERM_ORIGIN_MODE       = "representative"' in imp
    assert "NF_PERM_N_ORIGINS         = 12" in imp
    assert "_select_origins" in imp


def test_grouped_importance_covers_the_correlated_blocks_with_uncertainty():
    imp = _cell(IMPORTANCE_MARKER)
    for group in ("wh_lag_memory", "wind", "rainfall", "water_quality",
                  "static_habitat", "neighbour_advection"):
        assert group in imp, f"no grouped importance for {group}"
    assert "nf_grouped_importance" in imp
    assert "block_bootstrap_ci" in imp and "_ci_lo" in imp
    assert "NOT causal" in imp or "not causal" in imp


def test_feature_grouping_rules_assign_the_expected_blocks(origins_ns):
    """The grouping function is exercised directly, out of the notebook source."""
    src = _cell(IMPORTANCE_MARKER)
    start = src.index("    def nf_feature_group(col):")
    end = src.index("    nf_feature_groups = {}")
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in src[start:end].splitlines())
    ns: dict = {}
    exec(compile(body, "<groups>", "exec"), ns)
    group = ns["nf_feature_group"]
    assert group("wh_cover_lag1") == "wh_lag_memory"
    assert group("wh_cover_roll3_lag1") == "wh_lag_memory"
    assert group("wh_cover_neigh_lag1") == "neighbour_advection"
    assert group("wh_adv_upwind_flux_lag1") == "neighbour_advection"
    assert group("wind_speed_ms_lag1") == "wind"
    assert group("rain_chirps_30d_mm_lag1") == "rainfall"
    assert group("turb_ndti_s2_lag1") == "water_quality"
    assert group("chl_ndci_s2_lag1") == "water_quality"
    assert group("depth_m") == "static_habitat"
    assert group("wave_exposure_idx") == "static_habitat"
    assert group("month_num") == "calendar"


# ---------------------------------------------------------------------
# 8. Stale notices and superseded sections
# ---------------------------------------------------------------------
def test_the_no_metrics_are_committed_notice_is_gone():
    for src in _cells("markdown") + _cells():
        assert "no metrics are committed" not in src.lower(), \
            "a stale 'no metrics are committed' notice is still in the notebook"


def test_superseded_track_a_outputs_are_separated_from_the_primary_evaluation():
    md = "\n".join(_cells("markdown"))
    assert "SUPERSEDED" in md
    assert "§15e" in md and "§15b" in md


def test_the_valid_fraction_filter_cannot_be_silently_skipped():
    """A missing `valid_fraction` column must raise, not quietly disable the filter."""
    for marker in ("9d. Monthly coverage: measure it, audit it",
                   "Before habitat/valid-cell filters"):
        src = _cell(marker)
        assert 'MIN_VALID_FRACTION_PER_CELL_MONTH is None or' in src, \
            f"{marker} does not assert the valid-fraction filter actually ran"


def test_the_evaluation_records_the_mse_it_compares_on():
    """`beats_persistence` must come from a directly computed MSE, not from rmse**2.

    A square-root-then-square round trip perturbs the last bits, which would let
    persistence beat itself in a strict comparison.
    """
    ev = _cell(EVAL_MARKER)
    assert "mse_model = float(np.mean((obs - pred) ** 2))" in ev
    assert "beats_persistence=bool(mse_model < mse_pers)" in ev
    assert "beats_persistence=bool(rmse ** 2 < mse_pers)" not in ev


def test_every_code_cell_in_the_notebook_parses():
    """A cell that does not compile would fail only when the analyst reaches it."""
    import ast

    for i, src in enumerate(_cells()):
        stripped = src.lstrip()
        if stripped.startswith(("!", "%%", "%")) or "\n!" in src:
            continue   # IPython magics / shell escapes are not Python
        try:
            ast.parse(src)
        except SyntaxError as exc:  # pragma: no cover - the assertion is the report
            raise AssertionError(f"code cell {i} does not parse: {exc}") from exc
