"""Behavioural tests for the purely temporal driver notebook.

``winam_wh_temporal_driver_model.ipynb`` collapses the 500 m cell-month panel to
one AOI value per month and fits interpretable temporal models to it. Its §4/§5
helper cells are pure numpy/pandas/statsmodels, so they are executed here
straight out of the notebook JSON — the tests cannot drift from the code they
describe.

What is pinned:

* the AOI response is AREA-WEIGHTED (sum of WH area / sum of valid area), not a
  mean of per-cell cover, and the two differ whenever cells differ in size;
* the fixed cell set and coverage filter actually exclude what they claim to,
  so a cloudy month cannot change the *composition* of the average;
* the monthly series is calendar-complete, so a lag never reaches across an
  excluded month and calls a months-old value "last month";
* Newey-West bandwidth follows the standard rule and HAC standard errors are
  wider than naive ones on a persistent series;
* Shapley R^2 values sum to the full-model R^2 and split shared variance
  between collinear drivers instead of awarding it to whichever entered first;
* the synthetic self-test recovers the effects it was built with, and does NOT
  recover the decoys.

Everything temporal is pinned to CALENDAR MONTHS rather than row positions,
because the record has excluded months and complete-case rows compress them:

* the calendar HAC covariance reproduces statsmodels' HAC exactly on a gapless
  series, and a single missing month removes the two one-month covariance pairs
  that touched it;
* bootstrap blocks are contiguous runs of calendar months and keep the
  missing-month pattern, so separated observed months are never made adjacent;
* rolling-origin test windows span exactly the requested calendar interval,
  never train on or after the window start, never let a test month escape the
  window, are skipped rather than widened when too few months are evaluable,
  and score every compared model on the same response months;
* the seasonal-naive baseline looks up the response at exactly t-12 calendar
  months and reports "unavailable" instead of substituting a nearby row;
* the long-run multiplier is refused when the AR interval reaches a unit root,
  and multi-lag stationarity is decided on the AR polynomial's roots;
* shared-versus-unique variance may only merge a Shapley decomposition with the
  semi-partial values of the SAME fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "winam_wh_temporal_driver_model.ipynb"

pytest.importorskip("statsmodels")
import statsmodels.api as sm  # noqa: E402


def _cell(marker):
    """The first code cell containing `marker`, as source text."""
    for c in json.loads(NOTEBOOK.read_text())["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise AssertionError(f"no code cell contains {marker!r}")


@pytest.fixture(scope="module")
def ns():
    """§4 + §5 helper namespace, executed out of the notebook."""
    namespace = {"pd": pd, "np": np, "sm": sm, "Path": Path,
                 "display": lambda *a, **k: None}
    import scipy.stats as sstats
    namespace["sstats"] = sstats
    exec(compile(_cell("4. Loading and AOI aggregation"), "<agg>", "exec"), namespace)
    exec(compile(_cell("5. Calendar-aware, autocorrelation-robust inference helpers"),
                 "<inf>", "exec"), namespace)
    return namespace


def _panel(n_cells=40, n_months=24, seed=0, cloudy_months=(5, 6)):
    """A cell-month panel where cells differ in valid area and some months are cloudy."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for gid in range(n_cells):
        # Cell 0 is ten times the valid area of the rest, so an area-weighted
        # mean and an unweighted mean of cover cannot coincide.
        valid = 100.0 if gid else 1000.0
        for i, m in enumerate(months):
            if i in cloudy_months and gid >= n_cells // 4:
                continue                      # cloudy month: most cells missing
            cover = 0.05 + 0.02 * np.sin(2 * np.pi * i / 12) + rng.normal(0, 0.005)
            cover = float(np.clip(cover, 0.001, 0.5))
            rows.append({"grid_id": gid, "month": m,
                         "valid_area_ha": valid, "wh_area_ha": cover * valid,
                         "wh_cover": cover, "wh_present": float(cover > 0.02),
                         "rain_chirps_30d_mm": 50 + 30 * np.sin(2 * np.pi * i / 12)
                         + rng.normal(0, 5),
                         "air_temp_c": 25 + rng.normal(0, 1),
                         "depth_m": 3.0 + gid * 0.05})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The AOI response
# ---------------------------------------------------------------------------

def test_aoi_response_is_area_weighted_not_a_mean_of_cover(ns):
    """sum(WH area)/sum(valid area) — a mean of per-cell cover is a different number."""
    panel = _panel(cloudy_months=())
    out = ns["aoi_monthly_series"](panel, driver_cols=["rain_chirps_30d_mm"])
    row = out.iloc[0]
    month0 = panel[panel["month"] == row["month"]]
    expected = month0["wh_area_ha"].sum() / month0["valid_area_ha"].sum()
    assert row["wh_cover_aoi"] == pytest.approx(expected)
    # The unweighted mean of cover is genuinely different, so the choice matters.
    assert row["wh_cover_aoi"] != pytest.approx(month0["wh_cover"].mean(), rel=1e-6)


def test_driver_aggregation_uses_the_same_weights_as_the_response(ns):
    """A driver and the response must describe the same piece of lake."""
    panel = _panel(cloudy_months=())
    out = ns["aoi_monthly_series"](panel, driver_cols=["rain_chirps_30d_mm"])
    month0 = panel[panel["month"] == out["month"].iloc[0]]
    w = month0["valid_area_ha"]
    expected = float((month0["rain_chirps_30d_mm"] * w).sum() / w.sum())
    assert out["rain_chirps_30d_mm"].iloc[0] == pytest.approx(expected)


def test_equal_weighting_differs_from_area_weighting(ns):
    panel = _panel(cloudy_months=())
    area = ns["aoi_monthly_series"](panel, weighting="valid_area")
    equal = ns["aoi_monthly_series"](panel, weighting="equal")
    # The response is an area ratio either way; the DRIVER means are what change.
    a = ns["aoi_monthly_series"](panel, weighting="valid_area",
                                 driver_cols=["depth_m"])["depth_m"].iloc[0]
    e = ns["aoi_monthly_series"](panel, weighting="equal",
                                 driver_cols=["depth_m"])["depth_m"].iloc[0]
    assert a != pytest.approx(e)
    assert len(area) == len(equal)


def test_missing_driver_values_do_not_leak_zeros_into_the_mean(ns):
    """A NaN driver must be excluded from its month's mean, not counted as zero."""
    panel = _panel(cloudy_months=())
    panel.loc[panel["grid_id"] == 0, "air_temp_c"] = np.nan   # the huge cell
    out = ns["aoi_monthly_series"](panel, driver_cols=["air_temp_c"])
    month0 = panel[(panel["month"] == out["month"].iloc[0]) & panel["air_temp_c"].notna()]
    w = month0["valid_area_ha"]
    expected = float((month0["air_temp_c"] * w).sum() / w.sum())
    assert out["air_temp_c"].iloc[0] == pytest.approx(expected)
    assert out["air_temp_c"].iloc[0] > 20            # not dragged toward zero


# ---------------------------------------------------------------------------
# Coverage filter and the fixed cell set
# ---------------------------------------------------------------------------

def test_coverage_filter_excludes_the_cloudy_months(ns):
    panel = _panel(cloudy_months=(5, 6))
    cov = ns["monthly_coverage_table"](panel, 0.90)
    cloudy = cov.iloc[[5, 6]]
    assert not cloudy["retained"].any()
    assert cov.drop(index=[5, 6])["retained"].all()


def test_fixed_cell_set_reports_the_tradeoff_it_makes(ns):
    panel = _panel(cloudy_months=(5, 6))
    cov = ns["monthly_coverage_table"](panel, 0.90)
    kept_months = cov.loc[cov["retained"], "month"]
    cells, audit = ns["fixed_cell_set"](panel, kept_months, 0.80)
    assert audit["n_cells_kept"] <= audit["n_cells_total"]
    assert 0.0 <= audit["share_of_classified_area_kept"] <= 1.0
    assert audit["n_months_retained"] == len(kept_months)


def test_fixed_cell_set_drops_cells_observed_too_rarely(ns):
    """A cell seen in only a couple of months must not join the average."""
    panel = _panel(cloudy_months=())
    months = sorted(panel["month"].unique())
    # Cell 7 exists in the first two months only.
    panel = panel[~((panel["grid_id"] == 7) & (panel["month"].isin(months[2:])))]
    cells, _ = ns["fixed_cell_set"](panel, pd.Series(months), 0.80)
    assert 7 not in set(cells)
    assert 0 in set(cells)


# ---------------------------------------------------------------------------
# Calendar-correct lags
# ---------------------------------------------------------------------------

def test_reindex_makes_the_month_grid_unbroken(ns):
    monthly = pd.DataFrame({
        "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-05-01"]),
        "wh_cover_aoi": [0.1, 0.2, 0.3]})
    out = ns["reindex_calendar_months"](monthly)
    assert len(out) == 5                                    # Jan..May inclusive
    assert out["observed"].tolist() == [True, True, False, False, True]
    assert out["time_index"].tolist() == [0, 1, 2, 3, 4]


def test_lag_does_not_reach_across_an_excluded_month(ns):
    """The whole point of the calendar grid: a gap stays a gap."""
    monthly = pd.DataFrame({
        "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-05-01"]),
        "wh_cover_aoi": [0.1, 0.2, 0.3], "rain": [10.0, 20.0, 50.0]})
    out = ns["reindex_calendar_months"](monthly)
    out, made = ns["calendar_lag"](out, ["rain"], 1)
    assert made == ["rain_lag1"]
    may = out[out["month"] == pd.Timestamp("2020-05-01")].iloc[0]
    # April was never observed, so May's "last month" must be missing, NOT 20.0.
    assert pd.isna(may["rain_lag1"])
    feb = out[out["month"] == pd.Timestamp("2020-02-01")].iloc[0]
    assert feb["rain_lag1"] == pytest.approx(10.0)


def test_zero_lag_leaves_the_column_name_alone(ns):
    monthly = pd.DataFrame({"month": pd.date_range("2020-01-01", periods=4, freq="MS"),
                            "rain": [1.0, 2.0, 3.0, 4.0]})
    out, made = ns["calendar_lag"](monthly, ["rain"], 0)
    assert made == ["rain"]
    assert out["rain"].tolist() == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Response transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("how", ["logit", "log", "identity"])
def test_response_transform_is_finite_at_the_boundaries(ns, how):
    y, info = ns["transform_response"](pd.Series([0.0, 0.5, 1.0]), how, 1e-4)
    assert np.isfinite(y).all()
    assert info["transform"] == how
    if how != "identity":
        assert info["n_clipped"] >= 1


def test_logit_is_monotone_and_centred_at_a_half(ns):
    y, _ = ns["transform_response"](pd.Series([0.1, 0.5, 0.9]), "logit", 1e-4)
    assert y.iloc[0] < y.iloc[1] < y.iloc[2]
    assert y.iloc[1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Autocorrelation-robust inference
# ---------------------------------------------------------------------------

def test_hac_bandwidth_follows_the_standard_rule(ns):
    assert ns["hac_maxlags"](100) == 4
    assert ns["hac_maxlags"](12) >= 1
    assert ns["hac_maxlags"](100, override=7) == 7


def test_hac_standard_errors_exceed_naive_ones_on_a_persistent_series(ns):
    """The reason every SE in the notebook is HAC."""
    rng = np.random.default_rng(0)
    n = 150
    x = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):                     # both x and the error are AR(1)
        x[t] = 0.85 * x[t - 1] + rng.normal()
        e[t] = 0.85 * e[t - 1] + rng.normal()
    y = 0.3 * x + e
    X = pd.DataFrame({"x": x})
    hac = ns["fit_hac"](y, X)
    naive = sm.OLS(y, sm.add_constant(X)).fit()
    assert hac.bse["x"] > naive.bse["x"]


def test_bh_fdr_is_monotone_and_bounded(ns):
    q = ns["bh_fdr"]([0.001, 0.02, 0.3, 0.9])
    assert (q.to_numpy() >= np.array([0.001, 0.02, 0.3, 0.9]) - 1e-12).all()
    assert (q <= 1).all()
    assert q.is_monotonic_increasing


def test_bh_fdr_tolerates_missing_pvalues(ns):
    q = ns["bh_fdr"]([0.01, np.nan, 0.5])
    assert np.isnan(q.iloc[1])
    assert np.isfinite(q.iloc[0]) and np.isfinite(q.iloc[2])


# ---------------------------------------------------------------------------
# Calendar-aware HAC covariance
#
# The record has excluded months, so two rows that are neighbours in the
# complete-case table can be several calendar months apart. These pin that the
# sandwich pairs observations by CALENDAR distance.
# ---------------------------------------------------------------------------

def _ar1_series(n, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(scale=0.5)
    return x, 1.4 * x + e


def test_calendar_hac_matches_statsmodels_when_there_are_no_gaps(ns):
    """The estimator is the ordinary Newey-West sandwich, verified against statsmodels."""
    n = 96
    x, y = _ar1_series(n, seed=11)
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"x": x})
    ref = sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="HAC", cov_kwds={"maxlags": ns["hac_maxlags"](n), "use_correction": True})
    cal = ns["fit_hac"](y, X, months=months)
    assert np.allclose(np.asarray(cal.bse), np.asarray(ref.bse), rtol=1e-10, atol=1e-12)
    assert np.allclose(np.asarray(cal.conf_int()), np.asarray(ref.conf_int()),
                       rtol=1e-10, atol=1e-12)
    assert cal._hac_calendar_aware is True


def test_calendar_hac_matches_statsmodels_for_wls_too(ns):
    n = 96
    x, y = _ar1_series(n, seed=12)
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"x": x})
    w = np.abs(np.random.default_rng(3).normal(size=n)) + 0.5
    ref = sm.WLS(y, sm.add_constant(X), weights=w / w.mean()).fit(
        cov_type="HAC", cov_kwds={"maxlags": ns["hac_maxlags"](n), "use_correction": True})
    cal = ns["fit_hac"](y, X, weights=w, months=months)
    assert np.allclose(np.asarray(cal.bse), np.asarray(ref.bse), rtol=1e-10, atol=1e-12)


def test_a_missing_month_removes_the_pairs_that_cross_it(ns):
    """Observations either side of a gap must not contribute as a one-month pair."""
    n = 96
    x, y = _ar1_series(n, seed=13)
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    keep = np.r_[np.arange(40), np.arange(41, n)]          # delete one month
    gapped = ns["fit_hac"](y[keep], pd.DataFrame({"x": x[keep]}), months=months[keep])
    # 95 rows would give 94 consecutive pairs if adjacency were row-based; the
    # deleted month breaks the two pairs that touched it.
    assert gapped._hac_pair_counts[1] == (n - 1) - 2
    row_based = ns["fit_hac"](y[keep], pd.DataFrame({"x": x[keep]}))
    assert row_based._hac_calendar_aware is False
    assert not np.allclose(np.asarray(gapped.bse), np.asarray(row_based.bse))


def test_calendar_hac_refuses_unsorted_or_duplicated_months(ns):
    x, y = _ar1_series(24, seed=14)
    months = pd.date_range("2015-01-01", periods=24, freq="MS")[::-1]
    with pytest.raises(ValueError):
        ns["fit_hac"](y, pd.DataFrame({"x": x}), months=months)


def test_month_index_differences_are_calendar_months(ns):
    mi = ns["month_index"](pd.to_datetime(["2019-11-01", "2020-01-01", "2020-02-01"]))
    assert list(np.diff(mi)) == [2, 1]
    back = ns["months_from_index"](mi)
    assert list(back) == list(pd.to_datetime(["2019-11-01", "2020-01-01", "2020-02-01"]))


# ---------------------------------------------------------------------------
# Calendar moving-block bootstrap
# ---------------------------------------------------------------------------

def _gapped_months(n=96, drop=(40, 41, 70)):
    full = pd.date_range("2015-01-01", periods=n, freq="MS")
    keep = [i for i in range(n) if i not in set(drop)]
    return full, full[keep], np.asarray(keep)


def test_bootstrap_blocks_are_contiguous_in_calendar_months(ns):
    _, months, _ = _gapped_months()
    rng = np.random.default_rng(5)
    seen_lengths = set()
    for _idx, blocks in ns["calendar_block_indices"](months, block_months=6, rng=rng,
                                                     n_boot=30):
        for blk in blocks[:-1]:
            seen_lengths.add(len(blk))
        for blk in blocks:
            assert np.all(np.diff(np.asarray(blk, dtype=int)) == 1), \
                "a sampled block skips a calendar month"
        assert sum(len(b) for b in blocks) == ns["calendar_span_months"](months)
    assert seen_lengths == {6}


def test_bootstrap_blocks_preserve_the_missing_month_pattern(ns):
    """A block landing on an excluded month yields fewer rows; it does not close the gap."""
    _, months, _ = _gapped_months()
    mi = ns["month_index"](months)
    rng = np.random.default_rng(6)
    sizes, saw_short_block = set(), False
    for idx, blocks in ns["calendar_block_indices"](months, block_months=6, rng=rng,
                                                    n_boot=40):
        sizes.add(len(idx))
        for blk in blocks:
            n_rows = int(np.isin(np.asarray(blk, dtype=int), mi).sum())
            assert n_rows <= len(blk)
            if n_rows < len(blk):
                saw_short_block = True
    assert saw_short_block, "no block ever covered an excluded month"
    assert len(sizes) > 1, "replicate sample size never varied despite the gaps"


def test_bootstrap_brackets_the_point_estimate(ns):
    """The bootstrap distribution centres on the SAMPLE estimate, not on truth."""
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.3, size=n)
    months = pd.date_range("2012-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"x": x})
    point = float(sm.OLS(y, sm.add_constant(X)).fit().params["x"])
    draws, info = ns["bootstrap_coefficients"](y, X, months, n_boot=300, seed=3)
    summ = ns["bootstrap_summary"](draws, terms=["x"]).iloc[0]
    assert summ["boot_ci_lo"] < point < summ["boot_ci_hi"]
    assert summ["boot_median"] == pytest.approx(point, abs=0.05)
    assert summ["boot_median"] == pytest.approx(2.0, abs=0.15)
    assert summ["boot_sign_stability"] == pytest.approx(1.0)
    # The replicate count and fitted-size distribution must be reported.
    assert info["n_successful"] == len(draws) > 0
    assert info["block_months"] >= 1
    for key in ("n_rows_fitted_min", "n_rows_fitted_median", "n_rows_fitted_max"):
        assert key in info


# ---------------------------------------------------------------------------
# Calendar rolling-origin cross-validation
# ---------------------------------------------------------------------------

def _eval_months(n=96, drop=()):
    full = pd.date_range("2015-01-01", periods=n, freq="MS")
    return full.delete(list(drop)) if drop else full


def test_rolling_origin_windows_are_exactly_the_requested_calendar_interval(ns):
    months = _eval_months()
    folds, audit = ns["rolling_origin_month_folds"](
        months, n_folds=8, horizon_months=3, min_train_months=24, min_test_months=1)
    assert folds
    for f in folds:
        span = ((f["test_window_end"].year - f["test_window_start"].year) * 12
                + f["test_window_end"].month - f["test_window_start"].month + 1)
        assert span == 3 == f["horizon_months_requested"]
        # The origin is the calendar month immediately before the window.
        assert f["origin_month"] + pd.DateOffset(months=1) == f["test_window_start"]
    assert (audit["horizon_months_requested"] == 3).all()


def test_rolling_origin_never_trains_on_or_after_the_test_window(ns):
    months = _eval_months()
    folds, _ = ns["rolling_origin_month_folds"](
        months, n_folds=8, horizon_months=3, min_train_months=24, min_test_months=1)
    for f in folds:
        assert months[f["train_idx"]].max() < f["test_window_start"]
        assert len(f["train_idx"]) >= 24


def test_no_test_month_escapes_its_declared_window(ns):
    months = _eval_months(drop=(90, 93))
    folds, _ = ns["rolling_origin_month_folds"](
        months, n_folds=8, horizon_months=3, min_train_months=24, min_test_months=1)
    for f in folds:
        te = months[f["test_idx"]]
        assert te.min() >= f["test_window_start"]
        assert te.max() <= f["test_window_end"]
        assert len(te) <= f["horizon_months_requested"]


def test_three_rows_spanning_six_months_is_not_a_three_month_horizon(ns):
    """The bug this replaces: consecutive ROWS were treated as a 3-month window."""
    # 2022-07, -08 and -10 are three rows but four calendar months.
    months = pd.to_datetime(
        [f"2020-{m:02d}-01" for m in range(1, 13)]
        + [f"2021-{m:02d}-01" for m in range(1, 13)]
        + ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01",
           "2022-06-01", "2022-07-01", "2022-08-01", "2022-10-01"])
    folds, audit = ns["rolling_origin_month_folds"](
        months, n_folds=3, horizon_months=3, min_train_months=12, min_test_months=1)
    last = audit.iloc[-1]
    assert last["test_window_start"] == pd.Timestamp("2022-08-01")
    assert last["test_window_end"] == pd.Timestamp("2022-10-01")
    assert last["n_test_months_evaluable"] == 2          # 2022-09 is absent
    assert "2022-09" in last["omitted_test_months"]


def test_folds_without_enough_evaluable_months_are_skipped_not_widened(ns):
    months = _eval_months(drop=(93, 94))       # leaves one evaluable month in a window
    folds, audit = ns["rolling_origin_month_folds"](
        months, n_folds=8, horizon_months=3, min_train_months=24, min_test_months=2)
    skipped = audit[~audit["usable"]]
    assert len(skipped) >= 1
    assert (skipped["skip_reason"].str.len() > 0).all()
    assert "NOT widened" in " ".join(skipped["skip_reason"])
    # Every window still declares the requested horizon; none was stretched.
    assert (audit["n_calendar_months_in_window"] == 3).all()
    assert all(len(f["test_idx"]) >= 2 for f in folds)


def test_fold_audit_records_every_omitted_month_with_a_reason(ns):
    months = _eval_months(drop=(90,))
    folds, audit = ns["rolling_origin_month_folds"](
        months, n_folds=8, horizon_months=3, min_train_months=24, min_test_months=1,
        omission_reason={pd.Timestamp("2022-07-01"): "response not observed"})
    row = audit[audit["omitted_test_months"].str.len() > 0].iloc[0]
    assert row["n_test_months_omitted"] >= 1
    assert "=" in row["omitted_reasons"]
    for col in ("origin_month", "train_start", "train_end", "test_window_start",
                "test_window_end", "horizon_months_requested",
                "n_test_months_evaluable", "n_test_months_omitted"):
        assert col in audit.columns


def test_rolling_origin_refuses_folds_it_cannot_train(ns):
    months = pd.date_range("2015-01-01", periods=20, freq="MS")
    folds, _ = ns["rolling_origin_month_folds"](
        months, n_folds=8, horizon_months=3, min_train_months=24)
    assert folds == []


def test_cv_scores_beats_the_mean_on_a_learnable_series(ns):
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.2, size=n)
    months = pd.date_range("2012-01-01", periods=n, freq="MS")
    folds, _ = ns["rolling_origin_month_folds"](
        months, n_folds=6, horizon_months=3, min_train_months=24, min_test_months=1)
    sc, detail = ns["cv_scores"](y, pd.DataFrame({"x": x}), folds, months=months)
    assert sc["n_folds"] == len(folds)
    assert sc["r2_oos"] > 0.8
    assert len(detail) == sc["n_test"]
    # The real month of every prediction is retained.
    assert detail["month"].notna().all()
    assert set(detail["month"]) <= set(months)


def test_compared_models_are_scored_on_the_same_response_months(ns):
    rng = np.random.default_rng(15)
    n = 96
    x = rng.normal(size=n)
    y = 1.2 * x + rng.normal(scale=0.3, size=n)
    months = pd.date_range("2015-01-01", periods=n, freq="MS").delete([80, 88])
    y, x = y[: len(months)], x[: len(months)]
    folds, _ = ns["rolling_origin_month_folds"](
        months, n_folds=6, horizon_months=3, min_train_months=24, min_test_months=1)
    _, d1 = ns["cv_scores"](y, pd.DataFrame({"x": x}), folds, months=months)
    _, d2 = ns["cv_scores"](y, pd.DataFrame({"x2": x ** 2}), folds, months=months)
    assert list(d1["month"]) == list(d2["month"])


# ---------------------------------------------------------------------------
# Seasonal-naive baseline
# ---------------------------------------------------------------------------

def test_seasonal_naive_uses_the_same_calendar_month_a_year_earlier(ns):
    grid = pd.DataFrame({"month": pd.date_range("2015-01-01", periods=48, freq="MS")})
    grid["y"] = np.arange(48, dtype=float)
    pred_months = grid["month"].iloc[24:30]
    out = ns["seasonal_naive_predictions"](pred_months, grid)
    assert (out["source_month"] == out["month"] - pd.DateOffset(months=12)).all()
    assert out["yhat"].to_numpy() == pytest.approx(np.arange(12, 18, dtype=float))


def test_seasonal_naive_does_not_take_the_twelfth_previous_observed_row(ns):
    """With a gap, `shift(12)` and a calendar lookup give different answers."""
    grid = pd.DataFrame({"month": pd.date_range("2015-01-01", periods=48, freq="MS")})
    grid["y"] = np.arange(48, dtype=float)
    holed = grid.drop(index=[13, 14]).reset_index(drop=True)     # 2016-02, 2016-03 gone
    out = ns["seasonal_naive_predictions"](pd.to_datetime(["2017-01-01"]), holed)
    # Calendar answer: 2016-01 -> 12.0. The twelfth previous observed row would
    # be a different month entirely, because two rows were removed.
    assert out["source_month"].iloc[0] == pd.Timestamp("2016-01-01")
    assert out["yhat"].iloc[0] == pytest.approx(12.0)
    shifted = holed["y"].shift(12).iloc[-1]
    assert shifted != pytest.approx(12.0)


def test_seasonal_naive_reports_unavailable_rather_than_substituting(ns):
    grid = pd.DataFrame({"month": pd.date_range("2015-01-01", periods=48, freq="MS")})
    grid["y"] = np.arange(48, dtype=float)
    holed = grid.drop(index=[13]).reset_index(drop=True)          # 2016-02 gone
    out = ns["seasonal_naive_predictions"](pd.to_datetime(["2017-02-01"]), holed)
    assert out["available"].iloc[0] is np.False_ or not bool(out["available"].iloc[0])
    assert pd.isna(out["yhat"].iloc[0])
    assert out["source_month"].iloc[0] == pd.Timestamp("2016-02-01")


# ---------------------------------------------------------------------------
# Stationarity gate on the long-run multiplier
# ---------------------------------------------------------------------------

class _ARStub:
    """Minimal stand-in exposing just what `ar_stability` reads."""

    def __init__(self, phi, se):
        self.params = pd.Series(dict([("const", 0.0)]
                                     + [(f"y_lag{j}", v) for j, v in enumerate(phi, 1)]))
        self._V = np.diag(np.r_[0.0, np.asarray(se, dtype=float) ** 2])

    def cov_params(self):
        return pd.DataFrame(self._V, index=self.params.index, columns=self.params.index)


def test_long_run_multiplier_is_refused_when_the_interval_reaches_one(ns):
    """rho ~ 0.88 with a CI touching 1 -- exactly this notebook's situation."""
    out = ns["ar_stability"](_ARStub([0.88], [0.07]), ["y_lag1"])
    assert out["rho_ci_hi"] > 1.0
    assert out["stationary_supported"] is False
    assert out["reason"] == "AR confidence interval includes a unit root"


def test_long_run_multiplier_is_allowed_when_the_interval_is_comfortably_inside(ns):
    out = ns["ar_stability"](_ARStub([0.45], [0.08]), ["y_lag1"])
    assert out["stationary_supported"] is True
    assert out["reason"] == ""
    assert -1 < out["rho_ci_lo"] and out["rho_ci_hi"] < 1


def test_explosive_point_estimate_is_refused(ns):
    out = ns["ar_stability"](_ARStub([1.02], [0.01]), ["y_lag1"])
    assert out["stationary_supported"] is False
    assert "root" in out["reason"]


def test_multi_lag_stationarity_uses_the_roots_not_the_coefficient_sum(ns):
    """sum(phi) < 1 is not sufficient once there is more than one lag."""
    explosive = ns["ar_polynomial_roots"]([0.6, 0.5])       # sums to 1.1
    stationary = ns["ar_polynomial_roots"]([1.2, -0.35])    # sums to 0.85
    assert not np.all(np.abs(explosive) > 1.0)
    assert np.all(np.abs(stationary) > 1.0)
    # A pair whose coefficients sum to -0.95 -- comfortably inside a naive
    # "sum < 1" test -- can still sit outside the stationary region.
    assert not np.all(np.abs(ns["ar_polynomial_roots"]([0.1, -1.05])) > 1.0)


# ---------------------------------------------------------------------------
# Shared vs unique variance must come from ONE fit
# ---------------------------------------------------------------------------

def _partition_and_semi(ns, n=140, seed=21):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(scale=0.4, size=n)
    c = rng.normal(size=n)
    y = 1.0 * a + 0.4 * c + rng.normal(scale=0.4, size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    part = ns["shapley_r2"](y, X, {k: [k] for k in X.columns},
                            specification="spec", response="y", weighting="none")
    semi = ns["semi_partial_r2"](y, X, terms=list(X.columns),
                                 specification="spec", response="y", weighting="none")
    return part, semi


def test_shared_vs_unique_merges_only_matching_fits(ns):
    part, semi = _partition_and_semi(ns)
    out = ns["shared_vs_unique"](part, semi, ["a", "b", "c"], "spec")
    assert set(out["term"]) == {"a", "b", "c"}
    assert (out["specification"] == "spec").all()
    assert "last_entry_to_shapley_ratio" in out.columns
    # The old, misleading name must be gone.
    assert "unique_share_of_shapley" not in out.columns


def test_shared_vs_unique_refuses_a_mismatched_row_count(ns):
    """The §17 bug: a dynamic Shapley table beside a static semi-partial table."""
    part, _ = _partition_and_semi(ns, n=140, seed=21)
    _, semi_other = _partition_and_semi(ns, n=120, seed=21)
    with pytest.raises(AssertionError, match="row count"):
        ns["shared_vs_unique"](part, semi_other, ["a", "b", "c"], "spec")


def test_shared_vs_unique_refuses_mismatched_model_columns(ns):
    part, semi = _partition_and_semi(ns)
    semi = semi.copy()
    semi["model_columns"] = "a|b"                       # a different design matrix
    with pytest.raises(AssertionError, match="model columns"):
        ns["shared_vs_unique"](part, semi, ["a", "b", "c"], "spec")


def test_ratio_above_one_is_labelled_not_clipped(ns):
    """Suppression can push the last-entry contribution above the Shapley value."""
    part, semi = _partition_and_semi(ns)
    semi = semi.copy()
    semi.loc[semi["term"] == "a", "semi_partial_r2"] = (
        float(part.loc[part["group"] == "a", "shapley_r2"].iloc[0]) * 1.4)
    out = ns["shared_vs_unique"](part, semi, ["a", "b", "c"], "spec").set_index("term")
    assert out.loc["a", "last_entry_to_shapley_ratio"] == pytest.approx(1.4, rel=1e-6)
    assert "suppression" in out.loc["a", "ratio_note"]


def test_partition_and_semi_partial_carry_their_provenance(ns):
    part, semi = _partition_and_semi(ns)
    for tab in (part, semi):
        for col in ("specification", "n_obs", "model_columns", "response", "weighting"):
            assert col in tab.columns
    assert part["n_obs"].iloc[0] == semi["n_obs"].iloc[0]
    assert part["model_columns"].iloc[0] == semi["model_columns"].iloc[0]


# ---------------------------------------------------------------------------
# Variance partitioning
# ---------------------------------------------------------------------------

def test_shapley_values_sum_to_the_full_model_r2(ns):
    rng = np.random.default_rng(5)
    n = 150
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 1.0 * a + 0.5 * b + rng.normal(scale=0.4, size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    out = ns["shapley_r2"](y, X, {"a": ["a"], "b": ["b"], "c": ["c"]})
    full = float(out["r2_full_model"].iloc[0])
    assert out["shapley_r2"].sum() == pytest.approx(full, abs=1e-8)
    assert out.iloc[0]["group"] == "a"          # the strongest driver ranks first


def test_shapley_splits_variance_shared_by_collinear_drivers(ns):
    """Two near-identical drivers must share the credit, not race for it."""
    rng = np.random.default_rng(6)
    n = 200
    a = rng.normal(size=n)
    a_copy = a + rng.normal(scale=0.01, size=n)     # essentially the same series
    y = 1.0 * a + rng.normal(scale=0.3, size=n)
    out = ns["shapley_r2"](y, pd.DataFrame({"a": a, "a_copy": a_copy}),
                           {"a": ["a"], "a_copy": ["a_copy"]}).set_index("group")
    assert out.loc["a", "shapley_r2"] == pytest.approx(
        out.loc["a_copy", "shapley_r2"], rel=0.15)


def test_shapley_refuses_an_intractable_number_of_groups(ns):
    X = pd.DataFrame(np.random.default_rng(7).normal(size=(60, 15)),
                     columns=[f"v{i}" for i in range(15)])
    with pytest.raises(ValueError, match="reduce PARTITION_MAX_GROUPS"):
        ns["shapley_r2"](np.random.default_rng(8).normal(size=60), X,
                         {c: [c] for c in X.columns})


def test_semi_partial_r2_is_zero_for_a_duplicated_driver(ns):
    """A driver that duplicates another explains nothing UNIQUELY — the check §17 relies on."""
    rng = np.random.default_rng(9)
    n = 150
    a = rng.normal(size=n)
    y = a + rng.normal(scale=0.3, size=n)
    out = ns["semi_partial_r2"](y, pd.DataFrame({"a": a, "a_dup": a.copy()}),
                               terms=["a", "a_dup"]).set_index("term")
    assert out.loc["a_dup", "semi_partial_r2"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Cross-correlation and standardisation
# ---------------------------------------------------------------------------

def test_cross_correlation_finds_a_planted_lag(ns):
    rng = np.random.default_rng(10)
    n = 120
    driver = rng.normal(size=n)
    response = np.r_[np.nan, np.nan, driver[:-2]] * 2.0 + rng.normal(scale=0.2, size=n)
    ccf = ns["cross_correlation"](driver, response, max_lag=6)
    peak = ccf.loc[ccf["r"].abs().idxmax(), "lag"]
    assert peak == 2


def test_cross_correlation_positive_lag_means_the_driver_leads(ns):
    """Sign convention matters: only a leading driver can support a driver claim."""
    n = 60
    driver = np.arange(n, dtype=float)
    response = np.r_[np.nan, driver[:-1]]
    ccf = ns["cross_correlation"](driver, response, max_lag=3).set_index("lag")
    assert ccf.loc[1, "r"] == pytest.approx(1.0)


def test_cross_correlation_does_not_close_a_calendar_gap(ns):
    """A driver value cannot lead the response by 1 month across a missing month."""
    n = 60
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    driver = np.arange(n, dtype=float)
    response = np.r_[np.nan, driver[:-1]]
    keep = np.r_[np.arange(30), np.arange(31, n)]          # one month deleted
    ccf = ns["cross_correlation"](driver[keep], response[keep], months=months[keep],
                                  max_lag=2).set_index("lag")
    # 59 rows: row adjacency would give 58 lag-1 pairs. On the calendar the
    # deleted month costs the two pairs that touched it.
    assert ccf.loc[1, "n"] == (n - 1) - 2
    assert ccf.loc[1, "r"] == pytest.approx(1.0)


def test_acf_pairs_observations_by_calendar_distance(ns):
    n = 60
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(31)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    keep = np.r_[np.arange(30), np.arange(31, n)]
    acf = ns["acf_values"](x[keep], months=months[keep], nlags=3).set_index("lag")
    assert acf.loc[1, "n_pairs"] == (n - 1) - 2
    assert acf.loc[2, "n_pairs"] == (n - 2) - 2
    # Row-order ACF would silently use all 58 adjacent rows.
    row_based = ns["acf_values"](x[keep], nlags=3).set_index("lag")
    assert row_based.loc[1, "n_pairs"] == len(keep) - 1


def test_acf_does_not_drop_nan_rows_and_slide_the_series(ns):
    """A NaN month must cost pairs, not shift the months after it one lag closer."""
    n = 40
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    x = np.arange(n, dtype=float)
    x[20] = np.nan
    acf = ns["acf_values"](x, months=months, nlags=2).set_index("lag")
    assert acf.loc[1, "n_pairs"] == (n - 1) - 2


def test_zscore_frame_reports_reversible_scaling(ns):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out, scaling = ns["zscore_frame"](df, ["x"])
    assert out["x"].mean() == pytest.approx(0.0)
    assert out["x"].std(ddof=1) == pytest.approx(1.0)
    row = scaling.iloc[0]
    back = out["x"] * row["sd"] + row["mean"]
    assert back.tolist() == pytest.approx(df["x"].tolist())


def test_zscore_frame_marks_a_constant_column_instead_of_dividing_by_zero(ns):
    out, scaling = ns["zscore_frame"](pd.DataFrame({"c": [5.0] * 6}), ["c"])
    assert not np.isfinite(scaling["sd"].iloc[0])
    assert out["c"].isna().all()


def test_vif_flags_a_collinear_pair(ns):
    rng = np.random.default_rng(11)
    a = rng.normal(size=100)
    X = pd.DataFrame({"a": a, "a2": a + rng.normal(scale=0.01, size=100),
                      "b": rng.normal(size=100)})
    vif = ns["vif_table"](X).set_index("term")
    assert vif.loc["a", "vif"] > 10
    assert vif.loc["b", "vif"] < 5


# ---------------------------------------------------------------------------
# The synthetic self-test must actually be a test
# ---------------------------------------------------------------------------

def test_synthetic_series_recovers_its_planted_effects_and_not_the_decoys(ns):
    """§4's synthetic generator is the notebook's own self-check; it must work."""
    monthly, truth = ns["make_synthetic_monthly"](120, seed=1)
    monthly = ns["reindex_calendar_months"](monthly)
    monthly, season = ns["add_season_terms"](monthly, 2)
    monthly["y"], _ = ns["transform_response"](monthly["wh_cover_aoi"], "logit", 1e-4)
    monthly, _ = ns["calendar_lag"](monthly, ["rain_chirps_30d_mm"], 1)

    terms = ["rain_chirps_30d_mm_lag1", "wave_exposure_idx", "decoy_noise", "decoy_level"]
    monthly, _ = ns["zscore_frame"](monthly, terms)
    cols = terms + season + ["time_index"]
    sub = monthly[["y"] + cols].dropna()
    res = ns["fit_hac"](sub["y"], sub[cols])

    assert res.params["rain_chirps_30d_mm_lag1"] > 0        # planted positive
    assert res.pvalues["rain_chirps_30d_mm_lag1"] < 0.05
    assert res.params["wave_exposure_idx"] < 0              # planted negative
    for decoy in ("decoy_noise", "decoy_level"):
        assert res.pvalues[decoy] > 0.05, f"{decoy} is pure noise and must not be found"
    assert truth["decoy_noise"] == 0.0 and truth["decoy_level"] == 0.0


def test_synthetic_series_is_a_calendar_complete_monthly_frame(ns):
    monthly, _ = ns["make_synthetic_monthly"](36, seed=2)
    assert len(monthly) == 36
    assert monthly["month"].is_monotonic_increasing
    assert (monthly["wh_cover_aoi"].between(0, 1)).all()
    # The response must be reconstructible from the areas the notebook aggregates.
    assert (monthly["wh_area_ha"] / monthly["valid_area_ha"]).to_numpy() == pytest.approx(
        monthly["wh_cover_aoi"].to_numpy())


# ---------------------------------------------------------------------------
# Structural contract: the notebook must keep the guarantees it advertises
# ---------------------------------------------------------------------------

def test_notebook_declares_static_covariates_unidentifiable():
    """A temporal model cannot speak to time-invariant habitat variables."""
    cfg = _cell("3a. Where the data comes from")
    for col in ("depth_m", "dist_shore_m", "frac_cropland"):
        assert col in cfg, f"{col} must be listed in KNOWN_STATIC_COLS and dropped by name"
    build = _cell("7. Build the AOI monthly series")
    assert "KNOWN_STATIC_COLS" in build
    assert "cannot explain temporal" in build


def test_notebook_excludes_endogenous_optical_proxies_from_driver_claims():
    cfg = _cell("3a. Where the data comes from")
    assert "TEMPORAL_PROXY_TERMS" in cfg
    for name in ("chl_mci_s3", "turb_ndti_s2"):
        assert name in cfg
    synth = _cell("25. Synthesis table")
    # Proxies must never appear in the ranked driver table's term list.
    assert "PROXY_TERMS" not in synth


def test_notebook_uses_calendar_rolling_origin_not_random_cv():
    for marker in ("13. Elastic net", "16. Rolling-origin out-of-sample skill"):
        src = _cell(marker)
        assert "rolling_origin_month_folds" in src
        assert "KFold" not in src and "train_test_split" not in src


def test_notebook_scores_skill_against_persistence_and_seasonal_baselines():
    src = _cell("16. Rolling-origin out-of-sample skill")
    for baseline in ("mean baseline", "y_lag1", "seasonal-naive"):
        assert baseline in src
    # §16's y_lag1 specification is a FITTED regression and must not be sold as
    # literal persistence; the unfitted y_{t-1} rule is §21's baseline.
    assert "NOT literal persistence" in src


def test_notebook_reports_the_cv_design_in_calendar_months():
    src = _cell("16. Rolling-origin out-of-sample skill")
    assert "calendar-month rolling-origin windows" in src
    assert "n_evaluated" in src or "evaluated month" in src
    # The seasonal-naive baseline must go through the calendar lookup helper,
    # never a positional shift (the comment explaining why may still say so).
    assert "seasonal_naive_predictions" in src
    code = [ln for ln in src.split("\n") if not ln.strip().startswith("#")]
    assert not any("shift(12)" in ln for ln in code)


def test_notebook_asserts_a_common_evaluation_sample():
    src = _cell("16. Rolling-origin out-of-sample skill")
    assert "scored on different response months" in src
    assert "different n_test" in src


def test_notebook_gates_the_long_run_multiplier_on_stationarity():
    src = _cell("12. Model A - nested linear")
    assert "ar_stability" in src
    assert "long_run_estimable" in src
    assert "LONGRUN_REQUIRE_STATIONARITY" in src
    cfg = _cell("3a. Where the data comes from")
    assert "LONGRUN_REQUIRE_STATIONARITY = True" in cfg


def test_notebook_builds_shared_vs_unique_within_one_specification():
    src = _cell("17. Shapley R^2 partitioning")
    assert "SHARED_VS_UNIQUE_STATIC" in src and "SHARED_VS_UNIQUE_AR" in src
    assert "SEMI_PARTIAL_AR" in src
    # The primary reading is the dynamic specification.
    assert 'SHARED_VS_UNIQUE_SPEC = "with persistence"' in src
    model_a = _cell("12. Model A - nested linear")
    assert "SEMI_PARTIAL_AR = semi_partial_r2(" in model_a


def test_notebook_exports_carry_evidence_type_and_is_synthetic():
    src = _cell("26. Export tables and a run manifest")
    assert 'out["evidence_type"] = evidence' in src
    assert 'out["is_synthetic"] = SOURCE["is_synthetic"]' in src
    for name in ("semi_partial_r2_with_ar", "shared_vs_unique_variance_with_ar",
                 "cv_fold_audit", "seasonal_naive_predictions", "ar_stability"):
        assert f'"{name}"' in src


def test_notebook_self_tests_its_calendar_helpers():
    src = _cell("5b. Self-tests")
    assert "HELPER_SELFTESTS" in src
    assert "raise AssertionError" in src


def test_corrected_cells_carry_no_stale_outputs():
    """A cell whose numbers changed must not ship the previous run's output."""
    markers = [
        "# 3a. Where the data comes from",
        "# 5. Calendar-aware, autocorrelation-robust inference helpers",
        "# 5b. Self-tests for the calendar-aware helpers",
        "# 8. Response series diagnostics",
        "# 9. Identifiability audit",
        "# 10. Cross-correlation evidence",
        "# 11. Model dataset",
        "# 12. Model A - nested linear",
        "# 12b. Coefficient plot",
        "# 12c. Moving-block bootstrap",
        "# 12e. Descriptive proxy association",
        "# 13. Elastic net",
        "# 14. Natural-spline GLM",
        "# 15. Gradient boosting",
        "# 16. Rolling-origin out-of-sample skill",
        "# 17. Shapley R^2 partitioning",
        "# 18a. Response-definition",
        "# 18b. Leave-one-year-out",
        "# 18c. Deseasonalised anomalies",
        "# 25. Synthesis table",
        "# 26. Export tables and a run manifest",
    ]
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(m in src for m in markers):
            assert not c.get("outputs"), \
                f"stale outputs on a corrected cell: {src.splitlines()[1]!r}"


def test_synthesis_verdicts_cover_the_documented_categories():
    src = _cell("25. Synthesis table")
    for verdict in ("robust", "suggestive", "sign contradicts mechanism",
                    "not separable from season", "no evidence"):
        assert f'"{verdict}"' in src


def test_verdicts_ignore_selection_frequency_when_the_net_is_not_sparse():
    """A near-ridge fit selects everything; that must not read as robustness."""
    src = _cell("25. Synthesis table")
    assert "ENET_SPARSITY_OK" in src
    enet = _cell("13. Elastic net")
    assert "ENET_SPARSITY_OK" in enet


def test_notebook_regenerates_from_its_builder():
    """The builder script is the editable source of the notebook."""
    assert (REPO / "build_temporal_model_nb.py").exists()
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 30


# ---------------------------------------------------------------------------
# §5c / §19-§24: the state-space dynamic-regression model
#
# The static models put persistence on the right-hand side as `y_lag1`, which
# deletes every month whose predecessor is missing and assumes the disturbance
# is exactly AR(1). The principal model puts the dependence in the process
# instead. What is pinned here is the discipline around it, because that is what
# makes the driver estimates readable at all:
#
#   * the dependence structure is chosen from NULL dynamics, with no driver in
#     any candidate, and locked before the drivers enter;
#   * AICc is computed on the number of OBSERVED response months, not grid rows;
#   * the exogenous placeholder can only ever land on a month that makes no
#     likelihood contribution, and a month with an observed response but an
#     incomplete driver row is WITHHELD rather than imputed;
#   * every forecast is one calendar month ahead, trains strictly earlier, fits
#     its scaling inside the fold and uses only origin-time driver information;
#   * literal persistence is y_{t-1} with no fitted coefficient, and is reported
#     separately from a fitted AR(1);
#   * an RMSE difference is only called an improvement when its interval
#     excludes zero.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ss(ns):
    """§5c state-space helpers, executed on top of the §4/§5 namespace."""
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    namespace = dict(ns)
    namespace.update(warnings=warnings, SARIMAX=SARIMAX,
                     UnobservedComponents=UnobservedComponents)
    exec(compile(_cell("5c. State-space dynamic-regression helpers"), "<ss>", "exec"),
         namespace)
    return namespace


def _ss_series(n=90, seed=3, gaps=(11, 12, 30, 47)):
    """A gapped monthly series with an AR(1) error and one real driver."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2017-01-01", periods=n, freq="MS")
    ang = 2 * np.pi * months.month.to_numpy(dtype=float) / 12.0
    x = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + rng.normal(0, 0.4)
    y = 0.5 + 0.7 * np.sin(ang) + 0.5 * x + u
    s = pd.Series(y, index=months)
    s.iloc[list(gaps)] = np.nan
    return s, pd.Series(x, index=months), months


def test_aicc_uses_observed_months_not_grid_rows(ss):
    """A record that is a quarter missing must not get a quarter-sized penalty."""
    llf, k = -50.0, 8
    tight = ss["aicc_from"](llf, k, 60)
    loose = ss["aicc_from"](llf, k, 120)
    assert tight > loose, "AICc must penalise harder on fewer observed months"
    # The small-sample correction is the whole point of AICc over AIC.
    assert tight == pytest.approx(-2 * llf + 2 * k + 2 * k * (k + 1) / (60 - k - 1))
    # Refuse rather than return a negative-denominator nonsense value.
    assert ss["aicc_from"](llf, 60, 60) == np.inf


def test_only_models_without_a_stochastic_level_take_a_linear_trend(ss):
    """A random-walk level and a linear trend claim the same variation."""
    assert ss["ss_allows_linear_trend"]("sarimax_ar1")
    assert ss["ss_allows_linear_trend"]("sarimax_ar2")
    assert not ss["ss_allows_linear_trend"]("local_level")


def test_local_level_is_refused_a_long_run_multiplier(ss):
    """A unit-root latent state has no long-run multiplier to quote."""
    s, x, months = _ss_series()
    X = ss["fourier_terms"](months, 2)
    X.index = s.index
    res, _ = ss["ss_fit"](s, X, "local_level", cov_type=None, maxiter=120)
    diag = ss["ss_state_diagnostics"](res, "local_level")
    assert diag["stationary_supported"] is False
    assert "unit root" in diag["reason"]


def test_ar_structures_report_roots_and_gate_on_the_interval(ss):
    s, x, months = _ss_series()
    X = ss["fourier_terms"](months, 2)
    X.index = s.index
    res, info = ss["ss_fit"](s, X, "sarimax_ar1", cov_type=None, maxiter=200)
    diag = ss["ss_state_diagnostics"](res, "sarimax_ar1")
    assert diag["roots_outside_unit_circle"] is True
    assert diag["rho_ci_lo"] < diag["rho_sum"] < diag["rho_ci_hi"]
    assert diag["stationary_supported"] == (diag["roots_outside_unit_circle"]
                                            and diag["ci_within_unit_circle"])


def test_standardized_innovations_exist_only_on_observed_months(ss):
    """The filter makes no update on a missing month, so it has no innovation."""
    gaps = (11, 12, 30, 47)
    s, x, months = _ss_series(gaps=gaps)
    X = ss["fourier_terms"](months, 2)
    X.index = s.index
    res, _ = ss["ss_fit"](s, X, "sarimax_ar1", cov_type=None, maxiter=200)
    inn = ss["ss_standardized_innovations"](res, months=months)
    assert inn["std_innovation"].isna().to_numpy()[list(gaps)].all()
    assert inn["std_innovation"].notna().sum() >= len(s.dropna()) - 2


def test_calendar_ljung_box_drops_lags_with_no_calendar_pairs(ss):
    """A lag built from zero genuinely h-months-apart pairs is not evidence."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    x = pd.Series(rng.normal(size=8))
    out = ss["calendar_ljung_box"](x, months=months, nlags=12)
    assert out["df"] == len(out["lags_used"])
    assert all(l >= 8 for l in out["lags_dropped"] if l not in out["lags_used"])
    assert 0.0 <= out["lb_pvalue"] <= 1.0


def test_rmse_difference_bootstrap_is_paired_and_calendar_blocked(ss):
    """Both models are resampled on the SAME months, in contiguous blocks."""
    months = pd.date_range("2018-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1.0, 60)
    out = ss["block_bootstrap_rmse_difference"](months, a, a, n_boot=200, seed=0)
    # Identical error series: the difference is exactly zero in every replicate.
    assert out["ci_lo"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci_hi"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_successful"] > 100
    b = a * 2.0
    worse = ss["block_bootstrap_rmse_difference"](months, b, a, n_boot=200, seed=0)
    assert worse["ci_lo"] > 0, "a uniformly worse model must not straddle zero"


def test_inverse_response_transform_round_trips(ss):
    v = pd.Series([0.02, 0.15, 0.4])
    for how in ("logit", "log", "identity"):
        t, _ = ss["transform_response"](v, how, 1e-4)
        back = ss["inverse_response_transform"](t, how, 1e-4)
        assert back.to_numpy() == pytest.approx(v.to_numpy(), rel=1e-6)


def test_fourier_terms_depend_only_on_the_calendar_month(ss):
    """A fold must be able to build the season for a month it has never seen."""
    a = ss["fourier_terms"](pd.date_range("2019-01-01", periods=12, freq="MS"), 2)
    b = ss["fourier_terms"](pd.date_range("2024-01-01", periods=12, freq="MS"), 2)
    assert a.to_numpy() == pytest.approx(b.to_numpy())


def test_state_space_selection_uses_null_dynamics_only():
    """The dependence structure must not be chosen with driver p-values."""
    src = _cell("19b. Fit the candidate dependence structures WITHOUT any driver")
    assert "ss_null_exog" in src
    # The exogenous block of every candidate is season (+ trend); no driver term
    # is ever put in front of the selection.
    body = src.split("def ss_null_exog")[1].split("def ss_rolling_one_step_null")[0]
    assert "SS_DRIVER_TERMS" not in body
    assert "SS_SEASON_COLS" in body and "SS_TREND_COLS" in body
    assert "aicc" in src and "rmse_one_month_ahead" in src
    assert "LOCKED dependence structure" in src
    for cand in ("sarimax_ar1", "sarimax_ar2", "local_level"):
        cfg = _cell("3a. Where the data comes from")
        assert cand in cfg


def test_matched_null_and_full_models_are_asserted_identical():
    src = _cell("20. The locked structure, with and without the environmental drivers")
    assert "do not use identical response months" in src
    assert "is not a strict extension of the null model" in src
    assert "differs from the null by something other than the drivers" in src
    assert "ss_joint_lr_test" in src and "ss_joint_wald_test" in src


def test_exogenous_placeholder_is_confined_to_missing_response_months():
    src = _cell("19a. The state-space modelling frame")
    assert "placeholder would enter the likelihood" in src
    assert "a placeholder was written on a month with an observed response" in src
    assert "SS_WITHHELD_MONTHS" in src
    # The withheld months are recorded, not silently imputed.
    assert "withheld from the likelihood" in src


def test_one_month_ahead_design_never_reaches_forward():
    src = _cell("21b. Expanding-window, one-calendar-month-ahead rolling origin")
    assert "target is not exactly one calendar month after the origin" in src
    assert "training reaches the target month" in src
    assert "SS_MIN_TRAIN_MONTHS" in src
    # Scaling is fitted inside the fold, from the training months only.
    assert "_tr[_scale_cols].mean()" in src
    assert "_tr[_scale_cols].std(ddof=1)" in src
    # The globally z-scored §11 columns must not be reused for the forecast.
    assert "SS_FC_TERMS" in src and "DRIVER_TERMS" not in src


def test_forecast_drivers_are_lagged_to_origin_time():
    src = _cell("21a. Build the FORECAST driver set")
    assert "SS_FORECAST_MIN_LAG" in src
    assert "would use information from the target month or later" in src
    assert "SS_FORECAST_EXOG_OVERRIDE" in src
    # The a-priori (lag-0) specification survives for the association inference.
    assert "a-priori (lag-0) specification is retained unchanged" in src


def test_literal_persistence_is_not_a_fitted_regression():
    src = _cell("21b. Expanding-window, one-calendar-month-ahead rolling origin")
    assert "literal persistence (y_{t-1}, unfitted)" in src
    assert "fitted AR(1)" in src
    assert "NO fitted coefficient" in src
    val = _cell("26a. Validation")
    assert "literal persistence is y_{t-1} itself, with no fitted coefficient" in val
    assert "reported separately from literal persistence" in val


def test_seasonal_naive_is_exactly_twelve_calendar_months():
    src = _cell("21b. Expanding-window, one-calendar-month-ahead rolling origin")
    assert "seasonal naive (y_{t-12})" in src
    assert "not exactly 12 calendar months earlier" in src
    code = [ln for ln in src.split("\n") if not ln.strip().startswith("#")]
    assert not any("shift(12)" in ln for ln in code)


def test_rmse_improvement_wording_is_gated_on_the_interval():
    src = _cell("22. Paired loss differences and a calendar-aware bootstrap interval")
    assert "the point estimate was lower" in src
    assert "but the improvement was uncertain" in src
    assert "interval_excludes_zero" in src
    assert "block_bootstrap_rmse_difference" in src
    # Both scales are reported.
    assert "logit modelling scale" in src
    assert "raw WH cover (back-transformed)" in src


def test_lr_bootstrap_cannot_stop_the_notebook():
    src = _cell("23. Parametric bootstrap under the matched null")
    assert "SS_RUN_LR_BOOTSTRAP" in src
    assert "n_replicates_successful" in src
    assert "except Exception" in src
    assert "the rest of the notebook is unaffected" in src.lower()
    cfg = _cell("3a. Where the data comes from")
    assert "SS_LR_BOOTSTRAP_N = 399" in cfg          # inside the 300-500 range


def test_spline_section_fits_one_driver_at_a_time_within_a_df_ceiling():
    """25 columns on 64 months is an interpolation, not a nonlinear result."""
    cfg = _cell("3a. Where the data comes from")
    assert "SPLINE_DF_MAX = 3" in cfg
    assert "SPLINE_ONE_DRIVER_AT_A_TIME = True" in cfg
    src = _cell("14. Natural-spline GLM")
    assert "SPLINE_ONE_DRIVER_AT_A_TIME" in src
    assert "design_info" in src and "build_design_matrices" in src
    assert "SPLINE_MIN_ROWS_PER_COLUMN" in src
    # Centred, so the smooth carries no intercept.
    assert "centred: no intercept inside the smooth" in src
    oos = _cell("14b. Out-of-sample check on the non-linear shapes")
    assert "supported_out_of_sample" in oos
    assert "exploratory" in oos
    # A point estimate does not promote a curve: the same calendar-aware
    # moving-block interval §22 uses has to exclude zero.
    assert "block_bootstrap_rmse_difference" in oos
    assert '_b["ci_hi"] < 0' in oos


def test_three_month_window_skill_is_demoted_and_the_nine_percent_claim_withdrawn():
    src = _cell("16. Rolling-origin out-of-sample skill")
    assert "SENSITIVITY" in src
    assert "NOT a headline result" in src
    assert "IMPROVE out-of-sample RMSE by" not in src
    export = _cell("26. Export tables and a run manifest")
    assert "withdrawn" in export


def test_synthesis_answers_the_five_questions_and_names_its_sources():
    src = _cell("25a. The five questions")
    assert "SYNTHESIS_ANSWERS" in src
    for q in ("Which dependence structure", "jointly improve FIT",
              "one-calendar-month-ahead prediction", "Is any individual driver robust",
              "How much uncertainty remains"):
        assert q in src
    assert "PLAIN RESULT" in src
    table = _cell("25. Synthesis table")
    assert "coef_S1" in table and "coef_M3" in table
    assert "static only — not confirmed under persistence" in table


def test_effective_sample_size_is_labelled_approximate():
    src = _cell("8. Response series diagnostics")
    assert "APPROXIMATE effective sample size" in src
    assert "cannot" in src and "independent information" in src


def test_state_space_exports_and_validation_are_present():
    src = _cell("26. Export tables and a run manifest")
    for name in ("statespace_candidate_dynamics", "statespace_coefficients",
                 "statespace_joint_driver_block_test",
                 "statespace_rolling_origin_fold_audit",
                 "statespace_one_month_predictions",
                 "statespace_skill_common_sample",
                 "statespace_rmse_difference_bootstrap",
                 "statespace_lr_bootstrap",
                 "statespace_innovation_diagnostics",
                 "statespace_robustness_loyo",
                 "synthesis_answers", "validation_assertions"):
        assert f'"{name}"' in src, f"{name} is not exported"
    val = _cell("26a. Validation")
    for check in ("identical response months", "identical target months",
                  "exactly one calendar month ahead",
                  "standardisation was fitted on training months only",
                  "exactly t-12 calendar months",
                  "placeholder exogenous values occur only where the response is missing",
                  "real and synthetic outputs are distinguishable"):
        assert check in val, f"missing validation check: {check}"


def test_state_space_robustness_covers_the_required_sweeps():
    loyo = _cell("24a. Leave-one-year-out on the state-space driver model")
    assert "setting its responses missing" in loyo or "responses missing" in loyo
    var = _cell("24b. Alternative transforms and alternative dependence structures")
    assert "SS_ROBUST_TRANSFORMS" in var
    assert "SS_CANDIDATE_STRUCTURES" in var
    assert "never merged with §12/§18" in var


def test_synthetic_self_test_series_has_calendar_gaps(ns):
    """A gapless self-test never exercises the missing-month machinery."""
    monthly, truth = ns["make_synthetic_monthly"](108, seed=42, missing_fraction=0.2)
    assert truth["n_months_forced_missing"] > 10
    below = monthly["coverage_fraction"] < 0.9
    assert int(below.sum()) == truth["n_months_forced_missing"]
    # A month whose partner exactly 12 calendar months later survives the filter,
    # so seasonal-naive genuinely goes unavailable somewhere.
    gapped = set(monthly.loc[below, "month"])
    kept = set(monthly.loc[~below, "month"])
    assert any((m + pd.DateOffset(months=12)) in kept for m in gapped)
