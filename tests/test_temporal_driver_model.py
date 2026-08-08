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
* rolling-origin folds never train on the future;
* Shapley R^2 values sum to the full-model R^2 and split shared variance
  between collinear drivers instead of awarding it to whichever entered first;
* the synthetic self-test recovers the effects it was built with, and does NOT
  recover the decoys.
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
    exec(compile(_cell("5. Autocorrelation-robust inference helpers"), "<inf>", "exec"),
         namespace)
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


def test_moving_block_bootstrap_keeps_the_sample_size_and_uses_blocks(ns):
    rng = np.random.default_rng(1)
    idx = next(ns["moving_block_bootstrap_indices"](40, block=5, rng=rng, n_boot=1))
    assert len(idx) == 40
    # With blocks of 5 the resample must contain runs of consecutive indices.
    runs = np.sum(np.diff(idx) == 1)
    assert runs >= 20


def test_bootstrap_brackets_the_point_estimate(ns):
    """The bootstrap distribution centres on the SAMPLE estimate, not on truth."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=120)
    y = 2.0 * x + rng.normal(scale=0.3, size=120)
    X = pd.DataFrame({"x": x})
    point = float(sm.OLS(y, sm.add_constant(X)).fit().params["x"])
    draws = ns["bootstrap_coefficients"](y, X, n_boot=300, seed=3)
    summ = ns["bootstrap_summary"](draws, terms=["x"]).iloc[0]
    assert summ["boot_ci_lo"] < point < summ["boot_ci_hi"]
    assert summ["boot_median"] == pytest.approx(point, abs=0.05)
    assert summ["boot_median"] == pytest.approx(2.0, abs=0.15)
    assert summ["boot_sign_stability"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rolling-origin cross-validation
# ---------------------------------------------------------------------------

def test_rolling_origin_folds_never_train_on_the_future(ns):
    folds = ns["rolling_origin_folds"](100, n_folds=8, horizon=3, min_train=24)
    assert folds
    for train, test in folds:
        assert train.max() < test.min()
        assert len(train) >= 24
        assert len(test) <= 3


def test_rolling_origin_folds_expand_and_do_not_overlap(ns):
    folds = ns["rolling_origin_folds"](100, n_folds=8, horizon=3, min_train=24)
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)
    seen = np.concatenate([te for _, te in folds])
    assert len(seen) == len(set(seen.tolist()))


def test_rolling_origin_refuses_folds_it_cannot_train(ns):
    assert ns["rolling_origin_folds"](20, n_folds=8, horizon=3, min_train=24) == []


def test_cv_scores_beats_the_mean_on_a_learnable_series(ns):
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.2, size=n)
    folds = ns["rolling_origin_folds"](n, n_folds=6, horizon=3, min_train=24)
    sc, detail = ns["cv_scores"](y, pd.DataFrame({"x": x}), folds)
    assert sc["n_folds"] == len(folds)
    assert sc["r2_oos"] > 0.8
    assert len(detail) == sc["n_test"]


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
    synth = _cell("19. Synthesis table")
    # Proxies must never appear in the ranked driver table's term list.
    assert "PROXY_TERMS" not in synth


def test_notebook_uses_rolling_origin_not_random_cv():
    for marker in ("13. Elastic net", "16. Rolling-origin out-of-sample skill"):
        src = _cell(marker)
        assert "rolling_origin_folds" in src
        assert "KFold" not in src and "train_test_split" not in src


def test_notebook_scores_skill_against_persistence_and_seasonal_baselines():
    src = _cell("16. Rolling-origin out-of-sample skill")
    for baseline in ("mean baseline", "persistence (y_lag1)", "seasonal-naive"):
        assert baseline in src


def test_synthesis_verdicts_cover_the_documented_categories():
    src = _cell("19. Synthesis table")
    for verdict in ("robust", "suggestive", "sign contradicts mechanism",
                    "not separable from season", "no evidence"):
        assert f'"{verdict}"' in src


def test_verdicts_ignore_selection_frequency_when_the_net_is_not_sparse():
    """A near-ridge fit selects everything; that must not read as robustness."""
    src = _cell("19. Synthesis table")
    assert "ENET_SPARSITY_OK" in src
    enet = _cell("13. Elastic net")
    assert "ENET_SPARSITY_OK" in enet


def test_notebook_regenerates_from_its_builder():
    """The builder script is the editable source of the notebook."""
    assert (REPO / "build_temporal_model_nb.py").exists()
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 30
