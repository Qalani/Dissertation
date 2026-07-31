"""End-to-end execution of the predictive-ML evaluation cells against a synthetic panel.

The static tests in ``test_predictive_ml_validity.py`` pin what the notebook *says*;
these tests **run** the §15e (persistence-led evaluation), §15e2 (occurrence-threshold
sweep) and §15e3 (24-vs-36-month history) cells straight out of the notebook JSON, on
a small synthetic panel, with cheap scikit-learn stand-ins for the LightGBM generics
that §15b defines. That catches wiring errors — a renamed column, an unguarded merge,
a mis-scoped variable — that a text search cannot.

No Drive, no Earth Engine, no LightGBM. Outputs are written to a tmp directory.
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

N_CELLS = 24
N_MONTHS = 60
CELL_SIZE_M = 500
CELL_AREA_M2 = float(CELL_SIZE_M) ** 2


def _cell(marker):
    hits = [s for s in ("".join(c["source"])
                        for c in json.loads(NB_PATH.read_text())["cells"]
                        if c["cell_type"] == "code") if marker in s]
    assert hits, f"no code cell contains {marker!r}"
    return hits[0]


# ---------------------------------------------------------------------
# Synthetic panel + cheap stand-ins for the §15b LightGBM generics
# ---------------------------------------------------------------------
def _synthetic_ml_df(seed=0):
    """A small hard-class panel with an autoregressive signal and calendar lags."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2018-01-01", periods=N_MONTHS, freq="MS")
    gid = np.repeat(np.arange(N_CELLS), N_MONTHS)
    mo = pd.DatetimeIndex(np.tile(months.values, N_CELLS))

    depth = np.repeat(rng.uniform(1, 12, N_CELLS), N_MONTHS)
    season = np.sin(2 * np.pi * mo.month.to_numpy() / 12)
    level = 0.03 + 0.02 * season + rng.normal(0, 0.004, len(gid))
    cover = np.clip(level * np.exp(-depth / 8.0)
                    + rng.gamma(0.4, 0.01, len(gid)), 0, 1)
    # a fifth of the cell-months are genuinely empty
    cover[rng.random(len(gid)) < 0.35] = 0.0

    df = pd.DataFrame({
        "grid_id": gid, "month": mo, "month_num": mo.month.to_numpy(),
        "time_rank": np.tile(np.arange(N_MONTHS), N_CELLS),
        "wh_cover": cover,
        "valid_area_m2": CELL_AREA_M2,
        "wh_area_ha": cover * CELL_AREA_M2 / 1e4,
        "depth_m": depth,
        "wave_exposure_idx": np.repeat(rng.uniform(0, 1, N_CELLS), N_MONTHS),
        "x_km": np.repeat(rng.uniform(0, 20, N_CELLS), N_MONTHS),
        "y_km": np.repeat(rng.uniform(0, 20, N_CELLS), N_MONTHS),
        "rain_chirps_30d_mm_lag1": rng.gamma(2, 30, len(gid)),
        "wind_speed_ms_lag1": rng.gamma(3, 1.2, len(gid)),
    }).sort_values(["grid_id", "month"]).reset_index(drop=True)
    df["fold_sp"] = (df["grid_id"] % 3) + 1
    return df


def _add_lags(df, shared, presence_threshold=0.02):
    df = df.copy()
    df["wh_present"] = (df["wh_cover"] >= presence_threshold).astype(int)
    df = shared["calendar_lag"](df, ["wh_cover", "wh_present"], periods=1)
    df["wh_cover_neigh_lag1"] = df["wh_cover_lag1"].fillna(0.0) * 0.8
    df["wh_present_neigh_lag1"] = df["wh_present_lag1"].fillna(0.0) * 0.8
    return df


class _Clf:
    """A logistic stand-in for the §15b LightGBM classifier."""

    def __init__(self, model, prior):
        self.model, self.prior = model, prior

    def predict_proba(self, X):
        if self.model is None:
            p = np.full(len(X), self.prior)
        else:
            p = self.model.predict_proba(np.nan_to_num(X.to_numpy(float)))[:, 1]
        return np.column_stack([1 - p, p])


def _install_model_stubs(ns):
    """Cheap replacements for the §15b generics the evaluation cells call."""
    from sklearn.linear_model import LogisticRegression, Ridge

    def _fit_reg(X, y, w=None):
        m = Ridge(alpha=1.0)
        m.fit(np.nan_to_num(X.to_numpy(float)), np.asarray(y, float),
              sample_weight=None if w is None else np.asarray(w, float))
        return m

    def _predict_hurdle(Xtr, Xte, tr, w_tr=None):
        y = tr["wh_present"].astype(int).to_numpy()
        if np.unique(y).size < 2:
            clf = _Clf(None, float(y.mean()) if len(y) else 0.0)
        else:
            m = LogisticRegression(max_iter=200)
            m.fit(np.nan_to_num(Xtr.to_numpy(float)), y)
            clf = _Clf(m, float(y.mean()))
        p_te = clf.predict_proba(Xte)[:, 1]
        p_tr = clf.predict_proba(Xtr)[:, 1]
        pos = y == 1
        if pos.sum() >= 10:
            reg = _fit_reg(Xtr[pos], tr.loc[pos, "wh_cover"])
            c_te = np.clip(reg.predict(np.nan_to_num(Xte.to_numpy(float))), 0, 1)
            c_tr = np.clip(reg.predict(np.nan_to_num(Xtr.to_numpy(float))), 0, 1)
        else:
            reg = None
            c_te = np.full(len(Xte), float(tr["wh_cover"].mean()))
            c_tr = np.full(len(Xtr), float(tr["wh_cover"].mean()))
        return (np.clip(p_te * c_te, 0, 1), np.clip(p_tr * c_tr, 0, 1), p_te, clf, reg)

    def _predict_tweedie(Xtr, Xte, tr, w_tr=None):
        reg = _fit_reg(Xtr, tr["wh_cover"], w_tr)
        return (np.clip(reg.predict(np.nan_to_num(Xte.to_numpy(float))), 0, 1),
                np.clip(reg.predict(np.nan_to_num(Xtr.to_numpy(float))), 0, 1), reg)

    def _metrics(obs, pred):
        pred = np.clip(np.asarray(pred, float), 0, 1)
        obs = np.asarray(obs, float)
        return (ns["safe_spearman"](obs, pred),
                float(np.sqrt(np.mean((obs - pred) ** 2))),
                float(np.mean(np.abs(obs - pred))),
                float(pred.sum() / obs.sum()) if obs.sum() > 0 else np.nan)

    def _extra_skill(te, pred, clim_pred, cell_mean_tr=None):
        obs = te["wh_cover"].to_numpy(float)
        pred = np.clip(np.asarray(pred, float), 0, 1)
        within = [ns["safe_spearman"](obs[m], pred[m])
                  for m in (te["month"].to_numpy() == mm
                            for mm in pd.unique(te["month"]))]
        within = [v for v in within if np.isfinite(v)]
        return (ns["msss_vs_reference"](obs, pred, clim_pred),
                float(np.mean(within)) if within else np.nan)

    def _area_metrics(te, pred):
        frame = pd.DataFrame({"month": te["month"].to_numpy(),
                              "obs": te["wh_cover"].to_numpy(float),
                              "pred": np.clip(np.asarray(pred, float), 0, 1),
                              "valid_area_m2": te["valid_area_m2"].to_numpy(float)})
        _pm, s = ns["valid_area_month_metrics"](frame)
        return {k: s[k] for k in ("area_mae_ha", "area_rmse_ha", "area_bias_ha",
                                  "area_bias_pct")}

    class _Reg:
        """A ridge stand-in with the fit/predict surface §15i expects."""

        def __init__(self):
            self.m = Ridge(alpha=1.0)

        def fit(self, X, y, w=None):
            self.m.fit(np.nan_to_num(np.asarray(X, float)), np.asarray(y, float),
                       sample_weight=None if w is None else np.asarray(w, float))
            return self

        def predict(self, X):
            return self.m.predict(np.nan_to_num(np.asarray(X, float)))

    ns.update(_predict_hurdle=_predict_hurdle, _predict_tweedie=_predict_tweedie,
              _metrics=_metrics, _extra_skill=_extra_skill, _area_metrics=_area_metrics,
              _make_regressor=lambda objective="regression": _Reg(),
              _fit=lambda m, X, y, w=None, t=None: m.fit(X, y, w))


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """Run §15e, §15e2 and §15e3 out of the notebook on the synthetic panel."""
    out = tmp_path_factory.mktemp("nf_outputs")
    ns: dict = {}
    exec(compile(_cell(SHARED_MARKER), "<shared>", "exec"), ns)

    ml_df = _add_lags(_synthetic_ml_df(), ns)
    features = ["depth_m", "wave_exposure_idx", "x_km", "y_km",
                "rain_chirps_30d_mm_lag1", "wind_speed_ms_lag1",
                "wh_cover_lag1", "wh_present_lag1", "wh_cover_neigh_lag1",
                "wh_present_neigh_lag1", "month_num"]

    ns.update(
        RUN_ML_WORKHORSE=True, CELL_SIZE_M=CELL_SIZE_M, OUTPUT_DIR=out,
        ML_RANDOM_STATE=42, ML_MIN_FOLD_TRAIN_ROWS=50, ML_SPATIAL_BUFFER_KM=0.0,
        ML_TEMPORAL_EMBARGO_MONTHS=1, ML_ADD_CELL_CLIMATOLOGY=False,
        ML_ADD_HIER_PRIOR=False, ML_IMPORTANCE_PERMUTE_BLOCKS=True,
        CV_TEMPORAL_MIN_TRAIN_MONTHS=24, CV_TEMPORAL_MIN_ELAPSED_MONTHS=24,
        CV_TEMPORAL_SENSITIVITY_MIN_TRAIN_MONTHS=36,
        CV_TEMPORAL_SENSITIVITY_MIN_ELAPSED_MONTHS=36,
        PRESENCE_COVER_THRESHOLD=0.02, PRESENCE_AREA_HA_THRESHOLD=None,
        PRESENCE_COVER_THRESHOLD_GRID=(0.005, 0.01, 0.02),
        PRESENCE_AREA_HA_THRESHOLD_GRID=(0.125, 0.25, 0.5),
        PRESENCE_PRIMARY_DEFINITION="cover_0.02",
        PRESENCE_THRESHOLD_SELECTION_SOURCE="predeclared",
        ml_df=ml_df, grid_neighbours=None, display=lambda *a, **k: None,
    )
    exec(compile(_cell(ORIGIN_MARKER), "<origins>", "exec"), ns)
    _install_model_stubs(ns)

    ns["PRESENCE_DEFINITIONS"] = ns["build_presence_definitions"](
        ns["PRESENCE_COVER_THRESHOLD_GRID"], ns["PRESENCE_AREA_HA_THRESHOLD_GRID"],
        cell_size_m=CELL_SIZE_M)
    ns["PRIMARY_PRESENCE_DEFINITION"] = ns["resolve_primary_presence_definition"](
        ns["PRESENCE_COVER_THRESHOLD"], ns["PRESENCE_AREA_HA_THRESHOLD"],
        cell_size_m=CELL_SIZE_M)
    ns["nowcast_feature_cols"] = list(features)
    ns["forecast_feature_cols"] = list(features)
    ns["nf_contemporaneous_set"] = set()

    exec(compile(_cell(EVAL_MARKER), "<eval>", "exec"), ns)
    exec(compile(_cell(SWEEP_MARKER), "<sweep>", "exec"), ns)
    exec(compile(_cell(HISTORY_MARKER), "<history>", "exec"), ns)
    exec(compile(_cell(IMPORTANCE_MARKER), "<importance>", "exec"), ns)
    return ns


# ---------------------------------------------------------------------
# §15e — persistence-led evaluation
# ---------------------------------------------------------------------
def test_evaluation_scores_origins_with_the_declared_history(evaluated):
    summary = evaluated["nf_eval_origin_summary"]
    assert summary["n_origins"] > 0
    assert summary["min_train_months_used"] >= 24
    assert summary["min_elapsed_used"] >= 24
    assert summary["first_origin"] < summary["last_origin"]


def test_out_of_fold_predictions_carry_the_persistence_forecast(evaluated):
    oof = evaluated["nf_oof"]
    assert "persistence_pred" in oof.columns
    assert oof["persistence_pred"].notna().all()
    pers = oof[oof["model"] == "persistence"]
    np.testing.assert_allclose(pers["pred"].to_numpy(), pers["persistence_pred"].to_numpy())


def test_persistence_scores_zero_skill_against_itself(evaluated):
    skill = evaluated["nf_persistence_skill"]
    rows = skill[skill["model"] == "persistence"]
    assert len(rows)
    np.testing.assert_allclose(rows["msss_persistence"].to_numpy(), 0.0, atol=1e-12)


def test_persistence_intervals_bracket_the_point_estimate(evaluated):
    ci = evaluated["nf_persistence_ci"]
    assert len(ci) and (ci["n_blocks"] > 1).all()
    finite = ci.dropna(subset=["ci_lo", "ci_hi"])
    assert len(finite)
    assert (finite["ci_lo"] <= finite["msss_persistence"] + 1e-9).all()
    assert (finite["msss_persistence"] <= finite["ci_hi"] + 1e-9).all()
    # Persistence scores exactly zero in every resample, so its interval is a point;
    # every other model's interval has width.
    others = finite[finite["model"] != "persistence"]
    assert len(others) and (others["ci_hi"] > others["ci_lo"]).all()
    assert (finite.loc[finite["model"] == "persistence",
                       ["ci_lo", "ci_hi"]].abs().to_numpy() < 1e-12).all()


def test_win_rates_are_reported_per_origin_and_per_window(evaluated):
    wins = evaluated["nf_persistence_wins"]
    for col in ("origin_win_rate_vs_persistence", "window_win_rate_vs_persistence"):
        assert col in wins.columns, f"{col} missing from the win-rate table"
    rates = wins["origin_win_rate_vs_persistence"].dropna()
    assert len(rates) and rates.between(0, 1).all()
    assert (wins.loc[wins["model"] == "persistence",
                     "origin_win_rate_vs_persistence"] == 0).all()


def test_fold_level_rows_carry_the_persistence_metrics(evaluated):
    cv = evaluated["nf_cv"]
    for col in ("msss_persistence", "beats_persistence", "change_corr",
                "change_sign_hit"):
        assert col in cv.columns
    temporal = cv[cv["kind"] == "temporal"]
    assert temporal["msss_persistence"].notna().any()


def test_the_zero_baseline_still_appears_everywhere_skill_is_scored(evaluated):
    assert "zero" in set(evaluated["nf_cv"]["model"])
    assert "zero" in set(evaluated["nf_persistence_skill"]["model"])
    assert "zero" in set(evaluated["nf_area_summary"]["model"])


def test_the_temporal_residual_diagnostic_is_still_computed_and_climatology_is_na(evaluated):
    resid = evaluated["nf_temporal_resid"]
    assert "temporal_resid_corr" in resid.columns
    clim = resid[resid["model"] == "climatology_cell"]
    assert len(clim) and clim["temporal_resid_corr"].isna().all()


def test_the_evaluation_exported_the_persistence_tables(evaluated):
    names = [p.name for p in evaluated["OUTPUT_DIR"].iterdir()]
    assert any(n.startswith("nf_persistence_skill_") for n in names)
    assert any(n.startswith("nf_persistence_skill_bootstrap_ci_") for n in names)
    assert any(n.startswith("nf_persistence_win_rates_") for n in names)
    assert any(n.startswith("nf_forecast_origin_inventory_") for n in names)
    assert any("temporal_residual_corr_DIAGNOSTIC" in n for n in names)


# ---------------------------------------------------------------------
# §15e2 — occurrence-threshold sweep
# ---------------------------------------------------------------------
def test_every_definition_ran_on_the_same_origins(evaluated):
    folds = evaluated["nf_presence_sweep_folds"]
    per_def = folds.groupby("definition")["origin"].apply(lambda s: tuple(sorted(set(s))))
    assert per_def.nunique() == 1, "the definitions did not share identical folds"
    assert len(per_def) == 6


def test_the_sweep_reports_prevalence_and_the_share_of_wh_area(evaluated):
    sweep = evaluated["nf_presence_sweep"].set_index("definition")
    assert (sweep["prevalence"].between(0, 1)).all()
    assert (sweep["share_of_total_wh_area"].between(0, 1)).all()
    # a stricter cover threshold cannot label more cells
    assert sweep.loc["cover_0.02", "prevalence"] <= sweep.loc["cover_0.01", "prevalence"]
    assert sweep.loc["cover_0.01", "prevalence"] <= sweep.loc["cover_0.005", "prevalence"]


def test_the_sweep_reports_the_full_metric_set(evaluated):
    cols = set(evaluated["nf_presence_sweep"].columns)
    required = {"presence_auc", "presence_ap", "presence_brier",
                "colonisation_precision", "colonisation_recall", "colonisation_f1",
                "hurdle_gbm_rmse", "hurdle_gbm_mae", "hurdle_gbm_msss_clim",
                "hurdle_gbm_msss_persistence", "hurdle_gbm_area_mae_ha",
                "hurdle_gbm_area_bias_ha"}
    assert required <= cols, f"missing: {sorted(required - cols)}"


def test_the_predeclared_definition_is_flagged_and_is_the_only_one(evaluated):
    sweep = evaluated["nf_presence_sweep"]
    primary = sweep[sweep["is_primary_predeclared"]]
    assert len(primary) == 1
    assert primary["definition"].iloc[0] == "cover_0.02"


def test_the_probability_cut_comes_from_training_rows_only(evaluated):
    folds = evaluated["nf_presence_sweep_folds"]
    assert folds["train_only_prob_threshold"].between(0, 1).all()
    # It is a real, definition- and fold-specific quantity rather than a constant...
    assert folds.groupby("definition")["train_only_prob_threshold"].mean().nunique() > 1
    assert folds.groupby("origin")["train_only_prob_threshold"].mean().nunique() > 1
    # ... and it is not the cut that would have matched the HELD-OUT prevalence, which
    # is what selecting on the test month would have produced.
    ns_defs = {d["name"]: d for d in evaluated["PRESENCE_DEFINITIONS"]}
    ml_df = evaluated["ml_df"]
    apply_def = evaluated["apply_presence_definition"]
    mismatch = 0
    for row in folds.itertuples(index=False):
        te = ml_df[ml_df["month"] == pd.Timestamp(row.origin + "-01")]
        test_prevalence = float(apply_def(te, ns_defs[row.definition]).mean())
        if abs(row.train_only_prob_threshold - (1 - test_prevalence)) > 1e-6:
            mismatch += 1
    assert mismatch > 0, "the cut looks like it was matched to held-out prevalence"


def test_the_fixed_reference_cover_model_is_constant_across_definitions(evaluated):
    sweep = evaluated["nf_presence_sweep"]
    assert sweep["tweedie_gbm_rmse"].std() == pytest.approx(0.0, abs=1e-12)


def test_the_sweep_exported_its_tables(evaluated):
    names = [p.name for p in evaluated["OUTPUT_DIR"].iterdir()]
    assert any(n.startswith("nf_presence_definition_sweep_") for n in names)
    assert any(n.startswith("nf_presence_definition_sweep_perfold_") for n in names)


# ---------------------------------------------------------------------
# §15e3 — 24 vs 36 months of training history
# ---------------------------------------------------------------------
def test_the_history_inventory_reports_both_specifications(evaluated):
    inv = evaluated["nf_history_inventory"].set_index("spec")
    assert {"primary_24m", "sensitivity_36m"} <= set(inv.index)
    assert inv.loc["sensitivity_36m", "n_origins"] < inv.loc["primary_24m", "n_origins"]
    for spec in ("primary_24m", "sensitivity_36m"):
        assert inv.loc[spec, "first_origin"] and inv.loc[spec, "last_origin"]


def test_the_specifications_are_compared_on_three_origin_sets(evaluated):
    sens = evaluated["nf_history_sensitivity"]
    sets = set(sens["origin_set"])
    assert any(s.startswith("all_primary_24m") for s in sets)
    assert any(s.startswith("common_") for s in sets)
    assert any(s.endswith("_only") for s in sets)
    assert (sens["n_origins"] > 0).all()


def test_the_common_origin_comparison_is_exact(evaluated):
    """The 36-month run and the 24-month run restricted to common origins coincide."""
    sens = evaluated["nf_history_sensitivity"]
    common = sens[sens["origin_set"].str.startswith("common_")]
    assert len(common)
    # the §15e3 cell asserts this itself; re-check the reported numbers are usable
    assert common["msss_persistence"].notna().any()
    assert common["rmse"].gt(0).all()
    # the win rate must be populated for every model, not silently NaN because the
    # persistence reference was grouped out of the comparison frame
    assert sens["origin_win_rate_vs_persistence"].notna().all()
    assert sens["origin_win_rate_vs_persistence"].between(0, 1).all()
    assert (sens.loc[sens["model"] == "persistence",
                     "origin_win_rate_vs_persistence"] == 0).all()


def test_the_history_sensitivity_exported_its_tables(evaluated):
    names = [p.name for p in evaluated["OUTPUT_DIR"].iterdir()]
    assert any(n.startswith("nf_history_sensitivity_24_vs_36_") for n in names)
    assert any(n.startswith("nf_origin_inventory_by_spec_") for n in names)


# ---------------------------------------------------------------------
# §15i — importance across the series, with groups and intervals
# ---------------------------------------------------------------------
def test_importance_uses_origins_spanning_the_series(evaluated):
    baseline = evaluated["nf_perm_baseline"]
    assert len(baseline) > 4, "importance is still averaged over only a handful of origins"
    origins = sorted(baseline["origin"])
    inventory = evaluated["nf_history_inventory"].set_index("spec")
    assert origins[0] == inventory.loc["primary_24m", "first_origin"]
    assert origins[-1] == inventory.loc["primary_24m", "last_origin"]


def test_per_feature_importance_carries_spread_and_intervals(evaluated):
    imp = evaluated["nf_permutation_importance"]
    for col in ("rmse_increase", "rmse_increase_sd", "msss_drop",
                "area_mae_increase_ha", "group", "n_origins",
                "rmse_increase_ci_lo", "rmse_increase_ci_hi"):
        assert col in imp.columns, f"per-feature importance is missing {col}"
    assert (imp["n_origins"] > 4).all()
    ok = imp.dropna(subset=["rmse_increase_ci_lo", "rmse_increase_ci_hi"])
    assert len(ok)
    assert (ok["rmse_increase_ci_lo"] <= ok["rmse_increase"] + 1e-9).all()
    assert (ok["rmse_increase"] <= ok["rmse_increase_ci_hi"] + 1e-9).all()


def test_grouped_importance_covers_the_correlated_blocks(evaluated):
    grouped = evaluated["nf_grouped_importance"].set_index("group")
    assert {"wh_lag_memory", "neighbour_advection", "wind", "rainfall",
            "static_habitat", "calendar"} <= set(grouped.index)
    assert (grouped["n_features"] >= 1).all()
    assert "rmse_increase_ci_lo" in grouped.columns


def test_grouping_covers_every_feature_exactly_once(evaluated):
    grouped = evaluated["nf_grouped_importance"]
    per_feature = evaluated["nf_permutation_importance"]
    assert grouped["n_features"].sum() == per_feature["feature"].nunique()
    assert set(grouped["group"]) == set(per_feature["group"])


def test_a_group_the_model_really_uses_scores_positively(evaluated):
    """Destroying the block that generates the synthetic signal must hurt skill.

    Cover is built as a function of depth in the fixture, so permuting the whole
    static-habitat block across cells has to raise held-out RMSE. Grouped and
    per-feature importances are on the same scale (RMSE increase), so they are
    directly comparable.
    """
    grouped = evaluated["nf_grouped_importance"].set_index("group")
    assert grouped.loc["static_habitat", "rmse_increase"] > 0
    assert grouped["rmse_increase"].idxmax() == "static_habitat"


def test_importance_tables_were_exported(evaluated):
    names = [p.name for p in evaluated["OUTPUT_DIR"].iterdir()]
    assert any(n.startswith("nf_permutation_importance_") for n in names)
    assert any(n.startswith("nf_permutation_importance_perorigin_") for n in names)
    assert any(n.startswith("nf_grouped_importance_") for n in names)
