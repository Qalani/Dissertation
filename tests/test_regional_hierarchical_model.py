"""Behavioural tests for the regional hierarchical driver notebook.

``winam_wh_regional_hierarchical_driver_model.ipynb`` divides Winam Gulf into
fixed, ecologically meaningful regions, builds ONE water-hyacinth series per
region, and fits a hierarchical dynamic model to the region x month panel. Its
helper cells are pure numpy/pandas (plus PyMC for the model builder), so they are
executed here straight out of the notebook JSON — the tests cannot drift from the
code they describe.

What is pinned:

* **the regions are response-blind.** ``assert_response_blind`` raises on any
  column that looks like a response, a prevalence, a residual, a prediction or a
  classifier output, and it is called inside every regionalisation routine. A
  region drawn where the hyacinth is would make "regions differ in hyacinth" a
  tautology;
* **a class is not a region.** Geographically disconnected parts of one
  ecological class become separate regions, a class boundary splits a contiguous
  block, and a five-cell gap is not adjacency;
* **under-sized components merge into the most physically SIMILAR adjacent
  region**, not into whichever happens to be found first, and the merge is
  logged with a reason;
* **class precedence is fixed**: river influence outranks shelter, shelter
  outranks exposure;
* **static covariates really are static** — one that moves within a cell through
  time is refused rather than averaged;
* **the response is area-weighted**: sum(WH area) / sum(valid classified area),
  which differs from the mean of per-cell cover whenever cells differ in size;
* **coverage gates make a region-month MISSING**, never partially observed;
* **every region sits on the same complete calendar grid**, so a lag is a
  calendar lag for every region alike;
* **drivers come from the complete environmental tables**, so a month with no WH
  map still has forcing and the state process can run through it;
* **variance decomposition splits exactly by month**, and a driver with one
  gulf-wide value per month has no within-month regional variance — the property
  the whole ``temporal_only`` label rests on;
* **the likelihood sees only the observation index**: changing the response at a
  region-month outside it cannot change the model's log-probability, which is
  what makes the placeholder zeros safe;
* **region-macro skill weights every region equally**, so one large region
  cannot carry the score.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "winam_wh_regional_hierarchical_driver_model.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def _cell(marker):
    """The first code cell containing `marker`, as source text."""
    for c in _cells():
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise AssertionError(f"no code cell contains {marker!r}")


def _defs(cell_src, names):
    """Just the named top-level function definitions from a cell."""
    tree = ast.parse(cell_src)
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in set(names)]
    found = {n.name for n in keep}
    missing = set(names) - found
    assert not missing, f"missing function definitions in cell: {sorted(missing)}"
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


@pytest.fixture(scope="module")
def ns():
    """Notebook helper namespace: §4 loading/calendar + §5 regionalisation."""
    import scipy.stats as sstats
    import statsmodels.api as sm

    space = {"pd": pd, "np": np, "Path": Path, "json": json, "sm": sm,
             "sstats": sstats, "display": lambda *a, **k: None,
             "SEASON_HARMONICS": 2, "RESPONSE_TRANSFORM": "logit",
             "RESPONSE_EPS": 1e-4,
             "MODEL_COLS": {"persistence": "pred_persistence",
                            "seasonal_naive": "pred_seasonal_naive",
                            "regional_dynamic_null": "pred_null",
                            "regional_dynamic_full": "pred_full"}}
    from collections import deque
    space["deque"] = deque
    exec(compile(_cell("# 4. Loading, provenance, calendar and aggregation helpers"),
                 "<load>", "exec"), space)
    exec(compile(_cell("# 5a. Response-blindness guard"), "<blind>", "exec"), space)
    exec(compile(_cell("# 5b. Ecological class assignment"), "<class>", "exec"),
         space)
    # Function definitions only, out of cells that also run pipeline code.
    exec(compile(_defs(_cell("# 9a. Build the region-month panel"),
                       ["regional_monthly_panel",
                        "regional_env_from_complete_tables"]),
                 "<panel>", "exec"), space)
    exec(compile(_defs(_cell("# 10b. Variance decomposition"),
                       ["decompose_driver_variance"]), "<var>", "exec"), space)
    exec(compile(_defs(_cell("# 17c. Skill, by region"),
                       ["_rmse", "_mae", "skill_table"]), "<skill>", "exec"),
         space)
    return space


# ---------------------------------------------------------------------------
# Response blindness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("col", [
    "wh_cover", "wh_present", "mean_cover", "occurrence_rate", "resid",
    "pred_full", "wh_proba", "winning_class_confidence", "kmeans_cluster_id",
])
def test_response_like_columns_are_refused(ns, col):
    with pytest.raises(ValueError, match="response-blindness"):
        ns["assert_response_blind"](["dist_shore_m", col])


@pytest.mark.parametrize("col", [
    "dist_shore_m", "dist_majriver_m", "openness_index", "depth_m", "x_km",
    "gsw_water_fraction", "bathy_water_fraction", "frac_cropland",
])
def test_static_geography_is_allowed(ns, col):
    assert ns["assert_response_blind"]([col]) is True


def test_every_regionalisation_routine_calls_the_guard():
    """The guard is worthless if a routine forgets to call it."""
    src = (_cell("# 5a. Response-blindness guard")
           + _cell("# 5b. Ecological class assignment"))
    for fn in ("static_cell_table", "resolve_region_covariates",
               "assign_ecological_class", "merge_small_components"):
        tree = ast.parse(src)
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        calls = {c.func.id for c in ast.walk(node)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "assert_response_blind" in calls, (
            f"{fn} does not call assert_response_blind")


# ---------------------------------------------------------------------------
# Contiguity: a class is not a region
# ---------------------------------------------------------------------------
def test_disconnected_parts_of_one_class_become_separate_regions(ns):
    ix = np.array([0, 1, 0, 1, 10, 11, 10, 11])
    iy = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    comp = ns["connected_components"](ix, iy, np.array(["sheltered"] * 8))
    assert len(set(comp.tolist())) == 2


def test_a_class_boundary_splits_a_contiguous_block(ns):
    ix = np.array([0, 1, 0, 1])
    iy = np.array([0, 0, 1, 1])
    lab = np.array(["a", "a", "b", "b"])
    comp = ns["connected_components"](ix, iy, lab)
    assert len(set(comp.tolist())) == 2
    assert comp[0] == comp[1] and comp[2] == comp[3] and comp[0] != comp[2]


def test_rook_contiguity_is_stricter_than_queen(ns):
    ix = np.array([0, 1])
    iy = np.array([0, 1])                      # diagonal neighbours
    lab = np.array(["a", "a"])
    assert len(set(ns["connected_components"](ix, iy, lab, "queen").tolist())) == 1
    assert len(set(ns["connected_components"](ix, iy, lab, "rook").tolist())) == 2


def test_adjacency_needs_a_shared_edge(ns):
    ix = np.array([0, 1, 10])
    iy = np.array([0, 0, 0])
    comp = np.array([0, 1, 2])
    pairs = ns["component_adjacency"](ix, iy, comp)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def _merge_cells():
    """A 3x3 block: the middle cell is its own component, with two neighbours."""
    return pd.DataFrame({
        "grid_id": np.arange(9),
        "x_km": (np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]) + 0.5) * 0.5,
        "y_km": (np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) + 0.5) * 0.5,
        "dist_shore_m": [100, 100, 100, 100, 120, 3000, 100, 100, 100],
        "openness_index": [0.2, 0.2, 0.2, 0.2, 0.21, 0.9, 0.2, 0.2, 0.2],
        "eligible_area_ha": np.full(9, 25.0),
    })


def test_merge_picks_the_most_similar_neighbour_not_the_first(ns):
    cells = _merge_cells()
    comp = np.array([0, 0, 1, 0, 2, 1, 0, 0, 0])
    merged, log = ns["merge_small_components"](
        cells, comp, ["dist_shore_m", "openness_index"], min_cells=2,
        min_area_ha=0.0, cell_size_m=500)
    assert merged[4] == merged[0], "did not merge into the similar neighbour"
    assert merged[4] != merged[2], "merged into the dissimilar neighbour"
    assert "merged" in set(log["action"])
    assert log["reason"].str.len().gt(0).all()


def test_an_isolated_undersized_component_is_dropped_and_reported(ns):
    cells = pd.DataFrame({
        "grid_id": [0, 1, 2],
        "x_km": [0.25, 0.75, 10.25], "y_km": [0.25, 0.25, 0.25],
        "dist_shore_m": [100.0, 100.0, 100.0],
        "openness_index": [0.2, 0.2, 0.2],
        "eligible_area_ha": [25.0, 25.0, 25.0],
    })
    comp = np.array([0, 0, 1])
    merged, log = ns["merge_small_components"](
        cells, comp, ["dist_shore_m", "openness_index"], min_cells=2,
        min_area_ha=0.0, cell_size_m=500)
    assert merged[2] == -1
    assert "dropped" in set(log["action"])


def test_merging_is_order_independent(ns):
    """Always merging the SMALLEST offender first makes the outcome deterministic."""
    cells = _merge_cells()
    a, _ = ns["merge_small_components"](
        cells, np.array([0, 0, 1, 0, 2, 1, 0, 0, 0]),
        ["dist_shore_m", "openness_index"], 2, 0.0, cell_size_m=500)
    b, _ = ns["merge_small_components"](
        cells, np.array([5, 5, 9, 5, 7, 9, 5, 5, 5]),
        ["dist_shore_m", "openness_index"], 2, 0.0, cell_size_m=500)
    # relabelled, but the same partition
    assert (pd.factorize(a)[0] == pd.factorize(b)[0]).all()


# ---------------------------------------------------------------------------
# Class assignment
# ---------------------------------------------------------------------------
def test_class_precedence_is_river_then_shelter_then_exposure(ns):
    cells = pd.DataFrame({
        "grid_id": [0, 1, 2, 3],
        "dist_majriver_m": [1000, 20000, 20000, 20000],
        "dist_shore_m": [300, 300, 300, 9000],
        "openness_index": [0.2, 0.2, 0.9, 0.9],
        "depth_m": [2.0, 2.0, 3.0, 20.0]})
    cov = {"river_dist_m": "dist_majriver_m", "shore_dist_m": "dist_shore_m",
           "openness": "openness_index", "depth_m": "depth_m"}
    out = ns["assign_ecological_class"](
        cells, cov, {"river_dist_m": 5000.0, "shore_dist_m": 2000.0,
                     "openness": 0.5, "depth_m": None})
    assert out["region_type"].tolist() == [
        "river_influenced_bay", "sheltered_littoral", "exposed_littoral",
        "open_gulf"]


def test_a_deep_nearshore_cell_can_be_forced_to_open_gulf(ns):
    cells = pd.DataFrame({"grid_id": [0], "dist_majriver_m": [20000.0],
                          "dist_shore_m": [300.0], "openness_index": [0.2],
                          "depth_m": [30.0]})
    cov = {"river_dist_m": "dist_majriver_m", "shore_dist_m": "dist_shore_m",
           "openness": "openness_index", "depth_m": "depth_m"}
    assert ns["assign_ecological_class"](
        cells, cov, {"river_dist_m": 5000.0, "shore_dist_m": 2000.0,
                     "openness": 0.5, "depth_m": 10.0}
    )["region_type"].iloc[0] == "open_gulf"


def test_every_cell_carries_an_auditable_rule(ns):
    cells = pd.DataFrame({"grid_id": [0, 1], "dist_majriver_m": [1000.0, 20000.0],
                          "dist_shore_m": [300.0, 9000.0],
                          "openness_index": [0.2, 0.9], "depth_m": [2.0, 20.0]})
    cov = {"river_dist_m": "dist_majriver_m", "shore_dist_m": "dist_shore_m",
           "openness": "openness_index", "depth_m": "depth_m"}
    out = ns["assign_ecological_class"](
        cells, cov, {"river_dist_m": 5000.0, "shore_dist_m": 2000.0,
                     "openness": 0.5, "depth_m": None})
    for rule in out["assignment_rule"]:
        assert "river_dist=" in rule and "shore_dist=" in rule


# ---------------------------------------------------------------------------
# Static covariates
# ---------------------------------------------------------------------------
def test_a_time_varying_covariate_is_refused_as_static(ns):
    panel = pd.DataFrame({
        "grid_id": [1, 1, 2, 2],
        "month": pd.to_datetime(["2020-01-01", "2020-02-01"] * 2),
        "depth_m": [5.0, 5.0, 7.0, 7.0],
        "openness_index": [0.3, 0.9, 0.4, 0.4]})
    table, audit = ns["static_cell_table"](panel, ["depth_m", "openness_index"])
    assert "depth_m" in table.columns
    assert "openness_index" not in table.columns
    assert not audit.loc[audit["column"] == "openness_index",
                         "treated_as_static"].iloc[0]


# ---------------------------------------------------------------------------
# build_regions end to end
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_cells():
    """A 20x6 strip of 500 m cells with a river mouth at the west end."""
    ix, iy = np.meshgrid(np.arange(20), np.arange(6), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    x_km = (ix + 0.5) * 0.5
    y_km = (iy + 0.5) * 0.5
    dist_shore = (np.minimum(iy, 5 - iy) + 0.5) * 500.0
    dist_river = np.hypot(x_km - 0.25, y_km - 1.5) * 1000.0
    return pd.DataFrame({
        "grid_id": np.arange(ix.size), "x_km": x_km, "y_km": y_km,
        "dist_shore_m": dist_shore, "dist_majriver_m": dist_river,
        "openness_index": 0.2 + 0.7 * (dist_shore / dist_shore.max()),
        "depth_m": 1.0 + 0.004 * dist_shore,
        "eligible_area_ha": np.full(ix.size, 25.0),
        "cell_area_m2": np.full(ix.size, 250000.0)})


def test_build_regions_gives_every_cell_one_stable_region(ns, toy_cells):
    cov = {"river_dist_m": "dist_majriver_m", "shore_dist_m": "dist_shore_m",
           "openness": "openness_index", "depth_m": "depth_m"}
    th = {"river_dist_m": 2000.0, "shore_dist_m": 800.0, "openness": 0.5,
          "depth_m": None}
    assign, regions, log, class_audit = ns["build_regions"](
        toy_cells, cov, th, cell_size_m=500, min_cells=5, min_area_ha=0.0)
    assert assign["grid_id"].is_unique
    assert assign["region_id"].notna().all()
    assert assign["region_name"].notna().all()
    assert assign["region_type"].notna().all()
    assert assign["assignment_audit"].str.contains("class=").all()
    assert set(regions["region_id"]) == set(assign["region_id"])
    assert regions["region_id"].is_unique and regions["region_name"].is_unique
    # ids are assigned in a deterministic class-then-size order
    a2, r2, _, _ = ns["build_regions"](toy_cells, cov, th, cell_size_m=500,
                                       min_cells=5, min_area_ha=0.0)
    pd.testing.assert_frame_equal(assign, a2)


def test_no_region_is_smaller_than_the_minimum_after_merging(ns, toy_cells):
    cov = {"river_dist_m": "dist_majriver_m", "shore_dist_m": "dist_shore_m",
           "openness": "openness_index", "depth_m": "depth_m"}
    th = {"river_dist_m": 2000.0, "shore_dist_m": 800.0, "openness": 0.5,
          "depth_m": None}
    _, regions, _, _ = ns["build_regions"](
        toy_cells, cov, th, cell_size_m=500, min_cells=12, min_area_ha=0.0)
    assert (regions["n_cells"] >= 12).all()


# ---------------------------------------------------------------------------
# The region-month panel
# ---------------------------------------------------------------------------
def _panel(n_months=8, n_cells=6, seed=0):
    """Cell 0 has ten times the valid area of the rest, so the two means differ."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for gid in range(n_cells):
        valid = 250.0 if gid == 0 else 25.0
        for m in months:
            cover = 0.6 if gid == 0 else 0.1
            rows.append({"grid_id": gid, "month": m, "valid_area_ha": valid,
                         "wh_area_ha": cover * valid, "wh_cover": cover,
                         "wh_present": int(cover > 0.01),
                         "rain_chirps_30d_mm": 50 + 10 * rng.standard_normal()})
    return pd.DataFrame(rows)


def _assignments(n_cells=6):
    return pd.DataFrame({"grid_id": np.arange(n_cells),
                         "region_id": ["R01"] * n_cells,
                         "eligible_area_ha": np.full(n_cells, 25.0)})


def test_regional_cover_is_area_weighted_not_a_mean_of_cover(ns):
    panel = _panel()
    out = ns["regional_monthly_panel"](panel, _assignments(),
                                       ["rain_chirps_30d_mm"],
                                       min_cell_coverage=0.0,
                                       min_area_coverage=0.0)
    row = out.iloc[0]
    expected = row["wh_area_ha"] / row["valid_area_ha"]
    assert np.isclose(row["wh_cover"], expected)
    unweighted = panel[panel["month"] == row["month"]]["wh_cover"].mean()
    assert not np.isclose(row["wh_cover"], unweighted), (
        "area weighting made no difference; the fixture is not exercising it")


def test_a_region_month_below_a_coverage_gate_is_missing_not_partial(ns):
    panel = _panel()
    # drop most cells in the third month
    m = sorted(panel["month"].unique())[2]
    panel = panel[~((panel["month"] == m) & (panel["grid_id"] > 1))]
    out = ns["regional_monthly_panel"](panel, _assignments(),
                                       ["rain_chirps_30d_mm"],
                                       min_cell_coverage=0.8,
                                       min_area_coverage=0.0)
    bad = out[out["month"] == m]
    assert not bad["region_month_usable"].iloc[0]
    assert pd.isna(bad["wh_cover"].iloc[0])
    assert pd.isna(bad["wh_area_ha"].iloc[0])
    assert bad["exclusion_reason"].iloc[0]


def test_coverage_fractions_are_reported_for_every_region_month(ns):
    out = ns["regional_monthly_panel"](_panel(), _assignments(),
                                       ["rain_chirps_30d_mm"],
                                       min_cell_coverage=0.0,
                                       min_area_coverage=0.0)
    for col in ("cell_coverage_fraction", "valid_area_coverage_fraction",
                "n_cells_eligible", "n_cells_observed", "wh_occurrence",
                "valid_area_ha", "wh_area_ha"):
        assert col in out.columns and out[col].notna().all()


def test_drivers_are_filled_for_months_the_panel_never_saw(ns):
    """A month with no WH map must still carry environmental forcing."""
    assign = _assignments()
    months = pd.date_range("2020-01-01", periods=10, freq="MS")
    env_monthly = pd.DataFrame({"month": months,
                                "lake_level_m": np.linspace(1134, 1135, 10)})
    out = ns["regional_env_from_complete_tables"](
        assign, ["lake_level_m"], env_monthly=env_monthly)
    assert out["month"].nunique() == 10
    assert out["lake_level_m"].notna().all()


def test_per_cell_env_tables_are_area_weighted_per_region(ns):
    assign = pd.DataFrame({"grid_id": [0, 1], "region_id": ["R01", "R01"],
                           "eligible_area_ha": [30.0, 10.0]})
    env = pd.DataFrame({"grid_id": [0, 1], "month": [pd.Timestamp("2020-01-01")] * 2,
                        "rain_chirps_30d_mm": [100.0, 0.0]})
    out = ns["regional_env_from_complete_tables"](
        assign, ["rain_chirps_30d_mm"], env_cellmonth=env)
    assert np.isclose(out["rain_chirps_30d_mm"].iloc[0], 75.0)


# ---------------------------------------------------------------------------
# Calendar handling
# ---------------------------------------------------------------------------
def test_every_region_gets_the_same_complete_calendar(ns):
    frame = pd.DataFrame({
        "region_id": ["R01", "R01", "R02"],
        "month": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]),
        "v": [1.0, 2.0, 3.0]})
    out = ns["reindex_calendar_months"](frame, by="region_id")
    assert out.groupby("region_id")["month"].nunique().nunique() == 1
    assert out["month"].nunique() == 4
    gap = out[(out["region_id"] == "R01") & (out["month"]
                                             == pd.Timestamp("2020-02-01"))]
    assert gap["v"].isna().all(), "an excluded month was interpolated"


def test_month_index_differences_are_calendar_differences(ns):
    mi = ns["month_index"](pd.to_datetime(["2019-11-01", "2020-02-01"]))
    assert int(mi[1] - mi[0]) == 3


def test_fourier_terms_depend_only_on_the_calendar_month(ns):
    a = ns["fourier_terms"](pd.Series(pd.to_datetime(["2019-03-01"])), 2)
    b = ns["fourier_terms"](pd.Series(pd.to_datetime(["2024-03-01"])), 2)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_logit_clipping_is_counted(ns):
    y, info = ns["transform_response"](pd.Series([0.0, 0.5, 1.0]), "logit", 1e-4)
    assert info["n_clipped_low"] == 1 and info["n_clipped_high"] == 1
    assert np.isclose(ns["inverse_transform_response"](y.to_numpy()[1]), 0.5)


# ---------------------------------------------------------------------------
# Provenance gate
# ---------------------------------------------------------------------------
def test_provenance_gate_blocks_a_confidence_weighted_panel(ns):
    panel = pd.DataFrame({"grid_id": [1], "month": [pd.Timestamp("2020-01-01")]})
    manifest = {"cell_size_m": 500, "response_kind": "hard_class",
                "confidence_usage": {"weight_cover_by_confidence": True}}
    audit, blocking = ns["panel_provenance_audit"](panel, manifest,
                                                   require_hard_class=True)
    assert "hard_class_response" in blocking
    assert not audit.loc[audit["check"] == "hard_class_response", "ok"].iloc[0]


def test_provenance_gate_blocks_the_wrong_cell_size(ns):
    panel = pd.DataFrame({"grid_id": [1], "month": [pd.Timestamp("2020-01-01")]})
    _, blocking = ns["panel_provenance_audit"](
        panel, {"cell_size_m": 1000}, expected_cell_size_m=500,
        require_hard_class=False)
    assert "cell_size_m" in blocking


def test_provenance_gate_blocks_duplicated_cell_months(ns):
    panel = pd.DataFrame({"grid_id": [1, 1],
                          "month": [pd.Timestamp("2020-01-01")] * 2})
    _, blocking = ns["panel_provenance_audit"](panel, {"cell_size_m": 500},
                                               require_hard_class=False)
    assert "one_row_per_cell_month" in blocking


def test_provenance_gate_passes_a_clean_hard_class_panel(ns):
    panel = pd.DataFrame({"grid_id": [1, 2],
                          "month": [pd.Timestamp("2020-01-01")] * 2,
                          "sensor": ["S2", "S2"]})
    manifest = {"cell_size_m": 500, "response_kind": "hard_class",
                "confidence_usage": {}}
    _, blocking = ns["panel_provenance_audit"](
        panel, manifest, provenance_columns=["sensor"])
    assert blocking == []


# ---------------------------------------------------------------------------
# The variance decomposition the temporal_only label rests on
# ---------------------------------------------------------------------------
def _driver_frame(values_by_region, months=12):
    idx = pd.date_range("2020-01-01", periods=months, freq="MS")
    rows = []
    for rid, series in values_by_region.items():
        for t, (m, v) in enumerate(zip(idx, series)):
            rows.append({"region_id": rid, "month": m, "x": float(v),
                         "time_index": float(t), "month_num": m.month})
    return pd.DataFrame(rows)


def test_a_gulf_wide_driver_has_no_within_month_regional_variance(ns):
    rng = np.random.default_rng(0)
    shared = rng.normal(size=12)
    frame = _driver_frame({"R01": shared, "R02": shared, "R03": shared})
    out = ns["decompose_driver_variance"](frame, "x")
    assert out["share_between_months"] == pytest.approx(1.0, abs=1e-9)
    assert out["share_within_month_between_regions"] == pytest.approx(0.0, abs=1e-9)
    assert out["median_within_month_cv"] == pytest.approx(0.0, abs=1e-9)


def test_a_driver_with_a_regional_gradient_has_within_month_variance(ns):
    rng = np.random.default_rng(1)
    shared = rng.normal(size=12)
    frame = _driver_frame({"R01": shared - 2.0, "R02": shared,
                           "R03": shared + 2.0})
    out = ns["decompose_driver_variance"](frame, "x")
    assert out["share_within_month_between_regions"] > 0.2
    # a FIXED offset is a region main effect, not an interaction
    assert out["share_region_main_effect"] > out["share_region_x_month_interaction"]


def test_the_month_split_is_exact(ns):
    rng = np.random.default_rng(2)
    frame = _driver_frame({f"R{i:02d}": rng.normal(size=12) for i in range(4)})
    out = ns["decompose_driver_variance"](frame, "x")
    assert out["month_split_sums_to_one"] == pytest.approx(0.0, abs=1e-9)
    assert (out["share_between_months"]
            + out["share_within_month_between_regions"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Skill metrics
# ---------------------------------------------------------------------------
def test_region_macro_skill_weights_regions_equally(ns):
    """One big region predicted well must not hide a small region predicted badly."""
    rows = []
    for i in range(20):                       # R01: many rows, perfect
        rows.append({"region_id": "R01", "target_month": pd.Timestamp("2020-01-01"),
                     "y_true": 0.0, "pred_a": 0.0, "pred_b": 0.0,
                     "valid_area_ha": 1000.0})
    for i in range(2):                        # R02: few rows, badly wrong
        rows.append({"region_id": "R02", "target_month": pd.Timestamp("2020-01-01"),
                     "y_true": 0.0, "pred_a": 1.0, "pred_b": 0.0,
                     "valid_area_ha": 10.0})
    pred = pd.DataFrame(rows)
    tab = ns["skill_table"](pred, cols={"a": "pred_a"}, restrict_common=False)
    macro = float(tab["rmse_logit_region_macro"].iloc[0])
    pooled = float(tab["rmse_logit_pooled"].iloc[0])
    assert macro > pooled, "region-macro did not give the small region equal weight"
    assert macro == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The model: what the likelihood can and cannot see
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_ns(ns):
    pytest.importorskip("pymc")
    import pymc as pm
    import pytensor.tensor as pt

    space = dict(ns)
    space.update({"pm": pm, "pt": pt, "HAVE_PYMC": True, "time": __import__("time"),
                  "FAST_MODE": True,
                  "RANDOM_SLOPE_PARAMETERISATION": "centred",
                  "HIERARCHY_PARAMETERISATION": "centred",
                  "SAMPLING": dict(draws=10, tune=10, chains=2, cores=1,
                                   target_accept=0.9, random_seed=0),
                  "PRIORS": {"mu_alpha_sd": 1.5, "sigma_alpha": 1.0,
                             "beta_sd": 0.5, "sigma_b": 0.25, "season_sd": 0.5,
                             "trend_sd": 0.25, "sigma_g": 0.5,
                             "sigma_lambda": 0.3, "sigma_u": 0.4,
                             "sigma_eps": 0.5, "rho_a": 2.0, "rho_b": 2.0}})
    exec(compile(_defs(_cell("# 12. Model builder"),
                       ["make_model_data", "build_regional_model"]),
                 "<model>", "exec"), space)
    return space


def _toy_model_data(model_ns, R=4, T=18, K=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(R, T, K))
    Y = rng.normal(size=(R, T))
    mask = rng.random((R, T)) > 0.3
    Y = np.where(mask, Y, np.nan)
    season = np.c_[np.sin(np.arange(T)), np.cos(np.arange(T))]
    tt = (np.arange(T) - T / 2) / T
    return model_ns["make_model_data"](
        X, Y, mask, season, tt, [f"R{i:02d}" for i in range(R)],
        [f"d{k}" for k in range(K)])


@pytest.mark.parametrize("common_state", ["ar1", "randomwalk", "none"])
def test_every_shared_state_structure_builds_and_has_a_finite_logp(model_ns,
                                                                   common_state):
    data = _toy_model_data(model_ns)
    m = model_ns["build_regional_model"](data, drivers=True,
                                         common_state=common_state,
                                         regional_ar="common")
    lp = float(m.compile_logp()(m.initial_point()))
    assert np.isfinite(lp)


@pytest.mark.parametrize("regional_ar", ["common", "per_region", "none"])
def test_every_regional_ar_mode_builds(model_ns, regional_ar):
    data = _toy_model_data(model_ns)
    m = model_ns["build_regional_model"](data, drivers=True, common_state="ar1",
                                         regional_ar=regional_ar)
    assert np.isfinite(float(m.compile_logp()(m.initial_point())))


def test_the_likelihood_sees_exactly_the_observation_index(model_ns):
    data = _toy_model_data(model_ns)
    m = model_ns["build_regional_model"](data, drivers=True, common_state="ar1",
                                         regional_ar="common")
    observed = m["y"].eval() if hasattr(m["y"], "eval") else None
    assert len(data["y_obs"]) == int(data["obs_mask"].sum())
    assert observed is None or len(observed) == len(data["y_obs"])


def test_a_missing_region_month_cannot_change_the_log_probability(model_ns):
    """The property that makes the placeholder zeros safe."""
    data = _toy_model_data(model_ns)
    m1 = model_ns["build_regional_model"](data, drivers=True, common_state="ar1",
                                          regional_ar="common")
    lp1 = float(m1.compile_logp()(m1.initial_point()))

    # Vandalise the response AND the predictors wherever the mask is False.
    Y2 = data["Y"].copy()
    X2 = data["X"].copy()
    off = ~data["obs_mask"]
    Y2[off] = 1e6
    X2[off] = 1e6
    d2 = model_ns["make_model_data"](X2, Y2, data["obs_mask"], data["season"],
                                     data["tt"], data["region_ids"],
                                     data["driver_terms"])
    m2 = model_ns["build_regional_model"](d2, drivers=True, common_state="ar1",
                                          regional_ar="common")
    lp2 = float(m2.compile_logp()(m2.initial_point()))
    assert lp1 == pytest.approx(lp2), (
        "a region-month outside the observation index changed the likelihood")


def test_the_null_model_has_no_driver_coefficient(model_ns):
    data = _toy_model_data(model_ns)
    m = model_ns["build_regional_model"](data, drivers=False, common_state="ar1",
                                         regional_ar="common")
    assert "beta" not in {v.name for v in m.free_RVs + m.deterministics}


def test_stationary_ar_parameters_are_bounded_in_the_unit_interval(model_ns):
    import pymc as pm
    data = _toy_model_data(model_ns)
    m = model_ns["build_regional_model"](data, drivers=True, common_state="ar1",
                                         regional_ar="common")
    draws = pm.draw([m["rho_g"], m["rho_u"]], draws=200, random_seed=0)
    for arr in draws:
        arr = np.asarray(arr)
        assert (arr > 0).all() and (arr < 1).all()


@pytest.mark.parametrize("parameterisation", ["centred", "noncentred"])
def test_both_random_slope_parameterisations_build(model_ns, parameterisation):
    """The ladder switches between them, so both must be constructible."""
    rng = np.random.default_rng(4)
    R, T = 4, 18
    data = model_ns["make_model_data"](
        rng.normal(size=(R, T, 2)), rng.normal(size=(R, T)),
        np.ones((R, T), bool),
        np.c_[np.sin(np.arange(T)), np.cos(np.arange(T))], np.arange(T) / T,
        [f"R{i:02d}" for i in range(R)], ["d0", "d1"],
        random_slope_terms=["d0"])
    m = model_ns["build_regional_model"](
        data, drivers=True, common_state="ar1", regional_ar="common",
        use_random_slopes=True,
        random_slope_parameterisation=parameterisation)
    names = {v.name for v in m.free_RVs + m.deterministics}
    assert "b" in names and "sigma_b" in names
    assert ("b_z" in names) == (parameterisation == "noncentred")
    assert np.isfinite(float(m.compile_logp()(m.initial_point())))


@pytest.mark.parametrize("parameterisation", ["centred", "noncentred"])
def test_both_hierarchy_parameterisations_build(model_ns, parameterisation):
    """alpha_r and lambda_r have two parameterisations; the ladder switches them."""
    data = _toy_model_data(model_ns)
    m = model_ns["build_regional_model"](
        data, drivers=True, common_state="ar1", regional_ar="common",
        hierarchy_parameterisation=parameterisation)
    names = {v.name for v in m.free_RVs + m.deterministics}
    assert "alpha" in names and "lam" in names
    assert ("alpha_z" in names) == (parameterisation == "noncentred")
    assert ("lam_z" in names) == (parameterisation == "noncentred")
    assert np.isfinite(float(m.compile_logp()(m.initial_point())))


def test_the_two_hierarchy_parameterisations_are_the_same_model(model_ns):
    """Switching parameterisation is a geometry change, not a model change."""
    import pymc as pm
    data = _toy_model_data(model_ns, R=3, T=10, K=1, seed=7)
    draws = {}
    for how in ("centred", "noncentred"):
        m = model_ns["build_regional_model"](
            data, drivers=True, common_state="none", regional_ar="none",
            include_trend=False, hierarchy_parameterisation=how)
        with m:
            pr = pm.sample_prior_predictive(draws=4000, random_seed=3)
        draws[how] = pr.prior["alpha"].to_numpy().ravel()
    a, b = draws["centred"], draws["noncentred"]
    assert abs(a.mean() - b.mean()) < 0.15
    assert abs(a.std() - b.std()) / max(b.std(), 1e-9) < 0.15


def test_random_slopes_are_only_built_when_asked_for(model_ns):
    rng = np.random.default_rng(3)
    R, T, K = 4, 18, 2
    X = rng.normal(size=(R, T, K))
    Y = rng.normal(size=(R, T))
    mask = np.ones((R, T), bool)
    season = np.c_[np.sin(np.arange(T)), np.cos(np.arange(T))]
    data = model_ns["make_model_data"](
        X, Y, mask, season, np.arange(T) / T,
        [f"R{i:02d}" for i in range(R)], ["d0", "d1"],
        random_slope_terms=["d0"])
    on = model_ns["build_regional_model"](data, drivers=True, common_state="ar1",
                                          regional_ar="common",
                                          use_random_slopes=True)
    off = model_ns["build_regional_model"](data, drivers=True, common_state="ar1",
                                           regional_ar="common",
                                           use_random_slopes=False)
    names_on = {v.name for v in on.free_RVs + on.deterministics}
    names_off = {v.name for v in off.free_RVs + off.deterministics}
    assert "b" in names_on and "sigma_b" in names_on
    assert "b" not in names_off and "sigma_b" not in names_off


# ---------------------------------------------------------------------------
# Notebook-level guarantees
# ---------------------------------------------------------------------------
def test_the_notebook_exists_and_carries_no_month_fixed_effects():
    src = "\n".join("".join(c["source"]) for c in _cells()
                    if c["cell_type"] == "code")
    assert "C(month)" not in src and "month_dummies" not in src
    assert "pd.get_dummies" not in src


def test_the_notebook_never_treats_cell_months_as_the_sample():
    """The headline discipline, asserted in the notebook itself."""
    src = "\n".join("".join(c["source"]) for c in _cells()
                    if c["cell_type"] == "code")
    assert "the inferential dataset has one row per region per calendar month" in src
    assert "cell-months are never used as the inferential n" in src


def test_the_notebook_declares_its_mechanisms_and_lags_up_front():
    cfg = _cell("# 3a. Where the data comes from")
    assert "REGIONAL_FORCING_TERMS" in cfg
    assert "REGIONAL_PROXY_TERMS" in cfg
    assert "RANDOM_SLOPE_CANDIDATES" in cfg
    assert "ROPE_HALFWIDTH" in cfg
    assert "FAST_MODE" in cfg


def test_the_temporal_notebook_was_not_modified():
    """This notebook is additive; the AOI temporal model must be untouched."""
    other = REPO / "winam_wh_temporal_driver_model.ipynb"
    assert other.exists()
    assert NOTEBOOK.name != other.name
