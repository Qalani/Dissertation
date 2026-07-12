"""Tests for winam_diagnostics.shoreline (Workstream B1, B2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from winam_diagnostics import shoreline as sh

# A synthetic "water mask" polygon in the Winam Gulf lon/lat box. The shoreline is
# this square's boundary; the centre is far from it, the edge is near it.
WATER_SQUARE = {
    "type": "Polygon",
    "coordinates": [[
        [34.3, -0.45], [34.8, -0.45], [34.8, -0.10], [34.3, -0.10], [34.3, -0.45],
    ]],
}
CENTRE_LON, CENTRE_LAT = 34.55, -0.275  # middle of the square
SHORE_LON, SHORE_LAT = 34.305, -0.275   # ~0.005 deg inside the western edge


def _geo(lon, lat):
    return json.dumps({"type": "Point", "coordinates": [lon, lat]})


def test_compute_dist_to_shore_non_negative_and_centre_further():
    df = pd.DataFrame({
        ".geo": [_geo(CENTRE_LON, CENTRE_LAT), _geo(SHORE_LON, SHORE_LAT)],
    })
    d = sh.compute_dist_to_shore(df, WATER_SQUARE, crs="EPSG:32736")
    assert (d >= 0).all()
    # Lake-centre point must be much further from the shoreline than the near-shore point.
    assert d.iloc[0] > d.iloc[1]
    # Sanity: the near-shore point is within a few hundred metres of the edge.
    assert d.iloc[1] < 1000
    # The centre of a ~0.5x0.35 deg box is tens of km from the boundary.
    assert d.iloc[0] > 10000


def test_compute_dist_to_shore_uses_metric_crs_not_uk():
    # A UTM 36S distance for these coordinates is realistic (tens of km);
    # EPSG:27700 (UK OSGB) would be nonsensical here. Guard against a wrong CRS.
    df = pd.DataFrame({".geo": [_geo(CENTRE_LON, CENTRE_LAT)]})
    d = sh.compute_dist_to_shore(df, WATER_SQUARE, crs="EPSG:32736")
    assert 10000 < d.iloc[0] < 60000


def test_backfill_adds_missing_column():
    df = pd.DataFrame({".geo": [_geo(CENTRE_LON, CENTRE_LAT), _geo(SHORE_LON, SHORE_LAT)]})
    out, info = sh.backfill_dist_to_shore(df, WATER_SQUARE, verbose=False)
    assert "dist_to_shore_m" in out.columns
    assert out["dist_to_shore_m"].notna().all()
    assert info["source"] == "computed"


def test_backfill_prefers_baked_and_warns_on_disagreement(capsys):
    df = pd.DataFrame({
        ".geo": [_geo(CENTRE_LON, CENTRE_LAT), _geo(SHORE_LON, SHORE_LAT)],
        # Deliberately wrong baked values to force a disagreement warning.
        "dist_to_shore_m": [0.0, 99999.0],
    })
    out, info = sh.backfill_dist_to_shore(
        df, WATER_SQUARE, prefer_baked=True, tolerance_m=250.0, verbose=True
    )
    # Baked values are preserved when preferred.
    assert out["dist_to_shore_m"].tolist() == [0.0, 99999.0]
    assert info["source"] == "baked"
    assert info["n_disagree"] >= 1
    assert "WARNING" in capsys.readouterr().out


def test_oof_predictions_cover_all_rows(s2_frame, small_candidates):
    predictors = ["NDVI", "NDMI", "dist_to_shore_m", "nir_glcm_entropy_w5"]
    X = s2_frame[predictors].to_numpy()
    y = s2_frame["class"].to_numpy()
    groups = s2_frame["spatial_block"].to_numpy()
    model = small_candidates["Random Forest"]
    oof = sh.oof_predictions_spatial_cv(X, y, groups, model, n_splits=5, random_state=42)
    assert len(oof) == len(y)
    # Every row received an out-of-fold prediction (no -1 left).
    assert (oof >= 0).all()
    # Separable synthetic data -> out-of-fold accuracy well above chance.
    assert (oof == y).mean() > 0.5


def test_bin_shore_distance_labels_and_edges():
    cats, labels, edges = sh.bin_shore_distance(
        [0, 50, 100, 200, 400, 800], bins=(0, 100, 250, 500, np.inf)
    )
    assert labels == ["0-100", "100-250", "250-500", "500+"]
    # 0,50 -> 0-100 ; 100,200 -> 100-250 ; 400 -> 250-500 ; 800 -> 500+
    assert list(cats.astype(str)) == ["0-100", "0-100", "100-250", "100-250", "250-500", "500+"]


def test_stratified_accuracy_by_shore_bins_and_metrics():
    rng = np.random.default_rng(0)
    n = 400
    dist = rng.uniform(0, 900, n)
    # true classes 0/1/2 (open/LEV/floating)
    y_true = rng.integers(0, 3, n)
    # predictions: mostly correct, but near-shore (<250 m) we inject LEV<->Floating swaps
    y_pred = y_true.copy()
    near = dist < 250
    swap = near & (rng.random(n) < 0.4)
    y_pred[swap & (y_true == 1)] = 2  # LEV -> Floating
    y_pred[swap & (y_true == 2)] = 1  # Floating -> LEV

    df = pd.DataFrame({"dist_to_shore_m": dist, "class": y_true})
    strata = sh.stratified_accuracy_by_shore(
        df, y_pred, bins=(0, 100, 250, 500, np.inf), sensor_name="S2 test",
    )
    # One row per bin.
    assert list(strata["shore_bin_m"]) == ["0-100", "100-250", "250-500", "500+"]
    # Support sums to the number of rows.
    assert strata["support"].sum() == n
    # Near-shore bins carry the injected LEV<->Floating confusion in both directions.
    near_rows = strata[strata["dist_max_m"] <= 250]
    assert near_rows["lev_to_floating"].sum() > 0
    assert near_rows["floating_to_lev"].sum() > 0
    # Far bins have no injected swaps -> better floating recall than near-shore.
    far_recall = strata[strata["shore_bin_m"] == "500+"]["floating_recall"].iloc[0]
    near_recall = strata[strata["shore_bin_m"] == "0-100"]["floating_recall"].iloc[0]
    assert far_recall >= near_recall
    # Basis labelled as out-of-fold by default.
    assert (strata["prediction_basis"] == "out_of_fold").all()


def test_stratified_accuracy_resubstitution_label():
    df = pd.DataFrame({"dist_to_shore_m": [10, 300], "class": [2, 2]})
    strata = sh.stratified_accuracy_by_shore(
        df, [2, 2], resubstitution=True, sensor_name="S2 test",
    )
    assert (strata["prediction_basis"] == "resubstitution").all()
