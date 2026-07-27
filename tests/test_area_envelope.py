"""Tests for winam_diagnostics.area_envelope (Workstream B3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winam_diagnostics import area_envelope as ae


def test_soft_le_hard_and_ratio_in_unit_interval():
    # 5x5 raster: some floating (class 2) pixels at varying confidence.
    cls = np.array([
        [2, 2, 2, 0, 0],
        [2, 2, 1, 0, 0],
        [2, 2, 2, 3, 0],
        [0, 0, 0, 0, 0],
        [255, 255, 0, 0, 0],  # nodata row
    ])
    proba = np.array([
        [100, 80, 60, 90, 50],
        [70, 100, 40, 10, 20],
        [55, 65, 75, 88, 30],
        [10, 10, 10, 10, 10],
        [255, 255, 50, 50, 50],
    ])
    pixel_area_m2 = 100.0  # 10 m pixels
    env = ae.soft_hard_floating_area(cls, proba, pixel_area_m2)

    # Count of floating (class 2, valid) pixels.
    floating = (cls == 2) & (cls != 255) & (proba != 255)
    assert env["hard_pixels"] == int(floating.sum())
    # soft <= hard, ratio in (0, 1].
    assert env["soft_area_ha"] <= env["hard_area_ha"]
    assert 0.0 < env["soft_hard_ratio"] <= 1.0
    # gap% is exactly (hard - soft) / hard * 100.
    expected_gap_pct = (env["hard_area_ha"] - env["soft_area_ha"]) / env["hard_area_ha"] * 100.0
    assert env["gap_pct"] == pytest.approx(expected_gap_pct)
    # mean confidence equals soft/hard ratio (probability-weighted definition).
    assert env["mean_confidence"] == pytest.approx(env["soft_hard_ratio"])


def test_all_full_confidence_gives_zero_gap():
    cls = np.array([[2, 2], [2, 0]])
    proba = np.array([[100, 100], [100, 0]])
    env = ae.soft_hard_floating_area(cls, proba, pixel_area_m2=100.0)
    assert env["hard_pixels"] == 3
    assert env["gap_ha"] == pytest.approx(0.0)
    assert env["gap_pct"] == pytest.approx(0.0)
    assert env["soft_hard_ratio"] == pytest.approx(1.0)


def test_no_floating_pixels_yields_nan_ratio():
    cls = np.array([[0, 1], [3, 0]])
    proba = np.array([[90, 90], [90, 90]])
    env = ae.soft_hard_floating_area(cls, proba, pixel_area_m2=100.0)
    assert env["hard_pixels"] == 0
    assert env["hard_area_ha"] == 0.0
    assert np.isnan(env["gap_pct"])
    assert np.isnan(env["soft_hard_ratio"])


def test_probabilities_are_clipped_so_soft_never_exceeds_hard():
    # Even if a stray probability exceeds the 0..100 scale, soft must not exceed hard.
    cls = np.array([[2, 2]])
    proba = np.array([[120, 130]])  # out of range on purpose
    env = ae.soft_hard_floating_area(cls, proba, pixel_area_m2=100.0)
    assert env["soft_area_ha"] <= env["hard_area_ha"]
    assert env["soft_hard_ratio"] == pytest.approx(1.0)


def test_compute_area_envelope_without_proba_collapses_to_hard():
    # Rows lacking a probability raster fall back to soft = hard.
    rows = pd.DataFrame([
        {"sensor": "S2", "method": "RF", "start_date": "2020-01-01",
         "end_date": "2020-01-02", "area_ha": 12.5,
         "classification_output": "/nope/class.tif", "probability_output": ""},
        {"sensor": "S2", "method": "RF", "start_date": "2020-02-01",
         "end_date": "2020-02-02", "area_ha": 20.0,
         "classification_output": "/nope/class.tif", "probability_output": "/nope/proba.tif"},
    ])
    env = ae.compute_area_envelope(rows, path_exists=lambda p: False, verbose=False)
    assert len(env) == 2
    assert (~env["has_proba"]).all()
    assert (env["soft_area_ha"] == env["hard_area_ha"]).all()
    assert (env["gap_ha"] == 0.0).all()
    # Columns present and ordered.
    for col in ae.ENVELOPE_COLUMNS:
        assert col in env.columns


def test_compute_area_envelope_tolerates_nan_paths():
    # Reading the batch area CSV turns empty path cells into float NaN. NaN is
    # truthy, so the default path check must not hand it to pathlib (which raises
    # "argument should be a str or an os.PathLike object ... not 'float'"). Such
    # rows collapse to soft = hard instead of crashing the whole cell.
    rows = pd.DataFrame([
        {"sensor": "S2", "method": "RF", "start_date": "2020-01-01",
         "end_date": "2020-01-02", "area_ha": 12.5,
         "classification_output": np.nan, "probability_output": np.nan},
        {"sensor": "S2", "method": "RF", "start_date": "2020-02-01",
         "end_date": "2020-02-02", "area_ha": 20.0,
         "classification_output": "/nope/class.tif", "probability_output": np.nan},
    ])
    # Default path_exists (the one that crashed in the notebook) must be safe here.
    env = ae.compute_area_envelope(rows, verbose=False)
    assert len(env) == 2
    assert (~env["has_proba"]).all()
    assert (env["soft_area_ha"] == env["hard_area_ha"]).all()
    assert (env["gap_ha"] == 0.0).all()
    # Hard area is carried through from the area_ha column for the NaN-path rows.
    assert set(env["hard_area_ha"]) == {12.5, 20.0}


def test_clean_path_normalizes_bad_and_good_values():
    from pathlib import Path

    assert ae._clean_path(np.nan) is None
    assert ae._clean_path(None) is None
    assert ae._clean_path("") is None
    assert ae._clean_path("   ") is None
    assert ae._clean_path(3.0) is None
    assert ae._clean_path("  /data/class.tif  ") == "/data/class.tif"
    assert ae._clean_path(Path("/data/class.tif")) == "/data/class.tif"


def test_compute_area_envelope_reads_rasters_when_available(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # Write a tiny class + proba GeoTIFF pair (10 m pixels).
    cls = np.array([[2, 2, 0], [2, 1, 0]], dtype=np.uint8)
    proba = np.array([[100, 50, 0], [80, 90, 0]], dtype=np.uint8)
    transform = from_origin(0, 20, 10, 10)
    profile = dict(driver="GTiff", height=2, width=3, count=1, dtype="uint8",
                   crs="EPSG:32736", transform=transform)
    class_tif = tmp_path / "class.tif"
    proba_tif = tmp_path / "proba.tif"
    with rasterio.open(class_tif, "w", **profile) as dst:
        dst.write(cls, 1)
    with rasterio.open(proba_tif, "w", **profile) as dst:
        dst.write(proba, 1)

    rows = pd.DataFrame([
        {"sensor": "S2", "method": "RF", "start_date": "2020-01-01",
         "end_date": "2020-01-02", "area_ha": np.nan,
         "classification_output": str(class_tif), "probability_output": str(proba_tif)},
    ])
    env = ae.compute_area_envelope(rows, verbose=False)
    assert len(env) == 1
    r = env.iloc[0]
    assert bool(r["has_proba"]) is True
    # Floating (class 2) pixels are (0,0), (0,1), (1,0) with proba 100, 50, 80.
    # 3 floating pixels at 100 m^2 each = 300 m^2 = 0.03 ha hard.
    assert r["hard_area_ha"] == pytest.approx(0.03)
    # soft = (1.00 + 0.50 + 0.80) * 100 m^2 / 1e4 = 0.023 ha.
    assert r["soft_area_ha"] == pytest.approx(0.023)
    assert r["gap_ha"] == pytest.approx(0.007)
    assert r["soft_area_ha"] <= r["hard_area_ha"]


def test_plot_helpers_import_lazily_and_run():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    env = pd.DataFrame([
        {"sensor": "S2", "method": "RF", "start_date": "2020-01-01",
         "end_date": "2020-01-02", "hard_area_ha": 10.0, "soft_area_ha": 8.0,
         "gap_ha": 2.0, "gap_pct": 20.0, "soft_hard_ratio": 0.8, "mean_confidence": 0.8},
        {"sensor": "S2", "method": "RF", "start_date": "2020-02-01",
         "end_date": "2020-02-02", "hard_area_ha": 12.0, "soft_area_ha": 9.0,
         "gap_ha": 3.0, "gap_pct": 25.0, "soft_hard_ratio": 0.75, "mean_confidence": 0.75},
    ])
    ax = ae.plot_area_envelope_timeseries(env)
    assert ax is not None
