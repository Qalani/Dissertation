"""Tests for the local temporal-band backfill (Backfill_Temporal_Bands_Local.ipynb).

Everything runs on small synthetic rasters in ``tmp_path``: no Google Drive, no
Earth Engine, no network. The notebook is a thin wiring layer over
``winam_diagnostics.temporal_backfill``, so these tests exercise the logic the
notebook actually runs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
import rasterio.errors  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from winam_diagnostics import temporal_backfill as tb  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic raster helpers.
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 8, 6
ORIGIN_X, ORIGIN_Y = 500_000.0, 9_950_000.0
RES = 10.0
BASE_TRANSFORM = Affine(RES, 0.0, ORIGIN_X, 0.0, -RES, ORIGIN_Y)


def write_raster(path, arrays, descriptions, transform=BASE_TRANSFORM,
                 crs=tb.EXPORT_CRS, nodata=tb.PREDICTOR_NODATA_VALUE, dtype="float32"):
    """Write a small multi-band GeoTIFF with explicit band descriptions."""
    arrays = [np.asarray(a, dtype=dtype) for a in arrays]
    height, width = arrays[0].shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=len(arrays),
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        for index, array in enumerate(arrays, start=1):
            dst.write(array, index)
            dst.set_band_description(index, descriptions[index - 1])
    return path


def write_s2_snapshot(export_dir, date_iso, ndvi, token=tb.S2_LEGACY_SCHEMA_TOKEN,
                      suffix="", transform=BASE_TRANSFORM):
    """A full legacy-schema (20-band) S2 export whose NDVI band is ``ndvi``."""
    end_iso = (dt.date.fromisoformat(date_iso) + dt.timedelta(days=1)).isoformat()
    prefix = f"winam_s2_predictors_{token}_{date_iso}_to_{end_iso}"
    names = list(tb.S2_LEGACY_PREDICTORS) if token == tb.S2_LEGACY_SCHEMA_TOKEN else list(tb.S2_PREDICTORS)
    ndvi = np.asarray(ndvi, dtype="float32")
    bands = []
    for name in names:
        bands.append(ndvi if name == "NDVI" else np.full(ndvi.shape, 1.0, dtype="float32"))
    return write_raster(Path(export_dir) / f"{prefix}{suffix}.tif", bands, names, transform=transform)


def write_s1_snapshot(export_dir, date_iso, vh, temporal=False, suffix=""):
    """A full S1 export (legacy 3-band, or 5-band temporal schema)."""
    end_iso = (dt.date.fromisoformat(date_iso) + dt.timedelta(days=1)).isoformat()
    if temporal:
        prefix = f"winam_{tb.S1_TEMPORAL_SCHEMA_TOKEN}_{date_iso}_to_{end_iso}"
        names = list(tb.S1_PREDICTORS)
    else:
        prefix = f"winam_s1_scc_predictors_{date_iso}_to_{end_iso}"
        names = list(tb.S1_LEGACY_PREDICTORS)
    vh = np.asarray(vh, dtype="float32")
    bands = [vh if name == "VH_corrected" else np.full(vh.shape, -20.0, dtype="float32") for name in names]
    return write_raster(Path(export_dir) / f"{prefix}{suffix}.tif", bands, names)


def cache_source_band(cache_dir, snapshots, readonly_dirs=()):
    """Run Phase 1 for a set of snapshots and return the path resolver."""
    cache_dir = Path(cache_dir)
    paths = {}
    for snap in snapshots:
        out = cache_dir / snap.sensor / f"{snap.prefix}__source.tif"
        tb.extract_source_band(snap, out, readonly_dirs=readonly_dirs)
        paths[snap.prefix] = out
    return lambda snap: paths[snap.prefix]


# ---------------------------------------------------------------------------
# Window membership.
# ---------------------------------------------------------------------------

def test_window_bounds_are_start_inclusive_end_exclusive():
    start, end = tb.window_bounds("2020-06-02")
    assert start == dt.date(2020, 3, 4)
    assert end == dt.date(2020, 6, 2)
    assert (end - start).days == tb.TEMPORAL_LOOKBACK_DAYS


def test_window_includes_exactly_minus_90_days_and_excludes_minus_91():
    """filterDate is inclusive of start: -90 days is in, -91 days is out."""
    end = "2020-06-02"
    assert dt.date(2020, 3, 4) in tb.dates_in_window(end, ["2020-03-04"])
    assert tb.dates_in_window(end, ["2020-03-03"]) == []


def test_window_includes_the_targets_own_acquisition_date():
    """Each snapshot's end_date is its acquisition date + 1 day, so the 90-day
    window that ends at end_date contains the snapshot's own acquisition."""
    end = "2020-06-02"          # snapshot acquired 2020-06-01
    assert tb.dates_in_window(end, ["2020-06-01"]) == [dt.date(2020, 6, 1)]


def test_window_excludes_the_end_date_itself():
    assert tb.dates_in_window("2020-06-02", ["2020-06-02"]) == []
    assert tb.dates_in_window("2020-06-02", ["2020-06-03"]) == []


def test_snapshots_in_window_never_mixes_sensors(tmp_path):
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-06-01", np.zeros((HEIGHT, WIDTH)))
    write_s2_snapshot(export_dir, "2020-05-20", np.zeros((HEIGHT, WIDTH)))
    write_s1_snapshot(export_dir, "2020-05-21", np.full((HEIGHT, WIDTH), -20.0))
    write_s1_snapshot(export_dir, "2020-05-25", np.full((HEIGHT, WIDTH), -20.0))
    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))

    target = next(s for s in snapshots if s.sensor == "S2" and s.start_date == dt.date(2020, 6, 1))
    hits = tb.snapshots_in_window(target, snapshots)
    assert {s.sensor for s in hits} == {"S2"}
    assert [s.start_date for s in hits] == [dt.date(2020, 5, 20), dt.date(2020, 6, 1)]


# ---------------------------------------------------------------------------
# Discovery: tile shards vs Drive collision duplicates.
# ---------------------------------------------------------------------------

def test_s1_prefix_asymmetry_is_parsed_for_both_schemas():
    legacy = tb.parse_predictor_export_name("winam_s1_scc_predictors_2020-01-01_to_2020-01-02.tif")
    current = tb.parse_predictor_export_name("winam_s1_scc_temporal_v1_2020-01-01_to_2020-01-02.tif")
    assert legacy["sensor"] == "S1" and legacy["schema_token"] is None
    assert current["sensor"] == "S1" and current["schema_token"] == tb.S1_TEMPORAL_SCHEMA_TOKEN
    # The S1 bump dropped 'predictors' entirely; simple token substitution fails.
    assert tb.sensor_spec("S1").temporal_prefix("2020-01-01", "2020-01-02") == current["prefix"]
    assert "predictors" not in current["prefix"]


def test_s2_prefix_keeps_predictors_and_inserts_the_token():
    spec = tb.sensor_spec("S2")
    assert spec.temporal_prefix("2020-01-01", "2020-01-02") == (
        "winam_s2_predictors_s2_whlev_temporal_v1_2020-01-01_to_2020-01-02"
    )


def test_tiled_shards_are_grouped_into_one_snapshot(tmp_path):
    export_dir = tmp_path / "exports"
    left = np.full((HEIGHT, WIDTH), 0.2)
    right = np.full((HEIGHT, WIDTH), 0.4)
    shifted = Affine(RES, 0.0, ORIGIN_X + WIDTH * RES, 0.0, -RES, ORIGIN_Y)
    write_s2_snapshot(export_dir, "2020-06-01", left, suffix="-0000000000-0000000000")
    write_s2_snapshot(export_dir, "2020-06-01", right, suffix="-0000000000-0000000008",
                      transform=shifted)

    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    assert len(snapshots) == 1
    assert snapshots[0].is_tiled and len(snapshots[0].paths) == 2

    verdict = tb.classify_prefix_files(snapshots[0])
    assert verdict["kind"] == "tiled"

    # Shards mosaic onto the union grid; the cache is one full-width raster.
    out = tmp_path / "cache" / "s2.tif"
    tb.extract_source_band(snapshots[0], out)
    with rasterio.open(out) as src:
        assert src.width == WIDTH * 2 and src.height == HEIGHT
        data = src.read(1)
    assert np.allclose(data[:, :WIDTH], 0.2)
    assert np.allclose(data[:, WIDTH:], 0.4)


def test_drive_collision_duplicates_are_skipped_not_counted(tmp_path):
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.2))
    # Google Drive's collision copy of the very same export.
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.2), suffix=" (1)")

    inventory = tb.discover_predictor_files(export_dir)
    assert int(inventory["is_drive_duplicate"].sum()) == 1

    snapshots = tb.group_snapshot_files(inventory)
    assert len(snapshots) == 1
    assert len(snapshots[0].paths) == 1
    assert len(snapshots[0].duplicate_paths) == 1
    assert tb.classify_prefix_files(snapshots[0])["kind"] == "single"


def test_duplicate_never_double_counted_in_a_window(tmp_path):
    """A duplicated snapshot must not inflate the observation count."""
    export_dir = tmp_path / "exports"
    values = {"2020-05-10": 0.1, "2020-05-20": 0.2, "2020-06-01": 0.3}
    for date_iso, value in values.items():
        write_s2_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), value))
    # Duplicate one date twice over.
    write_s2_snapshot(export_dir, "2020-05-20", np.full((HEIGHT, WIDTH), 0.2), suffix=" (1)")
    write_s2_snapshot(export_dir, "2020-05-20", np.full((HEIGHT, WIDTH), 0.2), suffix=" (2)")

    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    chosen, collisions = tb.select_snapshots_by_date(snapshots)
    assert collisions == []
    assert len(chosen) == 3

    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    contributors = tb.snapshots_in_window(target, chosen)
    assert len(contributors) == 3

    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S2")
    out_paths = {
        spec.std_band: tmp_path / "out" / "std.tif",
        spec.count_band: tmp_path / "out" / "count.tif",
    }
    result = tb.compute_temporal_bands(target, contributors, resolver, out_paths)
    assert result["status"] == "completed"
    with rasterio.open(out_paths[spec.count_band]) as src:
        assert np.all(src.read(1) == 3)


def test_same_date_under_two_schemas_is_deduped_preferring_temporal(tmp_path):
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.3),
                      token=tb.S2_LEGACY_SCHEMA_TOKEN)
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.3),
                      token=tb.S2_TEMPORAL_SCHEMA_TOKEN)
    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    assert len(snapshots) == 2

    chosen, collisions = tb.select_snapshots_by_date(snapshots)
    assert len(chosen) == 1
    assert chosen[0].has_temporal_bands
    assert len(collisions) == 1
    assert collisions[0]["kept_prefix"] == chosen[0].prefix


def test_duplicate_grid_files_are_reported_not_treated_as_shards(tmp_path):
    """Two files on the identical grid are copies; calling them shards would
    double-count that observation in every overlapping window."""
    export_dir = tmp_path / "exports"
    write_s1_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), -20.0),
                      suffix="-0000000000-0000000000")
    write_s1_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), -20.0),
                      suffix="-0000000000-0000000008")  # same transform: not a real shard

    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    verdict = tb.classify_prefix_files(snapshots[0])
    assert verdict["kind"] == "duplicate_grid"
    assert "double-count" in verdict["message"]


# ---------------------------------------------------------------------------
# Statistics: nodata, ddof, min-obs.
# ---------------------------------------------------------------------------

def test_nodata_never_enters_any_statistic():
    layers = [
        np.array([[1.0, tb.PREDICTOR_NODATA_VALUE]]),
        np.array([[3.0, 2.0]]),
        np.array([[5.0, 4.0]]),
    ]
    count, mean, std = tb.temporal_block_stats(layers, ddof=0)
    assert count.tolist() == [[3, 2]]
    assert mean[0, 0] == pytest.approx(3.0)
    # -9999 excluded: the second pixel averages 2 and 4, not -9999/2/4.
    assert mean[0, 1] == pytest.approx(3.0)
    assert std[0, 0] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert std[0, 1] == pytest.approx(np.std([2.0, 4.0]))


def test_non_finite_values_are_excluded_like_nodata():
    layers = [np.array([[np.nan]]), np.array([[2.0]]), np.array([[4.0]]), np.array([[np.inf]])]
    count, mean, std = tb.temporal_block_stats(layers, ddof=0)
    assert count.tolist() == [[2]]
    assert mean[0, 0] == pytest.approx(3.0)
    assert std[0, 0] == pytest.approx(1.0)


def test_ddof_against_a_hand_computed_fixture():
    """Values 2, 4, 4, 4, 5, 5, 7, 9: population std 2, sample std ~2.13809."""
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    layers = [np.array([[v]]) for v in values]

    _, mean, std0 = tb.temporal_block_stats(layers, ddof=0)
    _, _, std1 = tb.temporal_block_stats(layers, ddof=1)
    assert mean[0, 0] == pytest.approx(5.0)
    assert std0[0, 0] == pytest.approx(2.0)                     # sqrt(32/8)
    assert std1[0, 0] == pytest.approx(np.sqrt(32.0 / 7.0))     # 2.138089...
    assert std1[0, 0] > std0[0, 0]


def test_std_is_undefined_when_count_does_not_exceed_ddof():
    layers = [np.array([[1.0]]), np.array([[tb.PREDICTOR_NODATA_VALUE]])]
    count, _, std = tb.temporal_block_stats(layers, ddof=1)
    assert count.tolist() == [[1]]
    assert np.isnan(std[0, 0])


@pytest.mark.parametrize("n_obs, expected_masked", [(2, True), (3, False), (4, False)])
def test_min_obs_three_masking(n_obs, expected_masked):
    count = np.array([[n_obs]], dtype=np.int32)
    values = np.array([[0.5]])
    out = tb.apply_min_obs(values, count, min_obs=tb.TEMPORAL_MIN_OBS)
    assert bool(np.isnan(out[0, 0])) == expected_masked


def test_min_obs_three_masking_end_to_end(tmp_path):
    """Counts of 2, 3 and 4 in the same block: only the 2 is masked."""
    export_dir = tmp_path / "exports"
    # Column 0 is valid on all 4 dates, column 1 on 3, column 2 on 2.
    presence = {
        "2020-03-20": [True, True, True],
        "2020-04-20": [True, True, True],
        "2020-05-20": [True, True, False],
        "2020-06-01": [True, False, False],
    }
    for index, (date_iso, cols) in enumerate(presence.items()):
        ndvi = np.full((1, 3), tb.PREDICTOR_NODATA_VALUE, dtype="float32")
        for col, present in enumerate(cols):
            if present:
                ndvi[0, col] = 0.1 * (index + 1)
        write_s2_snapshot(export_dir, date_iso, ndvi,
                          transform=Affine(RES, 0.0, ORIGIN_X, 0.0, -RES, ORIGIN_Y))

    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    chosen, _ = tb.select_snapshots_by_date(snapshots)
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    contributors = tb.snapshots_in_window(target, chosen)
    assert len(contributors) == 4

    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S2")
    out_paths = {
        spec.std_band: tmp_path / "out" / "std.tif",
        spec.count_band: tmp_path / "out" / "count.tif",
    }
    result = tb.compute_temporal_bands(target, contributors, resolver, out_paths, ddof=0)
    assert result["status"] == "completed"

    with rasterio.open(out_paths[spec.count_band]) as src:
        assert src.read(1).tolist() == [[4, 3, 2]]
    with rasterio.open(out_paths[spec.std_band]) as src:
        std = src.read(1)
        assert src.descriptions == (spec.std_band,)
        assert src.nodata == tb.PREDICTOR_NODATA_VALUE
    assert std[0, 0] != tb.PREDICTOR_NODATA_VALUE
    assert std[0, 1] != tb.PREDICTOR_NODATA_VALUE
    assert std[0, 2] == tb.PREDICTOR_NODATA_VALUE      # count 2 < min_obs 3
    assert std[0, 0] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4]), abs=1e-6)


def test_too_few_snapshots_in_window_is_recorded_not_computed(tmp_path):
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-05-25", np.full((HEIGHT, WIDTH), 0.2))
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.3))
    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    contributors = tb.snapshots_in_window(target, chosen)
    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S2")
    out_paths = {spec.std_band: tmp_path / "o" / "s.tif", spec.count_band: tmp_path / "o" / "c.tif"}

    result = tb.compute_temporal_bands(target, contributors, resolver, out_paths)
    assert result["status"] == "skipped_min_obs"
    assert result["n_contributing_scenes"] == 2
    assert not out_paths[spec.std_band].exists()


# ---------------------------------------------------------------------------
# S1 coefficient of variation.
# ---------------------------------------------------------------------------

def test_cv_uses_abs_mean_and_is_positive_for_negative_db():
    std = np.array([[2.0]])
    mean = np.array([[-20.0]])
    cv, near_zero = tb.temporal_cv(std, mean)
    assert cv[0, 0] == pytest.approx(0.1)
    assert cv[0, 0] > 0
    assert not near_zero[0, 0]


def test_cv_mean_and_std_come_from_the_identical_valid_set():
    layers = [
        np.array([[-18.0]]),
        np.array([[tb.PREDICTOR_NODATA_VALUE]]),   # excluded from BOTH mean and std
        np.array([[-22.0]]),
        np.array([[-20.0]]),
    ]
    count, mean, std = tb.temporal_block_stats(layers, ddof=0)
    assert count[0, 0] == 3
    assert mean[0, 0] == pytest.approx(-20.0)
    expected_std = np.std([-18.0, -22.0, -20.0])
    assert std[0, 0] == pytest.approx(expected_std)
    cv, _ = tb.temporal_cv(std, mean)
    assert cv[0, 0] == pytest.approx(expected_std / 20.0)


def test_cv_near_zero_mean_policy_masks_instead_of_emitting_inf():
    std = np.array([[1.0, 1.0, 1.0]])
    mean = np.array([[0.0, tb.S1_CV_MIN_ABS_MEAN / 2.0, -1.0]])
    cv, near_zero = tb.temporal_cv(std, mean)
    assert near_zero.tolist() == [[True, True, False]]
    assert np.isnan(cv[0, 0]) and np.isnan(cv[0, 1])
    assert np.isfinite(cv[0, 2])
    # The policy must never let an infinity through.
    assert not np.any(np.isinf(cv))


def test_cv_near_zero_policy_is_one_named_constant():
    std = np.array([[1.0]])
    mean = np.array([[1e-3]])
    strict, _ = tb.temporal_cv(std, mean, min_abs_mean=1.0)
    default, _ = tb.temporal_cv(std, mean, min_abs_mean=tb.S1_CV_MIN_ABS_MEAN)
    assert np.isnan(strict[0, 0])
    assert np.isfinite(default[0, 0])


def test_s1_end_to_end_writes_std_cv_and_count(tmp_path):
    export_dir = tmp_path / "exports"
    values = {"2020-03-20": -18.0, "2020-04-20": -22.0, "2020-05-20": -20.0, "2020-06-01": -20.0}
    for date_iso, vh in values.items():
        write_s1_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), vh))

    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    contributors = tb.snapshots_in_window(target, chosen)
    assert len(contributors) == 4

    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S1")
    out_paths = {
        spec.std_band: tmp_path / "out" / "std.tif",
        spec.cv_band: tmp_path / "out" / "cv.tif",
        spec.count_band: tmp_path / "out" / "count.tif",
    }
    result = tb.compute_temporal_bands(target, contributors, resolver, out_paths, ddof=0)
    assert result["status"] == "completed"
    assert result["n_nonfinite_cv"] == 0

    expected_std = float(np.std(list(values.values())))
    expected_mean = float(np.mean(list(values.values())))
    with rasterio.open(out_paths[spec.std_band]) as src:
        assert src.descriptions == (spec.std_band,)
        assert src.read(1)[0, 0] == pytest.approx(expected_std, abs=1e-5)
    with rasterio.open(out_paths[spec.cv_band]) as src:
        assert src.descriptions == (spec.cv_band,)
        assert src.read(1)[0, 0] == pytest.approx(expected_std / abs(expected_mean), abs=1e-6)
    with rasterio.open(out_paths[spec.count_band]) as src:
        assert src.descriptions == (spec.count_band,)
        assert np.all(src.read(1) == 4)


# ---------------------------------------------------------------------------
# Grid identity.
# ---------------------------------------------------------------------------

def test_grid_mismatch_raises_rather_than_resampling(tmp_path):
    export_dir = tmp_path / "exports"
    shifted = Affine(RES, 0.0, ORIGIN_X + 3.0, 0.0, -RES, ORIGIN_Y)  # sub-pixel offset
    for date_iso in ("2020-03-20", "2020-04-20", "2020-05-20"):
        write_s2_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), 0.2))
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.3), transform=shifted)

    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    contributors = tb.snapshots_in_window(target, chosen)
    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S2")
    out_paths = {spec.std_band: tmp_path / "o" / "s.tif", spec.count_band: tmp_path / "o" / "c.tif"}

    with pytest.raises(tb.GridMismatchError) as excinfo:
        tb.compute_temporal_bands(target, contributors, resolver, out_paths)
    assert "refusing to resample" in str(excinfo.value).lower()
    assert not out_paths[spec.std_band].exists()


def test_assert_grid_identity_names_the_mismatching_file():
    reference = tb.GridSpec.create(tb.EXPORT_CRS, (10, 0, 0, 0, -10, 0), 100, 100)
    other = tb.GridSpec.create(tb.EXPORT_CRS, (10, 0, 0, 0, -10, 0), 100, 99)
    with pytest.raises(tb.GridMismatchError) as excinfo:
        tb.assert_grid_identity([("good.tif", reference), ("bad.tif", other)], context="ctx")
    assert "bad.tif" in str(excinfo.value)
    assert "ctx" in str(excinfo.value)


def test_union_grid_rejects_unaligned_shards():
    a = tb.GridSpec.create(tb.EXPORT_CRS, (10, 0, 0, 0, -10, 0), 10, 10)
    b = tb.GridSpec.create(tb.EXPORT_CRS, (10, 0, 105, 0, -10, 0), 10, 10)  # 10.5 px offset
    with pytest.raises(tb.GridMismatchError):
        tb.union_grid([a, b])


def test_compute_refuses_to_mix_sensors(tmp_path):
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.3))
    write_s1_snapshot(export_dir, "2020-05-20", np.full((HEIGHT, WIDTH), -20.0))
    snapshots = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    target = next(s for s in snapshots if s.sensor == "S2")
    intruder = next(s for s in snapshots if s.sensor == "S1")
    spec = tb.sensor_spec("S2")
    with pytest.raises(tb.BackfillError, match="Refusing to mix sensors"):
        tb.compute_temporal_bands(
            target, [target, intruder, target], lambda s: tmp_path / "nope.tif",
            {spec.std_band: tmp_path / "s.tif", spec.count_band: tmp_path / "c.tif"},
        )


# ---------------------------------------------------------------------------
# Band resolution by description.
# ---------------------------------------------------------------------------

def test_band_resolved_by_description_not_index(tmp_path):
    """Reordering the file must move the resolved index, never silently pick 6."""
    names = list(reversed(tb.S2_LEGACY_PREDICTORS))
    bands = [np.full((2, 2), float(i)) for i in range(len(names))]
    path = write_raster(tmp_path / "reordered.tif", bands, names)
    with rasterio.open(path) as src:
        index = tb.band_index_by_description(src, "NDVI")
    assert index == names.index("NDVI") + 1
    assert index != tb.S2_LEGACY_PREDICTORS.index("NDVI") + 1


def test_missing_band_descriptions_fail_loudly(tmp_path):
    path = tmp_path / "nodesc.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=2, height=2, count=2, dtype="float32",
        crs=tb.EXPORT_CRS, transform=BASE_TRANSFORM, nodata=tb.PREDICTOR_NODATA_VALUE,
    ) as dst:
        dst.write(np.zeros((2, 2), dtype="float32"), 1)
        dst.write(np.zeros((2, 2), dtype="float32"), 2)
    with rasterio.open(path) as src:
        with pytest.raises(tb.BandDescriptionError, match="no complete band descriptions"):
            tb.band_index_by_description(src, "NDVI")


def test_unknown_band_name_fails_loudly(tmp_path):
    path = write_raster(tmp_path / "s.tif", [np.zeros((2, 2))], ["NDVI"])
    with rasterio.open(path) as src:
        with pytest.raises(tb.BandDescriptionError, match="no band named"):
            tb.band_index_by_description(src, "VH_corrected")


# ---------------------------------------------------------------------------
# Outputs: sidecars, VRT band order, read-only guard.
# ---------------------------------------------------------------------------

def test_vrt_band_order_matches_the_schema_and_reads_back(tmp_path):
    export_dir = tmp_path / "exports"
    values = {"2020-03-20": 0.1, "2020-04-20": 0.2, "2020-05-20": 0.3, "2020-06-01": 0.4}
    for date_iso, ndvi in values.items():
        write_s2_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), ndvi))
    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S2")
    out_paths = {
        spec.std_band: tmp_path / "sidecars" / f"{target.prefix}_{spec.std_band}.tif",
        spec.count_band: tmp_path / "sidecars" / f"{target.prefix}_{spec.count_band}.tif",
    }
    tb.compute_temporal_bands(target, tb.snapshots_in_window(target, chosen), resolver, out_paths)

    vrt = tb.write_snapshot_vrt(
        target, {spec.std_band: out_paths[spec.std_band]},
        tmp_path / "vrt" / f"{target.target_prefix()}.vrt",
    )
    assert vrt.name.startswith("winam_s2_predictors_s2_whlev_temporal_v1_")
    with rasterio.open(vrt) as src:
        assert src.count == len(tb.S2_PREDICTORS)
        assert list(src.descriptions) == list(tb.S2_PREDICTORS)
        assert src.nodata == tb.PREDICTOR_NODATA_VALUE
        assert src.read(list(tb.S2_PREDICTORS).index("NDVI") + 1)[0, 0] == pytest.approx(0.4)
        assert np.isfinite(src.read(len(tb.S2_PREDICTORS))[0, 0])


def test_s1_vrt_is_named_under_the_new_prefix(tmp_path):
    export_dir = tmp_path / "exports"
    for date_iso, vh in {"2020-03-20": -18.0, "2020-04-20": -22.0,
                         "2020-05-20": -20.0, "2020-06-01": -20.0}.items():
        write_s1_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), vh))
    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S1")
    out_paths = {
        spec.std_band: tmp_path / "sc" / "std.tif",
        spec.cv_band: tmp_path / "sc" / "cv.tif",
        spec.count_band: tmp_path / "sc" / "count.tif",
    }
    tb.compute_temporal_bands(target, tb.snapshots_in_window(target, chosen), resolver, out_paths)

    vrt = tb.write_snapshot_vrt(
        target,
        {spec.std_band: out_paths[spec.std_band], spec.cv_band: out_paths[spec.cv_band]},
        tmp_path / "vrt" / f"{target.target_prefix()}.vrt",
    )
    # The classifier's BATCH_S1_REQUIRE_SCHEMA_TOKEN is 's1_scc_temporal_v1'.
    assert tb.parse_predictor_export_name(vrt.with_suffix(".tif"))["s1_schema"] == (
        tb.S1_TEMPORAL_SCHEMA_TOKEN
    )
    with rasterio.open(vrt) as src:
        assert list(src.descriptions) == list(tb.S1_PREDICTORS)


def test_rewrite_in_place_produces_a_verified_full_schema_geotiff(tmp_path):
    export_dir = tmp_path / "exports"
    for date_iso, vh in {"2020-03-20": -18.0, "2020-04-20": -22.0,
                         "2020-05-20": -20.0, "2020-06-01": -20.0}.items():
        write_s1_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), vh))
    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S1")
    out_paths = {
        spec.std_band: tmp_path / "sc" / "std.tif",
        spec.cv_band: tmp_path / "sc" / "cv.tif",
        spec.count_band: tmp_path / "sc" / "count.tif",
    }
    tb.compute_temporal_bands(target, tb.snapshots_in_window(target, chosen), resolver, out_paths)

    rewritten = tb.rewrite_snapshot_geotiff(
        target,
        {spec.std_band: out_paths[spec.std_band], spec.cv_band: out_paths[spec.cv_band]},
        tmp_path / "rewrite" / f"{target.target_prefix()}.tif",
    )
    with rasterio.open(rewritten) as src:
        assert src.count == len(tb.S1_PREDICTORS)
        assert list(src.descriptions) == list(tb.S1_PREDICTORS)
        assert src.nodata == tb.PREDICTOR_NODATA_VALUE
        assert src.read(2)[0, 0] == pytest.approx(-20.0)   # VH_corrected passthrough
    # Source files are untouched.
    assert all(p.exists() for p in target.paths)
    assert not list((tmp_path / "rewrite").glob("*.tmp"))


def test_write_targets_inside_the_export_archive_are_refused(tmp_path):
    export_dir = tmp_path / "GEE_Exports_validated_snapshots"
    export_dir.mkdir()
    with pytest.raises(tb.ReadOnlyPathError):
        tb.assert_not_in_readonly_dir(export_dir / "sidecar.tif", [export_dir])
    with pytest.raises(tb.ReadOnlyPathError):
        tb.assert_not_in_readonly_dir(export_dir / "sub" / "deep.tif", [export_dir])
    # A sibling folder is fine.
    assert tb.assert_not_in_readonly_dir(tmp_path / "sidecars" / "x.tif", [export_dir])


def test_extract_source_band_refuses_to_write_into_the_archive(tmp_path):
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.3))
    snapshot = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))[0]
    with pytest.raises(tb.ReadOnlyPathError):
        tb.extract_source_band(snapshot, export_dir / "cache.tif", readonly_dirs=[export_dir])


# ---------------------------------------------------------------------------
# Idempotence, block planning and manifests.
# ---------------------------------------------------------------------------

def test_recompute_is_idempotent(tmp_path):
    export_dir = tmp_path / "exports"
    for date_iso, ndvi in {"2020-04-20": 0.2, "2020-05-20": 0.3, "2020-06-01": 0.4}.items():
        write_s2_snapshot(export_dir, date_iso, np.full((HEIGHT, WIDTH), ndvi))
    chosen, _ = tb.select_snapshots_by_date(
        tb.group_snapshot_files(tb.discover_predictor_files(export_dir))
    )
    target = next(s for s in chosen if s.start_date == dt.date(2020, 6, 1))
    resolver = cache_source_band(tmp_path / "cache", chosen)
    spec = tb.sensor_spec("S2")
    out_paths = {spec.std_band: tmp_path / "o" / "s.tif", spec.count_band: tmp_path / "o" / "c.tif"}
    contributors = tb.snapshots_in_window(target, chosen)

    first = tb.compute_temporal_bands(target, contributors, resolver, out_paths)
    with rasterio.open(out_paths[spec.std_band]) as src:
        before = src.read(1)
    second = tb.compute_temporal_bands(target, contributors, resolver, out_paths)
    with rasterio.open(out_paths[spec.std_band]) as src:
        after = src.read(1)
    assert first["valid_pixels"] == second["valid_pixels"]
    np.testing.assert_array_equal(before, after)


def test_blocks_tile_the_grid_exactly_and_stay_bounded():
    windows = list(tb.plan_blocks(10, 7, (4, 3)))
    covered = np.zeros((7, 10), dtype=int)
    for window in windows:
        covered[
            int(window.row_off):int(window.row_off) + int(window.height),
            int(window.col_off):int(window.col_off) + int(window.width),
        ] += 1
    assert np.all(covered == 1)

    # Block growth respects the byte budget: more layers -> smaller blocks.
    small = tb.choose_block_shape((256, 256), n_layers=64, max_bytes=64 * 1024 * 1024)
    large = tb.choose_block_shape((256, 256), n_layers=2, max_bytes=64 * 1024 * 1024)
    assert small[0] * small[1] <= large[0] * large[1]
    assert small[0] % 256 == 0 and small[1] % 256 == 0


def test_manifest_upsert_is_resumable(tmp_path):
    path = tmp_path / "run_manifest.csv"
    record = {
        "sensor": "S2", "prefix": "p1", "target_prefix": "t1",
        "start_date": "2020-06-01", "end_date": "2020-06-02",
        "n_contributing_scenes": 5, "valid_pixels": 10, "n_nonfinite_cv": 0,
        "status": "completed", "message": "ok", "recorded_at": tb.utc_now_iso(),
    }
    tb.upsert_manifest_record(record, path, tb.RUN_MANIFEST_COLUMNS)
    tb.upsert_manifest_record({**record, "status": "failed", "message": "boom"},
                              path, tb.RUN_MANIFEST_COLUMNS)
    frame = tb.load_manifest(path, tb.RUN_MANIFEST_COLUMNS)
    assert len(frame) == 1
    assert frame.iloc[0]["status"] == "failed"
    assert list(frame.columns) == tb.RUN_MANIFEST_COLUMNS


# ---------------------------------------------------------------------------
# Phase 0 comparison harness.
# ---------------------------------------------------------------------------

def test_compare_to_reference_reports_bias_rmse_and_masking_disagreement():
    local = np.array([[1.0, 2.0, 3.0, tb.PREDICTOR_NODATA_VALUE]])
    reference = np.array([[1.1, 2.1, tb.PREDICTOR_NODATA_VALUE, 4.0]])
    stats = tb.compare_to_reference(local, reference)
    assert stats["n_valid_both"] == 2
    assert stats["n_valid_local_only"] == 1
    assert stats["n_valid_reference_only"] == 1
    assert stats["mean_bias"] == pytest.approx(-0.1)
    assert stats["rmse"] == pytest.approx(0.1)
    assert stats["frac_local_only"] == pytest.approx(0.25)


def test_choose_ddof_prefers_the_better_fit():
    ddof, why = tb.choose_ddof({"rmse": 0.5}, {"rmse": 0.01})
    assert ddof == 1 and "ddof=1" in why
    ddof, why = tb.choose_ddof({"rmse": 0.01}, {"rmse": 0.5})
    assert ddof == 0 and "ddof=0" in why


def test_unvalidated_is_a_distinct_status_from_pass():
    assert tb.VALIDATION_UNVALIDATED == "UNVALIDATED"
    assert tb.VALIDATION_UNVALIDATED != tb.VALIDATION_PASS
    # No S1 ground truth exists, so the harness must return no dates to compare.
    assert tb.aggregate_comparisons([])["n_dates"] == 0


# ---------------------------------------------------------------------------
# Stale Drive mount vs genuinely unreadable file.
#
# Regression cover for a real failure: a whole-archive Phase 0 scan reported all
# 814 multi-file prefixes as 'unreadable' when in fact the Colab Drive FUSE mount
# had gone stale. rasterio's RasterioIOError subclasses OSError but carries
# errno=None and a GDAL message without the transport phrase, so the errno/message
# test alone silently misreports a dead mount as corrupt data.
# ---------------------------------------------------------------------------

def test_errno_107_is_detected_directly_and_when_wrapped():
    stale = OSError(107, "Transport endpoint is not connected")
    assert tb.is_transport_endpoint_error(stale)

    try:
        try:
            raise stale
        except OSError as exc:
            raise RuntimeError("failed while staging a raster") from exc
    except RuntimeError as wrapped:
        assert tb.is_transport_endpoint_error(wrapped)


def test_transport_phrase_without_errno_is_detected():
    assert tb.is_transport_endpoint_error(OSError("Transport endpoint is not connected"))
    assert tb.is_transport_endpoint_error(OSError("Drive FUSE endpoint is stale"))


def test_ordinary_errors_are_not_transport_failures():
    assert not tb.is_transport_endpoint_error(ValueError("nope"))
    assert not tb.is_transport_endpoint_error(FileNotFoundError("missing.tif"))


def test_rasterio_error_carries_no_errno_and_no_transport_phrase(tmp_path):
    """The property that made the errno/message test insufficient."""
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"definitely not a GeoTIFF")
    with pytest.raises(OSError) as excinfo:
        rasterio.open(broken)
    exc = excinfo.value
    assert isinstance(exc, OSError)
    assert getattr(exc, "errno", None) is None
    assert "Transport endpoint is not connected" not in str(exc)
    # ...so the errno/message test alone would not flag it either way.
    assert not tb.is_transport_endpoint_error(exc)


def test_read_error_on_a_vanished_drive_path_is_treated_as_a_stale_mount():
    """A Drive path that no longer stats means the mount died, not bad data."""
    drive_path = Path(tb.DRIVE_MOUNT_PREFIX) / "MyDrive" / "exports" / "scene.tif"

    def _vanished(_path):
        raise OSError(107, "Transport endpoint is not connected")

    exc = rasterio.errors.RasterioIOError(f"{drive_path}: not recognized as a supported file format")
    assert tb.looks_like_stale_mount(exc, drive_path, stat_probe=_vanished)


def test_read_error_on_a_healthy_drive_path_is_a_real_unreadable_file():
    """If the path still stats, the file really is bad and must not be retried."""
    drive_path = Path(tb.DRIVE_MOUNT_PREFIX) / "MyDrive" / "exports" / "scene.tif"
    exc = rasterio.errors.RasterioIOError(f"{drive_path}: not recognized as a supported file format")
    assert not tb.looks_like_stale_mount(exc, drive_path, stat_probe=lambda _p: None)


def test_non_drive_paths_are_never_blamed_on_the_mount(tmp_path):
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"definitely not a GeoTIFF")
    with pytest.raises(OSError) as excinfo:
        rasterio.open(broken)
    assert not tb.looks_like_stale_mount(excinfo.value, broken)


def test_stale_mount_check_ignores_non_oserror():
    drive_path = Path(tb.DRIVE_MOUNT_PREFIX) / "MyDrive" / "x.tif"

    def _vanished(_path):
        raise OSError(107, "Transport endpoint is not connected")

    assert not tb.looks_like_stale_mount(ValueError("bug"), drive_path, stat_probe=_vanished)


def test_classify_prefix_files_propagates_read_errors_for_the_caller_to_classify(tmp_path):
    """classify_prefix_files must not swallow a read failure into a verdict."""
    export_dir = tmp_path / "exports"
    write_s2_snapshot(export_dir, "2020-06-01", np.full((HEIGHT, WIDTH), 0.2),
                      suffix="-0000000000-0000000000")
    corrupt = export_dir / (
        "winam_s2_predictors_s2_whlev_texture_v1_2020-06-01_to_2020-06-02"
        "-0000000000-0000000008.tif"
    )
    corrupt.write_bytes(b"not a GeoTIFF")

    snapshot = tb.group_snapshot_files(tb.discover_predictor_files(export_dir))[0]
    assert len(snapshot.paths) == 2
    with pytest.raises(OSError):
        tb.classify_prefix_files(snapshot)


def test_import_pulls_in_no_earth_engine():
    import sys
    banned = [m for m in sys.modules if m == "ee" or m.startswith("ee.")]
    assert not banned, f"temporal_backfill import leaked Earth Engine modules: {banned}"


# ---------------------------------------------------------------------------
# Settling the ddof for the bulk run.
# ---------------------------------------------------------------------------

def test_settle_ddof_agreeing_sensors():
    ddof, note = tb.settle_ddof([
        {"sensor": "S1", "ddof": 0},
        {"sensor": "S2", "ddof": 0},
    ])
    assert ddof == 0
    assert "S1, S2" in note


def test_settle_ddof_majority_wins():
    ddof, note = tb.settle_ddof([
        {"sensor": "S1", "ddof": 0},
        {"sensor": "S2", "ddof": 0},
        {"sensor": "S3", "ddof": 1},
    ])
    assert ddof == 0
    assert "majority" in note


def test_settle_ddof_tie_goes_to_the_most_decisive_fit():
    """The real 2026-07-25 numbers: S1's fit differs by 270x between the two
    conventions, S2's by 0.16%. S1 must decide, and the old
    max(set(...), key=count) tie-break decided on set iteration order instead."""
    ddof, note = tb.settle_ddof([
        {"sensor": "S2", "ddof": 1, "rmse_by_ddof": {0: 0.012785, 1: 0.012764}},
        {"sensor": "S1", "ddof": 0, "rmse_by_ddof": {0: 0.000260761, 1: 0.0703581}},
    ])
    assert ddof == 0
    assert "S1" in note and "decisive" in note


def test_settle_ddof_tie_is_deterministic_whatever_the_input_order():
    fits = [
        {"sensor": "S2", "ddof": 1, "rmse_by_ddof": {0: 0.012785, 1: 0.012764}},
        {"sensor": "S1", "ddof": 0, "rmse_by_ddof": {0: 0.000260761, 1: 0.0703581}},
    ]
    assert tb.settle_ddof(fits)[0] == tb.settle_ddof(list(reversed(fits)))[0]


def test_settle_ddof_tie_with_no_usable_fit_quality_says_so():
    ddof, note = tb.settle_ddof([
        {"sensor": "S1", "ddof": 0},
        {"sensor": "S2", "ddof": 1},
    ])
    assert ddof == 0
    assert "could not be settled" in note


def test_settle_ddof_carries_a_previous_fit_forward_when_nothing_validates():
    """A run that can validate nothing must not silently flip the convention: the
    outputs already on Drive were written under the earlier fit, and mixing
    ddof=0 and ddof=1 vh_temporal_std_w90 values is undetectable downstream."""
    ddof, note = tb.settle_ddof([], previous_ddof=0)
    assert ddof == 0
    assert "carrying forward" in note


def test_settle_ddof_falls_back_to_the_documented_default_on_a_first_run():
    ddof, note = tb.settle_ddof([], previous_ddof=None)
    assert ddof == tb.TEMPORAL_DDOF
    assert "documented default" in note


def test_settle_ddof_ignores_sensors_without_a_fit():
    ddof, _ = tb.settle_ddof([{"sensor": "S1", "ddof": None}, {"sensor": "S2", "ddof": 1}])
    assert ddof == 1


# ---------------------------------------------------------------------------
# Phase 0 report digest: what an acknowledgement is keyed to.
# ---------------------------------------------------------------------------

def test_digest_ignores_the_session_that_wrote_the_report():
    base = {"recorded_at": "2026-07-27T12:00:00+00:00", "settled_ddof": 0, "verdicts": {}}
    assert (tb.phase0_report_digest({**base, "session_id": "a"})
            == tb.phase0_report_digest({**base, "session_id": "b"}))


def test_digest_is_key_order_independent():
    a = {"recorded_at": "t", "settled_ddof": 0, "verdicts": {"S1": {"status": "PASS"}}}
    b = {"verdicts": {"S1": {"status": "PASS"}}, "settled_ddof": 0, "recorded_at": "t"}
    assert tb.phase0_report_digest(a) == tb.phase0_report_digest(b)


@pytest.mark.parametrize("change", [
    {"settled_ddof": 1},
    {"recorded_at": "2026-07-28T09:00:00+00:00"},
    {"verdicts": {"S1": {"status": "UNVALIDATED"}}},
    {"n_files_discovered": 1},
])
def test_digest_changes_when_the_verdict_changes(change):
    base = {
        "session_id": "same",
        "recorded_at": "2026-07-27T12:00:00+00:00",
        "settled_ddof": 0,
        "n_files_discovered": 3121,
        "verdicts": {"S1": {"status": "PASS"}},
    }
    assert tb.phase0_report_digest(base) != tb.phase0_report_digest({**base, **change})


def test_digest_survives_values_json_cannot_serialise():
    """Reports carry numpy scalars and Paths; the digest must not raise on them."""
    report = {"recorded_at": Path("/x"), "settled_ddof": np.int64(0)}
    assert tb.phase0_report_digest(report)


# ---------------------------------------------------------------------------
# Archive-shrinkage guard.
# ---------------------------------------------------------------------------

def test_archive_intact_when_nothing_is_missing():
    assert tb.assess_archive_shrinkage(3121, 3121)["verdict"] == tb.ARCHIVE_INTACT


def test_archive_intact_when_it_grew():
    result = tb.assess_archive_shrinkage(3200, 3121)
    assert result["verdict"] == tb.ARCHIVE_INTACT
    assert "79 more" in result["message"]


def test_archive_with_no_baseline_is_not_an_error():
    assert tb.assess_archive_shrinkage(1, 0)["verdict"] == tb.ARCHIVE_NO_BASELINE


def test_archive_collapse_is_detected():
    """The 2026-07-27 run: 3121 files recorded, 1 file found, and it proceeded to
    report every sensor UNVALIDATED with nothing to backfill."""
    result = tb.assess_archive_shrinkage(1, 3121)
    assert result["verdict"] == tb.ARCHIVE_COLLAPSED
    assert result["n_missing"] == 3120
    assert "3121" in result["message"]


def test_small_shrink_is_flagged_separately_from_a_collapse():
    result = tb.assess_archive_shrinkage(3100, 3121)
    assert result["verdict"] == tb.ARCHIVE_SHRANK
    assert result["n_missing"] == 21


def test_shrink_threshold_is_configurable():
    assert tb.assess_archive_shrinkage(50, 100, collapse_ratio=0.4)["verdict"] == tb.ARCHIVE_SHRANK
    assert tb.assess_archive_shrinkage(50, 100, collapse_ratio=0.6)["verdict"] == tb.ARCHIVE_COLLAPSED


def test_cached_source_file_count_is_distinct_across_tiled_shards():
    frame = pd.DataFrame([
        {"status": "cached", "source_paths": "/d/a-0-0.tif;/d/a-0-8.tif"},
        {"status": "cached", "source_paths": "/d/b.tif"},
        # A re-cached prefix repeats its sources; they must not be double-counted.
        {"status": "cached", "source_paths": "/d/b.tif"},
        {"status": "failed", "source_paths": "/d/c.tif"},
    ])
    assert tb.count_cached_source_files(frame) == 3


def test_cached_source_file_count_tolerates_an_empty_or_absent_manifest():
    assert tb.count_cached_source_files(None) == 0
    assert tb.count_cached_source_files(pd.DataFrame()) == 0
    assert tb.count_cached_source_files(
        tb.empty_manifest(tb.CACHE_MANIFEST_COLUMNS)) == 0


def test_mtime_iso_reads_a_real_file_and_tolerates_a_missing_one(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("x")
    stamp = tb.mtime_iso(path)
    assert stamp and stamp.endswith("+00:00")
    assert tb.mtime_iso(tmp_path / "gone.csv") is None
