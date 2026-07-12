"""Shoreline distance and near-shore accuracy diagnostics (Workstream B).

Everything here is *diagnostic and additive* to
``Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb`` and runs offline: no
Earth Engine, no Google Drive, no network. Distances are computed with geopandas
in the AOI's metric CRS (**EPSG:32736 / UTM 36S — Lake Victoria**, never
EPSG:27700).

Contents
--------
* ``compute_dist_to_shore`` (B1) — parse ``.geo`` points, reproject, and measure
  distance to the water-mask boundary; used to backfill any frame lacking
  ``dist_to_shore_m`` (S1) and to sanity-check the baked S2 values.
* ``oof_predictions_spatial_cv`` — per-point out-of-fold predictions from the same
  spatial-block grouped-KFold design the notebook uses, so the stratified table is
  genuinely out-of-fold rather than resubstitution.
* ``stratified_accuracy_by_shore`` (B2) — floating-class precision/recall/F1 and the
  LEV<->Floating confusion (both directions) per shore-distance bin, isolating the
  near-shore (0-250 m) band.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# AOI metric CRS: UTM zone 36S over Lake Victoria. NOT EPSG:27700 (that is the UK).
DEFAULT_METRIC_CRS = "EPSG:32736"
DEFAULT_SOURCE_CRS = "EPSG:4326"
FLOATING_CLASS_CODE = 2
LEV_CLASS_CODE = 1


def extract_lonlat_from_geo(geo_value):
    """Return ``(lon, lat)`` from a CSV ``.geo`` cell.

    Faithful port of ``extract_lonlat_from_geo`` (notebook cell 15) so the module
    parses training-point geometry identically without importing the notebook.
    """
    geom = json.loads(geo_value) if isinstance(geo_value, str) else geo_value
    if isinstance(geom, dict):
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"][:2]
            return float(lon), float(lat)
        if geom.get("type") == "Feature":
            lon, lat = geom["geometry"]["coordinates"][:2]
            return float(lon), float(lat)
    coords = geom["coordinates"]
    return float(coords[0]), float(coords[1])


def _load_shoreline_geoseries(shoreline, source_crs=DEFAULT_SOURCE_CRS):
    """Coerce ``shoreline`` into a geopandas GeoSeries in a known CRS.

    Accepts a shapely geometry, a GeoDataFrame/GeoSeries, a GeoJSON dict/FeatureCollection,
    or a path to a GeoJSON file. GeoJSON without an embedded CRS is assumed to be
    lon/lat (``source_crs``), which is how the ``aoi/winam_gulf_water_mask_*.geojson``
    water masks are stored (CRS84).
    """
    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape
    from shapely.geometry.base import BaseGeometry

    if isinstance(shoreline, gpd.GeoSeries):
        gs = shoreline
        return gs if gs.crs is not None else gs.set_crs(source_crs)
    if isinstance(shoreline, gpd.GeoDataFrame):
        gs = shoreline.geometry
        return gs if gs.crs is not None else gs.set_crs(source_crs)
    if isinstance(shoreline, BaseGeometry):
        return gpd.GeoSeries([shoreline], crs=source_crs)
    if isinstance(shoreline, (str, bytes)) or hasattr(shoreline, "__fspath__"):
        return gpd.read_file(shoreline)
    if isinstance(shoreline, dict):
        if shoreline.get("type") == "FeatureCollection":
            geoms = [shapely_shape(f["geometry"]) for f in shoreline["features"]]
        elif shoreline.get("type") == "Feature":
            geoms = [shapely_shape(shoreline["geometry"])]
        else:
            geoms = [shapely_shape(shoreline)]
        return gpd.GeoSeries(geoms, crs=source_crs)
    raise TypeError(f"Unsupported shoreline input type: {type(shoreline)!r}")


def compute_dist_to_shore(
    df,
    shoreline_geom,
    crs=DEFAULT_METRIC_CRS,
    geo_column=".geo",
    source_crs=DEFAULT_SOURCE_CRS,
    to_boundary=True,
):
    """Distance (metres) from each training point to the water-mask shoreline (B1).

    Points are parsed from ``geo_column`` (``.geo``), reprojected to ``crs`` (a
    metric CRS — default EPSG:32736), and measured to the boundary of the water
    mask. ``to_boundary=True`` measures distance to the shoreline itself, so both
    open-water and land points get a non-negative distance and a lake-centre point
    is *further* from shore than a near-shore point.

    Returns a ``pandas.Series`` of distances in metres, indexed like ``df``.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if geo_column not in df.columns:
        raise ValueError(f"Frame is missing the geometry column {geo_column!r}.")

    lonlat = df[geo_column].apply(extract_lonlat_from_geo)
    lons = np.array([t[0] for t in lonlat], dtype=float)
    lats = np.array([t[1] for t in lonlat], dtype=float)
    points = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(lons, lats)], crs=source_crs, index=df.index
    ).to_crs(crs)

    shore = _load_shoreline_geoseries(shoreline_geom, source_crs=source_crs).to_crs(crs)
    shore_geom = shore.union_all() if hasattr(shore, "union_all") else shore.unary_union
    target = shore_geom.boundary if to_boundary else shore_geom

    distances = points.distance(target)
    return pd.Series(np.asarray(distances, dtype=float), index=df.index, name="dist_to_shore_m")


def backfill_dist_to_shore(
    df,
    shoreline_geom,
    crs=DEFAULT_METRIC_CRS,
    dist_column="dist_to_shore_m",
    prefer_baked=True,
    tolerance_m=250.0,
    geo_column=".geo",
    source_crs=DEFAULT_SOURCE_CRS,
    verbose=True,
):
    """Ensure ``df`` has a ``dist_to_shore_m`` column, computing it if absent (B1).

    * If the column is missing (S1 typically lacks it), it is computed and added.
    * If the column exists and ``prefer_baked`` (S2's baked value), the baked value
      is kept, but the geometric distance is computed anyway and a WARNING is
      printed for any point where the two disagree by more than ``tolerance_m``.

    Returns ``(out_df, info)`` where ``info`` records the source and, when a
    comparison was made, the disagreement count and maximum.
    """
    out = df.copy()
    info = {"source": None, "n_compared": 0, "n_disagree": 0, "max_abs_diff_m": np.nan}

    has_col = dist_column in out.columns and pd.to_numeric(out[dist_column], errors="coerce").notna().any()
    computed = compute_dist_to_shore(
        out, shoreline_geom, crs=crs, geo_column=geo_column, source_crs=source_crs
    )

    if has_col and prefer_baked:
        baked = pd.to_numeric(out[dist_column], errors="coerce")
        both = baked.notna() & computed.notna()
        diff = (baked[both] - computed[both]).abs()
        n_disagree = int((diff > tolerance_m).sum())
        info.update(
            source="baked",
            n_compared=int(both.sum()),
            n_disagree=n_disagree,
            max_abs_diff_m=float(diff.max()) if len(diff) else np.nan,
        )
        # Fill only rows where the baked value is missing.
        out[dist_column] = baked.where(baked.notna(), computed)
        if verbose and n_disagree:
            print(
                f"WARNING: baked vs computed dist_to_shore_m disagree by > {tolerance_m:g} m "
                f"at {n_disagree}/{int(both.sum())} point(s) (max {info['max_abs_diff_m']:.0f} m). "
                "Kept the baked S2 values; check the water-mask vintage if this is large."
            )
    else:
        out[dist_column] = computed
        info["source"] = "computed"
        if verbose:
            print(f"Computed dist_to_shore_m for {int(computed.notna().sum())} point(s).")

    return out, info


# ---------------------------------------------------------------------------
# Out-of-fold predictions from the notebook's spatial-block grouped KFold.
# ---------------------------------------------------------------------------

def assign_spatial_blocks_from_lonlat(lons, lats, block_degrees=0.1):
    """Reproduce the notebook's ``assign_spatial_blocks`` block id (cell 27)."""
    lon_bin = np.floor(np.asarray(lons, dtype=float) / block_degrees).astype(np.int64)
    lat_bin = np.floor(np.asarray(lats, dtype=float) / block_degrees).astype(np.int64)
    return (lon_bin * 100000 + lat_bin).astype(np.int64)


def oof_predictions_spatial_cv(
    X,
    y,
    groups,
    model,
    n_splits=5,
    random_state=42,
):
    """Per-point out-of-fold predictions from one grouped-KFold spatial split.

    The notebook's ``evaluate_with_spatial_cv`` aggregates metrics but does not
    expose per-point predictions. This runs a single pass of the same
    block-shuffle assignment it uses (blocks distributed round-robin across
    ``n_splits`` folds), so each point is predicted exactly once by a model that
    never saw its spatial block during training — a genuine out-of-fold label for
    the shoreline stratification (B2).

    Returns an array aligned with ``X``: ``oof[i]`` is the prediction for row ``i``.
    """
    from sklearn.base import clone

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    effective_splits = int(min(n_splits, len(unique_groups)))
    if effective_splits < 2:
        raise ValueError(
            f"Need >=2 spatial blocks for out-of-fold predictions, found {len(unique_groups)}."
        )

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_groups)
    group_to_fold = {g: i % effective_splits for i, g in enumerate(shuffled)}
    fold_assignments = np.array([group_to_fold[g] for g in groups])

    oof = np.full(len(y), -1, dtype=y.dtype if np.issubdtype(y.dtype, np.integer) else np.int64)
    for fold in range(effective_splits):
        test_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        est = clone(model)
        try:
            est.fit(X[train_idx], y[train_idx])
            oof[test_idx] = est.predict(X[test_idx])
        except Exception:
            # Degenerate train fold: fall back to the majority class so the row is
            # still assigned an out-of-fold label instead of crashing.
            vals, counts = np.unique(y[train_idx], return_counts=True)
            oof[test_idx] = vals[int(np.argmax(counts))]
    return oof


# ---------------------------------------------------------------------------
# Shore-distance stratified accuracy (B2).
# ---------------------------------------------------------------------------

def bin_shore_distance(distances, bins=(0, 100, 250, 500, np.inf)):
    """Bin shore distances into left-closed intervals with readable labels."""
    edges = list(bins)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append(f"{int(lo)}+")
        else:
            labels.append(f"{int(lo)}-{int(hi)}")
    cats = pd.cut(
        pd.to_numeric(pd.Series(distances), errors="coerce"),
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return cats, labels, edges


def stratified_accuracy_by_shore(
    df,
    cv_predictions,
    bins=(0, 100, 250, 500, np.inf),
    dist_column="dist_to_shore_m",
    class_column="class",
    sensor_name="Sensor",
    floating_class_code=FLOATING_CLASS_CODE,
    lev_class_code=LEV_CLASS_CODE,
    resubstitution=False,
):
    """Floating-class accuracy and LEV<->Floating confusion per shore band (B2).

    Parameters
    ----------
    df : DataFrame
        Must have ``dist_column`` (shore distance, m) and ``class_column`` (true class).
    cv_predictions : array-like
        Predicted class per row, aligned with ``df`` — ideally the out-of-fold
        predictions from :func:`oof_predictions_spatial_cv`. If they are
        resubstitution predictions, pass ``resubstitution=True`` so the table is
        labelled honestly.
    bins : sequence
        Left-closed shore-distance edges in metres.

    Returns
    -------
    pandas.DataFrame
        One row per bin with support, floating precision/recall/F1, LEV->Floating
        and Floating->LEV confusion counts (and rates), the LEV/Floating support in
        the bin, and the ``prediction_basis`` label. The near-shore rows are the
        point of the exercise; :func:`print_near_shore_summary` isolates them.
    """
    from sklearn.metrics import precision_recall_fscore_support

    work = df.copy()
    work = work.reset_index(drop=True)
    y_true = pd.to_numeric(work[class_column], errors="coerce").to_numpy()
    y_pred = np.asarray(cv_predictions)
    if len(y_pred) != len(work):
        raise ValueError(
            f"cv_predictions length {len(y_pred)} does not match df rows {len(work)}."
        )

    cats, labels, _ = bin_shore_distance(work[dist_column], bins=bins)
    basis = "resubstitution" if resubstitution else "out_of_fold"

    rows = []
    for lo, hi, label in zip(list(bins)[:-1], list(bins)[1:], labels):
        mask = (cats == label).to_numpy()
        support = int(mask.sum())
        yt = y_true[mask]
        yp = y_pred[mask]

        if support:
            p, r, f1, _ = precision_recall_fscore_support(
                yt, yp, labels=[floating_class_code], zero_division=0
            )
            floating_precision, floating_recall, floating_f1 = float(p[0]), float(r[0]), float(f1[0])
            lev_support = int((yt == lev_class_code).sum())
            floating_support = int((yt == floating_class_code).sum())
            lev_as_floating = int(((yt == lev_class_code) & (yp == floating_class_code)).sum())
            floating_as_lev = int(((yt == floating_class_code) & (yp == lev_class_code)).sum())
        else:
            floating_precision = floating_recall = floating_f1 = np.nan
            lev_support = floating_support = 0
            lev_as_floating = floating_as_lev = 0

        rows.append({
            "sensor": sensor_name,
            "shore_bin_m": label,
            "dist_min_m": float(lo),
            "dist_max_m": float(hi),
            "prediction_basis": basis,
            "support": support,
            "lev_support": lev_support,
            "floating_support": floating_support,
            "floating_precision": floating_precision,
            "floating_recall": floating_recall,
            "floating_f1": floating_f1,
            "lev_to_floating": lev_as_floating,
            "floating_to_lev": floating_as_lev,
            "lev_to_floating_rate": (lev_as_floating / lev_support) if lev_support else np.nan,
            "floating_to_lev_rate": (floating_as_lev / floating_support) if floating_support else np.nan,
        })

    return pd.DataFrame(rows)


def print_near_shore_summary(strata_df, near_shore_max_m=250.0, sensor_name=None):
    """Print the near-shore (0-``near_shore_max_m`` m) slice of the strata table (B2)."""
    df = strata_df
    if sensor_name is not None and "sensor" in df.columns:
        df = df[df["sensor"] == sensor_name]
    near = df[df["dist_max_m"] <= near_shore_max_m]
    tag = f" [{sensor_name}]" if sensor_name else ""
    print(f"\nNear-shore (0-{int(near_shore_max_m)} m) classifier performance{tag}:")
    if near.empty:
        print("  (no near-shore bins in the table)")
        return
    basis = near["prediction_basis"].iloc[0] if "prediction_basis" in near.columns else "?"
    for _, row in near.iterrows():
        print(
            f"  {row['shore_bin_m']:>8} m | n={row['support']:>4} | "
            f"floating F1={row['floating_f1']:.3f} "
            f"(P={row['floating_precision']:.3f}, R={row['floating_recall']:.3f}) | "
            f"LEV->Floating={row['lev_to_floating']} "
            f"Floating->LEV={row['floating_to_lev']}"
        )
    print(
        f"  Near-shore is where LEV and floating WH are hardest to separate; the "
        f"LEV<->Floating swaps above ({basis} predictions) quantify that band's error."
    )
