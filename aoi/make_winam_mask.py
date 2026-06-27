#!/usr/bin/env python3
"""
Build a vector (GeoJSON) mask of the Winam Gulf from the JRC Global Surface
Water product, clipped to the AOI defined in the project notebooks.

Reproduces, outside of Earth Engine, the "Winam water mask" used in the
notebooks:

  * AOI                 : ee.Geometry.Rectangle([34, -0.55, 34.9, 0])      (EPSG:4326)
                          == AOI_BBOX_WGS84 = (34.0, -0.55, 34.9, 0.0)
  * Water definition    : JRC/GSW1_4/GlobalSurfaceWater 'occurrence' >= T
                          (Batch_Export.ipynb T=5 ; spatial-panel model T=30)

Source raster: JRC GSW v1.4 (data 1984-2021) occurrence tile 30E_0N,
the exact product the `JRC/GSW1_4` EE asset is derived from.
"""
import json
import math
import os
import shutil
import subprocess
import urllib.request
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import shapes as rio_shapes
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

# JRC GSW v1.4 (data 1984-2021) occurrence tile covering 30-40 E / 0 to -10 N.
# This is the exact product the Earth Engine asset `JRC/GSW1_4` is built from.
TILE = "occurrence_30E_0N_v1_4_2021.tif"
TILE_URL = ("https://storage.googleapis.com/global-surface-water/"
            "downloads2021/occurrence/occurrence_30E_0Nv1_4_2021.tif")
AOI = (34.0, -0.55, 34.9, 0.0)          # xmin, ymin, xmax, ymax  (lon/lat, EPSG:4326)
THRESHOLDS = {5: "occ05", 30: "occ30"}   # Batch_Export=5, spatial-panel model=30
OUTDIR = "."

aoi_geom = box(*AOI)

try:
    from pyproj import Geod
    _GEOD = Geod(ellps="WGS84")
    def area_km2(geom):
        a, _ = _GEOD.geometry_area_perimeter(geom)
        return abs(a) / 1e6
except Exception:
    def area_km2(geom):  # crude fallback if pyproj missing
        latm = (AOI[1] + AOI[3]) / 2.0
        m_per_deg = 111320.0
        return geom.area * (m_per_deg ** 2) * math.cos(math.radians(latm)) / 1e6


def write_fc(path, geom, props):
    feature = {"type": "Feature", "properties": props, "geometry": mapping(geom)}
    fc = {
        "type": "FeatureCollection",
        "name": props.get("name", "winam_gulf_water_mask"),
        # RFC 7946 default CRS is WGS84; named CRS kept for GIS tools that want it explicit.
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [feature],
    }
    with open(path, "w") as f:
        json.dump(fc, f)
    return path


def count_vertices(geom):
    n = 0
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        if g.is_empty:
            continue
        n += len(g.exterior.coords)
        for r in g.interiors:
            n += len(r.coords)
    return n


def download_tile(url, dest, tries=4):
    """Fetch the source tile, preferring curl (robust through proxies),
    falling back to a streamed urllib download. Verifies the byte count."""
    tmp = dest + ".part"
    for attempt in range(1, tries + 1):
        try:
            if shutil.which("curl"):
                subprocess.run(["curl", "-fsS", "--retry", "3", "-o", tmp, url],
                               check=True)
            else:
                with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
            if os.path.getsize(tmp) > 1_000_000:   # sanity: tile is ~52 MB
                os.replace(tmp, dest)
                return
            raise IOError(f"download too small: {os.path.getsize(tmp)} bytes")
        except Exception as e:  # noqa: BLE001
            print(f"  download attempt {attempt} failed: {e}")
    raise RuntimeError(f"could not download {url}")


def main():
    if not os.path.exists(TILE):
        print(f"Downloading source tile -> {TILE} (~52 MB) ...")
        download_tile(TILE_URL, TILE)
    with rasterio.open(TILE) as src:
        win = from_bounds(*AOI, transform=src.transform)
        win = win.round_offsets().round_lengths()
        occ = src.read(1, window=win)
        win_transform = src.window_transform(win)
        nodata = src.nodata
    print(f"AOI window read: {occ.shape[1]} x {occ.shape[0]} px  "
          f"(min={occ.min()}, max={occ.max()})")

    # AOI rectangle itself (the notebooks' AOI geometry)
    write_fc(f"{OUTDIR}/winam_aoi_bbox.geojson", aoi_geom,
             {"name": "winam_aoi_bbox",
              "source": "ee.Geometry.Rectangle([34, -0.55, 34.9, 0])",
              "aoi_bbox_wgs84": list(AOI)})
    print("wrote winam_aoi_bbox.geojson")

    for thr, tag in THRESHOLDS.items():
        # occurrence is 0..100 (%); anything >100 (e.g. 255) is no-data.
        water = ((occ >= thr) & (occ <= 100)).astype("uint8")
        n_water_px = int(water.sum())

        polys = []
        for geom, val in rio_shapes(water, mask=water.astype(bool),
                                    transform=win_transform, connectivity=4):
            if val == 1:
                polys.append(shape(geom))
        n_raw = len(polys)

        dissolved = unary_union(polys)
        # clip to the exact AOI rectangle (window is pixel-aligned, slightly larger)
        clipped = dissolved.intersection(aoi_geom)
        # normalise to MultiPolygon, RFC-7946 winding (exterior CCW)
        if clipped.geom_type == "Polygon":
            clipped = orient(clipped, sign=1.0)
        else:
            from shapely.geometry import MultiPolygon
            clipped = MultiPolygon([orient(g, sign=1.0)
                                    for g in clipped.geoms if g.geom_type == "Polygon"])

        a = area_km2(clipped)
        parts = len(getattr(clipped, "geoms", [clipped]))
        verts = count_vertices(clipped)
        out = f"{OUTDIR}/winam_gulf_water_mask_{tag}.geojson"
        write_fc(out, clipped,
                 {"name": f"winam_gulf_water_mask_{tag}",
                  "source": "JRC/GSW1_4/GlobalSurfaceWater occurrence",
                  "occurrence_threshold_pct": thr,
                  "aoi_bbox_wgs84": list(AOI),
                  "area_km2": round(a, 2)})
        print(f"thr>={thr:>2}%  water_px={n_water_px:>9,}  raw_polys={n_raw:>6,}  "
              f"parts={parts:>5,}  vertices={verts:>7,}  area~{a:8.1f} km^2  -> {out}")


if __name__ == "__main__":
    main()
