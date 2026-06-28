#!/usr/bin/env python3
"""
Build a single-polygon AOI that covers ONLY the main Winam Gulf / Lake Victoria
water body (no smaller ponds) for restricting the classifier to the main lake.

Difference from ``make_planetscope_aoi.py``:
  * Source is the occ>=5 water mask (matches JRC_OCCURRENCE_THRESHOLD = 5 used
    in Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb), not occ>=30.
  * The single largest connected water body is selected *before* buffering, so
    the 484 separate ponds/specks in the AOI cannot be merged into the main body
    by the shoreline buffer -> they are dropped entirely.
  * The western edge is clipped to AOI_WEST_LON (moved east of the study bbox)
    so the open-lake south-west arm is excluded.

Pipeline (derived from the committed JRC GSW water mask, no raster needed):
  occ>=5 water mask -> project to UTM 36S -> keep largest connected body
  -> buffer outward a small collar (wrap shoreline) -> fill island holes
  -> simplify to a vertex budget -> back to WGS84 -> clip to the (west-adjusted)
  study bbox -> single Polygon.
"""
import json
from shapely.geometry import shape, mapping, box, Polygon, MultiPolygon
from shapely.ops import transform as shp_transform
from shapely.geometry.polygon import orient
from pyproj import Transformer, Geod

SRC = "winam_gulf_water_mask_occ05.geojson"   # base water mask (committed)
OUT = "winam_gulf_main_lake_aoi.geojson"
# Western edge moved east to line up with the point (lat, lon) clicked in the
# classifier notebook: (-0.41757704207237595, 34.2046673170698). Only the
# longitude defines a rectangle's western edge.
AOI_WEST_LON = 34.2046673170698
AOI_BBOX = (AOI_WEST_LON, -0.55, 34.9, 0.0)   # xmin(west), ymin, xmax, ymax
BUFFER_M = 90.0           # small shoreline collar (~3 JRC pixels); keeps the
                          # main-lake shoreline without reaching out to ponds
VERTEX_BUDGET = 1300      # keep the output compact
UTM = "EPSG:32736"        # UTM zone 36S, metres, covers the gulf

_to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform
_to_wgs = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform
_geod = Geod(ellps="WGS84")


def drop_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    return MultiPolygon([Polygon(g.exterior) for g in geom.geoms])


def largest(geom):
    if geom.geom_type == "Polygon":
        return geom
    return max(geom.geoms, key=lambda g: g.area)


def nverts(geom):
    n = len(geom.exterior.coords)
    for r in geom.interiors:
        n += len(r.coords)
    return n


def area_km2(geom):
    return abs(_geod.geometry_area_perimeter(geom)[0]) / 1e6


def main():
    fc = json.load(open(SRC))
    water_wgs = shape(fc["features"][0]["geometry"])
    n_parts = len(getattr(water_wgs, "geoms", [water_wgs]))

    # Keep ONLY the main connected water body first, so ponds can never be
    # pulled in by the shoreline buffer, then wrap the shoreline + fill islands.
    main_wgs = largest(water_wgs)
    dropped_area = area_km2(water_wgs) - area_km2(main_wgs)
    main_utm = shp_transform(_to_utm, main_wgs)
    wrapped = main_utm.buffer(BUFFER_M, join_style="round", quad_segs=4)
    wrapped = drop_holes(largest(wrapped))

    bbox_wgs = box(*AOI_BBOX)

    def build(tol):
        s = wrapped.simplify(tol, preserve_topology=True)
        g = shp_transform(_to_wgs, s)
        g = largest(g.intersection(bbox_wgs))      # clip to (west-adjusted) box
        g = orient(Polygon(g.exterior), sign=1.0)   # single ring, RFC-7946 CCW
        return g

    # binary search the simplify tolerance (m) for the largest detail under budget
    lo, hi, best = 1.0, 5000.0, None
    for _ in range(40):
        mid = (lo + hi) / 2
        g = build(mid)
        if nverts(g) > VERTEX_BUDGET:
            lo = mid                # too detailed -> simplify more
        else:
            best, hi = g, mid       # fits -> try to keep more detail
    aoi = best if best is not None else build(hi)

    area = area_km2(aoi)
    feat = {
        "type": "Feature",
        "properties": {
            "name": "winam_gulf_main_lake_aoi",
            "description": "Single-polygon AOI covering only the main Winam Gulf "
                           "/ Lake Victoria water body (no smaller ponds), with "
                           "the western edge clipped east to focus on the gulf.",
            "source": "JRC/GSW1_4 occurrence>=5% water mask, largest body only",
            "buffer_m": BUFFER_M,
            "west_edge_lon": AOI_WEST_LON,
            "clipped_to_bbox": list(AOI_BBOX),
            "ponds_dropped": n_parts - 1,
            "vertices": nverts(aoi),
            "area_km2": round(area, 1),
        },
        "geometry": mapping(aoi),
    }
    out = {"type": "FeatureCollection",
           "name": "winam_gulf_main_lake_aoi",
           "crs": {"type": "name",
                   "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
           "features": [feat]}
    json.dump(out, open(OUT, "w"))
    print(f"source parts = {n_parts}  ->  kept 1 main body, dropped {n_parts - 1} ponds "
          f"(~{dropped_area:.1f} km^2)")
    print(f"vertices = {nverts(aoi)} (budget {VERTEX_BUDGET})")
    print(f"area     = {area:.1f} km^2")
    print(f"valid    = {aoi.is_valid}, type = {aoi.geom_type}")
    print(f"bounds   = {tuple(round(x, 4) for x in aoi.bounds)}")
    print(f"wrote    = {OUT}")


if __name__ == "__main__":
    main()
