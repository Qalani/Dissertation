#!/usr/bin/env python3
"""
Build a simplified single-polygon AOI for PlanetScope ordering that wraps around
the Winam Gulf and the part of Lake Victoria inside the study box.

Pipeline (derived from the committed JRC GSW water mask, no raster needed):
  occ>=30 water mask -> project to UTM 36S -> buffer outward (wrap shoreline)
  -> keep largest connected body -> fill island holes -> simplify to a vertex
  budget -> back to WGS84 -> clip to the study bbox -> single Polygon.
"""
import json
import numpy as np
from shapely.geometry import shape, mapping, box, Polygon, MultiPolygon
from shapely.ops import transform as shp_transform, unary_union
from shapely.geometry.polygon import orient
from pyproj import Transformer, Geod

SRC = "winam_gulf_water_mask_occ30.geojson"   # base water mask (committed)
OUT = "winam_gulf_planetscope_aoi.geojson"
AOI_BBOX = (34.0, -0.55, 34.9, 0.0)
BUFFER_M = 300.0          # outward wrap around the water (metres)
VERTEX_BUDGET = 1300      # stay comfortably below the 1500 cap
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


def main():
    fc = json.load(open(SRC))
    water_wgs = shape(fc["features"][0]["geometry"])
    water_utm = shp_transform(_to_utm, water_wgs)

    # wrap the water: buffer out, merge nearby parts, keep the main body, fill islands
    wrapped = water_utm.buffer(BUFFER_M, join_style="round", quad_segs=4)
    wrapped = drop_holes(largest(wrapped))

    bbox_wgs = box(*AOI_BBOX)

    def build(tol):
        s = wrapped.simplify(tol, preserve_topology=True)
        g = shp_transform(_to_wgs, s)
        g = largest(g.intersection(bbox_wgs))     # clip to study box
        g = orient(Polygon(g.exterior), sign=1.0)  # single ring, RFC-7946 CCW
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

    area = abs(_geod.geometry_area_perimeter(aoi)[0]) / 1e6
    feat = {
        "type": "Feature",
        "properties": {
            "name": "winam_gulf_planetscope_aoi",
            "description": "Simplified AOI wrapping the Winam Gulf + adjacent "
                           "Lake Victoria, for PlanetScope ordering.",
            "source": "JRC/GSW1_4 occurrence>=30% water mask, largest body",
            "buffer_m": BUFFER_M,
            "clipped_to_bbox": list(AOI_BBOX),
            "vertices": nverts(aoi),
            "area_km2": round(area, 1),
        },
        "geometry": mapping(aoi),
    }
    out = {"type": "FeatureCollection",
           "name": "winam_gulf_planetscope_aoi",
           "crs": {"type": "name",
                   "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
           "features": [feat]}
    json.dump(out, open(OUT, "w"))
    print(f"vertices = {nverts(aoi)} (budget {VERTEX_BUDGET}, cap 1500)")
    print(f"area     = {area:.1f} km^2  (water mask was ~2435 km^2)")
    print(f"valid    = {aoi.is_valid}, type = {aoi.geom_type}")
    print(f"bounds   = {tuple(round(x,4) for x in aoi.bounds)}")
    print(f"wrote    = {OUT}")


if __name__ == "__main__":
    main()
