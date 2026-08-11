#!/usr/bin/env python3
"""
Build a vector (GeoJSON) layer of the rivers draining into the Winam Gulf, for
use as the river-distance covariate in the regional hierarchical driver model.

WHY THIS EXISTS
---------------
The 500 m cell-month panel carries `dist_majriver_m`, the distance to a
"major river" defined in Earth Engine as HydroSHEDS `RIV_ORD <= 7`. That order
cut is far too permissive for a water body this small: over the 5,879 eligible
Winam Gulf cells it produces distances of 77 m to 3,746 m, so a 5 km
river-influence threshold classifies EVERY cell as river-influenced and the
regionalisation collapses to a single region.

This script builds the alternative from a mapped river network — named
watercourses at 1:250,000 — so that "distance to a major river" varies across
the gulf by an order of magnitude rather than by a factor of fifty.

SELECTION RULE (response-blind by construction)
-----------------------------------------------
Every named watercourse in the window is written out, with two properties that
let the notebook apply its own declared thresholds:

  * `length_km`      - the river's total mapped course, dissolved across all
                       segments sharing a NAME within LENGTH_WINDOW. This is a
                       proxy for the size of the river, which the source has no
                       order attribute for.
  * `dist_to_gulf_km`- distance from that course to the analysed water body
                       (`winam_gulf_water_mask_occ05.geojson`). A river that
                       never approaches the gulf does not discharge into it.

Geometry is clipped to a window around the AOI, because only the reaches near
the gulf can ever be the nearest river to a gulf cell. `length_km` is measured
BEFORE that clip — over the wider LENGTH_WINDOW — so a long river truncated at
the window edge is not mistaken for a short one. The length window stops short
of a national dissolve because river NAMEs repeat across Kenya.

Nothing here reads WH cover, prevalence, residuals or model output. The layer
is a function of river geometry and the water mask alone.

SOURCE
------
KEN_Rivers (ILRI / OCHA Kenya, `KEN_Rivers-250_polyline`, ESRI shapefile,
EPSG:4326, 13,914 polyline features covering Kenya). Not redistributed here:
pass the path to the unpacked shapefile.

USAGE
-----
    python3 aoi/make_winam_major_rivers.py /path/to/KEN_Rivers/KEN_Rivers.shp

Requires `pyshp`, `shapely` and `pyproj` (no GDAL, no Earth Engine).
"""
import hashlib
import json
import sys
from pathlib import Path

import shapefile                      # pyshp
from pyproj import Transformer
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import transform, unary_union

HERE = Path(__file__).resolve().parent

# The AOI rectangle used throughout the project notebooks.
AOI = (34.0, -0.55, 34.9, 0.0)
# Rivers outside this window can never be the nearest river to a gulf cell.
# 0.40 deg ~ 44 km, comfortably wider than the gulf's own half-width.
CLIP_PAD_DEG = 0.40
# `length_km` must describe the WHOLE river, not the part that survives the clip,
# otherwise a long river truncated at the window edge looks like a short one.
# It is measured over this wider window instead of nationally, because river
# NAMEs repeat across Kenya and a national dissolve would weld unrelated
# watercourses together. This window covers the Kenyan Lake Victoria basin.
LENGTH_WINDOW = (33.8, -1.4, 35.6, 0.9)
# The analysed water body, from this directory.
WATER_MASK = HERE / "winam_gulf_water_mask_occ05.geojson"
OUT = HERE / "winam_major_rivers.geojson"
# Metric CRS for every length and distance below: UTM 36S, the CRS the regional
# notebook dissolves its region polygons in.
METRIC_CRS = "EPSG:32736"

# Only named watercourses shorter than this are dropped outright, to keep the
# file small. The notebook applies the REAL thresholds (declared in its §3c);
# this is just a floor so the layer does not carry thousands of 1 km ditches.
KEEP_MIN_LENGTH_KM = 5.0


def main(shp_path):
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise SystemExit(f"river shapefile not found: {shp_path}")
    if not WATER_MASK.exists():
        raise SystemExit(f"water mask not found: {WATER_MASK}")

    to_m = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform
    clip = box(AOI[0] - CLIP_PAD_DEG, AOI[1] - CLIP_PAD_DEG,
               AOI[2] + CLIP_PAD_DEG, AOI[3] + CLIP_PAD_DEG)

    mask_fc = json.loads(WATER_MASK.read_text())
    mask = unary_union([shape(f["geometry"]) for f in mask_fc["features"]])
    mask_m = transform(to_m, mask)

    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    if "NAME" not in fields:
        raise SystemExit(f"expected a NAME field, found {fields}")
    name_at = fields.index("NAME")

    # Dissolve segments by name over the LENGTH window, so `length_km` describes
    # the whole river; the clip below decides only which geometry is written.
    lw = box(*LENGTH_WINDOW)
    by_name = {}
    for sh, rec in zip(reader.shapes(), reader.records()):
        if len(sh.points) < 2:
            continue
        name = str(rec[name_at] or "").strip()
        if not name:
            continue                      # unnamed watercourses carry no size signal
        xmin, ymin, xmax, ymax = sh.bbox
        if (xmax < lw.bounds[0] or xmin > lw.bounds[2]
                or ymax < lw.bounds[1] or ymin > lw.bounds[3]):
            continue
        by_name.setdefault(name, []).append(LineString(sh.points))

    features, skipped = [], 0
    for name, lines in sorted(by_name.items()):
        full_m = unary_union([transform(to_m, ln) for ln in lines])
        length_km = full_m.length / 1000.0
        if length_km < KEEP_MIN_LENGTH_KM:
            skipped += 1
            continue
        clipped = unary_union(lines).intersection(clip)
        if clipped.is_empty:
            skipped += 1                  # whole course lies outside the AOI window
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "name": name,
                # the river's size, measured on its FULL mapped course
                "length_km": round(length_km, 3),
                # how close that course comes to the analysed water body
                "dist_to_gulf_km": round(full_m.distance(mask_m) / 1000.0, 3),
                "n_segments": len(lines),
            },
            "geometry": mapping(clipped),
        })

    features.sort(key=lambda f: -f["properties"]["length_km"])
    fc = {
        "type": "FeatureCollection",
        "name": "winam_major_rivers",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "properties": {
            "source": "KEN_Rivers (ILRI / OCHA Kenya, KEN_Rivers-250_polyline)",
            "source_scale": "1:250,000",
            "length_window_wgs84": list(LENGTH_WINDOW),
            "selection": (
                "named watercourses only, dissolved by NAME, total mapped "
                f"course >= {KEEP_MIN_LENGTH_KM} km; geometry clipped to the AOI "
                f"padded by {CLIP_PAD_DEG} deg"),
            "length_measured": ("on the full mapped course within "
                                "length_window_wgs84, before clipping"),
            "dist_to_gulf_reference": WATER_MASK.name,
            "metric_crs": METRIC_CRS,
            "response_blind": True,
            "note": ("length_km and dist_to_gulf_km are carried so the notebook "
                     "can apply its own declared major-river thresholds; this "
                     "file makes no such decision itself"),
        },
        "features": features,
    }
    OUT.write_text(json.dumps(fc))

    print(f"{len(by_name):,} named watercourses in the window; "
          f"{len(features):,} kept, {skipped:,} below "
          f"{KEEP_MIN_LENGTH_KM} km or outside the clip.")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"sha256 {digest}")
    print("  -> commit this file, then set RIVER_LAYER_EXPECTED_SHA256 to that "
          "digest\n     and RIVER_VECTOR_REFS[0] to the commit SHA, in the "
          "regional notebook's §3c.")
    print("\nLongest courses that reach within 1 km of the analysed water body:")
    near = [f["properties"] for f in features
            if f["properties"]["dist_to_gulf_km"] <= 1.0][:15]
    for p in near:
        print(f"  {p['name']:20s} {p['length_km']:8.1f} km  "
              f"{p['dist_to_gulf_km']:6.2f} km from the gulf")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-3])
    main(sys.argv[1])
