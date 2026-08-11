# Winam Gulf AOI & vector water mask

Vector (GeoJSON) mask of the **Winam Gulf** (a.k.a. Kavirondo Gulf, north-east
arm of Lake Victoria), clipped to the area of interest used throughout the
project notebooks. All geometries are **EPSG:4326 (WGS84 lon/lat)**.

![preview](winam_gulf_mask_preview.png)

## Files

| File | What it is |
|------|------------|
| `winam_gulf_water_mask_occ05.geojson` | **Primary mask** — water where JRC GSW `occurrence ≥ 5 %`, clipped to the AOI. This is the exact "Winam water mask" applied in `Batch_Export.ipynb`. |
| `winam_gulf_water_mask_occ30.geojson` | Stricter variant — `occurrence ≥ 30 %` (permanent water), matching `EE_WATER_OCCURRENCE_THRESHOLD` in `winam_wh_spatial_panel_test_model.ipynb`. |
| `winam_aoi_bbox.geojson` | The AOI rectangle itself, `[34.0, -0.55, 34.9, 0.0]`. |
| **`winam_gulf_planetscope_aoi.geojson`** | **Simplified ordering AOI** — single 1300-vertex polygon wrapping the gulf + adjacent Lake Victoria, for PlanetScope (see below). |
| **`winam_gulf_main_lake_aoi.geojson`** | **Classifier main-lake AOI** — single polygon covering only the main water body (no smaller ponds), with the western edge clipped east to focus on the gulf. Used to restrict classification in `Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb` (see below). |
| `make_winam_mask.py` | Self-contained generator for the masks above (downloads the source tile). |
| `make_planetscope_aoi.py` | Generator for the simplified PlanetScope AOI (reads the `occ30` mask). |
| `make_main_lake_aoi.py` | Generator for the classifier main-lake AOI (reads the `occ05` mask). |
| **`winam_major_rivers.geojson`** | **Mapped river network** — named watercourses draining towards the gulf, each carrying its total mapped course length and its distance to the `occ05` water body. Read by `winam_wh_regional_hierarchical_driver_model.ipynb` §7a-ii to build the `dist_majriver_local_m` covariate. |
| `make_winam_major_rivers.py` | Generator for the river layer (reads the `occ05` mask + the KEN_Rivers shapefile). |
| `winam_gulf_mask_preview.png` / `planetscope_aoi_preview.png` | Previews. |

Each mask is a single dissolved `(Multi)Polygon` feature; lake islands are
encoded as polygon holes.

## How it was derived (provenance)

The notebooks define the AOI and water mask in Earth Engine as:

```python
# Batch_Export.ipynb / Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb
winam = ee.Geometry.Rectangle([34, -0.55, 34.9, 0], geodesic=False)
JRC_OCCURRENCE_THRESHOLD = 5
get_jrc_water_mask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater') \
                       .select('occurrence').gte(JRC_OCCURRENCE_THRESHOLD)

# winam_wh_spatial_panel_test_model.ipynb
AOI_BBOX_WGS84 = (34.0, -0.55, 34.9, 0.0)
EE_WATER_OCCURRENCE_THRESHOLD = 30
```

These vectors reproduce that mask **outside** Earth Engine (no EE credentials
needed) from the published source raster:

- **Dataset:** JRC Global Surface Water v1.4 (Pekel et al., 2016), `occurrence`
  band, 1984–2021 — the same product the `JRC/GSW1_4` EE asset is built from.
- **Tile:** `occurrence_30E_0N` (30–40°E, 0 to −10°N), 0.00025° (~28 m) pixels.
- **Steps:** read the AOI window → `occurrence ≥ threshold` (0–100 %, no-data
  excluded) → polygonise → clip to the AOI rectangle → dissolve.

## Summary stats

| Threshold | Water area in AOI | Parts | Largest part (the gulf body) |
|-----------|------------------:|------:|-----------------------------:|
| `≥ 5 %`   | 2464.2 km²        | 485   | 2451.9 km² (99.5 %) |
| `≥ 30 %`  | 2435.2 km²        | 97    | 2428.1 km² (99.7 %) |

The two thresholds trace essentially the same gulf; the higher threshold just
drops scattered ephemeral specks and thin river channels. The "water area in
AOI" exceeds the gulf alone (~1400 km²) because the AOI's south-west corner
opens onto the main body of Lake Victoria.

## PlanetScope ordering AOI

`winam_gulf_planetscope_aoi.geojson` is a **single, valid `Polygon`** (WGS84,
right-hand rule, **1300 vertices**, under common Planet limits) intended as the
order/clip AOI. It is built from the `occ ≥ 30 %` water mask by keeping the main
connected water body, buffering **+300 m** outward to wrap the shoreline,
filling island holes, simplifying to the vertex budget, and clipping to the
study bbox.

- Area ≈ **2645 km²** (water body + 300 m wrap).
- **Contains 100 %** of the gulf water body (0.0 km² of water falls outside the
  simplified boundary), so nothing is clipped away in the bays.
- The open-lake (south-west) and equator (north) edges are straight because the
  AOI is clipped to the study box `[34, -0.55, 34.9, 0]`.

To change the wrap distance or vertex budget, edit `BUFFER_M` / `VERTEX_BUDGET`
in `make_planetscope_aoi.py` and re-run.

## Classifier main-lake AOI

`winam_gulf_main_lake_aoi.geojson` is the AOI the classifier notebook restricts
to (`AOI_GEOJSON_NAME` in section 10 of
`Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb`). It is built by
`make_main_lake_aoi.py` from the `occ ≥ 5 %` water mask (matching the notebook's
`JRC_OCCURRENCE_THRESHOLD = 5`):

- **Main lake only.** The single largest connected water body is selected
  *before* buffering, so the **484** separate ponds/specks (~12.2 km²) in the AOI
  are dropped rather than merged into the main body.
- **Western edge moved east.** Clipped to lon **34.2046673170698** (lining up
  with the point lat/lon `-0.41757704207237595, 34.2046673170698`), which drops
  the open-lake south-west arm and focuses the AOI on the gulf.
- Single valid `Polygon`, +90 m shoreline collar, island holes filled, 1300
  vertices, area ≈ **1552.6 km²**. It covers **100 %** of the main-lake water
  inside the new bbox while only **0.4 %** of the remaining pond water falls
  inside it.

Edit `AOI_WEST_LON` / `BUFFER_M` / `VERTEX_BUDGET` in `make_main_lake_aoi.py` to
change the western edge, shoreline collar, or vertex budget, then re-run.

## Reproduce

```bash
pip install rasterio shapely numpy pyproj pyshp
python make_winam_mask.py       # auto-downloads the ~52 MB source tile
python make_planetscope_aoi.py  # reads winam_gulf_water_mask_occ30.geojson
python make_main_lake_aoi.py    # reads winam_gulf_water_mask_occ05.geojson
python make_winam_major_rivers.py /path/to/KEN_Rivers/KEN_Rivers.shp
```

> Note: the source GeoTIFF (`occurrence_30E_0N_v1_4_2021.tif`) is not committed;
> it is fetched on demand and is git-ignored.

## The river network

`winam_major_rivers.geojson` exists because the cell-month panel's
`dist_majriver_m` — Earth Engine's HydroSHEDS `RIV_ORD <= 7` cut — is far too
permissive for a water body this small. Over the 5,879 eligible Winam Gulf cells
it spans only 77 m to 3,746 m, so the project's 5 km river-influence length scale
classifies **every** cell as river-influenced and the regional model's
regionalisation collapses to a single region.

Measured instead against the mapped network, the same 5 km cut selects about an
eighth of the gulf, and the covariate spans 0 to ~33 km.

- **Source:** KEN_Rivers (ILRI / OCHA Kenya, `KEN_Rivers-250_polyline`), an ESRI
  shapefile of 13,914 polylines covering Kenya at 1:250,000, EPSG:4326. The
  shapefile itself is **not committed** — pass its path to the generator.
- **Attributes:** the source carries only `NAME`, with no stream-order field, so
  "major" cannot be read off an attribute. The generator instead dissolves
  segments by name and records two response-blind quantities per watercourse:
  `length_km` (the size of the river) and `dist_to_gulf_km` (whether it actually
  reaches the analysed water body). **The file makes no selection itself** — the
  notebook applies its own declared thresholds (`RIVER_MAJOR_MIN_LENGTH_KM`,
  `RIVER_MAJOR_MAX_GULF_DIST_KM`) so the decision lives with every other
  threshold, in one configuration cell, fixed before anything is fitted.
- **At the notebook's defaults** (course ≥ 20 km, within 10 km of the gulf) this
  selects 18 watercourses, among them Nzoia, Migori, Yala, Nyando, Sondu-Miriu,
  Awach Kibuon, Awach Tende, Ombeyi and Kibos — the gulf's documented inflows.
- **Geometry** is clipped to the AOI padded by 0.40°, since only reaches near the
  gulf can be the nearest river to a gulf cell; `length_km` is measured *before*
  that clip, over a wider Lake Victoria basin window, so a long river truncated
  at the window edge is not mistaken for a short one.

### How the notebook gets it

`winam_wh_regional_hierarchical_driver_model.ipynb` fetches this file **from
GitHub over `raw.githubusercontent.com`**, so a fresh Colab runtime needs nothing
staged on Drive and every run reads the same versioned file. The notebook tries,
in order:

1. local paths (`MyDrive/WH_regional_hierarchical_model/`, `MyDrive/`, `aoi/`) —
   so a deliberately modified layer still wins;
2. `RIVER_VECTOR_PATH` on each ref in `RIVER_VECTOR_REFS`, built from
   `RIVER_VECTOR_REPO`.

Whatever is read is validated as a river layer before use — a `404: Not Found`
body or an unrelated FeatureCollection is skipped rather than accepted as a layer
with no qualifying rivers.

**The layer is pinned.** The first entry in `RIVER_VECTOR_REFS` is an immutable
commit SHA rather than a branch name, so the notebook reads the same bytes
whatever happens to `main` afterwards, and keeps working if the branch that
introduced the layer is deleted. `main` follows only as a safety net.

The SHA-256 of the bytes actually read is checked against
`RIVER_LAYER_EXPECTED_SHA256`, printed beside the source, written into the run
manifest (`regionalisation.river_layer`) and asserted in §23. The river network
defines the regions, so a different layer is a different partition — a run made
against unpinned bytes is not comparable with one made against these.

### Changing the layer

Regenerate it, commit it, then update **both** `RIVER_VECTOR_REFS[0]` (to the new
commit) and `RIVER_LAYER_EXPECTED_SHA256` (to the new digest, which
`make_winam_major_rivers.py` prints). Updating one without the other makes §23
fail, which is the intended behaviour — the pin is meant to notice.
