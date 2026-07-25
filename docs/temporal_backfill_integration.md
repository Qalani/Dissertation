# Classifier integration for the locally backfilled temporal bands

`Backfill_Temporal_Bands_Local.ipynb` reconstructs the snapshot-relative 90-day
temporal-persistence bands for every already-exported snapshot, without any Earth
Engine calls. This document describes **precisely what would change** in
`Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb` to consume them.

**No change has been made to the classifier notebook.** Everything below is a
specification, not an applied diff.

---

## 1. What the backfill produces

Written to `MyDrive/Winam_Temporal_Backfill/`, deliberately **not** into
`GEE_Exports_validated_snapshots`:

```
Winam_Temporal_Backfill/
  source_band_cache/{S2,S1}/{source_prefix}__{NDVI|VH_corrected}.tif
  sidecars/{S2,S1}/{source_prefix}_{band_name}.tif
  vrt/{S2,S1}/{new_prefix}.vrt
  rewritten/{S2,S1}/{new_prefix}.tif          # only if REWRITE_IN_PLACE
  reports/…                                    # manifests, Phase 0 report
```

Sidecars are named after the **source** prefix so provenance is visible on disk;
VRTs are named under the **new** prefix so the classifier's schema-token gate
passes. Per sensor:

| Sensor | Sidecar bands | Count band |
|---|---|---|
| S2 | `ndvi_temporal_std_w90` | `ndvi_temporal_count_w90` |
| S1 | `vh_temporal_std_w90`, `vh_temporal_cv_w90` | `vh_temporal_count_w90` |

Every sidecar is single-band float32, NoData `-9999`, cloud-optimised, on the same
grid as its source, with its band description set to exactly the schema band name.

The count band is **not** part of either predictor schema and is not in the VRT. It
exists so the obvious robustness check — does the result depend on observation
density? — is possible. This matters most for S1, where swath coverage makes
per-pixel counts strongly spatially structured.

### Why the export folder is left alone

`Batch_Export.ipynb`'s `export_prefix_exists` globs `{prefix}*` in the export dir, and
`discover_exported_predictor_sets` accepts any file whose stem parses as a predictor
export. A sidecar named `…_to_2020-01-02_ndvi_temporal_std_w90.tif` would not parse (so
it would be ignored by the classifier), but `export_prefix_exists` uses a `{prefix}-*`
/ `{prefix}` glob that a sidecar **would** match, which would make
`SKIP_PREFIXES_ALREADY_IN_DRIVE` believe a date had been exported when it had not.
Keeping sidecars in a separate folder avoids corrupting that resume state.

---

## 2. `_corr_discover_predictor_tifs` would accept `.vrt`

Current implementation (section 2c of the classifier):

```python
if path.suffix.lower() not in {'.tif', '.tiff', ''}:
    continue
```

The change is to add `.vrt` to the accepted suffixes and to search a second,
configurable directory:

```python
BACKFILL_VRT_DIR = Path('/content/drive/MyDrive/Winam_Temporal_Backfill/vrt')
PREDICTOR_SUFFIXES = {'.tif', '.tiff', '.vrt', ''}

def _corr_discover_predictor_tifs(prefix, export_dir, extra_dirs=()):
    search_dirs = [Path(export_dir)] + [Path(d) for d in extra_dirs]
    paths = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if not path.is_file():
                continue
            if path.suffix.lower() not in PREDICTOR_SUFFIXES:
                continue
            if _CORR_DRIVE_COLLISION_RE.search(path.name):
                continue
            stem = path.stem
            if stem != prefix and not stem.startswith(prefix + '-'):
                continue
            paths.append(path)
    ...
```

Three constraints on this change:

1. **A `.vrt` and a real `.tif` for the same prefix must never both be returned.**
   They describe the same scene, so returning both would double-sample every point in
   `_corr_sample_points` (its `drop_duplicates(subset=['_manual_index'], keep='first')`
   masks this for corrections but not for the batch path). Prefer a real GeoTIFF when
   one exists and fall back to the VRT:

   ```python
   real = [p for p in paths if p.suffix.lower() != '.vrt']
   return real if real else paths
   ```

2. **Sub-directory layout.** VRTs live under `vrt/{S2,S1}/`, so `extra_dirs` must be the
   per-sensor folder, or the scan must recurse one level. Passing the sensor folder
   explicitly is simpler and keeps the S1/S2 separation that the rest of the notebook
   relies on.

3. **`_as_geotiff_list`** (section on classification writing) sorts by name and checks
   `p.exists()`. A VRT satisfies both unchanged. No edit needed there.

---

## 3. `staged_drive_read` must stage sources *and* sidecars for a VRT

This is the subtle one, and getting it wrong turns a performance optimisation into a
correctness bug.

`staged_drive_read` copies a Drive path to local scratch and yields the local path, so
block I/O never crosses the FUSE mount:

```python
local = _local_scratch_path(path.name)
_copy_whole_file_through_drive(path, local, …)
yield local
```

A `.vrt` is a few kilobytes of XML that **references other files by absolute path**.
Staging only the VRT therefore:

- copies almost nothing (so the optimisation buys nothing), and
- leaves every pixel read going to the original Drive paths inside the XML — i.e. block
  I/O over FUSE, which is exactly what staging exists to avoid.

It still *works* (the paths resolve while Drive is mounted), but it is slow, and it
breaks outright if the Drive endpoint goes stale mid-read, because the retry logic in
`_run_with_drive_remount_retry` is wrapped around the staged copy, not the inner reads.

The fix is to make staging VRT-aware: stage every referenced source, then rewrite the
XML to point at the local copies.

```python
import xml.etree.ElementTree as ET

@contextmanager
def staged_drive_read(path):
    path = Path(path)
    if path.suffix.lower() == '.vrt':
        with _staged_vrt(path) as local_vrt:
            yield local_vrt
        return
    ...  # existing behaviour unchanged


@contextmanager
def _staged_vrt(vrt_path):
    """Stage a VRT and every raster it references, rewriting the paths."""
    tree = ET.parse(vrt_path)
    root = tree.getroot()
    with ExitStack() as stack:
        for node in root.iter('SourceFilename'):
            source = Path(node.text)
            if not source.is_absolute():
                source = (vrt_path.parent / source).resolve()
            local = stack.enter_context(staged_drive_read(source))
            node.text = str(local)
            node.set('relativeToVRT', '0')
        local_vrt = _local_scratch_path(vrt_path.name)
        tree.write(local_vrt)
        try:
            yield local_vrt
        finally:
            _safe_remove(local_vrt)
```

Notes:

- The backfill writes `SourceFilename` with `relativeToVRT="0"` and absolute paths, so
  the `is_absolute()` branch is defensive only.
- The recursion is safe: staged sources are plain GeoTIFFs, which take the existing
  branch.
- Inside a `drive_stage_cache()` scope the sources are cached and reused across the
  dataset exactly as GeoTIFFs are today. **This is the main win** — for S2 the 1.29 GB
  source is staged once and reused by the classification, area, confidence and quicklook
  passes.
- `staged_drive_write` needs no change: the classifier never writes VRTs.

---

## 4. How the S1 prefix change is handled in discovery

`parse_predictor_export_name` **already handles both S1 forms** (it was updated when the
schema was bumped) and needs no change:

```
winam_s1_scc_temporal_v1_YYYY-MM-DD_to_YYYY-MM-DD    (current, no _predictors_)
winam_s1_scc_predictors_YYYY-MM-DD_to_YYYY-MM-DD     (legacy)
```

The asymmetry is the thing to keep in mind everywhere else:

| | S2 | S1 |
|---|---|---|
| Old | `winam_s2_predictors_s2_whlev_texture_v1_…` | `winam_s1_scc_predictors_…` |
| New | `winam_s2_predictors_s2_whlev_temporal_v1_…` | `winam_s1_scc_temporal_v1_…` |

S2 kept `winam_s2_predictors_` and swapped the token. **S1 dropped `predictors`
entirely.** Any code that derives a new prefix from an old one by string substitution
of the token will silently produce `winam_s1_scc_predictors_s1_scc_temporal_v1_…`,
which matches nothing. Derive prefixes from the templates instead —
`winam_diagnostics.temporal_backfill.SensorSpec.temporal_prefix()` is the single source
of truth, and it mirrors `build_export_manifest` in `Batch_Export.ipynb` section 7.

Because the backfill names S1 VRTs `winam_s1_scc_temporal_v1_{start}_to_{end}.vrt`,
`parse_predictor_export_name` returns `s1_schema == 's1_scc_temporal_v1'` for them and
the existing gate in `discover_exported_predictor_sets` passes them unchanged:

```python
if (parsed['sensor'] == 'S1' and BATCH_S1_REQUIRE_SCHEMA_TOKEN is not None
        and parsed.get('s1_schema') != BATCH_S1_REQUIRE_SCHEMA_TOKEN):
    ...skip...
```

One real consequence: for a backfilled S1 date **both** the legacy
`winam_s1_scc_predictors_…tif` and the new `winam_s1_scc_temporal_v1_….vrt` exist. They
group under different prefixes, so `discover_exported_predictor_sets` produces two rows
for the same acquisition date — the legacy one skipped as `skipped_legacy_s1_schema`,
the VRT accepted. That is the desired behaviour and needs no extra deduplication, but
the inventory CSV will show both, and any downstream count of "S1 dates available"
should filter on status rather than counting rows.

---

## 5. Schema-token constants for backfilled dates

**No constant changes.** The backfill is designed so the current values stay correct:

```python
BATCH_S2_REQUIRE_SCHEMA_TOKEN = 's2_whlev_temporal_v1'
BATCH_S1_REQUIRE_SCHEMA_TOKEN = 's1_scc_temporal_v1'
```

A backfilled snapshot is presented under the new-schema prefix and its VRT carries all
21 (S2) / 5 (S1) band descriptions in exactly `S2_PREDICTORS` / `S1_PREDICTORS` order,
verified at write time. So:

- `_validate_s2_predictor_band_order` passes unchanged — it reads `src.descriptions` and
  compares to `S2_PREDICTORS`, and a VRT reports its `<Description>` elements there.
- `_source_band_descriptions` returns a complete list, so it never falls back.
- The `src.count != len(predictors)` check in `_write_model_classification_single`
  passes.
- `S2_CORRECTION_PREDICTORS` (section 2c) already lists 21 bands ending in
  `ndvi_temporal_std_w90`; it is only a fallback for rasters with no descriptions, which
  a backfill VRT never is.

The one genuinely new constant is where to find the VRTs:

```python
BACKFILL_ROOT = Path('/content/drive/MyDrive/Winam_Temporal_Backfill')
BACKFILL_VRT_DIRS = {'S2': BACKFILL_ROOT / 'vrt' / 'S2',
                     'S1': BACKFILL_ROOT / 'vrt' / 'S1'}
USE_BACKFILLED_TEMPORAL_BANDS = True   # False restores export-folder-only discovery
```

### Provenance in the outputs

Backfilled dates are reconstructed locally and are **not** byte-identical to what Earth
Engine would have written (section 7). Any results table that mixes them with genuine
Earth Engine exports should carry that distinction. The cheapest way is to read the
`reconstruction` tag the backfill writes into every sidecar and rewritten GeoTIFF:

```python
with rasterio.open(path) as src:
    is_backfilled = 'local backfill' in (src.tags().get('reconstruction') or '')
```

or simply to join on the backfill run manifest
(`Winam_Temporal_Backfill/reports/winam_temporal_backfill_run_manifest.csv`), whose
`message` column carries the per-sensor Phase 0 validation status on every row.

---

## 6. Known issues preserved faithfully, not fixed

Earth Engine's behaviour is reproduced exactly even where it is questionable;
consistency with the existing schema matters more than correctness at this stage. These
are flagged here rather than silently changed.

### 6.1 `vh_temporal_cv_w90` is not scale-free

`add_s1_temporal_stability` justifies dividing by `abs(mean)` like this:

> VH_corrected is in dB and negative over water, so a raw std/mean CV would flip sign;
> divide by the magnitude of the mean for a positive, scale-free coefficient of
> variation.

The sign argument is right; the scale-free claim is not. A coefficient of variation is
scale-free for a **ratio-scale** quantity, where multiplying the data by `k` multiplies
both the mean and the standard deviation by `k` and leaves the ratio unchanged. dB is a
logarithmic **interval** scale: a multiplicative change in linear backscatter
(`sigma0 -> k * sigma0`) is an *additive* shift in dB (`+10 log10 k`). That leaves the
standard deviation unchanged while moving the mean, so the ratio changes. Concretely, a
window with std 1.0 dB has CV 0.05 at a mean of −20 dB and CV 0.0667 at −15 dB, for
identical temporal variability.

Practical consequences for anyone using the band:

- It is not comparable across regions with different mean backscatter.
- It is monotonically related to `1/|mean|` at fixed std, so it partly encodes mean
  backscatter — information already carried by `VH_corrected` and `VH_smooth`. Expect
  collinearity.
- `vh_temporal_std_w90` alone is the interpretable variability measure.

Reproduced as written. If it is ever fixed, both the export and this backfill must
change together and the schema token must be bumped again.

### 6.2 The S1 stack mixes orbit geometries

`get_s1_source_collection` filters:

```python
.filter(ee.Filter.eq('instrumentMode', 'IW'))
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
```

It does **not** filter `orbitProperties_pass` (ASCENDING/DESCENDING) or
`relativeOrbitNumber_start`. A 90-day window therefore mixes ascending and descending
passes and several relative orbits. `add_s1_predictors_exact_scc` normalises incidence
angle to 38° via a `cos²` correction, which addresses angle but not **look direction**:
for a given target, ascending and descending passes see different local geometry, and
over water the wind/wave-driven backscatter is direction-sensitive.

`vh_temporal_std_w90` therefore conflates real temporal change with acquisition
geometry, and the mixture is not stable through time (Sentinel-1B failed in December
2021, which changed both the revisit interval and the ascending/descending balance mid
archive). This is present in the Earth Engine version too; it is reproduced, not fixed.

If it is ever addressed, the defensible options are to filter to a single orbit
direction (halving the observation count, which interacts badly with the min-obs-3
mask), or to compute the statistic within orbit direction and combine — both are schema
changes, not backfill changes.

### 6.3 S1 spatial coverage is uneven

The notebook already carries a diagnostic for eastern-AOI swath coverage
(`Batch_Export.ipynb` section 2a). Per-pixel observation counts vary far more across the
AOI for S1 than for S2, so the min-obs-3 mask has strong spatial structure: parts of the
AOI have a defined `vh_temporal_std_w90` on almost every date, and parts have it only
when two swaths happen to overlap in the window.

This is the main reason the count band is retained as an output. Any model using the S1
temporal bands should check whether its skill varies with
`vh_temporal_count_w90`, because a spatially structured mask is easily mistaken for a
spatially structured signal.

---

## 7. The near-zero-mean policy for `vh_temporal_cv_w90`

`vh_temporal_cv_w90 = std / abs(mean)` is undefined as `abs(mean) -> 0`. Earth Engine's
behaviour there **cannot currently be observed**, because `EXPORT_S1` is `False` and no
S1 snapshot has ever been exported with the temporal bands, so there is nothing in Drive
to compare against — and the backfill notebook must not export one to find out.

A policy therefore had to be chosen rather than measured:

> **Pixels whose `abs(mean)` falls below `S1_CV_MIN_ABS_MEAN` (default `1e-6`) are
> written as NoData (`-9999`), not as ±Inf.**

Rationale: the classifier's `_predictor_valid_mask` is

```python
finite = np.all(np.isfinite(stack), axis=0)
not_nodata = np.all(stack != predictor_nodata_value, axis=0)
return finite & not_nodata
```

so a non-finite value in `vh_temporal_cv_w90` already invalidates the **whole pixel
across all five S1 bands**. Emitting Inf and masking therefore have the same downstream
effect on classification, but masking makes it explicit and countable instead of letting
infinities propagate through any intermediate arithmetic.

Affected pixels are recorded per date in the run manifest column `n_nonfinite_cv`.

It is a **single named constant** in `winam_diagnostics/temporal_backfill.py`,
changeable in one place if a reference S1 export with Earth Engine's own temporal bands
ever becomes available. In practice the threshold should almost never bind:
`VH_corrected` over water is around −20 dB, so a 90-day window mean within 1e-6 of zero
implies something already pathological.

---

## 8. Where the backfill genuinely differs from Earth Engine

Two differences are structural and cannot be removed by any amount of local computation.
Both are quantified by Phase 0 of the backfill notebook rather than assumed away.

1. **Fewer observations in every window.** Earth Engine reduced over every scene passing
   the source-collection filters. The local stack sees only snapshots that passed the
   export coverage gate and reached Drive. `winam_snapshot_validated_predictor_manifest.csv`
   gives the `queued`/`completed` vs `skipped_low_coverage` split, and Phase 0 section 3b
   reports it per sensor. Fewer observations means a noisier standard deviation and more
   pixels falling below min-obs-3.

2. **One observation per acquisition date, not per granule.** `get_s2_collection` maps
   over individual granules, so Earth Engine's `stdDev` saw each granule separately,
   including same-date overlaps. The exported snapshot is `collection.median()` over the
   one-day window, so the local reconstruction sees one composited value per date.

Neither is a bug in the backfill; both are consequences of reconstructing from exports
rather than from source imagery. Phase 0 measures their combined effect directly by
recomputing dates that Earth Engine itself exported with the temporal bands, and
reporting correlation, bias, RMSE, difference percentiles and masking disagreement on
water pixels.

**Sensor coverage of that validation is asymmetric and must not be glossed over.** S2 has
reference exports and can be validated. S1 has none, so it reports `UNVALIDATED` — not
passing, not omitted. The S2 agreement does not transfer: the sensors differ in revisit
interval, swath coverage and source-band distribution. See
`docs/temporal_backfill_validation_summary.md`.

---

## 9. Stale Drive mounts are a data-integrity issue, not just an annoyance

Colab drops the Drive FUSE mount mid-run; every Drive path then raises `OSError` Errno
107. The classifier already handles this for its own reads via
`_is_transport_endpoint_error` / `_run_with_drive_remount_retry`. Two things are worth
carrying across, because the obvious implementation is wrong in a way that produces
confident nonsense rather than an error.

**The errno/message test alone is insufficient.** `rasterio` raises `RasterioIOError`,
which subclasses `OSError` but carries `errno=None` and a GDAL message such as
`…/scene.tif: not recognized as a supported file format` — no Errno 107, and none of the
phrases `_is_transport_endpoint_error` looks for. So a dead mount is reported as an
unreadable file. On a first real Phase 0 run this presented as **all 814 multi-file
prefixes resolving to `unreadable`**, which reads like archive corruption but was a
mount that had gone away a few seconds earlier.

The fix is to re-probe the path. It was listed successfully moments before, so:

- path no longer `stat`s → the mount died → remount and retry;
- path still `stat`s → the file really is unreadable → fail loudly.

That logic is `winam_diagnostics.temporal_backfill.looks_like_stale_mount`, unit-tested
in `tests/test_temporal_backfill.py`, and is what the backfill notebook uses. If the
classifier ever gains VRT staging (section 3), it should use the same test rather than
`_is_transport_endpoint_error` alone, since staging a VRT means opening several rasters
where a mid-sequence mount loss is likely.

**Never let an unresolved read become a silent assumption.** A multi-file prefix is
either genuine Earth Engine tile shards or copies of one scene. Reading copies as shards
double-counts that observation in every overlapping window; reading shards as copies
drops coverage. When the resolution cannot be obtained, the backfill refuses to guess —
it aborts Phase 0 by default, or, under `UNRESOLVED_PREFIX_POLICY = 'exclude'`, drops
those snapshots and records the loss. There is deliberately no "assume shards" option.
Any equivalent ambiguity on the classifier side deserves the same treatment.

---

## 10. Suggested rollout

1. Run Phase 0 of the backfill notebook against the real archive and read the verdict.
2. Run Phases 1–3 for S2 only (`SENSORS = ['S2']`), which has a validated estimator.
3. Add `USE_BACKFILLED_TEMPORAL_BANDS` and the discovery/staging changes in sections 2–3
   above to the classifier, behind the flag, defaulting to off.
4. Classify a handful of backfilled dates with `BATCH_MAX_DATASETS = 3` and compare area
   tables against the same dates classified from genuine Earth Engine exports, if any
   overlap exists.
5. Only then enable S1, and carry its `UNVALIDATED` status into anything reported from it.
