"""Local reconstruction of the snapshot-relative 90-day temporal-persistence bands.

``Batch_Export.ipynb`` bumped both predictor export schemas to add temporal
persistence bands computed over a 90-day lookback ending at each snapshot's
``end_date``:

* **S2** ``s2_whlev_texture_v1`` (20 bands) -> ``s2_whlev_temporal_v1`` (21 bands),
  adding ``ndvi_temporal_std_w90``.
* **S1** ``winam_s1_scc_predictors_*`` (3 bands) -> ``s1_scc_temporal_v1``
  (5 bands), adding ``vh_temporal_std_w90`` and ``vh_temporal_cv_w90``.

Re-exporting the ~610 GB archive from Earth Engine is not viable, and all three
new bands are deterministic functions of a band that is already present in every
exported snapshot (``NDVI`` for S2, ``VH_corrected`` for S1). This module holds
the pure, offline logic used by ``Backfill_Temporal_Bands_Local.ipynb`` to
rebuild them from the exports already sitting on Drive.

**Nothing here imports Earth Engine, mounts Google Drive, or touches the
network.** Every function works on ordinary filesystem paths so the whole module
is unit-tested offline against small synthetic rasters.

Faithfulness, not correctness
-----------------------------
The point of this module is to reproduce what Earth Engine wrote, including the
parts that are questionable. Three known issues are preserved deliberately and
documented in ``docs/temporal_backfill_integration.md``:

1. ``vh_temporal_cv_w90`` divides by ``abs(mean)`` of a **dB** quantity and is
   therefore not the scale-free coefficient of variation its source comment
   claims. Reproduced as written.
2. The S1 stack mixes ascending/descending orbit geometries, so the temporal
   standard deviation conflates real change with acquisition geometry.
3. S1 swath coverage is uneven across the AOI, so per-pixel observation counts
   (and hence the min-obs mask) are strongly spatially structured. This is why
   the observation-count band is retained as an output.

Known departure from Earth Engine
---------------------------------
Earth Engine reduced over **every scene passing the source-collection filters**,
whereas this reconstruction can only see snapshots that passed the export
coverage gate and actually reached Drive. It also sees one median-composited
observation per acquisition date, where Earth Engine saw each granule. Phase 0
of the notebook quantifies the resulting difference against real Earth Engine
output rather than assuming it is negligible.

The AOI metric CRS is **EPSG:32736 (UTM 36S)** - Lake Victoria, not the UK.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
from xml.dom import minidom

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

__all__ = [
    # schema
    "PREDICTOR_NODATA_VALUE",
    "EXPORT_CRS",
    "EXPORT_SCALE",
    "TEMPORAL_LOOKBACK_DAYS",
    "TEMPORAL_MIN_OBS",
    "TEMPORAL_DDOF",
    "S1_CV_MIN_ABS_MEAN",
    "S2_PREDICTORS",
    "S2_LEGACY_PREDICTORS",
    "S1_PREDICTORS",
    "S1_LEGACY_PREDICTORS",
    "SENSOR_SPECS",
    "sensor_spec",
    # errors
    "BackfillError",
    "BandDescriptionError",
    "GridMismatchError",
    "ReadOnlyPathError",
    # naming / discovery
    "DRIVE_COLLISION_RE",
    "is_drive_collision_duplicate",
    "parse_predictor_export_name",
    "discover_predictor_files",
    "group_snapshot_files",
    "classify_prefix_files",
    "select_snapshots_by_date",
    "Snapshot",
    # window semantics
    "parse_iso_date",
    "window_bounds",
    "dates_in_window",
    "snapshots_in_window",
    # grid
    "GridSpec",
    "grid_from_dataset",
    "assert_grid_identity",
    "union_grid",
    # blocks
    "choose_block_shape",
    "plan_blocks",
    # statistics
    "valid_mask",
    "temporal_block_stats",
    "apply_min_obs",
    "temporal_cv",
    # raster helpers
    "band_index_by_description",
    "single_band_profile",
    "extract_source_band",
    "compute_temporal_bands",
    # vrt
    "build_vrt_xml",
    "plan_vrt_bands",
    "write_snapshot_vrt",
    # manifests
    "CACHE_MANIFEST_COLUMNS",
    "RUN_MANIFEST_COLUMNS",
    "empty_manifest",
    "load_manifest",
    "save_manifest",
    "upsert_manifest_record",
    "utc_now_iso",
    # safety
    "assert_not_in_readonly_dir",
    "is_transport_endpoint_error",
    "looks_like_stale_mount",
    "DRIVE_MOUNT_PREFIX",
    "UnsafeRemovalError",
    "is_live_mount",
    "assert_safe_to_remove",
    "remove_stray_mount_dirs",
    "disk_headroom_bytes",
    "human_bytes",
]


# ---------------------------------------------------------------------------
# Schema constants. Mirrors Batch_Export.ipynb sections 2, 3, 5, 6 and 10.
# ---------------------------------------------------------------------------

PREDICTOR_NODATA_VALUE = -9999.0
EXPORT_CRS = "EPSG:32736"
EXPORT_SCALE = 10

#: ``ee.Date(end_date).advance(-90, 'day')`` .. ``end_date`` in both
#: ``add_s2_temporal_stability`` and ``add_s1_temporal_stability``.
TEMPORAL_LOOKBACK_DAYS = 90

#: ``S2_TEMPORAL_MIN_OBS`` / ``S1_TEMPORAL_MIN_OBS`` in Batch_Export.ipynb.
TEMPORAL_MIN_OBS = 3

#: Delta degrees of freedom for the standard deviation. Earth Engine's
#: ``ee.Reducer.stdDev()`` documents a *sample* standard deviation (ddof=1), but
#: Phase 0 of the notebook settles this empirically against real Earth Engine
#: output and overrides this default if the data says otherwise.
TEMPORAL_DDOF = 1

#: Policy constant for ``vh_temporal_cv_w90 = std / abs(mean)`` when the window
#: mean approaches zero.
#:
#: Earth Engine's behaviour here cannot currently be observed (no S1 snapshot has
#: been exported with the temporal bands), so this is a chosen, defensible
#: policy, not a measured one: pixels whose ``abs(mean)`` falls below this
#: threshold are written as NoData instead of emitting +/-Inf.
#:
#: Rationale: the classifier's ``_predictor_valid_mask`` drops any pixel with a
#: non-finite value in *any* band, so an Inf here would silently invalidate the
#: whole pixel across all 5 S1 bands anyway. Masking makes that explicit and
#: countable (see ``n_nonfinite_cv`` in the run manifest) instead of letting
#: infinities propagate. Change this in one place if a reference S1 export with
#: Earth Engine's own temporal bands ever becomes available.
S1_CV_MIN_ABS_MEAN = 1e-6

# Batch_Export.ipynb section 3, verbatim.
S2_BASE_PREDICTORS = [
    "AWEI_p95", "AWEI", "AWEInsh", "NDMI", "MNDWI", "NDVI",
    "B", "G", "R", "RE1", "RE2", "RE3", "RE4", "NIR", "SWIR", "SWIR2",
]
S2_WH_LEV_EXTRA_PREDICTORS = [
    "dist_to_shore_m",
    "nir_glcm_homogeneity_w5",
    "nir_glcm_entropy_w5",
    "nir_glcm_contrast_w5",
]
#: 20-band pre-bump S2 schema (``s2_whlev_texture_v1``).
S2_LEGACY_PREDICTORS = S2_BASE_PREDICTORS + S2_WH_LEV_EXTRA_PREDICTORS
#: 21-band current S2 schema (``s2_whlev_temporal_v1``).
S2_PREDICTORS = S2_LEGACY_PREDICTORS + ["ndvi_temporal_std_w90"]

#: 3-band pre-bump S1 schema (``winam_s1_scc_predictors_*``).
S1_LEGACY_PREDICTORS = ["VH_p5", "VH_corrected", "VH_smooth"]
#: 5-band current S1 schema (``s1_scc_temporal_v1``).
S1_PREDICTORS = S1_LEGACY_PREDICTORS + ["vh_temporal_std_w90", "vh_temporal_cv_w90"]

S2_TEMPORAL_SCHEMA_TOKEN = "s2_whlev_temporal_v1"
S2_LEGACY_SCHEMA_TOKEN = "s2_whlev_texture_v1"
S1_TEMPORAL_SCHEMA_TOKEN = "s1_scc_temporal_v1"


class BackfillError(RuntimeError):
    """Base class for every loud failure raised by this module."""


class BandDescriptionError(BackfillError):
    """A source raster has no usable band descriptions, or lacks a named band."""


class GridMismatchError(BackfillError):
    """Contributing rasters do not share an identical CRS/transform/shape."""


class ReadOnlyPathError(BackfillError):
    """A write was attempted inside a directory declared read-only."""


class UnsafeRemovalError(BackfillError):
    """A recursive delete was attempted on a path that may hold real Drive data."""


@dataclass(frozen=True)
class SensorSpec:
    """Everything that differs between the S1 and S2 backfills.

    ``prefix_template`` fields take ``start`` and ``end`` ISO date strings and
    reproduce ``build_export_manifest`` in Batch_Export.ipynb section 7. Note the
    asymmetry: S2 keeps ``winam_s2_predictors_`` and inserts the schema token,
    while S1 drops ``predictors`` entirely.
    """

    sensor: str
    source_band: str
    legacy_predictors: tuple[str, ...]
    temporal_predictors: tuple[str, ...]
    new_bands: tuple[str, ...]
    count_band: str
    temporal_schema_token: str
    legacy_prefix_template: str
    temporal_prefix_template: str
    lookback_days: int = TEMPORAL_LOOKBACK_DAYS
    min_obs: int = TEMPORAL_MIN_OBS

    def legacy_prefix(self, start: str, end: str) -> str:
        return self.legacy_prefix_template.format(start=start, end=end)

    def temporal_prefix(self, start: str, end: str) -> str:
        return self.temporal_prefix_template.format(start=start, end=end)

    @property
    def std_band(self) -> str:
        return self.new_bands[0]

    @property
    def cv_band(self) -> str | None:
        return self.new_bands[1] if len(self.new_bands) > 1 else None


SENSOR_SPECS: dict[str, SensorSpec] = {
    "S2": SensorSpec(
        sensor="S2",
        source_band="NDVI",
        legacy_predictors=tuple(S2_LEGACY_PREDICTORS),
        temporal_predictors=tuple(S2_PREDICTORS),
        new_bands=("ndvi_temporal_std_w90",),
        count_band="ndvi_temporal_count_w90",
        temporal_schema_token=S2_TEMPORAL_SCHEMA_TOKEN,
        legacy_prefix_template=(
            "winam_s2_predictors_" + S2_LEGACY_SCHEMA_TOKEN + "_{start}_to_{end}"
        ),
        temporal_prefix_template=(
            "winam_s2_predictors_" + S2_TEMPORAL_SCHEMA_TOKEN + "_{start}_to_{end}"
        ),
    ),
    "S1": SensorSpec(
        sensor="S1",
        source_band="VH_corrected",
        legacy_predictors=tuple(S1_LEGACY_PREDICTORS),
        temporal_predictors=tuple(S1_PREDICTORS),
        new_bands=("vh_temporal_std_w90", "vh_temporal_cv_w90"),
        count_band="vh_temporal_count_w90",
        temporal_schema_token=S1_TEMPORAL_SCHEMA_TOKEN,
        legacy_prefix_template="winam_s1_scc_predictors_{start}_to_{end}",
        temporal_prefix_template=(
            "winam_" + S1_TEMPORAL_SCHEMA_TOKEN + "_{start}_to_{end}"
        ),
    ),
}


def sensor_spec(sensor: str) -> SensorSpec:
    try:
        return SENSOR_SPECS[str(sensor).upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise BackfillError(f"Unknown sensor {sensor!r}; expected one of {sorted(SENSOR_SPECS)}") from exc


# ---------------------------------------------------------------------------
# Naming and discovery. Mirrors the classifier's _CORR_DRIVE_COLLISION_RE,
# _corr_discover_predictor_tifs and parse_predictor_export_name so this notebook
# and the classifier always agree on what a snapshot's files are.
# ---------------------------------------------------------------------------

#: Google Drive's collision marker, e.g. ``foo (1).tif``. Same pattern as
#: ``_CORR_DRIVE_COLLISION_RE`` in the classifier notebook.
DRIVE_COLLISION_RE = re.compile(r"\s\(\d+\)(\.[^.]+)?$")

_EXPORT_NAME_RE = re.compile(
    r"^(?P<prefix>winam_(?:"
    # S2: winam_s2_predictors_[<s2_schema>_]<dates>
    r"s2_predictors_(?:(?P<s2_schema>s2_whlev_texture_v1|s2_whlev_temporal_v1)_)?"
    r"|"
    # Current S1: winam_s1_scc_temporal_v1_<dates> (no _predictors_ segment)
    r"(?P<s1_schema>s1_scc_temporal_v1)_"
    r"|"
    # Legacy S1: winam_s1_scc_predictors_<dates> or winam_s1_predictors_<dates>
    r"s1_scc_predictors_|s1_predictors_"
    r")"
    r"(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2}))"
    r"(?P<tile>-\d{10}-\d{10})?$"
)

_RASTER_SUFFIXES = {".tif", ".tiff", ""}


def is_drive_collision_duplicate(path) -> bool:
    """True for Google Drive collision duplicates such as ``foo (1).tif``."""
    return bool(DRIVE_COLLISION_RE.search(Path(path).name))


def _normalise_name_for_matching(path) -> str:
    name = Path(path).name
    name = re.sub(r"\.(tif|tiff)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s\(\d+\)$", "", name)
    return name


def parse_predictor_export_name(path) -> dict | None:
    """Parse Batch_Export.ipynb's predictor export naming convention.

    Recognises, with or without a tiled ``-0000000000-0000008192`` suffix::

        winam_s2_predictors_s2_whlev_temporal_v1_YYYY-MM-DD_to_YYYY-MM-DD  (current S2)
        winam_s2_predictors_s2_whlev_texture_v1_YYYY-MM-DD_to_YYYY-MM-DD   (legacy S2)
        winam_s2_predictors_YYYY-MM-DD_to_YYYY-MM-DD                       (legacy S2, no token)
        winam_s1_scc_temporal_v1_YYYY-MM-DD_to_YYYY-MM-DD                  (current S1, no _predictors_)
        winam_s1_scc_predictors_YYYY-MM-DD_to_YYYY-MM-DD                   (legacy S1)

    Returns ``None`` for anything else. Kept deliberately identical to the
    classifier's ``parse_predictor_export_name`` so the two never disagree about
    the S1 prefix asymmetry.
    """
    stem = _normalise_name_for_matching(path)
    match = _EXPORT_NAME_RE.match(stem)
    if not match:
        return None
    prefix = match.group("prefix")
    sensor = "S2" if prefix.startswith("winam_s2_") else "S1"
    schema = match.group("s2_schema") if sensor == "S2" else match.group("s1_schema")
    return {
        "path": Path(path),
        "prefix": prefix,
        "sensor": sensor,
        "schema_token": schema,
        "s2_schema": match.group("s2_schema"),
        "s1_schema": match.group("s1_schema"),
        "start_date": match.group("start"),
        "end_date": match.group("end"),
        "is_tile": bool(match.group("tile")),
        "tile_suffix": match.group("tile") or "",
        "is_drive_duplicate": is_drive_collision_duplicate(path),
    }


def discover_predictor_files(export_dir, sensors: Sequence[str] | None = None) -> pd.DataFrame:
    """Inventory every parsable predictor GeoTIFF in ``export_dir``.

    Drive collision duplicates are **retained** in this frame (flagged by
    ``is_drive_duplicate``) so Phase 0 can report them; ``group_snapshot_files``
    is what drops them from the working set. Nothing is read from the files
    themselves, so this is cheap over a Drive mount.
    """
    export_dir = Path(export_dir)
    if not export_dir.exists():
        raise FileNotFoundError(f"Predictor export folder does not exist: {export_dir}")

    wanted = {s.upper() for s in sensors} if sensors else None
    rows = []
    for path in sorted(export_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _RASTER_SUFFIXES:
            continue
        parsed = parse_predictor_export_name(path)
        if parsed is None:
            continue
        if wanted is not None and parsed["sensor"] not in wanted:
            continue
        stat = path.stat()
        rows.append({**parsed, "path": path, "size_bytes": stat.st_size, "mtime": stat.st_mtime})

    columns = [
        "path", "prefix", "sensor", "schema_token", "s2_schema", "s1_schema",
        "start_date", "end_date", "is_tile", "tile_suffix", "is_drive_duplicate",
        "size_bytes", "mtime",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values(["sensor", "start_date", "prefix"]).reset_index(drop=True)


@dataclass(frozen=True)
class Snapshot:
    """One exported acquisition-date snapshot and the file(s) that hold it."""

    sensor: str
    prefix: str
    schema_token: str | None
    start_date: _dt.date
    end_date: _dt.date
    paths: tuple[Path, ...]
    duplicate_paths: tuple[Path, ...] = ()

    @property
    def is_tiled(self) -> bool:
        return len(self.paths) > 1

    @property
    def has_temporal_bands(self) -> bool:
        """True when this snapshot was exported by Earth Engine with the new bands."""
        return self.schema_token == sensor_spec(self.sensor).temporal_schema_token

    @property
    def start_iso(self) -> str:
        return self.start_date.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end_date.isoformat()

    def target_prefix(self) -> str:
        """The new-schema prefix this snapshot's backfilled outputs belong to."""
        spec = sensor_spec(self.sensor)
        return spec.temporal_prefix(self.start_iso, self.end_iso)


def group_snapshot_files(inventory: pd.DataFrame) -> list[Snapshot]:
    """Group an inventory frame into one :class:`Snapshot` per export prefix.

    Drive collision duplicates are separated out into ``duplicate_paths`` and
    never enter ``paths``, so they can never be counted twice in a window.
    Remaining files for a prefix are sorted by name, which puts Earth Engine's
    ``-<row>-<col>`` tile shards in a stable mosaic order.
    """
    if inventory is None or len(inventory) == 0:
        return []

    snapshots: list[Snapshot] = []
    for (sensor, prefix), group in inventory.groupby(["sensor", "prefix"], sort=True):
        dup = group[group["is_drive_duplicate"].astype(bool)]
        keep = group[~group["is_drive_duplicate"].astype(bool)]
        if keep.empty:
            # Every file for this prefix is a Drive duplicate; there is no
            # trustworthy original, so the snapshot is dropped entirely.
            continue
        first = keep.iloc[0]
        tokens = {t for t in keep["schema_token"].tolist() if t is not None}
        if len(tokens) > 1:
            raise BackfillError(
                f"Prefix {prefix!r} maps to more than one schema token: {sorted(tokens)}"
            )
        snapshots.append(
            Snapshot(
                sensor=str(sensor),
                prefix=str(prefix),
                schema_token=first["schema_token"],
                start_date=parse_iso_date(first["start_date"]),
                end_date=parse_iso_date(first["end_date"]),
                paths=tuple(sorted((Path(p) for p in keep["path"]), key=lambda p: p.name)),
                duplicate_paths=tuple(sorted((Path(p) for p in dup["path"]), key=lambda p: p.name)),
            )
        )
    snapshots.sort(key=lambda s: (s.sensor, s.start_date, s.prefix))
    return snapshots


def classify_prefix_files(snapshot: Snapshot, opener=rasterio.open) -> dict:
    """Decide whether a multi-file prefix holds genuine tile shards or copies.

    Filenames alone are not conclusive: Earth Engine tile shards carry a
    ``-<row>-<col>`` suffix, but an export folder can also accumulate
    same-content files that Drive did *not* mark with a ``(1)`` collision
    suffix. Getting this wrong silently double-counts observations in every
    overlapping window, so this reads each file's georeferencing and reports:

    ``kind``
        ``'single'``, ``'tiled'`` (distinct, non-overlapping grid windows),
        ``'duplicate_grid'`` (two or more files describing the same grid), or
        ``'inconsistent'`` (differing CRS or pixel size).
    ``grids``
        the :class:`GridSpec` read from each path.
    ``message``
        a human-readable explanation for the Phase 0 report.
    """
    grids = []
    for path in snapshot.paths:
        with opener(path) as src:
            grids.append(grid_from_dataset(src))

    if len(grids) == 1:
        return {
            "prefix": snapshot.prefix,
            "sensor": snapshot.sensor,
            "n_files": 1,
            "kind": "single",
            "grids": grids,
            "message": "single file",
        }

    crs_set = {g.crs for g in grids}
    res_set = {(round(g.transform[0], 6), round(g.transform[4], 6)) for g in grids}
    if len(crs_set) > 1 or len(res_set) > 1:
        return {
            "prefix": snapshot.prefix,
            "sensor": snapshot.sensor,
            "n_files": len(grids),
            "kind": "inconsistent",
            "grids": grids,
            "message": (
                f"{len(grids)} files disagree on CRS {sorted(crs_set)} or pixel size "
                f"{sorted(res_set)}; cannot mosaic"
            ),
        }

    signatures = {(g.transform[2], g.transform[5], g.width, g.height) for g in grids}
    if len(signatures) < len(grids):
        return {
            "prefix": snapshot.prefix,
            "sensor": snapshot.sensor,
            "n_files": len(grids),
            "kind": "duplicate_grid",
            "grids": grids,
            "message": (
                f"{len(grids)} files but only {len(signatures)} distinct grid window(s): "
                "these are copies, not tile shards. Treating them as shards would "
                "double-count observations in every window."
            ),
        }

    return {
        "prefix": snapshot.prefix,
        "sensor": snapshot.sensor,
        "n_files": len(grids),
        "kind": "tiled",
        "grids": grids,
        "message": f"{len(grids)} genuine tile shards covering distinct grid windows",
    }


def select_snapshots_by_date(snapshots: Sequence[Snapshot]) -> tuple[list[Snapshot], list[dict]]:
    """Keep exactly one snapshot per (sensor, acquisition date).

    A date can appear twice when it has been re-exported under the new schema
    while the legacy export is still on Drive. Both files carry the same source
    band, so counting both would double-count that observation in every
    overlapping window. Preference order: the temporal-schema export wins,
    otherwise the lexicographically last prefix, deterministically.

    Returns ``(chosen, collisions)`` where ``collisions`` describes every date
    that had more than one candidate, for the Phase 0 report.
    """
    by_key: dict[tuple[str, _dt.date], list[Snapshot]] = {}
    for snap in snapshots:
        by_key.setdefault((snap.sensor, snap.start_date), []).append(snap)

    chosen: list[Snapshot] = []
    collisions: list[dict] = []
    for (sensor, day), group in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(group) == 1:
            chosen.append(group[0])
            continue
        ranked = sorted(group, key=lambda s: (s.has_temporal_bands, s.prefix), reverse=True)
        chosen.append(ranked[0])
        collisions.append({
            "sensor": sensor,
            "date": day.isoformat(),
            "kept_prefix": ranked[0].prefix,
            "dropped_prefixes": [s.prefix for s in ranked[1:]],
        })
    chosen.sort(key=lambda s: (s.sensor, s.start_date))
    return chosen, collisions


# ---------------------------------------------------------------------------
# Window semantics.
#
# Both add_*_temporal_stability functions build the window as
#     ee.Date(end_date).advance(-LOOKBACK, 'day')  ..  end_date
# and Earth Engine's filterDate is inclusive of start, exclusive of end. Every
# snapshot's end_date is its acquisition date + 1 day, so the window INCLUDES
# the target snapshot's own acquisition. Acquisitions carry a time-of-day
# strictly after midnight UTC, so day-granularity membership is exact.
# ---------------------------------------------------------------------------

def parse_iso_date(value) -> _dt.date:
    """Coerce ``'YYYY-MM-DD'`` (or a date/datetime) to :class:`datetime.date`."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


def window_bounds(end_date, lookback_days: int = TEMPORAL_LOOKBACK_DAYS) -> tuple[_dt.date, _dt.date]:
    """Return ``(start, end)`` for a snapshot's lookback window.

    ``start`` is **inclusive**, ``end`` is **exclusive** - exactly
    ``filterDate(ee.Date(end_date).advance(-lookback, 'day'), end_date)``.
    """
    end = parse_iso_date(end_date)
    return end - _dt.timedelta(days=int(lookback_days)), end


def dates_in_window(end_date, candidate_dates: Iterable, lookback_days: int = TEMPORAL_LOOKBACK_DAYS) -> list[_dt.date]:
    """Acquisition dates falling inside the lookback window, sorted ascending.

    A date exactly ``lookback_days`` before ``end_date`` is **included**; the
    target snapshot's own acquisition date (``end_date - 1 day``) is
    **included**; ``end_date`` itself is **excluded**.
    """
    start, end = window_bounds(end_date, lookback_days)
    hits = {parse_iso_date(d) for d in candidate_dates}
    return sorted(d for d in hits if start <= d < end)


def snapshots_in_window(
    target: Snapshot,
    candidates: Sequence[Snapshot],
    lookback_days: int | None = None,
) -> list[Snapshot]:
    """Snapshots of the **same sensor** contributing to ``target``'s window.

    Sensors never cross-contaminate a window: candidates from another sensor are
    dropped before the date filter, not merely sorted after it.
    """
    spec = sensor_spec(target.sensor)
    lookback = spec.lookback_days if lookback_days is None else lookback_days
    start, end = window_bounds(target.end_date, lookback)
    hits = [
        snap for snap in candidates
        if snap.sensor == target.sensor and start <= snap.start_date < end
    ]
    hits.sort(key=lambda s: (s.start_date, s.prefix))
    return hits


# ---------------------------------------------------------------------------
# Grid identity.
# ---------------------------------------------------------------------------

_TRANSFORM_DP = 6


@dataclass(frozen=True)
class GridSpec:
    """CRS + affine transform + shape, rounded so equality is FP-noise free."""

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @classmethod
    def create(cls, crs, transform, width, height) -> "GridSpec":
        coeffs = tuple(round(float(v), _TRANSFORM_DP) for v in tuple(transform)[:6])
        return cls(crs=str(crs), transform=coeffs, width=int(width), height=int(height))

    def describe(self) -> str:
        return (
            f"crs={self.crs} size={self.width}x{self.height} "
            f"transform={self.transform}"
        )


def grid_from_dataset(src) -> GridSpec:
    """Read the :class:`GridSpec` of an open rasterio dataset."""
    return GridSpec.create(src.crs, src.transform, src.width, src.height)


def assert_grid_identity(items: Sequence[tuple[str, GridSpec]], context: str = "") -> GridSpec:
    """Require every ``(label, GridSpec)`` to be identical; raise if not.

    Aborting is deliberate: silently resampling mismatched grids would produce
    a temporal statistic over pixels that are not the same ground location.
    """
    if not items:
        raise GridMismatchError(f"No rasters to check grid identity for{f' ({context})' if context else ''}.")
    reference_label, reference = items[0]
    mismatches = [(label, grid) for label, grid in items[1:] if grid != reference]
    if mismatches:
        lines = [
            f"Grid mismatch{f' for {context}' if context else ''}; refusing to resample.",
            f"  reference {reference_label}: {reference.describe()}",
        ]
        lines += [f"  mismatch  {label}: {grid.describe()}" for label, grid in mismatches]
        raise GridMismatchError("\n".join(lines))
    return reference


def union_grid(grids: Sequence[GridSpec]) -> GridSpec:
    """Smallest grid containing every input grid.

    Used to mosaic Earth Engine tile shards back into one full-AOI raster. All
    inputs must share a CRS, pixel size and pixel alignment; anything else is a
    mismatch, not something to resample around.
    """
    if not grids:
        raise GridMismatchError("union_grid() needs at least one grid.")
    if len(grids) == 1:
        return grids[0]

    crs_set = {g.crs for g in grids}
    if len(crs_set) > 1:
        raise GridMismatchError(f"Cannot union grids with differing CRS: {sorted(crs_set)}")
    res_set = {(g.transform[0], g.transform[4]) for g in grids}
    if len(res_set) > 1:
        raise GridMismatchError(f"Cannot union grids with differing pixel sizes: {sorted(res_set)}")
    skew = {(g.transform[1], g.transform[3]) for g in grids}
    if skew != {(0.0, 0.0)}:
        raise GridMismatchError(f"Cannot union rotated/skewed grids: {sorted(skew)}")

    x_res, y_res = grids[0].transform[0], grids[0].transform[4]
    origin_x = min(g.transform[2] for g in grids)
    origin_y = max(g.transform[5] for g in grids)
    for grid in grids:
        col = (grid.transform[2] - origin_x) / x_res
        row = (grid.transform[5] - origin_y) / y_res
        if abs(col - round(col)) > 1e-3 or abs(row - round(row)) > 1e-3:
            raise GridMismatchError(
                "Tile shards are not aligned to a common pixel grid; refusing to resample.\n"
                f"  offending grid: {grid.describe()}"
            )

    right = max(g.transform[2] + g.width * x_res for g in grids)
    bottom = min(g.transform[5] + g.height * y_res for g in grids)
    width = int(round((right - origin_x) / x_res))
    height = int(round((bottom - origin_y) / y_res))
    return GridSpec.create(grids[0].crs, (x_res, 0.0, origin_x, 0.0, y_res, origin_y), width, height)


def _grid_offset(parent: GridSpec, child: GridSpec) -> tuple[int, int]:
    """Column/row offset of ``child``'s origin inside ``parent``."""
    col = (child.transform[2] - parent.transform[2]) / parent.transform[0]
    row = (child.transform[5] - parent.transform[5]) / parent.transform[4]
    return int(round(col)), int(round(row))


# ---------------------------------------------------------------------------
# Windowed block planning. Nothing here ever holds a whole stack in memory.
# ---------------------------------------------------------------------------

#: Peak bytes budgeted for one block of the temporal stack. Sized for a ~12 GB
#: Colab runtime with generous headroom for pandas/matplotlib alongside.
DEFAULT_MAX_BLOCK_BYTES = 512 * 1024 * 1024


def choose_block_shape(
    src_block_shape: tuple[int, int],
    n_layers: int,
    max_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
) -> tuple[int, int]:
    """Grow the source's internal tile shape to the largest affordable block.

    Blocks stay an integer multiple of the source's internal tiling so reads
    never straddle a compressed tile boundary. The per-pixel cost accounts for
    the float32 stack, its boolean valid masks and the float64 accumulators.
    """
    base_h = max(1, int(src_block_shape[0]))
    base_w = max(1, int(src_block_shape[1]))
    n_layers = max(1, int(n_layers))
    bytes_per_pixel = n_layers * (4 + 1) + 8 * 4  # stack + masks + accumulators
    max_pixels = max(base_h * base_w, int(max_bytes) // max(1, bytes_per_pixel))

    height, width = base_h, base_w
    # Alternate growth so blocks stay roughly square rather than one long strip.
    grow_rows = True
    while True:
        next_h = height + base_h if grow_rows else height
        next_w = width if grow_rows else width + base_w
        if next_h * next_w > max_pixels:
            if grow_rows and (height * (width + base_w)) <= max_pixels:
                grow_rows = False
                continue
            break
        height, width = next_h, next_w
        grow_rows = not grow_rows
    return height, width


def plan_blocks(width: int, height: int, block_shape: tuple[int, int]) -> Iterator[Window]:
    """Yield row-major :class:`rasterio.windows.Window` tiles covering the grid."""
    block_h = max(1, int(block_shape[0]))
    block_w = max(1, int(block_shape[1]))
    for row_off in range(0, int(height), block_h):
        rows = min(block_h, int(height) - row_off)
        for col_off in range(0, int(width), block_w):
            cols = min(block_w, int(width) - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=cols, height=rows)


# ---------------------------------------------------------------------------
# Statistics. Two-pass (mean, then squared deviations) in float64 so a VH stack
# around -20 dB with a ~1 dB spread does not lose precision to cancellation.
# ---------------------------------------------------------------------------

def valid_mask(array, nodata: float = PREDICTOR_NODATA_VALUE) -> np.ndarray:
    """Finite and not the export NoData sentinel.

    ``-9999`` never enters any statistic: it is excluded here, before the count,
    the mean and the standard deviation are formed.
    """
    arr = np.asarray(array)
    return np.isfinite(arr) & (arr != nodata)


def temporal_block_stats(
    layers: Sequence[np.ndarray],
    nodata: float = PREDICTOR_NODATA_VALUE,
    ddof: int = TEMPORAL_DDOF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel ``(count, mean, std)`` across a stack of equally-shaped blocks.

    The mean and the standard deviation are formed over the **identical** valid
    observation set, which is what ``add_s1_temporal_stability`` does (both
    ``ee.Reducer.mean()`` and ``ee.Reducer.stdDev()`` see the same masked
    collection), and is what makes ``std / abs(mean)`` a coherent ratio.

    ``mean`` is NaN where nothing is valid; ``std`` is NaN wherever
    ``count - ddof <= 0``. No min-observation masking is applied here - see
    :func:`apply_min_obs`.
    """
    if not layers:
        raise BackfillError("temporal_block_stats() needs at least one layer.")
    shape = np.asarray(layers[0]).shape
    for index, layer in enumerate(layers):
        if np.asarray(layer).shape != shape:
            raise BackfillError(
                f"Layer {index} has shape {np.asarray(layer).shape}, expected {shape}."
            )

    count = np.zeros(shape, dtype=np.int32)
    total = np.zeros(shape, dtype=np.float64)
    masks = []
    # NaN/Inf inputs are masked out before they reach an accumulator, so the
    # arithmetic that touches them is discarded; silence the warnings it raises.
    with np.errstate(invalid="ignore"):
        for layer in layers:
            arr = np.asarray(layer)
            mask = valid_mask(arr, nodata)
            masks.append(mask)
            count += mask
            total += np.where(mask, arr, 0.0)

        mean = np.full(shape, np.nan, dtype=np.float64)
        np.divide(total, count, out=mean, where=count > 0)

        sum_sq = np.zeros(shape, dtype=np.float64)
        for layer, mask in zip(layers, masks):
            arr = np.asarray(layer)
            deviation = np.where(mask, arr - mean, 0.0)
            sum_sq += deviation * deviation

        denominator = count.astype(np.float64) - float(ddof)
        variance = np.full(shape, np.nan, dtype=np.float64)
        np.divide(sum_sq, denominator, out=variance, where=denominator > 0)
        std = np.sqrt(
            variance, out=np.full(shape, np.nan, dtype=np.float64), where=np.isfinite(variance)
        )
    return count, mean, std


def apply_min_obs(values: np.ndarray, count: np.ndarray, min_obs: int = TEMPORAL_MIN_OBS) -> np.ndarray:
    """NaN out pixels observed fewer than ``min_obs`` times.

    Reproduces ``updateMask(count.gte(MIN_OBS))``; the caller then writes NaN as
    ``PREDICTOR_NODATA_VALUE``, exactly as the export's ``unmask(-9999)`` did.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    out[np.asarray(count) < int(min_obs)] = np.nan
    return out


def temporal_cv(
    std: np.ndarray,
    mean: np.ndarray,
    min_abs_mean: float = S1_CV_MIN_ABS_MEAN,
) -> tuple[np.ndarray, np.ndarray]:
    """``std / abs(mean)``, with the documented near-zero-mean policy applied.

    Reproduces ``vh_std.divide(vh_mean.abs())`` from
    ``add_s1_temporal_stability``. Note that this is *not* a scale-free
    coefficient of variation for a dB quantity - see the module docstring and
    ``docs/temporal_backfill_integration.md``. It is reproduced as written.

    Returns ``(cv, near_zero_mask)``. Where ``abs(mean) < min_abs_mean`` the CV
    is NaN rather than +/-Inf; ``near_zero_mask`` marks those pixels so the run
    manifest can count them per date instead of letting infinities propagate.
    """
    std_arr = np.asarray(std, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    abs_mean = np.abs(mean_arr)
    near_zero = np.isfinite(mean_arr) & (abs_mean < float(min_abs_mean))
    usable = np.isfinite(std_arr) & np.isfinite(abs_mean) & ~near_zero
    cv = np.full(std_arr.shape, np.nan, dtype=np.float64)
    np.divide(std_arr, abs_mean, out=cv, where=usable)
    cv[~usable] = np.nan
    return cv, near_zero


# ---------------------------------------------------------------------------
# Raster helpers.
# ---------------------------------------------------------------------------

def band_index_by_description(src, name: str, label: str | None = None) -> int:
    """1-based band index of ``name``, resolved by description, never by position.

    Raises :class:`BandDescriptionError` when the raster carries no complete set
    of band descriptions, or when ``name`` is not among them. Falling back to a
    hardcoded index here would silently compute a temporal statistic over the
    wrong band.
    """
    label = label or getattr(src, "name", "<raster>")
    descriptions = list(src.descriptions or ())
    if len(descriptions) != src.count or any(d in (None, "") for d in descriptions):
        raise BandDescriptionError(
            f"{label} has no complete band descriptions ({descriptions!r}); "
            f"cannot resolve source band {name!r} without guessing a band index."
        )
    if name not in descriptions:
        raise BandDescriptionError(
            f"{label} has no band named {name!r}. Available bands: {descriptions}"
        )
    return descriptions.index(name) + 1


#: Creation options for every raster this module writes: tiled, compressed,
#: cloud-optimised layout, float32 with an explicit NoData tag.
DEFAULT_CREATION_OPTIONS = {
    "driver": "GTiff",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "compress": "DEFLATE",
    "predictor": 3,          # floating-point predictor
    "zlevel": 6,
    "BIGTIFF": "IF_SAFER",
    "interleave": "band",
}


def single_band_profile(grid: GridSpec, dtype: str = "float32", nodata: float = PREDICTOR_NODATA_VALUE) -> dict:
    """A rasterio profile for a single-band output on ``grid``."""
    profile = dict(DEFAULT_CREATION_OPTIONS)
    profile.update(
        width=grid.width,
        height=grid.height,
        count=1,
        dtype=dtype,
        crs=grid.crs,
        transform=rasterio.Affine(*grid.transform),
        nodata=nodata,
    )
    if grid.width < 512 or grid.height < 512:
        # GTiff requires block sizes to be multiples of 16 and <= the raster.
        profile["blockxsize"] = max(16, (min(512, grid.width) // 16) * 16)
        profile["blockysize"] = max(16, (min(512, grid.height) // 16) * 16)
        if grid.width < 16 or grid.height < 16:
            profile.pop("tiled")
            profile.pop("blockxsize")
            profile.pop("blockysize")
    return profile


# ---------------------------------------------------------------------------
# Read-only safety. GEE_Exports_validated_snapshots is ~610 GB of exports that
# cannot be cheaply regenerated: this module must never write inside it.
# ---------------------------------------------------------------------------

def assert_not_in_readonly_dir(path, readonly_dirs: Sequence[Path]) -> Path:
    """Raise if ``path`` resolves inside any declared read-only directory."""
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except OSError:  # pragma: no cover - unresolvable path
        resolved = target.absolute()
    for readonly in readonly_dirs:
        readonly_path = Path(readonly).expanduser()
        try:
            readonly_resolved = readonly_path.resolve()
        except OSError:  # pragma: no cover
            readonly_resolved = readonly_path.absolute()
        if resolved == readonly_resolved or readonly_resolved in resolved.parents:
            raise ReadOnlyPathError(
                f"Refusing to write inside the read-only export archive.\n"
                f"  target:    {resolved}\n"
                f"  read-only: {readonly_resolved}"
            )
    return target


# ---------------------------------------------------------------------------
# Stale-mount detection.
#
# Colab drops the Drive FUSE mount mid-run and every Drive path then raises
# OSError Errno 107. The remount itself is Colab-specific and lives in the
# notebook; the *detection* lives here so it is unit-tested.
# ---------------------------------------------------------------------------

DRIVE_MOUNT_PREFIX = "/content/drive"


# ---------------------------------------------------------------------------
# Deletion safety.
#
# A failed mount leaves ordinary local directories sitting where the Drive mount
# belongs, and clearing them is a legitimate repair. The obvious command for that
# repair -- rm -rf /content/drive -- is also the single most destructive thing
# anyone can run on this project, because when the mount IS live it deletes
# straight through FUSE into real Drive data. Both states look identical to `ls`.
#
# That has already happened once here, so the distinction is enforced in code
# rather than left to whoever is reading the instructions at the time. No caller
# in this repo may delete a Drive path recursively without going through
# assert_safe_to_remove first.
# ---------------------------------------------------------------------------


def is_live_mount(path, ismount=None) -> bool:
    """True when ``path`` is itself a live mount point.

    ``ismount`` is injectable for tests; it defaults to :func:`os.path.ismount`.
    A path that cannot be probed is reported as mounted, because the safe answer
    under uncertainty is the one that refuses to delete.
    """
    probe = ismount if ismount is not None else os.path.ismount
    try:
        return bool(probe(str(path)))
    except OSError:
        return True


def assert_safe_to_remove(
    path,
    mount_prefix: str = DRIVE_MOUNT_PREFIX,
    ismount=None,
) -> Path:
    """Raise :class:`UnsafeRemovalError` unless ``path`` is safe to delete recursively.

    The rule is narrow on purpose. Deleting inside ``mount_prefix`` is permitted
    only while nothing is actually mounted there, which is exactly the stray
    local directories left behind by a failed mount. As soon as the mount is
    live, every path at or under it is real Drive data and is refused.

    Returns the resolved path so callers can use it directly.
    """
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except OSError:  # pragma: no cover - unresolvable path
        resolved = target.absolute()

    prefix = Path(mount_prefix).expanduser()
    try:
        prefix_resolved = prefix.resolve()
    except OSError:  # pragma: no cover
        prefix_resolved = prefix.absolute()

    inside_prefix = (
        resolved == prefix_resolved or prefix_resolved in resolved.parents
    )

    if is_live_mount(resolved, ismount=ismount):
        raise UnsafeRemovalError(
            f"Refusing to delete {resolved}: it is a live mount point.\n"
            "Deleting it would follow the mount into the real filesystem behind "
            "it. Unmount first (drive.flush_and_unmount()) if you truly mean to."
        )

    if inside_prefix and is_live_mount(prefix_resolved, ismount=ismount):
        raise UnsafeRemovalError(
            f"Refusing to delete {resolved}: it sits under the live Drive mount at "
            f"{prefix_resolved}, so it is real Drive data, not local scratch.\n"
            "If you meant to clear stray directories left by a failed mount, do it "
            "while Drive is NOT mounted."
        )

    return resolved


def remove_stray_mount_dirs(
    mount_prefix: str = DRIVE_MOUNT_PREFIX,
    ismount=None,
    rmtree=None,
) -> bool:
    """Clear the mount point when it holds only local directories, not a mount.

    This is the supported repair for "Mountpoint must not already contain files",
    and it is the reason no one should ever type a recursive delete against a
    Drive path by hand. Returns True when something was removed.
    """
    prefix = Path(mount_prefix).expanduser()
    if not prefix.exists():
        return False
    assert_safe_to_remove(prefix, mount_prefix=mount_prefix, ismount=ismount)
    (rmtree if rmtree is not None else shutil.rmtree)(str(prefix))
    return True


def is_transport_endpoint_error(exc: BaseException) -> bool:
    """True for a stale Colab/Drive FUSE endpoint (Errno 107).

    Errno 107 can be wrapped by helper functions, so the cause/context chain is
    walked rather than just the outermost exception.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if isinstance(current, OSError) and (
            getattr(current, "errno", None) == 107
            or "Transport endpoint is not connected" in message
            or "Drive FUSE endpoint is stale" in message
        ):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def looks_like_stale_mount(
    exc: BaseException,
    path=None,
    mount_prefix: str = DRIVE_MOUNT_PREFIX,
    stat_probe=None,
) -> bool:
    """True when ``exc`` is better explained by a dead mount than by bad data.

    The errno/message test alone is not enough. ``rasterio`` raises
    ``RasterioIOError``, which subclasses :class:`OSError` but carries
    ``errno=None`` and a GDAL message that does **not** contain "Transport
    endpoint is not connected". So a dead mount is otherwise misreported as an
    unreadable file — which is how an archive scan can come back claiming every
    single prefix is corrupt.

    When ``path`` was listed successfully moments earlier, re-probing it settles
    the question: a path that no longer stats means the mount went away, while a
    path that still stats means the file really is unreadable.

    ``stat_probe`` is injectable for tests; it defaults to ``Path.stat``.
    """
    if is_transport_endpoint_error(exc):
        return True
    if path is None or not isinstance(exc, OSError):
        return False
    if not str(Path(path)).startswith(str(mount_prefix)):
        return False
    probe = stat_probe if stat_probe is not None else (lambda p: Path(p).stat())
    try:
        probe(path)
    except OSError:
        return True
    return False


def disk_headroom_bytes(path) -> int:
    """Free bytes on the filesystem holding ``path`` (nearest existing parent)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def human_bytes(n: float) -> str:
    """Human-readable byte count, e.g. ``'1.29 GB'``."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024.0
    return f"{n:.2f} TB"  # pragma: no cover


# ---------------------------------------------------------------------------
# Phase 1: source-band cache.
# ---------------------------------------------------------------------------

def extract_source_band(
    snapshot: Snapshot,
    out_path,
    readonly_dirs: Sequence[Path] = (),
    opener=rasterio.open,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
    reader=None,
) -> dict:
    """Copy one snapshot's source band into a compressed single-band GeoTIFF.

    Tile shards are mosaicked onto their union grid so every cached snapshot of
    a sensor ends up on one identical grid, which is what makes the Phase 2
    grid-identity assertion meaningful. CRS, transform, shape and the ``-9999``
    NoData tag are preserved exactly; nothing is resampled.

    ``reader`` is an optional context manager (the classifier's
    ``staged_drive_read``) yielding a locally-readable path for a Drive path.
    Writes go to a temporary sibling and are moved into place only on success,
    so an interrupted run never leaves a half-written cache entry.
    """
    from contextlib import ExitStack, nullcontext

    spec = sensor_spec(snapshot.sensor)
    out_path = Path(assert_not_in_readonly_dir(out_path, readonly_dirs))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        sources = []
        for path in snapshot.paths:
            local = stack.enter_context(reader(path) if reader is not None else nullcontext(path))
            src = stack.enter_context(opener(local))
            band = band_index_by_description(src, spec.source_band, label=Path(path).name)
            sources.append((Path(path), src, band, grid_from_dataset(src)))

        target_grid = union_grid([grid for *_, grid in sources])
        profile = single_band_profile(target_grid)

        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.set_band_description(1, spec.source_band)
            dst.update_tags(
                source_band=spec.source_band,
                source_prefix=snapshot.prefix,
                sensor=snapshot.sensor,
                acquisition_date=snapshot.start_iso,
                n_source_files=str(len(sources)),
            )
            # Seed the whole grid with NoData so gaps between shards read as
            # NoData rather than an uninitialised zero. A single source that
            # already covers the union grid needs no seeding.
            if len(sources) > 1 or sources[0][3] != target_grid:
                nodata_block = np.full((1, 1), PREDICTOR_NODATA_VALUE, dtype=np.float32)
                for window in plan_blocks(target_grid.width, target_grid.height, (512, 512)):
                    dst.write(
                        np.broadcast_to(nodata_block, (int(window.height), int(window.width))),
                        1, window=window,
                    )
            for path, src, band, grid in sources:
                col_off, row_off = _grid_offset(target_grid, grid)
                block_shape = choose_block_shape(src.block_shapes[band - 1], 1, max_block_bytes)
                for window in plan_blocks(grid.width, grid.height, block_shape):
                    data = src.read(band, window=window)
                    dst.write(
                        data.astype(np.float32, copy=False), 1,
                        window=Window(
                            col_off=int(window.col_off) + col_off,
                            row_off=int(window.row_off) + row_off,
                            width=int(window.width),
                            height=int(window.height),
                        ),
                    )

    os.replace(tmp_path, out_path)
    return {
        "cache_path": out_path,
        "grid": target_grid,
        "n_source_files": len(snapshot.paths),
        "cache_bytes": out_path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Phase 2: rolling statistics.
# ---------------------------------------------------------------------------

def compute_temporal_bands(
    target: Snapshot,
    contributors: Sequence[Snapshot],
    cache_path_for,
    out_paths: Mapping[str, Path],
    readonly_dirs: Sequence[Path] = (),
    ddof: int = TEMPORAL_DDOF,
    min_obs: int | None = None,
    min_abs_mean: float = S1_CV_MIN_ABS_MEAN,
    opener=rasterio.open,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
) -> dict:
    """Compute and write one snapshot's temporal bands from cached source bands.

    ``contributors`` must already be the same-sensor snapshots inside the target's
    window (see :func:`snapshots_in_window`); this function re-checks the sensor
    and the minimum observation count but does not re-derive the window.

    ``cache_path_for(snapshot) -> Path`` resolves a snapshot to its cached
    source band. ``out_paths`` maps band name -> output path and must cover every
    band in the sensor's ``new_bands`` plus its ``count_band``.

    Processing is strictly block-by-block: only ``n_contributors`` blocks are
    ever resident, never the full stack. Grid identity across every contributing
    file is asserted before a single statistic is computed; a mismatch aborts
    this date with :class:`GridMismatchError` rather than resampling.
    """
    spec = sensor_spec(target.sensor)
    min_obs = spec.min_obs if min_obs is None else int(min_obs)

    cross_sensor = [s.prefix for s in contributors if s.sensor != target.sensor]
    if cross_sensor:
        raise BackfillError(
            f"Refusing to mix sensors in a window for {target.prefix}: "
            f"{target.sensor} target with contributors {cross_sensor}"
        )

    required = list(spec.new_bands) + [spec.count_band]
    missing = [band for band in required if band not in out_paths]
    if missing:
        raise BackfillError(f"compute_temporal_bands() is missing output paths for {missing}")

    n_contributing = len(contributors)
    if n_contributing < min_obs:
        return {
            "status": "skipped_min_obs",
            "n_contributing_scenes": n_contributing,
            "valid_pixels": 0,
            "n_nonfinite_cv": 0,
            "message": (
                f"only {n_contributing} snapshot(s) in the {spec.lookback_days}-day "
                f"window; need >= {min_obs}"
            ),
        }

    tmp_paths: dict[str, Path] = {}
    try:
        return _compute_temporal_bands_impl(
            target, contributors, cache_path_for, out_paths, readonly_dirs, ddof,
            min_obs, min_abs_mean, opener, max_block_bytes, spec, n_contributing, required,
            tmp_paths,
        )
    except BaseException:
        # A half-written block leaves the .tmp files behind; drop them so the
        # next run starts from a clean slate rather than a truncated raster.
        for tmp_path in tmp_paths.values():
            Path(tmp_path).unlink(missing_ok=True)
        raise


def _compute_temporal_bands_impl(
    target, contributors, cache_path_for, out_paths, readonly_dirs, ddof,
    min_obs, min_abs_mean, opener, max_block_bytes, spec, n_contributing, required,
    tmp_paths,
):
    """Body of :func:`compute_temporal_bands`, split out so its caller can clean
    up partially written ``.tmp`` outputs on any failure."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        datasets = []
        for snap in contributors:
            path = Path(cache_path_for(snap))
            if not path.exists():
                raise BackfillError(
                    f"Cached source band missing for {snap.prefix}: {path}. Re-run Phase 1."
                )
            datasets.append((snap, stack.enter_context(opener(path))))

        grid = assert_grid_identity(
            [(snap.prefix, grid_from_dataset(src)) for snap, src in datasets],
            context=f"{target.sensor} window ending {target.end_iso}",
        )

        block_shape = choose_block_shape(datasets[0][1].block_shapes[0], n_contributing, max_block_bytes)
        profile = single_band_profile(grid)
        count_profile = dict(profile)
        count_profile.update(dtype="int32", nodata=0, predictor=2)

        writers: dict[str, rasterio.io.DatasetWriter] = {}
        for band in required:
            out_path = Path(assert_not_in_readonly_dir(out_paths[band], readonly_dirs))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_paths[band] = out_path.with_name(out_path.name + ".tmp")
            band_profile = count_profile if band == spec.count_band else profile
            writer = stack.enter_context(rasterio.open(tmp_paths[band], "w", **band_profile))
            writer.set_band_description(1, band)
            writer.update_tags(
                band_name=band,
                sensor=target.sensor,
                source_band=spec.source_band,
                source_prefix=target.prefix,
                target_prefix=target.target_prefix(),
                acquisition_date=target.start_iso,
                window_start=window_bounds(target.end_date, spec.lookback_days)[0].isoformat(),
                window_end_exclusive=target.end_iso,
                lookback_days=str(spec.lookback_days),
                min_obs=str(min_obs),
                ddof=str(ddof),
                n_contributing_scenes=str(n_contributing),
                contributing_dates=",".join(s.start_iso for s in contributors),
                reconstruction="local backfill from exported source band; no Earth Engine",
            )
            writers[band] = writer

        valid_pixels = 0
        n_nonfinite_cv = 0
        for window in plan_blocks(grid.width, grid.height, block_shape):
            layers = [src.read(1, window=window) for _, src in datasets]
            count, mean, std = temporal_block_stats(layers, PREDICTOR_NODATA_VALUE, ddof=ddof)
            del layers

            std_masked = apply_min_obs(std, count, min_obs)
            enough = count >= min_obs
            valid_pixels += int(np.count_nonzero(enough & np.isfinite(std_masked)))

            writers[spec.std_band].write(
                np.where(np.isfinite(std_masked), std_masked, PREDICTOR_NODATA_VALUE).astype(np.float32),
                1, window=window,
            )
            if spec.cv_band is not None:
                cv, near_zero = temporal_cv(std, mean, min_abs_mean=min_abs_mean)
                cv_masked = apply_min_obs(cv, count, min_obs)
                n_nonfinite_cv += int(np.count_nonzero(near_zero & enough))
                writers[spec.cv_band].write(
                    np.where(np.isfinite(cv_masked), cv_masked, PREDICTOR_NODATA_VALUE).astype(np.float32),
                    1, window=window,
                )
            writers[spec.count_band].write(count.astype(np.int32), 1, window=window)

    for band in required:
        final = Path(out_paths[band])
        os.replace(tmp_paths[band], final)

    return {
        "status": "completed",
        "n_contributing_scenes": n_contributing,
        "valid_pixels": valid_pixels,
        "n_nonfinite_cv": n_nonfinite_cv,
        "grid": grid,
        "message": (
            f"{n_contributing} scene(s) in window; ddof={ddof}; min_obs={min_obs}"
        ),
    }


# ---------------------------------------------------------------------------
# Phase 3: VRT assembly.
# ---------------------------------------------------------------------------

def _pretty_xml(element: ET.Element) -> str:
    raw = ET.tostring(element, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def build_vrt_xml(grid: GridSpec, bands: Sequence[Mapping], nodata: float = PREDICTOR_NODATA_VALUE) -> str:
    """Serialise a GDAL ``.vrt`` stacking sources into a named band order.

    ``bands`` is an ordered sequence of mappings::

        {'name': 'NDVI', 'dtype': 'Float32',
         'sources': [{'path': Path(...), 'band': 6,
                      'src_window': (xoff, yoff, xsize, ysize),   # optional
                      'dst_window': (xoff, yoff, xsize, ysize)}]} # optional

    Every band gets its ``<Description>`` set to exactly the schema band name, so
    ``rasterio.open(vrt).descriptions`` satisfies the classifier's
    ``_validate_s2_predictor_band_order`` unchanged. Sources are written as
    ``ComplexSource`` with an explicit ``<NODATA>`` so mosaicked tile shards
    never paint NoData over a neighbour's valid data.
    """
    root = ET.Element("VRTDataset", rasterXSize=str(grid.width), rasterYSize=str(grid.height))
    ET.SubElement(root, "SRS").text = grid.crs
    a, b, c, d, e, f = grid.transform
    ET.SubElement(root, "GeoTransform").text = f"{c}, {a}, {b}, {f}, {d}, {e}"

    for index, band in enumerate(bands, start=1):
        band_el = ET.SubElement(
            root, "VRTRasterBand",
            dataType=str(band.get("dtype", "Float32")),
            band=str(index),
        )
        ET.SubElement(band_el, "Description").text = str(band["name"])
        band_nodata = band.get("nodata", nodata)
        if band_nodata is not None:
            ET.SubElement(band_el, "NoDataValue").text = repr(float(band_nodata))
        for source in band["sources"]:
            src_el = ET.SubElement(band_el, "ComplexSource")
            filename = ET.SubElement(src_el, "SourceFilename", relativeToVRT="0")
            filename.text = str(Path(source["path"]))
            ET.SubElement(src_el, "SourceBand").text = str(source.get("band", 1))
            src_window = source.get("src_window")
            dst_window = source.get("dst_window")
            if src_window:
                ET.SubElement(
                    src_el, "SrcRect",
                    xOff=str(src_window[0]), yOff=str(src_window[1]),
                    xSize=str(src_window[2]), ySize=str(src_window[3]),
                )
            if dst_window:
                ET.SubElement(
                    src_el, "DstRect",
                    xOff=str(dst_window[0]), yOff=str(dst_window[1]),
                    xSize=str(dst_window[2]), ySize=str(dst_window[3]),
                )
            if band_nodata is not None:
                ET.SubElement(src_el, "NODATA").text = repr(float(band_nodata))
    return _pretty_xml(root)


def plan_vrt_bands(
    snapshot: Snapshot,
    sidecar_paths: Mapping[str, Path],
    opener=rasterio.open,
    reader=None,
) -> tuple[GridSpec, list[dict]]:
    """Map every band of the sensor's new schema to a concrete (file, band).

    Source bands are resolved **by description** from the exported GeoTIFF(s);
    the new bands come from their sidecars. Returns the union grid and an ordered
    band plan in exactly ``S2_PREDICTORS`` / ``S1_PREDICTORS`` order.
    """
    from contextlib import ExitStack, nullcontext

    spec = sensor_spec(snapshot.sensor)
    with ExitStack() as stack:
        opened = []
        for path in snapshot.paths:
            local = stack.enter_context(reader(path) if reader is not None else nullcontext(path))
            src = stack.enter_context(opener(local))
            opened.append((Path(path), src, grid_from_dataset(src)))
        grid = union_grid([g for *_, g in opened])

        band_plan: list[dict] = []
        for name in spec.temporal_predictors:
            if name in spec.new_bands:
                sidecar = sidecar_paths.get(name)
                if sidecar is None:
                    raise BackfillError(f"No sidecar supplied for band {name!r} of {snapshot.prefix}")
                band_plan.append({
                    "name": name,
                    "dtype": "Float32",
                    "sources": [{
                        "path": Path(sidecar), "band": 1,
                        "src_window": (0, 0, grid.width, grid.height),
                        "dst_window": (0, 0, grid.width, grid.height),
                    }],
                })
                continue
            sources = []
            for path, src, src_grid in opened:
                band_index = band_index_by_description(src, name, label=path.name)
                col_off, row_off = _grid_offset(grid, src_grid)
                sources.append({
                    "path": path, "band": band_index,
                    "src_window": (0, 0, src_grid.width, src_grid.height),
                    "dst_window": (col_off, row_off, src_grid.width, src_grid.height),
                })
            band_plan.append({"name": name, "dtype": "Float32", "sources": sources})
    return grid, band_plan


def write_snapshot_vrt(
    snapshot: Snapshot,
    sidecar_paths: Mapping[str, Path],
    out_path,
    readonly_dirs: Sequence[Path] = (),
    opener=rasterio.open,
    reader=None,
) -> Path:
    """Write the per-snapshot ``.vrt`` and verify its band order before returning."""
    spec = sensor_spec(snapshot.sensor)
    out_path = Path(assert_not_in_readonly_dir(out_path, readonly_dirs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid, band_plan = plan_vrt_bands(snapshot, sidecar_paths, opener=opener, reader=reader)
    xml = build_vrt_xml(grid, band_plan)

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(xml, encoding="utf-8")
    with opener(tmp_path) as src:
        descriptions = list(src.descriptions or ())
        if descriptions != list(spec.temporal_predictors):
            tmp_path.unlink(missing_ok=True)
            raise BackfillError(
                f"VRT band order verification failed for {out_path.name}.\n"
                f"  expected: {list(spec.temporal_predictors)}\n"
                f"  actual:   {descriptions}"
            )
    os.replace(tmp_path, out_path)
    return out_path


def rewrite_snapshot_geotiff(
    snapshot: Snapshot,
    sidecar_paths: Mapping[str, Path],
    out_path,
    readonly_dirs: Sequence[Path] = (),
    opener=rasterio.open,
    reader=None,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
) -> Path:
    """Materialise a real full-schema GeoTIFF (21-band S2 / 5-band S1).

    Optional: the default deliverable is sidecars plus a ``.vrt``. This writes to
    a temporary path, verifies band count, band order, grid and NoData, and only
    then moves the file into place. It never deletes or overwrites a source file
    - the read-only guard refuses any target inside the export archive.
    """
    from contextlib import ExitStack, nullcontext

    spec = sensor_spec(snapshot.sensor)
    out_path = Path(assert_not_in_readonly_dir(out_path, readonly_dirs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid, band_plan = plan_vrt_bands(snapshot, sidecar_paths, opener=opener, reader=reader)

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    profile = single_band_profile(grid)
    profile.update(count=len(band_plan))

    with ExitStack() as stack:
        open_sources: dict[Path, rasterio.io.DatasetReader] = {}

        def _source(path: Path):
            if path not in open_sources:
                local = stack.enter_context(reader(path) if reader is not None else nullcontext(path))
                open_sources[path] = stack.enter_context(opener(local))
            return open_sources[path]

        dst = stack.enter_context(rasterio.open(tmp_path, "w", **profile))
        for index, band in enumerate(band_plan, start=1):
            dst.set_band_description(index, band["name"])
        dst.update_tags(
            sensor=snapshot.sensor,
            source_prefix=snapshot.prefix,
            target_prefix=snapshot.target_prefix(),
            band_order=",".join(spec.temporal_predictors),
            reconstruction="local backfill from exported source band; no Earth Engine",
        )
        block_shape = choose_block_shape((512, 512), 1, max_block_bytes)
        for index, band in enumerate(band_plan, start=1):
            for window in plan_blocks(grid.width, grid.height, block_shape):
                out_block = np.full(
                    (int(window.height), int(window.width)), PREDICTOR_NODATA_VALUE, dtype=np.float32
                )
                for source in band["sources"]:
                    src = _source(Path(source["path"]))
                    dst_x, dst_y, dst_w, dst_h = source.get(
                        "dst_window", (0, 0, grid.width, grid.height)
                    )
                    x0 = max(int(window.col_off), dst_x)
                    y0 = max(int(window.row_off), dst_y)
                    x1 = min(int(window.col_off) + int(window.width), dst_x + dst_w)
                    y1 = min(int(window.row_off) + int(window.height), dst_y + dst_h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    src_window = Window(
                        col_off=x0 - dst_x, row_off=y0 - dst_y, width=x1 - x0, height=y1 - y0
                    )
                    data = src.read(int(source.get("band", 1)), window=src_window).astype(np.float32)
                    target_slice = (
                        slice(y0 - int(window.row_off), y1 - int(window.row_off)),
                        slice(x0 - int(window.col_off), x1 - int(window.col_off)),
                    )
                    patch = out_block[target_slice]
                    out_block[target_slice] = np.where(data == PREDICTOR_NODATA_VALUE, patch, data)
                dst.write(out_block, index, window=window)

    with opener(tmp_path) as verify:
        problems = []
        if verify.count != len(spec.temporal_predictors):
            problems.append(f"band count {verify.count} != {len(spec.temporal_predictors)}")
        if list(verify.descriptions or ()) != list(spec.temporal_predictors):
            problems.append(f"band order {list(verify.descriptions or ())}")
        if grid_from_dataset(verify) != grid:
            problems.append("grid mismatch against the planned union grid")
        if verify.nodata != PREDICTOR_NODATA_VALUE:
            problems.append(f"nodata {verify.nodata} != {PREDICTOR_NODATA_VALUE}")
    if problems:
        Path(tmp_path).unlink(missing_ok=True)
        raise BackfillError(
            f"Rewritten GeoTIFF failed verification for {out_path.name}: " + "; ".join(problems)
        )
    os.replace(tmp_path, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Manifests. Mirrors Batch_Export.ipynb section 8 so runs stay resumable and
# auditable in the same shape as the existing export manifests.
# ---------------------------------------------------------------------------

CACHE_MANIFEST_COLUMNS = [
    "sensor",
    "prefix",
    "start_date",
    "end_date",
    "schema_token",
    "source_paths",
    "source_mtimes",
    "source_sizes",
    "cache_path",
    "cache_bytes",
    "width",
    "height",
    "crs",
    "status",
    "message",
    "recorded_at",
]

RUN_MANIFEST_COLUMNS = [
    "sensor",
    "prefix",
    "target_prefix",
    "start_date",
    "end_date",
    "n_contributing_scenes",
    "valid_pixels",
    "n_nonfinite_cv",
    "status",
    "message",
    "recorded_at",
]


def utc_now_iso() -> str:
    """Timestamp in the same shape as Batch_Export.ipynb's ``utc_now_iso``."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def empty_manifest(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def load_manifest(path, columns: Sequence[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return empty_manifest(columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(columns)]


def save_manifest(frame: pd.DataFrame, path, columns: Sequence[str], readonly_dirs: Sequence[Path] = ()) -> pd.DataFrame:
    path = Path(assert_not_in_readonly_dir(path, readonly_dirs))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    out = frame[list(columns)]
    tmp_path = path.with_name(path.name + ".tmp")
    out.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    return out


def upsert_manifest_record(
    record: Mapping,
    path,
    columns: Sequence[str],
    frame: pd.DataFrame | None = None,
    key: str = "prefix",
    readonly_dirs: Sequence[Path] = (),
) -> pd.DataFrame:
    """Insert or replace one row, keyed by ``key``, and persist the manifest."""
    if frame is None:
        frame = load_manifest(path, columns)
    frame = frame.copy()
    if len(frame) and key in frame.columns:
        match = frame[key].astype(str).eq(str(record[key]))
        if match.any():
            for column in columns:
                frame.loc[match, column] = record.get(column)
            return save_manifest(frame, path, columns, readonly_dirs)
    frame = pd.concat([frame, pd.DataFrame([dict(record)])], ignore_index=True)
    return save_manifest(frame, path, columns, readonly_dirs)


# ---------------------------------------------------------------------------
# Phase 0: sensor-generic estimator validation.
# ---------------------------------------------------------------------------

VALIDATION_UNVALIDATED = "UNVALIDATED"
VALIDATION_PASS = "PASS"
VALIDATION_FAIL = "FAIL"


def compare_to_reference(
    local: np.ndarray,
    reference: np.ndarray,
    water: np.ndarray | None = None,
    nodata: float = PREDICTOR_NODATA_VALUE,
) -> dict:
    """Compare a locally reconstructed band against Earth Engine's own values.

    ``local`` and ``reference`` are same-shaped arrays; ``water`` optionally
    restricts the comparison (the exports apply the JRC water mask, so a valid
    source-band pixel is already a water pixel).

    Reports Pearson correlation, mean bias (local - GEE), RMSE and the
    5th/50th/95th percentiles of the difference over pixels valid in **both**,
    plus masking disagreement split by direction.
    """
    local_arr = np.asarray(local, dtype=np.float64)
    ref_arr = np.asarray(reference, dtype=np.float64)
    local_ok = valid_mask(local_arr, nodata)
    ref_ok = valid_mask(ref_arr, nodata)
    if water is not None:
        water_mask = np.asarray(water, dtype=bool)
        local_ok &= water_mask
        ref_ok &= water_mask
        considered = int(np.count_nonzero(water_mask))
    else:
        considered = int(local_arr.size)

    both = local_ok & ref_ok
    n_both = int(np.count_nonzero(both))
    only_local = int(np.count_nonzero(local_ok & ~ref_ok))
    only_reference = int(np.count_nonzero(ref_ok & ~local_ok))

    out = {
        "n_considered": considered,
        "n_valid_both": n_both,
        "n_valid_local_only": only_local,
        "n_valid_reference_only": only_reference,
        "frac_local_only": (only_local / considered) if considered else np.nan,
        "frac_reference_only": (only_reference / considered) if considered else np.nan,
        "pearson_r": np.nan,
        "mean_bias": np.nan,
        "rmse": np.nan,
        "diff_p5": np.nan,
        "diff_p50": np.nan,
        "diff_p95": np.nan,
        "n_nonfinite_local": int(np.count_nonzero(~np.isfinite(local_arr))),
        "n_nonfinite_reference": int(np.count_nonzero(~np.isfinite(ref_arr))),
    }
    if n_both < 2:
        return out

    x = local_arr[both]
    y = ref_arr[both]
    diff = x - y
    out["mean_bias"] = float(np.mean(diff))
    out["rmse"] = float(np.sqrt(np.mean(diff * diff)))
    out["diff_p5"], out["diff_p50"], out["diff_p95"] = (
        float(v) for v in np.percentile(diff, [5, 50, 95])
    )
    if np.std(x) > 0 and np.std(y) > 0:
        out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    return out


def aggregate_comparisons(per_date: Sequence[Mapping]) -> dict:
    """Observation-weighted aggregate of :func:`compare_to_reference` results."""
    rows = [dict(r) for r in per_date if r and r.get("n_valid_both", 0) >= 2]
    if not rows:
        return {"n_dates": 0, "n_valid_both": 0}
    weights = np.array([r["n_valid_both"] for r in rows], dtype=np.float64)
    total = float(weights.sum())

    def _weighted(key):
        values = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
        ok = np.isfinite(values)
        return float(np.sum(values[ok] * weights[ok]) / weights[ok].sum()) if ok.any() else np.nan

    # RMSE pools in quadrature, not linearly: sqrt(sum(w * rmse^2) / sum(w)).
    rmse_values = np.array([r.get("rmse", np.nan) for r in rows], dtype=np.float64)
    rmse_ok = np.isfinite(rmse_values)
    pooled_rmse = (
        float(np.sqrt(np.sum(weights[rmse_ok] * rmse_values[rmse_ok] ** 2) / weights[rmse_ok].sum()))
        if rmse_ok.any() else np.nan
    )

    return {
        "n_dates": len(rows),
        "n_valid_both": int(total),
        "pearson_r": _weighted("pearson_r"),
        "mean_bias": _weighted("mean_bias"),
        "rmse": pooled_rmse,
        "diff_p5": _weighted("diff_p5"),
        "diff_p50": _weighted("diff_p50"),
        "diff_p95": _weighted("diff_p95"),
        "frac_local_only": _weighted("frac_local_only"),
        "frac_reference_only": _weighted("frac_reference_only"),
    }


def choose_ddof(ddof0: Mapping, ddof1: Mapping) -> tuple[int, str]:
    """Pick the reducer convention that actually fits Earth Engine's output.

    Compares aggregate RMSE (falling back to ``|mean bias|`` when RMSE is
    unavailable) and returns ``(ddof, explanation)``.
    """
    def _score(agg):
        rmse = agg.get("rmse", np.nan)
        if np.isfinite(rmse):
            return float(rmse)
        bias = agg.get("mean_bias", np.nan)
        return float(abs(bias)) if np.isfinite(bias) else np.inf

    score0, score1 = _score(ddof0), _score(ddof1)
    if not np.isfinite(score0) and not np.isfinite(score1):
        return TEMPORAL_DDOF, (
            "no comparable pixels; keeping the documented default "
            f"ddof={TEMPORAL_DDOF} (ee.Reducer.stdDev is a sample std dev)"
        )
    if score1 <= score0:
        return 1, f"ddof=1 fits better (RMSE {score1:.6g} vs {score0:.6g} for ddof=0)"
    return 0, f"ddof=0 fits better (RMSE {score0:.6g} vs {score1:.6g} for ddof=1)"
