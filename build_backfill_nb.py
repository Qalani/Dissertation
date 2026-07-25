"""Builder script that assembles Backfill_Temporal_Bands_Local.ipynb.

Kept in the repo (like build_inventory_nb.py) so the backfill notebook can be
regenerated/edited in one place instead of hand-editing notebook JSON.

The notebook is a thin wiring layer: every piece of logic it runs lives in
``winam_diagnostics/temporal_backfill.py`` and is unit-tested offline in
``tests/test_temporal_backfill.py``.
"""
import json

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    })


# ---------------------------------------------------------------------------
md("""# Backfill temporal-persistence bands locally (Sentinel-2 + Sentinel-1)

`Batch_Export.ipynb` bumped both predictor export schemas to add snapshot-relative
90-day temporal-persistence bands:

| | S2 | S1 |
|---|---|---|
| Old schema | `s2_whlev_texture_v1` (20 bands) | `winam_s1_scc_predictors_*` (3 bands) |
| New schema | `s2_whlev_temporal_v1` (21 bands) | `s1_scc_temporal_v1` (5 bands) |
| Old prefix | `winam_s2_predictors_s2_whlev_texture_v1_{start}_to_{end}` | `winam_s1_scc_predictors_{start}_to_{end}` |
| New prefix | `winam_s2_predictors_s2_whlev_temporal_v1_{start}_to_{end}` | `winam_s1_scc_temporal_v1_{start}_to_{end}` |
| Source band | `NDVI` | `VH_corrected` |
| New band(s) | `ndvi_temporal_std_w90` | `vh_temporal_std_w90`, `vh_temporal_cv_w90` |

Re-exporting the archive from Earth Engine is not viable: the project is out of
monthly EECU quota, with roughly 358 S2 snapshots (~448 GB) and 456 S1 snapshots
(~162 GB) to redo. **All three new bands are functions of a band that is already
present in every exported snapshot**, so they can be reconstructed locally at
zero Earth Engine cost.

> **Note the S1 prefix asymmetry.** S2 kept `winam_s2_predictors_` and inserted the
> schema token. S1 dropped `predictors` entirely. Discovery and VRT naming here
> never assume a simple token substitution for S1.

## What this notebook does

| Phase | What | Output |
|---|---|---|
| **0** | Inventory, manifest breakdown, estimator validation against Earth Engine's own values. **Blocking** — must finish and print its verdict before any bulk work, and not in the same run. | `phase0_report.json`, diagnostics figures |
| **1** | Extract `NDVI` / `VH_corrected` once per snapshot into a compressed single-band cache | cache GeoTIFFs + cache manifest |
| **2** | Rolling 90-day statistics per snapshot, in windowed blocks | in memory |
| **3** | Write sidecar GeoTIFFs + a `.vrt` per snapshot in the new-schema band order | sidecars, VRTs, run manifest |
| **4** | Classifier integration is **documented, not applied** | `docs/temporal_backfill_integration.md` |

## Hard constraints

- **`GEE_Exports_validated_snapshots` is read-only. No exceptions.** It holds ~610 GB
  of exports that cannot be cheaply regenerated. Every write path in this notebook
  passes through a guard that raises `ReadOnlyPathError` if it resolves inside it.
- **No Earth Engine.** No `import ee`, no authentication, no network calls to Earth
  Engine anywhere. The setup cell asserts this.
- Every phase is resumable and idempotent; safe to interrupt and re-run.
- Bounded memory, windowed I/O throughout. The full stack is never held in RAM.
- Sensors never mix in a window. S1 and S2 stacks are independent.

## Window semantics (easy to get wrong)

Both `add_s2_temporal_stability` and `add_s1_temporal_stability` build the window as
`ee.Date(end_date).advance(-90, 'day')` to `end_date`, and Earth Engine's `filterDate`
is **inclusive of start, exclusive of end**. Each snapshot's `end_date` is its
`start_date + 1 day`, so **the window includes the snapshot's own acquisition date**.
That is reproduced exactly and pinned by `tests/test_temporal_backfill.py`.

## Known departure from Earth Engine

Earth Engine reduced over **every scene passing the source-collection filters**. This
reconstruction can only see snapshots that passed the export coverage gate and reached
Drive, and it sees one median-composited observation per acquisition date where Earth
Engine saw each granule. Phase 0 measures the resulting difference against real Earth
Engine output rather than assuming it away.
""")

# ---------------------------------------------------------------------------
md("## 1. Setup — Drive, imports, and the offline logic package")

code("""# Mount Google Drive (Colab). Outside Colab this is skipped and you can point
# DRIVE_MYDRIVE at any folder mirroring the Drive layout.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception as exc:
    print('Google Drive not mounted automatically (not in Colab?):', exc)
    IN_COLAB = False

import json
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from IPython.display import display


def _ensure_winam_diagnostics():
    '''Make the winam_diagnostics package importable (no Drive, no Earth Engine).'''
    try:
        import winam_diagnostics  # noqa: F401
        return True
    except ModuleNotFoundError:
        pass
    for cand in ['/content/Dissertation', str(Path.cwd()), str(Path.cwd().parent),
                 '/content/Dissertation_diagnostics']:
        if (Path(cand) / 'winam_diagnostics' / '__init__.py').exists():
            sys.path.insert(0, cand)
            try:
                import winam_diagnostics  # noqa: F401
                return True
            except ModuleNotFoundError:
                continue
    clone = '/content/Dissertation_diagnostics'
    if not (Path(clone) / 'winam_diagnostics' / '__init__.py').exists():
        import subprocess
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/Qalani/Dissertation.git', clone], check=False)
    if (Path(clone) / 'winam_diagnostics' / '__init__.py').exists():
        sys.path.insert(0, clone)
        import winam_diagnostics  # noqa: F401
        return True
    return False


if not _ensure_winam_diagnostics():
    raise ImportError(
        'winam_diagnostics not found. Clone the Dissertation repo alongside this '
        'notebook (or run it from the repo root) so temporal_backfill is importable.'
    )

from winam_diagnostics import temporal_backfill as tb

REPO_ROOT = Path(tb.__file__).resolve().parent.parent

# Hard constraint: this notebook must never touch Earth Engine. Importing ee would
# also mean an authentication prompt and network calls, so fail loudly if anything
# in the import chain pulled it in.
_leaked_ee = sorted(m for m in sys.modules if m == 'ee' or m.startswith('ee.'))
assert not _leaked_ee, f'Earth Engine modules leaked into this runtime: {_leaked_ee}'

# Identifies this notebook run. Phase 0 stamps its report with it so the later
# phases can refuse to run in the same session (see require_phase0).
SESSION_ID = uuid.uuid4().hex

print('winam_diagnostics loaded from:', Path(tb.__file__).resolve().parent)
print('Repo root:', REPO_ROOT)
print('rasterio:', rasterio.__version__, '| numpy:', np.__version__)
print('Earth Engine imported:', bool(_leaked_ee), '(must be False)')
print('Session id:', SESSION_ID)""")

# ---------------------------------------------------------------------------
md("""## 2. Configuration

Everything tunable lives here. The defaults reproduce `Batch_Export.ipynb` exactly.

`TEMPORAL_DDOF` starts at the module default (1, matching Earth Engine's documented
*sample* standard deviation) but **Phase 0 settles it empirically** and writes the
fitted value into the Phase 0 report; Phase 2 reads it back from there.""")

code("""# --- Read-only source archive. Never written to, never moved, never deleted. ---
EE_EXPORT_FOLDER = 'GEE_Exports_validated_snapshots'
GEE_EXPORT_DIR = Path('/content/drive/MyDrive') / EE_EXPORT_FOLDER
READONLY_DIRS = [GEE_EXPORT_DIR]

EXPORT_RUN_MANIFEST_PATH = GEE_EXPORT_DIR / 'winam_snapshot_validated_predictor_manifest.csv'
EXPORT_PLANNED_MANIFEST_PATH = GEE_EXPORT_DIR / 'winam_snapshot_validated_predictor_planned_rows.csv'
EXPORT_PENDING_MANIFEST_PATH = GEE_EXPORT_DIR / 'winam_snapshot_validated_predictor_pending_rows.csv'

# --- Everything this notebook writes lives here, in its own Drive folder. ---
# Deliberately NOT inside the export folder: Batch_Export.ipynb's
# export_prefix_exists / SKIP_PREFIXES_ALREADY_IN_DRIVE and the classifier's
# discovery both scan the export dir by prefix, and sidecars there would confuse
# both resume states.
BACKFILL_ROOT = Path('/content/drive/MyDrive') / 'Winam_Temporal_Backfill'
CACHE_DIR = BACKFILL_ROOT / 'source_band_cache'      # per-sensor subfolders
SIDECAR_DIR = BACKFILL_ROOT / 'sidecars'             # per-sensor subfolders
VRT_DIR = BACKFILL_ROOT / 'vrt'                      # per-sensor subfolders
REWRITE_DIR = BACKFILL_ROOT / 'rewritten'            # per-sensor subfolders
REPORT_DIR = BACKFILL_ROOT / 'reports'

CACHE_MANIFEST_PATH = REPORT_DIR / 'winam_temporal_backfill_cache_manifest.csv'
RUN_MANIFEST_PATH = REPORT_DIR / 'winam_temporal_backfill_run_manifest.csv'
INVENTORY_CSV_PATH = REPORT_DIR / 'winam_temporal_backfill_inventory.csv'
PHASE0_REPORT_PATH = REPORT_DIR / 'phase0_report.json'

# Figures go in the repo alongside the other diagnostics, not on Drive.
DIAGNOSTICS_DIR = REPO_ROOT / 'outputs' / 'diagnostics' / 'temporal_backfill'

# Local scratch for staging Drive reads/writes (block I/O over the FUSE mount is
# far slower than one sequential copy). Ephemeral; safe to lose.
LOCAL_SCRATCH = Path('/content/winam_backfill_scratch') if IN_COLAB else Path('/tmp/winam_backfill_scratch')

# --- Which sensors to process. ---
SENSORS = ['S2', 'S1']

# --- Statistics. TEMPORAL_DDOF is overridden by the Phase 0 fit. ---
TEMPORAL_LOOKBACK_DAYS = tb.TEMPORAL_LOOKBACK_DAYS   # 90
TEMPORAL_MIN_OBS = tb.TEMPORAL_MIN_OBS               # 3
TEMPORAL_DDOF = tb.TEMPORAL_DDOF                     # 1 until Phase 0 says otherwise
S1_CV_MIN_ABS_MEAN = tb.S1_CV_MIN_ABS_MEAN           # near-zero-mean policy, one constant

# Peak bytes for one block of the temporal stack. Sized for a ~12 GB runtime.
MAX_BLOCK_BYTES = 512 * 1024 * 1024

# --- Drive resilience. ---
# Colab drops the Drive FUSE mount mid-run (OSError Errno 107, "Transport
# endpoint is not connected"). Every input and output here lives on Drive, so the
# only recovery is a force-remount and retry. Scanning a large archive can hit a
# stale endpoint several times, so the budget is generous.
MAX_DRIVE_REMOUNTS = 25

# --- Phase 0 controls. ---
PHASE0_MAX_VALIDATION_DATES = 8       # reference dates to recompute and compare
PHASE0_MAX_CLASSIFY_PREFIXES = None   # None = resolve every multi-file prefix
CV_EXTREME_THRESHOLD = 10.0           # |cv| above this counts as 'extreme' in the S1 report

# What to do when a multi-file prefix cannot be resolved into shards-vs-copies.
#   'abort'   — stop Phase 0 (default). Resolving this is Phase 0's whole job:
#               treating copies as shards double-counts observations in every
#               overlapping window, and treating shards as copies drops coverage.
#   'exclude' — drop the unresolved snapshots from the working set and carry on.
#               Their dates then contribute nothing to any window, which is a
#               known, recorded loss rather than a silent miscount.
# Never guess: there is no 'assume shards' option on purpose.
UNRESOLVED_PREFIX_POLICY = 'abort'

# --- Output controls. ---
# Default deliverable is sidecars + a .vrt, NOT a rewrite of the source rasters.
WRITE_SIDECARS = True
WRITE_VRT = True
# Materialise real 21-band (S2) / 5-band (S1) GeoTIFFs instead. Settable per sensor:
# S1 rewriting is roughly an order of magnitude cheaper than S2
# (~175 MB vs ~1.29 GB per file).
REWRITE_IN_PLACE = {'S2': False, 'S1': False}

# Section 7 always writes the Phase 0 validation summary to REPORT_DIR on Drive.
# When True it ALSO overwrites the committed baseline at
# docs/temporal_backfill_validation_summary.md with this run's real numbers — which
# is the point of the deliverable, but it does dirty the git tree. Set False to
# leave the repo copy alone.
UPDATE_REPO_VALIDATION_SUMMARY = True

# Resume behaviour. Both default to skipping work that is already on Drive.
SKIP_CACHED = True
SKIP_COMPLETED = True
MAX_TARGETS_PER_RUN = None            # set an int for a smoke test

for _dir in [BACKFILL_ROOT, CACHE_DIR, SIDECAR_DIR, VRT_DIR, REPORT_DIR, LOCAL_SCRATCH]:
    tb.assert_not_in_readonly_dir(_dir, READONLY_DIRS)
    _dir.mkdir(parents=True, exist_ok=True)
try:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    DIAGNOSTICS_DIR = REPORT_DIR / 'figures'
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

print('Read-only export archive :', GEE_EXPORT_DIR, '(exists:', GEE_EXPORT_DIR.exists(), ')')
print('Backfill output root     :', BACKFILL_ROOT)
print('Diagnostics figures      :', DIAGNOSTICS_DIR)
print('Sensors                  :', SENSORS)
print('Lookback / min obs / ddof:', TEMPORAL_LOOKBACK_DAYS, '/', TEMPORAL_MIN_OBS, '/', TEMPORAL_DDOF)
print('S1 CV near-zero policy   : mask when abs(mean) <', S1_CV_MIN_ABS_MEAN)
print('Rewrite in place         :', REWRITE_IN_PLACE)
print()
for sensor in SENSORS:
    spec = tb.sensor_spec(sensor)
    print(f'{sensor}: source band {spec.source_band!r} -> new bands {list(spec.new_bands)} '
          f'+ count band {spec.count_band!r}')
    print(f'    legacy prefix  : {spec.legacy_prefix(\"YYYY-MM-DD\", \"YYYY-MM-DD\")}')
    print(f'    new prefix     : {spec.temporal_prefix(\"YYYY-MM-DD\", \"YYYY-MM-DD\")}')""")

# ---------------------------------------------------------------------------
md("""### 2a. Helpers — path layout, Drive staging, and the Phase 0 gate

Every cell below is safe to re-run out of order: each either succeeds
idempotently or fails loudly with a reason.""")

code("""# ---------------------------------------------------------------------------
# Drive resilience. Colab drops the Drive FUSE mount mid-run; every input and
# output here lives on Drive, so a stale endpoint has to be recovered rather
# than recorded as a data problem.
# ---------------------------------------------------------------------------
_DRIVE_REMOUNTS_REMAINING = MAX_DRIVE_REMOUNTS


def _is_transport_endpoint_error(exc):
    '''True for a stale Colab/Drive FUSE endpoint. Tested in temporal_backfill.'''
    return tb.is_transport_endpoint_error(exc)


def _looks_like_stale_drive(exc, path=None):
    '''Transport failure, or a read error on a Drive path that no longer stats.

    rasterio raises RasterioIOError, which subclasses OSError but carries
    errno=None and a GDAL message that does NOT contain 'Transport endpoint is
    not connected'. The errno/message test alone therefore misclassifies a dead
    mount as an unreadable file -- which is how a whole archive scan can come
    back reporting every prefix as corrupt. Re-probing the path settles it.
    '''
    return tb.looks_like_stale_mount(exc, path)


def _remount_drive(mount_point='/content/drive'):
    '''Force-remount Drive to recover a stale endpoint. False when unavailable.'''
    if not IN_COLAB:
        return False
    try:
        drive.mount(mount_point, force_remount=True)
        return True
    except Exception as exc:
        print(f'Google Drive force-remount failed ({type(exc).__name__}: {exc}).')
        return False


def _remount_for_recovery(context):
    global _DRIVE_REMOUNTS_REMAINING
    if _DRIVE_REMOUNTS_REMAINING <= 0:
        print(f'Drive endpoint still stale during {context}, but the remount budget '
              f'(MAX_DRIVE_REMOUNTS={MAX_DRIVE_REMOUNTS}) is exhausted.')
        return False
    if not _remount_drive():
        print('Automatic Drive remount unavailable (not in Colab?); cannot recover.')
        return False
    _DRIVE_REMOUNTS_REMAINING -= 1
    used = MAX_DRIVE_REMOUNTS - _DRIVE_REMOUNTS_REMAINING
    print(f'Force-remounted Google Drive to recover from a stale endpoint during '
          f'{context} (remount {used}/{MAX_DRIVE_REMOUNTS}).')
    return True


def with_drive_retry(func, *args, context='operation', probe_path=None, **kwargs):
    '''Run ``func``; on a stale Drive endpoint, remount and retry.

    Non-transport errors and an exhausted remount budget propagate unchanged, so
    a genuinely corrupt file still fails loudly instead of looping.
    '''
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not _looks_like_stale_drive(exc, probe_path):
                raise
            print(f'WARNING: Drive endpoint went stale during {context} '
                  f'({type(exc).__name__}: {exc}).')
            if not _remount_for_recovery(context):
                raise


@contextmanager
def retrying_open(path, *args, **kwargs):
    '''rasterio.open with stale-Drive recovery around the open itself.'''
    dataset = with_drive_retry(
        rasterio.open, path, *args,
        context=f'opening {Path(path).name}', probe_path=path, **kwargs
    )
    try:
        yield dataset
    finally:
        dataset.close()


def _sensor_dir(root, sensor):
    path = Path(root) / sensor
    tb.assert_not_in_readonly_dir(path, READONLY_DIRS)
    with_drive_retry(lambda: path.mkdir(parents=True, exist_ok=True),
                     context=f'creating {path}', probe_path=path.parent)
    return path


def cache_path_for(snapshot):
    '''Cached single-band source raster for one snapshot.'''
    spec = tb.sensor_spec(snapshot.sensor)
    return _sensor_dir(CACHE_DIR, snapshot.sensor) / f'{snapshot.prefix}__{spec.source_band}.tif'


def sidecar_path_for(snapshot, band_name):
    '''Sidecar named after the SOURCE prefix, so provenance is visible on disk.'''
    return _sensor_dir(SIDECAR_DIR, snapshot.sensor) / f'{snapshot.prefix}_{band_name}.tif'


def sidecar_paths_for(snapshot, include_count=True):
    spec = tb.sensor_spec(snapshot.sensor)
    bands = list(spec.new_bands) + ([spec.count_band] if include_count else [])
    return {band: sidecar_path_for(snapshot, band) for band in bands}


def vrt_path_for(snapshot):
    '''VRT named under the NEW schema prefix so the classifier token gate passes.'''
    return _sensor_dir(VRT_DIR, snapshot.sensor) / f'{snapshot.target_prefix()}.vrt'


def rewrite_path_for(snapshot):
    return _sensor_dir(REWRITE_DIR, snapshot.sensor) / f'{snapshot.target_prefix()}.tif'


def _path_is_on_drive(path):
    return str(Path(path).resolve()).startswith('/content/drive/')


@contextmanager
def staged_drive_read(path):
    '''Yield a locally-readable path for ``path``.

    Same idea as the classifier's helper of this name: Drive paths are copied to
    local scratch in one sequential read and the local copy is opened for block
    I/O. Non-Drive paths are yielded unchanged, so this is safe to wrap around
    every raster read.
    '''
    path = Path(path)
    if not _path_is_on_drive(path):
        yield path
        return
    LOCAL_SCRATCH.mkdir(parents=True, exist_ok=True)
    local = LOCAL_SCRATCH / f'{uuid.uuid4().hex}_{path.name}'
    try:
        # One sequential copy is the access pattern Drive's FUSE layer handles
        # reliably; a stale endpoint mid-copy is remounted and the copy retried.
        with_drive_retry(shutil.copyfile, path, local,
                         context=f'staging {path.name}', probe_path=path)
        yield local
    finally:
        try:
            local.unlink(missing_ok=True)
        except OSError:
            pass


def _upsert_cache_record(record):
    return with_drive_retry(
        tb.upsert_manifest_record, record, CACHE_MANIFEST_PATH,
        tb.CACHE_MANIFEST_COLUMNS, frame=CACHE_MANIFEST, readonly_dirs=READONLY_DIRS,
        context='updating the cache manifest', probe_path=CACHE_MANIFEST_PATH.parent,
    )


def _upsert_run_record(record):
    return with_drive_retry(
        tb.upsert_manifest_record, record, RUN_MANIFEST_PATH,
        tb.RUN_MANIFEST_COLUMNS, frame=RUN_MANIFEST, readonly_dirs=READONLY_DIRS,
        context='updating the run manifest', probe_path=RUN_MANIFEST_PATH.parent,
    )


def require_phase0(phase_label):
    '''Refuse to run bulk work until Phase 0 has completed in an earlier run.'''
    if not PHASE0_REPORT_PATH.exists():
        raise RuntimeError(
            f'{phase_label} is blocked: Phase 0 has not been run.\\n'
            f'Run sections 3-3d to completion first; they write {PHASE0_REPORT_PATH}.'
        )
    report = json.loads(PHASE0_REPORT_PATH.read_text())
    if report.get('session_id') == SESSION_ID:
        raise RuntimeError(
            f'{phase_label} is blocked: Phase 0 completed in THIS run.\\n'
            'Phase 0 is a blocking gate — read its report, then start a fresh run '
            '(restart the runtime, or re-run the setup cell in section 1) before '
            'running the bulk phases.'
        )
    failed = [s for s, v in report.get('verdicts', {}).items() if v.get('status') == tb.VALIDATION_FAIL]
    if failed:
        raise RuntimeError(
            f'{phase_label} is blocked: Phase 0 validation FAILED for {failed}. '
            'Investigate before backfilling.'
        )
    return report


def load_phase0_report():
    if not PHASE0_REPORT_PATH.exists():
        return {}
    return json.loads(PHASE0_REPORT_PATH.read_text())


def validation_status_note(report=None):
    '''Short, honest per-sensor validation status for manifests and summaries.'''
    report = report if report is not None else load_phase0_report()
    verdicts = report.get('verdicts', {})
    if not verdicts:
        return 'validation=NOT_RUN'
    return 'validation=' + ','.join(
        f'{sensor}:{verdicts[sensor].get(\"status\", tb.VALIDATION_UNVALIDATED)}'
        for sensor in sorted(verdicts)
    )


print('Helpers defined. Phase 0 report path:', PHASE0_REPORT_PATH,
      '(exists:', PHASE0_REPORT_PATH.exists(), ')')""")

# ---------------------------------------------------------------------------
md("""## 3. Phase 0 — inventory (blocking)

Resolve prefix -> schema token -> file path(s) for every snapshot of every sensor.
Handles a single `{prefix}.tif`, tiled `{prefix}-<row>-<col>.tif` shards, extensionless
Earth Engine exports, and Google Drive collision duplicates (`foo (1).tif`), which are
**skipped, never counted**.

The multi-file resolution below matters: getting it wrong silently double-counts
observations in every window. Filenames alone are not conclusive, so each multi-file
prefix has its files' georeferencing read and compared — distinct grid windows are
genuine tile shards, identical grid windows are copies.""")

code("""inventory = tb.discover_predictor_files(GEE_EXPORT_DIR, sensors=SENSORS)
print(f'Parsable predictor files found: {len(inventory)}')

if inventory.empty:
    raise RuntimeError(f'No parsable predictor GeoTIFFs in {GEE_EXPORT_DIR}. Check the path.')

# --- Counts by sensor and schema token (duplicates shown separately) ---
inv = inventory.copy()
inv['schema_token'] = inv['schema_token'].fillna('(none / legacy tokenless)')
file_counts = (
    inv.groupby(['sensor', 'schema_token', 'is_drive_duplicate'])
       .size().rename('n_files').reset_index()
)
print()
print('=== Files by sensor / schema token ===')
display(file_counts)

ALL_SNAPSHOTS = tb.group_snapshot_files(inventory)
snapshot_rows = pd.DataFrame([{
    'sensor': s.sensor,
    'prefix': s.prefix,
    'schema_token': s.schema_token or '(none / legacy tokenless)',
    'start_date': s.start_iso,
    'end_date': s.end_iso,
    'n_files': len(s.paths),
    'n_drive_duplicates': len(s.duplicate_paths),
    'has_temporal_bands': s.has_temporal_bands,
    'target_prefix': s.target_prefix(),
} for s in ALL_SNAPSHOTS])

print()
print('=== Snapshots by sensor / schema token ===')
display(
    snapshot_rows.groupby(['sensor', 'schema_token'])
                 .agg(n_snapshots=('prefix', 'size'),
                      n_files=('n_files', 'sum'),
                      n_drive_duplicates=('n_drive_duplicates', 'sum'))
                 .reset_index()
)
print()
print('=== Files per prefix ===')
display(snapshot_rows.groupby(['sensor', 'n_files']).size().rename('n_prefixes').reset_index())""")

# ---------------------------------------------------------------------------
md("""### 3a. Resolve multi-file prefixes: tile shards or duplicates?

A previous inventory saw roughly 2 files per S1 prefix. Shards and duplicates need
opposite handling, so this reads each file's CRS, transform and shape and reports the
verdict per prefix rather than guessing from filenames.""")

code("""RESOLUTION_CSV_PATH = REPORT_DIR / 'phase0_multi_file_resolution.csv'
RESOLUTION_COLUMNS = ['sensor', 'prefix', 'n_files', 'kind', 'message', 'recorded_at']

multi_file = [s for s in ALL_SNAPSHOTS if len(s.paths) > 1]
print(f'Prefixes with more than one file: {len(multi_file)}')

# Resumable: a prefix resolved on an earlier run is not re-read. Only 'settled'
# verdicts are reused -- a prefix that failed because the Drive mount died is
# retried, because that is a mount problem, not a property of the data.
SETTLED_KINDS = {'single', 'tiled', 'duplicate_grid', 'inconsistent'}
prior = tb.load_manifest(RESOLUTION_CSV_PATH, RESOLUTION_COLUMNS)
resolved = {}
if len(prior):
    for _, row in prior.iterrows():
        if str(row['kind']) in SETTLED_KINDS:
            resolved[str(row['prefix'])] = row.to_dict()
    print(f'Reusing {len(resolved)} prefix resolution(s) from {RESOLUTION_CSV_PATH.name}')

to_check = multi_file if PHASE0_MAX_CLASSIFY_PREFIXES is None else multi_file[:PHASE0_MAX_CLASSIFY_PREFIXES]
pending = [s for s in to_check if s.prefix not in resolved]
print(f'Reading georeferencing for {len(pending)} prefix(es)...')

new_rows = []
for i, snap in enumerate(pending, start=1):
    try:
        # retrying_open recovers a stale Drive mount instead of recording it as
        # a data problem; a genuinely unreadable file still raises.
        verdict = with_drive_retry(
            tb.classify_prefix_files, snap, opener=retrying_open,
            context=f'resolving {snap.prefix}', probe_path=snap.paths[0],
        )
        row = {k: v for k, v in verdict.items() if k != 'grids'}
    except Exception as exc:
        stale = _looks_like_stale_drive(exc, snap.paths[0])
        row = {
            'sensor': snap.sensor, 'prefix': snap.prefix, 'n_files': len(snap.paths),
            'kind': 'unreadable_drive_stale' if stale else 'unreadable',
            'message': f'{type(exc).__name__}: {exc}',
        }
    row['recorded_at'] = tb.utc_now_iso()
    new_rows.append(row)
    resolved[snap.prefix] = row
    if i % 25 == 0 or i == len(pending):
        kinds = pd.Series([r['kind'] for r in new_rows]).value_counts().to_dict()
        print(f'  [{i}/{len(pending)}] {kinds}')
        # Checkpoint so an interrupted scan does not lose the work already done.
        with_drive_retry(
            tb.save_manifest,
            pd.DataFrame(list(resolved.values())), RESOLUTION_CSV_PATH,
            RESOLUTION_COLUMNS, readonly_dirs=READONLY_DIRS,
            context='saving prefix resolution', probe_path=REPORT_DIR,
        )

MULTI_FILE_VERDICTS = (
    pd.DataFrame(list(resolved.values()))[RESOLUTION_COLUMNS]
    if resolved else pd.DataFrame(columns=RESOLUTION_COLUMNS)
)
if len(MULTI_FILE_VERDICTS):
    with_drive_retry(
        tb.save_manifest, MULTI_FILE_VERDICTS, RESOLUTION_CSV_PATH,
        RESOLUTION_COLUMNS, readonly_dirs=READONLY_DIRS,
        context='saving prefix resolution', probe_path=REPORT_DIR,
    )

print()
if MULTI_FILE_VERDICTS.empty:
    print('=== Multi-file prefix resolution: every prefix maps to exactly one file. ===')
    print('Nothing to disambiguate: no tile shards and no unmarked duplicates.')
    UNRESOLVED_PREFIXES = set()
else:
    print('=== Multi-file prefix resolution ===')
    display(MULTI_FILE_VERDICTS.groupby(['sensor', 'kind']).size().rename('n_prefixes').reset_index())
    for kind in ['duplicate_grid', 'inconsistent', 'unreadable', 'unreadable_drive_stale']:
        bad = MULTI_FILE_VERDICTS[MULTI_FILE_VERDICTS['kind'] == kind]
        if len(bad):
            print()
            print(f'!!! {len(bad)} prefix(es) resolved as {kind!r}:')
            display(bad.head(10)[['sensor', 'prefix', 'n_files', 'message']])
            if kind == 'unreadable_drive_stale':
                print('    These failed because the Colab Drive FUSE mount went stale, not')
                print('    because the files are bad. Re-run this cell: settled verdicts are')
                print('    reused from the CSV and only these are retried.')
    tiled = MULTI_FILE_VERDICTS[MULTI_FILE_VERDICTS['kind'] == 'tiled']
    if len(tiled):
        print()
        print(f'{len(tiled)} prefix(es) are genuine Earth Engine tile shards covering distinct')
        print('grid windows. They are mosaicked onto their union grid when cached, so each')
        print('contributes exactly ONE observation per window.')
    UNRESOLVED_PREFIXES = set(
        MULTI_FILE_VERDICTS.loc[
            ~MULTI_FILE_VERDICTS['kind'].isin({'single', 'tiled'}), 'prefix'
        ].astype(str)
    )

# Drive collision duplicates never enter the working set.
dupes = inventory[inventory['is_drive_duplicate'].astype(bool)]
print()
print(f'Google Drive collision duplicates found and excluded: {len(dupes)}')
if len(dupes):
    display(dupes[['sensor', 'prefix', 'path']].head(10))

# Unresolved multi-file prefixes are exactly what silently double-counts (copies
# read as shards) or silently loses coverage (shards read as copies), so they are
# never guessed at.
if UNRESOLVED_PREFIXES:
    print()
    print(f'!!! {len(UNRESOLVED_PREFIXES)} multi-file prefix(es) are UNRESOLVED.')
    if UNRESOLVED_PREFIX_POLICY == 'exclude':
        before = len(ALL_SNAPSHOTS)
        ALL_SNAPSHOTS = [s for s in ALL_SNAPSHOTS if s.prefix not in UNRESOLVED_PREFIXES]
        print(f'UNRESOLVED_PREFIX_POLICY = \\'exclude\\': dropped {before - len(ALL_SNAPSHOTS)} '
              'snapshot(s). Their dates contribute nothing to any window; this is a '
              'recorded loss of coverage, not a silent miscount.')
    else:
        raise RuntimeError(
            f'{len(UNRESOLVED_PREFIXES)} multi-file prefix(es) could not be resolved into '
            'tile shards vs copies, and Phase 0 will not guess.\\n'
            'If the kind is \\'unreadable_drive_stale\\', just re-run this cell — the Drive '
            'mount is remounted automatically and settled verdicts are reused.\\n'
            'If they are genuinely unreadable, inspect them, or set '
            'UNRESOLVED_PREFIX_POLICY = \\'exclude\\' to drop those dates from the run.'
        )

# One snapshot per (sensor, acquisition date): a date re-exported under the new
# schema while the legacy file is still on Drive must not be counted twice.
SNAPSHOTS, DATE_COLLISIONS = tb.select_snapshots_by_date(ALL_SNAPSHOTS)
print()
print(f'Snapshots after de-duplicating by (sensor, acquisition date): {len(SNAPSHOTS)}')
if DATE_COLLISIONS:
    collisions_df = pd.DataFrame(DATE_COLLISIONS)
    print(f'{len(DATE_COLLISIONS)} date(s) had more than one export; kept the temporal-schema one:')
    display(collisions_df.groupby('sensor').size().rename('n_dates').reset_index())
    display(collisions_df.head(10))
else:
    print('No (sensor, date) collisions.')

with_drive_retry(snapshot_rows.to_csv, INVENTORY_CSV_PATH, index=False,
                 context='saving inventory CSV', probe_path=REPORT_DIR)
print()
print('Saved inventory:', INVENTORY_CSV_PATH)
print()
print('=== Reference exports available for validation ===')
for sensor in SENSORS:
    picked = [s for s in SNAPSHOTS if s.sensor == sensor]
    with_bands = [s for s in picked if s.has_temporal_bands]
    print(f'{sensor}: {len(picked)} snapshot(s); {len(with_bands)} already carry '
          f'Earth Engine temporal bands; {len(picked) - len(with_bands)} need backfilling.')
    if with_bands:
        print(f'    -> {sensor} CAN be validated against Earth Engine output in section 3c.')
    else:
        print(f'    -> {sensor} has no reference export; section 3c will report it '
              f'{tb.VALIDATION_UNVALIDATED}.')""")

# ---------------------------------------------------------------------------
md("""### 3b. Manifest breakdown

`winam_snapshot_validated_predictor_manifest.csv` records what Earth Engine was asked
to export and why some dates were not. The `queued` vs `skipped_low_coverage` ratio
quantifies how many observations the local reconstruction is **missing** relative to
Earth Engine, which reduced over every scene passing the source-collection filters
regardless of coverage.""")

code("""def _read_export_manifest(path):
    path = Path(path)
    if not path.exists():
        print(f'Export manifest not found: {path}')
        return pd.DataFrame()
    frame = pd.read_csv(path)
    print(f'Loaded {len(frame)} rows from {path.name}')
    return frame


EXPORT_MANIFEST = _read_export_manifest(EXPORT_RUN_MANIFEST_PATH)
PLANNED_MANIFEST = _read_export_manifest(EXPORT_PLANNED_MANIFEST_PATH)

MANIFEST_BREAKDOWN = pd.DataFrame()
COVERAGE_LOSS = {}
if not EXPORT_MANIFEST.empty and {'sensor', 'status'}.issubset(EXPORT_MANIFEST.columns):
    MANIFEST_BREAKDOWN = (
        EXPORT_MANIFEST.assign(status=EXPORT_MANIFEST['status'].astype(str).str.lower())
                       .groupby(['sensor', 'status']).size().rename('n_rows').reset_index()
    )
    print()
    print('=== Export manifest: rows by sensor and status ===')
    display(MANIFEST_BREAKDOWN)

    pivot = MANIFEST_BREAKDOWN.pivot(index='sensor', columns='status', values='n_rows').fillna(0)
    print()
    print('=== Observations the local reconstruction cannot see ===')
    for sensor in pivot.index:
        row = pivot.loc[sensor]
        exported = float(row.get('completed', 0) + row.get('queued', 0) + row.get('skipped_existing', 0))
        low_cov = float(row.get('skipped_low_coverage', 0))
        planned = 0
        if not PLANNED_MANIFEST.empty and 'sensor' in PLANNED_MANIFEST.columns:
            planned = int((PLANNED_MANIFEST['sensor'] == sensor).sum())
        denom = exported + low_cov
        frac = (low_cov / denom) if denom else float('nan')
        COVERAGE_LOSS[sensor] = {
            'planned_source_dates': planned,
            'exported_or_queued': exported,
            'skipped_low_coverage': low_cov,
            'frac_dates_missing_vs_gated': frac,
        }
        print(f'{sensor}: {int(exported)} date(s) exported/queued, {int(low_cov)} skipped for low '
              f'coverage -> {frac:.1%} of gated dates are invisible to this reconstruction.')
        if planned:
            print(f'      Earth Engine reduced over {planned} planned source date(s) for this sensor; '
                  f'the local stack sees at most {int(exported)} of them '
                  f'({exported / planned:.1%}).')
else:
    print('No usable export manifest; the coverage-loss estimate is unavailable.')
    print('This does not block the backfill, but the Phase 0 verdict records it.')""")

# ---------------------------------------------------------------------------
md("""### 3c. Estimator validation against Earth Engine's own values

**Sensor-generic by construction.** Any snapshot found carrying Earth Engine's own
temporal bands becomes ground truth and is validated automatically. Today that is S2
only, because `EXPORT_S1` is `False` and no S1 snapshot has ever been exported with
the temporal bands. If `s1_scc_temporal_v1` files ever appear in the export folder from
a separate run, S1 validation happens here with no further work.

When a sensor has no reference exports the harness reports it as **`UNVALIDATED`** —
not as passing, and not silently omitted.

For each reference date the harness reports, on water pixels only (the exports apply
the JRC water mask, so a valid source-band pixel *is* a water pixel):

- Pearson correlation, mean bias (local − GEE), RMSE, and 5th/50th/95th percentiles of the difference
- masking disagreement, split by direction
- the same comparison under both `ddof=0` and `ddof=1`, so the reducer convention is settled empirically
- observation-support agreement — whether Earth Engine masked a pixel where the local count says it should not have
- **(S1 only, dormant until inputs exist)** behaviour where the window mean approaches zero, and the spatial distribution of non-finite or extreme CV values in both versions

Recomputation needs cached source bands for the contributing dates, so this builds a
small, scoped slice of the Phase 1 cache. It is the same cache and the same helper, so
Phase 1 later skips whatever this already wrote.""")

code("""CACHE_MANIFEST = tb.load_manifest(CACHE_MANIFEST_PATH, tb.CACHE_MANIFEST_COLUMNS)


def _source_fingerprint(snapshot):
    sizes, mtimes = [], []
    for path in snapshot.paths:
        stat = Path(path).stat()
        sizes.append(str(stat.st_size))
        mtimes.append(f'{stat.st_mtime:.0f}')
    return ';'.join(str(p) for p in snapshot.paths), ';'.join(mtimes), ';'.join(sizes)


def ensure_cached(snapshot, verbose=False):
    '''Extract the source band once per snapshot. Idempotent and resumable.'''
    global CACHE_MANIFEST
    out_path = cache_path_for(snapshot)
    paths_str, mtimes_str, sizes_str = _source_fingerprint(snapshot)

    if SKIP_CACHED and out_path.exists() and len(CACHE_MANIFEST):
        rows = CACHE_MANIFEST[CACHE_MANIFEST['prefix'].astype(str) == snapshot.prefix]
        if len(rows):
            row = rows.iloc[-1]
            unchanged = (str(row.get('source_mtimes')) == mtimes_str
                         and str(row.get('source_sizes')) == sizes_str
                         and str(row.get('status')) == 'cached')
            if unchanged:
                return out_path, 'skipped_cached'

    started = time.time()
    result = with_drive_retry(
        tb.extract_source_band, snapshot, out_path, readonly_dirs=READONLY_DIRS,
        max_block_bytes=MAX_BLOCK_BYTES, reader=staged_drive_read,
        context=f'caching {snapshot.prefix}', probe_path=snapshot.paths[0],
    )
    grid = result['grid']
    CACHE_MANIFEST = _upsert_cache_record({
        'sensor': snapshot.sensor,
        'prefix': snapshot.prefix,
        'start_date': snapshot.start_iso,
        'end_date': snapshot.end_iso,
        'schema_token': snapshot.schema_token,
        'source_paths': paths_str,
        'source_mtimes': mtimes_str,
        'source_sizes': sizes_str,
        'cache_path': str(out_path),
        'cache_bytes': result['cache_bytes'],
        'width': grid.width,
        'height': grid.height,
        'crs': grid.crs,
        'status': 'cached',
        'message': f'{result[\"n_source_files\"]} source file(s) in {time.time() - started:.1f}s',
        'recorded_at': tb.utc_now_iso(),
    })
    if verbose:
        print(f'    cached {snapshot.prefix} -> {tb.human_bytes(result[\"cache_bytes\"])}')
    return out_path, 'cached'


print('Cache manifest rows:', len(CACHE_MANIFEST))
print('Cache dir:', CACHE_DIR)""")

code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read_band(path, band_name, reader=staged_drive_read):
    '''Read a named band by DESCRIPTION, never by hardcoded index.'''
    def _read():
        with reader(path) as local:
            with rasterio.open(local) as src:
                index = tb.band_index_by_description(src, band_name, label=Path(path).name)
                return src.read(index).astype(np.float64), tb.grid_from_dataset(src)
    return with_drive_retry(_read, context=f'reading {band_name} from {Path(path).name}',
                            probe_path=path)


def _save_comparison_figures(sensor, band_name, date_iso, local_values, gee_values, out_dir):
    '''Hexbin scatter + difference histogram for one reference date.'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if local_values.size < 10:
        return []
    diff = local_values - gee_values
    written = []

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.hexbin(gee_values, local_values, gridsize=60, bins='log', cmap='viridis')
    lo = float(min(gee_values.min(), local_values.min()))
    hi = float(max(gee_values.max(), local_values.max()))
    ax.plot([lo, hi], [lo, hi], color='crimson', lw=1.0, ls='--', label='1:1')
    ax.set_xlabel(f'Earth Engine {band_name}')
    ax.set_ylabel(f'Local {band_name}')
    ax.set_title(f'{sensor} {band_name}\\n{date_iso} (water pixels, n={local_values.size:,})')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    path = out_dir / f'{sensor}_{band_name}_{date_iso}_hexbin.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(diff, bins=80, color='steelblue')
    ax.axvline(0.0, color='crimson', lw=1.0, ls='--')
    ax.set_xlabel(f'local - Earth Engine ({band_name})')
    ax.set_ylabel('pixels')
    ax.set_title(f'{sensor} {band_name} difference, {date_iso}')
    fig.tight_layout()
    path = out_dir / f'{sensor}_{band_name}_{date_iso}_diffhist.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return written


def compare_observation_support(local_count, reference_values, water, min_obs=TEMPORAL_MIN_OBS):
    '''Agreement between the local observation count and Earth Engine's masking.

    Earth Engine discards the count band, so its observation support can only be
    seen through where it masked the output (count < min_obs). This reports the
    2x2 agreement, which for S1 is the interesting one: swath coverage makes
    per-pixel counts vary far more across the AOI than for S2.
    '''
    water = np.asarray(water, dtype=bool)
    local_enough = (np.asarray(local_count) >= int(min_obs)) & water
    reference_valid = tb.valid_mask(reference_values) & water
    n_water = int(np.count_nonzero(water))
    return {
        'n_water': n_water,
        'n_both_supported': int(np.count_nonzero(local_enough & reference_valid)),
        'n_local_only_supported': int(np.count_nonzero(local_enough & ~reference_valid)),
        'n_reference_only_supported': int(np.count_nonzero(~local_enough & reference_valid)),
        'n_neither_supported': int(np.count_nonzero(~local_enough & ~reference_valid)),
        'local_count_p5': float(np.percentile(local_count[water], 5)) if n_water else np.nan,
        'local_count_p50': float(np.percentile(local_count[water], 50)) if n_water else np.nan,
        'local_count_p95': float(np.percentile(local_count[water], 95)) if n_water else np.nan,
    }


def compare_cv_tails(local_cv, reference_cv, water, threshold=CV_EXTREME_THRESHOLD,
                     out_png=None, title=''):
    '''Non-finite / extreme CV behaviour in BOTH versions, with spatial layout.

    Dormant for S1 until a reference export with vh_temporal_cv_w90 exists; it
    runs automatically the moment one does.
    '''
    water = np.asarray(water, dtype=bool)
    out = {}
    for label, values in (('local', local_cv), ('reference', reference_cv)):
        if values is None:
            out[f'{label}_n_nonfinite'] = None
            out[f'{label}_n_extreme'] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        arr = np.where(arr == tb.PREDICTOR_NODATA_VALUE, np.nan, arr)
        nonfinite = water & ~np.isfinite(arr)
        extreme = water & np.isfinite(arr) & (np.abs(arr) > float(threshold))
        out[f'{label}_n_nonfinite'] = int(np.count_nonzero(nonfinite))
        out[f'{label}_n_extreme'] = int(np.count_nonzero(extreme))
        rows, cols = np.nonzero(nonfinite | extreme)
        out[f'{label}_affected_row_mean'] = float(rows.mean()) if rows.size else np.nan
        out[f'{label}_affected_col_mean'] = float(cols.mean()) if cols.size else np.nan

    if out_png is not None and reference_cv is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        for ax, (label, values) in zip(axes, (('local', local_cv), ('reference', reference_cv))):
            arr = np.asarray(values, dtype=np.float64)
            arr = np.where(arr == tb.PREDICTOR_NODATA_VALUE, np.nan, arr)
            flag = water & (~np.isfinite(arr) | (np.abs(arr) > float(threshold)))
            ax.imshow(flag, cmap='inferno', interpolation='nearest')
            ax.set_title(f'{label}: non-finite or |cv|>{threshold:g}')
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
    return out


print('Validation harness defined (sensor-generic; S1 comparisons dormant until inputs exist).')""")

code("""def validate_sensor(sensor, snapshots, max_dates=PHASE0_MAX_VALIDATION_DATES):
    '''Recompute the temporal bands locally for every reference date and compare.

    A reference date is a snapshot that Earth Engine itself exported with the new
    schema token, so band values written by Earth Engine are available in the file.
    '''
    spec = tb.sensor_spec(sensor)
    same_sensor = [s for s in snapshots if s.sensor == sensor]
    references = [s for s in same_sensor if s.has_temporal_bands]

    result = {
        'sensor': sensor,
        'source_band': spec.source_band,
        'new_bands': list(spec.new_bands),
        'n_snapshots': len(same_sensor),
        'n_reference_snapshots': len(references),
        'status': tb.VALIDATION_UNVALIDATED,
        'per_date': [],
        'aggregate': {},
        'ddof_choice': None,
        'ddof_reason': '',
        'figures': [],
        'message': '',
    }
    if not references:
        result['message'] = (
            f'No {sensor} snapshot carries Earth Engine\\'s own temporal bands '
            f'({spec.temporal_schema_token}), so there is nothing in Drive to compare '
            'against and this notebook must not export one. Status is UNVALIDATED — '
            'not passing, not omitted.'
        )
        print(f'--- {sensor}: {tb.VALIDATION_UNVALIDATED} ---')
        print('   ', result['message'])
        return result

    usable = []
    for ref in references:
        contributors = tb.snapshots_in_window(ref, same_sensor)
        if len(contributors) >= spec.min_obs:
            usable.append((ref, contributors))
    usable = usable[-int(max_dates):] if max_dates else usable
    print(f'--- {sensor}: {len(references)} reference snapshot(s); validating {len(usable)} ---')
    if not usable:
        result['message'] = (
            f'{len(references)} {sensor} reference snapshot(s) exist but none has >= '
            f'{spec.min_obs} snapshots in its 90-day window locally, so no comparison '
            'is possible. Status stays UNVALIDATED.'
        )
        print('   ', result['message'])
        return result

    scratch = REPORT_DIR / 'phase0_recompute' / sensor
    tb.assert_not_in_readonly_dir(scratch, READONLY_DIRS)
    scratch.mkdir(parents=True, exist_ok=True)

    per_date_by_ddof = {0: [], 1: []}
    for index, (ref, contributors) in enumerate(usable, start=1):
        print(f'  [{index}/{len(usable)}] {ref.start_iso}: {len(contributors)} scene(s) in window')
        for snap in contributors:
            ensure_cached(snap)

        # Water proxy: a valid source-band pixel in the reference export. The
        # export applies the JRC water mask, so this IS the water footprint.
        source_values, ref_grid = _read_band(ref.paths[0], spec.source_band)
        water = tb.valid_mask(source_values)

        for ddof in (0, 1):
            out_paths = {
                band: scratch / f'{ref.prefix}_ddof{ddof}_{band}.tif'
                for band in list(spec.new_bands) + [spec.count_band]
            }
            outcome = tb.compute_temporal_bands(
                ref, contributors, cache_path_for, out_paths,
                readonly_dirs=READONLY_DIRS, ddof=ddof, min_obs=spec.min_obs,
                min_abs_mean=S1_CV_MIN_ABS_MEAN, max_block_bytes=MAX_BLOCK_BYTES,
            )
            if outcome['status'] != 'completed':
                print(f'      ddof={ddof}: {outcome[\"status\"]} — {outcome[\"message\"]}')
                continue

            with rasterio.open(out_paths[spec.count_band]) as src:
                local_count = src.read(1)

            for band in spec.new_bands:
                with rasterio.open(out_paths[band]) as src:
                    local_values = src.read(1).astype(np.float64)
                gee_values, gee_grid = _read_band(ref.paths[0], band)
                if gee_grid != ref_grid:
                    raise tb.GridMismatchError(
                        f'{ref.prefix}: band {band} grid differs from {spec.source_band}'
                    )
                stats = tb.compare_to_reference(local_values, gee_values, water=water)
                stats.update(sensor=sensor, band=band, date=ref.start_iso, ddof=ddof,
                             n_contributing_scenes=len(contributors))
                stats.update({f'support_{k}': v for k, v in
                              compare_observation_support(local_count, gee_values, water).items()})
                if band == spec.cv_band:
                    stats.update({f'cv_{k}': v for k, v in compare_cv_tails(
                        local_values, gee_values, water,
                        out_png=DIAGNOSTICS_DIR / f'{sensor}_{band}_{ref.start_iso}_tails.png',
                        title=f'{sensor} {band} tail behaviour, {ref.start_iso}',
                    ).items()})
                per_date_by_ddof[ddof].append(stats)

                if ddof == 1:
                    both = tb.valid_mask(local_values) & tb.valid_mask(gee_values) & water
                    result['figures'] += [str(p) for p in _save_comparison_figures(
                        sensor, band, ref.start_iso,
                        local_values[both], gee_values[both], DIAGNOSTICS_DIR,
                    )]

    agg0 = tb.aggregate_comparisons(per_date_by_ddof[0])
    agg1 = tb.aggregate_comparisons(per_date_by_ddof[1])
    ddof_choice, ddof_reason = tb.choose_ddof(agg0, agg1)

    result['per_date'] = per_date_by_ddof[0] + per_date_by_ddof[1]
    result['aggregate'] = {'ddof0': agg0, 'ddof1': agg1}
    result['ddof_choice'] = int(ddof_choice)
    result['ddof_reason'] = ddof_reason
    chosen = agg1 if ddof_choice == 1 else agg0
    result['chosen_aggregate'] = chosen
    result['status'] = tb.VALIDATION_PASS if chosen.get('n_dates', 0) else tb.VALIDATION_UNVALIDATED
    result['message'] = (
        f'{chosen.get(\"n_dates\", 0)} date(s) compared on {chosen.get(\"n_valid_both\", 0):,} '
        f'water pixels; {ddof_reason}'
    )
    return result


VALIDATION = {sensor: validate_sensor(sensor, SNAPSHOTS) for sensor in SENSORS}

PER_DATE_TABLE = pd.DataFrame(
    [row for v in VALIDATION.values() for row in v['per_date']]
)
if not PER_DATE_TABLE.empty:
    with_drive_retry(PER_DATE_TABLE.to_csv, REPORT_DIR / 'phase0_per_date_comparison.csv',
                     index=False, context='saving the per-date comparison',
                     probe_path=REPORT_DIR)
    print()
    print('=== Per-date comparison (local vs Earth Engine) ===')
    display(PER_DATE_TABLE[[
        'sensor', 'band', 'date', 'ddof', 'n_contributing_scenes', 'n_valid_both',
        'pearson_r', 'mean_bias', 'rmse', 'diff_p5', 'diff_p50', 'diff_p95',
        'frac_local_only', 'frac_reference_only',
    ]])
else:
    print()
    print('No per-date comparison rows: no sensor has Earth Engine reference bands in Drive.')""")

# ---------------------------------------------------------------------------
md("""### 3d. Phase 0 verdict (blocking gate)

Prints a clear verdict block per sensor with the numbers behind it, then writes
`phase0_report.json`. The later phases refuse to run until this file exists **and was
written by an earlier run**.

S1 will read `UNVALIDATED` on a first run. That status is carried through to the run
manifest and the methods summary rather than dropped.""")

code("""def _fmt(value, spec='.6g'):
    if value is None:
        return 'n/a'
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return 'n/a'
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


print('=' * 78)
print('PHASE 0 VERDICT')
print('=' * 78)

verdict_payload = {}
for sensor in SENSORS:
    result = VALIDATION[sensor]
    spec = tb.sensor_spec(sensor)
    picked = [s for s in SNAPSHOTS if s.sensor == sensor]
    to_backfill = [s for s in picked if not s.has_temporal_bands]

    print()
    print(f'--- {sensor}  [{result[\"status\"]}] ---')
    print(f'  snapshots discovered      : {len(picked)}')
    print(f'  already carry GEE bands   : {result[\"n_reference_snapshots\"]}')
    print(f'  to backfill               : {len(to_backfill)}')
    print(f'  source band               : {spec.source_band}')
    print(f'  new band(s)               : {\", \".join(spec.new_bands)}')
    loss = COVERAGE_LOSS.get(sensor)
    if loss:
        print(f'  dates skipped low coverage: {int(loss[\"skipped_low_coverage\"])} '
              f'({_fmt(loss[\"frac_dates_missing_vs_gated\"], \".1%\")} of gated dates)')
        if loss['planned_source_dates']:
            seen = loss['exported_or_queued'] / loss['planned_source_dates']
            print(f'  local stack sees           : {_fmt(seen, \".1%\")} of the source dates '
                  f'Earth Engine reduced over')

    if result['status'] == tb.VALIDATION_UNVALIDATED:
        print(f'  VERDICT: {tb.VALIDATION_UNVALIDATED} — {result[\"message\"]}')
    else:
        chosen = result['chosen_aggregate']
        print(f'  ddof settled empirically  : ddof={result[\"ddof_choice\"]} ({result[\"ddof_reason\"]})')
        print(f'  dates compared            : {chosen.get(\"n_dates\", 0)}')
        print(f'  water pixels compared     : {chosen.get(\"n_valid_both\", 0):,}')
        print(f'  Pearson r                 : {_fmt(chosen.get(\"pearson_r\"), \".5f\")}')
        print(f'  mean bias (local - GEE)   : {_fmt(chosen.get(\"mean_bias\"), \".6f\")}')
        print(f'  RMSE                      : {_fmt(chosen.get(\"rmse\"), \".6f\")}')
        print(f'  diff p5 / p50 / p95       : {_fmt(chosen.get(\"diff_p5\"), \".6f\")} / '
              f'{_fmt(chosen.get(\"diff_p50\"), \".6f\")} / {_fmt(chosen.get(\"diff_p95\"), \".6f\")}')
        print(f'  valid locally, NoData GEE : {_fmt(chosen.get(\"frac_local_only\"), \".3%\")}')
        print(f'  valid GEE, NoData locally : {_fmt(chosen.get(\"frac_reference_only\"), \".3%\")}')
        print('  ddof comparison:')
        for label, agg in (('ddof=0', result['aggregate']['ddof0']),
                           ('ddof=1', result['aggregate']['ddof1'])):
            print(f'      {label}: RMSE {_fmt(agg.get(\"rmse\"), \".6f\")}, '
                  f'bias {_fmt(agg.get(\"mean_bias\"), \".6f\")}, '
                  f'r {_fmt(agg.get(\"pearson_r\"), \".5f\")}')
        print(f'  VERDICT: {result[\"status\"]} — {result[\"message\"]}')

    verdict_payload[sensor] = {
        'status': result['status'],
        'n_snapshots': len(picked),
        'n_reference_snapshots': result['n_reference_snapshots'],
        'n_to_backfill': len(to_backfill),
        'ddof_choice': result['ddof_choice'],
        'ddof_reason': result['ddof_reason'],
        'aggregate': result.get('aggregate', {}),
        'chosen_aggregate': result.get('chosen_aggregate', {}),
        'coverage_loss': loss,
        'message': result['message'],
        'figures': result['figures'],
    }

# The ddof used for the bulk run: taken from whichever sensor was actually
# validated. If none was, the documented default stands and says so.
fitted = [v['ddof_choice'] for v in verdict_payload.values() if v['ddof_choice'] is not None]
if fitted and len(set(fitted)) == 1:
    SETTLED_DDOF = int(fitted[0])
    ddof_note = f'fitted empirically against Earth Engine output (ddof={SETTLED_DDOF})'
elif fitted:
    SETTLED_DDOF = int(max(set(fitted), key=fitted.count))
    ddof_note = f'sensors disagreed {fitted}; using the majority fit ddof={SETTLED_DDOF}'
else:
    SETTLED_DDOF = int(tb.TEMPORAL_DDOF)
    ddof_note = (f'no sensor could be validated; keeping the documented default '
                 f'ddof={SETTLED_DDOF} (ee.Reducer.stdDev is a sample std dev)')

PHASE0_REPORT = {
    'session_id': SESSION_ID,
    'recorded_at': tb.utc_now_iso(),
    'export_dir': str(GEE_EXPORT_DIR),
    'sensors': SENSORS,
    'settled_ddof': SETTLED_DDOF,
    'settled_ddof_note': ddof_note,
    'min_obs': int(TEMPORAL_MIN_OBS),
    'lookback_days': int(TEMPORAL_LOOKBACK_DAYS),
    's1_cv_min_abs_mean': float(S1_CV_MIN_ABS_MEAN),
    'n_files_discovered': int(len(inventory)),
    'n_drive_duplicates_excluded': int(inventory['is_drive_duplicate'].sum()),
    'n_date_collisions': len(DATE_COLLISIONS),
    'multi_file_resolution': (
        MULTI_FILE_VERDICTS.groupby(['sensor', 'kind']).size().reset_index(name='n_prefixes')
        .to_dict('records') if not MULTI_FILE_VERDICTS.empty else []
    ),
    'manifest_breakdown': (
        MANIFEST_BREAKDOWN.to_dict('records') if not MANIFEST_BREAKDOWN.empty else []
    ),
    'verdicts': verdict_payload,
}
tb.assert_not_in_readonly_dir(PHASE0_REPORT_PATH, READONLY_DIRS)
with_drive_retry(PHASE0_REPORT_PATH.write_text,
                 json.dumps(PHASE0_REPORT, indent=2, default=str),
                 context='saving the Phase 0 report', probe_path=REPORT_DIR)

print()
print('=' * 78)
print('ddof for the bulk run:', SETTLED_DDOF, '-', ddof_note)
print('Wrote Phase 0 report:', PHASE0_REPORT_PATH)
print()
print('PHASE 0 IS A BLOCKING GATE. Read the verdict above, then START A FRESH RUN')
print('(restart the runtime, or re-run the setup cell in section 1) before running')
print('Phase 1 and beyond. The bulk cells will refuse to run in this same session.')
print('=' * 78)""")

# ---------------------------------------------------------------------------
md("""## 4. Phase 1 — source-band cache

Naively re-reading each snapshot once per overlapping window would move hundreds of GB
through the Drive mount: each snapshot participates in roughly a full window's worth of
targets. Instead the source band is extracted **once per snapshot** into a compressed,
tiled single-band GeoTIFF, preserving CRS, transform, shape and the `-9999` NoData tag
exactly. Tile shards are mosaicked onto their union grid so every cached snapshot of a
sensor lands on one identical grid.

The cache lives in its own Drive folder so it survives session resets, and is keyed by
sensor. Re-running skips anything already cached and verified against source size and
mtime.""")

code("""PHASE0 = require_phase0('Phase 1 (source-band cache)')
TEMPORAL_DDOF = int(PHASE0.get('settled_ddof', TEMPORAL_DDOF))
print('Phase 0 report from', PHASE0.get('recorded_at'))
print('Using ddof =', TEMPORAL_DDOF, '-', PHASE0.get('settled_ddof_note'))
print(validation_status_note(PHASE0))

# Rebuild the working snapshot set (safe to re-run; section 3 may not have run
# in this session).
if 'SNAPSHOTS' not in globals():
    _inv = tb.discover_predictor_files(GEE_EXPORT_DIR, sensors=SENSORS)
    SNAPSHOTS, DATE_COLLISIONS = tb.select_snapshots_by_date(tb.group_snapshot_files(_inv))
    print(f'Re-discovered {len(SNAPSHOTS)} snapshot(s).')

CACHE_MANIFEST = tb.load_manifest(CACHE_MANIFEST_PATH, tb.CACHE_MANIFEST_COLUMNS)

# --- Headroom check BEFORE writing anything. ---
already = CACHE_MANIFEST[CACHE_MANIFEST['status'].astype(str) == 'cached'] if len(CACHE_MANIFEST) else CACHE_MANIFEST
realised_bytes = float(pd.to_numeric(already.get('cache_bytes'), errors='coerce').fillna(0).sum()) if len(already) else 0.0
n_done = len(already)
n_total = len([s for s in SNAPSHOTS if s.sensor in SENSORS])
mean_bytes = (realised_bytes / n_done) if n_done else None
free_bytes = with_drive_retry(tb.disk_headroom_bytes, CACHE_DIR,
                              context='checking Drive headroom', probe_path=CACHE_DIR)

print()
print(f'Snapshots to cache : {n_total} ({n_done} already cached)')
print(f'Realised cache size: {tb.human_bytes(realised_bytes)} over {n_done} entr(y/ies)')
if mean_bytes:
    projected = mean_bytes * n_total
    print(f'Mean per snapshot  : {tb.human_bytes(mean_bytes)}')
    print(f'Projected total    : {tb.human_bytes(projected)}')
    print(f'Drive headroom     : {tb.human_bytes(free_bytes)}')
    if projected > free_bytes * 0.9:
        raise RuntimeError(
            f'Projected cache size {tb.human_bytes(projected)} would exceed 90% of the '
            f'available headroom {tb.human_bytes(free_bytes)}. Free space or narrow SENSORS '
            'before continuing; nothing has been written.'
        )
else:
    print(f'Drive headroom     : {tb.human_bytes(free_bytes)} '
          '(no realised sizes yet; re-run this cell after the first few writes '
          'for a projection)')""")

code("""targets_to_cache = [s for s in SNAPSHOTS if s.sensor in SENSORS]
counts = {'cached': 0, 'skipped_cached': 0, 'failed': 0}
first_write_checked = False

for index, snap in enumerate(targets_to_cache, start=1):
    try:
        path, outcome = ensure_cached(snap)
        counts[outcome] += 1
        if outcome == 'cached':
            print(f'[{index}/{len(targets_to_cache)}] {snap.sensor} {snap.start_iso} '
                  f'-> {tb.human_bytes(path.stat().st_size)}')
            if not first_write_checked:
                first_write_checked = True
                per = path.stat().st_size
                projected = per * len(targets_to_cache)
                free = tb.disk_headroom_bytes(CACHE_DIR)
                print(f'    projection from this write: {tb.human_bytes(projected)} total '
                      f'vs {tb.human_bytes(free)} free')
                if projected > free * 0.9:
                    raise RuntimeError(
                        f'Projected cache size {tb.human_bytes(projected)} would exceed 90% of '
                        f'the {tb.human_bytes(free)} available. Stopping after one write.'
                    )
        elif index % 50 == 0:
            print(f'[{index}/{len(targets_to_cache)}] ... {counts[\"skipped_cached\"]} already cached')
    except Exception as exc:
        counts['failed'] += 1
        print(f'[{index}/{len(targets_to_cache)}] FAILED {snap.prefix}: {type(exc).__name__}: {exc}')
        CACHE_MANIFEST = _upsert_cache_record({
            'sensor': snap.sensor, 'prefix': snap.prefix,
            'start_date': snap.start_iso, 'end_date': snap.end_iso,
            'schema_token': snap.schema_token, 'source_paths': ';'.join(str(p) for p in snap.paths),
            'source_mtimes': None, 'source_sizes': None, 'cache_path': str(cache_path_for(snap)),
            'cache_bytes': None, 'width': None, 'height': None, 'crs': None,
            'status': 'failed', 'message': f'{type(exc).__name__}: {exc}',
            'recorded_at': tb.utc_now_iso(),
        })

print()
print('Phase 1 complete:', counts)
print('Cache manifest:', CACHE_MANIFEST_PATH)
_cached = tb.load_manifest(CACHE_MANIFEST_PATH, tb.CACHE_MANIFEST_COLUMNS)
_cached = _cached[_cached['status'].astype(str) == 'cached']
print('Realised cache size:', tb.human_bytes(
    pd.to_numeric(_cached['cache_bytes'], errors='coerce').fillna(0).sum()))""")

# ---------------------------------------------------------------------------
md("""## 5. Phases 2 and 3 — rolling statistics, sidecars and VRTs

For each target snapshot, within its own sensor's collection only:

1. select cached source bands for every snapshot whose date falls in the window
   (including the target's own date);
2. skip and record the target if fewer than 3 snapshots fall in the window;
3. process in windowed blocks aligned to the source's internal tiling — never the whole stack;
4. per block, mask `-9999` and non-finite values, count valid observations per pixel, then
   compute the standard deviation (and, for S1, the mean over the **same** valid set, then
   `vh_temporal_cv_w90 = std / abs(mean)`);
5. apply the ddof settled in Phase 0, then mask pixels with count < 3;
6. assert grid identity across every contributing file before computing, aborting that
   date with a clear message on mismatch rather than silently resampling.

Outputs are **sidecars plus a `.vrt`**, in a separate Drive folder per sensor:

- `{source_prefix}_{band_name}.tif` — single band, float32, NoData `-9999`, band description
  set to exactly the schema band name, cloud-optimised, same grid as the source.
- `{source_prefix}_{ndvi|vh}_temporal_count_w90.tif` — the per-pixel observation count.
  Earth Engine discards this; keeping it costs almost nothing and supports the obvious
  robustness check on whether results depend on observation density. This matters more for
  S1, where swath coverage makes the count strongly spatially structured.
- `{new_prefix}.vrt` — the old-schema source plus its sidecars as a virtual dataset with band
  descriptions in exactly `S2_PREDICTORS` / `S1_PREDICTORS` order, verified on write, so
  `rasterio.open(vrt).descriptions` satisfies the classifier's band-order validation unchanged.

Sidecars are named after the **source** prefix (provenance is visible on disk); VRTs are
named under the **new** prefix, because the classifier's `BATCH_S1_REQUIRE_SCHEMA_TOKEN` is
`s1_scc_temporal_v1`.""")

code("""PHASE0 = require_phase0('Phase 2/3 (rolling statistics and outputs)')
TEMPORAL_DDOF = int(PHASE0.get('settled_ddof', TEMPORAL_DDOF))
VALIDATION_NOTE = validation_status_note(PHASE0)
print('Using ddof =', TEMPORAL_DDOF, '|', VALIDATION_NOTE)

if 'SNAPSHOTS' not in globals():
    _inv = tb.discover_predictor_files(GEE_EXPORT_DIR, sensors=SENSORS)
    SNAPSHOTS, DATE_COLLISIONS = tb.select_snapshots_by_date(tb.group_snapshot_files(_inv))

RUN_MANIFEST = tb.load_manifest(RUN_MANIFEST_PATH, tb.RUN_MANIFEST_COLUMNS)
CACHE_MANIFEST = tb.load_manifest(CACHE_MANIFEST_PATH, tb.CACHE_MANIFEST_COLUMNS)


def _record(snapshot, status, n_scenes=None, valid_pixels=None, n_nonfinite_cv=None, message=''):
    global RUN_MANIFEST
    RUN_MANIFEST = _upsert_run_record({
        'sensor': snapshot.sensor,
        'prefix': snapshot.prefix,
        'target_prefix': snapshot.target_prefix(),
        'start_date': snapshot.start_iso,
        'end_date': snapshot.end_iso,
        'n_contributing_scenes': n_scenes,
        'valid_pixels': valid_pixels,
        'n_nonfinite_cv': n_nonfinite_cv,
        'status': status,
        # The validation status travels with every row so a downstream reader
        # never has to assume the S2 result covered S1 too.
        'message': f'{message} | ddof={TEMPORAL_DDOF} | {VALIDATION_NOTE}'.strip(' |'),
        'recorded_at': tb.utc_now_iso(),
    })


def _already_done(snapshot):
    if not SKIP_COMPLETED or not len(RUN_MANIFEST):
        return False
    rows = RUN_MANIFEST[RUN_MANIFEST['prefix'].astype(str) == snapshot.prefix]
    if not len(rows) or str(rows.iloc[-1]['status']) not in {'completed', 'skipped_min_obs'}:
        return False
    if str(rows.iloc[-1]['status']) == 'skipped_min_obs':
        return True
    outputs = list(sidecar_paths_for(snapshot).values())
    if WRITE_VRT:
        outputs.append(vrt_path_for(snapshot))
    return all(Path(p).exists() for p in outputs)


# Targets are the snapshots that do NOT already carry Earth Engine's own bands.
TARGETS = [s for s in SNAPSHOTS if s.sensor in SENSORS and not s.has_temporal_bands]
if MAX_TARGETS_PER_RUN:
    TARGETS = TARGETS[:int(MAX_TARGETS_PER_RUN)]
print(f'Targets to backfill: {len(TARGETS)}')
print({sensor: sum(1 for s in TARGETS if s.sensor == sensor) for sensor in SENSORS})""")

code("""outcomes = {'completed': 0, 'skipped_existing': 0, 'skipped_min_obs': 0, 'failed': 0}

for index, target in enumerate(TARGETS, start=1):
    label = f'[{index}/{len(TARGETS)}] {target.sensor} {target.start_iso}'
    if _already_done(target):
        outcomes['skipped_existing'] += 1
        if index % 50 == 0:
            print(f'{label}: already complete ({outcomes[\"skipped_existing\"]} skipped so far)')
        continue

    try:
        contributors = tb.snapshots_in_window(target, SNAPSHOTS)
        spec = tb.sensor_spec(target.sensor)

        if len(contributors) < spec.min_obs:
            outcomes['skipped_min_obs'] += 1
            message = (f'only {len(contributors)} snapshot(s) in the '
                       f'{spec.lookback_days}-day window; need >= {spec.min_obs}')
            print(f'{label}: SKIP — {message}')
            _record(target, 'skipped_min_obs', n_scenes=len(contributors), message=message)
            continue

        for snap in contributors:
            ensure_cached(snap)

        out_paths = sidecar_paths_for(target)
        # Safe to retry: partial .tmp outputs are cleaned up on any failure.
        result = with_drive_retry(
            tb.compute_temporal_bands,
            target, contributors, cache_path_for, out_paths,
            readonly_dirs=READONLY_DIRS, ddof=TEMPORAL_DDOF, min_obs=spec.min_obs,
            min_abs_mean=S1_CV_MIN_ABS_MEAN, max_block_bytes=MAX_BLOCK_BYTES,
            context=f'computing {target.prefix}', probe_path=cache_path_for(target),
        )
        if result['status'] != 'completed':
            outcomes['skipped_min_obs'] += 1
            _record(target, result['status'], n_scenes=result['n_contributing_scenes'],
                    message=result['message'])
            continue

        if WRITE_VRT:
            with_drive_retry(
                tb.write_snapshot_vrt,
                target, {band: out_paths[band] for band in spec.new_bands},
                vrt_path_for(target), readonly_dirs=READONLY_DIRS,
                context=f'writing VRT for {target.prefix}', probe_path=target.paths[0],
            )

        outcomes['completed'] += 1
        _record(target, 'completed', n_scenes=result['n_contributing_scenes'],
                valid_pixels=result['valid_pixels'], n_nonfinite_cv=result['n_nonfinite_cv'],
                message=result['message'])
        print(f'{label}: {result[\"n_contributing_scenes\"]} scene(s), '
              f'{result[\"valid_pixels\"]:,} valid px'
              + (f', {result[\"n_nonfinite_cv\"]:,} near-zero-mean px masked'
                 if result['n_nonfinite_cv'] else ''))

    except tb.GridMismatchError as exc:
        outcomes['failed'] += 1
        print(f'{label}: GRID MISMATCH — aborting this date, not resampling.')
        print(f'    {exc}')
        _record(target, 'failed_grid_mismatch', message=str(exc).replace('\\n', ' '))
    except Exception as exc:
        outcomes['failed'] += 1
        print(f'{label}: FAILED — {type(exc).__name__}: {exc}')
        _record(target, 'failed', message=f'{type(exc).__name__}: {exc}')

print()
print('Phases 2 and 3 complete:', outcomes)
print('Run manifest:', RUN_MANIFEST_PATH)
print('Sidecars    :', SIDECAR_DIR)
print('VRTs        :', VRT_DIR)""")

# ---------------------------------------------------------------------------
md("""## 6. Optional — materialise real full-schema GeoTIFFs

`REWRITE_IN_PLACE` is `False` per sensor by default. When enabled it materialises real
21-band (S2) / 5-band (S1) GeoTIFFs instead of relying on the VRT. Each write goes to a
temp path, is verified for band count, band order, grid and NoData, and only then moved
into place.

It **never deletes or overwrites a source file** — the read-only guard refuses any
target inside the export archive, and outputs land in `Winam_Temporal_Backfill/rewritten/`.

S1 rewriting is roughly an order of magnitude cheaper than S2 (~175 MB vs ~1.29 GB per
file), which is why the flag is per sensor.""")

code("""rewrite_sensors = [s for s in SENSORS if REWRITE_IN_PLACE.get(s)]
if not rewrite_sensors:
    print('REWRITE_IN_PLACE is False for every sensor; nothing to do.')
    print('The .vrt files in', VRT_DIR, 'already present the full band stack.')
else:
    PHASE0 = require_phase0('Rewrite in place')
    # Safe to run without section 5 having run in this session.
    if 'SNAPSHOTS' not in globals():
        _inv = tb.discover_predictor_files(GEE_EXPORT_DIR, sensors=SENSORS)
        SNAPSHOTS, DATE_COLLISIONS = tb.select_snapshots_by_date(tb.group_snapshot_files(_inv))
    rewrite_targets = [
        s for s in SNAPSHOTS if s.sensor in rewrite_sensors and not s.has_temporal_bands
    ]
    estimate = {'S2': 1.29 * 1024 ** 3, 'S1': 175 * 1024 ** 2}
    projected = sum(estimate.get(t.sensor, 0) for t in rewrite_targets)
    free = tb.disk_headroom_bytes(REWRITE_DIR)
    print(f'Rewriting {len(rewrite_targets)} snapshot(s) for {rewrite_sensors}')
    print(f'Rough projection: {tb.human_bytes(projected)} vs {tb.human_bytes(free)} free')
    if projected > free * 0.9:
        raise RuntimeError(
            f'Rewriting would need about {tb.human_bytes(projected)}, more than 90% of the '
            f'{tb.human_bytes(free)} available. Nothing written.'
        )

    # Dates that section 5 legitimately skipped (too few scenes in the window)
    # have no sidecars by design and are not rewrite failures.
    _run = tb.load_manifest(RUN_MANIFEST_PATH, tb.RUN_MANIFEST_COLUMNS)
    skipped_by_design = set(
        _run.loc[_run['status'].astype(str) == 'skipped_min_obs', 'prefix'].astype(str)
    ) if len(_run) else set()

    rewritten = failed = skipped = no_bands = 0
    for index, target in enumerate(rewrite_targets, start=1):
        out_path = rewrite_path_for(target)
        if SKIP_COMPLETED and out_path.exists():
            skipped += 1
            continue
        if target.prefix in skipped_by_design:
            no_bands += 1
            continue
        spec = tb.sensor_spec(target.sensor)
        sidecars = {band: sidecar_path_for(target, band) for band in spec.new_bands}
        if not all(p.exists() for p in sidecars.values()):
            print(f'[{index}/{len(rewrite_targets)}] {target.prefix}: sidecars missing; '
                  'run section 5 first.')
            failed += 1
            continue
        try:
            with_drive_retry(
                tb.rewrite_snapshot_geotiff,
                target, sidecars, out_path, readonly_dirs=READONLY_DIRS,
                reader=staged_drive_read, max_block_bytes=MAX_BLOCK_BYTES,
                context=f'rewriting {target.prefix}', probe_path=target.paths[0],
            )
            rewritten += 1
            print(f'[{index}/{len(rewrite_targets)}] wrote and verified {out_path.name} '
                  f'({tb.human_bytes(out_path.stat().st_size)})')
        except Exception as exc:
            failed += 1
            print(f'[{index}/{len(rewrite_targets)}] FAILED {target.prefix}: '
                  f'{type(exc).__name__}: {exc}')

    print()
    print(f'Rewrite complete: {rewritten} written, {skipped} already present, '
          f'{no_bands} skipped (too few scenes in window; no temporal bands exist), '
          f'{failed} failed')
    # Sanity: the source archive is untouched.
    print('Source archive untouched:', GEE_EXPORT_DIR)""")

# ---------------------------------------------------------------------------
md("""## 7. Run summary and methods paragraph

Summarises the run manifest and prints a short markdown block that can be pasted
straight into a methods section. It states S1's validation status honestly rather than
implying the S2 result covers both sensors.""")

code("""RUN_MANIFEST = tb.load_manifest(RUN_MANIFEST_PATH, tb.RUN_MANIFEST_COLUMNS)
PHASE0 = load_phase0_report()

if RUN_MANIFEST.empty:
    print('Run manifest is empty; nothing to summarise yet.')
else:
    print('=== Run manifest: rows by sensor and status ===')
    display(RUN_MANIFEST.groupby(['sensor', 'status']).size().rename('n_rows').reset_index())

    done = RUN_MANIFEST[RUN_MANIFEST['status'].astype(str) == 'completed'].copy()
    if len(done):
        done['n_contributing_scenes'] = pd.to_numeric(done['n_contributing_scenes'], errors='coerce')
        done['valid_pixels'] = pd.to_numeric(done['valid_pixels'], errors='coerce')
        done['n_nonfinite_cv'] = pd.to_numeric(done['n_nonfinite_cv'], errors='coerce').fillna(0)
        print()
        print('=== Completed backfills ===')
        display(done.groupby('sensor').agg(
            n_dates=('prefix', 'size'),
            scenes_per_window_median=('n_contributing_scenes', 'median'),
            scenes_per_window_min=('n_contributing_scenes', 'min'),
            scenes_per_window_max=('n_contributing_scenes', 'max'),
            valid_pixels_median=('valid_pixels', 'median'),
            near_zero_mean_pixels_total=('n_nonfinite_cv', 'sum'),
        ).reset_index())


def methods_summary_markdown(report, run_manifest):
    '''Phase 0 validation numbers per sensor, ready to paste into a methods section.'''
    lines = [
        '# Temporal-band backfill — Phase 0 validation summary',
        '',
        f'Generated {tb.utc_now_iso()} from `{PHASE0_REPORT_PATH.name}`.',
        '',
        ('The 90-day temporal-persistence bands were reconstructed locally from the '
         'source band already present in every exported snapshot (`NDVI` for Sentinel-2, '
         '`VH_corrected` for Sentinel-1), with no Earth Engine calls. The lookback window '
         'reproduces the export code exactly: `[end_date - 90 days, end_date)`, which '
         "includes the snapshot's own acquisition date. Pixels with fewer than "
         f'{report.get(\"min_obs\", 3)} valid observations are masked, and '
         f'`ddof={report.get(\"settled_ddof\")}` was used '
         f'({report.get(\"settled_ddof_note\", \"\")}).'),
        '',
    ]
    for sensor, verdict in sorted(report.get('verdicts', {}).items()):
        spec = tb.sensor_spec(sensor)
        lines.append(f'## {sensor} — {verdict[\"status\"]}')
        lines.append('')
        lines.append(f'- Bands reconstructed: {\", \".join(spec.new_bands)} (from `{spec.source_band}`)')
        lines.append(f'- Snapshots discovered: {verdict[\"n_snapshots\"]}; backfilled: {verdict[\"n_to_backfill\"]}')
        agg = verdict.get('chosen_aggregate') or {}
        if verdict['status'] == tb.VALIDATION_UNVALIDATED:
            validated_elsewhere = sorted(
                other for other, v in report.get('verdicts', {}).items()
                if other != sensor and v.get('status') == tb.VALIDATION_PASS
            )
            caveat = (
                (f'The {\", \".join(validated_elsewhere)} result in this document does **not** '
                 f'transfer to {sensor}: the sensors differ in revisit interval, swath coverage '
                 'and source-band distribution, so agreement measured for one says nothing '
                 'about the other. ')
                if validated_elsewhere else
                ('No sensor in this run could be validated against Earth Engine output. ')
            )
            lines += [
                '',
                (f'**No validation was possible for {sensor}.** {verdict[\"message\"]} '
                 f'{caveat}{sensor} outputs are reported as `UNVALIDATED`.'),
                '',
            ]
        else:
            lines += [
                f'- Reference dates compared: {agg.get(\"n_dates\", 0)}',
                f'- Water pixels compared: {agg.get(\"n_valid_both\", 0):,}',
                f'- Pearson r: {agg.get(\"pearson_r\", float(\"nan\")):.5f}',
                f'- Mean bias (local - Earth Engine): {agg.get(\"mean_bias\", float(\"nan\")):.6f}',
                f'- RMSE: {agg.get(\"rmse\", float(\"nan\")):.6f}',
                (f'- Difference percentiles (5/50/95): {agg.get(\"diff_p5\", float(\"nan\")):.6f} / '
                 f'{agg.get(\"diff_p50\", float(\"nan\")):.6f} / {agg.get(\"diff_p95\", float(\"nan\")):.6f}'),
                (f'- Masking disagreement: {agg.get(\"frac_local_only\", float(\"nan\")):.3%} valid '
                 f'locally but NoData in Earth Engine; '
                 f'{agg.get(\"frac_reference_only\", float(\"nan\")):.3%} the other way'),
                f'- Reducer convention: {verdict.get(\"ddof_reason\", \"\")}',
                '',
            ]
        loss = verdict.get('coverage_loss')
        if loss and loss.get('planned_source_dates'):
            seen = loss['exported_or_queued'] / loss['planned_source_dates']
            lines.append(
                f'- Coverage caveat: Earth Engine reduced over {int(loss[\"planned_source_dates\"])} '
                f'source date(s); the local stack sees {seen:.1%} of them '
                f'({int(loss[\"skipped_low_coverage\"])} were skipped by the export coverage gate).'
            )
            lines.append('')
    lines += [
        '## Known issues preserved deliberately',
        '',
        ('1. `vh_temporal_cv_w90` divides by `abs(mean)` of a dB quantity, so it is not the '
         'scale-free coefficient of variation its source comment claims. Reproduced as written '
         'for schema consistency.'),
        ('2. The Sentinel-1 stack mixes ascending and descending orbit geometries, so its '
         'temporal standard deviation conflates real change with acquisition geometry.'),
        ('3. Sentinel-1 swath coverage is uneven across the AOI, so per-pixel observation counts '
         'and the minimum-observation mask are strongly spatially structured. The observation-'
         'count band is retained as an output so this can be checked.'),
        '',
        'See `docs/temporal_backfill_integration.md` for detail and for classifier integration.',
        '',
    ]
    return '\\n'.join(lines)


SUMMARY_MD = methods_summary_markdown(PHASE0, RUN_MANIFEST)
summary_path = REPORT_DIR / 'phase0_validation_summary.md'
tb.assert_not_in_readonly_dir(summary_path, READONLY_DIRS)
with_drive_retry(summary_path.write_text, SUMMARY_MD,
                 context='saving the validation summary', probe_path=REPORT_DIR)
if UPDATE_REPO_VALIDATION_SUMMARY:
    try:
        repo_copy = REPO_ROOT / 'docs' / 'temporal_backfill_validation_summary.md'
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        repo_copy.write_text(SUMMARY_MD)
        print('Also overwrote the committed baseline:', repo_copy)
    except OSError as exc:
        print('Could not write the repo copy of the summary:', exc)
else:
    print('UPDATE_REPO_VALIDATION_SUMMARY is False; the committed baseline at '
          'docs/temporal_backfill_validation_summary.md was left unchanged.')

print()
print(SUMMARY_MD)
print()
print('Saved summary:', summary_path)""")

# ---------------------------------------------------------------------------
md("""## 8. Notes and caveats

### Known issues preserved faithfully, not fixed

Earth Engine's behaviour is reproduced exactly even where it is questionable; consistency
with the existing schema matters more than correctness at this stage.

1. **`vh_temporal_cv_w90` is not scale-free.** The comment in `add_s1_temporal_stability`
   justifies dividing by `abs(mean)` as producing "a positive, scale-free coefficient of
   variation". That reasoning does not hold on a dB scale: a multiplicative change in linear
   backscatter is an *additive* shift in dB, which leaves the standard deviation unchanged
   but moves the mean, so the ratio is not invariant. Reproduced as written.
2. **The S1 stack mixes orbit geometries.** `get_s1_source_collection` filters IW mode and
   VV+VH polarisation but not orbit direction or relative orbit number, so a 90-day window
   mixes ascending and descending passes. The incidence-angle correction in
   `add_s1_predictors_exact_scc` normalises to 38° but does not address look direction.
   Temporal standard deviation therefore conflates real change with acquisition geometry.
   Present in the Earth Engine version too.
3. **S1 spatial coverage is uneven.** Per-pixel observation counts vary far more across the
   AOI for S1 than S2 because of swath coverage, so the min-obs-3 mask has strong spatial
   structure. This is the main reason the count band is retained.

### The near-zero-mean policy for `vh_temporal_cv_w90`

Earth Engine's behaviour at `abs(mean) -> 0` cannot currently be observed, because no S1
snapshot has been exported with the temporal bands. The chosen policy is to **mask** pixels
whose `abs(mean)` falls below `S1_CV_MIN_ABS_MEAN` rather than emit ±Inf. The classifier's
`_predictor_valid_mask` already drops any pixel with a non-finite value in any band, so an
infinity would silently invalidate the whole pixel across all five S1 bands; masking makes
that explicit and countable. Affected pixel counts are recorded per date in the run manifest
as `n_nonfinite_cv`. It is a single named constant in
`winam_diagnostics/temporal_backfill.py`, changeable in one place if a reference export ever
becomes available.

### Departure from Earth Engine, restated

Earth Engine reduced over every scene passing the source-collection filters. This
reconstruction sees only snapshots that passed the export coverage gate and reached Drive,
and one median-composited observation per acquisition date rather than each granule. Section
3b quantifies the first effect; section 3c measures the combined effect against real Earth
Engine output for whichever sensors have reference exports.

### What this notebook never does

- It never modifies, moves or deletes anything in `GEE_Exports_validated_snapshots`.
- It never imports Earth Engine, authenticates, or calls the Earth Engine API.
- It never edits `Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb`. The integration is
  described in `docs/temporal_backfill_integration.md` instead.

### Tests

`tests/test_temporal_backfill.py` covers window membership (including the boundary at
exactly −90 days and inclusion of the target's own date), NoData handling, min-obs-3 masking
at counts of 2/3/4, ddof against a hand-computed fixture, the S1 CV rules and near-zero-mean
policy, grid-mismatch detection, Drive collision duplicates and tiled shards, and
cross-sensor contamination. Run with `pytest tests/test_temporal_backfill.py`. No Drive, no
network.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "machine_shape": "hm"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open('Backfill_Temporal_Bands_Local.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('Wrote Backfill_Temporal_Bands_Local.ipynb with', len(cells), 'cells')
