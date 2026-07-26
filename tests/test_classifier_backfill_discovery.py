"""Tests for the classifier's discovery and staging of backfilled temporal bands.

`Backfill_Temporal_Bands_Local.ipynb` writes reconstructed temporal bands as
sidecar GeoTIFFs plus a `.vrt` that presents source + sidecar as one raster, in a
folder outside `GEE_Exports_validated_snapshots`. Consuming that in
`Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb` has two failure modes that
are silent rather than loud, so both are pinned here:

1. Returning a `.vrt` *and* a real GeoTIFF for one prefix. They describe the same
   scene, so every point in it would be sampled twice.
2. Staging a `.vrt` as if it were a GeoTIFF. It is a few kB of XML naming its
   sources by absolute path, so the copy succeeds, reads still cross the FUSE
   mount, and nothing reports a problem.

The functions under test live in the notebook, so they are extracted from its
JSON by name: these tests then track the real notebook source instead of a copy
that can drift away from it. No Drive, no Earth Engine, no network.
"""

from __future__ import annotations

import ast
import json
import re
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / \
    'Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb'


def _extract(names):
    """Return source text for the named top-level defs/assignments in the notebook."""
    wanted = set(names)
    found = {}
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if src.lstrip().startswith(('%', '!')):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover - notebook magics
            continue
        lines = src.splitlines(keepends=True)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                name = node.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            else:
                continue
            if name in wanted and name not in found:
                # node.lineno skips decorators, and @contextmanager is load-bearing.
                start = min([d.lineno for d in getattr(node, 'decorator_list', [])],
                            default=node.lineno)
                found[name] = ''.join(lines[start - 1:node.end_lineno])
    missing = wanted - set(found)
    if missing:
        raise AssertionError(
            f'not found in {NOTEBOOK.name}: {sorted(missing)} — the notebook was '
            'edited in a way these tests do not know about'
        )
    return found


@pytest.fixture(scope='module')
def nb_ns():
    """Notebook functions exec'd into a namespace, with config left settable."""
    names = [
        '_CORR_DRIVE_COLLISION_RE', 'PREDICTOR_READ_SUFFIXES',
        'BACKFILL_ROOT', 'BACKFILL_VRT_DIRS', 'USE_BACKFILLED_TEMPORAL_BANDS',
        'backfill_vrt_dirs_for_sensor', 'sensor_from_predictor_prefix',
        '_corr_discover_predictor_tifs',
        '_normalise_drive_name_for_matching', 'parse_predictor_export_name',
        '_staged_vrt_read',
    ]
    found = _extract(names)
    ns = {'Path': Path, 're': re, 'ET': ET, 'ExitStack': ExitStack,
          'contextmanager': contextmanager}
    for name in names:
        exec(found[name], ns)
    return ns


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return path


@pytest.fixture
def archive(tmp_path, nb_ns):
    """An export folder plus S2/S1 backfill VRT folders, wired into the namespace."""
    export = tmp_path / 'GEE_Exports_validated_snapshots'
    export.mkdir()
    vrt_s2 = tmp_path / 'Winam_Temporal_Backfill' / 'vrt' / 'S2'
    vrt_s1 = tmp_path / 'Winam_Temporal_Backfill' / 'vrt' / 'S1'
    vrt_s2.mkdir(parents=True)
    vrt_s1.mkdir(parents=True)
    nb_ns['BACKFILL_VRT_DIRS'] = {'S2': vrt_s2, 'S1': vrt_s1}
    nb_ns['USE_BACKFILLED_TEMPORAL_BANDS'] = False
    return {'export': export, 'S2': vrt_s2, 'S1': vrt_s1}


S2_PREFIX = 'winam_s2_predictors_s2_whlev_temporal_v1_2020-01-01_to_2020-01-02'
S1_PREFIX = 'winam_s1_scc_temporal_v1_2022-03-23_to_2022-03-24'


# --------------------------------------------------------------------------
# Imports the extracted code relies on
# --------------------------------------------------------------------------

def test_notebook_imports_everything_the_new_code_uses():
    """The tests inject these names, so only the notebook itself can prove it has them.

    `_staged_vrt_read` parses VRT XML with ET. Nothing else in the notebook did,
    so the import had to be added; without this check the tests would keep passing
    while the notebook raised NameError on the first backfilled scene.
    """
    src = '\n'.join(
        ''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text())['cells']
        if cell['cell_type'] == 'code'
    )
    expected = {
        'ET': 'import xml.etree.ElementTree as ET',
        'ExitStack': 'ExitStack',
        'contextmanager': 'contextmanager',
    }
    for alias, statement in expected.items():
        assert statement in src, f'{alias} is used but never imported ({statement!r})'


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def test_flag_off_ignores_backfill_entirely(archive, nb_ns):
    _touch(archive['S2'] / f'{S2_PREFIX}.vrt')
    with pytest.raises(FileNotFoundError):
        nb_ns['_corr_discover_predictor_tifs'](S2_PREFIX, archive['export'])


def test_flag_on_finds_the_vrt_when_no_export_exists(archive, nb_ns):
    vrt = _touch(archive['S2'] / f'{S2_PREFIX}.vrt')
    nb_ns['USE_BACKFILLED_TEMPORAL_BANDS'] = True
    assert nb_ns['_corr_discover_predictor_tifs'](S2_PREFIX, archive['export']) == [vrt]


def test_real_export_wins_over_a_backfill_for_the_same_prefix(archive, nb_ns):
    """The double-sampling guard: never both files for one scene."""
    tif = _touch(archive['export'] / f'{S2_PREFIX}.tif')
    _touch(archive['S2'] / f'{S2_PREFIX}.vrt')
    nb_ns['USE_BACKFILLED_TEMPORAL_BANDS'] = True
    assert nb_ns['_corr_discover_predictor_tifs'](S2_PREFIX, archive['export']) == [tif]


def test_tiled_shards_still_resolve_together(archive, nb_ns):
    a = _touch(archive['export'] / f'{S2_PREFIX}-0000000000-0000000000.tif')
    b = _touch(archive['export'] / f'{S2_PREFIX}-0000000000-0000008192.tif')
    _touch(archive['S2'] / f'{S2_PREFIX}.vrt')
    nb_ns['USE_BACKFILLED_TEMPORAL_BANDS'] = True
    assert nb_ns['_corr_discover_predictor_tifs'](S2_PREFIX, archive['export']) == [a, b]


def test_s1_prefix_searches_the_s1_folder_not_the_s2_one(archive, nb_ns):
    """S1 dropped the _predictors_ segment, so sensor routing is by prefix."""
    _touch(archive['S2'] / f'{S1_PREFIX}.vrt')      # wrong folder on purpose
    nb_ns['USE_BACKFILLED_TEMPORAL_BANDS'] = True
    with pytest.raises(FileNotFoundError):
        nb_ns['_corr_discover_predictor_tifs'](S1_PREFIX, archive['export'])

    vrt = _touch(archive['S1'] / f'{S1_PREFIX}.vrt')
    assert nb_ns['_corr_discover_predictor_tifs'](S1_PREFIX, archive['export']) == [vrt]


def test_sensor_routing_matches_the_export_naming_asymmetry(nb_ns):
    assert nb_ns['sensor_from_predictor_prefix'](S2_PREFIX) == 'S2'
    assert nb_ns['sensor_from_predictor_prefix'](S1_PREFIX) == 'S1'
    assert nb_ns['sensor_from_predictor_prefix'](
        'winam_s1_scc_predictors_2020-01-01_to_2020-01-02') == 'S1'


def test_drive_collision_duplicates_are_still_skipped(archive, nb_ns):
    tif = _touch(archive['export'] / f'{S2_PREFIX}.tif')
    _touch(archive['export'] / f'{S2_PREFIX} (1).tif')
    nb_ns['USE_BACKFILLED_TEMPORAL_BANDS'] = True
    assert nb_ns['_corr_discover_predictor_tifs'](S2_PREFIX, archive['export']) == [tif]


# --------------------------------------------------------------------------
# Name parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize('suffix', ['.tif', '.tiff', '.vrt', ''])
def test_export_names_parse_for_every_accepted_suffix(nb_ns, suffix):
    parsed = nb_ns['parse_predictor_export_name'](Path(f'{S2_PREFIX}{suffix}'))
    assert parsed is not None, f'{suffix!r} failed to parse'
    assert parsed['prefix'] == S2_PREFIX
    assert parsed['sensor'] == 'S2'
    assert parsed['s2_schema'] == 's2_whlev_temporal_v1'


def test_s1_vrt_carries_the_schema_token_the_gate_requires(nb_ns):
    """The gate in discover_exported_predictor_sets passes backfills unchanged."""
    parsed = nb_ns['parse_predictor_export_name'](Path(f'{S1_PREFIX}.vrt'))
    assert parsed['sensor'] == 'S1'
    assert parsed['s1_schema'] == 's1_scc_temporal_v1'


def test_unrelated_files_still_return_none(nb_ns):
    assert nb_ns['parse_predictor_export_name'](Path('notes.vrt')) is None
    # A sidecar is named after the source prefix plus a band name, so it must not
    # parse as an export in its own right.
    assert nb_ns['parse_predictor_export_name'](
        Path(f'{S2_PREFIX}_ndvi_temporal_std_w90.tif')) is None


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------

def _vrt_ns(nb_ns, scratch, staged_read, on_drive=False, copies=None):
    """_staged_vrt_read with its Drive-facing collaborators stubbed out."""
    def _copy(source, dest, context):
        if copies is not None:
            copies.append((Path(source), Path(dest)))
        Path(dest).write_bytes(Path(source).read_bytes())
        return Path(dest)

    ns = dict(nb_ns)
    ns['staged_drive_read'] = staged_read
    ns['_local_scratch_path'] = lambda name: scratch / f'local_{name}'
    ns['_safe_remove'] = lambda p: Path(p).unlink(missing_ok=True)
    ns['_path_is_on_drive'] = lambda p: on_drive
    ns['_copy_whole_file_through_drive'] = _copy
    exec(_extract(['_staged_vrt_read'])['_staged_vrt_read'], ns)
    return ns


def _write_vrt(path, sources):
    root = ET.Element('VRTDataset', {'rasterXSize': '10', 'rasterYSize': '10'})
    for i, source in enumerate(sources, 1):
        band = ET.SubElement(root, 'VRTRasterBand', {'band': str(i)})
        simple = ET.SubElement(band, 'SimpleSource')
        node = ET.SubElement(simple, 'SourceFilename', {'relativeToVRT': '0'})
        node.text = str(source)
    ET.ElementTree(root).write(path)
    return path


def test_staging_a_vrt_repoints_it_at_the_staged_sources(tmp_path, nb_ns):
    """Otherwise the copy is 4 kB of XML and every pixel read still hits Drive."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    sources = [_touch(drive / 'source.tif'), _touch(drive / 'sidecar.tif')]
    vrt = _write_vrt(drive / 'scene.vrt', sources)

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    staged = []

    @contextmanager
    def fake_staged_read(path):
        local = scratch / f'staged_{Path(path).name}'
        local.write_bytes(Path(path).read_bytes())
        staged.append(Path(path))
        yield local

    ns = _vrt_ns(nb_ns, scratch, fake_staged_read)

    with ns['_staged_vrt_read'](vrt) as local_vrt:
        assert Path(local_vrt).exists()
        names = [n.text for n in ET.parse(local_vrt).getroot().iter('SourceFilename')]
        rel = [n.get('relativeToVRT')
               for n in ET.parse(local_vrt).getroot().iter('SourceFilename')]

    assert staged == sources, 'every referenced raster must be staged'
    assert [Path(n).parent for n in names] == [scratch, scratch], \
        f'VRT still points outside local scratch: {names}'
    assert rel == ['0', '0']


def test_relative_source_paths_resolve_against_the_vrt(tmp_path, nb_ns):
    """Defensive branch: the backfill writes absolute paths, but GDAL allows both."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    _touch(drive / 'source.tif')
    vrt = _write_vrt(drive / 'scene.vrt', ['source.tif'])

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    staged = []

    @contextmanager
    def fake_staged_read(path):
        staged.append(Path(path))
        local = scratch / f'staged_{Path(path).name}'
        local.write_bytes(Path(path).read_bytes())
        yield local

    ns = _vrt_ns(nb_ns, scratch, fake_staged_read)

    with ns['_staged_vrt_read'](vrt):
        pass
    assert staged == [drive / 'source.tif']


def test_staged_vrt_is_cleaned_up(tmp_path, nb_ns):
    drive = tmp_path / 'drive'
    drive.mkdir()
    _touch(drive / 'source.tif')
    vrt = _write_vrt(drive / 'scene.vrt', [drive / 'source.tif'])

    scratch = tmp_path / 'scratch'
    scratch.mkdir()

    @contextmanager
    def fake_staged_read(path):
        local = scratch / f'staged_{Path(path).name}'
        local.write_bytes(Path(path).read_bytes())
        yield local

    ns = _vrt_ns(nb_ns, scratch, fake_staged_read)

    with ns['_staged_vrt_read'](vrt) as local_vrt:
        held = Path(local_vrt)
        assert held.exists()
    assert not held.exists(), 'staged VRT left behind in scratch'


def test_vrt_xml_on_drive_is_read_through_the_retrying_copy(tmp_path, nb_ns):
    """Parsing the XML in place is the one step the remount/retry wrapper misses."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    source = _touch(drive / 'source.tif')
    vrt = _write_vrt(drive / 'scene.vrt', [source])

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    copies = []

    @contextmanager
    def fake_staged_read(path):
        local = scratch / f'staged_{Path(path).name}'
        local.write_bytes(Path(path).read_bytes())
        yield local

    ns = _vrt_ns(nb_ns, scratch, fake_staged_read, on_drive=True, copies=copies)
    with ns['_staged_vrt_read'](vrt):
        pass

    assert [src for src, _ in copies] == [vrt], \
        'the VRT XML was parsed straight off Drive instead of a staged copy'
    # The temporary XML copy must not outlive the parse.
    assert not (scratch / f'local_{vrt.name}.xml').exists()
