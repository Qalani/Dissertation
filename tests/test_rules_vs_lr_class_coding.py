"""Tests for the class coding used by Rules_vs_Logistic_Regression_Comparison.ipynb.

The Winam classifier does not use one class scheme, it uses one per sensor
(``S2_CLASS_NAMES`` / ``S1_CLASS_NAMES`` in
``Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb``):

* **S2** has four classes, because surface algae is separable in optical data.
  Both the training labels (``'A' -> 3``) and the S2 rule branch
  ``NDMI >= 0.63`` emit code ``3``.
* **S1** has three, because surface algae is not separable from open water in
  SAR backscatter, so ``'A'`` is folded into code ``0`` and no S1 raster ever
  contains a ``3``.

The comparison notebook originally hard-coded a single ``(0, 1, 2)`` scheme for
both sensors, so every S2 scene aborted with::

    ValueError: ... LR contains unexpected non-NoData class codes: [3].

These tests pin the per-sensor behaviour:

1. an S2 pair containing surface algae compares cleanly and reports a 4x4
   agreement matrix;
2. an S1 raster containing a ``3`` is still rejected, because for S1 that
   really is a corrupt code;
3. the notebook's class maps do not drift away from the classifier notebook's;
4. disagreement transition codes stay inside uint8 and clear of NoData.

The code under test lives in the notebook, so it is extracted from the notebook
JSON by name and exec'd: the tests track the real notebook rather than a copy
that can drift. No Drive, no Earth Engine, no network.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('pyproj')

from pyproj import CRS, Geod  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.windows import Window  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / 'Rules_vs_Logistic_Regression_Comparison.ipynb'
CLASSIFIER_NOTEBOOK = REPO / 'Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb'

# UTM 36S: Winam Gulf sits here, and a projected CRS exercises the constant
# pixel-area path rather than the geodesic one.
TEST_CRS = 'EPSG:32736'
PIXEL_M = 10.0
PIXEL_AREA_HA = (PIXEL_M ** 2) / 10_000.0

NODATA = 255

# Every top-level name the comparison notebook must define for these tests.
WANTED = {
    'SENSOR_CLASS_LABELS', 'FLOATING_CLASS_CODE',
    'class_labels_for', 'class_codes_for',
    'iter_fixed_windows', 'assert_same_grid', 'staged_copy',
    'class_valid', 'check_unknown_codes', 'cohen_kappa_from_matrix',
    '_GEOD', 'projected_pixel_area_ha', 'geographic_row_pixel_areas_ha',
    'row_pixel_areas_ha', 'matrix_long_form', 'compare_scene',
}


def _cells(notebook):
    return json.loads(notebook.read_text())['cells']


def _extract(notebook, wanted):
    """Source text of the notebook's top-level defs/assignments named in ``wanted``."""
    found = {}
    for cell in _cells(notebook):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - a broken cell fails elsewhere
            continue
        for node in tree.body:
            names = []
            if isinstance(node, ast.FunctionDef):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if name in wanted:
                    found[name] = ast.get_source_segment(source, node)
    missing = wanted - set(found)
    assert not missing, f'{notebook.name} is missing {sorted(missing)}'
    # Preserve notebook order so helpers are defined before their callers.
    return '\n\n'.join(found[name] for name in found)


def _load_notebook_namespace(tmp_path, write_rasters=True, fail_on_unknown=True):
    """Exec the comparison notebook's class-coding and comparison code."""
    disagreement_dir = tmp_path / 'disagreement_geotiffs'
    disagreement_dir.mkdir(exist_ok=True)

    ns = {
        'np': np, 'pd': pd, 'Path': Path, 're': re,
        'hashlib': hashlib, 'shutil': shutil, 'warnings': warnings,
        'rasterio': rasterio, 'Window': Window, 'CRS': CRS, 'Geod': Geod,
        # Configuration the extracted code reads as module-level globals.
        'NODATA_VALUE': NODATA,
        'SENSOR_FILTER': ['S2', 'S1'],
        'FAIL_ON_UNKNOWN_CLASS_CODE': fail_on_unknown,
        'STRICT_GRID_MATCH': True,
        'READ_BLOCKSIZE': 2048,
        'USE_LOCAL_STAGING': False,
        'LOCAL_STAGE_DIR': tmp_path / 'stage',
        'DISAGREEMENT_DIR': disagreement_dir,
        'WRITE_DISAGREEMENT_RASTERS': write_rasters,
        'OVERWRITE_OUTPUTS': True,
    }
    exec(_extract(NOTEBOOK, WANTED), ns)
    return ns


def _write_raster(path, codes):
    """Write a single-band uint8 classification raster from a 2-D code array."""
    arr = np.asarray(codes, dtype=np.uint8)
    profile = {
        'driver': 'GTiff', 'height': arr.shape[0], 'width': arr.shape[1],
        'count': 1, 'dtype': 'uint8', 'nodata': NODATA, 'crs': TEST_CRS,
        'transform': from_origin(600_000.0, 9_950_000.0, PIXEL_M, PIXEL_M),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)
    return path


def _row(tmp_path, sensor, prefix, lr_codes, rule_codes):
    model_tif = _write_raster(tmp_path / f'{prefix}_lr.tif', lr_codes)
    rule_tif = _write_raster(tmp_path / f'{prefix}_rules.tif', rule_codes)
    return {
        'sensor': sensor,
        'prefix': prefix,
        'start_date': pd.Timestamp('2017-04-02'),
        'end_date': pd.Timestamp('2017-04-03'),
        'model_slug': 'logistic_regression_baseline',
        'model_patch_cleaned': False,
        'model_tif': str(model_tif),
        'rule_tif': str(rule_tif),
    }


# The exact scene from the reported traceback.
S2_PREFIX = 'winam_s2_predictors_s2_whlev_temporal_v1_2017-04-02_to_2017-04-03'
S1_PREFIX = 'winam_s1_scc_temporal_v1_2017-04-02_to_2017-04-03'


def test_s2_scene_containing_surface_algae_compares_without_error(tmp_path):
    """The reported failure: an S2 raster with code 3 must compare, not abort."""
    ns = _load_notebook_namespace(tmp_path)

    lr = [[0, 1, 2, 3],
          [3, 3, 2, 1],
          [0, 0, NODATA, 3]]
    rules = [[0, 1, 2, 3],
             [0, 3, 2, 1],
             [0, 1, NODATA, 2]]

    summary, class_rows, matrix_rows, matrix = ns['compare_scene'](
        _row(tmp_path, 'S2', S2_PREFIX, lr, rules)
    )

    # Four classes in, four-by-four agreement matrix out.
    assert matrix.shape == (4, 4)
    assert matrix.sum() == 11  # every pixel but the NoData one
    assert len(class_rows) == 4
    assert len(matrix_rows) == 16

    # Surface algae is carried as a real class, not dropped or relabelled.
    algae = next(r for r in class_rows if r['class_code'] == 3)
    assert algae['class_name'] == 'Surface algae'
    assert algae['lr_pixels'] == 4      # LR labelled four pixels as algae
    assert algae['rule_pixels'] == 2    # rules labelled two
    assert algae['intersection_pixels'] == 2
    assert algae['lr_area_ha'] == pytest.approx(4 * PIXEL_AREA_HA)
    assert algae['iou'] == pytest.approx(2 / 4)

    # 8 of 11 common-valid pixels agree.
    assert summary['overall_agreement'] == pytest.approx(8 / 11)
    assert summary['common_valid_pixels'] == 11
    assert np.trace(matrix) == 8


def test_s2_class_codes_are_not_silently_collapsed(tmp_path):
    """Surface-algae pixels must not be counted into any other class."""
    ns = _load_notebook_namespace(tmp_path)

    # Every pixel is surface algae under both classifiers.
    summary, class_rows, _, matrix = ns['compare_scene'](
        _row(tmp_path, 'S2', S2_PREFIX, [[3, 3], [3, 3]], [[3, 3], [3, 3]])
    )

    assert summary['overall_agreement'] == pytest.approx(1.0)
    assert matrix[3, 3] == 4
    assert matrix.sum() == 4

    for row in class_rows:
        if row['class_code'] != 3:
            assert row['lr_pixels'] == 0, row['class_name']
            assert row['rule_pixels'] == 0, row['class_name']

    # Floating plants stay code 2 on S2, so an all-algae scene maps none.
    assert summary['floating_lr_area_ha'] == 0.0
    assert summary['floating_rule_area_ha'] == 0.0


def test_s1_scene_uses_the_three_class_scheme(tmp_path):
    """S1 keeps three classes and its own open-water label."""
    ns = _load_notebook_namespace(tmp_path)

    summary, class_rows, matrix_rows, matrix = ns['compare_scene'](
        _row(tmp_path, 'S1', S1_PREFIX, [[0, 1, 2]], [[0, 1, 1]])
    )

    assert matrix.shape == (3, 3)
    assert len(class_rows) == 3
    assert len(matrix_rows) == 9
    assert [r['class_name'] for r in class_rows] == [
        'Open water / surface algae', 'LEV', 'Floating plants',
    ]
    assert summary['overall_agreement'] == pytest.approx(2 / 3)


def test_s1_raster_containing_code_three_is_still_rejected(tmp_path):
    """A 3 is valid on S2 and corrupt on S1; the check must stay sensor-aware."""
    ns = _load_notebook_namespace(tmp_path)

    with pytest.raises(ValueError, match=r'unexpected non-NoData class codes: \[3\]'):
        ns['compare_scene'](
            _row(tmp_path, 'S1', S1_PREFIX, [[0, 1, 3]], [[0, 1, 2]])
        )


def test_unknown_code_message_names_the_sensor_scheme(tmp_path):
    ns = _load_notebook_namespace(tmp_path)
    arr = np.array([[0, 1, 7]], dtype=np.uint8)

    with pytest.raises(ValueError) as excinfo:
        ns['check_unknown_codes'](arr, arr != NODATA, 'scene LR', 'S2')

    message = str(excinfo.value)
    assert '[7]' in message
    assert '(0, 1, 2, 3)' in message  # the S2 scheme, not a generic one
    assert 'S2' in message

    # A code that is merely unknown on S1 reports the S1 scheme instead.
    with pytest.raises(ValueError) as excinfo:
        ns['check_unknown_codes'](arr, arr != NODATA, 'scene LR', 'S1')
    assert '(0, 1, 2)' in str(excinfo.value)


def test_unknown_codes_are_excluded_when_failure_is_disabled(tmp_path):
    """With the guard off, stray codes warn and are dropped, never miscounted."""
    ns = _load_notebook_namespace(tmp_path, fail_on_unknown=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        summary, _, _, matrix = ns['compare_scene'](
            _row(tmp_path, 'S1', S1_PREFIX, [[0, 1, 3]], [[0, 1, 2]])
        )

    assert any('unexpected non-NoData class codes' in str(w.message) for w in caught)
    # The stray pixel is not common-valid, so it never reaches the matrix.
    assert matrix.sum() == 2
    assert summary['common_valid_pixels'] == 2


def test_class_maps_match_the_classifier_notebook():
    """The comparison notebook must not drift from the classifier's class maps."""
    ns = {}
    exec(_extract(NOTEBOOK, {'SENSOR_CLASS_LABELS', 'FLOATING_CLASS_CODE'}), ns)

    classifier = {}
    exec(
        _extract(CLASSIFIER_NOTEBOOK, {'S2_CLASS_NAMES', 'S1_CLASS_NAMES'}),
        classifier,
    )

    assert ns['SENSOR_CLASS_LABELS']['S2'] == classifier['S2_CLASS_NAMES']
    assert ns['SENSOR_CLASS_LABELS']['S1'] == classifier['S1_CLASS_NAMES']

    # Floating plants share one code across sensors; the notebook relies on it.
    for labels in ns['SENSOR_CLASS_LABELS'].values():
        assert labels[ns['FLOATING_CLASS_CODE']] == 'Floating plants'


def test_disagreement_transitions_stay_inside_uint8(tmp_path):
    """The 10*(LR+1)+(rules+1) encoding must survive the four-class S2 scheme."""
    ns = _load_notebook_namespace(tmp_path)

    # LR calls everything surface algae, the rules call it all open water:
    # the largest transition value the S2 scheme can produce.
    row = _row(tmp_path, 'S2', S2_PREFIX, [[3, 3], [3, 3]], [[0, 0], [0, 0]])
    summary, _, _, _ = ns['compare_scene'](row)

    with rasterio.open(summary['disagreement_tif']) as src:
        written = src.read(1)

    assert written.dtype == np.uint8
    assert np.all(written == 41)  # 10 * (3 + 1) + (0 + 1)
    assert 41 < NODATA
    assert summary['disagreement_fraction'] == pytest.approx(1.0)


def test_transition_values_are_unique_per_sensor_scheme():
    """No two class transitions may collide, and none may collide with NoData."""
    ns = {}
    exec(_extract(NOTEBOOK, {'SENSOR_CLASS_LABELS', 'class_labels_for',
                             'class_codes_for'}), ns)

    for sensor in ns['SENSOR_CLASS_LABELS']:
        values = {
            10 * (lr + 1) + (rule + 1)
            for lr in ns['class_codes_for'](sensor)
            for rule in ns['class_codes_for'](sensor)
            if lr != rule
        }
        expected = len(ns['class_codes_for'](sensor)) ** 2 - len(
            ns['class_codes_for'](sensor)
        )
        assert len(values) == expected, sensor
        assert 0 not in values, sensor       # 0 means agreement
        assert max(values) < NODATA, sensor


def test_class_lookup_rejects_an_unconfigured_sensor():
    ns = {}
    exec(_extract(NOTEBOOK, {'SENSOR_CLASS_LABELS', 'class_labels_for',
                             'class_codes_for'}), ns)

    with pytest.raises(KeyError, match='No class coding is configured'):
        ns['class_labels_for']('S3')


def test_matrix_long_form_is_sized_per_sensor(tmp_path):
    ns = _load_notebook_namespace(tmp_path)

    s2_rows = ns['matrix_long_form'](np.zeros((4, 4), dtype=np.int64), 'S2', 'p')
    s1_rows = ns['matrix_long_form'](np.zeros((3, 3), dtype=np.int64), 'S1', 'p')

    assert len(s2_rows) == 16
    assert len(s1_rows) == 9
    assert {r['lr_code'] for r in s2_rows} == {0, 1, 2, 3}
    assert {r['lr_code'] for r in s1_rows} == {0, 1, 2}
    assert {r['lr_class'] for r in s2_rows} >= {'Surface algae'}
    assert 'Surface algae' not in {r['lr_class'] for r in s1_rows}
