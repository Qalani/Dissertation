"""Tests for the reference-point accuracy section of
``Rules_vs_Logistic_Regression_Comparison.ipynb``.

Sections 4-12 of that notebook measure *concordance* between the rules-based and
Logistic Regression classifiers and explicitly refuse to call it accuracy. Section
13 adds the missing half: the rules-based classifier scored against the labelled
reference points (``SV_S2_Training.csv`` / ``SV_S1_Training.csv``) with the metric
suite ``Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb`` applies to every
candidate model.

These tests pin the things that make those numbers trustworthy:

1. the point-space rule replay reproduces the raster rule functions the classified
   GeoTIFFs were written with, branch for branch;
2. the metric suite exposes exactly the keys the classifier notebook's
   ``_score_split`` exposes, with the same values;
3. metrics are sensor-aware, so S1's three-class scheme is not scored against S2's
   four-class one;
4. the spatial-block folds are the classifier notebook's grouped k-fold: a block is
   never split across the train/test boundary, and every point is held out exactly
   once per repeat;
5. raster sampling at reference points reads the pixel a point actually falls in,
   and reports NoData outside the grid.

As in ``test_rules_vs_lr_class_coding.py``, the code under test lives in the
notebook and is extracted from the notebook JSON by name, so the tests track the
real notebook rather than a copy that can drift. No Drive, no Earth Engine, no
network.
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
pytest.importorskip('sklearn')

from pyproj import CRS, Geod, Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / 'Rules_vs_Logistic_Regression_Comparison.ipynb'
CLASSIFIER_NOTEBOOK = REPO / 'Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb'

NODATA = 255
FLOATING = 2

# UTM 36S: Winam Gulf, as in the sibling test module.
TEST_CRS = 'EPSG:32736'
PIXEL_M = 10.0
ORIGIN_X, ORIGIN_Y = 600_000.0, 9_950_000.0

# Everything section 13 must define for these tests.
WANTED = {
    'SENSOR_CLASS_LABELS', 'FLOATING_CLASS_CODE', 'class_labels_for', 'class_codes_for',
    'staged_copy',
    'S2_RULE_NDMI_ALGAE', 'S2_RULE_NDMI_FLOATING',
    'S1_RULE_VH_P5_LEV_DB', 'S1_RULE_VH_FLOATING_DB',
    'REFERENCE_RAW_CLASS_MAPPING', 'ACCURACY_METRICS',
    'extract_lonlat_from_geo', 'assign_spatial_blocks',
    's2_rule_class_at_points', 's1_rule_class_at_points',
    'classification_metrics', 'per_class_metrics', 'confusion_long_form',
    'spatial_block_folds', 'aggregate_fold_metrics',
    'sample_raster_at_points', 'match_reference_points_to_scenes',
}

CLASSIFIER_WANTED = {'classify_s2_rules_array', 'classify_s1_rules_array'}


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
    return '\n\n'.join(found[name] for name in found)


def _accuracy_namespace(tmp_path=None):
    """Exec the comparison notebook's section 13 helpers."""
    ns = {
        'np': np, 'pd': pd, 'Path': Path, 're': re, 'json': json,
        'hashlib': hashlib, 'shutil': shutil, 'warnings': warnings,
        'rasterio': rasterio, 'Window': Window,
        'CRS': CRS, 'Geod': Geod, 'Transformer': Transformer,
        'accuracy_score': accuracy_score,
        'balanced_accuracy_score': balanced_accuracy_score,
        'cohen_kappa_score': cohen_kappa_score,
        'confusion_matrix': confusion_matrix,
        'f1_score': f1_score,
        'precision_recall_fscore_support': precision_recall_fscore_support,
        # Configuration the extracted code reads as module-level globals.
        'NODATA_VALUE': NODATA,
        'SENSOR_FILTER': ['S2', 'S1'],
        'CLASSIFIER_ROOT': tmp_path or Path('.'),
        'USE_LOCAL_STAGING': False,
        'LOCAL_STAGE_DIR': (tmp_path / 'stage') if tmp_path else Path('.'),
        'SPATIAL_BLOCK_DEGREES': 0.1,
        'ACCURACY_CV_N_SPLITS': 5,
        'ACCURACY_CV_N_REPEATS': 10,
        'ACCURACY_CV_RANDOM_STATE': 42,
        'MAP_MATCH_TOLERANCE_DAYS': 0,
    }
    exec(_extract(NOTEBOOK, WANTED), ns)
    return ns


def _classifier_namespace():
    """Exec the classifier notebook's raster rule functions."""
    ns = {'np': np, 'NODATA_VALUE': NODATA, 'FLOATING_CLASS_CODE': FLOATING}
    exec(_extract(CLASSIFIER_NOTEBOOK, CLASSIFIER_WANTED), ns)
    return ns


# ---------------------------------------------------------------------------
# 1. The point-space replay must match the raster rules.
# ---------------------------------------------------------------------------

def test_s2_point_rules_match_the_raster_rule_classifier():
    """Every S2 point must get the class its pixel would have got."""
    ns = _accuracy_namespace()
    classifier = _classifier_namespace()

    rng = np.random.default_rng(7)
    n = 4000
    # Ranges that straddle every threshold: AWEI_p95 = 0, AWEI = 0, NDMI = 0.1/0.63.
    frame = pd.DataFrame({
        'AWEI_p95': rng.uniform(-1.0, 1.0, n),
        'AWEI': rng.uniform(-1.0, 1.0, n),
        'NDMI': rng.uniform(-0.4, 1.0, n),
    })

    point_codes = ns['s2_rule_class_at_points'](frame)

    bands = {name: frame[name].to_numpy().reshape(1, n) for name in frame.columns}
    raster_codes = classifier['classify_s2_rules_array'](
        bands, np.ones((1, n), dtype=bool)
    )

    assert np.array_equal(point_codes, raster_codes.ravel())
    # The random draw has to actually exercise all four classes, or this proves nothing.
    assert set(np.unique(point_codes)) == {0, 1, 2, 3}


def test_s1_point_rules_match_the_raster_rule_classifier():
    ns = _accuracy_namespace()
    classifier = _classifier_namespace()

    rng = np.random.default_rng(11)
    n = 4000
    frame = pd.DataFrame({
        'VH_p5': rng.uniform(-25.0, -10.0, n),
        'VH_corrected': rng.uniform(-25.0, -8.0, n),
    })

    point_codes = ns['s1_rule_class_at_points'](frame)

    bands = {name: frame[name].to_numpy().reshape(1, n) for name in frame.columns}
    raster_codes = classifier['classify_s1_rules_array'](
        bands, np.ones((1, n), dtype=bool)
    )

    assert np.array_equal(point_codes, raster_codes.ravel())
    assert set(np.unique(point_codes)) == {0, 1, 2}


def test_rule_thresholds_are_exactly_on_the_documented_boundaries():
    """The branch boundaries are inclusive/exclusive exactly as in the paper rules."""
    ns = _accuracy_namespace()

    # AWEI_p95 >= 0 is a water body; AWEI >= 0 is open water; NDMI >= 0.63 is algae;
    # 0.1 <= NDMI < 0.63 is floating; anything else valid stays LEV.
    frame = pd.DataFrame({
        'AWEI_p95': [0.0, 0.0, 0.0, 0.0, -1e-9, 0.0],
        'AWEI': [0.0, -0.1, -0.1, -0.1, -0.1, -0.1],
        'NDMI': [0.5, 0.63, 0.6299999, 0.1, 0.5, 0.0999999],
    })

    assert list(ns['s2_rule_class_at_points'](frame)) == [0, 3, 2, 2, 1, 1]

    # VH_p5 > -17 is LEV; otherwise VH_corrected > -14 is floating.
    s1 = pd.DataFrame({
        'VH_p5': [-17.0, -16.999, -17.0, -17.0],
        'VH_corrected': [-14.0, -10.0, -13.999, -20.0],
    })
    assert list(ns['s1_rule_class_at_points'](s1)) == [0, 1, 2, 0]


def test_points_missing_a_rule_predictor_are_nodata_not_guessed():
    """An unevaluable point must not be silently scored as LEV / open water."""
    ns = _accuracy_namespace()

    s2 = pd.DataFrame({
        'AWEI_p95': [0.5, np.nan, 0.5],
        'AWEI': [-0.2, -0.2, np.nan],
        'NDMI': [0.3, 0.3, 0.3],
    })
    assert list(ns['s2_rule_class_at_points'](s2)) == [2, NODATA, NODATA]

    s1 = pd.DataFrame({'VH_p5': [-20.0, np.nan], 'VH_corrected': [-10.0, -10.0]})
    assert list(ns['s1_rule_class_at_points'](s1)) == [2, NODATA]


def test_reference_class_mapping_matches_the_classifier_notebook():
    """The label -> code maps must not drift from the training mapping."""
    ns = {}
    exec(_extract(NOTEBOOK, {'REFERENCE_RAW_CLASS_MAPPING'}), ns)

    classifier = {}
    exec(
        _extract(CLASSIFIER_NOTEBOOK, {'S2_RAW_CLASS_MAPPING', 'S1_RAW_CLASS_MAPPING'}),
        classifier,
    )

    assert ns['REFERENCE_RAW_CLASS_MAPPING']['S2'] == classifier['S2_RAW_CLASS_MAPPING']
    assert ns['REFERENCE_RAW_CLASS_MAPPING']['S1'] == classifier['S1_RAW_CLASS_MAPPING']


# ---------------------------------------------------------------------------
# 2. The metric suite must be the classifier notebook's.
# ---------------------------------------------------------------------------

def _score_split_keys():
    """Keys the classifier notebook's _score_split() returns."""
    for cell in _cells(CLASSIFIER_NOTEBOOK):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'def _score_split' not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == '_score_split':
                returned = node.body[-1]
                assert isinstance(returned, ast.Return)
                return {key.value for key in returned.value.keys}
    raise AssertionError('_score_split not found in the classifier notebook')


def test_metric_suite_covers_every_score_split_metric():
    ns = _accuracy_namespace()

    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 2])

    metrics = ns['classification_metrics'](y_true, y_pred, labels=[0, 1, 2, 3])

    assert _score_split_keys() <= set(metrics)
    # ACCURACY_METRICS is what the CV aggregation averages; all of it must be produced.
    assert set(ns['ACCURACY_METRICS']) <= set(metrics)


def test_metric_values_match_scikit_learn_directly():
    ns = _accuracy_namespace()

    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 2])
    labels = [0, 1, 2, 3]

    metrics = ns['classification_metrics'](y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[FLOATING], zero_division=0
    )

    assert metrics['accuracy'] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics['balanced_accuracy'] == pytest.approx(
        balanced_accuracy_score(y_true, y_pred))
    assert metrics['kappa'] == pytest.approx(
        cohen_kappa_score(y_true, y_pred, labels=labels))
    assert metrics['macro_f1'] == pytest.approx(
        f1_score(y_true, y_pred, average='macro', zero_division=0))
    assert metrics['weighted_f1'] == pytest.approx(
        f1_score(y_true, y_pred, average='weighted', zero_division=0))
    assert metrics['floating_precision'] == pytest.approx(precision[0])
    assert metrics['floating_recall'] == pytest.approx(recall[0])
    assert metrics['floating_f1'] == pytest.approx(f1[0])
    assert metrics['floating_support'] == 2
    assert metrics['n_points'] == 8


def test_the_label_list_reaches_kappa_and_only_kappa():
    """`labels` is the sensor's class scheme; `_score_split` passes it to kappa alone.

    The macro/weighted F1 in the classifier notebook are computed over the classes
    actually present, so a rules row stays comparable with the model rows it is
    tabulated beside. Scoring them over the full scheme instead would penalise every
    fold that happens to contain no surface algae.
    """
    ns = _accuracy_namespace()

    # Perfect agreement on open water, LEV and surface algae: floating (code 2) is
    # in the S2 scheme but absent from the data.
    y_true = np.array([0, 0, 1, 3])
    y_pred = np.array([0, 0, 1, 3])

    metrics = ns['classification_metrics'](
        y_true, y_pred, labels=list(ns['class_codes_for']('S2')))

    assert metrics['kappa'] == pytest.approx(
        cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3]))
    # Over observed classes this is 1.0; over the full scheme the absent floating
    # class would drag it to 0.75.
    assert metrics['macro_f1'] == pytest.approx(1.0)
    assert metrics['weighted_f1'] == pytest.approx(1.0)
    # The floating metrics are still reported for the absent class, as zeros.
    assert metrics['floating_support'] == 0
    assert metrics['floating_f1'] == pytest.approx(0.0)


def test_per_class_metrics_are_sensor_sized_and_name_the_accuracy_conventions():
    ns = _accuracy_namespace()

    y_true = np.array([0, 1, 2, 2, 3])
    y_pred = np.array([0, 1, 2, 1, 3])

    rows = ns['per_class_metrics'](
        y_true, y_pred, 'S2', classifier='Rules', evaluation='test')

    assert len(rows) == 4
    assert [r['class_name'] for r in rows] == [
        'Open water', 'LEV', 'Floating plants', 'Surface algae']

    floating = next(r for r in rows if r['class_code'] == FLOATING)
    assert floating['reference_points'] == 2
    assert floating['predicted_points'] == 1
    # One of two floating reference points was mapped as floating, and the one
    # floating prediction was right: recall 0.5, precision 1.0.
    assert floating['recall_producers_accuracy'] == pytest.approx(0.5)
    assert floating['precision_users_accuracy'] == pytest.approx(1.0)

    s1_rows = ns['per_class_metrics'](
        np.array([0, 1, 2]), np.array([0, 1, 2]), 'S1',
        classifier='Rules', evaluation='test')
    assert len(s1_rows) == 3
    assert 'Surface algae' not in {r['class_name'] for r in s1_rows}


def test_confusion_long_form_orients_reference_as_rows():
    ns = _accuracy_namespace()

    # Three LEV reference points, two of them mapped as floating.
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 2, 2, 0])

    rows, matrix = ns['confusion_long_form'](
        y_true, y_pred, 'S1', classifier='Rules', evaluation='test')

    assert matrix.shape == (3, 3)
    assert matrix[1, 2] == 2  # reference LEV -> mapped floating
    assert matrix[2, 1] == 0  # and not the transpose

    confused = next(
        r for r in rows
        if r['reference_code'] == 1 and r['predicted_code'] == FLOATING
    )
    assert confused['reference_class'] == 'LEV'
    assert confused['predicted_class'] == 'Floating plants'
    assert confused['points'] == 2
    assert confused['row_percent'] == pytest.approx(100 * 2 / 3)


# ---------------------------------------------------------------------------
# 3. Spatial-block folds must be the classifier notebook's grouped k-fold.
# ---------------------------------------------------------------------------

def test_spatial_blocks_are_computed_like_the_classifier_notebook():
    ns = _accuracy_namespace()

    frame = pd.DataFrame({'lon': [34.25, 34.29, 34.31], 'lat': [-0.35, -0.31, -0.35]})
    blocks = ns['assign_spatial_blocks'](frame, block_degrees=0.1)

    # Same 0.1-degree cell -> same block; a different cell -> a different block.
    assert blocks.iloc[0] == blocks.iloc[1]
    assert blocks.iloc[0] != blocks.iloc[2]

    expected = np.floor(34.25 / 0.1).astype(np.int64) * 100000 + np.floor(-0.35 / 0.1).astype(np.int64)
    assert blocks.iloc[0] == expected


def test_spatial_block_folds_never_split_a_block_and_hold_out_everything_once():
    ns = _accuracy_namespace()

    groups = np.repeat(np.arange(20), 5)  # 20 blocks, 5 points each
    folds = list(ns['spatial_block_folds'](groups, n_splits=5, n_repeats=3))

    assert len(folds) == 15

    for repeat in range(3):
        held_out = [idx for r, _, idx in folds if r == repeat]
        combined = np.concatenate(held_out)
        # Every point is held out exactly once per repeat.
        assert np.array_equal(np.sort(combined), np.arange(len(groups)))

        for test_idx in held_out:
            test_blocks = set(groups[test_idx])
            train_idx = np.setdiff1d(np.arange(len(groups)), test_idx)
            # No block appears on both sides of the split.
            assert not (test_blocks & set(groups[train_idx]))


def test_spatial_block_folds_are_deterministic_and_reshuffle_per_repeat():
    ns = _accuracy_namespace()

    groups = np.repeat(np.arange(12), 3)

    first = [(r, f, idx.tolist()) for r, f, idx in ns['spatial_block_folds'](
        groups, n_splits=4, n_repeats=2, random_state=42)]
    second = [(r, f, idx.tolist()) for r, f, idx in ns['spatial_block_folds'](
        groups, n_splits=4, n_repeats=2, random_state=42)]
    other_seed = [(r, f, idx.tolist()) for r, f, idx in ns['spatial_block_folds'](
        groups, n_splits=4, n_repeats=2, random_state=7)]

    assert first == second          # same seed, same folds
    assert first != other_seed      # the seed actually drives the assignment

    repeat_0 = [idx for r, _, idx in first if r == 0]
    repeat_1 = [idx for r, _, idx in first if r == 1]
    assert repeat_0 != repeat_1     # blocks are reshuffled between repeats


def test_spatial_block_folds_refuse_fewer_blocks_than_folds():
    ns = _accuracy_namespace()

    with pytest.raises(ValueError, match='spatial blocks'):
        list(ns['spatial_block_folds'](np.array([1, 1, 2, 2]), n_splits=5, n_repeats=1))


def test_fold_aggregation_uses_the_classifier_leaderboard_schema():
    ns = _accuracy_namespace()

    per_fold = pd.DataFrame({
        metric: [0.6, 0.8] for metric in ns['ACCURACY_METRICS']
    })

    agg = ns['aggregate_fold_metrics'](per_fold, 'S2', 'Rules-based (paper thresholds)')

    assert agg['sensor'] == 'S2'
    assert agg['model'] == 'Rules-based (paper thresholds)'
    assert agg['n_folds_evaluated'] == 2
    for metric in ns['ACCURACY_METRICS']:
        assert agg[f'{metric}_mean'] == pytest.approx(0.7)
        # Sample standard deviation, as pandas .std() gives in the classifier notebook.
        assert agg[f'{metric}_std'] == pytest.approx(np.std([0.6, 0.8], ddof=1))


# ---------------------------------------------------------------------------
# 4. Map-space sampling.
# ---------------------------------------------------------------------------

def _write_raster(path, codes):
    arr = np.asarray(codes, dtype=np.uint8)
    profile = {
        'driver': 'GTiff', 'height': arr.shape[0], 'width': arr.shape[1],
        'count': 1, 'dtype': 'uint8', 'nodata': NODATA, 'crs': TEST_CRS,
        'transform': from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_M, PIXEL_M),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)
    return path


def _lonlat_for_pixel(row, col):
    """Centre of pixel (row, col) as EPSG:4326 lon/lat."""
    x = ORIGIN_X + (col + 0.5) * PIXEL_M
    y = ORIGIN_Y - (row + 0.5) * PIXEL_M
    transformer = Transformer.from_crs(TEST_CRS, 'EPSG:4326', always_xy=True)
    return transformer.transform(x, y)


def test_sample_raster_at_points_reads_the_pixel_under_each_point(tmp_path):
    ns = _accuracy_namespace(tmp_path)

    codes = [[0, 1, 2],
             [3, NODATA, 1]]
    tif = _write_raster(tmp_path / 'classified.tif', codes)

    targets = [(0, 0), (0, 2), (1, 0), (1, 1)]
    lons, lats = zip(*[_lonlat_for_pixel(r, c) for r, c in targets])

    values = ns['sample_raster_at_points'](tif, lons, lats)

    assert list(values) == [0, 2, 3, NODATA]


def test_sample_raster_at_points_reports_nodata_outside_the_grid(tmp_path):
    ns = _accuracy_namespace(tmp_path)

    tif = _write_raster(tmp_path / 'classified.tif', [[2, 2], [2, 2]])

    inside = _lonlat_for_pixel(0, 0)
    far_west = (inside[0] - 5.0, inside[1])

    values = ns['sample_raster_at_points'](
        tif, [inside[0], far_west[0]], [inside[1], far_west[1]])

    assert values[0] == FLOATING
    assert values[1] == NODATA


def _manifest(rows):
    manifest = pd.DataFrame(rows)
    manifest['start_date'] = pd.to_datetime(manifest['start_date'])
    manifest['end_date'] = pd.to_datetime(manifest['end_date'])
    return manifest


def test_reference_points_match_the_scene_covering_their_acquisition_date():
    ns = _accuracy_namespace()

    points = pd.DataFrame({
        'sensor': ['S2', 'S2'],
        'date': ['2021-02-10', '2019-01-01'],
        'lon': [34.5, 34.5],
        'lat': [-0.3, -0.3],
        'reference_code': [2, 2],
        'reference_class': ['Floating plants', 'Floating plants'],
        'spatial_block': [1, 1],
    })

    manifest = _manifest([
        {'sensor': 'S2', 'prefix': 'covers', 'start_date': '2021-02-10',
         'end_date': '2021-02-11', 'model_tif': 'lr.tif', 'rule_tif': 'rules.tif'},
        {'sensor': 'S2', 'prefix': 'later', 'start_date': '2021-03-10',
         'end_date': '2021-03-11', 'model_tif': 'lr.tif', 'rule_tif': 'rules.tif'},
    ])

    matched = ns['match_reference_points_to_scenes'](points, manifest)

    # The uncovered 2019 point is dropped rather than snapped to a distant scene.
    assert len(matched) == 1
    assert matched['prefix'].iloc[0] == 'covers'
    assert matched['n_candidate_scenes'].iloc[0] == 1


def test_a_point_inside_several_periods_is_matched_once_to_the_closest():
    ns = _accuracy_namespace()

    points = pd.DataFrame({
        'sensor': ['S1'],
        'date': ['2021-02-02'],
        'lon': [34.5], 'lat': [-0.3],
        'reference_code': [1], 'reference_class': ['LEV'], 'spatial_block': [1],
    })

    manifest = _manifest([
        {'sensor': 'S1', 'prefix': 'wide', 'start_date': '2021-01-01',
         'end_date': '2021-03-01', 'model_tif': 'lr.tif', 'rule_tif': 'rules.tif'},
        {'sensor': 'S1', 'prefix': 'tight', 'start_date': '2021-02-02',
         'end_date': '2021-02-03', 'model_tif': 'lr.tif', 'rule_tif': 'rules.tif'},
    ])

    matched = ns['match_reference_points_to_scenes'](points, manifest)

    assert len(matched) == 1  # counted once, not once per overlapping period
    assert matched['prefix'].iloc[0] == 'tight'
    assert matched['n_candidate_scenes'].iloc[0] == 2


def test_points_are_never_matched_across_sensors():
    ns = _accuracy_namespace()

    points = pd.DataFrame({
        'sensor': ['S2'],
        'date': ['2021-02-10'],
        'lon': [34.5], 'lat': [-0.3],
        'reference_code': [2], 'reference_class': ['Floating plants'],
        'spatial_block': [1],
    })

    manifest = _manifest([
        {'sensor': 'S1', 'prefix': 's1_scene', 'start_date': '2021-02-10',
         'end_date': '2021-02-11', 'model_tif': 'lr.tif', 'rule_tif': 'rules.tif'},
    ])

    assert ns['match_reference_points_to_scenes'](points, manifest).empty


def test_match_tolerance_widens_the_scene_period():
    ns = _accuracy_namespace()

    points = pd.DataFrame({
        'sensor': ['S2'],
        'date': ['2021-02-13'],
        'lon': [34.5], 'lat': [-0.3],
        'reference_code': [2], 'reference_class': ['Floating plants'],
        'spatial_block': [1],
    })

    manifest = _manifest([
        {'sensor': 'S2', 'prefix': 'near', 'start_date': '2021-02-10',
         'end_date': '2021-02-11', 'model_tif': 'lr.tif', 'rule_tif': 'rules.tif'},
    ])

    assert ns['match_reference_points_to_scenes'](
        points, manifest, tolerance_days=0).empty
    assert len(ns['match_reference_points_to_scenes'](
        points, manifest, tolerance_days=3)) == 1


def test_extract_lonlat_handles_the_csv_geo_forms():
    ns = _accuracy_namespace()

    point = json.dumps({'type': 'Point', 'coordinates': [34.5, -0.3]})
    feature = json.dumps({
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [34.5, -0.3]},
    })

    assert ns['extract_lonlat_from_geo'](point) == (34.5, -0.3)
    assert ns['extract_lonlat_from_geo'](feature) == (34.5, -0.3)
    assert ns['extract_lonlat_from_geo']({'type': 'Point', 'coordinates': [34.5, -0.3]}) == (
        34.5, -0.3)
