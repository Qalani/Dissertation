"""Tests for how the spatial-panel notebooks select classified GeoTIFFs.

The panel response must come from ONE classifier run and ONE batch-export schema
per sensor. Everything that previously let other rasters in is silent rather than
loud, so each route is pinned here:

1. picking the run log by mtime / a ``winam_full_stack_run_log_*.csv`` wildcard,
   which promotes whichever run finished last;
2. supplementing the run log from ``classified_geotiffs/``, which adds rasters
   from other classifier runs and other export schemas;
3. loose substring matching, which lets a superseded-schema prefix pass;
4. probability, paper-rule and pre-mosaic tile rasters entering the hard-class
   response;
5. two competing rasters for one sensor/acquisition being averaged together.

The functions under test live in the notebooks, so they are extracted from the
notebook JSON by name: the tests then track the real notebook source instead of a
copy that can drift away from it. No Drive, no Earth Engine, no network.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(REPO.glob('winam_wh_spatial_panel*.ipynb'))

CLASSIFIER_VERSION = 'route_b_s1_scc_spatialcv_proba_v4_whlev_temporal_corr_v1'
S1_PREFIX = 'winam_s1_scc_temporal_v1_'
S2_PREFIX = 'winam_s2_predictors_s2_whlev_temporal_v1_'
S1_TOKEN = 's1_scc_temporal_v1'
S2_TOKEN = 's2_whlev_temporal_v1'

RUN_LOG_COLUMNS = [
    'sensor', 'start_date', 'end_date', 'prefix', 'status',
    'model_classification_tif', 'model_probability_tif', 'rule_classification_tif',
]

# Every top-level name the selection block defines.
WANTED = {
    '_normalise_classifier_sensor', '_PROBABILITY_SUFFIX', '_RULE_SUFFIX',
    '_PATCH_CLEANED_SUFFIX', '_MODEL_SEGMENT', '_TILE_INTERMEDIATE_RE',
    '_GEOTIFF_SUFFIXES', 'required_classified_prefix', 'required_export_token',
    '_check_prefix_token_consistency', '_classifier_sensor_from_path',
    '_export_token_from_path', '_classifier_product_from_path',
    '_path_is_final_model_classification', '_patch_clean_base_stem',
    '_blank_run_log_value', 'read_active_classifier_run_log',
    '_records_from_classifier_run_log', '_resolve_canonical_records',
    'find_classified_tifs', '_build_selection_audit', 'validate_panel_provenance',
    'CLASSIFIED_SELECTION_AUDIT', 'PANEL_SOURCE_METADATA_COLUMNS',
    'panel_source_column', 'stamp_source_metadata',
}


def _extract_selection_source(notebook):
    """Source text of the selection block's top-level defs/assignments."""
    found = {}
    for cell in json.loads(notebook.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'def find_classified_tifs' not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            names = []
            if isinstance(node, ast.FunctionDef):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if name in WANTED:
                    found[name] = ast.get_source_segment(source, node)
    missing = WANTED - set(found)
    assert not missing, f'{notebook.name}: selection block is missing {sorted(missing)}'
    return '\n\n'.join(found[n] for n in found)


def _load_selection(notebook, tmp_path, sensor_filter=('S2', 'S1'),
                    prefer_patch_cleaned=True, supplement=False,
                    product_filter='model', use_run_log=True,
                    classifier_version=CLASSIFIER_VERSION):
    """Exec the notebook's selection block against a temporary Drive layout."""
    table_dir = tmp_path / 'tables'
    tif_dir = tmp_path / 'classified_geotiffs'
    table_dir.mkdir(exist_ok=True)
    tif_dir.mkdir(exist_ok=True)

    ns = {
        'pd': pd, 'Path': Path, 're': re,
        'ACTIVE_CLASSIFIER_VERSION': classifier_version,
        'CLASSIFIER_TABLE_DIR': table_dir,
        'CLASSIFIED_TIF_DIR': tif_dir,
        'CLASSIFIER_RUN_LOG_PATH':
            table_dir / f'winam_full_stack_run_log_{classifier_version}.csv',
        'REQUIRED_EXPORT_TOKEN_BY_SENSOR': {'S1': S1_TOKEN, 'S2': S2_TOKEN},
        'REQUIRED_CLASSIFIED_PREFIX_BY_SENSOR': {'S1': S1_PREFIX, 'S2': S2_PREFIX},
        'CLASSIFIER_SENSOR_FILTER': list(sensor_filter) if sensor_filter else None,
        'CLASSIFIER_PRODUCT_FILTER': product_filter,
        'PREFER_PATCH_CLEANED_S1': prefer_patch_cleaned,
        'USE_CLASSIFIER_RUN_LOG': use_run_log,
        'SUPPLEMENT_RUN_LOG_WITH_FOLDER': supplement,
    }
    exec(_extract_selection_source(notebook), ns)
    ns['_table_dir'] = table_dir
    ns['_tif_dir'] = tif_dir
    return ns


def _touch(directory, name):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'II*\x00')  # enough to exist; nothing reads the pixels here
    return path


def _row(sensor, start, end, model_tif, proba_tif='', rule_tif='', status='completed'):
    return {
        'sensor': sensor, 'start_date': start, 'end_date': end,
        'prefix': Path(model_tif).stem if model_tif else '', 'status': status,
        'model_classification_tif': str(model_tif),
        'model_probability_tif': str(proba_tif),
        'rule_classification_tif': str(rule_tif),
    }


def _write_run_log(ns, rows, name=None):
    path = ns['CLASSIFIER_RUN_LOG_PATH'] if name is None else ns['_table_dir'] / name
    pd.DataFrame(rows, columns=RUN_LOG_COLUMNS).to_csv(path, index=False)
    return path


def _s2_names(start='2021-06-01', end='2021-06-02', slug='random_forest'):
    stem = f'{S2_PREFIX}{start}_to_{end}_local_{slug}'
    return f'{stem}.tif', f'{stem}_proba.tif', f'{S2_PREFIX}{start}_to_{end}_local_rules.tif'


def _s1_names(start='2021-06-03', end='2021-06-04', slug='random_forest'):
    stem = f'{S1_PREFIX}{start}_to_{end}_local_{slug}'
    return f'{stem}.tif', f'{stem}_patch_cleaned.tif', f'{stem}_proba.tif'


@pytest.fixture(params=NOTEBOOKS, ids=lambda p: p.stem)
def notebook(request):
    return request.param


# ---------------------------------------------------------------------------
# Configuration is the same everywhere and matches the upstream notebooks.
# ---------------------------------------------------------------------------

def _code(notebook):
    """Concatenated source of the notebook's code cells."""
    return '\n'.join(
        ''.join(c['source'])
        for c in json.loads(notebook.read_text())['cells']
        if c['cell_type'] == 'code'
    )


def test_every_panel_notebook_pins_the_same_classifier_run(notebook):
    code = _code(notebook)
    assert f'"{CLASSIFIER_VERSION}"' in code
    assert f'"{S1_TOKEN}"' in code and f'"{S2_TOKEN}"' in code
    assert f'"{S1_PREFIX}"' in code and f'"{S2_PREFIX}"' in code
    assert 'SUPPLEMENT_RUN_LOG_WITH_FOLDER = False' in code


def test_no_mtime_or_wildcard_run_log_selection(notebook):
    """The run log is addressed by exact name, never discovered."""
    code = _code(notebook)
    # The wildcard may only survive inside explanatory strings, never as a glob.
    assert '.glob(' not in code.split('def create_grid_from_bbox')[0]
    assert 'CLASSIFIER_RUN_LOG_GLOB' not in code
    assert '_latest_classifier_run_log' not in code
    # No mtime-ordered run-log choice: the only remaining st_mtime uses key the
    # raster/EE caches, which are keyed on file identity by design.
    selection = code.split('def create_grid_from_bbox')[0]
    assert 'st_mtime' not in selection
    assert 'CLASSIFIER_RUN_LOG_PATH' in selection


def test_directory_scan_fallback_is_gone(notebook):
    assert '_records_from_classifier_folder' not in _code(notebook)


def test_active_version_matches_the_classifier_notebook():
    """ACTIVE_CLASSIFIER_VERSION == CLASSIFIER_VERSION + the correction suffix."""
    source = _code(REPO / 'Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb')
    assert "CLASSIFIER_VERSION = 'route_b_s1_scc_spatialcv_proba_v4_whlev_temporal'" in source
    assert "CLASSIFIER_VERSION = f'{CLASSIFIER_VERSION}_corr_{MANUAL_CORRECTION_VERSION}'" in source
    assert "MANUAL_CORRECTION_VERSION = 'v1'" in source


def test_export_tokens_match_batch_export():
    source = _code(REPO / 'Batch_Export.ipynb')
    assert f'S1_EXPORT_SCHEMA_VERSION = "{S1_TOKEN}"' in source
    assert f'S2_EXPORT_SCHEMA_VERSION = "{S2_TOKEN}"' in source
    # The prefixes the panel requires are the ones Batch_Export.ipynb builds.
    assert 'f"winam_{S1_EXPORT_SCHEMA_VERSION}_{r[\'start_date\']}' in source
    assert 'f"winam_s2_predictors_{S2_EXPORT_SCHEMA_VERSION}_{r[\'start_date\']}' in source


# ---------------------------------------------------------------------------
# Run-log addressing
# ---------------------------------------------------------------------------

def test_missing_run_log_raises_instead_of_scanning(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    # A classified raster IS on disk, and a run log for a DIFFERENT run exists.
    _touch(ns['_tif_dir'], _s2_names()[0])
    _write_run_log(ns, [], name='winam_full_stack_run_log_some_other_run.csv')
    with pytest.raises(FileNotFoundError, match='active classifier run'):
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')


def test_newer_run_log_of_another_version_is_ignored(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    model, _, _ = _s2_names()
    _touch(ns['_tif_dir'], model)
    _write_run_log(ns, [_row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / model)])

    other_model, _, _ = _s2_names(start='2022-06-01', end='2022-06-02')
    _touch(ns['_tif_dir'], other_model)
    newer = _write_run_log(
        ns,
        [_row('S2', '2022-06-01', '2022-06-02', ns['_tif_dir'] / other_model)],
        name='winam_full_stack_run_log_route_b_v5_newer.csv',
    )
    # Make the wrong log unambiguously the most recently modified one.
    import os
    os.utime(newer, (10 ** 10, 10 ** 10))

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert list(out['start_date']) == ['2021-06-01']
    assert set(out['source_run_log']) == {str(ns['CLASSIFIER_RUN_LOG_PATH'])}


def test_supplementing_from_the_directory_is_refused(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path, supplement=True)
    with pytest.raises(ValueError, match='SUPPLEMENT_RUN_LOG_WITH_FOLDER'):
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')


def test_run_log_records_missing_from_disk_raise(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    model, _, _ = _s2_names()
    _write_run_log(ns, [_row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / model)])
    with pytest.raises(FileNotFoundError, match='missing from disk'):
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')


# ---------------------------------------------------------------------------
# Product and provenance filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name, expected', [
    (f'{S2_PREFIX}2021-06-01_to_2021-06-02_local_random_forest.tif', 'model'),
    (f'{S2_PREFIX}2021-06-01_to_2021-06-02_local_random_forest_proba.tif', 'probability'),
    (f'{S2_PREFIX}2021-06-01_to_2021-06-02_local_rules.tif', 'rules'),
    (f'{S2_PREFIX}2021-06-01_to_2021-06-02_local_random_forest_tile_003.tif', 'tile_intermediate'),
    (f'{S1_PREFIX}2021-06-03_to_2021-06-04_local_random_forest_patch_cleaned.tif', 'model'),
    (f'{S1_PREFIX}2021-06-03_to_2021-06-04_local_random_forest_proba_tile_000.tif',
     'tile_intermediate'),
    (f'{S2_PREFIX}2021-06-01_to_2021-06-02_quicklook.tif', 'unknown'),
])
def test_product_classification(notebook, tmp_path, name, expected):
    ns = _load_selection(notebook, tmp_path)
    assert ns['_classifier_product_from_path'](name) == expected


@pytest.mark.parametrize('name', [
    # Superseded S1 export prefix (winam_s1_scc_predictors_...).
    'winam_s1_scc_predictors_2021-06-03_to_2021-06-04_local_random_forest.tif',
    # Superseded S2 export schema token.
    'winam_s2_predictors_s2_whlev_texture_v1_2021-06-01_to_2021-06-02_local_random_forest.tif',
    # Token present, but not as the prefix -- loose substring matching would pass this.
    f'reexport_{S2_TOKEN}_2021-06-01_to_2021-06-02_local_random_forest.tif',
])
def test_superseded_and_substring_only_prefixes_are_rejected(notebook, tmp_path, name):
    ns = _load_selection(notebook, tmp_path)
    assert ns['_classifier_sensor_from_path'](name) is None

    _touch(ns['_tif_dir'], name)
    sensor = 'S1' if 's1' in name else 'S2'
    _write_run_log(ns, [_row(sensor, '2021-06-01', '2021-06-02', ns['_tif_dir'] / name)])
    with pytest.raises(FileNotFoundError, match='No classified GeoTIFFs'):
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')


def test_probability_and_rule_rasters_never_become_the_response(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    model, proba, rules = _s2_names()
    for name in (model, proba, rules):
        _touch(ns['_tif_dir'], name)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / model,
             ns['_tif_dir'] / proba, ns['_tif_dir'] / rules),
    ])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert len(out) == 1
    assert Path(out.loc[0, 'path']).name == model
    assert set(out['product']) == {'model'}
    # The probability raster is carried as a SEPARATE confidence field.
    assert Path(out.loc[0, 'proba_path']).name == proba


def test_incomplete_and_failed_records_are_excluded(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    model, _, _ = _s2_names()
    _touch(ns['_tif_dir'], model)
    failed, _, _ = _s2_names(start='2021-07-01', end='2021-07-02')
    _touch(ns['_tif_dir'], failed)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / model),
        _row('S2', '2021-07-01', '2021-07-02', ns['_tif_dir'] / failed, status='failed'),
        _row('S2', '2021-08-01', '2021-08-02', '', status='completed'),
    ])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert list(out['start_date']) == ['2021-06-01']
    reasons = ns['CLASSIFIED_SELECTION_AUDIT']['excluded_reasons']
    assert reasons['status_failed'] == 1
    assert reasons['completed_row_without_model_classification_path'] == 1


def test_export_token_alone_is_not_proof_of_the_classifier_version(notebook, tmp_path):
    """A correctly-tokened raster listed under a different run log is not selected."""
    ns = _load_selection(notebook, tmp_path)
    model, _, _ = _s2_names()
    _touch(ns['_tif_dir'], model)
    _write_run_log(
        ns, [_row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / model)],
        name='winam_full_stack_run_log_route_b_s1_scc_spatialcv_proba_v4_whlev_temporal.csv',
    )
    with pytest.raises(FileNotFoundError, match='active classifier run'):
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')


# ---------------------------------------------------------------------------
# One canonical raster per sensor and acquisition
# ---------------------------------------------------------------------------

def test_patch_cleaned_s1_raster_wins_over_its_raw_input(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    raw, cleaned, _ = _s1_names()
    _touch(ns['_tif_dir'], raw)
    _touch(ns['_tif_dir'], cleaned)
    _write_run_log(ns, [
        _row('S1', '2021-06-03', '2021-06-04', ns['_tif_dir'] / raw),
        _row('S1', '2021-06-03', '2021-06-04', ns['_tif_dir'] / cleaned),
    ])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert len(out) == 1
    assert Path(out.loc[0, 'path']).name == cleaned
    assert bool(out.loc[0, 'is_patch_cleaned']) is True


def test_competing_outputs_for_one_acquisition_raise_and_list_the_files(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    a, _, _ = _s2_names(slug='random_forest')
    b, _, _ = _s2_names(slug='hist_gradient_boosting')
    _touch(ns['_tif_dir'], a)
    _touch(ns['_tif_dir'], b)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / a),
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / b),
    ])

    with pytest.raises(ValueError) as excinfo:
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    message = str(excinfo.value)
    assert 'ambiguous' in message
    assert a in message and b in message


def test_several_genuine_acquisitions_in_one_month_are_kept(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    first, _, _ = _s2_names(start='2021-06-01', end='2021-06-02')
    second, _, _ = _s2_names(start='2021-06-11', end='2021-06-12')
    _touch(ns['_tif_dir'], first)
    _touch(ns['_tif_dir'], second)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / first),
        _row('S2', '2021-06-11', '2021-06-12', ns['_tif_dir'] / second),
    ])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert len(out) == 2
    assert out['month'].nunique() == 1
    audit = ns['CLASSIFIED_SELECTION_AUDIT']
    assert audit['months_with_multiple_acquisitions'] == {'S2:2021-06': 2}
    assert audit['duplicate_sensor_acquisitions'] == []


def test_duplicated_identical_run_log_rows_collapse(notebook, tmp_path):
    """Incremental run logs append a row per attempt; identical paths are one raster."""
    ns = _load_selection(notebook, tmp_path)
    model, _, _ = _s2_names()
    _touch(ns['_tif_dir'], model)
    row = _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / model)
    _write_run_log(ns, [row, dict(row)])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Audit and panel-side validation
# ---------------------------------------------------------------------------

def test_audit_reports_versions_tokens_counts_and_coverage(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    s2, s2_proba, s2_rules = _s2_names()
    s1_raw, s1_cleaned, _ = _s1_names()
    for name in (s2, s2_proba, s2_rules, s1_raw, s1_cleaned):
        _touch(ns['_tif_dir'], name)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / s2,
             ns['_tif_dir'] / s2_proba, ns['_tif_dir'] / s2_rules),
        _row('S1', '2021-06-03', '2021-06-04', ns['_tif_dir'] / s1_raw),
        _row('S1', '2021-06-03', '2021-06-04', ns['_tif_dir'] / s1_cleaned),
    ])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2021-01-01', '2021-12-31')
    audit = ns['CLASSIFIED_SELECTION_AUDIT']

    assert audit['classifier_version'] == CLASSIFIER_VERSION
    assert audit['classifier_run_log_name'] == \
        f'winam_full_stack_run_log_{CLASSIFIER_VERSION}.csv'
    assert audit['required_export_token_by_sensor'] == {'S1': S1_TOKEN, 'S2': S2_TOKEN}
    assert audit['required_classified_prefix_by_sensor'] == {'S1': S1_PREFIX, 'S2': S2_PREFIX}
    assert audit['n_from_directory_supplementation'] == 0
    assert audit['supplement_run_log_with_folder'] is False
    assert audit['date_coverage']['first_month'] == '2021-06'
    assert len(audit['date_coverage']['months_without_any_raster']) == 11

    expected_sensors = set(out['sensor'])
    assert audit['n_selected_by_sensor'] == {
        s: int((out['sensor'] == s).sum()) for s in expected_sensors
    }
    for sensor in expected_sensors:
        assert audit['by_sensor'][sensor]['required_export_token'] == \
            {'S1': S1_TOKEN, 'S2': S2_TOKEN}[sensor]


def test_sensor_filter_is_honoured(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path, sensor_filter=('S2',))
    s2, _, _ = _s2_names()
    s1, _, _ = _s1_names()
    _touch(ns['_tif_dir'], s2)
    _touch(ns['_tif_dir'], s1)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', ns['_tif_dir'] / s2),
        _row('S1', '2021-06-03', '2021-06-04', ns['_tif_dir'] / s1),
    ])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')
    assert set(out['sensor']) == {'S2'}
    assert ns['CLASSIFIED_SELECTION_AUDIT']['excluded_reasons']['sensor_not_requested'] == 1


def test_rule_product_filter_is_refused(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path, product_filter='all')
    with pytest.raises(ValueError, match='CLASSIFIER_PRODUCT_FILTER'):
        ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')


def _panel(ns, **overrides):
    base = {
        'sensor': ['S2', 'S2'],
        'source_classifier_version': [CLASSIFIER_VERSION] * 2,
        'source_export_token': [S2_TOKEN] * 2,
        'source_run_log': [str(ns['CLASSIFIER_RUN_LOG_PATH'])] * 2,
        'source_from_directory_scan': [False, False],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_panel_provenance_validation_accepts_a_clean_panel(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    assert ns['validate_panel_provenance'](_panel(ns)) is True


@pytest.mark.parametrize('overrides, match', [
    ({'source_classifier_version': [CLASSIFIER_VERSION, 'route_b_v3_old']}, 'classifier version'),
    ({'source_export_token': [S2_TOKEN, 's2_whlev_texture_v1']}, 'export-token'),
    ({'source_from_directory_scan': [False, True]}, 'scanning'),
])
def test_panel_provenance_validation_rejects_mixed_panels(notebook, tmp_path, overrides, match):
    ns = _load_selection(notebook, tmp_path)
    with pytest.raises(ValueError, match=match):
        ns['validate_panel_provenance'](_panel(ns, **overrides))


def test_panel_provenance_validation_rejects_a_pre_provenance_panel(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    old = pd.DataFrame({'sensor': ['S2'], 'wh_cover': [0.1]})
    with pytest.raises(ValueError, match='missing provenance column'):
        ns['validate_panel_provenance'](old)


def test_stamping_then_validating_round_trips(notebook, tmp_path):
    """Section 7 stamps exactly the columns validate_panel_provenance requires.

    Regression: the stamping loop prefixed every metadata column with 'source_',
    so tif_index's already-prefixed 'source_run_log' landed as
    'source_source_run_log' and a freshly built panel failed its own validation.
    """
    ns = _load_selection(notebook, tmp_path)
    model, proba, _ = _s2_names()
    _touch(ns['_tif_dir'], model)
    _touch(ns['_tif_dir'], proba)
    _write_run_log(ns, [_row('S2', '2021-06-01', '2021-06-02',
                             ns['_tif_dir'] / model, ns['_tif_dir'] / proba)])

    out = ns['find_classified_tifs'](ns['_tif_dir'], '2017-01-01', '2026-12-31')

    # Mirror the Section 7 loop: reduce one raster, stamp it, validate.
    panel_parts = []
    for row in out.itertuples(index=False):
        df = pd.DataFrame({'grid_id': [1, 2], 'wh_cover': [0.1, 0.2]})
        row_dict = row._asdict()
        ns['stamp_source_metadata'](df, row_dict, ns['PANEL_SOURCE_METADATA_COLUMNS'])
        df['sensor'] = ns['_normalise_classifier_sensor'](row_dict.get('sensor'))
        panel_parts.append(df)
    panel_raw = pd.concat(panel_parts, ignore_index=True)

    assert 'source_run_log' in panel_raw.columns
    assert 'source_source_run_log' not in panel_raw.columns
    assert ns['validate_panel_provenance'](panel_raw, context='built panel') is True


def test_source_column_naming_does_not_double_prefix(notebook, tmp_path):
    ns = _load_selection(notebook, tmp_path)
    assert ns['panel_source_column']('sensor') == 'source_sensor'
    assert ns['panel_source_column']('source_run_log') == 'source_run_log'
    # Every configured metadata column must map to a distinct panel column.
    mapped = [ns['panel_source_column'](c) for c in ns['PANEL_SOURCE_METADATA_COLUMNS']]
    assert len(mapped) == len(set(mapped))


def test_panel_build_stamps_the_provenance_columns(notebook):
    """Section 7 must carry classifier version and export token onto every row."""
    cells = [''.join(c['source']) for c in json.loads(notebook.read_text())['cells']]
    build = [s for s in cells if '    metadata_cols = PANEL_SOURCE_METADATA_COLUMNS' in s]
    assert len(build) == 1
    assert 'stamp_source_metadata(df, row_dict, metadata_cols)' in build[0]
    assert 'validate_panel_provenance' in build[0]
    # The old blind-prefix loop is gone.
    assert 'df[f"source_{col}"]' not in build[0]

    checkpoint = [s for s in cells if 'def _panel_raw_fingerprint' in s]
    assert len(checkpoint) == 1
    assert 'validate_panel_provenance' in checkpoint[0]
    assert 'ACTIVE_CLASSIFIER_VERSION' in checkpoint[0]
