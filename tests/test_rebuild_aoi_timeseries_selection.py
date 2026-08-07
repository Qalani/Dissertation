"""Tests for which classified GeoTIFFs Rebuild_AOI_WH_TimeSeries.ipynb ingests.

``classified_geotiffs/`` accumulates every classifier run and every batch-export
schema ever written into it, so the AOI series must ingest only the CURRENT
version of each raster. Each way a superseded raster used to slip in is pinned
here:

1. a plain ``*.tif`` glob, which accepted legacy export schemas
   (``s2_whlev_texture_v1``, ``winam_s1_scc_predictors_...``) alongside current
   ones;
2. loose ``winam_s2_`` / ``winam_s1_`` prefix matching, which let a re-export that
   merely mentions a schema token pass;
3. ``drop_duplicates`` resolving two competing rasters by sort order rather than
   by which one is final;
4. two model slugs -- two classifier configurations -- landing in one series;
5. Google Drive collision copies (``foo (1).tif``) and pre-mosaic tiles being
   read as final classifications.

The functions under test live in the notebook, so they are extracted from the
notebook JSON by name: the tests then track the real notebook source rather than
a copy that can drift away from it. No Drive, no rasterio, no network.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / 'Rebuild_AOI_WH_TimeSeries.ipynb'

CLASSIFIER_VERSION = 'route_b_s1_scc_spatialcv_proba_v4_whlev_temporal_corr_v1'
S1_PREFIX = 'winam_s1_scc_temporal_v1_'
S2_PREFIX = 'winam_s2_predictors_s2_whlev_temporal_v1_'
S1_TOKEN = 's1_scc_temporal_v1'
S2_TOKEN = 's2_whlev_temporal_v1'
RUN_LOG_NAME = f'winam_full_stack_run_log_{CLASSIFIER_VERSION}.csv'

RUN_LOG_COLUMNS = [
    'sensor', 'start_date', 'end_date', 'prefix', 'status',
    'model_classification_tif', 'model_probability_tif', 'rule_classification_tif',
]

# Every top-level name the selection block defines.
WANTED = {
    'DATE_RE', '_PROBABILITY_SUFFIX', '_RULE_SUFFIX', '_PATCH_CLEANED_SUFFIX',
    '_MODEL_SEGMENT', '_TILE_INTERMEDIATE_RE', '_DRIVE_COLLISION_RE',
    '_GEOTIFF_SUFFIXES', 'SELECTION_AUDIT', 'blank_value', 'normalise_sensor',
    'check_prefix_token_consistency', 'sensor_from_path', 'product_from_path',
    'model_slug_from_path', 'patch_clean_base_stem', 'describe_classified_raster',
    'read_active_run_log', '_run_log_completed_rows', 'run_log_current_file_names',
    'run_log_files_missing_from_folder', 'apply_model_slug_filter',
    'resolve_canonical_rasters', 'select_classified_rasters',
}


def _cells(notebook):
    return json.loads(notebook.read_text())['cells']


def _code(notebook):
    """Concatenated source of the notebook's code cells."""
    return '\n'.join(
        ''.join(c['source']) for c in _cells(notebook) if c['cell_type'] == 'code'
    )


def _extract_selection_source(notebook):
    """Source text of the selection block's top-level defs/assignments."""
    found = {}
    for cell in _cells(notebook):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'def select_classified_rasters' not in source:
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
    return '\n\n'.join(found.values())


def _load_selection(tmp_path, run_log_mode='if_present', include_rules=True,
                    prefer_patch_cleaned=True, model_slug_by_sensor=None,
                    classifier_version=CLASSIFIER_VERSION):
    """Exec the notebook's selection block against a temporary Drive layout."""
    table_dir = tmp_path / 'tables'
    tif_dir = tmp_path / 'classified_geotiffs'
    table_dir.mkdir(exist_ok=True)
    tif_dir.mkdir(exist_ok=True)

    ns = {
        'pd': pd, 'Path': Path, 're': re,
        'ACTIVE_CLASSIFIER_VERSION': classifier_version,
        'CLASSIFIER_RUN_LOG_PATH':
            table_dir / f'winam_full_stack_run_log_{classifier_version}.csv',
        'RUN_LOG_MODE': run_log_mode,
        'RUN_LOG_COMPLETED_STATUSES': ('completed', 'completed_prior_run'),
        'REQUIRED_EXPORT_TOKEN_BY_SENSOR': {'S1': S1_TOKEN, 'S2': S2_TOKEN},
        'REQUIRED_CLASSIFIED_PREFIX_BY_SENSOR': {'S1': S1_PREFIX, 'S2': S2_PREFIX},
        'MODEL_SLUG_BY_SENSOR': model_slug_by_sensor or {'S1': None, 'S2': None},
        'INCLUDE_RULE_RASTERS': include_rules,
        'PREFER_PATCH_CLEANED_S1': prefer_patch_cleaned,
    }
    exec(_extract_selection_source(NOTEBOOK), ns)
    ns['_table_dir'] = table_dir
    ns['_tif_dir'] = tif_dir
    return ns


def _touch(directory, name):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'II*\x00')  # enough to exist; nothing reads the pixels here
    return path


def _row(sensor, start, end, prefix, model_tif='', rule_tif='', status='completed'):
    return {
        'sensor': sensor, 'start_date': start, 'end_date': end,
        'prefix': prefix, 'status': status,
        'model_classification_tif': str(model_tif),
        'model_probability_tif': '',
        'rule_classification_tif': str(rule_tif),
    }


def _write_run_log(ns, rows, name=None):
    path = ns['CLASSIFIER_RUN_LOG_PATH'] if name is None else ns['_table_dir'] / name
    pd.DataFrame(rows, columns=RUN_LOG_COLUMNS).to_csv(path, index=False)
    return path


def _s2_stem(start='2021-06-01', end='2021-06-02'):
    return f'{S2_PREFIX}{start}_to_{end}'


def _s1_stem(start='2021-06-03', end='2021-06-04'):
    return f'{S1_PREFIX}{start}_to_{end}'


def _select(ns):
    return ns['select_classified_rasters'](ns['_tif_dir'])


# ---------------------------------------------------------------------------
# The pinned run and schema match the notebooks that produce the rasters
# ---------------------------------------------------------------------------

def test_notebook_pins_the_active_run_and_current_export_schema():
    code = _code(NOTEBOOK)
    assert f'ACTIVE_CLASSIFIER_VERSION = "{CLASSIFIER_VERSION}"' in code
    assert f'"{S1_TOKEN}"' in code and f'"{S2_TOKEN}"' in code
    assert f'"{S1_PREFIX}"' in code and f'"{S2_PREFIX}"' in code


def test_the_pinned_values_are_the_ones_the_panel_notebooks_use():
    """One definition of 'current' across the AOI series and the spatial panels."""
    for panel in sorted(REPO.glob('winam_wh_spatial_panel*.ipynb')):
        panel_code = _code(panel)
        assert f'"{CLASSIFIER_VERSION}"' in panel_code
        assert f'"{S1_PREFIX}"' in panel_code and f'"{S2_PREFIX}"' in panel_code


def test_active_version_matches_the_classifier_notebook():
    source = _code(REPO / 'Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb')
    assert "CLASSIFIER_VERSION = 'route_b_s1_scc_spatialcv_proba_v4_whlev_temporal'" in source
    assert "CLASSIFIER_VERSION = f'{CLASSIFIER_VERSION}_corr_{MANUAL_CORRECTION_VERSION}'" in source
    assert "MANUAL_CORRECTION_VERSION = 'v1'" in source


def test_the_permissive_prefix_scan_is_gone():
    """The old gate accepted any winam_s1_/winam_s2_ file, schema regardless."""
    code = _code(NOTEBOOK)
    assert 'startswith("winam_s2_")' not in code
    assert 'startswith("winam_s1_")' not in code
    assert 'drop_duplicates' not in code
    assert 'REQUIRED_CLASSIFIED_PREFIX_BY_SENSOR' in code


def test_run_log_is_addressed_by_name_never_by_mtime_or_wildcard():
    """The wildcard may survive in a comment, never as a glob or an mtime sort."""
    code = _code(NOTEBOOK)
    assert 'st_mtime' not in code
    assert '.glob(' not in code
    assert 'f"winam_full_stack_run_log_{ACTIVE_CLASSIFIER_VERSION}.csv"' in code


def test_prefix_and_token_configuration_must_agree(tmp_path):
    ns = _load_selection(tmp_path)
    ns['REQUIRED_EXPORT_TOKEN_BY_SENSOR']['S2'] = 's2_whlev_texture_v1'
    with pytest.raises(ValueError, match='not a token of the required prefix'):
        ns['check_prefix_token_consistency']()


# ---------------------------------------------------------------------------
# Filename provenance gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', [
    # Superseded S1 export prefix.
    'winam_s1_scc_predictors_2021-06-03_to_2021-06-04_local_random_forest.tif',
    # Superseded S2 export schema token.
    'winam_s2_predictors_s2_whlev_texture_v1_2021-06-01_to_2021-06-02_local_random_forest.tif',
    # Legacy S2 export with no schema token at all.
    'winam_s2_predictors_2021-06-01_to_2021-06-02_local_random_forest.tif',
    # Token present, but not as the prefix: a substring search would pass this.
    f'reexport_{S2_TOKEN}_2021-06-01_to_2021-06-02_local_random_forest.tif',
])
def test_superseded_and_substring_only_prefixes_are_rejected(tmp_path, name):
    ns = _load_selection(tmp_path, run_log_mode='off')
    assert ns['sensor_from_path'](name) is None

    _touch(ns['_tif_dir'], name)
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_random_forest.tif')

    out = _select(ns)
    assert list(out['file_name']) == [f'{_s2_stem()}_local_random_forest.tif']
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {
        'prefix_is_not_the_current_export_schema': 1
    }


@pytest.mark.parametrize('suffix, expected', [
    ('_local_random_forest', 'model'),
    ('_local_random_forest_proba', 'probability'),
    ('_local_rules', 'rules'),
    ('_local_random_forest_tile_003', 'tile_intermediate'),
    ('_local_random_forest_patch_cleaned', 'model'),
    ('_local_random_forest_proba_tile_000', 'tile_intermediate'),
    ('_quicklook', 'unknown'),
])
def test_product_classification(tmp_path, suffix, expected):
    ns = _load_selection(tmp_path)
    assert ns['product_from_path'](f'{_s2_stem()}{suffix}.tif') == expected


def test_probability_tile_and_quicklook_rasters_never_enter_the_series(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    stem = _s2_stem()
    for name in (
        f'{stem}_local_random_forest.tif',
        f'{stem}_local_random_forest_proba.tif',
        f'{stem}_local_random_forest_tile_007.tif',
        f'{stem}_quicklook.tif',
        f'{stem}_local_random_forest.png',
    ):
        _touch(ns['_tif_dir'], name)

    out = _select(ns)
    assert list(out['file_name']) == [f'{stem}_local_random_forest.tif']
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {
        'not_a_final_classification_probability': 1,
        'not_a_final_classification_tile_intermediate': 1,
        'not_a_final_classification_unknown': 1,
        'not_a_geotiff': 1,
    }


def test_drive_collision_copies_are_rejected(tmp_path):
    """A re-downloaded 'foo (1).tif' is an older copy, not a current product."""
    ns = _load_selection(tmp_path, run_log_mode='off')
    stem = _s2_stem()
    _touch(ns['_tif_dir'], f'{stem}_local_random_forest.tif')
    _touch(ns['_tif_dir'], f'{stem}_local_random_forest (1).tif')

    out = _select(ns)
    assert list(out['file_name']) == [f'{stem}_local_random_forest.tif']
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {'drive_collision_copy': 1}


def test_rule_rasters_are_kept_as_their_own_product(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    stem = _s2_stem()
    _touch(ns['_tif_dir'], f'{stem}_local_random_forest.tif')
    _touch(ns['_tif_dir'], f'{stem}_local_rules.tif')

    out = _select(ns)
    assert sorted(out['kind']) == ['model', 'rules']
    assert sorted(out['method']) == ['Paper_rules', 'Route_B_random_forest']
    assert list(out.loc[out['kind'].eq('rules'), 'model_slug']) == ['']


def test_rule_rasters_can_be_switched_off(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off', include_rules=False)
    stem = _s2_stem()
    _touch(ns['_tif_dir'], f'{stem}_local_random_forest.tif')
    _touch(ns['_tif_dir'], f'{stem}_local_rules.tif')

    out = _select(ns)
    assert list(out['kind']) == ['model']
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {
        'rule_raster_excluded_by_INCLUDE_RULE_RASTERS': 1
    }


def test_s1_methods_and_dates_are_parsed(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    stem = _s1_stem()
    _touch(ns['_tif_dir'], f'{stem}_local_random_forest.tif')
    _touch(ns['_tif_dir'], f'{stem}_local_rules.tif')

    out = _select(ns).sort_values('kind').reset_index(drop=True)
    assert list(out['sensor']) == ['S1', 'S1']
    assert list(out['method']) == ['Route_B_SCC_random_forest', 'Paper_rules_SCC']
    assert set(out['start_date']) == {'2021-06-03'}
    assert set(out['end_date']) == {'2021-06-04'}
    assert set(out['export_token']) == {S1_TOKEN}
    assert set(out['classifier_version']) == {CLASSIFIER_VERSION}


# ---------------------------------------------------------------------------
# One canonical raster per sensor, product and acquisition
# ---------------------------------------------------------------------------

def test_patch_cleaned_s1_raster_supersedes_its_raw_input(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    stem = f'{_s1_stem()}_local_random_forest'
    _touch(ns['_tif_dir'], f'{stem}.tif')
    _touch(ns['_tif_dir'], f'{stem}_patch_cleaned.tif')

    out = _select(ns)
    assert list(out['file_name']) == [f'{stem}_patch_cleaned.tif']
    assert bool(out.loc[0, 'patch_cleaned']) is True
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {
        'superseded_by_patch_cleaned_raster': 1
    }


def test_patch_cleaned_preference_can_be_reversed(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off', prefer_patch_cleaned=False)
    stem = f'{_s1_stem()}_local_random_forest'
    _touch(ns['_tif_dir'], f'{stem}.tif')
    _touch(ns['_tif_dir'], f'{stem}_patch_cleaned.tif')

    out = _select(ns)
    assert list(out['file_name']) == [f'{stem}.tif']


def test_two_model_slugs_raise_instead_of_being_averaged(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_random_forest.tif')
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_hist_gradient_boosting.tif')

    with pytest.raises(ValueError) as excinfo:
        _select(ns)
    message = str(excinfo.value)
    assert 'more than one model slug' in message.lower()
    assert 'random_forest' in message and 'hist_gradient_boosting' in message


def test_pinning_the_model_slug_drops_the_superseded_one(tmp_path):
    ns = _load_selection(
        tmp_path, run_log_mode='off',
        model_slug_by_sensor={'S1': None, 'S2': 'random_forest'},
    )
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_random_forest.tif')
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_hist_gradient_boosting.tif')

    out = _select(ns)
    assert list(out['file_name']) == [f'{_s2_stem()}_local_random_forest.tif']
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {
        'model_slug_is_not_the_selected_random_forest': 1
    }
    assert ns['SELECTION_AUDIT']['model_slug_by_sensor'] == {'S2': 'random_forest'}


def test_several_genuine_acquisitions_are_all_kept(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    for start, end in (('2021-06-01', '2021-06-02'), ('2021-06-11', '2021-06-12')):
        _touch(ns['_tif_dir'], f'{_s2_stem(start, end)}_local_random_forest.tif')

    out = _select(ns)
    assert list(out['start_date']) == ['2021-06-01', '2021-06-11']


# ---------------------------------------------------------------------------
# Run-log intersection
# ---------------------------------------------------------------------------

def test_rasters_outside_the_active_run_log_are_dropped(tmp_path):
    """Same current export schema, but written by a run the log does not cover."""
    ns = _load_selection(tmp_path)
    current = f'{_s2_stem()}_local_random_forest.tif'
    stale = f'{_s2_stem("2019-05-01", "2019-05-02")}_local_hist_gradient_boosting.tif'
    _touch(ns['_tif_dir'], current)
    _touch(ns['_tif_dir'], stale)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', _s2_stem(),
             ns['_tif_dir'] / current),
    ])

    out = _select(ns)
    assert list(out['file_name']) == [current]
    assert ns['SELECTION_AUDIT']['selection_mode'] == 'run_log_intersection'
    assert ns['SELECTION_AUDIT']['excluded_reasons'] == {
        'not_listed_in_the_active_classifier_run_log': 1
    }
    assert out.loc[0, 'source_run_log'] == str(ns['CLASSIFIER_RUN_LOG_PATH'])


def test_failed_and_started_run_log_rows_do_not_admit_their_rasters(tmp_path):
    ns = _load_selection(tmp_path)
    failed = f'{_s2_stem("2021-07-01", "2021-07-02")}_local_random_forest.tif'
    ok = f'{_s2_stem()}_local_random_forest.tif'
    _touch(ns['_tif_dir'], failed)
    _touch(ns['_tif_dir'], ok)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', _s2_stem(), ns['_tif_dir'] / ok),
        _row('S2', '2021-07-01', '2021-07-02', _s2_stem('2021-07-01', '2021-07-02'),
             ns['_tif_dir'] / failed, status='failed'),
    ])

    out = _select(ns)
    assert list(out['file_name']) == [ok]


def test_a_resumed_dataset_keeps_its_rule_raster(tmp_path):
    """'completed_prior_run' rows log no rule raster; the name is derived instead."""
    ns = _load_selection(tmp_path)
    stem = _s2_stem()
    model = f'{stem}_local_random_forest.tif'
    rules = f'{stem}_local_rules.tif'
    _touch(ns['_tif_dir'], model)
    _touch(ns['_tif_dir'], rules)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', stem, ns['_tif_dir'] / model,
             status='completed_prior_run'),
    ])

    out = _select(ns)
    assert sorted(out['file_name']) == sorted([model, rules])


def test_run_log_of_another_classifier_version_is_never_substituted(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='require')
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_random_forest.tif')
    _write_run_log(
        ns, [], name='winam_full_stack_run_log_route_b_s1_scc_spatialcv_proba_v4_whlev_temporal.csv'
    )

    with pytest.raises(FileNotFoundError, match='active classifier run'):
        _select(ns)


def test_if_present_falls_back_to_the_filename_gate(tmp_path, capsys):
    ns = _load_selection(tmp_path, run_log_mode='if_present')
    _touch(ns['_tif_dir'], f'{_s2_stem()}_local_random_forest.tif')

    out = _select(ns)
    assert len(out) == 1
    assert ns['SELECTION_AUDIT']['selection_mode'] == 'filename_gate'
    assert ns['SELECTION_AUDIT']['run_log'] == ''
    assert list(out['source_run_log']) == ['']
    assert 'WARNING' in capsys.readouterr().out


def test_run_log_path_must_name_the_active_version(tmp_path):
    ns = _load_selection(tmp_path)
    ns['CLASSIFIER_RUN_LOG_PATH'] = ns['_table_dir'] / 'winam_full_stack_run_log_v3.csv'
    with pytest.raises(ValueError, match='must name the active classifier run'):
        ns['read_active_run_log']()


def test_recorded_rasters_missing_from_disk_are_reported(tmp_path):
    ns = _load_selection(tmp_path)
    present = f'{_s2_stem()}_local_random_forest.tif'
    absent = f'{_s2_stem("2021-07-01", "2021-07-02")}_local_random_forest.tif'
    _touch(ns['_tif_dir'], present)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', _s2_stem(), ns['_tif_dir'] / present),
        _row('S2', '2021-07-01', '2021-07-02', _s2_stem('2021-07-01', '2021-07-02'),
             ns['_tif_dir'] / absent),
    ])

    out = _select(ns)
    assert list(out['file_name']) == [present]
    assert ns['SELECTION_AUDIT']['run_log_files_missing_from_folder'] == [absent]


# ---------------------------------------------------------------------------
# Audit and empty-selection behaviour
# ---------------------------------------------------------------------------

def test_audit_reports_provenance_counts_and_reasons(tmp_path):
    ns = _load_selection(tmp_path)
    s2_stem, s1_stem = _s2_stem(), _s1_stem()
    s2_model = f'{s2_stem}_local_random_forest.tif'
    s2_rules = f'{s2_stem}_local_rules.tif'
    s1_model = f'{s1_stem}_local_random_forest_patch_cleaned.tif'
    legacy = 'winam_s2_predictors_s2_whlev_texture_v1_2019-01-01_to_2019-01-02_local_random_forest.tif'
    for name in (s2_model, s2_rules, s1_model, legacy):
        _touch(ns['_tif_dir'], name)
    _write_run_log(ns, [
        _row('S2', '2021-06-01', '2021-06-02', s2_stem,
             ns['_tif_dir'] / s2_model, ns['_tif_dir'] / s2_rules),
        _row('S1', '2021-06-03', '2021-06-04', s1_stem, ns['_tif_dir'] / s1_model),
    ])

    out = _select(ns)
    audit = ns['SELECTION_AUDIT']

    assert audit['classifier_version'] == CLASSIFIER_VERSION
    assert audit['run_log'] == str(ns['CLASSIFIER_RUN_LOG_PATH'])
    assert audit['run_log_mode'] == 'if_present'
    assert audit['required_export_token_by_sensor'] == {'S1': S1_TOKEN, 'S2': S2_TOKEN}
    assert audit['required_classified_prefix_by_sensor'] == {'S1': S1_PREFIX, 'S2': S2_PREFIX}
    assert audit['model_slug_by_sensor'] == {'S1': 'random_forest', 'S2': 'random_forest'}
    assert audit['n_files_scanned'] == 4
    assert audit['n_selected'] == len(out) == 3
    assert audit['n_selected_by_sensor_kind'] == {'S1:model': 1, 'S2:model': 1, 'S2:rules': 1}
    assert audit['excluded_reasons'] == {'prefix_is_not_the_current_export_schema': 1}


def test_nothing_current_raises_with_the_rejection_reasons(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    _touch(ns['_tif_dir'],
           'winam_s2_predictors_2019-01-01_to_2019-01-02_local_random_forest.tif')

    with pytest.raises(RuntimeError) as excinfo:
        _select(ns)
    message = str(excinfo.value)
    assert 'No current classified GeoTIFFs' in message
    assert 'prefix_is_not_the_current_export_schema' in message


def test_a_missing_classified_dir_says_which_setting_to_change(tmp_path):
    ns = _load_selection(tmp_path, run_log_mode='off')
    with pytest.raises(FileNotFoundError, match='CLASSIFIED_DIR'):
        ns['select_classified_rasters'](tmp_path / 'not_there')
