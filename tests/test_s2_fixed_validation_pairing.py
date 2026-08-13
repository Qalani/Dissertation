"""Tests for the paired S2 fixed-validation comparison (section 13h).

Section 13h of ``Rules_vs_Logistic_Regression_Comparison.ipynb`` is the notebook's
primary like-for-like accuracy comparison: the paper thresholds and the
expanded-data Logistic Regression model scored on *one identical set* of held-out
points. The split belongs to ``run_s2_fixed_validation_diagnostic()`` in
``Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb``, which exports a manifest;
section 13h consumes it and must never re-derive, re-split or re-fit anything.

That contract is only worth as much as the checks around it, so these tests pin:

1. the S2 point-space rule replay still reproduces the raster rule function, and
   section 13h reuses that one implementation rather than re-stating thresholds;
2. duplicate or missing validation ids fail loudly;
3. a missing or non-finite rule predictor fails loudly instead of deleting rows —
   deleting one would end the pairing;
4. both classifiers are scored on identical, identically ordered ids, with equal
   ``n_points`` and equal floating support;
5. the recomputed Logistic Regression row reproduces the classifier notebook's own
   fixed-validation result;
6. metric differences carry the documented ``rules_minus_lr`` sign convention;
7. no training-set or map-space Logistic Regression prediction can reach the
   comparison;
8. the manifest is written from the ``valid`` frame inside the diagnostic itself,
   and the producer's columns satisfy the consumer's loader end to end.

As in the sibling modules, the code under test is extracted from the notebook JSON
by name and exec'd, so the tests track the real notebooks rather than a copy that
can drift. No Drive, no Earth Engine, no network.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')

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

# Section 13h, plus the section 7/13a/13b names it builds on.
WANTED = {
    'SENSOR_CLASS_LABELS', 'FLOATING_CLASS_CODE', 'class_labels_for', 'class_codes_for',
    'ACCURACY_METRICS', 'S2_RULE_NDMI_ALGAE', 'S2_RULE_NDMI_FLOATING',
    's2_rule_class_at_points',
    'classification_metrics', 'per_class_metrics', 'confusion_long_form',
    'S2_FIXED_VALIDATION_SENSOR', 'CLASSIFIER_DIAGNOSTIC_DIR',
    'S2_FIXED_VALIDATION_MANIFEST_PATH', 'S2_FIXED_VALIDATION_METRICS_PATH',
    'S2_FIXED_VALIDATION_LR_TRAIN_ARM', 'S2_FIXED_VALIDATION_MODEL_NAME',
    'S2_FIXED_VALIDATION_ID_COLUMN', 'S2_FIXED_VALIDATION_RULE_PREDICTORS',
    'S2_FIXED_VALIDATION_REQUIRED_COLUMNS', 'S2_FIXED_VALIDATION_PROVENANCE_COLUMNS',
    'S2_FIXED_VALIDATION_EVALUATION', 'S2_FIXED_VALIDATION_LR_LABEL',
    'S2_FIXED_VALIDATION_RULES_LABEL', 'S2_FIXED_VALIDATION_CLASSIFIERS',
    'S2_FIXED_VALIDATION_DIFFERENCE_CONVENTION', 'S2_FIXED_VALIDATION_METRICS',
    'S2_FIXED_VALIDATION_COUNT_FIELDS', 'S2_FIXED_VALIDATION_METRIC_TOLERANCE',
    'S2_FIXED_VALIDATION_METRIC_CROSSCHECKS',
    'validate_fixed_validation_ids', 'validate_fixed_validation_rule_predictors',
    'load_s2_fixed_validation_manifest', 'load_s2_fixed_validation_metrics_row',
    's2_fixed_validation_predictions', 's2_fixed_validation_points_long_form',
    's2_fixed_validation_fingerprint', '_fingerprint_mismatch_detail',
    'assert_s2_fixed_validation_is_paired', 's2_fixed_validation_summary',
    'assert_lr_metrics_match_classifier_notebook',
    's2_fixed_validation_metric_differences',
}

# The producer side, in the classifier notebook.
CLASSIFIER_WANTED = {
    'classify_s2_rules_array', 'extract_lonlat_from_geo', 'S2_BAKED_SCHEMA_TOKEN',
    '_baked_scene_prefix', 'S2_FIXED_VALIDATION_MANIFEST_STEM',
    'S2_FIXED_VALIDATION_LR_TRAIN_ARM', 'S2_FIXED_VALIDATION_RULE_PREDICTORS',
    'S2_FIXED_VALIDATION_PROVENANCE_COLUMNS', '_s2_fixed_validation_manifest',
    'S2_DIAGNOSTIC_VALIDATION_FRACTION', 'S2_DIAGNOSTIC_RANDOM_STATE',
}

S2_CLASS_CODES = (0, 1, 2, 3)


def _cells(notebook):
    return json.loads(notebook.read_text())['cells']


def _extract(notebook, wanted):
    """Source text of the notebook's top-level defs/assignments named in ``wanted``.

    Discovery order is notebook order, so exec'ing the joined text reproduces the
    notebook's own definition order and its dependencies resolve.
    """
    found = {}
    for cell in _cells(notebook):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - magics; checked elsewhere
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


def _namespace(tmp_path=None):
    """Exec the comparison notebook's section 13h helpers and their dependencies."""
    ns = {
        'np': np, 'pd': pd, 'Path': Path, 'json': json,
        'accuracy_score': accuracy_score,
        'balanced_accuracy_score': balanced_accuracy_score,
        'cohen_kappa_score': cohen_kappa_score,
        'confusion_matrix': confusion_matrix,
        'f1_score': f1_score,
        'precision_recall_fscore_support': precision_recall_fscore_support,
        'NODATA_VALUE': NODATA,
        'SENSOR_FILTER': ['S2', 'S1'],
        # Only used to build the default artifact paths; every test passes explicit ones.
        'CLASSIFIER_ROOT': tmp_path or Path('.'),
    }
    exec(_extract(NOTEBOOK, WANTED), ns)
    return ns


def _classifier_namespace():
    """Exec the classifier notebook's rule function and manifest builder."""
    ns = {
        'np': np, 'pd': pd, 'json': json,
        'NODATA_VALUE': NODATA, 'FLOATING_CLASS_CODE': FLOATING,
        'CLASS_COLUMN': 'class',
        'SPATIAL_CV_RANDOM_STATE': 42,
        'S2_DIAGNOSTIC_SUFFIX': '_diagnostic_more_training_data_check',
        'S2_DIAGNOSTIC_MODEL_NAME': 'Logistic Regression baseline',
    }
    exec(_extract(CLASSIFIER_NOTEBOOK, CLASSIFIER_WANTED), ns)
    return ns


# ---------------------------------------------------------------------------
# Fixtures: a small but complete fixed-validation manifest.
# ---------------------------------------------------------------------------

def _geo(lon, lat):
    return json.dumps({'type': 'Point', 'coordinates': [float(lon), float(lat)]})


def make_manifest(n=24, seed=5, arm='corrected_full_excluding_validation'):
    """A manifest with the columns the classifier notebook exports.

    Rule predictors are drawn to straddle every threshold, so the replayed rules
    disagree with the Logistic Regression column often enough for the metric
    differences to be non-trivial.
    """
    rng = np.random.default_rng(seed)
    prefix = 'winam_s2_predictors_s2_whlev_temporal_v1_2021-02-10_to_2021-02-11'

    true_class = np.array([S2_CLASS_CODES[i % 4] for i in range(n)], dtype=np.int64)
    # Mostly-right Logistic Regression predictions, wrong on every 7th point.
    lr_pred = true_class.copy()
    lr_pred[::7] = (lr_pred[::7] + 1) % 4

    lon = 34.4 + rng.uniform(0, 0.5, n)
    lat = -0.45 + rng.uniform(0, 0.4, n)

    return pd.DataFrame({
        '_s2_diag_row_id': np.arange(n, dtype=np.int64),
        'correction_source': np.where(
            np.arange(n) % 3 == 0, 'original_accepted', 'manual_corrected_added'),
        'manual_correction': (np.arange(n) % 3 != 0).astype(int),
        'spatial_block': (np.arange(n) // 4).astype(np.int64),
        'true_class': true_class,
        'lr_predicted_class': lr_pred,
        'sensor': 'S2 Route B',
        'model': 'Logistic Regression baseline',
        'lr_train_arm': arm,
        'AWEI_p95': rng.uniform(-1.0, 1.0, n),
        'AWEI': rng.uniform(-1.0, 1.0, n),
        'NDMI': rng.uniform(-0.4, 1.0, n),
        'scene_prefix': prefix,
        'lon': lon,
        'lat': lat,
        'point_key': [f'{prefix}|{a:.7f}|{b:.7f}' for a, b in zip(lon, lat)],
        '.geo': [_geo(a, b) for a, b in zip(lon, lat)],
        'date': '2021-02-10',
        'dominant_class': ['W', 'T', 'F2', 'A'] * (n // 4),
    })


def write_manifest(tmp_path, manifest, name='manifest.csv'):
    path = tmp_path / name
    manifest.to_csv(path, index=False)
    return path


def scored(ns, manifest):
    """Load-free scoring pipeline: predictions, per-point rows, summary tables."""
    predictions = ns['s2_fixed_validation_predictions'](manifest)
    points = ns['s2_fixed_validation_points_long_form'](manifest, predictions)
    summary, per_class, confusion, matrices = ns['s2_fixed_validation_summary'](
        manifest, predictions)
    return predictions, points, summary, per_class, confusion, matrices


def write_metrics_csv(ns, tmp_path, summary, overrides=None,
                      arm='corrected_full_excluding_validation'):
    """A classifier-notebook metrics CSV agreeing with ``summary``'s LR row."""
    lr = summary[summary['classifier'].eq(ns['S2_FIXED_VALIDATION_LR_LABEL'])].iloc[0]
    row = {
        'sensor': 'S2 Route B',
        'diagnostic': 'fixed_validation_set',
        'model': 'Logistic Regression baseline',
        'train_arm': arm,
        'train_rows': 498,
        'train_blocks': 19,
        'validation_rows': int(lr['n_points']),
        'n_rows': int(lr['n_points']),
    }
    for metric in ns['S2_FIXED_VALIDATION_METRICS']:
        row[metric] = float(lr[metric])
    row.update(overrides or {})

    other = dict(row, train_arm='old_uncorrected_subset_only', train_rows=115)
    path = tmp_path / 'fixed_validation_metrics.csv'
    pd.DataFrame([other, row]).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. The rules used here are the raster rules, reused not restated.
# ---------------------------------------------------------------------------

def test_s2_rule_replay_still_matches_the_raster_rule_function():
    """The classifier scored in 13h must be the one the GeoTIFFs were written with."""
    ns = _namespace()
    classifier = _classifier_namespace()

    rng = np.random.default_rng(3)
    n = 4000
    frame = pd.DataFrame({
        'AWEI_p95': rng.uniform(-1.0, 1.0, n),
        'AWEI': rng.uniform(-1.0, 1.0, n),
        'NDMI': rng.uniform(-0.4, 1.0, n),
    })

    point_codes = ns['s2_rule_class_at_points'](frame)
    bands = {name: frame[name].to_numpy().reshape(1, n) for name in frame.columns}
    raster_codes = classifier['classify_s2_rules_array'](bands, np.ones((1, n), dtype=bool))

    assert np.array_equal(point_codes, raster_codes.ravel())
    assert set(np.unique(point_codes)) == set(S2_CLASS_CODES)


def _section_13h_code_source():
    """Joined source of the code cells between the 13h and 13i headings."""
    cells = _cells(NOTEBOOK)

    start = next(
        i for i, c in enumerate(cells)
        if c['cell_type'] == 'markdown'
        and '13h. S2 fixed-validation comparison' in ''.join(c['source'])
    )
    end = next(
        i for i, c in enumerate(cells)
        if i > start and c['cell_type'] == 'markdown'
        and '13i. Reporting the paired' in ''.join(c['source'])
    )

    sources = [
        ''.join(c['source'])
        for c in cells[start + 1:end]
        if c['cell_type'] == 'code'
    ]
    assert sources, 'section 13h has no code cells'
    return '\n'.join(sources)


def test_section_13h_reuses_the_single_rule_implementation():
    """13h must call s2_rule_class_at_points(), not restate the thresholds."""
    source = _section_13h_code_source()

    assert 's2_rule_class_at_points(' in source

    tree = ast.parse(source)
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert 's2_rule_class_at_points' not in defined, (
        'section 13h redefines the rule classifier instead of reusing section 13a'
    )

    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for threshold in ('S2_RULE_NDMI_ALGAE', 'S2_RULE_NDMI_FLOATING'):
        assert threshold not in assigned, f'section 13h re-assigns {threshold}'

    # No bare threshold constants either: a hard-coded 0.63 would be a second,
    # silently divergent copy of the branch boundaries.
    for literal in ('0.63', '-17.0', '-14.0'):
        assert literal not in source, f'section 13h hard-codes the threshold {literal}'


# ---------------------------------------------------------------------------
# 2. Validation ids must be present, complete and unique.
# ---------------------------------------------------------------------------

def test_duplicate_validation_ids_fail(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    manifest.loc[3, '_s2_diag_row_id'] = manifest.loc[2, '_s2_diag_row_id']

    with pytest.raises(ValueError, match='duplicated'):
        ns['validate_fixed_validation_ids'](manifest)

    with pytest.raises(ValueError, match='duplicated'):
        ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))


def test_missing_validation_ids_fail(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest().astype({'_s2_diag_row_id': 'float64'})
    manifest.loc[5, '_s2_diag_row_id'] = np.nan

    with pytest.raises(ValueError, match='missing or'):
        ns['validate_fixed_validation_ids'](manifest)

    with pytest.raises(ValueError, match='missing or'):
        ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))


def test_an_absent_id_column_fails(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest().drop(columns=['_s2_diag_row_id'])

    with pytest.raises(ValueError, match='_s2_diag_row_id'):
        ns['validate_fixed_validation_ids'](manifest)


def test_a_clean_manifest_round_trips_through_the_loader(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    loaded = ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))

    assert len(loaded) == len(manifest)
    # Sorted by id, so both classifiers inherit one deterministic row order.
    assert loaded['_s2_diag_row_id'].is_monotonic_increasing
    assert loaded['_s2_diag_row_id'].is_unique


# ---------------------------------------------------------------------------
# 3. Rule predictors: fail, never silently delete a point.
# ---------------------------------------------------------------------------

def test_non_finite_rule_predictors_fail_rather_than_dropping_rows(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    manifest.loc[4, 'NDMI'] = np.nan
    manifest.loc[9, 'AWEI'] = np.inf

    with pytest.raises(ValueError) as excinfo:
        ns['validate_fixed_validation_rule_predictors'](manifest)

    message = str(excinfo.value)
    assert '2 of 24' in message
    # The point ids are named, and the message says why deletion is not the fix.
    assert '4' in message and '9' in message
    assert 'deleting points' in message

    with pytest.raises(ValueError):
        ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))


def test_missing_rule_predictor_columns_fail(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest().drop(columns=['NDMI'])

    with pytest.raises(ValueError, match='NDMI'):
        ns['validate_fixed_validation_rule_predictors'](manifest)

    with pytest.raises(ValueError, match='NDMI'):
        ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))


def test_a_point_the_rules_cannot_classify_stops_the_comparison():
    """An unevaluable point must abort, not shrink one classifier's point set."""
    ns = _namespace()

    manifest = make_manifest()
    manifest.loc[6, 'AWEI_p95'] = np.nan  # -> NODATA from the rule replay

    with pytest.raises(ValueError, match='different point sets'):
        ns['s2_fixed_validation_predictions'](manifest)


# ---------------------------------------------------------------------------
# 4. The comparison is paired: identical ordered ids, counts and support.
# ---------------------------------------------------------------------------

def test_lr_and_rules_are_scored_on_identical_ordered_ids():
    ns = _namespace()

    manifest = make_manifest()
    _, points, _, _, _, _ = scored(ns, manifest)

    fingerprints = {
        classifier: ns['s2_fixed_validation_fingerprint'](points, classifier)
        for classifier in ns['S2_FIXED_VALIDATION_CLASSIFIERS']
    }
    lr, rules = fingerprints.values()

    assert lr['point_ids'] == rules['point_ids']
    assert lr['point_ids'] == tuple(manifest['_s2_diag_row_id'].tolist())
    assert lr['n_points'] == rules['n_points'] == len(manifest)
    assert lr['class_support'] == rules['class_support']
    assert lr['spatial_blocks'] == rules['spatial_blocks']
    assert lr['composition'] == rules['composition']

    # And the composition really does mix original and corrected points.
    assert dict(lr['composition']).keys() == {'original_accepted', 'manual_corrected_added'}

    assert ns['assert_s2_fixed_validation_is_paired'](points)['n_points'] == len(manifest)


def test_the_pairing_check_catches_a_dropped_point():
    ns = _namespace()

    manifest = make_manifest()
    _, points, _, _, _, _ = scored(ns, manifest)

    rules_label = ns['S2_FIXED_VALIDATION_RULES_LABEL']
    thinned = points.drop(
        points[points['classifier'].eq(rules_label)].index[:1]
    ).reset_index(drop=True)

    with pytest.raises(ValueError, match='not a paired comparison'):
        ns['assert_s2_fixed_validation_is_paired'](thinned)


def test_the_pairing_check_catches_reordered_ids():
    """Same ids in a different order is still not a paired, row-aligned evaluation."""
    ns = _namespace()

    manifest = make_manifest()
    _, points, _, _, _, _ = scored(ns, manifest)

    rules_label = ns['S2_FIXED_VALIDATION_RULES_LABEL']
    rules_rows = points[points['classifier'].eq(rules_label)]
    shuffled = pd.concat(
        [points[~points['classifier'].eq(rules_label)], rules_rows.iloc[::-1]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match='different order'):
        ns['assert_s2_fixed_validation_is_paired'](shuffled)


def test_both_summary_rows_share_n_points_and_floating_support():
    ns = _namespace()

    manifest = make_manifest()
    _, _, summary, _, _, _ = scored(ns, manifest)

    assert list(summary['classifier']) == list(ns['S2_FIXED_VALIDATION_CLASSIFIERS'])
    assert summary['n_points'].nunique() == 1
    assert int(summary['n_points'].iloc[0]) == len(manifest)
    # floating_support counts *reference* floating points, so it is a property of
    # the evaluation set and must be identical for both classifiers.
    assert summary['floating_support'].nunique() == 1
    assert int(summary['floating_support'].iloc[0]) == int(
        (manifest['true_class'] == FLOATING).sum())

    for field in ns['S2_FIXED_VALIDATION_COUNT_FIELDS']:
        assert summary[field].nunique() == 1, f'{field} differs between classifiers'

    # Both rows describe the same evaluation, by name as well as by content.
    assert summary['evaluation'].nunique() == 1
    assert 'fixed validation' in summary['evaluation'].iloc[0]

    assert list(summary['fitted_parameters']) == ['yes', 'no']


def test_the_accuracy_table_carries_one_row_per_classifier_with_the_agreed_labels():
    ns = _namespace()

    _, _, summary, _, _, _ = scored(ns, make_manifest())

    assert set(summary['classifier']) == {
        'Logistic Regression — corrected training data',
        'Rules-based — paper thresholds',
    }
    for metric in ns['S2_FIXED_VALIDATION_METRICS']:
        assert metric in summary.columns


# ---------------------------------------------------------------------------
# 5. The recomputed LR row must reproduce the classifier notebook's own result.
# ---------------------------------------------------------------------------

def test_lr_metric_row_matches_the_supplied_fixed_validation_result(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    _, _, summary, _, _, _ = scored(ns, manifest)

    metrics_path = write_metrics_csv(ns, tmp_path, summary)
    reference = ns['load_s2_fixed_validation_metrics_row'](metrics_path)

    assert ns['assert_lr_metrics_match_classifier_notebook'](summary, reference) is True


def test_a_disagreeing_metrics_row_is_rejected(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    _, _, summary, _, _, _ = scored(ns, manifest)

    lr = summary[summary['classifier'].eq(ns['S2_FIXED_VALIDATION_LR_LABEL'])].iloc[0]
    metrics_path = write_metrics_csv(
        ns, tmp_path, summary, overrides={'accuracy': float(lr['accuracy']) + 0.05})
    reference = ns['load_s2_fixed_validation_metrics_row'](metrics_path)

    with pytest.raises(ValueError, match='different evaluations'):
        ns['assert_lr_metrics_match_classifier_notebook'](summary, reference)


def test_a_metrics_row_with_a_different_validation_size_is_rejected(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    _, _, summary, _, _, _ = scored(ns, manifest)

    metrics_path = write_metrics_csv(
        ns, tmp_path, summary, overrides={'validation_rows': len(manifest) + 1})
    reference = ns['load_s2_fixed_validation_metrics_row'](metrics_path)

    with pytest.raises(ValueError, match='validation_rows'):
        ns['assert_lr_metrics_match_classifier_notebook'](summary, reference)


def test_only_the_corrected_arm_row_is_read(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    _, _, summary, _, _, _ = scored(ns, manifest)
    metrics_path = write_metrics_csv(ns, tmp_path, summary)

    row = ns['load_s2_fixed_validation_metrics_row'](metrics_path)

    assert row['train_arm'] == 'corrected_full_excluding_validation'
    # The old-arm row in the same file is trained on fewer rows and is not the one
    # this section may quote.
    assert int(row['train_rows']) == 498


# ---------------------------------------------------------------------------
# 6. Differences carry the documented sign convention.
# ---------------------------------------------------------------------------

def test_metric_differences_follow_the_documented_sign_convention():
    ns = _namespace()

    manifest = make_manifest()
    _, _, summary, _, _, _ = scored(ns, manifest)

    differences = ns['s2_fixed_validation_metric_differences'](summary)

    assert ns['S2_FIXED_VALIDATION_DIFFERENCE_CONVENTION'] == 'rules_minus_lr'
    assert set(differences['difference_convention']) == {'rules_minus_lr'}

    lr = summary[summary['classifier'].eq(ns['S2_FIXED_VALIDATION_LR_LABEL'])].iloc[0]
    rules = summary[summary['classifier'].eq(ns['S2_FIXED_VALIDATION_RULES_LABEL'])].iloc[0]

    reported = set(differences['metric'])
    expected = set(ns['S2_FIXED_VALIDATION_METRICS']) | set(
        ns['S2_FIXED_VALIDATION_COUNT_FIELDS'])
    assert reported == expected, 'every metric must be differenced, and only once'

    for _, row in differences.iterrows():
        metric = row['metric']
        # Same direction for every metric, with no per-metric flipping.
        assert row['difference'] == pytest.approx(float(rules[metric]) - float(lr[metric]))
        assert row['lr_value'] == pytest.approx(float(lr[metric]))
        assert row['rules_value'] == pytest.approx(float(rules[metric]))

        if row['difference'] > 0:
            assert row['favours'] == 'rules'
        elif row['difference'] < 0:
            assert row['favours'] == 'logistic_regression'
        else:
            assert row['favours'] == 'tie'

    # On a paired evaluation the count fields must difference to exactly zero.
    counts = differences[differences['metric'].isin(ns['S2_FIXED_VALIDATION_COUNT_FIELDS'])]
    assert len(counts) == len(ns['S2_FIXED_VALIDATION_COUNT_FIELDS'])
    assert (counts['difference'] == 0).all()


def test_the_difference_sign_actually_tracks_which_classifier_is_better():
    """A manifest where the rules are perfect and LR is not must report positives."""
    ns = _namespace()

    manifest = make_manifest()
    rule_codes = ns['s2_rule_class_at_points'](manifest)
    # Relabel so the rules are exactly right, then damage the LR column.
    manifest['true_class'] = np.asarray(rule_codes).astype(np.int64)
    manifest['lr_predicted_class'] = (manifest['true_class'] + 1) % 4

    _, _, summary, _, _, _ = scored(ns, manifest)
    differences = ns['s2_fixed_validation_metric_differences'](summary)

    accuracy = differences[differences['metric'].eq('accuracy')].iloc[0]
    assert accuracy['rules_value'] == pytest.approx(1.0)
    assert accuracy['difference'] > 0
    assert accuracy['favours'] == 'rules'


# ---------------------------------------------------------------------------
# 7. No training-set or map-space Logistic Regression predictions.
# ---------------------------------------------------------------------------

def test_a_manifest_from_another_training_arm_is_rejected(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest(arm='old_uncorrected_subset_only')

    with pytest.raises(ValueError, match='training-set prediction'):
        ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))


def test_a_manifest_mixing_arms_is_rejected(tmp_path):
    ns = _namespace(tmp_path)

    manifest = make_manifest()
    manifest.loc[0, 'lr_train_arm'] = 'old_uncorrected_subset_only'

    with pytest.raises(ValueError, match='may only score'):
        ns['load_s2_fixed_validation_manifest'](write_manifest(tmp_path, manifest))


def test_lr_predictions_come_from_the_manifest_column_only():
    ns = _namespace()

    manifest = make_manifest()
    predictions = ns['s2_fixed_validation_predictions'](manifest)

    assert np.array_equal(
        predictions[ns['S2_FIXED_VALIDATION_LR_LABEL']],
        manifest['lr_predicted_class'].to_numpy(),
    )
    # Changing the manifest column changes the scored predictions: nothing else
    # is consulted for the model's answers.
    manifest['lr_predicted_class'] = (manifest['lr_predicted_class'] + 1) % 4
    assert np.array_equal(
        ns['s2_fixed_validation_predictions'](manifest)[ns['S2_FIXED_VALIDATION_LR_LABEL']],
        manifest['lr_predicted_class'].to_numpy(),
    )


def test_section_13h_never_refits_a_model_or_samples_a_raster():
    """No fit, no raster sampling, no reuse of the map-space or CV point sets."""
    source = _section_13h_code_source()

    forbidden = [
        'sample_raster_at_points',       # 13f's map-space sampler
        'map_sampled_points',
        'MAP_PREDICTION_COLUMNS',
        'lr_map_code',
        'match_reference_points_to_scenes',
        'reference_points',              # 13a's original-CSV point set
        'spatial_block_folds',           # repeated folds belong to 13d, not here
        'aggregate_fold_metrics',
        '.fit(',
        'train_test_split',
        'LogisticRegression',
    ]
    for token in forbidden:
        assert token not in source, f'section 13h must not use {token!r}'

    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else
        node.func.id if isinstance(node.func, ast.Name) else None
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert 'fit' not in calls and 'predict' not in calls


def test_section_13h_reports_no_fold_spread():
    """A single fixed holdout has no fold standard deviation to report."""
    source = _section_13h_code_source()

    for token in ('_std', 'yerr', 'errorbar', 'capsize', 'n_folds_evaluated'):
        assert token not in source, (
            f'section 13h references {token!r}; a single fixed holdout has no folds'
        )


# ---------------------------------------------------------------------------
# 8. Producer side: the manifest comes from `valid`, inside the diagnostic.
# ---------------------------------------------------------------------------

def _classifier_function_source(name):
    for cell in _cells(CLASSIFIER_NOTEBOOK):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if f'def {name}' not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(source, node)
    raise AssertionError(f'{name} not found in the classifier notebook')


def test_the_manifest_is_built_from_the_valid_frame_inside_the_diagnostic():
    """Built where the split lives, not reconstructed later from another dataset."""
    source = _classifier_function_source('run_s2_fixed_validation_diagnostic')
    tree = ast.parse(source)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == '_s2_fixed_validation_manifest'
    ]
    assert len(calls) == 1, 'the manifest must be built exactly once, inside the diagnostic'

    first, second = calls[0].args[:2]
    assert isinstance(first, ast.Name) and first.id == 'valid', (
        'the manifest must be built from the already-created `valid` frame'
    )
    assert isinstance(second, ast.Name) and second.id == 'pred', (
        "the manifest must carry that run's own predictions"
    )

    assert 'S2_FIXED_VALIDATION_MANIFEST_STEM' in source
    assert 'to_csv' in source


def test_the_fixed_validation_split_is_untouched():
    """The seed and the validation fraction stay exactly as they were."""
    ns = _classifier_namespace()

    assert ns['S2_DIAGNOSTIC_VALIDATION_FRACTION'] == 0.25
    # SPATIAL_CV_RANDOM_STATE (42) + 711.
    assert 'SPATIAL_CV_RANDOM_STATE + 711' in _extract(
        CLASSIFIER_NOTEBOOK, {'S2_DIAGNOSTIC_RANDOM_STATE'})

    split = _classifier_function_source('_spatial_validation_row_ids')
    assert 'S2_DIAGNOSTIC_VALIDATION_FRACTION' in split
    assert 'S2_DIAGNOSTIC_RANDOM_STATE' in split


def test_the_manifest_filename_is_the_documented_one():
    ns = _classifier_namespace()

    assert ns['S2_FIXED_VALIDATION_MANIFEST_STEM'] == (
        'winam_s2_fixed_validation_manifest_diagnostic_more_training_data_check'
    )

    consumer = _namespace()
    assert Path(consumer['S2_FIXED_VALIDATION_MANIFEST_PATH']).name == (
        ns['S2_FIXED_VALIDATION_MANIFEST_STEM'] + '.csv'
    )


def test_the_exported_manifest_satisfies_the_consumers_loader(tmp_path):
    """End to end: the classifier notebook's own builder feeds section 13h."""
    producer = _classifier_namespace()
    consumer = _namespace(tmp_path)

    n = 12
    rng = np.random.default_rng(17)
    prefix = 'winam_s2_predictors_s2_whlev_temporal_v1_2021-02-10_to_2021-02-11'
    lon = 34.4 + rng.uniform(0, 0.4, n)
    lat = -0.4 + rng.uniform(0, 0.3, n)

    # A stand-in for the `valid` frame inside run_s2_fixed_validation_diagnostic().
    valid = pd.DataFrame({
        '_s2_diag_row_id': np.arange(n)[::-1],  # deliberately unsorted
        'correction_source': np.where(
            np.arange(n) % 2 == 0, 'original_accepted', 'manual_corrected_added'),
        'manual_correction': (np.arange(n) % 2).astype(int),
        'spatial_block': (np.arange(n) // 3).astype(np.int64),
        'class': [S2_CLASS_CODES[i % 4] for i in range(n)],
        '.geo': [_geo(a, b) for a, b in zip(lon, lat)],
        'AWEI_p95': rng.uniform(-1.0, 1.0, n),
        'AWEI': rng.uniform(-1.0, 1.0, n),
        'NDMI': rng.uniform(-0.4, 1.0, n),
        'date': '2021-02-10',
        'Date': '2021-02-10',
        'prefix': prefix,
        'dominant_class': ['W', 'T', 'F2', 'A'] * (n // 4),
        'Location': 'Winam',
        'lon': lon,
        'lat': lat,
    })
    pred = np.array([S2_CLASS_CODES[(i + 1) % 4] for i in range(n)])

    manifest = producer['_s2_fixed_validation_manifest'](valid, pred)

    for column in consumer['S2_FIXED_VALIDATION_REQUIRED_COLUMNS']:
        assert column in manifest.columns, f'the exported manifest lacks {column!r}'
    # Provenance the task asks the manifest to carry for auditing.
    for column in ('point_key', 'scene_prefix', 'lon', 'lat', '.geo', 'date'):
        assert column in manifest.columns

    loaded = consumer['load_s2_fixed_validation_manifest'](
        write_manifest(tmp_path, manifest))

    assert len(loaded) == n
    assert loaded['_s2_diag_row_id'].is_monotonic_increasing
    # The builder sorted the frame as a unit: labels still travel with their ids.
    by_id = dict(zip(valid['_s2_diag_row_id'], valid['class']))
    assert loaded['true_class'].tolist() == [
        by_id[i] for i in loaded['_s2_diag_row_id']]

    _, points, summary, _, _, matrices = scored(consumer, loaded)
    consumer['assert_s2_fixed_validation_is_paired'](points)

    assert summary['n_points'].nunique() == 1
    assert set(matrices) == set(consumer['S2_FIXED_VALIDATION_CLASSIFIERS'])
    assert all(matrix.shape == (4, 4) for matrix in matrices.values())


# ---------------------------------------------------------------------------
# 9. Both notebooks stay valid notebook JSON.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('notebook', [NOTEBOOK, CLASSIFIER_NOTEBOOK])
def test_notebook_json_is_well_formed(notebook):
    document = json.loads(notebook.read_text())

    assert document['nbformat'] == 4
    assert isinstance(document['cells'], list) and document['cells']

    for index, cell in enumerate(document['cells']):
        assert cell['cell_type'] in {'code', 'markdown', 'raw'}, index
        assert isinstance(cell['source'], list), index
        if cell['cell_type'] == 'code':
            assert isinstance(cell['outputs'], list), index
