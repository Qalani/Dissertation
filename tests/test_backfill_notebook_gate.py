"""Phase 0's blocking checks, tested against the notebook's own source.

Both live in notebook cells rather than in ``temporal_backfill``, because they are
about Drive paths and session state. That is exactly why they need tests here. Two
bugs the module tests could not see, both found the expensive way on 2026-07-27:

* Phase 1 refused to run in any session that had run Phase 0, and running the
  cells in order always produces that state -- so there was no reachable way
  through the notebook at all;
* a scan that found 1 file where 3121 had been went on to report both sensors
  UNVALIDATED with nothing to backfill, rather than saying the archive had gone.

These tests ``exec`` the real cells out of the generated notebook, so they fail if
the notebook and the builder drift apart, and they never touch Drive, Colab or the
network.
"""
from __future__ import annotations

import ast
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from winam_diagnostics import temporal_backfill as tb

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / 'Backfill_Temporal_Bands_Local.ipynb'


def _cells(cell_type='code'):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return [c for c in notebook['cells'] if c['cell_type'] == cell_type]


def _cell_containing(needle):
    matches = [c for c in _cells() if needle in ''.join(c['source'])]
    assert len(matches) == 1, f'expected exactly one cell containing {needle!r}, got {len(matches)}'
    return ''.join(matches[0]['source'])


def test_every_code_cell_parses():
    """The builder assembles cells as nested triple-quoted strings; one bad escape
    would ship a notebook that raises SyntaxError on the user's first run."""
    for index, cell in enumerate(_cells()):
        source = ''.join(cell['source'])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - only on a build regression
            pytest.fail(f'code cell {index} does not parse: {exc}')


@pytest.fixture()
def gate(tmp_path):
    """The helpers cell, exec'd with Drive paths pointed at a tmp directory."""
    report_dir = tmp_path / 'reports'
    report_dir.mkdir()
    namespace = {
        'contextmanager': contextmanager,
        'Path': Path,
        'json': json,
        'tb': tb,
        'MAX_DRIVE_REMOUNTS': 25,
        'READONLY_DIRS': [],
        'BACKFILL_ROOT': tmp_path,
        'CACHE_DIR': tmp_path / 'source_band_cache',
        'SIDECAR_DIR': tmp_path / 'sidecars',
        'VRT_DIR': tmp_path / 'vrt',
        'REPORT_DIR': report_dir,
        'PHASE0_REPORT_PATH': report_dir / 'phase0_report.json',
        'SESSION_ID': 'session-under-test',
    }
    exec(compile(_cell_containing('def require_phase0'), '<helpers cell>', 'exec'), namespace)
    return namespace


def _report(gate, session_id='session-under-test', statuses=(('S1', tb.VALIDATION_PASS),),
            settled_ddof=0, n_to_backfill=163):
    report = {
        'session_id': session_id,
        'recorded_at': '2026-07-27T12:00:00+00:00',
        'settled_ddof': settled_ddof,
        'settled_ddof_note': 'fitted empirically',
        'verdicts': {
            sensor: {'status': status, 'n_snapshots': 725,
                     'n_to_backfill': n_to_backfill, 'message': f'{sensor} {status}'}
            for sensor, status in statuses
        },
    }
    gate['PHASE0_REPORT_PATH'].write_text(json.dumps(report))
    return report


# --- The bug: no way through when Phase 0 ran in this session -----------------

def test_phase0_written_this_session_blocks_until_acknowledged(gate):
    _report(gate)
    with pytest.raises(RuntimeError, match='has not been acknowledged'):
        gate['require_phase0']('Phase 1')


def test_acknowledging_unlocks_the_bulk_phases_in_the_same_session(gate):
    """The regression test for the reported failure: running the cells in order,
    Phase 0 and Phase 1 in one session, must be completable."""
    expected = _report(gate)
    gate['acknowledge_phase0']()
    assert gate['require_phase0']('Phase 1')['recorded_at'] == expected['recorded_at']
    assert gate['require_phase0']('Phase 2/3')['settled_ddof'] == 0


def test_blocked_message_names_the_cell_that_unblocks_it(gate):
    _report(gate)
    with pytest.raises(RuntimeError) as excinfo:
        gate['require_phase0']('Phase 1')
    assert 'acknowledge_phase0()' in str(excinfo.value)
    assert 'section 3e' in str(excinfo.value)


# --- The gate still has to gate -----------------------------------------------

def test_report_from_an_earlier_run_still_passes_without_acknowledgement(gate):
    """Unchanged behaviour: a restart-then-run-the-bulk-phases workflow keeps
    working, because that verdict was read in the run that produced it."""
    _report(gate, session_id='some-earlier-session')
    assert gate['require_phase0']('Phase 1')['settled_ddof'] == 0


def test_rerunning_phase0_revokes_an_acknowledgement(gate):
    _report(gate)
    gate['acknowledge_phase0']()
    # Phase 0 runs again and writes a different verdict.
    _report(gate, statuses=(('S1', tb.VALIDATION_UNVALIDATED),), settled_ddof=1)
    with pytest.raises(RuntimeError, match='re-run since you acknowledged it'):
        gate['require_phase0']('Phase 1')


def test_failed_validation_cannot_be_acknowledged(gate):
    _report(gate, statuses=(('S1', tb.VALIDATION_FAIL),))
    with pytest.raises(RuntimeError, match='FAILED'):
        gate['acknowledge_phase0']()
    with pytest.raises(RuntimeError, match='FAILED'):
        gate['require_phase0']('Phase 1')


def test_failed_validation_blocks_even_from_an_earlier_run(gate):
    _report(gate, session_id='some-earlier-session', statuses=(('S1', tb.VALIDATION_FAIL),))
    with pytest.raises(RuntimeError, match='FAILED'):
        gate['require_phase0']('Phase 1')


def test_missing_report_reports_what_is_on_drive_instead(gate):
    (gate['CACHE_DIR'] / 'S1').mkdir(parents=True)
    (gate['CACHE_DIR'] / 'S1' / 'cached.tif').write_bytes(b'')
    with pytest.raises(RuntimeError) as excinfo:
        gate['require_phase0']('Phase 1')
    message = str(excinfo.value)
    assert 'does not exist' in message
    # Sibling outputs present -> a lost reports folder, not a first run.
    assert 'lost or' in message and 'Trash' in message
    assert 'source band cache' in message


def test_acknowledging_a_missing_report_fails_with_the_same_diagnosis(gate):
    with pytest.raises(RuntimeError, match='does not exist'):
        gate['acknowledge_phase0']()


def test_unvalidated_can_be_acknowledged_and_says_so(gate, capsys):
    """UNVALIDATED is not a failure -- there is nothing to compare against -- but
    it must not slip past silently either."""
    _report(gate, statuses=(('S1', tb.VALIDATION_UNVALIDATED), ('S2', tb.VALIDATION_PASS)))
    gate['acknowledge_phase0']()
    printed = capsys.readouterr().out
    assert 'could NOT be validated' in printed
    assert 'S1' in printed
    assert gate['require_phase0']('Phase 1')


def test_nothing_to_backfill_is_called_out_not_hidden(gate, capsys):
    """The 2026-07-27 run acknowledged a verdict with zero targets and would have
    reported 'Phase 1 complete' having done nothing at all."""
    _report(gate, n_to_backfill=0)
    gate['acknowledge_phase0']()
    printed = capsys.readouterr().out
    assert 'nothing to backfill' in printed
    assert 'archive scan in section 3' in printed


# --- Wiring: the notebook must actually use the gate --------------------------

def test_every_bulk_phase_calls_the_gate():
    bulk_markers = [
        "require_phase0('Phase 1 (source-band cache)')",
        "require_phase0('Phase 2/3 (rolling statistics and outputs)')",
        "require_phase0('Rewrite in place')",
    ]
    sources = [''.join(c['source']) for c in _cells()]
    for marker in bulk_markers:
        assert any(marker in s for s in sources), f'no cell calls {marker}'


def test_acknowledgement_cell_exists_and_is_a_single_call():
    """One cell, one call: nothing to configure, nothing to read past."""
    bodies = [''.join(c['source']).strip() for c in _cells()]
    assert bodies.count('PHASE0 = acknowledge_phase0()') == 1, [b[:60] for b in bodies]


def test_no_cell_still_tells_the_user_to_rerun_setup_to_get_past_the_gate():
    """The old instruction sent people to re-run the section 1 setup cell purely to
    churn SESSION_ID. Nothing should advertise that as the way through any more."""
    for cell in _cells():
        source = ''.join(cell['source'])
        assert 're-run the setup cell in section 1' not in source


# --- Phase 0's archive-collapse guard -----------------------------------------

def _export_file(export_dir, name):
    """A zero-byte file with a real export name.

    Nothing in the inventory cell opens these -- discovery and grouping work on
    names alone -- so the guard can be tested without synthesising rasters.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / name).write_bytes(b'')


def _s1_names(n):
    return [f'winam_s1_scc_temporal_v1_2020-{m // 28 + 1:02d}-{m % 28 + 1:02d}'
            f'_to_2020-{m // 28 + 1:02d}-{m % 28 + 1:02d}.tif' for m in range(n)]


@pytest.fixture()
def inventory_cell(tmp_path):
    """The section 3 inventory cell, wired to a synthetic export archive."""
    export_dir = tmp_path / 'GEE_Exports_validated_snapshots'
    report_dir = tmp_path / 'reports'
    report_dir.mkdir()
    remounts = []

    def _remount_for_recovery(context):
        remounts.append(context)
        return False

    namespace = {
        'Path': Path,
        'pd': pd,
        'tb': tb,
        'display': lambda *a, **k: None,
        'with_drive_retry': lambda func, *a, **k: func(
            *a, **{k2: v for k2, v in k.items() if k2 not in {'context', 'probe_path'}}),
        '_remount_for_recovery': _remount_for_recovery,
        'GEE_EXPORT_DIR': export_dir,
        'EE_EXPORT_FOLDER': 'GEE_Exports_validated_snapshots',
        'SENSORS': ['S2', 'S1'],
        'INVENTORY_CSV_PATH': report_dir / 'winam_temporal_backfill_inventory.csv',
        'CACHE_MANIFEST_PATH': report_dir / 'winam_temporal_backfill_cache_manifest.csv',
        'ARCHIVE_SHRINK_POLICY': 'abort',
        '__export_dir': export_dir,
        '__report_dir': report_dir,
        '__remounts': remounts,
    }
    namespace['__source'] = _cell_containing('def _scan_export_archive')
    return namespace


def _run_inventory_cell(namespace):
    exec(compile(namespace['__source'], '<inventory cell>', 'exec'), namespace)


def test_inventory_cell_runs_clean_on_a_first_run(inventory_cell):
    for name in _s1_names(5):
        _export_file(inventory_cell['__export_dir'], name)
    _run_inventory_cell(inventory_cell)
    assert inventory_cell['ARCHIVE_CHECK']['verdict'] == tb.ARCHIVE_NO_BASELINE
    assert len(inventory_cell['ALL_SNAPSHOTS']) == 5


def test_collapsed_archive_stops_phase0_instead_of_reporting_nothing_to_do(inventory_cell):
    """The 2026-07-27 failure: 3121 files became 1, and Phase 0 reported both
    sensors UNVALIDATED with 0 snapshots to backfill rather than saying the
    archive had gone."""
    _export_file(inventory_cell['__export_dir'], _s1_names(1)[0])
    # A cache manifest from the run that saw the full archive.
    tb.save_manifest(
        pd.DataFrame([{'status': 'cached', 'source_paths': f'/d/f{i}.tif'} for i in range(3121)]),
        inventory_cell['CACHE_MANIFEST_PATH'], tb.CACHE_MANIFEST_COLUMNS)

    with pytest.raises(RuntimeError) as excinfo:
        _run_inventory_cell(inventory_cell)
    message = str(excinfo.value)
    assert 'missing files it had before' in message
    assert '3121' in message
    assert 'Trash' in message
    assert 'ARCHIVE_SHRINK_POLICY' in message
    # It tried a remount first, because a partial listing looks identical.
    assert inventory_cell['__remounts']


def test_the_cache_manifest_baseline_survives_a_truncated_inventory_csv(inventory_cell):
    """The inventory CSV is rewritten by every Phase 0 run, so the truncated scan
    had already overwritten it with its own 1-file count. The guard must not be
    blinded by that."""
    _export_file(inventory_cell['__export_dir'], _s1_names(1)[0])
    pd.DataFrame([{'n_files': 1}]).to_csv(inventory_cell['INVENTORY_CSV_PATH'], index=False)
    tb.save_manifest(
        pd.DataFrame([{'status': 'cached', 'source_paths': f'/d/f{i}.tif'} for i in range(3121)]),
        inventory_cell['CACHE_MANIFEST_PATH'], tb.CACHE_MANIFEST_COLUMNS)

    with pytest.raises(RuntimeError, match='missing files it had before'):
        _run_inventory_cell(inventory_cell)


def test_accept_policy_lets_a_deliberately_pruned_archive_through(inventory_cell, capsys):
    for name in _s1_names(1):
        _export_file(inventory_cell['__export_dir'], name)
    pd.DataFrame([{'n_files': 5}]).to_csv(inventory_cell['INVENTORY_CSV_PATH'], index=False)
    inventory_cell['ARCHIVE_SHRINK_POLICY'] = 'accept'

    _run_inventory_cell(inventory_cell)
    printed = capsys.readouterr().out
    assert 'continuing anyway' in printed
    assert 'fewer' in printed
    assert len(inventory_cell['ALL_SNAPSHOTS']) == 1


def test_a_grown_archive_is_not_flagged(inventory_cell):
    for name in _s1_names(6):
        _export_file(inventory_cell['__export_dir'], name)
    pd.DataFrame([{'n_files': 5}]).to_csv(inventory_cell['INVENTORY_CSV_PATH'], index=False)
    _run_inventory_cell(inventory_cell)
    assert inventory_cell['ARCHIVE_CHECK']['verdict'] == tb.ARCHIVE_INTACT
    assert not inventory_cell['__remounts']


def test_a_remount_that_restores_the_listing_lets_the_run_continue(inventory_cell):
    """The benign half of the ambiguity: the mount was serving a partial listing,
    and re-scanning after a force-remount finds everything."""
    export_dir = inventory_cell['__export_dir']
    names = _s1_names(5)
    _export_file(export_dir, names[0])
    pd.DataFrame([{'n_files': 5}]).to_csv(inventory_cell['INVENTORY_CSV_PATH'], index=False)

    def _remount(context):
        for name in names[1:]:
            _export_file(export_dir, name)
        return True

    inventory_cell['_remount_for_recovery'] = _remount
    _run_inventory_cell(inventory_cell)
    assert inventory_cell['ARCHIVE_CHECK']['verdict'] == tb.ARCHIVE_INTACT
    assert len(inventory_cell['ALL_SNAPSHOTS']) == 5
