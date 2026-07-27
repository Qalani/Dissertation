"""Deletion safety around the Colab Drive mount.

Context, because these tests look paranoid without it. A failed Drive mount
leaves ordinary local directories sitting where the mount belongs, and clearing
them is a legitimate repair. The obvious command for that repair —
``rm -rf /content/drive`` — is also the most destructive thing anyone can run on
this project: when the mount is live it deletes straight through FUSE into real
Drive data. The two states are indistinguishable to ``ls``.

That has already happened here once, and every file in a Google Drive went to
the bin. So the distinction is enforced in code and pinned by these tests rather
than left to whoever is reading the instructions at the time.

No Drive, no Colab, no network: ``ismount`` and ``rmtree`` are injected.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from winam_diagnostics import temporal_backfill as tb


@pytest.fixture
def drive(tmp_path):
    """A fake /content/drive with MyDrive under it."""
    root = tmp_path / 'content' / 'drive'
    (root / 'MyDrive' / 'GEE_Exports_validated_snapshots').mkdir(parents=True)
    return root


def _mounted(*paths):
    """An ismount stub reporting exactly ``paths`` as live mounts."""
    wanted = {str(Path(p).resolve()) for p in paths}
    return lambda p: str(Path(p).resolve()) in wanted


def _nothing_mounted(_p):
    return False


# --------------------------------------------------------------------------
# The incident, directly
# --------------------------------------------------------------------------

def test_refuses_to_delete_the_mount_point_itself(drive):
    """This is the exact call that binned a Drive. It must raise, always."""
    with pytest.raises(tb.UnsafeRemovalError, match='live mount point'):
        tb.assert_safe_to_remove(drive, mount_prefix=drive, ismount=_mounted(drive))


def test_refuses_to_delete_anything_under_a_live_mount(drive):
    for victim in (drive / 'MyDrive',
                   drive / 'MyDrive' / 'GEE_Exports_validated_snapshots'):
        with pytest.raises(tb.UnsafeRemovalError, match='real Drive data'):
            tb.assert_safe_to_remove(victim, mount_prefix=drive,
                                     ismount=_mounted(drive))


def test_remove_stray_mount_dirs_is_a_no_op_when_the_mount_is_live(drive):
    called = []
    with pytest.raises(tb.UnsafeRemovalError):
        tb.remove_stray_mount_dirs(mount_prefix=drive, ismount=_mounted(drive),
                                   rmtree=called.append)
    assert called == [], 'rmtree must not run against a live mount'
    assert (drive / 'MyDrive' / 'GEE_Exports_validated_snapshots').exists()


# --------------------------------------------------------------------------
# The repair it exists to allow
# --------------------------------------------------------------------------

def test_allows_clearing_stray_dirs_when_nothing_is_mounted(drive):
    resolved = tb.assert_safe_to_remove(drive, mount_prefix=drive,
                                        ismount=_nothing_mounted)
    assert resolved == drive.resolve()


def test_remove_stray_mount_dirs_clears_a_poisoned_mount_point(drive):
    removed = tb.remove_stray_mount_dirs(mount_prefix=drive, ismount=_nothing_mounted)
    assert removed is True
    assert not drive.exists()


def test_remove_stray_mount_dirs_reports_nothing_to_do(tmp_path):
    assert tb.remove_stray_mount_dirs(mount_prefix=tmp_path / 'absent',
                                      ismount=_nothing_mounted) is False


def test_paths_outside_the_mount_prefix_are_unaffected(tmp_path):
    scratch = tmp_path / 'content' / 'winam_backfill_scratch'
    scratch.mkdir(parents=True)
    assert tb.assert_safe_to_remove(
        scratch, mount_prefix=tmp_path / 'content' / 'drive',
        ismount=_mounted(tmp_path / 'content' / 'drive')) == scratch.resolve()


# --------------------------------------------------------------------------
# Failing safe
# --------------------------------------------------------------------------

def test_an_unprobeable_path_counts_as_mounted(drive):
    """Under uncertainty the safe answer is the one that refuses to delete."""
    def explode(_p):
        raise OSError(107, 'Transport endpoint is not connected')

    assert tb.is_live_mount(drive, ismount=explode) is True
    with pytest.raises(tb.UnsafeRemovalError):
        tb.assert_safe_to_remove(drive, mount_prefix=drive, ismount=explode)


def test_a_symlink_into_the_mount_cannot_smuggle_a_delete_past_the_check(tmp_path, drive):
    """assert_safe_to_remove resolves before comparing, so aliases do not help."""
    alias = tmp_path / 'shortcut'
    alias.symlink_to(drive / 'MyDrive', target_is_directory=True)
    with pytest.raises(tb.UnsafeRemovalError, match='real Drive data'):
        tb.assert_safe_to_remove(alias, mount_prefix=drive, ismount=_mounted(drive))


# --------------------------------------------------------------------------
# Repo-wide invariant
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent

#: Recursive-delete callables. Matched on the AST, not on source text, so prose
#: warning against `rm -rf` does not trip the check that enforces the warning.
_RECURSIVE_DELETE_FUNCS = {'rmtree', 'removedirs'}

#: Shell magics are not parseable Python, so these are matched textually. `!rm -r`
#: in any form is banned outright: there is no safe hand-written recursive delete
#: in a notebook that lives beside a Drive mount.
_SHELL_RECURSIVE_RM = re.compile(r'^\s*[!%]\s*rm\s+(-\S*[rR]|--recursive)')


def _sources():
    """(provenance, source text) for every code cell and package module."""
    for nb_path in sorted(REPO.glob('*.ipynb')):
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                yield f'{nb_path.name} cell {i}', ''.join(cell['source'])
    for py in sorted((REPO / 'winam_diagnostics').glob('*.py')):
        yield py.name, py.read_text()


def _called_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            yield (func.attr if isinstance(func, ast.Attribute)
                   else getattr(func, 'id', None)), node.lineno


def test_no_unguarded_recursive_delete_anywhere():
    """Only the mount-checked helper may delete a tree.

    `remove_stray_mount_dirs` calls `assert_safe_to_remove` first and lives in
    temporal_backfill.py, so that module is the one allowed caller. Everything
    else — notebook cells above all — must go through it.
    """
    allowed = {'temporal_backfill.py'}
    offenders = []
    for where, src in _sources():
        if where.split(' ')[0] in allowed:
            continue
        for line in src.splitlines():
            if _SHELL_RECURSIVE_RM.match(line):
                offenders.append((where, line.strip()))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # notebook magics; the textual scan above still applied
        for name, lineno in _called_names(tree):
            if name in _RECURSIVE_DELETE_FUNCS:
                offenders.append((where, f'line {lineno}: {name}(...)'))
    assert not offenders, (
        'unguarded recursive delete:\n'
        + '\n'.join(f'  {w}: {ln}' for w, ln in offenders)
        + '\nUse winam_diagnostics.temporal_backfill.remove_stray_mount_dirs instead.'
    )


def test_the_check_would_catch_a_real_regression(tmp_path):
    """Guard the guard: prove the scan fails on code that should not pass."""
    bad_cell = "import shutil\nshutil.rmtree('/content/drive')\n"
    names = [n for n, _ in _called_names(ast.parse(bad_cell))]
    assert 'rmtree' in names
    assert _SHELL_RECURSIVE_RM.match('!rm -rf /content/drive')
    assert _SHELL_RECURSIVE_RM.match('  ! rm -r /content/drive')
    # ...and does not fire on prose describing the danger.
    assert not _SHELL_RECURSIVE_RM.match('    Do not run `rm -rf /content/drive`.')


def test_the_guarded_helper_checks_before_it_deletes():
    """Order matters: assert_safe_to_remove must precede the rmtree call."""
    import inspect
    src = inspect.getsource(tb.remove_stray_mount_dirs)
    assert src.index('assert_safe_to_remove') < src.index('rmtree if rmtree'), \
        'remove_stray_mount_dirs deletes before it checks'
