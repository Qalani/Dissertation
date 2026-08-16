"""How the AOI series draws the months it never observed.

``Rebuild_AOI_WH_TimeSeries.ipynb`` plots single-date classified snapshots, not
monthly composites, so the record's cadence is uneven: some calendar months
carry several acquisitions and some carry none. A plain ``ax.plot`` of the
observed rows joins every consecutive pair with the same solid line, which
draws the unmeasured months exactly as confidently as the measured ones -- the
long 2017-2018 stretch with no usable scene becomes a straight, authoritative
line through months that were never classified.

What is pinned here:

* a segment is SOLID only when its two endpoints sit in the same or in adjacent
  calendar months, and DASHED as soon as one whole calendar month in between
  carries no observation;
* the dashes are cosmetic joins, never data: they carry the measured endpoint
  values, no marker, and no legend entry of their own;
* observations keep their acquisition dates -- nothing is snapped to a monthly
  grid, resampled or interpolated, so no point moves and no invented point
  appears;
* rows the plot cannot use (no date, or a NaN from the centred rolling median)
  drop out, and a month left empty that way counts as unobserved like any
  other;
* the notebook actually routes BOTH plotted lines through the helper, and the
  switch that turns the behaviour off restores a single unbroken line.

The helpers live in the notebook and are executed straight out of its JSON, so
these tests track the real notebook rather than a copy that can drift from it.
No Drive, no rasterio, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / 'Rebuild_AOI_WH_TimeSeries.ipynb'

HELPER_MARKER = 'def plot_with_month_gaps'
PLOT_MARKER = 'fig, ax = plt.subplots(figsize=(11.5, 5.6))'


def _cells():
    return json.loads(NOTEBOOK.read_text())['cells']


def _cell(marker):
    """Source text of the first code cell containing `marker`."""
    for cell in _cells():
        if cell['cell_type'] == 'code' and marker in ''.join(cell['source']):
            return ''.join(cell['source'])
    raise AssertionError(f'{NOTEBOOK.name}: no code cell contains {marker!r}')


def _markdown():
    return '\n'.join(
        ''.join(c['source']) for c in _cells() if c['cell_type'] == 'markdown'
    )


@pytest.fixture(scope='module')
def ns():
    """The notebook's gap-aware plotting helpers, executed from the notebook."""
    namespace = {'pd': pd, 'np': np, 'print': lambda *a, **k: None}
    exec(compile(_cell(HELPER_MARKER), '<gaps>', 'exec'), namespace)
    return namespace


@pytest.fixture
def ax():
    plt = pytest.importorskip('matplotlib.pyplot')
    pytest.importorskip('matplotlib').use('Agg')
    fig, axes = plt.subplots()
    yield axes
    plt.close(fig)


def _dates(*values):
    return pd.Series(pd.to_datetime(list(values)))


# ---------------------------------------------------------------------------
# Which segments count as crossing an unobserved month
# ---------------------------------------------------------------------------

def test_a_skipped_calendar_month_breaks_the_solid_line(ns):
    dates = _dates('2020-01-08', '2020-02-11', '2020-04-03')
    assert ns['month_gap_after'](pd.DatetimeIndex(dates)).tolist() == [False, True]


def test_two_acquisitions_in_one_month_are_never_a_gap(ns):
    """Same month means nothing was missed, however the days fall."""
    dates = _dates('2020-01-02', '2020-01-29')
    assert ns['month_gap_after'](pd.DatetimeIndex(dates)).tolist() == [False]


def test_adjacent_months_are_not_a_gap_even_when_weeks_apart(ns):
    """31 Jan -> 1 Mar skips February; 1 Jan -> 28 Feb skips nothing.

    The rule is calendar months with no observation, not elapsed days, so the
    wider-in-days pair is the solid one.
    """
    assert ns['month_gap_after'](pd.DatetimeIndex(_dates('2021-01-01', '2021-02-28')))\
        .tolist() == [False]
    assert ns['month_gap_after'](pd.DatetimeIndex(_dates('2021-01-31', '2021-03-01')))\
        .tolist() == [True]


def test_the_month_gap_test_survives_a_year_boundary(ns):
    """December -> January is adjacent; December -> February is not."""
    assert ns['month_gap_after'](pd.DatetimeIndex(_dates('2020-12-20', '2021-01-04')))\
        .tolist() == [False]
    assert ns['month_gap_after'](pd.DatetimeIndex(_dates('2020-12-20', '2021-02-04')))\
        .tolist() == [True]


# ---------------------------------------------------------------------------
# The months with no observation, and the runs they form
# ---------------------------------------------------------------------------

def test_unobserved_months_lists_exactly_the_months_with_no_acquisition(ns):
    dates = _dates('2020-01-08', '2020-01-27', '2020-04-03', '2020-05-19')
    missing = ns['unobserved_months'](dates)
    assert [str(m) for m in missing] == ['2020-02', '2020-03']


def test_unobserved_months_stops_at_the_ends_of_the_record(ns):
    """Only months INSIDE the record can be missing from it."""
    dates = _dates('2020-03-08', '2020-04-27')
    assert len(ns['unobserved_months'](dates)) == 0
    assert len(ns['unobserved_months'](_dates('2020-03-08'))) == 0
    assert len(ns['unobserved_months'](pd.Series([], dtype='datetime64[ns]'))) == 0


def test_gap_spans_group_consecutive_missing_months_longest_first(ns):
    """One row per dashed stretch, so the caption can say how much is missing."""
    dates = _dates('2019-01-05', '2019-03-06', '2019-08-07', '2019-09-08')
    spans = ns['month_gap_spans'](dates)
    assert spans['n_months'].tolist() == [4, 1]
    longest = spans.iloc[0]
    assert longest['first_missing_month'] == '2019-04'
    assert longest['last_missing_month'] == '2019-07'
    shortest = spans.iloc[1]
    assert shortest['first_missing_month'] == shortest['last_missing_month'] == '2019-02'


def test_gap_spans_of_a_complete_record_are_empty_but_still_summable(ns):
    """The plot cell sums `n_months` unconditionally."""
    spans = ns['month_gap_spans'](_dates('2020-01-05', '2020-02-06', '2020-03-07'))
    assert len(spans) == 0
    assert int(spans['n_months'].sum()) == 0


# ---------------------------------------------------------------------------
# The solid line: breaks at the gaps, and keeps the acquisition dates
# ---------------------------------------------------------------------------

def test_the_solid_line_breaks_where_a_month_was_never_observed(ns):
    x, y = ns['split_on_month_gaps'](
        _dates('2020-01-08', '2020-02-11', '2020-04-03'),
        pd.Series([10.0, 20.0, 40.0]),
    )
    assert len(x) == 4                     # one NaT/NaN break inserted
    assert pd.isna(x[2]) and np.isnan(y[2])
    assert y[[0, 1, 3]].tolist() == [10.0, 20.0, 40.0]


def test_observations_keep_their_acquisition_dates(ns):
    """No monthly resampling: a point plots where the scene was acquired.

    The scatter of classified dates is drawn from the raw frame, so if the line
    were snapped to month starts the markers would sit off the line.
    """
    dates = _dates('2020-01-08', '2020-02-11', '2020-04-03')
    x, _ = ns['split_on_month_gaps'](dates, pd.Series([10.0, 20.0, 40.0]))
    assert [t for t in x if not pd.isna(t)] == list(pd.to_datetime(dates))


def test_nothing_is_interpolated_into_a_gap(ns):
    """A 17-month hole contributes no values at all, only a break."""
    dates = _dates('2017-06-14', '2018-12-02', '2019-01-09')
    x, y = ns['split_on_month_gaps'](dates, pd.Series([1.0, 2.0, 3.0]))
    assert np.isfinite(y).sum() == 3
    assert len(x) == 4


def test_a_gapless_record_is_left_as_one_unbroken_run(ns):
    x, y = ns['split_on_month_gaps'](
        _dates('2020-01-08', '2020-02-11', '2020-03-03'), pd.Series([1.0, 2.0, 3.0]))
    assert len(x) == 3 and np.isfinite(y).all()


def test_rows_are_sorted_and_the_frames_index_is_ignored(ns):
    """Call sites pass filtered groups, whose indices are not 0..n-1."""
    frame = pd.DataFrame(
        {'mid_date': pd.to_datetime(['2020-04-03', '2020-01-08', '2020-02-11']),
         'wh_area_ha': [40.0, 10.0, 20.0]},
        index=[7, 3, 11],
    )
    x, y = ns['split_on_month_gaps'](frame['mid_date'], frame['wh_area_ha'])
    assert y[np.isfinite(y)].tolist() == [10.0, 20.0, 40.0]
    assert pd.isna(x[2])                   # the break still lands before April


def test_a_row_without_a_value_is_not_an_observation(ns):
    """The centred rolling median leaves NaN at the ends of a short series.

    Dropping the row is what lets its month count as unobserved: February holds
    only the NaN here, so January -> March must be dashed, not solid.
    """
    dates = _dates('2020-01-08', '2020-02-11', '2020-03-03')
    x, y = ns['split_on_month_gaps'](dates, pd.Series([10.0, np.nan, 30.0]))
    assert y[np.isfinite(y)].tolist() == [10.0, 30.0]
    assert pd.isna(x[1])
    gx, _ = ns['month_gap_bridges'](dates, pd.Series([10.0, np.nan, 30.0]))
    assert len(gx) == 3


# ---------------------------------------------------------------------------
# The dashed connectors
# ---------------------------------------------------------------------------

def test_bridges_span_only_the_unobserved_stretches(ns):
    """One connector per gap, carrying the observed endpoint values."""
    dates = _dates('2019-01-05', '2019-03-06', '2019-04-07', '2019-08-08')
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    gx, gy = ns['month_gap_bridges'](dates, values)
    assert len(gx) == 6                    # two gaps, each with a NaT separator
    assert list(gx[:2]) == [pd.Timestamp('2019-01-05'), pd.Timestamp('2019-03-06')]
    assert pd.isna(gx[2]) and np.isnan(gy[2])
    assert list(gx[3:5]) == [pd.Timestamp('2019-04-07'), pd.Timestamp('2019-08-08')]
    assert gy[:2].tolist() == pytest.approx([1.0, 2.0])
    assert gy[3:5].tolist() == pytest.approx([3.0, 4.0])


def test_no_bridges_when_every_month_is_observed(ns):
    gx, gy = ns['month_gap_bridges'](
        _dates('2020-01-08', '2020-02-11', '2020-03-03'), pd.Series([1.0, 2.0, 3.0]))
    assert len(gx) == 0 and len(gy) == 0


def test_bridges_need_two_observations_to_span_anything(ns):
    """A one-scene series has nothing to bridge, and must not raise."""
    gx, _ = ns['month_gap_bridges'](_dates('2017-06-14'), pd.Series([1.0]))
    assert len(gx) == 0
    gx, _ = ns['month_gap_bridges'](pd.Series([], dtype='datetime64[ns]'),
                                    pd.Series([], dtype=float))
    assert len(gx) == 0


# ---------------------------------------------------------------------------
# What lands on the axes
# ---------------------------------------------------------------------------

def test_the_gap_is_drawn_dashed_and_the_data_solid(ns, ax):
    line = ns['plot_with_month_gaps'](
        ax, _dates('2020-01-08', '2020-02-11', '2020-04-03'),
        pd.Series([10.0, 20.0, 40.0]),
        linewidth=1.2, alpha=0.72, label='S2 | model',
    )
    solid, dashed = ax.lines
    assert solid is line
    assert solid.get_linestyle() == '-' and dashed.get_linestyle() == '--'
    # The dashes inherit colour and alpha, carry no marker, and stay out of the
    # legend, so each series still contributes exactly one legend handle.
    assert dashed.get_color() == solid.get_color()
    assert dashed.get_alpha() == solid.get_alpha()
    assert dashed.get_marker() in (None, 'None', '')
    assert solid.get_label() == 'S2 | model' and dashed.get_label().startswith('_')
    # The dashed artist spans February -> April only; the solid one is broken
    # there, so the unmeasured March is never drawn as measured.
    dashed_y = np.asarray(dashed.get_ydata(), dtype=float)
    assert dashed_y[np.isfinite(dashed_y)].tolist() == [20.0, 40.0]
    assert np.isnan(np.asarray(solid.get_ydata(), dtype=float)).sum() == 1


def test_a_complete_record_draws_no_dashed_artist_at_all(ns, ax):
    ns['plot_with_month_gaps'](ax, _dates('2020-01-08', '2020-02-11', '2020-03-03'),
                               pd.Series([1.0, 2.0, 3.0]))
    assert len(ax.lines) == 1


def test_the_switch_restores_a_single_unbroken_line(ns, ax):
    """DASH_UNOBSERVED_MONTHS = False is the pre-existing figure."""
    ns['plot_with_month_gaps'](ax, _dates('2020-01-08', '2020-02-11', '2020-04-03'),
                               pd.Series([10.0, 20.0, 40.0]), dash_gaps=False)
    assert len(ax.lines) == 1
    assert np.isfinite(np.asarray(ax.lines[0].get_ydata(), dtype=float)).all()


# ---------------------------------------------------------------------------
# The notebook is actually wired to the helper
# ---------------------------------------------------------------------------

def test_both_plotted_lines_go_through_the_helper():
    """The series and its rolling median must obey the same convention."""
    plot_cell = _cell(PLOT_MARKER)
    assert plot_cell.count('plot_with_month_gaps(') == 2
    assert 'dash_gaps=DASH_UNOBSERVED_MONTHS' in plot_cell
    # No line is drawn straight from the observed rows any more.
    assert 'ax.plot(' not in plot_cell


def test_the_markers_still_show_classified_dates_only():
    """The scatter comes from the raw group, so no dashed run carries a point."""
    plot_cell = _cell(PLOT_MARKER)
    assert 'ax.scatter(\n        group["mid_date"],\n        group["wh_area_ha"],' in plot_cell


def test_the_dash_convention_is_configurable_and_on_by_default():
    config = _cell('ROLLING_MEDIAN_POINTS = 5')
    assert 'DASH_UNOBSERVED_MONTHS = True' in config


def test_the_figure_says_what_the_dashes_mean():
    """A reader of the PNG alone must be able to tell data from connector."""
    plot_cell = _cell(PLOT_MARKER)
    assert 'Line2D(' in plot_cell
    assert 'dashed: no observation in those calendar months' in plot_cell
    assert 'from matplotlib.lines import Line2D' in _cell('import matplotlib.dates as mdates')
    assert 'dashed wherever at least one whole calendar month carries no observation' \
        in _markdown()
