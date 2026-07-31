"""Regression tests for the batched AOI-mean monthly climate reduction.

``aoi_monthly_dataframe`` used to build every month of the panel inside ONE
server-side ``FeatureCollection`` and pull it back with a single ``getInfo``. On
the multi-year panels that means ~96 multi-band CHIRPS/ERA5/MODIS composites in
one query, which Earth Engine rejects with "User memory limit exceeded" and takes
the whole ``monthly_climate`` covariate group down with it.

The helper now issues one request per bounded batch of months
(``EE_CLIMATE_MONTHS_PER_REQUEST``), halves a batch that still hits a capacity
error, and escalates ``tileScale`` for a lone month that fails on its own. These
tests pin that behaviour, plus the two things the fix must NOT disturb: the
covariate definitions and the cache signature (so the groups already cached are
reused and only ``monthly_climate`` is re-attempted).

The functions under test live in the notebooks, so they are extracted from the
notebook JSON by name and executed against a MOCKED Earth Engine client: no
Earth Engine, no Drive, no network.
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

# Every name the batched monthly reduction needs to run standalone.
WANTED = ('_is_ee_capacity_error', '_aoi_monthly_batch', 'aoi_monthly_dataframe')

# Payload keys of _ee_cache_signature. Pinned so a performance knob added to the
# extractor cannot silently join the fingerprint and invalidate every cached
# covariate group. 'ee_climate_per_cell' exists only in the GAM notebook.
# What the signature must cover: the DATES requested, the AOI, the RESOLUTION and
# every EE_* extraction parameter. The panel's month LIST is deliberately absent —
# a month appearing or being filtered out by the monthly-coverage gate does not
# change an already-extracted month's value, so it must not force a full rebuild.
CACHE_SIGNATURE_KEYS = {
    'cell_size_m', 'panel_crs', 'aoi_bbox_wgs84', 'n_grid_cells', 'grid_bounds',
    'test_start', 'test_end',
    'ee_layers', 'ee_s2_products', 'ee_s3_products', 'ee_rain_antecedent_days',
    'ee_rain_intensity', 'ee_rain_spike_thresholds_mm', 'ee_rain_wet_day_mm',
    'ee_rain_per_cell', 'ee_bay_axis_bearing_deg', 'ee_water_occurrence_threshold',
    'ee_s2_cloud_pct', 'ee_climate_aoi_scale', 'ee_air_temp_use_era5_land',
    'ee_water_temp_per_cell', 'ee_catchment_buffer_m',
    'ee_river_discharge_weighted', 'ee_river_major_max_ord', 'ee_static_year',
    'ee_s3_calib',
}
CACHE_SIGNATURE_OPTIONAL_KEYS = {'ee_climate_per_cell'}

# The AOI-mean climate bands climate_image_for_month may emit. The batching change
# must not add, drop or rename a covariate.
CLIMATE_BAND_NAMES = {
    'rain_chirps_mm', 'rain_era5_mm', 'rain_max_1d_mm', 'rain_wet_days',
    'wind_u_ms', 'wind_v_ms', 'wind_speed_ms', 'air_temp_c', 'water_temp_c',
    'chl_modis_mg_m3',
    's',  # per-image scratch name inside the wind-speed expression, not a covariate
}


# ---------------------------------------------------------------------------
# Mocked Earth Engine client
# ---------------------------------------------------------------------------

class FakeEEException(Exception):
    """Stands in for ee.ee_exception.EEException."""


class _Stats(dict):
    """A reduceRegion result that remembers the tileScale it was asked for."""

    tile_scale = None


class _FakeDate:
    def __init__(self, millis):
        self.millis = int(millis)


class _FakeReducer:
    def __init__(self, name):
        self.name = name


class _FakeReducerFactory:
    @staticmethod
    def mean():
        return _FakeReducer('mean')


class _FakeFeature:
    def __init__(self, props, meta=None):
        self.props = dict(props)
        self.meta = dict(meta or {})

    def set(self, key, value):
        out = _FakeFeature(self.props, self.meta)
        out.props[key] = value
        return out


class _FakeList:
    def __init__(self, values):
        self.values = list(values)

    def map(self, fn):
        return [fn(v) for v in self.values]


class _FakeImage:
    """One month's climate composite; reduceRegion records how it was called."""

    def __init__(self, client, month_millis):
        self.client = client
        self.month_millis = month_millis

    def reduceRegion(self, reducer=None, geometry=None, scale=None,
                     maxPixels=None, tileScale=None, bestEffort=None):
        self.client.reductions.append({
            'month_millis': self.month_millis, 'reducer': reducer,
            'geometry': geometry, 'scale': scale, 'maxPixels': maxPixels,
            'tileScale': tileScale, 'bestEffort': bestEffort,
        })
        stats = _Stats({'rain_chirps_mm': float(self.month_millis % 1000),
                        'air_temp_c': 25.0})
        stats.tile_scale = tileScale
        return stats


class _FakeFeatureCollection:
    def __init__(self, client, features):
        self.client = client
        self.features = list(features)

    def getInfo(self):
        months = tuple(f.props['month_millis'] for f in self.features)
        tile_scales = {f.meta.get('tile_scale') for f in self.features}
        tile_scale = tile_scales.pop() if len(tile_scales) == 1 else None
        return self.client.request(months, tile_scale, self.features)


class FakeEE:
    """Minimal stand-in for the ``ee`` module used by the monthly reduction.

    ``fail`` is called as ``fail(months, tile_scale)`` for every getInfo and
    returns an error message to raise (or None to succeed), which lets a test
    describe an Earth Engine capacity limit declaratively. Every request is
    recorded in ``requests`` as ``(months, tile_scale)``.
    """

    def __init__(self, fail=None):
        self._fail = fail or (lambda months, tile_scale: None)
        self.requests = []
        self.reductions = []
        self.Reducer = _FakeReducerFactory

    # --- ee.* surface used by the notebook helper ---
    def Number(self, value):
        return value

    def Date(self, millis):
        return _FakeDate(millis)

    def List(self, values):
        return _FakeList(values)

    def Feature(self, geometry, props):
        return _FakeFeature(props, {'tile_scale': getattr(props, 'tile_scale', None)})

    def FeatureCollection(self, features):
        return _FakeFeatureCollection(self, features)

    # --- server behaviour ---
    def request(self, months, tile_scale, features):
        self.requests.append((tuple(months), tile_scale))
        message = self._fail(months, tile_scale)
        if message:
            raise FakeEEException(message)
        return {'features': [{'properties': dict(f.props)} for f in features]}

    # --- convenience for assertions ---
    @property
    def request_sizes(self):
        return [len(months) for months, _ in self.requests]

    @property
    def months_returned(self):
        seen = []
        for months, _ in self.requests:
            seen.extend(months)
        return seen


def image_for_month(client):
    """An ``image_for_month(date)`` callable backed by the fake client."""
    return lambda date: _FakeImage(client, date.millis)


# ---------------------------------------------------------------------------
# Notebook source extraction
# ---------------------------------------------------------------------------

def _extract(notebook, wanted=WANTED):
    """Source text of the named top-level defs in the EE covariate-helper cell."""
    found = {}
    for cell in json.loads(notebook.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'def aoi_monthly_dataframe' not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                found[node.name] = ast.get_source_segment(source, node)
    missing = set(wanted) - set(found)
    assert not missing, f'{notebook.name}: monthly-climate helpers missing {sorted(missing)}'
    return '\n\n'.join(found[name] for name in wanted if name in found)


def _load(notebook, fail=None, tile_scale=4, max_tile_scale=16,
          months_per_request=12):
    """Exec the notebook's monthly-reduction helpers against a fake ``ee``."""
    client = FakeEE(fail=fail)
    ns = {
        'pd': pd,
        'ee': client,
        'EE_TILE_SCALE': tile_scale,
        'EE_TILE_SCALE_MAX': max_tile_scale,
        'EE_CLIMATE_MONTHS_PER_REQUEST': months_per_request,
    }
    exec(_extract(notebook), ns)
    ns['_client'] = client
    return ns


def _months(n, start='2018-01-01'):
    return list(pd.date_range(start, periods=n, freq='MS'))


def _code(notebook):
    return '\n'.join(
        ''.join(c['source'])
        for c in json.loads(notebook.read_text())['cells']
        if c['cell_type'] == 'code'
    )


def _function_source(notebook, name):
    for cell in json.loads(notebook.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if f'def {name}' not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(source, node)
    raise AssertionError(f'{notebook.name}: no top-level def {name}')


@pytest.fixture(params=NOTEBOOKS, ids=lambda p: p.stem)
def notebook(request):
    return request.param


# ---------------------------------------------------------------------------
# The whole panel is no longer one Earth Engine computation
# ---------------------------------------------------------------------------

def test_every_notebook_carries_the_batched_helper(notebook):
    code = _code(notebook)
    assert 'def _aoi_monthly_batch(' in code
    assert 'EE_CLIMATE_MONTHS_PER_REQUEST = 12' in code
    # The unbounded whole-panel form must be gone from every notebook: the months
    # now reach Earth Engine through _aoi_monthly_batch, never straight from the
    # caller's full month list.
    assert 'def aoi_monthly_dataframe(image_for_month, months, scale, aoi):' not in code
    assert 'months_per_request' in code


def test_full_panel_is_split_into_bounded_batches(notebook):
    ns = _load(notebook)
    client = ns['_client']
    months = _months(96)

    df = ns['aoi_monthly_dataframe'](image_for_month(client), months, 1000, 'AOI')

    assert client.request_sizes == [12] * 8, 'expected one request per 12 months'
    assert len(df) == 96
    assert list(df['month']) == list(pd.to_datetime(months))
    assert list(df.columns) == ['rain_chirps_mm', 'air_temp_c', 'month']
    # Every month was reduced exactly once, at the configured tileScale.
    assert sorted(client.months_returned) == sorted(
        int(pd.Timestamp(m).value // 10 ** 6) for m in months)
    assert {r['tileScale'] for r in client.reductions} == {4}


def test_final_batch_is_a_remainder_not_a_padded_batch(notebook):
    ns = _load(notebook)
    client = ns['_client']

    df = ns['aoi_monthly_dataframe'](image_for_month(client), _months(26), 1000, 'AOI')

    assert client.request_sizes == [12, 12, 2]
    assert len(df) == 26


def test_batch_size_is_configurable(notebook):
    ns = _load(notebook, months_per_request=6)
    client = ns['_client']

    ns['aoi_monthly_dataframe'](image_for_month(client), _months(24), 1000, 'AOI')
    assert client.request_sizes == [6] * 4

    # An explicit argument overrides the configured default.
    other = _load(notebook, months_per_request=6)
    other_client = other['_client']
    other['aoi_monthly_dataframe'](image_for_month(other_client), _months(24), 1000,
                                   'AOI', months_per_request=24)
    assert other_client.request_sizes == [24]


def test_reduction_arguments_are_unchanged(notebook):
    """Batching must not alter how a month is reduced (same covariate values)."""
    ns = _load(notebook)
    client = ns['_client']
    ns['aoi_monthly_dataframe'](image_for_month(client), _months(3), 1000, 'AOI')

    for call in client.reductions:
        assert call['reducer'].name == 'mean'
        assert call['geometry'] == 'AOI'
        assert call['scale'] == 1000
        assert call['maxPixels'] == 1e13
        assert call['bestEffort'] is True


# ---------------------------------------------------------------------------
# Recovery from Earth Engine capacity errors
# ---------------------------------------------------------------------------

def test_memory_limit_splits_the_batch_recursively(notebook):
    """The reported failure: Earth Engine refuses anything above a size limit."""
    def fail(months, tile_scale):
        if len(months) > 3:
            return 'User memory limit exceeded.'
        return None

    ns = _load(notebook, fail=fail)
    client = ns['_client']
    months = _months(24)

    df = ns['aoi_monthly_dataframe'](image_for_month(client), months, 1000, 'AOI')

    # Nothing is lost: every month still lands in the frame, in order, once.
    assert len(df) == 24
    assert list(df['month']) == list(pd.to_datetime(months))
    succeeded = [m for m, _ in client.requests if len(m) <= 3]
    assert sorted(sum(succeeded, ())) == sorted(
        int(pd.Timestamp(m).value // 10 ** 6) for m in months)
    # The batch was halved, not retried whole, and tileScale never had to move.
    assert 12 in client.request_sizes and 6 in client.request_sizes
    assert {t for _, t in client.requests} == {4}


def test_single_failing_month_escalates_tile_scale(notebook):
    """One heavy month is retried with more tiling instead of failing the group."""
    heavy = int(pd.Timestamp('2018-03-01').value // 10 ** 6)

    def fail(months, tile_scale):
        if heavy in months and tile_scale < 8:
            return 'Computation timed out.'
        return None

    ns = _load(notebook, fail=fail)
    client = ns['_client']
    months = _months(6)

    df = ns['aoi_monthly_dataframe'](image_for_month(client), months, 1000, 'AOI',
                                     months_per_request=6)

    assert len(df) == 6
    assert list(df['month']) == list(pd.to_datetime(months))
    # The lone month is the only thing that needed a bigger tileScale.
    escalated = [(m, t) for m, t in client.requests if t > 4]
    assert escalated == [((heavy,), 8)]


def test_single_month_gives_up_at_the_maximum_tile_scale(notebook):
    """A month that fails at every tileScale raises rather than looping."""
    def fail(months, tile_scale):
        return 'User memory limit exceeded.'

    ns = _load(notebook, fail=fail)
    client = ns['_client']

    with pytest.raises(FakeEEException, match='memory limit'):
        ns['aoi_monthly_dataframe'](image_for_month(client), _months(1), 1000, 'AOI')

    assert [t for _, t in client.requests] == [4, 8, 16]


def test_non_capacity_errors_are_not_retried(notebook):
    """A real bug (missing band, bad geometry) must surface on the first request."""
    def fail(months, tile_scale):
        return "Image.select: Pattern 'temperature_2m' did not match any bands."

    ns = _load(notebook, fail=fail)
    client = ns['_client']

    with pytest.raises(FakeEEException, match='did not match any bands'):
        ns['aoi_monthly_dataframe'](image_for_month(client), _months(24), 1000, 'AOI')

    assert len(client.requests) == 1, 'a non-capacity error must not be retried'


def test_capacity_error_classifier_covers_the_reported_failure(notebook):
    ns = _load(notebook)
    is_capacity = ns['_is_ee_capacity_error']
    for message in ('User memory limit exceeded.',
                    'Computation timed out.',
                    'Too many concurrent aggregations.'):
        assert is_capacity(FakeEEException(message)), message
    assert not is_capacity(FakeEEException('Image.load: asset not found'))


def test_no_months_returns_an_empty_month_indexed_frame(notebook):
    ns = _load(notebook)
    client = ns['_client']

    df = ns['aoi_monthly_dataframe'](image_for_month(client), [], 1000, 'AOI')

    assert list(df.columns) == ['month']
    assert len(df) == 0
    assert client.requests == []


# ---------------------------------------------------------------------------
# What the fix must NOT change: covariates and the cache signature
# ---------------------------------------------------------------------------

def test_climate_covariate_definitions_are_unchanged(notebook):
    """Batching is a transport change; the bands themselves must be untouched."""
    source = _function_source(notebook, 'climate_image_for_month')
    names = set(re.findall(r'\.rename\("([a-z0-9_]+)"\)', source))
    # Antecedent / intensity band names are built by f-string and stay templated.
    templated = set(re.findall(r'\.rename\(f"([a-z0-9_{}.()\[\]]+)"\)', source))
    assert names <= CLIMATE_BAND_NAMES, f'unexpected climate band(s): {sorted(names - CLIMATE_BAND_NAMES)}'
    assert {'rain_chirps_mm', 'wind_u_ms', 'wind_v_ms', 'wind_speed_ms'} <= names
    assert 'rain_chirps_{d}d_mm' in templated


def test_cache_signature_does_not_include_the_batching_knob(notebook):
    """Adding the batch size to the fingerprint would invalidate every cached
    group and force a full rebuild -- exactly what this fix must avoid."""
    source = _function_source(notebook, '_ee_cache_signature')
    keys = set(re.findall(r'^\s{8}"([a-z0-9_]+)":', source, re.M))
    assert 'ee_climate_months_per_request' not in keys
    assert 'ee_tile_scale' not in keys
    unexpected = keys - CACHE_SIGNATURE_KEYS - CACHE_SIGNATURE_OPTIONAL_KEYS
    assert not unexpected, f'cache signature gained {sorted(unexpected)}'
    assert CACHE_SIGNATURE_KEYS <= keys, f'cache signature lost {sorted(CACHE_SIGNATURE_KEYS - keys)}'


def test_cache_signature_tracks_dates_aoi_and_resolution_but_not_the_month_list(notebook):
    """Only dates / AOI / resolution (and the EE_* knobs) may invalidate the cache.

    Keying the signature on the panel's exact month set meant one added or filtered
    month discarded every already-extracted month and re-ran hours of Earth Engine
    work. The dates window, AOI and resolution stay in the signature -- they DO make
    cached values wrong -- while the month list moves to the manifest so a later run
    can top up only the months it is missing.
    """
    source = _function_source(notebook, '_ee_cache_signature')
    keys = set(re.findall(r'^\s{8}"([a-z0-9_]+)":', source, re.M))
    assert 'months' not in keys, 'the month list must not invalidate the whole cache'
    for required in ('test_start', 'test_end', 'aoi_bbox_wgs84', 'grid_bounds',
                     'cell_size_m', 'panel_crs'):
        assert required in keys, f'cache signature must still track {required}'


def test_cached_months_are_recorded_and_topped_up_incrementally(notebook):
    """The manifest records which months are cached; only missing ones are extracted."""
    code = _code(notebook)
    assert 'def _months_in_cache(' in code
    assert 'def _append_months(' in code
    assert '"months": [str(m.date()) for m in _cached_months],' in code
    assert 'missing_months = sorted(_panel_months_ts - set(cached_months or set()))' in code
    assert 'grid, missing_months, only_groups=_month_groups' in code


def test_monthly_climate_is_still_an_independently_retried_group(notebook):
    """A cached run missing only monthly_climate must re-attempt that group alone."""
    code = _code(notebook)
    assert '"monthly_climate",' in code
    assert 'def extract_ee_covariates(grid_gdf, months, only_groups=None):' in code
    assert 'only_groups=missing' in code
    assert 'aoi_monthly_dataframe(climate_image_for_month, months, EE_CLIMATE_AOI_SCALE, aoi)' in code
