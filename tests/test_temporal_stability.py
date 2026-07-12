"""Tests for the temporal-stability feature scaffold (Workstream B4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from winam_diagnostics import temporal_cv as tcv


def _stack():
    """A synthetic multi-date stack for two locations.

    point A: steady VH, never floating.
    point B: volatile VH, floating in two distinct months.
    """
    rows = []
    dates = ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"]
    vh_a = [-15.0, -15.1, -14.9, -15.0]       # steady
    vh_b = [-20.0, -8.0, -18.0, -6.0]          # volatile
    ndvi_a = [0.10, 0.11, 0.09, 0.10]          # steady optical
    ndvi_b = [0.10, 0.70, 0.20, 0.65]          # volatile optical
    cls_a = [0, 0, 0, 0]
    cls_b = [2, 2, 0, 2]                        # floating in Jan, Feb, Apr -> 3 months
    for d, va, vb, na, nb, ca, cb in zip(dates, vh_a, vh_b, ndvi_a, ndvi_b, cls_a, cls_b):
        rows.append({"point_id": "A", "date": d, "VH": va, "NDVI": na, "class": ca})
        rows.append({"point_id": "B", "date": d, "VH": vb, "NDVI": nb, "class": cb})
    return pd.DataFrame(rows)


def test_temporal_stability_features_keys():
    feats = tcv.temporal_stability_features("A", _stack())
    assert set(feats) == set(tcv.DEFAULT_TEMPORAL_STABILITY_FEATURES)


def test_volatile_point_has_higher_variability():
    stack = _stack()
    a = tcv.temporal_stability_features("A", stack)
    b = tcv.temporal_stability_features("B", stack)
    # Volatile point B has larger VH temporal std and optical-index std.
    assert b["vh_temporal_std"] > a["vh_temporal_std"]
    assert b["optical_index_temporal_std"] > a["optical_index_temporal_std"]
    # VH CV is std / |mean| and finite for both.
    assert np.isfinite(a["vh_temporal_cv"])
    assert np.isfinite(b["vh_temporal_cv"])


def test_months_present_as_floating_counts_distinct_months():
    stack = _stack()
    a = tcv.temporal_stability_features("A", stack)
    b = tcv.temporal_stability_features("B", stack)
    assert a["months_present_as_floating"] == 0
    # B is floating in Jan, Feb and Apr -> 3 distinct months.
    assert b["months_present_as_floating"] == 3


def test_rolling_window_limits_history():
    stack = _stack()
    # With window=2, only the last two observations count.
    b_full = tcv.temporal_stability_features("B", stack)
    b_win = tcv.temporal_stability_features("B", stack, window=2)
    # Last two dates for B are floating in Apr only (Mar class 0) -> 1 month.
    assert b_win["months_present_as_floating"] == 1
    assert b_full["months_present_as_floating"] == 3


def test_missing_point_returns_safe_defaults():
    feats = tcv.temporal_stability_features("does-not-exist", _stack())
    assert feats["months_present_as_floating"] == 0
    assert np.isnan(feats["vh_temporal_std"])


def test_empty_stack_returns_safe_defaults():
    feats = tcv.temporal_stability_features("A", pd.DataFrame())
    assert feats["months_present_as_floating"] == 0
    assert np.isnan(feats["vh_temporal_std"])
