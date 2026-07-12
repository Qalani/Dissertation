"""Shared synthetic fixtures mimicking the classifier's training-frame schema.

No Earth Engine, no Drive, no network — everything is generated in-memory so the
diagnostics package can be exercised offline. The synthetic frames match the real
schema: ``class`` (int), ``dominant_class`` (raw), ``date`` ('YYYY-MM-DD'),
``.geo`` (GeoJSON point), ``Location``, predictor columns and ``spatial_block``.
Coordinates sit in the Winam Gulf of Lake Victoria (lon ~34.2-34.9, lat ~-0.55-0).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# Winam Gulf class maps (see notebook cell 7).
S2_CLASS_NAMES = {0: "Open water", 1: "LEV", 2: "Floating plants", 3: "Surface algae"}
S1_CLASS_NAMES = {0: "Open water / surface algae", 1: "LEV", 2: "Floating plants"}
S2_RAW = {0: "W", 1: "T", 2: "F2", 3: "A"}
S1_RAW = {0: "W", 1: "T", 2: "F1"}

# A coarse Winam Gulf lon/lat box (matches the notebook AOI rectangle).
LON_MIN, LON_MAX = 34.2046673170698, 34.9
LAT_MIN, LAT_MAX = -0.55, 0.0


def _geo(lon, lat):
    """A GeoJSON Point string exactly like the CSV ``.geo`` column."""
    return json.dumps({"type": "Point", "coordinates": [float(lon), float(lat)]})


def make_synthetic_frame(
    n=240,
    class_names=S2_CLASS_NAMES,
    raw_map=S2_RAW,
    dates=("2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15",
           "2020-05-15", "2020-06-15"),
    predictors=("NDVI", "NDMI", "dist_to_shore_m", "nir_glcm_entropy_w5"),
    years=None,
    seed=0,
):
    """Build a schema-matching synthetic training frame with separable classes.

    ``predictors`` carry a class-dependent mean so classifiers score well above
    chance, which lets the CV tests assert sensible (not just non-crashing) output.
    ``years`` overrides ``dates`` with one date per given year (for LOYO tests).
    """
    rng = np.random.default_rng(seed)
    classes = sorted(class_names)
    if years is not None:
        dates = tuple(f"{y}-03-15" for y in years)

    rows = []
    for i in range(n):
        c = classes[i % len(classes)]
        lon = rng.uniform(LON_MIN, LON_MAX)
        lat = rng.uniform(LAT_MIN, LAT_MAX)
        date = dates[i % len(dates)]
        row = {
            "class": int(c),
            "dominant_class": raw_map[c],
            "date": date,
            ".geo": _geo(lon, lat),
            "Location": "Winam",
            "lon": lon,
            "lat": lat,
        }
        for j, p in enumerate(predictors):
            # Separable: each class shifted, plus noise. dist_to_shore_m kept positive.
            base = 100.0 * (j + 1) if p == "dist_to_shore_m" else 0.0
            row[p] = base + 2.0 * c + rng.normal(0, 0.3)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Spatial block id exactly as the notebook computes it (cell 27).
    block_deg = 0.1
    lon_bin = np.floor(df["lon"] / block_deg).astype(np.int64)
    lat_bin = np.floor(df["lat"] / block_deg).astype(np.int64)
    df["spatial_block"] = (lon_bin * 100000 + lat_bin).astype(np.int64)
    return df


@pytest.fixture
def s2_frame():
    return make_synthetic_frame(
        n=240, class_names=S2_CLASS_NAMES, raw_map=S2_RAW, seed=1
    )


@pytest.fixture
def s1_frame():
    return make_synthetic_frame(
        n=180,
        class_names=S1_CLASS_NAMES,
        raw_map=S1_RAW,
        predictors=("VH_p5", "VH_corrected", "VH_smooth"),
        seed=2,
    )


@pytest.fixture
def multi_year_frame():
    """A frame spanning three years, for LOYO tests."""
    return make_synthetic_frame(
        n=180, class_names=S2_CLASS_NAMES, raw_map=S2_RAW,
        years=(2019, 2020, 2023), seed=3,
    )


@pytest.fixture
def small_candidates():
    """A tiny, fast candidate set (subset of the notebook's models)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=40, random_state=42, n_jobs=1
        ),
        "Logistic Regression baseline": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced"),
        ),
    }
