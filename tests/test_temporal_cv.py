"""Tests for winam_diagnostics.temporal_cv (Workstream A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winam_diagnostics import temporal_cv as tcv

S2_CLASS_NAMES = {0: "Open water", 1: "LEV", 2: "Floating plants", 3: "Surface algae"}
S2_PREDICTORS = ["NDVI", "NDMI", "dist_to_shore_m", "nir_glcm_entropy_w5"]

# The metric columns temporal CV must return with mean/std, matching spatial CV.
EXPECTED_METRICS = [
    "accuracy", "balanced_accuracy", "kappa", "macro_f1",
    "weighted_f1", "floating_f1",
]


def test_derive_time_groups_lodo_and_loyo():
    dates = ["2020-01-15", "2020-02-15", "2019-03-15", "2023-03-15"]
    lodo = tcv.derive_time_groups(dates, scheme="LODO")
    loyo = tcv.derive_time_groups(dates, scheme="LOYO")
    assert list(lodo) == dates
    assert list(loyo) == [2020, 2020, 2019, 2023]


def test_temporal_cv_returns_expected_columns(s2_frame, small_candidates):
    res = tcv.run_temporal_cv_diagnostic(
        s2_frame, S2_PREDICTORS, S2_CLASS_NAMES, small_candidates,
        sensor_name="S2 test", scheme="LODO", verbose=False,
    )
    assert res is not None
    # One row per candidate model.
    assert set(res["model"]) == set(small_candidates)
    # All required metric columns present as mean/std.
    for m in EXPECTED_METRICS:
        assert f"{m}_mean" in res.columns
        assert f"{m}_std" in res.columns
    # Bookkeeping columns present.
    for col in ["scheme", "n_folds_evaluated", "n_folds_degenerate", "n_folds_skipped"]:
        assert col in res.columns
    assert (res["scheme"] == "LODO").all()
    # 6 synthetic dates -> 6 LODO folds evaluated per model.
    assert (res["n_folds_evaluated"] == 6).all()
    # Separable synthetic classes -> materially better than chance.
    assert res["balanced_accuracy_mean"].max() > 0.5


def test_loyo_skips_cleanly_on_single_year(s2_frame, small_candidates, capsys):
    # s2_frame is all 2020 -> LOYO should skip and return None with a message.
    res = tcv.run_temporal_cv_diagnostic(
        s2_frame, S2_PREDICTORS, S2_CLASS_NAMES, small_candidates,
        sensor_name="S2 test", scheme="LOYO", verbose=True,
    )
    assert res is None
    out = capsys.readouterr().out
    assert "LOYO skipped" in out


def test_loyo_runs_on_multi_year(multi_year_frame, small_candidates):
    res = tcv.run_temporal_cv_diagnostic(
        multi_year_frame, S2_PREDICTORS, S2_CLASS_NAMES, small_candidates,
        sensor_name="S2 test", scheme="LOYO", verbose=False,
    )
    assert res is not None
    # 3 years -> 3 LOYO folds.
    assert (res["n_folds_evaluated"] == 3).all()
    assert (res["scheme"] == "LOYO").all()


def test_degenerate_fold_does_not_crash(small_candidates):
    # Build a frame where one date is missing the floating class entirely, so a
    # LODO test fold is degenerate. It must be scored (not skipped) and flagged.
    rows = []
    dates = ["2020-01-15", "2020-02-15", "2020-03-15"]
    rng = np.random.default_rng(0)
    import json
    for i in range(90):
        date = dates[i % 3]
        # Date 3 only ever gets classes 0 and 1 (no floating class 2/3).
        if date == "2020-03-15":
            c = i % 2
        else:
            c = i % 4
        row = {
            "class": int(c), "dominant_class": "x", "date": date,
            ".geo": json.dumps({"type": "Point", "coordinates": [34.5, -0.3]}),
        }
        for j, p in enumerate(S2_PREDICTORS):
            row[p] = 2.0 * c + rng.normal(0, 0.3)
        rows.append(row)
    df = pd.DataFrame(rows)

    res = tcv.run_temporal_cv_diagnostic(
        df, S2_PREDICTORS, S2_CLASS_NAMES, small_candidates,
        sensor_name="S2 test", scheme="LODO", verbose=False,
    )
    assert res is not None
    # At least one degenerate fold flagged, and it did not crash.
    assert res["n_folds_degenerate"].max() >= 1
    assert res["n_folds_evaluated"].max() >= 1


def test_evaluate_temporal_cv_raises_on_single_group(s2_frame, small_candidates):
    X = s2_frame[S2_PREDICTORS].to_numpy()
    y = s2_frame["class"].to_numpy()
    groups = np.zeros(len(y))  # single group
    with pytest.raises(ValueError):
        tcv.evaluate_with_temporal_cv(
            X, y, groups, small_candidates, labels=sorted(S2_CLASS_NAMES),
            scheme="LODO",
        )


def test_build_scheme_comparison_joins_and_orders(s2_frame, small_candidates):
    lodo = tcv.run_temporal_cv_diagnostic(
        s2_frame, S2_PREDICTORS, S2_CLASS_NAMES, small_candidates,
        sensor_name="S2 test", scheme="LODO", verbose=False,
    )
    # Fake a "spatial" table by reusing the same columns.
    spatial = lodo.copy()
    long_df, wide = tcv.build_scheme_comparison(
        {"spatial": spatial, "LODO": lodo, "LOYO": None},
        sensor_name="S2 test",
        metrics=("floating_f1", "balanced_accuracy", "kappa"),
    )
    assert not long_df.empty
    # LOYO was None -> dropped; only spatial + LODO present.
    assert set(long_df["scheme"]) == {"spatial", "LODO"}
    assert set(long_df["metric"]) == {"floating_f1", "balanced_accuracy", "kappa"}
    # Wide table: rows = models, columns grouped by (metric, scheme).
    assert list(wide.index) == sorted(small_candidates) or set(wide.index) == set(small_candidates)
    assert ("floating_f1", "spatial") in wide.columns
    assert ("kappa", "LODO") in wide.columns


def test_loyo_caveat_text():
    assert "PESSIMISTIC" in tcv.LOYO_CAVEAT
    assert "error-targeted" in tcv.LOYO_CAVEAT or "error" in tcv.LOYO_CAVEAT
