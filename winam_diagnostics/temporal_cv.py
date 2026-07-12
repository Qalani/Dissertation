"""Temporal cross-validation diagnostics for the Winam Gulf WH classifier.

This module is a *diagnostic, additive* companion to
``Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb``. It mirrors the
notebook's spatial-CV evaluator (`evaluate_with_spatial_cv`, cell 27) but splits
the training points by *time* instead of by spatial block:

* **LODO** — Leave-One-Date-Out: each held-out fold is a single acquisition date.
* **LOYO** — Leave-One-Year-Out: each held-out fold is a calendar year.

The point is to answer "does classifier skill drop out-of-date / out-of-year?"
with a table that is directly comparable to the spatial-CV table: the same metric
columns (accuracy, balanced_accuracy, kappa, macro_f1, weighted_f1, floating_f1
and the floating precision/recall) are aggregated to per-fold ``*_mean`` / ``*_std``.

Design constraints (see the notebook task brief):

* NO Earth Engine, NO Google Drive, NO network — imports and runs offline.
* Do not reimplement the candidate models: callers pass in the exact ``candidates``
  dict built by the notebook's ``make_candidate_classifiers`` so the same
  estimators are evaluated.
* The metric block is a faithful port of the notebook's ``_score_split`` (cell 27)
  so temporal and spatial tables line up column-for-column.

The temporal-stability *feature* scaffold (Workstream B4) also lives here because
it is a temporal construct; it is gated behind a flag in the notebook and needs a
multi-date predictor stack that is not available offline (see the TODO in
``temporal_stability_features``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut

# Metric columns aggregated for every fold. Kept identical (and in the same order)
# to the notebook's spatial-CV aggregation so the two tables can be joined.
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "kappa",
    "macro_f1",
    "weighted_f1",
    "floating_precision",
    "floating_recall",
    "floating_f1",
]

# Default WH / floating-plant class code (matches FLOATING_CLASS_CODE in the notebook).
FLOATING_CLASS_CODE = 2


def score_classification(y_true, y_pred, labels, floating_class_code=FLOATING_CLASS_CODE):
    """Return the notebook's per-split metric dict for one fold.

    This mirrors the metric block of ``_score_split`` in cell 27 (minus the
    ``model.fit`` step, which the temporal evaluator handles per fold) so that
    temporal-CV rows carry exactly the columns the spatial-CV rows carry.
    ``labels`` fixes the class set for kappa/confusion so degenerate folds that
    are missing a class still score without crashing.
    """
    p, r, floating_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[floating_class_code], zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "floating_precision": p[0],
        "floating_recall": r[0],
        "floating_f1": floating_f1[0],
        "floating_support": int(support[0]),
    }


def derive_time_groups(dates, scheme="LODO"):
    """Derive fold-group ids from a sequence of dates.

    * ``scheme='LOYO'`` -> calendar year (int) per row.
    * ``scheme='LODO'`` -> the full ``YYYY-MM-DD`` date string per row.

    Accepts anything ``pandas.to_datetime`` understands ('YYYY-MM-DD' strings,
    datetimes, ...). Returns a numpy array aligned with ``dates``.
    """
    scheme = scheme.upper()
    parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    if scheme == "LOYO":
        return parsed.dt.year.to_numpy()
    if scheme == "LODO":
        return parsed.dt.strftime("%Y-%m-%d").to_numpy()
    raise ValueError(f"Unknown temporal CV scheme {scheme!r}; use 'LODO' or 'LOYO'.")


def evaluate_with_temporal_cv(
    X,
    y,
    time_groups,
    candidates,
    labels,
    scheme="LODO",
    sensor_name="Sensor",
    floating_class_code=FLOATING_CLASS_CODE,
):
    """Leave-One-Group-Out temporal CV, mirroring ``evaluate_with_spatial_cv``.

    Parameters
    ----------
    X, y : array-like
        Predictor matrix and integer class labels.
    time_groups : array-like
        Per-row temporal group id (year for LOYO, date for LODO). See
        :func:`derive_time_groups`.
    candidates : dict[str, estimator]
        Name -> unfitted sklearn estimator. Pass the notebook's
        ``make_candidate_classifiers(...)`` output so the same models are used.
    labels : sequence[int]
        Fixed class label set (e.g. ``sorted(class_names)``). Used for kappa and
        to keep degenerate folds scorable.
    scheme : str
        'LODO' or 'LOYO' (used only for labelling the output rows).

    Returns
    -------
    pandas.DataFrame
        One row per model with ``{metric}_mean`` / ``{metric}_std`` for every
        metric in :data:`METRIC_COLUMNS`, plus ``n_folds_evaluated``,
        ``n_folds_degenerate`` (test fold missing >=1 class) and
        ``n_folds_skipped`` (empty fold or a model that failed to fit, e.g. a
        train fold with a single class).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    time_groups = np.asarray(time_groups)
    labels = list(labels)
    label_set = set(labels)

    unique_groups = np.unique(time_groups)
    if len(unique_groups) < 2:
        raise ValueError(
            f"{sensor_name} {scheme}: need >=2 distinct time groups to run "
            f"Leave-One-Group-Out, found {len(unique_groups)}."
        )

    splitter = LeaveOneGroupOut()
    rows = []
    for model_name, template in candidates.items():
        per_fold = []
        n_degenerate = 0
        n_skipped = 0
        for train_idx, test_idx in splitter.split(X, y, groups=time_groups):
            if len(train_idx) == 0 or len(test_idx) == 0:
                n_skipped += 1
                continue
            y_train = y[train_idx]
            y_test = y[test_idx]
            # A fold whose test set is missing one of the fixed labels is
            # degenerate: it is scored (fixed labels + zero_division=0) but flagged.
            if not label_set.issubset(set(np.unique(y_test))):
                n_degenerate += 1
            try:
                model = clone(template)
                model.fit(X[train_idx], y_train)
                y_pred = model.predict(X[test_idx])
            except Exception:
                # e.g. a train fold with a single class that a booster rejects.
                # Skip that fold rather than crash the whole diagnostic.
                n_skipped += 1
                continue
            per_fold.append(score_classification(y_test, y_pred, labels, floating_class_code))

        per_fold_df = pd.DataFrame(per_fold)
        agg = {
            "sensor": sensor_name,
            "model": model_name,
            "scheme": scheme,
            "n_folds_evaluated": len(per_fold_df),
            "n_folds_degenerate": int(n_degenerate),
            "n_folds_skipped": int(n_skipped),
        }
        for col in METRIC_COLUMNS:
            if col in per_fold_df.columns and len(per_fold_df):
                agg[f"{col}_mean"] = per_fold_df[col].mean()
                agg[f"{col}_std"] = per_fold_df[col].std()
            else:
                agg[f"{col}_mean"] = np.nan
                agg[f"{col}_std"] = np.nan
        rows.append(agg)

    return pd.DataFrame(rows)


def run_temporal_cv_diagnostic(
    df,
    predictors,
    class_names,
    candidates,
    sensor_name="Sensor",
    scheme="LODO",
    date_column="date",
    class_column="class",
    floating_class_code=FLOATING_CLASS_CODE,
    sort_metric="floating_f1",
    verbose=True,
):
    """Frame-in / results-DataFrame-out wrapper, mirroring ``run_spatial_cv_diagnostic``.

    Validates the required columns, drops NaN predictor/label/date rows, derives
    temporal groups and runs :func:`evaluate_with_temporal_cv`. For ``LOYO`` it
    returns ``None`` (and prints a clear message) when fewer than two distinct
    years are present, which is the single-year (no-corrections) case.
    """
    scheme = scheme.upper()
    required = list(predictors) + [class_column, date_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{sensor_name} temporal CV is missing columns: {missing}")

    ready = df.dropna(subset=required).copy()
    if ready.empty:
        raise ValueError(f"{sensor_name}: no usable rows for temporal CV after dropping NaNs.")

    groups = derive_time_groups(ready[date_column], scheme=scheme)
    n_groups = len(np.unique(groups))

    if scheme == "LOYO" and n_groups < 2:
        if verbose:
            print(
                f"{sensor_name}: LOYO skipped — only {n_groups} distinct year(s) present "
                "(cross-year points come only from the manual corrections). Enable manual "
                "corrections to get a multi-year temporal split."
            )
        return None
    if n_groups < 2:
        if verbose:
            print(
                f"{sensor_name}: {scheme} skipped — only {n_groups} distinct time group(s) present."
            )
        return None

    X = ready[list(predictors)].astype(np.float32).to_numpy()
    y = ready[class_column].astype(int).to_numpy()
    labels = sorted(class_names.keys())

    if verbose:
        print(
            f"\n=== {sensor_name}: temporal CV ({scheme}) — "
            f"{len(ready)} rows across {n_groups} {'year' if scheme == 'LOYO' else 'date'} group(s) ==="
        )

    results = evaluate_with_temporal_cv(
        X, y, groups, candidates, labels, scheme=scheme,
        sensor_name=sensor_name, floating_class_code=floating_class_code,
    )
    sort_col = f"{sort_metric}_mean"
    if sort_col in results.columns:
        results = results.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return results


LOYO_CAVEAT = (
    "CAVEAT — LOYO is a PESSIMISTIC bound, not an unbiased out-of-year estimate. "
    "The 2020 points are the random supervised labels, but the cross-year (2019 & 2023) "
    "points come ONLY from the manual-correction GPKGs, which were digitised at "
    "low-confidence / error locations. Holding out a whole year therefore tests the "
    "model on an error-targeted, non-random sample, so a LOYO skill drop understates "
    "true out-of-year skill on ordinary points."
)


def print_loyo_caveat(prefix=""):
    """Print the standing LOYO interpretation caveat (Workstream A5)."""
    print(f"{prefix}{LOYO_CAVEAT}")


def build_scheme_comparison(
    results_by_scheme,
    sensor_name="Sensor",
    metrics=("floating_f1", "balanced_accuracy", "kappa"),
):
    """Join spatial / random / LODO / LOYO results into one comparison (Workstream A4).

    Parameters
    ----------
    results_by_scheme : dict[str, DataFrame | None]
        Ordered mapping of scheme label -> a CV results frame (each with a
        ``model`` column and ``{metric}_mean`` columns). ``None`` / empty entries
        are skipped, so LOYO can be dropped in on the single-year case.
    metrics : sequence[str]
        Metrics to line up across schemes.

    Returns
    -------
    (long_df, wide_df)
        ``long_df`` is tidy (model, metric, scheme, value) and easy to save;
        ``wide_df`` pivots to rows=models, columns grouped by (metric, scheme) in
        the given order — the at-a-glance "does skill drop out-of-date/out-of-year?"
        table.
    """
    metrics = list(metrics)
    frames = []
    scheme_order = []
    for scheme_label, res in results_by_scheme.items():
        if res is None or len(res) == 0:
            continue
        scheme_order.append(scheme_label)
        present = [m for m in metrics if f"{m}_mean" in res.columns]
        sub = res[["model"] + [f"{m}_mean" for m in present]].copy()
        sub = sub.melt(id_vars="model", var_name="metric", value_name="value")
        sub["metric"] = sub["metric"].str.replace("_mean", "", regex=False)
        sub["scheme"] = scheme_label
        frames.append(sub)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    long_df.insert(0, "sensor", sensor_name)

    wide = long_df.pivot_table(
        index="model", columns=["metric", "scheme"], values="value"
    )
    # Order columns metric-major (in requested order), scheme-minor (in input order).
    ordered_cols = [
        (m, s)
        for m in metrics
        for s in scheme_order
        if (m, s) in wide.columns
    ]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["metric", "scheme"]))
    return long_df, wide


# ---------------------------------------------------------------------------
# Workstream B4 — temporal-persistence discriminators (SCAFFOLD).
# ---------------------------------------------------------------------------
# These features are OPTIONAL candidate predictors. The extractor below is real
# and fully tested on synthetic data, but populating it in production needs a
# multi-date predictor stack per point (VH backscatter through time for S1;
# optical indices through time for S2) that is NOT available offline and is not
# baked into the current validated-snapshot exports.
#
# TODO (re-export requirement): to wire these into the classifier, re-export a
# per-point multi-date stack (mirroring Batch_Export.ipynb's validated snapshots)
# and sample each point's VH / optical-index time series with the EXACT
# baked-feature pattern used for the GLCM textures
# (`augment_s2_training_with_baked_features` in cell 17). Until that stack exists,
# do NOT add these columns to the production predictor list — empty columns would
# silently degrade the deployed model.

DEFAULT_TEMPORAL_STABILITY_FEATURES = (
    "vh_temporal_std",
    "vh_temporal_cv",
    "optical_index_temporal_std",
    "months_present_as_floating",
)


def temporal_stability_features(
    point_id,
    sensor_stack,
    *,
    point_id_column="point_id",
    date_column="date",
    vh_column="VH",
    optical_index_column="NDVI",
    class_column="class",
    floating_class_code=FLOATING_CLASS_CODE,
    window=None,
):
    """Compute temporal-persistence discriminators for one location.

    Water hyacinth mats drift and reform, whereas rooted LEV persists in place;
    per-location temporal variability of backscatter/indices plus a
    months-present-as-floating count therefore help separate the two. This is the
    real extractor behind Workstream B4.

    Parameters
    ----------
    point_id : hashable
        Location id to extract features for.
    sensor_stack : pandas.DataFrame
        Long-format multi-date stack with one row per (point, date). Must contain
        ``point_id_column`` and ``date_column``; ``vh_column`` (S1),
        ``optical_index_column`` (S2) and ``class_column`` are used when present.
    window : int, optional
        If given, keep only the most recent ``window`` observations (rolling
        window) before computing the features; otherwise use the whole series.

    Returns
    -------
    dict
        ``vh_temporal_std``, ``vh_temporal_cv`` (std/|mean|), ``optical_index_temporal_std``
        and ``months_present_as_floating``. Missing inputs yield ``nan`` (or 0 for
        the count) rather than fabricated values.
    """
    out = {
        "vh_temporal_std": np.nan,
        "vh_temporal_cv": np.nan,
        "optical_index_temporal_std": np.nan,
        "months_present_as_floating": 0,
    }
    if sensor_stack is None or len(sensor_stack) == 0 or point_id_column not in sensor_stack.columns:
        return out

    series = sensor_stack[sensor_stack[point_id_column] == point_id].copy()
    if series.empty:
        return out
    if date_column in series.columns:
        series[date_column] = pd.to_datetime(series[date_column], errors="coerce")
        series = series.sort_values(date_column)
    if window is not None and window > 0:
        series = series.tail(int(window))

    if vh_column in series.columns:
        vh = pd.to_numeric(series[vh_column], errors="coerce").dropna().to_numpy()
        if vh.size >= 2:
            std = float(np.std(vh, ddof=1))
            out["vh_temporal_std"] = std
            mean = float(np.mean(vh))
            out["vh_temporal_cv"] = std / abs(mean) if mean != 0 else np.nan

    if optical_index_column in series.columns:
        idx = pd.to_numeric(series[optical_index_column], errors="coerce").dropna().to_numpy()
        if idx.size >= 2:
            out["optical_index_temporal_std"] = float(np.std(idx, ddof=1))

    if class_column in series.columns and date_column in series.columns:
        floating = series[pd.to_numeric(series[class_column], errors="coerce") == floating_class_code]
        # Distinct months in which this location was classified as floating.
        months = floating[date_column].dropna().dt.to_period("M").unique()
        out["months_present_as_floating"] = int(len(months))

    return out
