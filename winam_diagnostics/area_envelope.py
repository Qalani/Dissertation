"""Hard vs confidence-weighted (soft) WH area envelope (Workstream B3).

Diagnostic, additive, offline (no Earth Engine / Drive / network). The classified
WH area is reported two ways per date:

* **hard** — every pixel classified as floating WH counts as one whole pixel
  (this is what ``area_by_class_from_geotiff`` produces).
* **soft** — each floating pixel is down-weighted by its winning-class
  probability, so low-confidence pixels contribute less. This is the classifier's
  own measurement error made explicit.

The soft definition mirrors the panel notebook's hard-vs-soft cover logic
(``winam_wh_spatial_panel_predictive_ml.ipynb``, "Classifier accuracy and response
measurement error"): there, ``wh_cover_soft`` is the probability-weighted cover and
the panel-total soft area is ``sum(wh_cover_soft * area)``. Per pixel that is
exactly ``sum(P_floating)`` over the hard WH pixels, which is what
:func:`soft_hard_floating_area` computes. ``soft <= hard`` always, and
``soft / hard`` is the mean winning-class confidence over the WH pixels.

This module never touches ``confidence_masked_area_by_class`` — that threshold-based
function is left intact. The envelope is a separate, additive output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOATING_CLASS_CODE = 2
CLASS_NODATA_VALUE = 255
PROBA_NODATA_VALUE = 255
# Probability rasters are written on the 0..100 uint8 scale (see the notebook's
# confidence-masked area code), so divide by 100 to recover a 0..1 probability.
PROBA_SCALE = 100.0


def soft_hard_floating_area(
    class_arr,
    proba_arr,
    pixel_area_m2,
    floating_class_code=FLOATING_CLASS_CODE,
    class_nodata=CLASS_NODATA_VALUE,
    proba_nodata=PROBA_NODATA_VALUE,
    proba_scale=PROBA_SCALE,
):
    """Hard, soft and gap WH area from a class array + winning-class probability array.

    Pure-numpy core (the tested unit). ``proba_arr`` holds the winning-class
    probability on the 0..``proba_scale`` scale. A pixel counts toward WH area only
    where both arrays are valid (not nodata) and the class equals
    ``floating_class_code``. Probabilities are clipped to [0, 1] so ``soft <= hard``
    and ``soft / hard in (0, 1]`` hold exactly.

    Returns a dict with hard/soft/gap areas in hectares, the soft/hard ratio, the
    mean winning-class confidence over WH pixels, and the WH pixel count.
    """
    class_arr = np.asarray(class_arr)
    proba_arr = np.asarray(proba_arr)
    if class_arr.shape != proba_arr.shape:
        raise ValueError(
            f"class and probability arrays differ in shape: {class_arr.shape} vs {proba_arr.shape}"
        )

    valid = (class_arr != class_nodata) & (proba_arr != proba_nodata)
    floating = valid & (class_arr == floating_class_code)
    hard_pixels = int(floating.sum())

    prob = np.clip(proba_arr[floating].astype(np.float64) / proba_scale, 0.0, 1.0)
    soft_pixels = float(prob.sum())

    hard_area_ha = hard_pixels * pixel_area_m2 / 1e4
    soft_area_ha = soft_pixels * pixel_area_m2 / 1e4
    gap_ha = hard_area_ha - soft_area_ha
    gap_pct = (gap_ha / hard_area_ha * 100.0) if hard_area_ha > 0 else float("nan")
    soft_hard_ratio = (soft_area_ha / hard_area_ha) if hard_area_ha > 0 else float("nan")
    mean_confidence = (soft_pixels / hard_pixels) if hard_pixels > 0 else float("nan")

    return {
        "hard_pixels": hard_pixels,
        "pixel_area_m2": float(pixel_area_m2),
        "hard_area_ha": hard_area_ha,
        "soft_area_ha": soft_area_ha,
        "gap_ha": gap_ha,
        "gap_pct": gap_pct,
        "soft_hard_ratio": soft_hard_ratio,
        "mean_confidence": mean_confidence,
    }


def accumulate_soft_hard_from_geotiffs(
    class_tif,
    proba_tif,
    floating_class_code=FLOATING_CLASS_CODE,
    class_nodata=CLASS_NODATA_VALUE,
    proba_nodata=PROBA_NODATA_VALUE,
    proba_scale=PROBA_SCALE,
    staged_reader=None,
):
    """Stream a class + probability GeoTIFF pair and compute the WH area envelope.

    rasterio is imported lazily so importing this module needs no raster stack.
    ``staged_reader`` is an optional context-manager factory (e.g. the notebook's
    ``staged_drive_read``) used to stage a Drive file to local disk before block
    reads; when ``None`` the files are opened in place. Reads block-by-block to keep
    peak memory bounded, matching the notebook's streaming area functions.
    """
    import contextlib

    import rasterio

    def _open(path):
        if staged_reader is not None:
            return staged_reader(path)
        return contextlib.nullcontext(path)

    hard_pixels = 0
    soft_pixels = 0.0
    with _open(class_tif) as class_local, _open(proba_tif) as proba_local:
        with rasterio.open(class_local) as cls_src, rasterio.open(proba_local) as proba_src:
            if cls_src.shape != proba_src.shape:
                raise ValueError(
                    "Class and probability rasters have different shapes; cannot form envelope."
                )
            pixel_area_m2 = abs(cls_src.transform.a * cls_src.transform.e)
            for _, window in cls_src.block_windows(1):
                cls = cls_src.read(1, window=window)
                proba = proba_src.read(1, window=window)
                valid = (cls != class_nodata) & (proba != proba_nodata)
                floating = valid & (cls == floating_class_code)
                if not np.any(floating):
                    continue
                hard_pixels += int(floating.sum())
                prob = np.clip(proba[floating].astype(np.float64) / proba_scale, 0.0, 1.0)
                soft_pixels += float(prob.sum())

    hard_area_ha = hard_pixels * pixel_area_m2 / 1e4
    soft_area_ha = soft_pixels * pixel_area_m2 / 1e4
    gap_ha = hard_area_ha - soft_area_ha
    return {
        "hard_pixels": hard_pixels,
        "pixel_area_m2": float(pixel_area_m2),
        "hard_area_ha": hard_area_ha,
        "soft_area_ha": soft_area_ha,
        "gap_ha": gap_ha,
        "gap_pct": (gap_ha / hard_area_ha * 100.0) if hard_area_ha > 0 else float("nan"),
        "soft_hard_ratio": (soft_area_ha / hard_area_ha) if hard_area_ha > 0 else float("nan"),
        "mean_confidence": (soft_pixels / hard_pixels) if hard_pixels > 0 else float("nan"),
    }


ENVELOPE_COLUMNS = [
    "sensor",
    "method",
    "start_date",
    "end_date",
    "hard_area_ha",
    "soft_area_ha",
    "gap_ha",
    "gap_pct",
    "soft_hard_ratio",
    "mean_confidence",
]


def compute_area_envelope(
    area_rows,
    floating_class_code=FLOATING_CLASS_CODE,
    class_tif_col="classification_output",
    proba_tif_col="probability_output",
    hard_area_col="area_ha",
    staged_reader=None,
    path_exists=None,
    verbose=True,
):
    """Build the per-date WH area-envelope table from the batch area rows (B3).

    ``area_rows`` is an iterable of dict-like rows (e.g. the floating-class rows of
    the batch ``winam_full_stack_area_by_class_all_dates_<version>.csv``), each with
    ``sensor``, ``method``, ``start_date``, ``end_date``, a hard-area column and the
    per-date class / probability GeoTIFF paths. For each row with an available
    probability raster the soft area is measured from the rasters; rows without a
    probability raster fall back to ``soft = hard`` (envelope collapses) and are
    flagged with ``has_proba = False``.

    Returns a DataFrame with :data:`ENVELOPE_COLUMNS` plus ``has_proba``.
    """
    from pathlib import Path

    if path_exists is None:
        path_exists = lambda p: bool(p) and Path(p).exists()

    if isinstance(area_rows, pd.DataFrame):
        iterator = (row for _, row in area_rows.iterrows())
    else:
        iterator = iter(area_rows)

    records = []
    for row in iterator:
        get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
        class_tif = get(class_tif_col)
        proba_tif = get(proba_tif_col)
        hard_area_ha = get(hard_area_col)
        base = {
            "sensor": get("sensor"),
            "method": get("method"),
            "start_date": get("start_date"),
            "end_date": get("end_date"),
        }

        has_proba = path_exists(proba_tif) and path_exists(class_tif)
        if has_proba:
            env = accumulate_soft_hard_from_geotiffs(
                class_tif, proba_tif, floating_class_code=floating_class_code,
                staged_reader=staged_reader,
            )
            base.update({
                "hard_area_ha": env["hard_area_ha"],
                "soft_area_ha": env["soft_area_ha"],
                "gap_ha": env["gap_ha"],
                "gap_pct": env["gap_pct"],
                "soft_hard_ratio": env["soft_hard_ratio"],
                "mean_confidence": env["mean_confidence"],
                "has_proba": True,
            })
        else:
            hard = float(hard_area_ha) if hard_area_ha is not None and pd.notna(hard_area_ha) else np.nan
            base.update({
                "hard_area_ha": hard,
                "soft_area_ha": hard,  # no probability raster -> envelope collapses to hard.
                "gap_ha": 0.0,
                "gap_pct": 0.0,
                "soft_hard_ratio": 1.0 if hard > 0 else np.nan,
                "mean_confidence": np.nan,
                "has_proba": False,
            })
            if verbose:
                print(
                    f"  {base['sensor']} {base['start_date']}: no probability raster; "
                    "soft area falls back to hard (envelope collapses for this date)."
                )
        records.append(base)

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out[[*ENVELOPE_COLUMNS, "has_proba"]]
        out = out.sort_values(["sensor", "method", "start_date"]).reset_index(drop=True)
    return out


def shade_area_envelope(ax, envelope_df, date_col="mid_date", color=None, alpha=0.18, label=None):
    """Shade the hard-soft band on an existing matplotlib Axes (B3).

    Fills between ``soft_area_ha`` and ``hard_area_ha`` versus date for one
    already-selected (sensor, method) series, so the classifier's WH-area
    measurement uncertainty is drawn onto the existing floating-area time-series
    figure. Returns the Axes.
    """
    df = envelope_df.copy()
    if df.empty:
        return ax
    if date_col not in df.columns:
        # Fall back to the acquisition mid-date if the caller did not precompute it.
        start = pd.to_datetime(df["start_date"])
        end = pd.to_datetime(df["end_date"])
        df[date_col] = start + (end - start) / 2
    df = df.sort_values(date_col)
    ax.fill_between(
        df[date_col],
        df["soft_area_ha"],
        df["hard_area_ha"],
        alpha=alpha,
        color=color,
        label=label or "hard-soft envelope",
    )
    return ax


def plot_area_envelope_timeseries(
    envelope_df,
    date_col="mid_date",
    title="Winam Gulf WH area — hard vs confidence-weighted (soft) envelope",
    ax=None,
):
    """Standalone hard/soft time-series with the shaded envelope band (B3).

    matplotlib is imported lazily. Draws, per (sensor, method), the hard-area and
    soft-area lines with the band shaded between them. Returns the Axes.
    """
    import matplotlib.pyplot as plt

    df = envelope_df.copy()
    if df.empty:
        raise ValueError("Empty envelope_df; nothing to plot.")
    if date_col not in df.columns:
        start = pd.to_datetime(df["start_date"])
        end = pd.to_datetime(df["end_date"])
        df[date_col] = start + (end - start) / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    for (sensor, method), group in df.groupby(["sensor", "method"], sort=True):
        group = group.sort_values(date_col)
        (line,) = ax.plot(
            group[date_col], group["hard_area_ha"], marker="o", linewidth=1.5,
            label=f"{sensor} | {method} hard",
        )
        ax.plot(
            group[date_col], group["soft_area_ha"], linewidth=1.5, linestyle="--",
            color=line.get_color(), label=f"{sensor} | {method} soft",
        )
        shade_area_envelope(ax, group, date_col=date_col, color=line.get_color(), label=None)

    ax.set_xlabel("Date")
    ax.set_ylabel("Floating plants / water hyacinth area (ha)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax
