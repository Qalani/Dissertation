"""Diagnostic, additive companions to the Winam Gulf WH classifier notebook.

``Classifier_Full_Stack_PostExport_TimeSeries_v4.ipynb`` deploys a Sentinel-1/2
pixel classifier for water hyacinth in the Winam Gulf of Lake Victoria. This
package holds the *diagnostic* extensions the notebook wires in through thin cells:

* :mod:`winam_diagnostics.temporal_cv` — temporal cross-validation (LODO / LOYO)
  comparable to the existing spatial CV, plus the scaffolded temporal-stability
  feature extractor.
* :mod:`winam_diagnostics.shoreline` — shore-distance computation and near-shore
  stratified accuracy (LEV<->Floating confusion) using out-of-fold predictions.
* :mod:`winam_diagnostics.area_envelope` — hard vs confidence-weighted (soft) WH
  area envelope per date.

Nothing here imports Earth Engine, mounts Google Drive, or hits the network, so
the whole package imports and is unit-tested offline. It never retrains or changes
which model the notebook deploys — these are read-only measurements alongside it.

The AOI metric CRS is **EPSG:32736 (UTM 36S)** — Lake Victoria, not the UK.
"""

from __future__ import annotations

from . import area_envelope, shoreline, temporal_cv
from .area_envelope import (
    accumulate_soft_hard_from_geotiffs,
    compute_area_envelope,
    plot_area_envelope_timeseries,
    shade_area_envelope,
    soft_hard_floating_area,
)
from .shoreline import (
    backfill_dist_to_shore,
    bin_shore_distance,
    compute_dist_to_shore,
    oof_predictions_spatial_cv,
    print_near_shore_summary,
    stratified_accuracy_by_shore,
)
from .temporal_cv import (
    build_scheme_comparison,
    derive_time_groups,
    evaluate_with_temporal_cv,
    print_loyo_caveat,
    run_temporal_cv_diagnostic,
    temporal_stability_features,
)

DEFAULT_METRIC_CRS = "EPSG:32736"

__all__ = [
    "area_envelope",
    "shoreline",
    "temporal_cv",
    # temporal_cv
    "evaluate_with_temporal_cv",
    "run_temporal_cv_diagnostic",
    "derive_time_groups",
    "build_scheme_comparison",
    "print_loyo_caveat",
    "temporal_stability_features",
    # shoreline
    "compute_dist_to_shore",
    "backfill_dist_to_shore",
    "oof_predictions_spatial_cv",
    "stratified_accuracy_by_shore",
    "bin_shore_distance",
    "print_near_shore_summary",
    # area_envelope
    "soft_hard_floating_area",
    "accumulate_soft_hard_from_geotiffs",
    "compute_area_envelope",
    "shade_area_envelope",
    "plot_area_envelope_timeseries",
    "DEFAULT_METRIC_CRS",
]
