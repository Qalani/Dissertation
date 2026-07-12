"""Guard: the diagnostics package imports with no Earth Engine / Drive / network."""

from __future__ import annotations

import sys


def test_no_earth_engine_or_drive_on_import():
    # Fresh import side-effects only.
    for name in list(sys.modules):
        if name.startswith(("winam_diagnostics", "ee", "geemap")):
            del sys.modules[name]
    import winam_diagnostics  # noqa: F401

    banned_prefixes = ("ee", "earthengine", "geemap", "google.colab")
    leaked = [
        m for m in sys.modules
        if any(m == p or m.startswith(p + ".") for p in banned_prefixes)
    ]
    assert not leaked, f"diagnostics import pulled in banned modules: {leaked}"


def test_public_api_is_exported():
    import winam_diagnostics as wd

    for name in [
        "evaluate_with_temporal_cv",
        "run_temporal_cv_diagnostic",
        "compute_dist_to_shore",
        "stratified_accuracy_by_shore",
        "oof_predictions_spatial_cv",
        "soft_hard_floating_area",
        "compute_area_envelope",
        "temporal_stability_features",
    ]:
        assert hasattr(wd, name), f"missing public export: {name}"
