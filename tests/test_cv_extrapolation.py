"""§13-CV must not extrapolate a held-out fold, and must not resume past a fix to it.

The failure this pins down: temporal fold 1 of the ``forcing`` spec came back with a
NaN Spearman and ``pred_min == pred_max == 2.22e-16`` for all 5,400 held-out rows, so
``assert_folds_scored`` failed the run. That value is ``.Machine$double.eps`` -- the
floor mgcv clamps betar's inverse logit to -- which means the linear predictor was
driven off the scale, not that the fit had shrunk to its intercept.

A rolling-origin fold predicts months lying entirely beyond its training window, so
every ``bs='tp'`` smooth and the raw ``time_index`` trend are evaluated outside the
range they were fitted on, where both are linear and unbounded. Two things therefore
have to hold, and both are checked here:

1. the sweep holds each test covariate at the edge of its own fold's training range
   before ``predict()``, and records how much it held so the restriction is reported;
2. the checkpoint fingerprint covers the fit/predict path, not just the data -- the
   run that produced the failure resumed all 12 units from disk and refitted nothing,
   so a fix that did not invalidate those checkpoints would have been a no-op.

The R helper is executed (not just grepped) wherever Rscript is available.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "winam_wh_spatial_panel_driver_gam.ipynb"


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


@pytest.fixture(scope="module")
def cv_cell(cells):
    """The §13-CV R sweep."""
    bodies = [
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith("%%R")
        and "fit_eval <- function" in "".join(c["source"])
    ]
    assert len(bodies) == 1, f"expected one §13-CV sweep cell, found {len(bodies)}"
    return bodies[0]


@pytest.fixture(scope="module")
def all_code(cells):
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


# ---------------------------------------------------------------------
# 1. The sweep refuses to extrapolate a held-out fold
# ---------------------------------------------------------------------
def test_test_covariates_are_held_at_the_training_range_before_predict(cv_cell):
    assert "clamp_to_train <- function" in cv_cell, (
        "the no-extrapolation helper is gone; a rolling-origin fold will saturate the "
        "link again and report an rmse from a constant prediction")

    clamp_at = cv_cell.index("cl <- clamp_to_train(tr, te_")
    predict_at = cv_cell.index("predict(m, newdata = te_")
    assert clamp_at < predict_at, "the clamp must run BEFORE predict(), not after"

    # It has to cover the terms that actually extrapolate: the fold's screened
    # drivers, the predictive spec's lagged-response smooths, and the raw trend.
    call = cv_cell[clamp_at:predict_at]
    assert "pf$drivers" in call
    assert "lagcols" in call
    assert "time_index" in call


def test_the_clamp_runs_after_the_predictive_spec_drops_its_rows(cv_cell):
    """The training range must be the one the model in hand was fitted on.

    The predictive spec drops rows with no observed previous month, so clamping
    before that subset would take the range from rows the fit never saw.
    """
    subset_at = cv_cell.index('tr <- tr[as.integer(tr$has_response_lag) == 1L')
    clamp_at = cv_cell.index("cl <- clamp_to_train(tr, te_")
    assert subset_at < clamp_at


def test_coordinates_are_not_clamped(cv_cell):
    """Holding x_km / y_km would change WHICH cell is predicted -- a different answer,
    not a conservative one. Only covariate VALUES may be held."""
    clamp_at = cv_cell.index("cl <- clamp_to_train(tr, te_")
    predict_at = cv_cell.index("predict(m, newdata = te_")
    call = cv_cell[clamp_at:predict_at]
    assert "x_km" not in call and "y_km" not in call


def test_how_much_was_held_is_recorded_on_the_fold(cv_cell):
    """A held row is a restriction on what was validated, so it is reported."""
    for col in ("extrap_clamped_frac", "extrap_clamped_cols"):
        assert cv_cell.count(col) >= 2, (
            f"{col} must be written by both the fitted and the failed-fold branch, so "
            f"rbind_aligned() never has to invent it")


def test_the_summary_cell_reports_what_was_held(cells):
    summary = [
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code"
        and "# --- CV summary, fold coverage, bound checks" in "".join(c["source"])
    ]
    assert len(summary) == 1, f"expected one §13-CV summary cell, found {len(summary)}"
    body = summary[0]
    assert "extrap_clamped_frac" in body, "nothing surfaces the holding to the reader"
    assert "Test rows HELD at the training range" in body, (
        "§13-CV summary must print the folds that were scored at their training edge")
    assert "No test row needed holding" in body, (
        "a sweep that held nothing must say so, so silence is never ambiguous")


# ---------------------------------------------------------------------
# 2. A fix to the sweep must not be resumed past
# ---------------------------------------------------------------------
def test_checkpoint_fingerprint_covers_the_cv_predict_path(all_code):
    """The fingerprint described only the DATA, so changing how a fold is fitted or
    predicted left every cached unit valid and nothing was refitted."""
    assert '"cv_predict_version"' in all_code, (
        "the checkpoint fingerprint does not version the CV fit/predict path; a change "
        "to it will be silently resumed away")
    version = re.search(r'"cv_predict_version":\s*(\d+)', all_code)
    assert version and int(version.group(1)) >= 2, (
        "cv_predict_version must be bumped past the pre-clamp path (1)")

    # And it must sit in the design dict that is actually hashed, unconditionally --
    # the cv_budget block below it is only recorded when a cap is active.
    design = all_code[all_code.index("_ckpt_design = {"):]
    design = design[:design.index("GAM_CKPT_FINGERPRINT")]
    assert '"cv_predict_version"' in design.split("if len(cv_df) < len(gam_df)")[0]


# ---------------------------------------------------------------------
# 3. The R helper, executed
# ---------------------------------------------------------------------
def _extract(cv_cell, name, end_marker):
    start = cv_cell.index(f"{name} <- function")
    return cv_cell[start:cv_cell.index(end_marker, start)]


R_CHECKS = r"""
fail <- function(msg) stop("FAIL: ", msg)
ok <- function(cond, msg) if (!isTRUE(cond)) fail(msg)

tr <- data.frame(rain = c(0, 5, 10, 20), temp = c(20, 21, 22, 23), time_index = 0:3)
te <- data.frame(rain = c(-3, 5, 99, 20), temp = rep(22, 4), time_index = 4:7)

r <- clamp_to_train(tr, te, c("rain", "temp", "time_index"))
ok(all(r$te$rain == c(0, 5, 20, 20)), "out-of-range values held to the training range")
ok(all(r$te$temp == te$temp), "an in-range column is left alone")
ok(all(r$te$time_index == 3), "a rolling-origin fold holds time_index at the train max")
ok(abs(r$frac - 1) < 1e-12, "every row was held")
ok(length(r$notes) == 2, "one note per column that needed holding")

r2 <- clamp_to_train(tr, tr, names(tr))
ok(r2$frac == 0 && length(r2$notes) == 0, "an in-range fold reports no holding")
ok(identical(r2$te, tr), "an in-range fold comes back untouched")

ok(length(clamp_to_train(tr, te, c("rain", "absent"))$notes) == 1, "absent col skipped")
ok(nrow(clamp_to_train(tr, te[0, ], "rain")$te) == 0, "an empty test fold is a no-op")

o <- c(0, 0.1, 0.4, 0.9)
ok(sp_why(o[1:2], o[1:2]) == "too_few", "too_few")
ok(sp_why(rep(0, 5), seq_len(5) / 10) == "constant_obs", "constant_obs")
ok(sp_why(o, rep(0.07, 4)) == "constant_pred", "an interior constant is constant_pred")
ok(sp_why(o, rep(0, 4)) == "constant_pred", "an exact 0 (zero baseline) is not saturation")
ok(sp_why(o, rep(.Machine$double.eps, 4)) == "saturated_pred", "betar floor is saturation")
ok(sp_why(o, rep(1 - .Machine$double.eps, 4)) == "saturated_pred", "betar ceiling too")
ok(sp_why(o, c(0.1, 0.2, 0.3, 0.4)) == "", "a healthy fold states no reason")
cat("PASS\n")
"""


def test_r_helpers_behave(cv_cell, tmp_path):
    """Run the extracted R helpers rather than trusting the text of them."""
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript not installed")
    script = tmp_path / "checks.R"
    script.write_text(
        _extract(cv_cell, "clamp_to_train", "\n\n  build_rhs")
        + "\n"
        + _extract(cv_cell, "sp_why", "\n  sp <- function")
        + "\n"
        + R_CHECKS
    )
    proc = subprocess.run([rscript, str(script)], capture_output=True, text=True)
    assert proc.returncode == 0 and "PASS" in proc.stdout, (
        f"R helper checks failed:\n{proc.stdout}\n{proc.stderr}")
