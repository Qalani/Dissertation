"""Structural contract for winam_wh_spatial_panel_driver_gam.ipynb.

These are static checks over the notebook JSON — no Drive, no Earth Engine, no R, no
execution. They pin the properties that made the previous driver model indefensible
and that must not silently regress:

1. the guardrail helpers exist and export the API the modelling sections rely on;
2. the forcing predictor set is exogenous by construction (no ``wh_*``, no optical
   proxy) and the lagged-response terms live only in the predictive specification;
3. classifier confidence has exactly one configured use;
4. the temporal fold design covers far more than the final four months;
5. every ``%%R -i`` variable is assigned by an earlier Python cell;
6. the sibling predictive-ML notebook is untouched by these rules (it is allowed to
   use lagged-response features — that is its job).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "winam_wh_spatial_panel_driver_gam.ipynb"
SIBLING = REPO / "winam_wh_spatial_panel_predictive_ml.ipynb"

RESPONSE_DERIVED = re.compile(r"^wh_|_neigh(_|$)")
ENDOGENOUS_OPTICAL = re.compile(r"(turb|ndti|ndci|chl|reflect|mci|mph)", re.IGNORECASE)


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


@pytest.fixture(scope="module")
def code_sources(cells):
    return [("".join(c["source"]), i) for i, c in enumerate(cells) if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def all_source(cells):
    return "\n".join("".join(c["source"]) for c in cells)


def _cell_containing(code_sources, marker):
    hits = [s for s, _ in code_sources if marker in s]
    assert hits, f"no code cell contains {marker!r}"
    return hits[0]


def _literal_from(source, name):
    """ast.literal_eval the first top-level assignment of ``name`` in ``source``."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at the top level of the cell")


# ---------------------------------------------------------------------
# 1. Guardrail helpers
# ---------------------------------------------------------------------
REQUIRED_HELPERS = [
    "is_response_derived", "is_endogenous_optical", "classify_predictor",
    "formula_variables", "assert_forcing_formula_exogenous",
    "split_forcing_and_response_terms", "is_ambiguous_column",
    "identify_ambiguous_columns", "screen_covariates", "assert_all_identified",
    "resolve_confidence_usage", "build_confidence_weights",
    "spatial_block_folds", "rolling_origin_folds", "assign_temporal_folds",
    "fold_coverage_table", "assert_folds_scored", "cover_metrics",
    "assert_predictions_in_bounds", "dependence_verdict", "residual_moran",
    "residual_lag1", "tag_evidence", "assert_evidence_labelled",
]


def test_guardrail_cell_defines_the_api(code_sources):
    src = _cell_containing(code_sources, "4d. Methodological guardrails")
    tree = ast.parse(src)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [h for h in REQUIRED_HELPERS if h not in defined]
    assert not missing, f"guardrail cell no longer defines: {missing}"


def test_guardrail_cell_has_no_heavy_imports(code_sources):
    """It must stay importable outside Colab so the unit tests can exercise it."""
    src = _cell_containing(code_sources, "4d. Methodological guardrails")
    tree = ast.parse(src)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"numpy", "pandas", "re"}, f"unexpected top-level imports: {top}"


# ---------------------------------------------------------------------
# 2. The forcing set is exogenous by construction
# ---------------------------------------------------------------------
def test_configured_forcing_terms_are_exogenous(code_sources):
    cfg = _cell_containing(code_sources, "GAM_FORCING_TERMS = {")
    terms = _literal_from(cfg, "GAM_FORCING_TERMS")
    extended = _literal_from(cfg, "GAM_FORCING_TERMS_EXTENDED")
    fallbacks = _literal_from(cfg, "GAM_FORCING_FALLBACKS")
    candidates = set(terms) | set(extended) | {c for v in fallbacks.values() for c in v}
    bad = [c for c in candidates
           if RESPONSE_DERIVED.search(c) or ENDOGENOUS_OPTICAL.search(c)]
    assert not bad, f"response-derived or endogenous terms configured as forcing: {bad}"
    assert len(terms) <= 12, "the 'parsimonious' forcing set has grown past 12 terms"
    for name, spec in terms.items():
        assert len(spec) == 2 and spec[0].strip(), f"{name} has no stated mechanism"
        assert spec[1] in {"+", "-", "?"}, f"{name} has no a-priori sign"


def test_current_and_lagged_forms_are_not_both_requested(code_sources):
    cfg = _cell_containing(code_sources, "GAM_FORCING_TERMS = {")
    terms = list(_literal_from(cfg, "GAM_FORCING_TERMS"))
    bases = [t[:-5] if t.endswith("_lag1") else t for t in terms]
    dupes = {b for b in bases if bases.count(b) > 1}
    assert not dupes, f"driver entered in both current and lagged form: {dupes}"


def test_lagged_response_terms_are_predictive_only(code_sources):
    cfg = _cell_containing(code_sources, "GAM_PREDICTIVE_LAG_TERMS")
    lags = _literal_from(cfg, "GAM_PREDICTIVE_LAG_TERMS")
    assert lags, "the predictive spec has no lagged-response terms left"
    assert all(RESPONSE_DERIVED.search(c) for c in lags)
    sec12 = _cell_containing(code_sources, "12. Model dataset: a parsimonious FORCING set")
    assert "predictive_lag_cols" in sec12
    # The forcing set is built from the exogenous partition, never from the lag list.
    assert "split_forcing_and_response_terms" in sec12


def test_every_forcing_formula_is_asserted_exogenous(all_source):
    """Each forcing specification must be gated, not merely documented."""
    for label in ["forcing (primary inferential)", "forcing transfer/CV",
                  "drivers-carry-space forcing", "§13e forcing",
                  "§13f robustness forcing", "ordered-Beta forcing"]:
        assert f'"{label}"' in all_source, f"no exogeneity assertion labelled: {label}"
    # One call per forcing specification (primary + CV share a loop), plus §13h.
    assert all_source.count("assert_forcing_formula_exogenous(") >= 6, (
        "a forcing specification is being built without the exogeneity gate")


def test_no_endogenous_term_is_hardcoded_into_a_forcing_formula(code_sources):
    for src, idx in code_sources:
        for m in re.finditer(r"r_formula_(forcing|cv|causal)\s*=\s*\(?f?\"", src):
            snippet = src[m.start():m.start() + 400]
            assert "wh_cover_lag1" not in snippet and "neigh_lag1" not in snippet, (
                f"cell {idx}: lagged-response term hardcoded into a forcing formula")


# ---------------------------------------------------------------------
# 3. Confidence used once
# ---------------------------------------------------------------------
def test_confidence_mode_is_single_use(code_sources):
    cfg = _cell_containing(code_sources, "GAM_CONFIDENCE_MODE = ")
    mode = _literal_from(cfg, "GAM_CONFIDENCE_MODE")
    assert mode in {"soft_response", "likelihood_weights", "none"}
    assert _literal_from(cfg, "GAM_CONFIDENCE_ALLOW_PARTIAL") is False, (
        "the flagged partial-confidence variant must not be the default")
    frame = _cell_containing(code_sources, "# --- Build the GAM modelling frame ---")
    assert "resolve_confidence_usage(" in frame
    assert "not (RESPONSE_IS_SOFT and WEIGHTS_ARE_CONFIDENCE)" in frame, (
        "the double-use assertion has been removed from the frame builder")


def test_absence_rows_are_never_given_full_confidence(all_source):
    """The old code did `wh_conf_mean.fillna(1.0)`; that must not come back."""
    assert not re.search(r"wh_conf_mean.*fillna\(\s*1(\.0)?\s*\)", all_source)
    assert not re.search(r'\["wh_conf_mean"\]\.fillna\(1', all_source)


# ---------------------------------------------------------------------
# 4. Fold design
# ---------------------------------------------------------------------
def test_temporal_folds_cover_more_than_four_months(code_sources):
    cfg = _cell_containing(code_sources, "GAM_TEMPORAL_HORIZON_MONTHS")
    n_folds = _literal_from(cfg, "GAM_N_TEMPORAL_FOLDS")
    horizon = _literal_from(cfg, "GAM_TEMPORAL_HORIZON_MONTHS")
    min_train = _literal_from(cfg, "GAM_TEMPORAL_MIN_TRAIN_MONTHS")
    assert n_folds * horizon >= 12, (
        f"temporal CV covers only {n_folds * horizon} months — the fault being fixed "
        f"was validating just the final four")
    assert min_train >= 12


def test_folds_are_built_once_and_reused(code_sources):
    frame = _cell_containing(code_sources, "# --- Build the GAM modelling frame ---")
    assert "spatial_block_folds(" in frame and "rolling_origin_folds(" in frame
    mfc = _cell_containing(code_sources, "13e. Model-family comparison")
    assert "mfc_df = gam_df.copy()" in mfc, (
        "§13e must reuse the §13 frame so the folds are identical by construction")


def test_fold_internal_preprocessing(all_source):
    assert "GAM_FOLD_INTERNAL_PREP" in all_source
    assert all_source.count("prep_fold <- function") >= 2, (
        "fold-internal imputation/collinearity screening is missing from a CV section")


# ---------------------------------------------------------------------
# 5. Sensor handling, sections present, R inputs resolvable
# ---------------------------------------------------------------------
def test_sensor_indicator_dropped_on_single_sensor_panel(code_sources):
    src = _cell_containing(code_sources, "RESPONSE SENSOR COMPOSITION")
    assert 'panel.drop(columns=["sensor_is_s1"])' in src
    assert "S1_GAPFILL_APPLIED" in src and "cloud_mnar_limitation" in src


def test_required_sections_exist(all_source):
    for marker in ["4d. Methodological guardrails",
                   "7c. What sensor(s) does this run ACTUALLY use",
                   "9c. Earth Engine covariate audit",
                   "13b. Residual dependence + the gate on inferential claims",
                   "13f. Robustness", "13g. Diagnostic tables",
                   "13h. Lightweight checks"]:
        assert marker in all_source, f"missing section: {marker}"


def test_r_cell_inputs_are_defined_earlier(cells):
    seen, problems = set(), []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("%%R"):
            header = s.split("\n", 1)[0]
            for v in re.findall(r"-i\s+([A-Za-z_][A-Za-z0-9_]*)", header):
                if v not in seen:
                    problems.append((i, v))
            seen |= set(re.findall(r"-o\s+([A-Za-z_][A-Za-z0-9_]*)", header))
        else:
            seen |= set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=[^=]", s, re.M))
            seen |= set(re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", s, re.M))
    assert not problems, f"%%R -i variables never assigned earlier: {problems}"


def _r_cells(cells):
    out = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("%%R"):
            out.append((i, s.split("\n", 1)[1] if "\n" in s else ""))
    return out


def test_r_cells_have_no_adjacent_string_literals(cells):
    """R does not concatenate adjacent string literals — Python and C do.

    A wrapped ``sprintf()`` format string written the Python way parses in neither
    language the same way: R raises "unexpected string constant" at the second
    literal, which only shows up when the cell is finally run in Colab.
    """
    pattern = re.compile(r'"\s*\n\s*"')
    offenders = [i for i, body in _r_cells(cells)
                 if pattern.search(re.sub(r"#[^\n]*", "", body))]
    assert not offenders, (
        f"R cells with adjacent string literals (join them with paste0()): {offenders}")


def test_r_cells_parse(cells, tmp_path):
    """Full R parse when Rscript is available; skipped otherwise."""
    import shutil
    import subprocess

    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript not installed")
    failures = []
    for i, body in _r_cells(cells):
        f = tmp_path / f"cell_{i}.R"
        f.write_text(body)
        proc = subprocess.run([rscript, "-e", f'invisible(parse("{f}"))'],
                              capture_output=True, text=True)
        if proc.returncode:
            failures.append((i, proc.stderr.strip().splitlines()[-3:]))
    assert not failures, f"R cells failed to parse: {failures}"


def test_ar1_rho_is_estimated_per_model(all_source):
    assert "fit_with_own_ar1" in all_source
    assert "rho_used" not in all_source, (
        "a single shared rho_used is back; each model must estimate its own AR1 rho")


def test_no_removed_helpers_are_still_referenced(all_source):
    for gone in ["GAM_WEIGHT_BY_CONFIDENCE", "GAM_COLLAPSE_REDUNDANT_DRIVERS",
                 "GAM_DROP_S1_FILL", "MFC_RUN_TWEEDIE_FULLFIT", "fit_one("]:
        assert gone not in all_source, f"stale reference to removed machinery: {gone}"


# ---------------------------------------------------------------------
# 6. Results hygiene and the sibling notebook
# ---------------------------------------------------------------------
def test_stale_results_are_labelled(all_source):
    assert "STALE RESULTS — DO NOT QUOTE" in all_source
    assert "have not been re-run" in all_source or "not yet been re-run" in all_source


def test_modelling_cells_carry_no_stale_outputs(cells):
    """Cells rewritten for the corrected workflow must not ship old outputs."""
    markers = ["12. Model dataset: a parsimonious FORCING set",
               "# --- Build the GAM modelling frame ---",
               "13b. Residual dependence",
               "13e. Model-family comparison",
               "13f. Robustness", "13g. Diagnostic tables", "13h. Lightweight checks"]
    for c in cells:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if any(m in s for m in markers):
            assert not c.get("outputs"), f"stale outputs on a rewritten cell: {s[:60]!r}"


@pytest.mark.skipif(not SIBLING.exists(), reason="sibling notebook not present")
def test_sibling_predictive_notebook_is_untouched_by_these_rules():
    """The predictive ML notebook is allowed lagged-response features — that is its job.

    This only asserts the separation still exists: the driver notebook must not import
    from it, and it must not have acquired the driver notebook's guardrail cell.
    """
    sib = SIBLING.read_text()
    assert "4d. Methodological guardrails" not in sib
    drv = NOTEBOOK.read_text()
    assert "winam_wh_spatial_panel_predictive_ml" in drv, (
        "the driver notebook should still point at its sibling")


# ---------------------------------------------------------------------
# Ordered-Beta memory contract (§13d and the ordbeta family in §13e)
# ---------------------------------------------------------------------
# glmmTMB maps every mgcv s() term onto a random-effect block over a DENSE spline
# basis, so TMB tapes rows x basis-columns entries for automatic differentiation.
# rpy2 runs R inside the Python process, so that peak is charged against one Colab
# RAM ceiling together with every frame the notebook holds. Asking for 60,000 rows
# against s(x_km, y_km, k=100) exhausted it and the runtime was killed mid-fit.
#
# An out-of-memory kill is a SIGKILL to the whole process: no R condition is
# raised and no tryCatch() runs, which is why the session died rather than
# reporting an error. These pin the two things that make it survivable — size the
# fit against the RAM actually free, and run it somewhere that is not this session.
def test_ordbeta_subsample_is_budgeted_against_free_ram(code_sources):
    s = _cell_containing(code_sources, "13d. Ordered-Beta cross-check")
    assert "available_ram_gb()" in s, "the subsample must be sized against RAM read at run time"
    assert "budget_tmb_fit(" in s
    assert "GAM_ORDBETA_RUN_R" in s, (
        "the R cell needs an EFFECTIVE run flag so an unaffordable fit is skipped, "
        "not attempted")


def test_ordbeta_fit_runs_in_a_child_process(code_sources):
    s = _cell_containing(code_sources, "fit_ordbeta_isolated <- function")
    assert "system2(" in s and "Rscript" in s, (
        "the fit must run in a child R process; a tryCatch() in this session cannot "
        "catch the SIGKILL that an out-of-memory kill delivers")
    assert "timeout_s" in s, "a child that never returns must be given up on"
    assert "137L" in s, "an out-of-memory kill (exit 137) must be named as such"


def test_the_child_reports_r_errors_so_a_missing_file_means_a_kill(code_sources):
    """The child saves output on BOTH paths, which is what makes the signal clean."""
    s = _cell_containing(code_sources, "fit_ordbeta_isolated <- function")
    child = s[s.index("writeLines("):s.index("rs <- Sys.which")]
    assert "error = function(e) list(ok = FALSE" in child
    assert child.count("saveRDS(out, a[2]") == 1, (
        "the child must write its result file on success AND on an R error, so that a "
        "MISSING file unambiguously means it was killed")


def test_ordbeta_isolation_is_on_by_default(code_sources):
    d = _cell_containing(code_sources, "13d. Ordered-Beta cross-check")
    e = _cell_containing(code_sources, "13e. Model-family comparison")
    assert re.search(r"^GAM_ORDBETA_ISOLATE\s*=\s*True", d, re.M)
    assert re.search(r"^MFC_ORDBETA_ISOLATE\s*=\s*True", e, re.M)


def test_family_comparison_ordbeta_goes_through_the_isolated_fitter(code_sources):
    """§13e fits one model per mode x fold x family. One killed ordered-Beta fold
    used to cost the session and every other fit in the sweep."""
    s = _cell_containing(code_sources, "13e comparison skipped")
    assert s.count("glmmTMB::glmmTMB(") == 1, (
        "the only glmmTMB call in §13e must be the one inside ob_fit_predict(); every "
        "caller goes through it so a killed fold degrades to an NA metric row")
    body = s[s.index("predict_family <- function"):]
    assert "glmmTMB::glmmTMB(" not in body, "predict_family must not fit glmmTMB directly"
    assert body.count("ob_fit_predict(") == 2, (
        "both the ordbeta family and the hurdle's ordered-Beta positive stage must use it")


def test_ordbeta_spatial_basis_is_budgeted_not_squared(all_source):
    assert "MFC_K_SPATIAL)[1]^2" not in all_source, (
        "the ordered-Beta spatial basis was derived as MFC_K_SPATIAL ** 2 (144, capped "
        "at 100); that dense basis on a 60,000-row subsample is what exhausted the runtime")
    assert "MFC_ORDBETA_K_SPATIAL_R" in all_source


def test_ordbeta_failure_is_reported_not_swallowed(code_sources):
    s = _cell_containing(code_sources, "fit_ordbeta_isolated <- function")
    assert "did not produce a fit" in s
    assert "NOT RUN" in s, "a cross-check that did not run must never read as one that did"


def test_isolated_fitter_actually_survives_a_killed_child(cells, tmp_path):
    """Execute the child-process contract for real, when Rscript is available.

    Static checks cannot see the two things that matter here: that a child killed
    the way the OOM killer kills it (SIGKILL) leaves the parent alive and
    correctly diagnosed, and that the formula survives the hand-off. The latter is
    easy to get wrong — ``as.character()`` on a formula yields ``c("~", lhs, rhs)``,
    so a ``[1]`` subscript ships "~" and every fit fails in the child.
    """
    import shutil
    import subprocess

    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript not installed")

    body = None
    for c in cells:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "fit_ordbeta_isolated <- function" in s:
            body = s[s.index("fit_ordbeta_isolated <- function"):s.index("\nrun_ob")]
    assert body, "no cell defines fit_ordbeta_isolated"
    (tmp_path / "helper.R").write_text(body)

    driver = r'''
setwd(commandArgs(trailingOnly = TRUE)[1])
source("helper.R")
set.seed(1); tr <- data.frame(y = runif(200), x = rnorm(200)); te <- tr[1:37, ]

# glmmTMB is not a test dependency: swap in a glm so the hand-off, the prediction
# vector and the summary text are exercised without it.
src <- paste(deparse(body(fit_ordbeta_isolated)), collapse = "\n")
src <- gsub("glmmTMB::glmmTMB(as.formula(p$form), data = p$train,",
            "glm(as.formula(p$form), data = p$train,", src, fixed = TRUE)
src <- gsub("family = glmmTMB::ordbeta())", "family = quasibinomial())", src, fixed = TRUE)
src <- gsub("isTRUE(m$sdr$pdHess)", "isTRUE(m$converged)", src, fixed = TRUE)
stub <- fit_ordbeta_isolated; body(stub) <- parse(text = src)[[1]]

# 1. success: a FORMULA object survives the hand-off and predicts onto newdata.
r <- stub(tr, y ~ x, newdata = te, timeout_s = 120)
stopifnot(isTRUE(r$ok), length(r$pred) == nrow(te), is.finite(r$fitted_max),
          length(r$summary_txt) > 1)
# 2. newdata = NULL predicts in-sample.
stopifnot(length(stub(tr, y ~ x, timeout_s = 120)$pred) == nrow(tr))
# 3. an R-level error in the child comes back as a message, not a dead session.
e <- fit_ordbeta_isolated(tr, y ~ x, timeout_s = 120)   # real one: no glmmTMB here
stopifnot(identical(e$ok, FALSE), nzchar(e$why))
# 4. a child killed exactly as the OOM killer kills it: parent alive and correct.
ksrc <- sub("writeLines\\(c\\(", 'writeLines(c("tools::pskill(Sys.getpid(), 9)", ',
            paste(deparse(body(fit_ordbeta_isolated)), collapse = "\n"))
killer <- fit_ordbeta_isolated; body(killer) <- parse(text = ksrc)[[1]]
k <- killer(tr, y ~ x, timeout_s = 120)
stopifnot(identical(k$ok, FALSE), isTRUE(k$killed), grepl("out of memory", k$why))
# 5. no temp files left behind by any of those paths.
stopifnot(length(list.files(tempdir(), pattern = "^ordbeta")) == 0)
cat("OK\n")
'''
    (tmp_path / "driver.R").write_text(driver)
    proc = subprocess.run([rscript, "--vanilla", str(tmp_path / "driver.R"), str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"child-process contract failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")


def test_recorded_k_spatial_survives_a_wrapped_formula(cells, tmp_path):
    """The k_spatial recorded in ordbeta_summary is read back out of the fitted
    formula, and R's deparse() WRAPS a long one. Rejoining the pieces leaves runs
    of whitespace at the wrap points, so a spatial term that happens to straddle a
    wrap reported k_spatial as NA — and a downsized cross-check that does not say
    what it was downsized to reads as the one that was requested.
    """
    import shutil
    import subprocess

    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript not installed")

    src = next("".join(c["source"]) for c in cells
               if c["cell_type"] == "code" and "k_sp <- suppressWarnings" in "".join(c["source"]))
    extract = src[src.index("form_txt <- gsub"):src.index("ordbeta_summary <- data.frame(engine")]

    # Where deparse() breaks depends on the whole formula, so this uses a driver
    # count whose wrap lands INSIDE the spatial term — the case that actually broke.
    # The two stopifnot()s keep the test honest: if a future R wraps elsewhere they
    # fail loudly rather than letting this pass without exercising anything.
    script = ('form <- as.formula("wh_cover ~ s(rain, k=6) + s(month_num, k=6) '
              '+ s(x_km, y_km, k=50)")\n'
              'raw <- paste(deparse(form), collapse = " ")\n'
              'stopifnot(length(deparse(form)) > 1)              # it really does wrap\n'
              'stopifnot(grepl("y_km,[[:space:]]{2,}", raw))     # and inside the spatial term\n'
              + extract +
              '\nstopifnot(identical(k_sp, 50L)); cat("OK\\n")\n')
    f = tmp_path / "k.R"
    f.write_text(script)
    proc = subprocess.run([rscript, "--vanilla", str(f)], capture_output=True, text=True)
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"k_spatial not recovered from a wrapped formula:\n{proc.stdout}{proc.stderr}")
