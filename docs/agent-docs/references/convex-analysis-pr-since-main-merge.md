# Summary: `convex-analysis-pr` Since `origin/main` Merged PR #132

**Date:** 2026-05-15
**Baseline on `origin/main`:** `2319476` — `Convex analysis pr (#132)` (2026-05-09)

## Important Interpretation Note

The local branch history still contains older convex-analysis port commits because
`origin/main` absorbed that work through PR #132 rather than preserving the
original branch SHA history. For "what happened after the merge", the meaningful
local work starts on **2026-05-12**. The April / early-May convex-analysis port
commits should not be presented to the developers as new post-merge work.

## TL;DR

Since the convex-analysis merge, the local `convex-analysis-pr` branch has moved
in four directions:

1. exact plot / slice rendering for quadratic subsets,
2. sphere-cap and DLI benchmark follow-up,
3. function-space hardening of Gaussian credible sets, including new weighted
   chi-square numerical backends,
4. agent-doc and archival cleanup.

The largest new technical addition is the function-space hardening stack in
`pygeoinf/gaussian_measure.py`, `pygeoinf/quadratic_form_quantile.py`,
`pygeoinf/matrix_function.py`, and `pygeoinf/spectral_operator.py`.

## Net Current Delta vs `origin/main`

- The current net diff is **61 files changed**.
- Approximate size: **9415 insertions / 213 deletions**.
- That delta includes source code, tests, tutorials, generated figures, work
  scripts, and agent-doc artifacts.
- Not every differing file is intended for upstream review; a noticeable share
  of the diff is notebooks, `work/` outputs, and planning / reference material.

## Post-Merge Workstreams

### 1. Exact Plotting and Finite-Dimensional Credible-Set Polish

**Relevant commits:** `585c2be`, `4c5b731` (2026-05-12)

- Added exact quadratic slice rendering for `Ball` and `Ellipsoid` in
  `pygeoinf/plot.py`.
- Tightened the finite-dimensional `GaussianMeasure.credible_set()` plotting and
  regression surface.
- Added `tests/test_plot_exact_quadratic_slices.py`.
- Expanded `tests/test_plot.py`, `tests/test_gaussian_measure_credible_set.py`,
  and `tests/test_subsets.py`.
- Refreshed `tutorials/gaussian_measure_to_sets_demo.ipynb`.

**Files to mention first:**

- `pygeoinf/plot.py`
- `pygeoinf/gaussian_measure.py`
- `tests/test_plot_exact_quadratic_slices.py`
- `tests/test_gaussian_measure_credible_set.py`
- `tutorials/gaussian_measure_to_sets_demo.ipynb`

### 2. Sphere-Cap / DLI Benchmark Follow-Up

**Relevant commit:** `d873575` (2026-05-12)

- Added an exact-vs-Monte-Carlo sphere-cap benchmark workflow.
- Extended `pygeoinf/symmetric_space/sphere.py` and
  `pygeoinf/symmetric_space/symmetric_space.py` to support the benchmark and
  associated exact-cap calculations.
- Added `tests/symmetric_space/test_sphere_cap_integrals.py`.
- Added or refreshed:
  - `work/sphere_cap_monte_carlo_benchmark.py`
  - `work/dli_euclidean_ellipsoids_demo.py`
  - `work/pli_euclidean_gaussian_analogue.py`
  - `work/sphere_dli_example.py`
- Generated benchmark figures and CSV summaries under `work/figures/`.

**Files to mention first:**

- `pygeoinf/symmetric_space/sphere.py`
- `pygeoinf/symmetric_space/symmetric_space.py`
- `tests/symmetric_space/test_sphere_cap_integrals.py`
- `work/sphere_cap_monte_carlo_benchmark.py`
- `work/sphere_dli_example.py`

### 3. Function-Space Hardening of Gaussian Credible Sets

**Relevant commits:** `2614f44`, `0a64998`, `e5f8632`, `c5f859c` (2026-05-14)

This is the main new line of work after the convex-analysis merge.

- Added two new function-space / infinite-dimensional credible-set geometries in
  `GaussianMeasure`:
  - `ambient_ball`
  - `weakened_ellipsoid`
- Added two radius-calibration routes for these geometries:
  - spectrum-based weighted-chi-square calibration,
  - sampling-based empirical calibration.
- Added the new weighted-chi-square backend module
  `pygeoinf/quadratic_form_quantile.py`.
- Added the matrix-free Lanczos backend in `pygeoinf/matrix_function.py` for
  applying fractional covariance powers.
- Added the low-rank spectral backend in `pygeoinf/spectral_operator.py`.
- Added theory and notebook support:
  - `docs/agent-docs/theory/hardening.md`
  - `docs/agent-docs/theory/function-space-hardening.md`
  - `tutorials/function_space_gaussian_measure_demo.ipynb`
- Added the benchmark / exploration script
  `work/function_space_hardening_demo.py`.
- Added dedicated tests:
  - `tests/test_quadratic_form_quantile.py`
  - `tests/test_matrix_function.py`
  - `tests/test_spectral_operator.py`
  - substantial additions to `tests/test_gaussian_measure_credible_set.py`

**Numerical follow-up within the same workstream:**

- `e5f8632`: fixed Imhof adaptive truncation for decaying spectra in
  `pygeoinf/quadratic_form_quantile.py`.
- `c5f859c`: added `method="auto"` plus tolerance-driven backend selection for
  weighted-chi-square quantiles, then threaded that through
  `pygeoinf/gaussian_measure.py`.

**Files to mention first:**

- `pygeoinf/gaussian_measure.py`
- `pygeoinf/quadratic_form_quantile.py`
- `pygeoinf/matrix_function.py`
- `pygeoinf/spectral_operator.py`
- `tests/test_quadratic_form_quantile.py`
- `tests/test_gaussian_measure_credible_set.py`
- `docs/agent-docs/theory/function-space-hardening.md`

### 4. Documentation, Plans, and Archival Cleanup

**Relevant commits:** `31c84ac`, `489d7ce`, `e59244b` (2026-05-14 to 2026-05-15)

- Recorded and archived the function-space hardening plan and phase-complete
  documents.
- Updated living references, especially
  `docs/agent-docs/references/living/convex-analysis-reference.md`.
- Archived the completed `port-convex-analysis` plan set and added the missing
  `port-convex-analysis-complete.md` summary.

This is mainly bookkeeping and should be described separately from the new code.

## Commit Timeline Since the Merge

| Date | Commit | Summary |
|---|---|---|
| 2026-05-12 | `d873575` | Sphere-cap exact-vs-MC benchmark, symmetric-space follow-up, new demos / figures |
| 2026-05-12 | `585c2be` | Exact quadratic slice rendering for `Ball` / `Ellipsoid`, credible-set plotting updates |
| 2026-05-12 | `4c5b731` | Plot/docs cleanup, tutorial cleanup, regression-test tightening |
| 2026-05-14 | `2614f44` | Core function-space hardening implementation (Phases 1–4) |
| 2026-05-14 | `0a64998` | Hardening theory docs, plan doc, and tutorial notebook |
| 2026-05-14 | `31c84ac` | Phase 5 hardening completion doc |
| 2026-05-14 | `e5f8632` | Imhof adaptive truncation fix for decaying spectra |
| 2026-05-14 | `489d7ce` | Hardening Phase 6 docs, living-reference update, archival |
| 2026-05-14 | `c5f859c` | `method="auto"` and tolerance-based weighted-chi-square backend selection |
| 2026-05-15 | `e59244b` | Agent-doc cleanup and convex-analysis plan archival |

## What to Emphasize When Explaining This to the Developers

If the developers ask for the short version, the clean explanation is:

> After `main` absorbed the convex-analysis PR, the branch did not keep growing
> the convex-analysis stack itself. Instead, it moved into follow-on work:
> exact plotting improvements, sphere-cap / DLI benchmarking, and then a new
> function-space credible-set hardening line built on top of Gaussian measures.
> The only substantial new library/runtime work relative to `main` is the
> hardening stack and its numerical backends; the rest is plotting polish,
> benchmark/demo work, tests, notebooks, and documentation cleanup.

## If the Developers Want a Code-Only Review Surface

Point them first to these files:

- `pygeoinf/gaussian_measure.py`
- `pygeoinf/quadratic_form_quantile.py`
- `pygeoinf/matrix_function.py`
- `pygeoinf/spectral_operator.py`
- `pygeoinf/plot.py`
- `pygeoinf/symmetric_space/sphere.py`
- `pygeoinf/symmetric_space/symmetric_space.py`

Then use these tests as the entry point for validation:

- `tests/test_gaussian_measure_credible_set.py`
- `tests/test_quadratic_form_quantile.py`
- `tests/test_matrix_function.py`
- `tests/test_spectral_operator.py`
- `tests/test_plot_exact_quadratic_slices.py`
- `tests/symmetric_space/test_sphere_cap_integrals.py`

## Validation Notes

- The post-merge workstream includes extensive new tests for the hardening and
  plotting surfaces.
- The hardening follow-up was previously validated with the full pygeoinf suite
  at **925 passing tests** at the time the weighted-chi-square auto-mode work was
  finished.
- The latest local commit (`e59244b`) is documentation-only.