# Fix package D: estimators

Findings file: `pygeoinf2/review/inference.md`. The reviewer's reproductions
are in the scratchpad's `inference_review/` subdirectory.

## Must-fix

1. **A nonzero noise expectation ignored** by
   `LowRankGaussianPosterior.from_linear_problem` (subtract it from the
   residual), `ConstrainedLeastSquares` (keep the inner estimator's
   translation) and `BackusGilbert` (add the translation); test each
   against its zero-mean twin on shifted data to `1e-12`, as the reviewer
   did.
2. **The low-rank linear mean keeps the prior in the dropped directions**
   (`examples.md`, "far off when the rank drops informative directions"):
   the update must be kept only in the retained directions, each shrunk
   by `1 / (1 + lambda_k)`: `m = m_0 + L V diag(1/(1+lambda)) V* L* A* R^-1
   (d - A m_0 - mu_e)`, exact at full rank and bounded below it. Test at
   full rank against `LinearGaussianInversion` to `1e-8` and at rank 2 of
   6 that the mean's norm stays below the full mean's norm times a small
   factor, and document in the class docstring that the mean at a
   truncated rank omits the update in the dropped directions.
3. **`LaplaceApproximation` always uses the Gauss-Newton curvature**:
   `linearized_normal` must follow the estimator's `gauss_newton=` as the
   MAP's derivative does; `from_problem` gains `gauss_newton=`; test on
   the reviewer's quadratic forward that the covariance matches the
   exact-Hessian inverse when `gauss_newton=False` and the Gauss-Newton
   inverse when `True`. The low-rank route (`rank=`) likewise takes the
   exact Hessian's eigenpairs when asked (the prior-preconditioned
   Hessian may then be indefinite; clip at zero and say so, or refuse
   `rank=` with `gauss_newton=False`; say which).
4. **An exact-data posterior carries a unit-noise density**: `potential()`
   with no error measure refuses, or `LinearGaussianInversion` returns the
   plain Gaussian with no density and no normalizer for exact data; choose
   the refusal unless something in the tests depends on the density.
5. **`resolution` a property, `propagated_covariance()` a method**: make
   them agree (both methods).

## Should-fix

From `inference.md`: `covariance=`/`reference=` on `LeastSquares` and
`MinimumNorm` so the apps' hand-rolled prior-norm whitening becomes a
knob (then the five apps' set/point parts may be simplified later, not
by you); `ConstrainedMinimumNorm` resetting `max_iterations`/`rtol` on
rebuild and lacking `MinimumNorm`'s surface; `LeastSquares` factorizing at
construction for zero-mean noise with a direct solver; `Minimization
.push_forward` returning a private copy; `log_evidence` dropping the
standard error; `harden_error` public with only tests calling it; the
mixture's loops without `n_jobs=`; the full-rank Laplace Gaussian having
no sampler and no evidence where the `rank=` route has both (give it the
Laplace evidence `-J(m) - log det(I + H) / 2` through the library's
log-determinant, and a sampler through the normal operator's solve if
local). Decision 2 in SUMMARY.md: do not change the potential's bypass;
package A owns the wording on the model-space formalism.

## Rules for every fix package

Working directory `/home/david/dev/pygeoinf`, package `pygeoinf2`, run with
`poetry run`. Read, in order: `pygeoinf2/review/BRIEF.md` (the principles
the library is held to), `pygeoinf2/review/SUMMARY.md` (the consolidated
findings and the **decisions taken with David**, which bind you), your
layer's findings file named below, and the code each finding points at.

- Fix every **Must-fix** item listed for your package. Each fix comes with a
  test that fails before and passes after, in the layer's test file, built
  where it matters on a non-orthonormal metric (`make_dense_metric_space`,
  `make_weighted_space` in `pygeoinf2/tests/conftest.py`) or a function
  space.
- Then the **Should-fix** items of your layer's findings file: fix the ones
  that are local and clear-cut; **skip** any that needs a design decision
  or an API change not already in SUMMARY.md's decisions, and say so.
- Stay inside your package's files, plus tests, examples and tutorials
  that your change breaks or that a listed item names. If a fix needs an
  edit elsewhere, record it in your report instead of making it.
- No decision numbers, history, planning or "v1" talk in code, docstrings,
  tests or examples. Docstrings self-contained and accurate. American
  spelling. Do not edit `DECISIONS.md` or `CURRENT_STATE.md`; the caller
  writes those from your report.
- `poetry run ruff check pygeoinf2/` and `ruff format` clean. Run your
  layer's test files as you go and the whole fast suite at the end
  (`MPLBACKEND=Agg poetry run pytest pygeoinf2/tests -q -p no:cacheprovider`,
  about four minutes; run it in the background if your tool allows, and
  never two at once). Keep every other command under two minutes;
  `MPLBACKEND=Agg`; never show a figure.
- Scratch files go in a subdirectory of your own under
  `/tmp/claude-1000/-home-david-dev-pygeoinf/4bec2589-8482-4f4c-843c-b27f331a8905/scratchpad/`,
  never in its root (other agents share it).
- Do not commit.

## Report

Append a section **Fixed** to your findings file: each item, what changed
(file and function), the test that covers it; then **Skipped**, each with
the reason. Report back to the caller with the counts, anything you found
on the way that the review missed, the suite's final line, and anything
you could not do.
