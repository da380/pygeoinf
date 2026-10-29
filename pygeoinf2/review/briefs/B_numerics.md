# Fix package B: absolute thresholds in numerics

Findings file: `pygeoinf2/review/numerics.md`.

One rule for the whole package: **every tolerance is relative to a scale
the problem supplies**, never absolute and never floored at one. Name the
scale in each docstring.

## Must-fix

1. **`weighted_chi2_quantile` and `_cdf` depend on the units of the
   weights**: normalize by the largest weight at entry, make the equality
   test relative, scale Imhof's floors, and test that the quantile in
   units of the weights is the same at scales `1`, `1e-9` and `1e9`, and
   that `GaussianMeasure.ambient_ball(level=0.95)` on deviations of `3e-5`
   covers 95% of draws (it covers 0.2% today).
2. **The Lanczos calculus preallocates `dim x max_iterations` with the cap
   at `dim`**: grow the buffer as steps are taken (geometrically or by
   blocks); test with a 120 000-dimensional diagonal operator with five
   distinct eigenvalues that `apply_operator_function` runs in a few steps
   and a few megabytes.
3. **Lanczos's breakdown test floored at one**: relative to the operator's
   scale (the first Lanczos coefficient, or the norm of the first image);
   test the square root of an operator of norm `1e-13`.
4. **GMRES and MINRES on a singular operator whose Krylov space closes
   exactly**: report convergence from the true residual (recompute it at
   the end, or test the recurrence's residual against it), never from the
   recurrence alone; test with a mask and a diagonal with a zero.
5. **Chambolle-Pock's stop**: feasibility relative to the data's scale and
   convergence requiring the primal-dual gap or iterate change too, never
   feasibility alone; test the rescaled problem (`1e-7`) gives the same
   support, and through the feasible set's splitting route that the inner
   end never exceeds the outer.
6. **The proximal bundle method returning its start as converged on a
   scaled objective**: its gap test relative to the objective's scale;
   test at scales `1` and `1e-4`.

## Should-fix

From `numerics.md`: the Woodbury default inner CG at `rtol=1e-3`
(D-47; the reviewer shows it failing a four-dimensional data space:
tighten the default or make it relative to the outer tolerance, and say
which in your report, since it touches a decision); `misfit_search` and
`MinimumNorm` refusing `minimum_damping` above one; `random_diagonal`
without an `Estimate`; `log_determinant`'s `rtol` against what
CURRENT_STATE §4 says; the Jacobi and diagonal floors; the trust-region
radii; and the rest of the list where local.

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
