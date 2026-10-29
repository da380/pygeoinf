# Fix package C: feasible property sets

Findings file: `pygeoinf2/review/geometry.md`. The reviewer's reproductions
are `e*.py` in the scratchpad root with `common.py` beside them; copy them
into your own subdirectory before running (the root is shared).

## Must-fix

1. **The outer ellipsoid** (decision 4 in SUMMARY.md): the model-space
   S-procedure becomes the default `outer_ellipsoid()`; the data-space form
   stays behind `formalism="data_space"` with its regime of validity in the
   docstring (it cancels when the data outnumber the model and the prior
   is slack); both routes check the returned ellipsoid against the set's
   support in a few directions before returning, inflating to cover or
   raising, so that the docstring's claim of a check is true. Test with
   the reviewer's `e8c` setting (dims 20 and 40) that the default never
   falls below an attained support, and that the data-space route either
   passes or raises there.
2. **A flat closed-form set**: test the rank of the shape once, and for a
   flat set refuse `ellipsoid`, `project` and the closed form's
   `outer_ellipsoid` with a `ValueError` saying the set is flat; take the
   level gradient through the least-squares solve `inclusion` already
   uses; a zero budget must not claim a positive definite covariance.
3. **A zero-radius noise ball with a centre**: fold the centre into the
   data (subtract it in `shifted_data` and the closed form's fit), or
   refuse a zero-radius ball with a nonzero centre; test that the set
   built with `noise=Ball(radius=0, center=b)` equals the one built on
   `d - b` with exact data.
4. **`outside()` at negative support**: `pairing > h + rtol * max(|h|,
   scale)` with the scale named, in `feasible.py` and `convex.py`; test the
   contact point of a set with negative support and the unit ball about
   `(-10, 0)`.
5. **The dual's emptiness answer** reporting a stand-in level of zero:
   report the feasible level as `Distance` does, catch only the
   unboundedness the dual raises, and write every level comparison as
   `level + rtol * max(|level|, 1)` consistently (`is_empty`, `admits`,
   `Search._general_fit`); test a prior written with level `-1`.
6. **The dual's absolute divergence threshold** (`-1e6` in property
   units): make it relative to the problem's scale (the prior-only bound
   `|h_prior(T* q)|` or the first dual value), or use the divergence test
   the docstrings describe; test the problem scaled by `1e7` with a short
   budget reports "not converged", never "empty".

## Should-fix

From `geometry.md`: degenerate (semidefinite) priors failing every support
algorithm and the emptiness check ignoring the instance's own search and
solver; `support_sweep(n_jobs=)` results not kept in the parent;
`n_jobs=` on `inner_hull`, `extent`, `outer_ellipsoid`; the quadratic
tier's `level_function()` without a gradient; `via=` forwarding on
`extent` and `fitting_model`; membership tolerances floored at one
(relative with a named scale, the same rule as package B); `Polytope &
Ball`; export consistency of the algorithm classes; `Ellipsoid` not
checking its covariance; the two docstrings that misstate the code. Note
the ±q duplication of the support sweep (CURRENT_STATE §7) if the fix is
local.

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
