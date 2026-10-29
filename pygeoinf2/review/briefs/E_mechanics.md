# Fix package E: mechanics in algebra and spaces

Findings files: `pygeoinf2/review/algebra.md` and `pygeoinf2/review/spaces.md`.
Package A has already changed the measure constructors in
`symmetric_space/base.py` and `spectral_element/base.py` (the identity
gate); build on the tree as you find it and do not touch those
constructors.

## Must-fix

1. **Block operators cannot be unpickled**: `BlockOperator`,
   `ColumnOperator`, `RowOperator`, `BlockDiagonalOperator` and their
   linear forms, and `SpectralBlockLinearOperator` in the spaces layer,
   have a `__new__` that requires arguments; make the first argument
   optional as `DirectSum` does; test a pickle round trip of each and
   `samples(4, n_jobs=2)` on a `from_product`, a `Layered.sobolev_measure`,
   a `correlated_measure` and a posterior built from one.
2. **`parallel_map(threads=)`** is public with no caller: remove it, and
   its docstring's mention.
3. **`AffineOperator` treating a callable translation as lazy**: Should,
   not Must, since no shipped backend has callable vectors; fix it if a
   clean way exists (an explicit `lazy=` flag), otherwise document.
4. **The ray travel-time operator**: add the order guard its own derivative
   applies (`path_integral_operator` refuses `L2`), and key the memo of
   traced rays on the model's components (`to_components(m).tobytes()` or
   a hash of them), not on object identity; test an in-place `axpy`
   followed by `F.at(m)`.
5. **A bounded `Box`, `Line` or `Plane` silently wraps points outside its
   domain**: refuse them as the spectral-element spaces do, with the same
   message shape; test both families.
6. **Matrix-free point evaluation** (decision 3 in SUMMARY.md): one keyword
   `route="auto" | "dense" | "local"` on both space families;
   `dense=` goes from the symmetric spaces and `form=` from the
   spectral-element ones; update every caller (grep `dense=` and
   `form="local"`/`form="dense"` across the package, tests, examples,
   tutorials and mini-apps) and the docstrings.
7. **The symmetric-space names leave the top level** (decision 3):
   `Box`, `Interval`, `Lebesgue`, `PeriodicBox`, `Sobolev` are no longer
   re-exported by `pygeoinf2/__init__.py` and live only in
   `symmetric_space`; update every caller (examples, tutorials, mini-apps,
   tests) to the scoped name. `ShellTransform` is no longer exported
   publicly (it stays importable from its module until the planetmodel
   update removes it).
8. **`HilbertSpace.mean` becomes `sample_expectation`** (decision 3), no
   alias; update every caller.

## Should-fix

From `algebra.md`: `check_operator` comparing a supplied
`components_action` with the operator's own action; `check_traits`
running a dense rank decomposition for every diagonal, sparse or
composed covariance claiming invertibility (only where the matrix is
already stored; otherwise a cheaper certificate or none); `D + I`
staying a generic sum (the same specialization as `D + 2I`); the
diagonal of a composition with a sparse factor densifying; Jacobi probing
a diagonal metric column by column (read `gram_diagonal` when the metric
is diagonal); `log_posterior(...).at(m)` costing two linearizations;
nonlinear block operators evaluating twice in `at()`; `matrix()` with
defaults failing on an operator without an adjoint when the codomain is
smaller; `Functional.from_callables(linearize=, hessian=)` reporting a
Hessian that `at()` does not carry; a check for a functional's Hessian;
a diagonal operator on a dense metric losing metric-independent traits.
From `spaces.md`: its Should list where local; `plot_set` taking
`n_jobs=` for its support solves and honoring `color=` on polytopes (the
examples review found both). The CURRENT_STATE §7 entry about dense point
evaluation being stale is the caller's to fix, not yours.

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
