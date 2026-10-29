# Fix package G: the mini-apps after the review's library changes

No findings file of its own; read `pygeoinf2/review/SUMMARY.md` and the
Fixed sections at the end of `probability.md`, `numerics.md`,
`geometry.md`, `inference.md`, `algebra.md` and `spaces.md` for what
changed in the library, then each app's `README.md`, `IMPLEMENTATION.md`
and source under `pygeoinf2/miniapps/` (point_data, tomography, mfem,
waveform, planet_density, cpp_backend) and the conventions in
`pygeoinf2/miniapps/README.md`. Every app's slow test passes on the
current tree; the question is whether what the READMEs and walkthroughs
say is still what runs.

## What to do, app by app

1. **Re-measure every number the README quotes** at the full scale, and
   every command-line override example, with the library as it now is.
   Two changes move numbers on purpose: the weighted chi-square behind
   `ambient_ball` is now the same in any units (radii change wherever the
   prior's variances were far from one), and the default ball and disk
   mesh at radius one changed (seeded draws on the tomography disk and
   ball move). Update the prose where a number changed and say what the
   new numbers show; no before/after, no history.
2. **Use what the library now offers where the app built it by hand**:
   `route="local"` for matrix-free point evaluation (already renamed by
   package E; check), `sample_expectation` (renamed), `n_jobs=` on
   `outer_ellipsoid`, `inner_hull`, `extent`, `plot_set` and the mixture;
   the tomography app's hand re-seating of its layered prior factor on the
   layered space is no longer needed (`from_product(domain=)` seats it)
   and goes; the C++ app's `radius_of` helper goes, the ellipsoid now
   carrying its `radius`; the waveform app's Laplace part may now use
   `rank=` where it helps, and says so if it does.
3. **Walkthroughs** (`IMPLEMENTATION.md`): follow the renames and the
   removed hand-built pieces; a sentence per app on the set hardening if
   it still describes the old ball radius computation.
4. **Keep the rules**: quick scale first; no single command over two
   minutes (point_data part E at full is measured direction by direction
   as before); every scale field a command-line option; figures per
   method, titled; no development narrative in any document.
5. Run each app's slow test at the end:
   `MPLBACKEND=Agg poetry run pytest pygeoinf2/tests/test_miniapp_<app>.py
   -m "" -q -p no:cacheprovider`.

Report per app: which numbers changed and by how much, what hand-built
code went, timings, test counts, anything in the library that got in the
way (with a reproduction), anything you could not do.

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
