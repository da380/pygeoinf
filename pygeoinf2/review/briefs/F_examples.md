# Fix package F: the examples and tutorials

Findings file: `pygeoinf2/review/examples.md`, whose coverage map and
proposed reordering are the plan, as amended by decision 5 in
SUMMARY.md. Packages A to E have already landed: the identity gate on
every measure route, `route=` for matrix-free point evaluation, the
symmetric-space names scoped under `symmetric_space`,
`sample_expectation`, the low-rank mean's fix, `LaplaceApproximation(
gauss_newton=)`, the model-space outer ellipsoid by default,
`covariance=` on the point estimators if package D built it (check
`inference.md`'s Fixed section). Build on the tree as you find it.

## What to do

1. **Four new examples**, one idea each, in the reviewer's words:
   `no_coordinates` (a `HilbertSpace` whose vectors are plain Python
   lists; CG and a Gaussian run on it; `CholeskySolver` and `matrix()`
   refuse by name), `correlated_product` (two fields tied through their
   latent noise: marginals unchanged, the cross-covariance `R_ij L_i L_j*`,
   an unmeasured field informed only through `R`), `point_estimates`
   (`LeastSquares`, `MinimumNorm`'s discrepancy search and `BackusGilbert`
   with `resolution`, `propagated_covariance` and `bias`, the problem
   parameterized by the prior's factor so size is the prior's norm, or
   `covariance=` if package D built it), `low_rank_posterior` (the
   eigenvalues of `L* A* R^-1 A L`, the low-rank posterior at ranks either
   side of the eigenvalue of order one against the full one,
   `pointwise_variance_reduction`, a draw at the cost of a prior draw).
   Each under two seconds, printing what it means.
2. **The framing of point answers** (decision 5, David's words): the
   library exposes point estimators because they are wanted, but its view
   is that they are tools within the proper methods with uncertainty
   quantification and not generally interesting in themselves; there is
   no "optimal regularization parameter" or the like in this library. The
   point-estimates example says so in its opening comment, in one or two
   sentences, and presents each estimator once as what it is.
3. **The renumbering** as the reviewer's table proposes (gap-free, in the
   brief's order), with `ambient_ball(level=)` and `ambient_ball(operator=)`
   in the cut-down feasible-set example, `LaplaceApproximation(rank=)` in
   the nonlinear one, `Layered.sobolev_measure(correlation=)` in the
   layered planet, examples 13 and 17 merged, the worked example rewritten
   to use the estimators, example 23 cut to one idea. Follow every
   "example N" citation through examples, tutorials, mini-app READMEs and
   walkthroughs, `examples/README.md`, `tests/test_examples.py`, and
   `CURRENT_STATE.md`'s counts (that file you may edit for the counts and
   citations only). Use `git mv` for renames.
4. **The three scripts that contradict their output**: tutorial 7's bias
   claim, tutorial 13's L-BFGS agreement and its citation of a tutorial-10
   operator that is not there, example 22's CG claim; rewrite the prose
   to what runs. Also example 12's degenerate metric demonstration, the
   history and planning lines in tutorials 2 and 9, tutorial 12's claim
   about white noise and the contract.
5. **The tutorial course reordered to point, measure, set** (decision 5),
   with its citations followed through (the reviewer counted 67 in the
   tutorials and examples and 70 in the app walkthroughs); tutorial 14's
   order fixed within itself. Regenerate the notebooks with
   `pygeoinf2/tutorials/make_notebooks.py`; the Colab badges and install
   cell are to point at a placeholder the release will fill (say
   `<release-branch>`), not at `main`.
6. **The checks in the examples** take `measure=` where a measure exists
   and keep or print their `CheckReport` rather than discard it.
7. The Should and Could lists in `examples.md` where local (the two
   spellings of the circle, example 20's place, the README drift, the
   missing disk and parallelism examples if cheap).

Run `tests/test_examples.py` and `tests/test_tutorials.py` fast and slow
(`-m ""`), then the fast suite.

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

## Two additions from package E

- Tutorial 12's refinement study is fragile with white-noise probes: at 11
  nodes the adjoint defect ranges from 0.01 to 0.89 across seeds. Give its
  checks smooth probes through `measure=` (a Sobolev measure on the space),
  so that "falls steadily" is true for every stream, or loosen its claim.
- The default ball and disk mesh at radius one changed (45 nodes rather
  than 49 at a length scale of 0.3), so seeded numbers in examples on a
  unit disk or ball may move; re-measure what the prose quotes.
