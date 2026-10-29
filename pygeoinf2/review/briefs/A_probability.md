# Fix package A: measures that give wrong answers

Findings file: `pygeoinf2/review/probability.md` (and the identity-gate
findings in `spaces.md` and `examples.md`, which are the same item seen from
other layers).

## Must-fix

1. **Derived samplers draw about the wrong mean.** `GaussianMeasure.sample`
   adds the expectation to the sampler's draw, and `_combine_affine`,
   `_combine_add` and `from_product` supply samplers that already carry it.
   Make the three samplers return centered draws, state in the `sample=`
   docstring that a supplied callable returns the centered fluctuation,
   and test `translate`, a push-forward, a sum and a product of a
   sampler-only measure with a nonzero mean through `testing.check_measure`
   and a sample mean; also `problem.joint_measure(posterior)` and
   `data_measure_from_model_measure` on a non-zero-mean prior.
2. **The correlated product's precision factor** (`_correlated_product`)
   assumes each factor's precision factor is the exact inverse of its
   covariance factor. Build the latent whitening from the fixed relation
   `L_i^-1 == L_i* P_i` (the factor's adjoint after the precision) when the
   factor has a precision, or keep the precision factor only when a probe
   confirms `Li_i L_i xi == xi`; whichever you choose, the precision must
   invert the covariance after `-1.0 * a` and with a hand-built
   Cholesky/QR pair. Add that test. Also make `check_measure` apply the
   precision to the covariance's image of a probe when both exist, so this
   class of defect is caught by the check (the algebra review found the
   same gap; it is yours because it is a measure identity).
3. **`from_covariance_matrix` symmetrizes before checking.** Refuse when
   the asymmetry exceeds a round-off tolerance (say `1e-8` relative) with
   a message naming the likely wrong `form=`; symmetrize only below it.
4. **The identity covariance on a function space, every route** (decision
   1 in SUMMARY.md): one helper refusing the literal identity, a constant
   spectrum, an identity or scaled-identity covariance or factor, a
   constant deviation, a flat power, on a space with `is_function_space`,
   without `unsafe=True`; applied in `from_standard_deviations`, the
   `GaussianMeasure` constructor, `from_covariance_matrix`,
   `invariant_measure`, `heat_measure` at time zero, `spectral_measure`,
   `correlated_measure`, `power_measure`, each gaining `unsafe=` where it
   lacks one, with docstrings saying the trace-class condition beyond the
   literal identity is the user's. Then re-pose `tutorials/09_nonlinear.py`
   on a trace-class Sobolev prior with Newton-CG preconditioned by the
   prior covariance, rewriting its optimizer argument honestly against
   what it then prints (the examples review measured about 10 MAP
   iterations against about 300 for the white prior with L-BFGS); re-pose
   `tutorials/13_*.py`'s whitened identity prior the same way (its own
   coda in tutorial 14 shows the pattern); fix `tests/test_laplace.py`'s
   copy of tutorial 9's prior. Regenerate the notebooks
   (`pygeoinf2/tutorials/make_notebooks.py`) and run the tutorial tests.

## Should-fix

From `probability.md`: `marginal` losing samplers and precisions;
`from_product` dropping precision factors (dead branch); scaling a
sampler-only measure losing the sampler; `AbsolutelyContinuousMeasure.can_sample`;
`credible_set` claiming definiteness; `hilbert_schmidt_norm` and
`log_normalizing_constant` returning estimates as floats; `method="spectral"`
meaning two routes; sampling routes holding every draw (a streaming
quantile or a block, your call if local). Decision 2 in SUMMARY.md: the
model-space formalism's docstrings say it is for explicitly
finite-dimensional problems and low-dimensional surrogates, and the mixture's
docstring recommends the direct determinant over the stochastic route;
`consistency_set` is made consistent with the potential, and both say the
error measure must be trace class for a limit to exist (the inference
package may touch the same lines; coordinate by keeping the wording in the
probability files only).

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
