# The documentation programme for 2.0

Approved by David on 2026-10-24 as the place to start; a working document
for the docs phase, deleted when the phase completes. Three tiers.

## Tier 1: reference examples (`examples/`)

The existing 33 stay as they are, one idea each. The pre-release work adds
four in the same style:

- a 3-D set sliced and drawn, with the restored slice plotter;
- `EvaluationError`: a forward that fails, a line search that survives it;
- `gradient=` against `derivative=` in `from_callables` (a coda to 5);
- a `CheckReport` walk for backend authors (a coda to 16 and 29).

## Tier 2: tutorials (scripts, replacing `tutorials/*.ipynb`)

An ordered course, narrative, runnable top to bottom, written against the
v2 names from the start.

> **Status, 2026-10-24: all twelve drafted, the notebooks not started.**
> First drafts are in `pygeoinf2/tutorials/` as percent-cell scripts
> (`# %%` / `# %% [markdown]`, so Jupyter and VS Code open them as notebooks
> and the two kept notebooks can be converted from them), with a README and
> `tests/test_tutorials.py`, which runs each one as `test_examples.py` runs
> the examples; `tutorials` joins `examples` in `test_code_practice`'s
> exemptions. Tutorials 10 (last section) and 11 need `planetmodel`, 12
> needs MFEM. Tutorial 7 uses `BackusGilbertParker`'s automatic choice only,
> so nothing in it depends on the `route=` spelling D-134 will change.
> Writing them turned up one library bug, fixed with a regression test:
> `GaussianMeasure.marginal` refused the marginal of any posterior on a
> direct sum (`_block_of` lost the covariance's claimed traits). And one
> wart, not fixed: a posterior on a direct sum lives on the forward
> operator's domain, which `RowLinearOperator` builds without the prior's
> labels, so its marginals are reached by position, not by name.

1. **First inversion** — a function on a circle from point observations,
   twenty lines end to end; the taste before any theory.
2. **Spaces** — vectors as backend objects, inner products, the Gram
   matrix, coordinates as a capability; why there are no dual spaces
   (`f.adjoint(1.0)`).
3. **Operators** — adjoint is not transpose, derivative is not gradient,
   taught slowly on a non-diagonal metric; Galerkin against component
   matrices (D-25).
4. **Solvers** — matrix-free Krylov, strictness, preconditioners, what an
   `Estimate` means.
5. **Gaussian measures on function spaces** — invariant, Sobolev, heat and
   power measures, sampling, push-forwards; what `log_density` refuses on
   a function space and why (the marker, taught as a feature).
6. **The linear forward problem** — error measures, the posterior as a
   measure with its density and evidence, credible sets on properties.
7. **Properties, not models** — property operators, Backus-Gilbert, the
   feasible set as the third kind of answer.
8. **Regularisation without priors** — the Tikhonov family, discrepancy
   principle, minimum norm; when and what they cost. No L-curve, by
   David's instruction.
9. **Nonlinear problems** — `at()`, linearization, MAP, the Laplace
   posterior, the failure protocol.
10. **Coupled problems** — direct sums, block operators, joint priors;
    `Layered` the worked case.
11. **Spaces whose structure varies** — sem1d from interval to layered
    planet: varying length scales, `r² dr`, balls and sections.
12. **Writing a backend** — the contract of CURRENT_STATE §6, the
    `check_*` ladder with `CheckReport`, MFEM the worked case, the
    lessons list.

Kept as notebooks, exactly two: a Colab quickstart (tutorial 1 in notebook
form, with the bootstrap cell) and a visual tour of the plotting layer,
where inline output earns the format.

## Tier 3: mini-apps (`miniapps/`, new)

MFEM-spirit: each a self-contained directory with a README stating the
problem, real or realistic data, several features composed, minutes to run.

1. **Great-circle tomography** — sphere, the shipped station and
   earthquake catalogues, path integrals, full Bayesian posterior with
   resolution analysis and corner plots. The flagship.
2. **A planet's density** — layered ball; mass, moment of inertia and
   discontinuity data; continuity by conditioning; then the
   Backus-Gilbert-Parker feasible set with route cross-checks — the
   Al-Attar (2021) ellipsoid on the problem it was written for.
3. **Flexure under load** — an MFEM space through `mfem_hilbert`, a
   spatially varying coefficient, a preconditioned large solve.
4. **An interface from gravity** — a genuinely nonlinear forward; MAP,
   Laplace, and the low-rank posterior with an information-gain budget
   once it exists.
5. **Two fields, one chain** — joint inversion for coupled fields on the
   sphere with a correlated prior; starts with independent factors,
   consumes `from_product(correlation=)` when it lands.
6. **Sampling the posterior** (post-release, the student's) — pCN against
   the Laplace Gaussian where the Gaussian is visibly wrong.

## Sequencing

Tier 2 and mini-apps 1-3 are docs-phase work before the release; mini-apps
4-6 may trail it. Coverage: every module of the package map appears except
the deeper convex machinery (deliberately quiet pending the redesign);
`datasets` is carried inside mini-app 1.
