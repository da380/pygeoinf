# pygeoinf 2.0: what it is now

This is the description of the package for someone who needs to use it or
review it: what it is, its conventions, what is exact and what is estimated,
where it still forms a dense matrix, and what is open. The reasons behind the
choices are in `DECISIONS.md`, cited here as D-nn. The history of how they
were reached is in git, at commit `83658d0` and before, where the design
journal, the review rounds and the functionality audit lived until they were
retired on 2026-09-18.

Last reconciled against the code on 2026-10-24, at 3053 passing tests in the
fast suite and 32 marked slow, the 35 examples all running. The release plan
in §10 was settled with David on that date and its first tranche landed the
same day as D-127 to D-133; the remaining items carry their D-numbers as
each change lands.

---

## 1. The one idea

Every space is a Hilbert space that knows its own inner product, and nothing
above it ever works in components unless it says so. That single constraint
is what most of the rest follows from:

- A **derivative** is a linear functional; a **gradient** is its Riesz
  representer. They are different objects and the library keeps them apart
  (D-26). Example 5 is about nothing else, and it is the mistake this design
  exists to make hard.
- A **Galerkin matrix** `G A_c` and a **component matrix** `A_c` are different
  representations of one operator. Anything that takes a matrix asks which
  (D-25).
- An **adjoint** is taken in the space's metric, so it is not a transpose. On
  a space with a non-diagonal Gram matrix the two differ, which is why the
  test suite runs so much of itself over one (D-106).

## 2. The package map

| module | what lives there |
|---|---|
| `algebra` | `HilbertSpace`, `CoordinateSpace`, `HilbertModule`, `MassWeightedSpace`, `DirectSum`; `Operator`, `LinearOperator`, `Functional`, the expression nodes, `DiagonalLinearOperator`, `Linearization` |
| `traits` | the seven trait flags and their closure and propagation rules |
| `probability` | `ProbabilityMeasure`, `GaussianMeasure`, `GaussianMixture`, `AbsolutelyContinuousMeasure` (D-132), products and push-forwards, conditioning, credible sets |
| `geometry` | convex sets (balls, ellipsoids, half-spaces, polytopes, intersections, oracle sets), level and sublevel sets, affine and linear subspaces, projectors, support functions |
| `numerics` | Krylov and direct solvers, preconditioners, randomised linear algebra, the operator calculus, log-determinants, optimisation and line searches, convex methods (bundle, KKT, Chambolle-Pock), quadratic programming, root finding, weighted chi-square quantiles |
| `symmetric_space` | the sphere on Driscoll-Healy or Gauss-Legendre grids; the periodic box in any dimension with the circle, torus, line and plane as named geometries; the bounded box and interval |
| `sem1d` | spaces in the eigenbasis of `1 - div(L² grad)` on a padded spectral-element mesh, for a length scale that varies: `interval` on the line under `dx`, `radial` for functions of radius under `r² dr`, `ball` for a ball or annulus, `layered` for piecewise-continuous fields over adjacent layers of any one of them (their direct sum, with points, jumps, integrals and layer-by-layer priors); Robin ends, a batched harmonic transform, a fit over the domain for sampled functions; wraps `planetmodel.randomfield`, an optional extra, and is imported explicitly (D-115 to D-124, D-126) |
| `inference` | forward problems with their likelihood `potential`, the estimator hierarchy, `LinearGaussianInversion` and the mixture inversion, `GaussianPosterior` (D-132), the point estimators and the Tikhonov family, `BackusGilbert` and `BackusGilbertParker`, the MAP and Laplace route, normal operators and their preconditioners |
| `plotting` | fields, points, paths, balls and networks on every symmetric geometry; shells, sections by any plane with their stations, and profiles of a `sem1d` ball, lines on its interval and radial profile (D-120); convex sets in one, two and three dimensions, with plotly as an optional 3-D engine (D-133); marginals, corner plots and error bounds |
| `backends` | MFEM two ways: `mfem` reads its matrices into a `CoordinateSpace`; `mfem_hilbert` keeps a plain `HilbertSpace` and lets MFEM do every computation (D-96) |
| `testing` | thirteen `check_*` functions, from the space axioms to affine operators and convexity (D-105), each taking `rtol=` and returning a `CheckReport` of what it measured (D-128) |
| `datasets` | the shipped station and earthquake tables, the cache directory, and the explicit downloads that refresh them (D-103) |
| `parallel` | `parallel_map` and `resolve_jobs`, the one place joblib is called (D-52) |
| `compat` | v1 class-name aliases and the v1-space adapter; goes with v1 (D-101) |

108 names are re-exported at the top level and the subpackages are importable
as `gi.inference`, `gi.numerics`, `gi.plotting` (D-5). `backends`, `testing`,
`compat` and `sem1d` are imported explicitly. Thirty-three examples under `examples/`
run as a slow test; their README says what each shows.

## 3. Conventions and units

These are the ones that bite.

- **American spelling in every identifier** (D-100): `minimize`, `center=`,
  `normalize=`, `Linearization`, `randomized`. Prose is not held to it.
- **`max_iterations` is the one name for a budget going in**; `iterations`
  is only the count on a result (D-99).
- **Every optional argument is keyword-only** (D-104), and `rng`, `n_jobs`
  and `solver` mean the same thing everywhere.
- **Points on a sphere are `(latitude, longitude)` in degrees** (D-2).
  `to_colatitude_radians` and `to_latitude_degrees` convert.
- **A sphere's vectors are `pyshtools.SHGrid` objects** (D-1); `grid_values`
  reaches the numbers and `from_grid_values` goes back. A box's vectors are
  arrays.
- **Coordinates on a box are physical lengths.** A circle point is an arc
  length; the circle and torus are unit by default, so there the coordinate
  is the angle (D-84). Radii are physical wherever a length is meant, and a
  method taking a radius says which.
- **Coordinate-free when it must be, and not otherwise** (D-15). Every
  algorithm runs on a space with no component map. On a `CoordinateSpace`
  the library may do its internal arithmetic on component arrays, with the
  metric entering through `apply_gram`; `uses_component_fast_paths` opts a
  space out.
- **Spherical harmonics are orthonormal, without the Condon-Shortley phase**
  (D-86). A box's `degrees` are `floor(|k|)`, so multiplicities are irregular.
- **Solvers are strict by default** (D-8): a solve that does not converge
  raises, and a non-finite residual raises whatever `strict` says (D-42).
- **A derived adjoint is never silent** (D-34): `with_probed_adjoint()` is
  the opt-in.
- **Densities refuse on a function-space discretisation** (D-131):
  `is_function_space` marks the spaces, and `log_density` and its family
  take `unsafe=True` to proceed, as point evaluation does under D-11. The
  dimension-robust object is a density relative to a reference measure.
- **An inference result carries its reference measure** (D-132): a Bayesian
  update returns a `GaussianPosterior`, a Gaussian that also knows the
  prior, `-Phi` against it, and the evidence; the algebra degrades it to a
  plain Gaussian rather than carrying a density it cannot transform.
- **A forward that cannot evaluate raises `EvaluationError`** (D-129): the
  line searches back off, a trust region shrinks, and an optimizer returns
  its last good state with `failure` set rather than crashing it away.
- **`power_measure` and `power_spectrum` agree on every space** (D-130):
  `power` means the `L2` spectrum whatever the host's order, and a draw's
  expected `power_spectrum` is the `power` it was given. The metric-native
  spelling is `invariant_measure`.

## 4. What is exact and what is estimated

Anything returning an `Estimate` is stochastic and carries its standard error
(D-51). `random_trace`, `random_diagonal` and `log_determinant` take an
`rtol` to sample *to*. The dense route reports zero error so callers treat
both alike.

`BackusGilbertParker` answers with a `FeasiblePropertySet` that may carry two
characterisations, chosen by the sets (D-79, D-80):

| support route | needs | what it is |
|---|---|---|
| `closed_form` | a ball prior, exact data | the ellipsoid of Al-Attar (2021), `dim(P) + 1` minimum-norm solves |
| `bisection` | two quadratic sets | the primal route, nested monotone root finds per direction, in the sets' own inner products |
| `dual` | any convex sets | a bundle minimisation over the data space per direction |
| `primal`, `kkt`, `smoothed` | as their solvers allow | the general engine's alternative solvers, for cross-checks |

| membership route | needs | what it is |
|---|---|---|
| `closed_form` | a ball prior, exact data | the joint map |
| `reduced` | two quadratic sets | the data-space reduction |
| `likelihood` | a ball prior, any differentiable confidence set | the minimum-norm fitting model against the prior's level |

Routes that both apply must agree, and the parity tests are the point of the
layer. On sixteen directions of the example the dual, primal, smoothed and
KKT solvers agreed to between 1e-3 and 1e-11 of each other in August 2026,
the KKT route's looser figure being its own limit with a tight noise set.

## 5. Where a dense matrix is still formed

Matrix-free is the default and the exceptions are deliberate. `matrix()` and
`diagonals()` are O(1) reads on a `MatrixLinearOperator` and on diagonal
operators, sums and scalings of them; elsewhere they cost one application per
column, and a composition never expands a low-rank product (D-33).

- `log_determinant(method="auto")` takes the dense route below
  `dense_limit=4000` only when the matrix is known or probing costs no more
  than the stochastic budget; `kl_divergence` refuses above 4000 rather than
  guessing; `ambient_ball` and `weakened_ellipsoid` take a dense generalised
  eigenproblem below 1024 (D-50).
- `GaussianMeasure.from_covariance_matrix`, `as_multivariate_normal` and
  `matrix(form=)` are dense by name, as is `nuclear_norm(method="dense")`.
- `PrimalKKTSolver` and `LevelKKTSolver` form two matrices on the **data**
  space, deliberately: that is what keeps the model space undiscretised.
- `LevelBundleMethod`'s master QP and LP bound are dense in the data dimension
  plus one (401 MB per master at 5000 data), v1's arrangement and a limit on
  the data size; `ProximalBundleMethod`'s subproblem is dense in the number of
  cuts, which is small by construction (D-56).
- `BackusGilbertParker`'s closed form computes the joint spectrum once per
  estimator; its property-space pseudo-inverse is a handful of rows (D-39).
- `backends.mfem.operator_from_bilinear_form` densifies and is documented as
  not the route for a real problem; `solver_from_bilinear_form` and
  `mfem_hilbert` are (D-96).
- `with_probed_adjoint()` and `DirectSolver` assemble what they are asked to,
  once, on first use (D-34, D-40).

## 6. Writing a backend

A backend supplies a `CoordinateSpace`. The contract is: `dim`,
`to_components`, `from_components`, `apply_gram`, `solve_gram`, `_key`, plus
whatever of the vector algebra (`add`, `subtract`, `scale`, `axpy`, `copy`,
`zero`) the vectors do not get for free. A coordinate-free backend supplies a
`HilbertSpace` and a mass operator, as `mfem_hilbert` does (D-22, D-96).

Then check it, in this order, and stop at the first failure:

```python
check_space(V, rng=rng, rebuild=lambda: build_again())
check_coordinates(V, rng=rng)
check_white_noise(V, rng=rng)
check_operator(A, rng=rng)
check_traits(A, rng=rng)
```

Each takes the `rtol` it enforces and returns a `CheckReport` of the worst
relative defect it measured (D-128). For a backend whose adjoint is
consistent only to discretisation level — a continuous adjoint — pass the
`rtol` it can meet and record the reported defect falling under refinement;
it is also the floor below which a solver tolerance asks for noise.

The lessons MFEM taught, which apply to any backend over foreign memory:

- **Who owns the buffer.** MFEM vectors can alias memory the library will
  free. Copy on the way in unless ownership is explicit.
- **Keep the solver alive.** `solver._pygeoinf_keepalive` is load-bearing:
  the wrapped object holds no reference of its own and the C++ side will free
  it.
- **The mass matrix is the Gram matrix.** That is the whole reason a finite
  element space fits without adaptation, and the case the design was built
  for.
- **Never densify to slice.** Take CSR rows and columns.
- **The form owns what `FormSystemMatrix` returns.** Under full assembly it
  eliminates in place; under partial assembly the constrained operator refers
  back to it. Operator, handle and form live and die together, or the next
  `Mult` segfaults.
- **Partial assembly of a matrix coefficient is broken in PyMFEM 4.8.**
  Scalar coefficients are fine; `mfem_hilbert.matern_measure` uses one when
  the field is isotropic and refuses otherwise (D-97).

## 7. Open questions

Recorded rather than settled, so nobody has to rediscover them.

- **`power_measure` on a Sobolev space.** Closed: D-130 made it mean the
  `L2` spectrum on every space, 2026-10-24.
- **The convex solvers' API.** Settled as D-134, 2026-10-24, in full:
  the estimator API stays; the algorithms move into the convex sets, whose
  probes dispatch through a form-ranked table (`via=` forces, an instance
  carries tuning); the dispatch machinery lives in the geometry layer with
  `FeasiblePropertySet` its first client and the rest of the hierarchy
  adopting incrementally; the solver zoo unifies underneath. Design fixed,
  implementation pending; D-82 and the dual-gap certificate become
  properties of the dispatch table's entries.
- **`route="auto"` on quadratic sets.** The KKT solver is the cheapest general
  solver on balls and ellipsoids and `auto` could prefer it over the bundle
  where both apply; left until measured (D-82).
- **`dense_limit` on `kl_divergence` and `ambient_ball`** keys on dimension,
  not application count (D-50). Fine until a covariance that applies a PDE
  solve meets it.
- **A posterior on a direct sum forgets the prior's labels.** Found
  2026-10-24 writing tutorial 10. `LinearGaussianInversion` puts the
  posterior on the forward operator's domain, and `RowLinearOperator` (like
  the other block constructors) builds that `DirectSum` with no labels, so
  `posterior.marginal("bias")` raises `KeyError` although the prior was
  built with `labels=` and compares equal to the domain. Marginals are
  reached by position meanwhile. To look into: whether the block operators
  should take `labels=`, or the estimator should prefer the prior's domain
  when the two are equal.
- **The public documentation is still v1's.** `README.md`, the Sphinx tree
  under `docs/` and the notebooks under `tutorials/` describe `pygeoinf`,
  not this package. Pointing them at 2.0 was deferred to the rename.

## 8. Decisions that change what a caller writes

`DECISIONS.md` has the full list. The ones a v1 user meets first:

| | |
|---|---|
| D-1 | sphere vectors are `SHGrid` |
| D-2 | points are `(lat, lon)` in degrees |
| D-4 | `from_matrix(..., form=...)` replaces the two matrix constructors |
| D-8 | solver defaults are `rtol=1e-8`, `strict=True` |
| D-11 | point evaluation is refused below the Sobolev order that admits it, with `unsafe=True` to override |
| D-12 | `path_integral_operator` is the integral; `path_average_operator` normalises |
| D-16 | there are no dual spaces; `f.adjoint(1.0)` is the representer |
| D-79 | `BackusGilbertParker` is the one feasible-set estimator; the sets choose the route |
| D-99 | v1's names where there was no strong reason to change: `forward_problem`, `property_operator`, `kalman_operator` |
| D-100 | American spelling |

## 9. Not in 2.0

Deferred by agreement, not by omission (D-112, D-113): `dynamical_system`
and sequential data assimilation, PETSc, `parallel=` inside operator actions,
full function-space MCMC (the hooks exist), and the v2 documentation set.

## 10. Between here and a release

The plan settled with David on 2026-10-24, reviewed issue by issue. Three
inputs informed it beyond the code: the sampling brief
(`docs/sampling_brief.pdf`, the student's proposed function-space MCMC API;
its reasoning document `sampling.tex` is to be obtained), the sl3dpp request
list (`~/dev/sl3dpp/docs/dev/pygeoinf-integration.md` §8), and a survey of
planetmodel 1.3.2.

**Pre-release code, decided:**

- DONE (D-127). `MaximumAPosteriori.objective` builds its functional with `linearize=` so
  the value and the derivative share one `forward.at`; today each runs the
  forward, which is two solves per line-search point.
- DONE (D-129). `EvaluationError` in `algebra`: a forward that cannot evaluate raises it,
  both line searches treat it as a step too far, and the failure field on
  the optimization result is named now even if populated later.
- DONE (D-128). The `check_*` functions take `rtol=`/`atol=` again (v1's checks had
  `check_rtol`/`check_atol`; losing them was a regression) and report what
  they measured: a small `CheckReport`, `check_operator`'s adjoint defect
  first. The measured defect is also how a continuous adjoint is validated
  under refinement, and what caps a solver's meaningful tolerance.
- DONE (D-131, as `is_function_space`). The function-space marker: a boolean property on `HilbertSpace`,
  propagating like the traits (`DirectSum` if any summand, preserved by
  `MassWeightedSpace`); `log_density`, `mahalanobis_squared`,
  `credible_set` and their family refuse on a function-space discretisation
  without `unsafe=True`, as D-11 does for point evaluation.
- DONE (D-132; the density member is `relative_log_density`, `log_density` staying the gated volume one, and `ForwardProblem.potential` came with it). `AbsolutelyContinuousMeasure` on `ProbabilityMeasure`: a `reference`, a
  `log_density` (a `Functional`), a lazy `log_normalizer`, `relative_to`;
  built from a reference and a density, never from two arbitrary measures.
  The linear Gaussian posterior and `LaplaceResult.measure` return a
  subclass of it and `GaussianMeasure` (reference the prior, normalizer the
  evidence, cheap in the data-space formalism). A push-forward degrades to
  the plain Gaussian — the answer to D-73's worry, not an override of it.
- DONE (D-130). `power_measure` means the `L2` spectrum everywhere (§7).
- DONE, and it turned out to predate the plan: `from_callables` already took `gradient=` with the exclusivity rule, and `Functional.gradient(x)` existed. `Functional.from_callables` takes `gradient=` as an exclusive alternative
  to `derivative=` — user code may apply the Riesz map more cheaply than
  the library can — and `gradient(x)` is the accessor. Everything
  downstream still consumes the derivative; the gradient is a doorway, not
  a second pathway.
- DONE (D-133). 3-D slice plotting returns (the audit's drop is
  reversed): `plot_set` grew the third dimension in v2's own
  route-by-capability style rather than as a port of v1's
  `SubspaceSlicePlotter`; plotly is the optional `plot3d` extra. Example 34
  shows it.
- The convex solvers' API is redesigned to David's specification (§7).
- planetmodel maintenance, half done: the pin is `planetmodel>=1.3`, the
  stale `[harmonics]` extra is gone, the venv and lock are on 1.3.2 (whose
  grids carry a `kind` that the transform now checks by name), and
  `test_sem1d_ball.py` no longer needs pyshtools. Still to do: the
  planetmodel 1.4 release, then the pin to `>=1.4` and the deletion of
  `sem1d/_transform.py`. The migration itself is built and tested:
  `planetmodel.harmonics.ShellTransform` sits uncommitted in that repo (on
  its `main`; its rules want the commit on `develop`), same names, with the
  version bump left for the release branch as its convention requires.

**Decided now, built after the release:**

- `GaussianMeasure.from_product(correlation=)`: a matrix on the latent
  white noise, unit diagonal, `None` the independent product of today;
  `correlated_measure` becomes a wrapper keeping its fused spectral path;
  sem1d spaces gain correlated fields. The student builds it.
- `formalism=` opens on the nonlinear route with `"data_space"` the default
  (D-74's rule everywhere), and `with_probed_derivative()` mirrors D-34:
  the derivative assembled from `dim(Y)` adjoint applications, exactly
  adjoint to the continuous adjoint that built it. For small data, and not
  privileged: the library offers the range of adjoint strategies — discrete
  adjoints are the backend's business, continuous adjoints are supported
  honestly through measured defects and solver floors, the representer
  route exists where data are few.
- One low-rank posterior class: a `GaussianMeasure` carrying the
  `(C, V, D)` structure of the prior-preconditioned misfit Hessian's
  leading eigenpairs, which is an `AbsolutelyContinuousMeasure` with an
  r-dimensional density. Designed jointly with the student before either
  side writes code; it serves the Laplace route, the sampling layer's
  references, and is the designated answer to deflated pointwise variance
  on spaces with no closed form (plain Monte Carlo until then).
- Sampling: the presets (`PCN`, `MALA`, `HMC`, `SMC`) and the result types
  (`SampledPosterior`, `WeightedEnsemble`) are the public contract; whether
  they are built as kernels-and-driver or as discretised SDE dynamics is
  the student's call, made against a working pCN benchmark. Unadjusted
  schemes on function spaces: leaning no.

**Confirmed drops** (v1 things that stay gone): derivative-free
minimization, BiCG, and a separate deflated pointwise-variance estimator.

**Docs phase:** three tiers. The reference examples here stay as they are,
one idea each. The v1 notebooks under `tutorials/` are replaced by
tutorial *scripts* covering the whole core, with a small number of
notebooks kept as entry points. Mini-apps in the MFEM spirit — each a
realistic inverse problem end to end, deeper than any one feature — are
new. The approved list is `EXAMPLES_PLAN.md`, a working document that is
deleted when the phase completes. `README.md` and the Sphinx tree are
rewritten against the new name, after the rename, not before.

**Release mechanics, in order:** tag `v1-final`; rename `pygeoinf2/` to
`pygeoinf/`; delete v1, `compat.py` and its test, `work/`, and v1's root
`tests/`; CI green on the renamed tree, including the MFEM extra on every
platform it installs on; version 2.0.0; readthedocs flips at the tag with
v1's docs kept under a version selector.

**Post-release tracks:** the student's sampling package (the brief's order
of work, starting with the marker-gated methods already done pre-release
and `from_product(correlation=)`); the sl3dpp items (the data-space route
and low-rank posterior above, then blocks with a size hint, process-safety
declarations, the sem1d separable-basis export, optimizer checkpointing,
resource-owning linearizations, and the Kronecker space, which sl3dpp has
offered to supply as a backend); and whatever the convex redesign defers.
