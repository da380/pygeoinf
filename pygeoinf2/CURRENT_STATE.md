# pygeoinf 2.0: what it is now

This is the description of the package for someone who needs to use it or
review it: what it is, its conventions, what is exact and what is estimated,
where it still forms a dense matrix, and what is open. The reasons behind the
choices are in `DECISIONS.md`, cited here as D-nn. The history of how they
were reached is in git, at commit `83658d0` and before, where the design
journal, the review rounds and the functionality audit lived until they were
retired on 2026-09-18.

Last reconciled against the code on 2026-10-28, at 3206 passing tests in the
fast suite and 32 marked slow; the 36 examples and 14 tutorials last ran in
full on 2026-10-29. The release plan in §10 was settled with David on
2026-10-24 and its first tranche landed the same day as D-127 to D-133; the
feasible-set rebuild and the ray operator followed on 2026-10-27 (D-134 to
D-156) and the review clean-up on 2026-10-28 (D-157 to D-160); the
remaining items carry their D-numbers as each change lands.

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
| `geometry` | convex sets (balls, ellipsoids including semidefinite ones, half-spaces, polytopes, intersections, oracle sets), the conversions between a set's descriptions (D-135), level and sublevel sets, affine and linear subspaces, projectors, support functions; the feasible property set with its algorithms `ClosedForm`, `Primal`, `Search`, `Dual`, `Splitting`, `Distance`, `SProcedure` (D-142), and the ellipsoidal descriptions of D-141: the S-procedure outer ellipsoid, the bound on a Minkowski sum of ellipsoids, a polytope's inscribed ellipsoid, a point set's enclosing ellipsoid |
| `numerics` | Krylov and direct solvers, the spectral family solver, preconditioners generic and structure-aware, the normal operators and the Tikhonov family with its misfit search (D-142), randomised linear algebra, the operator calculus, log-determinants, optimisation and line searches, convex methods (bundle, KKT, Chambolle-Pock), quadratic programming, root finding, weighted chi-square quantiles |
| `symmetric_space` | the sphere on Driscoll-Healy or Gauss-Legendre grids; the periodic box in any dimension with the circle, torus, line and plane as named geometries; the bounded box and interval; `evaluate_gradient` at scattered points (ducc0 on the sphere, the `ik`-scaled transform on the boxes) and the nonlinear `ray_travel_time_operator`, bending or shooting, on every geometry, with its linearization and its Hessian action (D-156) |
| `spectral_element`, `interval`, `ball` | spaces in the eigenbasis of `1 - div(L² grad)` on a padded spectral-element mesh, for a length scale that varies, reached by geometry (D-161): `interval` on the line under `dx` with `interval.radial` for functions of radius under `r² dr`, `ball` for a ball or annulus, each with its `Layered` for piecewise-continuous fields over adjacent layers (their direct sum, with points, jumps, integrals and layer-by-layer priors, D-126); `spectral_element` is the shared core (`SpectralElementSpace`, `AxisSpace`, the mesh, Robin ends, the batched harmonic transform, the fit over the domain for sampled functions), wrapping `planetmodel.randomfield`, an optional extra; all three are imported on demand (D-115 to D-124) |
| `disk` | fields on a disk or an annulus in the plane, the two-dimensional `ball`: Fourier modes of the angle times radial eigenfunctions of `1 - div(L² grad)` under `r dr` on a padded spectral-element mesh, with `Lebesgue`, `Sobolev` and `Layered` (jumps taken at angles), points `(radius, angle)` in degrees anticlockwise from the `x` axis; the radial family is `planetmodel`'s with the measure changed (`spectral_element._cylindrical`: `CylindricalOperatorFamily`, the FFT `RingTransform`, `PolarBasis`); drawn whole in the plane by `plotting`, with points and polyline paths; imported on demand |
| `inference` | forward problems with their likelihood `potential`; the estimator hierarchy, every class a `PointEstimator`, `SetEstimator` or `MeasureEstimator` (D-163): the linear point estimators `LeastSquares`, `MinimumNorm` (the discrepancy principle its method), their constrained forms and `BackusGilbert` (the averaging-kernel estimate, with its resolution, propagated covariance and bias), the generic `Minimization` with `NonlinearLeastSquares`, `MaximumAPosteriori` on it, `ConstrainedMinimization` with `Occam`; `LinearGaussianInversion`, the mixture inversion, `GaussianPosterior` (D-132) and `LaplaceApproximation` as the measure estimators; `BackusGilbertParker` over the geometry layer's feasible set, linearised about a model for a nonlinear forward; the normal operators, the Tikhonov family and the structure-aware preconditioners are re-exported from `numerics` |
| `plotting` | fields, points, paths, balls and networks on every symmetric geometry; shells, sections by any plane with their stations, and profiles of a `ball`, a `disk` whole in the plane with its points and paths (D-175), lines on an `interval` and its radial profile (D-120); convex sets in one, two and three dimensions, with plotly as an optional 3-D engine (D-133); marginals, corner plots and error bounds |
| `testing` | thirteen `check_*` functions, from the space axioms to affine operators and convexity (D-105), each taking `rtol=` and returning a `CheckReport` of what it measured (D-128) |
| `datasets` | the shipped station and earthquake tables, the cache directory, and the explicit downloads that refresh them (D-103) |
| `parallel` | `parallel_map` and `resolve_jobs`, the one place joblib is called (D-52) |
| `compat` | v1 class-name aliases and the v1-space adapter; goes with v1 (D-101) |

111 names are re-exported at the top level and the subpackages are importable
as `gi.inference`, `gi.numerics`, `gi.plotting` (D-5). `testing`,
`compat`, `spectral_element`, `interval`, `ball` and `disk` are imported explicitly. Thirty-six examples under `examples/`
run in the suite, two of them marked slow; their README says what each shows. Fourteen tutorials
under `tutorials/` with generated notebooks (two marked slow), and the first mini-app under
`pygeoinf2/miniapps/` (the C++ vector backend, D-169), run in the fast suite.
MFEM is not in the package: `miniapps/mfem/` wraps it, as a finite element
space and as a solve inside an operator on a box, and its test is slow and
needs the `mfem` extra, which is there for it.

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

`BackusGilbertParker` answers with the geometry layer's `FeasiblePropertySet`
(D-142, D-143): the two sets' form decides the algorithms, one per probe,
and `via=` on a probe names another or passes an instance with tuning
(D-136):

| form | support | inclusion | feasibility | outer ellipsoid |
|---|---|---|---|---|
| `closed_form` (quadratic prior, exact data) | `closed_form` | `closed_form` | `closed_form` | `closed_form` (the set itself) |
| `quadratic` (a ball or an ellipsoid each side) | `primal`, then `dual`, `splitting` | `search`, then `distance` | `search`, then `dual` | `s_procedure` (D-141) |
| `general` | `dual` (support functions, converted from a smooth level function if need be), `splitting` (projections) | `search` (smooth level functions), `distance` (a finite support function) | `search`, `dual` | none |

Every support value is an outer bound with a certified bracket (D-148);
the level side decides points and gives inner bounds. Algorithms that both
apply must agree, and the parity tests are the point of the layer: the
primal and the dual agree to `1e-6` on balls, the dual with the quadratic
tier to `1e-4` on a quartic prior, the splitting with the dual to `1e-5`.
The linear certificate, the Minkowski sum a fixed `X` allows, valid for
any `X` by weak duality (D-140), is the `linear_certificate` route of the
same set, never the automatic choice, with `X` from the quadratic surrogate
or supplied; `BackusGilbert` is the point estimator with that `X` and no
prior bound (D-163). A nonlinear forward enters through `model=`: the set
is built on the linearisation with one data shift, exact for the linearised
problem and local for the real one, in absolute property values. The
S-procedure ellipsoid, model-space by default, is checked against the
set's support along its principal axes before it is returned (D-203); a
polytope's `inscribed_ellipsoid()` describes the polytope and a point set's
`enclosing_ellipsoid()` the points, neither the set (D-141).

## 5. Where a dense matrix is still formed

Matrix-free is the default and the exceptions are deliberate. `matrix()` and
`diagonals()` are O(1) reads on a `MatrixLinearOperator` and on diagonal
operators, sums and scalings of them; elsewhere they cost one application per
column, and a composition never expands a low-rank product (D-33). A
diagonal factor in a composition scales the neighbouring matrix's rows or
columns and is never written out as a square (D-170).

- `log_determinant(method="auto")` takes the dense route below
  `dense_limit=4000` only when the matrix is known or probing costs no more
  than the stochastic budget; `kl_divergence` refuses above 4000 rather than
  guessing; `ambient_ball` takes a dense generalised
  eigenproblem below 1024 (D-50).
- `GaussianMeasure.from_covariance_matrix`, `as_multivariate_normal` and
  `matrix(form=)` are dense by name, as is `nuclear_norm(method="dense")`.
- `ProximalBundleMethod`'s subproblem is dense in the number of cuts, which
  is small by construction (D-56).
- The feasible set's closed form holds a `dim P × dim P` shape matrix and
  `Search` a `dim P × dim P` kernel Gram, shared across data; the only
  dense-in-`N_d` piece is the opt-in `SpectralFamilySolver` (D-145).
- Every ellipsoid on the property space is a `dim P × dim P` matrix: the
  S-procedure's shape per weight, `sum_of_ellipsoids`, a polytope's
  inscribed ellipsoid (Newton on `dim P (dim P + 3) / 2` unknowns) and a
  point set's enclosing ellipsoid (D-141).
- The MFEM mini-app's `operator_from_bilinear_form`
  (`miniapps/mfem/src/matrix_space.py`) keeps
  MFEM's sparse matrix (D-96); nothing in its space modules densifies.
- `with_probed_adjoint()` and `DirectSolver` assemble what they are asked to,
  once, on first use (D-34, D-40).
- `SpectralFamilySolver` factors a damped family's two Galerkin matrices
  once, dense in the family's dimension, and only when passed as the
  solver (D-145).

## 6. Writing a backend

A backend supplies a `CoordinateSpace`. The contract is: `dim`,
`to_components`, `from_components`, `apply_gram`, `solve_gram`, `_key`, plus
whatever of the vector algebra (`add`, `subtract`, `scale`, `axpy`, `copy`,
`zero`) the vectors do not get for free, and `random` and `white_noise`,
which every check and every randomised method draws from. A coordinate-free
backend supplies a `HilbertSpace` and a mass operator, as the MFEM
mini-app's `src/hilbert_space.py` does (D-22, D-96); its vectors need no array behaviour at all, which
`tests/test_no_array_backend.py` holds the library to over a space whose
vectors are plain Python objects. What such a space cannot be asked for
refuses by name (`assembled()`, the direct solvers, `random_diagonal`,
`matrix()`, the dense routes); the feasible set's membership side is the
one place the refusal is per probe, the support side being free.

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

A foreign library can also sit inside an operator between the library's own
spaces, with no space of its own: the MFEM mini-app's `src/box_operator.py` is a
finite element solve on a periodic Sobolev box, its adjoint written from the
transposes of the two transfers with the box's Gram factor and its inverse
on either side, and the ladder above proves it to the solver's tolerance.

The lessons MFEM taught, worked through in `miniapps/mfem/`, which apply to
any backend over foreign memory:

- **Who owns the buffer.** MFEM vectors can alias memory the library will
  free. Copy on the way in unless ownership is explicit.
- **Keep the solver alive.** The wrapped object holds no reference of its
  own and the C++ side will free what it points to; the references live on
  the pygeoinf object that owns the solver (the operator's closure, or the
  space), never as attributes on the foreign object (D-160).
- **The mass matrix is the Gram matrix.** That is the whole reason a finite
  element space fits without adaptation, and the case the design was built
  for.
- **Never densify to slice.** Take CSR rows and columns.
- **The form owns what `FormSystemMatrix` returns.** Under full assembly it
  eliminates in place; under partial assembly the constrained operator refers
  back to it. Operator, handle and form live and die together, or the next
  `Mult` segfaults.
- **Partial assembly of a matrix coefficient is broken in PyMFEM 4.8.**
  Scalar coefficients are fine; the mini-app's `matern_measure` (`src/hilbert_space.py`) uses one when
  the field is isotropic and refuses otherwise (D-97).

## 7. Open questions

Recorded rather than settled, so nobody has to rediscover them.

- **`power_measure` on a Sobolev space.** Closed: D-130 made it mean the
  `L2` spectrum on every space, 2026-10-24.
- **The convex solvers' API.** Settled as D-134 (2026-10-24) and in detail
  as D-135 to D-152 (2026-10-27): three forms decide the algorithm, `via=`
  forces one on a probe, every answer carries a certified bracket, the
  feasible set and its algorithms move to `geometry` and the damped-family
  machinery to `numerics`, matrix-free throughout with the dense spectrum
  as an opt-in solver. Built 2026-10-27 (D-153); the paper-scale
  benchmarks are deferred (D-155); their scripts were cut from the tree on
  2026-10-28 (David: come back to them later) and are in git at `8c53839`
  under `benchmarks/`. Left open from the build: `Cylinder` (D-138), a quadratic
  set with an explicit kernel basis for the true-kernel case, droppable;
  the affine constraint on the dual's certificate for degenerate priors;
  and the dual does not converge on the flexure–gravity problem (610 data
  in two blocks of different units, chi-squared noise set) at 1 500 bundle
  iterations, where the primal is certified and is what is reported.
  D-82's open question is closed by D-136 (the tiers are nested
  specialisations, nothing to rank by timing).
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
- **The splitting route's inner value is attained only to its tolerance.**
  Found 2026-10-28 when its test tolerance was loosened from 1e-9 to 1e-7:
  `Splitting` reads `inner` off a primal iterate that lies in the sets
  only to the feasibility residual, so `inner` exceeded `value` by 1.5e-8
  and the bracket was not a certificate on that side. The outer value is
  sound by weak duality. A certified inner value needs a feasible point,
  which for two balls a final exact correction could supply; documented on
  the class meanwhile.
- **The Woodbury preconditioner reads a Tikhonov member two ways.** Along
  a family (D-170) the multiplier is the factor on the noise and the prior
  covariance is the preconditioner's own, `t`-independent. Asked to invert
  one operator, `from_normal` reads `prior_covariance` off it, which for a
  Tikhonov operator is `(1/t) I`, and `_invert` divides by the member's
  `scale`: right for that one member, but a preconditioner built with
  `C = I` and handed a Tikhonov member at damping 10 inverts the member at
  damping 1 and divides by 10, a condition number of about 10 to pay (25
  iterations against 12 in the point-data mini-app). One reading should
  win: the multiplier read off the member, with `from_normal` stripping
  the damping from a Tikhonov prior. Not done on 2026-10-28.
- **Parallel randomised estimates oversubscribe.** Seen 2026-10-28 on the
  point-data mini-app at degree 128: `random_eig(n_jobs=8)` through
  `apply_block` sat at 580% CPU in the main process with the workers idle,
  where the serial run took about two minutes; BLAS threads and the worker
  pool contending, most likely (D-52's measurements point the same way).
  The mini-app no longer uses it (D-170); not investigated further.
- **The splitting route's inner end on a direction the data cannot see**
  (2026-10-28, the rest of that build's findings fixed as D-181; the
  dense point evaluation went with D-183's `route="local"`). It exceeded
  its outer by 1e-4 and took 28 s; D-202's relative stop keeps the inner
  end at or below the outer on the cases measured since.
- **Three findings from the revision of the mini-apps' figures** (2026-10-29).
  `plotting.plot_shell` draws on the ball's own angular grid, which at
  degree 14 is a 15 × 29 mosaic, and cannot be smoothed: passing
  `shading=` raises a duplicate-keyword `TypeError` because the sphere
  renderer sets it itself (`plotting/sphere.py`); the app used
  `contour=True`. There is no public way to draw a path on a section of a
  ball: the section frame is private and `plot_section_points` places
  markers only; the app mirrored the convention. A sampled pointwise
  standard deviation evaluated at a point off the grid is the projection
  of a grid product (the rule of D-183), which smooths it: at the
  tomography quick scale 2000 samples give 0.314 against an exact 0.380,
  while sampling at the point directly gives 0.379; worth a sentence in
  `sample_pointwise_std`'s docstring and perhaps a `points=` argument.
- **Two mixture findings from the tomography app** (2026-10-29). The
  plotting helpers `plot_densities` and `moments` sample a
  `GaussianMixture` (20 000 draws a panel, 40 to 80 s at the app's size)
  though a mixture's one-dimensional marginals are closed form, a
  weighted sum of Gaussian densities; worth a closed-form route.
  `LinearGaussianMixtureInversion` factorizes the identical normal
  operator once per component when the components share a covariance;
  harmless at five components, wasteful at fifty.
- **Signals from the mini-apps' implementation walkthroughs** (2026-10-29).
  A row of derivative components into a Euclidean data space is
  `from_matrix(..., form="components")` in the planet-density app and
  `form="galerkin"` in the tomography and MFEM apps, both right since the
  codomain's Gram is the identity: one convention across the apps, or a
  named constructor for "rows of derivative components", would spare the
  explanation. The point-data app runs no checks at all and the
  planet-density app checks its linear rows only in its test; each app
  should exercise the ladder on its own operator in part A.
  `OptimizationResult` could carry the
  gradient norm per iteration beside the values in `history`; the
  waveform app wraps its objective to record them. `test_code_practice`
  skips Markdown, so the no-decision-number rule is not enforced on the
  READMEs and walkthroughs.
- **Findings from re-posing the mini-apps** (2026-10-29, D-191; the
  layered seating and the sweep's threads fixed as D-195, the weakened
  ellipsoid's two gone with it, D-194). (4) A spectral-element measure
  calibrated by `pointwise_std=` is `S C S*`, with no precision and no
  diagonal, so its norm and its ellipsoid go through solvers and the
  Lanczos calculus at every application; the tomography app calibrates
  by `amplitude=` instead, uniform to 2%. (5) A symmetric space's
  invariant measure has a diagonal precision but no `precision_factor`,
  which the planet-density pattern of whitening reads; D-204's low-rank
  posterior builds its precision without one. (6) Done, D-203: the
  support sweep shares the kernel solve between `±q`.
  (7) The data-space `Primal`'s solve at its floor is most of a
  direction's cost on an ill-preconditioned family: 350 of about 770 CG
  iterations at the point-data quick scale at the default floor, half
  the applications at `floor=1e-4`.
- **planetmodel's spectral basis does not normalize eigenvector signs**
  (2026-10-29, D-199). `eig_banded` returns eigenvectors of either sign,
  so two meshes that differ by one element can flip a basis function, and
  a seeded draw of the same measure with it. `Radial(12,
  inner_radius=1221.5, radius=3480.0, order=2.0, length_scale=200.0)`
  against the same in Earth radii, before the element-length margin, had
  its first outer-core basis function flipped. For the planetmodel 1.4
  update after the launch: fix each eigenvector's sign (first nonzero
  component positive) in `SpectralBasis`.
- **Left open by the 2.0 review's fix packages** (2026-10-29, D-200 to
  D-205; each needs a decision or new API, and none blocks 2.0):
  - *Data-space S-procedure*: its repair through the resolvent identity
    and a floored walk, so that it is right where the data outnumber a
    slack prior; today it raises there and the model-space form is the
    default.
  - *`covariance=` and `reference=` on `LeastSquares` and `MinimumNorm`*,
    so the prior-norm whitening the apps build by hand through
    `parameterized(covariance_factor)` becomes a knob: the Tikhonov family
    would carry a model-space weighting and `MinimumNorm`'s derivative be
    re-derived.
  - Gates on `as_multivariate_normal` and `kl_divergence` on a
    function space: docstrings only for now.
  - *Degenerate (semidefinite) priors* still go through `_covariance_of`
    in the model-space primal and the splitting route, the one route that
    answers them; which cached pieces of a feasible structure may cross a
    process boundary is unruled; appending a half-space to a polytope
    (outer or inner bound) is undecided.
  - *Clarabel's absolute tolerances* still apply to the bundle subproblem
    at tiny scales (conservative, costs iterations only). The first trial
    step along a gradient direction is `|f| / ||g||^2`, Polyak's step with
    a target of zero, and one when the value at the start is zero: the one
    guess with the right units, corrected by the Wolfe search's secant
    extrapolation and clipped sectioning by as many decades as it takes.
  - *`Ball.contains` at radius zero* has no tolerance, so a zero-radius
    ball rejects its own center after rounding; `InclusionResult.multiplier`
    means opposite things on the quadratic and general searches.
  - *`Functional.gradient`'s cost model*: for a composed functional the
    derivative route can cost two forward evaluations where `at()` costs
    one. One `atol` default across the checks is a policy call.
  - *`random_diagonal` as an `Estimate`* needs an array-valued estimate;
    `log_determinant` samples to `sample_rtol`, which §4 and D-51 should
    say; LSQR's `damping` and the five tolerance spellings are renames not
    taken; Jacobi on a nonsymmetric operator; bundle reentrancy and
    `with_tolerance`; `max_cg_iterations` and the waveform `factor=`.
  - *Spaces*: the base `Layered` accepting disks; a domain quadrature on
    spectral-element spaces; `import pygeoinf2` loading planetmodel
    eagerly (a module `__getattr__` and lazy renderer registration would
    defer it); the length scale carried through `Lebesgue`; radial
    pointwise calibration.
  - *`check_operator`'s `measure=`* draws the domain's probes only; the
    adjoint identity's codomain probe is always white noise there.
  - *Correlated layers*: `from_product(correlation=)` ties factors on one
    latent space only (D-206); a layered correlation waits for sign-
    normalized spectral bases from the planetmodel update.
  - *From the apps' re-measurement (D-207)*: `ambient_ball(method=
    "randomized", rank=dim)` can warn that the kept eigenvalues carry a
    fraction of the trace although every eigenvalue is kept, since it
    compares with a 32-probe estimate whose scatter is about 25% when a
    few modes dominate (6 of 20 seeds on a 64-point circle); skip the
    check at full rank and allow for the estimate's standard error. A
    product measure's `L L*` is two block-diagonal applications not folded
    block by block, so on a layered spectral-element space a covariance
    application costs two transform round trips and a draw two syntheses
    and an analysis instead of one synthesis (256 prior draws 0.24 s
    against 0.61 s; posterior draws 1.12 s against 3.06 s). A process
    `support_sweep` leaves the parent's application counters empty.
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
| D-161 | the spectral-element spaces are `interval`, `interval.radial` and `ball` (and `disk`, D-175) |
| D-163 | every inference class is an estimator; `MinimumNorm` is the discrepancy principle; `BackusGilbert` is the point estimator; MAP returns the mode and `LaplaceApproximation` the Gaussian; `as_measure` is gone |

## 9. Not in 2.0

Deferred by agreement, not by omission (D-112, D-113): `dynamical_system`
and sequential data assimilation, PETSc, `parallel=` inside operator actions,
full function-space MCMC (the hooks exist), and the v2 documentation set.

## 10. Between here and a release

The plan settled with David on 2026-10-24, reviewed issue by issue. Three
inputs informed it beyond the code: the sampling brief
(`docs/sampling_brief.pdf`, the student's proposed function-space MCMC API;
sampling is post-release), the sl3dpp request
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
  The linear Gaussian posterior and `LaplaceApproximation` return a
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
  `SubspaceSlicePlotter`; plotly is the optional `plot3d` extra. Example 18
  shows it.
- DONE 2026-10-27 (D-153 to D-155). The feasible-set layer is rebuilt to
  D-135 to D-152 (§7); its plan was `CONVEX_PLAN.md`, retired 2026-10-28,
  and the benchmarks are deferred by D-155.
- DONE 2026-10-27 (D-156). The nonlinear ray operator on every symmetric
  space, bending or shooting, with its derivative and Hessian action;
  tutorial 13 is its showcase and the tomography mini-app's nonlinear half
  is to be written on it.
- DONE, as far as the release needs: the pin is `planetmodel>=1.3`, the
  stale `[harmonics]` extra is gone, the venv and lock are on 1.3.2 (whose
  grids carry a `kind` that the transform now checks by name), and
  `test_ball.py` no longer needs pyshtools. The planetmodel 1.4 release,
  the pin to `>=1.4` and the deletion of `spectral_element/_transform.py`
  are post-release (below); 2.0 ships the transform of its own.

**Pre-review actions, in order (settled with David 2026-10-29):**

1. DONE (D-195). The known §7 defects that change what a user sees.
2. DONE (D-196). `GaussianMeasure.from_product(correlation=)` and
   `Layered.sobolev_measure(correlation=)`; `correlated_measure` keeps its
   own fused path for the scale-dependent case.
3. DONE (D-198). The low-rank posterior, `LowRankGaussianPosterior`,
   with `LaplaceApproximation(rank=)` and `from_linear_problem`; to be
   shown to the student; examples 25 and 15 show it and the correlated
   product (D-206).
4. DONE (D-199). The planet-density app's unit scheme; David judged
   part D's weaker contrast better as it is.

DONE 2026-10-29 (D-200 to D-207): the review, seven layers by seven
reviewers, five decisions, seven fix packages, 28 Must-fix items fixed,
most Should items fixed, the rest listed above in §7 as left open. Its
working files are under `review/` (findings per layer, `SUMMARY.md` with
the decisions, `briefs/`) until tidying deletes them.
The external asks are out of the review: the student has seen the updates
and is happy, and sl3dpp's list is not a concern now.

**Decided now, built after the release:**

- `formalism=` opens on the nonlinear route with `"data_space"` the default
  (D-74's rule everywhere), and `with_probed_derivative()` mirrors D-34:
  the derivative assembled from `dim(Y)` adjoint applications, exactly
  adjoint to the continuous adjoint that built it. For small data, and not
  privileged: the library offers the range of adjoint strategies — discrete
  adjoints are the backend's business, continuous adjoints are supported
  honestly through measured defects and solver floors, the representer
  route exists where data are few. David's caution (2026-10-29): on a
  large backended problem little or nothing is done datum by datum, the
  adjoint acting on a general data-space vector through one adjoint
  solve, so this is not the picture to design around; the details wait.
- Sampling: the presets (`PCN`, `MALA`, `HMC`, `SMC`) and the result types
  (`SampledPosterior`, `WeightedEnsemble`) are the public contract; whether
  they are built as kernels-and-driver or as discretised SDE dynamics is
  the student's call, made against a working pCN benchmark. Unadjusted
  schemes on function spaces: leaning no.

**Confirmed drops** (v1 things that stay gone): derivative-free
minimization, BiCG, and a separate deflated pointwise-variance estimator.

**Docs phase:** three tiers. The reference examples here stay as they are,
one idea each. The v1 notebooks under `tutorials/` are replaced by
tutorial *scripts* covering the whole core, fourteen as of
2026-10-29 under `pygeoinf2/tutorials/` with notebooks generated from them
by `make_notebooks.py` (Colab entry points; the suite checks they are
current). Mini-apps in the MFEM spirit — each a realistic inverse problem
end to end, deeper than any one feature — are six, under `miniapps/`
(D-176 to D-199). The approved list is `EXAMPLES_PLAN.md`, a working document that is
deleted when the phase completes. `README.md` and the Sphinx tree are
rewritten against the new name, after the rename, not before.

**Release mechanics, in order:** tag `v1-final`; rename `pygeoinf2/` to
`pygeoinf/`; delete v1, `compat.py` and its test, `work/`, and v1's root
`tests/`; CI green on the renamed tree, including the MFEM extra on every
platform it installs on; version 2.0.0; readthedocs flips at the tag with
v1's docs kept under a version selector.

**Post-release tracks:** the planetmodel 1.4 release, then the pin to
`>=1.4` and the deletion of `spectral_element/_transform.py` (the
migration is built and tested: `planetmodel.harmonics.ShellTransform` sits
uncommitted in that repo on its `main`, where its rules want it on
`develop`, same names, the version bump left for its release branch) and
the sign-normalized spectral bases that come with it (§7); the student's
sampling package (the brief's order
of work, starting with the marker-gated methods already done pre-release
and `from_product(correlation=)`); the sl3dpp items (the data-space route
and low-rank posterior above, then blocks with a size hint, process-safety
declarations, the spectral-element separable-basis export, optimizer checkpointing,
resource-owning linearizations, and the tensor product space, which sl3dpp has
offered to supply as a backend); and whatever the convex redesign defers.
