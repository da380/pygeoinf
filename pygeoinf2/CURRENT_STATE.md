# pygeoinf 2.0: what it is now

This is the description of the package for someone who needs to use it or
review it: what it is, its conventions, what is exact and what is estimated,
where it still forms a dense matrix, and what is open. The reasons behind the
choices are in `DECISIONS.md`, cited here as D-nn. The history of how they
were reached is in git, at commit `83658d0` and before, where the design
journal, the review rounds and the functionality audit lived until they were
retired on 2026-09-18.

Last reconciled against the code on 2026-09-18, at 2763 passing tests in the
fast suite and 32 marked slow.

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
| `probability` | `ProbabilityMeasure`, `GaussianMeasure`, `GaussianMixture`, products and push-forwards, conditioning, credible sets |
| `geometry` | convex sets (balls, ellipsoids, half-spaces, polytopes, intersections, oracle sets), level and sublevel sets, affine and linear subspaces, projectors, support functions |
| `numerics` | Krylov and direct solvers, preconditioners, randomised linear algebra, the operator calculus, log-determinants, optimisation and line searches, convex methods (bundle, KKT, Chambolle-Pock), quadratic programming, root finding, weighted chi-square quantiles |
| `symmetric_space` | the sphere on Driscoll-Healy or Gauss-Legendre grids; the periodic box in any dimension with the circle, torus, line and plane as named geometries; the bounded box and interval |
| `sem1d` | spaces in the eigenbasis of `1 - div(L² grad)` on a padded spectral-element mesh, for a length scale that varies: `interval` on the line under `dx`, `radial` for functions of radius under `r² dr`, `ball` for a ball or annulus; Robin ends, a batched harmonic transform, a fit over the domain for sampled functions; wraps `planetmodel.randomfield`, an optional extra, and is imported explicitly (D-115 to D-124) |
| `inference` | forward problems, the estimator hierarchy, `LinearGaussianInversion` and the mixture inversion, the point estimators and the Tikhonov family, `BackusGilbert` and `BackusGilbertParker`, the MAP and Laplace route, normal operators and their preconditioners |
| `plotting` | fields, points, paths, balls and networks on every symmetric geometry; shells, sections by any plane with their stations, and profiles of a `sem1d` ball, lines on its interval and radial profile (D-120); convex sets; marginals, corner plots and error bounds |
| `backends` | MFEM two ways: `mfem` reads its matrices into a `CoordinateSpace`; `mfem_hilbert` keeps a plain `HilbertSpace` and lets MFEM do every computation (D-96) |
| `testing` | thirteen `check_*` functions, from the space axioms to affine operators and convexity (D-105) |
| `datasets` | the shipped station and earthquake tables, the cache directory, and the explicit downloads that refresh them (D-103) |
| `parallel` | `parallel_map` and `resolve_jobs`, the one place joblib is called (D-52) |
| `compat` | v1 class-name aliases and the v1-space adapter; goes with v1 (D-101) |

105 names are re-exported at the top level and the subpackages are importable
as `gi.inference`, `gi.numerics`, `gi.plotting` (D-5). `backends`, `testing`,
`compat` and `sem1d` are imported explicitly. Thirty-two examples under `examples/`
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
- **`power_measure` and `power_spectrum` agree on a Lebesgue space and
  differ on a Sobolev one**, by the Sobolev symbol. Open, see §7.

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

- **`power_measure` on a Sobolev space.** Its eigenvalues are the covariance
  operator's in that space's metric, so a draw's coefficients carry
  `eigenvalue / gram` and the spectrum comes out divided by the Sobolev
  symbol. Whether it should mean the `H^s` spectrum it currently means, or
  the `L2` one a modeller more often writes down, is a decision for the API.
- **The convex solvers' API.** Ported under D-13 and not redesigned; the
  numerics were restored to v1's in September. Mag's view is to be sought
  before the API changes, and the dual route's certified gap (REVIEW2's
  question 2, in git history) is his call too.
- **`route="auto"` on quadratic sets.** The KKT solver is the cheapest general
  solver on balls and ellipsoids and `auto` could prefer it over the bundle
  where both apply; left until measured (D-82).
- **`dense_limit` on `kl_divergence` and `ambient_ball`** keys on dimension,
  not application count (D-50). Fine until a covariance that applies a PDE
  solve meets it.
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

- Point `README.md`, `docs/` and `tutorials/` at the new package.
- Rename `pygeoinf2/` to `pygeoinf/`, delete v1, `compat.py` and its test.
- Confirm the MFEM extra passes in CI on every platform it installs on.
