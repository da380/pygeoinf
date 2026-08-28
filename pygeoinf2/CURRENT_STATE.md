# pygeoinf 2.0 — what it is now

**D-10.** `DESIGN.md` is the journal: it records how each decision was reached,
in the order it was reached, and it is long because that history is the point.
This document is the other thing — what the package *is* today, for someone
who needs to use it or review it without reading five thousand lines of
reasoning.

Last reconciled against the code on 2026-08-28, at 2085 passing tests.

---

## 1. The one idea

Every space is a Hilbert space that knows its own inner product, and nothing
above it ever works in components unless it says so. That single constraint is
what most of the rest follows from:

- A **derivative** is a linear functional; a **gradient** is its Riesz
  representer. They are different objects and the library keeps them apart.
  Example 5 is about nothing else, and it is the mistake this design exists to
  make hard.
- A **Galerkin matrix** `G A_c` and a **component matrix** `A_c` are different
  representations of one operator. Anything that takes a matrix asks which.
- An **adjoint** is taken in the space's metric, so it is not a transpose. On a
  space with a non-diagonal Gram matrix the two differ, which is why the test
  suite runs so much of itself over one.

## 2. The package map

| module | what lives there |
|---|---|
| `algebra` | `HilbertSpace`, `HilbertModule`, `CoordinateSpace`, `LinearOperator` and the operator nodes, `DirectSum`, `DiagonalLinearOperator` |
| `traits` | the trait flags and their closure rules |
| `probability` | `GaussianMeasure`, mixtures, push-forwards, conditioning |
| `geometry` | convex sets, subspaces, projectors, support functions |
| `numerics` | solvers, preconditioners, randomised linear algebra, functional calculus, optimisation, convex methods, quadratic programming, root finding |
| `symmetric_space` | sphere, periodic box, bounded box, circle, torus, line, plane |
| `inference` | forward problems, Gaussian inversion, point estimators, Backus–Gilbert, the Laplace/MAP route, preconditioners |
| `plotting` | field maps, marginals, corner plots, error bounds |
| `backends` | MFEM |
| `testing` | `check_space`, `check_coordinates`, `check_operator`, `check_traits`, `check_white_noise` |
| `compat` | a v1 parity shim, to be deleted at the rename (DESIGN §11.3) |

Ninety-eight names are re-exported at the top level; the submodules are also
importable as `gi.inference`, `gi.numerics`, `gi.plotting`.

## 3. Conventions and units

These are the ones that bite.

- **Points on a sphere are `(latitude, longitude)` in degrees** (**D-2**). Not
  colatitude, not radians. `to_colatitude_radians` and `to_latitude_degrees`
  convert.
- **A sphere's vectors are `pyshtools.SHGrid` objects** (**D-1**), not arrays.
  `grid_values` reaches the numbers; `from_grid_values` goes back.
- **Radii are physical, not angular**, wherever a length is meant. A method
  taking a radius says which in its docstring.
- **Spherical harmonics are orthonormal, without the Condon–Shortley phase.**
  pyshtools spells that `csphase=1`.
- **A box's `degrees` are `floor(|k|)`**, so several wavevectors share one and
  the multiplicities are irregular. Anything indexed by degree must not assume
  the sphere's `2l + 1`.
- **`power_measure` and `power_spectrum` agree on a Lebesgue space and differ
  on a Sobolev one**, by the Sobolev symbol. This is an open API question, not
  a defect — see §7.

## 4. What is exact and what is estimated

Anything returning an `Estimate` is stochastic and carries its standard error.
`random_trace`, `random_diagonal` and `log_determinant` take an `rtol` to
sample *to*, rather than a sample count to guess at.

Three routes answer a Backus–Gilbert support value, selectable by `route=`:

| route | needs | cost (16 directions) | agreement |
|---|---|---|---|
| `dual` | any convex sets | 10.96 s | 1.7e-8 |
| `primal` | any convex sets | 0.066 s | reference |
| `smoothed` | balls or ellipsoids | 0.013 s | 2.4e-9 |
| `kkt` | balls or ellipsoids | 0.008 s | 2.4e-3 |

The KKT route's looser agreement is its own limit: with a tight noise set its
second multiplier saturates. v1 behaves identically.

## 5. Where a dense matrix is still formed

Matrix-free is the default and the exceptions are deliberate. `matrix()` and
`diagonals()` are O(1) reads on a `MatrixLinearOperator` and on diagonal
operators, sums and scalings of them; elsewhere they cost one application per
column.

Dense fallbacks that remain, and why:

- `GaussianMeasure` — the dense route for small spaces (`from_covariance_matrix`,
  `as_multivariate_normal`, the dense log-determinant below `dense_limit=512`).
- `PrimalKKTSolver` — forms two matrices on the **data** space, deliberately:
  that is what keeps the model space undiscretised.
- `ProximalBundleMethod` / `LevelBundleMethod` — the subproblem is dense in the
  number of cuts, which is small by construction.
- `BackusInference` — the joint spectrum, cached per estimator.
- `InvariantDistancePreconditioner` and the localised preconditioners — sparse
  assembly, not dense; `BlockPreconditioner` and
  `ColumnThresholdedPreconditioner` probe columns and never form the full array.

## 6. Writing a backend

A backend supplies a `CoordinateSpace`. The contract is: `dim`, `to_components`,
`from_components`, `apply_gram`, `solve_gram`, `_key`, plus whatever of the
vector algebra (`add`, `subtract`, `scale`, `axpy`, `copy`, `zero`) the vectors
do not get for free.

Then check it, in this order, and stop at the first failure:

```python
check_space(V, rng=rng, rebuild=lambda: build_again())
check_coordinates(V, rng=rng)
check_white_noise(V, rng=rng)
check_operator(A, rng=rng)
check_traits(A, rng=rng)
```

The lessons MFEM taught, which apply to any backend over foreign memory:

- **Who owns the buffer.** MFEM vectors can alias memory the library will free.
  Copy on the way in unless ownership is explicit.
- **Keep the solver alive.** `solver._pygeoinf_keepalive` is load-bearing: the
  wrapped object holds no reference of its own and the C++ side will free it.
- **The mass matrix *is* the Gram matrix.** That is the whole reason a finite
  element space fits without adaptation, and the case the design was built for.
- **Never densify to slice.** Take CSR rows and columns.

## 7. Open questions

Recorded rather than settled, so nobody has to rediscover them.

- **`power_measure` on a Sobolev space.** Its eigenvalues are the covariance
  *operator's* in that space's metric, so a draw's coefficients carry
  `eigenvalue / gram` and the spectrum comes out divided by the Sobolev symbol.
  Whether it should mean the `H^s` spectrum it currently means, or the `L2` one
  a modeller more often writes down, is a decision for the API.
- **The convex solvers' API.** Ported under **D-13** and not redesigned. Mag's
  view is to be sought before anything beyond the port changes, and nothing has
  been cut.
- **`ProximalBundleMethod`'s subproblem accuracy.** Its Gram matrices are
  near-singular by construction, so the accelerated projected gradient has a
  residual floor. `best_available_qp_solver` is now available and would fix it,
  but that changes the behaviour of Mag's method.
- **Iteration-cap naming.** `maxiter`, `max_iterations` and `iterations` all
  appear. One should win.
- **`random_domain_points`** (land/ocean rejection sampling) is not ported.
- **Docstring contract.** `Raises:` where a function raises, and `Args:` for
  parameters carrying a choice, are enforced by `test_code_practice.py` against
  a *shrinking* list: a file at zero cannot regress, and no file may get worse.
  269 gaps remain across 35 files; `algebra/spaces.py` and
  `algebra/direct_sum.py` are clear.

## 8. Decisions

`DESIGN.md` §11 and `REVIEW.md` §11 carry the full list with reasons. The ones
that change what a caller writes:

| | |
|---|---|
| **D-1** | sphere vectors are `SHGrid` |
| **D-2** | points are `(lat, lon)` in degrees |
| **D-4** | `from_matrix(..., form=...)` replaces the two matrix constructors |
| **D-8** | solver defaults are `rtol=1e-8`, `strict=True` |
| **D-11** | point evaluation is refused below the Sobolev order that admits it, with `unsafe=True` to override |
| **D-12** | `path_integral_operator` is the integral; `path_average_operator` normalises |
| **D-13** | the convex solvers are restored |
| **D-7** | nonlinear MAP with a Laplace covariance is in scope, and is `inference.laplace` |

## 9. Not in 2.0

Deferred by agreement, not by omission: `dynamical_system`, sequential data
assimilation, PETSc, `parallel=` inside operator actions (**D-6** puts it
around them), and full MCMC sampling (**D-7** provides the hooks, not the
sampler).
