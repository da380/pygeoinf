# pygeoinf 2.0: the decisions that bind

What the code does is in the code and its docstrings. This file holds the
choices behind it that a reader cannot recover from the code alone: a
convention chosen among alternatives, a default fixed by a measurement, a v1
feature dropped on purpose, a rule the tests enforce. Each entry says what was
decided, why, and what it means for a caller or contributor. Docstrings cite
an entry as `DECISIONS.md D-nn`.

The entries are numbered in one sequence. D-1 to D-13 are the decisions of the
August 2026 review and keep their numbers because the code cites them. The
rest were extracted from the design journal and the functionality audit when
those were retired on 2026-09-18; the journal's reasoning survives in git
history, at commit `83658d0` and before, for anyone who needs the long form.

To add a decision: append an entry with the next number and the date, and
never renumber. To reverse one: edit the entry in place, say what replaced it
and when, and keep the number.

---

## 1. The August review (2026-08-27)

**D-1. Sphere vectors are `pyshtools.SHGrid` objects, not bare arrays.**
Vectors may be general objects; that is a point of the library. Grid
`sampling` defaults to 1 and is a user option. Every method that produces or
consumes a field takes an `SHGrid`; `grid_values` reaches the numbers and
`from_grid_values` goes back. Boxes stay arrays.

**D-2. Sphere points are `(latitude, longitude)` in degrees**, as pyshtools
uses. Every point-taking method on `Sphere` and in `plotting/sphere.py`
converts at its boundary; radius arguments say angular-degrees or physical;
`to_colatitude_radians` and `to_latitude_degrees` convert.

**D-3. Explicit geometries.** The submodules `sphere`, `circle`, `line`,
`torus`, `plane`, `box` each export `Lebesgue` and `Sobolev` as thin
subclasses of `Sphere`, `PeriodicBox` or `Box`, so `isinstance` works and the
class name names the geometry. `with_order` and `with_degree` return the
subclass their order names, not the bare geometry class.

**D-4. One matrix constructor, and it names its form.**
`LinearOperator.from_matrix(domain, codomain, M, *, form=...)` with `form`
required replaces `from_derivative_matrix` and `from_component_matrix`. See
D-25 for the two forms.

**D-5. Both flat and nested exports.** The subpackages are importable from
the top level and the workflow names are re-exported flat, as v1 did.

**D-6. Parallelism around operators, never inside them.** A `parallel_map`
helper on joblib, with `n_jobs=` at every loop whose iterations are
independent and each expensive; nothing inside an operator's action. See
D-52 for the reach.

**D-7. Nonlinear MAP and Laplace inference are in scope for 2.0**; full
function-space MCMC is later, so its seeds are laid now: `log_density`,
`grad_log_density` and the prior samplers are the interface a sampler will
consume. `inference.laplace` is the result.

**D-8. Solver defaults are `rtol=1e-8`, `strict=True`.** A solve that does
not converge raises `ConvergenceError`; `strict=False` downgrades it to a
warning. The same rule applied to an estimate with an error bar is why
`kl_divergence` refuses rather than silently going stochastic (D-63).

**D-9. `heat_measure` is parameterised by a length scale**, not a time:
`exp(-length_scale² λ)`, v1's meaning. The diffusion-time relation is
documented, not encoded.

**D-10. One current-state document, and docstrings cite it.** `DESIGN.md`
was the journal; it has since been retired in favour of this file and
`CURRENT_STATE.md`. Docstrings cite decisions here, never a journal section.

**D-11. The Sobolev-order guard is a hard error with `unsafe=True` to
override.** Point evaluation below the order that admits it is refused. The
guard is on the space, not the operation: `covariance_function` and
`pointwise_variance` converge for a decaying spectrum on any space, but a
statement about point values belongs to a space whose points have values
(David, 2026-08-29: choosing the space is the user's job, and doing it
implicitly through the prior is probably not right). Path integrals have
their own threshold, D-83.

**D-12. `path_integral_operator` is the integral; `path_average_operator`
normalises.** v1's action was right and its name wrong. An optional
`weight=` along the path; non-constant backgrounds by composition; ray
tracing out of scope.

**D-13. The convex solvers come back**, restored to v1's numerics:
`LevelBundleMethod`, the QP protocol and backends, `PrimalKKTSolver`,
Chambolle-Pock, the smoothed dual master, `solve_support_values`. This code is
Mag's; his views are sought before the API changes (the API is recorded as
open in `CURRENT_STATE.md`).

---

## 2. Spaces and vectors

**D-14. The core is coordinate-free; coordinates are a capability** (2026-08).
`HilbertSpace` requires only `dim`, `_key`, `zero()`, `copy`, `inner_product`,
`axpy`, `scale_inplace`; `CoordinateSpace` adds `to_components`,
`from_components` and the Gram map. Methods that need components call
`require_coordinates(...)` at construction and raise `TypeError` otherwise.
`tests/test_coordinate_free.py` runs the algebra and every Krylov solver over
`StrictSpace`, which raises if the component map is touched; only direct
solvers, `matrix()` and the randomised methods that need a basis may.

**D-15. Coordinate-free when it must be, and not otherwise** (2026-08-29,
REVIEW2 question 1). On a `CoordinateSpace` the library may do its internal
arithmetic on `(dim, k)` component arrays (Gram-Schmidt, the Lanczos basis,
the range finder, a low-rank factor's adjoint), because an inner product on a
spectral space transforms both arguments and the coordinate-free forms paid
O(k²) transforms for k vectors (at lmax 64: 2650 analyses to orthonormalise
fifty fields, against fifty on components). The metric enters those paths
only through `apply_gram`, so they are exactly as metric-correct as the inner
product. A space whose coordinate map is only formal opts out with
`uses_component_fast_paths`.

**D-16. Dual spaces are gone; the adjoint is the only map** (2026-08).
No `to_dual`, `from_dual`, `.dual`, `LinearForm`, `DualHilbertSpace`. The
Riesz distinction lives in two readings of one functional: `f.matrix()` is
the derivative, `f.adjoint(1.0)` the gradient. Dropping duals relocates the
mass matrix into the coordinate layer: `inner_product(x, y) = c_xᵀ G c_y`, the
adjoint's component matrix is `G_X⁻¹ A_cᵀ G_Y`, not `A_cᵀ`. What is lost is
holding a functional in load-vector form without a mass solve; `apply_gram`
and `solve_gram` are exposed so that is done deliberately.

**D-17. The Gram map is exposed as component actions, not as an operator**
(2026-08). `CoordinateSpace.apply_gram`, `solve_gram`, `gram_matrix()` and
`has_diagonal_metric` act on component arrays. The specialisations are
`OrthonormalSpace` (identity Gram: `EuclideanSpace`, `Reals`),
`DiagonalMetricSpace` (v1's `OrthogonalHilbertSpace`, its Gram written down as
`diag(metric_values)` and never probed) and the general `CoordinateSpace`.
`ArrayVectorMixin` supplies the vector algebra for array-backed spaces so a
concrete space needs only `dim`, `_key`, `to_components`, `from_components`.

**D-18. The pairing axiom binds every coordinate space** (2026-08).
`<f, x> == f.matrix() · X.to_components(x)` for every functional and vector,
so `representer(derivative_components)` is `solve_gram` on a derivative
array. `testing.check_coordinates` and `check_representer` assert it.

**D-19. `random` and `white_noise` are distinct, and white noise is white in
the space's inner product** (2026-08). `random(rng=)` is an arbitrary test
vector with no covariance claim; `white_noise(rng=)` promises
`E[(x,u)(x,v)] == (u,v)`. On a `CoordinateSpace` the draw is `c = L⁻ᵀ ξ` with
`G = L Lᵀ`, never `randn(dim)`, which v1 drew and which gives covariance `G`
on every mass-weighted space. Sampling a Gaussian through its factor is where
this pays. `testing.check_white_noise` verifies it.

**D-20. Identity comes from `_key()`; spaces are hashable** (2026-08).
`__eq__` is same type and equal key; `__hash__` follows. A space whose
identity depends on an operator supplies a structural key, because operator
equality is object identity. `check_space(rebuild=...)` asserts a rebuilt
space compares and hashes equal.

**D-21. `zero()` is a method, and in-place operations return their result**
(2026-08). `space.zero()` allocates, so it is a call rather than a property
that looks free in loops. `axpy` and `scale_inplace` return the updated
vector, and for an immutable backend (`Reals`, whose vectors are floats) the
result is a new object; callers use the return value. `Reals` stays in the
suite to keep that contract honest.

**D-22. `MassWeightedSpace(base, mass)` picks its variant by the base**
(2026-08, extended 2026-09-17). `(x, y)_V = (M x, y)_base` needs only the base
inner product and `M`; the inverse mass comes from an optional `mass_solver`,
and a diagonal mass inverts exactly. The constructor returns a coordinate
space (Gram `G_base M_c`) over a coordinate base, a module with `multiply` and
`sqrt` delegated over a module, both over both, and leaves a subclass such as
the MFEM space as written. The point is the automated lift (David: having
this step automated is key): `from_formal_adjoint`'s component route works
over it with fused actions kept. `CoordinateSpace.coordinate_selection(indices)`
is v1's `subspace_projection` on any coordinate space, with the metric in its
adjoint.

**D-23. Direct sums hold tuples and are labelled non-structurally; tensor
products of spaces are out** (2026-08). `DirectSum(spaces, *, labels=None)` is
a `CoordinateSpace` exactly when every summand is; its key is the summands
alone. No `X ⊗ Y` space exists because it has no backend-independent
representation; `LinearOperator.from_tensor_product(u, v)` is the rank-one
operator, a construction on operators, not a space.

**D-24. Pointwise multiplication is a capability** (2026-08). `HilbertModule`
sits beside `CoordinateSpace`; `require_module` names it when something needs
it. Nothing in the core assumes fields multiply, so an MFEM space opts in.
`is_element` is gone: with raw backend vectors it cannot tell two spaces
backed by the same array type apart; membership is a test-time question.

---

## 3. Operators and functionals

**D-25. Two matrix representations, and construction must name its form**
(2026-08). `matrix(form=)` returns `"components"` (`A_c`, with
`c_Ax = A_c c_x`) or `"galerkin"` (`G_Y A_c`, symmetric iff self-adjoint);
`form="auto"` picks Galerkin when `SELF_ADJOINT` is claimed, for extraction
only. No trait implies the form on construction, so `from_matrix` requires it
(D-4). `matrix()` means a dense array and nothing else; the scipy bridge is
`as_scipy(form=, n_jobs=)` and `from_scipy(domain, codomain, op, form=)`,
whose `rmatvec` carries the metric factors. `matrix(by="auto")` fills by the
smaller side.

**D-26. The derivative is primitive; the gradient is derived through the
adjoint** (2026-08). For a `Functional`, `derivative(x)` is a
`LinearFunctional` and `gradient(x)` is `derivative.adjoint(1.0)`, the Riesz
representer. The metric enters at the adjoint and nowhere else, so an adjoint
code's array `dJ/dm_i` goes in through
`LinearFunctional.from_derivative_components` and the gradient costs one Gram
solve. Supplying that array as a gradient is the error the design exists to
make hard; `check_gradient` fails it by exactly a factor of `G`. `from_callables`
accepts `gradient=` for a caller who genuinely holds one, and rejects both.

**D-27. Functionals have two named constructors and no bare `components=`**
(2026-08). `from_derivative_components(domain, g)` means `<f, x> = g · c_x`;
`from_representer(domain, v)` means `<f, x> = (v, x)`. v1's
`LinearForm(components=)` did not say which. Functionals are matrix-free by
default; v1 computed all `dim` components in `__init__`.

**D-28. Forms are subsumed, not removed** (2026-08). `Functional[X]` is
`Operator[X, Reals]` and `LinearFunctional[X]` is `LinearOperator[X, Reals]`,
so adjoints, traits, composition and `at()` are written once. `Functional`
carries the convex interface: `subgradient` (defaults to the gradient),
`prox(x, step)`, `conjugate()`.

**D-29. Subclass to define an operator; `from_callables` is the quick path**
(2026-08). The contract is `_value`, optional `_derivative`,
`_second_derivative` and `_linearise`. `__call__` is the value-only path and
must stay cheap. `at(x)` returns a `Linearization` sharing work when one
backend call yields value and derivative. `derivative(x)` calls the
derivative alone (2026-09-17), falling back to the linearisation only for a
class that supplies nothing else; the sum node sums the terms' derivatives and
the composition node linearises the inner operator and takes the outer
derivative alone, so a derivative-only query never pays for a value.

**D-30. Second derivatives are supported, curried and optional** (2026-08).
`second_derivative(x, dx)` returns `F''(x)[dx, ·]` as a `LinearOperator`,
propagates through composition by the differentiated chain rule and exists iff
every factor has one. Hessians propagate only where exact: `(φ ∘ A).hessian`
is `A* H A` for linear `A`; `(φ ∘ F).has_hessian` is false unless `F` has a
second derivative. Gauss-Newton is the named free function
`numerics.optimization.gauss_newton_hessian`, so the choice is visible at the
call site.

**D-31. Expression nodes do safe local simplification only; there is no
simplification engine** (2026-08). `_Identity`, `_Zero`, `_Scaled`, `_Sum`,
`_Composition`, `_Adjoint` compute traits from children, define adjoints
structurally and flatten nested sums and compositions. `adjoint` is memoised
so `A.adjoint.adjoint is A`; anything that joins the algebra and can be
rebuilt (`DirectSum.projection(i)`) must be memoised, because the palindrome
rule (D-36) compares factors by identity.

**D-32. The specialisation protocol asks both operands before a generic node
is built** (2026-08). `__add__` and `__matmul__` try `_combine_add` /
`_combine_radd` (and the compose pair) before `_Sum` / `_Composition`.
`DiagonalLinearOperator` (v1's `InvariantLinearAutomorphism`) and
`OrthogonalProjector.complement` stay in their class this way, independent of
operand order. Measures have the same hooks (D-60).

**D-33. A composition never expands a low-rank product; its diagonal is
O(nk)** (2026-09-16). The composition node declines to report a known matrix
when the product would be larger than its largest factor (a rank-10 factor
times its adjoint at n = 3000 was materialising 72 MB for any caller that
asked); the diagonal of a product comes from the factors as `Σ_j A_ij B_ji`.
Stacked rows in `l2_products_operator` and `(dim, k)` column blocks are kept:
they are the vectors themselves.

**D-34. A derived adjoint is an opt-in, never the default** (2026-09-16).
An operator built from its action alone refuses an adjoint, and the message
names `with_probed_adjoint()`, which assembles the component matrix once
(`dim(X)` applications, shared with `matrix()`) and derives
`G_X⁻¹ A_cᵀ G_Y`. v1 derived one silently at `dim(X)` applications per
adjoint application; a cost that size is what the library exists to make
visible.

**D-35. `from_formal_adjoint` claims nothing; `from_formally_self_adjoint` is
gone** (2026-08). A formally self-adjoint `A` is self-adjoint on the weighted
space only if it commutes with `M` (measured -131989 against +162009 on a
circle Sobolev space). The single constructor takes `traits=` and the caller
owns any claim; the case that does survive (both diagonal in one spectral
basis) is recognised by D-32. `lift_formal_adjoint` on the symmetric spaces
is the thin wrapper using the metric ratio read off the diagonals.

---

## 4. Traits

**D-36. Mathematical properties are traits; representational structure is a
class** (2026-08). `Traits` is `SELF_ADJOINT`, `POSITIVE_SEMIDEFINITE`,
`POSITIVE_DEFINITE`, `INVERTIBLE`, `ISOMETRY`, `UNITARY`, `IDEMPOTENT`; dense,
sparse, diagonal and low-rank stay classes because they carry data. `close()`
adds implied traits to a fixed point. The propagation rules are fixed: a sum
keeps the intersection of self-adjoint and PSD and is PD if one summand is PD
and the other PSD; a negative scale drops definiteness; the adjoint drops
`ISOMETRY`; the inverse keeps PD but not bare PSD. A flattened composition
whose factor list is adjoint-palindromic (`f_i.adjoint is f_{n+1-i}`) is
self-adjoint, and PSD when `n` is even or the middle factor is PSD; that is
how `L L*`, `A C A*` and `A Q A* + R` earn their traits with nothing
asserted. `from_vectors(orthonormal=True)` claims `ISOMETRY` so `U D U*` is
recognised.

**D-37. Traits are claims, verified only in tests** (2026-08).
`self_adjoint(...)`, `with_traits(...)` and `traits=` are assertions the
library cannot check; `testing.check_traits` verifies them numerically and
belongs in test suites. Solver errors on a false claim name it as the remedy.
`DiagonalLinearOperator` claims self-adjointness only when the domain's
metric is diagonal, since `diag(d)` commutes with `G` only then.

---

## 5. Linear solvers and preconditioners

**D-38. Solvers are stateless and declare their preconditions** (2026-08).
`solver(operator)` returns an `InverseOperator`; diagnostics come back from
`solve(y)` as a `SolveResult`, never stored on the solver. Each solver sets
`requires: Traits` (`CGSolver` and `CholeskySolver` need `POSITIVE_DEFINITE`,
`MinResSolver` and `EigenSolver` `SELF_ADJOINT`) and `requires_coordinates`
(true only for `DirectSolver`); CG raises on negative curvature.
`IterativeSolver` defaults to `max_iterations=max(2·dim, 20)`, not `dim`,
which loses MINRES to rounding. Non-square systems are the separate
`LeastSquaresSolver` / `LSQRSolver`.

**D-39. Iterative by default; a direct solver is a keyword away, chosen by
how many times the inverse is applied** (2026-08). Every inversion's `solver`
defaults to `CGSolver`, since a Cholesky default assembles the matrix and says
the library expects small problems. A posterior mean or a damping search wants
CG; a diagnostic that forms covariance blocks wants `with_solver(CholeskySolver())`
and one factorisation (on example 22 the wrong choice took a coupling
diagnostic from 30 s to 525 s). The one built-in direct solve is the
property-space pseudo-inverse in `inference/backus.py`, a handful of rows by
construction.

**D-40. A direct solver factorises on first use and keeps the factors; an
affine translation may be a thunk** (2026-09-16). `DirectSolver._invert`
factorises on the first solve, not at `solver(A)`, and shares the factors with
the adjoint and matrix reads (v1's laziness without v1's per-call
refactorisation). `AffineOperator` accepts `translation` as a zero-argument
callable resolved on first request, which is what lets
`LinearGaussianInversion` be built without a solve (model dim 3000, Cholesky:
construction 0.40 s to 0.001 s; default CG with a nonzero prior mean: 102
applications to one).

**D-41. An iterative inverse solves the adjoint with `P*` from the same
resolved `P`** (2026-09-16). `IterativeSolver._invert` hands its
`InverseOperator` an adjoint solve built lazily from the preconditioner
already resolved: the adjoint of a fixed operator, or the transposed solve
from the same factors for a deferred direct preconditioner, which is why
`DirectSolver` classes declare `transposes`. Before, `A*` was preconditioned
with `P` (thirteen BiCGStab steps where one sufficed) and a deferred
factorisation was done twice.

**D-42. A non-finite residual is a `ConvergenceError` regardless of
`strict`** (2026-09-16). The shared per-step hook refuses it for every
iterative solver, so the check cannot be forgotten by the next one;
`strict=False` downgrades a slow solve, and a poisoned one is wrong, not slow.
CG also refuses `(r, P r) ≤ 0`, naming the preconditioner. A NaN
preconditioner at dim 400 used to run 800 applications to the cap.

**D-43. A solver is fixed at construction; a factory callable closes the
ordering gap; nothing is mutable** (2026-08). `solver=` accepts a
`LinearSolver` or a callable receiving the assembled operator
(`resolve_solver`), so a preconditioner built from other factors can be built
from the operator it will precondition. There is no `set_solver`:
`with_traits`, `with_formalism`, `with_damping`, `with_solver`,
`with_preconditioner` return new objects, because an estimator's handed-out
`covariance` and `kalman_operator` must keep referring to the solve actually
performed. The solver callback receives a `SolveStep` whose `iterate` is
formed on demand and copied (2026-09-17); `SolutionTrackingCallback` keeps
them in `iterates`.

**D-44. Preconditioners are free-standing `LinearSolver`s that read the
normal operator's factors through `FactoredNormalOperator`** (2026-08, scale
added 2026-09-16). `NormalOperator` and `TikhonovNormalOperator` expose
`formalism`, `forward`, `prior_covariance`, `error_covariance` and `scale`,
the factor by which the assembled operator exceeds its Gaussian reading (`t`
on a data-space Tikhonov operator, 1 otherwise); every consumer divides by
`scale`, and a new factor-built preconditioner must too. Structure-aware
preconditioners check for that base with `isinstance`, not a runtime
`Protocol` (which would evaluate `prior_covariance`, which raises at zero
damping); `WoodburyPreconditioner.from_normal` duck-types because `numerics`
must not import `inference` (D-111). v1's four `surrogate_woodbury_*` entry
points collapse into passing cheap factors; `surrogate` returns a normal
operator.

**D-45. Woodbury: the model form needs only applications, the data form needs
inverses, and a damped `Q⁻¹` is the preconditioner's knob** (2026-08, damping
moved 2026-09-16). The model form applies `Q` and `R` only and survives a
Sobolev prior with an unbounded inverse; the data form needs `Q⁻¹` and `R⁻¹`
and takes `prior_damping=` to invert `Q + d I`. That replaced
`GaussianMeasure.with_regularized_inverse`, whose precision was not the
inverse of its covariance, so `log_density` and the normalising constant
described two different Gaussians. A caller wanting a floored measure builds
`N(m, C + d I)` by hand. The decisive test is exactness with exact inner
solves, not an iteration count.

**D-46. `BandedPreconditioner` requires its bandwidth and probes exactly by
default** (2026-08). A tridiagonal preconditioner on a dense operator took a
solve from 125 iterations to failure, and nothing can detect the assumed
structure is absent, so `bandwidth` has no default. `diagonals(probe="banded")`
sums out-of-band entries into the band and is a named option; `"exact"` is
the default. `incomplete=`, `drop_tol=`, `fill_factor=` (2026-09-17) give the
banded and block preconditioners an ILU through one shared factorisation.

**D-47. The distance preconditioner tapers by default** (2026-08). Truncating
`k(d(p_i, p_j))` beyond a radius is not positive (minimum eigenvalues -0.66
to -0.99 at every radius on a degree-48 sphere) and CG breaks on an indefinite
preconditioner, so `taper=True` (Gaspari-Cohn) where v1 defaulted off. An
approximation made column by column of a relational object must be
symmetrised or checked: `ColumnThresholdedPreconditioner` thresholds the
pattern and reads values off the symmetric Galerkin matrix; an inexact inner
Woodbury solve is the caller's to fix with `FlexibleCGSolver`.

**D-48. `LocalisedPreconditioner` drops the error covariance's off-diagonal
and says so; `NormalDiagonalPreconditioner` never applies the assembled
operator** (2026-08). Only `A Q A*` is treated block-wise; `R` contributes
its diagonal, and a test asserts exactness stops when `R` is correlated. The
diagonal of `A Q A* + R` is `<A* v, Q A* v>` per entry or block; on example 24
sixteen blocks beat the exact diagonal at a forty-fourth of the cost. A
diagonal costs `diagonals()`, never `np.diag(matrix())`.

---

## 6. Numerics: calculus, randomised methods, optimisation

**D-49. The operator calculus is gated on traits, dispatched on storage, and
stops on its tolerance** (2026-08, cap fixed 2026-09-16). `operator_function`
requires `SELF_ADJOINT`, evaluates `f` on the eigenvalues of a
`DiagonalLinearOperator` and uses Lanczos otherwise. `max_iterations=None`
means the dimension, where Lanczos terminates exactly, so `rtol=1e-10`
decides: the old cap of 50 never met the tolerance and stopped silently at
1e-3 to 1e-1 error; v1's `rtol=1e-3` delivered 1e-2. Both Lanczos kernels
hold Ritz values to the claimed spectrum (zero for semidefinite,
`eps·λ_max` for definite) and refuse a meaningfully negative one as a false
claim; no claim, no guard.

**D-50. A dense fallback should compare application counts, not dimension**
(2026-09-16). `log_determinant(method="auto")` takes the dense route only
below `dense_limit=4000` and only if the matrix is already known or probing
costs no more applications than the stochastic budget. `kl_divergence` (4000)
and `ambient_ball` (1024) still key on dimension alone; David's caution is
that a covariance applying a PDE solve is not cheap at 1024, and the first
pattern is the one to copy if a limit ever bites. `ambient_ball`'s dense
`eigh` stays because at its limit it is cheaper than 10,000-draw sampling
(0.28 s against 0.60 s) and exact where sampling gives an order statistic.

**D-51. Stochastic estimates return an `Estimate` with a standard error, and
sample to a tolerance** (2026-08). `random_trace`, `random_diagonal`,
`log_determinant` take `rtol=` to sample to rather than a count to guess at;
the dense route reports zero error so callers treat both uniformly. Tests
assert agreement within four sigma of the returned error, never a fixed
tolerance, which would be a statement about the seed. Stochastic Lanczos
quadrature at forty steps on a six-decade spectrum gives -1279 against the
exact -1382: the method's limit, and why the dense route is exact where it
fits.

**D-52. Every independent loop takes `n_jobs=`, serial by default, through
`parallel_map`** (2026-08, reach restored 2026-09-17). David: do not judge
from toy timings what is worth parallelising; the library is for problems
where one serial solve takes minutes. `n_jobs` reaches sampling, `matrix()`
and `diagonals()`, every preconditioner's probes, `sparse_approximation`,
`log_determinant`, `as_multivariate_normal`, the dense point, path and ball
operators, `pointwise_variance_at` and `support_values` on every route. joblib
is required; each worker gets one BLAS thread (8.6 s against 5.5 s for one
dense assembly at four workers with joblib's default); a `parallel_config`
context around the call is respected entirely. NumPy-bound work gains nothing
from processes; transform-bound work gains from processes only, because
pyshtools crashes under two Python threads. The NUFFT's `nthreads` is
threads inside one transform and stays separate. A new probe loop without
`n_jobs=` is a regression.

**D-53. Randomised methods: prior-weighted probes, `power=1`, Gaussian probes,
eigen-truncation** (2026-09-17). `random_range` and every factorisation over
it take `measure=` to draw probes from a measure on the domain: on a forward
operator with a decaying prior the unresolved fraction of the range the data
can see fell from 0.84 to 0.16 at rank 10 (David remembered this from v1).
`power` stays at 1: a second step buys 10 to 15 per cent for double the
applications, and the knob exists. Nyström is dropped: identical to the
eigen-truncation at equal output rank within 2 per cent and dearer to apply.
Rademacher probes are dropped: no variance gain (0.0376 against 0.0386 at
fifty probes), and only Gaussian white noise on the space makes
`random_trace` the trace on a weighted space. `deflated_diagonal` removes
leading eigenpairs before the diagonal estimate (relative error 0.36 to
0.0004 at equal probes on a 0.6-decay spectrum). Adaptive `rtol=` /
`max_samples=` / `block_size=` reach every caller with the fixed defaults
unchanged.

**D-54. Optimisation is written coordinate-free, not wrapped around SciPy**
(2026-08). A gradient and a direction are vectors and the slope is their
inner product, so there is no array to put in the wrong convention (v1 handed
SciPy `G⁻¹ dJ/dc` as `jac` beside a Galerkin `hess`). `SteepestDescent`,
`NonlinearCG` and `LBFGS` default to `StrongWolfeLineSearch` with the
slope-ratio initial step, because Armijo cannot grow a step and crawled by two
orders of magnitude on a badly scaled metric; its zoom interpolates
(2026-09-16: Rosenbrock nonlinear CG 224 to 113 evaluations, cond-1e4
quadratic 4321 to 2013). `truncated_cg` is Steihaug's and treats negative
curvature as a direction.

**D-55. Proximal and subgradient methods use converging step rules** (2026-08).
`SubgradientDescent` offers `sqrt`, `inverse`, `constant`, `polyak` because
v1's constant step did not converge; `ProximalGradient` starts from a
two-point Lipschitz estimate and never increases the step, because a doubling
step ran to 1e12. Proximal operators are written with norms and directions
and so are metric-aware on any space.

**D-56. QP backends in the order Clarabel, OSQP, projected gradient; the
bundle methods keep v1's numerics** (2026-09-16, under D-13). On the level
method's master QP, singular in the level variable, OSQP hit its
10,000-iteration cap on 40 per cent of solves and Clarabel solved every one;
on the proximal dual over a Backus sweep Clarabel took 0.37 s against OSQP's
0.89 s and was a thousand times closer to the primal. `ProximalBundleMethod`
solves its simplex-constrained dual in the number of cuts through that
backend and falls back to the built-in projected gradient, never SciPy's
SLSQP (15 s and less accurate than both). `LevelBundleMethod` keeps the
serious-step test, warm start and box; its master is dense in the data
dimension plus one, v1's arrangement and a limit on the data size.

**D-57. One root-finding kernel, `monotone_root`, with fences** (2026-08,
fences 2026-09-17). The discrepancy principle, the feasibility test and the
Backus primal route all delegate to it. Its contract: bracket at both ends
(widening only upward converges to a wrong answer with a fine-looking
residual); report saturation with the endpoint reached rather than raise,
because a missing root is the answer to a feasibility question; warm-start
consecutive solves (6504 to 4892 inner iterations on a 300-dimensional
system); rebuild a preconditioner only when the multiplier has moved by more
than a set factor. `minimum=` / `maximum=` fences stop the walk and report
the range exhausted; they surface as `atol=` and `minimum_damping=` on the
discrepancy searches, defaulting to zero.

**D-58. `weighted_chi2_cdf`: Imhof by default, saddlepoint by name, equal
weights short-circuited** (2026-08, saddlepoint 2026-09-17). Imhof is exact to
the requested tolerance and degrades worst at equal weights, the commonest
case, which goes to the exact chi-squared. Lugannani-Rice matches v1 to eight
figures at 0.2 to 0.6 ms against Imhof's 1 to 760 ms, but its tail error
follows the shape of the largest weights, not the count (a 512-mode slowly
decaying spectrum, effective degrees of freedom 164, had 2 per cent tail
error at 0.999, no better than eight modes), so v1's rule of switching on
effective degrees of freedom would trade accuracy silently; whoever wants the
speed names `method="saddlepoint"`. Array thresholds are not carried; a
caller maps.

---

## 7. Measures and probability

**D-59. A measure need only be samplable; everything else is optional**
(2026-08). `ProbabilityMeasure` has one abstract method, `sample(*, rng=)`;
`expectation`, `covariance`, `log_density`, `grad_log_density` may be absent.
Derived measures stay samplable: push-forwards, sums and affine images carry a
composed sampler even when no factor exists. `can_sample` is on the base and
wrapped measures delegate, because a default of true is a promise none of
them can make. `grad_log_density` returns a vector, `-P(x - m)` for a
Gaussian, never a functional (D-7). Randomness is an explicit
`numpy.random.Generator`, keyword-only, everywhere.

**D-60. A Gaussian built from a factor earns its traits; one built from a
covariance must claim them; a supplied sampler is centred** (2026-08).
`covariance_factor @ covariance_factor.adjoint` is PSD by the palindrome rule;
a `covariance=` passed directly must carry `SELF_ADJOINT | POSITIVE_SEMIDEFINITE`.
`sample` adds the expectation to whatever the `sample` callable returns, so
every sampler handed to a measure or a `GaussianEstimator` draws the
fluctuation (a randomise-then-optimise draw written the obvious way landed at
twice the posterior mean). Measures have the specialisation hooks
`_combine_affine`, `_combine_add`, `_combine_scale` so a spectral measure
stays spectral under `A @ μ`, `μ + ν`, `a μ`.

**D-61. The one-transform draw lives on the diagonal factor, not on a class**
(2026-09-17). `GaussianMeasure.sample` recognises a `DiagonalLinearOperator`
factor and draws white-noise components scaled by its eigenvalues before one
synthesis, so scaled, summed, translated, marginalised and conditioned
measures keep the fast draw with nothing carried. The invariant measure stays
a factory; its spectrum is `covariance.eigenvalues` (David preferred that to
a named accessor). A diagonal covariance yields its square-root factor for any
non-negative spectrum and a precision only for a positive one.

**D-62. Priors are calibrated by pointwise or norm standard deviation, and
the pointwise variance carries the metric** (2026-08). The invariant measure
constructors take `pointwise_std=` or `norm_std=`, mutually exclusive, because
that is what every example knows. `pointwise_variance` is
`Σ_k s_k φ_k(p)² / g_k`; without the metric it was out by 2.9 on `H²` and
right on every Lebesgue space, and a negative control pins the difference.
Several fields on one domain correlate scale by scale
(`correlated_measure`), their factor the block operator of symmetric square
roots.

**D-63. A trace is the component matrix's; norms are exact and matrix-free;
`kl_divergence` refuses rather than guesses** (2026-08, routes 2026-09-16).
`nuclear_norm` and `hilbert_schmidt_norm` are `tr(C_c)`, not `tr(G C_c)`;
`"auto"` reads a diagonal spectrum, spectral slices of a correlated measure
(recognised by shape, D-111), a stored matrix, or probes `diagonals()` in
linear memory; `"dense"` is opt-in. `kl_divergence(method="auto")` takes
`spectral`, then `dense` below 4000, and refuses above it with a message
naming `kl_divergence_estimate`, which returns an `Estimate`. Reference
covariances in tests are built with `form="galerkin"`, since a symmetric
component matrix is not self-adjoint on a weighted space.

**D-64. Two hardenings of a Gaussian into a set, and neither contains the
other** (2026-08, matrix-free 2026-09-16). `credible_set(level=, solver=)` is
the Mahalanobis ellipsoid; without a precision it builds one as the
covariance's inverse through `resolve_solver`, CG by default (the dense port
inverted `G C_c` and covered 46 per cent of a nominal 90 on a weighted space,
the tell that the construction was wrong), and a measure with a covariance
factor passes it as `Ellipsoid(factor=)`. `ambient_ball(level=, method=)` is
the smallest ball about the mean in the space's norm, its radius a weighted
chi-squared quantile. `weakened_ellipsoid(level=, power=)` interpolates
between them, exact on a diagonal covariance and through the calculus
otherwise; the Cameron-Martin credible set is the credible ellipsoid.
`condition` is the Bayesian update as a measure method and agrees exactly with
`LinearGaussianInversion`. No conversion happens inside a constructor, and
there is no set-to-measure method because it adds an assumption.

**D-65. A sparse approximation of a covariance is an operator, not a
measure** (2026-09-16). `numerics.preconditioners.sparse_approximation` probes
columns matrix-free with v1's correlation criterion and per-column cap and
returns a sparse-backed operator claiming self-adjointness only, since
thresholding does not preserve definiteness; `GaussianMeasure.sparse_covariance`
delegates. v1's regularised sparse-LU precision is not carried; the
thresholded preconditioner is that object.

**D-66. A Gaussian mixture keeps its between-component covariance as a
low-rank factor and its density as a log-sum-exp** (2026-08). The second term
of the law of total covariance is `from_vectors` at rank at most `K - 1`, and
a test asserts it exceeds either component's own variance on the example,
since dropping it turns a mixture into a blur silently. `from_family` takes a
finite support; `from_parameter_samples` is a Monte Carlo discretisation
named as one. The KKT push-forward precision of v1's `affine_mapping` is
dropped: `push_forward` keeps a precision only under the identity and
invertible diagonal maps, because a silent solve inside an algebraic
operation is the hidden cost the library is built not to have.

---

## 8. Sets and subspaces

**D-67. A convex set has three views, and `project` is the metric
projection** (2026-08). `contains`, `project`, `indicator()` (a `Functional`
whose `prox` is `project`) and `support_function()`, so a hard constraint
drops into `ProximalGradient(...).minimize(..., nonsmooth=set.indicator())`.
`project` is idempotent and leaves feasible points alone, so `HalfSpace.project`
differs from v1. `Ellipsoid.project` (2026-09-16) solves through
`resolve_solver`, CG at 1e-12 by default with a first-order predictor as warm
start (a dense Cholesky per Newton step cost 31 s at dimension 1500 with a
probed precision).

**D-68. A set carries what it was given and declares it** (2026-08, flags
2026-09-17). `has_membership`, `has_projection`, `has_support_function`,
`has_maximizer`, `has_level_function`, with a uniform refusal in the base
rather than `NotImplementedError` catching that swallowed real failures.
`from_support_function(domain, oracle, maximizer=, membership=)` builds an
oracle set; `SupportFunction.of_oracle` has a subgradient only with a
maximiser; `Polytope(outer=)` requires the flag because inner and outer
polytopes cannot be intersected with each other. `__add__` is the Minkowski
sum. `of_half_space` is extended-real, parallelism decided on the residual in
the space's norm at 1e-12, `ValueError` in an unbounded direction.

**D-69. `SublevelSet` and `LevelSet` are plain closed subsets; `boundary` is
a property refusing by default** (2026-09-17). Built from any `Functional`
and a level with v1's tolerance `rtol · max(|level|, 1)`; not `ConvexSet`s,
because a projection would be a constrained minimisation. `is_empty`,
`is_bounded`, `closure`, `is_open` and the `open_set` flag are dropped: v1's
`is_empty` returned false with a docstring saying false meant nothing, open
sets differ from closed by the tolerance in floating point. The one real
emptiness question is `FeasiblePropertySet.is_empty`. The convexity check
became `testing.check_convexity`.

**D-70. An affine subspace remembers the equation it was built from or
refuses to invent one; its dimension is the component-matrix trace**
(2026-08). `constraint_operator` and `constraint_value` raise when the
subspace came from a basis; `to_hyperplanes` gives an equation with the same
solutions. `dimension` sums the projector's component diagonal, never the
Galerkin one, which is meaningless on a weighted space.

---

## 9. Inference and estimators

**D-71. Every method is classified by data relation, prior kind and target,
and the prior kind fixes the answer kind** (2026-08). Data are a point, a
measure or a convex set; the prior is absent, a measure or a set; the target
is the model space or a property space through `property_operator`. No prior
gives a point, a measure prior a measure, a set prior a set: `PointEstimator`,
`MeasureEstimator`, `SetEstimator`. The property row is the model row pushed
forward, so `push_forward` is written once per kind and an inverse problem is
an inference problem with the identity as property operator. Mixed cells
(measure data with a set prior and the reverse) are out of scope, as BGP's
Table 1 says.

**D-72. The estimator is the mapping; the forward problem holds only the
observation model** (2026-08). `LinearForwardProblem(A, error=)` carries the
operator and the data uncertainty, a Gaussian measure or a convex set; the
prior and `property_operator` are arguments to the estimator, because the
prior is what selects the method. `LinearPointEstimator` is an
`AffineOperator` exposing `.operator` and `.resolution` (`X A`, the averaging
kernel), because resolution is the output of a Backus-Gilbert method, not a
diagnostic bolted on. `consistency_set(model, level=)` is the hardening of
the error at a level, with `chi_squared` and `critical_chi_squared` as
conveniences; `harden_error` is where a Gaussian problem and a set problem
meet. `parameterized` and `data_reduced` live on the problem and are lifted
onto the estimators.

**D-73. A linear Gaussian estimator is a pair, and its sampler is carried by
the estimator** (2026-08). `GaussianEstimator` holds `expectation_operator`
(data to posterior expectation) and a data-independent `covariance`, so
`push_forward(T)` is `T ∘ expectation_operator` and `T C T*`. The
randomise-then-optimise sampler is handed to the estimator as a centred draw
of the fluctuation, not attached on the way out of `__call__`, because a
sampler that only exists on the way out is one `push_forward` drops.
`LinearGaussianInversion` checks its inputs at the door with `isinstance` and
names the set-valued alternative. `LinearGaussianMixtureInversion` is a
`MeasureEstimator`, not a `GaussianEstimator`, because its weights depend on
the data.

**D-74. The formalism defaults to `"data_space"`; `"auto"` is opt-in**
(2026-08). Model spaces are usually larger, the model-space route needs
`Q⁻¹` which a function-space prior often lacks, and comparing dimensions would
let a grid refinement silently change which algebra runs. `choose_formalism`
implements `"auto"` for callers who ask. The two formalisms are tested to
assemble the same operator to machine precision. A low-rank prior has a
factor and no precision, so it works only in the data-space formalism, and
the check says so.

**D-75. Tikhonov is a family, kept as its own class** (2026-08). It is
exactly the Gaussian case with `Q⁻¹ = t I`, and a test asserts that; but
`TikhonovFamily` exists to be walked along (discrepancy search, L-curve) with
warm starts, and reading a damping as a prior variance is a claim the caller
need not make.

**D-76. `DiscrepancyPrinciple` is a nonlinear `Operator` with an exact
derivative, and the two ends of a saturated search mean different things**
(2026-08). Two data vectors needing different dampings are related by no
fixed matrix, so it is not a `LinearPointEstimator`; its derivative is the
fixed-damping estimator plus a rank-one correction for the damping moving,
omitted when the damping is pinned by the range. When every damping fits,
`MinimumNorm.for_data` returns the largest damping and the smallest model,
the correct discrepancy answer (the first version returned the smallest,
nine orders wrong); when none reaches the target, `DiscrepancyPrinciple`
raises naming the misfit reached, because the least-damped model solves a
singular system and is not a fallback.

**D-77. An exact constraint is a subspace substitution using the reduced
method** (2026-08). `ConstrainedLeastSquares` and `ConstrainedMinimumNorm`
write the subspace as `t + range(P)` and solve for `A P` and `d - A t`, so the
estimator stays affine in the data. `constraint_value_mapping` adds a
subspace point to the reduced solution, never the unconstrained one.

**D-78. Evidence is two terms, matrix-free, from the component determinant**
(2026-08). `evidence_terms` returns the Mahalanobis misfit and the
log-determinant separately; `log_evidence` sums them. The misfit is one solve
through the estimator's own solver; `normal_log_determinant` delegates to
`log_determinant`, which in the model-space formalism uses Sylvester's
identity so the data-space operator is never formed. The determinant is the
component matrix's (`det(G A_c) = det G · det A_c`): the dense route subtracts
`log det G`, the stochastic route needs no correction because white noise on
the space already probes `tr A_c`. `mahalanobis_squared` raises without a
precision and gets no dense fallback.

**D-79. One estimator for the feasible property set; the sets decide the
algorithm; `route=` forces one** (2026-09-17). `BackusGilbertParker(forward_problem,
property_operator, prior, confidence_set=)` returns a `FeasiblePropertySet`.
A ball prior with exact data takes the closed form (Al-Attar 2021, eq. 2.84);
two quadratic sets (balls or ellipsoids, centred anywhere) take the primal
bisection in the sets' own inner products, nothing whitened (David: just
allow more general sets and functionals); anything convex takes the dual
(BGP eq. 28); `route=` forces one for cross-checks and is refused, naming the
alternative, where the sets do not allow it. Where two routes apply they must
agree, and the parity tests are the point of the layer. `BackusInference`,
`FeasibleProperty` and `DualFeasibleProperty` are private engines and are not
aliased in `compat`; `BackusGilbert`, the point estimate, is a different
object with a two-part error bar (resolution and noise). The names credit
Backus and Parker; "SOLA" is not used.

**D-80. The answer is what gets probed, and it carries two
characterisations** (2026-09-17). Every question (`support`, `contains`,
`extent`, `inner_hull`, `fitting_model`, `extremal_model`, `is_empty`,
`push_forward`) is asked of the `FeasiblePropertySet` with no data argument,
matching the Bayesian side; nothing is computed at construction. The support
function bounds the set from outside (the certified `Polytope`); the level
function decides points and gives inner bounds; not both are required of
every route, and reporting the inner hull alone is BGP's Figure 4 mistake.
`membership=` chooses the membership engine as `route=` chooses the support
engine; a set with a level function and no support function has
`route is None`. The `level_function` / `level` contract: the prior's level
function at the smallest fitting model, against the prior's own level, so a
ball's level is its squared radius and an ellipsoid's is one;
`inclusion_norm` is the quadratic convenience.

**D-81. Feasibility is decided in the data space, matrix-free** (2026-09-16).
David: data spaces run to millions of dimensions and are small only relative
to model spaces, so forming `A A*` is never a safe default. `is_feasible` is a
damped minimum-norm root search through `misfit_search`, one warm-started
Krylov solve per probe; the discrepancy principle uses the same search with
the chi-square misfit. An unbounded dual is an empty feasible set, raised,
not a support value; bundle cuts store the point they were taken at.

**D-82. A Gaussian error is hardened to the ambient ball by default; general
level functions go through the likelihood and KKT engines** (2026-09-17).
The ball admits the cheap routes on both sides, so nothing gets slower
silently; the credible ellipsoid is asked for by passing it as the confidence
set. `LevelKKTSolver` handles differentiable non-quadratic sets on either side
with a cold-started convex minimisation per probe, two multipliers in log
coordinates (David's choice over a joint Newton). Both KKT solvers accept a
root only if the equations hold to 1e-6 of their levels, after `fsolve`
reported success 2 per cent low. Recorded as open: `"kkt"` is the cheapest
general solver on balls and ellipsoids and `auto` could prefer it over the
bundle, left until measured. The finite-difference dual gradient is dropped:
on a nonsmooth cost, one evaluation per data dimension, silently, it is not a
subgradient; a maximiser is asked for by name.

---

## 10. Symmetric spaces

**D-83. The package is `symmetric_space`; "invariant" is reserved for
operators** (2026-08). Circle, torus, periodic box and sphere are homogeneous
spaces; intervals and boxes are built by embedding into them. A
`DiagonalLinearOperator` on such a space is invariant under the group action;
`spaces` is left free for meshes. The path methods need Sobolev order above
`(d - 1)/2` (2026-09-16: the representer norm on a sphere diverges at orders
0 and 0.25, logarithmically at 0.5, converges at 0.75; v1's `d/2` was too
strict by half); ball averages need no order.

**D-84. Four v1 spaces are one N-dimensional `PeriodicBox` via `rfftn`, with
physical coordinates and unit circles and tori by default** (2026-08, defaults
2026-09-16). The `Lebesgue` basis is orthonormal (Parseval is the pinned
invariant; v1's factor-of-two bookkeeping is gone). A box's `degrees` are
`floor(|k|)`, so multiplicities are irregular. Coordinates are physical
lengths; the circle and torus default to period `2π` with `radius=` /
`radii=` beside `length=` / `lengths=`, so on a unit circle the coordinate is
the angle and v1's numbers hold. `degree_multiplicity` of one at the Nyquist
degree is right (Dan's v1 fix branch `af7f568`). `Box` and `Interval`
subclass `PeriodicBox`; `support_projection` on a Sobolev space is built on
the order-zero counterpart and lifted, claiming no symmetry.

**D-85. A Sobolev space is a diagonal-metric space, not a mass-weighted one**
(2026-08). Same coordinate map as `Lebesgue` with metric
`(1 + L²|k|²)^order`, so every `SymmetricSpace` is `HilbertModule` plus
`DiagonalMetricSpace`. Operators derived on L2 are reused through
`lift_formal_adjoint`. A multiplication operator is self-adjoint on L2 and
lifted, claiming nothing, on a Sobolev space. `spectral_operator(values)`
takes an array indexed by `degrees`, not a callable, because a symbol need
not be a function of the Laplacian.

**D-86. Sphere conventions** (2026-08; grid 2026-09-17). Orthonormal harmonics
without the Condon-Shortley phase (`csphase=1`); the radius is scaled into
the basis so the basis is orthonormal on that sphere; angles in degrees,
distances in units of the radius; `random_point` uniform in `cos(colatitude)`.
Great-circle distance is `atan2(|u × v|, u · v)`, never `arccos`, which loses
half its digits near zero and dropped points from their own neighbourhood.
The grid is a strategy: `grid="DH"` (Driscoll-Healy, `sampling` in the key)
or `"GLQ"` (Gauss-Legendre, `lmax + 1` rows at the Legendre zeros, refuses
`sampling` rather than ignoring it); a GLQ space borrows the fast point
evaluation from a Driscoll-Healy sibling of the same truncation; spaces on
different grids compare unequal. `extend=` adds the wrap column and pole row
for plotting and interchange, off by default, the extras carrying no weight.

**D-87. A product is left on the grid, untruncated; `truncate` is public**
(2026-08-30, David: the user picks a discretisation suited to the problem).
`multiply` and `sqrt` return the grid array: a product of band-limited
functions is not band-limited, truncation replaced the exact product with a
projection, and it cost two transforms on the hot path (4.72 ms against
0.018 ms at lmax 128). Every consumer that leaves the grid analyses, so
components are unchanged and self-adjointness holds under the grid's positive
quadrature weights. On the sphere's `Lebesgue` space the inner product is the
grid quadrature (5.44 ms to 0.055 ms); `Sobolev` keeps the component route.

**D-88. No operator has a `matrix_free` flag and no observation operator
writes down an adjoint** (2026-08). They are built through
`from_derivative_callables(domain, codomain, value, derivative_components)`,
the matrix-free counterpart of `from_matrix(form="galerkin")`, so the
framework applies `representer` once. The dense route is `assembled()` or
`dense=True`, filled from `basis_matrix` (60 paths at lmax 64: 153 ms to
build, then 1.6 ms against 66 ms per application). `path_average_operator` is
`W E`, point evaluation at pooled nodes then a sparse weight matrix;
`lazy_quadrature` is dropped (David: fine), nothing reaching the scale where
the node set does not fit before the applications are the problem.

**D-89. Scattered evaluation goes through a non-uniform FFT where one
exists** (2026-08). On a box `evaluate` is a finufft type-2 transform and
`accumulate` the type-1 (655× the direct sum at 256² with 3000 points), the
spectrum widened by two along each axis because folding the Nyquist mode
flips its phase off-grid. The sphere uses the double Fourier sphere extension
(exact to 2.5e-13; 35.6 s to 27 ms for 100 000 points at lmax 128).

**D-90. `gradient_dot_product` takes the positive-Laplacian sign, and the
flexure operator uses the Bochner identity** (2026-08). v1 computed the
negative, invisible with constant coefficients (6e-9 against 36 per cent on a
beam with varying `D`). `flexural_operator` forms
`tr(Hess D Hess w) + 2K ∇D · ∇w` from three `gradient_dot_product` calls with
no Hessian or tangent frame, the curvature term pinned by the degree-one
closed form `tr(Hess f Hess g) = 2 f g`.

**D-91. The spectral packing is public as arrays; no scalar wrappers; padding
is a composition** (2026-09-17). `orders` (signed) beside `degrees` on the
sphere, `wavevectors` and `phases` on the boxes, `component_of(label)`
vectorised; the reverse map is indexing. `coefficient_operator(components=)`
and its converse pick named coefficients, the property operator David named
as the main use. A band past the truncation is the operator on a space
resolved to that degree composed with the prolongation. `sufficient_degree`
walks past the space over the modes a degree would have, with `with_degree`
on the answer replacing v1's factories.

---

## 11. Plotting

**D-92. Plotting is a layer over the geometry, dispatching on the space**
(2026-08). Nothing in `symmetric_space` imports matplotlib or cartopy;
`plot`, `plot_points`, `plot_paths`, `plot_balls` are `singledispatch`
generics. The sphere's defaults are v1's (2026-08-29: `RdBu`, no colorbar,
gridlines, PlateCarree); pyslfp's Robinson default is its own business.
`plot_network(space, paths, sources=, receivers=)` is v1's styling on every
geometry, and one path is a network of one. `show()` guards `plt.show()` so
the suite stays warning-free.

**D-93. Exact Gaussians are drawn, everything else is sampled, with the
component covariance** (2026-08). `plot_densities` and `plot_corner` use
`as_multivariate_normal` (`G⁻¹ C_gal G⁻¹`, 75 per cent off on the weighted
test space without the corrections) for a Gaussian with a covariance, and
histograms with kernel density for anything with `can_sample`.

**D-94. `plot_set` is the compact port** (2026-09-17, David: Mag can expand
it later). One and two dimensions, matplotlib only, the route by capability
(support polygon, level contour, membership raster), slices through
`subspace=`, the support asked in the domain direction `G⁻¹ n`. No 3-D,
voxels, plotly or `Subset.plot` method.

---

## 12. Backends

**D-95. MFEM's mass matrix is the Gram matrix; its forms are Galerkin
matrices and derivatives** (2026-08). An assembled bilinear form goes in as
`from_matrix(form="galerkin")`, a linear form is a derivative whose
representer is `M⁻¹ b`, the mass solve is `solve_gram`. `to_components`
copies, because `GetDataArray` is a view that does not keep its owner alive.
A Dirichlet condition is a subspace: `MfemSpace(elements, essential_dofs=)`
takes the free-free block as Gram and vectors keep MFEM's full length with
zeros on constrained dofs.

**D-96. MFEM solves MFEM's systems; this library composes the results**
(2026-08). `solver_from_bilinear_form` wraps MFEM's own solve (components,
mass multiply, solve, components back; omitting the mass multiply is wrong by
a mass matrix); `operator_from_bilinear_form` densifies and is not the route
for a real problem. In the coordinate backend `FormSystemMatrix` is not used
because it takes ownership and a later `SpMat()` is a use-after-free;
`_default_mfem_solver` keeps explicit references because `SetPreconditioner`
stores a raw pointer. `MfemHilbertSpace` (`mfem_hilbert`) is the second,
matrix-free arrangement: a `MassWeightedSpace` over a bare dof space with
MFEM's `Mult` and CG, forms through `FormSystemMatrix` with operator, handle
and form retained together, `matrix()` refused; this is also the MPI path,
untested.

**D-97. Matérn fields come from MFEM's SPDE miniapp, integer exponents only**
(2026-08). `matern_measure` gives factor `η S^a`, covariance `η² A^{-2a}`,
precision `A^{2a}/η²`, nothing formed; a non-integer exponent is refused
listing the working `ν`; stationarity fails within a correlation length of
the boundary. Partial assembly of a matrix coefficient is broken in PyMFEM
4.8, so `mfem_hilbert.matern_measure` uses a scalar coefficient when the
field is isotropic and refuses otherwise.

**D-98. Foreign backends are optional extras; PETSc is not a dependency**
(2026-08). `mfem` and `sphere` are extras whose tests and examples skip when
absent. A PETSc adapter was written, proved the point (weighted adjoint
`M⁻¹ Aᵀ M` confirmed), and was withdrawn because the PyPI package builds a
private PETSc and MPI into the venv; it returns only as `petsc4py` against an
existing installation.

---

## 13. Naming, packaging, data

**D-99. v1 names unless there is a strong reason** (2026-09-17). Reverted to
v1: `forward_problem`, `property_operator`, `property_space` (`target`
collided with the target misfit), `kalman_operator` (still a property).
Kept from v2 with a reason: `LinearGaussianInversion` (the mixture is Bayesian
too), `prior` / `data_prior` / `joint_prior` and `error=` (the slot holds a
measure or a set), `precision`, `can_sample`, `affine_map` / `push_forward` /
`translate`, `from_product`, `credible_set(level=)` (0.95 is a confidence
level, not a significance level), callable estimators. Neither:
`expectation_operator` (David: expectation over mean). `max_iterations` is
the one name for a budget going in; `iterations` is only the count on a
result.

**D-100. American spelling for every identifier** (2026-09-17, David: we are
using other libraries that are American and cannot get away from that).
`Optimizer`, `minimize`, `Linearization`, `center=`, `has_maximizer`,
`randomized.py`, `optimization.py`; docstrings agree with the identifiers.
Prose documents stay as written. `centre=` or `minimise` in code is a second
spelling.

**D-101. `compat` carries class aliases only; a renamed keyword raises**
(2026-08, 2026-09-17). The five v1 inversion class names are aliased; a
renamed keyword raises `TypeError` at the call, the loud, easy failure a shim
would hide. The v1 adapter in the same module is scaffolding: nothing in the
package imports it, and it goes with its test when v1 is removed. It
deliberately does not delegate `white_noise` to v1 (D-19).

**D-102. All imports inside the package are relative** (2026-08), so the
rename of `pygeoinf2/` to `pygeoinf/` touches no import; only `tests/` and
`examples/` name the package.

**D-103. Datasets never download on their own; the cache is resolved at each
call** (2026-09-17). `download_gsn_stations` and `download_usgs_earthquakes`
write into `cache_directory()` (`PYGEOINF_CACHE_DIR`, else the platform
cache), resolved per call so a test can redirect it; `read_table` takes the
cached copy before the bundled one. `earthquakes(count=)` above the table's
size is refused naming the download: a fetch the caller did not ask for needs
a network and changes the next call's answer.

---

## 14. Testing and code practice

**D-104. Three house rules, enforced by `test_code_practice.py`** (2026-08).
Every public class and function has a docstring; every parameter and return
is annotated, private methods included; every optional argument is
keyword-only, because an optional positional argument freezes its position.
`Raises:` wherever a function raises and `Args:` for parameters carrying a
choice (278 gaps taken to zero). `tests/` and `examples/` are excluded; the
examples' test is that they run.

**D-105. Axiom checks live in `pygeoinf2.testing`, not the production MRO**
(2026-08). The thirteen `check_*` functions replace v1's mixins; every one but
`check_measure` takes `measure=` (2026-09-17, David: white noise is
unrealistic on a function space) through one helper that validates the
domain. `check_affine` covers translation recovery, affine combinations and
the linear part.

**D-106. Every metric-sensitive test needs a non-diagonal Gram matrix**
(2026-08). Five bugs of one kind (the credible-set inverse, a hand-symmetrised
eigendecomposition, the preconditioner diagonals, the KL reference, the
`as_multivariate_normal` broadcast) were each right on a diagonal metric and
wrong otherwise. The fixture `make_dense_metric_space` exists for this; a
sampling check is corroboration, not the test.

**D-107. Two independent routes to one number is the test pattern; a claim
of exactness is tested as exact** (2026-08). Formalism parity, the Backus
routes against each other, the inclusion test against the closed form,
`condition` against the inversion, mixtures against plain numpy. For a
preconditioner the admissible statements are an identity that must hold
exactly, two routes to one number, or an answer that must not depend on it;
an iteration count alone proves nothing. Negative controls accompany every
protected convention: a metric-free adjoint must fail `check_operator`, the
naive pointwise variance must differ, the banded probe must differ on a full
operator.

**D-108. The coordinate-free claim is tested with adversarial doubles**
(2026-08). `tests/doubles.py` has `Opaque` vectors with no arithmetic,
`OpaqueSpace` with a non-dot-product inner product, and `StrictSpace`;
`OpaqueSpace`, not `StrictSpace`, is used for randomised methods because
`white_noise` on a coordinate space is legitimately a coordinate operation.

**D-109. Expensive tests carry `slow` and the default run excludes them**
(2026-08). `addopts = "-m 'not slow'"`; `-m slow` runs only those, `-m ""`
everything. The marker is applied at parametrisation. Tests whose cost is the
point are not shrunk.

**D-110. No interactive figures in the suite or the examples** (2026-08-29).
Plotting and examples run with `MPLBACKEND=Agg`; `plotting.show()` is the
guard.

---

## 15. Layering

**D-111. `numerics`, `algebra` and `geometry` never import `inference` or
`probability`** (2026-08). The dependency runs one way, which is why
`WoodburyPreconditioner.from_normal` duck-types and the correlated-measure
norm routes recognise spectral slices by shape. The top-level `__init__`
exports `inference`, `numerics`, `plotting` and the flat workflow names
(D-5); `compat`, `backends` and `testing` are not pulled in.

---

## 16. Deferred, by agreement

**D-112. `dynamical_system` waits for the sequential-assimilation session**
(2026-09-17, David). v1's four classes were interface only, and an interface
with no consumer gets designed twice. That session starts from: a rule
`F(t, ·)` is an `Operator` with `at()`, a linear rule a `LinearOperator`, and
a Kalman step is the Gaussian inversion's push-forward and conditioning
iterated in time.

**D-113. Also later:** full function-space MCMC (D-7 lays the hooks), PETSc
(D-98), parallelism inside operator actions (D-6), the convex solvers' API
(D-13, with Mag), `DualFeasibleProperty`'s certified gap (REVIEW2 question 2,
Mag's call), and pointing Sphinx at the new package (REVIEW2 question 11).

---

## Appendix: measurements behind the defaults

Kept because they would be costly to redo. Timings are from a throttling
laptop and should be re-measured before being quoted.

- Operator calculus (D-49): `(dim, 1e-10)` costs 69 applications at dim 200
  cond 1e2 (sqrt), 149 at cond 1e4, 183 at cond 1e6 (log), 677 at dim 1000
  cond 1e6; `rtol=1e-8` delivered 1e-6 in the answer.
- Path-integral representer norm on a sphere of radius 2, lmax 16 to 256
  (D-83): order 0, 2.51 to 9.99; 0.25, 2.33 to 5.88; 0.5, 2.18 to 3.88; 0.75,
  2.05 to 2.89; 1.5, 1.75 to 1.90.
- Strong Wolfe zoom, evaluations to `gtol=1e-8`, bisection to interpolation
  (D-54): Rosenbrock nonlinear CG 224 to 113, L-BFGS 74 to 62; 60-D quadratic
  cond 1e2 CG 426 to 201; cond 1e4 CG 4321 to 2013, L-BFGS 907 to 870.
- Proximal bundle dual over sixteen Backus directions (D-56): projected
  gradient 10.84 s, deviation 1.7e-8; OSQP 0.89 s, 6.5e-8; Clarabel 0.37 s,
  6.5e-11; SciPy SLSQP 15.09 s, 4.6e-7. OSQP capped on 15 of 58 level-method
  masters.
- `ambient_ball` dense `eigh` against 10,000-draw sampling (D-50): dim 256,
  0.52 s against 0.16 s; dim 1024, 0.28 s against 0.60 s.
- Lazy factorisation (D-40): model dim 3000, data dim 1500, Cholesky:
  construction 0.40 s to 0.001 s, first call 0.40 s, second 0.009 s.
- `Ellipsoid.project` at dimension 1500 (D-67): 3.5 s with a matrix-backed
  precision, 31 s and 54,000 applications with a probed one, per dense
  Cholesky.
- Saddlepoint against Imhof (D-58): 0.2 to 0.6 ms against 1 to 760 ms; tail
  error a few parts in a thousand to a few in a hundred, following the shape
  of the largest weights.
- `random_range` probes with a `1/k²` prior (D-53): unresolved visible
  fraction at rank 10, 0.84 white noise against 0.16 prior-weighted; at rank
  20, 0.73 against 0.094.
- Dense assembly of 625 covariance columns on a 16-core laptop (D-52):
  serial with default BLAS threads 22.0 s; BLAS limited to one thread 13.9 s;
  loky four workers with joblib's default inner threads 8.6 s; four workers
  with one thread each 5.5 s; eight workers 5.3 s; any threading backend on
  the sphere crashes. Sampling sixteen posterior draws of 25 ms each: serial
  0.43 s, loky four workers warm 1.2 s (the process boundary costs more than
  the draw); eight draws of 0.5 s each: serial 3.9 s, loky four warm 2.2 s.
  A NumPy-bound loop with sixteen BLAS threads: 0.33 s serial against 0.36 s
  at eight threads and 1.03 s at eight processes.
- Component-space fast paths (D-15): at lmax 64, 2650 analyses to
  orthonormalise fifty fields coordinate-free, fifty on components.
- Point evaluation on the sphere (D-89): direct Legendre sum 35.6 s, double
  Fourier sphere NUFFT 27 ms, 100 000 points at lmax 128; on a 512² torus at
  1e5 points, the transform route against the direct sum, 2000 tomographic
  paths 1.59 s to 0.018 s through pooled nodes.
