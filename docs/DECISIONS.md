# pygeoinf 2.0: the decisions that bind

What the code does is in the code and its docstrings. This file holds the
choices behind it that a reader cannot recover from the code alone: a
convention chosen among alternatives, a default fixed by a measurement, a v1
feature dropped on purpose, a rule the tests enforce. Each entry says what was
decided, why, and what it means for a caller or contributor. The code does not
cite these entries (D-10): its docstrings and comments are self-contained, and
the reasoning behind them lives here.

The entries are numbered in one sequence. D-1 to D-13 are the decisions of the
August 2026 review and keep their numbers. The rest were extracted from the design journal and the functionality audit when
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

**D-10. Two documents of record, and the code cites neither** (2026-08;
reversed 2026-10-26, David). `DESIGN.md` was the journal; it has since been
retired in favour of this file and `CURRENT_STATE.md`. Docstrings used to cite
decisions here as `DECISIONS.md D-nn`; since 2026-10-26 nothing in the code
does -- not a docstring, a comment, an error message, a test or an example.
A docstring says what is done, what its terms mean and, where a caller needs
it to choose well, briefly why; the longer reasoning, the alternatives and the
measurements belong here. A docstring that needs a decision to make sense is
incomplete, and the fix is to say the point in the docstring. The same holds
for planning, development history and comparisons with v1, which belong in
these documents, a migration guide or git history.

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

**D-48. `LocalizedPreconditioner` drops the error covariance's off-diagonal
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
slope-ratio initial step, because Armijo cannot grow a step and costs about twice the
evaluations on a badly scaled metric (measured on a quadratic with a
metric spread of 1e3: Wolfe 3376 against Armijo 6744, and 417 against
743 at a spread of 1e2, same iterations and answer; an earlier draft of
this entry said two orders of magnitude, which the measurement does not
support); its zoom interpolates
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

**D-129. An evaluation failure is a step too far, not a crash** (2026-10-24,
from the sl3dpp request list). `EvaluationError` in `algebra.operators` is
the protocol for a forward that cannot converge at a trial model: raise it
from the value or the linearization, and the point is treated as bad rather
than the run as broken. The Armijo search treats it, and any non-finite
value, as a failed decrease and backtracks; the strong-Wolfe search treats
the trial as infinitely bad, closing its bracket on it from the high side,
which the sectioning already handles (interpolation through an infinite
value falls back to bisection); a trust region shrinks its radius, which is
its ordinary response to a model that was wrong. The one place with nowhere
to retreat — the derivative failing at a point the search just accepted by
value — returns the optimizer's result with the new `failure` field carrying
the error's message and `minimizer` still the last good point, so a run of
hours is handed back rather than lost. A failure at the *starting* point
propagates: there is no state worth saving and the caller must move. Only
`EvaluationError` gets this treatment; any other exception is a bug and
still propagates. In passing, `OptimisationResult` became
`OptimizationResult` — the one identifier that had escaped D-100.

---

## 7. Measures and probability

**D-132. An inference result carries the reference measure its density is
defined against** (2026-10-24, David, from the sampling brief, answering
D-73's worry). `AbsolutelyContinuousMeasure` is the dimension-robust density:
a `reference` measure, a `relative_log_density` (a `Functional`, so it
carries its derivative), and a lazy `log_normalizer`, under the convention
`dmu/dnu == exp(f - c)` with `c == log int exp(f) dnu`. It is built from a
reference and a density, never from two arbitrary measures, and
`relative_to` is defined only through a shared reference. On its own it is a
target and refuses to sample — drawing from a density is a sampler's job.
`GaussianPosterior(GaussianMeasure, AbsolutelyContinuousMeasure)` is what a
Bayesian update returns: `LinearGaussianInversion.__call__` hands back the
same Gaussian as before, now carrying the prior, `-Phi`, and the evidence in
the potential's convention, `(log|R| - log|S| - <v, S^-1 v>)/2`, lazily (two
log-determinants, the solve shared with `mahalanobis`); it relates to
`log_evidence` by the likelihood's constant, which the potential omits.
`MaximumAPosteriori.at` returns one whose two readings deliberately part
company: Gaussian moments are the approximation at the point, the relative
density is the *exact* posterior's, computed in the data space — the pair a
sampler starts from; its normalizer is `None` (a Laplace evidence estimate
is a separate claim, left open). D-73 is answered rather than overridden:
the algebra rebuilds through `GaussianMeasure._rebuild` and so degrades to a
plain Gaussian — a density does not survive a general linear map, and a
property-space push-forward of the *estimator* likewise returns plain
measures. Two deliberate namings: the member is `relative_log_density`, not
the brief's bare `log_density`, because that name already means the gated
volume density (D-131) and the subclass holds both; and
`ForwardProblem.potential(data, temperature=)` is the brief's likelihood
potential `Phi = chi^2 / (2T)`, built with one forward evaluation serving
value and derivative (D-127), the object samplers and tempering consume.

**D-131. A space says whether it discretises functions, and densities refuse
there** (2026-10-24, David, from the sampling brief). `HilbertSpace.
is_function_space` is a boolean property, `False` at the base: `True` on
`SymmetricSpace`, the `sem1d` spaces and both MFEM backends (`MfemDofSpace`
stays `False` — a dof vector under the dot product is a coefficient space by
its own admission); a `DirectSum` carries it if any summand does and
`MassWeightedSpace` takes its base's, with `MfemHilbertSpace` overriding
that delegation. On a marked space the density family refuses without
`unsafe=True`, exactly D-11's pattern for point evaluation: `log_density`,
`grad_log_density`, `mahalanobis_squared`, `log_normalizing_constant`,
`credible_set`, `weakened_ellipsoid` at `power=1` only (below one it is
covariance geometry, not the distribution's own set), and the mixture's
`log_density` and `marginal_probabilities` (whose attribution drifts with
the truncation wherever the components' covariances differ). The reason is
the reason the sampling layer is dimension-robust: a density against a
discretised space's own volume measure changes with the truncation and
vanishes in the limit, while a density relative to a *reference measure*
survives it, and `AbsolutelyContinuousMeasure` is where those will live.
`chi_squared` passes the gate with `unsafe=True` internally: a misfit is a
likelihood potential, defined whenever the residual lies in the error's
Cameron-Martin space, not a density on the data space.

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
reported success 2 per cent low. Recorded as open at the time: `"kkt"` is the cheapest
general solver on balls and ellipsoids and `auto` could prefer it over the
bundle; closed by D-136, where the sets' forms choose the algorithm and
the quadratic tier is matrix-free. The finite-difference dual gradient is dropped:
on a nonsmooth cost, one evaluation per data dimension, silently, it is not a
subgradient; a maximiser is asked for by name.

**D-127. The MAP objective is built from one linearization, and a functional
can be** (2026-10-24, from the sl3dpp request list). `Functional.from_callables`
takes `linearize=`, exactly as `Operator`'s does under D-29: one call yields
the value and the derivative together, `at()` uses it, and `derivative()`
falls back to it only when nothing cheaper was supplied. `value` stays
required and cheap, because a backtracking search pays for values alone.
`MaximumAPosteriori.objective` is the first consumer: its `linearize` runs
`forward.at(model)` once and reads the misfit from the linearization's value
and the gradient from its derivative, where it previously ran the forward
once for the value and again inside the derivative — with a forward that
takes minutes to hours, the double evaluation was the whole cost of a line
search. The scaled and summed functional nodes already linearise through
their bases, so `log_posterior` inherits the single evaluation. A per-point
cache of the last linearization was asked for too and is not built: the
Wolfe search already returns its final model to the optimizer, and a cache
keyed on vector identity is the kind of silent state D-38 keeps out of
solvers.

**D-134. The convex algorithms belong to the sets, and the sets dispatch by
form** (2026-10-24, David; the redesign CURRENT_STATE §7 held open, settled
without Mag). The inference API was already right and stays: `BackusGilbert`
maps data and an error measure to a point of the property space;
`BackusGilbertParker` maps data, a convex error set and a convex prior set
to a convex subset of it. Everything else — bundle, KKT, Chambolle-Pock,
bisection, the closed form, the dual — is computational convex geometry:
ways of learning about a convex set that is only implicitly defined, the
image under `T` of the prior intersected with the data's preimage. So the
algorithms move into the sets. A probe (`support`, `contains`, `extent`,
`project`, `is_empty`, ...) dispatches to the cheapest algorithm the set's
*form* admits, the form being determined by the error and prior sets;
forcing a named algorithm for a cross-check stays possible, the parity
tests being the point of the layer. The three questions it raised were
settled the same day, all David:

(1) **Choice surfaces on the probe alone.** No construction-time algorithm
state: `support(d)` dispatches, `via=` forces — a name for the cross-check,
an algorithm *instance* when tuning rides along. The default is not flat
but a **form-ranked table**: each algorithm declares the set forms it
accepts, and the probe takes the best applicable entry for this set's
form, "best" recorded per form from measurement rather than guessed. D-82
stops being an API question and becomes one row of that table, settled by
timing when the implementation lands; the dual gap certificate (REVIEW2
Q2) is likewise a property of the dual algorithm's entry.

(2) **The estimator stays, and it is no thinner than any other.** David:
the Bayesian estimator returns a measure with all the hard work in doing
something with it; this one returns a convex set with all the hard work in
probing it. The answer object carries the computation in both columns of
D-71's classification, and `problem + property_operator + prior` remains
the one spelling learned across point, measure and set inference.
`FeasiblePropertySet` additionally becomes constructible directly from
sets, for callers outside the `ForwardProblem` machinery.

(3) **The whole-hierarchy pattern is the architecture, built
incrementally.** The dispatch machinery and the form declarations live in
the geometry layer from day one — never inside `backus.py` — with
`FeasiblePropertySet` the first composite client; intersections, images
and preimages adopt it case by case as something wants them, each
adoption its own entry. The full generality is correct and will probably
be used sparingly, which is exactly why it is not built in one go.

Underneath, the solver zoo unifies into the engines the probes call:
`BundleResult`, `KKTResult` and `SaddlePointResult` fold into one result
shape and the bundle methods join the `Optimizer` hierarchy — the detailed
sub-decisions land with the implementation. Settled in detail as D-135 to
D-152 on 2026-10-27; the form-ranked registry of (1) is deferred by D-136.

**D-135. A set declares what it supplies; the convex machinery converts
where that is affordable** (2026-10-27, David). Every closed convex set has
a support function and a level function in principle; the `has_*` flags on
`ConvexSet` mean "supplied cheaply", and generic code never requires more.
Two conversions are provided as defaults behind the flags, labelled as
conversions: a set with a smooth level function gets a support function and
a maximiser by a one-multiplier maximisation of `(q, x)` under the level
constraint; a *finite-dimensional* set with a support function and a
maximiser gets a distance, hence membership with a certificate either way,
by the concave maximisation `d(p) = max over ‖q‖ ≤ 1 of (q, p) − h(q)` whose
supergradient is `p − p̂(q)`. The converse conversion in a model space would
be an optimisation over infinitely many directions and is not offered. So
the feasible property set always carries both characterisations, D-80
becoming a guarantee; sets known by projection alone reach the support side
through a primal-dual splitting, which is why Chambolle-Pock stays (D-139).
Built 2026-10-27: the second conversion is the Fenchel dual of the
nearest-point problem, ``½ d(p)² = max over q of (q, p) − h(q) − ½‖q‖²``,
unconstrained and strongly convex, so it yields the projection as well as
the distance and a proximal bundle takes it; its argument is accurate to
the square root of the bundle's gap tolerance.

**D-136. Three forms, one algorithm per form per probe, no registry yet**
(2026-10-27, David). The form of the pair `(prior, noise)` is decided by two
predicates, *exact* (the noise set is a single point) and *quadratic* (a
ball or an ellipsoid, degenerate forms included under D-138), giving three
tiers: the closed form (exact data, quadratic prior: Al-Attar 2021 eq. 2.84
in the prior's own inner product, which generalises the ball case to
ellipsoids and centres with no new mathematics and removes the construction
crash of an ellipsoid prior with exact data), the quadratic tier (two
quadratic sets, noise not a point), and the general tier (everything else,
routed by capability under D-135; exact data with a general prior is the
general tier with the point `{b}` as the noise set and needs nothing of its
own). The feasible set exposes `form` and `algorithms`; `via=` on a probe
forces an algorithm by name or instance (D-143). The dispatch is a predicate
and a per-probe choice of at most three entries, written as functions. The
form-ranked registry in `geometry` that D-134(1) and (3) described is
deferred until a second composite set wants the same dispatch; the algorithm
objects already have the shape it would register (`name`, `accepts`, the
probe methods), so adding it later changes no user-facing line. D-82's open
question is closed by this: the tiers are nested specialisations, not rival
algorithms for one form, so there is nothing to rank by timing. Built
2026-10-27 as `geometry/feasible.py`: the closed form's membership goes
through the property-space shape `S` (the level is `(C⁻¹ũ, ũ) + (S⁻¹(p − c),
p − c)`), which retires the joint spectrum of `(A; T)` the old route formed,
dense in `N_d + dim P`.

**D-137. Membership is one algorithm, the inner problem chosen by form**
(2026-10-27, David). Al-Attar (2021) §3.3: write `m = m̃ + u` with `m̃` the
model nearest the prior's centre having property `p` (`dim P` solves, cached)
and `u ∈ ker T`; the prior's level function separates and what remains is
the feasibility problem again with `A` restricted to `ker T` (eq. 3.28).
Emptiness and membership are therefore one search, with and without the
kernel restriction, and the fitting model its by-product. On quadratic sets
the inner minimiser is a damped least-squares solve, so the search is
`misfit_search` over a `TikhonovFamily` whose forward operator is `A ∘ P_K`
with the projector applied matrix-free; on general sets with smooth level
functions it is the convex minimisation the likelihood engine already does;
on the closed-form tier it is the ellipsoid's quadratic form; elsewhere it is
D-135's distance. The dense projected spectrum of the "reduced" engine is a
solver choice under D-145, not an engine, and is deleted; so are the
separate `_minimum_norm_fits` and the dual route's own feasibility test. The
search's multiplier is the gradient of the feasible set's level function in
`p`, which `level_function()` carries.

**D-138. Degenerate quadratic sets are admitted, and `+∞` is a legal support
value** (2026-10-27, David). Two sets from the papers do not fit a
positive-definite `Ellipsoid`: Mag's `L²` amplitude ball on an `H²` space
(positive definite on the truncation, compact relative to the identity, so
`A C A*` is severely ill-conditioned and the set unbounded in the model
norm) and Al-Attar (2021) §4.2's semi-norm bound (a genuine kernel, a
cylinder). `Ellipsoid` accepts a semidefinite precision for inference use;
a `Cylinder` with an explicit kernel basis is planned for the true-kernel
case and may be dropped if it proves hard. On the quadratic tier the family
`γQ + A*PA` needs `ker Q ∩ ker A = {0}` and the model-space formalism when
`C` does not exist; in the dual the certificate is constrained to keep
`T*q − A*λ` where the prior's support function is finite, an affine
constraint of `dim(ker Q)` equations enforced in the proximal step. Where
the data do not see a direction the support value is `+∞`, returned as a
value; `Polytope` omits that half-space and says so.

**D-139. The general tier keeps three algorithms, one per capability**
(2026-10-27, David). `Primal` for smooth level functions on both sides (two
monotone root finds around an inner problem that is linear on quadratic sets
and a Newton solve otherwise — the bisection route and `LevelKKTSolver`
were one algorithm with two inner solves), `Dual` for support functions and
maximisers (the proximal bundle by default; `method=` takes any
nonsmooth-capable `Optimizer`, `SubgradientDescent` a legitimate choice),
`Splitting` for projections (Chambolle-Pock), with `ClosedForm` on the exact
tier. Deleted as redundant or dominated: `PrimalKKTSolver` (`fsolve` on a
dense data Gram with a `1e-3` floor; Mag's own runs needed its
nested-bisection fallback, which the root finds are), the level bundle
(dense master QP in `N_d + 1`, D-56's restoration notwithstanding; in git if
a measured case wants it), and the smoothed dual (its one general use, the
Moreau envelope of a nonsmooth set, is not being built). The rule, David's:
where there is a genuine choice the user makes it, as with linear solvers;
what goes is only what is redundant or sub-optimal. What each gives back
differs and the result says so: `Primal` the contact point and extremal
model, `Dual` a certificate, `Splitting` both iterates.

**D-140. `BackusGilbert` returns the set, takes quadratic sets, and
evaluates any certificate** (2026-10-27, David). Both papers state the
method more generally than the two origin-centred balls the class demanded:
Al-Attar (2021) eq. 3.35 puts the noise covariance in the normal equation,
and Mag eq. 42 gives the uncertainty set `X d + (T − XA) S_prior + (−X)
S_noise` for any convex sets, centres handled by their support functions.
So: the automatic `X = T C_Q A*(A C_Q A* + α C_R)⁻¹` for a ball or ellipsoid
on each side (`dim P` data-space solves through the user's solver); the
uncertainty set as a real Minkowski-sum `ConvexSet`, which `__call__`
returns since the set is the method's value and the point `X d` is its
centre (`estimate` and the two-part `error_bars` as conveniences on it);
`from_operator(problem, T, X, prior, noise=)` to evaluate a user-chosen
linear certificate against any sets with support functions, which is where
a SOLA or Zaroli kernel can be assessed against a named prior. SOLA is this
estimator with the trade-off parameter chosen by hand and no prior bound
named, which is why it cannot certify a resolution error; that is a
sentence in the docstring and tutorial 7, not a class. Point estimators
(least squares, SOLA) remain for completeness and as tools; they take no
prior and give no uncertainty, and the library says so. Built 2026-10-27:
the set is `BackusGilbertSet`, `BackusGilbert` is a `SetEstimator` whose
affine part is `point_estimator`, the trade-off is `alpha=` (one by
default), and `error_bars` reports each part's larger reach on the two
sides of the estimate, which is the reach itself for a symmetric set.

**D-141. Ellipsoidal probes: the certified ones now, the descriptive ones if
workable** (2026-10-27, David). On the quadratic tier, for `θ ∈ (0, 1)` the
set `{θ·prior(m) + (1 − θ)·misfit(m) ≤ 1}` contains the feasible model set
(the S-procedure), and its image `T C_θ T*` with `C_θ = (θQ + (1 − θ)A*PA)⁻¹`
is a certified outer ellipsoid of the feasible property set at `dim P`
damped solves per `θ` and a coarse one-dimensional search in `θ` on the
trace or log-determinant: tighter than Backus-Gilbert's sum because the data
enter the constraint, checkable against exact support values, and the same
family the primal route sweeps. The Minkowski sum of two ellipsoids gets the
trace-optimal outer ellipsoid `(1 + 1/p)Q₁ + (1 + p)Q₂`, which makes
`BackusGilbert`'s set a single ellipsoid when both inputs are quadratic.
Planned as well, droppable: the maximum-volume inscribed ellipsoid of the
outer polytope (a `dim P` log-det program; a description of the bound, not
of the set) and a point-set minimum-volume enclosing ellipsoid (encloses the
inner hull and certifies nothing, documented as such). Built 2026-10-27, all
four: `SProcedure` as the `outer_ellipsoid` probe's algorithm (the closed
form answers with the set itself), `sum_of_ellipsoids` and the Minkowski
sum's `outer_ellipsoid()`, `Polytope.inscribed_ellipsoid()` and
`enclosing_ellipsoid()`; the last two dense in `dim P` as planned.

**D-142. The damped-family machinery moves to `numerics`; the feasible set
and its algorithms live in `geometry`; the estimators stay in `inference`**
(2026-10-27, David). `FactoredNormalOperator`, `NormalOperator`,
`choose_formalism`, `TikhonovNormalOperator`, `TikhonovFamily`, the
structure-aware preconditioners and `misfit_search` are regularised
least-squares machinery and belong in `numerics`, names unchanged and
re-exported from `inference`. `FeasiblePropertySet`, `ClosedForm`, `Primal`,
`Search`, `Dual`, `Splitting`, `Distance`, D-135's conversions and D-141's
ellipsoids are computational convex geometry and live in `geometry`, which
already imports `numerics`. `BackusGilbertParker` and `BackusGilbert` are
thin estimators in `inference`: a forward problem, hardening, and the
geometry object. D-111 holds: nothing below `inference` imports it.

**D-143. The answer object's API** (2026-10-27, David). Construction:
`BackusGilbertParker(problem, T, prior, /, *, noise=None, level=0.95,
solver=None)(data)`, or directly `FeasiblePropertySet(A, T, prior, data, /,
*, noise=None, solver=None)` with `noise=None` meaning exact data; `route=`,
`membership=`, `method=`, `optimizer=` and `max_iterations=` go, tuning
riding on an algorithm instance passed to a probe; `solver=` stays because
it is estimator-wide and carries the preconditioner. Probes: `form`,
`algorithms` and the read-only structure; `is_empty()`; the support side
`support`, `support_values(..., n_jobs=, warm_start=)`, `support_bracket`,
`support_maximizer`, `extremal_model`, `certificate`, `polytope`, `outside`,
`outer_ellipsoid`; the level side `contains`/`admits`, `inclusion_level`,
`level_function()` with gradient, `level`, `fitting_model`, `extent`,
`inner_hull(..., source="contact" | "admitted")`; and the `ConvexSet`
surface. `via=` takes an algorithm's snake-case name or an instance, the
same name serving every probe the algorithm answers; a refused name lists
the admissible ones. `inclusion_norm` is retired (`sqrt` of the level);
`route`, `membership` and `algorithm` are replaced by `form` and
`algorithms`. A construction-time default algorithm is not offered: it
would reintroduce the choice-before-data shape D-134 removed. Built
2026-10-27; the estimator also exposes `form` and `structure`, and its
`noise` is `None` for exact data.

**D-144. The quadratic tier's root find: one multiplier eliminated if it is
sound, nested monotone root finds as the fallback** (2026-10-27, David).
Dividing stationarity by `t`, the family `M(γ) = γQ + A*PA` depends on
`γ = s/t` alone and the model is affine in `1/t`: two warm-started solves at
a `γ` give the model for every `t`, and both constraints are quadratics in
`1/t` with closed-form roots. One multiplier is eliminated analytically and
what remains is a scalar root find in `γ`, roughly 20–40 solves per
direction against the 3600 fixed-depth bisection steps of the dense route.
Its monotonicity in `γ` is not established (the cross term `(Qu₁, u₂)` has
no sign) and is the first numerical task of the implementation; the nested
structure (`t` outer, `s` inner, both monotone by Al-Attar 2021 lemma 3.1
and the convexity of the partially minimised dual) is wired in from the
start as the fallback, with bracketed secant or Brent steps and warm starts
replacing fixed-depth bisection. A joint two-dimensional Newton stays
rejected (Mag's `fsolve` experience is the evidence).

**D-145. Two caches with clear owners, and the dense spectrum becomes a
solver** (2026-10-27, David). The estimator owns the data-independent part
(the kernel projector pieces and their `dim P` Cholesky, the closed-form
pieces, the family with its solver and preconditioner) and hands it to every
set it returns. The set owns the data-dependent part: a bounded cache of
`u₂(γ) = M(γ)⁻¹ A*P d′`, which is direction-independent so that half the
solves of a sweep are paid once per distinct `γ`; the last root per probe,
from which a neighbouring direction's bracket starts; the fitting model and
the probe results. Caches key on `γ` within the solver's tolerance, never on
vector identity (D-38), and nothing cached is reported converged unless it
was solved to the tolerance in force when stored. `with_data(d₂)` gives a
directly built set the same sharing. The present eigendecomposition of the
whitened data Gram is not lost: it is `SpectralFamilySolver`, a
`LinearSolver` for a one-parameter family that factors once at `O(N_d³)` and
answers every member by spectral division, passed as `solver=` by a user who
wants it and never assumed; it reproduces the present numerics exactly and
is how the parity bar is kept while the matrix-free path is built. Its
interface is a factorisation of the data Gram, so a randomised low-rank
version by `random_eig` can follow without changing callers.

**D-146. Preconditioning: connect what exists, defaults by formalism,
family-aware updates** (2026-10-27, David). The data-space family
`A C A* + γ P⁻¹` is the Gaussian normal operator with `C` as prior
covariance and `γ P⁻¹` as noise covariance, so built as a
`FactoredNormalOperator` it takes every existing preconditioner unchanged,
the structure-aware ones reading `A`, `Q`, `R` off the member through
`DampedSolves.assemble=`. A preconditioner may implement `at(multiplier)`
and `DampedSolves` then prefers it to the `refresh` rule; the probed
diagonal, the spectral deflation and the Woodbury surrogate all update
analytically in `γ` (the Woodbury one by keeping `Q_s⁻¹` and `A_s* P A_s`
separately and refactorising the surrogate-dimension matrix per
multiplier). Defaults: `C`-preconditioning in the model-space formalism
(the identity plus a decaying operator; a no-op for a ball), none in the
data-space one. The kernel-restricted family of D-137 reuses the
unrestricted family's preconditioner (a rank-`dim P` modification). No new
preconditioner classes; whether a randomised deflation of `A C A*` becomes
the data-space default is measured on the benchmark (D-152), not guessed.
David: in real applications the Woodbury preconditioner built around a
surrogate with a low-dimensional model space has been by far the best
general method, so it is the documented route and the benchmark includes
it. Built 2026-10-27 as `FamilySolver.for_family(base, shift)` returning a
`BoundFamily` whose `at(multiplier, member)` answers each member, the same
protocol for a solver (`SpectralFamilySolver`) and a preconditioner; the
probed diagonal and the spectral deflation implement it. Corrected the
same day: the Woodbury preconditioner needs no `at()`, because its build is
lazy and its inner factorization is of surrogate dimension, so
`refresh=1.0` already rebuilds it per member at negligible cost.

**D-147. `formalism="data_space"` here too** (2026-10-27, David). D-74's
rule holds: `"auto"` is the default nowhere. The first of D-74's reasons
reverses on this side (a quadratic set always has its precision; a
covariance is what may be missing) but the second stands (choosing by
dimension lets a refinement switch the algebra), and the data side is where
the preconditioners and the surrogate route live. The exceptions
(precision-only, degenerate, semi-norm priors) are the cases where a user
knows their prior is unusual and passes `formalism="model_space"` once; the
data-space route refuses with a message naming that when the prior cannot
supply `C`. "Whichever is smaller" is advice in the docstring, not a
default. Built 2026-10-27: the data-space formalism is refused outright
when the data outnumber the model, since its family is singular at `γ = 0`
and the data-only bound is out of reach there; the kernel-restricted
membership search is assembled in the data space whatever the formalism,
because the restriction of the prior's covariance to `ker T` is a
rank-`dim P` downdate there and has no precision.

**D-148. Reported values are outer bounds; every probe carries a certified
bracket; the solver floor is checked** (2026-10-27, David). On the
quadratic tier the primal root find yields a model scaled into both sets
(a rigorous inner, attainable value) and a dual certificate
`λ = t·P(d′ − Au)` whose cost is evaluated in closed form (a rigorous outer
value by weak duality whatever the solve's accuracy); on the general tier
the dual gives the outer value at any `λ` and `Splitting`'s scaled primal
iterate an inner one; the membership search's multiplier bounds its minimum
from below. `support()` returns the outer value on every tier so that a
quoted number is a bound; `support_bracket()` returns `(inner, outer)`,
`(−∞, outer)` where no inner point exists; `converged` means the bracket
closed to the instance's tolerance, a report rather than a second gate since
non-convergence already raises through the solvers under D-8. The parity
tests compare brackets, not floats. Mag's and D-82's acceptance rule,
constraints met to `1e-6` relative, is the default root-find tolerance. At
construction the estimator compares the solver's `rtol` with the noise
radius over `‖d‖` and the prior radius over the model scale, refusing under
D-8's strictness and warning otherwise: a radius below the solver floor is
one no fit can be certified to.

**D-149. The dual runs in the noise metric, stops on the bracket, warm-starts
its certificate, and detects emptiness by divergence** (2026-10-27, David).
The bundle's proximal term is an inner product on the data space and the
Euclidean one is wrong whenever the noise set is not a ball in it; `Dual`
runs the bundle on a `MassWeightedSpace` of the data space with the noise
precision as metric, no change to the bundle. Stopping on D-148's bracket
when an inner point exists and on the bundle gap otherwise, reported.
Cuts do not transfer between directions but the certificate does: the
neighbouring direction's `λ`, and `−λ(q)` for the antipodal direction of a
symmetric noise set. On the support-only tier an empty set has support
`−∞` everywhere, so the dual is unbounded below; a divergence test on the
cost decides it, certified, rather than the iteration cap. `Dual` takes
D-138's affine constraint on `λ` and refuses sets whose support function is
infinite on a non-subspace region (cones go through `Splitting`). `start=`
on the instance allows cross-seeding from a `Primal` certificate. Built
2026-10-27, with one addition: at the optimal certificate the prior's
maximizer at the residual `T*q − A*λ` is the model attaining the bound, so
the dual exhibits a contact point after all, reported as the inner value
when that model fits the data and as `−∞` otherwise; it is what makes the
distance conversion of D-135 available on the support-only tier. The
affine constraint for degenerate priors waits for `Cylinder`.

**D-150. One result shape at the solver level, two at the probe level, and a
sweep result** (2026-10-27, David). The bundle methods join the `Optimizer`
hierarchy and `BundleResult`, `KKTResult`, `SaddlePointResult` fold into
`OptimizationResult` subclasses (the bundle's gap and cut count, the
splitting's primal-dual gap; `gradient_norm` is `NaN` where a method has
none; Chambolle-Pock minimises `−(c, m)` so `minimizer` and `value` are
literally right and `.model` goes). Each algorithm's probe returns a frozen
`SupportResult` (outer value, inner, gap, converged, algorithm, evaluations,
certificate, extremal model, multipliers) or `InclusionResult` (level,
lower bound, converged, algorithm, evaluations, fitting model, multiplier),
cached by the set per direction or point; the float-returning probes are
views of one cached result, so one solve serves `support`,
`support_bracket`, `certificate`, `support_maximizer` and `extremal_model`.
A `SweepResult` carries a sweep's per-direction results for `polytope` and
`outer_ellipsoid`, which must know which half-spaces are infinite;
`support_values` keeps returning the array and `support_sweep` exposes the
wrapper. `converged`, `iterations`, `message`, `value` mean one thing at both
levels, and `iterations` is only ever a count (D-99). 2026-10-27: `KKTResult`
went with the two KKT solvers it reported for; `SaddlePointResult` stays as
the splitting's result, carrying its certificate and discrepancy.

**D-151. Parallelism: threads, chunked so warm starts survive** (2026-10-27,
David). `n_jobs=` on every independent loop (`support_values`, `polytope`,
`outer_ellipsoid`'s `dim P` solves at one `θ`, `inner_hull` over points,
`extent`'s two ends), through `parallel_map` (D-52). Directions are split
into `n_jobs` contiguous chunks in the order given, so that warm starts run
within a chunk and one cold start is paid per chunk; the `u₂(γ)` cache and
the result caches take a lock; the preconditioner and projector pieces are
read-only and shared. `n_jobs=1` and `n_jobs=k` must agree to tolerance and
the tests check it. Built 2026-10-27: `parallel_map` gained `threads=`,
the one way the library asks joblib for its threading backend (D-52 holds:
joblib is still called only there), because the sweep's closures hold the
set's caches and locks, which no process backend can pickle. A process backend, which needs picklable operators and
per-worker caches merged afterwards, is the sl3dpp "process-safety
declaration" item and waits for it after the release.

**D-152. The acceptance benchmark is the two papers' problems** (2026-10-27,
David). Three problems, each for a reason: point data on the sphere
(Al-Attar 2021 §2.5.2 and §3.3.2: 250 values, `H²` with `λ = 0.25`,
coefficients of degree ≤ 1 and ≤ 3, exact and noisy — the closed form
against the primal tier as noise → 0, extents against brackets,
`dim P = 15`); flexure–gravity (Mag §3.2, Table A2: two fields, the `L²`
amplitude ellipsoid, a chi-square ball in whitened coordinates, 156
directions — the degenerate prior, the noise metric, ill-conditioning, the
contact-point hull, the S-procedure ellipsoid, convergence under
refinement); phase-delay tomography with nested acquisition (Mag §3.1,
Fig. 9: `N_d` from `10²` to `10⁴`, ball prior, 12 directions — cost scaling
of matrix-free against the spectral solver counted in applications,
unpreconditioned, deflated and surrogate-Woodbury runs, `n_jobs`
invariance). Reduced versions are slow tests; paper-scale scripts live in
`benchmarks/`, run by hand, their measurements recorded in this file's
appendix and used to set the defaults D-146 leaves open; the flexure–gravity
problem is also the "two fields, one chain" mini-app of `EXAMPLES_PLAN.md`.
Acceptance: brackets overlap wherever two algorithms apply and the spectral
route reproduces the present numbers to `1e-10`; the phase-delay problem at
`N_d = 9000` runs matrix-free with the surrogate preconditioner, counts
recorded; the flexure bounds move by under a few per cent between the two
finest truncations; `n_jobs` and `via=` change nothing but cost.

**D-153. The one-multiplier reduction is sound, and it is a scalar
minimisation of the partially minimised dual** (2026-10-27, measured;
Stage 0.1 of `CONVEX_PLAN.md`, scripts in `benchmarks/`). Cleaner than
eliminating a multiplier through the constraint quadratics: with `s = γt`
the dual `ψ(s, t) = ½(b, M⁻¹b) + (s + t)/2 − (t/2)(Pd′, d′)` becomes
`A(γ)/(2t) + B(γ) + tC(γ)/2` with `A = (c, u₁)`, `B = (c, u₂)`,
`C = (A*Pd′, u₂) + γ + 1 − (Pd′, d′)` from the two solves
`u₁ = M(γ)⁻¹c`, `u₂ = M(γ)⁻¹A*Pd′`; the minimum over `t` is closed form,
`t* = sqrt(A/C)`, leaving `φ(γ) = B + sqrt(AC)` to minimise over `γ ≥ 0`,
with `φ(∞)` the prior-only bound and `φ(0)` the data-only bound where
`A*PA` is invertible. Measured on twelve problem families (balls,
ellipsoids, off-centre sets, prior condition numbers to `1e8`, noise from
loose to very tight, overdetermined, `dim P = 15`), 24 directions each:
`φ` evaluated in the data-space formalism is unimodal in every direction;
its minimum agrees with a direct minimisation of `ψ(s, t)` to `1e-13` and
with the present bisection route to `1e-13` wherever that route runs; a
warm-started bracketed Brent in `log γ` from the neighbouring direction's
root takes about 11 evaluations per direction at a tolerance of `1e-3` in
`log γ`, which is enough because the value is stationary there (second
order in the parameter error), against 3600 fixed-depth steps today.
Two cautions that shape Stage 3: the data-space evaluation must use the
resolvent identity, `A(γ) = κ/γ + (ACc, G⁻¹RK⁻¹ACc)` with the kernel part
`κ = (c, Cc) − (ACc, G⁻¹ACc)` computed once per direction (the dense
model-space solve of `γQ + A*PA` loses five digits at tight noise and
produces spurious minima below `γ ≈ 1e-7` in the underdetermined case); and
both ends `γ = 0` and `γ = ∞` are candidates to be evaluated explicitly,
since an overdetermined problem's prior is slack and the present route
crashes or returns a value off by `6e-5` there. `B(γ)` is
direction-independent apart from the pairing, so with `K(γ)⁻¹d′` cached by
`γ` (D-145) a sweep costs about one solve per `γ` per direction. Built
2026-10-27 as `Primal` in `geometry/feasible.py`; two refinements from the
implementation: the inner value is taken at the largest feasible point of
the ray `u₂ + x u₁` at the found `γ`, both constraints being quadratics in
`x = 1/t` with closed-form roots, which is first order in the search
tolerance where the outer value is second order, so the default tolerance
is `1e-4` in `log γ`; and `converged` reports the scalar search, the
bracket's width being the gap. The certificate is `λ = t·R⁻¹(Au − d′)`,
the dual's sign.

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

**D-130. `power_measure` means the `L2` spectrum on every space**
(2026-10-24, David, closing CURRENT_STATE §7's open question). The spectrum
a modeller writes down is the `L2` one, so `power_measure(p)` now carries
the gram diagonal into the eigenvalues it hands `invariant_measure`: a
draw's expected `power_spectrum` is `p` on a space of any Sobolev order,
where before the observed spectrum came out divided by the Sobolev symbol
and the same call meant a different random field on each host space. The
round trip `E[power_spectrum(sample)] == p` is now a universal identity and
tested as one; consistency with `power_spectrum` was worth more than metric
purity. `invariant_measure` stays the metric-native spelling for anyone who
wants covariance-operator eigenvalues in the host's own inner product. The
caution that replaces the old asymmetry: an `L2` spectrum is trace-class on
`H^s` in the limit only when `p` decays faster than the Sobolev symbol —
on a truncation everything runs, and the dimension-free reading is the
caller's to check, as with point evaluation under D-11.

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
voxels, plotly or `Subset.plot` method. The 3-D half of the drop was
reversed by D-133 on 2026-10-24; the rest stands.

**D-133. `plot_set` draws three dimensions, and plotly is an engine, not a
default** (2026-10-24, David, reversing half of D-94's drop for the 2.0
release). The same one function and the same route-by-capability, extended:
the support route samples directions on a Fibonacci sphere and draws the
convex hull of the *attaining* points — an inscribed approximation, every
vertex a point of the set, where the 2-D polygon of supporting lines
circumscribes — so in 3-D it also requires a maximizer, and a
support-function-only oracle falls through to its other routes. The level
and membership routes raster `resolution³` cells onto `Axes3D.voxels` with
the edges in parameter coordinates; the default resolution becomes 30 in
three dimensions (200 stays in one and two) and more than 30 warns with the
oracle-call count, v1's guard. Slices need nothing new — `_Picture` never
cared about the dimension. `engine="plotly"` is the optional interactive
route (extra `plot3d`), 3-D only, returning `(None, Figure)`; there is no
`engine="auto"`, because a silently switching backend is not this library's
style. Example 34 shows a sliced 4-D ellipsoid's rastered extents matching
the exact pullback ellipsoid to three decimals.

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
a mass matrix); `operator_from_bilinear_form` keeps the sparse matrix (`MfemSpace.restrict`
keeps CSR and `from_matrix` keeps a sparse input sparse; an earlier
version densified, and this entry said so until 2026-10-28). In the coordinate backend `FormSystemMatrix` is not used
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
package imports it, and it goes with its test when v1 is removed, so
the adapters do not ship in 2.0 (the release mechanics in
`CURRENT_STATE.md` §10 delete `compat.py` with v1). It
deliberately does not delegate `white_noise` to v1 (D-19).

**D-102. All imports inside the package are relative** (2026-08), so the
rename of `pygeoinf/` to `pygeoinf/` touches no import; only `tests/` and
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

**D-105. Axiom checks live in `pygeoinf.testing`, not the production MRO**
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

**D-125. A statistical check judges an entry against its own scale**
(2026-09-18, David). `check_white_noise` compared `E[(x, u)(x, v)]` with
`(u, v)` to within `rtol * max(|(u, v)|, 1)`. The sampling error of that mean
is at most `sqrt(2 (u, u)(v, v) / n)`, which grows with the metric, so a fixed
floor of one made it a different test on every space: on a large metric a
correct space failed by chance, the off-diagonal entries first (it happened on
a `sem1d` Sobolev space once a Robin condition had lifted its eigenvalues,
1.4 standard errors against the tolerance); on a small one -- an MFEM mass
matrix -- nothing could fail, the v1 construction of D-19 included. The entry
is now judged against `sqrt((u, u)(v, v))`, as `check_measure` already judged
a covariance, and one `rtol` serves a metric of any size: 0.06 at 20 000 draws
is six standard errors on the diagonal. Every existing caller passed
unchanged, the two on MFEM spaces for the first time as a real test.

---

**D-128. A check takes the tolerance it enforces and returns what it
measured** (2026-10-24, from the sl3dpp request list). Every `check_*` takes
`rtol=` (and `atol=` where vectors are compared), defaulting to exactly what
it required before the parameter existed — v1's checks had `check_rtol` and
`check_atol`, and losing them in the port was a regression, not a decision.
Every check returns a `CheckReport`: the check's name and, per identity
tested, the worst *relative* defect observed, the same quantity the tolerance
bounds. A failure still raises `AssertionError` first, so a report in hand
means the check passed. The point is the backend whose adjoint is consistent
only to discretisation level — a continuous adjoint: checked with the `rtol`
it can meet, its defect is read back, recorded as it falls under refinement,
and sets the floor below which a solver tolerance asks for noise.
`check_operator` applies one `rtol` to linearity and the adjoint alike,
deliberately: an operator that is a solve is linear only to its inner
tolerance, so the two degrade together (its default is now 1e-9 throughout,
where linearity was 1e-10 — the adjoint identity was always the binding
check). The statistical checks (`check_white_noise`, `check_measure`) keep
their D-125 scale logic and report deviations in the units their own `rtol`
bounds, the mean in standard errors. `check_coordinates` folds in
`check_representer`'s defects and `check_affine` folds in `check_operator`'s
on its linear part, so the caller of the umbrella sees everything measured
beneath it.

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

**D-114. pyshtools is always paired with ducc0** (2026-09-18, David). The
`sphere` extra lists `ducc0`, pyshtools' fast transform backend, which
pyshtools selects by default when it is importable; nothing in the code names
it. Measured on analysis plus synthesis of a Sobolev field: 8.5 ms to 1.3 ms
at lmax 128, 57 ms to 6.7 ms at lmax 256, coefficients agreeing to 1e-13.

**D-113. Also later:** full function-space MCMC (D-7 lays the hooks), PETSc
(D-98), parallelism inside operator actions (D-6), and pointing Sphinx at
the new package (REVIEW2 question 11). No longer deferred (2026-10-27): the
convex solvers' API and the certified gap, settled as D-134 to D-152
without Mag; the gap is the bracket of D-148.

---

## 17. Spectral-element spaces

**D-115. `sem1d` wraps `planetmodel.randomfield`; the numerics stay there**
(2026-09-18, David; renamed twice the same day. From `radial`, because a plain
interval in a package of that name reads as a function of radius, which it is
not: `interval` is the line under `dx`, `radial` is D-121, `ball` the ball.
Then from `spectral_element` to `sem1d`, David's `1d_sem` with the digit moved,
a module name being unable to start with one: shorter, and it says the
spectral elements are along one coordinate, the ball's angles being
harmonics, which leaves the name free for a three-dimensional mesh). Sobolev spaces on an interval, a ball and an annulus in
the eigenbasis of `A = 1 - div(L² grad)` on a Gauss-Lobatto-Legendre mesh,
for what the Fourier `line` cannot do: a length scale that varies, a mesh that
is not uniform, the `r² dr` measure. The optional extra is
`planetmodel>=1.3` (1.2 at first; 1.3's grids carry the `kind` the batched
transform of D-123 checks, and the stale `[harmonics]` extra went on
2026-10-24), which brings pyshtools and ducc0 for the ball
(D-114), and the package is imported on demand, as the sphere is. One submodule per
geometry exporting `Lebesgue` and `Sobolev` (D-3); `base.SpectralElementSpace`
holds everything but the geometry. It is *not* a `SymmetricSpace`: nothing
here is homogeneous, so there is no reference point, no geodesic and no single
pointwise variance. It keeps that class's names wherever the meaning is the
same (`with_order`, `spectral_operator`, `dirac`, `point_evaluation_operator`,
`multiplication_operator`, `truncate`, `sobolev_measure`).

**D-116. A vector is the field's values on the padded grid, not its
coefficients** (2026-09-18). The plan going in had components as the vectors
with an identity coordinate map; v2's convention is the other one (D-1, D-84,
D-87) and these spaces follow it: values at the nodes of the padded mesh --
`(radii, latitudes, longitudes)` in a ball, on the Gauss-Legendre grid of
`lmax` -- with the eigen-coefficients as components and a product left on the
grid. The domain is a restriction (`interior_values`, `interior_mask`);
`project_function` holds a function constant across the padding and never
calls it outside the domain. Multiplication is self-adjoint on `Lebesgue`
because analysis is the transpose of synthesis under the grid's weights, none
negative (the centre of a ball has weight zero), and is lifted on `Sobolev`.

**D-117. Calibration is by a standard-deviation field** (2026-09-18).
`sobolev_measure(order)` has covariance `A^-order` and no `scale`, the basis
being the eigenbasis of one `A`; the draws are Matern with
`nu = order + space.order - d/2`. `pointwise_std=` takes a number or a field
and multiplies the diagonal measure's draws by `std / raw_std`, so the
covariance is `S C S*` with factor `S C^1/2` and no closed-form precision.
`pointwise_variance` returns a field. In a ball a function of `A` has a
variance that depends on radius alone, by the addition theorem; variances that
depend on the order square the harmonics on the grid. **A function handed to a
ball must be regular at its centre** (David, 2026-09-18): the part of degree
`l` of an analytic field goes like `r^l` times a series in `r²`, the modes do
so by construction (they match `j_l(kr)` to 5e-5 at five GLL nodes and 1e-10
at eight, near the centre and at degree 8 included), so draws are regular for
nothing, and a profile in `r` alone, being of degree zero, must be even: one
linear in `r` is a cone and converges at the centre at first order (appendix).
Test targets are chosen to be regular, and the one that is not is there to
show it.

**D-118. The ball's default truncation is isotropic on the domain; the
padding is four length scales** (2026-09-18, David). With neither
`radial_modes` nor `max_eigenvalue`, a ball keeps at every degree the radial
modes of the padded mesh whose eigenvalue is at most the first of degree
`lmax` *on the domain without its padding*, found by one more banded
eigenvalue solve. The first version took that bound from the padded mesh,
whose first mode of degree `lmax` is longer by the ratio of the radii, so the
radial resolution in the domain fell short of the angular by that ratio and
the mode count ignored the padding altogether (appendix). The dimension now
grows with the padded volume, which is the price of padding in three
dimensions and was hidden before; a shorter padding with Robin ends is the
way out, and is not wired through. A mesh that reaches `r = 0` is a ball's
even under an annulus, and drops the axis node above degree zero, so one
`radial_modes` cannot be "all of them" at every degree there. Padding defaults
to two length scales at each end under the default Robin condition and four
under the natural one (D-122), inwards no further than the centre: a ball
is padded outwards only. The statistical checks of the ball carried `slow`
(D-109) while a draw was one harmonic transform per radial shell; since D-123
they take five seconds where they took 158, and run with the rest.

**D-119. A sampled function is continued across the padding by odd
reflection** (2026-09-18, David). `project_function` never calls the function
outside the domain. It gave the padding the value at the nearer end, which is
continuous with a kink there, and the kept modes come to a kink at first
order. It now gives `2 f(end) - f(mirror)`, which matches the slope too;
`extension="constant"` keeps the old behaviour, and is the one that keeps a
positive function positive. On an interval and an annulus the reflection is
better everywhere measured, and over the domain as good as knowing the
function on the padding. The mechanism is an identity: the two fills agree on
the domain and truncation is linear, so the reflected error is the constant
one plus the truncation of a field that lives on the padding alone. At a
boundary that leaked term cancels the kink's error almost exactly (appendix),
which is the whole improvement. **In a ball the centre is badly controlled
under either fill**, because a unit-norm radial mode of degree zero is worth
at the centre about 1.4 times its index and at the surface about one: any
truncation error is focused there, erratic in sign as modes are added and
falling only as their reciprocal. For two functions with angular dependence
the reflection was the worse within `r < 0.25`, by up to ten times, at every
padding tried -- so not because the mirror image runs out of domain, which
was the first guess and was tested; for a purely radial one the two were
alike. The reflection stays the default: the surface is where data are, the
norm over the domain is better, and the centre is not mended by either fill.
What would mend it is a fit over the domain alone and not a projection over
the padded mesh, which is not built. **All of this was measured under the natural condition and four length
scales of padding.** Under the Robin default (D-122) a continued function does
not satisfy the condition at the mesh's end, the padding is half as long, and
the two fills come out alike over the domain, the reflection still the better
at an end by about two (appendix); what does not care is the fit of D-124,
which is now what `project_function` does unless asked for a continuation.
This touches only the sampling of a given function; priors, draws and
inversions are in the span, where point evaluation is exact.

**D-120. A ball is drawn by cutting it, and a section needs no rotation**
(2026-09-18, David). `plot` on an interval is a line, with `padding=True` to
see the padded mesh shaded. A ball has three views: `plot_shell` at one
radius, which is what `plot` does there and is drawn by the sphere's own
renderer on a Gauss-Legendre sphere of the same `lmax`, so every map keyword
comes with it; `plot_section` by any plane through the centre, named by
`longitude=`, by `pole=` or by `through=` two points; and `plot_profile` along
a ray, with `section_values` for the numbers. A section is the harmonics
evaluated round the great circle times the radial coefficient functions, one
small product for any plane, so no coefficients are rotated. The plane is seen
from its pole with the direction nearest north up: a meridian has its own
longitude on the right, two points are met anticlockwise in order, and the
equator is seen from above the north pole. What is drawn in a ball is the
field with the vector's components, since off the grid there is no other. The
renderers are registered from `plotting` when `planetmodel` imports, keeping
D-92's direction (plotting knows the geometry, not the reverse), at 77 ms on
an `import pygeoinf` of about 270, with neither pyshtools nor matplotlib
loaded by it.

**D-121. A function of radius is its own space, under `r² dr`** (2026-09-18,
David). `sem1d.radial` holds profiles on `[inner_radius, radius]`:
the part of degree zero of the ball without an angular grid, with the volume
integral `4π ∫ u v r² dr` for its inner product, so that a profile's
components are the ball's components of degree zero to rounding and the two
can be copied across. It shares `axis.AxisSpace` with the interval, from which
it differs in the measure, in stopping its padding at the centre, and in where
a point has a value: a radius names a whole sphere, so away from the centre
evaluation needs an order above one half, and at the centre, a point of space,
three halves. The guard therefore takes the points (`point_evaluation_order(
points=)`); measured, the Dirac's representer at the centre still grows with
the modes at order 1.25 and has settled at 1.75, and at `r = 0.5` has settled
by 0.75, in a shell or a whole ball alike.

**D-122. The mesh ends with a Robin condition by default, which halves the
padding** (2026-09-18, David: the whole point of padding is that the specific
condition should not matter, and this is the more efficient). `boundary=` is `None` (natural), `"robin"`, or
the coefficients. The matched condition is the one the decaying solution of
`A u = 0` satisfies: `γ = 1/L` on a line, where it is exact for `A⁻¹` and a
mesh that stops at the domain needs no padding at all; `1/L ± 1/r` along a
radius, the curvature term a third of the other at a shell's inner end in the
case measured. Priors are higher powers, and for those `0.7/L` is better than
`1/L` by up to an order of magnitude and never much the worse, so `"robin"` is
`0.7/L`, with the curvature under `r²` and nothing at the centre; a
coefficient that would be negative is held at zero. With it two length scales
of padding do what four do under the natural condition, which is the default
padding it sets; in a ball every degree shares the one coefficient and it
holds there too, with a quarter fewer unknowns in the case measured and a
shorter mesh (appendix). It is the default, `boundary=None` being the natural
condition with its four length scales; the first version had them the other
way about. A Robin
condition lifts the long modes, on a short length scale past the bound the
domain sets for the default truncation (D-118), which is then raised to the
mesh's own first eigenvalue of degree `lmax` so that every degree keeps a
mode.

**D-123. The ball transforms all its shells at once** (2026-09-18, David).
`planetmodel.harmonics` goes to pyshtools one shell per call, and that Python
loop was the whole cost of a draw and of a covariance application.
`sem1d._transform.ShellTransform` contracts the degrees by one
batched matrix product against the harmonics on the prime meridian and the
longitudes by a second, with no loop at all; it agrees with the other route to
rounding and is taken up to degree 128, the largest measured, beyond which the
other is (its Legendre step is dense, ducc0's is not, and the gain falls from
150 times at degree 8 to 1.5 at 128). Example 31 went from 10.5 s to 1.3. It
could move to `planetmodel`'s `harmonics` as it stands, and is to, later
(David, 2026-09-18): for the moment it stays here. The move was made on
2026-10-24: `planetmodel.harmonics.ShellTransform` exists (uncommitted in
that repo, agreement to 1.3e-13, same class and method names), its grid
functions deliberately do NOT reroute through it (no caching precedent
there, and the table is `(lmax+1)^3` numbers); once planetmodel 1.4 is
released the pin becomes `>=1.4` and `sem1d/_transform.py` is deleted.
Meanwhile 1.3's grids carry a `kind`, which the local copy now checks by
name — under 1.3 every grid kind has `lmax` and weights, so duck-typing
them stopped selecting.

**D-124. A sampled function is fitted over the domain alone, by default**
(2026-09-18; the default David's, the same day). `project_function`, whose
`extension` is `"fit"` unless said otherwise, asks nothing of the padding:
it returns the field in the span of the kept modes that best matches the
function on the domain, in the domain's quadrature, harmonic by harmonic in a
ball. The modes are the padded mesh's and nearly dependent on the domain, so
the fit is a truncated SVD (`rcond` 1e-10) and picks the smallest answer;
the coefficients came out no larger than the other fills'. It converges as the
function is smooth and not as a continuation is, by four to ten orders of
magnitude in the cases measured, **the centre of a ball included**: what D-119
records as the centre's trouble was the projection of a continued function
over the whole padded ball. David chose `"odd"` when the choice lay between
two continuations under the natural condition, and the fit once it existed
and Robin (D-122) had taken most of the reflection's edge over the constant.
`"odd"` and `"constant"` remain, and return the function's own nodal values:
what a raw multiplier wants and, for `"constant"`, what keeps a positive field
positive on the padding, as a field handed to `pointwise_std=` has to be and a
fitted one need not be. The fit is what a mean, a reference model or a
synthetic truth wants, and those are what the method is mostly for. D-120 gains `plot_points` for a ball, on a map by latitude and
longitude, and `plot_section_points`, which marks the points within a
tolerance of a section's plane at the foot of each in it.

**D-126. A piecewise-continuous field is a vector of a direct sum of layers,
and `Layered` is that direct sum** (2026-09-18, David: the space of many
geophysical inversions, in one dimension or in annuli). One `sem1d` space per
layer, each on its own padded mesh with its own length scale, order and
truncation, the paddings overlapping as fictitious things may; a vector is a
tuple of layer fields. `sem1d.layered.Layered` subclasses the coordinate
direct sum and **compares equal to, and hashes as, the plain `DirectSum` of
its layers**: a block operator or a product measure builds a plain one and has
to land here, which is the reason a direct sum's identity is its summands
alone (D-23), carried one step further. What it adds is what a
direct sum cannot know, that the summands are adjacent in space:
`layer_index`, `evaluate`, `dirac` and `point_evaluation_operator` at points
anywhere in the medium, each layer keeping its own order guard;
`jump_operator` across an interface, at directions in balls;
`integral_functional` over the whole; `project_function` with one function or,
for one that jumps, a list of smooth pieces, since a single function is asked
about an interface by both layers and cannot tell them apart; layerwise
products; and `sobolev_measure` layer by layer, the layers **independent**,
which is what a discontinuity means. Continuity across an interface is that
prior conditioned on the jump vanishing (`measure.condition(jump, 0)`), exact
to 1e-14. **A point on an interface has two values, and evaluation there is
refused unless `side="below"` or `"above"` says which**; layers may leave a
gap, a field in a mantle and an inner core but not the fluid between, and a
point in a gap is in no layer. Arguments of the constructors
(`Layered.interval`, `.radial`, `.ball`) are given once for all layers or, **as
a list**, layer by layer, a tuple being one value (a padding pair); they build
the named `Lebesgue` and `Sobolev` layers of D-3. `plot` draws one line a
layer with a break at each jump; `plot_section`, `plot_profile`, `plot_shell`
(with `side=`) and the station markers take a layered ball, every shell on one
color scale.

---

**D-154. Noise-free problems stay at small scale** (2026-10-27, David).
Exact data are numerically brittle -- the closed form inverts `A C A*`
outright, the worst-conditioned member of the damped family -- and of no
practical interest, so they are not benchmark or example material at scale.
The closed form keeps its unit tests on small problems, where it is what
the quadratic tier is checked against; D-152's point-data benchmark
measures the quadratic tier at a working noise level (a tenth of the
data's size), its noise-to-zero sweep against the closed form demoted to an
opt-in check (`--limit`), and the slow tests assert the tier's parity, not
the limit.

**D-155. The paper-scale benchmarks are deferred; the examples build up
from what worked in v1** (2026-10-27, David). The three problems of D-152
were written and run at reduced size (`benchmarks/`), and they found three
defects that are now fixed, but the Mag-motivated problems are not ones
David has run, and tuning their regime cost more than it measured. They
stay as exploratory scripts, outside the suite, to be returned to fresh;
their slow tests are removed. The defaults D-146 left open are settled by
tutorial 10's numbers rather than a sweep: no preconditioner by default,
the surrogate Woodbury route documented as the one to reach for. What
replaces the benchmark as the proof of scale is a series of mini-apps that
grow out of problems known to work well in v1 -- tutorial 10's tomography
and the preconditioning example first -- with the pyslfp application, the
largest to date, taken up after the release.

**D-156. A nonlinear ray operator on every symmetric space** (2026-10-27,
David). Travel times through a velocity `c₀ e^{m}` along the rays traced
in it, as the first real nonlinear operator for the examples and the
tests: `ray_travel_time_operator(paths, reference_velocity, ...)` on the
symmetric-space base beside the linear path operators, on the sphere, the
torus and the plane (and the 1-D spaces for completeness, where a ray is
the interval), returning an `Operator` whose `linearize` traces once and
gives the value and the derivative together. The model is `δ ln c`, as the
v1 sphere example and the linear operators are written, so the speed stays
positive and the linearisation about `m = 0` is the constant-background
path-integral operator up to the scale `−1/c₀`, which is the first parity
test and keeps that operator as the special case it is. The derivative is
Fermat's principle: the path integral along the traced ray, so the adjoint
is the existing one. The two-point problem is solved by **bending** by
default -- the nodes of the geodesic relaxed in their transverse offsets
to make the discrete travel time stationary, a damped Newton whose
tridiagonal Hessian is recovered by three coloured finite differences of
the batched gradient, every path in one batch so that the field and its
gradient at all nodes come from one transform -- with **shooting** as the
option (`method="shooting"`: the Hamiltonian ray from a take-off angle,
Brent on the miss at the receiver, the geodesic's angle as the start),
since David has worked with it and knows its failure modes; the two are
checked against each other. Both find the ray continuously connected to
the geodesic, not provably the first arrival, which is documented;
neither converging raises `EvaluationError` (D-129). The Hessian is
Gauss–Newton from the derivative first; the true second derivative, which
needs the paraxial system, is a later addition. Gradients at scattered
points become a method of the spaces, `evaluate_gradient`, through ducc0's
spin-1 synthesis on the sphere (the scalar's coefficients scaled by
`sqrt(l(l+1))`, which measured 2e-8 against finite differences and 6 ms
for 20 000 points at degree 128, with the adjoint provided) and the
`ik`-scaled non-uniform transform on the boxes. Built on the plane first,
where everything is checked against finite differences cheaply, then the
torus and the sphere. Built 2026-10-27: `symmetric_space/rays.py`, the
spaces' `evaluate_gradient`, and `SymmetricSpace.ray_travel_time_operator`;
on the plane, torus and sphere the travel time at `m = 0` is the geodesic
length over `c₀` to rounding, the derivative matches finite differences to
`1e-5`, a traced ray is never slower than the geodesic through the same
field, and bending (48 segments) and shooting (384 steps) agree to `2e-5`,
both converging at second order to one limit; on the circle and the line
the ray is the interval. The Hessian action came the same day, on David's
asking: for bending the reduced Hessian `T_mm − T_mu T_uu⁻¹ T_um` of the
discrete travel time, its pieces the coloured differences the Newton step
already takes; for shooting the tangent linear (paraxial) system
integrated along the ray for the responses to `δm₁` and to the take-off
angle, combined so the receiver is kept, with the derivative operator
differenced along that response. Both match differences of the derivative
to `1e-3` or better and are symmetric in their two directions; the
second difference of the value matches `H[δm, δm]` to `3e-4`. With it,
`MaximumAPosteriori`'s objective carries the exact Hessian whenever the
forward operator has a second derivative -- the Gauss-Newton term plus
the operator's curvature weighted by the residual -- so a Newton optimizer
can be passed to it; a composition with a linear map keeps the second
derivative, which is what the whitened inversion of tutorial 13 uses.
Tutorial 13 is the operator's showcase: the rays, the kernels of the
adjoint, the Hessian with the ray's response, and the inversion by
gradients and by Newton against straight-ray tomography.

**D-157. Small choices recovered from the docstring review** (2026-10-28).
The review of 2026-10-26 took measurements and reasons out of docstrings
that this file did not hold. The measurements are in the appendix under
that date; the choices they settled, none of them recorded before, are:
- Iterative solvers run on a `ComponentView` and `assembled()` stores the
  components form, because on a sphere the vector algebra alone was seven
  transforms per iteration against the operator's two, and CG on a stored
  Galerkin matrix over a dense Gram matrix was three times slower than on
  stored components.
- `Optimizer` has `ftol` with no absolute floor and a precision-limited
  outcome counted as success, because with `rtol=1e-10` alone L-BFGS and
  nonlinear CG stopped at a gradient of 1e-6 to 1e-7 on a well-conditioned
  quadratic and reported failure, and steepest descent ran to its cap.
- `monotone_root` is Brent in `log t`, not bisection: half the solves per
  root on a chi-squared-like quantity.
- `Polytope.project` allows a thousand Dykstra cycles, since two hundred
  left a ball cut by a half-space 3.7e-4 off its boundary.
- The stochastic KL divergence is one trace of `g(M) = M - 1 - log M`, with
  a round-trip check on the reference solve: estimating the trace and the two
  log-determinants separately gave `KL(mu || mu)` as 369 +/- 35 at
  dimension 289, and a CG reference solve at `rtol=1e-8` on a covariance of
  condition 4e10 returned a trace of 217 against 289 without complaint.
- The chi-square quantile switches from Imhof to the moment-matched
  chi-square above 1000 positive weights or an effective rank above 100,
  where Imhof takes seconds and the match agrees to 1e-4 relative. `_imhof`
  integrates by a trapezoid rule because `quad` exhausted 800 evaluations on
  three weights and was thirty times slower.
- `plot_corner` densities are binned kernel estimates with the full kernel,
  not an axis-aligned filter, which is wrong by 8% of the peak on a
  correlated cloud; binned is three orders faster than `gaussian_kde`.
- The model-space normal operator refuses an error-free problem: `Q^-1 + A* A`
  is not the noise-free posterior, and its mean was wrong by a third of the
  true mean's norm on a random problem.
- Points are prepared once (`PreparedPoints`, `prepare_points`): the
  conversion was 40% of an application on a sphere, half on a torus and 91%
  on a bounded box at 1e5 points.
- `nthreads=1` for the NUFFT on every geometry, since finufft's default of
  every core was four to twenty times slower at every size measured.
- `Box.project_function` tapers by default: a hard cutoff puts a 5% error in
  a path integral that reaches the padding.
- The sphere's Lebesgue inner product is the grid quadrature even when both
  arguments are off-span grid products, where the two routes differ by up to
  2% and the quadrature is the more accurate.
- The direct Legendre sum is used below 200 points and dimension 256, the
  transform above, measured at `lmax` 4 to 64 across six point counts.
- An open point noted in the deleted likelihood route (why warm-started
  Newton-CG stopped a step short, and whether the stopping rules should
  change) is moot with that route (D-142); the question is kept here in case
  it recurs in `Search`.

**D-158. Private hooks that other subpackages needed are public accessors**
(2026-10-28, from the code-practice scan of 2026-10-27). The protected
extension hooks stay the override points; what other code reads goes
through a public, None-returning accessor: `LinearOperator.known_matrix(form=)`
(the matrix if it is already held), `components_action()` and
`components_adjoint_action()`, `is_identity`; `from_callables` takes
`components_action=` and `components_adjoint_action=` rather than a caller
writing private fields on its result. `resolve_rng` in `algebra.spaces`
returns a fresh `default_rng()` when given None, and there is no shared
module-level generator anywhere. `REORTHOGONALIZATION_THRESHOLD`,
`CoordinateSpace.orthonormalize_columns`, `HilbertSpace.orthogonalize_against`
(a separate method, since the range finder needs the residual and both norms
back), `CoordinateDirectSum` (its hash keyed on the class so a subclass
inherits it), `Sphere.plotting_grid` (the transform grid's colatitudes,
longitudes and values, without the wrap column and pole row, as copies),
`combine_log_normalizer`, `OrthogonalProjector.onto_kernel(inverse=)`,
`FactoredNormalOperator.prior_precision` and `error_precision` (None by
default) are all public for the same reason. `Ball` exposes `covariance`
and `precision` so it is interchangeable with `Ellipsoid` wherever a
quadratic set's shape is read, and the Ball-or-Ellipsoid chains are gone.
The capability flags (`has_membership`, `has_projection`,
`has_support_function`, `has_maximizer`, `has_level_function`,
`has_smooth_level_function`) with `level_function()` and `level` are
declared on `Subset`, so a `SublevelSet` is a level-function set like any
other and nothing probes with `getattr`; the conversions stay on
`ConvexSet`. The normal operators' property "this operator with its
structure forgotten" is `plain_operator`; it was `assembled`, which
shadowed the base method `assembled()` and broke substitutability.

**D-159. Refusals and failure reports settled by the docstring review**
(2026-10-28, from `CODE_ISSUES.md` of 2026-10-26). `from_covariance_matrix`
requires `form=` like every matrix constructor (it defaulted to Galerkin).
`from_standard_deviations` is defined only on a diagonal metric and refuses
otherwise, naming the alternatives, because its factor is not self-adjoint
on any other and no real use needs one. A missing ingredient (a subspace's
equation, an estimator's forward or error, a problem's error, a measure's
covariance) is a `ValueError` everywhere, never an `AttributeError`, which
`hasattr` would swallow. Non-convergence in the MFEM backends and an
indefinite preconditioner in MINRES raise `ConvergenceError`, as CG does;
LSQR warns when not strict, as every other iterative solver does.
`TrustRegionNewton` reports a collapsed trust region by its own message and
the true iteration count, and applies `ftol` to accepted steps.
`random_range` validates `block_size` and `rtol`, returns an empty basis for
a zero operator, and `random_diagonal` refuses `samples < 1`; the
chi-square quantile functions validate `method` before any shortcut.
`rescale_directional_variance` refuses a non-positive target. `sem1d`
requires planetmodel 1.3 (its grids carry `kind`) and says so at import;
`_sampled_with_extension` refuses `"fit"`, which is the callers' branch.
`joint_measure` and `from_direct_sum` take any samplable prior through the
generic product, a non-Gaussian prior giving a sampling-only joint measure.
`parallel._context_is_set` degrades to "no context" if joblib renames the
internals it reads.

**D-160. The rest of the clean-up of 2026-10-28.** `LocalizedPreconditioner`
is the spelling (it was `LocalisedPreconditioner`, against D-100), and an
index outside every block keeps its exact `A Q A*` diagonal entry, probed
off the factors, so a partial cover is never singular; it takes `n_jobs=`.
MFEM keep-alive references live on the pygeoinf object that owns the
solver (the operator's closure or the space), never as attributes on SWIG
objects, and the solver factories return the solver with its parts. The
point-evaluation gate on every space family shares one private signature,
`_require_point_evaluation(what, /, *, points=None, side=None, unsafe)`.
`Box.support_projection` on a Sobolev box now has the Sobolev space on
both sides (it had the Lebesgue space as its domain, a wrong adjoint that
the action-only test could not see). D-10 is enforced by a test that greps
the package, examples, tutorials and tests for a decision citation.
`tests/test_phase_four.py` is `tests/test_assorted.py`. The review
documents (`CODE_ISSUES.md`, `DOCSTRING_REVIEW.md`, `CODE_PRACTICE_SCAN.md`)
are deleted with this entry; their measurements are in the appendix under
D-157, and what was declined is recorded here: LSQR records no
iteration-0 history entry; the plotting functions keep setting
`center_line`, `contour_set` and `gridliner` on the Axes because their
two-tuple returns are unpacked by callers; `mahalanobis_squared` and
`grad_log_density` keep `NotImplementedError` for a missing precision; the
`estimate_truncation_degree` naming (v1's name means v2's
`sufficient_degree`) is open for David.

**D-161. `sem1d` splits by geometry: `interval`, `ball`, and the
`spectral_element` core** (2026-10-28, David). What the package provides is
further function spaces for use, alongside `symmetric_space` but for a
different class of geometries: functions on an interval under `dx` or
`r² dr` with a varying length scale, and functions in a ball or shell by
the same 1-D elements with harmonics. Users reach them by geometry, as they
reach `sphere` and `torus`, so the geometries are the packages:
`interval` (its `Lebesgue`, `Sobolev` and `Layered`, with `interval.radial`
carrying the `r² dr` line as its own classes, not a measure option) and
`ball` (ball and annulus, with its own `Layered`). The shared machinery
(`SpectralElementSpace`, `AxisSpace`, the mesh, the boundary and length-scale
types, the shell transform) is the public `spectral_element` package, which
a backend author may build on and a user never imports. The name
`spectral_element` was kept free by D-115 for a future 3-D mesh; that will
never be written in Python, since without a serious library behind it it
would not be fast enough, and such spaces arrive as wrapped C++ through the
backend interface instead, so the name is free to use. Plotting and tests
split the same way. All three need the planetmodel extra and are imported
on demand.

**D-162. `estimate_truncation_degree` means what it meant in v1**
(2026-10-28; David indifferent, the v1-name rule decides). v1's method walks
degrees upward until the shell energy ratio falls below a tolerance, past
any truncation; v2 had that as `sufficient_degree` and reused v1's name for
a different question, the degree holding all but a tolerance of this
space's own power, capped at its `lmax`, so a v1-style call ran silently
with another meaning. Now `estimate_truncation_degree` is the walk and the
capped question is `power_degree`; `sufficient_degree` is gone.

**D-163. The inference pattern: data in, a point, a set or a measure out,
and every class an estimator** (2026-10-28, David). An estimator maps data
to an answer about a property of the model; the kind of answer follows the
description of uncertainty. A prior measure and a noise measure give a
measure (the posterior, the Laplace Gaussian) or a point; a prior set and a
noise set give a set (the feasible property set) or a point. The decisions:
- `PointEstimator` is the base of every point-valued class, including those
  that are also operators: `LinearPointEstimator` (affine in the data) and
  the nonlinear-in-data ones with exact derivatives, which stay `Operator`s
  so a Bayesian inversion can be built on them by the chain rule.
- One minimum-norm class, `MinimumNorm`: the smallest model fitting the
  data, with the discrepancy principle as its method when errors are
  present and the misfit level as its knob. The fixed-damping class of that
  name goes (it is `LeastSquares` under another intent) and
  `DiscrepancyPrinciple` is renamed. `ConstrainedMinimumNorm` stays. The
  feasible set's `fitting_model` and this class share the misfit search and
  must agree in tests.
- A point estimator carries its error budget as two operators, `resolution`
  (`B A`) and `propagated_covariance` (`B R B*`), and the bias they leave
  out is made visible as `bias(m) = resolution(m) - m`. It never turns the
  budget into a measure: `as_measure` is deleted, because a Gaussian built
  from the propagated error alone looks like a posterior and is not one.
  The estimator is an operator and the push-forward machinery is general,
  so whoever wants that Gaussian pushes the error measure through it; the
  library does not advertise it.
  The error is `(B A - I) m + B e`, and bounding the first term takes a
  statement about `m`, which is what makes the two legitimate answers, the
  posterior under a prior measure and the certificate set under a prior
  set. Propagated error as "the uncertainty" is a view the library does not
  endorse.
- `BackusGilbert` is the point estimator: `X d` with `X` chosen to make
  `X A` close to `T` against a noise penalty, with its resolution kernel
  and propagated error, no prior bound; the name is the literature's for
  averaging-kernel estimates, and its docstring credits the original paper
  with already reaching for norm bounds and sends the reader to the
  feasible set as the completion of that thought. SOLA is this estimator
  and is not a name the library uses. The linear certificate with prior
  and noise sets (the Minkowski sum, valid for any `X` by weak duality) is
  no longer a separate estimator: it is a route inside the feasible
  property set's probes, a cheap outer bound asked for by `via=`, with the
  certified algorithms still the automatic choice.
- `MaximumAPosteriori` is a `PointEstimator` returning the mode;
  `LaplaceApproximation` is a `MeasureEstimator` built on it returning the
  Gaussian about the mode; the objective builder they share stays public
  for the sampling work.
- `Minimization` is the generic point estimator: an objective built from
  the data, minimised from a start by an optimizer, with MAP its first
  special case and nonlinear least squares (misfit potential plus Tikhonov
  damping) the second; the derivative with respect to the data by the
  implicit function theorem at the solution where the Hessian is available.
- `ConstrainedMinimization` by an augmented Lagrangian over the existing
  unconstrained optimizers, built now; Occam, the smallest model whose
  misfit meets a level, is its first special case and keeps that name. It
  is not a root search over the damping: once the forward is nonlinear the
  misfit at the damped minimizer need not be monotone in the damping, so
  the classic damping sweep is offered only as the heuristic it is.
- Nonlinear set estimation is open (a nonlinear forward makes the feasible
  set non-convex). What ships is the linearised feasible set: the feasible
  property set built on the forward's linearization about an estimate with
  the data shifted by the residual, documented as a local statement.
- The proof of the pattern is one tutorial answering one problem as a
  point, a set and a measure, showing the propagated error, the bias
  against a known truth, and the two completions.

**D-164. What the point-estimator base carries** (2026-10-28, implementing
D-163). `LinearPointEstimator` takes a `property_operator=` (`T`, the
identity by default) so that `bias(m) = resolution(m) - T m` is well typed
after a `push_forward`, which composes `T` along with `B`; a `B` landing
outside the model space without a `T` is refused. `ComposedPointEstimator`
is what `push_forward` returns for an estimator that is not affine in the
data: the inner estimator followed by `T`, a `PointEstimator` and an
`Operator` with the chain-rule derivative, flattened on a second push. The
renamed `MinimumNorm` keeps `search`, `estimator_at`, `for_data` and
`discrepancy_search` (the last two take per-call overrides of the search
settings, the estimator's own when omitted), gains `with_solver`,
`with_formalism`, `with_level`, `parameterized` and `data_reduced` like
`LeastSquares`, and accepts a problem without an error, where it is the fit
at zero damping and the search methods raise. `ConstrainedMinimumNorm`
follows it. A subclass that is both a `PointEstimator` and an `Operator`
lists the estimator base first and defines `__call__` itself, because the
abstract `__call__` otherwise shadows the operator's.

**D-165. The linear certificate as a route, the point estimator's signature,
and the linearized set in absolute coordinates** (2026-10-28, implementing
D-163). `LinearCertificate` in `geometry/feasible.py` is an `Algorithm`
named `linear_certificate` answering `support` (an outer value with no
inner, the dual cost at `λ = X* q`, so `support_bracket` is `(−∞, value)`)
and `outer_ellipsoid` (the sum of the two parts' ellipsoids bounded by
`sum_of_ellipsoids`, quadratic sets only); `terms(feasible, direction)`
gives the estimate, resolution and noise parts separately. `X` is the
caller's `operator=` against any sets with support values, or on quadratic
sets `linear_certificate_operator(A, T, covariance=, error_covariance=,
alpha=, solver=)`, built once per structure. `Algorithm.automatic` (true by
default) is false for it: `_resolve` skips it when no `via=` is given and
`algorithms` leaves it out, so it is reachable only by name or instance.
`BackusGilbertSet` is folded into that route and gone. `BackusGilbert` is
`BackusGilbert(problem, T, /, *, covariance=None, error_covariance=None,
alpha=1.0, solver=None)`, a `LinearPointEstimator` with `X = T C A* (A C A*
+ α R)⁻¹`, `C` the identity by default, `R` the problem's error covariance
(a measure's, a ball's or an ellipsoid's; dropped for exact data) and
`unresolved = T − X A`; `push_forward` composes `X`. The linearized set:
`BackusGilbertParker(problem, T, prior, *, model=None, linearization=None)`
takes a nonlinear `ForwardProblem` with one of the two (a linear problem
may take them too, with no effect on the set). Because the linearized
forward `F(m*) + F'(m*)(m − m*)` is affine in the model itself, the set is
built on `F'(m*)` with the data shifted once to `d − F(m*) + F'(m*) m*`,
and the constraint set and the property stay in their own coordinates; no
translation of the set is needed, so `ConvexSet.translate` (form-preserving
on balls and ellipsoids, a wrapper carrying the level function with its
gradient and Hessian otherwise) is unchanged and `SublevelSet` gets none.
The result is a `LinearizedFeasiblePropertySet` exposing `linearization`,
`linearization_point` (`m*`), `linearization_property` (`T m*`),
`observed_data`, and `data` as the shifted data; `with_data` and
`push_forward` keep the class and the point.

**D-166. Nonlinear estimators: the shape of `Minimization`, the
augmented Lagrangian's interface, and what the Laplace record keeps**
(2026-10-28, implementing the inference pattern). `Minimization` is
`Minimization(data_space, model_space, objective, *, optimizer, start,
mixed_derivative, solver)`, spaces first as every operator factory takes
them; it is both a `PointEstimator` and an `Operator`, with `__call__`
returning the model and `solve` the optimizer's record. Its derivative with
respect to the data is the implicit-function one, `-H^-1 M`, available when
a `mixed_derivative` is supplied and the objective carries a Hessian, and
`derivative_at(data, model)` gives it at a solution already in hand. For
`MaximumAPosteriori` and `NonlinearLeastSquares` the Hessian behind that
derivative is the exact one when the forward operator supplies its second
derivative and the Gauss-Newton one otherwise, which is what makes the MAP
derivative the linearized Kalman gain in the common case. `MaximumAPosteriori`
returns the mode; `LaplaceApproximation(estimator, *, solver)` with
`from_problem(problem, prior, *, optimizer, solver)` returns the Gaussian,
`at(model, data)` places it without searching, and `LaplaceResult` survives
as what `LaplaceApproximation.solve` returns (mode, Gaussian, optimizer
record), so the three stay one call away.
`AugmentedLagrangian(optimizer, *, equalities, inequalities, penalty,
penalty_growth, reduction, tolerance, initial_gtol, gtol_reduction, gtol,
max_iterations)` takes its constraints in the constructor so that
`minimize(functional, x0)` is kept exactly: a constrained problem whose
constraints follow the data builds one per data vector, as
`ConstrainedMinimization.method(data)` does. Its defaults are a penalty of
10 growing tenfold when the violation fails to fall by a quarter, an inner
gradient tolerance starting at 1e-2 and shrinking tenfold after each
adequate outer iteration down to `gtol` 1e-8, with the inner solves warm
started and run at an absolute tolerance (`Optimizer.with_tolerance`, added
for this, copies a method with new tolerances); it stops when the violation
is within `tolerance` (1e-6) and the final inner solve has met `gtol`. The
result reports the multipliers of both kinds, the final violation and the
penalty. `ConstrainedMinimization` and `Occam` expose the tunable half of
those knobs (`penalty`, `penalty_growth`, `reduction`, `tolerance`, `gtol`,
`max_iterations`). `Occam` offers no damping sweep: the heuristic was not
cheap to make trustworthy and the constrained solve is the method.

**D-167. What tutorial 14 found** (2026-10-28, from writing the proof of
D-163). Four small things, fixed the same day. `Occam`'s constraint is
`chi² / chi²_crit − 1 ≤ 0`, so its `tolerance` is relative to the critical
value (default `1e-4`): an absolute tolerance in misfit units was
unreachable for a forward operator evaluated iteratively, whose misfit is
noisy at its own solver's tolerance, and the penalty grew without bound. A
model with non-finite values raises `EvaluationError` at the ray tracer's
entry (D-129's protocol), where it previously reached the tracer and
crashed the process. A Gaussian whose covariance is the identity takes it
as its own factor, so a white measure samples without `covariance_factor=`
being repeated. Every optimizer's result at the iteration cap says
"gradient tolerance reached" when its `converged` flag is true, where the
flag and the message disagreed. Recorded as observations, not changed:
L-BFGS does not converge on the un-whitened ray problem (the line search
fails at a gradient norm of 200) where Newton-CG does in fifteen
iterations; `MinimumNorm.push_forward` carries no budget, the budget being
that of `for_data` at the damping chosen, which its docstring now says.

**D-168. Newton-CG takes a preconditioner; the Hessian is the operator, not
the form** (2026-10-28, David, from tutorial 14's coda). The coda had
whitened the ray problem as tutorial 13 does, which is exact for MAP and
Laplace (a preconditioner by another name) but changes Occam's question and
the ball prior from the Sobolev norm to the prior's Mahalanobis norm. That
is a fudge and goes: the tutorial works in the model space, and what it
needed was the inner solve preconditioned. `truncated_cg` and `NewtonCG`
take `preconditioner=`, an approximation to the inverse Hessian applied to
the residual, the prior covariance being the natural one for a posterior's
Hessian `P + J* R^-1 J` (un-preconditioned, Newton-CG stalled at a gradient
norm of 4 in budget on the rays where whitened it reached 6e-8; the prior
precision's spectrum on a Sobolev space spans many decades). Not combined
with a trust region, since the region would then have to be measured in the
preconditioner's norm; `TrustRegionNewton` stays unpreconditioned until
that recurrence is written. Convention stated on `Functional.hessian`: it
is the self-adjoint operator, the Riesz lift of the second derivative's
bilinear form, because the Newton step and the Laplace covariance both
need the operator; the form is what a finite difference of the gradient
checks.

**D-169. The backend claim is held to by a space with no array underneath,
in the suite and in C++** (2026-10-28, David). The claim of §1 and §6, that
nothing above a space works in components unless the space says it can, is
now tested rather than asserted. `tests/test_no_array_backend.py` runs the
checks, the operator algebra, the Krylov solvers, Gaussian measures, every
estimator, every optimizer, the randomised estimates and the convex sets
over a `HilbertSpace` whose vectors are plain Python objects with no array
behaviour at all, against numpy on the extracted lists; what a
coordinate-free space cannot be asked for must refuse by name. Mini-app 9,
`miniapps/cpp_backend/`, is the same proof with the vector type
and the sparse operators in C++ behind nanobind (a dev dependency; David
chose it over pybind11 as the successor by the same author, with lower
per-call overhead, faster compiles and the stable ABI, which is what a user
shipping a backend wants), everything of v2 staying under `pygeoinf/`
until the release distributes it, a
finite-element mass matrix as the Gram, and the inverse problem end to end
with a tally of every member the library called; nothing in the library
needed changing for it. The review found four reaches, fixed: the feasible
set keyed its result cache by the property's components for no reason (the
key is the value where the space has coordinates and the vector's identity
otherwise); its membership side, which forms matrices of the property's
dimension by design (§5), failed deep in an algorithm instead of refusing
by name (now `_require_property_coordinates` at each such probe, and both
docstrings say the support side is free and the membership side is not);
`hilbert_schmidt_norm(method="auto")` probed a diagonal with basis vectors
without checking for them (it takes the stochastic route on a space without
coordinates); and two scale probes built a ones-vector from components
where any nonzero vector does (`random` with a fixed seed). Contract
additions to §6: `random` and `white_noise` are part of what a backend
supplies. Mini-app build products and figures are ignored by git; the
mini-app's test compiles the extension into a temporary directory and skips
with the compiler's output where it cannot.
The top-level `benchmarks/` of D-155 is cut the same day (David: come
back to it later; in git at `8c53839`), so nothing of v2 sits outside
`pygeoinf/` until the release distributes it.

**D-170. The Woodbury family reads the multiplier, not the member's
scale; a diagonal factor scales a product; the point-data mini-app is
Woodbury-only** (2026-10-28, from the first full-scale run of mini-app 1).
Three things the run turned up. (1) `WoodburyPreconditioner.for_family`
inverted every member to within a factor of the damping on a Tikhonov
family (`t` in the data space, `1/t` in the model space) while being exact
on a `DampedNormalOperator` family: the bound family computed the Woodbury
form of `A C A* + t R` and then divided by the member's own `scale`, which
is one for the damped operator and `t` for the Tikhonov one in the data
space, the opposite in the model space. Conjugate gradients is blind to a
constant factor, which is how it survived every CG test; a solver or a
residual-reading outer method is not. The family now divides by the
multiplier in the model form and by one in the data form, whatever the
member says, with `C` the preconditioner's own prior covariance, so a
Tikhonov family's preconditioner is built with the identity (its member
at damping one). Along the family the data form's kernel `A* R⁻¹ A` is
read once as a stored matrix (`data_kernel`) where the factors can write
it down, and each multiplier's inner operator is a sum of stored
matrices. The regression test holds both families, both formalisms, at
three multipliers, to 1e-10. (2) A composition's known matrix multiplied
dense matrices, and a diagonal operator's known matrix is a dense square:
`A* R⁻¹ A` on 20 000 points with `R⁻¹` diagonal formed a 20 000-square
array (3.2 GB) and multiplied by it, 8.7 s for a 2 401-square result that
is a 1.1 s product. A `_known_scaling` hook (the diagonal operator, the
identity, scalings and compositions of them) lets a composition scale
rows or columns instead, and the size guard of D-33 counts a diagonal as
its length. (3) The mini-app, renamed `point_data/` from `oceans/` (David:
it is not about the oceans; the ocean mask is the geometry, point data the
problem), uses the Woodbury surrogate for everything, the Bayesian
operator and the damped families alike, and part B is reframed as the
linear algebra behind every answer rather than a way of obtaining one;
the spectral deflation is dropped from it (David: the Woodbury is good
even at this scale, the deflation is not). The part's build column now
holds the factorization, which the lazy direct solver of D-40 would
otherwise charge to the first iteration. Also fixed, from CI: the two
recorded-answer tests of `MaximumAPosteriori` asserted exact iteration
counts and values to 1e-9, which another platform's BLAS does not
reproduce (325 against 319 iterations on macOS); they now assert
convergence and the answer to the optimizer's tolerance. Steepest
descent's test tolerance on the no-array backend was 1e-11 on a quadratic,
below what a function-value line search can resolve; it is 1e-7.

**D-171. The test suite after its review: the precision floor follows
the curvature, tolerances sit above the arithmetic, the slow set runs on
a schedule, and threads are capped** (2026-10-28, David: look through and
address the review's issues; the review is `TEST_REVIEW.md` until its
remaining actions are taken). What the review found and what changed.
(1) The optimizers judged a failed line search converged only if
`||g|| <= sqrt(eps) (1 + |f|)`, a floor with no margin and the wrong
shape: the decrease a step can make is `||g||^2 / 2L`, so the floor is
`sqrt(eps (1 + |f|) L)`, and on the same quadratic the line-search methods
stop between one and four times that, which is why a verdict flipped
between platforms (macOS, 325 against 319 iterations; steepest descent at
`5.0e-8` against a floor of `4.8e-8`). The floor now reads the curvature
(the Hessian's Rayleigh quotient along the gradient where there is one,
the secant estimate from the last step otherwise, `1 + |f|` on the first
step without a Hessian) with a margin of four. Measured: asked for a
gradient of zero, L-BFGS, nonlinear CG and steepest descent stop at
`3e-10`, `3e-8` and `5e-8` on an objective of size two and at `4e-4`,
`1e-3` and `6e-4` on the same objective scaled by ten thousand, and all
six now report convergence; the Newton methods reach `1e-16`. (2) About
forty test sites asked for tolerances below that floor or below what a
Krylov recurrence can reach (`gtol` to `1e-14`, strict CG at `rtol`
`1e-13` under the default cap, first-order convex solvers at `1e-9`) and
asserted `converged`; they ask for `1e-8` to `1e-10` now, direct solvers
where an exact identity is the point, and comparisons at `1e-6` to
`1e-8`. Loosening the splitting route's tolerance exposed that its inner
value is attained only to that tolerance (CURRENT_STATE §7). (3)
Assertions about rounding-produced behaviour are gone: a MINRES
non-convergence, exact iteration counts between two arithmetic paths, a
speed contest of a private simplex routine (now one instance with a
residual bound), a warning-versus-raising distinction on a divergent run,
QP backends compared at their own solver tolerance. The C++ mini-app
draws its normals by Box-Muller on the engine's raw words rather than
`std::normal_distribution`, whose algorithm each standard library chooses.
(4) CI ran only the fast suite: `addopts` excludes `slow`, so the forty
parity and Monte Carlo checks had never run on macOS or Windows. A
scheduled weekly job (and `workflow_dispatch`) runs the slow set on all
three; the three whole-tutorial runs of a minute or more join the slow
set, fifteen sub-second tests leave it, the ray parity grids are halved,
the `gc.collect` loops are ten not two hundred, and the KDE and moment
Monte Carlos are a third their size. (5) Threads (David: "clear guidance,
and we apply it too"): the session caps the BLAS and OpenMP pools at
`PYGEOINF_TEST_THREADS` (two) through `threadpoolctl` in `conftest.py`,
whose docstring states the rule, and the test of the worker pools clears
the environment variables it then asserts on. (6) Two counting tests
counted nothing (the component route calls no inner product or `axpy`);
they count on the opaque space now and assert a positive count. A dead
import shim, a tautology, a false self-adjointness claim and three draws
rescaled by another draw's norm are fixed, and the narrative of earlier
versions and reviews is gone from every test outside `test_compat.py`,
whose subject it is. Not done, left in the review for a later pass: the
metric-rule fixtures (§D3, dense Gram matrices and a non-Euclidean data
space for the normal, convex, feasible and mixture tests), the coverage
gaps (§D2), the helper consolidation and the split of `test_assorted.py`.

**D-172. The metric rule applied to the fixtures that lacked it; what
it found was in the tests** (2026-10-28, David: proceed with the review's
open items in the order metric fixtures, guarantee tests, coverage). The
`setting`, `setup`, `pieces`, `measure` and `space` fixtures of the normal
operator, feasible set, estimator, point-estimator, mixture, convex,
probability, nonlinear, direct-sum, posterior-sampling and
column-thresholded tests now carry a dense Gram matrix on the model space
and, where there is one, a diagonal metric on the data space; twenty-five
convex tests take a parametrised space in place of a weighted one written
inline. Eighteen failures came out, all in the tests. Two fixtures built
an error measure from a Galerkin matrix of standard deviations as the
factor of a Galerkin matrix of variances, which is a factor only on a
Euclidean space (on a weighted one the noise came out the wrong size and
the minimum-norm estimator rightly could not reach its misfit); they let
`from_covariance_matrix` derive the factor. The mixture reference
inverted a covariance's components matrix as if it were the covariance of
the components. A test ellipsoid built as `G diag(s)` in the Galerkin
form was declared self-adjoint, which it is only on a diagonal metric;
the dual disagreed with the primal by 7e-3 on it, and a brute-force SLSQP
on the same problem sided with both once the ellipsoid was symmetric. A
prior calibrated to contain the truth on a weighted space did not on a
dense one, so the membership side rejected the truth's property
correctly; the helper now rescales the random shape through the point.
Three components-level claims (a localized preconditioner equal to the
diagonal one on an uncovered index, a diagonal sum's closed-form precision
and one-draw factor) are asserted on diagonal metrics only, where they
are claims. The primal's bracket on a dense metric is a few times wider
at the same tolerance, and the bound on it was calibrated for the
weighted case. Nothing in the library changed. Left on diagonal metrics
by construction: the symmetric-space and spectral-element fixtures, the
geometry file's Sobolev space, and the compatibility tests.
The coverage gaps the review listed are filled the same afternoon, two
of them guarantees the library made in words only: a primal, dual or
splitting run stopped at its budget still returns a bracket that holds
the converged answer (the primal's bracketing alone meets the default
tolerance on two balls, so the budget bites only against a tolerance
below the floor; and a `max_iterations` of one still spends the
bracketing's dozen evaluations, which is what the docstring says), and
an estimator built on a strict solver lets its `ConvergenceError` out,
the minimum-norm search wrapping it in a message that names it. Also
added: strict and downgraded reporting on MINRES, GMRES, BiCGStab and
flexible CG, trait refusals by name on Cholesky, eigen, MINRES and
flexible CG, the `unsafe` gate pinned on `credible_set` and on
`weakened_ellipsoid` at power one (the ambient ball and the weakened
ellipsoid below power one are quantiles that survive refinement and
carry no gate, which the test states), the ray operator's derivative
through `check_operator` and its two refusals (the method is read when
the rays are traced, on first application), and `misfit_search` called
directly, reaching its target and reporting a floored range exhausted.

**D-173. The review's housekeeping, and what of it was declined**
(2026-10-28, closing `TEST_REVIEW.md`, which is deleted with this entry;
the review itself is in git at bc15bc0 and after). Done: the eleven
random positive-definite builders are two in `conftest.py` (`spd_matrix`,
`spd_operator`), with the matrix-free `positive_definite` and
`nonsymmetric` beside them, and the inline dense metrics are
`make_dense_metric_space`; the nine hand-rolled patches of a class
attribute with a `try`/`finally` are `pytest.MonkeyPatch.context()`; the
shadowed figure-closing fixture, a dead `if False` branch, an unused
counting class and two unused names are gone; the point-data mini-app's
test unregisters the plain module names it had to register, since the
scripts import each other by them. Fourteen public names the suite never
mentioned have a direct test each (`from_tensor_product`,
`adjoints_are_linked`, `orthogonalize_against`, `orthonormalize_columns`,
`BlockOperator.blocks`, `PeriodicBox.with_shape`, `Sphere.domain_mask`,
`has_zero_expectation`, `GaussianMixture.with_weights`,
`ConvexSet.intersect`, `linear_certificate_operator`, the hardenings'
gates), the two names the review read as public (`excesses`, `slacks`)
being local functions, and the MFEM names left to the MFEM backend's
own Linux-only file. `test_assorted.py` is split by topic: the
preconditioners and the sparse approximation in `test_preconditioners.py`,
the measure operations (adjustments, quadratic forms, hardening,
conditioning, Imhof, quantile sampling, correlated measures) in
`test_measures.py`, and the rest into `test_solvers`, `test_operators`,
`test_geometry`, `test_observation` and `test_functional_calculus`.
Declined, with the reason: the feasible-set parity tests repeated across
`test_feasible` and `test_backus` test the algorithms and the estimator
layer on the same claims, which is what two layers are for, and a warm
start against a cold one is not a sweep against single probes; the
Monte Carlo sizes the review called larger than their tolerance needs
cost a few seconds each in a slow set of four minutes; the
geometry-independent tests parametrised over six geometries cost nothing
measurable; and a cross-file import of one ray problem builder is left
where it is. One real overlap, a stochastic re-check inside the test of
the spectral route, is trimmed.

**D-174. No route refuses or reroutes by dimension; a rank-deficient
family is walked to a floor** (2026-10-28, David: "the primal rejects
data-space formalism when data outnumber the model -- we definitely want
to remove this internal default"; found by tracing part E of the
point-data mini-app before running it). Two places keyed on the data
outnumbering the model: `Primal.support` refused the data-space
formalism outright, and `Search.feasibility` and `Search.inclusion` sent
the case to the general-tier fit, a Newton-CG minimization with
unpreconditioned inner solves on the full forward operator, which is
what ran for minutes at the mini-app's middle scale while the primal's
refusal never came into view. Both are gone. What the dimension stood
for is kept where it is true: on a family with more data than model
coefficients (or than the model less the property, restricted to a
value) the member `A C A* + t R` is singular at `t == 0`, and a walk
there, which a slack prior or an unfittable data set asks for, solves
singular members and returns either a breakdown or a bracket that holds
but says nothing (measured: an outer value of 6.9 against a limit of
-1.67). So the misfit search carries `floor=` (`1e-6` of the family's
scale `(A C A* d, d) / (R d, d)`) as the root find's `minimum`, applied
only to a rank-deficient family, and a walk that reaches it without a
fit is reported as no fit, exact to the order of the floor; a family of
full rank stays well conditioned as the multiplier vanishes and is
walked without one, which the vanishing-noise parity tests need. The
primal's existing `floor` (now `1e-6`, from `1e-4`) is the lower limit
of its walk on such a family too: the inner value is attained exactly
there and the outer value sits above the limit by a few hundred times
the floor, and the model-space formalism gives the limit where that
matters; the test that pinned the refusal now pins this. The sweep's
workers are threads (D-151), so the dense coarse surrogate is shared,
not pickled. Measured on the middle scale with 5000 points against 4225
coefficients, where the old route did not finish in five minutes: the
emptiness test in 0.5 s and a primal direction in 3.2 s, about ten
iterations per solve with the Woodbury family preconditioner. Also seen
and left: CG's curvature refusal is absolute (`(p, A p) <= 0`), which on
a member singular to rounding reads as a false claim of definiteness;
the floor keeps the routes away from such members.

**D-175. A disk geometry for the spectral-element spaces** (2026-10-28,
David: the tomography mini-app gets a two-dimensional test-bed in a disk
with an Earth-like radial structure, cheaper than the ball and with
figures that show something; built that night by an agent from a brief).
`pygeoinf.disk` mirrors `pygeoinf.ball` member for member: a radial
spectral-element basis per angular order times the circle's Fourier modes
(orthonormal on the unit circle, cosine then sine, one function at `m =
0`), in the leading eigenfunctions of `A = 1 − L² Δ` on a padded radial
mesh, with `Lebesgue`, `Sobolev` and `Layered`, point evaluation at
`(radius, angle in degrees anticlockwise from the x axis)`, projection,
the area integral, and plotting in the plane. The radial operator per
order, `u − L² (1/r)(r u')' + L² m² u / r²` against `r dr`, is not one
planetmodel's family offers (it has `r² dr` and `dr`), so
`spectral_element/_cylindrical.py` carries `CylindricalOperatorFamily`, a
subclass of planetmodel's family whose assembly uses `r dr` and whose
pencil uses `m²` for `l(l+1)`, everything else inherited, including the
treatment of the centre: condensed out at `m = 0`, dropped for `m ≥ 1`,
where a regular field goes like `r^|m|`. The cost is that the inherited
methods read planetmodel 1.3.2's private attribute names, which the tests
would catch changing. `RingTransform` does the angular analysis and
synthesis by real FFTs on `2 mmax + 1` equally spaced angles, analysis
the exact adjoint of synthesis. The Robin ends carry half the ball's
curvature term, `0.7/L ± 1/(2r)`, from the Bessel asymptotics in two
dimensions; measured on a padded disk the error is 2.3e-3 against 7.6e-3
with the ball's term, and an annulus whose inner pad ends near the
centre is the weak spot (0.02 at `r = 0.1`, `L = 0.2`), where the Bessel
ratio `I₁ / (L I₀)` would give 6.6e-3 for either geometry: left as is to
match the ball, noted as an improvement for both. Parity: eigenvalues
against `1 + L² j'²_{m,k}` to 1e-13 on the first modes, modes against
`J_m` to 1e-5, and against a 20 000-cell finite-volume solve on an
annulus to the reference's own 1e-7; the exponent of the modes at the
centre within 0.01 of `m`. Not done: a `heat_measure` on the
spectral-element base class (the disk's tests use
`spectral_measure(exp(-t (λ − 1)))`), and the mentions of the new
geometry in `spectral_element/__init__.py`, `layered.py`'s messages and
two rows of CURRENT_STATE; `Disk(0)` keeps only the constant, as the ball
at `lmax = 0` does.

**D-176. Mini-app 2, a planet's density, is built** (2026-10-28, by an
agent from the brief that settled its design: three layers on the
layered radial space, a reference by Adams-Williamson from PREM's
velocities with the CMB and ICB jumps put in, two data, the priors
played as the parts, and the boundary radii as unknowns through a
bilinear forward, `M(ρ, δ) = ∫ ρ dV − Σ_k 4π r_k² [ρ]_k δ_k` and its
moment analogue, with `δ` the outward displacement and `[ρ]` the jump
above minus below, whose derivative and second derivative pass the
checks to 5e-9). What it showed, recorded because it was not what the
brief expected: the reference is 9 % light (−90 standard deviations in
the mass), so the tight prior conflicts with the data and the feasible
set under it is empty; the loose prior shows the two-number constraint
as a slice through the space, the core and mantle means correlated at
−0.9998 with their volume-weighted sum known to 0.006 g/cm³; the mixture
rewards the reference with a 660 km jump for its added mass, not for
the jump, and the README says so; and Occam on the field-plus-radii sum
is dominated by the radii's cheapness in the norm. Two of these are open
questions in CURRENT_STATE §7, with two library findings from the same
build (the fit's leak into the padding, the multiplication operator on a
padded mesh). Units km and g/cm³, order 2, length scale 200 km, 16/24/48
modes, every part in seconds.

**D-177. MFEM leaves the core and becomes mini-app 7, in two forms**
(2026-10-28, David: the MFEM backend should be a mini-app, through PyMFEM,
with a record of how an MFEM-based function space is implemented kept
even though an operator between the library's own spaces is the better
illustration; built that night by an agent from the brief). The two
backend modules moved by `git mv` into `miniapps/mfem/` as
`matrix_space.py` and `hilbert_space.py`, class and function names
unchanged; examples 16, 27 and 29 became parts B, C and A; the two test
files became the app's slow test (114 tests, skipped by name without the
`mfem` extra, which stays in `pyproject.toml` as the app's dependency);
`pygeoinf/backends/` is gone and every reference to it fixed (tutorial
12 points at the app, the examples index and the tutorials index
likewise). What the library promises is unchanged: the backend contract
of CURRENT_STATE §6 and its proof on a space with no array (D-169). The
new form, `box_operator.ScreenedPoisson`, solves `(1 − s² Δ) u = f` with
the library's periodic Sobolev box as domain and codomain: the field is
loaded against MFEM's basis at its quadrature points, solved by MFEM's CG
on a periodic mesh, and read back at the box's grid, each step its own
operator with its adjoint written by hand (`G⁻¹ Bᵀ Qᵀ` for the load, the
same solve for the solve, `Pᵀ (V/N) T⁻¹ G` for the evaluation); the
ladder measures the whole operator's adjoint defect at 9e-12 and refuses
the plain transpose with the Gram factors left out at 0.63. The finding
worth the app: on a 64-element mesh aligned with the grid the discrete
operator commutes with the grid's translations and is a Fourier
multiplier, self-adjoint to rounding, so "the adjoint is the operator
itself" would pass the check there and be wrong on any other mesh; the
app's meshes are deliberately misaligned. An inverse problem on the box
operator ends the app, the three answers on a region among the sensors
and one in a gap. Three library findings are in CURRENT_STATE §7
(`check_traits` extracting the matrix of a definite operator,
`check_operator`'s absolute floor, `MinimumNorm.for_data` reassembling).

**D-178. Mini-app 3, ray tomography, in the disk and then the ball**
(2026-10-28, David: the two-dimensional test-bed in a disk with an
Earth-like radial structure, the ball on the same code at a toy size;
built that night by an agent from the brief, on the disk space of D-175).
The ray tracer is user code, not library code: a radially layered planet
with a linear velocity gradient in the mantle and a slow core, the ray
parameter, turning radius, distance and travel time by integrals with
the turning-point singularity removed, two-point rays by a root find on
the monotone branch, checked against the closed-form circular arcs of
the flattened gradient to 1e-11. The forward is the path integral of the
slowness perturbation along the reference ray through point evaluation
at the ray nodes, assembled ray by ray (the composition through the
point-evaluation operator at every node would hold a 3.4 GB matrix), the
adjoint the library's, defect 1e-13. Three hundred data against five
thousand coefficients, so every solve is a Cholesky on the data space
and there is no preconditioner story. The four parts show the classic
tomographic picture: a bump recovered at 0.63 of its height under good
coverage and 0.12 under poor, the core's posterior equal to its prior,
the feasible brackets a factor of six wider under poor coverage, the
truth inside every bracket; the whole app at full scale in 28 s, the
ball in 7 s. Mantle rays reach 80°, not the brief's 150°, because the
core shadow starts at 82°; reciprocal duplicate rays are dropped since
they make the data-space normal operator singular. The prior is one
diagonal operator over both layers' components rather than the layered
space's product measure, because the product lands on a direct sum
without the pointwise algebra: that and six other findings are in
CURRENT_STATE §7, the collapse of the support search's inner end on an
unseen direction a bug to fix. The third phase, bent rays in the disk
only: the ray re-traced in the perturbed slowness by a shooting
integrator (fourth-order Runge-Kutta in arc length, the two-point
problem by a secant search on the take-off angles, matching the radial
tracer at the reference to 1e-9), the derivative the path integral
along the current bent ray by Fermat's principle so the linearized
operator is the linear app's operator on the current rays, checked
against finite differences with the error falling as the step squared to
a floor of 1e-4 set by the integrator's stages sampling off the path;
MAP by Newton-CG on the Gauss-Newton Hessian in 7 iterations. What the
nonlinearity shows: the linear posterior mean fits its data to χ² 0.6 by
its own reckoning and 21.8 along its own bent rays, the mode repairs
that and differs by 6 % in the upper mantle where the rays bend most,
and the error against the truth barely moves because it is dominated by
what no ray resolves. Both this app and the waveform app had to attach
the Gauss-Newton Hessian themselves, which is the first open question
below.

**D-179. Mini-app 8, acoustic waveform inversion in from-scratch C++,
with MPI as its second phase** (2026-10-28, David: a 2-D finite-difference
wave code written for the app, forward and first- and second-order
adjoints, simple checkpointing, MPI optional with rank-0 ownership and
distributed fields as a stretch; built that night by an agent from the
brief). The model is the squared slowness on the library's periodic
Sobolev box, the forward a leapfrog scheme, second order in time and
fourth in space, with a two-level Cerjan sponge, `A(m) U = B f` affine in
`m`; the derivative, its adjoint, the second derivative and its adjoint
are discrete, derived from the scheme and bound with nanobind in one
translation unit with fused kernels that store three time steps instead
of whole adjoint fields. The brief's derivation was wrong twice and the
measurements settled it: the second adjoint's term carries a plus sign
(the brief's minus gave a pairing of +0.063 against −0.1128 for both
sides to fifteen digits), and the adjoint field at step `n` pairs with
the step that contains `u^{n+1}`; and the brief's single-level taper
damps the field but not its rate and sends back 40 % of the data's rms
where Cerjan's two-level form sends back 5 %, which no adjoint check
could see and only a padded reference grid found. Every check passes to
rounding: dot tests at 1e-15, the ladder's derivative checks at 1e-9
with central differences falling a hundredfold per decade. MAP by
Newton-CG on the Gauss-Newton Hessian in 7 iterations, then two exact
Newton steps that show quadratic convergence at quick scale and an
indefinite Hessian from the background; Laplace with the library's
spectral preconditioner at rank 100 (the prior alone leaves the solves at
the cap); Occam at 95 % in 5 outer iterations. Phase 2: the grid split in
columns across MPI ranks with a two-column halo, rank 0 running the
library and broadcasting commands, no mpi4py since the module does its
own communication, the serial build the one-block case and bitwise
identical to the serial module; the distributed operator matches the
serial one bit for bit on up to six ranks and the MAP mode is reproduced
to the last bit; no speedup at this size, the point being the ownership
pattern. Built here against PETSc's MPICH since the system Open MPI has
no headers; the test skips without `MPICXX`. Checkpointing is not
implemented; the snapshot stride is the hook. Findings in CURRENT_STATE
§7: NaN passes the checks, the checks' floor at one, the solve step's
iterate as a component array under a view, the missing Gauss-Newton
Hessian, Newton-CG discarding the truncated CG's stop reason.

**D-180. Three fixes from the mini-app builds** (2026-10-28, late; David:
do the quick ones tonight, the rest in the morning). (1) A check that
meets a NaN fails: `testing._record`, through which every defect passes,
refuses a defect that is not a finite number, where before the maximum
dropped the NaN and the comparison was false for it, so a NaN-valued
operator passed `check_operator` with both defects reported as zero. (2)
The primal's inner end no longer collapses on a direction the data
cannot see. The attaining model is built along the ray `u₂ + x u₁`, and
the ray was thrown away whenever `u₂` violated the misfit by more than
1e-12, which it does by the search's own tolerance (3e-4 at the default
on the three-dimensional reproduction); the fallback then walked from
the fitting model toward the optimum and stopped at once, giving the
fitting model's pairing, zero in the unseen direction. Now a violation
within 1e-6, the accuracy of the root find that places the fitting model
on the noise ball, counts as on the boundary, and when `u₂` is further
out than that the optimum is pulled back into the set along the segment
from the fitting model and the ray extended from there, which recovers
the whole extent where the ray never changes the misfit; the better of
that and the old pull-back is kept. The nine reproductions now agree
with the closed form on both ends. (3) `gauss_newton=True` on
`MaximumAPosteriori`, `NonlinearLeastSquares` and `Occam` makes the
objective (Occam's misfit constraint) carry the Gauss-Newton Hessian
whether or not the forward operator has a second derivative, the
curvature term dropped by choice and said so, which is what the
principle behind `gauss_newton_hessian` asked for: not a Hessian that
quietly omits a term, but one asked for by name. The default is
unchanged (exact when available, none otherwise), the estimators'
`hessian` methods use the objective's whenever it carries one, and the
derivative with respect to the data and the Laplace approximation follow
it. The tomography and waveform apps drop the subclasses that attached
it by hand.

**D-181. The mechanical findings from the mini-app builds, fixed**
(2026-10-29, David: address the remaining issues, the mechanical ones
first). `check_traits` verifies full rank only where the matrix is
already stored (`_known_matrix`), since extracting it from an operator
that is applied costs as many applications as the dimension and every
covariance claiming definiteness paid it. `MinimumNorm.estimator_at`,
and so `for_data`, hands `LeastSquares` the family's own member through
`TikhonovFamily.member_solver(damping)`, a solver that answers with the
family's solve, so a solver that assembled or factored the family once is
not asked to do it again; the test counts applications. `diagonals()` of
a composition with a diagonal factor scales rows or columns as
`_known_matrix` already did. `DirectSum.__new__` takes no arguments when
unpickling creates the instance bare, so a plain sum pickles and
process-parallel draws over a measure on one work. `ambient_ball`'s
exact route reads the covariance's known scaling of components (a
diagonal, a scaled identity, their compositions, a block diagonal of them
on a direct sum, where `BlockDiagonalLinearOperator` now reports the
concatenation) rather than the bare diagonal class alone; a measure
rescaled point by point is honestly not diagonal and still samples.
`GaussianMeasure.from_product(..., domain=)` puts the product on a given
direct sum whose summands are the factors' domains, and
`Layered.sobolev_measure` passes the layered space itself, so the prior,
and every posterior built from it, keeps the pointwise algebra;
`sample_pointwise_std` takes the variance before looking up the square
root, so a domain without one gets the variance's own `TypeError`. The
support search catches `LinAlgError` from a factorization at the floor
beside `ConvergenceError`. `NewtonCG` and `TrustRegionNewton` record why
each inner solve stopped in `OptimizationResult.inner_stops`, one entry
per iteration, so a run steered by negative-curvature exits can be seen
from its result. An iterative solve run on the component views hands its
callback the iterate as a vector of the operator's domain, through a
shallow copy of the solver with a lifting callback, so that a solver
shared between threads is not altered. The tomography app's hand-built
inner end for the core's direction is gone, D-180 having made it
unnecessary. Every item has a test.

**D-182. The padding of a spectral-element mesh: the integral reads the
domain, the fit warns when it leaks** (2026-10-29, from the planet-density
mini-app's two findings; measured before deciding). The integral
functional of every spectral-element geometry (`interval`, `interval.radial`,
`ball`, `disk`, and `Layered` as their sum) evaluates a grid vector by the
domain's own quadrature of its values (the radial weights times the scale
squared on an axis, times the Gauss-Legendre shell quadrature in a ball,
times the ring weight in a disk), and reads as before through the
derivative components on a vector in the span, where the two agree to
roundoff. A field not in the span, a pointwise product above all, was
integrated through its projection onto the modes over the padded mesh,
which read the padding: `+8e-5` for a uniform field and `-62 %` for a
6 % step on a three-layer radial space; now exact, so
`integral_functional() @ multiplication_operator(f)` is the integral of
`f x` over the domain. The fit over the domain (`project_function`'s
default `extension="fit"`) stays as it was, and warns: the fit is
ill-posed by construction, the kept modes restricted to the domain being
nearly dependent, and for a function with a jump or a kink the smallest
solution matching the domain values carries values thousands of times
larger on the padding (3e4 against 4 for a step on 48 radial modes).
Three remedies were measured and declined: a coarser rank cut-off
(1e-4 bounds the padding but costs a smooth fit five digits, 3e-10 to
1e-5); keeping the fewest modes within a tolerance of the best residual
(every ill-conditioned direction still lowers the domain residual, so
nothing is dropped); and a weak penalty pulling the padding toward the
constant continuation (1e-6 bounds the step's padding at 7 and costs a
smooth fit four digits, and ruins a field that is in the span, whose own
padding is not that continuation: whole-mesh error 0.5). No prior on the
padding is right for both a field in the span and a step, so the fit
keeps exact recovery of the former and `_checked_fit` warns when the
padding exceeds ten times the largest domain value, naming the remedy:
sample with `extension="constant"` or `"odd"`, whose domain values are
exact and whose padding is bounded. *The integral's part of this decision
was reversed the same day by D-183: the functional reads a grid array
through its components, as every operator must.*

**D-183. Matrix-free point evaluation on the spectral-element spaces,
and the rule that an operator reads a grid array through its components**
(2026-10-29, David: build it for the spectral-element spaces, where point
evaluation is a local operation, the linear forms sparse in the nodal
basis, wrapped by the Riesz map; built by an agent from a brief).
`point_evaluation_operator(points, form=)` on `Interval`, `Radial`,
`Ball`, `Disk` and their `Layered` spaces takes `"dense"` (the assembled
basis matrix, as before), `"local"` (the new route) or `"auto"`, which is
local when the dense matrix would exceed 256 MB (`_DENSE_EVALUATION_LIMIT`,
a memory budget: the two cost about the same per application and differ
in memory). The local route, in `spectral_element/_points.py`: each
point's Lagrange weights on its element from the mesh, so the radial
part is sparse with `ngll` entries per point; in a ball or a disk the
coefficient functions of the components contracted with those weights
and then with the harmonics or the circular modes at the point's angles,
grouped by element and chunked at 4e6 entries of work space, the
harmonics kept when their table fits the same 256 MB and recomputed per
chunk otherwise (a vectorized recurrence fifteen times faster than
planetmodel's `real_harmonics`, within 1e-13 of it); the adjoint the
transpose of the same contraction followed by the Riesz map, which in the
span of the kept modes is the modal projection with the diagonal Gram,
no system to solve. Against the dense form: 1e-15 forward and adjoint on
eleven geometries, the ladder's worst defect 1e-14; on the tomography
scale (dim 4223, 83 000 points) 0.05 s per application against the dense
form's 0.07 s and 415 MB against 2.8 GB. The chunks are not threaded:
threading pays only with the harmonics recomputed and BLAS capped, which
an operator's action must not do to the process. The rule the build
settled: the forward reads a vector through `to_components`, the field
with the components of the array, exactly as the dense form does. With
raw grid values the adjoint holds only in the span, and the pushforward
covariance of a pointwise-rescaled prior came out 4 % asymmetric;
through the components the local form equals the dense one on every
grid array, products included, and `local @ multiplication_operator(u)`
passes the ladder. The same test refutes the integral half of D-182:
with the integral functional reading raw grid values by quadrature,
`integral @ multiplication_operator(u)` fails the adjoint axiom on a
radial space and on a ball, and through the components it passes to
1e-14; so the integral functionals read through the components again,
their docstrings say so, and the exact integral of a grid product is the
caller's quadrature, which is what the planet-density app does. Two
surprises recorded: at the centre of a mesh that starts there the mass
vanishes and the degree-zero modes carry the condensed value, so
`Phi^T v` is `modes().T @ v` and not an analysis of `M^-1 v` (39 % wrong
at `r = 0.01`); `basis_matrix` stays on `real_harmonics` so that the
dense oracle is independent of the code it checks. Not done:
`jump_operator` still assembles a dense matrix.

**D-184. The mini-apps revised: answers in the order point, measure,
set; figures per method; the ball tomography three-dimensional**
(2026-10-29, David: "I would order them: point, measure, set in order of
treatment, this being roughly in order of algorithmic complexity"; the
set estimators' priors, the Gaussian hardened to its 95 % ball, match the
Bayesian prior in coverage level only, not as models, so "it would be
better not to compare the methods head to head in summary figures;
rather, each method gets its own set of figures", with prior against
posterior the comparison inside the measure and the set parts, and
titles on every figure "as examples produce several figures and it's
tricky to keep track"; and the ball part of the tomography app, which
ran its rays in one plane and drew sections through it so that it read
like the disk again, must be genuinely three-dimensional with
intersecting sections; done by an agent from a brief). The parts moved
by letter, not by file: point_data C point, D Bayesian, E set, F the
extension; tomography B point, C Bayesian, D set, E the ball, F bent
rays; waveform C the MAP, D Occam (a point estimate, no longer "D,
continued"), E Laplace, F the ranks; the three answers inside the MFEM
app's inverse problem likewise, where the point answer going first cut
its solves from 2434 to 1234. What feeds what was straightened: a set
part takes nothing from the point or the measure part any more. Every
figure is titled "App, part X: what it shows", enforced by each app's
`save`; the point figures show the estimate with two propagated standard
deviations, the value on exact data and the truth; the measure figures
set the prior's standard deviation and marginals against the
posterior's; the set figures set, per property, the bound the prior's
ball alone admits (its support function) against the certified bracket,
which corrected one README claim (Africa's bracket of ±46 is not the
whole range the ball allows; the ball alone admits ±69); the combined
comparison figures are gone. The ball tomography: twelve sources spread
over the sphere, thirty receivers on a continent and eight on islands,
174 rays each traced radially over its great-circle distance with its
nodes along that great circle, a layered ball of degree 8 in the core and
14 in the mantle, a truth of four mantle bumps and one in the core; the
figures a map of the network with the section planes, the truth on a
shell with the rays beneath it, and three sections (the equator and two
meridians, through the continent and away from it) each with the rays
within 6° of its plane; the answers show the coverage's dependence on
the network (pointwise standard deviation 0.12 under the continent's box
against 0.59 at its antipode, the prior's 1.0 in the core). Three
library findings in CURRENT_STATE §7: `plot_shell` draws on the ball's
own angular grid, coarse at low degree, and refuses a shading keyword
the sphere renderer sets itself; there is no public way to draw a path
on a section; a sampled standard-deviation field read at a point is the
projection of a grid product, which smooths it (the rule of D-183 at
work), so a pointwise standard deviation at a point should be sampled
there. The planet-density app was left to its rebuild (D-185).

**D-185. The planet-density mini-app rebuilt simpler, on PREM's own
layering** (2026-10-29, David: the results were too complex to read;
drop the Adams-Williamson reference and the mixture, generate the data
from PREM with suitable errors, a tight prior about PREM and a
deliberately wide bound, the boundary radii for the weakly nonlinear
part, normal-mode data through specnm later; then, on seeing the first
rebuild, the truth as PREM plus a seeded draw from the tight prior so
that the point answer moves, PREM's own discontinuities as the layers
since nothing stops it, and "basis functions" rather than "modes" since
the word must stay free for normal modes; built by an agent). Eleven
layers at PREM's boundaries with the ocean and the two crustal layers
merged (each under 20 km), 104 basis functions at the full scale, 200 km
length scale except in the merged crust, which gets a quarter of its
thickness; PREM fitted in every layer where it is smooth (within 4e-4,
the transition zone within 8e-5 against 0.2 before) and sampled with the
constant extension in the crust, whose three jumps three basis functions
cannot follow (0.84 there, its mass 0.1 % short); the reference's mass
and moment agree with PREM's to 1e-5. The truth is PREM plus a draw from
the tight prior (0.1 g/cm3 pointwise), the data its mass and moment with
0.1 % errors, rejecting PREM at chi-squared 266 and fitting the truth at
3; the seed is 2027 because at 2026 the draw left PREM fitting the data.
The parts: A the setup, B the tight prior and C the wide bound (a ball
whose radius is the 95 % radius of the Gaussian with pointwise std 3),
each answered point (the minimum-norm change from PREM in the prior's
Mahalanobis norm, through the parameterized problem, with its error
budget), measure (prior against posterior) and set (the ball alone
against the data's bracket), with their own titled figures; D the ICB
and CMB radii as unknowns in units of 50 km, MAP by Newton on the
Gauss-Newton Hessian, Laplace, Occam. What it shows now: the point
answer moves most of the way on the mantle mean and its error is almost
all bias, which its budget states; the wide prior leaves a thin ellipse
along the line of constant mass; the data prefer the tight prior by 5.5
nats; MAP recovers the CMB shift within two Laplace standard deviations
and explains the light Earth partly by moving it. Occam in D remains the
norm's answer and not the Earth's: it moves the ICB out by 1500 km
because the model space's Sobolev norm, with a 95 % ball of radius 3e5,
makes a unit of displacement thirty thousand times cheaper than the
density change with the same effect on the mass; the 50 km unit does not
cure that. David: this is a point about norms on direct sums, and about
Occam itself, since "simple" is norm dependent and nothing fixes the
norm; so it is the point of the part, shown rather than told. Part D runs
Occam twice from the same reference to the same level, in the model
space's own norm and in the prior's Mahalanobis norm through the
parameterized problem (as the point answers of B and C already measure
structure): the first leaves the density as PREM's and moves the ICB out
by 1547 km and the CMB in by 132 km, the second makes the mantle 0.038
lighter and moves the CMB 24 km in, near the MAP model; each is the
smaller change in its own norm (31 against 5e4 in the model space's,
31 against 0.98 in the prior's) and both fit the data equally, and the
README's paragraph on Occam is written around that.

**D-186. A mixture-prior part in the tomography app** (2026-10-29,
David: the mixture functionality, dropped from the planet-density app,
wants a home where a discrete hypothesis sits inside a continuous
problem; "just mixture on the linear problem; it makes the
functionality's existence clear and that is enough"; built by agents
from a brief). Part E of the tomography app, on the linear disk, after
the set answer and before the ball (now F) and the bent rays (now G):
a plume at one of five candidate locations, each component of the prior
the Gaussian whose mean is the reference plus a bump at its candidate
with the app's prior covariance shared by all, equal prior weights; the
truth a bump at the third candidate on half a draw from the prior, the
data with the app's noise. `LinearGaussianMixtureInversion` updates the
components and rescores the weights from their evidences: on the app's
data the true candidate carries 0.92 against the prior's 0.2, with the
two candidates under sparse coverage confusable at a quarter of each
other's weight however small the noise; across a sweep of the noise the
weights go from equal at a hundred times the app's noise to one
candidate as it falls, and stop sharpening below the app's level, where
the shared prior covariance dominates the noise. Figures per method with
the mixture drawn as its density; the README says what the weights are
(evidence ratios) and why a single Gaussian posterior could not say
this. The bent-ray version, a mixture of Laplace approximations, is
deliberately not built. Found on the way: the bent-ray part seeded its
derivative-check draws from the truth's generator (fixed, its numbers
refreshed); its shooting depth floor is a safety margin above the core
and 0.57 is safe at both scales with a margin of 0.009; the mixture's
marginals are sampled by the plotting helpers (20 000 draws a panel)
though they are closed form, and the mixture inversion factorizes the
same normal operator once per component when the covariance is shared,
both in CURRENT_STATE §7.

**D-187. The waveform app's absorbing boundary: a bigger box, a
retuned sponge, and results that are worse because they are honest**
(2026-10-29, David, on the snapshots: the absorbing boundaries are not
right; use a bigger box as a helper and show the physical domain in every
plot but the snapshot one; built by agents from a brief). The reflection
is measured against a run on a grid padded by 200 cells on every
absorbing side (300 gives the same reference to the last bit): before,
a 20-cell Cerjan sponge at rate 0.015 inside a grid the size of the
section sent back 4 % of the data's rms at the quick scale and 5 % at
the full, with an interior peak of 8 % of the direct wave; the bottom
reflection crossed the whole section at 0.87 s. Now the 2 by 0.8 km
section sits inside a 248 by 104 grid (148 by 64 at quick) with the
sponge beyond it, 24 cells at rate 0.008, the widest whose four stored
fields fit the 500 MB budget (40 cells would reach 0.2 % at 645 MB);
the reflection is 0.6 % of the data's rms at full and 0.7 % at quick,
the gather shows no reflected arrival, a faint arc remains in the
snapshot; the background velocity levels off below the section so that
draws from the prior cannot break the stability limit in the deep
sponge. Part C at full was cut to one polish iteration and two exact
iterations to stay under two minutes, which loses the quadratic
convergence demonstration at full (kept at quick); the inner solves'
stop reasons are now printed and confirm that from the background the
first two stop on negative curvature and at the Gauss-Newton mode none
does. The finding worth the work: the old sponge's echo was about the
noise's size, and since the operator models the sponge the inversion
was reading the model from the echo; a reflective sponge placed beyond
the section recovers the old, better numbers (Gauss-Newton error 0.45
against 0.67). So the full-scale answers are now worse (MAP error over
the old interior 0.51 against 0.27, Laplace ratios 0.10 to 0.36 against
0.05 to 0.10, three of four truths 1.8 to 1.9 posterior standard
deviations from the Laplace mean), and the README says why. The test's
MAP bound is over the whole section, which is the physical domain, with
the reflection checked at quick (data below 1 %, section below 3 %).
The MPI part was rerun against PETSc's MPICH: bit for bit on three and
six ranks at full; the serial second adjoint is now slower than three
ranks because the memory budget runs its shots one at a time.

**D-188. Each mini-app carries an implementation walkthrough, and its
README comments on the results shown and nothing else** (2026-10-29,
David: the apps teach people to use the library, the step from the
tutorials to their own applications, so each gets a separate document on
the library and library-adjacent aspects it uses, not the forward
code's internals, while the README stays brief; and "we don't need to
keep track of narrative during development, the README should comment
on the results shown and that is all"; written by an agent from one
brief so the five read alike). `IMPLEMENTATION.md` beside each README,
about 300 lines, in six parts: the spaces, the forward code wrapped as
an operator and what the checks verified, the priors and the error
measures, the solvers and preconditioners, the estimators in the order
point, measure, set, and what a reader would change to use the pattern
on their own problem; tutorials cited by number, code quoted only in
short fragments from the app. The READMEs lost every before-and-after
sentence and comparison with an earlier state of the app (the waveform
app's old sponge and its echo, the point-data app's first attempt at
the full scale) and keep the measurements against references and the
seed-sensitivity remarks, which are comments on the results. The ray
nodes are no longer drawn in the tomography app's setup figure: they are
quadrature points of the path integral, chosen dense enough relative to
the length scale that halving them changes nothing, which the app
measures. Signals from writing the walkthroughs, in CURRENT_STATE §7:
rows of derivative components into a Euclidean data space are declared
`form="components"` in one app and `form="galerkin"` in two others, both
right because the codomain's Gram is the identity and both needing
explanation; the point-data app runs no checks and the planet-density
app checks its linear rows only in its test, so the pattern is shown
rather than exercised; a layered space has no constructor for the
identity covariance scaled to a pointwise standard deviation, which the
tomography app builds by hand; an optimizer's result could carry the
gradient norms per iteration that the waveform app wraps the objective
to record.

**D-189. One layout and one command line for every mini-app**
(2026-10-29, David: "each mini-app has a folder, results or figures,
where the outputs are written to, to keep things tidy and separate;
similarly src for the module codes and cpp for any C++ code"; and "build
in a host of command line arguments through which interesting physical
parameters can be modified ... more granular control" beyond quick and
full; done by an agent from a brief, every move by `git mv`). Each app:
the part scripts, `run_all.py`, `README.md` and `IMPLEMENTATION.md` at
the top, where a reader runs them; `src/` a package holding the modules,
imported relatively among themselves and as `src.x` by the scripts;
`cpp/` for C++ sources and build files (the waveform kernel, whose
built modules land in `src/`); `results/` for everything a run writes,
with `results/quick/` beneath it, ignored by one pattern. The command
line: `--quick` or `--full` picks a preset `Scale`, and every field of
the scale is an option named after it, typed from its annotation, with
the preset's value as the default in `--help` and its help text from the
dataclass's docstring, applied with `dataclasses.replace`; one `parse`
in each `src/common.py` built from the dataclass, with a stated set of
fixed fields; the seed is a field of the scale; what was a module
constant and is size or physics moved into the scale, what is fixed
geometry, the truth or numerics stayed. The waveform app's saved mode
carries a digest of the scale in its file name when an option departs
from the preset, so a later part never reads a mode computed for
another problem. The tests load `src.*` and the scripts by name and
clear them afterwards. The quick runs print the same numbers before and
after the move.

**D-190. No identity covariance on a function space: the library
refuses a measure whose covariance is not trace class on the space**
(2026-10-29, David: "we cannot have identity covariances ever on a
function space; that is a cardinal sin for this library", one the
removal of `from_standard_deviation` for such spaces was meant to rule
out; and the problem is posed on a space appropriate to the forward
model, with the prior a measure of extra regularity on it, never
re-described as everything relative to `L^2` in the manner of the
Gaussian-process and SPDE literature). A Gaussian measure on a function
space has draws in the space exactly when its covariance is trace class
there: for a Sobolev covariance `(1 + L^2 lambda)^-order` in the
space's own metric that is `order > d/2`, with `d` the dimension the
eigenvalues grow in, whatever the space's own order (the draws then
have `order - d/2` derivatives more than the space asks); at order zero
it is the identity, white noise in the space's metric, and the
truncation only hides that the draws are not functions of the space.
So `sobolev_measure` on every geometry refuses `order <= d/2`, where `d`
is the spatial dimension except for a space of functions of one
coordinate in a higher-dimensional domain, a radial profile under
`r^2 dr`, whose eigenvalues grow as on a line and which says so through
`_eigenvalue_dimension = 1`; `Layered` passes the refusal through; and
`GaussianMeasure.from_standard_deviation` refuses a `HilbertModule`
domain, leaving it to Euclidean spaces, where it is the usual data
noise. Both take `unsafe=True` for a test or a tutorial that means it:
tutorial 5's table of the pointwise standard deviation under refinement
at orders 0.3 and 1.5 is the lesson itself and keeps the order-0.3
column through the escape; tutorial 8 shows the refusal where it read
Tikhonov's minimizer as the mean under the prior `I/t`, an algebraic
coincidence of finite dimensions and not a hidden prior; example 10 puts
a Sobolev measure on its interval; example 31 and tutorial 11 move
their ball priors from order 1.5 to 2; the ball and disk tests likewise,
and a test that needs order 1 for one banded solve says `unsafe`. The
mini-apps that committed the sin (point data on `H^4` with an order-zero
measure, chosen last night for the solvers' convenience; the tomography
app's hand-built diagonal identities; the MFEM box and the waveform box
at order zero) are re-posed on `H^2` with Sobolev measures of order 2,
their set answers hardened to the prior's credible ellipsoid rather
than the ambient norm ball, which coincides with it only for the
identity (D-191 records the outcome).

**D-191. The mini-apps re-posed on `H^2` with Sobolev measures of order
two, and their set answers on the prior's weakened ellipsoid** (2026-10-29,
done by an agent from the brief D-190 left; the hardening below departs
from that brief and is David's to confirm). Point data: `H^4` with an
order-zero measure became `H^2` with an order-2 measure, the same draws, so
parts C, D and F are unchanged to 1e-13; the point answers measure size in
the prior's norm through `problem.parameterized(prior.covariance_factor)`,
least squares at damping one being the posterior mean, and a model's `w` is
`L* Q^-1 m` (or the diagonal factor's inverse). Tomography: the hand-built
identity on each layer became `Layered.sobolev_measure(2, amplitude=...)`,
calibrated by an amplitude per layer so that it stays diagonal with a
precision (pointwise calibration gives `S C S*`, with neither) and
re-seated as one `DiagonalLinearOperator` on the layered space, because the
product measure's operators act on the bare direct sum and a posterior
built from it has no pointwise product; the bent-ray part's prior went from
3% to 5% for its smooth draws to bend the rays by about the noise. The MFEM
box and the waveform box went from order 0 to 2; the waveform's Occam lost
its prior-covariance preconditioner (its norm is the space's, whose Hessian
is the identity) and its Laplace solves are preconditioned by the leading
modes of `L* N L`, the normal operator seen through the prior's factor, 17
iterations a solve where a deflation of `N` itself found the prior
precision's eigenvalues and stalled. Planet density was already posed this
way; its set answers moved too, and its noise, two errors of different
sizes, went to its own credible ellipse. **The hardening.** D-190 asked for
`prior.credible_set`; D-131 refuses it on a function space, rightly: its
radius is a chi-squared quantile on as many degrees of freedom as there are
coefficients, and on the point-data sphere the bound it alone puts on a cap
average is ±35 at degree 64 and ±69 at 128 (the old app's ball, the
identity measure's credible set, admitted the same ±69). Every app instead
hardens to `prior.weakened_ellipsoid(level=0.95, power=p)`, the covariance
to a power with a radius carrying 95%, whose limit exists when
`sum lambda^(1-p)` converges, `p < 1 - d / 2s`: the scale's `set_power` is
half of that, 1/4 on the sphere, the disks and the boxes, 1/8 in the ball,
3/8 on the radial planet (bound ±12.0 at degree 64, ±12.9 at 128). Not the
ambient ball. Costs: the point-data set answer at full is about four and a
half minutes serially (six directions, 37 to 58 s each, the ellipsoid's
family preconditioned less well by the coarse surrogate), measured direction
by direction; the planet set answers half a minute each through the Lanczos
calculus on a dense covariance. Findings, in CURRENT_STATE §7.

**D-192. The checks compare on a relative scale with `atol` as the only
absolute floor** (2026-10-29, David's call on the signal from the MFEM and
waveform apps). `_assert_close` and the derivative, gradient and
second-derivative checks in `testing.py` divided every residual by
`max(|a|, |b|, 1)`, so on an operator whose values are small the
"relative" defect was absolute and a wrong adjoint passed: `x -> 1e-10 x`
against `y -> -1e-10 y` had a worst defect of 7e-10. The floor at one is
gone; the scale is the larger of the two sizes, `atol` the floor, and the
three finite-difference checks take an `atol` (default zero) for a
derivative the difference cannot resolve because the operator's values
dwarf it. A scalar pairing is judged against its own magnitude, not the
Cauchy-Schwarz bound `|Ax||y|` that was tried first: on random probes the
pairing is smaller than that bound by about the root of the dimension, so
the bound dilutes a wrong adjoint by the same factor (the MFEM ladder's
matrix-form operator, whose pairings disagree sevenfold, recorded 3e-4
against a scale of 2 065). The bound does set the rounding error of the
pairing, `eps |Ax||y|` times a factor for the accumulation, whatever the
pairing's own size, and a pairing can be structurally small: the
coefficient operator of a Sobolev line sees three low modes of a draw
whose norm lives in the high ones, so its pairings are `1e-6` of the bound
and agree to `1e-15` of it, which is `6e-9` of themselves. So the
comparison allows `1e3 eps` times the bound on top of `atol + rtol scale`
(`_ROUNDING`, with the measured ratios at most `2e-14`), while the recorded
defect stays the residual over the pairing. The sign test of positive
semidefiniteness keeps `|Ax||x|` alone, bounding roundoff rather than
agreement. A non-positive scale records the raw residual, exact for zero
vectors and the route by which a NaN still reaches the finite-number
guard.

**D-193. The C++ backend app re-posed on `H^1`, and the ellipsoid's power
counts the covariance's order against the space's own metric** (2026-10-29,
David's prompt: the app "has precisely the L^2 issue, the point
observations aren't defined on that space"; built by an agent from a
brief). `CppSpace` had the mass matrix as its inner product, so point
evaluation was unbounded on the space the problem was posed on, and its set
answer hardened with the ambient ball. Now the inner product is
`x^T (M + l^2 K) y`, the Riesz map the banded Cholesky of that matrix in
C++, white noise its transposed factor solve, and the Matérn covariance
`sigma^2 (I + l^2 Delta)^-2`, of order 2 against `L^2`, is a measure on
`H^1` with a factor, a precision and a precision factor, all four checked,
`sigma` calibrated to a pointwise standard deviation of one mid-interval.
Parts in the standard order with their own figures: the ladder (why `H^1`:
the norm of the evaluation functional converges, 2.4999 to 2.5000 from 250
to 4000 elements, where the `L^2` norm grows as the root of the count), the
point answer in the prior's norm, the posterior, the set with the weakened
ellipsoid; the standard layout (`src/`, `cpp/`, `results/`), every scale
field a command-line option, `README.md` and `IMPLEMENTATION.md`. **The
power.** D-191's rule, half of `1 - d / 2s`, takes `s` as the covariance's
order against the metric of the space it is a measure on, since that is
the spectrum whose `(1 - p)`-th powers the radius sums; the apps of D-191
use Sobolev measures whose order is stated that way, but the Matérn
covariance here is of order 1 against `H^1`, so the limit is `1/2` and the
power `1/4`, not `3/8`: the squared radius at 50, 100, 200 elements is
5.86, 6.18, 6.40 at `1/4` (converging like the root of the mesh size),
7.81, 8.50, 9.07 at `3/8`, and 10.8, 12.4, 13.9 at `1/2`, where it has no
limit. Findings in CURRENT_STATE §7: `from_standard_deviation` refuses only
a `HilbertModule`, not every function space, so it accepted this one;
`weakened_ellipsoid` on a space without coordinates is refused by its
default route and, at a truncated rank, silently short.

**D-194. The convenience hardening is the ambient ball, in the space's norm
or in the norm a bounded operator defines; the weakened ellipsoid is gone**
(2026-10-29, David). The set estimators take a constraint set, and a
Gaussian prior does not single one out: its draws lie outside the
Cameron-Martin space almost surely, so its credible set grows without bound
under refinement (D-131 refuses it on a function space), and any set is a
modeling choice, as the norm is for Occam (D-185). D-191's weakened
ellipsoid converged, at powers below `1 - d / 2s`, but its power was a
convention the answers moved with (squared radius 6.4 at `1/4` against 9.1
at `3/8` in the C++ app), its convergence slowed toward the limit, it
needed a gate the library did not have, and every application went through
Lanczos. David's ruling: this is a convenience, a means of generating a
constraint set, and within an application the physics and other
considerations decide the set, which the library leaves people free to
build; so the convenience is the simple one. `ambient_ball(level=)` is the
ball of the space's norm with the radius at which a draw lies inside with
the given probability: the classical Backus-Gilbert-Parker norm bound,
the prior calibrating the radius, no knob but the level, convergent for
any trace-class prior since a draw's squared norm has a finite
expectation. The same method takes `operator=B`, bounded, self-adjoint
and positive definite on the space, and then gives the ball of the inner
product `B` defines as an `Ellipsoid` (with `inverse=` for its covariance
and so its support function): a constraint set of another shape, the `L^2`
ball of a Sobolev space, a bound weighting one region or one summand of a
direct sum, convergent because `B` is bounded and the covariance trace
class. Its weights are the eigenvalues of `B C`: the product of the two
spectra on the diagonal route, those of `C^(1/2) B C^(1/2)` from the
matrices on the dense route, of `L* B L` through the factor on the
randomized route, and the empirical quantile of `(B (x - m), x - m)` on the
sampling route. `weakened_ellipsoid` and its findings (D-191's first two,
D-193's second) are removed with it. Every mini-app's set part moves to the
ball, and the C++ backend app shows the operator form with the `L^2` ball
of its `H^1` space.

**D-195. Four defects the mini-apps surfaced, fixed before the review**
(2026-10-29, David: "fix what we know needs fixing first"). (1) A block
diagonal operator takes `domain=` and `codomain=`, which must equal the
plain sums of its blocks' spaces, and `from_product(domain=)` seats the
covariance, precision and factors on the caller's space with them, a
factor whose blocks map each summand to itself at both ends: a product of
Sobolev measures on a `Layered` space had its operators on the bare
`CoordinateDirectSum`, which compares equal to the layered space but has no
pointwise product, so a posterior built from them lost `sample_pointwise_std`;
the self-adjoint factor was the subtle half, its adjoint being itself with
the plain domain. (2) `from_standard_deviation` refuses on
`is_function_space`, not on `HilbertModule`: the C++ backend's space is a
function space and not a module, and it accepted an identity covariance.
(3) The ambient ball's randomized route estimates the trace by Hutchinson
(32 probes, against the decomposition's hundreds of applications) and
warns when the kept eigenvalues carry less than 95% of it, since the
dropped tail is positive and the radius a lower bound. (4)
`FeasiblePropertySet.support_sweep` runs its workers as processes, as every
loop of the library does (threads crash the sphere transforms and bought
nothing); the set and its structure drop their locks and caches when
pickled and recreate them on arrival. With them, two gaps in D-194's
ellipsoid form: the ellipsoid carries the ball's `radius` when built as
one, and `inverse=` must claim the operator's traits and pass one probe
of `B B^-1 x == x`. The tomography app's hand re-seating of its layered
factor is no longer needed and goes when the app is next touched.

**D-196. Correlated products: `from_product(correlation=)` ties the
factors through their latent white noise** (2026-10-29, David: it "could be
built now"; the plan of 2026-10-24 had it after the release). Each factor
draws as `L_i xi_i`; a correlation matrix `R`, symmetric, positive
semidefinite, unit diagonal, makes the latent noises correlated mode for
mode through the symmetric root `S` of `R`: the draw is `L_i (sum_j S_ij
xi_j)`, the marginals are the factors unchanged, the cross-covariance of
`i` and `j` is `R_ij L_i L_j*`. The factor is the block diagonal of the
`L_i` composed with a mixing operator on the direct sum of the latent
spaces, matrix-free and self-adjoint; the precision factor is the
mixing's inverse composed with the factors' precision factors, when every
factor has one and `R` is definite. The latent spaces must correspond mode
for mode: all the same space, where the sum is taken directly, or all
coordinate spaces of one dimension with a diagonal metric, where it is
taken on standardized components `sqrt(g_k) c_k`, the ones white noise
makes standard normal; anything else is refused with the reason. `Layered
.sobolev_measure(correlation=)` passes it through, so a planet's layers
can share structure across an interface without being continuous there.
The scale-dependent correlation of fields on one symmetric space stays
`correlated_measure`'s; this is the constant one, on any summands. Not a
rewrite of `correlated_measure` as a wrapper, which the plan had: its
fused spectral path is the better implementation of its own case.

**D-197. `ball`, `disk` and `interval` are packages of modules, not of
one `__init__.py`** (2026-10-29, David: "check that all the `__init__.py`
files are doing what they should and don't contain module code"). Each
held its space and the Lebesgue, Sobolev and layered forms in
`__init__.py`, 850, 866 and 273 lines, as the agents that built them left
them. Each is now `space.py` (the space and its two forms), `layered.py`
(the layered wrapper) and an `__init__.py` of the package docstring and
re-exports, so no import path changes; `interval.radial` stays as it was.
Every other package's `__init__.py` was already a docstring, imports and
the export list.

**D-198. The low-rank posterior: the prior corrected in the directions the
data inform** (2026-10-29, David: build it now and show the student, rather
than wait to design it jointly). `LowRankGaussianPosterior` in
`inference.gaussian`, a `GaussianPosterior`, holds the `r` leading
eigenpairs `(lambda_k, v_k)` of the prior-preconditioned misfit Hessian
`L* A* R^-1 A L`, found by the randomized eigendecomposition, and is
`C == L (I - V D V*) L*` with `D == diag(lambda / (1 + lambda))`: exact at
the number of nonzero eigenvalues, the prior in the directions dropped,
between the two otherwise. Covariance factor `L (I - V a V*)`, so a draw
costs a prior draw; precision `Li* (I + V Lambda V*) Li` through the prior's
precision factor when it has one, and otherwise `P + (P W) Lambda (P W)*`
with `W == L V` the modes as models, since `Li* V == P L V`, which is what
the symmetric spaces' invariant measures need, having a precision and no
factor; precision factor `(I + V b V*) Li` when `Li` exists. Nothing is
assembled. The evidence in the potential's convention is `-J(m) - sum
log(1 + lambda) / 2`, exact for a linear problem at full rank (the test
checks it against `LinearGaussianInversion`'s to 1e-8) and the Laplace
estimate otherwise, which answers the question D-132 left open.
`from_linear_problem(problem, prior, data, rank=)` gives the mean in closed
form, `m_0 + C A* R^-1 (d - A m_0)`, with no solve; `LaplaceApproximation(
rank=)` returns it at the mode in place of the full Gaussian. `modes()` are
the fields the data inform and `pointwise_variance_reduction()` is `sum_k
d_k (L v_k)^2` on any space with a pointwise product: the posterior's
pointwise variance where no closed form exists, without sampling, which
retires the Monte Carlo stopgap for deflated pointwise variance. Not yet:
an example of its own and the sampling layer's use of it as a reference,
both post-review.

**D-199. The planet-density app's radius is in Earth radii, and a default
element length carries a rounding margin** (2026-10-29, David: check the
planetmodel-based examples "are using sensible unit schemes"). The other
planet-scale examples sit on a planet of radius one; this app kept
kilometers, which put the Earth's volume into every inner product: PREM's
norm on the space was `1.03e7`, now 20.3, every density norm scaling by
`6371^-1.5`; densities stay in g/cm^3, the data in `10^24 kg` and `10^37
kg m^2`, the options in kilometers converted inside, the plots labeled in
kilometers. Parts A to C reproduce to every digit. Part D does not, and
cannot: Occam in the model space's norm depends on that norm, and with a
displacement unit now costing 1 against 0.079 for the density change with
the same mass effect, where in kilometers displacements were nearly free
(ICB moved 1 547 km), the two Occam answers now differ by 24 km at the
core-mantle boundary rather than by more than 100 km at both; the lesson,
that simple is norm dependent, stands with less contrast, which David
judged better as it is. The derivative check's probes needed no
rebalancing once the norms were of order one, only centering (a draw from
the prior itself is PREM over its norm); worst defect 1.9e-13. Two things
in the coordinate change: planetmodel's mesh takes `ceil(span / drmax)`
elements with no tolerance, so a span that is a whole number of elements
in exact arithmetic can round up to one more in other units, and its
banded eigensolver does not normalize eigenvector signs, so the changed
mesh flipped one basis function and a seeded draw with it. The library's
default element length is lengthened by a part in `10^9` (`axis.py`), which
keeps the mesh the one exact arithmetic gives; the sign normalization is
for the planetmodel 1.4 update after the launch.

**D-200. The 2.0 review and its decisions** (2026-10-29, David). Seven
Opus reviewers read the layers against the principles (algebra with the
checks and the parallel module, numerics, probability, inference,
geometry, the concrete spaces with plotting, and the examples and
tutorials), each with a reproduction per claim; 28 distinct Must-fix
findings, 96 Should, 71 Could, the fourteen with the widest reach
reproduced by Claude as well. Five decisions settled the contested points:
(1) the identity covariance on a function space is refused on every
route, in its literal forms, with `unsafe=True` the escape, and the two
tutorials that taught it are re-posed; D-168's whitened identity prior is
superseded by D-190; (2) the chi-squared potential keeps its internal
bypass (a misfit is a potential, and infinite-dimensional data spaces are
not a concern of the library), and the model-space formalism's docstrings
say it is for explicitly finite-dimensional problems and low-dimensional
surrogates, where its evidence is a direct determinant; (3) matrix-free
point evaluation is `route="auto" | "dense" | "local"` on both space
families, the symmetric-space names leave the top level for their own
package, and `HilbertSpace.mean` is `sample_expectation`; (4) the
model-space S-procedure is the default outer ellipsoid, the data-space
form stays behind `formalism=` with its regime documented, and both check
themselves against support values before returning; (5) the examples are
renumbered without gaps in the brief's order with four added, the
tutorial course goes point, measure, set, and point estimators are
presented once as what they are: exposed because wanted, tools within the
methods that carry uncertainty, with nothing about optimal regularization
parameters in this library. Six fix packages followed, D-201 to D-206,
each built by an agent with a failing-then-passing test per item; what
each skipped for want of a decision is in CURRENT_STATE §7.

**D-201. Measures: centered samplers, a sound correlated precision, no
silent symmetrization, the identity gate everywhere** (2026-10-29, fix
package A). A supplied sampler returns the centered fluctuation and the
measure adds its expectation: the affine image, the sum and the product
of a sampler-only Gaussian (a posterior, a conditioned measure, most sums)
drew about twice their mean after a translation, a push-forward or a sum,
with the reported expectation right. The correlated product's whitening
uses a factor's precision factor only when one probe confirms it inverts
the covariance factor, otherwise `L* P`, otherwise the product has no
precision; `check_measure` now holds `P C u == u`, `L L* u == C u` and
`Li* Li u == P u` before drawing, and takes `atol=`. `from_covariance_matrix`
refuses an asymmetry above `1e-8` relative with a message naming the
likely wrong `form=`, and symmetrizes only below it. One helper refuses the
literal identity on a function space at every constructor and spectral
route, each with `unsafe=`, and `from_standard_deviation` raises
`ValueError` like the rest. Also: `marginal` keeps factors, block
precisions and samplers; a diagonal covariance given to the constructor
gets its factor and precision; `kl_divergence(method="diagonal")` and
`ambient_ball(method="randomized")` replace two meanings of `"spectral"`;
`log_normalizing_constant` is exact or refused, with `dense_limit=` and
`n_jobs=`; the default `hilbert_schmidt_norm` refuses without coordinates
rather than estimating; `AbsolutelyContinuousMeasure.can_sample` is
false; `-mu` works. Tutorial 9 on `sobolev_measure(1.0, 0.2,
pointwise_std=0.8)` with Newton-CG preconditioned by the prior covariance
(10 iterations against about 160 for L-BFGS, whose evaluation failures
now carry the failure-protocol section); tutorial 13 without the
whitening, and honest that L-BFGS does not converge at its cap and ends
further from the truth than straight rays.

**D-202. Numerics: every tolerance relative to a scale the problem
supplies** (2026-10-29, fix package B). The weighted chi-square divides
its weights by the largest on entry and Imhof does the same, so
`ambient_ball` is the same ball in any units (it covered 0.2% of draws at
deviations of `3e-5`). The Lanczos calculus starts its buffer at eight
columns and doubles it, and judges breakdown against the running largest
coefficient with no floor. GMRES and MINRES declare convergence on the
true residual, GMRES right-preconditioned, both detecting a closed Krylov
space and solving its singular triangle by least squares. Chambolle-Pock
sets its steps per solve from the operator's, the data's and the
constraint's norms and stops on relative feasibility and relative
stationarity together, `SaddlePointResult.stationarity` public. The bundle
method's gap is relative to the objective's scale, a one-cut model below
tolerance resets the weight rather than stopping, and its clamps and its
simplex solver are relative. The Woodbury preconditioner's default inner
CG is `rtol=1e-10` (revising D-47's `1e-3`, which failed a
four-dimensional data space); `misfit_search` starts at `max(1, 10 floor)`
so a floor above one is allowed; the Jacobi, normal-diagonal, column and
sparse-approximation floors are relative to the largest entry; the
invariant-distance preconditioner refuses Tikhonov and damped members by
name and takes `n_jobs`; family caches keep two members; `n_jobs` reaches
every probe site, with `DirectSolver.n_jobs` readable; trust-region radii
start at the Cauchy length with relative cap and collapse.

**D-203. Feasible sets: the model-space ellipsoid, a flat set refused,
emptiness by certificate** (2026-10-29, fix package C). `SProcedure`
defaults to the model-space formalism; both formalisms build the
ellipsoid from an eigendecomposition of its shape, refuse a flat or
indefinite one, and check it against `Primal`'s attained support along
the longest and shortest axes both ways, inflating a shortfall within
`1e-6` of the half-width and raising above it, so the claim of a check is
true (the data-space route raises on the slack, overdetermined cases where
it cancels). A flat closed-form set, whose kernel spans fewer dimensions
than the property, refuses `ellipsoid`, `project` and the closed form's
`outer_ellipsoid` by name, with inclusion and the level gradient sharing
one pseudo-inverse, and a zero budget is a single point. A zero-radius
noise ball's center is folded into the data for every algorithm and kept
by `push_forward`. `outside()` tests `pairing > h + rtol max(|h|, |q||x|)`
in both the set and the convex layers. The dual reports the prior's own
level, catches only its private unboundedness, and declares emptiness
only with a certificate, a ray `r` with `(r, d) + h_prior(-A* r) +
h_noise(-r) < 0`, never by a threshold; every level comparison is `level +
rtol max(|level|, 1)`. Also: `n_jobs=` on `outer_ellipsoid`, `inner_hull`,
`extent` and `outside`; the algorithms exported from `geometry`;
`Ellipsoid` probing its covariance; `Polytope & non-polytope` an
intersection; membership tolerances without a floor; `extent` using
`via=` for its membership tests only; the quadratic tier's level function
with a gradient; unbounded priors refused by name in the data-space
formalism; `Primal` and `SProcedure` deciding emptiness and the fallback
fitting model in their own formalism with their own solver, so a
semidefinite prior now has a default outer ellipsoid; `support_sweep(
n_jobs=)` keeping the workers' results; the kernel solve shared between a
direction and its opposite; cache keys treating `-0.0` as `0.0`.

**D-204. Estimators: the noise expectation everywhere, the Laplace
curvature as asked, the low-rank mean in the kept directions, exact data
without a density** (2026-10-29, fix package D). The low-rank posterior's
linear constructor, `ConstrainedLeastSquares` and `BackusGilbert` remove
the error's expectation as the other linear estimators do (eight
estimators now agree with their zero-mean twins to `1e-12`). The low-rank
mean is `m_0 + L V diag(1/(1 + lambda)) V* L* A* R^-1 (d - A m_0 - mu_e)`:
the update only in the retained directions, each shrunk, exact at full
rank and bounded below it, where the earlier form kept the prior in the
dropped directions and was 150 to 320 times too large at rank 2 of 6.
`LaplaceApproximation` inverts half the exact Hessian when the MAP carries
one and the Gauss-Newton normal operator otherwise, `from_problem` taking
`gauss_newton=`; on the `rank=` route with the exact Hessian, negative
eigenvalues are clipped at zero and the docstring says so;
`linearized_normal` keeps meaning the Gauss-Newton operator, which the
structure-aware preconditioners read. `potential()` refuses a problem
with no error measure and `LinearGaussianInversion` returns a plain
Gaussian for exact data. `resolution()` and `unresolved()` are methods
like `propagated_covariance()`. `bias()` is `B(A m + mu_e) + t - T m`,
right for an estimator with a constant part (it was 100 times off on a
subspace missing the origin). Found on the way: `MinimumNorm.estimator_at`
and `for_data` kept the family's member solver for the old damping, so
every rebuild answered at it (9% wrong). Also: `ConstrainedMinimumNorm`
keeps its settings and has `MinimumNorm`'s surface; `LeastSquares` does
not solve at construction; `Minimization.push_forward` returns a public
`ComposedPointEstimator`; `log_evidence` and the mixture's return an
`Estimate`; `harden_error` is deleted; the mixture takes `n_jobs=` with
its generator split per component so parallel matches serial; the
full-rank Laplace Gaussian samples on the Gauss-Newton route and always
carries the Laplace evidence; `consistency_set` passes `unsafe=True` as
the potential does and both say the error measure must be trace class
for a limit to exist.

**D-205. Mechanics: picklable block operators, `route=`, scoped
symmetric-space names, `sample_expectation`, honest boxes and rays**
(2026-10-29, fix package E). The block, column, row and block-diagonal
operators and the spectral block operator take an optional first argument
in `__new__`, so they unpickle and every `n_jobs=` loop over a product,
layered or correlated measure, or a posterior from one, runs in processes.
`parallel_map(threads=)` is gone. `AffineOperator` and
`LinearPointEstimator` take `lazy=` to say a callable translation is a
deferred value, instead of guessing. `HilbertSpace.mean` is
`sample_expectation`. Matrix-free point evaluation is `route=` on both
families. `Box`, `Interval`, `Lebesgue`, `PeriodicBox` and `Sobolev` are
no longer top-level names, and `ShellTransform` is not public. A bounded
box, line or plane refuses points outside its domain, and its path and
ball quadrature no longer wrap nodes modulo the period (which put nodes
outside the domain, undone only by the enclosure's own wrap); a ball that
leaves the domain is refused. The ray travel-time operator takes the
order guard its derivative applies and keys its memo of traced rays on the
model's component values, not object identity. The checks: `check_operator`
compares a supplied `components_action` and `apply_block` with the
operator's own action; `check_traits` ranks only a stored dense array or
reads a known scaling; `check_hessian` is public; `atol=` on
`check_traits`, `check_projection` and `check_affine`. Cost: `D + I` stays
diagonal; a sparse factor stays sparse through a composition's diagonal
(`_known_sparse`); the Galerkin diagonal is read on a diagonal metric;
sums and scalings of functionals reuse their parts' Hessians; nonlinear
block operators call `at()` once per block and a linear one's derivative
is itself; exact cap averages are matrix-free unless `dense=True`. The
sphere's key includes `extend`; a box's reference point is a grid node
inside the domain, degree transfer refuses a different domain; the ball
and disk meshes no longer depend on the unit of length through one shared
helper, which changes the default mesh at radius one (45 nodes rather
than 49 at a length scale of 0.3) and with it seeded results there.
`plot_set` takes `n_jobs=` and honors `color=` on polytopes;
`plot_network` marks the receiver; `moments`, `plot_densities` and
`plot_corner` take `n_jobs=`.

**D-206. The examples renumbered and completed, the tutorial course
point, measure, set; and the correlated product tied to one latent
space** (2026-10-29, fix package F, and Claude on its finding). Thirty-six
examples without gaps: four new (`07_no_coordinates`,
`15_correlated_product`, `23_point_estimates`, `25_low_rank_posterior`)
and a fifth, `33_disk`, since the disk had none; `13_convex` merged into
`16_sets`; `22_forward_problem` rewritten on the estimators;
`30_feasible_set` cut to `ambient_ball` in both forms against the
prior-alone bound; `31_nonlinear_map` with `LaplaceApproximation(rank=)`.
The point-estimates example opens with the library's view of them. The
tutorials run 6 regularization (which now introduces the forward problem),
7 posterior, 8 properties, and 14 goes point, measure, set throughout;
every check in the examples prints its report and probes with a measure
on a function space; tutorial 12's refinement study probes with a smooth
measure and its defect falls monotonically on nine seeds; the prose that
contradicted its output (tutorial 8's bias, tutorial 13's L-BFGS,
examples 11 and 27) now says what runs; `make_notebooks.BRANCH` and the
badges carry a placeholder the release fills, tied by a test. The
walkthroughs cite example 23 for the prior's-norm pattern. **The
correlated product**: pairing the modes of different layers by index
tied them with arbitrary signs, since the spectral bases' eigenvector
signs are not normalized (D-199), so a positive correlation gave layer
masses correlated `+0.78`, `-0.77` and `-0.70`. `from_product(
correlation=)` now requires every factor's latent space to be one space,
where a mode of one factor is the same mode of another, and refuses
otherwise by name; `Layered.sobolev_measure(correlation=)` is withdrawn
until the planetmodel update normalizes the signs, when mode-for-mode
pairing across layers of one geometry can be given a meaning. D-196 is
amended accordingly.

**D-207. The mini-apps after the review** (2026-10-29, fix package G).
Every README number re-measured at full scale and every override at
quick; every slow test passing. The answers did not move except in the
tomography app's bent-ray part, whose layered disk's radial mesh changed
with D-205's unit-independent mesh (49/57 radii, was 53/61), so its truth
and the numbers derived from it moved, and the README's "comes no closer"
became "barely closer". What the library now offers replaced hand-built
pieces: the C++ app reads the ellipsoid's `radius`; the MFEM app's
self-adjoint-only wrapper around its solve is gone, `check_traits`
testing both claims without a matrix; the point-data app's serial first
direction (there so threads could share factorizations) is gone, part E
running whole in 115 s on six worker processes, and its printout says the
parent's tally cannot see the workers' applications; the tomography app's
re-seating of its layered prior factor is gone, `sobolev_prior` returning
`space.sobolev_measure(...)` directly, at a sampling cost of two to three
times until the product's `L L*` is folded block by block (§7); its
mixture takes `n_jobs=`, and its resolution bumps use the constant
extension, which the fit extension leaked into the padding. The waveform
app evaluated `LaplaceApproximation(rank=)` and did not adopt it (rank
100 overstates the box standard deviations by 30 to 40% at full scale;
matching the solves to 2% at quick scale takes twice the app's rank at
520 applications against 345), and the walkthrough says why; its README
had stale check and override numbers from before the review, now the
measured ones. The MPI test passes with PETSc's `mpicxx`.

**D-208. Optimizers stop on relative tolerances** (2026-10-30, David). The
review's one disease in numerics (D-202) reached the smooth optimizers,
the line searches, the convex methods and the augmented Lagrangian, all
of which stopped on numbers in the problem's units. `Optimizer` takes
`rtol=1e-8`, the gradient tolerance relative to the gradient at the
start, and `gtol=0.0`, an absolute floor under it for a start already
close to the answer; the test is `||g|| <= max(rtol ||g0||, gtol)`. The
precision floor that turns a failed line search near the minimizer into
convergence is `sqrt(eps * scale * L)` with `scale` the larger of `|f|`
at the point and at the start and no absolute term. Newton-CG and the
trust region set their forcing from `sqrt(||g|| / ||g0||)`. The first
trial step along a gradient direction is `|f| / ||g||^2`, Polyak's step
with a target of zero, the one guess the value and the gradient can make
with the right units, and one when the value is zero; the slope-ratio
heuristic clips the dimensionless ratio of slopes, not the step. Armijo's
`min_step` and Wolfe's `max_step` are multiples of the first trial step;
the Wolfe bracketing extrapolates by a secant on the slope, exact on a
quadratic, at most tenfold a move (a thousandfold sent L-BFGS on an
exponential forward into overflow), and its sectioning moves an
interpolant that lands inside the margin to the margin instead of
bisecting, so a minimizer many decades inside the bracket is reached in a
decade a step. The proximal gradient and the proximal point stop relative
to their first move, the proximal gradient's Lipschitz probe is a
millionth of the start's length, and `SubgradientDescent` judges its
subgradient relative to the first.

The augmented Lagrangian's inner tolerances are relative to the size of
the terms that cancel at a stationary point of the augmented Lagrangian:
the objective's gradient and each multiplier, as the update is about to
set it, times its constraint's gradient. The gradient at the start is not
that scale, being dominated by the penalty on whatever the start
violates; so the first inner solve is relative to its own start, and
because the scale is read where an inner solve ends, a solve that
stopped on its gradient tolerance short of it is continued from its own
end (at most eight times). The scale is kept as the largest seen, so a
start that is already the answer is judged against the problem it solved
and not against rounding. Without the refinement a first solve a
hundredfold down from a penalty-dominated start left a multiplier
estimate that was noise, the outer loop then moved it by one penalty
unit a step, and the stopping test, which does not check complementary
slackness, declared an overfit model converged. The arguments are
`initial_rtol`, `rtol_reduction`, `rtol` and the floor `gtol`;
`initial_gtol` and `gtol_reduction` are gone, and `ConstrainedMinimization`
and `Occam` take `rtol=` and `gtol=` the same way. The violation
`tolerance` stays in the constraints' units, which the caller chose.

The ray tracer refuses, with `EvaluationError`, a slowness model whose
bent rays leave the finite plane; the transform behind the evaluation
segfaulted on them, which a far trial step of a line search had found.

Tests: `TestOptimizersDoNotDependOnUnits` (five methods on a dense
metric at scales `1e-12`, `1` and `1e12`, the Newton methods step for
step the same; the step heuristics; the line-search limits), the
precision floor at `1e-12` and `1e12`, and the convex methods' tests
through `rtol`. Callers that passed `gtol=` as their working tolerance
pass `rtol=`: examples 16 and 17, and `Occam` in tutorial 14 (`1e-4`) and
the waveform app (`1e-5`, the loosest that reproduces the printed
digits). Re-measured: the Newton counts fell everywhere with the relative
forcing (tutorial 9's MAP 10 to 8, tutorial 13's 14 to 11, tutorial 14's
14 to 12, the tomography app's bent-ray MAP 7 to 6 iterations); L-BFGS
counts moved by a few either way (tutorial 3's on-space 40/43/48/46 to
34/47/44/44 at four grids, its by-components 175/275/891/3481 to
170/282/885/3481); tutorial 9's L-BFGS no longer overshoots into the
range where its forward fails, its first step being sized by the value,
so the tutorial shows the refusal protocol and points at example 12 for
a run that meets it. Occam costs more: the waveform app's full-scale
Occam takes 21 Newton iterations and 44 s where it took 13 and 30 s, its
table unchanged, and its preconditioned-against-plain comparison reads
59 against 24 iterations. The planet-density app is unchanged step for
step.

**D-209. Definiteness is propagated, not stamped** (2026-10-30, David).
`Traits.SURJECTIVE`: the operator maps onto its codomain, so its adjoint
has a trivial kernel; implied by `INVERTIBLE`, kept by a nonzero scaling
and by composition of two, not carried to the adjoint (injectivity has
no trait), checked by `check_traits` where cheap (a codomain no larger
than the domain; the rank of a stored dense array). It is what a
congruence `S A S*` needs of `S` to keep `A`'s definiteness, so
`congruence_traits(outer_surjective=)` and `gramian_traits(surjective=)`
replace the invertibility flags, the palindrome rule reads the trait off
the outer factors, and a push-forward `T C T*` of a definite covariance
claims definiteness when `T` claims `SURJECTIVE`; a diagonal block of a
covariance on a direct sum, a congruence by a projection, keeps
definiteness outright. Where definiteness is earned it is now said:
`from_covariance_matrix`'s factor claims invertibility when its inverse
root exists, a spectral-element measure's pointwise rescaling claims it
(the multiplier is positive everywhere), the layered spaces' jump
operators and the disk's claim `SURJECTIVE` (jumps at distinct
interfaces are independent), a noisy conditioning of a definite measure
by definite noise is definite, and the linear posterior covariance
`(I - K A) Q` claims definiteness when the prior's and the error's do.

`credible_set` and `condition` no longer claim it for the caller.
`credible_set` without a precision refuses a covariance that does not
claim `POSITIVE_DEFINITE` under an iterative solver, naming the three
ways out (a `SURJECTIVE` push-forward, `with_traits`, a direct solver);
under a direct solver the factorization is the proof, Cholesky failing
on a singular matrix. `condition` refuses `A C A* + R` that is not known
to be invertible: an exact constraint needs a definite covariance and an
operator claiming `SURJECTIVE`, a noisy one a definite noise covariance.
Tutorial 7's property operator claims `SURJECTIVE`, and its credible sets
on the plane take `CholeskySolver()`, the heat prior's spectrum
underflowing at high wavenumber so that the prior itself claims only
semidefiniteness. Tests: the trait algebra, `check_traits`, push-forward
and conditioning through claimed and unclaimed operators, and the
refusals by name.

## Appendix: measurements behind the defaults

Kept because they would be costly to redo. Timings are from a throttling
laptop and should be re-measured before being quoted.

- Feasible-set benchmarks (D-152, 2026-10-27, `benchmarks/`, a 16-core
  desktop with four workers per sweep, applications counted on the forward
  operator and its adjoint together):
  - Flexure–gravity (`flexure_gravity.py`, truncations 24, 32, 48, 64; 610
    data in two blocks at a signal-to-noise of 20; cap averages in field
    units; 32 directions): the primal with the spectral solver and with
    unpreconditioned CG at `rtol=1e-10` agree to `1e-10` at every
    truncation; the spectral sweep costs 2 470 applications and 8 to 16 s,
    the CG sweep 50 000 to 53 000 and 100 to 270 s. The truth is inside, the
    contact hull of twelve directions lies in the polytope, the S-procedure's
    volume-optimal weight rises from 0.82 to 0.92 with the truncation. The
    dual (300 bundle iterations, and 1 500 on the small run) does not
    converge on this problem: open. The bounds change by at most 44%, 40%
    and 14% (9%, 10%, 4% on average) between consecutive truncations, so
    degree 64 does not yet meet the few-per-cent criterion for a cap of
    radius 0.15 R; degrees 96 and 128 were run next.
  - Tutorial 10's tomography (degree 64, 1 024 rays, noise 0.05 on data of
    rms 0.09, heat prior) as a check that the solves inside a support value
    cost what the Bayesian solve costs: the posterior mean with a Woodbury
    surrogate at degree 16 takes 3 CG iterations at `rtol=1e-6` and 4 at
    `1e-10` (57 and 89 unpreconditioned); each of the roughly forty solves
    behind one support value takes 6 and 9 with the same surrogate, for
    463 and 673 applications and brackets of `1.6e-4` and `5.5e-7`.

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
- Spectral-element spaces (D-116 to D-118). One white-noise draw with three
  inner products in a ball: 19.8 ms at `lmax` 8 on a `(65, 9, 17)` grid, 13.3
  ms at `lmax` 4 on `(65, 5, 9)`, 6.4 ms on `(29, 5, 9)`; the cost is the
  per-shell loop through pyshtools. Padding on `[0, 2]`, `L` 0.1, covariance
  `A^-2`, 100 modes: the variance at the endpoints moves 0.3 per cent between
  a padding of four length scales and eight, the interior 3.5e-4; with no
  padding it is 2.001 times as large, the doubling a reflecting end gives by
  the method of images. The interior sits within 1e-3 of the whole-line value
  `1 / 4L`. Relative `L2` error on the unit ball of `1 + z` sampled and
  truncated, `lmax` 4: 1.0e-2, 4.7e-3, 2.0e-3, 6.7e-4 at 4, 8, 16, 32 radial
  modes, the error sitting at the surface where the constant extension leaves
  a kink; at a single point it is not monotone. Standard deviation realized at
  the centre of the unit ball, `lmax` 5, covariance `A^-2.5`, at 12, 24, 48
  radial modes: target constant, 1e-4 throughout; `0.3 + 0.1 r`, 3.0, 1.4 and
  0.64 per cent (2.8 and 1.2 at `lmax` 10 for 12 and 24 modes, 1.3 at `lmax`
  16 for 24: the degree does not enter); `0.3 + 0.1 r²`, 0.32, 0.064 and 0.009
  per cent. One length scale, 0.3, and one order throughout.
- The ball's default truncation (D-118), unit ball, `lmax` 32: bound from the
  padded mesh, 11 radial modes at degree zero and one at degree 32, dimension
  3374, the same at `L` 0.3 (padded radius 2.2) and 0.1 (1.4), shortest
  wavelength 0.395 and 0.251 against an angular 0.196 at the surface; bound
  from the domain, 25 and 16 modes at degree zero, 11 and 3 at degree 32,
  dimension 16 252 and 7243, shortest wavelength 0.180 in both.
- Continuing a sampled function (D-119), error of the truncation against the
  function. `cos` on `[-1, 2]`, `L` 0.1, at 32, 64, 128 modes, error at the
  endpoint: constant 1.1e-2, 5.7e-3, 2.7e-3; reflected 1.6e-3, 5.4e-4, 1.3e-4;
  `cos` known on the padding 9.1e-4, 4.0e-4, 8.7e-5, with the reflected `L2`
  error within a few per cent of the last. `cos(3r)(1 + z)`, `lmax` 4, at 4,
  8, 16, 32, 64 radial modes, relative `L2` over the domain: annulus `[0.5,
  1]`, constant 9.7e-2, 1.9e-2, 7.0e-3, 2.0e-3, 7.5e-4 and reflected 6.1e-2,
  7.1e-3, 2.5e-3, 1.0e-3, 2.6e-4, the largest error in the inner third at 64
  modes 1.2e-2 against 6.4e-4; unit ball, padding 1.2, constant 6e-2, 2e-2,
  5e-3, 2e-3, 5e-4 and reflected 2e-1, 3e-2, 3e-3, 8e-4, 1e-4, the largest
  error within `r < 0.25` at 64 modes 5e-3 against 4e-3, and at padding 0.5,
  5e-4 against 3e-2. `1 + z` in the ball at 32 modes, by radius: at the
  surface 6.3e-3 constant against 5.8e-4 reflected, within `r < 0.25` 3.7e-3
  against 8.3e-3, unchanged when the element length is halved.
- The centre of a ball (D-119), `cos(3r)`, `L` 0.3, padding 1.2, at 6, 8, 12,
  16, 20, 24, 32, 40, 48, 64 radial modes. Error at `r = 0`, constant fill:
  +3.7e-2, -3.1e-2, -2.3e-2, -1.8e-2, -1.2e-2, -7.0e-3, +1.9e-3, +5.3e-3,
  +2.9e-3, -3.3e-3; reflected: +5.1e-3, -1.5e-1, -3.7e-2, -3.2e-3, +8.9e-4,
  -1.0e-3, +4.6e-3, +4.4e-3, -2.9e-3, +8.9e-5. The leaked term's norm over the
  domain falls steadily, 4.0e-3 to 1.2e-4, while its value at the centre does
  not. A unit-norm mode of degree zero at the centre and at the surface: mode
  1, 2.0 and 0.87; mode 16, 22 and 0.95; mode 63, 86 and 0.40. Annulus `[0.5,
  1]`, `L` 0.2, at 8, 16, 32, 64 modes, largest leaked term 6.7e-2, 3.2e-2,
  1.7e-2, 8.5e-3 against a constant-fill error of 7.1e-2, 3.5e-2, 1.7e-2,
  8.3e-3, leaving a reflected error of 7.2e-3, 3.2e-3, 1.4e-3, 2.9e-4.
- Robin ends (D-122), worst relative error of the prior variance on the domain
  against a mesh padded by twelve length scales, `L` 0.1. Interval `[0, 2]`,
  400 modes, at paddings 0, 1, 2, 4 `L`: `A⁻¹` natural 1.0, 1.6e-1, 3.0e-2,
  1.1e-2 and `γ = 1/L` 2.3e-2, 1.9e-2, 1.2e-2, 1.0e-2, the last figures being
  the truncation's; `A⁻²` natural 1.0, 4.1e-1, 9.2e-2, 3.0e-3 and `1/L` 5.0e-1,
  6.8e-2, 9.1e-3, 1.6e-4. The coefficient, `γ = c/L` at padding 2 `L`: `A⁻²`,
  c = 0.25, 0.5, 0.7, 1, 1.4, 2, 4 gives 4.9e-2, 2.2e-2, 7.3e-3, 9.1e-3, 2.4e-2,
  3.9e-2, 6.1e-2; `A⁻³`, 8.9e-2, 3.0e-2, 2.6e-3, 3.7e-2, 6.7e-2, 9.5e-2, 1.4e-1.
  Curvature, shell `[0.5, 1]` at 2 `L`, c = 0.7 flat against curved: `A⁻²·⁵`
  2e-2 and 6e-3, `A⁻³` 4e-2 and 3e-3, `A⁻⁴` 8e-2 and 3e-2; in a whole ball the
  two are alike from 2 `L`. Full ball, `lmax` 12, `L` 0.2, `A⁻³`, at 1, 2, 3,
  4 `L`: natural 7e-1, 2e-1, 4e-2, 7e-3 and `"robin"` 3e-2, 3e-3, 2e-3, 5e-4,
  dimension 4511, 5379, 6276, 7124; the annulus the same from 2 `L`. `A⁻¹` has
  no pointwise variance in three dimensions and was not a test there.
- The batched transform (D-123), against planetmodel's loop through pyshtools,
  per transform at `lmax` and shells (4, 29), (8, 65), (16, 105), (32, 105),
  (64, 150), (128, 100): 0.79, 2.6, 8.6, 18, 45, 60 ms against 0.01, 0.02,
  0.11, 2.1, 17, 39 ms; agreement 5e-15 rising to 5e-11.
- A fit over the domain (D-124), relative `L2` error on the domain. `cos` on
  `[-1, 2]` at 32, 64, 128 modes: reflected 1.0e-3, 2.4e-4, 5.9e-5; fitted
  9.7e-8, 7.1e-13, 1.2e-14, the smallest singular value kept 1e-4, 2.5e-9,
  3e-16 of the largest and the largest coefficient 0.86 to 0.89 against 0.85.
  `cos(3r)(1 + z)`, `lmax` 4, at 8, 16, 32 radial modes: unit ball, constant
  2e-2, 5e-3, 2e-3, reflected 3e-2, 3e-3, 8e-4, fitted 4e-9, 8e-10, 8e-10, the
  largest error within `r < 0.25` 5e-2 to 7e-3, 1e-1 to 1e-2 and 2e-6; annulus
  fitted 6e-8, 2e-10, 1e-10.
- The fills under the Robin default (D-119, D-122), `cos` on `[-1, 2]`, `L`
  0.1, two length scales of padding, at 32, 64, 128 modes, relative `L2` on
  the domain and error at `x = 2`: constant 4.1e-3, 1.2e-3, 3.2e-4 and 1.8e-2,
  7.2e-3, 3.1e-3; reflected 4.6e-3, 1.2e-3, 3.2e-4 and 1.3e-2, 4.0e-3, 1.2e-3;
  fitted 3.3e-5, 3.1e-8, 7.0e-14 and 3.1e-4, 4.1e-7, 2.2e-13. Under the
  natural condition and four length scales the reflection's `L2` was 1.0e-3,
  2.4e-4, 5.9e-5. The standard error of `check_white_noise`'s off-diagonal
  estimate is the geometric mean of the two metric values over root `n`, which
  a Robin condition raises: 0.048 against 0.021 at 20 000 draws on a radial
  `H²` with `L` 0.3, against the fixed 0.06 the check then had (D-125). With
  the metric `[1, 4, 9]` scaled by 1e4 a correct space gave an off-diagonal
  310 against that 0.06; scaled by 1e-4 the v1 construction passed it.

- Recovered from docstrings on 2026-10-28 (D-157; measured between
  2026-08 and 2026-10 on the same laptop unless said otherwise):
  - Orthonormalising 50 fields on a sphere: 1.9 s through `inner_product`
    against 0.1 s on components (100 transforms); a 30-step Lanczos run,
    1047 analyses against 152. CG on a 2000-dimensional operator over a
    dense Gram matrix: 320 ms stored as Galerkin against 95 ms as components.
  - `rtol` default (D-8): a 40 000-dimensional five-point Laplacian with a
    0.01 shift, CG takes 147 / 243 / 301 iterations at `rtol` 1e-5 / 1e-8 /
    1e-10 with solution errors 4e-5 / 4e-8 / 4e-10.
  - `BlockPreconditioner` in blocks of 20: 3.5 MB at dimension 2000 against
    97 MB for the dense matrix; a dict-of-sets pattern cost 11 MB where the
    pattern itself is 1.6. `ColumnThresholdedPreconditioner` at 2000 with
    threshold 0.1 peaks at 1.2 MB.
  - `random_diagonal`'s norm-ratio error estimate against the achieved
    error on a 120-dimensional operator: 0.042 against 0.045 at 20 probes,
    0.0094 against 0.0094 at 420.
  - Lanczos reorthogonalisation: on a spectrum 1e-6 to 1e6 over 200 steps,
    a run to 250 of 300 dimensions and tenfold repeated eigenvalues, a second
    pass moves `max |Q Q^T - I|` from 1.3e-15 to between 1.1e-15 and 1.8e-15.
    `eigh_tridiagonal` against `np.linalg.eigh` at k = 3/5/10/20/40:
    7.6/8.4/12.5/26.2/65.9 us against 3.5/4.5/8.6/22.7/80.4 us, one BLAS
    thread, crossover between 20 and 40, hence the limit 32. A convergence
    check every k steps is slower (200 probes, `log` at `rtol` 1e-3: dim 300,
    0.037/0.038/0.041 s for checks every 1/2/3 steps; dim 2000,
    1.06/1.23/1.37 s), the check itself 27% of a stochastic log-determinant
    at dimension 300.
  - `log_determinant`'s dense limit (D-50): slogdet at 4000 takes 0.8 s on
    one thread and 2.0 s at 4096 on this BLAS; dense against stochastic on a
    well-conditioned operator, 4.2x faster at 2000 and 3.6x at 3000; the
    stochastic estimate of a diagonal operator at dimension 1000, 455.4 +/-
    2.8 against 456.7; factorising the identity at 3000 cost 0.37 s and
    doubled the memory.
  - Import time: `scipy.stats` and `scipy.optimize` were 0.26 s of a 0.48 s
    `import pygeoinf`, 0.22 s once `quadratic_forms` and `inference.problem`
    import them lazily.
  - `_imhof` by trapezoid against `quad` at three weights: 3.7 ms against
    106 ms. Brent in `log t` against bisection on a chi-squared-like
    quantity from three starts: 10-13 against 23-28 solves per root.
  - `ftol` (D-157): on a 60-dimensional quadratic of condition 4.9, L-BFGS
    stopped at |g| = 1.8e-7 after 19 iterations and reported failure,
    nonlinear CG at 1.1e-6, steepest descent took 2000 iterations and 5330
    evaluations; with `ftol` and the precision outcome the four converge in
    16 to 25 iterations and 40 to 119 evaluations.
  - `_minimize_on_simplex`: a general nonlinear solver cost fifty seconds per
    bundle minimisation; over 30 random bundles at 400 iterations the
    accelerated method left a smaller KKT residual on 93 per cent, by a
    median factor of 3300, and ended behind on 2 of 30; keeping the best
    iterate took the median from 557 to 3300; across the Backus tests the
    median `Q` condition number was 1e23 and the residual 4.8e-7 at the
    median, 3.5e-3 at the 99th percentile, the reason `tolerance` and
    `warn_above` are separate.
  - The retired KKT route against `ChambollePockSolver` over sixteen
    directions at noise radius 0.05: fourteen agree to 1e-11, two where the
    multiplier hit its clip to 2.4e-3. The retired dual route against its
    primal: 1e-10 (strong duality checked); dual against primal 1.7e-8
    relative, primal against KKT 2.7e-11, on the support values. The retired
    likelihood route's fitted misfit sat 0.8 per cent above the level on one
    seed in twenty-five before the multiplier scaling.
  - KL divergence: dense at dimension 4000 exact in 4.4 s, a hundred probes
    37 s to reach +/- 1.5; dense 0.09 / 0.63 / 4.4 s against probes' 1.4 /
    9.3 / 37 s at 960 / 2000 / 4000. The folded stochastic route on a dense
    reference of dimension 400 and condition 1e6: 15.14 +/- 0.30 against
    15.67, where the old three-estimate route gave -7.0 +/- 22.8. The
    generalised symmetric eigenproblem `(C_gal, G)` for `ambient_ball`
    against a non-symmetric decomposition of `C_c`: 0.05 s against 0.26 s at
    1000, 4.0 s against 15.2 s at 4000. Imhof 0.8 s at 500 weights and
    10.4 s at 2000; the matched chi-square agrees to 9e-5 and 3e-5 relative
    in 0.001 s.
  - Binned density: an axis-aligned filter is wrong by 8% of the peak and
    moves contour levels by 6% on a correlated cloud; the full kernel 1.5%
    and 0.7%; 2.2 ms against `gaussian_kde`'s 5.0 s per panel at 20 000
    draws on a 160 x 160 grid (8.9 s of example 26's 12.2 s).
  - The false noise-free model-space posterior `Q^-1 + A* A`: mean wrong by
    0.86 against a mean of norm 2.6 on a random 12-by-8 problem.
  - Preparing points once (2026-08-29): the conversion was 14.5 of 37 ms on
    a sphere at 1e5 points, 32 of 61 ms on a torus, 230 of 254 ms on a
    bounded box. finufft at every core against one thread on a 16-core
    machine: 21 ms against 4.4 ms at `lmax` 128 for every point count from
    200 to 10 000. The sphere's point-evaluation thresholds: the old ones
    left a factor of 13 at `lmax` 64 and 1000 points (34 ms direct against
    2.6 ms transform). The sphere's quadrature inner product agrees with the
    component route to 6e-15 at every truncation, radius and sampling
    tested, 2e-14 with one raw pointwise product as an argument, 0.05% to 2%
    apart with both. `Box.project_function` without the taper: a path
    integral from 0.05 to 0.95 of the constant one over `[0, 1]` gives
    0.851 against 0.9, 7e-6 for a path from 0.4 to 0.6.
  - `_to_mfem` entry-by-entry conversion: 16 ms for 33 000 stored entries.
