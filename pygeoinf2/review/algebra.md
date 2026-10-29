# Review of the algebra layer for 2.0

Scope: `pygeoinf2/algebra/` (spaces, operators, nodes, diagonal, direct_sum,
linearization), `pygeoinf2/testing.py`, `pygeoinf2/parallel.py`,
`pygeoinf2/traits.py`. Every claim marked "verified" was run; the scripts are
`r1.py` to `r35.py` in
`/tmp/claude-1000/-home-david-dev-pygeoinf/4bec2589-8482-4f4c-843c-b27f331a8905/scratchpad/`,
and each reproduction below is the minimal form of one of them. Spaces named
`X` with a dense metric are `tests.conftest.make_dense_metric_space`.

## Must fix before 2.0

### Block operators cannot be unpickled, so every `n_jobs=` loop over them crashes
`pygeoinf2/algebra/direct_sum.py:521` (also `:646`, `:720`, `:812`). `BlockOperator`,
`ColumnOperator`, `RowOperator` and `BlockDiagonalOperator` define a `__new__`
that requires the blocks. Unpickling calls `cls.__new__(cls)` with no arguments,
so none of the four classes, linear or not, survives a pickle round trip.
`with_traits` already works around this for `copy` (comment at
`operators.py:591`). D-181 fixed the same fault for `DirectSum.__new__`, saying
"process-parallel draws over a measure on one work", but only for the space. The
covariance factor of every `GaussianMeasure.from_product` and of every
`Layered.sobolev_measure` is a `BlockDiagonalLinearOperator`. So process-parallel
sampling crashes on exactly the priors that use direct sums. The same goes for
`matrix(n_jobs=)` and `diagonals(n_jobs=)` on any block operator, and for any
process loop whose closure holds one (likely the D-195 process sweep over a set
built on a direct-sum model space too; I did not run that one). The tomography
app's hand re-seating onto one `DiagonalLinearOperator`, which D-195 says can now
go, is currently what keeps that app's parallel paths alive. Verified.
Reproduction:
```python
E = EuclideanSpace(2)
mu = GaussianMeasure.from_product([GaussianMeasure.from_standard_deviation(E, 1.0)] * 2)
mu.samples(4, rng=np.random.default_rng(0), n_jobs=2)
# BrokenProcessPool: A task has failed to un-serialize.
pygeoinf2.interval.Layered([0, .5, 1], 12, order=1.0, length_scale=0.2).sobolev_measure(2.0).samples(4, n_jobs=2)
# BrokenProcessPool, same
pickle.loads(pickle.dumps(BlockDiagonalLinearOperator([D1, D2])))
# TypeError: BlockDiagonalOperator.__new__() missing 1 required positional argument: 'operators'
```
Proposed fix: make the first argument of each `__new__` optional (`blocks=None` /
`operators=None`), as `DirectSum.__new__` now does, and add a pickling test for
all eight block classes and for `samples(n_jobs=2)` on a product measure.

### `parallel_map(threads=)` is a dead public keyword that contradicts processes-only
`pygeoinf2/parallel.py:151`. D-151 added `threads=` for the support sweep, and
D-195 moved the sweep to processes. Nothing in the library, the mini-apps, the
examples or the tests passes `threads=` any more. A grep for `threads` in
`geometry`, `numerics`, `inference` and `probability` finds one comment and no
caller, and no test exercises `threads=True`. The keyword still ships on a public
function, and its docstring justifies it by a caller that no longer exists ("a set
with its caches and locks"). The module docstring says the opposite: "There is no
`backend` argument on the loops themselves... `threading` crashes on a sphere".
The principle is processes, not threads, and removing a public keyword after 2.0
is a breaking change. Verified by grep.
Reproduction: `grep -rn threads pygeoinf2/{geometry,numerics,inference,probability}/*.py`
prints only `numerics/solvers.py:1227` (a comment).
Proposed fix: delete `threads=` and its branch. A caller who wants threads already
has `joblib.parallel_config(backend="threading")`, which the waveform app uses and
the module docstring documents.

### `AffineOperator` takes any callable translation for a thunk, so a callable vector breaks it
`pygeoinf2/algebra/operators.py:2710`. The translation slot holds either a vector
or a zero-argument callable, told apart by `callable(...)`. The library's contract
is that vectors are "whatever objects a backend provides" (D-1, D-169, §6), and some
backends' vectors are callable: a finite element `Function` that evaluates at a
point, or an interpolant. Such a vector is called with no arguments when it is
first read. The thunk is used internally (`inference/gaussian.py:180`,
`inference/estimators.py`), so the convention cannot be changed after 2.0 without
breaking callers. Verified with a 20-line callable-vector space. None of the
shipped backends (arrays, `SHGrid`, MFEM, the C++ space) has callable vectors, so
it is Must only if callable vectors are part of the contract, which is your call.
Reproduction:
```python
class Field:                       # a vector that evaluates at a point
    def __init__(self, c): self.c = np.asarray(c, float)
    def __call__(self, point): return float(np.polyval(self.c, point))
# FieldSpace: a HilbertSpace over Field (r26.py)
AffineOperator(LinearOperator.identity(V), Field([1, 2, 3]))(V.zero())
# TypeError: Field.__call__() missing 1 required positional argument: 'point'
```
Proposed fix: take the lazy form by name, either a keyword (`AffineOperator(A, b)`
plus `AffineOperator(A, translation_factory=f)`) or a classmethod
`AffineOperator.lazy(A, f)`, and never test a vector with `callable`.

## Should fix

### `check_traits` forms and decomposes a dense matrix for every diagonal, sparse or composed operator that claims definiteness
`pygeoinf2/testing.py:810`. D-181 meant full rank to be checked "only where the
matrix is already stored". The code reads `operator._known_matrix("components")`
(a private hook, while `known_matrix(form=)` is public). For a
`DiagonalLinearOperator` that hook writes `np.diag(d)`, for a sparse
`MatrixLinearOperator` it densifies, and for a composition of known factors it
multiplies them. `matrix_rank` then runs an SVD. Every diagonal covariance claims
positive definiteness, which implies `INVERTIBLE`, and those covariances are the
symmetric spaces' invariant measures, i.e. most priors in the library. So the
ladder step `check_traits(prior.covariance)` grows like n³: about ten seconds at
degree 64 on a sphere, and by extrapolation 2.2 GB and some ten minutes at
degree 128. Separately, the
docstring says the function verifies "every trait the operator claims", but
`INVERTIBLE` is checked only in this case. Verified.
Reproduction: `C = circle.Sobolev(n, 1.5, 0.05).sobolev_measure(2.0, 0.05).covariance`
(a `DiagonalLinearOperator` claiming `SA|PSD|PD|INVERTIBLE`):
`check_traits(C)` takes 0.8 s at n = 2048 and 9.6 s at n = 4096, which is all
`matrix_rank` (12.4 s on the bare `np.diag` at 4096 in the same run). A sparse
tridiagonal `from_matrix(..., traits=INVERTIBLE)` at n = 3000 took 9.6 s.
Proposed fix: decide full rank from `_known_scaling()` (no zero entries) where it
exists. Otherwise test only an array that is actually stored and dense
(`MatrixLinearOperator` with a dense array; for a sparse one use a sparse LU or
skip). Say in the docstring which claims are verified and which only partly.

### `check_measure` passes a measure whose precision or factors contradict its covariance
`pygeoinf2/testing.py:1040`. The check compares sample moments with `expectation`
and `covariance` to `rtol=0.08`, and that is all it does. It never applies the
exact identities a `GaussianMeasure` declares: `C == L L*`, `P C == I` and
`Lp L == I`. Downstream, the precision drives MAP objectives, Mahalanobis
distances, `log_density` and posterior precisions. D-107 says a claim of
exactness is tested as exact, but here the factor-covariance relation is tested
only statistically at 8%. The C++ mini-app writes the two identities by hand
(`miniapps/cpp_backend/ladder.py:96`) "since a wrong precision passes
`check_operator` on its own". Verified.
Reproduction:
```python
L = LinearOperator.from_matrix(X, X, np.tril(rng.normal(size=(4, 4))) + 3*np.eye(4), form="components")
mu = GaussianMeasure(X, covariance_factor=L, precision=(2.0 * (L @ L.adjoint)).with_traits(Traits.POSITIVE_DEFINITE))
check_measure(mu, rng=rng)          # CheckReport(check_measure, worst=1.01): passes
# |P C x - x| / |x| = 740
```
Proposed fix: in `check_measure`, before sampling, check each declared pair on a
few probes at a tight `rtol`: covariance against factor times factor adjoint,
precision after covariance, and precision factor after factor. Record each
identity in the report.

### `check_operator` never compares an operator's components action with its action
`pygeoinf2/testing.py:612`. `from_callables(components_action=...)`,
`from_formal_adjoint` and every structured operator supply `_components_action`.
A composition (`nodes._apply_chain`) and a sum use that action instead of
`__call__` whenever every neighbour has one. A components action that disagrees
with the value therefore passes the ladder and then gives a different answer
inside any product. The backend contract (§6) relies on `check_operator` for
exactly the hand-written pieces a backend supplies, and `apply_block` overrides
are not compared either. Verified.
Reproduction:
```python
G = X.gram_matrix()
B = LinearOperator.from_callables(X, X, lambda x: 2*x, adjoint=lambda y: 2*y,
                                  components_action=lambda c: 2*G@c, components_adjoint_action=lambda c: 2*c)
check_operator(B, rng=rng)                     # worst=0: passes
(DiagonalLinearOperator(X, np.ones(4)) @ B)(x) # differs from B(x): [-0.48 2.66 -0.07 -0.98] vs [-0.57 3.15 -0.87 -1.47]
```
Proposed fix: when `components_action()` or `components_adjoint_action()` is not
None, compare it with `to_components(A(x))` and `to_components(A*(y))` on the
probes. Also compare `apply_block` with single applications. Record both in the
report.

### `D + I` stays a generic sum and brings back the dense square that D-170 removed
`pygeoinf2/algebra/nodes.py:50` and `:424`. `_Identity` has no `_combine_add`, and
`_Sum` has no `_known_scaling`. So `D + I`, with D a `DiagonalLinearOperator` (as
in `C + I`, `R + I`, `sigma^2 I + D`), is a `_Sum`, not a diagonal. A composition
through it, such as `A* (D + I) A`, then reads two dense n×n matrices instead of
scaling rows. That is the 20 000-point failure D-170 fixed, back by way of one
`+`. `D + 2 I` is unaffected, because `2 I` is already diagonal. Verified.
Reproduction: n = 6000, m = 300, `(A.adjoint @ R @ A).matrix()` takes 0.03 s and
43 MB peak with `R = D + 2.0*I` (a `DiagonalLinearOperator`). With `R = D + I` or
`I + D` (a `_Sum`) it takes 0.29 s and 878 MB.
Proposed fix: fold the identity in `DiagonalLinearOperator._combine_add` (or give
`_Identity` a `_combine_add` / `_combine_radd` that goes through `1.0 * self`).
Give `_Sum` a `_known_scaling` that sums its terms' scalings when all have one.

### Compositions densify sparse factors to read a matrix or a diagonal
`pygeoinf2/algebra/nodes.py:699` (via `operators.py:2094`).
`_Composition._known_factors` asks every factor for `_known_matrix("components")`,
and a sparse `MatrixLinearOperator` answers by `todense()`. The size guard then
counts the densified n² as the "largest factor", so the product goes through.
`diagonals()`, which exists for when "the matrix would not fit", takes the same
path for a composition. The main diagonal of `A* R A` with a sparse `A` therefore
forms two n×n arrays to produce n numbers. This contradicts §5 and the lesson
"never densify to slice" from §6, and it is the shape of the MFEM and point-data
normal operators. Verified.
Reproduction: n = 5000, `A` tridiagonal CSR (14 998 nonzeros), `R` diagonal:
`(A.adjoint @ R @ A).diagonals()` peaks at 600 MB, where `A.diagonals()` peaks at
0.08 MB. The answer is correct.
Proposed fix: let `_known_factors` keep a stored sparse array sparse (scipy
products, `A.multiply(B.T).sum(axis=1)` for the diagonal). Alternatively, have a
sparse factor decline `_known_matrix` inside a composition so that `diagonals`
probes rather than densifies.

### The Galerkin diagonal of a composition is probed column by column on a diagonal metric
`pygeoinf2/algebra/nodes.py:771`. `_Composition._known_diagonals` reads the
Galerkin diagonal only on an orthonormal codomain. On a diagonal metric, which
includes every Sobolev and Lebesgue symmetric space and every spectral-element
space, the Galerkin diagonal is `g * diag(A_c)` and can be read for free. Instead
the base class probes n columns, each a full application of the chain.
`JacobiPreconditioner`, `NormalDiagonalPreconditioner` and `sparse_approximation`
all ask for `form="galerkin"`. `_Adjoint._known_diagonals` (`nodes.py:275`)
declines the components form for the same reason. Verified.
Reproduction: n = 3000, m = 200, `N = A.adjoint @ Rinv @ A` with matrix-backed
`A`. `N.diagonals(form="galerkin")` is read in 0.002 s on `EuclideanSpace` but
probed in 0.114 s on a `WeightedSpace` (`N._known_diagonals((0,), "galerkin")` is
None). The probe grows like n² m.
Proposed fix: where `codomain.has_diagonal_metric`, multiply the components
diagonal by `codomain.gram_diagonal()`. Do the same, with a division, for the
components form of `_Adjoint`.

### Scaling or summing a functional re-linearizes the forward model
`pygeoinf2/algebra/nodes.py:1045` and `:1068`. `_FunctionalSum._linearize` and
`_FunctionalScaled._linearize` first take each part's `at(x)` and then call
`self._hessian(x)`. That calls each part's `hessian(x)` afresh, and for a
composed or hand-built objective that means another `forward.at(x)`. Each level
of `+` or `*` around a functional with a Hessian therefore adds one forward
linearization per `at()` and per `gradient()`. In the library,
`MaximumAPosteriori.log_posterior` is `-0.5 * objective`, so a sampler or
optimizer on it pays two forward solves where `objective` pays one. This breaks
the "one linearization per point" promise in the `potential` and `objective`
docstrings. Verified.
Reproduction: with `F` built from one `linearize` callable,
`(phi @ F).at(x)` costs 1 forward linearization, `(0.5 * (phi @ F)).at(x)` costs
2, `(phi @ F + reg).at(x)` costs 2 and `(0.5*(phi @ F) + reg).at(x)` costs 3.
`MaximumAPosteriori(problem, prior).objective(d).at(m)` costs 1 and
`.log_posterior(d).at(m)` costs 2.
Proposed fix: build the Hessian from the quadratic models already in hand
(`sum(part.hessian)`, `alpha * part.hessian`) when every part carries one, and
call `_hessian(x)` only when one does not.

### Nonlinear block operators evaluate each block twice in `at()`
`pygeoinf2/algebra/direct_sum.py:513` (also `ColumnOperator`, `RowOperator`,
`BlockDiagonalOperator`). None of the four overrides `_linearize`, so
`Operator._linearize` calls `_value` (every block's value) and then `_derivative`
(every block's `derivative`). For a block built from one `linearize` callable,
which is a PDE solve yielding both, `at()` costs two solves.
`_OperatorSum` and `_OperatorComposition` honour D-29's one-call `at()`. The
module docstring's showcase, the joint model `[[I, 0], [F, I]]`, is the case
that pays. Verified.
Reproduction: `F.at(x)` costs 1 forward solve. `at()` on
`BlockOperator([[I, 0], [F, I]])`, `ColumnOperator([I, F])`,
`RowOperator([F, I])` or `BlockDiagonalOperator([F, I])` costs 2 forward solves.
Proposed fix: give each a `_linearize` that calls every block's `at()` once and
assembles the value and the block derivative from the results.

### `Functional.gradient` and `subgradient` evaluate the value and the Hessian too
`pygeoinf2/algebra/operators.py:2213` (`subgradient` at `:2240`). `gradient(x)` is
`self.at(x).gradient`, and `at()` builds a full `QuadraticModel`: value,
derivative, and Hessian when one exists. D-29 says a derivative-only query never
pays for a value. `subgradient` defaults to `gradient`, so the bundle and convex
methods (`numerics/convex.py:897`, `geometry/convex.py:321`, `:573`,
`geometry/feasible.py:1398`) pay a value evaluation per gradient. On an
objective over a PDE solve that is a second solve. Verified.
Reproduction: for `Functional.from_callables(X, value, derivative=..., hessian=...)`,
`f.gradient(x)` makes calls `{'value': 1, 'derivative': 1, 'hessian': 1}`, while
`f.derivative(x).representer` makes `{'value': 0, 'derivative': 1, 'hessian': 0}`.
Proposed fix: `return self.derivative(x).adjoint(1.0)`. Subclasses that share
work through `linearize` still have `at()`.

### `matrix()` with defaults fails on an operator that has no adjoint and a smaller codomain
`pygeoinf2/algebra/operators.py:770`. `by="auto"` chooses rows, which means
adjoint applications, whenever `codomain.dim < domain.dim`, without asking whether
an adjoint exists. A forward operator built with `from_callables` and no adjoint
into a data space smaller than the model space therefore cannot give its matrix
with the defaults, although `by="columns"` works. `assembled()` calls `matrix()`
with the defaults and fails the same way. The error tells the user to opt into
`with_probed_adjoint()`, an extra assembly, for something that needs none.
Verified.
Reproduction:
```python
A = LinearOperator.from_callables(EuclideanSpace(5), EuclideanSpace(3), lambda x: x[:3].copy())
A.matrix()               # NotImplementedError: No adjoint action was supplied ... with_probed_adjoint()
A.matrix(by="columns")   # (3, 5)
```
Proposed fix: under `"auto"`, fall back to columns when the adjoint is not
available. A `has_adjoint` property, or catching the `NotImplementedError` from
the first row, would do.

### `Functional.from_callables(linearize=, hessian=)` claims a Hessian its `at()` does not carry
`pygeoinf2/algebra/operators.py:2431`. When a `linearize` callable is given,
`_CallableFunctional._linearize` returns the caller's `QuadraticModel` unchanged,
even if it has no Hessian and a `hessian=` callable was supplied beside it. So
`has_hessian` is True and `at(x).hessian` is None. Newton-CG takes
`model.hessian` from `at()` and crashes. The docstring presents the two arguments
as independent. Verified.
Reproduction:
```python
f = Functional.from_callables(X, value, linearize=lambda x: QuadraticModel(x, value(x), deriv(x)),
                              hessian=lambda x: LinearOperator.identity(X))
f.has_hessian, f.at(x0).hessian     # True, None
NewtonCG().minimize(f, x0)          # AttributeError: 'NoneType' object has no attribute 'domain'
```
Proposed fix: in `_CallableFunctional._linearize`, attach `self._hessian_fn(x)`
when the returned model has none and a Hessian callable was given (or reject a
model that has none).

### A functional's Hessian is invisible to `second_derivative` and to the checks
`pygeoinf2/algebra/operators.py:2217`, `pygeoinf2/testing.py:965`. `Functional`
keeps its Hessian (`hessian`, `has_hessian`) apart from `Operator`'s
`second_derivative`, and never maps one onto the other. A functional with a
Hessian reports `has_second_derivative` False, and `check_second_derivative`
refuses it. None of the thirteen checks tests a Hessian against differences of
the gradient, although D-168 names exactly that as the check for one. Each caller
writes its own: `tests/test_laplace.py:400`, `tests/test_nonlinear.py:616` and
`tests/test_optimization.py:1057`. The waveform app checks only the Hessian's
symmetry, with `check_traits`. Verified.
Reproduction: for a `Functional.from_callables(..., hessian=...)`, `has_hessian` is
True and `has_second_derivative` is False, and `check_second_derivative(f, x)`
raises `AssertionError: Axiom failed: the operator carries a second derivative.
_CallableFunctional does not`.
Proposed fix: add `check_hessian(functional, point, *, rng, measure, step, rtol, atol)`,
comparing `H d` with central differences of `gradient`, and/or define
`Functional._second_derivative(x, dx)` as the functional `y -> (H dx, y)` when a
Hessian exists.

### A diagonal operator on a non-diagonal metric drops metric-independent traits, and solvers then refuse legitimate operators
`pygeoinf2/algebra/diagonal.py:78`. `_deduce_traits` returns `NONE` whenever the
metric is not diagonal. But invertibility (no zero entry) and idempotence (entries
0 or 1) do not depend on the metric, and a constant spectrum `s I` is self-adjoint
and definite on any metric. As a result `D* C D`, a congruence of a positive
definite `C` by an invertible diagonal on a finite element (dense Gram) space,
earns only PSD, and CG and Cholesky both refuse it. Verified.
Reproduction:
```python
D = DiagonalLinearOperator(X, rng.uniform(1, 2, 5)); C = spd_operator(X, rng)
D.traits, (D.adjoint @ C @ D).traits   # NONE, SELF_ADJOINT|POSITIVE_SEMIDEFINITE
CholeskySolver()(D.adjoint @ C @ D)    # ValueError: CholeskySolver requires Traits.POSITIVE_DEFINITE
DiagonalLinearOperator(X, np.full(4, 2.0)).traits   # NONE, while (2.0 * identity).traits is SA|PD
```
Proposed fix: deduce `INVERTIBLE` and `IDEMPOTENT` from the values on every metric.
Deduce the full set (as `_Identity._combine_scale` does) when all eigenvalues are
equal.

### `AffineOperator` drops out of the second-order and functional algebra
`pygeoinf2/algebra/operators.py:2679`. An affine map has a zero second
derivative, but `AffineOperator` does not override `_second_derivative`.
`F @ (A + b)` therefore loses the exact second derivative that `F @ A` keeps, and
`(phi @ (A + b)).has_hessian` is False while `.hessian(x)` works. Optimizers read
the flag. Separately, `LinearFunctional @ AffineOperator` goes through
`_combine_rcompose` (`:2754`) to an `AffineOperator`, not a `Functional`, so it has
no `gradient` and no quadratic model. The planet-density app composes a nonlinear
forward with `AffineOperator(prior.covariance_factor, reference)` and runs
Gauss-Newton, so it does not hit this, but an exact-Hessian run would. Verified.
Reproduction: `(phi @ A).has_hessian` True and `(phi @ (A + b)).has_hessian`
False. `(F @ A).has_second_derivative` True and `(F @ (A + b))` False.
`type(lf @ (A + b))` is `AffineOperator`, and `.gradient` raises `AttributeError`.
Proposed fix: give `AffineOperator` `_second_derivative` returning `_Zero`
(`has_second_derivative` then follows). In `_combine_rcompose`, return an affine
functional (or decline) when `other` is a `LinearFunctional`.

### Two `dim`-application loops take no `n_jobs=`
`pygeoinf2/algebra/operators.py:668` and `:1200`. `with_probed_adjoint()` assembles
with `matrix(form="components", by="columns")` (`:1973`) and takes no arguments.
That makes it one of the `dim(domain)` loops of independent, possibly PDE-solve,
applications that D-52 lists, with no way to parallelize it. The banded probe of
`diagonals(probe="banded")` loops over the offset span serially, although its
`n_jobs` argument exists and is only used by the exact probe. Read only.
Reproduction: `inspect.signature(LinearOperator.with_probed_adjoint)` is `(self)`.
Proposed fix: `with_probed_adjoint(*, n_jobs=None)`, forwarded to the assembly.
Route the banded residues through `parallel_map`.

### A mass-weighted module's multiplication operator refuses an adjoint the library can lift exactly
`pygeoinf2/algebra/spaces.py:860`. `_MassWeightedModule` and
`_CoordinateMassWeightedModule` delegate `multiply` and `sqrt` but inherit
`HilbertModule.multiplication_operator`, which is `from_callables` with no
adjoint. On `MassWeightedSpace(Sobolev, M)` the operator therefore fails
`check_operator`, although the base's multiplication operator has a lifted
adjoint and `from_formal_adjoint` is the automated lift D-22 calls key. The
`HilbertModule` docstring says no self-adjointness is claimed, but not that the
adjoint is refused altogether. Verified.
Reproduction: `W = MassWeightedSpace(circle.Sobolev(16, 1.5, 0.2), DiagonalLinearOperator(...))`.
Then `check_operator(W.multiplication_operator(f))` raises
`NotImplementedError: No adjoint action was supplied for this operator`.
Proposed fix: on the mass-weighted module, return
`LinearOperator.from_formal_adjoint(self, self, self.base.multiplication_operator(f))`.
State the refusal in `HilbertModule.multiplication_operator`'s docstring.

### The checks do not take or honour `atol=` uniformly
`pygeoinf2/testing.py:701`, `:1138`, `:1278`, `:356`. D-128 asks for `atol=`
wherever vectors are compared, and the brief asks every check for one. Three
checks compare vectors without one: `check_traits` (IDEMPOTENT), `check_projection`
(idempotency) and `check_affine` (the affine identities, which also cannot pass an
`atol` down to its nested `check_operator`). `check_operator`'s docstring says
`atol` applies to "the linearity residual", but the code also applies it to the
adjoint pairing. Defaults differ for the same keyword: `check_space` and
`check_coordinates` default `atol=1e-12` and apply it to scalar pairings in the
units of the inner product, while `check_operator` defaults 0 after D-192.
`check_coordinates` also uses seven bare `np.allclose` (rtol 1e-5, atol 1e-8)
that ignore the caller's `rtol` and `atol` and never enter the report
(`:399`–`:463`). Read only; in my runs the other axioms caught what these let
through (r6.py), so the cost is consistency, not a missed defect.
Reproduction: `inspect.signature(check_traits)` (and of `check_projection` and
`check_affine`) has no `atol`.
Proposed fix: add `atol=` to the three vector-comparing checks and forward it.
Default it to 0 for pairings everywhere. Replace the `np.allclose` calls with
`_assert_close` and `_assert_scalar` at the caller's tolerances.

### `HilbertSpace.mean` breaks both naming rules
`pygeoinf2/algebra/spaces.py:208`. v1 called this `sample_expectation`, and the
preferences are v1's names unless there is a strong reason, and "expectation not
mean". No decision records the rename. Callers are few:
`probability/base.py:147`, `probability/gaussian.py:465`, `testing.py:352` and
three tests. A rename after 2.0 is breaking, so it is worth deciding now. Read
only.
Reproduction: `grep -n sample_expectation pygeoinf/hilbert_space.py` gives line 403.
Proposed fix: rename to `sample_expectation`, or record the reason for `mean` in
DECISIONS.

## Could fix

### `matrix()` and `known_matrix()` hand out the operator's own array
`pygeoinf2/algebra/operators.py:2094`. For a dense `MatrixLinearOperator` asked for
its stored form, `_in_form` returns `self._stored` itself, so mutating the result
of `matrix()` changes the operator. `_ProbedAdjointOperator` does the same with
its cached assembly. `_ColumnOperator` copies. Only `stored_matrix` documents
"read-only", and no library code mutates the result (grep). Verified.
Reproduction: `D = from_matrix(Y, Y, np.eye(3), form="components"); m = D.matrix(); m[0, 0] = 99; D([1, 0, 0])`
gives `[99, 0, 0]`.
Proposed fix: return a copy from `matrix()`, or say "the returned array may be the
stored one; do not modify it" in `matrix` and `known_matrix`.

### The `known_matrix` contract is broken by the probed adjoint, and there is no cheap predicate
`pygeoinf2/algebra/operators.py:1984` and `:990`. `known_matrix` promises to apply
the operator zero times, but `_ProbedAdjointOperator._known_matrix` assembles,
which is `dim` applications, on first asking. Callers also use `known_matrix` as
an "is it free?" predicate and throw the array away
(`numerics/root_find.py:505`, "asking costs no applications"). When the answer is
not None, that predicate forms the dense matrix (densifying a sparse member, or
for a direct solver's inverse computing `A^-1` in full, `numerics/solvers.py:638`).
Verified for the first part, read only for the second.
Reproduction: `LinearOperator.from_callables(E5, E5, act).with_probed_adjoint().known_matrix(form="components")`
makes 5 applications of `act`.
Proposed fix: have `_ProbedAdjointOperator._known_matrix` answer only once it has
assembled. Add `has_known_matrix` (or a `_known_matrix_cost`) for callers that
only need the answer to "is it free?".

### `diagonals()` accepts any `form` despite its docstring
`pygeoinf2/algebra/operators.py:1169`. The docstring promises `ValueError` for an
unknown form, but only `probe` is validated, so an unknown form silently reads
the components diagonal. Verified.
Reproduction: `B.diagonals(form="nonsense")` gives `[[2. 2. 2. 2. 2.]]`.
Proposed fix: validate `form` as `matrix()` does.

### A negative power of a singular diagonal returns infinities and claims definiteness
`pygeoinf2/algebra/diagonal.py:399`. `__pow__` gates only fractional powers.
`D ** -1` with a zero eigenvalue returns `inf` and deduces `SELF_ADJOINT|PSD|PD|INVERTIBLE`
from it. `inverse` raises `ZeroDivisionError`, and its docstring says that
"returning infinities instead would push the failure somewhere further away". The
singular-spectrum errors also differ in class: `ZeroDivisionError` from
`inverse`, `ValueError` from `inverse_sqrt`, `log` and `log_determinant`.
Verified.
Reproduction: `(DiagonalLinearOperator(E3, [0, 1, 2]) ** -1).eigenvalues` gives `[inf 1. 0.5]`.
Proposed fix: gate negative powers on non-zero eigenvalues, and raise one class
for an inadmissible spectrum.

### Empty block constructors fail with `IndexError`
`pygeoinf2/algebra/direct_sum.py:689`, `:771` and the `__init__` of
`BlockDiagonalLinearOperator` (`:859`). `__new__` routes `[]` to
the linear subclass, whose `__init__` indexes `[0]` before the
nonlinear class's `ValueError` check can run. Verified.
Reproduction: `ColumnOperator([])`, `RowOperator([])` and `BlockDiagonalOperator([])`
raise `IndexError: tuple index out of range`, while `BlockOperator([])` raises
`ValueError`.
Proposed fix: check emptiness in the linear `__init__`s, or in `__new__`.

### A mass-weighted coordinate space with a diagonal metric draws white noise through a dense Cholesky
`pygeoinf2/algebra/spaces.py:1073`. `_CoordinateMassWeightedSpace` reports
`has_diagonal_metric` but inherits `CoordinateSpace.white_noise_components`, which
forms the Gram matrix column by column and factors it, O(n³) once and O(n²)
memory. It also inherits the column-by-column `apply_gram_to_columns` and
`solve_gram_to_columns`. Verified at n = 2000 (first draw 0.05 s, then 1.9 ms per
draw); the scaling is read.
Proposed fix: when `has_diagonal_metric`, draw `xi / sqrt(gram_diagonal())` and
broadcast the column forms.

### `from_scipy` refuses an adjoint with scipy's message, not the library's
`pygeoinf2/algebra/operators.py:951`. A scipy operator with only `matvec` gives an
adjoint that raises scipy's `NotImplementedError: rmatvec is not defined`.
`from_callables` raises the library's message naming `with_probed_adjoint()`,
D-34's rule. Verified.
Proposed fix: detect a missing `rmatvec` at construction and pass `adjoint=None`.

### Surface: a numpy name in `__all__`, and an unexported public-looking class
`pygeoinf2/algebra/spaces.py:36` and `:1238`. `spaces.__all__` re-exports numpy's
`default_rng`, which nothing imports from there. `ComponentView` is named as
public but is not exported. It is an implementation detail of the iterative
solvers, and its `is_function_space` is the base `False` even when it views a
function space. Read only.
Proposed fix: drop `default_rng` from `__all__`. Rename `ComponentView` to
`_ComponentView`, or export it and delegate `is_function_space`.

### Row, column and block operators cannot be seated on the caller's direct sum
`pygeoinf2/algebra/direct_sum.py:513`, `:639`, `:717`. D-195 gave
`BlockDiagonalLinearOperator` `domain=` and `codomain=` so that a block diagonal
on a `Layered` space keeps the pointwise product. `BlockOperator`,
`ColumnOperator` and `RowOperator` still build bare sums. That is the
seating side of §7's "a posterior on a direct sum forgets the prior's labels",
which also loses the layered space's module structure. The same `_seat` helper
would close both. I mention it only because the fix already exists one class
over, not to re-report §7. Read only.
Proposed fix: accept `domain=` and `codomain=` on the three constructors, through
`_seat`.

### Block operators and direct-sum projections give no components action
`pygeoinf2/algebra/direct_sum.py:238`, `:513`–`:972`. None of the block classes,
nor `DirectSum.projection`, define `_components_action`, although on a
`CoordinateDirectSum` each is a block or slice of arrays. So a fused product such as
`A Q A* + R` with `Q` a product-measure covariance converts at every block, and
on spectral summands every conversion is a transform. Read only; not measured.
Proposed fix: when every block has a components action, compose them over the
direct sum's `_slices`.

### The checks have no way to measure without judging
`pygeoinf2/testing.py` (all checks). D-128 makes a report available only after a
pass. To read a defect without judging it, the apps pass `rtol=np.inf`
(`miniapps/mfem/box_ladder.py:76`) or `rtol=1.0` (`miniapps/tomography/bent.py:381`),
a convention no docstring states. Read only.
Proposed fix: document `rtol=np.inf` as the "measure only" form, or add
`strict=False`.

### `check_coordinates` forms the dense Gram matrix and its eigendecomposition; the statistical checks take no `n_jobs`
`pygeoinf2/testing.py:398`, `:520`, `:1087`. `check_coordinates` calls
`gram_matrix()` (dim applications of `apply_gram`) and `eigvalsh`, O(n³), and its
docstring does not say so. `check_white_noise` and `check_measure` loop over
20 000 draws serially, although `measure.samples` takes `n_jobs`, against D-52.
Read only.
Proposed fix: say the cost in the docstring, or test the Gram on probes. Add
`n_jobs=` to the two statistical checks.

## What is sound

- **Matrix and diagonal readings.** `matrix()` and `diagonals()` (offsets −1, 0, 1;
  both forms; `by="rows"` and `"columns"`) agree with probing to within 2e-16
  relative on a dense Gram, a diagonal metric, a Euclidean space and a direct sum
  of a dense-metric and a weighted space. The operators covered are diagonal,
  Galerkin-stored dense, sparse Galerkin, rectangular, adjoint nodes,
  compositions with diagonal factors, sums, scalings, block, row, column,
  block-diagonal, `from_vectors`, `coordinate_selection`, `coordinate_projection`
  and `coordinate_inclusion`. The `check_operator` adjoint defect is at most
  7e-14 on all of them.
- **Block application.** `apply_block` and `_adjoint_apply_block` agree with
  single applications (≤ 1e-14) for compositions, sums, scalings, column
  operators and block diagonals on a dense metric.
- **Lifts and the scipy bridge.** `from_formal_adjoint` is correct by both routes:
  coordinate (mass-weighted over an SPD mass, a codomain lift, and a direct sum of
  a weighted and a Euclidean space) and coordinate-free (`MassWeightedSpace` over
  `OpaqueSpace` with a CG mass inverse). All defects are ≤ 1e-14. `as_scipy` and
  `from_scipy` round-trip on a dense metric in both forms, and `rmatvec` is the
  transpose of the form's matrix to 4e-16.
- **Spaces.** All three `MassWeightedSpace` variants pass `check_space`,
  `check_coordinates` and `check_white_noise`. A labelled `CoordinateDirectSum`
  of a Sobolev circle and a Euclidean space passes the same three, `rebuild`
  included, and its projections and inclusions are exactly adjoint.
  `is_function_space` propagates through sums and mass weighting.
- **Traits.** Palindrome recovery (`L L*`, `A C A*`, `U U*`, `2 A* A`) and
  the solvers' trait gates behave as D-36 says. The rules in `traits.py` (sum,
  scale, adjoint, compose, inverse, congruence, gramian, closure) were read
  against D-36 and found correct.
- **Nonlinear algebra.** The Hessian of `phi @ F` with a curved `F` between
  dense-metric spaces matches differences of the gradient to 7e-11 and is
  symmetric. The derivative, gradient and second-derivative checks pass on it.
- **Parallelism and pickling.** `matrix(n_jobs=2)` and `diagonals(n_jobs=2)` on
  a closure-built operator agree with serial runs. These objects pickle:
  `DirectSum`, `MassWeightedSpace`, diagonal, composition, adjoint and
  probed-adjoint operators, `LinearFunctional` and `AffineOperator`.
- **Code practice.** No decision numbers, history or v1 talk anywhere in the
  layer (grep), and identifiers use American spelling.
- **The layer's tests.** `test_spaces`, `test_operators`, `test_direct_sum`,
  `test_traits`, `test_known_structure`, `test_coordinate_free`,
  `test_parallel_knobs`, `test_code_practice`, `test_no_array_backend`,
  `test_nonlinear` and `test_gradients` all pass: 836 tests.
- **Not covered.** I did not exercise `compat`, the `Layered` internals beyond
  sampling, or the geometry and inference consumers beyond the call counts
  above.

## Fixed

Package E, 2026-10-29. Tests named by file and class under `pygeoinf2/tests/`.

**Must fix**

- **Block operators cannot be unpickled.** `algebra/direct_sum.py`: the
  `__new__` of `BlockOperator`, `ColumnOperator`, `RowOperator` and
  `BlockDiagonalOperator` takes its first argument as optional, as
  `DirectSum.__new__` does, so the linear subclasses unpickle too; the same
  for `SpectralBlockLinearOperator.__new__` in `symmetric_space/base.py`.
  The comment in `LinearOperator.with_traits` no longer gives the old
  reason. Tests: `test_direct_sum.py::TestBlockOperatorsPickle` (all eight
  classes, and `samples(4, n_jobs=2)` on a `from_product`),
  `test_layered.py::TestALayeredPriorCrossesProcesses` (a
  `Layered.sobolev_measure` round trip, its draws and a posterior's in
  worker processes), `test_spectral.py::TestACorrelatedMeasureCrossesProcesses`
  (the spectral block operator, a `correlated_measure` and a posterior).
- **`parallel_map(threads=)`.** Removed from `parallel.py`, with its branch
  and its docstring's mention. Test:
  `test_parallel_knobs.py::TestTheMapRunsProcesses`.
- **`AffineOperator` and a callable translation.** `algebra/operators.py`:
  `AffineOperator(A, b, *, lazy=False)`; the translation is a vector unless
  `lazy=True`, which takes a zero-argument callable and refuses anything
  else with `TypeError`. No vector is tested with `callable` any more.
  Callers that pass a thunk say so: `LinearGaussianInversion`
  (`inference/gaussian.py`), and `LinearPointEstimator`
  (`inference/estimators.py`), whose own `translation=` had the same
  ambiguity and now takes `lazy=` too, passed by `push_forward`,
  `LeastSquares`, `ConstrainedLeastSquares` (`inference/point.py`) and
  `BackusGilbert` (`inference/backus.py`). Tests:
  `test_nonlinear.py::TestAffineOperator::test_a_callable_vector_is_a_translation_not_a_thunk`
  (a space whose vectors are callable fields) and
  `::test_a_lazy_translation_must_be_callable`.
- **`HilbertSpace.mean` becomes `sample_expectation`**, no alias
  (`algebra/spaces.py`); callers in `probability/base.py`,
  `probability/gaussian.py`, `testing.py` and five tests updated, and
  `CORE_VECTOR_API` in `test_code_practice.py`. Test:
  `test_spaces.py::test_sample_expectation` (asserts `mean` is gone).

**Should fix**

- **`check_operator` compares the shortcuts.** `testing.py`
  `_check_shortcuts`: the components action and the adjoint's against
  `A x` and `A* y`, and an overridden `apply_block` or
  `_adjoint_apply_block` against single applications, each recorded in the
  report, on the probe pairs the adjoint identity used, so the check draws
  nothing more from the caller's generator. Test:
  `test_operators.py::TestTheShortcutsAreChecked`.
- **`check_traits` and `INVERTIBLE`.** `testing.py`: decided from
  `_known_scaling()` (no zero entry) where it exists, by `matrix_rank` only
  on a `MatrixLinearOperator` storing a dense array, otherwise left
  unverified; the docstring says so. Test:
  `test_operators.py::TestTheInvertibilityClaimIsCheckedOnlyWhereCheap`
  (no dense rank for a diagonal, a composition of scalings or a sparse
  array; a zero scaling and a singular dense array still caught).
- **`D + I` stays diagonal.** `algebra/diagonal.py` `_combine_add` folds the
  identity; `algebra/nodes.py` `_Sum._known_scaling` sums the terms'.
  Test: `test_known_structure.py::TestASumWithTheIdentityStaysDiagonal`.
- **Compositions densify sparse factors.** A new hook
  `LinearOperator._known_sparse()` (`algebra/operators.py`), answered by a
  sparse `MatrixLinearOperator` (components form, or Galerkin on a diagonal
  metric), by `_Adjoint` on diagonal metrics and by `_Scaled`
  (`algebra/nodes.py`); `_Composition._known_factors` keeps such factors
  sparse, and `_known_matrix` and `_known_diagonals` combine them through
  `_chain_product` and `_diagonal_of_product`, weighing a sparse factor by
  its stored entries. Test:
  `test_known_structure.py::TestASparseFactorIsNeverDensified` (the
  diagonal of `A* R A` with a tridiagonal `A` at dimension 5000, with
  densifying made to fail; both stored forms on a weighted space).
- **The Galerkin diagonal on a diagonal metric.** `_Composition._known_diagonals`
  multiplies the components diagonal by `gram_diagonal()`;
  `_Adjoint._known_diagonals` gives the components form by a row division
  there. Test: `test_known_structure.py::TestTheGalerkinDiagonalOnADiagonalMetric`
  (Jacobi on a normal operator on a weighted space costs no application).
- **Scaling or summing a functional re-linearizes.** `algebra/nodes.py`
  `_FunctionalSum._linearize` and `_FunctionalScaled._linearize` build the
  Hessian from the models in hand (`_model_hessian`), asking a term afresh
  only when its model has none. Test:
  `test_nonlinear.py::TestTheFunctionalAlgebraLinearizesOnce` (one
  linearization for `phi`, `0.5 phi`, `phi + reg`, `0.5 phi + reg` and
  `-0.5 (phi + reg)`, the shape of `log_posterior`).
- **Nonlinear block operators evaluate twice in `at()`.** `direct_sum.py`:
  a `_linearize` on each of the four nonlinear classes, one `at()` per
  block; the linear subclasses keep `LinearOperator`'s `_linearize` and
  `_derivative`, so a linear block operator is its own derivative rather
  than a rebuilt one. Test:
  `test_direct_sum.py::TestANonlinearBlockIsLinearizedOnce`.
- **`matrix()` without an adjoint.** `LinearOperator.matrix` under
  `by="auto"` falls back to columns when the first row finds no adjoint.
  Test: `test_operators.py::TestAMatrixWithoutAnAdjoint`;
  `TestProbedAdjoint::test_a_wide_operator_gets_its_matrix_by_rows_after_opting_in`
  updated to the new default.
- **`from_callables(linearize=, hessian=)`.** `_CallableFunctional._linearize`
  attaches the supplied Hessian to a model returned without one. Test:
  `test_nonlinear.py::TestALinearizationAndAHessianSuppliedApart` (Newton-CG
  runs on it).
- **A check for a functional's Hessian.** `testing.check_hessian` (in
  `__all__`): `H d` against central differences of the gradient,
  self-adjointness in the space's inner product, and the quadratic model
  carrying the Hessian. Test: `test_nonlinear.py::TestCheckHessian` (a
  Hessian missing the inverse metric fails on a dense metric).
- **A diagonal on a dense metric drops metric-independent traits.**
  `DiagonalLinearOperator._deduce_traits` deduces `INVERTIBLE` and
  `IDEMPOTENT` on any metric and the full set for a constant spectrum.
  Test: `test_operators.py::TestADiagonalOnADenseMetric` (the congruence
  `D* C D` is positive definite and Cholesky solves it); the fixture of
  `test_functional_calculus.py::TestTheDiagonalCalculusOnANonDiagonalMetric`
  now asserts no symmetry or definiteness rather than no trait at all.
- **`AffineOperator` in the second-order and functional algebra.**
  `_second_derivative` returns zero; `_combine_rcompose` declines a linear
  functional, which then composes to a functional. Test:
  `test_nonlinear.py::TestAnAffineMapInTheSecondOrderAlgebra`.
- **Two `dim` loops without `n_jobs=`.** `with_probed_adjoint(*, n_jobs=)`
  forwarded to its assembly; the banded probe of `diagonals` goes through
  `parallel_map`. Test: `test_parallel_knobs.py::TestTheAlgebrasProbes`.
- **The mass-weighted module's multiplication operator.**
  `_MassWeightedModule.multiplication_operator` lifts the base's with
  `from_formal_adjoint`; `HilbertModule.multiplication_operator`'s
  docstring states that the default supplies no adjoint. Test:
  `test_spaces.py::TestAMassWeightedModuleMultiplies`.
- **`atol=` across the checks.** `check_traits`, `check_projection` and
  `check_affine` take `atol=` (default 0) and apply it to every comparison,
  `check_affine` forwarding it to its nested `check_operator`;
  `check_coordinates` compares arrays by `_assert_arrays_close` at the
  caller's `rtol` and `atol`, recording each identity, in place of seven
  bare `np.allclose`; `check_operator`'s docstring says `atol` covers every
  comparison. Test: `test_operators.py::TestEveryCheckTakesBothTolerances`.

**Could fix, done because they were one-liners**

- `diagonals()` validates `form` (`test_operators.py::TestDiagonalsRefuseAnUnknownForm`).
- A negative power of a singular diagonal raises `ValueError`
  (`test_operators.py::TestANegativePowerOfASingularDiagonal`).
- Empty column, row and block-diagonal operators raise `ValueError`
  (`test_direct_sum.py::TestAnEmptyBlockOperatorIsRefusedByName`).
- `default_rng` dropped from `algebra.spaces.__all__`.
- `matrix()` and `known_matrix()` say the array may be the stored one.

## Skipped

- **`check_measure` and the precision**: done by package A.
- **`Functional.gradient` evaluating value and Hessian**: not clear-cut.
  For a composed functional `derivative(x)` needs the inner map's value
  anyway, and with an inner operator built from one `linearize` callable
  it pays two forward evaluations where `at()` pays one; which cost model
  `gradient` follows is a decision.
- **One default for `atol`**: `check_space` and `check_coordinates` keep
  `atol=1e-12`, the rest default 0; changing a default tolerance under every
  backend's ladder is a policy call.
- **Could**: the probed adjoint's `known_matrix` and a cheap
  `has_known_matrix` predicate (new API); `from_scipy`'s adjoint message;
  the mass-weighted coordinate space's white noise on a diagonal metric;
  `domain=`/`codomain=` on block, row and column operators (new API); block
  operators' components action (performance, unmeasured); a measure-only
  form of the checks (new API); `n_jobs=` on the statistical checks (new
  API); `ComponentView` (in the numerics package's solvers); the
  `ZeroDivisionError` of `DiagonalLinearOperator.inverse` against
  `ValueError` elsewhere (an exception class change).
