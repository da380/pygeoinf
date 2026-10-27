# Code issues found during the docstring review, 2026-10-26

Seen while the docstrings were being checked against the code. None of them
is fixed: the review changed no code. Each entry gives the place, what is
wrong, why it matters and a suggested fix. "Verified" means it was
reproduced in a few lines of Python during the review. Where a docstring
disagreed with the code, the docstring was made to describe what the code
does, and the disagreement is logged here when the code looks like the part
that is wrong.

9 bugs open and 1 fixed, 13 risks, 46 cleanups, 1 suggestion.

## Bugs

| where | what |
|---|---|
| `inference/backus.py` `_ClosedFormRoute` | the closed form ignores the prior ball's centre: an off-centre `Ball` prior with data `A c` is reported empty, though `c` fits (re-verified) |
| `numerics/convex.py` `PrimalKKTSolver` | ignores the noise set's centre; an off-centre noise ball gives a wrong maximum marked converged (4.4072 against 4.4903). Reaches `BackusGilbertParker.extremal_model` |
| `inference/backus.py` `BackusGilbertParker` `route="auto"` | exact data with an `Ellipsoid` prior picks bisection, which refuses the zero-radius noise ball, so construction raises |
| `inference/backus.py` `_BisectionRoute` | on an overdetermined problem (`N_d > N_m`, the prior slack at the optimum) `_fit_norm`'s bisection widens to a damping of zero and raises `ZeroDivisionError` in `_state`, or returns a support value off by about `6e-5` relative (found 2026-10-27 in the Stage 0.1 experiment, `benchmarks/`; the replacement route evaluates `γ = 0` explicitly, D-153). Not fixed: the route is being replaced |
| `symmetric_space/box.py` `Box.support_projection` | on a Sobolev box the operator's domain is the Lebesgue space: it cannot be composed with operators on the space, and its adjoint is wrong (re-verified) |
| `inference/problem.py` `joint_measure`, `synthetic_model_and_data` | typed for any prior, but fail with `AttributeError` on a non-Gaussian one (a `GaussianMixture`) when there is an error measure |
| `numerics/preconditioners.py` `LocalisedPreconditioner` (moved from `inference` 2026-10-27) | blocks that do not cover every index and no noise term give a singular factorization, though the docstring allows partial covers |
| `numerics/randomized.py` adaptive `random_range` | a zero operator on a coordinate space raises `AttributeError` |
| `numerics/optimization.py` `TrustRegionNewton` | a collapsed trust region is reported as "iteration limit reached" with `max_iterations` as the count |
| `plotting/fourier.py` 1-D `plot_balls` | a ball wider than the circle is drawn as two slivers at the ends |

## Fixed

| where | what |
|---|---|
| `algebra/spaces.py` `is_function_space` | was defined on `CoordinateSpace`, not `HilbertSpace`, so any density call on a coordinate-free space raised `AttributeError`. Moved to `HilbertSpace`, with the mass-weighted delegation moved to `MassWeightedSpace`; tested (2026-10-26) |

## Risks

| where | what |
|---|---|
| `inference/backus.py` `BackusGilbert`, `_minimum_norm_fits`, `_DualRoute.is_feasible` | the same blindness to ball centres, including the error-mean centre of the hardened ambient ball |
| `probability/gaussian.py` `from_standard_deviations` | on a non-diagonal metric the covariance is not `diag(sigma^2)` (verified) |
| `probability/gaussian.py` `from_covariance_matrix` | `form` defaults to `"galerkin"`, against the rule that matrix construction names its form |
| `numerics/solvers.py` `LSQRSolver` | `strict=False` returns a non-converged solve without the warning every other iterative solver gives (verified) |
| `backends/mfem.py`, `backends/mfem_hilbert.py` | non-convergence raises a bare `RuntimeError`, not `ConvergenceError` |
| `sem1d/__init__.py` import guard | accepts planetmodel 1.2, under which every ball fails later with a misleading grid error; the package needs 1.3 |
| `testing.py` `check_affine` | the nested `check_operator` gets neither the caller's `rtol` nor its `measure` |
| `geometry/convex.py` `_OracleSet.maximizer`, `Ellipsoid.support_maximizer` | raise `AttributeError` where the base contract is `NotImplementedError` |
| `symmetric_space/base.py` `estimate_truncation_degree` | the v1 name now means a different question; v1's is `sufficient_degree`, so a v1-style call runs with a different meaning |
| `numerics/randomized.py` `random_diagonal` | `samples=0` returns an all-NaN diagonal with no error |
| `numerics/convex.py` `ProximalBundleMethod._simplex_weights` | the warm start is misaligned once the bundle is full (performance only) |
| `parallel.py` `_context_is_set` | reads private joblib internals |

The cleanups (dead code, wrong annotations, missing `Raises:` on
constructors, error messages that cite decisions, and similar) are in the
detail below, with the risks and bugs, file by file.

## In detail, file by file

### algebra/spaces.py

- `CoordinateSpace.is_function_space` (**fixed 2026-10-26**; was a verified bug): the property is defined on `CoordinateSpace`, not on `HilbertSpace`, though D-131 says "`HilbertSpace.is_function_space` ... `False` at the base". `probability/base.py::_require_density` reads `space.is_function_space` unguarded, and `DirectSum.is_function_space` reads it from every summand. So a measure on a coordinate-free space (a plain `MassWeightedSpace` over a non-coordinate base, a user backend implementing `HilbertSpace` only) raises `AttributeError` from `log_density` (reproduced: `GaussianMeasure(OpaqueSpace(3), covariance=I, precision=I).log_density(x)` gives `AttributeError: 'OpaqueSpace' object has no attribute 'is_function_space'`) and the rest of the density family, and so does a `DirectSum` with such a summand. Fix: move the property (returning `False`) up to `HilbertSpace`; `MassWeightedSpace` (not only `_CoordinateMassWeightedSpace`) should delegate to its base.
  Fixed: the property is now on `HilbertSpace` (`False`), and the base delegation moved from `_CoordinateMassWeightedSpace` up to `MassWeightedSpace`, so every variant reads its base. Two tests in `test_probability.py` (`TestTheFunctionSpaceMarkerGatesDensities`) cover a coordinate-free space, a direct sum and a mass-weighted space over one; both fail without the fix.

### algebra/operators.py

- `if TYPE_CHECKING: pass` near the top (cleanup): dead block. `TYPE_CHECKING` is imported for nothing else. Delete both.

### algebra/nodes.py

- `_Identity._combine_scale` (cleanup): re-imports `numpy as np` locally, though the module already imports it, and imports `CoordinateSpace` as `..algebra.spaces` from inside `algebra` where everywhere else uses `.spaces`. Harmless; tidy when next touched.

### algebra/direct_sum.py

- no issues found.

### algebra/diagonal.py

- no issues found.

### algebra/linearization.py

- no issues found.

### algebra/__init__.py

- `EvaluationError` missing (cleanup): it is in `algebra.operators.__all__` and in the top-level `pygeoinf2.__all__`, but `pygeoinf2.algebra` neither imports nor exports it. `from pygeoinf2.algebra import EvaluationError` fails. Add it to the import list and to `__all__`.

### traits.py

- no issues found.

### parallel.py

- `_context_is_set` (risk): reads joblib internals (`joblib.parallel._backend`, `default_parallel_config`), which have no stability guarantee. A joblib upgrade that renames them breaks every parallel loop with an `AttributeError`/`ImportError`, or worse, silently changes whether the one-thread default applies. Worth a test pinned to the joblib version (the existing worker-environment test may already catch the second case), or a guarded fallback.

### compat.py

- no code issues. The docstring of `adapt_space` contradicted the code; the docstring was the wrong one and is fixed (see ledger).

### __init__.py

- `__all__` "subpackages" (cleanup): lists `geometry`, `inference`, `numerics`, `plotting`, `probability`, `symmetric_space` but not `algebra` or `sem1d`, both of which are loaded by `import pygeoinf2` (sem1d when planetmodel is installed). `gi.algebra` works as an attribute, but `from pygeoinf2 import *` does not bind it. Decide whether `algebra` belongs in the list.

### datasets.py

- no issues found.

### data/__init__.py

- no issues found.

### numerics/__init__.py

(none)

### numerics/solvers.py

- `LSQRSolver._invert` (risk, verified): with `strict=False` a non-converged solve returns silently, with no warning, though the `strict` docstring says "raise `ConvergenceError` rather than warn" and every `IterativeSolver` warns through `_finish`. Reproduced: `LSQRSolver(max_iterations=1, strict=False)(A).solve(b)` gives `converged=False` and no warning. Fix: emit the same `RuntimeWarning` as `IterativeSolver._finish` when `not result.converged and not self._strict`.
- `LSQRSolver` (cleanup): unlike the other iterative solvers, LSQR records no iteration-0 entry in `history`/the callback, and does not refuse a non-finite residual (`_note` lacks `_record`'s NaN check). So a NaN-producing operator runs LSQR to its cap (`4 * dim`) and is reported as non-convergence. Fix: share `IterativeSolver._record`'s finiteness check.
- `MinResSolver._solve` (cleanup): an indefinite preconditioner raises `ValueError` from the inner `m_norm`, while CG reports the analogous failure (`(r, P r) <= 0`) as `ConvergenceError`. Two exception types for one kind of failure. Consider `ConvergenceError` for consistency.

### numerics/preconditioners.py

- `__all__` (cleanup): `sparse_approximation` is defined here and re-exported by `numerics/__init__.py`, but is missing from this module's `__all__`, so `from pygeoinf2.numerics.preconditioners import *` drops it. Fix: add it.
- `ColumnThresholdedPreconditioner._invert` (cleanup): `dimension = domain.dim` followed by `del dimension` is dead code. Remove both lines.
- `ColumnThresholdedPreconditioner._invert` (cleanup): its `SolveResult` reports 0 iterations, where the banded and block preconditioners report 1 for the same kind of single sparse solve. Cosmetic, but `SolveResult.iterations` should mean one thing.
- `ColumnThresholdedPreconditioner._keep` and `sparse_approximation.keep` (cleanup): the same column test and capping logic is written twice (the "column" criterion of `sparse_approximation` is `_keep`). One shared helper would keep them from drifting.

### numerics/randomized.py

- `_adaptive_range_on_components` (bug, verified): when the first block of probes has zero norm (a zero operator) it returns `[]`, not an empty `(dim, 0)` array, and its callers then read `.shape`. `random_range(Z, rng=...)` with `Z` the zero operator on `EuclideanSpace(6)` raises `AttributeError: 'list' object has no attribute 'shape'`. The coordinate-free branch returns `[]` correctly, so only coordinate spaces hit this. The same path is behind `random_eig`/`random_svd`/`random_cholesky` with `rank=None`, which should instead reach their "range appears to be trivial" `ValueError`. Fix: `return np.zeros((codomain.dim, 0))`.
- `random_diagonal` (risk): `samples` is not validated. `samples=0` divides zero by zero and returns an all-NaN diagonal with no error, and a `JacobiPreconditioner(samples=...)` passes its own check (`samples >= 1`) but `deflated_diagonal` and direct callers do not. Fix: raise `ValueError` for `samples < 1`, as `random_trace` does for `samples < 2`.
- `random_range` (cleanup): `block_size` is not validated, though `Raises:` says a non-positive block size raises. With `block_size=0` on the adaptive route, the first block is empty and is treated like a zero operator: the coordinate-free loop returns an empty basis, and the component loop hits the `AttributeError` above. Fix: validate `block_size >= 1` (and `0 < rtol < 1`, which `Raises:` also promises but nothing checks).

### numerics/functional_calculus.py

(none found)

### numerics/quadratic_forms.py

- `weighted_chi2_cdf` (cleanup): an unknown `method` is refused only when the weights are unequal and `value > 0`. With equal weights or a non-positive value the function returns before the method is checked, so a misspelt method (`"imhoff"`) passes silently there, though `Raises:` promises a `ValueError`. `weighted_chi2_quantile` has the same gap for equal weights. Fix: validate `method` first.
- tests/test_code_practice.py `TestImportCost` (stale docstring, outside this group): its docstring says `pygeoinf2.inference.problem` "still imports `scipy.stats` eagerly"; it no longer does. Belongs to whoever reviews the tests.

### numerics/line_search.py

(none)

### numerics/root_find.py

(none)

### numerics/quadratic_programming.py

(none)

### numerics/optimization.py

- `TrustRegionNewton._minimize` (bug, minor): the three `break`s taken when the radius collapses below 1e-14 fall through to the final return, which reports `iterations=self._max_iterations` and the message "iteration limit reached". A run that stopped early because the trust region shrank to nothing is misreported as having hit the cap. Fix: return a result with the actual iteration count and a message such as "trust region collapsed" (and `converged` from `_precision_limited`, as the line-search methods do).
- `TrustRegionNewton._minimize` (cleanup): ignores `ftol` and the precision-limited outcome, though `Optimizer.__init__` documents `ftol` for every optimizer. The docstring now says so; either apply the value test here too or keep the exception documented.

### numerics/convex.py

- `PrimalKKTSolver` (bug, verified): the noise set's center is never read. `_quadratic` returns it, `_noise_parts[3]` is unused, and both `solve`'s slack test and `_model` use `G m - d` as if the set were centered at zero. An off-center noise ball gives a wrong maximum reported as converged. Reproduced with `EuclideanSpace(6)` model, `EuclideanSpace(3)` data, prior `Ball(radius=1.5)`, noise `Ball(center=c, radius=0.2)` with `c = [0.3, -0.2, 0.1]`: `PrimalKKTSolver` returns 4.407239 (`converged=True`), while the equivalent centered problem (data `d - c`) and `ChambollePockSolver` both give 4.490298. `LevelKKTSolver` handles the center correctly through the level function. `BackusGilbertParker._kkt_solver` hands its confidence set straight in, so `extremal_model` with an off-center error ball is affected. Fix: subtract the noise center from the data in `PrimalKKTSolver.__init__` (`self._data = data - center`), or refuse a nonzero center. The docstring now states the requirement.
- `ProximalBundleMethod.qp_solver` (cleanup): dead code after the `return`. The `self._cache` initialisation with its comment sits unreachable at the end of the property; it was evidently meant for `__init__`. Harmless, because `minimize` resets `_cache` before `_gram` is used, but `_cache` is undefined until then. Move it to `__init__`.
- `ProximalBundleMethod._simplex_weights` (risk, performance only): the warm start is misaligned once the bundle is full. At capacity a cut is appended and the oldest popped, so the size is unchanged, `self._warm.size == size`, and the first branch pads the previous weights without shifting them. The `elif` branch that shifts "when the bundle dropped its oldest cut" is unreachable (sizes differ only by 0 or 1). Correctness is unaffected (the result is solved and projected; Clarabel ignores `x0`), but OSQP gets a poor start. Fix: track whether a cut was dropped, not the size difference.
- `PrimalKKTSolver._quadratic` (cleanup): reads the private `given._covariance` and `given._precision` of `Ellipsoid`. A public accessor would be safer.

### geometry/convex.py

- `_OracleSet.maximizer` and `Ellipsoid.support_maximizer` (risk): both raise `AttributeError` when the maximizer or covariance is missing, while the base `ConvexSet.support_maximizer` raises `NotImplementedError` and both docstrings used to document `NotImplementedError` (`Ellipsoid`'s said `ValueError`). The docstrings now document the `AttributeError` the code actually raises; change them back when the code is fixed. A caller that handles the documented `NotImplementedError` (the base contract, which generic code over `ConvexSet` will rely on) does not catch it, and an `AttributeError` from a method call reads as a missing attribute. Nothing in the library catches either around these calls today, since callers ask `has_maximizer` first. Fix: raise `NotImplementedError` at both sites.
- `Polytope.contains` (cleanup): `plane.contains(x, rtol=rtol) if hasattr(plane, "contains") else False`. Every half-space is a `ConvexSet`, and `__init__` does not check the type, so the `hasattr` branch either never fires or silently answers "not contained" for a non-set argument. Fix: check `isinstance(plane, HalfSpace)` in `__init__` and drop the `hasattr`.
- `Polytope.__init__`, `Hyperplane.__init__`, `HalfSpace.__init__`, `BallSurface.__init__`, `EllipsoidSurface.__init__` (cleanup): they raise `ValueError` but their docstrings have no `Raises:` section. The code-practice test does not catch this because it skips dunder methods. Not changed: adding sections there is documentation, but outside the narrow brief; worth a sweep.

### geometry/subspaces.py

- No code issues found.

### geometry/sets.py

- `Intersection.__init__`, `Union.__init__` (cleanup): they raise `ValueError` on an empty list without a `Raises:` section (dunders are exempt from the house rule).

### testing.py

- `check_derivative`, `check_gradient`, `check_second_derivative` (cleanup): annotated `-> None` but return a `CheckReport`, as their docstrings and every other check say. Fix: annotate `-> CheckReport`.
- `check_affine` (risk): the nested `check_operator(linear, rng=rng, trials=trials)` passes neither the caller's `rtol` nor `measure`. The linear part is therefore always held to `check_operator`'s default `1e-9`, tighter than `check_affine`'s own `1e-8` default, and probed with `random` even when the caller asked for draws from a measure. An inexact backend that passes `rtol=1e-4` to `check_affine` still fails on the adjoint at `1e-9`. Fix: forward `rtol=rtol, measure=measure`.
- `check_coordinates` (cleanup): the nested `check_representer(space, cx, rng=rng, rtol=rtol)` does not forward `measure`, so the pairing probes ignore it.
- `check_projection` (cleanup): the membership tolerances are hard-coded (`rtol=1e-6` for "lands in the set", `1e-9` for the nearby points) and do not follow the `rtol` argument, which governs only idempotency. Consider exposing or documenting them.

### plotting/sets.py

- `_support_hull_3d` (cleanup): `except Exception` around `scipy.spatial.ConvexHull` turns any failure, including a bad maximizer return value, into "The supporting points span no volume". Fix: catch `scipy.spatial.QhullError` only.

### probability/base.py

- `_require_density`, error message (cleanup): the user-facing ValueError ends "... vanishes in the limit (D-131)." A user cannot resolve the number. Suggest dropping "(D-131)"; the rest of the message already explains the refusal.
- `ProbabilityMeasure.two_point_covariance` (cleanup, from reading): on a measure with no covariance (`self.covariance is None`) the last line calls `None(...)`, which raises `TypeError: 'NoneType' object is not callable`. Suggest checking first and raising ValueError as `directional_covariance` does.

### probability/absolutely_continuous.py

- `AbsolutelyContinuousMeasure.relative_to`, error message (cleanup): the ValueError ends "... defined by this construction (D-132)." Suggest dropping the decision number.

### probability/gaussian.py

- `GaussianMeasure.kl_divergence_estimate`, the "auto" refusal message (cleanup, stale): it justifies the refusal with "on a Sobolev spectrum it has returned -88.6 +/- 21.7 for a divergence whose true value is zero". That number came from the retired three-estimate route. The current folded route is non-negative by construction, so the claim is no longer possible, and the docstring now gives the real reason (an estimate reported as a bare number misleads). Suggest rewording the message to that reason.
- `GaussianMeasure.from_covariance_matrix` (risk): `form` defaults to `"galerkin"`. That contradicts the rule that matrix construction names its form (D-4, D-25: "no trait implies the form on construction, so `from_matrix` requires it"). The old docstring said "No default: guessing wrong is a silent error of one factor of the Gram matrix", so the default looks unintended. `with_dense_covariance` calls it without `form=`, passing a Galerkin matrix. Suggest making `form` required, as `LinearOperator.from_matrix` has it, and passing `form="galerkin"` at the internal call. The docstring was changed to describe the current default.
- `GaussianMeasure.from_standard_deviations` (risk, verified): on a space whose metric is not diagonal, the factor `DiagonalLinearOperator(domain, sigma)` is not self-adjoint, so the covariance `L L*` is not `diag(sigma^2)`. On `make_dense_metric_space(4)` with `sigma = [1, 2, 3, 4]`, the component matrix has off-diagonal entries up to 1.45 and a diagonal of 0.96, 3.71, 9.28, 16.39. The measure is self-consistent (its precision factor `diag(1/sigma)` inverts this `L L*`), but the old docstring promised `diag(sigma^2)`. Decide whether that is intended: if "a variance per basis direction" is wanted on non-diagonal metrics, a different construction is needed; otherwise refuse or warn when `not domain.has_diagonal_metric`. The docstring now describes the current behaviour.
- `GaussianMeasure.rescale_directional_variance` (cleanup): `standard_deviation` is not validated. Zero silently gives a point mass and a negative value is accepted. The old docstring claimed a non-positive target was refused. Suggest raising ValueError for `standard_deviation <= 0`, then restoring that sentence in Raises.
- `GaussianMeasure._combine_affine` (cleanup): annotated `-> GaussianMeasure[X]` but returns `None` for a precision-only measure under a map that does not preserve the precision. The annotation should be `GaussianMeasure[X] | None`, matching the base hook.
- `GaussianMeasure.mahalanobis_squared` / `grad_log_density` (cleanup): a missing precision raises NotImplementedError, while the parallel "missing covariance" case (`_require_covariance`) raises ValueError. Both mean "this measure lacks the ingredient", so one exception type would be easier to catch. Not changed; the docstrings now state NotImplementedError, which is what the code raises.

### probability/mixture.py

- `GaussianMixture.covariance` (cleanup, from reading): a component with no covariance operator (precision-only) makes `weight * component.covariance` raise an unhelpful `TypeError` (float times None). Suggest checking `has_covariance` on the live components and raising ValueError naming the component.

### plotting/distributions.py

(none)

### plotting/base.py

(none)

### inference/backus.py

- `BackusGilbertParker._choose` and `_choose_membership` with `"auto"` (bug, verified): with exact data (no error, so the confidence set is `Ball(radius=0)`) and an `Ellipsoid` prior, `_quadratic` is true, so `"auto"` picks `"bisection"` and membership `"reduced"`. Both build a `_BisectionRoute`, which refuses a zero-radius noise ball. Reproduced: `BackusGilbertParker(LinearForwardProblem(A), T, Ellipsoid(M, I, covariance=I))` raises `ValueError: The noise needs a positive radius for this route.` at construction. The forced routes already check `self._exact`; the auto branches do not. Fix: use `self._quadratic and not self._exact` in both auto branches, so exact data with an ellipsoid prior falls through to KKT, dual or likelihood (or generalise the closed form to ellipsoids).
- `_ClosedFormRoute` via `BackusGilbertParker` (bug, verified): the closed form ignores the prior ball's centre. `_ball_radius` reads only the radius, and `_ClosedFormRoute` assumes a ball centred at the origin (its private Args say so), but `BackusGilbertParker` sends any `Ball` with exact data there. Reproduced: with `prior = Ball(M, radius=1.0, center=c)` and data `d = A(c)`, `BackusGilbertParker(problem, T, prior)(d).is_empty()` is `True`, although `c` itself fits the data and lies in the prior. Fix: shift the data by `A c` and the answer by `T c`, or refuse a ball that is not centred on this route.
- `BackusGilbert`, `_minimum_norm_fits`, `_DualRoute.is_feasible` (risk): the same blindness to centres. `BackusGilbert` uses only the two radii, yet the noise ball `harden_error` returns is `ambient_ball`, centred at the error's expectation, so a non-zero-mean error is silently ignored; a ball prior's centre is ignored too. `_minimum_norm_fits`, which `_DualRoute.is_feasible` uses for two balls, assumes both are centred at zero. `_BisectionRoute` handles centres correctly. The docstrings now say the balls are taken as centred. Fix: subtract the centres, as `_BisectionRoute._shifted` does, or refuse off-centre balls.
- `harden_error` (cleanup): the user-facing `TypeError` says "a general convex one needs route (d). Pass noise= explicitly to bound it." "Route (d)" is the retired journal's lettering, and `harden_error` takes no `noise=`, only its callers do. Suggest: "... needs BackusGilbertParker's dual route; pass the set as the confidence set."
- `BackusGilbertParker._need_inclusion` and the forced `membership="likelihood"` refusal (cleanup): both messages say membership "needs a ball prior", but the reduced and likelihood engines accept ellipsoid and sublevel-set priors (`_likelihood_applies` checks only for level functions). The messages should name what is actually required.
- `_LikelihoodRoute.fitting_model` (cleanup): `if self._likelihood(data) <= self._level: pass` is a dead branch; it evaluates the likelihood and discards the result. Remove it, together with its comment.

### inference/laplace.py

(none)

### inference/mixture.py

(none)

### inference/estimators.py

(none)

### inference/__init__.py

(none)

Outside this group, noted in passing: `inference/problem.py::ForwardProblem.potential` documents `ValueError` for a set-valued data uncertainty, but the code raises `TypeError` (fork 07's file).

### inference/point.py


### inference/gaussian.py


### inference/normal.py


### inference/preconditioners.py

- `LocalisedPreconditioner._invert` (bug, verified): the class docstring says the blocks "need not cover every index", but an index outside every block gets a zero row and column in the assembled matrix, filled only by the noise diagonal. With no error measure, or a zero noise variance there, `splu` fails. Reproduced with `NormalOperator(A, prior)` (no error) on a 4-dimensional data space and `LocalisedPreconditioner([[0, 1]], rank=2)(N)`: `RuntimeError: Factor is exactly singular`. Fix: put the exact diagonal (or the `NormalDiagonalPreconditioner` probe) on uncovered indices, or refuse a partial cover when the noise diagonal does not cover it, with a message.

### inference/problem.py

- `LinearForwardProblem.joint_measure` / `synthetic_model_and_data` (bug, verified): annotated to take any `ProbabilityMeasure`, but with an error measure the code calls `GaussianMeasure.from_product([measure, error])`, which reads `covariance_factor`. A `GaussianMixture` prior gives `AttributeError: 'GaussianMixture' object has no attribute 'covariance_factor'`. `from_direct_sum` has the same dependence on Gaussian error measures. Fix: use the generic `probability.base.product` (with `push_forward`) when either factor is not Gaussian, or narrow the annotation and raise a `TypeError` naming the requirement.
- `ForwardProblem.chi_squared_from_residual` (cleanup): `centered = residual - expectation` and then `mahalanobis_squared(centered + expectation)` is a round trip back to `residual` (two vector operations for nothing), assuming `mahalanobis_squared` subtracts the expectation itself, as the round trip implies.
- `ForwardProblem.error_measure` (cleanup): the `TypeError` message says "This method needs a Gaussian error measure", but the check is only for a `ProbabilityMeasure`. Say "a measure-valued error".

### inference/tikhonov.py

### symmetric_space/base.py

- `SymmetricSpace.estimate_truncation_degree` and `SymmetricSpace.sufficient_degree` (risk): v1's `estimate_truncation_degree` corresponds to v2's `sufficient_degree`, the walk past the space's truncation. v2 reuses the name `estimate_truncation_degree` for a different question, the degree below which this space's own power lies, which cannot answer past the truncation. Checked against `pygeoinf/symmetric_space/torus.py`: v1's signature `(covariance_function, /, *, rtol=1e-6, min_degree=1, max_degree=None)` is `sufficient_degree`'s. A v1 call passing `rtol=` or `min_degree=` therefore fails loudly with `TypeError`, but a bare `estimate_truncation_degree(f)` still runs, silently answers a different question, and caps the result at the current `lmax`. Suggested fix: rename the v2 `estimate_truncation_degree` (for example `power_degree`), or make it the alias of `sufficient_degree` and record the change for migration.
- `SymmetricSpace.heat_measure` (cleanup): `expectation` is annotated `V | None`; `invariant_measure`, `sobolev_measure` and `power_measure` annotate it `np.ndarray | None`, and a sphere's vectors are not arrays. The annotations should agree (probably `V | None` throughout).
- `SymmetricSpace._path_nodes` (cleanup): reads `self.length_scale`, which `SymmetricSpace` does not declare. Every geometry defines it, so nothing fails, but a new geometry would only find out at the first path operator. Suggested fix: declare an abstract `length_scale` property on the base.

### symmetric_space/box.py

- `Box.support_projection` (bug, verified): on a Sobolev box the code builds `lift_formal_adjoint(base.support_projection(), base, codomain=self)`, which passes the *Lebesgue* space as the domain. The result maps `Lebesgue -> Sobolev`, not `Sobolev -> Sobolev`. Checked on `box.Sobolev((32,), 2.0, 0.1, bounds=((0, 1),))`: `P.domain == X` is False; `P @ X.laplacian` raises `ValueError: Cannot compose ...`; and `P.adjoint(x)` differs from the correct lift `lift_formal_adjoint(lebesgue.support_projection(), X).adjoint(x)` by a relative 630. The existing test (`test_bounded_and_sphere.py`) compares actions only, which agree because the two spaces share vectors. Suggested fix: `lift_formal_adjoint(base.support_projection(), self)`, plus a test that checks the domain and the adjoint (`check_operator`).
- `Box.with_order` (cleanup): the override only calls `self._rebuilt(...)`, exactly as `PeriodicBox.with_order` does, and `_rebuilt` is already overridden, so the override is redundant beyond narrowing the return annotation. Its docstring used to claim the base class would return the enclosing periodic domain, which was not true; the docstring is fixed. Either delete the override or keep it for the annotation.

### symmetric_space/line.py, plane.py, circle.py, torus.py, __init__.py

- No code issues found.

### symmetric_space/sphere.py

- `Sphere.from_components` (cleanup): annotated `-> np.ndarray` but returns a `pyshtools.SHGrid`; likewise `Sphere.evaluate(x: np.ndarray, ...)` takes a field, which is an `SHGrid`. The annotations contradict D-1 and the class docstring. Fix: annotate as `Any` (the class's vector type), as `copy`, `axpy` and `to_components` already do.
- `Sphere.geodesic_quadrature` (cleanup): for coincident endpoints it returns `[np.asarray(start, dtype=float)] * count`, the same array object `count` times (and, when `start` is already a float array, the caller's own array). A caller mutating one node mutates them all. `geodesic_ball_quadrature` does the same for a zero radius. Fix: build `count` separate copies.

### symmetric_space/fourier.py

No issues found.

### plotting/sphere.py

No issues found.

### plotting/fourier.py

- `plot_balls` for a one-dimensional `PeriodicBox` (bug, edge case, verified): a ball wider than the circle (`radius > length / 2`, so `low < 0` and `high > length` together) is wrapped to two thin spans at the ends and leaves the middle unshaded, though the ball covers the whole circle. Reproduced: `plot_balls(circle.Lebesgue(16), [[0.5 * L]], 0.6 * L)` draws spans starting at `0.9 L` and `0`, i.e. `[0.9 L, L]` and `[0, 0.1 L]`. Fix: when `2 * radius >= length`, draw one span over `[0, length]`.

### sem1d/__init__.py

- import guard (risk): the ImportError says "pygeoinf2.sem1d needs planetmodel 1.2 or later" and tests `hasattr(_randomfield, "SpectralBasis")`. But pyproject's extra is `planetmodel>=1.3`, and `sem1d/_transform.py::ShellTransform` needs 1.3's `grid.kind`. Under 1.2 the guard passes and then every `Ball` raises "The grid must be a Gauss-Legendre one, with weights." from `ShellTransform`, which hides the real cause. Fix: check for 1.3 (for instance `planetmodel.__version__` or the `kind` attribute) and say 1.3 in the message. DECISIONS.md D-115 also still says `planetmodel[harmonics]>=1.2`, which is stale against pyproject.

### sem1d/base.py

- `_sampled_with_extension` (cleanup): accepts `"fit"` in its validation and then returns the constant continuation, so a caller that forgot to branch on `"fit"` would silently get the constant fill rather than the fit. Both callers (`AxisSpace.project_function`, `Ball.project_function`) do branch first. Fix: validate `("odd", "constant")` here and let the callers validate `"fit"`, or raise on `"fit"`.

### backends/mfem.py

- `solver_from_bilinear_form` (risk): non-convergence under `strict=True` raises a bare `RuntimeError`, whereas the library's own solvers raise `numerics.solvers.ConvergenceError` (a `RuntimeError` subclass). A caller catching `ConvergenceError`, as the rest of the library's documentation invites, misses MFEM's failures. Fix: raise `ConvergenceError`.
- `operator_from_bilinear_form` (records): DECISIONS.md D-96 says this function "densifies and is not the route for a real problem". The code no longer densifies: `MfemSpace.restrict` keeps CSR and `from_matrix` keeps sparse input sparse. D-96 should be corrected; no code change.

### backends/mfem_hilbert.py

- `_solve_with` (risk): the same bare `RuntimeError` on non-convergence as the coordinate backend, here reaching both the mass solve (every adjoint and `white_noise`) and `solver_from_bilinear_form`. Fix: raise `ConvergenceError`.

### sem1d/axis.py, sem1d/ball.py, sem1d/interval.py, sem1d/radial.py, sem1d/layered.py, sem1d/_transform.py, backends/__init__.py, plotting/sem1d.py

(no code issues found)

### examples (printed text: code, so left alone)

- `examples/01_spaces.py:40` (cleanup): `print("In v1 this last line raises TypeError: unhashable type.")` is a v1 comparison in the program's output. Delete it and the `print()` before it.
- `examples/18_subspaces.py:50-52` (cleanup): the printed text says `A A*` is recognized as semidefinite "by the palindrome rule". That is the library's internal name for its trait propagation and is not defined for a reader. Suggested: "is recognized as semidefinite from its form, so CG is admissible with nothing claimed."
- `examples/26_mixture.py:96` and `:148` (cleanup): the scenarios are numbered both ways in the output. Line 96 prints "(scenario 2)" for the truth, one-based, while the evidence table and the closing paragraph number them 0 and 1 and call the truth "scenario 1". Make line 96 print "(scenario 1)".

### tests

- `tests/test_probability.py:1527, 1557, 1559` (cleanup, coupled to a fix): three `pytest.raises(ValueError, match="D-131")` assert that the density gate's error message cites the decision. When the message in `probability/base.py::_require_density` loses its "(D-131)", as it should under the revised D-10, these matches have to change too, e.g. to `match="discretises a function space"`.
- `tests/test_code_practice.py` (suggestion): D-10 is now a rule ("nothing in the code cites a decision"). The file's own principle is that a rule which is not checked drifts. A test that greps the package, examples and tests for `\bD-\d+` and `DECISIONS.md` would keep the rule. Not added, since this step makes no code changes.
- `tests/test_phase_four.py` (cleanup): the file is named after a development phase. Its contents are hardening, sets and measures tests, and a topical name (or merging it into the files that cover those modules) would say what it holds.
