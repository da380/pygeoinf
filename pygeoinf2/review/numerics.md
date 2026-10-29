# Review of the numerics layer

Scope: everything under `pygeoinf2/numerics/` (solvers, preconditioners,
normal, tikhonov, root_find, randomized, functional_calculus, optimization,
line_search, convex, quadratic_programming, quadratic_forms) and the names
the top level re-exports from it. Every bug claim below was reproduced by a
script run against the working tree on 2026-10-29; the code given is the
minimal form of it. Unless stated, `X = gi.EuclideanSpace(n)` and
`Traits` is `pygeoinf2.traits.Traits`.

Six of the findings share one root cause. A numerical threshold is absolute,
or measured against `max(|x|, 1)`, so the routine's answer depends on the
units the user chose. Those six are: the weighted chi-square, the Lanczos
breakdown test, the Chambolle-Pock stopping rule, the bundle gap test, the
Jacobi and diagonal floors, and the trust-region radii. D-192 removed this
pattern from the checks. Numerics still carries it.

## Must fix before 2.0

### The weighted chi-square depends on the units of its weights, and so does `ambient_ball`
`pygeoinf2/numerics/quadratic_forms.py:267`, `:327`, `:132-136` — The
equal-weights shortcut tests `np.allclose(live, live[0])` with NumPy's
default `atol=1e-8`. When every weight is below about 1e-8, the weights
count as equal, and the answer is an ordinary chi-square scaled by the
*first* weight. Imhof's quadrature also has absolute floors:
`max(value, 1e-6)` in the step and the truncation, the 200 000-point cap,
and `max(abs(integral), 1e-12)`. Those floors break the quadrature by 1e-12
even without the shortcut. `GaussianMeasure.ambient_ball` takes its radius
from `weighted_chi2_quantile`, so the convenience hardening of every set
answer is wrong for a prior whose variances are below about 1e-8 in the
space's units. A covariance in SI units of a gravity or potential field is
in that range. Verified by running.
Reproduction:
```python
w = np.array([5.0, 2.0, 1.0, 0.5, 0.1])
weighted_chi2_quantile(s * w, 0.95) / s
#   23.8467 at s = 1 and 1e-6 (Monte Carlo 23.89); 55.3525 at s = 1e-9 and 1e-12
from pygeoinf2.numerics.quadratic_forms import _imhof
_imhof(s * w, s * 23.8896, tolerance=1e-10)
#   0.950259 at s = 1; 0.951415 at 1e-9; 0.594856 at 1e-12
mu = gi.GaussianMeasure.from_standard_deviations(gi.EuclideanSpace(5), np.sqrt(s * w))
ball = mu.ambient_ball(level=0.95)
#   fraction of 20 000 draws inside: 0.9527 at s = 1, 0.9984 at s = 1e-9
```
Proposed fix: at the entry of both public functions, divide the weights
and the threshold by the largest weight, and multiply the quantile back.
The distribution scales exactly, so every floor below then acts on
normalized numbers. Test the equal-weights case with a relative tolerance
only.

### The Lanczos calculus allocates a `dim x dim` array by default
`pygeoinf2/numerics/functional_calculus.py:234` (with `:383`, `:487`) — On a
coordinate space, `_iter_lanczos_on_components` preallocates
`np.empty((space.dim, max_iterations))`. The default `max_iterations` of
`apply_operator_function`, `operator_quadratic_form` and `OperatorFunction`
is `space.dim`, so each Lanczos run reserves a dense square, however few
steps it takes. This affects `operator_sqrt`, `operator_inverse_sqrt`,
`operator_exp`, `operator_log` and `operator_power`. It also affects
`GaussianMeasure.kl_divergence(method="stochastic")`, which calls
`operator_function` without a cap (`probability/gaussian.py:1399`, read
only), and is the route meant for spaces above 4000. On Linux the
reservation is lazy until it exceeds RAM plus swap. Past that, the calculus
fails outright at exactly the sizes it exists for. It breaks "matrix-free by
default". Verified by running.
Reproduction:
```python
n = 120_000; X = gi.EuclideanSpace(n); d = 1.0 + np.arange(n) % 5   # five eigenvalues
A = gi.LinearOperator.from_callables(X, X, lambda x: d * x, adjoint=lambda x: d * x,
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
apply_operator_function(A, np.sqrt, np.ones(n))
# MemoryError: Unable to allocate 107. GiB for an array with shape (120000, 120000)
```
Proposed fix: grow the column buffer geometrically, or keep a list of
columns and stack per step. The basis held at any time is then the number
of steps actually taken.

### Lanczos stops after one step on an operator whose norm is below about 1e-12
`pygeoinf2/numerics/functional_calculus.py:201`, `:270` — The breakdown test
is `beta <= breakdown_tol * max(abs(alpha), 1.0)`. This is the floor at one
that D-192 took out of the checks. For an operator of norm about 1e-13, the
off-diagonal is always below `1e-12 * 1`. The Krylov space is then
declared closed after one step, and `f(A) x` is computed from one Ritz
value, with no warning. The result depends on the units of the covariance
or operator. Verified by running.
Reproduction: a 50-dimensional `M = Q diag(linspace(1, 3)) Q^T`, wrapped as
`from_callables` with `SELF_ADJOINT | POSITIVE_DEFINITE` and scaled by `s`.
`apply_operator_function(s*M, np.sqrt, x)` has a relative error of 1.1e-11
at `s = 1`, `1e-6` and `1e-11`, and 1.38e-01 at `s = 1e-13` and `1e-15`.
Proposed fix: measure `beta` against the running scale of the tridiagonal
matrix, the largest `|alpha|` and `beta` seen so far, with no absolute term.

### GMRES and MINRES report convergence when a singular operator closes its Krylov space exactly
`pygeoinf2/numerics/solvers.py:1627`, `:1745`, `:1764` — GMRES treats
`subdiagonal == 0.0` as a happy breakdown. It then solves the small
triangular system and reports `|rhs[k+1]|` as the residual, without checking
that the triangular factor is nonsingular. MINRES returns
`converged=True` on `gamma == 0.0` and on `beta_next == 0.0` in the same way.
On a singular operator whose Krylov space closes in exact arithmetic, with
an inconsistent right-hand side, both report a residual near 1e-16, but the
true residual is worse than the least-squares minimum. Exact closure happens
on a 0/1 mask, a coordinate projector, or a diagonal operator with an exact
zero. A random rank-deficient matrix does not close exactly, and both
solvers then raise correctly. A wrong answer marked converged gets past
`strict=True`. Verified by running.
Reproduction:
```python
mask = gi.DiagonalLinearOperator(gi.EuclideanSpace(6), np.array([1., 1, 1, 0, 0, 0])).with_traits(Traits.SELF_ADJOINT)
r = GMRESSolver(rtol=1e-10)(mask).solve(np.array([1., 2, 3, 1, 1, 1]))
# converged True, residual_norm 1.8e-16; true residual 5.74 (least-squares minimum 1.73)
A = gi.LinearOperator.from_matrix(gi.EuclideanSpace(3), gi.EuclideanSpace(3), np.diag([1., -2, 0]),
        form="components", traits=Traits.SELF_ADJOINT)
r = MinResSolver(rtol=1e-10)(A).solve(np.ones(3))
# converged True, residual_norm 7.2e-15; true residual 9.0
```
Proposed fix: at a breakdown, test the last diagonal of the triangular
factor (GMRES's `R_kk`, MINRES's `gamma`) against a threshold relative to
the factor's norm. If it is singular, form the true residual and report
non-convergence. Alternatively, solve the small problem by least squares
and report its own residual.

### Chambolle-Pock stops on an absolute feasibility residual and declares convergence on feasibility alone
`pygeoinf2/numerics/convex.py:1538`, `:1668` — `ChambollePockSolver` stops
when `||G m + v - d|| < tolerance`, with `tolerance=1e-6` in the data's own
units. It then reports `converged=True`. It never asks whether the iterate
is optimal. When the data are small in their units, one iteration meets the
test, and the reported support is not attained by any feasible model: it
exceeds the true maximum. This is not CURRENT_STATE §7's "inner value
attained only to its tolerance". That finding is about an error of the
tolerance's size. This one is a tolerance in the wrong units, and the error
it gives is of order one. Verified by running.
Reproduction: `n, m = 8, 5`, a random `G` with `from_matrix(..., form="components")`,
`Ball(X, radius=s)`, `Ball(D, radius=0.05*s)`, data `s * d0`, a fixed `c`.
`ChambollePockSolver(prior, noise, A, s*d0, max_iterations=20000).solve(c)`:
at `s = 1` the support is 0.604047 after 86 iterations, converged (SLSQP
gives 0.604047). At `s = 1e-7` the support divided by `s` is 3.267090 after
1 iteration, converged, with residual 4.2e-7 against data of size 1e-7.
Downstream, in `FeasiblePropertySet` on the weighted test setting scaled by
1e-5 (`via="splitting"`), the outer value is 4.8499 and the inner 4.8778,
with the inner above the outer. The primal and the dual both give 2.9972.
Proposed fix: make the tolerance relative to `||d||` (or to the noise
radius), and stop on the standard primal and dual residuals of the
iteration, so that `converged` means a saddle point, not only a feasible
point.

### The proximal bundle method stops at its starting point on a scaled objective
`pygeoinf2/numerics/convex.py:1075`, `:1191`, `:1213` — The gap test is
`gap <= tolerance * max(abs(best), 1.0)`, a floor at one. The proximal
weight starts at an absolute 1.0, and its clamps `1e-8` and `1e12` are
absolute. If the objective's values and subgradients are small in their
units, the first model gap, about `||g||^2 / 2`, is already below `1e-8`.
The method then returns the starting point marked converged. The
geometry layer's `Dual` route works around the weight by hand
(`geometry/feasible.py:2394`, `_scaled_start`). It cannot work around the
gap floor: on the scaled feasible setting below, `via="dual"` returns
`inner = -inf` with `converged=True`. Verified by running.
Reproduction: on `make_dense_metric_space(6)`, with `f(x) = s (||x - c|| + 0.3 ||x||)`
built by `SupportFunction.of_oracle` with its maximizer, run
`ProximalBundleMethod(max_iterations=500).minimize(f, X.zero())`. At `s = 1`
it converges in 4 iterations, with excess `(f - f*)/f* = 4.6e-9`. At
`s = 1e-4` and `1e-8` it is "converged" after 1 iteration at `x0 = 0`, with
excess 2.33 and distance to the minimizer 100 %. On the feasible setting
scaled by 1e-5, the dual's outer value is 2.997160 and its inner `-inf`,
with `converged=True`.
Proposed fix: make the gap test relative to `|best|` with no floor, or
relative to the first gap. Set the initial weight from the first
subgradient, as `_scaled_start` does, inside the method. `Dual` can then
drop its copy.

## Should fix

### The Woodbury preconditioner's default inner solve breaks the default outer solver
`pygeoinf2/numerics/preconditioners.py:1120`, `:1267` — The default inner
solver is CG at `rtol=1e-3`, and the result still claims `SELF_ADJOINT`.
That claim fails `check_traits` by 27 %. `CGSolver` needs a symmetric
preconditioner, so the combination that `from_normal`'s docstring calls
"the ordinary way to build the preconditioner",
`CGSolver(preconditioner=WoodburyPreconditioner.from_normal(N))`, fails on a
4-dimensional data space that CG alone solves in 4 iterations. D-47 makes
an inexact inner solve "the caller's to fix with FlexibleCGSolver". I think
that is wrong as the *default*: the library's own defaults should not
compose into a failure, and its trait claim should hold. Verified by
running.
Reproduction: the model is `make_dense_metric_space(12)`, the data
`make_weighted_space()` (dim 4), `A` random, the prior
`from_covariance_matrix(0.1 * spd, form="galerkin")`, the error standard
deviations `[0.1, 0.2, 0.3, 0.4]`, and
`N = NormalOperator(A, prior, error=error)`.
`CGSolver(rtol=1e-10)` takes 4 iterations. With
`preconditioner=WoodburyPreconditioner.from_normal(N, solver=CholeskySolver())`
it takes 1. With the default inner solver it raises
`ConvergenceError: did not converge in 20 iterations` (`FlexibleCGSolver`:
11). `check_traits` on the default-inner preconditioner fails with
`0.0725 != 0.0570`.
Proposed fix: either default the inner solve tight enough that the
asymmetry is below an outer `rtol` of 1e-8, or make the inexact form claim
no trait and have `CGSolver` refuse, by name, a preconditioner that does
not claim `SELF_ADJOINT`, naming `FlexibleCGSolver`. The second also needs
`IdentityPreconditioner` and the components-form sparse preconditioners to
state their claims.

### A damping floor above one makes the misfit search raise
`pygeoinf2/numerics/tikhonov.py:567`, `pygeoinf2/numerics/root_find.py:206` —
`misfit_search` hands `minimum_damping` to `monotone_root` as `minimum`,
with `initial` left at 1.0. `monotone_root` refuses `minimum > initial`.
A floor of 10, legitimate on a problem whose root is at 230, therefore
raises a `ValueError` about an `initial` the caller never set.
`MinimumNorm(minimum_damping=...)` goes through the same path. A damping
has units, so a floor above one is ordinary. `geometry/feasible.py:1281`
already works around this by passing `initial=max(1.0, 10.0 * minimum)`.
Verified by running.
Reproduction: `A` is 10 x 20 with entries of size 100, the error has
standard deviation 1, and `family = TikhonovFamily(A, error=error)`.
`misfit_search(family, rhs, misfit, 10.0)` finds damping 230.5.
`misfit_search(..., minimum_damping=10.0)` and
`gi.MinimumNorm(gi.LinearForwardProblem(A, error=error), minimum_damping=10.0)(d)`
raise `ValueError: The start must lie between the floor and the ceiling:
got minimum=10.0, initial=1.0, maximum=inf.`
Proposed fix: start the search at `max(1.0, 10.0 * minimum_damping)` inside
`misfit_search`, as the geometry layer does, or expose `initial=`.

### Preconditioned GMRES stops on the preconditioned residual
`pygeoinf2/numerics/solvers.py:1553-1557`, `:1641` — GMRES preconditions
from the left. Its stopping test and `residual_norm` are `||P r||` against
`rtol ||P b||`. `IterativeSolver`'s `rtol` documents "the residual norm
... `rtol * ||b||` measured in the space's own norm". That holds for CG,
flexible CG and BiCGStab (right-preconditioned), but not for GMRES, and
GMRES's docstring does not say so. With a badly scaled preconditioner,
`converged=True` says nothing about `A x - b`. The reproduction's
preconditioner is contrived, with a spread of 1e8, but the semantics are
what they are. Verified by running.
Reproduction: `M` random plus `8 I` (n = 50) and
`P = diag([1e-4]*25 + [1e4]*25) @ inv(M)`.
`GMRESSolver(rtol=1e-8, preconditioner=P)(A).solve(b)` gives
`converged=True` after 1 iteration, reported residual 1.5e-4, and true
relative residual 1.88.
Proposed fix: precondition from the right (`x = P u`), so that the residual
GMRES minimizes and reports is the true one, as for the other Krylov
solvers.

### Absolute floors make the diagonal preconditioners fail on small operators
`pygeoinf2/numerics/preconditioners.py:97`, `:161`, `:1575`, `:1646`,
`:638`, `:719` — `JacobiPreconditioner` and `NormalDiagonalPreconditioner`
leave uninverted, and set to one, every diagonal entry whose magnitude is
at most `floor=1e-14`. The column thresholds (`_column_rows`) and
`sparse_approximation` use the same absolute 1e-14. On an operator in
small units, every entry falls under the floor, and the preconditioner is
silently the identity times the wrong scale. Verified by running.
Reproduction: a 200-dimensional SPD `M0` with diagonal `logspace(0, 6)`,
scaled by `s`. `CGSolver(rtol=1e-8, preconditioner=JacobiPreconditioner(), strict=False)`
takes 5 iterations at `s = 1` and `1e-12`, and 400 iterations, unconverged,
at `s = 1e-16`.
Proposed fix: take each floor relative to the largest entry, for example
`floor * max|d|`, and document it as relative.

### `InvariantDistancePreconditioner` fails with `AttributeError` on Tikhonov and damped members
`pygeoinf2/numerics/preconditioners.py:2070`, `:2092` — The class accepts
any `FactoredNormalOperator` through `_require_normal`, then reads
`normal.prior`, a measure. Only `NormalOperator` has that attribute. The
members of every family the library sweeps, `TikhonovNormalOperator` and
`DampedNormalOperator`, fail with an `AttributeError` instead of a refusal
by name. Verified by running.
Reproduction: on `Sobolev(16, 2.0, 0.1)` with 30 random points,
`InvariantDistancePreconditioner(space, points, 0.3)` applied to
`TikhonovNormalOperator(forward, 0.1, error=error)` raises
`AttributeError: 'TikhonovNormalOperator' object has no attribute 'prior'`.
`DampedNormalOperator(forward, prior.covariance, error.covariance, 0.5)`
raises the same error.
Proposed fix: refuse non-`NormalOperator` inputs by name, or compute the
covariance function from `prior_covariance`, since `covariance_function`
only needs the operator's spectrum.

### The family sweeps keep a dense matrix or a factorization for every multiplier they visit
`pygeoinf2/numerics/root_find.py:456`, `:507`;
`pygeoinf2/numerics/preconditioners.py:1325`, `:1344`, `:1352` — With a
direct solver on a family whose member cannot read its own matrix,
`DampedSolves._sum_member` builds `base + t shift` as a dense matrix, and
`operator()` caches it in `_cache` for every multiplier. `_WoodburyFamily`
keeps the data form, with its inner factorization, in `_forms` for every
multiplier. Memory grows with the number of probes, at `dim^2` per probe:
128 MB per probe at dim 4000, over a search of up to 60. Only the last
member, or one about to be revisited, is ever reused. Verified by counting;
the per-entry memory is computed, not measured.
Reproduction: a `DampedSolves(A A*, I, CholeskySolver(n_jobs=4))` at
n = 300, solved at 12 multipliers, holds 12 cached members, each with a
dense Galerkin matrix (8.6 MB in all). A `misfit_search` through
`TikhonovFamily(A, error=error, solver=CGSolver(preconditioner=WoodburyPreconditioner(..., solver=CholeskySolver())))`
with 8 probes leaves 8 forms in `_bound_preconditioner._forms`.
Proposed fix: keep only the most recent member (or a cache of two), and
let the root find's own memo cover revisits.

### Several probe loops ignore `n_jobs`
`pygeoinf2/numerics/root_find.py:507-508`; `pygeoinf2/numerics/preconditioners.py:307`,
`:1616`, `:1631`, `:1673`, `:1879`, `:2053` — These loops run serially:
- `DampedSolves._matrices` extracts the base and shift matrices, which is
  `2 dim` applications, possibly of `A C A*`, even when the direct solver
  was built with `n_jobs`.
- `NormalDiagonalPreconditioner._base_diagonal` at scale zero probes the
  whole normal operator's diagonal, `dim` applications.
- `SpectralPreconditioner.for_family` applies the shift to each resolved
  mode, although `self._n_jobs` is set.
- The `error_covariance.diagonals(...)` and `shift.diagonals(...)` calls do
  not pass it.
- `InvariantDistancePreconditioner` takes no `n_jobs` at all.

D-52 calls a probe loop without `n_jobs=` a regression. Read only.
Proposed fix: forward the solver's or preconditioner's `n_jobs` at each of
these sites, using `apply_block` for the mode loop.

### The stochastic diagonal estimates return bare arrays, without their error
`pygeoinf2/numerics/randomized.py:1009`, `:907`, `:1112` — D-51's rule is
that stochastic estimates return an `Estimate` with a standard error, and
`random_trace` and `log_determinant` follow it. `random_diagonal` and
`deflated_diagonal` return a plain `ndarray`. The adaptive route computes
the per-entry standard error to decide when to stop, then discards it. A
caller cannot tell an estimated diagonal from an exact one, or how good it
is. Changing the return type is a breaking change, so it has to happen
before 2.0 if at all. Read only.
Proposed fix: return an array-valued estimate (`value`, `standard_error`,
`samples`), either an `Estimate` generalized to arrays or a sibling type,
or record the exception in D-51.

### `log_determinant`'s `rtol` does not mean what the documentation says
`pygeoinf2/numerics/functional_calculus.py:704-705` — CURRENT_STATE §4 and
D-51 say `random_trace`, `random_diagonal` and `log_determinant` take an
`rtol` "to sample to". In `log_determinant`, `rtol` (default 1e-3) is the
inner Lanczos tolerance, and the sampling tolerance is `sample_rtol`. The
same keyword means different things in two sibling functions, and the
documentation describes the other one. A caller who passes `rtol=1e-2`,
expecting the trace to 1 %, loosens every `log(A) z` instead. Read only.
Proposed fix: rename so that `rtol` is the sampling tolerance everywhere
(`rtol` and `lanczos_rtol`), carrying the change through
`GaussianMeasure.kl_divergence`. Otherwise correct CURRENT_STATE §4 and D-51.

### `damping` means `d^2` in LSQR and `t` in the Tikhonov family
`pygeoinf2/numerics/solvers.py:1877`, `:1892` against
`pygeoinf2/numerics/tikhonov.py:9` — `LSQRSolver(damping=d)` minimizes
`||Ax - b||^2 + d^2 ||x||^2`. `TikhonovNormalOperator(A, d)`, its family and
`minimum_damping` mean `A*A + d I`. The same public word gives two
different problems for the same number. Read only (the docstrings say so).
Proposed fix: make LSQR's damping the Tikhonov parameter and take its root
inside, or rename it (`damping_root`).

### Tolerance and budget keywords are named five ways
`pygeoinf2/numerics/convex.py:1076`, `:1538`, `:637`, `:756-759`;
`pygeoinf2/numerics/optimization.py:1281`;
`pygeoinf2/numerics/quadratic_forms.py:230`, `:294`;
`pygeoinf2/numerics/quadratic_programming.py:167`, `:253-254`, `:356-357`;
`pygeoinf2/numerics/line_search.py:102` — The library's convention is
`rtol`, `atol` and `max_iterations`. Alongside it:
- `tolerance=` means four different things: a relative gap with a floor in
  `ProximalBundleMethod`, an absolute residual in `ChambollePockSolver`, an
  absolute violation in `AugmentedLagrangian`, and Imhof's quadrature in
  `weighted_chi2_*`.
- `SciPyQPSolver(tolerance=)` is SLSQP's `ftol`.
- The OSQP and Clarabel wrappers use `absolute_tolerance=` and
  `relative_tolerance=`.
- `ArmijoLineSearch` and `ProximalGradient` use `max_backtracks=` for a
  budget.
- `SubgradientDescent` uses `step_size=` where `ProximalGradient` and
  `ProximalPoint` use `step=`.

These are all public names that become committed API at 2.0. Read only.
Proposed fix: use `rtol` or `atol` according to what each one is, rename
the QP wrappers' arguments to `atol`/`rtol`, use `max_iterations` for the
backtracking budgets, and use `step` throughout.

### The QP module documents a level bundle method that does not exist
`pygeoinf2/numerics/quadratic_programming.py:11-15`, `:473` — The module
docstring says "The level bundle method's subproblem has a general linear
constraint ... it needs a QP solver proper". `best_available_qp_solver`
argues its ordering from "the level method's master problem". There is no
level bundle method in the package. Only the proximal one uses these
backends. Read only (`grep -rn LevelBundle` finds nothing).
Proposed fix: rewrite both passages for the proximal bundle method's
simplex subproblem, the backends' only user.

## Could fix

### An unsolved QP is reported as solved
`pygeoinf2/numerics/quadratic_programming.py:334`, `:461` — The status maps
with `"solved" in reported.lower()`. OSQP's `unsolved` and Clarabel's
`Unsolved` both contain "solved", and so do `AlmostSolved` and
`solved inaccurate`. `ProximalBundleMethod` falls back to its own solver
only when `result.solved` is false. Verified (`'solved' in 'unsolved'` is
`True`; both status names exist in the installed osqp 1.1.1 and clarabel).
Proposed fix: compare against the backend's exact success status, and
decide explicitly whether the "almost" and "inaccurate" statuses count.

### `TrustRegionNewton`'s radii are absolute, and it ignores `line_search=`
`pygeoinf2/numerics/optimization.py:851-852` — `initial_radius=1.0`,
`max_radius=1e6` and the collapse test `radius < 1e-14` are in the space's
units. A quadratic whose minimizer is 2e8 from the start takes 243
iterations, where `NewtonCG` takes 1. The class also accepts the base
class's `line_search=` and never uses it, and it has no
`max_cg_iterations` where `NewtonCG` does. Verified by running
(`SquaredDistance(X, center=1e8*ones(5))`: 2, 15 and 243 iterations at
distances of about 2, 2e4 and 2e8).
Proposed fix: set the initial radius from the first Cauchy step, make the
maximum relative or unbounded, refuse `line_search=`, and add
`max_cg_iterations=`.

### `operator_quadratic_form` claims a role it does not play
`pygeoinf2/numerics/functional_calculus.py:450` — The docstring says it is
"the kernel of stochastic Lanczos quadrature, which is how a
log-determinant gets estimated". `log_determinant` goes through
`random_trace(OperatorFunction)`, which recombines the Krylov basis for
every probe. Nothing in the library calls `operator_quadratic_form`. Read
only.
Proposed fix: either route the stochastic `log_determinant` through it
(Gauss quadrature without forming `log(A) z`), or correct the docstring.

### `JacobiPreconditioner` refuses an operator that `BandedPreconditioner(0)` accepts
`pygeoinf2/numerics/preconditioners.py:90` — Jacobi requires
`SELF_ADJOINT`, so a nonsymmetric operator under GMRES is refused, while
`BandedPreconditioner(0)`, the same diagonal by its `"auto"` form, accepts
it. Verified by running.
Proposed fix: let Jacobi take the components diagonal of an operator that
does not claim self-adjointness, as the banded form does.

### `DampedSolves` takes optional and private arguments positionally
`pygeoinf2/numerics/root_find.py:375-436` — As a plain `@dataclass`, its
signature is `(base, shift, solver, traits=None, assemble=None,
refresh=10.0, _cache=..., _prepared=..., ..., _bound_preconditioner=None)`.
The optionals are positional, against D-104, and the six private state
fields are constructor arguments. Verified with `inspect.signature`.
`DampedSolves(base, shift, solver, None, None, 5.0)` sets `refresh`.
Proposed fix: `@dataclass(kw_only=True)` for the options, and
`field(init=False)` for the state.

### The range finder's measure probes include the measure's expectation
`pygeoinf2/numerics/randomized.py:104` — Probes are `measure.sample()`, so
a prior with a nonzero mean spends one direction on `A m`. The docstring
describes the range of the operator composed with the covariance factor.
The effect is small. Verified: the median unresolved fraction at rank 10
went from 0.066 to 0.070 with a mean of norm about 1e4.
Proposed fix: subtract the expectation from each draw, or say in the
docstring that it is included.

### The proximal bundle method keeps per-run state on the instance
`pygeoinf2/numerics/convex.py:1260`, `:1341` — `_warm`, `_dropped` and
the Gram cache keyed by `id()` live on `self` during `minimize`. A shared
instance, such as a user-supplied `method=` that `Dual` reuses across
directions under a threading backend (D-52 respects one), is not
reentrant. `IterativeSolver._lifting` copies the solver precisely to avoid
this. Separately, `with_tolerance` is inherited by `ProximalBundleMethod`,
`ProximalPoint` and `SubgradientDescent`, and silently changes nothing they
read. An `AugmentedLagrangian` over one of them never tightens its inner
solves. Read only.
Proposed fix: keep the per-run state in locals passed through `_minimize`,
and have `with_tolerance` refuse on methods it cannot steer.

### Preconditioners are not checked against the operator's spaces
`pygeoinf2/numerics/solvers.py:1245-1246` — `IterativeSolver._invert`
validates the operator but not a ready-made preconditioner. One on another
space of the same dimension is accepted silently. That is harmless to the
answer, which was verified by running: the residual is still 4e-16. A
preconditioner of another dimension fails deep in the arithmetic. Read
only for the second case.
Proposed fix: check `preconditioner.domain == operator.codomain` and
`preconditioner.codomain == operator.domain` in `_invert`.

### A dangling history comment ends `preconditioners.py`
`pygeoinf2/numerics/preconditioners.py:2116-2117` — The file ends with "The
former spelling, kept only until `inference/__init__.py` imports the new
one; not exported." No code follows it. This is history talk, against the
docstring rule.
Proposed fix: delete it.

### `SolverLike` is exported as a string
`pygeoinf2/numerics/solvers.py:51`, `:193` — `SolverLike` is in the
module's `__all__`, but it is a string constant, not a type, and
`numerics` does not re-export it.
Proposed fix: drop it from `__all__`, or make it a `TypeAlias`.

### Tests reach into numerics' private state
`tests/test_convex.py:576`, `:583` (`_dropped`, `_warm`);
`tests/test_point.py:1190`, `:1203`, `:1235`, `:1277` (`_base_diagonal`,
`_solves._bound_preconditioner`, `bound._factor`, `_prepared`);
`tests/test_solvers.py:1192` (`inner._rtol`);
`tests/test_known_structure.py:526` (`_use_component_view`); and the
private functions `_minimize_on_simplex`, `_eigh_tridiagonal` and
`_saddlepoint`. These are white-box tests, and most are reasonable as such.
They are listed because the brief asks for them. The family tests would
read better through a public `bound_preconditioner` or a solve count.
Proposed fix: none required. Optionally, add read-only properties where
a test needs to see a family's state.

### The waveform app builds a prior-whitened deflation preconditioner by hand
`pygeoinf2/miniapps/waveform/IMPLEMENTATION.md` §4 — The app builds
`factor @ SpectralPreconditioner(damping=1)(L* N L) @ factor.adjoint` by
hand, because deflating the normal operator itself finds the prior
precision's eigenvalues. D-198's low-rank posterior uses the same
construction. Read only.
Proposed fix: consider a `factor=` argument on `SpectralPreconditioner`,
which deflates `L* N L` and wraps the result, so the pattern has one name.

## What is sound

These parts were exercised and found right:
- **Krylov and direct solvers.** CG, flexible CG, MINRES, GMRES, BiCGStab,
  Cholesky, LU and Eigen, on `make_dense_metric_space(12)` and on
  `make_weighted_space()`, solve to 1e-13 or better. `check_operator`
  passes on every inverse and its adjoint, including the transposed solves
  that the direct solvers share with the adjoint.
- **Preconditioned solves.** Every iterative solver converges to rounding
  under the Jacobi, banded, block, column-thresholded, spectral and
  identity preconditioners on both metrics. Aliasing through the identity
  preconditioner is harmless.
- **Preconditioner traits.** `check_traits` and `check_operator` pass on
  the Jacobi, banded, block, column-thresholded, spectral,
  normal-diagonal (blocked and not), localized, and exact-inner Woodbury
  preconditioners on a dense model metric with a weighted data metric.
- **Woodbury exactness.** With exact inner solves, both Woodbury forms
  invert their normal operator to 6e-13 and 2e-13.
- **LSQR.** Damped and undamped, it matches the metric normal equations on
  a dense Gram.
- **Randomized methods.** `random_eig`, `random_svd` and `random_cholesky`
  recover a G-orthonormal spectrum exactly. `random_trace` lands within its
  standard error. `random_diagonal` and `deflated_diagonal` (Galerkin and
  components forms) agree with the dense diagonals, and the deflation cuts
  the error twentyfold.
- **Log-determinant.** The dense route is exact on a dense metric, and the
  stochastic route lands within 1.5 standard errors.
- **Functional calculus.** `operator_sqrt` squares back to 2e-9.
- **Smooth optimizers.** `SteepestDescent`, `NonlinearCG`, `LBFGS`,
  `NewtonCG` (plain and preconditioned) and `TrustRegionNewton` all
  minimize a dense-metric quadratic. `AugmentedLagrangian` reproduces the
  equality multiplier to 1e-11.
- **Convex methods.** `ProximalGradient` with `BallIndicator` and
  `NormFunctional`, `ProximalPoint`, and Polyak `SubgradientDescent` give
  the closed forms on a dense metric. Chambolle-Pock and the bundle method
  are correct at unit scale.
- **Misfit search.** `misfit_search` brackets and converges.
- **Test suites.** `test_solvers`, `test_preconditioners`,
  `test_randomized`, `test_functional_calculus`, `test_convex`,
  `test_optimization`, `test_normal`, `test_quadratic_programming` and
  `test_known_structure` pass (708 tests).
- **Hygiene.** There are no decision numbers or v1 talk in the numerics
  sources, apart from the one comment noted above.

## Fixed

Every fix below has a test that fails on the code at `4b27d35` and passes
now; the tests were run against an export of that commit to check the first
half. Tolerances in the tests are relative to a named scale.

### Must fix

1. **The weighted chi-square and `ambient_ball` depend on the units of the
   weights.** `numerics/quadratic_forms.py`: `weighted_chi2_cdf` and
   `weighted_chi2_quantile` divide the weights and the threshold by the
   largest weight on entry (`_normalized`) and multiply the quantile back
   (`_normalized_quantile`); `_imhof` normalizes on its own as well, so its
   step and truncation floors, grid cap and convergence floor all act on
   weights whose largest is one. The equal-weights shortcut is relative
   (`_common_weight`): a spread of at most `sqrt(tolerance)` of the largest
   weight is an ordinary chi-square at the mean, exact to second order in
   the spread. Tests: `tests/test_functional_calculus.py::TestWeightedChi2DoesNotDependOnUnits`
   (quantile at scales 1, 1e-9, 1e9 for Imhof, saddlepoint and matched;
   the CDF and `_imhof` at 1e-12, 1e-9, 1e9; nearly equal weights; and
   `GaussianMeasure.ambient_ball(level=0.95)` on deviations of `3e-5` on a
   dense metric covering 95% of 4 000 draws to four binomial standard
   errors, where it covered 99.8%).
2. **The Lanczos calculus preallocated `dim x max_iterations`.**
   `numerics/functional_calculus.py::_iter_lanczos_on_components`: the
   column buffer starts at eight columns (`_INITIAL_COLUMNS`) and doubles as
   steps are taken. Test: `TestLanczosHoldsOnlyTheStepsItTakes`, a
   120 000-dimensional weighted space with a matrix-free operator of five
   eigenvalues: at most six applications and a `tracemalloc` peak under 32
   vectors (measured 16 MB, 17 vectors), where it asked for 107 GiB.
3. **Lanczos's breakdown test was floored at one.** Both
   `iter_lanczos_tridiagonalize` and `_iter_lanczos_on_components` judge
   `beta` against the running scale of the tridiagonal matrix, the largest
   `|alpha|` and `beta` met, with no absolute term; `breakdown_tol`'s
   docstring says so. Test: `TestLanczosBreakdownIsRelative`, the square
   root of a 50-dimensional operator self-adjoint on a dense metric at
   scales 1, 1e-13, 1e-15 and 1e13, on the component and the coordinate-free
   iterations, to 1e-9 relative (it was 14% wrong at 1e-13).
4. **GMRES and MINRES reported convergence on a closed Krylov space.**
   `numerics/solvers.py`. Convergence is declared on the true residual in
   both. GMRES (`GMRESSolver._solve`) ends every cycle by forming
   `b - A x` and testing that; it detects closure relative to the image it
   orthogonalized (`_CLOSURE`), solves the small triangular system by least
   squares when it is singular (`_triangular_weights`), and stops when a
   cycle whose recurrence stopped (on its tolerance or on closure) gained
   less than half. MINRES (`MinResSolver._solve` and `_cycle`) runs its
   recurrences in cycles: closure is judged against the size of the new
   column of the tridiagonal matrix, a singular projected system ends the
   cycle without dividing by rounding error, the true `M`-norm residual
   decides, and a cycle that gained less than half ends the solve. Tests:
   `tests/test_solvers.py::TestASingularOperatorWhoseKrylovSpaceClosesIsNotConverged`
   (a 0/1 mask and a diagonal with a zero and a negative entry on a weighted
   space, both solvers: the inconsistent system raises under `strict`,
   reports its true residual, at least the least-squares minimum, under
   `strict=False`, and the consistent one still converges). The white-box
   `TestSolutionTracking::test_gmres_does_not_form_the_iterate_unless_asked`
   now counts `_triangular_weights` instead of `np.linalg.solve`.
5. **Chambolle-Pock stopped on an absolute feasibility residual.**
   `numerics/convex.py::ChambollePockSolver`. Unless both steps are given,
   they are chosen per solve from `||G||`, `||d||` and `||c||`, separately
   for the model, the data error and the dual
   (`theta rho / ||G||`, `theta rho ||G||`, `theta / (rho ||G||)` with
   `rho == ||d|| / ||c||`, `theta**2 == 0.495`), so rescaling the model, the
   data or the objective rescales the iterates and nothing else. The stop is
   relative: feasibility against the largest of `||G m||`, `||v||`, `||d||`,
   and both stationarity residuals (`||m_prev - m|| / tau` against
   `max(||c||, ||G* y||)`, `||v_prev - v|| / tau` against
   `max(||y||, ||c|| / ||G||)`) against theirs. `SaddlePointResult` gains a
   `stationarity` field; one given step still fixes the other as before.
   Tests: `tests/test_quadratic_programming.py::TestChambollePockDoesNotDependOnUnits`
   (support at scales 1, 1e-7 and 1e5 against SLSQP with both metrics
   explicit, a dense model metric and a weighted data metric; one iteration
   is feasible but not converged; through `FeasiblePropertySet` with
   `via="splitting"` at scales 1 and 1e-5, the inner end at or below the
   outer, the values equal across scales and to the primal's). At unit
   scale the stricter stop costs iterations: 315 against 87 on the
   reviewer's 8 x 5 problem, with the support right to 1e-9 against 2e-8.
   `tests/test_feasible.py::TestTheSplitting::test_it_agrees_with_the_primal_on_balls`
   checked the extremal model's misfit against the noise radius at
   `rtol=1e-6`, an absolute `1e-7`; it now allows ten times the tolerance
   times the data's size, which is what a relative feasibility promises.
6. **The proximal bundle method returned its start on a scaled objective.**
   `numerics/convex.py::ProximalBundleMethod._minimize`. The gap is judged
   against the larger of `|f|` at the best point and half `|f|` at the
   start, the first model's gap standing in where the start's value is
   zero (the projection dual of a point inside a set); no floor at one. A
   first model made of a single linearization is never taken as
   convergence: a gap below the tolerance from it means the weight is too
   heavy for the units, and the weight is reset to `||g||^2 / |f|`. The
   weight's clamps are relative to its starting value. The built-in simplex
   solver (`_minimize_on_simplex`) takes its stopping test, step and
   warning relative to the program's own scale. Tests:
   `tests/test_convex.py::TestTheBundleMethodDoesNotDependOnUnits` (a
   dense-metric objective at scales 1, 1e-4, 1e-8 and 1e4, Clarabel and the
   built-in solver: converged after more than one iteration, value to 1e-6
   and minimizer to 1e-6 relative; at 1e-4 and 1e-8 it stopped at the start
   with a value 2.3 times the minimum).

### Should fix

7. **The Woodbury preconditioner's default inner solve.** Tightened, not
   made relative to the outer tolerance (the preconditioner is built
   before, and independently of, the solver that will use it, so a
   relative inner tolerance would need the outer `rtol` plumbed into
   `solver(A)`): `WoodburyPreconditioner._inner_solver` is CG at
   `rtol=1e-10`, `strict=False`, the default cap; the class note and the
   `solver` argument say why. This revises the default D-47 describes; the
   caller's own loose inner solver still wants `FlexibleCGSolver`.
   Tests: `tests/test_solvers.py::TestWoodburyInnerDefault` (the default is
   tight and forgiving; on the reviewer's dense model metric and weighted
   four-dimensional data space the default preconditioner passes
   `check_traits` at `1e-6`, where it failed by 0.17% at `1e-3`, and CG with
   it converges to `1e-10` in no more steps than CG alone).
8. **A damping floor above one made the misfit search raise.**
   `numerics/tikhonov.py::misfit_search` starts the search at
   `max(1, 10 * minimum_damping)`; `MinimumNorm(minimum_damping=)` goes
   through it. Tests: `tests/test_solvers.py::TestADampingFloorAboveOne`
   (a floor of ten finds the same root, near 230, as no floor; `MinimumNorm`
   with the floor gives the unfloored model).
9. **Preconditioned GMRES stopped on the preconditioned residual.** Fixed
   with item 4: GMRES preconditions from the right, so its residual is
   `b - A x` and `rtol` means what `IterativeSolver` documents. Test:
   `TestGMRESStopsOnTheTrueResidual` (the reviewer's badly scaled
   preconditioner on a 50-dimensional dense metric: converged with the true
   residual within `1e-8 ||b||` and reported as such).
10. **Absolute floors in the diagonal preconditioners.**
    `numerics/preconditioners.py`: `_safe_diagonal` replaces entries at or
    below `floor` times the largest magnitude by that magnitude (all ones
    for a zero diagonal); `JacobiPreconditioner` and
    `NormalDiagonalPreconditioner` use it with `floor` documented as
    relative; `_column_rows` and `sparse_approximation` take
    `_RELATIVE_FLOOR` of the largest entry. Tests:
    `tests/test_preconditioners.py::TestTheDiagonalFloorsAreRelative`
    (Jacobi-preconditioned CG takes the same iterations at scales 1, 1e-16
    and 1e16 on a dense metric; the replacement rule at three scales; the
    sparse pattern under both criteria is the same at 1e-18 as at 1).
11. **`InvariantDistancePreconditioner` failed with `AttributeError`.** It
    refuses any normal operator that is not a `NormalOperator` with a
    `TypeError` naming the preconditioners that read factors instead.
    Test: `tests/test_normal.py::TestTheDistancePreconditionerNeedsAPriorMeasure`
    (Tikhonov and damped members on `Sobolev(16, 2.0, 0.1)`).
12. **The family sweeps kept a member per multiplier.**
    `numerics/root_find.py::DampedSolves.operator` keeps the two most
    recent members (`_KEPT_MEMBERS`); `numerics/preconditioners.py::_WoodburyFamily`
    keeps the two most recent forms (`_KEPT_FORMS`) and records the smallest
    positive multiplier on its own for the zero member. Tests:
    `tests/test_solvers.py::TestAFamilySweepKeepsOnlyRecentMembers`
    (twelve multipliers on a dense metric under Cholesky, eight through a
    Woodbury family; answers exact, at most two kept, zero answered at the
    smallest positive multiplier).
13. **Probe loops that ignored `n_jobs`.** `DampedSolves._matrices` extracts
    with the direct solver's workers (a new read-only `DirectSolver.n_jobs`);
    `NormalDiagonalPreconditioner` forwards `n_jobs` to the zero-scale
    diagonal, the error covariance's diagonal and a family's shift
    diagonal; `SpectralPreconditioner.for_family` applies the factor and the
    shift through `apply_block(n_jobs=)`; `InvariantDistancePreconditioner`
    takes `n_jobs` for the error covariance's diagonal. Tests:
    `tests/test_parallel_knobs.py::TestTheFamilyAndDiagonalProbesForwardTheirWorkers`
    (each site hands `n_jobs=2` to `parallel_map`, and the answer is the
    serial one).
14. **The QP module documented a level bundle method.** The module
    docstring and `best_available_qp_solver` argue from the proximal method's
    simplex subproblem. No test (docstrings).

### Could fix

15. **An unsolved QP reported as solved.** `quadratic_programming.py::_status`:
    only the backend's exact success status is `"solved"`; `unsolved`,
    `solved inaccurate` and `AlmostSolved` keep their names, so the bundle
    method falls back to its own solver for them, as `QPResult.status` now
    says. Tests: `TestOnlySuccessIsSolved` (the mapping; a capped OSQP run).
16. **`TrustRegionNewton`'s radii.** `numerics/optimization.py`: the initial
    radius defaults to the Cauchy length `||g||^3 / (g, H g)` (then
    `||x0||`, then one), the cap to a million times it, the collapse test is
    relative to the larger of the initial radius and `||x||`, and a
    `line_search=` is refused with a `TypeError`. Tests:
    `tests/test_optimization.py::TestTheTrustRegionTakesItsLengthsFromTheProblem`
    (a quadratic at distances of order 1, 1e4, 1e8 and 1e-8 on a dense
    metric in at most two iterations, where 1e8 took 243; the refusal).
17. **`operator_quadratic_form` claimed a role it does not play.** Its
    docstring now says `log_determinant` goes through `random_trace` and
    `OperatorFunction`.
18. **`DampedSolves` took options and private state positionally.** Options
    are keyword-only (`KW_ONLY`), the state fields `init=False`. Test:
    `TestDampedSolvesTakesItsOptionsByName`.
19. **The range finder's measure probes carried the expectation.**
    `numerics/randomized.py::_probe_range` takes the expectation off each
    draw. Test: `tests/test_randomized.py::TestMeasureProbesAreCentered`
    (a rank-two covariance with a mean of size 1e4 off its range: two basis
    vectors, none along the mean, where there were three).
20. **Preconditioners were not checked against the operator's spaces.**
    `IterativeSolver._invert` refuses a resolved preconditioner that does
    not map the operator's codomain to its domain. Test:
    `TestAPreconditionerOnTheWrongSpaces` (CG, GMRES, MINRES; another space
    of the same dimension and one of another dimension).
21. **The dangling history comment** at the end of `preconditioners.py` is
    deleted.
22. **`SolverLike` was a string.** It is a `type` alias.

## Skipped

- **`random_diagonal` and `deflated_diagonal` without an `Estimate`.** D-51
  says they should return one, but `Estimate.value` is a float: this needs
  an array-valued estimate (a design), and the return type reaches
  `symmetric_space/base.py`'s pointwise variance, outside this package.
- **`log_determinant`'s `rtol` against CURRENT_STATE §4.** The docstrings
  are right; renaming (`rtol` to the sampling tolerance) would reach
  `probability/gaussian.py` and `inference/gaussian.py`, which pass
  `sample_rtol` through. The fix is to CURRENT_STATE §4 and D-51:
  `random_trace` and `random_diagonal` take `rtol` to sample to;
  `log_determinant` takes `sample_rtol`, its `rtol` being the inner Lanczos
  tolerance.
- **`damping` meaning `d^2` in LSQR and `t` in the Tikhonov family**, and
  **the tolerance and budget keywords named five ways**: public renames,
  not in SUMMARY.md's decisions.
- **`JacobiPreconditioner` refusing a nonsymmetric operator**: which
  diagonal a nonsymmetric operator's Jacobi should take is a design call.
- **The bundle method's per-run state on the instance, and `with_tolerance`
  on methods it cannot steer**: the first is a refactor the warm-start
  white-box tests depend on, the second a choice between refusing and
  mapping onto each method's own tolerance; neither bears on units.
- **`TrustRegionNewton(max_cg_iterations=)`**: an API addition.
- **The waveform app's hand-built prior-whitened deflation**
  (`factor=` on `SpectralPreconditioner`): an API addition.
- **Tests reaching numerics' private state**: no change required.
