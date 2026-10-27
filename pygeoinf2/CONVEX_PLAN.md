# The feasible-set layer: implementation plan

A working document for the implementing sessions, deleted when the work is
done. It supersedes the plan of 2026-10-24 (in git at `d1afbd7`), whose
stages were overtaken by the design discussion recorded in its last section.
The design is now fixed by `DECISIONS.md` D-135 to D-152 (settled with David
on 2026-10-27, one question at a time); read those first, then D-79 to D-82
and D-134, then this file. The two papers behind the design are Al-Attar
(2021, arXiv:2104.12256) and Mag, Al-Attar, Koelemeijer and Jacob
(submitted 2026); neither is in the repository.

The implementer records any further sub-decision as D-153 onward (never
renumber), runs the fast suite between stages, and never commits: David
commits.

## The binding rules

- **Matrix-free is what the design must not exclude.** The user supplies the
  solver, as everywhere in the library; a dense factorisation is a solver
  they pass, never something an algorithm forms on its own. Nothing dense in
  `N_d` or `N_m` may be a default. Dense in `dim P` is free.
- **The inference sets are clients of the general convex machinery**, not a
  second implementation of it. A specialisation exists only where the sets'
  form admits one (D-136), and the general route must still work.
- **The parity bar is the certified bracket** (D-148): where two algorithms
  apply, their `(inner, outer)` brackets overlap at the instances'
  tolerances. The `SpectralFamilySolver` route reproduces the present
  bisection route's numbers to `1e-10` before the present code is deleted.
- **No decision citations in code** (D-10 as revised 2026-10-26): docstrings
  and comments are self-contained; the reasoning is in `DECISIONS.md`.
- House rules enforced by `tests/test_code_practice.py`: every optional
  argument keyword-only (D-104), American spelling in identifiers (D-100),
  `max_iterations` for budgets and `iterations` only as a count (D-99),
  every documented method's Args complete.
- Metric rule: anything touching inner products is tested on
  `tests/conftest.py`'s `make_dense_metric_space`, not only on Euclidean
  spaces.
- Before handing back: `poetry run ruff check pygeoinf2/`,
  `ruff format --check pygeoinf2/`, and the full fast suite
  (`MPLBACKEND=Agg poetry run pytest pygeoinf2 -q -m "not slow"`, all green,
  about 3070 tests at the start).

## The design on one page

A method takes the operators `(A, T)`, a convex constraint set on the model
space and a convex noise set on the data space; given data it returns the
feasible property set `S_P = T(S_prior ∩ A⁻¹(d − S_noise))` as an object that
answers questions about itself. Every closed convex set has, in principle,
both a support function and a level function; what a set *supplies* cheaply
is declared by its `has_*` flags, and the convex machinery converts between
the two where that is affordable (D-135). The sets' **form** chooses the
algorithm (D-136); `via=` on a probe forces one (D-143).

| tier | when | bounds (`support`) | membership (`contains`) |
|---|---|---|---|
| closed form | exact data, quadratic prior | the ellipsoid of Al-Attar (2021) eq. 2.84, in the prior's inner product; `dim P + 1` minimum-norm solves, then algebra | the ellipsoid's quadratic form |
| quadratic | ball or ellipsoid on each side (degenerate forms admitted, D-138) | `Primal`: two multipliers around the damped family `γQ + A*PA`, one eliminated analytically (D-144); contact point and extremal model for free | `Search`: Al-Attar (2021) §3.3, one multiplier around the same family restricted to `ker T` (D-137) |
| general | anything else, by capability | `Dual` given support functions and maximisers (Mag eq. 28, proximal bundle in the noise metric, D-149); `Splitting` given projections (Chambolle-Pock) | `Search` with a convex minimisation inside, given smooth level functions; else `Distance`, the `dim P` concave maximisation from the support side |

From bounds: polytopes (outer), the hull of contact points (inner), and the
S-procedure outer ellipsoid on the quadratic tier (D-141). From membership:
extents along lines through the fitting model, and admitted-point hulls.
`BackusGilbert` is the restricted-certificate special case and returns the
Minkowski-sum set (D-140). The feasible set and its algorithms live in
`geometry`, the damped-family machinery in `numerics`, the estimators in
`inference` (D-142). Every answer carries a certified bracket (D-148), and
reported values are the outer ones.

## Stage 0 — the numerical experiment, and the solver-level groundwork

Nothing user-facing changes in this stage.

1. **DONE (D-153).** The one-multiplier reduction is sound: with `s = γt`
   the dual minimised over `t` is closed form and `φ(γ) = B + sqrt(AC)` is
   unimodal in every direction of twelve problem families when evaluated
   in the data-space formalism through the resolvent identity; about 11
   evaluations per direction warm-started. `benchmarks/stage0_unimodality.py`
   and `benchmarks/stage0_checks.py` are the record. The nested root finds
   stay as the fallback (D-144) but are not expected to run.
2. **DONE.** `SpectralFamilySolver` in `numerics/solvers.py`: an
   `EigenSolver` standalone, and through `for_family(base, shift)` one
   whitened eigendecomposition serving every multiplier; the pseudo-inverse
   at `t = 0` is the minimum-norm solution in the shift's inner product.
   The whitening by the noise precision's factor is its business, not the
   route's.
3. **DONE.** `BundleResult`, `SaddlePointResult`, `KKTResult` are
   `OptimizationResult` subclasses (`value = -(c, m)` on the maximizing
   solvers, `minimizer` the model, `gradient_norm` NaN, `history` the best
   value or the residual per iteration); `ProximalBundleMethod` and
   `LevelBundleMethod` are `Optimizer`s with `minimize(functional, x0, /, *,
   subgradient=None)`; readers in `backus.py` and the tests re-spelled;
   the field-comparison test is an `issubclass` test.
4. **DONE.** The protocol is `FamilySolver.for_family(base, shift) ->
   BoundFamily` with `BoundFamily.at(multiplier, member) -> InverseOperator`,
   serving solvers and preconditioners alike; `DampedSolves` binds once and
   asks `at()` for each member, ahead of the `refresh` rule, and skips the
   member-matrix assembly for a family solver. `NormalDiagonalPreconditioner`
   (base diagonal probed off the first member's factors, the shift's
   diagonal read once) and `SpectralPreconditioner` (base deflated once, the
   shift's quadratic form on each resolved mode) implement it;
   `WoodburyPreconditioner` does not need to, since its build is lazy and
   its inner factorization is of surrogate dimension, so `refresh=1.0`
   already makes it per-member and cheap (D-146 corrected in place).

Fast suite green; D-entries for 1 and any surprise.

## Stage 1 — relocate the damped-family machinery (D-142)

**DONE 2026-10-27.** `numerics/normal.py` and `numerics/tikhonov.py` (the
latter now holding `misfit_search`), the structure-aware preconditioners
appended to `numerics/preconditioners.py`; `probability` is imported only
under `TYPE_CHECKING`, so D-111 holds at runtime. `pygeoinf2.inference`
re-exports every moved name; `pygeoinf2.numerics` exports them too. The
discrepancy searches stay in `inference/point.py` (they read the problem's
chi-square). Tests import from the new home.

## Stage 2 — the geometry layer gains what any convex set can use

**DONE 2026-10-27, except `Cylinder`.** In `geometry/convex.py`: the
support-function conversion (`_LevelSupport`, one multiplier around a
convex minimization, infinite where the set is unbounded) behind
`ConvexSet.support_function` and `support_maximizer` for any set with a
differentiable level function; the projection conversion
(`_project_by_support`, the Fenchel dual of the nearest-point problem,
strongly convex, by the proximal bundle) behind the oracle set's `project`
and `contains`, accurate to the square root of the bundle's gap tolerance;
`Ellipsoid` admits a semidefinite precision (`is_bounded` says which) and
falls back to the conversions without a covariance. The `has_*` flags stay
"supplied cheaply". In `geometry/feasible.py`: `SupportResult`,
`InclusionResult`, `SweepResult`; the shared `_Structure` with its
`form`; the `Algorithm` base; `ClosedForm` for any quadratic prior with
its centre, the inclusion level through the shape `S` on the property
space (no joint spectrum, nothing dense in `N_d`), the level function's
gradient from `S`; `FeasiblePropertySet` with every probe of D-143,
dispatch by form, `via=` by name or instance, value-keyed caches under
reentrant locks, `with_data`. Exported from `pygeoinf2.geometry`; not yet
at the top level (the inference set still owns that name until Stage 5).
Tests in `tests/test_feasible.py`: parity with the present closed form,
the ellipsoid prior against its whitened ball, centres, dispatch and
refusals, sharing, probes, the conversions.

Still to do here: **`Cylinder`** (D-138), a quadratic set with an explicit
kernel basis for the true-kernel case; deferred to after Stage 4, since
its purpose is the affine constraint on the dual's certificate, and
droppable.

## Stage 3 — the quadratic tier, matrix-free

**DONE 2026-10-27.** In `geometry/feasible.py`: `_QuadraticFamily` (the
damped family of the two sets in either formalism, members
`DampedNormalOperator`s from `numerics/normal.py` carrying `A`, `C`, `γR`
for the preconditioners, the Tikhonov-family interface so `misfit_search`
runs on it); `Primal` (the `φ(γ)` minimization of D-153 by bracketing and
Brent in `log γ`, the resolvent form in the data space, the prior-only
branch at no solve, `γ = 0` evaluated when the walk reaches the lower limit
in the model space, the certificate `λ = t·R⁻¹(Au − d′)` evaluated in closed
form as the outer bound, the inner value from the largest feasible point on
the ray `u₂ + x u₁` at the found `γ` — both constraints are quadratics in
`x`, D-144 — with the segment to the fitting model as the fallback);
`Search` (feasibility and membership by one misfit search, the kernel
restriction a rank-`dim P` downdate of `C` applied matrix-free, the lower
bound from the multiplier); caches: `u₂` by `γ` with warm starts within six
decades, the warm start of the scalar search per worker thread; the sweep
in contiguous chunks on threads (`parallel_map(threads=True)`, new in
`parallel.py`); the floor check of D-148. Defaults settled by the tests:
`CGSolver(rtol=1e-10, max_iterations=max(2·dim, 200))` per space, since
`1e-12` is below the rounding floor of an ill-conditioned family; `Primal`'s
tolerance `1e-4` in `log γ` (brackets of about `1e-5`). The data-space
formalism is refused by dimension when the data outnumber the model
(its family is singular at `γ = 0`), with the model-space advice; a solver
breakdown inside `φ` surfaces as the same advice. The kernel-restricted
search is assembled in the data space whatever the formalism, since the
restriction is the downdate there. Parity: outer values agree with the
present bisection route to `1e-8` with CG and with the spectral solver;
membership levels to `1e-7`; the closed form is the limit of vanishing
noise; the overdetermined case finds `γ = 0` with the prior slack.

Not done here: the structure-aware preconditioners' sharing between the
plain and the restricted family (each `DampedSolves` binds its own; a
family-aware preconditioner is probed twice). Revisit with the benchmark.

After the code-practice scan of the same day (`CODE_PRACTICE_SCAN.md`):
an `Algorithm` works through the public `feasible.structure`
(`FeasibleStructure`, exported) and `feasible.cached(key, build)`, and an
estimator builds sets with `FeasiblePropertySet.from_structure`; the floor
check reads `IterativeSolver.rtol`; `with_resolved_preconditioner` is the
public name of the preconditioner swap; the level-support memo is
thread-local.

## Stage 4 — the general tier

**DONE 2026-10-27.** In `geometry/feasible.py`: `Dual` (the bundle on a
`MassWeightedSpace` of the data space with the noise precision as metric
when the noise set is quadratic, the gradient mapped by the covariance;
`method=` any subgradient-taking optimizer; certificates warm-started per
worker thread, the opposite direction's negated when seen; an unbounded
dual raised as an empty set and used as the feasibility probe on the
support-only tier; the prior's maximizer at the residual `T*q − A*λ` as the
extremal model, reported as the inner value when it fits the data);
`Splitting` (Chambolle-Pock on projections, exact data as the zero-radius
ball, the outer value the better of the dual cost at `±` its dual variable
where the sets have support functions and infinite otherwise); `Distance`
(membership through the projection dual on the set's own support function
with contact points as subgradients; the level nominal, one inside and
infinite outside); `Search` extended to smooth level functions on both
sides (the likelihood route's inner minimization, the model-metric kernel
projector for a value); `SublevelSet` priors and noise admitted; the
conversions public as `ConvexSet.has_smooth_level_function`,
`computes_support_function`, `computes_maximizer`; `_dual_cost` through
the sets' support functions when they are not quadratic. Parity: the dual
agrees with the primal to `1e-6` on balls with overlapping brackets, with
the present KKT route to `1e-4` on a quartic prior, with the closed form
on exact data and an oracle prior to `1e-6`; the splitting agrees with the
primal to `1e-5`; the general search agrees with the present likelihood
route to `1e-5`. The dispatch order is `support: closed_form, primal, dual,
splitting`; `inclusion: closed_form, search, distance`; `feasibility:
closed_form, search, dual`.

Not done here: the affine constraint on the dual's certificate for
degenerate priors (D-138, with `Cylinder`), and an inner value for the
dual when the prior's maximizer does not fit the data (it is reported as
`−∞` then).

## Stage 5 — the estimators, and the deletions

**DONE 2026-10-27.** `inference/backus.py` rewritten over the geometry
layer: `BackusGilbertParker(problem, T, prior, /, *, noise=None,
level=0.95, solver=None)` resolves the confidence set (the problem's set,
the ambient ball of a Gaussian at `level`, or exact data), owns one
`FeasibleStructure`, and `__call__` returns
`FeasiblePropertySet.from_structure` on it, so sets for different data
share the data-independent work; `form` and `structure` on the estimator,
`push_forward` kept. `BackusGilbert` is a `SetEstimator`: the automatic
`X = T C A*(A C A* + α R)⁻¹` for a ball or an ellipsoid on each side with
centres (`alpha=` the trade-off, one by default), `from_operator(problem,
T, X, prior, noise=)` for any sets with support functions, and `__call__`
returns a `BackusGilbertSet`, the Minkowski sum `X d + (T − XA) S_prior +
(−X) S_noise` as a `ConvexSet` with `estimate`, `error_bars()`, `parts`,
`support`, and membership and projection through the support-function
oracle; `operator`, `resolution`, `unresolved`, `point_estimator`,
`estimate(data)` on the estimator. SOLA is named in the docstring, the
example and the tutorial. `harden_error` kept. Deleted: the five old
routes, the two choosers and the inference `FeasiblePropertySet` (the
geometry one is re-exported); `PrimalKKTSolver`, `LevelKKTSolver`,
`KKTResult`, `LevelBundleMethod` from `numerics/convex.py`.
`SaddlePointResult` stays: the splitting's certificate and discrepancy
have no other home (D-150). Tests re-spelled with `via=`; the old parity
bars are recorded in the test docstrings. Example 23 and tutorial 7 show
the bracket, the inner hull beside the polytope, and the SOLA sentence.
Top-level exports: 109 names (108 before; `BackusGilbertSet` added).

Fixed on the way: `ClosedForm.inclusion` proves unreachability on a flat
set (a least-squares solve with a rank cut, where the plain solve gave a
huge negative level); `Search` falls back to the model-space fit when the
data-space family is singular by dimension (more data than the model, or
than the kernel of `T`); `Dual` normalizes the scale of the noise metric
(a tiny noise ball reached the bundle as a huge proximal weight and
stopped it at once); `_LevelSupport` handles the zero direction;
`Distance` applies only where the support function is finite.

Open from here: the primal's inner value can be loose where the data
barely constrain a direction (example 23, cap 0: bracket width 0.8 on a
value of 6.4), and the default conjugate gradients do not reach their
tolerance on that example's family, which the example meets with
`solver=SpectralFamilySolver()`; both for Stage 7's benchmarks.

## Stage 6 — ellipsoidal probes (D-141)

**DONE 2026-10-27, all four items.** `SProcedure` in `geometry/feasible.py`
answers a new probe `outer_ellipsoid` (`ClosedForm` answers it with the set
itself); `FeasiblePropertySet.outer_ellipsoid(*, criterion=, via=)` and
`outer_ellipsoid_result` (an `EllipsoidResult` with the weight `θ` and the
evaluation count). The search is in `log γ`, `γ = θ/(1 − θ)`, from the
Rayleigh scale of `A C A*` against `R`, by the primal's bracketing-then-Brent
minimizer; both criteria were unimodal on 300 random problems. The two
formalisms agree on `θ` to `1e-11`. In `geometry/convex.py`:
`sum_of_ellipsoids(space, covariances, *, center, criterion)` (trace in
closed form for any number of summands, volume by a scalar search for two),
`(E₁ + E₂).outer_ellipsoid()` on a Minkowski sum of balls and ellipsoids,
`Polytope.inscribed_ellipsoid()` (Newton on the log barrier from the
Chebyshev centre, after an LP check that the polytope is bounded) and
`enclosing_ellipsoid(space, points)` (Khachiyan with away steps, inflated
to the furthest point so it encloses whatever the tolerance). All three are
dense in `dim P` and metric-aware (tested on a dense Gram).
`BackusGilbertSet.outer_ellipsoid()` is the sum bound about the estimate,
available when both sets were balls or ellipsoids with covariances. Example
23 and tutorial 7 draw the S-procedure ellipsoid; on tutorial 7's arc 0 its
half-width is 0.34 against the Backus-Gilbert ellipsoid's 0.46.

Observed, not a defect: a Backus-Gilbert ellipsoid can be narrower than the
S-procedure's in some directions (the criteria are global); the S-procedure
is tighter on the criterion and contains the set, which is the claim.

1. `feasible.outer_ellipsoid(*, criterion="trace" | "volume")` on the
   quadratic tier: the S-procedure ellipsoid `T C_θ T*`,
   `C_θ = (θQ + (1 − θ)A*PA)⁻¹`, `dim P` warm-started family solves per
   `θ`, a coarse one-dimensional search in `θ` with the solves warm-started
   along it; returns an `Ellipsoid` on the property space, certified outer.
2. The Minkowski sum of two ellipsoids gains `outer_ellipsoid()` by the
   trace-optimal family `(1 + 1/p)Q₁ + (1 + p)Q₂`, closed form; this is
   `BackusGilbert`'s set when both inputs are quadratic.
3. `Polytope.inscribed_ellipsoid()`: the maximum-volume inscribed
   ellipsoid by the log-det program (Boyd and Vandenberghe §8.4.2) in
   `dim P`; documented as a description of the polytope, not of the set.
4. A point-set minimum-volume enclosing ellipsoid (Khachiyan) in
   `geometry`, documented as enclosing the inner hull and certifying
   nothing.

Items 3 and 4 are droppable if the release schedule needs it (D-141: C if
workable, B the floor).

## Stage 7 — benchmarks, mini-app, documents (D-152)

**In progress, 2026-10-27.** Done so far: `benchmarks/_common.py`
(application counter, timing, tables), `point_data_sphere.py` (restructured
around the noisy quadratic tier per D-154, the closed form and the
noise-to-zero sweep behind `--limit`), `flexure_gravity.py` (the example-22
chain on an `L²` amplitude ellipsoid and the chi-squared noise set, cap
averages in field units, truncations as a sweep), `phase_delay.py` (nested
catalogue acquisition, spectral against CG plain, deflated and
surrogate-Woodbury), `operator_timings.py` (the three `work/` harnesses
ported); `tests/test_benchmarks.py` runs each at a reduced size under
`slow`. On the way: `WoodburyPreconditioner` is family-aware (`for_family`
/ `at`, refactorizing the surrogate-dimension inner operator per
multiplier, with a positive fallback at zero); the primal's kernel-part
solve moved from `γ = 0` to a floor `Primal(floor=1e-4)` of the family's
scale, since the undamped `A C A*` is the member CG cannot solve; the
dual runs in the noise covariance's metric (the curvature of its noise
term) rather than the precision's, and starts from a scalar line search
along its first subgradient that also sets the bundle's proximal weight.
Tutorial notebooks generated by `tutorials/make_notebooks.py` with Colab
badges (David's request), tutorial 8's prior passage rewritten.

Open from the small runs: the dual does not converge on the flexure
problem (610 data in two blocks of different units, chi-squared
ellipsoid noise) even at 1500 bundle iterations, its inner QP warning
throughout; the primal is certified there and is what the benchmark
reports. Unpreconditioned CG at `rtol=1e-10` needs more than twice the
data count in iterations on the phase-delay family and a quarter-degree
surrogate brings it to tens. Paper-scale numbers go into DECISIONS'
appendix once the runs finish; then the mini-apps (flexure–gravity,
tomography, point data, per David) and the document pass.

1. **`benchmarks/`** (new, outside the test suite, run by hand): the three
   problems of D-152 at paper scale — point data on the sphere (Al-Attar
   2021 §2.5.2 and §3.3.2), flexure–gravity (Mag §3.2, Table A2; the
   degree-diagonal gravity and flexure kernels of its Appendix B), and
   phase-delay tomography with nested acquisition (Mag §3.1, Fig. 9).
   Each script records application counts and wall time for the spectral
   route, unpreconditioned CG, deflated CG (10c's C, the `SpectralPreconditioner`
   by `random_eig`) and surrogate-Woodbury CG, with `n_jobs=1` and `k`.
   Seeds: `work/sphere_dli_example.py` (phase delay, real catalogues) and
   `work/dynamic_topography.py` (flexure–gravity, variable rigidity, the
   surrogate Woodbury preconditioner), both written against v1; the timing
   harnesses in `work/` join this directory.
2. **Measurements into DECISIONS.md's appendix**, and the defaults they
   set: whether deflation becomes the data-space default and at what rank,
   the `refresh` factor, any crossover between the spectral solver and CG.
3. **Reduced versions as slow tests** with the four acceptance criteria of
   D-152.
4. **Mini-app**: the flexure–gravity problem fills EXAMPLES_PLAN's
   "two fields, one chain" slot, end to end with a published figure to
   match.
5. **Documents**: CURRENT_STATE §2 (package map: `geometry/feasible`,
   `numerics` damped family), §4 (the tier table replaces the two route
   tables), §5 (dense inventory corrected: the spectral solver is the only
   dense-in-`N_d` piece and is opt-in; `LevelKKTSolver`'s entry was wrong),
   §7 (D-134 closed), §10 (item struck); DECISIONS.md D-82's open sentence
   closed by D-136; `EXAMPLES_PLAN.md` coverage note updated; this file
   deleted.

## Acceptance (D-152)

1. Parity by brackets on every problem where two algorithms apply; the
   spectral route reproduces the present numbers to `1e-10`.
2. The phase-delay problem at `N_d = 9000` runs its twelve support values
   matrix-free with the surrogate preconditioner; application count and wall
   time recorded against the spectral route at the same tolerance.
3. The flexure bounds change by under a few per cent between the two finest
   truncations.
4. `n_jobs=1` and `n_jobs=k` agree to tolerance; `via=` changes nothing but
   the algorithm name and the cost.

## Open, to be settled by the implementation

- Whether the one-multiplier reduction is monotone (Stage 0.1); the
  fallback is in either way.
- Whether `Cylinder` is worth its constructor (Stage 2.2); droppable.
- The name of `geometry/feasible.py` and of the module the family machinery
  moves to in `numerics`.
- Whether deflation becomes the data-space default (Stage 7.2); measured,
  not guessed.
- Process-backend parallelism, process-safety declarations, and the
  randomised low-rank `SpectralFamilySolver`: post-release (sl3dpp track).
