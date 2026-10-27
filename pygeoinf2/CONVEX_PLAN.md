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

Move `inference/normal.py` (`FactoredNormalOperator`, `NormalOperator`,
`choose_formalism`), `inference/tikhonov.py` (`TikhonovNormalOperator`,
`TikhonovFamily`), `inference/preconditioners.py`, and `misfit_search` with
its helpers from `inference/point.py`, into `numerics` (a module or small
subpackage; names unchanged). `inference` re-exports every moved name so
`pygeoinf2.inference.TikhonovFamily` and friends keep working;
`LeastSquares`, `MinimumNorm`, `DiscrepancyPrinciple`, the Gaussian
inversion and `laplace.py` import from the new home. Nothing in `numerics`
may import `inference` or `probability` afterwards (D-111): check with the
import-cost and layering tests. Docstrings of the moved modules lose any
"inference" framing that no longer fits.

Fast suite green; CURRENT_STATE §2 package map updated.

## Stage 2 — the geometry layer gains what any convex set can use

1. **The conversions (D-135)**, in `geometry`/`numerics`: a support
   function and maximiser for any set with a smooth level function, by the
   one-multiplier maximisation of `(q, x)` subject to the level constraint
   (the one-constraint case of `LevelKKTSolver`'s machinery; the Newton
   inner solve stays matrix-free); and the distance of a *finite-dimensional*
   set from its support function and maximiser, `d(p) = max over ‖q‖ ≤ 1 of
   (q, p) − h(q)`, a concave maximisation with supergradient `p − p̂(q)`,
   which gives membership with a certificate either way. `ConvexSet` grows
   the two as default implementations behind the existing flags, so a set
   that supplies neither cheaply still answers, and the answer says it is
   converted.
2. **Degenerate quadratic sets (D-138).** `Ellipsoid` accepts a precision
   claiming `POSITIVE_SEMIDEFINITE` when built for inference use; its
   support function is `+∞` off the right subspace and says so; its
   maximiser raises there. A `Cylinder` (quadratic set with an explicit
   kernel basis) for the true-kernel case, same behaviour, planned and
   droppable if it proves hard. `+∞` is a legal value of `support()`
   everywhere, and `Polytope` construction skips a half-space with infinite
   offset and records that it did.
3. **Probe results (D-150)**: `SupportResult`, `InclusionResult`,
   `SweepResult` as frozen dataclasses in `geometry` (fields in D-150).
4. **`FeasiblePropertySet`** in `geometry/feasible.py` (name open), built
   directly from `(A, T, prior, data, *, noise=None, solver=None)`, `noise=None`
   meaning exact data; `form`, `algorithms` and the read-only structure
   (`prior`, `noise`, `data`, `forward_operator`, `property_operator`);
   every probe of D-143 as a method that dispatches by form and `via=`,
   initially with only `ClosedForm` behind it.
5. **`ClosedForm`**, generalised from a ball to any quadratic prior
   (minimum-`Q`-norm model, `Q`-orthogonal projection onto `ker A`), centres
   included, `dim P + 1` solves through the supplied solver; serves bounds,
   membership, projection and `ellipsoid`.

Tests: the conversions on `make_dense_metric_space`; the closed form against
the present `_ClosedFormRoute` on balls, and on an ellipsoid prior against
the quadratic tier once Stage 3 lands; `+∞` handling.

## Stage 3 — the quadratic tier, matrix-free

1. **`Primal` (D-144, D-153)**: minimise `φ(γ) = B(γ) + sqrt(A(γ)C(γ))`
   over `γ ∈ [0, ∞]` around a `TikhonovFamily` built with the prior's `Q`
   and the noise's `P` (`formalism="data_space"` by default, D-147;
   `"model_space"` on request and required when the prior has no
   covariance). In the data-space formalism the three coefficients are
   `A = κ/γ + (ACc, G⁻¹RK⁻¹ACc)` with the kernel part `κ` once per
   direction, `B = (ACc, K⁻¹d′)`, `C = 1 + γ(1 − (d′, K⁻¹d′))`, where
   `K(γ) = G + γR` is the family member; in the model-space formalism the
   two direct solves. A bracketed Brent in `log γ` from the neighbouring
   direction's root (cold: expand from `γ = 1`), tolerance `1e-3` in
   `log γ` (the value is stationary, second order); `γ = ∞` (prior-only,
   zero solves) and `γ = 0` (data-only, when `A*PA` is invertible) tested
   as candidates; every solve warm-started through `DampedSolves`, with
   `K⁻¹d′` cached by `γ` (D-145). The nested monotone root finds remain as
   a fallback behind the same object. At the minimum `t* = sqrt(A/C)`,
   `s* = γt*`, the model `u = u₁/t* + u₂`. Returns a `SupportResult` with
   the outer value from the dual certificate `λ = t*·P(d′ − Au)` evaluated
   in closed form, the inner value from the model scaled into both sets,
   the contact point, the extremal model and the multipliers (D-148).
2. **`Search` (D-137)**: the membership and feasibility algorithm. Without
   a property value: `misfit_search` over the family on the shifted data,
   giving the fitting model and emptiness. With a value `p`: the
   minimum-`Q`-norm model with property `p` through the cached kernel
   pieces, then the same search with the forward operator restricted to
   `ker T` (`A ∘ P_K`, `P_K` applied matrix-free). Returns an
   `InclusionResult` with the level, its lower bound from the multiplier,
   and the fitting model; `level_function()` carries the gradient the
   multiplier gives.
3. **Sharing (D-145)**: the estimator-level part (kernel pieces, closed-form
   pieces, the family with its solver and preconditioner) built once and
   handed to every set; the set-level part (a bounded `u₂(γ)` cache keyed
   on `γ` to the solver's tolerance, the last roots, `InclusionResult`s)
   under a lock; `support_values` sweeps in the order given with
   warm-started roots in contiguous chunks per worker (D-151);
   `with_data(d₂)` returns a sibling sharing the estimator-level part.
4. **Preconditioning (D-146)**: the family is a `FactoredNormalOperator` so
   every existing preconditioner applies; `C`-preconditioning is the
   model-space default, none the data-space default; the kernel-restricted
   family reuses the unrestricted family's preconditioner.
5. **The floor check (D-148)**: at construction, the solver's `rtol` against
   the noise radius over `‖d‖` and the prior radius over the model scale;
   refuse under D-8's strictness, warn otherwise.
6. **`SpectralFamilySolver` parity**: with it passed as `solver=`, `Primal`
   and `Search` reproduce the present `_BisectionRoute` numbers to `1e-10`
   on every existing test; with CG they agree to the bracket.

Tests: brackets overlap between `Primal` with CG and with the spectral
solver; `ClosedForm` against `Primal` as the noise radius → 0; centres and
ellipsoids on both sides; the degenerate prior on a reduced flexure problem;
`n_jobs` invariance; the floor check.

## Stage 4 — the general tier

1. **`Dual` (D-149)**: the bundle run on a `MassWeightedSpace` of the data
   space with the noise precision as metric; `method=` any nonsmooth-capable
   `Optimizer`, the proximal bundle default; stopping on the certified
   bracket when an inner point exists and on the bundle gap otherwise;
   certificate warm starts across a sweep with the antipodal rule for a
   symmetric noise set; emptiness by a divergence test on the cost; an
   optional affine constraint on `λ` enforced in the proximal step for
   degenerate priors; refusal of sets whose support function is infinite on
   a non-subspace region (those go through `Splitting`); `start=` for
   cross-seeding from a `Primal` certificate.
2. **`Splitting`**: Chambolle-Pock on the support problem given projections,
   returning the primal iterate scaled in as the inner value and the dual
   iterate's cost as the outer.
3. **`Search` on general sets**: the inner convex minimisation of the
   present `_LikelihoodRoute`, under the same outer search.
4. **`Distance`**: Stage 2's conversion as the membership algorithm of the
   support-only tier, and the emptiness fallback there.
5. The general tier's `algorithms` listing by capability (D-135, D-136).

Tests: the existing cross-route parity tests re-spelled as `Primal` against
`Dual` against `Splitting` with brackets; the ellipsoid-prior-with-exact-data
case that crashes today; a polytope prior through `Dual`; a projection-only
intersection through `Splitting`; `Distance` against `Search` where both
apply.

## Stage 5 — the estimators, and the deletions

1. **`BackusGilbertParker`** (D-143): signature `(problem, T, prior, /, *,
   noise=None, level=0.95, solver=None)`; resolves the noise set as today
   (the problem's set, the ambient ball of a Gaussian at `level`, or exact);
   builds and owns the estimator-level part; `__call__` returns the geometry
   object. `route`, `membership`, `algorithm`, `method`, `optimizer`,
   `max_iterations` go. `push_forward` keeps working.
2. **`BackusGilbert`** (D-140): accepts a ball or ellipsoid on each side,
   centres included; `X = T C_Q A* (A C_Q A* + α C_R)⁻¹` with the user's
   solver, `dim P` data-space solves; `__call__` returns the Minkowski-sum
   `ConvexSet` built from the existing sum and image machinery, with the
   estimate as its centre, `error_bars` and `estimate` on it;
   `from_operator(problem, T, X, prior, noise=)` evaluates a given
   certificate against any sets with support functions. `resolution`,
   `unresolved` stay. The docstring and tutorial 7 say what SOLA is.
3. **Delete**: `_ClosedFormRoute`, `_BisectionRoute`, `_DualRoute`,
   `_LikelihoodRoute`, `_minimum_norm_fits`, `_Quadratic` and the two
   choosers from `inference/backus.py`; `PrimalKKTSolver`,
   `LevelBundleMethod` and the smoothed dual from `numerics/convex.py`;
   `KKTResult` and `SaddlePointResult` as separate types. The old parity
   figures are kept in a test comment as the bar they set.
4. **Tests re-spelled**: `test_backus`, `test_quadratic_programming`,
   `test_parallel_knobs`, `test_inference`, `test_convex`; forced routes
   become `via=`; `auto` picks the documented algorithm for each form; a
   refused `via=` names the admissible ones; an instance with tuning is
   honoured.
5. **Examples and tutorials**: example 23 and tutorial 7 use the automatic
   choice and need wording only (SOLA sentence; outer value and bracket;
   inner hull from contact points beside the polytope); the printed
   `route=` of any example goes.

Fast suite green; `inference/__init__.py` and the top-level exports
updated (108 names before; record the new count).

## Stage 6 — ellipsoidal probes (D-141)

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
