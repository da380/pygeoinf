# Review: pygeoinf2 inference / inversion layer against v1

> **Note (2026-08-27):** the decisions recorded in `pygeoinf2/REVIEW.md` §11 (D-1 … D-13) override the Must/Should/Consider ranking below wherever they conflict — in particular D-1 (sphere vectors are `SHGrid`, `sampling=1` default), D-2 (points in `(lat, lon)` degrees), D-3 (per-geometry submodules with `Lebesgue`/`Sobolev` subclasses), D-4 (`from_matrix(..., form=)`), D-6 (parallel hooks around operators), D-12 (path *integral* operator), D-13 (convex solvers restored).

Scope: `pygeoinf/{forward_problem,inversion,linear_bayesian,linear_optimisation,backus_gilbert}.py` (v1) vs `pygeoinf2/inference/*` (v2), with DESIGN.md §18, §21.9–10, §22–28, §31.2 and V1_CATALOGUE.md checked against the code. Five claims were verified by execution; results are marked **[verified]**.

---

## 1. Functionality: retained / extended / lost

### 1.1 `LinearForwardProblem` (v1 12 methods → `inference/problem.py`)

| v1 | v2 | Status |
|---|---|---|
| `from_direct_sum` (fp:128) | `from_direct_sum` (problem.py:193) | Retained. Same silent defect as v1: if *any* member lacks an error the joint problem becomes `error=None` (problem.py:212–219) — a noisy dataset joined with an exact one silently loses its noise **[verified]**. Also requires every error to be a `GaussianMeasure` (`Gaussian.from_product`), refuses set-valued errors with an `AttributeError`. |
| `data_measure_from_model` / `..._from_model_measure` / `joint_measure` | problem.py:222/226/235 | Retained; `rng=` plumbed through (extension). |
| `synthetic_data` / `synthetic_model_and_data` | problem.py:117/266 | Retained, now on `ForwardProblem` too (works with nonlinear operator). |
| `chi_squared*`, `critical_chi_squared`, `chi_squared_test` | problem.py:124–162 | Retained; `consistency_set` added (:164). `chi_squared_from_residual` (:143–146) subtracts and re-adds the expectation before calling `mahalanobis_squared`, which subtracts it again — correct but obfuscated. |
| `parameterized_problem(dense=, parallel=, n_jobs=)` | `parameterised(P)` (:277) | `dense=` **dropped** (v1's `with_dense_matrix`). `LinearGaussianInversion.parameterised(**kwargs)` forwards kwargs to a method that accepts none → `TypeError` **[verified]** (gaussian.py:299–317). |
| `data_reduced_problem` | `data_reduced` (:287) | Retained minus `dense=`. Set-valued error → `AttributeError` at :305. |

Extension: `error` may be a `Subset` (set-valued data relation, §18.1). But almost every method on the class then raises `AttributeError` from the `error_measure` property (:87–93) rather than a typed error saying "this needs a measure".

### 1.2 `LinearBayesianInversion` (24) → `LinearGaussianInversion` + `NormalOperator` + `preconditioners.py`

| v1 | v2 | Status |
|---|---|---|
| `model_prior_measure`, `data_prior_measure`, `joint_prior_measure` | `prior`, `data_prior`, `joint_prior` (gaussian.py:200/158/167) | Retained. |
| `normal_operator` (lb:157) | `NormalOperator` class (normal.py:136), `.normal_operator` (gaussian.py:177) | Retained and improved: factors kept, `surrogate`, `gain`, `posterior_covariance`, `right_hand_side` moved onto it. |
| `get_normal_equations_rhs` | `NormalOperator.right_hand_side` (normal.py:399) | Retained (takes the *shifted residual*, not the data — caller must shift). |
| `kalman_operator(solver, preconditioner)` | `.gain` property (gaussian.py:146) | Retained; solver now a constructor argument. |
| `model_posterior_measure(data, solver, preconditioner)` | `inversion(data)` (estimators.py:227) | Retained; sampler now data-independent (centred RTO, gaussian.py:467–489). |
| `posterior_expectation_operator` | `.mean_map` | Retained (always affine; v1 returned bare `LinearOperator` in the zero-mean case). |
| `with_formalism` | gaussian.py:225 | Retained; default `"data_space"` as v1; `"auto"` opt-in. |
| `normal_residual_callback` (lb:460) | **none on `LinearGaussianInversion`** | Lost here; `LeastSquares.residual_callback` exists (point.py:263). Substitute: `CGSolver(callback=)`. Catalogue row says "Ported" — only for least squares. |
| `mahalanobis_evidence_term`, `log_evidence` (lb:493, 688) | `mahalanobis`, `evidence_terms`, `log_evidence` (gaussian.py:331/428/442) | Retained, matrix-free, Woodbury in model space and Sylvester for the log-det (:371–426). v1's SLQ knobs (`size_estimate`, `block_size`, `parallel`, `n_jobs`) collapse to `samples`, `max_iterations`, `rtol`. `parallel=` **dropped**. |
| `estimate_log_determinant`, `_trace_log_slq` | `numerics.functional_calculus.log_determinant` | Retained; `"auto"` goes **dense** when `dim ≤ 512` (fc:460–464) — so on a modest data space `log_evidence` assembles `A Q A*+R` (dim(D) applications) regardless of the solver the user chose. Not wrong, but not what "matrix-free" suggests unless `method="stochastic"` is passed. |
| `diagonal_normal_preconditioner(blocks, parallel)` | `NormalDiagonalPreconditioner` (preconditioners.py:62) | Retained; `parallel=` dropped. |
| `sparse_localized_preconditioner` | `LocalisedPreconditioner` (:161) | Retained; `parallel=` dropped. |
| `woodbury_data/model_preconditioner` | `numerics.preconditioners.WoodburyPreconditioner` | Retained. |
| `surrogate_inversion` | `.surrogate` → `NormalOperator` (gaussian.py:239) | Retained; returns the operator not an inversion (deliberate, §23.2). |
| `surrogate_normal_preconditioner`, `surrogate_woodbury_*` (3) | subsumed: `solver(inv.surrogate(...))`, `WoodburyPreconditioner.from_normal(inv.surrogate(...))` | Subsumed, fine. |
| `low_rank_surrogate` | gaussian.py:260 | Retained. |
| `parameterized_inversion(parameter_prior=None → auto pull-back, dense=, formalism=)` (lb:1353) | `parameterised(P, *, prior)` (gaussian.py:299) | `prior` now mandatory (v1's auto "pull-back" via `P*` was not a pull-back anyway — defensible). `dense=`, `formalism=` override dropped. Catalogue says "Dropped"; DESIGN §23.7 says "lifted onto the inversion". Code has them; catalogue stale. |
| `data_reduced_inversion` | gaussian.py:319 | Retained minus `dense=`/`formalism=`. |

### 1.3 Least squares / minimum norm / constrained (`linear_optimisation.py` → `point.py`, `tikhonov.py`)

| v1 | v2 | Status |
|---|---|---|
| `LinearLeastSquaresInversion.normal_operator(damping)`, `normal_rhs` | `TikhonovNormalOperator`, `.right_hand_side` (tikhonov.py:46, 248) | Retained, plus `TikhonovFamily` for sweeps (:297). |
| `least_squares_operator(damping, solver, preconditioner=LinearOperator\|LinearSolver)` (lo:236) | `LeastSquares(problem, damping=, solver=)` is the operator (point.py:63) | Retained; preconditioner now via `solver.with_preconditioner`. |
| `woodbury_*`, `surrogate_*` on each of four classes | `surrogate()` + `WoodburyPreconditioner` | Subsumed. |
| `data_reduced_inversion`, `parameterized_inversion` | point.py:241/252 | Retained on `LeastSquares`; `ConstrainedLeastSquares.parameterised` raises `NotImplementedError` (:698–707) and has **no `data_reduced`** — v1 had both (lo:683, 749) including the constraint pull-back `B @ P`. **Lost.** |
| `LinearMinimumNormInversion.minimum_norm_operator(...)` returning a `NonLinearOperator` with derivative + adjoint (lo:835–1031) | `DiscrepancyPrinciple` (point.py:399), an `Operator` with `_derivative` (:534) | Retained. `minimum_damping`, `maxiter`, `atol` → `iterations`, `rtol`. No-error branch (lo:1033) → `MinimumNorm(problem)` with `damping=0`. |
| — | `MinimumNorm.for_data` (point.py:300) | **Duplicate of `DiscrepancyPrinciple._resolve` with different behaviour** — see §5. |
| `ConstrainedLinearMinimumNormInversion.minimum_norm_operator`, `constraint_value_mapping` | `ConstrainedMinimumNorm`, `.constraint_value_mapping` (point.py:724, 797) | Retained. `with_formalism`/`with_solver`/`parameterised`/`data_reduced` absent on both `DiscrepancyPrinciple` and `ConstrainedMinimumNorm`. |

### 1.4 Backus (`backus_gilbert.py` → `backus.py`)

v1's `BackusInference` (bg:275) was thin: bound + significance level + `test_data_compatibility` (bg:366–399, a minimum-norm solve compared against the bound). v1's `DualMasterCostFunction` (bg:23) is the route-(d) oracle.

v2 has four classes: `BackusGilbert` (route b, new), `BackusInference` (route a, error-free only), `FeasibleProperty` (route c), `DualFeasibleProperty` (route d ≈ `DualMasterCostFunction` + bundle method). Substantial **extension**. But:

- DESIGN §18.8 says `BackusInference` takes a `method=` selecting among routes; not so — four classes, and the user must know which applies (route a refuses noise; c refuses non-ball; c refuses non-orthonormal data space, backus.py:515–520).
- `test_data_compatibility` → only partially: `BackusInference.budget < 0` raises (error-free), `DualFeasibleProperty.support` raises on an unbounded dual (:963–969). `FeasibleProperty` has no feasibility test — its `_bisect` raises "could not bracket" (:656–657). §18.11's promised `feasible(data).is_empty()` does not exist (no `is_empty` anywhere in `geometry/`).
- `DualMasterCostFunction.value_and_subgradient` (fused evaluation, bg:139) and the finite-difference fallback (bg:209) have no counterpart; `dual_cost` (backus.py:918) builds separate `value`/`gradient` closures that each recompute `A* λ`.

### 1.5 Workflow, formalism, nonlinear, joint

- **Common workflow** — build problem → prior → `LinearGaussianInversion(problem, prior, solver=CGSolver().with_preconditioner(...))` → `post = inv(data)` → `post.sample(rng=)` → `inv.push_forward(T)(data)` → `inv.log_evidence(data)` → compare two inversions' evidences: **supported** and exercised by examples 21/22/24/26/27. A surrogate preconditioner needs the factory form `solver=lambda normal: ...` (§28.1). Mixture comparison via `LinearGaussianMixtureInversion.log_evidence`.
- **Formalism** — equivalent to v1 in scope; validation is better (needs `Q^-1`/`R^-1` is checked with a message, normal.py:190–204). One inconsistency inherited from v1, see §5.2.
- **Nonlinear** — **neither version** has a MAP / Gauss–Newton / Laplace estimator. v1: `ForwardProblem` accepts a `NonLinearOperator` but no inversion class uses it. v2: same; `ForwardProblem.chi_squared` works with a nonlinear `Operator`, `numerics/optimisation.py` has `NewtonCG`, `TrustRegionNewton`, `LBFGS`, and `gauss_newton_hessian` (optimisation.py:702), and `NormalOperator(forward=F.derivative(m), prior, error)` in model-space formalism *is* the Laplace precision. DESIGN §18.13 defers it explicitly. All ingredients are present; the estimator class is missing.
- **Joint inversion** — `from_direct_sum` + `LinearGaussianInversion` works end to end, including a blocked `NormalDiagonalPreconditioner` on the `DirectSum` data space and `log_evidence` **[verified]**. Not covered by any v2 test (test_inference.py:127 only checks the forward operator) or example (example 22 is a model-space direct sum).

---

## 2. Algorithmic performance

**Solve counts (data-space formalism, iterative solver)** — identical to v1 in every case I traced:
- Posterior mean: 1 solve (`gain(shift)` at construction, gaussian.py:116–117, is a solve with zero RHS when both means are zero; CG returns at iteration 0 — no waste; a direct solver factorises at `solver(normal)` regardless, as in v1).
- Covariance application: 1 solve per application (`Q - K A Q`, normal.py:395); property covariance `T C T*` costs dim(P) solves to assemble. Model-space: the inverse is the covariance.
- Sample: 1 prior draw + 1 noise draw + 1 forward + 1 solve (gaussian.py:483–489). Same as v1 (lb:374–389).
- `mahalanobis`: 1 solve (gaussian.py:358); the log-determinant is data-independent but not cached.
- `InverseOperator._value` (solvers.py:231) re-runs the Krylov iteration on every application; there is no caching. `adjoint_inverse` builds and caches a second inverse of `A*` (:237–242) — so `linear.adjoint(model)` in `DiscrepancyPrinciple._derivative` (point.py:574) starts a *second* Krylov solve on `N*`, which for a self-adjoint `N` is the same system solved twice without reuse. §27.5 documents the "how many applications" trade-off; example 22 switches to Cholesky for diagnostics.

**Normal operators are lazy** (`forward @ prior.covariance @ forward.adjoint + error.covariance`, normal.py:181–188; tikhonov.py:90–109); nothing is assembled unless a direct solver or `log_determinant(method="dense")` asks.

**Discrepancy sweep** — `TikhonovFamily`/`DampedSolves` (root_find.py:256–326): sums reassembled per multiplier (free), solutions warm-started, deferred preconditioner rebuilt only when the multiplier moves by >10× (`refresh`). Genuine improvement over v1 (which rebuilt `preconditioner(normal_operator)` on every probe, lo:871–876). **But** `DampedSolves.operator` builds `base + t*shift` as a plain sum (root_find.py:296–299), not a `TikhonovNormalOperator`, so a *structure-aware* deferred preconditioner (`NormalDiagonalPreconditioner`) is refused with a `TypeError` inside the sweep while the same solver works on fixed-damping `LeastSquares` **[verified]**. DESIGN §24.3/§25.1's claim that "every structure-aware preconditioner applies to the point estimators" is false for `DiscrepancyPrinciple`, `MinimumNorm.for_data`, `ConstrainedMinimumNorm`.

**`DiscrepancyPrinciple`** — `__call__` and `derivative` each re-run the full search (`_resolve`, point.py:487–519); `_linearise` is not overridden, so `op.at(data)` does two searches. The derivative additionally costs 2 solves (`h`, `L* u`) plus 1 per application. v1 had the same double search.

**Evidence** — matrix-free in both formalisms; Sylvester used in model space (gaussian.py:415–426). `"auto"` is dense below dim 512 (see §1.2).

**Mixture** — `__call__` (mixture.py:174) calls `weights` → `log_evidence_terms` → `log_evidence` per component, recomputing the *data-independent* log-determinant every call; `_PushedMixture.__call__` (:215) does it again. Should be cached per component (one `normal_log_determinant` per inversion, memoised).

**Backus routes**
- Route (b) `BackusGilbert`: one inverse of `A A* + αI` on the data space; per direction of `error_bars` (:236–246) 2 adjoint applications (one of which, `operator.adjoint(direction)`, is a solve through `inverse*`). Fine. `_harden` (:180–186) assembles the **full dense covariance matrix** to read its mean diagonal — dim(D) applications and O(D²) memory for one scalar, and the wrong scalar on a weighted space (see §5).
- Route (a) `BackusInference`: `A A*` inverse + kernel projector once; `__call__` inverts `budget · T P T*` by nested CG (each application of the shape = 1 data-space solve) — fine for small P. `inclusion_norm` (:354–391) performs a **dense generalised eigendecomposition of `C C*` on `D⊕P` on every call** — not cached (unlike `FeasibleProperty._reduced`, which is a `cached_property`), so `admits` in a loop is O(calls × (D+P)³ + (D+P) operator applications).
- Route (c) `FeasibleProperty`: `_data_gram` (dense eigh of `A A*`, dim(D) applications) is a `cached_property` and **is** reused across directions and multiplier values, as BGP §2.6 prescribes. Per direction: 3 operator applications in `_prepare` + 60×60 closed-form O(D²) evaluations + 2 adjoints in `_model`. Good. The `solver=` argument is *not* used for the damped operator (docstring :470 is wrong); only for `onto_kernel` and the property pseudo-inverse. Refuses non-orthonormal data spaces (:515–520).
- Route (d) `DualFeasibleProperty`: one bundle minimisation per direction; `certificate` re-minimises (:979–981); `dual_cost` recomputes `A* λ` separately in `value` and `gradient` (v1 fused them).

**Constrained variants** — `ConstrainedLeastSquares` applies the projector twice (`projector @ inner.operator`, point.py:635; the inner solution already lies in range(P) for t>0) — one extra cheap application; harmless. `_reduced_problem` passes `problem.error` which may be a `Subset` → `AttributeError` deep inside `LeastSquares`.

**Direct solvers** (`DirectSolver._invert`, solvers.py:274–289) factorise once at `solver(operator)` and apply cheaply — same as v1.

---

## 3. Code practice and quality

**Hierarchy** — `Estimator{Point,Measure,Set}Estimator` (estimators.py:38–69) is coherent and small. `LinearPointEstimator(AffineOperator)` with `PointEstimator.register(...)` (:162) is a mild hack (virtual subclass; `isinstance` works, nothing enforces the abstract contract). `GaussianEstimator` (mean map, covariance, centred sampler) is the right abstraction and makes `push_forward` one line. `FactoredNormalOperator`/`NormalOperator`/`TikhonovNormalOperator` is justified by the preconditioners; not over-engineered. `TikhonovFamily` is a reasonable extra. `MinimumNorm(LeastSquares)` adds nothing but a name and a divergent duplicate of the discrepancy search (point.py:290–359 vs 487–519).

**Duplication**
- `MinimumNorm.for_data`/`discrepancy_search` vs `DiscrepancyPrinciple.search`/`_resolve` — same `_discrepancy_search` call, but `for_data` skips the "zero model fits" check, the `exhausted == "low"` refusal, and the saturation bookkeeping. Result: `for_data` silently returns damping 1e-31 with χ² ≈ 1e39 on data no model can fit, while `DiscrepancyPrinciple` raises `ValueError` on the same input **[verified]**; with a direct solver `for_data` leaks a raw `LinAlgError` **[verified]**. The `for_data` docstring (:314–325) describes behaviour that §27.3 rejected.
- `with_solver`/`with_formalism` written out separately on `LeastSquares`, `ConstrainedLeastSquares`, `LinearGaussianInversion`; absent on `DiscrepancyPrinciple`, `ConstrainedMinimumNorm`, all Backus classes.
- `choose_formalism` exists twice (normal.py:50, point.py:46 wrapper) and is exported.
- `weighted_adjoint`/`right_hand_side`/`model_from` on both `TikhonovNormalOperator` and `TikhonovFamily`; `TikhonovFamily.weighted_adjoint` (tikhonov.py:383–389) returns `self._weighted`, which is never set → `AttributeError` **[verified]**. Dead public method.

**Naming** — `LinearGaussianInversion` is accurate (§23.3). `for_data` is opaque (it returns an estimator, not a result). `with_*` is consistent. `DiscrepancyPrinciple` names the rule, not the estimator (`MinimumNormByDiscrepancy` would read as a mapping). `BackusInference` in v2 means "route (a), error-free ellipsoid", which is narrower than v1's class of the same name and than DESIGN §18.8's description. `FeasibleProperty` vs `DualFeasibleProperty` say how, not what.

**Reach-through** — `FeasibleProperty.__init__` calls `BackusGilbert._harden` (backus.py:480), a private static of a sibling. `InverseOperator` pokes `__dict__["_adjoint_cache"]` (solvers.py:246). `InvariantDistancePreconditioner._invert` uses `normal.prior` (preconditioners.py:438, 458), which `FactoredNormalOperator` does not declare — works only for `NormalOperator`, fails on `TikhonovNormalOperator` with an `AttributeError` after `_require_normal` accepted it.

**Error handling** — `error_measure`/`error_set` raise `AttributeError` (problem.py:83, 90, 99); that propagates as an unhelpful `AttributeError` from `data_measure_from_model`, `data_reduced`, `_reduced_problem`, `LeastSquares` when the error is a set. `from_direct_sum` silently drops noise (see §1.1). `LinearForwardProblem.__init__` and `LinearGaussianInversion.__init__` type checks are good.

**Type hints** — vectors are `Any` throughout (package convention). Beyond that: `solver: Any` (mixture.py:60, point.py:177, gaussian.py:215), `operator: Any` (mixture.py:181, 194), `method: Any` (backus.py:878), `joint_prior -> Any` (gaussian.py:168). `Formalism` includes `"auto"` but the `.formalism` properties return `str`.

**Exports** — `pygeoinf2/__init__.py` exports neither `inference` nor `numerics`. The minimal workflow needs `from pygeoinf2.inference import ...`, `from pygeoinf2.numerics.solvers import CGSolver`, `from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner`, `from pygeoinf2.inference import NormalDiagonalPreconditioner` — preconditioners split across two packages by an implementation criterion (which factors they read) the user does not care about. v1 exported everything flat. §25.4 flags this as undecided; it is a usability problem for the primary use of the library.

**Design-document drift** — DESIGN §18.7 (sampler built in `__call__`), §18.8 (`method=` on `BackusInference`), §18.11 (`is_empty`), §24.3/§25.1 (structure-aware preconditioners on all point estimators), §22 (route c "reuses solvers that already exist"); catalogue rows for `LinearBayesianInversion` (`Bayesian`), `parameterized_inversion`/`data_reduced_inversion` ("Dropped"), `normal_residual_callback` ("Ported").

---

## 4. Documentation gaps (file:line)

Docstrings are generally good on purpose and rationale but thin on arguments, returns, cost, and exceptions. Concrete gaps:

- `problem.py:117` `synthetic_data`, `:124` `chi_squared`, `:148` `critical_chi_squared`, `:222` `data_measure_from_model`, `:226`, `:235`, `:277` `parameterised` — no Args/Returns/Raises; none states that a set-valued `error` raises.
- `problem.py:193` `from_direct_sum` — does not say the errors must all be Gaussian measures nor that a missing one drops noise for the joint problem.
- `estimators.py:135` `propagated_covariance` is a method while `resolution` (:126) is a property — no reason given; `:152` `as_measure` doesn't say the result cannot be sampled.
- `gaussian.py:53` class docstring one line; no summary of what is computed at construction (a solve is issued), no Raises.
- `gaussian.py:442` `log_evidence` — no statement of the reference measure for the density (it is the space's intrinsic volume: `log det N_c`, fc:466–476), no Raises (needs an error measure), no note that `method="auto"` assembles the operator below dim 512.
- `gaussian.py:299` `parameterised` — `**kwargs` undocumented and unusable.
- `point.py:300` `for_data` — docstring contradicts implementation (saturated-low case).
- `point.py:337` `discrepancy_search`, `:476` `search`, `:797` `constraint_value_mapping` — no Args; `:534` `_derivative` cost (search + 2 solves) undocumented on the class.
- `point.py:593` `ConstrainedLeastSquares` — does not say `damping` penalises `|u - t|²` within the subspace rather than `|u|²`.
- `tikhonov.py:383` `weighted_adjoint` — documented, broken. `:399` `solve`, `:408` `model` — no Args.
- `normal.py:374` `gain`, `:385` `posterior_covariance` — no Args; the `gain` parameter of the latter is redundant with `inverse` and unexplained.
- `backus.py:112` `BackusGilbert.__init__` — `level` hardening rule (`sqrt(χ²_crit · mean variance)`) not stated; `:199` `uncertainty`, `:224` `error_bars` — no Args; `:354` `inclusion_norm` — no cost note (dense eigendecomposition per call); `:429` `FeasibleProperty` class doc omits the orthonormal-data-space requirement stated only at `:516`; `:470` `solver:` "how to invert the damped normal operator" is false; `:871` `DualFeasibleProperty` — `method` type/contract not stated.
- `mixture.py:54` — no note that `__call__` recomputes every component's log-determinant.
- `preconditioners.py:347` `InvariantDistancePreconditioner` — needs `normal.prior`, i.e. a `NormalOperator` not any `FactoredNormalOperator`; undocumented.
- `inference/__init__.py:12` module example uses `MinimumNorm(problem)`; `post = ...` line says `LinearGaussianInversion` but the `D -> Measure(M)` comment column is misaligned — minor.

---

## 5. Correctness concerns

### 5.1 Metric handling (checked on weighted spaces)
- Normal operators, gain, posterior covariance, RHS: written in Hilbert adjoints (normal.py:181–207, 374–397; tikhonov.py:87–105). Tests run on a weighted data space (test_normal.py:58) and weighted model space (test_inference.py:39, test_point.py:200). **Correct.**
- `chi_squared` → `mahalanobis_squared` (gaussian.py:864–880) uses the space inner product with the Hilbert precision; `critical_chi_squared` uses `dim` dof. **Correct.** But `mahalanobis_squared` raises `NotImplementedError` without a precision — so `chi_squared`, hence `DiscrepancyPrinciple`, need `R^-1` even in the data-space formalism. v1 same.
- Evidence log-det: `log det N_gal − log det G = log det N_c`, which with the Mahalanobis term gives the density with respect to the metric's volume measure — coordinate-invariant, and constant across models on a fixed data space. **Correct**, tested on a weighted space (test_inference.py:467). Should be stated in the docstring.
- `NormalDiagonalPreconditioner`/`LocalisedPreconditioner`: Galerkin diagonal/blocks, `G y` on the way in (preconditioners.py:139–156, 249–307). **Correct.**
- `_self_adjoint_spectrum` (backus.py:41–59): generalised eigenproblem with `G`. **Correct.**
- `BackusGilbert._harden` (backus.py:180–186): `covariance.matrix(form="components").diagonal().mean()` — on a weighted data space the component-matrix diagonal is not the variance in the space's norm (the covariance of components is `C_c G^-1`); and `sqrt(χ²_crit(dim D) · mean var)` is not the radius of any credible ball. `GaussianMeasure.ambient_ball(level=)` (gaussian.py:755) is the correct hardening and exists. **Wrong on weighted spaces, crude on all.** Used by `BackusGilbert` and `FeasibleProperty` (:480).
- `FeasibleProperty._data_gram` (:523): `form="components"` guarded by `is_orthonormal` — refuses rather than errs. Acceptable.
- `BackusInference.__call__` (:336–352): ellipsoid `((T P T*)^-1 (p−p̃), p−p̃) ≤ budget` in the property space's inner product — consistent with `Ellipsoid`'s definition. OK.

### 5.2 Formalism inconsistency with no error measure (inherited from v1)
With `error=None`, the data-space normal operator is `A Q A*` (R = 0, exact-data conditioning) but the model-space one is `Q^-1 + A* A` (R = I) (normal.py:181 vs 205–207; v1 lb:178 vs 193). The two formalisms return **different posteriors**: mean difference 0.86 against a mean of norm 2.6 on a random 12×8 problem **[verified]**. Neither version documents this; v2's docstring (normal.py:158–159) states the model-space form as if it were the same problem. The Tikhonov pair is consistent (both R = I).

### 5.3 `from_direct_sum` silently drops noise (§1.1) **[verified]**.

### 5.4 `MinimumNorm.for_data` saturated-low garbage (§3) **[verified]**; and its `exhausted == "high"` case returns the largest *bracketed* damping (`found.argument`) rather than the zero model that `DiscrepancyPrinciple._resolve` returns (point.py:495–498) — a small but nonzero model where the principle says zero.

### 5.5 Constrained reductions
`ConstrainedLeastSquares` (point.py:607–651) and `ConstrainedMinimumNorm` (:737–795) reproduce v1's substitution `u = t + P w` correctly; `constraint_value_mapping` derivative `(I − P D A) B⁺` matches v1 (lo:1236) with the projector made explicit. Undamped `ConstrainedLeastSquares` in data-space formalism assembles `(AP)(AP)*`, singular whenever `dim(subspace) < dim(D)` (§27.1) — CG refuses, correctly; no guard or message in the class itself.

### 5.6 Minor
- `DiscrepancyPrinciple._derivative` shifts the residual by the error mean (point.py:568–569); v1 did not (lo:994–996). v2 is right.
- `LinearForwardProblem.joint_measure`/`from_direct_sum` type the argument as `ProbabilityMeasure` but call `Gaussian.from_product` (problem.py:215, 263).

---

## Recommendations

### Must
1. **Fix `MinimumNorm.for_data`** (point.py:300–335): after `discrepancy_search`, handle `found.exhausted == "low"` by raising the same `ValueError` as `DiscrepancyPrinciple._resolve` (point.py:500–514), and return damping ∞ / the zero model when the zero model already fits, exactly as `_resolve` does. Simplest: implement `for_data` by delegating to `DiscrepancyPrinciple(self._problem, level=, solver=, formalism=).search(data)` and reuse `_resolve`. Add a test mirroring `test_the_discrepancy_principle_has_no_answer` for `for_data`. Correct the docstring.
2. **Make `DampedSolves` build `TikhonovNormalOperator`s** so structure-aware preconditioners work in sweeps (root_find.py:295–299 builds `base + t*shift`). Either give `DampedSolves` an `assemble: Callable[[float], LinearOperator]` used instead of the sum, and pass `family.at` from `TikhonovFamily.__init__` (tikhonov.py:347–352); or have `_solver_for` resolve the deferred preconditioner against `family.at(t)`. Add a test: `DiscrepancyPrinciple(problem, solver=CGSolver().with_preconditioner(NormalDiagonalPreconditioner()))(data)` must run. Fix DESIGN §24.3/§25.1 or the code, not both.
3. **`from_direct_sum` must not silently drop noise** (problem.py:212–219): raise `ValueError("every problem needs an error measure, or none may")`, or build a zero-covariance Gaussian for the exact members. Same in v1 if it is still maintained.
4. **Delete or fix `TikhonovFamily.weighted_adjoint`** (tikhonov.py:383–389): return `self._template.weighted_adjoint()`.
5. **`BackusGilbert._harden`** (backus.py:168–187): replace the dense-matrix mean-variance rule with `problem.error_measure.ambient_ball(level=level)` and document that this is the hardening used; make it a module-level function so `FeasibleProperty` stops calling a sibling's private static (:480).
6. **Document or forbid the no-error model-space Gaussian formalism** (normal.py:189–207): either raise `ValueError("the model-space formalism needs an error measure; without one the data are exact and only the data-space form applies")` or state in the docstring that `error=None` means `R = I` there and `R = 0` in the data space. Apply the same to v1 if kept.
7. **Remove the dead `**kwargs`** from `LinearGaussianInversion.parameterised`/`data_reduced` (gaussian.py:299–329) and `LeastSquares.parameterised`/`data_reduced` (point.py:241–260), or add `error=` passthrough explicitly where `problem.data_reduced` takes it.

### Should
8. **Export the inference layer at top level** (`pygeoinf2/__init__.py`): `LinearForwardProblem`, `LinearGaussianInversion`, `LeastSquares`, `DiscrepancyPrinciple`, the Backus classes, plus `CGSolver`, `CholeskySolver`, `JacobiPreconditioner`, `WoodburyPreconditioner`, `NormalDiagonalPreconditioner`, `LocalisedPreconditioner`. Also consider moving `NormalDiagonalPreconditioner`/`LocalisedPreconditioner`/`InvariantDistancePreconditioner` next to `WoodburyPreconditioner` (they are all "read factors off a `FactoredNormalOperator`"; `numerics` already duck-types this in `from_normal`) so preconditioners live in one place.
9. **Cache the log-determinant** on `LinearGaussianInversion` (`functools.cached_property` keyed on `method`/`samples`, or a private memo) so `LinearGaussianMixtureInversion.__call__`/`weights` (mixture.py:147–178, 215–219) do not recompute it per call; pass an explicit `rng` so cached stochastic estimates are reproducible.
10. **Cache `_self_adjoint_spectrum(joint @ joint.adjoint)` in `BackusInference.inclusion_norm`** (backus.py:383) as a `cached_property`, as `FeasibleProperty._reduced` (:715) already does.
11. **Override `_linearise` in `DiscrepancyPrinciple`** (and `ConstrainedMinimumNorm`) so `op.at(data)` runs one search and reuses `(model, damping, moves)` for both value and derivative; and reuse `fixed.inverse_normal_operator` for `linear.adjoint(model)` when `N` is self-adjoint instead of building a second inverse (point.py:562–574, solvers.py:237–242).
12. **Restore `data_reduced` and `parameterised` on `ConstrainedLeastSquares`/`ConstrainedMinimumNorm`** by porting v1's constraint pull-back (lo:717–747: `AffineSubspace.from_linear_equation(B @ P, w)` when `has_explicit_equation`), raising only for geometric subspaces.
13. **Add `with_solver`/`with_formalism` to `DiscrepancyPrinciple` and `ConstrainedMinimumNorm`**, and a `residual_callback` (or a documented pointer to `CGSolver(callback=)`) on `LinearGaussianInversion`; fix the catalogue row for `normal_residual_callback`.
14. **Add an end-to-end joint-inversion test**: `from_direct_sum` of two problems with different noise, `LinearGaussianInversion` in both formalisms, blocked `NormalDiagonalPreconditioner` on the direct-sum data space, evidence, and a push-forward; compare against a dense reference.
15. **Typed errors for set-valued `error`**: make `ForwardProblem.error_measure` raise `TypeError` with a message naming the caller's need ("this method needs a Gaussian error measure; the problem's uncertainty is a set — see inference.backus"), and check `problem.has_error and isinstance(problem.error, ProbabilityMeasure)` in `_reduced_problem` (point.py:710–721), `data_reduced` (problem.py:304–305).
16. **`FeasibleProperty` docstring/signature** (backus.py:450–485): document the orthonormal data-space requirement on the class; rename or re-document `solver` (it is used for the kernel projector and property pseudo-inverse only). Consider a `feasible(data)` / `is_feasible(data)` predicate on the three noisy routes to replace v1's `test_data_compatibility` and DESIGN §18.11's `is_empty`.
17. **Update DESIGN.md and V1_CATALOGUE.md** for the drift listed in §3 (stale `Bayesian` name; `parameterized_inversion`/`data_reduced_inversion` "Dropped"; §18.7 sampler; §18.8 `method=`; §18.11 `is_empty`; §24.3/§25.1 preconditioner claim; §22's "reuses solvers" for route c).
18. **Docstring pass** per §4: Args/Returns/Raises on every public method in `problem.py`, `gaussian.py` (state the evidence reference measure and the dense-below-512 default), `point.py`, `backus.py`, `normal.py:374–397`; document `damping` semantics in `ConstrainedLeastSquares`; document `InvariantDistancePreconditioner`'s `NormalOperator`-only requirement or add `prior` to `FactoredNormalOperator`.

### Consider
19. **A nonlinear MAP/Laplace estimator** (`inference/laplace.py`): `MaximumAPosteriori(problem, prior, optimiser=NewtonCG(...))` minimising `chi²(m,d) + mahalanobis_prior(m)` using `Operator.at`/`gauss_newton_hessian` (optimisation.py:702), returning a `GaussianEstimator`-like object whose covariance is the inverse of `NormalOperator(F.derivative(m_map), prior, error=..., formalism="model_space")`. Every ingredient exists; DESIGN §18.13 sketches the signatures. This is the one substantive capability absent from both versions and directly asked about.
20. **Collapse `MinimumNorm` into `LeastSquares`** (it adds only `for_data`, which should delegate to `DiscrepancyPrinciple` per item 1), or give it a real distinction (e.g. default `damping=0` with the no-error pseudo-inverse semantics of v1 lo:1033–1037).
21. **Fuse value and subgradient in `DualFeasibleProperty.dual_cost`** (backus.py:936–948) as v1's `value_and_subgradient` (bg:139–196) did, if the bundle method can consume a fused oracle; and let `certificate` return from the same minimisation as `support`.
22. **`propagated_covariance` → property** for symmetry with `resolution` (estimators.py:126–141), or explain why not.
23. **Rename** `for_data` (→ `at_discrepancy(data)` or similar) and consider `MinimumNormByDiscrepancy` for `DiscrepancyPrinciple`; unify `choose_formalism` to one definition.
24. **`from_direct_sum` / `joint_measure` typing**: annotate `GaussianMeasure` where `Gaussian.from_product` is required (problem.py:194–197, 235, 263), or generalise to `product(...)` from `probability.base`.
