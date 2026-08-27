# Review: `pygeoinf2/numerics` (functional calculus, randomised, optimisation, line search, convex, root find, quadratic forms)

> **Note (2026-08-27):** the decisions recorded in `pygeoinf2/REVIEW.md` §11 (D-1 … D-13) override the Must/Should/Consider ranking below wherever they conflict — in particular D-1 (sphere vectors are `SHGrid`, `sampling=1` default), D-2 (points in `(lat, lon)` degrees), D-3 (per-geometry submodules with `Lebesgue`/`Sobolev` subclasses), D-4 (`from_matrix(..., form=)`), D-6 (parallel hooks around operators), D-12 (path *integral* operator), D-13 (convex solvers restored).

All line numbers are as of the current checkout. Claims marked **[measured]** were verified with throwaway scripts; everything else is from reading both codebases.

## 1. Functionality retained / extended / lost

### 1.1 Functional calculus (`pygeoinf/functional_calculus.py` → `pygeoinf2/numerics/functional_calculus.py`)

| Feature | v1 | v2 | Verdict |
|---|---|---|---|
| Lanczos with full reorthogonalisation | `iter_lanczos_tridiagonalize` 392–509, Kahan "twice is enough" 466–480 | `iter_lanczos_tridiagonalise` 73–134, **single pass** 123–125 | Weakened |
| Partial reorth / restarts / block Lanczos | none | none | Parity (neither) |
| `reorth="none"` (three-term only) | 418–419, 466 | `reorthogonalise=False` 79 | Retained |
| Breakdown detection | relative to running `max_scale` 449–502 | relative to `max(|alpha|,1)` 130; `breakdown_tol` exposed on the iterator only, not on `lanczos_tridiagonalise` 151–165 | Retained, inconsistent surface |
| Fixed Krylov size (`method="fixed"`) | 57, 190–191 | gone; only `max_iterations` + `rtol` | **Lost** |
| `max_k=None → dim`, `atol`, `check_interval` | 58–62, 192–193, 205–209 | `max_iterations=50` default, no `atol`, check every step 203–211 | Changed (see §2) |
| Convergence check location | coefficient vector `g ∈ R^k` 221–231 (O(k) scalars) | full-space vector 206–211 (O(k) axpys per step) | **Regressed** (§2.1) |
| Quadratic form `<v, f(A) v>` | `operator_function_quadratic_form` 255–351 with rtol/atol stopping 331–335 | `operator_quadratic_form` 239–269, **no convergence test at all**, always runs `max_iterations=30` | **Regressed** |
| `f(A)` as operator | `LanczosOperatorFunction` 40–138 | `OperatorFunction` 277–328 | Retained |
| Named helpers, trait gating | — | `operator_sqrt/inverse_sqrt/exp/log/power` 354–395, `_require` 398–403 | New, good |
| Diagonal fast path | (v1: `InvariantLinearAutomorphism` separately) | `operator_function` 331–351 via `isinstance(…, DiagonalLinearOperator)` | New, but see §2.3 |
| log-det via SLQ | in `linear_bayesian.py` (out of scope) | `log_determinant` 406–482 | New — but implemented as `random_trace(operator_log(A))` 478–482, i.e. `(z, log(A) z)` via `apply_operator_function`, not via `operator_quadratic_form` whose own docstring (248–253) says it is "the kernel of stochastic Lanczos quadrature" |

Neither version stores anything but the full basis list; Lanczos memory is O(k·n) in both (v1 `Q` 445; v2 `basis` 110). There is no two-pass or restart mode in either.

### 1.2 Randomised (`pygeoinf/low_rank.py` → `pygeoinf2/numerics/randomised.py`)

| Feature | v1 | v2 | Verdict |
|---|---|---|---|
| `random_range` power iterations | `power=2` default 502, re-orthonormalised each half-step 660–664 | `power=1` default 109, `_power_iterate` 82–100 | Retained |
| Oversampling | none (sampled exactly `size_estimate`) | `oversampling=10` 108 | New |
| Fixed vs adaptive rank | `method="fixed"/"variable"` 499, `block_size`, `rtol`, `max_rank` | `rank=None` → adaptive 149–164; `rank=k` → `k+oversampling` probes 142–147 | Retained; note `rank=k` returns up to `k+10` vectors, docstring 131–133 only warns it may be *shorter* |
| Component QR fast path (Path B) | 590–630 using LAPACK `qr` | **gone** — Python Gram–Schmidt on every space | **Lost** (§2.2) |
| "Geometric safety guard" | 547–548 routes non-orthonormal codomains to `white_noise_measure` | not needed: probes are `domain.white_noise` 74–79 and orthonormalisation is always in the space's inner product | Subsumed, correctly |
| `parallel`/`n_jobs` | everywhere | **gone** | **Lost** |
| Return type | `LinearOperator` `Q: R^k → codomain` 588, 628 | `list[Vector]` 131–134 | Changed; callers build `from_vectors` |
| `LowRankSVD` | `.u_factor/.sigma_factor/.v_factor/.singular_values/.rank` 87–110 | `.left_factor/.singular_values/.right_factor/.rank` 258–276; **no traits claimed** 253 | Renamed |
| `LowRankEig` | `.u_factor/.d_factor/.eigenvalues/.trace`, `apply_function(f, regularization=)` 230–362 | `.factor/.eigenvalues/.rank/.trace`, `apply_function(f)` 195–231; claims SA/PSD 188–191 | Renamed, `regularization` dropped |
| `LowRankCholesky` | Cholesky of `Q*AQ` with eigen fallback 465–475 | `random_cholesky` = `random_eig` + `U D^{1/2}` 433–463 | Simplified; fine |
| `random_trace` | Rademacher components + Euclidean dot 1121–1127, adaptive `rtol` stopping 1140–1154, parallel | white-noise probes 490–494, **fixed** `samples=100`, returns `Estimate` with standard error 495–499 | Adaptive stopping **lost**; error bar **gained** |
| `random_diagonal` | adaptive with `rtol`, Gaussian or Rademacher 860–958 | Rademacher only, fixed `samples`, explicit `form="galerkin"/"components"` 566–603 | Adaptivity lost; "which diagonal" now explicit (good) |
| `deflated_diagonal` | SVD deflation 961–1050 | eigen deflation 502–563; exact part correct for general `G` (546–560) | Retained, **but not in `__all__`** (35–46) nor `numerics/__init__.py`; the catalogue row (V1_CATALOGUE.md:171) says both "Ported" and "Not ported" |
| `white_noise_measure` | 34–47 | dropped in favour of `HilbertSpace.white_noise` | Correct decision |

### 1.3 Optimisation (`pygeoinf/nonlinear_optimisation.py` → `optimisation.py`, `line_search.py`)

v1 offered only a SciPy wrapper (`ScipyUnconstrainedOptimiser` 17–112: Newton-CG, trust-*, BFGS, L-BFGS-B, CG, Nelder-Mead, Powell) and a SciPy `line_search` bridge (115–218). v2 replaces it with native `SteepestDescent`, `NonlinearCG` (PR+/FR), `LBFGS`, `NewtonCG`, `TrustRegionNewton` (Steihaug), `truncated_cg`, `gauss_newton_hessian`, and `ArmijoLineSearch`/`StrongWolfeLineSearch`. The DESIGN §11.7 argument for dropping the wrapper (gradient vs derivative components) is correct.

Lost relative to v1's wrapper: derivative-free methods (Powell), bound constraints (L-BFGS-B), `ftol`-type stopping, and SciPy's `callback`. Not present in either: callbacks, iterate history (v1 convex had `store_iterates`), constrained optimisation by projection. Note that `ProximalGradient` + `BallIndicator` (convex.py 165–215, 513–706) *is* projected gradient and is the natural home for "constrained by projection"; it is not advertised as such.

Stopping criteria: `Optimiser` uses `max(gtol, rtol·||g0||)` (optimisation.py 122–123) only; no value-decrease or step-length criterion for the gradient methods. See §2.5 and §5 for why the defaults are unreachable.

### 1.4 Convex (`convex_analysis.py`, `convex_optimisation.py` → `convex.py`, plus `geometry/convex.py`)

Retained/extended: `SupportFunction` algebra (`+`, positive `*`, `compose_with`, `of_ball`, `of_point`; convex.py 218–385, domain-direction bug fixed per DESIGN §11.8), `SubgradientDescent` with diminishing/Polyak rules (402–510; v1 had only a constant step), `ProximalGradient`/FISTA with monotone backtracking (513–706), `ProximalPoint` (709–767), `ProximalBundleMethod` with linearisation errors (787–928), closed-form prox for `SquaredDistance`/`NormFunctional`/`BallIndicator`.

**Lost** (all in v1 `convex_optimisation.py`): `LevelBundleMethod` with its HiGHS LP *global* lower bound (1014–1387, 1104–1161); `QPSolver` protocol and `SciPyQPSolver`/`OSQPQPSolver`/`ClarabelQPSolver`/`best_available_qp_solver` (333–695); `PrimalKKTSolver` (2230–2601); `ChambollePockSolver`/`solve_primal_feasibility` (1542–1860); `SmoothedDualMaster`/`SmoothedLBFGSSolver` (1868–2223); `solve_support_values` with warm-start across directions and joblib (1393–1483); `value_and_support_point`/`_get_value_and_subgradient` fused evaluation (convex_analysis.py 80–105, convex_optimisation.py 736–758); `Bundle.compress`/`Cut` as objects; `CallableSupportFunction` (catalogue says it became `ConvexSet.from_support_function` — true, geometry/convex.py 104–129).

The catalogue rows are inaccurate here: `PrimalKKTSolver`, `solve_support_values`, `solve_primal_feasibility` are marked **Ported** (V1_CATALOGUE.md 216–220) but no KKT, Chambolle–Pock or support-values code exists anywhere in v2 (grep). Routes (c) and (d) in `inference/backus.py` use `monotone_root` (630–660) and `ProximalBundleMethod` (890–905) instead — a legitimate design choice, but "Subsumed" not "Ported". `QPSolver` and backends are marked "Planned" while DESIGN §22.7 records the decision to use projected gradient instead; the two documents disagree.

### 1.5 Weighted chi² (`quadratic_form_quantile.py` → `quadratic_forms.py`)

| v1 | v2 |
|---|---|
| `imhof`, `ws`, `saddlepoint`, `mc`, `auto` (ν_eff-based selection 428–462) | `imhof`, `matched`, `monte_carlo`, `auto` (= Imhof with fallback 102–108); **saddlepoint gone** despite catalogue 308 ("Imhof, Wood–Saddlepoint, and Monte Carlo") |
| Vectorised `t` 315–337 | scalar `value` only 64–73 |
| Imhof via vectorised fixed-step trapezoid, explicitly chosen over `quad` (489–494) | Imhof via `scipy.integrate.quad` with a scalar Python integrand 38–52 |
| `rtol`, `n_samples`, `rng` on both cdf and quantile | quantile takes no `samples`/`rng` (112–119), so `method="monte_carlo"` runs `brentq` on a noisy function |
| Empty/all-zero weights → degenerate point mass 318–321, 381–382 | `ValueError` 26–35 |

**[measured]** On `w_j = 1/j²`: v1 and v2 quantiles agree to 1e-6 for n = 20, 300, 2000, so v2's `quad` route is *not* the failure v1's comment predicts on this spectrum. Timing: n=20 v1 12 ms vs v2 314 ms; n=300 118 ms vs 315 ms; n=2000 845 ms vs 699 ms.

### 1.6 Root finding (`root_find.py`) — new

`monotone_root` (105–252) and `DampedSolves` (255–326) implement DESIGN §18.6/§24.2 as described: two-sided geometric bracketing, saturation reported via `exhausted`, warm start by carrying `previous`, preconditioner reuse with `refresh`. Not exported from `numerics/__init__.py`.

## 2. Algorithmic performance

### 2.1 Lanczos / `apply_operator_function`
- Per application v2 does: `k` operator applications, ~`k²/2` inner products (reorth), **plus ~`k²/2` axpys** because `_combine` (215–228) rebuilds the full-space result at every step to test convergence (206–211), plus `k` tridiagonal eigensolves. v1 checked in coefficient space (221–231: `||g_k − pad(g_{k−1})||`, exactly equivalent since `Q` is orthonormal) every `check_interval` steps and recombined once (248–250). With the default `rtol=1e-10` the test rarely fires on a large problem, so the O(k²) axpy cost is paid in full on nearly every call.
- `operator_quadratic_form` never stops early (262–269).
- `OperatorFunction` recomputes the tridiagonalisation per application in both versions — inherent to `f(A)x`, correctly documented at 278–283.
- Single-pass reorthogonalisation (123–125) is less robust than v1's Kahan test for the same cost class.

### 2.2 `random_range` orthonormalisation
- Adaptive mode calls `codomain.orthonormal_basis(basis + block)` (161) which re-orthogonalises the **entire** accumulated basis every block (spaces.py 178–192 loops over all vectors), after already having orthogonalised the block against the basis to measure the residual (158–160). Cost O(k³/b) inner products. v1 orthogonalised only the new residuals against the existing basis (669–737: O(k·b) per block).
- **[measured]** Full-rank 400×400 SPD operator, `rtol=1e-2, power=0`: basis 380, **1,170,897 inner products**, 380 applications. v1's incremental scheme would be ~7×10⁴; the v1 Path-B QR would be one LAPACK call.
- No component/QR fast path on `CoordinateSpace`, so a `EuclideanSpace` pays Python-loop Gram–Schmidt.
- `random_cholesky` (457–462) applies `factor` to `k` basis vectors (O(k²) axpys) to recover eigenvectors that `random_eig` had as a list.
- `random_eig`/`random_svd` truncate to `rank` only *after* doing `rank+oversampling` operator applications and a `(k+p)²` inner-product Gram matrix — standard, fine.

### 2.3 Diagonal fast path dispatch **[measured]**
`operator_function` dispatches for `D`, `2*D`, `D+D`, `D@D` (the `_combine_*` protocol in diagonal.py 126–148 works). It does **not** dispatch for `D.with_traits(...)` (returns `_RetraitedOperator`, operators.py 386–390, 920–932) → falls to `OperatorFunction`. Both in-package call sites that take a log-determinant of a covariance wrap it in `.with_traits(definite)` first: `inference/gaussian.py:417–420` and `probability/gaussian.py:531, 539–545`. So a diagonal prior covariance (the common case, and the case DESIGN §26.2 calls "the cheap one") goes through 100 Hutchinson probes × ≤40 Lanczos steps instead of `np.sum(np.log(eigenvalues))`. Separately, `log_determinant` (406–482) never consults `DiagonalLinearOperator.log_determinant` (diagonal.py 202–205) even for a bare `D`: measured `Estimate(128.678 ± 1.3)` for an exact value of 129.4845 on a 200-dim diagonal.

### 2.4 L-BFGS two-loop
Correct and standard (365–388): `2m` inner products + `2m` axpys per direction, initial scaling `(s,y)/(y,y)`, curvature guard in `_update` (390–397). Nothing to fix.

### 2.5 Line search evaluation counts **[measured]**
60-dim SPD quadratic, cond ≈ 4.9:

| method | iters | derivative calls | value calls | converged |
|---|---|---|---|---|
| LBFGS / StrongWolfe | 19 | 75 | 79 | False |
| LBFGS / Armijo | 500 | 501 | 17,322 | False |
| NonlinearCG | 18 | 141 | 160 | False |
| SteepestDescent | 2000 | 2001 | 5,330 | False |
| NewtonCG | 8 | 9 | 20 | True |

Causes: (a) `_DescentMethod._minimise` calls `functional.at(new_x)` (optimisation.py 192) after the Wolfe search already evaluated value *and* gradient at that point; `LineSearchResult` (line_search.py 31–39) carries no gradient, so one full evaluation per iteration is wasted; (b) `_zoom` re-evaluates `low_value` (243), which the caller already knows, and does not count it; (c) `_zoom` is pure bisection (246) — no interpolation, so ~4 gradient evaluations/iteration where SciPy's `dcsrch` averages 1–2; (d) the non-convergence is the termination defect in §5.4, not the line search per se — the Armijo case burned 480 iterations × ~35 backtracks accepting rounding-noise steps.

### 2.6 Bundle QP subproblem
`_solve_model` (882–928) rebuilds the full `k×k` Gram matrix of subgradients every iteration (900–904: O(k²) inner products on full-space vectors, k ≤ 40 → up to 1600 per iteration) when only one row changes per iteration. `_minimise_on_simplex` (945–963) is projected gradient with step `1/λ_max`, capped at 400 iterations, no accuracy check. As cuts become nearly parallel near convergence the Gram matrix is ill-conditioned and 400 projected-gradient steps may not converge; the dual objective at an inexact feasible `w` **over-estimates** the gap (it is a lower bound on the dual optimum, negated), so termination is conservative rather than wrong, but the candidate is not the model minimiser and the descent test (874) sees an inflated gap → extra null steps. DESIGN §22.7's "80× speed-up" compares v1's *primal* `(d+1)`-variable SLSQP against a *dual* `k`-variable problem; the win is dualisation, and an OSQP/Clarabel solve of the k-variable dual would be equally fast and accurate.

### 2.7 `monotone_root` warm start
Works as designed. Minor: `previous` (155–162) is shared between the upward and downward walks, so the first downward probe (234–238) warm-starts from the solution at the *largest* multiplier reached rather than from `initial`. `DampedSolves.operator` caches one assembled operator per distinct multiplier with no bound (297–302).

## 3. Code practice / quality

- **API consistency.** `Optimiser.minimise(functional, x0)` → `OptimisationResult`; `ProximalGradient.minimise(smooth, x0, *, nonsmooth)` (550–573); `ProximalBundleMethod.minimise(functional, start, *, subgradient)` → `BundleResult` (770–784) with different field names (`minimum` vs `value`, no `evaluations`, no `history`, no `gradient_norm`), and it is not an `Optimiser`. Its `tolerance` is relative to `max(|f|,1)` (870) while `gtol` elsewhere is absolute. Cap is `iterations` in bundle/`monotone_root` but `max_iterations` in `Optimiser`.
- **Callbacks**: none in any method. History is a list of values only.
- **Exports.** `numerics/__init__.py` omits `ProximalBundleMethod`, `BundleResult`, `deflated_diagonal`, `monotone_root`, `DampedSolves`, `Evaluation`, `RootResult`, `weighted_chi2_cdf/quantile`; it exports `truncated_cg` (54) which is absent from `optimisation.__all__` (46–55). `randomised.__all__` omits `deflated_diagonal` though `symmetric_space/base.py:633` imports it.
- **Dead code.** `_ConvexResult` (convex.py 393–399) unused; `random_eig` 364–367 (`space.mean([]) for _ in range(0)`); `truncated_cg` computes `squared_norm(residual)` twice (667, 673).
- **Reach-through.** `random_range` calls `codomain._orthogonalise_against` (randomised.py 159), a private method of `HilbertSpace`.
- **Duplication.** The result-building tail is repeated in `_DescentMethod._minimise`, `NewtonCG._minimise`, `TrustRegionNewton._minimise`, `SubgradientDescent._minimise`, `ProximalGradient._run`, `ProximalPoint._minimise`. `SubgradientDescent` evaluates `functional(x)` twice per iteration (491, 495). `ProximalGradient` evaluates `total(x)` for history (630) and `smooth(trial)` per backtrack (703) without counting them.
- **Error handling.** Non-convergence returns flags (good) but: `TrustRegionNewton` reports `"iteration limit reached"` with `iterations=max_iterations` after a radius-collapse `break` (587–589, 608–620); Armijo reports `evaluations=max_backtracks` even when it stopped early on `min_step` (128); `apply_operator_function`/`OperatorFunction` give **no** convergence signal when `max_iterations` is hit (212); `weighted_chi2_cdf` swallows every exception in auto mode (105); `weighted_chi2_quantile` surfaces a raw `brentq` `ValueError` on bracket failure (150–161); `monotone_root` treats any `ConvergenceError` during widening as "edge of the usable range" (42, 204–207) — an iterative solver that merely ran out of iterations is misreported as saturation.
- **Type hints.** Vectors are `Any` throughout although the algebra is generic in `V`; `space: Any` in the direction hooks.
- **Defaults that are wrong for large problems.** `Optimiser(gtol=1e-8, rtol=1e-10)` — unreachable by line-search methods in double precision on O(1) objectives (§5.4). `apply_operator_function(rtol=1e-10, max_iterations=50)` — effectively "always 50 applications" per `f(A)x`, so each Gaussian sample through `operator_sqrt` costs 50 covariance applications. `random_range(rank=None, max_rank=None)` — ceiling is `min(dim)` and growth is O(k³). `random_trace(samples=100)`/`log_determinant(samples=100)` — no tolerance-driven sampling; ±1.3 on a log-det of 129 **[measured]**. `weighted_chi2_cdf(tolerance=1e-10)` with a scalar-Python integrand.
- **DESIGN vs code.** §11.7 says the steepest-descent default "is now a strong Wolfe search with the slope-ratio heuristic"; `SteepestDescent` (249–271) does not override `_default_line_search`, so it inherits `ArmijoLineSearch` (103–105) — consistent with the measured 5,330 value / 2,001 gradient calls. §11.7 also says the Lanczos port "added trait gating and dispatch rather than restructuring" — it also dropped fixed mode, `atol`, `check_interval`, the coefficient-space test, Kahan reorth and the quadratic-form convergence test.

## 4. Documentation gaps (file:line)

- functional_calculus.py:151 `lanczos_tridiagonalise` — one line, no Args/Returns, `breakdown_tol` not exposed or mentioned. :173 `apply_operator_function` — no Args/Returns; nothing says non-convergence is silent. :239 `operator_quadratic_form` — no Args; does not say there is no convergence test. :286 `OperatorFunction.__init__` — no docstring. :331 `operator_function`, :354–395 helpers — `**kwargs` never enumerated.
- randomised.py:49 `Estimate` — fields undocumented. :103 `random_range` — does not say the result may be *longer* than `rank`; `rtol` "relative residual" relative to what (first block's max norm, 155). :328/:379/:433 — `**kwargs` undocumented (they go to `random_range`); `random_svd`'s 1e-14 relative cut (412) undocumented; `random_eig` docstring says "largest first" (202) but orders by `|λ|` (359). :471 `random_trace`, :566 `random_diagonal` — no Args/Returns. :478–483 `random_trace` claims v1's estimator gave `tr(GA)`; v1's `random_trace` (low_rank.py 1121–1127) used Euclidean Rademacher probes on components and was already unbiased for `tr(A_c)` — the §9 defect concerns `white_noise_measure` in `random_range`, not the trace.
- optimisation.py:58 `OptimisationResult` — `iterations`/`evaluations`/`history` semantics undocumented (and `evaluations` counts different things in Armijo vs Wolfe). :79 `Optimiser` — convergence formula only readable from `_tolerance` (122); line-search failure → `converged=False` unstated. :623 `truncated_cg` — Args (`rtol` relative to `||rhs||`, `radius`) undocumented. :516 `TrustRegionNewton` — radius-collapse termination undocumented and mis-reported.
- line_search.py:31 `LineSearchResult` — fields undocumented, especially what `evaluations` counts.
- convex.py:402 `SubgradientDescent` — inherits `rtol` but ignores it. :513 `ProximalGradient` and :709 `ProximalPoint` — reuse `gtol` as a *step* tolerance on `||x_{k+1}−x_k||/step` (616, 631) / `||x_{k+1}−x_k||` (743) without saying so. :770 `BundleResult` — fields undocumented. :817 `tolerance` — relative (870) but documented as absolute; :799–801 "The gap is a genuine bound on the distance to the minimum" overstates: the predicted decrease of a proximal bundle model certifies approximate stationarity, not `f − f*` (v1's level method LP bound 1104–1161 was a genuine global lower bound). :654–671 `_initial_step` — absolute 1e-6 offset (667) is scale-dependent, unmentioned.
- quadratic_forms.py:112 `weighted_chi2_quantile` — `method` undocumented; `"monte_carlo"` unsupported in practice; empty/zero-weight behaviour changed from v1 without note.
- root_find.py — good. `DampedSolves.solve` (320–326) return type (`SolveResult`) unstated.
- numerics/__init__.py — no module docstring; nothing lists what is deliberately internal.

## 5. Correctness (metric-sensitive first)

1. **Lanczos, Gram–Schmidt, random_range, random_eig/svd, bundle Gram, prox operators, optimiser slopes** all go through `space.inner_product`/`norm`/`axpy`; I found no Euclidean-dot leak. Lanczos is tested on a dense-metric space (test_functional_calculus.py 274–300, 347–354). **But** `test_randomised.py`, `test_optimisation.py`, `test_convex.py`, `test_backus.py` contain **zero** dense-metric tests (only diagonal-weight "weighted" and `OpaqueSpace`, which is diagonal too: doubles.py 82–93). This violates the metric-bug rule for exactly the code whose Galerkin/component distinctions (`random_diagonal` 599–600, `deflated_diagonal` 556–560) and quasi-Newton metric claims are the point.
2. `random_trace` (490–494): white-noise probe `p` with `E[(p,u)(p,v)]=(u,v)` gives `E[(Ap,p)] = tr(A_c)` — correct, and consistent with `log_determinant`'s dense route `slogdet(Galerkin) − slogdet(G)` (466–476).
3. `random_diagonal` (596–603): Rademacher on *components*, optional `apply_gram` → `(A_c)_ii` or `(G A_c)_ii` as documented. Correct. `deflated_diagonal`'s exact part: component matrix of `UΛU*` is `Σ λ_i c_i (G c_i)^T`, Galerkin `Σ λ_i (G c_i)(G c_i)^T`; the einsums at 558/560 match. `symmetric_space/base.py:635` asks for `form="components"`, which is the right one for a pointwise variance.
4. **Optimiser termination.** **[measured]** With `f = O(1)` and `|g| ≈ 1e-7`, `c1·α·slope ≈ 1e-18` is below the rounding floor of `f`, so the Armijo test fails; Wolfe returns `converged=False` → "line search failed" while `NonlinearCG` at `gtol=1e-6` stopped at `|g|=1.12e-6` with the same message; Armijo instead accepts `2^-35` steps for hundreds of iterations. There is no "precision loss" outcome and no value-based criterion. Any large-scale user will see either spurious failures or thousands of wasted evaluations.
5. `operator_function` diagonal branch (348–350) skips `_require_self_adjoint`: on a non-diagonal-metric space a `DiagonalLinearOperator` is *not* self-adjoint (diagonal.py 66–72 correctly declines the trait) but `operator_function(D, f)` silently returns a component-wise `f(D)` that is not the spectral calculus of any self-adjoint operator. The named helpers are safe because they also require PSD/PD.
6. `random_eig` orders by `|λ|` (359) — for an indefinite operator followed by `apply_function` the sign is what matters; docstring says "largest".
7. `SupportFunction.__mul__` (274–278) rejects `np.integer` scalars and silently falls to generic `Functional` scaling, losing the support algebra.
8. `monotone_root` classifying `ConvergenceError` as saturation (root_find.py 42, 204–207) can convert an under-iterated CG into a wrong "exhausted" answer — the failure mode §24.1 warns about, reintroduced through the exception handler.
9. Lanczos `breakdown_tol` relative to `max(|alpha|,1)` (130) rather than to the spectral scale; for an operator with `|alpha| ≪ 1` but large `||v||` this is looser than v1's `max_scale`.

## Recommendations

**Must**
1. Fix optimiser termination (optimisation.py 133–215, 444–513): add a value-decrease/step criterion (`ftol`, as v1's SciPy path had) and, when the line search fails with `|g|` already below a looser tolerance (e.g. `sqrt(eps)·(1+|f|)`-scaled), return `converged=True, message="precision loss"`; change default `gtol` to something reachable or relative (`rtol` on `||g0||` at 1e-6 rather than 1e-10).
2. Make the diagonal fast path survive `with_traits`: override `DiagonalLinearOperator.with_traits` to `_rebuild` with merged traits (diagonal.py 126–128), and make `log_determinant` (functional_calculus.py 406) return `Estimate(D.log_determinant, 0, 0)` for a `DiagonalLinearOperator`. Then remove or relax the `.with_traits(definite)` wrapping at inference/gaussian.py 417–420 and probability/gaussian.py 531–545, or at least verify they now hit the exact path.
3. Restore the coefficient-space convergence test in `apply_operator_function` (compare `g_k` with padded `g_{k−1}` as v1 221–231; recombine once at the end) and add a convergence test to `operator_quadratic_form` (v1 331–335). Both currently cost O(k²) vector ops or never stop early.
4. Fix `random_range` adaptive orthogonalisation (randomised.py 149–164): orthogonalise only the new block's residuals against the existing basis (as v1 712–733) and keep the already-computed residuals rather than calling `orthonormal_basis(basis + block)`; add a `CoordinateSpace` QR path (`gram_matrix` Cholesky-weighted QR on components) for the common case.
5. Return the gradient (or the `QuadraticModel`) in `LineSearchResult` and use it in `_DescentMethod._minimise` (192) instead of re-evaluating; count the `_zoom` initial evaluation (line_search.py 243) or pass the known value in.
6. Add dense (non-diagonal) Gram-matrix tests for `random_diagonal`, `deflated_diagonal`, `random_eig/svd`, `LBFGS`, `ProximalGradient`, `ProximalBundleMethod`, using `make_dense_metric_space` from `tests/conftest.py` as `test_functional_calculus.py` already does.
7. Correct the catalogue: `PrimalKKTSolver`, `solve_support_values`, `solve_primal_feasibility` → Subsumed (by `monotone_root`/`ProximalBundleMethod` in backus.py); `weighted_chi2_quantile` → no saddlepoint; `deflated_diagonal` → Ported; `QPSolver*` → Dropped per DESIGN §22.7. Correct DESIGN §11.7's steepest-descent line-search claim (code inherits Armijo) or make `SteepestDescent` override `_default_line_search` to `StrongWolfeLineSearch`.

**Should**
8. Export `ProximalBundleMethod`, `BundleResult`, `deflated_diagonal`, `monotone_root`, `DampedSolves`, `Evaluation`, `RootResult`, `weighted_chi2_cdf/quantile` from `numerics/__init__.py`; add `truncated_cg` to `optimisation.__all__` and `deflated_diagonal` to `randomised.__all__`.
9. Make `ProximalBundleMethod` an `Optimiser` returning `OptimisationResult` (or give `BundleResult` the same field names), and document that `tolerance` is relative; soften the "genuine bound on the distance to the minimum" claim (799–801).
10. In `_solve_model`, keep the Gram matrix incrementally (one new row per iteration) and give `_minimise_on_simplex` a duality-gap or KKT-residual stopping test with a warning when 400 iterations are exhausted; consider an optional QP backend (OSQP/Clarabel on the k-variable dual) for accuracy.
11. Reinstate two-pass ("twice is enough") reorthogonalisation in Lanczos (functional_calculus.py 123–125) using `space._orthogonalise_against`-style logic, and expose `breakdown_tol` on `lanczos_tridiagonalise`.
12. Restore tolerance-driven sampling for `random_trace`/`log_determinant` (stop when `standard_error ≤ rtol·|value|`, with `max_samples`), which the `Estimate` type makes trivial; likewise for `random_diagonal`.
13. Add an `Optimiser`-level `callback(iteration, x, model)` hook and an opt-in iterate history; document `evaluations` units.
14. Fix `TrustRegionNewton`'s radius-collapse message/count (587–589, 608–620); fix Armijo's failure `evaluations` (128); stop `SubgradientDescent` evaluating `f(x)` twice per iteration (491/495); count `ProximalGradient`'s extra evaluations (630, 703).
15. In `monotone_root`, only treat `LinAlgError` (and a solver-reported *singularity*) as saturation; re-raise a plain iteration-cap `ConvergenceError` or surface it in `RootResult`.
16. `weighted_chi2_quantile`: accept `samples`/`rng`, reject `method="monte_carlo"` or implement it via empirical quantile (v1 698–706); either vectorise the Imhof integrand or port v1's trapezoid (486–569) to recover the 25× at small n; replace `except Exception` (105) with the specific `IntegrationWarning`/`ValueError`.
17. `operator_function`'s diagonal branch should also require `SELF_ADJOINT` (or document that it is component-wise, not spectral, on a non-diagonal metric).

**Consider**
18. Add `max_rank` defaults tied to a budget (e.g. `min(dim, 200)`) for adaptive `random_range`, and a warning when the ceiling is hit.
19. Restore parallel probe evaluation (`n_jobs`) in `random_trace`/`random_range`/`random_diagonal` and multi-direction warm-started support-value evaluation (v1 `solve_support_values`) in backus.py.
20. Interpolating (cubic/quadratic) `_zoom` in `StrongWolfeLineSearch`.
21. Remove dead code (`_ConvexResult`, `random_eig` 364–367), dedupe the result-construction tails, replace `codomain._orthogonalise_against` with a public method, and type vectors with the space's `V` rather than `Any`.
22. Give `LowRankSVD` traits when `left is right` and the singular values are non-negative; restore `LowRankEig.apply_function(regularization=)` or document the replacement (`(D + εI).apply_function`).
23. Consider a `LevelBundleMethod` port (v1 1014–1387) if a certified global gap is wanted for the dual route; the proximal method's gap is not that.
