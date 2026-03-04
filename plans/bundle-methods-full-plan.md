# Plan: Bundle Methods for DualMasterCostFunction

Implements proximal bundle, level bundle, optional OSQP/Clarabel QP backends, a multi-direction batch
helper, a smoothing + L-BFGS-B fast path for ball/ellipsoid sets, and a Chambolle–Pock primal
feasibility solver — all inside `pygeoinf/convex_optimisation.py`. Supersedes the draft in
`bundle-methods-optimizer-plan.md`.

**Phases (7)**

---

### Phase 1: Core Data Structures and QP Abstraction

- **Objective:** Define the building blocks shared by all bundle variants: `Cut`, `Bundle`,
  `BundleResult`, `QPResult`, the `QPSolver` Protocol, and the always-available `SciPyQPSolver`.
  No solver loop yet — only infrastructure and their tests.

- **Files/Functions to Modify/Create:**
  - [pygeoinf/pygeoinf/convex_optimisation.py](../pygeoinf/convex_optimisation.py):
    - `@dataclass Cut` — fields `x: Vector, f: float, g: Vector, iteration: int`
    - `class Bundle` — `add_cut(cut)`, `lower_bound() -> float`, `upper_bound() -> float`,
      `best_point() -> Vector`, `linearization_matrix() -> tuple[np.ndarray, np.ndarray]`
      (returns `A, b` such that cut constraints are `A @ lam >= b`),
      `compress(max_size: int)`, `__len__`
    - `@dataclass QPResult` — `x: np.ndarray, obj: float, status: str`
    - `class QPSolver(Protocol)` — `solve(P, q, A, l, u, x0=None) -> QPResult`
    - `class SciPyQPSolver` — implements `QPSolver` using scipy SLSQP; converts Hilbert-space
      vectors to/from numpy arrays via `domain.to_components` / `domain.from_components`
    - `@dataclass BundleResult` — `x_best: Vector, f_best: float, f_low: float, gap: float,
      converged: bool, num_iterations: int, num_serious_steps: int, function_values: list[float],
      iterates: Optional[list[Vector]]`
  - [pygeoinf/tests/test_bundle_core.py](../tests/test_bundle_core.py) *(new file)*

- **Tests to Write (`test_bundle_core.py`):**
  - `test_cut_fields` — construct a `Cut`, check all fields stored correctly
  - `test_bundle_lower_bound_single_cut` — one linear cut; `lower_bound()` equals the cut's
    unconstrained minimum
  - `test_bundle_lower_bound_two_cuts` — two cuts that agree at a point; lower bound at that point
  - `test_bundle_linearization_matrix_shape` — `A` has shape `(num_cuts, n_vars)`, `b` shape
    `(num_cuts,)`
  - `test_bundle_compress_keeps_recent` — after compress to `max_size=2`, only 2 cuts remain
  - `test_scipy_qp_solver_simple` — solve a 2D QP with known solution, check residual < 1e-6
  - `test_scipy_qp_solver_infeasible` — detect infeasibility; `QPResult.status != 'solved'`

- **Steps:**
  1. Write `test_bundle_core.py` with all seven tests (all fail: classes don't exist yet).
  2. Implement `Cut`, `Bundle`, `QPResult`, `QPSolver`, `SciPyQPSolver`, `BundleResult` in
     `convex_optimisation.py` (append after `SubgradientDescent`).
  3. Run `python -m pytest tests/test_bundle_core.py -x` — all seven tests pass.
  4. Run full test suite; confirm no regressions.

---

### Phase 2: `value_and_subgradient` on DualMasterCostFunction + ProximalBundleMethod ✅ COMPLETE

- **Objective:** (a) Add a `value_and_subgradient(lam) -> tuple[float, Vector]` method to
  `DualMasterCostFunction` that avoids computing `Gstar_lam` and the hilbert residual twice
  (once for the value, once for the subgradient). (b) Implement `ProximalBundleMethod` which
  solves $\min_{\lambda,t}\; t + \tfrac{\rho}{2}\|\lambda - \hat\lambda\|^2$ s.t. cuts.

- **Implemented:**
  - `DualMasterCostFunction.value_and_subgradient(lam)` in `backus_gilbert.py` — shares
    `Gstar_lam`, `hilbert_residual`, support-point queries; falls back to finite-diff if None.
  - `_get_value_and_subgradient(oracle, x)` helper in `convex_optimisation.py` that duck-types:
    calls `oracle.value_and_subgradient(x)` if available, else `(oracle(x), oracle.subgradient(x))`.
  - `ProximalBundleMethod` in `convex_optimisation.py`:
    - Constructor: `(oracle, /, *, rho0, rho_factor, tolerance, max_iterations, bundle_size, store_iterates, qp_solver)`
    - `_solve_master(bundle, lam_hat, rho, domain, x0_comps, t_warm) -> (lam_next, t_opt)`:
      builds OSQP-form QP, returns `(lam_next, result.x[d])` where `result.x[d]` is the
      cutting-plane model value `hat_phi(lam_next)` (correct t-variable from QP solution).
    - `solve(x0) -> BundleResult`: proximal bundle loop.
    - Convergence criterion: after a **null step**, if `f_hat - f_low ≤ tolerance`.
      `f_low` (cutting-plane lower bound) is reset to `-inf` on each **serious step** and
      updated by `max(f_low, t_opt)` on each **null step**.  This prevents spurious
      convergence caused by the cutting-plane model being exact on linear regions.
  - Tests in `tests/test_proximal_bundle.py` (6 tests) and `tests/test_dual_master_cost.py`.

- **Key implementation decisions:**
  - `t_opt = result.x[d]` (the QP `t`-variable at the solution), NOT `qp_obj - (rho/2)||prox||^2`.
    The SLSQP objective is `(rho/2)||λ||^2 - rho*lam_hat·λ + t` (missing the constant
    `+(rho/2)||lam_hat||^2`), so subtracting `(rho/2)*prox_dist^2` from `qp_obj` gives the
    wrong `t_opt`. Reading `result.x[d]` directly is correct.
  - Warm-start: `t_warm = f_hat` (always feasible since cuts at lam_hat satisfy ≤ f_hat).
  - `f_low` reset on serious steps prevents false gap-convergence when the model is tight
    by coincidence (e.g. affine function = exact cutting plane at the evaluated point).

- **Test results (all passing):**
  - `test_proximal_bundle_quadratic_1d` ✅
  - `test_proximal_bundle_convex_nonsmooth_1d` ✅
  - `test_proximal_bundle_gap_certificate` ✅
  - `test_proximal_bundle_serious_steps` ✅
  - `test_proximal_bundle_dual_master` ✅
  - `test_proximal_bundle_dual_master_gap` ✅
  - `test_value_and_subgradient_consistency` ✅

---

### Phase 3: LevelBundleMethod (parallel with Phase 2)

- **Objective:** Implement `LevelBundleMethod` which adds level management on top of the proximal
  bundle structure. Key additions: level computation $f_\text{lev} = \alpha f_\text{low} + (1-\alpha)
  f_\text{up}$, level infeasibility handling (fall back to tighter level or proximal step), and the
  constraint $t \le f_\text{lev}$ in the QP.

- **Files/Functions to Modify/Create:**
  - [pygeoinf/pygeoinf/convex_optimisation.py](../pygeoinf/convex_optimisation.py):
    - `class LevelBundleMethod` — constructor: `oracle, /, *, alpha=0.1, tolerance=1e-6,
      max_iterations=500, bundle_size=100, store_iterates=False, qp_solver=None`
      Method: `solve(x0: Vector) -> BundleResult`
      Internal: `_compute_level(f_low, f_up) -> float`, `_handle_infeasible(bundle, lam_hat, f_lev)`
      (tighten level or fall back to proximity center)
  - [pygeoinf/tests/test_level_bundle.py](../tests/test_level_bundle.py) *(new file)*

- **Tests to Write (`test_level_bundle.py`):**
  - `test_level_bundle_quadratic_1d` — same setup as proximal bundle quadratic test; check
    convergence to $\lambda^* = -1$
  - `test_level_bundle_nonsmooth_1d` — minimize $|\lambda - 0.5|$; converge to 0.5
  - `test_level_bundle_gap_certificate` — gap ≤ tolerance at convergence
  - `test_level_bundle_infeasibility_recovery` — construct a scenario where initial level is
    infeasible (set `alpha=1e-6`); verify solver does not crash and gap eventually decreases
  - `test_level_bundle_dual_master` — same `DualMasterCostFunction` fixture; result ≤ initial
  - `test_level_vs_proximal_bundle_agreement` — on the 1D quadratic, gap at convergence is
    with a factor of 3 across both methods for the same tolerance

- **Steps:**
  1. Write tests (all fail).
  2. Implement `LevelBundleMethod` in `convex_optimisation.py`.
  3. Run tests; iterate.
  4. Full suite regression check.

---

### Phase 4: Optional QP Backends (OSQP and Clarabel)

- **Objective:** Add `OSQPQPSolver` and `ClarabelQPSolver` implementations of the `QPSolver`
  protocol. Both are optional (guarded by `try: import osqp` / `try: import clarabel`). Both
  support warm-starting via `x0` parameter. Add a factory function `best_available_qp_solver()`
  that auto-detects and returns the best available backend.

- **Files/Functions to Modify/Create:**
  - [pygeoinf/pygeoinf/convex_optimisation.py](../pygeoinf/convex_optimisation.py):
    - `class OSQPQPSolver` — wraps `osqp.OSQP`; calls `osqp_solver.warm_start(x=x0)` when
      `x0 is not None`; sets `osqp_solver.update_settings(verbose=False)`; single `osqp_solver`
      instance reused across calls to amortise setup
    - `class ClarabelQPSolver` — wraps `clarabel`; builds `clarabel.DefaultSolver`; no warm-start
      (interior-point), but re-sets `initial_variable` if provided
    - `def best_available_qp_solver() -> QPSolver` — returns `OSQPQPSolver` if osqp importable,
      else `ClarabelQPSolver` if clarabel importable, else `SciPyQPSolver`
  - [pygeoinf/pyproject.toml](../pyproject.toml):
    - Add optional extras `fast-bundle = ["osqp"]` and `bundle-alt = ["clarabel"]`
  - [pygeoinf/tests/test_qp_backends.py](../tests/test_qp_backends.py) *(new file)*

- **Tests to Write (`test_qp_backends.py`):**
  - `test_scipy_qp_known_solution` — same 2D QP; check `|x - x_true|_inf < 1e-5`
  - `test_osqp_qp_known_solution` — same QP with `OSQPQPSolver`; skip if osqp not installed
  - `test_clarabel_qp_known_solution` — same QP with `ClarabelQPSolver`; skip if clarabel not installed
  - `test_osqp_warm_start_reduces_iterations` — run same QP twice with `OSQPQPSolver`, second time
    with `x0` from first; check `result2.iters <= result1.iters`
  - `test_best_available_returns_solver` — `best_available_qp_solver()` returns a `QPSolver`
    instance; `solve()` works; skip if osqp/clarabel not installed with `pytest.importorskip`
  - `test_backends_agree_on_solution` — for all available backends, solve same QP and check all
    solutions agree up to 1e-4

- **Steps:**
  1. Write tests with `pytest.importorskip('osqp')` / `pytest.importorskip('clarabel')` guards.
  2. Implement `OSQPQPSolver` and `ClarabelQPSolver` with optional import guards at class
     definition time (raise `ImportError` in `__init__` if not installed).
  3. Implement `best_available_qp_solver`.
  4. Update `pyproject.toml` extras.
  5. Run `python -m pytest tests/test_qp_backends.py` — SciPy tests pass; OSQP/Clarabel tests
     skip cleanly if not installed.
  6. Run both `ProximalBundleMethod` and `LevelBundleMethod` with each backend; confirm
     `test_proximal_bundle_dual_master` and `test_level_bundle_dual_master` still pass using
     `best_available_qp_solver()` as the default.

---

### Phase 5: Multi-Direction Batch Helper (`solve_support_values`)

- **Objective:** Implement `solve_support_values` which takes a list of directions $q_1,\ldots,q_p$
  and runs the bundle solver on each, warm-starting $\lambda$ from the previous direction's optimum.
  Supports `n_jobs > 1` via joblib for independent parallel evaluation.

- **Files/Functions to Modify/Create:**
  - [pygeoinf/pygeoinf/convex_optimisation.py](../pygeoinf/convex_optimisation.py):
    - `def solve_support_values(cost: DualMasterCostFunction, qs: list[Vector] | np.ndarray,
        solver, lambda0: Vector, *, warm_start: bool = True, n_jobs: int = 1)
        -> tuple[np.ndarray, list[Vector], list[BundleResult]]` —
      When `n_jobs == 1`: sequential loop with $\lambda$ warm-start.
      When `n_jobs > 1`: uses `joblib.Parallel`; each job gets a deep copy of `cost` and
      `solver` (no shared state); warm-start is per-job only.
  - [pygeoinf/tests/test_solve_support_values.py](../tests/test_solve_support_values.py) *(new)*

- **Tests to Write (`test_solve_support_values.py`):**
  - `test_single_direction` — single $q$; result shape is `(1,)`, matches direct `solver.solve`
    result; gap ≤ tolerance
  - `test_multiple_directions_sequential` — three orthogonal unit-vector directions; all support
    values finite; `np.testing.assert_allclose` that values equal individually solved results
    (rtol=1e-4)
  - `test_warm_start_vs_cold_start_agreement` — same three directions; `warm_start=True` and
    `warm_start=False` agree on support values up to 1e-4
  - `test_parallel_agrees_with_sequential` — `n_jobs=2` (or skip if `joblib` not available);
    parallel result agrees with sequential up to 1e-4
  - `test_returns_correct_types` — check return types: `values` is `np.ndarray`, `lambdas` is
    `list[Vector]`, `diagnostics` is `list[BundleResult]`

- **Steps:**
  1. Write tests (fail).
  2. Implement `solve_support_values`.
  3. Run tests; iterate.
  4. Full suite regression check.

---

### Phase 6: Smoothing + L-BFGS-B Fast Path ✅ COMPLETE

- **Objective:** Add an alternative optimizer for the case where both `model_prior_support` and
  `data_error_support` are `BallSupportFunction` or `EllipsoidSupportFunction`. In these cases
  the objective is smoothable to $L_\varepsilon$-smooth, enabling L-BFGS-B. Implement a
  `SmoothedDualMaster` wrapper that computes $\varphi_\varepsilon$ and its gradient, and a
  `SmoothedLBFGSSolver` that runs the continuation schedule.

- **Implemented:**
  - `class SmoothedDualMaster` in `convex_optimisation.py`:
    - Constructor: `(cost, epsilon: float)`
    - `__call__(lam) -> float` — smooth $\varphi_\varepsilon(\lambda)$
    - `gradient(lam) -> Vector` — analytic gradient via chain rule through
      $z_1 = T^*q - G^*\lambda$ and $z_2 = -\lambda$
    - `_smoothed_ball_value_and_grad(z, sigma)` — value + grad of
      $\sigma_{B,\varepsilon}(z) = \langle z,c\rangle + r\sqrt{\|z\|^2+\varepsilon^2}$
    - `_smoothed_ellipsoid_value_and_grad(z, sigma)` — value + grad of
      $\sigma_{E,\varepsilon}(z) = \langle z,c\rangle + r\sqrt{\langle z,A^{-1}z\rangle+\varepsilon^2}$;
      raises `NotImplementedError` if `sigma._A_inv is None`
    - `_eval_support(z, sigma)` — dispatches on type; raises `NotImplementedError` for unknown types
  - `class SmoothedLBFGSSolver` in `convex_optimisation.py`:
    - Constructor: `(cost, /, *, epsilon0=1e-2, n_levels=5, tolerance=1e-6, max_iter_per_level=500)`
    - `solve(lam0) -> BundleResult` — geometric continuation
      $\varepsilon_i = \varepsilon_0 \times 10^{-i}$; each level solved with
      `scipy.optimize.minimize(..., method='L-BFGS-B', jac=True)`; warm-start between levels;
      returns `BundleResult` with `gap=nan`, `f_low=nan`
  - Tests in `tests/test_smoothed_lbfgs.py` (5 tests, all passing)
  - Import added to `convex_optimisation.py`:
    `from .convex_analysis import BallSupportFunction, EllipsoidSupportFunction`

- **Key implementation decisions:**
  - Used `_eval_support()` internal dispatch to avoid code duplication between `__call__` and
    `gradient`.
  - Gradient formula: $\nabla_\lambda\varphi_\varepsilon = \tilde{d} - G\nabla_{z_1}\sigma_{B,\varepsilon} - \nabla_{z_2}\sigma_{V,\varepsilon}$ via chain rule through $z_1 = T^*q - G^*\lambda$ (factor $-G^*$, adjoint $-G$) and $z_2=-\lambda$ (factor $-I$).
  - `test_smoothed_raises_for_unsupported_support` uses a custom `_UnsupportedSupport(SupportFunction)`
    subclass rather than `HalfSpaceSupportFunction` (the latter does not subclass `SupportFunction`
    and would be rejected by `DualMasterCostFunction`'s validator).
  - `_smoothed_ellipsoid_value_and_grad` uses `max(inner_term, 0.0)` before `sqrt` to guard
    against tiny negative values from floating-point noise.

- **Test results (all passing):**
  - `test_smoothed_ball_gradient_consistency` ✅ — FD rtol=1e-3
  - `test_smoothed_ellipsoid_gradient_consistency` ✅ — FD rtol=1e-3
  - `test_smoothed_lbfgs_converges_ball` ✅
  - `test_smoothed_lbfgs_agrees_with_proximal_bundle` ✅ — rtol=1e-2
  - `test_smoothed_raises_for_unsupported_support` ✅

---

### Phase 7: ChambollePockSolver (Primal Feasibility)

- **Objective:** Implement `ChambollePockSolver` that solves the primal-feasibility form

  $$\max_{m \in B,\, v \in V}\; \langle c,\, m\rangle \quad\text{s.t.}\quad Gm + v = \tilde d$$

  via the Chambolle–Pock algorithm (Chambolle & Pock 2011). The solver exploits the direction-independence
  of the feasible set: a single solve returns $(m^*, v^*, \mu^*)$ and evaluating $\langle T^* q_i, m^*\rangle$
  for multiple $q_i$ is then free. Convergence: $O(1/N)$ with $\tau\sigma\|G\|^2 \le 1$.

- **Files/Functions to Modify/Create:**
  - [pygeoinf/pygeoinf/convex_optimisation.py](../pygeoinf/convex_optimisation.py):
    - `@dataclass ChambollePockResult` — `m: Vector, v: Vector, mu: Vector, primal_dual_gap: float,
      converged: bool, num_iterations: int`
    - `class ChambollePockSolver` — constructor: `B: SupportFunction, V: SupportFunction,
      G: LinearOperator, d_tilde: Vector, /, *, sigma: float | None = None, tau: float | None = None,
      theta: float = 1.0, max_iterations: int = 1000, tolerance: float = 1e-8`
      (if `sigma` and `tau` are None, auto-compute as $\tau = \sigma = 1 / (\|G\|_\text{est} + \varepsilon)$
      using a power-iteration estimate of $\|G\|$).
      Method: `solve(c: Vector, m0: Vector | None = None) -> ChambollePockResult`
      Internal: `_prox_B(m, tau) -> Vector` using `B`'s underlying subset's `project` or
      `_ball_prox` / `_ellipsoid_prox` directly; `_prox_V(v, tau) -> Vector` similarly;
      `_prox_equality(y, sigma) -> Vector` (projection onto $\{\tilde d\}$: trivially return $\tilde d$).
    - `def solve_primal_feasibility(cost: DualMasterCostFunction, qs: list[Vector],
        solver: ChambollePockSolver) -> np.ndarray` — convenience wrapper that calls
      `solver.solve(c = T.adjoint(q_i))` is NOT right (c appears in the objective, not the
      constraints) — need to solve one feasibility problem first, then evaluate linear objectives.
      Actually: solve once to get feasible $(m^*, v^*)$, then return
      `[domain.inner_product(cost.T.adjoint(q), m_star) for q in qs]`.
  - [pygeoinf/tests/test_chambolle_pock.py](../tests/test_chambolle_pock.py) *(new file)*

- **Tests to Write (`test_chambolle_pock.py`):**
  - `test_chambolle_pock_feasibility` — after convergence, verify $\|Gm^* + v^* - \tilde d\|
    \le 10 \cdot \text{tol}$
  - `test_chambolle_pock_m_in_B` — $m^* \in B$ (check via `B.support_point(G.adjoint(mu^*))`
    or `B(m^*) \le r^2`)
  - `test_chambolle_pock_v_in_V` — $v^* \in V$
  - `test_chambolle_pock_primal_dual_gap` — `primal_dual_gap <= tolerance * 10`
  - `test_chambolle_pock_objective` — $\langle c, m^*\rangle$ agrees with `LevelBundleMethod`
    on same problem (within factor of 2 of tolerance, since C-P requires more iterations for
    tight primal optimality)
  - `test_solve_primal_feasibility_multiple_directions` — run `solve_primal_feasibility` on 3
    directions; values agree with `solve_support_values` (same solver for reference) to within
    1e-3

- **Steps:**
  1. Write tests (fail).
  2. Implement `ChambollePockResult`, `ChambollePockSolver`, `solve_primal_feasibility`.
  3. Run tests; iterate.
  4. Full suite regression check.
  5. Update `pygeoinf/plans/pygeoinf-reference.md` to document all new classes/functions.

---

## Open Questions

1. **`NonLinearForm` oracle interface**: Should `ProximalBundleMethod` / `LevelBundleMethod` accept
   any `NonLinearForm` with `has_subgradient=True`, or be typed directly to
   `DualMasterCostFunction`? The former is more reusable; the latter enables `value_and_subgradient`.
   **Proposed**: Accept `NonLinearForm`, but special-case `DualMasterCostFunction` by duck-typing
   `hasattr(oracle, 'value_and_subgradient')`.

2. **Vector ↔ numpy conversion**: The `QPSolver` interface uses `np.ndarray`; the Hilbert space
   layer uses abstract `Vector`. Conversion via `domain.to_components(v) -> np.ndarray` and
   `domain.from_components(arr) -> Vector` — confirm these methods exist on all `EuclideanSpace`
   instances before Phase 1 implementation begins.

3. **`Subset.project` vs prox via support function**: `ChambollePockSolver` needs $\text{prox}_{\tau\delta_B}(m) = \text{proj}_B(m)$.
   The `SupportFunction` classes don't expose `project` directly — it lives on the `Subset` base
   class in `subsets.py`. How to link a `SupportFunction` to its underlying `Subset.project`?
   **Proposed**: add `underlying_subset: Optional[Subset]` property to `SupportFunction` subclasses,
   or pass the `Subset` directly to `ChambollePockSolver`.
