# Plan Complete: Bundle Methods for DualMasterCostFunction

Bundle methods, optional QP backends (SciPy / OSQP / Clarabel), a multi-direction batch helper,
a smoothing + L-BFGS-B fast path, and a Chambolle–Pock primal feasibility solver are all fully
implemented in `pygeoinf/convex_optimisation.py`. All 386 tests pass (excluding one pre-existing
`pytest.fail` TODO stub unrelated to this work).

**Phases Completed:** 7 of 7
1. ✅ Phase 1: Core data structures and QP abstraction
2. ✅ Phase 2: `value_and_subgradient` on DualMasterCostFunction + ProximalBundleMethod
3. ✅ Phase 3: LevelBundleMethod
4. ✅ Phase 4: Optional QP backends (OSQP + Clarabel)
5. ✅ Phase 5: `solve_support_values` multi-direction batch helper
6. ✅ Phase 6: Smoothing + L-BFGS-B fast path
7. ✅ Phase 7: ChambollePockSolver (primal feasibility form)

---

**All Files Created/Modified:**

- `pygeoinf/pygeoinf/convex_optimisation.py` — all new classes/functions appended
- `pygeoinf/pygeoinf/backus_gilbert.py` — `DualMasterCostFunction.value_and_subgradient` added
- `pygeoinf/pyproject.toml` — `fast-bundle = ["osqp"]` and `bundle-alt = ["clarabel"]` optional extras
- `pygeoinf/tests/test_bundle_core.py` — Phase 1 tests (10 tests)
- `pygeoinf/tests/test_proximal_bundle.py` — Phase 2 tests (6 tests)
- `pygeoinf/tests/test_dual_master_cost.py` — Phase 2 tests replacing stub (1 test)
- `pygeoinf/tests/test_level_bundle.py` — Phase 3 tests (6 tests)
- `pygeoinf/tests/test_qp_backends.py` — Phase 4 tests (6 tests)
- `pygeoinf/tests/test_solve_support_values.py` — Phase 5 tests (5 tests)
- `pygeoinf/tests/test_smoothed_lbfgs.py` — Phase 6 tests (5 tests)
- `pygeoinf/tests/test_chambolle_pock.py` — Phase 7 tests (5 tests)
- `pygeoinf/plans/pygeoinf-reference.md` — updated throughout with all new public symbols

---

**Key Functions/Classes Added:**

### Phase 1 (data structures)
- `Cut` — `@dataclass(x, f, g, iteration)`, one linear cut of the objective
- `Bundle` — manages cuts; provides `linearization_matrix`, `compress`, `upper_bound`, `best_point`
- `QPResult` — `@dataclass(x, obj, status)`, standardised QP output
- `QPSolver` — `@runtime_checkable Protocol` for OSQP standard form
- `SciPyQPSolver` — SLSQP backend (always available)
- `BundleResult` — `@dataclass` with `x_best, f_best, f_low, gap, converged, ...`

### Phase 2 (proximal bundle)
- `DualMasterCostFunction.value_and_subgradient` — shares G*λ computation for value+subgradient
- `_get_value_and_subgradient` — duck-typed helper (uses `value_and_subgradient` if available)
- `ProximalBundleMethod` — `solve(x0) -> BundleResult` with ρ adaptation, serious/null steps, gap cert

### Phase 3 (level bundle)
- `LevelBundleMethod` — level QP with `f_lev = α f_low + (1-α) f_up`; infeasibility recovery (3
  attempts widening α, then proximal fallback); lower bound via regularised LP

### Phase 4 (backends)
- `OSQPQPSolver` — ADMM solver; warm-start; ±inf → ±1e30; `polishing=True`
- `ClarabelQPSolver` — interior-point; native QP path; converts to cone form
- `best_available_qp_solver()` — factory: OSQP > Clarabel > SciPy

### Phase 5 (batch)
- `solve_support_values(cost, qs, solver, lambda0, *, warm_start, n_jobs)` — sequential warm-start
  or joblib parallel; returns `(values, lambdas, diagnostics)`

### Phase 6 (smoothing)
- `SmoothedDualMaster` — Moreau-Yosida smoothing for ball and ellipsoid supports; analytic gradient
- `SmoothedLBFGSSolver` — continuation schedule + L-BFGS-B; `BundleResult` with `gap=nan`

### Phase 7 (Chambolle–Pock)
- `ChambollePockResult` — `@dataclass(m, v, mu, primal_dual_gap, converged, num_iterations)`
- `ChambollePockSolver` — primal-dual algorithm; auto step-sizes via power iteration; ball
  projection; `solve(c, m0) -> ChambollePockResult`
- `solve_primal_feasibility(cost, qs, cp_solver)` — batch wrapper; feasible set solved once per q

---

**Test Coverage:**
- Total tests written: 44 new tests across 7 test files
- All tests passing: ✅ (386 total in full suite)

---

**Recommendations for Next Steps:**
- Add `underlying_subset` property to `EllipsoidSupportFunction` and implement ellipsoid projection
  in `ChambollePockSolver` (needed for the ellipsoid prior case)
- Benchmark `ProximalBundleMethod` vs `LevelBundleMethod` vs `SmoothedLBFGSSolver` on a realistic
  Backus-Gilbert problem (vary $n_\text{data}$, $n_\text{model}$)
- Wire `best_available_qp_solver()` as the default in `ProximalBundleMethod` and `LevelBundleMethod`
  constructors (currently defaults to `SciPyQPSolver`)
- Add the `underlying_subset` property to `SupportFunction` as planned (open question 3 from the plan)
