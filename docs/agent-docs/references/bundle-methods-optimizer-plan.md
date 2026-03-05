# Plan: Bundle Methods Optimizer for Non-Smooth Convex Optimization

**TL;DR:** Implement level bundle methods in pygeoinf to solve non-smooth convex minimization problems with superior convergence compared to subgradient descent. Bundle methods build cutting-plane models, solve quadratic master problems, and provide automatic step sizing with gap-based optimality certificates. Target application: DualMasterCostFunction minimization for computing support values h_U(q).

**Phases: 7**

## Phase 1: Data Structures & QP Interface

**Objective:** Establish foundational components for bundle management and quadratic programming solver interface.

**Files/Functions to Modify/Create:**
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Add `Cut`, `Bundle`, `QPSolver` classes
- [tests/test_bundle_method.py](tests/test_bundle_method.py) - New test file

**Tests to Write:**
- `test_cut_dataclass_creation()` - Create Cut with vector, value, subgradient
- `test_bundle_add_cut()` - Add cuts to bundle, verify storage
- `test_bundle_get_constraints()` - Extract linearization constraints for QP
- `test_qp_solver_toy_problem()` - Solve simple QP min ||x||² s.t. x ≥ 5
- `test_qp_solver_bundle_constraints()` - QP with cutting-plane constraints

**Steps:**
1. **Define Cut dataclass** (~10 LOC)
   ```python
   @dataclass
   class Cut:
       x: Vector          # Point where linearization computed
       f_x: float         # Function value f(x)
       g: Vector          # Subgradient g ∈ ∂f(x)
       iteration: int     # When cut was generated
   ```

2. **Implement Bundle class** (~40 LOC)
   - `__init__(max_size: int = 100)` - Initialize empty bundle with size limit
   - `add_cut(x, f_x, g, iteration)` - Append cut to bundle
   - `get_constraints(x_current)` - Return list of (a_j, b_j) for f(x) ≥ ⟨g_j, x - x_j⟩ + f(x_j)
   - `__len__()` - Return number of cuts in bundle
   - Properties: `best_point`, `best_value` (tracking minimum over bundle)

3. **Implement QPSolver wrapper** (~30 LOC)
   - `solve_master_problem(bundle, stability_center, f_level, domain)` → (x_next, success)
   - Use `scipy.optimize.minimize(method='SLSQP')` for constrained QP
   - QP formulation:
     ```python
     # min (1/2)||x - x_hat||²
     # s.t. f(x_j) + ⟨g_j, x - x_j⟩ ≤ r_j  for all j
     #      sum_j r_j ≤ f_level
     ```
   - Return (x_next, True) if feasible, (None, False) if infeasible

4. **Write tests** - Test each component independently
   - Cuts store data correctly
   - Bundle accumulates cuts and tracks best point
   - QPSolver solves toy QP problems correctly

5. **Run tests** - Verify all tests pass (green)

6. **Lint/format** - Run ruff/black formatting tools

**Acceptance Criteria:**
- Cut dataclass stores vector, value, subgradient
- Bundle class manages cuts with size limit
- QPSolver successfully solves 2D toy quadratic programs with linear constraints
- All tests pass with ✅

---

## Phase 2: Basic Bundle Iteration Loop

**Objective:** Implement core BundleMethod class with basic iteration structure that computes next iterates via QP master problem.

**Files/Functions to Modify/Create:**
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Add `BundleMethod` class, `BundleResult` dataclass
- [tests/test_bundle_method.py](tests/test_bundle_method.py) - Add iteration tests

**Tests to Write:**
- `test_bundle_method_single_iteration()` - Run 1 iteration on absolute value function
- `test_bundle_method_5_iterations()` - Run 5 iterations, verify bundle grows
- `test_bundle_method_tracks_bounds()` - Verify f_up, f_low computed correctly
- `test_bundle_method_absolute_value()` - Solve min |x - 5| to optimum x* = 5
- `test_bundle_method_l1_norm_2d()` - Solve min ||x - c||_1 on 2D problem

**Steps:**
1. **Define BundleResult dataclass** (~15 LOC)
   ```python
   @dataclass
   class BundleResult:
       x_best: Vector           # Stability center (best point)
       f_best: float            # f(x_best) (upper bound)
       gap: float               # Δ_k = f_up - f_low (optimality gap)
       converged: bool          # True if gap ≤ tolerance
       num_iterations: int      # Total iterations
       num_serious_steps: int   # Serious step count
       num_null_steps: int      # Null step count
       function_values: List[float]  # f_up_k history
       iterates: Optional[List[Vector]] = None  # x_k history (if stored)
   ```

2. **Implement BundleMethod.__init__** (~20 LOC)
   ```python
   def __init__(self, oracle: NonLinearForm, *,
                alpha: float = 0.1,
                tolerance: float = 1e-6,
                max_iterations: int = 500,
                bundle_size: int = 100,
                store_iterates: bool = False):
       """
       Level bundle method for non-smooth convex optimization.

       Parameters
       ----------
       oracle : NonLinearForm
           Objective function with .subgradient(x) method
       alpha : float, default=0.1
           Level parameter ∈ (0,1) for f_lev_k computation
       tolerance : float, default=1e-6
           Gap tolerance for convergence (Δ_k ≤ tolerance)
       max_iterations : int, default=500
           Maximum number of iterations
       bundle_size : int, default=100
           Maximum cuts in bundle (compression threshold)
       store_iterates : bool, default=False
           Whether to store x_k history
       """
   ```

3. **Implement BundleMethod.solve(x0)** (~80 LOC)
   - Initialize: stability_center = x0, f_up = f(x0), f_low = -∞, bundle = empty
   - **Main loop:**
     ```python
     for k in range(max_iterations):
         # 1. Compute level: f_lev_k = alpha * f_low + (1-alpha) * f_up
         # 2. Solve QP master problem to get x_next
         # 3. Evaluate oracle: f_next = f(x_next), g_next ∈ ∂f(x_next)
         # 4. Add cut to bundle: (x_next, f_next, g_next)
         # 5. Update f_up = min(f_up, f_next)  (monotonic descent)
         # 6. Check QP feasibility: if infeasible, update f_low = f_lev_k
         # 7. Compute gap: Δ_k = f_up - f_low
         # 8. Check convergence: if Δ_k ≤ tolerance, break
     ```
   - For now: NO serious/null step distinction (Phase 3)
   - For now: NO bundle compression (Phase 5)

4. **Test on 1D absolute value:** min f(x) = |x - 5|
   - Analytical solution: x* = 5, f* = 0
   - Verify bundle method converges to x* within tolerance
   - Subgradient: g = sign(x - 5) = {-1 if x < 5, [−1,1] if x=5, +1 if x > 5}

5. **Test on 2D L1 norm:** min f(x) = ||x - c||_1 for c = (3, 7)
   - Analytical solution: x* = c, f* = 0
   - Verify convergence

6. **Run tests** - Verify all tests pass

7. **Lint/format**

**Acceptance Criteria:**
- BundleMethod iterates via QP master problem
- Bundle grows with each iteration
- f_up decreases monotonically
- Converges on toy 1D/2D problems within 50 iterations
- Gap Δ_k → 0 demonstrating convergence

---

## Phase 3: Stability Center & Serious/Null Step Logic

**Objective:** Implement stability center management with descent test to distinguish serious steps (update stability center) from null steps (add cut only).

**Files/Functions to Modify/Create:**
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Enhance `BundleMethod.solve()` with descent test
- [tests/test_bundle_method.py](tests/test_bundle_method.py) - Add step type tests

**Tests to Write:**
- `test_stability_center_updated_on_serious_step()` - Verify ˆx updated when sufficient descent
- `test_stability_center_unchanged_on_null_step()` - Verify ˆx unchanged when descent insufficient
- `test_serious_null_step_counts()` - Count step types over full solve
- `test_monotonic_best_value()` - Verify f(ˆx_k) is non-increasing
- `test_stability_prevents_oscillation()` - Compare with/without stability center on oscillatory problem

**Steps:**
1. **Add stability center tracking** (~15 LOC)
   - `stability_center = x0` (initialized at start)
   - `f_stability = f(x0)` (best value found)
   - Update only on **serious steps**

2. **Implement descent test** (~25 LOC)
   ```python
   # After evaluating f(x_next):
   delta_k = f_up_k - f_low_k
   descent_threshold = f_stability - alpha * delta_k

   if f_next < descent_threshold:
       # SERIOUS STEP: Sufficient descent achieved
       stability_center = x_next
       f_stability = f_next
       serious_steps += 1
   else:
       # NULL STEP: Insufficient descent, just add cut
       null_steps += 1
   ```

3. **Modify QP center** (~5 LOC)
   - Change QP objective from `min ||x - x_k||²` to `min ||x - stability_center||²`
   - QP always pulls toward best point found, not previous iterate

4. **Update BundleResult** - Add `num_serious_steps`, `num_null_steps`

5. **Test descent behavior:**
   - Create test problem where naive iteration would oscillate
   - Verify stability center provides robustness
   - Track serious vs null step ratio (typically ~1:3 to 1:5)

6. **Run tests** - Verify descent logic correct

7. **Lint/format**

**Acceptance Criteria:**
- Stability center ˆx_k updated only when f(x_{k+1}) < f(ˆx_k) - α·Δ_k
- Null steps add cuts without changing ˆx
- f(ˆx_k) decreases monotonically (never increases)
- Serious/null step counts tracked and reasonable (~20-40% serious)
- Stability center prevents oscillation on test problems

---

## Phase 4: Gap-Based Convergence & Termination

**Objective:** Implement robust gap-based termination criterion with optimality certificate.

**Files/Functions to Modify/Create:**
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Add gap computation and convergence checks
- [tests/test_bundle_method.py](tests/test_bundle_method.py) - Add convergence tests

**Tests to Write:**
- `test_gap_convergence_to_zero()` - Verify Δ_k → 0 on convex problem
- `test_gap_certificate_abs_value()` - For min |x-5|, verify |f(ˆx) - 0| ≤ gap when converged
- `test_gap_certificate_l1_norm()` - For min ||x-c||_1, verify optimality certificate
- `test_convergence_flag_true_when_gap_small()` - Verify result.converged = True
- `test_no_convergence_when_gap_large()` - Verify result.converged = False if max_iter reached
- `test_gap_tolerance_parameter()` - Test different tolerance values (1e-3, 1e-6, 1e-9)

**Steps:**
1. **Robust gap computation** (~15 LOC)
   ```python
   # Track f_up_k = min{f(x_j) : j ∈ bundle} (best value found)
   f_up_k = min(bundle.best_value, f_stability)

   # Track f_low_k (lower bound from QP feasibility)
   # Initially f_low_0 = -∞
   # When QP infeasible at level f_lev, update: f_low = f_lev
   if qp_infeasible:
       f_low_k = f_lev_k

   # Gap (optimality certificate)
   gap_k = f_up_k - f_low_k
   ```

2. **Convergence check** (~10 LOC)
   ```python
   if gap_k <= tolerance:
       converged = True
       break  # Optimal within tolerance

   # Also check stagnation as backup
   if stagnation_window and no_improvement_for_N_iterations:
       converged = False
       break  # Stagnated without convergence
   ```

3. **Add gap history tracking** (~5 LOC)
   - Store `gaps: List[float] = []` during iteration
   - Useful for convergence plots and diagnostics

4. **Test optimality certificate:**
   - For problems with known optimum f*, verify: |f(ˆx_k) - f*| ≤ gap_k
   - This is the KEY property: gap provides rigorous optimality bound

5. **Test with varying tolerances:**
   - tolerance = 1e-3: Should converge quickly (~20 iterations)
   - tolerance = 1e-6: Moderate iterations (~50-100)
   - tolerance = 1e-9: Many iterations (~200+) but high accuracy

6. **Run tests** - Verify gap-based termination correct

7. **Lint/format**

**Acceptance Criteria:**
- Gap Δ_k = f_up_k - f_low_k computed correctly each iteration
- Convergence terminates when Δ_k ≤ tolerance
- Optimality certificate validated: |f(ˆx_k) - f*| ≤ Δ_k on test problems
- result.converged flag accurate
- Smaller tolerance → more iterations but higher accuracy

---

## Phase 5: Bundle Compression

**Objective:** Implement bundle size management to prevent unbounded memory growth and maintain computational efficiency.

**Files/Functions to Modify/Create:**
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Add `Bundle.compress()` method
- [tests/test_bundle_method.py](tests/test_bundle_method.py) - Add compression tests

**Tests to Write:**
- `test_bundle_compression_preserves_active_cuts()` - Active cuts (tight at ˆx) kept
- `test_bundle_compression_removes_old_cuts()` - Oldest inactive cuts removed
- `test_bundle_size_bounded()` - Run 1000 iterations, verify |bundle| ≤ max_size
- `test_convergence_not_degraded_by_compression()` - Compare with/without compression
- `test_compression_strategy_aggregate()` - Test aggregation of cuts (optional)

**Steps:**
1. **Identify active cuts** (~20 LOC)
   ```python
   def get_active_cuts(self, x_hat, tolerance=1e-8):
       """Return cuts that are nearly active at x_hat.

       Cut j is active if: f(x_j) + ⟨g_j, x_hat - x_j⟩ ≈ f(x_hat)
       i.e., the linearization is tight at the stability center.
       """
       active = []
       f_hat = self.best_value
       for cut in self.cuts:
           linearization = cut.f_x + self.domain.inner_product(
               cut.g, self.domain.subtract(x_hat, cut.x)
           )
           if abs(linearization - f_hat) < tolerance:
               active.append(cut)
       return active
   ```

2. **Implement compression strategy** (~30 LOC)
   ```python
   def compress(self, stability_center, max_size):
       """Reduce bundle size while preserving convergence.

       Strategy:
       1. Keep all active cuts (always!)
       2. Keep most recent cuts (warm information)
       3. Remove oldest inactive cuts until |bundle| ≤ max_size
       """
       if len(self.cuts) <= max_size:
           return  # No compression needed

       active_cuts = self.get_active_cuts(stability_center)
       recent_cuts = sorted(self.cuts, key=lambda c: c.iteration, reverse=True)[:max_size//2]

       # Keep union of active and recent cuts
       keep = set(active_cuts) | set(recent_cuts)
       self.cuts = sorted(list(keep), key=lambda c: c.iteration)

       # Ensure we're under limit (remove oldest if needed)
       while len(self.cuts) > max_size:
           # Remove oldest non-active cut
           for i, cut in enumerate(self.cuts):
               if cut not in active_cuts:
                   self.cuts.pop(i)
                   break
   ```

3. **Trigger compression** (~5 LOC)
   ```python
   # In BundleMethod.solve(), after adding cut:
   if len(bundle) > bundle_size:
       bundle.compress(stability_center, bundle_size)
   ```

4. **Test long runs:**
   - Run 1000-5000 iterations on a problem
   - Verify bundle never exceeds max_size
   - Verify convergence rate not significantly degraded

5. **Compare with/without compression:**
   - Run same problem with bundle_size = ∞ (no compression) vs bundle_size = 50
   - Should achieve similar final accuracy, similar iteration counts

6. **Run tests** - Verify compression works correctly

7. **Lint/format**

**Acceptance Criteria:**
- Bundle size bounded: |J_k| ≤ max_size even after 1000+ iterations
- Active cuts (tight at ˆx_k) always preserved
- Convergence not degraded (iteration count within 10% of uncompressed)
- Memory usage constant after compression kicks in

---

## Phase 6: Integration with DualMasterCostFunction

**Objective:** Integrate bundle methods with DualMasterCostFunction for solving dual master problems and compare performance against SubgradientDescent.

**Files/Functions to Modify/Create:**
- [tests/test_bundle_dual_master.py](tests/test_bundle_dual_master.py) - New test file for integration
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Minor tweaks for NonLinearForm compatibility

**Tests to Write:**
- `test_bundle_with_dual_master_simple()` - Solve toy dual master problem
- `test_bundle_vs_subgradient_accuracy()` - Compare final gap/accuracy
- `test_bundle_vs_subgradient_iterations()` - Compare iteration counts
- `test_bundle_dual_master_support_value()` - Compute h_U(q) and verify bounds
- `test_bundle_with_ball_constraint()` - Use BallSupportFunction in dual master
- `test_bundle_with_ellipsoid_constraint()` - Use EllipsoidSupportFunction in dual master

**Steps:**
1. **Create dual master test problem** (~30 LOC)
   ```python
   # Backus-Gilbert setup with known solution
   model_space = EuclideanSpace(dim=10)
   data_space = EuclideanSpace(dim=5)
   property_space = EuclideanSpace(dim=3)

   G = # Forward operator (design)
   T = # Property operator
   d_obs = # Synthetic data

   B = Ball(model_space, center=0, radius=5.0)
   V = Ball(data_space, center=0, radius=0.1)

   cost = DualMasterCostFunction(G, T, d_obs, B.support_function, V.support_function)
   ```

2. **Test bundle method on dual master** (~20 LOC)
   ```python
   cost.set_direction(q)  # Direction in property space

   # Solve with bundle method
   bundle_solver = BundleMethod(cost, alpha=0.1, tolerance=1e-6)
   bundle_result = bundle_solver.solve(lambda_init)

   # Solve with subgradient descent
   sg_solver = SubgradientDescent(cost, step_size=0.01, max_iterations=5000)
   sg_result = sg_solver.solve(lambda_init)

   # Compare
   assert bundle_result.gap < 1e-6        # Bundle achieves target tolerance
   assert bundle_result.gap < sg_result.gap * 10  # Bundle 10x more accurate
   ```

3. **Performance comparison** (~30 LOC)
   - Measure iterations: bundle should converge in fewer iterations
   - Measure accuracy: bundle should achieve smaller gap
   - Measure oracle calls: bundle may use more (QP overhead), but each call better utilized
   - Expected: Bundle ~5-10x more accurate in ~2-3x fewer iterations

4. **Verify support value computation:**
   - h_U(q) = inf_λ φ(λ; q) should match known bounds
   - For simple problems, can compute h_U analytically
   - Verify bundle_result.f_best ≈ h_U_analytical(q)

5. **Test multiple support functions:**
   - Ball, Ellipsoid, Box constraints
   - Verify bundle method handles different SupportFunction implementations

6. **Run tests** - Verify integration successful

7. **Lint/format**

**Acceptance Criteria:**
- BundleMethod successfully minimizes DualMasterCostFunction
- Achieves gap < 1e-6 (typical target accuracy)
- Outperforms SubgradientDescent in both iterations and accuracy
- Compatible with all existing SupportFunction implementations
- Support values h_U(q) computed accurately

---

## Phase 7: Documentation, Polishing & Advanced Features

**Objective:** Complete implementation with comprehensive documentation, usage examples, convergence diagnostics, and optional advanced features.

**Files/Functions to Modify/Create:**
- [pygeoinf/convex_optimisation.py](pygeoinf/convex_optimisation.py) - Add docstrings, examples
- [pygeoinf/docs/theory_map.md](pygeoinf/docs/theory_map.md) - Update status from 🔲 Future to ✅ Complete
- [pygeoinf/docs/theory_papers_index.md](pygeoinf/docs/theory_papers_index.md) - Update code status
- [tests/test_bundle_method.py](tests/test_bundle_method.py) - Add doctests
- [pygeoinf/plot.py](pygeoinf/plot.py) - Add `plot_bundle_convergence()` utility (optional)

**Tests to Write:**
- `test_docstring_examples_run()` - Run all docstring examples
- `test_bundle_convergence_plot()` - Generate convergence diagnostic plot
- `test_warm_start()` - Test warm starting from previous solution (optional)
- `test_bundle_with_constraints()` - Test with feasible set X constraints (optional)

**Steps:**
1. **Write comprehensive docstrings** (~100 LOC)
   ```python
   class BundleMethod:
       """Level bundle method for non-smooth convex optimization.

       Solves min_{x ∈ X} f(x) where f is convex but non-differentiable,
       using cutting-plane models and quadratic master problems.

       Algorithm
       ---------
       1. Build cutting-plane model: ˇf_k(x) = max_j {f(x_j) + ⟨g_j, x-x_j⟩}
       2. Solve QP: x_{k+1} = argmin_{x : ˇf_k(x) ≤ f_lev_k} ||x - ˆx_k||²
       3. Descent test: f(x_{k+1}) < f(ˆx_k) - α·Δ_k → serious step (update ˆx)
       4. Terminate when gap Δ_k = f_up_k - f_low_k ≤ tolerance

       Advantages over subgradient methods:
       - Automatic step sizing (no manual tuning)
       - Model-based search directions (better descent)
       - Reliable gap-based termination (certificate of optimality)
       - Stability center prevents oscillation

       Parameters
       ----------
       oracle : NonLinearForm
           Objective function with .subgradient(x) method.
       alpha : float, default=0.1
           Level parameter ∈ (0,1). Smaller α → more null steps, more stable.
           Typical values: 0.1 (aggressive), 0.5 (balanced), 0.9 (conservative).
       tolerance : float, default=1e-6
           Gap tolerance for convergence. When Δ_k ≤ tolerance,
           guarantees |f(ˆx_k) - f*| ≤ tolerance.
       max_iterations : int, default=500
           Maximum number of iterations.
       bundle_size : int, default=100
           Maximum cuts in bundle before compression.
       store_iterates : bool, default=False
           If True, store full iterate history (memory intensive).

       Returns
       -------
       result : BundleResult
           Optimization result with fields:
           - x_best : Vector (stability center, best point found)
           - f_best : float (best value f(x_best))
           - gap : float (optimality gap Δ_k)
           - converged : bool (True if gap ≤ tolerance)
           - num_iterations : int
           - num_serious_steps, num_null_steps : int

       References
       ----------
       - Lemaréchal (1975): "An extension of Davidon methods to non-differentiable problems"
       - Kiwiel (1995): "Proximal level bundle methods for convex nondifferentiable optimization"
       - theory/bundle_methods.pdf: Asynchronous level bundle methods

       Examples
       --------
       Minimize L1 norm:

       >>> from pygeoinf import EuclideanSpace, BundleMethod
       >>> space = EuclideanSpace(dim=5)
       >>> def l1_norm(x): return space.norm(x, ord=1)
       >>> def l1_subgradient(x): return np.sign(x)  # Subdifferential at x≠0
       >>>
       >>> oracle = ... # NonLinearForm wrapper
       >>> solver = BundleMethod(oracle, alpha=0.1, tolerance=1e-6)
       >>> result = solver.solve(x_init)
       >>> print(f"Optimal value: {result.f_best:.6f}, gap: {result.gap:.2e}")
       Optimal value: 0.000000, gap: 9.87e-07

       Minimize DualMasterCostFunction:

       >>> cost = DualMasterCostFunction(G, T, d_obs, sigma_B, sigma_V)
       >>> cost.set_direction(q)
       >>> solver = BundleMethod(cost, alpha=0.1, tolerance=1e-6)
       >>> result = solver.solve(lambda_init)
       >>> h_U_q = result.f_best  # Support value with certificate |h_U_q - h*| ≤ 1e-6

       See Also
       --------
       SubgradientDescent : Simpler method with manual step sizing
       DualMasterCostFunction : Primary use case (convex dual problems)
       """
   ```

2. **Add usage examples to theory_map.md** (~20 LOC)
   - Update Bundle Methods section with complete working example
   - Show comparison with SubgradientDescent

3. **Update theory_papers_index.md** (~10 LOC)
   - Change code status from 🔲 Future to ✅ Complete
   - Update references to point to implementation

4. **Add convergence diagnostic plot** (~40 LOC, optional)
   ```python
   def plot_bundle_convergence(result: BundleResult, ax=None):
       """Plot convergence diagnostics for bundle method.

       Shows:
       - f_up_k (best value) over iterations
       - Gap Δ_k over iterations (log scale)
       - Serious vs null step markers
       """
   ```

5. **Add warm start capability** (~20 LOC, optional)
   ```python
   def solve(self, x0, warm_start_bundle=None):
       """Solve with optional warm start from previous bundle."""
       if warm_start_bundle:
           bundle = copy.deepcopy(warm_start_bundle)
       else:
           bundle = Bundle(self.bundle_size)
       # ... continue as normal
   ```

6. **Test all docstring examples** - Ensure doctests pass

7. **Run full test suite** - All 35+ tests pass

8. **Lint/format** - Final cleanup

**Acceptance Criteria:**
- All classes/methods have comprehensive Google-style docstrings with LaTeX
- Docstring examples run and produce expected output
- theory_map.md and theory_papers_index.md updated with ✅ Complete status
- Optional: Convergence plot utility available
- Optional: Warm start functionality working
- All tests pass (35+ tests covering all features)
- Code formatted and linted
- Ready for production use in dual master problems

---

## Open Questions

1. **QP Solver Choice:** Start with scipy.optimize.minimize(method='SLSQP') or add cvxpy dependency?
   - **Option A (scipy):** No new dependencies, but SLSQP may be slow for large bundles
   - **Option B (cvxpy):** Specialized QP solver (faster), but requires cvxpy + solver backend
   - **Recommended:** Start with scipy (Option A), add cvxpy as optional enhancement if needed

2. **Bundle Compression Strategy:** Simple (keep recent + active) or advanced (aggregation)?
   - **Option A (Simple):** Keep N most recent cuts + active cuts at ˆx_k
   - **Option B (Aggregate):** Aggregate old cuts into single averaged cut
   - **Recommended:** Start with Option A (simpler, proven), add Option B in Phase 7 if beneficial

3. **Feasible Set Constraints:** Implement general X constraints or defer?
   - **Option A (Defer):** Start with X = full space (no constraints), add later if needed
   - **Option B (Include):** Add X constraints to QP from start (more general)
   - **Recommended:** Option A (defer) - most dual master problems have X = R^n (no box constraints)

4. **Performance Target:** How much faster/more accurate than SubgradientDescent?
   - **Target:** 5-10x smaller gap in 2-3x fewer iterations on dual master problems
   - **Metric:** For tolerance=1e-6, bundle should converge in ~50-100 iterations vs subgradient ~500+
   - **Tradeoff:** More oracle calls per iteration (QP overhead), but better utilization

5. **Asynchronous Extensions:** Implement parallel oracle calls (from bundle_methods.pdf)?
   - **Option A (Defer):** Serial implementation only (simpler, sufficient for most problems)
   - **Option B (Include):** Add async oracle interface for parallel subgradient evaluations
   - **Recommended:** Option A (defer) - synchronous is sufficient, async is research feature
