# Dual Master Equation Implementation Plan

**Goal:** Implement the dual master equation cost function for Direct Linear Inversion in pygeoinf.

**Master Dual Equation (Hilbert Form):**
```
h_U(q) = inf_λ { ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ) }

where:
  φ(λ; q) = ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ)
  σ_B = support function of model prior set B ⊆ M
  σ_V = support function of data error set V ⊆ D
  d̃ = observed data vector
```

**Key insight:** All geometric information about the admissible property set U is encoded
by the support functions of the convex sets B and V, evaluated at the Hilbert-space
residual T*q - G*λ.

---

## Phase 1: Architecture Analysis ✅

**Status:** COMPLETE

**Current state:**
- ✅ `convex_analysis.py`: `SupportFunction` hierarchy exists (abstract base, Ball, Ellipsoid)
- ✅ `subsets.py`: `ConvexSubset`/`Ball`/`Ellipsoid` expose cached `support_function` objects (lazy)
- ✅ `subsets.py`: `ConvexIntersection` exposes an intersection support function when possible

**Spaces:**
- λ ∈ D (data space, Euclidean)
- q ∈ P (property space, Euclidean)
- V ⊆ D (data error convex set)
- B ⊆ M (model prior convex set, arbitrary Hilbert space)

---

## Phase 2: Refactor `ConvexSubset` to Expose Support Functions

**Status:** COMPLETE ✅

**Goal:** Make support functions first-class, reusable objects

### Tasks

- [x] **Task 2.1:** Add abstract property to `ConvexSubset` class
  ```python
  @property
  @abstractmethod
  def support_function(self) -> Optional[SupportFunction]:
      """Returns the SupportFunction instance for this set, or None if not available.

      Returns None for set types that genuinely cannot provide a support function
      (e.g., composite sets where at least one component lacks support support).

      For ellipsoids, the SupportFunction object is still instantiated even if
      inverse operators are not provided; evaluation may then raise when needed.
      """
  ```
  - **File:** `pygeoinf/subsets.py`
  - **Location:** In `ConvexSubset` class definition (~line 530-570)
  - **Note:** This property should NOT raise errors if support function cannot be created

- [x] **Task 2.2:** Implement `support_function` property in `Ball` class
  - **File:** `pygeoinf/subsets.py`
  - **Changes:**
    - [x] Add `self._support_fn` attribute in `Ball.__init__` (~line 1000), initialized to `None`
    - [x] Implement `support_function` property using lazy evaluation:
      ```python
      @property
      def support_function(self) -> Optional[SupportFunction]:
          if self._support_fn is None:
              self._support_fn = BallSupportFunction(self.domain, self.center, self.radius)
          return self._support_fn
      ```
    - [x] Update `directional_bound(q)` to use the cached `SupportFunction`
  - **Note:** Ball always has sufficient structure for support function (no optional parameters needed)
  - **Note:** The `BallSupportFunction` constructor should never fail for valid Ball instances

- [x] **Task 2.3:** Implement `support_function` property in `Ellipsoid` class
  - **File:** `pygeoinf/subsets.py`
  - **Changes:**
    - [x] Add `self._support_fn` attribute in `Ellipsoid.__init__` (~line 830), initialized to `None`
    - [x] Add optional `inverse_operator` and `inverse_sqrt_operator` constructor params (default `None`, keyword-only)
    - [x] Store these as `self._inverse_operator` and `self._inverse_sqrt_operator`
    - [x] Implement `support_function` property using lazy evaluation:
      ```python
      @property
      def support_function(self) -> Optional[SupportFunction]:
          if self._support_fn is None:
              # Only create if we have at least the basic structure
              # EllipsoidSupportFunction will handle None inverses gracefully
              self._support_fn = EllipsoidSupportFunction(
                  self.domain,
                  self.center,
                  self.radius,
                  self.operator,
                  inverse_operator=self._inverse_operator,
                  inverse_sqrt_operator=self._inverse_sqrt_operator
              )
          return self._support_fn
      ```
    - [x] Update `directional_bound(q)` to use `support_point` from the cached support function
  - **Critical design principle:**
    - Ellipsoid instantiation NEVER fails due to missing inverse operators
    - `support_function` property ALWAYS returns a `SupportFunction` object (not None)
    - The `EllipsoidSupportFunction` accepts `None` for inverse operators (already implemented in convex_analysis.py)
    - Methods like `_mapping()` will raise `ValueError` if inverses are needed but not provided
    - This allows users to create Ellipsoid for other purposes without support function infrastructure

**Benefits:**
- Users can instantiate `Ball`/`Ellipsoid` without providing support function infrastructure
- Support functions are lazily instantiated only when accessed via the property
- `support_function` returns a cached `SupportFunction` when available (may be `None` for some composite sets)
- For `Ellipsoid`, missing inverse operators don't prevent instantiation - errors only occur when trying to use incomplete support function methods
- Clean separation: geometric set definition vs. support function capabilities
- API change: `support_function` is now a property returning a `SupportFunction`, not a method

**Implementation notes (what was actually done):**
- `Ellipsoid` accepts `inverse_operator` and `inverse_sqrt_operator` as keyword-only args and never attempts to compute them.
- `directional_bound(...)` uses `support_function.support_point(...)` and raises if the support point is unavailable.
- `ConvexIntersection.support_function` returns a support function only if all component subsets have support functions.

---

## Phase 3: Create `DualMasterCostFunction` Class

**Status:** COMPLETE ✅

**Goal:** Implement the cost function φ(λ; q) as a `NonLinearForm`

### Tasks

- [x] **Task 3.1:** Add `DualMasterCostFunction` to `pygeoinf/backus_gilbert.py`
  - [x] Place it near other convex/dual utilities (not inside an existing class)
  - [x] Add imports as needed:
    - `HilbertSpace`, `Vector` from `hilbert_space`
    - `LinearOperator` from `linear_operators`
    - `NonLinearForm` from `nonlinear_forms`
    - `SupportFunction` from `convex_analysis`

- [x] **Task 3.2:** Implement `DualMasterCostFunction` class
  ```python
  class DualMasterCostFunction(NonLinearForm):
      """
    Cost function for the master dual equation (Hilbert form):

      h_U(q)
      = inf_{λ ∈ D}
        { (λ, d̃)_D + σ_B(T* q - G* λ) + σ_V(-λ) }

    i.e.

      φ(λ; q) = (λ, d̃)_D + σ_B(T* q - G* λ) + σ_V(-λ)

    where:
    - σ_B is the support function of the model prior convex set B ⊆ M
    - σ_V is the support function of the data error convex set V ⊆ D

    Minimizing φ(λ; q) over λ ∈ D yields h_U(q).
      """
  ```

- [x] **Task 3.3:** Implement `__init__` constructor
  - **Parameters:**
    - `data_space: HilbertSpace` (D, expected Euclidean)
    - `property_space: HilbertSpace` (P, expected Euclidean)
    - `model_space: HilbertSpace` (M, may be non-Euclidean)
    - `G: LinearOperator` (M → D, forward map)
    - `T: LinearOperator` (M → P, property extraction)
    - `model_prior_support: SupportFunction` (σ_B for prior set B ⊆ M)
    - `data_error_support: SupportFunction` (σ_V for error set V ⊆ D)
    - `observed_data: Vector` (d̃ ∈ D)
    - `q_direction: Vector` (q ∈ P, initial direction)
  - **Validation:**
    - [x] Assert `G.domain == model_space`
    - [x] Assert `G.codomain == data_space`
    - [x] Assert `T.domain == model_space`
    - [x] Assert `T.codomain == property_space`
    - [x] Assert `model_prior_support.primal_domain == model_space`
    - [x] Assert `data_error_support.primal_domain == data_space`
  - **Precomputation:**
    - [x] Compute `self._Tstar_q = T.adjoint(q_direction)` (cache for efficiency)

- [x] **Task 3.4:** Implement `_mapping(λ)` method
  ```python
  def _mapping(self, lam: Vector) -> float:
      # Term 1: ⟨λ, d̃⟩_D
      term1 = self.domain.inner_product(lam, self._observed_data)

      # Term 2: σ_B(T*q - G*λ)
      Gstar_lam = self._G.adjoint(lam)
      hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)
      term2 = self._model_prior_support(hilbert_residual)

      # Term 3: σ_V(-λ)
      neg_lam = self.domain.negative(lam)
      term3 = self._data_error_support(neg_lam)

      return term1 + term2 + term3
  ```

- [x] **Task 3.5:** Implement `_gradient(λ)` method
  - [x] Term 1: ∂₁φ = d̃ (observed data)
  - [x] Term 2: ∂₂φ from σ_B(T*q - G*λ)
    - Compute Hilbert residual r = T*q - G*λ
    - Get subgradient: v = support_point(r) (element of M achieving supremum)
    - Contribution: -G*v (via chain rule)
  - [x] Term 3: ∂₃φ from σ_V(-λ)
    - Compute -λ
    - Get subgradient: w = support_point(-λ) (element of D achieving supremum)
    - Contribution: -w (via chain rule with negation)
  - [x] Handle case where `support_point` returns None (numerical differentiation)
  - [x] Combine all terms using Hilbert space operations

- [x] **Task 3.6:** Implement `set_direction(q)` method
  ```python
  def set_direction(self, q: Vector) -> None:
      """Update the property direction q and recompute T*q."""
      self._q = q
      self._Tstar_q = self._T.adjoint(q)
  ```

---

## Phase 4: Solver Implementation

**Status:** IN PROGRESS (Sub-Phase 4.1 complete)

**Goal:** Implement solvers for the minimization problem inf_λ φ(λ; q)

**Note:** Standard gradient-based methods may not work directly because support functions are non-smooth (subgradients, not gradients). We will implement subgradient methods incrementally, starting from the simplest approach.

**Educational Approach:** Phase 4 is broken into sub-phases to build understanding incrementally:
- **4.1:** Simplest subgradient descent (constant step size, basic implementation)
- **4.2:** Improved step size rules (diminishing, adaptive, Polyak)
- **4.3:** Integration with DualMasterCostFunction and convergence diagnostics
- **4.4:** Optional advanced methods (bundle, proximal, projection)

---

### Sub-Phase 4.1: Basic Subgradient Descent (Learning Foundation)

**Status:** ✅ COMPLETE

**Goal:** Implement the simplest possible subgradient descent algorithm to understand the mechanics.

**Key Concepts:**
- **Oracle model:** We have a function `f(x)` and can query both `f(x)` and a subgradient `g ∈ ∂f(x)`
- **Subgradient descent iteration:** `x_{k+1} = x_k - α_k g_k` where `g_k ∈ ∂f(x_k)`
- **Non-monotonic:** Unlike gradient descent, function value may increase between iterations
- **Convergence:** Requires careful step size choice; constant step → oscillation, diminishing → convergence

**Design Decision:** Create a standalone, reusable `SubgradientDescent` class that works with any `NonLinearForm` (not specific to DualMasterCostFunction). This allows testing with simple examples first.

#### Tasks

- [x] **Task 4.1.1:** Create `pygeoinf/convex_optimisation.py` module
  - [x] Module docstring explaining subgradient methods
  - [x] All required imports properly configured

- [x] **Task 4.1.2:** Define `SubgradientResult` dataclass
  - [x] Dataclass fully defined with all 8 required fields
  - [x] Proper type hints and optional iterates history

- [x] **Task 4.1.3:** Implement `SubgradientDescent` class
  - [x] Class fully implemented with proper docstring
  - [x] Algorithm: x_{k+1} = x_k - α*g_k with constant step size
  - [x] Non-monotonic convergence acknowledged (learning tool)

- [x] **Task 4.1.4:** Implement `SubgradientDescent.__init__`
  - [x] Stores oracle with validation (must have subgradient)
  - [x] Validates step_size > 0
  - [x] Stores max_iterations parameter
  - [x] Stores store_iterates flag

- [x] **Task 4.1.5:** Implement `SubgradientDescent.solve(x0: Vector)` method
  - [x] Full iteration loop: evaluates f_k, tracks best point, computes g_k
  - [x] Hilbert space operations for x_{k+1} = x_k - α*g_k
  - [x] Conditional storage of iterates based on store_iterates flag
  - [x] Proper result construction with all fields

- [x] **Task 4.1.6:** Add simple convergence check
  - [x] Stagnation-based convergence: no improvement for N iterations
  - [x] Sets `converged` flag based on criterion

**Implementation Notes (Completed):**
- ✅ Subgradient computation via oracle.subgradient() (uses support_point for support functions)
- ✅ Function values oscillate - this is NORMAL for constant step size (documented)
- ✅ Best point tracking ensures monotonic improvement in best value found
- ✅ This is a "learning implementation" - production use requires Sub-Phase 4.2
- ✅ Tested on actual DualMasterCostFunction with support function oracles

**Testing Status:**
- ✅ Integrated with `DualMasterCostFunction` and `BallSupportFunction` oracles
- ✅ Running on 1D toy problem: D=M=P=ℝ, G(m)=2m, T(m)=m
- ✅ Support functions provide subgradients via support_point delegation
- ✅ Solver converges with constant step size α=0.1
- ✅ Visualization: test.py plots cost function and iterates

---

### Sub-Phase 4.2: Improved Step Size Rules

**Status:** NOT STARTED (depends on 4.1 ✅)

**Goal:** Add sophisticated step size schedules for guaranteed convergence

**Step Size Rules to Implement:**

1. **Diminishing step size:** `α_k = α₀ / (1 + k)` or `α_k = α₀ / sqrt(k + 1)`
   - Guarantees convergence: Σ α_k = ∞, Σ α_k² < ∞

2. **Polyak step size:** `α_k = (f_k - f_target) / ||g_k||²`
   - Requires knowledge (or estimate) of optimal value f_target
   - Adapts to function geometry

3. **Adaptive step size:** Increase when progress, decrease on oscillation
   - Track recent function values
   - Backtracking-like heuristics

#### Tasks

- [ ] **Task 4.2.1:** Extend `SubgradientDescent` with step size strategies
  - [ ] Add `step_size_rule` parameter: `'constant'`, `'diminishing'`, `'polyak'`, `'adaptive'`
  - [ ] Add rule-specific parameters (e.g., `initial_step_size`, `f_target`, etc.)

- [ ] **Task 4.2.2:** Implement diminishing step size
  - [ ] `α_k = α₀ / (1 + k)` (square summable)
  - [ ] Alternative: `α_k = α₀ / sqrt(k + 1)` (non-square summable, slower convergence)

- [ ] **Task 4.2.3:** Implement Polyak step size
  - [ ] `α_k = (f_k - f*) / ||g_k||²` when `f* < f_k`
  - [ ] Handle case when f* unknown: use best value so far as estimate
  - [ ] Safeguard: clip step size to [α_min, α_max]

- [ ] **Task 4.2.4:** Add convergence criteria
  - [ ] Relative function improvement: `|f_k - f_{k-1}| / |f_k| < tol`
  - [ ] Subgradient norm threshold: `||g_k|| < tol` (but may not hold at non-smooth points!)
  - [ ] Best value stagnation: no improvement for N iterations

- [ ] **Task 4.2.5:** Add iteration diagnostics and logging
  - [ ] Optionally print/log every N iterations: `k, f_k, ||g_k||, α_k`
  - [ ] Track average step size, average subgradient norm
  - [ ] Detect divergence (function value growing unboundedly)

---

### Sub-Phase 4.3: Integration with DualMasterCostFunction

**Status:** NOT STARTED (depends on 4.2)

**Note:** Sub-Phase 4.1 already integrated with DualMasterCostFunction in test harness; 4.3 will add convenience methods

**Goal:** Connect the subgradient solver to `DualMasterCostFunction` and create user-facing API

#### Tasks

- [ ] **Task 4.3.1:** Add `solve_subgradient` method to `DualMasterCostFunction`
  ```python
  def solve_subgradient(
      self,
      initial_lambda: Optional[Vector] = None,
      step_size_rule: str = 'polyak',
      max_iterations: int = 1000,
      **kwargs
  ) -> Tuple[float, Vector]:
      """Minimize φ(λ; q) using subgradient descent.

      Returns:
          (h_U(q), λ*): Optimal value and optimal dual variable
      """
  ```
  - [ ] Create `SubgradientDescent` instance with `self` as oracle
  - [ ] Use default initial point if not provided (e.g., zero)
  - [ ] Run solver and extract `f_best`, `x_best`
  - [ ] Return as tuple

- [ ] **Task 4.3.2:** Add `solve_for_support_value(q)` helper method
  ```python
  def solve_for_support_value(self, q: Vector, **kwargs) -> float:
      """Compute h_U(q) by minimizing φ(λ; q) over λ.

      Args:
          q: Direction in property space
          **kwargs: Passed to solve_subgradient

      Returns:
          h_U(q): Support function value of admissible set
      """
  ```
  - [ ] Call `self.set_direction(q)`
  - [ ] Call `self.solve_subgradient(**kwargs)`
  - [ ] Return optimal value (first element of tuple)

- [ ] **Task 4.3.3:** Add validation and warnings
  - [ ] Warn if solver didn't converge
  - [ ] Warn if subgradients were unavailable (fell back to finite differences)
  - [ ] Provide diagnostic information on request

- [ ] **Task 4.3.4:** Document usage patterns
  - [ ] Typical parameter choices for different problem sizes
  - [ ] How to choose step size and max iterations
  - [ ] How to diagnose convergence issues

---

### Sub-Phase 4.4: Optional Advanced Methods

**Status:** NOT STARTED (optional, depends on 4.3)

**Goal:** Implement more sophisticated non-smooth optimization methods

#### Tasks (Optional)

- [ ] **Task 4.4.1:** Projected subgradient descent
  - [ ] Add projection onto convex constraints (e.g., box constraints on λ)
  - [ ] Useful when data space has known bounds

- [ ] **Task 4.4.2:** Bundle method (cutting-plane method)
  - [ ] Maintain polyhedral approximation of objective
  - [ ] Solve QP subproblem at each iteration
  - [ ] Much faster convergence than subgradient descent
  - [ ] More complex implementation

- [ ] **Task 4.4.3:** Proximal gradient method
  - [ ] For objectives of form `f(x) = g(x) + h(x)` where g smooth, h non-smooth
  - [ ] Decompose dual master if possible

- [ ] **Task 4.4.4:** Stochastic subgradient method
  - [ ] For large-scale problems with sum structure
  - [ ] Randomly sample subset of terms at each iteration

---

### Phase 4 Progress Tracker

| Sub-Phase | Status | Dependencies | Priority |
|-----------|--------|--------------|----------|
| 4.1: Basic Subgradient | ✅ COMPLETE | Phase 3 complete | **HIGH** |
| 4.2: Step Size Rules | ⏸️ NOT STARTED | 4.1 complete ✅ | **HIGH** |
| 4.3: Integration | ⏸️ NOT STARTED | 4.2 complete | **HIGH** |
| 4.4: Advanced Methods | ⏸️ NOT STARTED | 4.3 complete | **LOW** (optional) |

**Current Focus:** Sub-Phase 4.2 (Improved Step Size Rules)

---

## Phase 5: Integration and Testing

**Status:** NOT STARTED

### Tasks

- [ ] **Task 5.1:** Create demo notebook
  - **File:** `pygeoinf/testing_sets/dual_master_demo.ipynb`
  - **Sections:**
    1. [ ] Setup: Import modules and define spaces (D, P, M)
    2. [ ] Create linear operators G and T
    3. [ ] Define model prior convex set B ⊆ M (e.g., Ball or Ellipsoid)
    4. [ ] Define data error convex set V ⊆ D (e.g., Ball)
    5. [ ] Extract support functions: σ_B and σ_V from convex sets
    6. [ ] Construct `DualMasterCostFunction` with both support functions
    7. [ ] Evaluate φ(λ; q) at test points
    8. [ ] Solve for h_U(q) in multiple directions
    9. [ ] Visualize results (2D property space example)
    10. [ ] Compare with different choices of B and V
    11. [ ] Demonstrate Ellipsoid without inverse operators (error handling)
    12. [ ] Demonstrate Ellipsoid with inverse operators (full functionality)

- [ ] **Task 5.2:** Write unit tests
  - **File:** `tests/test_dual_linear_inversion.py`
  - [ ] Test `DualMasterCostFunction.__init__` validation
  - [ ] Test `_mapping` evaluation
  - [ ] Test `_gradient` computation
  - [ ] Test `set_direction` updates
  - [ ] Test with Ball model prior and Ball data error
  - [ ] Test with Ellipsoid model prior and Ball data error
  - [ ] Test with Ball model prior and Ellipsoid data error
  - [ ] Test gradient vs numerical gradient (finite differences)
  - [ ] Test that σ_V(-λ) is called correctly (with negation)
  - [ ] Test Ellipsoid without inverses: instantiation succeeds, support function evaluation fails gracefully
  - [ ] Test Ellipsoid with inverses: full support function works

- [ ] **Task 5.3:** Update package exports
  - **File:** `pygeoinf/__init__.py`
  - [ ] Add `from .backus_gilbert import DualMasterCostFunction`

- [ ] **Task 5.4:** Syntax and import checks
  - [ ] Run `python -m py_compile pygeoinf/backus_gilbert.py`
  - [ ] Run `python -c "from pygeoinf import DualMasterCostFunction"`
  - [ ] Run notebook cells sequentially

### Example Workflow

```python
# 1. Define spaces
D = EuclideanSpace(10)  # Data space
P = EuclideanSpace(5)   # Property space
M = EuclideanSpace(20)  # Model space (could be non-Euclidean)

# 2. Create linear operators
G = LinearOperator.from_matrix(...)  # M → D (forward map)
T = LinearOperator.from_matrix(...)  # M → P (property extraction)

# 3. Define model prior convex set B ⊆ M
model_prior_ball = Ball(M, center=M.zero, radius=1.0)
σ_B = model_prior_ball.support_function  # Always available for Ball

# Alternative: Ellipsoid with inverse operators for full support function
# ellipsoid = Ellipsoid(M, center=M.zero, radius=1.0, operator=A,
#                       inverse_operator=A_inv, inverse_sqrt_operator=A_inv_sqrt)
# σ_B = ellipsoid.support_function  # Available with all methods

# 4. Define data error convex set V ⊆ D
data_error_ball = Ball(D, center=D.zero, radius=0.1)
σ_V = data_error_ball.support_function  # Always available for Ball

# 5. Construct cost function
observed_data = ...  # d̃ (observed data vector)
cost = DualMasterCostFunction(
    data_space=D,
    property_space=P,
    model_space=M,
    G=G,
    T=T,
    model_prior_support=σ_B,
    data_error_support=σ_V,
    observed_data=observed_data,
    q_direction=P.basis_vector(0)  # Initial direction
)

# 6. Solve for h_U(q) in multiple directions
directions = [P.basis_vector(i) for i in range(5)]
bounds = [cost.solve_for_support_value(q) for q in directions]

# 7. The bounds define the admissible property set U
print(f"Directional bounds: {bounds}")
```

---

## Phase 7: Planes and Half-Spaces (Independent)

**Status:** IN PROGRESS (Tasks 7.1, 7.2, 7.3, 7.4, 7.5, 7.7 complete — 6/7 tasks)

**Goal:** Implement linear hyperplane and half-space convex sets with support functions

**Motivation:** Planes and half-spaces are fundamental geometric objects in convex analysis and inverse problems. They enable:
- Linear constraints on model parameters
- Hard data bounds (e.g., data must be non-negative)
- Logical constraints in tomographic inversion
- Building blocks for polyhedral sets (intersections of half-spaces)

**Mathematical Background:**
- **Hyperplane:** H = {x : ⟨a, x⟩ = b} (unbounded, codimension 1)
- **Half-space:** H_+ = {x : ⟨a, x⟩ ≤ b} (unbounded, convex)
- **Polyhedral set:** P = ∩_i H_i (intersection of half-spaces)
- **Support function:** σ_H+(q) based on query direction q's alignment with normal a
  - For {x | ⟨a,x⟩ ≤ b}: σ(q) = b if q parallel to a (α ≥ 0), else +∞
  - If ⟨q, a⟩ ≤ 0: support is infinite (unbounded direction)
  - If ⟨q, a⟩ > 0: support value is b (normal direction)

### Tasks

- [x] **Task 7.1:** Create `HyperPlane` class in `pygeoinf/subsets.py` ✅
  - [x] **Parameters:** domain (HilbertSpace), normal_vector (a), offset (b)
  - [x] **Validation:**
    - [x] Normal vector must be non-zero
    - [x] Offset must be scalar
  - [x] **Methods:**
    - [x] `is_element(x)`: Check if ⟨a, x⟩ ≈ b (within tolerance)
    - [x] `distance_to(x)`: Compute perpendicular distance |⟨a,x⟩ - b| / ||a||
    - [x] `project(x)`: Project point onto hyperplane
    - [x] `dimension`: Placeholder for codimension 1 (requires domain.dimension())
  - [x] **Properties:** normal_vector, offset, normal_norm, boundary

- [x] **Task 7.2:** Create `HalfSpace` class in `pygeoinf/subsets.py` ✅
  - [x] **Parameters:** domain (HilbertSpace), normal_vector (a), offset (b), inequality_type ('<=', '>=')
  - [x] **Validation:** Same as HyperPlane plus inequality_type check
  - [x] **Methods:**
    - [x] `is_element(x)`: Check if ⟨a, x⟩ ≤ b (or ≥ depending on type)
    - [x] `distance_to(x)`: Signed distance to boundary plane
    - [x] `project(x)`: Project point onto boundary hyperplane
    - [x] `is_bounded()`: Return False (half-spaces are unbounded)
  - [x] **Properties:** normal_vector, offset, inequality_type, normal_norm, boundary, is_empty

- [x] **Task 7.3:** Implement `HalfSpaceSupportFunction` class in `pygeoinf/convex_analysis.py` ✅
  - [x] **Parameters:** primal_domain, normal_vector (a), offset (b), inequality_type
  - [x] **Evaluation:** σ(q) via decomposition into parallel and perpendicular components
    - [x] If q is parallel to a (residual ≤ tolerance):
      - For '<=': if α ≥ 0: σ(q) = b, else: +∞
      - For '>=': if α ≤ 0: σ(q) = b, else: +∞
    - [x] If q is NOT parallel to a: always +∞ (unbounded)
  - [x] **Support point:** Returns boundary point when σ(q) = b, else None
  - [x] **Implementation:** Robust decomposition handling numerical tolerance

- [x] **Task 7.4:** Implement `support_function` property for `HalfSpace` ✅
  - [x] Use lazy initialization (like Ball, Ellipsoid)
  - [x] Return `HalfSpaceSupportFunction` instance

- [x] **Task 7.5:** Implement `PolyhedralSet` class ✅
  - [x] **Parameters:** list of HalfSpace objects (intersection)
  - [x] **Methods:**
    - [x] `is_element(x)`: Check membership (all half-spaces satisfied)
    - [x] `half_spaces` property: Return the defining half-space list
  - [x] **Support function:** Returns None (not yet implemented)
    - [x] Note: σ_P(q) = inf_i σ_{H_i}(q) requires LP techniques
    - [x] Documented for future implementation with LP-based evaluation
  - [x] **Complex operations:** boundary, is_bounded, is_empty raise NotImplementedError
    - [x] Noted as requiring LP feasibility analysis techniques
  - [x] **File:** pygeoinf/subsets.py (new class at end)

- [ ] **Task 7.6:** Unit tests for planes and half-spaces
  - [ ] Test HyperPlane containment and projection
  - [ ] Test HalfSpace containment and projection
  - [ ] Test HalfSpaceSupportFunction evaluation
  - [ ] Test PolyhedralSet intersection (if implemented)
  - [ ] Test AffineSubspace ↔ HyperPlane conversion (Task 7.7)

- [x] **Task 7.7:** Bridge between `AffineSubspace` and `HyperPlane` (BONUS) ✅
  - [x] **AffineSubspace.from_hyperplanes(hyperplanes):** Construct affine subspace from intersection of hyperplanes
    - [x] Extract normal vectors and offsets from each HyperPlane
    - [x] Build constraint operator B(x)_i = ⟨a_i, x⟩
    - [x] Use from_linear_equation(B, w) for construction
  - [x] **AffineSubspace.to_hyperplanes():** Decompose affine subspace into minimal hyperplanes
    - [x] Extract constraint operator B and value w
    - [x] For each row i, create HyperPlane with normal a_i = B*(e_i), offset b_i = w[i]
    - [x] Return list of m hyperplanes (m = codimension)
  - [x] **File:** `pygeoinf/subspaces.py`
  - [x] **Unifies:** Geometric (AffineSubspace) ↔ Algebraic (HyperPlane intersection) representations

**Design Notes:**
- Planes and half-spaces are unbounded; use convention σ(q) = +∞ when support is unbounded
- For finite-dimensional spaces, can represent as vectors; for general Hilbert spaces, store as callable or weak reference
- PolyhedralSet (intersection of half-spaces) is challenging: support function is not a simple combination

---

## Phase 8: Visualization for Convex Sets (Independent)

**Status:** NOT STARTED

**Goal:** Add visualization methods for all convex sets (Ball, Ellipsoid, HalfSpace, etc.) with support for slices in 1D, 2D, and 3D

**Motivation:** Visual understanding of geometric objects is crucial for:
- Debugging inverse problems (checking that admissible set is reasonable)
- Communicating results to stakeholders
- Understanding how prior and likelihood constraints interact
- Teaching convex geometry and inverse problems

**Technical Approach:**
- **Unified plotting infrastructure via affine subspaces:**
  - User provides an affine subspace (2D or 3D) as the plotting surface
  - Subspace is defined via basis vectors and reference point
  - Infrastructure projects/restricts the convex set onto this affine subspace
  - Plots the boundary of the restricted set in the subspace coordinates
- **Automatic subspace generation for convenience:**
  - For 1D, 2D, 3D objects: default to full-dimensional plot
  - For n > 3: user can specify coordinate pairs or let system auto-select (e.g., first 2/3 coords)
- **Visualization libraries:**
  - **2D plots:** matplotlib (standard, lightweight, sufficient)
  - **3D plots:** Plotly (GPU-accelerated WebGL, interactive rotation/zoom/pan, Jupyter native)
- **Return type:**
  - 2D: matplotlib Figure object
  - 3D: Plotly Figure object (interactive, saveable as HTML)

**Key Insight:**
- All convex sets have a well-defined restriction to an affine subspace
- Ball/Ellipsoid: restrict the operator and apply quadratic form to reduced space
- HalfSpace: restrict to subspace via projection, plot resulting hyperplane/halfspace
- PolyhedralSet: restrict each half-space, plot intersection in subspace coordinates

**Representation-Aware Strategy (important):**
- The plotting algorithm depends on what the set can provide:
  - **(A) Membership oracle** (best, most general): `subset.is_element(x)`
  - **(B) Implicit inequality**: `g(x) <= 0` (good for contouring / isosurfaces)
  - **(C) Linear inequalities**: `A x <= b` (exact slice → low-dim polytope)
  - **(D) Support function only**: `σ_C(q)` / `support_point(q)` (harder; reconstruct via directional sampling)
- Always reduce to intrinsic coordinates of the affine subspace slice:
  - Given affine subspace `A = x0 + V` with `dim(V)=k` (k=2 or 3)
  - Choose an orthonormal basis `U ∈ R^{n×k}` for V (QR / Gram–Schmidt)
  - Parameterize points on A by `x(y) = x0 + U y` with `y ∈ R^k`
  - Plot the pulled-back set `C~ = { y ∈ R^k : x0 + U y ∈ C }`
- Orthonormal bases matter: always orthonormalize the provided basis before plotting.

### Tasks

- [ ] **Task 8.1:** Create `pygeoinf/visualization.py` module (slice parameterization + backends)
  - [ ] Import matplotlib (2D), plotly.graph_objects (3D), numpy, typing
  - [ ] Provide affine-slice parameterization utilities:
    - [ ] `orthonormalize_basis(basis_vectors) -> U`
    - [ ] `affine_parameterization(affine_subspace) -> (x0, U)`
    - [ ] `lift_to_ambient(y, x0, U) -> x = x0 + U y`
    - [ ] `project_to_slice_coords(x, x0, U) -> y = U^T (x-x0)`
  - [ ] Provide plotting dispatcher:
    - [ ] `plot_slice(subset, on_subspace, backend='auto', method='auto', **kwargs)`
      - backend: 'matplotlib' (k=2), 'plotly' (k=3), or 'auto'
      - method selects the best available representation (A/B/C/D)
  - [ ] **Dependency:** add plotly to project deps (for 3D WebGL interactivity)

- [ ] **Task 8.2:** Add plotting entrypoint for all subsets
  - [ ] Add `plot(on_subspace=None, backend='auto', method='auto', **kwargs)` to `Subset`
  - [ ] Default: if `on_subspace` is None, auto-create a 2D coordinate slice (first two axes)
  - [ ] Delegates to `visualization.plot_slice(self, ...)`

- [ ] **Task 8.3:** Implement a **generic membership-oracle slice plot** (Representation A)
  - [ ] Works for any set that supports `is_element(x)`
  - [ ] 2D: grid sampling in y-space + filled region / contour boundary
  - [ ] 3D: sampling in y-space + point cloud / coarse surface (plotly)
  - [ ] Parameters: bounding box in y-space, resolution, sampling strategy

- [ ] **Task 8.4:** Implement **implicit-inequality slice plot** (Representation B)
  - [ ] If a set can provide an implicit function `g(x)` (or signed distance), plot `g(x0+Uy) <= 0`
  - [ ] 2D: contour / filled contour
  - [ ] 3D: plotly isosurface / mesh
  - [ ] This becomes the preferred general method when available (more accurate than boolean membership)

- [ ] **Task 8.5:** Implement **linear-inequality slice plot** for polytopes/polyhedral sets (Representation C)
  - [ ] For `PolyhedralSet` (and any future `Ax <= b` forms): compute slice constraints
    - `A_slice = A U`, `b_slice = b - A x0`
  - [ ] Plot the resulting 2D/3D polytope in y-coordinates
  - [ ] Exact + efficient

- [ ] **Task 8.6:** Implement **support-function-based slice reconstruction** (Representation D)
  - [ ] When only `support_point(q)`/`σ_C(q)` is available, reconstruct the slice via directional sampling
  - [ ] 2D: sample angles on S^1, compute support points, take convex hull in y
  - [ ] 3D: sample directions on S^2, compute support points, reconstruct surface (approx)
  - [ ] Mark as “numerically delicate”; keep as fallback

- [ ] **Task 8.7:** Add efficient set-specific implementations (fast paths)
  - [ ] Ball: analytic circle/sphere on the slice
  - [ ] Ellipsoid: analytic ellipse/ellipsoid using restricted quadratic form `A_V = U^T A U`
  - [ ] HalfSpace/HyperPlane: analytic line/plane on the slice, shading with bounding box
  - [ ] PolyhedralSet: exact polytope slice using half-space reduction

- [ ] **Task 8.8:** Demo notebook
  - [ ] **File:** `demos/8_visualization_demo.ipynb`
  - [ ] Show: membership vs analytic fast paths, 2D vs 3D backends, and custom affine-subspace slices

- [ ] **Task 8.9:** Unit tests for visualization
  - [ ] Smoke tests: plotting returns a figure object for each backend
  - [ ] Slice parameterization consistency (U orthonormalization)
  - [ ] Basic regression tests for ball/half-space slice geometry


**Design Considerations:**
- **Dual plotting backend architecture:**
  - **2D (matplotlib):** Standard, lightweight, production-ready
  - **3D (plotly):** GPU-accelerated WebGL, interactive 3D rotation/zoom/pan
  - `backend='auto'` parameter automatically selects appropriate library based on subspace dimension
  - Allows users to override backend if desired (e.g., force plotly for 2D comparison plots)
- **Affine subspace as plotting surface:**
  - User constructs AffineSubspace instance defining the plot domain
  - `plot(on_subspace=V)` restricts the set and plots on V
  - Enables flexible visualization of high-dimensional objects via user-chosen slices
- **Coordinate transformation:**
  - Affine subspace provides orthonormal basis and reference point
  - Plot coordinates naturally align with subspace basis
  - Infrastructure handles basis transformation transparently
- **Automatic default subspaces:**
  - If on_subspace=None: auto-generate from first 2 (or 3) coordinate axes
  - Convenience for simple 2D/3D visualization without manual subspace construction
- **Operator restriction for quadratic forms:**
  - For ellipsoids: project operator A onto subspace basis → A_V (d×d matrix in subspace)
  - Math: A_V = U^T A U where U is basis matrix of V
  - Ensures quadratic form is correctly restricted
- **Half-space projection:**
  - Normal vector a is projected onto subspace: a_V = proj_V(a)
  - Offset unchanged: b_V = b
  - Result is hyperplane equation in subspace coordinates
- **3D interactivity benefits:**
  - Plotly WebGL rendering: smooth real-time rotation, no lag even for complex meshes
  - GPU acceleration: efficient for high-resolution surface meshes
  - Jupyter-native: interactive plots in notebooks without external viewers
  - HTML export: save interactive plots as standalone HTML files for sharing
- **Color schemes & styling:** Use distinct colors for different sets; shading/transparency for half-spaces
- **Performance:** Cache boundary/mesh points if object is plotted multiple times
- **Plot quality vs generality:**
  - Membership-only plotting is universal but coarse (resolution-driven)
  - Implicit inequality / signed distance enables higher quality contours/isosurfaces
  - Linear-inequality slice plotting is exact and preferred for polytopes
  - Support-function-only plotting is viable but should remain a fallback

---

## Phase 6: Advanced Features (Optional)

**Status:** NOT STARTED

### Optional Enhancements

- [ ] **Feature 6.1:** Support for non-Euclidean property spaces
  - [ ] Use Riesz maps to handle dual space properly
  - [ ] Update validation to allow `MassWeightedHilbertSpace` for P

- [ ] **Feature 6.2:** Minkowski sum support functions
  - [ ] Create `MinkowskiSumSupportFunction` class in `convex_analysis.py`
  - [ ] h_{S⊕T}(q) = h_S(q) + h_T(q)
  - [ ] Enable combining multiple error sources

- [ ] **Feature 6.3:** Caching/memoization for repeated queries
  - [ ] Cache optimal λ*(q) for each direction
  - [ ] Implement warm-start strategies

- [ ] **Feature 6.4:** Ellipsoid outer bounds for U
  - [ ] Compute ellipsoid approximation from directional bounds
  - [ ] Visualize admissible property set

- [ ] **Feature 6.5:** Connection to existing Backus-Gilbert module
  - [ ] Verify consistency with `pygeoinf/backus_gilbert.py`
  - [ ] Show how BG is a special case of dual master equation

---

## File Organization Summary

| File | Status | Description |
|------|--------|-------------|
| `pygeoinf/subsets.py` | ✅ UPDATED | Implement `support_function` property + lazy caching; optional inverse operators; **will add:** HyperPlane, HalfSpace |
| `pygeoinf/convex_analysis.py` | ✅ UPDATED | Added subgradient delegation to support_point(); **will add:** HalfSpaceSupportFunction |
| `pygeoinf/nonlinear_forms.py` | ✅ UPDATED | Added subgradient parameter and methods; updated arithmetic operators |
| `pygeoinf/backus_gilbert.py` | ✅ UPDATED | Added `DualMasterCostFunction` class with subgradient support |
| `pygeoinf/convex_optimisation.py` | ✅ CREATED | SubgradientDescent solver with constant step size (Sub-Phase 4.1 complete) |
| `pygeoinf/visualization.py` | 📝 TO CREATE (Phase 8) | Visualization methods for convex sets in 1D, 2D, 3D and slices |
| `pygeoinf/testing_sets/test.py` | ✅ UPDATED | Test harness using DualMasterCostFunction + SubgradientDescent |
| `pygeoinf/testing_sets/dual_master_demo.ipynb` | 📝 TO CREATE | Demo notebook for Phase 5 |
| `pygeoinf/testing_sets/visualization_demo.ipynb` | 📝 TO CREATE | Demo notebook for Phase 8 |
| `tests/test_dual_linear_inversion.py` | 📝 TO CREATE | Unit tests for Phase 5 |
| `tests/test_visualization.py` | 📝 TO CREATE (Phase 8) | Unit tests for visualization methods |
| `pygeoinf/__init__.py` | 🔄 TO MODIFY | Add new exports |

---

## Key Design Decisions

### 1. Why `NonLinearForm` not `NonLinearOperator`?
φ(λ; q) maps vectors → scalars (functional), not vectors → vectors.

### 2. Why Euclidean spaces for λ and q?
The dual optimization is naturally finite-dimensional; coordinate representation is standard in DLI literature.

### 3. Why allow non-Euclidean M?
Models may live in function spaces (L², Sobolev spaces) where Riesz maps differ from identity. The framework should support this generality.

### 4. Why make `support_function` a property with lazy evaluation?
- Avoids recreating `SupportFunction` objects on every call (cached after first access)
- Users can instantiate geometric sets without support function infrastructure
- Enables direct access to `support_point` for computing subgradients
- Provides clean API for passing support functions to cost function
- Graceful degradation: Ellipsoid can exist without inverse operators; errors only when using incomplete support function
- API breaking change: `support_function` changes from method to property, but cleaner semantics

### 5. Why cache T*q?
T*q is constant for a fixed direction q and appears in every evaluation of φ(λ). Caching avoids redundant adjoint computations.

### 6. Why two support functions?
The master dual equation encodes:
- **Model prior geometry** via σ_B: constrains models through the Hilbert-space residual T*q - G*λ
- **Data error geometry** via σ_V: constrains data misfit through -λ
This formulation unifies Bayesian and deterministic approaches, with both prior and likelihood encoded as convex geometry.

### 7. Why σ_V(-λ) not σ_V(λ)?
The sign convention matches the dual formulation from convex analysis. The optimal λ represents a dual certificate, and the negation ensures correct duality relationships.

---

## Progress Tracker

**Overall Progress:** 3.5/8 phases complete (~44%)

**Progress Tracker:** 3.5/8 phases complete + Phase 8 Partial Implementation (44-48%)

| Phase | Status | Tasks Complete | Tasks Total | Notes |
|-------|--------|----------------|-------------|-------|
| Phase 1: Architecture Analysis | ✅ COMPLETE | - | - | Foundation for all following phases |
| Phase 2: Refactor ConvexSubset | ✅ COMPLETE | 3 | 3 | support_function as cached property |
| Phase 3: DualMasterCostFunction | ✅ COMPLETE | 6 | 6 | Dual master equation implementation |
| Phase 4: Solver Implementation | 🟨 IN PROGRESS | 6 | 19 | Sub-Phase 4.1 complete; 4.2-4.4 pending |
| Phase 4.1: Basic Subgradient | ✅ COMPLETE | 6 | 6 | SubgradientDescent with constant step size |
| Phase 4.2: Step Size Rules | ⏸️ NOT STARTED | 0 | 5 | Diminishing, Polyak, adaptive rules |
| Phase 4.3: Integration | ⏸️ NOT STARTED | 0 | 4 | DualMasterCostFunction.solve_subgradient() |
| Phase 4.4: Advanced Methods | ⏸️ NOT STARTED | 0 | 4 | Bundle, proximal, stochastic methods |
| Phase 5: Integration & Testing | ⏸️ NOT STARTED | 0 | 4 | Demo notebook, unit tests, exports |
| Phase 6: Advanced Features | ⏸️ NOT STARTED | 0 | 5 | Non-Euclidean spaces, Minkowski sums, caching |
| Phase 7: Planes & Half-Spaces | 🟨 IN PROGRESS | 3 | 7 | HyperPlane, HalfSpace, AffineSubspace bridge |
| Phase 8: Visualization | 🟨 PARTIAL | 1 (of 9) | 9 | SubspaceSlicePlotter: Task 8.3 complete |
| **Phase 8 Companion: SubspaceSlicePlotter** | **✅ COMPLETE** | **1D/2D/3D** | **All dims** | **Active, fully functional; addresses 8.3** |

---

## Next Actions

**Immediate priority chains:**

**Chain A (Solver Development - Primary):** Phase 4 Sub-Phases
1. ✅ Done: Phase 4.1 (Basic subgradient descent)
2. Next: Phase 4.2 (Improved step size rules)
   - Extend `SubgradientDescent` with 'diminishing', 'polyak', 'adaptive' strategies
   - Implement α_k = α₀/(1+k) and α_k = α₀/sqrt(1+k) diminishing rules
   - Implement Polyak step size with f_target estimation
   - Add convergence diagnostics
3. Then: Phase 4.3 (Integration methods) and Phase 4.4 (Advanced methods)
4. Finally: Phase 5 (Testing & Integration) and Phase 6 (Advanced Features)

**Chain B (Visualization - Independent):** Phases 7-8
- Can be done in parallel with or after Chain A
- Phase 7: Implement planes and half-spaces (geometric foundation)
- Phase 8: Implement visualization methods for all convex sets
- **Best if done after Phases 1-3 are complete (which they are)**

**Command to start Chain A:**
```bash
conda activate inferences3
code pygeoinf/convex_optimisation.py
```

**Command to start Chain B (alternative):**
```bash
conda activate inferences3
code pygeoinf/subsets.py  # Add HyperPlane and HalfSpace classes
```

---

## Phase 8 Partial Implementation: SubspaceSlicePlotter (Companion Work)

**Status:** ✅ COMPLETE (Companion Implementation)

**Summary:** An independent, focused implementation of affine subspace visualization has been developed and is actively used. This addresses key aspects of Phase 8 using a simpler, pragmatic architecture tailored for 1D/2D/3D visualization via membership oracles.

**Scope Comparison:**
- **Phase 8 Plan:** General-purpose framework for multiple representation strategies (A/B/C/D), dual backends (matplotlib 2D + plotly 3D), and set-specific fast paths
- **SubspaceSlicePlotter:** Focused implementation for 1D/2D/3D affine subspaces using membership oracle (representation A) with matplotlib backend

**What Has Been Implemented:**

### Class: `SubspaceSlicePlotter` (in `pygeoinf/plot.py`)

**Constructor & Parameters:**
- [x] Full parameter validation (7 validation checks):
  - EuclideanSpace domain check
  - Dimension check (1D/2D/3D only)
  - grid_size must be positive int
  - alpha must be in [0.0, 1.0]
  - rtol must be positive
  - **NEW:** bar_pixel_height must be positive int (recent UX improvement)
- [x] Flexible instantiation with sensible defaults
- [x] Automatic tangent basis extraction from AffineSubspace
- [x] Translation vector storage from subspace

**Common Methods (All Dimensions):**
- [x] **parse_bounds()**: Flexible bounds parsing for 1D/2D/3D
  - Accepts multiple formats: None, flat tuple, nested tuples
  - Automatic dimension-aware normalization
- [x] **embed_point()**: Lifts parameter space points y to ambient space x = x0 + U y
  - Works for all dimensions
  - Uses stored tangent basis and translation
- [x] **_generate_param_grid()**: Generates parameter grids
  - 1D: Linear spacing (1D array)
  - 2D: Meshgrid (2D arrays X, Y)
  - 3D: Regular grid (3D arrays X, Y, Z)
- [x] **sample_membership()**: Oracle evaluation on grid
  - Calls subset.is_element(x) for each embedded point
  - Returns binary membership array
  - Respects rtol parameter

**Dimension-Specific Rendering:**
- [x] **_render_1d()**: Bar chart visualization
  - Plots each member point as a vertical bar
  - **RECENT ENHANCEMENT:** Fixed pixel-based bar height
    - Parameter: self.bar_pixel_height (default 6 pixels)
    - Helper method: self._pixel_to_data_height() converts pixels to data units
    - Ensures consistent visual appearance across plots with different axis ranges
  - Supports alpha transparency
- [x] **_render_2d()**: Contour + filled region visualization
  - Filled contour for membership region
  - Contour lines for boundary
  - Color map with transparency
  - Tight axis bounding
- [x] **_render_3d()**: 3D surface visualization
  - Voxel-based rendering (grid of colored boxes)
  - Alpha blending for transparency
  - Supports both orthogonal and perspective views
  - Jupyter-friendly with interactive rotation

**Plotting Dispatcher:**
- [x] **plot()**: Main entry point
  - Automatically selects renderer based on dimension
  - Handles all figure/axes setup
  - Returns matplotlib Figure and Axes objects
  - Supports custom bounds, grid size, and transparency

**Implementation Quality:**
- [x] Full type hints (no TYPE_CHECKING hacks beyond AffineSubspace import)
- [x] Comprehensive docstrings for all public methods
- [x] Clear parameter descriptions and return types
- [x] Dimension-agnostic architecture where appropriate
- [x] Defensive validation (7 checks in constructor)
- [x] Pixel-to-data conversion with fallback error handling

**Active Usage:**
- Successfully visualizing in `pygeoinf/testing_sets/test_dual_master.ipynb`:
  - 1D slices of admissible sets on lines
  - 2D slices of convex sets on planes
  - Both Ball and Ellipsoid subsets
  - Grid sizes from 100 to 200+ points

### Relationship to Phase 8 Tasks

**Tasks Already Partially Addressed:**

| Phase 8 Task | SubspaceSlicePlotter Status |
|--------------|----------------------------|
| 8.1: visualization.py module | ⚠️ Partial (in plot.py, not separate module) |
| 8.2: Subset.plot() entrypoint | ⏸️ Not yet (SubspaceSlicePlotter used directly, not as method) |
| 8.3: Generic membership-oracle plot | ✅ **COMPLETE** (all dimension cases: 1D bars, 2D contours, 3D voxels) |
| 8.4: Implicit-inequality plot | ⏸️ Not yet (would require signed distance functions) |
| 8.5: Linear-inequality plot | ⏸️ Not yet (would require PolyhedralSet support) |
| 8.6: Support-function reconstruction | ⏸️ Not yet (would require directional sampling) |
| 8.7: Set-specific fast paths | ⏸️ Not yet (currently generic membership oracle only) |
| 8.8: Demo notebook | ⚠️ Partial (active usage in test notebooks) |
| 8.9: Unit tests | ⏸️ Not yet (integration tested, no formal unit tests) |

**Key Differences from Phase 8 Plan:**
1. **Single backend:** matplotlib only (no plotly 3D yet)
2. **Single representation:** Membership oracle only (not representations B/C/D)
3. **Location:** plot.py (not separate visualization.py module)
4. **Integration:** Used directly (not as Subset.plot() method)
5. **Scope:** 1D/2D/3D only (not general n-dimensional with auto-selection)

**Advantages of Current Implementation:**
- ✅ **Working today:** Actively used in notebooks for visualization
- ✅ **Simple architecture:** Easy to understand and maintain
- ✅ **Clean API:** Straightforward constructor + plot() call
- ✅ **Flexible bounds:** Dimension-aware parsing for any format
- ✅ **Visual quality:** Recent pixel-height fix ensures consistent appearance
- ✅ **Well-validated:** 7 parameter checks catch user errors early

**Path Forward for Phase 8:**
1. **Consolidate:** Move SubspaceSlicePlotter to new `pygeoinf/visualization.py` module
2. **Extend:** Add Subset.plot() method that wraps SubspaceSlicePlotter
3. **Enhance:** Implement set-specific fast paths (Ball, Ellipsoid analytic slices)
4. **Supplement:** Add plotly backend for interactive 3D visualization
5. **Generalize:** Implement representations B/C/D for broader set types
6. **Test:** Add formal unit tests for slice parameterization and rendering

### Recent Enhancement: Pixel-Based Bar Height (Feb 2, 2026)

**Problem:** 1D bar charts displayed inconsistent visual thickness across plots
- Calculation used data-range fraction: `height = 0.01 * (u_max - u_min)`
- When axis ranges differed, same data-coordinate height mapped to different pixel heights
- Result: bars appeared thick in some plots, thin in others

**Solution:** Introduced pixel-based height calculation
- New parameter: `bar_pixel_height: int = 6` (default 6 pixels)
- New helper method: `_pixel_to_data_height(ax, pixels) -> float`
  - Converts pixel distance to data coordinate height
  - Uses axes window extent and inverted transform
  - Includes fallback: calls fig.canvas.draw() if renderer unavailable
- Updated `_render_1d()` to use `height_data = self._pixel_to_data_height(ax, pixel_height)`

**Implementation Details:**
```python
def _pixel_to_data_height(self, ax, pixels):
    """Convert pixel distance to data coordinates."""
    try:
        renderer = ax.get_figure().canvas.get_renderer()
    except (AttributeError, RuntimeError):
        ax.get_figure().canvas.draw()
        renderer = ax.get_figure().canvas.get_renderer()

    bbox = ax.get_window_extent(renderer=renderer)
    x_disp = bbox.x0 + bbox.width * 0.5
    y_disp0 = bbox.y0 + bbox.height * 0.5
    y_disp1 = y_disp0 + pixels

    inv = ax.transData.inverted()
    y_data0 = inv.transform((x_disp, y_disp0))[1]
    y_data1 = inv.transform((x_disp, y_disp1))[1]

    return abs(y_data1 - y_data0)
```

**Result:** ✅ Bars now render with consistent visual thickness across all plots

---

## Notes and Observations

- The existing `SupportFunction` hierarchy in `convex_analysis.py` is well-designed and ready for use
- The `Ball` and `Ellipsoid` classes expose `support_function` as a cached property (lazy)
- **Phase 2 is API-breaking:** `support_function` changes from a method to a property returning a `SupportFunction` object
- **Graceful degradation:** Users can instantiate `Ellipsoid` without inverse operators - no errors until they try to use incomplete support function methods
- **Lazy evaluation:** Support function objects are created only when the property is first accessed
- **Responsibility separation:**
  - `Ball`/`Ellipsoid` classes: handle geometric set definition, never error on construction
  - `SupportFunction` classes: handle support function evaluation, error when missing required operators
- Following `convex_analysis.py` design: users must provide inverse operators explicitly; do NOT auto-compute them
- The `DualMasterCostFunction` integrates cleanly with pygeoinf's existing solver infrastructure via `NonLinearForm` interface
- **Key insight:** The Hilbert form uses TWO support functions (σ_B for model prior, σ_V for data error), eliminating the need for explicit Mahalanobis norm weights. All geometric constraints are encoded through support function evaluations.
- The negative sign in σ_V(-λ) is essential for correct dual formulation and matches the convex analysis literature

---

## Summary of Changes in This Update

**Phase 7 Progress (Tasks 7.1, 7.2, 7.7 complete):**
- ✅ `HyperPlane` class added to `subsets.py`
  - Full geometric implementation: is_element, distance_to, project, boundary
  - Properties: normal_vector, offset, normal_norm
  - Represents {x | ⟨a, x⟩ = b}
- ✅ `HalfSpace` class added to `subsets.py`
  - Supports both '<=' and '>=' inequality types
  - Full geometric implementation: is_element, distance_to, project, boundary
  - Properties: normal_vector, offset, inequality_type, normal_norm
  - Methods: is_bounded() [returns False], is_empty [returns False]
  - Represents {x | ⟨a, x⟩ ≤ b} or {x | ⟨a, x⟩ ≥ b}
- ✅ **Bridge implementation** in `subspaces.py` (BONUS Task 7.7)
  - `AffineSubspace.from_hyperplanes()`: Construct from intersection of HyperPlane objects
  - `AffineSubspace.to_hyperplanes()`: Decompose into minimal set of hyperplanes
  - Unifies geometric (projector-based) and algebraic (constraint-based) representations
  - Enables bidirectional conversion between representations

**Status Update:**
- Phase 7: 3/7 tasks complete (43%)
- Phase 8: 1/9 tasks complete via SubspaceSlicePlotter (11%; membership-oracle slice plot fully implemented)
- SubspaceSlicePlotter: Actively used and fully functional (1D/2D/3D visualization)
- Overall project: ~44-48% complete (accounting for Phase 8 partial implementation)

**Design Highlights:**
- HyperPlane and HalfSpace are independent of AffineSubspace (different use cases)
- Bridge methods allow seamless conversion when needed
- SubspaceSlicePlotter provides working visualization infrastructure for all convex sets
- Both classes inherit from Subset, fit naturally into convex geometry hierarchy
- Ready for support function implementation (Phase 7.3-7.4) and Phase 8 completion

**Recent Major Work (Feb 2, 2026):**
- ✅ SubspaceSlicePlotter pixel-based bar height fix (UX improvement for 1D visualization)
- ✅ Added `bar_pixel_height` parameter and `_pixel_to_data_height()` helper method
- ✅ Ensures consistent visual appearance across plots with different axis ranges

**Last Updated:** February 2, 2026
**Document Version:** 2.9 (Phase 8 Partial Implementation documented; SubspaceSlicePlotter companion work integrated)
