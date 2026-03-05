---
handoff:
  - label: Start implementation with Atlas
    agent: Atlas
    prompt: Implement the plan
---

> ⚠️ **ARCHIVED** (2026-03-05)
> DualMasterCostFunction was fully implemented as Phase 2 of bundle-methods-full-plan.md.
> See: docs/agent-docs/completed-plans/bundle-methods-full-plan-phase-2-complete.md
> Kept for reference to understand implementation approach.

# Dual Master — Implementation Plan (Atlas)

Location: `pygeoinf/docs/agent-docs/references/legacy/superseded-plans/dual_master_implementation.md`

## Purpose
Concrete, phase-by-phase TDD plan for implementing the dual master equation cost function for Direct Linear Inversion in pygeoinf. This plan is meant for Atlas orchestration and Sisyphus implementation.

## Mathematical Background

**Master Dual Equation (Hilbert Form):**
```
h_U(q) = inf_λ { ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ) }

where:
  φ(λ; q) = ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ)
  λ ∈ D (data space dual variables)
  q ∈ P (property space direction)
  d̃ = observed data vector
  σ_B = support function of model prior set B ⊆ M
  σ_V = support function of data error set V ⊆ D
  G: M → D (forward operator)
  T: M → P (property extraction operator)
```

**Key insight:** All geometric information about the admissible property set U is encoded by the support functions of the convex sets B and V, evaluated at the Hilbert-space residual T*q - G*λ.

---

## Progress Summary

**Overall Progress:** 4/8 phases complete (~50%)

| Phase | Status | Progress | Notes |
|-------|--------|----------|-------|
| Phase 1 | ✅ COMPLETE | - | Architecture analysis (foundation) |
| Phase 2 | ✅ COMPLETE | 3/3 | ConvexSubset support_function refactor |
| Phase 3 | ✅ COMPLETE | 6/6 | DualMasterCostFunction implementation |
| Phase 4 | 🟨 IN PROGRESS | 6/19 | Solver (Sub-Phase 4.1 complete) |
| Phase 5 | ⏸️ NOT STARTED | 0/4 | Integration & Testing |
| Phase 6 | ⏸️ NOT STARTED | 0/5 | Advanced Features (optional) |
| Phase 7 | ✅ COMPLETE | 7/7 | Planes & Half-Spaces (35 tests passing) |
| Phase 8 | 🟨 PARTIAL | 1/9 | Visualization (SubspaceSlicePlotter done) |

---

## Phase 2: ConvexSubset support_function refactor ✅

**Status:** COMPLETE
**Owner:** Sisyphus
**Files:** `pygeoinf/subsets.py`, `pygeoinf/convex_analysis.py`
**Tests:** None (refactoring existing functionality)

### Goal
Make support functions first-class, reusable objects that are lazily computed via properties.

### Tasks Completed
- [x] Add abstract `support_function` property to `ConvexSubset` class
- [x] Implement `support_function` property in `Ball` class (always available)
- [x] Implement `support_function` property in `Ellipsoid` class (requires inverse operators)
- [x] Lazy caching: support function created on first access
- [x] Graceful degradation: Ellipsoid can exist without inverses; errors only when using incomplete support function

### Acceptance
`Ball` and `Ellipsoid` expose cached `support_function` property. Support functions are lazily instantiated only when accessed. API change: `support_function` is now a property (not a method).

---

## Phase 3: DualMasterCostFunction ✅

**Status:** COMPLETE
**Owner:** Sisyphus
**Files:** `pygeoinf/backus_gilbert.py`, `pygeoinf/nonlinear_forms.py`
**Tests:** `tests/test_dual_master_cost.py`

### Goal
Implement the cost function φ(λ; q) as a `NonLinearForm` with subgradient oracle.

### Tasks Completed
- [x] Create `DualMasterCostFunction` class in `pygeoinf/backus_gilbert.py`
- [x] Implement `__init__` constructor with parameters:
  - data_space, property_space, forward_operator G, property_operator T
  - observed_data d̃, model_prior_support_fn σ_B, data_error_support_fn σ_V
  - q_direction (initial direction)
- [x] Implement `_mapping(λ)`: evaluate φ(λ; q) = ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)
- [x] Implement `_gradient(λ)`: return subgradient via support function delegation
- [x] Implement `set_direction(q)`: update q and cache T*q for efficiency
- [x] Integration with support function subgradient mechanism

### Example Usage
```python
cost = DualMasterCostFunction(
    data_space=D, property_space=P,
    forward_operator=G, property_operator=T,
    observed_data=d_tilde,
    model_prior_support_fn=ball_B.support_function,
    data_error_support_fn=ball_V.support_function,
    q_direction=P.basis_vector(0)
)
value = cost(lambda_trial)
subgrad = cost.subgradient(lambda_trial)
```

### Acceptance
Cost function evaluates and exposes subgradient oracle for simple examples. Tested with `BallSupportFunction` oracles on 1D toy problem.

---

## Phase 4: Solver Implementation 🟨

**Status:** IN PROGRESS (Sub-Phase 4.1 complete ✅)
**Owner:** Sisyphus
**Files:** `pygeoinf/convex_optimisation.py`
**Tests:** `tests/test_subgradient.py`

### Goal
Implement solvers for the minimization problem inf_λ φ(λ; q) using subgradient methods.

**Note:** Standard gradient-based methods don't work because support functions are non-smooth. We implement subgradient methods incrementally in sub-phases.

---

### Sub-Phase 4.1: Basic Subgradient Descent ✅

**Status:** COMPLETE
**Progress:** 6/6 tasks

#### Goal
Implement the simplest possible subgradient descent algorithm to understand mechanics.

#### Key Concepts
- **Subgradient descent iteration:** x_{k+1} = x_k - α_k g_k where g_k ∈ ∂f(x_k)
- **Non-monotonic:** Function value may increase between iterations (normal behavior)
- **Convergence:** Requires careful step size; constant step → oscillation, diminishing → convergence
- **Best point tracking:** Track best value found so far (monotonic improvement in best)

#### Tasks Completed
- [x] Create `pygeoinf/convex_optimisation.py` module
- [x] Define `SubgradientResult` dataclass (8 fields: solution, value, converged, iterations, etc.)
- [x] Implement `SubgradientDescent` class with oracle-based design
- [x] Implement `__init__` (stores oracle, max_iter, tol, step_size, store_iterates)
- [x] Implement `solve(x0)` method (full iteration loop with best-point tracking)
- [x] Add stagnation-based convergence check

#### Acceptance
`SubgradientDescent` runs with constant step size on `DualMasterCostFunction`. Tested on 1D toy problem with visualization. This is a "learning implementation" – production use requires Sub-Phase 4.2.

---

### Sub-Phase 4.2: Improved Step Size Rules ⏸️

**Status:** NOT STARTED (depends on 4.1 ✅)
**Progress:** 0/5 tasks

#### Goal
Add sophisticated step size schedules for guaranteed convergence.

#### Step Size Rules to Implement
1. **Diminishing:** α_k = α₀ / (1 + k) or α_k = α₀ / sqrt(k + 1)
   - Guarantees convergence: Σ α_k = ∞, Σ α_k² < ∞
2. **Polyak:** α_k = (f_k - f_target) / ||g_k||²
   - Requires knowledge/estimate of optimal value
3. **Adaptive:** Increase on progress, decrease on oscillation
   - Track recent function values, backtracking-like heuristics

#### Tasks
- [ ] Extend `SubgradientDescent` with `step_size_rule` parameter ('constant', 'diminishing', 'polyak', 'adaptive')
- [ ] Implement diminishing step size (square summable)
- [ ] Implement Polyak step size with safeguards (clip to [α_min, α_max])
- [ ] Add convergence criteria (relative improvement, gradient norm, stagnation)
- [ ] Add iteration diagnostics and logging (print every N iterations)

#### Acceptance
`SubgradientDescent` supports 'constant', 'diminishing', 'polyak', 'adaptive' rules and unit tests validate behavior on toy problems.

---

### Sub-Phase 4.3: Integration helpers for DualMasterCostFunction ⏸️

**Status:** NOT STARTED (depends on 4.2)
**Progress:** 0/4 tasks
**Files:** `pygeoinf/backus_gilbert.py`
**Tests:** `tests/test_dual_master_cost.py`

#### Goal
Connect the subgradient solver to `DualMasterCostFunction` and create user-facing API.

#### Tasks
- [ ] Add `solve_subgradient(initial_lambda, solver_kwargs)` method to `DualMasterCostFunction`
  - Return tuple: (optimal_lambda, SubgradientResult)
- [ ] Add `solve_for_support_value(q)` helper method
  - Set direction to q, solve, return optimal value h_U(q)
- [ ] Add validation and warnings (non-convergence, numerical issues)
- [ ] Document usage patterns (parameter choices, diagnostics)

#### Acceptance
`solve_subgradient` and `solve_for_support_value` convenience methods added with warnings and diagnostics.

---

### Sub-Phase 4.4: Optional Advanced Methods ⏸️

**Status:** NOT STARTED (optional, depends on 4.3)
**Progress:** 0/4 tasks

#### Optional Tasks
- [ ] Projected subgradient descent (box constraints on λ)
- [ ] Bundle method (cutting-plane, polyhedral approximation)
- [ ] Proximal gradient method (decompose f = g + h)
- [ ] Stochastic subgradient (large-scale sum structure)

---

## Phase 5: Integration & Testing ⏸️

**Status:** NOT STARTED
**Owner:** Sisyphus
**Progress:** 0/4 tasks
**Files:** `pygeoinf/testing_sets/dual_master_demo.ipynb`, `tests/test_dual_linear_inversion.py`
**Tests:** `tests/test_dual_linear_inversion.py`

### Goal
Create end-to-end demo notebook and comprehensive unit tests.

### Tasks
- [ ] Create demo notebook `dual_master_demo.ipynb`
  - Define spaces D, P, M (Euclidean)
  - Create operators G: M→D, T: M→P
  - Define convex sets B (model prior), V (data error)
  - Construct DualMasterCostFunction
  - Solve for h_U(q) in multiple directions
  - Visualize admissible property set
- [ ] Write unit tests in `tests/test_dual_linear_inversion.py`
  - Test Ball support functions
  - Test Ellipsoid with inverses
  - Test composition and edge cases
- [ ] Update package exports in `pygeoinf/__init__.py`
  - Add `from .backus_gilbert import DualMasterCostFunction`
  - Add `from .convex_optimisation import SubgradientDescent`
- [ ] Syntax and import checks
  - Run `python -m py_compile` on all modified files
  - Run notebook cells sequentially

### Example Workflow
```python
# 1. Define spaces
D = EuclideanSpace(10)  # Data space
P = EuclideanSpace(5)   # Property space
M = EuclideanSpace(20)  # Model space

# 2. Create operators
G = LinearOperator.from_matrix(...)  # M → D
T = LinearOperator.from_matrix(...)  # M → P

# 3. Define convex sets
model_prior = Ball(M, center=M.zero, radius=1.0)
data_error = Ball(D, center=D.zero, radius=0.1)

# 4. Construct cost function
cost = DualMasterCostFunction(
    data_space=D, property_space=P,
    forward_operator=G, property_operator=T,
    observed_data=d_tilde,
    model_prior_support_fn=model_prior.support_function,
    data_error_support_fn=data_error.support_function,
    q_direction=P.basis_vector(0)
)

# 5. Solve for directional bounds
directions = [P.basis_vector(i) for i in range(5)]
bounds = [cost.solve_for_support_value(q) for q in directions]
print(f"Admissible property set bounds: {bounds}")
```

### Acceptance
Demo notebook runs end-to-end on toy problem; integration tests added and passing.

---

## Phase 6: Advanced Features (Optional) ⏸️

**Status:** NOT STARTED
**Progress:** 0/5 tasks

### Optional Enhancements
- [ ] Support for non-Euclidean property spaces (use Riesz maps)
- [ ] Minkowski sum support functions (`MinkowskiSumSupportFunction`)
- [ ] Caching/memoization for repeated queries (cache optimal λ*(q))
- [ ] Ellipsoid outer bounds for U (compute from directional bounds)
- [ ] Connection to existing Backus-Gilbert module (show BG as special case)

---

## Phase 7: Planes & Half-Spaces ✅

**Status:** COMPLETE (7/7 tasks complete)
**Owner:** Sisyphus
**Progress:** 7/7 tasks
**Files:** `pygeoinf/subsets.py`, `pygeoinf/convex_analysis.py`
**Tests:** `tests/test_halfspaces.py` (35 tests, all passing)

### Goal
Implement linear hyperplane and half-space convex sets with support functions.

### Motivation
Planes and half-spaces are fundamental geometric objects enabling:
- Linear constraints on model parameters
- Hard data bounds (e.g., non-negativity)
- Logical constraints in tomographic inversion
- Building blocks for polyhedral sets (intersections of half-spaces)

### Mathematical Background
- **Hyperplane:** H = {x : ⟨a, x⟩ = b} (unbounded, codimension 1)
- **Half-space:** H₊ = {x : ⟨a, x⟩ ≤ b} (unbounded, convex)
- **Polyhedral set:** P = ∩ᵢ Hᵢ (intersection of half-spaces)
- **Support function:**
  - For {x | ⟨a,x⟩ ≤ b}: σ(q) = αb if q = αa with α ≥ 0, else +∞
  - For {x | ⟨a,x⟩ ≥ b}: σ(q) = αb if q = αa with α ≤ 0, else +∞

### Tasks
- [x] Create `HyperPlane` class in `pygeoinf/subsets.py`
  - Parameters: domain, normal_vector (a), offset (b)
  - Properties: normal_vector, offset, normal_norm, boundary
  - Methods: contains(x), project(x), distance(x)
- [x] Create `HalfSpace` class in `pygeoinf/subsets.py`
  - Parameters: domain, normal_vector (a), offset (b), inequality_type ('<=', '>=')
  - Properties: normal_vector, offset, inequality_type, boundary, is_empty
  - Methods: contains(x), support_function (lazy property)
- [x] Implement `HalfSpaceSupportFunction` in `pygeoinf/convex_analysis.py`
  - support_value(q): handle unbounded cases (return +∞)
  - support_point(q): robust decomposition with numerical tolerance
- [x] Implement `support_function` property for `HalfSpace`
  - Lazy initialization, return `HalfSpaceSupportFunction`
- [x] Implement `PolyhedralSet` class (intersection of half-spaces)
  - Parameters: list of HalfSpace objects
  - Methods: contains(x), support_function (intersection logic)
- [x] **Unit tests for planes and half-spaces** ✅
  - 35 comprehensive tests in `tests/test_halfspaces.py`
  - Test suite breakdown:
    - `TestHyperPlane`: 12 tests (initialization, membership, projection, distance, properties)
    - `TestHalfSpace`: 13 tests (initialization, membership, support functions, boundaries)
    - `TestPolyhedralSet`: 7 tests (initialization, intersection semantics, simplex constraints)
    - `TestNumericalRobustness`: 3 tests (parallel directions, large offsets, small normals)
  - All 35 tests passing ✅
- [x] Bridge between `AffineSubspace` and `HyperPlane` (BONUS)
  - `AffineSubspace.from_hyperplanes(hyperplanes)`: construct from intersection
  - Unifies geometric ↔ algebraic representations

### Acceptance
✅ `HalfSpace` and `PolyhedralSet` support functions implemented; comprehensive unit tests pass (35/35).

---

## Phase 8: Visualization API and demos 🟨

**Status:** PARTIAL (SubspaceSlicePlotter companion complete)
**Owner:** Sisyphus
**Progress:** 1/9 tasks (Task 8.3 addressed by companion implementation)
**Files:** `pygeoinf/visualization.py`, `pygeoinf/testing_sets/visualization_demo.ipynb`
**Tests:** `tests/test_visualization.py`

### Goal
Add visualization methods for all convex sets (Ball, Ellipsoid, HalfSpace, etc.) with support for slices in 1D, 2D, and 3D.

### Motivation
Visual understanding of geometric objects is crucial for:
- Debugging inverse problems (checking admissible set is reasonable)
- Communicating results to stakeholders
- Understanding prior/likelihood constraint interactions
- Teaching convex geometry and inverse problems

### Technical Approach
- **Unified plotting via affine subspaces:**
  - User provides affine subspace (2D or 3D) as plotting surface
  - Plot restricted set in subspace coordinates
- **Multiple representation strategies:**
  - (A) Membership oracle: `subset.is_element(x)` (general, grid-based)
  - (B) Implicit inequality: `g(x) ≤ 0` (more accurate than membership)
  - (C) Linear inequality: `Ax ≤ b` (exact for polytopes)
  - (D) Support function only: reconstruct via directional sampling (fallback)
- **Dual backends:**
  - 2D: matplotlib (standard, lightweight)
  - 3D: Plotly (GPU WebGL, interactive rotation/zoom)

### Tasks
- [ ] Create `pygeoinf/visualization.py` module (slice parameterization + backends)
- [ ] Add `plot(on_subspace, backend, method, **kwargs)` to `Subset` base class
- [x] **Implement membership-oracle slice plot** (ADDRESSED by `SubspaceSlicePlotter`)
  - Works for any set with `is_element(x)`
  - Grid-based sampling in subspace coordinates
  - **Note:** `SubspaceSlicePlotter` in `pygeoinf/plot.py` provides this functionality
- [ ] Implement implicit-inequality slice plot (representation B)
- [ ] Implement linear-inequality slice plot for polytopes (representation C)
- [ ] Implement support-function-based slice reconstruction (representation D)
- [ ] Add efficient set-specific fast paths (Ball: analytic circle/sphere, etc.)
- [ ] Demo notebook `visualization_demo.ipynb`
- [ ] Unit tests `tests/test_visualization.py`

### SubspaceSlicePlotter (Companion Implementation) ✅
**Status:** COMPLETE and currently in active use

A focused implementation of affine subspace visualization has been developed:
- **Class:** `SubspaceSlicePlotter` in `pygeoinf/plot.py`
- **Scope:** 1D/2D/3D affine subspaces using membership oracle (representation A)
- **Backend:** matplotlib
- **Features:** Full parameter validation, flexible instantiation, bar plots (1D), contour/filled (2D), voxel (3D)

This addresses key aspects of Phase 8 Task 8.3 using a simpler, pragmatic architecture.

### Acceptance
Plotting backends return figure objects; demo notebook renders example slices. Current partial implementation (`SubspaceSlicePlotter`) provides working 1D/2D/3D visualization via membership oracles.

---

## Design Decisions

### Why `NonLinearForm` not `NonLinearOperator`?
φ(λ; q) maps vectors → scalars (functional), not vectors → vectors.

### Why Euclidean spaces for λ and q?
The dual optimization is naturally finite-dimensional; coordinate representation is standard in DLI literature.

### Why allow non-Euclidean M?
Models may live in function spaces (L², Sobolev spaces) where Riesz maps differ from identity.

### Why make `support_function` a property with lazy evaluation?
- Avoids recreating objects on every call (cached after first access)
- Users can instantiate geometric sets without support function infrastructure
- Enables direct access to `support_point` for computing subgradients
- Graceful degradation: Ellipsoid can exist without inverse operators
- API breaking change: cleaner semantics (property vs method)

### Why cache T*q?
T*q is constant for fixed direction q and appears in every evaluation of φ(λ). Caching avoids redundant adjoint computations.

### Why two support functions (σ_B and σ_V)?
The master dual equation encodes:
- **Model prior geometry** via σ_B: constrains models through residual T*q - G*λ
- **Data error geometry** via σ_V: constrains data misfit through -λ
Unifies Bayesian and deterministic approaches with both prior and likelihood encoded as convex geometry.

### Why σ_V(-λ) not σ_V(λ)?
Sign convention matches dual formulation from convex analysis. The optimal λ represents a dual certificate, and negation ensures correct duality relationships.

---

## Run Instructions

### Run a single test file
```bash
cd pygeoinf
PYTHONPATH=. pytest -q tests/test_dual_master_cost.py
```

### Run all tests
```bash
cd pygeoinf
PYTHONPATH=. pytest tests/
```

### Test current implementation
```bash
cd pygeoinf/testing_sets
python test.py  # Runs DualMasterCostFunction + SubgradientDescent
```

---

## Per-Phase Commit Message Template

```
Phase {id}: {title}

What changed: {files modified}
Why: Follow plan in plans/dual_master_implementation.md
Tests: pytest -q tests/{matching_test}.py
Status: {acceptance criteria met}
```

---

## Next Actions

**Immediate Priority Chains:**

**Chain A (Solver Development - Primary):**
1. ✅ Done: Phase 4.1 (Basic subgradient descent)
2. **Next:** Phase 4.2 (Improved step size rules)
   - Extend `SubgradientDescent` with diminishing/Polyak/adaptive strategies
   - Add convergence diagnostics and logging
3. Then: Phase 4.3 (Integration methods) → Phase 5 (Testing)

**Chain B (Geometric Foundations - Independent):**
1. **Next:** Phase 7 final task (unit tests for planes/half-spaces)
2. Then: Phase 8 (Visualization - remaining 8/9 tasks)

**Recommended Start:**
```bash
conda activate inferences3
code pygeoinf/convex_optimisation.py
# Begin Phase 4.2 implementation
```
