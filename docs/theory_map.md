# Theory-to-Code Mapping Reference

**Last Updated:** 2026-02-17
**Purpose:** Map mathematical concepts from theory documents to pygeoinf implementations

This document serves as a living reference connecting theory (papers, theory.txt) to code. When implementing new features or reviewing existing code, consult this map to ensure mathematical correctness.

---

## Table of Contents

1. [Fundamental Spaces](#fundamental-spaces)
2. [Operators](#operators)
3. [Support Functions](#support-functions)
4. [Dual Master Equation](#dual-master-equation)
5. [Convex Sets](#convex-sets)
6. [Optimization](#optimization)
7. [Forward Problems](#forward-problems)
8. [Inversion Methods](#inversion-methods)
9. [Notation Translation](#notation-translation)
10. [Agent Checklists](#agent-checklists)

---

## Fundamental Spaces

**Theory Reference:** theory.txt §1 "Spaces and maps"

### Model Space M (Banach/Hilbert)

**Theory:** Separable Banach space of models, typically Hilbert (§4)
**Code:** `HilbertSpace` (abstract base class)
**File:** `pygeoinf/hilbert_space.py`

**Concrete Implementations:**
- `EuclideanSpace(dim)` - ℝⁿ with standard inner product
- `MassWeightedHilbertSpace(base_space, mass_operator)` - Weighted inner product
- Custom spaces via overriding abstract methods

**Key Properties:**
- Inner product: `space.inner_product(x, y)` → float
- Riesz map: `space.to_dual(x)` (M → M*), `space.from_dual(xp)` (M* → M)
- Component representation: `space.to_components(x)` → np.ndarray
- Dimension: `space.dim` (finite-dimensional restriction)

**Mathematical Requirements:**
- Inner product must be symmetric, positive-definite, linear
- Riesz map satisfies: ⟨x, y⟩ = ⟨R(x), y⟩_dual
- Component maps are isomorphisms to ℝⁿ

### Data Space D (Finite-Dimensional)

**Theory:** D = ℝ^{N_d} with Euclidean inner product
**Code:** `EuclideanSpace(N_d)`
**Usage:** `data_space = EuclideanSpace(50)` for 50 data points

### Property Space P (Finite-Dimensional)

**Theory:** P = ℝ^{N_p} for extracted properties
**Code:** `EuclideanSpace(N_p)`
**Usage:** `property_space = EuclideanSpace(10)` for 10 properties

### Dual Spaces M*

**Theory:** Continuous linear functionals on M
**Code (Hilbert):** Identified with M via Riesz map
**Code (General):** `DualHilbertSpace(space)` wrapper
**File:** `pygeoinf/hilbert_space.py`

**Key Methods:**
- `space.dual` - returns dual space
- `space.to_dual(x)` - primal to dual (via Riesz)
- `space.from_dual(xp)` - dual to primal (inverse Riesz)

**Caution:** In Banach spaces, M* ≠ M. Hilbert: M* ≅ M via Riesz.

---

## Operators

**Theory Reference:** theory.txt §1-2, operator algebra

### Forward Operator G: M → D

**Theory:** Bounded linear map, typically non-injective (ker G ≠ {0})
**Code:** `LinearOperator(model_space, data_space, mapping)`
**File:** `pygeoinf/linear_operators.py`

**Construction:**
```python
# From matrix
G = LinearOperator.from_matrix(model_space, data_space, G_matrix)

# From function
def forward_map(m):
    return ...  # Returns element of data_space

G = LinearOperator(model_space, data_space, forward_map)
```

**Key Properties:**
- `G.domain` - model space M
- `G.codomain` - data space D
- `G.adjoint` - G*: D → M (satisfies ⟨Gx, y⟩_D = ⟨x, G*y⟩_M)
- `G.dual` - G': D* → M* (Banach dual)
- `G.kernel()` - null space (approximate for large ops)
- `G.range()` - image space (approximate)

**Mathematical Requirements:**
- Adjoint correctness: Test ⟨Gx, y⟩ = ⟨x, G*y⟩ for random x, y
- Composition order: (A @ B).adjoint = B.adjoint @ A.adjoint
- Matrix representation: `G.matrix(domain_basis, codomain_basis)` consistent

### Property Operator T: M → P

**Theory:** Linear map extracting properties from models
**Code:** `LinearOperator(model_space, property_space, mapping)`
**Usage:** See `backus_gilbert.py` DualMasterCostFunction

**Example:**
```python
# Point evaluation: Extract model value at location x₀
T = property_space.coordinate_inclusion @ point_evaluation_form(x₀)
```

### Adjoint G* (Hilbert case)

**Theory:** G*: D → M satisfying ⟨Gx, y⟩_D = ⟨x, G*y⟩_M
**Code:** `G.adjoint`
**Automatic:** Computed via dual_mapping in LinearOperator constructor

**Verification Test:**
```python
# Should be satisfied (up to numerical tolerance)
x = model_space.random()
y = data_space.random()
lhs = data_space.inner_product(G(x), y)
rhs = model_space.inner_product(x, G.adjoint(y))
np.testing.assert_allclose(lhs, rhs, rtol=1e-10)
```

### Operator Composition

**Theory:** (A ∘ B)(x) = A(B(x)), adjoint: (A ∘ B)* = B* ∘ A*
**Code:** `A @ B` (uses `__matmul__`)

**Critical:** Adjoint reverses order!
```python
composed = A @ B
composed_adjoint = composed.adjoint  # Automatically B.adjoint @ A.adjoint
```

### Null Space ker(G)

**Theory:** {m ∈ M : G(m) = 0}
**Code:** `G.kernel()` returns subspace projector
**Implementation:** Via SVD for matrix operators, iterative for large

### Range im(G)

**Theory:** {G(m) : m ∈ M} ⊆ D
**Code:** `G.range()` returns subspace projector
**Implementation:** Randomized methods for large operators

---

## Support Functions

**Theory Reference:** theory.txt §2, Rockafellar Convex Analysis

### Definition

**Theory:** σ_S(q) = sup_{x∈S} ⟨q, x⟩
**Code:** `SupportFunction` abstract base class
**File:** `pygeoinf/convex_analysis.py`

**Abstract Interface:**
```python
class SupportFunction(NonLinearForm, ABC):
    def _mapping(self, q: Vector) -> float:
        """Compute σ_S(q). Returns +∞ if q outside domain."""

    def support_point(self, q: Vector) -> Vector:
        """Return x* ∈ ∂σ(q) = argmax_{x∈S} ⟨q,x⟩."""
```

**Mathematical Properties (must verify in tests):**
1. **Convex:** σ(λq₁ + (1-λ)q₂) ≤ λσ(q₁) + (1-λ)σ(q₂)
2. **Positive Homogeneous:** σ(tq) = t·σ(q) for t ≥ 0
3. **Subgradient:** x* = support_point(q) satisfies ⟨q, x*⟩ = σ(q)

### Ball Support Function

**Theory:** For B = {x : ‖x - c‖ ≤ r},
σ_B(q) = ⟨q, c⟩ + r·‖q‖

**Code:** `BallSupportFunction(domain, center, radius)`
**Implementation:**
```python
def _mapping(self, q):
    return self.domain.inner_product(q, self.center) + \
           self.radius * self.domain.norm(q)

def support_point(self, q):
    q_norm = self.domain.norm(q)
    if q_norm < 1e-14:  # Handle q=0
        return self.center
    return self.center + self.radius * (q / q_norm)
```

**Edge Cases:**
- q=0: σ(0) = ⟨0, c⟩ = 0 if c is center (usually)
- Large r: No overflow protection (recommend r < 1e10)

### Ellipsoid Support Function

**Theory:** For E = {x : ⟨A(x-c), x-c⟩ ≤ r²},
σ_E(q) = ⟨q, c⟩ + r·‖A^{-1/2} R(q)‖
where R is Riesz map M → M*

**Code:** `EllipsoidSupportFunction(domain, center, radius, operator)`
**Requirements:** `operator` must support `.inverse.sqrt`

**Implementation:**
```python
def _mapping(self, q):
    qp = self.domain.to_dual(q)  # Riesz map
    A_inv_sqrt_qp = self.operator.inverse.sqrt(qp)
    norm_term = self.domain.dual.norm(A_inv_sqrt_qp)
    return self.domain.inner_product(q, self.center) + \
           self.radius * norm_term
```

**Caution:** Requires operator A to be positive-definite and invertible

### Half-Space Support Function

**Theory:** For H = {x : ⟨a, x⟩ ≤ b},
σ_H(q) = α·b when q = α·a with correct sign constraint, else +∞

**Code:** `HalfSpaceSupportFunction(normal, offset, inequality_type)`
**File:** `pygeoinf/convex_analysis.py` lines 194-350

**Implementation Details:**
- inequality_type='leq': Returns finite for α ≥ 0 (q = αa, α ≥ 0)
- inequality_type='geq': Returns finite for α ≤ 0 (q = αa, α ≤ 0)
- Returns +∞ for q not parallel to normal or wrong sign

**Edge Cases:**
- Unbounded direction: σ(q) = +∞ for most q
- Bounded direction: σ(q) = α·b for q = α·a, correct sign

---

## Dual Master Equation

**Theory Reference:** theory.txt §2, Al-Attar & Crawford (2021)

### Master Equation (Hilbert Form)

**Theory (Equation 2.1):**
```latex
h_U(q) = inf_{λ ∈ D} { ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ) }
```

Where:
- U = T(F) - admissible property set
- F = B ∩ G⁻¹(d̃ - V) - feasible model set
- B ⊆ M - model prior set (convex)
- V ⊆ D - data error set (convex)
- σ_B, σ_V - support functions of B, V
- d̃ - observed data vector

### Code Implementation

**Class:** `DualMasterCostFunction` in `backus_gilbert.py`

**Constructor:**
```python
cost = DualMasterCostFunction(
    forward_operator,        # G: M → D (LinearOperator)
    property_operator,       # T: M → P (LinearOperator)
    observed_data,           # d̃ ∈ D (np.ndarray)
    model_prior_support,     # σ_B (SupportFunction)
    data_error_support,      # σ_V (SupportFunction)
)
```

**Usage Workflow:**
```python
# 1. Fix query direction q ∈ P
cost.set_direction(q)

# 2. Define cost function φ_q(λ) for optimization
def phi_q(lambda_):
    return cost._mapping(lambda_)

# 3. Compute subgradient
subgrad = cost._subgradient(lambda_)

# 4. Minimize over λ using SubgradientDescent
from pygeoinf import SubgradientDescent
solver = SubgradientDescent(step_size=0.01, max_iter=1000)
result = solver.solve(cost, lambda_init)

# 5. Extract support value
h_U_q = result.best_value  # inf_λ φ_q(λ)
lambda_star = result.best_point  # Optimal certificate
```

**Implementation Details:**
```python
def _mapping(self, lambda_):
    # φ_q(λ) = ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ)
    term1 = data_space.inner_product(lambda_, observed_data)

    residual = property_operator.adjoint(q) - forward_operator.adjoint(lambda_)
    term2 = model_prior_support._mapping(residual)

    term3 = data_error_support._mapping(-lambda_)

    return term1 + term2 + term3

def _subgradient(self, lambda_):
    # ∂φ_q(λ) = d̃ - G*(∂σ_B) - ∂σ_V(-λ)
    residual = property_operator.adjoint(q) - forward_operator.adjoint(lambda_)
    subgrad_B = model_prior_support.support_point(residual)
    subgrad_V = data_error_support.support_point(-lambda_)

    return observed_data - forward_operator.adjoint(subgrad_B) - subgrad_V
```

**Mathematical Guarantees:**
- φ_q(λ) is convex in λ (composition of convex functions)
- Subgradient ∈ ∂φ_q(λ) (verified via support function subgradient rules)
- h_U(q) = inf_λ φ_q(λ) characterizes admissible set boundary

**Development Status:** Phase 3 complete (implemented), Phase 4-5 in progress (solver integration)

---

## Convex Sets

**Theory Reference:** theory.txt §3, subsets.py module

### Convex Subset Base Class

**Theory:** S ⊆ M convex ⇔ λx + (1-λ)y ∈ S for all x,y ∈ S, λ ∈ [0,1]
**Code:** `ConvexSubset(domain)` abstract class
**File:** `pygeoinf/subsets.py`

**Key Interface:**
```python
class ConvexSubset(SublevelSet):
    def is_element(self, x, rtol=1e-6) -> bool:
        """Check if x ∈ S."""

    @property
    def support_function(self) -> SupportFunction:
        """Lazy-computed support function σ_S."""

    @property
    def boundary(self) -> Subset:
        """Return ∂S (boundary set)."""
```

### Ball

**Theory:** B = {x : ‖x - c‖ ≤ r}
**Code:** `Ball(domain, center, radius)`
**Actually:** Implemented via `NormalisedEllipsoid` with identity operator

**Support Function:** See BallSupportFunction above

### Ellipsoid

**Theory:** E = {x : ⟨A(x-c), x-c⟩ ≤ r²}
**Code:** `Ellipsoid(domain, center, radius, operator)`

**Requirements:**
- operator must be positive-definite, self-adjoint
- Defines quadratic form via inner product

**Support Function:** See EllipsoidSupportFunction above

**Related:** `EllipsoidSurface` for boundary ∂E

### HalfSpace

**Theory:** H = {x : ⟨a, x⟩ ≤ b}
**Code:** `HalfSpace(domain, normal_vector, offset, inequality_type='leq')`
**File:** `pygeoinf/subsets.py` lines 1275-1488

**Inequality Types:**
- 'leq': ⟨a, x⟩ ≤ b
- 'geq': ⟨a, x⟩ ≥ b

**Implementation:**
```python
def is_element(self, x, rtol=1e-6):
    value = self.domain.inner_product(self.normal_vector, x)
    if self.inequality_type == 'leq':
        return value <= self.offset + rtol
    else:
        return value >= self.offset - rtol
```

**Support Function:** Returns +∞ for unbounded directions (see HalfSpaceSupportFunction)

### PolyhedralSet

**Theory:** P = ∩ⱼ Hⱼ where Hⱼ are half-spaces
**Code:** `PolyhedralSet(domain, half_spaces)`
**File:** `pygeoinf/subsets.py` lines 1490-1634

**Constructor:**
```python
half_spaces = [
    HalfSpace(space, a1, b1, 'leq'),
    HalfSpace(space, a2, b2, 'leq'),
    ...
]
poly = PolyhedralSet(space, half_spaces)
```

**Membership:** x ∈ P ⇔ x ∈ Hⱼ for all j

**Support Function:** (Future) Should use infimal convolution, currently not optimized

### Affine Subspace

**Theory:** A = x₀ + V where V is linear subspace
**Code:** `AffineSubspace(point, projector)`
**File:** `pygeoinf/subspaces.py`

**Constructor:**
```python
# V defined by orthogonal projector P: M → M
P = OrthogonalProjector(space, range_basis)
A = AffineSubspace(x0, P)
```

**Membership:** x ∈ A ⇔ P(x - x₀) = x - x₀

**Support Function:**
```python
σ_A(ξ) = {
    ⟨ξ, x₀⟩  if ξ ∈ (ker P)^⊥ = range(P*)
    +∞       otherwise
}
```

**Caution:** Enforces hard constraint in dual space (rank-deficient systems)

---

## Optimization

**Theory Reference:** theory.txt §2.1, convex_optimisation.py

### Subgradient Method

**Theory:** Non-smooth convex optimization
x_{k+1} = x_k - α_k g_k, where g_k ∈ ∂f(x_k)

**Code:** `SubgradientDescent(step_size, max_iter)`
**File:** `pygeoinf/convex_optimisation.py`

**Usage:**
```python
solver = SubgradientDescent(step_size=0.01, max_iter=1000)
result = solver.solve(nonlinear_form, x_init)

print(result.best_value)    # min f(x) found
print(result.best_point)    # x* achieving minimum
print(result.num_iterations)  # Actual iterations
print(result.converged)     # Convergence flag
```

**Requirements:**
- `nonlinear_form` must implement `._mapping(x)` and `._subgradient(x)`
- Subgradient must be in ∂f(x) (non-empty)
- Convexity of f (otherwise no guarantees)

**Convergence:**
- Non-monotonic: f(x_k) may increase
- Tracks best value seen: result.best_value ≤ f(x_k) for all k
- Constant step: ε-optimal with sufficient iterations
- Diminishing step: α_k → 0, Σα_k = ∞ ⇒ converges (not yet implemented)

**Phase 4 Status:**
- ✅ Phase 4.1: Basic constant step implementation
- ⏸️ Phase 4.2: Diminishing step rules (α_k = 1/k, Polyak step)
- ⏸️ Phase 4.3: Integration helpers (solve_for_support_value)
- ⏸️ Phase 4.4: Advanced methods (bundle, proximal)

---

### Bundle Methods

**Theory:** Level bundle methods for non-smooth convex optimization
Build cutting-plane model: ˇf_k(x) = max_{j ∈ J_k} {f(x_j) + ⟨g_j, x - x_j⟩}
Solve QP: x_{k+1} = argmin_{x : ˇf_k(x) ≤ f_lev_k} ||x - ˆx_k||²

**Code:** `BundleMethod(oracle, alpha, max_iterations, bundle_size)`
**File:** `pygeoinf/convex_optimisation.py`

**Usage:**
```python
solver = BundleMethod(nonlinear_form, alpha=0.1, max_iterations=500)
result = solver.solve(x_init)

print(result.gap)        # Δ_k = f_up - f_low (optimality certificate)
print(result.converged)  # True if Δ_k ≤ tolerance
print(result.num_serious_steps)  # Stability center updates
```

**Requirements:**
- `nonlinear_form` must implement `._mapping(x)` and `.subgradient(x)`
- Quadratic programming solver (scipy.optimize.minimize or cvxpy)
- Feasible set X (default: no constraints)

**Key Concepts:**
- **Bundle:** Collection of cuts (x_j, f(x_j), g_j) from past oracle calls
- **Cutting-plane model:** Piecewise-linear underestimator ˇf_k(x) ≤ f(x)
- **Stability center:** ˆx_k = best point found (argmin f(x_j) over bundle)
- **Level set:** L_k = {x : ˇf_k(x) ≤ f_lev_k} where f_lev_k = α·f_low + (1-α)·f_up
- **Serious step:** f(x_{k+1}) < f(ˆx_k) - α·Δ_k → update ˆx_{k+1} = x_{k+1}
- **Null step:** Add cut to bundle, keep ˆx_{k+1} = ˆx_k

**Convergence:**
- Monotonic descent of upper bound: f_up_k ↓ f*
- Gap-based stopping: Δ_k ≤ tol guarantees |f(ˆx_k) - f*| ≤ tol
- Automatic step sizing (implicit in QP, no manual tuning)
- Stability center prevents oscillation

**Advantages over Subgradient:**
- No step size tuning required
- Better descent through model-based search direction
- Reliable termination criterion (gap Δ_k)
- Accumulates information (bundle not discarded)
- Natural handling of constraints via QP

**Status:**
- 🔲 Future: Bundle methods implementation (see plan: bundle-methods-optimizer-plan.md)
- 📊 Theory: theory/bundle_methods.pdf (asynchronous level methods)
- 🎯 Target: Replace SubgradientDescent for dual master problems (higher accuracy)

---

## Forward Problems

**Theory Reference:** theory.txt §1 eq:intro-observation

### Observation Model

**Theory:** d̃ = G(m̄) + η where η ~ noise distribution
**Code:** `ForwardProblem` base class
**File:** `pygeoinf/forward_problem.py`

**Linear Case:**
```python
problem = LinearForwardProblem(
    model_space,          # HilbertSpace
    data_space,           # HilbertSpace
    forward_operator,     # G: M → D (LinearOperator)
    data_error_measure    # p(η) (GaussianMeasure, optional)
)
```

**Attributes:**
- `problem.model_space` - M
- `problem.data_space` - D
- `problem.forward_operator` - G
- `problem.data_error_measure` - Noise model (if Gaussian)

### Gaussian Noise Model

**Theory:** η ~ N(0, Γ) where Γ is covariance
**Code:** `GaussianMeasure(data_space, mean=0, covariance=Gamma)`
**File:** `pygeoinf/gaussian_measure.py`

**Confidence Set:**
```python
# V = {η : ‖η‖_Γ⁻¹ ≤ χ²_{Nd}(1-α)}
V = data_error_measure.confidence_ellipsoid(alpha=0.05)  # 95% confidence
```

**Support Function:**
```python
# For V = {η : ‖Γ^{-1/2}η‖ ≤ r}
sigma_V = EllipsoidSupportFunction(
    data_space,
    center=data_space.zero(),
    radius=chi2_quantile,
    operator=Gamma
)
```

---

## Inversion Methods

### Bayesian Inversion

**Theory Reference:** Stuart (2010), linear_bayesian.py

**Prior:** m ~ N(m₀, C_prior)
**Likelihood:** d = G(m) + η, η ~ N(0, Γ)
**Posterior:** m|d ~ N(m_post, C_post)

**Code:**
```python
inversion = LinearBayesianInversion(
    forward_problem,
    prior_measure
)

posterior = inversion.posterior_measure  # GaussianMeasure
m_map = posterior.mean                   # MAP estimate
C_post = posterior.covariance_operator   # Posterior covariance
```

**Formula (from theory):**
- m_post = m₀ + C_prior G* (G C_prior G* + Γ)⁻¹ (d - G m₀)
- C_post = C_prior - C_prior G* (G C_prior G* + Γ)⁻¹ G C_prior

**Code Implementation:** Uses normal operator and Kalman gain

### Least Squares Inversion

**Theory Reference:** Tikhonov regularization

**Formulation:** min ‖G m - d‖²_Γ⁻¹ + α²‖m - m₀‖²

**Code:**
```python
inversion = LinearLeastSquaresInversion(forward_problem)
m_solution = inversion.solve(data_vector, prior_mean, regularization=alpha)
```

### Minimum Norm Inversion

**Theory Reference:** Discrepancy principle

**Formulation:** min ‖m‖ subject to ‖G m - d‖ ≤ τ

**Code:**
```python
inversion = LinearMinimumNormInversion(forward_problem)
m_solution = inversion.solve(data_vector, tolerance=tau)
```

---

## Notation Translation

Quick reference for theory ↔ code notation:

| Math (LaTeX) | Theory.txt | Python Code | Notes |
|--------------|-----------|-------------|-------|
| M | `\modelspace` | `model_space` | HilbertSpace instance |
| D | `\dataspace` | `data_space` | EuclideanSpace(N_d) |
| P | `\propertyspace` | `property_space` | EuclideanSpace(N_p) |
| G | G | `forward_operator` | LinearOperator |
| T | `\Tau` | `property_operator` | LinearOperator |
| G* | G* | `.adjoint` | Hilbert adjoint |
| G' | G' | `.dual` | Banach dual |
| ⟨x, y⟩_M | - | `space.inner_product(x, y)` | Inner product |
| ‖x‖_M | - | `space.norm(x)` | Norm |
| σ_S(q) | `\sigma_{S}` | `support_fn._mapping(q)` | Support function |
| ∂σ(q) | - | `support_fn.support_point(q)` | Subgradient |
| ker(G) | `\ker G` | `G.kernel()` | Null space |
| im(G) | `\im G` | `G.range()` | Image/range |
| B | `\Bset` | `model_prior_set` | ConvexSubset |
| V | `\Vdata` | `data_error_set` | ConvexSubset |
| d̃ | `\dt` | `observed_data` | np.ndarray |
| U | `\Uset` | (computed) | Admissible property set |
| F | `\Fset` | (computed) | Feasible model set |

---

## Agent Checklists

Use these checklists when implementing or reviewing mathematical code.

### For HilbertSpace Subclasses

- [ ] Riesz map preserves inner product: ⟨x, y⟩ = ⟨Rx, y⟩_dual
- [ ] `to_components` / `from_components` are inverses
- [ ] Random vectors span the space (test with Gram-Schmidt)
- [ ] Coordinate projection / inclusion are adjoints
- [ ] Reference theory.txt §1 or relevant paper in docstring

### For LinearOperator

- [ ] Adjoint satisfies ⟨Ax, y⟩ = ⟨x, A*y⟩ (test with random vectors, 10 trials)
- [ ] Domain and codomain are correct HilbertSpace instances
- [ ] Composition preserves adjoint: (A @ B)* = B* @ A*
- [ ] Matrix representation (if finite) matches mapping
- [ ] Kernel and range methods consistent with SVD
- [ ] Reference theory.txt §1-2 or specific paper

### For SupportFunction

- [ ] Convex: σ(λq₁ + (1-λ)q₂) ≤ λσ(q₁) + (1-λ)σ(q₂) (test 100 random cases)
- [ ] Positively homogeneous: σ(tq) = t·σ(q) for t > 0
- [ ] Returns +∞ for q outside domain (if unbounded set)
- [ ] `support_point(q)` satisfies ⟨q, x*⟩ = σ(q) (subgradient condition)
- [ ] Handle q=0 case: σ(0) = 0 iff 0 ∈ S
- [ ] Reference theory.txt §2 and Rockafellar Convex Analysis

### For ConvexSubset

- [ ] `is_element()` respects convex combinations (test random λ, x, y)
- [ ] `support_function` property returns valid SupportFunction
- [ ] `boundary` property returns lower-dimensional set
- [ ] CSG operations (intersect, union, complement) preserve topology
- [ ] Reference theory.txt §3

### For Optimization Solvers

- [ ] Converges on convex problems (test with quadratics)
- [ ] Subgradient ∈ subdifferential at each iteration
- [ ] Handles +∞ function values gracefully (constraint violations)
- [ ] Returns convergence diagnostics (best value, iterations, residuals)
- [ ] Non-monotonic for subgradient methods (track best value separately)
- [ ] Reference theory.txt §2.1 or convergence theorem from papers

---

## Using This Map

**For Oracle-subagent:**
Consult relevant sections when researching. Cross-reference theory.txt equation numbers with code modules.

**For Sisyphus-subagent:**
Before implementing, read theory section. Add LaTeX docstring linking to theory.txt. Follow agent checklists.

**For Code-Review-subagent:**
Verify implementations against checklists. Ensure theory references are accurate and complete.

**For Theory-Validator-subagent:**
Use notation table to parse LaTeX. Check properties from agent checklists. Validate edge cases.

**For Users:**
Navigate from mathematical concept to implementation. Understand assumptions and limitations.

---

**Maintenance:** Update this document when:
- New mathematical classes are added
- Theory documents are updated (theory.txt changes)
- Papers are added to theory/ directory
- Implementation details change (API refactoring)

Report issues or suggest improvements via project issue tracker.
