# Convex Analysis Extension for pygeoinf

**Date**: 19 January 2026
**Branch**: `convex_analysis`
**Reference Document**: `/disks/data/PhD/Inferences/convex_analysis.txt`

## Executive Summary

The convex analysis framework documented in `convex_analysis.txt` represents a **highly compatible and valuable extension** for pygeoinf. Recent developments (subsets module, inference base classes) have built **75-80% of the required infrastructure**. The remaining work involves implementing:

1. Support function computational machinery
2. Master dual equation solvers
3. DLI/Backus-Gilbert specific algorithms

This document provides complete context for continuing this work.

---

## Background

### What is pygeoinf?

A Python library for solving geophysical inverse problems using:
- **Abstract Hilbert space formulation**: `HilbertSpace`, `LinearOperator`, `GaussianMeasure`
- **Bayesian inversion**: Posterior distributions via Kalman operators
- **Optimization methods**: Tikhonov regularization, minimum-norm solutions
- **Coordinate-free design**: Mathematics first, NumPy arrays second

**Current Version**: 1.3.9
**Repository**: https://github.com/da380/pygeoinf

### What is the Convex Analysis Framework?

A rigorous mathematical framework for **Data-driven Localized Inference (DLI)** that:

1. **Generalizes inference settings**: Works in Banach/Hilbert spaces
2. **Introduces property inference**: Maps model space → property space via operator $\mathcal{T}$
3. **Uses support functions**: Characterizes admissible property sets via dual formulation
4. **Handles convex constraints**: Arbitrary model priors $\mathcal{B}$ and data-confidence sets $\mathcal{V}_{\mathcal{D}}$
5. **Unifies classical methods**: Shows Backus-Gilbert, SOLA, DLI emerge from master equation

**Key Innovation**: The **Master Dual Equation**

```
h_U(q) = inf_{λ ∈ D'} { ⟨λ, d̃⟩ + σ_B(T'q - G'λ) + σ_V(-λ) }
```

Where:
- `h_U(q)`: Support function of admissible property set
- `σ_B`: Support function of model constraint set
- `σ_V`: Support function of data confidence set
- `T`: Property operator
- `G`: Forward operator

---

## Current State Analysis

### Recent Major Additions (Since Initial Review)

#### 1. Complete `subsets.py` Module (846 lines)

**What's Implemented:**

```python
# Base hierarchy
Subset (ABC)
├── EmptySet / UniversalSet
├── LevelSet (f(x) = c)
│   └── EllipsoidSurface → Sphere
├── SublevelSet (f(x) ≤ c)
│   ├── ConvexSubset
│   │   ├── Ellipsoid → Ball
│   │   ├── NormalisedEllipsoid
│   │   └── ConvexIntersection
│   └── Generic sublevel sets
├── Intersection (CSG)
├── Union (CSG)
└── Complement (S^c)
```

**Key Features:**
- ✅ Convex sets defined by functionals
- ✅ Ellipsoid/Ball with quadratic forms `⟨A(x-c), x-c⟩ ≤ r²`
- ✅ Intersection via max-functional: `F(x) = max_i(f_i(x) - c_i)`
- ✅ CSG operations (Union, Intersection, Complement)
- ✅ Membership testing with tolerances
- ✅ Convexity verification tools

**Maps to convex_analysis.txt:**
- Model constraint sets $\mathcal{B}$: `ConvexSubset`, `Ellipsoid`, `Ball`
- Norm balls: `{m : ∥m-m₀∥ ≤ M}` → `Ball(domain, m0, M, identity)`
- Intersections of constraints → `ConvexIntersection`

#### 2. Property Inference Framework

**Added to `inversion.py`:**

```python
class Inference(Inversion):
    """Base class with property_operator: M → P"""
    def __init__(self, forward_problem, property_operator):
        self._property_operator = property_operator

    @property
    def property_space(self) -> HilbertSpace:
        return self.property_operator.codomain

class LinearInference(Inference):
    """For linear property operators T: M → P"""
```

**Significance:**
- ✅ Property space $\mathcal{P}$ now formalized
- ✅ Property operator $\mathcal{T}$ integrated into class hierarchy
- ✅ Framework ready for localized inference methods

#### 3. Existing Infrastructure (Already Present)

| Component | Implementation | Status |
|-----------|----------------|--------|
| Model space $\mathcal{M}$ | `HilbertSpace` | ✅ Complete |
| Data space $\mathcal{D}$ | `HilbertSpace` | ✅ Complete |
| Forward operator $G$ | `LinearOperator` with adjoint $G^*$ | ✅ Complete |
| Gaussian measures | `GaussianMeasure` with covariance | ✅ Complete |
| Ellipsoidal sets | `Ellipsoid`, `Ball` in `subsets.py` | ✅ Complete |
| Inner products | `HilbertSpace.inner_product` | ✅ Complete |
| Optimization tools | `ScipyUnconstrainedOptimiser` | ✅ Complete |

---

## Gap Analysis

### Critical Gaps (Must Implement)

#### 1. Support Functions

**What's Missing:**
```python
class ConvexSubset:
    def support_function(self, dual_vector) -> float:
        """
        Compute σ_S(ξ) = sup_{x ∈ S} ⟨ξ, x⟩

        For sublevel set S = {x : f(x) ≤ c}, need to solve:
            max_x  ⟨ξ, x⟩
            s.t.   f(x) ≤ c
        """
        pass
```

**Needed Implementations:**
- `Ellipsoid.support_function`: Analytical formula for `σ_{Ball}(ξ) = r·∥ξ∥ + ⟨ξ, c⟩`
- `ConvexIntersection.support_function`: Via infimal convolution
- `Ball.support_function`: Simplified for isotropic case
- Generic numerical solver for general `ConvexSubset`

**Priority**: **CRITICAL** - Unlocks entire framework

#### 2. Master Dual Equation Solver

**What's Missing:**
```python
class DualLocalizedInference(LinearInference):
    """
    Implements the master dual support equation.

    Computes directional bounds on admissible property set U:
        h_U(q) = inf_{λ ∈ D} { ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ) }
    """

    def __init__(self,
                 forward_problem: LinearForwardProblem,
                 property_operator: LinearOperator,
                 model_constraint: ConvexSubset,
                 data_confidence: ConvexSubset):
        super().__init__(forward_problem, property_operator)
        self.model_constraint = model_constraint
        self.data_confidence = data_confidence

    def directional_support(self, direction, data) -> float:
        """Solve the dual optimization problem."""
        pass

    def compute_certificate(self, direction, data):
        """Return optimal λ*(q) for direction q."""
        pass

    def membership_test(self, property_value, data, *, n_directions=100):
        """Test if p ∈ U via separating hyperplanes."""
        pass
```

**Implementation Strategy:**
1. Define objective function: `φ(λ; q) = ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)`
2. Use `ScipyUnconstrainedOptimiser` with gradient
3. Gradient requires `∂σ_B` (subgradient of support function)
4. For smooth cases, can use second-order methods

**Priority**: **CRITICAL** - Core algorithmic contribution

### Medium Priority Gaps

#### 3. DLI Ellipsoid Constructors

**What's Needed:**

From convex_analysis.txt Section 8-9, implement:

```python
class DLIEllipsoid:
    """
    Compute DLI ellipsoid for noiseless data with norm ball prior.

    Mathematical Form (from text):
        U_DLI = {p ∈ P̃ + im(H) : ⟨H†(p - P̃), p - P̃⟩ ≤ ρ²}

    Where:
        - P̃ = T(m̃), m̃ = minimum-norm solution to Gm = d
        - H = T ∘ P_ker(G) ∘ T*  (resolution operator)
        - ρ² = M² - ∥m̃∥²  (feasibility radius)
    """

    @staticmethod
    def from_noiseless_data(forward_operator: LinearOperator,
                           property_operator: LinearOperator,
                           data,
                           model_radius: float) -> Ellipsoid:
        """
        Construct DLI ellipsoid for case:
            - B = {m : ∥m∥ ≤ M}
            - V = {0}  (noiseless)
            - m₀ = 0   (centered prior)

        Returns Ellipsoid in property space.
        """
        pass

    @staticmethod
    def from_noisy_data(forward_operator: LinearOperator,
                       property_operator: LinearOperator,
                       data,
                       model_radius: float,
                       model_center,
                       data_covariance) -> Ellipsoid:
        """
        General case with:
            - B = {m : ∥m - m₀∥ ≤ M}
            - V = Mahalanobis ellipsoid
        """
        pass
```

**References in convex_analysis.txt:**
- Lines 1000-1200: Noiseless case derivation
- Lines 700-900: Noisy case with Mahalanobis sets

**Priority**: **MEDIUM** - High scientific value

#### 4. Backus-Gilbert / SOLA Implementation

**Current State**: `backus_gilbert.py` has only `HyperEllipsoid` (basic)

**What's Needed:**

```python
class SOLAEstimator(LinearInference):
    """
    SOLA (Subtractive Optimally Localized Averages) estimator.

    From convex_analysis.txt Section 12-13:
        X_SOLA = T·G*·(αC_D + βGG*)^(-1)

    Provides linear mapping from data to properties with:
        - Resolution control via averaging kernels
        - Noise propagation analysis
        - Trade-off parameters α (data weight), β (resolution weight)
    """

    def __init__(self,
                 forward_problem: LinearForwardProblem,
                 property_operator: LinearOperator,
                 alpha: float,  # data weight
                 beta: float):  # resolution weight
        super().__init__(forward_problem, property_operator)
        self.alpha = alpha
        self.beta = beta

    def compute_estimator(self) -> LinearOperator:
        """Return X: D → P mapping."""
        pass

    def averaging_kernel(self) -> LinearOperator:
        """Return A = X·G: M → P (resolution operator)."""
        pass

    def resolution_analysis(self, target_locations):
        """Compute averaging kernel values at target points."""
        pass

    def variance_analysis(self, data_covariance):
        """Propagate data uncertainty to property estimates."""
        pass
```

**References in convex_analysis.txt:**
- Lines 982-1000: Link to Backus-Gilbert
- Lines 1175-1260: SOLA from dual support
- Lines 1262-1356: SOLA with unimodularity

**Priority**: **MEDIUM** - Widely used in geophysics

### Low Priority / Nice-to-Have

#### 5. Advanced Features

- **Quadratic surrogate bounds**: Conservative outer approximations (lines 741-900)
- **Directional sampling**: Sample support function on sphere for visualization
- **Polyhedral approximations**: Outer bounds via finite direction samples
- **Certificate analysis**: Study optimal dual variables λ*(q)
- **Parallel directional solver**: Compute multiple directions simultaneously

---

## Mathematical Correspondence

### Notation Mapping

| convex_analysis.txt | pygeoinf | Notes |
|---------------------|----------|-------|
| $\mathcal{M}$ (model space) | `HilbertSpace` (model_space) | |
| $\mathcal{D}$ (data space) | `HilbertSpace` (data_space) | Typically `EuclideanSpace` |
| $\mathcal{P}$ (property space) | `HilbertSpace` (property_space) | Via `Inference.property_space` |
| $G : \mathcal{M} \to \mathcal{D}$ | `LinearOperator` (forward_operator) | |
| $\mathcal{T} : \mathcal{M} \to \mathcal{P}$ | `LinearOperator` (property_operator) | Via `Inference` |
| $G^* : \mathcal{D} \to \mathcal{M}$ | `forward_operator.adjoint` | |
| $\mathcal{T}^* : \mathcal{P} \to \mathcal{M}$ | `property_operator.adjoint` | |
| $\mathcal{B} \subset \mathcal{M}$ | `ConvexSubset` | Model constraint |
| $\mathcal{V}_{\mathcal{D}} \subset \mathcal{D}$ | `ConvexSubset` | Data confidence |
| $\mathcal{U} \subset \mathcal{P}$ | Target to compute | Admissible property set |
| $\sigma_{\mathcal{B}}(\xi)$ | `model_constraint.support_function(xi)` | **TO IMPLEMENT** |
| $\sigma_{\mathcal{V}}(\lambda)$ | `data_confidence.support_function(lam)` | **TO IMPLEMENT** |
| $h_{\mathcal{U}}(q)$ | `directional_support(q)` | **TO IMPLEMENT** |
| $\langle \cdot, \cdot \rangle_{\mathcal{M}}$ | `model_space.inner_product(·, ·)` | |
| $\\|\cdot\\|_{\mathcal{M}}$ | `model_space.norm(·)` | |

### Key Formulas to Implement

#### 1. Master Dual Equation (Hilbert form)

**From convex_analysis.txt line 500:**

```
h_U(q) = inf_{λ ∈ D} { ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ) }
```

**Implementation:**
```python
def directional_support(self, q, data):
    G = self.forward_problem.forward_operator
    T = self.property_operator

    def objective(lambda_components):
        lam = self.data_space.from_components(lambda_components)

        # Term 1: ⟨λ, d̃⟩
        term1 = self.data_space.inner_product(lam, data)

        # Term 2: σ_B(T*q - G*λ)
        residual = self.model_space.subtract(
            T.adjoint(q),
            G.adjoint(lam)
        )
        term2 = self.model_constraint.support_function(residual)

        # Term 3: σ_V(-λ)
        neg_lam = self.data_space.multiply(-1, lam)
        term3 = self.data_confidence.support_function(neg_lam)

        return term1 + term2 + term3

    # Minimize using scipy
    result = minimize(objective, x0=initial_lambda_components)
    return result.fun
```

#### 2. Support Function for Ellipsoid

**From convex_analysis.txt (Mahalanobis ellipsoid, line 707):**

For $\mathcal{V} = \\{\eta : \frac{1}{2}\langle C^{-1}\eta, \eta \rangle \leq s^2\\}$:

```
σ_V(λ) = √2 · s · ∥C^{1/2}λ∥
```

**Implementation:**
```python
class MahalanobisEllipsoid(Ellipsoid):
    def support_function(self, dual_vector):
        """
        For V = {x : (1/2)⟨C^{-1}x, x⟩ ≤ s²}
        Returns σ_V(ξ) = √2·s·∥C^{1/2}ξ∥
        """
        # C = operator (covariance)
        # radius = s (confidence level)
        sqrt_C_xi = self.operator.sqrt()(dual_vector)
        norm = self.domain.norm(sqrt_C_xi)
        return np.sqrt(2) * self.radius * norm
```

#### 3. Support Function for Norm Ball

**Standard formula:**

For $\mathcal{B} = \\{m : \\|m - m_0\\| \leq M\\}$:

```
σ_B(ξ) = M·∥ξ∥ + ⟨ξ, m₀⟩
```

**Implementation:**
```python
class Ball(Ellipsoid):
    def support_function(self, dual_vector):
        """
        For B = {m : ∥m - m₀∥ ≤ M}
        Returns σ_B(ξ) = M·∥ξ∥ + ⟨ξ, m₀⟩
        """
        norm_xi = self.domain.norm(dual_vector)
        inner = self.domain.inner_product(dual_vector, self.center)
        return self.radius * norm_xi + inner
```

#### 4. DLI Ellipsoid (Noiseless, Zero Center)

**From convex_analysis.txt lines 1150-1174:**

```
U_DLI = {p ∈ P̃ + im(H) : ⟨H†(p - P̃), p - P̃⟩ ≤ ρ²}

Where:
    P̃ = T(m̃)  (m̃ = min-norm solution)
    H = T ∘ P_ker(G) ∘ T*
    ρ² = M² - ∥m̃∥²
```

**Implementation:**
```python
def compute_dli_ellipsoid(G, T, data, M):
    # 1. Minimum-norm solution
    m_tilde = G.minimum_norm_inverse()(data)
    p_tilde = T(m_tilde)

    # 2. Kernel projector
    P_ker = G.kernel_projector()

    # 3. Resolution operator H = T ∘ P_ker ∘ T*
    H = T @ P_ker @ T.adjoint

    # 4. Radius
    rho_squared = M**2 - model_space.norm(m_tilde)**2
    if rho_squared < 0:
        raise ValueError("Infeasible: data inconsistent with prior")
    rho = np.sqrt(rho_squared)

    # 5. Construct ellipsoid
    return Ellipsoid(
        domain=property_space,
        center=p_tilde,
        radius=rho,
        operator=H.pseudo_inverse()  # H†
    )
```

---

## Implementation Roadmap

### Phase 1: Support Functions (Week 1-2)

**Tasks:**
1. Add `support_function` abstract method to `Subset` base class
2. Implement for `Ball`:
   ```python
   σ_B(ξ) = M·∥ξ∥ + ⟨ξ, c⟩
   ```
3. Implement for `Ellipsoid` (general quadratic case)
4. Implement for `ConvexIntersection` via infimal convolution
5. Add unit tests comparing to analytical solutions

**Files to Modify:**
- `pygeoinf/subsets.py`
- `tests/test_subsets.py` (new)

**Deliverable**: All convex sets have computable support functions

### Phase 2: Master Dual Solver (Week 3-4)

**Tasks:**
1. Create new file `pygeoinf/convex_inference.py`
2. Implement `DualLocalizedInference` class
3. Implement directional support solver
4. Add gradient computation for support functions
5. Test on simple 2D examples (visualize property sets)

**Files to Create:**
- `pygeoinf/convex_inference.py`
- `tests/test_convex_inference.py`

**Deliverable**: Working directional bound solver

### Phase 3: DLI Ellipsoids (Week 5)

**Tasks:**
1. Add `LinearOperator.kernel_projector()` method if missing
2. Add `LinearOperator.minimum_norm_inverse()` method if missing
3. Implement `DLIEllipsoid` factory class
4. Add examples comparing to Bayesian posteriors

**Files to Modify:**
- `pygeoinf/linear_operators.py` (if needed)
- `pygeoinf/convex_inference.py`
- Create example notebook

**Deliverable**: DLI ellipsoid constructor

### Phase 4: Backus-Gilbert/SOLA (Week 6-7)

**Tasks:**
1. Extend `pygeoinf/backus_gilbert.py`
2. Implement `SOLAEstimator` class
3. Implement averaging kernel computation
4. Add resolution/variance analysis tools
5. Create tutorial notebook

**Files to Modify:**
- `pygeoinf/backus_gilbert.py`
- `tests/test_backus_gilbert.py`

**Deliverable**: Complete SOLA implementation

### Phase 5: Documentation & Examples (Week 8)

**Tasks:**
1. Write comprehensive docstrings
2. Create tutorial notebooks:
   - DLI vs Bayesian comparison
   - SOLA resolution analysis
   - Support function visualization
3. Update main README
4. Submit PR to develop branch

**Deliverable**: Publication-ready extension

---

## Technical Notes

### Computing Subgradients of Support Functions

For optimization, we need $\partial \sigma_S(\xi)$ where:

```
σ_S(ξ) = sup_{x ∈ S} ⟨ξ, x⟩
```

**Theorem (Subgradient)**: If $x^*$ achieves the supremum, then $x^* \in \partial \sigma_S(\xi)$.

**For Ellipsoid** $S = \\{x : \\|x - c\\| \leq r\\}$:

```python
def support_function_gradient(self, dual_vector):
    """∂σ_S(ξ) = {c + r·ξ/∥ξ∥}  (if ξ ≠ 0)"""
    norm = self.domain.norm(dual_vector)
    if norm < 1e-14:
        # Subgradient is the whole ball at ξ=0
        return self.center  # Return any element

    direction = self.domain.multiply(1.0/norm, dual_vector)
    offset = self.domain.multiply(self.radius, direction)
    return self.domain.add(self.center, offset)
```

### Mahalanobis Ellipsoid Details

The data confidence set in Hilbert form (convex_analysis.txt line 700):

```
V = {η ∈ D : (1/2)⟨C_D^{-1}η, η⟩ ≤ s²}
```

Can be represented as `Ellipsoid` with:
- `operator = C_D^{-1}` (precision)
- `radius = s·√2` (note the √2 factor!)
- `center = 0`

Or equivalently with:
- `operator = C_D` (covariance)
- Define support directly: `σ(λ) = √2·s·∥C_D^{1/2}λ∥`

### Connection to Existing GaussianMeasure

pygeoinf's `GaussianMeasure` already has:
- Covariance operator
- Covariance factor (Cholesky-like)

For data noise $\eta \sim \mathcal{N}(0, C_D)$, the $1-\alpha$ confidence set is:

```
V = {η : ∥C_D^{-1/2}η∥² ≤ χ²_{N_d}(1-α)}
```

So `s² = (1/2)·χ²_{N_d}(1-α)` for conversion.

---

## Key Design Decisions

### 1. Where to Place New Code?

**Option A**: Extend existing modules
- `subsets.py`: Add support functions
- `backus_gilbert.py`: Add SOLA/DLI
- `inversion.py`: Add dual inference base class

**Option B**: New module `convex_inference.py`
- Keep convex analysis methods together
- Cleaner separation of concerns
- Can import from other modules

**Recommendation**: **Option B** for phase 2-3, then integrate successful patterns into existing modules.

### 2. API Design Philosophy

Follow pygeoinf conventions:
- Return operators, not matrices
- Work with abstract vectors, not components
- Provide both low-level (operators) and high-level (inference classes) APIs
- Use type hints extensively
- Document with NumPy-style docstrings

### 3. Testing Strategy

For each new feature:
1. **Unit tests**: Test individual methods on `EuclideanSpace`
2. **Integration tests**: Combine with existing forward problems
3. **Numerical tests**: Verify against analytical solutions
4. **Comparison tests**: DLI vs Bayesian, SOLA vs least-squares
5. **Performance tests**: Scaling with dimension

---

## References in convex_analysis.txt

### Critical Sections

- **Lines 1-200**: Introduction, problem setup, notation
- **Lines 200-400**: Master dual equation derivation and properties
- **Lines 500-700**: Hilbert specialization
- **Lines 700-900**: Mahalanobis data sets, quadratic surrogate
- **Lines 1000-1174**: DLI ellipsoid derivation (noiseless)
- **Lines 1175-1260**: SOLA from dual support
- **Lines 1262-1356**: SOLA with unimodularity

### Example Applications

The text includes several worked examples that should be reproduced as tests/tutorials:
1. Noiseless DLI with norm ball prior
2. Noisy DLI with Mahalanobis data
3. SOLA estimator construction
4. Backus-Gilbert with resolution constraints

---

## Contact & Collaboration

**Original Analysis Date**: 19 January 2026
**Branch**: `convex_analysis`
**Status**: Infrastructure complete, algorithms pending

### Quick Start for New Session

1. **Review this document** to understand current state
2. **Check branch status**: `git status` in `/disks/data/PhD/Inferences/pygeoinf`
3. **Read convex_analysis.txt**: Start with introduction and Section 2
4. **Examine subsets.py**: Understand existing convex set infrastructure
5. **Run existing tests**: `pytest tests/` to ensure baseline
6. **Start with Phase 1**: Implement support functions

### Questions to Resolve

- [ ] Should support functions handle both smooth and non-smooth cases?
- [ ] Performance targets for directional bound solver?
- [ ] Integration with existing Bayesian infrastructure?
- [ ] Visualization tools for property sets in high dimensions?
- [ ] Parallelization strategy for multi-directional solvers?

---

## Appendix: Code Snippets

### A. Minimal Working Example (Target)

```python
import pygeoinf as pg
import numpy as np

# 1. Define spaces
model_space = pg.EuclideanSpace(100)
data_space = pg.EuclideanSpace(50)
property_space = pg.EuclideanSpace(10)

# 2. Define operators
G = pg.DenseMatrixLinearOperator(
    model_space, data_space,
    np.random.randn(50, 100)
)
T = pg.DenseMatrixLinearOperator(
    model_space, property_space,
    np.random.randn(10, 100)
)

# 3. Define constraints
model_prior = pg.Ball(model_space, center=model_space.zero, radius=5.0)
data_confidence = pg.Ball(data_space, center=data_space.zero, radius=1.0)

# 4. Synthetic data
true_model = model_space.random()
data = G(true_model) + 0.1 * data_space.random()

# 5. DLI inference (TARGET - NOT YET IMPLEMENTED)
dli = pg.DualLocalizedInference(
    forward_problem=pg.LinearForwardProblem(G),
    property_operator=T,
    model_constraint=model_prior,
    data_confidence=data_confidence
)

# 6. Compute admissible property set
directions = property_space.random_directions(n=100)
bounds = [dli.directional_support(q, data) for q in directions]

# 7. Get ellipsoidal approximation
property_ellipsoid = dli.outer_ellipsoid(data, directions)

print(f"Property center: {property_ellipsoid.center}")
print(f"Property radius: {property_ellipsoid.radius}")
```

### B. Support Function Template

```python
# In pygeoinf/subsets.py

class ConvexSubset(SublevelSet):
    # ... existing code ...

    @abstractmethod
    def support_function(self, dual_vector: Vector) -> float:
        """
        Compute the support function σ_S(ξ) = sup_{x∈S} ⟨ξ, x⟩.

        Args:
            dual_vector: An element ξ from the dual space (identified
                        with the primal via Riesz map in Hilbert spaces).

        Returns:
            float: The supremum value, or +∞ if unbounded.
        """
        pass

    def support_function_gradient(self, dual_vector: Vector) -> Vector:
        """
        Compute a subgradient of the support function.

        Returns an element x* ∈ S that achieves (or approximates)
        the supremum in σ_S(ξ).

        Args:
            dual_vector: Direction ξ

        Returns:
            Vector: An element of ∂σ_S(ξ)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide support "
            "function gradients"
        )
```

---

## Conclusion

The pygeoinf library is **architecturally ready** for the convex analysis extension. With ~75-80% of infrastructure in place, the remaining work is primarily:

1. **Algorithmic**: Implement support function computations
2. **Optimization**: Build dual solvers for master equation
3. **High-level**: Create convenience classes for DLI/SOLA

This represents a **natural evolution** of the library that will:
- Complement existing Bayesian methods with set-based uncertainty
- Unify classical geophysical inference methods (Backus-Gilbert, SOLA)
- Provide rigorous bounds on localized properties
- Maintain the elegant coordinate-free design philosophy

**Estimated Effort**: 6-8 weeks for full implementation and documentation.

**Scientific Impact**: High - bridges classical and modern inference methods in geophysics.
