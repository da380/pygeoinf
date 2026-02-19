# Theory Papers Index

**Last Updated:** 2026-02-17
**Purpose:** Map papers in `pygeoinf/theory/` to code implementations and development status

This document catalogs all theoretical papers used for pygeoinf development, their key contributions, and how they're realized in code. Use this as a roadmap to understand the theoretical foundations of the package.

---

## Table of Contents

1. [Core Methodological Papers](#core-methodological-papers)
2. [Foundational Mathematics](#foundational-mathematics)
3. [Specialized Methods](#specialized-methods)
4. [Recent Developments](#recent-developments)
5. [Paper-to-Code Quick Reference](#paper-to-code-quick-reference)

---

## Core Methodological Papers

### Backus & Gilbert (1967, 1968)

**Files:**
- `Backus and Gilbert - 1967 - Numerical applications of a formalism for geophysi.pdf`
- `Backus and Gilbert - 1968 - The resolving power of gross earth data.pdf`

**Key Concepts:**
1. **Averaging Kernels** (1967):
   - Linear combinations of data that approximate properties
   - Trade-off between resolution and amplification of errors
   - Spread function as measure of resolution

2. **Resolving Power** (1968):
   - Optimal approximation of delta function δ(r - r₀)
   - Resolving length as fundamental limit
   - Lagrange multiplier approach for constrained optimization

**Mathematical Framework:**
```
Property estimate: p̂ = Σᵢ aᵢ dᵢ
Averaging kernel: A(r; r₀) = Σᵢ aᵢ Gᵢ(r)
Goal: A(r; r₀) ≈ δ(r - r₀)
Spread: σ² = ∫ (r - r₀)² A(r; r₀)² dr
```

**Code Status:**
- ✅ Partial: `HyperEllipsoid` class in backus_gilbert.py (constraint sets)
- ✅ Partial: `LinearForm` represents delta functionals
- ⏸️ Incomplete: Full Backus-Gilbert solver with trade-off curves
- ⏸️ Future: Direct integration with DualMasterCostFunction

**Recommended Implementation Path:**
1. Define property operator T as weighted sum (averaging kernel)
2. Express spread constraint as quadratic form in λ
3. Use DualMasterCostFunction with trade-off parameter
4. Sweep trade-off to generate resolution-amplification curves

**Citation in Code:**
```python
# Backus & Gilbert (1967) averaging kernel approach
# See: theory/Backus and Gilbert - 1967 - Numerical applications...pdf
```

---

### Backus (1970) - Trilogy on Inadequate Data

**Files:**
- `Backus - 1970 - Inference from Inadequate and Inaccurate Data, I.pdf`
- `Backus - 1970 - Inference from Inadequate and Inaccurate Data, II.pdf`
- `Backus - 1970 - Inference from Inadequate and Inaccurate Data, III.pdf`

**Part I: Null Space vs Data Space**

**Key Result:** Any model m can be uniquely decomposed as:
```
m = m_N + m_D
where m_N ∈ N(G) (null space), m_D ∈ N(G)^⊥ (data space)
```

**Code Implementation:**
- ✅ Complete: `LinearOperator.kernel()` computes projector onto null space
- ✅ Complete: `LinearOperator.range()` computes projector onto data space
- ✅ Complete: SVD-based decomposition in matrix linear operators

**Usage Example:**
```python
# Decompose model into null + data components
P_null = forward_operator.kernel()      # Project to N(G)
P_data_space = forward_operator.range()  # Project to N(G)^⊥

m_null = P_null(m)
m_data_space = m - m_null

# Verify: G(m_null) = 0
```

**Part II: Confidence Regions**

**Key Concepts:**
- Convex geometry approach to uncertainty quantification
- Confidence set for model: {m : m ∈ B, Gm ∈ d̃ ± V}
- Support functions characterize boundaries

**Code Implementation:**
- ✅ Complete: `ConvexSubset` hierarchy (Ball, Ellipsoid, etc.)
- ✅ Complete: `GaussianMeasure.confidence_ellipsoid(alpha)`
- ✅ Active: `DualMasterCostFunction` implements confidence bounds
- ⏸️ Future: Visualization of confidence regions in property space

**Part III: Property Extraction**

**Key Idea:** Extract stable properties rather than full model recovery

**Code Implementation:**
- ✅ Conceptual: Property operator T in DualMasterCostFunction
- ⏸️ Future: Library of common property operators (point values, averages, integrals)

---

### Stuart (2010) - Bayesian Inverse Problems

**File:** `Stuart - 2010 - Inverse problems A Bayesian perspective.pdf`

**Key Contributions:**
1. **Well-Posedness on Function Spaces:**
   - Bayesian solution exists even for ill-posed deterministic problem
   - Posterior measure well-defined in infinite dimensions
   - Continuous dependence on data

2. **Measure-Theoretic Foundations:**
   - Gaussian measures on Hilbert/Banach spaces
   - Radon-Nikodym derivatives for likelihood
   - Cameron-Martin space for regularity

3. **Posterior Characterization:**
   - Prior: μ₀ (Gaussian with covariance C)
   - Likelihood: p(d|m) ∝ exp(-Φ(m; d))
   - Posterior: dμ^d/dμ₀ ∝ exp(-Φ(m; d))

**Code Implementation:**
- ✅ Complete: `LinearBayesianInversion` in linear_bayesian.py
- ✅ Complete: `GaussianMeasure` for prior/posterior
- ✅ Complete: Posterior mean/covariance computation
- ✅ Complete: Sampling from posterior via Cholesky
- ✅ Complete: Conditional Gaussian (affine constraints)

**Mathematical Formula (Linear Case):**
```python
# Posterior mean
m_post = m_prior + C_prior @ G.adjoint @ (
    G @ C_prior @ G.adjoint + Gamma
).inverse(d - G(m_prior))

# Posterior covariance
C_post = C_prior - C_prior @ G.adjoint @ (
    G @ C_prior @ G.adjoint + Gamma
).inverse @ G @ C_prior
```

**Code Example:**
```python
from pygeoinf import LinearBayesianInversion, GaussianMeasure

prior = GaussianMeasure(model_space, mean=m0, covariance=C)
problem = LinearForwardProblem(model_space, data_space, G, Gamma)

inversion = LinearBayesianInversion(problem, prior)
posterior = inversion.posterior_measure

# MAP estimate
m_map = posterior.mean

# Uncertainty (marginal standard deviations)
marginal_vars = posterior.covariance_operator.diagonal()
```

**Citation:**
```python
# Stuart (2010) Bayesian framework for inverse problems
# Well-posed posterior measure in infinite dimensions
```

---

### Al-Attar & Crawford (2021) - Deterministic Linear Inference

**File:** `Al-Attar - 2021 - Linear inference problems with deterministic const.pdf`

**Key Innovation:** Convex analysis approach without probability

**Master Dual Equation:**
```latex
h_U(q) = inf_{λ ∈ D} { ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ) }
```

**Geometric Interpretation:**
- U = admissible property set (what can be said with confidence)
- Characterized by support function h_U(q)
- Directional bounds: inf_U⟨q,·⟩, sup_U⟨q,·⟩
- Membership test via separating hyperplanes

**Code Implementation:**
- ✅ Phase 1 (Complete): Architecture analysis (see dual_master_implementation.md)
- ✅ Phase 2 (Complete): Support function refactor with `.support_function` property
- ✅ Phase 3 (Complete): DualMasterCostFunction class
- 🟨 Phase 4 (Partial): SubgradientDescent (4.1 done, 4.2-4.4 pending)
- ⏸️ Phase 5: Integration and testing
- ⏸️ Phase 6: Advanced features
- ✅ Phase 7 (Complete): Plane/half-space support functions (35 tests)
- 🟨 Phase 8 (Partial): Visualization (SubspaceSlicePlotter done)

**Current Status:** ~50% complete, actively developed

**File:** `pygeoinf/backus_gilbert.py` (DualMasterCostFunction)

**Usage Pattern:**
```python
# Define model prior set B (e.g., norm ball)
B = Ball(model_space, center=m0, radius=10.0)
sigma_B = B.support_function

# Define data error set V (e.g., confidence ellipsoid)
V = Ellipsoid(data_space, center=0, radius=chi2, operator=Gamma)
sigma_V = V.support_function

# Create dual master cost function
cost = DualMasterCostFunction(G, T, d_obs, sigma_B, sigma_V)

# Compute directional bound
cost.set_direction(q)  # Direction in property space
solver = SubgradientDescent(step_size=0.01, max_iter=1000)
result = solver.solve(cost, lambda_init)
h_U_q = result.best_value  # sup_{p ∈ U} ⟨q, p⟩
```

**Citation:**
```python
# Al-Attar & Crawford (2021) - Deterministic linear inference
# Master dual equation for confidence bounds
# See: theory.txt §2, theory/Al-Attar - 2021...pdf
```

---

### Bundle Methods for Non-Smooth Optimization (2020s)

**File:** `theory/bundle_methods.pdf` (21 pages)

**Key Contributions:**
1. **Level Bundle Algorithm:**
   - Cutting-plane model approximates non-smooth function
   - Quadratic master problem determines next iterate
   - Stability center provides robustness

2. **Asynchronous Extensions:**
   - Distributed computation with delayed oracle responses
   - Upper-bound estimation without full evaluations
   - Coordination strategies for convergence

3. **Inexact Oracles:**
   - Handle noisy function/subgradient evaluations
   - On-demand accuracy control
   - Convergence to ε-optimal solutions

**Mathematical Framework:**
```latex
Master QP: min  (1/2)||x - ˆx||²
           s.t. f(x_j) + ⟨g_j, x - x_j⟩ ≤ r_j  ∀j ∈ J_k
                sum_j r_j ≤ f_lev_k
                x ∈ X

Descent test: Δ_k = f_up_k - f_low_k ≤ tolerance
```

**Code Status:**
- 🔲 Future: `BundleMethod` class in convex_optimisation.py
- 🔲 Future: Quadratic master problem solver
- 🔲 Future: Bundle compression/aggregation
- 📋 Planned: See pygeoinf/plans/bundle-methods-optimizer-plan.md

**Comparison with Pygeoinf:**
- ✅ Compatible: Uses `NonLinearForm` oracle interface
- ✅ Compatible: Works with `SupportFunction` for dual problems
- ➕ New: Requires QP solver (scipy.optimize or cvxpy)
- 🆕 Advanced: Asynchronous/parallel computation (future extension)

**Advantages over Current SubgradientDescent:**
- Automatic step sizing (no manual tuning)
- Model-based search directions (better descent)
- Reliable gap-based termination (certificate of optimality)
- Stability center prevents oscillation
- Bundle accumulates information (not discarded)

**Application to Dual Master:**
Level bundle methods solve the dual master cost minimization:
```python
# Minimize: φ(λ) = ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)
cost = DualMasterCostFunction(G, T, d_obs, sigma_B, sigma_V)
cost.set_direction(q)

# Bundle method (automatic step sizing, gap-based stopping)
solver = BundleMethod(cost, alpha=0.1, tolerance=1e-6)
result = solver.solve(lambda_init)
h_U_q = result.best_value  # Certified: |h_U_q - h_U*(q)| ≤ 1e-6
```

**Citation in Code:**
```python
# Level bundle method (Lemaréchal et al., 1995; Kiwiel, 1995)
# Asynchronous extensions: theory/bundle_methods.pdf
```

**References:**
- Kiwiel (1995): Proximal level bundle methods
- Lemaréchal (1975): Extension of Davidon methods to non-differentiable problems
- van Ackooij & de Oliveira (2014): Level bundle methods with various oracles

---

### Bui-Thanh et al. (2013) - Infinite-Dimensional Bayesian Computation

**File:** `Bui-Thanh et al. - 2013 - A Computational Framework for Infinite-Dimensional.pdf`

**Key Contributions:**
1. **Hessian-Based Uncertainty Quantification:**
   - Prior-preconditioned Hessian H̃ = C^{1/2} H C^{1/2}
   - Spectrum of H̃ determines posterior uncertainty
   - Low-rank structure for efficient computation

2. **Randomized Methods:**
   - Randomized eigenvalue decomposition
   - Hutchinson trace estimator
   - Scalable to large-scale problems

3. **Geometric Insights:**
   - Data-informed subspace (eigenvectors with λ > 1)
   - Data-uninformed subspace (λ ≈ 0)
   - Posterior variance reduction only in informed subspace

**Code Implementation:**
- ✅ Partial: `random_matrix.py` (randomized SVD, eigenvalues)
- ✅ Partial: `GaussianMeasure.sample_from_posterior()` can use Hessian
- ✅ Complete: `SpectralPreconditioningMethod` in preconditioners.py
- ⏸️ Future: Full Hessian-based UQ workflow

**Recommended Usage:**
```python
# Randomized eigenvalue decomposition of prior-preconditioned Hessian
from pygeoinf import random_eig

# H̃ = C^{1/2} H C^{1/2}
H_tilde = (C_prior.sqrt @ hessian @ C_prior.sqrt)
eigenvalues, eigenvectors = random_eig(H_tilde, num_eigs=50)

# Posterior variance in data-informed directions
post_var_informed = 1 / (1 + eigenvalues)
```

**Citation:**
```python
# Bui-Thanh et al. (2013) - Hessian-based uncertainty quantification
# Low-rank approximation for infinite-dimensional posteriors
```

---

## Foundational Mathematics

### Bogachev (1996) - Gaussian Measures

**File:** `Bogachev - 1996 - Gaussian measures on linear spaces.pdf`

**Scope:** Comprehensive reference on Gaussian measures in infinite dimensions

**Key Topics:**
- Covariance operators (trace-class, Hilbert-Schmidt)
- Cameron-Martin space (RKHS of covariance)
- Integration and Fernique's theorem
- Measure equivalence and Radon-Nikodym derivatives

**Code Implementation:**
- ✅ Complete: `GaussianMeasure` class in gaussian_measure.py
- ✅ Complete: Covariance operator as positive-definite LinearOperator
- ✅ Complete: Sampling via Cholesky decomposition
- ✅ Complete: Affine transformations of Gaussian measures
- ⏸️ Not implemented: Cameron-Martin space as explicit object

**Theoretical Foundation For:**
- All Bayesian inversion methods
- Prior and noise modeling
- Posterior sampling

**Citation:**
```python
# Bogachev (1996) - Gaussian measures on linear spaces
# Foundation for infinite-dimensional probability
```

---

### Eldredge (2016) - Analysis on Infinite-Dimensional Spaces

**File:** `Eldredge - 2016 - Analysis and Probability on Infinite-Dimensional S.pdf`

**Scope:** Textbook on functional analysis and probability

**Relevance:**
- Conceptual foundation for `HilbertSpace` abstraction
- Sobolev spaces (related to `MassWeightedHilbertSpace`)
- Weak convergence and integration theory

**Code Implementation:**
- Conceptual: Not directly translated (textbook reference)
- Influences: Design of HilbertSpace API
- Related: `symmetric_space/` module for Sobolev spaces on manifolds

**Use Case:**
- Reference for developers extending to new spaces
- Background for understanding measure-theoretic aspects

---

## Specialized Methods

### SOLA Method for Helioseismic Inversion

**File:** `The SOLA method for helioseismic inversion,.pdf`

**Key Concept:** Subtractive Optimally Localized Averages

**Differences from Backus-Gilbert:**
- Different cost function (subtractive vs additive)
- Target function specification
- Used extensively in helioseismology

**Code Implementation:**
- ⏸️ Not implemented
- ⏸️ Future: Could express via DualMasterCostFunction with custom T

**Potential Integration:**
1. Define SOLA cost as NonLinearForm
2. Use SubgradientDescent or custom optimizer
3. Property operator T encodes target function
4. Compare with Backus-Gilbert on same problem

---

### Parker (1977) - Underparameterized Models

**File:** `Parker - 1977 - Linear inference and underparameterized models.pdf`

**Key Idea:** Focus on null space characterization

**Relation to Backus (1970):**
- Complements null-space decomposition
- Strategies for choosing regularization

**Code Implementation:**
- ✅ Complete: `LinearOperator.kernel()` for null space
- ✅ Complete: Regularization in LinearLeastSquaresInversion
- Used implicitly in all inversion methods

---

### Backus (1988) - Bayesian Geomagnetism

**File:** `Backus - 1988 - Bayesian inference in geomagnetism.pdf`

**Key Contribution:** Bridge between Bayesian and deterministic approaches

**Relates:**
- Gaussian prior → norm ball prior (limiting case)
- Probabilistic confidence → deterministic confidence sets

**Code Implementation:**
- Conceptual: Both `LinearBayesianInversion` and `DualMasterCostFunction` available
- Users can choose probabilistic or deterministic framework

---

### Backus (1988) - Hard vs Soft Prior Bounds

**File:** `Backus - 1988 - Comparing hard and soft prior bounds in geophysica.pdf`

**Key Question:** When to use hard constraints vs soft (probabilistic)?

**Trade-offs:**
- Hard (ConvexSubset): Sharp boundaries, computationally cheaper
- Soft (GaussianMeasure): Smooth, incorporates graded beliefs, easier derivatives

**Code Implementation:**
- ✅ Both frameworks available:
  - Hard: via ConvexSubset and support functions
  - Soft: via GaussianMeasure

**User Choice:**
```python
# Option 1: Hard constraint (support function)
prior = Ball(model_space, center=m0, radius=10.0)
sigma_B = prior.support_function

# Option 2: Soft constraint (Gaussian)
prior = GaussianMeasure(model_space, mean=m0, covariance=C)
```

---

### Backus (1989) - Quadratic Bound Confidence Sets

**File:** `Backus - 1989 - Confidence Set Inference with a Prior Quadratic Bo.pdf`

**Key Concept:** Ellipsoidal constraints on models

**Mathematical Form:**
```
B = {m : ⟨A(m - m₀), m - m₀⟩ ≤ M²}
```

**Code Implementation:**
- ✅ Complete: `Ellipsoid` class in subsets.py
- ✅ Complete: `EllipsoidSupportFunction` in convex_analysis.py

**Usage:**
```python
# Define ellipsoidal prior set
A = build_ellipse_operator()  # Positive-definite
B = Ellipsoid(model_space, center=m0, radius=M, operator=A)
```

---

### Uniqueness in Gross Earth Data Inversion

**File:** `Uniqueness in the inversion of inaccurate gross Ea.pdf`

**Key Result:** Conditions for unique solutions in geophysical inverse problems

**Relevance:**
- Injectivity of G (when ker G = {0})
- Impact of noise on uniqueness
- Stability estimates

**Code Implementation:**
- Diagnostic: Use `LinearOperator.kernel()` to check uniqueness
- Regularization automatically handles non-uniqueness

---

## Recent Developments

### Mag et al. (2025) - Bridging SOLA and DLI

**File:** `Mag et al. - 2025 - Bridging the gap between SOLA and deterministic li.pdf`

**Key Contribution:** Unified framework connecting SOLA and deterministic linear inference

**Status:** Recent paper, implementation roadmap TBD

**Future Work:**
- Implement unified SOLA-DLI cost function
- Compare both methods on benchmark problems
- Identify when each method is superior

---

### Stuart (2020c)

**File:** `stuart20c.pdf`

**Content:** (Supplement to Stuart 2010, specifics TBD based on file content)

**Status:** Available in theory/ directory

---

## Paper-to-Code Quick Reference

| Paper | Code Module | Status | Priority |
|-------|-------------|--------|----------|
| Backus & Gilbert 1967 | backus_gilbert.py (HyperEllipsoid) | Partial | High |
| Backus & Gilbert 1968 | (Future: full solver) | Not started | Medium |
| Backus 1970 I | linear_operators.py (kernel, range) | Complete | N/A |
| Backus 1970 II | subsets.py, convex_analysis.py | Complete | N/A |
| Backus 1970 III | backus_gilbert.py | Active dev | High |
| Stuart 2010 | linear_bayesian.py, gaussian_measure.py | Complete | N/A |
| Al-Attar 2021 | backus_gilbert.py (DualMasterCostFunction) | 50% | Critical |
| Bui-Thanh 2013 | random_matrix.py, preconditioners.py | Partial | Medium |
| Bogachev 1996 | gaussian_measure.py | Complete | N/A |
| Eldredge 2016 | (Conceptual foundation) | N/A | N/A |
| SOLA paper | (Not implemented) | Not started | Low |
| Parker 1977 | linear_operators.py | Complete | N/A |
| Backus 1988 (Bayes) | Both frameworks available | Complete | N/A |
| Backus 1988 (Hard/Soft) | Both approaches available | Complete | N/A |
| Backus 1989 (Quadratic) | subsets.py (Ellipsoid) | Complete | N/A |
| Uniqueness paper | Diagnostic tools available | Complete | N/A |
| Mag et al. 2025 | (Future work) | Not started | Low-Medium |

---

## Using This Index

### For Oracle-subagent

When researching a task:
1. Identify relevant mathematical concepts
2. Search this index for papers addressing those concepts
3. Check "Code Implementation" status
4. Read paper sections if implementing new features
5. Include paper citations in research summary

### For Sisyphus-subagent

When implementing:
1. Check index for relevant papers
2. Read "Mathematical Framework" sections
3. Add paper citations to docstrings:
   ```python
   """
   Implements Backus & Gilbert (1967) averaging kernel approach.

   See: theory/Backus and Gilbert - 1967 - Numerical applications...pdf
        Section 3.2, Equation (15)
   """
   ```
4. Follow notation from papers (LaTeX in docstrings)

### For Theory-Validator-subagent

Use index to:
1. Locate paper when implementation cites it
2. Verify mathematical formula matches paper
3. Check if assumptions from paper are satisfied in code
4. Cross-reference theorem numbers

### For Code-Review-subagent

Check that:
1. Paper citations in docstrings are accurate
2. Implementation status matches index claims
3. Code follows paper conventions
4. Related papers are cross-referenced

---

## Maintenance

Update this index when:
- New papers added to `theory/` directory
- Implementation status changes (Not started → Active → Complete)
- New connections discovered between papers and code
- Papers superseded by newer work

**Maintainer:** Check index against `ls pygeoinf/theory/*.pdf` quarterly

**Last Full Audit:** 2026-02-17
