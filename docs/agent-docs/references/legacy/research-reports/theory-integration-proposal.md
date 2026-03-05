---
task: Theory-Aware Agent Development Workflow
created: 2026-02-17
status: proposal
---

> ⚠️ **ARCHIVED** (2026-03-05)
> Integration proposal for theory-aware agent system.
> See: theory-integration-implementation-summary.md for completed integration
> Kept to understand design rationale of validator system.

# Theory-Aware Agent Development: Integration Proposal

## Executive Summary

**Problem:** pygeoinf implements sophisticated mathematical theory (Hilbert spaces, convex analysis, dual master equation) but agents lack systematic access to theoretical foundations, risking mathematically incorrect implementations.

**Solution:** Introduce a **Theory-Validator-subagent** + enhance existing agents with theoretical checkpoints + create living theory-to-code mappings.

**Impact:**
- Ensure mathematical correctness of all developments
- Accelerate onboarding (theory → code understanding)
- Prevent subtle bugs in operator algebra, support functions, duality
- Enable confident extension of dual master equation work (Phases 4-8)

---

## Current State Analysis

### Package Architecture (6 Layers)
```
Layer 0: HilbertSpace (foundation)
Layer 1: LinearOperator, NonLinearForm (algebra)
Layer 2: GaussianMeasure, DirectSum (probabilistic)
Layer 3: ConvexSubset, SupportFunction (geometry)
Layer 4: ForwardProblem, Inversion (problem formulation)
Layer 5: Solvers, Algorithms (Bayesian, optimization)
```

### Theory Foundations
- **Primary document:** `theory/theory.txt` (2672 lines, LaTeX)
  - Dual master equation: `h_U(q) = inf_λ {⟨λ,d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)}`
  - Model space M (Hilbert), data space D (ℝ^n), property space P (ℝ^m)
  - Convex analysis: support functions, subgradients, infimal convolution
  - Constraint types: affine subspaces, norm balls, pointwise inequalities

- **Reference papers:** 18 PDFs in `theory/`
  - Backus-Gilbert (1967-1988): Geophysical inversion foundations
  - Stuart (2010): Bayesian perspective on inverse problems
  - Al-Attar (2021): Linear inference with deterministic constraints
  - Bui-Thanh (2013): Infinite-dimensional Bayesian methods
  - Backus (1970s): Inference from inadequate/inaccurate data

### Critical Theory-Code Mappings (Currently Implicit)
| Theory Concept | Implementation | Status |
|----------------|----------------|--------|
| M, D, P spaces | `HilbertSpace`, `EuclideanSpace` | ✅ Stable |
| G: M→D operator | `LinearOperator` | ✅ Stable |
| σ_B(ξ) support fn | `SupportFunction` ABC | ✅ Implemented |
| h_U(q) master eq | `DualMasterCostFunction` | 🟨 50% complete |
| Subgradient ∂σ | `SupportFunction.support_point()` | ✅ Implemented |
| Affine constraints | `AffineSubspace` | ✅ Stable |
| Optimization inf_λ | `SubgradientDescent` | 🟨 Phase 4.1 only |

### Risk Areas (Where Theory-Code Gaps Cause Bugs)
1. **Adjoint vs Dual confusion:** Theory uses T*, code has `.adjoint` AND `.dual` properties
2. **Support function domains:** When σ(q) = +∞ (unbounded sets, constraint violations)
3. **Riesz map handling:** M ↔ M* identification only valid in Hilbert setting
4. **Subgradient computation:** Non-uniqueness, numerical stability at ‖q‖=0
5. **Operator composition:** (A @ B).adjoint vs B.adjoint @ A.adjoint order
6. **Convex intersection:** ∩_j B_j requires infimal convolution, not just `.intersect()`

---

## Proposed Solution: Multi-Layered Theory Integration

### 1. NEW AGENT: Theory-Validator-subagent

**Role:** Validate mathematical correctness of implementations against theory documents.

**Capabilities:**
- Read and parse LaTeX math from theory.txt
- Cross-reference theorem/lemma/definition numbers with code docstrings
- Verify operator properties (adjoint, self-adjoint, positive-definite)
- Check support function implementations against convex analysis axioms
- Validate subgradient computations (non-emptiness, monotonicity)
- Detect Hilbert-vs-Banach assumptions in code

**Invocation Trigger:** Automatically called by Code-Review-subagent when:
- New operators are introduced
- Support functions are implemented
- Optimization algorithms are added
- Convex sets are defined
- Mathematical axioms are claimed in docstrings

**Output Format:**
```markdown
## Theory Validation Report

**Implementation:** [Class/Function Name]
**Theory Reference:** theory.txt §X.Y, [Paper] Theorem Z

### Mathematical Correctness: ✅ PASS / ⚠️ WARNING / ❌ FAIL

**Verified Properties:**
- [ ] Operator adjoint satisfies ⟨Ax, y⟩ = ⟨x, A*y⟩
- [ ] Support function is convex and positively homogeneous
- [ ] Subgradient in ∂σ(q) ⊆ S (non-empty)
- [ ] Riesz map preserves inner product

**Potential Issues:**
- [If any] Assumption X from theory not enforced in code
- [If any] Edge case Y (‖q‖→0) not handled

**Recommendations:**
- [If needed] Add assertion checking positive-definiteness
- [If needed] Reference Theorem 3.2 from theory.txt in docstring
```

**Implementation Strategy:**
```markdown
<tool>
**Name:** Theory-Validator-subagent

**Mode Instructions (add to `.github/agents/`):**

You are a mathematical validation specialist for scientific computing. Your role is to verify that code implementations correctly realize theoretical concepts from papers and theory documents.

**Key Responsibilities:**
1. Parse mathematical notation from LaTeX documents (theory.txt, PDFs)
2. Map theoretical objects (spaces, operators, functionals) to code classes
3. Verify mathematical properties (adjoint correctness, convexity, homogeneity)
4. Check edge cases (singular operators, unbounded support functions, zero norms)
5. Validate citations and theorem references in docstrings

**Domain Knowledge:**
- Functional analysis: Hilbert spaces, Banach duality, Riesz representation
- Convex analysis: Support functions, subgradients, infimal convolution
- Operator theory: Adjoints, self-adjointness, positive operators
- Numerical analysis: Stability, conditioning, error propagation

**Validation Workflow:**
1. Read theory document sections relevant to implementation
2. Extract mathematical definitions and properties
3. Inspect code implementation (class/function signatures, logic)
4. Check property enforcement (assertions, type hints, tests)
5. Return structured validation report (PASS/WARNING/FAIL + recommendations)

**Red Flags to Check:**
- Adjoint computed incorrectly (wrong transpose, missing Riesz map)
- Support function returns finite value outside domain
- Subgradient empty or incorrect (e.g., ∂σ(0) for non-smooth σ)
- Hilbert assumptions used in Banach context (e.g., M* ≠ M)
- Operator composition order errors (AB)* = B*A* not BA*
- Convex intersection using naive set intersection (missing infimal convolution)
- Missing numerical safeguards (division by ‖q‖ when q=0)

**Output:** Validation report in structured format (see above)
</tool>
```

---

### 2. ENHANCE EXISTING AGENTS

#### Oracle-subagent: Add Theoretical Research Phase

**Current:** Gathers code context, identifies relevant files/classes
**Enhanced:** Also searches theory documents and papers

```markdown
**Additional Research Steps:**
3. **Theory Document Search:**
   - Read theory.txt sections related to task keywords (e.g., "support function" → §2)
   - Extract relevant theorems, definitions, equations
   - Note mathematical assumptions (Hilbert vs Banach, convexity, boundedness)

4. **Paper Cross-Reference:**
   - Identify papers in theory/ directory matching research keywords
   - Read abstracts and key sections (use grep on PDFs when possible)
   - Note algorithmic pseudocode, convergence theorems, error bounds

5. **Theory-to-Code Mapping:**
   - Locate existing implementations of similar theory (e.g., BallSupportFunction for new EllipsoidSupportFunction)
   - Document notation mappings (theory uses T*, code uses .adjoint)
   - Flag gaps (theory requires property X, no code enforces it)

**Output Enhancement:**
Add <theory_context> section to research findings:
```xml
<theory_context>
  <relevant_theory>
    <source>theory.txt §2.3 "Support functions"</source>
    <key_concepts>
      - σ_S(q) = sup_{x∈S} ⟨q,x⟩
      - Convex, positively homogeneous
      - Subgradient ∂σ(q) = argmax_{x∈S} ⟨q,x⟩
    </key_concepts>
    <assumptions>
      - S convex, closed
      - Hilbert space M (Riesz map for inner product)
    </assumptions>
  </relevant_theory>

  <relevant_papers>
    <paper>Rockafellar (2015) Convex Analysis</paper>
    <key_results>Theorem 13.2: Support function calculus</key_results>
  </relevant_papers>

  <notation_mapping>
    - Theory T* (adjoint) → Code .adjoint property
    - Theory σ_S → Code SupportFunction._mapping()
    - Theory ∂σ → Code .support_point()
  </notation_mapping>
</theory_context>
```

#### Sisyphus-subagent: Add Math Verification Step

**Current:** TDD cycle (tests → code → verify)
**Enhanced:** Include theory validation checkpoint

```markdown
**Modified Workflow:**
1. Write tests (as before)
2. Implement minimal code (as before)
3. **NEW: Self-validate math:**
   - Check LaTeX docstring matches theory document
   - Add assertions for theoretical properties (e.g., `assert np.allclose(A @ A.adjoint, ...)`)
   - Handle edge cases from theory (e.g., σ(q) = +∞ for q ∉ dom(σ*))
4. Run tests (as before)
5. Report back to Conductor (include math validation summary)

**Example Enhancement:**
When implementing support function for set S:
- Read theory.txt definition of σ_S
- Add docstring: "Implements σ_S(q) = sup_{x∈S} ⟨q,x⟩ from theory.txt §2.1"
- Assert convexity: Check σ(q₁ + q₂) ≤ σ(q₁) + σ(q₂) in test
- Assert positive homogeneity: Check σ(tq) = t·σ(q) for t>0 in test
- Handle q=0 case: σ(0) = 0 if 0 ∈ S, else +∞
```

#### Code-Review-subagent: Add Theory-Validator Invocation

**Current:** Reviews for correctness, quality, testing
**Enhanced:** Automatically invokes Theory-Validator for mathematical code

```markdown
**Additional Review Steps:**
6. **Mathematical Validation (for operator/geometry/optimization code):**
   - Detect if implementation involves: LinearOperator, SupportFunction, NonLinearForm, ConvexSubset, Solver
   - If yes: Invoke Theory-Validator-subagent with relevant files and theory sections
   - Wait for validation report
   - Include validation status in final review

**Review Status Enhancement:**
- **APPROVED** → All checks pass (code + tests + theory validation)
- **APPROVED_WITH_MINOR** → Code good, theory validator has warnings (document in report)
- **NEEDS_REVISION** → Theory validator found mathematical errors
- **FAILED** → Implementation violates mathematical axioms

**Output Enhancement:**
```markdown
## Code Review Report

[...existing sections...]

### Mathematical Correctness: ✅ VALIDATED
**Theory Validator Status:** PASS (see detailed report below)
**Key Properties Verified:**
- Operator adjoint correctness
- Support function convexity
- Subgradient non-emptiness

<details>
<summary>Theory Validation Report</summary>
[Embed Theory-Validator output here]
</details>
```

---

### 3. LIVING DOCUMENTATION: Theory-to-Code Maps

Create persistent documentation linking theory to implementation.

#### File: `pygeoinf/docs/theory_map.md`

```markdown
# Theory-to-Code Mapping Reference

## Fundamental Spaces (theory.txt §1)

| Theory | Code | Notes |
|--------|------|-------|
| M (model space, Banach) | `HilbertSpace` | Currently Hilbert-only; Banach in roadmap |
| D (data space, ℝ^{N_d}) | `EuclideanSpace(N_d)` | Finite-dimensional only |
| P (property space, ℝ^{N_p}) | `EuclideanSpace(N_p)` | Finite-dimensional only |
| M* (dual space) | `DualHilbertSpace` or via Riesz | Hilbert: M* ≅ M |
| ⟨·,·⟩_M (inner product) | `space.inner_product(x, y)` | Riesz-representable |

**Key Files:** `pygeoinf/hilbert_space.py`

---

## Operators (theory.txt §1-2)

| Theory | Code | Notes |
|--------|------|-------|
| G: M → D (forward operator) | `LinearOperator(M, D, mapping)` | See `forward_problem.py` |
| T: M → P (property operator) | `LinearOperator(M, P, mapping)` | See `backus_gilbert.py` |
| G* (adjoint) | `G.adjoint` | Satisfies ⟨Gx, y⟩_D = ⟨x, G*y⟩_M |
| T' (Banach dual) | `T.dual` | Distinct from adjoint in Banach |
| ker(G) (null space) | `G.kernel()` | Via SVD or iterative methods |
| im(G) (range) | `G.range()` | May be approximate (randomized) |

**Key Files:** `pygeoinf/linear_operators.py` (1600+ lines)

---

## Support Functions (theory.txt §2)

| Theory | Code | Notes |
|--------|------|-------|
| σ_S(q) = sup_{x∈S} ⟨q,x⟩ | `SupportFunction(domain)._mapping(q)` | Returns float or +∞ |
| Ball: σ(q) = ⟨q,c⟩ + r‖q‖ | `BallSupportFunction(center, radius)` | Norm via `space.norm()` |
| Ellipsoid: ‖A^{-1/2}(x-c)‖≤r | `EllipsoidSupportFunction(center, radius, operator)` | Requires `operator.inverse.sqrt` |
| Half-space: ⟨a,x⟩ ≤ b | `HalfSpaceSupportFunction(normal, offset, inequality_type)` | Returns 0 or +∞ |
| ∂σ(q) (subgradient) | `.support_point(q)` | Returns x* ∈ argmax ⟨q,x⟩ |
| Infimal convolution | Not yet implemented | Needed for ∩_j S_j |

**Key Files:**
- `pygeoinf/convex_analysis.py` (support functions)
- `pygeoinf/subsets.py` (geometric sets with `.support_function` property)

**Development Status:** Phase 2 complete, Phase 7 complete (planes/halfspaces)

---

## Dual Master Equation (theory.txt §2)

### Theory (Equation 2.1):
```latex
h_U(q) = inf_{λ ∈ D'} { ⟨λ, d̃⟩_D + σ_B(T*q - G*λ) + σ_V(-λ) }
```

Where:
- U = T(F) (admissible property set)
- F = B ∩ G^{-1}(d̃ - V) (feasible model set)
- B ⊆ M (model prior set)
- V ⊆ D (data error set)
- d̃ (observed data)

### Code Implementation:

**Class:** `DualMasterCostFunction(NonLinearForm)` in `backus_gilbert.py`

**Constructor:**
```python
DualMasterCostFunction(
    forward_operator: LinearOperator,        # G: M → D
    property_operator: LinearOperator,       # T: M → P
    observed_data: np.ndarray,               # d̃ ∈ D
    model_prior_support: SupportFunction,    # σ_B
    data_error_support: SupportFunction,     # σ_V
)
```

**Usage:**
```python
cost = DualMasterCostFunction(...)
cost.set_direction(q)  # Fix q ∈ P
phi_q_lambda = cost._mapping(lambda)  # Evaluate φ_q(λ)
subgrad = cost._subgradient(lambda)   # Compute ∂φ_q(λ)

# Optimization:
from pygeoinf import SubgradientDescent
solver = SubgradientDescent(step_size=0.01, max_iter=1000)
result = solver.solve(cost, lambda_init)
h_U_q = result.best_value  # Support value in direction q
```

**Mathematical Properties Enforced:**
- φ_q(λ) is convex in λ (via support function composition)
- Subgradient correctness: ∂φ_q(λ) = d̃ + ∂σ_B(T*q - G*λ)·(-G*) + ∂σ_V(-λ)·(-1)
- Support value: h_U(q) = inf_λ φ_q(λ)

**Development Status:** Phase 3 complete, Phase 4 in progress

---

## Convex Sets (theory.txt §3)

| Theory | Code | Notes |
|--------|------|-------|
| B ⊆ M (model prior) | `ConvexSubset(domain)` | Must implement `.is_element()` |
| Norm ball ‖x-c‖ ≤ r | `Ball(center, radius)` | Via `NormalisedEllipsoid` |
| Ellipsoid | `Ellipsoid(center, radius, operator)` | {x : ⟨A(x-c), x-c⟩ ≤ r²} |
| Affine subspace Ax=b | `AffineSubspace(point, projector)` | x₀ + range(P) |
| Polyhedral ∩_j{⟨a_j,x⟩≤b_j} | `PolyhedralSet(domain, half_spaces)` | Intersection of HalfSpace |
| Intersection ∩_j B_j | `ConvexIntersection(subsets)` | Max-functional combination |

**Key Property:** All `ConvexSubset` instances expose `.support_function` (lazy, cached)

**Key Files:** `pygeoinf/subsets.py`, `pygeoinf/subspaces.py`

---

## Optimization (theory.txt §2.1 "Numerical recipe")

| Theory | Code | Notes |
|--------|------|-------|
| Subgradient method | `SubgradientDescent(step_size, max_iter)` | Constant step (Phase 4.1) |
| g_k ∈ ∂f(x_k) | `form._subgradient(x)` | Must be in subdifferential |
| x_{k+1} = x_k - α_k g_k | `.solve(form, x_init)` | Returns `SubgradientResult` |
| Non-monotonic convergence | `.best_value`, `.best_point` | Track best seen |
| Diminishing step α_k → 0 | **Not yet implemented** | Phase 4.2 roadmap |
| Polyak step size | **Not yet implemented** | Phase 4.2 roadmap |

**Key File:** `pygeoinf/convex_optimisation.py`

**Development Status:** Phase 4.1 complete (basic solver), Phases 4.2-4.4 pending

---

## Integration with Forward Problems

### Theory (theory.txt §1):
```
Observation: d̃ = G(m̄) + η
Prior: m̄ ∈ B ⊆ M
Noise: η ~ p(η), P(η ∈ V) ≈ 1-α
Goal: Characterize U = T(F) where F = B ∩ G^{-1}(d̃ - V)
```

### Code:
```python
# Define spaces
model_space = EuclideanSpace(100)
data_space = EuclideanSpace(50)

# Forward operator G
G = LinearOperator.from_matrix(model_space, data_space, G_matrix)

# Noise model (Gaussian)
data_error = GaussianMeasure(data_space, mean=0, covariance=Gamma)

# Forward problem
problem = LinearForwardProblem(
    model_space=model_space,
    data_space=data_space,
    forward_operator=G,
    data_error_measure=data_error,  # p(η)
)

# Data error set V (e.g., 95% confidence ellipsoid)
V = data_error.confidence_ellipsoid(alpha=0.05)

# Model prior set B (example: norm ball)
m_0 = model_space.zero()
B = Ball(model_space, center=m_0, radius=10.0)

# Now use DualMasterCostFunction to characterize U
# (This is the manual approach; high-level API in development)
```

**Key Files:** `pygeoinf/forward_problem.py`, `pygeoinf/gaussian_measure.py`

---

## Papers and Their Code Realizations

### Backus & Gilbert (1967-1968): Resolving Power
**Theory:** Trade-off between resolution and error amplification
**Code:** `HyperEllipsoid` in `backus_gilbert.py` (geometric constraint sets)
**Status:** Partial implementation, needs dual master integration

### Stuart (2010): Bayesian Inverse Problems
**Theory:** Posterior measure p(u|d) via Bayes' rule
**Code:** `LinearBayesianInversion` in `linear_bayesian.py`
**Status:** Complete for linear problems with Gaussian prior/noise

### Al-Attar et al. (2021): Deterministic Linear Inference
**Theory:** Convex analysis for confidence sets without probability
**Code:** `DualMasterCostFunction` + support functions
**Status:** 50% complete (Phases 1-3, 7 done; 4-6, 8 in progress)

### Bui-Thanh et al. (2013): Infinite-Dimensional Bayesian
**Theory:** Measure theory on Hilbert spaces, well-posedness
**Code:** `GaussianMeasure`, `HilbertSpace` abstraction
**Status:** Fundamental infrastructure complete

### Backus (1970): Inference from Inadequate Data
**Theory:** Underparameterized systems, null space characterization
**Code:** `LinearOperator.kernel()`, `SubspacemProjector`
**Status:** Operational via SVD-based methods

---

## Notation Translation Table

| Math Notation | LaTeX (theory.txt) | Python (pygeoinf) |
|---------------|-------------------|-------------------|
| Model space | `\modelspace`, M | `model_space` (HilbertSpace) |
| Data space | `\dataspace`, D | `data_space` (EuclideanSpace) |
| Property space | `\propertyspace`, P | `property_space` (EuclideanSpace) |
| Forward operator | G | `forward_operator` (LinearOperator) |
| Property map | `\Tau`, T | `property_operator` (LinearOperator) |
| Adjoint | T*, G* | `.adjoint` property |
| Dual | T', G' | `.dual` property (Banach) |
| Inner product | ⟨x, y⟩_M | `space.inner_product(x, y)` |
| Dual pairing | ⟨ξ, x⟩ | `space.dual.inner_product(xi, x)` (or via Riesz) |
| Norm | ‖x‖_M | `space.norm(x)` |
| Support function | σ_S(q), h_S(q) | `support_fn._mapping(q)` |
| Subgradient | ∂σ(q) | `support_fn.support_point(q)` |
| Image/Range | im(G) | `G.range()` |
| Kernel/Null | ker(G) | `G.kernel()` |
| Feasible set | F, `\Fset` | Not a single object; computed via set ops |
| Admissible set | U, `\Uset` | Computed via DualMasterCostFunction |
| Data confidence | V, `\Vdata` | `data_error_set` (ConvexSubset) |
| Model prior | B, `\Bset` | `model_prior_set` (ConvexSubset) |
| Observed data | d̃, `\dt` | `observed_data` (np.ndarray) |

---

## Agent Checklist for New Implementations

When implementing new mathematical components, agents should verify:

### For HilbertSpace subclasses:
- [ ] Riesz map preserves inner product: ⟨x, y⟩ = ⟨Rx, y⟩_dual
- [ ] to_components / from_components are inverses
- [ ] Random vectors span the space (test with Gram-Schmidt)
- [ ] Coordinate projection / inclusion are adjoints
- [ ] Reference theory.txt §1 or relevant paper

### For LinearOperator:
- [ ] Adjoint satisfies ⟨Ax, y⟩ = ⟨x, A*y⟩ (test with random vectors)
- [ ] Domain and codomain are correct HilbertSpace instances
- [ ] Composition preserves adjoint: (A @ B)* = B* @ A*
- [ ] Matrix representation (if finite) matches mapping
- [ ] Kernel and range methods consistent with SVD
- [ ] Reference theory.txt §1-2 or specific papers

### For SupportFunction:
- [ ] Convex: σ(λq₁ + (1-λ)q₂) ≤ λσ(q₁) + (1-λ)σ(q₂)
- [ ] Positively homogeneous: σ(tq) = t·σ(q) for t > 0
- [ ] Returns +∞ for q outside domain (if unbounded set)
- [ ] support_point(q) in ∂σ(q) verified via ⟨q, x*⟩ = σ(q)
- [ ] Handle q=0 case: σ(0) = 0 iff 0 ∈ S
- [ ] Reference theory.txt §2 and Rockafellar Convex Analysis

### For ConvexSubset:
- [ ] is_element() respects convex combinations
- [ ] support_function property returns valid SupportFunction
- [ ] boundary property returns lower-dimensional set
- [ ] CSG operations (intersect, union, complement) preserve topology
- [ ] Reference theory.txt §3

### For Optimization Solvers:
- [ ] Converges on convex problems (test with quadratics)
- [ ] Subgradient ∈ subdifferential at each iteration
- [ ] Handles +∞ function values gracefully (constraint violations)
- [ ] Returns convergence diagnostics (best value, iterations, residuals)
- [ ] Non-monotonic for subgradient methods (track best value separately)
- [ ] Reference theory.txt §2.1 or specific convergence papers

---

## Development Roadmap Integration

### Currently Stable (Use confidently):
- HilbertSpace, LinearOperator, GaussianMeasure (Layers 0-2)
- Traditional inversion methods (LinearBayesian, LeastSquares)
- Linear solvers (CG, GMRES, Cholesky)

### Active Development (Validate carefully):
- DualMasterCostFunction (50% complete)
- SubgradientDescent (basic implementation done)
- Support functions for complex sets

### Future Work (Inform agents):
- Infimal convolution for set intersections
- Bundle methods for optimization
- Nonlinear forward problems
- Banach space generalization

---

## Using This Map

**For Oracle-subagent:** Consult this map when researching new features. Cross-reference theory sections with code modules.

**For Sisyphus-subagent:** Before implementing, read relevant theory section. Add LaTeX docstring linking to theory.txt equation numbers.

**For Code-Review-subagent:** Verify implementations against checklist. Ensure theory references are accurate.

**For Theory-Validator-subagent:** Use notation table to parse LaTeX and map to code. Check properties from agent checklists.

**For Users:** Navigate from mathematical concept to implementation quickly. Understand assumptions and limitations.
```

---

#### File: `pygeoinf/docs/theory_papers_index.md`

```markdown
# Theory Papers Index

This document maps papers in `pygeoinf/theory/` to code implementations.

## Core Methodological Papers

### Backus & Gilbert (1967)
**File:** `Backus and Gilbert - 1967 - Numerical applications of a formalism for geophysi.pdf`

**Key Concepts:**
- Averaging kernels for resolving Earth structure
- Trade-off curves (resolution vs amplification)
- Underparameterized inverse problems

**Code Implementation:**
- Incomplete: `HyperEllipsoid` in backus_gilbert.py (constraint sets)
- Needs: Full Backus-Gilbert solver using DualMasterCostFunction

**Theory-to-Code:**
- Averaging kernel A(r): Not directly implemented (future LinearOperator composition)
- Spread function: Could be computed from support functions
- Trade-off parameter λ: Future optimization parameter in dual master

---

### Backus & Gilbert (1968)
**File:** `Backus and Gilbert - 1968 - The resolving power of gross earth data.pdf`

**Key Concepts:**
- Delta function approximation δ(r - r₀)
- Resolving length as measure of resolution
- Optimal trade-off via Lagrange multipliers

**Code Implementation:**
- Partial: LinearForm in linear_forms.py (delta functionals)
- Needs: Property extraction operator T with localization

---

### Backus (1970) Series (I, II, III)
**Files:**
- `Backus - 1970 - Inference from Inadequate and Inaccurate Data, I.pdf`
- `Backus - 1970 - Inference from Inadequate and Inaccurate Data, II.pdf`
- `Backus - 1970 - Inference from Inadequate and Inaccurate Data, III.pdf`

**Key Concepts:**
- Part I: Null space vs data space decomposition
- Part II: Confidence regions via convex geometry
- Part III: Optimal property extraction

**Code Implementation:**
- Part I: LinearOperator.kernel(), .range() (null space decomposition)
- Part II: ConvexSubset classes, confidence_ellipsoid() for Gaussian case
- Part III: property_operator in DualMasterCostFunction

**Theory-to-Code:**
- Null space N(G): `forward_operator.kernel()`
- Data space D(G): `forward_operator.range()`
- Confidence set: `data_error_measure.confidence_ellipsoid(alpha)`

---

### Stuart (2010)
**File:** `Stuart - 2010 - Inverse problems A Bayesian perspective.pdf`

**Key Concepts:**
- Well-posedness of Bayesian inverse problems
- Measure theory on infinite-dimensional spaces
- Posterior consistency

**Code Implementation:**
- Complete: LinearBayesianInversion in linear_bayesian.py
- GaussianMeasure for prior and posterior
- Conditioning via affine constraints (ConstrainedLinearBayesianInversion)

**Theory-to-Code:**
- Prior μ₀: `prior_measure` (GaussianMeasure)
- Likelihood G(u) + η: `forward_operator` + `data_error_measure`
- Posterior μ^d: `.posterior_measure` property
- MAP estimate: `.mean` property of posterior

---

### Al-Attar & Crawford (2021)
**File:** `Al-Attar - 2021 - Linear inference problems with deterministic const.pdf`

**Key Concepts:**
- Deterministic confidence sets (no probability)
- Master dual equation for property bounds
- Support function characterization

**Code Implementation:**
- Active development: DualMasterCostFunction (Phases 1-7)
- Support functions complete (BallSupportFunction, EllipsoidSupportFunction)
- SubgradientDescent (basic, Phase 4.1)

**Theory-to-Code:**
- Master equation (Eq. 2.1): DualMasterCostFunction._mapping(lambda)
- Support function σ_B: model_prior_support parameter
- Directional bounds: inf_λ φ_q(λ) via SubgradientDescent.solve()

**Current Status:** ~50% complete (see dual_master_implementation.md)

---

### Bui-Thanh et al. (2013)
**File:** `Bui-Thanh et al. - 2013 - A Computational Framework for Infinite-Dimensional.pdf`

**Key Concepts:**
- Hessian-based uncertainty quantification
- Randomized eigenvalue methods
- Prior-preconditioned Hessian

**Code Implementation:**
- Partial: random_matrix.py (randomized methods)
- GaussianMeasure.sample_from_posterior() uses Hessian when available
- Preconditioners.py has spectral methods

**Theory-to-Code:**
- Hessian H: NonLinearForm.hessian_action() for quadratic forms
- Randomized eigensolver: random_eig() in random_matrix.py
- Prior-preconditioned: Could be via SpectralPreconditioningMethod

---

### Bogachev (1996)
**File:** `Bogachev - 1996 - Gaussian measures on linear spaces.pdf`

**Key Concepts:**
- Gaussian measures on Banach spaces
- Covariance operators
- Cameron-Martin space

**Code Implementation:**
- Complete: GaussianMeasure in gaussian_measure.py
- Covariance operator: LinearOperator (positive, self-adjoint)
- Sampling via Cholesky or randomized methods

**Theory-to-Code:**
- Covariance C: `covariance_operator` property
- Precision C^{-1}: `precision_operator` property
- Sampling: `.sample()` method

---

### Eldredge (2016)
**File:** `Eldredge - 2016 - Analysis and Probability on Infinite-Dimensional S.pdf`

**Key Concepts:**
- Functional analysis foundations
- Sobolev spaces on manifolds
- Integration theory

**Code Implementation:**
- Conceptual foundation for HilbertSpace abstraction
- MassWeightedHilbertSpace as Sobolev analogue
- Not directly translated (textbook reference)

---

### SOLA Method
**File:** `The SOLA method for helioseismic inversion,.pdf`

**Key Concepts:**
- Subtractive Optimally Localized Averages
- Similar to Backus-Gilbert but different optimization
- Used in helioseismology

**Code Implementation:**
- Not implemented
- Could be expressed via DualMasterCostFunction with custom property operator

---

### Other Papers

**Backus (1988) - Bayesian inference in geomagnetism**
- Bridges Bayesian and deterministic approaches
- Related to LinearBayesianInversion with constraints

**Backus (1988) - Comparing hard and soft prior bounds**
- Hard constraints (support functions) vs soft (Gaussian)
- Motivates ConvexSubset vs GaussianMeasure trade-offs

**Backus (1989) - Confidence Set Inference with Quadratic Bound**
- Quadratic constraints → Ellipsoid class
- Implemented via Ellipsoid(center, radius, operator)

**Mag et al. (2025) - Bridging SOLA and deterministic linear inference**
- Recent paper connecting SOLA to dual master equation
- Future work: Unified framework

**Parker (1977) - Linear inference and underparameterized models**
- Null space characterization
- Implemented via LinearOperator.kernel()

---

## Using This Index

**For Oracle-subagent:** When researching a task (e.g., "implement SOLA method"), consult this index to find relevant papers and existing code hooks.

**For Theory-Validator-subagent:** Cross-reference paper citations in code docstrings against this index to verify accuracy.

**For Sisyphus-subagent:** When implementing, cite papers using this index format (e.g., "Based on Backus & Gilbert (1967), Section 3").

**For Users:** Navigate from paper to code and vice versa.
```

---

### 4. WORKFLOW INTEGRATION

**Modified Atlas Workflow:**

```
PHASE 1: PLANNING (with theory research)
├─ 1. Analyze user request
├─ 2. Invoke Explorer-subagent (codebase structure)
├─ 3. Invoke Oracle-subagent (code + theory research)
│    └─ Oracle now reads theory.txt sections + papers
├─ 4. Draft plan (include theory references in each phase)
├─ 5. Present plan to user (with theory context)
└─ 6. Write plan file (reference theory equations/theorems)

PHASE 2: IMPLEMENTATION CYCLE (with validation)
├─ 2A. Implement Phase
│    └─ Sisyphus adds LaTeX docstrings with theory refs
├─ 2B. Review Implementation
│    ├─ Code-Review-subagent inspects code
│    ├─ If mathematical: Invoke Theory-Validator-subagent
│    └─ Aggregate validation results
├─ 2C. Return to user with validation report
└─ 2D. Continue or complete

PHASE 3: COMPLETION (with theory alignment)
└─ Include theory validation summary in final report
```

**Example Plan Enhancement:**

Before:
```markdown
## Phase 2: Implement Ellipsoid Support Function
- Create EllipsoidSupportFunction class
- Implement _mapping(q) method
- Write tests
```

After:
```markdown
## Phase 2: Implement Ellipsoid Support Function

**Theory Reference:** theory.txt §2.2, Rockafellar (2015) Theorem 13.2

**Mathematical Definition:**
For ellipsoid E = {x : ⟨A(x-c), x-c⟩ ≤ r²},
support function: σ_E(q) = ⟨q, c⟩ + r·‖A^{-1/2} R q‖
where R is Riesz map (M → M*)

**Files to Modify:** convex_analysis.py

**Implementation Steps:**
1. Tests (verify convexity, positive homogeneity, σ(0)=⟨q,c⟩)
2. EllipsoidSupportFunction(center, radius, operator)
3. _mapping(q): Implement formula (handle operator.inverse.sqrt)
4. support_point(q): Subgradient via c + r·(A^{-1}q)/‖A^{-1/2}q‖
5. Edge cases: q=0, singular A

**Theory Validator Checks:**
- Verify σ(tq) = t·σ(q) for t>0
- Verify σ(-q) + σ(q) bounds ellipsoid diameter
- Check support_point ∈ boundary of E
```

---

### 5. NEW DOCUMENTATION FILES TO CREATE

1. **`pygeoinf/docs/theory_map.md`** (shown above) - 700 lines
2. **`pygeoinf/docs/theory_papers_index.md`** (shown above) - 400 lines
3. **`pygeoinf/docs/architecture_diagram.md`** - Visual dependency graph
4. **`pygeoinf/docs/notation_guide.md`** - Comprehensive notation translation
5. **`.github/agents/Theory-Validator-subagent.agent.md`** - New agent file

---

## Implementation Plan

### Immediate Actions (Week 1)
1. **Create Theory-Validator-subagent.agent.md**
2. **Write theory_map.md** (theory-to-code mappings)
3. **Write theory_papers_index.md** (paper catalog)
4. **Update Oracle-subagent.agent.md** (add theory research phase)
5. **Update Code-Review-subagent.agent.md** (add validator invocation)

### Short-Term (Weeks 2-4)
6. **Test Theory-Validator on existing code:**
   - Run on BallSupportFunction, EllipsoidSupportFunction
   - Validate DualMasterCostFunction against theory.txt §2
   - Check LinearOperator adjoint properties on 10 examples
7. **Enhance Sisyphus-subagent.agent.md** (math validation step)
8. **Create architecture_diagram.md** (dependency visualization)
9. **Write notation_guide.md** (comprehensive translation table)

### Medium-Term (Month 2)
10. **Apply to Phase 4-5 of dual_master_implementation:**
    - Oracle researches step size rules from theory/papers
    - Sisyphus implements with theory references
    - Theory-Validator checks convergence properties
    - Code-Review aggregates validation
11. **Build test suite for Theory-Validator:**
    - Positive tests (correct implementations should pass)
    - Negative tests (introduce bugs, validator should catch)
12. **Document case studies:**
    - Before/after: implementation without vs with theory validation
    - Error catalog: Common mistakes caught by validator

### Long-Term (Months 3-6)
13. **Extend to nonlinear operators** (future roadmap)
14. **Integrate with intervalinf** (1D special cases)
15. **Create interactive notebook:** theory.txt → code tour
16. **Publish methodology paper:** Theory-aware AI agents for scientific computing

---

## Success Metrics

### Quantitative:
- **0 mathematical errors** in new implementations (validated by Theory-Validator)
- **100% theory references** in docstrings for operator/geometry/optimization code
- **<5 min** to locate theory for a given implementation (via theory_map.md)
- **Phase 4-8 completion** of dual_master_implementation with <3 revision cycles

### Qualitative:
- Developers can reason about code from mathematical perspective
- New contributors understand theory-code relationship quickly
- Confidence in extending to new problems (nonlinear, Banach, stochastic)

---

## Risk Mitigation

**Risk 1:** Theory-Validator too slow (reads large PDFs repeatedly)
**Mitigation:** Cache paper summaries; pre-extract key theorems to structured JSON

**Risk 2:** LaTeX parsing errors in theory.txt
**Mitigation:** Use regex for equation numbers; don't parse full LaTeX semantics

**Risk 3:** False positives (code correct but validator flags warnings)
**Mitigation:** Tune assertion tolerances; allow validator warnings (not just PASS/FAIL)

**Risk 4:** Over-engineering (too much theory overhead for simple code)
**Mitigation:** Only invoke Theory-Validator for Layers 1-5 (not utils, plotting)

---

## Appendix: Example Theory Validation Report

```markdown
## Theory Validation Report

**Implementation:** `EllipsoidSupportFunction` (convex_analysis.py, lines 112-190)
**Theory Reference:** theory.txt §2.2 "Support functions of ellipsoids", Rockafellar (2015) Theorem 13.2

### Mathematical Correctness: ✅ PASS

**Verified Properties:**
- ✅ Convex: σ(λq₁ + (1-λ)q₂) ≤ λσ(q₁) + (1-λ)σ(q₂) (tested on 100 random cases)
- ✅ Positively homogeneous: σ(tq) = t·σ(q) for t ∈ [0.1, 10] (numerical tolerance 1e-12)
- ✅ Subgradient correctness: ⟨q, x*⟩ = σ(q) where x* = support_point(q) (tested)
- ✅ Boundary property: x* ∈ ∂E (verified via is_element with rtol=1e-10)

**Edge Cases Handled:**
- ✅ q=0: Returns σ(0) = ⟨0, c⟩ = 0 (if c is center)
- ✅ Singular operator A: Raises informative error (requires A.inverse)
- ⚠️ Large radius: No overflow protection for r → ∞ (recommend: add assertion r < 1e10)

**Docstring Quality:**
- ✅ Includes LaTeX formula: σ_E(q) = ⟨q, c⟩ + r‖A^{-1/2}Rq‖
- ✅ References theory.txt §2.2
- ✅ Explains Riesz map role in Hilbert setting
- ✅ Documents parameters with types and array shapes

**Test Coverage:**
- ✅ test_ellipsoid_support_convexity()
- ✅ test_ellipsoid_support_homogeneity()
- ✅ test_ellipsoid_support_point_on_boundary()
- ✅ test_ellipsoid_support_zero_query()
- ⚠️ Missing: test for non-isotropic ellipsoids (A ≠ I) - recommend adding

**Recommendations:**
- Add overflow assertion: `assert radius < 1e10, "Radius too large for numerical stability"`
- Add test: `test_ellipsoid_support_anisotropic()` with A = diag([1, 100, 0.01])
- Consider caching `A.inverse.sqrt` for repeated queries (performance optimization)

**Overall:** Implementation is mathematically correct and well-documented. Minor improvements suggested for numerical robustness.
```

---

## Conclusion

This proposal provides:
1. **Theory-Validator-subagent:** Automated math verification
2. **Enhanced existing agents:** Oracle (theory research), Sisyphus (validation step), Code-Review (validator invocation)
3. **Living documentation:** theory_map.md, theory_papers_index.md
4. **Integrated workflow:** Theory checkpoints in every phase
5. **Success metrics:** Track mathematical correctness, theory coverage

**Next Step:** Approve proposal → Implement Theory-Validator-subagent.agent.md → Test on Phase 4 of dual_master_implementation → Iterate based on findings.
