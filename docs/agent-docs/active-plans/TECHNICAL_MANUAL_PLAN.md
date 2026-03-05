# Technical Manual Plan: pygeoinf User & Theory Manual

**Created:** 2026-02-27
**Format:** LaTeX (.tex), compiling to PDF
**Style:** Gentle-then-rigorous (each chapter opens with motivation/intuition, then full mathematical rigor)
**Code integration:** Inline `codebox` callouts mapping math → API (same pattern as existing `theoretical_manual.txt`)
**Scope:** pygeoinf only (no intervalinf coverage, though EuclideanSpace examples throughout)

---

## TL;DR

Build a ~100+ page LaTeX technical manual for pygeoinf organized in a layered, bottom-up structure: foundational concepts (spaces, operators, forms) → geometric structures (subsets, subspaces, convex analysis) → probabilistic structures (Gaussian measures) → problem formulation (forward problems) → inversion algorithms (Bayesian, deterministic, Backus-Gilbert) → advanced topics (visualization, computational methods). Each chapter follows a consistent template: *Why this matters* → *Intuitive explanation* → *Rigorous mathematics* → *Code mapping* → *Worked examples*. The existing `theoretical_manual.txt` content is integrated (with corrections) into the appropriate chapters. Implementation is phased: skeleton first, then chapter-by-chapter expansion.

---

## Document Structure

### Front Matter
- **Title page** — "pygeoinf: Technical Manual — Theory, Algorithms, and API Reference"
- **Preface** — Who this manual is for, how to read it (two tracks: intuitive vs. rigorous), notation guide
- **Table of contents**
- **List of notation** — Comprehensive symbol table (spaces, operators, maps, matrices)

### Part I: Foundations (Chapters 1–5)

These chapters establish the mathematical vocabulary required to use pygeoinf. They answer the question every geophysicist will have: *"Why do I need to think about spaces and duals? Can't I just pass in a matrix?"*

---

#### Chapter 1: Introduction and Motivation
- **1.1 What is pygeoinf?** — Package goals, design philosophy, what makes it different
- **1.2 The geophysical inversion problem** — Informal description: you have data, you want a model, the problem is ill-posed
- **1.3 Why abstract spaces matter** — The central pitch: working at the level of Hilbert spaces (not just matrices) gives you correct adjoints, proper uncertainty quantification, and basis-independent algorithms. Anticipate and address the "why bother?" objection head-on with a concrete example showing how naïve matrix inversion gives wrong uncertainty when the inner product is non-trivial.
- **1.4 Package architecture overview** — Layer diagram (foundation → geometry → algorithms → output), dependency graph
- **1.5 How to read this manual** — Two-track guide: practitioners can follow motivation + codeboxes; theorists read the full proofs
- **1.6 Notation and conventions** — Complete notation table (boldface for vectors/matrices, calligraphic for spaces, etc.)

---

#### Chapter 2: Hilbert Spaces
*This chapter covers the most fundamental concept in pygeoinf. Every other piece — operators, forms, measures, inversions — builds on it.*

- **2.1 Why Hilbert spaces?** — Intuitive explanation: a Hilbert space is a vector space with a notion of length and angle. Why geophysicists need it: the choice of inner product encodes your beliefs about what "close" means in model space.
  - **Motivating example:** Two models that look identical in ℓ² norm but very different in a Sobolev norm — the inner product matters!

- **2.2 Definition and axioms** — Full rigorous definition: vector space + inner product + completeness. Separability. Real vs. complex (pygeoinf is real-only).

- **2.3 Inner products, norms, and distances** — Definitions, Cauchy-Schwarz, parallelogram law. Connection to `HilbertSpace.inner_product()`, `norm()`, `distance()`.

- **2.4 The dual space and dual pairing** — What is a continuous linear functional? The dual space M'. The pairing ⟨·,·⟩ vs. the inner product (·,·). Why they are different and when they coincide.
  - **Key insight for practitioners:** "A dual vector is a measurement — it takes a model and returns a number."

- **2.5 The Riesz representation theorem** — Statement, proof sketch, geometric interpretation. The Riesz map R: M' → M and its inverse R⁻¹: M → M'. Why this is the most important theorem in the package.
  - **Codebox:** `to_dual` ≡ R⁻¹, `from_dual` ≡ R

- **2.6 Bases and representations** — Riesz bases (not just orthonormal!), biorthogonal duals, coefficient sequences in ℓ². The analysis operator Π and synthesis operator π.
  - **Codebox:** `to_components` ≡ Π, `from_components` ≡ π

- **2.7 The Gram matrix** — Definition, properties (SPD), connection between basis and biorthogonal basis via M⁻¹. The formula Π = M⁻¹ π*.
  - **Codebox:** `coordinate_projection`, `coordinate_inclusion`

- **2.8 Finite-dimensional approximation** — The subspace M_N = span{φ₁,...,φ_N}, truncated analysis/synthesis, the projection P_N = π_N Π_N.

- **2.9 Concrete spaces in pygeoinf**
  - **2.9.1 EuclideanSpace** — ℝⁿ with standard inner product. The simplest case: Riesz map = identity.
  - **2.9.2 MassWeightedHilbertSpace** — Weighted inner product ⟨u,v⟩_M = ⟨Mu,v⟩_H. Why this matters: Sobolev spaces, non-uniform grids.
  - **2.9.3 HilbertSpaceDirectSum** — Product space H₁ ⊕ H₂.
  - **2.9.4 Implementing your own HilbertSpace** — Step-by-step guide to subclassing, the 8 abstract methods, common pitfalls.

- **2.10 Summary diagram** — Commutative diagram showing M, M', ℝⁿ, (ℝⁿ)', Riesz maps, analysis/synthesis, and how they all relate. (Refine and integrate the existing tikz-cd diagram from theoretical_manual.txt)

---

#### Chapter 3: Linear Forms (Dual Vectors)
*A linear form is the mathematical representation of a single measurement.*

- **3.1 Motivation** — A seismometer measures a linear functional of the displacement field. A gravity measurement is a linear functional of the density field. Linear forms are *data* at the abstract level.

- **3.2 Definition** — A continuous linear functional ℓ: M → ℝ. The dual space M' revisited.

- **3.3 Representation in coordinates** — Components [ℓ(φ₁), ..., ℓ(φ_N)]. The relationship ℓ(m) = ⟨ℓ, m⟩ = cᵀ M⁻¹ m_coeffs.

- **3.4 Gradient and Hessian** — For a linear form: gradient = Riesz representative, Hessian = 0. Why this matters for optimization later.

- **3.5 Linear forms as rank-1 operators** — `as_linear_operator()`: viewing ℓ as an operator M → ℝ.
  - **Codebox:** `LinearForm`, `DualVector`, `components`, `as_linear_operator()`

---

#### Chapter 4: Linear Operators
*The forward operator G connecting models to data is the heart of geophysical inversion.*

- **4.1 Motivation** — The forward operator maps a model to predicted data. Examples from geophysics: gravity, seismology, magnetics. The fundamental equation d = Gm + noise.

- **4.2 Definition and properties** — Bounded linear maps between Hilbert spaces. Domain, codomain, composition, operator algebra (+, scalar ×, @).

- **4.3 The Banach dual (transpose) G'** — Definition via the pairing: ⟨G'd', m⟩ = ⟨d', Gm⟩. Why this is NOT the same as the adjoint when inner products are non-trivial!
  - **Key insight:** "If you're used to thinking of the transpose as the adjoint, that's only true in Euclidean space. On general Hilbert spaces, the distinction matters — and getting it wrong leads to incorrect inversions."

- **4.4 The Hilbert adjoint G\*** — Definition: (Gm, d)_D = (m, G*d)_M. The fundamental commuting relation G* = R_M ∘ G' ∘ R_D⁻¹. Full derivation.
  - **Codebox:** `LinearOperator.adjoint` vs. `LinearOperator.dual`

- **4.5 Matrix representations** — Standard matrix A = Π_D G π_N (components to components). Galerkin matrix. The discrete adjoint formula A* = M_φ⁻¹ Aᵀ M_b.
  - **Codebox:** `L.matrix(galerkin=False)` vs `L.matrix(galerkin=True)`

- **4.6 Operators as stacks of linear forms** — G as [ℓ₁, ..., ℓ_M]ᵀ where each ℓᵢ is a measurement functional.
  - **Codebox:** `LinearOperator.from_linear_forms()`

- **4.7 Kernel and range** — Null space ker(G), range im(G), the fundamental decomposition m = m_null + m_data.

- **4.8 Constructing operators in pygeoinf** — From callables, from matrices, from linear forms. Providing vs. deriving adjoint/dual mappings.

- **4.9 The big commutative diagram** — Full diagram with M, D, M', D', ℝⁿ, ℝᴺᵈ, all Riesz maps, analysis/synthesis, G, G', G*, reusing and correcting the tikz-cd figure from the existing manual.

---

#### Chapter 5: Nonlinear Operators and Forms
*Brief chapter for completeness — the package supports nonlinear operators and forms, used primarily in optimization contexts.*

- **5.1 NonLinearOperator** — Definition, interface (`__call__`, `jacobian`, `adjoint_jacobian`).
- **5.2 NonLinearForm** — Scalar-valued nonlinear functions on Hilbert spaces. Used by support functions and cost functions.
- **5.3 Gradient and Hessian** — Fréchet derivatives, connection to optimization.

---

### Part II: Geometry and Structure (Chapters 6–9)

These chapters cover the geometric objects that constrain and describe what we know about models.

---

#### Chapter 6: Subsets and Convex Sets
*Prior information about the model — bounds, constraints, regularity — is encoded as convex sets.*

- **6.1 Why convex sets?** — Prior information is geometric: "the model lies in some set." Convexity is the computationally tractable case. Examples: bound constraints, energy bounds, smoothness constraints.

- **6.2 The Subset ABC** — `is_element()`, `project()`, `boundary`.

- **6.3 Balls and Ellipsoids** — Definition, projection formulas, usage for norm constraints.
  - **Codebox:** `Ball`, `Ellipsoid`

- **6.4 Half-spaces and polyhedral sets** — Linear inequality constraints. Intersections of half-spaces.
  - **Codebox:** `HalfSpace`, `PolyhedralSet`

- **6.5 EmptySet and FullSpace** — Degenerate cases.

- **6.6 Set operations** — Intersection (and its limitations in general). Minkowski sums (via support functions).

---

#### Chapter 7: Subspaces
*Linear and affine subspaces represent equality constraints: "the model satisfies these exact relationships."*

- **7.1 Motivation** — Linear constraints arise from conservation laws, boundary conditions, known relationships between parameters.

- **7.2 Orthogonal projectors** — Definition, construction from basis vectors, properties (P² = P = P*).
  - **Codebox:** `OrthogonalProjector`, `OrthogonalProjector.from_basis()`

- **7.3 Linear subspaces** — As subsets with projectors.
  - **Codebox:** `LinearSubspace`

- **7.4 Affine subspaces** — Translation of a linear subspace. The solution set of a linear equation Bu = w.
  - **Codebox:** `AffineSubspace`, `AffineSubspace.from_linear_equation()`

- **7.5 Tangent spaces and dimension** — `get_tangent_basis()`, tolerant Gram-Schmidt.

---

#### Chapter 8: Convex Analysis and Support Functions
*The mathematical engine for deterministic inversion. Support functions dually characterize convex sets.*

- **8.1 Why convex analysis?** — The deterministic linear inference (DLI) framework characterizes what we can and cannot learn about the model *without probability*. Support functions are the key tool.

- **8.2 Support functions — intuition** — Geometric picture: σ_C(q) = max_{x∈C} ⟨q,x⟩ is "how far C extends in direction q." The support function encodes the shape of C completely.

- **8.3 Formal definition and properties** — Convexity, positive homogeneity, subadditivity. Conjugacy (Fenchel-Legendre).

- **8.4 Subgradients and support points** — ∂σ_C(q) = argmax, geometric interpretation.

- **8.5 Concrete support functions**
  - **8.5.1 BallSupportFunction** — σ(q) = ⟨q,c⟩ + r‖q‖
  - **8.5.2 EllipsoidSupportFunction**
  - **8.5.3 PolyhedralSupportFunction**
  - **8.5.4 SobolevBallSupportFunction**
  - **Codebox** for each

- **8.6 Combinators** — Infimal convolution (Minkowski sum), sum (intersection), scaled sum. Mathematical properties and code.
  - **Codebox:** `InfimalConvolution`, `SupportFunctionSum`, `SupportFunctionScaledSum`

- **8.7 The dual master equation** — The central theoretical result:
  h_U(q) = inf_λ { ⟨λ, d̃⟩ + σ_B(T'q - G'λ) + σ_V(-λ) }
  Full derivation, geometric interpretation, connection to Al-Attar & Crawford (2021).
  - **Codebox:** `DualMasterCostFunction` in `backus_gilbert.py`

---

#### Chapter 9: Gaussian Measures
*The probabilistic framework for Bayesian inversion.*

- **9.1 Why Gaussian measures?** — The natural probabilistic building block for linear problems. If your prior is Gaussian and your forward operator is linear and your noise is Gaussian, the posterior is Gaussian. The full posterior is computable in closed form.

- **9.2 Gaussian measures on Hilbert spaces** — Mean, covariance operator. Difference from finite-dimensional Gaussian (covariance is an *operator*, not a matrix). Cameron-Martin space.

- **9.3 Sampling** — How to draw samples from a Gaussian measure. Cholesky-based and randomized approaches.

- **9.4 Push-forward and marginals** — If m ~ N(μ, C) and A is linear, then Am ~ N(Aμ, ACA*). Connection to data prediction.
  - **Codebox:** `push_forward()`, `marginals()`

- **9.5 Conditioning (Bayesian update)** — The posterior formula: given d = Gm + ε with ε ~ N(0, Γ), the posterior is Gaussian with known mean and covariance. Full derivation.
  - **Codebox:** `GaussianMeasure.condition()`

---

### Part III: Problems and Algorithms (Chapters 10–14)

These chapters cover the problem formulations and solution algorithms.

---

#### Chapter 10: Forward Problems
*Packaging the model space, data space, forward operator, and noise into a single object.*

- **10.1 The forward modelling framework** — d = Gm + ε. The forward problem as a data structure.
- **10.2 ForwardProblem and LinearForwardProblem** — Constructor, properties, validation.
  - **Codebox:** `ForwardProblem`, `LinearForwardProblem`

---

#### Chapter 11: Deterministic Inversion (Optimisation-Based)
*When you don't have (or don't want) a probabilistic prior.*

- **11.1 Least-squares inversion** — min ‖Gm - d‖². The normal equations. Uniqueness and non-uniqueness.
  - **Codebox:** `LinearLeastSquaresInversion`

- **11.2 Minimum-norm inversion** — min ‖m‖ subject to Gm = d. The pseudo-inverse. Why the norm matters (connects back to Chapter 2).
  - **Codebox:** `LinearMinimumNormInversion`

- **11.3 Constrained inversions** — Adding affine constraints Bu = w. The constrained least-squares and minimum-norm problems.

- **11.4 Connection to the dual master equation** — How the deterministic inversions are special cases of the general DLI framework (Chapter 8).

---

#### Chapter 12: Bayesian Inversion
*When you have a probabilistic prior.*

- **12.1 The Bayesian framework for inverse problems** — Prior measure → forward model → likelihood → posterior measure. Stuart (2010) well-posedness.

- **12.2 Linear Bayesian inversion** — Closed-form posterior: Kalman gain formula. Posterior mean = MAP estimate (in the linear-Gaussian case).
  - **Codebox:** `LinearBayesianInversion`

- **12.3 Constrained Bayesian inversion** — Adding hard constraints (affine subspace). The conditional Gaussian.
  - **Codebox:** `ConstrainedLinearBayesianInversion`

- **12.4 Posterior uncertainty** — Posterior covariance, marginal variances, confidence ellipsoids.

---

#### Chapter 13: Backus-Gilbert and DLI Inference
*The crown jewel: deterministic inference using convex analysis.*

- **13.1 The inference problem** — We don't want the full model m; we want properties p = Tm. What can we learn about p given data?

- **13.2 Backus-Gilbert estimators** — Averaging kernels, the resolution-variance trade-off. Classical approach.
  - **Codebox:** `BackusGilbertInversion`

- **13.3 The admissible property set U** — The set of all property values consistent with the data and prior constraints. Characterized by support function h_U(q).

- **13.4 Computing U via the dual master equation** — Numerical evaluation of h_U. Gradient-based optimization. Support points as extreme models.

- **13.5 Polytope outer approximation** — Approximating U by a polytope. Three operational regimes: underdetermined, critical, overdetermined. Connection to visualization.

---

#### Chapter 14: Convex Optimisation
*Algorithms for solving the optimisation problems arising in Chapters 11–13.*

- **14.1 Overview of optimisation in pygeoinf** — What problems arise and which solvers handle them.
- **14.2 Linear solvers** — CG, MINRES, direct methods.
  - **Codebox:** `LinearSolver`
- **14.3 Convex optimisation algorithms** — Bundle methods, proximal methods.
  - **Codebox:** `convex_optimisation.py` classes
- **14.4 Preconditioners** — Why preconditioning matters, available preconditioners.

---

### Part IV: Computational Methods and Visualization (Chapters 15–16)

---

#### Chapter 15: Computational Utilities
*Randomized linear algebra, parallel computation, and helper functions.*

- **15.1 Randomized matrix decompositions** — Randomized SVD, range finder, Cholesky. When and why to use them.
  - **Codebox:** `random_matrix.py`
- **15.2 Direct sum operators** — Block-diagonal structures.
  - **Codebox:** `direct_sum.py`
- **15.3 Parallel computation** — `parallel.py` helpers.
- **15.4 Auxiliary functions** — Mathematical helpers in `auxiliary.py`.

---

#### Chapter 16: Visualization
*Seeing the geometry of inverse problems.*

- **16.1 The SubspaceSlicePlotter** — Visualizing convex sets by slicing with low-dimensional affine subspaces.
  - **Codebox:** `SubspaceSlicePlotter`
- **16.2 1D slices** — Bar plots of intervals.
- **16.3 2D slices** — Polygon rendering (exact for polyhedral, raster for others).
- **16.4 3D slices** — Surface rendering.
- **16.5 Worked example** — Visualizing the admissible property set from a DLI inversion.

---

### Back Matter
- **Appendix A: Convex Analysis Primer** — Self-contained introduction to convex analysis for readers who need it. Definitions, key theorems (separation, Fenchel duality), examples. (Draws from theory.txt appendices)
- **Appendix B: Functional Analysis Prerequisites** — Banach spaces, weak topologies, compact operators — the background needed for the rigorous parts.
- **Appendix C: Proofs** — Deferred proofs that would interrupt the main narrative.
- **Appendix D: Complete API Quick Reference** — One-page-per-module cheat sheet of all classes and methods.
- **Bibliography** — All papers in theory/, plus standard references (Rockafellar, Brezis, Stuart, etc.)
- **Index**

---

## Issues in Existing theoretical_manual.txt to Fix During Integration

1. **Duplicate equation labels** — `eq:gram-def`, `eq:PG-proj`, `eq:gramN`, `eq:Riesz-RN`, `eq:MN-def`, `eq:PiN-piN` appear twice.
2. **Duplicate content** — The "Computational surrogate" paragraph near the end repeats definitions from earlier (M_N, Π_N, π_N, P_N, Gram matrix, Riesz on ℝᴺ).
3. **Incomplete standard matrix formula** — The equation for G_d uses undefined operator `Pi_D G i_M π_N` without defining `i_M` (inclusion operator). Needs clarification.
4. **Galerkin adjoint formula** — Equation has a spurious double `=` sign and uses `A` vs `G_d` inconsistently.
5. **Missing definition** — The `A` matrix in the Galerkin adjoint formula is not formally related to `G_d`.
6. **Title mismatch** — Document title is "Relations Between Spaces" but it covers much more. Will be absorbed into the new structure.
7. **Missing bibliography file** — References `bibliography.bib` which doesn't exist yet.

---

## Implementation Plan

### Phase 0: Skeleton and Infrastructure (Priority: First)
- **Objective:** Create the LaTeX project structure, build system, and document skeleton with all chapter/section headings.
- **Files to create:**
  - `pygeoinf/theory/manual/main.tex` — Master document with `\input` for each chapter
  - `pygeoinf/theory/manual/preamble.tex` — Shared packages, commands, environments (codebox, remark, etc.)
  - `pygeoinf/theory/manual/ch01_introduction.tex` through `ch16_visualization.tex` — One file per chapter (stub with section headings only)
  - `pygeoinf/theory/manual/appendixA.tex` through `appendixD.tex`
  - `pygeoinf/theory/manual/bibliography.bib`
  - `pygeoinf/theory/manual/Makefile` or `latexmk` config
- **Tests:** Document compiles cleanly with `latexmk -pdf main.tex`

### Phase 1: Chapter 1 — Introduction and Motivation
- **Objective:** Write the complete introduction chapter (motivation, philosophy, architecture, notation).
- **Content:** All of §1.1–1.6 as outlined above.
- **Key deliverable:** The "Why abstract spaces matter" section with a concrete worked example showing incorrect results from naïve matrix inversion.

### Phase 2: Chapter 2 — Hilbert Spaces
- **Objective:** Write the foundational Hilbert space chapter, integrating and correcting content from the existing theoretical_manual.txt §1–2.
- **Existing content to integrate:** Riesz bases, biorthogonal duals, analysis/synthesis, Gram operator, inner products, dual pairings, Riesz maps, all codeboxes, the commutative diagram.
- **Fixes:** Deduplicate definitions, fix equation labels, clarify the analysis-via-Gram formula.
- **New content:** §2.1 motivation, §2.2 axioms, §2.9 concrete spaces guide, §2.10 refined diagram.

### Phase 3: Chapters 3–4 — Linear Forms and Linear Operators
- **Objective:** Write these two tightly related chapters.
- **Existing content to integrate:** Linear operators section from theoretical_manual.txt §3 (dual vs. adjoint, derivation, matrix representations, operators as forms).
- **Fixes:** Fix the Galerkin adjoint formula, clarify G_d notation, define inclusion operator.
- **New content:** Motivation sections, kernel/range discussion, operator construction guide, full worked example.

### Phase 4: Chapter 5 — Nonlinear Operators and Forms
- **Objective:** Brief chapter covering the nonlinear operator/form interfaces.
- **Content:** Relatively short — interface descriptions, gradient/Hessian, connection to later optimization chapters.

### Phase 5: Chapters 6–7 — Subsets and Subspaces
- **Objective:** Write the geometric structure chapters.
- **Content:** All convex set types, projectors, affine subspaces. Geometric intuition with figures.

### Phase 6: Chapter 8 — Convex Analysis and Support Functions
- **Objective:** Comprehensive treatment of support functions and the dual master equation.
- **Source material:** Heavy drawing from theory.txt §1–7 and appendices.
- **Content:** This is one of the most important chapters — full derivation of the master equation, all concrete support function types, combinators.

### Phase 7: Chapter 9 — Gaussian Measures
- **Objective:** Probabilistic framework chapter.
- **Source material:** Stuart (2010), theory.txt §4 specializations.
- **Content:** Gaussian measures on Hilbert spaces, sampling, push-forward, conditioning.

### Phase 8: Chapters 10–11 — Forward Problems and Deterministic Inversion
- **Objective:** Problem formulation and deterministic solution methods.
- **Content:** Forward problem packaging, least-squares, minimum-norm, constrained variants. The least-norm solution from existing manual integrated here.

### Phase 9: Chapters 12–13 — Bayesian Inversion and Backus-Gilbert/DLI
- **Objective:** The main inversion algorithm chapters.
- **Source material:** Stuart (2010), Al-Attar & Crawford (2021), Backus (1970), theory.txt §8–12.
- **Content:** Full Bayesian derivation, Backus-Gilbert, admissible sets, polytope approximation.

### Phase 10: Chapters 14–16 — Optimization, Computation, Visualization
- **Objective:** The remaining chapters on algorithms and output.
- **Content:** Solver interfaces, randomized methods, SubspaceSlicePlotter, worked visualization example.

### Phase 11: Appendices and Back Matter
- **Objective:** Write appendices (convex analysis primer, functional analysis prerequisites, deferred proofs, API reference), compile bibliography, build index.
- **Source material:** theory.txt appendices A–E, all papers in theory/.

### Phase 12: Final Review and Polish
- **Objective:** Cross-reference check, notation consistency, figure quality, compile test, proof-read.

---

## Chapter Template

Each chapter follows this consistent structure:
```
\chapter{Title}

\section{Why [topic] matters}
% 0.5–1 page of motivation and intuition
% Addresses: "Why would I bother with this?"
% Uses a concrete geophysical example

\section{Definitions}
% Full rigorous mathematical definitions
% Theorems with proof sketches (full proofs in Appendix C if long)

\section{Key properties and results}
% Important theorems, formulas, relationships

\section{Implementation in pygeoinf}
% codebox environments mapping every concept to API calls
% Constructor signatures, key methods

\section{Worked example}
% A complete mini-example: define spaces, create operators, run algorithm
% Shows both the math and the code side-by-side
```

---

## Estimated Size

| Part | Chapters | Est. Pages |
|------|----------|-----------|
| Front matter | — | 5 |
| Part I: Foundations | Ch. 1–5 | 45–55 |
| Part II: Geometry | Ch. 6–9 | 30–40 |
| Part III: Algorithms | Ch. 10–14 | 35–45 |
| Part IV: Computation | Ch. 15–16 | 10–15 |
| Appendices | A–D | 20–30 |
| **Total** | | **~145–190 pages** |

---

## Open Questions

1. **Figures/diagrams budget** — Should we invest in high-quality TikZ figures for geometric intuition (convex sets, projections, support hyperplanes), or keep figures minimal and focus on equations? *Recommendation: Invest in key diagrams — they are worth 1000 equations for the geophysicist audience.*
2. **Worked examples** — Should worked examples use a single running geophysical scenario (e.g. gravity inversion) threaded through the whole manual, or independent examples per chapter? *Recommendation: One running example + chapter-specific extras.*
3. **Cross-references to theory.txt** — Should the manual supersede theory.txt entirely, or reference it as a companion "research notes" document? *Recommendation: The manual should be self-contained; theory.txt remains as detailed derivation scratch notes for developers.*
4. **PDF-only or also HTML?** — LaTeX can be compiled to HTML via tools like `tex4ht` or `pandoc`. Is HTML output desired? *Recommendation: PDF primary, HTML as a future nice-to-have.*
5. **Version coupling** — Should the manual be versioned with the package (e.g. "Manual for pygeoinf v0.5")? *Recommendation: Yes, include git tag/version on title page.*
