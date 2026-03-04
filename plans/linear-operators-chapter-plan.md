## Plan: Linear Operators Chapter

Fill the manual’s “Linear Operators” chapter scaffold with Riesz-first, API-faithful content: clearly distinguish $G$, $G'$ (Banach dual), and $G^*$ (Hilbert adjoint); explain standard vs Galerkin matrix representations; include one simple worked example using `LinearOperator.from_matrix`; briefly discuss the advanced randomized routines; and finish with a full commutative diagram based on the provided TikZ starting point. The kernel/range subsection is removed.

**Phases**
1. **Phase 1: Chapter Core + Notation ↔ API**
    - **Objective:** Replace the current outline comments with complete definitions and consistent notation tied to the actual `pygeoinf` API.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex` (Linear Operators chapter); reference `pygeoinf/pygeoinf/linear_operators.py`, `pygeoinf/pygeoinf/hilbert_space.py`.
    - **Tests to Write:** Manual build check (2 LaTeX passes) with no new errors.
    - **Steps:**
        1. Write the motivation paragraph (operators as forward maps in inverse problems).
        2. Add a “Notation ↔ API” box mapping: `LinearOperator`, `.dual`, `.adjoint`, `to_dual`, `from_dual`.
        3. Define the evaluation convention `G(x)` and the operator type signature (domain/codomain).

2. **Phase 2: Dual vs Adjoint (Riesz-first derivation)**
    - **Objective:** Make $G$, $G'$, and $G^*$ distinct in type and meaning, and align the text with how the library derives one from the other via Riesz maps.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`; reference `LinearOperator(domain, codomain, mapping, dual_mapping=..., adjoint_mapping=...)`.
    - **Tests to Write:** Manual build check.
    - **Steps:**
        1. Define $G':\mathcal{D}'\to\mathcal{M}'$ and $G^*: \mathcal{D}\to\mathcal{M}$, with careful typing (linear forms vs primal vectors).
        2. State and explain the identity matching the code path: $G^* = \mathcal{R}_{\mathcal{M}} \circ G' \circ \mathcal{R}_{\mathcal{D}}^{-1}$.
        3. Add a pitfall box: dual vs adjoint confusion, and the role of the Riesz maps.

3. **Phase 3: Matrices and Coordinates (standard vs Galerkin)**
    - **Objective:** Explain `matrix(..., galerkin=False)` vs `matrix(..., galerkin=True)` precisely, without assuming orthonormality, and clarify where “symmetry/self-adjointness” lives.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`; reference `LinearOperator.matrix` and coordinate maps in `HilbertSpace`.
    - **Tests to Write:** Manual build check.
    - **Steps:**
        1. Document the standard component-to-component representation (`galerkin=False`) and that its `rmatvec` corresponds to the Banach-dual operator.
        2. Document the Galerkin representation (`galerkin=True`) and that its `rmatvec` corresponds to the Hilbert adjoint.
        3. Add a pitfall box: transpose intuition failures plus Galerkin vs standard matrix mismatch.

4. **Phase 4: Simple Worked Example (Option A: from_matrix end-to-end)**
    - **Objective:** Provide one concrete example demonstrating construction, evaluation, dual action, adjoint action, and both matrix views.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`; reference `LinearOperator.from_matrix` (and `EuclideanSpace` as needed).
    - **Tests to Write:** Manual build check.
    - **Steps:**
        1. Introduce a tiny, explicit matrix-backed operator via `LinearOperator.from_matrix`.
        2. Demonstrate what inputs each of `G`, `G.dual`, and `G.adjoint` consumes (primal vector vs linear form vs primal vector).
        3. Show both matrix modes (`galerkin=False` and `galerkin=True`) and explain what each represents.

5. **Phase 5: Constructing Operators in pygeoinf (implementer guidance)**
    - **Objective:** Convert the “Constructing operators” outline into actionable guidance for library users/implementers.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`; reference `LinearOperator` constructor and factories (especially `from_formal_adjoint`).
    - **Tests to Write:** Manual build check.
    - **Steps:**
        1. Explain when to provide `dual_mapping` vs `adjoint_mapping`, and what the library derives via Riesz maps.
        2. Explain “formal adjoint” vs “Hilbert adjoint” and connect to mass-weighted geometry (`MassWeightedHilbertSpace`, `LinearOperator.from_formal_adjoint`).
        3. Mention `from_matrix` and `self_adjoint_from_matrix` as safe, explicit construction paths.

6. **Phase 6: Advanced Routines (brief, oriented)**
    - **Objective:** Briefly discuss the advanced randomized routines as tools for approximation/diagnostics (not a full algorithms chapter).
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`; reference `LinearOperator.random_svd`, `random_eig`, `random_cholesky`.
    - **Tests to Write:** Manual build check.
    - **Steps:**
        1. Explain at a high level what each routine returns (low-rank factors / approximate eigensystem / approximate Cholesky factor).
        2. State intended use-cases (structure discovery, cheap surrogates, operator diagnostics, preconditioning intuition).
        3. Add a short caution about approximation and dependence on the chosen geometry.

7. **Phase 7: Full Commutative Diagram (TikZ starting point)**
    - **Objective:** Replace the “full commutative diagram” stub with a diagram and short explanation of what commutes and why, using the provided TikZ diagram as the baseline.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`.
    - **Tests to Write:** Manual build check.
    - **Steps:**
        1. Incorporate the diagram and adjust only notation needed to match the manual’s established symbols.
        2. Add a concise caption/paragraph emphasizing $G^*=\mathcal{R}_{\mathcal{M}}\circ G'\circ \mathcal{R}_{\mathcal{D}}^{-1}$ and the role of analysis/synthesis maps.
        3. Ensure finite-dimensional surrogates are clearly framed as surrogates, preserving the manual’s infinite-dimensional emphasis.

**Open Questions**
1. None remaining for the example choice (Option A approved).