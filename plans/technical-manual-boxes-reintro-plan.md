## Plan: Reintroduce Technical Manual Boxes

Re-add the previously removed pedagogical/API/example boxes to the LaTeX technical manual while preserving your newer, more intuitive + rigorous exposition. The key is to restore “bridge to code” content without reverting the narrative to a purely API-driven style.

**Phases**
1. **Phase 1: Notation-to-API alignment**
    - **Objective:** Ensure the manual’s operators ($\Pi,\pi,\pi^*,\pi',\Pi'$) map cleanly to current `pygeoinf` objects, consistent with your decision that $\Pi$ is the primary coefficient route.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`
    - **Tests to Write:** LaTeX build check (two-pass `pdflatex`)
    - **Steps:**
        1. Locate the rewritten sections defining $\Pi$ and $\pi$ and confirm the intended meanings (coefficients vs inner-product vector).
        2. Verify the corresponding implementation points in code: `HilbertSpace.coordinate_projection`, `HilbertSpace.coordinate_inclusion`, and `LinearOperator.adjoint` / `.dual` behavior.
        3. Adjust manual wording as needed to avoid claiming non-existent APIs (e.g., no `gram_matrix()` method; Gram/mass is a composition).

2. **Phase 2: Reintroduce missing boxes (pedagogical + accurate)**
    - **Objective:** Restore the removed boxes at the correct conceptual points, updated to reflect current `pygeoinf` implementation.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`
    - **Tests to Write:** LaTeX build check (two-pass `pdflatex`)
    - **Steps:**
        1. Add a `codebox` near the first introduction of analysis/synthesis mapping $\Pi,\pi$ to `coordinate_projection` / `coordinate_inclusion`, clarifying that $\pi^*$ corresponds to `coordinate_inclusion.adjoint`.
        2. Add a `codebox` in the Gram/finite-analysis discussion explaining the computational recipe used in `pygeoinf` (apply `pi.adjoint` to get the inner-product vector, then solve a Gram/mass system to obtain coefficients).
        3. Add a `whybox` in the finite-dimensional approximation section emphasizing the conceptual link between infinite-dimensional theory and finite computation (convergence as `dim` increases), without over-specifying an $N$-truncation API that `pygeoinf` does not expose directly.
        4. Re-add a worked `examplebox` in “Implementing your own HilbertSpace”, but update it to the current canonical pattern: using `MassWeightedHilbertSpace` (optionally with a diagonal mass operator) to model a quadrature-weighted $L^2$ inner product.

3. **Phase 3: Compile and polish**
    - **Objective:** Ensure the updated manual compiles and the added boxes are consistent, readable, and aligned with the pedagogical tone.
    - **Files/Functions to Modify/Create:** `pygeoinf/theory/TECHNICAL_MANUAL.tex`
    - **Tests to Write:** LaTeX build check (two-pass `pdflatex`)
    - **Steps:**
        1. Run `pdflatex` twice and resolve any LaTeX issues introduced by the edits.
        2. Consistency sweep for primes vs stars and for dual vs adjoint terminology.

**Open Questions**
1. Should the reintroduced API `codebox`es mention `to_components/from_components` explicitly, or prefer `coordinate_projection/coordinate_inclusion` as the primary API names?
