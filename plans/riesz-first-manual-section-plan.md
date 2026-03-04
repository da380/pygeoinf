## Plan: Riesz-First in Technical Manual

Add a short, explicit “Riesz-first” explanation immediately after the Riesz representation theorem section, so the manual matches the core pygeoinf design: implement `to_dual`/`from_dual` first, then derive the inner product, norms, and adjoints from them.

**Phases**
1. **Phase 1: Insert Riesz-first design section + rebuild**
    - **Objective:** Explain that `to_dual`/`from_dual` are the primitive structure; `inner_product` is derived from the duality pairing; and operator adjoints depend on the Riesz maps. Place this right after the Riesz representation theorem (before the basis discussion).
    - **Files/Functions to Modify/Create:**
        - `pygeoinf/theory/TECHNICAL_MANUAL.tex` (insert section + codebox)
        - `pygeoinf/plans/pygeoinf-reference.md` (correct `HilbertSpace` API description: `inner_product` is derived; `to_dual`/`from_dual` are abstract)
    - **Tests to Write:** None (documentation-only change)
    - **Steps:**
        1. Add a new section after the Riesz theorem explaining:
            - The inner product is constructed as $(x,y)_H := \langle \mathcal{R}^{-1}x, y\rangle$.
            - In pygeoinf, `HilbertSpace.to_dual` and `HilbertSpace.from_dual` define the geometry; `HilbertSpace.inner_product` uses them.
            - The Hilbert adjoint of operators is computed via the spaces’ Riesz maps; changing Riesz changes all adjoints consistently.
        2. Add a compact “notation ↔ code” codebox mapping the Riesz maps and inner product relationship to the API.
        3. Update the package reference to match the actual code contract (Riesz-first, derived inner product).
        4. Rebuild the technical manual (2 pdflatex passes) to ensure compilation.

**Open Questions**
1. None (placement clarified: immediately after the Riesz representation theorem section).
