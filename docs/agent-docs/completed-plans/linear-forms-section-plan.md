## Plan: Linear Forms section (components + geometry)

Rewrite Chapter “Linear Forms” to focus on the component representation of functionals and how it interacts with the domain’s coordinate maps and geometry (Riesz maps). Avoid presenting linear forms primarily as rank-1 operators; if `as_linear_operator` is mentioned, treat it as an adapter for interoperability.

**Phases**
1. **Phase 1: Rewrite the Linear Forms chapter content**
    - **Objective:** Replace placeholder text with a components/geometry-first narrative, including a concrete motivating example, and explicitly document both ways to define a form (components or callable mapping).
    - **Files/Functions to Modify/Create:**
        - `pygeoinf/theory/TECHNICAL_MANUAL.tex` (Chapter “Linear Forms”, sections: motivation/definition/coordinates/Riesz representative; rename or refocus the “rank-1 operators” section)
    - **Tests to Write:** None (documentation-only)
    - **Steps:**
        1. Motivation: state that each datum is evaluation of a linear functional; include one concrete example (e.g. point sensor or averaging kernel) without overcommitting to a specific physics.
        2. Definition: define a continuous linear functional `\ell \in H'` and duality pairing `\langle \ell, x\rangle := \ell(x)`.
        3. Components: introduce the component vector `c_\ell` via a chosen computational basis/coordinate system (do not assume orthonormality). Explain the evaluation formula used in pygeoinf: `\ell(x)` is computed as a dot product between `c_\ell` and the coordinate vector `\Pi x` returned by `to_components`.
        4. User-facing construction: explain two equivalent ways in the API:
            - Supply `components` directly.
            - Supply a callable `mapping(x)`; pygeoinf computes the components by evaluating on `domain.basis_vector(i)` (with optional parallelism).
        5. Geometry: explain the Riesz representative (gradient) `g_\ell = \mathcal{R}(\ell)` and the identity `\ell(x) = (g_\ell, x)_H`. Map to code: `domain.from_dual(ell)`.
        6. Convenience: document arithmetic (`+,-,*,/`, in-place ops), `copy`, and the adapter `as_linear_operator` as a secondary interoperability tool.

2. **Phase 2: Build + consistency pass**
    - **Objective:** Ensure LaTeX compiles and notation is consistent with the earlier Riesz-first section.
    - **Files/Functions to Modify/Create:**
        - `pygeoinf/theory/TECHNICAL_MANUAL.tex`
    - **Tests to Write:** None
    - **Steps:**
        1. Rebuild the manual with two `pdflatex` passes.
        2. Fix any broken references introduced by section rename.

**Open Questions**
1. None — decisions captured: include one motivating example; defer operator pushforward/pullback identities; do not assume orthonormality.
