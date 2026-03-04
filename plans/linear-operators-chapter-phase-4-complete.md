## Phase 4 Complete: Simple Worked Example (from_matrix end-to-end)

Added a compact worked example constructed via `LinearOperator.from_matrix` that demonstrates the type distinctions between the forward operator, its Banach dual, and its Hilbert adjoint, including a mass-weighted codomain case where dual and adjoint differ. The example also shows how to obtain dense matrices via `matrix(dense=True, galerkin=...)` and how to interpret the two `galerkin` modes.

**Files created/changed:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/theory/TECHNICAL_MANUAL.aux
- pygeoinf/theory/TECHNICAL_MANUAL.log
- pygeoinf/theory/TECHNICAL_MANUAL.out
- pygeoinf/theory/TECHNICAL_MANUAL.toc
- pygeoinf/theory/TECHNICAL_MANUAL.pdf
- pygeoinf/plans/linear-operators-chapter-phase-4-complete.md

**Functions created/changed:**
- None

**Tests created/changed:**
- LaTeX manual build check (2 `pdflatex` passes)

**Review Status:** APPROVED (with minor wording clarification applied)

**Git Commit Message:**
docs: add from_matrix worked example

- Add mass-weighted example distinguishing dual vs adjoint
- Clarify dense=True for materialized matrices
- Rebuild technical manual artifacts
