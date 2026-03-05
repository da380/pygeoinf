## Phase 3 Complete: Matrices and Coordinates (standard vs Galerkin)

Implemented the matrix-representation section for linear operators, documenting the two `LinearOperator.matrix(..., galerkin=...)` modes and clarifying how `rmatvec` corresponds to the Banach dual vs Hilbert adjoint. Added explicit warnings that coordinate maps are not isometries and that symmetry/self-adjointness lives in the Galerkin representation.

**Files created/changed:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/theory/TECHNICAL_MANUAL.aux
- pygeoinf/theory/TECHNICAL_MANUAL.log
- pygeoinf/theory/TECHNICAL_MANUAL.out
- pygeoinf/theory/TECHNICAL_MANUAL.toc
- pygeoinf/theory/TECHNICAL_MANUAL.pdf
- pygeoinf/plans/linear-operators-chapter-phase-3-complete.md

**Functions created/changed:**
- None

**Tests created/changed:**
- LaTeX manual build check (2 `pdflatex` passes)

**Review Status:** APPROVED

**Git Commit Message:**
docs: document matrix representations for operators

- Explain standard vs Galerkin matrix views
- Clarify rmatvec corresponds to dual vs adjoint
- Add pitfalls about coordinates vs geometry
