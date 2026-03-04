## Phase 2 Complete: Dual vs Adjoint (Riesz-first derivation)

Completed the Banach-dual and Hilbert-adjoint sections for linear operators, with explicit typing and a Riesz-first derivation of the identity $G^*=\mathcal R_{\mathcal M}\circ G'\circ \mathcal R_{\mathcal D}^{-1}$ consistent with the library’s `to_dual`/`from_dual` design.

**Files created/changed:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/theory/TECHNICAL_MANUAL.aux
- pygeoinf/theory/TECHNICAL_MANUAL.log
- pygeoinf/theory/TECHNICAL_MANUAL.out
- pygeoinf/theory/TECHNICAL_MANUAL.toc
- pygeoinf/theory/TECHNICAL_MANUAL.pdf
- pygeoinf/plans/linear-operators-chapter-phase-2-complete.md

**Functions created/changed:**
- None

**Tests created/changed:**
- LaTeX manual build check (2 `pdflatex` passes)

**Review Status:** APPROVED

**Git Commit Message:**
docs: define dual and adjoint operators

- Add Banach-dual and Hilbert-adjoint definitions
- Derive adjoint via Riesz maps (Riesz-first)
- Rebuild technical manual artifacts
