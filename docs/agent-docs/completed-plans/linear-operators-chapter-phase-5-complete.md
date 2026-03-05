## Phase 5 Complete: Constructing operators in pygeoinf

Expanded the “Constructing operators in pygeoinf” section into implementer-facing guidance: what `mapping`, `dual_mapping`, and `adjoint_mapping` mean; when the library derives one from the other via Riesz maps; and how mass-weighted geometries connect “formal adjoints” to Hilbert adjoints via `MassWeightedHilbertSpace` and `LinearOperator.from_formal_adjoint`. Also highlighted safe factory routes.

**Files created/changed:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/theory/TECHNICAL_MANUAL.aux
- pygeoinf/theory/TECHNICAL_MANUAL.log
- pygeoinf/theory/TECHNICAL_MANUAL.out
- pygeoinf/theory/TECHNICAL_MANUAL.toc
- pygeoinf/theory/TECHNICAL_MANUAL.pdf
- pygeoinf/plans/linear-operators-chapter-phase-5-complete.md

**Functions created/changed:**
- None

**Tests created/changed:**
- LaTeX manual build check (2 `pdflatex` passes)

**Review Status:** APPROVED

**Git Commit Message:**
docs: document operator construction patterns

- Explain mapping vs dual_mapping vs adjoint_mapping
- Describe derived dual/adjoint via Riesz maps
- Connect formal adjoints to mass-weighted geometry
