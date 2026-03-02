## Phase 6 Complete: Advanced Routines (randomized surrogates)

Added a concise section documenting `LinearOperator.random_svd`, `random_eig`, and `random_cholesky`: what they return at a high level, when to use them (diagnostics, cheap surrogates, preconditioning intuition), and a clear caution that results are randomized/approximate and depend on the chosen geometry (component vs Galerkin views) and operator properties.

**Files created/changed:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/theory/TECHNICAL_MANUAL.aux
- pygeoinf/theory/TECHNICAL_MANUAL.log
- pygeoinf/theory/TECHNICAL_MANUAL.out
- pygeoinf/theory/TECHNICAL_MANUAL.toc
- pygeoinf/theory/TECHNICAL_MANUAL.pdf
- pygeoinf/plans/linear-operators-chapter-phase-6-complete.md

**Functions created/changed:**
- None

**Tests created/changed:**
- LaTeX manual build check (2 `pdflatex` passes)

**Review Status:** APPROVED

**Git Commit Message:**
docs: document randomized operator routines

- Describe random_svd/random_eig/random_cholesky outputs
- Add geometry-dependent caution and use cases
- Rebuild technical manual artifacts
