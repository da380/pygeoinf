## Phase 7 Complete: Full Commutative Diagram

Added the full commutative TikZ diagram connecting the infinite-dimensional spaces and their finite-N surrogates, based on the user's provided starting point. Fixed three bugs identified in review: removed the incorrect `blue` from the forward operator $G$ (not geometry-dependent), added `blue` to the Hilbert adjoint $\Gadj$ (which is geometry-dependent), and corrected the $\mathcal{R}_{\modelspace_N}$ arrow direction (moved from the dual node upward to the primal node downward). An explanatory paragraph frames the key identity $\Gadj = \RieszM \circ G' \circ \RieszD^{-1}$ and the role of blue arrows.

**Files created/changed:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/theory/TECHNICAL_MANUAL.aux
- pygeoinf/theory/TECHNICAL_MANUAL.log
- pygeoinf/theory/TECHNICAL_MANUAL.out
- pygeoinf/theory/TECHNICAL_MANUAL.toc
- pygeoinf/theory/TECHNICAL_MANUAL.pdf
- pygeoinf/plans/linear-operators-chapter-phase-7-complete.md

**Functions created/changed:**
- None

**Tests created/changed:**
- LaTeX manual build check (2 `pdflatex` passes, 0 hard errors, 670K PDF)

**Review Status:** APPROVED (after three diagram fixes applied)

**Git Commit Message:**
docs: add full commutative diagram for operators

- Insert tikzcd diagram linking M, D, duals, and surrogates
- Fix blue-arrow coloring to correctly mark geometry-dependent maps
- Fix Riesz map arrow directions in diagram
