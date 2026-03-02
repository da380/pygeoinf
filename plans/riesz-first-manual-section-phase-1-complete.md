## Phase 1 Complete: Insert Riesz-first design section + rebuild

Added an explicit “Riesz-first” explanation immediately after the Riesz representation theorem, clarifying that `to_dual`/`from_dual` are the primitive geometry in pygeoinf and that inner products and (default) adjoints are derived from them. Updated the package reference to match the actual `HilbertSpace` contract and rebuilt the technical manual successfully.

**Files created/changed:**
- `pygeoinf/theory/TECHNICAL_MANUAL.tex`
- `pygeoinf/theory/TECHNICAL_MANUAL.pdf`
- `pygeoinf/theory/TECHNICAL_MANUAL.aux`
- `pygeoinf/theory/TECHNICAL_MANUAL.log`
- `pygeoinf/theory/TECHNICAL_MANUAL.out`
- `pygeoinf/theory/TECHNICAL_MANUAL.toc`
- `pygeoinf/plans/pygeoinf-reference.md`
- `pygeoinf/plans/riesz-first-manual-section-plan.md`

**Functions created/changed:**
- None (documentation-only change)

**Tests created/changed:**
- None

**Review Status:** APPROVED

**Git Commit Message:**
docs: document Riesz-first geometry

- Add Riesz-first section after Riesz theorem
- Qualify adjoint dependence on Riesz maps
- Align package reference with HilbertSpace API
