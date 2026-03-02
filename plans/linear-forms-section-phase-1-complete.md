## Phase 1 Complete: Linear Forms (components + geometry)

Rewrote the “Linear Forms” chapter to focus on component representations and how linear forms interact with the domain’s coordinate maps and Riesz-induced geometry. The old “rank-1 operator” framing was removed; `as_linear_operator` is presented only as an interoperability adapter. The technical manual builds successfully.

**Files created/changed:**
- `pygeoinf/theory/TECHNICAL_MANUAL.tex`
- `pygeoinf/theory/TECHNICAL_MANUAL.pdf`
- `pygeoinf/theory/TECHNICAL_MANUAL.aux`
- `pygeoinf/theory/TECHNICAL_MANUAL.log`
- `pygeoinf/theory/TECHNICAL_MANUAL.out`
- `pygeoinf/theory/TECHNICAL_MANUAL.toc`
- `pygeoinf/plans/linear-forms-section-plan.md`

**Functions created/changed:**
- None (documentation-only change)

**Tests created/changed:**
- None

**Review Status:** APPROVED

**Git Commit Message:**
docs: rewrite linear forms chapter

- Explain linear forms via components and dual pairing
- Document mapping vs components construction paths
- Reframe as_linear_operator as an adapter
