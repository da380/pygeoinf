## Plan Complete: Linear Forms section (components + geometry)

Completely rewrote the "Linear Forms" chapter to focus on component representations and how linear forms interact with the domain's coordinate maps and Riesz-induced geometry. The old "rank-1 operator" framing was removed; `as_linear_operator` is presented only as an interoperability adapter. Both phases completed in a single focused effort; technical manual builds successfully with no errors.

**Phases Completed:** 2 of 2
1. ✅ Phase 1: Rewrite the Linear Forms chapter content
2. ✅ Phase 2: Build + consistency pass

**All Files Created/Modified:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex (171 lines changed)
- pygeoinf/theory/TECHNICAL_MANUAL.pdf (rebuilt)

**Key Changes:**
- Motivation: evaluation as basis for data terms
- Definition: continuous linear functionals in dual space
- Components: component vector via coordinate system (non-orthonormal basis support)
- User-facing construction: two paths (components direct or computed from callable mapping)
- Geometry: Riesz representative and mapping to `domain.from_dual`
- Convenience: arithmetic and adapters

**Test Coverage:**
- Total tests written: 0 (documentation-only)
- Validation: LaTeX compiles cleanly, no broken references

**Recommendations for Next Steps:**
- None (plan is complete)
