## Plan Complete: Reintroduce Technical Manual Boxes

The LaTeX technical manual now balances intuition and rigor in the basis/coefficient/Gram discussion while restoring the missing “bridge to code” and worked-example boxes. The restored boxes match the current pygeoinf API conventions and preserve the decision that $\Pi$ is the primary coefficient route.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Notation-to-API alignment
2. ✅ Phase 2: Reintroduce missing boxes
3. ✅ Phase 3: Compile and polish

**All Files Created/Modified:**
- pygeoinf/theory/TECHNICAL_MANUAL.tex
- pygeoinf/plans/pygeoinf-reference.md
- pygeoinf/plans/technical-manual-boxes-reintro-plan.md
- pygeoinf/plans/technical-manual-boxes-reintro-phase-1-complete.md

**Key Functions/Classes Added:**
- N/A (documentation-only change)

**Test Coverage:**
- Total tests written: 0
- All tests passing: ✅ (LaTeX build)

**Recommendations for Next Steps:**
- If desired, reduce overfull hbox warnings by adding discretionary breaks inside long \texttt{...} identifiers.
- Proceed with a single scoped doc commit in the pygeoinf submodule, then update the top-level submodule pointer.
