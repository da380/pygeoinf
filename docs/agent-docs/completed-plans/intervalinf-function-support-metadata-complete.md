## Plan Complete: Function Support Metadata Preservation

Four phases of TDD implementation ensuring `Function.support` metadata is correctly set and propagated throughout the intervalinf package. The `support` field (semantics: `None` = unknown/global, `[]` = known empty/zero, `[(a,b),...]` = compact intervals) is now set by all basis providers and preserved by all arithmetic operations in Lebesgue spaces.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Function.restrict support intersection + disjoint-product `support=[]`
2. ✅ Phase 2: Lebesgue algebra support propagation (`from_components`, `zero`, `multiply`, `add`, `ax`, `axpy`)
3. ✅ Phase 3: Hat/Spline providers set support on all returned basis functions
4. ✅ Phase 4: Reference documentation updated

**All Files Created/Modified:**
- `intervalinf/intervalinf/core/functions.py` (Phase 1)
- `intervalinf/intervalinf/spaces/lebesgue.py` (Phase 2)
- `intervalinf/intervalinf/providers/functions/fem.py` (Phase 3)
- `intervalinf/tests/core/test_functions.py` (Phase 1)
- `intervalinf/tests/spaces/test_lebesgue.py` (Phase 2)
- `intervalinf/tests/providers/test_standalone_providers.py` (Phase 3)
- `intervalinf/plans/intervalinf-reference.md` (Phase 4)
- `pygeoinf/plans/intervalinf-function-support-metadata-phase-1-complete.md`
- `pygeoinf/plans/intervalinf-function-support-metadata-phase-3-complete.md`
- `pygeoinf/plans/intervalinf-function-support-metadata-complete.md`

**Key Functions/Classes Added or Modified:**
- `Function.restrict`: intersects `.support` with restricted domain's interval
- `Function.__mul__`: short-circuits to `support=[]` when supports are disjoint and both compact
- `Function._intersect_supports`, `Function._union_supports`: static helpers for support algebra
- `Lebesgue.from_components`: infers support from active basis functions (tol = 1e-14)
- `Lebesgue.zero`: returns `support=[]`
- `Lebesgue.multiply`: `a==0 → support=[]`, else preserves
- `Lebesgue.add`: union of operand supports
- `Lebesgue.ax`: in-place; sets `support=[]` when `a == 0`
- `Lebesgue.axpy`: in-place; unions support only when `a != 0`
- `HatFunctionProvider.get_function_by_index`: sets `support=(nodes[i-1], nodes[i+1])`, clamped at boundaries
- `SplineFunctionProvider.get_function_by_index`: sets `support=(knots[i], knots[i+degree+1])`; `None` for degenerate spans
- `SplineFunctionProvider.get_function_by_parameters`: infers support when `knots`/`degree`/`index` keys present

**Test Coverage:**
- Total tests written across all phases: ~29 new tests
- All tests passing: ✅ (302 passed, 0 failed, 0 regressions as of Phase 3)

**Recommendations for Next Steps:**
- `BumpFunctionProvider` and `WaveletFunctionProvider` could also set `support=` (their functions are compactly supported by construction) following the same pattern
- `DiscontinuousFunctionProvider` (piecewise constant) could set support per sub-interval similarly to `BoxCarFunctionProvider`
- Consider adding a `support_width` convenience property to `Function` for interval-length arithmetic
