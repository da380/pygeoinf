## Plan Complete: Support-Function Algebra and Notebook Refactor

This plan completed the support-function algebra work by adding constructor helpers, algebraic composition, support-point propagation, and a declarative notebook refactor for the reduced DLI example. The result is a reusable support-function API in `pygeoinf` and a notebook that expresses the admissible-region construction directly through composed support objects rather than handwritten pullback logic.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Core Constructors and Public Entry Points
2. ✅ Phase 2: Algebraic Composition and Scaling
3. ✅ Phase 3: Support-Point Propagation and Edge Cases
4. ✅ Phase 4: Declarative Notebook Refactor

**All Files Created/Modified:**
- pygeoinf/pygeoinf/convex_analysis.py
- pygeoinf/pygeoinf/__init__.py
- pygeoinf/tests/test_support_function_constructors.py
- pygeoinf/tests/test_support_function_algebra.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md
- intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb

**Key Functions/Classes Added:**
- SupportFunction.callable
- SupportFunction.point
- SupportFunction.image
- SupportFunction.translate
- SupportFunction.scale
- SupportFunction.__add__
- SupportFunction.__mul__
- SupportFunction.__rmul__
- CallableSupportFunction
- PointSupportFunction
- LinearImageSupportFunction
- MinkowskiSumSupportFunction
- ScaledSupportFunction

**Test Coverage:**
- Constructor tests added
- Algebra tests added
- Support-point propagation tests added
- Notebook cells re-executed successfully: ✅

**Recommendations for Next Steps:**
- Keep notebook comparisons explicit about confidence calibration when varying data dimension
- Consider additional examples that use the support-function algebra outside the DLI notebook
