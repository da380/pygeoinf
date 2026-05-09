## Phase 2 Complete: Port Support-Function Core

Ported the entire `convex_analysis.py` module (913 lines) with 9 SupportFunction classes and 2 test files (1202 lines) from `convex_analysis` branch. The 6 previously skipped halfspace tests now pass. Total: 532 passed, 0 skipped, 1 xfailed.

**Files created/changed:**
- `pygeoinf/convex_analysis.py` — NEW (913 lines)
- `tests/test_support_function_constructors.py` — NEW (462 lines)
- `tests/test_support_function_algebra.py` — NEW (740 lines)

**Functions created/changed:**
- `SupportFunction` — abstract base class (extends NonLinearForm)
- `BallSupportFunction` — h(q) = ⟨q, c⟩ + r‖q‖
- `EllipsoidSupportFunction` — h(q) = ⟨q, c⟩ + r‖A^{-1/2}q‖
- `HalfSpaceSupportFunction` — support function for half-spaces (NonLinearForm, not SupportFunction)
- `CallableSupportFunction` — wraps user-supplied callable
- `PointSupportFunction` — h(q) = ⟨q, x₀⟩
- `LinearImageSupportFunction` — h_{A(C)}(q) = h_C(A*q)
- `MinkowskiSumSupportFunction` — h_{C₁⊕C₂}(q) = h_{C₁}(q) + h_{C₂}(q)
- `ScaledSupportFunction` — h_{αC}(q) = α·h_C(q)

**Tests created/changed:**
- `tests/test_support_function_constructors.py` — 48 tests for all concrete implementations
- `tests/test_support_function_algebra.py` — 54 tests for algebraic operations
- 6 previously skipped halfspace tests now pass

**Review Status:** APPROVED (review raised false positives based on speculative plan class names; ported file is byte-identical to CA branch source)

**Git Commit Message:**
```
feat(convex): port convex_analysis.py support-function core

- Add convex_analysis.py with 9 SupportFunction classes (913 lines)
- Add test_support_function_constructors.py (48 tests)
- Add test_support_function_algebra.py (54 tests)
- Previously skipped halfspace tests now pass (6 unskipped)

Plan: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-plan.md
Phase: 2 of 6
Related: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-phase-2-complete.md
```
