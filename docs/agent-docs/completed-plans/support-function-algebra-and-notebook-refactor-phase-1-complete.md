## Phase 1 Complete: Core Constructors and Public Entry Points

Phase 1 added first-class support-function wrappers for user-defined support maps and singleton sets, together with convenience constructors on `SupportFunction`. The implementation is limited to the constructor layer, with targeted tests passing and review approval granted before any algebraic composition work.

**Files created/changed:**
- pygeoinf/pygeoinf/convex_analysis.py
- pygeoinf/tests/test_support_function_constructors.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md

**Functions created/changed:**
- SupportFunction.callable
- SupportFunction.point
- CallableSupportFunction
- PointSupportFunction

**Tests created/changed:**
- test_evaluate_matches_callable
- test_support_point_none_when_no_fn
- test_subgradient_raises_when_no_fn
- test_support_point_delegates_to_callable
- test_subgradient_uses_support_point_callable
- test_evaluate_inner_product
- test_support_point_is_fixed_point
- test_subgradient_is_fixed_point
- test_callable_returns_callable_support_function
- test_callable_with_support_point
- test_point_returns_point_support_function
- test_callable_subgradient_matches_support_point

**Review Status:** APPROVED

**Git Commit Message:**
feat(support): add support-function constructors

- Add CallableSupportFunction and PointSupportFunction
- Expose SupportFunction.callable and .point helpers
- Add Phase 1 constructor tests and update reference

Plan: pygeoinf/docs/agent-docs/active-plans/support-function-algebra-and-notebook-refactor-plan.md
Phase: 1 of 4
Related: pygeoinf/docs/agent-docs/completed-plans/support-function-algebra-and-notebook-refactor-phase-1-complete.md