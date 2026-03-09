## Phase 2 Complete: Algebraic Composition and Scaling

Phase 2 added reusable support-function algebra for linear images, Minkowski sums, translation, and nonnegative scaling, while keeping the results inside the `SupportFunction` hierarchy rather than degrading to generic nonlinear forms. The implementation includes explicit wrapper classes, convenience methods and operators, targeted regression tests, and the zero-scaling fix for extended-real-valued supports.

**Files created/changed:**
- pygeoinf/pygeoinf/convex_analysis.py
- pygeoinf/pygeoinf/__init__.py
- pygeoinf/tests/test_support_function_algebra.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md

**Functions created/changed:**
- SupportFunction.image
- SupportFunction.translate
- SupportFunction.scale
- SupportFunction.__add__
- SupportFunction.__mul__
- SupportFunction.__rmul__
- LinearImageSupportFunction
- MinkowskiSumSupportFunction
- ScaledSupportFunction

**Tests created/changed:**
- test_linear_image_matches_adjoint_pullback
- test_image_requires_matching_operator_domain
- test_translate_matches_point_sum_identity
- test_minkowski_sum_matches_value_sum
- test_add_returns_support_function_wrapper
- test_scale_matches_scalar_identity
- test_zero_scaling_returns_zero_support
- test_zero_scaling_on_unbounded_support_is_zero
- test_negative_scaling_raises_value_error
- test_mul_returns_support_function_wrapper

**Review Status:** APPROVED

**Git Commit Message:**
feat(support): add support-function algebra

- Add image, sum, translate, and scale support wrappers
- Override SupportFunction arithmetic to preserve support types
- Add Phase 2 algebra tests and update reference docs

Plan: pygeoinf/docs/agent-docs/active-plans/support-function-algebra-and-notebook-refactor-plan.md
Phase: 2 of 4
Related: pygeoinf/docs/agent-docs/completed-plans/support-function-algebra-and-notebook-refactor-phase-2-complete.md