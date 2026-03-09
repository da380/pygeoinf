## Phase 3 Complete: Support-Point Propagation and Edge Cases

Phase 3 implemented support-point propagation through all algebraic wrappers with conservative semantics. The algebra is now fully usable for higher-level constructions where knowing the support points (subgradients) matters for convergence analysis or visualization. Edge cases including zero scaling and unavailable operands are handled safely.

**Files created/changed:**
- pygeoinf/pygeoinf/convex_analysis.py
- pygeoinf/tests/test_support_function_algebra.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md

**Functions created/changed:**
- LinearImageSupportFunction.support_point
- MinkowskiSumSupportFunction.support_point
- ScaledSupportFunction.support_point

**Tests created/changed:**
- test_linear_image_support_point_propagates
- test_linear_image_support_point_unavailable
- test_minkowski_support_point_both_available
- test_minkowski_support_point_left_unavailable
- test_minkowski_support_point_right_unavailable
- test_scaled_support_point_positive_alpha
- test_scaled_support_point_zero_alpha
- test_scaled_support_point_unavailable
- test_nested_composition_support_points
- test_support_point_edge_cases (4 variants)

**Review Status:** APPROVED

**Git Commit Message:**
feat(support): add support-point propagation

- Propagate support points through linear image, sum, scale
- Conservative None fallback when operands unavailable
- Add Phase 3 edge-case tests and update reference

Plan: pygeoinf/docs/agent-docs/active-plans/support-function-algebra-and-notebook-refactor-plan.md
Phase: 3 of 4
Related: pygeoinf/docs/agent-docs/completed-plans/support-function-algebra-and-notebook-refactor-phase-3-complete.md
