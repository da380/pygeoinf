## Phase 1 Complete: Preserve support under restrict and disjoint products

Phase 1 preserves `Function.support` metadata in two critical places: restriction now intersects (rather than discarding) compact support, and multiplication of disjointly-supported functions now returns an identically-zero function with `support=[]`. Targeted tests were added and the review passed.

**Files created/changed:**
- intervalinf/intervalinf/core/functions.py
- intervalinf/tests/core/test_functions.py
- intervalinf/plans/intervalinf-reference.md
- pygeoinf/plans/intervalinf-function-support-metadata-plan.md

**Functions created/changed:**
- Function.restrict
- Function.__mul__ (disjoint-support fast path)

**Tests created/changed:**
- TestFunctionArithmetic.test_mul_disjoint_support_gives_zero_with_empty_support
- TestFunctionRestrict.test_restrict_intersects_support_with_domain
- TestFunctionRestrict.test_restrict_can_yield_empty_support
- TestFunctionRestrict.test_restrict_preserves_no_compact_support

**Review Status:** APPROVED

**Git Commit Message:**
fix: preserve Function support metadata

- Intersect compact support in Function.restrict
- Return zero function with support=[] for disjoint products
- Add tests and update intervalinf reference notes

(Commit the plan file separately in pygeoinf: "docs: add intervalinf support-metadata plan".)
