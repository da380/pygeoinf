## Phase 4 Complete: Declarative Notebook Refactor

Phase 4 refactored the reduced DLI notebook so the admissible-region support is expressed directly with the new support-function algebra rather than a handwritten oracle. The notebook now composes the estimator translation, model-prior image, and data-confidence image declaratively, and the affected cells execute successfully with the same admissible-region behavior.

**Files created/changed:**
- intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb

**Functions created/changed:**
- Notebook support construction using `SupportFunction.point`
- Notebook support construction using `SupportFunction.image`
- Notebook halfspace construction using the composed `admissible_support`

**Tests created/changed:**
- Re-executed notebook Phase 2 support-construction cell
- Re-executed notebook polyhedral-region cell
- Re-executed notebook plot cell
- Verified cardinal containment with `P.inner_product`
- Verified `p_t` and `p_bar` remain in the admissible region

**Review Status:** APPROVED

**Git Commit Message:**
feat(notebook): refactor admissible-region support algebra

- Replace handwritten support oracle with composed supports
- Use SupportFunction.point and .image in the DLI notebook
- Re-run admissible-region cells and preserve plot output

Plan: pygeoinf/docs/agent-docs/active-plans/support-function-algebra-and-notebook-refactor-plan.md
Phase: 4 of 4
Related: pygeoinf/docs/agent-docs/completed-plans/support-function-algebra-and-notebook-refactor-phase-4-complete.md
