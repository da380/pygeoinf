## Plan: Remove HyperEllipsoid And Audit Redundancy

Remove the unused `HyperEllipsoid` class from `backus_gilbert.py`, update the public exports and reference docs that still mention it, and then perform a package-level audit for other high-confidence redundant or superseded code in `pygeoinf`. The goal is to make the removal safe, keep the package reference accurate, and separate confirmed cleanup from lower-confidence candidates.

**Phases 3**
1. **Phase 1: Remove HyperEllipsoid**
    - **Objective:** Delete the unused `HyperEllipsoid` class and clean up all direct package references to it.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/backus_gilbert.py`, `pygeoinf/pygeoinf/__init__.py`, `pygeoinf/docs/theory_papers_index.md`, `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
    - **Tests to Write:** None if no behavioral surface remains; verify by search and existing test runs.
    - **Steps:**
        1. Confirm `HyperEllipsoid` has no runtime users or tests.
        2. Remove the class and top-level export.
        3. Update package docs and reference material to point to `subsets.Ellipsoid` and `DualMasterCostFunction`.
        4. Run focused validation to confirm imports and tests still pass.
2. **Phase 2: Deep Redundancy Audit**
    - **Objective:** Inspect `pygeoinf` for other high-confidence redundant, dead, or superseded code.
    - **Files/Functions to Modify/Create:** Read-only audit across `pygeoinf/pygeoinf/`, tests, and docs.
    - **Tests to Write:** None.
    - **Steps:**
        1. Search for unreferenced exports, isolated modules, and legacy directories.
        2. Cross-check candidate findings against tests and public exports.
        3. Separate confirmed dead code from code that is merely optional, experimental, or weakly integrated.
3. **Phase 3: Report And Recommendations**
    - **Objective:** Summarize confirmed removals, validation results, and high-confidence follow-up cleanup candidates.
    - **Files/Functions to Modify/Create:** `pygeoinf/docs/agent-docs/completed-plans/hyperellipsoid-removal-and-redundancy-audit-phase-1-complete.md`
    - **Tests to Write:** None.
    - **Steps:**
        1. Record the implemented removal and validation evidence.
        2. Document audit findings with confidence and rationale.
        3. Provide a commit message template and next-step recommendations.

**Open Questions 1**
1. Should experimental directories such as `data_assimilation/` and `testing_sets/` remain inside the installable source tree, or should they move to top-level examples/research folders?