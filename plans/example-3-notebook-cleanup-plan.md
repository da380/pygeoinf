## Plan: Clean example_3 notebook

Replace the current polluted `example_3.ipynb` with a clean copy of `discretization.ipynb`, then remove the naive-case strategy so the notebook contains only continuous vs discretized PLI.

**Phases**
1. **Phase 1: Replace Notebook Base**
    - **Objective:** Restore `example_3.ipynb` from `discretization.ipynb` so there is no unrelated old content.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/paper_demos/example_3.ipynb`
    - **Tests to Write:** None (notebook maintenance)
    - **Steps:**
        1. Copy `discretization.ipynb` over `example_3.ipynb`.
        2. Clear outputs/execution metadata.

2. **Phase 2: Remove Naive Strategy**
    - **Objective:** Remove the naive prior/PLI strategy and adapt markdown + plots to the 2-strategy case.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/paper_demos/example_3.ipynb`
    - **Tests to Write:** None
    - **Steps:**
        1. Delete naive PLI cell.
        2. Remove naive blocks from prior measure + prior visualization.
        3. Update comparison plot to show only continuous vs discrete.

3. **Phase 3: Validate Notebook JSON**
    - **Objective:** Ensure the notebook is valid nbformat and opens cleanly.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/paper_demos/example_3.ipynb`
    - **Tests to Write:** None
    - **Steps:**
        1. Validate with `nbformat`.
        2. (Optional) Execute via notebook runner if available.

**Open Questions**
1. None (confirmed: full replacement + clear outputs).
