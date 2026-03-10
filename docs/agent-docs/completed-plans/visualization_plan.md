## Plan: Repair and Extend Visualization

Rebase Phase 8 on the current package state before adding new API surface. The first objective is to restore importability and make the existing plotting module testable, then layer convenience APIs, 3D support, and documentation on top of a verified baseline.

**Phases 5**
1. **Phase 1: Restore Plot Module Baseline**
  - **Objective:** Make `pygeoinf.plot` importable again and establish a truthful visualization test baseline for the code that already exists.
  - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/plot.py`; `pygeoinf/pygeoinf/__init__.py`; `pygeoinf/tests/test_plot.py`
  - **Tests to Write:** `test_visualization_module_imports`; `test_subspace_slice_plotter_imports`; `test_subspace_slice_plotter_ball_2d`; `test_subspace_slice_plotter_polyhedral_2d`
  - **Steps:**
    1. Write or adjust import and smoke tests to capture the current failure during test collection.
    2. Fix the runtime import error in `pygeoinf.plot` with the minimal code changes needed.
    3. Run the targeted visualization tests again and confirm the module imports and current 1D/2D paths pass.

2. **Phase 2: Add plot_slice Convenience API**
  - **Objective:** Add a supported top-level `plot_slice()` wrapper around `SubspaceSlicePlotter` for currently implemented plotting paths.
  - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/plot.py`; `pygeoinf/pygeoinf/__init__.py`; `pygeoinf/tests/test_plot.py`
  - **Tests to Write:** `test_plot_slice_exported`; `test_plot_slice_ball_2d_returns_figure`; `test_plot_slice_polyhedral_2d_returns_vertices`
  - **Steps:**
    1. Write failing tests for a public `plot_slice()` wrapper using supported 1D and 2D cases.
    2. Implement the wrapper and export it from the package API.
    3. Re-run the targeted visualization tests and confirm the convenience path matches direct plotter usage.

3. **Phase 3: Add Subset.plot Entry Point**
  - **Objective:** Add a user-facing plotting method on `Subset` that delegates to `plot_slice()` without inventing a separate visualization module.
  - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/subsets.py`; `pygeoinf/pygeoinf/plot.py`; `pygeoinf/tests/test_subsets.py`; `pygeoinf/tests/test_plot.py`
  - **Tests to Write:** `test_subset_plot_requires_subspace_or_builds_default`; `test_ball_plot_delegates_to_plot_slice`; `test_subset_plot_preserves_kwargs`
  - **Steps:**
    1. Write tests that define the exact public API for `Subset.plot()` in the current package architecture.
    2. Implement `Subset.plot()` by delegating to `pygeoinf.plot.plot_slice`.
    3. Run the subset and visualization tests together to confirm the API is stable and consistent.

4. **Phase 4: Implement 3D Visualization Backend**
  - **Objective:** Replace the current `NotImplementedError` 3D path with a supported backend and explicit dependency handling.
  - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/plot.py`; `pygeoinf/pyproject.toml`; `pygeoinf/tests/test_plot.py`
  - **Tests to Write:** `test_plot_slice_ball_3d_backend`; `test_plot_slice_3d_missing_optional_dependency`; `test_plot_slice_3d_returns_figure_like`
  - **Steps:**
    1. Write failing tests that describe the expected 3D behavior and fallback semantics.
    2. Implement the 3D backend and optional dependency handling in `plot.py`.
    3. Run targeted visualization tests to confirm 3D plotting now works or fails cleanly when the optional backend is unavailable.

5. **Phase 5: Demo and Documentation Sync**
  - **Objective:** Document the final visualization API, add a runnable demo, and update living references so future agent work starts from accurate docs.
  - **Files/Functions to Modify/Create:** `pygeoinf/README.md`; `pygeoinf/pygeoinf/testing_sets/visualization_demo.ipynb`; `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
  - **Tests to Write:** `test_readme_visualization_examples_smoke` if documentation examples become executable; otherwise no new automated tests required.
  - **Steps:**
    1. Add user-facing quick-start documentation for the final visualization API.
    2. Create the end-to-end demo notebook covering supported set types and export paths.
    3. Update the living reference so it accurately reflects the final plotting module, public API, and 3D support status.

**Open Questions 3**
1. Should `Subset.plot()` require `on_subspace` initially, or should it auto-build a default subspace only for finite-dimensional `EuclideanSpace` domains?
2. For Phase 4, should the first supported 3D backend be `matplotlib` voxels/surfaces or optional `plotly` with graceful fallback?
3. Should Phase 2 restrict `plot_slice()` to the currently implemented 1D/2D behavior until Phase 4 lands, or should it expose 3D immediately with a documented temporary error?
