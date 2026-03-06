## Plan: Add Interactive 3D Rendering

Add an optional interactive 3D backend for subset visualization, using Plotly isosurfaces/meshes when available and falling back to Matplotlib when it is not. The change will preserve existing exact polyhedral behavior, keep backend-dependent return types explicit, and document the fallback path clearly.

**Phases 3**
1. **Phase 1: Add Backend Selection API**
   - **Objective:** Thread a `backend` option through the public plotting surface.
   - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/plot.py`; `pygeoinf/pygeoinf/subsets.py`; `pygeoinf/tests/test_plot.py`
   - **Tests to Write:** `test_plot_slice_backend_parameter_defaults`; `test_subset_plot_forwards_backend`; `test_plot_slice_3d_auto_uses_existing_matplotlib_path_when_needed`
   - **Steps:**
       1. Write failing tests that define `backend="auto" | "matplotlib" | "plotly"` on `plot_slice()`, `SubspaceSlicePlotter.plot()`, and `Subset.plot()`.
       2. Implement the new parameter and delegation plumbing without changing the current Matplotlib rendering behavior.
       3. Run the targeted plotting tests to confirm the backend parameter is wired correctly.

2. **Phase 2: Implement Plotly 3D Backend With Fallback**
   - **Objective:** Render 3D subsets interactively with Plotly isosurfaces/meshes when requested or auto-selected, with Matplotlib fallback when Plotly is unavailable.
   - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/plot.py`; `pygeoinf/pyproject.toml`; `pygeoinf/tests/test_plot.py`
   - **Tests to Write:** `test_plot_slice_3d_plotly_missing_dependency`; `test_plot_slice_3d_auto_falls_back_to_matplotlib`; `test_plot_slice_3d_plotly_returns_figure`; `test_plot_slice_polyhedral_3d_plotly_exact_path`; `test_plot_slice_plotly_rejects_matplotlib_ax`
   - **Steps:**
       1. Add Plotly as an optional dependency and expose a matching extra in `pyproject.toml`.
       2. Implement Plotly rendering for sampled 3D masks via isosurface traces and exact polyhedral 3D rendering via mesh traces.
       3. Add import/fallback/error handling so `backend="auto"` tries Plotly first and falls back to Matplotlib with a warning if Plotly is unavailable.
       4. Run targeted plotting tests to confirm both requested and fallback paths behave correctly.

3. **Phase 3: Sync Docs and Examples**
   - **Objective:** Document the interactive backend option and update the demo notebook/reference docs.
   - **Files/Functions to Modify/Create:** `pygeoinf/README.md`; `pygeoinf/pygeoinf/testing_sets/visualization_demo.ipynb`; `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
   - **Tests to Write:** No new automated tests unless a small docs smoke test becomes necessary.
   - **Steps:**
       1. Update README examples to show `backend="auto"` for 3D plotting and document Plotly fallback semantics.
       2. Extend the visualization demo notebook with one interactive 3D example.
       3. Update the living reference so backend options and return-type semantics are explicit for Plotly vs Matplotlib.

**Open Questions 2**
1. Default backend for 3D is approved as `backend="auto"`; 1D/2D should remain Matplotlib-only unless explicitly broadened later.
2. Plotly sampled rendering should use an interactive isosurface trace rather than attempting to mimic voxels.