## Phase 2 Complete: Implement Plotly 3D Backend With Fallback

Added an interactive 3D plotting path that returns Plotly figures for sampled and exact polyhedral 3D subset slices, while preserving Matplotlib for explicit `backend="matplotlib"` and for automatic fallback when Plotly is unavailable. The phase also adds the optional dependency wiring, locks in the fallback and trace-type behavior with tests, and updates the package reference and public docstrings to match the new runtime semantics.

**Files created/changed:**
- `pygeoinf/pyproject.toml`
- `pygeoinf/pygeoinf/plot.py`
- `pygeoinf/tests/test_plot.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- `SubspaceSlicePlotter._resolve_backend`
- `SubspaceSlicePlotter._render_3d_plotly`
- `SubspaceSlicePlotter._render_3d_polyhedral_plotly`
- `SubspaceSlicePlotter.plot`
- `SubspaceSlicePlotter._plot_polyhedral_exact`
- `plot_slice`

**Tests created/changed:**
- `test_plot_slice_3d_auto_uses_plotly_when_available`
- `test_plot_slice_3d_plotly_missing_dependency`
- `test_plot_slice_3d_auto_falls_back_to_matplotlib`
- `test_plot_slice_3d_plotly_returns_figure`
- `test_plot_slice_polyhedral_3d_plotly_exact_path`
- `test_plot_slice_plotly_rejects_matplotlib_ax`
- Existing Matplotlib-only 3D tests pinned to `backend="matplotlib"`

**Review Status:** APPROVED

**Git Commit Message:**
feat(visualization): add Plotly 3D backend with Matplotlib fallback

- Add optional `plotly` interactive extra and route 3D plotting through backend resolution
- Render sampled 3D slices with `go.Isosurface` and exact polyhedral 3D slices with `go.Mesh3d`
- Add fallback, return-type, and trace-type tests and sync plotting reference docs

Plan: docs/agent-docs/active-plans/interactive-3d-visualization-plan.md
Phase: 2 of 3
Related: docs/agent-docs/completed-plans/interactive-3d-visualization-phase-2-complete.md