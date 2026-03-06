## Phase 1 Complete: Add Backend Selection API

Added a `backend` parameter across the public 3D plotting surface without changing current rendering behavior. The new parameter now flows through `Subset.plot()`, `plot_slice()`, and `SubspaceSlicePlotter.plot()`, defaults to `"auto"`, and is documented as a no-op selector until the interactive backend lands in Phase 2.

**Files created/changed:**
- `pygeoinf/pygeoinf/plot.py`
- `pygeoinf/pygeoinf/subsets.py`
- `pygeoinf/tests/test_plot.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- `Subset.plot`
- `SubspaceSlicePlotter.plot`
- `plot_slice`

**Tests created/changed:**
- `test_plot_slice_backend_parameter_defaults`
- `test_plot_slice_backend_valid_values`
- `test_subspace_slice_plotter_plot_backend_parameter`
- `test_subset_plot_backend_parameter_exists`
- `test_subset_plot_forwards_backend`
- `test_plot_slice_3d_auto_uses_existing_matplotlib_path`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(visualization): add backend selector to 3D plotting API

- Add `backend="auto"` to `Subset.plot()`, `plot_slice()`, and `SubspaceSlicePlotter.plot()`
- Preserve existing Matplotlib rendering behavior while wiring the new selector through the public API
- Add backend-parameter and delegation tests and sync the living reference docs

Plan: docs/agent-docs/active-plans/interactive-3d-visualization-plan.md
Phase: 1 of 3
Related: docs/agent-docs/completed-plans/interactive-3d-visualization-phase-1-complete.md