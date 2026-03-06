## Phase 2 Complete: Add plot_slice Convenience API

Phase 2 added a public `plot_slice()` wrapper over `SubspaceSlicePlotter` and exported it from both `pygeoinf.plot` and top-level `pygeoinf`. The phase kept the public API honest by supporting 1D and 2D slices only, explicitly rejecting 3D through the wrapper while documenting the internal partial 3D state accurately.

**Files created/changed:**
- pygeoinf/pygeoinf/plot.py
- pygeoinf/pygeoinf/__init__.py
- pygeoinf/tests/test_plot.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md

**Functions created/changed:**
- plot_slice
- TestPlotSliceWrapper.test_plot_slice_exported
- TestPlotSliceWrapper.test_plot_slice_ball_2d_returns_figure
- TestPlotSliceWrapper.test_plot_slice_polyhedral_2d_returns_vertices
- TestPlotSliceWrapper.test_plot_slice_ball_1d_returns_mask
- TestPlotSliceWrapper.test_plot_slice_3d_raises_not_implemented

**Tests created/changed:**
- test_plot_slice_exported
- test_plot_slice_ball_2d_returns_figure
- test_plot_slice_polyhedral_2d_returns_vertices
- test_plot_slice_ball_1d_returns_mask
- test_plot_slice_3d_raises_not_implemented

**Review Status:** APPROVED

**Git Commit Message:**
feat(visualization): add plot_slice wrapper API

- Add public plot_slice convenience wrapper
- Export plot_slice from module and package root
- Cover 1D and 2D wrapper behavior in tests

Plan: pygeoinf/docs/agent-docs/active-plans/visualization_plan.md
Phase: 2 of 5
Related: pygeoinf/docs/agent-docs/completed-plans/visualization-plan-phase-2-complete.md