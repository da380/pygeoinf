## Phase 4 Complete: Implement 3D Visualization Backend

Enabled public 3D slice plotting through the existing Matplotlib stack and removed the wrapper-level 3D rejection. The implementation now supports sampled 3D rendering for general subsets, preserves the exact polyhedral 3D path, respects parameter-space bounds, and documents the distinct payload semantics for sampled versus exact rendering.

**Files created/changed:**
- `pygeoinf/pygeoinf/plot.py`
- `pygeoinf/tests/test_plot.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- `SubspaceSlicePlotter._generate_param_grid`
- `SubspaceSlicePlotter._render_3d`
- `plot_slice`

**Tests created/changed:**
- `test_plot_slice_3d_supported`
- `test_plot_slice_ball_3d_backend`
- `test_plot_slice_3d_returns_figure_like`
- `test_plot_slice_ball_3d_mask_center`
- `test_subspace_slice_plotter_3d_direct`
- `test_plot_slice_polyhedral_3d_exact_path`
- `test_plot_slice_3d_mask_uses_param_coords`
- `test_3d_large_grid_warns`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(visualization): enable 3D slice plotting backend

- Enable public 3D plotting in `plot_slice()` using Matplotlib's 3D backend
- Render sampled 3D slices in parameter coordinates and preserve exact polyhedral 3D rendering
- Add 3D backend, exact-path, coordinate-space, and warning coverage in plotting tests

Plan: docs/agent-docs/active-plans/visualization_plan.md
Phase: 4 of 5
Related: docs/agent-docs/completed-plans/visualization-plan-phase-4-complete.md