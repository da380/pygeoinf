## Phase 3 Complete: Add Subset.plot Entry Point

Added a user-facing `Subset.plot()` convenience method that delegates to `plot_slice()` and keeps the plotting API inside the existing visualization module. The implementation supports an automatic canonical slice only for 1D and 2D Euclidean domains, preserves forwarded plotting kwargs, and now has explicit tests for delegation and warning-free default behavior.

**Files created/changed:**
- `pygeoinf/pygeoinf/subsets.py`
- `pygeoinf/tests/test_subsets.py`
- `pygeoinf/tests/test_plot.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- `Subset.plot`

**Tests created/changed:**
- `test_subset_plot_requires_subspace_or_builds_default`
- `test_ball_plot_delegates_to_plot_slice`
- `test_subset_plot_preserves_kwargs`
- `test_ball_1d_default_subspace_no_warning`
- `test_plot_delegates_to_plot_slice_function`
- `test_ball_plot_auto_default_2d`
- `test_ball_plot_requires_subspace_for_3d`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(visualization): add Subset.plot convenience entry point

- Add `Subset.plot()` as a thin delegation layer over `pygeoinf.plot.plot_slice`
- Auto-build a canonical slice only for 1D and 2D Euclidean subsets and require explicit subspaces otherwise
- Add delegation, kwarg-forwarding, and warning-free default-slice tests for subset plotting

Plan: docs/agent-docs/active-plans/visualization_plan.md
Phase: 3 of 5
Related: docs/agent-docs/completed-plans/visualization-plan-phase-3-complete.md