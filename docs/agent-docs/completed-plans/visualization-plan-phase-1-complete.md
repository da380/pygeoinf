## Phase 1 Complete: Restore Plot Module Baseline

Phase 1 established a truthful visualization baseline without expanding the plotting API. The work added headless-safe smoke coverage for the existing plotting module imports and for supported 2D `SubspaceSlicePlotter` paths, and the phase review approved the result.

**Files created/changed:**
- pygeoinf/tests/test_plot.py

**Functions created/changed:**
- TestVisualizationModuleImports.test_visualization_module_imports
- TestVisualizationModuleImports.test_subspace_slice_plotter_imports
- TestSubspaceSlicePlotter2D.test_subspace_slice_plotter_ball_2d
- TestSubspaceSlicePlotter2D.test_subspace_slice_plotter_polyhedral_2d

**Tests created/changed:**
- test_visualization_module_imports
- test_subspace_slice_plotter_imports
- test_subspace_slice_plotter_ball_2d
- test_subspace_slice_plotter_polyhedral_2d

**Review Status:** APPROVED

**Git Commit Message:**
test(visualization): add phase 1 baseline coverage

- Add import smoke coverage for pygeoinf.plot
- Add 2D slice tests for Ball and PolyhedralSet
- Make plotting tests reliably headless with Agg

Plan: pygeoinf/docs/agent-docs/active-plans/visualization_plan.md
Phase: 1 of 5
Related: pygeoinf/docs/agent-docs/completed-plans/visualization-plan-phase-1-complete.md