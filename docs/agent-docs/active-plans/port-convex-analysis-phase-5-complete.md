## Phase 5 Complete: Port Visualisation & Public Surface

Ported plot.py (SubspaceSlicePlotter, plot_slice), updated __init__.py exports, pyproject.toml optional extras, test_plot.py, test_subsets.py, and cleaned test_halfspaces.py skip guards. 624 tests pass.

**Files created/changed:**
- `pygeoinf/plot.py` — MODIFIED: SubspaceSlicePlotter, plot_slice (+1087 lines, byte-identical to CA)
- `pygeoinf/__init__.py` — MODIFIED: exports for all CA classes (symmetric_space import removed)
- `pyproject.toml` — MODIFIED: optional extras for osqp, clarabel, plotly
- `tests/test_plot.py` — MODIFIED: +686 lines for SubspaceSlicePlotter/plot_slice tests (byte-identical to CA)
- `tests/test_subsets.py` — MODIFIED: +123 lines for Subset.plot() tests (byte-identical to CA)
- `tests/test_halfspaces.py` — MODIFIED: removed Phase 1 skip guards (-19 lines)

**Functions created/changed:**
- `SubspaceSlicePlotter` — NEW class for visualising subsets on affine subspaces
- `plot_slice()` — NEW function for 1D/2D/3D subset slicing
- `__init__.py` — exports: SupportFunction classes, optimisation classes, DualMasterCostFunction, SubspaceSlicePlotter, plot_slice, ProximalBundleMethod

**Tests created/changed:**
- `tests/test_plot.py` — ~25 new tests for plot_slice/SubspaceSlicePlotter
- `tests/test_subsets.py` — 5 new Subset.plot() entry-point tests

**Review Status:** APPROVED (plot.py, test_plot.py, test_subsets.py byte-identical to CA; __init__.py intentionally differs by removing symmetric_space import and adding ProximalBundleMethod to __all__)

**Git Commit Message:**
```
feat(viz): port SubspaceSlicePlotter, plot_slice, and public exports

- Add SubspaceSlicePlotter and plot_slice to plot.py (1087 lines)
- Update __init__.py with all CA class exports
- Add optional extras (osqp, clarabel, plotly) to pyproject.toml
- Add plot tests (25+) and Subset.plot() tests (5)
- Remove redundant skip guards from test_halfspaces.py

Plan: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-plan.md
Phase: 5 of 6
Related: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-phase-5-complete.md
```
