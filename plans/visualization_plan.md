# Visualization Plan — pygeoinf

Location: `pygeoinf/plans/visualization_plan.md`

Purpose
- Consolidate what has been implemented for visualization, list remaining work, and provide a small TDD-ready roadmap so Atlas/Sisyphus can pick up Phase 8.

Summary of current state (from `DUAL_MASTER_IMPLEMENTATION_PLAN.md`)
- SubspaceSlicePlotter (companion implementation) exists and is actively used (located in `pygeoinf/visualization.py`).
\n+Changelog:
- Refactor: `pygeoinf/plot.py` renamed to `pygeoinf/visualization.py` and imports updated.
- Membership-oracle slice plotting (1D/2D/3D) is implemented and working: bars (1D), filled contours (2D), voxel/surface (3D).
- UX enhancement: pixel-based bar height conversion implemented for consistent 1D rendering.
- `pygeoinf/visualization.py` is NOT yet created; `Subset.plot()` entrypoint not added; Plotly 3D backend not yet integrated.

Goals (Phase 8 core)
1. Consolidate SubspaceSlicePlotter into the official `pygeoinf/visualization.py` module.
2. Add a `Subset.plot(on_subspace=None, backend='auto', method='auto', **kwargs)` wrapper so users call `some_subset.plot(...)`.
3. Implement Plotly backend for interactive 3D rendering (optional for MVP but desirable).
4. Add demo notebook `pygeoinf/testing_sets/visualization_demo.ipynb` showing Ball/Ellipsoid/HalfSpace slices.
5. Add unit tests and smoke tests (`tests/test_visualization.py`) verifying API and basic outputs.

Concrete tasks (small, TDD friendly)
- Task V1 (MVP): Move `SubspaceSlicePlotter` into `pygeoinf/visualization.py` and expose `plot()` function.
  - Tests: `tests/test_visualization.py::test_visualization_smoke` should import `pygeoinf.visualization` and call `plot_slice` on a simple `Ball` subset slice (use a tiny Euclidean space stub).
- Task V2: Add `Subset.plot()` entrypoint that calls into `pygeoinf.visualization.plot_slice`.
  - Tests: new test asserting `callable(getattr(some_subset, 'plot'))` and that calling it returns a Figure-like object for 2D.
- Task V3: Add Plotly 3D backend.
  - Update `pyproject.toml` (add `plotly` to `extras` or dependencies) or add `requirements-visualization.txt` for notebooks.
  - Tests: smoke test that 3D backend returns a `plotly.graph_objs.Figure` object (or at least doesn't raise).
- Task V4: Demo notebook `pygeoinf/testing_sets/visualization_demo.ipynb` showing typical workflows and saving an HTML export for plotly.
- Task V5: Polishing + docs: add a short README section in `pygeoinf/README.md` or `pygeoinf/plans/visualization_plan.md` explaining how to call `Subset.plot()` and how agents should use it.

Suggested file locations
- Implementation: `pygeoinf/visualization.py` (move code from `pygeoinf/plot.py` into this module).
- API: add `plot(self, on_subspace=None, backend='auto', method='auto', **kwargs)` to the `Subset` base class in `pygeoinf/subsets.py` (simple wrapper delegating to visualization module).
- Tests: `tests/test_visualization.py` (exists as a failing stub; update to import and exercise the new API).
- Demo: `pygeoinf/testing_sets/visualization_demo.ipynb`.

Run/Test instructions (local)
```
cd pygeoinf
PYTHONPATH=. pytest -q tests/test_visualization.py::test_visualization_smoke
```

Agent handoffs and usage
- Prometheus: can be asked to research Plotly vs Matplotlib tradeoffs (we already have a Prometheus research plan entry).
- Explorer: find any existing plotting helpers and where `plot.py` currently lives.
- Atlas/Sisyphus: implement phases V1→V5 using TDD: create failing tests, implement minimal code, iterate.

Acceptance criteria (MVP)
- `pygeoinf/visualization.py` exists and exports `plot_slice` / `SubspaceSlicePlotter`.
- `Subset.plot()` exists and returns a matplotlib Figure for 2D slices.
- `tests/test_visualization.py` passes smoke test without external GUI dependencies (use Agg backend in CI if needed).

Notes
- Keep the Plotly dependency optional for users who do not need interactive 3D. Provide instructions for installing extras.
- Prefer small incremental commits: (1) move code, (2) add wrapper, (3) add tests, (4) add plotly backend, (5) add demo notebook.
