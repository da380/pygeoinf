## Phase 3 Complete: Sync Docs and Examples

Updated the user-facing visualization documentation and demo assets to reflect the new interactive 3D backend behavior. The README now documents `backend="auto"` and the optional `interactive` extra, the demo notebook includes an interactive 3D example that uses Plotly when available, and the living reference stays aligned with the actual warning-and-fallback semantics.

**Files created/changed:**
- `pygeoinf/README.md`
- `pygeoinf/pygeoinf/testing_sets/visualization_demo.ipynb`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
- `pygeoinf/docs/agent-docs/completed-plans/interactive-3d-visualization-phase-3-complete.md`

**Functions created/changed:**
- No production functions changed in this phase

**Tests created/changed:**
- No new automated tests added
- Updated demo notebook cells to cover `backend="matplotlib"` and interactive `backend="auto"` usage
- Re-validated `tests/test_plot.py`

**Review Status:** APPROVED

**Git Commit Message:**
docs(visualization): document interactive 3D backend and demo workflow

- Update README with `backend="auto"` 3D usage, return-type notes, and `interactive` install instructions
- Extend the visualization demo notebook with explicit Matplotlib and Plotly-backed 3D examples
- Sync the living reference with the warning-and-fallback semantics for Plotly auto-selection

Plan: docs/agent-docs/active-plans/interactive-3d-visualization-plan.md
Phase: 3 of 3
Related: docs/agent-docs/completed-plans/interactive-3d-visualization-phase-3-complete.md