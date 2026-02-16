# Dual Master — Prometheus Research Plan

Reference: `pygeoinf/pygeoinf/DUAL_MASTER_IMPLEMENTATION_PLAN.md`

Purpose
- Capture research tasks, open questions, and recommended experiments required before implementation.

Scope
- Validate numerical design choices for SubgradientDescent (Phase 4.2).
- Survey plotting backends and strategy for `visualization.py` (Phase 8).
- Enumerate operator inverse availability for `Ellipsoid` support functions.
- Identify candidate dependencies and performance considerations (SciPy, NumPy, Plotly, joblib).

Research Tasks
- Step-size rules: implement and benchmark `diminishing`, `polyak`, and `adaptive` on toy problems.
- Robust subgradient oracles: confirm `support_point` behavior for complex `ConvexIntersection` and `PolyhedralSet`.
- Ellipsoid inverses: search for `inverse_operator` availability patterns across codebase; propose safe API for missing inverses.
- Visualization trade-offs: matplotlib (2D) vs plotly (3D interactive) and optional lightweight HTML export.
- Numerical stability: test with ill-conditioned operators; propose preconditioning or scaling heuristics.

Deliverables (Prometheus)
- A short report (2–4 bullets) per task with recommendations and sample code snippets.
- An updated `dual_master_implementation.md` with prioritized TDD phases and per-phase test names.

Delegation Recommendations
- Delegate file discovery to `Explorer` (3 parallel searches): locations of inverse computations, operator factories, and existing visualization helpers.
- Delegate deeper numerical analysis to `Oracle` (one task per major question).

Acceptance Criteria
- Prometheus returns: updated implementation plan, list of test names to add, and a short rationale for chosen step-size rules.

Run hints
```
# Ask Prometheus to run the research phase interactively in VS Code Copilot Chat:
@Prometheus Plan: dual_master research
```
