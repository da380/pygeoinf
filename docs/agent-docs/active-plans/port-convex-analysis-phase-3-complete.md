## Phase 3 Complete: Port Optimisation Layer

Ported convex_optimisation.py (2205 lines, 5 solver classes + QP infrastructure) and backus_gilbert.py DualMasterCostFunction, plus 8 test files. test_subgradient.py was rewritten with real tests (CA branch had only a TODO stub). All 579 tests pass.

**Files created/changed:**
- `pygeoinf/convex_optimisation.py` — NEW (2205 lines, byte-identical to CA)
- `pygeoinf/backus_gilbert.py` — MODIFIED: DualMasterCostFunction added (byte-identical to CA)
- `tests/test_subgradient.py` — NEW (103 lines, 4 tests, fresh implementation)
- `tests/test_proximal_bundle.py` — NEW (198 lines, byte-identical to CA)
- `tests/test_level_bundle.py` — NEW (213 lines, byte-identical to CA)
- `tests/test_smoothed_lbfgs.py` — NEW (258 lines, byte-identical to CA)
- `tests/test_chambolle_pock.py` — NEW (211 lines, byte-identical to CA)
- `tests/test_bundle_core.py` — NEW (220 lines, byte-identical to CA)
- `tests/test_qp_backends.py` — NEW (76 lines, byte-identical to CA)
- `tests/test_solve_support_values.py` — NEW (182 lines, byte-identical to CA)

**Functions created/changed:**
- `SubgradientDescent` — subgradient descent solver
- `ProximalBundleMethod` — proximal bundle method
- `LevelBundleMethod` — level bundle method
- `SmoothedLBFGSMethod` — smoothed L-BFGS continuation method
- `ChambolleChockMethod` — Chambolle-Pock primal-dual method
- `QPBackend`, `OSQPBackend`, `ClarabelBackend` — QP solver backends
- `solve_support_function_values` — multi-direction support function solver
- `DualMasterCostFunction` — dual master cost oracle for BG formulations

**Tests created/changed:**
- 47 new optimisation tests across 8 files
- test_subgradient.py: 4 tests (fresh, replacing TODO stub)

**Review Status:** APPROVED (review concerns are pre-existing CA design decisions, not port defects; noted for reviewer guide)

**Notes:**
- backus_gilbert.py was pulled into Phase 3 (originally Phase 4) because 5 test files depend on DualMasterCostFunction
- __init__.py exports deferred to Phase 5

**Git Commit Message:**
```
feat(optimisation): port convex_optimisation.py and DualMasterCostFunction

- Add convex_optimisation.py with 5 solver classes + QP infrastructure (2205 lines)
- Add DualMasterCostFunction to backus_gilbert.py
- Add 8 test files with 47 new tests
- Replace test_subgradient.py TODO stub with 4 real tests

Plan: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-plan.md
Phase: 3 of 6
Related: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-phase-3-complete.md
```
