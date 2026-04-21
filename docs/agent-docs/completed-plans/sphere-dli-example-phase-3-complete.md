## Phase 3 Complete: DLI solve with norm-ball support functions

Assembled the full DLI cost function using `BallSupportFunction` for both the model
prior and data-confidence sets, then solved for upper and lower bounds on each target
property via `DualMasterCostFunction` + `ProximalBundleMethod` + `solve_support_values`.

**Files created/changed:**
- `pygeoinf/work/sphere_dli_example.py` (extended)
- `pygeoinf/tests/test_sphere_dli_example.py` (extended — 3 Phase 3 tests added)

**Functions created/changed:**
- `solve_dli(model_space, forward_operator, property_operator, truth_model, data_vector, *, sigma_noise, prior_radius_multiplier, max_iter, tol) -> dict`

**Tests created/changed:**
- `test_dli_bounds_finite`
- `test_dli_bounds_ordered`
- `test_truth_inside_bounds`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(sphere-dli): Phase 3 — DLI norm-ball solve

- Add solve_dli: BallSupportFunction for prior (3*||truth||) and data (3σ√n)
- Wire DualMasterCostFunction + ProximalBundleMethod + solve_support_values
- Solve ±eᵢ basis directions for upper/lower property bounds
- Add 3 Phase 3 tests (finite, ordered, truth-inside); 9 passed total

Plan: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-plan.md
Phase: 3 of 4
Related: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-phase-3-complete.md
```
