## Phase 4 Complete: Visualisation, orchestrator, and end-to-end validation

Wrapped the full pipeline in `run_example()`, added a `plot_results()` function for
three diagnostic figures, guarded all heavy code under `if __name__ == '__main__':`,
and added a normalization regression test plus the end-to-end and importability tests.
Living reference updated to cover all four phases.

**Files created/changed:**
- `pygeoinf/work/sphere_dli_example.py` (extended — `run_example`, `plot_results`, `__main__` guard, updated docstring)
- `pygeoinf/tests/test_sphere_dli_example.py` (extended — 3 Phase 4 tests + normalization regression)
- `pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md` (fully updated)

**Functions created/changed:**
- `run_example(*, min_degree, n_sources, n_receivers, n_target, n_cap, sigma_noise, seed) -> dict`
- `plot_results(result) -> None`
- `test_forward_operator_constant_field_is_path_average` (normalization regression)

**Tests created/changed:**
- `test_forward_operator_constant_field_is_path_average`
- `test_end_to_end_tiny`
- `test_script_importable`

**Review Status:** APPROVED (post-revision: stale header fixed, normalization regression added, living reference updated)

**Git Commit Message:**
```
feat(sphere-dli): Phase 4 — visualisation, orchestrator, review fixes

- Add run_example(): full pipeline orchestrator callable from tests
- Add plot_results(): ray-network map, true-field map, interval bar chart
- Guard heavy code under if __name__ == '__main__'
- Add test_forward_operator_constant_field_is_path_average (normalisation regression)
- Add test_end_to_end_tiny and test_script_importable
- Fix stale module docstring (was "Phase 1 skeleton")
- Update sphere-dli-example living reference to cover all 4 phases and 12 tests

Plan: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-plan.md
Phase: 4 of 4
Related: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-phase-4-complete.md
```
