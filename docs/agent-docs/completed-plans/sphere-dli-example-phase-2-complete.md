## Phase 2 Complete: Forward operator and synthetic data generation

Built the normalised geodesic-path forward operator using real IRIS GSN stations and
random USGS earthquakes, dividing each `geodesic_integral` form by its arc length to
produce true path averages. Added a synthetic data generator that samples from the
heat-kernel prior and adds clipped Gaussian noise.

**Files created/changed:**
- `pygeoinf/work/sphere_dli_example.py` (extended)
- `pygeoinf/tests/test_sphere_dli_example.py` (extended — 3 Phase 2 tests added)

**Functions created/changed:**
- `build_forward_operator(model_space, *, n_sources, n_receivers, seed) -> (LinearOperator, paths)`
- `generate_synthetic_data(model_space, forward_operator, *, sigma_noise, seed) -> (truth_model, data_vector)`

**Tests created/changed:**
- `test_forward_operator_shape`
- `test_forward_operator_finite`
- `test_synthetic_data_shape`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(sphere-dli): Phase 2 — forward operator and synthetic data

- Add build_forward_operator: normalised arc-length path averages, IRIS+USGS geometry
- Add generate_synthetic_data: heat-kernel prior sample + clipped Gaussian noise
- Normalise each geodesic_integral by geodesic_distance for true path averages
- Save/restore stdlib random and numpy legacy RNG state for reproducibility
- Add 3 Phase 2 tests (shape, finite, data length); 6 passed total

Plan: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-plan.md
Phase: 2 of 4
Related: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-phase-2-complete.md
```
