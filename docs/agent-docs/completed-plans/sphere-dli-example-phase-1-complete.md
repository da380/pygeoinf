## Phase 1 Complete: Sphere Sobolev model space and spherical-cap property operator

Built the Sobolev model space (order 1.5, pyshtools backend) and a finite-dimensional
property operator that returns spherical-cap averages of the phase-velocity perturbation
field at six target locations spread over the sphere.

**Files created/changed:**
- `pygeoinf/work/sphere_dli_example.py` (new — Phase 1 skeleton)
- `pygeoinf/tests/test_sphere_dli_example.py` (new — 3 Phase 1 tests)
- `pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md` (new)

**Functions created/changed:**
- `build_model_space(min_degree=64) -> Sobolev`
- `build_cap_property_operator(model_space, target_latlon_list, cap_radius_rad, n_cap, seed) -> LinearOperator`
- `_latlon_to_unit_xyz`, `_unit_xyz_to_latlon`, `_sample_cap_points` (internal helpers)

**Tests created/changed:**
- `test_model_space_builds`
- `test_cap_property_operator_shape`
- `test_cap_property_operator_constant_field`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(sphere-dli): Phase 1 — Sobolev model space and spherical-cap property op

- Add build_model_space: Sobolev.from_heat_kernel_prior, order=1.5, scale=0.1
- Add build_cap_property_operator: DiracForm averages inside spherical caps
- Add 3D rejection sampling helpers for uniform cap sampling
- Create test_sphere_dli_example.py with 3 Phase 1 smoke tests (3 passed)
- Create sphere-dli-example living reference

Plan: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-plan.md
Phase: 1 of 4
Related: pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-phase-1-complete.md
```
