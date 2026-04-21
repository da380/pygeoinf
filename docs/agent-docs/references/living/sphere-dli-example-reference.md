# Sphere DLI Example Living Reference

> Status: All 4 phases complete as of 2026-04-20. Covers the full sphere DLI work-script slice on the `port_convex_analysis` branch.

## Scope

- Work script: `pygeoinf/work/sphere_dli_example.py`
- Test file: `pygeoinf/tests/test_sphere_dli_example.py`
- Plan anchor: `pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-plan.md`

## Purpose

End-to-end example and test bench for the norm-ball-specialised DLI solver described in `pygeoinf/docs/agent-docs/theory/support_function.pdf`. Infers spherical-cap averages of a phase-velocity perturbation field on the two-sphere from great-circle ray travel-time data.

## Module constants

- `ORDER = 1.5` — Sobolev order (ensures Dirac measure and point evaluation are continuous)
- `SCALE = 0.1` — characteristic length scale in radians (~5.7°)
- `PRIOR_SCALE = 0.05` — prior energy scale for the heat-kernel Gaussian measure
- `N_TARGET = 6`, `N_SOURCES = 10`, `N_RECEIVERS = 30`, `SIGMA_NOISE = 0.02`
- `DEFAULT_TARGET_LATLON`: six (lat°, lon°) target locations spread over the sphere

## Public API

All functions are import-safe; no expensive computation runs at module load time.

### Phase 1 — model space and property operator

- `build_model_space(min_degree=64) -> Sobolev`
  `Sobolev.from_heat_kernel_prior(PRIOR_SCALE, ORDER, SCALE, power_of_two=True, min_degree=min_degree)`.
- `build_cap_property_operator(model_space, target_latlon_list=DEFAULT_TARGET_LATLON, cap_radius_rad=0.15, n_cap=40, seed=42) -> LinearOperator`
  Rows are empirical spherical-cap averages: `n_cap` random `dirac` components inside each cap averaged and assembled with `LinearOperator.from_linear_forms`.

Internal helpers: `_latlon_to_unit_xyz`, `_unit_xyz_to_latlon`, `_sample_cap_points`.

### Phase 2 — forward operator and synthetic data

- `build_forward_operator(model_space, *, n_sources=N_SOURCES, n_receivers=N_RECEIVERS, seed=0) -> (LinearOperator, paths)`
  Fetches real IRIS GSN stations and random USGS earthquakes (≥ M6.5). Builds normalised path-average operator: each `geodesic_integral` form divided by `geodesic_distance` so rows are true path averages, not raw arc-length integrals.
- `generate_synthetic_data(model_space, forward_operator, *, sigma_noise=SIGMA_NOISE, seed=42) -> (truth_model, data_vector)`
  Samples truth from the heat-kernel prior, adds Gaussian noise clipped to ±3σ.

### Phase 3 — DLI solve

- `solve_dli(model_space, forward_operator, property_operator, truth_model, data_vector, *, sigma_noise=SIGMA_NOISE, prior_radius_multiplier=3.0, max_iter=300, tol=1e-5) -> dict`
  Returns `{'lower': ndarray, 'upper': ndarray, 'true_values': ndarray}`.
  Prior ball: `BallSupportFunction(model_space, model_space.zero, 3*||truth||)`.
  Data ball: `BallSupportFunction(data_space, data_space.zero, 3*sigma_noise*sqrt(n_paths))`.
  Solver: `DualMasterCostFunction` + `ProximalBundleMethod` + `solve_support_values` for ±eᵢ directions.

### Phase 4 — orchestrator and visualisation

- `run_example(*, min_degree=64, n_sources=N_SOURCES, n_receivers=N_RECEIVERS, n_target=N_TARGET, n_cap=40, sigma_noise=SIGMA_NOISE, seed=42) -> dict`
  Runs the full pipeline; returns the bounds dict plus all intermediate objects.
- `plot_results(result) -> None`
  Three figures: cartopy ray-network map, true-field colour map, horizontal interval bar chart.

`if __name__ == '__main__':` guard calls `run_example()` then `plot_results()`.

## Implementation notes

- `LinearForm` lives in `pygeoinf.linear_forms` on this branch (not `pygeoinf.hilbert_space`).
- Cap sampling uses `np.random.default_rng(seed)` with 3-D rejection sampling.
- `iris_stations` / `random_earthquakes` use stdlib `random`; the script saves/restores state.
- `prior.sample()` uses legacy global numpy RNG; the script saves/restores numpy state.
- `model_space.project_function(lambda _: c)` is the canonical way to make a constant field.

## Tests (12 total, all passing)

`pygeoinf/tests/test_sphere_dli_example.py` prepends `pygeoinf/work` to `sys.path`.

| Test | Phase | What it checks |
|------|-------|----------------|
| `test_model_space_builds` | 1 | `dim > 0`, `order == 1.5` |
| `test_cap_property_operator_shape` | 1 | codomain and matrix shape |
| `test_cap_property_operator_constant_field` | 1 | constant field → constant caps (rtol/atol 1e-3) |
| `test_forward_operator_shape` | 2 | domain, codomain, path count |
| `test_forward_operator_finite` | 2 | finite output on random model |
| `test_forward_operator_constant_field_is_path_average` | 2 | constant field → constant path averages (normalisation regression) |
| `test_synthetic_data_shape` | 2 | data length == n_paths |
| `test_dli_bounds_finite` | 3 | all bounds finite (tiny problem) |
| `test_dli_bounds_ordered` | 3 | lower ≤ upper |
| `test_truth_inside_bounds` | 3 | truth inside interval (near-zero noise) |
| `test_end_to_end_tiny` | 4 | full pipeline smoke test |
| `test_script_importable` | 4 | import-safe, has `run_example`, `solve_dli`, `plot_results` |