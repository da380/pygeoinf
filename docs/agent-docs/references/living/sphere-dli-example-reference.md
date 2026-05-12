# Sphere DLI Example Living Reference

> Status: All 4 phases complete as of 2026-04-20. Covers the full sphere DLI work-script slice on the `port_convex_analysis` branch.

## Scope

- Work script: `pygeoinf/work/sphere_dli_example.py`
- Benchmark script: `pygeoinf/work/sphere_cap_monte_carlo_benchmark.py`
- Test file: `pygeoinf/tests/test_sphere_dli_example.py`
- Related core API: `pygeoinf/pygeoinf/symmetric_space/symmetric_space.py` and `pygeoinf/pygeoinf/symmetric_space/sphere.py`
- Related core tests: `pygeoinf/tests/symmetric_space/test_sphere_cap_integrals.py`
- Plan anchor: `pygeoinf/docs/agent-docs/active-plans/sphere-dli-example-plan.md`

## Purpose

End-to-end example and test bench for the norm-ball-specialised DLI solver described in `pygeoinf/docs/agent-docs/theory/support_function.pdf`. Infers spherical-cap averages of a phase-velocity perturbation field on the two-sphere from great-circle ray travel-time data.

## Module constants

- `ORDER = 1.5` — Sobolev order (ensures Dirac measure and point evaluation are continuous)
- `SCALE = 0.1` — characteristic length scale in radians (~5.7°)
- `PRIOR_SCALE = 0.05` — prior energy scale for the heat-kernel Gaussian measure
- `REFERENCE_VELOCITY_BASE = 1.0`, `REFERENCE_VELOCITY_LATITUDE_PERTURBATION = 0.12`, `REFERENCE_VELOCITY_LONGITUDE_PERTURBATION = 0.08` — smooth positive synthetic background field parameters for `c_0`
- `N_TARGET = 6`, `N_SOURCES = 5`, `N_RECEIVERS = 10`, `SIGMA_NOISE = 0.02`
- `DEFAULT_TARGET_LATLON`: six (lat°, lon°) target locations spread over the sphere

## Public API

All functions are import-safe; no expensive computation runs at module load time.

### Phase 1 — model space and property operator

- `build_model_space(min_degree=64) -> Sobolev`
  `Sobolev.from_heat_kernel_prior(PRIOR_SCALE, ORDER, SCALE, power_of_two=True, min_degree=min_degree)`.
- `reference_phase_velocity(point) -> float`
  Smooth positive synthetic reference phase velocity `c_0(lat, lon)` with mild large-scale latitude/longitude variation.
- `build_cap_property_operator(model_space, target_latlon_list=DEFAULT_TARGET_LATLON, cap_radius_rad=0.15, n_cap=40, seed=42, exact=True) -> LinearOperator`
  Rows are exact spherical-cap averages by default, assembled via `model_space.geodesic_ball_average(centre, model_space.radius * cap_radius_rad)`. Set `exact=False` to retain the previous empirical Monte Carlo average of `n_cap` random `dirac` components inside each cap.

Internal helpers: `_latlon_to_unit_xyz`, `_unit_xyz_to_latlon`, `_sample_cap_points`, `_weighted_geodesic_integral_form`.

Core cap-integral additions:
- `SymmetricHilbertSpace.geodesic_ball_quadrature(...)` is an additive hook for geometry-specific volume quadrature; the default raises `NotImplementedError`.
- `SymmetricHilbertSpace.geodesic_ball_integral(...)` and `geodesic_ball_average(...)` assemble linear forms from a geodesic-ball quadrature rule when one exists.
- `SymmetricSobolevSpace.geodesic_ball_integral(...)` delegates to the underlying Lebesgue space and wraps the resulting components on the Sobolev domain.
- `sphere.Lebesgue.spherical_cap_integral(...)` / `spherical_cap_average(...)` use `pyshtools.SHCoeffs.from_cap` to compute exact spherical-harmonic cap coefficients. `sphere.Lebesgue.geodesic_ball_integral(...)` maps geodesic radius to angular cap radius and uses this exact implementation.

### Phase 2 — forward operator and synthetic data

- `build_forward_operator(model_space, *, n_sources=N_SOURCES, n_receivers=N_RECEIVERS, seed=0, normalize_by_arclength=True, reference_velocity=reference_phase_velocity) -> (LinearOperator, paths)`
  Fetches real IRIS GSN stations and random USGS earthquakes (≥ M6.5). Builds a reference-weighted operator by sampling `1 / c_0` along each geodesic quadrature rule and accumulating `weight / c_0(point) * dirac(point)` components. If `normalize_by_arclength=True`, divides each row by total arc length so rows are reference-weighted path averages rather than raw integrals.
- `generate_synthetic_data(model_space, forward_operator, *, sigma_noise=SIGMA_NOISE, seed=42) -> (truth_model, data_vector)`
  Samples truth from the heat-kernel prior, adds Gaussian noise clipped to ±3σ.

### Phase 3 — DLI solve

- `solve_dli(model_space, forward_operator, property_operator, truth_model, data_vector, *, sigma_noise=SIGMA_NOISE, prior_radius_multiplier=3.0, max_iter=300, tol=1e-5) -> dict`
  Returns `{'lower': ndarray, 'upper': ndarray, 'true_values': ndarray}`.
  Prior ball: `BallSupportFunction(model_space, model_space.zero, 3*||truth||)`.
  Data ball: `BallSupportFunction(data_space, data_space.zero, 3*sigma_noise*sqrt(n_paths))`.
  Solver: `PrimalKKTSolver`; each support direction `q` is solved via
  `kkt_solver.solve(property_operator.adjoint(q))` for both ±eᵢ directions.

### Phase 4 — orchestrator and visualisation

- `run_example(*, min_degree=32, n_sources=N_SOURCES, n_receivers=N_RECEIVERS, n_target=N_TARGET, n_cap=40, sigma_noise=SIGMA_NOISE, seed=42, exact_cap_average=True) -> dict`
  Runs the full pipeline; returns the bounds dict plus all intermediate objects.
- `plot_results(result) -> None`
  Three figures: cartopy ray-network map, true-field colour map, horizontal interval bar chart.

`if __name__ == '__main__':` guard calls `run_example()` then `plot_results()`.

### Exact-vs-Monte-Carlo cap benchmark

- `work/sphere_cap_monte_carlo_benchmark.py` compares exact cap averages against the retained empirical Monte Carlo cap sampler.
- Default run: `lmax=32`, six target caps, `n_cap = 4, 8, ..., 2048`, eight Monte Carlo repeats, 24 smooth probe fields.
- Ground truth: rows assembled by `space.geodesic_ball_average(centre, radius).components`.
- Monte Carlo comparator: rows assembled by `_sample_cap_points(...)` and averaging `space.dirac(point).components`, matching `build_cap_property_operator(..., exact=False)`.
- Metrics: component relative L2 error, relative RMSE of cap averages applied to smooth probe fields, construction time for all target caps, and Monte Carlo/exact time ratio.
- Outputs under `work/figures/`: `sphere_cap_monte_carlo_benchmark_records.csv`, `sphere_cap_monte_carlo_benchmark_summary.csv`, `sphere_cap_monte_carlo_accuracy.png`, and `sphere_cap_monte_carlo_cost.png`.
- Reference result from 2026-05-12: exact construction median `1.339e-3 s`; at `n_cap=2048`, Monte Carlo median field-output relative RMSE `1.365e-2`, component relative L2 `5.061e-2`, and construction time `3.248 s` (~2426x exact time).

## Implementation notes

- `LinearForm` lives in `pygeoinf.linear_forms` on this branch (not `pygeoinf.hilbert_space`).
- Cap properties use exact spherical-harmonic cap averages by default. The previous `np.random.default_rng(seed)` 3-D rejection sampler is retained for `build_cap_property_operator(..., exact=False)`.
- `iris_stations` / `random_earthquakes` use stdlib `random`; the script saves/restores state.
- `prior.sample()` uses legacy global numpy RNG; the script saves/restores numpy state.
- `model_space.project_function(lambda _: c)` is the canonical way to make a constant field.

## Tests (13 DLI tests + 5 cap-integral tests, all passing)

`pygeoinf/tests/test_sphere_dli_example.py` prepends `pygeoinf/work` to `sys.path`.

`pygeoinf/tests/symmetric_space/test_sphere_cap_integrals.py` validates exact spherical-cap integrals against analytic formulas for area, first moments, second moments, zonal harmonics, non-zonal zeros, physical radius scaling, Sobolev-domain wrapping, and invalid-radius errors.

| Test | Phase | What it checks |
|------|-------|----------------|
| `test_model_space_builds` | 1 | `dim > 0`, `order == 1.5` |
| `test_cap_property_operator_shape` | 1 | codomain and matrix shape |
| `test_cap_property_operator_constant_field` | 1 | constant field → constant caps (rtol/atol 1e-3) |
| `test_cap_property_operator_exact_default_is_deterministic` | 1 | exact default cap operator is independent of old `n_cap` / `seed` sampling controls |
| `test_forward_operator_shape` | 2 | domain, codomain, path count |
| `test_forward_operator_finite` | 2 | finite output on random model |
| `test_forward_operator_constant_field_is_reference_weighted_path_average` | 2 | constant field → ray-average of `1 / c_0` (weighting regression) |
| `test_synthetic_data_shape` | 2 | data length == n_paths |
| `test_dli_bounds_finite` | 3 | all bounds finite (tiny problem) |
| `test_dli_bounds_ordered` | 3 | lower ≤ upper |
| `test_truth_inside_bounds` | 3 | truth inside interval (near-zero noise) |
| `test_end_to_end_tiny` | 4 | full pipeline smoke test |
| `test_script_importable` | 4 | import-safe, has `run_example`, `solve_dli`, `plot_results` |