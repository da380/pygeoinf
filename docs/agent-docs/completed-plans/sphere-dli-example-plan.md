# Plan: Sphere DLI Phase-Velocity Example Script

Implement a standalone example script (`pygeoinf/work/sphere_dli_example.py`) that solves
a deterministic linear inference (DLI) problem for phase-velocity perturbations on the
two-sphere using real IRIS station geometry, great-circle ray paths, spherical-cap
target properties, and norm-ball support functions. The script is intended as the
canonical test bench for the new norm-ball-specialised support-function solver described
in `pygeoinf/docs/agent-docs/theory/support_function.pdf`.

---

## Phases

### Phase 1: Sphere Sobolev model space and spherical-cap property operator

- **Objective:** Build the model space and define the finite-dimensional property
  operator that returns spherical-cap averages of the phase-velocity perturbation field
  at a handful of target locations.
- **Files/Functions to modify/create:**
  - NEW: `pygeoinf/work/sphere_dli_example.py` (skeleton through end of Phase 1)
  - Uses: `Sobolev.from_heat_kernel_prior` from `pygeoinf/pygeoinf/symmetric_space/sphere.py`
  - Uses: `dirac` / `point_evaluation_operator` from `pygeoinf/pygeoinf/symmetric_space/symmetric_space.py`
  - Uses: `LinearOperator.from_linear_forms` from `pygeoinf/pygeoinf/linear_operators.py`
- **Tests to write (in a separate test file `pygeoinf/tests/test_sphere_dli_example.py`):**
  - `test_model_space_builds`: Sobolev space has expected `dim` and `order`.
  - `test_cap_property_operator_shape`: property operator has shape `(n_properties, model_space.dim)`.
  - `test_cap_property_operator_constant_field`: applying the property operator to a
    constant-value model returns a vector of that constant repeated (linearity smoke check).
- **Steps:**
  1. Write failing tests first, run them to confirm red.
  2. In `sphere_dli_example.py`:
     - Build `Sobolev.from_heat_kernel_prior(prior_scale, order, scale, power_of_two=True, min_degree=64)`
       with `order=1.5`, `scale=0.1`, `prior_scale=0.05`.
     - Define `n_target = 6` target points spread over the sphere.
     - For each target point build a spherical-cap property form:
       sample `n_cap=40` uniformly distributed points within `cap_radius=0.15` radians
       of the target using simple rejection sampling in 3-D, average their `dirac` forms
       (i.e. sum component arrays, divide by `n_cap`), and construct a `LinearForm`.
     - Assemble the property operator with `LinearOperator.from_linear_forms`.
  3. Run tests to confirm green.

---

### Phase 2: Forward operator and synthetic data generation

- **Objective:** Build the geodesic-path forward operator using real IRIS stations and
  random USGS earthquakes; normalise ray integrals to true path averages; generate
  synthetic phase-velocity perturbation data with additive noise.
- **Files/Functions to modify/create:**
  - `pygeoinf/work/sphere_dli_example.py` (extend through end of Phase 2)
  - Uses: `iris_stations`, `random_earthquakes`, `geodesic_distance`,
    `path_average_operator` from `pygeoinf/pygeoinf/symmetric_space/sphere.py`
  - Uses: `geodesic_integral` from `pygeoinf/pygeoinf/symmetric_space/symmetric_space.py`
  - Uses: `LinearOperator.from_linear_forms` from `pygeoinf/pygeoinf/linear_operators.py`
  - Uses: `EuclideanSpace` from `pygeoinf/pygeoinf/hilbert_space.py`
- **Tests to write:**
  - `test_forward_operator_shape`: operator maps from `model_space` to `EuclideanSpace(n_paths)`.
  - `test_forward_operator_finite`: applying the operator to a random model yields finite values.
  - `test_synthetic_data_shape`: synthetic data vector has length `n_paths`.
- **Steps:**
  1. Write failing tests, confirm red.
  2. In `sphere_dli_example.py`:
     - Fetch `n_receivers=30` IRIS stations, `n_sources=10` random earthquakes (magnitude ≥ 6.5).
     - Build all `n_paths = n_sources * n_receivers` source-receiver pairs.
     - Build the raw path integral operator via `model_space.path_average_operator(paths)`.
     - Normalise each row by the corresponding arc length to convert line integrals to
       true path averages: for each path compute
       `arc = model_space.geodesic_distance(p1, p2)` and scale the row by `1/arc`.
       Do this by constructing normalised `LinearForm` objects and reassembling with
       `LinearOperator.from_linear_forms`.
     - Generate a synthetic truth model by sampling from the heat-kernel prior measure.
     - Compute noise-free data, add Gaussian noise with `sigma_noise=0.02` (km/s),
       clipped to `±3*sigma_noise` to stay within the ball data-confidence set.
  3. Run tests to confirm green.

---

### Phase 3: DLI solve with norm-ball support functions

- **Objective:** Assemble the DLI cost function and compute upper/lower bounds for each
  target property component using `DualMasterCostFunction`, `ProximalBundleMethod`, and
  `solve_support_values`.
- **Files/Functions to modify/create:**
  - `pygeoinf/work/sphere_dli_example.py` (extend through end of Phase 3)
  - Uses: `BallSupportFunction` from `pygeoinf/pygeoinf/convex_analysis.py`
  - Uses: `DualMasterCostFunction` from `pygeoinf/pygeoinf/backus_gilbert.py`
  - Uses: `ProximalBundleMethod`, `solve_support_values`, `best_available_qp_solver`
    from `pygeoinf/pygeoinf/convex_optimisation.py`
- **Tests to write:**
  - `test_dli_bounds_finite`: upper and lower bounds are all finite for a tiny problem
    (`n_sources=2`, `n_receivers=3`, `n_target=2`, `n_cap=5`).
  - `test_dli_bounds_ordered`: for every property, lower bound ≤ upper bound.
  - `test_truth_inside_bounds`: on a deterministic tiny instance where noise is zero and
    data-confidence ball radius is set to 0, the true property value lies within
    `[lower, upper]` to a loose tolerance.
- **Steps:**
  1. Write failing tests, confirm red.
  2. In `sphere_dli_example.py`:
     - Define a model prior ball in model space:
       `model_ball = BallSupportFunction(model_space, radius=prior_energy_radius)`
       where `prior_energy_radius` is the Sobolev norm of the prior-sampled truth.
     - Define a data-confidence ball in data space:
       `data_ball = BallSupportFunction(data_space, radius=3*sigma_noise*sqrt(n_paths))`.
     - Build `cost = DualMasterCostFunction(forward_operator, property_operator,
       model_ball, data_ball, data_vector)`.
     - Build the bundle solver:
       `solver = ProximalBundleMethod(best_available_qp_solver(), max_iter=300, tol=1e-5)`.
     - Solve with `bounds = solve_support_values(cost, solver, n_properties)`.
     - Print a table of `[lower_i, true_i, upper_i]` for each property.
  3. Run tests to confirm green.

---

### Phase 4: Visualisation and end-to-end validation

- **Objective:** Add a self-contained visualisation section using `cartopy`/`matplotlib`
  showing the sphere ray network, the true field, and the inferred property intervals.
- **Files/Functions to modify/create:**
  - `pygeoinf/work/sphere_dli_example.py` (extend through end of Phase 4; add `if __name__
    == '__main__':` guard so the script is importable for tests)
  - Uses: `plot_geodesic_network`, `create_map_figure`, `plot` from
    `pygeoinf/pygeoinf/symmetric_space/sphere.py`
  - Uses: `plot_1d_distributions` or basic matplotlib bar charts
- **Tests to write:**
  - `test_end_to_end_tiny`: run the entire pipeline on a minimal problem
    (`n_sources=2`, `n_receivers=3`, `n_target=2`, `min_degree=16`) without plotting;
    check that output bounds dict has the right keys and all values are finite.
  - `test_script_importable`: import `sphere_dli_example` without error.
- **Steps:**
  1. Write failing tests, confirm red.
  2. Wrap the main script body in `if __name__ == '__main__':`.
  3. Add a `run_example(…)` function with configurable parameters that can be called
     by both the main block and the tests.
  4. Add visualisation inside `__main__`:
     - Map figure showing stations (triangles), earthquake epicentres (stars), and ray paths.
     - Colour map of the true phase-velocity perturbation field.
     - Horizontal bar chart showing property intervals `[lower, upper]` with the true
       value marked as a vertical line.
  5. Run tests to confirm green.
  6. Run the full touched-slice suite:
     ```
     conda activate inferences3 && cd pygeoinf && python -m pytest tests/test_sphere_dli_example.py -x -q
     ```

---

## Open Questions (resolved before writing)

1. Unknown variable: **phase-velocity perturbation** (user confirmed).
2. Target properties: **spherical-cap averages** (user confirmed).
3. Geometry: **real IRIS stations + random earthquakes** (user confirmed).
4. Artefact location: **`pygeoinf/work/sphere_dli_example.py`** only, with a matching test
   file `pygeoinf/tests/test_sphere_dli_example.py` (standard pygeoinf practice).
