## Plan Complete: Sphere DLI Phase-Velocity Example Script

A fully-tested, runnable DLI example for phase-velocity perturbation inference on the
two-sphere has been created at `pygeoinf/work/sphere_dli_example.py`. The script covers
the complete DLI pipeline — Sobolev model space, real ray geometry, spherical-cap target
properties, and norm-ball convex-analysis solve — and serves as the canonical test bench
for the new support-function solver described in the theory document.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Sobolev model space and spherical-cap property operator
2. ✅ Phase 2: Forward operator (normalised path averages) and synthetic data
3. ✅ Phase 3: DLI norm-ball solve with ProximalBundleMethod
4. ✅ Phase 4: Visualisation, orchestrator, and end-to-end validation

**All Files Created/Modified:**
- `pygeoinf/work/sphere_dli_example.py` (new)
- `pygeoinf/tests/test_sphere_dli_example.py` (new)
- `pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md` (new)

**Key Functions/Classes Added:**
- `build_model_space` — Sobolev.from_heat_kernel_prior, order 1.5
- `build_cap_property_operator` — spherical-cap Dirac averages
- `build_forward_operator` — normalised geodesic-path operator, IRIS+USGS geometry
- `generate_synthetic_data` — heat-kernel prior sample + clipped Gaussian noise
- `solve_dli` — BallSupportFunction + DualMasterCostFunction + ProximalBundleMethod
- `run_example` — full pipeline orchestrator with configurable parameters
- `plot_results` — three-figure diagnostic output

**Test Coverage:**
- Total tests written: 12
- All tests passing: ✅ (`12 passed, 351 warnings in 119.55s`)
- Broader suite unaffected: ✅ (`728 passed, 1 xfailed`)

**Recommendations for Next Steps:**
- Implement the norm-ball-specialised solver from `theory/support_function.pdf` and add
  it as an alternative backend in `solve_dli` for direct speed comparison.
- Optionally reduce the test suite runtime by caching the model space and forward
  operator with `@pytest.fixture(scope="module")`.
- Consider promoting to a notebook once the new solver is benchmarked.
