# Plan: Sphere-Cap Exact-vs-Monte-Carlo Benchmark

Add a reproducible benchmark for the new exact spherical-cap average functional
against the retained Monte Carlo cap sampler. The benchmark should quantify
accuracy, convergence, and computational cost using the same public interfaces
and internal sampling path already exposed by
`pygeoinf/work/sphere_dli_example.py`, and it should produce paper-ready plots
plus machine-readable summary tables under `pygeoinf/work/figures/`.

---

## Phases

### Phase 1: Benchmark exact cap averages against Monte Carlo sampling

- **Objective:** Compare the exact spherical-harmonic cap average against the
  legacy Monte Carlo cap-average construction across increasing sample counts,
  and generate figures showing both convergence/accuracy and computational
  cost.
- **Files/Functions to modify/create:**
  - NEW: `pygeoinf/work/sphere_cap_monte_carlo_benchmark.py`
  - UPDATE: `pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md`
  - OUTPUT: `pygeoinf/work/figures/sphere_cap_monte_carlo_accuracy.png`
  - OUTPUT: `pygeoinf/work/figures/sphere_cap_monte_carlo_cost.png`
  - OUTPUT: `pygeoinf/work/figures/sphere_cap_monte_carlo_benchmark_records.csv`
  - OUTPUT: `pygeoinf/work/figures/sphere_cap_monte_carlo_benchmark_summary.csv`
  - Reuse exact path: `geodesic_ball_average` from
    `pygeoinf/pygeoinf/symmetric_space/symmetric_space.py`
  - Reuse sphere exact implementation: `spherical_cap_integral` /
    `geodesic_ball_integral` from
    `pygeoinf/pygeoinf/symmetric_space/sphere.py`
  - Reuse Monte Carlo comparator path: `_sample_cap_points` and
    `build_cap_property_operator(..., exact=False)` logic from
    `pygeoinf/work/sphere_dli_example.py`
- **Tests / validation to run:**
  - Focused static correctness check on the new benchmark script.
  - Smoke benchmark run on a tiny configuration to confirm plotting and CSV
    generation.
  - Full benchmark run with increasing `n_cap` values.
  - Regression check that the exact cap-integral test suite still passes.
- **Steps:**
  1. Build a benchmark harness that assembles exact cap-average rows and Monte
     Carlo rows for the same target caps and Sobolev space.
  2. Measure error in two complementary ways:
     - relative component-space error of the cap linear forms,
     - relative RMSE when those forms are applied to a fixed ensemble of smooth
       probe fields.
  3. Measure construction time for the exact rows and Monte Carlo rows across
     a sweep of sample counts.
  4. Write CSV outputs and generate two figures: convergence/accuracy and cost.
  5. Record the benchmark methodology and headline numbers in the living
     reference document.

---

## Open Questions (resolved before writing)

1. Exact reference: use the new public `geodesic_ball_average` API or raw
   coefficient internals? Resolved: use the public API.
2. Monte Carlo comparator: invent a new sampler or benchmark the retained
   legacy path? Resolved: benchmark the retained legacy sampler exactly as kept
   in the work script.
3. Accuracy metric: compare only coefficients or also operator action on fields?
   Resolved: report both.
4. Output format: terminal-only summary or durable artefacts? Resolved: save
   CSV summaries and PNG figures in `work/figures/`.