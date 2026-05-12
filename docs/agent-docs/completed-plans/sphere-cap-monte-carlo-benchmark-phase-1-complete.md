## Phase 1 Complete: Benchmark exact cap averages against Monte Carlo sampling

Implemented a standalone benchmark script that compares exact spherical-cap
averages against the retained Monte Carlo cap sampler over increasing sample
counts, writes CSV summaries, and generates figures for both convergence and
runtime scaling. The benchmark uses the exact `geodesic_ball_average` rows as
ground truth and measures both component-space error and field-output error on
smooth probe fields.

**Files created/changed:**
- `pygeoinf/work/sphere_cap_monte_carlo_benchmark.py`
- `pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_accuracy.png`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_cost.png`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_benchmark_records.csv`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_benchmark_summary.csv`

**Functions created/changed:**
- `run_benchmark(args)`
- `_exact_cap_rows(...)`
- `_monte_carlo_cap_rows(...)`
- `_smooth_probe_components(...)`
- `_plot_accuracy(...)`
- `_plot_cost(...)`

**Tests created/changed:**
- No new unit tests added for the benchmark harness itself.
- Validation run: `python work/sphere_cap_monte_carlo_benchmark.py`
- Static check: `python -m ruff check --select F work/sphere_cap_monte_carlo_benchmark.py`
- Regression check: `python -m pytest tests/symmetric_space/test_sphere_cap_integrals.py -x -q`

**Review Status:** APPROVED (retroactive documentation; benchmark execution and focused checks completed)

**Git Commit Message:**
```
feat(sphere-cap): add exact-vs-MC cap benchmark

- Add benchmark script for exact vs Monte Carlo cap averages
- Generate convergence and runtime comparison figures plus CSVs
- Update living reference with benchmark method and headline results
- Validate script with smoke/full runs and cap-integral regression

Plan: pygeoinf/docs/agent-docs/completed-plans/sphere-cap-monte-carlo-benchmark-plan.md
Phase: 1 of 1
Related: pygeoinf/docs/agent-docs/completed-plans/sphere-cap-monte-carlo-benchmark-phase-1-complete.md
```