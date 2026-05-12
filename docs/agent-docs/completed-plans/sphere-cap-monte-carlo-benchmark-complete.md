## Plan Complete: Sphere-Cap Exact-vs-Monte-Carlo Benchmark

The exact spherical-cap average implementation is now benchmarked against the
retained Monte Carlo comparator with durable outputs suitable for review and
future regression checks. The benchmark shows the expected Monte Carlo
`N^{-1/2}` error decay, while also demonstrating that the exact method is both
substantially more accurate and dramatically faster for this spherical-harmonic
setting.

**Phases Completed:** 1 of 1
1. ✅ Phase 1: Benchmark exact cap averages against Monte Carlo sampling

**All Files Created/Modified:**
- `pygeoinf/work/sphere_cap_monte_carlo_benchmark.py`
- `pygeoinf/docs/agent-docs/references/living/sphere-dli-example-reference.md`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_accuracy.png`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_cost.png`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_benchmark_records.csv`
- `pygeoinf/work/figures/sphere_cap_monte_carlo_benchmark_summary.csv`

**Key Functions/Classes Added:**
- `run_benchmark`
- `_exact_cap_rows`
- `_monte_carlo_cap_rows`
- `_smooth_probe_components`
- `_plot_accuracy`
- `_plot_cost`

**Test Coverage:**
- Benchmark script static check passing: ✅
- Smoke benchmark run passing: ✅
- Full benchmark run passing: ✅
- Exact cap regression suite passing: ✅ (`5 passed, 1 warning`)

**Recommendations for Next Steps:**
- If needed, add a reduced-size benchmark test that checks CSV/figure creation
  without committing to expensive runtime thresholds.
- If paper figures are needed, reuse the generated CSV summary to produce a
  layout that matches the manuscript typography exactly.