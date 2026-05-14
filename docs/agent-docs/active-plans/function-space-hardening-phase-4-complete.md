## Phase 4 Complete: Wire spectrum-aware paths into credible_set

Extended `GaussianMeasure.credible_set()` with full spectrum-aware calibration infrastructure. Supports ambient norm balls, weakened covariance ellipsoids, and three spectrum-resolution paths (user-supplied, callable, randomized). All 98 Phases 1–4 tests pass.

**Files created/changed:**
- `pygeoinf/gaussian_measure.py` (+525 lines; modified lines 428–522)
- `tests/test_gaussian_measure_credible_set.py` (+218 lines; 29 new tests for new modes)

**Functions created/changed:**
- `GaussianMeasure.credible_set()` — extended signature with `geometry`, `theta`, `spectrum`, `spectrum_size`, `radius_method`, `quantile_method`, `fractional_apply`, `n_samples`, `n_lanczos`, `spectrum_low_rank_kwargs`, `rng`
- `GaussianMeasure.ambient_ball()` — convenience wrapper for ambient norm ball
- `GaussianMeasure.weakened_ellipsoid()` — convenience wrapper for weakened ellipsoid
- `GaussianMeasure._resolve_spectrum()` — spectrum resolution logic (array, LowRankEig, callable, randomized fallback)
- `GaussianMeasure._spectral_radius()` — compute $r_p^2$ via weighted-chi-square quantile
- `GaussianMeasure._sampling_radius()` — compute $r_p^2$ via Monte Carlo gauge sampling
- `GaussianMeasure._build_gauge_squared()` — construct gauge function for sampling radius

**Tests created/changed:**
- `test_theta_weakened_ellipsoid_mutual()` — `theta` and `geometry="weakened_ellipsoid"` mutually required
- `test_equal_weight_degenerate_case()` — `spectrum = np.ones(k)`, `theta = 0` matches $\chi^2_k(p)$
- `test_ambient_ball_spectral()` — ambient ball with spectral radius calibration
- `test_weakened_ellipsoid_spectral()` — weakened ellipsoid with Lanczos + spectral paths
- `test_backward_compatibility_chi2()` — existing `geometry="ellipsoid"` unchanged
- `test_coverage_empirical_5d()` — 5000 samples, 5 geometries, coverage within binomial tolerance
- 23 additional tests for spectrum resolution, sampling, quantile methods, edge cases

**Verification:**
```bash
conda activate inferences3
cd pygeoinf && python -m pytest tests/test_quadratic_form_quantile.py \
  tests/test_spectral_operator.py tests/test_matrix_function.py \
  tests/test_gaussian_measure_credible_set.py -v
# Result: 98 passed in 41.06s
```

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(hardening): Phase 4 — wire spectrum-aware paths into credible_set

- Extend GaussianMeasure.credible_set() signature (backward compatible)
- New geometries: "ambient_ball", "weakened_ellipsoid" (theta in [0, 1))
- Three spectrum paths: user-supplied, callable eigenvalues, randomized LowRankEig
- Three radius methods: spectral (Imhof/WS/saddlepoint), sampling (MC), auto routing
- Fractional-apply paths: spectral (fast), Lanczos (robust), auto selection
- Convenience wrappers: ambient_ball(), weakened_ellipsoid()
- 29 new tests; empirical coverage validated on EuclideanSpace(5) to 3σ tolerance
- Backward compatibility verified: chi2 calibration unchanged for rank-k truncations

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: 4 of 6
Related: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-phase-4-complete.md
```
