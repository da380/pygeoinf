## Plan Complete: Function-Space Hardening

Extended `GaussianMeasure.credible_set` to support infinite-dimensional
function spaces via two new geometry modes (`ambient_ball`,
`weakened_ellipsoid`) backed by three new modules: a weighted chi-square
quantile engine (`quadratic_form_quantile`), a spectral fractional
operator (`spectral_operator`), and a Lanczos matrix-function applicator
(`matrix_function`). End-to-end integration tests on `intervalinf`'s
basis-free `Lebesgue` spaces confirm correctness. A critical Imhof
adaptive-truncation fix eliminates OOM kills on decaying spectra (625×
smaller integration domain for N=50 InverseLaplacian).

**Phases Completed:** 6 of 6
1. ✅ Phase 1: `quadratic_form_quantile.py` — Imhof/WS/saddlepoint/MC weighted chi-square (42 tests)
2. ✅ Phase 2: `spectral_operator.py` — `SpectralFractionalOperator` (14 tests)
3. ✅ Phase 3: `matrix_function.py` — Lanczos `apply_matrix_function` (13 tests)
4. ✅ Phase 4: Extended `credible_set()` with `ambient_ball`/`weakened_ellipsoid` (29 tests)
5. ✅ Phase 5: `intervalinf` basis-free integration tests (6 tests); Imhof adaptive truncation fix
6. ✅ Phase 6: Living references updated, demo verified, plan archived

**All Files Created/Modified:**
- `pygeoinf/pygeoinf/quadratic_form_quantile.py` (new module, then perf-fixed)
- `pygeoinf/pygeoinf/spectral_operator.py` (new module)
- `pygeoinf/pygeoinf/matrix_function.py` (new module)
- `pygeoinf/pygeoinf/gaussian_measure.py` (extended +525 lines)
- `pygeoinf/tests/test_quadratic_form_quantile.py` (new, 42 tests)
- `pygeoinf/tests/test_spectral_operator.py` (new, 14 tests)
- `pygeoinf/tests/test_matrix_function.py` (new, 13 tests)
- `pygeoinf/tests/test_gaussian_measure_credible_set.py` (extended, +29 tests)
- `intervalinf/tests/spaces/test_lebesgue_hardening.py` (new, 6 tests)
- `pygeoinf/docs/agent-docs/references/living/convex-analysis-reference.md` (updated)
- `pygeoinf/work/function_space_hardening_demo.py` (verified)

**Key Functions/Classes Added:**
- `weighted_chi2_cdf(weights, t, *, method, rtol, ...)` — public CDF
- `weighted_chi2_quantile(weights, probability, *, method, rtol, ...)` — public quantile
- `_imhof_cdf(weights, t, *, rtol)` — adaptive truncation (1/(π U ρ(U)) < rtol)
- `SpectralFractionalOperator(LinearOperator)` — finite-rank f(C) from `LowRankEig`
- `SpectralFractionalOperator.from_low_rank_eig(eig, power)` — factory
- `lanczos_tridiagonalize(op, v, k, *, reorth)` — Lanczos reduction
- `apply_matrix_function(op, v, func, k, *, reorth)` — matrix-free f(C)v
- `GaussianMeasure.credible_set` — extended: `geometry∈{"ambient_ball","weakened_ellipsoid"}`, `theta`, `spectrum`, `spectrum_size`, `radius_method`, `quantile_method`, `fractional_apply`, `n_samples`, `n_lanczos`, `spectrum_low_rank_kwargs`, `rng`
- `GaussianMeasure.ambient_ball(probability, /, **kwargs)` — convenience wrapper
- `GaussianMeasure.weakened_ellipsoid(probability, /, *, theta, **kwargs)` — convenience wrapper

**Test Coverage:**
- Total new tests: 104 (98 in pygeoinf + 6 in intervalinf)
- All tests passing: ✅
- pygeoinf hardening suite runtime: ~10 s (was 41 s before Imhof fix)
- intervalinf hardening suite runtime: ~11 s (was 70 s / OOM-killed before fix)

**Performance Fix Summary:**
- `_imhof_cdf`: old `U = 16/w_min` → U ≈ 400 000 (N=50 InverseLaplacian)
- New adaptive: find U where `1/(π U ρ(U)) < rtol` via binary search → U ≈ 640
- Speedup: credible_set call N=50 from ~6.5 s → ~10 ms (650×)

**Recommendations for Next Steps:**
- Consider a batched `_sampling_radius` for basis-free function spaces (current per-sample L² quadrature costs ~13 ms/call; vectorising over samples would reduce to ~0.1 ms/sample)
- The WS approximation is 1000× faster than Imhof for large N; add a warning or auto-switch when N > 500
- For N → ∞ theoretical analysis: the ambient-ball radius converges to `sqrt(trace(C))` once the spectrum is summable; document this in `credible_set` docstring
