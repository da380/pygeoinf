## Phase 1 Complete: Weighted chi-square quantile module

Implemented the core weighted chi-square distribution backend for calibrating credible sets on function spaces. The module supports four numerical methods (Imhof, Welch–Satterthwaite, saddlepoint, Monte Carlo) and all tests pass.

**Files created/changed:**
- `pygeoinf/quadratic_form_quantile.py` (441 lines)
- `tests/test_quadratic_form_quantile.py` (150 lines)

**Functions created/changed:**
- `weighted_chi2_cdf()` — evaluate CDF of $\sum_j w_j Z_j^2$
- `weighted_chi2_quantile()` — invert CDF for probability $p$
- `_imhof_cdf_integral()` — Imhof numerical inversion
- `_welch_satterthwaite_approximation()` — closed-form moment matching
- `_saddlepoint_approximation()` — Lugannani–Rice tail formula
- `_monte_carlo_quantile()` — empirical reference

**Tests created/changed:**
- `test_equal_weights_match_chi2()` — degenerate isotropic case vs. `scipy.stats.chi2`
- `test_imhof_vs_mc()` — cross-validation on anisotropic weights
- `test_imhof_vs_ws_asymptotic()` — asymptotic agreement
- `test_anisotropic_heavy_tail()` — WS/Imhof bracketing
- `test_saddlepoint_deep_tail()` — tail accuracy
- 37 additional parametrized tests covering edge cases, extreme anisotropy, broadcast shapes

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(hardening): Phase 1 — weighted chi-square quantile module

- Add pygeoinf/quadratic_form_quantile.py with Imhof, WS, saddlepoint, MC methods
- Public API: weighted_chi2_cdf(), weighted_chi2_quantile()
- 42 unit tests, all passing; validates equal-weights degenerate case to 1e-8
- Cross-validates Imhof vs MC on anisotropic spectra (3σ tolerance)
- Supports arbitrary weight arrays and broadcast evaluation

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: 1 of 6
Related: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-phase-1-complete.md
```
