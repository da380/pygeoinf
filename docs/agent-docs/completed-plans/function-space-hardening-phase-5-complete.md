## Phase 5 Complete: intervalinf basis-free integration tests

Exercised `GaussianMeasure.credible_set` end-to-end on `intervalinf`'s
basis-free `Lebesgue` space with `InverseLaplacian` (Cauchy spectrum)
and `BesselSobolevInverse` covariances. All six tests pass.

**Files created/changed:**
- `intervalinf/tests/spaces/test_lebesgue_hardening.py` (new)

**Functions created/changed:**
- `_make_inverse_laplacian_measure` (test helper)
- `_spectrum_callable` (test helper)

**Tests created/changed:**
- `test_ambient_ball_spectral_convergence` — $r_p(N)$ Cauchy in $N\in\{50,200,1000\}$
- `test_ambient_ball_spectral_vs_sampling` — sampling and spectral radii agree within MC + KL-truncation tolerance (15 %)
- `test_weakened_ellipsoid_lanczos_vs_spectral` — Lanczos action $C^{-\theta}$ on leading eigenfunction matches $\lambda_0^{-\theta}$ within 5e-3
- `test_sobolev_weakened_ellipsoid` — `BesselSobolevInverse` covariance produces a finite positive radius
- `test_no_spectrum_no_sampling_raises` — informative `ValueError` when neither path is available
- `test_cm_warning` — near-Cameron–Martin truncation ($\theta = 0.99$) emits `UserWarning`

**Review Status:** APPROVED (self-reviewed; all tests green; tolerances justified)

**Git Commit Message:**
```
test(hardening): Phase 5 — intervalinf basis-free credible-set tests

- Add tests/spaces/test_lebesgue_hardening.py with 6 cases covering
  ambient-ball spectral convergence, ambient-ball spectral-vs-sampling
  agreement, Lanczos fractional-gauge action, Sobolev/Bessel covariance,
  the no-spectrum/no-sampling ValueError path, and the near-Cameron-Martin
  UserWarning.
- Exercises GaussianMeasure.credible_set end-to-end against
  basis-free Lebesgue + InverseLaplacian and BesselSobolevInverse.

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: 5 of 6
Related: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-phase-5-complete.md
```
