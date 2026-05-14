## Phase 3 Complete: Lanczos matrix-function module

Implemented Lanczos-based matrix-function application for computing $f(C) v$ without explicit eigendecomposition. Provides robust fallback when spectrum is unavailable. Full reorthogonalisation preserves numerical stability. All tests pass.

**Files created/changed:**
- `pygeoinf/matrix_function.py` (216 lines)
- `tests/test_matrix_function.py` (178 lines)

**Functions created/changed:**
- `lanczos_tridiagonalize()` — classical Lanczos with full reorthogonalisation
- `apply_matrix_function()` — evaluate $f(C) v$ via Krylov projection
- `_lanczos_full_reorth()` — full reorthogonalisation kernel
- Helper `_lanczos_convergence_check()` — eigenvalue definiteness validation

**Tests created/changed:**
- `test_dense_fractional_power_match()` — dense $C^{-0.5}$ vs. `scipy.linalg.fractional_matrix_power` to 1e-10
- `test_convergence_rate_smooth_spectrum()` — geometric convergence vs. Krylov dimension $k$
- `test_full_vs_no_reorth()` — orthogonality parity with/without reorthogonalisation
- `test_breakdown_detection()` — invariant subspace identification ($\beta_j = 0$)
- `test_high_condition_number()` — robustness on $\kappa \sim 10^6$
- 8 additional tests for edge cases, breakdown, and orthogonality monitoring

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(hardening): Phase 3 — Lanczos matrix-function module

- Add pygeoinf/matrix_function.py: lanczos_tridiagonalize(), apply_matrix_function()
- Full reorthogonalisation by default; breakdown detection + fallback handling
- Numerically stable for high-condition covariances ($\kappa > 10^6$)
- 13 unit tests; dense $C^{-0.5}$ match verified to 1e-10
- Convergence rate (geometric in Krylov dim) validated on synthetic SPD matrices

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: 3 of 6
Related: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-phase-3-complete.md
```
