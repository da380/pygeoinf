## Phase 2 Complete: Spectral fractional operator

Implemented matrix-free fractional-power operators constructed from low-rank eigendecompositions. Enables efficient application of $C^{-\theta/2}$ to vectors for weakened-ellipsoid gauge evaluation. All tests pass.

**Files created/changed:**
- `pygeoinf/spectral_operator.py` (261 lines)
- `tests/test_spectral_operator.py` (203 lines)

**Functions created/changed:**
- `SpectralFractionalOperator.__init__()` — construct from $(U, \Lambda, f)$
- `SpectralFractionalOperator.__matmul__()` — matrix-free forward application
- `SpectralFractionalOperator.adjoint` — adjoint operator (symmetric $f$)
- `SpectralFractionalOperator.matrix()` — dense matrix (lazy evaluation)
- `SpectralFractionalOperator.from_low_rank_eig()` — factory from `LowRankEig` instance
- `SpectralFractionalOperator.from_callable()` — factory from eigenpairs + callable power

**Tests created/changed:**
- `test_round_trip_identity()` — $C^\theta \cdot C^{-\theta} \approx I$ on eigenspace
- `test_adjoint_symmetry()` — adjointness for symmetric $f$
- `test_diagonal_match()` — exact match vs `DiagonalSparseMatrixLinearOperator ** power`
- `test_fractional_power_sanity()` — $C^{0.5} @ C^{0.5} \approx C$ (Cholesky-like)
- 10 additional edge cases and broadcast tests

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(hardening): Phase 2 — spectral fractional operator

- Add pygeoinf/spectral_operator.py: SpectralFractionalOperator for matrix-free $f(C)$
- Factories: from_low_rank_eig(), from_callable()
- Supports symmetric matrix functions: adjoint + dense matrix materialization
- 14 unit tests; round-trip $C^\theta \cdot C^{-\theta}$ verified to machine precision
- Exact agreement with DiagonalSparseMatrixLinearOperator on isotropic cases

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: 2 of 6
Related: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-phase-2-complete.md
```
