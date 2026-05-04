## Plan Complete: UndiscretizedPrimalKKTSolver

`PrimalKKTSolver` has been implemented, reviewed, and all tests pass.
The new class is appended to `convex_optimisation.py` alongside — and without
any modification to — the existing `PrimalKKTSolver` benchmark.  The solver
operates entirely in abstract Hilbert-space vectors; only the $M \times M$
data-space Cholesky solve is numerical.  End-to-end agreement with the dense
solver is verified for all four B/V combinations.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Core implementation + Euclidean regression tests (4 Woodbury + 5 solve tests)
2. ✅ Phase 2: Basis-freedom guard tests (structural shape checks; Lebesgue monkey-patch in Phase 3)
3. ✅ Phase 3: intervalinf integration tests (Lebesgue + SOLAOperator, monkey-patch proof)
4. ✅ Phase 4: Export, docstring with ball/ball simplification, final regression

**All Files Created/Modified:**
- `pygeoinf/pygeoinf/convex_optimisation.py` — appended `PrimalKKTSolver`
- `pygeoinf/pygeoinf/__init__.py` — export added to import block and `__all__`
- `pygeoinf/tests/test_primal_kkt_solver_basis_free.py` — 18 tests
- `intervalinf/tests/operators/test_primal_kkt_intervalinf.py` — 6 tests

**Key Functions/Classes Added:**
- `PrimalKKTSolver.__init__` — precomputes `P_mat`, `AV_inv_mat`, `A_B_u0`, `G_adj_AV_d`
- `PrimalKKTSolver._woodbury_solve` — abstract H-space Woodbury; M×M Cholesky only
- `PrimalKKTSolver._residuals` — abstract KKT residuals via `inner_product`
- `PrimalKKTSolver.solve` — two-branch fsolve driver, warm-start, abstract throughout

**Test Coverage:**
- Total tests written: 24 (18 pygeoinf + 6 intervalinf)
- All tests passing: ✅
- Full pygeoinf suite: 819 passed, 1 xfailed — no regressions
- Full intervalinf suite: 489 passed — no regressions

**Key design correction vs plan:**
The plan incorrectly included $A_V$ in `P_mat` for ellipsoid V.
Correct derivation: `P_mat = (G @ A_B_inv @ G.adjoint).matrix(dense=True)` (no $A_V$);
$A_V$ enters only through `AV_inv_mat` in `K = (1/μ) A_V^{-1} + (1/λ) P_mat`.
The correction term uses plain `G.adjoint(z_D)`, not `G_adj_AV(z_D)`.

**Recommendations for Next Steps:**
- Use `PrimalKKTSolver` as the primary solver in any `Lebesgue`/`Sobolev` inversion pipeline
- The existing `PrimalKKTSolver` remains available as a numerical benchmark for Euclidean problems
