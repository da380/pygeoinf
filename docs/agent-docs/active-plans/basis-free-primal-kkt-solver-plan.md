## Plan: Basis-Free PrimalKKTSolver

Refactor `PrimalKKTSolver` so the **model space is touched only through abstract Hilbert / operator API** (`inner_product`, `to_dual`, `from_dual`, operator `@`, `.adjoint`) — no `to_components` / `from_components` / dense `.matrix()` calls on the model side. The 2-D `fsolve` over `(λ, μ)` is unchanged; only the inner KKT solve and norm evaluations are rewritten basis-free. Tested end-to-end on `intervalinf` (Lebesgue + SOLA) where the model space genuinely has no canonical basis.

### Design Note — what "basis-free" means here

The fundamental KKT system is

$$
(\lambda A_B + \mu G^* A_V G)\, u = c + \lambda A_B u_0 + \mu G^* A_V \tilde d,
$$

with residuals

$$
\rho_1(\lambda,\mu) = \|u - u_0\|_{A_B}^2 - \eta^2,\qquad
\rho_2(\lambda,\mu) = \|G u - \tilde d\|_{A_V}^2 - r^2.
$$

The rewritten solver only requires:
- `model_space.inner_product`, `model_space.to_dual`, `model_space.from_dual`
- An abstract `LinearOperator` for `A_B` and `A_B^{-1}` derived from `B`:
  - **Ball:** `A_B = model_space.to_dual`, `A_B^{-1} = model_space.from_dual` (the Riesz map). For a `MassWeightedHilbertSpace` these *are* the mass / inverse-mass actions, but accessed abstractly via `to_dual`/`from_dual`.
  - **Ellipsoid:** `A_B = B._A`, `A_B^{-1} = B._A_inv` (already `LinearOperator`s).
- `G` and `G.adjoint` (already abstract).
- The **data space stays in components** — it is finite-dimensional Euclidean / a small `EuclideanSpace`. Using `to_components` on the data side is *not* discretization.

The Woodbury system $K\,z = p$ lives in the data space ($M\times M$), so it is legitimately finite-dimensional and may use a dense Cholesky as before — no extra discretization tax there.

### Files / Components

- **New solver class:** `pygeoinf/pygeoinf/convex_optimisation.py` — add `PrimalKKTSolverMatrixFree` (or rename existing to `PrimalKKTSolverDense` and make `PrimalKKTSolver` a thin alias dispatching by space type — open question 1).
- **Public exports:** `pygeoinf/pygeoinf/__init__.py`.
- **Existing infra reused (no edits):**
  - [`pygeoinf/pygeoinf/linear_solvers.py`](pygeoinf/pygeoinf/linear_solvers.py) — `CGSolver`, `IterativeLinearSolver`.
  - [`pygeoinf/pygeoinf/linear_operators.py`](pygeoinf/pygeoinf/linear_operators.py) — `LinearOperator` algebra (`+`, scalar `*`, composition `@`, `.adjoint`).
  - [`pygeoinf/pygeoinf/convex_analysis.py`](pygeoinf/pygeoinf/convex_analysis.py) — `BallSupportFunction`, `EllipsoidSupportFunction`.
- **New tests (basis-free):**
  - `pygeoinf/tests/test_primal_kkt_solver_matrix_free.py` — Euclidean + `MassWeightedHilbertSpace` parity tests against the dense reference.
  - `intervalinf/tests/operators/test_primal_kkt_intervalinf.py` — Lebesgue / Sobolev + `SOLAOperator` integration tests.

### Phases

#### Phase 1: Abstract `(A_B, A_B_inv, A_V, A_V_inv)` extraction
- **Objective:** Replace the matrix-extracting block (lines ~2317-2409 of `convex_optimisation.py`) with a function that returns four `LinearOperator`s (or two pairs of operator-callables) without ever calling `.to_components` on the model space.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/pygeoinf/convex_optimisation.py` — new helper `_metric_operators_from_support_function(B, space)` returning `(A, A_inv)` for ball / ellipsoid.
- **Tests to Write:**
  - `test_metric_ops_ball_euclidean_match_riesz` — Ball on `EuclideanSpace`: `A_B(u)` equals `to_dual(u)`.
  - `test_metric_ops_ball_mass_weighted_match_riesz` — Ball on `MassWeightedHilbertSpace`: `A_B(u)` matches mass operator, `A_B_inv(ξ)` matches inverse-mass.
  - `test_metric_ops_ellipsoid` — Ellipsoid: `A_B == B._A`, `A_B_inv == B._A_inv`.
  - `test_metric_ops_reject_ellipsoid_without_inverse` — Same `ValueError` as today.
- **Steps:**
  1. Write the four tests against the helper signature.
  2. Run them (red).
  3. Implement `_metric_operators_from_support_function` returning callables backed by `to_dual`/`from_dual` for balls and `B._A`/`B._A_inv` for ellipsoids.
  4. Run tests (green).

#### Phase 2: Basis-free Woodbury solve in data space
- **Objective:** Reimplement `_woodbury_solve` so that, given `(λ, μ)`, it returns `u*` as a model-space `Vector` using only `A_B`, `A_B_inv`, `A_V`, `A_V_inv`, `G`, `G.adjoint`, and a dense `M × M` Cholesky on the data space.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/pygeoinf/convex_optimisation.py` — new `PrimalKKTSolverMatrixFree._woodbury_solve(lam, mu, c)` returning a model-space `Vector`. RHS built in the dual space via `to_dual`. Build `P = G ∘ A_B_inv ∘ G.adjoint` as a dense data-space matrix once (probing `M` standard basis vectors *of the data space*, which is finite-dim — no model-space discretization).
- **Tests to Write:**
  - `test_woodbury_matches_dense_reference_euclidean` — Random Euclidean fixture; compare against existing `PrimalKKTSolver` dense path at fixed `(λ, μ)`.
  - `test_woodbury_matches_dense_reference_mass_weighted` — Same on `MassWeightedHilbertSpace`.
  - `test_woodbury_special_case_mu_zero` — `μ=0` short-circuit returns `(1/λ) A_B_inv (rhs)` with no data-space solve.
- **Steps:**
  1. Write the three tests (call new method, parametrised by `(λ, μ)`).
  2. Run them (red).
  3. Implement `_woodbury_solve` using only abstract operator applications + a dense `cho_factor`/`cho_solve` on the data-space `K` matrix.
  4. Run tests (green).

#### Phase 3: Basis-free residuals and `solve()` driver
- **Objective:** Implement `_residuals(lam, mu, c)` and `solve(c)` end-to-end with the same fsolve/log-space/warm-start logic as the dense version, but no `to_components` on the model space.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/pygeoinf/convex_optimisation.py` — `PrimalKKTSolverMatrixFree._residuals` and `.solve`. Branch 1 (prior-only) reuses `B.value_and_support_point`. Branch 2 (both-active) calls `fsolve` on log-space residuals computed via abstract norms (`model_space.inner_product`, `data_space.inner_product`).
- **Tests to Write:**
  - `test_solve_branch1_prior_only_euclidean` — Loose data ball; result matches `B.support_point(c)`.
  - `test_solve_branch2_both_active_euclidean` — Tight data ball; result matches dense reference within `rtol=1e-8`.
  - `test_solve_branch2_both_active_mass_weighted` — Same on `MassWeightedHilbertSpace`.
  - `test_solve_warm_start_persistence` — Two consecutive `solve` calls update `_lambda_prev`, `_mu_prev`.
  - `test_solve_ellipsoid_prior` — `EllipsoidSupportFunction` for `B`.
  - `test_solve_ellipsoid_data` — `EllipsoidSupportFunction` for `V`.
- **Steps:**
  1. Write all six tests (red).
  2. Implement `_residuals`, `solve`, warm-start state, fallback guesses identical to dense version.
  3. Run tests (green).

#### Phase 4: Public surface, config, and integration with existing class
- **Objective:** Expose the new solver, decide its name and how it relates to the existing `PrimalKKTSolver`, and ensure the existing dense tests still pass.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/pygeoinf/__init__.py` — export `PrimalKKTSolverMatrixFree` (final name TBD per open question 1).
  - `pygeoinf/pygeoinf/convex_optimisation.py` — optional dispatch helper / docstring updates.
- **Tests to Write:**
  - `test_importable_from_pygeoinf` for the new symbol.
  - Re-run the existing `tests/test_primal_kkt_solver.py` to confirm no regression.
- **Steps:**
  1. Add export and import-test (red on import test until export added).
  2. Add export, run import test (green).
  3. Run full `pygeoinf/tests/` suite to confirm no regressions.

#### Phase 5: intervalinf integration tests (the real basis-free target)
- **Objective:** Demonstrate the solver on `intervalinf` where the model space is `Lebesgue` / `Sobolev` on an interval and `to_components` would be either expensive or undesirable. This is the acceptance test of the whole refactor.
- **Files/Functions to Modify/Create:**
  - `intervalinf/tests/operators/test_primal_kkt_intervalinf.py` — new test file.
- **Tests to Write:**
  - `test_kkt_lebesgue_sola_branch1` — Lebesgue interval space + `SOLAOperator`, loose data ball, prior-only branch. Compare returned `m` against `B.support_point(c)` evaluated in Lebesgue.
  - `test_kkt_lebesgue_sola_branch2` — Same setup with tight data ball; verify both KKT residuals are below tolerance after solve, computed via `model_space.inner_product` and `data_space.inner_product`.
  - `test_kkt_sobolev_sola_branch2` — Sobolev (mass-weighted) model space + `SOLAOperator`; same validation.
  - `test_kkt_no_to_components_called_on_model_space` — Monkey-patch `Lebesgue.to_components` / `from_components` to raise; verify a full `solve` call still succeeds (the proof that we are basis-free).
- **Steps:**
  1. Write the four tests (red on the to_components patch + on convergence checks until Phases 2-3 land).
  2. Run them (green if Phases 1-4 are correct).
  3. If `to_components` is unexpectedly called, trace the offending line and either lift it into the data side or refactor to use abstract operators.

### Open Questions

1. **Naming / dispatch:** Should the new class be `PrimalKKTSolverMatrixFree` (separate, explicit), or should we rename the current one to `PrimalKKTSolverDense` and have `PrimalKKTSolver` auto-dispatch by inspecting whether the model space exposes `squared_norms` / `dim`? Recommendation: keep both classes side-by-side, document the trade-off.
2. **Woodbury vs. matrix-free CG on M:** When `M` is large (rare in BG), should we offer a CG fallback for the data-space `K` solve too? Initial scope: dense Cholesky only; revisit if needed.
3. **Ellipsoid V `A_V_inv` requirement:** The Woodbury form needs `A_V^{-1}`. Should `EllipsoidSupportFunction` gain an `inverse_operator` requirement-message tweak, or do we just keep the existing `ValueError`?
4. **Should `PrimalKKTSolverMatrixFree` keep `dense_threshold` and switch to abstract CG (`CGSolver`) for the model-space solve when both `M` and the model space are large?** Initial scope: Woodbury-only; CG path can be a Phase 6 if needed.
5. **Where do intervalinf integration tests live** — `intervalinf/tests/operators/` (closer to SOLA) or `intervalinf/tests/spaces/`? Recommendation: `tests/operators/`.
