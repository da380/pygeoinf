## Plan: UndiscretizedPrimalKKTSolver

Add a new class `PrimalKKTSolver` alongside the **existing** `PrimalKKTSolver`
(which must NOT be touched — it serves as the numerical benchmark). The new solver is
mathematically identical but operates on abstract Hilbert-space vectors throughout:
no `to_components`, `from_components`, or `.matrix()` on the **model** space, ever.
The data space is explicitly finite-dimensional and is freely discretized.

---

## Mathematical Derivation (all four prior/data combinations)

### Setup

| Symbol | Meaning |
|--------|---------|
| H | Model Hilbert space (may be Lebesgue, Sobolev, or any HilbertSpace) |
| D | Data Hilbert space (finite-dimensional, EuclideanSpace or MassWeighted on it) |
| u ∈ H | Model vector |
| G : H → D | Forward operator (LinearOperator with known adjoint) |
| B | BallSupportFunction or EllipsoidSupportFunction for the model prior |
| V | BallSupportFunction or EllipsoidSupportFunction for the data error |
| c ∈ H | Objective direction (e.g. T* q) |
| u₀ ∈ H | Center of B (abstract model vector) |
| d̃ ∈ D | Observed data (abstract data vector) |
| η | Radius of B |
| r | Radius of V |

### KKT stationarity (all cases)

The optimality condition at the primal maximizer u* is (in primal model space H):

```
(λ A_B_op + μ G_adj_AV ∘ G) u* = c + λ A_B_op(u₀) + μ G_adj_AV(d̃)
```

where:

| Case | A_B_op | G_adj_AV |
|------|--------|----------|
| Ball B, Ball V | identity (u ↦ u) | G.adjoint |
| Ball B, Ellipsoid V | identity | G.adjoint ∘ V._A |
| Ellipsoid B, Ball V | B._A | G.adjoint |
| Ellipsoid B, Ellipsoid V | B._A | G.adjoint ∘ V._A |

All four are abstract operator applications — no components on H.

### Woodbury inversion (data-space M×M system)

Apply the Woodbury identity `(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}` with:
- A = λ·A_B_op (automorphism on H)
- U = G_adj_AV : D → H
- C = μ·I_D
- V_mat = G : H → D

Result:

```
u*(λ,μ) = w - (1/λ) A_B_inv_op( G_adj_AV( K(λ,μ)^{-1} · G(w) ) )
```

where:

```
w           = (1/λ) A_B_inv_op(rhs)            [H-vector, abstract]
rhs         = c + λ A_B_op(u₀) + μ G_adj_AV(d̃)  [H-vector, abstract, precompute parts]
K(λ,μ)      = (1/μ) I_M + (1/λ) P_mat          [M×M dense matrix, cheap per call]
P_mat       = (G ∘ A_B_inv_op ∘ G_adj_AV).matrix(dense=True)  [precomputed once!]
```

P_mat expands to:

| Case | P_mat |
|------|-------|
| Ball B, Ball V | `(G @ G.adjoint).matrix(dense=True)` |
| Ball B, Ellipsoid V | `(G @ (G.adjoint @ V._A)).matrix(dense=True)` |
| Ellipsoid B, Ball V | `(G @ B._A_inv @ G.adjoint).matrix(dense=True)` |
| Ellipsoid B, Ellipsoid V | `(G @ B._A_inv @ (G.adjoint @ V._A)).matrix(dense=True)` |

`.matrix(dense=True)` probes the **data space** only (M×M applications of the composed
operator to data-space basis vectors). No model-space coordinates touched.

### Residuals (both abstract)

```
ρ₁(λ,μ) = model_space.inner_product(u* - u₀, u* - u₀) - η²
ρ₂(λ,μ) = data_space.inner_product(A_V_op(Gu* - d̃), Gu* - d̃) - r²
```

where `A_V_op = identity` for ball V and `A_V_op = V._A` for ellipsoid V. Both are
abstract inner-product evaluations — completely basis-free even for the data side.

### Physics-based initial λ

| Case | λ_init |
|------|--------|
| Ball B | `model_space.norm(c) / η` |
| Ellipsoid B | `sqrt(model_space.inner_product(B._A_inv(c), c)) / η` |

Both purely abstract.

### What is precomputed at construction time

1. `P_mat` (M×M numpy array) — one operator probing, never repeated.
2. `A_B_u0 = A_B_op(u₀)` — one abstract H-vector, zero cost for ball (identity).
3. `G_adj_AV_d = G_adj_AV(d̃)` — one H-vector from a single operator application.
4. `cho_factor(P_mat)` is NOT precomputed because K depends on (λ,μ); factored per call.
5. Warm-start multipliers `(λ_prev, μ_prev)`.

---

## Implementation

### New class location

`pygeoinf/pygeoinf/convex_optimisation.py` — append after existing `PrimalKKTSolver`.
**Zero edits to the existing class.**

### Constructor signature

```python
class PrimalKKTSolver:
    def __init__(
        self,
        B: "SupportFunction",
        V: "SupportFunction",
        G: "LinearOperator",
        d_tilde: "Vector",
        /,
        *,
        dense_threshold: int = 5000,   # data-space dim above which Cholesky → CG
        fsolve_tol: float = 1e-10,
        fsolve_maxfev: int = 200,
    ) -> None:
```

Constructor body (no model-space components):
1. Validate types of B and V (same TypeError as today).
2. For ellipsoid B/V: require `_A` and `_A_inv` are set (same ValueError as today).
3. Build abstract callables `A_B_op`, `A_B_inv_op`, `G_adj_AV`, `A_V_op`.
4. Precompute `P_mat = (G @ A_B_inv_op_lop @ G_adj_AV_lop).matrix(dense=True)`.
5. Precompute `A_B_u0 = A_B_op(B._center)` (H-vector).
6. Precompute `G_adj_AV_d = G_adj_AV(d_tilde)` (H-vector).
7. Store η, r, M, G, d_tilde, model_space, data_space, warm-start state.

### `_woodbury_solve(lam, mu, c)` → Vector

```python
def _woodbury_solve(self, lam: float, mu: float, c: "Vector") -> "Vector":
    ms = self._model_space
    ds = self._data_space
    # rhs (abstract H-vector)
    rhs = ms.add(c, ms.add(ms.multiply(lam, self._A_B_u0),
                            ms.multiply(mu, self._G_adj_AV_d)))
    # w = (1/λ) A_B_inv_op(rhs)
    w = ms.multiply(1.0 / lam, self._A_B_inv_op(rhs))
    if mu == 0.0:
        return w
    # p = G(w)  →  data-space vector
    p_vec = self._G(w)
    p_comps = ds.to_components(p_vec)
    # K = (1/μ) I_M + (1/λ) P_mat
    K_mat = (1.0 / mu) * np.eye(self._M) + (1.0 / lam) * self._P_mat
    cho = cho_factor(K_mat)
    z_comps = cho_solve(cho, p_comps)
    z_vec = ds.from_components(z_comps)
    # correction = A_B_inv_op( G_adj_AV(z_vec) )
    correction = self._A_B_inv_op(self._G_adj_AV(z_vec))
    return ms.subtract(w, ms.multiply(1.0 / lam, correction))
```

### `_residuals(lam, mu, c)` → (float, float)

```python
def _residuals(self, lam: float, mu: float, c: "Vector") -> tuple:
    ms = self._model_space
    ds = self._data_space
    u = self._woodbury_solve(lam, mu, c)
    diff_u = ms.subtract(u, self._u0)
    rho1 = ms.inner_product(diff_u, diff_u) - self._eta ** 2
    res_d = ds.subtract(self._G(u), self._d_tilde)
    rho2 = ds.inner_product(self._A_V_op(res_d), res_d) - self._r ** 2
    return rho1, rho2
```

### `solve(c)` → KKTResult

Identical logic to `PrimalKKTSolver.solve()`:
- **Branch 1:** `u_ball = B.value_and_support_point(c)` → if data constraint not active, return.
- Physics-based λ_init: `model_space.norm(c) / eta` (ball) or inner-product form (ellipsoid).
- Warm-start.
- `fsolve` in log space with clamp and fallback guesses.
- Update warm start on success.
- Return `KKTResult(m=u_star, ...)`.

---

## Phases

### Phase 1: Core implementation and Euclidean regression tests

- **Objective:** Implement `PrimalKKTSolver` (all four B/V combinations) and verify numerical parity against the existing dense solver on Euclidean fixtures.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/pygeoinf/convex_optimisation.py` — add the new class (append only).
  - `pygeoinf/pygeoinf/__init__.py` — export `PrimalKKTSolver`.
  - `pygeoinf/tests/test_primal_kkt_solver_basis_free.py` — new test file.
- **Tests to Write:**
  - `test_woodbury_ball_ball_euclidean_matches_dense` — at fixed (λ, μ), compare `_woodbury_solve` output against existing solver.
  - `test_woodbury_ball_ellipsoid_euclidean_matches_dense` — same for ellipsoid V.
  - `test_woodbury_ellipsoid_ball_euclidean_matches_dense` — ellipsoid B.
  - `test_woodbury_ellipsoid_ellipsoid_euclidean_matches_dense` — both ellipsoid.
  - `test_solve_branch1_euclidean` — prior-only branch returns B.support_point(c).
  - `test_solve_branch2_ball_ball_euclidean_matches_dense` — full solve, compare u* against dense.
  - `test_solve_branch2_ball_ball_mass_weighted_matches_dense` — MassWeightedHilbertSpace model.
  - `test_residuals_zero_at_solution` — verify ρ₁ ≈ ρ₂ ≈ 0 at returned u*.
  - `test_warm_start_updates` — two consecutive solve calls update _lambda_prev, _mu_prev.
- **Steps:**
  1. Write all tests (red).
  2. Implement the class stub (constructor + stubs raising NotImplementedError).
  3. Implement `_woodbury_solve` first — run Woodbury tests (green).
  4. Implement `_residuals` — run residual test (green).
  5. Implement `solve` — run full solve tests (green).
  6. Run full `pygeoinf/tests/` suite to confirm no regressions to existing solver.

### Phase 2: Monkey-patch guard — proof of basis-freedom

- **Objective:** Prove that the model space `to_components` and `from_components` are never called during a full `solve`.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/tests/test_primal_kkt_solver_basis_free.py` — add tests.
- **Tests to Write:**
  - `test_no_model_to_components_called_ball_ball` — monkey-patch `EuclideanSpace.to_components` and `from_components` on the model space to raise `AssertionError`; verify `solve(c)` still completes without raising.
  - `test_no_model_to_components_called_mass_weighted` — same on a `MassWeightedHilbertSpace` model.
- **Steps:**
  1. Write tests (they may unexpectedly pass if Phase 1 is correctly implemented; force fail first by inserting a `to_components` call, observe the raise, remove it).
  2. Run (green = fully basis-free; red = still touching model components somewhere).
  3. Fix any offending call identified in step 2.

### Phase 3: intervalinf integration tests

- **Objective:** Run the solver end-to-end with `Lebesgue` model space + `SOLAOperator` from intervalinf, where `to_components` is genuinely expensive and semantically wrong to call.
- **Files/Functions to Modify/Create:**
  - `intervalinf/tests/operators/test_primal_kkt_intervalinf.py` — new test file.
- **Tests to Write:**
  - `test_kkt_lebesgue_sola_branch1` — Lebesgue([0,1]) + SOLAOperator with loose data ball → prior-only branch; verify `model_space.inner_product(u* - B.support_point(c), u* - B.support_point(c)) < 1e-10`.
  - `test_kkt_lebesgue_sola_branch2_ball_ball` — tight data ball; verify ρ₁ ≈ ρ₂ ≈ 0 at returned u*.
  - `test_kkt_lebesgue_sola_branch2_ellipsoid_data` — EllipsoidSupportFunction for V (Euclidean data space with A_V ≠ I).
  - `test_kkt_sobolev_sola_branch2` — Sobolev (MassWeightedHilbertSpace) model + SOLAOperator; same validation.
  - `test_no_model_to_components_lebesgue` — monkey-patch `Lebesgue.to_components` to raise; full `solve` succeeds.
- **Steps:**
  1. Write tests (red until Phase 1-2 complete).
  2. Run under `conda activate inferences3` from `intervalinf/` directory.
  3. Fix any import issues (ensure `PrimalKKTSolver` is accessible from `pygeoinf`).

### Phase 4: Export, docs, final regression

- **Objective:** Ensure public surface is clean, docstrings are correct, and the full test suites for both packages pass.
- **Files/Functions to Modify/Create:**
  - `pygeoinf/pygeoinf/__init__.py` — confirm export.
  - `pygeoinf/pygeoinf/convex_optimisation.py` — add full Google-style docstring to `PrimalKKTSolver` with LaTeX math (ball/ball case shown explicitly).
- **Tests to Write:**
  - `test_importable_from_pygeoinf` for `PrimalKKTSolver`.
- **Steps:**
  1. Run `cd pygeoinf && python -m pytest tests/ -x -q` — must be all green.
  2. Run `cd intervalinf && python -m pytest tests/ -x -q` — must be all green.
  3. Write docstring with math for the ball/ball case (reference KKT + Woodbury).

---

## Open Questions

1. **Dense threshold for K:** Currently Cholesky for all M ≤ dense_threshold. Should it fall back to CG (via `ScipyIterativeSolver`) for large M? Initial scope: Cholesky only; CG path deferred.
2. **Naming:** `PrimalKKTSolver` vs `PrimalKKTSolverAbstract` vs `PrimalKKTSolverV2`. Decision: `PrimalKKTSolver` is most self-documenting.
3. **EllipsoidSupportFunction V: is `V._A` self-adjoint w.r.t. D?** The plan assumes yes (SPD shape operator). The existing dense solver makes the same assumption. No change needed.
4. **Precomputing `cho_factor(P_mat)` per call vs. caching per (λ,μ)?** K depends on (λ,μ) so the factorization must be done per call. P_mat itself is constant and cached.
