# Re-review N — recipes for the items that are easy to get wrong

Companion to `rereview_N.md`. Each recipe was executed in `work/review_r2/N/` against the current code; the numbers quoted are from those runs. Vectors are components `c` in the space's basis; `G` is the Gram matrix, reached only through `space.apply_gram`. The one rule that matters everywhere below: **an inner product is `c_x · (G c_y)`; never `c_x · c_y`** — every prototype here applies `G` exactly once per pair, and that is the whole difference between metric-correct and metric-wrong.

## 1. O1 — component path for Gram–Schmidt, Lanczos, `from_vectors`, `random_eig` (`p3b_lanczos_proto.py`, `p1_random_eig.py`, `p3_sphere.py`)

**Where it goes.** Add to `CoordinateSpace` (spaces.py) overrides of `_orthogonalise_against`, `orthonormal_basis`, `gram_schmidt`, and a new `inner_products(xs, ys)` (matrix of pairwise products); leave the `HilbertSpace` versions as the coordinate-free fallback. `iter_lanczos_tridiagonalise`, `random_eig`/`random_svd`, `from_vectors.adjoint` and the bundle `_gram` then branch on `isinstance(space, CoordinateSpace)`. Nothing in the algebra changes its contract.

**Gram–Schmidt on components** (19× at lmax 64; 23–30× on a dense metric at dim 3000):
```python
C = np.stack([space.to_components(v) for v in vectors], axis=1)   # (dim, k): k transforms, once
def gram_apply(Y): return np.stack([space.apply_gram(y) for y in Y.T], axis=1)  # diagonal spaces: metric_values[:, None] * Y
# Cholesky-QR, twice ("twice is enough" applies to CholQR as to MGS):
Q = C
for _ in range(2):
    R = np.linalg.cholesky(Q.T @ gram_apply(Q))       # k x k; fails only on numerical dependence
    Q = np.linalg.solve(R, Q.T).T                       # Q R^-T
basis = [space.from_components(q) for q in Q.T]         # k transforms, once
```
Check: `Q.T @ gram_apply(Q) - I` was 4e-15 on the dense metric. For the rank-revealing use (`orthonormal_basis` with dropping), use MGS on the arrays instead of CholQR — the `p3` prototype (`proto_orth`) does two MGS sweeps with `v -= (q @ (g*v)) * q` and drops when the residual norm falls below `rtol * original`; same 19×. `_orthogonalise_against(vector, basis)` for the adaptive `random_range` block: keep the basis as a `(dim, k)` array `Qc` alongside the vector list, and do `coef = Qc.T @ space.apply_gram(c); c -= Qc @ coef` (one `apply_gram`, two GEMVs) with the second pass under the existing threshold.

**Lanczos** (3.1–3.7× at lmax 64/128; 1146 → 152 transforms per 30 steps): keep `Q` as a list of component arrays; per step one `from_components` (to apply the operator) and one `to_components` (its result):
```python
w = space.to_components(operator(space.from_components(Q[-1])))
alpha = w @ g_apply(Q[-1]);  w -= alpha*Q[-1];  if step: w -= beta*Q[-2]
Gw = g_apply(w)                      # one metric application for the whole sweep …
for q in Q: w -= (q @ Gw) * q        # … is classical GS; for *modified* GS recompute Gw inside the loop
beta = sqrt(w @ g_apply(w))
```
The current code is modified GS with a full basis and no second pass. On components, classical GS with one `apply_gram` per step and a second pass when `beta < 0.5 * ||w_before||` (the same threshold `_orthogonalise_against` uses) is the standard choice and is what the prototype's numbers assume; modified GS on components costs k metric applications per step, which on a *diagonal* metric is still only O(k·dim) and is what `p3b` actually ran (result identical to v2 at 1e-14). Recombine once at the end: `space.from_components(norm * (np.stack(Q, axis=1) @ weights))`. Yield contract unchanged (`(basis, T)`), but let the iterator yield the component array too, or give `apply_operator_function` its own loop — the `list(basis)` copy per yield goes away with it.

**`random_eig` projection**: `T = Q.T @ gram_apply(AQ)` where `AQ` is the `(dim, k)` array of `to_components(operator(q))`; symmetrise; eigenvectors `U = Q @ S` (one GEMM instead of k² `axpy`s); `from_vectors` built from the columns of `U` and, on a `CoordinateSpace`, storing `U` so that `adjoint(y) = U.T @ apply_gram(to_components(y))` (one transform, not k). `random_svd`: the same with `pulled` as an array and `C = P.T @ gram_apply_domain(P)`.

**Tests to add (rule 2, rule 3).** Every function above on `make_dense_metric_space(40)` against the closed form (generalised eigenproblem `S v = λ G v` for `random_eig`, orthonormality `Uᵀ G U = I`), and a transform-count test on `Sphere` (patch `pyshtools.expand.SHExpandDH` as `common.Counts.patch_sh` does): `apply_operator_function` with `max_iterations=k` must call it `≤ 2k + operator's own` times, `orthonormal_basis(k)` exactly `k`. Fix B4 first or the dense fixture is unusable above dim 300.

## 2. O4 — the level bundle master as a k-variable dual (`p7_recipes.py`, verified to 8e-15)

Primal (convex.py:1336–1369), with cuts `(g_j, f_j, x_j)` and centre `c`:
```
minimise  ½‖x − c‖²   subject to   f_j + (g_j, x − x_j) ≤ level   for all j
```
Dual, over `λ ≥ 0` (no simplex constraint — the level is a fixed number, not a free `t`):
```
minimise  ½ λᵀ Q λ + qᵀ λ,   Q_ij = (g_i, g_j),   q_j = level − f_j + (g_j, x_j − c)
x = c − Σ_j λ_j g_j
```
`Q` is the Gram matrix `ProximalBundleMethod._gram` already keeps incrementally; `q` needs k inner products per iteration. Infeasibility of the level (the case `_master` returns `None` for) shows up as an **unbounded** dual (OSQP status `dual infeasible`), so keep the `alpha`-widening logic keyed on that status. The proximal fallback (`level=None`, `t` free) is exactly the proximal method's simplex problem with `weight=1`: `w = _minimise_on_simplex(Q, −e)` with `e_j = max_i m_i − m_j`, `m_j = f_j + (g_j, c − x_j)`, and `x = c − Σ w_j g_j` (verified 3e-14). Cost measured: 2 ms at 5000 data against 125 ms and 401 MB for the dense primal; at 10⁵ data the primal cannot be formed at all. The LP lower bound (`_lower_bound`) stays as it is — its dual still has `dim` equality rows — but cache `to_components(g_j)` with the cut instead of re-transforming in `_cut_rows` on every call. This makes `LevelBundleMethod` coordinate-free except for the LP, which is worth saying in its docstring.

## 3. O3 — the proximal bundle subproblem (`p7`, second script; `p2c_osqp_simplex.py`)

Do **not** warm-start FISTA: measured 957 vs 976 iterations, no gain — on `cond(Q) ≈ 1e13` it hits the cap from any start. Hand the simplex QP to OSQP with polishing:
```python
A = vstack([ones(1,k), I_k]);  l = [1, 0…0];  u = [1, ∞…∞]
OSQP.setup(P=csc(Q), q=−linear, A, l, u, eps_abs=eps_rel=1e-6 … 1e-8, polishing=True)
w = clip(x, 0, None); w /= w.sum()
```
On 20 bundles of 40 near-parallel cuts: OSQP 0.9 ms and KKT residual 4e-16 (polishing makes it exact) against FISTA 13.8 ms and 2.6e-5; Clarabel 0.7 ms but 4e-3 (its interior point stops at its tolerance — do not use it here). Keep FISTA as the fallback when `osqp` is absent (`best_available_qp_solver` already orders them), and keep the existing residual warning for the fallback. `eps=1e-10` costs 50× more per call (the earlier 55 ms) and buys nothing after polishing. Expected on the (300, 60) run: 5.4 s → ≈ 0.5 s in the subproblem, and fewer null steps (61 → 42 outer iterations measured at 1e-10).

## 4. O5 — `monotone_root` (`p4_root.py`)

Keep the decade bracketing (and its `exhausted`/`breakdown` semantics) exactly as it is; replace the geometric bisection loop (root_find.py:276–295) by `scipy.optimize.brentq(lambda u: sign * probe(exp(u)) − goal, log(low), log(high), xtol=…)` with `xtol` chosen so that `|Δt|/t ≤ rtol` (`xtol = 2·rtol` in `u`), the warm start still carried through `probe`'s `previous`. Measured 23–28 → 10–13 solves per root. `converged` = Brent's `converged` flag; the bracket in the result is `(exp(u−xtol), exp(u+xtol))`.

## 5. B1 — what to do about the dual route's false convergence

Not a numerics fix. The proximal gap (convex.py:866–876) is a stopping heuristic; on the (300 model, 2000 data, tight noise) problem it says 1e-10 at a value 2× the truth, and 3000 iterations still leave 5 %. What *does* work there is Chambolle–Pock run to a feasibility residual (`ChambollePockSolver(iterations=20000, tolerance=1e-9)`: 0.181539, agreeing with a feasible SLSQP point to 1e-6, 5.3 s). Two defensible defaults for `DualFeasibleProperty.support_values`: (a) run the primal to `tolerance` with a large cap and report `converged`; (b) keep the dual but return the level method's certified `upper − lower` alongside, refusing `converged=True` unless that gap is below tolerance — which needs O4 first, since the level method as it stands does not converge on this problem. This is Mag's API and needs his view.
