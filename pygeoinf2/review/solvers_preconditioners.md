# Review: linear solvers and preconditioners, v1 → v2

> **Note (2026-08-27):** the decisions recorded in `pygeoinf2/REVIEW.md` §11 (D-1 … D-13) override the Must/Should/Consider ranking below wherever they conflict — in particular D-1 (sphere vectors are `SHGrid`, `sampling=1` default), D-2 (points in `(lat, lon)` degrees), D-3 (per-geometry submodules with `Lebesgue`/`Sobolev` subclasses), D-4 (`from_matrix(..., form=)`), D-6 (parallel hooks around operators), D-12 (path *integral* operator), D-13 (convex solvers restored).

Scope: `pygeoinf/linear_solvers.py`, `pygeoinf/preconditioners.py`, the preconditioner/surrogate methods of `pygeoinf/linear_bayesian.py`, `linear_optimisation.py`, `symmetric_space.py` versus `pygeoinf2/numerics/solvers.py`, `numerics/preconditioners.py`, `inference/preconditioners.py`, `inference/normal.py`, `numerics/root_find.py`. All empirical claims below were checked with a throwaway script (results quoted inline); nothing in the repo was modified.

---

## 1. Functionality retained / extended / lost

### 1.1 Solvers

| v1 | v2 | Status |
|---|---|---|
| `CGSolver` (`linear_solvers.py:439`) | `CGSolver` (`solvers.py:508`) | Retained. Same recurrence; v2 skips the `A(0)` application when `x0 is None` (`solvers.py:495-502`). |
| `FCGSolver` (`:943`) | `FlexibleCGSolver` (`solvers.py:570`) | Retained but **not exported**: absent from `solvers.__all__` (`solvers.py:43-60`) and from `numerics/__init__.py`. Only reachable by importing the module directly. `WoodburyPreconditioner`'s docstring (`preconditioners.py:320`) tells users to use it. |
| `MinResSolver` (`:586`) | `MinResSolver` (`solvers.py:757`) | **Preconditioning lost.** v1 supported an SPD preconditioner (`:624, :664`); v2 raises `NotImplementedError` (`solvers.py:774-778`). |
| `BICGStabSolver` (`:705`) | `BiCGStabSolver` (`solvers.py:849`) | Retained. |
| `LSQRSolver` (`:816`) | `LSQRSolver` (`solvers.py:942`) | Retained as a `LeastSquaresSolver` sibling. **`x0` silently ignored** (`solvers.py:966-967`: `solve_fn(b, x0)` never passes `x0` to `_solve`). Verified: warm start from the exact solution still costs 28 iterations. No `atol`, no callback, no history. Damped-case residual tracking (`res2 += psi**2`, v1 `:923-925`) dropped, so `phi_bar` alone is used as the residual test (`solvers.py:1039`). |
| `ScipyIterativeSolver` + `CG/BICG/BICGStab/GMRESMatrixSolver` (`:325-436`) | dropped | See §2.3: measured, nothing lost for CG. `bicg` has no v2 counterpart (minor). |
| — | `GMRESSolver` (`solvers.py:639`) | **New**, left-preconditioned restarted GMRES, but also **not exported**. |
| `LUSolver`, `CholeskySolver`, `EigenSolver` (`:100-218`) | same (`solvers.py:295-336`) | Retained. Lost: `galerkin=` choice (now fixed per class via `form` ClassVar), `parallel`/`n_jobs` matrix assembly, `check_finite`. |
| `ProgressCallback`, `SolutionTrackingCallback`, `ResidualTrackingCallback` (`:1016-1108`) | `callback=(iteration, residual)` + `SolveResult.history` | **Partially lost.** The callback no longer receives the iterate, so `SolutionTrackingCallback` cannot be rebuilt on top of it. Only CG/FlexibleCG/GMRES call `_record`; **MINRES, BiCGStab and LSQR never invoke the callback and return an empty history** (verified: 0 callback calls, `len(history)==0`), contradicting the `IterativeSolver.__init__` docstring (`solvers.py:376-380`). |
| `IterativeLinearSolver(preconditioning_method=)` + `__call__(preconditioner=)` | `IterativeSolver(preconditioner=LinearSolver|LinearOperator)`, `with_preconditioner`, `resolved_for`, `resolve_solver` factories | Retained and extended. |
| `iterations` property on solver | `SolveResult` from `InverseOperator.solve` | Retained (better). |
| v1 default `rtol=1e-5`, `maxiter=10*dim` | v2 default `rtol=1e-10`, `maxiter=max(2*dim,20)` | **Changed silently.** Measured on the same 40 000-dim system: v1 default 150 iterations, v2 default 305. Combined with `strict=True` this is what broke example 22 (DESIGN §27.1). |

### 1.2 Generic preconditioners

| v1 (`preconditioners.py`) | v2 (`numerics/preconditioners.py`) | Status |
|---|---|---|
| `Identity` (`:20`) | `IdentityPreconditioner` (`:41`) | Retained. Returns the input object itself, not a copy (`:46`) — see §5. |
| `Jacobi` (`:35`) — **default Hutchinson estimate**, `num_samples=20`, `method`, `rtol`, `block_size`, `parallel` | `JacobiPreconditioner` (`:51`) — exact only | **Stochastic option lost.** v2 always does `dim` exact applications (`:71`; verified 2000 applications for dim 2000). `numerics.randomised.random_diagonal` exists and is unused here. v1 also had an exact path (`num_samples=None`). |
| `Spectral` (`:81`) — `method`, `max_rank`, `power`, `rtol`, `block_size`, `parallel` | `SpectralPreconditioner` (`:84`) — `rank`, `damping`, `rng` | Retained, options lost. `random_eig` accepts `**kwargs` (`randomised.py:328-335`) but `SpectralPreconditioner` forwards none (`:122`). `damping` semantics changed: v1 uses `damping**2` (`:156`), v2 uses `damping` as the eigenvalue floor (`:134`) — undocumented. |
| `Iterative` (`:171`) | subsumed | Any `IterativeSolver` is a `LinearSolver` and can be passed as `preconditioner=`. Works, but must be built with `strict=False` or the inner `ConvergenceError` aborts the outer solve (`solvers.py:461-473`). Not documented anywhere; catalogue row (`V1_CATALOGUE.md:152`) still says "M5". |
| `Banded` (`:201`) — `incomplete` ILU, `parallel` | `BandedPreconditioner` (`:146`) — adds `probe="banded"` fast extraction | Retained; ILU option lost. |
| `ExactBlock` (`:286`) — column probing keeping only block rows (O(nnz) memory), **overlapping blocks allowed**, ILU | `BlockPreconditioner` (`:216`) | **Regressed.** Forms the full dense matrix (`:255`, `dim²` memory), requires an exact partition (`:249-254`), `np.linalg.inv` per block. |
| `ColumnThresholded` (`:408`) — sparse column-by-column, `parallel` | `ColumnThresholdedPreconditioner` (`:513`) — pattern symmetrised (good) | Retained, but forms the full dense Galerkin matrix (`:596`) where v1 kept only the sparse entries. |

### 1.3 Structure-aware / inversion-bound preconditioners

| v1 | v2 | Status |
|---|---|---|
| `LinearBayesianInversion.diagonal_normal_preconditioner` (`linear_bayesian.py:791`), `parallel` | `NormalDiagonalPreconditioner` (`inference/preconditioners.py:62`) | Retained, free-standing; `parallel` lost. |
| `sparse_localized_preconditioner` (`:897`), `parallel` | `LocalisedPreconditioner` (`:161`) | Retained; `parallel` lost. |
| `woodbury_data/model_preconditioner` (`:1019`, `:1082`; `linear_optimisation.py:322`, `:385`) | `WoodburyPreconditioner` (`numerics/preconditioners.py:275`) with `from_normal` | Retained, unified. Tikhonov `1/damping` scaling is absorbed via `TikhonovNormalOperator.prior_covariance = (1/t) I` (`tikhonov.py:188`). |
| `surrogate_inversion`, `surrogate_*_preconditioner` (`:1126-1290`) | `NormalOperator.surrogate` (`normal.py:310`) + `from_normal` | Retained, cleaner. |
| `distance_localized_preconditioner` (`symmetric_space.py:2725`) | `InvariantDistancePreconditioner` (`:347`); taper now default | Retained. |
| Handing an inversion a preconditioner built from its own normal operator | `solver=lambda normal: CGSolver().with_preconditioner(...)` via `resolve_solver` (`solvers.py:134`); `LinearGaussianInversion.__init__` (`gaussian.py:104-112`); `with_solver` (`:215`) | Works for `LinearGaussianInversion`. **Broken for the damping sweep** — see §5.1. |

Catalogue rows `V1_CATALOGUE.md:148,150-152` (Banded, ExactBlock, Spectral, Iterative) still read "M5" although those classes exist; the catalogue is stale.

---

## 2. Algorithmic performance

### 2.1 Krylov loops
- **CG** (`solvers.py:540-565`): 1 operator application, 1 preconditioner application, 2 inner products + 1 norm per iteration; `axpy`/`scale_inplace` used, 2 fresh vectors per iteration (`operator(p)`, `preconditioner(r)`). Verified: 143 PCG iterations = 143 operator applications = 143 preconditioner applications. Equivalent to v1.
- **FlexibleCG** (`:607-634`): 2 extra allocations per iteration (`space.copy(r)` at `:617`, `space.subtract` at `:629`). `(r_new - r_old, z)` can be computed as `(r_new, z) - (r_old, z)` with one extra inner product and no allocation.
- **MINRES** (`:802-843`): `space.scale(1/beta_next, p)` at `:841` copies `p` although `p` is dead — one avoidable allocation per iteration.
- **GMRES** (`:697-745`): modified Gram-Schmidt, cost grows with column index as expected; restart recomputes the true residual (`:749`). Fine.
- **BiCGStab** (`:874-909`): 2 operator + 2 preconditioner applications per iteration, standard.
- Every preconditioner application allocates a frozen `SolveResult` dataclass (`InverseOperator._value`, `:231-232`). Negligible against a matvec, but it is per-iteration Python overhead.

### 2.2 `InverseOperator` re-runs the iteration per application
`InverseOperator._value` (`solvers.py:231-232`) calls `_solve_fn(y, None)` — a fresh Krylov run from zero every time. Verified: 5 applications of a CG inverse = 713 operator applications. Same as v1. DESIGN §27.5 documents the consequence (example 22: 30 s → 525 s when forming posterior covariance blocks) and the mitigation (`with_solver(CholeskySolver())`). There is no block/multi-RHS solve, no Krylov recycling, and `matrix()` on an iterative inverse runs `dim` independent solves. The library should at least say this on `InverseOperator`.

### 2.3 scipy vs hand-written CG, coordinate space
40 000-dim 5-point Laplacian + 0.01 I, `rtol=1e-8`, all three from the same `b`:

```
scipy.sparse.linalg.cg   244 it   0.046 s
v1 CGSolver              245 it   0.051 s
v2 CGSolver              244 it   0.044 s
```
scipy's `cg` is itself a Python loop over numpy since 1.12, so dropping `ScipyIterativeSolver` costs nothing for CG on numpy-backed spaces. The catalogue's worry is unfounded for CG; `bicg` is the only algorithm with no v2 equivalent.

### 2.4 Direct solvers
- Factor is computed once per `solver(operator)` and captured in the closure (`solvers.py:277-286`); each application is one triangular solve. Good.
- `matrix()` costs `dim` operator applications (`operators.py:449-514`, `by="auto"` picks the cheaper side). Same as v1, minus v1's `parallel`.
- **Adjoint of an LU inverse re-assembles and re-factorises** (`solvers.py:245-250` → `self._solver(self._operator.adjoint)`). Verified: 60 adjoint applications + a second `lu_factor` where v1 reused the factor with `lu_solve(..., trans=1)` (`linear_solvers.py:125-126`). Worse, `_adjoint_value` goes through a *second* cache `_adjoint_inverse_op` (`:237-243`), so `inv.adjoint(x)` and `inv._adjoint_value(x)` each build their own inverse — verified 120 adjoint applications after both paths. For self-adjoint operators `adjoint is self`, so the usual case is unaffected; for LU on a non-symmetric operator, and for any iterative inverse of a non-self-adjoint operator (the preconditioner is rebuilt for `A*`), it is a 2× cost.

### 2.5 `diagonals()` cost
`LinearOperator.diagonals(probe="exact")` (`operators.py:516-599`) is `dim` applications, each converted to components and Gram-weighted. There is **no override anywhere** (`grep "def diagonals"` finds one definition), so a diagonal error covariance `R` costs `dim` matvecs to extract its own diagonal. v1 had O(1) overrides (`linear_operators.py:1117`, `:1290`, `:1470`). This hits `NormalDiagonalPreconditioner` (`inference/preconditioners.py:146`), `LocalisedPreconditioner` (`:299`), `InvariantDistancePreconditioner` (`:431`) and `JacobiPreconditioner` — the diagonal of `R` costs as much as the diagonal of `A Q A*` was designed to save.

### 2.6 Woodbury and localised
- **Woodbury model form** (`preconditioners.py:454-465`): `prior - cross @ inverse @ cross.adjoint` applies `Q` three times per outer application (verified: Q=3). The first `Q y` and the `Q` inside `cross.adjoint = A Q` are the same product; computing `q = Q y` once and returning `q - Q A* N_d^{-1} A q` saves one `Q` per outer iteration. Same in `data_form` (`:467-485`) for `R^{-1}`. v1 had the same redundancy.
- **Woodbury inner solver default** is `CGSolver()` (`:438-441`): `rtol=1e-10`, `strict=True`. The class docstring says "the inner solve is cheap"; the default is a full-accuracy inner solve whose `ConvergenceError` will abort the outer solve. `data_form` with no precisions nests CG three deep (`:474-481`).
- **Localised** (`inference/preconditioners.py:243-287`): each block runs `random_eig` on `P (A Q A*) P*`, so roughly `nblocks × (rank + oversampling) × (1 + 2·power)` full applications of `A*`, `Q`, `A`. When `size <= rank` a randomised decomposition is still used (`:279`) where `size` exact probes would be cheaper and exact.

### 2.7 `DampedSolves`
`solve()` (`root_find.py:320-326`) creates a new `InverseOperator` each call; with a `DirectSolver` that is one factorisation per probe, including the repeated final probe at `monotone_root:250-252` and any repeat of the same multiplier. Operators are cached per multiplier (`:295-302`) but inverses are not.

---

## 3. Code practice / API

- **Too many routes, one of them silent.** Constructor `preconditioner=`, `with_preconditioner`, `resolved_for`, `resolve_solver` factories, `with_solver`, `WoodburyPreconditioner(...)` vs `.from_normal(...)`. Each has a rationale (DESIGN §28), but `with_preconditioner` **returns `self` unchanged when a preconditioner is already set** (`solvers.py:399-400`; verified) — a method named `with_X` that ignores its argument. Should raise or replace, and the "library keeps the caller's" policy should live at the call site that wants it.
- `SolverLike = "LinearSolver | Callable[...]"` (`solvers.py:130`) is a *string*, not a type alias; it cannot be used in annotations or checked.
- `IterativeSolver.__init__` docstring says `maxiter` "defaults to the dimension of the space" (`:371`); `_limit` uses `max(2*dim, 20)` (`:443-453`).
- `LinearSolver._validate` (`:106-123`) checks `domain.dim == codomain.dim` only, not `domain == codomain`; the Krylov loops then use `operator.domain` for the residual norm of a codomain vector (`:526-531`). Cheap to tighten.
- Reach-through: `InverseOperator._make_adjoint` writes `result.__dict__["_adjoint_cache"] = self` (`:249`); `NormalOperator.__repr__` inspects `self.__dict__` (`normal.py:299`). Two adjoint caches on `InverseOperator` (`_adjoint_cache`, `_adjoint_inverse_op`) is duplication; `adjoint_inverse` (`:237-243`) should just call `self.adjoint(x)`.
- Error handling: CG/FCG raise `ConvergenceError` on non-positive curvature (good); BiCGStab reports breakdown as `converged=False` (`:876-877`, `:889-890`) so the user sees "did not converge in N iterations", not "breakdown". MINRES/BiCGStab/LSQR have no `history`, so non-convergence gives no diagnostic trail.
- Trait strictness: `with_traits` is easy, but it returns `_RetraitedOperator` (`operators.py:386-391`, `:920-931`), which **loses the concrete type**. A `NormalOperator.with_traits(...)` is no longer a `FactoredNormalOperator`, so every structure-aware preconditioner refuses it (verified). See §5.1.
- Duplication: CG and FlexibleCG share ~40 lines verbatim (`:526-566` vs `:593-635`); Jacobi/NormalDiagonal/Localised/Invariant/ColumnThresholded each re-implement the same `apply_gram → solve → from_components` closure.
- `BandedPreconditioner`/`BlockPreconditioner` claim `Traits.SELF_ADJOINT` on the inverse regardless of `form="components"` on a non-self-adjoint operator (`preconditioners.py:213`, `:272`).
- `IdentityPreconditioner` returns the input object (`:46`); any solver that mutates `z` in place would corrupt `r`. Current solvers don't, but nothing enforces it.

---

## 4. Documentation gaps (public API)

`numerics/solvers.py`
- `SolveResult` (`:72-80`): fields undocumented. `residual_norm` means different things per solver (CG: recurrence residual in the space norm; GMRES: *preconditioned* residual; LSQR: normal-equation residual `:1038`); `history` only populated by three solvers.
- `InverseOperator.__init__` (`:195-215`): no docstring; nothing says each application re-runs the solve.
- `IterativeSolver.__init__` (`:367-381`): `maxiter` wrong (above); `rtol` is relative to `||b||` in the space norm — say "not `||b - A x0||`"; `callback` claim false for MINRES/BiCGStab.
- `LUSolver`/`CholeskySolver` (`:295-318`): no statement that construction costs `dim` operator applications and `dim²` memory, nor that the LU adjoint refactorises.
- `EigenSolver.__init__` (`:327`): `rtol` undocumented.
- `MinResSolver` (`:757-762`): should state up front that preconditioning is unsupported.
- `BiCGStabSolver` (`:850`): one line; no cost (2 applications/iteration), no breakdown behaviour.
- `LSQRSolver.__init__` (`:949-963`): **no docstring at all**; `x0` ignored is unstated; `rtol` applies to two different quantities (`:995-997`).
- `FlexibleCGSolver`, `GMRESSolver`: good docstrings, but unreachable via the package.

`numerics/preconditioners.py`
- `JacobiPreconditioner.__init__` (`:63`): `floor` undocumented; class doesn't state cost (`dim` applications).
- `SpectralPreconditioner` (`:107-113`): `damping` is an eigenvalue floor, not v1's squared damping — say so; no way to pass `power`/oversampling.
- `BlockPreconditioner`/`ColumnThresholdedPreconditioner`: should say they form the full dense matrix.
- `WoodburyPreconditioner.__init__` (`:341-357`): default inner solver's tolerance/strictness unstated.

`inference/preconditioners.py`
- `LocalisedPreconditioner` (`:176-180`): "overlapping contributions add, which is what makes an overlapping cover behave sensibly" is wrong (§5.4).
- `InvariantDistancePreconditioner`: no statement that the data space must be Euclidean/orthonormal (§5.3).

`numerics/root_find.py`
- `DampedSolves` (`:255-293`): `base`, `shift`, `solver`, `traits` fields undocumented; `solve` return annotated `Any` but returns `SolveResult`.

---

## 5. Correctness concerns

### 5.1 Structure-aware preconditioners are defeated by `with_traits` and by `DampedSolves` (Must)
`DampedSolves.operator()` builds `base + t*shift` then `.with_traits(...)` (`root_find.py:298-300`), producing a `_RetraitedOperator(_Sum)`. `_require_normal` (`inference/preconditioners.py:45-59`) demands a `FactoredNormalOperator`. Verified: `TikhonovFamily(A, error=err, solver=CGSolver().with_preconditioner(NormalDiagonalPreconditioner())).solve(1.0, b)` raises `TypeError`. So the DESIGN §24.3/§25.1 claim that "every structure-aware preconditioner applies to the point estimators" holds only for the factory route, which is resolved once against `t=1.0` (`tikhonov.py:337-346`) and never refreshed — the very thing the comment at `tikhonov.py:338-340` says is worse. Fix: have `DampedSolves` take an `assemble: Callable[[float], LinearOperator]` (e.g. `family.at`) instead of `base`/`shift`, and/or make `_RetraitedOperator` preserve `FactoredNormalOperator` (delegate `formalism/forward/prior_covariance/error_covariance` when the base has them).

### 5.2 `monotone_root` reports `converged=True` after exhausting its iterations (Must)
`root_find.py:250-252`: after the `for` loop ends without meeting the bracket tolerance, one more probe is taken and `finish(..., True)` is returned. Verified: `iterations=2, rtol=0` → `converged=True`, bracket width 1.38.

### 5.3 `InvariantDistancePreconditioner` is only right on an orthonormal data space (Should)
With point evaluation `A` into a data space with Gram `G_Y`, `(A Q A*)_c = K G_Y` and the Galerkin matrix is `G_Y K G_Y`, where `K_ij = k(d_ij)`. The class assembles `K + diag(R_gal)` (`inference/preconditioners.py:462-465`) and applies `K^{-1} G_Y c_y` (`:469`), which equals `(A Q A*)_c^{-1} c_y = G_Y^{-1} K^{-1} c_y` only when `G_Y = I`. It also mixes a Galerkin noise diagonal into a non-Galerkin `K`. Every other preconditioner in that file is careful about this; this one is not, and nothing documents the assumption. Per the metric-bug rule, test it on a `DiagonalMetricSpace` data space.

### 5.4 `LocalisedPreconditioner` double-counts overlaps (Should)
Overlapping blocks each contribute their full Nyström sub-block to the COO assembly, and COO duplicates *sum* (`:289-295`). An entry `(i,j)` in two blocks is counted twice. The docstring claims this is desirable (`:176-180`); it is not — it inflates the diagonal in overlaps by the multiplicity. Divide each entry by its multiplicity, or document that blocks should not overlap (v1 had the same behaviour but did not advertise overlap as a feature).

### 5.5 Metric handling that is right
- Jacobi/Banded/ColumnThresholded/NormalDiagonal/Localised: Galerkin diagonal/block, applied as `D^{-1} G c_y` — correct, and self-adjoint w.r.t. the space inner product.
- Cholesky/Eigen on the Galerkin matrix with `G_Y c_y` on the right-hand side (`solvers.py:282-284`) — correct.
- CG/FlexibleCG/MINRES/GMRES/BiCGStab/LSQR are written entirely in `inner_product`/`norm`/`axpy`, so they are the Hilbert-space algorithms; PCG's `(r, z)` pairing is correct provided the preconditioner is self-adjoint in the space inner product (all v2 preconditioners claim it).
- Woodbury is written in Hilbert adjoints; DESIGN §22.12 says it is tested on weighted spaces.
- FlexibleCG's Polak-Ribière `β = (r_{k+1} - r_k, z_{k+1})/(r_k, z_k)` (`:629-631`) is correct and reduces to Fletcher-Reeves for a fixed SPD preconditioner.

### 5.6 LSQR damped stopping (Consider)
Without v1's `res2` accumulation (`linear_solvers.py:923-925`), `abs(phi_bar) <= rtol*beta` (`solvers.py:1039`) tests only the reducible part of the damped residual, not `||A x - b||² + d²||x||²`. Combined by OR with the normal-residual test it will not give a wrong answer, but the reported `residual_norm` is the normal residual and there is no true residual in the result.

---

## Prioritised recommendations

### Must
1. **Export `FlexibleCGSolver` and `GMRESSolver`.** Add both names to `__all__` in `pygeoinf2/numerics/solvers.py:43-60` and to the import list and `__all__` in `pygeoinf2/numerics/__init__.py`.
2. **Make structure-aware preconditioners work inside the damping sweep.** In `pygeoinf2/numerics/root_find.py`, change `DampedSolves` to take a callable `assemble(multiplier) -> LinearOperator` (in `tikhonov.py:347-352` pass `self.at`) so the operator handed to a deferred preconditioner is a `TikhonovNormalOperator`, not `_RetraitedOperator(_Sum)`. Additionally, in `pygeoinf2/algebra/operators.py:920-931`, make `_RetraitedOperator` forward `formalism`, `forward`, `prior_covariance`, `error_covariance` to its base and register as a `FactoredNormalOperator` when the base is one, so `normal.with_traits(...)` does not silently disable them. Add a test: `TikhonovFamily(..., solver=CGSolver().with_preconditioner(NormalDiagonalPreconditioner())).solve(1.0, b)` must succeed.
3. **Fix `monotone_root`'s final return.** `pygeoinf2/numerics/root_find.py:250-252`: pass `converged=False` (or `high - low <= atol + rtol*(low+high)`) instead of `True`. Add a test with `iterations=1, rtol=0`.
4. **Make `callback`/`history` work in every iterative solver.** Add `self._record(iteration, residual, history)` calls and pass `tuple(history)` into every `SolveResult` in `MinResSolver._solve` (`solvers.py:802-846`) and `BiCGStabSolver._solve` (`:874-911`); add `callback`/`history` to `LSQRSolver` (`:949-1044`), or correct the `IterativeSolver.__init__` docstring (`:376-380`) and say which solvers support it.
5. **Honour `x0` in `LSQRSolver`.** `solvers.py:966-967`: pass `x0` into `_solve`, initialise `x = x0`, `u = b - A x0`; or document that LSQR cannot warm-start and drop `x0` from the signature.
6. **Restore MINRES preconditioning** (v1 `linear_solvers.py:609-702` is a working reference: preconditioned Lanczos with `y = M r`, `beta = sqrt((r, y))`), or make the class docstring (`solvers.py:757-762`) say it is unsupported rather than discovering it at solve time.
7. **Stop the dense-matrix regressions in preconditioners.** `BlockPreconditioner._invert` (`numerics/preconditioners.py:255`) and `ColumnThresholdedPreconditioner._invert` (`:596`) call `operator.matrix(...)`. Replace with column probing that keeps only the needed rows (v1 `pygeoinf/preconditioners.py:349-361`, `:467-508` show how), so memory is O(nnz). For Block, allow overlapping blocks as v1 did (`:335-344`).

### Should
8. **Give `JacobiPreconditioner` the stochastic option back.** Add `samples: int | None = None` (or v1's default 20) and use `numerics.randomised.random_diagonal` when set; document that exact costs `dim` applications.
9. **Add O(1) `diagonals()` for operators that know their diagonal.** Add an overridable hook on `LinearOperator` (e.g. `_diagonal_components()` returning `None` by default) and implement it for diagonal/matrix-backed operators (`from_component_matrix`/`from_derivative_matrix` in `operators.py:777-850`) so `error_covariance.diagonals(...)` in `inference/preconditioners.py:146, 299, 431` and Jacobi stop costing `dim` matvecs.
10. **Fix `InvariantDistancePreconditioner` for weighted data spaces** (`inference/preconditioners.py:455-474`): either assemble `G K G + diag(R_gal)` and solve that (Galerkin, consistent with the other classes), or `require` an `OrthonormalSpace` data space and say so in the docstring. Add a `DiagonalMetricSpace` test comparing against `Traits`-checked `A Q A*`.
11. **Fix or document overlap in `LocalisedPreconditioner`** (`:243-295`): divide assembled entries by their multiplicity (count per `(i,j)` pair), and rewrite `:176-180`.
12. **Stop re-factorising for the adjoint of a direct inverse.** In `DirectSolver._invert` (`solvers.py:274-288`), have `_factorise` return both `solve` and `solve_transposed` (`lu_solve(factor, c, trans=1)`, `cho_solve` is symmetric) and build the `InverseOperator` with an `adjoint_solve_fn`; make `InverseOperator._adjoint_value` use `self.adjoint` so there is one cache (`:237-250`).
13. **Change `with_preconditioner` to raise `ValueError` when a preconditioner is already set** (`solvers.py:399-400`), and give the library routines that want "keep the caller's" an explicit check instead.
14. **Loosen and de-strict the Woodbury inner solver default.** `numerics/preconditioners.py:438-441`: default to e.g. `CGSolver(rtol=1e-3, maxiter=…, strict=False)` and document that a `strict=True` inner solver aborts the outer solve.
15. **Reconsider `IterativeSolver` default `rtol=1e-10`** (`solvers.py:360`). v1 used `1e-5`; `1e-10` is often unreachable for ill-conditioned normal operators and, with `strict=True`, turns a usable answer into an exception. `1e-6`–`1e-8` plus `strict=True` is a safer default; whatever is chosen, record it in DESIGN with the measured iteration counts (150 vs 305 on the test problem).
16. **Fix the docstrings listed in §4**, in particular `IterativeSolver.__init__` `maxiter` (`:371`), `LSQRSolver.__init__` (`:949`), `SolveResult` field semantics (`:72-80`), `SpectralPreconditioner.damping` (`:110`), and add cost notes to the direct solvers and `InverseOperator`.
17. **Tighten `LinearSolver._validate`** (`solvers.py:107-113`) to require `operator.domain == operator.codomain`.
18. **Update `V1_CATALOGUE.md:148,150-152`** ("M5" rows) to point at the built classes and note `IterativePreconditioningMethod` is subsumed by passing an `IterativeSolver(strict=False)` as `preconditioner=`.

### Consider
19. Make `SolverLike` a real `TypeAlias` (`solvers.py:130`).
20. Share one `Q y` / `R^{-1} y` in `WoodburyPreconditioner.model_form`/`data_form` (`:454-485`) to save one covariance application per outer iteration; write the preconditioner as an explicit closure rather than an operator expression.
21. Remove the per-iteration allocations in `FlexibleCGSolver` (`:617, :629`: use `(r,z) - (r_old,z)` with `r_old` kept from the previous step) and `MinResSolver` (`:841`: `scale_inplace` on the dead `p`).
22. Forward `**kwargs` from `SpectralPreconditioner`/`LocalisedPreconditioner` to `random_eig` so power iterations and oversampling are reachable; in `LocalisedPreconditioner`, probe exactly when `size <= rank`.
23. Cache the `InverseOperator` per multiplier in `DampedSolves` (`root_find.py:320-326`) so a `DirectSolver` factorises once per distinct damping.
24. Add an `IterativeSolver` callback variant that receives the iterate (or a `track_solution=True` flag storing copies), restoring v1's `SolutionTrackingCallback`; add `ResidualTrackingCallback`-style exact residual as an option.
25. Have `IdentityPreconditioner` return `space.copy(y)` (`preconditioners.py:46`), or document the no-mutation contract solvers must honour for `z`.
26. Factor the duplicated `apply_gram → factorised.solve → from_components` closure used by Jacobi/Banded/Block/ColumnThresholded/NormalDiagonal/Localised/Invariant into one helper, and the shared CG/FlexibleCG loop body into a common method.
27. Add a `bicg` equivalent only if someone needs it; the measured parity with scipy (`244 it / 0.044 s` vs `0.046 s`) says the scipy wrapper is not missed for CG.
