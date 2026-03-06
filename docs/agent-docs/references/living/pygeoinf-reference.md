# pygeoinf — Package Reference

**Version:** see `pygeoinf/pyproject.toml`
**License:** see `pygeoinf/LICENSE`
**Requires:** Python ≥ 3.10, numpy, scipy

---

## Overview

`pygeoinf` is a **Hilbert-space-first library** for geophysical inversion and Bayesian inference. It separates three concerns cleanly:

1. **Abstract mathematical structure** — spaces, operators, measures (base classes + protocols)
2. **Numerical implementations** — matrix backends, randomized decompositions, iterative solvers
3. **Problem orchestration** — forward problems and inversion algorithms that accept any concrete space/operator

The core design goal is that all inversion algorithms are written against abstract interfaces (`HilbertSpace`, `LinearOperator`, `GaussianMeasure`) and work automatically with any concrete space — including `intervalinf`'s `Lebesgue` and `Sobolev` spaces.

## Documentation

- **Technical manual (LaTeX):** `pygeoinf/theory/TECHNICAL_MANUAL.tex`.
    Chapter 2 (Hilbert spaces) includes the coefficient-space induced inner product
    $(\cdot,\cdot)_\pi$ and discusses the adjoint $\pi^{*_{\ell^2}}$ of the synthesis
    map with respect to the standard $\ell^2$ inner product, alongside the distinction
    between Hilbert adjoints and Banach duals in the API. The Gram/mass operator is
    $\mathbf{M}_\phi = \pi^{*_{\ell^2}}\pi$.
    The manual also includes “notation ↔ code” boxes mapping $\Pi,\pi,\pi^{*_{\ell^2}}$
    to `HilbertSpace.coordinate_projection`, `HilbertSpace.coordinate_inclusion`, and
    `LinearOperator.adjoint` / `LinearOperator.dual`.
    The discussion is structured so the infinite-dimensional coefficient-space viewpoint comes first (bases/representations and Gram operator), with truncation and finite-dimensional computation deferred to the later “Finite-dimensional approximation” section.

---

## Architecture

```
pygeoinf/pygeoinf/
├── hilbert_space.py       ← HilbertSpace ABC, EuclideanSpace,
│                             MassWeightedHilbertSpace, HilbertSpaceDirectSum
├── linear_forms.py        ← LinearForm ABC, DualVector
├── linear_operators.py    ← LinearOperator, DenseMatrixLinearOperator, …
├── nonlinear_operators.py ← NonLinearOperator
├── nonlinear_forms.py     ← NonLinearForm
│
├── gaussian_measure.py    ← GaussianMeasure (prior/posterior representations)
├── forward_problem.py     ← ForwardProblem, LinearForwardProblem
├── inversion.py           ← Inversion, LinearInversion, Inference (ABCs)
│
├── linear_bayesian.py     ← LinearBayesianInversion (+ constrained)
├── linear_optimisation.py ← LinearLeastSquaresInversion,
│                             LinearMinimumNormInversion (+ constrained)
├── linear_solvers.py      ← LinearSolver ABC + iterative solvers
├── backus_gilbert.py      ← Backus-Gilbert inversion
│
├── subsets.py             ← Subset ABC, Ball, Ellipsoid, HalfSpace,
│                             PolyhedralSet, EmptySet, FullSpace
├── subspaces.py           ← OrthogonalProjector, LinearSubspace, AffineSubspace
├── convex_analysis.py     ← SupportFunction hierarchy + combinators
├── convex_optimisation.py ← Convex optimisation algorithms
│
├── plot.py                ← SubspaceSlicePlotter, plot_slice (1D/2D/3D slices); plot_1d_distributions, plot_corner_distributions
│
├── auxiliary.py           ← Miscellaneous mathematical helpers
├── direct_sum.py          ← DirectSumLinearOperator, DirectSumLinearForm
├── preconditioners.py     ← Preconditioner operators for iterative solvers
├── random_matrix.py       ← Randomized range finder, SVD, Cholesky
├── parallel.py            ← joblib-based parallel compute helpers
└── utils.py               ← General utilities
```

**Dependency layering** (no circular imports):
```
hilbert_space + linear_forms + linear_operators  (foundation layer)
         ↓
gaussian_measure + subsets + subspaces + convex_analysis  (geometry layer)
         ↓
forward_problem + inversion + linear_bayesian + linear_optimisation  (algorithm layer)
         ↓
visualization + backus_gilbert  (output layer)
```

---

## Module Reference

### `hilbert_space.py` — Core Space Abstractions

#### `HilbertSpace` (ABC)

The abstract base class for ALL spaces in pygeoinf and its extensions (including `intervalinf`).

**Abstract methods** — subclasses must implement the core interface:

| Method | Signature | Description |
|---|---|---|
| `dim` | `@property → int` | Finite approximation dimension |
| `to_dual` | `(x) → LinearForm` | Riesz map: $x \mapsto \langle x, \cdot\rangle$ |
| `from_dual` | `(xp) → Vector` | Inverse Riesz map |
| `to_components` | `(x) → ndarray` | Project onto basis → coefficient array |
| `from_components` | `(c) → Vector` | Reconstruct from coefficient array |
| `zero` | `@property → Vector` | Additive identity |
| `basis_vector` | `(i) → Vector` | $i$-th orthonormal basis vector |

**Concrete methods** — inherited for free by any subclass:

| Method | Description |
|---|---|
| `inner_product(x, y)` | Default: $\langle x, y \rangle = \langle \texttt{to_dual}(x), y\rangle$ (duality pairing) |
| `norm(x)` | $\|x\| = \sqrt{\langle x, x\rangle}$ |
| `distance(x, y)` | $\|x - y\|$ |
| `axpy(a, x, y)` | In-place $y \leftarrow ax + y$ |
| `ax(a, x)` | In-place $x \leftarrow ax$ |
| `copy(x)` | Deep copy of vector |
| `add(x, y)` | $x + y$ (new vector) |
| `subtract(x, y)` | $x - y$ (new vector) |
| `multiply(a, x)` | $ax$ (new vector) |
| `gram_schmidt(vs)` | Strict Gram-Schmidt (raises `ValueError` on linear dependence) |

---

#### `EuclideanSpace(HilbertSpace)`

Concrete $n$-dimensional Euclidean space. Vectors are `np.ndarray` of shape `(n,)`.

**Constructor:** `EuclideanSpace(n)`

Inner product: $\langle x, y \rangle = x^T y$. Riesz map is the identity (self-dual).

---

#### `MassWeightedHilbertSpace(HilbertSpace)`

Abstract base for spaces with a mass-weighted inner product:

$$\langle u, v \rangle_M = \langle M u, v \rangle_H$$

where $M$ is a bounded, positive-definite, self-adjoint operator on underlying space $H$.

**Constructor:** `MassWeightedHilbertSpace(underlying_space, M_op, M_op_inv)`

Used by `intervalinf.Sobolev` to implement Sobolev spaces.

---

#### `HilbertSpaceDirectSum(HilbertSpace)`

Direct sum $H_1 \oplus H_2$ with product inner product.

**Constructor:** `HilbertSpaceDirectSum(space1, space2)`

---

### `linear_forms.py` — Linear Functionals

#### `LinearForm` (ABC)

Abstract linear functional $\ell: V \to \mathbb{R}$.

| Method | Description |
|---|---|
| `__call__(x)` | Evaluate $\ell(x)$ |
| `components` | `@property` — coefficient array $[\ell(\phi_0), \ldots, \ell(\phi_{n-1})]$ |

#### `DualVector(LinearForm)`

Concrete linear form from a coefficient array. `DualVector(space, components)`.

---

### `linear_operators.py` — Linear Operators

#### `LinearOperator`

A bounded linear map $A: V \to W$.

**Constructor:** `LinearOperator(domain, codomain, mapping_callable)`

**Factory:** `LinearOperator.from_matrix(domain, codomain, matrix)` — wraps a 2D numpy array.

**Key methods:**

| Method | Description |
|---|---|
| `__call__(x)` / `__matmul__(x)` | Apply operator: $y = Ax$ |
| `adjoint` | `@property` → adjoint $A^*: W \to V$ (computed via Riesz maps) |
| `compose(B)` | Composition: $A \circ B$ |
| `matrix()` | Build dense matrix (applies $A$ to each basis vector) |
| `domain`, `codomain` | Source and target spaces |

Operator algebra: `A + B`, `A @ B` (composition), `a * A`, `A.T` (adjoint shorthand).

---

### `subsets.py` — Convex Sets

All sets derive from `Subset(ABC)`:

| Method | Description |
|---|---|
| `is_element(x)` | Membership test → `bool` |
| `boundary` | The set's boundary (returns a `Subset`) |
| `plot(on_subspace=None, *, bounds=None, grid_size=200, rtol=1e-6, alpha=0.5, cmap="Blues", color="steelblue", show_plot=True, ax=None, backend="auto")` | Visualize via `plot_slice()`; auto-builds a default subspace for 1D/2D `EuclideanSpace`, requires an explicit subspace otherwise; `backend` is `"auto"`, `"matplotlib"`, or `"plotly"`; `"auto"` prefers Plotly for 3D when installed, warns and falls back to Matplotlib otherwise; 1D/2D always use Matplotlib |

**Concrete classes:**

| Class | Constructor | Mathematical set |
|---|---|---|
| `Ball` | `Ball(space, center, radius, open_set=True)` | $\{x : \|x - c\| < r\}$ by default; set `open_set=False` for the closed ball |
| `Ellipsoid` | `Ellipsoid(space, center, radius, operator)` | $\{x : \|A(x-c)\| \leq r\}$ |
| `HalfSpace` | `HalfSpace(space, normal, offset)` | $\{x : n^T x \leq \alpha\}$ |
| `PolyhedralSet` | `PolyhedralSet(space, halfspaces)` | Intersection of `HalfSpace` objects |
| `EmptySet` | `EmptySet(space)` | $\emptyset$ |
| `FullSpace` | `FullSpace(space)` | $V$ |

---

### `subspaces.py` — Affine/Linear Subspaces

#### `OrthogonalProjector`

Callable orthogonal projection operator onto a linear subspace.

**Factory:** `OrthogonalProjector.from_basis(space, [v1, v2, ...])` — builds $P = \sum_i \hat{v}_i \hat{v}_i^T$.

**Methods:** `__call__(x)` (projection $Px$), `complement(x)` ($(I-P)x$).

---

#### `LinearSubspace(Subset)`

A linear subspace $V \leq H$.

**Constructor:** `LinearSubspace(projector)`

---

#### `AffineSubspace(Subset)`

An affine subspace $x_0 + V$ defined by an orthogonal projector and translation.

**Constructor:** `AffineSubspace(projector, translation=None)`

**Factory methods:**

| Factory | Description |
|---|---|
| `from_tangent_basis(space, [v1,...], translation=None)` | Build from spanning vectors |
| `from_linear_equation(B, w)` | Kernel of linear map: $\{u : B(u) = w\}$ |

**Key methods:**

| Method | Description |
|---|---|
| `project(x)` | $P_A(x) = P(x - x_0) + x_0$ |
| `is_element(x, tol)` | Membership test |
| `get_tangent_basis()` | Orthonormal basis for tangent space (tolerant Gram-Schmidt) |
| `dimension` | `len(get_tangent_basis())` |
| `tangent_space` | The corresponding `LinearSubspace` |
| `domain` | Ambient `HilbertSpace` |
| `boundary` | Returns `EmptySet` |

**Important:** `get_tangent_basis()` uses two-phase tolerant Gram-Schmidt (fixed Feb 2026) — correctly reports dimension for non-axis-aligned subspaces. Do NOT replace with the strict `gram_schmidt()` from `HilbertSpace`.

---

### `convex_analysis.py` — Support Functions

A support function $h_C(u) = \sup_{x \in C} \langle x, u \rangle$ encodes a convex set $C$ dually.

#### `SupportFunction` (ABC)

| Method | Description |
|---|---|
| `evaluate(u)` | Compute $h_C(u)$ |
| `subgradient(u)` | A subgradient $\in \partial h_C(u)$ |
| `conjugate()` | Conjugate support function |

**Concrete implementations:**

| Class | Convex set $C$ |
|---|---|
| `BallSupportFunction` | $\{x: \|x\| \leq r\}$ |
| `EllipsoidSupportFunction` | $\{x: \|Ax\| \leq r\}$ |
| `PolyhedralSupportFunction` | Polyhedral set |
| `SobolevBallSupportFunction` | Ball in Sobolev (mass-weighted) norm |

**Combinators:**

| Class | Description |
|---|---|
| `InfimalConvolution(h1, h2)` | $(h_1 \square h_2)(u)$ |
| `SupportFunctionSum(h1, h2)` | $h_1(u) + h_2(u)$ |
| `SupportFunctionScaledSum(hs, ws)` | $\sum_i w_i h_i(u)$ |

---

### `gaussian_measure.py` — Gaussian Measures

#### `GaussianMeasure`

A Gaussian measure $\mathcal{N}(\mu, C)$ on a `HilbertSpace`.

**Constructor:** `GaussianMeasure(space, mean, covariance_operator)`

| Method | Description |
|---|---|
| `sample(n)` | Draw $n$ samples |
| `mean` | Mean $\mu \in H$ |
| `covariance_operator` | Covariance $C: H \to H$ |
| `push_forward(A)` | $A_\# \mathcal{N}(\mu, C) = \mathcal{N}(A\mu, ACA^*)$ |
| `marginals(A)` | Marginal distribution under linear map $A$ |
| `condition(A, y, error_measure)` | Bayesian update given $y = Ax + \varepsilon$ |

---

### `forward_problem.py` — Forward Problems

#### `ForwardProblem`

Bundles model space, data space, forward operator, and optional data error measure.

**Constructor:** `ForwardProblem(model_space, data_space, forward_operator, data_error_measure=None)`

#### `LinearForwardProblem(ForwardProblem)`

Specialised for `LinearOperator` forward operators. Used by all linear inversion classes.

---

### `inversion.py` — Inversion Abstractions

| Class | Description |
|---|---|
| `Inversion(ABC)` | Base interface: `.invert(data) → solution` |
| `LinearInversion(Inversion)` | For linear forward problems |
| `Inference` | Wrapper adding uncertainty quantification |

---

### `linear_bayesian.py` — Bayesian Inversion

#### `LinearBayesianInversion(LinearInversion)`

Computes the posterior Gaussian measure from a Gaussian prior and linear observations.

**Constructor:** `LinearBayesianInversion(forward_problem, prior)`

**Method:** `.invert(data) → GaussianMeasure`

**Algorithm** (Kalman gain):
$$\mu_{\text{post}} = \mu_{\text{prior}} + C_{\text{prior}} A^* (A C_{\text{prior}} A^* + C_\varepsilon)^{-1}(y - A\mu_{\text{prior}})$$

#### `ConstrainedLinearBayesianInversion`

Adds an `AffineSubspace` constraint $Bu = w$ to the inversion.

---

### `linear_optimisation.py` — Deterministic Inversion

| Class | Solves |
|---|---|
| `LinearLeastSquaresInversion` | $\min_u \| Au - y \|^2$ |
| `LinearMinimumNormInversion` | $\min \|u\|$ subject to $Au = y$ |
| Constrained variants | Add `AffineSubspace` equality constraint |

#### `LinearMinimumNormInversion.minimum_norm_operator` — Discrepancy-Principle Bisection

The discrepancy principle chooses the regularisation parameter $\alpha^*$ satisfying
$$\chi^2(u^\dagger(\alpha^*)) = \chi^2_\text{critical}$$
where $\chi^2_\text{critical}$ is set from a $\chi^2$ distribution at the chosen significance level.
The algorithm brackets $\alpha^*$ by halving/doubling from `damping = 1.0`, then bisects.

**Feasibility pre-condition:**
The problem has a solution only when the *chi-squared floor* — the residual variance
that persists even at $\alpha \to 0$ (i.e. the noise projected onto the null space of
$G^\top$) — is strictly below $\chi^2_\text{critical}$.  In expectation the floor equals
the number of rays minus the rank of the forward operator.  If the forward-model grid is
too coarse relative to the ray coverage the floor can exceed the critical value, making
$\alpha^*$ non-existent.

**Bugs fixed (2026-03):**

| Bug | Location | Symptom | Root cause | Fix |
|---|---|---|---|---|
| **Silent false bracket** | halving/doubling loops after `while` | `RuntimeError: Bracketing search failed to converge` even when the halving loop ran to `maxiter` without chi-squared ever crossing critical | `damping_lower = damping` set unconditionally — when the loop exhausted `maxiter` without a crossing, a meaningless value `≈ 1/2^{100}` was placed in `damping_lower` and bisection started on a completely unbracketed interval | Added `if chi_squared > critical_value: raise RuntimeError(…)` guard after each loop |
| **Degenerate convergence criterion** | bisection stopping test | bisection exhausts all `maxiter` iterations when crossover $\alpha^* \lesssim 10^{-6}$ | Test was `width < atol + rtol*(lower+upper)`.  When `lower → 0` this collapses to `width < rtol*upper`, e.g. for $\alpha^*=10^{-8}$: threshold $\approx 10^{-14}$ while width $\approx 5\times10^{-9}$ — never satisfied with default `atol=0.0` | Changed to `width < atol + rtol*upper` (scale on the larger endpoint) |

**Client-side feasibility guard (recommended):**
Before calling `minimum_norm_operator`, estimate the chi-squared floor by evaluating the
forward model at near-zero damping (or via a pseudoinverse), then re-scale the noise
standard deviation upward so that the floor falls below the critical value.  See
`manuel_solution/seismic_tomo/inversion.py → run_minimum_norm_inversion` for a worked
example: `_estimate_chi2_floor` / `_scale_noise_measure` utilities implement this pattern.

---

### `plot.py` — Set Visualization

#### `SubspaceSlicePlotter`

Slice any `Subset` along a 1D, 2D, or 3D `AffineSubspace` for visualization.

**Constructor:** `SubspaceSlicePlotter(subset, on_subspace, *, grid_size=200, rtol=1e-6, alpha=0.5, bar_pixel_height=6)`

**Method:** `.plot(bounds, cmap="Blues", color="steelblue", show_plot=True, ax=None, backend="auto") → (fig, ax, payload)`

`backend`: `"auto"` (default) | `"matplotlib"` | `"plotly"`. `"auto"` prefers Plotly for 3D when installed, warns and falls back to Matplotlib otherwise; 1D/2D always use Matplotlib.

| Subspace dimension | `payload` type | Method |
|---|---|---|
| 1D | boolean mask `(grid_size,)` or `[lo, hi]` interval (PolyhedralSet) | Bar plot |
| 2D | boolean mask `(grid_size, grid_size)` or `(n_verts, 2)` polygon vertices (PolyhedralSet) | Filled region |
| 3D | boolean mask `(grid_size, grid_size, grid_size)` (oracle path) or `(n_verts, 3)` vertex array (PolyhedralSet exact path) | Matplotlib: `mplot3d` voxels / Poly3D; Plotly: `go.Isosurface` (sampled) / `go.Mesh3d` (PolyhedralSet) |

**Two rendering paths:**
- `PolyhedralSet` → exact affine slice via `scipy.spatial.HalfspaceIntersection` + convex hull → payload is vertex array (1D, 2D, and 3D); **never uses the grid** (no oracle calls).
- All other sets → raster membership-oracle sampling on a `grid_size^n` grid → payload is boolean mask (1D, 2D, and 3D). 3D mask uses `indexing='ij'` so `mask[i,j,k]` = membership at `(u[i], v[j], w[k])`. 3D with Matplotlib uses `Axes3D.voxels()` with parameter-coordinate edge arrays (not raw voxel indices); 3D with Plotly uses `go.Isosurface` (sampled) or `go.Mesh3d` (PolyhedralSet). `UserWarning` emitted when `grid_size > 30` for non-`PolyhedralSet` 3D sets.

#### `plot_slice()` — Convenience Wrapper

```python
plot_slice(subset, on_subspace, bounds=None, grid_size=200, rtol=1e-6, alpha=0.5,
           cmap="Blues", color="steelblue", show_plot=True, ax=None, backend="auto")
    → (fig, ax, payload)
```

Thin wrapper over `SubspaceSlicePlotter`. Supports **1D, 2D, and 3D subspaces**.
For 3D with Matplotlib backend, `ax` is an `Axes3D` instance; with `backend='plotly'` (or `"auto"` when Plotly is installed), `ax` is `None`.
Exported from both `pygeoinf.plot` and the top-level `pygeoinf` namespace.

---

### Other Modules

| Module | Key exports | Description |
|---|---|---|
| `backus_gilbert.py` | `BackusGilbertInversion` | Classic Backus-Gilbert averaging kernels |
| `convex_optimisation.py` | `SubgradientDescent`, `Cut`, `Bundle`, `QPSolver`, `QPResult`, `SciPyQPSolver`, `OSQPQPSolver`, `ClarabelQPSolver`, `best_available_qp_solver`, `BundleResult`, `ProximalBundleMethod`, `LevelBundleMethod`, `ChambollePockResult`, `ChambollePockSolver`, `solve_primal_feasibility`, `solve_support_values`, `SmoothedDualMaster`, `SmoothingScheduleSolver` | Non-smooth convex optimisation — subgradient, bundle-method infrastructure, proximal bundle solver, level bundle solver; optional OSQP/Clarabel QP backends; Chambolle-Pock primal-dual solver for dual master primal feasibility form |
| `direct_sum.py` | `DirectSumLinearOperator`, `DirectSumLinearForm` | Block operators on direct sum spaces |
| `linear_solvers.py` | `LinearSolver(ABC)` + CG/MINRES | Abstract solver + iterative implementations |
| `preconditioners.py` | Various preconditioner classes | Used with iterative solvers |
| `random_matrix.py` | Randomized range finder, SVD, Cholesky | Low-rank approximations |
| `parallel.py` | `parallel_map` etc. | joblib-based parallelism helpers |
| `auxiliary.py` | Mathematical helper functions | Miscellaneous utilities |

#### `convex_optimisation.py` — Convex Optimisation (non-smooth)

**Phase 1 — Core Bundle Infrastructure** (implemented 2026-03-04)

| Class | Description |
|---|---|
| `SubgradientResult` | Result dataclass for `SubgradientDescent` |
| `SubgradientDescent` | Constant-step subgradient descent |
| `Cut` | Dataclass: `x`, `f`, `g`, `iteration` — one linearisation |
| `Bundle` | Collection of `Cut` objects; builds QP constraint data |
| `QPResult` | Dataclass: `x`, `obj`, `status` |
| `QPSolver` | `@runtime_checkable Protocol` for `solve(P,q,A,l,u,x0)→QPResult` |
| `SciPyQPSolver` | SLSQP-backed implementation of `QPSolver` |
| `OSQPQPSolver` | ADMM-based implementation via `osqp` (optional); warm-start via `x0`; inf→1e30 substitution |
| `ClarabelQPSolver` | Interior-point implementation via `clarabel` (optional); converts `l≤Ax≤u` to ZeroCone/NonnegativeCone form; tolerance attrs `tol_gap_abs`/`tol_gap_rel` |
| `best_available_qp_solver()` | Factory: returns `OSQPQPSolver` > `ClarabelQPSolver` > `SciPyQPSolver` by availability |
| `BundleResult` | Result dataclass for bundle method solvers |

**`Bundle` key methods:**

| Method | Returns | Notes |
|---|---|---|
| `add_cut(cut)` | `None` | Appends a `Cut` |
| `upper_bound()` | `float` | $\min_j f_j$ — best evaluated value |
| `best_point()` | `Vector` | $x_j$ achieving `upper_bound()` |
| `lower_bound()` | `float` | Placeholder `-inf`; real value set by master QP in Phase 2–3 |
| `linearization_matrix(center, domain)` | `(A, b)` | $A \in \mathbb{R}^{m\times(d+1)}$, $b\in\mathbb{R}^m$; encodes $A[\lambda;t]\leq b$ |
| `compress(max_size)` | `None` | Keeps last `max_size` cuts |
| `__len__()` | `int` | Number of cuts |

**`SciPyQPSolver` standard form:** $\min \tfrac{1}{2}x^\top P x + q^\top x$ s.t. $l \leq Ax \leq u$.

**Tests:** `tests/test_bundle_core.py` — 10 tests, all passing.

**Phase 2 — Proximal Bundle Method** (implemented 2026-03-04)

| Class / Function | Description |
|---|---|
| `_get_value_and_subgradient(oracle, x)` | Duck-typed helper: calls `oracle.value_and_subgradient(x)` if available, else `(oracle(x), oracle.subgradient(x))` |
| `ProximalBundleMethod` | Proximal bundle for $\min f(\lambda)$; constructor `(oracle, /, *, rho0, rho_factor, tolerance, max_iterations, bundle_size, store_iterates, qp_solver)`; public `solve(x0) → BundleResult` |

**`ProximalBundleMethod` convergence logic:**
- Solves master QP with *t* variable extracted directly as `result.x[d]` (not derived from `qp_obj` — the SLSQP objective omits the constant `+(ρ/2)‖λ̂‖²`).
- Warm-start: `t = f_hat` (always feasible).
- Step classification: serious if `f_next < f_hat`; null otherwise.
- `f_low` reset to $-\infty$ on each **serious step** (avoids spurious convergence when the model is exact on linear pieces); updated by `max(f_low, t_opt)` on **null steps**.
- Gap check `f_hat - f_low ≤ tolerance` done only after updating `f_low` (i.e., after null steps only).

**`DualMasterCostFunction.value_and_subgradient`** (in `backus_gilbert.py`):
Shares `G*λ`, `hilbert_residual`, and support-point queries in one pass; returns `(f, g)`.

**Tests:** `tests/test_proximal_bundle.py` (6 tests), `tests/test_dual_master_cost.py` (1 test) — all passing.

**Phase 3 — Level Bundle Method** (implemented 2026-03-04)

| Class / Helper | Description |
|---|---|
| `LevelBundleMethod` | Level bundle for $\min f(\lambda)$; constructor `(oracle, /, *, alpha, tolerance, max_iterations, bundle_size, store_iterates, qp_solver)`; public `solve(x0) → BundleResult` |

**`LevelBundleMethod` algorithm:**
- Level: $f_{\text{lev}} = \alpha f_{\text{low}} + (1-\alpha) f_{\text{up}}$, default $\alpha = 0.1$.
- Master QP objective: $\tfrac{1}{2}\|\lambda-\hat{\lambda}\|^2$ (no penalty on $t$); decision var $z=[\lambda, t]$.
- Constraints: cut constraints **plus** level constraint $t \leq f_{\text{lev}}$.
- Lower bound: computed each iteration by solving the regularised LP $\min t + \tfrac{\varepsilon}{2}\|\lambda\|^2$ s.t. cuts (tiny $\varepsilon=10^{-8}$); `f_low = max(f_low, f_LP)`.
- Infeasibility recovery: if QP fails, widen alpha by ×1.5 (capped at 0.9) for up to 3 attempts; fallback to an emergency proximal step.
- Serious step: `f_next < f_hat` → update stability centre; null step: add cut only.

**Tests:** `tests/test_level_bundle.py` — 6 tests, all passing:
- `test_level_bundle_quadratic_1d` — minimises $\lambda^2+2\lambda$ to $\lambda^*=-1$
- `test_level_bundle_nonsmooth_1d` — minimises $|\lambda-0.5|$ to $\lambda^*=0.5$
- `test_level_bundle_gap_certificate` — converged gap ≤ 10×tolerance
- `test_level_bundle_infeasibility_recovery` — $\alpha=10^{-8}$, no crash
- `test_level_bundle_dual_master` — finite $f_{\text{best}}$ on `DualMasterCostFunction`
- `test_level_vs_proximal_agreement` — agrees with `ProximalBundleMethod` within 0.01

**Phase 7 — Chambolle-Pock Primal-Dual Solver** (implemented 2026-03-04)

Solves the **primal feasibility form** of the dual master:

$$h_U(q) = \max_{m \in B,\, v \in V} \langle T^*q, m\rangle \quad\text{s.t.}\quad Gm + v = \tilde{d}$$

via the first-order saddle-point algorithm of Chambolle & Pock (2011).

| Class / Function | Description |
|---|---|
| `ChambollePockResult` | Dataclass: `m` (model primal), `v` (data primal), `mu` (dual), `primal_dual_gap` (feasibility residual), `converged`, `num_iterations` |
| `ChambollePockSolver` | Solves $\max_{m\in B, v\in V}\langle c,m\rangle$ s.t. $Gm+v=\tilde{d}$ via Chambolle-Pock; constructor `(B, V, G, d_tilde, /, *, sigma, tau, theta, max_iterations, tolerance)` |
| `solve_primal_feasibility(cost, qs, cp_solver)` | Batch wrapper: computes $h_U(q_i)$ for each $q_i$ by calling `cp_solver.solve(T^*q_i)` and returning $\langle T^*q_i, m^*\rangle$ as `np.ndarray` |

**`ChambollePockSolver` key details:**
- Operator $K = [G;\; I_D]$: primal var $x=(m,v)$, dual var $\mu\in D$.
- Step-size auto-selection via 20-step power iteration on $G^\top G$: `G_norm_est = sqrt(dominant_eigenvalue)`, `K_norm = sqrt(G_norm_est² + 1) × 1.01`, `tau = sigma = 0.99 / K_norm`.
- Iterations: dual update $\mu \mathrel{+}= \sigma(G\bar m + \bar v - \tilde d)$; primal update $m \leftarrow \operatorname{proj}_B(m - \tau G^\top\mu + \tau c)$; $v \leftarrow \operatorname{proj}_V(v - \tau\mu)$; over-relaxation $\bar m = m + \theta\Delta m$.
- Projection support: `BallSupportFunction` only (ball projection $c + r(z-c)/\max(\|z-c\|,r)$); `EllipsoidSupportFunction` raises `NotImplementedError`.
- Convergence: feasibility residual $\|Gm+v-\tilde d\| < \text{tolerance}$.

**Tests:** `tests/test_chambolle_pock.py` — 5 tests, all passing:
- `test_chambolle_pock_feasibility` — $\|Gm+v-\tilde d\| \leq 10\times\text{tol}$
- `test_chambolle_pock_m_in_B` — $\|m - c_B\| \leq r_B + 10^{-3}$
- `test_chambolle_pock_v_in_V` — $\|v - c_V\| \leq r_V + 10^{-3}$
- `test_chambolle_pock_returns_result` — result type and finite attributes
- `test_solve_primal_feasibility_multiple_directions` — agrees with `ProximalBundleMethod` at rtol=0.15, atol=1e-2

---

## Key Patterns and Conventions

### Adding a new concrete HilbertSpace

1. Inherit from `HilbertSpace` (or `MassWeightedHilbertSpace` for Sobolev-type)
2. Implement all 8 abstract methods
3. All other methods (`norm`, `axpy`, `ax`, `copy`, `gram_schmidt`, …) free from base class

### Adding a new LinearOperator

```python
# Functional style (any callable)
op = LinearOperator(domain_space, codomain_space, lambda x: ...)

# Matrix-backed
op = LinearOperator.from_matrix(domain, codomain, numpy_2d_array)
```

### Passing intervalinf spaces to pygeoinf algorithms

`Lebesgue` and `Sobolev` inherit from `HilbertSpace` / `MassWeightedHilbertSpace` — they can be passed directly to any pygeoinf algorithm without modification.

### Docstring convention

```python
def method(self, x: np.ndarray) -> float:
    r"""
    Brief description.

    Computes $f(x) = \langle x, Ax \rangle$ where $A$ is positive definite.

    Args:
        x: Array of shape (n,).

    Returns:
        Scalar value $f(x)$.

    References:
        Author (Year), Title, Journal. doi:...
        theory.txt §3.2
    """
```

---

## Public API Summary

```python
from pygeoinf import (
    HilbertSpace,               # ABC for all spaces
    EuclideanSpace,             # R^n
    MassWeightedHilbertSpace,   # H^s-type spaces
    HilbertSpaceDirectSum,      # H1 ⊕ H2
    LinearForm,                 # Abstract linear functional
    LinearOperator,             # Bounded linear map
)
from pygeoinf.subsets import (Ball, Ellipsoid, HalfSpace, PolyhedralSet)
from pygeoinf.subspaces import (AffineSubspace, LinearSubspace, OrthogonalProjector)
from pygeoinf.convex_analysis import SupportFunction
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInversion
from pygeoinf.linear_optimisation import LinearLeastSquaresInversion
from pygeoinf.plot import SubspaceSlicePlotter, plot_slice
```

---

## Theory Documents

Located in `pygeoinf/theory/`:
- `theory.txt` (2672 lines, LaTeX) — master reference: dual master equation, support functions, Bayesian formulas
- `*.pdf` — 18 reference papers (Backus-Gilbert, Stuart, Al-Attar, bundle methods, etc.)
- `theory_papers_index.md` — paper catalog (if present)

The `Theory-Validator-subagent` reads these automatically when reviewing mathematical code. Cite specific sections in docstrings as `theory.txt §X.Y`.

---

## Plans (Selected)

Located in `pygeoinf/plans/`:
- `bundle-methods-optimizer-plan.md` — implementation plan for level bundle methods targeting dual master problems.
- `dual_master_implementation.md` — implementation + testing plan for `DualMasterCostFunction` and related tooling.
- `dual-master-fast-convex-optimisation-report.md` — algorithm/API proposals for faster, more robust minimisation of `DualMasterCostFunction` (bundle, proximal bundle, primal-dual, smoothing, warm-start).

---

## Test Patterns

```bash
cd pygeoinf && conda run -n inferences3 python -m pytest tests/ -v
```

| Pattern | Use |
|---|---|
| `np.testing.assert_allclose(a, b, rtol=1e-9)` | All floating-point comparisons |
| `np.testing.assert_array_equal(a, b)` | Exact arrays |
| `pytest.raises(ValueError)` | Expected exceptions |
| `np.random.seed(42)` | Reproducibility in stochastic tests |
