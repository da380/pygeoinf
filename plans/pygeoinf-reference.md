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
├── visualization.py       ← SubspaceSlicePlotter (1D/2D/3D set slices)
├── plot.py                ← Lower-level plotting utilities
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

**Abstract methods** — subclasses must implement all 8:

| Method | Signature | Description |
|---|---|---|
| `dim` | `@property → int` | Finite approximation dimension |
| `inner_product` | `(x, y) → float` | $\langle x, y \rangle$ |
| `to_dual` | `(x) → LinearForm` | Riesz map: $x \mapsto \langle x, \cdot\rangle$ |
| `from_dual` | `(xp) → Vector` | Inverse Riesz map |
| `to_components` | `(x) → ndarray` | Project onto basis → coefficient array |
| `from_components` | `(c) → Vector` | Reconstruct from coefficient array |
| `zero` | `@property → Vector` | Additive identity |
| `basis_vector` | `(i) → Vector` | $i$-th orthonormal basis vector |

**Concrete methods** — inherited for free by any subclass:

| Method | Description |
|---|---|
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
| `project(x)` | Euclidean projection onto the set |
| `boundary` | The set's boundary (returns a `Subset`) |

**Concrete classes:**

| Class | Constructor | Mathematical set |
|---|---|---|
| `Ball` | `Ball(space, center, radius)` | $\{x : \|x - c\| \leq r\}$ |
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

---

### `visualization.py` — Set Visualization

#### `SubspaceSlicePlotter`

Visualise any `Subset` by slicing it with a 1D, 2D, or 3D `AffineSubspace`.

**Constructor:** `SubspaceSlicePlotter(subset, on_subspace, *, grid_size=50, alpha=1.0, cmap=None, color=None)`

**Method:** `.plot(bounds, show_plot=True, ax=None) → (fig, ax, result)`

| Subspace dimension | `result` type | Method |
|---|---|---|
| 1D | `(lo, hi)` interval | Bar plot |
| 2D | `(n_verts, 2)` polygon vertices | Filled polygon (exact for `PolyhedralSet`) |
| 3D | `(n_verts, 3)` vertices | 3D surface |

**Two rendering paths:**
- `PolyhedralSet` → Chebyshev center + `scipy.spatial.HalfspaceIntersection` + convex hull
- All other sets → raster membership-oracle sampling on a `grid_size × grid_size` grid

---

### Other Modules

| Module | Key exports | Description |
|---|---|---|
| `backus_gilbert.py` | `BackusGilbertInversion` | Classic Backus-Gilbert averaging kernels |
| `direct_sum.py` | `DirectSumLinearOperator`, `DirectSumLinearForm` | Block operators on direct sum spaces |
| `linear_solvers.py` | `LinearSolver(ABC)` + CG/MINRES | Abstract solver + iterative implementations |
| `preconditioners.py` | Various preconditioner classes | Used with iterative solvers |
| `random_matrix.py` | Randomized range finder, SVD, Cholesky | Low-rank approximations |
| `parallel.py` | `parallel_map` etc. | joblib-based parallelism helpers |
| `auxiliary.py` | Mathematical helper functions | Miscellaneous utilities |

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
from pygeoinf.visualization import SubspaceSlicePlotter
```

---

## Theory Documents

Located in `pygeoinf/theory/`:
- `theory.txt` (2672 lines, LaTeX) — master reference: dual master equation, support functions, Bayesian formulas
- `*.pdf` — 18 reference papers (Backus-Gilbert, Stuart, Al-Attar, bundle methods, etc.)
- `theory_papers_index.md` — paper catalog (if present)

The `Theory-Validator-subagent` reads these automatically when reviewing mathematical code. Cite specific sections in docstrings as `theory.txt §X.Y`.

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
