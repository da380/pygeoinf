# pygeoinf Explorer Report — Comprehensive Module Inventory

**Date:** 2026-02-27
**Purpose:** Full exploration of `pygeoinf` for planning a technical user manual.

---

## 1. Complete Module-by-Module Inventory

### Layer 1: Foundation — Spaces, Forms, and Operators

#### `hilbert_space.py` (848 lines)
**Mathematical concept:** Real Hilbert spaces, inner products, Riesz representation.

| Class | Key Methods | Description |
|---|---|---|
| `HilbertSpace` (ABC) | `dim`, `inner_product`, `to_dual`, `from_dual`, `to_components`, `from_components`, `zero`, `basis_vector` (abstract); `norm`, `distance`, `add`, `subtract`, `multiply`, `ax`, `axpy`, `copy`, `gram_schmidt`, `random`, `identity_operator`, `zero_operator`, `coordinate_inclusion`, `coordinate_projection`, `riesz`, `inverse_riesz`, `squared_norm`, `sample_expectation` (concrete) | Core ABC — all spaces inherit from this. Defines the contract: 8 abstract methods + rich concrete API. |
| `DualHilbertSpace` | Full `HilbertSpace` interface wrapping `LinearForm` objects | Wraps a primal space's dual; vectors are `LinearForm` instances. |
| `HilbertModule` (ABC) | `vector_multiply`, `vector_sqrt` | Extension for pointwise multiplication (Banach algebra). |
| `EuclideanSpace` | All 8 abstract methods; `subspace_projection` | Concrete $\mathbb{R}^n$; vectors are `np.ndarray`. Self-dual. |
| `MassWeightedHilbertSpace` | Delegates to underlying space + mass operator | $\langle u,v\rangle_M = \langle Mu,v\rangle_X$. Used by Sobolev spaces. |
| `MassWeightedHilbertModule` | Adds `vector_multiply`, `vector_sqrt` via delegation | Mass-weighted analog with pointwise multiplication. |

**Docstring quality:** Excellent. Module docstring + all methods documented with Google-style docstrings.

---

#### `linear_forms.py` (264 lines)
**Mathematical concept:** Linear functionals $\ell: V \to \mathbb{R}$, elements of dual spaces.

| Class | Key Methods | Description |
|---|---|---|
| `LinearForm` | `__call__`, `components`, `as_linear_operator`, `copy`; arithmetic (`+`, `-`, `*`, `/`, `+=`, `*=`), `from_linear_operator` (static) | Component-based linear functional. Gradient is constant (Riesz representative); Hessian is zero operator. |

**Docstring quality:** Excellent.

---

#### `nonlinear_forms.py` (317 lines)
**Mathematical concept:** General functionals $f: V \to \mathbb{R}$, gradients, Hessians, subgradients.

| Class | Key Methods | Description |
|---|---|---|
| `NonLinearForm` | `__call__`, `gradient`, `derivative`, `hessian`, `subgradient`; arithmetic (`+`, `-`, `*`) | Base for all functionals. Supports both smooth (gradient/Hessian) and non-smooth (subgradient) forms. |

**Docstring quality:** Good. Subgradient definition included.

---

#### `nonlinear_operators.py` (220 lines)
**Mathematical concept:** Nonlinear mappings $F: V \to W$, Fréchet derivatives.

| Class | Key Methods | Description |
|---|---|---|
| `NonLinearOperator` | `__call__`, `derivative`, `domain`, `codomain`; arithmetic (`+`, `-`, `*`, `@`) | General operator with optional Fréchet derivative. Chain rule for `@` composition. |

**Docstring quality:** Good.

---

#### `linear_operators.py` (1855 lines) — **Largest module**
**Mathematical concept:** Bounded linear maps, adjoints, duals, Galerkin forms, operator algebra.

| Class | Key Methods | Description |
|---|---|---|
| `LinearOperator` | `__call__`, `adjoint`, `dual`, `matrix()`, `compose`, `__add__`, `__sub__`, `__mul__`, `__matmul__` (`@`); Factories: `from_matrix`, `self_adjoint`, `self_dual`, `from_formal_adjoint`, `from_formally_self_adjoint`, `from_linear_forms`, `from_tensor_product`, `self_adjoint_from_tensor_product`, `self_adjoint_from_matrix`, `from_trace`; Decompositions: `random_svd`, `random_cholesky`, `random_eig`, `extract_diagonal` | Central workhorse. Supports Hilbert adjoint + Banach dual, standard and Galerkin matrix representations. |
| `MatrixLinearOperator` | `matrix()`, `galerkin_matrix`, internal SciPy interop | Wraps a SciPy `LinearOperator`; handles Galerkin vs standard form. |
| `DenseMatrixLinearOperator` | Dense matrix storage + operations | For dense `np.ndarray` matrices. |
| `SparseMatrixLinearOperator` | Sparse matrix storage | For `scipy.sparse` arrays. |
| `DiagonalSparseMatrixLinearOperator` | `inverse`, `sqrt`, `from_diagonal_values`, functional calculus | Diagonal operators with efficient functional calculus ($A^{-1}$, $A^{1/2}$). Key for covariance factors. |
| `NormalSumOperator` | Efficient $ACA^* + R$ computation | Specialized for Bayesian normal equations. |

**Docstring quality:** Excellent. Factory methods well-documented.

---

### Layer 2: Geometry — Sets, Subspaces, Measures, and Convex Analysis

#### `subsets.py` (1634 lines)
**Mathematical concept:** Subset hierarchy, sublevel sets, convex sets, CSG operations.

| Class | Key Methods | Mathematical Set |
|---|---|---|
| `Subset` (ABC) | `is_element`, `boundary`, `complement`, `intersect`, `union` | Abstract subset of $H$ |
| `EmptySet` | | $\emptyset$ |
| `UniversalSet` | | $H$ |
| `Complement` | | $S^c$ |
| `Intersection` | | $\bigcap_i S_i$ (generic) |
| `Union` | | $\bigcup_i S_i$ |
| `SublevelSet` | `form`, `level` | $\{x : f(x) \leq c\}$ |
| `LevelSet` | `form`, `level` | $\{x : f(x) = c\}$ |
| `ConvexSubset` | `support_function` | Convex sublevel set with dual support function |
| `ConvexIntersection` | Combines via max-functional | $\{x : \max(f_i(x)) \leq c\}$ |
| `Ellipsoid` | `center`, `radius`, `shape_operator`, `project` | $\{x : \langle A(x-c), x-c\rangle \leq r^2\}$ |
| `NormalisedEllipsoid` | Pre-normalised variant | Unit Ellipsoid |
| `EllipsoidSurface` | | $\{x : \langle A(x-c), x-c\rangle = r^2\}$ |
| `Ball` | `center`, `radius`, `project` | $\{x :\|x-c\| \leq r\}$ |
| `Sphere` | | $\{x : \|x-c\| = r\}$ |
| `HyperPlane` | `normal_vector`, `offset` | $\{x : \langle a,x\rangle = b\}$ |
| `HalfSpace` | `normal_vector`, `offset`, `inequality_type`, intersect logic | $\{x : \langle a,x\rangle \leq b\}$ or $\geq$ |
| `PolyhedralSet` | `halfspaces`, `chebyshev_center` | Intersection of half-spaces |

**Docstring quality:** Very good. Full CSG hierarchy documented.

---

#### `subspaces.py` (777 lines)
**Mathematical concept:** Orthogonal projections, affine subspaces, hyperplane-based subspaces, Gaussian conditioning.

| Class | Key Methods | Description |
|---|---|---|
| `OrthogonalProjector` | `__call__`, `complement`, `from_basis` | $P = P^* = P^2$, orthogonal projector onto span of basis vectors |
| `AffineSubspace` | `project`, `is_element`, `get_tangent_basis`, `condition_gaussian_measure`, `to_hyperplanes`; Factories: `from_linear_equation`, `from_tangent_basis`, `from_complement_basis`, `from_hyperplanes` | $x_0 + V$. Central for constraints. 4 factory methods with different constraint encodings. |
| `LinearSubspace` | `complement`, `from_kernel`, `from_basis`, `from_complement_basis` | Linear subspace (through origin) |

**Docstring quality:** Excellent. Tolerant Gram-Schmidt algorithm fully documented.

---

#### `convex_analysis.py` (347 lines)
**Mathematical concept:** Support functions $h_C(q) = \sup_{x \in C}\langle q,x\rangle$, subgradients.

| Class | Key Methods | Mathematical Object |
|---|---|---|
| `SupportFunction` (ABC) | `__call__` (evaluate $h_C$), `support_point`, `subgradient` | Abstract support function |
| `BallSupportFunction` | | $h(q) = \langle q,c\rangle + r\|q\|$ |
| `EllipsoidSupportFunction` | | $h(q) = \langle q,c\rangle + r\|A^{-1/2}q\|$ |
| `HalfSpaceSupportFunction` | | Extended-real-valued support of half-space. Returns $+\infty$ for non-parallel directions. |

**Docstring quality:** Good. Mathematical formulas included.

---

#### `gaussian_measure.py` (775 lines)
**Mathematical concept:** Gaussian measures $\mathcal{N}(\mu, C)$, affine transformations, Bayesian conditioning.

| Class | Key Methods | Description |
|---|---|---|
| `GaussianMeasure` | **Properties:** `domain`, `expectation`, `covariance`, `inverse_covariance`, `covariance_factor`, `sample_set`; **Sampling:** `sample`, `samples`, `sample_expectation`, `sample_pointwise_variance`, `sample_pointwise_std`; **Transformations:** `affine_mapping`, `as_multivariate_normal`, `low_rank_approximation`, `with_dense_covariance`, `two_point_covariance`; **Arithmetic:** `+`, `-`, `*`, `/`, neg; **Factories:** `from_standard_deviation`, `from_standard_deviations`, `from_covariance_matrix`, `from_samples`, `from_direct_sum` | Full Gaussian measure toolkit. Supports covariance factor-based sampling ($C = LL^*$). |

**Docstring quality:** Excellent. All factory methods documented.

---

### Layer 3: Problem Formulation and Algorithms

#### `forward_problem.py` (300 lines)
**Mathematical concept:** Forward model $d = A(u) + \varepsilon$.

| Class | Key Methods | Description |
|---|---|---|
| `ForwardProblem` | `forward_operator`, `data_error_measure`, `model_space`, `data_space` | General forward problem container |
| `LinearForwardProblem` | `from_direct_sum`, `data_measure`, `synthetic_data`, `chi_squared`, `critical_chi_squared` | Specialized for `LinearOperator`. Supports joint inversions via `from_direct_sum`. |

**Docstring quality:** Very good.

---

#### `inversion.py` (165 lines)
**Mathematical concept:** Abstract inversion/inference framework.

| Class | Key Methods | Description |
|---|---|---|
| `Inversion` | `forward_problem`, `model_space`, `data_space`, `assert_data_error_measure` | Base class for all inversion algorithms |
| `LinearInversion` | | Requires `LinearForwardProblem` |
| `Inference` | `property_operator`, `property_space` | Adds property estimation: estimate $T(u)$ from data |
| `LinearInference` | | Linear property + linear forward |

**Docstring quality:** Good.

---

#### `linear_bayesian.py` (300 lines)
**Mathematical concept:** Bayesian inversion — posterior $p(u|d)$ via Kalman gain.

| Class | Key Methods | Description |
|---|---|---|
| `LinearBayesianInversion` | `normal_operator`, `kalman_operator`, `model_posterior_measure` | $\mu_{post} = \mu_{prior} + CA^*(ACA^* + C_\varepsilon)^{-1}(d - A\mu_{prior})$. Full posterior as `GaussianMeasure`. |
| `ConstrainedLinearBayesianInversion` | `conditioned_prior_measure`, `model_posterior_measure` | Bayesian inversion + affine constraint $u \in A$. Two modes: conditioning vs. geometric projection. |

**Docstring quality:** Very good. Mathematical formula in class-level docs.

---

#### `linear_optimisation.py` (349 lines)
**Mathematical concept:** Deterministic inversion (Tikhonov, minimum norm, discrepancy principle).

| Class | Key Methods | Description |
|---|---|---|
| `LinearLeastSquaresInversion` | `normal_operator`, `normal_rhs`, `least_squares_operator` | Tikhonov: $\min_u \|Au-d\|^2 + \alpha^2\|u\|^2$ |
| `ConstrainedLinearLeastSquaresInversion` | `least_squares_operator` | + affine constraint |
| `LinearMinimumNormInversion` | `minimum_norm_operator` | $\min\|u\|$ s.t. $\chi^2 \leq c$. Bisection on damping $\alpha$. |
| `ConstrainedLinearMinimumNormInversion` | | + affine constraint |

**Docstring quality:** Good.

---

#### `linear_solvers.py` (957 lines)
**Mathematical concept:** Solving $Ax = b$ via direct and iterative methods.

| Class | Type | Description |
|---|---|---|
| `LinearSolver` (ABC) | — | Base interface |
| `DirectLinearSolver` | Direct | Base for factorization-based solvers |
| `LUSolver` | Direct | LU decomposition |
| `CholeskySolver` | Direct | Cholesky decomposition (SPD operators) |
| `EigenSolver` | Direct | Eigendecomposition; supports pseudo-inverse for singular systems |
| `IterativeLinearSolver` | Iterative | Base for matrix-free iterative solvers |
| `ScipyIterativeSolver` | Iterative | Wrapper for SciPy's matrix-based iterative solvers |
| `CGMatrixSolver`, `BICGMatrixSolver`, `BICGStabMatrixSolver`, `GMRESMatrixSolver` | Iterative (matrix) | Specialized scipy wrappers |
| `CGSolver` | Iterative (matrix-free) | Pure Hilbert-space Conjugate Gradient |
| `MinResSolver` | Iterative (matrix-free) | Minimum-residual for symmetric/singular systems |
| `BICGStabSolver` | Iterative (matrix-free) | BiCGStab for non-symmetric systems |
| `LSQRSolver` | Iterative | LSQR for least-squares problems |
| `FCGSolver` | Iterative (matrix-free) | Flexible CG (variable preconditioner per iteration) |

**Docstring quality:** Good.

---

#### `backus_gilbert.py` (314 lines)
**Mathematical concept:** Backus-Gilbert averaging kernels, dual master cost function.

| Class | Key Methods | Description |
|---|---|---|
| `HyperEllipsoid` | `space`, `radius`, `operator`, `centre`, `quadratic_form`, `is_point` | Model-space hyper-ellipsoid for prior constraints |
| `DualMasterCostFunction` | `__call__`, `set_direction`, `observed_data`, `direction` | $h_U(q) = \inf_\lambda\{\langle\lambda,\tilde{d}\rangle + \sigma_B(T^*q - G^*\lambda) + \sigma_V(-\lambda)\}$ |

**Docstring quality:** Moderate. `DualMasterCostFunction` has mathematical formula but light prose.

---

#### `convex_optimisation.py` (161 lines)
**Mathematical concept:** Subgradient descent for non-smooth convex minimization.

| Class | Key Methods | Description |
|---|---|---|
| `SubgradientResult` (dataclass) | `x_best`, `f_best`, `converged`, `function_values` | Result container |
| `SubgradientDescent` | `solve(x0)` | $x_{k+1} = x_k - \alpha g_k$ with constant step size. Learning/testing tool. |

**Docstring quality:** Good. Explicit about limitations (constant step size).

---

#### `nonlinear_optimisation.py` (219 lines)
**Mathematical concept:** Non-linear optimisation via SciPy, Wolfe line search.

| Class / Function | Key Methods | Description |
|---|---|---|
| `ScipyUnconstrainedOptimiser` | `minimize(form, x0)` | Wraps `scipy.optimize.minimize` for `NonLinearForm`. Supports Newton-CG, BFGS, L-BFGS-B, Powell, etc. |
| `line_search` (function) | | Wolfe line search wrapper for `NonLinearForm`. |

**Docstring quality:** Good.

---

### Layer 4: Infrastructure and Utilities

#### `direct_sum.py` (534 lines)
**Mathematical concept:** Direct sums $H_1 \oplus H_2 \oplus \cdots$, block operators.

| Class | Key Methods | Description |
|---|---|---|
| `HilbertSpaceDirectSum` | All `HilbertSpace` methods; `subspace_projection`, `subspace_inclusion`, `canonical_dual_isomorphism` | Vectors are `List[Any]`; inner product = sum of component inner products |
| `BlockStructure` (ABC) | `blocks` | Interface for block-structured operators |
| `BlockLinearOperator` | Full 2D block grid | General block operator between direct sums |
| `ColumnLinearOperator` | | Maps single space → direct sum (stacks operators vertically) |
| `RowLinearOperator` | | Maps direct sum → single space (applies operators horizontally) |
| `BlockDiagonalLinearOperator` | | Efficient diagonal-block operator |

**Docstring quality:** Good.

---

#### `preconditioners.py` (148 lines)
**Mathematical concept:** Preconditioning for iterative solvers.

| Class | Description |
|---|---|
| `IdentityPreconditioningMethod` | No-op (returns identity) |
| `JacobiPreconditioningMethod` | Diagonal preconditioning via Hutchinson's trace estimator or exact diagonal extraction |
| `SpectralPreconditioningMethod` | Low-rank spectral preconditioning using randomized eigendecomposition |
| `IterativePreconditioningMethod` | Uses an inner iterative solver as preconditioner (for FCG) |

**Docstring quality:** Moderate.

---

#### `random_matrix.py` (505 lines)
**Mathematical concept:** Randomized low-rank decompositions (Halko, Martinsson, Tropp 2011).

| Function | Description |
|---|---|
| `fixed_rank_random_range` | Fixed-rank approximation of matrix range |
| `variable_rank_random_range` | Adaptive-rank range approximation with tolerance |
| `random_range` | Dispatcher (fixed or variable) |
| `random_svd` | Randomized SVD |
| `random_eig` | Randomized eigendecomposition for symmetric operators |
| `random_cholesky` | Randomized Cholesky factorization |
| `random_diagonal` | Randomized diagonal estimation (Hutchinson's method) |

**Docstring quality:** Excellent. References to Halko et al. 2011.

---

#### `visualization.py` (1215 lines)
**Mathematical concept:** Plotting probability distributions and convex set slices.

| Class / Function | Description |
|---|---|
| `plot_1d_distributions` | Plot prior/posterior 1D Gaussians with dual y-axes |
| `plot_corner_distributions` | Corner plot (pairwise marginals) |
| `SubspaceSlicePlotter` | Slice arbitrary `Subset` along 1D, 2D, or 3D `AffineSubspace`. Two rendering paths: exact (polyhedral) and raster (membership oracle). |

**Docstring quality:** Good.

---

#### `auxiliary.py` (34 lines)
| Function | Description |
|---|---|
| `empirical_data_error_measure` | Generate data error covariance from model prior samples pushed through forward operator |

---

#### `utils.py` (13 lines)
| Function | Description |
|---|---|
| `configure_threading` | Controls BLAS/MKL thread count via `threadpoolctl` |

---

#### `parallel.py` (79 lines)
| Function | Description |
|---|---|
| `parallel_mat_mat` | Parallel matrix-matrix multiply via column-wise joblib |
| `parallel_compute_dense_matrix_from_scipy_op` | Parallel dense matrix construction from `ScipyLinOp` |

---

### Subdirectories

#### `checks/` — Self-testing mixins
- `hilbert_space.py` (202 lines): `HilbertSpaceAxiomChecks` — randomized axiom tests for any `HilbertSpace` (vector space, inner product, Riesz round-trip, Gram-Schmidt, in-place ops).
- `linear_operators.py` (198 lines): `LinearOperatorAxiomChecks` — linearity check, adjoint identity $\langle Ax,y\rangle = \langle x,A^*y\rangle$, algebraic identities.
- `nonlinear_operators.py` (198 lines): `NonLinearOperatorAxiomChecks` — finite-difference derivative check.

#### `data_assimilation/` — Standalone dynamical systems toolkit
- `core.py` (1496 lines): Dimension-agnostic ODE integration, probability grids, Kalman filters, ensemble methods, animation utilities. **Largely self-contained** — uses scipy, not the rest of pygeoinf.
- `pendulum/`: Example applications (pendulum dynamics).

#### `symmetric_space/` — Function spaces on symmetric manifolds
- `symmetric_space.py` (614 lines): `AbstractInvariantLebesgueSpace`, `AbstractInvariantSobolevSpace` — abstract framework using Laplace-Beltrami eigenvalues for invariant operators and isotropic Gaussian measures.
- `sphere.py` (1097 lines): `Lebesgue` (L²(S²)) and `Sobolev` (Hˢ(S²)) — concrete function spaces on the 2-sphere using `pyshtools`. Includes `cartopy`-based plotting. `SphereHelper` mixin.
- `circle.py`: Function spaces on the circle.
- `sh_tools.py`: Spherical harmonic vector conversion utilities.
- `wigner.py`: Wigner-D function utilities (rotational symmetry).

#### `testing_sets/` — Demos and test scripts
- Contains demo notebooks and test scripts for the dual master equation, polyhedral sets, and the admissible set.

---

## 2. User-Facing API Surface

What a geophysicist would import and use directly:

### Core Building Blocks
```python
from pygeoinf import (
    EuclideanSpace,                     # R^n
    HilbertSpaceDirectSum,              # H1 ⊕ H2
    LinearForm,                         # Dual vectors / measurement functionals
    LinearOperator,                     # Forward operators, adjoints
    GaussianMeasure,                    # Priors, posteriors, noise models
    ForwardProblem, LinearForwardProblem, # Problem setup
)
```

### Inversion Algorithms
```python
from pygeoinf import (
    LinearBayesianInversion,            # Full posterior p(u|d)
    ConstrainedLinearBayesianInversion,  # + affine constraint
    LinearLeastSquaresInversion,         # Tikhonov regularization
    LinearMinimumNormInversion,          # Discrepancy principle
    ConstrainedLinearLeastSquaresInversion,
    ConstrainedLinearMinimumNormInversion,
)
```

### Solvers & Preconditioners
```python
from pygeoinf import (
    CholeskySolver, LUSolver, EigenSolver,  # Direct
    CGSolver, MinResSolver, FCGSolver,       # Iterative (matrix-free)
    JacobiPreconditioningMethod,
    SpectralPreconditioningMethod,
)
```

### Geometry & Constraints
```python
from pygeoinf import (
    Ball, Ellipsoid, HalfSpace, PolyhedralSet,  # Convex sets
    AffineSubspace, LinearSubspace,               # Subspace constraints
    OrthogonalProjector,
    BallSupportFunction, EllipsoidSupportFunction, # Support functions
)
```

### Visualization
```python
from pygeoinf import (
    SubspaceSlicePlotter,               # Slice sets through subspaces
    plot_1d_distributions,              # Prior/posterior PDFs
    plot_corner_distributions,          # Corner plots
)
```

### Symmetric Spaces (optional — requires pyshtools+cartopy)
```python
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev  # S² function spaces
```

---

## 3. Internal/Helper API

| Module / Class | Role |
|---|---|
| `NonLinearOperator` | Base class — users rarely instantiate directly |
| `NonLinearForm` | Base class for all forms |
| `DualHilbertSpace` | Auto-created by `HilbertSpace.dual` |
| `MatrixLinearOperator`, `DenseMatrixLinearOperator`, `SparseMatrixLinearOperator` | Internal dispatch targets of `LinearOperator.from_matrix` |
| `DiagonalSparseMatrixLinearOperator` | Internal; exposed for covariance factor construction |
| `NormalSumOperator` | Internal optimization for $ACA^*$ sums |
| `checks/` (all mixins) | Inherited automatically — never called by users directly |
| `parallel.py` | Internal parallelism helpers |
| `random_matrix.py` functions | Called internally by `LinearOperator.random_svd()` etc. |
| `data_assimilation/core.py` | Self-contained toolkit, loosely coupled to core pygeoinf |
| `symmetric_space/sh_tools.py`, `wigner.py` | Internal utilities for sphere module |
| `Complement`, `Union`, `Intersection` | CSG combinators — created internally by `intersect()` / `union()` |
| `SublevelSet`, `LevelSet` | Base classes for convex set hierarchy |
| `BlockStructure` (ABC) | Interface for block operators |

---

## 4. Documentation Infrastructure Status

### Sphinx Setup
- **Location:** `pygeoinf/docs/`
- **Configuration:** `docs/source/conf.py` — fully configured with:
  - `sphinx.ext.autodoc` (auto-generate from docstrings)
  - `sphinx.ext.napoleon` (Google/NumPy docstring parsing)
  - `sphinx.ext.viewcode` (source links)
  - `nbsphinx` (Jupyter notebook integration)
  - Theme: **Furo** (modern, clean)
- **Source files:**
  - `index.rst` — complete landing page with overview and feature list
  - `modules.rst` → `pygeoinf.rst` — auto-generated API reference stubs
  - `pygeoinf.checks.rst`, `pygeoinf.symmetric_space.rst` — subpackage docs
  - `tutorials/` directory (empty or minimal)
  - `tutorials.rst` — toctree for tutorials
- **Requirements:** Full `docs/requirements.txt` (142 lines) with all dependencies pinned
- **State:** Infrastructure is complete but **no hand-written tutorials or theory explanation exist**. API docs are autodoc stubs only.

### Existing Documentation
- `README.md` — brief project description
- `CONTRIBUTING.md` — contribution guidelines
- `docs/theory_map.md` — may contain theory-to-code mapping
- `docs/theory_papers_index.md` — index of papers

---

## 5. Mathematical Topics in Theory Files

### `theory/theory.txt` (2672 lines) — **Main Theory Document**
Title: "DLI as Convex Analysis problems"

**Section structure:**
1. AI Statement
2. **Introduction** — Banach model space, linear forward maps $G$, property maps $T$, noisy observations
3. **Dual support characterization of the admissible property set** — the master dual equation $h_U(q) = \inf_\lambda\{\ldots\}$
   - Consequences
4. **Model-space constraints and their effect on the master dual equation**
   - General form of the master equation
   - Affine subspaces and linear constraints
   - Norm balls and gauge constraints
   - Pointwise inequality constraints
   - Intersections of convex constraints
   - Implications for computation
5. **Hilbert-space specialization of the master equation** — Inner products, Riesz maps
6. **Model-space constraints in the Hilbert setting**
   - Affine subspaces, norm balls, pointwise inequality, intersections
7. **Hilbert specialisation: L² model norm and Mahalanobis data set**
8. **Quadratic surrogate and its connection to BG estimators**
   - Closed-form surrogate certificates
   - Plug certificates, additive slack, geometric interpretation
   - Link to Backus-Gilbert and unconstrained SOLA
9. **Recovering the DLI ellipsoid** from dual support (noiseless case, with/without prior centre)
10. **Recovering the noiseless SOLA map from dual support**
11. **SOLA with unimodularity (equality constraints)**
12. **Polytope approximations of $\mathcal{U}$**
    - Certificates → half-spaces → polyhedra → polytope outer approximation
    - Direction selection strategies (simplex, incremental, grouped)
13. **Plotting affine slices of $\mathcal{U}$**
    - Slice rasterisation via membership oracle
    - Slicing explicit polyhedral approximations
14. **Geometric interpretation of support functions and certificates**
    - Supporting hyperplanes, sliding half-spaces
    - Certificates as conservative supporting hyperplanes
15. **Derivation of the support formula for $\mathcal{U}$**
16. **Support of a Mahalanobis ellipsoid** (appendix)
17. **Support functions, Minkowski sums, ellipsoids, and matrix order**
    - Cheat sheet for support function identities

### `theory/theoretical_manual.txt` (441 lines) — **Code-linked Companion**
Title: "Relations Between Spaces"

**Section structure:**
1. **Introduction** — Hilbert spaces, forward operator, adjoint vs dual, Riesz maps
   - Includes `\codebox` sections linking math to pygeoinf API methods
2. **Model space as Hilbert space and finite approximation** — Riesz bases, coefficient representations, analysis/synthesis operators, Gram operator
3. **More on Linear Operators** — Banach dual vs Hilbert adjoint derivation, Galerkin form, operators as stacks of linear forms
   - Commutative diagram relating primal/dual/finite spaces

**Key feature:** Each mathematical section includes `\codebox` callouts mapping formulas to exact pygeoinf API calls (e.g., `to_dual ≡ R⁻¹`, `from_components ≡ π`).

---

## 6. Gaps Between Code and Documentation

### Documentation Gaps
1. **No tutorials exist** — the Sphinx `tutorials/` directory is empty. The most critical need.
2. **No theory chapter in docs** — theory.txt and theoretical_manual.txt exist as standalone LaTeX but are not integrated into Sphinx.
3. **No user manual** — the `index.rst` has a good overview but no "getting started", "cookbook", or worked examples.
4. **Missing module docs for:**
   - `data_assimilation/` (entirely undocumented in Sphinx)
   - `convex_analysis.py` (module docstring absent — only class docstrings)
   - `convex_optimisation.py` (minimal docstring for SubgradientDescent)
   - `backus_gilbert.py` (placeholder module docstring: "To be done...")
5. **No cross-references** between theory.txt mathematical derivations and code.
6. **`symmetric_space/`** has good per-class docstrings but no narrative guide.
7. **Preconditioners** lack usage guidance — when to choose Jacobi vs Spectral.

### Code Capabilities Not Documented
1. **Axiom checking system** (`checks/`) — powerful self-testing but no user guide.
2. **Galerkin vs standard matrix** distinction — critical for mass-weighted spaces, only documented in theoretical_manual.txt.
3. **`MassWeightedHilbertSpace`** interaction with `from_formal_adjoint` — crucial pattern, only in reference doc.
4. **Joint inversions** via `LinearForwardProblem.from_direct_sum` — no tutorial.
5. **Constrained inversions** (both Bayesian and optimization) — no worked example.
6. **Polytope approximation / visualization pipeline** — theory.txt describes it thoroughly but no code tutorial.
7. **Randomized decompositions** — when and how to use `random_svd`, `random_cholesky`, etc.
8. **Preconditioning strategies** — no guidance on solver/preconditioner selection.
9. **`empirical_data_error_measure`** in auxiliary.py — undocumented utility.
10. **`line_search` function** for custom optimization loops — no tutorial.

### Mathematical Coverage Alignment
| Theory Topic | Code Module(s) | Gap |
|---|---|---|
| Dual master equation | `backus_gilbert.py` (`DualMasterCostFunction`) | Module stub docstring only |
| Support functions | `convex_analysis.py` | No module-level docstring |
| Polytope approximation | `subsets.py` (`PolyhedralSet`), `visualization.py` | No tutorial |
| Bayesian posterior | `linear_bayesian.py`, `gaussian_measure.py` | Well-documented |
| Tikhonov/min-norm | `linear_optimisation.py` | Well-documented |
| Galerkin form / mass matrices | `linear_operators.py`, `theoretical_manual.txt` | Only in LaTeX |
| Riesz maps / duality | `hilbert_space.py`, `theoretical_manual.txt` | Only in LaTeX |
| SOLA / BG estimators | theory.txt §8, §10–11 | Code incomplete ("to be done") |
| Affine constraints | `subspaces.py` | Good docstrings, no narrative |
| Data assimilation | `data_assimilation/core.py` | Entirely separate, no docs |

---

## Summary Statistics

| Metric | Value |
|---|---|
| Total .py files in `pygeoinf/pygeoinf/` | ~25 (main) + 8 (subdirs) |
| Total lines of Python | ~12,000+ |
| Public API symbols (`__all__`) | 62 |
| Classes | ~55 |
| Theory documents | 2 LaTeX files (3113 lines total) + 18 reference PDFs |
| Sphinx setup | Complete but empty of tutorials |
| Docstring coverage | ~85% (majority of public methods documented) |
