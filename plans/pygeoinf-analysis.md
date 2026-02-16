---
task: pygeoinf-architecture-analysis
repo: pygeoinf
created: 2026-02-13
status: draft
---

## pygeoinf architecture summary

pygeoinf is a Hilbert-space-first library for geophysical inference/inverse problems. The core design separates (1) abstract mathematical structure (spaces/operators/measures) from (2) numerical implementations (matrix backends, randomized decompositions, iterative solvers, parallelization) and (3) problem-level orchestration (forward problems and inversion algorithms).

### Core abstractions (the “type system”)

- **Spaces**: `HilbertSpace` (ABC) plus concrete spaces like `EuclideanSpace` define vector representation, the Riesz map, and basic linear algebra primitives.
  - Primary file: `pygeoinf/hilbert_space.py`

- **Operators**:
  - `NonLinearOperator` / `LinearOperator` represent mappings between spaces, with adjoint/dual support and operator algebra.
  - Primary file: `pygeoinf/linear_operators.py` (large module; most downstream code depends on it)

- **Measures**: `GaussianMeasure` generalizes multivariate normals to Hilbert spaces; provides sampling and affine transforms.
  - Primary file: `pygeoinf/gaussian_measure.py`

- **Forward problems**: `ForwardProblem` and `LinearForwardProblem` bind a model space, data space, forward operator, and optional data error measure.
  - Primary file: `pygeoinf/forward_problem.py`

- **Inversions / inference**:
  - Base: `Inversion`, `LinearInversion`, `Inference` provide shared structure and precondition checks.
  - Primary file: `pygeoinf/inversion.py`

### Problem-solving layers

- **Linear Bayesian inversion** (`LinearBayesianInversion`, constrained variant): computes posterior Gaussian measure via normal operator and Kalman gain.
  - File: `pygeoinf/linear_bayesian.py`

- **Linear optimisation inversions** (`LinearLeastSquaresInversion`, `LinearMinimumNormInversion`, constrained variants): classic deterministic solvers via normal equations and (optionally) error precision.
  - File: `pygeoinf/linear_optimisation.py`

- **Solvers**: abstracted linear solver interfaces + iterative variants; used by both Bayesian and optimisation modules.
  - File: `pygeoinf/linear_solvers.py`

### Geometry / convex analysis (active development area)

- **Convex sets + support functions**: `SupportFunction` hierarchy (e.g., `BallSupportFunction`, `EllipsoidSupportFunction`) provide support-function-based geometry.
  - File: `pygeoinf/convex_analysis.py`

- **Subsets / subspaces**: `ConvexSubset`-style sets (`Ball`, `Ellipsoid`, intersections, etc.) and affine/linear subspaces.
  - Files: `pygeoinf/subsets.py`, `pygeoinf/subspaces.py`

- **Dual master equation work**: detailed roadmap and refactors around support functions and convex sets.
  - File: `pygeoinf/DUAL_MASTER_IMPLEMENTATION_PLAN.md`

### Performance / utilities

- **Randomized linear algebra**: randomized range/SVD/Cholesky utilities.
  - File: `pygeoinf/random_matrix.py`

- **Parallelization**: joblib-based parallel compute patterns.
  - Files: `pygeoinf/parallel.py` and helpers in operator code

- **Plotting**: convenience plotting wrappers.
  - File: `pygeoinf/plot.py`

### Symmetric spaces

There is a dedicated subpackage for circle/sphere style spaces with optional dependencies (`pyshtools`, `Cartopy`).
- Directory: `pygeoinf/symmetric_space/`

## Entrypoints

- **Primary user entrypoint:** `import pygeoinf as inf`
  - `pygeoinf/__init__.py` re-exports the public API via unified imports and an explicit `__all__`.

- **Tutorial-driven entrypoints:** Jupyter notebooks referenced from `README.md` and `docs/`.

No CLI entrypoint is evident from `pyproject.toml` (no `[tool.poetry.scripts]` section in the first 120 lines).

## Key data models / “core objects” for other agents to know

- `HilbertSpace`, `DualHilbertSpace`, `EuclideanSpace`, `HilbertModule` (spaces)
- `LinearOperator` (+ sparse/dense/diagonal variants), `NonLinearOperator` (operators)
- `GaussianMeasure` (priors/noise)
- `ForwardProblem`, `LinearForwardProblem` (problem definition)
- `Inversion`, `LinearInversion`, `Inference` (algorithm bases)
- `LinearBayesianInversion`, `ConstrainedLinearBayesianInversion` (Bayesian)
- `LinearLeastSquaresInversion`, `LinearMinimumNormInversion`, constrained variants (optimisation)
- `SupportFunction` + concrete support functions; `Ball`/`Ellipsoid` and other convex sets (geometry)
- `AffineSubspace`, projectors, subset/set operations (constraints)

## Critical files (high fan-in / start-here)

- `pygeoinf/__init__.py` (public API surface)
- `pygeoinf/hilbert_space.py` (space abstraction)
- `pygeoinf/linear_operators.py` (operator algebra + matrices)
- `pygeoinf/gaussian_measure.py` (probabilistic modeling)
- `pygeoinf/forward_problem.py` (problem binding)
- `pygeoinf/inversion.py` (algorithm base)
- `pygeoinf/linear_bayesian.py` (Bayesian inversion)
- `pygeoinf/linear_optimisation.py` (least squares / minimum norm)
- `pygeoinf/linear_solvers.py` (solver backends)

## Testing notes / likely gaps

Existing tests cover many core concepts (operators, Gaussian measures, forward problems, inversion, random matrices, subspaces). Likely weaker or missing coverage areas:

- `convex_analysis.py` and `subsets.py` support-function integration (especially failure modes when inverse operators are absent)
- constrained inversion variants (conditioning/projection paths)
- parallelization and randomized algorithm correctness under different backends
- some plotting utilities (often intentionally untested)
- the “dual master equation” roadmap items (new cost functions, support-function composition rules)

## Suggested next tasks for agents

1. **Generate a precise public API map** from `pygeoinf/__init__.py.__all__` (grouped by module), and identify stability expectations.
2. **Convex geometry audit**: verify which convex sets expose a `support_function` object; add tests for ball/ellipsoid support functions + intersections.
3. **Dual master equation implementation alignment**: extract the next unfinished phase from `DUAL_MASTER_IMPLEMENTATION_PLAN.md` and convert into a 3–6 phase TDD plan with explicit tests.
4. **Architecture index for navigation**: create a `plans/pygeoinf-index.md` listing the above “critical files” with 1–2 line responsibilities.

