# Convex Analysis Port — Reviewer Guide

## Overview

This branch ports **convex analysis and optimisation capabilities** from the `convex_analysis` development branch onto `port_convex_analysis` (based on `main`). The port adds support functions, convex optimisation solvers, geometric subset types, and subset visualisation — enabling deterministic linear inversion (DLI) formulated as convex analysis problems.

**What changed:** ~10,600 lines added across 2 new modules, 8 modified modules, and 16 new test files.
**Test count:** 388 → 624 (236 new tests, all passing).
**No regressions:** All pre-existing tests pass unchanged.

---

## Architecture: Dependency Diagram

```
Layer 4 – Application
  backus_gilbert.py (DualMasterCostFunction)
  plot.py (SubspaceSlicePlotter, plot_slice)
       │
Layer 3 – Optimisation
  convex_optimisation.py (5 solvers, QP backends, solve_support_function_values)
       │
Layer 2 – Support Functions
  convex_analysis.py (SupportFunction hierarchy, 9 classes)
       │
Layer 1 – Foundations (modified existing modules)
  hilbert_space.py  nonlinear_forms.py  subsets.py  direct_sum.py
  linear_optimisation.py  subspaces.py
       │
Layer 0 – Unchanged Core
  linear_operators.py  forward_problem.py  inversion.py  ...
```

Each layer depends only on layers below it. Reviewing bottom-up ensures you see dependencies before dependents.

---

## Component Reference

### Layer 1: Foundation Changes

#### `hilbert_space.py`
- **`axpy()` now returns Vector** — previously returned `None`. Enables fluent accumulation patterns (`y = H.axpy(a, x, y)`). `MassWeightedHilbertSpace` updated to match.
- **`distance(x1, x2)`** — new `@final` method computing `‖x₁ - x₂‖`. Convenience wrapper.
- **Dependencies:** None new.  **Dependents:** `direct_sum.py` (axpy return), `convex_optimisation.py` (distance).

#### `nonlinear_forms.py`
- **Subgradient infrastructure** — `NonLinearForm` gains `subgradient` parameter, `has_subgradient` property, and `subgradient(x)` method. Algebraic operations (`__neg__`, `__mul__`, `__add__`) propagate subgradients.
- **Why:** Support functions and convex cost functions need subgradients for non-smooth optimisation.
- **Dependencies:** None new.  **Dependents:** `convex_analysis.py` (all SupportFunctions), `backus_gilbert.py` (DualMasterCostFunction).

#### `subsets.py`
- **`ConvexSubset` abstract extensions** — new abstract methods: `support_function` (property), `directional_bound(direction)`. New concrete methods: `closure()`, `_warn_if_open(operation)`.
- **`Ellipsoid` / `Ball`** — gain `support_function` property (lazy, deferred import from `convex_analysis.py`), `directional_bound()`, optional `inverse_operator`/`inverse_sqrt_operator` constructor params.
- **`ConvexIntersection`** — gains `support_upper_bound()`, `feasible_lower_bound()`.
- **`HyperPlane`** — new class: `{x | ⟨a, x⟩ = b}`. Provides `is_element`, `distance_to`, `project`.
- **`HalfSpace`** — new class: `{x | ⟨a, x⟩ ≤ b}` or `≥`. Provides `is_element`, `distance_to`, `project`, `support_function`.
- **`PolyhedralSet`** — new class: intersection of `HalfSpace` objects.
- **`Subset.plot()`** — convenience method that delegates to `plot_slice()`.
- **Dependencies:** `convex_analysis.py` (lazy imports only).  **Dependents:** `convex_analysis.py` (BallSupportFunction etc.), `plot.py`.

#### `direct_sum.py`
- **`HilbertSpaceDirectSum`** — new `zero` property, `inner_product()`, `random()` overrides that work componentwise without routing through `to_components`/`from_components`. Enables basis-free subspaces (dim=0).
- **`axpy()` returns `List[Any]`** — matches base class contract.
- **`ColumnLinearOperator` / `RowLinearOperator`** — use `axpy` return value correctly.
- **Dependencies:** None new.  **Dependents:** Used by direct-sum operator constructions.

#### `linear_optimisation.py`
- **Bisection convergence fix** — the tolerance criterion for the damping bisection used `rtol * (lower + upper)` which collapses when `lower → 0`. Fixed to `rtol * upper`.
- **Dependencies / Dependents:** No new ones.

#### `subspaces.py`
- **`AffineSubspace.get_tangent_basis()`** — extracts orthonormal tangent basis vectors.
- **`AffineSubspace.from_hyperplanes()`** — constructs subspace from hyperplane intersection.
- **`AffineSubspace.to_hyperplanes()`** — converts subspace to hyperplane representation.
- **Dependencies:** `subsets.py` (HyperPlane).  **Dependents:** `plot.py`, `subsets.py` (Subset.plot default subspace).

---

### Layer 2: Support Functions (`convex_analysis.py` — NEW, 913 lines)

The **support function** of a closed convex set $C$ is $h_C(q) = \sup\{\langle q, x\rangle : x \in C\}$. This module provides 9 classes implementing the `SupportFunction` abstract base.

| Class | Formula | Set |
|---|---|---|
| `BallSupportFunction` | $h(q) = \langle q, c\rangle + r\|q\|$ | Ball $B(c, r)$ |
| `EllipsoidSupportFunction` | $h(q) = \langle q, c\rangle + r\|A^{-1/2}q\|$ | Ellipsoid $E(c, r, A)$ |
| `HalfSpaceSupportFunction`* | $h(q) = b \cdot \langle q, a/\|a\|\rangle$ if aligned | Half-space $\{x \mid \langle a, x\rangle \le b\}$ |
| `CallableSupportFunction` | User-supplied callable | Any |
| `PointSupportFunction` | $h(q) = \langle q, x_0\rangle$ | Singleton $\{x_0\}$ |
| `LinearImageSupportFunction` | $h_{A(C)}(q) = h_C(A^*q)$ | Affine image |
| `MinkowskiSumSupportFunction` | $h_{C_1 \oplus C_2}(q) = h_{C_1}(q) + h_{C_2}(q)$ | Minkowski sum |
| `ScaledSupportFunction` | $h_{\alpha C}(q) = \alpha \cdot h_C(q)$ | Scaled set |

\* `HalfSpaceSupportFunction` extends `NonLinearForm` rather than `SupportFunction` because half-spaces are unbounded (infinite in some directions).

**Dependencies:** `nonlinear_forms.py`, `hilbert_space.py`.
**Dependents:** `subsets.py` (Ellipsoid/Ball `.support_function`), `convex_optimisation.py`, `backus_gilbert.py`.

---

### Layer 3: Convex Optimisation (`convex_optimisation.py` — NEW, 2205 lines)

Five optimisation methods for minimising non-smooth convex functions, plus QP infrastructure.

| Solver | Method | Use Case |
|---|---|---|
| `SubgradientDescent` | Projected subgradient with step-size rules | Simple convex, non-smooth |
| `ProximalBundleMethod` | Proximal bundle (cutting-plane model + QP master) | General non-smooth convex |
| `LevelBundleMethod` | Level-set bundle (constrainted QP master) | Non-smooth, level control |
| `SmoothedLBFGSSolver` | Moreau-smoothed continuation + L-BFGS | Large-scale smooth approx |
| `ChambollePockSolver` | Chambolle-Pock primal-dual splitting | Saddle-point / constrained |

**QP Backends:**
- `SciPyQPSolver` — always available (scipy.optimize)
- `OSQPQPSolver` — optional (`pip install osqp`)
- `ClarabelQPSolver` — optional (`pip install clarabel`)

**`solve_support_function_values()`** — evaluates a support function in multiple directions using parallelised optimisation.

**Dependencies:** `convex_analysis.py`, `nonlinear_forms.py`, `hilbert_space.py`.
**Optional:** `osqp`, `clarabel`, `joblib` (parallelism).
**Dependents:** `backus_gilbert.py` (DualMasterCostFunction used with these solvers).

---

### Layer 4: Application

#### `backus_gilbert.py` — `DualMasterCostFunction`
The dual master cost function for convex Backus-Gilbert formulations:

$$\varphi(\lambda; q) = \langle \lambda, \tilde{d}\rangle_D + \sigma_B(T^*q - G^*\lambda) + \sigma_V(-\lambda)$$

where $\sigma_B$, $\sigma_V$ are support functions of the model-prior and data-error convex sets. Minimising over $\lambda$ yields the support-function value for the induced property uncertainty set.

**Dependencies:** `convex_analysis.py` (SupportFunction), `nonlinear_forms.py`, `forward_problem.py`.
**Dependents:** Used with any Layer 3 solver.

#### `plot.py` — `SubspaceSlicePlotter`, `plot_slice()`
Visualises any `Subset` along a 1D, 2D, or 3D affine subspace:
- 1D: line plot of membership
- 2D: filled contour / mask plot
- 3D: Matplotlib voxels or Plotly isosurface (if installed)

**Dependencies:** `subsets.py`, `subspaces.py`, `matplotlib`, optionally `plotly`.
**Dependents:** `Subset.plot()` delegates here.

---

## Packaging Changes (`pyproject.toml`)

New optional dependency groups:
```toml
[project.optional-dependencies]
qp = ["osqp>=0.6", "clarabel>=0.6"]
interactive = ["plotly>=5.0"]
```

Install with: `pip install -e ".[qp,interactive]"` or `pip install -e ".[all]"`.

---

## Test Coverage Summary

| Test File | Tests | Covers |
|---|---|---|
| `test_support_function_constructors.py` | 48 | All SupportFunction concrete classes |
| `test_support_function_algebra.py` | 54 | Algebraic operations (image, sum, scale) |
| `test_halfspaces.py` | 35 | HyperPlane, HalfSpace, PolyhedralSet |
| `test_direct_sum.py` | +7 | Basis-free direct sums |
| `test_subgradient.py` | 4 | SubgradientDescent |
| `test_proximal_bundle.py` | ~8 | ProximalBundleMethod |
| `test_level_bundle.py` | ~8 | LevelBundleMethod |
| `test_smoothed_lbfgs.py` | ~8 | SmoothedLBFGSSolver |
| `test_chambolle_pock.py` | ~8 | ChambollePockSolver |
| `test_bundle_core.py` | ~8 | Bundle/Cut data structures |
| `test_qp_backends.py` | 5 | SciPy, OSQP, Clarabel QP solvers |
| `test_solve_support_values.py` | ~8 | Multi-direction support evaluation |
| `test_dual_master_cost.py` | 9 | DualMasterCostFunction oracle |
| `test_subspaces.py` | +4 | get_tangent_basis |
| `test_plot.py` | +25 | SubspaceSlicePlotter, plot_slice |
| `test_subsets.py` | +5 | Subset.plot() entry point |

**Total new tests:** 236
**All passing:** 624 passed, 1 xfailed (pre-existing)

---

## Known Gaps & Design Notes

1. **`from_hyperplanes` / `to_hyperplanes` round-trip tests** — these `AffineSubspace` methods lack dedicated round-trip tests. They are exercised indirectly by `plot_slice` and `Subset.plot()` tests but could use explicit coverage.

2. **QP solver failure handling** — `ProximalBundleMethod` and `LevelBundleMethod` do not always verify QP solver success status before using `result.x`. In practice the QP subproblems are well-conditioned, but robust error propagation would be an improvement.

3. **`HalfSpaceSupportFunction` is not a `SupportFunction`** — it extends `NonLinearForm` directly because half-spaces are unbounded (infinite support in some directions). This is a deliberate design choice but means it cannot participate in algebraic support-function operations.

4. **`SmoothedLBFGSSolver.tolerance`** — documented as determining the final smoothing level, but the continuation schedule is driven by `epsilon0` and `n_levels` only. The `tolerance` parameter currently affects the L-BFGS convergence criterion, not the smoothing schedule.

5. **Negative radius in direct construction** — `BallSupportFunction(domain, center, -1.0)` does not raise. The geometric `Ball` constructor validates radius, but the support-function constructor does not duplicate the check.

---

## Bug Fixes Applied During Port

Three bugs in the CA branch were fixed during porting (Phase 1 review):

1. **`PolyhedralSet.boundary`** — missing `@property` decorator, causing `poly.boundary` to return a bound method instead of raising `NotImplementedError`.
2. **`HilbertSpaceDirectSum.axpy`** — returned `None` instead of the result list, breaking nested direct-sum accumulation.
3. **`HyperPlane.dimension()`** — returned `domain.dim` instead of `domain.dim - 1`, contradicting its own docstring.

---

## Intentional Divergences from CA Branch

| File | Divergence | Reason |
|---|---|---|
| `direct_sum.py` | `axpy` returns `List[Any]` | Bug fix (CA returns None) |
| `subsets.py` | `HyperPlane.dimension` returns `dim-1`; `PolyhedralSet.boundary` has `@property` | Bug fixes |
| `__init__.py` | No `from . import symmetric_space` | `symmetric_space` refactor excluded from port |
| `__init__.py` | `ProximalBundleMethod` in `__all__` | Missing from CA's `__all__` |
| `test_subgradient.py` | 4 real tests instead of TODO stub | CA had only `pytest.fail("TODO: ...")` |
| `test_gaussian_measure.py` | Keeps main's affine_operator tests | CA removed tests that main added |
| `README.md` | Not updated | User decision to keep minimal |

---

## How to Review

1. **Start bottom-up**: Layer 1 foundations → Layer 2 support functions → Layer 3 optimisation → Layer 4 application.
2. **Run tests**: `python -m pytest tests/ -q` (expects 624 passed, 1 xfailed).
3. **Check optional deps**: `pip install osqp clarabel plotly` for full test coverage.
4. **Key mathematical checks**:
   - Support function formulas match convex analysis theory (see `docs/agent-docs/theory/theory.txt`)
   - Subgradient chain rule in `DualMasterCostFunction`
   - Bundle method convergence (tests verify function value decrease)
5. **Focus areas**: `convex_analysis.py` (correctness of support formulas), `convex_optimisation.py` (solver robustness), `subsets.py` (abstract interface extensions).
