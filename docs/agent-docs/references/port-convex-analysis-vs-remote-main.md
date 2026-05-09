# Branch Comparison: `port_convex_analysis` vs `origin/main`

**Date:** 2026-05-04
**Context:** The developer on `origin/main` (da380) wants to merge only purely convex
analysis additions from `port_convex_analysis`. This document records what is in each
branch and which changes are safe to propose for merge.

---

## Divergence Point

Both branches share common ancestor **`1c909e1` (tag: v1.7.7)**. Remote `main` has
advanced to v1.7.8 via PR #130, while `port_convex_analysis` has not yet incorporated
those commits.

---

## New on `origin/main` (not yet in `port_convex_analysis`)

**PR #130 "Develop" → v1.7.8** (`eeb5db8`)

| File | Change |
|---|---|
| `pygeoinf/symmetric_space/symmetric_space.py` | `l2_products_operator()` added to two classes — new operator for computing vectors of L2 inner products |
| `pygeoinf/plot.py` | Minor legend cleanup: numerical true-parameter values removed from labels |
| `work/inverse_covariance.py` | Deleted (not library code) |
| `work/preconditioning_examples.py`, `work/tomo_example.py` | Minor tweaks |

**Action required:** Rebase or merge `origin/main` into `port_convex_analysis` to pick
up v1.7.8 before proposing any PR.

---

## Changes in `port_convex_analysis` Relative to `origin/main`

### Purely Convex Analysis — Suitable for PR to `main`

| File | Nature |
|---|---|
| `pygeoinf/convex_analysis.py` (**NEW**) | Core support-function hierarchy: `SupportFunction`, `BallSupportFunction`, `EllipsoidSupportFunction`, `HalfSpaceSupportFunction`, `CallableSupportFunction`, `PointSupportFunction`, `LinearImageSupportFunction`, `MinkowskiSumSupportFunction`, `ScaledSupportFunction` |
| `pygeoinf/convex_optimisation.py` (**NEW**) | `SubgradientDescent`, `ProximalBundleMethod`, `PrimalKKTSolver`, `KKTResult` |
| `pygeoinf/backus_gilbert.py` | Rewritten: `DualMasterCostFunction` — dual master cost oracle for Backus-Gilbert inversion |
| `pygeoinf/subsets.py` | Large additions: convex subset extensions, `Subset.plot()` visualization, halfspace support, new concrete subsets |
| `pygeoinf/subspaces.py` (**NEW**) | `AffineSubspace` and geometric constructors |
| `pygeoinf/plot.py` | `SubspaceSlicePlotter`, `plot_slice` |
| `pygeoinf/__init__.py` | All new convex analysis symbols exported |
| `pyproject.toml` | Optional deps: `osqp` (fast-bundle), `clarabel` (bundle-alt), `plotly` (interactive); dev deps cleaned up |
| `pygeoinf/nonlinear_forms.py` | `subgradient` parameter added to `NonLinearForm` — directly required by `DualMasterCostFunction` |
| `tests/test_bundle_core.py` | New (220 lines) |
| `tests/test_chambolle_pock.py` | New (211 lines) |
| `tests/test_dual_master_cost.py` | New (305 lines) |
| `tests/test_halfspaces.py` | New (426 lines) |
| `tests/test_level_bundle.py` | New (213 lines) |
| `tests/test_primal_kkt_solver.py` | New (429 lines) |
| `tests/test_proximal_bundle.py` | New (198 lines) |
| `tests/test_qp_backends.py` | New (76 lines) |
| `tests/test_smoothed_lbfgs.py` | New (258 lines) |
| `tests/test_solve_support_values.py` | New (182 lines) |
| `tests/test_sphere_dli_example.py` | New (200 lines) |
| `tests/test_subgradient.py` | New (103 lines) |
| `tests/test_subsets.py` | Extended (123+ lines) |
| `tests/test_subspaces.py` | New (59 lines) |
| `tests/test_support_function_algebra.py` | New (740 lines) |
| `tests/test_support_function_constructors.py` | New (462 lines) |
| `tests/test_direct_sum.py` | Extended (218+ lines) |
| `tests/test_plot.py` | Extended (701+ lines) |

---

### Non-Convex Changes — Developer Does NOT Want in PR (yet)

These are changes that are **not** purely convex analysis additions. They should either
be held back, stripped into a separate PR, or negotiated individually.

#### Foundational (coupled to convex stack — needs discussion)

| File | Change | Why it matters |
|---|---|---|
| `pygeoinf/hilbert_space.py` | `axpy()` signature changed: `None` → `Vector` return type | Base-class API break; used internally by convex algorithms |
| `pygeoinf/hilbert_space.py` | `dim` docstring revised to clarify "finite representation dimension" | Doc-only, not functionally breaking |
| `pygeoinf/direct_sum.py` | `axpy()` returns `List[Any]` instead of `None` | Mirrors `HilbertSpace.axpy()` change |
| `pygeoinf/direct_sum.py` | `zero`, `inner_product()`, `random()` methods added | Required by `PrimalKKTSolver` with basis-free (dim=0) component spaces |

> **Key tension:** The `axpy()` return-value change and the `direct_sum.py`
> `zero`/`inner_product`/`random` additions are **required** for `PrimalKKTSolver` to
> operate on basis-free spaces. If the developer wants `PrimalKKTSolver` in the PR,
> these foundational changes must come with it — they cannot be cleanly stripped.
> Worth flagging this dependency explicitly when proposing the PR.

#### Cleanly Separable — Hold Back or Submit as Separate PRs

| File | Change | Nature |
|---|---|---|
| `pygeoinf/hilbert_space.py` | `distance(x1, x2)` method added | Convenience utility; not needed for convex analysis |
| `pygeoinf/linear_optimisation.py` | Bug fix: damping convergence criterion uses `rtol * damping_upper` instead of `rtol * (lower + upper)` | Unrelated to convex analysis; prevents maxiter exhaustion when lower→0 |
| `pygeoinf/symmetric_space/sphere.py` | `src_style`/`rec_style` `.pop()` moved before the geodesic loop in `plot_geodesic_network` | Bug fix: original code called `.pop()` after `kwargs` was consumed, causing a `KeyError` |

---

## Recommended PR Strategy

1. **Rebase `port_convex_analysis` onto `origin/main`** to pick up v1.7.8 first.
2. **Negotiate the foundational changes** (`axpy` return, `direct_sum` additions) with
   the developer — frame them as "required infrastructure for `PrimalKKTSolver`", not
   general API improvements.
3. **Exclude from PR** (or separate PR): `distance()`, the damping bugfix, the
   `sphere.py` bugfix.
4. **Include in PR** (convex analysis stack): everything in the "Purely Convex Analysis"
   table above, plus the `nonlinear_forms.py` subgradient additions and the coupled
   foundational changes if negotiated.
