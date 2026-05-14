# Backus-Gilbert Reference

## Scope

This reference covers [pygeoinf/backus_gilbert.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/backus_gilbert.py), the Backus-Gilbert public surface in [pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py), and relevant tests in [tests/test_backus_gilbert_*.py](/home/adrian/PhD/Inferences/pygeoinf/tests/).

The module implements Backus' method for parameter estimation with uncertainty quantification, using convex support-function algebra to represent model priors and data errors. The core is the dual master cost function $\varphi(\lambda; q)$ minimized over the Lagrange multiplier $\lambda \in D$ to yield the admissible region for a property $q \in P$ of the model.

## Core Classes

### `DualMasterCostFunction`

**Role:** Oracle for the Backus dual optimization, implementing $\varphi(\lambda; q)$ as a `NonLinearForm` on the data space $D$.

**Construction:**
```python
from pygeoinf import EuclideanSpace
from pygeoinf.convex_analysis import BallSupportFunction

cost = DualMasterCostFunction(
    data_space=EuclideanSpace(m),
    property_space=EuclideanSpace(1),
    model_space=model_space,
    G=forward_operator,  # model → data
    T=property_operator,  # model → property
    model_prior_support=BallSupportFunction(model_space, center, radius),
    data_error_support=BallSupportFunction(data_space, center, radius),
    observed_data=d_observed,
    q_direction=q_test
)
```

**Attributes:**
- `observed_data` — the observed data vector $\tilde{d} \in D$
- `direction` — the current property direction $q \in P$

**Methods:**
- `set_direction(q)` — update the property direction and cache $T^* q$
- `_mapping(lam)` — evaluate $\varphi(\lambda; q) = \langle \lambda, \tilde{d} \rangle_D + \sigma_B(T^* q - G^* \lambda) + \sigma_V(-\lambda)$
- `_subgradient(lam)` — subgradient via support-point delegatio to $\sigma_B$ and $\sigma_V$
- `value_and_subgradient(lam)` — fused evaluation sharing $G^* \lambda$ and support-point queries
- `_finite_difference_gradient(lam, eps=1e-6)` — fallback when support points unavailable

**Design notes:**
- Composition of three support-function terms, each evaluable independently
- Caches $T^* q$ to avoid recomputing on repeated property evaluations
- Caches the adjoint `G.adjoint` for efficiency
- Falls back to finite-difference gradients if support points are unavailable (e.g., for degenerate sets)
- Inherits from `NonLinearForm` so it works directly with `ProximalBundleMethod` and `SubgradientDescent`

### `BackusInference`

**Role:** High-level problem wrapper for Backus inference (legacy/convenience layer).

**Construction:**
```python
inference = BackusInference(
    forward_problem=fp,
    property_operator=T,
    prior_norm_bound=M,
    significance_level=0.95,
    constraint_solver=CholeskySolver  # optional
)
```

**Methods:**
- `test_data_compatibility(data, solver)` — check whether a model exists satisfying both the data and prior norm bound

**Status:** Minimal implementation; most BG workflows use `DualMasterCostFunction` directly with optimization routines.

## Typical Workflow

1. **Define the forward problem** $G: M \to D$ and property operator $T: M \to P$
2. **Choose priors/errors** as support functions $\sigma_B$ (model) and $\sigma_V$ (data)
3. **Create the cost function** via `DualMasterCostFunction`
4. **Optimize over $\lambda$** using `ProximalBundleMethod` or `SubgradientDescent`, sweeping over property directions $q$
5. **Extract the admissible region** as the $(T^* q, h_U(q))$ pairs where $h_U(q) = \inf_\lambda \varphi(\lambda; q)$

## Mission Context

**Last Updated:** 2026-05-13

The *Lowering Execution Framework* mission (started 2026-04-03) includes a Phase 5 to integrate reduced BG operators into the lowering planner. The semantic `DualMasterCostFunction` code is **stable and unchanged** since its last update. Phase 5 work planned a `ReducedDualMasterCostFunction` subclass that evaluates the full BG cost entirely in data/property space using pre-computed Gram/cross-Gram matrices for `BallSupportFunction` priors — achieving 30× speedup on test workloads. That implementation is currently planned but not yet merged to main.

See `docs/agent-docs/completed-plans/lowering-execution-framework-phase-5-complete.md` for mission progress.

---

## See Also

- [[convex-analysis-reference]] — support functions and algebraic combinators
- [[sphere-dli-example-reference]] — concrete Backus-Gilbert workflow example on a simple 2D sphere domain
