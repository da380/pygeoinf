# Polyhedral Approximation Reference

## Scope

This reference covers [pygeoinf/polyhedral_approximation.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/polyhedral_approximation.py) and relevant tests in [tests/test_polyhedral_approximation.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_polyhedral_approximation.py).

The module implements incremental polyhedral approximations of DLI feasible regions in property space. Each direction $q$ on the unit sphere yields one supporting hyperplane constraint via DLI bounds, producing a PolyhedralSet that approximates the admissible parameter region.

## Core Classes

### `DirectionSampler`

**Role:** Factory for generating direction sets on the unit sphere in property space.

**Key Methods:**

- `box(n_dims)` — Cardinal axis-aligned directions $\pm e_i$ (returns $2n$ unit vectors)
- `simplex(n_dims, random_state)` — Random non-degenerate simplex ($n+1$ directions, det > 1e-3)
- `random_uniform(n_dims, n_directions, random_state)` — Uniformly random directions via normalized Gaussian
- `get(strategy, n_dims, n_new, **kwargs)` — Factory dispatcher for "box", "simplex", "random_uniform"

All methods return arrays of shape `(n_directions, n_dims)` with unit-norm rows.

### `PolyhedralApproximation`

**Role:** Incremental polyhedral approximation via directional DLI bound caching.

**Construction:**
```python
approx = PolyhedralApproximation(
    property_space=EuclideanSpace(6),
    cost_function=DualMasterCostFunction(...),
    solver=ProximalBundleMethod(...),  # or PrimalKKTSolver
    property_operator=T  # required when using PrimalKKTSolver
)
```

**Key Methods:**

- `initialize(strategy)` — Add initial directions via "box" or "simplex"
- `refine(strategy, n_new, **kwargs)` — Add more directions for refinement
- `add_directions(directions)` — Solve and cache bounds for explicit direction set
- `as_polyhedral_set()` — Return current approximation as PolyhedralSet for visualization
- `plot(dims=[i, j], **kwargs)` — Plot 2D slice via plot_slice with fast scipy.spatial.HalfspaceIntersection

**Dual-Solver Support:**

The class detects solver type at construction via `_is_primal_kkt_solver()`:

- **PrimalKKTSolver** (fast, direct): Given c = T*q, solves primal KKT system directly returning optimal u*. Bound h_U(q) = <c, u*>. No iteration, basis-free.
- **DualMasterCostFunction + bundle** (iterative): Sets direction q, minimizes φ(λ; q) over data space λ, returns f_best as bound.

**Deduplication:** Directions normalized to unit length; cached by 12-decimal rounding tuple to skip near-duplicates.

**Attributes:**
- `n_constraints` — Number of cached half-spaces
- `_cache` — Maps rounded direction tuples → h_U values
- `_property_space` — Domain for constraints and plotting

## Typical Workflow

```python
# 1. Build problem (forward operator G, property operator T)
model_space = ... # Hilbert space
property_space = EuclideanSpace(6)

# 2. Choose solver (PrimalKKTSolver for ball constraints is faster)
solver = PrimalKKTSolver(
    model_ball, data_ball, forward_operator, observed_data
)

# 3. Create approximation (property_operator only needed for PrimalKKTSolver)
approx = PolyhedralApproximation(
    property_space, cost=None, solver=solver, property_operator=T
)

# 4. Initialize with box (axis-aligned), then refine
approx.initialize("box")
approx.refine("random_uniform", n_new=60, random_state=42)

# 5. Visualize
fig, ax, _ = approx.plot(dims=[0, 1])

# 6. Extract as PolyhedralSet for downstream use
feasible_set = approx.as_polyhedral_set()
```

## Mission Context

**Last Updated:** 2026-05-19

The *Lowering Execution Framework* mission (Phase 5 integration) uses polyhedral approximation to visualize DLI feasible regions and validate reduced operator implementations. The module supports both iterative bundle methods (DualMasterCostFunction) and direct KKT solvers (PrimalKKTSolver) for ball/ellipsoid-constrained problems.

**Performance:** With PrimalKKTSolver on 6D property space, 72 directional solves complete in ~2.5 minutes (includes model space dimension ~16k), demonstrating the speedup of direct KKT solving over iterative optimization for sphere problems.

See `docs/agent-docs/completed-plans/lowering-execution-framework-phase-5-complete.md` for mission progress.

---

## See Also

- [[backus-gilbert-reference]] — DualMasterCostFunction and DLI bounds
- [[convex-analysis-reference]] — Support functions and BallSupportFunction
- [pygeoinf/plot.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/plot.py) — plot_slice implementation using scipy.spatial.HalfspaceIntersection
