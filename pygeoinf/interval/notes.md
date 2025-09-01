# Notes for Cambridge Visit

## Why functions.py and interval_domain.py?
The core idea is simple: represent functions on a 1D interval as first-class
objects so evaluation, algebra and integration are consistent and efficient.

### Key reasons (short):

- Represent a function as an object with an `evaluate` callable and optional
	coefficient representation for projections and spectral methods.
- Support scalar and array evaluation and arithmetic (`+`, `-`, `*`) that
	returns `Function` instances preserving compact-support semantics.
- Track compact support so evaluation and integration avoid unnecessary work
	(speedups for localised functions).
- Keep quadrature and meshes in `IntervalDomain` so geometry and numeric
	parameters live in one place and can be reused across functions.

Responsibilities (one line): provide a stable, minimal API for evaluation,
algebra and integration of 1D functions while delegating geometry/meshing to
`IntervalDomain`.

### Future work — integration profiling (short)

- Profile integration for representative function families (bumps, splines,
	narrow/large support) to measure error vs cost for parameter sets
	(n_points, method, vectorized).
- Use a high-accuracy reference (dense grid or SciPy quad) or self-convergence
	to estimate errors and select conservative defaults.
- Record results in a small, machine-readable summary so defaults are
	auditable and reproducible.
- Expose chosen defaults in providers (e.g. `default_integration_params`) and
	allow caller overrides; cache normalization constants deterministically.
- Keep the profiling tool lightweight and optional (live under
	`pygeoinf.interval.profiling`) so basic integration remains dependency-light.

## Why functions_providers.py?

### Purpose

- There are function classes that I use very often. I want to  be able to create instances of those functions easily

### Future work

-  Use profiling to find optimal default parameters (for bump function specifically)

## Why l2_space.py?

I think this one is obvious. But I need to refactor this to work with the new implementation of hilbert spaces.

I still think that giving basis functions and inner product is more convenient than giving to and from maps. This class will use basis functions and create to and from maps from them so it is compatible to the original hilbert space class.

## Why providers.py ?

This is simply because besides simple functions it may be easier to have a class for basis functions as a collective. This is a wrapper for function providers that is aware of the dimensions of the space and uses indexing to select the functions, so people don't need to worry about parameters.

## Why operators.py ?

This is maninly for the sola operator and covariance operators.

## Why fem_solvers ?

This separates the fem solvers from the implementation of the operators that need FEM. It is just cleaner and easier for debugging.

## Why boundary_conditions ?

This one is quite small but it is useful to store the information about bcs in a more structured way, so I created a class for that.

## Why sobolev providers ?

This one will be the workhosr, but now it does not work well ...

Right now everything works with l2space because the inner product of the sobolev space gets in the way.

## The L2 vs Sobolev problem (theoretical gap)

### Current situation
Most code in the `interval/` folder uses `L2Space` even when theoretically we should use Sobolev spaces:
- FEM solvers use L2 inner products for stiffness matrix assembly
- Operators like `LaplacianInverseOperator` work with L2 spaces
- Function providers create functions in L2 spaces

### Theoretical issue
- **FEM**: should use H¹ Sobolev space (derivatives exist) but we use L2 inner product for convenience
- **Laplacian operators**: map H^s → H^(s-2) but we treat everything as L2 functions
- **Boundary conditions**: naturally live in Sobolev spaces (traces) but we handle them in L2

### Design choice needed
Options:
1. **Pragmatic**: keep using L2 everywhere since it works numerically and is simpler
2. **Theoretical**: refactor to use proper Sobolev spaces but deal with inner product complexity
3. **Hybrid**: use L2 for basis construction but Sobolev for operators/BCs when needed

Current preference: option 1 (pragmatic) until the theoretical benefits become essential.

## Boundary conditions → trace operators (design consideration)

### Current boundary conditions approach
- `BoundaryConditions` class stores parameter values (Dirichlet: u(a)=α, Neumann: u'(a)=β, Robin: αu+βu'=γ)
- Operators and FEM solvers check boundary condition types and apply discrete constraints
- Works by modifying matrices/equations after assembly

### Trace operator approach (more mathematical)
**Concept**: replace discrete boundary conditions with continuous trace operators:
- **Dirichlet**: trace operator T₀: H¹ → ℝ² where T₀(u) = (u(a), u(b))
- **Neumann**: normal trace T₁: H² → ℝ² where T₁(u) = (u'(a), u'(b))
- **Robin**: combination operator T_R: H¹ → ℝ² where T_R(u) = (αu(a)+βu'(a), ...)

### How to implement trace operators practically

**Step 1: Create trace operator classes**
```python
class TraceOperator(LinearOperator):
    def __init__(self, domain: Sobolev, trace_type: str, boundary_points: list):
        # Maps H^s → ℝ^k where k = number of boundary points

class DirichletTrace(TraceOperator):
    # T₀: H¹ → ℝ² where T₀(u) = (u(a), u(b))

class NeumannTrace(TraceOperator):
    # T₁: H² → ℝ² where T₁(u) = (u'(a), u'(b))
```

**Step 2: Problem formulation with both operators**
The mathematical problem becomes: find u such that
```
Lu = f     (differential equation)
Tu = g     (boundary/trace constraint)
```
where L is Laplacian, T is trace operator, f is source, g is boundary data.

**Step 3: Solver construction**
- Instead of: `LaplacianInverseOperator(space, BoundaryConditions.dirichlet(0,0))`
- Use: `BoundaryValueProblem(laplacian_op, trace_op, boundary_data)`
- Or: `ConstrainedSolver([laplacian_op, trace_op], [source_f, boundary_g])`

**Step 4: Equal footing implementation**
Both operators work together in problem assembly:
- Laplacian provides stiffness matrix A
- Trace operator provides constraint matrix B
- Solve constrained system: [A B^T; B 0][u; λ] = [f; g]

### Benefits vs complexity
**Benefits:**
- Mathematically rigorous (proper operator relationships)
- Trace and differential operators have equal status in problem formulation
- Generalizes to higher dimensions naturally
- Separates PDE physics (Laplacian) from boundary geometry (trace)
- Supports multiple constraints naturally (mixed BCs, contact problems)

**Complexity:**
- Need separate problem assembly class (`BoundaryValueProblem` or `ConstrainedSolver`)
- Saddle-point linear systems require specialized solvers
- More complex than current "modify matrix after assembly" approach

### Mathematical insight
Your confusion highlights the key insight: boundary conditions aren't "properties" of differential operators, they're separate constraints that work together with the PDE to define a well-posed problem. The trace operator approach makes this mathematically explicit.

### Recommendation
**Phase 1**: Keep current `BoundaryConditions` for now
**Phase 2**: Implement `TraceOperator` classes as alternative interface
**Phase 3**: Gradually migrate FEM/operators to use trace operators where beneficial

This is lower priority than the L2/Sobolev issue since current BC approach works well numerically.


L2 space has no bcs (obviously). Sobolev spaces might have bcs included.