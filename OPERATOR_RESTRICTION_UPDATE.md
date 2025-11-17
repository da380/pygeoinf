# Operator Restriction Framework Implementation

## Summary

Implemented a fundamental operator restriction capability that allows restricting operators (Laplacian) and spaces (Sobolev) to subdomains with flexible boundary conditions. This is architecturally superior to ad-hoc creation of operators on subdomains.

## Motivation

When splitting a domain at discontinuities, we need to specify boundary conditions at the new interior boundaries. Initially, `Sobolev.with_discontinuities()` simply copied the same boundary conditions to all subspaces, but this is insufficient - we need independent control. For example:
- Left subspace: Dirichlet at left boundary, Neumann at discontinuity (DN)
- Right subspace: Neumann at discontinuity, Dirichlet at right boundary (ND)

Rather than adding ad-hoc logic to handle this, we recognized that **operator restriction should be a fundamental capability** that can be composed cleanly.

## Implementation

### 1. `Laplacian.restrict(restricted_space, new_bcs=None)`

Added to `pygeoinf/interval/operators.py` (lines ~543-605):

```python
def restrict(self, restricted_space, new_bcs=None):
    """
    Restrict Laplacian operator to a subspace with new boundary conditions.

    Args:
        restricted_space: Lebesgue or Sobolev space on subdomain.
        new_bcs: New boundary conditions for the restricted operator.
                 If None, uses the same boundary conditions as original.

    Returns:
        New Laplacian operator on the restricted space.
    """
    # Validates subdomain containment
    # Creates new Laplacian preserving alpha, method, dofs, etc.
    # Allows changing boundary conditions
```

**Key features:**
- Validates that restricted domain is contained in original domain
- Preserves all operator parameters: `alpha`, `method`, `dofs`, `fd_order`, `n_samples`, etc.
- Supports arbitrary boundary condition changes
- Creates a proper new Laplacian instance (not just a reference)

### 2. `Sobolev.restrict(restricted_space, new_bcs=None)`

Added to `pygeoinf/interval/sobolev_space.py` (lines ~127-202):

```python
def restrict(self, restricted_space: "Sobolev", new_bcs=None):
    """
    Restrict Sobolev space to a subspace with new boundary conditions.

    Args:
        restricted_space: Target Sobolev space with restricted domain.
        new_bcs: New boundary conditions for the Laplacian operator.

    Returns:
        Updated restricted_space with new Laplacian and mass operators.
    """
    # Validates s and k match
    # Uses Laplacian.restrict() to create restricted Laplacian
    # Updates mass operators (BesselSobolev, BesselSobolevInverse)
```

**Key features:**
- Leverages `Laplacian.restrict()` for operator restriction
- Validates regularity parameters (`s`, `k`) match
- Updates both the Laplacian and the mass operators
- Clean composition pattern

### 3. Enhanced `Sobolev.with_discontinuities()`

The existing classmethod was enhanced with `bcs_per_subspace` parameter (lines ~225-398):

```python
@classmethod
def with_discontinuities(
    cls,
    dim: int,
    function_domain: "IntervalDomain",
    discontinuity_points: list,
    s: float,
    k: float,
    bcs: BoundaryConditions,
    alpha: float,
    bcs_per_subspace: Optional[List[BoundaryConditions]] = None,
    ...
)
```

**Key feature:**
- `bcs_per_subspace`: Optional list allowing different boundary conditions per subspace
- If provided, must have length equal to number of subspaces
- Enables DN+ND configurations at discontinuities

## Testing

Comprehensive test suite validates:

1. **Discontinuous Sobolev space creation**: `with_discontinuities()` works correctly
2. **Laplacian restriction**: Domain restriction with BC changes preserves parameters
3. **Sobolev restriction**: Leverages Laplacian restriction, updates mass operators
4. **BC flexibility**: Supports DD, NN, DN, ND combinations

All tests pass successfully.

## Design Benefits

### 1. Fundamental Capability
Operator restriction is now a first-class operation, not ad-hoc logic scattered throughout.

### 2. Clean Composition
Higher-level operations (Sobolev restriction) naturally leverage lower-level capabilities (Laplacian restriction).

### 3. Flexibility
Boundary conditions at discontinuities can be specified independently per subspace.

### 4. Correctness
When we split an operator at a discontinuity, we don't "just copy the same operator" - we properly specify behavior at new boundaries.

## Usage Examples

### Example 1: Restricting a Laplacian with BC changes

```python
from pygeoinf.interval import Lebesgue, Laplacian, BoundaryConditions
from pygeoinf.interval.interval_domain import IntervalDomain

# Create full-domain Laplacian with DD boundary conditions
domain_full = IntervalDomain(0, 1)
L_full = Lebesgue(100, domain_full, basis=None)
bcs_dd = BoundaryConditions.dirichlet()
Lap_full = Laplacian(L_full, bcs_dd, 0.1, method='spectral', dofs=50)

# Restrict to subdomain [0.2, 0.8] with DN boundary conditions
domain_sub = IntervalDomain(0.2, 0.8)
L_sub = Lebesgue(50, domain_sub, basis=None)
bcs_dn = BoundaryConditions.mixed_dirichlet_neumann()

Lap_restricted = Lap_full.restrict(L_sub, new_bcs=bcs_dn)
# Result: Laplacian on [0.2, 0.8] with DN BCs, preserving alpha=0.1, etc.
```

### Example 2: Creating discontinuous Sobolev with custom BCs per subspace

```python
from pygeoinf.interval import Sobolev, BoundaryConditions
from pygeoinf.interval.interval_domain import IntervalDomain

# Create Sobolev space with discontinuity at x=0.5
# Left subspace: DN boundary conditions
# Right subspace: ND boundary conditions
domain = IntervalDomain(0, 1)
bcs_dn = BoundaryConditions.mixed_dirichlet_neumann()
bcs_nd = BoundaryConditions.mixed_neumann_dirichlet()

M = Sobolev.with_discontinuities(
    100,                          # Total dimension
    domain,
    [0.5],                        # Discontinuity at x=0.5
    s=1.0, k=1.0, alpha=0.1,
    bcs=bcs_dn,                   # Default (will be overridden)
    bcs_per_subspace=[bcs_dn, bcs_nd]  # Per-subspace BCs
)

# Left subspace: (0, 0.5) with DN BCs
# Right subspace: (0.5, 1] with ND BCs
```

### Example 3: Restricting a Sobolev space

```python
from pygeoinf.interval import Sobolev, Laplacian, BoundaryConditions
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue

# Create full Sobolev space
domain_full = IntervalDomain(0, 1)
L_full = Lebesgue(100, domain_full, basis=None)
bcs = BoundaryConditions.dirichlet()
Lap = Laplacian(L_full, bcs, 0.1, method='spectral', dofs=50)
M_full = Sobolev(100, domain_full, 1.0, 1.0, Lap, basis=None)

# Create target restricted space
domain_sub = IntervalDomain(0.2, 0.8)
L_sub = Lebesgue(50, domain_sub, basis=None)
Lap_sub = Laplacian(L_sub, bcs, 0.1, method='spectral', dofs=50)
M_sub = Sobolev(50, domain_sub, 1.0, 1.0, Lap_sub, basis=None)

# Restrict with new boundary conditions
bcs_new = BoundaryConditions.mixed_neumann_dirichlet()
M_restricted = M_full.restrict(M_sub, new_bcs=bcs_new)
# Result: M_restricted has Laplacian with ND BCs on [0.2, 0.8]
```

## Files Modified

1. **`pygeoinf/interval/operators.py`**
   - Added `Laplacian.restrict()` method (lines ~543-605)
   - Fixed domain containment check to use endpoint comparison

2. **`pygeoinf/interval/sobolev_space.py`**
   - Added `Sobolev.restrict()` method (lines ~127-202)
   - Enhanced `with_discontinuities()` with `bcs_per_subspace` parameter (lines ~225-398)

## Technical Notes

### Positional-Only Parameters
The `Laplacian.__init__()` signature uses positional-only parameters:
```python
def __init__(
    self,
    domain: Union[Lebesgue, Sobolev],
    boundary_conditions: BoundaryConditions,
    alpha: float = 1.0,
    /,  # Everything before this MUST be positional
    *,
    method: Literal['spectral', 'fd'] = 'spectral',
    ...
)
```

Therefore, when creating a new Laplacian in `restrict()`, we must use:
```python
Laplacian(restricted_space, bcs_to_use, self._alpha, method=...)
```
NOT:
```python
Laplacian(restricted_space, bcs_to_use, alpha=self._alpha, ...)  # ERROR!
```

### Domain Containment
`IntervalDomain` only has `contains(x)` for point containment, not domain containment.
We check containment via endpoint comparison:
```python
if not (orig_domain.a <= rest_domain.a and rest_domain.b <= orig_domain.b):
    raise ValueError(...)
```

## Future Considerations

### Potential Enhancement to `with_discontinuities()`
Currently, `with_discontinuities()` creates Laplacians directly on each subdomain.
It could potentially be refactored to:
1. Create a full-domain Laplacian
2. Use `Laplacian.restrict()` to create subspace Laplacians
3. This would ensure even more consistency

However, the current implementation is clear and works well.

### Extension to Other Operators
The restriction pattern could be extended to other operators:
- `InverseLaplacian.restrict()` (already has it)
- `BesselSobolev.restrict()`
- `BesselSobolevInverse.restrict()`

## Conclusion

The operator restriction framework provides a clean, composable way to handle domain splitting with flexible boundary conditions. By making restriction a fundamental operator capability rather than ad-hoc logic, we achieve:

- **Correctness**: Proper specification of BCs at discontinuities
- **Clarity**: Intent is explicit in the code
- **Composability**: Higher-level operations naturally leverage lower-level primitives
- **Flexibility**: Arbitrary BC combinations at discontinuities

This architectural improvement was suggested by the user's insight: "Would it not be better to define restrictions for all the operators in the first place? Then the sobolev restriction could leverage that functionality" - and it proved to be exactly the right approach.
