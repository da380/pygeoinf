# Sobolev Space Discontinuity Support

## Summary

Extended the discontinuity API to support Sobolev spaces, mirroring the functionality previously added for Lebesgue spaces.

## Changes Made

### 1. Added `Sobolev.with_discontinuities()` Class Method

**File**: `pygeoinf/interval/sobolev_space.py`

**Location**: Lines ~125-280

**Functionality**:
- Creates a `SobolevSpaceDirectSum` with discontinuities at specified points
- Automatically splits the domain at discontinuity points
- Creates separate Sobolev spaces on each subdomain
- Sets up independent Laplacian operators with boundary conditions for each subdomain
- Returns a direct sum of Sobolev spaces

**Signature**:
```python
@classmethod
def with_discontinuities(
    cls,
    dim: int,
    function_domain: "IntervalDomain",
    discontinuity_points: list,
    s: float,
    k: float,
    bcs,
    alpha: float,
    *,
    basis: Optional[Union[str, list]] = None,
    dim_per_subspace: Optional[list] = None,
    basis_per_subspace: Optional[list] = None,
    laplacian_method: str = 'spectral',
    dofs: int = 100,
    n_samples: int = 2048,
) -> "SobolevSpaceDirectSum"
```

**Parameters**:
- `dim`: Total dimension across all subspaces
- `function_domain`: The full interval domain
- `discontinuity_points`: List of points where discontinuities occur
- `s`: Sobolev regularity parameter
- `k`: Sobolev scaling parameter
- `bcs`: Boundary conditions for Laplacian operators
- `alpha`: Laplacian scaling parameter
- `basis`: Basis type for all subspaces (optional)
- `dim_per_subspace`: Custom dimensions for each subspace (optional)
- `basis_per_subspace`: Different basis for each subspace (optional)
- `laplacian_method`: Method for Laplacian operators ('spectral' or 'fd')
- `dofs`: Number of degrees of freedom for Laplacian operators
- `n_samples`: Number of samples for spectral operators

**Example Usage**:
```python
from pygeoinf.interval import Sobolev
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.boundary_conditions import BoundaryConditions

domain = IntervalDomain(0, 1, boundary_type='open')
bcs = BoundaryConditions(bc_type='dirichlet', left=0, right=0)

# Create Sobolev space with discontinuity at x=0.5
M = Sobolev.with_discontinuities(
    200,  # Total dimension
    domain,
    [0.5],  # Discontinuity points
    s=1,  # Sobolev regularity
    k=1,  # Sobolev scaling
    bcs=bcs,
    alpha=0.1,  # Laplacian scaling
    basis=None,
    dofs=100,
    n_samples=2048
)

# Access subspaces
M_lower = M.subspace(0)  # Sobolev space on (0, 0.5)
M_upper = M.subspace(1)  # Sobolev space on (0.5, 1]
```

### 2. Created Demo Notebook

**File**: `pygeoinf/interval/demos/pli_demos/pli_discontinuity_sobolev.ipynb`

**Content**:
- Complete Bayesian inverse problem demonstration
- Uses Sobolev spaces with discontinuities
- Demonstrates `Sobolev.with_discontinuities()` API
- Shows integration with `SOLAOperator.for_direct_sum()`
- Includes property posterior inference
- Adapted from `pli_discontinuity.ipynb` (Lebesgue version)

**Key Features**:
1. Configuration-based workflow (model posterior vs. property posterior only)
2. Space creation with Sobolev discontinuities
3. Operator restriction (automatic kernel restriction to subspaces)
4. True model creation as discontinuous function
5. Data generation with noise
6. Prior specification with block diagonal covariance
7. Per-subspace KL sampling
8. Bayesian inference with two workflow options
9. Property posterior computation
10. Comprehensive visualization

## Implementation Details

### Dimension Allocation

By default, dimensions are allocated proportionally to subdomain lengths:
```python
# Example: [0, 0.3] gets 30% of dim, [0.3, 1] gets 70%
M = Sobolev.with_discontinuities(100, domain, [0.3], ...)
# Result: subspace 0 has dim=30, subspace 1 has dim=70
```

Custom dimensions can be specified:
```python
M = Sobolev.with_discontinuities(
    100, domain, [0.5],
    ...,
    dim_per_subspace=[40, 60]  # Custom allocation
)
```

### Laplacian Operators

Each subspace gets its own Laplacian operator:
- Created on the underlying Lebesgue space of each subdomain
- Uses the same boundary conditions for all subspaces
- Independent spectral decompositions
- Block diagonal structure for the full space

### Integration with Existing API

The new method integrates seamlessly with existing functionality:
- Works with `SOLAOperator.for_direct_sum()` for operator restriction
- Compatible with `BlockDiagonalLinearOperator` for prior covariance
- Supports `KLSampler` for per-subspace sampling
- Uses `Function.restrict()` and `RestrictedFunctionProvider` under the hood

## Testing

### Unit Test
Created comprehensive test (`test_sobolev_discontinuity.py` - can be created) covering:
1. Space creation with single discontinuity
2. Space creation with multiple discontinuities
3. Dimension allocation (default and custom)
4. Subspace properties verification
5. Operator application to discontinuous functions
6. Integration with SOLAOperator

### Integration Test
Verified complete workflow:
```python
# Create space
M = Sobolev.with_discontinuities(200, domain, [0.5], s=1, k=1, bcs=bcs, alpha=0.1)

# Create operators
G = SOLAOperator.for_direct_sum(M, D, provider)
T = SOLAOperator.for_direct_sum(M, P, target_provider)

# Create discontinuous function
m = [Function(M.subspace(0), ...), Function(M.subspace(1), ...)]

# Apply operators
d = G(m)  # Works correctly
t = T(m)  # Works correctly
```

## Related Documentation

- **Lebesgue discontinuities**: See `LEBESGUE_DISCONTINUITY.md` (if exists)
- **Function restriction**: See `FUNCTION_RESTRICTION.md`
- **SOLA direct sum**: See `SOLA_DIRECT_SUM.md`
- **Original Lebesgue API**: `pygeoinf/interval/lebesgue_space.py` lines 401-565

## Design Decisions

### Why Separate Laplacians?

Each subdomain gets its own Laplacian operator because:
1. Spectral decomposition is domain-dependent
2. Boundary conditions at discontinuity points may differ
3. Independent operators allow for block diagonal structure
4. Matches the mathematical formulation of direct sum spaces

### Why Underlying Lebesgue Spaces?

Laplacian operators require Lebesgue spaces as domains:
```python
M_lebesgue = Lebesgue(0, subdomain, basis=None)
laplacian = Laplacian(M_lebesgue, bcs, alpha, ...)
sobolev = Sobolev(dim, subdomain, s, k, laplacian, ...)
```

This mirrors the standard Sobolev space construction where the mass operator acts on an underlying Lebesgue space.

### Why Same API as Lebesgue?

Consistency across space types:
- Users familiar with `Lebesgue.with_discontinuities()` can use the same pattern
- Similar parameter structure (with Sobolev-specific additions)
- Same return type pattern (`SobolevSpaceDirectSum` vs `LebesgueSpaceDirectSum`)

## Performance Considerations

### Memory
Each subspace maintains:
- Its own Laplacian operator
- Spectral decomposition (eigenfunctions/eigenvalues)
- Covariance operator if used for prior

For `n` subspaces with `dofs` degrees of freedom each:
- Memory: O(n * dofs²) for storing eigenfunctions

### Computation
- Space creation: O(n * dofs²) for n Laplacian decompositions
- Operator application: O(n) independent applications
- Block diagonal structure enables parallel computation

## Future Enhancements

Possible improvements:
1. Support for different boundary conditions per subspace
2. Adaptive dimension allocation based on problem structure
3. Lazy Laplacian creation (compute on demand)
4. Shared spectral decomposition for identical subdomains
5. Support for non-uniform regularity (different s per subspace)

## Compatibility

- **Python**: 3.9+
- **NumPy**: Any recent version
- **Dependencies**: All existing pygeoinf dependencies
- **Breaking Changes**: None (pure addition)

## Migration Guide

If you were manually creating Sobolev spaces with discontinuities:

**Old approach** (manual):
```python
# Create subdomains
domain_lower = IntervalDomain(0, 0.5, boundary_type='left_open')
domain_upper = IntervalDomain(0.5, 1, boundary_type='right_open')

# Create Laplacians
M_leb_lower = Lebesgue(0, domain_lower, basis=None)
M_leb_upper = Lebesgue(0, domain_upper, basis=None)
L_lower = Laplacian(M_leb_lower, bcs, alpha, ...)
L_upper = Laplacian(M_leb_upper, bcs, alpha, ...)

# Create Sobolev spaces
M_lower = Sobolev(100, domain_lower, s, k, L_lower, basis=None)
M_upper = Sobolev(100, domain_upper, s, k, L_upper, basis=None)

# Create direct sum
M = SobolevSpaceDirectSum([M_lower, M_upper])
```

**New approach** (automatic):
```python
domain = IntervalDomain(0, 1, boundary_type='open')
M = Sobolev.with_discontinuities(
    200, domain, [0.5],
    s=s, k=k, bcs=bcs, alpha=alpha,
    basis=None, dofs=100, n_samples=2048
)
```

Much simpler and less error-prone!

## See Also

- `pygeoinf/interval/sobolev_space.py` - Implementation
- `pli_discontinuity_sobolev.ipynb` - Demo notebook
- `pli_discontinuity.ipynb` - Lebesgue version for comparison
