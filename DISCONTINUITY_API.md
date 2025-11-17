# Discontinuity Functionality for pygeoinf

This document describes the new API for modeling functions with jump discontinuities in the `pygeoinf` library.

## Overview

When modeling physical systems with discontinuous properties (e.g., material interfaces, layered media), we need to represent function spaces that allow for jump discontinuities. The new API makes this simple by providing tools to:

1. Split interval domains at discontinuity points
2. Restrict Lebesgue spaces to subintervals
3. Create direct sum spaces with automatic discontinuity handling

## API Reference

### `IntervalDomain.split_at_discontinuities(discontinuity_points)`

Splits an interval domain into non-overlapping subintervals separated by discontinuity points.

**Parameters:**
- `discontinuity_points` (list): Points where discontinuities occur, must lie in the interior (a, b)

**Returns:**
- List of `IntervalDomain` objects representing the subintervals

**Example:**
```python
from pygeoinf.interval import IntervalDomain

domain = IntervalDomain(0, 1, boundary_type='closed')
subdomains = domain.split_at_discontinuities([0.5])
# Returns: [IntervalDomain(0, 0.5), IntervalDomain(0.5, 1)]

# Multiple discontinuities
subdomains = domain.split_at_discontinuities([0.25, 0.5, 0.75])
# Returns 4 subdomains
```

### `Lebesgue.restrict_to_subinterval(subdomain)`

Creates a new Lebesgue space restricted to a subinterval of the function domain.

**Parameters:**
- `subdomain` (IntervalDomain): The subinterval to restrict to

**Returns:**
- New `Lebesgue` space defined on the subinterval

**Example:**
```python
from pygeoinf.interval import IntervalDomain, Lebesgue

domain = IntervalDomain(0, 1)
space = Lebesgue(100, domain, basis='none')

subdomain = IntervalDomain(0, 0.5)
restricted = space.restrict_to_subinterval(subdomain)
# restricted.dim == 100, restricted.function_domain == subdomain
```

### `Lebesgue.with_discontinuities(dim, domain, discontinuity_points, **kwargs)`

**Class method** that creates a `LebesgueSpaceDirectSum` with discontinuities at specified points.

**Parameters:**
- `dim` (int): Total dimension across all subspaces
- `domain` (IntervalDomain): The full interval domain
- `discontinuity_points` (list): Points where discontinuities occur
- `basis` (optional): Basis type for **all** subspaces (string like 'fourier', 'sine', 'none'). Ignored if `basis_per_subspace` is provided. **Cannot be a list of functions** - raises error.
- `weight` (optional): Weight function for inner product
- `dim_per_subspace` (optional list): Custom dimension allocation. If None, dimensions are allocated proportionally to subinterval lengths
- `basis_per_subspace` (optional list): Basis specification for **each** subspace individually. Must have length equal to number of subspaces. Overrides `basis` parameter.

**Returns:**
- `LebesgueSpaceDirectSum` with subspaces for each interval segment

**Examples:**

```python
from pygeoinf.interval import IntervalDomain, Lebesgue

domain = IntervalDomain(0, 1, boundary_type='closed')

# Single discontinuity - dimensions allocated proportionally
M = Lebesgue.with_discontinuities(200, domain, [0.5])
# Creates: LebesgueSpaceDirectSum([Lebesgue(100, [0, 0.5]),
#                                    Lebesgue(100, [0.5, 1])])

# Multiple discontinuities
M = Lebesgue.with_discontinuities(400, domain, [0.25, 0.5, 0.75])
# Creates 4 subspaces of 100 dimensions each

# Custom dimension allocation
M = Lebesgue.with_discontinuities(
    200, domain, [0.5],
    dim_per_subspace=[80, 120]
)
# First subspace gets 80 dims, second gets 120

# Same basis for all subspaces
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis='fourier')
# Both subspaces use Fourier basis

# Different basis per subspace
M = Lebesgue.with_discontinuities(
    200, domain, [0.5],
    basis_per_subspace=['fourier', 'sine']
)
# First subspace uses Fourier, second uses sine

# No basis initially - set manually later
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis='none')
# Later: M.subspace(0).set_basis_provider(my_provider)

# Access subspaces
M.number_of_subspaces  # 2
M.subspace(0)  # First Lebesgue space
M.subspace(1)  # Second Lebesgue space
```

## Use Cases

### 1. Material Interfaces

Model a layered medium with different properties in each layer:

```python
domain = IntervalDomain(0, 10)  # Depth from 0 to 10 km
interfaces = [3.5, 7.2]  # Layer boundaries at 3.5 km and 7.2 km

# Create space with discontinuities at interfaces
M = Lebesgue.with_discontinuities(300, domain, interfaces)
# Now M has 3 subspaces for the 3 layers
```

### 2. Piecewise Functions

Define functions that can jump at specific locations:

```python
domain = IntervalDomain(0, 1)
M = Lebesgue.with_discontinuities(100, domain, [0.3, 0.7])

# Functions in M can be discontinuous at x=0.3 and x=0.7
# Each function is represented as a list [f1, f2, f3] where:
#   f1 is defined on [0, 0.3)
#   f2 is defined on (0.3, 0.7)
#   f3 is defined on (0.7, 1]
```

### 3. Inverse Problems with Discontinuities

The API integrates seamlessly with the Bayesian inverse problem framework:

```python
from pygeoinf.interval import Lebesgue, IntervalDomain
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInference

# Model space with discontinuity
domain = IntervalDomain(0, 1)
M = Lebesgue.with_discontinuities(200, domain, [0.5])

# Define forward operator, data, prior, etc.
# The rest of the workflow is identical to continuous case!
```

## Basis Handling

### Overview

When creating spaces with discontinuities, you have several options for handling basis functions:

1. **Same basis for all subspaces**: Use `basis='string'` parameter
2. **Different bases per subspace**: Use `basis_per_subspace=[...]` parameter
3. **No basis initially**: Use `basis='none'` and set manually later
4. **List of basis functions**: **NOT SUPPORTED** - raises error

### Why Lists of Functions Are Not Supported

When you provide a list of basis functions like `basis=[func1, func2, ...]`, these functions are typically defined on the full domain. Restricting these functions to subdomains is non-trivial:

- Functions may have compact support that doesn't align with subdomain boundaries
- Orthogonality properties may not be preserved after restriction
- It's unclear how to distribute functions across subspaces

Instead, use `basis_per_subspace` to specify the basis type for each subdomain, or create the space with `basis='none'` and set basis providers manually.

### Option 1: Same Basis for All Subspaces

Use the `basis` parameter with a string value:

```python
M = Lebesgue.with_discontinuities(
    200, domain, [0.5],
    basis='fourier'  # Both subspaces get Fourier basis
)
```

**Supported basis types:**
- `'fourier'`: Fourier basis (sine + cosine)
- `'sine'`: Sine basis
- `'cosine'`: Cosine basis
- `'hat'`: Hat functions (finite elements)
- `'none'`: No basis

Each subspace automatically gets the appropriate basis for its subdomain.

### Option 2: Different Bases Per Subspace

Use `basis_per_subspace` to specify different bases:

```python
M = Lebesgue.with_discontinuities(
    400, domain, [0.25, 0.5, 0.75],
    basis_per_subspace=['fourier', 'sine', 'cosine', 'fourier']
)

# Now each subspace has its specified basis:
M.subspace(0)  # Fourier basis on [0, 0.25)
M.subspace(1)  # Sine basis on (0.25, 0.5)
M.subspace(2)  # Cosine basis on (0.5, 0.75)
M.subspace(3)  # Fourier basis on (0.75, 1]
```

**Note**: `basis_per_subspace` overrides the `basis` parameter if both are provided.

### Option 3: Manual Basis Setting

For maximum flexibility, create a baseless space and set basis providers manually:

```python
from pygeoinf.interval.providers import CustomBasisProvider
from pygeoinf.interval.function_providers import FourierFunctionProvider

# Step 1: Create baseless space
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis='none')

# Step 2: Create custom basis providers for each subspace
sub0 = M.subspace(0)
sub1 = M.subspace(1)

provider0 = CustomBasisProvider(
    sub0,
    functions_provider=FourierFunctionProvider(sub0),
    orthonormal=True
)

provider1 = CustomBasisProvider(
    sub1,
    functions_provider=YourCustomProvider(sub1),
    orthonormal=False
)

# Step 3: Set basis providers
sub0.set_basis_provider(provider0)
sub1.set_basis_provider(provider1)
```

This workflow is useful when:
- You need custom basis functions
- You want different basis configurations (e.g., different numbers of modes)
- You're using specialized basis providers

### Example: Material Interface with Different Bases

```python
# Model Earth's crust-mantle boundary at 35 km depth
domain = IntervalDomain(0, 100)  # 0 to 100 km depth
interface = 35  # Crust-mantle boundary

# Use different bases for different physical properties
M = Lebesgue.with_discontinuities(
    300, domain, [interface],
    basis_per_subspace=['fourier', 'sine']
)

# Crust (0-35 km): Fourier basis captures complex shallow structure
# Mantle (35-100 km): Sine basis for smoother deeper structure
```

## Implementation Details

### Dimension Allocation

When `dim_per_subspace` is not provided, dimensions are allocated proportionally to subinterval lengths:

- Each subinterval gets `floor(total_dim * length_i / total_length)` dimensions
- Remaining dimensions are distributed to the largest subintervals first

### Boundary Types

The `split_at_discontinuities` method automatically determines appropriate boundary types:
- First subinterval: inherits left boundary, right is open
- Last subinterval: left is open, inherits right boundary
- Middle subintervals: both boundaries open (to avoid overlap)

## Migration Guide

### Old Way (Manual Construction)

```python
# Old: Manual construction
domain_lower = IntervalDomain(0, 0.5, boundary_type='left_open')
domain_upper = IntervalDomain(0.5, 1, boundary_type='right_open')
M_lower = Lebesgue(100, domain_lower, basis=None)
M_upper = Lebesgue(100, domain_upper, basis=None)
M = LebesgueSpaceDirectSum([M_lower, M_upper])
```

### New Way (Discontinuity API)

```python
# New: Using discontinuity API
domain = IntervalDomain(0, 1, boundary_type='open')
M = Lebesgue.with_discontinuities(200, domain, [0.5])
```

The new API is:
- **More concise**: One line instead of five
- **Less error-prone**: No manual boundary type management
- **More flexible**: Easy to add more discontinuities
- **Clearer intent**: Code explicitly states "I want discontinuities at these points"

## Testing

Comprehensive unit tests are provided in `tests/test_discontinuities.py`. Run with:

```bash
pytest tests/test_discontinuities.py -v
```

## Future Extensions

Possible future enhancements:
- Automatic basis function restriction for subintervals
- Support for derivative discontinuities (C^0 continuity)
- Visualization tools for discontinuous functions
- Optimization of operators on direct sum spaces

## References

- See `examples/pli_discontinuity.ipynb` for a complete worked example
- Mathematical background: Direct sum Hilbert spaces (Section 2.3 of pygeoinf documentation)
