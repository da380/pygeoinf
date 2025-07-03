"""
Implementation Summary: SobolevFunction Basis Functions

This document summarizes the implementation of SobolevFunction basis functions
in the create_standard_sobolev method.

## Overview

The `create_standard_sobolev` method in `interval_space.py` has been enhanced to:
1. Create standard Sobolev spaces with different basis types
2. Return basis functions as `SobolevFunction` instances
3. Support point evaluation and domain checking
4. Provide proper mathematical representations

## Key Changes

### 1. Enhanced create_standard_sobolev method

The method now:
- Creates the Sobolev space with appropriate transformations
- Generates basis functions as `SobolevFunction` instances
- Stores them in the space for later access
- Supports three basis types: 'fourier', 'sine', 'chebyshev'

### 2. New methods added to Sobolev class

- `_create_basis_functions(basis_type)`: Creates basis functions as SobolevFunction instances
- `get_basis_functions()`: Returns the list of basis functions
- `basis_functions` property: Convenient access to basis functions

### 3. Basis function implementations

#### Fourier basis (cosine functions):
- f_k(x) = cos(k * π * (x - a) / L) for k = 0, 1, 2, ...
- Suitable for periodic-like functions
- Includes the constant function (k=0)

#### Sine basis:
- f_k(x) = sin(k * π * (x - a) / L) for k = 1, 2, 3, ...
- Satisfies zero boundary conditions
- Suitable for functions that vanish at endpoints

#### Chebyshev basis:
- f_k(x) = T_k(t) where t = 2*(x-a)/L - 1
- T_k are Chebyshev polynomials of the first kind
- Optimal for approximation on intervals

## Mathematical Properties

### Point Evaluation
- Point evaluation is only well-defined for s > d/2
- For intervals (d=1), requires s > 1/2
- The implementation checks this condition and raises errors appropriately

### Domain Checking
- Each SobolevFunction knows its domain (IntervalDomain)
- Can check if evaluation points are within the domain
- Supports both single points and arrays of points

### Sobolev Scaling
- Each basis function is associated with a scaling factor
- Fourier: (1 + (scale * k * π / L)²)^order
- Sine: (1 + (scale * (k+1) * π / L)²)^order
- Chebyshev: (1 + (scale * k)²)^order

## Usage Examples

```python
from pygeoinf.other_space.interval_space import Sobolev

# Create a Sobolev space with Fourier basis
space = Sobolev.create_standard_sobolev(
    order=1.5,    # Sobolev order s > 0.5 for point evaluation
    scale=1.0,    # Length scale parameter
    dim=5,        # Number of basis functions
    interval=(0, 1),      # Domain interval
    basis_type='fourier'  # Basis type
)

# Access basis functions
basis_functions = space.basis_functions

# Each basis function is a SobolevFunction instance
first_basis = basis_functions[0]
print(f"Function name: {first_basis.name}")
print(f"Sobolev order: {first_basis.sobolev_order}")
print(f"Domain: {first_basis.domain}")

# Evaluate at points
x_test = 0.5
value = first_basis.evaluate(x_test)
# or equivalently: value = first_basis(x_test)

# Evaluate at multiple points
x_array = np.linspace(0, 1, 100)
values = first_basis.evaluate(x_array)
```

## Testing

The implementation includes comprehensive tests:
- `test_sobolev_functions.py`: Verifies that all basis functions are SobolevFunction instances
- `demo_sobolev_functions.py`: Demonstrates usage and properties
- Tests all three basis types
- Verifies point evaluation and domain checking

## Integration with Existing Code

The implementation is fully backward compatible:
- Existing `to_coefficient` and `from_coefficient` methods unchanged
- All existing functionality preserved
- New basis functions add functionality without breaking existing code

## Future Extensions

The framework supports easy extension to:
- Additional basis types (wavelets, splines, etc.)
- Custom basis functions
- Basis functions on different domains
- Boundary conditions and constraints
"""
