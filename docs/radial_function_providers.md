# Radial Laplacian Function Providers

This document describes the new function providers for radial Laplacian eigenfunctions.

## Overview

The radial Laplacian operator in 3D spherical coordinates is:
```
L = -d²/dr² - (2/r)d/dr = -(1/r²)d/dr(r² d/dr)
```

This operator is self-adjoint with respect to the weighted inner product:
```
⟨f,g⟩ = ∫ f(r)g(r) r² dr
```

## Available Providers

### Domain (0, R) with regularity at r=0:

1. **`RadialLaplacianDirichletProvider`**: Dirichlet BC at r=R
   - Eigenfunctions: `y_n(r) = √(2/R) sin(nπr/R) / r`
   - Eigenvalues: `λ_n = (nπ/R)²`
   - Basis type: `'radial_dirichlet'`

2. **`RadialLaplacianNeumannProvider`**: Neumann BC at r=R
   - Zero mode: `y_0(r) = √(3/R³)` (constant)
   - Nonzero modes: `y_n(r) = c_n sin(k_n r) / r` where `tan(k_n R) = k_n R`
   - Basis type: `'radial_neumann'`

### Domain (a, b) with 0 < a < b:

3. **`RadialLaplacianDDProvider`**: Dirichlet at both endpoints
   - Eigenfunctions: `y_n(r) = √(2/L) sin(nπ(r-a)/L) / r`
   - Eigenvalues: `λ_n = (nπ/L)²`
   - Basis type: `'radial_DD'`

4. **`RadialLaplacianDNProvider`**: Dirichlet at a, Neumann at b
   - Eigenfunctions: `y_n(r) = c_n sin(k_n(r-a)) / r` where `tan(k_n L) = k_n b`
   - Basis type: `'radial_DN'`

5. **`RadialLaplacianNDProvider`**: Neumann at a, Dirichlet at b
   - Eigenfunctions: `y_n(r) = c_n sin(k_n(b-r)) / r` where `tan(k_n L) = -a k_n`
   - Basis type: `'radial_ND'`

6. **`RadialLaplacianNNProvider`**: Neumann at both endpoints
   - Zero mode: `y_0(r) = √(3/(b³-a³))` (constant)
   - Nonzero modes: `y_n(r) = c_n [sin(k_n(r-a)) + ak_n cos(k_n(r-a))] / r`
   - Basis type: `'radial_NN'`

## Usage

### 1. Direct use as function providers

```python
from pygeoinf.interval import IntervalDomain, Lebesgue
from pygeoinf.interval.function_providers import RadialLaplacianDirichletProvider

# Create domain and space
domain = IntervalDomain(0.0, 2.0)
space = Lebesgue(50, domain)

# Create provider
provider = RadialLaplacianDirichletProvider(space)

# Get eigenfunctions
func0 = provider.get_function_by_index(0)
func1 = provider.get_function_by_index(1)

# Evaluate
r = np.linspace(0.01, 2.0, 100)
values = func0.evaluate(r)
```

### 2. As basis functions in Lebesgue spaces

```python
from pygeoinf.interval import IntervalDomain, Lebesgue

# Create space with radial Laplacian basis
domain = IntervalDomain(0.0, 2.0)
space = Lebesgue(50, domain, basis='radial_dirichlet')

# Get basis functions (same as provider)
func0 = space.get_basis_function(0)
func1 = space.get_basis_function(1)

# Space has proper metric for these basis functions
metric = space.metric  # Identity matrix (orthonormal basis)
```

### 3. Consistency with RadialLaplacian operator

The providers generate exactly the same eigenfunctions as the `RadialLaplacian` operator:

```python
from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
from pygeoinf.interval.operators import RadialLaplacian
from pygeoinf.interval.function_providers import RadialLaplacianDirichletProvider

domain = IntervalDomain(0.0, 2.0)
bc = BoundaryConditions(bc_type='dirichlet')
space = Lebesgue(50, domain)

# Method 1: Via operator
operator = RadialLaplacian(space, bc, method='spectral')
func_op = operator.get_eigenfunction(0)

# Method 2: Via provider
provider = RadialLaplacianDirichletProvider(space)
func_prov = provider.get_function_by_index(0)

# They are identical
assert np.allclose(func_op.evaluate(r), func_prov.evaluate(r))
```

## Examples

### Example 1: (0, R) Neumann case with zero mode

```python
from pygeoinf.interval import IntervalDomain, Lebesgue
from pygeoinf.interval.function_providers import RadialLaplacianNeumannProvider

domain = IntervalDomain(0.0, 1.0)
space = Lebesgue(50, domain)
provider = RadialLaplacianNeumannProvider(space)

# Zero mode (constant function)
y0 = provider.get_function_by_index(0)
eigenval0 = provider.get_eigenvalue(0)  # λ_0 = 0

# First nonzero mode
y1 = provider.get_function_by_index(1)
eigenval1 = provider.get_eigenvalue(1)  # λ_1 = k_1² where tan(k_1) = k_1
```

### Example 2: (a, b) Mixed boundary conditions

```python
from pygeoinf.interval import IntervalDomain, Lebesgue

# Dirichlet-Neumann on (0.5, 2.0)
domain = IntervalDomain(0.5, 2.0)
space_dn = Lebesgue(50, domain, basis='radial_DN')

# Neumann-Neumann on (0.5, 2.0)
space_nn = Lebesgue(50, domain, basis='radial_NN')

# Get zero mode for NN case
y0_nn = space_nn.get_basis_function(0)  # Constant function
```

### Example 3: Creating a Gaussian measure prior

```python
from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
from pygeoinf.interval.operators import InverseRadialLaplacian
from pygeoinf import GaussianMeasure

# Domain and boundary conditions
domain = IntervalDomain(0.0, 1.0)
bc = BoundaryConditions(bc_type='neumann')

# Create space with radial basis (optional but natural)
space = Lebesgue(100, domain, basis='radial_neumann')

# Create inverse operator as covariance
cov_op = InverseRadialLaplacian(space, bc, alpha=0.1, method='spectral')

# Create Gaussian measure
prior = GaussianMeasure(space, covariance_operator=cov_op)

# Sample from prior
sample = prior.sample()
```

## Implementation Details

### Architecture

The implementation uses a three-layer architecture:

1. **Function Providers** (`pygeoinf/interval/function_providers/radial.py`):
   - Generate eigenfunctions on demand
   - Cache computed functions
   - Implement `IndexedFunctionProvider` interface

2. **Spectrum Provider** (`pygeoinf/interval/operators/radial_operators.py`):
   - `RadialLaplacianSpectrumProvider` delegates to appropriate function provider
   - Provides eigenvalues via `RadialLaplacianEigenvalueProvider`

3. **Lebesgue Space Integration** (`pygeoinf/interval/lebesgue_space.py`):
   - `create_basis_provider()` recognizes radial basis types
   - Wraps function providers in `CustomBasisProvider`

### Benefits

1. **Modularity**: Function providers work independently of operators
2. **Reusability**: Same functions used by operators and as basis functions
3. **Consistency**: No code duplication, single source of truth
4. **Flexibility**: Can use radial eigenfunctions as basis in any Lebesgue space

## Testing

Run the test suite:
```bash
# Test operators still work
python test_radial_eigenfunctions.py

# Test providers work independently and as basis
python test_radial_providers.py
```

## References

- Radial Laplacian derivation: See `radial_operators.py` docstrings
- Boundary condition specifications: See `boundary_conditions.py`
- Function provider interface: See `function_providers/base.py`
