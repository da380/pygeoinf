# Function and Provider Restriction

## Overview

This document describes the function and provider restriction functionality for working with discontinuous spaces. This provides an explicit, clean way to restrict functions and kernels defined on a full domain to subdomains when building operators on direct sum spaces.

## Motivation

When working with discontinuous functions represented as direct sums:
- Operators need kernels defined on each subdomain
- Starting with kernels on the full domain `(0, 1)` is more natural
- Explicitly restricting kernels to subdomains `(0, 0.5)` and `(0.5, 1]` is clearer than implicit domain handling

## Features

### 1. Function Restriction

Any evaluable Function can be restricted to a subdomain:

```python
# Function on full domain [0, 1]
f_full = Function(space_full, evaluate_callable=lambda x: x**2)

# Restrict to subdomain [0, 0.5]
f_restricted = f_full.restrict(space_lower)

# f_restricted evaluates to same values within subdomain
assert f_restricted.evaluate(0.3) == f_full.evaluate(0.3)
```

**Key points:**
- Only works for functions with `evaluate_callable` (not coefficient-based)
- Creates new Function on restricted space
- Original support constraints are not copied (restricted domain defines support)

### 2. Provider Restriction

Function providers can be restricted to generate functions on subdomains:

```python
# Provider on full domain
provider_full = NormalModesProvider(space_full, ...)

# Restrict provider to subdomain
provider_restricted = provider_full.restrict(space_lower)

# Get restricted kernel
kernel = provider_restricted.get_function_by_index(0)
# kernel.space.function_domain == space_lower.function_domain
```

**How it works:**
- Creates `RestrictedFunctionProvider` wrapper
- Automatically restricts each function from original provider
- Preserves all provider functionality (indexing, caching, etc.)

### 3. Automatic Restriction in SOLAOperator.for_direct_sum()

When creating operators on direct sum spaces, kernels are automatically restricted:

```python
# Create spaces
M_full = Lebesgue(100, IntervalDomain(0, 1), basis=None)
M = Lebesgue.with_discontinuities(200, IntervalDomain(0, 1), [0.5], basis=None)

# Create provider on FULL domain
provider = NormalModesProvider(M_full, ...)

# Create operator - automatic kernel restriction!
G = SOLAOperator.for_direct_sum(M, D, provider)

# Kernels are automatically restricted to each subspace:
# G.block(0,0) has kernels on (0, 0.5)
# G.block(0,1) has kernels on (0.5, 1]
```

**Benefits:**
- No need to manually create separate providers for each subspace
- Clear semantics: start with full-domain kernels, explicitly restrict
- No reliance on implicit domain checking during integration

## Implementation Details

### Function.restrict(restricted_space)

Located in `pygeoinf/interval/functions.py`:

- Validates restricted domain is subset of original domain
- Creates new Function with same `evaluate_callable`
- Sets `support=None` to let restricted domain define support
- Returns new Function on `restricted_space`

### RestrictedFunctionProvider

Located in `pygeoinf/interval/function_providers.py`:

- Implements `IndexedFunctionProvider` interface
- Wraps original provider
- `get_function_by_index(i)` gets function from original, then restricts it

### IndexedFunctionProvider.restrict(restricted_space)

Added to base provider class:

- Returns `RestrictedFunctionProvider(self, restricted_space)`
- Available for all providers (NormalModesProvider, BumpFunctionProvider, etc.)

### SOLAOperator.for_direct_sum() Enhancement

Located in `pygeoinf/interval/operators.py`:

- Detects if kernels is `IndexedFunctionProvider` or list of Functions
- For provider: calls `provider.restrict(subspace)` for each subspace
- For list: calls `func.restrict(subspace)` on each function
- Creates SOLAOperator for each subspace with restricted kernels
- Returns RowLinearOperator

## Usage Example

```python
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.interval.operators import SOLAOperator
from pygeoinf.interval.function_providers import NormalModesProvider
from pygeoinf.interval.functions import Function
import numpy as np

# Setup
domain = IntervalDomain(0, 1, boundary_type='open')
M_full = Lebesgue(100, domain, basis=None)
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis=None)
D = EuclideanSpace(20)

# Create provider on full domain
provider = NormalModesProvider(M_full, n_modes_range=(1, 50), random_state=42)

# Create operator with automatic restriction
G = SOLAOperator.for_direct_sum(M, D, provider)

# Verify kernel domains
print(G.block(0,0).domain.function_domain)  # (0.0, 0.5)
print(G.block(0,1).domain.function_domain)  # (0.5, 1.0]

# Apply to discontinuous function
m_lower = Function(M.subspace(0), evaluate_callable=lambda x: np.sin(np.pi*x))
m_upper = Function(M.subspace(1), evaluate_callable=lambda x: np.cos(np.pi*x))
m = [m_lower, m_upper]

data = G(m)  # Works perfectly!
```

## Design Rationale

This approach provides several advantages:

1. **Explicit over implicit**: Clear intention to restrict rather than relying on integration behavior
2. **Composability**: Restriction is a first-class operation on functions and providers
3. **Reusability**: Same provider can be restricted to multiple subdomains
4. **Type safety**: Restricted functions know their domain
5. **Simplicity**: `for_direct_sum()` handles restriction automatically

## Related Files

- `pygeoinf/interval/functions.py`: Function.restrict()
- `pygeoinf/interval/function_providers.py`: Provider restriction classes
- `pygeoinf/interval/operators.py`: SOLAOperator.for_direct_sum()
- `SOLA_DIRECT_SUM.md`: Documentation for SOLAOperator with direct sums

## Testing

See `test_function_restriction.py` for comprehensive tests of:
- Function restriction to subdomains
- Provider restriction
- Automatic restriction in operators
- Integration with discontinuous spaces
