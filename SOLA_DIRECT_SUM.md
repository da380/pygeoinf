# SOLAOperator with Direct Sum Spaces (Discontinuous Functions)

## Overview

When working with discontinuous functions (e.g., functions with jumps at material interfaces), the model space is represented as a direct sum of Lebesgue spaces on different subdomains. The `SOLAOperator` now supports these discontinuous spaces through the `for_direct_sum()` class method.

## Problem

Previously, if you wanted to use `SOLAOperator` with a discontinuous model space (created via `Lebesgue.with_discontinuities()`), you had two problems:

1. **Direct construction fails**: `SOLAOperator(M, D, kernels)` where `M` is a `LebesgueSpaceDirectSum` would fail with unclear error messages
2. **Manual construction was complex**: You had to manually create a `RowLinearOperator` with one `SOLAOperator` per subspace

## Solution

Use the **`SOLAOperator.for_direct_sum()`** class method:

```python
# Create a discontinuous model space
M = Lebesgue.with_discontinuities(
    dim=200,
    function_domain=IntervalDomain(0, 1, boundary_type='open'),
    discontinuity_points=[0.5],
    basis=None
)

# M is now a LebesgueSpaceDirectSum with 2 subspaces:
# - M.subspace(0): Lebesgue space on (0, 0.5)
# - M.subspace(1): Lebesgue space on (0.5, 1)

# Create data space
D = EuclideanSpace(50)

# Define kernels (as callables, Functions, or via a Provider)
kernels = [lambda x, i=i: np.sin((i+1) * np.pi * x) for i in range(50)]

# Create SOLA operator for the direct sum space
G = SOLAOperator.for_direct_sum(M, D, kernels)

# G is a RowLinearOperator that can act on discontinuous functions
# Discontinuous functions are lists: [f_lower, f_upper]
data = G([f_lower, f_upper])
```

## How It Works

The `for_direct_sum()` method:

1. **Validates** that the domain is a `HilbertSpaceDirectSum`
2. **Creates** one `SOLAOperator` per subspace
3. **Returns** a `RowLinearOperator` that:
   - Takes a list of functions `[f_1, f_2, ..., f_n]` (one per subspace)
   - Applies each subspace operator: `G_i(f_i)`
   - Sums the results: `G([f_1, ..., f_n]) = G_1(f_1) + ... + G_n(f_n)`

Each kernel is integrated independently over each subdomain:

```
G([f_1, f_2]) = [∫_{Ω_1} f_1(x)·k_i(x) dx + ∫_{Ω_2} f_2(x)·k_i(x) dx]_{i=1}^{N_d}
```

## Kernel Definition

**Important**: Kernels should be defined on the **full undivided domain**, not on the direct sum space.

### Option 1: Simple Callables (Recommended)

```python
# Define kernels as Python callables
kernels = []
for i in range(N_d):
    freq = (i + 1) * np.pi
    if i % 2 == 0:
        def kernel(x, f=freq):
            return np.sin(f * x)
    else:
        def kernel(x, f=freq):
            return np.cos(f * x)
    kernels.append(kernel)

G = SOLAOperator.for_direct_sum(M, D, kernels)
```

### Option 2: Function Objects

```python
# Create a Lebesgue space on the FULL domain (not the direct sum)
M_full = Lebesgue(dim=200, function_domain=IntervalDomain(0, 1))

# Create Function objects
kernels = [
    Function(M_full, evaluate_callable=lambda x, i=i: np.sin((i+1)*np.pi*x))
    for i in range(N_d)
]

G = SOLAOperator.for_direct_sum(M, D, kernels)
```

### Option 3: Function Provider (Advanced)

```python
# Create a provider for the FULL undivided domain
M_full = Lebesgue(dim=200, function_domain=IntervalDomain(0, 1))

provider = NormalModesProvider(
    M_full,  # NOT the direct sum M!
    n_modes_range=(1, 50),
    coeff_range=(-5, 5),
    gaussian_width_percent_range=(1, 5),
    freq_range=(0.1, 20),
    random_state=42
)

G = SOLAOperator.for_direct_sum(M, D, provider)
```

**Why use the full domain?** The kernels need to be evaluable at any point in the domain. During integration, each subspace operator will evaluate the kernel only over its subdomain, so the kernel should be defined on at least that subdomain (which is part of the full domain).

## Complete Example

```python
import numpy as np
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.interval.operators import SOLAOperator
from pygeoinf.interval.functions import Function

# 1. Create discontinuous model space
function_domain = IntervalDomain(0, 1, boundary_type='open')
M = Lebesgue.with_discontinuities(
    dim=200,
    function_domain=function_domain,
    discontinuity_points=[0.5],  # Jump at x=0.5
    basis=None
)

# 2. Create data space
N_d = 50
D = EuclideanSpace(N_d)

# 3. Define kernel functions (simple trigonometric functions)
kernels = []
for i in range(N_d):
    freq = (i + 1) * np.pi
    if i % 2 == 0:
        def kernel(x, f=freq):
            return np.sin(f * x)
    else:
        def kernel(x, f=freq):
            return np.cos(f * x)
    kernels.append(kernel)

# 4. Create SOLA operator for direct sum
G = SOLAOperator.for_direct_sum(M, D, kernels)

# 5. Create a discontinuous test function
# (constant=1 on lower subdomain, linear=2x on upper subdomain)
f_lower = Function(
    M.subspace(0),
    evaluate_callable=lambda x: 1.0 if np.isscalar(x) else np.ones_like(x)
)
f_upper = Function(
    M.subspace(1),
    evaluate_callable=lambda x: 2 * x
)

# 6. Apply operator
data = G([f_lower, f_upper])  # Result: array of shape (50,)

print(f"Data shape: {data.shape}")
print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
```

## API Reference

### `SOLAOperator.for_direct_sum()`

```python
@staticmethod
def for_direct_sum(
    domain,              # HilbertSpaceDirectSum (e.g., LebesgueSpaceDirectSum)
    codomain,            # EuclideanSpace
    kernels,             # IndexedFunctionProvider or List[Function or Callable]
    cache_kernels=False, # Cache kernels for repeated access
    integration_method='simpson',  # Integration method
    n_points=1000        # Number of integration points
) -> RowLinearOperator
```

**Parameters:**
- `domain`: Direct sum space (created via `Lebesgue.with_discontinuities()`)
- `codomain`: Euclidean space defining output dimension
- `kernels`:
  - `IndexedFunctionProvider` (e.g., `NormalModesProvider`) on full domain
  - `List[Function]`: Function objects on full domain
  - `List[Callable]`: Simple Python callables (will be wrapped in Functions)
- `cache_kernels`: Whether to cache kernel evaluations
- `integration_method`: `'simpson'` or `'trapezoid'`
- `n_points`: Number of points for numerical integration

**Returns:**
- `RowLinearOperator`: Operator mapping `[f_1, ..., f_n] → data`

**Raises:**
- `TypeError`: If `domain` is not a `HilbertSpaceDirectSum`
- `TypeError`: If subspaces are not `Lebesgue` or `Sobolev` spaces

## Error Handling

### Direct Construction Attempt

If you try to use the regular constructor with a direct sum space:

```python
M = Lebesgue.with_discontinuities(...)
try:
    G = SOLAOperator(M, D, kernels)  # This will fail!
except TypeError as e:
    print(e)
    # "SOLAOperator constructor does not directly support HilbertSpaceDirectSum domains.
    #  Use the class method SOLAOperator.for_direct_sum() instead."
```

This gives you a clear message about what to do instead.

## Integration with Bayesian Inference

The `RowLinearOperator` returned by `for_direct_sum()` is a full `LinearOperator`, so it works seamlessly with the Bayesian inference framework:

```python
# Create discontinuous model space and forward operator
M = Lebesgue.with_discontinuities(200, domain, [0.5])
G = SOLAOperator.for_direct_sum(M, D, provider)

# Use in Bayesian inference (same as before!)
forward_problem = LinearForwardProblem(G, data_error_measure=gaussian_D)
bayesian_inference = LinearBayesianInference(forward_problem, M_prior, T)
posterior = bayesian_inference.model_posterior_measure(data, solver)
```

The operator automatically handles the fact that the model space is a direct sum, and all the matrix operations, adjoints, etc. work correctly.

## Design Notes

### Why Not Automatic Detection?

We could have made `SOLAOperator.__init__()` automatically detect direct sum spaces and call `for_direct_sum()`. However:

1. **Explicit is better than implicit**: Users should know they're working with discontinuous functions
2. **Clearer error messages**: The current approach gives helpful guidance
3. **Type safety**: Separating the methods makes the type signatures clearer

### Why RowLinearOperator?

The mathematical structure is:

```
G: M₁ ⊕ M₂ ⊕ ... ⊕ Mₙ → D
```

This is naturally represented as a row operator:

```
G = [G₁ | G₂ | ... | Gₙ]
```

where each `Gᵢ: Mᵢ → D` is a standard `SOLAOperator`.

The row structure means:
- Input: List of functions `[f₁, f₂, ..., fₙ]`
- Output: Single data vector `G₁(f₁) + G₂(f₂) + ... + Gₙ(fₙ)`

This matches the physics: measurements integrate over the entire domain, including all subdomain contributions.

## See Also

- `DISCONTINUITY_API.md`: Guide to creating discontinuous spaces with `Lebesgue.with_discontinuities()`
- `BASIS_HANDLING_DESIGN.md`: How to configure basis functions for each subspace
- `direct_sum.py`: Implementation of `RowLinearOperator` and block operators
