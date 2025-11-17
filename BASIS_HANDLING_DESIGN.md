# Basis Handling in Discontinuous Spaces - Design Summary

## Problem Statement

When creating Lebesgue spaces with discontinuities using `Lebesgue.with_discontinuities()`, we need to handle basis functions for each subspace. The user asked:

1. **What happens if the user provides a basis to the initial space?**
2. **What should be the bases of the underlying subspaces?**
3. **What if the first space has `none` as basis, but users want to add basis to each subspace manually?**

## Solution Design

### Three Basis Configuration Options

#### 1. **Uniform Basis Across All Subspaces**

**Usage:**
```python
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis='fourier')
```

**Behavior:**
- The same basis type (as a string) is applied to all subspaces
- Each subspace gets a basis provider appropriate for its subdomain
- Supported types: `'fourier'`, `'sine'`, `'cosine'`, `'hat'`, `'none'`

**Implementation:**
```python
bases = [basis] * n_subspaces  # Replicate basis for all subspaces
```

#### 2. **Individual Basis Per Subspace**

**Usage:**
```python
M = Lebesgue.with_discontinuities(
    400, domain, [0.25, 0.5, 0.75],
    basis_per_subspace=['fourier', 'sine', 'cosine', 'fourier']
)
```

**Behavior:**
- Each subspace gets its own specified basis type
- List must have length equal to number of subspaces
- Overrides the `basis` parameter if both are provided

**Validation:**
- Length check: `len(basis_per_subspace) == n_subspaces`
- Each element can be any valid basis type string

#### 3. **Manual Basis Setting (Deferred Configuration)**

**Usage:**
```python
# Step 1: Create baseless space
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis='none')

# Step 2: Set basis providers manually
from pygeoinf.interval.providers import CustomBasisProvider
sub0 = M.subspace(0)
sub1 = M.subspace(1)
sub0.set_basis_provider(my_provider_0)
sub1.set_basis_provider(my_provider_1)
```

**Behavior:**
- All subspaces start baseless (`_basis_type='none'`)
- User has full control over basis configuration
- Useful for custom basis providers or complex setups

### Why Lists of Functions Are Not Supported

**Rejected Design:**
```python
# This raises ValueError
M = Lebesgue.with_discontinuities(
    200, domain, [0.5],
    basis=[lambda x: 1, lambda x: x, lambda x: x**2]
)
```

**Rationale:**

1. **Function Restriction Problem**: Basis functions are typically defined on the full domain. Restricting them to subdomains is non-trivial:
   - Which functions belong to which subdomain?
   - How to handle functions with support spanning multiple subdomains?
   - Orthogonality properties may not be preserved

2. **Dimension Mismatch**: If you provide N basis functions, how should they be distributed across subspaces with potentially different dimensions?

3. **Alternative Solution**: Users can:
   - Define separate function lists per subspace
   - Use `basis_per_subspace` with string types
   - Create baseless space and set basis providers manually

### Implementation Details

**Parameter Precedence:**
```python
if basis_per_subspace is not None:
    bases = list(basis_per_subspace)  # Use per-subspace specification
else:
    bases = [basis] * n_subspaces      # Replicate single basis
```

**Validation Checks:**
1. If `basis` is a list → raise `ValueError`
2. If `basis_per_subspace` provided → check length matches `n_subspaces`
3. Each subspace gets validated basis during construction

**Subspace Creation:**
```python
subspaces = [
    cls(d, subdomain, basis=b, weight=weight)
    for d, subdomain, b in zip(dims, subdomains, bases)
]
```

## API Reference

### Updated Signature

```python
@classmethod
def with_discontinuities(
    cls,
    dim: int,
    function_domain: IntervalDomain,
    discontinuity_points: list,
    /,
    *,
    basis: Optional[Union[str, list]] = None,
    weight: Optional[Callable] = None,
    dim_per_subspace: Optional[list] = None,
    basis_per_subspace: Optional[list] = None,  # NEW PARAMETER
) -> LebesgueSpaceDirectSum
```

### New Parameter

- **`basis_per_subspace`** (Optional[list]):
  - List of basis specifications, one per subspace
  - Each element is a string (basis type) or `'none'`
  - Length must equal number of subspaces
  - Overrides `basis` parameter
  - Default: `None` (use `basis` for all subspaces)

### Modified Behavior of `basis`

- **Before**: Could be string or list of functions
- **After**:
  - String: Applied to all subspaces ✓
  - List: Raises `ValueError` with helpful message ✗
  - Suggests using `basis_per_subspace` instead

## Use Cases

### Use Case 1: Homogeneous Layered Medium

Same basis everywhere (e.g., Fourier for all layers):

```python
M = Lebesgue.with_discontinuities(
    300, domain, [10, 20, 30],  # 3 interfaces
    basis='fourier'  # Same basis for all 4 layers
)
```

### Use Case 2: Heterogeneous Layered Medium

Different bases for different physical properties:

```python
# Earth model: crust (complex) vs mantle (smooth)
M = Lebesgue.with_discontinuities(
    300, IntervalDomain(0, 100), [35],
    basis_per_subspace=['fourier', 'sine']
)
# Crust (0-35 km): Fourier captures complexity
# Mantle (35-100 km): Sine for smooth variations
```

### Use Case 3: Custom Basis Providers

Maximum flexibility with manual configuration:

```python
M = Lebesgue.with_discontinuities(200, domain, [0.5], basis='none')

# Configure each subspace with specialized basis
for i in range(M.number_of_subspaces):
    sub = M.subspace(i)
    provider = create_custom_provider(sub, config[i])
    sub.set_basis_provider(provider)
```

## Testing

Comprehensive tests in `tests/test_discontinuities.py`:

1. ✓ Same basis for all subspaces
2. ✓ `basis='none'` creates baseless subspaces
3. ✓ `basis_per_subspace` with different bases
4. ✓ `basis_per_subspace` overrides `basis`
5. ✓ Wrong length `basis_per_subspace` raises error
6. ✓ List of functions raises error with helpful message
7. ✓ Multiple discontinuities with mixed bases

## Migration Impact

**Backward Compatibility:**
- Existing code using `basis='string'` continues to work unchanged
- Existing code with `basis='none'` continues to work unchanged
- No breaking changes to existing functionality

**New Capabilities:**
- Per-subspace basis configuration
- Clear error message for unsupported list-based basis
- Explicit workflow for manual basis setting

## Documentation Updates

1. ✓ Updated `DISCONTINUITY_API.md` with basis handling section
2. ✓ Added examples for all three configuration options
3. ✓ Explained why list of functions is not supported
4. ✓ Updated notebook with basis handling examples
5. ✓ Added comprehensive tests

## Summary

The solution provides **three clear pathways** for basis configuration:

| Approach | When to Use | Example |
|----------|-------------|---------|
| **Uniform** | Same basis everywhere | `basis='fourier'` |
| **Per-subspace** | Different bases needed | `basis_per_subspace=[...]` |
| **Manual** | Custom providers | `basis='none'` + manual setup |

**Key Design Principle:** Make the simple case simple (`basis='fourier'`) while supporting complex cases (`basis_per_subspace` or manual configuration) without adding confusion.
