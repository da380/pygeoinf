Refactoring Summary: SobolevFunction to use canonical Sobolev space
==================================================================

PROBLEM:
- The original sobolev_functions.py had a redundant SobolevSpace class
- The interval.py file already contains the canonical Sobolev class
- SobolevFunction was not space-aware, leading to potential inconsistencies

SOLUTION:
✓ Removed redundant SobolevSpace class from sobolev_functions.py
✓ Refactored SobolevFunction to accept a Sobolev space object (from interval.py)
✓ Updated SobolevFunction to delegate mathematical operations to its space
✓ Fixed factory function to work with canonical Sobolev spaces
✓ Removed old BasisTransform classes (functionality is in Sobolev class)

KEY CHANGES:

1. SobolevFunction constructor now takes a Sobolev space:
   OLD: SobolevFunction(domain, order, ...)
   NEW: SobolevFunction(space, ...)

2. Factory function updated:
   ```python
   # Create space first
   space = Sobolev.create_standard_sobolev(
       order=1.5, scale=0.1, dim=50, interval=(0, np.pi)
   )

   # Create function in that space
   f = create_sobolev_function(
       space,
       evaluate_callable=lambda x: np.sin(x),
       sobolev_order=1.5
   )
   ```

3. Mathematical operations now respect space membership:
   - Inner products delegate to space.inner_product()
   - Addition/multiplication check space compatibility
   - Point evaluation uses space's mathematical restrictions

4. Eliminated code duplication:
   - No more redundant basis transforms
   - Single source of truth for Sobolev space properties
   - Consistent mathematical behavior

MATHEMATICAL CORRECTNESS:
✓ Point evaluation restricted to s > d/2 (s > 1/2 for intervals)
✓ Inner products use proper Sobolev metric from space
✓ Function operations preserve space membership
✓ Basis transformations handled by canonical Sobolev class

USAGE EXAMPLE:
```python
from interval import Sobolev
from sobolev_functions import create_sobolev_function

# Create canonical Sobolev space
space = Sobolev.create_standard_sobolev(
    order=2.0, scale=0.1, dim=64, interval=(0, 2*np.pi)
)

# Create functions in this space
f1 = create_sobolev_function(
    space,
    evaluate_callable=lambda x: np.sin(x),
    sobolev_order=2.0,
    name="sine"
)

f2 = create_sobolev_function(
    space,
    evaluate_callable=lambda x: np.cos(x),
    sobolev_order=2.0,
    name="cosine"
)

# Mathematical operations work within space
f_sum = f1 + f2  # Both functions in same space
inner_prod = f1.inner_product(f2)  # Uses space's inner product
```

BENEFITS:
1. Eliminates redundancy between SobolevSpace and Sobolev classes
2. Ensures mathematical consistency through single source of truth
3. Makes functions aware of their mathematical context
4. Improves maintainability by reducing code duplication
5. Aligns with established mathematical abstractions in interval.py

The refactoring makes SobolevFunction objects properly space-aware while
eliminating redundant code and ensuring mathematical correctness.
