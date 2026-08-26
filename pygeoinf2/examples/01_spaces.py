"""
1. Spaces do the arithmetic; vectors are whatever the backend hands you.

The one idea here: a vector is not asked to know how to add itself. The space
knows. That is what lets a PETSc ``Vec`` or an MFEM ``GridFunction`` be a vector
of a pygeoinf space without being wrapped in anything.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace

X = EuclideanSpace(4)
rng = np.random.default_rng(0)

x = X.random(rng=rng)
y = X.random(rng=rng)

print("a vector is just a NumPy array here:", type(x).__name__, x.shape)
print()

# Arithmetic goes through the space, never through the vector.
print("x + y        ", X.add(x, y).round(4))
print("2 x          ", X.scale(2.0, x).round(4))
print("(x, y)       ", round(X.inner_product(x, y), 6))
print("||x||        ", round(X.norm(x), 6))
print()

# The in-place forms are the ones a Krylov method wants. They update where the
# backend allows it and RETURN the result, so always use the return value.
target = X.copy(x)
result = X.axpy(3.0, y, target)
print("axpy(3, y, x) == x + 3y ?", np.allclose(result, x + 3.0 * y))
print()

# Spaces are values: two spaces of the same shape are equal, and hashable.
print("EuclideanSpace(4) == EuclideanSpace(4):", X == EuclideanSpace(4))
print("usable as a dict key:", bool({X: "yes"}))
print()
print("In v1 this last line raises TypeError: unhashable type.")
