"""
2. Coordinates are a capability, not a requirement.

A Hilbert space needs an inner product. It does *not* need a basis. Spaces that
have one advertise it by being a ``CoordinateSpace``, and that is what numerical
methods check before asking for a matrix.

The Gram matrix lives here too, and it is the thing that makes an inner product
differ from a dot product.
"""

import numpy as np

from pygeoinf2 import CoordinateSpace, EuclideanSpace
from pygeoinf2.symmetric_space import Lebesgue, Sobolev

rng = np.random.default_rng(0)

euclidean = EuclideanSpace(4)
lebesgue = Lebesgue((16,))  # L2 on the unit circle
sobolev = Sobolev((16,), 2.0, 0.3)  # H2 on the same grid

for name, space in [("Euclidean", euclidean), ("L2", lebesgue), ("H2", sobolev)]:
    print(
        f"{name:10s} dim={space.dim:3d}  coordinates={isinstance(space, CoordinateSpace)}"
        f"  orthonormal={space.is_orthonormal}"
    )
print()

# The Gram matrix is what separates the two. On an orthonormal basis it is the
# identity and the inner product IS the dot product; on H2 it is the Sobolev
# symbol, and the difference is the whole point of using H2.
x = sobolev.random(rng=rng)
components = sobolev.to_components(x)

print("(x, x) on L2 :", round(lebesgue.inner_product(x, x), 6))
print("(x, x) on H2 :", round(sobolev.inner_product(x, x), 6))
print(
    "dot(c, c)    :", round(float(components @ components), 6), "  <- matches L2 only"
)
print()

# The same field, the same components. Only the metric differs.
print(
    "same components on both spaces:",
    np.allclose(components, lebesgue.to_components(x)),
)
print()
print("A space with no coordinates is still a perfectly good Hilbert space;")
print("it simply cannot be handed to a method that needs a matrix.")
