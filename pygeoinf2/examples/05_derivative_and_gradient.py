"""
5. A derivative is not a gradient. This is the one to read twice.

An adjoint solve gives you dJ/dm_i: the components of the *derivative*, in the
basis you chose. Step along that and you are not going downhill in the metric
you chose -- you are going downhill in an artefact of the discretisation, and
the direction changes when the mesh does.

The gradient is the derivative with the inverse metric applied. In pygeoinf 2
the adjoint is where that happens, and it is the only place it happens.
"""

import numpy as np

from pygeoinf2 import LinearFunctional
from pygeoinf2.symmetric_space import Sobolev
from pygeoinf2.testing import check_gradient
from pygeoinf2.algebra.operators import Functional

rng = np.random.default_rng(0)
X = Sobolev((16,), 2.0, 0.3)

# What an adjoint code returns: an array of partial derivatives.
g = rng.normal(size=X.dim)

f = LinearFunctional.from_derivative_components(X, g)

print("two readings of one object:")
print("  f.matrix()      -> the derivative, the row vector g")
print("  f.adjoint(1.0)  -> the gradient, the Riesz representer")
print()
print("  derivative components:", f.matrix().ravel()[:4].round(4), "...")
print("  gradient components  :", X.to_components(f.representer)[:4].round(4), "...")
print(
    "  they differ by G^-1  :",
    np.allclose(X.to_components(f.representer), X.solve_gram(g)),
)
print()

# The distinction is invisible on an orthonormal basis, which is exactly why
# the error survives: it does not show up in the toy problem.
from pygeoinf2 import EuclideanSpace  # noqa: E402

flat = EuclideanSpace(X.dim)
flat_functional = LinearFunctional.from_derivative_components(flat, g)
print(
    "on a Euclidean space the two coincide:",
    np.allclose(flat.to_components(flat_functional.representer), g),
)
print()

# And the check that catches it. Supply the derivative where a gradient was
# wanted and this fails by exactly a factor of the Gram matrix.
matrix = 0.5 * (lambda M: M + M.T)(rng.normal(size=(X.dim, X.dim)))


def value(x):
    c = X.to_components(x)
    return 0.5 * float(c @ matrix @ c)


right = Functional.from_callables(
    X,
    value,
    derivative=lambda x: LinearFunctional.from_derivative_components(
        X, matrix @ X.to_components(x)
    ),
)
wrong = Functional.from_callables(
    X, value, gradient=lambda x: X.from_components(matrix @ X.to_components(x))
)

check_gradient(right, X.random(rng=rng), rng=rng)
print("check_gradient passes on the correct functional.")
try:
    check_gradient(wrong, X.random(rng=rng), rng=rng)
except AssertionError:
    print("check_gradient FAILS when the derivative is passed as a gradient.")
