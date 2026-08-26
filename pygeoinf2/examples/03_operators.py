"""
3. Operators carry their adjoint, and the adjoint carries the metric.

There are no dual spaces in pygeoinf 2. An operator has an adjoint, defined by
``(A x, y)_Y == (x, A* y)_X``, and that identity is where the metric of both
spaces enters. Get the adjoint wrong and everything downstream is wrong, which
is why there is a check for it.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, LinearOperator
from pygeoinf2.spaces import Sobolev
from pygeoinf2.testing import check_operator

rng = np.random.default_rng(0)

X = Sobolev((16,), 2.0, 0.3)  # a weighted space: the metric is not the identity
Y = EuclideanSpace(3)
matrix = rng.normal(size=(3, X.dim))

# from_component_matrix means "c_{Ax} == M c_x". The adjoint is then derived,
# and it is NOT the transpose: it is G_X^-1 M^T G_Y.
A = LinearOperator.from_component_matrix(X, Y, matrix)

x, y = X.random(rng=rng), Y.random(rng=rng)
print("(A x, y)_Y  =", round(Y.inner_product(A(x), y), 10))
print("(x, A* y)_X =", round(X.inner_product(x, A.adjoint(y)), 10))
print()

# The naive answer, for comparison.
naive = X.from_components(matrix.T @ y)
print(
    "the transpose is a different vector:",
    not np.allclose(X.to_components(A.adjoint(y)), X.to_components(naive)),
)
print()

check_operator(A, rng=rng)
print("check_operator passed: linear, and the adjoint identity holds.")
print()

# Adjoints are memoised, which matters more than it looks -- see example 4.
print("A.adjoint is A.adjoint      :", A.adjoint is A.adjoint)
print("A.adjoint.adjoint is A      :", A.adjoint.adjoint is A)
