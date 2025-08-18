import matplotlib.pyplot as plt

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev


X = Sobolev(128, 2, 0.5)


Y = inf.HilbertSpaceDirectSum([X, X])

A = inf.LinearOperator(
    X, Y, lambda x: [x, x], formal_adjoint_mapping=lambda y: y[0] + y[1]
)

x = X.random()
y = Y.random()

lhs = Y.inner_product(y, A(x))
rhs = X.inner_product(A.adjoint(y), x)

print(lhs, rhs)
