import matplotlib.pyplot as plt

import numpy as np
from numpy import pi


import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

X = Sobolev(32, 2, 0.5)
Y = Sobolev(32, 1, 0.1)


Z = inf.HilbertSpaceDirectSum([Y, Y])

w = X.random()

X_base = X.underlying_space
Z_base = inf.HilbertSpaceDirectSum(
    [subspace.underlying_space for subspace in Z.subspaces]
)

print(X_base)

A_L2 = inf.LinearOperator(
    X_base, Z_base, lambda u: [u, 2 * u], adjoint_mapping=lambda v: v[0] + 2 * v[1]
)


u = X_base.random()
v = Z_base.random()

lhs = Z_base.inner_product(A_L2(u), v)
rhs = X_base.inner_product(u, A_L2.adjoint(v))

print(lhs, rhs)


A = inf.LinearOperator.from_formal_adjoint(X, Z, A_L2)

u = X.random()
v = Z.random()

lhs = Z.inner_product(A(u), v)
rhs = X.inner_product(u, A.adjoint(v))

print(lhs, rhs)
