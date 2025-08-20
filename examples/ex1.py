import matplotlib.pyplot as plt

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

order = 4
X = Sobolev(64, order, 0.5, radius=10)
Y = Sobolev(64, order - 2, 1.0, radius=1)

mu = X.heat_gaussian_measure(0.5, 1)
nu = Y.heat_gaussian_measure(0.5, 1)

w = nu.sample()


def mapping(u):
    ulm = u.expand()
    for l in range(u.lmax + 1):
        fac = l * (l + 1)
        ulm.coeffs[:, l, :] *= fac
    return w * ulm.expand(grid=X.grid, extend=X.extend)


def formal_adjoint_mapping(v):
    z = w * v
    zlm = z.expand()
    for l in range(u.lmax + 1):
        fac = l * (l + 1)
        zlm.coeffs[:, l, :] *= fac
    return zlm.expand(grid=X.grid, extend=X.extend)


A = inf.LinearOperator(X, Y, mapping, formal_adjoint_mapping=formal_adjoint_mapping)


u = mu.sample()
v = nu.sample()

w = A(v)

lhs = Y.inner_product(v, A(u))
rhs = X.inner_product(A.adjoint(v), u)

print(lhs, rhs)
