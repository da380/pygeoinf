import numpy as np
from pygeoinf.linalg import (
    EuclideanSpace,
    LinearOperator,
    DiagonalLinearOperator,
    CGSolver,
    CGMatrixSolver,
)


import matplotlib.pyplot as plt

from pygeoinf.sphere import Lebesgue, Sobolev
from scipy.stats import norm, uniform


X = Sobolev(4, 2, 0.1)
# X = Lebesgue(64)

mu = X.sobolev_gaussian_measure(2, 0.6, 1)

m = 6
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)


A = mu.covariance


U, D = A.random_eig(4, power=2)

IX = X.inclusion
V = IX @ IX.adjoint @ U
B = V @ D.inverse @ V.adjoint

M = X.identity - A @ B


plt.matshow(M.matrix(dense=True, galerkin=True))
plt.colorbar()
plt.show()


"""


def pre(x):
    x1 = U(V.adjoint(x))
    x0 = x - x1
    return x0 + x1


P = LinearOperator.self_adjoint(X, pre)
# P = X.identity()


u = mu.sample()


v = A(u)

solver = CGSolver(atol=1e-20)
C = solver(A, preconditioner=P)
# w = C(v)
w = P(v)


plt.figure()
plt.pcolormesh(u.lons(), v.lats(), v.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(w.lons(), w.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.show()

"""
