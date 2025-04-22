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


X = Sobolev(32, 1.5, 0.1)

mu = X.sobolev_gaussian_measure(2.0, 0.5, 1)

m = 6
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)


A = mu.covariance


U, D = A.random_eig(100, power=0)
B = U @ D @ U.adjoint

I = X.inclusion
V = I @ I.adjoint @ U
C = V @ D.inverse @ V.adjoint


u = mu.sample()
v = A(u)
w = B(u)

D = CGSolver(atol=1.0e-7)(A, preconditioner=X.identity, x0=C(v))
z = D(v)

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(u.lons(), v.lats(), v.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(w.lons(), w.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(z.lons(), z.lats(), z.data, cmap="seismic")
plt.colorbar()

plt.show()
