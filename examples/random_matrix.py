import numpy as np
from pygeoinf.hilbert import (
    EuclideanSpace,
    LinearOperator,
    DiagonalLinearOperator,
    CGSolver,
    CGMatrixSolver,
)


import matplotlib.pyplot as plt

from pygeoinf.sphere import Lebesgue, Sobolev
from scipy.stats import norm, uniform


X = Sobolev(32, 0.0, 0.5)

mu = X.sobolev_gaussian_measure(2.0, 0.5, 1)

m = 6
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)


u0 = mu.sample()

A = mu.covariance

u = mu.sample()
v = A(u)

P = A.random_preconditioner(100, power=0)


B = CGSolver(rtol=1.0e-10)(A, preconditioner=P)
C = CGSolver(rtol=1.0e-10)(A, preconditioner=None)
w = P(v)
z = C(v)
print(X.norm(u - w))
print(X.norm(u - z))

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(w.lons(), w.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(z.lons(), z.lats(), z.data, cmap="seismic")
plt.colorbar()


plt.show()
