import numpy as np
from pygeoinf.linalg import EuclideanSpace, LinearOperator, DiagonalLinearOperator


import matplotlib.pyplot as plt

from pygeoinf.sphere import Lebesgue, Sobolev
from scipy.stats import norm, uniform


X = Sobolev(64, 2, 0.1)
# X = Lebesgue(64)

mu = X.sobolev_gaussian_measure(2, 0.4, 1)

m = 6
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)


A = mu.covariance

# L, D, R = A.random_svd(10, galerkin=True)
# B = L @ D @ R

U, D = A.random_eig(10, power=2, inverse=True)
C = U @ D @ U.adjoint


# u = X.dirac_representation(0, 180)
u = mu.sample()

# v = A(u)
# w = B(u)

v = u
w = (C @ A)(u)

plt.figure()
plt.pcolormesh(v.lons(), v.lats(), v.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(w.lons(), w.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.show()
