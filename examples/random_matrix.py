import numpy as np
from pygeoinf.linalg import EuclideanSpace, LinearOperator


import matplotlib.pyplot as plt

from pygeoinf.sphere import Lebesgue, Sobolev
from scipy.stats import norm, uniform


"""
X = EuclideanSpace(4)


def mapping(x):
    y = x.copy()
    for i, val in enumerate(y):
        y[i] /= (i + 1) ** 4
    return y

a = LinearOperator.self_adjoint(X, mapping)
A = a.matrix()
"""

X = Sobolev(64, 2, 0.1)


mu = X.sobolev_gaussian_measure(5, 0.4, 1)

m = 6
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)


a = mu.covariance
f = a.random_cholesky(10, power=2)

b = f @ f.adjoint

# b = a.random_eig_approximation(100, power=2)


u = X.dirac_representation(0, 180)

v = a(u)
w = b(u)

plt.figure()
plt.pcolormesh(v.lons(), v.lats(), v.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(w.lons(), w.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.show()
