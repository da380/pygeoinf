import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from pygeoinf.hilbert import CGSolver
from pygeoinf.sphere import Sobolev
from pygeoinf.optimisation import LeastSquaresInversion
import time


# Set the model space.
X = Sobolev(128, 2.0, 0.1)


# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(3.0, 0.1, 1)


# Set up the forward operator.
n = 500
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)

# Set up the error distribution.
Y = A.codomain
sigma = 0.1
nu = Y.standard_gaussisan_measure(sigma)

u = mu.sample()
umax = np.max(np.abs(u.data))

v = A(u)

damping = 0.1
least_squares_inversion = LeastSquaresInversion(A, nu)


B = least_squares_inversion.least_squares_operator(
    damping, solver=CGSolver(rtol=1.0e-7)
)

w = B(v)

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-umax, umax])
plt.colorbar()

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), w.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-umax, umax])
plt.colorbar()

plt.show()
