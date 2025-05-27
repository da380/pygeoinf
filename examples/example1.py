import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearLeastSquaresInversion,
    CGSolver,
)

from pygeoinf.geometry.sphere import Sobolev


# Set the model space.
X = Sobolev(128, 2.0, 0.4)


# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(3.0, 0.1, 1)


# Set up the forward operator.
n = 25
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)

# Set up the error distribution.
Y = A.codomain
standard_deviation = 0.1
nu = GaussianMeasure.from_standard_deviation(Y, standard_deviation)


forward_problem = LinearForwardProblem(A, nu)

u = mu.sample()
v = forward_problem.data_measure(u).sample()


least_squares_inversion = LinearLeastSquaresInversion(forward_problem)


damping = 0.1
B = least_squares_inversion.least_squares_operator(damping, CGSolver(rtol=1.0e-7))

w = B(v)


umax = np.max(np.abs(u.data))

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
