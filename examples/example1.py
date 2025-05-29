import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearLeastSquaresInversion,
    LinearMinimumNormInversion,
    CGSolver,
    CholeskySolver,
    CGMatrixSolver,
)

from pygeoinf.homogeneous_space.sphere import Sobolev


# Set the model space.
X = Sobolev(128, 2.0, 0.2)


# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(3.0, 0.2, 1)


# Set up the forward operator.
n = 100
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)
Y = A.codomain

# Set the error distribution
sigma = 0.1
nu = GaussianMeasure.from_standard_deviation(Y, sigma)


# Set up forward problem.
forward_problem = LinearForwardProblem(A, nu)

# Make synthetic data
u, v = forward_problem.synthetic_model_and_data(mu)


# damping = 0.1
# least_squares_inversion = LinearLeastSquaresInversion(forward_problem)
# B = least_squares_inversion.least_squares_operator(damping, CGSolver(rtol=1.0e-7))

inversion = LinearMinimumNormInversion(forward_problem)
B = inversion.minimum_norm_operator(CGSolver())
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
