import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import uniform

from pygeoinf import (
    LinearBayesianInversion,
    CholeskySolver,
    GaussianMeasure,
    LinearForwardProblem,
    pointwise_variance,
)
from pygeoinf.homogeneous_space.sphere import Sobolev


# Set the model space.
X = Sobolev(64, 2, 0.1)


# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(2.0, 0.1, 1)


# Set up the forward operator.
n = 25
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)


# Set up the error distribution.
Y = A.codomain
# stds = np.random.uniform(0.05, 0.2, Y.dim)
# nu = GaussianMeasure.from_standard_deviations(Y, stds)
sigma = 0.01
nu = GaussianMeasure.from_standard_deviation(Y, sigma) if sigma > 0 else None


forward_problem = LinearForwardProblem(A, nu)
inverse_problem = LinearBayesianInversion(forward_problem, mu)


# Generate synthetic data.
u, v = forward_problem.synthetic_model_and_data(mu)


solver = CholeskySolver()
pi = inverse_problem.model_posterior_measure(v, solver)


ubar = pi.expectation
uvar = pointwise_variance(pi.low_rank_approximation(50, power=3), 20)
ustd = uvar.copy()
ustd.data = np.sqrt(uvar.data)


umax = np.max(np.abs(u.data))
uvmax = np.max(np.abs(ustd.data))

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-umax, umax])
plt.colorbar()

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), ubar.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-umax, umax])
plt.colorbar()

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), 2 * ustd.data, cmap="Reds")
plt.plot(lons, lats, "ko")
plt.clim([0, 2 * uvmax])
plt.colorbar()

w = X.dirac_representation(lats[0], lons[0])


plt.show()
