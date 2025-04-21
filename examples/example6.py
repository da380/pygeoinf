import numpy as np
import pygeoinf.linalg as la
from scipy.stats import chi2
from pygeoinf.linalg import GaussianMeasure, CholeskySolver
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.bayesian import BayesianInversion, BayesianInference
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 1.5, 0.1)

# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(2, 0.5, 1)


# Set up the forward operator.
n = 50
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)


# Set up the error distribution.
Y = A.codomain
sigma = 0.01
nu = Y.standard_gaussisan_measure(sigma)

# Set up the inference problem
problem = BayesianInversion(A, mu, nu)

# Generate synthetic data.
u = mu.sample()
v = problem.data_measure(u).sample()
pi = problem.model_posterior_measure(v, solver=CholeskySolver()).low_rank_approximation(
    20, power=2
)

ub = pi.expectation

ns = 20
uvar = X.zero
for _ in range(n):
    us = pi.sample()
    uvar += (us - ub) * (us - ub)
uvar /= ns - 1
uvar.data = np.sqrt(uvar.data)

umax = np.max(np.abs(u.data))

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-umax, umax])
plt.colorbar()


plt.figure()
plt.pcolormesh(u.lons(), u.lats(), ub.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-umax, umax])
plt.colorbar()

uvar *= 100 / umax
uvmax = np.max(np.abs(uvar.data))

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), uvar.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.clim([-uvmax, uvmax])
plt.colorbar()


plt.show()
