import numpy as np
import pygeoinf.linalg as la
from scipy.stats import chi2
from pygeoinf.linalg import GaussianMeasure
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.bayesian import BayesianInversion, BayesianInference
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 1.5, 0.1)

# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(2, 1.5, 0.1)

# Set up the property operator.
m = 4
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)
B = X.point_evaluation_operator(lats, lons)


# Set up the forward operator.
n = 30
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
pi = problem.model_posterior_measure(v)

# xi = GaussianMeasure.low_rank_measure_by_factored_covariance(pi, 10)


plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.colorbar()

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), pi.expectation.data, cmap="seismic")
plt.plot(lons, lats, "ko")
plt.colorbar()


plt.show()
