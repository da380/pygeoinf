import numpy as np
import pygeoinf.linalg as la
from scipy.stats import chi2
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.occam import OccamInversion
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 2, 0.3)

# Set up the forward operator.
n = 250
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)

# Set up the error distribution.
Y = A.codomain
sigma = 0.1
nu = Y.standard_gaussisan_measure(sigma)

# Set up the inversion method.
problem = OccamInversion(A, nu)

# Make synthetic data.
mu = X.sobolev_gaussian_measure(2, 0.3, 1)
u = mu.sample()
v = problem.data_measure(u).sample()

u2 = problem.minimum_norm_operator(0.95)(v)

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.plot(lons, lats, 'ko')
plt.colorbar()

plt.figure()
plt.pcolormesh(u2.lons(), u2.lats(), u2.data, cmap="seismic")
plt.plot(lons, lats, 'ko')
plt.colorbar()

plt.show()
