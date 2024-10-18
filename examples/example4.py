import numpy as np
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(128, 2, 0.1)

mu = X.sobolev_gaussian_measure(2, 0.4, 1)
u = mu.sample()

# Set up the forward operator.
n = 5
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)


# Set up the error distribution.
Y = A.codomain
sigma = 0.1
nu = Y.standard_gaussisan_measure(sigma)


forward_problem = ForwardProblem(A, nu)

v = forward_problem.data_measure(u).sample()

print(forward_problem.chi_squared(u, v))


# plt.figure()
# plt.pcolormesh(u.lons(), u.lats(), (u-u2).data, cmap="seismic")
# plt.plot(lons, lats, 'ko')
# plt.clim([-umax, umax])
# plt.colorbar()
# plt.show()
