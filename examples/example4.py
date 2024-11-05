import numpy as np
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.optimisation import LeastSquaresInversion
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 2, 0.2)

mu = X.sobolev_gaussian_measure(2, 0.2, 1)
u = mu.sample()

# Set up the forward operator.
n = 500
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)


# Set up the error distribution.
Y = A.codomain
sigma = 0.1
nu = Y.standard_gaussisan_measure(sigma)

problem = LeastSquaresInversion(A, nu)

v = problem.data_measure(u).sample()

# problem.trade_off_curve(0.1, 10.0, 100, v)

damping = 1.0
B = problem.least_squares_operator(damping)

u2 = B(v)

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.plot(lons, lats, 'ko')

plt.colorbar()

plt.figure()
plt.pcolormesh(u2.lons(), u2.lats(), u2.data, cmap="seismic")
plt.plot(lons, lats, 'ko')

plt.colorbar()
plt.show()

'''
w = X.dirac_representation(20, 180)

R = problem.resolution_operator(damping)
z = R.adjoint(w)

plt.figure()
plt.pcolormesh(z.lons(), z.lats(), w.data, cmap="seismic")
plt.plot(lons, lats, 'ko')
plt.colorbar()

plt.figure()
plt.pcolormesh(z.lons(), z.lats(), z.data, cmap="seismic")
plt.plot(lons, lats, 'ko')
plt.colorbar()


plt.show()
'''
