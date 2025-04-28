import numpy as np
from pygeoinf.hilbert import CholeskySolver, CGSolver, CGMatrixSolver
from pygeoinf.bayesian import BayesianInversion, BayesianInference
from pygeoinf.sphere import Sobolev, Lebesgue
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 1.1, 0.1)


# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(2.0, 0.25, 1)


# Set up the forward operator.
n = 50
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)


# Set up the error distribution.
Y = A.codomain
stds = np.random.uniform(0.05, 0.2, Y.dim)
nu = Y.diagonal_gaussian_measure(stds)

# Set up the inference problem
problem = BayesianInversion(A, mu, nu)

# Set up the preconditioner
P = X.low_degree_projection(16)
Z = P.codomain
At = Z.point_evaluation_operator(lats, lons) @ P
Q = mu.covariance
R = nu.covariance
Nt = At @ Q @ At.adjoint + R
prec = CholeskySolver()(Nt)

# Generate synthetic data.
u = mu.sample()
v = problem.data_measure(u).sample()

solver = CholeskySolver()
# solver = CGMatrixSolver()
pi = problem.model_posterior_measure(
    v, solver=solver, preconditioner=prec
).low_rank_approximation(20, power=2)

ubar = pi.expectation
ustd = X.sample_std(pi.samples(100), expectation=ubar)


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
plt.clim([0, umax])
plt.colorbar()

# plt.figure()
# plt.pcolormesh(u.lons(), u.lats(), np.abs((ubar - u).data), cmap="Reds")
# plt.plot(lons, lats, "ko")
# plt.clim([0, umax])
# plt.colorbar()


plt.show()
