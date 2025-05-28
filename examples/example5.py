import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.homogeneous_space.line import Sobolev
from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearBayesianInversion,
    CholeskySolver,
    sample_variance,
)

# Set the model space
X = Sobolev(0, 2, 0.001, 2, 0.01)

# Set the model prior
mu = X.sobolev_measure(2, 0.01)

# Generate model and plot
u = mu.sample()
X.plot(u, "k--")

# Set up the forward operator
n = 30
x = X.random_points(n)
A = X.point_evaluation_operator(x)

# Set up the data error measure
sigma = 0.2
nu = GaussianMeasure.from_standard_deviation(A.codomain, sigma)

# Set up the forward problem
forward_problem = LinearForwardProblem(A, nu)

# Generate and plot synthetic data.
v = forward_problem.data_measure(u).sample()
plt.errorbar(x, v, sigma, fmt="ko")

# Set up the inverse problem
inversion = LinearBayesianInversion(forward_problem, mu)

# Get the posterior model distribtution.
pi = inversion.model_posterior_measure(v, CholeskySolver()).low_rank_approximation(50)


uvar = sample_variance(pi, 20)
ustd = np.sqrt(uvar)

umax = np.max(np.abs(u))

# Plot the posterior expectation.
ubar = pi.expectation
X.plot(ubar, "b")
plt.fill_between(X.sample_points(), ubar - ustd, ubar + ustd, alpha=0.2)
plt.ylim([-1.1 * umax, 1.1 * umax])
plt.grid()


plt.show()
