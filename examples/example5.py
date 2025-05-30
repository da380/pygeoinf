import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.homogeneous_space.line import Sobolev
from pygeoinf import (
    FactoredGaussianMeasure,
    LinearForwardProblem,
    LinearBayesianInversion,
    CholeskySolver,
    sample_variance,
)


# Set the model space
X = Sobolev(0, 2, 0.001, 2, 0.01)

# Set the model prior
mu = X.sobolev_measure(2, 0.01)


# Set up the forward operator
n = 100
x = X.random_points(n)
A = X.point_evaluation_operator(x)
Y = A.codomain

# Set up the data error measure
"""
sigma = 0.3
nu = (
    FactoredGaussianMeasure.from_standard_deviation(A.codomain, sigma)
    if sigma > 0
    else None
)
"""
sigma0 = 0.05
sigma1 = 0.4
standard_deviations = sigma0 + np.random.rand(Y.dim) * (sigma1 - sigma0)
nu = FactoredGaussianMeasure.from_standard_deviations(Y, standard_deviations)

# Set up the forward problem
forward_problem = LinearForwardProblem(A, nu)

# Generate and plot synthetic data.
u, v = forward_problem.synthetic_model_and_data(mu)
X.plot(u, "k")

plt.errorbar(x, v, 2 * standard_deviations, fmt="ko", capsize=2)

# Set up the inverse problem
inversion = LinearBayesianInversion(forward_problem, mu)

# Get the posterior model distribtution.
pi = inversion.model_posterior_measure(v, CholeskySolver())


# Estimate the pointwise standard deviation.
uvar = sample_variance(pi.low_rank_approximation(300, power=2), 200)
ustd = np.sqrt(uvar)


# Plot the posterior expectation.
ubar = pi.expectation
X.plot(ubar, "b")
X.plot_error_bounds(ubar, 2 * ustd, alpha=0.2, color="b")


umax = np.max(np.abs(ubar) + 2 * ustd)
plt.ylim([-1.3 * umax, 1.3 * umax])
plt.xlim([0, 2])
plt.grid()


plt.show()
