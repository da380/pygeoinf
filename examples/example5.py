import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.homogeneous_space.line import Sobolev
from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearBayesianInversion,
    CholeskySolver,
    pointwise_variance,
    DiagonalLinearOperator,
)


# Set the model space
X = Sobolev(0, 2, 0.001, 0, 0.01)
print(X.dim)

# Set the model prior
mu = X.sobolev_gaussian_measure(2, 0.1)


# Set up the forward operator
n = 1
x = X.random_points(n)
A = X.point_evaluation_operator(x)
Y = A.codomain

# Set up the data error measure
sigma0 = 0.1
sigma1 = 0.1
standard_deviations = sigma0 + np.random.rand(Y.dim) * (sigma1 - sigma0)
nu = GaussianMeasure.from_standard_deviations(Y, standard_deviations)


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
pi_approx = pi.low_rank_approximation(
    5,
    method="variable",
    rtol=1e-4,
)
uvar = pointwise_variance(pi_approx, 200)
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
