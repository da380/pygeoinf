import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.homogeneous_space.line import Sobolev
from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearMinimumNormInversion,
    CGSolver,
)


# Set the model space
X = Sobolev(0, 2, 0.001, 2, 0.1)

# Set the model prior
mu = X.sobolev_measure(2, 0.01)


# Set up the forward operator
n = 50
x = X.random_points(n)
A = X.point_evaluation_operator(x)

# Set up the data error measure
sigma = 0.1
nu = GaussianMeasure.from_standard_deviation(A.codomain, sigma)

# Set up the forward problem
forward_problem = LinearForwardProblem(A, nu)

# Generate and plot synthetic data.
u, v = forward_problem.synthetic_model_and_data(mu)
# X.plot(u, "k")

plt.errorbar(x, v, 2 * sigma, fmt="ko", capsize=2)

# Set up the inverse problem
inversion = LinearMinimumNormInversion(forward_problem)

umin = inversion.minimum_norm_operator(CGSolver())(v)

X.plot(umin)

umax = np.max(np.abs(u))
plt.ylim([-1.3 * umax, 1.3 * umax])
plt.xlim([0, 2])
plt.grid()


plt.show()
