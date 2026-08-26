"""
10. Gaussian measures: sampling, moments, and pushing forward.

A covariance built from a factor gets its structure for free, sampling uses
white noise on the space, and the pushforward under an operator stays Gaussian
with ``A C A*`` recognised as positive semidefinite.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, GaussianMeasure, LinearOperator
from pygeoinf2.symmetric_space import Sobolev
from pygeoinf2.testing import check_measure

rng = np.random.default_rng(0)
X = Sobolev((16,), 2.0, 0.3)

# An isotropic measure: covariance sigma^2 times the identity ON THE SPACE.
mu = GaussianMeasure.from_standard_deviation(X, 1.5)
print("covariance traits:", mu.covariance.traits, " <- nothing was claimed")
print()

u = X.basis_vector(3)
draws = mu.samples(4000, rng=rng)
empirical = np.mean([X.inner_product(x, u) ** 2 for x in draws])
print(f"E[(x, u)^2] : {empirical:.4f}")
print(f"sigma^2 (u, u) : {1.5**2 * X.inner_product(u, u):.4f}   <- identity covariance")
print(f"what a components draw gives: {1.5**2 * X.inner_product(u, u) ** 2:.4f}")
print()

# Pushforward through a forward operator.
Y = EuclideanSpace(4)
A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(4, X.dim)))
data_measure = A @ mu
print("A @ mu is a", type(data_measure).__name__)
print("its covariance:", data_measure.covariance.traits)
check_measure(data_measure, rng=rng, samples=8000, rtol=0.1)
print("check_measure passed: sampled moments match the declared covariance.")
print()

# The log density and its gradient. The gradient is a VECTOR, because the
# precision maps the space to itself -- no Riesz map appears anywhere.
x = X.random(rng=rng)
print("log density   :", round(mu.log_density(x), 4))
print(
    "its gradient is a field, not an array of partials:",
    type(mu.grad_log_density(x)).__name__,
    mu.grad_log_density(x).shape,
)
