"""
10. Gaussian measures: sampling, moments, and pushing forward.

A covariance built from a factor gets its structure for free, sampling uses
white noise on the space, and the pushforward under an operator stays Gaussian
with ``A C A*`` recognized as positive semidefinite.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, GaussianMeasure, LinearOperator
from pygeoinf2.symmetric_space import Sobolev
from pygeoinf2.testing import check_measure

rng = np.random.default_rng(0)
X = Sobolev((16,), 2.0, 0.3)

# A measure on a function space must have a covariance that is trace class
# on it; the identity is not, and the library refuses it. The workhorse is
# the Sobolev measure: covariance (1 + L^2 lambda)^-order IN THE SPACE'S
# OWN METRIC, here of order one on a line.
try:
    GaussianMeasure.from_standard_deviation(X, 1.5)
except TypeError as refusal:
    print("an identity covariance is refused:", str(refusal)[:60], "...")
mu = X.sobolev_measure(1.0, 0.3, amplitude=1.5)
print("covariance traits:", mu.covariance.traits)
print()

# The second moment of a pairing is (C u, u) in the space's inner product,
# which is the covariance's claim and not a components formula.
u = X.basis_vector(3)
draws = mu.samples(4000, rng=rng)
empirical = np.mean([X.inner_product(x, u) ** 2 for x in draws])
print(f"E[(x, u)^2] : {empirical:.4f}")
print(
    f"(C u, u)    : {X.inner_product(mu.covariance(u), u):.4f}   <- the covariance's claim"
)
print()

# Pushforward through a forward operator.
Y = EuclideanSpace(4)
A = LinearOperator.from_matrix(X, Y, rng.normal(size=(4, X.dim)), form="components")
data_measure = A @ mu
print("A @ mu is a", type(data_measure).__name__)
print("its covariance:", data_measure.covariance.traits)
check_measure(data_measure, rng=rng, samples=8000, rtol=0.1)
print("check_measure passed: sampled moments match the declared covariance.")
print()

# The log density and its gradient. The gradient is a VECTOR, because the
# precision maps the space to itself -- no Riesz map appears anywhere.
# On a space that discretises functions the density is refused outright:
# its value is an artefact of the truncation. unsafe=True says
# "show me the discretisation's own number anyway".
x = X.random(rng=rng)
try:
    mu.log_density(x)
except ValueError as error:
    print("log density refused on a function space:", str(error)[:60], "...")
print("log density   :", round(mu.log_density(x, unsafe=True), 4), " <- unsafe=True")
print(
    "its gradient is a field, not an array of partials:",
    type(mu.grad_log_density(x, unsafe=True)).__name__,
    mu.grad_log_density(x, unsafe=True).shape,
)
