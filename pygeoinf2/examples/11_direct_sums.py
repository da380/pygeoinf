"""
11. Direct sums, and the joint model they exist for.

The joint law of model and data is not assembled from a covariance. It is a
product measure pushed through a block operator:

    op = [[I, 0], [A, I]]  applied to  prior (x) noise  gives  (m, A m + e)

Put a nonlinear F where A is and the same line gives the law of (m, F(m) + e).
That is the whole bridge to nonlinear inference.
"""

import numpy as np

from pygeoinf2 import (
    BlockOperator,
    DirectSum,
    EuclideanSpace,
    GaussianMeasure,
    LinearOperator,
    Operator,
)
from pygeoinf2.probability import product

rng = np.random.default_rng(0)
X, Y = EuclideanSpace(4), EuclideanSpace(3)

S = DirectSum([X, Y], labels=("model", "data"))
print("a direct-sum vector is a tuple:", type(S.random(rng=rng)).__name__)
print("with named access:", S.component(S.random(rng=rng), "model").shape)
print()

A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
prior = GaussianMeasure.from_standard_deviation(X, 1.0)
noise = GaussianMeasure.from_standard_deviation(Y, 0.2)

joint_operator = BlockOperator(
    [
        [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
        [A, LinearOperator.identity(Y)],
    ]
)
joint = joint_operator @ product([prior, noise], labels=("model", "data"))

print("linear case  ->", type(joint).__name__)
print("  covariance :", joint.covariance.traits)

# The data block of the joint covariance is A C A* + R, as it should be.
inclusion = joint.domain.inclusion(1)
expected = A @ prior.covariance @ A.adjoint + noise.covariance
y = Y.random(rng=rng)
print(
    "  data block == A C A* + R ?",
    np.allclose(joint.domain.component(joint.covariance(inclusion(y)), 1), expected(y)),
)
print()

# Now the same expression with a nonlinear forward map.
F = Operator.from_callables(X, Y, lambda m: np.array([m @ m, m[0], m[1]]))
nonlinear_operator = BlockOperator(
    [
        [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
        [F, LinearOperator.identity(Y)],
    ]
)
nonlinear_joint = nonlinear_operator @ product([prior, noise])

print("nonlinear case ->", type(nonlinear_joint).__name__)
print("  has a covariance in closed form:", nonlinear_joint.has_covariance)
print("  but it samples, which is what matters:")
model, data = nonlinear_joint.sample(rng=rng)
print("    m =", model.round(3), " d =", data.round(3))
print()
print("That is prior predictive sampling, from the same object as the linear case.")
