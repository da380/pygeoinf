"""
9. Randomised linear algebra, on the space's own geometry.

The probes are white noise *on the space*: a draw whose covariance is the
identity there, not in the components. On a weighted space those are different
distributions, and using the wrong one biases every answer by the Gram matrix.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, LinearOperator, Traits
from pygeoinf2.numerics import random_eig, random_range, random_trace
from pygeoinf2.spaces import Sobolev

rng = np.random.default_rng(0)

# An operator of exactly rank six, hidden in forty dimensions.
X = EuclideanSpace(40)
factor = rng.normal(size=(40, 6))
matrix = factor @ factor.T
A = LinearOperator.from_component_matrix(
    X, X, matrix, traits=Traits.POSITIVE_SEMIDEFINITE
)

basis = random_range(A, rng=rng, rtol=1e-8, block_size=4)
print("adaptive range finding found rank", len(basis), "(true rank 6)")

decomposition = random_eig(A, rank=6, rng=rng)
print(
    "eigenvalues match to",
    np.max(
        np.abs(
            np.sort(decomposition.eigenvalues)[::-1]
            - np.sort(np.linalg.eigvalsh(matrix))[::-1][:6]
        )
    ),
)
print("the factor is an isometry:", Traits.ISOMETRY & decomposition.factor.traits)
print("so U D U* is recognised as:", decomposition.traits)
print()

# The trace, on a space where the metric is not the identity.
H = Sobolev((16,), 2.0, 0.3)
values = np.arange(1.0, H.dim + 1.0)
B = H.invariant_operator(lambda _: values)

estimate = random_trace(B, samples=8000, rng=rng)
exact = float(values.sum())
wrong = float((H.metric_values * values).sum())

print(f"Hutchinson trace : {estimate}")
print(f"  exact          : {exact:.4f}")
print(f"  what components-space probes would give: {wrong:.4f}")
print()
print("The two differ by a factor of the metric. Drawing probes in the")
print("components rather than on the space gives the second number.")
