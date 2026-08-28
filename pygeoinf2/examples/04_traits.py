"""
4. Structure survives the algebra.

An operator carries traits -- self-adjoint, positive definite, and so on. They
propagate through sums, scalings and compositions by rule, so a covariance
pushforward is *recognised* as positive semidefinite rather than asserted.

Traits are claims. ``check_traits`` verifies them.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, LinearOperator, Traits
from pygeoinf2.testing import check_traits

rng = np.random.default_rng(0)
X, Y = EuclideanSpace(6), EuclideanSpace(4)


def spd(n):
    root = rng.normal(size=(n, n))
    return root @ root.T + n * np.identity(n)


A = LinearOperator.from_matrix(X, Y, rng.normal(size=(4, 6)), form="components")
Q = LinearOperator.from_matrix(
    X, X, spd(6), traits=Traits.POSITIVE_SEMIDEFINITE, form="components"
)
R = LinearOperator.from_matrix(
    Y, Y, spd(4), traits=Traits.POSITIVE_DEFINITE, form="components"
)

print("A          ", A.traits)
print("A @ A*     ", (A @ A.adjoint).traits, " <- a Gramian is always semidefinite")
print("A @ Q @ A* ", (A @ Q @ A.adjoint).traits, " <- a congruence preserves it")
print()

# The operator every Bayesian inversion inverts, recognised with nothing claimed.
normal = A @ Q @ A.adjoint + R
print("A Q A* + R ", normal.traits)
print()
print("Note how that was assembled: (A @ Q) @ A.adjoint, in two steps.")
print("The pattern only exists once the composition is complete, which is why")
print("the algebra keeps its operands instead of collapsing into a closure.")
print()

check_traits(normal, rng=rng)
print("check_traits passed. A false claim would have been caught here,")
print("and CGSolver would refuse an operator that had not earned the trait.")
