"""
6. Nonlinear operators, and why ``at()`` exists.

A PDE solve usually yields the value and the derivative together. Asking for
them separately means solving twice. ``F.at(x)`` returns both from one call;
``F(x)`` returns only the value, which is what a line search wants.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, LinearOperator, Operator
from pygeoinf2.algebra.linearisation import Linearisation
from pygeoinf2.testing import check_derivative

rng = np.random.default_rng(0)
X, Y = EuclideanSpace(3), EuclideanSpace(2)

solves = {"count": 0}


def expensive_solve(m):
    """Stands in for a PDE solve: gives everything needed for both answers."""
    solves["count"] += 1
    return m @ m, 2.0 * m


def linearise(m):
    squared, jacobian_row = expensive_solve(m)
    value = np.array([squared, float(m[0])])
    rows = np.vstack([jacobian_row, np.eye(3)[0]])
    return Linearisation(
        m, value, LinearOperator.from_matrix(X, Y, rows, form="components")
    )


F = Operator.from_callables(
    X, Y, lambda m: np.array([m @ m, m[0]]), linearise=linearise
)

m = X.random(rng=rng)

solves["count"] = 0
model = F.at(m)
_ = model.value, model.derivative
print("at(m) gives value and derivative from", solves["count"], "solve")

solves["count"] = 0
for _ in range(20):
    F(X.random(rng=rng))
print("20 value-only evaluations cost", solves["count"], "solves")
print("  -- a line search is not charged for Jacobians it will discard.")
print()

check_derivative(F, m, rng=rng)
print("check_derivative passed: the derivative matches finite differences.")
print()

# The chain rule composes, and the derivative of the composition is built for
# you. A second derivative is optional and propagates the same way.
A = LinearOperator.from_matrix(
    Y, EuclideanSpace(1), rng.normal(size=(1, 2)), form="components"
)
composed = A @ F
print("(A @ F).has_derivative:", composed.has_derivative)
check_derivative(composed, m, rng=rng)
print("and its derivative is right too.")
