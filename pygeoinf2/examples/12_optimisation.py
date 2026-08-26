"""
12. Optimisation in the space, not in the components.

The methods here are written against the inner product, so the gradient they
step along is the one your metric defines. The consequence is practical: the
iteration count does not change when the discretisation is rescaled.
"""

import numpy as np

from pygeoinf2 import (
    EuclideanSpace,
    Functional,
    LinearFunctional,
    LinearOperator,
    Traits,
)
from pygeoinf2.numerics import LBFGS, NewtonCG, SteepestDescent
from pygeoinf2.numerics.convex import SquaredDistance
from pygeoinf2.symmetric_space import Sobolev

rng = np.random.default_rng(0)


def quadratic(space, matrix, offset):
    symmetric = 0.5 * (matrix + matrix.T)

    def value(x):
        c = space.to_components(x) - offset
        return 0.5 * float(c @ symmetric @ c)

    def derivative(x):
        c = space.to_components(x) - offset
        return LinearFunctional.from_derivative_components(space, symmetric @ c)

    def hessian(x):
        return LinearOperator.self_adjoint(
            space,
            lambda v: space.from_components(
                space.solve_gram(symmetric @ space.to_components(v))
            ),
            traits=Traits.POSITIVE_DEFINITE,
        )

    return Functional.from_callables(
        space, value, derivative=derivative, hessian=hessian
    )


X = EuclideanSpace(20)
root = rng.normal(size=(20, 20))
phi = quadratic(X, root @ root.T + 20.0 * np.identity(20), rng.normal(size=20))
start = X.random(rng=rng)

for method in (SteepestDescent(max_iterations=3000), LBFGS(), NewtonCG()):
    result = method.minimise(phi, start)
    print(f"{type(method).__name__:16s} {result.iterations:4d} iterations")
print()

# The metric point. phi(x) = ||x - a||^2 / 2 in the SPACE's norm has the
# identity as its Hessian, so it is perfectly conditioned whatever the metric.
print("the same problem, with the metric spread over four decades:")
for spread in (1.0, 100.0, 10000.0):
    space = Sobolev((16,), 2.0, spread)
    centre = space.random(rng=rng)
    objective = SquaredDistance(space, centre=centre)
    result = SteepestDescent(max_iterations=2000).minimise(
        objective, space.random(rng=rng)
    )
    condition = space.metric_values.max() / space.metric_values.min()
    print(
        f"  metric condition number {condition:12.1f} -> "
        f"{result.iterations} iterations"
    )
print()
print("In components the same function has the Gram matrix as its Hessian,")
print("so its condition number is the spread and the count grows with it.")
