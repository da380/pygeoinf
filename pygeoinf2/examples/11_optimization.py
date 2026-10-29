"""
11. Optimization in the space, not in the components.

The methods here are written against the inner product, so the gradient they
step along is the one your metric defines. The consequence is practical: the
iteration count does not follow the conditioning of the metric, as it would
for the same function handed over as a function of its component array.
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
from pygeoinf2.symmetric_space import circle

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
    result = method.minimize(phi, start)
    print(f"{type(method).__name__:16s} {result.iterations:4d} iterations")
print()

# The metric point. phi(x) = ||x - a||^2 / 2 in the SPACE's norm has the
# identity as its Hessian in the space, so it is perfectly conditioned
# whatever the metric. Handed to the same method as a function of the
# component array, it is (c - a)' G (c - a) / 2 with the Gram matrix G as its
# Hessian, and the conditioning is the metric's.
print("||x - a||^2 / 2 on H^1 of a circle, by steepest descent:")
print(
    f"{'length scale':>13} {'metric condition':>17} {'in the space':>13} {'on the arrays':>14}"
)
for length in (0.01, 0.1, 1.0):
    space = circle.Sobolev(32, 1.0, length)
    center, start = space.random(rng=rng), space.random(rng=rng)
    in_space = SteepestDescent(max_iterations=5000).minimize(
        SquaredDistance(space, center=center), start
    )
    metric = space.metric_values
    a = space.to_components(center)
    arrays = EuclideanSpace(space.dim)
    on_arrays = Functional.from_callables(
        arrays,
        lambda c, a=a, metric=metric: 0.5 * float((c - a) @ (metric * (c - a))),
        gradient=lambda c, a=a, metric=metric: metric * (c - a),
    )
    by_components = SteepestDescent(max_iterations=5000).minimize(
        on_arrays, space.to_components(start)
    )
    condition = metric.max() / metric.min()
    print(
        f"{length:13.2f} {condition:17.1f} {in_space.iterations:13d}"
        f" {by_components.iterations:14d}"
    )
print()
print("The same function, the same method, the same minimum. In the space the")
print("count does not move; on the arrays it follows the metric's condition")
print("number, which grows with the length scale and with the resolution.")
