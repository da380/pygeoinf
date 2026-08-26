"""
13. Non-smooth problems: proximal operators are geometry, not indexing.

The proximal operators that matter are written with a norm and a direction:

    prox of  t w ||.||     x -> max(0, 1 - t w / ||x||) x
    ball projection        x -> min(1, r / ||x||) x

So they are metric-aware for free, and they mean the same thing under
refinement. Nothing here refers to a basis.
"""

import numpy as np

from pygeoinf2.numerics.convex import (
    BallIndicator,
    NormFunctional,
    ProximalGradient,
    SquaredDistance,
    SupportFunction,
)
from pygeoinf2.symmetric_space import Sobolev

rng = np.random.default_rng(0)
X = Sobolev((16,), 2.0, 0.3)

centre = X.random(rng=rng)
smooth = SquaredDistance(X, centre=centre)
penalty = NormFunctional(X, weight=0.3)

# min 0.5||x - c||^2 + w||x||. The answer is the shrinkage of c, in closed form.
result = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
    smooth, X.random(rng=rng), nonsmooth=penalty
)
print(f"FISTA converged in {result.iterations} iterations")
print(
    "matches the closed form:",
    X.norm(X.subtract(result.minimiser, penalty.prox(centre, 1.0))) < 1e-8,
)
print()

# A hard constraint enters as an indicator, whose prox is the projection.
constraint = BallIndicator(X, radius=0.2)
constrained = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
    smooth, X.random(rng=rng), nonsmooth=constraint
)
print(f"with ||x|| <= 0.2 : ||x*|| = {X.norm(constrained.minimiser):.6f}")
print()

# Support functions live on the same space, which is a small dividend of
# working without duals -- and their algebra is closed.
ball = SupportFunction.of_ball(X, radius=2.0)
shifted = ball + SupportFunction.of_point(X, centre)
y = X.random(rng=rng)
print(
    "h_ball(y)            :",
    round(ball(y), 6),
    "== 2||y|| ?",
    np.isclose(ball(y), 2.0 * X.norm(y)),
)
print("Minkowski sum stays a support function:", isinstance(shifted, SupportFunction))
print(
    "its subgradient attains the supremum  :",
    np.isclose(X.inner_product(ball.subgradient(y), y), ball(y)),
)
