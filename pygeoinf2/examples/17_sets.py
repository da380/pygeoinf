"""
17. A convex set, its indicator, and its support function are one object.

v1 keeps these in three places and nothing connects them. Here they are three
views of the same set, which is what lets a hard constraint enter a proximal
method with no extra machinery at all.

Note also what ``project`` means: the *nearest* point of the set. A point
already inside is left where it is, so the map is idempotent -- which is what a
proximal method relies on, and what a projection onto the boundary would not
give.
"""

import numpy as np

from pygeoinf2.geometry import Ball, HalfSpace, Hyperplane, UniversalSet
from pygeoinf2.numerics.convex import ProximalGradient, SquaredDistance
from pygeoinf2.spaces import Sobolev
from pygeoinf2.testing import check_projection

rng = np.random.default_rng(0)
X = Sobolev((16,), 2.0, 0.3)

ball = Ball(X, radius=1.0)
normal = X.random(rng=rng)
half_space = HalfSpace(X, normal, offset=0.5)

# --- the set algebra, which needs nothing but a membership test ----------
print("set algebra:")
print("  0 in ball                :", ball.contains(X.zero()))
print("  0 in ball complement     :", ball.complement().contains(X.zero()))
print(
    "  double complement is the original object:",
    ball.complement().complement() is ball,
)
print("  intersection             :", type(ball & half_space).__name__)
print(
    "  and it flattens          :", len((ball & half_space & UniversalSet(X)).subsets)
)
print()

# --- projection means NEAREST, so it fixes points already inside ---------
print("projection:")
inside = X.zero()
print(
    "  a feasible point is left alone:",
    X.norm(X.subtract(half_space.project(inside), inside)) < 1e-12,
)
print(
    "  the boundary hyperplane would move it:",
    X.norm(X.subtract(half_space.boundary.project(inside), inside)) > 1e-6,
)

far = X.scale(50.0 / X.norm(X.random(rng=rng)), X.random(rng=rng))
print(
    "  and an outside point lands on the sphere:", round(X.norm(ball.project(far)), 8)
)
for subset in (ball, half_space, Hyperplane(X, normal, offset=0.5)):
    check_projection(subset, rng=rng)
print("  check_projection passes: in the set, idempotent, nearest.")
print()

# --- the three views ------------------------------------------------------
print("three views of the ball:")
y = X.random(rng=rng)
support = ball.support_function()
indicator = ball.indicator()
print("  support_function(y) == ||y|| ?", np.isclose(support(y), X.norm(y)))
print("  indicator is 0 inside, inf outside:", indicator(X.zero()), indicator(far))
print(
    "  indicator.prox IS the projection  :",
    X.norm(X.subtract(indicator.prox(far, 1.0), ball.project(far))) < 1e-12,
)
print(
    "  and its conjugate is the support  :",
    np.isclose(indicator.conjugate()(y), support(y)),
)
print()

# --- so a hard constraint costs one argument -----------------------------
centre = X.random(rng=rng)
constrained = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
    SquaredDistance(X, centre=centre),
    X.random(rng=rng),
    nonsmooth=Ball(X, radius=0.25).indicator(),
)
print("min ||x - c||^2/2 subject to ||x|| <= 0.25")
print(
    f"  ||x*|| = {X.norm(constrained.minimiser):.6f}, in {constrained.iterations} iterations"
)
