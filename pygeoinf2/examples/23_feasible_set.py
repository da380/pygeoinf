"""
23. What the data cannot rule out.

The third kind of answer: a *set*. No probability anywhere -- the prior is a norm bound, the noise is a
norm bound, and what comes back is the collection of property values that no
model consistent with both can be shown to exclude.

The estimate is not a number with an error bar bolted on. It is a convex set,
and the honest way to report it is by its extent in chosen directions, which is
what a support function is. The example computes it, one constrained
optimization per direction with a certified bracket, and compares the result
with a linear certificate, which bounds it from outside.

Needs pyshtools and cartopy, which come with the 'sphere' extra.
"""

import numpy as np

from pygeoinf2 import plotting
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import BackusGilbert, BackusGilbertParker, LinearForwardProblem
from pygeoinf2.numerics.solvers import CholeskySolver, SpectralFamilySolver
from pygeoinf2.symmetric_space.sphere import Sobolev

rng = np.random.default_rng(3)

# H^{3/2} on the sphere: the Sobolev embedding makes a point value a bounded
# functional there, which is what lets a path average have a representer.
X = Sobolev(24, 1.5, 0.1)
print(f"model space: H^1.5, dimension {X.dim}")

# ---------------------------------------------------------------------------
# The problem: phase-delay paths, and four spherical caps as the properties.
# ---------------------------------------------------------------------------

paths = X.source_receiver_paths(
    sources=25, receivers=8, minimum_separation=0.4, rng=rng
)
forward = X.path_average_operator(paths, count=12, dense=True)
print(f"{len(paths)} paths -> data space of dimension {forward.codomain.dim}")

# (latitude, longitude) in degrees, which is what every method of the sphere
# takes at its boundary.
centers = [
    np.array([lat, lon])
    for lat, lon in [(-60.0, 0.0), (5.0, 143.0), (0.0, -120.0), (46.0, 104.0)]
]
target = X.geodesic_ball_average_operator(centers, 0.15, dense=True)
print(f"{len(centers)} spherical caps as the property operator")
print()

# ---------------------------------------------------------------------------
# A truth inside the prior ball, and data inside the noise ball.
# ---------------------------------------------------------------------------

PRIOR_RADIUS, NOISE_RADIUS = 3.0, 0.02

raw = X.heat_measure(0.01).sample(rng=rng)
truth = X.scale(0.6 * PRIOR_RADIUS / X.norm(raw), raw)

data_space = forward.codomain
raw_noise = data_space.random(rng=rng)
data = data_space.add(
    forward(truth),
    data_space.scale(0.7 * NOISE_RADIUS / data_space.norm(raw_noise), raw_noise),
)
print(f"||truth|| = {X.norm(truth):.4f} <= {PRIOR_RADIUS}")
print(
    f"data misfit = {data_space.norm(data_space.subtract(data, forward(truth))):.5f}"
    f" <= {NOISE_RADIUS}"
)
print()

problem = LinearForwardProblem(forward, error=Ball(data_space, radius=NOISE_RADIUS))
prior = Ball(X, radius=PRIOR_RADIUS)

# ---------------------------------------------------------------------------
# The exact bounds, one constrained optimization per direction.
# ---------------------------------------------------------------------------

# The damped family behind each direction is ill-conditioned with a noise
# ball this tight, and conjugate gradients will not reach their tolerance on
# it. The data space is small, so the family is solved once for every
# multiplier by a dense eigendecomposition instead.
exact = BackusGilbertParker(problem, target, prior, solver=SpectralFamilySolver())
P = target.codomain

answer = exact(data)
print(f"the sets are {answer.form!r}: {answer.algorithms['support'][0]} answers")
print("cap averages: the feasible interval (outer values), against the truth")
truth_values = target(truth)
lower, upper = [], []
for index in range(P.dim):
    direction = P.basis_vector(index)
    high = answer.support(direction)
    low = -answer.support(P.scale(-1.0, direction))
    lower.append(low)
    upper.append(high)
    inside = low - 1e-9 <= truth_values[index] <= high + 1e-9
    inner, outer = answer.support_bracket(direction)
    print(
        f"   cap {index}: [{low:+.5f}, {high:+.5f}]   truth {truth_values[index]:+.5f}"
        f"   {'inside' if inside else 'OUTSIDE'}   bracket width {outer - inner:.1e}"
    )
lower, upper = np.array(lower), np.array(upper)
print()

# ---------------------------------------------------------------------------
# A linear certificate, which bounds the exact interval from outside.
# ---------------------------------------------------------------------------

# A direct solve, not CG: with a noise ball this tight the data-space
# operator is nearly singular. The data space is small, so it can simply be
# factored. SOLA is this estimator with the trade-off chosen by hand and no
# prior ball named, which is why SOLA cannot certify its resolution error.
certificate = BackusGilbert(problem, target, prior, solver=CholeskySolver())
estimate, resolution, noise = certificate(data).error_bars()
print("the same, from a linear certificate -- valid always, sharp never:")
for index in range(P.dim):
    width = resolution[index] + noise[index]
    print(
        f"   cap {index}: [{estimate[index]-width:+.5f}, {estimate[index]+width:+.5f}]"
        f"   resolution {resolution[index]:.5f}  noise {noise[index]:.5f}"
    )
print(
    f"   it is wider by a factor of "
    f"{np.mean(2*(resolution+noise)/(upper-lower)):.2f} on average"
)
print()

# The split says what to do about it: the resolution term is coverage, the
# noise term is measurement quality, and they respond to different remedies.
print(
    f"resolution accounts for {np.mean(resolution/(resolution+noise)):.0%} of the"
    " certificate's width, so the limit here is coverage, not noise"
)
print()

# Which is checkable rather than assertable. Count the ray samples passing near
# each cap and compare with the width of its interval: a cap the rays miss
# should be, and is, unconstrained by them.
print("interval width against ray coverage:")
coverage = []
for index, center in enumerate(centers):
    near = sum(
        1
        for start, end in paths
        for node in X.geodesic_quadrature(start, end, count=12)[0]
        if X.geodesic_distance(node, center) < 0.3
    )
    coverage.append(near)
    print(
        f"   cap {index}: width {upper[index] - lower[index]:.3f}"
        f"   ray samples within 0.3 rad: {near}"
    )
least, widest = int(np.argmin(coverage)), int(np.argmax(upper - lower))
print(f"   least covered: cap {least}; widest interval: cap {widest}")
print()


# ---------------------------------------------------------------------------
# The set is not a box: off-axis directions find the trade-offs.
# ---------------------------------------------------------------------------

angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
diagonal = [
    P.from_components(np.array([np.cos(a), np.sin(a), 0.0, 0.0])) for a in angles
]
box = answer.polytope(
    [P.scale(s, P.basis_vector(i)) for i in range(P.dim) for s in (1.0, -1.0)]
)
tighter = box & answer.polytope(diagonal)
print(f"axis-aligned box: {box}")
print(f"with 12 off-axis directions: {tighter}")
print(
    f"the truth survives both: {box.contains(truth_values)}"
    f" and {tighter.contains(truth_values)}"
)
# The contact points of the same directions are points of the set, so their
# hull is an inner bound: the set lies between the hull and the polytope.
pair = LinearOperator.from_matrix(
    P, EuclideanSpace(2), np.eye(2, P.dim), form="components"
)
projected = answer.push_forward(pair)
hull = projected.inner_hull(
    [projected.domain.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles]
)
print(f"inner hull of the contact points on caps 0 and 1: {hull}")
print()

# ---------------------------------------------------------------------------
# Two panels: the truth with the network, and the caps that were measured.
# ---------------------------------------------------------------------------

figure, axes = plotting.subplots(X, rows=1, columns=2)
ax, _ = plotting.plot(
    X,
    truth,
    ax=axes[0],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="d ln c",
)
plotting.plot_network(X, paths, ax=ax, alpha=0.08)
ax.set_title("Truth, with the ray network")

extremal = answer.extremal_model(P.basis_vector(0))
ax, _ = plotting.plot(
    X,
    extremal,
    ax=axes[1],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="d ln c",
)
plotting.plot_balls(X, centers, 0.15, ax=ax, color="black")
plotting.plot_points(X, centers, ax=ax, color="black", marker="o", size=20.0)
ax.set_title("The model that maximizes cap 0, with the caps")

# ---------------------------------------------------------------------------
# The feasible set itself, projected onto two of the caps: the convex set
# the estimator returns, drawn from its support function, with the truth.
# ---------------------------------------------------------------------------

ax, _ = plotting.plot_set(projected, color="tab:blue", alpha=0.4)
plotting.plot_set(
    hull, ax=ax, bounds=(ax.get_xlim(), ax.get_ylim()), color="tab:green", alpha=0.4
)
ax.plot(truth_values[0], truth_values[1], "k*", markersize=12, label="truth")
ax.set_xlabel("cap 0 average")
ax.set_ylabel("cap 1 average")
ax.set_title("The feasible set on caps 0 and 1, with the inner hull (green)")
ax.legend()

print("three figures drawn")
plotting.show()
