"""
30. What the data cannot rule out.

The third kind of answer: a *set*. The prior is a norm bound, the noise is a
norm bound, and what comes back is the collection of property values that no
model consistent with both can be shown to exclude. It is reported by its
extent in chosen directions -- support values, one constrained optimization
each, with a certified bracket -- and the extent is worth reading only beside
what the prior bound allows on its own: the difference is what the data
bought.

The bounds here are Gaussian beliefs hardened into sets. ``ambient_ball(level=)``
is the smallest ball about the prior's mean that carries the given
probability, in the space's own norm. Given an ``operator=``, it is the ball of
the norm that operator defines, an ellipsoid: here the ``L^2`` norm, which
bounds size and not roughness. The two answer differently, because a norm
bound is an assumption about smoothness as much as about size.

Needs pyshtools and cartopy, which come with the 'sphere' extra.
"""

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    BackusGilbertParker,
    GaussianMeasure,
    LinearForwardProblem,
    plotting,
)
from pygeoinf.numerics.solvers import SpectralFamilySolver
from pygeoinf.symmetric_space.sphere import Sobolev

rng = np.random.default_rng(3)

# H^{3/2} on the sphere: the Sobolev embedding makes a path average a bounded
# functional there, which is what lets it have a representer.
X = Sobolev(24, 1.5, 0.1)

# ---------------------------------------------------------------------------
# The problem: phase-delay paths, and four spherical caps as the properties.
# ---------------------------------------------------------------------------

paths = X.source_receiver_paths(
    sources=25, receivers=8, minimum_separation=0.4, rng=rng
)
forward = X.path_average_operator(paths, count=12, dense=True)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.002)
problem = LinearForwardProblem(forward, error=noise)

# (latitude, longitude) in degrees, which is what every method of the sphere
# takes at its boundary.
centers = [
    np.array([lat, lon])
    for lat, lon in [(-60.0, 0.0), (5.0, 143.0), (0.0, -120.0), (46.0, 104.0)]
]
caps = X.geodesic_ball_average_operator(centers, 0.15, dense=True)
P = caps.codomain
print(f"model space H^1.5, dimension {X.dim}; {len(paths)} paths; {P.dim} caps")

# A Gaussian belief about the model, and a truth drawn from it with its data.
prior = X.heat_measure(0.1, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
print()

# ---------------------------------------------------------------------------
# Two constraint sets from the one measure.
# ---------------------------------------------------------------------------

ball = prior.ambient_ball(level=0.95)
# The L^2 inner product read through the space's own: (B x, y) in H^1.5 is the
# L^2 product of x and y. Its inverse gives the ellipsoid a support function.
l2_form = X.spectral_operator(1.0 / X.metric_values)
ellipsoid = prior.ambient_ball(
    level=0.95, operator=l2_form, inverse=X.spectral_operator(X.metric_values)
)
print(f"the 95% ball in H^1.5 has radius {ball.radius:.4f}; the truth's norm is")
print(f"  {X.norm(truth):.4f}; the 95% ball in L^2 is an {type(ellipsoid).__name__}")
print("the truth is in both:", ball.contains(truth), ellipsoid.contains(truth))
print()

# ---------------------------------------------------------------------------
# The feasible set under each, in the eight directions +-e_i.
# ---------------------------------------------------------------------------

# The damped family behind each direction is ill-conditioned with noise this
# small, and conjugate gradients would struggle on it. The data space is small,
# so the family is solved once for every multiplier by a dense
# eigendecomposition instead. The noise measure is hardened into its 95% ball
# the same way, by the estimator.
directions = [
    P.scale(sign, P.basis_vector(i)) for i in range(P.dim) for sign in (1, -1)
]
truth_values = caps(truth)
intervals = {}
for name, constraint in (("H^1.5 ball", ball), ("L^2 ellipsoid", ellipsoid)):
    answer = BackusGilbertParker(
        problem, caps, constraint, solver=SpectralFamilySolver()
    )(data)
    # The directions are independent, so they are swept by worker processes.
    sweep = answer.support_sweep(directions, n_jobs=4)
    alone = [constraint.support_function()(caps.adjoint(q)) for q in directions]
    print(f"under the {name}, the cap averages lie in")
    rows = []
    for i in range(P.dim):
        high, low = sweep.results[2 * i], sweep.results[2 * i + 1]
        rows.append((-low.value, high.value, -alone[2 * i + 1], alone[2 * i]))
        gap = max(high.value - high.inner, low.value - low.inner)
        print(
            f"   cap {i}: [{-low.value:+.3f}, {high.value:+.3f}]"
            f"   the bound alone [{-alone[2 * i + 1]:+.3f}, {alone[2 * i]:+.3f}]"
            f"   truth {truth_values[i]:+.3f}   bracket {gap:.0e}"
        )
    intervals[name] = np.array(rows)
print()
print("The L^2 bound admits rougher fields than the H^1.5 one, so both its")
print("prior-alone bound and its feasible intervals are the wider.")

# ---------------------------------------------------------------------------
# Two figures: the network and the caps, and the intervals.
# ---------------------------------------------------------------------------

figure, axes = plotting.subplots(X, rows=1, columns=1)
ax, _ = plotting.plot(
    X,
    truth,
    ax=axes,
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="d ln c",
)
plotting.plot_network(X, paths, ax=ax, alpha=0.08)
plotting.plot_balls(X, centers, 0.15, ax=ax, color="black")
ax.set_title("The truth, the ray network, and the four caps")

figure, ax = plt.subplots(figsize=(7.0, 3.5), layout="constrained")
for offset, color, (name, rows) in zip((-0.12, 0.12), ("C0", "C1"), intervals.items()):
    for i, (low, high, alone_low, alone_high) in enumerate(rows):
        ax.plot([i + offset] * 2, [alone_low, alone_high], color=color, lw=1, alpha=0.4)
        ax.plot(
            [i + offset] * 2,
            [low, high],
            color=color,
            lw=6,
            alpha=0.7,
            solid_capstyle="butt",
            label=name if i == 0 else None,
        )
ax.plot(range(P.dim), truth_values, "k*", ms=10, label="truth")
ax.set_xticks(range(P.dim), [f"cap {i}" for i in range(P.dim)])
ax.set_ylabel("average of d ln c")
ax.set_title("Feasible intervals (thick) inside what each bound alone allows (thin)")
ax.legend(fontsize=8)

print()
print("two figures drawn")
plotting.show()
