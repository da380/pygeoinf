"""
33. A field on an annulus, drawn whole.

``pygeoinf2.disk`` is the two-dimensional ball: a disk or an annulus in the
eigenbasis of ``A = 1 - div(L^2 grad)`` under the measure ``r dr``, with a
length scale ``L`` that may vary with radius. Here it is a planet's mantle in
cross-section, its correlation length longer at depth, observed at stations
on its surface.

Unlike the ball of example 34, a field on it is drawn whole, and so is what
the data did: the posterior is the low-rank one of example 25, whose
``pointwise_variance_reduction`` is a field, the variance the data removed,
at no cost beyond the eigenpairs.

Needs planetmodel, which comes with the 'planetmodel' extra.
"""

import numpy as np

from pygeoinf2 import (
    GaussianMeasure,
    LinearForwardProblem,
    LowRankGaussianPosterior,
    plotting,
)
from pygeoinf2.disk import Sobolev

rng = np.random.default_rng(33)

INNER = 0.55  # the core-mantle boundary, as a fraction of the outer radius


def length_scale(r):
    """0.2 at the core-mantle boundary, 0.08 at the surface."""
    depth = (1.0 - np.asarray(r)) / (1.0 - INNER)
    return 0.08 + 0.12 * depth**2


X = Sobolev(32, 2.0, length_scale, inner_radius=INNER)
print(X)

# Order 2 is above half the dimension, so the measure is trace class; the
# pointwise standard deviation is calibrated to one throughout.
prior = X.sobolev_measure(2.0, pointwise_std=1.0)

# Forty stations on the surface. A point is (radius, angle in degrees).
stations = [(1.0, angle) for angle in np.sort(rng.uniform(0.0, 360.0, 40))]
forward = X.point_evaluation_operator(stations)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.05)
problem = LinearForwardProblem(forward, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

posterior = LowRankGaussianPosterior.from_linear_problem(problem, prior, data, rng=rng)
print(
    f"{len(stations)} surface stations; the posterior differs from the prior in"
    f" {posterior.rank} directions of {X.dim}"
)
print()

# Shell by shell: the standard deviation, exact from the low-rank covariance
# (rank-r corrections of the prior's factor, no solve), and the error of the
# posterior mean.
print("shell by shell, prior and posterior standard deviation, error of the mean:")
for radius in (1.0, 0.95, 0.9, 0.8, 0.7, INNER):
    points = [(radius, angle) for angle in np.linspace(0.0, 360.0, 24, endpoint=False)]
    look = X.point_evaluation_operator(points)

    def spread(measure):
        covariance = (look @ measure.covariance @ look.adjoint).matrix()
        return np.sqrt(np.mean(np.diag(covariance)))

    exact = look(truth)
    error = np.linalg.norm(look(posterior.expectation) - exact) / np.linalg.norm(exact)
    print(
        f"   r = {radius:.2f}   {spread(prior):.2f} -> {spread(posterior):.2f}"
        f"   error {error:.2f}"
    )
print("the surface data reach a correlation length or two down, and no further")

# ---------------------------------------------------------------------------
# Three panels, each the whole annulus.
# ---------------------------------------------------------------------------

limit = float(np.max(np.abs(X.interior_values(truth))))
figure, axes = plotting.subplots(X, columns=3)
plotting.plot(X, truth, ax=axes[0], vmin=-limit, vmax=limit, title="Truth")
plotting.plot_points(X, stations, ax=axes[0], size=8.0)
plotting.plot(
    X,
    posterior.expectation,
    ax=axes[1],
    vmin=-limit,
    vmax=limit,
    title="Posterior mean",
)
plotting.plot(
    X,
    posterior.pointwise_variance_reduction(),
    ax=axes[2],
    cmap="viridis",
    title="Variance the data removed",
)
print()
print("three panels drawn")
plotting.show()
