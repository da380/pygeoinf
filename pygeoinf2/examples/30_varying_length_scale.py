"""
30. An interval whose correlation length varies.

The Fourier interval of example 14 diagonalizes the Laplacian, and so has one
length scale everywhere. The interval here diagonalizes ``A = 1 - d/dx (L^2
d/dx)`` on a spectral-element mesh, for whatever ``L(x)`` it is given: a prior
whose draws wriggle at one end and roll at the other, which is what a profile
through a layered medium wants.

Two things are worth seeing. The mesh is longer than the domain, because ``A``
needs a boundary condition at each end and the padding keeps its mark away
from where it matters. And nothing here is homogeneous, so a prior is
calibrated by a standard deviation *field*: the raw variance of ``A^-p`` is
larger where ``L`` is smaller, and ``pointwise_std=`` flattens it -- or shapes
it, given a field -- without touching the correlation structure.

This is an interval of the *line*: the measure is ``dx``, the coordinate is a
position and not a radius, and the mesh is padded at both ends. A function of
radius in a planet, under ``r^2 dr``, is example 32.

Needs planetmodel, which comes with the 'planetmodel' extra.
"""

import numpy as np

from pygeoinf2 import plotting
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.sem1d.interval import Sobolev

rng = np.random.default_rng(3)

# ---------------------------------------------------------------------------
# The space: H^1 on [0, 2], correlation length from 0.04 to 0.3.
# ---------------------------------------------------------------------------


def length_scale(x):
    return 0.04 + 0.26 * (np.asarray(x) / 2.0) ** 2


X = Sobolev(160, 1.0, length_scale, upper=2.0)
print(X)
print(
    f"a mesh of {X.nodes.size} nodes on [{X.nodes[0]:.2f}, {X.nodes[-1]:.2f}]: "
    f"padded by {X.padding[0]:.2f} below and {X.padding[1]:.2f} above, two "
    "length scales at each end, under a Robin condition"
)
print()

# ---------------------------------------------------------------------------
# A prior, raw and calibrated.
# ---------------------------------------------------------------------------

# Covariance A^-1 on H^1: Matern draws of smoothness 1 + 1 - 1/2 = 3/2.
raw = X.interior_values(X.pointwise_variance(X.eigenvalues**-1.0))
print(
    f"raw pointwise variance of A^-1: {raw[0]:.2f} at x = 0 and {raw[-1]:.2f} "
    f"at x = 2, a ratio of {raw[0] / raw[-1]:.1f} for a length scale ratio of "
    f"{length_scale(2.0) / length_scale(0.0):.1f}"
)

prior = X.sobolev_measure(1.0, pointwise_std=0.5)
draws = [prior.sample(rng=rng) for _ in range(400)]
spread = np.std([X.interior_values(draw) for draw in draws], axis=0)
print(
    f"calibrated to 0.5: the sample standard deviation over 400 draws runs "
    f"from {spread.min():.3f} to {spread.max():.3f}"
)
print()

# ---------------------------------------------------------------------------
# Point data, and the posterior.
# ---------------------------------------------------------------------------

stations = np.sort(rng.uniform(0.0, 2.0, size=14))
forward = X.point_evaluation_operator(stations)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.03)
problem = LinearForwardProblem(forward, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

posterior = LinearGaussianInversion(problem, prior)(data)

# The posterior standard deviation along the interval, through the algebra:
# the diagonal of E C E*, with E evaluation at the plotting points.
along = np.linspace(0.0, 2.0, 161)
look = X.point_evaluation_operator(along)
band = np.sqrt(
    np.diag((look @ posterior.covariance @ look.adjoint).matrix(form="components"))
)
mean = look(posterior.expectation)
inside = np.mean(np.abs(look(truth) - mean) <= 2.0 * band)
print(
    f"{stations.size} noisy point values: the truth lies within two posterior "
    f"standard deviations at {inside:.0%} of the interval"
)
near = band[np.argmin(np.abs(along[:, None] - stations[None, :]), axis=0)]
print(
    f"the standard deviation falls from the prior's 0.5 to {near.mean():.3f} "
    f"at the stations, and recovers between them faster where L is short"
)
print()

# ---------------------------------------------------------------------------
# Two panels.
# ---------------------------------------------------------------------------

figure, (left, right) = plotting.subplots(X, columns=2)

for draw in draws[:3]:
    plotting.plot(X, draw, ax=left, padding=True, linewidth=1.0)
left.set_title("Three prior draws, with the padding shaded")
left.set_xlabel("x")

right.fill_between(
    along,
    mean - 2.0 * band,
    mean + 2.0 * band,
    alpha=0.25,
    label="posterior, two standard deviations",
)
right.plot(along, mean, label="posterior mean")
plotting.plot(X, truth, ax=right, color="black", linewidth=0.8, label="truth")
right.plot(stations, data, "o", markersize=4, label="data")
right.set_title("Fourteen point values")
right.set_xlabel("x")
right.legend(fontsize=7)

print("two panels drawn")
plotting.show()
