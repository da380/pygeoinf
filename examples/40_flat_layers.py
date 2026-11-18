"""
40. Fields of a flat layer: a strip and a slab.

``pygeoinf.strip`` and ``pygeoinf.slab`` are the disk and the ball laid flat:
a layer between two depths in the eigenbasis of ``A = 1 - div(L^2 grad)``,
Fourier across it and spectral elements in depth, with a length scale ``L``
that may vary with depth. A strip is a field of ``(depth, x)``, a vertical
section; a slab a field of ``(depth, x, y)``. Depth is positive downward and
comes first in a point, as a radius does in a disk or a ball.

Here a crust whose correlation length grows with depth. On the strip a
prior, calibrated to a standard deviation that depends on depth, is drawn as
a depth section, and conditioned on a line of stations at the surface: the
data reach a correlation length or two down, and no further. On the slab the
same prior is drawn as two lateral maps, one near the surface and one at
depth, where the same field is visibly smoother.

Needs matplotlib.
"""

import matplotlib.pyplot as pyplot
import numpy as np

from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LowRankGaussianPosterior,
    plotting,
    slab,
    strip,
)

rng = np.random.default_rng(40)

WIDTH, BASE = 8.0, 2.0  # the lateral period and the depth of the base


def length_scale(depth):
    """0.15 at the surface, 0.5 at the base."""
    return 0.15 + 0.35 * (np.asarray(depth) / BASE) ** 2


def deviation(point):
    """The prior's standard deviation: largest at the surface."""
    return 1.0 - 0.3 * point[0] / BASE


# ---------------------------------------------------------------------------
# The strip: H^2 of (depth, x), so that points have values (order > 1).
# ---------------------------------------------------------------------------

# An odd number of points across: the grid then holds a sine for every
# cosine, and a prior's pointwise variance is exact between its points too.
X = strip.Sobolev(161, 2.0, length_scale, lateral=WIDTH, bottom=BASE)
print(X)
print(
    f"grid {X.grid_shape}: {X.depths.size} nodes in depth from {X.depths[0]:.2f} "
    f"to {X.depths[-1]:.2f}, of which {X.interior_depths.size} lie in the layer; "
    f"{len(X.basis)} lateral wavenumbers, {X.basis.nmodes.max()} modes of depth "
    f"at the longest and {X.basis.nmodes.min()} at the shortest"
)

# The standard deviation is continued across the padding in depth by its
# value at the nearer end, which keeps it positive there.
sigma = X.project_function(deviation, extension="constant")
prior = X.sobolev_measure(2.0, pointwise_std=sigma)

# Sixty stations at the surface. A point is (depth, x).
stations = [(0.0, x) for x in np.sort(rng.uniform(0.0, WIDTH, 60))]
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

print("level by level, prior and posterior standard deviation, error of the mean:")
across = np.linspace(0.0, WIDTH, 48, endpoint=False)
for depth in (0.0, 0.1, 0.25, 0.5, 1.0, BASE):
    look = X.point_evaluation_operator([(depth, x) for x in across])

    def spread(measure):
        covariance = (look @ measure.covariance @ look.adjoint).matrix()
        return np.sqrt(np.mean(np.diag(covariance)))

    exact = look(truth)
    error = np.linalg.norm(look(posterior.expectation) - exact) / np.linalg.norm(exact)
    print(
        f"   depth {depth:.2f}   {spread(prior):.2f} -> {spread(posterior):.2f}"
        f"   error {error:.2f}"
    )
print("the surface data reach a correlation length or two down, and no further")

# ---------------------------------------------------------------------------
# Depth sections: the layer alone, depth increasing downward.
# ---------------------------------------------------------------------------


def section(axis, field, /, *, title, **options):
    """One depth section of a strip's field, its padding left out."""
    image = axis.pcolormesh(
        X.x, X.interior_depths, X.interior_values(field), shading="gouraud", **options
    )
    axis.set_ylim(BASE, 0.0)
    axis.set_aspect("equal")
    axis.set_ylabel("depth")
    axis.set_title(title)
    return image


limit = float(np.max(np.abs(X.interior_values(truth))))
shared = dict(cmap="RdBu", vmin=-limit, vmax=limit)
figure, panels = pyplot.subplots(3, 1, figsize=(9.0, 8.0), layout="constrained")
image = section(panels[0], truth, title="Truth", **shared)
panels[0].scatter(
    [x for _, x in stations], [0.0] * len(stations), s=10.0, c="k", clip_on=False
)
section(panels[1], posterior.expectation, title="Posterior mean", **shared)
figure.colorbar(image, ax=panels[:2], label="field")
removed = section(
    panels[2],
    posterior.pointwise_variance_reduction(),
    title="Variance the data removed",
    cmap="viridis",
)
figure.colorbar(removed, ax=panels[2])
panels[2].set_xlabel("x")
print()
print("three depth sections drawn")

# ---------------------------------------------------------------------------
# The slab: the same prior over two lateral axes, as lateral maps.
# ---------------------------------------------------------------------------

Y = slab.Sobolev((45, 45), 2.0, length_scale, lateral=(4.0, 4.0), bottom=BASE)
print()
print(Y)
print(
    f"grid {Y.grid_shape}; {len(Y.basis)} distinct lateral wavenumbers share the "
    f"eigenproblems of {Y.transform.size} Fourier components"
)

# Order 2 is above three halves, so the measure is trace class.
draw = Y.sobolev_measure(
    2.0, pointwise_std=Y.project_function(deviation, extension="constant")
).sample(rng=rng)

# A lateral map at any depth is an evaluation at the points of a level.
x, y = np.meshgrid(Y.x, Y.y, indexing="ij")
limit = float(np.max(np.abs(Y.interior_values(draw))))
figure, maps = pyplot.subplots(1, 2, figsize=(10.0, 4.4), layout="constrained")
for axis, depth in zip(maps, (0.05, 1.5)):
    points = np.column_stack([np.full(x.size, depth), x.ravel(), y.ravel()])
    values = Y.evaluate(draw, points).reshape(x.shape)
    image = axis.pcolormesh(
        x, y, values, shading="gouraud", cmap="RdBu", vmin=-limit, vmax=limit
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_title(f"A draw of the prior at depth {depth:g}")
    print(
        f"   depth {depth:.2f}: length scale {length_scale(depth):.2f}, "
        f"standard deviation of the draw across the map {values.std():.2f}"
    )
maps[0].set_ylabel("y")
figure.colorbar(image, ax=maps, label="field")
print("two lateral maps drawn")
plotting.show()
