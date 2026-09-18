"""
31. A field in a mantle, and the three ways of looking at it.

An annulus from the core-mantle boundary to the surface, in units of the outer
radius, with a prior whose correlation length and amplitude both depend on
depth -- the kind of statement a one-dimensional reference model lets you
make, and one no periodic box can hold, the measure being ``r^2 dr``.

A field in a ball cannot be drawn whole. ``plotting`` cuts it three ways: a
*shell* at one radius, drawn as a map; a *section* by any plane through the
centre, named by a longitude, by its pole, or by two points it passes through;
and a *profile* along one ray.

Then a small inversion, chosen to show something true about the geometry:
point values on the surface alone. Shell by shell, the error of the posterior
mean is smallest at the surface, holds to a depth of about a correlation
length, and has reached that of knowing nothing by the core-mantle boundary;
the section through two stations shows where the data stop and the prior
takes over.

Needs planetmodel and cartopy: the 'planetmodel' and 'sphere' extras.
"""

import matplotlib.pyplot as pyplot
import numpy as np

from pygeoinf2 import plotting
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.radial.ball import Sobolev

rng = np.random.default_rng(5)

CMB = 0.55  # the core-mantle boundary, as a fraction of the outer radius

# ---------------------------------------------------------------------------
# The space: H^2 on the mantle, so that points have values (order > 3/2).
# ---------------------------------------------------------------------------


def length_scale(r):
    """Longer at depth: 0.25 at the core-mantle boundary, 0.12 at the surface."""
    depth = (1.0 - np.asarray(r)) / (1.0 - CMB)
    return 0.12 + 0.13 * depth**2


# Left to itself the truncation is isotropic: the radial modes kept are those
# no shorter than the angular wavelength of degree lmax at the surface, fewer
# at each higher degree. A mantle wants finer resolution in depth than along
# it, and the same at every degree, which is what `radial_modes` says.
X = Sobolev(16, 2.0, length_scale, inner_radius=CMB, radial_modes=10)
isotropic = Sobolev(16, 2.0, length_scale, inner_radius=CMB)
print(X)
print(
    f"grid {X.grid_shape}: {X.radii.size} radial nodes from {X.radii[0]:.2f} to "
    f"{X.radii[-1]:.2f}, of which {X.interior_radii.size} lie in the mantle"
)
print(
    f"ten radial modes at every degree, dimension {X.dim}; the isotropic default "
    f"would keep {isotropic.basis.nmodes[0]} at degree 0 and "
    f"{isotropic.basis.nmodes[16]} at degree 16, dimension {isotropic.dim}"
)
print()

# ---------------------------------------------------------------------------
# A prior with a standard deviation that depends on depth.
# ---------------------------------------------------------------------------

# Twice as variable in the boundary layers as in the mid-mantle, and given on
# the padding too. Smooth in r, which in an annulus is all it has to be.
depth = (1.0 - X.radii) / (1.0 - CMB)
sigma = 0.01 * (1.0 + np.cos(2.0 * np.pi * np.clip(depth, 0.0, 1.0)) ** 2)
prior = X.sobolev_measure(1.5, pointwise_std=sigma[:, None, None] + X.zero())

raw = X.interior_values(X.pointwise_variance(X.eigenvalues**-1.5))[:, 0, 0]
print(
    "raw variance of A^-1.5, a function of radius alone: "
    f"{raw[0]:.3f} at the core-mantle boundary, {raw[-1]:.3f} at the surface"
)

# ---------------------------------------------------------------------------
# Surface stations, and the posterior.
# ---------------------------------------------------------------------------

stations = [(1.0, *X.random_point(rng=rng)[1:]) for _ in range(150)]
forward = X.point_evaluation_operator(stations)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.002)
problem = LinearForwardProblem(forward, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

posterior = LinearGaussianInversion(problem, prior)(data)
mean = posterior.expectation
print(
    f"{len(stations)} surface stations; the posterior mean fits them to "
    f"chi-squared {problem.chi_squared(mean, data):.1f}"
)


def misfit_at(radius):
    """Relative error of the posterior mean on the shell of one radius."""
    points = [(radius, *X.random_point(rng=rng)[1:]) for _ in range(400)]
    exact = X.evaluate(truth, points)
    return np.linalg.norm(X.evaluate(mean, points) - exact) / np.linalg.norm(exact)


print("relative error of the posterior mean, shell by shell:")
for radius in (1.0, 0.95, 0.9, 0.8, 0.7, CMB):
    print(f"   r = {radius:.2f}   {misfit_at(radius):.2f}")
print()

# ---------------------------------------------------------------------------
# Shells, as maps.
# ---------------------------------------------------------------------------

limit = 2.5 * sigma.max()
shared = dict(symmetric=True, vmin=-limit, vmax=limit)

figure, maps = plotting.subplots(X, columns=2)
plotting.plot(
    X,
    truth,
    ax=maps[0],
    title="Truth at the surface",
    contour=True,
    levels=24,
    **shared,
)
plotting.plot_shell(
    X,
    truth,
    radius=0.7,
    ax=maps[1],
    title="Truth at r = 0.7",
    contour=True,
    levels=24,
    colorbar_label="field",
    **shared,
)

# ---------------------------------------------------------------------------
# A section through two stations, and a profile beneath one of them.
# ---------------------------------------------------------------------------

first, second = stations[0][1:], stations[1][1:]

figure, panels = pyplot.subplots(1, 3, figsize=(13.0, 4.0), layout="constrained")
plotting.plot_section(
    X,
    truth,
    through=(first, second),
    ax=panels[0],
    colorbar=False,
    title="Truth, on the plane through two stations",
    **shared,
)
plotting.plot_section(
    X,
    mean,
    through=(first, second),
    ax=panels[1],
    colorbar_label="field",
    title="Posterior mean, same plane",
    **shared,
)
# One ray is one ray: it can do better or worse than its shell, and the errors
# printed above are the fair summary.
plotting.plot_profile(X, truth, *first, ax=panels[2], color="black", label="truth")
plotting.plot_profile(X, mean, *first, ax=panels[2], label="posterior mean")
panels[2].set_title("Beneath the first station")
panels[2].legend(fontsize=8)

print("two shells, two sections and a profile drawn")
plotting.show()
