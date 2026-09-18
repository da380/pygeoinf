"""
32. A density profile from a planet's mass and moment of inertia.

The oldest inverse problem in planetary science, and the one that shows what
the ``r^2 dr`` measure is for. A density that depends on radius alone is a
field in a ball, and the two numbers anyone can measure from outside are
integrals of it over the volume:

    mass                 M = 4 pi int rho r^2 dr
    moment of inertia    C = (8 pi / 3) int rho r^4 dr

Two data against a function: the answer is a distribution. Both integrals
weigh a shell by its volume or more, so they tie the outer half of the profile
down and say almost nothing about the centre, where a sphere of small radius
holds no mass. Add a handful of densities at known depths and the rest comes.

The space is ``sem1d.radial``: functions of radius under the volume
measure, the part of degree zero of a ball. It is not the interval of example
30, whose measure is ``dx`` -- there the mass would be ``int rho dx``, and a
shell at the surface would count for no more than one at the centre.

Needs planetmodel, which comes with the 'planetmodel' extra.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, LinearOperator, plotting
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.sem1d.radial import Sobolev

rng = np.random.default_rng(1)

# ---------------------------------------------------------------------------
# The space: H^1 of a unit ball, correlation length a fifth of the radius.
# ---------------------------------------------------------------------------

# Order one: above one half, so a density at a radius r > 0 has a value. At
# the centre itself that would take three halves, the centre being a point of
# space and not a sphere.
X = Sobolev(60, 1.0, 0.2)
print(X)
print(
    f"padded to r = {X.nodes[-1]:.2f} under the default Robin condition, two "
    f"length scales where the natural condition wants four; none at the centre"
)
print(
    f"a point has a value above order {X.point_evaluation_order(points=[0.5]):g} "
    f"at r = 0.5 and above {X.point_evaluation_order(points=[0.0]):g} at the centre"
)
print()

# ---------------------------------------------------------------------------
# The reference model, the prior about it, and the two functionals.
# ---------------------------------------------------------------------------


def reference(r):
    """Denser towards the centre, and even in r, as a field regular there is."""
    return 3.0 - 2.0 * r**2


# Fitted over the domain alone, which is what `project_function` does unless
# asked to continue the function's own values across the padding instead.
expectation = X.project_function(reference)
prior = X.sobolev_measure(1.5, expectation=expectation, pointwise_std=0.4)

volume = X.integral_functional()
radius_squared = X.project_function(lambda r: r**2, extension="constant")
mass = volume
inertia = (2.0 / 3.0) * (volume @ X.multiplication_operator(radius_squared))

exact_mass = 4.0 * np.pi * (3.0 / 3.0 - 2.0 / 5.0)
exact_inertia = (8.0 * np.pi / 3.0) * (3.0 / 5.0 - 2.0 / 7.0)
print(
    f"mass of the reference {mass(expectation):.6f} (exactly {exact_mass:.6f}), "
    f"moment of inertia {inertia(expectation):.6f} (exactly {exact_inertia:.6f})"
)
print()

# ---------------------------------------------------------------------------
# Two data, then seven.
# ---------------------------------------------------------------------------

truth = prior.sample(rng=rng)
depths = np.array([0.95, 0.8, 0.65, 0.45, 0.25])

# The rows of the forward operator are derivative components: what each datum
# does to the coefficients. `from_matrix(form="galerkin")` puts the metric in
# the adjoint once, which is the point of example 5.
integrals = np.vstack([mass.matrix(form="galerkin"), inertia.matrix(form="galerkin")])
points = X.basis_matrix(depths)


def posterior_from(rows, sigmas):
    forward = LinearOperator.from_matrix(
        X, EuclideanSpace(rows.shape[0]), rows, form="galerkin"
    )
    noise = GaussianMeasure.from_standard_deviations(
        forward.codomain, np.asarray(sigmas)
    )
    problem = LinearForwardProblem(forward, error=noise)
    data = problem.synthetic_data(truth, rng=rng)
    return LinearGaussianInversion(problem, prior)(data), problem, data


along = np.linspace(0.02, 1.0, 120)
look = X.point_evaluation_operator(along)


def band(measure):
    covariance = (look @ measure.covariance @ look.adjoint).matrix(form="components")
    return look(measure.expectation), np.sqrt(np.diag(covariance))


two, problem, data = posterior_from(integrals, [1e-3, 1e-3])
print(
    f"mass and moment of inertia alone: measured {data[0]:.4f} and {data[1]:.4f}, "
    f"the posterior mean gives {mass(two.expectation):.4f} and "
    f"{inertia(two.expectation):.4f}, the truth {mass(truth):.4f} and "
    f"{inertia(truth):.4f}"
)
mean_two, band_two = band(two)
mean_prior, band_prior = band(prior)
inner, outer = np.argmin(np.abs(along - 0.1)), np.argmin(np.abs(along - 0.8))
print(
    f"   standard deviation at r = 0.8: {band_prior[outer]:.3f} before, "
    f"{band_two[outer]:.3f} after; at r = 0.1: {band_prior[inner]:.3f} before, "
    f"{band_two[inner]:.3f} after"
)
print(
    "   two numbers tie down the outer half, where a shell holds mass, and leave "
    "the centre nearly as free as it was: a sphere of small radius holds none"
)

seven, _, _ = posterior_from(np.vstack([integrals, points]), [1e-3, 1e-3] + [0.02] * 5)
mean_seven, band_seven = band(seven)
print(
    f"with densities at five depths as well it falls to {band_seven.mean():.3f}, "
    f"and the truth lies within two standard deviations at "
    f"{np.mean(np.abs(look(truth) - mean_seven) <= 2.0 * band_seven):.0%} of radii"
)
print()

# ---------------------------------------------------------------------------
# Two panels.
# ---------------------------------------------------------------------------

figure, panels = plotting.subplots(X, columns=2)
for panel, mean, spread, title in (
    (panels[0], mean_two, band_two, "Mass and moment of inertia"),
    (panels[1], mean_seven, band_seven, "And five densities"),
):
    panel.fill_between(along, mean - 2.0 * spread, mean + 2.0 * spread, alpha=0.25)
    panel.plot(along, mean, label="posterior mean")
    plotting.plot(X, truth, ax=panel, color="black", linewidth=0.8, label="truth")
    plotting.plot(
        X, expectation, ax=panel, linestyle=":", color="gray", label="reference"
    )
    panel.set_xlabel("radius")
    panel.set_title(title)
panels[1].plot(depths, X.evaluate(truth, depths), "o", markersize=4)
panels[0].legend(fontsize=7)

print("two panels drawn")
plotting.show()
