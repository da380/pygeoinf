"""
33. A planet in layers: fields that are smooth in each and jump between them.

The density of example 32 was one smooth profile, and no planet's is: it jumps
at the boundary of the core and again at the inner core. A field like that is
in none of the ``sem1d`` spaces, whose smoothness is what each is built on. It
is in their *direct sum*: one space per layer, each on its own padded mesh with
its own correlation length, and a vector a tuple of layer profiles.
``sem1d.layered.Layered`` is that direct sum, with what a direct sum cannot
know added -- that the layers are adjacent. Which layer a radius is in, the
two values at an interface, the jump across it, integrals over the whole, and
a prior given layer by layer, under which the layers are independent: which is
what a discontinuity means.

The data here are of three kinds at once, and the space makes no difference
between them: integrals over the whole planet (mass and moment of inertia),
the jump across the core-mantle boundary (as a reflected wave sees it), and a
few densities in the mantle.

Last, the converse. Two layers made continuous across the interface they share
are the prior *conditioned* on the jump vanishing there.

Needs planetmodel, which comes with the 'planetmodel' extra.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, LinearOperator, plotting
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.sem1d.layered import Layered

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Three layers, each with its own resolution and correlation length.
# ---------------------------------------------------------------------------

ICB, CMB = 0.19, 0.55
X = Layered.radial(
    [0.0, ICB, CMB, 1.0],
    [10, 20, 30],  # a list is layer by layer; anything else is given to all
    order=1.0,
    length_scale=[0.08, 0.15, 0.2],
    labels=["inner core", "outer core", "mantle"],
)
print(X)
print(f"interfaces at {X.interfaces}, where a field has two values")
print()

# ---------------------------------------------------------------------------
# A reference that jumps, given piece by piece, and the prior about it.
# ---------------------------------------------------------------------------

# One smooth piece per layer, each even in r as a field regular at the centre
# is. A single function would do for a field that does not jump; one that does
# is asked about the interface by both layers, and cannot tell them apart.
reference = X.project_function(
    [
        lambda r: 13.1 - 8.8 * r**2,
        lambda r: 12.6 - 8.0 * r**2,
        lambda r: 6.2 - 2.8 * r**2,
    ]
)
for name, radius in (("inner-core boundary", ICB), ("core-mantle boundary", CMB)):
    below = X.evaluate(reference, [radius], side="below")[0]
    above = X.evaluate(reference, [radius], side="above")[0]
    print(f"reference at the {name}: {below:.3f} below, {above:.3f} above")

# More uncertain in the core than in the mantle. Independent layers.
prior = X.sobolev_measure(1.5, expectation=reference, pointwise_std=[0.6, 0.5, 0.25])
print()

# ---------------------------------------------------------------------------
# Three kinds of data.
# ---------------------------------------------------------------------------

mass = X.integral_functional()
radius_squared = X.project_function(lambda r: r**2, extension="constant")
inertia = (2.0 / 3.0) * (mass @ X.multiplication_operator(radius_squared))
jump = X.jump_operator(1)  # the second interface: the core-mantle boundary
depths = np.array([0.6, 0.7, 0.8, 0.9, 0.98])

rows = np.vstack(
    [
        mass.matrix(form="galerkin"),
        inertia.matrix(form="galerkin"),
        jump.matrix(form="galerkin"),
        X.basis_matrix(depths),
    ]
)
forward = LinearOperator.from_matrix(
    X, EuclideanSpace(rows.shape[0]), rows, form="galerkin"
)
noise = GaussianMeasure.from_standard_deviations(
    forward.codomain, np.array([1e-3, 1e-3, 0.05] + [0.03] * depths.size)
)
problem = LinearForwardProblem(forward, error=noise)

truth = prior.sample(rng=rng)
data = problem.synthetic_data(truth, rng=rng)
posterior = LinearGaussianInversion(problem, prior)(data)

print(
    f"mass {mass(truth):.4f} and moment of inertia {inertia(truth):.4f}; the "
    f"posterior mean gives {mass(posterior.expectation):.4f} and "
    f"{inertia(posterior.expectation):.4f}"
)
print(
    f"jump at the core-mantle boundary: truth {jump(truth)[0]:+.3f}, measured "
    f"{data[2]:+.3f}, posterior mean {jump(posterior.expectation)[0]:+.3f}"
)


def spread(measure, radii, side):
    look = X.point_evaluation_operator(radii, side=side)
    covariance = (look @ measure.covariance @ look.adjoint).matrix(form="components")
    return np.sqrt(np.diag(covariance))


print("pointwise standard deviation, prior and posterior, layer by layer:")
for name, radius in (("inner core", 0.1), ("outer core", 0.4), ("mantle", 0.8)):
    before = spread(prior, [radius], None)[0]
    after = spread(posterior, [radius], None)[0]
    print(f"   {name:10s} r = {radius:.1f}   {before:.3f} -> {after:.3f}")
print(
    "   the mantle is seen directly; the core only through the integrals and "
    "the jump, and the inner core, which holds little mass, hardly at all"
)
print()

# ---------------------------------------------------------------------------
# Continuity, by conditioning.
# ---------------------------------------------------------------------------

welded = prior.condition(X.jump_operator(0), np.zeros(1))
draws = [welded.sample(rng=rng) for _ in range(200)]
at_icb = np.array([X.jump_operator(0)(x)[0] for x in draws])
at_cmb = np.array([jump(x)[0] - jump(reference)[0] for x in draws])
print(
    "conditioned on no jump at the inner-core boundary: the jump there is "
    f"{np.abs(at_icb).max():.1e} at most over 200 draws, and at the core-mantle "
    f"boundary still varies by {at_cmb.std():.2f} about the reference's"
)
print()

# ---------------------------------------------------------------------------
# Two panels.
# ---------------------------------------------------------------------------

figure, (left, right) = plotting.subplots(X, columns=2)

plotting.plot(X, reference, ax=left, color="black", label="reference")
for draw in (prior.sample(rng=rng) for _ in range(3)):
    plotting.plot(X, draw, ax=left, linewidth=0.8)
left.set_title("The prior: independent layers")
left.set_xlabel("radius")
left.legend(fontsize=7)

# The band layer by layer, so that it breaks where the field does. At the two
# ends of a layer the side is named, which matters only on an interface; the
# centre is left out, a point there wanting an order of three halves.
for lower, upper in X.layer_bounds:
    radii = np.linspace(max(lower, 0.01), upper, 60)
    sides = ["above"] + [None] * (radii.size - 2) + ["below"]
    rows = np.vstack([X.basis_matrix([r], side=s) for r, s in zip(radii, sides)])
    look = LinearOperator.from_matrix(
        X, EuclideanSpace(radii.size), rows, form="galerkin"
    )
    centre = look(posterior.expectation)
    width = np.sqrt(
        np.diag((look @ posterior.covariance @ look.adjoint).matrix(form="components"))
    )
    right.fill_between(
        radii,
        centre - 2.0 * width,
        centre + 2.0 * width,
        color="tab:blue",
        alpha=0.25,
        linewidth=0.0,
    )
plotting.plot(
    X, posterior.expectation, ax=right, color="tab:blue", label="posterior mean"
)
plotting.plot(X, truth, ax=right, color="black", linewidth=0.8, label="truth")
right.plot(depths, data[3:], "o", markersize=4, color="tab:orange", label="densities")
right.set_title("Mass, moment of inertia, a jump, five densities")
right.set_xlabel("radius")
right.legend(fontsize=7)

print("two panels drawn")
plotting.show()
