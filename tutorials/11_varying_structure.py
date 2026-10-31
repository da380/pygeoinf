# %% [markdown]
# # Tutorial 11: spaces whose structure varies
#
# The circle, the torus and the sphere of the earlier tutorials are
# *homogeneous*: every point looks like every other, so the Laplacian's
# eigenfunctions are Fourier modes or spherical harmonics, and a prior has
# one correlation length everywhere. Real media are not like that. A
# sedimentary column is finely layered near the top and massive at depth; a
# planet's mantle has boundary layers; the density of a planet is a function
# of radius, where a shell near the surface holds far more mass than one near
# the centre.
#
# The spaces in `pygeoinf.interval` and `pygeoinf.ball` are built for that.
# Each is the eigenbasis of
#
# $$A = 1 - \nabla \cdot \big(L(x)^2\, \nabla\big)$$
#
# for a length scale $L(x)$ *you* choose, computed on a spectral-element
# mesh. Everything else is as before: priors are built from $A$'s spectrum,
# point evaluation and integrals are operators, and the inversions of
# tutorials 6 to 10 run unchanged. Needs `planetmodel`
# (`poetry install --extras planetmodel`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    CholeskySolver,
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    plotting,
)
from pygeoinf.ball import Sobolev as BallSobolev
from pygeoinf.interval import Sobolev as IntervalSobolev
from pygeoinf.interval.radial import Layered
from pygeoinf.interval.radial import Sobolev as RadialSobolev

rng = np.random.default_rng(11)

# %% [markdown]
# ## An interval with a varying length scale
#
# A profile on $[0, 2]$ whose correlation length grows from 0.04 at the top
# to 0.3 at the bottom. The space is $H^1$ built from that $A$, with 160
# modes.


# %%
def length_scale(x):
    return 0.04 + 0.26 * (np.asarray(x) / 2.0) ** 2


X = IntervalSobolev(160, 1.0, length_scale, upper=2.0)
print(X)
print(
    f"the mesh runs from {X.nodes[0]:.2f} to {X.nodes[-1]:.2f}: padded by {X.padding}"
)

# %% [markdown]
# The mesh is longer than the interval. $A$ needs a boundary condition at
# each end, and a boundary condition leaves its mark on the eigenfunctions
# for a length scale or two; the padding keeps that mark off the domain you
# care about. It is sized from the length scale at each end, which is why it
# is short at the top and long at the bottom.
#
# Nothing here is homogeneous, so a prior's variance is not the same
# everywhere: the raw covariance $A^{-p}$ is more variable where $L$ is short.
# `pointwise_std=` calibrates it to a given standard deviation — a number, or
# a field — without changing the correlation structure.

# %%
raw = X.interior_values(X.pointwise_variance(X.eigenvalues**-1.0))
print(f"raw variance of A^-1: {raw[0]:.2f} at the top, {raw[-1]:.2f} at the bottom")
prior = X.sobolev_measure(1.0, pointwise_std=0.5)

fig, ax = plt.subplots(figsize=(8.0, 3.0), layout="constrained")
for _ in range(3):
    plotting.plot(X, prior.sample(rng=rng), ax=ax, padding=True, lw=1.0)
ax.set_xlabel("depth")
ax.set_title(
    "Prior draws: short correlation at the top, long at the bottom; padding shaded"
)

# %% [markdown]
# ## A function of radius
#
# The density of a planet as a function of radius, on the unit ball. The
# natural inner product of such functions is the one the ball's volume gives
# them, $(f, g) = \int_0^1 f(r)\, g(r)\, 4\pi r^2\, dr$ — so a shell near the
# surface counts for more than one near the centre, because it is bigger.
# That is what `interval.radial` is: the same construction, under $r^2\,dr$.
#
# Two data a planet always supplies are its mass and its moment of inertia,
# both integrals of the density: $\int \rho\, dV$ and
# $\tfrac{2}{3}\int \rho\, r^2\, dV$. The space provides the first as a
# functional, and the second is the first composed with a multiplication.

# %%
R = RadialSobolev(60, 1.0, 0.2)
print(R)
print(
    f"point values exist above order {R.point_evaluation_order(points=[0.5]):g} at r = 0.5,"
    f" and above {R.point_evaluation_order(points=[0.0]):g} at the centre"
)

reference = R.project_function(lambda r: 3.0 - 2.0 * r**2)
density_prior = R.sobolev_measure(1.5, expectation=reference, pointwise_std=0.4)
mass = R.integral_functional()
radius_squared = R.project_function(lambda r: r**2, extension="constant")
inertia = (2.0 / 3.0) * (mass @ R.multiplication_operator(radius_squared))

rows = np.vstack([mass.matrix(form="galerkin"), inertia.matrix(form="galerkin")])
forward = LinearOperator.from_matrix(R, EuclideanSpace(2), rows, form="galerkin")
problem = LinearForwardProblem(
    forward, error=GaussianMeasure.from_standard_deviation(forward.codomain, 1e-3)
)
truth = density_prior.sample(rng=rng)
posterior = LinearGaussianInversion(problem, density_prior)(
    problem.synthetic_data(truth, rng=rng)
)

radii = np.linspace(0.02, 1.0, 100)
look = R.point_evaluation_operator(radii)


def spread(measure):
    covariance = (look @ measure.covariance @ look.adjoint).matrix(form="components")
    return np.sqrt(np.diag(covariance))


before, after = spread(density_prior), spread(posterior)
for r in (0.1, 0.5, 0.9):
    i = np.argmin(np.abs(radii - r))
    print(f"standard deviation at r = {r}: {before[i]:.3f} -> {after[i]:.3f}")

# %%
fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
mean = look(posterior.expectation)
ax.fill_between(
    radii, mean - 2 * after, mean + 2 * after, alpha=0.3, label="posterior, 2 sd"
)
ax.fill_between(
    radii,
    look(reference) - 2 * before,
    look(reference) + 2 * before,
    alpha=0.12,
    color="gray",
    label="prior, 2 sd",
)
plotting.plot(R, truth, ax=ax, color="black", lw=0.8, label="truth")
ax.set_xlabel("radius")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_title("Two numbers, mass and moment of inertia, and what they say")

# %% [markdown]
# Two integrals tie down the outer half of the planet and leave the centre
# almost as free as the prior had it: a sphere of small radius holds almost
# no mass, so no integral over the planet can see it. The $r^2$ in the inner
# product is what makes the space know that — on the interval of the
# previous section, a point near zero would count as much as any other.
#
# ## A ball
#
# The same in three dimensions: a field in the unit ball, written as
# spherical harmonics in angle (to degree 12) times eigenfunctions of the
# radial $A$ (eight per degree). Point values need order above 3/2 in three
# dimensions, so this is $H^2$. With sixty data the normal operator is a
# sixty-by-sixty matrix, so a direct solver is the right tool, where the
# default conjugate gradients would sit near their iteration cap.

# %%
Y = BallSobolev(12, 2.0, 0.2, radial_modes=8)
print(Y)
ball_prior = Y.sobolev_measure(2.0, pointwise_std=1.0)
field_truth = ball_prior.sample(rng=rng)

stations = [(1.0, *Y.random_point(rng=rng)[1:]) for _ in range(60)]
evaluate = Y.point_evaluation_operator(stations)
problem = LinearForwardProblem(
    evaluate, error=GaussianMeasure.from_standard_deviation(evaluate.codomain, 0.05)
)
data = problem.synthetic_data(field_truth, rng=rng)
ball_posterior = LinearGaussianInversion(problem, ball_prior, solver=CholeskySolver())(
    data
)
print(f"{len(stations)} stations on the surface")


def error_at(radius, count=300):
    points = [(radius, *Y.random_point(rng=rng)[1:]) for _ in range(count)]
    exact = Y.evaluate(field_truth, points)
    return np.linalg.norm(
        Y.evaluate(ball_posterior.expectation, points) - exact
    ) / np.linalg.norm(exact)


for radius in (1.0, 0.9, 0.7, 0.4):
    print(
        f"relative error of the posterior mean on the shell r = {radius}: {error_at(radius):.2f}"
    )

# %% [markdown]
# Surface data say less and less with depth, as they must. A field in a
# ball is looked at by cutting it: a **section** through any plane — here the
# one through the first two stations and the centre — and a **profile** along
# a radius.

# %%
first, second = stations[0][1:], stations[1][1:]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), layout="constrained")
limit = 2.5
plotting.plot_section(
    Y,
    field_truth,
    through=(first, second),
    ax=axes[0],
    colorbar=False,
    symmetric=True,
    vmin=-limit,
    vmax=limit,
    title="Truth",
)
plotting.plot_section(
    Y,
    ball_posterior.expectation,
    through=(first, second),
    ax=axes[1],
    symmetric=True,
    vmin=-limit,
    vmax=limit,
    title="Posterior mean",
)
plotting.plot_section_points(
    Y, stations, ax=axes[1], through=(first, second), size=18.0
)
plotting.plot_profile(Y, field_truth, *first, ax=axes[2], color="black", label="truth")
plotting.plot_profile(
    Y, ball_posterior.expectation, *first, ax=axes[2], label="posterior mean"
)
axes[2].set_title("Beneath the first station")
axes[2].legend(fontsize=8)

# %% [markdown]
# ## Layers
#
# Finally, the structures of this tutorial and the last together: a planet
# in layers, each its own radial space with its own length scale and
# resolution, joined in a `Layered` direct sum. A prior on it is independent
# layer by layer, so its draws jump at the boundaries — the inner-core
# boundary at 0.19 and the core-mantle boundary at 0.55.

# %%
planet = Layered(
    [0.0, 0.19, 0.55, 1.0],
    [10, 20, 30],
    order=1.0,
    length_scale=[0.08, 0.15, 0.2],
    labels=["inner core", "outer core", "mantle"],
)
planet_reference = planet.project_function(
    [
        lambda r: 13.1 - 8.8 * r**2,
        lambda r: 12.6 - 8.0 * r**2,
        lambda r: 6.2 - 2.8 * r**2,
    ]
)
planet_prior = planet.sobolev_measure(
    2.0, expectation=planet_reference, pointwise_std=[0.6, 0.5, 0.25]
)

fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
plotting.plot(planet, planet_reference, ax=ax, color="black", label="reference")
for _ in range(3):
    plotting.plot(planet, planet_prior.sample(rng=rng), ax=ax, lw=0.8)
ax.set_xlabel("radius")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_title("A layered planet: one space per layer, jumps between")

# %% [markdown]
# Example 36 inverts this planet's mass, moment of inertia, a jump and a
# handful of densities together, and welds two layers by conditioning, as
# tutorial 10 did on an interval; example 34 does the mantle of the ball
# section above at a resolution worth looking at, with a standard deviation
# that varies with depth.

# %%
plotting.show()
