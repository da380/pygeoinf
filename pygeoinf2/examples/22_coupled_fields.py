"""
22. Two coupled fields, seen through one shared physical chain.

v1's ``work/dynamic_topography.py`` on v2: a density anomaly and a basal
traction, both on a sphere, observed only through the geoid and the surface
topography they jointly produce.

What makes it different from example 21 is that the unknown is a *pair*. The
model space is a direct sum, the prior is a product measure, and the forward
operator is a block matrix built from two degree-dependent symbols and the
flexure operator of example 20. Nothing in the inference layer notices: a
direct sum is a Hilbert space, so ``LinearGaussianInversion(problem, prior)`` is the same call
it was before.

The physics is a two-layer gravity calculation, kept only in enough detail to
be a real coupled problem. The constants are not the point.

Needs pyshtools and cartopy, which come with the 'sphere' extra.
"""

import numpy as np
import matplotlib.pyplot as plt

from pygeoinf2 import plotting
from pygeoinf2.algebra.direct_sum import (
    DirectSum,
    BlockDiagonalLinearOperator,
    BlockLinearOperator,
    ColumnLinearOperator,
)
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.inference import LinearGaussianInversion, LinearForwardProblem
from pygeoinf2.numerics.preconditioners import JacobiPreconditioner
from pygeoinf2.numerics.solvers import CGSolver, CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev

rng = np.random.default_rng(2)

GRAVITY = 9.8
NEWTON = 6.6743e-11
LAYER = 2.0e4
RADIUS = 6.371e6
POISSON = 0.25
DENSITY_LOAD, DENSITY_MANTLE = 3000.0, 3400.0

X = Sobolev(32, 2.0, 0.1 * RADIUS, radius=RADIUS)
model_space = DirectSum([X, X], labels=("density", "traction"))
print(f"field space dimension {X.dim}; model space {model_space.dim}")

# ---------------------------------------------------------------------------
# The flexure step: a traction deflects the plate.
# ---------------------------------------------------------------------------

rigidity = 1.0e23
buoyancy = (DENSITY_MANTLE - DENSITY_LOAD) * GRAVITY
flexure = X.inverse_flexural_operator(rigidity, POISSON, buoyancy)

# [density, traction] -> [density, flexure]. The projections onto the two
# summands are the only place the pair structure is mentioned.
density_of, traction_of = model_space.projection(0), model_space.projection(1)
step_one = ColumnLinearOperator([density_of, flexure @ traction_of])

# ---------------------------------------------------------------------------
# The gravity step: both fields make potential, with degree-dependent symbols.
# ---------------------------------------------------------------------------

degrees = X.degrees.astype(float)
density_to_potential = X.spectral_operator(
    -4.0 * np.pi * NEWTON * RADIUS * LAYER / (2.0 * degrees + 1.0)
)
flexure_to_potential = X.spectral_operator(
    -4.0
    * np.pi
    * NEWTON
    * RADIUS
    * DENSITY_LOAD
    * (
        1.0
        + (DENSITY_MANTLE / DENSITY_LOAD - 1.0)
        * (1.0 - LAYER / RADIUS) ** (degrees + 2.0)
    )
    / (2.0 * degrees + 1.0)
)

step_two = BlockLinearOperator(
    [
        [density_to_potential, flexure_to_potential],
        [LinearOperator.zero(X), LinearOperator.identity(X)],
    ]
)
physics = step_two @ step_one

# ---------------------------------------------------------------------------
# What is actually observed: low-degree coefficients of geoid and topography.
# ---------------------------------------------------------------------------

geoid = (-1.0 / GRAVITY) * X.coefficient_operator(lmax=12)
topography = X.coefficient_operator(lmax=20)
observation = BlockDiagonalLinearOperator([geoid, topography])

forward = observation @ physics
print(f"forward operator: {model_space.dim} -> {forward.codomain.dim}")
print()

# ---------------------------------------------------------------------------
# A product prior: two fields, independent a priori, coupled only by the data.
# ---------------------------------------------------------------------------

prior = GaussianMeasure.from_product(
    [
        X.sobolev_measure(2.0, 5.0e5, pointwise_std=10.0),  # density anomaly
        X.sobolev_measure(1.25, 1.0e5, pointwise_std=2.0e6),  # basal traction
    ]
)

noise = GaussianMeasure.from_standard_deviation(forward.codomain, 1.0e-3)
problem = LinearForwardProblem(forward, error=noise)

truth, data = problem.synthetic_model_and_data(prior, rng=rng)
density_truth, traction_truth = truth
# Through grid_values, because a sphere's vectors are SHGrid objects (D-1)
# rather than arrays and do not carry numpy's methods.
print(f"density anomaly rms {X.grid_values(density_truth).std():.3f} kg/m^3")
print(f"basal traction rms  {X.grid_values(traction_truth).std():.3e} Pa")
print(
    f"chi-squared of the truth: {problem.chi_squared(truth, data):.1f} "
    f"on {problem.data_space.dim} data"
)
print()

# ---------------------------------------------------------------------------
# One call. The direct sum is a Hilbert space, so nothing here is special.
# ---------------------------------------------------------------------------

# The solve is iterative, as it should be for a problem of this size — nothing
# here is ever assembled. But the two fields carry wildly different units, a
# density in kg/m^3 against a traction in Pa, so the normal operator's
# eigenvalues span 9e12 to 7e18: badly *scaled* rather than badly conditioned.
# Unpreconditioned conjugate gradients loses orthogonality on that and reports
# a residual of nan. Jacobi is the whole fix, because a scaling problem is
# exactly what a diagonal preconditioner is for.
solver = CGSolver(rtol=1e-10, maxiter=5000).with_preconditioner(JacobiPreconditioner())
estimator = LinearGaussianInversion(problem, prior, solver=solver)
print(f"assembled in the {estimator.formalism}")
posterior = estimator(data)
density_mean, traction_mean = posterior.expectation

for name, truth_field, mean in [
    ("density", density_truth, density_mean),
    ("traction", traction_truth, traction_mean),
]:
    error = X.norm(X.subtract(mean, truth_field)) / X.norm(truth_field)
    print(f"  {name:9s} relative error of the posterior mean {error:.3f}")
print()


# ---------------------------------------------------------------------------
# What the second data channel buys.
# ---------------------------------------------------------------------------


def coupling_of(estimator):
    """How correlated the two fields are a posteriori, as a Frobenius ratio.

    A trace would not do: a cross-covariance block sums signed correlations and
    can vanish while every one of them is large.

    This *forms* three covariance blocks, so it applies the inverse normal
    operator about three thousand times. That is the case where a direct solver
    earns its keep: one factorisation, then three thousand cheap triangular
    solves, against three thousand independent Krylov runs. The posterior above
    is iterative because a posterior mean is one solve and the problem might be
    large; this diagnostic is dense because a dense answer is what it asks for.
    Which solver is right is a question about how many times you will use the
    inverse, not about how large the problem is.
    """
    covariance = estimator.with_solver(CholeskySolver()).covariance

    def block(first, second):
        return (first @ covariance @ second.adjoint).matrix(form="components")

    def size(matrix):
        return float(np.sqrt(np.sum(matrix**2)))

    return size(block(density_of, traction_of)) / np.sqrt(
        size(block(density_of, density_of)) * size(block(traction_of, traction_of))
    )


# Topography sees the flexure, and so the traction, on its own. That breaks the
# trade-off: given the topography, the geoid is left constraining the density.
print(f"joint data, posterior coupling of the two fields: {coupling_of(estimator):.4f}")

# Take the topography away and the fields have to share one channel.
geoid_only = LinearForwardProblem(
    geoid @ physics.codomain.projection(0) @ physics,
    error=GaussianMeasure.from_standard_deviation(geoid.codomain, 1.0e-3),
)
print(
    "geoid alone,                                    "
    f"{coupling_of(LinearGaussianInversion(geoid_only, prior, solver=solver)):.4f}"
)
print("which is what a second, differently-sensitive observable is worth")
print()

# ---------------------------------------------------------------------------
# Four panels.
# ---------------------------------------------------------------------------

figure, axes = plotting.subplots(X, rows=2, columns=2)
panels = axes.ravel()
for index, (name, field, unit) in enumerate(
    [
        ("Density anomaly, truth", density_truth, "kg/m^3"),
        ("Density anomaly, posterior mean", density_mean, "kg/m^3"),
        ("Basal traction, truth", traction_truth, "Pa"),
        ("Basal traction, posterior mean", traction_mean, "Pa"),
    ]
):
    plotting.plot(
        X,
        field,
        ax=panels[index],
        cmap="RdBu_r",
        symmetric=True,
        coasts=True,
        colorbar_label=unit,
    )
    panels[index].set_title(name)

print("four panels drawn; matplotlib.pyplot.show() displays them")
plt.show()
