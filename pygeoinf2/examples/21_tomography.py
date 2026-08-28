"""
21. A Bayesian inversion, end to end.

v1's ``work/tomo.py`` on v2: phase-velocity tomography on a sphere, from
travel-time anomalies along great-circle paths between real sources and
receivers.

The point of the example is what is *not* written. There is no mass matrix, no
Galerkin flag, and no conversion between a derivative and a gradient anywhere
in it -- and there are two of each hiding in the arithmetic. The forward
operator's adjoint returns a function because it was built from derivative
components; the posterior mean comes out of a solve whose normal equations were
assembled in whichever space is smaller, without being asked.

Needs pyshtools and cartopy, which come with the 'sphere' extra.
"""

import numpy as np

from pygeoinf2 import plotting
from pygeoinf2.inference import LinearGaussianInversion, LinearForwardProblem
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev

rng = np.random.default_rng(1)

# H^2 on the unit sphere: smooth enough that a path average is a bounded
# functional, which is what gives it a representer at all.
X = Sobolev(48, 2.0, 0.1)
print(f"model space: dimension {X.dim}")

# ---------------------------------------------------------------------------
# A real acquisition geometry.
# ---------------------------------------------------------------------------

receivers = X.stations(count=24, rng=rng)
sources = X.earthquakes(count=40, minimum_magnitude=5.5, rng=rng)
paths = [(source, receiver) for source in sources for receiver in receivers]
print(f"{len(sources)} sources x {len(receivers)} receivers = {len(paths)} paths")

# Assembled: the rows are known in closed form, so this costs one pass rather
# than one adjoint application per datum.
forward = X.path_average_operator(paths, count=16, dense=True)

noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
problem = LinearForwardProblem(forward, error=noise)

# ---------------------------------------------------------------------------
# A prior, and a synthetic truth drawn jointly with its data.
# ---------------------------------------------------------------------------

# A heat-kernel prior with a correlation length of roughly 15 degrees, and
# a pointwise standard deviation you can actually have an opinion about.
prior = X.heat_measure(0.17, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
print(
    f"truth has pointwise rms {X.grid_values(truth).std():.4f}; "
    f"data rms {data.std():.4f}"
)
print(
    f"chi-squared of the truth: {problem.chi_squared(truth, data):.1f} "
    f"on {problem.data_space.dim} data"
)
print()

# ---------------------------------------------------------------------------
# The posterior. One line, and the solve inside it is iterative and
# matrix-free: nothing here is ever assembled.
# ---------------------------------------------------------------------------

# The normal equations go in the data space by default, and here that is also
# the smaller side. It is the default because it is nearly always the smaller
# side -- a model space is a discretised function and grows with resolution,
# while the data are however many observations there are.
estimator = LinearGaussianInversion(problem, prior)
print(
    f"normal equations assembled in the {estimator.formalism}"
    f" (dim {problem.data_space.dim} rather than {X.dim})"
)

posterior = estimator(data)
error = X.norm(X.subtract(posterior.expectation, truth)) / X.norm(truth)
print(f"relative error of the posterior mean: {error:.3f}")
print("  the prior mean alone would give 1.000, so that is what the data bought")

# The data are fitted to within their errors, which is the check that the whole
# chain -- operator, adjoint, metric, solve -- is consistent.
fit = problem.chi_squared(posterior.expectation, data)
print(
    f"the posterior mean fits the data to chi-squared {fit:.0f}"
    f" on {problem.data_space.dim} data"
)
print()

# ---------------------------------------------------------------------------
# A property of the model, without forming the model-space covariance.
# ---------------------------------------------------------------------------

caps = X.geodesic_ball_average_operator(sources[:4], 0.15, dense=True)
property_posterior = estimator.push_forward(caps)(data)
truth_values = caps(truth)
means = property_posterior.expectation
deviations = np.sqrt(np.diag(property_posterior.covariance.matrix(form="components")))
print("cap averages, posterior against truth:")
for value, mean, deviation in zip(truth_values, means, deviations):
    inside = abs(value - mean) < 2.0 * deviation
    print(
        f"   truth {value:+.5f}   posterior {mean:+.5f} +/- {deviation:.5f}"
        f"   {'within 2 sd' if inside else 'OUTSIDE 2 sd'}"
    )
print()

# ---------------------------------------------------------------------------
# Four panels.
# ---------------------------------------------------------------------------

figure, axes = plotting.subplots(X, rows=2, columns=2)
panels = axes.ravel()

ax, _ = plotting.plot(
    X,
    truth,
    ax=panels[0],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="d ln c",
)
plotting.plot_paths(X, paths, ax=ax)
ax.set_title("Truth, with the ray network")

plotting.plot(
    X,
    posterior.expectation,
    ax=panels[1],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="d ln c",
)
panels[1].set_title("Posterior mean")

plotting.plot(
    X,
    X.subtract(posterior.expectation, truth),
    ax=panels[2],
    cmap="PRGn",
    symmetric=True,
    coasts=True,
    colorbar_label="error",
)
panels[2].set_title("Posterior mean minus truth")

ax, _ = plotting.plot(
    X,
    posterior.sample(rng=rng),
    ax=panels[3],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="d ln c",
)
plotting.plot_points(X, receivers, ax=ax, color="black")
ax.set_title("One posterior draw, with the receivers")

print("four panels drawn; matplotlib.pyplot.show() displays them")
