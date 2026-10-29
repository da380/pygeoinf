"""
15. Two fields tied through their latent noise.

Example 14 built a joint law as an independent product. Two fields are often
not independent: warm water tends to be salty, a density anomaly tends to come
with a velocity one. ``GaussianMeasure.from_product`` takes a ``correlation``
matrix ``R`` for that. Each field is drawn as ``L_i xi_i`` from its own factor,
and ``R`` correlates the white noises ``xi_i`` mode for mode, so that

    the marginal of each field is its own measure, unchanged;
    the cross-covariance of fields i and j is  R_ij L_i L_j*.

The consequence worth seeing: a field that is never measured is still informed,
through ``R`` and only through it.
"""

import numpy as np

from pygeoinf2 import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    RowLinearOperator,
)
from pygeoinf2.symmetric_space import circle

rng = np.random.default_rng(15)
X = circle.Sobolev(256, 2.0, 0.2)

# Two priors of different shape and size: a heat measure for the temperature,
# a Sobolev measure for the salinity, each calibrated by its pointwise
# standard deviation.
temperature = X.heat_measure(0.3, pointwise_std=1.0)
salinity = X.sobolev_measure(2.0, 0.3, pointwise_std=0.5)
R = np.array([[1.0, 0.8], [0.8, 1.0]])
joint = GaussianMeasure.from_product(
    [temperature, salinity], correlation=R, labels=("temperature", "salinity")
)
print("the joint prior lives on", joint.domain)
print()

# ---------------------------------------------------------------------------
# The marginals are the factors, and the cross-covariance is R_12 L_1 L_2*.
# ---------------------------------------------------------------------------

u = X.random(rng=rng)


def relative(a, b):
    return X.norm(X.subtract(a, b)) / X.norm(b)


for index, measure in enumerate((temperature, salinity)):
    marginal = joint.marginal(index)
    print(
        f"marginal {index} against its own measure: "
        f"{relative(marginal.covariance(u), measure.covariance(u)):.1e}"
    )
cross = joint.cross_covariance(0, 1)
expected = 0.8 * (temperature.covariance_factor @ salinity.covariance_factor.adjoint)
print(f"cross-covariance against 0.8 L_1 L_2*: {relative(cross(u), expected(u)):.1e}")

# At a point the correlation is 0.8 only when the two spectra have one shape;
# these two nearly do.
point = [1.0]
look = X.point_evaluation_operator(point)
pair = [
    (look @ c @ look.adjoint).matrix()[0, 0]
    for c in (cross, temperature.covariance, salinity.covariance)
]
draws = joint.samples(1000, rng=rng)
values = np.array(
    [[X.evaluate(t, point)[0], X.evaluate(s, point)[0]] for t, s in draws]
)
print(
    f"correlation of the two fields at a point: {pair[0] / np.sqrt(pair[1] * pair[2]):.3f},"
    f" sampled {np.corrcoef(values.T)[0, 1]:.3f}"
)
print()

# ---------------------------------------------------------------------------
# Measure the temperature only.
# ---------------------------------------------------------------------------

sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 40))
A = X.point_evaluation_operator(sensors)
D = A.codomain
forward = RowLinearOperator([A, LinearOperator.zero(X, codomain=D)])
problem = LinearForwardProblem(
    forward, error=GaussianMeasure.from_standard_deviation(D, 0.02)
)
truth, data = problem.synthetic_model_and_data(joint, rng=rng)
posterior = LinearGaussianInversion(problem, joint)(data)

unseen = posterior.marginal(1)
spread = np.sqrt(X.pointwise_variance_at(unseen, X.grid_axes[0][::16]))
print("forty temperature values, salinity never measured")
print(
    f"  salinity's pointwise standard deviation: 0.5 before, {spread.min():.3f}"
    f" to {spread.max():.3f} after"
)
print(
    f"  the floor a perfectly known temperature leaves: 0.5 sqrt(1 - 0.8^2)"
    f" = {0.5 * np.sqrt(1.0 - 0.8**2):.3f}"
)
error = relative(unseen.expectation, truth[1])
print(
    f"  relative error of its posterior mean {error:.2f}, against 1.00 for the"
    " prior's, from no salinity data at all"
)
print()

# With R the identity the same data would leave the salinity at its prior.
independent = GaussianMeasure.from_product([temperature, salinity])
alone = LinearGaussianInversion(problem, independent)(data).marginal(1)
print(
    "independent fields: salinity's standard deviation stays at "
    f"{np.sqrt(X.pointwise_variance_at(alone, [1.0]))[0]:.3f}"
)
