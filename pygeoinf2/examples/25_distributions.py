"""
25. Looking at the answer: marginals and corner plots.

An inversion on a function space produces a posterior nobody can look at — it
is a measure on thousands of dimensions. ``push_forward`` turns it into a
measure on a handful of *properties*, and this is what you do with that: the
end of the estimator chain, and the reason a property space is the thing worth
asking about rather than a convenience.

Two kinds of measure are drawn here, and the difference matters. A Gaussian is
drawn exactly, from its mean and covariance. A measure that has been pushed
through a *non-linear* property map is no longer Gaussian and has no covariance
to draw from, so it is drawn from samples instead — the same picture, made a
different way, and honest about which it is.

Needs pyshtools and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

from pygeoinf2 import plotting
from pygeoinf2.algebra.operators import Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.numerics.solvers import CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev

rng = np.random.default_rng(2)

# ---------------------------------------------------------------------------
# A tomographic posterior, as in example 21.
# ---------------------------------------------------------------------------

X = Sobolev(32, 2.0, 0.1)
receivers = X.stations(count=12, rng=rng)
sources = X.earthquakes(count=30, minimum_magnitude=5.5, rng=rng)
paths = [(source, receiver) for source in sources for receiver in receivers]

forward = X.path_average_operator(paths, count=16, dense=True)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
problem = LinearForwardProblem(forward, error=noise)
prior = X.heat_measure(0.03, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

# Cholesky, deliberately, and this is the only place in the examples where a
# direct solver is the right default rather than an exception. Sampling by
# randomise-then-optimise costs *one solve per draw* — a prior sample, a noise
# sample, and one application of the Kalman gain — so the few thousand draws
# below are a few thousand solves. That is section 27.5's criterion, not the
# problem's size: apply an inverse once and iterate, apply it thousands of
# times and factorise.
estimator = LinearGaussianInversion(problem, prior, solver=CholeskySolver())
print(f"model space {X.dim}, data space {problem.data_space.dim}")

# ---------------------------------------------------------------------------
# Four properties, and their joint posterior.
# ---------------------------------------------------------------------------

# Average velocity anomaly in four caps. A property space of dimension four is
# something a person can look at; the model space is not.
caps = X.geodesic_ball_average_operator(sources[:4], 0.2, dense=True)
posterior = estimator.push_forward(caps)(data)
prior_properties = prior.push_forward(caps)
cap_truth = caps.codomain.to_components(caps(truth))

print(f"property space dimension {caps.codomain.dim}")
print("cap averages, posterior against truth:")
mean, covariance, _ = plotting.moments(posterior)
for index, (value, centre) in enumerate(zip(cap_truth, mean)):
    deviation = np.sqrt(covariance[index, index])
    inside = abs(value - centre) < 2.0 * deviation
    print(
        f"   cap {index}: truth {value:+.5f}   posterior {centre:+.5f}"
        f" +/- {deviation:.5f}   {'within 2 sd' if inside else 'OUTSIDE 2 sd'}"
    )
print()

axes = plotting.plot_corner(
    posterior,
    prior=prior_properties,
    truth=cap_truth,
    labels=[f"cap {index}" for index in range(caps.codomain.dim)],
)
axes[0, 0].figure.suptitle("Joint posterior of four cap averages")
print("corner plot drawn: marginals on the diagonal, pairwise contours below,")
print("  the prior dotted on its own axis, and the truth starred")

# ---------------------------------------------------------------------------
# One property on its own, prior against posterior.
# ---------------------------------------------------------------------------

figure, axis = plt.subplots(figsize=(7, 4))
plotting.plot_densities(
    posterior,
    prior=prior_properties,
    truth=float(cap_truth[0]),
    index=0,
    ax=axis,
    xlabel="cap 0 average anomaly",
)
axis.set_title("What the data bought, for one property")
print()
print("density plot drawn: the prior is on a second axis because it is far the")
print("  wider of the two, and sharing one would make the posterior a spike")

# ---------------------------------------------------------------------------
# A non-linear property, which has no covariance at all.
# ---------------------------------------------------------------------------

# The largest of the four cap anomalies, and the spread between them. Neither
# is a linear function of the model, so the posterior for them is not Gaussian
# however Gaussian the model posterior is.
#
# It is pushed forward from the *model* posterior rather than from the property
# one, because that is the measure that can be sampled: the model posterior
# carries a randomise-then-optimise sampler (DESIGN section 18.7), while the
# property posterior is a covariance with no factor and so cannot be drawn
# from. A push-forward can be sampled exactly when what it pushes can.
summary = EuclideanSpace(2)


def extremes(model):
    values = caps.codomain.to_components(caps(model))
    return summary.from_components(np.array([values.max(), np.ptp(values)]))


model_posterior = estimator(data)
nonlinear = model_posterior.push_forward(Operator.from_callables(X, summary, extremes))
print()
print(f"pushed through a non-linear map: {type(nonlinear).__name__},")
print(f"  Gaussian: no, covariance: none, can be sampled: {nonlinear.can_sample}")

plotting.plot_corner(
    nonlinear,
    truth=np.array([cap_truth.max(), np.ptp(cap_truth)]),
    labels=["largest anomaly", "spread"],
    samples=4000,
    rng=np.random.default_rng(7),
)
print("  drawn from four thousand draws instead, by kernel density —")
print("  and the marginals are visibly skewed, which is the thing a Gaussian")
print("  summary of this measure would have thrown away")

print()
print("three figures drawn; matplotlib.pyplot.show() displays them")
plt.show()
