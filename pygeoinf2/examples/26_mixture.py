"""
26. A prior that cannot make up its mind.

Every prior so far has been a single Gaussian, which can say "about this, give
or take that much" and nothing else. It cannot say *either-or*. But an
either-or is the honest state of a great deal of prior knowledge: either the
structure is smooth or it is rough, either the anomaly is in the crust or in
the mantle, either this fault slipped or the other one did.

A Gaussian mixture says it. Couple a parameterised Gaussian with a distribution
over the parameter — here a discrete choice between two scenarios — and the
data do the rest: under a linear Gaussian likelihood the posterior is again a
mixture, with the same components updated in the usual way and the weights
rescored by each component's *evidence*. Exact, not approximate.

The weights are the interesting output. They are a quantitative answer to "which
scenario?", and they come out of the same evidence calculation that model
comparison uses.
"""

import numpy as np

from pygeoinf2 import plotting
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference import (
    LinearForwardProblem,
    LinearGaussianMixtureInversion,
)
from pygeoinf2.numerics.solvers import CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.probability.mixture import GaussianMixture
from pygeoinf2.traits import Traits

rng = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# Two scenarios, and a prior that admits both.
# ---------------------------------------------------------------------------

X = EuclideanSpace(2)
chol = CholeskySolver()


def scenario(centre, spread):
    """A Gaussian around one scenario's expected model."""
    covariance = LinearOperator.from_derivative_matrix(
        X,
        X,
        spread * np.identity(2),
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
    )
    return GaussianMeasure(
        X,
        covariance=covariance,
        precision=chol(covariance),
        covariance_factor=LinearOperator.from_derivative_matrix(
            X, X, np.sqrt(spread) * np.identity(2)
        ),
        expectation=X.from_components(np.asarray(centre, float)),
    )


# Two scenarios that predict *nearly the same datum* by different means — one
# splits the signal evenly, the other loads it onto the first parameter. The
# observation cannot separate them cleanly, which is the case worth showing:
# a decisive posterior would be a mixture doing nothing a single Gaussian
# could not.
prior = GaussianMixture.from_family(
    lambda spec: scenario(*spec),
    [((0.6, 0.6), 0.05), ((1.9, -0.5), 0.30)],
    weights=[0.5, 0.5],
)
print(f"prior: {len(prior)} scenarios, weights {np.round(prior.weights, 3)}")
print(f"  prior mean {np.round(X.to_components(prior.expectation), 3)},")
print("  which is a point neither scenario considers likely -- a mixture's mean")
print("  is not where its mass is, and that is the whole difficulty a single")
print("  Gaussian has with this kind of knowledge")
print()

# ---------------------------------------------------------------------------
# One observation, and the likelihood is deliberately weak.
# ---------------------------------------------------------------------------

# Sensitive to the sum of the two parameters only: it cannot separate them, so
# what discriminates the scenarios is the prior geometry, not the data alone.
forward = LinearOperator.from_derivative_matrix(
    X, EuclideanSpace(1), np.array([[1.0, 1.0]])
)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.25)
problem = LinearForwardProblem(forward, error=noise)

truth = X.from_components(np.array([1.9, -0.5]))  # the second scenario
data = problem.synthetic_data(truth, rng=rng)
print(f"truth {np.round(X.to_components(truth), 3)} (scenario 2)")
print(f"one datum: {float(forward.codomain.to_components(data)[0]):+.4f}")
print()

# ---------------------------------------------------------------------------
# The posterior, which is a mixture again.
# ---------------------------------------------------------------------------

inversion = LinearGaussianMixtureInversion(problem, prior, solver=chol)
posterior = inversion(data)

terms = inversion.log_evidence_terms(data)
print("scenario   log evidence   prior weight   posterior weight")
for index, (term, before, after) in enumerate(
    zip(terms, prior.weights, posterior.weights)
):
    print(f"    {index}       {term:+8.3f}         {before:.3f}            {after:.3f}")
print()
print(f"mixture log evidence {inversion.log_evidence(data):+.3f}, which is below the")
print(f"  better component's {terms.max():+.3f}: hedging costs something, and this is")
print("  where a model comparison sees the cost")
print()

for index, component in enumerate(posterior.components):
    print(
        f"  scenario {index} posterior mean "
        f"{np.round(X.to_components(component.expectation), 3)}"
    )
print(f"  mixture mean       {np.round(X.to_components(posterior.expectation), 3)}")
print(f"  truth              {np.round(X.to_components(truth), 3)}")
print()

# ---------------------------------------------------------------------------
# And it looks like what it is.
# ---------------------------------------------------------------------------

# A mixture can be sampled exactly -- choose a component by weight, draw from
# it -- so the corner plot of example 25 draws it from samples and shows the
# shape a Gaussian summary would have destroyed.
plotting.plot_corner(
    posterior,
    truth=X.to_components(truth),
    labels=["parameter 1", "parameter 2"],
    samples=20000,
    rng=np.random.default_rng(5),
)
print("corner plot drawn from samples: the posterior is genuinely bimodal, with")
print("  two separated lobes rather than the single blurred one a Gaussian fit")
print("  would report -- and the mixture mean above sits between them, at a")
print("  point neither lobe considers likely, which is why the mean of a")
print("  multimodal posterior is the wrong thing to quote")
print()
print("note which scenario the data prefer. The truth is scenario 1, and yet")
print("  scenario 0 carries the larger weight: it is the *tighter* prior, so it")
print("  earns more evidence for the fit it achieves. That is Occam arithmetic")
print("  working correctly, not a failure -- and it is why the answer keeps both")
print("  scenarios rather than choosing. A single Gaussian could only have")
print("  collapsed to the middle, which is where the truth is not.")
print()
print("one figure drawn; matplotlib.pyplot.show() displays it")
