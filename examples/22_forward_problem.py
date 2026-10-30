"""
22. A worked problem, end to end: the forward problem and its posterior.

A smooth field on a circle, observed at a handful of points with noise, and
recovered by Bayesian inversion. Everything here has appeared already; this is
what it looks like assembled, through the two objects every later example
uses. The *forward problem* is a forward operator and an error measure, and
says nothing about the unknown. The *estimator* adds what is assumed about the
unknown -- here a prior measure -- and maps data to an answer.

Note what is absent: no Galerkin flag, no mass matrix written out, no
conversion between derivatives and gradients. Each of those is handled where it
belongs and nothing has to be remembered at the call site.
"""

import numpy as np

from pygeoinf import (
    CGSolver,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
)
from pygeoinf.symmetric_space import circle

rng = np.random.default_rng(3)

# --- the model space, the forward operator, the noise ---------------------
X = circle.Sobolev(128, 2.0, 0.15)
points = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
A = X.point_evaluation_operator(points)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.05)
problem = LinearForwardProblem(A, error=noise)
print(problem)

# --- a prior, and a truth drawn from it with its data ---------------------
prior = X.sobolev_measure(2.0, 0.15, pointwise_std=1.0)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

# The problem alone answers one question: is a model consistent with the data?
# Its weighted misfit should be about the number of data, and below the 95%
# point of the chi-squared distribution.
misfit, critical = problem.chi_squared(truth, data), problem.critical_chi_squared()
print(f"chi-squared of the truth {misfit:.1f}, critical value {critical:.1f}")
print("the truth passes:", problem.chi_squared_test(truth, data))
print("the zero model passes:", problem.chi_squared_test(X.zero(), data))
print()

# --- the posterior ----------------------------------------------------------
inversion = LinearGaussianInversion(problem, prior)
posterior = inversion(data)
print(
    f"a {type(posterior).__name__}, its normal equations in the {inversion.formalism}"
)
print("  the normal operator, recognized rather than asserted:")
print("  ", inversion.normal_operator.traits)

mean = posterior.expectation
error = X.norm(X.subtract(mean, truth)) / X.norm(truth)
print(f"relative error of the posterior mean {error:.3f}, of the prior mean 1.000")
print()

# What the estimator does inside, by hand: m = Q A* (A Q A* + R)^-1 d, with the
# normal operator applied and never assembled, and conjugate gradients
# admissible on it because its traits come from its form.
normal = A @ prior.covariance @ A.adjoint + noise.covariance
weights = CGSolver(rtol=1e-12)(normal).solve(data).solution
by_hand = (prior.covariance @ A.adjoint)(weights)
agreement = X.norm(X.subtract(by_hand, mean)) / X.norm(mean)
print(f"by hand, the same mean to {agreement:.0e}")
print()

# --- and how sure ------------------------------------------------------------
# The posterior standard deviation at a datum and between data, against the
# prior's one: one solve per point.
widest = int(np.argmax(np.diff(points)))
gap = 0.5 * (points[widest] + points[widest + 1])
at_datum, in_gap = np.sqrt(X.pointwise_variance_at(posterior, [points[0], gap]))
print("posterior standard deviation, against the prior's 1.000 everywhere:")
print(f"  {at_datum:.3f} at a datum, {in_gap:.3f} in the middle of the widest gap")
