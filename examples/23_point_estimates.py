"""
23. Point estimates, and what each one is for.

The library offers point estimators because they are wanted, but its view is
that they are tools inside the methods that quantify uncertainty -- the
posterior measure and the feasible set -- and not answers of much interest in
themselves; it offers nothing like an "optimal regularization parameter". Each
estimator appears here once, as what it is.

The problem is example 22's kind, point values on a circle, parameterized by
the prior's covariance factor, ``m = L w``, so that a Tikhonov term ``|w|^2``
measures a model in the prior's own norm rather than in the model space's:
small then means likely.
"""

import numpy as np

from pygeoinf import (
    BackusGilbert,
    GaussianMeasure,
    LeastSquares,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
)
from pygeoinf.symmetric_space import circle

rng = np.random.default_rng(23)

X = circle.Sobolev(128, 2.0, 0.2)
prior = X.sobolev_measure(2.0, 0.3, pointwise_std=1.0)
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 24))
A = X.point_evaluation_operator(sensors)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.1)
problem = LinearForwardProblem(A, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

L = prior.covariance_factor
parameterized = problem.parameterized(L)
W = parameterized.model_space
print(
    f"{problem.data_space.dim} point values; the model is m = L w, L the prior's factor"
)
print()

# ---------------------------------------------------------------------------
# LeastSquares: the Tikhonov minimizer of |A L w - d|^2_R + t |w|^2 at a
# damping t you name. An affine map from data to w. At t = 1 it is the
# posterior mean, the same linear system solved for the same vector: the
# measure answer computes it on the way.
# ---------------------------------------------------------------------------

least_squares = LeastSquares(parameterized, damping=1.0)
posterior = LinearGaussianInversion(problem, prior)(data)
mean = posterior.expectation
from_least_squares = L(least_squares(data))
agreement = X.norm(X.subtract(from_least_squares, mean)) / X.norm(mean)
print(
    f"LeastSquares at damping 1, mapped by L, against the posterior mean: {agreement:.1e}"
)
print()

# ---------------------------------------------------------------------------
# MinimumNorm: the smallest w whose misfit meets the discrepancy principle at
# the confidence level, the damping found from the data by a one-dimensional
# search. It is the smallest model the data cannot reject, which is a question
# a feasible set asks; it is not an estimate of the truth.
# ---------------------------------------------------------------------------

minimum_norm = MinimumNorm(parameterized, level=0.95)
search = minimum_norm.discrepancy_search(data)
smallest = minimum_norm(data)
print(
    f"MinimumNorm: damping {search.argument:.2f} after {search.evaluations} probes"
    f" and {search.inner_iterations} Krylov iterations"
)
for name, w in (("smallest model", smallest), ("posterior mean", least_squares(data))):
    print(
        f"  {name:15s} chi-squared {problem.chi_squared(L(w), data):5.1f},"
        f" size in the prior's norm {W.norm(w):.2f}"
    )
print(f"  the critical value at 95%: {problem.critical_chi_squared(level=0.95):.1f}")
print()

# ---------------------------------------------------------------------------
# BackusGilbert: an estimate of a property, two arc averages, as X d with
# X = T C A* (A C A* + alpha R)^-1. Its error is (X A - T) m + X e: the
# resolution X A against T, the propagated covariance X R X*, and the bias
# (X A - T) m, which needs the model and so cannot be quoted.
# ---------------------------------------------------------------------------

T = X.geodesic_ball_average_operator([np.array([1.0]), np.array([4.0])], 0.3)
estimator = BackusGilbert(problem, T, covariance=prior.covariance)
estimate = estimator(data)
propagated = np.sqrt(np.diag(estimator.propagated_covariance().matrix()))
bias = estimator.bias(truth)
print("BackusGilbert, with C the prior covariance and alpha one:")
for i in range(2):
    print(
        f"  arc {i}: estimate {estimate[i]:+.3f}, propagated sd {propagated[i]:.3f},"
        f" bias on the truth {bias[i]:+.3f}, truth {T(truth)[i]:+.3f}"
    )

# Completed by the prior measure, the budget is the posterior: the estimate is
# the posterior mean of T m, and the bias's covariance under the prior plus
# the propagated covariance is the posterior covariance of T m. Completed by a
# norm bound instead, the same X is a linear certificate (tutorial 8).
unresolved = T - estimator.resolution()
budget = (unresolved @ prior.covariance @ unresolved.adjoint).matrix()
budget += estimator.propagated_covariance().matrix()
pushed = posterior.push_forward(T)
print(
    "  against the posterior of T m: mean to "
    f"{np.max(np.abs(estimate - pushed.expectation)):.0e}, covariance to "
    f"{np.max(np.abs(budget - pushed.covariance.matrix())) / np.max(np.abs(budget)):.0e}"
)
print()
print("ConstrainedLeastSquares and ConstrainedMinimumNorm are the same two")
print("estimators on an affine subspace, for what is known exactly (example 17).")
