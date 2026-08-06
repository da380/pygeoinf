"""Cross-field Bayesian updating with a correlated invariant prior.

Observes field u at scattered points and checks that the posterior mean of
the *unobserved* field v is informed through the cross-covariance, using
only existing library machinery on the joint measure.
"""

import numpy as np
from pygeoinf.symmetric_space import sphere
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInversion
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.hilbert_space import EuclideanSpace

np.random.seed(7)

X = sphere.Sobolev(32, 2.0, 0.1)

# Joint prior: same marginal spectrum, correlation decaying with scale
f = X.sobolev_kernel(2.0, 0.1)
rho = lambda lam: 0.9 * np.exp(-lam / 800.0)
prior = X.correlated_invariant_gaussian_measure(f, rho)
prior = prior.rescale_norm_variance(np.sqrt(2.0))  # unit norm variance per field

# Synthetic truth drawn from the prior
u_true, v_true = prior.sample()

# Observe u only, at scattered points, with noise
points = X.random_points(60)
A = X.point_evaluation_operator(points) @ prior.domain.subspace_projection(0)
noise = GaussianMeasure.from_standard_deviation(EuclideanSpace(len(points)), 0.05)
problem = LinearForwardProblem(A, data_error_measure=noise)
data = problem.synthetic_data([u_true, v_true])

posterior = LinearBayesianInversion(problem, prior).model_posterior_measure(
    data, CholeskySolver()
)
u_post, v_post = posterior.expectation


def correlation(a, b):
    ca, cb = X.to_components(a) * np.sqrt(X.metric_values), X.to_components(
        b
    ) * np.sqrt(X.metric_values)
    return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))


print(f"corr(u_post, u_true) = {correlation(u_post, u_true):.3f}   (observed field)")
print(
    f"corr(v_post, v_true) = {correlation(v_post, v_true):.3f}   (unobserved field, via cross-covariance)"
)

# Control: with an independent prior the posterior mean of v must vanish
prior0 = X.correlated_invariant_gaussian_measure(f, 0.0).rescale_norm_variance(
    np.sqrt(2.0)
)
posterior0 = LinearBayesianInversion(problem, prior0).model_posterior_measure(
    data, CholeskySolver()
)
_, v_post0 = posterior0.expectation
print(f"||v_post|| with rho = 0:  {X.norm(v_post0):.2e}   (should be ~ 0)")
print(f"||v_post|| with rho(lam): {X.norm(v_post):.2e}")

# Posterior sampling on the joint space also works out of the box
us, vs = posterior.sample()
print("posterior joint sample drawn:", type(us).__name__, type(vs).__name__)
