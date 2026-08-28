"""
15. A worked problem, end to end.

A smooth field on a circle, observed at a handful of points with noise, and
recovered by Bayesian inversion. Everything here has appeared already; this is
what it looks like assembled.

Note what is absent: no Galerkin flag, no mass matrix written out, no
conversion between derivatives and gradients. Each of those is handled where it
belongs and nothing has to be remembered at the call site.
"""

import numpy as np

from pygeoinf2 import GaussianMeasure
from pygeoinf2.numerics import CGSolver
from pygeoinf2.symmetric_space import Sobolev

rng = np.random.default_rng(3)

# --- the model space and the prior ---------------------------------------
X = Sobolev((128,), 2.0, 0.15)
prior = X.sobolev_measure(2.0, 0.15, amplitude=1.0)
print(f"model space: {X.dim} dimensions, prior covariance {prior.covariance.traits}")

# --- the forward problem: evaluation at twelve points --------------------
points = [np.array([p]) for p in np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))]
A = X.point_evaluation_operator(points)
Y = A.codomain

noise_level = 0.05
noise = GaussianMeasure.from_standard_deviation(Y, noise_level)

# --- synthetic data ------------------------------------------------------
truth = prior.sample(rng=rng)
data = Y.add(A(truth), noise.sample(rng=rng))

# --- the normal operator, recognised rather than asserted ----------------
normal = A @ prior.covariance @ A.adjoint + noise.covariance
print(f"normal operator: {normal.traits}")
print("  so CG accepts it without anyone claiming anything.")

# --- solve, and map back through the prior covariance --------------------
weights = CGSolver(rtol=1e-12)(normal).solve(data)
posterior_mean = (prior.covariance @ A.adjoint)(weights.solution)
print(f"CG converged in {weights.iterations} iterations")
print()

# --- how did we do? ------------------------------------------------------
error = X.norm(X.subtract(posterior_mean, truth)) / X.norm(truth)
# Against the prior mean, which is what the inversion had to beat. Written as
# the comparison rather than as the 1.0 it currently equals, so it stays
# correct if the prior ever gets a non-zero mean.
prior_error = X.norm(X.subtract(prior.expectation, truth)) / X.norm(truth)
print(f"relative error of the posterior mean : {error:.3f}")
print(f"relative error of the prior mean     : {prior_error:.3f}")
print(
    f"data residual                        : "
    f"{Y.norm(Y.subtract(A(posterior_mean), data)):.4f}"
)
print(
    f"noise level                          : {noise_level * np.sqrt(len(points)):.4f}"
)
print()

# The posterior covariance, as an operator. Its traits come out right too.
kalman = prior.covariance @ A.adjoint @ CGSolver(rtol=1e-10)(normal)
posterior_covariance = prior.covariance - kalman @ A @ prior.covariance
spread = np.sqrt(
    max(
        X.inner_product(posterior_covariance(X.basis_vector(0)), X.basis_vector(0)), 0.0
    )
)
prior_spread = np.sqrt(
    X.inner_product(prior.covariance(X.basis_vector(0)), X.basis_vector(0))
)
print(f"prior spread in mode 0     : {prior_spread:.4f}")
print(f"posterior spread in mode 0 : {spread:.4f}  <- the data reduced it")
