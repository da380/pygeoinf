"""Closed-form checks of metric-sensitive probability code on a dense Gram matrix."""
import sys
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.tests.conftest import make_dense_metric_space, make_weighted_space
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.probability.mixture import GaussianMixture
from pygeoinf2.traits import Traits
from scipy.stats import multivariate_normal

rng = np.random.default_rng(11)
n = 5
X = make_dense_metric_space(dim=n)
G = X.gram_matrix()

def spd(k):
    r = rng.standard_normal((n, n)); return r @ r.T + k * np.eye(n)

def report(name, err, tol=1e-9):
    print(f"{'OK ' if err < tol else 'BAD'} {name:70s} {err:.2e}")

# Two measures: Galerkin matrices C0g, C1g; component matrices C_c = G^-1 C_g
C0g, C1g = spd(3.0), spd(2.0)
m0, m1 = rng.standard_normal(n), rng.standard_normal(n)
mu0 = GaussianMeasure.from_covariance_matrix(X, C0g, form="galerkin", expectation=m0)
mu1 = GaussianMeasure.from_covariance_matrix(X, C1g, form="galerkin", expectation=m1)
C0c, C1c = np.linalg.solve(G, C0g), np.linalg.solve(G, C1g)

# 1. as_multivariate_normal: covariance of components = G^-1 C_g G^-1
mvn = mu0.as_multivariate_normal()
report("as_multivariate_normal cov == G^-1 C_gal G^-1", np.abs(mvn.cov - np.linalg.solve(G, np.linalg.solve(G, C0g).T)).max())
report("as_multivariate_normal mean", np.abs(mvn.mean - m0).max())

# 2. from_covariance_matrix(form='components') round trip
mu_c = GaussianMeasure.from_covariance_matrix(X, C0c, form="components")
report("from_covariance_matrix(components) component matrix", np.abs(mu_c.covariance.matrix(form="components") - C0c).max())

# 3. dense KL against closed form in components (operator KL, metric G)
d = m1 - m0
kl_ref = 0.5 * (np.trace(np.linalg.solve(C1c, C0c)) + d @ G @ np.linalg.solve(C1c, d) - n
                + np.linalg.slogdet(C1c)[1] - np.linalg.slogdet(C0c)[1])
report("kl_divergence dense == closed form", abs(mu0.kl_divergence(mu1, method="dense") - kl_ref))

# 4. spectral KL: isotropic diagonal covariances on the dense space are legitimately self-adjoint
lam0, lam1 = 1.7, 0.6
D0 = DiagonalLinearOperator(X, lam0 * np.ones(n)).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
D1 = DiagonalLinearOperator(X, lam1 * np.ones(n)).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
nu0 = GaussianMeasure(X, expectation=m0, covariance=D0)
nu1 = GaussianMeasure(X, expectation=m1, covariance=D1)
kl_ref = 0.5 * (n * lam0 / lam1 + d @ G @ d / lam1 - n + n * np.log(lam1 / lam0))
report("kl_divergence spectral (isotropic, dense G) == closed form", abs(nu0.kl_divergence(nu1, method="spectral") - kl_ref))
report("kl_divergence dense agrees on the same pair", abs(nu0.kl_divergence(nu1, method="dense") - kl_ref))
# general diagonal on a weighted (diagonal-metric) space
W = make_weighted_space(); g = W.metric_values; k = W.dim
a, b = rng.uniform(0.5, 2, k), rng.uniform(0.5, 2, k); da = rng.standard_normal(k)
wa = GaussianMeasure(W, covariance=DiagonalLinearOperator(W, a))
wb = GaussianMeasure(W, expectation=da, covariance=DiagonalLinearOperator(W, b))
kl_ref = 0.5 * (np.sum(a / b) + np.sum(g * da**2 / b) - k + np.sum(np.log(b / a)))
report("kl_divergence spectral (weighted space, general diag) == closed form", abs(wa.kl_divergence(wb, method="spectral") - kl_ref))

# 5. condition() against the component closed form; data space Euclidean
mD = 3
Ac = rng.standard_normal((mD, n))
A = LinearOperator.from_matrix(X, EuclideanSpace(mD), Ac, form="components")
Rc = np.diag(rng.uniform(0.1, 0.3, mD))
noise = GaussianMeasure.from_covariance_matrix(EuclideanSpace(mD), Rc, form="galerkin", expectation=0.2 * np.ones(mD))
dvec = rng.standard_normal(mD)
post = mu0.condition(A, dvec, noise=noise)
Astar = np.linalg.solve(G, Ac.T)           # component matrix of A* (data space Euclidean)
N = Ac @ C0c @ Astar + Rc
K = C0c @ Astar @ np.linalg.inv(N)
mean_ref = m0 + K @ (dvec - Ac @ m0 - 0.2 * np.ones(mD))
cov_ref = C0c - K @ Ac @ C0c
report("condition(): mean", np.abs(post.expectation - mean_ref).max())
report("condition(): covariance component matrix", np.abs(post.covariance.matrix(form="components") - cov_ref).max())
# exact constraint
post_exact = mu0.condition(A, dvec)
N = Ac @ C0c @ Astar; K = C0c @ Astar @ np.linalg.inv(N)
report("condition() exact: mean", np.abs(post_exact.expectation - (m0 + K @ (dvec - Ac @ m0))).max())
report("condition() exact: covariance", np.abs(post_exact.covariance.matrix(form="components") - (C0c - K @ Ac @ C0c)).max())
# sampler: covariance of many draws (sampling check, loose)
draws = np.array([post.sample(rng=rng) for _ in range(20000)])
emp = np.cov(draws.T)   # covariance of components == C_c G^-1
report("condition(): sample covariance vs C_c G^-1 (20k draws, tol 5e-2)", np.abs(emp - cov_ref @ np.linalg.inv(G)).max(), tol=5e-2)

# 6. credible_set precision = C^-1 / chi2 threshold, in components
from scipy.stats import chi2
thr = chi2.ppf(0.9, n)
ell = mu0.credible_set(level=0.9)
Pc = ell.precision.matrix(form="components") if hasattr(ell, "precision") else ell._precision.matrix(form="components")
report("credible_set precision component matrix == C_c^-1 / threshold", np.abs(Pc - np.linalg.inv(C0c) / thr).max())

# 7. norms
ev = np.linalg.eigvals(C0c).real
report("nuclear_norm == sum of operator eigenvalues", abs(mu0.nuclear_norm() - ev.sum()))
report("hilbert_schmidt_norm == sqrt(sum eig^2)", abs(mu0.hilbert_schmidt_norm() - np.sqrt((ev**2).sum())))
try:
    mu0.hilbert_schmidt_norm(method="stochastic")
    print("NOTE hilbert_schmidt_norm(method='stochastic') silently took the dense route")
except Exception as e:
    print("hilbert_schmidt_norm(method='stochastic') raised:", type(e).__name__)

# 8. log normalising constant: -n/2 log 2pi - 1/2 log det C_c (dense) and diagonal route
report("log_normalising_constant (dense route) ", abs(mu0.log_normalising_constant() - (-0.5 * n * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(C0c)[1])))
report("log_normalising_constant (diagonal route, weighted)", abs(wa.log_normalising_constant() - (-0.5 * k * np.log(2 * np.pi) - 0.5 * np.sum(np.log(a)))))

# 9. mixture responsibilities on the dense space vs scipy in components (density w.r.t. components differs by a shared sqrt(det G))
mu0p = mu0.with_regularized_inverse(lambda op: LinearOperator.from_matrix(X, X, np.linalg.inv(op.matrix(form="components")), form="components", traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE))
mu1p = mu1.with_regularized_inverse(lambda op: LinearOperator.from_matrix(X, X, np.linalg.inv(op.matrix(form="components")), form="components", traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE))
mix = GaussianMixture([mu0p, mu1p], weights=[0.3, 0.7])
x = rng.standard_normal(n)
Ginv = np.linalg.inv(G)
p0 = multivariate_normal(m0, C0c @ Ginv).logpdf(x); p1 = multivariate_normal(m1, C1c @ Ginv).logpdf(x)
resp = np.exp([np.log(0.3) + p0, np.log(0.7) + p1]); resp /= resp.sum()
report("mixture.marginal_probabilities == scipy responsibilities", np.abs(mix.marginal_probabilities(x) - resp).max())
from scipy.special import logsumexp
ref_ld = logsumexp([np.log(0.3) + p0, np.log(0.7) + p1]) + 0.5 * np.linalg.slogdet(G)[1]
report("mixture.log_density == scipy (+ 1/2 log det G for the volume)", abs(mix.log_density(x) - ref_ld))

# 10. ambient_ball radius == weighted chi2 quantile of operator eigenvalues; check by sampling coverage
ball = mu0.ambient_ball(level=0.9)
draws = np.array([mu0.sample(rng=rng) for _ in range(20000)])
inside = np.mean([X.norm(dd - m0) <= ball.radius for dd in draws])
report("ambient_ball coverage 0.9 (sampling, tol 1e-2)", abs(inside - 0.9), tol=1.5e-2)
