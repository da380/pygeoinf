import sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion, DiscrepancyPrinciple, ConstrainedLeastSquares
from pygeoinf2.geometry.subspaces import AffineSubspace
from pygeoinf2.numerics.solvers import CholeskySolver, CGSolver, LinearSolver
import pygeoinf2

counts = Counter()
def counted(op, name):
    def value(x): counts[name] += 1; return op(x)
    def adjoint(y): counts[name + "*"] += 1; return op.adjoint(y)
    return LinearOperator.from_callables(op.domain, op.codomain, value, adjoint=adjoint, traits=op.traits)
class CountingSolver(LinearSolver):
    def __init__(self, inner): self.inner = inner
    def _invert(self, operator): return counted(self.inner(operator), "solve")
def snap(label):
    print(f"{label:60s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))); counts.clear()

rng = np.random.default_rng(5)
X = make_dense_metric_space(dim=12); D = EuclideanSpace(6)
A = counted(LinearOperator.from_matrix(X, D, rng.standard_normal((6, 12)), form="components"), "A")
root = rng.standard_normal((12, 12)); gal = root @ root.T + 12 * np.eye(12)
base = GaussianMeasure.from_covariance_matrix(X, gal, form="galerkin", expectation=rng.standard_normal(12))
prior = GaussianMeasure(X, expectation=base.expectation, covariance=counted(base.covariance, "Q"), covariance_factor=base.covariance_factor)
noise = GaussianMeasure.from_standard_deviation(D, 0.3)
problem = LinearForwardProblem(A, error=noise)
data = rng.standard_normal(6)
T = LinearOperator.from_matrix(X, EuclideanSpace(2), rng.standard_normal((2, 12)), form="components")

print("top-level export LinearGaussianInversion:", hasattr(pygeoinf2, "LinearGaussianInversion"))
est = LinearGaussianInversion(problem, prior, solver=CountingSolver(CholeskySolver()))
post = est(data); counts.clear()
via_estimator = est.push_forward(T)(data); snap("estimator.push_forward(T)(data)")
via_measure = post.push_forward(T); snap("posterior.push_forward(T)")
print("  means agree:", np.abs(via_estimator.expectation - via_measure.expectation).max(),
      "| covariances agree:", np.abs(via_estimator.covariance.matrix(form='components') - via_measure.covariance.matrix(form='components')).max(),
      "| measure route samplable:", via_measure.can_sample)
counts.clear()

# dead kwargs on ConstrainedLeastSquares.parameterised
sub = AffineSubspace.from_linear_equation(LinearOperator.from_matrix(X, EuclideanSpace(1), rng.standard_normal((1, 12)), form="components"), np.array([1.0]))
cls = ConstrainedLeastSquares(problem, sub, damping=0.1, solver=CholeskySolver())
P = LinearOperator.from_matrix(EuclideanSpace(4), X, rng.standard_normal((12, 4)), form="components")
try:
    cls.parameterised(P, dense=True); print("ConstrainedLeastSquares.parameterised(**kwargs): accepted")
except TypeError as e:
    print("ConstrainedLeastSquares.parameterised(P, dense=True) -> TypeError:", str(e)[:70])

# DiscrepancyPrinciple: solves for __call__, derivative, at
counts.clear()
dp = DiscrepancyPrinciple(problem, solver=CountingSolver(CholeskySolver()))
snap("DiscrepancyPrinciple construction")
m = dp(data); snap("dp(data)")
lin = dp.at(data); snap("dp.at(data)")
d = lin.derivative(rng.standard_normal(6)); snap("derivative(v)")
d = lin.derivative.adjoint(rng.standard_normal(12)); snap("derivative.adjoint(u)")

# dense KL: how many times is the reference covariance assembled when it has no precision?
mu0 = GaussianMeasure.from_covariance_matrix(X, gal, form="galerkin", expectation=rng.standard_normal(12))
gal1 = root.T @ root + 6 * np.eye(12)
mu1raw = GaussianMeasure.from_covariance_matrix(X, gal1, form="galerkin", expectation=rng.standard_normal(12))
mu1 = GaussianMeasure(X, expectation=mu1raw.expectation, covariance=counted(mu1raw.covariance, "C1"))
mu0c = GaussianMeasure(X, expectation=mu0.expectation, covariance=counted(mu0.covariance, "C0"))
counts.clear(); mu0c.kl_divergence(mu1, method="dense"); snap("kl_divergence dense, dim 12, no precisions")
counts.clear(); mu0c.as_multivariate_normal(); snap("as_multivariate_normal dim 12")
counts.clear(); mu0c.credible_set(level=0.9); snap("credible_set (no precision) dim 12")
