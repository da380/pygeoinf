import sys, time
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.inference import LinearForwardProblem, DiscrepancyPrinciple
from pygeoinf2.numerics.solvers import CholeskySolver, LinearSolver, InverseOperator
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.traits import Traits

counts = Counter()
def counted(op, name):
    def value(x): counts[name] += 1; return op(x)
    def adjoint(y): counts[name + "*"] += 1; return op.adjoint(y)
    return LinearOperator.from_callables(op.domain, op.codomain, value, adjoint=adjoint, traits=op.traits)
class CountingSolver(LinearSolver):
    def __init__(self, inner): self.inner = inner
    def _invert(self, operator):
        inv = self.inner(operator)
        def solve_fn(y, x0):
            counts["solve"] += 1; return inv.solve(y, x0=x0)
        def adj_fn(y, x0):
            counts["solve*"] += 1; return InverseOperator.solve(inv, y, x0=x0)
        return InverseOperator(operator, self, solve_fn, traits=inv.traits, adjoint_solve_fn=adj_fn)
def snap(label):
    print(f"{label:60s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))); counts.clear()

rng = np.random.default_rng(5)
X = make_dense_metric_space(dim=12); D = EuclideanSpace(6)
A = counted(LinearOperator.from_matrix(X, D, rng.standard_normal((6, 12)), form="components"), "A")
noise = GaussianMeasure.from_standard_deviation(D, 0.3)
problem = LinearForwardProblem(A, error=noise)
data = rng.standard_normal(6) * 3
dp = DiscrepancyPrinciple(problem, solver=CountingSolver(CholeskySolver()))
counts.clear(); m = dp(data); snap("dp(data)")
lin = dp.at(data); snap("dp.at(data)")
lin.derivative(rng.standard_normal(6)); snap("derivative(v)")
lin.derivative.adjoint(rng.standard_normal(12)); snap("derivative.adjoint(u)")

# dense KL: assemblies of the reference covariance without a precision
root = rng.standard_normal((12, 12)); gal = root @ root.T + 12 * np.eye(12); gal1 = root.T @ root + 6 * np.eye(12)
mu0 = GaussianMeasure.from_covariance_matrix(X, gal, form="galerkin", expectation=rng.standard_normal(12))
mu1raw = GaussianMeasure.from_covariance_matrix(X, gal1, form="galerkin", expectation=rng.standard_normal(12))
mu1 = GaussianMeasure(X, expectation=mu1raw.expectation, covariance=counted(mu1raw.covariance, "C1"))
mu0c = GaussianMeasure(X, expectation=mu0.expectation, covariance=counted(mu0.covariance, "C0"))
counts.clear(); mu0c.kl_divergence(mu1, method="dense"); snap("kl_divergence dense, dim 12, no precisions")
counts.clear(); mu0c.as_multivariate_normal(); snap("as_multivariate_normal dim 12")
counts.clear(); mu0c.credible_set(level=0.9); snap("credible_set (no precision) dim 12")

# stochastic KL, opt-in: KL(mu||mu) with a non-diagonal wrapper forcing the route
S = Sobolev(16, 2.0, 0.3)
mu = S.heat_measure(0.3)
wrapped = GaussianMeasure(S, covariance=LinearOperator.from_callables(S, S, mu.covariance, adjoint=mu.covariance, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE))
t = time.perf_counter()
est = mu.kl_divergence_estimate(wrapped, method="stochastic", samples=50, rng=rng)
print(f"stochastic KL(mu||mu) on Sobolev(16) dim {S.dim}, 50 probes: {est.value:.3f} +/- {est.standard_error:.3f} (true 0) in {time.perf_counter()-t:.1f} s")
t = time.perf_counter()
est = wrapped.kl_divergence_estimate(mu, method="stochastic", samples=50, rng=rng)
print(f"stochastic KL with the reference's precision available: {est.value:.3f} +/- {est.standard_error:.3f} (true 0) in {time.perf_counter()-t:.1f} s")
