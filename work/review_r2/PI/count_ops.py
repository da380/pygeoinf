"""Count A, A*, Q, L and solve applications through the Gaussian inversion."""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.probability.mixture import GaussianMixture
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.inference.mixture import LinearGaussianMixtureInversion
from pygeoinf2.numerics.solvers import CGSolver, CholeskySolver, LinearSolver
from pygeoinf2.traits import Traits

counts = Counter()

def counted(op, name):
    def value(x):
        counts[name] += 1
        return op(x)
    def adjoint(y):
        counts[name + "*"] += 1
        return op.adjoint(y)
    return LinearOperator.from_callables(op.domain, op.codomain, value, adjoint=adjoint, traits=op.traits)

class CountingSolver(LinearSolver):
    def __init__(self, inner):
        self.inner = inner
    def _invert(self, operator):
        return counted(self.inner(operator), "solve")

def snapshot(label):
    global counts
    print(f"{label:52s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    counts = Counter()

rng = np.random.default_rng(3)
X = make_dense_metric_space(dim=12)
D = EuclideanSpace(6)
A = counted(LinearOperator.from_matrix(X, D, rng.standard_normal((6, 12)), form="components"), "A")
root = rng.standard_normal((12, 12)); gal = root @ root.T + 12 * np.eye(12)
base = GaussianMeasure.from_covariance_matrix(X, gal, form="galerkin")
m0 = rng.standard_normal(12)
prior = GaussianMeasure(X, expectation=m0, covariance=counted(base.covariance, "Q"), covariance_factor=counted(base.covariance_factor, "L"))
noise = GaussianMeasure.from_standard_deviation(D, 0.3, expectation=0.1 * np.ones(6))
problem = LinearForwardProblem(A, error=noise)
data = rng.standard_normal(6)
T = LinearOperator.from_matrix(X, EuclideanSpace(2), rng.standard_normal((2, 12)), form="components")

for solver_name, make in (("Cholesky", lambda: CountingSolver(CholeskySolver())), ("CG", lambda: CountingSolver(CGSolver()))):
    print(f"\n=== {solver_name}, data-space formalism, nonzero prior and noise means ===")
    counts.clear()
    est = LinearGaussianInversion(problem, prior, solver=make())
    snapshot("construction")
    post = est(data)
    snapshot("est(data)")
    post2 = est(data)
    snapshot("est(data) again")
    prop = est.push_forward(T)(data)
    snapshot("est.push_forward(T)(data)")
    x = post.sample(rng=rng)
    snapshot("post.sample()")
    v = post.covariance(X.random(rng=rng) if hasattr(X, 'random') else rng.standard_normal(12))
    snapshot("post.covariance(x)")
    M = post.covariance.matrix(form="components")
    snapshot("post.covariance.matrix()  (dim 12)")
    est.log_evidence(data)
    snapshot("log_evidence(data)  [first]")
    est.log_evidence(data)
    snapshot("log_evidence(data)  [second]")
    est.log_evidence(data, method="stochastic", samples=4)
    snapshot("log_evidence stochastic samples=4")

print("\n=== zero-mean prior and noise, Cholesky ===")
prior0 = GaussianMeasure(X, covariance=counted(base.covariance, "Q"), covariance_factor=counted(base.covariance_factor, "L"))
noise0 = GaussianMeasure.from_standard_deviation(D, 0.3)
problem0 = LinearForwardProblem(A, error=noise0)
counts.clear()
est = LinearGaussianInversion(problem0, prior0, solver=CountingSolver(CholeskySolver()))
snapshot("construction")
est(data); snapshot("est(data)")
est.log_evidence(data); snapshot("log_evidence(data)")

print("\n=== mixture of 2 components, Cholesky ===")
mix = GaussianMixture([prior, GaussianMeasure(X, expectation=-m0, covariance=counted((2.0*base.covariance), "Q2"), covariance_factor=counted(np.sqrt(2.0)*base.covariance_factor, "L2"))])
counts.clear()
minv = LinearGaussianMixtureInversion(problem, mix, solver=CountingSolver(CholeskySolver()))
snapshot("construction")
minv(data); snapshot("mixture(data)")
minv.push_forward(T)(data); snapshot("mixture.push_forward(T)(data)")

print("\n=== condition(): covariance application and sampling ===")
counts.clear()
cond = prior.condition(A, data, noise=noise, solver=CountingSolver(CholeskySolver()))
snapshot("prior.condition(A, d, noise)")
cond.covariance(rng.standard_normal(12)); snapshot("cond.covariance(x)")
cond.sample(rng=rng); snapshot("cond.sample()")
cond.covariance.matrix(form="components"); snapshot("cond.covariance.matrix() dim 12")
