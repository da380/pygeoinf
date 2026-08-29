"""Transforms per CG iteration in example 21's normal operator, and a fused component-space version."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
import pyshtools
from pygeoinf2.inference import LinearGaussianInversion, LinearForwardProblem
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.numerics.solvers import CGSolver
from pygeoinf2.traits import Traits

calls = {"n": 0}
import pyshtools.expand as ex
for name in ("SHExpandDH", "MakeGridDH", "SHExpandGLQ", "MakeGridGLQ"):
    if hasattr(ex, name):
        orig = getattr(ex, name)
        def wrap(f):
            def g(*a, **k):
                calls["n"] += 1
                return f(*a, **k)
            return g
        setattr(ex, name, wrap(orig))
import pygeoinf2.symmetric_space.sphere as sph

rng = np.random.default_rng(1)
X = Sobolev(48, 2.0, 0.1)
receivers = X.stations(count=24, rng=rng)
sources = X.earthquakes(count=40, minimum_magnitude=5.5, rng=rng)
paths = [(s, r) for s in sources for r in receivers]
forward = X.path_average_operator(paths, count=16, dense=True)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
problem = LinearForwardProblem(forward, error=noise)
prior = X.heat_measure(0.17, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
est = LinearGaussianInversion(problem, prior)
N = est.normal_operator
v = rng.standard_normal(forward.codomain.dim)

calls["n"] = 0
t = time.perf_counter(); N(v); t_apply = time.perf_counter() - t
print(f"one N application: {calls['n']} SH transforms, {t_apply*1e3:.2f} ms")

# fused: A_c (lam * g^-1 * (A_c^T v)) + sigma^2 v
Ac = forward.matrix(form="components")
lam = prior.covariance.eigenvalues
g = X.metric_values if hasattr(X, "metric_values") else None
print("A_c shape", Ac.shape, "| diag metric:", g is not None)
def fused(v):
    return Ac @ (lam * (v @ Ac / g)) + 0.02**2 * v
err = np.abs(fused(v) - N(v)).max() / np.abs(N(v)).max()
print(f"fused agrees with N to {err:.1e}")

# interleaved timing of 20 applications each, 5 rounds
rounds = []
for _ in range(5):
    t = time.perf_counter()
    for _ in range(20): N(v)
    a = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(20): fused(v)
    b = time.perf_counter() - t
    rounds.append((a, b))
print("N vs fused per application (ms):", [(f"{a/20*1e3:.2f}", f"{b/20*1e3:.2f}") for a, b in rounds])

# full solve timing: as built vs fused operator
fusedN = LinearOperator.from_callables(N.domain, N.codomain, fused, adjoint=fused, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
sol = CGSolver()
res = []
for _ in range(3):
    calls["n"] = 0
    t = time.perf_counter(); r1 = sol(N).solve(data); a = time.perf_counter() - t; n1 = calls["n"]
    t = time.perf_counter(); r2 = sol(fusedN).solve(data); b = time.perf_counter() - t
    res.append((a, b, r1.iterations, r2.iterations, n1))
print("solve as built vs fused (s, iterations, transforms):", [(f"{a:.3f}", f"{b:.3f}", i1, i2, n1) for a, b, i1, i2, n1 in res])
print("solutions agree to", np.abs(r1.solution - r2.solution).max() / np.abs(r1.solution).max())

# where the time goes in one application: transforms alone
c = X.to_components(X.random(rng=rng)) if hasattr(X, "random") else None
x = X.from_components(rng.standard_normal(X.dim))
t = time.perf_counter()
for _ in range(50): X.from_components(X.to_components(x))
print(f"one to_components+from_components round trip: {(time.perf_counter()-t)/50*1e3:.2f} ms")
