"""Backus support sweep: applications of A/A* per direction on each route, and timings."""
import sys, time
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import LinearForwardProblem
from pygeoinf2.inference.backus import DualFeasibleProperty, FeasibleProperty, BackusGilbert
counts = Counter()
def counted(op, name):
    def value(x): counts[name] += 1; return op(x)
    def adjoint(y): counts[name + "*"] += 1; return op.adjoint(y)
    return LinearOperator.from_callables(op.domain, op.codomain, value, adjoint=adjoint, traits=op.traits)
rng = np.random.default_rng(2)
nM, nD, nP = 400, 40, 3
X, D, Pp = EuclideanSpace(nM), EuclideanSpace(nD), EuclideanSpace(nP)
A = counted(LinearOperator.from_matrix(X, D, rng.standard_normal((nD, nM)) / np.sqrt(nM), form="components"), "A")
T = LinearOperator.from_matrix(X, Pp, rng.standard_normal((nP, nM)) / np.sqrt(nM), form="components")
truth = rng.standard_normal(nM) * 0.8 / np.sqrt(nM) * np.sqrt(nM) * 0.05
truth = truth / np.linalg.norm(truth) * 0.8
data = A(truth) + 0.02 * rng.standard_normal(nD)
noise = Ball(D, radius=0.05 * np.sqrt(nD) * 0.5)
problem = LinearForwardProblem(A, error=noise)
prior = Ball(X, radius=1.0)
dirs = [rng.standard_normal(nP) for _ in range(16)]
dfp = DualFeasibleProperty(problem, T, prior)
for route in ("kkt", "smoothed", "primal", "dual"):
    counts.clear(); t = time.perf_counter()
    vals = dfp.support_values(dirs, data, route=route)
    dt = time.perf_counter() - t
    print(f"{route:9s} 16 directions: {dt:7.3f} s, per direction A={counts['A']/16:.0f}, A*={counts['A*']/16:.0f}; values[:3]={np.round(vals[:3], 6)}")
fp = FeasibleProperty(problem, T, prior)
counts.clear(); t = time.perf_counter()
vals = [fp.support(d, data) for d in dirs]; dt = time.perf_counter() - t
print(f"{'primal-c':9s} 16 directions: {dt:7.3f} s, total A={counts['A']}, A*={counts['A*']} (incl. one-off _data_gram = {nD} each); values[:3]={np.round(vals[:3], 6)}")
bg = BackusGilbert(problem, T, prior)
counts.clear(); t = time.perf_counter(); est, r, n = bg.error_bars(data); dt = time.perf_counter() - t
print(f"{'certif.':9s} error_bars: {dt:7.3f} s, A={counts['A']}, A*={counts['A*']}")
