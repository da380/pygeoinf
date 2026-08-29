"""Can dual_cost's id()-keyed memo return a stale residual for a new certificate?"""
import sys, gc
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import LinearForwardProblem
from pygeoinf2.inference.backus import DualFeasibleProperty
rng = np.random.default_rng(0)
X, D = EuclideanSpace(8), EuclideanSpace(3)
A = LinearOperator.from_matrix(X, D, rng.standard_normal((3, 8)), form="components")
T = LinearOperator.from_matrix(X, EuclideanSpace(1), rng.standard_normal((1, 8)), form="components")
problem = LinearForwardProblem(A, error=Ball(D, radius=0.1))
dfp = DualFeasibleProperty(problem, T, Ball(X, radius=1.0))
data = rng.standard_normal(3)
cost = dfp.dual_cost(np.array([1.0]), data)
hits = 0
for trial in range(200):
    lam1 = rng.standard_normal(3)
    v1 = cost(lam1)
    id1 = id(lam1)
    del lam1; gc.collect()
    lam2 = rng.standard_normal(3)
    if id(lam2) == id1:
        fresh = dfp.dual_cost(np.array([1.0]), data)   # uncached oracle
        g_cached = cost.gradient(lam2)
        g_fresh = fresh.gradient(lam2)
        if not np.allclose(g_cached, g_fresh):
            hits += 1
print("id reuse produced a wrong gradient in", hits, "of 200 trials")
