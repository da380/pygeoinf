import sys, time, numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
sys.path.insert(0, "/home/david/dev/pygeoinf/pygeoinf2/tests")
from scipy.linalg import cho_solve
from conftest import make_dense_metric_space, DenseMetricSpace, WeightedSpace

class CachedDenseMetricSpace(DenseMetricSpace):
    """The fixture, with solve_gram through the Cholesky it already holds."""
    def solve_gram(self, c):
        return cho_solve((self._chol, True), c)

def dense_space(dim):
    """make_dense_metric_space for dim <= 60 (cond < 1e3); beyond that its root
    is exponentially ill-conditioned (cond 1e17 at dim 200), so a random
    orthogonal conjugation of eigenvalues in [0.5, 2] stands in: dense,
    non-diagonal, cond 4."""
    if dim <= 60:
        return CachedDenseMetricSpace(make_dense_metric_space(dim)._gram)
    rng = np.random.default_rng(20260829)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    lam = rng.uniform(0.5, 2.0, dim)
    return CachedDenseMetricSpace((Q * lam) @ Q.T)

def weighted_space(dim, seed=1):
    g = np.random.default_rng(seed).uniform(0.5, 2.0, dim)
    return WeightedSpace(g)

def timeit(fn, repeats=3, number=1):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t) / number)
    return best

def interleave(named, repeats=3, number=1):
    """Run alternatives interleaved; return dict of best times."""
    best = {k: float("inf") for k in named}
    for _ in range(repeats):
        for k, fn in named.items():
            t = time.perf_counter()
            for _ in range(number):
                fn()
            best[k] = min(best[k], (time.perf_counter() - t) / number)
    return best

class Counter:
    def __init__(self): self.n = 0
    def wrap(self, f):
        def g(*a, **k):
            self.n += 1
            return f(*a, **k)
        return g
