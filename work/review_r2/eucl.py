import time, numpy as np
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed, parallel_config
from joblib.parallel import get_active_backend
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.numerics.solvers import CGSolver
from pygeoinf2.traits import Traits

def nest_probe(i):
    b, n = get_active_backend()
    return type(b).__name__, getattr(b, "nesting_level", "n/a"), n
print("inside a loky worker, get_active_backend() ->", Parallel(n_jobs=2)(delayed(nest_probe)(i) for i in range(2))[0])
print("at top level ->", nest_probe(0))

rng = np.random.default_rng(0)
dim = 1500
M = rng.normal(size=(dim, dim)); M = M @ M.T / dim + np.eye(dim)
X = EuclideanSpace(dim)
A = LinearOperator.from_matrix(X, X, M, form="components").with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
Ainv = CGSolver(rtol=1e-8)(A)
cols = 200
def column(i): return X.to_components(Ainv(X.basis_vector(i)))
def timeit(label, fn):
    t = time.perf_counter(); fn(); print(f"{label:50s} {time.perf_counter()-t:7.2f} s")
with threadpool_limits(limits=1):
    timeit("serial, 200 CG columns, BLAS 1", lambda: [column(i) for i in range(cols)])
    timeit("threading n_jobs=4", lambda: Parallel(n_jobs=4, backend="threading")(delayed(column)(i) for i in range(cols)))
    timeit("threading n_jobs=8", lambda: Parallel(n_jobs=8, backend="threading")(delayed(column)(i) for i in range(cols)))
    Parallel(n_jobs=4)(delayed(column)(i) for i in range(4))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        timeit("loky n_jobs=4, inner 1", lambda: Parallel(n_jobs=4)(delayed(column)(i) for i in range(cols)))
        timeit("loky n_jobs=8, inner 1", lambda: Parallel(n_jobs=8)(delayed(column)(i) for i in range(cols)))
timeit("serial, BLAS 16 threads (numpy default)", lambda: [column(i) for i in range(cols)])
