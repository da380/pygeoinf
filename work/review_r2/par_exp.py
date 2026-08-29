"""Parallelism experiments on the v2 posterior of example 21 (smaller lmax)."""
import os, sys, time, pickle
import numpy as np

mode = sys.argv[1]
lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 24

from pygeoinf2.inference import LinearGaussianInversion, LinearForwardProblem
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev

rng = np.random.default_rng(1)
X = Sobolev(lmax, 2.0, 0.1)
receivers = X.stations(count=24, rng=rng)
sources = X.earthquakes(count=40, minimum_magnitude=5.5, rng=rng)
paths = [(s, r) for s in sources for r in receivers]
forward = X.path_average_operator(paths, count=16, dense=(mode != "matfree"))
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
problem = LinearForwardProblem(forward, error=noise)
prior = X.heat_measure(0.17, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
estimator = LinearGaussianInversion(problem, prior)
posterior = estimator(data)
print(f"dim X = {X.dim}, data dim = {problem.data_space.dim}")

def timeit(label, fn):
    t = time.perf_counter(); out = fn(); dt = time.perf_counter() - t
    print(f"{label:55s} {dt:8.2f} s"); return out

if mode in ("samples", "matfree"):
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    timeit(f"serial samples({n})", lambda: posterior.samples(n, rng=np.random.default_rng(0)))
    timeit(f"loky n_jobs=4 samples({n}) [first call, spawns]", lambda: posterior.samples(n, rng=np.random.default_rng(0), n_jobs=4))
    timeit(f"loky n_jobs=4 samples({n}) [second call]", lambda: posterior.samples(n, rng=np.random.default_rng(0), n_jobs=4))
    timeit(f"loky n_jobs=-1 samples({n})", lambda: posterior.samples(n, rng=np.random.default_rng(0), n_jobs=-1))
    import cloudpickle
    blob = cloudpickle.dumps(lambda stream: posterior.sample(rng=stream))
    print(f"cloudpickle size of the sample closure: {len(blob)/1e6:.2f} MB")
    timeit("cloudpickle dumps of that closure", lambda: cloudpickle.dumps(lambda stream: posterior.sample(rng=stream)))

elif mode == "threads":
    n = 16
    timeit(f"serial samples({n})", lambda: posterior.samples(n, rng=np.random.default_rng(0)))
    timeit(f"threading n_jobs=4 samples({n})", lambda: posterior.samples(n, rng=np.random.default_rng(0), n_jobs=4, backend="threading"))
    print("threading survived")

elif mode == "threads_limited":
    from threadpoolctl import threadpool_limits
    n = 16
    with threadpool_limits(limits=1):
        timeit(f"threading n_jobs=4 samples({n}) under threadpool_limits(1)", lambda: posterior.samples(n, rng=np.random.default_rng(0), n_jobs=4, backend="threading"))
    print("threading survived")

elif mode == "matrix":
    cov = posterior.covariance
    timeit("serial covariance.matrix(galerkin)", lambda: cov.matrix(form="galerkin"))
    timeit("loky n_jobs=4 covariance.matrix(galerkin) [first]", lambda: cov.matrix(form="galerkin", n_jobs=4))
    timeit("loky n_jobs=4 covariance.matrix(galerkin) [second]", lambda: cov.matrix(form="galerkin", n_jobs=4))
    timeit("loky n_jobs=8 covariance.matrix(galerkin)", lambda: cov.matrix(form="galerkin", n_jobs=8))
    timeit("loky n_jobs=-1 covariance.matrix(galerkin)", lambda: cov.matrix(form="galerkin", n_jobs=-1))
    timeit("threading n_jobs=4 covariance.matrix(galerkin)", lambda: cov.matrix(form="galerkin", n_jobs=4, backend="threading"))

elif mode == "workerenv":
    from joblib import Parallel, delayed
    def probe(i):
        import threadpoolctl, os
        info = {(d["internal_api"], d["num_threads"]) for d in threadpoolctl.threadpool_info()}
        return os.getpid(), os.environ.get("OMP_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"), sorted(info)
    for r in Parallel(n_jobs=4)(delayed(probe)(i) for i in range(4)):
        print(r)
    print("parent:", os.environ.get("OMP_NUM_THREADS"), sorted({(d["internal_api"], d["num_threads"]) for d in __import__("threadpoolctl").threadpool_info()}))

if mode == "pickle":
    from joblib.externals import cloudpickle
    fn = lambda stream: posterior.sample(rng=stream)
    blob = timeit("cloudpickle dumps of the sample closure", lambda: cloudpickle.dumps(fn))
    print(f"  size {len(blob)/1e6:.2f} MB")
    blob2 = cloudpickle.dumps(lambda i: posterior.covariance(X.basis_vector(i)))
    print(f"  covariance column closure {len(blob2)/1e6:.2f} MB")
    print(f"  prior alone {len(cloudpickle.dumps(prior))/1e6:.2f} MB, space alone {len(cloudpickle.dumps(X))/1e6:.2f} MB, forward {len(cloudpickle.dumps(forward))/1e6:.2f} MB")

elif mode == "crash":
    import faulthandler; faulthandler.enable()
    cov = posterior.covariance
    timeit("loky n_jobs=16 covariance.matrix(galerkin)", lambda: cov.matrix(form="galerkin", n_jobs=16))
    print("survived")

elif mode == "blas1":
    from threadpoolctl import threadpool_limits
    cov = posterior.covariance
    with threadpool_limits(limits=1):
        timeit("serial covariance.matrix(galerkin), BLAS 1 thread", lambda: cov.matrix(form="galerkin"))
        timeit("loky n_jobs=4, parent BLAS 1 thread [first]", lambda: cov.matrix(form="galerkin", n_jobs=4))
        timeit("loky n_jobs=4, parent BLAS 1 thread [second]", lambda: cov.matrix(form="galerkin", n_jobs=4))
        timeit("loky n_jobs=8, parent BLAS 1 thread", lambda: cov.matrix(form="galerkin", n_jobs=8))

if mode == "diag":
    from threadpoolctl import threadpool_limits
    from joblib import Parallel, delayed
    cov = posterior.covariance
    def column(index):
        return X.to_components(cov(X.basis_vector(index)))
    def chunk(indices):
        return [column(i) for i in indices]
    with threadpool_limits(limits=1):
        timeit("parent: 80 columns, BLAS 1", lambda: [column(i) for i in range(80)])
        # warm the executor
        Parallel(n_jobs=8)(delayed(column)(i) for i in range(8))
        timeit("loky 8: 625 columns, default batching", lambda: Parallel(n_jobs=8)(delayed(column)(i) for i in range(625)))
        timeit("loky 8: 625 columns, batch_size=40", lambda: Parallel(n_jobs=8, batch_size=40)(delayed(column)(i) for i in range(625)))
        parts = np.array_split(np.arange(625), 8)
        timeit("loky 8: 8 chunk tasks of ~78 columns", lambda: Parallel(n_jobs=8)(delayed(chunk)(p) for p in parts))
        parts = np.array_split(np.arange(625), 32)
        timeit("loky 8: 32 chunk tasks of ~20 columns", lambda: Parallel(n_jobs=8)(delayed(chunk)(p) for p in parts))
        timeit("loky 8 verbose: 625 columns", lambda: Parallel(n_jobs=8, verbose=5)(delayed(column)(i) for i in range(625)))
    # worker-side compute rate
    def timed_chunk(indices):
        t = time.perf_counter(); out = [column(i) for i in indices]; return time.perf_counter() - t
    parts = np.array_split(np.arange(625), 8)
    t = Parallel(n_jobs=8)(delayed(timed_chunk)(p) for p in parts)
    print("in-worker seconds per ~78-column chunk (OMP capped at 2 each):", np.round(t, 2))

if mode == "inner1":
    from threadpoolctl import threadpool_limits
    from joblib import Parallel, delayed, parallel_config
    cov = posterior.covariance
    def column(index):
        return X.to_components(cov(X.basis_vector(index)))
    def timed_chunk(indices):
        t = time.perf_counter(); out = [column(i) for i in indices]; return time.perf_counter() - t
    with threadpool_limits(limits=1):
        for jobs in (4, 8):
            with parallel_config(backend="loky", inner_max_num_threads=1):
                Parallel(n_jobs=jobs)(delayed(column)(i) for i in range(8))
                timeit(f"loky {jobs}, inner threads 1: 625 columns", lambda: Parallel(n_jobs=jobs)(delayed(column)(i) for i in range(625)))
                parts = np.array_split(np.arange(625), jobs)
                t = Parallel(n_jobs=jobs)(delayed(timed_chunk)(p) for p in parts)
                print(f"  in-worker ms per column:", np.round(1000*np.array(t)/len(parts[0]), 1))

elif mode == "cols80":
    from threadpoolctl import threadpool_limits
    cov = posterior.covariance
    with threadpool_limits(limits=1):
        t = time.perf_counter(); [X.to_components(cov(X.basis_vector(i))) for i in range(80)]
        print(f"pid {os.getpid()} 80 columns: {1000*(time.perf_counter()-t)/80:.1f} ms per column")
