import os, sys, time, numpy as np
from joblib import Parallel, delayed, parallel_config, cpu_count
print("os.cpu_count", os.cpu_count(), "| affinity", len(os.sched_getaffinity(0)), "| joblib cpu_count", cpu_count(), "| physical", cpu_count(only_physical_cores=True))

# 1. nested Parallel inside a loky worker
def inner(i):
    from joblib.parallel import get_active_backend
    def leaf(j):
        return os.getpid()
    with Parallel(n_jobs=4) as p:
        pids = p(delayed(leaf)(j) for j in range(4))
    backend, n = get_active_backend()
    return os.getpid(), type(backend).__name__, n, len(set(pids)), p.n_jobs
print("nested:", Parallel(n_jobs=2)(delayed(inner)(i) for i in range(2)))

# 2. does a parallel_config context reach a library call that passes n_jobs but not backend?
def which(i):
    import threading
    return type(threading.current_thread()).__name__, os.getpid()
with parallel_config(backend="threading"):
    print("context threading ->", set(Parallel(n_jobs=2)(delayed(which)(i) for i in range(4))))
with parallel_config(backend="loky", inner_max_num_threads=1):
    print("context loky inner=1 ->", set(Parallel(n_jobs=2)(delayed(lambda i: os.environ.get("OMP_NUM_THREADS"))(i) for i in range(4))))
with parallel_config(n_jobs=4):
    p = Parallel()  # no n_jobs given: takes the context's
    print("context n_jobs=4, Parallel() sees", p.n_jobs, "| Parallel(n_jobs=2) sees", Parallel(n_jobs=2).n_jobs)

# 3. large array inside a closure: memmapped?
big = np.random.default_rng(0).normal(size=(2000, 2000))  # 32 MB
fn = lambda i: (type(big).__name__, float(big[i, i]))
t = time.perf_counter(); out = Parallel(n_jobs=2)(delayed(fn)(i) for i in range(20)); dt = time.perf_counter() - t
print("closure over 32 MB array -> worker sees", out[0][0], f"({dt:.2f} s for 20 tasks)")
t = time.perf_counter(); out = Parallel(n_jobs=2)(delayed(fn)(i) for i in range(20)); print(f"  second call {time.perf_counter()-t:.2f} s")
