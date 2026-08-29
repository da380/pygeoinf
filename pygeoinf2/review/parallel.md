# Parallelism in pygeoinf2 — review appendix (2026-08-29)

> Follows up REVIEW.md D-6 ("parallel hooks *around* operators, never inside
> their actions"). Everything marked *(measured)* was run on this machine: an
> AMD Ryzen AI 7 PRO 350, 8 physical cores / 16 threads, joblib 1.5.3, Python
> 3.12, pyshtools 4.14.1, finufft 2.5.1. The scripts are throwaway; the numbers
> are quoted so that the recommendations can be checked against them.

## 0. Verdict in five lines

1. The v2 design — one `parallel_map`, `n_jobs=None` by default, joblib's
   process backend, seeds spawned per worker — is the right shape, and it is
   what v1 does minus the ~250 `parallel=`/`n_jobs=` signatures. Keep it.
2. Two defaults are wrong and cost real time or safety: workers are allowed
   several OpenMP/BLAS threads each (fixing that took `matrix(n_jobs=4)` from
   8.6 s to 5.5 s *(measured)*), and a nested `n_jobs` call silently becomes
   *threads* inside a worker — which on a sphere is a crash, not a slowdown.
3. D-6 is about half delivered. `samples`, `matrix`, `random_trace`,
   `pointwise_variance_at` and the support sweep have hooks; `diagonals`,
   `assembled`, every direct solver, the `random_range` family,
   `random_diagonal`/`deflated_diagonal`, `sample_expectation` and every
   `GaussianMeasure` route that forms a dense covariance do not. Two of the
   three pyslfp call sites that motivated D-6 are in the uncovered half.
4. `backend=` on six public signatures is a foot-gun that duplicates what
   `joblib.parallel_config` already provides. Recommend removing it.
5. The one thing the library cannot do is choose for the user: NumPy-bound
   work wants BLAS threads (or the threading backend) and gains *nothing* from
   processes; transform-bound work on a sphere wants processes with one thread
   each and *crashes* under threads. Serial-by-default with an explicit
   `n_jobs` is therefore correct, and the documentation should say which
   regime a problem is in rather than which flag to pass.

## 1. What v1 does

* `joblib.Parallel(n_jobs=...)` with the default loky (process) backend at
  every embarrassingly parallel loop: `samples`, dense matrix assembly
  (`matrix(parallel=, n_jobs=)`), diagonal extraction, sparse column
  probing, randomised range/eig/Cholesky block products (`parallel_mat_mat`),
  Hutchinson probes, support values across directions, chunked point
  evaluation on the sphere, preconditioner assembly.
* Two flags on ~250 signatures (`parallel: bool = False, n_jobs: int = -1`),
  each threaded by hand through every constructor and method that might call
  a loop. That duplication is what DESIGN.md §21 removed, correctly.
* A module of top-level worker functions (`pygeoinf/parallel.py`) so that
  the work pickles under processes; and `configure_threading(n)` in
  `utils.py`, a thin `threadpoolctl.threadpool_limits` wrapper that the user
  calls by hand before an outer loop.
* No seeding: draws use NumPy's legacy global state, so no parallel result is
  reproducible and, worse, forked workers can share a stream.

## 2. What v2 does now

`pygeoinf2/parallel.py`: `resolve_jobs` and `parallel_map(function, items,
n_jobs=, backend=)`. Serial with one job or without joblib; otherwise
`Parallel(n_jobs=workers[, backend=backend])(delayed(function)(item) ...)`.

Hooks (`n_jobs`, `backend`) exist at:

| entry point | file | note |
|---|---|---|
| `ProbabilityMeasure.samples` | probability/base.py:76 | spawns one stream per draw *only* in the parallel branch |
| `LinearOperator.matrix` | algebra/operators.py:666 | column or row loop |
| `random_trace` | numerics/randomised.py:532 | first block only; the adaptive `rtol` continuation is serial |
| `SymmetricSpace.pointwise_variance_at` | symmetric_space/base.py:969 | exact route only |
| `DualFeasibleProperty.support_values` | inference/backus.py:1256 | loses the warm start when parallel (documented) |

Tests: `tests/test_probability.py::TestParallelLoops` — the law of parallel
draws (threading backend on a weighted space), matrix equality (threading),
the process backend on a sphere (`@slow`), `resolve_jobs` validation, and the
no-joblib fallback. Nothing tests nesting, affinity, or the inner thread cap.

Two documentation inconsistencies:

* `parallel.py` and `samples` call joblib "an optional extra"; `pyproject.toml`
  line 16 lists `joblib>=1.5.3` in the *required* dependencies (as v1 needs
  it). Pick one. If it is to become optional for v2, the silent serial
  fallback in `parallel_map` needs a warning (see R6); if it stays required,
  drop the fallback and the sentence.
* `parallel.py`'s oversubscription paragraph says "this module does not do it
  for you". It does, partly and badly: joblib's loky backend sets
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` in each worker
  to `cpu_count() // n_jobs` *(measured: 4 with `n_jobs=4`, 2 with 8)*. That
  is the wrong number for this library (§3.2).

## 3. Measurements

Test problem: example 21 at `lmax=24` (dim 625, 960 travel-time data, dense
path operator), the posterior from `LinearGaussianInversion`. One column of
`posterior.covariance` is a prior covariance application, a forward/adjoint
pair and a solve; ~20 ms alone.

### 3.1 The threading backend on a sphere

`posterior.samples(16, n_jobs=4, backend="threading")` kills the interpreter
— SIGFPE on one run, SIGSEGV on the next — with or without
`threadpool_limits(1)` around it *(measured; three of three runs)*. The
docstring's claim is right and the test that asserts it is right to be
process-only. This is the fact that decides everything else: the *default*
must be processes, and nothing may turn a request into threads behind the
user's back (§3.5).

### 3.2 Dense assembly, `posterior.covariance.matrix(form="galerkin")`, 625 columns

| configuration | wall time |
|---|---|
| serial, NumPy's default 16 BLAS threads | 22.0 s |
| serial, BLAS limited to 1 thread | 13.9 s |
| loky `n_jobs=4`, joblib's default inner cap (4 threads each) | 8.6 s |
| loky `n_jobs=8`, default cap (2 each) | 6.5–8.1 s |
| loky `n_jobs=16`, default cap (1 each) | 7.0 s |
| loky `n_jobs=4`, `inner_max_num_threads=1` | **5.5 s** |
| loky `n_jobs=8`, `inner_max_num_threads=1` | **5.3 s** |
| threading, any `n_jobs` | crash |

Three things to take from this.

*BLAS threading is a net loss on these sizes even serially* (22 s → 14 s by
turning it off). The transforms and matvecs are too small; sixteen threads
spin on each. The same effect inside workers is why joblib's default inner
cap costs 3 s of 8.6.

*The speed-up ceiling on this laptop is about 2.6×, and it is the chip, not
joblib.* Per-column time inside a worker rises from 20 ms alone to 29–30 ms
with four workers and 32–45 ms with eight *(measured)*; `lscpu` reports the
cores running at 34 % of nominal under load. Chunking the columns into eight
tasks, changing `batch_size`, or bypassing `parallel_map` entirely changes
nothing *(measured, all within 5 %)*: dispatch overhead is not where the time
goes. On a workstation or a node with sustained clocks the same code should
approach the core count. Do not add batching knobs to fix a problem the
machine has.

*Hyperthreads buy nothing.* `n_jobs=16` was no better than 8. `-1` should mean
physical cores, or at least affinity-aware cores (§3.6).

### 3.3 Sampling

| | serial | loky 4, first call | loky 4, warm | loky 16 |
|---|---|---|---|---|
| 16 posterior draws, dense forward (≈25 ms each) | 0.43 s | 2.6 s | 1.2 s | 3.5 s |
| 8 posterior draws, matrix-free forward (≈0.5 s each) | 3.9 s | 3.6 s | 2.2 s | 4.4 s |

The first call pays 1–2 s to spawn workers and import the package in each;
loky keeps them alive (300 s idle) so the second call does not. Cheap draws
should stay serial; the docstrings say so, and the numbers agree. `n_jobs=-1`
is worse than 4 in both rows — sixteen workers importing pyshtools and
throttling the chip for 16 tasks.

### 3.4 What crosses the process boundary

The closure `parallel_map` ships is `lambda stream: self.sample(rng=stream)`
with `self` the posterior. cloudpickle serialises it at 4.9 MB — 4.85 MB of
which is the dense path operator; the prior and the space are 0.06 MB and
0.05 MB *(measured)*. joblib's pickler replaces any array over 1 MB with a
memmap in `/dev/shm`, keyed by content hash, so the 5 MB is written once and
each worker sees an `np.memmap` *(measured: a closure over a 32 MB array
ships 20 tasks in 0.12 s warm)*. So large dense operators in a measure are not
a per-task cost. Objects that hold unpicklable state (an MPI communicator, a
PETSc matrix, a Fortran module handle in `__main__`) fail at dispatch with a
clear error; that is the case the v1 note about top-level worker functions
is about, and it stays true.

### 3.5 Nesting *(measured)*

Inside a loky worker, `get_active_backend()` returns
`(ThreadingBackend, nesting_level=1)`. A `Parallel(n_jobs=4)` call made there
runs on **threads in that worker**, with no warning. Today no v2 loop
forwards `n_jobs` into a call that has its own loop, so it is unreachable —
but the moment `sample_expectation(n_jobs=)` forwards to `samples`, or a
sample's `gain` uses a solver that assembles with `matrix(n_jobs=)`, it is
reachable, and on a sphere it is §3.1.

### 3.6 Core counts *(measured)*

`os.cpu_count()` is 16 under `taskset -c 0-3`; `joblib.cpu_count()` and
`len(os.sched_getaffinity(0))` are 4. `resolve_jobs(-1)` uses the first, so
in a SLURM allocation of 4 cores on a 128-core node it would start 128
workers. `joblib.cpu_count(only_physical_cores=True)` gives 8 here.

### 3.7 `parallel_config` reaches the library *(measured)*

Because `parallel_map` passes `n_jobs` but not `backend` unless asked,
joblib's context manager steers it: inside
`with joblib.parallel_config(backend="threading"):` v2's loops run on
threads; inside `parallel_config(backend="loky", inner_max_num_threads=1)`
workers see `OMP_NUM_THREADS=1`; a context `n_jobs` is overridden by the
explicit one, as it should be. This is the extension point for dask, ray, or
a user's registered backend, and it already works with no v2 code.

### 3.8 Where threads *are* the right tool *(measured)*

200 CG columns of a 1500-dim dense SPD Euclidean operator:

| | wall time |
|---|---|
| serial, BLAS 1 thread | 1.32 s |
| threading `n_jobs=4` / `8` | 0.50 s / 0.36 s |
| loky `n_jobs=4` / `8`, inner 1 | 1.39 s / 1.03 s |
| serial, BLAS 16 threads (NumPy's default) | **0.33 s** |

NumPy releases the GIL in the matvec, so threads give 3.7× and processes give
nothing over a 1500×1500 matrix. And just letting BLAS thread the serial loop
is as good as either. This is the regime pyslfp's Euclidean codomains and any
`from_matrix` operator live in.

### 3.9 Environment variables before Python starts *(measured)*

pyslfp's `Heathcote2026/run_all.sh` exports `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to `N_THREADS` (default
1) before invoking Python, and sets `--max-jobs` to `cores / N_THREADS`.
That is the right discipline and it composes with joblib: loky honours a
variable that is *already set* in the parent and fills in only the ones that
are not. With only `OMP_NUM_THREADS=1` exported, workers here saw
`OMP_NUM_THREADS=1` but `OPENBLAS_NUM_THREADS=4` — so exporting the whole
set, as the script does, is what makes it work. The script therefore already
achieves R1 from outside; R1 makes the library do the same when the script
is absent, and an exported variable still wins because joblib defers to it.
The one thing the variables cannot reach is finufft, whose thread count is a
call argument — which v2 now pins to one itself.

## 4. What is possible — the design space, honestly

There are exactly four ways to use more than one core here, and they do not
compose freely.

1. **BLAS/OpenMP threads inside a single operator application.** Free, on by
   default, good for dense NumPy work (§3.8), bad for small spectral
   transforms (§3.2, and the finufft `nthreads=1` default already in v2).
   Controlled by `threadpoolctl` or environment variables, by the *user*.
2. **Python threads across independent applications** (joblib `threading`).
   Good for GIL-releasing NumPy work; fatal on pyshtools (§3.1); useless for
   pure-Python-bound work. Never the default.
3. **Processes across independent applications** (joblib `loky`). The general
   tool: safe on the sphere, needs picklable work, pays a spawn cost once per
   session and a per-batch pickle of the closure (cheap, memmapped). Its
   workers must be told to use one thread each (§3.2). Cannot host an
   operator that is itself MPI-parallel, and should not host one that is
   itself OpenMP-parallel at scale — the cap in R1 would strangle it, which
   is the correct outcome *if the user asked for `n_jobs`* and a surprise
   otherwise; hence serial by default.
4. **Scheduler-level parallelism**: N array jobs, each running the serial
   code with its own stream. This is the only route for operators that run
   large parallel codes (the MFEM/PETSc/MPI case), and v2's explicit `rng`
   makes it clean: `SeedSequence(seed).spawn(N)[k]` gives job `k` an
   independent, reproducible stream. Nothing in the library needs to know.

Nesting 3 inside 3 is turned into 2 by joblib (§3.5). Nesting 1 inside 3 is
the oversubscription case, handled by capping. Nesting 3 inside a user's own
MPI program is a non-starter and should be documented as such.

## 5. Recommendations

> **Status (2026-08-29, merged to `refactor`):** R1–R6 implemented in `parallel.py` and the five loops (`542de76`); R7's hooks are in (`6620637`: `apply_block`, `n_jobs` on `diagonals`, `assembled`, the direct solvers, Jacobi, the `random_*` routines, `deflated_diagonal`, `sample_expectation`, the sampling route of `pointwise_variance_at`) — and `with_dense_covariance(n_jobs=)` landed on 2026-08-30 (`99f4b25`) — though `n_jobs` gave no measurable gain there, the parallel overhead dominating `matrix()` for both cheap and CG-backed covariances, so R7 is complete but its last item is of unproven value; R8–R10 as recorded, R10's nesting and affinity tests written, the seed-invariance test written. The `Parallel(inner_max_num_threads=)` keyword turned out to be ignored by joblib 1.5.3's reusable executor — the `parallel_config` context form works and is what is used, pinned by a test that reads the workers' environment.

Ordered by value; the first four are small edits to `parallel.py`.

**R1. Cap inner threads to one, by default.** In `parallel_map`, run under
`parallel_config(backend=backend or "loky", inner_max_num_threads=1)` (or
pass `inner_max_num_threads=1` to `Parallel` with an explicit backend). 8.6 s
→ 5.5 s at `n_jobs=4` *(measured)*. Let a user override with their own
`parallel_config` context around the call, and say so in the docstring. This
replaces the paragraph that says the module does not manage threads.

**R2. Refuse to nest.** At the top of `parallel_map`:
`backend, _ = joblib.parallel.get_active_backend(); if getattr(backend,
"nesting_level", 0) > 0: run serially`. A nested request then costs nothing
and can never become threads on a sphere. One test: a `parallel_map` whose
function calls `parallel_map(n_jobs=2)` must report a single thread in the
worker.

**R3. `-1` means the cores you actually have.** `resolve_jobs(-1)` →
`joblib.cpu_count()` (affinity- and cgroup-aware), and consider
`only_physical_cores=True`: hyperthreads gained nothing here and cost
16 imports of pyshtools. If joblib is optional, fall back to
`len(os.sched_getaffinity(0))` on Linux and `os.cpu_count()` elsewhere.

**R4. Drop `backend=` from the six public signatures.** It is on
`samples`, `matrix`, `random_trace`, `pointwise_variance_at`,
`support_values` and `parallel_map`. It exists to offer `"threading"`, which
crashes on the sphere and is never faster than BLAS threads where it is safe
(§3.8). `joblib.parallel_config(backend=...)` gives the same control from
outside, reaches every v2 loop today (§3.7), and is where dask/ray backends
plug in. Keep `backend` on `parallel_map` only if a test needs it; the
public API carries `n_jobs` alone. Document the `parallel_config` idiom once,
in `parallel.py`.

**R5. Spawn streams in the serial path too.** `samples` spawns a stream per
draw only when parallel, so a result changes when `n_jobs` changes; the
docstring has to explain this. Spawning always (`rng.spawn(n)` is
microseconds) makes the draws a function of the seed alone, the same at any
`n_jobs`, and lets a test pin values while running parallel. The same for
`random_trace`'s probes — and its adaptive continuation should draw its extra
blocks through the same `parallel_map`, not the serial loop it has now.

**R6. Decide whether joblib is optional, then act on it.** If required (as
`pyproject.toml` says), delete the `ImportError` fallback and the "optional
extra" sentences. If optional, `warnings.warn` once when `n_jobs > 1` and
joblib is missing: a user who asked for eight workers and got one should be
told. Either is fine; the current state promises both.

**R7. Finish D-6's list.** In order of pyslfp's need:

* `DirectSolver` (`LUSolver`, `CholeskySolver`, `EigenSolver`): take
  `n_jobs` in the constructor and pass it to `operator.matrix(...)` in
  `_invert` (solvers.py:397). This is the Woodbury inner solve on a ~3k-dim
  surrogate normal operator — one CG solve per column — the largest single
  parallel win in the pyslfp scripts and currently unreachable.
* `GaussianMeasure` routes that form the dense covariance — `as_multivariate_normal`
  (gaussian.py:1097), `credible_set`, `ambient_ball`, `hilbert_schmidt_norm`,
  `nuclear_norm`, `with_sparse_approximation`, `_symmetric_matrix` — are
  v1's `with_dense_covariance(parallel=)`. Rather than an `n_jobs` on each,
  give `GaussianMeasure` one method that assembles and caches the dense
  covariance with `n_jobs` (`with_dense_covariance(n_jobs=)` returning a
  measure whose covariance is `from_matrix`), which is exactly v1's shape and
  what every pyslfp script calls before plotting.
* `LinearOperator.diagonals(probe="exact")` (operators.py:768): the column
  loop is the one in `matrix`; the preconditioners
  (`JacobiPreconditioner`, `BandedPreconditioner`, `NormalDiagonalPreconditioner`)
  build on it. `assembled()` should forward too.
* `random_range` and everything on it (`random_eig`, `random_svd`,
  `random_cholesky`): the block product `A @ Omega` is `k` independent
  applications — v1's `parallel_mat_mat`. `deflated_diagonal` and
  `random_diagonal` as `random_trace` is.
* `sample_expectation` and the sample route of `pointwise_variance_at`
  forward `n_jobs` to `samples`.

Every one is a `parallel_map` over an existing comprehension; none needs a
new mechanism. Because of R2 the forwarding cannot nest into threads.

**R8. Do not add chunking, batch sizes, or an executor protocol.** Measured
not to matter (§3.2); joblib's auto-batching and `parallel_config` cover the
rest. v1 needed none of them either.

**R9. Write the three-regime note where users will find it** — in
`parallel.py`'s docstring and one paragraph in `CURRENT_STATE.md`:

> *NumPy-bound* (dense matrices, Euclidean spaces): leave `n_jobs` unset and
> let BLAS thread; processes will not help. *Transform-bound* (sphere, boxes
> with finufft): `n_jobs=k` runs `k` processes with one thread each, and
> `backend="threading"` will crash. *Your operator is itself parallel* (MPI,
> a threaded PDE solver): leave `n_jobs` unset, or parallelise across the
> scheduler with `SeedSequence(seed).spawn(N)[k]` per job.

**R10. Tests.** Un-`@slow` a small version of the sphere process test (lmax 4,
two draws); add the nesting test (R2); test `resolve_jobs(-1)` under a
restricted affinity with `os.sched_setaffinity` on Linux; test that the same
seed gives the same draws at `n_jobs=1` and `2` once R5 is in.

## 6. What I did not do

* No cluster or MPI measurement; §4.4 is reasoning, not data.
* The uncommitted example edits (`plt.show()` appended to six examples,
  reformatting in `28_nonlinear_map.py`) were skimmed, not reviewed.
* Nothing here touches D-6's boundary: parallelism inside an operator's own
  action stays the operator's business, as decided.
