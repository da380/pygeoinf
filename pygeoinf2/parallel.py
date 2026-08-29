"""Running independent work in parallel, *around* operators.

The distinction that decides where this is used, from DESIGN.md and REVIEW.md
D-6: parallelism belongs at the embarrassingly parallel loops that surround an
operator, never inside its action. Drawing a thousand samples is a thousand
independent applications of one factor; applying that factor once is not
something this can help with, and threading a ``parallel=`` flag down into an
operator's own arithmetic is how v1 ended up with the flag on methods that
could not use it.

So the loops that take an ``n_jobs`` are the ones whose iterations are
genuinely independent and each expensive:

* drawing samples from a measure, where each draw may be a solve;
* filling in a matrix column by column, and everything built on that --
  assembling an operator, extracting its diagonals, forming a dense covariance;
* the probes of a randomised estimator;
* pointwise variance at a list of points;
* the support values of a feasible set in many directions.

``joblib`` does the work. It is a required dependency, as it was in v1. With
``n_jobs`` unset the loop is a list comprehension and joblib is never
touched.

**Which regime a problem is in decides whether to ask for this at all.**
Measured in ``review/parallel.md``:

* *NumPy-bound* work -- dense matrices, Euclidean spaces, anything whose cost
  is a BLAS call -- gains nothing from processes: the serial loop with BLAS
  threading is as fast as any parallel run of it. Leave ``n_jobs`` unset.
* *Transform-bound* work -- a sphere, a box with finufft -- gains from
  processes, and *only* processes: the ``pyshtools`` transforms crash the
  interpreter when called from two Python threads at once, whatever the
  OpenMP settings, so a threading backend must not be used there.
* *An operator that is itself parallel* -- MPI, a threaded PDE solver --
  cannot be shipped to worker processes at all. Leave ``n_jobs`` unset and
  parallelise across the scheduler instead: ``SeedSequence(seed).spawn(N)[k]``
  gives job ``k`` an independent, reproducible stream, and every sampling
  routine here takes an explicit ``rng``.

**Threads inside the workers.** Each worker is given one OpenMP/BLAS thread.
joblib's own default is ``cores // n_jobs``, which for the small transforms
this library does is a net loss -- measured at 8.6 s against 5.5 s for one
dense assembly at four workers -- and, for the ``pyshtools`` transforms, the
same oversubscription the ``nthreads=1`` default on the NUFFT avoids. A
variable already exported in the environment (``OMP_NUM_THREADS`` and its
BLAS siblings, set before Python starts as pyslfp's ``run_all.sh`` does) is
respected by joblib and wins over this.

**Steering from outside.** Everything here goes through
``joblib.Parallel`` with only ``n_jobs`` set, so a ``joblib.parallel_config``
context around the call decides the rest::

    with joblib.parallel_config(backend="loky", inner_max_num_threads=2):
        measure.samples(1000, rng=rng, n_jobs=8)

That is where a threading backend, a dask or ray cluster, or a different
thread budget is chosen; when a context is set it is respected entirely and
the one-thread default above is not applied. There is no ``backend``
argument on the loops themselves, because the one value it was there to
offer -- ``"threading"`` -- is the one that crashes on a sphere and is never
faster than BLAS threading where it is safe.

**Nesting.** A loop that finds itself already inside a worker runs serially.
joblib would otherwise turn the inner request into *threads* inside that
worker, silently -- which on a sphere is the crash above, and elsewhere is
oversubscription. So ``n_jobs`` can be forwarded freely from one loop to
another without ever producing threads.

**What a process backend asks of you.** The work has to be picklable. Bound
methods of ordinary objects are, and so are closures over them, which is why
the loops here can pass small lambdas; a function defined in ``__main__`` is
serialised *by value* and drags its module globals with it, so one that has a
Fortran extension in scope fails with ``cannot pickle 'fortran' object``. That
is the reason v1 keeps its workers in a module of top-level functions. Large
arrays reached through the closure -- a dense forward operator inside a
posterior -- are written once to shared memory by joblib and mapped by every
worker, not copied per task.

**Seeding.** Every sampling loop here spawns one generator per item from the
caller's ``rng`` *whether or not it runs in parallel*, so a result is a
function of the seed alone and does not change with ``n_jobs``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

__all__ = ["parallel_map", "resolve_jobs"]


def resolve_jobs(n_jobs: int | None) -> int:
    """How many workers to use, given what the caller asked for.

    Args:
        n_jobs: the caller's request. ``None`` or ``1`` means serial; ``-1``
            means every core this process may use -- joblib's count, which
            respects a CPU affinity mask or a container limit where
            ``os.cpu_count`` reports the whole machine.

    Returns:
        The number of workers, with ``1`` meaning "run it here".

    Raises:
        ValueError: if *n_jobs* is zero or below minus one.
    """
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(
            f"n_jobs is a positive count, or -1 for all cores; got {n_jobs}."
        )
    if n_jobs == -1:
        from joblib import cpu_count

        return max(int(cpu_count()), 1)
    return n_jobs


def _inside_a_worker() -> bool:
    """Whether this code is already running under a joblib worker."""
    from joblib.parallel import get_active_backend

    backend, _ = get_active_backend()
    return bool(getattr(backend, "nesting_level", 0))


def _context_is_set() -> bool:
    """Whether a ``joblib.parallel_config`` context has chosen a backend."""
    from joblib.parallel import _backend, default_parallel_config

    config = getattr(_backend, "config", None)
    if config is None:
        return False
    return config.get("backend") is not default_parallel_config["backend"]


def parallel_map(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    /,
    *,
    n_jobs: int | None = None,
) -> list[Any]:
    """``[function(item) for item in items]``, possibly in parallel.

    Serial with one job, with fewer than two items, or when already inside a
    worker (see the module docstring on nesting); joblib is not imported on
    the serial path. Otherwise ``joblib.Parallel`` with the process backend
    and one thread per worker, unless a ``joblib.parallel_config`` context is
    active, in which case that context decides.

    Args:
        function: called once per item. Must be picklable under a process
            backend.
        items: the work.
        n_jobs: workers. Serial by default.

    Returns:
        The results, in the order of *items*.
    """
    workers = resolve_jobs(n_jobs)
    materialised: Sequence[Any] = list(items)
    if workers == 1 or len(materialised) < 2 or _inside_a_worker():
        return [function(item) for item in materialised]

    from joblib import Parallel, delayed, parallel_config

    def run() -> list[Any]:
        return list(
            Parallel(n_jobs=workers)(delayed(function)(item) for item in materialised)
        )

    if _context_is_set():
        return run()
    # The context form, not the ``Parallel(inner_max_num_threads=)`` keyword:
    # in joblib 1.5 the keyword is accepted and then ignored by the reusable
    # executor, and the workers keep joblib's cores // n_jobs. Pinned by a
    # test that reads the workers' environment.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return run()
