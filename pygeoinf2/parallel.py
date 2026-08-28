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
* pointwise variance at a list of points.

``joblib`` does the work when it is installed and more than one job is asked
for. It is an optional extra: with ``n_jobs`` unset, or joblib absent, the loop
runs serially and nothing is imported.

**Do not use the threading backend with the spectral transforms.** This is not
a performance note. ``pyshtools``' transforms are compiled with OpenMP and are
not re-entrant: calling them from two Python threads at once segfaults the
interpreter, and it does so silently -- no exception, no traceback, a dropped
core file. Measured here on ``pointwise_variance_at`` over a handful of points
on a sphere, which crashed under ``backend="threading"`` and was correct under
the default.

The default backend is joblib's, which is processes, and that is safe: each
worker has its own OpenMP state. The cost is that the work must be picklable,
which a closure over a local is not. ``backend="threading"`` remains available
because it is right for anything that stays in NumPy -- a Euclidean space, a
dense operator -- and wrong for anything that reaches a transform.

The related performance note stands too: ``finufft`` and ``pyshtools``
parallelise internally and by default take every core, so nesting them inside
an outer parallel loop oversubscribes the machine. The review measured a single
sphere evaluation at 173-794 ms with the default thread count against 20 ms
with one. Set the inner library to one thread when running an outer loop; this
module does not do it for you, because it cannot know what a caller's operator
reaches for.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

__all__ = ["parallel_map", "resolve_jobs"]


def resolve_jobs(n_jobs: int | None) -> int:
    """How many workers to use, given what the caller asked for.

    Args:
        n_jobs: the caller's request. ``None`` or ``1`` means serial; ``-1``
            means every core.

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
        import os

        return os.cpu_count() or 1
    return n_jobs


def parallel_map(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    /,
    *,
    n_jobs: int | None = None,
    backend: str | None = None,
) -> list[Any]:
    """``[function(item) for item in items]``, possibly in parallel.

    Serial whenever it can be: with one job, or with joblib absent, this is the
    list comprehension and joblib is never imported. That keeps the dependency
    genuinely optional and keeps a serial run free of the overhead of pretending
    otherwise.

    Args:
        function: called once per item. Must be picklable under a process
            backend, which a closure over a local is not -- pass ``backend=
            "threading"`` for those, or for work that already releases the GIL.
        items: the work.
        n_jobs: workers. Serial by default.
        backend: joblib's backend. Its own default -- processes -- when
            omitted, which is the safe choice: see the module docstring on why
            ``"threading"`` must not be used with the spectral transforms.

    Returns:
        The results, in the order of *items*.
    """
    workers = resolve_jobs(n_jobs)
    materialised: Sequence[Any] = list(items)
    if workers == 1 or len(materialised) < 2:
        return [function(item) for item in materialised]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [function(item) for item in materialised]

    settings: dict[str, Any] = {"n_jobs": workers}
    if backend is not None:
        settings["backend"] = backend
    return list(Parallel(**settings)(delayed(function)(item) for item in materialised))
