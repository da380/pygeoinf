"""Shared helpers: a spherical-harmonic transform counter and interleaved timing."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np


class TransformCounter:
    """Counts SHExpandDH / MakeGridDH calls, however pyshtools is reached.

    v2 does ``from pyshtools.expand import SHExpandDH`` at call time; v1 goes
    through ``SHGrid.expand`` which resolves ``backend_module().SHExpandDH``
    dynamically. Patching the module attributes covers both.
    """

    names = ("SHExpandDH", "MakeGridDH")

    def __init__(self):
        self.counts = {n: 0 for n in self.names}
        self._saved = []

    def __enter__(self):
        import pyshtools
        import pyshtools.expand
        import pyshtools.backends.shtools
        mods = [pyshtools.expand, pyshtools.backends.shtools]
        for m in mods:
            for n in self.names:
                if hasattr(m, n):
                    orig = getattr(m, n)
                    self._saved.append((m, n, orig))
                    setattr(m, n, self._wrap(n, orig))
        return self

    def _wrap(self, n, orig):
        def wrapped(*a, **k):
            self.counts[n] += 1
            return orig(*a, **k)
        return wrapped

    def __exit__(self, *exc):
        for m, n, o in self._saved:
            setattr(m, n, o)

    def reset(self):
        for n in self.names:
            self.counts[n] = 0

    @property
    def total(self):
        return sum(self.counts.values())

    def __repr__(self):
        return f"A={self.counts['SHExpandDH']} S={self.counts['MakeGridDH']}"


def count(fn, *args, **kwargs):
    """Run fn once under a fresh counter, return (result, counter)."""
    with TransformCounter() as c:
        out = fn(*args, **kwargs)
    return out, c


def bench(fns, reps=5, warm=1):
    """Interleaved timing. fns: dict name -> zero-arg callable. Returns medians (s)."""
    times = {k: [] for k in fns}
    for _ in range(warm):
        for k, f in fns.items():
            f()
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            times[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in times.items()}


def fmt(medians, ref=None):
    lines = []
    for k, v in medians.items():
        s = f"  {k:48s} {1e3*v:10.2f} ms"
        if ref is not None and ref in medians and medians[ref] > 0:
            s += f"   x{v/medians[ref]:.2f} of {ref}"
        lines.append(s)
    return "\n".join(lines)
