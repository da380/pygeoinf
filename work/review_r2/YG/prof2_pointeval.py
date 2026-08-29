"""Point evaluation operator forward/adjoint at 1e4-1e5 points: v2 NUFFT vs direct vs dense vs v1."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import TransformCounter, bench, fmt
import numpy as np
import time
import cProfile, pstats, io

from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2
import pygeoinf2.symmetric_space.sphere as sphmod
import pygeoinf.symmetric_space.sphere as sph1

rng = np.random.default_rng(2)

sizes = [(64, 2000), (64, 10000), (64, 100000), (128, 2000), (128, 10000), (128, 100000), (256, 10000)]
if len(sys.argv) > 1 and sys.argv[1] == "quick":
    sizes = [(64, 10000), (128, 10000)]

for lmax, n in sizes:
    X = Sob2(lmax, 2.0, 0.2)
    X1 = sph1.Sobolev(lmax, 2.0, 0.2)
    pts = X.random_points(n, rng=rng)
    pts1 = [(float(p[0]), float(p[1])) for p in pts]
    x = X.heat_measure(0.3).sample(rng=rng)
    # same field in v1: transfer through coefficients
    c = X.to_components(x)
    x1 = X1.from_components(c)  # v1 components scale may differ; only timing matters
    w = rng.standard_normal(n)

    t0 = time.perf_counter(); A2 = X.point_evaluation_operator(pts); tb2 = time.perf_counter() - t0
    t0 = time.perf_counter(); A1 = X1.point_evaluation_operator(pts1, matrix_free=True); tb1 = time.perf_counter() - t0

    fns = {
        "v2 fwd (NUFFT, nthreads=1)": lambda: A2(x),
        "v2 adj (NUFFT, nthreads=1)": lambda: A2.adjoint(w),
        "v1 fwd (SHCoeffs.expand)": lambda: A1(x1),
        "v1 adj (per-point Legendre)": lambda: A1.adjoint(w),
        "v2 _angles (point conversion) only": lambda: X._angles(pts),
        "v2 evaluate nthreads=4": lambda: X.evaluate(x, pts, nthreads=4),
        "v2 accumulate nthreads=4": lambda: X.accumulate(w, pts, nthreads=4),
    }
    if n <= 10000 and lmax <= 128:
        A2d = None
        t0 = time.perf_counter(); A2d = X.point_evaluation_operator(pts, dense=True); tbd = time.perf_counter() - t0
        fns["v2 fwd dense=True"] = lambda: A2d(x)
        fns["v2 adj dense=True"] = lambda: A2d.adjoint(w)
    else:
        tbd = float("nan")
    if n <= 10000:
        # force the direct route
        saved = sphmod._TRANSFORM_MIN_POINTS
        def direct_fwd():
            sphmod._TRANSFORM_MIN_POINTS = 10**9
            try:
                return X.evaluate(x, pts)
            finally:
                sphmod._TRANSFORM_MIN_POINTS = saved
        def direct_adj():
            sphmod._TRANSFORM_MIN_POINTS = 10**9
            try:
                return X.accumulate(w, pts)
            finally:
                sphmod._TRANSFORM_MIN_POINTS = saved
        fns["v2 fwd direct (basis_matrix)"] = direct_fwd
        fns["v2 adj direct (basis_matrix)"] = direct_adj
    reps = 3 if n >= 100000 else 5
    print(f"\n-- lmax {lmax} (dim {X.dim}), n={n}; build v2 {1e3*tb2:.1f} ms, v1 {1e3*tb1:.1f} ms, v2 dense {1e3*tbd:.0f} ms --")
    print(fmt(bench(fns, reps=reps, warm=1)))
    # accuracy of v2 NUFFT vs direct at n=2000
    if n == 2000:
        sphmod._TRANSFORM_MIN_POINTS = 10**9
        ref = X.evaluate(x, pts)
        sphmod._TRANSFORM_MIN_POINTS = 200
        got = A2(x)
        print(f"  NUFFT vs direct max abs err {np.max(np.abs(ref-got)):.2e} (field max {np.max(np.abs(ref)):.2e})")
    with TransformCounter() as c:
        A2(x)
    cf = repr(c)
    with TransformCounter() as c:
        A2.adjoint(w)
    print(f"  transforms: v2 fwd {cf}; v2 adj {c!r}")

# profile breakdown at lmax 128, n=1e5
lmax, n = 128, 100000
X = Sob2(lmax, 2.0, 0.2)
pts = X.random_points(n, rng=rng)
x = X.heat_measure(0.3).sample(rng=rng)
A2 = X.point_evaluation_operator(pts)
w = rng.standard_normal(n)
A2(x); A2.adjoint(w)
for label, fn in (("forward", lambda: A2(x)), ("adjoint", lambda: A2.adjoint(w))):
    pr = cProfile.Profile(); pr.enable(); fn(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
    print(f"\n=== cProfile {label}, lmax {lmax}, n={n} ===")
    print("\n".join(s.getvalue().splitlines()[:30]))
