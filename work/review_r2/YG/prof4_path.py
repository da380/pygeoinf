"""Path integral / average operators and geodesic ball averages: construction and application."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import TransformCounter, bench, fmt
import numpy as np
import time
import cProfile, pstats, io

from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2
import pygeoinf.symmetric_space.sphere as sph1

rng = np.random.default_rng(4)

for lmax, npaths in ((64, 200), (128, 200), (128, 2000)):
    X = Sob2(lmax, 2.0, 0.1); X1 = sph1.Sobolev(lmax, 2.0, 0.1)
    src = X.random_points(int(np.sqrt(npaths)) + 1, rng=rng)
    rec = X.random_points(int(np.sqrt(npaths)) + 1, rng=rng)
    paths = [(s, r) for s in src for r in rec][:npaths]
    paths1 = [((float(s[0]), float(s[1])), (float(r[0]), float(r[1]))) for s, r in paths]
    x = X.heat_measure(0.3).sample(rng=rng)
    x1 = X1.from_components(X.to_components(x))

    t0 = time.perf_counter(); P2 = X.path_integral_operator(paths); tb2 = time.perf_counter() - t0
    t0 = time.perf_counter(); P1 = X1.path_average_operator(paths1, matrix_free=True); tb1 = time.perf_counter() - t0
    nodes = P2.domain.dim  # not nodes; get node count from the weight operator
    # count nodes
    total_nodes = sum(len(X._path_nodes(s, e, None)[0]) for s, e in paths)
    w = rng.standard_normal(npaths)
    fns = {
        "v2 path_integral fwd": lambda: P2(x),
        "v2 path_integral adj": lambda: P2.adjoint(w),
        "v1 path_average(matrix_free) fwd": lambda: P1(x1),
        "v1 path_average(matrix_free) adj": lambda: P1.adjoint(w),
    }
    tbd = float("nan")
    if total_nodes * X.dim <= 60_000_000:
        t0 = time.perf_counter(); P2d = X.path_integral_operator(paths, dense=True); tbd = time.perf_counter() - t0
        fns["v2 dense=True fwd"] = lambda: P2d(x)
        fns["v2 dense=True adj"] = lambda: P2d.adjoint(w)
    print(f"\n-- lmax {lmax}, {npaths} paths, {total_nodes} nodes; build v2 {1e3*tb2:.0f} ms, v1 {1e3*tb1:.0f} ms, v2 dense {1e3*tbd:.0f} ms --")
    print(fmt(bench(fns, reps=3)))
    if lmax == 128 and npaths == 2000:
        pr = cProfile.Profile(); pr.enable(); X.path_integral_operator(paths); pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
        print("=== cProfile construction ===")
        print("\n".join(s.getvalue().splitlines()[:26]))

print("\n=== geodesic ball averages (lmax 128, 100 centres) ===")
X = Sob2(128, 2.0, 0.1)
centres = X.random_points(100, rng=rng)
x = X.heat_measure(0.3).sample(rng=rng)
t0 = time.perf_counter(); Be = X.geodesic_ball_average_operator(centres, 0.3); te = time.perf_counter() - t0
t0 = time.perf_counter(); Bq = X.geodesic_ball_average_operator(centres, 0.3, count=100); tq = time.perf_counter() - t0
t0 = time.perf_counter(); Bd = X.geodesic_ball_average_operator(centres, 0.3, count=100, dense=True); td = time.perf_counter() - t0
w = rng.standard_normal(100)
print(f"  build: exact {1e3*te:.0f} ms, quadrature {1e3*tq:.0f} ms, quadrature dense {1e3*td:.0f} ms")
print(fmt(bench({
    "exact fwd": lambda: Be(x), "exact adj": lambda: Be.adjoint(w),
    "quadrature fwd": lambda: Bq(x), "quadrature adj": lambda: Bq.adjoint(w),
    "quadrature dense fwd": lambda: Bd(x), "quadrature dense adj": lambda: Bd.adjoint(w),
}, reps=3)))
print(f"  exact vs quadrature: max diff {np.max(np.abs(Be(x)-Bq(x))):.2e} of {np.max(np.abs(Be(x))):.3g}")
pr = cProfile.Profile(); pr.enable(); X.geodesic_ball_average_operator(centres, 0.3); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)
print("\n".join(s.getvalue().splitlines()[:20]))
