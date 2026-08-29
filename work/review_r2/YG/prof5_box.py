"""Periodic box NUFFT path: where the time goes at 1e4-1e5 points."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import bench, fmt
import numpy as np
import time
import cProfile, pstats, io

from pygeoinf2.symmetric_space.torus import Sobolev as TSob
from pygeoinf2.symmetric_space.plane import Sobolev as PSob
import pygeoinf.symmetric_space.torus as tor1

rng = np.random.default_rng(5)

for shape, n in (((256, 256), 10000), ((256, 256), 100000), ((512, 512), 100000)):
    X = TSob(shape, 2.0, 0.05, lengths=(1.0, 1.0))
    pts = X.random_points(n, rng=rng)
    x = X.heat_measure(0.05).sample(rng=rng)
    w = rng.standard_normal(n)
    t0 = time.perf_counter(); A = X.point_evaluation_operator(pts); tb = time.perf_counter() - t0
    layout = X._nufft_layout
    sizes, plus, minus, amplitude, fixed = layout
    comps = X.to_components(x)
    def assemble_spectrum():
        real = np.concatenate([comps[:fixed], comps[fixed::2]])
        imaginary = np.concatenate([np.zeros(fixed), comps[fixed + 1 :: 2]])
        weights = amplitude * (real + 1j * imaginary)
        spectrum = np.zeros(int(np.prod(sizes)), dtype=complex)
        np.add.at(spectrum, plus, 0.5 * weights)
        np.add.at(spectrum, minus, 0.5 * np.conj(weights))
        return spectrum
    def assemble_spectrum_assign():
        real = np.concatenate([comps[:fixed], comps[fixed::2]])
        imaginary = np.concatenate([np.zeros(fixed), comps[fixed + 1 :: 2]])
        weights = amplitude * (real + 1j * imaginary)
        spectrum = np.zeros(int(np.prod(sizes)), dtype=complex)
        spectrum[plus] = 0.5 * weights
        spectrum[minus] += 0.5 * np.conj(weights)  # overlap only where plus==minus (origin)
        return spectrum
    angles = X._angles(pts)
    import finufft
    spec = assemble_spectrum().reshape(sizes)
    fns = {
        "v2 fwd (operator)": lambda: A(x),
        "v2 adj (operator)": lambda: A.adjoint(w),
        "  to_components (rfftn)": lambda: X.to_components(x),
        "  spectrum assembly (np.add.at)": assemble_spectrum,
        "  spectrum assembly (assign)": assemble_spectrum_assign,
        "  _angles (point conversion)": lambda: X._angles(pts),
        "  nufft2d2 alone, nthreads=1": lambda: finufft.nufft2d2(*angles, spec, isign=+1, eps=1e-12, nthreads=1),
        "  nufft2d2 alone, nthreads=4": lambda: finufft.nufft2d2(*angles, spec, isign=+1, eps=1e-12, nthreads=4),
        "  nufft2d2 eps=1e-8": lambda: finufft.nufft2d2(*angles, spec, isign=+1, eps=1e-8, nthreads=1),
    }
    print(f"\n-- torus {shape} (dim {X.dim}), n={n}; build {1e3*tb:.1f} ms --")
    print(fmt(bench(fns, reps=3)))
    same = np.allclose(assemble_spectrum(), assemble_spectrum_assign())
    print(f"  assign == add.at: {same}")

# v1 torus point evaluation for comparison (256x256, 1e4)
try:
    X1 = tor1.Sobolev(256, 256, 2.0, 0.05)
    print("\nv1 torus Sobolev signature ok:", X1)
except Exception as e:
    print("\nv1 torus construct failed:", e)
    import inspect; print(inspect.signature(tor1.Sobolev.__init__))

# plane (Box) point evaluation: the _angles path with _to_enclosing per point
X = PSob((256, 256), 2.0, 0.05, bounds=((0, 1), (0, 1)))
n = 100000
pts = X.random_points(n, rng=rng)
x = X.heat_measure(0.05).sample(rng=rng)
A = X.point_evaluation_operator(pts)
w = rng.standard_normal(n)
print(f"\n-- plane/Box 256x256, n={n} --")
print(fmt(bench({
    "Box fwd": lambda: A(x), "Box adj": lambda: A.adjoint(w),
    "  Box._angles": lambda: X._angles(pts),
}, reps=3)))
pr = cProfile.Profile(); pr.enable(); A(x); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
print("\n".join(s.getvalue().splitlines()[:20]))
