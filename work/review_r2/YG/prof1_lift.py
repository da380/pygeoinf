"""Transform counts and timings for the formal-adjoint lift and the flexure operator, v1 vs v2."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import TransformCounter, bench, fmt
import numpy as np
import time

import pygeoinf2 as gi2
from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2, Lebesgue as Leb2
import pygeoinf.symmetric_space.sphere as sph1

rng = np.random.default_rng(1)


def smooth_field(space, rng, scale=0.3):
    mu = space.heat_measure(scale)
    return mu.sample(rng=rng)


def v2_setup(lmax, order=2.0, ls=0.2):
    X = Sob2(lmax, order, ls)
    L2 = X.with_order(0.0)
    f = L2.heat_measure(0.3).sample(rng=rng)
    D = L2.add(L2.project_function(lambda p: 2.0), L2.heat_measure(0.3).sample(rng=rng))
    rho = L2.add(L2.project_function(lambda p: 1.0), L2.scale(0.3, L2.heat_measure(0.3).sample(rng=rng)))
    x = X.heat_measure(0.3).sample(rng=rng)
    y = X.heat_measure(0.3).sample(rng=rng)
    return X, L2, f, D, rho, x, y


def v1_setup(lmax, order=2.0, ls=0.2):
    X = sph1.Sobolev(lmax, order, ls)
    L2 = sph1.Lebesgue(lmax)
    mu = L2.heat_kernel_gaussian_measure(0.3)
    f = mu.sample()
    D = L2.project_function(lambda p: 2.0)
    L2.axpy(1.0, mu.sample(), D)
    rho = L2.project_function(lambda p: 1.0)
    L2.axpy(0.3, mu.sample(), rho)
    x = X.heat_kernel_gaussian_measure(0.3).sample()
    y = X.heat_kernel_gaussian_measure(0.3).sample()
    return X, L2, f, D, rho, x, y


print("=== transform counts per application (lmax 32) ===")
X, L2, f, D, rho, x, y = v2_setup(32)
X1, L21, f1, D1, rho1, x1, y1 = v1_setup(32)

ops2 = {
    "v2 multiply (L2)": (lambda: L2.multiply(f, L2.grid_values(x) if False else x), None),
}
ops2 = {}
M2 = X.multiplication_operator(f)
F2 = X.flexural_operator(D, 0.25, rho)
I2 = X.order_inclusion_operator(X.with_order(1.0))
Lap2 = X.laplacian
FL2 = L2.flexural_operator(D, 0.25, rho)
G2 = lambda: X.gradient_dot_product(f, x)

M1 = X1.spatial_multiplication_operator(f1)
F1 = X1.flexural_operator(D1, 0.25, rho1)
I1 = X1.order_inclusion_operator(1.0)
FL1 = L21.flexural_operator(D1, 0.25, rho1)

cases = [
    ("v2 multiplication_operator fwd", lambda: M2(x)),
    ("v2 multiplication_operator adj", lambda: M2.adjoint(y)),
    ("v1 spatial_multiplication_operator fwd", lambda: M1(x1)),
    ("v1 spatial_multiplication_operator adj", lambda: M1.adjoint(y1)),
    ("v2 order_inclusion fwd", lambda: I2(x)),
    ("v2 order_inclusion adj", lambda: I2.adjoint(X.with_order(1.0).from_components(X.to_components(y)))),
    ("v1 order_inclusion fwd", lambda: I1(x1)),
    ("v1 order_inclusion adj", lambda: I1.adjoint(X1.with_order(1.0).from_components(X.to_components(y)))),
    ("v2 laplacian fwd", lambda: Lap2(x)),
    ("v2 gradient_dot_product", G2),
    ("v1 gradient_dot_product", lambda: X1.gradient_dot_product(f1, x1)),
    ("v2 flexural (Sobolev) fwd", lambda: F2(x)),
    ("v2 flexural (Sobolev) adj", lambda: F2.adjoint(y)),
    ("v1 flexural (Sobolev) fwd", lambda: F1(x1)),
    ("v1 flexural (Sobolev) adj", lambda: F1.adjoint(y1)),
    ("v2 flexural (L2) fwd", lambda: FL2(f)),
    ("v1 flexural (L2) fwd", lambda: FL1(f1)),
]
for name, fn in cases:
    with TransformCounter() as c:
        fn()
    print(f"  {name:45s} {c!r:16s} total={c.total}")

print("\n=== timings ===")
for lmax in (64, 128):
    X, L2, f, D, rho, x, y = v2_setup(lmax)
    X1, L21, f1, D1, rho1, x1, y1 = v1_setup(lmax)
    M2 = X.multiplication_operator(f); M1 = X1.spatial_multiplication_operator(f1)
    F2 = X.flexural_operator(D, 0.25, rho); F1 = X1.flexural_operator(D1, 0.25, rho1)
    fns = {
        "v2 to_components": lambda: X.to_components(x),
        "v2 from_components": lambda: X.from_components(X.to_components(x)),
        "v1 to_components": lambda: X1.to_components(x1),
        "v2 multiplication fwd": lambda: M2(x),
        "v1 multiplication fwd": lambda: M1(x1),
        "v2 multiplication adj": lambda: M2.adjoint(y),
        "v1 multiplication adj": lambda: M1.adjoint(y1),
        "v2 flexural fwd": lambda: F2(x),
        "v1 flexural fwd": lambda: F1(x1),
        "v2 flexural adj": lambda: F2.adjoint(y),
        "v1 flexural adj": lambda: F1.adjoint(y1),
        "v2 multiply": lambda: X.multiply(f, x),
        "v2 grid product only": lambda: X.grid_values(f) * X.grid_values(x),
    }
    print(f"\n-- lmax {lmax} (dim {X.dim}) --")
    print(fmt(bench(fns, reps=5)))

print("\n=== construction costs ===")
for lmax in (128, 256):
    t0 = time.perf_counter(); X = Sob2(lmax, 2.0, 0.2); t1 = time.perf_counter()
    L2 = X.with_order(0.0); t2 = time.perf_counter()
    _ = L2._quadrature; t3 = time.perf_counter()
    L2b = X.with_order(0.0); _ = L2b._quadrature; t4 = time.perf_counter()
    f = L2.heat_measure(0.3).sample(rng=rng)
    t5 = time.perf_counter(); M = X.multiplication_operator(f); t6 = time.perf_counter()
    print(f"  lmax {lmax}: Sphere() {1e3*(t1-t0):.1f} ms, with_order {1e3*(t2-t1):.1f} ms, "
          f"first _quadrature {1e3*(t3-t2):.1f} ms, second (cached) {1e3*(t4-t3):.1f} ms, "
          f"multiplication_operator() {1e3*(t6-t5):.1f} ms")
