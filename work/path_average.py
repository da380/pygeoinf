"""
Demo: correctness and timing of `Sobolev.path_average_operator` on the
sphere, comparing the dense-matrix and matrix-free implementations, with an
optional `--fast-ylm` flag that monkey-patches the accelerated
`laplacian_eigenvectors_at_point` (Fortran Legendre + vectorized gather)
into the library before anything is built. This lets you measure the effect
of the patch without editing pygeoinf.

Checks (as in the point-evaluation demo)
----------------------------------------
1. Forward agreement between the two operators on random fields.
2. Adjoint agreement between the two operators.
3. The adjoint identity <Au, y>_{R^n} = <u, A*y>_H for each operator.

Usage
-----
    python path_avg_demo.py --lmax 128 --n-paths 100
    python path_avg_demo.py --lmax 128 --n-paths 100 --fast-ylm
    python path_avg_demo.py --lmax 256 --n-paths 200 --n-quad 30 --fast-ylm

Dependencies: pygeoinf (and its dependencies pyshtools, joblib, numpy).
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from statistics import mean

import numpy as np


# --------------------------------------------------------------------------
# Fast Y_lm assembly, applied as a monkey-patch so the demo measures the
# library's own code paths with only this one routine replaced.
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _ylm_index_maps(lmax: int):
    """Gather indices mapping the packed Legendre array and cos/sin order
    tables onto the SHVector ordering (l^2+m for m>=0, l^2+l+|m| for m<0)."""
    dim = (lmax + 1) ** 2
    pidx = np.empty(dim, dtype=np.intp)
    midx = np.empty(dim, dtype=np.intp)
    is_cos = np.empty(dim, dtype=bool)
    for l in range(lmax + 1):
        base = l * (l + 1) // 2
        m = np.arange(l + 1)
        pidx[l * l : l * l + l + 1] = base + m
        midx[l * l : l * l + l + 1] = m
        is_cos[l * l : l * l + l + 1] = True
        pidx[l * l + l + 1 : (l + 1) ** 2] = base + m[1:]
        midx[l * l + l + 1 : (l + 1) ** 2] = m[1:]
        is_cos[l * l + l + 1 : (l + 1) ** 2] = False
    return pidx, midx, is_cos


def apply_fast_ylm_patch() -> None:
    import pyshtools as sh
    from pygeoinf.symmetric_space import sphere

    def fast_eigvecs(self, point):
        latitude, longitude = point
        pidx, midx, is_cos = _ylm_index_maps(self.lmax)
        z = np.sin(np.deg2rad(latitude))  # cos(colatitude)
        p = sh.legendre.PlmON(self.lmax, z, csphase=self.csphase)
        m_phi = np.arange(self.lmax + 1) * np.deg2rad(longitude)
        trig = np.where(is_cos, np.cos(m_phi)[midx], np.sin(m_phi)[midx])
        return p[pidx] * trig

    sphere.Lebesgue.laplacian_eigenvectors_at_point = fast_eigvecs


def timed(fn, repeats: int):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), mean(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=128)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument(
        "--n-quad",
        type=int,
        default=None,
        help="quadrature points per path (default: heuristic)",
    )
    parser.add_argument("--order", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--fast-ylm",
        action="store_true",
        help="monkey-patch the fast Y_lm assembly first",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.fast_ylm:
        apply_fast_ylm_patch()

    from pygeoinf.symmetric_space.sphere import Sobolev

    np.random.seed(args.seed)
    space = Sobolev(args.lmax, args.order, args.scale)
    paths = [(space.random_point(), space.random_point()) for _ in range(args.n_paths)]

    # report the quadrature load implied by the settings
    n_quad_per_path = []
    for p1, p2 in paths:
        if args.n_quad is None:
            arc = space.geodesic_distance(p1, p2)
            n_quad_per_path.append(max(2, int(np.ceil(2.0 * arc / space.scale))))
        else:
            n_quad_per_path.append(args.n_quad)
    n_total = int(np.sum(n_quad_per_path))

    print(
        f"lmax={args.lmax}  dim={space.dim}  n_paths={args.n_paths}  "
        f"quad pts/path ~{np.mean(n_quad_per_path):.0f}  total quad pts={n_total}"
    )
    print(
        f"fast_ylm={'ON' if args.fast_ylm else 'off'}   "
        f"dense matrix size: {args.n_paths * space.dim * 8 / 1e6:.1f} MB\n"
    )

    t0 = time.perf_counter()
    A_mat = space.path_average_operator(paths, n_points=args.n_quad)
    t_mat = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_free = space.path_average_operator(
        paths,
        n_points=args.n_quad,
        matrix_free=True,
        parallel=args.parallel,
        n_jobs=args.n_jobs,
    )
    t_free = time.perf_counter() - t0
    print(f"construction:  matrix {t_mat:8.3f} s   matrix-free {t_free:8.3f} s")

    u = space.random()
    y = np.random.standard_normal(args.n_paths)
    data_space = A_mat.codomain

    fwd_mat, fwd_free = A_mat(u), A_free(u)
    err_fwd = np.linalg.norm(fwd_mat - fwd_free) / np.linalg.norm(fwd_mat)
    print(f"\nforward  relative difference: {err_fwd:.3e}")

    adj_mat, adj_free = A_mat.adjoint(y), A_free.adjoint(y)
    err_adj = space.norm(space.subtract(adj_mat, adj_free)) / space.norm(adj_mat)
    print(f"adjoint  relative difference: {err_adj:.3e}")

    for label, op, adj in (
        ("matrix", A_mat, adj_mat),
        ("matrix-free", A_free, adj_free),
    ):
        lhs = data_space.inner_product(op(u), y)
        rhs = space.inner_product(u, adj)
        print(
            f"adjoint identity ({label:11s}): rel err "
            f"{abs(lhs - rhs) / abs(lhs):.3e}"
        )

    print(f"\napplication times over {args.repeats} repeats (best / mean):")
    for label, fn in (
        ("matrix       forward", lambda: A_mat(u)),
        ("matrix-free  forward", lambda: A_free(u)),
        ("matrix       adjoint", lambda: A_mat.adjoint(y)),
        ("matrix-free  adjoint", lambda: A_free.adjoint(y)),
    ):
        best, avg = timed(fn, args.repeats)
        print(f"  {label}: {best*1e3:9.1f} / {avg*1e3:9.1f} ms")


if __name__ == "__main__":
    main()
