"""
Demo: correctness and timing of the finufft-backed matrix-free
`point_evaluation_operator` and `path_average_operator` on the torus and
the plane, compared against the dense-matrix implementations.

Checks
------
1. Forward values against a direct eigenfunction sum at 5 sample points
   (an absolute check, independent of the dense operator).
2. Forward and adjoint agreement with the dense operators (skipped with
   --skip-dense, for problem sizes where the matrix will not fit).
3. The adjoint identity <Au, y>_{R^n} = <u, A*y>_H for each matrix-free
   operator (needs no dense reference).

Usage
-----
    python nufft_demo.py --space torus --kmax 128 --n-points 2000
    python nufft_demo.py --space plane --kmax 64 --n-paths 100
    python nufft_demo.py --space torus --kmax 256 --n-points 100000 --skip-dense

Dependencies: pygeoinf with the finufft-backed operators.
"""

from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np


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
    parser.add_argument("--space", choices=("torus", "plane"), default="torus")
    parser.add_argument("--kmax", type=int, default=128)
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument(
        "--n-quad",
        type=int,
        default=None,
        help="quadrature points per path (default: heuristic)",
    )
    parser.add_argument("--order", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--skip-dense", action="store_true", help="skip building the dense operators"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.space == "torus":
        from pygeoinf.symmetric_space.torus import Sobolev
    else:
        from pygeoinf.symmetric_space.plane import Sobolev

    np.random.seed(args.seed)
    space = Sobolev(args.kmax, args.order, args.scale)
    mf_kwargs = dict(
        matrix_free=True, parallel=args.parallel, n_jobs=args.n_jobs, eps=args.eps
    )

    points = [space.random_point() for _ in range(args.n_points)]
    paths = [(space.random_point(), space.random_point()) for _ in range(args.n_paths)]
    u = space.random()
    y_pts = np.random.standard_normal(args.n_points)
    y_paths = np.random.standard_normal(args.n_paths)

    print(
        f"{args.space}: kmax={args.kmax}  dim={space.dim}  "
        f"n_points={args.n_points}  n_paths={args.n_paths}  eps={args.eps}"
    )
    print(
        f"dense point-eval matrix would be "
        f"{args.n_points * space.dim * 8 / 1e6:.0f} MB\n"
    )

    # --- construction -------------------------------------------------------
    t0 = time.perf_counter()
    A_nu = space.point_evaluation_operator(points, **mf_kwargs)
    P_nu = space.path_average_operator(paths, n_points=args.n_quad, **mf_kwargs)
    print(
        f"finufft operator construction (both): "
        f"{(time.perf_counter() - t0)*1e3:8.1f} ms"
    )

    if not args.skip_dense:
        t0 = time.perf_counter()
        A_mat = space.point_evaluation_operator(points)
        t_A = time.perf_counter() - t0
        t0 = time.perf_counter()
        P_mat = space.path_average_operator(paths, n_points=args.n_quad)
        print(
            f"dense construction: point eval {t_A:6.2f} s   "
            f"path avg {time.perf_counter() - t0:6.2f} s"
        )

    # --- absolute forward check at 5 sample points --------------------------
    c = space.to_components(u)
    direct = np.array(
        [np.dot(space.laplacian_eigenvectors_at_point(p), c) for p in points[:5]]
    )
    fwd = A_nu(u)
    err = np.abs(fwd[:5] - direct).max() / np.abs(direct).max()
    print(f"\nforward vs direct eigenfunction sum (5 pts): {err:.3e}")

    # --- agreement with dense operators -------------------------------------
    if not args.skip_dense:
        err = np.linalg.norm(fwd - A_mat(u)) / np.linalg.norm(fwd)
        print(f"point eval forward vs dense:  {err:.3e}")
        d = space.subtract(A_nu.adjoint(y_pts), A_mat.adjoint(y_pts))
        print(
            f"point eval adjoint vs dense:  "
            f"{space.norm(d) / space.norm(A_mat.adjoint(y_pts)):.3e}"
        )
        err = np.linalg.norm(P_nu(u) - P_mat(u)) / np.linalg.norm(P_mat(u))
        print(f"path avg forward vs dense:    {err:.3e}")
        d = space.subtract(P_nu.adjoint(y_paths), P_mat.adjoint(y_paths))
        print(
            f"path avg adjoint vs dense:    "
            f"{space.norm(d) / space.norm(P_mat.adjoint(y_paths)):.3e}"
        )

    # --- adjoint identities (dense reference not needed) ---------------------
    for name, op, yv in (("point eval", A_nu, y_pts), ("path avg", P_nu, y_paths)):
        lhs = float(np.dot(op(u), yv))
        rhs = space.inner_product(u, op.adjoint(yv))
        print(
            f"adjoint identity ({name:10s}): rel err "
            f"{abs(lhs - rhs) / abs(lhs):.3e}"
        )

    # --- timings -------------------------------------------------------------
    print(f"\napplication times over {args.repeats} repeats (best / mean):")
    rows = [
        ("finufft point eval forward", lambda: A_nu(u)),
        ("finufft point eval adjoint", lambda: A_nu.adjoint(y_pts)),
        ("finufft path avg   forward", lambda: P_nu(u)),
        ("finufft path avg   adjoint", lambda: P_nu.adjoint(y_paths)),
    ]
    if not args.skip_dense:
        rows += [
            ("dense   point eval forward", lambda: A_mat(u)),
            ("dense   point eval adjoint", lambda: A_mat.adjoint(y_pts)),
            ("dense   path avg   forward", lambda: P_mat(u)),
            ("dense   path avg   adjoint", lambda: P_mat.adjoint(y_paths)),
        ]
    for label, fn in rows:
        best, avg = timed(fn, args.repeats)
        print(f"  {label}: {best*1e3:9.1f} / {avg*1e3:9.1f} ms")


if __name__ == "__main__":
    main()
