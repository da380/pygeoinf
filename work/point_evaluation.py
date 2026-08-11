"""
Demo: correctness and timing of `Sobolev.point_evaluation_operator` on the
sphere, comparing the dense-matrix and matrix-free implementations.

Checks
------
1. Forward agreement between the two operators on random fields.
2. Adjoint agreement between the two operators.
3. The adjoint identity <Au, y>_{R^n} = <u, A*y>_H for each operator
   separately (validates each implementation internally, not just mutually).

Timings
-------
Construction, forward application, and adjoint application for each variant,
plus the memory footprint of the dense matrix. Applications are timed over
`--repeats` runs after one warm-up; best and mean are reported (best is the
more stable statistic on a shared machine).

Usage
-----
    python point_eval_demo.py --lmax 128 --n-points 500 --repeats 5
    python point_eval_demo.py --lmax 256 --n-points 1000 --parallel

Dependencies: pygeoinf (and its dependencies pyshtools, joblib, numpy).
"""

from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np

from pygeoinf.symmetric_space.sphere import Sobolev


def timed(fn, repeats: int) -> tuple[float, float, object]:
    """Returns (best, mean) times in seconds over `repeats` runs, one warm-up."""
    result = fn()  # warm-up, also the returned value for reuse
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), mean(times), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=128)
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--order", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--parallel", action="store_true", help="parallel matrix-free applications"
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)  # random_point() uses the global RNG

    space = Sobolev(args.lmax, args.order, args.scale)
    points = [space.random_point() for _ in range(args.n_points)]
    dim = space.dim

    print(
        f"lmax={args.lmax}  dim={dim}  n_points={args.n_points}  "
        f"order={args.order}  scale={args.scale}"
    )
    print(f"dense matrix size: {args.n_points * dim * 8 / 1e6:.1f} MB\n")

    # --- construction -------------------------------------------------------
    t0 = time.perf_counter()
    A_mat = space.point_evaluation_operator(points)
    t_mat_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_free = space.point_evaluation_operator(
        points, matrix_free=True, parallel=args.parallel, n_jobs=args.n_jobs
    )
    t_free_build = time.perf_counter() - t0

    print(
        f"construction:  matrix {t_mat_build:8.3f} s   "
        f"matrix-free {t_free_build:8.3f} s"
    )

    # --- test data ----------------------------------------------------------
    u = space.random()
    y = np.random.standard_normal(args.n_points)
    data_space = A_mat.codomain

    # --- forward agreement --------------------------------------------------
    fwd_mat = A_mat(u)
    fwd_free = A_free(u)
    ref = np.linalg.norm(fwd_mat)
    err_fwd = np.linalg.norm(fwd_mat - fwd_free) / ref
    print(f"\nforward  relative difference: {err_fwd:.3e}")

    # --- adjoint agreement --------------------------------------------------
    adj_mat = A_mat.adjoint(y)
    adj_free = A_free.adjoint(y)
    diff = space.subtract(adj_mat, adj_free)
    err_adj = space.norm(diff) / space.norm(adj_mat)
    print(f"adjoint  relative difference: {err_adj:.3e}")

    # --- adjoint identity for each operator ---------------------------------
    for label, op, adj in (
        ("matrix", A_mat, adj_mat),
        ("matrix-free", A_free, adj_free),
    ):
        lhs = data_space.inner_product(op(u), y)
        rhs = space.inner_product(u, adj)
        rel = abs(lhs - rhs) / abs(lhs)
        print(
            f"adjoint identity ({label:11s}): "
            f"<Au,y>={lhs:+.6e}  <u,A*y>={rhs:+.6e}  rel err {rel:.3e}"
        )

    # --- timings ------------------------------------------------------------
    print(f"\napplication times over {args.repeats} repeats (best / mean):")
    for label, fn in (
        ("matrix       forward", lambda: A_mat(u)),
        ("matrix-free  forward", lambda: A_free(u)),
        ("matrix       adjoint", lambda: A_mat.adjoint(y)),
        ("matrix-free  adjoint", lambda: A_free.adjoint(y)),
    ):
        best, avg, _ = timed(fn, args.repeats)
        print(f"  {label}: {best*1e3:9.1f} / {avg*1e3:9.1f} ms")


if __name__ == "__main__":
    main()
