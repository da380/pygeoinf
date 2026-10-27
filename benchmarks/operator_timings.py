"""Point evaluations and path averages: matrix-free against dense, timed.

The two acquisition operators every tomographic example is built from, on
the sphere, the torus and the plane, in their matrix-free form (a
non-uniform transform behind the point evaluations) and assembled as a
matrix. Each pair is checked to agree, forward and adjoint, and each
operator is checked against the adjoint identity on its own; then the
construction and the applications are timed. The matrix-free form is the
one that scales to real acquisition geometries; the dense one wins when
the operator is applied many times and the matrix fits, and this script
says where the crossover is on this machine.

Run from the repository root::

    poetry run python benchmarks/operator_timings.py --space sphere --lmax 128 --points 1000 --paths 200
    poetry run python benchmarks/operator_timings.py --space torus --shape 256 256 --points 20000
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import parser, table  # noqa: E402


def best_and_mean(function: Callable[[], Any], repeats: int, /) -> tuple[float, float]:
    """Wall times over several runs after one warm-up: the best and the mean."""
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times), mean(times)


def space_for(settings: Any) -> Any:
    if settings.space == "sphere":
        from pygeoinf2.symmetric_space.sphere import Sobolev

        return Sobolev(settings.lmax, settings.order, settings.scale)
    if settings.space == "torus":
        from pygeoinf2.symmetric_space.torus import Sobolev

        return Sobolev(settings.shape, settings.order, settings.scale)
    from pygeoinf2.symmetric_space.plane import Sobolev

    return Sobolev(
        settings.shape, settings.order, settings.scale, bounds=[(0.0, 1.0), (0.0, 1.0)]
    )


def main() -> None:
    arguments = parser(__doc__.split("\n")[0])
    arguments.add_argument(
        "--space", choices=("sphere", "torus", "plane"), default="sphere"
    )
    arguments.add_argument("--lmax", type=int, default=128, help="the sphere's degree")
    arguments.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[128, 128],
        help="the grid of a torus or plane",
    )
    arguments.add_argument("--order", type=float, default=2.0)
    arguments.add_argument("--scale", type=float, default=0.1)
    arguments.add_argument("--points", type=int, default=500)
    arguments.add_argument("--paths", type=int, default=100)
    arguments.add_argument(
        "--nodes", type=int, default=16, help="quadrature nodes per path"
    )
    arguments.add_argument("--repeats", type=int, default=5)
    arguments.add_argument(
        "--skip-dense", action="store_true", help="for sizes the matrix will not fit"
    )
    settings = arguments.parse_args()
    rng = np.random.default_rng(settings.seed)
    space = space_for(settings)
    points = [space.random_point(rng=rng) for _ in range(settings.points)]
    paths = [
        (space.random_point(rng=rng), space.random_point(rng=rng))
        for _ in range(settings.paths)
    ]
    print(
        f"{settings.space}: dimension {space.dim}, {settings.points} points, "
        f"{settings.paths} paths of {settings.nodes} nodes; a dense point matrix would be "
        f"{settings.points * space.dim * 8 / 1e6:.0f} MB"
    )
    print()

    builders = {
        "point evaluation": lambda dense: space.point_evaluation_operator(
            points, dense=dense
        ),
        "path average": lambda dense: space.path_average_operator(
            paths, count=settings.nodes, dense=dense
        ),
    }
    rows = []
    for name, build in builders.items():
        start = time.perf_counter()
        free = build(False)
        free_build = time.perf_counter() - start
        u = space.random(rng=rng)
        y = free.codomain.random(rng=rng)
        forms = {"matrix-free": (free, free_build)}
        if not settings.skip_dense:
            start = time.perf_counter()
            dense = build(True)
            forms["dense"] = (dense, time.perf_counter() - start)
        reference = None
        for form, (operator, build_time) in forms.items():
            forward = operator(u)
            adjoint = operator.adjoint(y)
            identity = abs(
                free.codomain.inner_product(forward, y)
                - space.inner_product(u, adjoint)
            ) / abs(free.codomain.inner_product(forward, y))
            if reference is None:
                reference = (forward, adjoint)
                agreement = 0.0
            else:
                agreement = max(
                    free.codomain.norm(free.codomain.subtract(forward, reference[0]))
                    / free.codomain.norm(reference[0]),
                    space.norm(space.subtract(adjoint, reference[1]))
                    / space.norm(reference[1]),
                )
            forward_best, forward_mean = best_and_mean(
                lambda: operator(u), settings.repeats
            )
            adjoint_best, adjoint_mean = best_and_mean(
                lambda: operator.adjoint(y), settings.repeats
            )
            rows.append(
                [
                    name,
                    form,
                    build_time,
                    forward_best * 1e3,
                    forward_mean * 1e3,
                    adjoint_best * 1e3,
                    adjoint_mean * 1e3,
                    identity,
                    agreement,
                ]
            )
    print(
        table(
            [
                "operator",
                "form",
                "build s",
                "forward best ms",
                "forward mean ms",
                "adjoint best ms",
                "adjoint mean ms",
                "adjoint identity",
                "rel. diff to matrix-free",
            ],
            rows,
        )
    )


if __name__ == "__main__":
    main()
