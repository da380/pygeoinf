"""Phase-delay tomography with nested acquisition: cost against the data count.

Mag et al. (submitted), section 3.1 and figure 9: path averages of a phase
velocity perturbation on the sphere along great circles between the Global
Seismograph Network's stations and a USGS catalogue's epicentres, with a
norm-ball prior and a norm-ball noise set, and four cap averages as the
properties. The acquisition is nested: each data set contains the smaller
ones, so the sets shrink as the data grow and the cost of each support
value is measured against the number of data alone.

The sizes follow the tomography tutorial of the first release: ``H²`` with
length scale 0.2, and a noise ball of a third of the data's size, which is
the regime a real acquisition is in. A noise ball of one per cent is the
tight-noise regime, where the damped family's members approach the
undamped ``A C A*`` at every multiplier the walk visits and no surrogate
of lower rank can precondition them; ``--noise`` reaches it, and the cost
then rises by an order of magnitude on every iterative route.

What is measured, at each data count:

1. the support values in twelve directions by the primal with the dense
   spectral solver (up to a data count it is still affordable at), with
   unpreconditioned conjugate gradients, with conjugate gradients deflated
   by the leading modes of ``A C A*``, and with conjugate gradients
   preconditioned by the Woodbury inverse of a surrogate on a coarser
   sphere; applications of the forward operator and wall time for each,
   and their agreement;
2. the same sweep with one worker and several, which must agree.

Run from the repository root::

    poetry run python benchmarks/phase_delay.py --lmax 64 --counts 100 300 1000 3000 9000

The catalogues are read from the cache; ``pygeoinf2.datasets`` downloads
larger ones than the shipped tables when asked. The slow tests run
:func:`build` and :func:`measure` on synthetic acquisition at a small size.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    Counter,
    axis_directions,
    counted,
    parser,
    planar_directions,
    table,
    timed,
)

from pygeoinf2.algebra.operators import LinearOperator  # noqa: E402
from pygeoinf2.geometry import Ball, FeasiblePropertySet  # noqa: E402
from pygeoinf2.numerics.preconditioners import (  # noqa: E402
    SpectralPreconditioner,
    WoodburyPreconditioner,
)
from pygeoinf2.numerics.solvers import (  # noqa: E402
    CGSolver,
    CholeskySolver,
    SpectralFamilySolver,
)
from pygeoinf2.symmetric_space.sphere import Sobolev  # noqa: E402
from pygeoinf2.traits import Traits  # noqa: E402

CAPS = [
    np.array([lat, lon])
    for lat, lon in [(-60.0, 0.0), (5.0, 143.0), (0.0, -120.0), (46.0, 104.0)]
]
CAP_RADIUS = 0.15
NODES_PER_PATH = 12


@dataclass
class Acquisition:
    """Sources and receivers in a fixed random order, so prefixes nest."""

    sources: list[np.ndarray]
    receivers: list[np.ndarray]

    def paths(
        self, space: Sobolev, count: int, /, *, minimum_separation: float = 0.4
    ) -> list[tuple[Any, Any]]:
        """About ``count`` paths from the first sources and receivers, nested in the count."""
        receivers = min(len(self.receivers), max(4, int(np.sqrt(count / 2.0))))
        sources = min(len(self.sources), int(np.ceil(count / receivers)))
        pairs = [
            (source, receiver)
            for source in self.sources[:sources]
            for receiver in self.receivers[:receivers]
            if space.geodesic_distance(source, receiver) > minimum_separation
        ]
        return pairs


def acquisition(
    space: Sobolev, /, *, synthetic: bool = False, rng: np.random.Generator
) -> Acquisition:
    """The catalogues, shuffled once; or random points when asked."""
    if synthetic:
        return Acquisition(
            [space.random_point(rng=rng) for _ in range(200)],
            [space.random_point(rng=rng) for _ in range(100)],
        )
    sources = space.earthquakes(minimum_magnitude=5.0)
    receivers = space.stations()
    rng.shuffle(sources)
    rng.shuffle(receivers)
    return Acquisition(sources, receivers)


@dataclass
class Problem:
    """One data count of the nested acquisition."""

    count: int
    space: Sobolev
    forward: Any
    raw_forward: Any
    counter: Counter
    paths: list
    prior: Ball
    noise: Ball
    target: Any
    data: Any
    truth: Any


def build(
    space: Sobolev,
    acquired: Acquisition,
    count: int,
    /,
    *,
    truth: Any,
    prior_radius: float,
    noise_relative: float = 0.3,
    rng: np.random.Generator,
) -> Problem:
    """The problem at one data count."""
    paths = acquired.paths(space, count)
    raw = space.path_average_operator(paths, count=NODES_PER_PATH)
    forward, counter = counted(raw)
    data_space = forward.codomain
    clean = forward(truth)
    radius = noise_relative * data_space.norm(clean)
    error = data_space.random(rng=rng)
    error = data_space.scale(0.7 * radius / data_space.norm(error), error)
    return Problem(
        count=data_space.dim,
        space=space,
        forward=forward,
        raw_forward=raw,
        counter=counter,
        paths=paths,
        prior=Ball(space, radius=prior_radius),
        noise=Ball(data_space, radius=radius),
        target=space.geodesic_ball_average_operator(CAPS, CAP_RADIUS, dense=True),
        data=data_space.add(clean, error),
        truth=truth,
    )


def truth_for(
    space: Sobolev, prior_radius: float, /, *, rng: np.random.Generator
) -> Any:
    """A smooth field inside the prior ball."""
    raw = space.heat_measure(0.1).sample(rng=rng)
    return space.scale(0.6 * prior_radius / space.norm(raw), raw)


def surrogate_preconditioner(
    problem: Problem, /, *, coarse_degree: int
) -> WoodburyPreconditioner:
    """The Woodbury inverse of the same problem on a coarser sphere, sharing the data space."""
    coarse = problem.space.with_degree(coarse_degree)
    coarse_forward = coarse.path_average_operator(
        problem.paths, count=NODES_PER_PATH, dense=True
    )
    traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    prior = (LinearOperator.identity(coarse) * problem.prior.radius**2).with_traits(
        traits
    )
    prior_inverse = (
        LinearOperator.identity(coarse) * (1.0 / problem.prior.radius**2)
    ).with_traits(traits)
    data_space = problem.forward.codomain
    noise = (LinearOperator.identity(data_space) * problem.noise.radius**2).with_traits(
        traits
    )
    noise_inverse = (
        LinearOperator.identity(data_space) * (1.0 / problem.noise.radius**2)
    ).with_traits(traits)
    return WoodburyPreconditioner(
        coarse_forward,
        prior,
        noise,
        solver=CholeskySolver(),
        prior_inverse=prior_inverse,
        noise_inverse=noise_inverse,
    )


def directions_for(space: Any, /) -> list[Any]:
    """The eight axis directions and four in the plane of the first two caps."""
    return axis_directions(space) + planar_directions(space, 0, 1, 4)


def measure(
    problems: list[Problem],
    *,
    spectral_limit: int = 3000,
    deflation_rank: int = 20,
    coarse_degree: int | None = None,
    n_jobs: int = 4,
    quiet: bool = False,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Every table of the module docstring, returned by name and printed."""
    out: dict[str, Any] = {}
    if coarse_degree is None:
        coarse_degree = max(problems[0].space.lmax // 4, 4)

    def say(text: str = "") -> None:
        if not quiet:
            print(text)

    rows = []
    for problem in problems:
        directions = directions_for(problem.target.codomain)
        runs: list[tuple[str, Any]] = []
        if problem.count <= spectral_limit:
            runs.append(("spectral", SpectralFamilySolver()))
        # A generous cap, so that a slow solve is measured rather than refused.
        cg = CGSolver(rtol=1e-10, max_iterations=max(20 * problem.count, 400))
        runs.append(("CG", cg))
        runs.append(
            (
                f"CG, deflated rank {deflation_rank}",
                cg.with_preconditioner(
                    SpectralPreconditioner(rank=deflation_rank, rng=rng)
                ),
            )
        )
        runs.append(
            (
                f"CG, Woodbury surrogate at degree {coarse_degree}",
                cg.with_preconditioner(
                    surrogate_preconditioner(problem, coarse_degree=coarse_degree)
                ),
            )
        )
        reference = None
        for label, solver in runs:
            feasible = FeasiblePropertySet(
                problem.forward,
                problem.target,
                problem.prior,
                problem.data,
                noise=problem.noise,
                solver=solver,
            )
            problem.counter.reset()
            try:
                sweep, seconds = timed(
                    lambda: feasible.support_sweep(directions, n_jobs=n_jobs)
                )
            except Exception as error:  # noqa: BLE001 - recorded, not hidden
                rows.append(
                    [
                        problem.count,
                        label,
                        f"failed: {type(error).__name__}: {str(error)[:90]}",
                        "",
                        "",
                        problem.counter.total,
                        "",
                        False,
                    ]
                )
                continue
            values = sweep.values
            if reference is None:
                reference = values
            gaps = np.array([r.gap for r in sweep.results])
            gaps = gaps[np.isfinite(gaps)]
            rows.append(
                [
                    problem.count,
                    label,
                    float(
                        np.max(
                            np.abs(values - reference)
                            / np.maximum(np.abs(reference), 1e-300)
                        )
                    ),
                    float(np.max(gaps)) if gaps.size else np.inf,
                    float(np.mean([r.evaluations for r in sweep.results])),
                    problem.counter.total,
                    seconds,
                    sweep.converged,
                ]
            )
        say(f"... {problem.count} data done")
    out["cost"] = rows
    say()
    say("## Each data count: the solvers, their agreement, applications and wall time")
    say(
        table(
            [
                "N_d",
                "solver",
                "max rel. diff to first",
                "max bracket",
                "mean evaluations",
                "applications",
                "s",
                "converged",
            ],
            rows,
        )
    )
    say()

    # 2. Workers, on the largest count the spectral solver still serves.
    problem = max(
        (p for p in problems if p.count <= spectral_limit),
        key=lambda p: p.count,
        default=problems[0],
    )
    directions = directions_for(problem.target.codomain)
    rows = []
    results = []
    for workers in (1, n_jobs):
        feasible = FeasiblePropertySet(
            problem.forward,
            problem.target,
            problem.prior,
            problem.data,
            noise=problem.noise,
            solver=SpectralFamilySolver() if problem.count <= spectral_limit else None,
        )
        values, seconds = timed(
            lambda: feasible.support_values(directions, n_jobs=workers)
        )
        results.append(values)
        rows.append([problem.count, workers, seconds])
    agreement = float(np.max(np.abs(results[0] - results[1])))
    out["workers"] = rows + [["max |Δ|", agreement]]
    say("## One worker against several")
    say(table(["N_d", "workers", "s"], rows))
    say(f"max |Δ| between them: {agreement:.3g}")
    say()
    return out


def main() -> None:
    arguments = parser(__doc__.split("\n")[0])
    arguments.add_argument("--lmax", type=int, default=64)
    arguments.add_argument(
        "--counts", type=int, nargs="+", default=[100, 300, 1000, 3000, 9000]
    )
    arguments.add_argument(
        "--noise", type=float, default=0.3, help="the noise radius relative to the data"
    )
    arguments.add_argument("--spectral-limit", type=int, default=3000)
    arguments.add_argument("--rank", type=int, default=20, help="deflation rank")
    arguments.add_argument(
        "--coarse",
        type=int,
        default=None,
        help="the surrogate's degree; a quarter of lmax by default",
    )
    arguments.add_argument(
        "--synthetic",
        action="store_true",
        help="random points instead of the catalogues",
    )
    settings = arguments.parse_args()
    rng = np.random.default_rng(settings.seed)
    space = Sobolev(settings.lmax, 2.0, 0.2)
    acquired = acquisition(space, synthetic=settings.synthetic, rng=rng)
    truth = truth_for(space, 3.0, rng=rng)
    problems = [
        build(
            space,
            acquired,
            count,
            truth=truth,
            prior_radius=3.0,
            noise_relative=settings.noise,
            rng=rng,
        )
        for count in settings.counts
    ]
    print(
        f"H^1.5 on the sphere at degree {settings.lmax} ({space.dim} coefficients); "
        f"{len(acquired.sources)} sources and {len(acquired.receivers)} receivers available; "
        f"data counts {[p.count for p in problems]}"
    )
    print()
    measure(
        problems,
        spectral_limit=settings.spectral_limit,
        deflation_rank=settings.rank,
        coarse_degree=settings.coarse,
        n_jobs=settings.n_jobs,
        rng=rng,
    )


if __name__ == "__main__":
    main()
