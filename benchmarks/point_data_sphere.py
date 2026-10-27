"""Point data on the sphere: the closed form against the quadratic tier.

Al-Attar (2021), sections 2.5.2 and 3.3.2: a field in ``H²`` on the sphere
with length scale 0.25, seen through 250 point values, a norm-ball prior, a
norm-ball noise set, and the spherical-harmonic coefficients of degree at
most one (four of them) and of degrees one to three (fifteen) as the
properties: the quadratic tier, answered by the primal and the dual
algorithms, on the problem the closed form was first written for.

What is measured, in this order, at a noise radius of a tenth of the data's
size:

1. the support values of the primal with the dense spectral solver and
   with conjugate gradients, and of the dual, their agreement, brackets,
   evaluations, applications of the forward operator and wall time, and
   the S-procedure ellipsoid's weight;
2. the extents along the axes, which are inner bounds, against the
   brackets of the same directions;
3. the same sweep with one worker and several, which must agree.

With ``--limit`` the exact-data closed form is run as well, by both
solvers, and the quadratic tier's support values are followed towards it
as the noise ball shrinks. Exact data are numerically brittle and of no
practical interest, so this is a check rather than the measurement.

Run from the repository root::

    poetry run python benchmarks/point_data_sphere.py --lmax 64 --points 250

The slow tests run :func:`build` and :func:`measure` at a reduced size.
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
    table,
    timed,
)

from pygeoinf2.geometry import Ball, FeasiblePropertySet  # noqa: E402
from pygeoinf2.numerics.solvers import SpectralFamilySolver  # noqa: E402
from pygeoinf2.symmetric_space.sphere import Sobolev  # noqa: E402


@dataclass
class Problem:
    """The pieces of one instance, built once and probed many ways."""

    space: Sobolev
    forward: Any
    counter: Counter
    prior: Ball
    truth: Any
    exact_data: Any
    properties: dict[int, Any]

    def noisy_data(
        self, relative_radius: float, /, *, rng: np.random.Generator
    ) -> tuple[Any, Ball]:
        """Data inside a noise ball of the given radius relative to the data's size, and the ball."""
        data_space = self.forward.codomain
        radius = relative_radius * data_space.norm(self.exact_data)
        raw = data_space.random(rng=rng)
        error = data_space.scale(0.7 * radius / data_space.norm(raw), raw)
        return data_space.add(self.exact_data, error), Ball(data_space, radius=radius)


def build(
    *,
    lmax: int = 64,
    points: int = 250,
    order: float = 2.0,
    scale: float = 0.25,
    prior_radius: float = 1.0,
    rng: np.random.Generator,
) -> Problem:
    """The problem at a given truncation and number of points."""
    space = Sobolev(lmax, order, scale)
    forward, counter = counted(
        space.point_evaluation_operator(
            [space.random_point(rng=rng) for _ in range(points)]
        )
    )
    raw = space.heat_measure(0.3).sample(rng=rng)
    truth = space.scale(0.6 * prior_radius / space.norm(raw), raw)
    return Problem(
        space=space,
        forward=forward,
        counter=counter,
        prior=Ball(space, radius=prior_radius),
        truth=truth,
        exact_data=forward(truth),
        properties={
            4: space.coefficient_operator(lmax=1),
            15: space.coefficient_operator(lmin=1, lmax=3),
        },
    )


def measure(
    problem: Problem,
    *,
    relative_radius: float = 0.1,
    limit_radii: tuple[float, ...] = (),
    n_jobs: int = 4,
    rng: np.random.Generator,
    quiet: bool = False,
) -> dict[str, Any]:
    """Every table of the module docstring, returned by name and printed.

    Args:
        problem: what :func:`build` made.
        relative_radius: the noise ball's radius relative to the data's
            size, for the working measurement.
        limit_radii: radii for the optional noise-to-zero sweep against the
            closed form; none by default, the exact-data problem being
            brittle and of no practical interest at scale.
        n_jobs: workers for the parallel sweeps.
        rng: the generator for the noise.
        quiet: print nothing.
    """
    out: dict[str, Any] = {}
    counter = problem.counter

    def say(text: str = "") -> None:
        if not quiet:
            print(text)

    # 1. The quadratic tier at the working noise level.
    rows = []
    data, noise = problem.noisy_data(relative_radius, rng=rng)
    for dim, target in problem.properties.items():
        directions = axis_directions(target.codomain)
        reference = None
        for label, solver, via in (
            ("primal, spectral", SpectralFamilySolver(), "primal"),
            ("primal, CG", None, "primal"),
            ("dual, spectral", SpectralFamilySolver(), "dual"),
        ):
            feasible = FeasiblePropertySet(
                problem.forward, target, problem.prior, data, noise=noise, solver=solver
            )
            counter.reset()
            sweep, seconds = timed(
                lambda: feasible.support_sweep(directions, via=via, n_jobs=n_jobs)
            )
            values = sweep.values
            if reference is None:
                reference = values
                truth_inside = feasible.contains(target(problem.truth))
                ellipsoid, ellipsoid_seconds = timed(
                    lambda: feasible.outer_ellipsoid_result(criterion="volume")
                )
            gaps = np.array([r.gap for r in sweep.results])
            gaps = gaps[np.isfinite(gaps)]
            rows.append(
                [
                    dim,
                    label,
                    float(
                        np.max(
                            np.abs(values - reference)
                            / np.maximum(np.abs(reference), 1e-300)
                        )
                    ),
                    float(np.max(gaps)) if gaps.size else np.inf,
                    float(np.mean([r.evaluations for r in sweep.results])),
                    counter.total,
                    seconds,
                    sweep.converged,
                    truth_inside if label == "primal, spectral" else "",
                    f"{ellipsoid.weight:.3f} ({ellipsoid_seconds:.1f} s)"
                    if label == "primal, spectral"
                    else "",
                ]
            )
    out["working"] = rows
    say(
        f"## The quadratic tier at a noise radius of {relative_radius:g} of the data's size"
    )
    say(
        table(
            [
                "dim P",
                "algorithm",
                "max rel. diff to primal/spectral",
                "max bracket",
                "mean evaluations",
                "applications",
                "s",
                "converged",
                "truth inside",
                "S-procedure θ",
            ],
            rows,
        )
    )
    say()

    # 2. Extents against brackets, on the 4-dimensional property.
    target = problem.properties[4]
    feasible = FeasiblePropertySet(
        problem.forward,
        target,
        problem.prior,
        data,
        noise=noise,
        solver=SpectralFamilySolver(),
    )
    rows = []
    for index in range(target.codomain.dim):
        q = target.codomain.basis_vector(index)
        minus = target.codomain.negative(q)
        inner_hi, outer_hi = feasible.support_bracket(q)
        inner_lo, outer_lo = feasible.support_bracket(minus)
        (low, high), seconds = timed(lambda: feasible.extent(q))
        rows.append(
            [
                index,
                -outer_lo,
                -inner_lo,
                low,
                high,
                inner_hi,
                outer_hi,
                low >= -outer_lo - 1e-9 and high <= outer_hi + 1e-9,
                seconds,
            ]
        )
    out["extents"] = rows
    say(
        "## Extents (inner, by bisection on membership) against the brackets (dim P = 4)"
    )
    say(
        table(
            [
                "axis",
                "−outer(−q)",
                "−inner(−q)",
                "extent low",
                "extent high",
                "inner(q)",
                "outer(q)",
                "within",
                "s",
            ],
            rows,
        )
    )
    say()

    # 3. Workers.
    target = problem.properties[15]
    directions = axis_directions(target.codomain)
    rows = []
    for workers in (1, n_jobs):
        feasible = FeasiblePropertySet(
            problem.forward,
            target,
            problem.prior,
            data,
            noise=noise,
            solver=SpectralFamilySolver(),
        )
        values, seconds = timed(
            lambda: feasible.support_values(directions, n_jobs=workers)
        )
        rows.append([workers, seconds, values])
    agreement = float(np.max(np.abs(rows[0][2] - rows[1][2])))
    out["workers"] = [[row[0], row[1]] for row in rows] + [["max |Δ|", agreement]]
    say("## One worker against several (dim P = 15, primal, spectral)")
    say(table(["workers", "s"], [[row[0], row[1]] for row in rows]))
    say(f"max |Δ| between them: {agreement:.3g}")
    say()

    if not limit_radii:
        return out

    # 4. Optional: the exact-data closed form, and the quadratic tier's limit towards it.
    rows = []
    for dim, target in problem.properties.items():
        directions = axis_directions(target.codomain)
        counter.reset()
        default = FeasiblePropertySet(
            problem.forward, target, problem.prior, problem.exact_data
        )
        values, seconds = timed(lambda: default.support_values(directions))
        applications = counter.total
        counter.reset()
        spectral = FeasiblePropertySet(
            problem.forward,
            target,
            problem.prior,
            problem.exact_data,
            solver=SpectralFamilySolver(),
        )
        values_spectral, seconds_spectral = timed(
            lambda: spectral.support_values(directions)
        )
        rows.append(
            [
                dim,
                float(
                    np.max(
                        np.abs(values - values_spectral)
                        / np.maximum(np.abs(values), 1e-300)
                    )
                ),
                applications,
                seconds,
                counter.total,
                seconds_spectral,
                default.contains(target(problem.truth)),
            ]
        )
    out["closed_form"] = rows
    say("## The exact-data closed form: default solver against the spectral one")
    say(
        table(
            [
                "dim P",
                "max rel. diff",
                "CG applications",
                "CG s",
                "spectral applications",
                "spectral s",
                "truth inside",
            ],
            rows,
        )
    )
    say()

    rows = []
    out["limit"] = rows
    for dim, target in problem.properties.items():
        directions = axis_directions(target.codomain)
        exact = FeasiblePropertySet(
            problem.forward,
            target,
            problem.prior,
            problem.exact_data,
            solver=SpectralFamilySolver(),
        ).support_values(directions)
        for relative in limit_radii:
            limit_data, limit_noise = problem.noisy_data(relative, rng=rng)
            feasible = FeasiblePropertySet(
                problem.forward,
                target,
                problem.prior,
                limit_data,
                noise=limit_noise,
                solver=SpectralFamilySolver(),
            )
            counter.reset()
            sweep, seconds = timed(
                lambda: feasible.support_sweep(directions, n_jobs=n_jobs)
            )
            gaps = np.array([r.gap for r in sweep.results])
            gaps = gaps[np.isfinite(gaps)]
            rows.append(
                [
                    dim,
                    relative,
                    float(np.max(np.abs(sweep.values - exact))),
                    float(np.max(gaps)) if gaps.size else np.inf,
                    float(np.mean([r.evaluations for r in sweep.results])),
                    counter.total,
                    seconds,
                    sweep.converged,
                ]
            )
    say("## Noise → 0: the primal with the spectral solver against the closed form")
    say(
        table(
            [
                "dim P",
                "noise / |d|",
                "max |h − h_exact|",
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
    return out


def main() -> None:
    arguments = parser(__doc__.split("\n")[0])
    arguments.add_argument("--lmax", type=int, default=64)
    arguments.add_argument("--points", type=int, default=250)
    arguments.add_argument(
        "--noise",
        type=float,
        default=1e-2,
        help="the noise radius relative to the data",
    )
    arguments.add_argument(
        "--limit",
        action="store_true",
        help="also run the exact-data closed form and the noise-to-zero sweep",
    )
    settings = arguments.parse_args()
    rng = np.random.default_rng(settings.seed)
    problem = build(lmax=settings.lmax, points=settings.points, rng=rng)
    print(
        f"H^2 on the sphere at degree {settings.lmax} ({problem.space.dim} coefficients), "
        f"{settings.points} point values, prior radius {problem.prior.radius}"
    )
    print()
    measure(
        problem,
        relative_radius=settings.noise,
        limit_radii=(1e-1, 1e-2, 1e-3, 1e-4) if settings.limit else (),
        n_jobs=settings.n_jobs,
        rng=rng,
    )


if __name__ == "__main__":
    main()
