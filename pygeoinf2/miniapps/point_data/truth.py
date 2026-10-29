"""Part A: the setup, and a map of the truth with the observation points."""

from __future__ import annotations

import pathlib

from pygeoinf2 import plotting

from common import map_panels, output_directory, parse, save
from problem import CAP_RADIUS, FULL, QUICK, REGIONS, Problem, Scale, build


def truth_map(problem: Problem, /, *, outdir: pathlib.Path) -> pathlib.Path:
    """The truth, the observation points and the three caps asked about."""
    space = problem.space
    fig, axes = map_panels(space, 1, 1)
    ax = axes[0, 0]
    plotting.plot(
        space,
        problem.truth,
        ax=ax,
        symmetric=True,
        coasts=True,
        gridlines=False,
        colorbar_label="truth",
    )
    many = len(problem.points) > 2000
    plotting.plot_points(
        space, problem.points, ax=ax, marker=".", size=1.0 if many else 8.0
    )
    plotting.plot_balls(
        space, [c for _, c in REGIONS], CAP_RADIUS, ax=ax, color="black", linewidth=1.2
    )
    path = outdir / "truth.png"
    save(
        fig,
        path,
        title=f"Point data, part A: the truth, the {len(problem.points)} ocean "
        "observation points and the three caps asked about",
    )
    return path


def run(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> Problem:
    problem = build(scale=scale, seed=seed)
    truth_map(problem, outdir=outdir)
    return problem


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
