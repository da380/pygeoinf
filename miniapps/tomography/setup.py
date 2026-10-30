"""Part A: the setup, with the rays over the velocity and the truth under them."""

from __future__ import annotations

import pathlib

import numpy as np

from pygeoinf import plotting

from src.common import (
    draw,
    outline_boxes,
    panels,
    parse,
    save,
    stations,
)
from src.problem import (
    BOXES,
    CORE_VELOCITY,
    FULL,
    MANTLE,
    QUICK,
    Problem,
    Scale,
    build,
)


def setup_figure(problem: Problem, /, *, outdir: pathlib.Path) -> pathlib.Path:
    """The reference velocity with the rays, and the truth with the nodes."""
    space, scale = problem.space, problem.scale
    velocity = space.project_function(
        [lambda point: CORE_VELOCITY, lambda point: float(MANTLE.velocity(point[0]))]
    )
    fig, axes = panels(space, 1, 2)
    left, right = axes[0]
    draw(space, velocity, left, title="", cmap="viridis_r", label="velocity")
    plotting.plot_paths(
        space,
        [ray.nodes for ray in problem.rays],
        ax=left,
        color="white",
        linewidth=0.5,
        alpha=0.5,
    )
    stations(left, scale.sources, scale.receivers, radius=space.layers[-1].radius)
    left.set_title(f"{len(problem.rays)} rays over the reference velocity", fontsize=10)

    limit = float(np.abs(space.layers[-1].interior_values(problem.truth[-1])).max())
    draw(
        space,
        problem.truth,
        right,
        title="",
        limit=limit,
        label="slowness perturbation (%)",
    )
    outline_boxes(right, BOXES, color="black", linewidth=1.2)
    right.set_title("The truth and the two boxes", fontsize=10)
    path = outdir / "setup.png"
    save(
        fig,
        path,
        title="Tomography, part A: the rays over the reference velocity, and the "
        "truth with the two boxes",
    )
    return path


def run(scale: Scale, /, *, outdir: pathlib.Path) -> Problem:
    problem = build(scale=scale)
    setup_figure(problem, outdir=outdir)
    return problem


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
