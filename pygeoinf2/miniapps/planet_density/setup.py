"""Part A: the setup, with PREM, the reference, the truth and the layers.

The reference is PREM in the space of PREM's own layers; the truth is the
reference plus a seeded draw from the tight prior about it, and the data are
the truth's mass and moment of inertia with errors drawn from the error
measure. The figure shows how closely the space holds PREM, layer by layer,
and how far the truth departs from it.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from src.common import (
    draw_prem,
    draw_profile,
    finish_profile_axes,
    layer_radii,
    parse,
    save,
    title,
)
from src.problem import FULL, QUICK, Problem, Scale, build
from src.reference import R_SURFACE, layering, prem_layer_functions

# The upper mantle and the crust, drawn again on their own.
ZOOM = (5400.0, R_SURFACE)


def setup_figure(problem: Problem, /, *, outdir: pathlib.Path) -> pathlib.Path:
    """PREM, the reference, the truth, the layers, and the differences, over
    the whole radius and over the upper mantle and the crust."""
    space = problem.space
    _, boundaries = layering()
    functions = prem_layer_functions()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 6.5),
        sharex="col",
        layout="constrained",
        width_ratios=[1.6, 1.0],
        height_ratios=[2.0, 1.0],
    )
    for column in range(2):
        top, bottom = axes[:, column]
        for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            for ax in (top, bottom):
                ax.axvspan(
                    lower, upper, color="0.5", alpha=0.1 * (index % 2), linewidth=0.0
                )
        draw_prem(top, color="black", lw=1.0, label="PREM")
        draw_profile(
            space, problem.reference, top, color="C3", lw=1.4, label="reference"
        )
        draw_profile(space, problem.truth, top, color="C0", lw=1.0, label="truth")
        finish_profile_axes(top, legend=False)
        top.set_xlabel("")
        for index, radii in enumerate(layer_radii(space)):
            layer = space.layers[index]
            reference = layer.evaluate(problem.reference[index], radii)
            truth = layer.evaluate(problem.truth[index], radii)
            prem = np.array([functions[index](r) for r in radii])
            first = index == 0
            bottom.plot(
                radii,
                reference - prem,
                color="C3",
                lw=1.0,
                label="reference less PREM" if first else None,
            )
            bottom.plot(
                radii,
                truth - reference,
                color="C0",
                lw=1.0,
                label="truth less the reference" if first else None,
            )
        bottom.axhline(0.0, color="black", lw=0.6)
        finish_profile_axes(bottom, legend=False)
        bottom.set_ylabel("difference (g/cm³)")
    axes[0, 0].set_ylim(0.0, 14.5)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1, 0].legend(fontsize=7, frameon=False, loc="lower left")
    axes[0, 0].set_title(
        "the whole radius; PREM's layers shaded alternately", fontsize=10
    )
    axes[0, 1].set_xlim(*ZOOM)
    axes[0, 1].set_ylim(0.8, 4.8)
    axes[0, 1].set_title("the upper mantle and the crust", fontsize=10)
    fig.suptitle(title("A", "PREM, the reference in the space, and the truth"))
    path = outdir / "setup.png"
    save(fig, path)
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
