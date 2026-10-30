"""Every part in order: A to E on one problem in the disk, F in the ball, and
G with bent rays.

``python run_all.py`` for the full scale, ``--quick`` for the small one; the
ball is a toy at both, and G is a few dozen rays at its own two sizes. The
scale's fields are options: applied to the disk's preset for A to E, and the
same options given are applied to the ball's preset for F, its stations
excepted, and to the bent rays' for G.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time

import ball
import bayesian
import bent
import mixture
import point
import sets
import setup
from src.common import parse
from src.problem import FULL, QUICK, Scale


def given(scale: Scale, /) -> dict:
    """The fields in which a scale departs from its preset, the one its name
    names: the options given on the command line."""
    preset = QUICK if scale.name == QUICK.name else FULL
    return {
        field.name: getattr(scale, field.name)
        for field in dataclasses.fields(scale)
        if getattr(scale, field.name) != getattr(preset, field.name)
    }


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    changes = given(scale)
    ball_scale = dataclasses.replace(
        ball.BALL,
        **{
            name: value
            for name, value in changes.items()
            if name not in ball.BALL_FIXED
        },
    )
    bent_scale = dataclasses.replace(
        bent.BENT_QUICK if scale.name == QUICK.name else bent.BENT, **changes
    )
    problem = setup.run(scale, outdir=outdir)
    results = {"a": problem}
    results["b"] = point.run(scale, outdir=outdir, problem=problem)
    results["c"] = bayesian.run(scale, outdir=outdir, problem=problem)
    results["d"] = sets.run(scale, outdir=outdir, problem=problem)
    results["e"] = mixture.run(scale, outdir=outdir, problem=problem)
    results["f"] = ball.run(ball_scale, outdir=outdir)
    results["g"] = bent.run(bent_scale, outdir=outdir)
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
