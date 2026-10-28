"""Every part in order: A to D on one problem in the disk, E in the ball, and
F with bent rays.

``python run_all.py`` for the full scale, ``--quick`` for the small one; the
ball is a toy at both, and F is a few dozen rays at its own two sizes.
"""

from __future__ import annotations

import pathlib
import time

import ball
import bayesian
import bent
import point
import sets
import setup
from common import output_directory, parse
from problem import BALL, FULL, QUICK, Scale


def run_all(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    problem = setup.run(scale, seed=seed, outdir=outdir)
    results = {"a": problem}
    results["b"] = bayesian.run(scale, seed=seed, outdir=outdir, problem=problem)
    results["c"] = point.run(scale, seed=seed, outdir=outdir, problem=problem)
    results["d"] = sets.run(
        scale,
        seed=seed,
        outdir=outdir,
        problem=problem,
        bayesian=results["b"],
        point=results["c"],
    )
    results["e"] = ball.run(BALL, seed=seed, outdir=outdir)
    results["f"] = bent.run(
        bent.BENT_QUICK if scale is QUICK else bent.BENT, seed=seed, outdir=outdir
    )
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__)
    run_all(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
