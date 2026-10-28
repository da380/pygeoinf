"""Part E: the same planet in a ball.

The questions of B, C and D asked again in three dimensions, at a toy size:
the model a layered ball of degree 12, the rays the quick network's in the
equatorial plane, traced by the same radial tracer with the angle read as a
longitude, and the figures sections through that plane. Nothing in the parts
changes; the space does.
"""

from __future__ import annotations

import pathlib
import time

from bayesian import bayesian_answer
from common import output_directory, parse, part_rng
from point import point_answers
from problem import BALL, Scale, build
from sets import set_answer
from setup import setup_figure


def ball_answers(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """Every part on the ball, with its figures."""
    started = time.perf_counter()
    print("\n=== E. The same planet in a ball ===")
    problem = build(scale=scale, seed=seed)
    setup_figure(problem, outdir=outdir)
    results = {"a": problem}
    results["b"] = bayesian_answer(problem, rng=part_rng(seed, "e"), outdir=outdir)
    results["c"] = point_answers(problem, rng=part_rng(seed, "e"), outdir=outdir)
    results["d"] = set_answer(
        problem,
        rng=part_rng(seed, "e"),
        outdir=outdir,
        bayesian=results["b"],
        point=results["c"],
    )
    print(f"E took {time.perf_counter() - started:.1f} s")
    return results


def run(scale: Scale = BALL, /, *, seed: int, outdir: pathlib.Path) -> dict:
    return ball_answers(scale, seed=seed, outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    # The ball is a toy at either scale; --quick only moves the figures.
    run(BALL, seed=args.seed, outdir=output_directory(args.quick, args.outdir))


if __name__ == "__main__":
    main()
