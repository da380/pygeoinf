"""Part C: a deliberately wide bound.

A norm bound on the departure from PREM that says almost nothing: a ball in
the model space's norm whose radius is the 95% radius of the Sobolev
measure with a pointwise standard deviation of 3 g/cm^3, and that measure as
the Gaussian. The same three answers as in part B, in the same order. Two
data cut a slice through the space, and this part shows what each answer
makes of the slice when the prior hardly narrows it.
"""

from __future__ import annotations

import pathlib

from pygeoinf2 import LinearGaussianInversion

from answers import three_answers
from common import output_directory, parse, part_rng
from problem import FULL, QUICK, TIGHT_STD, WIDE_STD, Problem, Scale, build


def wide_answers(problem: Problem, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """The three answers under the wide bound, and the evidence against the
    tight prior's."""
    print(f"\n=== C. A deliberately wide bound ({problem.scale.name} scale) ===")
    print(f"prior: about PREM, pointwise std {WIDE_STD:g} g/cm^3 in every layer")
    result = three_answers(
        problem,
        problem.prior(WIDE_STD),
        part="C",
        prior_name="wide",
        rng=part_rng(seed, "c"),
        outdir=outdir,
    )
    tight = LinearGaussianInversion(problem.problem, problem.prior(TIGHT_STD))
    result["log_bayes_factor"] = (
        tight.log_evidence(problem.data) - result["measure"]["log_evidence"]
    )
    print(
        f"the data prefer the tight prior to the wide one by "
        f"{result['log_bayes_factor']:.2f} nats"
    )
    return result


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return wide_answers(problem, seed=seed, outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
