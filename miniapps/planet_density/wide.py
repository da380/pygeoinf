"""Part C: a deliberately wide bound.

A bound on the departure from PREM that says almost nothing: the Sobolev
measure with a pointwise standard deviation of 3 g/cm^3 as the Gaussian, and
its 95% ball as the set. The same three answers as in part B, in the
same order. Two
data cut a slice through the space, and this part shows what each answer
makes of the slice when the prior hardly narrows it.
"""

from __future__ import annotations

import pathlib

from pygeoinf import LinearGaussianInversion

from src.answers import three_answers
from src.common import parse, part_rng
from src.problem import FULL, QUICK, Problem, Scale, build


def wide_answers(problem: Problem, /, *, outdir: pathlib.Path) -> dict:
    """The three answers under the wide bound, and the evidence against the
    tight prior's."""
    scale = problem.scale
    print(f"\n=== C. A deliberately wide bound ({scale.name} scale) ===")
    print(f"prior: about PREM, pointwise std {scale.wide_std:g} g/cm^3 in every layer")
    result = three_answers(
        problem,
        problem.prior(scale.wide_std),
        part="C",
        prior_name="wide",
        rng=part_rng(scale.seed, "c"),
        outdir=outdir,
    )
    tight = LinearGaussianInversion(problem.problem, problem.prior(scale.tight_std))
    result["log_bayes_factor"] = (
        tight.log_evidence(problem.data).value - result["measure"]["log_evidence"]
    )
    print(
        f"the data prefer the tight prior to the wide one by "
        f"{result['log_bayes_factor']:.2f} nats"
    )
    return result


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return wide_answers(problem, outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
