"""Part B: a prior tight about PREM.

The Sobolev measure about PREM, the reference, with a pointwise standard
deviation of 0.1 g/cm^3 in every layer, independent between the layers: the
prior the truth was drawn from. The three answers under it, in order: the point, the smallest change to PREM that fits
the data, measured in the prior's precision, with its error budget; the
measure, the posterior against the prior and the log evidence; the set, the
brackets on the three properties under the prior's 95% ellipsoid and the
noise's 95% ball.
"""

from __future__ import annotations

import pathlib

from src.answers import three_answers
from src.common import parse, part_rng
from src.problem import FULL, QUICK, Problem, Scale, build


def tight_answers(problem: Problem, /, *, outdir: pathlib.Path) -> dict:
    """The three answers under the tight prior."""
    scale = problem.scale
    print(f"\n=== B. A prior tight about PREM ({scale.name} scale) ===")
    print(f"prior: about PREM, pointwise std {scale.tight_std:g} g/cm^3 in every layer")
    return three_answers(
        problem,
        problem.prior(scale.tight_std),
        part="B",
        prior_name="tight",
        rng=part_rng(scale.seed, "b"),
        outdir=outdir,
    )


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return tight_answers(problem, outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
