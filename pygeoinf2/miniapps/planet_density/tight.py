"""Part B: a prior tight about PREM.

The Sobolev measure about PREM, the reference, with a pointwise standard
deviation of 0.1 g/cm^3 in every layer, independent between the layers: the
prior the truth was drawn from. The three answers under it, in order: the point, the smallest change to PREM that fits
the data, measured in the prior's precision, with its error budget; the
measure, the posterior against the prior and the log evidence; the set, the
brackets on the three properties under the prior's 95% ball and the
noise's.
"""

from __future__ import annotations

import pathlib

from answers import three_answers
from common import output_directory, parse, part_rng
from problem import FULL, QUICK, TIGHT_STD, Problem, Scale, build


def tight_answers(problem: Problem, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """The three answers under the tight prior."""
    print(f"\n=== B. A prior tight about PREM ({problem.scale.name} scale) ===")
    print(f"prior: about PREM, pointwise std {TIGHT_STD:g} g/cm^3 in every layer")
    return three_answers(
        problem,
        problem.prior(TIGHT_STD),
        part="B",
        prior_name="tight",
        rng=part_rng(seed, "b"),
        outdir=outdir,
    )


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return tight_answers(problem, seed=seed, outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
