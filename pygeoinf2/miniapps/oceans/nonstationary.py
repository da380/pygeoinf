"""Part F: a non-stationary prior.

The stationary prior pushed through a multiplication operator, its
amplitude larger in the southern hemisphere, and the Bayesian answer rerun
under it. The Woodbury surrogate keeps the stationary prior, as a surrogate
may.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from bayesian import posterior_summary
from common import draw_map, map_panels, output_directory, parse, part_rng, save
from problem import FULL, PRIOR_STD, PROBES, QUICK, REGIONS, Problem, Scale, build


def shaped_prior(problem: Problem) -> tuple:
    """The amplitude field and the prior multiplied by it."""
    space = problem.space
    latitudes = 90.0 - np.degrees(space.colatitudes)
    profile = 1.0 + 0.5 * np.tanh(-latitudes / 10.0)
    amplitude = space.from_grid_values(
        np.repeat(profile[:, None], space.longitudes.size, axis=1)
    )
    return amplitude, problem.prior.affine_map(space.multiplication_operator(amplitude))


def nonstationary_answer(
    problem: Problem,
    /,
    *,
    rng: np.random.Generator,
    outdir: pathlib.Path,
    bayesian: dict | None = None,
) -> dict:
    """The posterior under the shaped prior, and what changed."""
    started = time.perf_counter()
    space = problem.space
    print(
        f"\n=== F. Extension: a non-stationary prior ({problem.scale.name} scale) ==="
    )
    if bayesian is None:
        bayesian = posterior_summary(
            problem, problem.prior, rng=rng, label="stationary prior"
        )
    amplitude, shaped = shaped_prior(problem)
    print(
        "the prior's draws are multiplied by an amplitude of 1.5 in the southern "
        "hemisphere and 0.5 in the northern, changing over about 10 degrees at the "
        "equator; the Woodbury surrogate still uses the stationary prior"
    )
    result = posterior_summary(problem, shaped, rng=rng, label="non-stationary prior")
    print("what changed against the stationary posterior:")
    for name, _ in PROBES:
        print(
            f"  std at {name:<16s} {bayesian['exact_std'][name]:.3f} -> "
            f"{result['exact_std'][name]:.3f}"
        )
    for index, (name, _) in enumerate(REGIONS):
        print(
            f"  {name:<16s} {bayesian['center'][index]:+.3f} +/- "
            f"{bayesian['deviation'][index]:.3f} -> {result['center'][index]:+.3f} +/- "
            f"{result['deviation'][index]:.3f} (truth {problem.true_averages[index]:+.3f})"
        )

    fig, axes = map_panels(space, 1, 2)
    draw_map(
        space,
        amplitude,
        axes[0, 0],
        title="the amplitude the prior is multiplied by",
        vmin=0.0,
        vmax=1.6,
    )
    draw_map(
        space,
        result["std_values"],
        axes[0, 1],
        title="posterior pointwise std under the non-stationary prior",
        vmin=0.0,
        vmax=1.5 * PRIOR_STD,
    )
    save(fig, outdir / "nonstationary.png")
    print(f"F took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale,
    /,
    *,
    seed: int,
    outdir: pathlib.Path,
    problem: Problem | None = None,
    bayesian: dict | None = None,
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return nonstationary_answer(
        problem, rng=part_rng(seed, "f"), outdir=outdir, bayesian=bayesian
    )


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
