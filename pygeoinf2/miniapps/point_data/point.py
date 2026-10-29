"""Part C: the point answers.

Least squares at a guessed damping, the minimum-norm model under the
discrepancy principle, the error budget of the latter on the regional
averages, and its resolution of a localized bump under the sea and under
land.
"""

from __future__ import annotations

import pathlib
import time

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import LeastSquares, MinimumNorm

from common import (
    draw_map,
    error_budget_panels,
    map_axes,
    output_directory,
    parse,
    part_rng,
    save,
)
from ocean_mask import geodesic_bump
from problem import FULL, LEVEL, QUICK, REGIONS, Problem, Scale, build
from solvers import coarse_identity, woodbury_family_solver

GUESSED_DAMPING = 10.0


def family_solver(problem: Problem, /) -> Any:
    """The Woodbury solver for the families ``A A* + t R`` of both estimators."""
    identity = coarse_identity(problem)
    error = problem.problem.error_measure
    return woodbury_family_solver(
        problem,
        prior_covariance=identity,
        prior_precision=identity,
        noise_covariance=error.covariance,
        noise_precision=error.precision,
    )


def point_answers(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """Both estimators, their budget on the averages, their resolution of a bump."""
    started = time.perf_counter()
    scale, space, counted, forward = (
        problem.scale,
        problem.space,
        problem.counted,
        problem.problem,
    )
    print(f"\n=== C. The point answers ({scale.name} scale) ===")
    solver = family_solver(problem)

    counted.reset()
    t0 = time.perf_counter()
    guess = LeastSquares(forward, damping=GUESSED_DAMPING, solver=solver)(problem.data)
    print(
        f"LeastSquares at damping {GUESSED_DAMPING:g}: {time.perf_counter() - t0:.1f} s, "
        f"{counted.forward} forward, {counted.adjoint} adjoint applications; "
        f"chi-squared {forward.chi_squared(guess, problem.data):.0f} against the "
        f"critical {forward.critical_chi_squared(level=LEVEL):.0f}, norm "
        f"{space.norm(guess):.2f}, relative error {problem.relative_error(guess):.3f}"
    )

    counted.reset()
    t0 = time.perf_counter()
    estimator = MinimumNorm(forward, level=LEVEL, solver=solver, rtol=1e-2)
    search = estimator.discrepancy_search(problem.data)
    chosen = estimator.for_data(problem.data)
    smallest = chosen(problem.data)
    print(
        f"MinimumNorm by the discrepancy principle: damping {search.argument:.4g} after "
        f"{search.evaluations} misfit evaluations and {search.inner_iterations} CG "
        f"iterations in all, {time.perf_counter() - t0:.1f} s, {counted.forward} "
        f"forward, {counted.adjoint} adjoint applications; chi-squared "
        f"{forward.chi_squared(smallest, problem.data):.0f}, norm "
        f"{space.norm(smallest):.2f} (the truth's {space.norm(problem.truth):.2f}), "
        f"relative error {problem.relative_error(smallest):.3f}"
    )

    pushed = chosen.push_forward(problem.averages)
    estimate = pushed(problem.data)
    propagated = np.sqrt(pushed.propagated_covariance().diagonals()[0])
    bias = pushed.bias(problem.truth)
    print("the minimum-norm estimate on the regional averages:")
    print(
        f"  {'region':<16s} {'estimate':>9s} {'propagated sd':>14s} {'bias':>8s} "
        f"{'truth':>8s}"
    )
    for (name, _), e, s, b, t in zip(
        REGIONS, estimate, propagated, bias, problem.true_averages
    ):
        print(f"  {name:<16s} {e:+9.3f} {s:14.3f} {b:+8.3f} {t:+8.3f}")

    t0 = time.perf_counter()
    bumps = {
        name: geodesic_bump(space, center, scale.bump_width)
        for name, center in (
            ("North Pacific", REGIONS[0][1]),
            ("Africa", REGIONS[2][1]),
        )
    }
    seen = {name: chosen.resolution(bump) for name, bump in bumps.items()}
    print(
        f"the resolution of a unit bump of width {scale.bump_width} "
        f"({time.perf_counter() - t0:.1f} s):"
    )
    peaks = {}
    for name, image in seen.items():
        values = space.grid_values(image)
        peaks[name] = float(values.max())
        print(
            f"  under {name:<14s} the estimate sees a peak of {values.max():.3f}, "
            f"lowest value {values.min():+.3f}"
        )

    fig = plt.figure(figsize=(12.0, 9.6), layout="constrained")
    grid = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.9])
    maps = [
        [map_axes(fig, grid[row, 3 * k : 3 * k + 3]) for k in range(2)]
        for row in range(2)
    ]
    limit = float(np.abs(space.grid_values(problem.truth)).max())
    draw_map(
        space,
        guess,
        maps[0][0],
        title=f"least squares at damping {GUESSED_DAMPING:g}",
        limit=limit,
    )
    draw_map(
        space,
        smallest,
        maps[0][1],
        title=f"minimum norm, damping {search.argument:.3g}",
        limit=limit,
    )
    for ax, name in zip(maps[1], ("North Pacific", "Africa")):
        draw_map(
            space,
            seen[name],
            ax,
            title=f"resolution of a unit bump under {name}",
            limit=1.0,
        )
    error_budget_panels(
        [fig.add_subplot(grid[2, 2 * k : 2 * k + 2]) for k in range(len(REGIONS))],
        [f"average over {name}" for name, _ in REGIONS],
        estimate,
        propagated,
        bias,
        problem.true_averages,
        ylabel="cap average",
    )
    save(
        fig,
        outdir / "point.png",
        title="Point data, part C: the point answers. Least squares and the "
        "minimum-norm model,\nhow the latter sees a bump under the sea and under "
        "land, and its estimate of the averages with its error budget",
    )
    print(f"C took {time.perf_counter() - started:.1f} s")
    return {
        "guess": guess,
        "smallest": smallest,
        "damping": search.argument,
        "estimate": estimate,
        "propagated": propagated,
        "bias": bias,
        "peaks": peaks,
    }


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return point_answers(problem, rng=part_rng(seed, "c"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
