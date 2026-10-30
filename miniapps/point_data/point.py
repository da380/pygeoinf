"""Part C: the point answers.

Least squares at the damping a Bayesian would guess, the minimum-norm model
under the discrepancy principle, the error budget of the latter on the
regional averages, and its resolution of a localized bump under the sea and
under land. Size is measured in the prior's norm: the problem is
parameterized by the prior's covariance factor, ``m = L w``, and both
estimators measure ``|w|``.
"""

from __future__ import annotations

import pathlib
import time

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import LeastSquares, MinimumNorm

from src.common import (
    draw_map,
    error_budget_panels,
    map_axes,
    parse,
    part_rng,
    save,
)
from src.ocean_mask import geodesic_bump
from src.problem import FULL, QUICK, REGIONS, Problem, Scale, build
from src.solvers import coarse_prior, woodbury_family_solver

# In the prior's norm, least squares at a damping of one is the posterior
# mean of part D: the guess a Bayesian would make.
GUESSED_DAMPING = 1.0


def family_solver(problem: Problem, /) -> Any:
    """The Woodbury solver for the families ``A Q A* + t R`` of both estimators."""
    prior = coarse_prior(problem)
    error = problem.problem.error_measure
    return woodbury_family_solver(
        problem,
        prior_covariance=prior.covariance,
        prior_precision=prior.precision,
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
    # The problem in w, with m = L w: the norm of w is the prior's norm of m.
    # The factor is diagonal in the harmonic basis, so its inverse is exact.
    factor = problem.prior.covariance_factor
    whiten = factor.inverse
    whitened = forward.parameterized(factor)
    norm = factor.domain.norm
    truth_norm = norm(whiten(problem.truth))

    counted.reset()
    t0 = time.perf_counter()
    guess = factor(
        LeastSquares(whitened, damping=GUESSED_DAMPING, solver=solver)(problem.data)
    )
    print(
        f"LeastSquares at damping {GUESSED_DAMPING:g}: {time.perf_counter() - t0:.1f} s, "
        f"{counted.forward} forward, {counted.adjoint} adjoint applications; "
        f"chi-squared {forward.chi_squared(guess, problem.data):.0f} against the "
        f"critical {forward.critical_chi_squared(level=scale.level):.0f}, prior norm "
        f"{norm(whiten(guess)):.2f}, relative error {problem.relative_error(guess):.3f}"
    )

    counted.reset()
    t0 = time.perf_counter()
    estimator = MinimumNorm(whitened, level=scale.level, solver=solver, rtol=1e-2)
    search = estimator.discrepancy_search(problem.data)
    chosen = estimator.for_data(problem.data)
    change = chosen(problem.data)
    smallest = factor(change)
    print(
        f"MinimumNorm by the discrepancy principle: damping {search.argument:.4g} after "
        f"{search.evaluations} misfit evaluations and {search.inner_iterations} CG "
        f"iterations in all, {time.perf_counter() - t0:.1f} s, {counted.forward} "
        f"forward, {counted.adjoint} adjoint applications; chi-squared "
        f"{forward.chi_squared(smallest, problem.data):.0f}, prior norm "
        f"{norm(change):.2f} (the truth's {truth_norm:.2f}), "
        f"relative error {problem.relative_error(smallest):.3f}"
    )

    pushed = chosen.push_forward(problem.averages @ factor)
    estimate = pushed(problem.data)
    propagated = np.sqrt(pushed.propagated_covariance().diagonals()[0])
    bias = pushed.bias(whiten(problem.truth))
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
    # The model-space resolution: L B A L applied to the bump's w.
    resolving = chosen.push_forward(factor)
    seen = {name: resolving.resolution()(whiten(bump)) for name, bump in bumps.items()}
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
        "norm": norm(change),
        "truth_norm": truth_norm,
        "damping": search.argument,
        "estimate": estimate,
        "propagated": propagated,
        "bias": bias,
        "peaks": peaks,
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return point_answers(problem, rng=part_rng(scale.seed, "c"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
