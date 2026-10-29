"""Part D: the Bayesian answer.

The posterior under the Sobolev prior: its mean, its pointwise standard
deviation by sampling, a few samples, and the posterior on three regional
averages with their credible intervals, each against the prior's.
``posterior_summary`` and ``measure_figure`` are also what the
non-stationary extension reruns under its own prior.
"""

from __future__ import annotations

import pathlib
import time

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import GaussianMeasure, LinearGaussianInversion, plotting

from common import draw_map, map_axes, output_directory, parse, part_rng, save
from ocean_mask import area_weights
from problem import (
    CAP_RADIUS,
    FULL,
    PRIOR_STD,
    PROBES,
    QUICK,
    REGIONS,
    Problem,
    Scale,
    build,
)
from solvers import woodbury_solver


def posterior_summary(
    problem: Problem,
    prior: GaussianMeasure,
    /,
    *,
    rng: np.random.Generator,
    label: str,
) -> dict:
    """The posterior under a prior: mean, standard deviation, samples, averages."""
    scale, space, counted = problem.scale, problem.space, problem.counted
    counted.reset()
    t0 = time.perf_counter()
    inversion = LinearGaussianInversion(
        problem.problem, prior, solver=woodbury_solver(problem)
    )
    posterior = inversion(problem.data)
    mean = posterior.expectation
    print(
        f"{label}: posterior mean in {time.perf_counter() - t0:.1f} s "
        f"({counted.forward} forward, {counted.adjoint} adjoint applications), "
        f"relative error {problem.relative_error(mean):.3f}, chi-squared "
        f"{problem.problem.chi_squared(mean, problem.data):.0f}"
    )

    counted.reset()
    t0 = time.perf_counter()
    std = posterior.sample_pointwise_std(
        scale.std_samples, rng=rng, n_jobs=scale.n_jobs
    )
    std_values = space.grid_values(std)
    weights = area_weights(space)
    land, sea = problem.on_land, ~problem.on_land
    std_land = float(np.average(std_values[land], weights=weights[land]))
    std_sea = float(np.average(std_values[sea], weights=weights[sea]))
    workers = "serial" if scale.n_jobs is None else f"{scale.n_jobs} worker processes"
    print(
        f"  pointwise std by {scale.std_samples} posterior samples in "
        f"{time.perf_counter() - t0:.1f} s ({workers}; one solve per sample, relative "
        f"standard error about {1.0 / np.sqrt(2.0 * scale.std_samples):.0%}): "
        f"area-weighted mean {std_land:.3f} over land, {std_sea:.3f} over the sea; "
        f"the prior's is {PRIOR_STD:.3f}"
    )
    t0 = time.perf_counter()
    places = [p for _, p in PROBES]
    exact = np.sqrt(space.pointwise_variance_at(posterior, places))
    before = np.sqrt(space.pointwise_variance_at(prior, places))
    print(
        f"  exact pointwise std ({time.perf_counter() - t0:.1f} s, one solve per point):"
    )
    for (name, _), value, was in zip(PROBES, exact, before):
        print(f"    {name:<16s} {value:.3f}  (prior {was:.3f})")

    t0 = time.perf_counter()
    samples = posterior.samples(2, rng=rng)
    averages = posterior.affine_map(problem.averages)
    prior_averages = prior.affine_map(problem.averages)
    center = averages.expectation
    deviation = np.sqrt(averages.covariance.diagonals()[0])
    print(f"  regional averages ({time.perf_counter() - t0:.1f} s):")
    print(
        f"    {'region':<16s} {'mean':>8s} {'std':>7s} {'95% from':>9s} {'to':>8s} "
        f"{'truth':>8s}"
    )
    for (name, _), m, s, t in zip(REGIONS, center, deviation, problem.true_averages):
        print(
            f"    {name:<16s} {m:+8.3f} {s:7.3f} {m - 1.96 * s:+9.3f} "
            f"{m + 1.96 * s:+8.3f} {t:+8.3f}"
        )
    return {
        "posterior": posterior,
        "mean": mean,
        "std_values": std_values,
        "std_land": std_land,
        "std_sea": std_sea,
        "exact_std": dict(zip((name for name, _ in PROBES), exact)),
        "samples": samples,
        "averages": averages,
        "prior_averages": prior_averages,
        "center": center,
        "deviation": deviation,
    }


def measure_figure(
    problem: Problem,
    result: dict,
    prior_std: np.ndarray,
    /,
    *,
    path: pathlib.Path,
    title: str,
) -> None:
    """Prior against posterior: the pointwise standard deviation before and
    after on one color scale, the posterior mean and a sample, and the
    marginal of each average before and after, with the truth."""
    space = problem.space
    # The maps and the marginals in two subfigures, each laid out on its
    # own: the marginals carry a second axis for the prior.
    fig = plt.figure(figsize=(12.0, 9.8), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[2.0, 0.85])
    grid = upper.add_gridspec(2, 2)
    maps = [[map_axes(upper, grid[row, k]) for k in range(2)] for row in range(2)]
    top = 1.05 * float(max(np.max(prior_std), np.max(result["std_values"])))
    draw_map(
        space, prior_std, maps[0][0], title="prior pointwise std", vmin=0.0, vmax=top
    )
    draw_map(
        space,
        result["std_values"],
        maps[0][1],
        title="posterior pointwise std (sampled)",
        vmin=0.0,
        vmax=top,
    )
    for ax in maps[0]:
        plotting.plot_balls(
            space, [c for _, c in REGIONS], CAP_RADIUS, ax=ax, color="white"
        )
    limit = float(np.abs(space.grid_values(problem.truth)).max())
    draw_map(space, result["mean"], maps[1][0], title="posterior mean", limit=limit)
    draw_map(
        space, result["samples"][0], maps[1][1], title="a posterior sample", limit=limit
    )
    for index, ax in enumerate(lower.subplots(1, len(REGIONS))):
        marginal(
            ax,
            result["averages"],
            result["prior_averages"],
            index,
            truth=problem.true_averages[index],
            title=f"average over {REGIONS[index][0]}",
        )
    save(fig, path, title=title)


def marginal(
    ax: Any,
    posterior: GaussianMeasure,
    prior: GaussianMeasure,
    index: int,
    /,
    *,
    truth: float,
    title: str,
) -> None:
    """One average's marginal, the prior's on its own axis to the right."""
    plotting.plot_densities(
        posterior,
        prior=prior,
        truth=truth,
        index=index,
        ax=ax,
        labels=["posterior"],
        prior_labels=["prior"],
        xlabel="cap average",
        title=title,
    )


def bayesian_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The posterior under the stationary prior, with its figure."""
    started = time.perf_counter()
    space = problem.space
    print(f"\n=== D. The Bayesian answer ({problem.scale.name} scale) ===")
    result = posterior_summary(
        problem, problem.prior, rng=rng, label="stationary prior"
    )

    prior_std = np.full(
        space.grid_values(result["mean"]).shape,
        float(np.sqrt(space.pointwise_variance_at(problem.prior, [PROBES[0][1]])[0])),
    )
    measure_figure(
        problem,
        result,
        prior_std,
        path=outdir / "posterior.png",
        title="Point data, part D: the Bayesian answer, prior against posterior\n"
        "the standard deviation returns to the prior's under the continents, and "
        "Africa's average keeps the prior's width",
    )
    print(f"D took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return bayesian_answer(problem, rng=part_rng(seed, "d"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
