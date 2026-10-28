"""Part C: the Bayesian answer.

The posterior under the Sobolev prior: its mean, its pointwise standard
deviation by sampling, a few samples, and the posterior on three regional
averages with their credible intervals. ``posterior_summary`` is also what
the non-stationary extension reruns under its own prior.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from pygeoinf2 import GaussianMeasure, LinearGaussianInversion, plotting

from common import draw_map, map_panels, output_directory, parse, part_rng, save
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
        "center": center,
        "deviation": deviation,
    }


def bayesian_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The posterior under the stationary prior, with its figure."""
    started = time.perf_counter()
    space = problem.space
    print(f"\n=== C. The Bayesian answer ({problem.scale.name} scale) ===")
    result = posterior_summary(
        problem, problem.prior, rng=rng, label="stationary prior"
    )

    fig, axes = map_panels(space, 2, 2)
    limit = float(np.abs(space.grid_values(problem.truth)).max())
    draw_map(space, result["mean"], axes[0, 0], title="posterior mean", limit=limit)
    draw_map(
        space,
        result["std_values"],
        axes[0, 1],
        title="posterior pointwise std",
        vmin=0.0,
        vmax=PRIOR_STD,
    )
    plotting.plot_balls(
        space, [c for _, c in REGIONS], CAP_RADIUS, ax=axes[0, 1], color="white"
    )
    for ax, sample, title in zip(
        axes[1], result["samples"], ("a posterior sample", "another posterior sample")
    ):
        draw_map(space, sample, ax, title=title, limit=limit)
    fig.suptitle(
        "The posterior: the standard deviation returns to the prior's under the continents"
    )
    save(fig, outdir / "posterior.png")
    print(f"C took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return bayesian_answer(problem, rng=part_rng(seed, "c"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
