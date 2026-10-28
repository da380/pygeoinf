"""Part B: the Bayesian answer.

The posterior under the prior: its mean, its pointwise standard deviation,
exactly at a few points and by sampling over the whole disk, two samples, and
the posterior on the three properties with their 95% intervals. Every solve
is a Cholesky factorization of the data-space normal operator ``A Q A* + R``,
which is a few hundred square: the data are far fewer than the model's
coefficients, and a direct solve is the natural one.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import numpy as np

from pygeoinf2 import CholeskySolver, LinearGaussianInversion

from common import (
    draw,
    outline_boxes,
    output_directory,
    panels,
    parse,
    part_rng,
    save,
    variances,
)
from problem import (
    BOXES,
    CORE_RADIUS,
    FULL,
    PRIOR_STD,
    PROPERTIES,
    QUICK,
    Problem,
    Scale,
    build,
)

#: Where the standard deviation is read exactly: the middle of each box, the
#: middle of the core, and the mantle under the stations near the surface.
PROBES = (
    ("well covered", (0.825, 75.0)),
    ("poorly covered", (0.825, 270.0)),
    ("core", (0.2, 0.0)),
    ("under the stations", (0.95, 75.0)),
)


def exact_pointwise_std(space: Any, measure: Any, points: list, /) -> np.ndarray:
    """The standard deviation at points, exactly: the covariance pushed
    through point evaluation, one application per point."""
    pushed = measure.affine_map(space.point_evaluation_operator(points))
    return np.sqrt(variances(pushed.covariance))


def posterior(problem: Problem, /) -> Any:
    """The posterior, every solve a Cholesky factorization of ``A Q A* + R``."""
    inversion = LinearGaussianInversion(
        problem.problem, problem.prior, solver=CholeskySolver()
    )
    return inversion(problem.data)


def property_posterior(problem: Problem, measure: Any, /) -> dict:
    """The posterior pushed to the three properties, and the prior's spread
    there for comparison."""
    pushed = measure.affine_map(problem.properties)
    return {
        "center": pushed.expectation,
        "deviation": np.sqrt(variances(pushed.covariance)),
        "prior_deviation": np.sqrt(
            variances(problem.prior.affine_map(problem.properties).covariance)
        ),
    }


def posterior_summary(
    problem: Problem, /, *, rng: np.random.Generator, say: Any = print
) -> dict:
    """The posterior: mean, standard deviation, samples, properties."""
    scale, space = problem.scale, problem.space
    t0 = time.perf_counter()
    measure = posterior(problem)
    mean = measure.expectation
    say(
        f"posterior mean in {time.perf_counter() - t0:.2f} s, one Cholesky "
        f"factorization of the {len(problem.rays)}-square data-space normal "
        f"operator; relative error over the mantle {problem.relative_error(mean):.3f}, "
        f"chi-squared {problem.problem.chi_squared(mean, problem.data):.1f} on "
        f"{len(problem.rays)} data"
    )

    t0 = time.perf_counter()
    places = problem.lift([point for _, point in PROBES])
    exact = exact_pointwise_std(space, measure, places)
    before = exact_pointwise_std(space, problem.prior, places)
    say(
        f"  exact pointwise std ({time.perf_counter() - t0:.2f} s, one solve per point):"
    )
    for (name, point), value, was in zip(PROBES, exact, before):
        say(f"    {name:<20s} at r = {point[0]:<5g} {value:.3f}  (prior {was:.3f})")

    t0 = time.perf_counter()
    std = measure.sample_pointwise_std(scale.std_samples, rng=rng, n_jobs=scale.n_jobs)
    workers = "serial" if scale.n_jobs is None else f"{scale.n_jobs} worker processes"
    sampled = space.evaluate(std, places)
    say(
        f"  pointwise std by {scale.std_samples} posterior samples in "
        f"{time.perf_counter() - t0:.1f} s ({workers}; relative standard error "
        f"about {1.0 / np.sqrt(2.0 * scale.std_samples):.0%}): at the probes "
        + ", ".join(f"{value:.3f}" for value in sampled)
    )

    t0 = time.perf_counter()
    samples = measure.samples(2, rng=rng)
    properties = property_posterior(problem, measure)
    say(f"  the three properties ({time.perf_counter() - t0:.2f} s):")
    say(
        f"    {'mean over':<16s} {'posterior':>9s} {'std':>7s} {'95% from':>9s} "
        f"{'to':>8s} {'truth':>8s} {'prior std':>10s}"
    )
    for name, m, s, t, p in zip(
        PROPERTIES,
        properties["center"],
        properties["deviation"],
        problem.true_properties,
        properties["prior_deviation"],
    ):
        say(
            f"    {name:<16s} {m:+9.3f} {s:7.3f} {m - 1.96 * s:+9.3f} "
            f"{m + 1.96 * s:+8.3f} {t:+8.3f} {p:10.3f}"
        )
    return {
        "posterior": measure,
        "mean": mean,
        "std": std,
        "exact_std": dict(zip((name for name, _ in PROBES), exact)),
        "prior_std": dict(zip((name for name, _ in PROBES), before)),
        "sampled_std": dict(zip((name for name, _ in PROBES), sampled)),
        "samples": samples,
        **properties,
    }


def bayesian_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The posterior, with its figure."""
    started = time.perf_counter()
    space = problem.space
    print(f"\n=== B. The Bayesian answer ({problem.scale.name} scale) ===")
    result = posterior_summary(problem, rng=rng)

    fig, axes = panels(space, 2, 3)
    mantle = space.layers[-1]
    limit = float(np.abs(mantle.interior_values(problem.truth[-1])).max())
    draw(space, problem.truth, axes[0, 0], title="the truth", limit=limit)
    draw(space, result["mean"], axes[0, 1], title="posterior mean", limit=limit)
    draw(
        space,
        result["std"],
        axes[0, 2],
        title="posterior pointwise std (sampled)",
        vmin=0.0,
        vmax=1.1 * PRIOR_STD,
    )
    outline_boxes(axes[0, 2], BOXES, color="white", linewidth=1.2)
    for ax, sample, title in zip(
        axes[1, :2],
        result["samples"],
        ("a posterior sample", "another posterior sample"),
    ):
        draw(space, sample, ax, title=title, limit=limit)
    error = space.subtract(result["mean"], problem.truth)
    draw(space, error, axes[1, 2], title="posterior mean less the truth", limit=limit)
    for ax in axes.ravel():
        outline_boxes(ax, BOXES, color="black", linewidth=0.6)
    fig.suptitle(
        "The posterior: pinned under the stations, the prior's in the core "
        f"(r < {CORE_RADIUS}) and where no ray goes"
    )
    save(fig, outdir / f"{problem.tag}posterior.png")
    print(f"B took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return bayesian_answer(problem, rng=part_rng(seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
