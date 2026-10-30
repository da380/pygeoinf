"""Part C: the Bayesian answer.

The posterior under the prior: its mean, its pointwise standard deviation,
exactly at a few points and by sampling over the whole disk, two samples, and
the posterior on the three properties with their 95% intervals, each against
the prior's. Every solve
is a Cholesky factorization of the data-space normal operator ``A Q A* + R``,
which is a few hundred square: the data are far fewer than the model's
coefficients, and a direct solve is the natural one.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import CholeskySolver, LinearGaussianInversion

from src.common import (
    draw,
    marginal_panels,
    outline_boxes,
    parse,
    part_rng,
    save,
    variances,
)
from src.problem import (
    BOXES,
    CORE_RADIUS,
    FULL,
    PROPERTIES,
    QUICK,
    Problem,
    Scale,
    build,
    prior_pointwise_std,
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
    """The posterior pushed to the three properties, and the prior pushed
    there for comparison."""
    pushed = measure.affine_map(problem.properties)
    before = problem.prior.affine_map(problem.properties)
    return {
        "on_properties": pushed,
        "prior_on_properties": before,
        "center": pushed.expectation,
        "deviation": np.sqrt(variances(pushed.covariance)),
        "prior_deviation": np.sqrt(variances(before.covariance)),
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
    space, scale = problem.space, problem.scale
    print(f"\n=== C. The Bayesian answer ({scale.name} scale) ===")
    result = posterior_summary(problem, rng=rng)

    # The disks and the marginals in two subfigures, each laid out on its
    # own: the marginals carry a second axis for the prior.
    fig = plt.figure(figsize=(13.2, 12.0), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[2.2, 0.9])
    axes = np.asarray(upper.subplots(2, 3, subplot_kw={"aspect": "equal"}))
    mantle = space.layers[-1]
    limit = float(np.abs(mantle.interior_values(problem.truth[-1])).max())
    for ax, field, title in (
        (
            axes[0, 0],
            prior_pointwise_std(space, order=scale.prior_order, std=scale.prior_std),
            "prior pointwise std",
        ),
        (axes[0, 1], result["std"], "posterior pointwise std (sampled)"),
    ):
        draw(space, field, ax, title=title, vmin=0.0, vmax=1.1 * scale.prior_std)
        outline_boxes(ax, BOXES, color="white", linewidth=1.2)
    draw(space, result["mean"], axes[0, 2], title="posterior mean", limit=limit)
    draw(space, problem.truth, axes[1, 0], title="the truth", limit=limit)
    draw(
        space, result["samples"][0], axes[1, 1], title="a posterior sample", limit=limit
    )
    error = space.subtract(result["mean"], problem.truth)
    draw(space, error, axes[1, 2], title="posterior mean less the truth", limit=limit)
    for ax in axes[1]:
        outline_boxes(ax, BOXES, color="black", linewidth=0.6)
    marginal_panels(
        lower.subplots(1, len(PROPERTIES)),
        [
            f"mean over the {name}" + (" box" if name != "core" else "")
            for name in PROPERTIES
        ],
        result["on_properties"],
        result["prior_on_properties"],
        problem.true_properties,
        xlabel="mean slowness perturbation (%)",
    )
    save(
        fig,
        outdir / "posterior.png",
        title="Tomography, part C: the Bayesian answer, prior against posterior\n"
        "pinned under the stations, the prior's in the core "
        f"(r < {CORE_RADIUS}) and where no ray goes",
    )
    print(f"C took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return bayesian_answer(problem, rng=part_rng(scale.seed, "c"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
