"""Part B: an effectively non-informative prior.

The prior is centered on the Adams-Williamson reference with a pointwise
standard deviation of 3 g/cm^3 in every layer, enough to admit any density
the Earth could have and many it could not. The posterior on the three
properties shows what two numbers say through a function space: the mass is
exactly the volume-weighted sum of the core's and the mantle's mean
densities, so the posterior on that pair is a thin ellipse along the line of
constant mass, which the moment of inertia, saying how mass is spread with
radius rather than how it splits, shortens only a little; the jump at the
core-mantle boundary is barely touched. The feasible set under the prior's
95% ball says the same from the set side, in brackets that are wider still.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker, LinearGaussianInversion, plotting

from common import (
    draw_band,
    draw_prem,
    draw_profile,
    finish_profile_axes,
    output_directory,
    parse,
    part_rng,
    pointwise_std,
    save,
)
from problem import (
    CORE_VOLUME,
    FULL,
    LEVEL,
    LOOSE_STD,
    MANTLE_VOLUME,
    PROPERTY_SHORT,
    QUICK,
    Problem,
    Scale,
    build,
)


def posterior_summary(problem: Problem, prior, /, *, label: str) -> dict:
    """The posterior under a prior, on the profile and on the three properties."""
    t0 = time.perf_counter()
    inversion = LinearGaussianInversion(problem.problem, prior)
    posterior = inversion(problem.data)
    mean = posterior.expectation
    on_properties = posterior.push_forward(problem.properties)
    center = on_properties.expectation
    covariance = on_properties.covariance.matrix(form="components")
    deviation = np.sqrt(np.diag(covariance))
    before = prior.push_forward(problem.properties)
    prior_deviation = np.sqrt(np.diag(before.covariance.matrix(form="components")))
    log_evidence = inversion.log_evidence(problem.data)
    sigmas = problem.misfit(mean)
    print(
        f"{label}: posterior in {time.perf_counter() - t0:.2f} s; its mean misses "
        f"the mass by {sigmas[0]:+.2f} sd and the moment by {sigmas[1]:+.2f} sd; "
        f"log evidence {log_evidence:.1f}"
    )
    print(
        f"  {'property':<12s} {'prior':>13s} {'posterior':>17s} {'95% from':>9s} "
        f"{'to':>7s} {'PREM':>7s}"
    )
    reference = before.expectation
    for name, r, s0, m, s, p in zip(
        PROPERTY_SHORT,
        reference,
        prior_deviation,
        center,
        deviation,
        problem.prem_properties,
    ):
        print(
            f"  {name:<12s} {r:6.3f} ± {s0:5.3f} {m:8.3f} ± {s:6.3f} "
            f"{m - 1.96 * s:9.3f} {m + 1.96 * s:7.3f} {p:7.3f}"
        )
    return {
        "posterior": posterior,
        "mean": mean,
        "properties": on_properties,
        "center": center,
        "deviation": deviation,
        "covariance": covariance,
        "prior_properties": before,
        "prior_deviation": prior_deviation,
        "log_evidence": log_evidence,
        "misfit": sigmas,
    }


def feasible_brackets(problem: Problem, prior, /, *, label: str) -> dict:
    """The feasible property set under the prior's 95% ball and the noise's.

    Brackets on each property when the set is not empty; when it is, how far
    beyond the ball the nearest model that fits the data lies.
    """
    t0 = time.perf_counter()
    ball = prior.ambient_ball(level=LEVEL)
    feasible = BackusGilbertParker(problem.problem, problem.properties, ball)(
        problem.data
    )
    nearest = feasible.feasibility_result()
    reach = float(np.sqrt(nearest.level / feasible.level))
    empty = feasible.is_empty()
    print(
        f"{label}: the prior's {LEVEL:.0%} ball has radius {ball.radius:.4g} in "
        f"H^2 (km^3/2 g/cm^3); the nearest model that fits the data lies "
        f"{reach:.2f} radii from its center"
    )
    result = {"radius": ball.radius, "empty": empty, "reach": reach}
    if empty:
        print(
            "  the feasible set is empty: no model within the ball fits the data "
            f"({time.perf_counter() - t0:.2f} s)"
        )
        result["brackets"] = None
        return result
    P = problem.properties.codomain
    brackets = np.zeros((3, 2))
    gaps = np.zeros(3)
    for index in range(3):
        q = P.basis_vector(index)
        inner_high, high = feasible.support_bracket(q)
        inner_low, low = feasible.support_bracket(P.scale(-1.0, q))
        brackets[index] = (-low, high)
        gaps[index] = max(high - inner_high, low - inner_low)
    print(f"  brackets ({time.perf_counter() - t0:.2f} s):")
    print(f"    {'property':<12s} {'from':>8s} {'to':>8s} {'gap':>8s} {'PREM':>7s}")
    for name, (low, high), gap, p in zip(
        PROPERTY_SHORT, brackets, gaps, problem.prem_properties
    ):
        print(f"    {name:<12s} {low:8.3f} {high:8.3f} {gap:8.1e} {p:7.3f}")
    result.update({"brackets": brackets, "gaps": gaps})
    return result


def uninformative_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The posterior and the feasible set under the loose prior, with a figure."""
    started = time.perf_counter()
    space = problem.space
    print(
        f"\n=== B. An effectively non-informative prior ({problem.scale.name} scale) ==="
    )
    prior = problem.prior(LOOSE_STD)
    print(
        f"prior: about the reference, pointwise std {LOOSE_STD:g} g/cm^3 in every layer"
    )
    result = posterior_summary(problem, prior, label="loose prior")
    covariance = result["covariance"]
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    weights = np.array([CORE_VOLUME, MANTLE_VOLUME]) / (CORE_VOLUME + MANTLE_VOLUME)
    whole = float(np.sqrt(weights @ covariance[:2, :2] @ weights))
    print(
        f"  the core and mantle means are correlated at {correlation:+.4f}: their "
        f"volume-weighted sum, the Earth's mean density, has a posterior sd of "
        f"{whole:.2e} g/cm^3, which is the mass's error and nothing else"
    )
    result["correlation"] = correlation
    result["whole_earth_std"] = whole
    result.update(feasible_brackets(problem, prior, label="loose prior"))

    std = pointwise_std(space, result["posterior"])
    fig = plt.figure(figsize=(13.0, 5.2), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.0, 1.15])
    ax = left.subplots()
    draw_prem(ax, color="black", lw=1.0, label="PREM")
    draw_band(
        space,
        result["mean"],
        std,
        ax,
        color="C0",
        alpha=0.25,
        label="posterior mean ± 2 pointwise sd",
    )
    draw_profile(space, result["mean"], ax, color="C0", lw=1.6, label="posterior mean")
    draw_profile(
        space,
        problem.reference_model,
        ax,
        color="C1",
        lw=1.0,
        ls="--",
        label="reference",
    )
    finish_profile_axes(
        ax, title=f"Prior std {LOOSE_STD:g} g/cm³: the profile is barely constrained"
    )
    corner = right.subplots(3, 3)
    plotting.plot_corner(
        result["properties"],
        prior=result["prior_properties"],
        truth=problem.prem_properties,
        labels=[f"{name} (g/cm³)" for name in PROPERTY_SHORT],
        axes=corner,
        truth_label="PREM",
        rng=rng,
    )
    if result["brackets"] is not None:
        # The brackets are far wider than the panels, which are scaled to the
        # posterior; they are written in rather than drawn off the edge.
        for index, (low, high) in enumerate(result["brackets"]):
            corner[index, index].text(
                0.03,
                0.95,
                f"set: [{low:.2f}, {high:.2f}]",
                transform=corner[index, index].transAxes,
                fontsize=7,
                color="C3",
                va="top",
            )
    right.suptitle(
        "The three properties: two numbers through a function space", fontsize=10
    )
    save(fig, outdir / "uninformative.png")
    print(f"B took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return uninformative_answer(problem, rng=part_rng(seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
