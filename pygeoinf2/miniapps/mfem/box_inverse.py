"""Part E: an inverse problem on the box operator.

The source ``f`` of the screened Poisson equation on the periodic box, from
noisy readings of the solution ``u`` at scattered sensors, none of them
inside one disc: the shape of the point-data mini-app in miniature, with
MFEM computing every forward and adjoint application and the library never
seeing a finite element. The prior is the identity covariance on the box's
``H^2``, so that the space's norm is the prior's and the three answers
measure size alike. Two properties are asked about, the average source over
a small disc among the sensors and over one in the empty disc, and each is
answered three ways: the minimum-norm estimate under the discrepancy
principle, the Bayesian posterior, and the feasible set's certified brackets
under the prior's and the noise's balls.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    BackusGilbertParker,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
    plotting,
)
from pygeoinf2.numerics import SpectralFamilySolver

from box_operator import ScreenedPoisson
from common import (
    FULL,
    QUICK,
    Scale,
    ball_alone,
    bound_panels,
    box_prior,
    box_space,
    error_budget_panels,
    marginal_panels,
    output_directory,
    parse,
    part_rng,
    save,
)

NOISE_STD = 0.05
LEVEL = 0.95
# The sensors avoid a disc of this radius about the gap's center; the
# properties are averages over smaller discs about the two centers.
GAP_RADIUS = 0.18
REGION_RADIUS = 0.08
REGIONS = (
    ("among the sensors", np.array([0.3, 0.65])),
    ("in the gap", np.array([0.7, 0.3])),
)


def sensor_points(
    space: object, count: int, /, *, rng: np.random.Generator
) -> np.ndarray:
    """``count`` points uniform over the box, none within the gap's radius."""
    center = REGIONS[1][1]
    kept: list[np.ndarray] = []
    while len(kept) < count:
        point = rng.uniform(0.0, 1.0, size=2)
        if space.geodesic_distance(point, center) > GAP_RADIUS:
            kept.append(point)
    return np.stack(kept)


def box_inverse(
    scale: Scale, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The forward problem, then the three answers on the two averages."""
    started = time.perf_counter()
    print(f"\n=== E. An inverse problem on the box operator ({scale.name} scale) ===")
    X = box_space(scale)
    prior = box_prior(X)
    box = ScreenedPoisson(
        X, elements=scale.box_elements, screening_length=scale.screening_length
    )
    points = sensor_points(X, scale.sensors, rng=rng)
    forward = X.point_evaluation_operator(points) @ box.operator
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, NOISE_STD)
    problem = LinearForwardProblem(forward, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    averages = X.geodesic_ball_average_operator(
        [center for _, center in REGIONS], REGION_RADIUS
    )
    true_averages = np.asarray(averages(truth))
    print(
        f"the source in H^{X.order:g} on a {X.shape[0]} x {X.shape[1]} grid under the "
        f"identity covariance, length scale {X.length_scale}; u = A f by MFEM on "
        f"{box.size} degrees of freedom, screening length {box.screening_length}; "
        f"{len(points)} sensors outside a disc of radius {GAP_RADIUS}, noise "
        f"{NOISE_STD}; chi-squared of the truth {problem.chi_squared(truth, data):.0f}"
    )

    # Every answer below solves in the data space, which has a few hundred
    # dimensions; conjugate gradients there take more iterations than that,
    # each a forward and an adjoint solve in MFEM. Assembling the data-space
    # operators once, a column per datum, and diagonalizing them costs fewer
    # MFEM solves than one iterative solve, and every solve after it, for a
    # sample or a damping, is a pair of small matrix products.
    solver = SpectralFamilySolver()
    box.reset_counts()
    t0 = time.perf_counter()
    estimator = MinimumNorm(problem, level=LEVEL, solver=solver)
    chosen = estimator.for_data(data)
    smallest = chosen(data)
    pushed = chosen.push_forward(averages)
    estimate = np.asarray(pushed(data))
    propagated = np.sqrt(pushed.propagated_covariance().diagonals()[0])
    bias = np.asarray(pushed.bias(truth))
    print(
        f"the point: minimum norm by the discrepancy principle, damping "
        f"{chosen.damping:.3g}, "
        f"chi-squared {problem.chi_squared(smallest, data):.0f} against the critical "
        f"{problem.critical_chi_squared(level=LEVEL):.0f}, {box.solves} MFEM solves, "
        f"{time.perf_counter() - t0:.1f} s"
    )
    for (name, _), e, s, b, t in zip(
        REGIONS, estimate, propagated, bias, true_averages
    ):
        print(
            f"   {name:<18s} estimate {e:+.3f} +/- {2 * s:.3f} (2 propagated sd), "
            f"bias {b:+.3f}, truth {t:+.3f}"
        )

    box.reset_counts()
    t0 = time.perf_counter()
    inversion = LinearGaussianInversion(problem, prior, solver=solver)
    posterior = inversion(data)
    mean = posterior.expectation
    mean_solves, mean_time = box.solves, time.perf_counter() - t0
    std = np.asarray(posterior.sample_pointwise_std(scale.std_samples, rng=rng))
    print(
        f"the measure: the posterior mean in {mean_time:.1f} s, {mean_solves} MFEM "
        f"solves; relative error "
        f"{np.linalg.norm(mean - truth) / np.linalg.norm(truth):.3f} on the grid, "
        f"chi-squared {problem.chi_squared(mean, data):.0f} on {len(points)} data; "
        f"the pointwise standard deviation from {scale.std_samples} samples, "
        f"{box.solves - mean_solves} solves more"
    )
    exact_std = np.sqrt(
        X.pointwise_variance_at(posterior, [center for _, center in REGIONS])
    )
    on_averages = posterior.affine_map(averages)
    center = np.asarray(on_averages.expectation)
    deviation = np.sqrt(on_averages.covariance.diagonals()[0])
    for (name, _), sd, m, s, t in zip(
        REGIONS, exact_std, center, deviation, true_averages
    ):
        print(
            f"   {name:<18s} pointwise std at the center {sd:.3f} (prior 1.000); "
            f"average {m:+.3f}, 95% [{m - 1.96 * s:+.3f}, {m + 1.96 * s:+.3f}], "
            f"truth {t:+.3f}"
        )

    box.reset_counts()
    t0 = time.perf_counter()
    ball = prior.ambient_ball(level=LEVEL)
    noise_ball = noise.ambient_ball(level=LEVEL)
    feasible = BackusGilbertParker(
        problem, averages, ball, noise=noise_ball, solver=solver
    )(data)
    P = averages.codomain
    directions = []
    for index in range(len(REGIONS)):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    sweep = feasible.support_sweep(directions)
    brackets = np.array(
        [
            (-sweep.results[2 * i + 1].value, sweep.results[2 * i].value)
            for i in range(len(REGIONS))
        ]
    )
    inside = bool(feasible.polytope(directions).contains(true_averages))
    alone = ball_alone(ball, averages)
    print(
        f"the set: the feasible set under the prior's {LEVEL:.0%} ball (radius "
        f"{ball.radius:.1f}) and the noise's: {box.solves} MFEM solves, "
        f"{time.perf_counter() - t0:.1f} s; the true averages in the polytope: {inside}"
    )
    for (name, _), (low, high), (a0, a1), t in zip(
        REGIONS, brackets, alone, true_averages
    ):
        print(
            f"   {name:<18s} certified bracket [{low:+.3f}, {high:+.3f}], the "
            f"ball alone [{a0:+.2f}, {a1:+.2f}], truth {t:+.3f}"
        )

    names = [f"the average source {name}" for name, _ in REGIONS]
    limit = float(np.abs(truth).max())

    def source_map(ax: Any, field: Any, title: str, /, **options: Any) -> None:
        """A field of the box with the sensors and the two discs."""
        plotting.plot(X, np.asarray(field), ax=ax, **options)
        plotting.plot_points(X, points, ax=ax, marker=".", size=4.0, color="black")
        plotting.plot_balls(
            X, [c for _, c in REGIONS], REGION_RADIUS, ax=ax, color="black"
        )
        ax.set_title(title, fontsize=10)

    signed = {"symmetric": True, "cmap": "RdBu_r", "vmin": -limit, "vmax": limit}

    # The point: the estimate, and its error budget on the two averages.
    fig = plt.figure(figsize=(11.0, 8.0), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[1.0, 0.9])
    left, right = upper.subplots(1, 2)
    source_map(left, truth, "the true source", **signed)
    source_map(right, smallest, f"minimum norm, damping {chosen.damping:.3g}", **signed)
    error_budget_panels(
        lower.subplots(1, len(REGIONS)),
        names,
        estimate,
        propagated,
        bias,
        true_averages,
        ylabel="average of f",
    )
    save(
        fig,
        outdir / "box_inverse_point.png",
        title="MFEM, part E: the point answer. The minimum-norm estimate, and its "
        "estimate of each average with its error budget",
    )

    # The measure: prior against posterior.
    prior_std = float(np.sqrt(X.pointwise_variance_at(prior, [REGIONS[0][1]])[0]))
    fig = plt.figure(figsize=(13.0, 7.6), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[1.0, 0.85])
    maps = upper.subplots(1, 3)
    top = 1.05 * max(prior_std, float(std.max()))
    source_map(
        maps[0],
        np.full(std.shape, prior_std),
        "prior pointwise std",
        vmin=0.0,
        vmax=top,
    )
    source_map(maps[1], std, "posterior pointwise std (sampled)", vmin=0.0, vmax=top)
    source_map(maps[2], mean, "posterior mean", **signed)
    marginal_panels(
        lower.subplots(1, len(REGIONS)),
        names,
        on_averages,
        prior.affine_map(averages),
        true_averages,
        xlabel="average of f",
    )
    save(
        fig,
        outdir / "box_inverse_posterior.png",
        title="MFEM, part E: the Bayesian answer, prior against posterior. The "
        "standard deviation before and after, and each average's marginal",
    )

    # The set: each bracket against the bound of the prior's ball alone.
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(8.0, 3.8), layout="constrained")
    bound_panels(axes, names, alone, brackets, true_averages, ylabel="average of f")
    save(
        fig,
        outdir / "box_inverse_brackets.png",
        title=f"MFEM, part E: the set answer. The bound the prior's {LEVEL:.0%} "
        "ball admits alone,\nand the certified bracket the data leave",
    )
    print(f"E took {time.perf_counter() - started:.1f} s")
    return {
        "points": points,
        "truth": truth,
        "true_averages": true_averages,
        "chi_squared": problem.chi_squared(mean, data),
        "critical": problem.critical_chi_squared(level=LEVEL),
        "exact_std": exact_std,
        "std": std,
        "center": center,
        "deviation": deviation,
        "smallest_chi_squared": problem.chi_squared(smallest, data),
        "estimate": estimate,
        "propagated": propagated,
        "bias": bias,
        "brackets": brackets,
        "alone": alone,
        "inside": inside,
        "data": len(points),
    }


def run(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    return box_inverse(scale, rng=part_rng(seed, "e"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
