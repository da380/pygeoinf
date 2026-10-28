"""Part E: an inverse problem on the box operator.

The source ``f`` of the screened Poisson equation on the periodic box, from
noisy readings of the solution ``u`` at scattered sensors, none of them
inside one disc: the shape of the point-data mini-app in miniature, with
MFEM computing every forward and adjoint application and the library never
seeing a finite element. The prior is the identity covariance on the box's
``H^2``, so that the space's norm is the prior's and the three answers
measure size alike. Two properties are asked about, the average source over
a small disc among the sensors and over one in the empty disc, and each is
answered three ways: the Bayesian posterior, the minimum-norm estimate under
the discrepancy principle, and the feasible set's certified brackets under
the prior's and the noise's balls.
"""

from __future__ import annotations

import pathlib
import time

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
    box_prior,
    box_space,
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
    print(
        f"the set: the feasible set under the prior's {LEVEL:.0%} ball (radius "
        f"{ball.radius:.1f}) and the noise's: {box.solves} MFEM solves, "
        f"{time.perf_counter() - t0:.1f} s; the true averages in the polytope: {inside}"
    )
    for (name, _), (low, high), t in zip(REGIONS, brackets, true_averages):
        print(
            f"   {name:<18s} certified bracket [{low:+.3f}, {high:+.3f}], truth {t:+.3f}"
        )

    fig = plt.figure(figsize=(13.0, 7.0), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    maps = [fig.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3)]
    limit = float(np.abs(truth).max())
    for ax, field, title, options in (
        (maps[0], truth, "the true source", {"symmetric": True, "cmap": "RdBu_r"}),
        (maps[1], mean, "posterior mean", {"symmetric": True, "cmap": "RdBu_r"}),
        (maps[2], std, "posterior pointwise std", {"vmin": 0.0, "vmax": 1.0}),
    ):
        if options.get("symmetric"):
            options = {**options, "vmin": -limit, "vmax": limit}
        plotting.plot(X, np.asarray(field), ax=ax, **options)
        plotting.plot_points(X, points, ax=ax, marker=".", size=4.0, color="black")
        plotting.plot_balls(
            X, [c for _, c in REGIONS], REGION_RADIUS, ax=ax, color="black"
        )
        ax.set_title(title, fontsize=10)
    for index, (name, _) in enumerate(REGIONS):
        ax = fig.add_subplot(grid[1, 3 * index : 3 * index + 3])
        ax.errorbar(
            0,
            estimate[index],
            yerr=2 * propagated[index],
            fmt="o",
            color="C1",
            capsize=4,
            label="point, 2 propagated sd",
        )
        ax.plot(
            [1, 1], brackets[index], color="C0", lw=6, alpha=0.6, solid_capstyle="butt"
        )
        ax.plot(
            1,
            brackets[index].mean(),
            "_",
            color="C0",
            ms=14,
            label="set, certified bracket",
        )
        ax.errorbar(
            2,
            center[index],
            yerr=1.96 * deviation[index],
            fmt="s",
            color="C2",
            capsize=4,
            label="measure, 95% interval",
        )
        ax.axhline(true_averages[index], color="black", ls="--", lw=1.0, label="truth")
        # The bracket is far wider than the other two answers; the axis is
        # fitted to those, and the bracket's ends are written where it leaves.
        spread = max(2.5 * 1.96 * deviation[index], 2.5 * 2 * propagated[index])
        middle = center[index]
        ax.set_ylim(middle - spread, middle + spread)
        low, high = brackets[index]
        ax.annotate(
            f"set: [{low:+.2f}, {high:+.2f}]",
            (1, middle + 0.92 * spread),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color="C0",
            va="center",
        )
        ax.set_xticks([0, 1, 2], ["point", "set", "measure"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(f"the average source {name}", fontsize=10)
        if index == 0:
            ax.legend(fontsize=7, loc="best")
    save(fig, outdir / "box_inverse.png")
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
