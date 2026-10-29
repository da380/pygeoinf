"""Part E: a discrete hypothesis inside the continuous problem, by a mixture prior.

A plume rises through the mantle at one of a handful of candidate angles,
and which one is the question. The plume is a conduit, an idealized one: a
slowness perturbation along a radial line from the core-mantle boundary to
the surface, Gaussian across the line and flat along it. Each candidate is a
component of the prior: the Gaussian whose mean is the conduit at that angle
and whose covariance is the app's prior covariance, the same for every
component; the prior weights are equal. Under the linear forward problem the
posterior is again a mixture: each component is updated by its own linear
Gaussian inversion, and the weights are rescored by the components'
evidences, so that they are the evidence ratios of the candidates. The
covariance being shared, the volume halves of the evidences are equal and
the weights compare the data's misfits to each candidate's travel times in
the metric of ``A C A* + R``.

The truth is the plume at one candidate, under moderate coverage, over a
small draw from the shared covariance. The answer: the weights against the
prior's, the posterior mixture on the three properties against the prior
mixture, the winning component's mean; then the weights across a sweep of
the noise level, the data redrawn many times at each, and the weights each
candidate's own travel times would earn, which show where the coverage
leaves two candidates hard to tell apart. Every solve is a Cholesky
factorization of the data-space normal operator, one per component and
noise level.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    CholeskySolver,
    GaussianMeasure,
    GaussianMixture,
    LinearForwardProblem,
    LinearGaussianMixtureInversion,
)

from src.common import (
    draw,
    draw_rays,
    mixture_marginal,
    mixture_marginal_panels,
    parse,
    part_rng,
    save,
    stations,
    variances,
)
from src.problem import (
    CORE_RADIUS,
    FULL,
    PROPERTIES,
    QUICK,
    RADIUS,
    Problem,
    Scale,
    build,
)

# The candidates, the plume's width, peak and background, and the redraws of
# the sweep are the scale's.
#: Where the plume rises: the candidate under moderate coverage, one of the
#: scale's candidate angles, in degrees.
TRUE_CANDIDATE = 330.0
#: The noise levels of the sweep, as multiples of the app's.
NOISE_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
#: The colors of the candidates that are not the truth, which is red.
OTHERS = ("C0", "C1", "C2", "C4", "C5", "C6")


def amplitude(scale: Scale, /) -> float:
    """The plume's peak, in percent of the surface slowness: the scale's
    multiple of the prior's pointwise standard deviation."""
    return scale.plume_peak * scale.prior_std


def distance_to_axis(radius: Any, angle: Any, axis: float, /) -> Any:
    """The distance from points ``(radius, angle)`` of the disk to a
    conduit's axis, the radial segment at ``axis`` degrees from the
    core-mantle boundary to the surface: across the segment beside it, and to
    its nearer end beyond either. Angles in degrees; arrays or numbers."""
    x = radius * np.cos(np.radians(angle))
    y = radius * np.sin(np.radians(angle))
    c, s = np.cos(np.radians(axis)), np.sin(np.radians(axis))
    along = np.clip(x * c + y * s, CORE_RADIUS, RADIUS)
    return np.hypot(x - along * c, y - along * s)


def conduit(space: Any, axis: float, width: float, /) -> Any:
    """A conduit of unit peak on the layered disk: ``exp(-d^2 / (2 width^2))``
    with ``d`` the distance to its axis at ``axis`` degrees, so that it is
    Gaussian across the axis and flat along it from the core-mantle boundary
    to the surface.

    Nothing in the core, which no ray enters and no box reaches. Its values
    on the mantle's grid are continued at their end values across the
    padding: a fit over the domain alone of a profile this narrow, at full
    height at both edges of the layer, leaks into the padding at the full
    scale.
    """

    def value(point: tuple[float, float]) -> float:
        d = distance_to_axis(point[0], point[1], axis)
        return float(np.exp(-0.5 * (d / width) ** 2))

    return space.project_function([lambda point: 0.0, value], extension="constant")


def plume(problem: Problem, axis: float, /) -> Any:
    """The plume at a candidate angle: the conduit there, of the scale's width
    and of :func:`amplitude`."""
    scale = problem.scale
    shape = conduit(problem.space, axis, scale.plume_width)
    return problem.space.scale(amplitude(scale), shape)


def plume_prior(problem: Problem, angles: Any, /) -> GaussianMixture:
    """One component a candidate: the app's prior translated to the plume
    there, every component with the same covariance, the weights equal."""
    return GaussianMixture(
        [problem.prior.translate(plume(problem, axis)) for axis in angles]
    )


def coverage(problem: Problem, axis: float, /) -> int:
    """How many rays pass within the plume's width of the conduit's axis."""
    width = problem.scale.plume_width
    return sum(
        bool(distance_to_axis(ray.nodes[:, 0], ray.nodes[:, 1], axis).min() < width)
        for ray in problem.rays
    )


def noisy_problem(problem: Problem, factor: float, /) -> LinearForwardProblem:
    """The forward problem with the noise at a multiple of the app's."""
    noise = GaussianMeasure.from_standard_deviation(
        problem.forward.codomain, factor * problem.scale.noise_std
    )
    return LinearForwardProblem(problem.forward, error=noise)


def inversion(
    forward: LinearForwardProblem, prior: GaussianMixture, /
) -> LinearGaussianMixtureInversion:
    """The mixture's posterior, each component's normal operator inverted
    by a Cholesky factorization in the data space."""
    return LinearGaussianMixtureInversion(forward, prior, solver=CholeskySolver())


def effective_number(weights: np.ndarray, /) -> float:
    """``exp`` of the weights' entropy: one when a single candidate holds all
    the weight, the number of candidates when the weights are equal."""
    live = weights[weights > 0.0]
    return float(np.exp(-np.sum(live * np.log(live))))


def noise_sweep(
    problem: Problem,
    prior: GaussianMixture,
    truth: Any,
    /,
    *,
    redraws: int,
    rng: np.random.Generator,
) -> dict:
    """The weights at each noise level of :data:`NOISE_FACTORS`, the data
    redrawn ``redraws`` times at each; and at each level the weights each
    candidate's own travel times would earn, noise-free and with no
    background, a row a candidate: the weights that data from a plume there
    give at the mean of their log evidences."""
    count = len(prior)
    weights = np.empty((len(NOISE_FACTORS), redraws, count))
    confusion = np.empty((len(NOISE_FACTORS), count, count))
    predictions = [problem.forward(c.expectation) for c in prior.components]
    for row, factor in enumerate(NOISE_FACTORS):
        forward = noisy_problem(problem, factor)
        mixture = inversion(forward, prior)
        for column in range(redraws):
            weights[row, column] = mixture.weights(
                forward.synthetic_data(truth, rng=rng)
            )
        confusion[row] = [mixture.weights(data) for data in predictions]
    return {
        "factors": np.array(NOISE_FACTORS),
        "weights": weights,
        "confusion": confusion,
    }


def mixture_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The plume's prior, truth and data, the posterior mixture, the noise
    sweep, and the figures."""
    started = time.perf_counter()
    scale, space = problem.scale, problem.space
    print(
        f"\n=== E. A plume among candidates, by a mixture prior ({scale.name} scale) ==="
    )
    angles = scale.candidates
    true_index = angles.index(TRUE_CANDIDATE)

    t0 = time.perf_counter()
    prior = plume_prior(problem, angles)
    background = space.scale(scale.background, problem.prior.sample(rng=rng))
    truth = space.add(prior.components[true_index].expectation, background)
    data = problem.problem.synthetic_data(truth, rng=rng)
    # The disk's geometry, prior and noise, with this part's truth and data.
    problem = dataclasses.replace(
        problem,
        truth=truth,
        data=data,
        true_properties=problem.properties(truth),
    )
    counts = [coverage(problem, axis) for axis in angles]
    print(
        f"{len(angles)} candidates for a plume, a conduit from the core-mantle "
        f"boundary to the surface of peak {amplitude(scale):g}% and width "
        f"{scale.plume_width}, each a component of the prior with the app's "
        f"covariance, weights equal ({time.perf_counter() - t0:.2f} s):"
    )
    for index, (axis, count) in enumerate(zip(angles, counts)):
        mark = "  <- the truth" if index == true_index else ""
        print(
            f"  {index + 1}: at {axis:5.1f} degrees, {count:3d} rays within its "
            f"width{mark}"
        )
    spread = problem.prior.affine_map(problem.forward).covariance
    print(
        f"truth: the plume at candidate {true_index + 1} over {scale.background:g} of a "
        f"draw from the shared covariance; travel-time perturbations of std "
        f"{np.std(problem.forward(truth)):.3f}, the plume's alone "
        f"{np.std(problem.forward(prior.components[true_index].expectation)):.3f}, "
        f"against noise of {scale.noise_std} and the shared covariance's "
        f"{np.sqrt(variances(spread).mean()):.3f} a datum"
    )

    t0 = time.perf_counter()
    estimator = inversion(problem.problem, prior)
    terms = estimator.log_evidence_terms(data)
    weights = estimator.weights(data)
    posterior = estimator(data)
    winner = int(np.argmax(weights))
    mean = posterior.components[winner].expectation
    misfits = [component.mahalanobis(data) for component in estimator.inversions]
    print(
        f"the posterior mixture in {time.perf_counter() - t0:.2f} s, a factorization "
        f"of the {len(problem.rays)}-square normal operator for each component:"
    )
    print(
        f"  {'candidate':>9s} {'misfit':>9s} {'log evidence':>13s} "
        f"{'less the best':>14s} {'prior weight':>13s} {'posterior weight':>17s}"
    )
    for index in range(len(angles)):
        print(
            f"  {index + 1:9d} {misfits[index]:9.1f} {terms[index]:13.2f} "
            f"{terms[index] - terms.max():14.2f} {prior.weights[index]:13.3f} "
            f"{weights[index]:17.3g}"
        )
    print(
        f"  the winner is candidate {winner + 1}; the relative error of its mean "
        f"over the mantle {problem.relative_error(mean):.3f}, chi-squared "
        f"{problem.problem.chi_squared(mean, data):.1f} on {len(problem.rays)} data"
    )

    t0 = time.perf_counter()
    prior_on = prior.push_forward(problem.properties)
    posterior_on = estimator.push_forward(problem.properties)(data)
    true_properties = problem.true_properties
    print(
        f"  on the three properties ({time.perf_counter() - t0:.2f} s): the "
        "mixture's mean and standard deviation, before and after"
    )
    for index, name in enumerate(PROPERTIES):
        before = _moments(prior_on, index)
        after = _moments(posterior_on, index)
        print(
            f"    {name:<16s} prior {before[0]:+.3f} +- {before[1]:.3f}, posterior "
            f"{after[0]:+.3f} +- {after[1]:.3f}, truth {true_properties[index]:+.3f}"
        )
        _, means, deviations = mixture_marginal(prior_on, index)
        print(
            "      the prior's components at "
            + ", ".join(f"{value:+.3f}" for value in np.round(means, 3) + 0.0)
            + f", each of standard deviation {deviations[0]:.3f}"
        )

    t0 = time.perf_counter()
    sweep = noise_sweep(problem, prior, truth, redraws=scale.redraws, rng=rng)
    average = sweep["weights"].mean(axis=1)
    print(
        f"the noise sweep ({time.perf_counter() - t0:.1f} s, {scale.redraws} "
        f"draws of the data at each level): the average weights, and the "
        f"effective number of candidates"
    )
    for factor, row in zip(NOISE_FACTORS, average):
        print(
            f"  noise x{factor:<5g} "
            + " ".join(f"{value:6.3f}" for value in row)
            + f"   {effective_number(row):.2f}"
        )
    print(
        "the weights each candidate's own travel times would earn, noise-free and "
        f"with no background, a row each: at the app's noise | at x{NOISE_FACTORS[0]:g}"
    )
    app = NOISE_FACTORS.index(1.0)
    for index, (row, low) in enumerate(
        zip(sweep["confusion"][app], sweep["confusion"][0])
    ):
        print(
            f"  {index + 1}: "
            + " ".join(f"{value:6.3f}" for value in row)
            + "  |"
            + " ".join(f"{value:6.3f}" for value in low)
        )

    candidates_figure(problem, prior, truth, true_index, outdir=outdir)
    weights_figure(prior, weights, sweep, true_index, outdir=outdir)
    marginals_figure(prior_on, posterior_on, true_properties, outdir=outdir)
    winner_figure(problem, truth, mean, winner, true_index, outdir=outdir)
    print(f"E took {time.perf_counter() - started:.1f} s")
    return {
        "angles": angles,
        "true_index": true_index,
        "coverage": counts,
        "truth": truth,
        "data": data,
        "prior": prior,
        "posterior": posterior,
        "log_evidence": terms,
        "misfits": np.array(misfits),
        "prior_weights": prior.weights,
        "weights": weights,
        "winner": winner,
        "mean": mean,
        "prior_on_properties": prior_on,
        "on_properties": posterior_on,
        "true_properties": true_properties,
        "sweep": sweep,
        "average_weights": average,
    }


def _moments(mixture: GaussianMixture, index: int, /) -> tuple[float, float]:
    """A mixture's mean and standard deviation in one coordinate of a
    Euclidean space, by the law of total variance."""
    weights, means, deviations = mixture_marginal(mixture, index)
    mean = float(weights @ means)
    spread = weights @ deviations**2 + weights @ (means - mean) ** 2
    return mean, float(np.sqrt(spread))


# --------------------------------------------------------------------------- #
#                                  Figures                                    #
# --------------------------------------------------------------------------- #


def _label_candidates(ax: Any, angles: Any, true_index: int, /) -> None:
    """Each candidate's conduit drawn as its axis, from the core-mantle
    boundary to the surface, dashed so that the field shows through, and
    numbered at its foot, inside the core: the true one in red, the others
    in their colors of the weights figure."""
    for index, axis in enumerate(angles):
        c, s = np.cos(np.radians(axis)), np.sin(np.radians(axis))
        tone = "red" if index == true_index else OTHERS[index % len(OTHERS)]
        ax.plot(
            [CORE_RADIUS * c, RADIUS * c],
            [CORE_RADIUS * s, RADIUS * s],
            color=tone,
            lw=1.2,
            ls=(0, (4, 2)),
        )
        ax.text(
            0.85 * CORE_RADIUS * c,
            0.85 * CORE_RADIUS * s,
            str(index + 1),
            ha="center",
            va="center",
            fontsize=9,
            color=tone,
            fontweight="bold",
        )


def candidates_figure(
    problem: Problem,
    prior: GaussianMixture,
    truth: Any,
    true_index: int,
    /,
    *,
    outdir: pathlib.Path,
) -> None:
    """The prior mixture's mean, a share of each plume, under the rays; and
    the truth, with the candidates' axes drawn."""
    space, scale = problem.space, problem.scale
    angles = scale.candidates
    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 4.8), subplot_kw={"aspect": "equal"}, layout="constrained"
    )
    draw(
        space,
        prior.expectation,
        axes[0],
        title=f"the prior mixture's mean, {1.0 / len(angles):.2g} of each plume, "
        "and the rays",
        limit=amplitude(scale) / len(angles),
    )
    draw_rays(axes[0], problem.rays, colors="black", linewidths=0.4, alpha=0.5)
    stations(axes[0], scale.sources, scale.receivers, radius=space.layers[-1].radius)
    draw(
        space,
        truth,
        axes[1],
        title=f"the truth: the plume at candidate {true_index + 1} over a small "
        "draw from the covariance",
        limit=amplitude(scale),
        label="slowness perturbation (%)",
    )
    for ax in axes:
        _label_candidates(ax, angles, true_index)
    save(
        fig,
        outdir / "mixture_candidates.png",
        title="Tomography, part E: the candidate conduits of a plume, under the "
        "rays, and the truth",
    )


def weights_figure(
    prior: GaussianMixture,
    weights: np.ndarray,
    sweep: dict,
    true_index: int,
    /,
    *,
    outdir: pathlib.Path,
) -> None:
    """The weights after against before; the average weights across the
    noise levels, with the spread of the true candidate's; and the weights
    each candidate's own travel times would earn."""
    count = len(prior)
    labels = [str(index + 1) for index in range(count)]
    fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 4.4), layout="constrained", width_ratios=[0.8, 1.3, 1.0]
    )
    ax = axes[0]
    positions = np.arange(count)
    ax.bar(positions - 0.2, prior.weights, width=0.4, color="0.7", label="prior")
    ax.bar(positions + 0.2, weights, width=0.4, color="C0", label="posterior")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("candidate")
    ax.set_ylabel("weight")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("the weights at the app's noise", fontsize=10)

    ax = axes[1]
    factors = sweep["factors"]
    every = sweep["weights"]
    average = every.mean(axis=1)
    low, high = np.percentile(every[:, :, true_index], [10.0, 90.0], axis=1)
    ax.fill_between(factors, low, high, color="red", alpha=0.15, lw=0)
    for index in range(count):
        ax.plot(
            factors,
            average[:, index],
            "o-",
            ms=4,
            color="red" if index == true_index else OTHERS[index % len(OTHERS)],
            label=f"candidate {index + 1}"
            + (", the truth" if index == true_index else ""),
        )
    ax.axhline(1.0 / count, color="0.5", ls=":", lw=1.0, label="the prior's weight")
    ax.axvline(1.0, color="black", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("noise, in multiples of the app's")
    ax.set_ylabel("posterior weight")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title(
        f"the average over {every.shape[1]} draws of the data at each level,\n"
        "the true candidate's 10% to 90% shaded",
        fontsize=10,
    )

    ax = axes[2]
    confusion = sweep["confusion"][NOISE_FACTORS.index(1.0)]
    image = ax.imshow(confusion, vmin=0.0, vmax=1.0, cmap="Blues")
    for i in range(count):
        for j in range(count):
            value = confusion[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.6 else "black",
            )
    ax.set_xticks(positions, labels)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("weight on candidate")
    ax.set_ylabel("the plume at candidate")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(
        "the weights each plume's own travel times earn,\nnoise-free and with no "
        "background, at the app's noise",
        fontsize=10,
    )
    save(
        fig,
        outdir / "mixture_weights.png",
        title="Tomography, part E: the weights, evidence ratios, against the prior's "
        "and against the noise level:\nthey sharpen as the noise falls until the "
        "prior's own variability, not the noise, is what limits them",
    )


def marginals_figure(
    prior_on: GaussianMixture,
    posterior_on: GaussianMixture,
    truth: np.ndarray,
    /,
    *,
    outdir: pathlib.Path,
) -> None:
    """Each property's marginal before and after, as the mixture's density."""
    _, means, deviations = mixture_marginal(prior_on, 0)
    shift = float((means.max() - np.median(means)) / deviations[0])
    fig, axes = plt.subplots(
        1, len(PROPERTIES), figsize=(13.2, 4.0), layout="constrained"
    )
    mixture_marginal_panels(
        axes,
        [
            f"mean over the {name}" + (" box" if name != "core" else "")
            for name in PROPERTIES
        ],
        posterior_on,
        prior_on,
        truth,
        xlabel="mean slowness perturbation (%)",
    )
    save(
        fig,
        outdir / "mixture_marginals.png",
        title="Tomography, part E: the prior and posterior mixtures on the three "
        "means, drawn as their densities over their weighted components:\na plume "
        f"in the well-covered box shifts its component of the box's mean by {shift:.1f} "
        "prior standard deviations, and the data remove it",
    )


def winner_figure(
    problem: Problem,
    truth: Any,
    mean: Any,
    winner: int,
    true_index: int,
    /,
    *,
    outdir: pathlib.Path,
) -> None:
    """The truth, the winning component's posterior mean, and the difference."""
    space, scale = problem.space, problem.scale
    fig, axes = plt.subplots(
        1, 3, figsize=(14.4, 4.6), subplot_kw={"aspect": "equal"}, layout="constrained"
    )
    limit = amplitude(scale)
    draw(space, truth, axes[0], title="the truth", limit=limit)
    draw(
        space,
        mean,
        axes[1],
        title=f"the posterior mean of the winner, candidate {winner + 1}",
        limit=limit,
    )
    draw(
        space,
        space.subtract(mean, truth),
        axes[2],
        title="the winner's mean less the truth",
        limit=limit,
        label="slowness perturbation (%)",
    )
    for ax in axes:
        _label_candidates(ax, scale.candidates, true_index)
    save(
        fig,
        outdir / "mixture_mean.png",
        title="Tomography, part E: the winning component's posterior mean, with "
        "the truth",
    )


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return mixture_answer(problem, rng=part_rng(scale.seed, "e"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
