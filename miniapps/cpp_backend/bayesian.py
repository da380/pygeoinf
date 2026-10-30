"""Part C: the Bayesian answer.

The posterior under the prior on ``H^1``: its mean, its pointwise standard
deviation, a few samples, and the posterior on the two local averages with
their credible intervals, each against the prior's. The space has no
pointwise product, so the standard deviation is read off the covariance of
the measure pushed through evaluation at every node, before and after.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import LinearGaussianInversion, plotting
from pygeoinf.numerics import SpectralFamilySolver

from src.common import parse, part_rng, report, save
from src.problem import FULL, INTERVALS, QUICK, Problem, Scale, build
from src.space import point_evaluation_operator


def pointwise_std(measure, space, /) -> np.ndarray:
    """The standard deviation at every node: the measure pushed through
    evaluation there, and the diagonal of that covariance."""
    at_nodes = point_evaluation_operator(space, space.nodes())
    return np.sqrt(measure.affine_map(at_nodes).covariance.diagonals()[0])


def bayesian_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The posterior against the prior, with its figure."""
    started = time.perf_counter()
    scale, space, forward = problem.scale, problem.space, problem.problem
    prior = problem.prior
    print(f"\n=== C. The Bayesian answer ({scale.name} scale) ===")
    space.calls.clear()
    t0 = time.perf_counter()
    posterior = LinearGaussianInversion(forward, prior, solver=SpectralFamilySolver())(
        problem.data
    )
    mean = posterior.expectation
    error = problem.relative_error(mean)
    print(
        f"the posterior mean in {time.perf_counter() - t0:.2f} s: relative error "
        f"{error:.3f} in L^2, chi-squared {forward.chi_squared(mean, problem.data):.2f} "
        f"on {scale.n_points} data"
    )
    t0 = time.perf_counter()
    after = pointwise_std(posterior, space)
    before = pointwise_std(prior, space)
    near = np.abs(np.subtract.outer(space.nodes(), problem.points)).min(axis=1)
    observed = near < 0.25 * scale.prior_length
    print(
        f"the pointwise standard deviation at {space.n_elements + 1} nodes in "
        f"{time.perf_counter() - t0:.2f} s: the prior's {before.min():.3f} to "
        f"{before.max():.3f}; the posterior's {after[observed].max():.3f} at most "
        f"within a quarter of the prior's length of a point, "
        f"{after.max():.3f} at most anywhere"
    )
    samples = posterior.samples(3, rng=rng)
    prior_samples = prior.samples(3, rng=rng)
    on_averages = posterior.affine_map(problem.averages)
    prior_averages = prior.affine_map(problem.averages)
    center = np.asarray(on_averages.expectation)
    deviation = np.sqrt(on_averages.covariance.diagonals()[0])
    prior_deviation = np.sqrt(prior_averages.covariance.diagonals()[0])
    print("the averages:")
    print(
        f"  {'':<24s} {'mean':>8s} {'std':>7s} {'95% from':>9s} {'to':>8s} "
        f"{'prior std':>10s} {'truth':>8s}"
    )
    for name, m, s, s0, t in zip(
        problem.names, center, deviation, prior_deviation, problem.true_averages
    ):
        print(
            f"  {name:<24s} {m:+8.3f} {s:7.3f} {m - 1.96 * s:+9.3f} "
            f"{m + 1.96 * s:+8.3f} {s0:10.3f} {t:+8.3f}"
        )
    calls = report("the Bayesian answer", space.calls)

    fig = plt.figure(figsize=(12.0, 7.8), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[1.0, 0.85])
    left, right = upper.subplots(1, 2, sharey=True)
    x = np.array(space.nodes())
    data = problem.evaluate.codomain.to_components(problem.data)
    for ax, center_values, std, draws, title in (
        (left, np.zeros_like(x), before, prior_samples, "the prior"),
        (right, space.values(mean), after, samples, "the posterior"),
    ):
        for a, b in INTERVALS:
            ax.axvspan(a, b, color="0.92", lw=0)
        ax.fill_between(
            x,
            center_values - 2 * std,
            center_values + 2 * std,
            color="C0",
            alpha=0.2,
            lw=0,
            label="2 pointwise std",
        )
        for k, draw in enumerate(draws):
            ax.plot(
                x,
                space.values(draw),
                color="C0",
                lw=0.8,
                alpha=0.6,
                label="samples" if k == 0 else None,
            )
        ax.plot(x, center_values, color="C0", lw=2, label="mean")
        ax.plot(x, space.values(problem.truth), color="C1", lw=1.5, label="truth")
        ax.set_xlim(0, 1)
        ax.set_xlabel("x")
        ax.set_title(title, fontsize=10)
    right.errorbar(
        problem.points,
        data,
        yerr=scale.noise_std,
        fmt="o",
        color="black",
        ms=4,
        capsize=2,
        lw=1,
        label="data",
    )
    left.set_ylabel("u(x)")
    left.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    right.legend(fontsize=8, frameon=False, loc="lower left", ncol=3)
    for index, ax in enumerate(lower.subplots(1, len(INTERVALS))):
        plotting.plot_densities(
            on_averages,
            prior=prior_averages,
            truth=float(problem.true_averages[index]),
            index=index,
            ax=ax,
            labels=["posterior"],
            prior_labels=["prior"],
            xlabel="local average",
            title=problem.names[index],
        )
    save(
        fig,
        outdir / "posterior.png",
        title="C++ backend, part C: the Bayesian answer, prior against posterior. "
        "Mean, two pointwise standard deviations\nand samples before and after, and "
        "each average's marginal before and after",
    )
    print(f"C took {time.perf_counter() - started:.1f} s")
    return {
        "mean": mean,
        "error": error,
        "chi_squared": forward.chi_squared(mean, problem.data),
        "before": before,
        "after": after,
        "observed": observed,
        "center": center,
        "deviation": deviation,
        "prior_deviation": prior_deviation,
        "true_averages": problem.true_averages,
        "calls": calls,
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale)
    return bayesian_answer(problem, rng=part_rng(scale.seed, "c"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
