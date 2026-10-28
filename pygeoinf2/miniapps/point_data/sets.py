"""Part E: the set answer.

The feasible property set of the three regional averages, with the prior
hardened to its 95% ball and the noise to its credible ball: support
brackets, whether the truth lies in the polytope they bound, and the
brackets against the posterior intervals of C and the point bars of D on
one plot.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker
from pygeoinf2.geometry.feasible import Primal

from bayesian import posterior_summary
from common import output_directory, parse, part_rng, save
from point import minimum_norm_on_averages
from problem import FULL, LEVEL, ORDER, QUICK, REGIONS, Problem, Scale, build
from solvers import coarse_identity, woodbury_family_solver

PRIMAL = Primal(tolerance=1e-3)


def set_answer(
    problem: Problem,
    /,
    *,
    rng: np.random.Generator,
    outdir: pathlib.Path,
    bayesian: dict | None = None,
    point: dict | None = None,
) -> dict:
    """The feasible brackets, and the comparison with the other two answers."""
    started = time.perf_counter()
    scale, space, counted, forward = (
        problem.scale,
        problem.space,
        problem.counted,
        problem.problem,
    )
    print(f"\n=== E. The set answer ({scale.name} scale) ===")
    if bayesian is None:
        bayesian = posterior_summary(
            problem, problem.prior, rng=rng, label="stationary prior"
        )
    if point is None:
        point = minimum_norm_on_averages(problem)

    ball = problem.prior.ambient_ball(level=LEVEL)
    noise_ball = forward.error_measure.ambient_ball(level=LEVEL)
    print(
        f"the prior hardened to its {LEVEL:.0%} ball: radius {ball.radius:.2f} in "
        f"H^{ORDER:g}; the truth's norm is {space.norm(problem.truth):.2f}, inside: "
        f"{ball.contains(problem.truth)}; the noise to its {LEVEL:.0%} ball of "
        f"radius {noise_ball.radius:.2f}"
    )
    # The family swept is ``A C A* + t R`` with ``C = r² I`` and ``R = ρ² I``
    # the two balls' shape operators; the surrogate uses the same on the
    # coarse sphere.
    solver = woodbury_family_solver(
        problem,
        prior_covariance=coarse_identity(problem, scale=ball.radius**2),
        prior_precision=coarse_identity(problem, scale=ball.radius**-2),
        noise_covariance=noise_ball.covariance,
        noise_precision=noise_ball.precision,
    )
    counted.reset()
    t0 = time.perf_counter()
    feasible = BackusGilbertParker(
        forward, problem.averages, ball, noise=noise_ball, solver=solver
    )(problem.data)
    print(f"{feasible}; support by {feasible.algorithms['support'][0]}")
    P = problem.averages.codomain
    directions = []
    for index in range(len(REGIONS)):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    # The first direction serially, so the family and its factorizations are
    # built once here rather than once per worker process.
    feasible.support_result(directions[0], via=PRIMAL)
    sweep = feasible.support_sweep(directions, via=PRIMAL, n_jobs=scale.n_jobs)
    brackets = np.zeros((len(REGIONS), 2))
    gaps = np.zeros(len(REGIONS))
    cost = (
        f"{counted.forward} forward, {counted.adjoint} adjoint applications"
        if scale.n_jobs is None
        else f"{scale.n_jobs} worker processes, whose applications are not counted"
    )
    print(
        f"support in {len(directions)} directions in {time.perf_counter() - t0:.1f} s "
        f"({cost}):"
    )
    print(
        f"  {'region':<16s} {'from':>8s} {'to':>8s} {'gap':>8s} {'truth':>8s} "
        f"{'posterior 95%':>22s}"
    )
    for index, (name, _) in enumerate(REGIONS):
        high, low = sweep.results[2 * index], sweep.results[2 * index + 1]
        brackets[index] = (-low.value, high.value)
        gaps[index] = max(high.value - high.inner, low.value - low.inner)
        m, s = bayesian["center"][index], bayesian["deviation"][index]
        print(
            f"  {name:<16s} {brackets[index, 0]:+8.3f} {brackets[index, 1]:+8.3f} "
            f"{gaps[index]:8.1e} {problem.true_averages[index]:+8.3f} "
            f"[{m - 1.96 * s:+8.3f}, {m + 1.96 * s:+8.3f}]"
        )
    polytope = feasible.polytope(directions, via=PRIMAL)
    inside = polytope.contains(problem.true_averages)
    print(
        f"the true averages lie in the polytope of those {len(directions)} "
        f"half-spaces: {inside}"
    )

    fig, axes = plt.subplots(1, len(REGIONS), figsize=(10.0, 3.6), layout="constrained")
    for index, (ax, (name, _)) in enumerate(zip(axes, REGIONS)):
        ax.errorbar(
            0,
            point["estimate"][index],
            yerr=2.0 * point["propagated"][index],
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
        m, s = bayesian["center"][index], bayesian["deviation"][index]
        ax.errorbar(
            2,
            m,
            yerr=1.96 * s,
            fmt="s",
            color="C2",
            capsize=4,
            label="measure, 95% interval",
        )
        ax.axhline(
            problem.true_averages[index], color="black", ls="--", lw=1.0, label="truth"
        )
        ax.set_xticks([0, 1, 2], ["point", "set", "measure"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("cap average")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("The same three averages, answered three ways")
    save(fig, outdir / "answers.png")
    print(f"E took {time.perf_counter() - started:.1f} s")
    return {"radius": ball.radius, "brackets": brackets, "gaps": gaps, "inside": inside}


def run(
    scale: Scale,
    /,
    *,
    seed: int,
    outdir: pathlib.Path,
    problem: Problem | None = None,
    bayesian: dict | None = None,
    point: dict | None = None,
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return set_answer(
        problem, rng=part_rng(seed, "e"), outdir=outdir, bayesian=bayesian, point=point
    )


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
