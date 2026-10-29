"""Part E: the set answer.

The feasible property set of the three regional averages, with the prior
hardened to its 95% ball and the noise to its credible ball: support
brackets, whether the truth lies in the polytope they bound, and each
bracket against the bound the prior's ball admits on its own, before any
datum.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker
from pygeoinf2.geometry.feasible import Primal

from common import bound_panels, output_directory, parse, part_rng, save
from problem import FULL, LEVEL, ORDER, QUICK, REGIONS, Problem, Scale, build
from solvers import coarse_identity, woodbury_family_solver

PRIMAL = Primal(tolerance=1e-3)


def ball_alone(ball: Any, properties: Any, /) -> np.ndarray:
    """The bound on each property that the ball admits with no data: its
    support function either way along the property's representer."""
    support = ball.support_function()
    P = properties.codomain
    bounds = np.zeros((P.dim, 2))
    for index in range(P.dim):
        q = P.basis_vector(index)
        bounds[index] = (
            -support(properties.adjoint(P.scale(-1.0, q))),
            support(properties.adjoint(q)),
        )
    return bounds


def set_answer(
    problem: Problem,
    /,
    *,
    rng: np.random.Generator,
    outdir: pathlib.Path,
) -> dict:
    """The feasible brackets, against the bounds of the prior's ball alone."""
    started = time.perf_counter()
    scale, space, counted, forward = (
        problem.scale,
        problem.space,
        problem.counted,
        problem.problem,
    )
    print(f"\n=== E. The set answer ({scale.name} scale) ===")

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
    # built once before the threads share them.
    feasible.support_result(directions[0], via=PRIMAL)
    sweep = feasible.support_sweep(directions, via=PRIMAL, n_jobs=scale.n_jobs)
    brackets = np.zeros((len(REGIONS), 2))
    gaps = np.zeros(len(REGIONS))
    cost = f"{counted.forward} forward, {counted.adjoint} adjoint applications" + (
        "" if scale.n_jobs is None else f", in {scale.n_jobs} worker threads"
    )
    print(
        f"support in {len(directions)} directions in {time.perf_counter() - t0:.1f} s "
        f"({cost}):"
    )
    alone = ball_alone(ball, problem.averages)
    print(
        f"  {'region':<16s} {'from':>8s} {'to':>8s} {'gap':>8s} {'truth':>8s} "
        f"{'the ball alone':>22s}"
    )
    for index, (name, _) in enumerate(REGIONS):
        high, low = sweep.results[2 * index], sweep.results[2 * index + 1]
        brackets[index] = (-low.value, high.value)
        gaps[index] = max(high.value - high.inner, low.value - low.inner)
        print(
            f"  {name:<16s} {brackets[index, 0]:+8.3f} {brackets[index, 1]:+8.3f} "
            f"{gaps[index]:8.1e} {problem.true_averages[index]:+8.3f} "
            f"[{alone[index, 0]:+8.3f}, {alone[index, 1]:+8.3f}]"
        )
    polytope = feasible.polytope(directions, via=PRIMAL)
    inside = polytope.contains(problem.true_averages)
    print(
        f"the true averages lie in the polytope of those {len(directions)} "
        f"half-spaces: {inside}"
    )

    fig, axes = plt.subplots(1, len(REGIONS), figsize=(10.0, 3.8), layout="constrained")
    bound_panels(
        axes,
        [f"average over {name}" for name, _ in REGIONS],
        alone,
        brackets,
        problem.true_averages,
        ylabel="cap average",
    )
    save(
        fig,
        outdir / "brackets.png",
        title="Point data, part E: the set answer. For each average, the bound the "
        f"prior's {LEVEL:.0%} ball admits alone\nand the certified bracket the data "
        "leave, which they narrow least under Africa",
    )
    print(f"E took {time.perf_counter() - started:.1f} s")
    return {
        "radius": ball.radius,
        "alone": alone,
        "brackets": brackets,
        "gaps": gaps,
        "inside": inside,
    }


def run(
    scale: Scale,
    /,
    *,
    seed: int,
    outdir: pathlib.Path,
    problem: Problem | None = None,
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return set_answer(problem, rng=part_rng(seed, "e"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
