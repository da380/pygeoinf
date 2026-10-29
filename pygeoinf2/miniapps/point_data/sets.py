"""Part E: the set answer.

The feasible property set of the three regional averages, with the prior
hardened to its 95% ball and the noise to its 95% ball: support brackets,
whether the truth lies in the polytope they bound, and each bracket against
the bound the prior's ball admits on its own, before any datum.

The prior's ball is ``{m : ||m|| <= r}`` in the space's own norm, with ``r``
the radius at which a draw from the prior lies inside with probability
0.95: the classical norm bound, with the prior calibrating the radius. A
draw's squared norm has the covariance's trace as its expectation, finite
for a trace-class covariance, so the radius has a limit however fine the
truncation, where the Mahalanobis ellipsoid's grows with it.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker
from pygeoinf2.geometry.feasible import Primal

from src.common import bound_panels, parse, part_rng, save
from src.problem import FULL, QUICK, REGIONS, Problem, Scale, build
from src.solvers import coarse_prior, woodbury_family_solver

# The primal walk with a looser tolerance than its default, and the kernel
# part of each direction taken at a ten-thousandth of the family's scale
# rather than a millionth: the solve there is the one most iterations go to,
# and every walk asked here ends above it, which the gaps confirm.
PRIMAL = Primal(tolerance=1e-3, floor=1e-4)


def prior_alone(prior_set: Any, properties: Any, /) -> np.ndarray:
    """The bound on each property that the prior's set admits with no data:
    its support function either way along the property's representer."""
    support = prior_set.support_function()
    P = properties.codomain
    bounds = np.zeros((P.dim, 2))
    for index in range(P.dim):
        q = P.basis_vector(index)
        bounds[index] = (
            -support(properties.adjoint(P.scale(-1.0, q))),
            support(properties.adjoint(q)),
        )
    return bounds


def hardened(problem: Problem, /) -> tuple[Any, Any, Any]:
    """The prior's ball, the noise's ball, and the Woodbury solver for the
    family they make.

    The prior's covariance is diagonal in the sphere's harmonics, so the
    ball's radius is read off its eigenvalues with no applications.
    """
    scale, forward = problem.scale, problem.problem
    level = scale.level
    ball = problem.prior.ambient_ball(level=level)
    noise_ball = forward.error_measure.ambient_ball(level=level)
    # The family swept is ``A A* r^2 + t R``, the shape operators of the two
    # balls being ``r^2 I`` and ``R``; the surrogate uses the coarse prior's
    # ball, its own radius being close enough.
    coarse = coarse_prior(problem).ambient_ball(level=level)
    solver = woodbury_family_solver(
        problem,
        prior_covariance=coarse.covariance,
        prior_precision=coarse.precision,
        noise_covariance=noise_ball.covariance,
        noise_precision=noise_ball.precision,
    )
    return ball, noise_ball, solver


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

    level = scale.level
    ball, noise_ball, solver = hardened(problem)
    gauge = (space.norm(problem.truth) / ball.radius) ** 2
    print(
        f"the prior hardened to its {level:.0%} ball ||m|| <= r, r = "
        f"{ball.radius:.2f}: the truth's gauge (||m|| / r)^2 is {gauge:.2f} of the "
        f"boundary's 1, inside: {ball.contains(problem.truth)}; the noise to its "
        f"{level:.0%} ball of radius {noise_ball.radius:.2f}"
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
    # In worker processes, each building the family and its factorizations
    # for its own directions.
    sweep = feasible.support_sweep(directions, via=PRIMAL, n_jobs=scale.n_jobs)
    brackets = np.zeros((len(REGIONS), 2))
    gaps = np.zeros(len(REGIONS))
    cost = (
        f"{counted.forward} forward, {counted.adjoint} adjoint applications"
        if scale.n_jobs is None
        else f"in {min(scale.n_jobs, len(directions))} worker processes, whose "
        "applications this tally does not see"
    )
    print(
        f"support in {len(directions)} directions in {time.perf_counter() - t0:.1f} s "
        f"({cost}):"
    )
    alone = prior_alone(ball, problem.averages)
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
        f"prior's {level:.0%} ball admits alone\nand the certified bracket the "
        "data leave, which they narrow least under Africa",
    )
    print(f"E took {time.perf_counter() - started:.1f} s")
    return {
        "radius": ball.radius,
        "gauge": gauge,
        "alone": alone,
        "brackets": brackets,
        "gaps": gaps,
        "inside": inside,
    }


def run(
    scale: Scale,
    /,
    *,
    outdir: pathlib.Path,
    problem: Problem | None = None,
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return set_answer(problem, rng=part_rng(scale.seed, "e"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
