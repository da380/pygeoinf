"""Part D: the set answer.

The feasible set of the three properties, with the prior hardened to its 95%
ball and the noise to its credible ball: certified support brackets in the
six coordinate directions, whether the truth lies in the polytope they
bound, and each bracket against the bound the prior's ball admits on its
own, before any datum. The searches work in the data space, where the family
``A C A* + t R`` is a few hundred square and full rank, and every member is
inverted by a Cholesky factorization.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker, CholeskySolver

from common import bound_panels, output_directory, parse, part_rng, save
from problem import FULL, LEVEL, ORDER, PROPERTIES, QUICK, Problem, Scale, build


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


def support_brackets(problem: Problem, /, *, names: Sequence[str] = PROPERTIES) -> dict:
    """The certified brackets on the properties, the bounds of the prior's
    ball alone, and whether the truth lies in the polytope of the brackets.

    Each answer is a bracket: an outer value with a dual certificate and an
    inner one attained by a feasible model, which in a direction no ray
    sees, the core's, is the smallest fitting model plus as much of the
    property's representer as the ball holds.
    """
    scale, space, forward = problem.scale, problem.space, problem.problem
    ball = problem.prior.ambient_ball(level=LEVEL)
    noise_ball = forward.error_measure.ambient_ball(level=LEVEL)
    print(
        f"the prior hardened to its {LEVEL:.0%} ball: radius {ball.radius:.2f} in "
        f"H^{ORDER:g}; the truth's norm is {space.norm(problem.truth):.2f}, inside: "
        f"{ball.contains(problem.truth)}; the noise to its {LEVEL:.0%} ball of "
        f"radius {noise_ball.radius:.3f}"
    )
    t0 = time.perf_counter()
    feasible = BackusGilbertParker(
        forward, problem.properties, ball, noise=noise_ball, solver=CholeskySolver()
    )(problem.data)
    print(f"{feasible}; support by {feasible.algorithms['support'][0]}")
    P = problem.properties.codomain
    directions = []
    for index in range(P.dim):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    sweep = feasible.support_sweep(directions, n_jobs=scale.n_jobs)
    workers = "" if scale.n_jobs is None else f", {scale.n_jobs} workers"
    print(
        f"support in {len(directions)} directions in {time.perf_counter() - t0:.1f} s"
        f"{workers}"
    )
    outer = np.array([result.value for result in sweep.results])
    inner = np.array([result.inner for result in sweep.results])
    brackets = np.zeros((P.dim, 2))
    gaps = np.zeros(P.dim)
    alone = ball_alone(ball, problem.properties)
    print(
        f"  {'mean over':<16s} {'from':>8s} {'to':>8s} {'gap':>8s} {'truth':>8s} "
        f"{'the ball alone':>20s}"
    )
    for index, name in enumerate(names):
        high, low = 2 * index, 2 * index + 1
        brackets[index] = (-outer[low], outer[high])
        gaps[index] = max(outer[high] - inner[high], outer[low] - inner[low])
        print(
            f"  {name:<16s} {brackets[index, 0]:+8.3f} {brackets[index, 1]:+8.3f} "
            f"{gaps[index]:8.1e} {problem.true_properties[index]:+8.3f} "
            f"[{alone[index, 0]:+7.2f}, {alone[index, 1]:+7.2f}]"
        )
    inside = feasible.polytope(directions).contains(problem.true_properties)
    print(
        f"the true properties lie in the polytope of those {len(directions)} "
        f"half-spaces: {inside}"
    )
    return {
        "radius": ball.radius,
        "truth_in_ball": bool(ball.contains(problem.truth)),
        "alone": alone,
        "brackets": brackets,
        "gaps": gaps,
        "inside": inside,
    }


def brackets_figure(
    problem: Problem,
    result: dict,
    /,
    *,
    names: Sequence[str],
    path: pathlib.Path,
    title: str,
) -> None:
    """For each property, the bound the prior's ball admits alone against
    the bracket the data leave, with the truth."""
    fig, axes = plt.subplots(1, len(names), figsize=(10.0, 4.0), layout="constrained")
    bound_panels(
        axes,
        names,
        result["alone"],
        result["brackets"],
        problem.true_properties,
        ylabel="mean slowness perturbation (%)",
    )
    save(fig, path, title=title)


def set_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The feasible brackets, against the bounds of the prior's ball alone."""
    started = time.perf_counter()
    print(f"\n=== D. The set answer ({problem.scale.name} scale) ===")
    result = support_brackets(problem)
    brackets_figure(
        problem,
        result,
        names=[
            f"mean over the {name}" + (" box" if name != "core" else "")
            for name in PROPERTIES
        ],
        path=outdir / "brackets.png",
        title="Tomography, part D: the set answer. For each mean, the bound the "
        f"prior's {LEVEL:.0%} ball admits alone\nand the certified bracket the "
        "data leave: narrow where the rays cross, barely narrowed in the core",
    )
    print(f"D took {time.perf_counter() - started:.1f} s")
    return result


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
    return set_answer(problem, rng=part_rng(seed, "d"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
