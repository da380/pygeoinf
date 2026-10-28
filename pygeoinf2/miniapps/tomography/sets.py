"""Part D: the set answer.

The feasible set of the three properties, with the prior hardened to its 95%
ball and the noise to its credible ball: certified support brackets in the
six coordinate directions, whether the truth lies in the polytope they
bound, and the brackets against the posterior intervals of B and the point
bars of C on one plot. The searches work in the data space, where the family
``A C A* + t R`` is a few hundred square and full rank, and every member is
inverted by a Cholesky factorization.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker, CholeskySolver

from bayesian import posterior, property_posterior
from common import output_directory, parse, part_rng, save
from point import error_budget, minimum_norm
from problem import FULL, LEVEL, ORDER, PROPERTIES, QUICK, Problem, Scale, build

#: A bracket wider than this, relative to its outer value, is loose.
GAP = 1e-3


def smallest_fitting_model(problem: Problem, /) -> Any:
    """The smallest model whose misfit is at the noise ball's edge, the
    damping found to roundoff."""
    return minimum_norm(problem, rtol=1e-12).for_data(problem.data)(problem.data)


def unseen_end(
    problem: Problem, ball: Any, noise_ball: Any, smallest: Any, direction: Any, /
) -> tuple[float, float, float] | None:
    """An inner end for a direction the data cannot see, attained by a model.

    Where the forward operator sends the property's representer to zero, as
    it does the core's mean, the models of the set that reach furthest along
    the direction are the smallest one that fits the data, which spends the
    least of the ball, plus as much of the representer as the rest of the
    ball holds. That model is built here and measured against both
    constraints.

    Returns:
        The value it attains, its distance from the ball's center over the
        radius, and its residual over the noise ball's radius; or None if the
        data see the direction or the smallest model leaves no room.
    """
    space = problem.space
    representer = problem.properties.adjoint(direction)
    seen = problem.forward(representer)
    if np.linalg.norm(seen) > 1e-12 * space.norm(representer):
        return None
    offset = space.subtract(smallest, ball.center)
    room = ball.radius**2 - space.squared_norm(offset)
    if room <= 0.0:
        return None
    step = np.sqrt(room) / space.norm(representer)
    model = space.axpy(step, representer, space.copy(smallest))
    value = float(
        problem.properties.codomain.inner_product(direction, problem.properties(model))
    )
    residual = problem.data - problem.forward(model)
    return (
        value,
        space.norm(space.subtract(model, ball.center)) / ball.radius,
        float(np.linalg.norm(residual)) / noise_ball.radius,
    )


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
    scale, space, forward = problem.scale, problem.space, problem.problem
    print(f"\n=== D. The set answer ({scale.name} scale) ===")
    if bayesian is None:
        bayesian = property_posterior(problem, posterior(problem))
    if point is None:
        chosen = minimum_norm(problem).for_data(problem.data)
        point = error_budget(problem, chosen)

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
    for index in range(len(PROPERTIES)):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    sweep = feasible.support_sweep(directions, n_jobs=scale.n_jobs)
    workers = "" if scale.n_jobs is None else f", {scale.n_jobs} workers"
    print(
        f"support in {len(directions)} directions in {time.perf_counter() - t0:.1f} s"
        f"{workers}"
    )
    # Each answer is a bracket: an outer value with a dual certificate and an
    # inner one attained by a feasible model. Where the inner end is loose
    # and the data cannot see the direction, the model that attains it is
    # built directly.
    outer = np.array([result.value for result in sweep.results])
    inner = np.array([result.inner for result in sweep.results])
    smallest = None
    for index, direction in enumerate(directions):
        if outer[index] - inner[index] <= GAP * max(1.0, abs(outer[index])):
            continue
        if smallest is None:
            # The smallest model that fits, its damping found to roundoff so
            # that it lies on the noise ball and not a search tolerance
            # outside it.
            smallest = smallest_fitting_model(problem)
        end = unseen_end(problem, ball, noise_ball, smallest, direction)
        message = (
            f"  direction {index}: the {sweep.results[index].algorithm} bracket "
            f"[{inner[index]:+.4f}, {outer[index]:+.4f}] is loose"
        )
        if end is None:
            print(message)
            continue
        value, norm, misfit = end
        print(
            f"{message}; the data cannot see it, and the smallest fitting model "
            f"plus the property's representer reaches {value:+.4f} at "
            f"{norm:.6f} of the ball's radius and {misfit:.6f} of the noise's"
        )
        if norm <= 1.0 + 1e-9 and misfit <= 1.0 + 1e-9:
            inner[index] = max(inner[index], value)
    brackets = np.zeros((len(PROPERTIES), 2))
    gaps = np.zeros(len(PROPERTIES))
    print(
        f"  {'mean over':<16s} {'from':>8s} {'to':>8s} {'gap':>8s} {'truth':>8s} "
        f"{'posterior 95%':>20s}"
    )
    for index, name in enumerate(PROPERTIES):
        high, low = 2 * index, 2 * index + 1
        brackets[index] = (-outer[low], outer[high])
        gaps[index] = max(outer[high] - inner[high], outer[low] - inner[low])
        m, s = bayesian["center"][index], bayesian["deviation"][index]
        print(
            f"  {name:<16s} {brackets[index, 0]:+8.3f} {brackets[index, 1]:+8.3f} "
            f"{gaps[index]:8.1e} {problem.true_properties[index]:+8.3f} "
            f"[{m - 1.96 * s:+7.3f}, {m + 1.96 * s:+7.3f}]"
        )
    polytope = feasible.polytope(directions)
    inside = polytope.contains(problem.true_properties)
    print(
        f"the true properties lie in the polytope of those {len(directions)} "
        f"half-spaces: {inside}"
    )

    fig, axes = plt.subplots(
        1, len(PROPERTIES), figsize=(10.0, 4.0), layout="constrained"
    )
    for index, (ax, name) in enumerate(zip(axes, PROPERTIES)):
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
            problem.true_properties[index],
            color="black",
            ls="--",
            lw=1.0,
            label="truth",
        )
        ax.set_xticks([0, 1, 2], ["point", "set", "measure"])
        ax.set_xlim(-0.6, 2.6)
        # Linear within one percent, logarithmic beyond: the brackets are
        # tens of times wider than the intervals.
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(
            f"mean over the {name}" + (" box" if name != "core" else ""), fontsize=10
        )
    axes[0].set_ylabel("mean slowness perturbation (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncols=4)
    fig.suptitle("The same three properties, answered three ways")
    save(fig, outdir / f"{problem.tag}answers.png")
    print(f"D took {time.perf_counter() - started:.1f} s")
    return {
        "radius": ball.radius,
        "truth_in_ball": bool(ball.contains(problem.truth)),
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
    bayesian: dict | None = None,
    point: dict | None = None,
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return set_answer(
        problem, rng=part_rng(seed, "d"), outdir=outdir, bayesian=bayesian, point=point
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
