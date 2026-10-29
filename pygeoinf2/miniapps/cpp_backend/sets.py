"""Part D: the set answer.

The feasible property set of the two local averages under two constraint
sets the prior calibrates, with the noise hardened to its 95% ball:

- the prior's 95% ball in the space's own norm, ``{u : ||u|| <= r}`` with
  ``||u||^2 = u^T (M + l^2 K) u``, the radius ``r`` the one at which a draw
  lies inside with probability 0.95;
- its 95% ball in the ``L^2`` norm, ``{u : u^T M u <= r^2}``, an ellipsoid of
  ``H^1`` whose shape is the operator ``B = G^-1 M``, the mass inner product
  seen from the space.

For each, certified support brackets, whether the truth lies in the
polytope they bound, and each bracket against the bound the set admits
alone, before any datum; and that bound under refinement, which settles for
both, a draw's squared norm having a finite expectation in either. The space
has no coordinates, so each radius comes from a randomized
eigendecomposition taken to the full dimension: of the covariance for the
first, of ``L* B L`` through the prior's factor for the second.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker
from pygeoinf2.numerics import SpectralFamilySolver
from pygeoinf2.testing import check_operator, check_traits

from src.common import bound_panels, parse, part_rng, prior_alone, report, save
from src.problem import FULL, INTERVALS, QUICK, Problem, Scale, build
from src.space import CppSpace, l2_form, local_average_operator, matern_measure

# The meshes the bounds alone are followed through, as fractions of the
# scale's.
REFINEMENTS = (0.25, 0.5, 1.0)
# The two constraint sets, by the names the printout and the figure use.
H1, L2 = "the H^1 ball", "the L^2 ball"


def hardened(
    prior: Any, level: float, rng: np.random.Generator, /, *, l2: bool = False
) -> Any:
    """The prior's ball at a level, in the space's norm or in ``L^2``.

    The space has no coordinates, so the weights of the radius come from a
    randomized eigendecomposition to the full dimension: they decay too
    slowly for a truncated spectrum to give it. In ``L^2`` the form's
    inverse goes with it, so that the ellipsoid has a covariance and so a
    support function.
    """
    operator, inverse = l2_form(prior.domain) if l2 else (None, None)
    return prior.ambient_ball(
        level=level,
        operator=operator,
        inverse=inverse,
        method="spectral",
        rank=prior.domain.dim,
        rng=rng,
    )


def radius_of(prior_set: Any, form: Any, u: Any, /) -> float:
    """The radius of a ball about zero in the norm of ``form``: the ratio of
    ``(B u, u)`` to ``(P u, u)``, with ``P`` its precision ``B / r^2``,
    along any vector."""
    space = prior_set.domain
    return float(
        np.sqrt(
            space.inner_product(form(u), u)
            / space.inner_product(prior_set.precision(u), u)
        )
    )


def bounds_alone(
    scale: Scale, n_elements: int, rng: np.random.Generator, /
) -> tuple[float, float]:
    """The bound each prior set alone admits on the first average, on a mesh
    of its own."""
    space = CppSpace(n_elements, scale.space_length)
    prior = matern_measure(
        space, length_scale=scale.prior_length, pointwise_std=scale.prior_std
    )
    average = local_average_operator(space, INTERVALS[:1])
    first, second = (
        float(prior_alone(hardened(prior, scale.level, rng, l2=l2), average)[0, 1])
        for l2 in (False, True)
    )
    return first, second


def feasible_brackets(problem: Problem, prior_set: Any, noise_ball: Any, /) -> dict:
    """The certified brackets under one prior set, its bounds alone, and the
    truth against the polytope of the brackets."""
    P = problem.averages.codomain
    directions = []
    for index in range(len(INTERVALS)):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    feasible = BackusGilbertParker(
        problem.problem,
        problem.averages,
        prior_set,
        noise=noise_ball,
        solver=SpectralFamilySolver(),
    )(problem.data)
    sweep = feasible.support_sweep(directions)
    brackets = np.zeros((len(INTERVALS), 2))
    gaps = np.zeros(len(INTERVALS))
    for index in range(len(INTERVALS)):
        high, low = sweep.results[2 * index], sweep.results[2 * index + 1]
        brackets[index] = (-low.value, high.value)
        gaps[index] = max(high.value - high.inner, low.value - low.inner)
    return {
        "brackets": brackets,
        "gaps": gaps,
        "alone": prior_alone(prior_set, problem.averages),
        "inside": bool(feasible.polytope(directions).contains(problem.true_averages)),
        "algorithm": feasible.algorithms["support"][0],
    }


def set_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The feasible brackets under both prior sets, against the bounds of
    each set alone."""
    started = time.perf_counter()
    scale, space, truth = problem.scale, problem.space, problem.truth
    print(f"\n=== D. The set answer ({scale.name} scale) ===")
    space.calls.clear()

    # The L^2 form and its inverse, checked on the space: their adjoints in
    # the space's inner product, and the traits the ball relies on.
    t0 = time.perf_counter()
    form, inverse = l2_form(space)
    prior = problem.prior
    checks = {
        "the L^2 form": check_operator(form, rng=rng, measure=prior),
        "its traits": check_traits(form, rng=rng, measure=prior),
        "its inverse": check_operator(inverse, rng=rng, measure=prior),
        "the inverse's traits": check_traits(inverse, rng=rng, measure=prior),
    }
    print(
        f"the L^2 form B = G^-1 M and its inverse M^-1 G on the space, probes "
        f"drawn from the prior ({time.perf_counter() - t0:.1f} s):"
    )
    for name, result in checks.items():
        print(f"  {name:<21s} {result}")

    noise_ball = problem.noise.ambient_ball(level=scale.level)
    print(
        f"the noise hardened to its {scale.level:.0%} ball of radius "
        f"{noise_ball.radius:.3f}"
    )
    answers = {}
    for name, l2 in ((H1, False), (L2, True)):
        t0 = time.perf_counter()
        prior_set = hardened(prior, scale.level, rng, l2=l2)
        hardening = time.perf_counter() - t0
        radius = radius_of(prior_set, form, truth) if l2 else prior_set.radius
        gauge = space.inner_product(prior_set.precision(truth), truth)
        t0 = time.perf_counter()
        answer = feasible_brackets(problem, prior_set, noise_ball)
        answer.update(radius=radius, gauge=gauge)
        answers[name] = answer
        print(
            f"{name}, its spectrum to rank {space.dim} in {hardening:.1f} s: radius "
            f"{radius:.3f}, the truth's gauge {gauge:.3f} of the boundary's 1; support "
            f"in four directions by {answer['algorithm']}, "
            f"{time.perf_counter() - t0:.1f} s:"
        )
        print(
            f"  {'':<24s} {'from':>8s} {'to':>8s} {'gap':>8s} {'truth':>8s} "
            f"{'the ball alone':>21s}"
        )
        for label, (low, high), gap, t, (a0, a1) in zip(
            problem.names,
            answer["brackets"],
            answer["gaps"],
            problem.true_averages,
            answer["alone"],
        ):
            print(
                f"  {label:<24s} {low:+8.3f} {high:+8.3f} {gap:8.1e} {t:+8.3f} "
                f"[{a0:+8.3f}, {a1:+8.3f}]"
            )
        print(
            f"  the true averages lie in the polytope of the four half-spaces: "
            f"{answer['inside']}"
        )
    calls = report("the set answer", space.calls)

    # The bounds alone under refinement, each mesh a space of its own.
    t0 = time.perf_counter()
    meshes = [int(round(f * scale.n_elements)) for f in REFINEMENTS]
    refinement = [
        (
            n,
            *(
                (answers[H1]["alone"][0, 1], answers[L2]["alone"][0, 1])
                if n == scale.n_elements
                else bounds_alone(scale, n, rng)
            ),
        )
        for n in meshes
    ]
    print(
        f"the bound each ball alone admits on the {problem.names[0]}, under "
        f"refinement ({time.perf_counter() - t0:.1f} s):"
    )
    print(f"  {'elements':>8s} {'in H^1':>9s} {'in L^2':>9s}")
    for n, in_h1, in_l2 in refinement:
        print(f"  {n:8d} {in_h1:9.3f} {in_l2:9.3f}")

    fig, axes = plt.subplots(
        1, len(INTERVALS), figsize=(10.0, 4.2), layout="constrained"
    )
    bound_panels(
        axes,
        problem.names,
        [
            ("the prior's\n$H^1$ ball", answers[H1]["alone"], answers[H1]["brackets"]),
            ("the prior's\n$L^2$ ball", answers[L2]["alone"], answers[L2]["brackets"]),
        ],
        problem.true_averages,
        ylabel="local average",
    )
    save(
        fig,
        outdir / "brackets.png",
        title="C++ backend, part D: the set answer. For each average and each of the "
        f"prior's {scale.level:.0%} balls, in $H^1$ and in $L^2$,\nthe bound the "
        "ball admits alone and the certified bracket the data leave",
    )
    print(f"D took {time.perf_counter() - started:.1f} s")
    return {
        "checks": checks,
        "answers": answers,
        "refinement": refinement,
        "true_averages": problem.true_averages,
        "calls": calls,
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale)
    return set_answer(problem, rng=part_rng(scale.seed, "d"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
