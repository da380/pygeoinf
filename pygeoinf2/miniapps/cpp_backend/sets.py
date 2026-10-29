"""Part D: the set answer.

The feasible property set of the two local averages, with the prior
hardened to its 95% ellipsoid and the noise to its 95% ball: certified
support brackets, whether the truth lies in the polytope they bound, and
each bracket against the bound the prior's ellipsoid admits on its own,
before any datum.

The ellipsoid is ``{u : (Q^-p u, u) <= r^2}``, the prior's covariance ``Q``
to a power ``p``, with ``r`` the radius that carries 95% of the prior. The
radius is a quantile of ``sum lambda^(1-p) Z^2`` over the covariance's
eigenvalues, which decay as the inverse square of the mode number on
``H^1``: the sum converges for ``p`` below one half, and the scale's power
is half of that. The space has no coordinates, so the spectrum comes from a
randomized eigendecomposition taken to the full dimension, and the
fractional powers of the covariance from the Lanczos calculus.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import BackusGilbertParker
from pygeoinf2.numerics import SpectralFamilySolver

from src.common import bound_panels, parse, part_rng, prior_alone, report, save
from src.problem import FULL, INTERVALS, QUICK, Problem, Scale, build
from src.space import CppSpace, local_average_operator, matern_measure

# The meshes the ellipsoid's bound is followed through, as fractions of the
# scale's, and the power at which its radius stops having a limit on H^1.
REFINEMENTS = (0.25, 0.5, 1.0)
LIMIT = 0.5


def hardened(prior: Any, level: float, power: float, rng: np.random.Generator) -> Any:
    """The prior's ellipsoid of a power at a level.

    The space has no coordinates, so the covariance's spectrum is taken by a
    randomized eigendecomposition to the full dimension: the weights of the
    radius decay too slowly for a truncated spectrum to give it.
    """
    return prior.weakened_ellipsoid(
        level=level,
        power=power,
        method="spectral",
        rank=prior.domain.dim,
        quantile_method="matched",
        rng=rng,
    )


def bound_alone(
    scale: Scale, n_elements: int, power: float, rng: np.random.Generator, /
) -> float:
    """The bound the prior's ellipsoid alone admits on the first average, on
    a mesh of its own."""
    space = CppSpace(n_elements, scale.space_length)
    prior = matern_measure(
        space, length_scale=scale.prior_length, pointwise_std=scale.prior_std
    )
    ellipsoid = hardened(prior, scale.level, power, rng)
    return float(
        prior_alone(ellipsoid, local_average_operator(space, INTERVALS[:1]))[0, 1]
    )


def set_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The feasible brackets, against the bounds of the prior's ellipsoid alone."""
    started = time.perf_counter()
    scale, space, forward = problem.scale, problem.space, problem.problem
    print(f"\n=== D. The set answer ({scale.name} scale) ===")
    space.calls.clear()
    t0 = time.perf_counter()
    ellipsoid = hardened(problem.prior, scale.level, scale.set_power, rng)
    noise_ball = problem.noise.ambient_ball(level=scale.level)
    gauge = space.inner_product(ellipsoid.precision(problem.truth), problem.truth)
    print(
        f"the prior hardened to its {scale.level:.0%} ellipsoid (Q^-p u, u) <= r^2 "
        f"with p = {scale.set_power:g}, its spectrum to rank {space.dim}, in "
        f"{time.perf_counter() - t0:.2f} s: the truth's gauge is {gauge:.3f} of the "
        f"boundary's 1; the noise to its {scale.level:.0%} ball of radius "
        f"{noise_ball.radius:.3f}"
    )
    t0 = time.perf_counter()
    feasible = BackusGilbertParker(
        forward,
        problem.averages,
        ellipsoid,
        noise=noise_ball,
        solver=SpectralFamilySolver(),
    )(problem.data)
    P = problem.averages.codomain
    directions = []
    for index in range(len(INTERVALS)):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    sweep = feasible.support_sweep(directions)
    brackets = np.zeros((len(INTERVALS), 2))
    gaps = np.zeros(len(INTERVALS))
    for index in range(len(INTERVALS)):
        high, low = sweep.results[2 * index], sweep.results[2 * index + 1]
        brackets[index] = (-low.value, high.value)
        gaps[index] = max(high.value - high.inner, low.value - low.inner)
    inside = bool(feasible.polytope(directions).contains(problem.true_averages))
    alone = prior_alone(ellipsoid, problem.averages)
    print(
        f"support in {len(directions)} directions by "
        f"{feasible.algorithms['support'][0]}, {time.perf_counter() - t0:.1f} s:"
    )
    print(
        f"  {'':<24s} {'from':>8s} {'to':>8s} {'gap':>8s} {'truth':>8s} "
        f"{'the ellipsoid alone':>21s}"
    )
    for name, (low, high), gap, t, (a0, a1) in zip(
        problem.names, brackets, gaps, problem.true_averages, alone
    ):
        print(
            f"  {name:<24s} {low:+8.3f} {high:+8.3f} {gap:8.1e} {t:+8.3f} "
            f"[{a0:+8.3f}, {a1:+8.3f}]"
        )
    print(
        f"the true averages lie in the polytope of those {len(directions)} "
        f"half-spaces: {inside}"
    )
    calls = report("the set answer", space.calls)

    # The bound alone under refinement, at the scale's power and at the
    # power where the radius stops converging, each mesh a space of its own.
    t0 = time.perf_counter()
    meshes = [int(round(f * scale.n_elements)) for f in REFINEMENTS]
    refinement = [
        (
            n,
            alone[0, 1]
            if n == scale.n_elements
            else bound_alone(scale, n, scale.set_power, rng),
            bound_alone(scale, n, LIMIT, rng),
        )
        for n in meshes
    ]
    print(
        f"the bound the ellipsoid alone admits on the {problem.names[0]}, under "
        f"refinement ({time.perf_counter() - t0:.1f} s):"
    )
    print(f"  {'elements':>8s} {f'p = {scale.set_power:g}':>9s} {f'p = {LIMIT:g}':>9s}")
    for n, at_power, at_limit in refinement:
        print(f"  {n:8d} {at_power:9.3f} {at_limit:9.3f}")

    fig, axes = plt.subplots(
        1, len(INTERVALS), figsize=(8.0, 3.8), layout="constrained"
    )
    bound_panels(
        axes,
        problem.names,
        alone,
        brackets,
        problem.true_averages,
        ylabel="local average",
    )
    save(
        fig,
        outdir / "brackets.png",
        title="C++ backend, part D: the set answer. For each average, the bound the "
        f"prior's {scale.level:.0%} ellipsoid\nadmits alone and the certified "
        "bracket the data leave",
    )
    print(f"D took {time.perf_counter() - started:.1f} s")
    return {
        "gauge": gauge,
        "brackets": brackets,
        "gaps": gaps,
        "alone": alone,
        "inside": inside,
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
