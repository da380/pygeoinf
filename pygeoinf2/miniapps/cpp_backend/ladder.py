"""Part A: the backend, its space and its check ladder.

The space is ``H^1`` on ``[0, 1]`` over ``cppvec.Vector``, a C++ vector with
no buffer, no indexing and no arithmetic, and the library can do nothing
with it but call the space. This part shows why the space is ``H^1``: the
norm of a value at a point settles under refinement there and grows as the
root of the number of elements in ``L^2``, the same space at length zero.
Then the ladder: the space's axioms, its white noise, the forward and
property operators, and the prior on the space, its covariance, factor,
precision and precision factor, and its samples against its covariance.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import CoordinateSpace
from pygeoinf2.testing import (
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

from src.common import parse, part_rng, report, save
from src.problem import FULL, INTERVALS, QUICK, Problem, Scale, build
from src.space import CppSpace, point_evaluation_operator

# Where the representer of a value is drawn, and the mesh refinements the
# norm of that value is followed through, as multiples of the scale's mesh.
POINT = 0.5
REFINEMENTS = (0.25, 0.5, 1, 2, 4)


def evaluation_norm(n_elements: int, length_scale: float, /) -> tuple[float, list]:
    """The norm of the value at ``POINT`` on a mesh, under the inner product
    of a length, and its representer's nodal values."""
    space = CppSpace(n_elements, length_scale)
    evaluate = point_evaluation_operator(space, [POINT])
    representer = evaluate.adjoint(evaluate.codomain.from_components(np.ones(1)))
    return space.norm(representer), space.values(representer)


def ladder(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The space, the refinement of a value's norm, and the check ladder."""
    started = time.perf_counter()
    scale, space, prior = problem.scale, problem.space, problem.prior
    print(f"\n=== A. The backend and its check ladder ({scale.name} scale) ===")
    vector = space.zero()
    print(
        f"{space!r}: a CoordinateSpace? {isinstance(space, CoordinateSpace)}; "
        f"vectors are {type(vector).__name__} from {type(vector).__module__}"
    )
    space.calls.clear()

    print(f"the norm of the value at x = {POINT} under refinement:")
    print(f"  {'elements':>8s} {'in H^1':>9s} {'in L^2':>9s}")
    norms = []
    for factor in REFINEMENTS:
        n = int(round(factor * scale.n_elements))
        h1, _ = evaluation_norm(n, scale.space_length)
        l2, _ = evaluation_norm(n, 0.0)
        norms.append((n, h1, l2))
        print(f"  {n:8d} {h1:9.4f} {l2:9.2f}")

    t0 = time.perf_counter()
    reports = {
        "space": check_space(
            space,
            rng=rng,
            measure=prior,
            rebuild=lambda: CppSpace(scale.n_elements, scale.space_length),
        ),
        "white noise": check_white_noise(space, rng=rng, measure=prior),
        "point evaluation": check_operator(problem.evaluate, rng=rng, measure=prior),
        "local averages": check_operator(problem.averages, rng=rng, measure=prior),
        "covariance": check_operator(prior.covariance, rng=rng, measure=prior),
        "covariance traits": check_traits(prior.covariance, rng=rng, measure=prior),
        "factor": check_operator(prior.covariance_factor, rng=rng, measure=prior),
        "precision": check_operator(prior.precision, rng=rng, measure=prior),
        "precision traits": check_traits(prior.precision, rng=rng, measure=prior),
        "precision factor": check_operator(
            prior.precision_factor, rng=rng, measure=prior
        ),
        "measure": check_measure(prior, rng=rng),
    }
    draw = prior.sample(rng=rng)
    inverses = {
        "the precision after the covariance": space.norm(
            space.subtract(prior.precision(prior.covariance(draw)), draw)
        )
        / space.norm(draw),
        "the precision factor after the factor": space.norm(
            space.subtract(prior.precision_factor(prior.covariance_factor(draw)), draw)
        )
        / space.norm(draw),
    }
    print(
        f"the check ladder, probes drawn from the prior ({time.perf_counter() - t0:.1f} s):"
    )
    for name, result in reports.items():
        print(f"  {name:<20s} {result}")
    for name, defect in inverses.items():
        print(f"  {name} returns a draw to {defect:.1e}")
    calls = report("the ladder", space.calls)

    # The problem, and the representer of a value on two meshes, in H^1 and
    # in L^2.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), layout="constrained")
    x = np.array(space.nodes())
    ax = axes[0]
    for a, b in INTERVALS:
        ax.axvspan(a, b, color="0.92", lw=0)
    for k, sample in enumerate(prior.samples(3, rng=rng)):
        ax.plot(
            x,
            space.values(sample),
            color="0.6",
            lw=0.8,
            label="prior draws" if k == 0 else None,
        )
    ax.plot(x, space.values(problem.truth), color="C1", lw=2, label="truth")
    ax.errorbar(
        problem.points,
        problem.evaluate.codomain.to_components(problem.data),
        yerr=scale.noise_std,
        fmt="o",
        color="black",
        ms=4,
        capsize=2,
        lw=1,
        label="data",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("the truth, a draw from the prior, and the data", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    meshes = (scale.n_elements, 4 * scale.n_elements)
    for ax, length, where in (
        (axes[1], scale.space_length, "in $H^1$"),
        (axes[2], 0.0, "in $L^2$ (length zero)"),
    ):
        for n, style in zip(meshes, ("-", "--")):
            norm, values = evaluation_norm(n, length)
            ax.plot(
                np.linspace(0.0, 1.0, n + 1),
                values,
                style,
                lw=1.5,
                label=f"{n} elements, norm {norm:.3g}",
            )
        ax.set_xlim(0, 1)
        ax.set_xlabel("x")
        ax.set_title(
            f"the representer of the value at x = {POINT}, {where}", fontsize=10
        )
        ax.legend(fontsize=8, frameon=False)
    save(
        fig,
        outdir / "ladder.png",
        title="C++ backend, part A: the problem on $H^1$. The truth and the data, and "
        "the representer of a value at a point,\nwhich settles under refinement in "
        "$H^1$ and grows without bound in $L^2$",
    )
    print(f"A took {time.perf_counter() - started:.1f} s")
    return {"norms": norms, "reports": reports, "inverses": inverses, "calls": calls}


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale)
    return ladder(problem, rng=part_rng(scale.seed, "a"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
