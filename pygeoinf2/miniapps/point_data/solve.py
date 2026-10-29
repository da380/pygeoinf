"""Part B: the linear algebra behind every answer.

One solve underlies every method here: conjugate gradients on the data-space
normal operator ``A Q A* + R``, which the Bayesian answer inverts once, and
whose damped relatives ``A C A* + t R`` the point and set answers sweep. At
this scale the plain solve does not finish, and the part is about what makes
it finish: the Woodbury form of a surrogate on a coarse sphere, factorized
once, as the preconditioner. Plain and capped against preconditioned, with
the iteration counts, the application counts and the wall times side by side.
Every solve is serial so that the counts and the times compare.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import CGSolver, LinearGaussianInversion
from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner
from pygeoinf2.numerics.solvers import CholeskySolver

from src.common import machine_name, parse, part_rng, save
from src.problem import FULL, QUICK, Problem, Scale, build
from src.solvers import coarse_prior


def the_solve(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The one solve, plain and preconditioned; the table and the residual histories."""
    started = time.perf_counter()
    scale, space, counted = problem.scale, problem.space, problem.counted
    print(f"\n=== B. The linear algebra ({scale.name} scale) ===")
    print(
        f"data-space normal operator A Q A* + R on {len(problem.points)} dimensions, "
        f"never formed; every solve is serial so that the counts and the times compare"
    )
    print(f"machine: {machine_name()}, {os.cpu_count()} cpus")

    inversion = LinearGaussianInversion(problem.problem, problem.prior)
    normal = inversion.normal_operator
    residual = problem.problem.data_space.subtract(
        problem.data, problem.problem.forward_operator(problem.prior.expectation)
    )
    right_hand_side = normal.right_hand_side(residual)

    coarse, coarse_forward = problem.surrogate()
    surrogate_prior = coarse_prior(problem)

    def plain() -> Any:
        return CGSolver(max_iterations=scale.cg_cap, strict=False)

    def woodbury() -> Any:
        # Built here, and applied once, so that the factorization on the
        # coarse space is paid for in the build column and not in the first
        # iteration of the solve.
        surrogate = normal.surrogate(forward=coarse_forward, prior=surrogate_prior)
        preconditioner = WoodburyPreconditioner.from_normal(
            surrogate, solver=CholeskySolver()
        )(normal)
        preconditioner(right_hand_side)
        return CGSolver().with_preconditioner(preconditioner)

    rows: list[dict] = []
    means: dict[str, Any] = {}

    def run(label: str, make_solver: Callable[[], Any]) -> None:
        counted.reset()
        t0 = time.perf_counter()
        solver = make_solver()
        build_forward, build_adjoint = counted.forward, counted.adjoint
        built = time.perf_counter()
        result = solver(normal).solve(right_hand_side)
        finished = time.perf_counter()
        mean = space.add(
            problem.prior.expectation, normal.model_update(result.solution)
        )
        rows.append(
            {
                "label": label,
                "iterations": result.iterations,
                "converged": result.converged,
                "build_forward": build_forward,
                "build_adjoint": build_adjoint,
                "solve_forward": counted.forward - build_forward,
                "solve_adjoint": counted.adjoint - build_adjoint,
                "build_time": built - t0,
                "solve_time": finished - built,
                "history": np.asarray(result.history),
                "error": problem.relative_error(mean),
            }
        )
        means[label] = mean

    run(f"plain CG, capped at {scale.cg_cap}", plain)
    run(f"Woodbury surrogate, degree {coarse.lmax}", woodbury)

    print(
        f"\n{'solver':<30s} {'iterations':>10s} {'converged':>9s} "
        f"{'build A/A*':>11s} {'solve A/A*':>11s} {'build s':>8s} {'solve s':>8s} "
        f"{'rel. error':>10s}"
    )
    for row in rows:
        print(
            f"{row['label']:<30s} {row['iterations']:>10d} {str(row['converged']):>9s} "
            f"{row['build_forward']:>5d}/{row['build_adjoint']:<5d} "
            f"{row['solve_forward']:>5d}/{row['solve_adjoint']:<5d} "
            f"{row['build_time']:>8.1f} {row['solve_time']:>8.1f} "
            f"{row['error']:>10.3f}"
        )
    plain_label, woodbury_label = (row["label"] for row in rows)
    agreement = space.norm(
        space.subtract(means[plain_label], means[woodbury_label])
    ) / space.norm(means[woodbury_label])
    print(f"the plain answer against the preconditioned one: {agreement:.1e}")

    fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
    for row in rows:
        history = row["history"]
        ax.semilogy(history / history[0], label=row["label"], lw=1.4)
    ax.axhline(1e-8, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.set_title(f"{len(problem.points)} data, degree {space.lmax}", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    save(
        fig,
        outdir / "solve.png",
        title="Point data, part B: the one solve behind every answer, plain and "
        "preconditioned",
    )
    print(f"B took {time.perf_counter() - started:.1f} s")
    return {"rows": rows, "agreement": agreement}


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return the_solve(problem, rng=part_rng(scale.seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
