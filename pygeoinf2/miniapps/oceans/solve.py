"""Part B: the posterior mean three ways, counted and timed.

Conjugate gradients on the data-space normal operator ``A Q A* + R``: plain
and capped, with a randomized spectral deflation, and with the Woodbury form
of a surrogate on a coarse sphere. Every solve is serial so that the
application counts and the wall times compare.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import CGSolver, LinearGaussianInversion

from common import machine_name, output_directory, parse, part_rng, save
from problem import FULL, QUICK, Problem, Scale, build
from solvers import spectral_solver, woodbury_solver


def solve_three_ways(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """Run the three solvers, print the table and draw the residual histories."""
    started = time.perf_counter()
    scale, space, counted = problem.scale, problem.space, problem.counted
    print(f"\n=== B. The solve ({scale.name} scale) ===")
    print(
        f"data-space normal operator A Q A* + R on {len(problem.points)} dimensions; "
        f"every solve is serial so that the counts and the times compare"
    )
    print(f"machine: {machine_name()}, {os.cpu_count()} cpus")

    coarse, _ = problem.surrogate()
    rows: list[dict] = []
    means: dict[str, Any] = {}

    def run(label: str, solver: Any) -> None:
        counted.reset()
        t0 = time.perf_counter()
        inversion = LinearGaussianInversion(
            problem.problem, problem.prior, solver=solver
        )
        build_forward, build_adjoint = counted.forward, counted.adjoint
        built = time.perf_counter()
        # The one solve the posterior mean needs, done as the estimator does it
        # inside, but by hand so that its diagnostics are in view: the mean
        # is the prior's plus the Kalman update of the data residual.
        normal = inversion.normal_operator
        residual = problem.problem.data_space.subtract(
            problem.data, problem.problem.forward_operator(problem.prior.expectation)
        )
        result = inversion.inverse_normal_operator.solve(
            normal.right_hand_side(residual)
        )
        mean = space.add(
            problem.prior.expectation, normal.model_update(result.solution)
        )
        finished = time.perf_counter()
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

    run("plain CG (capped)", CGSolver(max_iterations=scale.cg_cap, strict=False))
    run(
        f"spectral deflation, rank {scale.spectral_rank}",
        spectral_solver(problem, rng=rng),
    )
    run(f"Woodbury surrogate, degree {coarse.lmax}", woodbury_solver(problem))

    print(
        f"\n{'solver':<34s} {'iterations':>10s} {'converged':>9s} "
        f"{'build A/A*':>11s} {'solve A/A*':>11s} {'build s':>8s} {'solve s':>8s} "
        f"{'rel. error':>10s}"
    )
    for row in rows:
        print(
            f"{row['label']:<34s} {row['iterations']:>10d} {str(row['converged']):>9s} "
            f"{row['build_forward']:>5d}/{row['build_adjoint']:<5d} "
            f"{row['solve_forward']:>5d}/{row['solve_adjoint']:<5d} "
            f"{row['build_time']:>8.1f} {row['solve_time']:>8.1f} "
            f"{row['error']:>10.3f}"
        )
    labels = [row["label"] for row in rows]
    reference = means[labels[2]]
    agreement = {}
    for label in labels[:2]:
        agreement[label] = space.norm(
            space.subtract(means[label], reference)
        ) / space.norm(reference)
        print(f"{label} against the Woodbury answer: {agreement[label]:.1e}")

    fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
    for row in rows:
        history = row["history"]
        ax.semilogy(history / history[0], label=row["label"], lw=1.4)
    ax.axhline(1e-8, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.set_title(
        f"The same solve three ways: {len(problem.points)} data, degree {space.lmax}"
    )
    ax.legend(frameon=False, fontsize=9)
    save(fig, outdir / "solve.png")
    print(f"B took {time.perf_counter() - started:.1f} s")
    return {"rows": rows, "agreement": agreement}


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return solve_three_ways(problem, rng=part_rng(seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
