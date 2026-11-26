"""Part B: a point answer, the most probable masses and lengths.

The unknowns are the free coordinates of the four parameters, the
logarithms of the masses and lengths, under a Gaussian prior about a
nominal pendulum. The maximum a posteriori model minimizes the misfit to
the readings plus the prior's penalty, and is found by a Newton method
from the prior's expectation. This part shows

* the search with the exact Hessian, which takes the solution's second
  derivative, beside the Gauss-Newton one, which drops it;
* the estimate against the truth, with its two errors. The estimator's
  derivative in the data carries the readings' noise to the parameters;
  and the estimate from readings with no noise at all is still not the
  truth, by a bias the prior puts there, which only the truth reveals.
  Together they are the error of a point estimate, and not a posterior;
* why a longer or a livelier record is harder: the misfit along the
  straight line from the prior's expectation to the truth, which has one
  valley from the release the data come from and many from a livelier one,
  and the same search run on the livelier record.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import ForwardProblem, MaximumAPosteriori
from pygeoinf.numerics.optimization import NewtonCG

from src.common import log_normal_interval, parameter_panels, parse, part_rng, save
from src.problem import FULL, QUICK, Problem, Scale, build

# The search on the livelier record is stopped here: by then it has shown
# where it is going.
LIVELY_ITERATIONS = 40
# Where along the line from the prior's expectation, 0, to the truth, 1.
LINE = np.linspace(-0.5, 1.5, 201)


def search(
    problem: Problem,
    /,
    *,
    gauss_newton: bool,
    data: object = None,
    max_iterations: int | None = None,
) -> dict:
    """The mode for the problem's data, or for others, and how the search went.

    With ``max_iterations`` the search is stopped there if it has not ended.
    """
    forward_problem = ForwardProblem(problem.forward, error=problem.noise)
    prior = problem.parameter_prior
    optimizer = None
    if max_iterations is not None:
        optimizer = NewtonCG(
            preconditioner=prior.covariance, max_iterations=max_iterations
        )
    estimator = MaximumAPosteriori(
        forward_problem, prior, optimizer=optimizer, gauss_newton=gauss_newton
    )
    data = problem.data if data is None else data
    before = dict(problem.solver.solves)
    started = time.perf_counter()
    result = estimator.solve(data)
    seconds = time.perf_counter() - started
    model = result.minimizer
    # The noise carried to the estimate by the estimator's derivative in the data.
    carried = estimator.derivative_at(data, model)
    spread = (carried @ problem.noise.covariance @ carried.adjoint).matrix(
        form="components"
    )
    chi_squared = 2.0 * float(forward_problem.potential(data)(model))
    return {
        "model": model,
        "parameters": problem.parameter_map.parameters(model),
        "std": np.sqrt(np.maximum(np.diag(spread), 0.0)),
        "spread": spread,
        "iterations": result.iterations,
        "converged": bool(result.converged),
        "chi_squared": chi_squared,
        "seconds": seconds,
        "solves": {
            k: v - before[k] for k, v in problem.solver.solves.items() if v > before[k]
        },
    }


def misfit_along_the_line(problem: Problem, /) -> np.ndarray:
    """Half the chi-squared at models between the prior's expectation and the truth."""
    potential = ForwardProblem(problem.forward, error=problem.noise).potential(
        problem.data
    )
    return np.array([float(potential(s * problem.truth)) for s in LINE])


def valleys(values: np.ndarray, /) -> int:
    """How many local minima a curve has, its ends not counted."""
    inner = values[1:-1]
    return int(np.sum((inner < values[:-2]) & (inner < values[2:])))


def point(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The point answer and the figures that go with it."""
    started = time.perf_counter()
    scale = problem.scale
    size = problem.data_space.dim
    print(f"\n=== B. The most probable masses and lengths ({scale.name} scale) ===")
    print(
        f"{size} readings with noise {scale.noise_std:g}; four coordinates, the "
        f"logarithms of {', '.join(problem.parameter_names)}, each with prior "
        f"standard deviation {scale.prior_std:g}"
    )

    found = {}
    for label, gauss_newton in (("exact Hessian", False), ("Gauss-Newton", True)):
        found[label] = search(problem, gauss_newton=gauss_newton)
        run = found[label]
        print(
            f"  {label:14s} {run['iterations']:3d} iterations, converged "
            f"{run['converged']}, chi-squared {run['chi_squared']:.1f} for {size} "
            f"readings, {run['seconds']:.1f} s, integrations {run['solves']}"
        )
    best = found["exact Hessian"]
    apart = float(np.abs(found["Gauss-Newton"]["model"] - best["model"]).max())
    print(f"  the two searches end {apart:.1e} apart")

    truth = np.array(scale.true_parameters)
    # The estimate from readings with no noise: what is left is the bias.
    exact = search(problem, gauss_newton=True, data=problem.clean)
    bias = exact["parameters"] / truth - 1.0
    print("the estimate against the truth, with the noise it carries and its bias:")
    print(f"  {'':4s} {'truth':>8s} {'estimate':>9s} {'noise':>8s} {'bias':>8s}")
    for name, true, estimate, std, offset in zip(
        problem.parameter_names, truth, best["parameters"], best["std"], bias
    ):
        print(
            f"  {name:4s} {true:8.4f} {estimate:9.4f} {100.0 * std:7.2f}% "
            f"{100.0 * offset:+7.2f}%"
        )
    # The masses enter through their ratio alone: the readings do not move
    # the sum of their logarithms, so no noise reaches it and the estimate
    # of it is the prior's, which is where the masses' bias comes from.
    together = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    ratio = np.array([-1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    moved = {
        "the masses' common size": float(
            np.sqrt(max(together @ best["spread"] @ together, 0.0))
        ),
        "their ratio": float(np.sqrt(max(ratio @ best["spread"] @ ratio, 0.0))),
    }
    for name, value in moved.items():
        print(f"  noise carried to {name}: {100.0 * value:.3f}%")
    print(
        f"  estimated m2/m1 {best['parameters'][1] / best['parameters'][0]:.4f} "
        f"against {truth[1] / truth[0]:.4f}; estimated m1 m2 "
        f"{best['parameters'][0] * best['parameters'][1]:.4f}, the prior's own 1"
    )

    # The livelier release: the same pendulum, the same instrument, new data.
    lively = build(dataclasses.replace(scale, release=scale.lively_release))
    lines = {
        "release": misfit_along_the_line(problem),
        "lively": misfit_along_the_line(lively),
    }
    again = search(lively, gauss_newton=True, max_iterations=LIVELY_ITERATIONS)
    print("the misfit along the line from the prior's expectation to the truth:")
    for label, values in lines.items():
        print(
            f"  {label:8s} {valleys(values)} valleys; at the truth {values[np.argmin(np.abs(LINE - 1.0))]:.1f}"
        )
    print(
        f"the same search on the livelier record, {again['iterations']} iterations: "
        f"chi-squared {again['chi_squared']:.3g} for {size} readings, parameters "
        + ", ".join(f"{value:.3f}" for value in again["parameters"])
    )

    figure, axes = plt.subplots(
        1, 5, figsize=(14.0, 3.4), layout="constrained", width_ratios=[1, 1, 1, 1, 3]
    )
    parameter_panels(
        axes[:4],
        problem.parameter_names,
        truth,
        [
            log_normal_interval(
                "estimate, 2 sd of carried noise", best["model"], best["std"], "C1"
            )
        ],
    )
    for (label, values), color in zip(lines.items(), ("C0", "C3")):
        angles = scale.release if label == "release" else scale.lively_release
        axes[4].semilogy(
            LINE, values, color=color, lw=1.2, label=f"released at {angles}"
        )
    axes[4].axvline(1.0, color="black", ls="--", lw=0.8)
    axes[4].set_xlabel("from the prior's expectation (0) to the truth (1)")
    axes[4].set_ylabel("half chi-squared")
    axes[4].set_title("The misfit along a line", fontsize=10)
    axes[4].legend(fontsize=8, frameon=False)
    save(
        figure,
        outdir / "B_point.png",
        title="The most probable parameters, and why a livelier record is harder to fit",
    )
    print(f"part B took {time.perf_counter() - started:.1f} s")
    return {
        "searches": found,
        "apart": apart,
        "moved": moved,
        "bias": bias,
        "valleys": {label: valleys(values) for label, values in lines.items()},
        "lively": again,
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    """Run the part on a scale, building the problem unless it is given."""
    problem = build(scale) if problem is None else problem
    return point(problem, rng=part_rng(scale.seed, "B"), outdir=outdir)


def main(argv: list[str] | None = None) -> dict:
    """Run the part from the command line."""
    arguments = parse(__doc__, quick=QUICK, full=FULL, argv=argv)
    return run(arguments.scale, outdir=arguments.outdir)


if __name__ == "__main__":
    main()
