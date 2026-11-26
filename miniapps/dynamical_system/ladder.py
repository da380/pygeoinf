"""Part A: the forward operator, piece by piece, and its checks.

The forward operator takes masses and lengths to noisy readings of a
double pendulum's angles, and is built from four operators. This part
checks each, and then the whole:

* how well the trajectory's space holds the solution between the nodes of
  its mesh, which is what the element length buys;
* the operators that are exact, between tables, functions of time and
  readings, against the adjoint identity to rounding;
* the solution operator's derivative against finite differences, and its
  adjoint against its derivative, by both routes: the adjoint equation,
  accurate to the integrator's tolerance, and the sensitivities, whose
  transpose is exact;
* the same with an unknown forcing, an argument with hundreds of
  components where the parameters have four;
* how fast the trajectory's sensitivity to the parameters grows with time,
  from the release the data come from and from a livelier one. It is this
  growth that decides how long a record an inversion can use.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import GaussianMeasure
from pygeoinf.testing import check_derivative, check_operator

from src.common import parse, part_rng, save
from src.problem import FULL, QUICK, Problem, Scale, build
from src.spaces import from_table, to_table

ADJOINT = "the adjoint satisfies (A x, y) == (x, A* y)"


def representation(problem: Problem, /, *, rng: np.random.Generator) -> dict:
    """The largest error of the lifted trajectory against the integrator's own, at random times."""
    scale, solver, space = problem.scale, problem.solver, problem.space
    parameters = np.array(scale.true_parameters)
    found = {}
    for label, release in (
        ("release", scale.release),
        ("lively", scale.lively_release),
    ):
        state = problem.solver_for(release).solve(parameters)
        lifted = problem.lift(solver.sample(state))
        times = rng.uniform(0.0, scale.duration, 400)
        exact = state(times)
        errors = [
            float(
                np.abs(
                    space.evaluate(space.from_components(lifted[k]), list(times))
                    - exact[k]
                ).max()
            )
            for k in range(problem.system.state_dim)
        ]
        found[label] = (max(errors[:2]), max(errors[2:]))
    return found


def sensitivity_growth(problem: Problem, /) -> dict:
    """The size of the angles' derivative in the parameters, at each node, for two releases."""
    parameters = np.array(problem.scale.true_parameters)
    found = {}
    for label, release in (
        ("release", problem.scale.release),
        ("lively", problem.scale.lively_release),
    ):
        solver = problem.solver_for(release)
        jacobian = solver.sensitivities(solver.solve(parameters), parameters)
        found[label] = np.linalg.norm(jacobian[:2], axis=(0, 2))
    return found


def ladder(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The checks of the forward operator and what they measured."""
    started = time.perf_counter()
    scale, space, grid = problem.scale, problem.space, problem.grid
    print(f"\n=== A. The forward operator and its checks ({scale.name} scale) ===")
    print(
        f"a function of time: {space!r}, {grid.nodes.size} nodes in "
        f"[0, {scale.duration:g}] s; a trajectory has {problem.trajectory.dim} components"
    )

    held = representation(problem, rng=rng)
    print("the trajectory between the nodes, against the integrator's own:")
    for label, (angles, rates) in held.items():
        print(
            f"  {label:8s} largest error {angles:.1e} rad in the angles, {rates:.1e} rad/s in the rates"
        )

    exact = {
        "table to function": check_operator(from_table(space), rng=rng),
        "function to table": check_operator(to_table(space), rng=rng),
        "lift": check_operator(problem.lift, rng=rng),
        "observation": check_operator(problem.observe, rng=rng),
    }
    print("operators that are exact, the worst defect of each:")
    for name, report in exact.items():
        print(f"  {name:20s} {report.worst:.1e}")

    truth = problem.truth
    routes = {}
    print("the forward operator at the truth, by its two routes:")
    for name, operator in (
        ("sensitivities", problem.forward),
        ("adjoint equation", problem.forward_by_adjoint),
    ):
        t0 = time.perf_counter()
        linearization = operator.at(truth)
        linearization.derivative.adjoint(problem.data)
        seconds = time.perf_counter() - t0
        slope = check_derivative(operator, truth, rng=rng, rtol=1e-4)
        adjoint = check_operator(linearization.derivative, rng=rng, rtol=1e-5)
        routes[name] = (slope.worst, adjoint.defects[ADJOINT], seconds)
        print(
            f"  {name:17s} derivative against differences {slope.worst:.1e}, "
            f"adjoint against derivative {adjoint.defects[ADJOINT]:.1e}, "
            f"a gradient in {seconds:.2f} s"
        )
    matrices = [
        operator.at(truth).derivative.matrix(form="components")
        for operator in (problem.forward, problem.forward_by_adjoint)
    ]
    agreement = float(
        np.abs(matrices[0] - matrices[1]).max() / np.abs(matrices[0]).max()
    )
    print(f"  the two routes' Jacobians differ by {agreement:.1e} of the largest entry")

    prior = GaussianMeasure.from_product(
        [
            GaussianMeasure.from_standard_deviation(
                problem.parameter_map.space, scale.prior_std
            ),
            problem.forcing_prior,
        ],
        domain=problem.forced_forward.domain,
    )
    point = (truth, problem.forcing_prior.sample(rng=rng))
    t0 = time.perf_counter()
    linearization = problem.forced_forward.at(point)
    linearization.derivative.adjoint(problem.data)
    seconds = time.perf_counter() - t0
    slope = check_derivative(
        problem.forced_forward, point, rng=rng, measure=prior, rtol=1e-4
    )
    adjoint = check_operator(
        linearization.derivative, rng=rng, measure=prior, rtol=1e-5
    )
    forced = (slope.worst, adjoint.defects[ADJOINT], seconds)
    print(
        f"with an unknown forcing, {problem.forcing_space.dim} more unknowns: "
        f"derivative against differences {slope.worst:.1e}, adjoint against "
        f"derivative {adjoint.defects[ADJOINT]:.1e}, a gradient in {seconds:.2f} s"
    )

    growth = sensitivity_growth(problem)
    print(
        "the angles' sensitivity to the parameters, its size at the end over its size at one second:"
    )
    one = int(np.searchsorted(grid.nodes, 1.0))
    for label, sizes in growth.items():
        print(
            f"  {label:8s} {sizes[-1] / sizes[one]:8.1f}  (largest {sizes.max():.1f} rad per unit parameter)"
        )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), layout="constrained")
    state = problem.solver.solve(np.array(scale.true_parameters))
    fine = np.linspace(0.0, scale.duration, 800)
    for index, color in zip(scale.observed, ("C0", "C1")):
        name = problem.system.state_names[index]
        axes[0].plot(fine, state(fine)[index], color=color, lw=1.2, label=name)
    for record, (index, times) in enumerate(zip(scale.observed, problem.times)):
        axes[0].plot(
            times, problem.data[record], "o", ms=3.5, color=("C0", "C1")[record]
        )
    axes[0].set_xlabel("time, s")
    axes[0].set_ylabel("angle, rad")
    axes[0].set_title("The motion and its readings", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    for (label, sizes), color in zip(growth.items(), ("C0", "C3")):
        angles = scale.release if label == "release" else scale.lively_release
        axes[1].semilogy(
            grid.nodes[1:],
            sizes[1:],
            color=color,
            lw=1.2,
            label=f"released at {angles}",
        )
    axes[1].set_xlabel("time, s")
    axes[1].set_ylabel("rad per unit parameter")
    axes[1].set_title("Sensitivity of the angles to the parameters", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    save(
        figure,
        outdir / "A_forward_operator.png",
        title="A double pendulum, read at irregular times, and how its sensitivity grows",
    )
    print(f"part A took {time.perf_counter() - started:.1f} s")
    return {
        "representation": held,
        "exact": {name: report.worst for name, report in exact.items()},
        "routes": routes,
        "agreement": agreement,
        "forced": forced,
        "growth": {
            label: float(sizes[-1] / sizes[one]) for label, sizes in growth.items()
        },
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    """Run the part on a scale, building the problem unless it is given."""
    problem = build(scale) if problem is None else problem
    return ladder(problem, rng=part_rng(scale.seed, "A"), outdir=outdir)


def main(argv: list[str] | None = None) -> dict:
    """Run the part from the command line."""
    arguments = parse(__doc__, quick=QUICK, full=FULL, argv=argv)
    return run(arguments.scale, outdir=arguments.outdir)


if __name__ == "__main__":
    main()
