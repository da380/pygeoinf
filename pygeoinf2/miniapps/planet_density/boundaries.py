"""Part E: the two boundaries as unknowns, a nonlinear problem.

The unknown is the density together with ``delta``, the outward displacements
in km of the inner-core boundary and the core-mantle boundary from their
places in the mesh. The mesh does not move. Moving a boundary out by
``delta_k`` turns a shell of thickness ``delta_k`` from the material above it
into the material below it, so to first order

    M(rho, delta) = int rho dV - sum_k 4 pi r_k^2 [rho]_k delta_k,
    I(rho, delta) = (2/3) int rho r^2 dV - sum_k (8 pi / 3) r_k^4 [rho]_k delta_k,

with ``[rho]_k`` the jump at interface ``k``, the value above less the value
below, and ``r_k`` the mesh radius. That is bilinear: the derivative in
``rho`` is the linear part plus the jumps weighted by ``delta``, the
derivative in ``delta_k`` is the number ``-4 pi r_k^2 [rho]_k``, and the
second derivative couples the jump of one direction with the displacement of
the other and is otherwise zero. All three are checked against finite
differences before they are used.

The prior is the tight density prior of part C, independent of a Gaussian on
the displacements with a standard deviation of 20 km. Then the mode, the
Laplace Gaussian about it on the three properties and on the displacements,
and Occam's smallest model in the model space's own norm.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    CGSolver,
    DirectSum,
    EuclideanSpace,
    ForwardProblem,
    GaussianMeasure,
    LinearOperator,
    MaximumAPosteriori,
    Occam,
    Operator,
    testing,
)
from pygeoinf2.inference import LaplaceApproximation
from pygeoinf2.numerics.optimization import NewtonCG

from common import (
    draw_prem,
    draw_profile,
    finish_profile_axes,
    output_directory,
    parse,
    part_rng,
    save,
)
from problem import (
    FULL,
    LEVEL,
    PROPERTY_SHORT,
    QUICK,
    TIGHT_STD,
    TO_MASS,
    TO_MOMENT,
    Problem,
    Scale,
    build,
)
from reference import R_CMB, R_ICB

RADII_STD = 20.0
INTERFACE_NAMES = ("ICB", "CMB")
NEWTON_ITERATIONS = 50
CHECK_POINTS = 3


def boundary_operator(problem: Problem, /) -> tuple[DirectSum, Operator]:
    """The model space ``(density, radii)`` and the bilinear forward operator on it."""
    space = problem.space
    model_space = DirectSum([space, EuclideanSpace(2)], labels=("density", "radii"))
    data_space = problem.forward.codomain
    linear = problem.forward.matrix(form="components")
    jumps = problem.jumps
    radii = np.array([R_ICB, R_CMB])
    # Data per unit of jump times displacement: rows mass and moment, columns
    # the two interfaces.
    shells = np.vstack(
        [TO_MASS * 4.0 * np.pi * radii**2, TO_MOMENT * 8.0 * np.pi / 3.0 * radii**4]
    )
    n = space.dim

    def parts(x: Any) -> tuple[np.ndarray, np.ndarray]:
        c = model_space.to_components(x)
        return c[:n], c[n:]

    def value(x: Any) -> np.ndarray:
        c, delta = parts(x)
        return linear @ c - shells @ (delta * (jumps @ c))

    def matrix(c: np.ndarray, delta: np.ndarray, constant: bool) -> np.ndarray:
        density_block = -shells @ (delta[:, None] * jumps)
        if constant:
            density_block = density_block + linear
        return np.hstack([density_block, -shells * (jumps @ c)[None, :]])

    def derivative(x: Any) -> LinearOperator:
        c, delta = parts(x)
        return LinearOperator.from_matrix(
            model_space, data_space, matrix(c, delta, True), form="components"
        )

    def second_derivative(x: Any, direction: Any) -> LinearOperator:
        dc, ddelta = parts(direction)
        return LinearOperator.from_matrix(
            model_space, data_space, matrix(dc, ddelta, False), form="components"
        )

    operator = Operator.from_callables(
        model_space,
        data_space,
        value,
        derivative=derivative,
        second_derivative=second_derivative,
    )
    return model_space, operator


def check(
    operator: Operator, prior: GaussianMeasure, rng: np.random.Generator
) -> float:
    """The derivatives against finite differences at a few draws from the prior,
    and the derivative's adjoint; the worst defect seen."""
    worst = 0.0
    for _ in range(CHECK_POINTS):
        point = prior.sample(rng=rng)
        reports = (
            testing.check_operator(operator.derivative(point), rng=rng, measure=prior),
            testing.check_derivative(operator, point, rng=rng, measure=prior),
            testing.check_second_derivative(operator, point, rng=rng, measure=prior),
        )
        worst = max(worst, *(max(r.defects.values()) for r in reports))
    return worst


def boundaries_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The mode, the Laplace Gaussian and Occam's model, with a figure."""
    started = time.perf_counter()
    space = problem.space
    print(f"\n=== E. The boundaries as unknowns ({problem.scale.name} scale) ===")
    model_space, operator = boundary_operator(problem)
    density_prior = problem.prior(TIGHT_STD)
    prior = GaussianMeasure.from_product(
        [
            density_prior,
            GaussianMeasure.from_standard_deviations(
                EuclideanSpace(2), np.full(2, RADII_STD)
            ),
        ],
        labels=("density", "radii"),
    )
    worst = check(operator, prior, rng)
    print(
        f"at {CHECK_POINTS} draws from the prior the derivative and the second "
        f"derivative agree with central differences, and the derivative's adjoint "
        f"with its defining identity: worst relative defect {worst:.1e}"
    )
    forward = ForwardProblem(operator, error=problem.problem.error_measure)
    critical = forward.critical_chi_squared(level=LEVEL)
    start = prior.expectation

    t0 = time.perf_counter()
    estimator = MaximumAPosteriori(
        forward,
        prior,
        optimizer=NewtonCG(
            preconditioner=prior.covariance,
            max_iterations=NEWTON_ITERATIONS,
            max_cg_iterations=4 * model_space.dim,
        ),
    )
    search = estimator.solve(problem.data)
    mode = search.minimizer
    mode_misfit = forward.chi_squared(mode, problem.data)
    print(
        f"MAP: {search.iterations} Newton iterations in {time.perf_counter() - t0:.2f} "
        f"s, converged {search.converged}; chi-squared {mode_misfit:.2f} against the "
        f"critical {critical:.2f}"
    )

    # The three properties of the density on the fixed mesh, and the two
    # displacements.
    properties = problem.properties.matrix(form="components")
    readout = LinearOperator.from_matrix(
        model_space,
        EuclideanSpace(5),
        np.block(
            [
                [properties, np.zeros((3, 2))],
                [np.zeros((2, space.dim)), np.eye(2)],
            ]
        ),
        form="components",
    )
    t0 = time.perf_counter()
    laplace = LaplaceApproximation(
        estimator, solver=CGSolver().with_preconditioner(prior.covariance)
    )
    gaussian = laplace.at(mode, problem.data).push_forward(readout)
    center = gaussian.expectation
    deviation = np.sqrt(np.diag(gaussian.covariance.matrix(form="components")))
    print(f"Laplace Gaussian about the mode ({time.perf_counter() - t0:.2f} s):")
    names = list(PROPERTY_SHORT) + [f"{name} shift (km)" for name in INTERFACE_NAMES]
    prior_values = np.concatenate(
        [problem.properties(problem.reference_model), np.zeros(2)]
    )
    prior_deviation = np.concatenate(
        [
            np.sqrt(
                np.diag(
                    density_prior.push_forward(problem.properties).covariance.matrix(
                        form="components"
                    )
                )
            ),
            np.full(2, RADII_STD),
        ]
    )
    yardstick = np.concatenate([problem.prem_properties, np.zeros(2)])
    print(
        f"  {'':<16s} {'prior':>16s} {'Laplace':>17s} {'95% from':>9s} {'to':>8s} "
        f"{'PREM':>7s}"
    )
    for name, p, s0, m, s, y in zip(
        names, prior_values, prior_deviation, center, deviation, yardstick
    ):
        print(
            f"  {name:<16s} {p:8.3f} ± {s0:5.3f} {m:8.3f} ± {s:6.3f} "
            f"{m - 1.96 * s:9.3f} {m + 1.96 * s:8.3f} {y:7.3f}"
        )

    t0 = time.perf_counter()
    occam = Occam(
        forward,
        level=LEVEL,
        reference=start,
        start=mode,
        optimizer=NewtonCG(
            max_iterations=NEWTON_ITERATIONS, max_cg_iterations=4 * model_space.dim
        ),
        max_iterations=30,
    ).solve(problem.data)
    smallest = occam.minimizer
    occam_shift = np.asarray(smallest[1])
    change = model_space.subtract(smallest, start)
    density_part = space.norm(change[0])
    print(
        f"Occam ({time.perf_counter() - t0:.1f} s, {occam.outer_iterations} outer "
        f"iterations, converged {occam.converged}): chi-squared "
        f"{forward.chi_squared(smallest, problem.data):.2f}; shifts ICB "
        f"{occam_shift[0]:+.0f} km and CMB {occam_shift[1]:+.0f} km; the density "
        f"changed by {density_part:.3g} in the space's norm, the radii by "
        f"{np.linalg.norm(occam_shift):.3g} km"
    )
    print(
        "  the model space's norm adds kilometers of displacement to a density "
        "norm that carries the volume in km^3: a kilometer is nearly free, and "
        "Occam fits the data by moving the boundaries"
    )

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.0, 4.2), layout="constrained", width_ratios=[1.7, 1.0]
    )
    draw_prem(left, color="black", lw=1.0, label="PREM")
    draw_profile(
        space,
        problem.reference_model,
        left,
        color="C1",
        lw=1.0,
        ls="--",
        label="reference",
    )
    draw_profile(space, mode[0], left, color="C0", lw=1.6, label="MAP density")
    for radius, shift in zip((R_ICB, R_CMB), mode[1]):
        left.axvline(radius + shift, color="C0", lw=1.0, ls="-.")
    left.plot([], [], color="C0", lw=1.0, ls="-.", label="MAP boundaries")
    finish_profile_axes(
        left,
        title="The MAP density on the fixed mesh, and where it puts the boundaries",
    )
    for index, name in enumerate(INTERFACE_NAMES):
        right.errorbar(
            index - 0.12,
            0.0,
            yerr=1.96 * RADII_STD,
            fmt="o",
            color="0.5",
            capsize=4,
            label="prior 95%" if index == 0 else None,
        )
        right.errorbar(
            index + 0.12,
            center[3 + index],
            yerr=1.96 * deviation[3 + index],
            fmt="s",
            color="C0",
            capsize=4,
            label="Laplace 95%" if index == 0 else None,
        )
    right.axhline(0.0, color="black", lw=0.8, ls="--", label="PREM's radii")
    right.set_xticks([0, 1], list(INTERFACE_NAMES))
    right.set_xlim(-0.6, 1.6)
    right.set_ylabel("outward shift (km)")
    right.set_title("The radii (Occam's shifts are off the scale)", fontsize=10)
    right.legend(fontsize=7, frameon=False)
    save(fig, outdir / "boundaries.png")
    print(f"E took {time.perf_counter() - started:.1f} s")
    return {
        "worst_defect": worst,
        "mode": mode,
        "converged": bool(search.converged),
        "iterations": search.iterations,
        "mode_chi_squared": mode_misfit,
        "critical": critical,
        "center": center,
        "deviation": deviation,
        "occam": smallest,
        "occam_shift": occam_shift,
        "occam_converged": bool(occam.converged),
        "occam_chi_squared": forward.chi_squared(smallest, problem.data),
    }


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return boundaries_answer(problem, rng=part_rng(seed, "e"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
