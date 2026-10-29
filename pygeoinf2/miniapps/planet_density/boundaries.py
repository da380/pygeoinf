"""Part D: the two boundary radii as further unknowns, a weakly nonlinear problem.

The unknown is the density together with the outward displacements of the
inner-core boundary and the core-mantle boundary, in units of 50 km, and the
forward operator is :mod:`radii`'s, first order in the displacements and
bilinear. Its derivative, the derivative's adjoint and its second derivative
are checked against finite differences at draws from the prior before they
are used.

The truth is part A's, PREM in the space plus a draw from the tight prior,
with the core-mantle boundary 30 km out and the inner-core boundary 20 km
in, and the data are its mass and moment through this operator, with errors
drawn from part A's error measure. The prior is part B's on the density,
independent of a standard normal on the scaled displacements. Then the MAP
model, by Newton-CG on the Gauss-Newton Hessian, and the Laplace Gaussian
about it; and Occam's model at the 95% level twice, from the same reference
to the same level, in two norms. In the model space's own norm, which adds
the squared displacements to the squared ``H^2`` norm of the density change;
and in the prior's Mahalanobis norm, where a unit of displacement costs
what a unit of density departure does: there the model is written
``m0 + L w`` with ``L`` the prior's covariance factor and Occam asks for the
smallest ``w``.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    AffineOperator,
    CholeskySolver,
    EuclideanSpace,
    ForwardProblem,
    GaussianMeasure,
    LinearOperator,
    MaximumAPosteriori,
    Occam,
)
from pygeoinf2.inference import LaplaceApproximation
from pygeoinf2.numerics.optimization import NewtonCG

from common import (
    draw_profile,
    finish_profile_axes,
    output_directory,
    parse,
    part_rng,
    save,
    title,
)
from problem import (
    FULL,
    LEVEL,
    PROPERTY_NAMES,
    QUICK,
    TIGHT_STD,
    Problem,
    Scale,
    build,
)
from radii import CHECK_POINTS, RADIUS_SCALE, boundary_operator, check

# The truth's displacements in km, outward: the inner-core boundary 20 km
# in, the core-mantle boundary 30 km out.
TRUE_SHIFTS = np.array([-20.0, 30.0])
INTERFACE_NAMES = ("ICB", "CMB")
OCCAM_LABELS = {
    "space": "Occam, the model space's norm",
    "prior": "Occam, the prior's norm",
}
LEGEND_ORDER = ("prior 95%", "MAP, Laplace 95%", *OCCAM_LABELS.values(), "truth")
# The displacement panel's reach, in km: past the prior's 95%.
LIMIT = 150.0
NEWTON_ITERATIONS = 50


def occam_answers(
    forward: ForwardProblem,
    prior: GaussianMeasure,
    data: np.ndarray,
    /,
    *,
    optimizer: Callable[[], NewtonCG],
) -> dict[str, dict]:
    """Occam's model in the model space's own norm and in the prior's
    Mahalanobis norm, from the prior's expectation to the same level.

    The second is Occam on the problem parameterized by ``m = m0 + L w``,
    whose smallest ``w`` is the smallest change in the prior's precision.
    Each answer is mapped back to the model space, and its change from the
    reference measured both ways.
    """
    reference = prior.expectation
    space = forward.model_space
    to_model = AffineOperator(prior.covariance_factor, reference)
    runs = (
        ("space", forward, reference, lambda x: x),
        (
            "prior",
            ForwardProblem(forward.forward_operator @ to_model, error=forward.error),
            None,
            to_model,
        ),
    )
    answers = {}
    for norm, problem, start, back in runs:
        t0 = time.perf_counter()
        result = Occam(
            problem,
            level=LEVEL,
            reference=start,
            optimizer=optimizer(),
            gauss_newton=True,
        ).solve(data)
        model = back(result.minimizer)
        change = space.subtract(model, reference)
        answers[norm] = {
            "model": model,
            "shifts": RADIUS_SCALE * np.asarray(model[1]),
            "chi_squared": forward.chi_squared(model, data),
            "converged": bool(result.converged),
            "outer_iterations": result.outer_iterations,
            "space_norm": space.norm(change),
            "density_norm": space.subspaces[0].norm(change[0]),
            "prior_norm": float(
                np.sqrt(space.inner_product(change, prior.precision(change)))
            ),
            "seconds": time.perf_counter() - t0,
        }
    return answers


def boundaries_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The MAP model, the Laplace Gaussian about it and Occam's model in two
    norms, with a figure."""
    started = time.perf_counter()
    space = problem.space
    print(f"\n=== D. The boundaries as unknowns ({problem.scale.name} scale) ===")
    model_space, operator = boundary_operator(problem)
    prior = GaussianMeasure.from_product(
        [
            problem.prior(TIGHT_STD),
            GaussianMeasure.from_standard_deviation(EuclideanSpace(2), 1.0),
        ],
        labels=("density", "radii"),
    )
    worst = check(operator, prior, rng)
    print(
        f"at {CHECK_POINTS} draws from the prior the derivative and the second "
        f"derivative agree with central differences, and the derivative's adjoint "
        f"with its defining identity: worst relative defect {worst:.1e}"
    )
    truth = (problem.truth, TRUE_SHIFTS / RADIUS_SCALE)
    error = problem.problem.error_measure
    data = error.domain.add(operator(truth), error.sample(rng=rng))
    forward = ForwardProblem(operator, error=error)
    critical = forward.critical_chi_squared(level=LEVEL)
    print(
        f"the truth: part A's, with the ICB moved {TRUE_SHIFTS[0]:+g} km and the "
        f"CMB {TRUE_SHIFTS[1]:+g} km; the data's chi-squared at the reference, "
        f"PREM at its own radii, {forward.chi_squared(prior.expectation, data):.1f}, "
        f"at the truth's density with PREM's radii "
        f"{forward.chi_squared((problem.truth, np.zeros(2)), data):.1f}, at the "
        f"truth {forward.chi_squared(truth, data):.2f}, against the critical "
        f"{critical:.2f}"
    )

    def newton(**options: Any) -> NewtonCG:
        return NewtonCG(
            max_iterations=NEWTON_ITERATIONS,
            max_cg_iterations=4 * model_space.dim,
            **options,
        )

    t0 = time.perf_counter()
    estimator = MaximumAPosteriori(
        forward,
        prior,
        optimizer=newton(preconditioner=prior.covariance),
        gauss_newton=True,
    )
    search = estimator.solve(data)
    map_model = search.minimizer
    map_misfit = forward.chi_squared(map_model, data)
    print(
        f"MAP: {search.iterations} Newton iterations on the Gauss-Newton Hessian in "
        f"{time.perf_counter() - t0:.2f} s, converged {search.converged}; "
        f"chi-squared {map_misfit:.2f}"
    )

    # The three properties of the density on the fixed mesh, and the two
    # displacements in km.
    readout = LinearOperator.from_matrix(
        model_space,
        EuclideanSpace(5),
        np.block(
            [
                [problem.properties.matrix(form="components"), np.zeros((3, 2))],
                [np.zeros((2, space.dim)), RADIUS_SCALE * np.eye(2)],
            ]
        ),
        form="components",
    )
    t0 = time.perf_counter()
    laplace = LaplaceApproximation(estimator, solver=CholeskySolver())
    gaussian = laplace.at(map_model, data).push_forward(readout)
    center = gaussian.expectation
    deviation = np.sqrt(np.diag(gaussian.covariance.matrix(form="components")))
    before = prior.push_forward(readout)
    prior_deviation = np.sqrt(np.diag(before.covariance.matrix(form="components")))
    true_values = readout(truth)
    print(f"Laplace Gaussian about the MAP model ({time.perf_counter() - t0:.2f} s):")
    names = list(PROPERTY_NAMES) + [f"{name} (km)" for name in INTERFACE_NAMES]
    print(f"  {'':<12s} {'prior':>16s} {'Laplace':>17s} {'truth':>8s}")
    for name, p, s0, m, s, t in zip(
        names, before.expectation, prior_deviation, center, deviation, true_values
    ):
        print(f"  {name:<12s} {p:8.3f} ± {s0:5.3f} {m:8.3f} ± {s:6.3f} {t:8.3f}")

    occams = occam_answers(forward, prior, data, optimizer=newton)
    print(
        f"Occam's model from PREM at its own radii to chi-squared {critical:.2f}, in "
        f"two norms; the sizes are of the change from PREM, of its density part "
        f"and of the whole in the model space's norm, and of the whole in the "
        f"prior's"
    )
    print(
        f"  {'norm':<14s} {'chi2':>5s} {'ICB km':>8s} {'CMB km':>8s} "
        f"{'density':>9s} {'whole':>9s} {'prior':>8s} {'core':>7s} {'mantle':>7s} "
        f"{'jump':>7s} {'outer':>5s}"
    )
    for norm, label in (("space", "model space's"), ("prior", "prior's")):
        answer = occams[norm]
        answer["properties"] = readout(answer["model"])[:3]
        core, mantle, jump = answer["properties"]
        print(
            f"  {label:<14s} {answer['chi_squared']:5.2f} {answer['shifts'][0]:+8.1f} "
            f"{answer['shifts'][1]:+8.1f} {answer['density_norm']:9.3g} "
            f"{answer['space_norm']:9.3g} {answer['prior_norm']:8.3g} {core:7.3f} "
            f"{mantle:7.3f} {jump:7.3f} {answer['outer_iterations']:5d}"
        )

    result = {
        "worst_defect": worst,
        "truth": true_values,
        "truth_model": truth,
        "map_model": map_model,
        "converged": bool(search.converged),
        "iterations": search.iterations,
        "map_chi_squared": map_misfit,
        "critical": critical,
        "center": center,
        "deviation": deviation,
        "prior_center": before.expectation,
        "prior_deviation": prior_deviation,
        "occam": occams,
    }
    boundaries_figure(problem, result, outdir=outdir)
    print(f"D took {time.perf_counter() - started:.1f} s")
    return result


def boundaries_figure(
    problem: Problem, result: dict, /, *, outdir: pathlib.Path
) -> pathlib.Path:
    """The density less PREM and the displacements: the truth, the MAP model
    with its Laplace intervals, and Occam's model in the two norms."""
    space = problem.space
    occams = result["occam"]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), layout="constrained", width_ratios=[1.6, 1.0]
    )
    for model, color, style, label in (
        (result["truth_model"], "black", "--", "truth"),
        (result["map_model"], "C0", "-", "MAP"),
        (occams["space"]["model"], "C2", "-", OCCAM_LABELS["space"]),
        (occams["prior"]["model"], "C1", "-", OCCAM_LABELS["prior"]),
    ):
        draw_profile(
            space,
            space.subtract(model[0], problem.reference),
            left,
            color=color,
            ls=style,
            lw=1.3,
            label=label,
        )
    left.axhline(0.0, color="0.5", lw=1.0, ls=":", label="PREM, the reference")
    finish_profile_axes(left, subtitle="the density on the fixed mesh, less PREM")
    left.set_ylabel("density less PREM (g/cm³)")
    for index, name in enumerate(INTERFACE_NAMES):
        k = 3 + index
        for offset, center, spread, marker, color, label in (
            (
                -0.24,
                result["prior_center"][k],
                result["prior_deviation"][k],
                "o",
                "0.5",
                "prior 95%",
            ),
            (
                -0.08,
                result["center"][k],
                result["deviation"][k],
                "s",
                "C0",
                "MAP, Laplace 95%",
            ),
        ):
            right.errorbar(
                index + offset,
                center,
                yerr=1.96 * spread,
                fmt=marker,
                color=color,
                capsize=4,
                label=label if index == 0 else None,
            )
        for offset, norm, color in ((0.08, "space", "C2"), (0.24, "prior", "C1")):
            value = occams[norm]["shifts"][index]
            # A displacement far past the prior's range is drawn at the edge
            # of the panel with its value written.
            shown = float(np.clip(value, -0.95 * LIMIT, 0.95 * LIMIT))
            right.plot(
                index + offset,
                shown,
                "D",
                color=color,
                label=OCCAM_LABELS[norm] if index == 0 else None,
            )
            if shown != value:
                right.annotate(
                    f"{value:+.0f} km",
                    (index + offset, shown),
                    xytext=(6, -3),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
        right.plot(
            [index - 0.35, index + 0.35],
            [result["truth"][k]] * 2,
            color="black",
            ls="--",
            lw=1.0,
            label="truth" if index == 0 else None,
        )
    right.axhline(0.0, color="0.7", lw=0.6)
    right.set_xticks([0, 1], list(INTERFACE_NAMES))
    right.set_xlim(-0.6, 1.6)
    right.set_ylim(-LIMIT, LIMIT)
    right.set_ylabel("outward displacement (km)")
    right.set_title("the boundaries", fontsize=10)
    handles, labels = right.get_legend_handles_labels()
    order = [labels.index(name) for name in LEGEND_ORDER]
    right.legend([handles[i] for i in order], LEGEND_ORDER, fontsize=7, frameon=False)
    fig.suptitle(
        title(
            "D", "the boundary radii as unknowns: MAP, Laplace, and Occam in two norms"
        )
    )
    path = outdir / "boundaries.png"
    save(fig, path)
    return path


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale, seed=seed)
    return boundaries_answer(problem, rng=part_rng(seed, "d"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
