"""Part B: the point answers.

Least squares at the damping a Bayesian would guess, the minimum-norm
model under the discrepancy principle, the latter's resolution of a bump
under good and under poor coverage, and its error budget on the three
properties: the noise it propagates and the bias only the truth reveals.
Size is measured in the prior's norm: the problem is parameterized by the
prior's covariance factor, ``m = L w``, and both estimators measure ``|w|``.
Every solve is a Cholesky factorization of the damped data-space operator
``A Q A* + t R``, redone at each damping the search tries.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import CholeskySolver, LeastSquares, MinimumNorm

from src.common import (
    draw,
    error_budget_panels,
    outline_boxes,
    parse,
    part_rng,
    save,
    variances,
)
from src.problem import (
    BOXES,
    FULL,
    PROPERTIES,
    QUICK,
    Problem,
    Scale,
    build,
)

# In the prior's norm, least squares at a damping of one is the posterior
# mean of part C: the guess a Bayesian would make.
GUESSED_DAMPING = 1.0


def cartesian(point: Sequence[float], /) -> np.ndarray:
    """A point ``(radius, angle)`` of a disk, or ``(radius, latitude,
    longitude)`` of a ball, as a position in space."""
    if len(point) == 2:
        r, a = point
        return np.array([r * np.cos(np.radians(a)), r * np.sin(np.radians(a)), 0.0])
    r, b, a = point
    b, a = np.radians(b), np.radians(a)
    return r * np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])


def bump(
    problem: Problem,
    center: tuple[float, ...],
    width: float,
    /,
    *,
    extension: str = "fit",
) -> Any:
    """A Gaussian bump of unit peak about a point of the model space.

    ``extension`` is how the space's projection treats the padding beyond
    the domain: ``"fit"`` is the best fit on the domain in the kept modes,
    ``"constant"`` the values on the domain's grid continued at their end
    values, which keeps a bump much narrower than the layer out of the
    padding, where a fit of it may grow large.
    """
    middle = cartesian(problem.lift([center])[0])

    def value(point: Sequence[float]) -> float:
        offset = cartesian(point) - middle
        return float(np.exp(-0.5 * offset @ offset / width**2))

    # Nothing in the core, which no ray enters and no box reaches.
    return problem.space.project_function(
        [lambda point: 0.0, value], extension=extension
    )


def box_center(box: tuple[float, float, float, float], /) -> tuple[float, float]:
    """The middle of a polar box ``(r1, r2, angle1, angle2)``."""
    r1, r2, a1, a2 = box
    return 0.5 * (r1 + r2), 0.5 * (a1 + a2)


def whitened(problem: Problem, /) -> Any:
    """The problem in ``w``, with ``m = L w`` and ``L`` the prior's covariance
    factor: the norm of ``w`` is the prior's norm of ``m``."""
    return problem.problem.parameterized(problem.prior.covariance_factor)


def whiten(problem: Problem, model: Any, /) -> Any:
    """The ``w`` of a model, ``L^-1 m = L* Q^-1 m``, exact through the prior's
    precision."""
    prior = problem.prior
    return prior.covariance_factor.adjoint(prior.precision(model))


def minimum_norm(problem: Problem, /, *, rtol: float = 1e-3) -> MinimumNorm:
    """The minimum-norm estimator under the discrepancy principle, smallest in
    the prior's norm, its damping found to ``rtol``."""
    return MinimumNorm(
        whitened(problem),
        level=problem.scale.level,
        solver=CholeskySolver(),
        rtol=rtol,
    )


def error_budget(problem: Problem, chosen: Any, /) -> dict:
    """The estimator the data chose, pushed through the prior's factor to the
    properties: its estimate, the noise it propagates and its bias against
    the truth."""
    pushed = chosen.push_forward(problem.properties @ problem.prior.covariance_factor)
    return {
        "estimate": pushed(problem.data),
        "propagated": np.sqrt(variances(pushed.propagated_covariance())),
        "bias": pushed.bias(whiten(problem, problem.truth)),
    }


def point_answers(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """Both estimators, the budget on the properties, the resolution of bumps."""
    started = time.perf_counter()
    scale, space, forward = problem.scale, problem.space, problem.problem
    print(f"\n=== B. The point answers ({scale.name} scale) ===")

    factor = problem.prior.covariance_factor
    norm = factor.domain.norm
    truth_norm = norm(whiten(problem, problem.truth))

    t0 = time.perf_counter()
    guess_w = LeastSquares(
        whitened(problem), damping=GUESSED_DAMPING, solver=CholeskySolver()
    )(problem.data)
    guess = factor(guess_w)
    print(
        f"LeastSquares at damping {GUESSED_DAMPING:g}: {time.perf_counter() - t0:.2f} s; "
        f"chi-squared {forward.chi_squared(guess, problem.data):.1f} against the "
        f"critical {forward.critical_chi_squared(level=scale.level):.1f}, prior norm "
        f"{norm(guess_w):.2f}, relative error over the mantle "
        f"{problem.relative_error(guess):.3f}"
    )

    t0 = time.perf_counter()
    estimator = minimum_norm(problem)
    search = estimator.discrepancy_search(problem.data)
    chosen = estimator.for_data(problem.data)
    change = chosen(problem.data)
    smallest = factor(change)
    print(
        f"MinimumNorm by the discrepancy principle: damping {search.argument:.4g} after "
        f"{search.evaluations} misfit evaluations, a factorization each, "
        f"{time.perf_counter() - t0:.2f} s; chi-squared "
        f"{forward.chi_squared(smallest, problem.data):.1f}, prior norm "
        f"{norm(change):.2f} (the truth's {truth_norm:.2f}), "
        f"relative error over the mantle {problem.relative_error(smallest):.3f}"
    )

    t0 = time.perf_counter()
    budget = error_budget(problem, chosen)
    print(
        f"its error budget on the three properties ({time.perf_counter() - t0:.2f} s):"
    )
    print(
        f"  {'mean over':<16s} {'estimate':>9s} {'propagated sd':>14s} {'bias':>8s} "
        f"{'error':>8s} {'truth':>8s}"
    )
    for name, e, s, b, t in zip(
        PROPERTIES,
        budget["estimate"],
        budget["propagated"],
        budget["bias"],
        problem.true_properties,
    ):
        print(f"  {name:<16s} {e:+9.3f} {s:14.3f} {b:+8.3f} {e - t:+8.3f} {t:+8.3f}")

    t0 = time.perf_counter()
    mantle = space.index("mantle")
    seen, peaks = {}, {}
    for name, box in BOXES:
        feature = bump(problem, box_center(box), scale.bump_width)
        # The model-space resolution: L B A L applied to the bump's w.
        image = chosen.push_forward(factor).resolution(whiten(problem, feature))
        seen[name] = image
        peaks[name] = float(space.layers[mantle].interior_values(image[mantle]).max())
    print(
        f"the resolution of a unit bump of width {scale.bump_width} "
        f"({time.perf_counter() - t0:.2f} s):"
    )
    for name in seen:
        print(f"  in the {name} box the estimate sees a peak of {peaks[name]:.3f}")

    # The maps and the error budget in two subfigures, each laid out on its own.
    fig = plt.figure(figsize=(9.6, 11.4), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[2.2, 0.9])
    axes = np.asarray(upper.subplots(2, 2, subplot_kw={"aspect": "equal"}))
    limit = float(
        np.abs(space.layers[mantle].interior_values(problem.truth[mantle])).max()
    )
    draw(
        space,
        guess,
        axes[0, 0],
        title=f"least squares at damping {GUESSED_DAMPING:g}",
        limit=limit,
    )
    draw(
        space,
        smallest,
        axes[0, 1],
        title=f"minimum norm, damping {search.argument:.3g}",
        limit=limit,
    )
    # One color scale for both, set by the better resolved.
    reach = max(peaks.values())
    for ax, (name, box) in zip(axes[1], BOXES):
        draw(
            space,
            seen[name],
            ax,
            title=f"a unit bump, {name}: peak {peaks[name]:.2f}",
            limit=reach,
        )
        center = box_center(box)
        t = np.radians(np.linspace(0.0, 360.0, 121))
        x0 = center[0] * np.cos(np.radians(center[1]))
        y0 = center[0] * np.sin(np.radians(center[1]))
        ax.plot(
            x0 + 2.0 * scale.bump_width * np.cos(t),
            y0 + 2.0 * scale.bump_width * np.sin(t),
            color="black",
            linewidth=0.8,
            linestyle="--",
        )
    for ax in axes[0]:
        outline_boxes(ax, BOXES, color="black", linewidth=0.6)
    error_budget_panels(
        lower.subplots(1, len(PROPERTIES)),
        [
            f"mean over the {name}" + (" box" if name != "core" else "")
            for name in PROPERTIES
        ],
        budget["estimate"],
        budget["propagated"],
        budget["bias"],
        problem.true_properties,
        ylabel="mean slowness perturbation (%)",
    )
    save(
        fig,
        outdir / "point.png",
        title="Tomography, part B: the point answers. The two estimates, how the "
        "minimum-norm one sees\na bump where the rays cross and where they do "
        "not, and its estimate of each mean with its error budget",
    )
    print(f"B took {time.perf_counter() - started:.1f} s")
    return {
        "guess": guess,
        "smallest": smallest,
        "norm": norm(change),
        "truth_norm": truth_norm,
        "damping": search.argument,
        "evaluations": search.evaluations,
        **budget,
        "peaks": peaks,
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return point_answers(problem, rng=part_rng(scale.seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
