"""Part D, continued: Occam's inversion at the 95% level.

The model closest to the background, in the model space's own Sobolev norm,
whose chi-squared does not exceed its critical value at 95%: the library's
``Occam``, an augmented Lagrangian over Newton-CG inner solves, on the
Gauss-Newton Hessian of the misfit, started from the MAP mode of part C.
The prior covariance here is the identity of the model space, so Occam's
norm is the prior's, and the two answers differ only in how far each lets
the data pull: the MAP fits to the noise's mean, Occam only to its 95%
level, and is the smoother for it.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import Occam
from pygeoinf2.numerics.optimization import NewtonCG

from common import (
    Stopwatch,
    describe,
    output_directory,
    parse,
    save,
    section,
    sponge_outline,
)
from inversion import map_mode
from survey import BOXES, FULL, LEVEL, QUICK, Scale, box_average_operator
from synthetic import Setup, setup


def occam_answer(state: Setup, mode: np.ndarray, /, *, outdir: pathlib.Path) -> dict:
    """Occam's model against the MAP mode and the truth, and the figure."""
    experiment, wave, problem = state.experiment, state.wave, state.problem
    scale = experiment.survey.scale
    space = experiment.space
    estimator = Occam(
        state.first_order,
        gauss_newton=True,
        level=LEVEL,
        reference=experiment.background,
        optimizer=NewtonCG(
            max_iterations=scale.occam_newton_iterations,
            max_cg_iterations=scale.occam_cg_iterations,
            preconditioner=experiment.prior.covariance,
        ),
        start=mode,
        max_iterations=scale.occam_iterations,
        tolerance=1e-4,
        gtol=1e-5,
    )
    watch = Stopwatch(wave.calls)
    result = estimator.solve(state.data)
    seconds, calls = watch.lap()
    model = result.minimizer
    target = estimator.target

    def distance(m: np.ndarray) -> float:
        return float(space.norm(space.subtract(m, experiment.background)))

    rows = {
        "MAP mode": mode,
        "Occam": model,
        "truth": experiment.truth,
    }
    print(
        f"Occam: {result.outer_iterations} outer iterations, {result.iterations} "
        f"Newton iterations ({result.message}); violation {result.violation:.1e}, "
        f"multiplier {result.inequality_multipliers[0]:.4g}; {seconds:.1f} s; "
        f"kernels {describe(calls)}"
    )
    print(f"  critical chi-squared at {LEVEL:.0%}: {target:.0f}")
    print(
        f"  {'model':<10s} {'chi-squared':>12s} {'distance':>10s} {'rel. error':>11s}"
    )
    for name, m in rows.items():
        print(
            f"  {name:<10s} {problem.chi_squared(m, state.data):12.0f} "
            f"{distance(m):10.4f} {experiment.relative_error(m):11.3f}"
        )
    boxes = box_average_operator(space, experiment.survey)
    means = {name: np.asarray(boxes(m)) for name, m in rows.items()}
    print(
        "  box means: "
        + ", ".join(
            f"{name} " + "/".join(f"{v:.4f}" for v in values)
            for name, values in means.items()
        )
    )

    survey = experiment.survey
    background = 1.0 / np.sqrt(experiment.background)
    anomaly = 1.0 / np.sqrt(experiment.truth) - background
    limit = float(np.abs(anomaly).max())
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.1), constrained_layout=True)
    for ax, (name, m) in zip(
        axes, (("truth", experiment.truth), ("MAP mode", mode), ("Occam", model))
    ):
        image = section(
            ax,
            survey,
            1.0 / np.sqrt(m) - background,
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
        )
        sponge_outline(ax, survey)
        for _, (x0, x1, z0, z1) in BOXES:
            ax.add_patch(
                plt.Rectangle(
                    (x0, z0), x1 - x0, z1 - z0, fill=False, color="0.3", lw=0.6
                )
            )
        ax.set_title(
            f"{name}: chi-squared {problem.chi_squared(m, state.data):.0f}", fontsize=10
        )
    fig.colorbar(image, ax=axes, label="velocity less background (km/s)", shrink=0.9)
    save(fig, outdir / "occam.png")
    return {
        "model": model,
        "result": result,
        "target": target,
        "chi2": problem.chi_squared(model, state.data),
        "distance": {name: distance(m) for name, m in rows.items()},
        "error": experiment.relative_error(model),
        "means": means,
        "seconds": seconds,
    }


def run(
    scale: Scale,
    /,
    *,
    seed: int,
    outdir: pathlib.Path,
    state: Setup | None = None,
    mode: np.ndarray | None = None,
) -> dict:
    started = time.perf_counter()
    print(f"\n=== D, continued. Occam's inversion ({scale.name} scale) ===")
    if state is None:
        state = setup(scale, seed=seed)
    if mode is None:
        mode = map_mode(state, outdir=outdir, seed=seed)
    result = occam_answer(state, mode, outdir=outdir)
    print(f"D, continued, took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
