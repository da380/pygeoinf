"""Part C: the MAP model by Newton-CG, on the Gauss-Newton Hessian and then the exact one.

Twice the negative log posterior, ``chi^2(m, d) + (m - m0, P (m - m0))``,
minimized by the library's ``MaximumAPosteriori`` with Newton-CG, the prior
covariance as its preconditioner. First from the background with the
Gauss-Newton Hessian ``2 (P + J* R^-1 J)``; then, from where that stopped,
with the exact Hessian, which adds the forward operator's second derivative
weighted by the residual, ``-2 F''(m)[., .]* R^-1 r``, and tighter inner
solves. The exact Hessian is not definite far from the mode, and truncated
conjugate gradients stop at the first direction of negative curvature it
shows, so a short run of the exact Newton from the background is the third
run, to show what that costs. The recovered models against the truth, and
what each run cost in kernel runs and seconds. The modes are saved for
part D.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import MaximumAPosteriori

from common import (
    Stopwatch,
    describe,
    output_directory,
    parse,
    save,
    section,
    sponge_outline,
)
from inversion import modes_path, newton
from survey import FULL, QUICK, Scale
from synthetic import Setup, setup

COLORS = {"gauss-newton": "#2a78d6", "polish": "#1baf7a", "exact": "#eb6834"}


def _run(label: str, estimator, optimizer, state: Setup, start) -> dict:
    wave, experiment = state.wave, state.experiment
    watch = Stopwatch(wave.calls)
    result = estimator.solve(state.data, start=start)
    seconds, calls = watch.lap()
    model = result.minimizer
    chi2 = state.problem.chi_squared(model, state.data)
    error = experiment.relative_error(model)
    print(
        f"{label}: {result.iterations} iterations ({result.message}); value "
        f"{result.value:.2f}, |grad| {result.gradient_norm:.3g}; chi-squared "
        f"{chi2:.0f}; relative error {error:.3f}; {seconds:.1f} s; kernels "
        f"{describe(calls)}"
    )
    return {
        "model": model,
        "result": result,
        "log": list(optimizer.log),
        "chi2": chi2,
        "error": error,
        "seconds": seconds,
        "calls": calls,
    }


def map_answer(
    state: Setup, /, *, outdir: pathlib.Path, seed: int, figure: bool = True
) -> dict:
    """The three Newton runs, the saved modes and the figure."""
    experiment = state.experiment
    scale = experiment.survey.scale
    prior = experiment.prior
    truth_chi2 = state.problem.chi_squared(experiment.truth, state.data)
    print(
        f"chi-squared of the truth {truth_chi2:.0f}, critical "
        f"{state.problem.critical_chi_squared():.0f}; the background's relative "
        f"error is 1 by definition"
    )
    runs = {}
    optimizer = newton(experiment, scale)
    runs["gauss-newton"] = _run(
        "Gauss-Newton Hessian from the background",
        MaximumAPosteriori(
            state.first_order, prior, optimizer=optimizer, gauss_newton=True
        ),
        optimizer,
        state,
        experiment.background,
    )
    optimizer = newton(
        experiment,
        scale,
        iterations=scale.polish_iterations,
        forcing=0.01,
        ftol=0.0,
        cg_iterations=scale.polish_cg_iterations,
    )
    runs["polish"] = _run(
        "exact Hessian from there",
        MaximumAPosteriori(state.problem, prior, optimizer=optimizer),
        optimizer,
        state,
        runs["gauss-newton"]["model"],
    )
    optimizer = newton(experiment, scale, iterations=scale.exact_iterations)
    runs["exact"] = _run(
        "exact Hessian from the background",
        MaximumAPosteriori(state.problem, prior, optimizer=optimizer),
        optimizer,
        state,
        experiment.background,
    )
    np.savez(
        modes_path(outdir, scale, seed),
        gauss_newton=runs["gauss-newton"]["model"],
        polish=runs["polish"]["model"],
    )
    if figure:
        draw(state, runs, outdir=outdir)
    return runs


def draw(state: Setup, runs: dict, /, *, outdir: pathlib.Path) -> None:
    """The anomalies recovered, and the convergence of the three runs."""
    experiment = state.experiment
    survey = experiment.survey
    background = 1.0 / np.sqrt(experiment.background)
    anomaly = 1.0 / np.sqrt(experiment.truth) - background
    limit = float(np.abs(anomaly).max())
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    panels = (
        ("the truth", anomaly),
        ("Gauss-Newton", 1.0 / np.sqrt(runs["gauss-newton"]["model"]) - background),
        ("then the exact Hessian", 1.0 / np.sqrt(runs["polish"]["model"]) - background),
        (
            f"exact Hessian from the background ({runs['exact']['result'].iterations}"
            " iterations)",
            1.0 / np.sqrt(runs["exact"]["model"]) - background,
        ),
    )
    for k, (title, values) in enumerate(panels):
        ax = fig.add_subplot(grid[k // 2, k % 2])
        image = section(ax, survey, values, cmap="RdBu", vmin=-limit, vmax=limit)
        sponge_outline(ax, survey)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, label="km/s", shrink=0.8)
    ax = fig.add_subplot(grid[2, :])
    first = len(runs["gauss-newton"]["log"]) - 1
    for name, run in runs.items():
        norms = [norm for _, norm in run["log"]]
        offset = first if name == "polish" else 0
        ax.semilogy(
            np.arange(len(norms)) + offset,
            norms,
            "o-",
            color=COLORS[name],
            label={
                "gauss-newton": "Gauss-Newton",
                "polish": "then the exact Hessian",
                "exact": "the exact Hessian from the background",
            }[name],
        )
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("gradient norm")
    ax.legend(frameon=False)
    fig.suptitle("The velocity less the background, and how the three runs converged")
    fig.tight_layout()
    save(fig, outdir / "map.png")


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, state: Setup | None = None
) -> dict:
    started = time.perf_counter()
    print(f"\n=== C. The MAP model by Newton-CG ({scale.name} scale) ===")
    if state is None:
        state = setup(scale, seed=seed)
    runs = map_answer(state, outdir=outdir, seed=seed)
    print(f"C took {time.perf_counter() - started:.1f} s")
    return {"runs": runs, "setup": state}


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
