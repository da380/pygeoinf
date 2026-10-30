"""Part E: the Laplace Gaussian at the MAP mode, on the mean of m over four boxes.

The library's ``LaplaceApproximation`` about the polished mode of part C: a
Gaussian whose covariance is the inverse of the linearized normal operator
``P + J* R^-1 J`` there, in the model-space formalism, each application of
the inverse a conjugate-gradient solve. Pushed through the box means it is a
Gaussian on four numbers, whose covariance costs four solves; the prior's,
which costs nothing, is beside it. The boxes are one over each anomaly and
one over plain background, and the truth is marked against both.

The normal operator is the prior's precision, whose spectrum spans many
decades on a Sobolev space, plus the data's term, which informs some
hundreds of directions. Seen through the prior's covariance factor ``L``,
``L* (P + J* R^-1 J) L = I + L* J* R^-1 J L``, it is the identity but for
those directions, and they are what a randomized eigendecomposition finds:
the library's ``SpectralPreconditioner`` on that operator, its leading modes
inverted exactly and the rest by the identity's one, and ``L`` on either
side to take it back to the model space. It is paid once and shared by the
four solves.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

import joblib

from pygeoinf import LaplaceApproximation, LinearOperator, MaximumAPosteriori
from pygeoinf.numerics.preconditioners import SpectralPreconditioner
from pygeoinf.numerics.solvers import CGSolver
from pygeoinf.traits import Traits

from src.common import Stopwatch, describe, parse, part_rng, save
from src.inversion import map_mode
from src.survey import BOXES, FULL, QUICK, Scale, box_average_operator
from synthetic import Setup, setup

# CG stopped short of convergence underestimates a variance, the Krylov
# space not yet holding the directions it misses; at this tolerance the
# standard deviations have settled to three digits.
LAPLACE_RTOL = 1e-3


def laplace_answer(
    state: Setup,
    mode: np.ndarray,
    /,
    *,
    rng: np.random.Generator,
    outdir: pathlib.Path,
) -> dict:
    """The Laplace Gaussian on the box means, against the prior's, and the figure."""
    experiment, wave = state.experiment, state.wave
    scale = experiment.survey.scale
    prior = experiment.prior
    boxes = box_average_operator(experiment.space, experiment.survey)
    watch = Stopwatch(wave.calls)
    # Per solve, the residual and the box means of the iterate at each step:
    # the solves run one column at a time, so the k-th solve's k-th box mean
    # is the k-th variance as conjugate gradients see it at that step.
    solves: list[dict[str, list]] = []

    def record(step) -> None:
        if not solves or (step.iteration <= 1 and len(solves[-1]["residual"]) > 1):
            solves.append({"residual": [], "means": []})
        solves[-1]["residual"].append(step.residual)
        iterate = np.asarray(step.iterate)
        if iterate.shape != experiment.survey.shape:
            # The solver runs on component arrays where the space has them.
            iterate = experiment.space.from_components(iterate)
        solves[-1]["means"].append(np.asarray(boxes(iterate)))

    factor = prior.covariance_factor

    def solver(normal: LinearOperator) -> CGSolver:
        """Conjugate gradients on the normal operator, preconditioned by the
        prior-whitened operator's leading modes."""
        whitened = (factor.adjoint @ normal @ factor).with_traits(
            Traits.POSITIVE_DEFINITE
        )
        deflated = SpectralPreconditioner(
            rank=scale.laplace_rank, damping=1.0, rng=rng, n_jobs=scale.probe_jobs
        )(whitened)
        return CGSolver(
            rtol=LAPLACE_RTOL,
            max_iterations=scale.laplace_cg_iterations,
            preconditioner=factor @ deflated @ factor.adjoint,
            strict=False,
            callback=record,
        )

    # The decomposition's probes are independent applications of the normal
    # operator, run on threads: the kernels release the interpreter lock, and
    # the operator, which holds the C++ solver, cannot be sent to a process.
    with joblib.parallel_config(backend="threading"):
        laplace = LaplaceApproximation(
            MaximumAPosteriori(state.first_order, prior, gauss_newton=True),
            solver=solver,
        ).at(mode, state.data)
    build_seconds, build_calls = watch.lap()
    shots = experiment.survey.n_shots
    print(
        f"the spectral preconditioner, rank {scale.laplace_rank}, in "
        f"{build_seconds:.1f} s: {build_calls['born'] // shots} applications of the "
        f"normal operator"
    )
    on_boxes = laplace.affine_map(boxes)
    covariance = on_boxes.covariance.matrix(form="components")
    seconds, calls = watch.lap()
    applications = calls["born"] // shots
    print(
        f"the Laplace covariance on {len(BOXES)} box means in {seconds:.1f} s: "
        f"{applications} applications of the normal operator, kernels "
        f"{describe(calls)}"
    )
    iterations = [len(solve["residual"]) - 1 for solve in solves]
    reached = [solve["residual"][-1] / solve["residual"][0] for solve in solves]
    print(
        f"  {len(solves)} conjugate-gradient solves of {iterations} iterations, "
        f"residuals reduced to {max(reached):.1e} at worst (asked {LAPLACE_RTOL:g}, "
        f"cap {scale.laplace_cg_iterations})"
    )
    # How the standard deviations settled: each as conjugate gradients had it
    # at half and three quarters of its solve, against the end.
    settling = []
    for k, solve in enumerate(solves):
        variances = [means[k] for means in solve["means"]]
        last = np.sqrt(max(variances[-1], 0.0))
        settling.append(
            [
                np.sqrt(max(variances[int(fraction * (len(variances) - 1))], 0.0))
                / last
                for fraction in (0.5, 0.75)
            ]
        )
    settling = np.array(settling)
    print(
        "  each standard deviation at half and three quarters of its solve, over its "
        "final value: " + ", ".join(f"{a:.3f}/{b:.3f}" for a, b in settling)
    )
    prior_covariance = prior.affine_map(boxes).covariance.matrix(form="components")
    center = np.asarray(on_boxes.expectation)
    std = np.sqrt(np.diag(covariance))
    prior_center = np.asarray(boxes(experiment.background))
    prior_std = np.sqrt(np.diag(prior_covariance))
    truth = np.asarray(boxes(experiment.truth))
    inside = np.abs(truth - center) <= 2.0 * std
    print(
        f"  {'box':<14s} {'truth':>8s} {'prior':>8s} {'+/-2sd':>7s} {'Laplace':>8s} "
        f"{'+/-2sd':>7s} {'sd ratio':>8s}  truth within 2 sd"
    )
    for k, (name, _) in enumerate(BOXES):
        print(
            f"  {name:<14s} {truth[k]:8.4f} {prior_center[k]:8.4f} {2 * prior_std[k]:7.4f} "
            f"{center[k]:8.4f} {2 * std[k]:7.4f} {std[k] / prior_std[k]:8.3f}  "
            f"{'yes' if inside[k] else 'no'}"
        )
    correlation = covariance / np.outer(std, std)
    off = correlation[~np.eye(len(BOXES), dtype=bool)]
    print(
        f"  posterior correlations between boxes from {off.min():+.2f} to {off.max():+.2f}"
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = np.arange(len(BOXES))[::-1]
    ax.errorbar(
        prior_center,
        rows + 0.15,
        xerr=2 * prior_std,
        fmt="s",
        color="0.55",
        capsize=3,
        label="prior, 2 sd",
    )
    ax.errorbar(
        center,
        rows - 0.15,
        xerr=2 * std,
        fmt="o",
        color="#2a78d6",
        capsize=3,
        label="Laplace at the mode, 2 sd",
    )
    ax.plot(truth, rows, "|", color="#eb6834", ms=18, mew=2.5, label="truth")
    ax.set_yticks(rows, [name for name, _ in BOXES])
    ax.set_xlabel("mean of m over the box (s²/km²)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("The mean slowness squared in four boxes", fontsize=10)
    save(
        fig,
        outdir / "laplace.png",
        title="Waveform, part E: the Laplace Gaussian at the mode, prior against "
        "posterior on four box means",
    )
    return {
        "center": center,
        "std": std,
        "prior_center": prior_center,
        "prior_std": prior_std,
        "truth": truth,
        "inside": inside,
        "correlation": correlation,
        "seconds": seconds + build_seconds,
        "applications": applications + build_calls["born"] // shots,
        "iterations": iterations,
        "settling": settling,
    }


def run(
    scale: Scale,
    /,
    *,
    outdir: pathlib.Path,
    state: Setup | None = None,
    mode: np.ndarray | None = None,
) -> dict:
    started = time.perf_counter()
    print(f"\n=== E. The Laplace Gaussian on four box means ({scale.name} scale) ===")
    if state is None:
        state = setup(scale)
    if mode is None:
        mode = map_mode(state, outdir=outdir)
    result = laplace_answer(state, mode, rng=part_rng(scale.seed, "e"), outdir=outdir)
    print(f"E took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
