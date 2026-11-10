"""Part G: GeometricHMC about the Laplace Gaussian, on the Gauss-Newton and the exact Hessian.

The library's ``GeometricHMC`` samples the posterior itself, not an
approximation of it: its reference is a Gaussian equivalent to the prior,
whose part of the dynamics is solved exactly, and the Metropolis step
corrects the rest, so the reference decides how fast the chains mix and
never what they converge to. Here the reference is the Laplace Gaussian at
the polished mode of part C, in the low-rank chart form that
``LaplaceApproximation.reference`` builds: the prior with the law of the
leading directions of the misfit's curvature, seen through the prior's
covariance factor, replaced by the curvature's. That curvature is the
Gauss-Newton ``J* R^-1 J`` for one sampler and the exact Hessian for the
other, the forward operator's second derivative weighted by the residual
added by second-order adjoints, its negative eigenvalues clipped. The rank
is part E's, the scale's ``laplace_rank``.

Both samplers run on the same data and from the same seed, their chains one
after another, each step a trajectory of ``hmc_length`` leapfrog steps whose
time step adapts during burn-in. What is compared is how they converge, the
chains' autocorrelation and effective sample size for the potential and the
four box means of part E, and what they find, the box means' marginals and
the pointwise spread of the velocity, against the two Laplace Gaussians and,
run in order, part E's full-rank Laplace Gaussian.

At the quick scale only, and on the serial build of the solver: the chains
evaluate the forward map some thousands of times.
"""

from __future__ import annotations

import pathlib
import time
import warnings
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from pygeoinf import LaplaceApproximation
from pygeoinf.inference.sampling import Chains, GeometricHMC
from pygeoinf.numerics.chain_diagnostics import autocorrelation

from src.common import Stopwatch, describe, parse, part_rng, save, section
from src.inversion import map_mode
from src.survey import BOXES, FULL, QUICK, Scale, box_average_operator
from synthetic import Setup, setup

COLORS = {"gauss-newton": "#2a78d6", "exact": "#eb6834"}
DASHES = {"gauss-newton": "--", "exact": ":"}
LABELS = {"gauss-newton": "Gauss-Newton Hessian", "exact": "exact Hessian"}
SHORT = {"gauss-newton": "Gauss-Newton", "exact": "exact"}


def references(state: Setup, mode: np.ndarray, /, *, rng: np.random.Generator) -> dict:
    """The low-rank Laplace reference at the mode for each Hessian, timed."""
    experiment, wave = state.experiment, state.wave
    scale = experiment.survey.scale
    shots = experiment.survey.n_shots
    built = {}
    for (name, problem, gauss_newton), stream in zip(
        (
            ("gauss-newton", state.first_order, True),
            ("exact", state.problem, False),
        ),
        rng.spawn(2),
    ):
        watch = Stopwatch(wave.calls)
        laplace = LaplaceApproximation.from_problem(
            problem,
            experiment.prior,
            gauss_newton=gauss_newton,
            rank=scale.laplace_rank,
            rng=stream,
            n_jobs=scale.probe_jobs,
        )
        # The probes are applications at the one model, run on threads as
        # part E runs its own.
        with joblib.parallel_config(backend="threading"):
            reference = laplace.reference(mode, state.data)
        seconds, calls = watch.lap()
        print(
            f"the {LABELS[name]} reference, rank {scale.laplace_rank}, chart of "
            f"{reference.rank}, in {seconds:.1f} s: {calls['adjoint'] // shots} "
            f"applications of its curvature, kernels {describe(calls)}"
        )
        built[name] = {"reference": reference, "seconds": seconds, "calls": calls}
    return built


def chains(
    state: Setup, reference: Any, /, *, rng: np.random.Generator, name: str
) -> dict:
    """GeometricHMC about one reference: the sampled posterior and its traces."""
    experiment, wave = state.experiment, state.wave
    scale = experiment.survey.scale
    boxes = box_average_operator(experiment.space, experiment.survey)
    sampler = GeometricHMC(
        state.problem, reference, step=scale.hmc_step, length=scale.hmc_length
    )
    # What calling the preset does, with its kernel kept: the callback reads
    # the potential its target has already evaluated at each state, so that
    # the burn-in is traced at no cost.
    kernel = sampler.kernel(state.data)
    total = scale.hmc_burn_in + scale.hmc_steps
    traces = np.full((scale.hmc_chains, total), np.nan)

    def trace(chain: int, iteration: int, step: Any) -> None:
        traces[chain, iteration] = kernel.target.potential(step.state)

    watch = Stopwatch(wave.calls)
    posterior = Chains(scale.hmc_chains)(
        kernel,
        start=reference,
        steps=scale.hmc_steps,
        burn_in=scale.hmc_burn_in,
        adapt=sampler.adaptation,
        thin=1,
        record=boxes,
        record_names=[box for box, _ in BOXES],
        rng=rng,
        callback=trace,
    )
    seconds, calls = watch.lap()
    gradients = calls["adjoint"] // experiment.survey.n_shots
    diagnostics = posterior.diagnostics()
    step = float(np.mean([chain.step_size for chain in posterior.chains]))
    print(
        f"GeometricHMC about the {LABELS[name]} reference: {scale.hmc_chains} chains "
        f"of {scale.hmc_burn_in} + {scale.hmc_steps} steps of {scale.hmc_length} "
        f"leapfrog steps, step {step:.3f} after burn-in, acceptance "
        f"{posterior.acceptance_rate:.2f}; {gradients} gradients in {seconds:.1f} s, "
        f"kernels {describe(calls)}"
    )
    print(
        f"  {'scalar':<14s} {'tau':>6s} {'ESS':>7s} {'R-hat':>6s}  ESS per 100 gradients"
    )
    for k, scalar in enumerate(diagnostics.names):
        ess = diagnostics.effective_sample_size[k]
        print(
            f"  {scalar:<14s} {diagnostics.autocorrelation_time[k]:6.2f} {ess:7.1f} "
            f"{diagnostics.rhat[k]:6.3f}  {100 * ess / gradients:.2f}"
        )
    return {
        "posterior": posterior,
        "traces": traces,
        "diagnostics": diagnostics,
        "step": step,
        "seconds": seconds,
        "gradients": gradients,
    }


def velocity_spread(models: list, /) -> tuple[np.ndarray, np.ndarray, int]:
    """The pointwise mean and standard deviation of the velocity over models,
    and how many of the models are not slownesses squared everywhere.

    A Gaussian's draw of ``m`` can fall to zero or below where the posterior
    is wide, and has no velocity there; each cell's statistics are over the
    models that have one, and are NaN where none does."""
    m = np.stack([np.asarray(model) for model in models])
    physical = m > 0.0
    velocity = np.where(physical, 1.0 / np.sqrt(np.where(physical, m, 1.0)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(velocity, axis=0)
        spread = np.nanstd(velocity, axis=0, ddof=1)
    unphysical = int(np.sum(~physical.reshape(len(models), -1).all(axis=1)))
    return mean, spread, unphysical


def hmc_answer(
    state: Setup,
    mode: np.ndarray,
    /,
    *,
    rng: np.random.Generator,
    outdir: pathlib.Path,
    laplace: dict | None = None,
) -> dict:
    """The two samplers, their comparison with the Laplace Gaussians, and the figures."""
    experiment = state.experiment
    boxes = box_average_operator(experiment.space, experiment.survey)
    reference_rng, chain_rng, draw_rng = rng.spawn(3)
    built = references(state, mode, rng=reference_rng)
    # The same chain seed for both samplers: they differ in the reference alone.
    chain_seed = int(chain_rng.integers(2**62))
    runs = {
        name: chains(
            state, entry["reference"], rng=np.random.default_rng(chain_seed), name=name
        )
        for name, entry in built.items()
    }

    truth = np.asarray(boxes(experiment.truth))
    at_mode = np.asarray(boxes(mode))
    gaussians = {}
    for name, entry in built.items():
        on_boxes = entry["reference"].affine_map(boxes)
        covariance = on_boxes.covariance.matrix(form="components")
        gaussians[name] = (
            np.asarray(on_boxes.expectation),
            np.sqrt(np.diag(covariance)),
        )
    sampled = {}
    for name, run in runs.items():
        values = run["posterior"].scalars()[1:]
        errors = [
            estimate.standard_error for estimate in run["posterior"].scalar_means()[1:]
        ]
        sampled[name] = (
            values.mean(axis=(1, 2)),
            values.reshape(len(BOXES), -1).std(axis=1, ddof=1),
            np.array(errors),
        )

    columns = [
        ("Laplace GN", gaussians["gauss-newton"]),
        ("Laplace ex", gaussians["exact"]),
    ]
    if laplace is not None:
        columns.append(("part E", (laplace["center"], laplace["std"])))
    columns += [
        ("HMC GN", sampled["gauss-newton"][:2]),
        ("HMC ex", sampled["exact"][:2]),
    ]
    print(
        "the box means, mean +/- 2 sd; GN the Gauss-Newton Hessian, ex the exact, "
        "part E the full-rank Gauss-Newton Laplace Gaussian"
    )
    print(
        f"  {'box':<14s} {'truth':>7s} {'MAP':>7s} "
        + " ".join(f"{title:>18s}" for title, _ in columns)
    )
    for k, (box, _) in enumerate(BOXES):
        print(
            f"  {box:<14s} {truth[k]:7.4f} {at_mode[k]:7.4f} "
            + " ".join(f"{c[k]:8.4f} +/-{2 * s[k]:.4f}" for _, (c, s) in columns)
        )
    for name in runs:
        center, spread, error = sampled[name]
        gaussian_center, gaussian_spread = gaussians[name]
        print(
            f"  HMC about the {LABELS[name]} reference against its Laplace Gaussian: "
            f"mean offsets "
            + ", ".join(
                f"{d:+.2f}" for d in (center - gaussian_center) / gaussian_spread
            )
            + " sd (Monte Carlo errors "
            + ", ".join(f"{e:.2f}" for e in error / gaussian_spread)
            + "); sd ratios "
            + ", ".join(f"{r:.2f}" for r in spread / gaussian_spread)
        )

    # The velocity's pointwise spread: the chains' stored states against as
    # many draws of the reference, so that the two carry noise of about the
    # same size.
    fields = {}
    inside = experiment.survey.interior()
    for (name, run), stream in zip(runs.items(), draw_rng.spawn(len(runs))):
        stored = list(run["posterior"].stored_positions)
        draws = built[name]["reference"].samples(len(stored), rng=stream)
        mean, spread, _ = velocity_spread(stored)
        _, laplace_spread, unphysical = velocity_spread(draws)
        shift = mean - 1.0 / np.sqrt(mode)
        fields[name] = {"laplace": laplace_spread, "sampled": spread, "shift": shift}
        ratio = spread[inside] / laplace_spread[inside]
        ratio = ratio[np.isfinite(ratio)]
        print(
            f"  the velocity's pointwise sd over the section, HMC about the "
            f"{LABELS[name]} reference over its Laplace Gaussian's, from "
            f"{len(stored)} states and draws each: median {np.median(ratio):.2f}, "
            f"5-95% {np.percentile(ratio, 5):.2f}-{np.percentile(ratio, 95):.2f}; "
            f"the sampled mean less the MAP velocity, rms "
            f"{1000 * np.sqrt(np.nanmean(shift[inside] ** 2)):.1f} m/s against a "
            f"median sd of {1000 * np.nanmedian(spread[inside]):.1f} m/s"
        )
        if unphysical:
            print(
                f"  {unphysical} of the {len(draws)} Laplace draws have m <= 0 in "
                "some cell, which has no velocity; each cell's sd is over the "
                "draws that have one"
            )

    draw_convergence(runs, outdir=outdir)
    draw_posterior(
        state, runs, gaussians, fields, truth, at_mode, laplace, outdir=outdir
    )
    return {
        "references": built,
        "runs": runs,
        "gaussians": gaussians,
        "sampled": sampled,
        "fields": fields,
        "truth": truth,
        "map": at_mode,
    }


def draw_convergence(runs: dict, /, *, outdir: pathlib.Path) -> None:
    """The potential along every chain, burn-in included, and the scalars'
    autocorrelation, for each reference."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
    lags = 25
    for column, (name, run) in enumerate(runs.items()):
        ax = axes[0, column]
        traces = run["traces"]
        burn_in = traces.shape[1] - run["posterior"].chains[0].steps
        for chain in traces:
            ax.plot(chain, lw=0.9, color=COLORS[name], alpha=0.8)
        ax.axvspan(0, burn_in, color="0.92", zorder=0)
        ax.set_xlabel("step (burn-in shaded)")
        ax.set_title(
            f"about the {LABELS[name]} reference: acceptance "
            f"{run['posterior'].acceptance_rate:.2f}, step {run['step']:.3f}",
            fontsize=10,
        )
        ax = axes[1, column]
        diagnostics = run["diagnostics"]
        values = run["posterior"].scalars()
        for k, scalar in enumerate(diagnostics.names):
            rho = autocorrelation(values[k]).mean(axis=0)[: lags + 1]
            ax.plot(
                rho,
                "o-" if k == 0 else ".-",
                ms=3,
                lw=1.2 if k == 0 else 0.9,
                color="k" if k == 0 else plt.cm.viridis(k / len(diagnostics.names)),
                label=f"{scalar} (ESS {diagnostics.effective_sample_size[k]:.0f})",
            )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("lag (steps)")
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("likelihood potential")
    axes[1, 0].set_ylabel("autocorrelation")
    save(
        fig,
        outdir / "hmc_convergence.png",
        title="Waveform, part G: GeometricHMC about the Laplace Gaussian on the "
        "Gauss-Newton and on the exact Hessian",
    )


def draw_posterior(
    state: Setup,
    runs: dict,
    gaussians: dict,
    fields: dict,
    truth: np.ndarray,
    at_mode: np.ndarray,
    laplace: dict | None,
    /,
    *,
    outdir: pathlib.Path,
) -> None:
    """The box means' marginals against the Laplace Gaussians, and the
    velocity's pointwise spread, sampled and Laplace, with the sampled mean
    less the MAP velocity."""
    survey = state.experiment.survey
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])
    for k, (box, _) in enumerate(BOXES):
        ax = fig.add_subplot(grid[0, k])
        pooled = {
            name: run["posterior"].scalars()[1 + k].ravel()
            for name, run in runs.items()
        }
        low = min(v.min() for v in pooled.values())
        high = max(v.max() for v in pooled.values())
        width = max(s[k] for _, s in gaussians.values())
        x = np.linspace(min(low, truth[k]) - width, max(high, truth[k]) + width, 300)
        bins = np.linspace(low, high, 25)
        for name, values in pooled.items():
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.4,
                color=COLORS[name],
                label=f"HMC, {LABELS[name]}",
            )
            center, spread = gaussians[name][0][k], gaussians[name][1][k]
            ax.plot(
                x,
                np.exp(-0.5 * ((x - center) / spread) ** 2)
                / (spread * np.sqrt(2 * np.pi)),
                color=COLORS[name],
                lw=1.3,
                ls=DASHES[name],
                label=f"Laplace, {LABELS[name]}",
            )
        if laplace is not None:
            center, spread = laplace["center"][k], laplace["std"][k]
            ax.plot(
                x,
                np.exp(-0.5 * ((x - center) / spread) ** 2)
                / (spread * np.sqrt(2 * np.pi)),
                color="0.35",
                lw=1.0,
                ls="-.",
                label="Laplace, part E's full-rank Gauss-Newton",
            )
        ax.axvline(truth[k], color="k", lw=1.5, label="truth")
        ax.axvline(at_mode[k], color="0.5", lw=1.0, label="MAP")
        ax.set_title(box, fontsize=10)
        ax.set_xlabel("mean of m (s²/km²)")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_yticks([])
        if k == 0:
            fig.legend(frameon=False, fontsize=8, ncols=4, loc="outside lower center")
    for row, (name, field) in enumerate(fields.items(), start=1):
        spread_limit = 1000 * max(
            float(np.nanmax(field[key][survey.window]))
            for key in ("laplace", "sampled")
        )
        shift_limit = 1000 * float(np.nanmax(np.abs(field["shift"][survey.window])))
        panels = (
            ("laplace", f"Laplace sd, {SHORT[name]}", "magma", 0.0, spread_limit),
            ("sampled", f"sampled sd, {SHORT[name]}", "magma", 0.0, spread_limit),
            (
                "shift",
                "sampled mean less the MAP velocity",
                "RdBu",
                -shift_limit,
                shift_limit,
            ),
        )
        for column, (key, title, cmap, low, high) in enumerate(panels):
            ax = fig.add_subplot(grid[row, column])
            image = section(
                ax, survey, 1000 * field[key], cmap=cmap, vmin=low, vmax=high
            )
            ax.set_title(title, fontsize=9)
            fig.colorbar(image, ax=ax, label="m/s", shrink=0.8)
        ax = fig.add_subplot(grid[row, 3])
        inside = survey.interior()
        ratio = field["sampled"][inside] / field["laplace"][inside]
        ax.hist(ratio[np.isfinite(ratio)], bins=40, color=COLORS[name])
        ax.axvline(1.0, color="k", lw=1.0)
        ax.set_xlabel("sampled sd / Laplace sd, per cell")
        ax.set_yticks([])
    save(
        fig,
        outdir / "hmc_posterior.png",
        title="Waveform, part G: the sampled posterior against the Laplace Gaussians. "
        "The box means' marginals; the velocity's pointwise spread",
    )


def run(
    scale: Scale,
    /,
    *,
    outdir: pathlib.Path,
    state: Setup | None = None,
    mode: np.ndarray | None = None,
    laplace: dict | None = None,
) -> dict:
    """Part G on a quick scale and the serial solver.

    Raises:
        ValueError: if the scale is not the quick preset's, or the state's
            solver is not the serial build.
    """
    if scale.name != QUICK.name:
        raise ValueError(f"Part G runs at the quick scale only, not the {scale.name}.")
    started = time.perf_counter()
    print(f"\n=== G. GeometricHMC about the Laplace Gaussian ({scale.name} scale) ===")
    if state is None:
        state = setup(scale)
    if state.experiment.survey.backend is not None:
        raise ValueError("Part G runs on the serial build of the solver only.")
    if mode is None:
        mode = map_mode(state, outdir=outdir)
    result = hmc_answer(
        state, mode, rng=part_rng(scale.seed, "g"), outdir=outdir, laplace=laplace
    )
    print(f"G took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    if not args.quick:
        raise SystemExit("Part G runs at the quick scale only: pass --quick.")
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
