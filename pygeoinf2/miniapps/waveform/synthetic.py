"""Part B: synthetic data from the true model, with noise.

The truth's receiver data for every shot, Gaussian noise of a fixed fraction
of the data's root-mean-square added sample by sample, and the forward
problem the later parts invert. How much the anomalies matter is read off
the misfit of the known background against the data, and where they are
seen from is the gradient of that misfit at the background: the classic
first image of waveform inversion, here the model-space gradient, so smooth
in the Sobolev metric.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import GaussianMeasure
from pygeoinf2.inference import ForwardProblem

from src.common import parse, part_rng, save, section
from src.survey import (
    BOXES,
    DEPTH_CELLS,
    FULL,
    QUICK,
    Experiment,
    Scale,
    build,
)
from src.wave_operator import WaveformOperator


@dataclass
class Setup:
    """The experiment, its operator, the data and the two forward problems."""

    experiment: Experiment
    wave: WaveformOperator
    clean: np.ndarray
    data: np.ndarray
    noise_std: float
    problem: ForwardProblem
    first_order: ForwardProblem


def setup(scale: Scale, /, *, quiet: bool = False, backend: Any = None) -> Setup:
    """The data and the forward problems, the same for every part from one seed.

    Args:
        scale: the size of the problem, and the seed of the noise.
        quiet: say nothing.
        backend: ``acoustic_mpi`` to split the grid across MPI ranks, the shots
            then running one after another; the serial ``acoustic`` if omitted.
    """
    say = (lambda *args: None) if quiet else print
    started = time.perf_counter()
    experiment = build(scale, backend=backend)
    survey = experiment.survey
    serial = backend is None
    wave = WaveformOperator(
        survey,
        experiment.space,
        n_jobs=scale.n_jobs if serial else None,
        curvature_jobs=scale.curvature_jobs if serial else None,
    )
    clean = wave.value(experiment.truth)
    noise_std = scale.noise_fraction * float(np.sqrt(np.mean(clean**2)))
    noise = GaussianMeasure.from_standard_deviation(wave.data_space, noise_std)
    data = clean + noise.sample(rng=part_rng(scale.seed, "b"))
    problem = ForwardProblem(wave.operator, error=noise)
    first_order = ForwardProblem(wave.first_order, error=noise)
    width, depth = survey.section_lengths
    jobs = scale.curvature_jobs if serial and scale.curvature_jobs else 1
    say(
        f"section {width:.1f} km by {depth:.1f} km, in a grid of {survey.shape[0]} x "
        f"{survey.shape[1]} cells of {survey.scale.h * 1000:.0f} m with a sponge "
        f"{scale.sponge} cells wide beyond it; {survey.scale.steps} steps of "
        f"{survey.scale.dt * 1000:.1f} ms; Ricker {survey.scale.peak_frequency:g} Hz"
    )
    say(
        f"{survey.n_shots} shots, {survey.n_receivers} receivers each, "
        f"{wave.data_space.dim} data; noise std {noise_std:.3g} "
        f"({scale.noise_fraction:.0%} of the data's rms); one stored field "
        f"{survey.field_megabytes:.0f} MB, {survey.n_shots} cached and {jobs} more "
        f"for the curvature: {(survey.n_shots + jobs) * survey.field_megabytes:.0f} MB "
        "at most"
    )
    say(f"setup took {time.perf_counter() - started:.1f} s")
    return Setup(experiment, wave, clean, data, noise_std, problem, first_order)


def data_answer(state: Setup, /, *, outdir: pathlib.Path) -> dict:
    """The misfits that say how much the anomalies matter, and the figure."""
    experiment, wave, problem = state.experiment, state.wave, state.problem
    survey = experiment.survey
    t0 = time.perf_counter()
    background_data = wave.value(experiment.background)
    misfit = {
        "truth": problem.chi_squared(experiment.truth, state.data),
        "background": problem.chi_squared(experiment.background, state.data),
    }
    level = survey.scale.level
    critical = problem.critical_chi_squared(level=level)
    signal = float(
        np.sqrt(np.mean((state.clean - background_data) ** 2))
        / np.sqrt(np.mean(state.clean**2))
    )
    print(
        f"chi-squared of the truth {misfit['truth']:.0f}, of the background "
        f"{misfit['background']:.0f}, on {wave.data_space.dim} data (critical at "
        f"{level:.0%}: {critical:.0f}); the anomalies are {signal:.1%} of the "
        f"data's rms"
    )
    # The misfit's gradient at the background, which the prior term does not
    # touch there: -2 F'(m0)* R^-1 (d - F(m0)).
    residual = (state.data - background_data) / state.noise_std**2
    gradient = experiment.space.scale(
        -2.0, wave.derivative(experiment.background).adjoint(residual)
    )
    print(
        f"the misfit's gradient at the background in {time.perf_counter() - t0:.1f} s"
    )

    shot = survey.n_shots // 2
    shape = survey.data_shape
    gathers = [
        state.data.reshape(shape)[shot],
        (state.clean - background_data).reshape(shape)[shot],
    ]
    velocity = 1.0 / np.sqrt(experiment.truth)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    image = section(axes[0, 0], survey, velocity, cmap="viridis")
    fig.colorbar(image, ax=axes[0, 0], label="km/s", shrink=0.8)
    for name, (x0, x1, z0, z1) in BOXES:
        axes[0, 0].add_patch(
            plt.Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, color="w", lw=1)
        )
        axes[0, 0].text(x0, z0 - 0.01, name, color="w", fontsize=7, va="bottom")
    axes[0, 0].plot(
        survey.x[survey.source_x],
        np.full(survey.n_shots, survey.z[DEPTH_CELLS]),
        "*",
        color="k",
        ms=9,
    )
    axes[0, 0].set_title("the true velocity, the boxes asked about, the shots")
    limit = float(np.abs(gradient[survey.window]).max())
    image = section(
        axes[0, 1], survey, gradient, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    fig.colorbar(image, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].set_title("the misfit's gradient at the background")
    extent = (
        survey.x[survey.receiver_x[0]],
        survey.x[survey.receiver_x[-1]],
        survey.times[-1],
        survey.times[0],
    )
    for ax, gather, title in zip(
        axes[1],
        gathers,
        ("the data of the middle shot", "what the anomalies add to it"),
    ):
        clip = np.percentile(np.abs(gather), 98)
        ax.imshow(
            gather, aspect="auto", cmap="Greys", vmin=-clip, vmax=clip, extent=extent
        )
        ax.set_xlabel("receiver x (km)")
        ax.set_ylabel("time (s)")
        ax.set_title(f"{title} (clipped at {clip:.2g})", fontsize=10)
    save(
        fig,
        outdir / "data.png",
        title="Waveform, part B: the synthetic data, and what the anomalies add to "
        "the background's",
    )
    return {
        "misfit": misfit,
        "critical": critical,
        "signal": signal,
        "gradient": gradient,
    }


def run(scale: Scale, /, *, outdir: pathlib.Path, state: Setup | None = None) -> dict:
    started = time.perf_counter()
    print(f"\n=== B. Synthetic data ({scale.name} scale) ===")
    if state is None:
        state = setup(scale)
    result = data_answer(state, outdir=outdir)
    result["setup"] = state
    print(f"B took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
