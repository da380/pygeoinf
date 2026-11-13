"""Part D: Gibbs over the fields, on the whole problem.

The library's ``Gibbs``, one move to each summand of the model space, each
against its conditional under the coupled prior, run as chains by the Monte
Carlo estimator. The main run has two blocks.
HMC moves the part of the model the forward map is linear in, the coupled thermal
fields and the contrasts inside the provinces, as one summand of a nested direct
sum; pCN moves the level set, which needs no gradient and has none. Each block's
step adapts to its own acceptance during burn-in. The forward map's derivative is
zero in the level set, and only the linear block ever follows it.

Why the linear part is one block is measured here, against the three blocks
that the model's fields suggest. A thermal field can absorb a smooth part of a
contrast, so the two trade off along a valley. Given the thermal fields a contrast
is pinned to a small fraction of its marginal spread, and a sampler that moves them
in turn creeps along the valley where one that moves them together walks it.

The other hard block is the level set. A sharp indicator gives the data a say over
the provinces' edges and over nothing else, so that a pCN step small enough to keep
an edge where the data put it barely moves the rest of the field. What that costs is
measured by how far each block moves per forward evaluation: with more pCN steps a
visit, under a symmetric scan, and from chains started from the prior, which find
another arrangement of provinces and stay there. The comparisons are short, far
shorter than their chains' autocorrelation times, so that a jump distance is
what they can honestly report and an effective sample size is not.

The main chains start from a guess the data make: the provinces the slowest quarter
of the thermal-only posterior mean of part C, and the linear part its exact
conditional mean given those provinces, a linear Gaussian problem. Each chain
adds a tenth of a prior draw of its own.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    HMC,
    MALA,
    PCN,
    Chains,
    ForwardProblem,
    GaussianMeasure,
    Gibbs,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    MonteCarloEstimator,
)
from pygeoinf.algebra.direct_sum import RowLinearOperator
from pygeoinf.numerics.solvers import CGSolver

from src.common import Stopwatch, parse, part_rng, save
from src.prior import grouped, linear_prior, ungrouped
from src.record import NAMES, area_weights, record_operator, summary
from src.shell import FULL, QUICK, Scale
from synthetic import Setup, setup

SOLVER_RTOL = 1e-8
# The scalars the tables print, and what a comparison measures the moves of.
SHOWN = ("c_s", "c_p", "c_rho", "share", "ratio")
MOVES = ("c_s", "c_rho", "share")
CHAIN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


def conditional_mean(state: Setup, phi: Any, /) -> tuple[Any, np.ndarray]:
    """The thermal fields and the contrasts given a level set: the mean of the
    linear Gaussian posterior their data make when the provinces are held."""
    forward = state.forward
    joint = linear_prior(state.scale, state.space)
    gain = LinearOperator.from_matrix(
        joint.domain.subspace(1),
        forward.data_space,
        forward.response(phi),
        form="components",
    )
    problem = LinearForwardProblem(
        RowLinearOperator([forward.thermal, gain]), error=state.error
    )
    mean = LinearGaussianInversion(problem, joint, solver=CGSolver(rtol=SOLVER_RTOL))(
        state.data
    ).expectation
    return mean[0], np.asarray(mean[1])


def informed_start(state: Setup, /) -> tuple:
    """The data's guess at ``(thermal, level set, contrasts)``: provinces where
    the thermal-only posterior mean is slowest, and the conditional mean of the
    rest given them."""
    scale, space = state.scale, state.space
    inversion = LinearGaussianInversion(
        state.thermal_problem, state.thermal, solver=CGSolver(rtol=SOLVER_RTOL)
    )
    slow = -space.grid_values(inversion(state.data).expectation[0])
    weights = area_weights(space).ravel()
    order = np.argsort(slow.ravel())[::-1]
    filled = np.cumsum(weights[order])
    level = slow.ravel()[order[np.searchsorted(filled, scale.start_share)]]
    phi = space.from_grid_values((slow - level) / np.std(slow))
    thermal, contrasts = conditional_mean(state, phi)
    return thermal, phi, contrasts


def start_measure(
    prior: GaussianMeasure, start: Any, spread: float, /
) -> GaussianMeasure:
    """Where each chain starts: the guess, plus a little of a prior draw."""
    return prior.affine_map(
        spread * LinearOperator.identity(prior.domain), translation=start
    )


class Blocks(NamedTuple):
    """A Gibbs move with the problem and prior on the sum its blocks cut."""

    problem: ForwardProblem
    prior: GaussianMeasure
    move: Gibbs


def two_blocks(
    state: Setup, /, *, level_steps: int, order: str = "systematic"
) -> Blocks:
    """Gibbs on the grouped sum: HMC on the linear part, pCN on the level set."""
    scale = state.scale
    move = Gibbs(
        {
            "linear": HMC(step=scale.linear_step, length=scale.linear_length),
            "level_set": PCN(step=scale.level_step),
        },
        order=order,
        repeat={"level_set": level_steps},
    )
    return Blocks(state.grouped_problem, state.grouped, move)


def three_blocks(state: Setup, /) -> Blocks:
    """Gibbs on the fields as the model names them: the thermal triple, the
    level set and the contrasts each its own block."""
    scale = state.scale
    move = Gibbs(
        {
            "thermal": HMC(step=scale.thermal_step, length=scale.thermal_length),
            "level_set": PCN(step=scale.level_step),
            "chemistry": MALA(step=scale.contrast_step),
        },
        repeat={"level_set": scale.level_steps, "chemistry": scale.contrast_steps},
    )
    return Blocks(state.problem, state.prior, move)


def run_chains(
    state: Setup,
    blocks: Blocks,
    /,
    *,
    start: Any,
    steps: int,
    burn_in: int,
    rng: np.random.Generator,
    thin: int | None = None,
) -> Any:
    """A Gibbs run recording :data:`~src.record.NAMES`, whichever sum it is on."""
    domain = blocks.problem.model_space
    read = ungrouped if domain == state.grouped.domain else tuple
    sampler = Chains(
        blocks.move,
        steps=steps,
        burn_in=burn_in,
        chains=state.scale.gibbs_chains,
        thin=thin,
        rng=rng,
        n_jobs=state.scale.n_jobs,
    )
    estimator = MonteCarloEstimator(
        blocks.problem,
        blocks.prior,
        sampler=sampler,
        record=record_operator(state.space, domain, read=read),
        record_names=list(NAMES),
    )
    return estimator(state.data, start=start)


def jump_rates(posterior: Any, evaluations: int, /) -> np.ndarray:
    """The mean squared move of each scalar of :data:`MOVES` per sweep, per
    thousand forward evaluations."""
    values = posterior.scalars()
    picks = [1 + NAMES.index(name) for name in MOVES]
    moves = np.diff(values[picks], axis=2)
    return 1000.0 * np.mean(moves**2, axis=(1, 2)) / evaluations


def compare(
    state: Setup,
    label: str,
    blocks: Blocks,
    evaluations: int,
    start: Any,
    /,
    *,
    rng: np.random.Generator,
) -> dict:
    """A short run from the guess or the prior, and its printed row."""
    scale = state.scale
    watch = Stopwatch()
    posterior = run_chains(
        state,
        blocks,
        start=start,
        steps=scale.compare_steps,
        burn_in=scale.compare_burn_in,
        rng=rng,
    )
    values = posterior.scalars()
    last = values[:, :, -len(values[0][0]) // 2 :].mean(axis=(1, 2))
    row = {
        "posterior": posterior,
        "rates": jump_rates(posterior, evaluations),
        "acceptance": posterior.block_acceptance_rates,
        "steps": np.mean([chain.step_sizes for chain in posterior.chains], axis=0),
        "potential": float(last[0]),
        "c_s": float(last[1 + NAMES.index("c_s")]),
        "evaluations": evaluations,
    }
    print(
        f"  {label:<30s} {evaluations:5d} "
        + " ".join(f"{value:9.2e}" for value in row["rates"])
        + f" {row['potential']:8.0f} {row['c_s']:7.2f}  "
        + "/".join(f"{value:.2f}" for value in row["acceptance"])
        + f" {watch.lap():6.0f}"
    )
    return row


def gibbs_answer(state: Setup, /, *, outdir: pathlib.Path) -> dict:
    """The main run, what it costs to run otherwise, and the figure."""
    scale, space = state.scale, state.space
    rng = part_rng(scale.seed, "d")
    potential = state.problem.potential(state.data)
    watch = Stopwatch()
    start = informed_start(state)
    guess = summary(space, start)
    print(
        f"the data's guess in {watch.lap():.1f} s: {guess[NAMES.index('share')]:.0%} of "
        f"the sphere inside {guess[NAMES.index('provinces')]:.0f} provinces, contrasts "
        f"{np.round(start[2], 2)} percent; potential {potential(start):.0f} against "
        f"the truth's {potential(state.truth):.0f}"
    )
    measure = start_measure(state.grouped, grouped(start), scale.start_spread)
    main = run_chains(
        state,
        two_blocks(state, level_steps=scale.level_steps),
        start=measure,
        steps=scale.gibbs_steps,
        burn_in=scale.gibbs_burn_in,
        thin=scale.gibbs_thin,
        rng=rng.spawn(1)[0],
    )
    seconds = watch.lap()
    diagnostics = main.diagnostics()
    steps = np.mean([chain.step_sizes for chain in main.chains], axis=0)
    print(
        f"the main run: {scale.gibbs_chains} chains of {scale.gibbs_burn_in} + "
        f"{scale.gibbs_steps} sweeps of HMC x {scale.linear_length} on the linear part "
        f"and pCN x {scale.level_steps} on the level set, {seconds:.0f} s; steps "
        f"{steps[0]:.3f} and {steps[1]:.5f}, acceptance "
        f"{main.block_acceptance_rates[0]:.2f} and "
        f"{main.block_acceptance_rates[1]:.2f}; mean potential "
        f"{main.scalars()[0].mean():.0f}"
    )
    print(f"  {'scalar':<10s} {'tau':>7s} {'ESS':>7s} {'R-hat':>6s}")
    for name in SHOWN:
        k = 1 + NAMES.index(name)
        print(
            f"  {name:<10s} {diagnostics.autocorrelation_time[k]:7.1f} "
            f"{diagnostics.effective_sample_size[k]:7.1f} {diagnostics.rhat[k]:6.2f}"
        )
    print(
        "  what each run moves in its squared jumps per sweep, per 1000 forward "
        "evaluations; its potential and c_s over the second half of "
        f"{scale.compare_steps} sweeps; each block's acceptance"
    )
    print(
        f"  {'run':<30s} {'evals':>5s} "
        + " ".join(f"{name:>9s}" for name in MOVES)
        + f" {'Phi':>8s} {'c_s':>7s}  acceptance {'s':>6s}"
    )
    linear_evaluations = scale.linear_length
    rows = {}
    for level_steps in scale.level_ladder:
        rows[f"{level_steps} pCN"] = compare(
            state,
            f"{level_steps} pCN steps per visit",
            two_blocks(state, level_steps=level_steps),
            linear_evaluations + level_steps,
            measure,
            rng=rng.spawn(1)[0],
        )
    rows["symmetric"] = compare(
        state,
        f"symmetric scan, {scale.level_steps} pCN",
        two_blocks(state, level_steps=scale.level_steps, order="symmetric"),
        # There and back, linear then level set then linear: the level set,
        # in the middle, is visited once.
        2 * linear_evaluations + scale.level_steps,
        measure,
        rng=rng.spawn(1)[0],
    )
    rows["three"] = compare(
        state,
        "three blocks",
        three_blocks(state),
        scale.thermal_length + scale.level_steps + scale.contrast_steps,
        start_measure(state.prior, start, scale.start_spread),
        rng=rng.spawn(1)[0],
    )
    rows["prior"] = compare(
        state,
        "two blocks, from the prior",
        two_blocks(state, level_steps=scale.level_steps),
        linear_evaluations + scale.level_steps,
        state.grouped,
        rng=rng.spawn(1)[0],
    )
    draw(state, main, rows, outdir=outdir)
    return {
        "main": main,
        "start": start,
        "rows": rows,
        "diagnostics": diagnostics,
        "seconds": seconds,
    }


def draw(state: Setup, main: Any, rows: dict, /, *, outdir: pathlib.Path) -> None:
    """Each block's scalars along the chains, how the blocks and starts compare,
    and what more steps on the level set buy."""
    truth = summary(state.space, state.truth)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), layout="constrained")
    values = main.scalars()
    for ax, name, title in zip(
        axes[0],
        ("c_s", "c_rho", "share"),
        (
            "the shear contrast, in the linear block",
            "the density contrast, in the linear block",
            "the provinces' share, from the level set's block",
        ),
    ):
        k = NAMES.index(name)
        for chain, color in zip(values[1 + k], CHAIN_COLORS):
            ax.plot(chain, lw=0.8, color=color, alpha=0.9)
        ax.axhline(truth[k], color="k", lw=1.2, label="truth")
        ax.set_xlabel("sweep, after burn-in")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    for label, row, color in (
        ("two blocks", rows[f"{state.scale.level_steps} pCN"], "#2a78d6"),
        ("three blocks", rows["three"], "#eb6834"),
    ):
        k = 1 + NAMES.index("c_s")
        scalars = row["posterior"].scalars()
        for chain in scalars[k]:
            ax.plot(chain, lw=0.9, color=color, alpha=0.8)
        ax.plot([], [], color=color, label=label)
    ax.axhline(truth[NAMES.index("c_s")], color="k", lw=1.2, label="truth")
    ax.set_xlabel("sweep, after a short burn-in")
    ax.set_ylabel("c_s, percent")
    ax.set_title(
        "the contrast walks its valley only with the thermal fields", fontsize=10
    )
    ax.margins(y=0.15)
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncols=3)
    ax = axes[1, 1]
    for label, row, color in (
        ("from the data's guess", rows[f"{state.scale.level_steps} pCN"], "#2a78d6"),
        ("from the prior", rows["prior"], "#eb6834"),
    ):
        for chain in row["posterior"].scalars()[0]:
            ax.plot(chain, lw=0.9, color=color, alpha=0.8)
        ax.plot([], [], color=color, label=label)
    ax.axhline(
        state.problem.potential(state.data)(state.truth),
        color="k",
        lw=1.2,
        label="the truth's",
    )
    ax.set_yscale("log")
    ax.set_xlabel("sweep, after a short burn-in")
    ax.set_ylabel("likelihood potential")
    ax.set_title("where the chains settle", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1, 2]
    ladder = state.scale.level_ladder
    k = MOVES.index("share")
    ax.plot(
        ladder,
        [rows[f"{n} pCN"]["rates"][k] for n in ladder],
        "o-",
        lw=2.0,
        color="#2a78d6",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ladder, [str(n) for n in ladder])
    ax.set_xlabel("pCN steps per visit to the level set")
    ax.set_ylabel("squared move of the share, per 1000 evaluations")
    ax.set_title("what more steps on the level set buy", fontsize=10)
    for ax in axes.ravel():
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    save(
        fig,
        outdir / "gibbs.png",
        title="LLSVP, part D: Gibbs over the fields, the coupled thermal fields and "
        "the contrasts against the level set",
    )


def run(scale: Scale, /, *, outdir: pathlib.Path, state: Setup | None = None) -> dict:
    started = time.perf_counter()
    print(f"\n=== D. Gibbs over the fields ({scale.name} scale) ===")
    if state is None:
        state = setup(scale)
    result = gibbs_answer(state, outdir=outdir)
    print(f"D took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
