"""Part C: the thermal-only problem, whose posterior is known exactly.

Leave the chemistry out and the forward map is ``T``, linear, under a Gaussian
prior and Gaussian noise: the posterior is Gaussian, and its means and
variances on the regional means are a few conjugate-gradient solves. So every
sampler can be marked against the exact answer, before the hard problem of
part D, where nothing can.

Two problems, the thermal fields under the same data. **As given**, the data
inform about a hundred directions, the leading one thousands of times over;
every sampler whose reference is the prior moves at the pace of that direction,
and crawls. **Weak**, the same noise draw made many times larger, the leading
direction is informed some tens of times over and those samplers mix, so that
the check against the exact answer has power. On both, the chains start from
exact posterior draws, so that what is measured is the mixing at equilibrium
and no chain has anywhere to travel to.

Then a ladder of truncation degrees, on the weak problem and on one data set:
the acceptance at a fixed step and the step the burn-in settles on, against
a plain random walk with pCN's proposal size. Every sampler of the library is
defined on the function space, and so does not notice the refinement.

Each prior-reference sampler gets the same forward evaluations per chain.
``GeometricHMC`` gets its own count of steps: its reference pairs every state
with every chart vector by the space's inner product, whose cost on the sphere
is two spherical-harmonic transforms a field, and that, not the forward map, is
what it spends its time on.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    GaussianMeasure,
    GeneralizedHMC,
    GeometricHMC,
    HMC,
    LinearForwardProblem,
    LinearGaussianInversion,
    LowRankGaussianPosterior,
    MALA,
    PCN,
)
from pygeoinf.inference.sampling import pcn_beta
from pygeoinf.numerics.chain_diagnostics import integrated_autocorrelation_time
from pygeoinf.numerics.solvers import CGSolver

from src.common import Stopwatch, parse, part_rng, save
from src.forward import Forward, record_operator
from src.prior import sphere, thermal_prior
from src.shell import FIELDS, FULL, QUICK, REGIONS, Scale
from synthetic import Setup, setup

# The samplers, each with the slot of the palette it keeps through the figure.
COLORS = {
    "PCN": "#2a78d6",
    "MALA": "#eb6834",
    "HMC": "#1baf7a",
    "GeneralizedHMC": "#eda100",
    "GeometricHMC": "#e87ba4",
    "random walk": "#6b6b66",
}
# The ladder's trajectories are short: it measures an acceptance, not mixing.
LADDER_LENGTH = 3
# R-hat above this, and a chain's own error bar on its mean means nothing.
MIXED = 1.1
SOLVER_RTOL = 1e-9


@dataclass
class Posed:
    """One linear problem on one data set, with its exact answer.

    Attributes:
        name: what the printed tables call it.
        problem: the linear forward problem.
        data: the data.
        inversion: the exact estimator.
        posterior: the exact posterior.
        mean: its means of the recorded scalars.
        std: its standard deviations of them.
        prior_std: the prior's standard deviations of them.
        eigenvalues: the leading eigenvalues of the misfit's curvature through
            the prior's factor.
    """

    name: str
    problem: LinearForwardProblem
    data: np.ndarray
    inversion: LinearGaussianInversion
    posterior: GaussianMeasure
    mean: np.ndarray
    std: np.ndarray
    prior_std: np.ndarray
    eigenvalues: np.ndarray


def names() -> list[str]:
    """The recorded scalars: every field over every region."""
    return [f"{field}, {region}" for field in FIELDS for region, _ in REGIONS]


def pose(
    state: Setup, name: str, factor: float, /, *, rng: np.random.Generator
) -> Posed:
    """The thermal-only problem with its noise made ``factor`` times larger.

    The same draw of the noise, scaled, so that the two problems differ in how
    much the data say and in nothing else.
    """
    scale = state.scale
    data = state.clean + factor * (state.data - state.clean)
    error = GaussianMeasure.from_standard_deviations(
        state.forward.data_space, factor * state.noise
    )
    problem = LinearForwardProblem(state.forward.thermal, error=error)
    inversion = LinearGaussianInversion(
        problem, state.thermal, solver=CGSolver(rtol=SOLVER_RTOL)
    )
    posterior = inversion(data)
    record = record_operator(scale, state.space)
    pushed = posterior.affine_map(record)
    covariance = pushed.covariance.matrix(form="components")
    prior_pushed = state.thermal.affine_map(record).covariance.matrix(form="components")
    eigenvalues = LowRankGaussianPosterior.from_linear_problem(
        problem, state.thermal, data, rank=2 * scale.reference_rank, rng=rng
    ).eigenvalues
    return Posed(
        name,
        problem,
        data,
        inversion,
        posterior,
        record.codomain.to_components(pushed.expectation),
        np.sqrt(np.diag(covariance)),
        np.sqrt(np.diag(prior_pushed)),
        np.sort(eigenvalues)[::-1],
    )


def samplers(scale: Scale, posed: Posed, reference: Any, /) -> dict:
    """Every sampler on a problem, with the forward evaluations of its step."""
    problem, prior = posed.problem, posed.inversion.prior
    return {
        "PCN": (PCN(problem, prior, step=scale.pcn_step), 1),
        "MALA": (MALA(problem, prior, step=scale.mala_step), 1),
        "HMC": (
            HMC(problem, prior, step=scale.hmc_step, length=scale.hmc_length),
            scale.hmc_length,
        ),
        "GeneralizedHMC": (
            GeneralizedHMC(problem, prior, step=scale.hmc_step),
            1,
        ),
        "GeometricHMC": (
            GeometricHMC(
                problem, reference, step=scale.ghmc_step, length=scale.ghmc_length
            ),
            scale.ghmc_length,
        ),
    }


def run_sampler(
    state: Setup,
    posed: Posed,
    name: str,
    sampler: Any,
    per_step: int,
    /,
    *,
    rng: np.random.Generator,
) -> dict:
    """One sampler's chains from exact draws, marked against the exact answer."""
    scale = state.scale
    if name == "GeometricHMC":
        steps, burn_in = scale.geometric_steps, scale.geometric_burn_in
    else:
        total = scale.budget // per_step
        burn_in = int(scale.burn_share * total)
        steps = total - burn_in
    watch = Stopwatch()
    posterior = sampler(
        posed.data,
        steps=steps,
        chains=scale.chains,
        burn_in=burn_in,
        start=posed.posterior,
        record=record_operator(scale, state.space),
        record_names=names(),
        rng=rng,
        n_jobs=scale.n_jobs,
    )
    seconds = watch.lap()
    diagnostics = posterior.diagnostics()
    values = posterior.scalars()[1:]
    means = values.mean(axis=(1, 2))
    spreads = values.reshape(values.shape[0], -1).std(axis=1, ddof=1)
    errors = np.array([e.standard_error for e in posterior.scalar_means()[1:]])
    evaluations = scale.chains * (steps + burn_in) * per_step
    ess = float(diagnostics.effective_sample_size[1:].min())
    return {
        "steps": steps,
        "acceptance": posterior.acceptance_rate,
        "step": float(np.mean([chain.step_size for chain in posterior.chains])),
        "tau": float(diagnostics.autocorrelation_time[1:].max()),
        "tau_potential": float(diagnostics.autocorrelation_time[0]),
        "ess": ess,
        "rhat": float(diagnostics.rhat[1:].max()),
        "offsets": (means - posed.mean) / posed.std,
        "z": (means - posed.mean) / errors,
        "ratios": spreads / posed.std,
        "seconds": seconds,
        "evaluations": evaluations,
        "ess_rate": 1000.0 * ess / evaluations,
    }


def run_problem(
    state: Setup, posed: Posed, rank: int, /, *, rng: np.random.Generator
) -> dict:
    """Every sampler on one problem, and its printed table."""
    scale = state.scale
    watch = Stopwatch()
    reference = posed.inversion.reference(posed.data, rank=rank, rng=rng)
    build = watch.lap()
    eigenvalues = posed.eigenvalues
    picks = [k for k in (0, 9, 29, 49, 79) if k < eigenvalues.size]
    print(
        f"{posed.name}: the curvature through the prior's factor has eigenvalues "
        + ", ".join(f"{eigenvalues[k]:.3g} (no. {k + 1})" for k in picks)
    )
    print(
        f"  the regional means' posterior sd over the prior's: "
        f"{np.min(posed.std / posed.prior_std):.2f} to "
        f"{np.max(posed.std / posed.prior_std):.2f}; the reference of rank {rank} "
        f"in {build:.1f} s"
    )
    print(
        f"  {'sampler':<18s} {'evals':>6s} {'step':>9s} {'accept':>6s} {'tau':>8s} "
        f"{'tau Phi':>8s} "
        f"{'ESS':>7s} {'R-hat':>6s} {'ESS/1000':>8s} {'|offset|':>8s} {'|z|':>5s} "
        f"{'sd ratio':>11s} {'s':>6s}"
    )
    rows = {}
    for (name, (sampler, per_step)), stream in zip(
        samplers(scale, posed, reference).items(), rng.spawn(5)
    ):
        row = run_sampler(state, posed, name, sampler, per_step, rng=stream)
        rows[name] = row
        print(
            f"  {name:<18s} {row['evaluations']:6d} {row['step']:9.2e} "
            f"{row['acceptance']:6.2f} {row['tau']:8.1f} {row['tau_potential']:8.1f} "
            f"{row['ess']:7.1f} "
            f"{row['rhat']:6.2f} {row['ess_rate']:8.2f} "
            f"{np.abs(row['offsets']).max():8.2f} {np.abs(row['z']).max():5.1f} "
            f"{row['ratios'].min():5.2f}-{row['ratios'].max():5.2f} "
            f"{row['seconds']:6.1f}"
        )
    print(
        "  tau the worst regional mean's autocorrelation time, tau Phi the "
        "likelihood potential's, which sees a slow direction the means can miss"
    )
    print(
        f"  |offset| the worst chain mean's distance from the exact mean, in exact "
        f"sd; |z| in its own Monte Carlo errors, meaningless above R-hat {MIXED}"
    )
    return {"rows": rows, "reference_seconds": build, "rank": rank}


def random_walk(
    posed_problem: LinearForwardProblem,
    prior: GaussianMeasure,
    data: np.ndarray,
    start: Any,
    step: float,
    steps: int,
    /,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random-walk Metropolis in the fields themselves: the acceptance
    probabilities of a proposal of the given size.

    A proposal ``m + s xi`` with ``xi`` a prior draw is pCN's without the
    contraction towards the prior's mean. Its ratio needs the prior's density
    against the grid, ``|m|^2_{C^-1}``, which exists at every degree and has no
    limit; the library refuses to build this sampler.
    """
    space = prior.domain
    potential = posed_problem.potential(data)

    def log_density(m: Any) -> float:
        return -potential(m) - 0.5 * space.inner_product(prior.precision(m), m)

    current, value = start, log_density(start)
    probabilities = np.empty(steps)
    for k in range(steps):
        proposal = space.axpy(step, prior.sample(rng=rng), space.copy(current))
        proposed = log_density(proposal)
        ratio = proposed - value
        probabilities[k] = np.exp(min(0.0, ratio))
        if np.log(rng.random()) < ratio:
            current, value = proposal, proposed
    return probabilities


def mean_and_error(probabilities: np.ndarray, /) -> tuple[float, float]:
    """The mean acceptance probability and its Monte Carlo standard error."""
    tau = integrated_autocorrelation_time(probabilities)
    return float(probabilities.mean()), float(
        probabilities.std() * np.sqrt(tau / probabilities.size)
    )


def ladder(
    state: Setup, weak: Posed, steps: dict, /, *, rng: np.random.Generator
) -> dict:
    """The acceptance at a fixed step and the adapted step, at every degree."""
    scale = state.scale
    factor = scale.weak_noise
    error = GaussianMeasure.from_standard_deviations(
        state.forward.data_space, factor * state.noise
    )
    walk_step = pcn_beta(steps["PCN"])
    results: dict[str, dict] = {
        name: {"acceptance": [], "error": [], "step": []}
        for name in ("PCN", "MALA", "HMC", "random walk")
    }
    n = scale.ladder_steps
    for degree in scale.ladder:
        watch = Stopwatch()
        space = sphere(scale, degree)
        forward = Forward(space, state.network, scale, n_jobs=scale.n_jobs)
        prior = thermal_prior(scale, space)
        problem = LinearForwardProblem(forward.thermal, error=error)
        start = LinearGaussianInversion(
            problem, prior, solver=CGSolver(rtol=SOLVER_RTOL)
        )(weak.data).expectation
        fixed = {
            "PCN": PCN(problem, prior, step=steps["PCN"], acceptance=None),
            "MALA": MALA(problem, prior, step=steps["MALA"], acceptance=None),
            "HMC": HMC(
                problem,
                prior,
                step=steps["HMC"],
                length=LADDER_LENGTH,
                acceptance=None,
            ),
        }
        adapted = {
            "PCN": PCN(problem, prior, step=steps["PCN"]),
            "MALA": MALA(problem, prior, step=steps["MALA"]),
            "HMC": HMC(problem, prior, step=steps["HMC"], length=LADDER_LENGTH),
        }
        row = f"  {degree:6d} {prior.domain.dim:6d}"
        for name in fixed:
            posterior = fixed[name](
                weak.data,
                steps=n,
                chains=2,
                burn_in=n // 4,
                start=start,
                rng=rng.spawn(1)[0],
                n_jobs=scale.n_jobs,
            )
            value, spread = mean_and_error(
                np.stack([chain.acceptance for chain in posterior.chains])
            )
            settled = adapted[name](
                weak.data,
                steps=n // 4,
                chains=2,
                burn_in=n,
                start=start,
                rng=rng.spawn(1)[0],
                n_jobs=scale.n_jobs,
            )
            step = float(np.mean([chain.step_size for chain in settled.chains]))
            results[name]["acceptance"].append(value)
            results[name]["error"].append(spread)
            results[name]["step"].append(step)
            row += f" {value:6.3f}+-{spread:.3f} {step:8.2e}"
        value, spread = mean_and_error(
            random_walk(
                problem, prior, weak.data, start, walk_step, 2 * n, rng=rng.spawn(1)[0]
            )
        )
        results["random walk"]["acceptance"].append(value)
        results["random walk"]["error"].append(spread)
        row += f" {value:9.3f}+-{spread:.3f}"
        print(row + f" {watch.lap():6.1f}")
    return {"degrees": list(scale.ladder), "walk_step": walk_step, **results}


def linear_answer(state: Setup, /, *, outdir: pathlib.Path) -> dict:
    """The two problems, the ladder and the figure."""
    scale = state.scale
    rng = part_rng(scale.seed, "c")
    given = pose(state, "the data as given", 1.0, rng=rng.spawn(1)[0])
    weak = pose(
        state,
        f"the data with {scale.weak_noise:g} times the noise",
        scale.weak_noise,
        rng=rng.spawn(1)[0],
    )
    chi_squared = given.problem.chi_squared(given.posterior.expectation, given.data)
    excess = (chi_squared - given.data.size) / np.sqrt(2.0 * given.data.size)
    print(
        f"the exact posterior mean's chi-squared on the data as given {chi_squared:.0f} "
        f"on {given.data.size} data: {excess:.1f} of its standard deviations too many, "
        "the misfit of thermal fields made to hold the chemistry"
    )
    weak_rows = run_problem(state, weak, scale.weak_rank, rng=rng.spawn(1)[0])
    given_rows = run_problem(state, given, scale.reference_rank, rng=rng.spawn(1)[0])
    steps = {name: weak_rows["rows"][name]["step"] for name in ("PCN", "MALA", "HMC")}
    walk_step = pcn_beta(steps["PCN"])
    print(
        f"the ladder, on the weak problem: at pCN's step {steps['PCN']:.2e}, "
        f"MALA's {steps['MALA']:.2e} and HMC's {steps['HMC']:.2e} x {LADDER_LENGTH}, "
        f"and a random walk of pCN's size {walk_step:.3f}; each acceptance at the "
        "fixed step, and each step as the burn-in leaves it"
    )
    print(
        f"  {'degree':>6s} {'dim':>6s}"
        + "".join(
            f" {name + ' accept':>13s} {'step':>8s}" for name in ("pCN", "MALA", "HMC")
        )
        + f" {'random walk':>15s} {'s':>6s}"
    )
    steps_ladder = ladder(state, weak, steps, rng=rng.spawn(1)[0])
    draw(weak_rows, given_rows, steps_ladder, outdir=outdir)
    return {
        "weak": weak,
        "given": given,
        "weak_rows": weak_rows,
        "given_rows": given_rows,
        "ladder": steps_ladder,
        "chi_squared": chi_squared,
        "excess": excess,
    }


def draw(weak: dict, given: dict, steps: dict, /, *, outdir: pathlib.Path) -> None:
    """What each sampler buys, its check against the exact answer, and the
    ladder."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    labels = list(weak["rows"])
    rows = np.arange(len(labels))[::-1]

    ax = axes[0, 0]
    for y, name in zip(rows, labels):
        a, b = weak["rows"][name]["ess_rate"], given["rows"][name]["ess_rate"]
        ax.plot([a, b], [y, y], color="0.8", lw=1.5, zorder=1)
        ax.plot(a, y, "o", color=COLORS[name], ms=9, zorder=2)
        ax.plot(b, y, "o", mfc="white", color=COLORS[name], ms=9, mew=1.8, zorder=2)
    ax.plot([], [], "o", color="0.35", ms=8, label="weak data")
    ax.plot(
        [], [], "o", mfc="white", color="0.35", ms=8, mew=1.8, label="data as given"
    )
    ax.set_xscale("log")
    ax.set_yticks(rows, labels)
    ax.set_xlabel("smallest ESS of a regional mean, per 1000 forward evaluations")
    ax.set_title("what a forward evaluation buys", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[0, 1]
    for y, name in zip(rows, labels):
        offsets = weak["rows"][name]["offsets"]
        ax.plot(
            offsets, np.full(offsets.size, y), "o", color=COLORS[name], ms=6, alpha=0.8
        )
    ax.axvline(0.0, color="0.45", lw=1.0)
    ax.set_yticks(rows, labels)
    ax.set_xlabel("each regional mean's distance from the exact one, in exact sd")
    ax.set_title("the check on the weak data, nine regional means each", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1, 0]
    degrees = steps["degrees"]
    for name in ("PCN", "MALA", "HMC", "random walk"):
        ax.errorbar(
            degrees,
            steps[name]["acceptance"],
            yerr=2.0 * np.array(steps[name]["error"]),
            marker="o",
            lw=2.0,
            capsize=3,
            color=COLORS[name],
            label=name,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(degrees, [str(d) for d in degrees])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("truncation degree")
    ax.set_ylabel("acceptance at a fixed step, 2 Monte Carlo errors")
    ax.set_title("the ladder: what refinement does to acceptance", fontsize=10)
    ax.legend(
        frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.5, 0.42), ncols=4
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1, 1]
    for name in ("PCN", "MALA", "HMC"):
        ax.plot(
            degrees, steps[name]["step"], "o-", lw=2.0, color=COLORS[name], label=name
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(degrees, [str(d) for d in degrees])
    ax.set_xlabel("truncation degree")
    ax.set_ylabel("step after burn-in")
    ax.set_title("the ladder: what refinement does to the step", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(
        fig,
        outdir / "samplers.png",
        title="LLSVP, part C: the thermal-only problem, sampled and marked against "
        "its exact posterior",
    )


def run(scale: Scale, /, *, outdir: pathlib.Path, state: Setup | None = None) -> dict:
    started = time.perf_counter()
    print(f"\n=== C. The thermal-only problem, exactly ({scale.name} scale) ===")
    if state is None:
        state = setup(scale)
    result = linear_answer(state, outdir=outdir)
    print(f"C took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
