"""Part D: a pendulum that was pushed, and what that does to the answer.

The readings now come from a pendulum under a forcing nobody recorded: an
angular acceleration of each arm, a smooth random function of time. Two
answers for the masses and lengths are compared.

* **The forcing ignored.** The model of parts B and C, fitted to these
  readings. It cannot fit them, and it reports the parameters to a
  precision it has not earned.
* **The forcing unknown.** The model is the pair of the parameters and the
  forcing, a function of time with a Gaussian prior of its own; both are
  inferred together, and the answer for the parameters is the marginal of
  the joint posterior. It is found by the Laplace approximation, a Newton
  search over more than a thousand unknowns whose gradient is one adjoint
  integration, and then by chains about that Gaussian.

The forcing itself is recovered where the readings constrain it, and its
posterior is the prior where they do not.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    ForwardProblem,
    GaussianMeasure,
    LaplaceApproximation,
    MonteCarloEstimator,
)
from pygeoinf.sampling import PCN, Chains

from src.common import log_normal_interval, parameter_panels, parse, part_rng, save
from src.observation import first_of
from src.problem import FULL, QUICK, Problem, Scale, build
from src.spaces import to_table

DRAWS = 20
# The sampler's step about the joint Laplace Gaussian.
STEP = 0.5


def marginal(measure: GaussianMeasure, problem: Problem, /) -> np.ndarray:
    """The covariance of the parameters' coordinates under a measure on the pair."""
    covariance, size = measure.covariance, problem.parameter_map.space.dim
    columns = [
        covariance((np.eye(size)[k], problem.forcing_space.zero()))[0]
        for k in range(size)
    ]
    return np.array(columns).T


def summary(
    label: str, mean: np.ndarray, covariance: np.ndarray, truth: np.ndarray
) -> dict:
    """A posterior for the coordinates, printed against the truth."""
    std = np.sqrt(np.diag(covariance))
    away = np.abs(mean - truth) / std
    print(
        f"  {label:18s}"
        + "".join(f"{100.0 * value:8.3f}%" for value in std)
        + "   truth at "
        + ", ".join(f"{value:.1f}" for value in away)
        + " sd"
    )
    return {"mean": mean, "std": std, "covariance": covariance, "away": away}


def forcing(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The two answers for a forced pendulum, and their figures."""
    started = time.perf_counter()
    scale, names = problem.scale, problem.parameter_names
    truth, prior = problem.truth, problem.parameter_prior
    forced = problem.forced_forward
    space = forced.codomain
    print(f"\n=== D. A pendulum that was pushed ({scale.name} scale) ===")

    # The readings: the true parameters, a forcing drawn from its prior, noise.
    pushed = problem.forcing_prior.sample(rng=rng)
    clean = forced((truth, pushed))
    data = space.add(clean, problem.noise.sample(rng=rng))
    shift = space.norm(space.subtract(clean, problem.clean))
    noise_size = scale.noise_std * np.sqrt(space.dim)
    print(
        f"a forcing of standard deviation {scale.forcing_std:g} rad/s^2 and "
        f"correlation time {scale.forcing_time:g} s moves the readings by {shift:.2f}, "
        f"against {noise_size:.2f} of noise"
    )
    print("the standard deviation of each logarithm:")
    print(f"  {'':18s}" + "".join(f"{name:>9s}" for name in names))
    true_parameters = np.array(scale.true_parameters)
    prior_std = np.full(4, scale.prior_std)

    # ---- the forcing ignored ---- #
    t0 = time.perf_counter()
    plain = ForwardProblem(problem.forward, error=problem.noise)
    ignored = LaplaceApproximation.from_problem(plain, prior, gauss_newton=True).solve(
        data
    )
    ignored_found = summary(
        "forcing ignored",
        np.asarray(ignored.model),
        ignored.measure.covariance.matrix(form="components"),
        truth,
    )
    ignored_found["chi_squared"] = 2.0 * float(plain.potential(data)(ignored.model))
    ignored_found["seconds"] = time.perf_counter() - t0

    # ---- the forcing unknown: the joint Laplace approximation ---- #
    t0 = time.perf_counter()
    before = dict(problem.forced_solver.solves)
    joint_prior = GaussianMeasure.from_product(
        [prior, problem.forcing_prior], domain=forced.domain
    )
    joint = ForwardProblem(forced, error=problem.noise)
    # The curvature of the misfit has no more directions than there are
    # readings, so the Gaussian is the prior with that many corrected, and
    # nothing has to be solved for its covariance.
    laplace = LaplaceApproximation.from_problem(
        joint, joint_prior, gauss_newton=True, rank=space.dim, rng=rng
    )
    result = laplace.solve(data)
    mode_parameters, mode_forcing = result.model
    unknown_found = summary(
        "forcing unknown",
        np.asarray(mode_parameters),
        marginal(result.measure, problem),
        truth,
    )
    unknown_found["chi_squared"] = 2.0 * float(joint.potential(data)(result.model))
    unknown_found["seconds"] = time.perf_counter() - t0
    unknown_found["iterations"] = result.optimization.iterations
    used = {
        key: value - before[key]
        for key, value in problem.forced_solver.solves.items()
        if value > before[key]
    }
    print(
        f"  {'prior':18s}" + "".join(f"{100.0 * scale.prior_std:8.3f}%" for _ in names)
    )
    print(
        f"  chi-squared for {space.dim} readings: {ignored_found['chi_squared']:.0f} with "
        f"the forcing ignored, {unknown_found['chi_squared']:.0f} with it unknown"
    )
    print(
        f"  the joint search over {forced.domain.dim} unknowns: "
        f"{unknown_found['iterations']} iterations, {unknown_found['seconds']:.1f} s, "
        f"integrations {used}"
    )

    # The forcing: the truth, the mode, and draws of the Laplace Gaussian.
    read = problem.forcing_space.lift(to_table(problem.space), axis=1)
    draws = [result.measure.sample(rng=rng) for _ in range(DRAWS)]
    tables = np.array([read(draw[1]) for draw in draws])
    true_table, mode_table = read(pushed), read(mode_forcing)
    spread = tables.std(axis=0)
    error = float(np.sqrt(np.mean((mode_table - true_table) ** 2)))
    size = float(np.sqrt(np.mean(true_table**2)))
    covered = float(np.mean(np.abs(mode_table - true_table) <= 2.0 * spread))
    print(
        f"  the forcing: root mean square {size:.3f}, the mode's error {error:.3f}; "
        f"the truth within two posterior sd of the mode at {100.0 * covered:.0f}% of "
        f"the nodes; posterior sd between {spread.min():.3f} and {spread.max():.3f}"
    )

    figure, axes = plt.subplots(1, 4, figsize=(11.0, 3.4), layout="constrained")
    parameter_panels(
        axes,
        names,
        true_parameters,
        [
            log_normal_interval("prior, 2 sd", prior.expectation, prior_std, "0.6"),
            log_normal_interval(
                "posterior, 2 sd", ignored_found["mean"], ignored_found["std"], "C3"
            ),
        ],
    )
    for ax, k in zip(axes[2:], (2, 3)):
        center = np.exp(ignored_found["mean"][k])
        reach = max(abs(center - true_parameters[k]) * 1.6, 0.02)
        ax.set_ylim(true_parameters[k] - reach, true_parameters[k] + reach)
    save(
        figure,
        outdir / "D_forcing_ignored.png",
        title="The forcing ignored: prior against posterior (the lengths' panels are drawn close in)",
    )

    figure, axes = plt.subplots(
        1,
        6,
        figsize=(16.0, 3.4),
        layout="constrained",
        width_ratios=[1, 1, 1, 1, 2.2, 2.2],
    )
    parameter_panels(
        axes[:4],
        names,
        true_parameters,
        [
            log_normal_interval("prior, 2 sd", prior.expectation, prior_std, "0.6"),
            log_normal_interval(
                "posterior, 2 sd", unknown_found["mean"], unknown_found["std"], "C0"
            ),
        ],
    )
    nodes = problem.grid.nodes
    for ax, k in zip(axes[4:], range(2)):
        for table in tables:
            ax.plot(nodes, table[k], color="C0", lw=0.5, alpha=0.4)
        ax.plot(nodes, mode_table[k], color="C0", lw=1.4, label="mode")
        ax.plot(nodes, true_table[k], color="black", ls="--", lw=1.0, label="truth")
        ax.set_xlabel("time, s")
        ax.set_title(
            f"forcing of arm {k + 1}, rad/s$^2$: mode and {DRAWS} draws", fontsize=10
        )
        ax.legend(fontsize=8, frameon=False)
    save(
        figure,
        outdir / "D_forcing_unknown.png",
        title="The forcing unknown, by the Laplace approximation: prior against posterior, and the forcing",
    )

    # ---- the forcing unknown: chains about the Laplace Gaussian ---- #
    t0 = time.perf_counter()
    before = problem.forced_solver.solves["state"]
    sampler = Chains(
        PCN(step=STEP, acceptance=None),
        steps=scale.steps,
        burn_in=scale.burn_in,
        chains=scale.chains,
        rng=rng,
    )
    estimator = MonteCarloEstimator(
        joint,
        joint_prior,
        sampler=sampler,
        reference=laplace.reference(data),
        record=first_of(forced.domain),
        record_names=[f"log {name}" for name in names],
    )
    posterior = estimator(data)
    values = posterior.scalars()[1:].reshape(len(names), -1)
    diagnostics = posterior.diagnostics()
    sampled_found = summary("sampled", values.mean(axis=1), np.cov(values), truth)
    sampled_found["rhat"] = diagnostics.rhat[1:]
    sampled_found["effective"] = diagnostics.effective_sample_size[1:]
    sampled_found["acceptance"] = float(posterior.acceptance_rate)
    print(
        f"  sampling: {scale.chains} chains of {scale.steps} steps after "
        f"{scale.burn_in}, acceptance {posterior.acceptance_rate:.2f}, R-hat up to "
        f"{sampled_found['rhat'].max():.3f}, effective size from "
        f"{sampled_found['effective'].min():.0f}, "
        f"{problem.forced_solver.solves['state'] - before} integrations, "
        f"{time.perf_counter() - t0:.1f} s"
    )
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 3.4), layout="constrained")
    parameter_panels(
        axes,
        names,
        true_parameters,
        [
            log_normal_interval("prior, 2 sd", prior.expectation, prior_std, "0.6"),
            log_normal_interval(
                "posterior, 2 sd", sampled_found["mean"], sampled_found["std"], "C0"
            ),
        ],
    )
    save(
        figure,
        outdir / "D_forcing_sampled.png",
        title="The forcing unknown, by sampling: prior against posterior",
    )
    print(f"part D took {time.perf_counter() - started:.1f} s")
    return {
        "shift": float(shift),
        "noise": float(noise_size),
        "ignored": ignored_found,
        "unknown": unknown_found,
        "sampled": sampled_found,
        "forcing": {"size": size, "error": error, "covered": covered},
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    """Run the part on a scale, building the problem unless it is given."""
    problem = build(scale) if problem is None else problem
    return forcing(problem, rng=part_rng(scale.seed, "D"), outdir=outdir)


def main(argv: list[str] | None = None) -> dict:
    """Run the part from the command line."""
    arguments = parse(__doc__, quick=QUICK, full=FULL, argv=argv)
    return run(arguments.scale, outdir=arguments.outdir)


if __name__ == "__main__":
    main()
