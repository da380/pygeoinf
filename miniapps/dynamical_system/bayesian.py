"""Part C: a measure answer, the posterior of the masses and lengths.

The same prior and the same readings as the point answer, and now the whole
posterior, two ways.

* **The Laplace approximation**: the Gaussian about the mode whose
  precision is the curvature there, with the exact Hessian and with the
  Gauss-Newton one. It costs a search and a handful of integrations.
* **Sampling**: chains about the Laplace Gaussian, by default of
  preconditioned Crank-Nicolson steps, which need the misfit alone and no
  derivative; ``--sampler`` chooses another move, and ``--sampler-step``
  its step. Their target is the exact posterior, whatever the reference; a
  good reference makes them fast. They start from draws of that Gaussian: the posterior
  is a hundred times narrower than the prior in three directions of four,
  and a chain started from a draw of the prior does not find it.

Each is shown as the prior against the posterior for every parameter, and
through something the instrument never saw: the motion of a component of
the state that was not read, under draws from the posterior.

The masses enter the motion through their ratio alone, so along their
common size the posterior is the prior, whatever the readings; along their
ratio it is as narrow as the lengths'.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import ForwardProblem, LaplaceApproximation, LinearOperator

from src.common import log_normal_interval, parameter_panels, parse, part_rng, save
from src.problem import FULL, QUICK, Problem, Scale, build
from src.sampling import sample

# The two directions the masses' logarithms are read along.
TOGETHER = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
RATIO = np.array([-1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
DRAWS = 30
# The step of the "pcn" sampler where the scale gives none: after it,
# exp(-STEP) of the old point is left. Long, because the reference fits the
# posterior and a long step is accepted.
STEP = 2.0


def unseen_component(problem: Problem, /) -> int:
    """A component of the state the instrument does not read; the first rate if it reads them all."""
    scale = problem.scale
    seen = {0, 1} if scale.instrument == "bob" else set(scale.observed)
    hidden = [k for k in range(problem.system.state_dim) if k not in seen]
    return hidden[0] if hidden else 2


def motion_panel(ax, problem: Problem, models: list, /, *, color: str) -> None:
    """A component the instrument did not read, under several models, against the truth."""
    scale, index = problem.scale, unseen_component(problem)
    fine = np.linspace(0.0, scale.duration, 500)
    for model in models:
        state = problem.solver.solve(problem.parameter_map.parameters(model))
        ax.plot(fine, state(fine)[index], color=color, lw=0.5, alpha=0.4)
    state = problem.solver.solve(np.array(scale.true_parameters))
    ax.plot(fine, state(fine)[index], color="black", ls="--", lw=1.0, label="truth")
    ax.set_xlabel("time, s")
    ax.set_title(
        f"{problem.system.state_names[index]}, not read: {len(models)} posterior draws",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)


def figure_for(
    problem: Problem,
    mean: np.ndarray,
    std: np.ndarray,
    models: list,
    path,
    /,
    *,
    title: str,
) -> None:
    """The prior against the posterior for each parameter, and the unseen motion."""
    prior = problem.parameter_prior
    figure, axes = plt.subplots(
        1, 5, figsize=(14.0, 3.4), layout="constrained", width_ratios=[1, 1, 1, 1, 3]
    )
    prior_std = np.full(4, problem.scale.prior_std)
    parameter_panels(
        axes[:4],
        problem.parameter_names,
        np.array(problem.scale.true_parameters),
        [
            log_normal_interval("prior, 2 sd", prior.expectation, prior_std, "0.6"),
            log_normal_interval("posterior, 2 sd", mean, std, "C0"),
        ],
    )
    motion_panel(axes[4], problem, models, color="C0")
    save(figure, path, title=title)


def bayesian(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The Laplace approximation and the sampled posterior, with their figures."""
    started = time.perf_counter()
    scale = problem.scale
    names, prior = problem.parameter_names, problem.parameter_prior
    forward_problem = ForwardProblem(problem.forward, error=problem.noise)
    truth = problem.truth
    print(f"\n=== C. The posterior of the masses and lengths ({scale.name} scale) ===")

    # ---- the Laplace approximation, with each Hessian ---- #
    laplace = {}
    for label, gauss_newton in (("exact Hessian", False), ("Gauss-Newton", True)):
        t0 = time.perf_counter()
        estimator = LaplaceApproximation.from_problem(
            forward_problem, prior, gauss_newton=gauss_newton
        )
        result = estimator.solve(problem.data)
        covariance = result.measure.covariance.matrix(form="components")
        laplace[label] = {
            "estimator": estimator,
            "measure": result.measure,
            "mean": np.asarray(result.model),
            "covariance": covariance,
            "std": np.sqrt(np.diag(covariance)),
            "seconds": time.perf_counter() - t0,
        }
    exact, gauss = laplace["exact Hessian"], laplace["Gauss-Newton"]
    print("the Laplace approximation, the standard deviation of each logarithm:")
    print(f"  {'':16s}" + "".join(f"{name:>9s}" for name in names) + "   seconds")
    for label, found in laplace.items():
        row = "".join(f"{100.0 * value:8.3f}%" for value in found["std"])
        print(f"  {label:16s}{row}   {found['seconds']:6.1f}")
    print(
        f"  {'prior':16s}" + "".join(f"{100.0 * scale.prior_std:8.3f}%" for _ in names)
    )

    def along(covariance: np.ndarray, direction: np.ndarray) -> float:
        return float(np.sqrt(direction @ covariance @ direction))

    directions = {
        "the masses' common size": along(exact["covariance"], TOGETHER),
        "their ratio": along(exact["covariance"], RATIO),
    }
    for name, value in directions.items():
        print(
            f"  along {name}: {100.0 * value:.3f}% (prior {100.0 * scale.prior_std:.1f}%)"
        )
    inside = np.abs(exact["mean"] - truth) / exact["std"]
    print(
        "  the truth lies "
        + ", ".join(f"{value:.1f}" for value in inside)
        + " posterior standard deviations from the mode"
    )
    figure_for(
        problem,
        gauss["mean"],
        gauss["std"],
        [gauss["measure"].sample(rng=rng) for _ in range(DRAWS)],
        outdir / "C_laplace.png",
        title="The Laplace approximation: prior against posterior, and a motion that was not read",
    )

    # ---- sampling, about the Laplace Gaussian ---- #
    t0 = time.perf_counter()
    before = problem.solver.solves["state"]
    posterior, move = sample(
        forward_problem,
        prior,
        problem.data,
        scale,
        at_mode=gauss["estimator"],
        start=gauss["measure"],
        record=LinearOperator.identity(problem.parameter_map.space),
        record_names=[f"log {name}" for name in names],
        pcn_step=STEP,
        rng=rng,
    )
    seconds = time.perf_counter() - t0
    values = posterior.scalars()[1:]
    flat = values.reshape(len(names), -1)
    mean, covariance = flat.mean(axis=1), np.cov(flat)
    std = np.sqrt(np.diag(covariance))
    diagnostics = posterior.diagnostics()
    print(
        f"sampling ({move}): {scale.chains} chains of {scale.steps} steps after {scale.burn_in}, "
        f"acceptance {posterior.acceptance_rate:.2f}, "
        f"{problem.solver.solves['state'] - before} integrations, {seconds:.1f} s"
    )
    print(f"  {'':16s}" + "".join(f"{name:>9s}" for name in names))
    print(f"  {'sampled sd':16s}" + "".join(f"{100.0 * value:8.3f}%" for value in std))
    print(
        f"  {'mean - mode':16s}"
        + "".join(f"{100.0 * value:+8.3f}%" for value in mean - exact["mean"])
    )
    print(
        f"  {'R-hat':16s}" + "".join(f"{value:9.3f}" for value in diagnostics.rhat[1:])
    )
    print(
        f"  {'effective size':16s}"
        + "".join(f"{value:9.0f}" for value in diagnostics.effective_sample_size[1:])
    )
    sampled_directions = {
        "the masses' common size": along(covariance, TOGETHER),
        "their ratio": along(covariance, RATIO),
    }
    for name, value in sampled_directions.items():
        print(f"  along {name}: {100.0 * value:.3f}%")
    picks = rng.choice(flat.shape[1], size=DRAWS, replace=False)
    figure_for(
        problem,
        mean,
        std,
        [flat[:, pick] for pick in picks],
        outdir / "C_sampling.png",
        title="Sampling: prior against posterior, and a motion that was not read",
    )
    print(f"part C took {time.perf_counter() - started:.1f} s")
    return {
        "laplace": {
            label: {k: found[k] for k in ("mean", "std", "covariance", "seconds")}
            for label, found in laplace.items()
        },
        "directions": directions,
        "inside": inside,
        "sampled": {
            "mean": mean,
            "std": std,
            "covariance": covariance,
            "rhat": diagnostics.rhat[1:],
            "effective": diagnostics.effective_sample_size[1:],
            "acceptance": float(posterior.acceptance_rate),
            "directions": sampled_directions,
        },
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    """Run the part on a scale, building the problem unless it is given."""
    problem = build(scale) if problem is None else problem
    return bayesian(problem, rng=part_rng(scale.seed, "C"), outdir=outdir)


def main(argv: list[str] | None = None) -> dict:
    """Run the part from the command line."""
    arguments = parse(__doc__, quick=QUICK, full=FULL, argv=argv)
    return run(arguments.scale, outdir=arguments.outdir)


if __name__ == "__main__":
    main()
