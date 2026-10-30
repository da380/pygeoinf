"""Part B: the point answer.

The smallest model that fits the data under the discrepancy principle, with
size measured in the prior's norm: the problem is parameterized by the
prior's covariance factor, ``u = L w``, and ``MinimumNorm`` measures ``|w|``.
Then its estimate of the two local averages with its error budget: the noise
it propagates, and its bias, which only the truth reveals.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import MinimumNorm
from pygeoinf.numerics import SpectralFamilySolver

from src.common import error_budget_panels, parse, part_rng, report, save
from src.problem import FULL, INTERVALS, QUICK, Problem, Scale, build


def point_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The minimum-norm model, its error budget on the averages, its figure."""
    started = time.perf_counter()
    scale, space, forward = problem.scale, problem.space, problem.problem
    print(f"\n=== B. The point answer ({scale.name} scale) ===")
    space.calls.clear()
    # The problem in w, with u = L w: the norm of w is the prior's norm of u.
    # The precision factor is the factor's inverse, w = L^-1 u.
    factor = problem.prior.covariance_factor
    whiten = problem.prior.precision_factor
    estimator = MinimumNorm(
        forward.parameterized(factor),
        level=scale.level,
        solver=SpectralFamilySolver(),
    )
    search = estimator.discrepancy_search(problem.data)
    chosen = estimator.for_data(problem.data)
    change = chosen(problem.data)
    smallest = factor(change)
    norm, truth_norm = space.norm(change), space.norm(whiten(problem.truth))
    print(
        f"MinimumNorm by the discrepancy principle: damping {search.argument:.4g} "
        f"after {search.evaluations} misfit evaluations; chi-squared "
        f"{forward.chi_squared(smallest, problem.data):.2f} against the critical "
        f"{forward.critical_chi_squared(level=scale.level):.2f}; prior norm "
        f"{norm:.3f}, the truth's {truth_norm:.3f}; relative error "
        f"{problem.relative_error(smallest):.3f} in L^2"
    )

    pushed = chosen.push_forward(problem.averages @ factor)
    estimate = np.asarray(pushed(problem.data))
    propagated = np.sqrt(pushed.propagated_covariance().diagonals()[0])
    bias = np.asarray(pushed.bias(whiten(problem.truth)))
    print("the minimum-norm estimate of the averages:")
    print(
        f"  {'':<24s} {'estimate':>9s} {'2 prop. sd':>11s} {'bias':>8s} {'truth':>8s}"
    )
    for name, e, s, b, t in zip(
        problem.names, estimate, propagated, bias, problem.true_averages
    ):
        print(f"  {name:<24s} {e:+9.3f} {2 * s:11.3f} {b:+8.3f} {t:+8.3f}")
    calls = report("the point answer", space.calls)

    fig = plt.figure(figsize=(10.0, 7.4), layout="constrained")
    upper, lower = fig.subfigures(2, 1, height_ratios=[1.0, 0.8])
    ax = upper.subplots()
    x = np.array(space.nodes())
    for a, b in INTERVALS:
        ax.axvspan(a, b, color="0.92", lw=0)
    ax.plot(
        x, space.values(problem.truth), color="black", lw=1.5, ls="--", label="truth"
    )
    ax.plot(
        x,
        space.values(smallest),
        color="C1",
        lw=2,
        label=f"minimum norm, damping {search.argument:.3g}",
    )
    ax.errorbar(
        problem.points,
        problem.evaluate.codomain.to_components(problem.data),
        yerr=scale.noise_std,
        fmt="o",
        color="black",
        ms=4,
        capsize=2,
        lw=1,
        label="data",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("the estimate, smallest in the prior's norm", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    error_budget_panels(
        lower.subplots(1, len(INTERVALS)),
        problem.names,
        estimate,
        propagated,
        bias,
        problem.true_averages,
        ylabel="local average",
    )
    save(
        fig,
        outdir / "point.png",
        title="C++ backend, part B: the point answer. The minimum-norm model under "
        "the discrepancy principle,\nand its estimate of each average with its "
        "error budget",
    )
    print(f"B took {time.perf_counter() - started:.1f} s")
    return {
        "damping": search.argument,
        "chi_squared": forward.chi_squared(smallest, problem.data),
        "critical": forward.critical_chi_squared(level=scale.level),
        "norm": norm,
        "truth_norm": truth_norm,
        "estimate": estimate,
        "propagated": propagated,
        "bias": bias,
        "smallest": smallest,
        "true_averages": problem.true_averages,
        "calls": calls,
    }


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale)
    return point_answer(problem, rng=part_rng(scale.seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
