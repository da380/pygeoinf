"""Part C: a prior tight about the Adams-Williamson reference.

The same prior with a pointwise standard deviation of 0.1 g/cm^3 in every
layer. The reference is nine percent short of the Earth's mass, and over a
whole mantle a pointwise 0.1 g/cm^3 averages down to a few hundredths: the
prior and the data disagree, and each kind of answer shows it in its own
way. The posterior travels some eight prior standard deviations from the
reference towards the data and still misses them, and its narrow intervals
miss PREM; the evidence falls by a hundred nats against the loose prior; the
minimum-norm change to the reference fits the data with an error budget
that cannot see its bias; and the feasible set under the prior's 95% ball is
empty, because the nearest model that fits the data lies several radii out.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import LinearForwardProblem, MinimumNorm

from common import (
    bracket_bar,
    draw_band,
    draw_prem,
    draw_profile,
    finish_profile_axes,
    interval_bar,
    output_directory,
    parse,
    part_rng,
    pointwise_std,
    save,
)
from problem import (
    FULL,
    LEVEL,
    LOOSE_STD,
    PROPERTY_SHORT,
    QUICK,
    TIGHT_STD,
    Problem,
    Scale,
    build,
)
from uninformative import feasible_brackets, posterior_summary


def minimum_norm_change(problem: Problem, /) -> dict:
    """The smallest change to the reference, in the space's norm, that fits
    the data at the discrepancy level; its estimate of the three properties
    with the error budget: the noise it propagates, and its bias were PREM
    the truth."""
    space = problem.space
    reference = problem.reference_model
    residual = problem.data - problem.forward(reference)
    change = LinearForwardProblem(problem.forward, error=problem.problem.error_measure)
    estimator = MinimumNorm(change, level=LEVEL)
    search = estimator.discrepancy_search(residual)
    chosen = estimator.for_data(residual)
    model = space.add(reference, chosen(residual))
    pushed = chosen.push_forward(problem.properties)
    estimate = problem.properties(reference) + pushed(residual)
    propagated = np.sqrt(
        np.diag(pushed.propagated_covariance().matrix(form="components"))
    )
    bias = pushed.bias(space.subtract(problem.prem_model, reference))
    chi_squared = problem.problem.chi_squared(model, problem.data)
    print(
        f"minimum-norm change to the reference: damping {search.argument:.4g}, "
        f"chi-squared {chi_squared:.2f} against the critical "
        f"{problem.problem.critical_chi_squared(level=LEVEL):.2f}; the change has "
        f"norm {space.norm(space.subtract(model, reference)):.4g}"
    )
    print(
        f"  {'property':<12s} {'estimate':>9s} {'propagated sd':>14s} "
        f"{'bias at PREM':>13s} {'PREM':>7s}"
    )
    for name, e, s, b, p in zip(
        PROPERTY_SHORT, estimate, propagated, bias, problem.prem_properties
    ):
        print(f"  {name:<12s} {e:9.3f} {s:14.4f} {b:+13.3f} {p:7.3f}")
    return {
        "model": model,
        "damping": search.argument,
        "estimate": estimate,
        "propagated": propagated,
        "bias": bias,
        "chi_squared": chi_squared,
    }


def tight_answer(
    problem: Problem,
    /,
    *,
    rng: np.random.Generator,
    outdir: pathlib.Path,
    loose: dict | None = None,
) -> dict:
    """The three answers under the tight prior, beside the loose prior's."""
    started = time.perf_counter()
    space = problem.space
    print(
        f"\n=== C. A prior tight about the reference ({problem.scale.name} scale) ==="
    )
    if loose is None:
        loose_prior = problem.prior(LOOSE_STD)
        loose = posterior_summary(problem, loose_prior, label="loose prior")
        loose.update(feasible_brackets(problem, loose_prior, label="loose prior"))
    prior = problem.prior(TIGHT_STD)
    print(
        f"prior: about the reference, pointwise std {TIGHT_STD:g} g/cm^3 in every layer"
    )
    result = posterior_summary(problem, prior, label="tight prior")
    shift = (result["center"] - result["prior_properties"].expectation) / result[
        "prior_deviation"
    ]
    outside = np.abs(result["center"] - problem.prem_properties) / result["deviation"]
    print(
        f"  the posterior moved the core mean by {shift[0]:.1f} and the mantle mean "
        f"by {shift[1]:.1f} prior sd; PREM lies {outside[0]:.1f} and {outside[1]:.1f} "
        f"posterior sd from them"
    )
    bayes = loose["log_evidence"] - result["log_evidence"]
    print(
        f"  log evidence {result['log_evidence']:.1f} against the loose prior's "
        f"{loose['log_evidence']:.1f}: the data prefer the loose prior by "
        f"{bayes:.0f} nats"
    )
    result["shift"] = shift
    result["log_bayes_factor"] = bayes
    result["minimum_norm"] = minimum_norm_change(problem)
    result.update(feasible_brackets(problem, prior, label="tight prior"))

    fig, axes = plt.subplots(
        1, 4, figsize=(15.0, 4.2), layout="constrained", width_ratios=[1.6, 1, 1, 1]
    )
    ax = axes[0]
    draw_prem(ax, color="black", lw=1.0, label="PREM")
    draw_band(
        space,
        result["mean"],
        pointwise_std(space, result["posterior"]),
        ax,
        color="C0",
        alpha=0.3,
        label="posterior ± 2 pointwise sd",
    )
    draw_profile(space, result["mean"], ax, color="C0", lw=1.4, label="posterior mean")
    draw_profile(
        space,
        result["minimum_norm"]["model"],
        ax,
        color="C2",
        lw=1.2,
        ls="-.",
        label="minimum-norm change",
    )
    draw_profile(
        space,
        problem.reference_model,
        ax,
        color="C1",
        lw=1.0,
        ls="--",
        label="reference",
    )
    finish_profile_axes(ax, title=f"Prior std {TIGHT_STD:g} g/cm³ about the reference")
    for index, ax in enumerate(axes[1:]):
        interval_bar(
            ax,
            0,
            loose["center"][index],
            1.96 * loose["deviation"][index],
            marker="s",
            color="C0",
            label="B: posterior 95%",
        )
        if loose["brackets"] is not None:
            bracket_bar(ax, 1, loose["brackets"][index], color="C0", label="B: bracket")
        interval_bar(
            ax,
            2,
            result["center"][index],
            1.96 * result["deviation"][index],
            marker="s",
            color="C3",
            label="C: posterior 95%",
        )
        interval_bar(
            ax,
            3,
            result["minimum_norm"]["estimate"][index],
            2.0 * result["minimum_norm"]["propagated"][index],
            marker="o",
            color="C2",
            label="C: minimum norm, 2 sd",
        )
        if result["brackets"] is None:
            ax.text(
                4,
                problem.prem_properties[index],
                "empty",
                rotation=90,
                ha="center",
                va="center",
                fontsize=8,
                color="C3",
            )
        else:
            bracket_bar(
                ax, 4, result["brackets"][index], color="C3", label="C: bracket"
            )
        ax.axhline(
            problem.prem_properties[index], color="black", ls="--", lw=1.0, label="PREM"
        )
        ax.set_xticks([0, 1, 2, 3, 4], ["B", "B set", "C", "C point", "C set"])
        ax.set_xlim(-0.6, 4.6)
        ax.set_title(f"{PROPERTY_SHORT[index]} (g/cm³)", fontsize=10)
        # Zoomed on C and PREM: B's answers run off the panel, as wide as they are.
        values = [
            result["center"][index],
            result["minimum_norm"]["estimate"][index],
            problem.prem_properties[index],
        ]
        spread = max(np.ptp(values), 6.0 * result["deviation"][index], 0.05)
        middle = 0.5 * (max(values) + min(values))
        ax.set_ylim(middle - 1.2 * spread, middle + 1.2 * spread)
    axes[1].legend(fontsize=6, loc="lower left", frameon=False)
    fig.suptitle(
        "A tight prior about a reference the data reject: narrow intervals that miss PREM"
    )
    save(fig, outdir / "tight.png")
    print(f"C took {time.perf_counter() - started:.1f} s")
    return result


def run(
    scale: Scale,
    /,
    *,
    seed: int,
    outdir: pathlib.Path,
    problem: Problem | None = None,
    loose: dict | None = None,
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return tight_answer(problem, rng=part_rng(seed, "c"), outdir=outdir, loose=loose)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
