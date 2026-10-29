"""The three answers under one prior, each with its own figure: the point,
the measure and the set.

Parts B and C ask the same three questions under two Gaussians about PREM,
the reference, a tight one and a deliberately wide one; this module answers
them for whichever it is handed.

- **The point**: the smallest change to the prior's expectation that fits
  the data at the discrepancy level, measured in the prior's precision, with
  its error budget on the three properties.
- **The measure**: the posterior, against the prior, on the profile and on
  the three properties, and the log evidence.
- **The set**: the prior hardened to its 95% ellipsoid, the ellipsoid of its
  covariance to a power below three quarters, which carries its 95% however
  fine the space; the noise to its 95% ellipse; and the brackets that leave on
  the three properties, against what the prior's ellipsoid admits on its
  own.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    BackusGilbertParker,
    GaussianMeasure,
    LinearGaussianInversion,
    MinimumNorm,
    plotting,
)

from .common import (
    draw_band,
    draw_profile,
    finish_profile_axes,
    pointwise_std,
    save,
    title,
)
from .problem import PROPERTY_NAMES, Problem

UNITS = "g/cm³"


def _deviations(covariance: Any, /) -> np.ndarray:
    """The standard deviations a covariance on a Euclidean space of a few
    dimensions gives each component, read off its matrix."""
    return np.sqrt(np.clip(np.diag(covariance.matrix(form="components")), 0.0, None))


def _truth_line(ax: Any, value: float, /) -> None:
    ax.axhline(value, color="black", ls="--", lw=1.0, label="truth")


def point_answer(
    problem: Problem,
    prior: GaussianMeasure,
    /,
    *,
    part: str,
    prior_name: str,
    outdir: pathlib.Path,
) -> dict:
    """The minimum-norm change from the prior's expectation, with its budget.

    A change is written ``L w``, with ``L`` the prior's covariance factor,
    so that ``|w|`` is the change's norm in the prior's precision; the
    estimator is :class:`MinimumNorm` on the problem in ``w``, applied to
    the data less the reference's prediction. The budget is the noise the
    estimate propagates, and its bias at the truth, whose departure from the
    reference is ``L w`` with ``w`` read off by the prior's precision factor.
    """
    space, properties = problem.space, problem.properties
    reference, factor = prior.expectation, prior.covariance_factor
    residual = problem.forward.codomain.subtract(
        problem.data, problem.forward(reference)
    )
    level = problem.scale.level
    estimator = MinimumNorm(problem.problem.parameterized(factor), level=level)
    search = estimator.discrepancy_search(residual)
    chosen = estimator.for_data(residual)
    change = chosen(residual)
    model = space.add(reference, factor(change))
    pushed = chosen.push_forward(properties @ factor)
    estimate = properties(reference) + pushed(residual)
    propagated = _deviations(pushed.propagated_covariance())
    departure = prior.precision_factor(space.subtract(problem.truth, reference))
    bias = pushed.bias(departure)
    misfit = problem.problem.chi_squared(model, problem.data)
    critical = problem.problem.critical_chi_squared(level=level)
    if search.exhausted == "high":
        print(
            f"point: the reference already fits the data, chi-squared "
            f"{problem.problem.chi_squared(reference, problem.data):.2f} against "
            f"the critical {critical:.2f}, and the smallest change is none"
        )
    else:
        print(
            f"point: damping {search.argument:.4g} after {search.evaluations} "
            f"misfit evaluations; chi-squared {misfit:.2f} against the critical "
            f"{critical:.2f}, from {problem.problem.chi_squared(reference, problem.data):.1f} "
            f"at the reference; the change has precision norm "
            f"{factor.domain.norm(change):.3g}, the truth's departure "
            f"{factor.domain.norm(departure):.3g}"
        )
    print(
        f"  {'property':<12s} {'reference':>9s} {'estimate':>9s} "
        f"{'propagated sd':>14s} {'bias':>8s} {'error':>8s} {'truth':>7s}"
    )
    for name, r, e, s, b, t in zip(
        PROPERTY_NAMES,
        properties(reference),
        estimate,
        propagated,
        bias,
        problem.true_properties,
    ):
        print(
            f"  {name:<12s} {r:9.3f} {e:9.3f} {s:14.4f} {b:+8.3f} {e - t:+8.3f} "
            f"{t:7.3f}"
        )

    fig = plt.figure(figsize=(13.0, 4.4), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.0, 1.25])
    ax = left.subplots()
    for vector, color, label in (
        (problem.truth, "black", "truth"),
        (model, "C2", "estimate"),
    ):
        offset = space.subtract(vector, reference)
        draw_profile(space, offset, ax, color=color, lw=1.3, label=label)
    ax.axhline(0.0, color="0.5", lw=1.0, ls=":", label="PREM, the reference")
    finish_profile_axes(ax, subtitle="the profile less the reference", legend=False)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.set_ylabel(f"density less PREM ({UNITS})")
    axes = right.subplots(1, 3)
    for index, ax in enumerate(axes):
        ax.errorbar(
            0,
            estimate[index],
            yerr=2.0 * propagated[index],
            fmt="o",
            color="C2",
            capsize=4,
            label="estimate, 2 propagated sd",
        )
        _truth_line(ax, problem.true_properties[index])
        ax.axhline(
            properties(reference)[index], color="0.5", ls=":", lw=1.0, label="PREM"
        )
        expected = problem.true_properties[index] + bias[index]
        ax.annotate(
            "",
            xy=(1, expected),
            xytext=(1, problem.true_properties[index]),
            arrowprops={"arrowstyle": "->", "color": "C3"},
        )
        ax.plot(1, expected, "o", mfc="none", color="C3", label="truth plus the bias")
        ax.set_xticks([0, 1], ["these data", "exact data"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{PROPERTY_NAMES[index]} ({UNITS})", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    right.legend(
        handles, labels, loc="outside lower center", ncols=2, fontsize=7, frameon=False
    )
    fig.suptitle(
        title(
            f"{part}, the {prior_name} prior",
            "the point answer, the minimum-norm change from PREM, and its error budget",
        )
    )
    save(fig, outdir / f"{prior_name}_point.png")
    return {
        "model": model,
        "damping": search.argument,
        "saturated": search.exhausted == "high",
        "chi_squared": misfit,
        "estimate": estimate,
        "propagated": propagated,
        "bias": bias,
    }


def measure_answer(
    problem: Problem,
    prior: GaussianMeasure,
    /,
    *,
    part: str,
    prior_name: str,
    rng: np.random.Generator,
    outdir: pathlib.Path,
) -> dict:
    """The posterior against the prior: bands on the profile, the three
    properties, and the log evidence."""
    space, properties = problem.space, problem.properties
    inversion = LinearGaussianInversion(problem.problem, prior)
    posterior = inversion(problem.data)
    before = prior.push_forward(properties)
    after = posterior.push_forward(properties)
    prior_deviation = _deviations(before.covariance)
    deviation = _deviations(after.covariance)
    center = after.expectation
    covariance = after.covariance.matrix(form="components")
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    log_evidence = inversion.log_evidence(problem.data)
    misfit = problem.problem.chi_squared(posterior.expectation, problem.data)
    print(
        f"measure: posterior mean chi-squared {misfit:.2f}; log evidence "
        f"{log_evidence:.2f}; core and mantle means correlated at {correlation:+.4f}"
    )
    print(
        f"  {'property':<12s} {'prior':>15s} {'posterior':>17s} {'95% from':>9s} "
        f"{'to':>7s} {'truth':>7s}"
    )
    for name, p, s0, m, s, t in zip(
        PROPERTY_NAMES,
        before.expectation,
        prior_deviation,
        center,
        deviation,
        problem.true_properties,
    ):
        print(
            f"  {name:<12s} {p:7.3f} ± {s0:5.3f} {m:8.3f} ± {s:6.3f} "
            f"{m - 1.96 * s:9.3f} {m + 1.96 * s:7.3f} {t:7.3f}"
        )

    fig = plt.figure(figsize=(13.0, 5.2), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.0, 1.15])
    ax = left.subplots()
    # Drawn less the reference, which the prior is centered on: a band of 0.1
    # g/cm^3 is invisible on the scale of the profile itself.
    reference = prior.expectation
    for measure, color, alpha, label in (
        (prior, "0.6", 0.3, "prior, mean ± 2 sd"),
        (posterior, "C0", 0.35, "posterior, mean ± 2 sd"),
    ):
        offset = space.subtract(measure.expectation, reference)
        draw_band(
            space,
            offset,
            pointwise_std(space, measure),
            ax,
            color=color,
            alpha=alpha,
            label=label,
        )
    draw_profile(
        space, space.subtract(posterior.expectation, reference), ax, color="C0"
    )
    draw_profile(
        space,
        space.subtract(problem.truth, reference),
        ax,
        color="black",
        lw=1.0,
        ls="--",
        label="truth",
    )
    finish_profile_axes(
        ax, subtitle="the profile less the reference, pointwise", legend=False
    )
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_ylabel(f"density less PREM ({UNITS})")
    corner = right.subplots(3, 3)
    plotting.plot_corner(
        after,
        prior=before,
        truth=problem.true_properties,
        labels=[f"{name} ({UNITS})" for name in PROPERTY_NAMES],
        axes=corner,
        rng=rng,
    )
    right.suptitle("the three properties", fontsize=10)
    fig.suptitle(
        title(
            f"{part}, the {prior_name} prior",
            "the measure answer, the prior against the posterior",
        )
    )
    save(fig, outdir / f"{prior_name}_measure.png")
    return {
        "posterior": posterior,
        "center": center,
        "deviation": deviation,
        "prior_deviation": prior_deviation,
        "correlation": correlation,
        "log_evidence": log_evidence,
        "chi_squared": misfit,
    }


def set_answer(
    problem: Problem,
    prior: GaussianMeasure,
    /,
    *,
    part: str,
    prior_name: str,
    outdir: pathlib.Path,
) -> dict:
    """The brackets the data leave under the prior's 95% ellipsoid and the
    noise's, against the bound the prior's ellipsoid admits alone.

    The ellipsoid is ``{m : (Q^-p (m - m0), m - m0) <= r^2}``, the prior's
    covariance ``Q`` to the scale's power ``p``, with ``r`` the radius that
    carries 95% of the prior. At ``p = 1`` it would be the Mahalanobis
    ellipsoid, whose radius grows with the number of basis functions; below
    three quarters, for a measure of order two whose spectrum grows as on a
    line, it carries its probability in the limit.
    """
    properties = problem.properties
    level, power = problem.scale.level, problem.scale.set_power
    ellipsoid = prior.weakened_ellipsoid(level=level, power=power)
    # The errors are independent but of two sizes, so the noise's 95% set is
    # its own Mahalanobis ellipse on the two data, not a ball.
    noise = problem.problem.error_measure.credible_set(level=level)
    feasible = BackusGilbertParker(problem.problem, properties, ellipsoid, noise=noise)(
        problem.data
    )
    P = properties.codomain
    directions = []
    for index in range(P.dim):
        q = P.basis_vector(index)
        directions += [q, P.scale(-1.0, q)]
    sweep = feasible.support_sweep(directions)
    outer = np.array([result.value for result in sweep.results])
    inner = np.array([result.inner for result in sweep.results])
    support = ellipsoid.support_function()
    alone = np.array([support(properties.adjoint(q)) for q in directions])
    brackets = np.column_stack([-outer[1::2], outer[0::2]])
    bounds = np.column_stack([-alone[1::2], alone[0::2]])
    gaps = np.maximum(outer[0::2] - inner[0::2], outer[1::2] - inner[1::2])
    space = problem.space
    departure = space.subtract(problem.truth, prior.expectation)
    gauge = space.inner_product(ellipsoid.precision(departure), departure)
    print(
        f"set: the prior's {level:.0%} ellipsoid (Q^-p (m - m0), m - m0) <= r^2 "
        f"with p = {power:g}, the truth's gauge {gauge:.2f} of the boundary's 1, "
        f"inside: {ellipsoid.contains(problem.truth)}; the noise to its own "
        f"{level:.0%} ellipse"
    )
    print(
        f"  {'property':<12s} {'the ellipsoid alone':>19s} {'with the data':>19s} "
        f"{'gap':>8s} {'truth':>7s}"
    )
    for name, (a, b), (low, high), gap, t in zip(
        PROPERTY_NAMES, bounds, brackets, gaps, problem.true_properties
    ):
        print(
            f"  {name:<12s} [{a:7.3f}, {b:7.3f}] [{low:7.3f}, {high:7.3f}] "
            f"{gap:8.1e} {t:7.3f}"
        )

    fig, axes = plt.subplots(1, P.dim, figsize=(10.0, 4.0), layout="constrained")
    for index, ax in enumerate(axes):
        for position, (low, high), color, label in (
            (0, bounds[index], "0.6", "the prior's ellipsoid alone"),
            (1, brackets[index], "C1", "with the data"),
        ):
            ax.plot(
                [position, position],
                [low, high],
                color=color,
                lw=8,
                solid_capstyle="butt",
                label=label,
            )
        _truth_line(ax, problem.true_properties[index])
        ax.set_xticks([0, 1], ["prior ellipsoid", "with the data"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{PROPERTY_NAMES[index]} ({UNITS})", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(
        title(
            f"{part}, the {prior_name} prior",
            "the set answer, the prior's ellipsoid against the brackets",
        )
    )
    save(fig, outdir / f"{prior_name}_set.png")
    return {
        "gauge": gauge,
        "truth_in_ellipsoid": bool(ellipsoid.contains(problem.truth)),
        "bounds": bounds,
        "brackets": brackets,
        "gaps": gaps,
    }


def three_answers(
    problem: Problem,
    prior: GaussianMeasure,
    /,
    *,
    part: str,
    prior_name: str,
    rng: np.random.Generator,
    outdir: pathlib.Path,
) -> dict:
    """The point, the measure and the set under one prior, in that order.

    Args:
        problem: the problem.
        prior: the Gaussian, which the set answer hardens to its ellipsoid.
        part: the part's letter, for the titles.
        prior_name: what the prior is called, for the titles and the files.
        rng: the generator for the corner plot, should it need draws.
        outdir: where the figures go.
    """
    labels = {"part": part, "prior_name": prior_name, "outdir": outdir}
    answers = (
        ("point", lambda: point_answer(problem, prior, **labels)),
        ("measure", lambda: measure_answer(problem, prior, rng=rng, **labels)),
        ("set", lambda: set_answer(problem, prior, **labels)),
    )
    result = {}
    for kind, answer in answers:
        started = time.perf_counter()
        result[kind] = answer()
        print(f"  the {kind} answer took {time.perf_counter() - started:.2f} s")
    return result
