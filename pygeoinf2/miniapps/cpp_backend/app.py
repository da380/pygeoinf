"""An inverse problem on a C++ vector backend, end to end.

Twelve point observations of a random field on ``[0, 1]``, under Gaussian
noise, with a Matérn-type prior: the posterior by ``LinearGaussianInversion``,
the smallest model fitting the data by ``MinimumNorm`` under the discrepancy
principle, and the feasible range of two local averages by
``BackusGilbertParker`` with the prior's ambient ball and the noise's credible
ball. Every vector is a ``cppvec.Vector``: the library never sees an array of
nodal values, and the space counts what it was asked so the run can say so.

Run ``python build.py`` first, then ``python app.py``. The figure is written
to ``cpp_backend.png`` next to this file; nothing is shown on screen.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections import Counter

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2.algebra.spaces import CoordinateSpace
from pygeoinf2.inference import (
    BackusGilbertParker,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
)
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import (
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

from space import (
    CppSpace,
    laplacian_operator,
    local_average_operator,
    matern_measure,
    point_evaluation_operator,
)

HERE = pathlib.Path(__file__).resolve().parent
N_ELEMENTS = 200
N_DATA = 12
NOISE_STD = 0.1
LENGTH_SCALE = 0.08
LEVEL = 0.95
INTERVALS = [(0.2, 0.35), (0.6, 0.85)]


def report(title: str, calls: Counter) -> None:
    """Print the backend members the library called since the last report."""
    print(f"  backend calls during {title}:")
    for name, count in sorted(calls.items()):
        print(f"    {name:24s} {count:7d}")
    calls.clear()


def main() -> None:
    """Run the checks, the three estimators and draw the figure."""
    rng = np.random.default_rng(2026)
    started = time.perf_counter()

    space = CppSpace(N_ELEMENTS)
    print(space)
    print("  a CoordinateSpace?", isinstance(space, CoordinateSpace))
    print(
        "  vectors are",
        type(space.zero()).__name__,
        "from",
        type(space.zero()).__module__,
    )
    prior = matern_measure(space, length_scale=LENGTH_SCALE)
    points = sorted(rng.uniform(0.03, 0.97, N_DATA).tolist())
    evaluate = point_evaluation_operator(space, points)
    laplacian = laplacian_operator(space)
    averages = local_average_operator(space, INTERVALS)
    space.calls.clear()

    # --- the check ladder --------------------------------------------------
    print("\nThe check ladder:")
    print(" ", check_space(space, rng=rng, rebuild=lambda: CppSpace(N_ELEMENTS)))
    print(" ", check_white_noise(space, rng=rng, measure=prior))
    for name, operator in (
        ("point evaluation", evaluate),
        ("Laplacian", laplacian),
        ("local averages", averages),
        ("prior covariance", prior.covariance),
    ):
        print(f"  {name}:", check_operator(operator, rng=rng, measure=prior))
        print(f"  {name}:", check_traits(operator, rng=rng, measure=prior))
    report("the checks", space.calls)
    print(f"  ({time.perf_counter() - started:.1f} s so far)")

    # --- the problem ---------------------------------------------------------
    noise = GaussianMeasure.from_standard_deviation(evaluate.codomain, NOISE_STD)
    problem = LinearForwardProblem(evaluate, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    print(
        f"\n{N_DATA} point observations, noise {NOISE_STD}, prior length scale {LENGTH_SCALE}"
    )
    print(
        f"  chi-squared of the truth: {problem.chi_squared(truth, data):.2f} on {N_DATA} data"
    )
    space.calls.clear()

    # --- the posterior -------------------------------------------------------
    t0 = time.perf_counter()
    posterior = LinearGaussianInversion(problem, prior)(data)
    mean = posterior.expectation
    error = space.norm(space.subtract(mean, truth)) / space.norm(truth)
    print("\nLinearGaussianInversion:")
    print(f"  relative error of the posterior mean  {error:.3f}")
    print(
        f"  chi-squared of the mean               {problem.chi_squared(mean, data):.2f}"
    )
    # The pointwise standard deviation: push the posterior through evaluation
    # at every node and read the diagonal of the covariance there. The space
    # has no pointwise product, so this, or sampling, is the route.
    nodes = space.nodes()
    at_nodes = point_evaluation_operator(space, nodes)
    pointwise = posterior.affine_map(at_nodes)
    std = np.sqrt(pointwise.covariance.diagonals()[0])
    samples = posterior.samples(3, rng=rng)
    print(f"  pointwise standard deviation          {std.min():.3f} to {std.max():.3f}")
    print(f"  posterior in {time.perf_counter() - t0:.1f} s")
    report("the inversion", space.calls)

    # --- the minimum-norm fit ------------------------------------------------
    t0 = time.perf_counter()
    estimator = MinimumNorm(problem, level=LEVEL)
    search = estimator.discrepancy_search(data)
    smallest = estimator(data)
    print("\nMinimumNorm under the discrepancy principle:")
    print(
        f"  damping {search.argument:.4g} after {search.evaluations} misfit evaluations"
    )
    print(
        f"  chi-squared of the estimate           {problem.chi_squared(smallest, data):.2f}"
    )
    print(f"  norm of the estimate                  {space.norm(smallest):.3f}")
    print(f"  norm of the truth                     {space.norm(truth):.3f}")
    print(f"  in {time.perf_counter() - t0:.1f} s")
    report("the minimum-norm fit", space.calls)

    # --- the feasible set of two local averages ------------------------------
    t0 = time.perf_counter()
    ball = prior.ambient_ball(level=LEVEL, rng=rng)
    print("\nBackusGilbertParker on two local averages:")
    print(
        f"  prior ambient ball radius {ball.radius:.3f}; truth norm {space.norm(truth):.3f}"
    )
    feasible = BackusGilbertParker(problem, averages, ball, level=LEVEL)(data)
    print(
        f"  the sets are {feasible.form!r}; support by {feasible.algorithms['support'][0]}"
    )
    property_space = averages.codomain
    true_averages = property_space.to_components(averages(truth))
    posterior_averages = posterior.affine_map(averages)
    posterior_center = property_space.to_components(posterior_averages.expectation)
    posterior_std = np.sqrt(posterior_averages.covariance.diagonals()[0])
    brackets = []
    for index, (a, b) in enumerate(INTERVALS):
        direction = property_space.basis_vector(index)
        hi_inner, hi_outer = feasible.support_bracket(direction)
        lo_inner, lo_outer = feasible.support_bracket(
            property_space.scale(-1.0, direction)
        )
        brackets.append((-lo_outer, hi_outer))
        print(
            f"  average over [{a}, {b}]: feasible [{-lo_outer:+.3f}, {hi_outer:+.3f}] "
            f"(bracket widths {lo_outer - lo_inner:.1e}, {hi_outer - hi_inner:.1e}); "
            f"truth {true_averages[index]:+.3f}; posterior "
            f"{posterior_center[index]:+.3f} +/- {posterior_std[index]:.3f}"
        )
    print(f"  in {time.perf_counter() - t0:.1f} s")
    report("the feasible set", space.calls)

    # --- the figure ------------------------------------------------------------
    # to_list() is called here and nowhere else: the plotting boundary.
    x = np.array(nodes)
    truth_values = np.array(truth.to_list())
    mean_values = np.array(mean.to_list())
    smallest_values = np.array(smallest.to_list())
    data_values = evaluate.codomain.to_components(data)

    blue, orange, aqua, ink, muted = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#0b0b0b",
        "#52514e",
    )
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 7.5),
        gridspec_kw={"height_ratios": [3, 1.3]},
        constrained_layout=True,
    )
    top.fill_between(
        x, mean_values - 2 * std, mean_values + 2 * std, color=blue, alpha=0.18, lw=0
    )
    for k, sample in enumerate(samples):
        top.plot(
            x,
            sample.to_list(),
            color=blue,
            lw=0.8,
            alpha=0.45,
            label="posterior samples" if k == 0 else None,
        )
    top.plot(x, truth_values, color=orange, lw=2, label="truth")
    top.plot(x, mean_values, color=blue, lw=2, label="posterior mean, 2 std band")
    top.plot(
        x,
        smallest_values,
        color=aqua,
        lw=2,
        ls="--",
        label="minimum norm (discrepancy)",
    )
    top.errorbar(
        points,
        data_values,
        yerr=NOISE_STD,
        fmt="o",
        color=ink,
        ms=5,
        capsize=2,
        lw=1,
        label="data",
    )
    top.set_xlim(0, 1)
    top.set_xlabel("x")
    top.set_ylabel("u(x)")
    top.set_title("A random field from 12 noisy point values, on cppvec.Vector")
    top.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    for side in ("top", "right"):
        top.spines[side].set_visible(False)

    for index, ((a, b), (lo, hi)) in enumerate(zip(INTERVALS, brackets)):
        y = 1 - index
        bottom.plot([lo, hi], [y, y], color=ink, lw=4, solid_capstyle="butt")
        bottom.errorbar(
            posterior_center[index],
            y + 0.22,
            xerr=2 * posterior_std[index],
            fmt="s",
            color=blue,
            ms=5,
            capsize=3,
            lw=1.5,
        )
        bottom.plot(true_averages[index], y, marker="|", color=orange, ms=18, mew=2.5)
        bottom.text(
            hi,
            y - 0.22,
            f"average over [{a}, {b}]",
            color=muted,
            fontsize=9,
            ha="right",
            va="top",
        )
    bottom.set_yticks([])
    bottom.set_ylim(-0.6, 1.7)
    bottom.set_xlabel("local average")
    bottom.set_title(
        "Feasible interval (black), posterior 2 std (blue), truth (orange)", fontsize=10
    )
    for side in ("top", "right", "left"):
        bottom.spines[side].set_visible(False)

    out = HERE / "cpp_backend.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nfigure written to {out}")
    print(f"total {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
