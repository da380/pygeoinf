"""Part F: the answers, from the chains of part D.

Everything asked of the model here is nonlinear in it, which is why a sampler and
not a Gaussian about a mode: where the provinces are, as the probability that a
point lies inside one; the contrasts inside them, and whether the provinces are
dense, as the probability that ``c_rho`` is positive; the jump in ``dlnVs`` across
their edges over the jump in ``dlnVp``, against the thermal scaling, which a
thermal origin would give and a chemical one need not; and how many provinces
there are. Each is read off the chains' records or their stored states, and each
is shown by chain, since where the chains disagree the pooled answer is not yet one.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs

from pygeoinf import LinearGaussianInversion, plotting
from pygeoinf.numerics.solvers import CGSolver

import gibbs as part_d
from src.common import parse, save
from src.prior import correlation_matrix, ungrouped
from src.record import NAMES, area_weights, summary
from src.shell import FIELDS, FULL, QUICK, Scale
from synthetic import Setup, setup

SOLVER_RTOL = 1e-8


def thermal_ratio(scale: Scale, /) -> float:
    """The shear jump over the compressional that the thermal prior's scaling
    makes: the regression of ``dlnVs`` on ``dlnVp``."""
    correlation = correlation_matrix(scale)[0, 1]
    return float(correlation * scale.thermal_std[0] / scale.thermal_std[1])


def province_probability(state: Setup, stored: Any, /) -> np.ndarray:
    """The share of the stored states that hold each grid cell in a province."""
    space = state.space
    return np.mean(
        [space.grid_values(ungrouped(model)[1]) > 0.0 for model in stored], axis=0
    )


def answers(state: Setup, main: Any, /, *, outdir: pathlib.Path) -> dict:
    """The answers, by chain and pooled, and the figure."""
    scale, space = state.scale, state.space
    truth = summary(space, state.truth)
    values = main.scalars()
    errors = {name: e for name, e in zip(NAMES, main.scalar_means()[1:])}
    rhat = main.diagnostics().rhat[1:]
    print(
        f"  {'scalar':<10s} {'truth':>7s} {'mean':>7s} {'sd':>6s} {'R-hat':>6s}  by chain"
    )
    rows = {}
    for k, name in enumerate(NAMES):
        pooled = values[1 + k]
        rows[name] = {
            "truth": float(truth[k]),
            "mean": float(pooled.mean()),
            "std": float(pooled.std(ddof=1)),
            "chains": pooled.mean(axis=1),
            "rhat": float(rhat[k]),
            "error": errors[name].standard_error,
        }
        shown = "   --" if name == "provinces" else f"{rhat[k]:6.2f}"
        print(
            f"  {name:<10s} {truth[k]:7.3f} {pooled.mean():7.3f} {pooled.std(ddof=1):6.3f} "
            f"{shown:>6s}  " + " ".join(f"{m:6.2f}" for m in pooled.mean(axis=1))
        )
    potentials = values[0].mean(axis=1)
    print(
        "  the mean potential by chain: "
        + " ".join(f"{p:.0f}" for p in potentials)
        + f"; the truth's {state.problem.potential(state.data)(state.truth):.0f}"
    )
    c_rho = values[1 + NAMES.index("c_rho")]
    ratio = values[1 + NAMES.index("ratio")]
    count = values[1 + NAMES.index("provinces")]
    thermal = thermal_ratio(scale)
    dense = float((c_rho > 0.0).mean())
    breaks = float((ratio > thermal).mean())
    # What thermal fields alone make of the same data, on the true provinces.
    thermal_mean = LinearGaussianInversion(
        state.thermal_problem, state.thermal, solver=CGSolver(rtol=SOLVER_RTOL)
    )(state.data).expectation
    alone = summary(space, (thermal_mean, state.truth[1], np.zeros(len(FIELDS))))
    alone_ratio = float(alone[NAMES.index("ratio")])
    counts = {int(n): float((count == n).mean()) for n in np.unique(count)}
    print(
        f"  P(c_rho > 0), the provinces dense: {dense:.3f}; by chain "
        + " ".join(f"{m:.2f}" for m in (c_rho > 0.0).mean(axis=1))
    )
    print(
        f"  the shear jump over the compressional: {ratio.mean():.2f} +- "
        f"{ratio.std(ddof=1):.2f}, the truth's {truth[NAMES.index('ratio')]:.2f}; the "
        f"thermal scaling {thermal:.2f}, and {alone_ratio:.2f} from the thermal-only "
        f"posterior mean across the true edges; P(above the scaling) {breaks:.3f}"
    )
    print(
        "  how many provinces: "
        + ", ".join(f"{n}: {share:.0%}" for n, share in counts.items())
        + "; by chain "
        + " ".join(f"{m:.2f}" for m in count.mean(axis=1))
    )
    probability = province_probability(state, main.stored_positions)
    inside = space.grid_values(state.truth[1]) > 0.0
    weights = area_weights(space)
    agreement = float(
        np.sum(weights[inside] * probability[inside]) / weights[inside].sum()
    )
    outside = float(
        np.sum(weights[~inside] * probability[~inside]) / weights[~inside].sum()
    )
    print(
        f"  from {len(main.stored_positions)} stored states: a point inside the true "
        f"provinces is inside the sampled ones with mean probability {agreement:.2f}, a "
        f"point outside them with {outside:.2f}"
    )
    draw(state, values, probability, thermal, alone_ratio, outdir=outdir)
    return {
        "rows": rows,
        "dense": dense,
        "breaks": breaks,
        "thermal": thermal,
        "alone": alone_ratio,
        "counts": counts,
        "probability": probability,
        "agreement": agreement,
        "outside": outside,
        "potentials": potentials,
    }


def draw(
    state: Setup,
    values: np.ndarray,
    probability: np.ndarray,
    thermal: float,
    alone: float,
    /,
    *,
    outdir: pathlib.Path,
) -> None:
    """The province probability, and the marginals by chain against the truth."""
    space, scale = state.space, state.scale
    truth = summary(space, state.truth)
    colors = part_d.CHAIN_COLORS
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0])
    ax = fig.add_subplot(grid[0, :2], projection=crs.Robinson())
    plotting.plot(
        space,
        space.from_grid_values(probability),
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        coasts=True,
        gridlines=False,
        colorbar=True,
    )
    latitudes = 90.0 - np.degrees(space.colatitudes)
    longitudes = np.degrees(space.longitudes)
    ax.contour(
        longitudes,
        latitudes,
        space.grid_values(state.truth[1]),
        levels=[0.0],
        colors="k",
        linewidths=1.2,
        transform=crs.PlateCarree(),
    )
    ax.set_title(
        "P(a point lies inside a province); the true edges in black", fontsize=10
    )
    panels = (
        ("c_rho", "the density contrast, percent", grid[0, 2], scale.chemistry_std[2]),
        ("ratio", "the shear jump over the compressional", grid[0, 3], None),
        ("c_s", "the shear contrast, percent", grid[1, 0], scale.chemistry_std[0]),
        (
            "c_p",
            "the compressional contrast, percent",
            grid[1, 1],
            scale.chemistry_std[1],
        ),
        ("share", "the provinces' share of the sphere", grid[1, 2], None),
    )
    for name, label, cell, prior_std in panels:
        k = NAMES.index(name)
        ax = fig.add_subplot(cell)
        bins = np.linspace(values[1 + k].min(), values[1 + k].max(), 30)
        for chain, color in zip(values[1 + k], colors):
            ax.hist(
                chain, bins=bins, histtype="step", lw=1.3, density=True, color=color
            )
        ax.axvline(truth[k], color="k", lw=1.5, label="truth")
        if prior_std is not None:
            ax.axvline(0.0, color="0.6", lw=1.0, ls=":", label="the prior's mean")
        if name == "ratio":
            ax.axvline(thermal, color="0.35", lw=1.2, ls="--", label="thermal scaling")
            ax.axvline(alone, color="0.6", lw=1.2, ls=":", label="thermal fields alone")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    ax = fig.add_subplot(grid[1, 3])
    count = values[1 + NAMES.index("provinces")]
    numbers = np.arange(count.min(), count.max() + 1)
    width = 0.8 / max(len(count), 1)
    for j, (chain, color) in enumerate(zip(count, colors)):
        shares = [(chain == n).mean() for n in numbers]
        ax.bar(
            numbers + j * width - 0.4, shares, width=width, align="edge", color=color
        )
    ax.axvline(truth[NAMES.index("provinces")], color="k", lw=1.5)
    ax.set_xticks(numbers, [f"{n:.0f}" for n in numbers])
    ax.set_xlabel("how many provinces, by chain")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(
        fig,
        outdir / "answers.png",
        title="LLSVP, part F: where the provinces are and what they are made of, "
        "each chain its own colour",
    )


def run(
    scale: Scale,
    /,
    *,
    outdir: pathlib.Path,
    state: Setup | None = None,
    gibbs: dict | None = None,
) -> dict:
    started = time.perf_counter()
    print(f"\n=== F. The answers ({scale.name} scale) ===")
    if state is None:
        state = setup(scale)
    if gibbs is None:
        gibbs = part_d.run(scale, outdir=outdir, state=state)
    result = answers(state, gibbs["main"], outdir=outdir)
    print(f"F took {time.perf_counter() - started:.1f} s")
    return result


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
