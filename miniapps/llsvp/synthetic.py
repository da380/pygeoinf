"""Part B: the coupled prior, the truth and the data.

Draws of the three thermal fields, which share their latent noise and so their
pattern, side by side with draws of the level set, whose provinces are the
places it is positive. The truth: a thermal draw from the prior, a level set
with a province under Africa and one under the Pacific, and contrasts inside
them that break the thermal scaling, slow in shear and dense. What the data
see of it, kind by kind, with the coverage of the base of the mantle that the
world's earthquakes and stations leave.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pygeoinf import ForwardProblem, GaussianMeasure, LinearForwardProblem, plotting

from src.common import draw_map, map_panels, parse, part_rng, save
from src.forward import Forward
from src.prior import (
    correlation_matrix,
    full_prior,
    grouped_prior,
    level_set_prior,
    sphere,
    thermal_prior,
    true_level_set,
)
from src.shell import (
    FIELDS,
    FULL,
    QUICK,
    REGIONS,
    Network,
    Scale,
    network,
    separation,
    to_vector,
)

# The kinds of data, each with the scale's field of its noise fraction.
NOISE = {
    "noise_s": ("s_arcs", "s_caps"),
    "noise_p": ("p_arcs", "p_caps"),
    "noise_modes": ("modes",),
}


@dataclass
class Setup:
    """Everything the parts share, from the seed alone.

    Attributes:
        scale: the size of the problem.
        network: where the data are.
        space: the sphere of one field.
        forward: the forward map.
        prior: the prior on the whole model space.
        thermal: the prior on the thermal fields alone.
        truth: the true model, ``(thermal, level set, contrasts)``.
        clean: the truth's data.
        data: the truth's data with noise.
        noise: the standard deviation of each datum.
        error: the noise's measure.
        problem: the whole nonlinear forward problem.
        grouped: the same prior, the linear part one summand.
        grouped_problem: the same problem on that sum.
        thermal_problem: the linear problem of the thermal fields alone.
    """

    scale: Scale
    network: Network
    space: Any
    forward: Forward
    prior: GaussianMeasure
    thermal: GaussianMeasure
    truth: Any
    clean: np.ndarray
    data: np.ndarray
    noise: np.ndarray
    error: GaussianMeasure
    problem: ForwardProblem
    grouped: GaussianMeasure
    grouped_problem: ForwardProblem
    thermal_problem: LinearForwardProblem


def noise_levels(scale: Scale, forward: Forward, clean: np.ndarray, /) -> np.ndarray:
    """Each datum's noise: a fraction of the rms of its kind's clean data."""
    levels = np.zeros_like(clean)
    for field, kinds in NOISE.items():
        rows = np.concatenate(
            [np.arange(clean.size)[forward.slices[kind]] for kind in kinds]
        )
        if rows.size:
            levels[rows] = getattr(scale, field) * np.sqrt(np.mean(clean[rows] ** 2))
    return levels


def setup(scale: Scale, /, *, quiet: bool = False) -> Setup:
    """The network, the spaces, the truth and the data, the same for every part.

    Args:
        scale: the size of the problem, and the seed.
        quiet: say nothing.

    Returns:
        What the parts share.
    """
    say = (lambda *args: None) if quiet else print
    started = time.perf_counter()
    grid = network(scale)
    space = sphere(scale)
    forward = Forward(space, grid, scale, n_jobs=scale.n_jobs)
    prior = full_prior(scale, space)
    thermal = thermal_prior(scale, space)
    rng = part_rng(scale.seed, "b")
    truth = (
        thermal.sample(rng=rng),
        true_level_set(scale, space),
        np.array(scale.contrasts, dtype=float),
    )
    clean = forward(truth)
    noise = noise_levels(scale, forward, clean)
    error = GaussianMeasure.from_standard_deviations(forward.data_space, noise)
    data = clean + error.sample(rng=rng)
    problem = ForwardProblem(forward.operator(prior.domain), error=error)
    together = grouped_prior(scale, space)
    grouped_problem = ForwardProblem(
        forward.grouped_operator(together.domain), error=error
    )
    thermal_problem = LinearForwardProblem(forward.thermal, error=error)
    say(
        f"the shell: degree {space.lmax}, {space.dim} coefficients a field and "
        f"{prior.domain.dim} in all; {len(grid.events)} events and "
        f"{len(grid.stations)} stations; {len(grid.s_arcs)} Sdiff and "
        f"{len(grid.p_arcs)} Pdiff arcs, {len(grid.s_caps)} ScS and "
        f"{len(grid.p_caps)} PcP caps, {forward.slices['modes'].stop - forward.slices['modes'].start} "
        f"coefficients of {len(grid.kernels)} modes: {forward.data_space.dim} data"
    )
    say(f"setup took {time.perf_counter() - started:.1f} s")
    return Setup(
        scale,
        grid,
        space,
        forward,
        prior,
        thermal,
        truth,
        clean,
        data,
        noise,
        error,
        problem,
        together,
        grouped_problem,
        thermal_problem,
    )


def area_weights(space: Any, /) -> np.ndarray:
    """``cos(latitude)`` per grid cell, for area-weighted statistics of a map."""
    weights = np.cos(np.radians(90.0 - np.degrees(space.colatitudes)))
    return np.repeat(weights[:, None], space.longitudes.size, axis=1)


def weighted_correlation(fields: np.ndarray, weights: np.ndarray, /) -> np.ndarray:
    """The correlation matrix of fields, one per row, under area weights."""
    covariance = np.cov(fields, aweights=weights)
    spread = np.sqrt(np.diag(covariance))
    return covariance / np.outer(spread, spread)


def province_share(space: Any, phi: Any, /) -> float:
    """The share of the sphere's area inside ``{phi > 0}``."""
    return float(np.average(space.grid_values(phi) > 0.0, weights=area_weights(space)))


def data_answer(state: Setup, /, *, outdir: pathlib.Path) -> dict:
    """The prior's draws against its correlation, the data's budget, and the
    figures."""
    scale, space, forward = state.scale, state.space, state.forward
    rng = part_rng(scale.seed, "p")
    target = correlation_matrix(scale)
    draws = state.thermal.samples(64, rng=rng)
    grids = np.hstack(
        [
            np.stack([space.grid_values(field).ravel() for field in draw])
            for draw in draws
        ]
    )
    measured = weighted_correlation(
        grids, np.tile(area_weights(space).ravel(), len(draws))
    )
    print(f"  {'':8s}" + "".join(f"{name:>16s}" for name in FIELDS))
    print("  the thermal fields' correlation, asked / over 64 draws:")
    for i, name in enumerate(FIELDS):
        print(
            f"  {name:8s}"
            + "".join(
                f"{target[i, j]:8.2f}/{measured[i, j]:7.2f}" for j in range(len(FIELDS))
            )
        )
    level = level_set_prior(scale, space)
    level_draws = level.samples(64, rng=rng)
    shares = np.array([province_share(space, draw) for draw in level_draws])
    print(
        f"  the level set's draws: from {shares.min():.0%} to {shares.max():.0%} of "
        f"the sphere inside a province, {shares.mean():.0%} on average"
    )
    thermal, phi, truth_contrasts = state.truth
    share = province_share(space, phi)
    response = forward.response(phi)
    print(
        f"the truth: {share:.0%} of the sphere inside its provinces; contrasts "
        f"{tuple(float(c) for c in truth_contrasts)} percent in (vs, vp, rho)"
    )
    print(
        f"  {'kind':<8s} {'count':>6s} {'rms':>7s} {'thermal':>8s} {'chemical':>9s} "
        f"{'noise':>7s}"
    )
    budget = {}
    linear = forward.thermal(thermal)
    chemical = response @ truth_contrasts
    for kind, rows in forward.slices.items():
        if rows.stop == rows.start:
            continue
        row = {
            "count": rows.stop - rows.start,
            "rms": float(np.sqrt(np.mean(state.clean[rows] ** 2))),
            "thermal": float(np.sqrt(np.mean(linear[rows] ** 2))),
            "chemical": float(np.sqrt(np.mean(chemical[rows] ** 2))),
            "noise": float(state.noise[rows][0]),
        }
        budget[kind] = row
        print(
            f"  {kind:<8s} {row['count']:6d} {row['rms']:7.3f} {row['thermal']:8.3f} "
            f"{row['chemical']:9.3f} {row['noise']:7.3f}"
        )
    chi_truth = state.problem.chi_squared(state.truth, state.data)
    chi_none = state.thermal_problem.chi_squared(thermal, state.data)
    print(
        f"  chi-squared of the truth {chi_truth:.0f} on {state.data.size} data; of "
        f"its thermal part alone, the chemistry left out, {chi_none:.0f}"
    )
    coverage = arc_density(state)
    crossings = arcs_through_regions(state)
    print(
        "  the share of the arcs entering a cap of radius "
        f"{scale.region_radius:g} about: "
        + ", ".join(f"{name} {share:.0%}" for name, share in crossings.items())
    )
    draw_prior(state, draws, level_draws, outdir=outdir)
    draw_truth(state, coverage, outdir=outdir)
    return {
        "correlation": measured,
        "target": target,
        "shares": shares,
        "share": share,
        "budget": budget,
        "chi_truth": chi_truth,
        "chi_thermal": chi_none,
        "coverage": coverage,
        "crossings": crossings,
    }


def arcs_through_regions(state: Setup, /) -> dict[str, float]:
    """The share of the arcs that enter each region's cap."""
    space, grid, scale = state.space, state.network, state.scale
    arcs = (*grid.s_arcs, *grid.p_arcs)
    if not arcs:
        return {name: 0.0 for name, _ in REGIONS}
    nodes = np.array(
        [space.geodesic_quadrature(start, end, count=32)[0] for start, end in arcs]
    )
    vectors = to_vector(nodes.reshape(-1, 2))
    shares = {}
    for name, center in REGIONS:
        distance = separation(vectors, to_vector([center]))
        inside = (distance <= scale.region_radius).reshape(len(arcs), -1)
        shares[name] = float(inside.any(axis=1).mean())
    return shares


def arc_density(state: Setup, /) -> np.ndarray:
    """How much path crosses each grid cell of the sphere, per unit area: the
    arcs' nodes counted into the grid, weighted by the length each carries."""
    space, grid = state.space, state.network
    latitudes = 90.0 - np.degrees(space.colatitudes)
    longitudes = np.degrees(space.longitudes)
    counts = np.zeros(space.grid_shape)
    for start, end in (*grid.s_arcs, *grid.p_arcs):
        points, weights = space.geodesic_quadrature(start, end, count=32)
        points = np.asarray(points)
        rows = np.abs(latitudes[:, None] - points[:, 0]).argmin(axis=0)
        columns = (
            np.abs(((longitudes[:, None] - points[:, 1] + 180.0) % 360.0) - 180.0)
        ).argmin(axis=0)
        np.add.at(counts, (rows, columns), weights)
    return counts / np.maximum(area_weights(space), 0.05)


def draw_prior(
    state: Setup, draws: list, level_draws: list, /, *, outdir: pathlib.Path
) -> None:
    """Two draws of the thermal triple, and three of the level set."""
    space = state.space
    limits = [3.0 * std for std in state.scale.thermal_std]
    fig, axes = map_panels(space, 3, 3)
    for row, draw in enumerate(draws[:2]):
        for column, name in enumerate(FIELDS):
            draw_map(
                space,
                draw[column],
                axes[row, column],
                title=f"draw {row + 1}, {name} (percent)",
                limit=limits[column],
            )
    for column, draw in enumerate(level_draws[:3]):
        inside = space.from_grid_values((space.grid_values(draw) > 0.0).astype(float))
        draw_map(
            space,
            inside,
            axes[2, column],
            title=f"draw {column + 1} of the level set: its provinces",
            vmin=0.0,
            vmax=1.0,
            cmap="Greys",
        )
    save(
        fig,
        outdir / "prior.png",
        title="LLSVP, part B: the coupled thermal fields, which share their "
        "pattern, and the provinces of the level set",
    )


def draw_truth(state: Setup, coverage: np.ndarray, /, *, outdir: pathlib.Path) -> None:
    """The true anomalies, the coverage and the network."""
    space, grid = state.space, state.network
    thermal, phi, contrasts = state.truth
    inside = (space.grid_values(phi) > 0.0).astype(float)
    fig, axes = map_panels(space, 2, 3)
    for column, name in enumerate(FIELDS):
        anomaly = space.grid_values(thermal[column]) + contrasts[column] * inside
        draw_map(
            space,
            space.from_grid_values(anomaly),
            axes[0, column],
            title=f"the true d ln {name}, percent",
            limit=float(np.abs(anomaly).max()),
        )
    draw_map(
        space,
        space.from_grid_values(inside),
        axes[1, 0],
        title="the true provinces",
        vmin=0.0,
        vmax=1.0,
        cmap="Greys",
    )
    draw_map(
        space,
        space.from_grid_values(coverage),
        axes[1, 1],
        title="path length through each cell, per unit area",
        vmin=0.0,
        vmax=float(np.percentile(coverage, 99)),
    )
    ax = axes[1, 2]
    plotting.plot(
        space,
        space.zero(),
        ax=ax,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        coasts=True,
        gridlines=False,
        colorbar=False,
    )
    plotting.plot_points(
        space, grid.events, ax=ax, marker="*", size=8.0, color="#eb6834"
    )
    plotting.plot_points(
        space, grid.stations, ax=ax, marker="^", size=6.0, color="#2a78d6"
    )
    ax.set_title("the events (stars) and the stations (triangles)", fontsize=10)
    save(
        fig,
        outdir / "truth.png",
        title="LLSVP, part B: the truth, and what the network sees of the base of "
        "the mantle",
    )


def run(scale: Scale, /, *, outdir: pathlib.Path, state: Setup | None = None) -> dict:
    started = time.perf_counter()
    print(
        f"\n=== B. The coupled prior, the truth and the data ({scale.name} scale) ==="
    )
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
